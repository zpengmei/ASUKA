from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from asuka.cuguga import build_drt
from asuka.mrpt2.df_pair_block import build_df_pair_blocks
from asuka.mrpt2.nevpt2_sc import nevpt2_sc_total_energy_df
from asuka.mrpt2.nevpt2_sc_df_tiled import sr_h1e_v_correction_df_tiled
from asuka.mrpt2.semicanonical import semicanonicalize_core_virt_from_generalized_fock
from asuka.rdm.contract4pdm import _make_f3ca_f3ac_pyscf
from asuka.rdm.rdm123 import _make_rdm123_pyscf
from asuka.solver import GUGAFCISolver


@dataclass(frozen=True)
class NEVPT2SCDFResult:
    """Result object for SC-NEVPT2(DF) on a CSF reference."""

    e_corr: float
    breakdown: dict[str, float]
    e_cas: float | None = None
    e_tot: float | None = None


def _as_f64_2d(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array")
    return arr


def _as_f64_1d(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).ravel()
    return arr


def nevpt2_sc_df_from_mc(
    mc,
    *,
    auxbasis: Any = "weigend+etb",
    twos: int = 0,
    semicanonicalize: bool = True,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    max_memory_mb: float = 4000.0,
    guga_tol: float = 1e-14,
    guga_max_cycle: int = 400,
    guga_max_space: int = 30,
    guga_pspace_size: int = 0,
    verbose: int = 0,
) -> NEVPT2SCDFResult:
    """Compute SC-NEVPT2(DF) from a PySCF CASCI/CASSCF object using a CSF reference.

    Notes
    -----
    - This driver is **reference-oriented**: it solves the CAS problem in the CSF
      basis (via :class:`asuka.solver.GUGAFCISolver`) even if `mc.ci` is already
      available in determinant form.
    - By default, core/virtual orbitals are semicanonicalized by diagonalizing
      the generalized Fock in the core-core and virtual-virtual blocks.
    """

    mol = mc.mol
    ncore = int(mc.ncore)
    ncas = int(mc.ncas)
    nelecas = mc.nelecas
    nocc = ncore + ncas

    mo = _as_f64_2d(mc.mo_coeff)
    if mo.shape[1] < nocc:
        raise ValueError("mo_coeff has too few orbitals for (ncore+ncas)")
    mo_core = mo[:, :ncore]
    mo_act = mo[:, ncore:nocc]
    mo_virt = mo[:, nocc:]
    if mo_core.size == 0 or mo_virt.size == 0:
        raise ValueError("need at least 1 core and 1 virtual orbital for SC-NEVPT2")

    eps = _as_f64_1d(getattr(mc, "mo_energy", getattr(mc._scf, "mo_energy")))
    if eps.size < mo.shape[1]:
        raise ValueError("mo_energy has too few entries")
    eps_core = eps[:ncore]
    eps_virt = eps[nocc : nocc + mo_virt.shape[1]]

    # Active-space integrals:
    # - eri_chem matches the Hamiltonian convention expected by the FCI/CSF solvers (chemist notation).
    # - h2e_nevpt matches the einsum conventions in `asuka.mrpt2.nevpt2_sc` (Physicist-like transpose).
    h1e = np.asarray(mc.h1e_for_cas()[0], dtype=np.float64, order="C")
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("SC-NEVPT2 active-space integrals currently require mol.cart=True (cuERI dense CPU)")

    from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder  # noqa: PLC0415
    from asuka.cueri.mol_basis import pack_cart_shells_from_mol  # noqa: PLC0415

    # cuERI returns an ordered-pair matrix with row pair ij=i*ncas+j and col pair kl=k*ncas+l.
    # Reshape it to a 4-index tensor matching PySCF AO2MO `restore(1, ..., ncas)` conventions.
    ao_basis = None
    try:
        ao_basis = pack_cart_shells_from_mol(mol, expand_contractions=True)
    except Exception:
        ao_basis = None

    if ao_basis is None:
        raise NotImplementedError(
            "SC-NEVPT2 cuERI CPU path requires a packable cartesian ao_basis from mol."
        )
    builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis)

    eri_mat = builder.build_eri_mat(mo_act, eps_ao=0.0, eps_mo=0.0)
    eri_chem = np.asarray(eri_mat.reshape(ncas, ncas, ncas, ncas), dtype=np.float64, order="C")
    h2e_nevpt = np.asarray(eri_chem.transpose(0, 2, 1, 3), dtype=np.float64, order="C")

    # CSF reference solve.
    solver = GUGAFCISolver(twos=int(twos))
    _e_act, ci_csf = solver.kernel(
        h1e,
        eri_chem,
        ncas,
        nelecas,
        tol=float(guga_tol),
        max_cycle=int(guga_max_cycle),
        max_space=int(guga_max_space),
        pspace_size=int(guga_pspace_size),
    )
    drt = build_drt(norb=ncas, nelec=int(np.sum(nelecas)), twos_target=int(twos))

    # Active density tensors and contracted-4PDM intermediates in PySCF NEVPT2 conventions.
    #
    # Important: PySCF SC-NEVPT2 uses the *raw* spin-traced dm2/dm3 returned by
    # `pyscf.fci.rdm.make_dm123('FCI3pdm_kern_sf', ...)` (i.e. no `reorder_dm123`).
    # See `pyscf.mrpt.nevpt2.NEVPT.kernel()`.
    dm1, dm2, dm3 = _make_rdm123_pyscf(drt, ci_csf, max_memory_mb=float(max_memory_mb), reorder=False)
    f3ca, f3ac = _make_f3ca_f3ac_pyscf(drt, ci_csf, eri_chem, max_memory_mb=float(max_memory_mb))

    if bool(semicanonicalize):
        sc = semicanonicalize_core_virt_from_generalized_fock(
            mc,
            mo_core=mo_core,
            mo_act=mo_act,
            mo_virt=mo_virt,
            dm1_act=dm1,
        )
        mo_core = sc.mo_core
        mo_virt = sc.mo_virt
        eps_core = sc.eps_core
        eps_virt = sc.eps_virt

    # DF pair blocks for mixed integrals.
    l_cv, l_vc, l_va, l_ac, l_aa = build_df_pair_blocks(
        mol,
        [
            (mo_core, mo_virt),
            (mo_virt, mo_core),
            (mo_virt, mo_act),
            (mo_act, mo_core),
            (mo_act, mo_act),
        ],
        auxbasis=auxbasis,
        max_memory=int(max_memory_mb),
        verbose=int(verbose),
        compute_pair_norm=False,
    )

    # One-electron blocks coupling inactive/virtual with active space.
    core_dm = np.dot(mo_core, mo_core.T) * 2.0
    from asuka.mrpt2.semicanonical import build_hcore_ao, build_vhf_df  # noqa: PLC0415

    _vhf_ctx = get_df_cholesky_context(
        mol, auxbasis=auxbasis, max_memory=int(max_memory_mb), verbose=int(verbose),
    )
    _b_ao = np.asarray(_vhf_ctx.B_ao, dtype=np.float64)
    core_vhf = build_vhf_df(core_dm, b_ao=_b_ao)
    hcore = build_hcore_ao(mol)
    h1e_v_sir = reduce(np.dot, (mo_virt.T, hcore + core_vhf, mo_core))  # (virt, core)
    h1e_v_si = reduce(np.dot, (mo_act.T, hcore + core_vhf, mo_core))  # (act, core)

    # Sr needs the (virt,act) block corrected by -Σ_b (r b|b a) with the *same* DF block.
    h1e_v_sr = reduce(np.dot, (mo_virt.T, hcore + core_vhf, mo_act))  # (virt, act)
    h1e_v_sr = h1e_v_sr - sr_h1e_v_correction_df_tiled(l_va, l_aa)

    # Energy breakdown.
    pt2_backend = str(pt2_backend).lower()
    if pt2_backend == "cpu":
        breakdown = nevpt2_sc_total_energy_df(
            l_cv=l_cv,
            l_vc=l_vc,
            l_va=l_va,
            l_ac=l_ac,
            l_aa=l_aa,
            eps_core=eps_core,
            eps_virt=eps_virt,
            h1e_v_sir=h1e_v_sir,
            h1e_v_sr=h1e_v_sr,
            h1e_v_si=h1e_v_si,
            dm1=dm1,
            dm2=dm2,
            dm3=dm3,
            h1e=h1e,
            h2e=h2e_nevpt,
            f3ca=f3ca,
            f3ac=f3ac,
        )
    elif pt2_backend in ("cuda", "cupy"):
        from asuka.mrpt2.nevpt2_sc_df_cuda import nevpt2_sc_total_energy_df_cuda  # noqa: PLC0415

        breakdown = nevpt2_sc_total_energy_df_cuda(
            l_cv=l_cv,
            l_vc=l_vc,
            l_va=l_va,
            l_ac=l_ac,
            l_aa=l_aa,
            eps_core=eps_core,
            eps_virt=eps_virt,
            h1e_v_sir=h1e_v_sir,
            h1e_v_sr=h1e_v_sr,
            h1e_v_si=h1e_v_si,
            dm1=dm1,
            dm2=dm2,
            dm3=dm3,
            h1e=h1e,
            h2e=h2e_nevpt,
            f3ca=f3ca,
            f3ac=f3ac,
            device=cuda_device,
        )
    else:
        raise ValueError("pt2_backend must be one of: 'cpu', 'cuda'")

    e_corr = float(breakdown["e_total"])
    e_cas = getattr(mc, "e_cas", None)
    e_tot = None
    if e_cas is not None:
        e_tot = float(e_cas) + e_corr
    return NEVPT2SCDFResult(e_corr=e_corr, breakdown=breakdown, e_cas=e_cas, e_tot=e_tot)
