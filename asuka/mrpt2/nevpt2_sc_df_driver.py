from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from asuka.cuguga import build_drt
from asuka.mrpt2.df_pair_block import build_df_pair_blocks_from_df_B
from asuka.mrpt2.nevpt2_sc import nevpt2_sc_total_energy_df
from asuka.mrpt2.nevpt2_sc_df_tiled import sr_h1e_v_correction_df_tiled
from asuka.mrpt2.semicanonical import build_vhf_df, semicanonicalize_core_virt_from_fock_ao
from asuka.rdm.contract4pdm import _make_f3ca_f3ac_pyscf
from asuka.rdm.rdm123 import _make_rdm123_pyscf


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


def nevpt2_sc_df_from_ref(
    ref,
    *,
    scf_out: Any | None = None,
    state: int = 0,
    auxbasis: Any | None = None,
    twos: int | None = None,
    semicanonicalize: bool = True,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    max_memory_mb: float = 4000.0,
    verbose: int = 0,
) -> NEVPT2SCDFResult:
    """Compute SC-NEVPT2(DF) from an ASUKA CAS reference (CASSCF/CASCI result).

    Notes
    -----
    - By default, core/virtual orbitals are semicanonicalized by diagonalizing
      the generalized Fock in the core-core and virtual-virtual blocks.
    """

    if scf_out is None:
        scf_out = getattr(ref, "scf_out", None)
    if scf_out is None:
        casci = getattr(ref, "casci", None)
        if casci is not None:
            scf_out = getattr(casci, "scf_out", None)
    if scf_out is None:
        raise ValueError("scf_out must be provided (or available as ref.scf_out/ref.casci.scf_out)")

    mol = getattr(ref, "mol", None)
    if mol is None:
        raise ValueError("ref.mol must be available")

    ncore = int(getattr(ref, "ncore"))
    ncas = int(getattr(ref, "ncas"))
    nelecas = getattr(ref, "nelecas")
    nocc = ncore + ncas

    mo = _as_f64_2d(getattr(ref, "mo_coeff"))
    if mo.shape[1] < nocc:
        raise ValueError("mo_coeff has too few orbitals for (ncore+ncas)")
    mo_core = mo[:, :ncore]
    mo_act = mo[:, ncore:nocc]
    mo_virt = mo[:, nocc:]
    if mo_core.size == 0 or mo_virt.size == 0:
        raise ValueError("need at least 1 core and 1 virtual orbital for SC-NEVPT2")

    if twos is None:
        twos = getattr(mol, "spin", None)
    if twos is None:
        raise ValueError("twos must be provided (or available as ref.mol.spin)")
    twos = int(twos)

    ci = getattr(ref, "ci", None)
    if ci is None:
        raise ValueError("ref.ci must be available")
    if isinstance(ci, (list, tuple)):
        st = int(state)
        if st < 0 or st >= len(ci):
            raise IndexError("state out of range for ref.ci")
        ci_csf = np.asarray(ci[st], dtype=np.float64).ravel()
    else:
        if int(state) != 0:
            raise ValueError("state must be 0 for a single-root ref.ci")
        ci_csf = np.asarray(ci, dtype=np.float64).ravel()

    # Active-space integrals:
    # - eri_chem matches the Hamiltonian convention expected by the FCI/CSF solvers (chemist notation).
    # - h2e_nevpt matches the einsum conventions in `asuka.mrpt2.nevpt2_sc` (Physicist-like transpose).
    if auxbasis is not None:
        # The ASUKA-native DF path uses `scf_out.df_B`. Ensure the user's auxbasis
        # selection is consistent with the SCF/CASSCF preparation.
        if isinstance(auxbasis, str) and isinstance(getattr(scf_out, "auxbasis_name", None), str):
            if str(auxbasis) != str(getattr(scf_out, "auxbasis_name")):
                raise ValueError("auxbasis must match scf_out.auxbasis_name; rebuild scf_out with the desired auxbasis")
        else:
            raise NotImplementedError("auxbasis override is not supported in the ASUKA-native NEVPT2 driver")

    hcore = np.asarray(getattr(scf_out, "int1e").hcore, dtype=np.float64)
    df_B = getattr(scf_out, "df_B", None)
    if df_B is None:
        raise ValueError("scf_out.df_B is required for SC-NEVPT2(DF)")
    B_ao = np.asarray(df_B, dtype=np.float64)

    core_dm = np.dot(mo_core, mo_core.T) * 2.0
    core_vhf = build_vhf_df(core_dm, b_ao=B_ao)
    h1e = np.asarray(mo_act.T @ (hcore + core_vhf) @ mo_act, dtype=np.float64, order="C")

    if not bool(getattr(mol, "cart", True)):
        raise NotImplementedError("SC-NEVPT2 active-space integrals currently require cart=True (dense cuERI CPU)")

    from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder  # noqa: PLC0415

    # cuERI returns an ordered-pair matrix with row pair ij=i*ncas+j and col pair kl=k*ncas+l.
    # Reshape it to a 4-index tensor matching the standard AO2MO `restore(1, ..., ncas)` layout.
    ao_basis = getattr(scf_out, "ao_basis", None)
    if ao_basis is None:
        raise ValueError("scf_out.ao_basis is required for SC-NEVPT2 active-space ERIs")
    builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis)

    eri_mat = builder.build_eri_mat(mo_act, eps_ao=0.0, eps_mo=0.0)
    eri_chem = np.asarray(eri_mat.reshape(ncas, ncas, ncas, ncas), dtype=np.float64, order="C")
    h2e_nevpt = np.asarray(eri_chem.transpose(0, 2, 1, 3), dtype=np.float64, order="C")

    drt = build_drt(norb=ncas, nelec=int(np.sum(nelecas)), twos_target=int(twos))
    if int(ci_csf.size) != int(drt.ncsf):
        raise ValueError("ref.ci has wrong length for the constructed DRT")

    # Active density tensors and contracted-4PDM intermediates in NEVPT2 conventions.
    # Important: use the raw spin-traced dm2/dm3 without `reorder_dm123`.
    dm1, dm2, dm3 = _make_rdm123_pyscf(drt, ci_csf, max_memory_mb=float(max_memory_mb), reorder=False)
    f3ca, f3ac = _make_f3ca_f3ac_pyscf(drt, ci_csf, eri_chem, max_memory_mb=float(max_memory_mb))

    if bool(semicanonicalize):
        nao = int(mo.shape[0])
        dm_act = mo_act @ dm1 @ mo_act.T
        dm_tot = core_dm + dm_act
        vhf_tot = build_vhf_df(dm_tot, b_ao=B_ao)
        f_ao = hcore + vhf_tot
        sc = semicanonicalize_core_virt_from_fock_ao(f_ao, mo_core=mo_core, mo_virt=mo_virt)
        mo_core = sc.mo_core
        mo_virt = sc.mo_virt
        eps_core = sc.eps_core
        eps_virt = sc.eps_virt
    else:
        dm_act = mo_act @ dm1 @ mo_act.T
        dm_tot = core_dm + dm_act
        vhf_tot = build_vhf_df(dm_tot, b_ao=B_ao)
        f_ao = hcore + vhf_tot
        f_cc = 0.5 * (mo_core.T @ f_ao @ mo_core + (mo_core.T @ f_ao @ mo_core).T)
        f_vv = 0.5 * (mo_virt.T @ f_ao @ mo_virt + (mo_virt.T @ f_ao @ mo_virt).T)
        eps_core = np.diag(f_cc).copy()
        eps_virt = np.diag(f_vv).copy()

    # DF pair blocks for mixed integrals.
    l_cv, l_vc, l_va, l_ac, l_aa = build_df_pair_blocks_from_df_B(
        df_B,
        [
            (mo_core, mo_virt),
            (mo_virt, mo_core),
            (mo_virt, mo_act),
            (mo_act, mo_core),
            (mo_act, mo_act),
        ],
        max_memory=int(max_memory_mb),
        compute_pair_norm=False,
    )

    # One-electron blocks coupling inactive/virtual with active space.
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
    e_cas = None
    if hasattr(ref, "e_roots"):
        e_roots = np.asarray(getattr(ref, "e_roots"), dtype=np.float64).ravel()
        if int(state) < int(e_roots.size):
            e_cas = float(e_roots[int(state)])
    if e_cas is None:
        e_tot_ref = getattr(ref, "e_tot", None)
        if e_tot_ref is not None:
            e_arr = np.asarray(e_tot_ref, dtype=np.float64).ravel()
            if int(e_arr.size) == 1:
                e_cas = float(e_arr[0])
            elif int(state) < int(e_arr.size):
                e_cas = float(e_arr[int(state)])

    e_tot = None if e_cas is None else float(e_cas) + e_corr
    return NEVPT2SCDFResult(e_corr=e_corr, breakdown=breakdown, e_cas=e_cas, e_tot=e_tot)
