"""Analytic nuclear gradients for (uncontracted) MRCISD.

This module provides an analytic nuclear gradient implementation for the
**uncontracted** MRCISD energy produced by :func:`asuka.mrci.driver.mrci_from_mc`.

Notes
-----
*   The implementation assumes a frozen-core reference consistent with the
    MRCISD Hamiltonian used in the driver.
*   The target energy (MRCISD) is variational w.r.t. the MRCISD CI vector but is
    *not* variational w.r.t. the underlying CASSCF (or SA-CASSCF) orbital and CI
    parameters. We therefore solve a Z-vector / Lagrange multiplier equation
    using ASUKA's internal CP-CASSCF machinery.
*   Currently supported: uncontracted MRCISD (``method='mrcisd'``) with optional
    P-space projection (``projection=True``).
*   Validation support: internally contracted ic-MRCISD (``method='ic_mrcisd'``)
    via **reconstruction** into an uncontracted CSF vector (tiny systems only).
*   Not yet supported *in this module*: Davidson +Q correction (``plus_q=True``)
    as a fully consistent analytic Lagrangian. The public driver
    (`asuka/mrci/grad_driver.py`) can optionally provide a **heuristic**
    `plus_q=True` gradient by scaling the uncorrected correlation gradient with a
    constant Davidson denominator.
*   Not yet supported analytically: production-grade contracted density builders.

The key quantities we need from the target (MRCISD) wavefunction are the 1- and
2-RDM in the correlated MO space (all non-core orbitals). From those we build
the generalized Fock matrix (``gfock``), which (i) defines the orbital gradient
``2*pack(gfock-gfock.T)`` used as the Lagrange RHS and (ii) provides the
energy-weighted density used in the Pulay (overlap-derivative) term.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np

from asuka.mcscf import sacasscf_response as sacasscf
from asuka.mrci.driver import assign_roots_by_overlap, mrci_from_mc, mrci_states_from_mc


def _get_hcore_ao(mf, mol):
    """Return AO-core Hamiltonian for *mf*.

    Some PySCF objects accept ``mol`` as an argument, others do not.
    """

    try:
        return mf.get_hcore(mol)
    except TypeError:
        return mf.get_hcore()


def _nao_cart_from_basis(ao_basis: Any) -> int:
    from asuka.integrals.gto_cart import ncart  # noqa: PLC0415

    shell_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int64).ravel()
    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int64).ravel()
    if shell_start.size == 0:
        return 0
    if shell_start.shape != shell_l.shape:
        raise RuntimeError("invalid AO basis shell arrays")
    nfunc = np.asarray([int(ncart(int(l))) for l in shell_l.tolist()], dtype=np.int64)
    return int(np.max(shell_start + nfunc))


def _build_internal_newton_adapter(mc: Any) -> Any:
    """Build an ASUKA-internal Newton adapter for Hessian/ao2mo calls."""

    ao2mo_fn = getattr(mc, "ao2mo", None)
    if callable(ao2mo_fn):
        mod = str(getattr(getattr(ao2mo_fn, "__func__", ao2mo_fn), "__module__", ""))
        if mod.startswith("asuka.mcscf.newton_"):
            return mc

    mo_coeff = np.asarray(getattr(mc, "mo_coeff"), dtype=np.float64)
    fcisolver = getattr(mc, "fcisolver", None)
    if fcisolver is None:
        raise RuntimeError("mc must provide fcisolver for internal Newton adapter")

    scf = getattr(mc, "_scf", None)
    df_B = None if scf is None else getattr(scf, "df_B", None)
    hcore_src = scf if scf is not None else mc

    if df_B is not None:
        from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter  # noqa: PLC0415

        return DFNewtonCASSCFAdapter(
            df_B=df_B,
            hcore_ao=np.asarray(_get_hcore_ao(hcore_src, mc.mol), dtype=np.float64),
            ncore=int(getattr(mc, "ncore")),
            ncas=int(getattr(mc, "ncas")),
            nelecas=getattr(mc, "nelecas"),
            mo_coeff=mo_coeff,
            fcisolver=fcisolver,
            weights=list(np.asarray(getattr(mc, "weights", [1.0]), dtype=np.float64).ravel().tolist()),
            frozen=getattr(mc, "frozen", None),
            internal_rotation=bool(getattr(mc, "internal_rotation", False)),
            extrasym=getattr(mc, "extrasym", None),
        )

    from asuka.cueri.mol_basis import get_cached_or_pack_cart_ao_basis  # noqa: PLC0415
    from asuka.mcscf.newton_dense import DenseNewtonCASSCFAdapter  # noqa: PLC0415

    ao_basis = get_cached_or_pack_cart_ao_basis(cache_owner=mc, mol=getattr(mc, "mol", None), cache_attr="_asuka_ao_basis")
    if ao_basis is None:
        raise RuntimeError(
            "ASUKA internal Hessian path requires a packable cart AO basis. "
            "Provide mc._asuka_ao_basis or a mol with cartesian basis introspection."
        )
    nao_cart = int(_nao_cart_from_basis(ao_basis))
    if int(mo_coeff.shape[0]) != nao_cart:
        raise RuntimeError(
            "ASUKA internal dense Newton adapter requires cartesian MO coefficients "
            f"(nao_cart={nao_cart}, mo_rows={int(mo_coeff.shape[0])})."
        )

    return DenseNewtonCASSCFAdapter(
        ao_basis=ao_basis,
        atom_coords_bohr=np.asarray(mc.mol.atom_coords(), dtype=np.float64),
        hcore_ao=np.asarray(_get_hcore_ao(hcore_src, mc.mol), dtype=np.float64),
        ncore=int(getattr(mc, "ncore")),
        ncas=int(getattr(mc, "ncas")),
        nelecas=getattr(mc, "nelecas"),
        mo_coeff=mo_coeff,
        fcisolver=fcisolver,
        weights=list(np.asarray(getattr(mc, "weights", [1.0]), dtype=np.float64).ravel().tolist()),
        frozen=getattr(mc, "frozen", None),
        internal_rotation=bool(getattr(mc, "internal_rotation", False)),
        extrasym=getattr(mc, "extrasym", None),
    )


def _build_internal_hessian_context(mc: Any) -> tuple[Any, Any]:
    mc_hess = _build_internal_newton_adapter(mc)
    eris = mc_hess.ao2mo(np.asarray(mc.mo_coeff, dtype=np.float64))
    return mc_hess, eris


def _make_rdm12(
    drt,
    ci,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (spin-summed, spatial) 1- and 2-RDM from a cuGUGA DRT+CI."""

    if rdm_backend == "cuda":
        try:
            from asuka.cuda.rdm_gpu import make_rdm12_cuda

            return make_rdm12_cuda(drt, ci)
        except Exception:
            # Fall back to CPU if CUDA path is unavailable.
            pass

    from asuka.rdm.stream import make_rdm12_streaming

    return make_rdm12_streaming(drt, ci)


def _build_gfock_mrcisd(
    mol,
    mf,
    mo_coeff: np.ndarray,
    ncore: int,
    dm1_corr: np.ndarray,
    dm2_corr: np.ndarray,
    eri4_corr: Optional[np.ndarray] = None,
    aapa_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build generalized Fock matrix for frozen-core MRCISD.

    Parameters
    ----------
    mol
        PySCF Mole object.
    mf
        Underlying SCF object (used for J/K builds).
    mo_coeff
        Full MO coefficients (nao, nmo).
    ncore
        Number of doubly occupied (frozen) core orbitals.
    dm1_corr, dm2_corr
        1- and 2-RDM in the correlated MO subspace (all orbitals after ncore).
        Shapes: (ncor,ncor) and (ncor,ncor,ncor,ncor).
    eri4_corr
        Optional 4-index ERI tensor in correlated MO space (ncor^4).
    aapa_core
        Optional (corr corr|core corr) tensor with shape (ncor,ncor,ncore,ncor),
        used only when ncore > 0.

    Returns
    -------
    gfock
        Generalized Fock matrix in the full MO basis (nmo,nmo).

    Notes
    -----
    The implementation mirrors PySCF's CASSCF generalized Fock construction,
    generalized to a correlated (non-active-only) 2-RDM.
    """

    from asuka.cueri.ao2mo import ao2mo_kernel  # noqa: PLC0415

    nao, nmo = mo_coeff.shape
    ncor = nmo - ncore
    if ncor != dm1_corr.shape[0]:
        raise ValueError(
            f"Inconsistent correlated space: nmo-ncore={ncor} but dm1 has {dm1_corr.shape[0]}"
        )

    mo_core = mo_coeff[:, :ncore]
    mo_corr = mo_coeff[:, ncore:]

    # AO densities
    dm_core = 2.0 * (mo_core @ mo_core.T) if ncore > 0 else np.zeros((nao, nao))
    dm_corr = mo_corr @ dm1_corr @ mo_corr.T

    # AO core Hamiltonian and Coulomb/exchange potentials
    hcore_ao = _get_hcore_ao(mf, mol)
    h1e_mo = mo_coeff.T @ hcore_ao @ mo_coeff

    # Core potential (J-K/2) from frozen core density
    if ncore > 0:
        vj_c, vk_c = mf.get_jk(mol, dm_core, hermi=1)
        vhf_c_ao = vj_c - 0.5 * vk_c
    else:
        vhf_c_ao = np.zeros((nao, nao))

    # Mean-field potential generated by *correlated* 1-RDM (used only for core columns)
    vj_a, vk_a = mf.get_jk(mol, dm_corr, hermi=1)
    vhf_a_ao = vj_a - 0.5 * vk_a

    vhf_c_mo = mo_coeff.T @ vhf_c_ao @ mo_coeff
    vhf_a_mo = mo_coeff.T @ vhf_a_ao @ mo_coeff

    # Assemble generalized Fock
    gfock = np.zeros((nmo, nmo))

    # Core columns: doubly occupied, include mean-field of correlated density
    if ncore > 0:
        gfock[:, :ncore] = (h1e_mo[:, :ncore] + vhf_c_mo[:, :ncore] + vhf_a_mo[:, :ncore]) * 2.0

    # Correlated columns: 1-body + core potential contracted with dm1_corr
    heff_mo = h1e_mo + vhf_c_mo
    gfock[:, ncore:] = heff_mo[:, ncore:] @ dm1_corr

    # 2-RDM contribution to correlated columns
    if eri4_corr is None:
        eri4_corr = ao2mo_kernel(mol, mo_corr, compact=False)
        eri4_corr = eri4_corr.reshape(ncor, ncor, ncor, ncor)
    else:
        eri4_corr = np.asarray(eri4_corr, dtype=np.float64)
        if eri4_corr.shape != (ncor, ncor, ncor, ncor):
            raise ValueError("eri4_corr has wrong shape")

    # Rows in correlated space (place into full MO rows ncore:)
    # gfock_{p,t} += sum_{u,v,w} (u v|p w) * dm2[v,u,w,t]
    g2_corr = np.einsum("uvpw,vuwt->pt", eri4_corr, dm2_corr, optimize=True)
    gfock[ncore:, ncore:] += g2_corr

    # Rows in core space (p in core, t in correlated): need (u v|i w)
    if ncore > 0:
        if aapa_core is None:
            aapa_core = ao2mo_kernel(mol, (mo_corr, mo_corr, mo_core, mo_corr), compact=False)
            aapa_core = aapa_core.reshape(ncor, ncor, ncore, ncor)
        else:
            aapa_core = np.asarray(aapa_core, dtype=np.float64)
            if aapa_core.shape != (ncor, ncor, ncore, ncor):
                raise ValueError("aapa_core has wrong shape")
        g2_core = np.einsum("uviw,vuwt->it", aapa_core, dm2_corr, optimize=True)
        gfock[:ncore, ncore:] += g2_core

    return gfock


@dataclass(frozen=True)
class MRCIIntegralCache:
    eri4_corr: np.ndarray
    aapa_core: np.ndarray | None


def _build_mrcisd_integral_cache(mol, mo_coeff: np.ndarray, ncore: int) -> MRCIIntegralCache:
    from asuka.cueri.ao2mo import ao2mo_kernel  # noqa: PLC0415

    mo_coeff = np.asarray(mo_coeff, dtype=np.float64)
    ncore = int(ncore)

    nao, nmo = mo_coeff.shape
    if ncore < 0 or ncore > nmo:
        raise ValueError("ncore out of range")
    ncor = nmo - ncore
    if ncor <= 0:
        raise ValueError("empty correlated space")

    mo_core = mo_coeff[:, :ncore]
    mo_corr = mo_coeff[:, ncore:]

    eri4_corr = ao2mo_kernel(mol, mo_corr, compact=False)
    eri4_corr = np.asarray(eri4_corr, dtype=np.float64).reshape(ncor, ncor, ncor, ncor)

    aapa_core = None
    if ncore > 0:
        aapa_core = ao2mo_kernel(mol, (mo_corr, mo_corr, mo_core, mo_corr), compact=False)
        aapa_core = np.asarray(aapa_core, dtype=np.float64).reshape(ncor, ncor, ncore, ncor)

    return MRCIIntegralCache(eri4_corr=eri4_corr, aapa_core=aapa_core)


def _grad_elec_mrcisd(
    mol,
    mf_grad,
    mo_coeff: np.ndarray,
    ncore: int,
    dm1_corr: np.ndarray,
    dm2_corr: np.ndarray,
    gfock: np.ndarray,
    atmlst: Sequence[int],
) -> np.ndarray:
    """Electronic part of the MRCISD analytic gradient.

    This follows the structure of :func:`pyscf.grad.casscf.grad_elec`, replacing
    the active-space RDMs with correlated-space RDMs.
    """

    from asuka.cueri.ao2mo import nr_e2_half_transform  # noqa: PLC0415
    from asuka.mcscf.sacasscf_response import pack_tril, shell_prange  # noqa: PLC0415

    nao, nmo = mo_coeff.shape
    ncor = nmo - ncore
    if ncor != dm1_corr.shape[0]:
        raise ValueError(
            f"Inconsistent correlated space: nmo-ncore={ncor} but dm1 has {dm1_corr.shape[0]}"
        )

    mo_core = mo_coeff[:, :ncore]
    mo_corr = mo_coeff[:, ncore:]

    dm_core = 2.0 * (mo_core @ mo_core.T) if ncore > 0 else np.zeros((nao, nao))
    dm_corr = mo_corr @ dm1_corr @ mo_corr.T
    dm1 = dm_core + dm_corr

    # Energy-weighted density (Pulay term)
    dme0 = mo_coeff @ ((gfock + gfock.T) * 0.5) @ mo_coeff.T

    # 2e potential derivatives for (dm_core, dm_corr)
    vj, vk = mf_grad.get_jk(mol, (dm_core, dm_corr))
    vhf1c, vhf1a = vj - 0.5 * vk

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    aoslices = mol.aoslice_by_atom()

    max_memory = getattr(mf_grad, "max_memory", 2000)
    nao_pair = nao * (nao + 1) // 2
    max_p = int((aoslices[:, 3] - aoslices[:, 2]).max())
    # Memory footprint is dominated by eri1 + dm2_ao-like intermediates. This
    # is the same conservative heuristic used by PySCF's CASSCF gradient.
    blksize = int(max_memory * 0.9e6 / 8 / (4 * max_p * nao_pair))
    blksize = min(nao, max(2, blksize))

    dm2_contr = os.getenv("CUGUGA_MRCI_GRAD_DM2_CONTRACTION", "auto").strip().lower()
    if dm2_contr not in {"auto", "ao_pair", "mo_pair"}:
        raise ValueError(
            "CUGUGA_MRCI_GRAD_DM2_CONTRACTION must be one of {'auto','ao_pair','mo_pair'}"
        )

    # Heuristic: AO-pair buffering allocates an intermediate of size
    # (ncor^2, nao^2), which can dominate memory for large correlated spaces.
    use_mo_pair = dm2_contr == "mo_pair"
    if dm2_contr == "auto":
        dm2buf_peak_bytes = int(ncor) * int(ncor) * int(nao) * int(nao) * 8
        use_mo_pair = dm2buf_peak_bytes > int(max_memory * 0.9e6)

    dm2buf = None
    mo_corr_f = None
    if not use_mo_pair:
        # Prepare dm2 buffer in AO-pair form (same strategy as PySCF CASSCF gradient)
        dm2_cc = dm2_corr + dm2_corr.transpose(0, 1, 3, 2)
        dm2buf = nr_e2_half_transform(
            dm2_cc.reshape(ncor * ncor, ncor * ncor), mo_corr.T, (0, nao, 0, nao)
        )
        dm2buf = dm2buf.reshape(ncor * ncor, nao, nao)
        dm2buf = pack_tril(dm2buf)
        diag_idx = np.arange(nao) + np.arange(nao) * (np.arange(nao) + 1) // 2
        dm2buf[:, diag_idx] *= 0.5
        dm2buf = dm2buf.reshape(ncor, ncor, nao_pair)
    else:
        # For AO->MO half-transforms of (∂(pq|kl)/∂R) in the k,l indices.
        mo_corr_f = np.asarray(mo_corr, dtype=np.float64, order="F")

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)

        # 1e contribution
        de[k] += np.einsum("xij,ij->x", h1ao, dm1)

        # Pulay (overlap) contribution
        de[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2.0

        # 2e contribution from correlated 2-RDM
        q1 = 0
        for b0, b1, nf in shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice
            ).reshape(3, p1 - p0, nf, nao_pair)

            if dm2buf is not None:
                # Transform dm2_{ij,kl} where ij in correlated MOs, kl in AO pairs.
                dm2_ao = np.einsum("ijw,pi,qj->pqw", dm2buf, mo_corr[p0:p1], mo_corr[q0:q1])
                de[k] -= np.einsum("xijw,ijw->x", eri1, dm2_ao) * 2.0
            else:
                if mo_corr_f is None:  # pragma: no cover
                    raise RuntimeError("internal error: mo_corr_f not initialized")

                # Memory-saving path: transform the k,l indices of the derivative ERIs to the
                # correlated MO basis and contract against dm2 in the mixed (AO,AO,MO,MO) form.
                dm2_pqkl = np.einsum("ijkl,pi,qj->pqkl", dm2_corr, mo_corr[p0:p1], mo_corr[q0:q1])
                dm2_pqkl = dm2_pqkl + dm2_pqkl.transpose(0, 1, 3, 2)

                nrow = (p1 - p0) * nf
                for x in range(3):
                    eri1_x = np.ascontiguousarray(eri1[x].reshape(nrow, nao_pair))
                    eri1_mo = nr_e2_half_transform(
                        eri1_x, mo_corr_f, (0, ncor, 0, ncor), aosym="s2kl", mosym="s1"
                    ).reshape(p1 - p0, nf, ncor, ncor)
                    de[k, x] -= np.einsum("pqkl,pqkl->", eri1_mo, dm2_pqkl)

        # Mean-field derivative terms involving the frozen core (core-core and core-corr)
        de[k] += np.einsum("xij,ij->x", vhf1c[:, p0:p1], dm1[p0:p1]) * 2.0
        if ncore > 0:
            de[k] += np.einsum("xij,ij->x", vhf1a[:, p0:p1], dm_core[p0:p1]) * 2.0

    return de


@dataclass
class MRCIAnalyticGradData:
    """Cached target-wavefunction quantities for analytic gradients."""

    dm1_corr: np.ndarray
    dm2_corr: np.ndarray
    gfock: np.ndarray


def mrcisd_energy_and_grad(
    mc,
    *,
    plus_q: bool = False,
    use_cuda: bool = True,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    atmlst: Optional[Sequence[int]] = None,
    max_cycle: int = 80,
    conv_tol: float = 1e-10,
    max_virt_e: int = 2,
    verbose: Optional[int] = None,
    hop_backend: str | None = None,
    correlate_inactive: int = 0,
):
    """Compute uncontracted MRCISD energy and analytic nuclear gradient.

    Parameters
    ----------
    mc : Any
        A converged PySCF CASSCF / SA-CASSCF object.
    plus_q : bool, optional
        Davidson +Q correction (currently **not supported** for analytic gradients).
        Default is False.
    use_cuda : bool, optional
        Passed through to :func:`mrci_from_mc` (energy/CI backend). Default is True.
    rdm_backend : {"cuda", "cpu"}, optional
        Backend for RDM construction. "cuda" is preferred, falls back to "cpu".
        Default is "cuda".
    atmlst : Sequence[int] | None, optional
        Atom indices to compute gradients for. Default is None (all atoms).
    max_cycle : int, optional
        Maximum number of iterations for the solver. Default is 80.
    conv_tol : float, optional
        Convergence tolerance for the solver. Default is 1e-10.
    max_virt_e : int, optional
        Maximum number of electrons in the virtual space. Default is 2.
    verbose : int | None, optional
        Verbosity level.
    hop_backend : str | None, optional
        Backend for Hamiltonian-vector product ("fast", "augmented", or "cuda").
    correlate_inactive : int, optional
        Number of inactive (core) orbitals to correlate. Default is 0.

    Returns
    -------
    mrci_res : MRCISDResult
        The result object returned by :func:`mrci_from_mc`.
    grad : np.ndarray
        Nuclear gradient array with shape (natm, 3) or (len(atmlst), 3).
    """

    if plus_q:
        raise NotImplementedError(
            "Analytic gradients for Davidson +Q corrected energies are not implemented yet."
        )

    profile = bool(int(os.getenv("CUGUGA_MRCI_GRAD_PROFILE", "0")))
    t0 = time.perf_counter()

    # Run target MRCISD
    mrci_res = mrci_from_mc(
        mc,
        method="mrcisd",
        plus_q=False,
        return_integrals=True,
        hop_backend=hop_backend,
        max_virt_e=int(max_virt_e),
        max_cycle=max_cycle,
        tol=conv_tol,
        correlate_inactive=int(correlate_inactive),
    )
    t1 = time.perf_counter()

    from asuka.mrci.rdm_mrcisd import make_rdm12_mrcisd, prepare_mrcisd_rdm_workspace  # noqa: PLC0415

    n_act_mc = int(getattr(mc, "ncas", 0))
    if n_act_mc <= 0:
        raise ValueError("mc.ncas must be positive")

    ncore_mc = int(getattr(mc, "ncore", 0))
    correlate_inactive_i = int(correlate_inactive)
    if correlate_inactive_i < 0 or correlate_inactive_i > ncore_mc:
        raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= mc.ncore")
    ncore_frozen = ncore_mc - correlate_inactive_i
    n_act_int = n_act_mc + correlate_inactive_i

    n_virt = int(mrci_res.result.drt.norb) - int(n_act_int)
    if n_virt < 0:
        raise RuntimeError("invalid correlated space: drt.norb < n_act_int")

    rdm_ws = prepare_mrcisd_rdm_workspace(
        mrci_res.result.drt,
        n_act=n_act_int,
        n_virt=n_virt,
        nelec=int(mrci_res.result.drt.nelec),
        twos=int(mrci_res.result.drt.twos_target),
        max_virt_e=int(max_virt_e),
    )
    t2 = time.perf_counter()
    dm1_corr, dm2_corr = make_rdm12_mrcisd(rdm_ws, mrci_res.result.ci, rdm_backend=rdm_backend)
    t3 = time.perf_counter()

    ints_cache = None
    eri4_corr = None
    aapa_core = None
    if getattr(mrci_res, "integrals", None) is not None:
        eri4_corr = np.asarray(mrci_res.integrals.eri4, dtype=np.float64, order="C")
        if ncore_frozen > 0:
            from asuka.cueri.ao2mo import ao2mo_kernel  # noqa: PLC0415

            mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
            mo_core = mo_coeff[:, :ncore_frozen]
            mo_corr = mo_coeff[:, ncore_frozen:]
            ncor = int(mo_corr.shape[1])

            aapa_core = ao2mo_kernel(mc.mol, (mo_corr, mo_corr, mo_core, mo_corr), compact=False)
            aapa_core = np.asarray(aapa_core, dtype=np.float64).reshape(ncor, ncor, int(ncore_frozen), ncor)
    else:
        ints_cache = _build_mrcisd_integral_cache(mc.mol, mc.mo_coeff, ncore_frozen)
        eri4_corr = ints_cache.eri4_corr
        aapa_core = ints_cache.aapa_core
    t4 = time.perf_counter()
    gfock = _build_gfock_mrcisd(
        mc.mol,
        mc._scf,
        mc.mo_coeff,
        ncore_frozen,
        dm1_corr,
        dm2_corr,
        eri4_corr=eri4_corr,
        aapa_core=aapa_core,
    )
    t5 = time.perf_counter()

    # Unrelaxed (variational) part for the target energy functional.
    mf_grad = mc._scf.nuc_grad_method()
    try:
        gnuc = mf_grad.grad_nuc(atmlst=atmlst)
    except TypeError:
        gnuc = mf_grad.grad_nuc(mc.mol, atmlst=atmlst)

    de = _grad_elec_mrcisd(
        mc.mol,
        mf_grad,
        mc.mo_coeff,
        ncore_frozen,
        dm1_corr,
        dm2_corr,
        gfock,
        list(range(mc.mol.natm)) if atmlst is None else list(atmlst),
    )
    ham = np.asarray(gnuc) + np.asarray(de)
    t6 = time.perf_counter()

    # Z-vector (Lagrange) response in the reference CASSCF parameter space.
    rhs_orb = mc.pack_uniq_var(gfock - gfock.T) * 2.0
    from asuka.mcscf.zvector import solve_mcscf_zvector  # noqa: PLC0415

    mc_hess, eris = _build_internal_hessian_context(mc)
    zres = solve_mcscf_zvector(
        mc_hess,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
        rhs_ci=None,
        eris=eris,
        tol=1e-10,
        maxiter=200,
        use_newton_hessian=True,
    )
    t7 = time.perf_counter()

    Lorb = mc.unpack_uniq_var(zres.z_orb)
    from asuka.mcscf.zvector import project_ci_rhs_normalized  # noqa: PLC0415

    z_ci = project_ci_rhs_normalized(mc.ci, zres.z_ci)
    weights = getattr(mc, "weights", np.asarray([1.0], dtype=np.float64))
    de_Lorb = sacasscf.Lorb_dot_dgorb_dx(
        Lorb,
        mc,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        atmlst=list(range(mc.mol.natm)) if atmlst is None else list(atmlst),
        mf_grad=mf_grad,
        eris=eris,
        verbose=0 if verbose is None else int(verbose),
    )
    de_Lci = sacasscf.Lci_dot_dgci_dx(
        z_ci,
        weights,
        mc,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        atmlst=list(range(mc.mol.natm)) if atmlst is None else list(atmlst),
        mf_grad=mf_grad,
        eris=eris,
        verbose=0 if verbose is None else int(verbose),
    )
    t8 = time.perf_counter()

    grad = ham + np.asarray(de_Lorb) + np.asarray(de_Lci)

    if profile:
        print(
            "[asuka.mrci.grad] stages (s): "
            f"mrci={t1-t0:.3f} rdm_ws={t2-t1:.3f} rdm={t3-t2:.3f} "
            f"ints={t4-t3:.3f} gfock={t5-t4:.3f} ham={t6-t5:.3f} "
            f"z={t7-t6:.3f} resp={t8-t7:.3f} total={t8-t0:.3f}"
        )
    return mrci_res, np.asarray(grad)


def mrcisd_energy_and_grad_states_from_mc(
    mc,
    *,
    states: Sequence[int] | None = None,
    nroots: int | None = None,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
    plus_q: bool = False,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    atmlst: Optional[Sequence[int]] = None,
    max_cycle: int = 80,
    conv_tol: float = 1e-10,
    max_virt_e: int = 2,
    verbose: Optional[int] = None,
    hop_backend: str | None = None,
    correlate_inactive: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    z_method: Literal["gmres", "gcrotmk"] | None = None,
    z_restart: int | None = None,
    z_gcrotmk_k: int | None = None,
    z_recycle: bool = True,
    z_warm_start: bool = True,
):
    """Compute uncontracted MRCISD energies and analytic gradients for multiple states.

    Parameters
    ----------
    mc : Any
        A converged PySCF CASSCF / SA-CASSCF object.
    states : Sequence[int] | None, optional
        Indices of states to compute.
    nroots : int | None, optional
        Number of roots to solve for.
    root_follow : {"hungarian", "greedy"}, optional
        Algorithm to assign roots to reference states. Default is "hungarian".
    plus_q : bool, optional
        Davidson +Q correction (not supported). Default is False.
    rdm_backend : {"cuda", "cpu"}, optional
        Backend for RDM construction. Default is "cuda".
    atmlst : Sequence[int] | None, optional
        Atom indices to compute gradients for.
    max_cycle : int, optional
        Maximum iterations for the solver. Default is 80.
    conv_tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_virt_e : int, optional
        Maximum virtual electrons. Default is 2.
    verbose : int | None, optional
        Verbosity level.
    hop_backend : str | None, optional
        Hop backend.
    correlate_inactive : int, optional
        Number of inactive orbitals to correlate. Default is 0.
    z_tol : float, optional
        Tolerance for Z-vector solver. Default is 1e-10.
    z_maxiter : int, optional
        Maximum iterations for Z-vector solver. Default is 200.
    z_method : {"gmres", "gcrotmk"} | None, optional
        Iterative solver for Z-vector equation.
    z_restart : int | None, optional
        Restart parameter for Z-vector solver.
    z_gcrotmk_k : int | None, optional
        Subspace size for GCRO-DR.
    z_recycle : bool, optional
        Whether to recycle Z-vector subspace. Default is True.
    z_warm_start : bool, optional
        Whether to warm start Z-vector solver. Default is True.

    Returns
    -------
    mrci_states : MRCIFromMCStatesResult
        Result object containing energies and wavefunctions.
    roots : np.ndarray
        Assigned root indices.
    grads : list[np.ndarray]
        List of gradient arrays for each state.
    """

    if plus_q:
        raise NotImplementedError(
            "Analytic gradients for Davidson +Q corrected energies are not implemented yet."
        )

    profile = bool(int(os.getenv("CUGUGA_MRCI_GRAD_PROFILE", "0")))
    t0 = time.perf_counter()

    if str(root_follow).strip().lower() not in ("hungarian", "greedy"):
        raise ValueError("root_follow must be 'hungarian' or 'greedy'")

    mrci_states = mrci_states_from_mc(
        mc,
        method="mrcisd",
        states=states,
        nroots=nroots,
        return_integrals=True,
        hop_backend=hop_backend,
        max_virt_e=int(max_virt_e),
        tol=conv_tol,
        max_cycle=max_cycle,
        correlate_inactive=int(correlate_inactive),
    )
    t1 = time.perf_counter()

    # Map each requested reference state (row) to a unique MRCI root (column).
    roots = assign_roots_by_overlap(mrci_states.mrci.overlap_ref_root, method=root_follow)

    ncore_frozen = int(mrci_states.ncore)
    ints_cache = None
    eri4_corr = None
    aapa_core = None
    if getattr(mrci_states, "integrals", None) is not None:
        eri4_corr = np.asarray(mrci_states.integrals.eri4, dtype=np.float64, order="C")
        if ncore_frozen > 0:
            from asuka.cueri.ao2mo import ao2mo_kernel  # noqa: PLC0415

            mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
            mo_core = mo_coeff[:, :ncore_frozen]
            mo_corr = mo_coeff[:, ncore_frozen:]
            ncor = int(mo_corr.shape[1])

            aapa_core = ao2mo_kernel(mc.mol, (mo_corr, mo_corr, mo_core, mo_corr), compact=False)
            aapa_core = np.asarray(aapa_core, dtype=np.float64).reshape(ncor, ncor, int(ncore_frozen), ncor)
    else:
        ints_cache = _build_mrcisd_integral_cache(mc.mol, mc.mo_coeff, ncore_frozen)
        eri4_corr = ints_cache.eri4_corr
        aapa_core = ints_cache.aapa_core
    t2 = time.perf_counter()
    mf_grad = mc._scf.nuc_grad_method()
    try:
        gnuc = mf_grad.grad_nuc(atmlst=atmlst)
    except TypeError:
        gnuc = mf_grad.grad_nuc(mc.mol, atmlst=atmlst)
    gnuc = np.asarray(gnuc)

    from asuka.mcscf.zvector import build_mcscf_hessian_operator, solve_mcscf_zvector  # noqa: PLC0415

    mc_hess, eris = _build_internal_hessian_context(mc)
    hess_op = build_mcscf_hessian_operator(
        mc_hess,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        eris=eris,
        use_newton_hessian=True,
    )
    weights = getattr(mc, "weights", np.asarray([1.0], dtype=np.float64))
    atmlst_eff = list(range(mc.mol.natm)) if atmlst is None else list(atmlst)
    nsolve = int(len(mrci_states.states))
    if z_method is None:
        method_z = "gcrotmk" if nsolve > 1 else "gmres"
    else:
        method_z = str(z_method).strip().lower()
    if method_z not in ("gmres", "gcrotmk"):
        raise ValueError("z_method must be 'gmres' or 'gcrotmk'")
    gcrotmk_k_use = z_gcrotmk_k
    if method_z == "gcrotmk" and gcrotmk_k_use is None:
        gcrotmk_k_use = 10
    recycle_space = [] if (method_z == "gcrotmk" and bool(z_recycle)) else None
    x0_z = None

    from asuka.mcscf.zvector import project_ci_rhs_normalized  # noqa: PLC0415

    from asuka.mrci.rdm_mrcisd import make_rdm12_mrcisd, prepare_mrcisd_rdm_workspace  # noqa: PLC0415

    rdm_ws = prepare_mrcisd_rdm_workspace(
        mrci_states.mrci.drt,
        n_act=int(mrci_states.n_act),
        n_virt=int(mrci_states.n_virt),
        nelec=int(mrci_states.nelec),
        twos=int(mrci_states.twos),
        max_virt_e=int(max_virt_e),
    )
    t3 = time.perf_counter()

    grads: list[np.ndarray] = []
    for k, state in enumerate(mrci_states.states):
        root = int(roots[k])
        ci = mrci_states.mrci.ci[root]
        ts0 = time.perf_counter()
        dm1_corr, dm2_corr = make_rdm12_mrcisd(rdm_ws, ci, rdm_backend=rdm_backend)
        ts1 = time.perf_counter()
        gfock = _build_gfock_mrcisd(
            mc.mol,
            mc._scf,
            mc.mo_coeff,
            ncore_frozen,
            dm1_corr,
            dm2_corr,
            eri4_corr=eri4_corr,
            aapa_core=aapa_core,
        )
        ts2 = time.perf_counter()

        # Unrelaxed part
        de = _grad_elec_mrcisd(
            mc.mol,
            mf_grad,
            mc.mo_coeff,
            ncore_frozen,
            dm1_corr,
            dm2_corr,
            gfock,
            atmlst_eff,
        )
        ham = gnuc + np.asarray(de)
        ts3 = time.perf_counter()

        # Z-vector response
        rhs_orb = mc.pack_uniq_var(gfock - gfock.T) * 2.0
        zres = solve_mcscf_zvector(
            mc,
            rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
            rhs_ci=None,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            method=method_z,
            restart=z_restart,
            gcrotmk_k=gcrotmk_k_use,
            recycle_space=recycle_space,
            x0=x0_z if bool(z_warm_start) else None,
            hessian_op=hess_op,
        )
        ts4 = time.perf_counter()
        if bool(z_warm_start):
            x0_z = np.asarray(zres.z_packed, dtype=np.float64).ravel()
        Lorb = mc.unpack_uniq_var(zres.z_orb)
        z_ci = project_ci_rhs_normalized(mc.ci, zres.z_ci)
        de_Lorb = sacasscf.Lorb_dot_dgorb_dx(
            Lorb,
            mc,
            mo_coeff=mc.mo_coeff,
            ci=mc.ci,
            atmlst=atmlst_eff,
            mf_grad=mf_grad,
            eris=eris,
            verbose=0 if verbose is None else int(verbose),
        )
        de_Lci = sacasscf.Lci_dot_dgci_dx(
            z_ci,
            weights,
            mc,
            mo_coeff=mc.mo_coeff,
            ci=mc.ci,
            atmlst=atmlst_eff,
            mf_grad=mf_grad,
            eris=eris,
            verbose=0 if verbose is None else int(verbose),
        )
        grads.append(np.asarray(ham) + np.asarray(de_Lorb) + np.asarray(de_Lci))
        ts5 = time.perf_counter()

        if profile:
            print(
                "[asuka.mrci.grad] state "
                f"{int(state)} (root {root}) stages (s): "
                f"rdm={ts1-ts0:.3f} gfock={ts2-ts1:.3f} ham={ts3-ts2:.3f} "
                f"z={ts4-ts3:.3f} resp={ts5-ts4:.3f} total={ts5-ts0:.3f}"
            )

    t4 = time.perf_counter()
    if profile:
        print(
            "[asuka.mrci.grad] common stages (s): "
            f"mrci_states={t1-t0:.3f} ints={t2-t1:.3f} rdm_ws={t3-t2:.3f} total={t4-t0:.3f}"
        )

    return mrci_states, roots, grads


# Alias for convenience
mrcisd_energy_and_grad_from_mc = mrcisd_energy_and_grad


def ic_mrcisd_energy_and_grad_states_from_mc(
    mc,
    *,
    states: Sequence[int] | None = None,
    contraction: str = "fic",
    backend: str = "semi_direct",
    sc_backend: str = "otf",
    symmetry: bool = True,
    allow_same_external: bool = True,
    allow_same_internal: bool = True,
    norm_min_singles: float = 0.0,
    norm_min_doubles: float = 0.0,
    s_tol: float = 1e-12,
    solver: str = "davidson",
    dense_nlab_max: int = 250,
    n_virt: int | None = None,
    max_virt_e: int = 2,
    hop_backend: str | None = None,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    plus_q: bool = False,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    atmlst: Optional[Sequence[int]] = None,
    max_cycle: int = 80,
    conv_tol: float = 1e-10,
    verbose: Optional[int] = None,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    z_method: Literal["gmres", "gcrotmk"] | None = None,
    z_restart: int | None = None,
    z_gcrotmk_k: int | None = None,
    z_recycle: bool = True,
    z_warm_start: bool = True,
):
    """Compute ic-MRCISD energies and analytic gradients for multiple states.

    Current implementation notes
    ----------------------------
    - Uses Phase-3 direct contracted densities for `dm1_corr/dm2_corr` when available
      (no CSF-space reconstruction for densities).
    - Still uses CSF-space reconstruction to obtain the **reference CI RHS** for the
      CP-CASSCF Z-vector solve (validation backend).
    """

    if plus_q:
        raise NotImplementedError(
            "Analytic gradients for Davidson +Q corrected energies are not implemented yet."
        )

    contraction_s = str(contraction).strip().lower()
    backend_s = str(backend).strip().lower()
    if contraction_s not in ("fic", "sc"):
        raise ValueError("contraction must be 'fic' or 'sc'")
    if backend_s != "semi_direct":
        raise NotImplementedError(
            "Analytic ic-MRCISD gradients (reconstruction backend) require backend='semi_direct'."
        )

    # Default: all available CASSCF roots, else [0].
    if states is None:
        if isinstance(getattr(mc, "ci", None), (list, tuple)):
            states = list(range(len(mc.ci)))
        else:
            states = [0]
    states = [int(s) for s in states]
    if not states:
        raise ValueError("states must be non-empty")

    # For now, require full correlated (non-core) orbital space.
    ncore = int(getattr(mc, "ncore", 0))
    n_act = int(getattr(mc, "ncas", 0))
    nmo = int(np.asarray(mc.mo_coeff).shape[1])
    nvirt_all = nmo - ncore - n_act
    if n_virt is not None and int(n_virt) != int(nvirt_all):
        raise NotImplementedError(
            "Analytic ic-MRCISD gradients currently require n_virt=None (all virtuals)."
        )

    # Integral caches (geometry-level).
    ints_cache = _build_mrcisd_integral_cache(mc.mol, mc.mo_coeff, ncore)
    mf_grad = mc._scf.nuc_grad_method()
    try:
        gnuc = mf_grad.grad_nuc(atmlst=atmlst)
    except TypeError:
        gnuc = mf_grad.grad_nuc(mc.mol, atmlst=atmlst)
    gnuc = np.asarray(gnuc)

    # Frozen-core correlated integrals for the residual-based CI RHS.
    from asuka.mrci.frozen_core import _build_frozen_core_mo_integrals_pyscf  # noqa: PLC0415

    mo = np.asarray(mc.mo_coeff, dtype=np.float64)
    mo_core = mo[:, :ncore]
    mo_corr = mo[:, ncore:]
    mf_or_mc = getattr(mc, "_scf", mc)
    ints = _build_frozen_core_mo_integrals_pyscf(mol=mc.mol, mf_or_mc=mf_or_mc, mo_core=mo_core, mo_corr=mo_corr)

    # Z-vector setup
    mc_hess, eris = _build_internal_hessian_context(mc)
    weights = getattr(mc, "weights", np.asarray([1.0], dtype=np.float64))
    atmlst_eff = list(range(mc.mol.natm)) if atmlst is None else list(atmlst)

    from asuka.mcscf.zvector import (  # noqa: PLC0415
        build_mcscf_hessian_operator,
        prepare_ci_rhs_for_zvector,
        project_ci_rhs_normalized,
        solve_mcscf_zvector,
    )
    from asuka.mrci.ic_rdm import ic_mrcisd_make_rdm12, ic_mrcisd_make_rdm12_phase3  # noqa: PLC0415
    from asuka.mrci.ic_reconstruct import (  # noqa: PLC0415
        ic_mrcisd_reference_ci_rhs_from_residual,
        reconstruct_uncontracted_ci_from_ic_mrcisd,
        reconstruct_uncontracted_ci_from_ic_mrcisd_staged,
    )

    hess_op = build_mcscf_hessian_operator(
        mc_hess,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        eris=eris,
        use_newton_hessian=True,
    )
    nsolve = int(len(states))
    if z_method is None:
        method_z = "gcrotmk" if nsolve > 1 else "gmres"
    else:
        method_z = str(z_method).strip().lower()
    if method_z not in ("gmres", "gcrotmk"):
        raise ValueError("z_method must be 'gmres' or 'gcrotmk'")
    gcrotmk_k_use = z_gcrotmk_k
    if method_z == "gcrotmk" and gcrotmk_k_use is None:
        gcrotmk_k_use = 10
    recycle_space = [] if (method_z == "gcrotmk" and bool(z_recycle)) else None
    x0_z = None

    mrci_results: list[Any] = []
    grads: list[np.ndarray] = []
    for state in states:
        # Run contracted ic-MRCISD for this reference state.
        mrci_res = mrci_from_mc(
            mc,
            method="ic_mrcisd",
            state=int(state),
            n_virt=None,
            max_virt_e=int(max_virt_e),
            hop_backend=hop_backend,
            tol=float(conv_tol),
            max_cycle=int(max_cycle),
            contract_nthreads=int(contract_nthreads),
            contract_blas_nthreads=contract_blas_nthreads,
            precompute_epq=bool(precompute_epq),
            contraction=str(contraction_s),
            backend=str(backend_s),
            sc_backend=str(sc_backend),
            symmetry=bool(symmetry),
            allow_same_external=bool(allow_same_external),
            allow_same_internal=bool(allow_same_internal),
            norm_min_singles=float(norm_min_singles),
            norm_min_doubles=float(norm_min_doubles),
            s_tol=float(s_tol),
            solver=str(solver),
            dense_nlab_max=int(dense_nlab_max),
        )
        ic_res = mrci_res.result

        # Normalize reference CI for reconstruction.
        ci0 = mc.ci[int(state)] if isinstance(getattr(mc, "ci", None), (list, tuple)) else mc.ci
        ci0 = np.asarray(ci0, dtype=np.float64).ravel()
        nrm = float(np.linalg.norm(ci0))
        if not np.isfinite(nrm) or nrm <= 0.0:
            raise ValueError("invalid reference CI vector (zero norm)")
        ci0 = ci0 / nrm

        drt_mrci, ci_mrci = reconstruct_uncontracted_ci_from_ic_mrcisd_staged(ic_res, ci_cas=ci0)
        reconstructed = (drt_mrci, ci_mrci)

        try:
            dm1_corr, dm2_corr = ic_mrcisd_make_rdm12_phase3(
                ic_res,
                ci_cas=ci0,
                backend="direct",
                rdm_backend=rdm_backend,
            )
        except NotImplementedError:
            dm1_corr, dm2_corr = ic_mrcisd_make_rdm12(
                ic_res,
                ci_cas=ci0,
                contraction=contraction_s,  # type: ignore[arg-type]
                backend="reconstruct",
                reconstructed=reconstructed,
                rdm_backend=rdm_backend,
            )

        gfock = _build_gfock_mrcisd(
            mc.mol,
            mc._scf,
            mc.mo_coeff,
            ncore,
            dm1_corr,
            dm2_corr,
            eri4_corr=ints_cache.eri4_corr,
            aapa_core=ints_cache.aapa_core,
        )

        # Unrelaxed electronic gradient contribution.
        de = _grad_elec_mrcisd(
            mc.mol,
            mf_grad,
            mc.mo_coeff,
            ncore,
            dm1_corr,
            dm2_corr,
            gfock,
            atmlst_eff,
        )
        ham = gnuc + np.asarray(de)

        # CP-CASSCF Z-vector: orbital RHS from gfock, CI RHS from residual reconstruction.
        rhs_orb = mc.pack_uniq_var(gfock - gfock.T) * 2.0
        rhs_ci_state = ic_mrcisd_reference_ci_rhs_from_residual(
            ic_res,
            ci_cas=ci0,
            h1e=ints.h1e,
            eri=ints.eri4,
            reconstructed=reconstructed,
            contract_nthreads=int(contract_nthreads),
            contract_blas_nthreads=contract_blas_nthreads,
        )

        if isinstance(getattr(mc, "ci", None), (list, tuple)):
            rhs_ci = [np.zeros_like(np.asarray(v)) for v in mc.ci]
            rhs_ci[int(state)] = rhs_ci_state.reshape(np.asarray(mc.ci[int(state)]).shape)
        else:
            rhs_ci = rhs_ci_state.reshape(np.asarray(mc.ci).shape)

        rhs_ci = prepare_ci_rhs_for_zvector(ci0=mc.ci, rhs_ci=rhs_ci, project_normalized=True)

        zres = solve_mcscf_zvector(
            mc,
            rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
            rhs_ci=rhs_ci,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            method=method_z,
            restart=z_restart,
            gcrotmk_k=gcrotmk_k_use,
            recycle_space=recycle_space,
            x0=x0_z if bool(z_warm_start) else None,
            hessian_op=hess_op,
        )
        if bool(z_warm_start):
            x0_z = np.asarray(zres.z_packed, dtype=np.float64).ravel()

        Lorb = mc.unpack_uniq_var(zres.z_orb)
        z_ci = project_ci_rhs_normalized(mc.ci, zres.z_ci)
        de_Lorb = sacasscf.Lorb_dot_dgorb_dx(
            Lorb,
            mc,
            mo_coeff=mc.mo_coeff,
            ci=mc.ci,
            atmlst=atmlst_eff,
            mf_grad=mf_grad,
            eris=eris,
            verbose=0 if verbose is None else int(verbose),
        )
        de_Lci = sacasscf.Lci_dot_dgci_dx(
            z_ci,
            weights,
            mc,
            mo_coeff=mc.mo_coeff,
            ci=mc.ci,
            atmlst=atmlst_eff,
            mf_grad=mf_grad,
            eris=eris,
            verbose=0 if verbose is None else int(verbose),
        )

        mrci_results.append(mrci_res)
        grads.append(np.asarray(ham) + np.asarray(de_Lorb) + np.asarray(de_Lci))

    return mrci_results, grads
