from __future__ import annotations

"""CASCI drivers.

This module supports two workflows:
- 1-electron integrals: from `asuka.frontend.scf` / `asuka.frontend.one_electron`.
- 2-electron integrals (active space):
    - DF: streamed density fitting build via `cuERI` active space DF (GPU).
    - Dense CPU: exact active-space eris via `cuERI` CPU backend (Cartesian AOs).

The initial goal is a CASCI energy driver that can be used as a building block for the CASSCF implementation.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import time
import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuda.active_space_df.active_space_integrals import (
    build_device_dfmo_integrals_cueri_dense_rys,
    build_device_dfmo_integrals_cueri_df,
)
from asuka.hf import df_scf as _df_scf
from asuka.hf import df_jk
from asuka.solver import GUGAFCISolver

from asuka.frontend.molecule import Molecule
from asuka.frontend.scf import RHFDFRunResult, ROHFDFRunResult, UHFDFRunResult

from .orbital_grad import orbital_gradient_dense, orbital_gradient_df
from .state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from .uhf_guess import spatialize_uhf_mo_coeff


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return np.asarray(a, dtype=np.float64)
    if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        return np.asarray(cp.asnumpy(a), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _infer_max_l_from_ao_basis(ao_basis: Any) -> int:
    shell_l = getattr(ao_basis, "shell_l", None)
    if shell_l is None:
        return 5
    arr = np.asarray(shell_l, dtype=np.int32).ravel()
    if int(arr.size) == 0:
        return 0
    return int(np.max(arr))


def _nelecas_total(nelecas: int | tuple[int, int]) -> int:
    """Return total active electrons from `nelecas` specification."""
    if isinstance(nelecas, (int, np.integer)):
        total = int(nelecas)
    elif isinstance(nelecas, (tuple, list)) and len(nelecas) == 2:
        na = int(nelecas[0])
        nb = int(nelecas[1])
        if na < 0 or nb < 0:
            raise ValueError("nelecas tuple entries must be >= 0")
        total = na + nb
    else:
        raise ValueError("nelecas must be an int or a length-2 tuple/list")
    if total < 0:
        raise ValueError("nelecas total must be >= 0")
    return total


def validate_active_space_configuration(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    context: str = "CAS",
) -> None:
    """Validate active-space electron accounting and orbital bounds.

    This guard catches invalid configurations early, before costly CI/CASSCF work.
    """
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    if ncore_i < 0:
        raise ValueError(f"{context}: ncore must be >= 0")
    if ncas_i <= 0:
        raise ValueError(f"{context}: ncas must be > 0")

    C = mo_coeff if mo_coeff is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError(f"{context}: scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        if len(C) != 2:
            raise ValueError(f"{context}: mo_coeff tuple must be (Ca, Cb)")
        shape0 = getattr(C[0], "shape", None)
        if shape0 is None or len(shape0) != 2:
            raise ValueError(f"{context}: mo_coeff[0] must be 2D")
        nmo = int(shape0[1])
    else:
        shape = getattr(C, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(f"{context}: mo_coeff must be 2D")
        nmo = int(shape[1])

    if ncore_i + ncas_i > nmo:
        raise ValueError(
            f"{context}: invalid active space: ncore+ncas={ncore_i + ncas_i} exceeds nmo={nmo}"
        )

    nelec_tot = int(getattr(scf_out.mol, "nelectron"))
    nelec_act = _nelecas_total(nelecas)
    if nelec_act > 2 * ncas_i:
        raise ValueError(
            f"{context}: invalid active space: nelecas={nelec_act} exceeds 2*ncas={2 * ncas_i}"
        )

    lhs = 2 * ncore_i + nelec_act
    if lhs != nelec_tot:
        suggest_core_msg = ""
        remain = nelec_tot - nelec_act
        if remain >= 0 and remain % 2 == 0:
            suggest_core = remain // 2
            suggest_core_msg = f" Suggested ncore={suggest_core} for nelecas={nelec_act}."
        suggest_nele_msg = ""
        suggest_nele = nelec_tot - 2 * ncore_i
        if 0 <= suggest_nele <= 2 * ncas_i:
            suggest_nele_msg = f" Suggested nelecas={suggest_nele} for ncore={ncore_i}."
        raise ValueError(
            f"{context}: invalid electron accounting: 2*ncore + nelecas_total = {lhs}, "
            f"but molecule has nelectron = {nelec_tot}."
            f"{suggest_core_msg}{suggest_nele_msg}"
        )


def _build_dfmo_integrals_from_df_B(
    B_ao: np.ndarray,
    C_active: np.ndarray,
    *,
    profile: dict | None = None,
) -> DFMOIntegrals:
    """Build CPU DF MO integrals from AO DF factors and active MOs.

    Parameters
    ----------
    B_ao : np.ndarray
        AO density fitting factors, shape (nao, nao, naux).
    C_active : np.ndarray
        Active molecular orbital coefficients, shape (nao, ncas).
    profile : dict | None, optional
        Dictionary to store profiling information.

    Returns
    -------
    DFMOIntegrals
        The constructed density-fitted MO integrals.
    """

    B_ao = np.asarray(B_ao, dtype=np.float64, order="C")
    C_active = np.asarray(C_active, dtype=np.float64, order="C")
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, ncas)")
    nao0, nao1, naux = map(int, B_ao.shape)
    if nao0 != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if int(C_active.shape[0]) != int(nao0):
        raise ValueError("B_ao and C_active nao mismatch")

    ncas = int(C_active.shape[1])
    if ncas <= 0:
        raise ValueError("C_active must have ncas > 0")

    t0 = None
    if profile is not None:
        import time

        t0 = time.perf_counter()

    # l[p,q,Q] = sum_{μ,ν} C[μ,p] B[μ,ν,Q] C[ν,q]
    tmp = np.tensordot(B_ao, C_active, axes=([1], [0]))  # (nao, naux, ncas)
    tmp = np.transpose(tmp, (0, 2, 1))  # (nao, ncas, naux)
    l_pqQ = np.tensordot(C_active.T, tmp, axes=([1], [0]))  # (ncas, ncas, naux)
    l_full = np.asarray(l_pqQ.reshape(ncas * ncas, naux), dtype=np.float64, order="C")

    pair_norm = np.linalg.norm(l_full, axis=1)
    j_ps = np.einsum("pql,qsl->ps", l_pqQ, l_pqQ, optimize=True)

    if profile is not None and t0 is not None:
        import time

        profile["t_build_dfmo_s"] = float(time.perf_counter() - float(t0))
        profile["ncas"] = int(ncas)
        profile["naux"] = int(naux)
        profile["l_full_nbytes"] = int(getattr(l_full, "nbytes", 0))

    return DFMOIntegrals(
        norb=int(ncas),
        l_full=l_full,
        j_ps=np.asarray(j_ps, dtype=np.float64, order="C"),
        pair_norm=np.asarray(pair_norm, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class CASCIResult:
    """Result of a CASCI calculation.

    Attributes
    ----------
    mol : Molecule
        The molecule object.
    basis_name : str
        Name of the basis set.
    auxbasis_name : str
        Name of the auxiliary basis set.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    nroots : int
        Number of roots requested.
    ecore : float
        Core energy (nuclear repulsion + core hamiltonian).
    e_tot : float | np.ndarray
        Total energy. Scalar if nroots=1, else array of shape (nroots,).
    ci : Any
        CI vector(s).
    mo_coeff : Any
        Molecular orbital coefficients.
    h1eff : np.ndarray
        Effective active space one-electron hamiltonian.
    eri : Any
        Active space two-electron integrals.
    scf : Any
        The underlying SCF result object from the frontend calculation.
    scf_out : Any | None
        The full frontend run result object (includes DF factors, hcore, etc).
    profile : dict | None
        Performance profiling data, if collected.
    """
    mol: Molecule
    basis_name: str
    auxbasis_name: str
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    nroots: int
    ecore: float
    e_tot: float | np.ndarray
    ci: Any
    mo_coeff: Any
    h1eff: np.ndarray  # (ncas,ncas), float64
    eri: Any
    scf: Any
    profile: dict | None = None
    scf_out: Any | None = None


@dataclass(frozen=True)
class _CASCIDFIntegrals:
    """Intermediate result structure for DF integral construction.

    Attributes
    ----------
    h1eff : np.ndarray
        Effective one-electron Hamiltonian.
    eri : Any
        Two-electron integrals (format depends on backend).
    ecore : float
        Core energy.
    mo_coeff : Any
        Validated/resolved MO coefficients.
    C_cas : Any
        Active MO slice of coefficients.
    """
    h1eff: np.ndarray
    eri: Any
    ecore: float
    mo_coeff: Any  # validated/resolved MO coefficients
    C_cas: Any     # active MO slice


def _build_casci_df_integrals(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    mo_coeff: Any | None = None,
    want_eri_mat: bool = True,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    profile: dict | None = None,
    cached_b_whitened: Any | None = None,
    cache_out: dict | None = None,
) -> _CASCIDFIntegrals:
    """Build effective 1-electron, 2-electron integrals, and core energy for DF-CASCI.

    This helper function is shared by `run_casci_df` and autotuning workflows.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        The SCF result object containing density fitting factors and integrals.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients. If None, uses `scf_out.scf.mo_coeff`.
    want_eri_mat : bool, optional
        Whether to build the ERI matrix. Defaults to True.
    aux_block_naux : int, optional
        Block size for auxiliary basis functions. Defaults to 256.
    max_tile_bytes : int, optional
        Maximum size in bytes for ERI tiles. Defaults to 256MB.
    profile : dict | None, optional
        Dictionary to store profiling information.
    cached_b_whitened : Any | None, optional
        Precomputed whitened B matrix for reuse.
    cache_out : dict | None, optional
        Output cache dictionary for storing reusable intermediates.

    Returns
    -------
    _CASCIDFIntegrals
        The constructed integrals and related data.
    """

    if not bool(getattr(scf_out.scf, "converged", False)):
        raise RuntimeError("SCF must be converged before CASCI")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    C = mo_coeff if mo_coeff is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError("scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        mo_occ = getattr(scf_out.scf, "mo_occ", None)
        if not isinstance(mo_occ, tuple) or len(mo_occ) != 2:
            raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
        C, _occ_no = spatialize_uhf_mo_coeff(S_ao=scf_out.int1e.S, mo_coeff=C, mo_occ=mo_occ)

    C = C.astype(C.dtype, copy=False)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nao, nmo = map(int, C.shape)
    if ncore + ncas > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    C_core = C[:, :ncore]
    C_cas = C[:, ncore : ncore + ncas]

    B = scf_out.df_B

    _t_core_start = time.perf_counter() if profile is not None else 0.0
    xp, _is_gpu = _df_scf._get_xp(B, C_cas)  # noqa: SLF001
    h_ao = xp.asarray(scf_out.int1e.hcore, dtype=xp.float64)

    cached_b_whitened_use = cached_b_whitened
    if cached_b_whitened_use is None and bool(_is_gpu) and B is not None:
        # Reuse SCF's whitened DF factors for active-space DF when available.
        # This avoids rebuilding metric/int3c2e/whitening inside the CAS pipeline.
        try:
            import cupy as cp  # type: ignore
        except Exception:  # pragma: no cover
            cp = None  # type: ignore
        if cp is not None and isinstance(B, cp.ndarray):  # type: ignore[attr-defined]
            if B.dtype == cp.float64 and B.ndim == 3 and int(B.shape[0]) == int(nao) and int(B.shape[1]) == int(nao):
                if hasattr(B, "flags") and not bool(B.flags.c_contiguous):
                    B = cp.ascontiguousarray(B)
                cached_b_whitened_use = B

    if ncore == 0:
        vhf_core = xp.zeros((nao, nao), dtype=xp.float64)
        ecore = float(scf_out.mol.energy_nuc())
    else:
        D_core = 2.0 * (C_core @ C_core.T)
        if bool(_is_gpu):
            import cupy as cp  # type: ignore

            B_mnQ = cp.asarray(B, dtype=cp.float64)
            BQ = cp.ascontiguousarray(B_mnQ.transpose((2, 0, 1)))
            Jc = df_jk.df_J_from_BQ_D(BQ, D_core)
            Jc = 0.5 * (Jc + Jc.T)
            Cc = cp.ascontiguousarray(C_core)
            occ_vals = cp.full((int(ncore),), 2.0, dtype=cp.float64)
            Kc = df_jk.df_K_from_BQ_Cocc(BQ, Cc, occ_vals, q_block=128)
        else:
            Jc, Kc = _df_scf._df_JK(B, D_core, want_J=True, want_K=True)  # noqa: SLF001
        vhf_core = Jc - 0.5 * Kc
        e_one = xp.sum(D_core * h_ao)
        e_two = 0.5 * xp.sum(D_core * vhf_core)
        ecore = float(float(scf_out.mol.energy_nuc()) + float(e_one.item()) + float(e_two.item()))

    if profile is not None:
        _t_core_fock = float(time.perf_counter() - _t_core_start)
        profile["t_core_fock_s"] = profile.get("t_core_fock_s", 0.0) + _t_core_fock
    h1eff_dev = C_cas.T @ (h_ao + vhf_core) @ C_cas
    h1eff = _asnumpy_f64(h1eff_dev)

    eri_prof = None
    if profile is not None:
        eri_prof = profile.setdefault("active_df", {})

    # cuERI builder works in Cartesian AO space -- transform MO coefficients
    # from spherical to Cartesian when mol.cart=False.
    C_cas_for_eri = C_cas
    cached_b_for_eri = cached_b_whitened_use
    sph_map = getattr(scf_out, "sph_map", None)
    if sph_map is not None:
        T_c2s = xp.asarray(sph_map[0], dtype=xp.float64)  # (nao_cart, nao_sph)
        C_cas_for_eri = T_c2s @ C_cas  # (nao_cart, ncas)
        # Spherical-basis cached B cannot be reused in Cartesian cuERI builder
        cached_b_for_eri = None

    eri = build_device_dfmo_integrals_cueri_df(
        scf_out.ao_basis,
        scf_out.aux_basis,
        C_cas_for_eri,
        aux_block_naux=int(aux_block_naux),
        max_tile_bytes=int(max_tile_bytes),
        want_eri_mat=bool(want_eri_mat),
        want_pair_norm=False,
        profile=eri_prof,
        cached_b_whitened=cached_b_for_eri,
        cache_out=cache_out,
    )

    return _CASCIDFIntegrals(h1eff=h1eff, eri=eri, ecore=ecore, mo_coeff=C, C_cas=C_cas)


def eval_casci_energy_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any,
    ci: Any,
    fcisolver: GUGAFCISolver,
    nroots: int = 1,
    want_eri_mat: bool = True,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    cached_b_whitened: Any | None = None,
    cache_out: dict | None = None,
) -> CASCIResult:
    """Compute CASCI energy from existing CI vectors without Davidson.

    Builds h1eff/eri/ecore at the given MOs, then evaluates
    E = ecore + <ci|H|ci> via a single contract_2e call per root.
    This is ~10x cheaper than a full Davidson solve when CI vectors
    are already good approximations (e.g. from update_orb_ci).
    """
    ncore = int(ncore)
    ncas = int(ncas)
    nroots = int(nroots)

    integrals = _build_casci_df_integrals(
        scf_out,
        ncore=ncore,
        ncas=ncas,
        mo_coeff=mo_coeff,
        want_eri_mat=want_eri_mat,
        aux_block_naux=aux_block_naux,
        max_tile_bytes=max_tile_bytes,
        cached_b_whitened=cached_b_whitened,
        cache_out=cache_out,
    )
    h1eff = integrals.h1eff
    eri = integrals.eri
    ecore = integrals.ecore
    C = integrals.mo_coeff

    ci_list = ci_as_list(ci, nroots=nroots)
    e_roots = np.empty(nroots, dtype=np.float64)
    for r, ci_r in enumerate(ci_list):
        ci_arr = np.asarray(ci_r, dtype=np.float64)
        Hci = fcisolver.contract_2e(eri, ci_arr, ncas, nelecas, h1e=h1eff)
        Hci = np.asarray(Hci, dtype=np.float64)
        e_cas = float(ci_arr.ravel().dot(Hci.ravel()))
        e_roots[r] = ecore + e_cas

    if nroots == 1:
        e_tot_val: float | np.ndarray = float(e_roots[0])
    else:
        e_tot_val = e_roots.copy()

    ci_out = ci_list if nroots > 1 else ci_list[0]
    return CASCIResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        ncore=ncore,
        ncas=ncas,
        nelecas=nelecas,
        nroots=nroots,
        ecore=float(ecore),
        e_tot=e_tot_val,
        ci=ci_out,
        mo_coeff=C,
        h1eff=h1eff,
        eri=eri,
        scf=scf_out.scf,
        scf_out=scf_out,
    )


def run_casci_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    ci0: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    matvec_backend: str = "cuda_eri_mat",
    want_eri_mat: bool = True,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    profile: dict | None = None,
    cached_b_whitened: Any | None = None,
    cache_out: dict | None = None,
    **solver_kwargs,
) -> CASCIResult:
    """Run DF-CASCI using `cuERI`-built GPU active-space DF integrals.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`. If SCF used UHF, the alpha/beta orbitals
        are converted to spatial natural orbitals of the total density matrix.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space. Can be an int (total electrons)
        or a tuple of (nalpha, nbeta).
    mo_coeff : Any | None, optional
        Molecular orbital coefficients. If None, uses `scf_out.scf.mo_coeff`.
    ci0 : Any | None, optional
        Initial guess for the CI vector.
    fcisolver : GUGAFCISolver | None, optional
        Custom solver instance. If None, a new `GUGAFCISolver` is created.
    twos : int | None, optional
        Target 2S (spin multiplicity minus 1). If None, defaults to `mol.spin`.
    nroots : int, optional
        Number of roots to solve for. Defaults to 1.
    matvec_backend : str, optional
        Backend for matrix-vector multiplication in the solver. Defaults to "cuda_eri_mat".
    want_eri_mat : bool, optional
        Whether to build the ERI matrix. Defaults to True.
    aux_block_naux : int, optional
        Block size for auxiliary basis functions. Defaults to 256.
    max_tile_bytes : int, optional
        Maximum size in bytes for ERI tiles. Defaults to 256MB.
    profile : dict | None, optional
        Dictionary to store profiling information.
    cached_b_whitened : Any | None, optional
        Precomputed whitened B matrix for reuse.
    cache_out : dict | None, optional
        Output cache dictionary for storing reusable intermediates.
    **solver_kwargs
        Additional keyword arguments passed to the solver's kernel method.

    Returns
    -------
    CASCIResult
        The result of the CASCI calculation.
    """

    validate_active_space_configuration(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        context="run_casci_df",
    )

    integrals = _build_casci_df_integrals(
        scf_out,
        ncore=ncore,
        ncas=ncas,
        mo_coeff=mo_coeff,
        want_eri_mat=want_eri_mat,
        aux_block_naux=aux_block_naux,
        max_tile_bytes=max_tile_bytes,
        profile=profile,
        cached_b_whitened=cached_b_whitened,
        cache_out=cache_out,
    )
    h1eff = integrals.h1eff
    eri = integrals.eri
    ecore = integrals.ecore
    C = integrals.mo_coeff

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        # Respect a caller-supplied solver instance, but align root count for consistency.
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    t0_ci = time.perf_counter() if profile is not None else 0.0
    e_tot, ci = fcisolver.kernel(
        h1eff,
        eri,
        int(ncas),
        nelecas,
        ci0=ci0,
        ecore=float(ecore),
        nroots=int(nroots),
        matvec_backend=str(matvec_backend),
        **solver_kwargs,
    )
    if profile is not None:
        _t_ci = float(time.perf_counter() - float(t0_ci))
        profile["t_ci_solve_s"] = profile.get("t_ci_solve_s", 0.0) + _t_ci
        # Propagate solver's internal kernel profile (Davidson stats) if available
        kprof = getattr(fcisolver, "_last_kernel_profile", None)
        if kprof is not None:
            dav_stats = kprof.get("davidson_stats", None)
            if dav_stats is not None:
                profile["davidson_stats"] = dict(dav_stats)
            if "matvec_cuda_ws_reused" in kprof:
                profile["solver_matvec_cuda_ws_reused"] = bool(kprof["matvec_cuda_ws_reused"])
            if "matvec_cuda_ws_rebuild_mismatches" in kprof:
                profile["solver_matvec_cuda_ws_rebuild_mismatches"] = list(kprof["matvec_cuda_ws_rebuild_mismatches"])
            else:
                profile["solver_matvec_cuda_ws_rebuild_mismatches"] = None
            # Accumulate key solver timing breakdowns across iterations
            for _k in ("total_s", "davidson_s", "make_hdiag_s",
                        "matvec_cuda_ws_init_s", "h_eff_eri_to_gpu_s",
                        "matvec_cuda_epq_table_build_s", "pspace_s"):
                if _k in kprof:
                    pk = f"solver_{_k}"
                    profile[pk] = profile.get(pk, 0.0) + float(kprof[_k])

    e_tot_val: float | np.ndarray
    if int(nroots) == 1:
        e_tot_val = float(e_tot)
    else:
        e_tot_val = np.asarray(e_tot, dtype=np.float64).ravel()

    return CASCIResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        nroots=int(nroots),
        ecore=float(ecore),
        e_tot=e_tot_val,
        ci=ci,
        mo_coeff=C,
        h1eff=h1eff,
        eri=eri,
        scf=scf_out.scf,
        profile=profile,
        scf_out=scf_out,
    )


@dataclass(frozen=True)
class AutoTuneCASCIResult:
    """Result of the high-level autotune driver.

    Attributes
    ----------
    solver : GUGAFCISolver
        The tuned solver instance.
    result : Any
        The `AutoTuneResult` object from `cuguga.autotune`.
    h1eff : np.ndarray
        Effective one-electron Hamiltonian.
    eri : Any
        Two-electron integrals.
    ecore : float
        Core energy.
    """
    solver: GUGAFCISolver
    result: Any  # AutoTuneResult from cuguga.autotune
    h1eff: np.ndarray
    eri: Any
    ecore: float


def autotune_casci_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    want_eri_mat: bool = True,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    # autotune-specific params
    metric: str = "hop_total_s",
    max_cycle: int = 10,
    refine: bool = False,
    apply_best: bool = True,
    verbose: bool = True,
    **autotune_kwargs,
) -> AutoTuneCASCIResult:
    """Autotune CUDA matvec settings using DF-CASCI integrals.

    Builds h1e/eri/ecore from `scf_out` (same pipeline as `run_casci_df`),
    then delegates to `cuguga.autotune` for the tuning sweep.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance.
    twos : int | None, optional
        Spin multiplicity minus 1.
    nroots : int, optional
        Number of roots. Defaults to 1.
    want_eri_mat : bool, optional
        Whether to build ERI matrix. Defaults to True.
    aux_block_naux : int, optional
        Block size for aux basis. Defaults to 256.
    max_tile_bytes : int, optional
        Max tile size. Defaults to 256MB.
    metric : str, optional
        Metric to optimize ("hop_total_s"). Defaults to "hop_total_s".
    max_cycle : int, optional
        Max tuning cycles. Defaults to 10.
    refine : bool, optional
        Whether to refine the tuning. Defaults to False.
    apply_best : bool, optional
        Whether to apply the best settings to the solver. Defaults to True.
    verbose : bool, optional
        Whether to print verbose output. Defaults to True.
    **autotune_kwargs
        Additional arguments for autotuning.

    Returns
    -------
    AutoTuneCASCIResult
        Result object whose `.solver` is configured with tuned parameters,
        ready to be passed to `run_casci_df` or `run_casscf_df`.
        The cached `.h1eff`/.`eri`/.`ecore` can be reused to avoid rebuilding integrals.
    """
    from asuka.cuguga import autotune as _low_level_autotune  # noqa: PLC0415

    integrals = _build_casci_df_integrals(
        scf_out,
        ncore=ncore,
        ncas=ncas,
        mo_coeff=mo_coeff,
        want_eri_mat=want_eri_mat,
        aux_block_naux=aux_block_naux,
        max_tile_bytes=max_tile_bytes,
    )

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))

    at_result = _low_level_autotune(
        fcisolver,
        integrals.h1eff,
        integrals.eri,
        int(ncas),
        nelecas,
        ecore=integrals.ecore,
        nroots=int(nroots),
        metric=metric,
        max_cycle=max_cycle,
        refine=refine,
        apply_best=apply_best,
        verbose=verbose,
        **autotune_kwargs,
    )

    return AutoTuneCASCIResult(
        solver=fcisolver,
        result=at_result,
        h1eff=integrals.h1eff,
        eri=integrals.eri,
        ecore=integrals.ecore,
    )


def run_casci_df_cpu(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    ci0: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    matvec_backend: str = "contract",
    eri_mat_max_bytes: int = 0,
    profile: dict | None = None,
    **solver_kwargs,
) -> CASCIResult:
    """Run DF-CASCI on CPU using DF factors from `scf_out.df_B`.

    This does not require CUDA.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients.
    ci0 : Any | None, optional
        Initial CI guess.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance.
    twos : int | None, optional
        Target 2S.
    nroots : int, optional
        Number of roots. Defaults to 1.
    matvec_backend : str, optional
        Matvec backend ("contract" or "row_oracle_df"). Defaults to "contract".
    eri_mat_max_bytes : int, optional
        Max bytes for ERI matrix if precomputed. Defaults to 0 (no limit).
    profile : dict | None, optional
        Profiling dictionary.
    **solver_kwargs
        Additional solver arguments.

    Returns
    -------
    CASCIResult
        The CASCI result.
    """

    matvec_backend_s = str(matvec_backend).strip().lower()
    if matvec_backend_s not in {"contract", "row_oracle_df"}:
        raise ValueError("run_casci_df_cpu requires matvec_backend in {'contract','row_oracle_df'}")

    validate_active_space_configuration(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        context="run_casci_df_cpu",
    )

    if not bool(getattr(scf_out.scf, "converged", False)):
        raise RuntimeError("SCF must be converged before CASCI")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    C = mo_coeff if mo_coeff is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError("scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        mo_occ = getattr(scf_out.scf, "mo_occ", None)
        if not isinstance(mo_occ, tuple) or len(mo_occ) != 2:
            raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
        C, _occ_no = spatialize_uhf_mo_coeff(S_ao=scf_out.int1e.S, mo_coeff=C, mo_occ=mo_occ)

    C = np.asarray(_asnumpy_f64(C), dtype=np.float64, order="C")
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nao, nmo = map(int, C.shape)
    if ncore + ncas > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    C_core = C[:, :ncore]
    C_cas = C[:, ncore : ncore + ncas]

    B = np.asarray(_asnumpy_f64(scf_out.df_B), dtype=np.float64, order="C")
    h_ao = np.asarray(scf_out.int1e.hcore, dtype=np.float64, order="C")

    if ncore == 0:
        D_core = np.zeros((nao, nao), dtype=np.float64)
        vhf_core = np.zeros((nao, nao), dtype=np.float64)
        ecore = float(scf_out.mol.energy_nuc())
    else:
        D_core = 2.0 * (C_core @ C_core.T)
        Jc, Kc = _df_scf._df_JK(B, D_core, want_J=True, want_K=True)  # noqa: SLF001
        vhf_core = np.asarray(Jc - 0.5 * Kc, dtype=np.float64)
        e_one = float(np.sum(D_core * h_ao))
        e_two = 0.5 * float(np.sum(D_core * vhf_core))
        ecore = float(float(scf_out.mol.energy_nuc()) + e_one + e_two)

    h1eff = np.asarray(C_cas.T @ (h_ao + vhf_core) @ C_cas, dtype=np.float64, order="C")

    eri_prof = None
    if profile is not None:
        eri_prof = profile.setdefault("active_df_cpu", {})
    eri = _build_dfmo_integrals_from_df_B(B, C_cas, profile=eri_prof)
    if int(eri_mat_max_bytes) > 0:
        _ = eri._maybe_build_eri_mat(int(eri_mat_max_bytes))  # noqa: SLF001

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    t0_ci = time.perf_counter() if profile is not None else 0.0
    e_tot, ci = fcisolver.kernel(
        h1eff,
        eri,
        int(ncas),
        nelecas,
        ci0=ci0,
        ecore=float(ecore),
        nroots=int(nroots),
        matvec_backend=str(matvec_backend_s),
        **solver_kwargs,
    )
    if profile is not None:
        _t_ci = float(time.perf_counter() - float(t0_ci))
        profile["t_ci_solve_s"] = profile.get("t_ci_solve_s", 0.0) + _t_ci
        # Propagate solver's internal kernel profile (Davidson stats) if available
        kprof = getattr(fcisolver, "_last_kernel_profile", None)
        if kprof is not None:
            dav_stats = kprof.get("davidson_stats", None)
            if dav_stats is not None:
                profile["davidson_stats"] = dict(dav_stats)
            # Accumulate key solver timing breakdowns across iterations
            for _k in ("total_s", "davidson_s", "make_hdiag_s",
                        "matvec_cuda_ws_init_s", "h_eff_eri_to_gpu_s",
                        "matvec_cuda_epq_table_build_s", "pspace_s"):
                if _k in kprof:
                    pk = f"solver_{_k}"
                    profile[pk] = profile.get(pk, 0.0) + float(kprof[_k])

    e_tot_val: float | np.ndarray
    if int(nroots) == 1:
        e_tot_val = float(e_tot)
    else:
        e_tot_val = np.asarray(e_tot, dtype=np.float64).ravel()

    return CASCIResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        nroots=int(nroots),
        ecore=float(ecore),
        e_tot=e_tot_val,
        ci=ci,
        mo_coeff=C,
        h1eff=h1eff,
        eri=eri,
        scf=scf_out.scf,
        profile=profile,
        scf_out=scf_out,
    )


def run_casci_dense_cpu(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    ci0: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    matvec_backend: str = "contract",
    eps_ao: float = 0.0,
    eps_mo: float = 0.0,
    threads: int = 0,
    blas_nthreads: int | None = None,
    max_tile_bytes: int = 256 * 1024 * 1024,
    builder: Any | None = None,
    profile: dict | None = None,
    **solver_kwargs,
) -> CASCIResult:
    """Run dense (non-DF) CASCI using `cuERI` CPU active-space ERIs.

    The SCF part is still assumed to come from the DF frontend; the dense path
    refers to the *active-space* 2-electron integrals used in the CI solve.
    This requires the `cuERI` CPU ERI extension and `mol.cart=True` convention.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients.
    ci0 : Any | None, optional
        Initial CI guess.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance.
    twos : int | None, optional
        Target 2S.
    nroots : int, optional
        Number of roots. Defaults to 1.
    matvec_backend : str, optional
        Matvec backend. Must be "contract". Defaults to "contract".
    eps_ao : float, optional
        Screening threshold for AO integrals. Defaults to 0.0.
    eps_mo : float, optional
        Screening threshold for MO integrals. Defaults to 0.0.
    threads : int, optional
        Number of threads for integral building. Defaults to 0 (auto).
    blas_nthreads : int | None, optional
        Number of BLAS threads. Use None for default.
    max_tile_bytes : int, optional
        Max tile size in bytes. Defaults to 256MB.
    builder : Any | None, optional
        Custom ERI builder instance.
    profile : dict | None, optional
        Profiling dictionary.
    **solver_kwargs
        Additional solver arguments.

    Returns
    -------
    CASCIResult
        The dense CASCI result.
    """

    matvec_backend_s = str(matvec_backend).strip().lower()
    if matvec_backend_s != "contract":
        raise ValueError("run_casci_dense_cpu currently requires matvec_backend='contract'")

    validate_active_space_configuration(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        context="run_casci_dense_cpu",
    )

    if not bool(getattr(scf_out.scf, "converged", False)):
        raise RuntimeError("SCF must be converged before CASCI")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    C = mo_coeff if mo_coeff is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError("scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        mo_occ = getattr(scf_out.scf, "mo_occ", None)
        if not isinstance(mo_occ, tuple) or len(mo_occ) != 2:
            raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
        C, _occ_no = spatialize_uhf_mo_coeff(S_ao=scf_out.int1e.S, mo_coeff=C, mo_occ=mo_occ)

    C = np.asarray(_asnumpy_f64(C), dtype=np.float64, order="C")
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nao, nmo = map(int, C.shape)
    if ncore + ncas > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    C_core = C[:, :ncore]
    C_cas = C[:, ncore : ncore + ncas]

    # Core density and core Fock potential (AO basis).
    # Use DF JK when available, otherwise fall back to dense AO ERIs.
    B = scf_out.df_B
    ao_eri = getattr(scf_out, "ao_eri", None)
    _xp_probe = B if B is not None else (ao_eri if ao_eri is not None else C_cas)
    xp, _is_gpu = _df_scf._get_xp(_xp_probe, C_cas)  # noqa: SLF001
    h_ao = xp.asarray(scf_out.int1e.hcore, dtype=xp.float64)

    if ncore == 0:
        D_core = xp.zeros((nao, nao), dtype=xp.float64)
        vhf_core = xp.zeros((nao, nao), dtype=xp.float64)
        ecore = float(scf_out.mol.energy_nuc())
    else:
        D_core = 2.0 * (xp.asarray(C_core) @ xp.asarray(C_core).T)
        from asuka.mcscf.jk_util import jk_from_2e_source  # noqa: PLC0415

        _B_xp = xp.asarray(B, dtype=xp.float64) if B is not None else None
        _ao_eri_xp = xp.asarray(ao_eri, dtype=xp.float64) if ao_eri is not None else None
        Jc, Kc = jk_from_2e_source(_B_xp, _ao_eri_xp, D_core, want_J=True, want_K=True)
        vhf_core = Jc - 0.5 * Kc
        e_one = xp.sum(D_core * h_ao)
        e_two = 0.5 * xp.sum(D_core * vhf_core)
        ecore = float(float(scf_out.mol.energy_nuc()) + float(e_one.item()) + float(e_two.item()))

    C_cas_xp = xp.asarray(C_cas, dtype=xp.float64)
    h1eff_dev = C_cas_xp.T @ (h_ao + vhf_core) @ C_cas_xp
    h1eff = _asnumpy_f64(h1eff_dev)

    # Build dense active-space ERIs on CPU (packed pair matrix, sym=4).
    if builder is None:
        from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder  # noqa: PLC0415

        builder = CuERIActiveSpaceDenseCPUBuilder(
            ao_basis=scf_out.ao_basis,
            max_l=int(_infer_max_l_from_ao_basis(scf_out.ao_basis)),
            max_tile_bytes=int(max_tile_bytes),
            threads=int(threads),
        )

    eri_prof = None
    if profile is not None:
        eri_prof = profile.setdefault("active_dense_cpu", {})
    eri = builder.build_eri_packed(
        np.asarray(C_cas, dtype=np.float64, order="C"),
        eps_ao=float(eps_ao),
        eps_mo=float(eps_mo),
        blas_nthreads=None if blas_nthreads is None else int(blas_nthreads),
        profile=eri_prof,
    )

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    e_tot, ci = fcisolver.kernel(
        h1eff,
        eri,
        int(ncas),
        nelecas,
        ci0=ci0,
        ecore=float(ecore),
        nroots=int(nroots),
        matvec_backend=str(matvec_backend),
        **solver_kwargs,
    )

    e_tot_val: float | np.ndarray
    if int(nroots) == 1:
        e_tot_val = float(e_tot)
    else:
        e_tot_val = np.asarray(e_tot, dtype=np.float64).ravel()

    return CASCIResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        nroots=int(nroots),
        ecore=float(ecore),
        e_tot=e_tot_val,
        ci=ci,
        mo_coeff=C,
        h1eff=h1eff,
        eri=eri,
        scf=scf_out.scf,
        profile=profile,
        scf_out=scf_out,
    )


def run_casci_dense_gpu(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    ci0: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    matvec_backend: str = "cuda_eri_mat",
    dense_gpu_ao_rep: str = "auto",
    dense_gpu_builder_mol: Any | None = None,
    dense_gpu_builder: Any | None = None,
    dense_exact_jk: bool = False,
    threads: int = 256,
    eps_ao: float = 0.0,
    max_tile_bytes: int = 256 * 1024 * 1024,
    profile: dict | None = None,
    **solver_kwargs,
) -> CASCIResult:
    """Run dense (non-DF) CASCI using `cuERI` GPU dense ERIs (ordered-pair `ERI_mat`).

    The SCF part is still assumed to come from the DF frontend; the dense path
    refers to the *active-space* 2-electron integrals used in the CI solve.
    This requires CUDA, CuPy, and the `cuERI` CUDA extension.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients.
    ci0 : Any | None, optional
        Initial CI guess.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance.
    twos : int | None, optional
        Target 2S.
    nroots : int, optional
        Number of roots. Defaults to 1.
    matvec_backend : str, optional
        Matvec backend. Defaults to "cuda_eri_mat".
    dense_gpu_ao_rep : str, optional
        AO representation for GPU builder. Defaults to "auto".
    dense_gpu_builder_mol : Any | None, optional
        Molecule object for GPU builder.
    dense_gpu_builder : Any | None, optional
        Custom GPU ERI builder.
    dense_exact_jk : bool, optional
        If True, use exact JK for core. Defaults to False.
    threads : int, optional
        Number of threads. Defaults to 256.
    eps_ao : float, optional
        Screening threshold for integrals. Defaults to 0.0.
    max_tile_bytes : int, optional
        Max tile size in bytes. Defaults to 256MB.
    profile : dict | None, optional
        Profiling dictionary.
    **solver_kwargs
        Additional solver arguments.

    Returns
    -------
    CASCIResult
        The dense CASCI result.
    """

    matvec_backend_s = str(matvec_backend).strip().lower()
    if matvec_backend_s not in {"cuda_eri_mat", "cuda"}:
        raise ValueError("run_casci_dense_gpu currently requires matvec_backend='cuda_eri_mat' (or 'cuda')")

    validate_active_space_configuration(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        context="run_casci_dense_gpu",
    )

    if not bool(getattr(scf_out.scf, "converged", False)):
        raise RuntimeError("SCF must be converged before CASCI")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    C = mo_coeff if mo_coeff is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError("scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        mo_occ = getattr(scf_out.scf, "mo_occ", None)
        if not isinstance(mo_occ, tuple) or len(mo_occ) != 2:
            raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
        C, _occ_no = spatialize_uhf_mo_coeff(S_ao=scf_out.int1e.S, mo_coeff=C, mo_occ=mo_occ)

    C = C.astype(C.dtype, copy=False)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nao, nmo = map(int, C.shape)
    if ncore + ncas > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    C_core = C[:, :ncore]
    C_cas = C[:, ncore : ncore + ncas]

    # Core density and core Fock potential (AO basis).
    # Use DF JK when available, otherwise fall back to dense AO ERIs.
    B = scf_out.df_B
    ao_eri = getattr(scf_out, "ao_eri", None)
    _xp_probe = B if B is not None else (ao_eri if ao_eri is not None else C_cas)
    xp, _is_gpu = _df_scf._get_xp(_xp_probe, C_cas)  # noqa: SLF001
    h_ao = xp.asarray(scf_out.int1e.hcore, dtype=xp.float64)

    dense_exact_jk = bool(dense_exact_jk)

    def _jk_for_density(D_in):
        if dense_exact_jk:
            mol_exact = getattr(scf_out, "mol", None)
            scf_exact = getattr(scf_out, "scf", None)
            get_jk = getattr(scf_exact, "get_jk", None)
            if mol_exact is None or not callable(get_jk):
                raise ValueError("dense_exact_jk=True requires scf_out.scf.get_jk")
            d_h = np.asarray(_asnumpy_f64(D_in), dtype=np.float64, order="C")
            try:
                J_h, K_h = get_jk(mol_exact, d_h, hermi=1)
            except TypeError:
                J_h, K_h = get_jk(d_h, hermi=1)
            return xp.asarray(J_h, dtype=xp.float64), xp.asarray(K_h, dtype=xp.float64)
        if B is not None:
            return _df_scf._df_JK(B, D_in, want_J=True, want_K=True)  # noqa: SLF001
        if ao_eri is not None:
            from asuka.hf.dense_jk import dense_JK_from_eri_mat_D  # noqa: PLC0415

            _ao_eri_xp = xp.asarray(ao_eri, dtype=xp.float64)
            return dense_JK_from_eri_mat_D(_ao_eri_xp, D_in, want_J=True, want_K=True)
        raise ValueError("No 2e integral source available (need df_B or ao_eri)")

    _t_core_start = time.perf_counter() if profile is not None else 0.0
    if ncore == 0:
        D_core = xp.zeros((nao, nao), dtype=xp.float64)
        vhf_core = xp.zeros((nao, nao), dtype=xp.float64)
        ecore = float(scf_out.mol.energy_nuc())
    else:
        D_core = 2.0 * (C_core @ C_core.T)
        Jc, Kc = _jk_for_density(D_core)
        vhf_core = Jc - 0.5 * Kc
        e_one = xp.sum(D_core * h_ao)
        e_two = 0.5 * xp.sum(D_core * vhf_core)
        ecore = float(float(scf_out.mol.energy_nuc()) + float(e_one.item()) + float(e_two.item()))
    if profile is not None:
        _t_core = float(time.perf_counter() - float(_t_core_start))
        profile["t_core_fock_s"] = profile.get("t_core_fock_s", 0.0) + _t_core

    # Effective 1e Hamiltonian in CAS space: C^T (hcore + vhf_core) C
    h1eff_dev = C_cas.T @ (h_ao + vhf_core) @ C_cas
    h1eff = _asnumpy_f64(h1eff_dev)

    eri_prof = None
    _t_eri_start = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        eri_prof = profile.setdefault("active_dense_gpu", {})
    eri = build_device_dfmo_integrals_cueri_dense_rys(
        scf_out.ao_basis,
        C_cas,
        mol=dense_gpu_builder_mol if dense_gpu_builder_mol is not None else getattr(scf_out, "mol", None),
        ao_rep=str(dense_gpu_ao_rep),
        builder=dense_gpu_builder,
        threads=int(threads),
        max_tile_bytes=int(max_tile_bytes),
        eps_ao=float(eps_ao),
        profile=eri_prof,
    )
    if profile is not None and isinstance(eri_prof, dict):
        _t_eri = float(time.perf_counter() - float(_t_eri_start))
        eri_prof["t_build_eri_s"] = eri_prof.get("t_build_eri_s", 0.0) + _t_eri

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    t0_ci = time.perf_counter() if profile is not None else 0.0
    e_tot, ci = fcisolver.kernel(
        h1eff,
        eri,
        int(ncas),
        nelecas,
        ci0=ci0,
        ecore=float(ecore),
        nroots=int(nroots),
        matvec_backend=str(matvec_backend_s),
        **solver_kwargs,
    )
    if profile is not None:
        _t_ci = float(time.perf_counter() - float(t0_ci))
        profile["t_ci_solve_s"] = profile.get("t_ci_solve_s", 0.0) + _t_ci
        kprof = getattr(fcisolver, "_last_kernel_profile", None)
        if kprof is not None:
            dav_stats = kprof.get("davidson_stats", None)
            if dav_stats is not None:
                profile["davidson_stats"] = dict(dav_stats)
            if "matvec_cuda_ws_reused" in kprof:
                profile["solver_matvec_cuda_ws_reused"] = bool(kprof["matvec_cuda_ws_reused"])
            if "matvec_cuda_ws_rebuild_mismatches" in kprof:
                profile["solver_matvec_cuda_ws_rebuild_mismatches"] = list(kprof["matvec_cuda_ws_rebuild_mismatches"])
            else:
                profile["solver_matvec_cuda_ws_rebuild_mismatches"] = None
            for _k in (
                "total_s",
                "davidson_s",
                "make_hdiag_s",
                "matvec_cuda_ws_init_s",
                "h_eff_eri_to_gpu_s",
                "matvec_cuda_epq_table_build_s",
                "pspace_s",
            ):
                if _k in kprof:
                    pk = f"solver_{_k}"
                    profile[pk] = profile.get(pk, 0.0) + float(kprof[_k])

    e_tot_val: float | np.ndarray
    if int(nroots) == 1:
        e_tot_val = float(e_tot)
    else:
        e_tot_val = np.asarray(e_tot, dtype=np.float64).ravel()

    return CASCIResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        nroots=int(nroots),
        ecore=float(ecore),
        e_tot=e_tot_val,
        ci=ci,
        mo_coeff=C,
        h1eff=h1eff,
        eri=eri,
        scf=scf_out.scf,
        profile=profile,
        scf_out=scf_out,
    )

def run_casci(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    backend: str = "cuda",
    df: bool = True,
    mo_coeff: Any | None = None,
    **kwargs,
) -> CASCIResult:
    """Unified CASCI driver with support for different backends and integral strategies.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of electrons in the active space.
    backend : str, optional
        "cpu" or "cuda". Defaults to "cuda".
    df : bool, optional
        If True, use DF (approximate) active-space integrals.
        If False, build dense (exact) active-space ERIs.
        Defaults to True.
    mo_coeff : Any | None, optional
        Molecular orbital coefficients.
    **kwargs
        Additional keyword arguments passed to the specific backend driver.

    Returns
    -------
    CASCIResult
        The result of the CASCI calculation.
    """

    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    df_b = bool(df)

    validate_active_space_configuration(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        context="run_casci",
    )

    C = getattr(scf_out.scf, "mo_coeff", None)
    _xp_probe = scf_out.df_B if scf_out.df_B is not None else getattr(scf_out, "ao_eri", C)
    _xp, is_gpu = _df_scf._get_xp(_xp_probe, C)  # noqa: SLF001
    if backend_s == "cuda" and not bool(is_gpu):
        raise ValueError("backend='cuda' requires scf_out with GPU arrays")

    if backend_s == "cuda" and df_b:
        return run_casci_df(scf_out, ncore=int(ncore), ncas=int(ncas), nelecas=nelecas, mo_coeff=mo_coeff, **kwargs)
    if backend_s == "cpu" and df_b:
        return run_casci_df_cpu(scf_out, ncore=int(ncore), ncas=int(ncas), nelecas=nelecas, mo_coeff=mo_coeff, **kwargs)
    if backend_s == "cuda" and not df_b:
        return run_casci_dense_gpu(scf_out, ncore=int(ncore), ncas=int(ncas), nelecas=nelecas, mo_coeff=mo_coeff, **kwargs)
    return run_casci_dense_cpu(scf_out, ncore=int(ncore), ncas=int(ncas), nelecas=nelecas, mo_coeff=mo_coeff, **kwargs)


def casci_orbital_gradient_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    casci: CASCIResult,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    root_weights: Sequence[float] | None = None,
    **solver_kwargs,
):
    """Compute the DF-based CASCI orbital gradient.

    This computes the antisymmetric orbital rotation gradient matrix used
    inside the CASSCF macro-iteration loop, but exposed for standalone CASCI workflows.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    casci : CASCIResult
        The result of the CASCI calculation.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance used to compute RDMs.
    twos : int | None, optional
        Target 2S.
    root_weights : Sequence[float] | None, optional
        Weights for state-averaged gradient.
    **solver_kwargs
        Additional arguments for the solver.

    Returns
    -------
    np.ndarray
        The computed orbital gradient matrix.
    """

    nroots = int(casci.nroots)
    weights = normalize_weights(root_weights, nroots=nroots)
    ci_list = ci_as_list(casci.ci, nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(casci.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(casci.ncas),
        nelecas=casci.nelecas,
        solver_kwargs=solver_kwargs,
    )

    return orbital_gradient_df(
        scf_out,
        C=casci.mo_coeff,
        ncore=int(casci.ncore),
        ncas=int(casci.ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
    )


def casci_orbital_gradient_dense(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    casci: CASCIResult,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    root_weights: Sequence[float] | None = None,
    dense_eps_ao: float = 0.0,
    dense_max_tile_bytes: int = 256 * 1024 * 1024,
    dense_cpu_threads: int = 0,
    dense_cpu_blas_nthreads: int | None = None,
    dense_cpu_p_block_nmo: int = 64,
    dense_gpu_threads: int = 256,
    dense_gpu_builder: Any | None = None,
    dense_exact_jk: bool = False,
    profile: dict | None = None,
    **solver_kwargs,
):
    """Compute the dense-consistent CASCI orbital gradient.

    This uses the same SCF (DF) mean-field pieces as :func:`casci_orbital_gradient_df`,
    but computes the 2-RDM contraction term using *exact* (non-DF) mixed-index ERIs
    `(p u|w x)` via `cuERI` dense tiles.

    Parameters
    ----------
    scf_out : RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult
        Output from `frontend.run_*_df`.
    casci : CASCIResult
        The result of the CASCI calculation.
    fcisolver : GUGAFCISolver | None, optional
        Solver instance.
    twos : int | None, optional
        Target 2S.
    root_weights : Sequence[float] | None, optional
        Weights for state-averaged gradient.
    dense_eps_ao : float, optional
        AO screening threshold.
    dense_max_tile_bytes : int, optional
        Max tile size in bytes.
    dense_cpu_threads : int, optional
        Number of CPU threads.
    dense_cpu_blas_nthreads : int | None, optional
        Number of BLAS threads.
    dense_cpu_p_block_nmo : int, optional
        Block size for p-index.
    dense_gpu_threads : int, optional
        Number of GPU threads.
    dense_gpu_builder : Any | None, optional
        Custom GPU builder.
    dense_exact_jk : bool, optional
        Whether to use exact JK.
    profile : dict | None, optional
        Profiling dictionary.
    **solver_kwargs
        Additional solver arguments.

    Returns
    -------
    np.ndarray
        The computed orbital gradient matrix.
    """

    nroots = int(casci.nroots)
    weights = normalize_weights(root_weights, nroots=nroots)
    ci_list = ci_as_list(casci.ci, nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(casci.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(casci.ncas),
        nelecas=casci.nelecas,
        solver_kwargs=solver_kwargs,
    )

    return orbital_gradient_dense(
        scf_out,
        C=casci.mo_coeff,
        ncore=int(casci.ncore),
        ncas=int(casci.ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        dense_eps_ao=float(dense_eps_ao),
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_cpu_threads=int(dense_cpu_threads),
        dense_cpu_blas_nthreads=None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads),
        dense_cpu_p_block_nmo=int(dense_cpu_p_block_nmo),
        dense_gpu_threads=int(dense_gpu_threads),
        dense_gpu_builder=dense_gpu_builder,
        dense_exact_jk=bool(dense_exact_jk),
        profile=profile,
    )


__all__ = [
    "CASCIResult",
    "casci_orbital_gradient_df",
    "casci_orbital_gradient_dense",
    "eval_casci_energy_df",
    "run_casci",
    "run_casci_df",
    "run_casci_df_cpu",
    "run_casci_dense_cpu",
    "run_casci_dense_gpu",
]
