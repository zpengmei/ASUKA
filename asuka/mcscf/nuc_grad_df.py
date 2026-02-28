from __future__ import annotations

import os as _os

"""Nuclear gradients for DF-CASSCF/CASCI.

This module implements a nuclear gradient path for CASSCF using:
  - Analytic AO 1e integral derivatives (dS, dT, dV) from
    :mod:`asuka.integrals.int1e_cart`
  - A Pulay overlap term using the energy-weighted density built from the
    generalized Fock matrix (same `g` as used in orbital gradients)
  - Two-electron DF derivative contractions (analytic) using
    :func:`asuka.integrals.grad.compute_df_gradient_contributions_analytic_packed_bases`
    with a finite-difference fallback on backends without derivative kernels.

Notes
-----
* The DF 2e part is **analytic** for:
  - `df_backend="cpu"` (cuERI CPU derivative tiles)
  - `df_backend="cuda"` (cuERI CUDA derivative contraction kernels; requires the cuERI CUDA extension)
  If analytic kernels are unavailable, this module falls back to FD on `B`.
* CASCI supports both:
  - **unrelaxed** (no SCF/CPHF response): :func:`casci_nuc_grad_df_unrelaxed`
  - **relaxed** (includes RHF/DF CPHF response): :func:`casci_nuc_grad_df_relaxed`
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import time
import warnings
import numpy as np

from asuka.frontend.periodic_table import atomic_number
from asuka.hf import df_scf as _df_scf
from asuka.integrals.grad import (
    compute_df_gradient_contributions_analytic_packed_bases,
    compute_df_gradient_contributions_analytic_sph,
    compute_df_gradient_contributions_fd_packed_bases,
)
from asuka.integrals.int1e_cart import contract_dS_cart, contract_dhcore_cart, contract_dS_ip_cart, shell_to_atom_map
from asuka.solver import GUGAFCISolver

from .state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from .cphf_df import solve_rhf_cphf_df

# ---------------------------------------------------------------------------
# xp (numpy / cupy) dispatch utilities
# ---------------------------------------------------------------------------
from asuka.hf.df_scf import _get_xp as _get_xp_arrays  # noqa: E402


def _resolve_xp(df_backend: str):
    """Return ``(xp, is_gpu)`` based on *df_backend* string."""
    if str(df_backend).strip().lower() == "cuda":
        try:
            import cupy as cp  # type: ignore[import-not-found]

            return cp, True
        except ImportError:
            raise RuntimeError("df_backend='cuda' requires CuPy")
    return np, False


def _log_vram(label: str) -> None:
    """Print current GPU VRAM usage (for debugging). Controlled by ASUKA_VRAM_DEBUG=1."""
    if not _os.environ.get("ASUKA_VRAM_DEBUG"):
        return
    try:
        import cupy as cp  # noqa: PLC0415
        pool = cp.get_default_memory_pool()
        free, total = cp.cuda.runtime.memGetInfo()
        used_nvidia = (total - free) / 1e9
        pool_used = pool.used_bytes() / 1e9
        pool_total = pool.total_bytes() / 1e9
        print(f"[VRAM] {label}: nvidia={used_nvidia:.2f}GB  pool_active={pool_used:.2f}GB  pool_total={pool_total:.2f}GB")
    except Exception:
        pass


def _flush_gpu_pool() -> None:
    """Return all freed CuPy pool blocks to the GPU driver to reduce memory high-water mark."""
    try:
        import cupy as cp  # noqa: PLC0415
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _as_xp_f64(xp, a):
    """Convert any array-like to *xp* float64."""
    # When target is numpy but input is a CuPy array, pull to CPU first.
    if xp is np and hasattr(a, "get"):
        a = a.get()
    return xp.asarray(a, dtype=xp.float64)


def _get_sph_T(scf_out):
    """Return the cart-to-sph matrix T or None if mol.cart=True."""
    mol = getattr(scf_out, "mol", None)
    if bool(getattr(mol, "cart", True)):
        return None
    sph_map = getattr(scf_out, "sph_map", None)
    if sph_map is None:
        raise ValueError("scf_out.sph_map is required for spherical AO gradients (cart=False)")
    return sph_map[0]  # (nao_cart, nao_sph)


def _sph_to_cart_final(T_c2s, *, D_tot_ao, W, bar_L_ao, C, B_ao_sph, scf_out, xp, coords, charges):
    """Back-transform final gradient quantities from spherical to Cartesian.

    For the DF gradient, bar_L and B must be in Cartesian AO basis to match
    the Cartesian derivative integrals.  The back-transforms are:

        D_cart  = T @ D_sph  @ T^T
        W_cart  = T @ W_sph  @ T^T
        bar_L_cart[Q,:,:] = T @ bar_L_sph[Q,:,:] @ T^T
        C_cart  = T @ C_sph

    For B, we rebuild from ao_basis/aux_basis (rebuilding is cheaper than
    storing a second copy, and back-transform T @ B_sph @ T^T is WRONG
    because T @ T^T is a projector, not identity, for l >= 2).
    """
    T_np = np.asarray(T_c2s, dtype=np.float64)
    nao_cart = int(T_np.shape[0])

    # -- D_tot and W (2D matrices) --
    D_np = _asnumpy_f64(D_tot_ao)
    D_cart = T_np @ D_np @ T_np.T
    W_np = np.asarray(W, dtype=np.float64)
    W_cart = T_np @ W_np @ T_np.T

    # -- C (MO coefficients) --
    C_np = _asnumpy_f64(C)
    C_cart = T_np @ C_np

    # -- bar_L (Q, nao, nao) → (Q, nao_cart, nao_cart) --
    bar_L_np = _asnumpy_f64(bar_L_ao)
    naux = int(bar_L_np.shape[0])
    bar_L_cart = np.empty((naux, nao_cart, nao_cart), dtype=np.float64)
    for q in range(naux):
        bar_L_cart[q] = T_np @ bar_L_np[q] @ T_np.T

    # -- B_cart: rebuild from ao_basis/aux_basis (Cartesian) --
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    B_cart = _rebuild_B_cart(ao_basis, aux_basis, xp)

    return D_cart, W_cart, bar_L_cart, C_cart, B_cart


def _rebuild_B_cart(ao_basis, aux_basis, xp):
    """Rebuild Cartesian whitened DF factors B[mu,nu,Q] from packed bases."""
    try:
        import cupy as cp  # noqa: PLC0415
        if xp is cp:
            from asuka.integrals.cueri_df import build_df_B_from_cueri_packed_bases  # noqa: PLC0415
            B = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis)
            # GPU builder may return Qmn layout — ensure mnQ
            if B.ndim == 3 and int(B.shape[0]) != int(B.shape[1]):
                B = cp.ascontiguousarray(B.transpose((1, 2, 0)))
            return _asnumpy_f64(B)
    except Exception:
        pass
    from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu  # noqa: PLC0415
    return np.asarray(build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis), dtype=np.float64)


def _transform_bar_L_to_cart(T_np: np.ndarray, bar_L_sph: Any) -> np.ndarray:
    """Transform bar_L from spherical to Cartesian AO basis (FD fallback only).

    Parameters
    ----------
    T_np : np.ndarray, shape (nao_cart, nao_sph)
    bar_L_sph : array, shape (naux, nao_sph, nao_sph)

    Returns
    -------
    np.ndarray, shape (naux, nao_cart, nao_cart)
    """
    bl = np.asarray(bar_L_sph, dtype=np.float64)
    if hasattr(bl, "get"):
        bl = bl.get()
    bl = np.asarray(bl, dtype=np.float64)
    return np.einsum("mi,Qij,nj->Qmn", T_np, bl, T_np, optimize=True)


def _asnumpy_f64(a: Any) -> np.ndarray:
    """Ensure array is numpy.float64 (moves from GPU if needed).

    Parameters
    ----------
    a : Any
        Input array.

    Returns
    -------
    np.ndarray
        Numpy array (float64).
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return np.asarray(a, dtype=np.float64)
    if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        return np.asarray(cp.asnumpy(a), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _mol_coords_charges_bohr(mol: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extracrt atomic coordinates (Bohr) and charges from Molecule object.

    Parameters
    ----------
    mol : Any
        The molecule object.

    Returns
    -------
    coords : np.ndarray
        Atomic coordinates in Bohr (natm, 3).
    charges : np.ndarray
        Nuclear charges (natm,).

    Raises
    ------
    TypeError
        If mol is not a valid Molecule-like object.
    """
    atoms = getattr(mol, "atoms_bohr", None)
    if atoms is None:
        raise TypeError("mol must be an asuka.frontend.molecule.Molecule-like object")
    coords = np.asarray([xyz for _sym, xyz in atoms], dtype=np.float64).reshape((-1, 3))
    charges = np.asarray([atomic_number(sym) for sym, _xyz in atoms], dtype=np.float64)
    return coords, charges


@dataclass(frozen=True)
class DFNucGradResult:
    """Container for a DF-based nuclear gradient.

    Attributes
    ----------
    e_tot : float
        Total energy.
    e_nuc : float
        Nuclear repulsion energy.
    grad : np.ndarray
        Gradient array (natm, 3) in Eh/Bohr.
    """

    e_tot: float
    e_nuc: float
    grad: np.ndarray


def _build_gfock_casscf_df(
    B_ao: np.ndarray,
    h_ao: np.ndarray,
    C: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (gfock_mo, D_core_ao, D_act_ao, D_tot_ao, C_act) for DF-CASSCF.

    Constructs the generalized Fock matrix in MO basis using DF intermediates.

    Parameters
    ----------
    B_ao : np.ndarray
        Density fitting tensor (nao, nao, naux).
    h_ao : np.ndarray
        Core Hamiltonian (nao, nao).
    C : np.ndarray
        MO coefficients (nao, nmo).
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    dm1_act : np.ndarray
        Active space 1-RDM (ncas, ncas).
    dm2_act : np.ndarray
        Active space 2-RDM (ncas, ncas, ncas, ncas).

    Returns
    -------
    tuple
        (gfock, D_core, D_act, D_tot, C_act)
    """

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0:
        raise ValueError("invalid ncore/ncas")

    xp, _ = _get_xp_arrays(B_ao, C)
    B_ao = xp.asarray(B_ao, dtype=xp.float64)
    h_ao = xp.asarray(h_ao, dtype=xp.float64)
    C = xp.asarray(C, dtype=xp.float64)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if h_ao.ndim != 2 or h_ao.shape[0] != h_ao.shape[1]:
        raise ValueError("h_ao must be a square 2D matrix")
    if C.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    nao, nmo = map(int, C.shape)
    if tuple(h_ao.shape) != (nao, nao):
        raise ValueError("h_ao/C nao mismatch")
    if tuple(B_ao.shape[:2]) != (nao, nao):
        raise ValueError("B_ao/C nao mismatch")

    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    dm1_act = xp.asarray(dm1_act, dtype=xp.float64)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")

    dm2_arr = xp.asarray(dm2_act, dtype=xp.float64)
    if dm2_arr.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act must have shape (ncas,ncas,ncas,ncas)")

    # AO densities
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = xp.zeros((nao, nao), dtype=xp.float64)
    D_act_ao = C_act @ dm1_act @ C_act.T
    D_tot_ao = D_core_ao + D_act_ao

    # AO potentials
    Jc, Kc = _df_scf._df_JK(B_ao, D_core_ao, want_J=True, want_K=True)  # noqa: SLF001
    Ja, Ka = _df_scf._df_JK(B_ao, D_act_ao, want_J=True, want_K=True)  # noqa: SLF001
    vhf_c_ao = Jc - 0.5 * Kc
    vhf_ca_ao = (Jc + Ja) - 0.5 * (Kc + Ka)

    # Transform AO matrices to MO
    h_mo = C.T @ h_ao @ C
    vhf_c_mo = C.T @ vhf_c_ao @ C
    vhf_ca_mo = C.T @ vhf_ca_ao @ C

    # DF MO factors for dm2 contraction (same construction as orbital_grad_df)
    X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
    L_pact = xp.einsum("mp,mvQ->pvQ", C, X, optimize=True)  # (nmo,ncas,naux)
    L_act = L_pact[ncore:nocc]  # (ncas,ncas,naux)

    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    L2 = L_act.reshape(ncas * ncas, -1)

    # T[Q,u,v] = sum_{w,x} L[w,x,Q] * dm2[w,x,u,v]
    T_flat = L2.T @ dm2_flat  # (naux,ncas^2)
    T = T_flat.reshape(L2.shape[1], ncas, ncas)
    g_dm2 = xp.einsum("puQ,Quv->pv", L_pact, T, optimize=True)  # (nmo,ncas)

    # Generalized Fock (g) matrix: only core+active columns are defined.
    gfock = xp.zeros((nmo, nmo), dtype=xp.float64)
    if ncore:
        gfock[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gfock[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1_act + g_dm2

    return gfock, D_core_ao, D_act_ao, D_tot_ao, C_act


def _build_dme0_lorb_response(
    B_ao: np.ndarray,
    h_ao: np.ndarray,
    C: np.ndarray,
    Lorb: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ppaa: np.ndarray,
    papa: np.ndarray,
    *,
    ncore: int,
    ncas: int,
) -> np.ndarray:
    """Compute response energy-weighted density from orbital Lagrange multiplier.

    Follows PySCF's ``Lorb_dot_dgorb_dx`` gfock construction.
    Returns ``dme0 = 0.5*(gfock + gfock^T)`` in AO-like basis (nao, nao).
    GPU-aware: keeps J/K builds on device when inputs are CuPy arrays.
    """
    xp, _ = _get_xp_arrays(B_ao, h_ao)

    ncore, ncas = int(ncore), int(ncas)
    nocc = ncore + ncas
    C = xp.asarray(C, dtype=xp.float64)
    L = xp.asarray(Lorb, dtype=xp.float64)
    h_ao_x = xp.asarray(h_ao, dtype=xp.float64)
    B_x = xp.asarray(B_ao, dtype=xp.float64)
    nao, nmo = int(C.shape[0]), int(C.shape[1])

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    C_L = C @ L
    C_L_core = C_L[:, :ncore]
    C_L_act = C_L[:, ncore:nocc]

    dm1 = xp.asarray(dm1_act, dtype=xp.float64)
    dm2 = xp.asarray(dm2_act, dtype=xp.float64)

    # AO densities and L-effective densities.
    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core = 2.0 * (C_L_core @ C_core.T)
        D_L_core = D_L_core + D_L_core.T
    else:
        D_core = xp.zeros((nao, nao), dtype=xp.float64)
        D_L_core = xp.zeros((nao, nao), dtype=xp.float64)

    D_act = C_act @ dm1 @ C_act.T
    D_L_act = C_L_act @ dm1 @ C_act.T
    D_L_act = D_L_act + D_L_act.T
    D_L = D_L_core + D_L_act

    # J/K builds using DF (GPU-aware via _df_JK xp dispatch).
    Jc, Kc = _df_scf._df_JK(B_x, D_core, want_J=True, want_K=True)  # noqa: SLF001
    Ja, Ka = _df_scf._df_JK(B_x, D_act, want_J=True, want_K=True)  # noqa: SLF001
    if ncore:
        JcL, KcL = _df_scf._df_JK(B_x, D_L_core, want_J=True, want_K=True)  # noqa: SLF001
    else:
        JcL = xp.zeros((nao, nao), dtype=xp.float64)
        KcL = xp.zeros((nao, nao), dtype=xp.float64)
    JaL, KaL = _df_scf._df_JK(B_x, D_L_act, want_J=True, want_K=True)  # noqa: SLF001

    vhf_c = Jc - 0.5 * Kc
    vhf_a = Ja - 0.5 * Ka
    vhfL_c = JcL - 0.5 * KcL
    vhfL_a = JaL - 0.5 * KaL

    # Build gfock in AO basis (PySCF Lorb_dot_dgorb_dx formula).
    gfock = h_ao_x @ D_L
    gfock += (vhf_c + vhf_a) @ D_L_core
    gfock += (vhfL_c + vhfL_a) @ D_core
    gfock += vhfL_c @ D_act
    gfock += vhf_c @ D_L_act

    # S^{-1} ≈ CC^T (orthogonal MO basis).
    s0_inv = C @ C.T
    gfock = s0_inv @ gfock

    # 2e active-active ERI contribution (from ppaa/papa) — small, keep on CPU.
    ppaa_np = _asnumpy_f64(ppaa)
    papa_np = _asnumpy_f64(papa)
    L_np = _asnumpy_f64(L)
    L_act_slice = L_np[:, ncore:nocc]

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=np.float64)
    aapaL = np.zeros_like(aapa)
    for i in range(nmo):
        jbuf = ppaa_np[i]
        kbuf = papa_np[i]
        aapa[:, :, i, :] = jbuf[ncore:nocc, :, :].transpose(1, 2, 0)
        aapaL[:, :, i, :] += np.tensordot(jbuf, L_act_slice, axes=((0,), (0,)))
        kk = np.tensordot(kbuf, L_act_slice, axes=((1,), (0,))).transpose(1, 2, 0)
        aapaL[:, :, i, :] += kk + kk.transpose(1, 0, 2)

    dm2_np = _asnumpy_f64(dm2)
    t1 = np.einsum("uviw,uvtw->it", aapaL, dm2_np, optimize=True)
    t2 = np.einsum("uviw,vuwt->it", aapa, dm2_np, optimize=True)
    C_np = _asnumpy_f64(C)
    C_act_np = _asnumpy_f64(C_act)
    C_L_act_np = _asnumpy_f64(C_L_act)
    gfock_np = _asnumpy_f64(gfock)
    gfock_np += C_np @ t1 @ C_act_np.T
    gfock_np += C_np @ t2 @ C_L_act_np.T

    return np.asarray(0.5 * (gfock_np + gfock_np.T), dtype=np.float64)


def _build_bar_L_casscf_df(
    B_ao: np.ndarray,
    *,
    D_core_ao: np.ndarray,
    D_act_ao: np.ndarray,
    C_act: np.ndarray,
    dm2_act: np.ndarray,
    L_act: Any | None = None,
    rho_core: Any | None = None,
) -> np.ndarray:
    """Return bar_L_ao[Q,μ,ν] = ∂E_2e/∂B[μ,ν,Q] for DF-CASSCF.

    Parameters
    ----------
    B_ao : np.ndarray
        Density fitting tensor (nao, nao, naux).
    D_core_ao : np.ndarray
        Core density matrix in AO basis.
    D_act_ao : np.ndarray
        Active density matrix in AO basis.
    C_act : np.ndarray
        Active MO coefficients (nao, ncas).
    dm2_act : np.ndarray
        Active space 2-RDM.
    L_act : Any | None, optional
        Precomputed active DF factors ``L_act[u,v,Q] = C_act^T B_Q C_act`` with
        shape ``(ncas, ncas, naux)``. If provided, avoids recomputing the
        expensive active DF contraction.
    rho_core : Any | None, optional
        Precomputed ``rho_core[Q] = B2.T @ D_core_ao.reshape(-1)`` with
        shape ``(naux,)``. If provided, avoids recomputing the core DF density
        projection.

    Returns
    -------
    np.ndarray
        The partial derivative of energy w.r.t B tensor elements.
    """

    xp, _ = _get_xp_arrays(B_ao, D_core_ao, D_act_ao, C_act, dm2_act, L_act, rho_core)
    B_ao = xp.asarray(B_ao, dtype=xp.float64)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_core_ao = xp.asarray(D_core_ao, dtype=xp.float64)
    D_act_ao = xp.asarray(D_act_ao, dtype=xp.float64)
    if D_core_ao.shape != (nao, nao) or D_act_ao.shape != (nao, nao):
        raise ValueError("AO density shape mismatch")

    # Mean-field interaction with the core potential:
    #   E_cc + E_ca = Tr(D_w * (J(Dc) - 0.5 K(Dc))), where D_w = D_act + 0.5 D_core
    D_w = D_act_ao + 0.5 * D_core_ao

    B2 = B_ao.reshape(nao * nao, naux)
    if rho_core is None:
        rho = B2.T @ D_core_ao.reshape(nao * nao)  # (naux,)
    else:
        rho = xp.asarray(rho_core, dtype=xp.float64)
        if rho.shape != (naux,):
            raise ValueError("rho_core shape mismatch")
    sigma = B2.T @ D_w.reshape(nao * nao)  # (naux,)

    # Fused bar_J + bar_K: avoids separate (naux,nao,nao) allocations for
    # bar_J, t1, t2, bar_K.  Peak is now 2× sizeof(B) instead of ~6×.
    _log_vram("    casscf_df: before bar_mean")
    bar_mean = sigma[:, None, None] * D_core_ao[None, :, :]
    bar_mean += rho[:, None, None] * D_w[None, :, :]
    _log_vram("    casscf_df: after bar_mean J")

    BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao) — view, no copy
    # Chunk over aux index so intermediates are (chunk,nao,nao) instead of
    # (naux,nao,nao), reducing peak VRAM by ~2×sizeof(B).
    _chunk = max(1, naux // 4)
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _bq = BQ[_q0:_q1]
        _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_w)
        _t += xp.matmul(xp.matmul(D_w[None, :, :], _bq), D_core_ao)
        _t *= -0.5
        bar_mean[_q0:_q1] += _t
        del _t
    _log_vram("    casscf_df: after exchange")

    # Active-active 2-RDM term:
    #   E_aa = 0.5 Σ_{uvwx} dm2_uvwx (uv|wx)
    # with (uv|wx) ≈ Σ_Q L_uv,Q L_wx,Q and L_uv,Q = C^T B_Q C in active MO space.
    C_act = xp.asarray(C_act, dtype=xp.float64)
    if C_act.ndim != 2 or int(C_act.shape[0]) != int(nao):
        raise ValueError("C_act shape mismatch")
    ncas = int(C_act.shape[1])
    if ncas <= 0:
        raise ValueError("empty active space")

    dm2_arr = xp.asarray(dm2_act, dtype=xp.float64)
    if dm2_arr.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    if L_act is None:
        # L_uv,Q in active MO indices
        X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
        L_act = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)  # (ncas,ncas,naux)
    else:
        L_act = xp.asarray(L_act, dtype=xp.float64)
        if L_act.shape != (ncas, ncas, naux):
            raise ValueError("L_act shape mismatch")

    L2 = L_act.reshape(ncas * ncas, naux)
    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2  # (ncas^2,naux), M_uv,Q = sum_{wx} dm2_uvwx L_wx,Q
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    bar_act = xp.einsum("mvQ,nv->Qmn", tmp, C_act, optimize=True)  # (naux,nao,nao)
    _log_vram("    casscf_df: after bar_act")

    bar_mean += bar_act
    del bar_act
    bar_L = bar_mean + xp.transpose(bar_mean, (0, 2, 1))
    del bar_mean
    bar_L *= 0.5
    _log_vram("    casscf_df: after sym")
    return xp.asarray(bar_L, dtype=xp.float64)


def _build_bar_L_delta_casscf_df(
    B_ao: Any,
    *,
    D_core_ao: Any,
    C_act: Any,
    dm1_delta: Any,
    dm2_delta: Any,
    rho_core: Any,
    L2: Any,
) -> Any:
    """Return Hellmann–Feynman DF derivative delta ``bar_L_K - bar_L_SA``.

    This delta is linear in the active-space RDM deltas and avoids constructing
    the full per-root ``bar_L_K`` only to subtract ``bar_L_SA``.

    Assumes the core density is root-invariant (shared orbitals), so core-core
    contributions cancel exactly in the delta.
    """

    xp, _ = _get_xp_arrays(B_ao, D_core_ao, C_act, dm1_delta, dm2_delta, rho_core, L2)
    B_ao = xp.asarray(B_ao, dtype=xp.float64)
    D_core_ao = xp.asarray(D_core_ao, dtype=xp.float64)
    C_act = xp.asarray(C_act, dtype=xp.float64)
    dm1_delta = xp.asarray(dm1_delta, dtype=xp.float64)
    dm2_delta = xp.asarray(dm2_delta, dtype=xp.float64)
    rho_core = xp.asarray(rho_core, dtype=xp.float64)
    L2 = xp.asarray(L2, dtype=xp.float64)

    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if D_core_ao.shape != (nao, nao):
        raise ValueError("D_core_ao shape mismatch")
    if C_act.ndim != 2 or int(C_act.shape[0]) != int(nao):
        raise ValueError("C_act shape mismatch")
    ncas = int(C_act.shape[1])
    if ncas <= 0:
        raise ValueError("empty active space")
    if dm1_delta.shape != (ncas, ncas):
        raise ValueError("dm1_delta shape mismatch")
    if dm2_delta.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_delta shape mismatch")
    if rho_core.shape != (naux,):
        raise ValueError("rho_core shape mismatch")
    if L2.shape != (ncas * ncas, naux):
        raise ValueError("L2 shape mismatch")

    # AO active density delta
    D_act_delta = C_act @ dm1_delta @ C_act.T

    # Mean-field delta (D_w_delta = D_act_delta because D_core cancels)
    B2 = B_ao.reshape(nao * nao, naux)
    delta_sigma = B2.T @ D_act_delta.reshape(nao * nao)  # (naux,)

    # Fused bar_J_delta + bar_K_delta to reduce peak VRAM.
    bar_mean_delta = delta_sigma[:, None, None] * D_core_ao[None, :, :]
    bar_mean_delta += rho_core[:, None, None] * D_act_delta[None, :, :]

    BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
    _chunk = max(1, naux // 4)
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _bq = BQ[_q0:_q1]
        _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_act_delta)
        _t += xp.matmul(xp.matmul(D_act_delta[None, :, :], _bq), D_core_ao)
        _t *= -0.5
        bar_mean_delta[_q0:_q1] += _t
        del _t

    # Active-active 2-RDM delta (linear in dm2)
    dm2_flat = dm2_delta.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2  # (ncas^2,naux)
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    bar_act_delta = xp.einsum("mvQ,nv->Qmn", tmp, C_act, optimize=True)  # (naux,nao,nao)

    bar_mean_delta += bar_act_delta
    del bar_act_delta
    bar_L_delta = bar_mean_delta + xp.transpose(bar_mean_delta, (0, 2, 1))
    del bar_mean_delta
    bar_L_delta *= 0.5
    return xp.asarray(bar_L_delta, dtype=xp.float64)


def _build_bar_L_df_cross(
    B_ao: np.ndarray,
    *,
    D_left: np.ndarray,
    D_right: np.ndarray,
    coeff_J: float,
    coeff_K: float,
) -> np.ndarray:
    """Return bar_L for E = coeff_J*Tr(D_left·J(D_right)) + coeff_K*Tr(D_left·K(D_right)).

    This helper is used to build the DF two-electron derivative contributions
    arising from RHF/CPHF orbital-response terms (e.g. CASCI gradients).

    Parameters
    ----------
    B_ao : np.ndarray
        Density fitting tensor.
    D_left : np.ndarray
        Left-hand density matrix.
    D_right : np.ndarray
        Right-hand density matrix.
    coeff_J : float
        Coefficient for Coulomb term.
    coeff_K : float
        Coefficient for Exchange term.

    Returns
    -------
    np.ndarray
        Contribution to bar_L.
    """

    xp, _ = _get_xp_arrays(B_ao)
    B_ao = xp.asarray(B_ao, dtype=xp.float64)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_left = xp.asarray(D_left, dtype=xp.float64)
    D_right = xp.asarray(D_right, dtype=xp.float64)
    if D_left.shape != (nao, nao) or D_right.shape != (nao, nao):
        raise ValueError("D_left/D_right shape mismatch")

    # Coulomb-like part: Tr(D_left J(D_right)) = u(D_left)·u(D_right)
    B2 = B_ao.reshape(nao * nao, naux)
    rho = B2.T @ D_right.reshape(nao * nao)  # (naux,)
    sigma = B2.T @ D_left.reshape(nao * nao)  # (naux,)

    _log_vram("    cross: before Coulomb")
    cJ = float(coeff_J)
    if cJ:
        bar = (cJ * sigma)[:, None, None] * D_right[None, :, :]
        bar += (cJ * rho)[:, None, None] * D_left[None, :, :]
    else:
        bar = xp.zeros((naux, nao, nao), dtype=xp.float64)
    _log_vram("    cross: after Coulomb")

    # Exchange-like part: Tr(D_left K(D_right)) = Σ_Q Tr(D_left B_Q D_right B_Q)
    # Chunked over aux index to keep intermediates at (chunk,nao,nao).
    cK = float(coeff_K)
    if cK:
        BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
        _chunk = max(1, naux // 4)
        for _q0 in range(0, naux, _chunk):
            _q1 = min(_q0 + _chunk, naux)
            _bq = BQ[_q0:_q1]
            _t = xp.matmul(xp.matmul(D_left[None, :, :], _bq), D_right)
            _t += xp.matmul(xp.matmul(D_right[None, :, :], _bq), D_left)
            _t *= cK
            bar[_q0:_q1] += _t
            del _t
        _log_vram("    cross: after exchange")

    _sym = bar + xp.transpose(bar, (0, 2, 1))
    del bar
    _sym *= 0.5
    _log_vram("    cross: after sym")
    return xp.asarray(_sym, dtype=xp.float64)


def casscf_nuc_grad_df(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    atmlst: Sequence[int] | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    int1e_contract_backend: Literal["auto", "cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    solver_kwargs: dict[str, Any] | None = None,
    profile: dict | None = None,
) -> DFNucGradResult:
    """DF-based nuclear gradient for a (SA-)CASSCF result.

    Combines analytic 1-electron derivatives with DF 2-electron derivatives
    (Analytic or FD) and Pulay overlap terms.

    Parameters
    ----------
    scf_out : Any
        SCF result object (provides DF tensors).
    casscf : Any
        CASSCF result object (mo_coeff, ci, etc.).
    fcisolver : GUGAFCISolver | None, optional
        FCI solver for RDM calculation.
    twos : int | None, optional
        Spin multiplicity (2S).
    atmlst : Sequence[int] | None, optional
        List of atoms to compute gradient for.
    df_backend : Literal["cpu", "cuda"], optional
        Backend for DF derivative contraction.
    int1e_contract_backend : Literal["auto", "cpu", "cuda"], optional
        Backend for AO 1e derivative contractions used in ``contract_dhcore_cart``.
    df_config : Any | None, optional
        Configuration for DF backend.
    df_threads : int, optional
        Number of threads for DF backend.
    delta_bohr : float, optional
        Step size for finite difference fallback (Bohr).
    solver_kwargs : dict, optional
        Arguments for FCI solver RDM calculation.
    profile : dict, optional
        Dictionary to store timing/profile data.

    Returns
    -------
    DFNucGradResult
        The computed nuclear gradient.
    """

    # When df_backend='cuda', auto-promote the 1e contraction backend so the
    # full gradient pipeline stays GPU-resident (avoids CPU↔GPU ping-pong).
    if str(int1e_contract_backend).strip().lower() == "cpu" and str(df_backend).strip().lower() == "cuda":
        int1e_contract_backend = "cuda"

    t0_total = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.clear()
        profile["df_backend"] = str(df_backend).strip().lower()
        profile["int1e_contract_backend"] = str(int1e_contract_backend).strip().lower()
        profile["df_threads"] = int(df_threads)

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])
    if natm <= 0:
        return DFNucGradResult(e_tot=float(getattr(casscf, "e_tot", 0.0)), e_nuc=float(mol.energy_nuc()), grad=np.zeros((0, 3)))
    if profile is not None:
        profile["natm"] = int(natm)

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")

    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(casscf, "ci"), nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )
    t_rdms = time.perf_counter() if profile is not None else 0.0

    xp, _is_gpu = _resolve_xp(df_backend)
    C = _as_xp_f64(xp, getattr(casscf, "mo_coeff"))
    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))

    # Detect spherical AO mode.  gfock / D / bar_L are computed in the native
    # AO basis (spherical when cart=False), then back-transformed to Cartesian
    # before the derivative-integral contractions.
    T_c2s = _get_sph_T(scf_out)

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
    )
    t_gfock = time.perf_counter() if profile is not None else 0.0

    # ── GPU phase: bar_L build + DF 2e contraction (before 1e to keep GPU busy) ──
    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=dm2_act,
    )
    t_barL = time.perf_counter() if profile is not None else 0.0

    # Energy-weighted density W (in native AO basis)
    _C_np = _asnumpy_f64(C)
    _gfock_np = _asnumpy_f64(gfock)
    _nocc = ncore + ncas
    _C_occ = _C_np[:, :_nocc]
    _tmp_w = _C_np @ _gfock_np[:, :_nocc]  # (nao, nocc)
    W = 0.5 * (_tmp_w @ _C_occ.T + _C_occ @ _tmp_w.T)

    # ── Native spherical DF gradient (no B_cart rebuild / bar_L back-transform) ──
    _T_np = None if T_c2s is None else np.asarray(T_c2s, dtype=np.float64)

    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        t0_df = time.perf_counter() if profile is not None else 0.0
        if _T_np is not None:
            de_df = compute_df_gradient_contributions_analytic_sph(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_sph=B_ao,
                bar_L_sph=bar_L_ao,
                T_c2s=_T_np,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=df_prof,
            )
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_ao,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=df_prof,
            )
    except (NotImplementedError, RuntimeError) as _analytic_exc:
        # Fallback for backends without analytic DF derivative kernels (e.g. CUDA).
        import warnings as _w
        _w.warn(
            f"Analytic DF gradient failed ({type(_analytic_exc).__name__}: {_analytic_exc}); "
            "falling back to finite-difference (much slower). "
            "Set ASUKA_GRAD_DEBUG=1 for traceback.",
            stacklevel=1,
        )
        import os as _os_grad
        if _os_grad.environ.get("ASUKA_GRAD_DEBUG"):
            import traceback as _tb
            _tb.print_exc()
        t0_df = time.perf_counter() if profile is not None else 0.0
        _bar_L_fd = _transform_bar_L_to_cart(_T_np, bar_L_ao) if _T_np is not None else bar_L_ao
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            bar_L_ao=_bar_L_fd,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(delta_bohr),
            profile=df_prof,
        )
    t_df = time.perf_counter() if profile is not None else 0.0

    # ── CPU phase: 1e AO derivative contractions ──
    # For spherical: transform D/W to Cartesian for the 1e kernels.
    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    _D_tot_1e = _T_np @ _asnumpy_f64(D_tot_ao) @ _T_np.T if _T_np is not None else _asnumpy_f64(D_tot_ao)
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=_D_tot_1e,
        shell_atom=shell_atom,
        contract_backend=str(int1e_contract_backend),
    )

    t_1e = time.perf_counter() if profile is not None else 0.0

    try:
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)
    except Exception:
        de_nuc = np.zeros((natm, 3), dtype=np.float64)
    t_nuc = time.perf_counter() if profile is not None else 0.0

    # Pulay (overlap-derivative) term: -Tr(W · dS/dR).
    _W_1e = _T_np @ W @ _T_np.T if _T_np is not None else W
    de_pulay = -1.0 * contract_dS_cart(
        ao_basis, atom_coords_bohr=coords,
        M=np.asarray(_W_1e, dtype=np.float64),
        shell_atom=shell_atom,
        contract_backend=str(int1e_contract_backend),
    )
    t_pulay = time.perf_counter() if profile is not None else 0.0

    _de_h1_np = np.asarray(de_h1, dtype=np.float64)
    _de_df_np = _asnumpy_f64(de_df)
    _de_nuc_np = np.asarray(de_nuc, dtype=np.float64)
    import os as _os
    if _os.environ.get("ASUKA_GRAD_DEBUG"):
        print(f"[grad_debug] atom0 x: de_h1={_de_h1_np[0,0]:+.8f}  de_df={_de_df_np[0,0]:+.8f}"
              f"  de_nuc={_de_nuc_np[0,0]:+.8f}  de_pulay={de_pulay[0,0]:+.8f}"
              f"  pre_pulay={(_de_h1_np+_de_df_np+_de_nuc_np)[0,0]:+.8f}"
              f"  total={(_de_h1_np+_de_df_np+_de_nuc_np+de_pulay)[0,0]:+.8f}", flush=True)

    # Full nuclear gradient: de_h1 + de_df + de_nuc + de_pulay.
    grad = np.asarray(_de_h1_np + _de_df_np + _de_nuc_np + de_pulay, dtype=np.float64)
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    if profile is not None:
        profile["t_rdms_s"] = float(t_rdms - t0_total)
        profile["t_gfock_s"] = float(t_gfock - t_rdms)
        profile["t_barL_s"] = float(t_barL - t_gfock)
        profile["t_df_s"] = float(t_df - t0_df)
        profile["t_1e_s"] = float(t_1e - t_df)
        profile["t_nuc_s"] = float(t_nuc - t_1e)
        profile["t_pulay_s"] = float(t_pulay - t_nuc)
        profile["t_total_s"] = float(t_pulay - t0_total)

    return DFNucGradResult(
        e_tot=float(getattr(casscf, "e_tot", 0.0)),
        e_nuc=float(mol.energy_nuc()),
        grad=grad,
    )


def casci_nuc_grad_df_unrelaxed(
    scf_out: Any,
    casci: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    root_weights: Sequence[float] | None = None,
    atmlst: Sequence[int] | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    solver_kwargs: dict[str, Any] | None = None,
    profile: dict | None = None,
) -> DFNucGradResult:
    """Unrelaxed DF-based nuclear gradient for CASCI (no SCF/CPHF response).

    This is primarily a debugging utility. For the SCF-relaxed CASCI
    nuclear gradient (includes RHF/CPHF response), use
    :func:`casci_nuc_grad_df_relaxed`.

    Parameters
    ----------
    scf_out : Any
        SCF result object.
    casci : Any
        CASCI result object.
    fcisolver : GUGAFCISolver | None, optional
        FCI solver.
    twos : int | None, optional
        Spin multiplicity.
    root_weights : Sequence[float] | None, optional
        State-average weights.
    atmlst : Sequence[int] | None, optional
        Atom list.
    df_backend : Literal["cpu", "cuda"], optional
        DF backend.
    df_config : Any | None, optional
        DF config.
    df_threads : int, optional
        DF threads.
    delta_bohr : float, optional
        FD step size.
    solver_kwargs : dict, optional
        Solver kwargs.
    profile : dict, optional
        Profiling dict.

    Returns
    -------
    DFNucGradResult
        The computed gradient.
    """

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])
    if natm <= 0:
        return DFNucGradResult(e_tot=float(getattr(casci, "e_tot", 0.0)), e_nuc=float(mol.energy_nuc()), grad=np.zeros((0, 3)))

    ncore = int(getattr(casci, "ncore"))
    ncas = int(getattr(casci, "ncas"))
    nelecas = getattr(casci, "nelecas")

    nroots = int(getattr(casci, "nroots", 1))
    weights_in = root_weights
    if weights_in is None:
        weights_in = getattr(casci, "root_weights", None)
    if weights_in is None:
        weights_in = getattr(casci, "weights", None)
    weights = normalize_weights(weights_in, nroots=nroots)
    ci_list = ci_as_list(getattr(casci, "ci"), nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )

    C = _asnumpy_f64(getattr(casci, "mo_coeff"))
    B_ao = _asnumpy_f64(getattr(scf_out, "df_B"))
    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
    )

    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=D_tot_ao,
        shell_atom=shell_atom,
    )

    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=dm2_act,
    )

    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        de_df = compute_df_gradient_contributions_analytic_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            B_ao=B_ao,
            bar_L_ao=bar_L_ao,
            L_chol=getattr(scf_out, "df_L", None),
            backend=str(df_backend),
            df_threads=int(df_threads),
            profile=df_prof,
        )
    except (NotImplementedError, RuntimeError):
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_ao,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(delta_bohr),
            profile=df_prof,
        )

    try:
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)
    except Exception:
        de_nuc = np.zeros((natm, 3), dtype=np.float64)

    grad = np.asarray(de_h1 + de_df + de_nuc, dtype=np.float64)
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    e_raw = np.asarray(getattr(casci, "e_tot", 0.0), dtype=np.float64).ravel()
    if int(e_raw.size) == 1:
        e_tot = float(e_raw[0])
    elif int(e_raw.size) >= int(nroots):
        e_tot = float(np.dot(np.asarray(weights, dtype=np.float64), e_raw[: int(nroots)]))
    elif int(e_raw.size):
        w = np.ones((int(e_raw.size),), dtype=np.float64) / float(int(e_raw.size))
        e_tot = float(np.dot(w, e_raw))
    else:
        e_tot = 0.0

    return DFNucGradResult(
        e_tot=float(e_tot),
        e_nuc=float(mol.energy_nuc()),
        grad=grad,
    )


def casci_nuc_grad_df_relaxed(
    scf_out: Any,
    casci: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    root_weights: Sequence[float] | None = None,
    atmlst: Sequence[int] | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    solver_kwargs: dict[str, Any] | None = None,
    cphf_max_cycle: int = 30,
    cphf_tol: float = 1e-10,
    cphf_diis_space: int = 8,
    profile: dict | None = None,
) -> DFNucGradResult:
    """Relaxed DF-based nuclear gradient for CASCI (includes RHF/DF CPHF response).

    Computes the nuclear gradient including the orbital response contribution
    via coupled-perturbed Hartree-Fock (CPHF) equations.

    Parameters
    ----------
    scf_out : Any
        SCF result object.
    casci : Any
        CASCI result object.
    fcisolver : GUGAFCISolver | None, optional
        FCI solver.
    twos : int | None, optional
        Spin multiplicity.
    root_weights : Sequence[float] | None, optional
        State-average weights.
    atmlst : Sequence[int] | None, optional
        Atom list.
    df_backend : Literal["cpu", "cuda"], optional
        DF backend.
    df_config : Any | None, optional
        DF config.
    df_threads : int, optional
        DF threads.
    delta_bohr : float, optional
        FD step size.
    solver_kwargs : dict, optional
        Solver kwargs.
    cphf_max_cycle : int, optional
        CPHF max cycles.
    cphf_tol : float, optional
        CPHF tolerance.
    cphf_diis_space : int, optional
        CPHF DIIS space size.
    profile : dict, optional
        Profiling dict.

    Returns
    -------
    DFNucGradResult
        The computed gradient.
    """

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])
    if natm <= 0:
        return DFNucGradResult(e_tot=float(getattr(casci, "e_tot", 0.0)), e_nuc=float(mol.energy_nuc()), grad=np.zeros((0, 3)))

    ncore = int(getattr(casci, "ncore"))
    ncas = int(getattr(casci, "ncas"))
    nelecas = getattr(casci, "nelecas")
    nocc = int(ncore + ncas)

    nroots = int(getattr(casci, "nroots", 1))
    weights_in = root_weights
    if weights_in is None:
        weights_in = getattr(casci, "root_weights", None)
    if weights_in is None:
        weights_in = getattr(casci, "weights", None)
    weights = normalize_weights(weights_in, nroots=nroots)
    ci_list = ci_as_list(getattr(casci, "ci"), nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )

    # Spherical AO support
    T_c2s = _get_sph_T(scf_out)
    _T_np = None if T_c2s is None else np.asarray(T_c2s, dtype=np.float64)

    C = _asnumpy_f64(getattr(casci, "mo_coeff"))
    B_ao = _asnumpy_f64(getattr(scf_out, "df_B"))
    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
    )

    # ------------------------------------------------------------------
    # RHF/CPHF response (matches pyscf.grad.casci)
    # ------------------------------------------------------------------
    scf = getattr(scf_out, "scf", None)
    if scf is None:
        raise TypeError("scf_out must have a .scf attribute for relaxed CASCI gradients")
    mo_energy = _asnumpy_f64(getattr(scf, "mo_energy"))
    mo_occ = _asnumpy_f64(getattr(scf, "mo_occ"))
    if mo_energy.ndim != 1 or mo_occ.ndim != 1:
        raise NotImplementedError("Relaxed CASCI gradients currently require RHF mo_energy/mo_occ as 1D arrays")

    nelec = float(np.sum(mo_occ).item())
    nelec_i = int(round(nelec))
    if abs(nelec - float(nelec_i)) > 1e-8 or nelec_i % 2 != 0:
        raise ValueError("RHF CPHF requires an even integer electron count")
    neleca = int(nelec_i // 2)
    if neleca <= 0:
        raise ValueError("empty occupied space")
    if neleca > int(C.shape[1]):
        raise ValueError("neleca exceeds number of MOs")

    if nocc > int(C.shape[1]):
        raise ValueError("ncore+ncas exceeds nmo")

    orbo = C[:, :neleca]
    orbv = C[:, neleca:]
    eps_occ = mo_energy[:neleca]
    eps_vir = mo_energy[neleca:]

    Imat = np.asarray(gfock, dtype=np.float64)
    ee = mo_energy[:, None] - mo_energy[None, :]
    denom = np.where(np.abs(ee) < 1e-12, np.sign(ee) * 1e-12 + (ee == 0) * 1e-12, ee)

    zvec = np.zeros_like(Imat)
    if ncore and ncore < neleca:
        zvec[:ncore, ncore:neleca] = Imat[:ncore, ncore:neleca] / (-denom[:ncore, ncore:neleca])
        zvec[ncore:neleca, :ncore] = Imat[ncore:neleca, :ncore] / (-denom[ncore:neleca, :ncore])
    if nocc > neleca:
        zvec[nocc:, neleca:nocc] = Imat[nocc:, neleca:nocc] / (-denom[nocc:, neleca:nocc])
        zvec[neleca:nocc, nocc:] = Imat[neleca:nocc, nocc:] / (-denom[neleca:nocc, nocc:])

    zvec_ao = C @ (zvec + zvec.T) @ C.T
    Jz, Kz = _df_scf._df_JK(B_ao, zvec_ao, want_J=True, want_K=True)  # noqa: SLF001
    vhf = 2.0 * (Jz - 0.5 * Kz)

    xvo = orbv.T @ vhf @ orbo
    xvo += Imat[neleca:, :neleca] - Imat[:neleca, neleca:].T

    cphf_res = solve_rhf_cphf_df(
        B_ao,
        orbo=orbo,
        orbv=orbv,
        eps_occ=eps_occ,
        eps_vir=eps_vir,
        rhs_vo=xvo,
        max_cycle=int(cphf_max_cycle),
        tol=float(cphf_tol),
        diis_space=int(cphf_diis_space),
    )
    if not bool(cphf_res.converged):
        raise RuntimeError(f"CPHF did not converge (residual {cphf_res.residual_norm:g})")
    zvec[neleca:, :neleca] = cphf_res.x_vo

    zvec_ao = C @ (zvec + zvec.T) @ C.T
    zeta = C @ (zvec * mo_energy[None, :]) @ C.T

    p1 = orbo @ orbo.T
    Jz, Kz = _df_scf._df_JK(B_ao, zvec_ao, want_J=True, want_K=True)  # noqa: SLF001
    veff_z = Jz - 0.5 * Kz
    vhf_s1occ = p1 @ veff_z @ p1

    Imat_m = np.asarray(Imat, dtype=np.float64, copy=True)
    if ncore and ncore < neleca:
        Imat_m[:ncore, ncore:neleca] = 0.0
        Imat_m[ncore:neleca, :ncore] = 0.0
    if nocc > neleca:
        Imat_m[nocc:, neleca:nocc] = 0.0
        Imat_m[neleca:nocc, nocc:] = 0.0
    Imat_m[neleca:, :neleca] = Imat_m[:neleca, neleca:].T
    im1 = C @ Imat_m @ C.T

    # ------------------------------------------------------------------
    # Nuclear gradient contractions
    # ------------------------------------------------------------------
    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    _M_h1 = np.asarray(D_tot_ao + zvec_ao, dtype=np.float64)
    _M_h1_cart = _T_np @ _M_h1 @ _T_np.T if _T_np is not None else _M_h1
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=_M_h1_cart,
        shell_atom=shell_atom,
    )

    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=dm2_act,
    )

    hf_dm1 = _df_scf._density_from_C_occ(C, mo_occ)  # noqa: SLF001
    bar_L_resp = _build_bar_L_df_cross(
        B_ao,
        D_left=np.asarray(hf_dm1, dtype=np.float64),
        D_right=np.asarray(zvec_ao, dtype=np.float64),
        # Match pyscf.grad.casci: response term uses Veff = J - 0.5 K (not 2J-K).
        coeff_J=1.0,
        coeff_K=-0.5,
    )
    bar_L_tot = np.asarray(bar_L_ao + bar_L_resp, dtype=np.float64)

    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        if _T_np is not None:
            de_df = compute_df_gradient_contributions_analytic_sph(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_sph=B_ao,
                bar_L_sph=bar_L_tot,
                T_c2s=_T_np,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=df_prof,
            )
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_tot,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=df_prof,
            )
    except (NotImplementedError, RuntimeError):
        _bar_L_tot_fd = _transform_bar_L_to_cart(_T_np, bar_L_tot) if _T_np is not None else bar_L_tot
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            bar_L_ao=_bar_L_tot_fd,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(delta_bohr),
            profile=df_prof,
        )

    try:
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)
    except Exception:
        de_nuc = np.zeros((natm, 3), dtype=np.float64)

    grad = np.asarray(de_h1 + de_df + de_nuc, dtype=np.float64)
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    e_raw = np.asarray(getattr(casci, "e_tot", 0.0), dtype=np.float64).ravel()
    if int(e_raw.size) == 1:
        e_tot = float(e_raw[0])
    elif int(e_raw.size) >= int(nroots):
        e_tot = float(np.dot(np.asarray(weights, dtype=np.float64), e_raw[: int(nroots)]))
    elif int(e_raw.size):
        w = np.ones((int(e_raw.size),), dtype=np.float64) / float(int(e_raw.size))
        e_tot = float(np.dot(w, e_raw))
    else:
        e_tot = 0.0

    return DFNucGradResult(
        e_tot=float(e_tot),
        e_nuc=float(mol.energy_nuc()),
        grad=grad,
    )


@dataclass(frozen=True)
class DFNucGradMultirootResult:
    """Container for per-root DF-based nuclear gradients from SA-CASSCF.

    Attributes
    ----------
    e_roots : np.ndarray
        Per-root energies, shape ``(nroots,)``.
    e_sa : float
        State-averaged energy.
    e_nuc : float
        Nuclear repulsion energy.
    grads : np.ndarray
        Per-root gradients, shape ``(nroots, natm, 3)`` in Eh/Bohr.
    grad_sa : np.ndarray
        State-averaged gradient, shape ``(natm, 3)`` in Eh/Bohr.
    root_weights : np.ndarray
        SA weights, shape ``(nroots,)``.
    """

    e_roots: np.ndarray
    e_sa: float
    e_nuc: float
    grads: np.ndarray
    grad_sa: np.ndarray
    root_weights: np.ndarray


def casscf_nuc_grad_df_per_root(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    int1e_contract_backend: Literal["auto", "cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    solver_kwargs: dict[str, Any] | None = None,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> DFNucGradMultirootResult:
    """Per-root DF-based nuclear gradients for SA-CASSCF with CP-MCSCF response.

    For SA-CASSCF, individual root energies are *not* variational w.r.t.
    orbitals, so the per-root gradient requires a CP-MCSCF (Z-vector) orbital
    response correction.  This function computes exact per-root analytic
    gradients using the same Lagrangian approach as PySCF's
    ``sacasscf.Gradients``.

    For single-state CASSCF (nroots=1) the response is zero and the result
    matches :func:`casscf_nuc_grad_df` exactly.

    Parameters
    ----------
    scf_out : Any
        SCF result object (provides DF tensors).
    casscf : Any
        CASSCF result object (mo_coeff, ci, etc.).
    fcisolver : GUGAFCISolver | None, optional
        FCI solver for RDM / transition-RDM calculation.
    twos : int | None, optional
        Spin quantum number 2S.
    df_backend : Literal["cpu", "cuda"], optional
        Backend for DF derivative contraction.
    int1e_contract_backend : Literal["auto", "cpu", "cuda"], optional
        Backend for AO 1e derivative contractions used in ``contract_dhcore_cart``.
    df_config : Any | None, optional
        Configuration for DF backend.
    df_threads : int, optional
        Number of threads for DF backend.
    delta_bohr : float, optional
        Step size for finite difference fallback (Bohr).
    solver_kwargs : dict, optional
        Arguments for FCI solver RDM calculation.
    z_tol : float, optional
        Convergence tolerance for the Z-vector solve.
    z_maxiter : int, optional
        Maximum iterations for the Z-vector solve.

    Returns
    -------
    DFNucGradMultirootResult
        Per-root energies and gradients.
    """
    from contextlib import contextmanager, nullcontext  # noqa: PLC0415
    import os  # noqa: PLC0415

    from .newton_df import DFNewtonCASSCFAdapter  # noqa: PLC0415
    from . import newton_casscf as _newton_casscf  # noqa: PLC0415
    from .zvector import build_mcscf_hessian_operator, solve_mcscf_zvector  # noqa: PLC0415
    from .nac._df import (  # noqa: PLC0415
        _FixedRDMFcisolver,
        _grad_elec_active_df,
        _Lorb_dot_dgorb_dx_df,
        _build_bar_L_net_active_df,
        _build_bar_L_lorb_df,
    )

    @contextmanager
    def _force_internal_newton():
        k_prefer = "CUGUGA_NEWTON_CASSCF"
        k_impl = "CUGUGA_NEWTON_CASSCF_IMPL"
        old_prefer = os.environ.get(k_prefer)
        old_impl = os.environ.get(k_impl)
        os.environ[k_prefer] = "internal"
        os.environ[k_impl] = "internal"
        try:
            yield
        finally:
            if old_prefer is None:
                os.environ.pop(k_prefer, None)
            else:
                os.environ[k_prefer] = old_prefer
            if old_impl is None:
                os.environ.pop(k_impl, None)
            else:
                os.environ[k_impl] = old_impl

    # ------------------------------------------------------------------
    # Setup (shared across all roots)
    # ------------------------------------------------------------------
    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(casscf, "ci"), nroots=nroots)
    e_roots = np.asarray(getattr(casscf, "e_roots"), dtype=np.float64).ravel()

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    # Spherical AO support: detect cart=False and prepare T_c2s.
    T_c2s = _get_sph_T(scf_out)
    _T_np = None if T_c2s is None else np.asarray(T_c2s, dtype=np.float64)

    xp, _is_gpu = _resolve_xp(df_backend)
    C = _as_xp_f64(xp, getattr(casscf, "mo_coeff"))
    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))

    # Per-root RDMs
    per_root_rdms: list[tuple[np.ndarray, np.ndarray]] = []
    for K in range(nroots):
        dm1_K, dm2_K = fcisolver_use.make_rdm12(ci_list[K], int(ncas), nelecas, **(solver_kwargs or {}))
        per_root_rdms.append((np.asarray(dm1_K, dtype=np.float64), np.asarray(dm2_K, dtype=np.float64)))

    # SA RDMs (needed for orbital response)
    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver_use, ci_list, weights, ncas=int(ncas), nelecas=nelecas, solver_kwargs=solver_kwargs,
    )

    # Nuclear repulsion gradient (shared)
    try:
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)
    except Exception:
        de_nuc = np.zeros((natm, 3), dtype=np.float64)

    # Build SA adapter and Hessian operator (shared across all roots)
    mc_sa = DFNewtonCASSCFAdapter(
        df_B=B_ao,
        hcore_ao=h_ao,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=C,
        fcisolver=fcisolver_use,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
        frozen=getattr(casscf, "frozen", None),
        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
        extrasym=getattr(casscf, "extrasym", None),
    )
    eris_sa = mc_sa.ao2mo(C)
    with _force_internal_newton():
        hess_op = build_mcscf_hessian_operator(
            mc_sa, mo_coeff=C, ci=ci_list, eris=eris_sa, use_newton_hessian=True,
        )

    # Prepare DF gradient contraction context (reuse across roots)
    from asuka.integrals.df_grad_context import DFGradContractionContext  # noqa: PLC0415

    df_grad_ctx: DFGradContractionContext | None = None
    try:
        df_grad_ctx = DFGradContractionContext.build(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            backend=str(df_backend),
            df_threads=int(df_threads),
            L_chol=getattr(scf_out, "df_L", None),
        )
    except (NotImplementedError, RuntimeError):
        df_grad_ctx = None

    # AO basis objects for 1e derivative contractions
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    n_orb = int(hess_op.n_orb)
    w_arr = np.asarray(weights, dtype=np.float64).ravel()
    warned_df_fd = False

    # ------------------------------------------------------------------
    # Profiling + CUDA stream configuration
    # ------------------------------------------------------------------
    _prof_env = str(os.environ.get("ASUKA_PROFILE_DF_PER_ROOT", "")).strip().lower()
    _profile_df_per_root = _prof_env not in ("", "0", "false", "no", "off")
    _t_bar_L_sa = 0.0
    _t_contract_sa = 0.0
    _t_bar_L_delta = 0.0
    _t_contract_delta = 0.0
    _t_z_solve = 0.0
    _t_trans_rdm_lci = 0.0
    _z_solver: str | None = None
    _z_matvec_calls = 0
    _z_niter = 0

    # Multi-stream DF contraction (CUDA only). Escape hatch: ASUKA_DF_CONTRACT_STREAMS=1.
    _use_multistream_contract = False
    _n_streams = 1
    _cp = None  # type: ignore[assignment]
    _main_stream = None  # type: ignore[assignment]
    _main_stream_cm = nullcontext()
    _contract_streams: list[Any] = []

    if (
        str(df_backend).strip().lower() == "cuda"
        and df_grad_ctx is not None
        and str(getattr(df_grad_ctx, "backend", "")).strip().lower() == "cuda"
        and int(nroots) > 1
    ):
        try:
            _n_streams_env = int(os.environ.get("ASUKA_DF_CONTRACT_STREAMS", "4"))
        except Exception:
            _n_streams_env = 4
        _n_streams = max(1, min(int(nroots), int(_n_streams_env)))
        # Auto-scale: if sizeof(B) * n_streams > 50% free VRAM, reduce streams.
        try:
            import cupy as _cp_probe  # noqa: PLC0415

            _sizeof_B = int(B_ao.size) * 8  # float64
            _free_vram = int(_cp_probe.cuda.runtime.memGetInfo()[0])
            _vram_cap = max(1, int(0.5 * _free_vram // max(1, _sizeof_B)))
            _n_streams = min(_n_streams, max(1, _vram_cap))
        except Exception:
            pass
        if _n_streams > 1:
            try:
                import cupy as cp  # noqa: PLC0415
            except Exception:
                cp = None  # type: ignore
            if cp is not None:
                _cp = cp
                _main_stream = cp.cuda.Stream()
                _main_stream_cm = _main_stream
                _contract_streams = [cp.cuda.Stream() for _ in range(int(_n_streams))]
                _use_multistream_contract = True

    # ------------------------------------------------------------------
    # Z-vector + CI-response CUDA configuration
    # ------------------------------------------------------------------
    _z_method_env = str(os.environ.get("ASUKA_ZVECTOR_METHOD", "auto")).strip().lower()
    _z_use_x0_env = str(os.environ.get("ASUKA_ZVECTOR_USE_X0", "1")).strip().lower()
    _z_use_x0 = _z_use_x0_env not in ("0", "false", "no", "off", "disable", "disabled")

    # Default to GCROTMK unless explicitly overridden.
    if _z_method_env in ("gmres", "gcrotmk"):
        _z_method = _z_method_env
    else:
        _z_method = "gcrotmk"
    _z_recycle_space: list[tuple[np.ndarray | None, np.ndarray]] | None = [] if _z_method == "gcrotmk" else None
    _z_prev_x0: np.ndarray | None = None

    # Per-root CI-response transition RDM accumulation: optionally keep on GPU.
    _ci_rdm_device_env = str(os.environ.get("ASUKA_PER_ROOT_CI_RDM_DEVICE", "auto")).strip().lower()
    try:
        _ci_rdm_thresh = int(os.environ.get("ASUKA_PER_ROOT_CI_RDM_CUDA_THRESHOLD_NCSF", "4096"))
    except Exception:
        _ci_rdm_thresh = 4096

    _use_ci_rdm_device = False
    _cp_ci = None  # type: ignore[assignment]
    ci_list_dev: list[Any] | None = None

    if str(df_backend).strip().lower() == "cuda":
        if _ci_rdm_device_env in ("1", "true", "yes", "on", "enable", "enabled"):
            _use_ci_rdm_device = True
        elif _ci_rdm_device_env in ("0", "false", "no", "off", "disable", "disabled"):
            _use_ci_rdm_device = False
        else:
            ncsf_hint = int(getattr(hess_op, "n_ci", 0)) // max(int(nroots), 1)
            _use_ci_rdm_device = int(ncsf_hint) >= int(_ci_rdm_thresh)

        if _use_ci_rdm_device:
            try:
                from asuka.cuda.cuda_backend import has_cuda_ext  # noqa: PLC0415

                if not has_cuda_ext():
                    _use_ci_rdm_device = False
            except Exception:
                _use_ci_rdm_device = False

        if _use_ci_rdm_device:
            try:
                import cupy as cp  # type: ignore[import-not-found]  # noqa: PLC0415
            except Exception:
                _use_ci_rdm_device = False
            else:
                _cp_ci = cp
                # Pre-upload ket CI vectors once. Keep on the same "main" stream used elsewhere.
                with _main_stream_cm:
                    ci_list_dev = [cp.asarray(ci, dtype=cp.float64).ravel() for ci in ci_list]

    # ------------------------------------------------------------------
    # Precompute shared quantities for lightweight Z-vector RHS
    # ------------------------------------------------------------------
    nmo = int(C.shape[1])
    nocc = ncore + ncas
    ppaa_sa = _asnumpy_f64(getattr(eris_sa, "ppaa"))  # (nmo,nmo,ncas,ncas)
    papa_sa = _asnumpy_f64(getattr(eris_sa, "papa"))  # (nmo,ncas,nmo,ncas)
    vhf_c_sa = _asnumpy_f64(getattr(eris_sa, "vhf_c"))  # (nmo,nmo)
    h1e_mo_sa = _asnumpy_f64(C).T @ _asnumpy_f64(h_ao) @ _asnumpy_f64(C)
    h1cas_0 = h1e_mo_sa[ncore:nocc, ncore:nocc] + vhf_c_sa[ncore:nocc, ncore:nocc]
    eri_cas_sa = ppaa_sa[ncore:nocc, ncore:nocc]

    # Precompute hci0, eci0 (root-independent: H_act |ci_r> for each root)
    _linkstrl_rhs = _newton_casscf._maybe_gen_linkstr(fcisolver_use, ncas, nelecas, True)
    _hci0_all = _newton_casscf._ci_h_op(
        fcisolver_use,
        h1cas=h1cas_0,
        eri_cas=eri_cas_sa,
        ncas=ncas,
        nelecas=nelecas,
        ci_list=ci_list,
        link_index=_linkstrl_rhs,
    )
    _eci0_all = np.array(
        [float(np.dot(np.asarray(c).ravel(), np.asarray(hc).ravel())) for c, hc in zip(ci_list, _hci0_all)],
        dtype=np.float64,
    )
    # Pre-reshape ppaa for GEMM in orbital gradient
    _ppaa_2d = ppaa_sa.reshape(nmo * nmo, ncas * ncas)

    # ------------------------------------------------------------------
    # SA base gradient (shared across all roots, includes Pulay term)
    # This is the correct SA-CASSCF gradient (variational w.r.t. SA energy).
    # Per-root gradients are expressed as: grad_K = grad_sa_base + delta_K
    # ------------------------------------------------------------------
    with _main_stream_cm:
        C_np = _asnumpy_f64(C)
        gfock_sa, D_core_sa, D_act_sa, D_tot_sa, C_act_sa = _build_gfock_casscf_df(
            B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas), dm1_act=dm1_sa, dm2_act=dm2_sa,
        )

        # Precompute root-invariant DF intermediates for active contractions.
        nao = int(B_ao.shape[0])
        naux = int(B_ao.shape[2])
        B2 = B_ao.reshape(nao * nao, naux)
        rho_core = B2.T @ D_core_sa.reshape(nao * nao)  # (naux,)

        X_act = xp.einsum("mnQ,nv->mvQ", B_ao, C_act_sa, optimize=True)
        L_act = xp.einsum("mu,mvQ->uvQ", C_act_sa, X_act, optimize=True)  # (ncas,ncas,naux)
        L2 = L_act.reshape(int(ncas) * int(ncas), naux)  # (ncas^2,naux)
        del X_act

        t0 = time.perf_counter()
        bar_L_sa = _build_bar_L_casscf_df(
            B_ao,
            D_core_ao=D_core_sa,
            D_act_ao=D_act_sa,
            C_act=C_act_sa,
            dm2_act=dm2_sa,
            L_act=L_act,
            rho_core=rho_core,
        )
        if _profile_df_per_root:
            _t_bar_L_sa += time.perf_counter() - t0

        try:
            t0 = time.perf_counter()
            if _T_np is not None:
                # ── Spherical path: compute DF adjoint in sph, transform bar_X to cart ──
                if df_grad_ctx is not None:
                    de_df_sa = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_sa, T_c2s=_T_np)
                else:
                    de_df_sa = compute_df_gradient_contributions_analytic_sph(
                        ao_basis, aux_basis, atom_coords_bohr=coords, B_sph=B_ao, bar_L_sph=bar_L_sa,
                        T_c2s=_T_np, L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend), df_threads=int(df_threads), profile=None,
                    )
            else:
                if df_grad_ctx is not None:
                    de_df_sa = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_sa)
                else:
                    de_df_sa = compute_df_gradient_contributions_analytic_packed_bases(
                        ao_basis, aux_basis, atom_coords_bohr=coords, B_ao=B_ao, bar_L_ao=bar_L_sa,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend), df_threads=int(df_threads), profile=None,
                    )
            if _profile_df_per_root:
                _t_contract_sa += time.perf_counter() - t0
        except (NotImplementedError, RuntimeError) as e:
            warnings.warn(
                f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                "expect noisy/non-conservative forces in MD. "
                f"Reason: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            warned_df_fd = True
            _bar_L_sa_fd = _transform_bar_L_to_cart(_T_np, bar_L_sa) if _T_np is not None else bar_L_sa
            de_df_sa = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=_bar_L_sa_fd,
                backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                delta_bohr=float(delta_bohr), profile=None,
            )

    # ── 1e and Pulay: transform D/W to Cartesian if spherical ──
    _D_tot_sa_1e = _T_np @ _asnumpy_f64(D_tot_sa) @ _T_np.T if _T_np is not None else _asnumpy_f64(D_tot_sa)
    de_h1_sa = contract_dhcore_cart(
        ao_basis, atom_coords_bohr=coords, atom_charges=charges,
        M=_D_tot_sa_1e, shell_atom=shell_atom, contract_backend=str(int1e_contract_backend),
    )
    # SA Pulay term: -Tr(W_sa · dS/dR).
    gfock_sa_np = _asnumpy_f64(gfock_sa)
    C_occ_np = C_np[:, :nocc]
    _tmp_sa = C_np @ gfock_sa_np[:, :nocc]  # (nao, nocc)
    W_sa = 0.5 * (_tmp_sa @ C_occ_np.T + C_occ_np @ _tmp_sa.T)
    _W_sa_1e = _T_np @ W_sa @ _T_np.T if _T_np is not None else W_sa
    de_pulay_sa = -2.0 * contract_dS_ip_cart(
        ao_basis,
        atom_coords_bohr=coords,
        M=_W_sa_1e,
        shell_atom=shell_atom,
        contract_backend=str(int1e_contract_backend),
    )

    # SA base gradient: de_h1_sa + de_df_sa + de_nuc + de_pulay_sa.
    grad_sa_base = np.asarray(
        de_h1_sa + _asnumpy_f64(de_df_sa) + de_nuc + de_pulay_sa, dtype=np.float64,
    )
    del bar_L_sa, de_df_sa
    _log_vram("after SA base gradient")
    _flush_gpu_pool()

    # ------------------------------------------------------------------
    # Per-root gradient loop
    # ------------------------------------------------------------------
    grads_out: list[np.ndarray | None] = [None] * int(nroots)
    _in_flight: list[dict[str, Any]] = []

    # Precompute gfock with zero active density (for CI response Pulay subtraction).
    _dm1_zero = xp.zeros((int(ncas), int(ncas)), dtype=xp.float64)
    _dm2_zero = xp.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=xp.float64)
    _gfock_zero, _, _, _, _ = _build_gfock_casscf_df(
        B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
        dm1_act=_dm1_zero, dm2_act=_dm2_zero,
    )
    _gfock_zero_np = _asnumpy_f64(_gfock_zero)

    for K in range(nroots):
        _log_vram(f"root {K} start")
        dm1_K, dm2_K = per_root_rdms[K]

        # For single-state CASSCF: no response — SA base gradient IS the exact per-root gradient.
        if nroots == 1:
            grads_out[int(K)] = np.asarray(grad_sa_base, dtype=np.float64)
            continue

        # Step A: Build per-root densities and bar_L delta for the Hellmann–Feynman term (GPU).
        with _main_stream_cm:
            gfock_K, _D_core_K, _D_act_K, D_tot_K, _C_act_K = _build_gfock_casscf_df(
                B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas), dm1_act=dm1_K, dm2_act=dm2_K,
            )
            dm1_delta = dm1_K - dm1_sa
            dm2_delta = dm2_K - dm2_sa
            t0 = time.perf_counter()
            bar_L_delta_hf = _build_bar_L_delta_casscf_df(
                B_ao,
                D_core_ao=D_core_sa,
                C_act=C_act_sa,
                dm1_delta=dm1_delta,
                dm2_delta=dm2_delta,
                rho_core=rho_core,
                L2=L2,
            )
            if _profile_df_per_root:
                _t_bar_L_delta += time.perf_counter() - t0
            _flush_gpu_pool()

        # Step B: Z-vector RHS
        fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=dm1_K, dm2=dm2_K)
        mc_K = DFNewtonCASSCFAdapter(
            df_B=B_ao,
            hcore_ao=h_ao,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            mo_coeff=C,
            fcisolver=fcisolver_fixed,
            frozen=getattr(casscf, "frozen", None),
            internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
            extrasym=getattr(casscf, "extrasym", None),
        )

        with _main_stream_cm:
            with _force_internal_newton():
                g_K, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                    mc_K, C, ci_list[K], eris_sa, verbose=0, implementation="internal",
                )
            g_K = _asnumpy_f64(g_K).ravel()
        del _gupd, _hop, _hdiag, mc_K, fcisolver_fixed
        _log_vram(f"root {K} after gen_g_hop cleanup")

        rhs_orb = g_K[:n_orb]
        rhs_ci_K = g_K[n_orb:]

        rhs_ci: list[np.ndarray] = []
        for r in range(nroots):
            rhs_ci.append(np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()))
        ndet_K = int(np.asarray(ci_list[K]).size)
        rhs_ci[K] = rhs_ci_K[:ndet_K]

        # Step C: Z-vector solve
        t0 = time.perf_counter()
        _x0_use = _z_prev_x0 if (_z_use_x0 and _z_prev_x0 is not None) else None
        z_K = solve_mcscf_zvector(
            mc_sa,
            rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
            rhs_ci=rhs_ci,
            hessian_op=hess_op,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            method=str(_z_method),
            x0=_x0_use,
            recycle_space=_z_recycle_space,
        )
        if _profile_df_per_root:
            _t_z_solve += time.perf_counter() - t0
            try:
                _z_solver = str(z_K.info.get("solver", _z_solver or "")).strip() or _z_solver
                _z_matvec_calls += int(z_K.info.get("matvec_calls", 0) or 0)
                _z_niter += int(z_K.info.get("niter", 0) or 0)
            except Exception:
                pass
        Lvec = np.asarray(z_K.z_packed, dtype=np.float64).ravel()
        if _z_use_x0:
            _z_prev_x0 = Lvec.copy()
        Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])

        # Step D: CI response — accumulate transition RDMs, then build bar_L without contract().
        Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
        del z_K
        _log_vram(f"root {K} after z_solve cleanup")
        t0 = time.perf_counter()
        if _use_ci_rdm_device and _cp_ci is not None and ci_list_dev is not None:
            cp = _cp_ci
            with _main_stream_cm:
                dm1_lci = cp.zeros((int(ncas), int(ncas)), dtype=cp.float64)
                dm2_lci = cp.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=cp.float64)
                _rdm_kw = dict(solver_kwargs or {})
                _rdm_kw["rdm_backend"] = "cuda"
                _rdm_kw["return_cupy"] = True
                for r in range(nroots):
                    wr = float(w_arr[r])
                    if abs(wr) < 1e-14:
                        continue
                    Lci_r_dev = cp.asarray(Lci_list[r], dtype=cp.float64).ravel()
                    dm1_r, dm2_r = fcisolver_use.trans_rdm12(
                        Lci_r_dev,
                        ci_list_dev[r],
                        int(ncas),
                        nelecas,
                        **_rdm_kw,
                    )
                    dm1_lci += wr * (dm1_r + dm1_r.T)
                    dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))
        else:
            # CPU accumulation path (default) — keep legacy behavior.
            _use_cuda_rdm_compute = False
            if bool(getattr(hess_op, "gpu_mode", False)):
                try:
                    from asuka.cuda.cuda_backend import has_cuda_ext  # noqa: PLC0415

                    _use_cuda_rdm_compute = bool(has_cuda_ext())
                except Exception:
                    _use_cuda_rdm_compute = False

            dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
            dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
            for r in range(nroots):
                wr = float(w_arr[r])
                if abs(wr) < 1e-14:
                    continue
                _rdm_kw = dict(solver_kwargs or {})
                if _use_cuda_rdm_compute:
                    _rdm_kw["rdm_backend"] = "cuda"
                dm1_r, dm2_r = fcisolver_use.trans_rdm12(
                    np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                    np.asarray(ci_list[r], dtype=np.float64).ravel(),
                    int(ncas),
                    nelecas,
                    **_rdm_kw,
                )
                dm1_r = np.asarray(dm1_r, dtype=np.float64)
                dm2_r = np.asarray(dm2_r, dtype=np.float64)
                dm1_lci += wr * (dm1_r + dm1_r.T)
                dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))
        if _profile_df_per_root:
            _t_trans_rdm_lci += time.perf_counter() - t0

        # ── Single fused contract(): HF delta + lci response + lorb response ──
        # Interleave build + fuse: build each bar_L component and immediately
        # accumulate into bar_L_delta_total before building the next, so at most
        # 2 (naux,nao,nao) arrays are alive at once instead of 3.
        _flush_gpu_pool()
        _log_vram(f"root {K} pre-barL flush")
        if _os.environ.get("ASUKA_VRAM_DUMP") and K == 0:
            import gc as _gc  # noqa: PLC0415
            _gc.collect()
            _cupy_arrays = []
            for _obj in _gc.get_objects():
                try:
                    if hasattr(_obj, '__class__') and _obj.__class__.__module__ == 'cupy' and hasattr(_obj, 'nbytes'):
                        _mb = _obj.nbytes / 1e6
                        if _mb >= 0.5:
                            _cupy_arrays.append((_mb, _obj.shape, _obj.dtype))
                except Exception:
                    pass
            _cupy_arrays.sort(reverse=True)
            print(f"[VRAM_DUMP] root {K} pre-barL: {len(_cupy_arrays)} arrays >= 0.5 MB")
            for _mb, _sh, _dt in _cupy_arrays[:30]:
                print(f"  {_mb:10.2f} MB  shape={_sh}  dtype={_dt}")
            del _cupy_arrays
        with _main_stream_cm:
            bar_L_delta_total = bar_L_delta_hf
            _log_vram(f"root {K} before lci_net")

            # Build bar_L_lci_net = bar_L_lci_active - bar_L_lci_core (fused, GPU).
            bar_L_lci_net, D_act_lci = _build_bar_L_net_active_df(
                B_ao, C, dm1_lci, dm2_lci, ncore=int(ncore), ncas=int(ncas), xp=xp, L_act=L_act, rho_core=rho_core,
            )
            _log_vram(f"root {K} after lci_net (before accum)")
            bar_L_delta_total += bar_L_lci_net
            del bar_L_lci_net
            _log_vram(f"root {K} after lci_net accum")
            _flush_gpu_pool()
            _log_vram(f"root {K} after lci_net flush")

            # Step E: Orbital response — build bar_L_lorb without contract() (GPU).
            bar_L_lorb, D_L_lorb = _build_bar_L_lorb_df(
                B_ao, C, np.asarray(Lorb_mat, dtype=np.float64), dm1_sa, dm2_sa,
                ncore=int(ncore), ncas=int(ncas), xp=xp,
            )
            _log_vram(f"root {K} after lorb (before accum)")
            bar_L_delta_total += bar_L_lorb
            del bar_L_lorb
            _log_vram(f"root {K} bar_L fused")

        # Multi-stream CUDA path: launch contract_device() asynchronously and
        # overlap CPU work while kernels run on independent streams.
        if _use_multistream_contract and df_grad_ctx is not None and _cp is not None and hasattr(df_grad_ctx, "contract_device"):
            cp = _cp
            evt_ready = cp.cuda.Event()
            with _main_stream_cm:
                evt_ready.record()
            stream = _contract_streams[int(K) % int(_n_streams)]
            stream.wait_event(evt_ready)
            try:
                with stream:
                    if _T_np is not None:
                        grad_dev = df_grad_ctx.contract_device_sph(B_sph=B_ao, bar_L_sph=bar_L_delta_total, T_c2s=_T_np)
                    else:
                        grad_dev = df_grad_ctx.contract_device(B_ao=B_ao, bar_L_ao=bar_L_delta_total)
            except (NotImplementedError, RuntimeError, ValueError):
                grad_dev = None

            if grad_dev is not None:
                # ── Single fused 1e call: delta from SA + lci + lorb ──
                with _main_stream_cm:
                    D_1e_delta = _asnumpy_f64(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb)
                _D_1e_delta_ms = _T_np @ D_1e_delta @ _T_np.T if _T_np is not None else D_1e_delta
                de_h1_delta = contract_dhcore_cart(
                    ao_basis, atom_coords_bohr=coords, atom_charges=charges,
                    M=_D_1e_delta_ms, shell_atom=shell_atom, contract_backend=str(int1e_contract_backend),
                )

                # ── Per-root Pulay delta: -Tr((W_K - W_sa) · dS/dR) ──
                with _main_stream_cm:
                    gfock_K_np = _asnumpy_f64(gfock_K)
                _tmp_K = C_np @ gfock_K_np[:, :nocc]  # (nao, nocc)
                W_K = 0.5 * (_tmp_K @ C_occ_np.T + C_occ_np @ _tmp_K.T)

                # ── Response Pulay (multi-stream path) ──
                _gfock_lci_raw, _, _, _, _ = _build_gfock_casscf_df(
                    B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
                    dm1_act=xp.asarray(dm1_lci, dtype=xp.float64),
                    dm2_act=xp.asarray(dm2_lci, dtype=xp.float64),
                )
                _dgfock_lci = _asnumpy_f64(_gfock_lci_raw) - _gfock_zero_np
                _dme0_lci = C_np @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C_np.T
                _dme0_lorb = _build_dme0_lorb_response(
                    B_ao, h_ao, C_np, np.asarray(Lorb_mat, dtype=np.float64),
                    _asnumpy_f64(dm1_sa), _asnumpy_f64(dm2_sa),
                    ppaa_sa, papa_sa,
                    ncore=int(ncore), ncas=int(ncas),
                )

                _W_pulay_ms = W_K - W_sa + _dme0_lci + _dme0_lorb
                _W_pulay_ms = _T_np @ _W_pulay_ms @ _T_np.T if _T_np is not None else _W_pulay_ms
                de_pulay_delta = -2.0 * contract_dS_ip_cart(
                    ao_basis,
                    atom_coords_bohr=coords,
                    M=_W_pulay_ms,
                    shell_atom=shell_atom,
                    contract_backend=str(int1e_contract_backend),
                )

                _in_flight.append(
                    {
                        "K": int(K),
                        "stream": stream,
                        "grad_dev": grad_dev,
                        "bar_L": bar_L_delta_total,
                        "de_h1_delta": np.asarray(de_h1_delta, dtype=np.float64),
                        "de_pulay_delta": np.asarray(de_pulay_delta, dtype=np.float64),
                    }
                )

                if len(_in_flight) >= int(_n_streams):
                    job = _in_flight.pop(0)
                    t0 = time.perf_counter()
                    try:
                        job["stream"].synchronize()
                        try:
                            de_df_delta = job["grad_dev"].get(stream=job["stream"])
                        except (TypeError, AttributeError):
                            de_df_delta = cp.asnumpy(job["grad_dev"])
                        de_df_delta = np.asarray(de_df_delta, dtype=np.float64)
                    except Exception as e:
                        if not warned_df_fd:
                            warnings.warn(
                                f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                                "expect noisy/non-conservative forces in MD. "
                                f"Reason: {type(e).__name__}: {e}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            warned_df_fd = True
                        _bar_L_fd_ms = _transform_bar_L_to_cart(_T_np, job["bar_L"]) if _T_np is not None else job["bar_L"]
                        de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                            ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=_bar_L_fd_ms,
                            backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                            delta_bohr=float(delta_bohr), profile=None,
                        )
                        de_df_delta = _asnumpy_f64(de_df_delta)

                    if _profile_df_per_root:
                        _t_contract_delta += time.perf_counter() - t0

                    grad_done = np.asarray(
                        grad_sa_base + de_df_delta + job["de_h1_delta"] + job["de_pulay_delta"],
                        dtype=np.float64,
                    )
                    grads_out[int(job["K"])] = grad_done
                    _flush_gpu_pool()
                continue

        # Synchronous contraction (CPU / single-stream CUDA fallback)
        try:
            t0 = time.perf_counter()
            if _T_np is not None:
                # ── Spherical path ──
                if df_grad_ctx is not None:
                    de_df_delta = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_delta_total, T_c2s=_T_np)
                else:
                    de_df_delta = compute_df_gradient_contributions_analytic_sph(
                        ao_basis, aux_basis, atom_coords_bohr=coords, B_sph=B_ao, bar_L_sph=bar_L_delta_total,
                        T_c2s=_T_np, L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend), df_threads=int(df_threads), profile=None,
                    )
            else:
                if df_grad_ctx is not None:
                    de_df_delta = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_delta_total)
                else:
                    de_df_delta = compute_df_gradient_contributions_analytic_packed_bases(
                        ao_basis, aux_basis, atom_coords_bohr=coords, B_ao=B_ao, bar_L_ao=bar_L_delta_total,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend), df_threads=int(df_threads), profile=None,
                    )
            if _profile_df_per_root:
                _t_contract_delta += time.perf_counter() - t0
        except (NotImplementedError, RuntimeError) as e:
            if not warned_df_fd:
                warnings.warn(
                    f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                    "expect noisy/non-conservative forces in MD. "
                    f"Reason: {type(e).__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_df_fd = True
            _bar_L_dt_fd = _transform_bar_L_to_cart(_T_np, bar_L_delta_total) if _T_np is not None else bar_L_delta_total
            de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=_bar_L_dt_fd,
                backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                delta_bohr=float(delta_bohr), profile=None,
            )

        # ── Single fused 1e call: delta from SA + lci + lorb ──
        with _main_stream_cm:
            D_1e_delta = _asnumpy_f64(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb)
        _D_1e_delta_1e = _T_np @ D_1e_delta @ _T_np.T if _T_np is not None else D_1e_delta
        de_h1_delta = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=_D_1e_delta_1e, shell_atom=shell_atom, contract_backend=str(int1e_contract_backend),
        )

        # ── Per-root Pulay delta: -Tr((W_K - W_sa) · dS/dR) ──
        with _main_stream_cm:
            gfock_K_np = _asnumpy_f64(gfock_K)
        _tmp_K = C_np @ gfock_K_np[:, :nocc]  # (nao, nocc)
        W_K = 0.5 * (_tmp_K @ C_occ_np.T + C_occ_np @ _tmp_K.T)

        # ── Response Pulay: energy-weighted density from CI + orbital response ──
        # CI response: perturbation gfock from response RDMs (MO basis).
        _gfock_lci_raw, _, _, _, _ = _build_gfock_casscf_df(
            B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
            dm1_act=xp.asarray(dm1_lci, dtype=xp.float64),
            dm2_act=xp.asarray(dm2_lci, dtype=xp.float64),
        )
        _dgfock_lci = _asnumpy_f64(_gfock_lci_raw) - _gfock_zero_np
        _dme0_lci = C_np @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C_np.T

        # Orbital response: L-effective gfock (AO-like basis → dme0).
        _dme0_lorb = _build_dme0_lorb_response(
            B_ao, h_ao, C_np, np.asarray(Lorb_mat, dtype=np.float64),
            _asnumpy_f64(dm1_sa), _asnumpy_f64(dm2_sa),
            ppaa_sa, papa_sa,
            ncore=int(ncore), ncas=int(ncas),
        )

        _W_delta = W_K - W_sa + _dme0_lci + _dme0_lorb
        _W_delta_1e = _T_np @ _W_delta @ _T_np.T if _T_np is not None else _W_delta
        de_pulay_delta = -2.0 * contract_dS_ip_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=_W_delta_1e,
            shell_atom=shell_atom,
            contract_backend=str(int1e_contract_backend),
        )

        # Step F: Combine — grad_sa_base + delta 2e + delta 1e + delta Pulay
        del bar_L_delta_total
        grad_K = np.asarray(
            grad_sa_base + _asnumpy_f64(de_df_delta) + de_h1_delta + de_pulay_delta,
            dtype=np.float64,
        )
        grads_out[int(K)] = grad_K
        _log_vram(f"root {K} done")
        _flush_gpu_pool()

    # Drain any remaining async contractions.
    if _use_multistream_contract and df_grad_ctx is not None and _cp is not None:
        cp = _cp
        while _in_flight:
            job = _in_flight.pop(0)
            t0 = time.perf_counter()
            try:
                job["stream"].synchronize()
                try:
                    de_df_delta = job["grad_dev"].get(stream=job["stream"])
                except (TypeError, AttributeError):
                    de_df_delta = cp.asnumpy(job["grad_dev"])
                de_df_delta = np.asarray(de_df_delta, dtype=np.float64)
            except Exception as e:
                if not warned_df_fd:
                    warnings.warn(
                        f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                        "expect noisy/non-conservative forces in MD. "
                        f"Reason: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_df_fd = True
                _bar_L_fd_drain = _transform_bar_L_to_cart(_T_np, job["bar_L"]) if _T_np is not None else job["bar_L"]
                de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                    ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=_bar_L_fd_drain,
                    backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                    delta_bohr=float(delta_bohr), profile=None,
                )
                de_df_delta = _asnumpy_f64(de_df_delta)

            if _profile_df_per_root:
                _t_contract_delta += time.perf_counter() - t0

            grad_done = np.asarray(
                grad_sa_base + de_df_delta + job["de_h1_delta"] + job["de_pulay_delta"],
                dtype=np.float64,
            )
            grads_out[int(job["K"])] = grad_done
            _flush_gpu_pool()

    missing = [i for i, g in enumerate(grads_out) if g is None]
    if missing:  # pragma: no cover
        raise RuntimeError(f"internal error: missing per-root gradients for roots {missing!r}")

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    grads = np.stack([np.asarray(g, dtype=np.float64) for g in grads_out], axis=0)  # (nroots, natm, 3)
    grad_sa = np.asarray(grad_sa_base, dtype=np.float64)  # directly from SA (not weighted sum)

    e_sa = float(np.dot(w_arr, e_roots))
    e_nuc_val = float(mol.energy_nuc())

    if _profile_df_per_root:
        import sys  # noqa: PLC0415

        nstreams = int(_n_streams) if bool(_use_multistream_contract) else 1
        print(
            "[ASUKA_PROFILE_DF_PER_ROOT] "
            f"nroots={int(nroots)} streams={nstreams} "
            f"bar_L_sa={_t_bar_L_sa:.3f}s contract_sa={_t_contract_sa:.3f}s "
            f"bar_L_delta={_t_bar_L_delta:.3f}s contract_delta={_t_contract_delta:.3f}s "
            f"z_solve_total={_t_z_solve:.3f}s z_solver={_z_solver or 'unknown'} "
            f"z_niter={int(_z_niter)} z_matvec_calls={int(_z_matvec_calls)} "
            f"t_trans_rdm_lci={_t_trans_rdm_lci:.3f}s",
            file=sys.stderr,
        )

    return DFNucGradMultirootResult(
        e_roots=e_roots,
        e_sa=e_sa,
        e_nuc=e_nuc_val,
        grads=grads,
        grad_sa=grad_sa,
        root_weights=np.asarray(weights, dtype=np.float64),
    )


__all__ = [
    "DFNucGradResult",
    "DFNucGradMultirootResult",
    "casscf_nuc_grad_df",
    "casscf_nuc_grad_df_per_root",
    "casci_nuc_grad_df_unrelaxed",
    "casci_nuc_grad_df_relaxed",
]
