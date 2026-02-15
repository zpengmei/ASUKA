from __future__ import annotations

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
import numpy as np

from asuka.frontend.periodic_table import atomic_number
from asuka.hf import df_scf as _df_scf
from asuka.integrals.grad import (
    compute_df_gradient_contributions_analytic_packed_bases,
    compute_df_gradient_contributions_fd_packed_bases,
)
from asuka.integrals.int1e_cart import contract_dS_cart, contract_dhcore_cart, shell_to_atom_map
from asuka.solver import GUGAFCISolver

from .state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from .cphf_df import solve_rhf_cphf_df


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

    B_ao = np.asarray(B_ao, dtype=np.float64)
    h_ao = np.asarray(h_ao, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
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

    dm1_act = np.asarray(dm1_act, dtype=np.float64)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")

    dm2_arr = np.asarray(dm2_act, dtype=np.float64)
    if dm2_arr.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act must have shape (ncas,ncas,ncas,ncas)")

    # AO densities
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = np.zeros((nao, nao), dtype=np.float64)
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
    X = np.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
    L_pact = np.einsum("mp,mvQ->pvQ", C, X, optimize=True)  # (nmo,ncas,naux)
    L_act = L_pact[ncore:nocc]  # (ncas,ncas,naux)

    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    L2 = L_act.reshape(ncas * ncas, -1)

    # T[Q,u,v] = sum_{w,x} L[w,x,Q] * dm2[w,x,u,v]
    T_flat = L2.T @ dm2_flat  # (naux,ncas^2)
    T = T_flat.reshape(L2.shape[1], ncas, ncas)
    g_dm2 = np.einsum("puQ,Quv->pv", L_pact, T, optimize=True)  # (nmo,ncas)

    # Generalized Fock (g) matrix: only core+active columns are defined.
    gfock = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        gfock[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gfock[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1_act + g_dm2

    return gfock, D_core_ao, D_act_ao, D_tot_ao, C_act


def _build_bar_L_casscf_df(
    B_ao: np.ndarray,
    *,
    D_core_ao: np.ndarray,
    D_act_ao: np.ndarray,
    C_act: np.ndarray,
    dm2_act: np.ndarray,
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

    Returns
    -------
    np.ndarray
        The partial derivative of energy w.r.t B tensor elements.
    """

    B_ao = np.asarray(B_ao, dtype=np.float64)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_core_ao = np.asarray(D_core_ao, dtype=np.float64)
    D_act_ao = np.asarray(D_act_ao, dtype=np.float64)
    if D_core_ao.shape != (nao, nao) or D_act_ao.shape != (nao, nao):
        raise ValueError("AO density shape mismatch")

    # Mean-field interaction with the core potential:
    #   E_cc + E_ca = Tr(D_w * (J(Dc) - 0.5 K(Dc))), where D_w = D_act + 0.5 D_core
    D_w = D_act_ao + 0.5 * D_core_ao

    B2 = B_ao.reshape(nao * nao, naux)
    rho = B2.T @ D_core_ao.reshape(nao * nao)  # (naux,)
    sigma = B2.T @ D_w.reshape(nao * nao)  # (naux,)

    bar_J = sigma[:, None, None] * D_core_ao[None, :, :] + rho[:, None, None] * D_w[None, :, :]

    BQ = np.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
    t1 = np.matmul(np.matmul(D_core_ao[None, :, :], BQ), D_w)  # D_core * B_Q * D_w
    t2 = np.matmul(np.matmul(D_w[None, :, :], BQ), D_core_ao)  # D_w * B_Q * D_core
    bar_K = -0.5 * (t1 + t2)

    bar_mean = bar_J + bar_K

    # Active-active 2-RDM term:
    #   E_aa = 0.5 Σ_{uvwx} dm2_uvwx (uv|wx)
    # with (uv|wx) ≈ Σ_Q L_uv,Q L_wx,Q and L_uv,Q = C^T B_Q C in active MO space.
    C_act = np.asarray(C_act, dtype=np.float64)
    if C_act.ndim != 2 or int(C_act.shape[0]) != int(nao):
        raise ValueError("C_act shape mismatch")
    ncas = int(C_act.shape[1])
    if ncas <= 0:
        raise ValueError("empty active space")

    dm2_arr = np.asarray(dm2_act, dtype=np.float64)
    if dm2_arr.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    # L_uv,Q in active MO indices
    X = np.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
    L_act = np.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)  # (ncas,ncas,naux)

    L2 = L_act.reshape(ncas * ncas, naux)
    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2  # (ncas^2,naux), M_uv,Q = sum_{wx} dm2_uvwx L_wx,Q
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = np.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    bar_act = np.einsum("mvQ,nv->Qmn", tmp, C_act, optimize=True)  # (naux,nao,nao)

    bar_L = bar_mean + bar_act
    bar_L = 0.5 * (bar_L + np.transpose(bar_L, (0, 2, 1)))
    return np.asarray(bar_L, dtype=np.float64, order="C")


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

    B_ao = np.asarray(B_ao, dtype=np.float64)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_left = np.asarray(D_left, dtype=np.float64)
    D_right = np.asarray(D_right, dtype=np.float64)
    if D_left.shape != (nao, nao) or D_right.shape != (nao, nao):
        raise ValueError("D_left/D_right shape mismatch")

    # Coulomb-like part: Tr(D_left J(D_right)) = u(D_left)·u(D_right)
    B2 = B_ao.reshape(nao * nao, naux)
    rho = B2.T @ D_right.reshape(nao * nao)  # (naux,)
    sigma = B2.T @ D_left.reshape(nao * nao)  # (naux,)

    bar = np.zeros((naux, nao, nao), dtype=np.float64)

    cJ = float(coeff_J)
    if cJ:
        bar += cJ * (sigma[:, None, None] * D_right[None, :, :] + rho[:, None, None] * D_left[None, :, :])

    # Exchange-like part: Tr(D_left K(D_right)) = Σ_Q Tr(D_left B_Q D_right B_Q)
    cK = float(coeff_K)
    if cK:
        BQ = np.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
        t1 = np.matmul(np.matmul(D_left[None, :, :], BQ), D_right)  # D_left * B_Q * D_right
        t2 = np.matmul(np.matmul(D_right[None, :, :], BQ), D_left)  # D_right * B_Q * D_left
        bar += cK * (t1 + t2)

    bar = 0.5 * (bar + np.transpose(bar, (0, 2, 1)))
    return np.asarray(bar, dtype=np.float64, order="C")


def casscf_nuc_grad_df(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    atmlst: Sequence[int] | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
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

    t0_total = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.clear()
        profile["df_backend"] = str(df_backend).strip().lower()
        profile["df_threads"] = int(df_threads)

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("DF nuclear gradients currently require cart=True")

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

    C = _asnumpy_f64(getattr(casscf, "mo_coeff"))
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
    t_gfock = time.perf_counter() if profile is not None else 0.0

    # Energy-weighted density for the overlap (Pulay) term.
    dme0 = C @ ((gfock + gfock.T) * 0.5) @ C.T

    # 1e AO derivatives (analytic), contracted on the fly to avoid allocating
    # (natm,3,nao,nao) tensors.
    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=D_tot_ao,
        shell_atom=shell_atom,
    )
    de_S = -contract_dS_cart(
        ao_basis,
        atom_coords_bohr=coords,
        M=dme0,
        shell_atom=shell_atom,
    )
    t_1e = time.perf_counter() if profile is not None else 0.0

    # 2e DF derivative contraction (FD on B)
    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=dm2_act,
    )
    t_barL = time.perf_counter() if profile is not None else 0.0

    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        t0_df = time.perf_counter() if profile is not None else 0.0
        de_df = compute_df_gradient_contributions_analytic_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            B_ao=B_ao,
            bar_L_ao=bar_L_ao,
            backend=str(df_backend),
            df_threads=int(df_threads),
            profile=df_prof,
        )
    except (NotImplementedError, RuntimeError):
        # Fallback for backends without analytic DF derivative kernels (e.g. CUDA).
        t0_df = time.perf_counter() if profile is not None else 0.0
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
    t_df = time.perf_counter() if profile is not None else 0.0

    try:
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)
    except Exception:
        de_nuc = np.zeros((natm, 3), dtype=np.float64)
    t_nuc = time.perf_counter() if profile is not None else 0.0

    grad = np.asarray(de_h1 + de_S + de_df + de_nuc, dtype=np.float64)
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    if profile is not None:
        profile["t_rdms_s"] = float(t_rdms - t0_total)
        profile["t_gfock_s"] = float(t_gfock - t_rdms)
        profile["t_1e_s"] = float(t_1e - t_gfock)
        profile["t_barL_s"] = float(t_barL - t_1e)
        profile["t_df_s"] = float(t_df - t0_df)
        profile["t_nuc_s"] = float(t_nuc - t_df)
        profile["t_total_s"] = float(t_nuc - t0_total)

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

    dme0 = C @ ((gfock + gfock.T) * 0.5) @ C.T

    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=D_tot_ao,
        shell_atom=shell_atom,
    )
    de_S = -contract_dS_cart(
        ao_basis,
        atom_coords_bohr=coords,
        M=dme0,
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

    grad = np.asarray(de_h1 + de_S + de_df + de_nuc, dtype=np.float64)
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
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("DF nuclear gradients currently require cart=True")

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

    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=np.asarray(D_tot_ao + zvec_ao, dtype=np.float64),
        shell_atom=shell_atom,
    )

    mS = 0.5 * (im1 + im1.T) + 2.0 * (0.5 * (zeta + zeta.T)) + 2.0 * (0.5 * (vhf_s1occ + vhf_s1occ.T))
    de_S = -contract_dS_cart(
        ao_basis,
        atom_coords_bohr=coords,
        M=np.asarray(mS, dtype=np.float64),
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
        de_df = compute_df_gradient_contributions_analytic_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            B_ao=B_ao,
            bar_L_ao=bar_L_tot,
            backend=str(df_backend),
            df_threads=int(df_threads),
            profile=df_prof,
        )
    except (NotImplementedError, RuntimeError):
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            getattr(scf_out, "ao_basis"),
            getattr(scf_out, "aux_basis"),
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_tot,
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

    grad = np.asarray(de_h1 + de_S + de_df + de_nuc, dtype=np.float64)
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


__all__ = [
    "DFNucGradResult",
    "casscf_nuc_grad_df",
    "casci_nuc_grad_df_unrelaxed",
    "casci_nuc_grad_df_relaxed",
]
