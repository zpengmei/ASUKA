from __future__ import annotations

"""SA-CASSCF nonadiabatic coupling vectors (DF).

This module computes SA-CASSCF derivative couplings using:
- ASUKA DF integral derivatives (1e analytic + DF 2e analytic/FD backends)
- ASUKA Newton-CASSCF operator (`asuka.mcscf.newton_casscf`, internal mode)
- ASUKA Z-vector solver (`asuka.mcscf.zvector.solve_mcscf_zvector`)

Current scope:
- Hamiltonian-response terms are analytic on DF integrals.
- Response terms support both `split_orbfd` (recommended) and
  `fd_jacobian` (validation-oriented, more expensive).
"""

from contextlib import contextmanager
from typing import Any, Callable, Literal, Sequence
import warnings

import numpy as np

from asuka.frontend.molecule import Molecule
from asuka.frontend.periodic_table import atomic_number
from asuka.hf import df_scf as _df_scf
from asuka.integrals.df_context import DFCholeskyContext
from asuka.integrals.df_grad_context import DFGradContractionContext
from asuka.integrals.grad import (
    compute_df_gradient_contributions_analytic_packed_bases,
    compute_df_gradient_contributions_fd_packed_bases,
)
from asuka.integrals.int1e_cart import (
    build_int1e_cart,
    contract_dS_ip_cart,
    contract_dhcore_cart,
    shell_to_atom_map,
)
from asuka.hf.df_scf import _get_xp as _get_xp_arrays
from asuka.mcscf.nuc_grad_df import (
    _build_bar_L_casscf_df,
    _build_bar_L_df_cross,
    _build_gfock_casscf_df,
    _resolve_xp,
    _as_xp_f64,
    _apply_df_pool_policy,
    _log_vram,
    _flush_gpu_pool,
    _symmetrize_bar_L_inplace,
)
from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter, DFNewtonERIs
from asuka.mcscf import newton_casscf as _newton_casscf
from asuka.mcscf.state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from asuka.mcscf.zvector import build_mcscf_hessian_operator, solve_mcscf_zvector, _project_sa_ci_components
from asuka.solver import GUGAFCISolver


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def _warn_df_fd_fallback(*, where: str, backend: str, delta_bohr: float, err: Exception) -> None:
    warnings.warn(
        f"DF NAC derivative contraction fell back to finite-difference on B in {where} "
        f"(backend={backend!s}, delta_bohr={float(delta_bohr):.3e}); "
        "this may introduce noisy/non-conservative couplings in dynamics. "
        f"Reason: {type(err).__name__}: {err}",
        RuntimeWarning,
        stacklevel=3,
    )


def _base_fcisolver_method(fcisolver: Any, name: str):
    """Return the (possibly unwrapped) fcisolver method ``name``.

    PySCF state-average fcisolver wrappers implement ``trans_rdm12``/``trans_rdm1``
    with list semantics. For pairwise transition densities we need the underlying
    single-root method (on the base solver class).
    """

    base_cls = getattr(fcisolver, "_base_class", None)
    if base_cls is not None and hasattr(base_cls, name):
        fn_unbound = getattr(base_cls, name)

        def _call(solver_obj: Any, *args: Any, **kwargs: Any):
            return fn_unbound(solver_obj, *args, **kwargs)

        return _call

    if not hasattr(fcisolver, name):
        raise AttributeError(f"fcisolver does not implement {name}")

    fn_bound = getattr(fcisolver, name)

    def _call(_solver_obj: Any, *args: Any, **kwargs: Any):
        return fn_bound(*args, **kwargs)

    return _call


def _symm_sqrt(a: np.ndarray, *, inv: bool, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    w, v = np.linalg.eigh(a)
    w = np.maximum(w, float(eps))
    if inv:
        s = 1.0 / np.sqrt(w)
    else:
        s = np.sqrt(w)
    return (v * s[None, :]) @ v.T


@contextmanager
def _force_internal_newton():
    import os

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


class _FixedRDMFcisolver:
    """Delegate wrapper that overrides make_rdm* to return fixed transition densities."""

    def __init__(self, base: Any, dm1: np.ndarray, dm2: np.ndarray):
        self._base = base
        self._dm1 = np.asarray(dm1, dtype=np.float64)
        self._dm2 = np.asarray(dm2, dtype=np.float64)

    def make_rdm12(self, *_a: Any, **_k: Any):
        return self._dm1, self._dm2

    def make_rdm1(self, *_a: Any, **_k: Any):
        return self._dm1

    def make_rdm2(self, *_a: Any, **_k: Any):
        return self._dm2

    # Important: `newton_casscf` prefers `states_make_rdm12` when available. Many solvers
    # (including ASUKA's) implement it, so we must override it here to prevent the
    # fixed transition densities from being bypassed.
    def states_make_rdm12(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm1 = np.asarray(self._dm1, dtype=np.float64)
        dm2 = np.asarray(self._dm2, dtype=np.float64)
        return np.broadcast_to(dm1, (n,) + dm1.shape).copy(), np.broadcast_to(dm2, (n,) + dm2.shape).copy()

    def states_make_rdm1(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm1 = np.asarray(self._dm1, dtype=np.float64)
        return np.broadcast_to(dm1, (n,) + dm1.shape).copy()

    def states_make_rdm2(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm2 = np.asarray(self._dm2, dtype=np.float64)
        return np.broadcast_to(dm2, (n,) + dm2.shape).copy()

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def _eris_patch_active(eris: DFNewtonERIs, *, mo_coeff: np.ndarray, hcore_ao: np.ndarray, ncore: int) -> DFNewtonERIs:
    ncore = int(ncore)
    if ncore <= 0:
        return eris
    mo = np.asarray(mo_coeff, dtype=np.float64)
    moH = mo.T
    vnocore = np.asarray(getattr(eris, "vhf_c"), dtype=np.float64).copy()
    vnocore[:, :ncore] = -moH @ np.asarray(hcore_ao, dtype=np.float64) @ mo[:, :ncore]
    return DFNewtonERIs(ppaa=eris.ppaa, papa=eris.papa, vhf_c=vnocore, j_pc=eris.j_pc, k_pc=eris.k_pc)


def _mol_coords_charges_bohr(mol: Any) -> tuple[np.ndarray, np.ndarray]:
    mol0 = mol
    coords = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("mol.coords_bohr must have shape (natm,3)")
    charges = np.asarray([float(atomic_number(sym)) for sym in getattr(mol0, "elements")], dtype=np.float64)
    if charges.shape != (int(coords.shape[0]),):
        raise ValueError("invalid mol.elements for charge construction")
    return coords, charges


def _displaced_molecule(mol: Molecule, *, ia: int, axis: int, delta_bohr: float) -> Molecule:
    ia = int(ia)
    axis = int(axis)
    delta_bohr = float(delta_bohr)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0,1,2")
    if abs(delta_bohr) <= 0.0:
        raise ValueError("delta_bohr must be non-zero")

    atoms = []
    for sym, xyz in mol.atoms_bohr:
        atoms.append((str(sym), np.asarray(xyz, dtype=np.float64).copy()))
    atoms[ia][1][axis] += delta_bohr
    return Molecule.from_atoms(
        atoms,
        unit="Bohr",
        charge=int(getattr(mol, "charge", 0)),
        spin=int(getattr(mol, "spin", 0)),
        basis=getattr(mol, "basis", None),
        cart=bool(getattr(mol, "cart", True)),
    )


def _core_energy_weighted_density(
    *,
    mo_coeff: np.ndarray,
    hcore_ao: np.ndarray,
    B_ao: np.ndarray,
    ncore: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (D_core_ao, dme_sf_core) for the core-only RHF-like term."""

    xp, _ = _get_xp_arrays(B_ao, mo_coeff)
    mo = xp.asarray(mo_coeff, dtype=xp.float64)
    nao, nmo = map(int, mo.shape)
    ncore = int(ncore)
    if ncore < 0 or ncore > nmo:
        raise ValueError("invalid ncore")

    if ncore:
        mo_core = mo[:, :ncore]
        D_core_ao = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = _df_scf._df_JK(B_ao, D_core_ao, want_J=True, want_K=True)  # noqa: SLF001
        v_ao = xp.asarray(Jc - 0.5 * Kc, dtype=xp.float64)
    else:
        D_core_ao = xp.zeros((nao, nao), dtype=xp.float64)
        v_ao = xp.zeros((nao, nao), dtype=xp.float64)

    f0 = mo.T @ xp.asarray(hcore_ao, dtype=xp.float64) @ mo
    f0 = xp.asarray(f0, dtype=xp.float64)
    if ncore:
        f0 = f0 + (mo.T @ v_ao @ mo)

    mo_occ = xp.zeros((nmo,), dtype=xp.float64)
    mo_occ[:ncore] = 2.0
    f0_occ = f0 * mo_occ[None, :]
    dme_sf = mo @ ((f0_occ + f0_occ.T) * 0.5) @ mo.T
    return xp.asarray(D_core_ao, dtype=xp.float64), xp.asarray(dme_sf, dtype=xp.float64)


def _build_bar_L_net_active_df(
    B_ao: Any,
    C: Any,
    dm1_act: Any,
    dm2_act: Any,
    *,
    ncore: int,
    ncas: int,
    xp: Any,
    L_act: Any | None = None,
    rho_core: Any | None = None,
) -> tuple[Any, Any]:
    """Return (bar_L_net, D_act_ao) for the active DF gradient contribution.

    bar_L_net = bar_L_casscf - bar_L_core_sub (fused single-pass).

    Instead of building bar_L_casscf (sizeof_B) and bar_L_core (sizeof_B)
    separately and subtracting, this computes the net directly.  Peak VRAM is
    1 sizeof_B (the result) instead of 2 sizeof_B.

    Math:
        D_w = D_act + 0.5*D_core              (weighted density)
        D_ah = D_w - D_core = D_act - 0.5*D_core   (net offset)
        net_J  = u(D_w)*D_core + u(D_core)*D_ah
        net_K  = -0.5*(D_core @ BQ @ D_ah + D_w @ BQ @ D_core)
        net_act = bar_act from 2-RDM          (unchanged, core has no active term)
        bar_L_net = 0.5 * (net + net^T)
    """
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas
    nao = int(C.shape[0])

    C = xp.asarray(C, dtype=xp.float64)
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    D_core_ao = 2.0 * (C_core @ C_core.T) if ncore else xp.zeros((nao, nao), dtype=xp.float64)
    dm1 = xp.asarray(dm1_act, dtype=xp.float64)
    D_act_ao = C_act @ dm1 @ C_act.T

    # Weighted density and net offset (both nao×nao — negligible VRAM).
    D_w = D_act_ao + 0.5 * D_core_ao
    D_ah = D_w - D_core_ao  # = D_act - 0.5*D_core

    B_ao = xp.asarray(B_ao, dtype=xp.float64)
    nao_b, _, naux = map(int, B_ao.shape)
    B2 = B_ao.reshape(nao * nao, naux)
    if rho_core is None:
        rho = B2.T @ D_core_ao.reshape(nao * nao)  # (naux,)
    else:
        rho = xp.asarray(rho_core, dtype=xp.float64)
    sigma_w = B2.T @ D_w.reshape(nao * nao)  # (naux,)

    # --- Net Coulomb: sigma_w * D_core + rho * D_ah ---
    _log_vram("  net_active_fused: before net Coulomb")
    bar_net = sigma_w[:, None, None] * D_core_ao[None, :, :]
    bar_net += rho[:, None, None] * D_ah[None, :, :]
    _log_vram("  net_active_fused: after net Coulomb")

    # --- Net Exchange (chunked): -0.5*(D_core @ BQ @ D_ah + D_w @ BQ @ D_core) ---
    if ncore:
        BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao) view
        _chunk = max(1, naux // 4)
        for _q0 in range(0, naux, _chunk):
            _q1 = min(_q0 + _chunk, naux)
            _bq = BQ[_q0:_q1]
            _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_ah)
            _t += xp.matmul(xp.matmul(D_w[None, :, :], _bq), D_core_ao)
            _t *= -0.5
            bar_net[_q0:_q1] += _t
            del _t
    _log_vram("  net_active_fused: after net exchange")

    # --- Active 2-RDM term (identical to _build_bar_L_casscf_df) ---
    dm2_arr = xp.asarray(dm2_act, dtype=xp.float64)
    if L_act is None:
        X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
        L_act_val = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)
    else:
        L_act_val = xp.asarray(L_act, dtype=xp.float64)

    L2 = L_act_val.reshape(ncas * ncas, naux)
    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)
    # Accumulate bar_act into bar_net in chunks to avoid sizeof_B temporary.
    _chunk = max(1, naux // 4)
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        bar_net[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_act, optimize=True)
    del tmp
    _log_vram("  net_active_fused: after bar_act")

    # --- Symmetrize in-place ---
    _symmetrize_bar_L_inplace(bar_net, xp)
    _log_vram("  net_active_fused: after sym")
    return xp.asarray(bar_net, dtype=xp.float64), D_act_ao


def _build_bar_L_lorb_df(
    B_ao: Any,
    C: Any,
    Lorb: Any,
    dm1_act: Any,
    dm2_act: Any,
    *,
    ncore: int,
    ncas: int,
    xp: Any,
) -> tuple[Any, Any]:
    """Return (bar_L_lorb, D_L_ao) for the orbital Lagrange DF gradient contribution.

    bar_L_lorb is the bar_L tensor from _Lorb_dot_dgorb_dx_df, extracted without
    the gfock/JK construction and without calling contract().
    D_L_ao is the L-effective density for the 1e term (contract_dhcore_cart M=D_L_ao).

    The caller can sum bar_L_lorb with other bar_L tensors and issue a single
    contract() call, replacing the separate contract() inside _Lorb_dot_dgorb_dx_df.
    """
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas
    nao = int(C.shape[0])

    L = xp.asarray(Lorb, dtype=xp.float64)
    dm1 = xp.asarray(dm1_act, dtype=xp.float64)
    dm2 = xp.asarray(dm2_act, dtype=xp.float64)

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    C_L = C @ L
    C_L_core = C_L[:, :ncore]
    C_L_act = C_L[:, ncore:nocc]

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

    # Mean-field bar_L (two _build_bar_L_df_cross calls, no _df_JK needed).
    D_w = D_act + 0.5 * D_core
    D_wL = D_L_act + 0.5 * D_L_core
    _log_vram("  lorb: before cross1")
    bar_mean = _build_bar_L_df_cross(
        B_ao, D_left=D_wL, D_right=D_core, coeff_J=1.0, coeff_K=-0.5, symmetrize=False,
    )
    _log_vram("  lorb: after cross1")
    _flush_gpu_pool()
    _log_vram("  lorb: after cross1 flush")
    _build_bar_L_df_cross(
        B_ao, D_left=D_w, D_right=D_L_core, coeff_J=1.0, coeff_K=-0.5,
        out=bar_mean, symmetrize=False,
    )
    _log_vram("  lorb: after cross2")

    # Active-active DF term: linearize bar_L_casscf active block w.r.t C_act along C_L_act.
    dm2_flat = dm2.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)

    X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
    L_act_mo = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)
    L2 = L_act_mo.reshape(ncas * ncas, -1)
    del L_act_mo
    M_mat = dm2_flat @ L2
    del L2
    M_uvQ = M_mat.reshape(ncas, ncas, -1)
    del M_mat

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)
    tmp_L = xp.einsum("mu,uvQ->mvQ", C_L_act, M_uvQ, optimize=True)
    del M_uvQ

    X_L = xp.einsum("mnQ,nv->mvQ", B_ao, C_L_act, optimize=True)
    dL_act = (
        xp.einsum("mu,mvQ->uvQ", C_L_act, X, optimize=True)
        + xp.einsum("mu,mvQ->uvQ", C_act, X_L, optimize=True)
    )
    del X, X_L
    dL2 = dL_act.reshape(ncas * ncas, -1)
    del dL_act
    dM = dm2_flat @ dL2
    del dL2
    dM_uvQ = dM.reshape(ncas, ncas, -1)
    del dM
    tmp_M = xp.einsum("mu,uvQ->mvQ", C_act, dM_uvQ, optimize=True)
    del dM_uvQ

    # Accumulate all three bar_act einsum terms directly into bar_mean in chunks.
    _naux = int(bar_mean.shape[0])
    _chunk = max(1, _naux // 4)
    for _q0 in range(0, _naux, _chunk):
        _q1 = min(_q0 + _chunk, _naux)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp_L[:, :, _q0:_q1], C_act, optimize=True)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_L_act, optimize=True)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp_M[:, :, _q0:_q1], C_act, optimize=True)
    del tmp_L, tmp, tmp_M

    _symmetrize_bar_L_inplace(bar_mean, xp)
    return bar_mean, D_L


def _grad_elec_active_df(
    *,
    scf_out: Any,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    df_backend: str,
    df_config: Any | None,
    df_threads: int,
    df_grad_ctx: DFGradContractionContext | None = None,
    fd_delta_bohr: float = 1e-4,
    return_terms: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return active-electron part of <dH/dR> using DF integrals (no nuclear term)."""

    xp, _ = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_grad_elec_active_df")
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    C = _as_xp_f64(xp, mo_coeff)

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=xp.asarray(dm1_act, dtype=xp.float64),
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
    )

    # ── Phase 1: Build both bar_L tensors on GPU (cheap, before any contract call) ──
    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
    )
    # Core-only RHF-like contribution (GPU).
    D_core_only, dme_core = _core_energy_weighted_density(mo_coeff=C, hcore_ao=h_ao, B_ao=B_ao, ncore=int(ncore))
    bar_L_core = _build_bar_L_df_cross(
        B_ao,
        D_left=D_core_only,
        D_right=D_core_only,
        coeff_J=0.5,
        coeff_K=-0.25,
    )

    if not bool(return_terms):
        # ── Fast path: single fused contract() call (bar_L_ao - bar_L_core) ──
        bar_L_ao -= bar_L_core
        del bar_L_core
        bar_L_net = bar_L_ao
        try:
            if df_grad_ctx is not None:
                de_df_net = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_net)
            else:
                de_df_net = compute_df_gradient_contributions_analytic_packed_bases(
                    ao_basis,
                    aux_basis,
                    atom_coords_bohr=coords,
                    B_ao=B_ao,
                    bar_L_ao=bar_L_net,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend),
                    df_threads=int(df_threads),
                    profile=None,
                )
        except (NotImplementedError, RuntimeError) as e:
            _warn_df_fd_fallback(
                where="_grad_elec_active_df(total)",
                backend=str(df_backend),
                delta_bohr=float(fd_delta_bohr),
                err=e,
            )
            de_df_net = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                bar_L_ao=bar_L_net,
                backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                delta_bohr=float(fd_delta_bohr),
                profile=None,
            )
        D_tot_cpu = _asnumpy_f64(D_tot_ao)
        D_core_cpu = _asnumpy_f64(D_core_only)
        de_h1 = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=D_tot_cpu, shell_atom=shell_atom,
        )
        de_h1_core = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=D_core_cpu, shell_atom=shell_atom,
        )
        _restore_pool()
        return np.asarray(de_h1 + _asnumpy_f64(de_df_net) - de_h1_core, dtype=np.float64)

    # ── Debug path (return_terms=True): keep 2 separate contract() calls for breakdown ──
    try:
        if df_grad_ctx is not None:
            de_df = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_ao)
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_ao,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        _warn_df_fd_fallback(
            where="_grad_elec_active_df(df2e)",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_ao,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(fd_delta_bohr),
            profile=None,
        )

    try:
        if df_grad_ctx is not None:
            de_df_core = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_core)
        else:
            de_df_core = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_core,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        _warn_df_fd_fallback(
            where="_grad_elec_active_df(df2e_core_sub)",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df_core = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_core,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(fd_delta_bohr),
            profile=None,
        )

    D_tot_cpu = _asnumpy_f64(D_tot_ao)
    D_core_cpu = _asnumpy_f64(D_core_only)
    de_h1 = contract_dhcore_cart(
        ao_basis, atom_coords_bohr=coords, atom_charges=charges,
        M=D_tot_cpu, shell_atom=shell_atom,
    )
    de_h1_core = contract_dhcore_cart(
        ao_basis, atom_coords_bohr=coords, atom_charges=charges,
        M=D_core_cpu, shell_atom=shell_atom,
    )
    total = np.asarray(de_h1 + _asnumpy_f64(de_df) - de_h1_core - _asnumpy_f64(de_df_core), dtype=np.float64)
    terms = {
        "dhcore": np.asarray(de_h1, dtype=np.float64),
        "df2e": np.asarray(de_df, dtype=np.float64),
        # Core-only RHF-like subtraction pieces (returned as the *contribution to total*).
        "dhcore_core_sub": np.asarray(-de_h1_core, dtype=np.float64),
        "df2e_core_sub": np.asarray(-de_df_core, dtype=np.float64),
        # Generalized Fock (MO basis, nmo×nmo) and core energy-weighted density (AO basis)
        # for computing the Pulay (overlap-derivative) term externally.
        "gfock": _asnumpy_f64(gfock),
        "dme_core": _asnumpy_f64(dme_core),
    }
    _restore_pool()
    return total, terms


def _grad_elec_casscf_df(
    *,
    scf_out: Any,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    df_backend: str,
    df_config: Any | None,
    df_threads: int,
    df_grad_ctx: DFGradContractionContext | None = None,
    fd_delta_bohr: float = 1e-4,
) -> np.ndarray:
    """Return electronic SA-CASSCF/CASSCF gradient (no nuclear term), DF-only."""

    xp, _ = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_grad_elec_casscf_df")
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    C = _as_xp_f64(xp, mo_coeff)

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=xp.asarray(dm1_act, dtype=xp.float64),
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
    )
    # ── GPU phase: bar_L + DF 2e contraction (before 1e to keep GPU busy) ──
    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
    )
    try:
        if df_grad_ctx is not None:
            de_df = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_ao)
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_ao,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        _warn_df_fd_fallback(
            where="_grad_elec_casscf_df",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_ao,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(fd_delta_bohr),
            profile=None,
        )

    # ── CPU phase: 1e contractions (device synced by contract()) ──
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=_asnumpy_f64(D_tot_ao),
        shell_atom=shell_atom,
    )

    _restore_pool()
    return np.asarray(de_h1 + _asnumpy_f64(de_df), dtype=np.float64)


def _Lorb_dot_dgorb_dx_df(
    *,
    scf_out: Any,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    Lorb: np.ndarray,
    ncore: int,
    ncas: int,
    eris: DFNewtonERIs,
    df_backend: str,
    df_config: Any | None,
    df_threads: int,
    df_grad_ctx: DFGradContractionContext | None = None,
    fd_delta_bohr: float = 1e-4,
    return_terms: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Analytic DF orbital Lagrange term contribution (PySCF Lorb_dot_dgorb_dx analogue).

    This computes the nuclear derivative contribution corresponding to the orbital
    Lagrange multipliers, in the same spirit as PySCF's
    ``pyscf.grad.sacasscf.Lorb_dot_dgorb_dx`` but using ASUKA's DF derivative
    contractions.

    Notes
    -----
    - This is used for the split response backend in NACs (``split_orbfd``).
    - The active-space RDMs are treated as fixed inputs (state-averaged).
    """

    xp, _ = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_Lorb_dot_dgorb_dx_df")
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))

    C = _as_xp_f64(xp, mo_coeff)
    L = _as_xp_f64(xp, Lorb)
    if C.ndim != 2 or L.ndim != 2:
        raise ValueError("mo_coeff and Lorb must be 2D arrays")
    nao, nmo = map(int, C.shape)
    if tuple(L.shape) != (nmo, nmo):
        raise ValueError("Lorb shape mismatch")

    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas
    if ncore < 0 or ncas <= 0 or nocc > nmo:
        raise ValueError("invalid ncore/ncas for Lorb response")

    dm1_act = xp.asarray(dm1_act, dtype=xp.float64)
    dm2_act = xp.asarray(dm2_act, dtype=xp.float64)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")
    if dm2_act.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    # Core/active blocks and L-contracted MO coefficients.
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    C_L = C @ L
    C_L_core = C_L[:, :ncore]
    C_L_act = C_L[:, ncore:nocc]

    # AO densities: D and the L-effective ~D (PySCF dm1L).
    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core = 2.0 * (C_L_core @ C_core.T)
        D_L_core = D_L_core + D_L_core.T
    else:
        D_core = xp.zeros((nao, nao), dtype=xp.float64)
        D_L_core = xp.zeros((nao, nao), dtype=xp.float64)

    D_act = C_act @ dm1_act @ C_act.T
    D_L_act = C_L_act @ dm1_act @ C_act.T
    D_L_act = D_L_act + D_L_act.T

    D_tot = D_core + D_act
    D_L = D_L_core + D_L_act

    # Build the L-effective generalized Fock matrix for the orbital Lagrange term.
    # This mirrors PySCF's construction in Lorb_dot_dgorb_dx, but uses DF J/K.
    if ncore:
        Jc, Kc = _df_scf._df_JK(B_ao, D_core, want_J=True, want_K=True)  # noqa: SLF001
    else:
        Jc = xp.zeros((nao, nao), dtype=xp.float64)
        Kc = xp.zeros((nao, nao), dtype=xp.float64)
    Ja, Ka = _df_scf._df_JK(B_ao, D_act, want_J=True, want_K=True)  # noqa: SLF001
    if ncore:
        JcL, KcL = _df_scf._df_JK(B_ao, D_L_core, want_J=True, want_K=True)  # noqa: SLF001
    else:
        JcL = xp.zeros((nao, nao), dtype=xp.float64)
        KcL = xp.zeros((nao, nao), dtype=xp.float64)
    JaL, KaL = _df_scf._df_JK(B_ao, D_L_act, want_J=True, want_K=True)  # noqa: SLF001

    vhf_c = Jc - 0.5 * Kc
    vhf_a = Ja - 0.5 * Ka
    vhfL_c = JcL - 0.5 * KcL
    vhfL_a = JaL - 0.5 * KaL

    gfock = h_ao @ D_L
    gfock = gfock + (vhf_c + vhf_a) @ D_L_core
    gfock = gfock + (vhfL_c + vhfL_a) @ D_core
    gfock = gfock + vhfL_c @ D_act
    gfock = gfock + vhf_c @ D_L_act

    # Convert AO->(MO definition) by left-multiplying S^{-1} ≈ C C^T (PySCF convention).
    s0_inv = C @ C.T
    gfock = s0_inv @ gfock

    # Two-electron (active-active) part: reproduce PySCF's aapa/aapaL contraction using DF ERIs.
    # This contraction is validation-oriented and can be costly for large ncas.
    ppaa = xp.asarray(getattr(eris, "ppaa"), dtype=xp.float64)
    papa = xp.asarray(getattr(eris, "papa"), dtype=xp.float64)
    if ppaa.ndim != 4 or papa.ndim != 4:
        raise ValueError("unexpected ERI tensor ndim for Lorb response")

    aapa = xp.zeros((ncas, ncas, nmo, ncas), dtype=xp.float64)
    aapaL = xp.zeros_like(aapa)
    L_act_slice = L[:, ncore:nocc]
    for i in range(nmo):
        jbuf = ppaa[i]  # (nmo,ncas,ncas)
        kbuf = papa[i]  # (ncas,nmo,ncas)
        aapa[:, :, i, :] = xp.asarray(jbuf[ncore:nocc, :, :], dtype=xp.float64).transpose(1, 2, 0)
        aapaL[:, :, i, :] += xp.tensordot(jbuf, L_act_slice, axes=((0,), (0,)))
        kk = xp.tensordot(kbuf, L_act_slice, axes=((1,), (0,))).transpose(1, 2, 0)
        aapaL[:, :, i, :] += kk + kk.transpose(1, 0, 2)

    dm2 = xp.asarray(dm2_act, dtype=xp.float64)
    t1 = xp.einsum("uviw,uvtw->it", aapaL, dm2, optimize=True)
    t2 = xp.einsum("uviw,vuwt->it", aapa, dm2, optimize=True)
    gfock = gfock + (C @ t1 @ C_act.T)
    gfock = gfock + (C @ t2 @ C_L_act.T)

    # ── GPU phase: bar_L build + DF 2e contraction (before 1e to keep GPU busy) ──
    # Two-electron DF derivative term: build bar_L for the L-effective contraction.
    D_w = D_act + 0.5 * D_core
    D_wL = D_L_act + 0.5 * D_L_core
    bar_mean = _build_bar_L_df_cross(
        B_ao, D_left=D_wL, D_right=D_core, coeff_J=1.0, coeff_K=-0.5, symmetrize=False,
    )
    _build_bar_L_df_cross(
        B_ao, D_left=D_w, D_right=D_L_core, coeff_J=1.0, coeff_K=-0.5,
        out=bar_mean, symmetrize=False,
    )

    # Active-active DF term: linearize _build_bar_L_casscf_df active block w.r.t C_act along C_L_act.
    X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)  # (nao,ncas,naux)
    L_act_mo = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)  # (ncas,ncas,naux)
    L2 = L_act_mo.reshape(ncas * ncas, -1)

    dm2_flat = dm2.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)

    M_mat = dm2_flat @ L2  # (ncas^2,naux)
    M_uvQ = M_mat.reshape(ncas, ncas, -1)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    tmp_L = xp.einsum("mu,uvQ->mvQ", C_L_act, M_uvQ, optimize=True)

    # δM from δL_act induced by δC_act = C_L_act
    X_L = xp.einsum("mnQ,nv->mvQ", B_ao, C_L_act, optimize=True)
    dL_act = xp.einsum("mu,mvQ->uvQ", C_L_act, X, optimize=True) + xp.einsum("mu,mvQ->uvQ", C_act, X_L, optimize=True)
    dL2 = dL_act.reshape(ncas * ncas, -1)
    dM = dm2_flat @ dL2
    dM_uvQ = dM.reshape(ncas, ncas, -1)
    tmp_M = xp.einsum("mu,uvQ->mvQ", C_act, dM_uvQ, optimize=True)

    # Accumulate all three bar_act einsum terms directly into bar_mean in chunks.
    naux_lorb = int(bar_mean.shape[0])
    _chunk = max(1, naux_lorb // 4)
    for _q0 in range(0, naux_lorb, _chunk):
        _q1 = min(_q0 + _chunk, naux_lorb)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp_L[:, :, _q0:_q1], C_act, optimize=True)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_L_act, optimize=True)
        bar_mean[_q0:_q1] += xp.einsum("mvQ,nv->Qmn", tmp_M[:, :, _q0:_q1], C_act, optimize=True)
    del tmp_L, tmp, tmp_M

    _symmetrize_bar_L_inplace(bar_mean, xp)
    bar_L_ao = bar_mean

    try:
        if df_grad_ctx is not None:
            de_df = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_ao)
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao,
                bar_L_ao=bar_L_ao,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        _warn_df_fd_fallback(
            where="_Lorb_dot_dgorb_dx_df",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            bar_L_ao=bar_L_ao,
            backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(fd_delta_bohr),
            profile=None,
        )

    # ── CPU phase: 1e derivative contractions (device synced by contract()) ──
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M=_asnumpy_f64(D_L),
        shell_atom=shell_atom,
    )
    # de_df already on CPU (contract() returns numpy).
    total = np.asarray(de_h1 + _asnumpy_f64(de_df), dtype=np.float64)
    if not bool(return_terms):
        _restore_pool()
        return total
    terms = {
        "dhcore": np.asarray(de_h1, dtype=np.float64),
        "df2e": _asnumpy_f64(de_df),
        # Generalized Fock in AO-like basis (nao×nao, with S^{-1} left-multiplied + 2e terms)
        # for computing the Pulay (overlap-derivative) term externally.
        "gfock": _asnumpy_f64(gfock),
    }
    _restore_pool()
    return total, terms


def sacasscf_nonadiabatic_couplings_df(
    scf_out: Any,
    casscf: Any,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    atmlst: Sequence[int] | None = None,
    use_etfs: bool = False,
    mult_ediff: bool = False,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    auxbasis: Any | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    response_term: Literal["fd", "fd_jacobian", "split_orbfd"] = "split_orbfd",
    delta_bohr: float = 1e-4,
    fd_integrals_builder: Callable[[Molecule], tuple[Any, Any, Any]] | None = None,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> np.ndarray:
    """Compute SA-CASSCF NACVs (<bra|d/dR|ket>) using ASUKA DF integrals.

    Parameters
    ----------
    scf_out
        DF-SCF output providing at least: ``mol``, ``ao_basis``, ``aux_basis``,
        ``df_B``, and ``int1e.hcore``.
    casscf
        SA-CASSCF result providing at least: ``mo_coeff``, ``ci`` (list),
        ``ncore``, ``ncas``, ``nelecas``, and energies (``e_states`` or ``e_roots``).
    pairs
        Optional list of (ket, bra) root pairs. If None, compute all off-diagonal pairs.
    atmlst
        Atom indices to compute. If None, compute all atoms.
    use_etfs
        If True, skip the explicit AO-overlap (CSF) term.
    mult_ediff
        If True, return numerator (multiplied by energy difference).
    response_term
        Lagrange response backend:
        - ``'split_orbfd'`` (default): PySCF-style split response
          ``LdotJ = Lci_dot_dgci + Lorb_dot_dgorb`` with orbital-FD.
        - ``'fd_jacobian'``: finite-difference Jacobian of SA stationarity conditions (debug).
        - ``'fd'``: alias for ``'fd_jacobian'`` (kept for historical tests).
    delta_bohr
        Displacement (Bohr) for finite-difference response term. Also used as
        the fallback DF-B finite-difference step if an analytic DF derivative
        contraction path is unavailable at runtime.
    fd_integrals_builder
        Optional callback to build displaced-geometry integral arrays for the FD
        response term. Signature::

            (B_ao, hcore_ao, S_ao) = fd_integrals_builder(mol_disp)

        This hook is intended for *parity testing* (e.g., building B_ao from
        PySCF DF) and advanced backends. If omitted, displaced integrals are
        rebuilt via :class:`~asuka.integrals.df_context.DFCholeskyContext` and
        :func:`~asuka.integrals.int1e_cart.build_int1e_cart`.
    z_tol, z_maxiter
        Z-vector linear solve controls.
    """

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("DF NAC currently requires cart=True")

    coords, _charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])

    if atmlst is None:
        atmlst_use = list(range(natm))
    else:
        atmlst_use = [int(a) for a in atmlst]

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nocc = ncore + ncas

    ci_list = None
    ci_raw = getattr(casscf, "ci", None)
    if isinstance(ci_raw, (list, tuple)):
        nroots = int(len(ci_raw))
        ci_list = ci_as_list(ci_raw, nroots=nroots)
    else:
        nroots = int(getattr(casscf, "nroots", 1))
        ci_list = ci_as_list(ci_raw, nroots=nroots)

    if nroots <= 1:
        return np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    weights_in = getattr(casscf, "root_weights", None)
    if weights_in is None:
        weights_in = getattr(casscf, "weights", None)
    weights = normalize_weights(weights_in, nroots=nroots)

    e_raw = getattr(casscf, "e_states", None)
    if e_raw is None:
        e_raw = getattr(casscf, "e_roots", None)
    if e_raw is None:
        raise ValueError("casscf must provide per-root energies as e_states or e_roots")
    e_states = np.asarray(e_raw, dtype=np.float64).ravel()
    if int(e_states.size) != nroots:
        raise ValueError("energy array length mismatch with nroots")

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

    C_ref = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    if C_ref.ndim != 2:
        raise ValueError("casscf.mo_coeff must be a 2D array (nao,nmo)")
    nao, nmo = map(int, C_ref.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    B_ref = _asnumpy_f64(getattr(scf_out, "df_B"))
    hcore_ref = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    ao_basis_ref = getattr(scf_out, "ao_basis")
    aux_basis_ref = getattr(scf_out, "aux_basis")  # required for DF gradient contractions

    df_grad_ctx: DFGradContractionContext | None = None
    try:
        df_grad_ctx = DFGradContractionContext.build(
            ao_basis_ref,
            aux_basis_ref,
            atom_coords_bohr=coords,
            backend=str(df_backend),
            df_threads=int(df_threads),
            L_chol=getattr(scf_out, "df_L", None),
        )
    except (NotImplementedError, RuntimeError):
        df_grad_ctx = None

    # SA reference "mc" adapter and a reusable Hessian operator.
    mc_sa = DFNewtonCASSCFAdapter(
        df_B=B_ref,
        hcore_ao=hcore_ref,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=C_ref,
        fcisolver=fcisolver_use,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
        frozen=getattr(casscf, "frozen", None),
        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
        extrasym=getattr(casscf, "extrasym", None),
    )
    # Build the SA objective Hessian with *unpatched* ERIs (matches PySCF's
    # `sacasscf.Gradients.make_fcasscf_sa()` + `newton_casscf.gen_g_hop` usage).
    eris_sa = mc_sa.ao2mo(C_ref)
    with _force_internal_newton():
        hess_op = build_mcscf_hessian_operator(
            mc_sa,
            mo_coeff=C_ref,
            ci=ci_list,
            eris=eris_sa,
            use_newton_hessian=True,
        )

    response = str(response_term).strip().lower()
    # Keep "fd" as an alias for finite-difference Jacobian response.
    # This path is numerically robust and mainly used for validation/parity checks.
    if response == "fd":
        response = "fd_jacobian"
    if response not in ("fd_jacobian", "split_orbfd"):
        raise NotImplementedError("response_term must be 'split_orbfd' or 'fd_jacobian'")
    fd_delta = float(delta_bohr)
    if fd_delta <= 0.0:
        raise ValueError("delta_bohr must be > 0")

    dg = None
    if response == "fd_jacobian":
        # FD Jacobian of SA stationarity conditions w.r.t nuclear coordinates.
        delta = float(fd_delta)

        auxbasis_spec = auxbasis
        if auxbasis_spec is None:
            auxbasis_spec = getattr(scf_out, "auxbasis", None)
        if auxbasis_spec is None:
            auxbasis_spec = "autoaux"

        # Keep MO coefficients fixed in an orthonormal AO representation: C(R) = S(R)^(-1/2) U, U constant.
        try:
            S0 = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "S"))
        except Exception:
            # Minimal scf_out objects might omit overlap; rebuild from reference basis.
            from asuka.integrals.int1e_cart import build_S_cart  # noqa: PLC0415

            S0 = build_S_cart(ao_basis_ref)
        S0_sqrt = _symm_sqrt(S0, inv=False)
        U = S0_sqrt @ C_ref  # orthonormal-AO representation

        def _fd_build_arrays(mol_disp: Molecule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if fd_integrals_builder is not None:
                B_ao, hcore_ao, S_ao = fd_integrals_builder(mol_disp)
                B_ao = _asnumpy_f64(B_ao)
                hcore_ao = _asnumpy_f64(hcore_ao)
                S_ao = _asnumpy_f64(S_ao)
                if B_ao.ndim != 3:
                    raise ValueError("fd_integrals_builder must return B_ao with shape (nao,nao,naux)")
                if hcore_ao.ndim != 2 or S_ao.ndim != 2:
                    raise ValueError("fd_integrals_builder must return hcore_ao and S_ao as 2D arrays")
                if int(hcore_ao.shape[0]) != int(hcore_ao.shape[1]) or int(S_ao.shape[0]) != int(S_ao.shape[1]):
                    raise ValueError("fd_integrals_builder returned non-square hcore_ao/S_ao")
                nao_x = int(S_ao.shape[0])
                if int(B_ao.shape[0]) != nao_x or int(B_ao.shape[1]) != nao_x or int(hcore_ao.shape[0]) != nao_x:
                    raise ValueError("fd_integrals_builder returned inconsistent nao dimensions")
                return (
                    np.asarray(B_ao, dtype=np.float64),
                    np.asarray(hcore_ao, dtype=np.float64),
                    np.asarray(S_ao, dtype=np.float64),
                )

            ctx = DFCholeskyContext.build(mol_disp, auxbasis=auxbasis_spec, threads=int(df_threads))
            coords_d, charges_d = _mol_coords_charges_bohr(mol_disp)
            int1e_d = build_int1e_cart(ctx.ao_basis, atom_coords_bohr=coords_d, atom_charges=charges_d)
            return _asnumpy_f64(ctx.B_ao), _asnumpy_f64(int1e_d.hcore), _asnumpy_f64(int1e_d.S)

        dg = np.zeros((natm, 3, int(hess_op.n_tot)), dtype=np.float64)
        for ia in atmlst_use:
            for ax in range(3):
                # Use a 4th-order central stencil:
                #   g'(0) ≈ (-g(+2h) + 8g(+h) - 8g(-h) + g(-2h)) / (12h)
                # where h = delta_bohr.
                mol_p1 = _displaced_molecule(mol, ia=int(ia), axis=int(ax), delta_bohr=+delta)
                mol_m1 = _displaced_molecule(mol, ia=int(ia), axis=int(ax), delta_bohr=-delta)
                mol_p2 = _displaced_molecule(mol, ia=int(ia), axis=int(ax), delta_bohr=+2.0 * delta)
                mol_m2 = _displaced_molecule(mol, ia=int(ia), axis=int(ax), delta_bohr=-2.0 * delta)

                def _g_at(mol_disp: Molecule) -> np.ndarray:
                    B_d, hcore_d, S_d = _fd_build_arrays(mol_disp)
                    X = _symm_sqrt(S_d, inv=True)
                    C_d = X @ U

                    mc_d = DFNewtonCASSCFAdapter(
                        df_B=B_d,
                        hcore_ao=hcore_d,
                        ncore=int(ncore),
                        ncas=int(ncas),
                        nelecas=nelecas,
                        mo_coeff=np.asarray(C_d, dtype=np.float64),
                        fcisolver=fcisolver_use,
                        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
                        frozen=getattr(casscf, "frozen", None),
                        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
                        extrasym=getattr(casscf, "extrasym", None),
                    )
                    eris_d = mc_d.ao2mo(C_d)
                    g_d, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                        mc_d,
                        C_d,
                        ci_list,
                        eris_d,
                        verbose=0,
                        weights=weights,
                        implementation="internal",
                    )
                    g_d = np.asarray(g_d, dtype=np.float64).ravel()
                    if int(g_d.size) != int(hess_op.n_tot):
                        raise RuntimeError("gen_g_hop packed size mismatch during FD response build")
                    if bool(hess_op.is_sa) and hess_op.ci_ref_list is not None:
                        g_orb = g_d[: int(hess_op.n_orb)]
                        g_ci = hess_op.ci_unflatten(g_d[int(hess_op.n_orb) :])
                        g_ci = _project_sa_ci_components(hess_op.ci_ref_list, g_ci, gram_inv=hess_op.sa_gram_inv)
                        g_d = np.concatenate([g_orb, np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in g_ci])])
                    return np.asarray(g_d, dtype=np.float64).ravel()

                g_p1 = _g_at(mol_p1)
                g_m1 = _g_at(mol_m1)
                g_p2 = _g_at(mol_p2)
                g_m2 = _g_at(mol_m2)

                dg[int(ia), int(ax)] = (-g_p2 + 8.0 * g_p1 - 8.0 * g_m1 + g_m2) / (12.0 * delta)

    # Output tensor
    nac = np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    def _unpack_state(state: tuple[int, int]) -> tuple[int, int]:
        ket, bra = state
        ket = int(ket)
        bra = int(bra)
        if ket < 0 or bra < 0 or ket >= nroots or bra >= nroots:
            raise ValueError("state indices out of range")
        return ket, bra

    pair_list: list[tuple[int, int]]
    if pairs is None:
        pair_list = [(ket, bra) for ket in range(nroots) for bra in range(nroots) if ket != bra]
    else:
        pair_list = [(int(ket), int(bra)) for (ket, bra) in pairs if int(ket) != int(bra)]

    # Cache AO mappings used by the CSF term.
    shell_atom_ref = shell_to_atom_map(ao_basis_ref, atom_coords_bohr=coords)

    dm1_sa = None
    dm2_sa = None
    if response == "split_orbfd":
        dm1_sa, dm2_sa = make_state_averaged_rdms(
            fcisolver_use,
            ci_list,
            weights,
            ncas=int(ncas),
            nelecas=nelecas,
        )

    for ket, bra in pair_list:
        ket, bra = _unpack_state((ket, bra))
        if ket == bra:
            continue

        ediff = float(e_states[bra] - e_states[ket])

        # Transition densities for pair (bra,ket)
        trans_rdm12 = _base_fcisolver_method(fcisolver_use, "trans_rdm12")
        dm1_t, dm2_t = trans_rdm12(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
        dm1_t = 0.5 * (np.asarray(dm1_t, dtype=np.float64) + np.asarray(dm1_t, dtype=np.float64).T)
        dm2_t = 0.5 * (
            np.asarray(dm2_t, dtype=np.float64) + np.asarray(dm2_t, dtype=np.float64).transpose(1, 0, 3, 2)
        )

        # Hamiltonian response term (<bra|dH/dR|ket> without nuclear term)
        ham = _grad_elec_active_df(
            scf_out=scf_out,
            mo_coeff=C_ref,
            dm1_act=dm1_t,
            dm2_act=dm2_t,
            ncore=int(ncore),
            ncas=int(ncas),
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            df_grad_ctx=df_grad_ctx,
            fd_delta_bohr=float(fd_delta),
        )

        # CSF / AO-overlap term (numerator form).
        if not bool(use_etfs):
            trans_rdm1 = _base_fcisolver_method(fcisolver_use, "trans_rdm1")
            dm1 = trans_rdm1(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
            castm1 = np.asarray(dm1, dtype=np.float64).T - np.asarray(dm1, dtype=np.float64)
            mo_cas = C_ref[:, ncore:nocc]
            tm1 = mo_cas @ castm1 @ mo_cas.T
            nac_csf = 0.5 * contract_dS_ip_cart(
                ao_basis_ref,
                atom_coords_bohr=coords,
                M=tm1,
                shell_atom=shell_atom_ref,
            )
            ham = np.asarray(ham, dtype=np.float64) + np.asarray(nac_csf * ediff, dtype=np.float64)

        # Pair-specific Z-vector RHS in SA parameter space.
        fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=dm1_t, dm2=dm2_t)
        mc_trans = DFNewtonCASSCFAdapter(
            df_B=B_ref,
            hcore_ao=hcore_ref,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            mo_coeff=C_ref,
            fcisolver=fcisolver_fixed,
            frozen=getattr(casscf, "frozen", None),
            internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
            extrasym=getattr(casscf, "extrasym", None),
        )
        # Match PySCF NAC RHS construction: remove core-orbital contributions from the Newton operator.
        eris_act = _eris_patch_active(eris_sa, mo_coeff=C_ref, hcore_ao=hcore_ref, ncore=int(ncore))

        g_ket, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
            mc_trans,
            C_ref,
            ci_list[ket],
            eris_act,
            verbose=0,
            implementation="internal",
        )
        g_ket = np.asarray(g_ket, dtype=np.float64).ravel()
        g_bra, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
            mc_trans,
            C_ref,
            ci_list[bra],
            eris_act,
            verbose=0,
            implementation="internal",
        )
        g_bra = np.asarray(g_bra, dtype=np.float64).ravel()

        n_orb = int(hess_op.n_orb)
        g_orb = g_ket[:n_orb]

        g_ci_bra = 0.5 * g_ket[n_orb:].copy()
        g_ci_ket = 0.5 * g_bra[n_orb:].copy()

        ndet_ket = int(np.asarray(ci_list[ket]).size)
        ndet_bra = int(np.asarray(ci_list[bra]).size)
        if ndet_ket == ndet_bra:
            ket2bra = float(np.dot(np.asarray(ci_list[bra], dtype=np.float64).ravel(), g_ci_ket))
            bra2ket = float(np.dot(np.asarray(ci_list[ket], dtype=np.float64).ravel(), g_ci_bra))
            g_ci_ket = g_ci_ket - ket2bra * np.asarray(ci_list[bra], dtype=np.float64).ravel()
            g_ci_bra = g_ci_bra - bra2ket * np.asarray(ci_list[ket], dtype=np.float64).ravel()

        rhs_ci_list: list[np.ndarray] = []
        for r in range(nroots):
            rhs_ci_list.append(np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()))
        rhs_ci_list[ket] = g_ci_ket[:ndet_ket]
        rhs_ci_list[bra] = g_ci_bra[:ndet_bra]

        z = solve_mcscf_zvector(
            mc_sa,
            rhs_orb=np.asarray(g_orb, dtype=np.float64),
            rhs_ci=rhs_ci_list,
            hessian_op=hess_op,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
        )
        Lvec = np.asarray(z.z_packed, dtype=np.float64).ravel()
        if int(Lvec.size) != int(hess_op.n_tot):
            raise RuntimeError("unexpected Z-vector packed length")

        resp_full = np.zeros((natm, 3), dtype=np.float64)
        if response == "fd_jacobian":
            if dg is None:  # pragma: no cover
                raise RuntimeError("internal error: dg is missing for fd_jacobian response")
            for ia in atmlst_use:
                for ax in range(3):
                    resp_full[int(ia), int(ax)] = float(np.dot(Lvec, dg[int(ia), int(ax)]))
        else:
            # PySCF-style split response: Lci_dot_dgci + Lorb_dot_dgorb.
            n_orb = int(hess_op.n_orb)
            Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])
            Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
            if not isinstance(Lci_list, list) or len(Lci_list) != int(nroots):
                raise RuntimeError("unexpected CI unpack structure in Z-vector solution")

            # CI response: build (weighted) transition RDMs between Lci[root] and ci[root], then reuse the DF gradient.
            trans_rdm12 = _base_fcisolver_method(fcisolver_use, "trans_rdm12")
            dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
            dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64).ravel()
            for r in range(int(nroots)):
                wr = float(w_arr[r])
                if abs(wr) < 1e-14:
                    continue
                dm1_r, dm2_r = trans_rdm12(
                    fcisolver_use,
                    np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                    np.asarray(ci_list[r], dtype=np.float64).ravel(),
                    int(ncas),
                    nelecas,
                )
                dm1_r = np.asarray(dm1_r, dtype=np.float64)
                dm2_r = np.asarray(dm2_r, dtype=np.float64)
                dm1_lci += wr * (dm1_r + dm1_r.T)
                dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

            de_lci = _grad_elec_active_df(
                scf_out=scf_out,
                mo_coeff=C_ref,
                dm1_act=dm1_lci,
                dm2_act=dm2_lci,
                ncore=int(ncore),
                ncas=int(ncas),
                df_backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                df_grad_ctx=df_grad_ctx,
                fd_delta_bohr=float(fd_delta),
            )

            if dm1_sa is None or dm2_sa is None:  # pragma: no cover
                raise RuntimeError("internal error: missing SA RDMs for split_orbfd response")

            # Orbital response: analytic DF analogue of PySCF's Lorb_dot_dgorb_dx.
            # This avoids the finite-difference directional derivative previously used here.
            de_lorb = _Lorb_dot_dgorb_dx_df(
                scf_out=scf_out,
                mo_coeff=C_ref,
                dm1_act=dm1_sa,
                dm2_act=dm2_sa,
                Lorb=np.asarray(Lorb_mat, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                eris=eris_sa,
                df_backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                df_grad_ctx=df_grad_ctx,
                fd_delta_bohr=float(fd_delta),
            )

            resp_full = np.asarray(de_lci, dtype=np.float64) + np.asarray(de_lorb, dtype=np.float64)

        nac_num = (
            np.asarray(ham, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
            + np.asarray(resp_full, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
        )

        if not bool(mult_ediff):
            if abs(ediff) < 1e-12:
                raise ZeroDivisionError("E_bra - E_ket is too small; use mult_ediff=True for numerator mode")
            nac_num = nac_num / ediff

        nac[bra, ket] = np.asarray(nac_num, dtype=np.float64)

    return nac


__all__ = ["sacasscf_nonadiabatic_couplings_df"]
