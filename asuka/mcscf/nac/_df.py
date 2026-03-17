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
import copy
import time
from typing import Any, Callable, Literal, Sequence
import os
import warnings

import numpy as np

from asuka.chem.periodic_table import atomic_number
from asuka.hf import df_jk as _df_jk
from asuka.hf import df_scf as _df_scf
from asuka.integrals.df_context import DFCholeskyContext
from asuka.integrals.df_grad_context import DFGradContractionContext
from asuka.integrals.grad import (
    compute_df_gradient_contributions_analytic_packed_bases,
    compute_df_gradient_contributions_analytic_sph,
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
    _barl_coulomb_add_inplace,
    _barl_coulomb_add_inplace_qp,
    _df_B_dims,
    _pack_qmn_block_to_qp,
    _resolve_barl_hybrid,
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
from asuka.mcscf.zvector import (
    build_mcscf_hessian_operator,
    solve_mcscf_zvector,
    solve_mcscf_zvector_batch,
    _project_sa_ci_components,
)
from asuka.solver import GUGAFCISolver


_LAST_NACV_TIMING: dict[str, Any] | None = None


def _set_last_nacv_timing(payload: dict[str, Any] | None) -> None:
    global _LAST_NACV_TIMING
    if payload is None:
        _LAST_NACV_TIMING = None
    else:
        _LAST_NACV_TIMING = copy.deepcopy(payload)


def get_last_nacv_timing(*, clear: bool = False) -> dict[str, Any] | None:
    """Return a deep-copied timing payload from the latest DF NAC call."""
    global _LAST_NACV_TIMING
    out = copy.deepcopy(_LAST_NACV_TIMING)
    if clear:
        _LAST_NACV_TIMING = None
    return out


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(str(name))
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


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
    xp, _ = _get_xp_arrays(mo_coeff, hcore_ao, getattr(eris, "vhf_c", None))
    mo = xp.asarray(mo_coeff, dtype=xp.float64)
    moH = mo.T
    vnocore = xp.asarray(getattr(eris, "vhf_c"), dtype=xp.float64).copy()
    vnocore[:, :ncore] = -moH @ xp.asarray(hcore_ao, dtype=xp.float64) @ mo[:, :ncore]
    return DFNewtonERIs(
        ppaa=eris.ppaa,
        papa=eris.papa,
        vhf_c=vnocore,
        j_pc=eris.j_pc,
        k_pc=eris.k_pc,
        L_pu=getattr(eris, "L_pu", None),
        L_pi=getattr(eris, "L_pi", None),
        L_uv=getattr(eris, "L_uv", None),
        L_pq=getattr(eris, "L_pq", None),
        eri_provider=getattr(eris, "eri_provider", None),
        mo_coeff=getattr(eris, "mo_coeff", None),
        C_act=getattr(eris, "C_act", None),
    )


def _mol_coords_charges_bohr(mol: Any) -> tuple[np.ndarray, np.ndarray]:
    mol0 = mol
    coords = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("mol.coords_bohr must have shape (natm,3)")
    charges = np.asarray([float(atomic_number(sym)) for sym in getattr(mol0, "elements")], dtype=np.float64)
    if charges.shape != (int(coords.shape[0]),):
        raise ValueError("invalid mol.elements for charge construction")
    return coords, charges


def _atoms_bohr_from_mol_like(mol: Any) -> list[tuple[str, np.ndarray]]:
    atoms_bohr = getattr(mol, "atoms_bohr", None)
    if atoms_bohr is not None:
        out: list[tuple[str, np.ndarray]] = []
        for sym, xyz in atoms_bohr:
            out.append((str(sym), np.asarray(xyz, dtype=np.float64).reshape((3,))))
        return out

    natm = getattr(mol, "natm", None)
    atom_symbol = getattr(mol, "atom_symbol", None)
    atom_coord = getattr(mol, "atom_coord", None)
    if natm is None or not callable(atom_symbol) or not callable(atom_coord):
        raise TypeError("mol must provide atoms_bohr or natm/atom_symbol(i)/atom_coord(i)")

    out: list[tuple[str, np.ndarray]] = []
    for i in range(int(natm)):
        out.append((str(atom_symbol(int(i))), np.asarray(atom_coord(int(i)), dtype=np.float64).reshape((3,))))
    return out


def _displaced_molecule(mol: Any, *, ia: int, axis: int, delta_bohr: float) -> Any:
    ia = int(ia)
    axis = int(axis)
    delta_bohr = float(delta_bohr)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0,1,2")
    if abs(delta_bohr) <= 0.0:
        raise ValueError("delta_bohr must be non-zero")

    atoms = []
    for sym, xyz in _atoms_bohr_from_mol_like(mol):
        atoms.append((str(sym), np.asarray(xyz, dtype=np.float64).copy()))
    atoms[ia][1][axis] += delta_bohr

    # ASUKA Molecule (and compatible containers) path.
    from_atoms = getattr(type(mol), "from_atoms", None)
    if callable(from_atoms):
        return from_atoms(
            atoms,
            unit="Bohr",
            charge=int(getattr(mol, "charge", 0)),
            spin=int(getattr(mol, "spin", 0)),
            basis=getattr(mol, "basis", None),
            cart=bool(getattr(mol, "cart", True)),
        )

    # PySCF-like fallback: copy and set geometry in Bohr.
    mol_copy = getattr(mol, "copy", None)
    set_geom = getattr(mol, "set_geom_", None)
    if callable(mol_copy) and callable(set_geom):
        mol2 = mol_copy()
        mol2.set_geom_([(sym, xyz.tolist()) for sym, xyz in atoms], unit="Bohr", inplace=True)
        return mol2

    raise TypeError("unable to build displaced molecule from the provided mol object")


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
        use_cocc_k = (xp is not np) and _bool_env("ASUKA_MCSCF_DF_K_COCC", True)
        if use_cocc_k:
            try:
                q_block = int(os.environ.get("ASUKA_DF_JK_K_QBLOCK", "128"))
            except Exception:
                q_block = 128
            _is_qp, _nao_i, naux, _ntri = _df_B_dims(B_ao, nao=int(nao), where="_core_energy_weighted_density")
            q_block = max(1, min(int(naux), int(q_block)))

            Jc, _ = _df_scf._df_JK(B_ao, D_core_ao, want_J=True, want_K=False)  # noqa: SLF001
            occ_core = xp.full((int(ncore),), 2.0, dtype=xp.float64)
            Kc = _df_jk.df_K_from_BmnQ_Cocc(B_ao, mo_core, occ_core, q_block=int(q_block))
        else:
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
    out: Any | None = None,
    symmetrize: bool = True,
    work_dtype: Any | None = None,
    out_dtype: Any | None = None,
    qblock: int | None = None,
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

    _wd = xp.float64 if work_dtype is None else work_dtype
    _od = xp.float64 if out_dtype is None else out_dtype
    C = xp.asarray(C, dtype=_wd)
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    D_core_ao = 2.0 * (C_core @ C_core.T) if ncore else xp.zeros((nao, nao), dtype=_wd)
    dm1 = xp.asarray(dm1_act, dtype=_wd)
    D_act_ao = C_act @ dm1 @ C_act.T

    # Weighted density and net offset (both nao×nao — negligible VRAM).
    D_w = D_act_ao + 0.5 * D_core_ao
    D_ah = D_w - D_core_ao  # = D_act - 0.5*D_core

    B_ao = xp.asarray(B_ao, dtype=_wd)
    if int(getattr(B_ao, "ndim", 0)) == 2:
        # Packed Qp path: keep bar_L in (naux,ntri).
        is_qp, _nao_i, naux, ntri = _df_B_dims(B_ao, nao=int(nao), where="_build_bar_L_net_active_df")
        if not bool(is_qp):  # pragma: no cover
            raise RuntimeError("internal error: expected packed df_B")
        from asuka.integrals.tri_packed import pack_tril, tri_weights  # noqa: PLC0415
        from asuka.integrals.df_packed_s2 import (  # noqa: PLC0415
            apply_Qp_to_C_block,
            fused_qp_exchange_sym_f64,
            fused_qp_l_act_f64,
        )

        w = tri_weights(xp, int(nao), dtype=_wd)
        D_core_p = pack_tril(xp, D_core_ao)
        D_w_p = pack_tril(xp, D_w)
        D_ah_p = pack_tril(xp, D_ah)

        if rho_core is None:
            rho = B_ao @ (w * D_core_p)  # (naux,)
        else:
            rho = xp.asarray(rho_core, dtype=_wd)
        sigma_w = B_ao @ (w * D_w_p)  # (naux,)

        if out is None:
            bar_net = xp.zeros((naux, ntri), dtype=_od)
        else:
            bar_net = xp.asarray(out, dtype=_od)
            if tuple(map(int, bar_net.shape)) != (naux, ntri):
                raise ValueError("out shape mismatch")

        # --- Net Coulomb: sigma_w * D_core + rho * D_ah ---
        _log_vram("  net_active_fused_qp: before net Coulomb")
        _barl_coulomb_add_inplace_qp(
            bar_net,
            a_Q=sigma_w,
            M1_p=D_core_p,
            b_Q=rho,
            M2_p=D_ah_p,
            xp=xp,
        )
        _log_vram("  net_active_fused_qp: after net Coulomb")

        # Tier B-2: fused exchange kernel — no (q,nao,nao) intermediate.
        # Computes -0.5 * (D_core @ B_q @ D_ah + D_ah @ B_q @ D_core) in packed Qp format.
        if ncore:
            B_ao_f64 = xp.asarray(B_ao, dtype=xp.float64)
            D_core_f64 = xp.asarray(D_core_ao, dtype=xp.float64)
            D_ah_f64 = xp.asarray(D_ah, dtype=xp.float64)
            _exch_f64 = xp.zeros((naux, ntri), dtype=xp.float64)
            _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
            for _q0 in range(0, naux, _chunk):
                _q1 = min(_q0 + _chunk, naux)
                _q = int(_q1 - _q0)
                if _q <= 0:
                    continue
                fused_qp_exchange_sym_f64(
                    B_ao_f64, D_core_f64, D_ah_f64, _exch_f64,
                    nao=int(nao), q0=int(_q0), q_count=int(_q), alpha=-0.5,
                )
            bar_net += _exch_f64.astype(_od, copy=False)
            del _exch_f64
        _log_vram("  net_active_fused_qp: after net exchange")

        # --- Active 2-RDM term (identical to _build_bar_L_casscf_df) ---
        dm2_arr = xp.asarray(dm2_act, dtype=_wd)
        if L_act is None:
            # Tier B-1: fused kernel — no (q,nao,nao) or (q,nao,ncas) intermediate.
            B_ao_f64 = xp.asarray(B_ao, dtype=xp.float64)
            L_act_val = xp.empty((ncas, ncas, naux), dtype=_wd)
            _chunkL = int(max(1, qblock if qblock is not None else (naux // 4)))
            for _q0 in range(0, naux, _chunkL):
                _q1 = min(_q0 + _chunkL, naux)
                _q = int(_q1 - _q0)
                if _q <= 0:
                    continue
                _l = fused_qp_l_act_f64(B_ao_f64, C_act, nao=int(nao), q0=int(_q0), q_count=int(_q))  # (q,ncas,ncas)
                L_act_val[:, :, _q0:_q1] = _l.transpose(1, 2, 0)
                del _l
        else:
            L_act_val = xp.asarray(L_act, dtype=_wd)

        L2 = L_act_val.reshape(ncas * ncas, naux)
        dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
        dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
        M = dm2_flat @ L2
        M_uvQ = M.reshape(ncas, ncas, naux)

        tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)
        _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
        for _q0 in range(0, naux, _chunk):
            _q1 = min(_q0 + _chunk, naux)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_act, optimize=True)
            _bp = _pack_qmn_block_to_qp(xp, _blk.astype(xp.float64, copy=False), nao=int(nao))
            bar_net[_q0:_q1] += _bp.astype(_od, copy=False)
            del _blk, _bp
        del tmp
        _log_vram("  net_active_fused_qp: after bar_act")

        return xp.asarray(bar_net, dtype=_od), xp.asarray(D_act_ao, dtype=xp.float64)

    # Full mnQ path.
    nao_b, _, naux = map(int, B_ao.shape)
    B2 = B_ao.reshape(nao * nao, naux)
    if rho_core is None:
        rho = B2.T @ D_core_ao.reshape(nao * nao)  # (naux,)
    else:
        rho = xp.asarray(rho_core, dtype=_wd)
    sigma_w = B2.T @ D_w.reshape(nao * nao)  # (naux,)

    if out is None:
        bar_net = xp.zeros((naux, nao, nao), dtype=_od)
    else:
        bar_net = xp.asarray(out, dtype=_od)
        if bar_net.shape != (naux, nao, nao):
            raise ValueError("out shape mismatch")

    # --- Net Coulomb: sigma_w * D_core + rho * D_ah ---
    _log_vram("  net_active_fused: before net Coulomb")
    _barl_coulomb_add_inplace(
        bar_net,
        a_Q=sigma_w,
        M1_mn=D_core_ao,
        b_Q=rho,
        M2_mn=D_ah,
        xp=xp,
    )
    _log_vram("  net_active_fused: after net Coulomb")

    # --- Net Exchange (chunked): -0.5*(D_core @ BQ @ D_ah + D_w @ BQ @ D_core) ---
    if ncore:
        BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao) view
        _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
        for _q0 in range(0, naux, _chunk):
            _q1 = min(_q0 + _chunk, naux)
            _bq = BQ[_q0:_q1]
            _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_ah)
            _t += xp.matmul(xp.matmul(D_w[None, :, :], _bq), D_core_ao)
            _t *= -0.5
            bar_net[_q0:_q1] += _t.astype(_od, copy=False)
            del _t
    _log_vram("  net_active_fused: after net exchange")

    # --- Active 2-RDM term (identical to _build_bar_L_casscf_df) ---
    dm2_arr = xp.asarray(dm2_act, dtype=_wd)
    if L_act is None:
        X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
        L_act_val = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)
    else:
        L_act_val = xp.asarray(L_act, dtype=_wd)

    L2 = L_act_val.reshape(ncas * ncas, naux)
    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)
    # Accumulate bar_act into bar_net in chunks to avoid sizeof_B temporary.
    _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_act, optimize=True)
        bar_net[_q0:_q1] += _blk.astype(_od, copy=False)
        del _blk
    del tmp
    _log_vram("  net_active_fused: after bar_act")

    # --- Symmetrize in-place ---
    if bool(symmetrize):
        _symmetrize_bar_L_inplace(bar_net, xp)
        _log_vram("  net_active_fused: after sym")
    return xp.asarray(bar_net, dtype=_od), xp.asarray(D_act_ao, dtype=xp.float64)


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
    out: Any | None = None,
    symmetrize: bool = True,
    work_dtype: Any | None = None,
    out_dtype: Any | None = None,
    qblock: int | None = None,
    act_resp_scale: float | None = None,
    dml_sym_mode: str | None = None,
    C_L_override: Any | None = None,
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

    _wd = xp.float64 if work_dtype is None else work_dtype
    _od = xp.float64 if out_dtype is None else out_dtype
    B_ao = xp.asarray(B_ao, dtype=_wd)
    is_qp, _nao_i, naux, ntri = _df_B_dims(B_ao, nao=int(nao), where="_build_bar_L_lorb_df")
    C = xp.asarray(C, dtype=_wd)
    L = xp.asarray(Lorb, dtype=_wd)
    dm1 = xp.asarray(dm1_act, dtype=_wd)
    dm2 = xp.asarray(dm2_act, dtype=_wd)
    if act_resp_scale is None:
        try:
            _act_resp_scale = float(os.environ.get("ASUKA_CASPT2_LORB_ACTIVE_RESPONSE_SCALE", "1.0"))
        except Exception:
            _act_resp_scale = 1.0
    else:
        _act_resp_scale = float(act_resp_scale)
    if abs(float(_act_resp_scale) - 1.0) > 1e-12:
        dm1 = float(_act_resp_scale) * dm1
        dm2 = float(_act_resp_scale) * dm2
    nmo = int(C.shape[1])

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    C_L = C @ L
    C_L_core = C_L[:, :ncore]
    C_L_act = C_L[:, ncore:nocc]
    C_L_act_dm2 = C_L_act
    if C_L_override is not None:
        C_L_alt = xp.asarray(C_L_override, dtype=_wd)
        if tuple(map(int, C_L_alt.shape)) != tuple(map(int, C.shape)):
            raise ValueError(
                "C_L_override shape mismatch: "
                f"expected {tuple(map(int, C.shape))}, got {tuple(map(int, C_L_alt.shape))}"
            )
        C_L_act_dm2 = C_L_alt[:, ncore:nocc]

    if dml_sym_mode is None:
        _dml_sym_mode = str(os.environ.get("ASUKA_CASPT2_LORB_DML_SYM_MODE", "full")).strip().lower()
    else:
        _dml_sym_mode = str(dml_sym_mode).strip().lower()
    if _dml_sym_mode not in {
        "full",
        "core_raw",
        "act_raw",
        "raw",
        "core_asym",
        "act_asym",
        "asym",
        "core_oitd",
        "act_oitd",
        "molcas_oitd",
    }:
        _dml_sym_mode = "full"

    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core_raw = 2.0 * (C_L_core @ C_core.T)
        D_L_core_sym = D_L_core_raw + D_L_core_raw.T
        D_L_core_asym = D_L_core_raw - D_L_core_raw.T
        _need_core_oitd = _dml_sym_mode in {"core_oitd", "molcas_oitd"}
        if _need_core_oitd:
            d_core_mo = xp.zeros((nmo, nmo), dtype=_wd)
            d_core_mo[:ncore, :ncore] = 2.0 * xp.eye(int(ncore), dtype=_wd)
            dL_core_mo = d_core_mo @ L.T - L.T @ d_core_mo
            D_L_core_oitd = C @ dL_core_mo @ C.T
        else:
            D_L_core_oitd = xp.zeros((nao, nao), dtype=_wd)
        if _dml_sym_mode in {"core_raw", "raw"}:
            D_L_core = D_L_core_raw
        elif _dml_sym_mode in {"core_asym", "asym"}:
            D_L_core = D_L_core_asym
        elif _dml_sym_mode in {"core_oitd", "molcas_oitd"}:
            D_L_core = D_L_core_oitd
        else:
            D_L_core = D_L_core_sym
    else:
        D_core = xp.zeros((nao, nao), dtype=_wd)
        D_L_core = xp.zeros((nao, nao), dtype=_wd)

    D_act = C_act @ dm1 @ C_act.T
    D_L_act_raw = C_L_act @ dm1 @ C_act.T
    D_L_act_sym = D_L_act_raw + D_L_act_raw.T
    D_L_act_asym = D_L_act_raw - D_L_act_raw.T
    _need_act_oitd = _dml_sym_mode in {"act_oitd", "molcas_oitd"}
    if _need_act_oitd:
        d_act_mo = xp.zeros((nmo, nmo), dtype=_wd)
        d_act_mo[ncore:nocc, ncore:nocc] = dm1
        dL_act_mo = d_act_mo @ L.T - L.T @ d_act_mo
        D_L_act_oitd = C @ dL_act_mo @ C.T
    else:
        D_L_act_oitd = xp.zeros((nao, nao), dtype=_wd)
    if _dml_sym_mode in {"act_raw", "raw"}:
        D_L_act = D_L_act_raw
    elif _dml_sym_mode in {"act_asym", "asym"}:
        D_L_act = D_L_act_asym
    elif _dml_sym_mode in {"act_oitd", "molcas_oitd"}:
        D_L_act = D_L_act_oitd
    else:
        D_L_act = D_L_act_sym
    D_L = D_L_core + D_L_act

    # Mean-field bar_L (two _build_bar_L_df_cross calls, no _df_JK needed).
    D_w = D_act + 0.5 * D_core
    D_wL = D_L_act + 0.5 * D_L_core
    _log_vram("  lorb: before cross1")
    if out is None:
        bar_mean = _build_bar_L_df_cross(
            B_ao,
            D_left=D_wL,
            D_right=D_core,
            coeff_J=1.0,
            coeff_K=-0.5,
            symmetrize=False,
            work_dtype=_wd,
            out_dtype=_od,
            qblock=qblock,
        )
    else:
        bar_mean = xp.asarray(out, dtype=_od)
        if bool(is_qp):
            if tuple(map(int, bar_mean.shape)) != (int(naux), int(ntri)):
                raise ValueError("out shape mismatch")
        else:
            if tuple(map(int, bar_mean.shape)) != (int(naux), int(nao), int(nao)):
                raise ValueError("out shape mismatch")
        _build_bar_L_df_cross(
            B_ao,
            D_left=D_wL,
            D_right=D_core,
            coeff_J=1.0,
            coeff_K=-0.5,
            out=bar_mean,
            symmetrize=False,
            work_dtype=_wd,
            out_dtype=_od,
            qblock=qblock,
        )
    _log_vram("  lorb: after cross1")
    _flush_gpu_pool()
    _log_vram("  lorb: after cross1 flush")
    _build_bar_L_df_cross(
        B_ao,
        D_left=D_w,
        D_right=D_L_core,
        coeff_J=1.0,
        coeff_K=-0.5,
        out=bar_mean,
        symmetrize=False,
        work_dtype=_wd,
        out_dtype=_od,
        qblock=qblock,
    )
    _log_vram("  lorb: after cross2")

    # Active-active DF term: linearize bar_L_casscf active block w.r.t C_act along C_L_act.
    dm2_flat = dm2.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)

    if not bool(is_qp):
        X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
        L_act_mo = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)
        L2 = L_act_mo.reshape(ncas * ncas, -1)
        del L_act_mo
        M_mat = dm2_flat @ L2
        del L2
        M_uvQ = M_mat.reshape(ncas, ncas, -1)
        del M_mat

        tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)
        tmp_L = xp.einsum("mu,uvQ->mvQ", C_L_act_dm2, M_uvQ, optimize=True)
        del M_uvQ

        X_L = xp.einsum("mnQ,nv->mvQ", B_ao, C_L_act_dm2, optimize=True)
        dL_act = (
            xp.einsum("mu,mvQ->uvQ", C_L_act_dm2, X, optimize=True)
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
        _chunk = int(max(1, qblock if qblock is not None else (_naux // 4)))
        for _q0 in range(0, _naux, _chunk):
            _q1 = min(_q0 + _chunk, _naux)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp_L[:, :, _q0:_q1], C_act, optimize=True)
            bar_mean[_q0:_q1] += _blk.astype(_od, copy=False)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_L_act_dm2, optimize=True)
            bar_mean[_q0:_q1] += _blk.astype(_od, copy=False)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp_M[:, :, _q0:_q1], C_act, optimize=True)
            bar_mean[_q0:_q1] += _blk.astype(_od, copy=False)
            del _blk
        del tmp_L, tmp, tmp_M
    else:
        # Packed-Qp path: build the same intermediates using aux-block unpacking and
        # accumulate bar_act into (naux,ntri) storage.
        from asuka.integrals.df_packed_s2 import unpack_Qp_to_Qmn_block  # noqa: PLC0415

        _chunk = int(max(1, qblock if qblock is not None else (int(naux) // 4)))
        _chunk = max(1, min(int(naux), int(_chunk)))

        # Build X=B*C_act and X_L=B*C_L_act, plus L_act= C_act^T X, in aux blocks.
        X = xp.empty((nao, ncas, naux), dtype=_wd)
        X_L = xp.empty((nao, ncas, naux), dtype=_wd)
        L_act_mo = xp.empty((ncas, ncas, naux), dtype=_wd)
        for _q0 in range(0, int(naux), int(_chunk)):
            _q1 = min(int(naux), int(_q0) + int(_chunk))
            _q = int(_q1 - _q0)
            if _q <= 0:
                continue
            BQc = unpack_Qp_to_Qmn_block(B_ao, nao=int(nao), q0=int(_q0), q_count=int(_q))  # (q,nao,nao)
            X_blk = xp.matmul(BQc, C_act)  # (q,nao,ncas)
            X_L_blk = xp.matmul(BQc, C_L_act_dm2)  # (q,nao,ncas)
            L_blk = xp.matmul(C_act.T[None, :, :], X_blk)  # (q,ncas,ncas)
            X[:, :, int(_q0) : int(_q1)] = X_blk.transpose(1, 2, 0)
            X_L[:, :, int(_q0) : int(_q1)] = X_L_blk.transpose(1, 2, 0)
            L_act_mo[:, :, int(_q0) : int(_q1)] = L_blk.transpose(1, 2, 0)
            del BQc, X_blk, X_L_blk, L_blk

        L2 = L_act_mo.reshape(ncas * ncas, int(naux))
        M_mat = dm2_flat @ L2
        M_uvQ = M_mat.reshape(ncas, ncas, int(naux))
        del L2, M_mat

        # tmp = C_act @ M, tmp_L = C_L_act @ M (batch matmul over Q)
        M_batch = M_uvQ.transpose(2, 0, 1)  # (naux,ncas,ncas)
        tmp = xp.matmul(C_act[None, :, :], M_batch).transpose(1, 2, 0)  # (nao,ncas,naux)
        tmp_L = xp.matmul(C_L_act_dm2[None, :, :], M_batch).transpose(1, 2, 0)
        del M_uvQ, M_batch

        # dL_act = C_L_act^T X + C_act^T X_L (batch matmul over Q)
        X_batch = X.transpose(2, 0, 1)  # (naux,nao,ncas)
        X_L_batch = X_L.transpose(2, 0, 1)
        dL1 = xp.matmul(C_L_act_dm2.T[None, :, :], X_batch)  # (naux,ncas,ncas)
        dL2 = xp.matmul(C_act.T[None, :, :], X_L_batch)
        dL_act = (dL1 + dL2).transpose(1, 2, 0)  # (ncas,ncas,naux)
        del X_batch, X_L_batch, dL1, dL2

        dL2_flat = dL_act.reshape(ncas * ncas, int(naux))
        dM = dm2_flat @ dL2_flat
        dM_uvQ = dM.reshape(ncas, ncas, int(naux))
        del dL_act, dL2_flat, dM

        dM_batch = dM_uvQ.transpose(2, 0, 1)  # (naux,ncas,ncas)
        tmp_M = xp.matmul(C_act[None, :, :], dM_batch).transpose(1, 2, 0)  # (nao,ncas,naux)
        del dM_uvQ, dM_batch

        for _q0 in range(0, int(naux), int(_chunk)):
            _q1 = min(int(naux), int(_q0) + int(_chunk))
            # Each einsum("mvQ,nv->Qmn") produces a non-symmetric per-Q block;
            # symmetrize before packing to match the Qmn path (which symmetrizes
            # the full accumulated bar_mean at the end via _symmetrize_bar_L_inplace).
            _blk = xp.einsum("mvQ,nv->Qmn", tmp_L[:, :, _q0:_q1], C_act, optimize=True)
            _blk = 0.5 * (_blk + _blk.transpose(0, 2, 1))
            bar_mean[_q0:_q1] += _pack_qmn_block_to_qp(xp, _blk.astype(xp.float64, copy=False), nao=int(nao)).astype(_od, copy=False)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_L_act_dm2, optimize=True)
            _blk = 0.5 * (_blk + _blk.transpose(0, 2, 1))
            bar_mean[_q0:_q1] += _pack_qmn_block_to_qp(xp, _blk.astype(xp.float64, copy=False), nao=int(nao)).astype(_od, copy=False)
            _blk = xp.einsum("mvQ,nv->Qmn", tmp_M[:, :, _q0:_q1], C_act, optimize=True)
            _blk = 0.5 * (_blk + _blk.transpose(0, 2, 1))
            bar_mean[_q0:_q1] += _pack_qmn_block_to_qp(xp, _blk.astype(xp.float64, copy=False), nao=int(nao)).astype(_od, copy=False)
            del _blk
        del tmp_L, tmp, tmp_M, X, X_L, L_act_mo

    if bool(symmetrize):
        if not bool(is_qp):
            _symmetrize_bar_L_inplace(bar_mean, xp)
    return xp.asarray(bar_mean, dtype=_od), xp.asarray(D_L, dtype=xp.float64)


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
    core_cache: dict[str, Any] | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return active-electron part of <dH/dR> using DF integrals (no nuclear term)."""

    xp, _is_gpu = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    _is_sph = not bool(getattr(mol, "cart", True))
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_grad_elec_active_df")
    _nao_hint = int(getattr(mo_coeff, "shape", (0, 0))[0])
    _is_qp, _nao_i, _naux_hint, _ntri_hint = _df_B_dims(B_ao, nao=int(_nao_hint), where="_grad_elec_active_df")
    _barl_policy = _resolve_barl_hybrid(
        xp=xp,
        is_gpu=bool(_is_gpu),
        naux_hint=int(_naux_hint),
    )
    if bool(_barl_policy.get("enabled", False)):
        _barl_work_dtype = _barl_policy.get("work_dtype", xp.float64)
        _barl_out_dtype = _barl_policy.get("out_dtype", xp.float64)
        _barl_qblock = int(_barl_policy.get("qblock", 0))
    else:
        _barl_work_dtype = xp.float64
        _barl_out_dtype = xp.float64
        _barl_qblock = 0
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    C = _as_xp_f64(xp, mo_coeff)
    cache = core_cache if isinstance(core_cache, dict) else None

    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao,
        h_ao,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=xp.asarray(dm1_act, dtype=xp.float64),
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
        vhf_cache_in=cache,
    )

    # ── Phase 1: Build both bar_L tensors on GPU (cheap, before any contract call) ──
    barl_L_act = None
    barl_rho_core = None
    if cache is not None:
        barl_L_act = cache.get("barl_L_act")
        barl_rho_core = cache.get("barl_rho_core")
    if barl_L_act is None or barl_rho_core is None:
        if not bool(_is_qp):
            nao_i = int(C_act.shape[0])
            naux_i = int(B_ao.shape[2])
            B2 = B_ao.reshape(nao_i * nao_i, naux_i)
            barl_rho_core = B2.T @ D_core_ao.reshape(nao_i * nao_i)
            X_act = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
            barl_L_act = xp.einsum("mu,mvQ->uvQ", C_act, X_act, optimize=True)
            del X_act
        else:
            from asuka.integrals.tri_packed import pack_tril, tri_weights  # noqa: PLC0415
            from asuka.integrals.df_packed_s2 import unpack_Qp_to_Qmn_block  # noqa: PLC0415

            nao_i = int(C_act.shape[0])
            naux_i = int(B_ao.shape[0])
            w_tri = tri_weights(xp, int(nao_i), dtype=xp.float64)
            barl_rho_core = B_ao @ (w_tri * pack_tril(xp, D_core_ao))
            qblk = int(max(1, int(_barl_qblock) if int(_barl_qblock) > 0 else max(1, int(naux_i) // 4)))
            qblk = max(1, min(int(naux_i), int(qblk)))
            barl_L_act = xp.empty((int(ncas), int(ncas), int(naux_i)), dtype=xp.float64)
            for q0 in range(0, int(naux_i), int(qblk)):
                q1 = min(int(naux_i), int(q0) + int(qblk))
                q_count = int(q1 - q0)
                if q_count <= 0:
                    continue
                bq = unpack_Qp_to_Qmn_block(B_ao, nao=int(nao_i), q0=int(q0), q_count=int(q_count))
                x_blk = xp.matmul(bq, C_act)
                l_blk = xp.matmul(C_act.T[None, :, :], x_blk)
                barl_L_act[:, :, int(q0):int(q1)] = l_blk.transpose(1, 2, 0)
                del bq, x_blk, l_blk
        if cache is not None:
            cache["barl_L_act"] = barl_L_act
            cache["barl_rho_core"] = barl_rho_core
    bar_L_ao = _build_bar_L_casscf_df(
        B_ao,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=xp.asarray(dm2_act, dtype=xp.float64),
        L_act=barl_L_act,
        rho_core=barl_rho_core,
        work_dtype=_barl_work_dtype,
        out_dtype=_barl_out_dtype,
        qblock=_barl_qblock,
    )
    D_core_only = dme_core = bar_L_core = de_h1_core = None
    if cache is not None:
        D_core_only = cache.get("D_core_only")
        dme_core = cache.get("dme_core")
        bar_L_core = cache.get("bar_L_core")
        de_h1_core = cache.get("de_h1_core")

    if D_core_only is None or dme_core is None or bar_L_core is None or de_h1_core is None:
        # Core-only RHF-like contribution (GPU).
        D_core_only, dme_core = _core_energy_weighted_density(
            mo_coeff=C,
            hcore_ao=h_ao,
            B_ao=B_ao,
            ncore=int(ncore),
        )
        bar_L_core = _build_bar_L_df_cross(
            B_ao,
            D_left=D_core_only,
            D_right=D_core_only,
            coeff_J=0.5,
            coeff_K=-0.25,
            work_dtype=_barl_work_dtype,
            out_dtype=_barl_out_dtype,
            qblock=_barl_qblock,
        )
        D_core_cpu_cache = _asnumpy_f64(D_core_only)
        if _is_sph:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1_core = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=D_core_cpu_cache,
                shell_atom=shell_atom,
            )
        else:
            de_h1_core = contract_dhcore_cart(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M=D_core_cpu_cache,
                shell_atom=shell_atom,
            )
        if cache is not None:
            cache["D_core_only"] = D_core_only
            cache["dme_core"] = dme_core
            cache["bar_L_core"] = bar_L_core
            cache["de_h1_core"] = np.asarray(de_h1_core, dtype=np.float64)
    de_h1_core = np.asarray(de_h1_core, dtype=np.float64)

    if not bool(return_terms):
        # ── Fast path: single fused contract() call (bar_L_ao - bar_L_core) ──
        bar_L_ao -= bar_L_core
        del bar_L_core
        bar_L_net = bar_L_ao
        bar_L_contract = bar_L_net
        if str(getattr(bar_L_contract, "dtype", "")) != str(np.float64):
            bar_L_contract = xp.asarray(bar_L_contract, dtype=xp.float64)
        try:
            if _is_sph:
                if df_grad_ctx is not None:
                    de_df_net = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_contract, T_c2s=None)
                else:
                    de_df_net = compute_df_gradient_contributions_analytic_sph(
                        ao_basis, aux_basis,
                        atom_coords_bohr=coords,
                        B_sph=B_ao, bar_L_sph=bar_L_contract,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend),
                        df_threads=int(df_threads),
                        profile=None,
                    )
            elif df_grad_ctx is not None:
                de_df_net = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_contract)
            else:
                de_df_net = compute_df_gradient_contributions_analytic_packed_bases(
                    ao_basis,
                    aux_basis,
                    atom_coords_bohr=coords,
                    B_ao=B_ao,
                    bar_L_ao=bar_L_contract,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend),
                    df_threads=int(df_threads),
                    profile=None,
                )
        except (NotImplementedError, RuntimeError) as e:
            if _is_sph:
                raise
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
                bar_L_ao=bar_L_contract,
                backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                delta_bohr=float(fd_delta_bohr),
                profile=None,
            )
        D_tot_cpu = _asnumpy_f64(D_tot_ao)
        if _is_sph:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415
            de_h1 = contract_dhcore_sph(
                ao_basis, atom_coords_bohr=coords, atom_charges=charges,
                M_sph=D_tot_cpu, shell_atom=shell_atom,
            )
        else:
            de_h1 = contract_dhcore_cart(
                ao_basis, atom_coords_bohr=coords, atom_charges=charges,
                M=D_tot_cpu, shell_atom=shell_atom,
            )
        dme_tot_ao = C @ (0.5 * (gfock + gfock.T)) @ C.T
        dme_act_ao = dme_tot_ao - dme_core
        if _is_sph:
            from asuka.integrals.int1e_sph import contract_dS_ip_sph  # noqa: PLC0415

            de_pulay = -2.0 * contract_dS_ip_sph(
                ao_basis,
                atom_coords_bohr=coords,
                M_sph=_asnumpy_f64(dme_act_ao),
                shell_atom=shell_atom,
            )
        else:
            de_pulay = -2.0 * contract_dS_ip_cart(
                ao_basis,
                atom_coords_bohr=coords,
                M=_asnumpy_f64(dme_act_ao),
                shell_atom=shell_atom,
            )
        _restore_pool()
        return np.asarray(de_h1 + _asnumpy_f64(de_df_net) - de_h1_core + np.asarray(de_pulay, dtype=np.float64), dtype=np.float64)

    # ── Debug path (return_terms=True): keep 2 separate contract() calls for breakdown ──
    bar_L_ao_contract = bar_L_ao if str(getattr(bar_L_ao, "dtype", "")) == str(np.float64) else xp.asarray(bar_L_ao, dtype=xp.float64)
    bar_L_core_contract = bar_L_core if str(getattr(bar_L_core, "dtype", "")) == str(np.float64) else xp.asarray(bar_L_core, dtype=xp.float64)

    def _contract_df_2e(bar_L, *, where_label):
        try:
            if _is_sph:
                if df_grad_ctx is not None:
                    return df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L, T_c2s=None)
                return compute_df_gradient_contributions_analytic_sph(
                    ao_basis, aux_basis, atom_coords_bohr=coords,
                    B_sph=B_ao, bar_L_sph=bar_L,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend), df_threads=int(df_threads), profile=None,
                )
            if df_grad_ctx is not None:
                return df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L)
            return compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords,
                B_ao=B_ao, bar_L_ao=bar_L,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend), df_threads=int(df_threads), profile=None,
            )
        except (NotImplementedError, RuntimeError) as e:
            if _is_sph:
                raise
            _warn_df_fd_fallback(where=where_label, backend=str(df_backend), delta_bohr=float(fd_delta_bohr), err=e)
            return compute_df_gradient_contributions_fd_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords,
                bar_L_ao=bar_L, backend=str(df_backend), df_config=df_config,
                df_threads=int(df_threads), delta_bohr=float(fd_delta_bohr), profile=None,
            )

    de_df = _contract_df_2e(bar_L_ao_contract, where_label="_grad_elec_active_df(df2e)")
    de_df_core = _contract_df_2e(bar_L_core_contract, where_label="_grad_elec_active_df(df2e_core_sub)")

    D_tot_cpu = _asnumpy_f64(D_tot_ao)
    if _is_sph:
        from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415
        de_h1 = contract_dhcore_sph(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M_sph=D_tot_cpu, shell_atom=shell_atom,
        )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=D_tot_cpu, shell_atom=shell_atom,
        )
    dme_tot_ao = C @ (0.5 * (gfock + gfock.T)) @ C.T
    dme_act_ao = dme_tot_ao - dme_core
    if _is_sph:
        from asuka.integrals.int1e_sph import contract_dS_ip_sph  # noqa: PLC0415

        de_pulay = -2.0 * contract_dS_ip_sph(
            ao_basis,
            atom_coords_bohr=coords,
            M_sph=_asnumpy_f64(dme_act_ao),
            shell_atom=shell_atom,
        )
    else:
        de_pulay = -2.0 * contract_dS_ip_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=_asnumpy_f64(dme_act_ao),
            shell_atom=shell_atom,
        )
    total = np.asarray(
        de_h1 + _asnumpy_f64(de_df) - de_h1_core - _asnumpy_f64(de_df_core) + np.asarray(de_pulay, dtype=np.float64),
        dtype=np.float64,
    )
    terms = {
        "dhcore": np.asarray(de_h1, dtype=np.float64),
        "df2e": np.asarray(de_df, dtype=np.float64),
        # Core-only RHF-like subtraction pieces (returned as the *contribution to total*).
        "dhcore_core_sub": np.asarray(-de_h1_core, dtype=np.float64),
        "df2e_core_sub": np.asarray(-de_df_core, dtype=np.float64),
        "dS_act_pulay": np.asarray(de_pulay, dtype=np.float64),
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

    xp, _is_gpu = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    _is_sph_c = not bool(getattr(mol, "cart", True))
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_grad_elec_casscf_df")
    _is_qp, _nao_i, _naux_hint, _ntri = _df_B_dims(B_ao, nao=int(mo_coeff.shape[0]), where="_grad_elec_active_df")
    _barl_policy = _resolve_barl_hybrid(
        xp=xp,
        is_gpu=bool(_is_gpu),
        naux_hint=int(_naux_hint),
    )
    if bool(_barl_policy.get("enabled", False)):
        _barl_work_dtype = _barl_policy.get("work_dtype", xp.float64)
        _barl_out_dtype = _barl_policy.get("out_dtype", xp.float64)
        _barl_qblock = int(_barl_policy.get("qblock", 0))
    else:
        _barl_work_dtype = xp.float64
        _barl_out_dtype = xp.float64
        _barl_qblock = 0
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
        work_dtype=_barl_work_dtype,
        out_dtype=_barl_out_dtype,
        qblock=_barl_qblock,
    )
    bar_L_contract = bar_L_ao
    if str(getattr(bar_L_contract, "dtype", "")) != str(np.float64):
        bar_L_contract = xp.asarray(bar_L_contract, dtype=xp.float64)
    try:
        if _is_sph_c:
            if df_grad_ctx is not None:
                de_df = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_contract, T_c2s=None)
            else:
                de_df = compute_df_gradient_contributions_analytic_sph(
                    ao_basis, aux_basis, atom_coords_bohr=coords,
                    B_sph=B_ao, bar_L_sph=bar_L_contract,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend), df_threads=int(df_threads), profile=None,
                )
        elif df_grad_ctx is not None:
            de_df = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_contract)
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords,
                B_ao=B_ao, bar_L_ao=bar_L_contract,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend), df_threads=int(df_threads), profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        if _is_sph_c:
            raise
        _warn_df_fd_fallback(
            where="_grad_elec_casscf_df",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis, aux_basis, atom_coords_bohr=coords,
            bar_L_ao=bar_L_contract, backend=str(df_backend), df_config=df_config,
            df_threads=int(df_threads), delta_bohr=float(fd_delta_bohr), profile=None,
        )

    # ── CPU phase: 1e contractions (device synced by contract()) ──
    if _is_sph_c:
        from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415
        de_h1 = contract_dhcore_sph(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M_sph=_asnumpy_f64(D_tot_ao), shell_atom=shell_atom,
        )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=_asnumpy_f64(D_tot_ao), shell_atom=shell_atom,
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

    xp, _is_gpu = _resolve_xp(df_backend)

    mol = getattr(scf_out, "mol")
    _is_sph_l = not bool(getattr(mol, "cart", True))
    coords, charges = _mol_coords_charges_bohr(mol)

    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    _restore_pool = _apply_df_pool_policy(B_ao, label="_Lorb_dot_dgorb_dx_df")
    _is_qp, _nao_i, _naux_hint, _ntri = _df_B_dims(B_ao, nao=int(mo_coeff.shape[0]), where="_Lorb_dot_dgorb_dx_df")
    _barl_policy = _resolve_barl_hybrid(
        xp=xp,
        is_gpu=bool(_is_gpu),
        naux_hint=int(_naux_hint),
    )
    if bool(_barl_policy.get("enabled", False)):
        _barl_work_dtype = _barl_policy.get("work_dtype", xp.float64)
        _barl_out_dtype = _barl_policy.get("out_dtype", xp.float64)
        _barl_qblock = int(_barl_policy.get("qblock", 0))
    else:
        _barl_work_dtype = xp.float64
        _barl_out_dtype = xp.float64
        _barl_qblock = 0
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
    dm1_act = 0.5 * (dm1_act + dm1_act.T)
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
    use_cocc_k = (xp is not np) and _bool_env("ASUKA_MCSCF_DF_K_COCC", True)
    if use_cocc_k:
        try:
            q_block = int(os.environ.get("ASUKA_DF_JK_K_QBLOCK", "128"))
        except Exception:
            q_block = 128
        _is_qp, _nao_i, _naux, _ntri = _df_B_dims(B_ao, nao=int(nao), where="_Lorb_dot_dgorb_dx_df")
        q_block = max(1, min(int(_naux), int(q_block)))

        if ncore:
            Jc, _ = _df_scf._df_JK(B_ao, D_core, want_J=True, want_K=False)  # noqa: SLF001
            occ_core = xp.full((int(ncore),), 2.0, dtype=xp.float64)
            Kc = _df_jk.df_K_from_BmnQ_Cocc(B_ao, C_core, occ_core, q_block=int(q_block))
        else:
            Jc = xp.zeros((nao, nao), dtype=xp.float64)
            Kc = xp.zeros((nao, nao), dtype=xp.float64)

        Ja, _ = _df_scf._df_JK(B_ao, D_act, want_J=True, want_K=False)  # noqa: SLF001
        dm1_h = np.asarray(dm1_act if xp is np else xp.asnumpy(dm1_act), dtype=np.float64)
        w_h, U_h = np.linalg.eigh(dm1_h)
        if float(np.min(w_h)) < -1e-8:
            _, Ka = _df_scf._df_JK(B_ao, D_act, want_J=False, want_K=True)  # noqa: SLF001
        else:
            w_h = np.clip(w_h, 0.0, None)
            w = xp.asarray(w_h, dtype=xp.float64)
            U = xp.asarray(U_h, dtype=xp.float64)
            C_no = C_act @ U
            Ka = _df_jk.df_K_from_BmnQ_Cocc(B_ao, C_no, w, q_block=int(q_block))
    else:
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
    # Build the bar_L tensor for the DF derivative contraction. Use the shared
    # helper that supports both full mnQ and packed Qp layouts.
    bar_L_ao, _ = _build_bar_L_lorb_df(
        B_ao,
        C,
        L,
        dm1_act,
        dm2_act,
        ncore=int(ncore),
        ncas=int(ncas),
        xp=xp,
        symmetrize=True,
        work_dtype=_barl_work_dtype,
        out_dtype=_barl_out_dtype,
        qblock=int(_barl_qblock) if int(_barl_qblock) > 0 else None,
        act_resp_scale=1.0,
        dml_sym_mode="full",
    )
    bar_L_contract = bar_L_ao
    if str(getattr(bar_L_contract, "dtype", "")) != str(np.float64):
        bar_L_contract = xp.asarray(bar_L_contract, dtype=xp.float64)

    try:
        if _is_sph_l:
            if df_grad_ctx is not None:
                de_df = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_contract, T_c2s=None)
            else:
                de_df = compute_df_gradient_contributions_analytic_sph(
                    ao_basis, aux_basis, atom_coords_bohr=coords,
                    B_sph=B_ao, bar_L_sph=bar_L_contract,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend), df_threads=int(df_threads), profile=None,
                )
        elif df_grad_ctx is not None:
            de_df = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_contract)
        else:
            de_df = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords,
                B_ao=B_ao, bar_L_ao=bar_L_contract,
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend), df_threads=int(df_threads), profile=None,
            )
    except (NotImplementedError, RuntimeError) as e:
        if _is_sph_l:
            raise
        _warn_df_fd_fallback(
            where="_Lorb_dot_dgorb_dx_df",
            backend=str(df_backend),
            delta_bohr=float(fd_delta_bohr),
            err=e,
        )
        de_df = compute_df_gradient_contributions_fd_packed_bases(
            ao_basis, aux_basis, atom_coords_bohr=coords,
            bar_L_ao=bar_L_contract, backend=str(df_backend), df_config=df_config,
            df_threads=int(df_threads), delta_bohr=float(fd_delta_bohr), profile=None,
        )

    # ── CPU phase: 1e derivative contractions (device synced by contract()) ──
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    if _is_sph_l:
        from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415
        de_h1 = contract_dhcore_sph(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M_sph=_asnumpy_f64(D_L), shell_atom=shell_atom,
        )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis, atom_coords_bohr=coords, atom_charges=charges,
            M=_asnumpy_f64(D_L), shell_atom=shell_atom,
        )
    dme0 = 0.5 * (gfock + gfock.T)
    if _is_sph_l:
        from asuka.integrals.int1e_sph import contract_dS_ip_sph  # noqa: PLC0415

        de_pulay = -2.0 * contract_dS_ip_sph(
            ao_basis,
            atom_coords_bohr=coords,
            M_sph=_asnumpy_f64(dme0),
            shell_atom=shell_atom,
        )
    else:
        de_pulay = -2.0 * contract_dS_ip_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=_asnumpy_f64(dme0),
            shell_atom=shell_atom,
        )
    # de_df already on CPU (contract() returns numpy).
    total = np.asarray(de_h1 + _asnumpy_f64(de_df) + np.asarray(de_pulay, dtype=np.float64), dtype=np.float64)
    if not bool(return_terms):
        _restore_pool()
        return total
    terms = {
        "dhcore": np.asarray(de_h1, dtype=np.float64),
        "df2e": _asnumpy_f64(de_df),
        "dS_pulay": np.asarray(de_pulay, dtype=np.float64),
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
    fd_integrals_builder: Callable[[Any], tuple[Any, Any, Any]] | None = None,
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
        Optional list of (bra, ket) root pairs. If None, compute all off-diagonal pairs.
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

    t_total_start = time.perf_counter()
    t_setup_start = t_total_start
    timing_payload: dict[str, Any] = {
        "status": "running",
        "response_term_input": str(response_term),
        "timings_s": {},
        "pair_timings_s": [],
    }

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    _is_sph_nac = not bool(getattr(mol, "cart", True))
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
        out = np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)
        timing_payload.update(
            {
                "status": "ok",
                "nroots": int(nroots),
                "natm": int(natm),
                "nac_shape": [int(x) for x in out.shape],
                "timings_s": {"total": float(time.perf_counter() - t_total_start)},
            }
        )
        _set_last_nacv_timing(timing_payload)
        return out

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

    xp_ref, _ = _resolve_xp(str(df_backend))

    C_ref = _as_xp_f64(xp_ref, getattr(casscf, "mo_coeff"))
    if C_ref.ndim != 2:
        raise ValueError("casscf.mo_coeff must be a 2D array (nao,nmo)")
    nao, nmo = map(int, C_ref.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    B_ref = _as_xp_f64(xp_ref, getattr(scf_out, "df_B"))
    hcore_ref = _as_xp_f64(xp_ref, getattr(getattr(scf_out, "int1e"), "hcore"))
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
    timing_payload["hessian_gpu_mode"] = bool(getattr(hess_op, "gpu_mode", False))

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
    timing_payload.update(
        {
            "response_term": str(response),
            "natm": int(natm),
            "nroots": int(nroots),
            "ncore": int(ncore),
            "ncas": int(ncas),
            "atmlst_len": int(len(atmlst_use)),
            "df_backend": str(df_backend),
            "df_threads": int(df_threads),
            "use_etfs": bool(use_etfs),
            "mult_ediff": bool(mult_ediff),
        }
    )
    timing_payload["timings_s"]["setup_pre_fd"] = float(time.perf_counter() - t_setup_start)

    dg = None
    if response == "fd_jacobian":
        t_fd_jacobian_start = time.perf_counter()
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
        U = S0_sqrt @ _asnumpy_f64(C_ref)  # orthonormal-AO representation

        def _fd_build_arrays(mol_disp: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

                def _g_at(mol_disp: Any) -> np.ndarray:
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
        timing_payload["timings_s"]["fd_jacobian_build"] = float(time.perf_counter() - t_fd_jacobian_start)
    else:
        timing_payload["timings_s"]["fd_jacobian_build"] = 0.0

    # Output tensor
    nac = np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    def _unpack_state(state: tuple[int, int]) -> tuple[int, int]:
        bra, ket = state
        bra = int(bra)
        ket = int(ket)
        if ket < 0 or bra < 0 or ket >= nroots or bra >= nroots:
            raise ValueError("state indices out of range")
        return bra, ket

    pair_list: list[tuple[int, int]]
    pair_fill_symmetric = False
    if pairs is None:
        pair_list = [(bra, ket) for bra in range(nroots) for ket in range(bra + 1, nroots)]
        pair_fill_symmetric = True
        timing_payload["pair_count"] = int(nroots * (nroots - 1))
    else:
        pair_list = [(int(bra), int(ket)) for (bra, ket) in pairs if int(ket) != int(bra)]
        timing_payload["pair_count"] = int(len(pair_list))
    timing_payload["pair_eval_count"] = int(len(pair_list))

    # Cache AO mappings used by the CSF term.
    shell_atom_ref = shell_to_atom_map(ao_basis_ref, atom_coords_bohr=coords)
    atmlst_idx = np.asarray(atmlst_use, dtype=np.int32)
    mo_cas = _asnumpy_f64(C_ref[:, ncore:nocc])
    trans_rdm12 = _base_fcisolver_method(fcisolver_use, "trans_rdm12")
    trans_rdm1 = None if bool(use_etfs) else _base_fcisolver_method(fcisolver_use, "trans_rdm1")
    eris_act = _eris_patch_active(eris_sa, mo_coeff=C_ref, hcore_ao=hcore_ref, ncore=int(ncore))
    active_core_cache: dict[str, Any] = {}

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

    timing_payload["timings_s"]["setup_total"] = float(time.perf_counter() - t_setup_start)
    t_pair_loop_start = time.perf_counter()
    pair_timings: list[dict] = []
    z_recycle_space: list[tuple[np.ndarray | None, np.ndarray]] = []
    pair_records: list[dict[str, Any]] = []
    rhs_orb_all: list[np.ndarray] = []
    rhs_ci_all: list[list[np.ndarray]] = []

    for bra, ket in pair_list:
        bra, ket = _unpack_state((bra, ket))
        if ket == bra:
            continue

        pair_timing: dict = {"bra": bra, "ket": ket}
        ediff = float(e_states[bra] - e_states[ket])

        # Transition densities for pair (bra,ket)
        t0 = time.perf_counter()
        dm1_t, dm2_t = trans_rdm12(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
        dm1_t = 0.5 * (np.asarray(dm1_t, dtype=np.float64) + np.asarray(dm1_t, dtype=np.float64).T)
        dm2_t = 0.5 * (
            np.asarray(dm2_t, dtype=np.float64) + np.asarray(dm2_t, dtype=np.float64).transpose(1, 0, 3, 2)
        )
        pair_timing["trans_rdm"] = float(time.perf_counter() - t0)

        # Hamiltonian response term (<bra|dH/dR|ket> without nuclear term)
        t0 = time.perf_counter()
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
            core_cache=active_core_cache,
        )
        pair_timing["ham_response"] = float(time.perf_counter() - t0)

        # CSF / AO-overlap term (numerator form).
        t0 = time.perf_counter()
        if not bool(use_etfs):
            if trans_rdm1 is None:  # pragma: no cover
                raise RuntimeError("internal error: missing trans_rdm1 builder")
            dm1 = trans_rdm1(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
            castm1 = np.asarray(dm1, dtype=np.float64).T - np.asarray(dm1, dtype=np.float64)
            tm1 = mo_cas @ castm1 @ mo_cas.T
            if _is_sph_nac:
                from asuka.integrals.int1e_sph import contract_dS_ip_sph  # noqa: PLC0415

                nac_csf = 0.5 * contract_dS_ip_sph(
                    ao_basis_ref,
                    atom_coords_bohr=coords,
                    M_sph=tm1,
                    shell_atom=shell_atom_ref,
                )
            else:
                nac_csf = 0.5 * contract_dS_ip_cart(
                    ao_basis_ref,
                    atom_coords_bohr=coords,
                    M=tm1,
                    shell_atom=shell_atom_ref,
                )
            # CSF overlap term: (E_bra - E_ket) * nac_csf_raw, added to the
            # Hamiltonian derivative numerator (matches PySCF convention).
            ham = np.asarray(ham, dtype=np.float64) + np.asarray(nac_csf * ediff, dtype=np.float64)
        pair_timing["csf_term"] = float(time.perf_counter() - t0)

        # Pair-specific Z-vector RHS in SA parameter space.
        t0 = time.perf_counter()
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
        g_ket, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
            mc_trans,
            C_ref,
            ci_list[ket],
            eris_act,
            verbose=0,
            implementation="internal",
        )
        g_ket = _asnumpy_f64(g_ket).ravel()
        g_bra, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
            mc_trans,
            C_ref,
            ci_list[bra],
            eris_act,
            verbose=0,
            implementation="internal",
        )
        g_bra = _asnumpy_f64(g_bra).ravel()

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
        pair_timing["zvector_rhs_build"] = float(time.perf_counter() - t0)

        pair_records.append(
            {
                "bra": bra,
                "ket": ket,
                "ediff": ediff,
                "ham": np.asarray(ham, dtype=np.float64),
                "pair_timing": pair_timing,
            }
        )
        rhs_orb_all.append(np.asarray(g_orb, dtype=np.float64))
        rhs_ci_all.append(rhs_ci_list)

    # Use GMRES when GPU mode is active — the GPU GCROTMK implementation
    # can stall on large orbital spaces (n_orb > 1000).
    _z_method = "gmres" if bool(getattr(hess_op, "gpu_mode", False)) else "gcrotmk"

    z_results: list[Any] = []
    if pair_records:
        t_z_solve_start = time.perf_counter()
        if len(pair_records) == 1:
            z_single = solve_mcscf_zvector(
                mc_sa,
                rhs_orb=rhs_orb_all[0],
                rhs_ci=rhs_ci_all[0],
                hessian_op=hess_op,
                tol=float(z_tol),
                maxiter=int(z_maxiter),
                recycle_space=z_recycle_space,
                method=_z_method,
            )
            z_results = [z_single]
            z_total_time = float(time.perf_counter() - t_z_solve_start)
            pair_records[0]["pair_timing"]["zvector_solve"] = z_total_time
            timing_payload["zvector_batch_info"] = {
                "mode": "single",
                "n_rhs": 1,
                "solver": str(z_single.info.get("solver", "unknown")),
                "backend": str(z_single.info.get("backend", "unknown")),
                "total_matvec_calls": int(z_single.info.get("matvec_calls", 0) or 0),
                "total_niter": int(z_single.info.get("niter", 0) or 0),
            }
        else:
            z_batch = solve_mcscf_zvector_batch(
                mc_sa,
                rhs_orb_list=rhs_orb_all,
                rhs_ci_list=rhs_ci_all,
                hessian_op=hess_op,
                tol=float(z_tol),
                maxiter=int(z_maxiter),
                recycle_space=z_recycle_space,
                reorder="input",
                shared_recycle=True,
                chain_x0=True,
                method=_z_method,
            )
            z_results = list(z_batch.results)
            z_total_time = float(time.perf_counter() - t_z_solve_start)
            share_weights = np.asarray(
                [float(res.info.get("matvec_calls", 0) or 0) for res in z_results],
                dtype=np.float64,
            )
            if not np.isfinite(share_weights).all() or float(share_weights.sum()) <= 0.0:
                share_weights = np.asarray(
                    [float(res.info.get("niter", 0) or 0) for res in z_results],
                    dtype=np.float64,
                )
            if not np.isfinite(share_weights).all() or float(share_weights.sum()) <= 0.0:
                share_weights = np.ones(len(z_results), dtype=np.float64)
            share_weights = share_weights / float(share_weights.sum())
            for rec, w in zip(pair_records, share_weights, strict=True):
                rec["pair_timing"]["zvector_solve"] = 0.0
                rec["pair_timing"]["zvector_solve_shared"] = float(z_total_time * float(w))
            timing_payload["zvector_batch_info"] = dict(z_batch.info)
            timing_payload["zvector_batch_info"]["mode"] = "batch"
        timing_payload["timings_s"]["zvector_total"] = float(z_total_time)
        timing_payload["timings_s"]["zvector_batch_total"] = float(z_total_time)

    for rec, z in zip(pair_records, z_results, strict=True):
        bra = int(rec["bra"])
        ket = int(rec["ket"])
        ediff = float(rec["ediff"])
        ham = np.asarray(rec["ham"], dtype=np.float64)
        pair_timing = rec["pair_timing"]

        Lvec = np.asarray(z.z_packed, dtype=np.float64).ravel()
        if int(Lvec.size) != int(hess_op.n_tot):
            raise RuntimeError("unexpected Z-vector packed length")
        t0 = time.perf_counter()

        resp_full = np.zeros((natm, 3), dtype=np.float64)
        if response == "fd_jacobian":
            if dg is None:  # pragma: no cover
                raise RuntimeError("internal error: dg is missing for fd_jacobian response")
            for ia in atmlst_use:
                for ax in range(3):
                    resp_full[int(ia), int(ax)] = float(np.dot(Lvec, dg[int(ia), int(ax)]))
            pair_timing["response_assemble"] = 0.0
            pair_timing["response_unpack"] = 0.0
            pair_timing["response_ci_rdm"] = 0.0
            pair_timing["response_lorb"] = float(time.perf_counter() - t0)
        else:
            # PySCF-style split response: Lci_dot_dgci + Lorb_dot_dgorb.
            n_orb = int(hess_op.n_orb)
            Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])
            Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
            if not isinstance(Lci_list, list) or len(Lci_list) != int(nroots):
                raise RuntimeError("unexpected CI unpack structure in Z-vector solution")

            # CI response: build (weighted) transition RDMs between Lci[root] and ci[root], then reuse the DF gradient.
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
            pair_timing["response_ci_rdm"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()

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
                core_cache=active_core_cache,
            )
            pair_timing["response_assemble"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()

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
            pair_timing["response_lorb"] = float(time.perf_counter() - t0)
            pair_timing["response_unpack"] = 0.0

            resp_full = np.asarray(de_lci, dtype=np.float64) + np.asarray(de_lorb, dtype=np.float64)

        nac_num = (
            np.asarray(ham, dtype=np.float64)[atmlst_idx]
            + np.asarray(resp_full, dtype=np.float64)[atmlst_idx]
        )

        if not bool(mult_ediff):
            if abs(ediff) < 1e-12:
                raise ZeroDivisionError("E_bra - E_ket is too small; use mult_ediff=True for numerator mode")
            nac_num = nac_num / ediff

        nac_num = np.asarray(nac_num, dtype=np.float64)
        nac[bra, ket] = nac_num
        if bool(pair_fill_symmetric):
            nac[ket, bra] = nac_num if bool(mult_ediff) else -nac_num
        pair_timing["total"] = float(
            sum(float(v) for k, v in pair_timing.items() if k not in ("bra", "ket"))
        )
        pair_timings.append(copy.copy(pair_timing))

    # Post-loop timing aggregation
    timing_payload["timings_s"]["pair_loop_total"] = float(time.perf_counter() - t_pair_loop_start)
    timing_payload["timings_s"]["total"] = float(time.perf_counter() - t_total_start)

    if pair_timings:
        stage_keys = [k for k in pair_timings[0] if k not in ("bra", "ket")]
        pair_stage_totals_s = {k: sum(pt.get(k, 0.0) for pt in pair_timings) for k in stage_keys}
        pair_stage_max_s = {k: max(pt.get(k, 0.0) for pt in pair_timings) for k in stage_keys}
        slowest_pair = max(pair_timings, key=lambda pt: float(pt.get("total", 0.0)))
        timing_payload["pair_stage_totals_s"] = pair_stage_totals_s
        timing_payload["pair_stage_max_s"] = pair_stage_max_s
        timing_payload["slowest_pair"] = {"bra": slowest_pair["bra"], "ket": slowest_pair["ket"],
                                           "total_s": float(slowest_pair.get("total", 0.0))}
    timing_payload["per_pair_timings"] = pair_timings
    timing_payload["pair_timings_s"] = pair_timings
    timing_payload["status"] = "ok"

    _set_last_nacv_timing(timing_payload)
    return nac


__all__ = ["sacasscf_nonadiabatic_couplings_df", "get_last_nacv_timing"]
