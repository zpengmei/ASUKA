from __future__ import annotations

"""ASUKA-native analytic nuclear gradients for MRCI.

This module computes spin-free analytic gradients for native ASUKA MRCI
targets on top of an SA-CASSCF reference. The current implementation supports:

- uncontracted `mrcisd`
- contracted `ic_mrcisd` by expanding shared-basis roots back to the
  uncontracted MRCI CSF space for target densities
- DF-backed and THC-backed target two-electron derivatives

Common restrictions
-------------------
- `correlate_inactive == 0`
- `n_virt is None` (all virtual orbitals correlated)

THC-specific restrictions
-------------------------
- global `THCFactors` and gradient-capable `LocalTHCFactors`
- no point downselect
- solve method in `{'inv_metric', 'fit_metric_gram'}`

When these conditions are not met, callers should use the FD gradient backend.
"""

import contextlib
import os
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.result import MRCIStatesResult


@contextlib.contextmanager
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


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    if hasattr(a, "get"):
        try:
            return np.asarray(a.get(), dtype=np.float64)
        except Exception:
            pass
    return np.asarray(a, dtype=np.float64)


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("THC analytic MRCI gradients require CuPy") from e
    return cp


def _ensure_df_response_factors(scf_out: Any, *, df_config: Any | None) -> tuple[Any, Any | None]:
    """Return (B_ao, L_chol), building DF factors on demand for THC references."""

    B_ao = getattr(scf_out, "df_B", None)
    L_chol = getattr(scf_out, "df_L", None)
    if B_ao is not None:
        return B_ao, L_chol

    from asuka.integrals.cueri_df import build_df_B_from_cueri_packed_bases  # noqa: PLC0415

    ao_basis = getattr(scf_out, "ao_basis", None)
    aux_basis = getattr(scf_out, "aux_basis", None)
    if ao_basis is None or aux_basis is None:
        raise ValueError("analytic gradients require scf_out.ao_basis and scf_out.aux_basis")
    ao_rep = "cart" if bool(getattr(getattr(scf_out, "mol", None), "cart", True)) else "sph"
    B_ao, L_chol = build_df_B_from_cueri_packed_bases(
        ao_basis,
        aux_basis,
        config=df_config,
        layout="mnQ",
        ao_rep=str(ao_rep),
        return_L=True,
    )
    return B_ao, L_chol


def _infer_target_backend(scf_out: Any) -> str:
    if getattr(scf_out, "df_B", None) is None and getattr(scf_out, "thc_factors", None) is not None:
        return "thc"
    return "df"


def _add_thc_bar_components(bar_X_a, bar_Y_a, bar_X_b, bar_Y_b):
    if isinstance(bar_X_a, list):
        return (
            [ax + bx for ax, bx in zip(bar_X_a, bar_X_b, strict=True)],
            [ay + by for ay, by in zip(bar_Y_a, bar_Y_b, strict=True)],
        )
    return bar_X_a + bar_X_b, bar_Y_a + bar_Y_b


def _scale_thc_bar_components(bar_X, bar_Y, alpha: float):
    alpha_f = float(alpha)
    if isinstance(bar_X, list):
        return [alpha_f * v for v in bar_X], [alpha_f * v for v in bar_Y]
    return alpha_f * bar_X, alpha_f * bar_Y


def _mask_local_left_density(M_sub, blk):
    B_eff = M_sub.copy()
    nloc = int(B_eff.shape[0])
    n_early = int(getattr(blk, "n_early", 0))
    n_primary = int(getattr(blk, "n_primary", 0))
    if n_early < 0 or n_early > nloc:
        raise ValueError("invalid blk.n_early")
    if n_primary < 0 or (n_early + n_primary) > nloc:
        raise ValueError("invalid blk.n_primary")
    tail = int(n_early + n_primary)
    if n_early > 0:
        B_eff[:n_early, :] = 0.0
        B_eff[:, :n_early] = 0.0
    if tail < nloc:
        B_eff[tail:, tail:] = 0.0
    return B_eff


def _rebuild_mrci_integral_payload(
    scf_out: Any,
    ref: Any,
    *,
    mrci_states: MRCIStatesResult,
) -> tuple[np.ndarray, Any]:
    """Rebuild the correlated-space MRCI integral payload for residual-based helpers."""

    from asuka.mrci.driver_asuka import (  # noqa: PLC0415
        _dfmo_integrals_from_df_B,
        _device_dfmo_integrals_from_thc,
        _frozen_core_h1e_ecore_df,
        _frozen_core_h1e_ecore_thc,
    )

    mo = _asnumpy_f64(getattr(ref, "mo_coeff"))
    ncore_ref = int(getattr(ref, "ncore", 0))
    n_act_ref = int(getattr(ref, "ncas", 0))
    ncore_frozen = int(mrci_states.ncore)
    correlate_inactive = int(mrci_states.n_act) - int(n_act_ref)
    if int(correlate_inactive) != 0:
        raise NotImplementedError(
            "Analytic native MRCI gradients currently require correlate_inactive==0 for residual-based CI RHS"
        )
    nvirt_use = int(mrci_states.n_virt)

    mo_core_frozen = mo[:, :ncore_frozen]
    mo_core_corr = mo[:, ncore_frozen:ncore_ref]
    mo_act = mo[:, ncore_ref : ncore_ref + n_act_ref]
    mo_virt = mo[:, ncore_ref + n_act_ref : ncore_ref + n_act_ref + nvirt_use]
    mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

    target_backend = str(getattr(mrci_states, "integrals_backend", _infer_target_backend(scf_out))).strip().lower()
    if target_backend == "thc":
        h1e_corr, _ecore = _frozen_core_h1e_ecore_thc(
            scf_out=scf_out,
            mo_core=mo_core_frozen,
            mo_corr=mo_corr,
        )
        eri_payload = _device_dfmo_integrals_from_thc(scf_out, mo_corr, want_pair_norm=True)
        return np.asarray(h1e_corr, dtype=np.float64), eri_payload

    B_ao = getattr(scf_out, "df_B", None)
    if B_ao is None:
        raise ValueError("DF-backed residual rebuild requires scf_out.df_B")
    h1e_corr, _ecore = _frozen_core_h1e_ecore_df(
        scf_out=scf_out,
        mo_core=mo_core_frozen,
        mo_corr=mo_corr,
    )
    try:
        import cupy as cp  # noqa: PLC0415

        use_cuda = isinstance(B_ao, cp.ndarray)
    except Exception:
        use_cuda = False
    eri_payload = _dfmo_integrals_from_df_B(B_ao if use_cuda else _asnumpy_f64(B_ao), mo_corr, device="cuda" if use_cuda else "cpu")
    return np.asarray(h1e_corr, dtype=np.float64), eri_payload


def _build_bar_xy_target_thc(
    scf_out: Any,
    *,
    D_core_ao: Any,
    D_corr_ao: Any,
    C_corr: Any,
    dm2_corr: Any,
    q_block: int = 256,
    pair_p_block: int = 8,
) -> tuple[Any, Any]:
    cp = _require_cupy()
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415
    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_thc import (  # noqa: PLC0415
        _dm2_sym_flat,
        _thc_energy_adjoint_active_local_block,
        _thc_energy_adjoint_active_global,
        _thc_energy_adjoint_jk_bilinear,
    )

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    D_core = cp.asarray(D_core_ao, dtype=cp.float64)
    D_corr = cp.asarray(D_corr_ao, dtype=cp.float64)
    C_corr = cp.asarray(C_corr, dtype=cp.float64)
    D_w = D_corr + 0.5 * D_core
    ncor = int(C_corr.shape[1])
    dm2_flat = _dm2_sym_flat(cp, dm2_corr, ncas=int(ncor))
    if isinstance(thc, THCFactors):
        bar_X_mean, bar_Y_mean = _thc_energy_adjoint_jk_bilinear(
            D_core,
            D_w,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_aa, bar_Y_aa = _thc_energy_adjoint_active_global(
            thc.X,
            thc.Y,
            C_corr,
            dm2_flat,
            pair_p_block=int(pair_p_block),
        )
        return (
            cp.ascontiguousarray(bar_X_mean + bar_X_aa),
            cp.ascontiguousarray(bar_Y_mean + bar_Y_aa),
        )

    bar_X_list = []
    bar_Y_list = []
    for blk in thc.blocks:
        idx_np = np.asarray(getattr(blk, "ao_idx_global"), dtype=np.int32).ravel()
        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_core_sub = D_core[idx[:, None], idx[None, :]]
        D_w_sub = D_w[idx[:, None], idx[None, :]]
        B_eff = _mask_local_left_density(D_w_sub, blk)

        bar_X_mean, bar_Y_mean = _thc_energy_adjoint_jk_bilinear(
            D_core_sub,
            B_eff,
            blk.X,
            blk.Z,
            blk.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_aa, bar_Y_aa = _thc_energy_adjoint_active_local_block(
            blk.X,
            blk.Z,
            blk.Y,
            ao_idx_global=idx_np,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(getattr(blk, "n_primary", 0)),
            C_act=C_corr,
            dm2_flat_sym=dm2_flat,
            pair_p_block=int(pair_p_block),
            q_block=int(q_block),
        )
        bar_X_list.append(cp.ascontiguousarray(bar_X_mean + bar_X_aa))
        bar_Y_list.append(cp.ascontiguousarray(bar_Y_mean + bar_Y_aa))
    return bar_X_list, bar_Y_list


def _build_bar_xy_net_active_thc(
    scf_out: Any,
    *,
    C: Any,
    dm1_act: Any,
    dm2_act: Any,
    ncore: int,
    ncas: int,
    q_block: int = 256,
    pair_p_block: int = 8,
) -> tuple[Any, Any, Any]:
    cp = _require_cupy()
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415
    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_thc import (  # noqa: PLC0415
        _dm2_sym_flat,
        _thc_energy_adjoint_active_local_block,
        _thc_energy_adjoint_active_global,
        _thc_energy_adjoint_jk_bilinear,
    )

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    C = cp.asarray(C, dtype=cp.float64)
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = int(ncore + ncas)
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    dm1 = cp.asarray(dm1_act, dtype=cp.float64)
    dm1 = 0.5 * (dm1 + dm1.T)

    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
    else:
        D_core = cp.zeros((int(C.shape[0]), int(C.shape[0])), dtype=cp.float64)
    D_act = C_act @ dm1 @ C_act.T
    D_w = D_act + 0.5 * D_core
    D_ah = D_w - D_core

    dm2_flat = _dm2_sym_flat(cp, dm2_act, ncas=int(ncas))
    if isinstance(thc, THCFactors):
        bar_X_1, bar_Y_1 = _thc_energy_adjoint_jk_bilinear(
            D_core,
            D_w,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_2, bar_Y_2 = _thc_energy_adjoint_jk_bilinear(
            D_ah,
            D_core,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_3, bar_Y_3 = _thc_energy_adjoint_active_global(
            thc.X,
            thc.Y,
            C_act,
            dm2_flat,
            pair_p_block=int(pair_p_block),
        )
        return (
            cp.ascontiguousarray(bar_X_1 + bar_X_2 + bar_X_3),
            cp.ascontiguousarray(bar_Y_1 + bar_Y_2 + bar_Y_3),
            cp.ascontiguousarray(D_act, dtype=cp.float64),
        )

    bar_X_list = []
    bar_Y_list = []
    for blk in thc.blocks:
        idx_np = np.asarray(getattr(blk, "ao_idx_global"), dtype=np.int32).ravel()
        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_core_sub = D_core[idx[:, None], idx[None, :]]
        D_w_sub = D_w[idx[:, None], idx[None, :]]
        D_ah_sub = D_ah[idx[:, None], idx[None, :]]
        B_eff_w = _mask_local_left_density(D_w_sub, blk)
        B_eff_core = _mask_local_left_density(D_core_sub, blk)

        bar_X_1, bar_Y_1 = _thc_energy_adjoint_jk_bilinear(
            D_core_sub,
            B_eff_w,
            blk.X,
            blk.Z,
            blk.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_2, bar_Y_2 = _thc_energy_adjoint_jk_bilinear(
            D_ah_sub,
            B_eff_core,
            blk.X,
            blk.Z,
            blk.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_3, bar_Y_3 = _thc_energy_adjoint_active_local_block(
            blk.X,
            blk.Z,
            blk.Y,
            ao_idx_global=idx_np,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(getattr(blk, "n_primary", 0)),
            C_act=C_act,
            dm2_flat_sym=dm2_flat,
            pair_p_block=int(pair_p_block),
            q_block=int(q_block),
        )
        bar_X_list.append(cp.ascontiguousarray(bar_X_1 + bar_X_2 + bar_X_3))
        bar_Y_list.append(cp.ascontiguousarray(bar_Y_1 + bar_Y_2 + bar_Y_3))
    return bar_X_list, bar_Y_list, cp.ascontiguousarray(D_act, dtype=cp.float64)


def _build_bar_xy_lorb_thc(
    scf_out: Any,
    *,
    C: Any,
    Lorb: np.ndarray,
    dm1_act: Any,
    dm2_act: Any,
    ncore: int,
    ncas: int,
    q_block: int = 256,
    pair_p_block: int = 8,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    cp = _require_cupy()
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415
    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_thc import (  # noqa: PLC0415
        _dm2_sym_flat,
        _thc_energy_adjoint_active_local_block,
        _thc_energy_adjoint_active_global,
        _thc_energy_adjoint_jk_bilinear,
    )

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    C = cp.asarray(C, dtype=cp.float64)
    L = cp.asarray(Lorb, dtype=cp.float64)
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = int(ncore + ncas)
    nao, nmo = map(int, C.shape)

    dm1 = cp.asarray(dm1_act, dtype=cp.float64)
    dm2 = cp.asarray(dm2_act, dtype=cp.float64)
    try:
        scale = float(os.environ.get("ASUKA_CASPT2_LORB_ACTIVE_RESPONSE_SCALE", "0.0"))
    except Exception:
        scale = 0.0
    dm1 = scale * dm1
    dm2 = scale * dm2

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    C_L = C @ L
    C_L_core = C_L[:, :ncore]
    C_L_act = C_L[:, ncore:nocc]

    dml_sym_mode = str(os.environ.get("ASUKA_CASPT2_LORB_DML_SYM_MODE", "full")).strip().lower()
    if dml_sym_mode not in {"full", "core_raw", "act_raw", "raw", "core_asym", "act_asym", "asym"}:
        dml_sym_mode = "full"

    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core_raw = 2.0 * (C_L_core @ C_core.T)
        D_L_core_sym = D_L_core_raw + D_L_core_raw.T
        D_L_core_asym = D_L_core_raw - D_L_core_raw.T
        if dml_sym_mode in {"core_raw", "raw"}:
            D_L_core = D_L_core_raw
        elif dml_sym_mode in {"core_asym", "asym"}:
            D_L_core = D_L_core_asym
        else:
            D_L_core = D_L_core_sym
    else:
        D_core = cp.zeros((nao, nao), dtype=cp.float64)
        D_L_core = cp.zeros((nao, nao), dtype=cp.float64)

    D_act = C_act @ dm1 @ C_act.T
    D_L_act_raw = C_L_act @ dm1 @ C_act.T
    D_L_act_sym = D_L_act_raw + D_L_act_raw.T
    D_L_act_asym = D_L_act_raw - D_L_act_raw.T
    if dml_sym_mode in {"act_raw", "raw"}:
        D_L_act = D_L_act_raw
    elif dml_sym_mode in {"act_asym", "asym"}:
        D_L_act = D_L_act_asym
    else:
        D_L_act = D_L_act_sym
    D_L = D_L_core + D_L_act

    D_w = D_act + 0.5 * D_core
    D_wL = D_L_act + 0.5 * D_L_core
    dm2_flat = _dm2_sym_flat(cp, dm2, ncas=int(ncas))
    if isinstance(thc, THCFactors):
        bar_X_1, bar_Y_1 = _thc_energy_adjoint_jk_bilinear(
            D_core,
            D_wL,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_2, bar_Y_2 = _thc_energy_adjoint_jk_bilinear(
            D_L_core,
            D_w,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )

        bar_X_act = cp.zeros_like(cp.asarray(thc.X, dtype=cp.float64))
        bar_Y_act = cp.zeros_like(cp.asarray(thc.Y, dtype=cp.float64))
        if abs(float(scale)) > 0.0:
            eps_f = float(eps)
            bar_X_p, bar_Y_p = _thc_energy_adjoint_active_global(
                thc.X,
                thc.Y,
                C_act + eps_f * C_L_act,
                dm2_flat,
                pair_p_block=int(pair_p_block),
            )
            bar_X_m, bar_Y_m = _thc_energy_adjoint_active_global(
                thc.X,
                thc.Y,
                C_act - eps_f * C_L_act,
                dm2_flat,
                pair_p_block=int(pair_p_block),
            )
            bar_X_act = (bar_X_p - bar_X_m) / (2.0 * eps_f)
            bar_Y_act = (bar_Y_p - bar_Y_m) / (2.0 * eps_f)

        return (
            cp.ascontiguousarray(bar_X_1 + bar_X_2 + bar_X_act),
            cp.ascontiguousarray(bar_Y_1 + bar_Y_2 + bar_Y_act),
            cp.ascontiguousarray(D_L, dtype=cp.float64),
        )

    bar_X_list = []
    bar_Y_list = []
    for blk in thc.blocks:
        idx_np = np.asarray(getattr(blk, "ao_idx_global"), dtype=np.int32).ravel()
        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_core_sub = D_core[idx[:, None], idx[None, :]]
        D_w_sub = D_w[idx[:, None], idx[None, :]]
        D_wL_sub = D_wL[idx[:, None], idx[None, :]]
        D_L_core_sub = D_L_core[idx[:, None], idx[None, :]]
        B_eff_wL = _mask_local_left_density(D_wL_sub, blk)
        B_eff_Lcore = _mask_local_left_density(D_L_core_sub, blk)

        bar_X_1, bar_Y_1 = _thc_energy_adjoint_jk_bilinear(
            D_core_sub,
            B_eff_wL,
            blk.X,
            blk.Z,
            blk.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_2, bar_Y_2 = _thc_energy_adjoint_jk_bilinear(
            D_w_sub,
            B_eff_Lcore,
            blk.X,
            blk.Z,
            blk.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )

        bar_X_act = cp.zeros_like(cp.asarray(blk.X, dtype=cp.float64))
        bar_Y_act = cp.zeros_like(cp.asarray(blk.Y, dtype=cp.float64))
        if abs(float(scale)) > 0.0:
            eps_f = float(eps)
            bar_X_p, bar_Y_p = _thc_energy_adjoint_active_local_block(
                blk.X,
                blk.Z,
                blk.Y,
                ao_idx_global=idx_np,
                n_early=int(getattr(blk, "n_early", 0)),
                n_primary=int(getattr(blk, "n_primary", 0)),
                C_act=C_act + eps_f * C_L_act,
                dm2_flat_sym=dm2_flat,
                pair_p_block=int(pair_p_block),
                q_block=int(q_block),
            )
            bar_X_m, bar_Y_m = _thc_energy_adjoint_active_local_block(
                blk.X,
                blk.Z,
                blk.Y,
                ao_idx_global=idx_np,
                n_early=int(getattr(blk, "n_early", 0)),
                n_primary=int(getattr(blk, "n_primary", 0)),
                C_act=C_act - eps_f * C_L_act,
                dm2_flat_sym=dm2_flat,
                pair_p_block=int(pair_p_block),
                q_block=int(q_block),
            )
            bar_X_act = (bar_X_p - bar_X_m) / (2.0 * eps_f)
            bar_Y_act = (bar_Y_p - bar_Y_m) / (2.0 * eps_f)

        bar_X_list.append(cp.ascontiguousarray(bar_X_1 + bar_X_2 + bar_X_act))
        bar_Y_list.append(cp.ascontiguousarray(bar_Y_1 + bar_Y_2 + bar_Y_act))

    return bar_X_list, bar_Y_list, cp.ascontiguousarray(D_L, dtype=cp.float64)


def _contract_thc_bar_adjoint(
    scf_out: Any,
    *,
    bar_X: Any,
    bar_Y: Any,
    df_threads: int = 0,
) -> np.ndarray:
    cp = _require_cupy()
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415
    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_thc import (  # noqa: PLC0415
        _thc_factor_vjp_atomgrad_fit_metric_gram,
        _thc_factor_vjp_atomgrad_inv_metric,
    )

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    meta = {} if thc.meta is None else dict(thc.meta)
    solve_method = str(meta.get("solve_method", "fit_metric_qr")).strip().lower()
    inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
    fit_metric_gram_methods = {"fit_metric_gram", "gram"}
    if solve_method in inv_metric_methods:
        solve_kind = "inv_metric"
    elif solve_method in fit_metric_gram_methods:
        solve_kind = "fit_metric_gram"
    else:
        raise NotImplementedError(
            "Analytic MRCI THC gradients currently support solve_method in {'inv_metric','fit_metric_gram'} "
            f"(got {solve_method!r})"
        )
    is_spherical = not bool(getattr(mol, "cart", True))
    sph_map = getattr(scf_out, "sph_map", None)
    ao_basis_cart = getattr(scf_out, "ao_basis")
    aux_basis_cart = getattr(scf_out, "aux_basis")
    if isinstance(thc, THCFactors):
        if bool(meta.get("downselected", False)):
            raise NotImplementedError("Analytic MRCI THC gradients require THC factors built without point downselect")
        point_atom = meta.get("point_atom", None)
        if point_atom is None:
            raise ValueError("THC factors are missing meta['point_atom']; rebuild with gradient-capable metadata")
        grid_kind = str(meta.get("grid_kind", "")).strip().lower()
        if grid_kind not in {"becke", "rdvr"}:
            raise NotImplementedError("Analytic MRCI THC gradients currently support only grid_kind in {'becke','rdvr'}")
        becke_n = int(meta.get("becke_n", 3))

        if solve_kind == "inv_metric":
            g_thc, g_metric = _thc_factor_vjp_atomgrad_inv_metric(
                mol=mol,
                ao_basis_cart=ao_basis_cart,
                aux_basis_cart=aux_basis_cart,
                sph_map=sph_map,
                is_spherical=bool(is_spherical),
                pts=thc.points,
                w=thc.weights,
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=thc.X,
                Y=thc.Y,
                L_metric=thc.L_metric,
                bar_X=bar_X,
                bar_Y=bar_Y,
                df_threads=int(df_threads),
            )
        else:
            g_thc, g_metric = _thc_factor_vjp_atomgrad_fit_metric_gram(
                mol=mol,
                ao_basis_cart=ao_basis_cart,
                aux_basis_cart=aux_basis_cart,
                sph_map=sph_map,
                is_spherical=bool(is_spherical),
                pts=thc.points,
                w=thc.weights,
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=thc.X,
                Y=thc.Y,
                L_metric=thc.L_metric,
                bar_X=bar_X,
                bar_Y=bar_Y,
                solve_rcond=float(meta.get("solve_rcond", 1e-12)),
                df_threads=int(df_threads),
            )
        cp.cuda.get_current_stream().synchronize()
        return np.asarray(cp.asnumpy(g_thc + g_metric), dtype=np.float64)

    from asuka.cueri.basis_subset import subset_cart_basis_by_shells  # noqa: PLC0415

    if bool(meta.get("downselected", False)):
        raise NotImplementedError("Analytic local-THC MRCI gradients require factors built without point downselect")
    grad_thc_dev = cp.zeros((int(getattr(mol, "natm", len(getattr(mol, "atoms_bohr", [])))), 3), dtype=cp.float64)
    grad_metric_dev = cp.zeros_like(grad_thc_dev)
    for blk, bar_X_blk, bar_Y_blk in zip(thc.blocks, bar_X, bar_Y, strict=True):
        bmeta = {} if getattr(blk, "meta", None) is None else dict(getattr(blk, "meta"))
        if bool(bmeta.get("downselected", False)):
            raise NotImplementedError("Analytic local-THC MRCI gradients require blocks built without point downselect")
        point_atom = bmeta.get("point_atom", None)
        if point_atom is None:
            raise ValueError("LocalTHCBlock.meta['point_atom'] is missing; rebuild local THC factors with gradient metadata")
        grid_kind = str(bmeta.get("grid_kind", "")).strip().lower()
        if grid_kind not in {"becke", "rdvr"}:
            raise NotImplementedError("Analytic local-THC MRCI gradients currently support only grid_kind in {'becke','rdvr'}")
        becke_n = int(bmeta.get("becke_n", 3))
        ao_shells = bmeta.get("ao_shells", None)
        aux_shells = bmeta.get("aux_shells", None)
        if ao_shells is None or aux_shells is None:
            raise ValueError("LocalTHCBlock.meta is missing ao_shells/aux_shells; rebuild local THC factors with metadata")
        ao_basis_blk = subset_cart_basis_by_shells(ao_basis_cart, list(map(int, ao_shells)))
        aux_basis_blk = subset_cart_basis_by_shells(aux_basis_cart, list(map(int, aux_shells)))

        blk_sph_map = None
        if bool(is_spherical):
            from asuka.integrals.cart2sph import (  # noqa: PLC0415
                build_cart2sph_matrix,
                compute_sph_layout_from_cart_basis,
            )
            from asuka.cueri.cart import ncart  # noqa: PLC0415

            shell_l_blk = np.asarray(getattr(ao_basis_blk, "shell_l"), dtype=np.int32).ravel()
            shell_start_cart_blk = np.asarray(getattr(ao_basis_blk, "shell_ao_start"), dtype=np.int32).ravel()
            shell_start_sph_blk, nao_sph_blk = compute_sph_layout_from_cart_basis(ao_basis_blk)
            if int(shell_l_blk.size):
                nfn_cart = np.asarray([ncart(int(l)) for l in shell_l_blk.tolist()], dtype=np.int32)
                nao_cart_blk = int(np.max(shell_start_cart_blk + nfn_cart))
            else:
                nao_cart_blk = 0
            T = build_cart2sph_matrix(
                shell_l_blk,
                shell_start_cart_blk,
                np.asarray(shell_start_sph_blk, dtype=np.int32).ravel(),
                int(nao_cart_blk),
                int(nao_sph_blk),
            )

            class _TmpSphMap:
                T_c2s = T

            blk_sph_map = _TmpSphMap()

        if solve_kind == "inv_metric":
            g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_inv_metric(
                mol=mol,
                ao_basis_cart=ao_basis_blk,
                aux_basis_cart=aux_basis_blk,
                sph_map=blk_sph_map,
                is_spherical=bool(is_spherical),
                pts=getattr(blk, "points"),
                w=getattr(blk, "weights"),
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=getattr(blk, "X"),
                Y=getattr(blk, "Y"),
                L_metric=getattr(blk, "L_metric"),
                bar_X=bar_X_blk,
                bar_Y=bar_Y_blk,
                df_threads=int(df_threads),
            )
        else:
            g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_fit_metric_gram(
                mol=mol,
                ao_basis_cart=ao_basis_blk,
                aux_basis_cart=aux_basis_blk,
                sph_map=blk_sph_map,
                is_spherical=bool(is_spherical),
                pts=getattr(blk, "points"),
                w=getattr(blk, "weights"),
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=getattr(blk, "X"),
                Y=getattr(blk, "Y"),
                L_metric=getattr(blk, "L_metric"),
                bar_X=bar_X_blk,
                bar_Y=bar_Y_blk,
                solve_rcond=1e-12,
                df_threads=int(df_threads),
            )
        grad_thc_dev += g_thc_blk
        grad_metric_dev += g_metric_blk

    cp.cuda.get_current_stream().synchronize()
    return np.asarray(cp.asnumpy(grad_thc_dev + grad_metric_dev), dtype=np.float64)


def mrci_grad_states_from_ref_analytic(
    scf_out: Any,
    ref: Any,
    *,
    mrci_states: MRCIStatesResult,
    roots: np.ndarray,
    states: Sequence[int],
    max_virt_e: int = 2,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> list[np.ndarray]:
    """Compute analytic MRCISD gradients for the requested reference states.

    Parameters
    ----------
    scf_out
        ASUKA HF/DF output (provides DF factors, AO bases, hcore).
    ref
        ASUKA CASSCF/CASCI result (mo_coeff, ci, ncore, ncas, nelecas, etc.).
    mrci_states
        Spin-free MRCISD result from :func:`asuka.mrci.driver_asuka.mrci_states_from_ref`.
    roots
        Assigned MRCI root indices for each requested reference state (same order as `states`).
    states
        Reference state indices (for bookkeeping only; order must match `roots` and mrci_states.states).
    max_virt_e
        MRCISD truncation parameter used to build the DRT.
    rdm_backend
        Backend for RDM evaluation ("cpu" or "cuda").
    df_backend
        Backend for DF derivative contraction ("cpu" or "cuda").
    df_config, df_threads
        DF contraction knobs (passed to DF derivative kernels / FD fallback).
    z_tol, z_maxiter
        Z-vector solver controls.
    """

    # ── Imports (local to keep module import lightweight) ─────────────────────
    from asuka.integrals.grad import (  # noqa: PLC0415
        compute_df_gradient_contributions_analytic_sph,
        compute_df_gradient_contributions_analytic_packed_bases,
        compute_df_gradient_contributions_fd_packed_bases,
    )
    from asuka.integrals.int1e_cart import (  # noqa: PLC0415
        contract_dS_cart,
        contract_dhcore_cart,
        shell_to_atom_map,
    )
    from asuka.integrals.df_grad_context import DFGradContractionContext  # noqa: PLC0415
    from asuka.mcscf.nac._df import (  # noqa: PLC0415
        _build_bar_L_lorb_df,
        _build_bar_L_net_active_df,
    )
    from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_df import (  # noqa: PLC0415
        _apply_df_pool_policy,
        _as_xp_f64,
        _build_bar_L_casscf_df,
        _build_dme0_lorb_response,
        _build_gfock_casscf_df,
        _mol_coords_charges_bohr,
        _resolve_barl_hybrid,
        _resolve_xp,
    )
    from asuka.mcscf.state_average import (  # noqa: PLC0415
        ci_as_list,
        make_state_averaged_rdms,
        normalize_weights,
    )
    from asuka.mcscf.zvector import (  # noqa: PLC0415
        build_mcscf_hessian_operator,
        solve_mcscf_zvector,
    )
    from asuka.mrci.rdm_mrcisd import (  # noqa: PLC0415
        make_rdm12_mrcisd,
        prepare_mrcisd_rdm_workspace,
    )
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    # ── Validate basic shapes / invariance assumptions ───────────────────────
    states_list = [int(s) for s in states]
    roots = np.asarray(roots, dtype=np.int64).ravel()
    if roots.shape != (len(states_list),):
        raise ValueError("roots must have shape (len(states),)")

    ncore_ref = int(getattr(ref, "ncore", 0))
    ncas_ref = int(getattr(ref, "ncas", 0))
    if ncas_ref <= 0:
        raise ValueError("ref.ncas must be positive")

    n_act_int = int(mrci_states.n_act)
    correlate_inactive = int(n_act_int - ncas_ref)
    if correlate_inactive != 0:
        raise NotImplementedError(
            "Analytic MRCISD gradients currently require correlate_inactive==0. "
            "Split inactive spaces make the energy depend on redundant core-core rotations."
        )

    C_full = _asnumpy_f64(getattr(ref, "mo_coeff"))
    nmo = int(C_full.shape[1])
    nvirt_all = nmo - ncore_ref - ncas_ref
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    nvirt_corr = int(mrci_states.n_virt)
    if nvirt_corr != int(nvirt_all):
        raise NotImplementedError(
            "Analytic MRCISD gradients currently require n_virt=None (all virtual orbitals correlated). "
            f"Got n_virt={nvirt_corr} but total virtuals={nvirt_all}."
        )

    # ── Common SCF/CASSCF objects ───────────────────────────────────────────
    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])

    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    is_spherical = not bool(getattr(mol, "cart", True))

    method_s = str(getattr(mrci_states, "method", "mrcisd")).strip().lower()
    if method_s not in {"mrcisd", "ic_mrcisd"}:
        raise ValueError("mrci_states.method must be 'mrcisd' or 'ic_mrcisd'")
    target_backend = str(getattr(mrci_states, "integrals_backend", _infer_target_backend(scf_out))).strip().lower()
    if target_backend not in {"df_b", "thc"}:
        raise ValueError("mrci_states.integrals_backend must be 'df_B' or 'thc'")
    target_backend = "thc" if target_backend == "thc" else "df"

    B_ao, L_chol = _ensure_df_response_factors(scf_out, df_config=df_config)
    h_ao = getattr(getattr(scf_out, "int1e"), "hcore", None)
    if h_ao is None:
        raise ValueError("scf_out.int1e.hcore is required")
    _restore_pool = _apply_df_pool_policy(B_ao, label="mrci_grad_states_from_ref_analytic")
    xp, _is_gpu = _resolve_xp(str(df_backend))
    B_ao_x = _as_xp_f64(xp, B_ao)
    h_ao_x = _as_xp_f64(xp, h_ao)
    C_full_x = _as_xp_f64(xp, C_full)
    _b_shape = tuple(map(int, getattr(B_ao_x, "shape", (0, 0, 0))))
    _naux_hint = int(_b_shape[0]) if len(_b_shape) == 2 else int(_b_shape[2])
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
    B_ao_np = _asnumpy_f64(B_ao)
    h_ao_np = _asnumpy_f64(h_ao)

    # Reference SA-CASSCF ingredients (Hessian operator for Z-vector solve).
    nroots_ref = int(getattr(ref, "nroots", 1))
    weights = normalize_weights(getattr(ref, "root_weights", None), nroots=nroots_ref)
    ci_list = ci_as_list(getattr(ref, "ci"), nroots=nroots_ref)
    nelecas = getattr(ref, "nelecas")
    twos = int(getattr(getattr(scf_out, "mol", None), "spin", 0))

    fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots_ref))
    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(ncas_ref),
        nelecas=nelecas,
    )
    dm1_sa = np.asarray(dm1_sa, dtype=np.float64)
    dm2_sa = np.asarray(dm2_sa, dtype=np.float64)

    mc_sa = DFNewtonCASSCFAdapter(
        df_B=B_ao_np,
        hcore_ao=h_ao_np,
        ncore=int(ncore_ref),
        ncas=int(ncas_ref),
        nelecas=nelecas,
        mo_coeff=C_full,
        fcisolver=fcisolver,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel()],
        frozen=getattr(ref, "frozen", None),
        internal_rotation=bool(getattr(ref, "internal_rotation", False)),
        extrasym=getattr(ref, "extrasym", None),
    )
    eris_sa = mc_sa.ao2mo(C_full)
    ppaa_sa = np.asarray(getattr(eris_sa, "ppaa"), dtype=np.float64)
    papa_sa = np.asarray(getattr(eris_sa, "papa"), dtype=np.float64)
    with _force_internal_newton():
        hess_op = build_mcscf_hessian_operator(
            mc_sa,
            mo_coeff=C_full,
            ci=ci_list,
            eris=eris_sa,
            use_newton_hessian=True,
        )
    n_orb = int(hess_op.n_orb)

    # Optional cached DF contraction context.
    df_grad_ctx = None
    try:
        df_grad_ctx = DFGradContractionContext.build(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            backend=str(df_backend),
            df_threads=int(df_threads),
            L_chol=L_chol,
        )
    except Exception:
        df_grad_ctx = None

    # MRCI RDM workspace (correlated-space DRT).
    rdm_drt = getattr(mrci_states.mrci, "drt", None)
    if rdm_drt is None:
        rdm_drt = getattr(mrci_states.mrci, "drt_work", None)
    if rdm_drt is None:
        raise ValueError("mrci_states.mrci must provide drt or drt_work")
    rdm_ws = prepare_mrcisd_rdm_workspace(
        rdm_drt,
        n_act=int(mrci_states.n_act),
        n_virt=int(mrci_states.n_virt),
        nelec=int(mrci_states.nelec),
        twos=int(mrci_states.twos),
        max_virt_e=int(max_virt_e),
    )

    ci_ref_for_solver = [np.asarray(ci_list[int(s)], dtype=np.float64).ravel() for s in mrci_states.states]
    if method_s in {"mrcisd", "ic_mrcisd"}:
        h1e_resid, eri_resid = _rebuild_mrci_integral_payload(scf_out, ref, mrci_states=mrci_states)
    else:
        h1e_resid = None
        eri_resid = None

    # ── Per-state gradient loop ─────────────────────────────────────────────
    grads: list[np.ndarray] = []

    ncore_frozen = int(mrci_states.ncore)
    ncor = int(mrci_states.n_act) + int(mrci_states.n_virt)
    q_block_thc = 256
    pair_p_block_thc = 8

    grad_base_ref: np.ndarray | None = None
    D_tot_ref_ao: np.ndarray | None = None
    bar_L_ref = None
    bar_X_ref = None
    bar_Y_ref = None

    if target_backend == "df":
        gfock_ref, D_core_ref_ao, D_act_ref_ao, D_tot_ref_x, C_act_ref = _build_gfock_casscf_df(
            B_ao_x,
            h_ao_x,
            C_full_x,
            ncore=int(ncore_ref),
            ncas=int(ncas_ref),
            dm1_act=dm1_sa,
            dm2_act=dm2_sa,
        )
        bar_L_ref = _build_bar_L_casscf_df(
            B_ao_x,
            D_core_ao=D_core_ref_ao,
            D_act_ao=D_act_ref_ao,
            C_act=C_act_ref,
            dm2_act=dm2_sa,
            work_dtype=_barl_work_dtype,
            out_dtype=_barl_out_dtype,
            qblock=_barl_qblock,
        )
        D_tot_ref_ao = _asnumpy_f64(D_tot_ref_x)
        if is_spherical:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1_ref = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=np.asarray(D_tot_ref_ao, dtype=np.float64),
                shell_atom=shell_atom,
            )
        else:
            de_h1_ref = contract_dhcore_cart(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M=np.asarray(D_tot_ref_ao, dtype=np.float64),
                shell_atom=shell_atom,
            )
        if is_spherical:
            if df_grad_ctx is not None:
                de_df_ref = df_grad_ctx.contract_sph(B_sph=B_ao_x, bar_L_sph=bar_L_ref, T_c2s=None)
            else:
                de_df_ref = compute_df_gradient_contributions_analytic_sph(
                    ao_basis,
                    aux_basis,
                    atom_coords_bohr=coords,
                    B_sph=B_ao_x,
                    bar_L_sph=bar_L_ref,
                    T_c2s=None,
                    L_chol=L_chol,
                    backend=str(df_backend),
                    df_threads=int(df_threads),
                    profile=None,
                )
        elif df_grad_ctx is not None:
            de_df_ref = df_grad_ctx.contract(B_ao=B_ao_x, bar_L_ao=bar_L_ref)
        else:
            de_df_ref = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao_x,
                bar_L_ao=bar_L_ref,
                L_chol=L_chol,
                backend=str(df_backend),
                df_threads=int(df_threads),
                profile=None,
            )
        grad_base_ref = np.asarray(
            np.asarray(de_h1_ref, dtype=np.float64)
            + _asnumpy_f64(de_df_ref)
            + np.asarray(mol.energy_nuc_grad(), dtype=np.float64),
            dtype=np.float64,
        )
    else:
        from asuka.mcscf.nuc_grad_thc import _build_gfock_casscf_thc  # noqa: PLC0415

        gfock_ref, D_core_ref_ao, D_act_ref_ao, D_tot_ref_x, C_act_ref = _build_gfock_casscf_thc(
            scf_out,
            C=C_full,
            ncore=int(ncore_ref),
            ncas=int(ncas_ref),
            dm1_act=dm1_sa,
            dm2_act=dm2_sa,
            q_block=int(q_block_thc),
            pair_p_block=int(pair_p_block_thc),
        )
        bar_X_ref, bar_Y_ref = _build_bar_xy_target_thc(
            scf_out,
            D_core_ao=D_core_ref_ao,
            D_corr_ao=D_act_ref_ao,
            C_corr=C_act_ref,
            dm2_corr=dm2_sa,
            q_block=int(q_block_thc),
            pair_p_block=int(pair_p_block_thc),
        )
        D_tot_ref_ao = _asnumpy_f64(D_tot_ref_x)
        de_2e_ref = _contract_thc_bar_adjoint(
            scf_out,
            bar_X=bar_X_ref,
            bar_Y=bar_Y_ref,
            df_threads=int(df_threads),
        )
        if is_spherical:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1_ref = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=np.asarray(D_tot_ref_ao, dtype=np.float64),
                shell_atom=shell_atom,
            )
        else:
            de_h1_ref = contract_dhcore_cart(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M=np.asarray(D_tot_ref_ao, dtype=np.float64),
                shell_atom=shell_atom,
            )
        grad_base_ref = np.asarray(
            np.asarray(de_h1_ref, dtype=np.float64)
            + np.asarray(de_2e_ref, dtype=np.float64)
            + np.asarray(mol.energy_nuc_grad(), dtype=np.float64),
            dtype=np.float64,
        )

    for k, _st in enumerate(states_list):
        root = int(roots[k])
        if root < 0 or root >= int(mrci_states.nroots):
            raise ValueError(f"root index out of range: {root}")

        if method_s == "mrcisd":
            ci_mrci = np.asarray(mrci_states.mrci.ci[root], dtype=np.float64).ravel()
        else:
            from asuka.mrci.ic_mrcisd import expand_ic_mrcisd_multi_root  # noqa: PLC0415

            _drt_u, ci_mrci = expand_ic_mrcisd_multi_root(
                mrci_states.mrci,
                ci_cas=ci_ref_for_solver,
                root=int(root),
            )
            ci_mrci = np.asarray(ci_mrci, dtype=np.float64).ravel()
        dm1_corr, dm2_corr = make_rdm12_mrcisd(rdm_ws, ci_mrci, rdm_backend=rdm_backend)
        dm1_corr = np.asarray(dm1_corr, dtype=np.float64)
        dm2_corr = np.asarray(dm2_corr, dtype=np.float64)

        if target_backend == "df":
            gfock, D_core_ao, D_corr_ao, D_tot_ao, C_corr = _build_gfock_casscf_df(
                B_ao_x,
                h_ao_x,
                C_full_x,
                ncore=int(ncore_frozen),
                ncas=int(ncor),
                dm1_act=dm1_corr,
                dm2_act=dm2_corr,
            )

            bar_L_target = _build_bar_L_casscf_df(
                B_ao_x,
                D_core_ao=D_core_ao,
                D_act_ao=D_corr_ao,
                C_act=C_corr,
                dm2_act=dm2_corr,
                work_dtype=_barl_work_dtype,
                out_dtype=_barl_out_dtype,
                qblock=_barl_qblock,
            )

            nocc = int(ncore_frozen + ncor)
            C_occ = C_full[:, :nocc]
            gfock_np = _asnumpy_f64(gfock)
            tmp_w = C_full @ gfock_np[:, :nocc]
            W = 0.5 * (tmp_w @ C_occ.T + C_occ @ tmp_w.T)
        else:
            from asuka.mcscf.nuc_grad_thc import _build_gfock_casscf_thc  # noqa: PLC0415

            gfock, D_core_ao, D_corr_ao, D_tot_ao, C_corr = _build_gfock_casscf_thc(
                scf_out,
                C=C_full,
                ncore=int(ncore_frozen),
                ncas=int(ncor),
                dm1_act=dm1_corr,
                dm2_act=dm2_corr,
                q_block=int(q_block_thc),
                pair_p_block=int(pair_p_block_thc),
            )
            bar_X_target, bar_Y_target = _build_bar_xy_target_thc(
                scf_out,
                D_core_ao=D_core_ao,
                D_corr_ao=D_corr_ao,
                C_corr=C_corr,
                dm2_corr=dm2_corr,
                q_block=int(q_block_thc),
                pair_p_block=int(pair_p_block_thc),
            )
            gfock_np = _asnumpy_f64(gfock)
            de_pulay = np.zeros((natm, 3), dtype=np.float64)

        # Z-vector RHS from orbital gradient matrix 2*(gfock - gfock^T), packed in SA-CASSCF variables.
        g_orb = gfock_np - gfock_np.T
        rhs_orb = 2.0 * np.asarray(mc_sa.pack_uniq_var(g_orb), dtype=np.float64).ravel()

        rhs_ci_z = None
        if method_s == "ic_mrcisd":
            from asuka.mrci.ic_reconstruct import ic_mrcisd_multi_reference_ci_rhs_from_residual  # noqa: PLC0415

            rhs_ci_blocks = ic_mrcisd_multi_reference_ci_rhs_from_residual(
                mrci_states.mrci,
                ci_cas=ci_ref_for_solver,
                root=int(root),
                h1e=h1e_resid,
                eri=eri_resid,
            )
            rhs_ci_full = [np.zeros_like(np.asarray(ci, dtype=np.float64).ravel()) for ci in ci_list]
            for blk_idx, rhs_blk in enumerate(rhs_ci_blocks):
                st_idx = int(mrci_states.states[blk_idx])
                rhs_ci_full[st_idx] = np.asarray(rhs_blk, dtype=np.float64).ravel()
            rhs_ci_z = rhs_ci_full

        if target_backend == "thc":
            Lorb = np.zeros((nmo, nmo), dtype=np.float64)
            dm1_lci = np.zeros((ncas_ref, ncas_ref), dtype=np.float64)
            dm2_lci = np.zeros((ncas_ref, ncas_ref, ncas_ref, ncas_ref), dtype=np.float64)
        else:
            z = solve_mcscf_zvector(
                mc_sa,
                rhs_orb=rhs_orb,
                rhs_ci=rhs_ci_z,
                hessian_op=hess_op,
                tol=float(z_tol),
                maxiter=int(z_maxiter),
            )
            Lvec = np.asarray(z.z_packed, dtype=np.float64).ravel()
            Lorb = mc_sa.unpack_uniq_var(Lvec[:n_orb])
            Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])

            # CI-response transition RDMs (state-averaged weights).
            dm1_lci = np.zeros((ncas_ref, ncas_ref), dtype=np.float64)
            dm2_lci = np.zeros((ncas_ref, ncas_ref, ncas_ref, ncas_ref), dtype=np.float64)
            for r in range(nroots_ref):
                wr = float(np.asarray(weights, dtype=np.float64).ravel()[r])
                if abs(wr) < 1e-14:
                    continue
                dm1_r, dm2_r = fcisolver.trans_rdm12(
                    np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                    np.asarray(ci_list[r], dtype=np.float64).ravel(),
                    int(ncas_ref),
                    nelecas,
                    rdm_backend=str(rdm_backend),
                    return_cupy=False,
                )
                dm1_r = np.asarray(dm1_r, dtype=np.float64)
                dm2_r = np.asarray(dm2_r, dtype=np.float64)
                dm1_lci += wr * (dm1_r + dm1_r.T)
                dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

        if target_backend == "df":
            bar_L_lci_net, D_act_lci = _build_bar_L_net_active_df(
                B_ao_x,
                C_full_x,
                dm1_lci,
                dm2_lci,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
                xp=xp,
                work_dtype=_barl_work_dtype,
                out_dtype=_barl_out_dtype,
                qblock=_barl_qblock,
            )
            bar_L_lorb, D_L_lorb = _build_bar_L_lorb_df(
                B_ao_x,
                C_full_x,
                np.asarray(Lorb, dtype=np.float64),
                dm1_sa,
                dm2_sa,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
                xp=xp,
                work_dtype=_barl_work_dtype,
                out_dtype=_barl_out_dtype,
                qblock=_barl_qblock,
            )

            bar_L_total = xp.asarray(bar_L_target, dtype=_barl_out_dtype)
            bar_L_total += xp.asarray(bar_L_lci_net, dtype=_barl_out_dtype)
            bar_L_total += xp.asarray(bar_L_lorb, dtype=_barl_out_dtype)
            bar_L_delta = bar_L_total - xp.asarray(bar_L_ref, dtype=_barl_out_dtype)
            bar_L_contract = bar_L_delta
            if str(getattr(bar_L_contract, "dtype", "")) != str(np.float64):
                bar_L_contract = xp.asarray(bar_L_contract, dtype=xp.float64)
            del bar_L_target, bar_L_lci_net, bar_L_lorb
            D_h1_delta = _asnumpy_f64(D_tot_ao) - np.asarray(D_tot_ref_ao, dtype=np.float64)
            D_h1_delta += _asnumpy_f64(D_act_lci)
            D_h1_delta += _asnumpy_f64(D_L_lorb)
            del D_act_lci, D_L_lorb

            gfock_lci_raw, _, _, _, _ = _build_gfock_casscf_df(
                B_ao_x,
                h_ao_x,
                C_full_x,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
                dm1_act=np.asarray(dm1_lci, dtype=np.float64),
                dm2_act=np.asarray(dm2_lci, dtype=np.float64),
            )
            h_mo = C_full.T @ h_ao_np @ C_full
            if int(ncore_ref) > 0:
                D_core_zero = 2.0 * (C_full[:, :ncore_ref] @ C_full[:, :ncore_ref].T)
                from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

                Jc_zero_x, Kc_zero_x = _df_scf._df_JK(
                    B_ao_x,
                    xp.asarray(D_core_zero, dtype=xp.float64),
                    want_J=True,
                    want_K=True,
                )
                Jc_zero = _asnumpy_f64(Jc_zero_x)
                Kc_zero = _asnumpy_f64(Kc_zero_x)
                vhf_c_mo = C_full.T @ (Jc_zero - 0.5 * Kc_zero) @ C_full
            else:
                vhf_c_mo = np.zeros_like(h_mo)
            gfock_zero = np.zeros((nmo, nmo), dtype=np.float64)
            if int(ncore_ref) > 0:
                gfock_zero[:, :ncore_ref] = 2.0 * (h_mo + vhf_c_mo)[:, :ncore_ref]
            dgfock_lci = _asnumpy_f64(gfock_lci_raw) - gfock_zero
            dme0_lci = C_full @ (0.5 * (dgfock_lci + dgfock_lci.T)) @ C_full.T
            dme0_lorb = _build_dme0_lorb_response(
                B_ao_np,
                h_ao_np,
                C_full,
                np.asarray(Lorb, dtype=np.float64),
                dm1_sa,
                dm2_sa,
                ppaa_sa,
                papa_sa,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
            )
            W_delta = np.asarray(dme0_lci, dtype=np.float64) + np.asarray(dme0_lorb, dtype=np.float64)

            try:
                if is_spherical:
                    if df_grad_ctx is not None:
                        de_df = df_grad_ctx.contract_sph(B_sph=B_ao_x, bar_L_sph=bar_L_contract, T_c2s=None)
                    else:
                        de_df = compute_df_gradient_contributions_analytic_sph(
                            ao_basis,
                            aux_basis,
                            atom_coords_bohr=coords,
                            B_sph=B_ao_x,
                            bar_L_sph=bar_L_contract,
                            T_c2s=None,
                            L_chol=L_chol,
                            backend=str(df_backend),
                            df_threads=int(df_threads),
                            profile=None,
                        )
                elif df_grad_ctx is not None:
                    de_df = df_grad_ctx.contract(B_ao=B_ao_x, bar_L_ao=bar_L_contract)
                else:
                    de_df = compute_df_gradient_contributions_analytic_packed_bases(
                        ao_basis,
                        aux_basis,
                        atom_coords_bohr=coords,
                        B_ao=B_ao_x,
                        bar_L_ao=bar_L_contract,
                        L_chol=L_chol,
                        backend=str(df_backend),
                        df_threads=int(df_threads),
                        profile=None,
                    )
            except (NotImplementedError, RuntimeError, ValueError):
                de_df = compute_df_gradient_contributions_fd_packed_bases(
                    ao_basis,
                    aux_basis,
                    atom_coords_bohr=coords,
                    bar_L_ao=bar_L_contract,
                    backend=str(df_backend),
                    df_config=df_config,
                    df_threads=int(df_threads),
                    delta_bohr=1e-4,
                    profile=None,
                )
            de_2e_delta = _asnumpy_f64(de_df)
        else:
            bar_X_lci, bar_Y_lci, D_act_lci = _build_bar_xy_net_active_thc(
                scf_out,
                C=C_full,
                dm1_act=dm1_lci,
                dm2_act=dm2_lci,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
                q_block=int(q_block_thc),
                pair_p_block=int(pair_p_block_thc),
            )
            bar_X_lorb, bar_Y_lorb, D_L_lorb = _build_bar_xy_lorb_thc(
                scf_out,
                C=C_full,
                Lorb=np.asarray(Lorb, dtype=np.float64),
                dm1_act=dm1_sa,
                dm2_act=dm2_sa,
                ncore=int(ncore_ref),
                ncas=int(ncas_ref),
                q_block=int(q_block_thc),
                pair_p_block=int(pair_p_block_thc),
            )
            bar_X_total, bar_Y_total = _add_thc_bar_components(bar_X_target, bar_Y_target, bar_X_lci, bar_Y_lci)
            bar_X_total, bar_Y_total = _add_thc_bar_components(bar_X_total, bar_Y_total, bar_X_lorb, bar_Y_lorb)
            bar_X_ref_neg, bar_Y_ref_neg = _scale_thc_bar_components(bar_X_ref, bar_Y_ref, -1.0)
            bar_X_delta, bar_Y_delta = _add_thc_bar_components(
                bar_X_total,
                bar_Y_total,
                bar_X_ref_neg,
                bar_Y_ref_neg,
            )
            D_h1_delta = _asnumpy_f64(D_tot_ao) - np.asarray(D_tot_ref_ao, dtype=np.float64)
            D_h1_delta += _asnumpy_f64(D_act_lci)
            D_h1_delta += _asnumpy_f64(D_L_lorb)
            de_2e_delta = _contract_thc_bar_adjoint(
                scf_out,
                bar_X=bar_X_delta,
                bar_Y=bar_Y_delta,
                df_threads=int(df_threads),
            )
            de_pulay_delta = np.zeros((natm, 3), dtype=np.float64)

        if target_backend == "df":
            if is_spherical:
                from asuka.integrals.int1e_sph import contract_dS_sph  # noqa: PLC0415

                de_pulay_delta = -1.0 * contract_dS_sph(
                    ao_basis,
                    atom_coords_bohr=coords,
                    M_sph=np.asarray(W_delta, dtype=np.float64),
                    shell_atom=shell_atom,
                )
            else:
                de_pulay_delta = -1.0 * contract_dS_cart(
                    ao_basis,
                    atom_coords_bohr=coords,
                    M=np.asarray(W_delta, dtype=np.float64),
                    shell_atom=shell_atom,
                )

        if is_spherical:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1_delta = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=np.asarray(D_h1_delta, dtype=np.float64),
                shell_atom=shell_atom,
            )
        else:
            de_h1_delta = contract_dhcore_cart(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M=np.asarray(D_h1_delta, dtype=np.float64),
                shell_atom=shell_atom,
            )

        grad = np.asarray(
            np.asarray(grad_base_ref, dtype=np.float64)
            + np.asarray(de_h1_delta, dtype=np.float64)
            + np.asarray(de_2e_delta, dtype=np.float64)
            + np.asarray(de_pulay_delta, dtype=np.float64),
            dtype=np.float64,
        )
        grads.append(grad)

    _restore_pool()
    return grads


__all__ = ["mrci_grad_states_from_ref_analytic"]
