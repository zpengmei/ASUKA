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
from asuka.integrals.cart2sph import coerce_sph_map
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


def _normalize_int1e_backend_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError("int1e_contract_backend must be one of {'auto','cpu','cuda'}")
    return mode


def _has_sph_int1e_cuda_kernels() -> bool:
    try:
        from asuka.integrals.int1e_sph_cuda import has_int1e_sph_cuda_kernels  # noqa: PLC0415
    except Exception:
        return False
    try:
        return bool(has_int1e_sph_cuda_kernels())
    except Exception:
        return False


def _select_sph_int1e_backend(*, contract_backend: str, df_backend: str) -> Literal["cpu", "cuda"]:
    mode = _normalize_int1e_backend_mode(contract_backend)
    want_cuda = bool(mode == "cuda" or (mode == "auto" and str(df_backend).strip().lower() == "cuda"))
    if not want_cuda:
        return "cpu"

    if _has_sph_int1e_cuda_kernels():
        return "cuda"

    if mode == "cuda":
        raise RuntimeError(
            "Spherical int1e CUDA backend requested but fused spherical 1e CUDA kernels are unavailable. "
            "Rebuild cuERI CUDA extension with cueri_cuda_kernels_int1e_sph.cu."
        )
    return "cpu"


def _build_sph_int1e_prebuilt(
    ao_basis: Any,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray | None,
    shell_atom: np.ndarray,
    need_overlap: bool,
    need_hcore: bool,
    to_gpu: bool,
) -> tuple[Any | None, Any | None, Any | None]:
    from asuka.integrals.int1e_sph import build_dS_sph, build_dT_sph, build_dV_sph  # noqa: PLC0415

    dS_pre = None
    dT_pre = None
    dV_pre = None

    if bool(need_overlap):
        dS_pre = build_dS_sph(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            shell_atom=shell_atom,
        )
    if bool(need_hcore):
        dT_pre = build_dT_sph(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            shell_atom=shell_atom,
        )
        dV_pre = build_dV_sph(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            atom_charges=np.asarray(atom_charges, dtype=np.float64),
            shell_atom=shell_atom,
            include_operator_deriv=True,
        )

    if bool(to_gpu):
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Spherical int1e CUDA backend requires CuPy") from e
        if dS_pre is not None:
            dS_pre = cp.asarray(dS_pre, dtype=cp.float64)
        if dT_pre is not None:
            dT_pre = cp.asarray(dT_pre, dtype=cp.float64)
        if dV_pre is not None:
            dV_pre = cp.asarray(dV_pre, dtype=cp.float64)

    return dS_pre, dT_pre, dV_pre


def _require_df_mnq_layout(scf_out: Any, *, where: str) -> tuple[int, int, int]:
    """Validate that cached DF factors are in mnQ layout: ``(nao, nao, naux)``."""
    B = getattr(scf_out, "df_B", None)
    if B is None:
        raise ValueError(f"{where} requires scf_out.df_B in df_layout='mnQ' with shape (nao,nao,naux); got None")

    B_shape = getattr(B, "shape", None)
    if B_shape is None or len(B_shape) != 3:
        raise ValueError(
            f"{where} requires scf_out.df_B in df_layout='mnQ' with shape (nao,nao,naux); "
            f"got shape={B_shape!r}"
        )

    nao_int1e = None
    try:
        nao_int1e = int(getattr(getattr(scf_out, "int1e", None), "S").shape[0])
    except Exception:
        nao_int1e = None

    n0, n1, naux = map(int, B_shape)
    if n0 != n1:
        raise ValueError(
            f"{where} requires df_layout='mnQ' (B[mu,nu,Q] with shape (nao,nao,naux)); "
            f"got df_B.shape={tuple(map(int, B_shape))}. If SCF used Qmn, rerun with df_layout='mnQ'."
        )
    if nao_int1e is not None and (n0 != nao_int1e or n1 != nao_int1e):
        raise ValueError(
            f"{where} requires df_layout='mnQ' with AO dimensions matching int1e.S "
            f"(expected nao={int(nao_int1e)}). Got df_B.shape={tuple(map(int, B_shape))}. "
            "Rerun SCF with df_layout='mnQ'."
        )
    return n0, n1, naux


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


def _normalize_df_vram_policy(value: str | None) -> str:
    policy = str(value or "auto").strip().lower()
    if policy in ("", "auto"):
        return "auto"
    if policy in ("off", "0", "false", "no", "disabled"):
        return "off"
    if policy in ("aggressive", "tight", "low"):
        return "aggressive"
    return "auto"


def _apply_df_pool_policy(B_ao: Any, *, label: str = ""):
    """Apply an adaptive CuPy memory-pool cap for DF-heavy phases.

    Returns a restore callback that resets the previous pool limit and flushes
    cached free blocks.
    """
    policy = _normalize_df_vram_policy(_os.environ.get("ASUKA_DF_VRAM_POLICY"))
    if policy == "off":
        return lambda: None

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        return lambda: None

    if B_ao is None or not isinstance(B_ao, cp.ndarray):
        return lambda: None

    try:
        pool = cp.get_default_memory_pool()
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        free_b = int(free_b)
        total_b = int(total_b)
        used_b = max(0, int(total_b) - int(free_b))
        sizeof_B = int(getattr(B_ao, "size", 0)) * 8

        old_limit: int | None = None
        try:
            old_limit = int(pool.get_limit())
        except Exception:
            old_limit = None

        limit_env = _os.environ.get("ASUKA_DF_POOL_LIMIT_GB")
        if limit_env:
            try:
                limit_b = max(1, int(float(limit_env) * 1024**3))
            except Exception:
                limit_b = 0
        else:
            if policy == "aggressive":
                floor_b = int(0.60 * total_b)
                ceil_b = int(0.90 * total_b)
                slack_b = max(int(6 * max(1, sizeof_B)), int(5 * 1024**3))
            else:
                floor_b = int(0.72 * total_b)
                ceil_b = int(0.95 * total_b)
                slack_b = max(int(10 * max(1, sizeof_B)), int(8 * 1024**3))
            limit_b = max(floor_b, min(ceil_b, int(used_b) + int(slack_b)))

        if limit_b > 0:
            pool.set_limit(size=int(limit_b))
            if _os.environ.get("ASUKA_VRAM_DEBUG"):
                print(f"[VRAM_POLICY] {label}: policy={policy} limit={limit_b/1e9:.2f}GB")

    except Exception:
        return lambda: None

    def _restore() -> None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            if old_limit is not None:
                cp.get_default_memory_pool().set_limit(size=int(old_limit))
        except Exception:
            pass

    return _restore


def _normalize_bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    sval = str(value).strip().lower()
    if sval in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if sval in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return bool(default)


def _normalize_sym_kernel_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "enable", "enabled", "force"):
        return "on"
    if mode in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    return "auto"


def _prefer_sym_kernel(*, arr_nbytes: int, nchunks: int) -> bool:
    mode = _normalize_sym_kernel_mode(_os.environ.get("ASUKA_DF_SYM_KERNEL"))
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        import cupy as cp  # noqa: PLC0415

        free_b, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        return False
    tmp_b = max(1, int(arr_nbytes) // max(1, int(nchunks)))
    margin_b = max(int(768 * 1024**2), tmp_b // 4)
    return int(free_b) < int(tmp_b + margin_b)


def _normalize_fused_contract_precision(value: str | None) -> str:
    mode = str(value or "fp64").strip().lower()
    if mode in ("tf32",):
        return "tf32"
    if mode in ("fp32_acc64", "fp32-acc64", "fp32acc64", "mixed_fp32_acc64"):
        return "fp32_acc64"
    return "fp64"


def _apply_df_grad_optimal_defaults() -> dict[str, Any]:
    """Apply the single production DF-gradient preset."""
    applied: dict[str, str] = {}

    def _setdefault_env(name: str, value: str) -> None:
        if name in _os.environ:
            return
        _os.environ[name] = str(value)
        applied[name] = str(value)

    # Single production preset: fused fp64 contraction + scheduler-guided 3c launches.
    _setdefault_env("ASUKA_DF_FUSED_CONTRACT", "1")
    _setdefault_env("ASUKA_DF_FUSED_CONTRACT_PRECISION", "fp64")
    _setdefault_env("ASUKA_DF_3C_SCHED_MODE", "on")
    _setdefault_env("ASUKA_DF_3C_AB_TILE", "8")

    if _os.environ.get("ASUKA_VRAM_DEBUG") or _os.environ.get("ASUKA_PROFILE_DF_PER_ROOT"):
        applied_s = "none"
        if applied:
            applied_s = ",".join(f"{k}={v}" for k, v in sorted(applied.items()))
        print(f"[ASUKA_DF_GRAD_DEFAULTS] preset=optimal applied={applied_s}")

    return {"preset": "optimal", "applied": dict(applied)}


def _normalize_barl_fused_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in ("", "auto"):
        return "auto"
    if mode in ("1", "true", "yes", "on", "enable", "enabled"):
        return "on"
    if mode in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    return "auto"


def _normalize_barl_stage(value: str | None) -> str:
    stage = str(value or "").strip().lower()
    if stage in ("tf32",):
        return "tf32"
    if stage in ("fp32_acc64", "fp32-acc64", "fp32acc64", "mixed_fp32_acc64"):
        return "fp32_acc64"
    if stage in ("fp64", "double", "f64"):
        return "fp64"
    return ""


def _parse_barl_stage_ladder(value: str | None) -> tuple[str, ...]:
    raw = str(value or "tf32,fp32_acc64,fp64").strip()
    out: list[str] = []
    for tok in raw.split(","):
        st = _normalize_barl_stage(tok)
        if st and st not in out:
            out.append(st)
    if "fp64" not in out:
        out.append("fp64")
    if not out:
        out = ["fp64"]
    return tuple(out)


def _barl_stage_dtypes(xp: Any, stage: str) -> tuple[Any, Any]:
    stage_n = _normalize_barl_stage(stage)
    if stage_n == "tf32":
        return xp.float32, xp.float32
    if stage_n == "fp32_acc64":
        return xp.float32, xp.float64
    return xp.float64, xp.float64


def _resolve_barl_hybrid(
    *,
    xp: Any,
    is_gpu: bool,
    naux_hint: int,
) -> dict[str, Any]:
    fused_mode = _normalize_barl_fused_mode(_os.environ.get("ASUKA_DF_BARL_FUSED"))
    hybrid_mode = str(_os.environ.get("ASUKA_DF_HYBRID_MODE", "off")).strip().lower()
    enabled = bool(is_gpu) and fused_mode != "off" and hybrid_mode in (
        "aggressive",
        "on",
        "1",
        "true",
        "yes",
        "enable",
        "enabled",
    )
    ladder = _parse_barl_stage_ladder(_os.environ.get("ASUKA_DF_HYBRID_LADDER"))
    # When hybrid mode is disabled, default to fp64 end-to-end. Low-precision
    # bar_L without a matching low-precision adjoint path can *increase* peak
    # VRAM due to implicit fp32->fp64 casts inside the DF contraction.
    if not bool(enabled):
        ladder = ("fp64",)

    ab_check = _normalize_bool_env(_os.environ.get("ASUKA_DF_BARL_AB_CHECK"), default=True)
    try:
        ab_tol = float(_os.environ.get("ASUKA_DF_BARL_AB_TOL", _os.environ.get("ASUKA_DF_HYBRID_ERROR_ATOL", "1e-6")))
    except Exception:
        ab_tol = 1e-6
    ab_tol = max(0.0, float(ab_tol))
    warn = _normalize_bool_env(_os.environ.get("ASUKA_DF_HYBRID_WARN"), default=True)
    _qblock_env = _os.environ.get("ASUKA_DF_BARL_FUSED_QBLOCK")
    try:
        qblock = int(_qblock_env) if _qblock_env is not None else 0
    except Exception:
        qblock = 0
    qblock_source = "env" if _qblock_env is not None else "auto_naux"
    if qblock <= 0:
        qblock = max(1, int(naux_hint) // 8)

    stage0 = str(ladder[0])
    work_dtype, out_dtype = _barl_stage_dtypes(xp, stage0)
    return {
        "enabled": bool(enabled),
        "fused_mode": str(fused_mode),
        "ladder": tuple(ladder),
        "stage_index": 0,
        "stage": stage0,
        "work_dtype": work_dtype,
        "out_dtype": out_dtype,
        "ab_check": bool(ab_check),
        "ab_tol": float(ab_tol),
        "warn": bool(warn),
        "qblock": int(qblock),
        "qblock_source": str(qblock_source),
        "calibrated": False,
    }


def _advance_barl_stage(policy: dict[str, Any], xp: Any) -> None:
    ladder = tuple(policy.get("ladder", ("fp64",)))
    idx = int(policy.get("stage_index", 0))
    idx = min(max(0, idx + 1), max(0, len(ladder) - 1))
    stage = str(ladder[idx]) if ladder else "fp64"
    work_dtype, out_dtype = _barl_stage_dtypes(xp, stage)
    policy["stage_index"] = int(idx)
    policy["stage"] = str(stage)
    policy["work_dtype"] = work_dtype
    policy["out_dtype"] = out_dtype


def _auto_qblock_for_aux_tile(
    *,
    nao: int,
    naux: int,
    itemsize: int,
    target_bytes: int | None = None,
) -> int:
    """Choose an aux chunk size so (qblock, nao, nao) temporaries fit in VRAM.

    Several bar_L builders allocate intermediates shaped (qblock, nao, nao).
    For large bases this can exceed VRAM even when the final bar_L fits.
    """
    nao = int(nao)
    naux = int(naux)
    itemsize = int(itemsize)
    if target_bytes is None:
        try:
            target_mb = int(_os.environ.get("ASUKA_DF_TILE_AUX_TARGET_MB", "256"))
        except Exception:
            target_mb = 256
        target_mb = max(16, int(target_mb))
        target_bytes = int(target_mb) * 1024**2
    target_bytes = max(1, int(target_bytes))
    denom = max(1, int(nao) * int(nao) * int(itemsize))
    q = int(target_bytes // denom)
    q = max(1, min(int(naux), int(q)))
    return int(q)


_BARL_COULOMB_ADD_KERNELS: dict[str, Any] = {}


def _barl_coulomb_add_inplace(
    bar_Qmn: Any,
    *,
    a_Q: Any,
    M1_mn: Any,
    b_Q: Any,
    M2_mn: Any,
    xp: Any,
) -> Any:
    """In-place: bar[Q,m,n] += a[Q]*M1[m,n] + b[Q]*M2[m,n]."""
    if xp is np:
        bar_Qmn += a_Q[:, None, None] * M1_mn[None, :, :] + b_Q[:, None, None] * M2_mn[None, :, :]
        return bar_Qmn

    import cupy as cp  # noqa: PLC0415

    bar = cp.asarray(bar_Qmn)
    if not bool(getattr(bar, "flags", None).c_contiguous):
        bar = cp.ascontiguousarray(bar)
    dt = cp.dtype(bar.dtype)
    if dt not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
        raise ValueError("bar_L Coulomb kernel supports only float32/float64")

    a = cp.asarray(a_Q, dtype=dt).ravel()
    b = cp.asarray(b_Q, dtype=dt).ravel()
    m1 = cp.asarray(M1_mn, dtype=dt).ravel()
    m2 = cp.asarray(M2_mn, dtype=dt).ravel()

    naux = int(bar.shape[0])
    nao = int(bar.shape[1])
    nao2 = int(nao * nao)
    if int(a.size) != naux or int(b.size) != naux:
        raise ValueError("a_Q/b_Q length mismatch")
    if int(m1.size) != nao2 or int(m2.size) != nao2:
        raise ValueError("M1/M2 shape mismatch")

    key = str(dt)
    k = _BARL_COULOMB_ADD_KERNELS.get(key)
    if k is None:
        if dt == cp.dtype(cp.float32):
            k = cp.ElementwiseKernel(
                "raw float32 a, raw float32 m1, raw float32 b, raw float32 m2, int64 nao2",
                "float32 y",
                "const long long q = (long long)(i / nao2); const long long mn = (long long)(i - q * nao2); y += a[q] * m1[mn] + b[q] * m2[mn];",
                "asuka_barl_coulomb_add_f32",
            )
        else:
            k = cp.ElementwiseKernel(
                "raw float64 a, raw float64 m1, raw float64 b, raw float64 m2, int64 nao2",
                "float64 y",
                "const long long q = (long long)(i / nao2); const long long mn = (long long)(i - q * nao2); y += a[q] * m1[mn] + b[q] * m2[mn];",
                "asuka_barl_coulomb_add_f64",
            )
        _BARL_COULOMB_ADD_KERNELS[key] = k

    k(a, m1, b, m2, int(nao2), bar.reshape(-1))
    return bar


def _symmetrize_bar_L_inplace(bar, xp, nchunks: int | None = None) -> None:
    """Compute bar = 0.5 * (bar + bar^T) in-place, chunked over the aux index.

    This avoids allocating a full (naux, nao, nao) temporary for the transpose,
    reducing the peak VRAM from 2 sizeof_B to sizeof_B + sizeof_B/nchunks.
    The (0, 2, 1) transpose swaps the last two (AO) dims independently per Q-slice,
    so chunking over Q is safe.
    """
    if nchunks is None:
        try:
            nchunks = int(_os.environ.get("ASUKA_DF_SYM_CHUNKS", "4"))
        except Exception:
            nchunks = 4
    nchunks = max(1, int(nchunks))

    # CUDA fast path: use extension kernel to avoid transpose-copy temporaries.
    if xp is not np:
        try:
            import cupy as cp  # noqa: PLC0415

            if xp is cp and isinstance(bar, cp.ndarray) and int(bar.ndim) == 3:
                naux_i, nao0, nao1 = map(int, bar.shape)
                if nao0 == nao1 and bar.dtype == cp.float64 and bool(getattr(bar, "flags", None).c_contiguous):
                    if _prefer_sym_kernel(arr_nbytes=int(bar.nbytes), nchunks=int(nchunks)):
                        from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415

                        _ext.df_symmetrize_qmn_inplace_device(
                            bar.reshape(-1),
                            int(naux_i),
                            int(nao0),
                            256,
                            int(cp.cuda.get_current_stream().ptr),
                            False,
                        )
                        return
        except Exception:
            pass

    naux = int(bar.shape[0])
    _chunk = max(1, naux // nchunks)
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _slc = bar[_q0:_q1].copy()
        bar[_q0:_q1] += xp.transpose(_slc, (0, 2, 1))
        del _slc
    bar *= 0.5


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
    sph_map = coerce_sph_map(getattr(scf_out, "sph_map", None))
    if sph_map is None:
        raise ValueError("scf_out.sph_map is required for spherical AO gradients (cart=False)")
    return sph_map.T_c2s  # (nao_cart, nao_sph)


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


_GPU_PRECISION_POLICIES = ("fp64", "mixed_conservative", "mixed_aggressive")


@dataclass(frozen=True)
class GpuRuntimeConfig:
    memory_cap_gb: float | None = None
    workspace_cache_fraction: float | None = None
    precision_policy: str | None = None
    fused_df: bool | None = None
    fused_matvec: bool | None = None
    krylov_recycle_max_vectors: int | None = None
    tile_aux: int = 0
    tile_rows: int = 0


def _parse_env_float(name: str) -> float | None:
    raw = _os.environ.get(name)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def _parse_env_int(name: str) -> int | None:
    raw = _os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _normalize_precision_policy(value: str | None) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    aliases = {
        "off": "fp64",
        "none": "fp64",
        "double": "fp64",
        "conservative": "mixed_conservative",
        "balanced": "mixed_conservative",
        "aggressive": "mixed_aggressive",
        "mixed": "mixed_conservative",
    }
    mode = aliases.get(mode, mode)
    if mode not in _GPU_PRECISION_POLICIES:
        return None
    return mode


def _normalize_ws_cache_fraction_local(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(max(0.0, min(0.8, float(value))))


def _resolve_gpu_runtime_config() -> GpuRuntimeConfig:
    memory_cap = _parse_env_float("ASUKA_GPU_MEM_CAP_GB")
    ws_cache_frac = _normalize_ws_cache_fraction_local(_parse_env_float("ASUKA_GPU_WS_CACHE_FRAC"))
    precision_policy = _normalize_precision_policy(_os.environ.get("ASUKA_GPU_PRECISION_POLICY"))

    fused_df = None
    if _os.environ.get("ASUKA_GPU_FUSED_DF") is not None:
        fused_df = _normalize_bool_env(_os.environ.get("ASUKA_GPU_FUSED_DF"), default=False)

    fused_matvec = None
    if _os.environ.get("ASUKA_GPU_FUSED_MATVEC") is not None:
        fused_matvec = _normalize_bool_env(_os.environ.get("ASUKA_GPU_FUSED_MATVEC"), default=True)

    recycle_max = _parse_env_int("ASUKA_GPU_KRYLOV_RECYCLE_MAX")
    if recycle_max is not None and recycle_max <= 0:
        recycle_max = None

    tile_aux = _parse_env_int("ASUKA_GPU_TILE_AUX")
    if tile_aux is None:
        tile_aux = 0
    tile_rows = _parse_env_int("ASUKA_GPU_TILE_ROWS")
    if tile_rows is None:
        tile_rows = 0

    return GpuRuntimeConfig(
        memory_cap_gb=memory_cap,
        workspace_cache_fraction=ws_cache_frac,
        precision_policy=precision_policy,
        fused_df=fused_df,
        fused_matvec=fused_matvec,
        krylov_recycle_max_vectors=recycle_max,
        tile_aux=max(0, int(tile_aux)),
        tile_rows=max(0, int(tile_rows)),
    )


def _apply_gpu_runtime_config_to_solver(fcisolver: Any, cfg: GpuRuntimeConfig) -> None:
    if cfg.memory_cap_gb is not None:
        setattr(fcisolver, "matvec_cuda_memory_cap_gib", float(cfg.memory_cap_gb))
        setattr(fcisolver, "matvec_cuda_mem_hard_cap_gib", float(cfg.memory_cap_gb))
    if cfg.workspace_cache_fraction is not None:
        setattr(fcisolver, "matvec_cuda_ws_cache_fraction", float(cfg.workspace_cache_fraction))
    if cfg.tile_rows > 0:
        setattr(fcisolver, "matvec_cuda_j_tile", int(cfg.tile_rows))
    if cfg.fused_matvec is not None:
        setattr(fcisolver, "matvec_cuda_use_fused_hop", bool(cfg.fused_matvec))
        if bool(cfg.fused_matvec):
            setattr(fcisolver, "matvec_cuda_path_mode", "fused_epq_hybrid")
        else:
            setattr(fcisolver, "matvec_cuda_path_mode", "epq_blocked")

    if cfg.precision_policy == "fp64":
        setattr(fcisolver, "matvec_cuda_dtype", "float64")
        setattr(fcisolver, "approx_cuda_dtype", "float64")
    elif cfg.precision_policy == "mixed_conservative":
        setattr(fcisolver, "matvec_cuda_dtype", "mixed")
        setattr(fcisolver, "approx_cuda_dtype", "float32")
        setattr(fcisolver, "matvec_cuda_mixed_threshold", 1e-3)
        setattr(fcisolver, "matvec_cuda_mixed_low_precision_max_iter", 1)
        setattr(fcisolver, "matvec_cuda_mixed_force_final_full_hop", True)
    elif cfg.precision_policy == "mixed_aggressive":
        setattr(fcisolver, "matvec_cuda_dtype", "mixed")
        setattr(fcisolver, "approx_cuda_dtype", "float32")
        setattr(fcisolver, "matvec_cuda_mixed_threshold", 1e-6)
        setattr(fcisolver, "matvec_cuda_mixed_low_precision_max_iter", 4)
        setattr(fcisolver, "matvec_cuda_mixed_force_final_full_hop", True)
        setattr(fcisolver, "matvec_cuda_gemm_backend", "cublaslt_tf32")

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
    vhf_cache: dict | None = None,
    lorb_cache: dict | None = None,
    return_xp: bool = False,
    return_parts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Compute response energy-weighted density from orbital Lagrange multiplier.

    Follows PySCF's ``Lorb_dot_dgorb_dx`` gfock construction.
    Returns ``dme0 = 0.5*(gfock + gfock^T)`` in AO-like basis (nao, nao).
    GPU-aware: keeps J/K builds on device when inputs are CuPy arrays.

    Parameters
    ----------
    vhf_cache : dict, optional
        Precomputed root-invariant J/K results. Keys: ``"vhf_c"``, ``"vhf_a"``.
        When provided, skips the 2 root-invariant ``_df_JK`` calls (Jc/Kc from
        D_core and Ja/Ka from D_act), saving ~40% of per-root DF work.
    lorb_cache : dict, optional
        Optional cache for root-invariant CPU intermediates used by the
        active-active response terms. Recognized keys:
        ``"ppaa_np"``, ``"papa_np"``, ``"aapa"`` and GPU variants
        ``"ppaa_xp"``, ``"papa_xp"``, ``"aapa_xp"``.
    return_xp : bool, optional
        When True, return the matrix in the native ``xp`` backend dtype
        (float64). When False (default), return a NumPy float64 array.
    return_parts : bool, optional
        When True, also return a dictionary with symmetrized AO-channel
        components (h1/mean-field/aa1/aa2/total) for response debugging.
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
    # Use cached root-invariant Jc/Kc and Ja/Ka when available.
    if vhf_cache is not None and "vhf_c" in vhf_cache and "vhf_a" in vhf_cache:
        vhf_c = vhf_cache["vhf_c"]
        vhf_a = vhf_cache["vhf_a"]
    else:
        Jc, Kc = _df_scf._df_JK(B_x, D_core, want_J=True, want_K=True)  # noqa: SLF001
        Ja, Ka = _df_scf._df_JK(B_x, D_act, want_J=True, want_K=True)  # noqa: SLF001
        vhf_c = Jc - 0.5 * Kc
        vhf_a = Ja - 0.5 * Ka
    if ncore:
        JcL, KcL = _df_scf._df_JK(B_x, D_L_core, want_J=True, want_K=True)  # noqa: SLF001
    else:
        JcL = xp.zeros((nao, nao), dtype=xp.float64)
        KcL = xp.zeros((nao, nao), dtype=xp.float64)
    JaL, KaL = _df_scf._df_JK(B_x, D_L_act, want_J=True, want_K=True)  # noqa: SLF001

    vhfL_c = JcL - 0.5 * KcL
    vhfL_a = JaL - 0.5 * KaL

    # Build gfock in AO basis (PySCF Lorb_dot_dgorb_dx formula).
    import os

    _reco_mode = str(os.environ.get("ASUKA_CASPT2_LORB_RECO_MODE", "pyscf")).strip().lower()
    if _reco_mode not in {"pyscf", "core_only", "core_minus_active"}:
        _reco_mode = "pyscf"
    if _reco_mode == "core_only":
        _vhf_mix = vhf_c
        _vhfL_mix = vhfL_c
    elif _reco_mode == "core_minus_active":
        _vhf_mix = vhf_c - vhf_a
        _vhfL_mix = vhfL_c - vhfL_a
    else:
        _vhf_mix = vhf_c + vhf_a
        _vhfL_mix = vhfL_c + vhfL_a

    g_h1 = h_ao_x @ D_L
    g_vmix = _vhf_mix @ D_L_core
    g_vLmix = _vhfL_mix @ D_core
    g_vL_c = vhfL_c @ D_act
    g_vc_L = vhf_c @ D_L_act

    # S^{-1} ≈ CC^T (orthogonal MO basis).
    s0_inv = C @ C.T
    g_h1 = s0_inv @ g_h1
    g_vmix = s0_inv @ g_vmix
    g_vLmix = s0_inv @ g_vLmix
    g_vL_c = s0_inv @ g_vL_c
    g_vc_L = s0_inv @ g_vc_L
    gfock = g_h1 + g_vmix + g_vLmix + g_vL_c + g_vc_L

    # 2e active-active ERI contribution (from ppaa/papa).
    cache = lorb_cache if isinstance(lorb_cache, dict) else None
    if xp is np:
        if cache is not None and cache.get("ppaa_np") is not None:
            ppaa_np = np.asarray(cache["ppaa_np"], dtype=np.float64)
        else:
            ppaa_np = np.asarray(_asnumpy_f64(ppaa), dtype=np.float64)
            if cache is not None:
                cache["ppaa_np"] = ppaa_np
        if cache is not None and cache.get("papa_np") is not None:
            papa_np = np.asarray(cache["papa_np"], dtype=np.float64)
        else:
            papa_np = np.asarray(_asnumpy_f64(papa), dtype=np.float64)
            if cache is not None:
                cache["papa_np"] = papa_np

        L_np = _asnumpy_f64(L)
        L_act_slice = L_np[:, ncore:nocc]

        if cache is not None and cache.get("aapa") is not None:
            aapa = np.asarray(cache["aapa"], dtype=np.float64)
        else:
            aapa = np.asarray(ppaa_np[:, ncore:nocc, :, :].transpose(2, 3, 0, 1), dtype=np.float64)
            if cache is not None:
                cache["aapa"] = aapa

        # Vectorized replacement for the per-i Python loop:
        #   aapaL[:, :, i, :] = sum_j ppaa[i,j,:,:] L[j,:]
        #                     + sym_{u,v} sum_j papa[i,u,j,:] L[j,v]
        aapaL_j = np.einsum("ijuv,jt->uvit", ppaa_np, L_act_slice, optimize=True)
        aapaL_k = np.einsum("iujw,jt->wtiu", papa_np, L_act_slice, optimize=True)
        aapaL = np.asarray(aapaL_j + aapaL_k + aapaL_k.transpose(1, 0, 2, 3), dtype=np.float64)

        dm2_np = _asnumpy_f64(dm2)
        t1 = np.einsum("uviw,uvtw->it", aapaL, dm2_np, optimize=True)
        t2 = np.einsum("uviw,vuwt->it", aapa, dm2_np, optimize=True)
        C_np = _asnumpy_f64(C)
        C_act_np = _asnumpy_f64(C_act)
        C_L_act_np = _asnumpy_f64(C_L_act)
        gfock_np = _asnumpy_f64(gfock)
        g_aa1_np = np.asarray(C_np @ t1 @ C_act_np.T, dtype=np.float64)
        g_aa2_np = np.asarray(C_np @ t2 @ C_L_act_np.T, dtype=np.float64)
        gfock_np += g_aa1_np
        gfock_np += g_aa2_np
        out_np = np.asarray(0.5 * (gfock_np + gfock_np.T), dtype=np.float64)
        if bool(return_parts):
            g_h1_np = _asnumpy_f64(g_h1)
            g_vmix_np = _asnumpy_f64(g_vmix)
            g_vLmix_np = _asnumpy_f64(g_vLmix)
            g_vL_c_np = _asnumpy_f64(g_vL_c)
            g_vc_L_np = _asnumpy_f64(g_vc_L)
            g_mean_np = np.asarray(g_vmix_np + g_vLmix_np + g_vL_c_np + g_vc_L_np, dtype=np.float64)
            parts_np: dict[str, Any] = {
                "dme0_h1": np.asarray(0.5 * (g_h1_np + g_h1_np.T), dtype=np.float64),
                "dme0_mean": np.asarray(0.5 * (g_mean_np + g_mean_np.T), dtype=np.float64),
                "dme0_aa1": np.asarray(0.5 * (g_aa1_np + g_aa1_np.T), dtype=np.float64),
                "dme0_aa2": np.asarray(0.5 * (g_aa2_np + g_aa2_np.T), dtype=np.float64),
                "dme0_total": np.asarray(out_np, dtype=np.float64),
            }
            if return_xp:
                return np.asarray(out_np, dtype=np.float64), parts_np
            return out_np, parts_np
        if return_xp:
            return np.asarray(out_np, dtype=np.float64)
        return out_np

    # GPU path: keep the full response construction on device.
    if cache is not None and cache.get("ppaa_xp") is not None:
        ppaa_x = xp.asarray(cache["ppaa_xp"], dtype=xp.float64)
    else:
        ppaa_x = xp.asarray(ppaa, dtype=xp.float64)
        if cache is not None:
            cache["ppaa_xp"] = ppaa_x
    if cache is not None and cache.get("papa_xp") is not None:
        papa_x = xp.asarray(cache["papa_xp"], dtype=xp.float64)
    else:
        papa_x = xp.asarray(papa, dtype=xp.float64)
        if cache is not None:
            cache["papa_xp"] = papa_x

    if cache is not None and cache.get("aapa_xp") is not None:
        aapa_x = xp.asarray(cache["aapa_xp"], dtype=xp.float64)
    else:
        aapa_x = xp.asarray(ppaa_x[:, ncore:nocc, :, :].transpose(2, 3, 0, 1), dtype=xp.float64)
        if cache is not None:
            cache["aapa_xp"] = aapa_x

    L_act_slice_x = L[:, ncore:nocc]
    aapaL_j_x = xp.einsum("ijuv,jt->uvit", ppaa_x, L_act_slice_x, optimize=True)
    aapaL_k_x = xp.einsum("iujw,jt->wtiu", papa_x, L_act_slice_x, optimize=True)
    aapaL_x = aapaL_j_x + aapaL_k_x + aapaL_k_x.transpose(1, 0, 2, 3)

    t1_x = xp.einsum("uviw,uvtw->it", aapaL_x, dm2, optimize=True)
    t2_x = xp.einsum("uviw,vuwt->it", aapa_x, dm2, optimize=True)
    g_aa1_x = C @ t1_x @ C_act.T
    g_aa2_x = C @ t2_x @ C_L_act.T
    gfock += g_aa1_x
    gfock += g_aa2_x
    out_x = xp.asarray(0.5 * (gfock + gfock.T), dtype=xp.float64)
    if bool(return_parts):
        g_mean_x = g_vmix + g_vLmix + g_vL_c + g_vc_L
        parts_x: dict[str, Any] = {
            "dme0_h1": xp.asarray(0.5 * (g_h1 + g_h1.T), dtype=xp.float64),
            "dme0_mean": xp.asarray(0.5 * (g_mean_x + g_mean_x.T), dtype=xp.float64),
            "dme0_aa1": xp.asarray(0.5 * (g_aa1_x + g_aa1_x.T), dtype=xp.float64),
            "dme0_aa2": xp.asarray(0.5 * (g_aa2_x + g_aa2_x.T), dtype=xp.float64),
            "dme0_total": xp.asarray(out_x, dtype=xp.float64),
        }
        if return_xp:
            return out_x, parts_x
        return _asnumpy_f64(out_x), {k: _asnumpy_f64(v) for k, v in parts_x.items()}
    if return_xp:
        return out_x
    return _asnumpy_f64(out_x)


def _build_bar_L_casscf_df(
    B_ao: np.ndarray,
    *,
    D_core_ao: np.ndarray,
    D_act_ao: np.ndarray,
    C_act: np.ndarray,
    dm2_act: np.ndarray,
    L_act: Any | None = None,
    rho_core: Any | None = None,
    work_dtype: Any | None = None,
    out_dtype: Any | None = None,
    qblock: int | None = None,
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
    _wd = xp.float64 if work_dtype is None else work_dtype
    _od = xp.float64 if out_dtype is None else out_dtype
    B_ao = xp.asarray(B_ao, dtype=_wd)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_core_ao = xp.asarray(D_core_ao, dtype=_wd)
    D_act_ao = xp.asarray(D_act_ao, dtype=_wd)
    if D_core_ao.shape != (nao, nao) or D_act_ao.shape != (nao, nao):
        raise ValueError("AO density shape mismatch")

    # Mean-field interaction with the core potential:
    #   E_cc + E_ca = Tr(D_w * (J(Dc) - 0.5 K(Dc))), where D_w = D_act + 0.5 D_core
    D_w = D_act_ao + 0.5 * D_core_ao

    B2 = B_ao.reshape(nao * nao, naux)
    if rho_core is None:
        rho = B2.T @ D_core_ao.reshape(nao * nao)  # (naux,)
    else:
        rho = xp.asarray(rho_core, dtype=_wd)
        if rho.shape != (naux,):
            raise ValueError("rho_core shape mismatch")
    sigma = B2.T @ D_w.reshape(nao * nao)  # (naux,)

    # Coulomb-like part without materializing full-size broadcast temporaries.
    _log_vram("    casscf_df: before bar_mean")
    bar_mean = xp.zeros((naux, nao, nao), dtype=_od)
    _barl_coulomb_add_inplace(
        bar_mean,
        a_Q=sigma,
        M1_mn=D_core_ao,
        b_Q=rho,
        M2_mn=D_w,
        xp=xp,
    )
    _log_vram("    casscf_df: after bar_mean J")

    BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao) — view, no copy
    # Chunk over aux index so intermediates are (chunk,nao,nao) instead of
    # (naux,nao,nao), reducing peak VRAM by ~2×sizeof(B).
    _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _bq = BQ[_q0:_q1]
        _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_w)
        _t += xp.matmul(xp.matmul(D_w[None, :, :], _bq), D_core_ao)
        _t *= -0.5
        bar_mean[_q0:_q1] += _t.astype(_od, copy=False)
        del _t
    _log_vram("    casscf_df: after exchange")

    # Active-active 2-RDM term:
    #   E_aa = 0.5 Σ_{uvwx} dm2_uvwx (uv|wx)
    # with (uv|wx) ≈ Σ_Q L_uv,Q L_wx,Q and L_uv,Q = C^T B_Q C in active MO space.
    C_act = xp.asarray(C_act, dtype=_wd)
    if C_act.ndim != 2 or int(C_act.shape[0]) != int(nao):
        raise ValueError("C_act shape mismatch")
    ncas = int(C_act.shape[1])
    if ncas <= 0:
        raise ValueError("empty active space")

    dm2_arr = xp.asarray(dm2_act, dtype=_wd)
    if dm2_arr.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    if L_act is None:
        # L_uv,Q in active MO indices
        X = xp.einsum("mnQ,nv->mvQ", B_ao, C_act, optimize=True)
        L_act = xp.einsum("mu,mvQ->uvQ", C_act, X, optimize=True)  # (ncas,ncas,naux)
    else:
        L_act = xp.asarray(L_act, dtype=_wd)
        if L_act.shape != (ncas, ncas, naux):
            raise ValueError("L_act shape mismatch")

    L2 = L_act.reshape(ncas * ncas, naux)
    dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2  # (ncas^2,naux), M_uv,Q = sum_{wx} dm2_uvwx L_wx,Q
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    # Accumulate bar_act into bar_mean in chunks to avoid sizeof_B temporary.
    _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_act, optimize=True)
        bar_mean[_q0:_q1] += _blk.astype(_od, copy=False)
        del _blk
    del tmp
    _log_vram("    casscf_df: after bar_act")

    _symmetrize_bar_L_inplace(bar_mean, xp)
    _log_vram("    casscf_df: after sym")
    return xp.asarray(bar_mean, dtype=_od)


def _build_bar_L_delta_casscf_df(
    B_ao: Any,
    *,
    D_core_ao: Any,
    C_act: Any,
    dm1_delta: Any,
    dm2_delta: Any,
    rho_core: Any,
    L2: Any,
    out: Any | None = None,
    symmetrize: bool = True,
    work_dtype: Any | None = None,
    out_dtype: Any | None = None,
    qblock: int | None = None,
) -> Any:
    """Return Hellmann–Feynman DF derivative delta ``bar_L_K - bar_L_SA``.

    This delta is linear in the active-space RDM deltas and avoids constructing
    the full per-root ``bar_L_K`` only to subtract ``bar_L_SA``.

    Assumes the core density is root-invariant (shared orbitals), so core-core
    contributions cancel exactly in the delta.
    """

    xp, _ = _get_xp_arrays(B_ao, D_core_ao, C_act, dm1_delta, dm2_delta, rho_core, L2, out)
    _wd = xp.float64 if work_dtype is None else work_dtype
    _od = xp.float64 if out_dtype is None else out_dtype
    B_ao = xp.asarray(B_ao, dtype=_wd)
    D_core_ao = xp.asarray(D_core_ao, dtype=_wd)
    C_act = xp.asarray(C_act, dtype=_wd)
    dm1_delta = xp.asarray(dm1_delta, dtype=_wd)
    dm2_delta = xp.asarray(dm2_delta, dtype=_wd)
    rho_core = xp.asarray(rho_core, dtype=_wd)
    L2 = xp.asarray(L2, dtype=_wd)

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

    if out is None:
        bar_mean_delta = xp.zeros((naux, nao, nao), dtype=_od)
    else:
        bar_mean_delta = xp.asarray(out, dtype=_od)
        if bar_mean_delta.shape != (naux, nao, nao):
            raise ValueError("out shape mismatch")

    # Fused bar_J_delta + bar_K_delta without materializing full-size broadcast temporaries.
    _barl_coulomb_add_inplace(
        bar_mean_delta,
        a_Q=delta_sigma,
        M1_mn=D_core_ao,
        b_Q=rho_core,
        M2_mn=D_act_delta,
        xp=xp,
    )

    BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
    _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _bq = BQ[_q0:_q1]
        _t = xp.matmul(xp.matmul(D_core_ao[None, :, :], _bq), D_act_delta)
        _t += xp.matmul(xp.matmul(D_act_delta[None, :, :], _bq), D_core_ao)
        _t *= -0.5
        bar_mean_delta[_q0:_q1] += _t.astype(_od, copy=False)
        del _t

    # Active-active 2-RDM delta (linear in dm2)
    dm2_flat = dm2_delta.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)
    M = dm2_flat @ L2  # (ncas^2,naux)
    M_uvQ = M.reshape(ncas, ncas, naux)

    tmp = xp.einsum("mu,uvQ->mvQ", C_act, M_uvQ, optimize=True)  # (nao,ncas,naux)
    # Accumulate bar_act_delta into bar_mean_delta in chunks.
    _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
    for _q0 in range(0, naux, _chunk):
        _q1 = min(_q0 + _chunk, naux)
        _blk = xp.einsum("mvQ,nv->Qmn", tmp[:, :, _q0:_q1], C_act, optimize=True)
        bar_mean_delta[_q0:_q1] += _blk.astype(_od, copy=False)
        del _blk
    del tmp

    if bool(symmetrize):
        _symmetrize_bar_L_inplace(bar_mean_delta, xp)
    return xp.asarray(bar_mean_delta, dtype=_od)


def _build_bar_L_df_cross(
    B_ao: np.ndarray,
    *,
    D_left: np.ndarray,
    D_right: np.ndarray,
    coeff_J: float,
    coeff_K: float,
    out: Any | None = None,
    symmetrize: bool = True,
    work_dtype: Any | None = None,
    out_dtype: Any | None = None,
    qblock: int | None = None,
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
    out : Any | None, optional
        Optional output buffer with shape ``(naux,nao,nao)``. If provided,
        contributions are accumulated in-place.
    symmetrize : bool, optional
        Whether to symmetrize in-place before returning.

    Returns
    -------
    np.ndarray
        Contribution to bar_L.
    """

    xp, _ = _get_xp_arrays(B_ao)
    _wd = xp.float64 if work_dtype is None else work_dtype
    _od = xp.float64 if out_dtype is None else out_dtype
    B_ao = xp.asarray(B_ao, dtype=_wd)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao, nao1, naux = map(int, B_ao.shape)
    if nao != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    D_left = xp.asarray(D_left, dtype=_wd)
    D_right = xp.asarray(D_right, dtype=_wd)
    if D_left.shape != (nao, nao) or D_right.shape != (nao, nao):
        raise ValueError("D_left/D_right shape mismatch")

    # Coulomb-like part: Tr(D_left J(D_right)) = u(D_left)·u(D_right)
    B2 = B_ao.reshape(nao * nao, naux)
    rho = B2.T @ D_right.reshape(nao * nao)  # (naux,)
    sigma = B2.T @ D_left.reshape(nao * nao)  # (naux,)

    _log_vram("    cross: before Coulomb")
    cJ = float(coeff_J)
    if out is None:
        bar = xp.zeros((naux, nao, nao), dtype=_od)
    else:
        bar = xp.asarray(out, dtype=_od)
        if bar.shape != (naux, nao, nao):
            raise ValueError("out shape mismatch")
    if cJ:
        _barl_coulomb_add_inplace(
            bar,
            a_Q=(cJ * sigma),
            M1_mn=D_right,
            b_Q=(cJ * rho),
            M2_mn=D_left,
            xp=xp,
        )
    _log_vram("    cross: after Coulomb")

    # Exchange-like part: Tr(D_left K(D_right)) = Σ_Q Tr(D_left B_Q D_right B_Q)
    # Chunked over aux index to keep intermediates at (chunk,nao,nao).
    cK = float(coeff_K)
    if cK:
        BQ = xp.transpose(B_ao, (2, 0, 1))  # (naux,nao,nao)
        _chunk = int(max(1, qblock if qblock is not None else (naux // 4)))
        for _q0 in range(0, naux, _chunk):
            _q1 = min(_q0 + _chunk, naux)
            _bq = BQ[_q0:_q1]
            _t = xp.matmul(xp.matmul(D_left[None, :, :], _bq), D_right)
            _t += xp.matmul(xp.matmul(D_right[None, :, :], _bq), D_left)
            _t *= cK
            bar[_q0:_q1] += _t.astype(_od, copy=False)
            del _t
        _log_vram("    cross: after exchange")

    if bool(symmetrize):
        _symmetrize_bar_L_inplace(bar, xp)
        _log_vram("    cross: after sym")
    return xp.asarray(bar, dtype=_od)


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

    t0_total = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.clear()
        profile["df_threads"] = int(df_threads)

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    _require_df_mnq_layout(scf_out, where="casscf_nuc_grad_df")
    is_spherical = not bool(getattr(mol, "cart", True))

    # Keep default behavior convenient on CUDA runs:
    # - Cartesian: promote CPU -> CUDA
    # - Spherical: promote CPU -> AUTO (uses CUDA fused kernels when available)
    if str(int1e_contract_backend).strip().lower() == "cpu" and str(df_backend).strip().lower() == "cuda":
        int1e_contract_backend = "auto" if bool(is_spherical) else "cuda"

    _int1e_mode = _normalize_int1e_backend_mode(str(int1e_contract_backend))
    _sph_int1e_backend: Literal["cpu", "cuda"] = "cpu"
    if bool(is_spherical):
        _sph_int1e_backend = _select_sph_int1e_backend(
            contract_backend=str(int1e_contract_backend),
            df_backend=str(df_backend),
        )

    if profile is not None:
        profile["df_backend"] = str(df_backend).strip().lower()
        profile["int1e_contract_backend"] = _int1e_mode
        if bool(is_spherical):
            profile["int1e_sph_backend"] = str(_sph_int1e_backend)

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
    _restore_pool = _apply_df_pool_policy(B_ao, label="casscf_nuc_grad_df")

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
    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        t0_df = time.perf_counter() if profile is not None else 0.0
        if is_spherical:
            de_df = compute_df_gradient_contributions_analytic_sph(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_sph=B_ao,
                bar_L_sph=bar_L_ao,
                T_c2s=None,
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
        if is_spherical:
            raise
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

    # ── CPU phase: 1e AO derivative contractions ──
    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    _dS_sph_pre = None
    _dT_sph_pre = None
    _dV_sph_pre = None
    if bool(is_spherical) and str(_sph_int1e_backend) == "cuda":
        _dS_sph_pre, _dT_sph_pre, _dV_sph_pre = _build_sph_int1e_prebuilt(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            shell_atom=shell_atom,
            need_overlap=True,
            need_hcore=True,
            to_gpu=True,
        )

    if is_spherical:
        if str(_sph_int1e_backend) == "cuda":
            from asuka.integrals.int1e_sph_cuda import contract_dhcore_sph_prebuilt_cuda  # noqa: PLC0415

            de_h1 = contract_dhcore_sph_prebuilt_cuda(
                _dT_sph_pre,
                _dV_sph_pre,
                D_tot_ao,
            )
        else:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1 = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=_asnumpy_f64(D_tot_ao),
                shell_atom=shell_atom,
            )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=_asnumpy_f64(D_tot_ao),
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
    if is_spherical:
        if str(_sph_int1e_backend) == "cuda":
            from asuka.integrals.int1e_sph_cuda import contract_dS_sph_prebuilt_cuda  # noqa: PLC0415

            de_pulay = -1.0 * contract_dS_sph_prebuilt_cuda(
                _dS_sph_pre,
                W,
            )
        else:
            from asuka.integrals.int1e_sph import contract_dS_sph  # noqa: PLC0415

            de_pulay = -1.0 * contract_dS_sph(
                ao_basis,
                atom_coords_bohr=coords,
                M_sph=np.asarray(W, dtype=np.float64),
                shell_atom=shell_atom,
            )
    else:
        de_pulay = -1.0 * contract_dS_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=np.asarray(W, dtype=np.float64),
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

    _restore_pool()
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
    int1e_contract_backend: Literal["auto", "cpu", "cuda"] = "cpu",
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
    int1e_contract_backend : Literal["auto", "cpu", "cuda"], optional
        Backend for AO 1e derivative contractions.
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
    _require_df_mnq_layout(scf_out, where="casci_nuc_grad_df_unrelaxed")
    is_spherical = not bool(getattr(mol, "cart", True))
    if str(int1e_contract_backend).strip().lower() == "cpu" and str(df_backend).strip().lower() == "cuda":
        int1e_contract_backend = "auto" if bool(is_spherical) else "cuda"
    _int1e_mode = _normalize_int1e_backend_mode(str(int1e_contract_backend))
    _sph_int1e_backend: Literal["cpu", "cuda"] = "cpu"
    if bool(is_spherical):
        _sph_int1e_backend = _select_sph_int1e_backend(
            contract_backend=str(int1e_contract_backend),
            df_backend=str(df_backend),
        )
    if profile is not None:
        profile["int1e_contract_backend"] = _int1e_mode
        if bool(is_spherical):
            profile["int1e_sph_backend"] = str(_sph_int1e_backend)

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
    _restore_pool = _apply_df_pool_policy(B_ao, label="casci_nuc_grad_df_unrelaxed")

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
    if is_spherical:
        if str(_sph_int1e_backend) == "cuda":
            from asuka.integrals.int1e_sph_cuda import contract_dhcore_sph_prebuilt_cuda  # noqa: PLC0415

            _dS_pre, _dT_pre, _dV_pre = _build_sph_int1e_prebuilt(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                shell_atom=shell_atom,
                need_overlap=False,
                need_hcore=True,
                to_gpu=True,
            )
            de_h1 = contract_dhcore_sph_prebuilt_cuda(_dT_pre, _dV_pre, D_tot_ao)
        else:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1 = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=D_tot_ao,
                shell_atom=shell_atom,
            )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=D_tot_ao,
            shell_atom=shell_atom,
            contract_backend=str(int1e_contract_backend),
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
        if is_spherical:
            de_df = compute_df_gradient_contributions_analytic_sph(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_sph=B_ao,
                bar_L_sph=bar_L_ao,
                T_c2s=None,
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
    except (NotImplementedError, RuntimeError):
        if is_spherical:
            raise
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

    _restore_pool()
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
    int1e_contract_backend: Literal["auto", "cpu", "cuda"] = "cpu",
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
    int1e_contract_backend : Literal["auto", "cpu", "cuda"], optional
        Backend for AO 1e derivative contractions.
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
    _require_df_mnq_layout(scf_out, where="casci_nuc_grad_df_relaxed")
    is_spherical = not bool(getattr(mol, "cart", True))
    if str(int1e_contract_backend).strip().lower() == "cpu" and str(df_backend).strip().lower() == "cuda":
        int1e_contract_backend = "auto" if bool(is_spherical) else "cuda"
    _int1e_mode = _normalize_int1e_backend_mode(str(int1e_contract_backend))
    _sph_int1e_backend: Literal["cpu", "cuda"] = "cpu"
    if bool(is_spherical):
        _sph_int1e_backend = _select_sph_int1e_backend(
            contract_backend=str(int1e_contract_backend),
            df_backend=str(df_backend),
        )
    if profile is not None:
        profile["int1e_contract_backend"] = _int1e_mode
        if bool(is_spherical):
            profile["int1e_sph_backend"] = str(_sph_int1e_backend)

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
    _restore_pool = _apply_df_pool_policy(B_ao, label="casci_nuc_grad_df_relaxed")

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

    # Energy-weighted density W for the Pulay (overlap-derivative) term.
    #
    # PySCF formula (grad/casci.py): W = im1 + 2*zeta + 2*vhf_s1occ
    # where im1 = C @ Imat_m @ C.T, zeta = C @ (zvec * eps) @ C.T,
    # vhf_s1occ = P1 @ Veff[zvec_ao] @ P1.
    #
    # IMPORTANT: contract_dS_{cart,sph}(M) computes sum_ij dS[A,x,i,j]*M[i,j]
    # where dS includes BOTH bra and ket derivative sides, effectively computing
    # Tr(dS * (M + M.T)) per atom.  Therefore W must NOT be pre-symmetrized.
    W = im1 + 2.0 * zeta + 2.0 * vhf_s1occ

    # ------------------------------------------------------------------
    # Nuclear gradient contractions
    # ------------------------------------------------------------------
    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    _M_h1 = np.asarray(D_tot_ao + zvec_ao, dtype=np.float64)
    if is_spherical:
        if str(_sph_int1e_backend) == "cuda":
            from asuka.integrals.int1e_sph_cuda import contract_dhcore_sph_prebuilt_cuda  # noqa: PLC0415

            _dS_pre, _dT_pre, _dV_pre = _build_sph_int1e_prebuilt(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                shell_atom=shell_atom,
                need_overlap=True,
                need_hcore=True,
                to_gpu=True,
            )
            de_h1 = contract_dhcore_sph_prebuilt_cuda(_dT_pre, _dV_pre, _M_h1)
        else:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1 = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=_M_h1,
                shell_atom=shell_atom,
            )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=_M_h1,
            shell_atom=shell_atom,
            contract_backend=str(int1e_contract_backend),
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
    bar_L_tot = np.asarray(bar_L_ao, dtype=np.float64, order="C")
    bar_L_tot += np.asarray(bar_L_resp, dtype=np.float64)
    del bar_L_ao, bar_L_resp

    df_prof = None if profile is None else profile.setdefault("df_2e", {})
    try:
        if is_spherical:
            de_df = compute_df_gradient_contributions_analytic_sph(
                getattr(scf_out, "ao_basis"),
                getattr(scf_out, "aux_basis"),
                atom_coords_bohr=coords,
                B_sph=B_ao,
                bar_L_sph=bar_L_tot,
                T_c2s=None,
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
        if is_spherical:
            raise
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

    # Pulay (overlap-derivative) term: -Tr(W · dS/dR).
    if is_spherical:
        if str(_sph_int1e_backend) == "cuda":
            from asuka.integrals.int1e_sph_cuda import contract_dS_sph_prebuilt_cuda  # noqa: PLC0415

            de_pulay = -1.0 * contract_dS_sph_prebuilt_cuda(_dS_pre, W)
        else:
            from asuka.integrals.int1e_sph import contract_dS_sph  # noqa: PLC0415

            de_pulay = -1.0 * contract_dS_sph(
                ao_basis,
                atom_coords_bohr=coords,
                M_sph=np.asarray(W, dtype=np.float64),
                shell_atom=shell_atom,
            )
    else:
        de_pulay = -1.0 * contract_dS_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=np.asarray(W, dtype=np.float64),
            shell_atom=shell_atom,
            contract_backend=str(int1e_contract_backend),
        )

    grad = np.asarray(de_h1 + de_df + de_nuc + de_pulay, dtype=np.float64)
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

    _restore_pool()
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
    from .zvector import build_mcscf_hessian_operator, solve_mcscf_zvector, solve_mcscf_zvector_batch  # noqa: PLC0415
    from .nac._df import (  # noqa: PLC0415
        _FixedRDMFcisolver,
        _grad_elec_active_df,
        _Lorb_dot_dgorb_dx_df,
        _build_bar_L_net_active_df,
        _build_bar_L_lorb_df,
    )

    _apply_df_grad_optimal_defaults()

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
    _require_df_mnq_layout(scf_out, where="casscf_nuc_grad_df_per_root")
    is_spherical = not bool(getattr(mol, "cart", True))
    if str(int1e_contract_backend).strip().lower() == "cpu" and str(df_backend).strip().lower() == "cuda":
        int1e_contract_backend = "auto" if bool(is_spherical) else "cuda"

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
    _gpu_runtime_cfg = _resolve_gpu_runtime_config()

    xp, _is_gpu = _resolve_xp(df_backend)
    C = _as_xp_f64(xp, getattr(casscf, "mo_coeff"))
    B_ao = _as_xp_f64(xp, getattr(scf_out, "df_B"))
    h_ao = _as_xp_f64(xp, getattr(getattr(scf_out, "int1e"), "hcore"))
    if bool(_is_gpu):
        _apply_gpu_runtime_config_to_solver(fcisolver_use, _gpu_runtime_cfg)
    _restore_pool = _apply_df_pool_policy(B_ao, label="casscf_nuc_grad_df_per_root")
    _preflush_grad_pool = _normalize_bool_env(_os.environ.get("ASUKA_DF_GRAD_PRE_FLUSH"), default=True)
    if bool(_is_gpu) and bool(_preflush_grad_pool):
        _flush_gpu_pool()
        _log_vram("casscf_nuc_grad_df_per_root.preflush")
    _barl_policy = _resolve_barl_hybrid(
        xp=xp,
        is_gpu=bool(_is_gpu),
        naux_hint=int(getattr(B_ao, "shape", (0, 0, 1))[2]),
    )
    if int(_gpu_runtime_cfg.tile_aux) > 0:
        _barl_policy["qblock"] = int(_gpu_runtime_cfg.tile_aux)
        _barl_policy["qblock_source"] = "runtime"
    else:
        # Refine qblock using nao to bound (qblock, nao, nao) temporaries.
        # Keep explicit env overrides intact.
        if str(_barl_policy.get("qblock_source", "")) != "env":
            _nao_hint = int(getattr(B_ao, "shape", (0, 0, 1))[0])
            _naux_hint = int(getattr(B_ao, "shape", (0, 0, 1))[2])
            try:
                _itemsize = int(xp.dtype(_barl_policy.get("work_dtype", xp.float64)).itemsize)
            except Exception:
                _itemsize = 8
            _q_auto = _auto_qblock_for_aux_tile(nao=_nao_hint, naux=_naux_hint, itemsize=_itemsize)
            try:
                _q_cur = int(_barl_policy.get("qblock", 0) or 0)
            except Exception:
                _q_cur = 0
            _barl_policy["qblock"] = int(_q_auto if _q_cur <= 0 else min(_q_cur, _q_auto))
    _barl_hybrid_enabled = bool(_barl_policy.get("enabled", False))

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

    _fused_contract_requested = _normalize_bool_env(_os.environ.get("ASUKA_DF_FUSED_CONTRACT"), default=False)
    if _gpu_runtime_cfg.fused_df is not None:
        _fused_contract_requested = bool(_gpu_runtime_cfg.fused_df)
    _fused_contract_precision = _normalize_fused_contract_precision(_os.environ.get("ASUKA_DF_FUSED_CONTRACT_PRECISION"))
    try:
        _fused_contract_ab_tol = float(_os.environ.get("ASUKA_DF_FUSED_CONTRACT_AB_TOL", "1e-6"))
    except Exception:
        _fused_contract_ab_tol = 1e-6
    _fused_contract_ab_tol = max(0.0, float(_fused_contract_ab_tol))
    _fused_contract_enabled = (
        bool(_fused_contract_requested)
        and str(df_backend).strip().lower() == "cuda"
        and df_grad_ctx is not None
        and hasattr(df_grad_ctx, "contract_fused_terms")
    )
    _fused_contract_warned_inert = False

    # AO basis objects for 1e derivative contractions
    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    _int1e_backend_mode = _normalize_int1e_backend_mode(str(int1e_contract_backend))
    _sph_int1e_backend = (
        _select_sph_int1e_backend(contract_backend=str(int1e_contract_backend), df_backend=str(df_backend))
        if bool(is_spherical)
        else "cpu"
    )
    _gpu_int1e_path = (not bool(is_spherical)) and bool(_is_gpu) and _int1e_backend_mode in ("cuda", "auto")
    _gpu_sph_int1e_path = bool(is_spherical) and bool(_is_gpu) and str(_sph_int1e_backend) == "cuda"
    _gpu_1e_tensor_path = bool(_is_gpu) and bool(_gpu_int1e_path or _gpu_sph_int1e_path)
    dm1_sa_xp = _as_xp_f64(xp, dm1_sa) if _gpu_1e_tensor_path else np.asarray(dm1_sa, dtype=np.float64)
    dm2_sa_xp = _as_xp_f64(xp, dm2_sa) if _gpu_1e_tensor_path else np.asarray(dm2_sa, dtype=np.float64)

    _int1e_prebuild = _normalize_bool_env(_os.environ.get("ASUKA_INT1E_PREBUILD"), default=(int(nroots) > 1))
    if bool(_gpu_sph_int1e_path):
        _int1e_prebuild = True
    _dS_pre: Any | None = None
    _dT_pre: Any | None = None
    _dV_pre: Any | None = None
    if _int1e_prebuild:
        try:
            if bool(is_spherical):
                from asuka.integrals.int1e_sph import build_dS_sph, build_dT_sph, build_dV_sph  # noqa: PLC0415

                _dS_pre = build_dS_sph(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
                _dT_pre = build_dT_sph(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
                _dV_pre = build_dV_sph(
                    ao_basis,
                    atom_coords_bohr=coords,
                    atom_charges=charges,
                    shell_atom=shell_atom,
                    include_operator_deriv=True,
                )
                if bool(_gpu_sph_int1e_path):
                    _dS_pre = xp.asarray(_dS_pre, dtype=xp.float64)
                    _dT_pre = xp.asarray(_dT_pre, dtype=xp.float64)
                    _dV_pre = xp.asarray(_dV_pre, dtype=xp.float64)
            else:
                from asuka.integrals.int1e_cart import build_dS_cart, build_dT_cart, build_dV_cart  # noqa: PLC0415

                _dS_np = build_dS_cart(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
                _dT_np = build_dT_cart(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
                _dV_np = build_dV_cart(
                    ao_basis,
                    atom_coords_bohr=coords,
                    atom_charges=charges,
                    shell_atom=shell_atom,
                    include_operator_deriv=True,
                )
                if _gpu_int1e_path:
                    _dS_pre = xp.asarray(_dS_np, dtype=xp.float64)
                    _dT_pre = xp.asarray(_dT_np, dtype=xp.float64)
                    _dV_pre = xp.asarray(_dV_np, dtype=xp.float64)
                    del _dS_np, _dT_np, _dV_np
                else:
                    _dS_pre = np.asarray(_dS_np, dtype=np.float64)
                    _dT_pre = np.asarray(_dT_np, dtype=np.float64)
                    _dV_pre = np.asarray(_dV_np, dtype=np.float64)
        except Exception:
            _dS_pre = None
            _dT_pre = None
            _dV_pre = None

    def _contract_hcore_fast(Mmat: Any) -> np.ndarray:
        if bool(is_spherical):
            if _dT_pre is not None and _dV_pre is not None:
                if bool(_gpu_sph_int1e_path):
                    from asuka.integrals.int1e_sph_cuda import contract_dhcore_sph_prebuilt_cuda  # noqa: PLC0415

                    return np.asarray(
                        contract_dhcore_sph_prebuilt_cuda(
                            _dT_pre,
                            _dV_pre,
                            Mmat,
                        ),
                        dtype=np.float64,
                    )
                Mn = np.asarray(_asnumpy_f64(Mmat), dtype=np.float64)
                gn = np.einsum("axij,ij->ax", _dT_pre, Mn, optimize=True) + np.einsum(
                    "axij,ij->ax", _dV_pre, Mn, optimize=True
                )
                return np.asarray(gn, dtype=np.float64)
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            return contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=_asnumpy_f64(Mmat),
                shell_atom=shell_atom,
            )
        if _dT_pre is None or _dV_pre is None:
            return contract_dhcore_cart(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M=Mmat,
                shell_atom=shell_atom,
                contract_backend=str(int1e_contract_backend),
            )
        if _gpu_int1e_path:
            Mx = xp.asarray(Mmat, dtype=xp.float64)
            gx = xp.einsum("axij,ij->ax", _dT_pre, Mx, optimize=True) + xp.einsum(
                "axij,ij->ax", _dV_pre, Mx, optimize=True
            )
            return np.asarray(_asnumpy_f64(gx), dtype=np.float64)
        Mn = np.asarray(Mmat, dtype=np.float64)
        gn = np.einsum("axij,ij->ax", _dT_pre, Mn, optimize=True) + np.einsum(
            "axij,ij->ax", _dV_pre, Mn, optimize=True
        )
        return np.asarray(gn, dtype=np.float64)

    def _contract_pulay_fast(Wmat: Any) -> np.ndarray:
        if bool(is_spherical):
            if _dS_pre is not None:
                if bool(_gpu_sph_int1e_path):
                    from asuka.integrals.int1e_sph_cuda import contract_dS_sph_prebuilt_cuda  # noqa: PLC0415

                    Wv = xp.asarray(Wmat, dtype=xp.float64)
                    Wv = 0.5 * (Wv + Wv.T)
                    return -np.asarray(
                        contract_dS_sph_prebuilt_cuda(
                            _dS_pre,
                            Wv,
                        ),
                        dtype=np.float64,
                    )
                Wn = _asnumpy_f64(Wmat)
                Wn = 0.5 * (Wn + Wn.T)
                gn = np.einsum("axij,ij->ax", _dS_pre, Wn, optimize=True)
                return -np.asarray(gn, dtype=np.float64)
            from asuka.integrals.int1e_sph import contract_dS_sph  # noqa: PLC0415

            Wn = _asnumpy_f64(Wmat)
            Wn = 0.5 * (Wn + Wn.T)
            return -np.asarray(
                contract_dS_sph(
                    ao_basis,
                    atom_coords_bohr=coords,
                    M_sph=Wn,
                    shell_atom=shell_atom,
                ),
                dtype=np.float64,
            )
        if _dS_pre is None:
            return -2.0 * contract_dS_ip_cart(
                ao_basis,
                atom_coords_bohr=coords,
                M=Wmat,
                shell_atom=shell_atom,
                contract_backend=str(int1e_contract_backend),
            )
        if _gpu_int1e_path:
            Wx = xp.asarray(Wmat, dtype=xp.float64)
            Wx = 0.5 * (Wx + Wx.T)
            gx = xp.einsum("axij,ij->ax", _dS_pre, Wx, optimize=True)
            return -np.asarray(_asnumpy_f64(gx), dtype=np.float64)
        Wn = np.asarray(Wmat, dtype=np.float64)
        Wn = 0.5 * (Wn + Wn.T)
        gn = np.einsum("axij,ij->ax", _dS_pre, Wn, optimize=True)
        return -np.asarray(gn, dtype=np.float64)

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
    _t_gen_g_hop = 0.0
    _t_gfock = 0.0
    _t_1e_pulay = 0.0
    _t_response_pulay = 0.0
    _z_solver: str | None = None
    _z_solver_detail: str | None = None
    _z_backend: str | None = None
    _z_matvec_calls = 0
    _z_niter = 0
    _barl_stage_effective = str(_barl_policy.get("stage", "fp64"))
    _barl_ab_diff_max = 0.0
    _barl_fallback = False
    _barl_ab_checked = False

    # Multi-stream DF contraction (CUDA only). Escape hatch: ASUKA_DF_CONTRACT_STREAMS=1.
    _use_multistream_contract = False
    _n_streams = 1
    _cp = None  # type: ignore[assignment]
    _main_stream = None  # type: ignore[assignment]
    _main_stream_cm = nullcontext()
    _contract_streams: list[Any] = []

    if (
        not bool(_barl_hybrid_enabled)
        and not bool(_fused_contract_enabled)
        and
        str(df_backend).strip().lower() == "cuda"
        and df_grad_ctx is not None
        and str(getattr(df_grad_ctx, "backend", "")).strip().lower() == "cuda"
        and int(nroots) > 1
    ):
        _n_streams_env_raw = os.environ.get("ASUKA_DF_CONTRACT_STREAMS")
        _n_streams_default = 1 if int(nroots) <= 2 else 4
        if _n_streams_env_raw is None:
            _n_streams_env = int(_n_streams_default)
        else:
            try:
                _n_streams_env = int(_n_streams_env_raw)
            except Exception:
                _n_streams_env = int(_n_streams_default)
        _n_streams = max(1, min(int(nroots), int(_n_streams_env)))
        # Auto-scale: if sizeof(B) * n_streams > 50% free VRAM, reduce streams.
        try:
            import cupy as _cp_probe  # noqa: PLC0415

            _sizeof_B = int(B_ao.size) * 8  # float64
            _free_vram = int(_cp_probe.cuda.runtime.memGetInfo()[0])
            try:
                _foot_mult = float(_os.environ.get("ASUKA_DF_STREAM_FOOTPRINT_MULT", "2.5"))
            except Exception:
                _foot_mult = 2.5
            _foot_mult = max(1.0, float(_foot_mult))
            _vram_cap = max(1, int(0.5 * _free_vram // max(1, int(_foot_mult * _sizeof_B))))
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
    _z_recycle_max = 0
    if _gpu_runtime_cfg.krylov_recycle_max_vectors is not None:
        _z_recycle_max = max(0, int(_gpu_runtime_cfg.krylov_recycle_max_vectors))
    _z_prev_x0: np.ndarray | None = None
    _mixed_policy_active = bool(_gpu_runtime_cfg.precision_policy in ("mixed_conservative", "mixed_aggressive"))
    _mixed_fallback_rtol = 5e-7 if _gpu_runtime_cfg.precision_policy == "mixed_conservative" else 5e-6
    _fallback_rtol_env = _parse_env_float("ASUKA_GPU_PRECISION_FALLBACK_RREL")
    if _fallback_rtol_env is not None and _fallback_rtol_env > 0.0:
        _mixed_fallback_rtol = float(_fallback_rtol_env)

    _MISSING = object()

    @contextmanager
    def _temporary_solver_attrs(obj: Any, updates: dict[str, Any]):
        prev: dict[str, Any] = {}
        for key, val in updates.items():
            prev[key] = getattr(obj, key, _MISSING)
            setattr(obj, key, val)
        try:
            yield
        finally:
            for key, old in prev.items():
                if old is _MISSING:
                    try:
                        delattr(obj, key)
                    except Exception:
                        pass
                else:
                    setattr(obj, key, old)

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
    ppaa_sa_np = _asnumpy_f64(getattr(eris_sa, "ppaa"))  # (nmo,nmo,ncas,ncas)
    papa_sa_np = _asnumpy_f64(getattr(eris_sa, "papa"))  # (nmo,ncas,nmo,ncas)
    ppaa_sa = _as_xp_f64(xp, ppaa_sa_np) if _gpu_1e_tensor_path else np.asarray(ppaa_sa_np, dtype=np.float64)
    papa_sa = _as_xp_f64(xp, papa_sa_np) if _gpu_1e_tensor_path else np.asarray(papa_sa_np, dtype=np.float64)
    _lorb_resp_cache: dict[str, Any] = {
        "ppaa_np": np.asarray(ppaa_sa_np, dtype=np.float64),
        "papa_np": np.asarray(papa_sa_np, dtype=np.float64),
    }
    _lorb_resp_cache["aapa"] = np.asarray(
        _lorb_resp_cache["ppaa_np"][:, ncore:nocc, :, :].transpose(2, 3, 0, 1),
        dtype=np.float64,
    )
    if _gpu_1e_tensor_path:
        _lorb_resp_cache["ppaa_xp"] = _as_xp_f64(xp, ppaa_sa)
        _lorb_resp_cache["papa_xp"] = _as_xp_f64(xp, papa_sa)
        _lorb_resp_cache["aapa_xp"] = _as_xp_f64(
            xp,
            _lorb_resp_cache["ppaa_xp"][:, ncore:nocc, :, :].transpose(2, 3, 0, 1),
        )
    vhf_c_sa = _asnumpy_f64(getattr(eris_sa, "vhf_c"))  # (nmo,nmo)
    h1e_mo_sa = _asnumpy_f64(C).T @ _asnumpy_f64(h_ao) @ _asnumpy_f64(C)
    h1cas_0 = h1e_mo_sa[ncore:nocc, ncore:nocc] + vhf_c_sa[ncore:nocc, ncore:nocc]
    eri_cas_sa = ppaa_sa_np[ncore:nocc, ncore:nocc]

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
    _ppaa_2d = ppaa_sa_np.reshape(nmo * nmo, ncas * ncas)

    # ------------------------------------------------------------------
    # SA base gradient (shared across all roots, includes Pulay term)
    # This is the correct SA-CASSCF gradient (variational w.r.t. SA energy).
    # Per-root gradients are expressed as: grad_K = grad_sa_base + delta_K
    # ------------------------------------------------------------------
    # Drain CASSCF pool residue — the optimizer leaves ~10 GB in the CuPy
    # pool that is no longer referenced.  Flushing here reduces the peak
    # driver-visible VRAM for the entire gradient phase.
    _flush_gpu_pool()
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
            if bool(is_spherical):
                if df_grad_ctx is not None:
                    de_df_sa = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_sa, T_c2s=None)
                else:
                    de_df_sa = compute_df_gradient_contributions_analytic_sph(
                        ao_basis,
                        aux_basis,
                        atom_coords_bohr=coords,
                        B_sph=B_ao,
                        bar_L_sph=bar_L_sa,
                        T_c2s=None,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend),
                        df_threads=int(df_threads),
                        profile=None,
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
            if bool(is_spherical):
                raise
            warnings.warn(
                f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                "expect noisy/non-conservative forces in MD. "
                f"Reason: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            warned_df_fd = True
            de_df_sa = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=bar_L_sa,
                backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                delta_bohr=float(delta_bohr), profile=None,
            )

    # ── 1e and Pulay contractions ──
    W_sa_xp: Any | None = None
    C_occ_xp: Any | None = None
    if _gpu_1e_tensor_path:
        with _main_stream_cm:
            _D_tot_sa_x = xp.asarray(D_tot_sa, dtype=xp.float64)
        de_h1_sa = _contract_hcore_fast(_D_tot_sa_x)

        # SA Pulay term: -Tr(W_sa · dS/dR).
        with _main_stream_cm:
            C_occ_xp = C[:, :nocc]
            gfock_sa_xp = xp.asarray(gfock_sa, dtype=xp.float64)
            _tmp_sa = C @ gfock_sa_xp[:, :nocc]  # (nao, nocc)
            W_sa_xp = 0.5 * (_tmp_sa @ C_occ_xp.T + C_occ_xp @ _tmp_sa.T)
        de_pulay_sa = _contract_pulay_fast(W_sa_xp)
        W_sa = _asnumpy_f64(W_sa_xp)
        C_occ_np = _asnumpy_f64(C_occ_xp)
    else:
        _D_tot_sa_1e = _asnumpy_f64(D_tot_sa)
        de_h1_sa = _contract_hcore_fast(_D_tot_sa_1e)
        gfock_sa_np = _asnumpy_f64(gfock_sa)
        C_occ_np = C_np[:, :nocc]
        _tmp_sa = C_np @ gfock_sa_np[:, :nocc]  # (nao, nocc)
        W_sa = 0.5 * (_tmp_sa @ C_occ_np.T + C_occ_np @ _tmp_sa.T)
        de_pulay_sa = _contract_pulay_fast(W_sa)

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

    # Precompute root-invariant vhf_c/vhf_a from SA-averaged densities.
    # These are reused by _build_dme0_lorb_response every root, avoiding 2
    # redundant _df_JK calls per root (~3s each on cc-pVDZ).
    with _main_stream_cm:
        _D_core_cached = 2.0 * (C[:, :ncore] @ C[:, :ncore].T) if ncore else xp.zeros((int(B_ao.shape[0]), int(B_ao.shape[0])), dtype=xp.float64)
        _D_act_sa_cached = C_act_sa @ xp.asarray(dm1_sa, dtype=xp.float64) @ C_act_sa.T
        _Jc_cached, _Kc_cached = _df_scf._df_JK(B_ao, _D_core_cached, want_J=True, want_K=True)  # noqa: SLF001
        _Ja_cached, _Ka_cached = _df_scf._df_JK(B_ao, _D_act_sa_cached, want_J=True, want_K=True)  # noqa: SLF001
    _vhf_cache = {
        "vhf_c": _Jc_cached - 0.5 * _Kc_cached,
        "vhf_a": _Ja_cached - 0.5 * _Ka_cached,
    }
    del _D_core_cached, _D_act_sa_cached, _Jc_cached, _Kc_cached, _Ja_cached, _Ka_cached

    # Precompute gfock with zero active density (for CI response Pulay subtraction).
    _dm1_zero = xp.zeros((int(ncas), int(ncas)), dtype=xp.float64)
    _dm2_zero = xp.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=xp.float64)
    _gfock_zero, _, _, _, _ = _build_gfock_casscf_df(
        B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
        dm1_act=_dm1_zero, dm2_act=_dm2_zero,
    )
    _gfock_zero_xp = xp.asarray(_gfock_zero, dtype=xp.float64) if _gpu_1e_tensor_path else None
    _gfock_zero_np = _asnumpy_f64(_gfock_zero)

    def _contract_df_delta_sync(_bar_L: Any) -> np.ndarray:
        nonlocal warned_df_fd
        try:
            if bool(is_spherical):
                if df_grad_ctx is not None:
                    _de = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=_bar_L, T_c2s=None)
                else:
                    _de = compute_df_gradient_contributions_analytic_sph(
                        ao_basis,
                        aux_basis,
                        atom_coords_bohr=coords,
                        B_sph=B_ao,
                        bar_L_sph=_bar_L,
                        T_c2s=None,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend),
                        df_threads=int(df_threads),
                        profile=None,
                    )
            else:
                if df_grad_ctx is not None:
                    _de = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=_bar_L)
                else:
                    _de = compute_df_gradient_contributions_analytic_packed_bases(
                        ao_basis,
                        aux_basis,
                        atom_coords_bohr=coords,
                        B_ao=B_ao,
                        bar_L_ao=_bar_L,
                        L_chol=getattr(scf_out, "df_L", None),
                        backend=str(df_backend),
                        df_threads=int(df_threads),
                        profile=None,
                    )
            return np.asarray(_asnumpy_f64(_de), dtype=np.float64)
        except (NotImplementedError, RuntimeError) as e:
            if bool(is_spherical):
                raise
            if not warned_df_fd:
                warnings.warn(
                    f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                    "expect noisy/non-conservative forces in MD. "
                    f"Reason: {type(e).__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_df_fd = True
            _de = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                bar_L_ao=_bar_L,
                backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                delta_bohr=float(delta_bohr),
                profile=None,
            )
            return np.asarray(_asnumpy_f64(_de), dtype=np.float64)

    _z_results_by_root: list[Any] | None = None
    _z_batch_enabled = bool(int(nroots) > 1) and _normalize_bool_env(_os.environ.get("ASUKA_ZVECTOR_BATCH_SOLVE"), default=True)
    _fast_rhs_enabled = _normalize_bool_env(_os.environ.get("ASUKA_ZVECTOR_FAST_RHS"), default=True)

    if _z_batch_enabled:
        _rhs_orb_all: list[np.ndarray] = []
        _rhs_ci_all: list[list[np.ndarray]] = []
        _rhs_ci_zeros = [np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()) for r in range(int(nroots))]
        for K in range(nroots):
            _flush_gpu_pool()
            _log_vram(f"root {K} rhs start")
            t0 = time.perf_counter()
            fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=per_root_rdms[K][0], dm2=per_root_rdms[K][1])
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
            if _fast_rhs_enabled:
                try:
                    with _main_stream_cm:
                        g_K = _newton_casscf.compute_mcscf_gradient_vector(
                            mc_K,
                            np.asarray(C_np, dtype=np.float64),
                            ci_list[K],
                            eris_sa,
                            gauge="none",
                            strict_weights=False,
                            enforce_absorb_h1e_direct=True,
                        )
                    g_K = np.asarray(g_K, dtype=np.float64).ravel()
                except Exception:
                    with _main_stream_cm:
                        with _force_internal_newton():
                            g_K, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                                mc_K, C, ci_list[K], eris_sa, verbose=0, implementation="internal",
                            )
                        g_K = _asnumpy_f64(g_K).ravel()
                    del _gupd, _hop, _hdiag
            else:
                with _main_stream_cm:
                    with _force_internal_newton():
                        g_K, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                            mc_K, C, ci_list[K], eris_sa, verbose=0, implementation="internal",
                        )
                    g_K = _asnumpy_f64(g_K).ravel()
                del _gupd, _hop, _hdiag
            del mc_K, fcisolver_fixed
            if _profile_df_per_root:
                _t_gen_g_hop += time.perf_counter() - t0
            _flush_gpu_pool()
            _log_vram(f"root {K} after rhs build")

            rhs_orb = np.asarray(g_K[:n_orb], dtype=np.float64)
            rhs_ci_K = np.asarray(g_K[n_orb:], dtype=np.float64)
            rhs_ci = [arr.copy() for arr in _rhs_ci_zeros]
            ndet_K = int(np.asarray(ci_list[K]).size)
            rhs_ci[K] = rhs_ci_K[:ndet_K]
            _rhs_orb_all.append(rhs_orb)
            _rhs_ci_all.append(rhs_ci)

        t0 = time.perf_counter()
        _gcrotmk_k = int(_z_recycle_max) if (_z_method == "gcrotmk" and _z_recycle_max > 0) else None
        _x0_list = [None for _ in range(int(nroots))]
        if _z_use_x0 and _z_prev_x0 is not None:
            _x0_list[0] = np.asarray(_z_prev_x0, dtype=np.float64).ravel()
        z_batch = solve_mcscf_zvector_batch(
            mc_sa,
            rhs_orb_list=_rhs_orb_all,
            rhs_ci_list=_rhs_ci_all,
            hessian_op=hess_op,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            method=str(_z_method),
            restart=None,
            x0_list=_x0_list,
            recycle_space=_z_recycle_space,
            gcrotmk_k=_gcrotmk_k,
            reorder="norm_desc",
            shared_recycle=True,
            chain_x0=bool(_z_use_x0),
        )
        _z_results_by_root = list(z_batch.results)
        if _z_recycle_space is not None and _z_recycle_max > 0 and len(_z_recycle_space) > _z_recycle_max:
            del _z_recycle_space[:-int(_z_recycle_max)]
        if _profile_df_per_root:
            _t_z_solve += time.perf_counter() - t0
            _z_solver = str(z_batch.info.get("solver", _z_solver or "")).strip() or _z_solver
            _z_backend = str(z_batch.info.get("backend", _z_backend or "")).strip() or _z_backend
            _solver_detail = z_batch.info.get("solver_detail", None)
            if isinstance(_solver_detail, (list, tuple)):
                _z_solver_detail = "|".join(str(s).strip() for s in _solver_detail if str(s).strip()) or _z_solver_detail
            elif _solver_detail is not None:
                _z_solver_detail = str(_solver_detail).strip() or _z_solver_detail
            _z_matvec_calls += int(z_batch.info.get("total_matvec_calls", 0) or 0)
            _z_niter += int(z_batch.info.get("total_niter", 0) or 0)

        if _mixed_policy_active:
            _gcrotmk_k = int(_z_recycle_max) if (_z_method == "gcrotmk" and _z_recycle_max > 0) else None
            for K in range(int(nroots)):
                _z_cur = _z_results_by_root[K]
                _res_rel = float(_z_cur.info.get("residual_rel", np.inf))
                _needs_fp64_fallback = bool(
                    (not bool(_z_cur.converged))
                    or (not np.isfinite(_res_rel))
                    or (float(_res_rel) > float(_mixed_fallback_rtol))
                )
                if not _needs_fp64_fallback:
                    continue
                _fallback_start = time.perf_counter()
                _fallback_updates = {
                    "matvec_cuda_dtype": "float64",
                    "approx_cuda_dtype": "float64",
                    "matvec_cuda_mixed_low_precision_max_iter": 0,
                }
                with _temporary_solver_attrs(fcisolver_use, _fallback_updates):
                    z_fp64 = solve_mcscf_zvector(
                        mc_sa,
                        rhs_orb=_rhs_orb_all[K],
                        rhs_ci=_rhs_ci_all[K],
                        hessian_op=hess_op,
                        tol=float(z_tol),
                        maxiter=int(z_maxiter),
                        method=str(_z_method),
                        x0=np.asarray(_z_cur.z_packed, dtype=np.float64).ravel(),
                        recycle_space=_z_recycle_space,
                        gcrotmk_k=_gcrotmk_k,
                    )
                if _z_recycle_space is not None and _z_recycle_max > 0 and len(_z_recycle_space) > _z_recycle_max:
                    del _z_recycle_space[:-int(_z_recycle_max)]
                try:
                    z_fp64.info["precision_fallback"] = "mixed_to_fp64"
                    z_fp64.info["precision_fallback_trigger_residual_rel"] = float(_res_rel)
                    z_fp64.info["precision_fallback_threshold_residual_rel"] = float(_mixed_fallback_rtol)
                    z_fp64.info["precision_fallback_time_s"] = float(time.perf_counter() - _fallback_start)
                except Exception:
                    pass
                _z_results_by_root[K] = z_fp64
                if _profile_df_per_root:
                    _t_z_solve += float(time.perf_counter() - _fallback_start)

        if _z_use_x0 and _z_results_by_root:
            _z_prev_x0 = np.asarray(_z_results_by_root[-1].z_packed, dtype=np.float64).ravel().copy()

    for K in range(nroots):
        # Drain free CuPy blocks between roots so pool caching does not inflate
        # driver-visible VRAM and trigger avoidable OOM on small GPUs.
        _flush_gpu_pool()
        _log_vram(f"root {K} start")
        dm1_K, dm2_K = per_root_rdms[K]
        _barl_stage = str(_barl_policy.get("stage", "fp64"))
        _barl_work_dtype = _barl_policy.get("work_dtype", xp.float64)
        _barl_out_dtype = _barl_policy.get("out_dtype", xp.float64)
        _barl_stage_effective = _barl_stage

        # For single-state CASSCF: no response — SA base gradient IS the exact per-root gradient.
        if nroots == 1:
            grads_out[int(K)] = np.asarray(grad_sa_base, dtype=np.float64)
            continue

        # Step A: Build per-root densities (GPU) and cache RDM deltas (CPU).
        t0 = time.perf_counter()
        with _main_stream_cm:
            gfock_K, _D_core_K, _D_act_K, D_tot_K, _C_act_K = _build_gfock_casscf_df(
                B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas), dm1_act=dm1_K, dm2_act=dm2_K,
            )
            dm1_delta = dm1_K - dm1_sa
            dm2_delta = dm2_K - dm2_sa
            _flush_gpu_pool()
        if _profile_df_per_root:
            _t_gfock += time.perf_counter() - t0

        # Step B/C: Z-vector RHS + solve (legacy) or batched pre-solved path.
        rhs_orb = None
        rhs_ci_K = None
        rhs_ci: list[np.ndarray] | None = None
        g_K = None
        if _z_results_by_root is not None:
            z_K = _z_results_by_root[int(K)]
        else:
            t0 = time.perf_counter()
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

            if _fast_rhs_enabled:
                try:
                    with _main_stream_cm:
                        g_K = _newton_casscf.compute_mcscf_gradient_vector(
                            mc_K,
                            np.asarray(C_np, dtype=np.float64),
                            ci_list[K],
                            eris_sa,
                            gauge="none",
                            strict_weights=False,
                            enforce_absorb_h1e_direct=True,
                        )
                    g_K = np.asarray(g_K, dtype=np.float64).ravel()
                except Exception:
                    with _main_stream_cm:
                        with _force_internal_newton():
                            g_K, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                                mc_K, C, ci_list[K], eris_sa, verbose=0, implementation="internal",
                            )
                        g_K = _asnumpy_f64(g_K).ravel()
                    del _gupd, _hop, _hdiag
            else:
                with _main_stream_cm:
                    with _force_internal_newton():
                        g_K, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                            mc_K, C, ci_list[K], eris_sa, verbose=0, implementation="internal",
                        )
                    g_K = _asnumpy_f64(g_K).ravel()
                del _gupd, _hop, _hdiag
            del mc_K, fcisolver_fixed
            if _profile_df_per_root:
                _t_gen_g_hop += time.perf_counter() - t0
            _flush_gpu_pool()
            _log_vram(f"root {K} after gen_g_hop cleanup")

            rhs_orb = g_K[:n_orb]
            rhs_ci_K = g_K[n_orb:]
            rhs_ci = []
            for r in range(nroots):
                rhs_ci.append(np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()))
            ndet_K = int(np.asarray(ci_list[K]).size)
            rhs_ci[K] = rhs_ci_K[:ndet_K]

            t0 = time.perf_counter()
            _x0_use = _z_prev_x0 if (_z_use_x0 and _z_prev_x0 is not None) else None
            _gcrotmk_k = int(_z_recycle_max) if (_z_method == "gcrotmk" and _z_recycle_max > 0) else None
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
                gcrotmk_k=_gcrotmk_k,
            )
            if _z_recycle_space is not None and _z_recycle_max > 0 and len(_z_recycle_space) > _z_recycle_max:
                del _z_recycle_space[:-int(_z_recycle_max)]

            if _mixed_policy_active:
                _res_rel = float(z_K.info.get("residual_rel", np.inf))
                _needs_fp64_fallback = bool(
                    (not bool(z_K.converged))
                    or (not np.isfinite(_res_rel))
                    or (float(_res_rel) > float(_mixed_fallback_rtol))
                )
                if _needs_fp64_fallback:
                    _fallback_start = time.perf_counter()
                    _fallback_updates = {
                        "matvec_cuda_dtype": "float64",
                        "approx_cuda_dtype": "float64",
                        "matvec_cuda_mixed_low_precision_max_iter": 0,
                    }
                    with _temporary_solver_attrs(fcisolver_use, _fallback_updates):
                        z_fp64 = solve_mcscf_zvector(
                            mc_sa,
                            rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
                            rhs_ci=rhs_ci,
                            hessian_op=hess_op,
                            tol=float(z_tol),
                            maxiter=int(z_maxiter),
                            method=str(_z_method),
                            x0=np.asarray(z_K.z_packed, dtype=np.float64).ravel(),
                            recycle_space=_z_recycle_space,
                            gcrotmk_k=_gcrotmk_k,
                        )
                    if _z_recycle_space is not None and _z_recycle_max > 0 and len(_z_recycle_space) > _z_recycle_max:
                        del _z_recycle_space[:-int(_z_recycle_max)]
                    try:
                        z_fp64.info["precision_fallback"] = "mixed_to_fp64"
                        z_fp64.info["precision_fallback_trigger_residual_rel"] = float(_res_rel)
                        z_fp64.info["precision_fallback_threshold_residual_rel"] = float(_mixed_fallback_rtol)
                        z_fp64.info["precision_fallback_time_s"] = float(time.perf_counter() - _fallback_start)
                    except Exception:
                        pass
                    z_K = z_fp64

            if _profile_df_per_root:
                _t_z_solve += time.perf_counter() - t0
                try:
                    _z_solver = str(z_K.info.get("solver", _z_solver or "")).strip() or _z_solver
                    _z_solver_detail = str(z_K.info.get("solver", _z_solver_detail or "")).strip() or _z_solver_detail
                    _z_backend = str(z_K.info.get("backend", _z_backend or "")).strip() or _z_backend
                    _z_matvec_calls += int(z_K.info.get("matvec_calls", 0) or 0)
                    _z_niter += int(z_K.info.get("niter", 0) or 0)
                except Exception:
                    pass
            if _z_use_x0:
                _z_prev_x0 = np.asarray(z_K.z_packed, dtype=np.float64).ravel().copy()

        Lvec = np.asarray(z_K.z_packed, dtype=np.float64).ravel()
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
        bar_L_delta_total: Any | None = None
        bar_L_contract: Any | None = None
        bar_L_terms_contract: list[Any] | None = None
        de_df_delta_precomputed: np.ndarray | None = None
        with _main_stream_cm:
            qblock_eff = int(_barl_policy.get("qblock", 0))
            t0 = time.perf_counter()
            nao = int(getattr(B_ao, "shape", (0, 0, 1))[0])
            naux = int(getattr(B_ao, "shape", (0, 0, 1))[2])
            # Always use a single accumulation buffer for bar_L construction.
            # The old per-term path (3 separate buffers) wastes 2×sizeof(B) VRAM
            # with no benefit: contraction now sums terms before the kernel anyway.
            if False:  # kept for reference; the 3-buffer path is strictly dominated
                pass
            else:
                # Default path: allocate one accumulation buffer and build all terms into it.
                bar_L_delta_total = xp.zeros((naux, nao, nao), dtype=_barl_out_dtype)

                _build_bar_L_delta_casscf_df(
                    B_ao,
                    D_core_ao=D_core_sa,
                    C_act=C_act_sa,
                    dm1_delta=dm1_delta,
                    dm2_delta=dm2_delta,
                    rho_core=rho_core,
                    L2=L2,
                    out=bar_L_delta_total,
                    symmetrize=False,
                    work_dtype=_barl_work_dtype,
                    out_dtype=_barl_out_dtype,
                    qblock=qblock_eff,
                )
                _log_vram(f"root {K} after delta_hf")
                _flush_gpu_pool()
                _log_vram(f"root {K} after delta_hf flush")

                # CI response term: accumulate directly into bar_L_delta_total.
                _, D_act_lci = _build_bar_L_net_active_df(
                    B_ao,
                    C,
                    dm1_lci,
                    dm2_lci,
                    ncore=int(ncore),
                    ncas=int(ncas),
                    xp=xp,
                    L_act=L_act,
                    rho_core=rho_core,
                    out=bar_L_delta_total,
                    symmetrize=False,
                    work_dtype=_barl_work_dtype,
                    out_dtype=_barl_out_dtype,
                    qblock=qblock_eff,
                )
                _log_vram(f"root {K} after lci_net")
                _flush_gpu_pool()
                _log_vram(f"root {K} after lci_net flush")

                # Orbital response term: accumulate directly into bar_L_delta_total.
                _, D_L_lorb = _build_bar_L_lorb_df(
                    B_ao,
                    C,
                    np.asarray(Lorb_mat, dtype=np.float64),
                    dm1_sa,
                    dm2_sa,
                    ncore=int(ncore),
                    ncas=int(ncas),
                    xp=xp,
                    out=bar_L_delta_total,
                    symmetrize=False,
                    work_dtype=_barl_work_dtype,
                    out_dtype=_barl_out_dtype,
                    qblock=qblock_eff,
                )
                _log_vram(f"root {K} after lorb")

                _symmetrize_bar_L_inplace(bar_L_delta_total, xp)
                _log_vram(f"root {K} bar_L fused")
                if _profile_df_per_root:
                    _t_bar_L_delta += time.perf_counter() - t0

        if bool(_fused_contract_enabled):
            if bar_L_terms_contract is None:
                # Single-buffer accumulation was used; fused term-list not needed.
                _fused_contract_enabled = False
            else:
                _terms_fp64: list[Any] = []
                for _term in bar_L_terms_contract:
                    if _term is None:
                        continue
                    _term64 = _term if str(getattr(_term, "dtype", "")) == str(np.float64) else xp.asarray(_term, dtype=xp.float64)
                    _terms_fp64.append(_term64)
                if len(_terms_fp64) == 0:
                    if not bool(_fused_contract_warned_inert):
                        warnings.warn(
                            "ASUKA_DF_FUSED_CONTRACT requested but produced an empty term list; "
                            "falling back to non-fused DF contraction.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _fused_contract_warned_inert = True
                    _fused_contract_enabled = False
                    bar_L_terms_contract = None
                else:
                    bar_L_terms_contract = _terms_fp64

        if (
            not bool(_fused_contract_enabled)
            and bar_L_delta_total is None
            and bar_L_terms_contract is not None
            and len(bar_L_terms_contract) > 0
        ):
            bar_L_delta_total = xp.zeros((naux, nao, nao), dtype=xp.float64)
            for _term in bar_L_terms_contract:
                bar_L_delta_total += xp.asarray(_term, dtype=xp.float64)
            _symmetrize_bar_L_inplace(bar_L_delta_total, xp)
            bar_L_terms_contract = None

        bar_L_contract = bar_L_delta_total
        if bar_L_contract is None and not bool(_fused_contract_enabled):  # pragma: no cover
            raise RuntimeError("internal error: bar_L buffer was not constructed")
        if bar_L_contract is not None and str(getattr(bar_L_contract, "dtype", "")) != str(np.float64):
            bar_L_contract = xp.asarray(bar_L_contract, dtype=xp.float64)

        if (
            bool(_barl_hybrid_enabled)
            and int(K) == 0
            and bool(_barl_policy.get("ab_check", True))
            and not bool(_barl_policy.get("calibrated", False))
            and str(_barl_policy.get("stage", "fp64")) != "fp64"
        ):
            # Reference fp64 bar_L built in a single buffer to avoid holding multiple
            # sizeof(B) arrays simultaneously.
            _nao = int(getattr(B_ao, "shape", (0, 0, 1))[0])
            _naux = int(getattr(B_ao, "shape", (0, 0, 1))[2])
            bar_L_ref = xp.zeros((_naux, _nao, _nao), dtype=xp.float64)
            _build_bar_L_delta_casscf_df(
                B_ao,
                D_core_ao=D_core_sa,
                C_act=C_act_sa,
                dm1_delta=dm1_delta,
                dm2_delta=dm2_delta,
                rho_core=rho_core,
                L2=L2,
                out=bar_L_ref,
                symmetrize=False,
                work_dtype=xp.float64,
                out_dtype=xp.float64,
                qblock=int(_barl_policy.get("qblock", 0)),
            )
            _, D_act_lci_ref = _build_bar_L_net_active_df(
                B_ao,
                C,
                dm1_lci,
                dm2_lci,
                ncore=int(ncore),
                ncas=int(ncas),
                xp=xp,
                L_act=L_act,
                rho_core=rho_core,
                out=bar_L_ref,
                symmetrize=False,
                work_dtype=xp.float64,
                out_dtype=xp.float64,
                qblock=int(_barl_policy.get("qblock", 0)),
            )
            _, D_L_lorb_ref = _build_bar_L_lorb_df(
                B_ao,
                C,
                np.asarray(Lorb_mat, dtype=np.float64),
                dm1_sa,
                dm2_sa,
                ncore=int(ncore),
                ncas=int(ncas),
                xp=xp,
                out=bar_L_ref,
                symmetrize=False,
                work_dtype=xp.float64,
                out_dtype=xp.float64,
                qblock=int(_barl_policy.get("qblock", 0)),
            )
            _symmetrize_bar_L_inplace(bar_L_ref, xp)

            de_df_ref = _contract_df_delta_sync(bar_L_ref)
            if de_df_delta_precomputed is not None:
                de_df_stage = np.asarray(de_df_delta_precomputed, dtype=np.float64)
            elif bool(_fused_contract_enabled) and bar_L_terms_contract is not None and df_grad_ctx is not None:
                if bool(is_spherical):
                    de_df_stage = np.asarray(
                        df_grad_ctx.contract_fused_terms_sph(
                            B_sph=B_ao,
                            bar_L_terms_sph=bar_L_terms_contract,
                            T_c2s=None,
                            precision=str(_fused_contract_precision),
                        ),
                        dtype=np.float64,
                    )
                else:
                    de_df_stage = np.asarray(
                        df_grad_ctx.contract_fused_terms(
                            B_ao=B_ao,
                            bar_L_terms=bar_L_terms_contract,
                            precision=str(_fused_contract_precision),
                        ),
                        dtype=np.float64,
                    )
            else:
                de_df_stage = _contract_df_delta_sync(bar_L_contract)
            _barl_diff = float(np.max(np.abs(de_df_stage - de_df_ref)))
            _barl_ab_checked = True
            _barl_ab_diff_max = max(float(_barl_ab_diff_max), float(_barl_diff))
            _ab_tol_eff = float(_fused_contract_ab_tol) if bool(_fused_contract_enabled) else float(_barl_policy.get("ab_tol", 1e-6))
            if _barl_diff > _ab_tol_eff:
                if bool(_barl_policy.get("warn", True)):
                    warnings.warn(
                        "[DF_HYBRID] root-0 A/B check exceeded tolerance; falling back to fp64 "
                        f"(stage={_barl_policy.get('stage')} max_abs={_barl_diff:.3e} tol={_ab_tol_eff:.3e})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                _barl_policy["stage_index"] = max(0, len(tuple(_barl_policy.get("ladder", ("fp64",)))) - 1)
                _advance_barl_stage(_barl_policy, xp)
                bar_L_contract = bar_L_ref
                bar_L_delta_total = bar_L_ref
                bar_L_terms_contract = None
                D_act_lci = D_act_lci_ref
                D_L_lorb = D_L_lorb_ref
                de_df_delta_precomputed = de_df_ref
                _barl_stage_effective = str(_barl_policy.get("stage", "fp64"))
                _barl_fallback = True
                _fused_contract_enabled = False
            else:
                de_df_delta_precomputed = de_df_stage
                del bar_L_ref
            _barl_policy["calibrated"] = True

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
                    if bool(is_spherical):
                        grad_dev = df_grad_ctx.contract_device_sph(B_sph=B_ao, bar_L_sph=bar_L_contract, T_c2s=None)
                    else:
                        grad_dev = df_grad_ctx.contract_device(B_ao=B_ao, bar_L_ao=bar_L_contract)
            except (NotImplementedError, RuntimeError, ValueError):
                grad_dev = None

            if grad_dev is not None:
                # ── Single fused 1e call: delta from SA + lci + lorb ──
                if _gpu_1e_tensor_path:
                    with _main_stream_cm:
                        D_1e_delta = xp.asarray(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb, dtype=xp.float64)
                else:
                    with _main_stream_cm:
                        D_1e_delta = _asnumpy_f64(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb)
                de_h1_delta = _contract_hcore_fast(D_1e_delta)

                # ── Per-root Pulay delta: -Tr((W_K - W_sa) · dS/dR) ──
                # ── Response Pulay (multi-stream path) ──
                _gfock_lci_raw, _, _, _, _ = _build_gfock_casscf_df(
                    B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
                    dm1_act=xp.asarray(dm1_lci, dtype=xp.float64),
                    dm2_act=xp.asarray(dm2_lci, dtype=xp.float64),
                )
                if _gpu_1e_tensor_path and W_sa_xp is not None and C_occ_xp is not None:
                    with _main_stream_cm:
                        gfock_K_xp = xp.asarray(gfock_K, dtype=xp.float64)
                        _tmp_K = C @ gfock_K_xp[:, :nocc]  # (nao, nocc)
                        W_K_xp = 0.5 * (_tmp_K @ C_occ_xp.T + C_occ_xp @ _tmp_K.T)
                        _dgfock_lci = xp.asarray(_gfock_lci_raw, dtype=xp.float64) - xp.asarray(_gfock_zero_xp, dtype=xp.float64)
                        _dme0_lci = C @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C.T
                        _dme0_lorb = _build_dme0_lorb_response(
                            B_ao, h_ao, C, np.asarray(Lorb_mat, dtype=np.float64),
                            dm1_sa_xp, dm2_sa_xp,
                            ppaa_sa, papa_sa,
                            ncore=int(ncore), ncas=int(ncas),
                            vhf_cache=_vhf_cache,
                            lorb_cache=_lorb_resp_cache,
                            return_xp=True,
                        )
                        _W_pulay_ms = W_K_xp - W_sa_xp + _dme0_lci + _dme0_lorb
                    gfock_K_np = None
                    W_K = None
                else:
                    with _main_stream_cm:
                        gfock_K_np = _asnumpy_f64(gfock_K)
                    _tmp_K = C_np @ gfock_K_np[:, :nocc]  # (nao, nocc)
                    W_K = 0.5 * (_tmp_K @ C_occ_np.T + C_occ_np @ _tmp_K.T)
                    _dgfock_lci = _asnumpy_f64(_gfock_lci_raw) - _gfock_zero_np
                    _dme0_lci = C_np @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C_np.T
                    _dme0_lorb = _build_dme0_lorb_response(
                        B_ao, h_ao, C_np, np.asarray(Lorb_mat, dtype=np.float64),
                        _asnumpy_f64(dm1_sa), _asnumpy_f64(dm2_sa),
                        ppaa_sa, papa_sa,
                        ncore=int(ncore), ncas=int(ncas),
                        vhf_cache=_vhf_cache,
                        lorb_cache=_lorb_resp_cache,
                    )
                    _W_pulay_ms = W_K - W_sa + _dme0_lci + _dme0_lorb
                de_pulay_delta = _contract_pulay_fast(_W_pulay_ms)

                _in_flight.append(
                    {
                        "K": int(K),
                        "stream": stream,
                        "grad_dev": grad_dev,
                        "bar_L": bar_L_contract,
                        "de_h1_delta": np.asarray(de_h1_delta, dtype=np.float64),
                        "de_pulay_delta": np.asarray(de_pulay_delta, dtype=np.float64),
                    }
                )
                # Current root payload is now queued; release local refs before
                # deciding whether to drain an older job.
                del (
                    de_h1_delta, de_pulay_delta,
                    D_1e_delta,
                    gfock_K_np, _tmp_K, W_K,
                    _gfock_lci_raw, _dgfock_lci, _dme0_lci, _dme0_lorb, _W_pulay_ms,
                    rhs_orb, rhs_ci_K, rhs_ci, g_K,
                    Lci_list, Lorb_mat,
                    D_tot_K, D_act_lci, D_L_lorb,
                    dm1_lci, dm2_lci,
                    bar_L_delta_total, bar_L_contract,
                )
                _flush_gpu_pool()

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
                        if bool(is_spherical):
                            raise
                        if not warned_df_fd:
                            warnings.warn(
                                f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                                "expect noisy/non-conservative forces in MD. "
                                f"Reason: {type(e).__name__}: {e}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            warned_df_fd = True
                        de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                            ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=job["bar_L"],
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
                    del de_df_delta, grad_done, job
                    _flush_gpu_pool()
                continue

        # Synchronous contraction (CPU / single-stream CUDA fallback)
        if de_df_delta_precomputed is not None:
            de_df_delta = np.asarray(de_df_delta_precomputed, dtype=np.float64)
        else:
            try:
                t0 = time.perf_counter()
                if bool(_fused_contract_enabled) and df_grad_ctx is not None and bar_L_terms_contract is not None:
                    if bool(is_spherical):
                        de_df_delta = df_grad_ctx.contract_fused_terms_sph(
                            B_sph=B_ao,
                            bar_L_terms_sph=bar_L_terms_contract,
                            T_c2s=None,
                            precision=str(_fused_contract_precision),
                        )
                    else:
                        de_df_delta = df_grad_ctx.contract_fused_terms(
                            B_ao=B_ao,
                            bar_L_terms=bar_L_terms_contract,
                            precision=str(_fused_contract_precision),
                        )
                else:
                    if bool(is_spherical):
                        if df_grad_ctx is not None:
                            de_df_delta = df_grad_ctx.contract_sph(B_sph=B_ao, bar_L_sph=bar_L_contract, T_c2s=None)
                        else:
                            de_df_delta = compute_df_gradient_contributions_analytic_sph(
                                ao_basis, aux_basis, atom_coords_bohr=coords, B_sph=B_ao, bar_L_sph=bar_L_contract,
                                T_c2s=None, L_chol=getattr(scf_out, "df_L", None),
                                backend=str(df_backend), df_threads=int(df_threads), profile=None,
                            )
                    else:
                        if df_grad_ctx is not None:
                            de_df_delta = df_grad_ctx.contract(B_ao=B_ao, bar_L_ao=bar_L_contract)
                        else:
                            de_df_delta = compute_df_gradient_contributions_analytic_packed_bases(
                                ao_basis, aux_basis, atom_coords_bohr=coords, B_ao=B_ao, bar_L_ao=bar_L_contract,
                                L_chol=getattr(scf_out, "df_L", None),
                                backend=str(df_backend), df_threads=int(df_threads), profile=None,
                            )
                if _profile_df_per_root:
                    _t_contract_delta += time.perf_counter() - t0
            except (NotImplementedError, RuntimeError, ValueError) as e:
                if bool(is_spherical):
                    raise
                if not warned_df_fd:
                    warnings.warn(
                        f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                        "expect noisy/non-conservative forces in MD. "
                        f"Reason: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_df_fd = True
                if bool(_fused_contract_enabled) and bar_L_terms_contract is not None:
                    _bar_L_dt_fd = None
                    for _term in bar_L_terms_contract:
                        _term64 = xp.asarray(_term, dtype=xp.float64)
                        if _bar_L_dt_fd is None:
                            _bar_L_dt_fd = _term64
                        else:
                            _bar_L_dt_fd += _term64
                    if _bar_L_dt_fd is None:
                        raise RuntimeError("internal error: fused term list was empty")
                else:
                    _bar_L_dt_fd = bar_L_contract
                de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                    ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=_bar_L_dt_fd,
                    backend=str(df_backend), df_config=df_config, df_threads=int(df_threads),
                    delta_bohr=float(delta_bohr), profile=None,
                )

        # ── Single fused 1e call: delta from SA + lci + lorb ──
        if _gpu_1e_tensor_path:
            with _main_stream_cm:
                D_1e_delta = xp.asarray(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb, dtype=xp.float64)
        else:
            with _main_stream_cm:
                D_1e_delta = _asnumpy_f64(D_tot_K - D_tot_sa + D_act_lci + D_L_lorb)
        de_h1_delta = _contract_hcore_fast(D_1e_delta)

        # ── Per-root Pulay delta: -Tr((W_K - W_sa) · dS/dR) ──
        t0 = time.perf_counter()
        # ── Response Pulay: energy-weighted density from CI + orbital response ──
        # CI response: perturbation gfock from response RDMs (MO basis).
        _gfock_lci_raw, _, _, _, _ = _build_gfock_casscf_df(
            B_ao, h_ao, C, ncore=int(ncore), ncas=int(ncas),
            dm1_act=xp.asarray(dm1_lci, dtype=xp.float64),
            dm2_act=xp.asarray(dm2_lci, dtype=xp.float64),
        )
        if _gpu_1e_tensor_path and W_sa_xp is not None and C_occ_xp is not None:
            with _main_stream_cm:
                gfock_K_xp = xp.asarray(gfock_K, dtype=xp.float64)
                _tmp_K = C @ gfock_K_xp[:, :nocc]  # (nao, nocc)
                W_K_xp = 0.5 * (_tmp_K @ C_occ_xp.T + C_occ_xp @ _tmp_K.T)
                _dgfock_lci = xp.asarray(_gfock_lci_raw, dtype=xp.float64) - xp.asarray(_gfock_zero_xp, dtype=xp.float64)
                _dme0_lci = C @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C.T
                _dme0_lorb = _build_dme0_lorb_response(
                    B_ao, h_ao, C, np.asarray(Lorb_mat, dtype=np.float64),
                    dm1_sa_xp, dm2_sa_xp,
                    ppaa_sa, papa_sa,
                    ncore=int(ncore), ncas=int(ncas),
                    vhf_cache=_vhf_cache,
                    lorb_cache=_lorb_resp_cache,
                    return_xp=True,
                )
                _W_delta = W_K_xp - W_sa_xp + _dme0_lci + _dme0_lorb
        else:
            with _main_stream_cm:
                gfock_K_np = _asnumpy_f64(gfock_K)
            _tmp_K = C_np @ gfock_K_np[:, :nocc]  # (nao, nocc)
            W_K = 0.5 * (_tmp_K @ C_occ_np.T + C_occ_np @ _tmp_K.T)
            _dgfock_lci = _asnumpy_f64(_gfock_lci_raw) - _gfock_zero_np
            _dme0_lci = C_np @ (0.5 * (_dgfock_lci + _dgfock_lci.T)) @ C_np.T
            _dme0_lorb = _build_dme0_lorb_response(
                B_ao, h_ao, C_np, np.asarray(Lorb_mat, dtype=np.float64),
                _asnumpy_f64(dm1_sa), _asnumpy_f64(dm2_sa),
                ppaa_sa, papa_sa,
                ncore=int(ncore), ncas=int(ncas),
                vhf_cache=_vhf_cache,
                lorb_cache=_lorb_resp_cache,
            )
            _W_delta = W_K - W_sa + _dme0_lci + _dme0_lorb
        de_pulay_delta = _contract_pulay_fast(_W_delta)
        if _profile_df_per_root:
            _t_response_pulay += time.perf_counter() - t0

        # Step F: Combine — grad_sa_base + delta 2e + delta 1e + delta Pulay
        if bar_L_terms_contract is not None:
            bar_L_terms_contract.clear()
            bar_L_terms_contract = None
        if bar_L_contract is not None and bar_L_contract is not bar_L_delta_total:
            del bar_L_contract
        if bar_L_delta_total is not None:
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
                if bool(is_spherical):
                    raise
                if not warned_df_fd:
                    warnings.warn(
                        f"DF 2e gradient contraction fell back to finite-difference on B (backend={df_backend!s}); "
                        "expect noisy/non-conservative forces in MD. "
                        f"Reason: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_df_fd = True
                de_df_delta = compute_df_gradient_contributions_fd_packed_bases(
                    ao_basis, aux_basis, atom_coords_bohr=coords, bar_L_ao=job["bar_L"],
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
            del de_df_delta, grad_done, job
            _flush_gpu_pool()

    _restore_pool()
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
            f"barL_hybrid={int(bool(_barl_hybrid_enabled))} "
            f"barL_stage={_barl_stage_effective} "
            f"barL_ab_checked={int(bool(_barl_ab_checked))} "
            f"barL_ab_max={float(_barl_ab_diff_max):.3e} "
            f"barL_fallback={int(bool(_barl_fallback))} "
            f"fused_contract={int(bool(_fused_contract_requested))} "
            f"fused_contract_live={int(bool(_fused_contract_enabled))} "
            f"fused_precision={str(_fused_contract_precision)} "
            f"z_solve_total={_t_z_solve:.3f}s z_solver={_z_solver or 'unknown'} "
            f"z_solver_detail={_z_solver_detail or _z_solver or 'unknown'} "
            f"z_backend={_z_backend or 'unknown'} "
            f"z_niter={int(_z_niter)} z_matvec_calls={int(_z_matvec_calls)} "
            f"t_trans_rdm_lci={_t_trans_rdm_lci:.3f}s "
            f"t_gen_g_hop={_t_gen_g_hop:.3f}s t_gfock={_t_gfock:.3f}s "
            f"t_response_pulay={_t_response_pulay:.3f}s",
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
