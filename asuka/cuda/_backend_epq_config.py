from __future__ import annotations

import numpy as np


EPQ_I32_MAX_NNZ = int(np.iinfo(np.int32).max)


def normalize_epq_indptr_mode(indptr_dtype) -> str:
    """Normalize EPQ indptr dtype selector to one of: auto|int32|int64."""
    if indptr_dtype is None:
        return "auto"
    if isinstance(indptr_dtype, str):
        mode = indptr_dtype.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("int32", "i4"):
            return "int32"
        if mode in ("int64", "i8"):
            return "int64"
        raise ValueError("indptr_dtype must be one of: auto, int32, int64")
    dt = np.dtype(indptr_dtype)
    if dt == np.dtype(np.int32):
        return "int32"
    if dt == np.dtype(np.int64):
        return "int64"
    raise ValueError("indptr_dtype must be one of: auto, int32, int64")


def normalize_epq_blocked_transpose_mode(mode) -> str:
    """Normalize blocked EPQ transpose mode to one of: auto|on|off."""
    if mode is None:
        return "auto"
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m in ("", "auto"):
            return "auto"
        if m in ("on", "true", "1", "yes", "force", "transpose"):
            return "on"
        if m in ("off", "false", "0", "no", "disable", "disabled", "fallback"):
            return "off"
        raise ValueError("epq_blocked_transpose must be one of: auto, on, off")
    return "on" if bool(mode) else "off"


def resolve_epq_blocked_transpose_mode_with_env(mode, env_mode) -> str:
    """Resolve blocked-transpose mode using env when caller leaves mode in auto/default."""
    env = str(env_mode or "").strip()
    raw = mode
    if env:
        raw_s = str(raw).strip().lower() if raw is not None else ""
        if raw is None or raw_s in ("", "auto"):
            raw = env
    return normalize_epq_blocked_transpose_mode(raw)


def resolve_epq_blocked_transpose_reserve_mib_with_env(reserve_mib, env_reserve_mib) -> int:
    """Resolve reserve MiB using env when caller leaves default/auto-like reserve."""
    raw = 512 if reserve_mib is None else int(reserve_mib)
    env = str(env_reserve_mib or "").strip()
    if env and (reserve_mib is None or int(raw) <= 0 or int(raw) == 512):
        try:
            raw = int(env)
        except Exception as e:
            raise ValueError("ASUKA_CUGUGA_EPQ_BLOCKED_TRANSPOSE_RESERVE_MIB must be an integer") from e
    return max(0, int(raw))


def normalize_matvec_cuda_path_mode(mode) -> str:
    """Normalize CUDA matvec path mode to one of: auto|epq_blocked|fused_epq_hybrid."""
    if mode is None:
        return "auto"
    m = str(mode).strip().lower()
    if m in ("", "auto"):
        return "auto"
    if m in ("fused_coo", "fused-coo", "coo"):
        raise ValueError(
            "matvec_cuda_path_mode='fused_coo' is disabled (no-go path due to performance). "
            "Use 'auto', 'fused_epq_hybrid', or 'epq_blocked'."
        )
    if m in ("epq_blocked", "epq-blocked", "epq"):
        return "epq_blocked"
    if m in ("fused_epq_hybrid", "fused-epq-hybrid", "fused_epq"):
        return "fused_epq_hybrid"
    raise ValueError("matvec_cuda_path_mode must be one of: auto, epq_blocked, fused_epq_hybrid")


def epq_indptr_cp_dtype_for_total_nnz(cp, *, mode: str, total_nnz: int):
    """Resolve concrete CuPy dtype for EPQ indptr under runtime nnz guard."""
    total_nnz = int(total_nnz)
    if total_nnz < 0:
        raise ValueError("total_nnz must be >= 0")
    if mode == "int64":
        return cp.int64
    if mode == "int32":
        if total_nnz > EPQ_I32_MAX_NNZ:
            raise ValueError(
                f"indptr_dtype=int32 requires total_nnz <= {EPQ_I32_MAX_NNZ}, got {total_nnz}"
            )
        return cp.int32
    if total_nnz <= EPQ_I32_MAX_NNZ:
        return cp.int32
    return cp.int64
