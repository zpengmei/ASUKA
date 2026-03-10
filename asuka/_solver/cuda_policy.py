from __future__ import annotations

import os
from typing import Any

import numpy as np

from .warm_state import WARM_CUDA_MATVEC_BACKENDS


def cuda_budget_free_bytes(cp_mod: Any, hard_cap_gib: float) -> int | None:
    """Return available CUDA bytes after applying an optional hard-cap budget.

    Uses CuPy runtime free/total memory and adds reclaimable bytes from the CuPy
    default memory pool so long-lived processes are not penalized by cached blocks.
    """

    try:
        free_b, total_b = cp_mod.cuda.runtime.memGetInfo()
        free_b = int(free_b)
        total_b = int(total_b)
        try:
            pool = cp_mod.get_default_memory_pool()
            pool_free_b = int(pool.total_bytes()) - int(pool.used_bytes())
            if pool_free_b > 0:
                free_b += int(pool_free_b)
        except Exception:
            pass
    except Exception:
        return None

    budget_free_b = int(free_b)
    if float(hard_cap_gib) > 0.0:
        hard_cap_b = int(float(hard_cap_gib) * 1024 * 1024 * 1024)
        used_b = max(0, int(total_b) - int(free_b))
        budget_free_b = min(int(budget_free_b), max(0, int(hard_cap_b) - int(used_b)))
    return max(0, int(budget_free_b))


def apply_cuda_pool_hard_cap(cp_mod: Any, hard_cap_gib: float) -> int | None:
    """Apply CuPy memory-pool hard cap (bytes) for the current device."""

    if float(hard_cap_gib) <= 0.0:
        return None
    hard_cap_b = int(float(hard_cap_gib) * 1024 * 1024 * 1024)
    try:
        pool = cp_mod.get_default_memory_pool()
        pool.set_limit(size=int(hard_cap_b))
    except Exception:
        return None
    return int(hard_cap_b)


def estimate_epq_peak_bytes(ncsf: int, norb: int) -> int:
    n_pairs = int(norb) * (int(norb) - 1)
    ntasks = int(ncsf) * int(n_pairs)
    est_counts_bytes = int(ntasks) * 4
    est_table_bytes = (int(ncsf) + 1) * 8 + int(ntasks) * 16
    return int(est_counts_bytes) + int(est_table_bytes)


def normalize_cuda_user_policy_mode(value: bool | str | None) -> str:
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "disabled"):
            return "off"
        raise ValueError("matvec_cuda_policy must be bool or one of: auto/on/off")
    if value is None:
        return "auto"
    return "on" if bool(value) else "off"


def normalize_cuda_accuracy_mode(value: str | None) -> str:
    if value is None:
        return "balanced"
    mode = str(value).strip().lower()
    aliases = {
        "default": "balanced",
        "normal": "balanced",
        "performance": "fast",
        "accurate": "strict",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("fast", "balanced", "strict"):
        raise ValueError("matvec_cuda_accuracy_mode must be one of: fast, balanced, strict")
    return mode


def normalize_matvec_cuda_path_mode(value: str | None) -> str:
    mode = "auto" if value is None else str(value).strip().lower()
    aliases = {
        "fused-coo": "fused_coo",
        "coo": "fused_coo",
        "epq-blocked": "epq_blocked",
        "epq": "epq_blocked",
        "fused-epq-hybrid": "fused_epq_hybrid",
        "fused_epq": "fused_epq_hybrid",
    }
    mode = aliases.get(mode, mode)
    if mode in ("", "auto"):
        return "auto"
    if mode == "fused_coo":
        raise ValueError(
            "matvec_cuda_path_mode='fused_coo' is disabled (no-go path due to performance). "
            "Use 'auto', 'fused_epq_hybrid', or 'epq_blocked'."
        )
    if mode not in ("epq_blocked", "fused_epq_hybrid"):
        raise ValueError(
            "matvec_cuda_path_mode must be one of: auto, epq_blocked, fused_epq_hybrid"
        )
    return mode


def apply_cuda_user_policy(
    *,
    matvec_backend: str,
    policy_mode: str,
    accuracy_mode: str,
    dtype_hint: str,
    memory_cap_gib: float | None,
    kwargs: dict[str, Any],
    policy_explicit: bool,
    policy_configured: bool,
) -> tuple[bool, str, dict[str, Any]]:
    """Apply high-level CUDA policy defaults without overriding expert knobs."""

    backend = str(matvec_backend).strip().lower()
    if backend not in WARM_CUDA_MATVEC_BACKENDS:
        return False, "non_cuda_backend", {}

    mode = normalize_cuda_user_policy_mode(policy_mode)
    acc_mode = normalize_cuda_accuracy_mode(accuracy_mode)
    if mode == "off":
        return False, "policy_off", {}
    if mode == "auto" and (not policy_explicit) and (not policy_configured):
        return False, "policy_auto_inactive", {}

    dtype_mode = str(dtype_hint).strip().lower()
    if dtype_mode in ("fp64", "f64", "double"):
        dtype_mode = "float64"
    elif dtype_mode in ("fp32", "f32", "single"):
        dtype_mode = "float32"
    elif dtype_mode in ("mixed_fp32", "float32_mixed"):
        dtype_mode = "mixed"
    if dtype_mode not in ("float64", "float32", "mixed"):
        dtype_mode = "float64"

    resolved: dict[str, Any] = {}

    def _set_default(key: str, value: Any) -> None:
        if key in kwargs:
            return
        kwargs[key] = value
        resolved[key] = value

    if memory_cap_gib is not None:
        _set_default("matvec_cuda_mem_hard_cap_gib", float(memory_cap_gib))

    _set_default("matvec_cuda_aggregate_offdiag", True)
    _set_default("matvec_cuda_use_epq_table", True)
    _set_default("matvec_cuda_epq_streaming", "off")
    _set_default("matvec_cuda_cache_csr_tiles", False)
    _set_default("matvec_cuda_prefilter_trivial_tasks", "auto")

    if acc_mode == "fast":
        _set_default("matvec_cuda_max_g_mib", 512.0)
        if dtype_mode == "mixed":
            _set_default("matvec_cuda_mixed_threshold", 1e-7)
    elif acc_mode == "strict":
        _set_default("matvec_cuda_max_g_mib", 128.0)
        if dtype_mode == "mixed":
            _set_default("matvec_cuda_mixed_threshold", 1e-3)
    else:
        _set_default("matvec_cuda_max_g_mib", 256.0)
        if dtype_mode == "mixed":
            _set_default("matvec_cuda_mixed_threshold", 1e-5)

    if dtype_mode == "float64":
        _set_default("matvec_cuda_gemm_backend", "cublaslt_fp64")
    elif dtype_mode == "float32":
        _set_default("matvec_cuda_gemm_backend", "cublaslt_tf32")
    else:
        _set_default("matvec_cuda_gemm_backend", "cublaslt_fp64")

    return True, ("policy_on" if mode == "on" else "policy_auto"), resolved


def resolve_kernel_cuda_policy(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    matvec_backend: str,
    strict_gpu: bool,
) -> dict[str, Any]:
    """Resolve/apply kernel CUDA policy inputs and return profiling metadata."""

    cuda_policy_key_present = "matvec_cuda_policy" in kwargs
    cuda_acc_mode_key_present = "matvec_cuda_accuracy_mode" in kwargs
    cuda_mem_cap_alias_key_present = "matvec_cuda_memory_cap_gib" in kwargs

    matvec_cuda_policy_mode = normalize_cuda_user_policy_mode(
        kwargs.pop("matvec_cuda_policy", getattr(defaults, "matvec_cuda_policy", "auto"))
    )
    matvec_cuda_accuracy_mode = normalize_cuda_accuracy_mode(
        kwargs.pop("matvec_cuda_accuracy_mode", getattr(defaults, "matvec_cuda_accuracy_mode", "balanced"))
    )
    matvec_cuda_mem_cap_alias_in = kwargs.pop(
        "matvec_cuda_memory_cap_gib",
        getattr(defaults, "matvec_cuda_memory_cap_gib", None),
    )
    matvec_cuda_mem_cap_alias = None
    if matvec_cuda_mem_cap_alias_in is not None:
        matvec_cuda_mem_cap_alias = float(matvec_cuda_mem_cap_alias_in)
        if float(matvec_cuda_mem_cap_alias) <= 0.0:
            raise ValueError("matvec_cuda_memory_cap_gib must be > 0 when provided")

    cuda_policy_explicit = bool(
        cuda_policy_key_present or cuda_acc_mode_key_present or cuda_mem_cap_alias_key_present
    )
    cuda_policy_configured = bool(
        str(matvec_cuda_accuracy_mode) != "balanced" or matvec_cuda_mem_cap_alias is not None
    )
    matvec_cuda_dtype_hint = str(
        kwargs.get("matvec_cuda_dtype", getattr(defaults, "matvec_cuda_dtype", "float64"))
    ).strip().lower()
    cuda_policy_applied, cuda_policy_reason, cuda_policy_resolved = apply_cuda_user_policy(
        matvec_backend=matvec_backend,
        policy_mode=matvec_cuda_policy_mode,
        accuracy_mode=matvec_cuda_accuracy_mode,
        dtype_hint=matvec_cuda_dtype_hint,
        memory_cap_gib=matvec_cuda_mem_cap_alias,
        kwargs=kwargs,
        policy_explicit=cuda_policy_explicit,
        policy_configured=cuda_policy_configured,
    )

    matvec_cuda_aggregate_offdiag_preview = None
    if str(matvec_backend) in ("cuda_eri_mat", "cuda"):
        aggregate_preview_in = kwargs.get(
            "matvec_cuda_aggregate_offdiag",
            getattr(defaults, "matvec_cuda_aggregate_offdiag", None),
        )
        if aggregate_preview_in is None:
            matvec_cuda_aggregate_offdiag_preview = True
        else:
            matvec_cuda_aggregate_offdiag_preview = bool(aggregate_preview_in)
        enforce_cuda_aggregate_offdiag_guard(
            bool(matvec_cuda_aggregate_offdiag_preview),
            context="kernel(cuda)",
        )

    profile: dict[str, Any] = {
        "strict_gpu": bool(strict_gpu),
        "matvec_cuda_policy": str(matvec_cuda_policy_mode),
        "matvec_cuda_accuracy_mode": str(matvec_cuda_accuracy_mode),
        "matvec_cuda_policy_applied": bool(cuda_policy_applied),
        "matvec_cuda_policy_reason": str(cuda_policy_reason),
    }
    if matvec_cuda_mem_cap_alias is not None:
        profile["matvec_cuda_memory_cap_gib_alias"] = float(matvec_cuda_mem_cap_alias)
    if cuda_policy_resolved:
        profile["matvec_cuda_policy_resolved"] = {
            str(kk): vv for kk, vv in dict(cuda_policy_resolved).items()
        }

    return {
        "matvec_cuda_policy_mode": str(matvec_cuda_policy_mode),
        "matvec_cuda_accuracy_mode": str(matvec_cuda_accuracy_mode),
        "matvec_cuda_mem_cap_alias": matvec_cuda_mem_cap_alias,
        "cuda_policy_applied": bool(cuda_policy_applied),
        "cuda_policy_reason": str(cuda_policy_reason),
        "cuda_policy_resolved": dict(cuda_policy_resolved),
        "matvec_cuda_aggregate_offdiag_preview": matvec_cuda_aggregate_offdiag_preview,
        "profile": profile,
    }


def resolve_cuda_memory_controls(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    consume: bool = False,
) -> dict[str, float]:
    """Resolve CUDA memory hard-cap and workspace-cache fraction controls."""

    if consume:
        mem_cap_in = kwargs.pop(
            "matvec_cuda_mem_hard_cap_gib",
            getattr(defaults, "matvec_cuda_mem_hard_cap_gib", 11.5),
        )
        ws_cache_frac_in = kwargs.pop(
            "matvec_cuda_ws_cache_fraction",
            getattr(defaults, "matvec_cuda_ws_cache_fraction", 0.2),
        )
    else:
        mem_cap_in = kwargs.get(
            "matvec_cuda_mem_hard_cap_gib",
            getattr(defaults, "matvec_cuda_mem_hard_cap_gib", 11.5),
        )
        ws_cache_frac_in = kwargs.get(
            "matvec_cuda_ws_cache_fraction",
            getattr(defaults, "matvec_cuda_ws_cache_fraction", 0.2),
        )
    return {
        "matvec_cuda_mem_hard_cap_gib": float(mem_cap_in),
        "matvec_cuda_ws_cache_fraction": float(normalize_ws_cache_fraction(ws_cache_frac_in)),
    }


def enforce_cuda_fp32_large_cas_epq_policy(
    *,
    context: str,
    matvec_cuda_dtype: str,
    matvec_cuda_use_epq_table: bool,
    matvec_cuda_aggregate_offdiag: bool,
    ncsf: int,
) -> None:
    dtype_mode = str(matvec_cuda_dtype).strip().lower()
    if dtype_mode != "mixed":
        return
    if not bool(matvec_cuda_aggregate_offdiag):
        return
    if bool(matvec_cuda_use_epq_table):
        return
    if int(ncsf) < 1_000_000:
        return
    raise ValueError(
        f"{context}: matvec_cuda_dtype='{dtype_mode}' with large CAS (ncsf={int(ncsf)}) "
        "requires matvec_cuda_use_epq_table=True when matvec_cuda_aggregate_offdiag=True; "
        "the no-EPQ large-CAS mixed path is disabled to prevent CUDA crashes. "
        "Use matvec_cuda_dtype='float64', reduce CAS size, or increase memory budget so EPQ stays enabled."
    )


def resolve_epq_overbudget_action(
    *,
    matvec_cuda_dtype: str,
    matvec_cuda_aggregate_offdiag: bool,
    ncsf: int,
    epq_table_forced: bool,
    epq_streaming_mode: str,
    has_epq_table_device_build: bool,
) -> tuple[str, str]:
    """Decide EPQ policy when estimated materialization exceeds budget."""

    dtype_mode = str(matvec_cuda_dtype).strip().lower()
    stream_mode = str(epq_streaming_mode).strip().lower()
    streaming_explicit = stream_mode in ("on", "manual")
    mixed_guarded = bool(
        dtype_mode == "mixed"
        and bool(matvec_cuda_aggregate_offdiag)
        and int(ncsf) >= 1_000_000
    )

    if bool(epq_table_forced):
        return "keep_materialized", "forced_use_epq_table"

    if mixed_guarded:
        if streaming_explicit and bool(has_epq_table_device_build):
            return "streaming", "explicit_streaming_mode"
        return "keep_materialized", "mixed_guarded_keep_materialized"

    if streaming_explicit:
        if bool(has_epq_table_device_build):
            return "streaming", "explicit_streaming_mode"
        return "disable_epq", "streaming_requested_but_unavailable"

    can_stream_auto = bool(
        dtype_mode in ("float32", "mixed")
        and bool(matvec_cuda_aggregate_offdiag)
        and stream_mode != "off"
        and bool(has_epq_table_device_build)
    )
    if can_stream_auto:
        return "streaming", "auto_streaming_low_precision_overbudget"
    return "disable_epq", "disable_epq_overbudget"


def resolve_mixed_low_workspace_oom_fallback(
    *,
    can_stream_fallback: bool,
    can_noepq_fallback: bool,
    guarded_requires_epq: bool,
) -> tuple[str, str]:
    """Pick mixed low-workspace fallback action after materialized-EPQ OOM."""

    if bool(can_noepq_fallback):
        if bool(guarded_requires_epq):
            return "no_epq", "oom_materialized_epq_prefer_no_epq_guard_override"
        return "no_epq", "oom_materialized_epq_prefer_no_epq"
    if bool(can_stream_fallback):
        if bool(guarded_requires_epq):
            return "streaming", "oom_materialized_epq_guarded_requires_epq"
        return "streaming", "oom_materialized_epq_streaming_fallback"
    return "raise", "oom_no_supported_fallback"


def normalize_csr_host_cache_mode(value: bool | str | None) -> str:
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "host", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "none", "disabled"):
            return "off"
        raise ValueError("matvec_cuda_csr_host_cache must be bool or one of: auto/on/off")
    return "on" if bool(value) else "off"


def normalize_prefilter_trivial_tasks_mode(value: bool | str | None) -> str:
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "none", "disabled"):
            return "off"
        raise ValueError("matvec_cuda_prefilter_trivial_tasks must be bool or one of: auto/on/off")
    return "on" if bool(value) else "off"


def enforce_cuda_aggregate_offdiag_guard(value: bool, *, context: str) -> bool:
    """Hard guard: CUDA paths must keep aggregate_offdiag enabled."""
    if not bool(value):
        raise ValueError(
            f"{context}: matvec_cuda_aggregate_offdiag=False is forbidden by hard guard; "
            "set matvec_cuda_aggregate_offdiag=True"
        )
    return True


def normalize_ws_cache_fraction(value: Any) -> float:
    try:
        frac = float(value)
    except Exception:
        frac = 0.2
    if not np.isfinite(frac):
        frac = 0.2
    return max(0.0, min(0.8, float(frac)))


_DETECTED_GPU_MEM_CAP_GIB: float | None = None


def auto_gpu_mem_hard_cap() -> float:
    """Return a safe VRAM hard cap based on the current CUDA device."""
    global _DETECTED_GPU_MEM_CAP_GIB
    if _DETECTED_GPU_MEM_CAP_GIB is not None:
        return _DETECTED_GPU_MEM_CAP_GIB

    env_val = os.environ.get("ASUKA_CUDA_MEM_HARD_CAP_GIB")
    if env_val is not None:
        try:
            cap = float(env_val)
            if cap > 0.0:
                _DETECTED_GPU_MEM_CAP_GIB = cap
                return cap
        except (ValueError, TypeError):
            pass

    try:
        from asuka.cuguga.autotune import detect_cuda_device_info

        info = detect_cuda_device_info()
        if info is not None and isinstance(info.get("total_mem_gib"), (int, float)):
            total = float(info["total_mem_gib"])
            if total > 0.0:
                reserve = 1.5 if total >= 20.0 else (1.0 if total >= 12.0 else 0.75)
                cap = max(2.0, min(total * 0.90, total - reserve))
                _DETECTED_GPU_MEM_CAP_GIB = round(cap, 3)
                return _DETECTED_GPU_MEM_CAP_GIB
    except Exception:
        pass

    _DETECTED_GPU_MEM_CAP_GIB = 11.5
    return 11.5
