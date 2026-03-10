from __future__ import annotations

import numpy as np
from typing import Any

from .cuda_policy import (
    enforce_cuda_aggregate_offdiag_guard,
    normalize_ws_cache_fraction as _normalize_ws_cache_fraction,
)


CUDA_MATVEC_BACKENDS = frozenset(
    ("cuda_eri_mat", "cuda", "cuda_fixed_ell", "cuda_ell", "cuda_fixed_sell", "cuda_sell")
)


def ws_needs_rebuild(
    ws: Any,
    *,
    expected_dtype: Any,
    j_tile: int,
    csr_capacity_mult: float,
    threads_enum: int,
    threads_g: int,
    threads_w: int,
    threads_apply: int,
    max_g_bytes: int,
    coalesce: bool,
    include_diagonal_rs: bool,
    fuse_count_write: bool,
    fp32_coeff_data: bool,
    path_mode: str,
    use_fused_hop: bool,
    use_epq_table: bool,
    aggregate_offdiag_k: bool,
    l_full_d: Any,
    enable_fp64_emulation: bool,
    gemm_backend: str,
    emulation_strategy: str,
    cublas_workspace_cap_mb: int,
    apply_mode: str,
    epq_build_device: bool,
    epq_build_j_tile: int,
    epq_streaming: bool,
    epq_stream_j_tile: int,
    epq_stream_use_recompute: str,
    cache_csr_tiles: bool,
    csr_host_cache_mode: str,
    csr_host_cache_budget_gib: float,
    csr_host_cache_min_ncsf: int,
    csr_pipeline_streams_mode: str,
    csr_pipeline_streams_value: int | None,
    csr_pipeline_min_ncsf: int,
    prefilter_trivial_tasks_mode: str,
    prefilter_trivial_tasks_min_ncsf: int,
) -> bool:
    """Return True if a CUDA matvec workspace is incompatible with requested settings."""
    if ws is None:
        return True
    ws_path_mode = str(getattr(ws, "path_mode", "auto"))
    return (
        np.dtype(getattr(ws, "dtype", np.float64)) != np.dtype(expected_dtype)
        or int(getattr(ws, "j_tile", -1)) != j_tile
        or float(getattr(ws, "csr_capacity_mult", -1.0)) != float(csr_capacity_mult)
        or int(getattr(ws, "threads_enum", -1)) != threads_enum
        or int(getattr(ws, "threads_g", -1)) != threads_g
        or int(getattr(ws, "threads_w", -1)) != threads_w
        or int(getattr(ws, "threads_apply", -1)) != threads_apply
        or int(getattr(ws, "max_g_bytes", -1)) != int(max_g_bytes)
        or bool(getattr(ws, "coalesce", False)) != coalesce
        or bool(getattr(ws, "include_diagonal_rs", False)) != include_diagonal_rs
        or bool(getattr(ws, "fuse_count_write", False)) != bool(fuse_count_write)
        or bool(getattr(ws, "fp32_coeff_data", False)) != bool(fp32_coeff_data)
        or str(getattr(ws, "path_mode_requested", "auto")) != str(path_mode)
        or bool(getattr(ws, "use_fused_hop", True)) != bool(use_fused_hop)
        or (
            ws_path_mode != "fused_coo"
            and bool(getattr(ws, "use_epq_table", False)) != bool(use_epq_table)
        )
        or (
            ws_path_mode not in ("fused_coo", "fused_epq_hybrid")
            and bool(getattr(ws, "aggregate_offdiag_k", False)) != bool(aggregate_offdiag_k)
        )
        or bool(getattr(ws, "l_full", None) is not None) != bool(l_full_d is not None)
        or bool(getattr(ws, "offdiag_enable_fp64_emulation", False)) != bool(enable_fp64_emulation)
        or str(getattr(ws, "gemm_backend", "")) != str(gemm_backend)
        or str(getattr(ws, "offdiag_emulation_strategy", "")) != str(emulation_strategy)
        or int(getattr(ws, "offdiag_cublas_workspace_cap_mb", 0)) != int(cublas_workspace_cap_mb)
        or str(getattr(ws, "apply_mode", "")) != str(apply_mode)
        or bool(getattr(ws, "epq_build_device", False)) != bool(epq_build_device)
        or int(getattr(ws, "epq_build_j_tile", 0)) != int(epq_build_j_tile)
        or bool(getattr(ws, "epq_streaming", False)) != bool(epq_streaming)
        or int(getattr(ws, "epq_stream_j_tile", 0)) != int(epq_stream_j_tile)
        or str(getattr(ws, "epq_stream_use_recompute", "auto")) != str(epq_stream_use_recompute)
        or bool(getattr(ws, "cache_csr_tiles", False)) != bool(cache_csr_tiles)
        or str(getattr(ws, "csr_host_cache_mode", "off")) != str(csr_host_cache_mode)
        or float(getattr(ws, "csr_host_cache_budget_gib", -1.0)) != float(csr_host_cache_budget_gib)
        or int(getattr(ws, "csr_host_cache_min_ncsf", -1)) != int(csr_host_cache_min_ncsf)
        or str(getattr(ws, "csr_pipeline_streams_mode", "off")) != str(csr_pipeline_streams_mode)
        or (
            csr_pipeline_streams_value is not None
            and int(getattr(ws, "csr_pipeline_streams", 0)) != int(csr_pipeline_streams_value)
        )
        or int(getattr(ws, "csr_pipeline_min_ncsf", -1)) != int(csr_pipeline_min_ncsf)
        or str(getattr(ws, "prefilter_trivial_tasks_mode", "off")) != str(prefilter_trivial_tasks_mode)
        or int(getattr(ws, "prefilter_trivial_tasks_min_ncsf", -1)) != int(prefilter_trivial_tasks_min_ncsf)
        or (
            l_full_d is not None
            and int(getattr(ws, "naux", 0)) != int(getattr(l_full_d, "shape", (0, 0))[1])
        )
    )


def release_matvec_cuda_workspace(ws: Any) -> None:
    """Best-effort release for CUDA matvec workspace buffers."""
    if ws is None:
        return
    release_fn = getattr(ws, "release", None)
    if callable(release_fn):
        try:
            release_fn()
            return
        except Exception:
            pass
    for attr in (
        "_g_buf",
        "_w_block",
        "_w_offdiag",
        "_epq_table",
        "_epq_apply_tile_cache",
        "_csr_tile_cache",
        "_csr_host_tile_cache",
        "_diag_g_cache",
        "_k25_ws",
        "_offdiag_gemm_ws",
        "_gdf_ws",
    ):
        if hasattr(ws, attr):
            try:
                setattr(ws, attr, None)
            except Exception:
                pass


def estimate_matvec_cuda_workspace_bytes(ws: Any) -> int:
    """Best-effort byte estimate for workspace memory accounting."""
    if ws is None:
        return 0
    try:
        est_fn = getattr(ws, "workspace_nbytes_estimate", None)
        if callable(est_fn):
            est = int(est_fn())
            if est >= 0:
                return est
    except Exception:
        pass

    seen: set[int] = set()
    stack: list[Any] = [ws]
    total = 0
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        nbytes = getattr(obj, "nbytes", None)
        if nbytes is not None:
            try:
                total += int(nbytes)
                continue
            except Exception:
                pass
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    return int(max(0, total))


def resolve_matvec_cuda_ws_cache_budget_bytes(
    *,
    cp_mod: Any | None,
    hard_cap_gib: float,
    fraction: float,
) -> int:
    """Resolve workspace-cache budget from VRAM cap/total and configured fraction."""
    frac = _normalize_ws_cache_fraction(fraction)
    if frac <= 0.0:
        return 0
    total_b = 0
    if cp_mod is not None:
        try:
            _free_b, _total_b = cp_mod.cuda.runtime.memGetInfo()
            total_b = int(_total_b)
        except Exception:
            total_b = 0
    cap_b = int(float(hard_cap_gib) * (1024**3)) if float(hard_cap_gib) > 0.0 else 0
    base_b = int(cap_b if cap_b > 0 else total_b)
    if base_b <= 0:
        return 0
    reserve_b = int(1.5 * 1024**3)
    usable_b = max(0, int(base_b) - int(reserve_b))
    return int(max(0, usable_b) * float(frac))


def resolve_kernel_cuda_execution_mode(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    matvec_backend: str,
    strict_gpu: bool,
    ncsf: int,
    matvec_cuda_davidson_subspace_eigh_cpu_in: bool | None,
    matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff: int,
) -> dict[str, Any]:
    if str(matvec_backend) in CUDA_MATVEC_BACKENDS:
        if strict_gpu and matvec_cuda_davidson_subspace_eigh_cpu_in is True:
            raise ValueError("strict_gpu=True is incompatible with matvec_cuda_davidson_subspace_eigh_cpu=True")
        if strict_gpu:
            matvec_cuda_davidson_subspace_eigh_cpu = False
        elif matvec_cuda_davidson_subspace_eigh_cpu_in is None:
            matvec_cuda_davidson_subspace_eigh_cpu = bool(
                int(ncsf) <= int(matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff)
            )
        else:
            matvec_cuda_davidson_subspace_eigh_cpu = bool(matvec_cuda_davidson_subspace_eigh_cpu_in)

        matvec_cuda_make_hdiag_cpu_in = kwargs.pop(
            "matvec_cuda_make_hdiag_cpu",
            getattr(defaults, "matvec_cuda_make_hdiag_cpu", None),
        )
        matvec_cuda_make_hdiag_cpu_ncsf_cutoff = int(
            kwargs.pop(
                "matvec_cuda_make_hdiag_cpu_ncsf_cutoff",
                getattr(defaults, "matvec_cuda_make_hdiag_cpu_ncsf_cutoff", 10_000),
            )
        )
        if matvec_cuda_make_hdiag_cpu_ncsf_cutoff < 0:
            matvec_cuda_make_hdiag_cpu_ncsf_cutoff = 0
        if strict_gpu and matvec_cuda_make_hdiag_cpu_in is True:
            raise ValueError("strict_gpu=True is incompatible with matvec_cuda_make_hdiag_cpu=True")
        if strict_gpu:
            matvec_cuda_make_hdiag_cpu = False
        elif matvec_cuda_make_hdiag_cpu_in is None:
            matvec_cuda_make_hdiag_cpu = bool(int(ncsf) <= int(matvec_cuda_make_hdiag_cpu_ncsf_cutoff))
        else:
            matvec_cuda_make_hdiag_cpu = bool(matvec_cuda_make_hdiag_cpu_in)

        matvec_cuda_dtype_in = str(
            kwargs.pop("matvec_cuda_dtype", getattr(defaults, "matvec_cuda_dtype", "float64"))
        ).strip().lower()
        if matvec_cuda_dtype_in in ("float64", "fp64", "f64", "double"):
            matvec_cuda_dtype = "float64"
        elif matvec_cuda_dtype_in in ("float32", "fp32", "f32", "single"):
            matvec_cuda_dtype = "float32"
        elif matvec_cuda_dtype_in in ("mixed", "mixed_fp32", "float32_mixed"):
            matvec_cuda_dtype = "mixed"
        else:
            raise ValueError("matvec_cuda_dtype must be one of: float64, float32, mixed")

        mixed_thr_in = kwargs.pop(
            "matvec_cuda_mixed_threshold",
            getattr(defaults, "matvec_cuda_mixed_threshold", 1e-5),
        )
        mixed_force_final_full_in = kwargs.pop(
            "matvec_cuda_mixed_force_final_full_hop",
            getattr(defaults, "matvec_cuda_mixed_force_final_full_hop", True),
        )
        mixed_final_full_subspace_refresh_in = kwargs.pop(
            "matvec_cuda_mixed_final_full_subspace_refresh",
            getattr(defaults, "matvec_cuda_mixed_final_full_subspace_refresh", False),
        )
        mixed_low_max_iter_in = kwargs.pop(
            "matvec_cuda_mixed_low_precision_max_iter",
            getattr(defaults, "matvec_cuda_mixed_low_precision_max_iter", 2),
        )
        matvec_cuda_mixed_threshold = None if mixed_thr_in is None else float(mixed_thr_in)
        matvec_cuda_mixed_force_final_full_hop = bool(mixed_force_final_full_in)
        matvec_cuda_mixed_final_full_subspace_refresh = bool(mixed_final_full_subspace_refresh_in)
        if mixed_low_max_iter_in is None:
            matvec_cuda_mixed_low_precision_max_iter = None
        else:
            mixed_low_max_iter_i = int(mixed_low_max_iter_in)
            matvec_cuda_mixed_low_precision_max_iter = (
                None if mixed_low_max_iter_i <= 0 else int(mixed_low_max_iter_i)
            )
        if matvec_cuda_dtype == "mixed":
            if matvec_cuda_mixed_threshold is None:
                matvec_cuda_mixed_threshold = 1e-5
            if float(matvec_cuda_mixed_threshold) <= 0.0:
                raise ValueError("matvec_cuda_mixed_threshold must be > 0 for matvec_cuda_dtype='mixed'")
    else:
        matvec_cuda_davidson_subspace_eigh_cpu = None
        matvec_cuda_make_hdiag_cpu = None
        matvec_cuda_make_hdiag_cpu_ncsf_cutoff = None
        matvec_cuda_dtype = "float64"
        matvec_cuda_mixed_threshold = None
        matvec_cuda_mixed_force_final_full_hop = False
        matvec_cuda_mixed_final_full_subspace_refresh = False
        matvec_cuda_mixed_low_precision_max_iter = None

    return {
        "matvec_cuda_davidson_subspace_eigh_cpu": matvec_cuda_davidson_subspace_eigh_cpu,
        "matvec_cuda_make_hdiag_cpu": matvec_cuda_make_hdiag_cpu,
        "matvec_cuda_make_hdiag_cpu_ncsf_cutoff": matvec_cuda_make_hdiag_cpu_ncsf_cutoff,
        "matvec_cuda_dtype": matvec_cuda_dtype,
        "matvec_cuda_mixed_threshold": matvec_cuda_mixed_threshold,
        "matvec_cuda_mixed_force_final_full_hop": matvec_cuda_mixed_force_final_full_hop,
        "matvec_cuda_mixed_final_full_subspace_refresh": matvec_cuda_mixed_final_full_subspace_refresh,
        "matvec_cuda_mixed_low_precision_max_iter": matvec_cuda_mixed_low_precision_max_iter,
    }


def resolve_approx_cuda_frontend(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    matvec_backend: str,
) -> dict[str, Any]:
    if str(matvec_backend) in ("cuda_eri_mat", "cuda"):
        aggregate_preview_in = kwargs.get(
            "matvec_cuda_aggregate_offdiag",
            getattr(defaults, "matvec_cuda_aggregate_offdiag", None),
        )
        if aggregate_preview_in is None:
            aggregate_preview = True
        else:
            aggregate_preview = bool(aggregate_preview_in)
        enforce_cuda_aggregate_offdiag_guard(
            bool(aggregate_preview),
            context="approx_kernel(cuda)",
        )
    else:
        aggregate_preview = None

    approx_cuda_dtype_in = str(kwargs.pop("approx_cuda_dtype", getattr(defaults, "approx_cuda_dtype", "float32"))).strip().lower()
    if approx_cuda_dtype_in in ("float64", "fp64", "f64", "double"):
        approx_cuda_dtype = "float64"
    elif approx_cuda_dtype_in in ("float32", "fp32", "f32", "single"):
        approx_cuda_dtype = "float32"
    else:
        raise ValueError("approx_cuda_dtype must be one of: float32, float64")

    return {
        "approx_cuda_dtype": approx_cuda_dtype,
        "matvec_cuda_aggregate_offdiag_preview": aggregate_preview,
    }


def resolve_approx_kernel_iteration_caps(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    nroots: int,
    matvec_backend: str,
) -> dict[str, int]:
    nroots_i = int(nroots)
    if str(matvec_backend) in ("cuda_eri_mat", "cuda"):
        default_cap_cycle = 2
        default_cap_space = max(4, 2 * nroots_i)
    else:
        default_cap_cycle = 2
        default_cap_space = max(4, 2 * nroots_i)

    cap_cycle_attr = getattr(defaults, "approx_kernel_max_cycle", None)
    cap_space_attr = getattr(defaults, "approx_kernel_max_space", None)
    cap_cycle = int(default_cap_cycle if cap_cycle_attr is None else cap_cycle_attr)
    cap_space = int(default_cap_space if cap_space_attr is None else cap_space_attr)

    max_cycle = int(kwargs.pop("max_cycle", cap_cycle))
    max_space = int(kwargs.pop("max_space", cap_space))
    if cap_cycle > 0:
        max_cycle = min(max_cycle, cap_cycle)
    if cap_space > 0:
        max_space = min(max_space, cap_space)

    return {
        "nroots": int(nroots_i),
        "max_cycle": int(max_cycle),
        "max_space": int(max_space),
    }
