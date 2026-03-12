from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_policy import (
    cap_cuda_max_g_mib_by_hard_cap,
    cuda_budget_free_bytes,
    enforce_cuda_fp32_large_cas_epq_policy,
    maybe_promote_cuda_apply_mode_scatter,
)
from .matvec_runtime import (
    resolve_cuda_cache_csr_tiles,
    resolve_cuda_threads_apply,
    resolve_cuda_threads_w,
    tune_cuda_threads_for_large_cas_noepq,
)


def auto_select_use_epq_table(
    *,
    cp: Any,
    norb: int,
    ncsf: int,
    aggregate_offdiag: bool,
    has_epq_table_device_build: bool,
    mem_hard_cap_gib: float,
    dtype_mode: str,
    eri_mat_present: bool,
) -> bool:
    """Auto-select EPQ-table usage for CUDA matvec when user did not force it."""

    norb_i = int(norb)
    ncsf_i = int(ncsf)
    if norb_i <= 16 and ncsf_i <= 300_000:
        return True

    if not (norb_i <= 16 and bool(aggregate_offdiag) and bool(has_epq_table_device_build)):
        return False

    n_pairs = int(norb_i) * (int(norb_i) - 1)
    ntasks = int(ncsf_i) * int(n_pairs)
    est_counts_bytes = int(ntasks) * 4
    est_table_bytes = (int(ncsf_i) + 1) * 8 + int(ntasks) * 16
    est_peak_bytes = int(est_counts_bytes) + int(est_table_bytes)
    try:
        free_bytes = cuda_budget_free_bytes(cp, float(mem_hard_cap_gib))
    except Exception:
        free_bytes = 0
    if not free_bytes or int(est_peak_bytes) > int(float(free_bytes) * 0.80):
        return False

    # Guard: if EPQ materialization would crowd out the full W buffer, prefer no-EPQ aggregate path.
    if bool(aggregate_offdiag) and int(free_bytes) > 0:
        nops_i = int(norb_i) * int(norb_i)
        dtype_bytes = 4 if str(dtype_mode) == "float32" else 8
        w_bytes = int(ncsf_i) * int(nops_i) * int(dtype_bytes) if bool(eri_mat_present) else 0
        if (int(est_table_bytes) + int(w_bytes)) > int(float(free_bytes) * 0.85):
            return False
    return True


def resolve_epq_streaming_controls(
    *,
    epq_streaming_in: Any,
    epq_stream_j_tile_in: Any,
    epq_stream_use_recompute_in: Any,
) -> dict[str, Any]:
    """Normalize EPQ streaming controls shared by kernel CUDA orchestration paths."""

    if isinstance(epq_streaming_in, str):
        stream_mode = str(epq_streaming_in).strip().lower()
        if stream_mode in ("", "auto"):
            streaming_mode = "auto"
            streaming = False
        elif stream_mode in ("1", "true", "yes", "on"):
            streaming_mode = "on"
            streaming = True
        elif stream_mode in ("0", "false", "no", "off"):
            streaming_mode = "off"
            streaming = False
        else:
            raise ValueError("matvec_cuda_epq_streaming must be bool or one of: auto/on/off")
    elif epq_streaming_in is None:
        streaming_mode = "auto"
        streaming = False
    else:
        streaming_mode = "manual"
        streaming = bool(epq_streaming_in)

    stream_j_tile = int(epq_stream_j_tile_in)
    if stream_j_tile < 0:
        stream_j_tile = 0

    if isinstance(epq_stream_use_recompute_in, str):
        recompute_mode = str(epq_stream_use_recompute_in).strip().lower()
        if recompute_mode in ("", "auto"):
            stream_use_recompute: bool | str = "auto"
        elif recompute_mode in ("1", "true", "yes", "on"):
            stream_use_recompute = True
        elif recompute_mode in ("0", "false", "no", "off"):
            stream_use_recompute = False
        else:
            raise ValueError(
                "matvec_cuda_epq_stream_use_recompute must be bool or one of: auto/on/off"
            )
    elif epq_stream_use_recompute_in is None:
        stream_use_recompute = "auto"
    else:
        stream_use_recompute = bool(epq_stream_use_recompute_in)

    return {
        "streaming_mode": str(streaming_mode),
        "streaming": bool(streaming),
        "stream_j_tile": int(stream_j_tile),
        "stream_use_recompute": stream_use_recompute,
    }


def resolve_cuda_workspace_policy_common(
    *,
    apply_mode: str,
    apply_mode_forced: bool,
    use_epq_table: bool,
    dtype_mode: str,
    ncsf: int,
    nops: int,
    threads_enum: int,
    threads_g: int,
    threads_w: int,
    threads_apply: int,
    threads_enum_forced: bool,
    threads_g_forced: bool,
    threads_apply_auto: bool,
    eri_mat_present: bool,
    aggregate_offdiag: bool,
    max_g_mib: float,
    mem_hard_cap_gib: float,
    cache_csr_tiles_in: Any,
    j_tile: int,
    norb: int,
    csr_capacity_mult: float,
    noepq_large_ncsf_uses_64: bool,
) -> dict[str, Any]:
    """Resolve shared CUDA apply/thread/cache policy knobs used by kernel and approx."""

    apply_mode_out = maybe_promote_cuda_apply_mode_scatter(
        apply_mode=str(apply_mode),
        apply_mode_forced=bool(apply_mode_forced),
        use_epq_table=bool(use_epq_table),
        dtype_mode=str(dtype_mode),
        ncsf=int(ncsf),
    )
    thread_tuned = tune_cuda_threads_for_large_cas_noepq(
        threads_enum=int(threads_enum),
        threads_g=int(threads_g),
        threads_enum_forced=bool(threads_enum_forced),
        threads_g_forced=bool(threads_g_forced),
        eri_mat_present=bool(eri_mat_present),
        use_epq_table=bool(use_epq_table),
        aggregate_offdiag=bool(aggregate_offdiag),
        nops=int(nops),
        ncsf=int(ncsf),
        dtype_mode=str(dtype_mode),
    )
    threads_enum_out = int(thread_tuned["threads_enum"])
    threads_g_out = int(thread_tuned["threads_g"])
    threads_w_out = resolve_cuda_threads_w(
        threads_w=int(threads_w),
        threads_g=int(threads_g_out),
    )
    threads_apply_out = resolve_cuda_threads_apply(
        threads_apply=int(threads_apply),
        use_epq_table=bool(use_epq_table),
        dtype_mode=str(dtype_mode),
        ncsf=int(ncsf),
        nops=int(nops),
        noepq_large_ncsf_uses_64=bool(noepq_large_ncsf_uses_64),
    )
    max_g_out = cap_cuda_max_g_mib_by_hard_cap(
        max_g_mib=float(max_g_mib),
        hard_cap_gib=float(mem_hard_cap_gib),
    )
    cache_csr_tiles_out = resolve_cuda_cache_csr_tiles(
        cache_csr_tiles_in=cache_csr_tiles_in,
        aggregate_offdiag=bool(aggregate_offdiag),
        use_epq_table=bool(use_epq_table),
        ncsf=int(ncsf),
        j_tile=int(j_tile),
        norb=int(norb),
        csr_capacity_mult=float(csr_capacity_mult),
    )
    if bool(threads_apply_auto) and int(threads_apply_out) <= 0:
        threads_apply_out = 64 if int(ncsf) >= 1_000_000 else 32
    return {
        "apply_mode": str(apply_mode_out),
        "threads_enum": int(threads_enum_out),
        "threads_g": int(threads_g_out),
        "threads_w": int(threads_w_out),
        "threads_apply": int(threads_apply_out),
        "max_g_mib": float(max_g_out),
        "cache_csr_tiles": cache_csr_tiles_out,
    }


def validate_low_precision_cuda_path(
    *,
    context: str,
    dtype_mode: str,
    use_epq_table: bool,
    aggregate_offdiag: bool,
    ncsf: int,
    eri_mat_present: bool,
    enable_fp64_emulation: bool,
    use_graph: bool,
) -> dict[str, Any]:
    """Validate low-precision CUDA constraints and return possibly adjusted controls."""

    dtype_mode_s = str(dtype_mode)
    use_graph_out = bool(use_graph)
    if dtype_mode_s in ("float32", "mixed"):
        if dtype_mode_s == "mixed":
            if not bool(use_epq_table):
                raise ValueError("matvec_cuda_dtype float32/mixed requires matvec_cuda_use_epq_table=True")
            if not bool(aggregate_offdiag):
                raise ValueError("matvec_cuda_dtype float32/mixed requires matvec_cuda_aggregate_offdiag=True")
        elif (not bool(use_epq_table)) and (not bool(eri_mat_present)):
            raise ValueError(
                "matvec_cuda_dtype='float32' with matvec_cuda_use_epq_table=False requires dense eri_mat"
            )
        if bool(enable_fp64_emulation):
            raise ValueError(
                "matvec_cuda_enable_fp64_emulation is incompatible with matvec_cuda_dtype float32/mixed"
            )
        if bool(use_graph_out):
            use_graph_out = False
    enforce_cuda_fp32_large_cas_epq_policy(
        context=str(context),
        matvec_cuda_dtype=str(dtype_mode_s),
        matvec_cuda_use_epq_table=bool(use_epq_table),
        matvec_cuda_aggregate_offdiag=bool(aggregate_offdiag),
        ncsf=int(ncsf),
    )
    return {
        "use_graph": bool(use_graph_out),
    }


def apply_low_precision_and_workspace_policy(
    *,
    context: str,
    dtype_mode: str,
    use_epq_table: bool,
    aggregate_offdiag: bool,
    ncsf: int,
    eri_mat_present: bool,
    enable_fp64_emulation: bool,
    use_graph: bool,
    apply_mode: str,
    apply_mode_forced: bool,
    nops: int,
    threads_enum: int,
    threads_g: int,
    threads_w: int,
    threads_apply: int,
    threads_enum_forced: bool,
    threads_g_forced: bool,
    threads_apply_auto: bool,
    max_g_mib: float,
    mem_hard_cap_gib: float,
    cache_csr_tiles_in: Any,
    j_tile: int,
    norb: int,
    csr_capacity_mult: float,
    noepq_large_ncsf_uses_64: bool,
) -> dict[str, Any]:
    """Apply low-precision guards + shared workspace policy and return normalized CUDA knobs."""

    lowp = validate_low_precision_cuda_path(
        context=str(context),
        dtype_mode=str(dtype_mode),
        use_epq_table=bool(use_epq_table),
        aggregate_offdiag=bool(aggregate_offdiag),
        ncsf=int(ncsf),
        eri_mat_present=bool(eri_mat_present),
        enable_fp64_emulation=bool(enable_fp64_emulation),
        use_graph=bool(use_graph),
    )
    use_graph_out = bool(lowp["use_graph"])
    policy = resolve_cuda_workspace_policy_common(
        apply_mode=str(apply_mode),
        apply_mode_forced=bool(apply_mode_forced),
        use_epq_table=bool(use_epq_table),
        dtype_mode=str(dtype_mode),
        ncsf=int(ncsf),
        nops=int(nops),
        threads_enum=int(threads_enum),
        threads_g=int(threads_g),
        threads_w=int(threads_w),
        threads_apply=int(threads_apply),
        threads_enum_forced=bool(threads_enum_forced),
        threads_g_forced=bool(threads_g_forced),
        threads_apply_auto=bool(threads_apply_auto),
        eri_mat_present=bool(eri_mat_present),
        aggregate_offdiag=bool(aggregate_offdiag),
        max_g_mib=float(max_g_mib),
        mem_hard_cap_gib=float(mem_hard_cap_gib),
        cache_csr_tiles_in=cache_csr_tiles_in,
        j_tile=int(j_tile),
        norb=int(norb),
        csr_capacity_mult=float(csr_capacity_mult),
        noepq_large_ncsf_uses_64=bool(noepq_large_ncsf_uses_64),
    )
    return {
        "graph_disabled": bool(use_graph) and (not bool(use_graph_out)),
        "use_graph": bool(use_graph_out),
        "apply_mode": str(policy["apply_mode"]),
        "threads_enum": int(policy["threads_enum"]),
        "threads_g": int(policy["threads_g"]),
        "threads_w": int(policy["threads_w"]),
        "threads_apply": int(policy["threads_apply"]),
        "max_g_mib": float(policy["max_g_mib"]),
        "cache_csr_tiles": policy["cache_csr_tiles"],
    }


def reset_cuda_graph_capture_buffers(*, ws: Any) -> None:
    """Drop cached CUDA Graph capture handles on a matvec workspace."""

    setattr(ws, "_cuda_graph", None)
    setattr(ws, "_cuda_graph_x", None)
    setattr(ws, "_cuda_graph_y", None)


def refresh_cuda_workspace_hamiltonian_inplace(
    *,
    cp: Any,
    ws: Any,
    eri_mat_d: Any,
    l_full_d: Any,
    direct_op_d: Any = None,
    h_eff_d: Any,
    use_cuda_graph: bool,
    refresh_diag_cache_for_graph: bool,
) -> dict[str, bool]:
    """Refresh a cached CUDA matvec workspace for a new Hamiltonian.

    This keeps pointer-stable buffers where possible for CUDA Graph reuse and
    invalidates graph capture handles only when buffer replacement is required.
    """

    graph_invalidated = False
    h_eff_replaced = False
    diag_cache_refreshed = False

    ws_dtype_obj = np.dtype(getattr(ws, "dtype", np.float64))
    ws.eri_mat = None if eri_mat_d is None else cp.ascontiguousarray(cp.asarray(eri_mat_d, dtype=ws_dtype_obj))
    ws.l_full = None if l_full_d is None else cp.ascontiguousarray(cp.asarray(l_full_d, dtype=ws_dtype_obj))
    ws.direct_op = direct_op_d

    h_eff_flat_new = ws._as_h_eff_flat(h_eff_d)
    if getattr(ws, "h_eff_flat", None) is None or tuple(getattr(ws.h_eff_flat, "shape", ())) != tuple(
        getattr(h_eff_flat_new, "shape", ())
    ):
        ws.h_eff_flat = h_eff_flat_new
        h_eff_replaced = True
        if getattr(ws, "_cuda_graph", None) is not None:
            reset_cuda_graph_capture_buffers(ws=ws)
            graph_invalidated = True
    else:
        cp.copyto(ws.h_eff_flat, h_eff_flat_new)

    # Diagonal-rs contribution depends on `eri_diag_t` extracted from `eri_mat`.
    ws._eri_diag_t = None

    if eri_mat_d is not None:
        if getattr(ws, "_eri_mat_t", None) is not None:
            cp.copyto(ws._eri_mat_t, ws.eri_mat.T)
        elif getattr(ws, "_cuda_graph", None) is not None:
            reset_cuda_graph_capture_buffers(ws=ws)
            graph_invalidated = True
    else:
        ws._eri_mat_t = None
        if getattr(ws, "_cuda_graph", None) is not None:
            reset_cuda_graph_capture_buffers(ws=ws)
            graph_invalidated = True

    if (
        bool(refresh_diag_cache_for_graph)
        and bool(use_cuda_graph)
        and getattr(ws, "_cuda_graph", None) is not None
        and bool(getattr(ws, "include_diagonal_rs", False))
    ):
        ws._build_diag_g_cache()
        diag_cache_refreshed = True

    ws.use_cuda_graph = bool(use_cuda_graph)
    return {
        "graph_invalidated": bool(graph_invalidated),
        "h_eff_replaced": bool(h_eff_replaced),
        "diag_cache_refreshed": bool(diag_cache_refreshed),
    }
