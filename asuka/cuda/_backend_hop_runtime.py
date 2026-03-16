from __future__ import annotations

import numpy as np
import time
from typing import Any


def resolve_hop_runtime_flags(
    workspace: Any,
    *,
    path_mode: str,
    eri_mat: Any,
) -> dict[str, Any]:
    fused_eligible_runtime = bool(
        bool(getattr(workspace, "use_fused_hop", True))
        and bool(getattr(workspace, "_fused_hop_kernel_available", False))
        and int(getattr(workspace, "norb", 0)) <= 20
        and (eri_mat is not None or getattr(workspace, "eri_mat", None) is not None or getattr(workspace, "l_full", None) is not None)
    )
    use_fused_hop = bool(str(path_mode) in ("fused_coo", "fused_epq_hybrid") and fused_eligible_runtime)
    use_aggregate_offdiag = bool(getattr(workspace, "aggregate_offdiag_k", False)) and (not use_fused_hop)
    return {
        "use_fused_hop": bool(use_fused_hop),
        "use_aggregate_offdiag": bool(use_aggregate_offdiag),
    }


def normalize_hop_x(workspace: Any, *, cp: Any, x: Any):
    out = cp.ascontiguousarray(cp.asarray(x, dtype=workspace._dtype).ravel())
    if out.shape != (int(workspace.ncsf),):
        raise ValueError("x must have shape (ncsf,)")
    return out


def try_cuda_graph_fast_path(
    workspace: Any,
    *,
    cp: Any,
    x: Any,
    y: Any,
    eri_mat: Any,
    h_eff: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
) -> dict[str, Any]:
    if not (
        bool(getattr(workspace, "use_cuda_graph", False))
        and getattr(workspace, "_cuda_graph", None) is not None
        and getattr(workspace, "_cuda_graph_x", None) is not None
        and getattr(workspace, "_cuda_graph_y", None) is not None
        and eri_mat is None
        and h_eff is None
        and profile is None
        and not bool(check_overflow)
    ):
        return {"executed": False, "y": y, "stream": stream}

    stream_out = stream
    if stream_out is None:
        stream_out = cp.cuda.get_current_stream()
    with stream_out:
        cp.copyto(workspace._cuda_graph_x, x)
        workspace._cuda_graph.launch(stream=stream_out)
        if y is None:
            y_out = workspace._cuda_graph_y.copy()
        else:
            y_out = cp.ascontiguousarray(cp.asarray(y, dtype=workspace._dtype).ravel())
            if y_out.shape != (int(workspace.ncsf),):
                raise ValueError("y must have shape (ncsf,)")
            cp.copyto(y_out, workspace._cuda_graph_y)
    if bool(sync):
        stream_out.synchronize()
    return {"executed": True, "y": y_out, "stream": stream_out}


def prepare_hop_runtime_inputs(
    workspace: Any,
    *,
    cp: Any,
    y: Any,
    eri_mat: Any,
    direct_op: Any = None,
    h_eff: Any,
    use_fused_hop: bool,
) -> dict[str, Any]:
    if y is None:
        y_out = cp.empty((int(workspace.ncsf),), dtype=workspace._dtype)
    else:
        y_out = cp.ascontiguousarray(cp.asarray(y, dtype=workspace._dtype).ravel())
        if y_out.shape != (int(workspace.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    eri_mat_use = workspace.eri_mat if eri_mat is None else cp.ascontiguousarray(cp.asarray(eri_mat, dtype=workspace._dtype))
    l_full_use = None
    direct_op_use = getattr(workspace, "direct_op", None) if direct_op is None else direct_op
    use_df = False
    use_direct = False
    if eri_mat_use is not None:
        if eri_mat_use.shape != (int(workspace.nops), int(workspace.nops)):
            raise ValueError("eri_mat must have shape (nops,nops)")
    else:
        l_full_use = workspace.l_full
        if l_full_use is not None:
            l_full_use = cp.ascontiguousarray(cp.asarray(l_full_use, dtype=workspace._dtype))
            if l_full_use.ndim != 2 or tuple(l_full_use.shape)[0] != int(workspace.nops):
                raise ValueError("l_full must have shape (norb*norb, naux)")
            use_df = True
        elif direct_op_use is None:
            raise ValueError("eri_mat, l_full, or direct_op must be provided (workspace has none)")
        else:
            use_direct = True

    if bool(use_fused_hop) and bool(use_df) and eri_mat_use is None:
        eri_mat_use = cp.ascontiguousarray(cp.dot(l_full_use, l_full_use.T).astype(workspace._dtype))

    use_epq_streaming = bool(
        bool(getattr(workspace, "use_epq_table", False))
        and bool(getattr(workspace, "epq_streaming", False))
        and getattr(workspace, "_epq_table", None) is None
    )
    epq_stream_panic_requested = False
    epq_stream_panic_active = False
    if use_epq_streaming:
        panic_mode = str(getattr(workspace, "epq_stream_panic_mode", "off")).strip().lower()
        if panic_mode == "on":
            epq_stream_panic_requested = True
            epq_stream_panic_active = True
    use_epq_streaming_tiles = bool(use_epq_streaming and (not epq_stream_panic_active))

    if h_eff is None:
        h_eff_flat = workspace.h_eff_flat
    else:
        h_eff_flat = workspace._as_h_eff_flat(h_eff)
    if h_eff_flat is None:
        raise ValueError("h_eff must be provided (workspace h_eff is None)")

    return {
        "y": y_out,
        "eri_mat_use": eri_mat_use,
        "l_full_use": l_full_use,
        "direct_op_use": direct_op_use,
        "use_df": bool(use_df),
        "use_direct": bool(use_direct),
        "h_eff_flat": h_eff_flat,
        "use_epq_streaming": bool(use_epq_streaming),
        "use_epq_streaming_tiles": bool(use_epq_streaming_tiles),
        "epq_stream_panic_requested": bool(epq_stream_panic_requested),
        "epq_stream_panic_active": bool(epq_stream_panic_active),
    }


def run_fused_hop_path(
    workspace: Any,
    *,
    cp: Any,
    ext: Any,
    x: Any,
    y: Any,
    eri_mat_use: Any,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    path_mode: str,
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
):
    """Run the fused-hop / fused-hybrid controller loop.

    Extracted from GugaMatvecEriMatWorkspace.hop() to keep the workspace
    class focused on low-level tile kernels while the orchestration for
    fused Phase-1, COO hybrid, and EPQ-hybrid execution lives here.
    """

    t_total0 = time.perf_counter() if profile is not None else None

    # Fused kernel computes both one-body and two-body; start from zeroed output.
    t0 = time.perf_counter() if profile is not None else None
    cp.cuda.runtime.memsetAsync(
        int(y.data.ptr),
        0,
        int(y.size) * int(y.itemsize),
        int(stream.ptr),
    )
    if profile is not None and t0 is not None:
        stream.synchronize()
        profile["one_body_s"] = profile.get("one_body_s", 0.0) + (time.perf_counter() - t0)

    # Detect available phase-1 paths.
    panic_mode_str = str(getattr(workspace, "epq_stream_panic_mode", "off")).strip().lower()
    has_phase1_coo = bool(
        ext is not None
        and hasattr(ext, "fused_hop_phase1_coo_device")
        and hasattr(ext, "coo_scatter_device")
    )
    if path_mode == "fused_epq_hybrid":
        has_phase1_coo = False

    has_phase1 = bool(
        ext is not None
        and hasattr(ext, "fused_hop_phase1_device")
        and panic_mode_str != "on"
        and workspace._epq_table is not None
    ) and (not has_phase1_coo)
    if path_mode == "fused_coo":
        has_phase1 = False

    g_tile = workspace._g_buf if (has_phase1_coo or has_phase1) else None
    set_matvec_path_profile(
        profile=profile,
        path_mode=path_mode,
        effective_mode=(
            "fused_coo" if has_phase1_coo else ("fused_epq_hybrid" if has_phase1 else "fused_hop_fallback")
        ),
        fallback_reason=(
            ""
            if (has_phase1_coo or has_phase1)
            else str(getattr(workspace, "path_mode_fallback_reason", "fused_phase1_unavailable"))
        ),
    )

    # Allocate COO buffers once (lazy) for the COO path.
    if has_phase1_coo and g_tile is not None:
        n_offdiag = int(workspace.norb) * (int(workspace.norb) - 1)
        avg_conn = 20  # empirical average for CAS14
        max_coo = int(workspace.j_tile) * n_offdiag * avg_conn
        workspace._ensure_coo_buffers(max_coo)

    coo_overflow = False
    for j0 in range(0, int(workspace.ncsf), int(workspace.j_tile)):
        j1 = min(int(workspace.ncsf), j0 + int(workspace.j_tile))
        j_count = j1 - j0
        check_overflow_tile = bool(check_overflow)
        if check_overflow_tile and bool(workspace.check_overflow_first_tile_only) and j0 != 0:
            check_overflow_tile = False

        if has_phase1_coo and g_tile is not None:
            # COO hybrid path: Phase 1 DFS + ERI contraction + COO output,
            # then trivial COO scatter for Phase 2.
            g_tile_slice = g_tile[:j_count]
            nnz = workspace._fused_hop_phase1_coo_tile(
                j_start=j0,
                j_count=j_count,
                x=x,
                eri_mat=eri_mat_use,
                h_eff_flat=h_eff_flat,
                y=y,
                g_out=g_tile_slice,
                stream=stream,
                sync=False,
                check_overflow=check_overflow_tile,
                profile=profile,
            )
            # Check for COO overflow — fall back to original fused kernel
            if nnz > workspace._coo_max:
                import warnings

                warnings.warn(
                    f"COO overflow: nnz={nnz} > max_coo={workspace._coo_max}, "
                    f"falling back to fused kernel. Growing buffer 2x.",
                    stacklevel=2,
                )
                workspace._ensure_coo_buffers(workspace._coo_max * 2)
                coo_overflow = True
                break
            # Track max nnz across tiles for adaptive calibration
            if not getattr(workspace, "_coo_calibrated", False):
                cal_max_nnz = max(getattr(workspace, "_coo_cal_max_nnz", 0), nnz)
                workspace._coo_cal_max_nnz = cal_max_nnz
            # Phase 2: COO scatter
            workspace._coo_scatter_tile(
                g_tile=g_tile_slice,
                nops=int(workspace.nops),
                nnz=nnz,
                y=y,
                stream=stream,
                sync=False,
                profile=profile,
            )
        elif has_phase1 and g_tile is not None:
            # Hybrid path: Phase 1 DFS + ERI contraction → g_tile,
            # then EPQ table scatter for Phase 2.
            g_tile_slice = g_tile[:j_count]
            workspace._fused_hop_phase1_tile(
                j_start=j0,
                j_count=j_count,
                x=x,
                eri_mat=eri_mat_use,
                h_eff_flat=h_eff_flat,
                y=y,
                g_out=g_tile_slice,
                stream=stream,
                sync=False,
                check_overflow=check_overflow_tile,
                profile=profile,
            )
            # Build EPQ tile for this j-range
            t_build0 = time.perf_counter() if profile is not None else None
            epq_tile = build_epq_action_table_tile_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                j_start=j0,
                j_count=j_count,
                threads=int(workspace.threads_enum),
                stream=stream,
                sync=True,
                check_overflow=check_overflow_tile,
                use_recompute=workspace.epq_stream_use_recompute,
                recompute_warp_coop=bool(workspace.epq_recompute_warp_coop),
                global_indptr=False,
                pq_block=0,
                dtype=workspace._dtype,
            )
            if profile is not None and t_build0 is not None:
                dt = time.perf_counter() - t_build0
                profile["fused_epq_build_s"] = profile.get("fused_epq_build_s", 0.0) + dt
            # Phase 2: scatter G via EPQ table
            t_scatter0 = time.perf_counter() if profile is not None else None
            local_indptr, indices, pq_ids, epq_data = epq_tile
            apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                local_indptr,
                indices,
                pq_ids,
                epq_data,
                g_tile_slice,
                j_start=j0,
                j_count=j_count,
                y=y,
                zero_y=False,
                stream=stream,
                sync=False,
                check_overflow=False,
                dtype=workspace._dtype,
            )
            if profile is not None and t_scatter0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t_scatter0
                profile["fused_epq_scatter_s"] = profile.get("fused_epq_scatter_s", 0.0) + dt
        else:
            # Fallback: original fused kernel with Phase 2 DFS
            workspace._fused_hop_tile(
                j_start=j0,
                j_count=j_count,
                x=x,
                eri_mat=eri_mat_use,
                h_eff_flat=h_eff_flat,
                y=y,
                stream=stream,
                sync=bool(sync),
                check_overflow=check_overflow_tile,
                profile=profile,
            )

    # Adaptive COO buffer sizing: after the first successful hop() completes
    # all tiles, use the observed max nnz to right-size the buffer.
    if (
        not coo_overflow
        and not getattr(workspace, "_coo_calibrated", False)
        and hasattr(workspace, "_coo_cal_max_nnz")
    ):
        cal_nnz = workspace._coo_cal_max_nnz
        new_max = max(int(cal_nnz * 2.0), 1024)
        if new_max < workspace._coo_max:
            workspace._ensure_coo_buffers(new_max, force=True)
        workspace._coo_calibrated = True
        del workspace._coo_cal_max_nnz

    # If COO overflow occurred, re-zero y and redo with original fused kernel.
    if coo_overflow:
        y.fill(0)
        for j0 in range(0, int(workspace.ncsf), int(workspace.j_tile)):
            j1 = min(int(workspace.ncsf), j0 + int(workspace.j_tile))
            j_count = j1 - j0
            check_overflow_tile = bool(check_overflow)
            if check_overflow_tile and bool(workspace.check_overflow_first_tile_only) and j0 != 0:
                check_overflow_tile = False
            workspace._fused_hop_tile(
                j_start=j0,
                j_count=j_count,
                x=x,
                eri_mat=eri_mat_use,
                h_eff_flat=h_eff_flat,
                y=y,
                stream=stream,
                sync=bool(sync),
                check_overflow=check_overflow_tile,
                profile=profile,
            )

    if profile is not None and t_total0 is not None:
        stream.synchronize()
        profile["total_s"] = profile.get("total_s", 0.0) + (time.perf_counter() - t_total0)

    return y


def build_epq_stream_tile(
    workspace: Any,
    *,
    build_epq_action_table_tile_device_fn: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    j_start: int,
    j_count: int,
    global_indptr: bool = False,
    stream_override: Any = None,
    sync_override: bool | None = None,
    check_overflow_override: bool | None = None,
):
    stream_build = stream if stream_override is None else stream_override
    sync_build = bool(sync) if sync_override is None else bool(sync_override)
    check_overflow_build = bool(check_overflow) if check_overflow_override is None else bool(check_overflow_override)
    pq_block_build = int(getattr(workspace, "epq_stream_pq_block", 0))
    if pq_block_build > 0:
        sync_build = True
        use_recompute_build = False
    else:
        use_recompute_build = workspace.epq_stream_use_recompute

    epq_tile = build_epq_action_table_tile_device_fn(
        workspace.drt,
        workspace.drt_dev,
        workspace.state_dev,
        j_start=int(j_start),
        j_count=int(j_count),
        threads=int(workspace.threads_enum),
        stream=stream_build,
        sync=bool(sync_build),
        check_overflow=bool(check_overflow_build),
        use_recompute=use_recompute_build,
        recompute_warp_coop=bool(workspace.epq_recompute_warp_coop),
        global_indptr=bool(global_indptr),
        pq_block=int(pq_block_build),
        dtype=workspace._dtype,
    )
    return epq_tile


def build_epq_stream_tile_profiled(
    workspace: Any,
    *,
    build_epq_action_table_tile_device_fn: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    j_start: int,
    j_count: int,
    global_indptr: bool = False,
    stream_override: Any = None,
    sync_override: bool | None = None,
    check_overflow_override: bool | None = None,
):
    t_build0 = time.perf_counter() if profile is not None else None
    stream_build = stream if stream_override is None else stream_override
    epq_tile = build_epq_stream_tile(
        workspace,
        build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        j_start=int(j_start),
        j_count=int(j_count),
        global_indptr=bool(global_indptr),
        stream_override=stream_override,
        sync_override=sync_override,
        check_overflow_override=check_overflow_override,
    )
    if profile is not None and t_build0 is not None:
        stream_build.synchronize()
        dt = time.perf_counter() - t_build0
        profile["epq_stream_build_s"] = profile.get("epq_stream_build_s", 0.0) + dt
        profile["tile_apply_build_s"] = profile.get("tile_apply_build_s", 0.0) + dt
    return epq_tile


def set_matvec_path_profile(
    *,
    profile: dict[str, float] | None,
    path_mode: str,
    effective_mode: str,
    fallback_reason: str,
) -> None:
    if profile is None:
        return
    profile["matvec_path_mode"] = str(path_mode)
    profile["matvec_path_mode_effective"] = str(effective_mode)
    profile["matvec_path_fallback_reason"] = str(fallback_reason)


def resolve_epq_streaming_runtime(
    workspace: Any,
    *,
    use_epq_streaming: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    epq_stream_panic_requested: bool,
    epq_stream_panic_active: bool,
) -> dict[str, Any]:
    epq_stream_db_requested = False
    epq_stream_db_active = False
    epq_stream_j_tile = int(getattr(workspace, "epq_stream_j_tile", 0))

    if use_epq_streaming:
        db_mode = str(getattr(workspace, "epq_stream_double_buffer_mode", "off")).strip().lower()
        if db_mode == "on":
            epq_stream_db_requested = True
        elif db_mode == "off":
            epq_stream_db_requested = False
        else:
            epq_stream_db_requested = bool(int(getattr(workspace, "ncsf", 0)) > int(epq_stream_j_tile))
        # Safe policy: overlap build/apply only when overflow checks are disabled and profiling is off.
        epq_stream_db_active = bool(
            epq_stream_db_requested
            and int(epq_stream_j_tile) < int(getattr(workspace, "ncsf", 0))
            and profile is None
            and (not bool(check_overflow))
            and (not bool(epq_stream_panic_active))
        )

    if profile is not None:
        profile["epq_streaming"] = float(1.0 if use_epq_streaming else 0.0)
        if use_epq_streaming:
            profile["epq_stream_j_tile"] = float(int(epq_stream_j_tile))
            profile["epq_stream_pq_block"] = float(int(getattr(workspace, "epq_stream_pq_block", 0)))
            profile["epq_stream_panic_requested"] = float(1.0 if epq_stream_panic_requested else 0.0)
            profile["epq_stream_panic"] = float(1.0 if epq_stream_panic_active else 0.0)
            profile["epq_stream_double_buffer_requested"] = float(1.0 if epq_stream_db_requested else 0.0)
            profile["epq_stream_double_buffer"] = float(1.0 if epq_stream_db_active else 0.0)

    return {
        "epq_stream_j_tile": int(epq_stream_j_tile),
        "epq_stream_db_requested": bool(epq_stream_db_requested),
        "epq_stream_db_active": bool(epq_stream_db_active),
    }


def resolve_one_body_mode(
    *,
    use_fused_hop: bool,
    use_epq_streaming: bool,
    epq_stream_panic_active: bool,
    epq_stream_db_active: bool,
) -> str:
    if bool(use_fused_hop):
        return "fused_zero"
    if not bool(use_epq_streaming):
        return "single_scatter"
    if bool(epq_stream_panic_active):
        return "epq_panic"
    if bool(epq_stream_db_active):
        return "epq_double_buffer"
    return "epq_single_buffer"


def run_one_body_epq_streaming(
    workspace: Any,
    *,
    cp: Any,
    one_body_mode: str,
    x: Any,
    y: Any,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    epq_stream_j_tile: int,
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
) -> None:
    """Run EPQ-streaming one-body apply loop for panic/single/double-buffer modes."""

    step = int(epq_stream_j_tile)
    if step <= 0:
        raise ValueError("epq_stream_j_tile must be > 0")
    mode_s = str(one_body_mode)
    if mode_s not in {"epq_panic", "epq_double_buffer", "epq_single_buffer"}:
        raise ValueError(f"unsupported one-body EPQ mode: {mode_s!r}")

    if mode_s == "epq_panic":
        zero_y = True
        for j0 in range(0, int(workspace.ncsf), int(step)):
            j1 = min(int(workspace.ncsf), int(j0 + int(step)))
            j_d = workspace.task_csf_all[int(j0) : int(j1)]
            t_apply0 = time.perf_counter() if profile is not None else None
            apply_g_flat_scatter_atomic_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                j_d,
                h_eff_flat,
                task_scale=cp.asarray(x[int(j0) : int(j1)], dtype=workspace._dtype),
                epq_table=None,
                apply_mode="scatter",
                y=y,
                overflow=workspace.overflow_apply,
                threads=int(workspace.threads_apply),
                zero_y=bool(zero_y),
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                dtype=workspace._dtype,
                use_kahan=bool(getattr(workspace, "kahan_compensation", False)),
            )
            if profile is not None and t_apply0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t_apply0
                profile["epq_stream_panic_apply_s"] = profile.get("epq_stream_panic_apply_s", 0.0) + dt
            zero_y = False
        return

    if mode_s == "epq_double_buffer":
        stream_build = cp.cuda.Stream(non_blocking=True)
        stream_apply = cp.cuda.Stream(non_blocking=True)
        evt_build_done = [cp.cuda.Event(disable_timing=True), cp.cuda.Event(disable_timing=True)]
        evt_apply_done = [cp.cuda.Event(disable_timing=True), cp.cuda.Event(disable_timing=True)]

        last_slot = 0
        nt = 0
        for tile_idx, j0 in enumerate(range(0, int(workspace.ncsf), int(step))):
            slot = int(tile_idx & 1)
            if tile_idx >= 2:
                stream_build.wait_event(evt_apply_done[slot])

            j1 = min(int(workspace.ncsf), int(j0 + int(step)))
            jc = int(j1 - j0)
            local_indptr, tile_indices, tile_pq, tile_data = build_epq_stream_tile_profiled(
                workspace,
                build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                profile=profile,
                j_start=int(j0),
                j_count=int(jc),
                global_indptr=False,
                stream_override=stream_build,
                sync_override=True,
                check_overflow_override=False,
            )
            evt_build_done[slot].record(stream_build)

            stream_apply.wait_event(evt_build_done[slot])
            apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                local_indptr,
                tile_indices,
                tile_pq,
                tile_data,
                h_eff_flat,
                task_scale=cp.asarray(x[int(j0) : int(j1)], dtype=workspace._dtype),
                j_start=int(j0),
                j_count=int(jc),
                y=y,
                overflow=workspace.overflow_apply,
                threads=int(workspace.threads_apply),
                zero_y=bool(tile_idx == 0),
                stream=stream_apply,
                sync=False,
                check_overflow=False,
                dtype=workspace._dtype,
                use_kahan=bool(getattr(workspace, "kahan_compensation", False)),
            )
            evt_apply_done[slot].record(stream_apply)
            last_slot = int(slot)
            nt += 1

        if nt > 0:
            stream.wait_event(evt_apply_done[last_slot])
        return

    # epq_single_buffer
    zero_y = True
    for j0 in range(0, int(workspace.ncsf), int(step)):
        j1 = min(int(workspace.ncsf), int(j0 + int(step)))
        jc = int(j1 - j0)
        local_indptr, tile_indices, tile_pq, tile_data = build_epq_stream_tile_profiled(
            workspace,
            build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            j_start=int(j0),
            j_count=int(jc),
            global_indptr=False,
        )
        t_apply0 = time.perf_counter() if profile is not None else None
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn(
            workspace.drt,
            workspace.drt_dev,
            workspace.state_dev,
            local_indptr,
            tile_indices,
            tile_pq,
            tile_data,
            h_eff_flat,
            task_scale=cp.asarray(x[int(j0) : int(j1)], dtype=workspace._dtype),
            j_start=int(j0),
            j_count=int(jc),
            y=y,
            overflow=workspace.overflow_apply,
            threads=int(workspace.threads_apply),
            zero_y=bool(zero_y),
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            dtype=workspace._dtype,
            use_kahan=bool(getattr(workspace, "kahan_compensation", False)),
        )
        if profile is not None and t_apply0 is not None:
            stream.synchronize()
            dt = time.perf_counter() - t_apply0
            profile["tile_apply_apply_s"] = profile.get("tile_apply_apply_s", 0.0) + dt
        zero_y = False


def run_one_body_single_scatter(
    workspace: Any,
    *,
    cp: Any,
    x: Any,
    y: Any,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    use_aggregate_offdiag: bool,
    epq_table: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
) -> dict[str, Any]:
    """Run one-body single-scatter path and resolve overlap stream policy."""

    one_body_overlap = bool(
        bool(use_aggregate_offdiag)
        and epq_table is not None
        and getattr(workspace, "_w_offdiag", None) is None
        and profile is None
    )
    stream_onebody = None
    if one_body_overlap:
        stream_onebody = getattr(workspace, "_stream_onebody", None)
        if stream_onebody is None:
            stream_onebody = cp.cuda.Stream(non_blocking=True)
            setattr(workspace, "_stream_onebody", stream_onebody)

    apply_g_flat_scatter_atomic_inplace_device_fn(
        workspace.drt,
        workspace.drt_dev,
        workspace.state_dev,
        workspace.task_csf_all,
        h_eff_flat,
        task_scale=x,
        epq_table=epq_table,
        apply_mode=str(getattr(workspace, "apply_mode", "scatter")),
        y=y,
        overflow=workspace.overflow_apply,
        threads=int(workspace.threads_apply),
        zero_y=True,
        stream=stream_onebody if one_body_overlap else stream,
        sync=False if one_body_overlap else bool(sync),
        check_overflow=False if one_body_overlap else bool(check_overflow),
        dtype=workspace._dtype,
        use_kahan=bool(getattr(workspace, "kahan_compensation", False)),
    )
    return {
        "one_body_overlap": bool(one_body_overlap),
        "stream_onebody": stream_onebody,
    }


def finalize_hop_profile(
    *,
    profile: dict[str, float] | None,
    stream: Any,
    t_total0: float | None,
    csr_host_cache_hits: int,
    csr_host_cache_misses: int,
    epq_apply_cache_hits: int,
    epq_apply_cache_misses: int,
    epq_apply_cache_bytes: int,
) -> None:
    if profile is None:
        return

    offdiag_gemm_s = float(profile.get("offdiag_gemm_s", 0.0))
    offdiag_gemm_flops = float(profile.get("offdiag_gemm_flops", 0.0))
    if offdiag_gemm_s > 0.0 and offdiag_gemm_flops > 0.0:
        profile["offdiag_gemm_tflops"] = float(offdiag_gemm_flops / offdiag_gemm_s / 1.0e12)

    offdiag_df_gemm_s = float(profile.get("offdiag_df_gemm_s", 0.0))
    offdiag_df_gemm_flops = float(profile.get("offdiag_df_gemm_flops", 0.0))
    if offdiag_df_gemm_s > 0.0 and offdiag_df_gemm_flops > 0.0:
        profile["offdiag_df_gemm_tflops"] = float(offdiag_df_gemm_flops / offdiag_df_gemm_s / 1.0e12)

    csr_total = max(1, int(csr_host_cache_hits) + int(csr_host_cache_misses))
    profile["csr_host_cache_hit_rate"] = float(int(csr_host_cache_hits)) / float(csr_total)

    epq_total = max(1, int(epq_apply_cache_hits) + int(epq_apply_cache_misses))
    profile["epq_apply_cache_hit_rate"] = float(int(epq_apply_cache_hits)) / float(epq_total)
    profile["epq_apply_cache_bytes"] = float(int(epq_apply_cache_bytes))

    if t_total0 is not None:
        stream.synchronize()
        profile["total_s"] = profile.get("total_s", 0.0) + (time.perf_counter() - t_total0)


def resolve_epq_table_for_tile(
    workspace: Any,
    *,
    use_epq_streaming_tiles: bool,
    j_start: int,
    j_count: int,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    build_epq_action_table_tile_device_fn: Any,
):
    if not bool(use_epq_streaming_tiles):
        return getattr(workspace, "_epq_table", None)
    return build_epq_stream_tile_profiled(
        workspace,
        build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        profile=profile,
        j_start=int(j_start),
        j_count=int(j_count),
        global_indptr=True,
    )


def finalize_one_body_phase(
    *,
    cp: Any,
    stream: Any,
    one_body_overlap: bool,
    stream_onebody: Any,
    profile: dict[str, float] | None,
    t0: float | None,
):
    one_body_event = cp.cuda.Event(disable_timing=True)
    one_body_event.record(stream_onebody if bool(one_body_overlap) else stream)

    if profile is not None and t0 is not None:
        stream.synchronize()
        if stream_onebody is not None:
            stream_onebody.synchronize()
        profile["one_body_s"] = profile.get("one_body_s", 0.0) + (time.perf_counter() - t0)
    return one_body_event


def resolve_tile_runtime_flags(
    workspace: Any,
    *,
    j0: int,
    check_overflow: bool,
    sync: bool,
    use_csr_pipeline: bool,
) -> dict[str, Any]:
    check_overflow_tile = bool(check_overflow)
    check_overflow_mode_tile = int(getattr(workspace, "check_overflow_mode", 0))
    if check_overflow_tile and bool(getattr(workspace, "check_overflow_first_tile_only", False)) and int(j0) != 0:
        check_overflow_tile = False
    if not check_overflow_tile:
        check_overflow_mode_tile = 0

    check_overflow_apply_tile = bool(check_overflow)
    if check_overflow_apply_tile and bool(getattr(workspace, "check_overflow_first_tile_only", False)) and int(j0) != 0:
        check_overflow_apply_tile = False

    tile_sync_apply = bool(sync)
    if bool(use_csr_pipeline) and (not bool(check_overflow_apply_tile)):
        tile_sync_apply = False

    return {
        "check_overflow_tile": bool(check_overflow_tile),
        "check_overflow_mode_tile": int(check_overflow_mode_tile),
        "check_overflow_apply_tile": bool(check_overflow_apply_tile),
        "tile_sync_apply": bool(tile_sync_apply),
    }


def stamp_csr_prefilter_profile(
    *,
    profile: dict[str, float] | None,
    use_csr_pipeline: bool,
    pipeline_nslots: int,
    use_prefilter_trivial: bool,
) -> None:
    if profile is None:
        return
    profile["csr_pipeline_active"] = float(1.0 if bool(use_csr_pipeline) else 0.0)
    profile["csr_pipeline_slots"] = float(int(pipeline_nslots))
    profile["csr_prefilter_active"] = float(1.0 if bool(use_prefilter_trivial) else 0.0)


def increment_profile_counter(
    *,
    profile: dict[str, float] | None,
    key: str,
    delta: float = 1.0,
) -> None:
    if profile is None:
        return
    profile[str(key)] = profile.get(str(key), 0.0) + float(delta)


def stamp_csr_prefilter_tile_profile(
    *,
    profile: dict[str, float] | None,
    total_tasks: int,
    kept_tasks: int,
) -> dict[str, Any]:
    skipped_tasks = int(total_tasks) - int(kept_tasks)
    increment_profile_counter(profile=profile, key="csr_prefilter_tiles", delta=1.0)
    increment_profile_counter(profile=profile, key="csr_prefilter_total_tasks", delta=float(total_tasks))
    increment_profile_counter(profile=profile, key="csr_prefilter_kept_tasks", delta=float(kept_tasks))
    increment_profile_counter(profile=profile, key="csr_prefilter_skipped_tasks", delta=float(skipped_tasks))
    is_zero_tile = bool(int(kept_tasks) <= 0)
    if is_zero_tile:
        increment_profile_counter(profile=profile, key="csr_prefilter_zero_tiles", delta=1.0)
    return {
        "skipped_tasks": int(skipped_tasks),
        "is_zero_tile": bool(is_zero_tile),
    }


def resolve_x_tile_skip_policy(
    workspace: Any,
    *,
    cp: Any,
    x: Any,
    use_fused_hop: bool,
    profile: dict[str, float] | None,
) -> dict[str, Any]:
    tile_active_mask = None
    skip_two_body_tiles = False

    if (
        (not bool(use_fused_hop))
        and bool(getattr(workspace, "skip_zero_x_tiles_enabled", False))
        and int(getattr(workspace, "j_tile", 0)) < int(getattr(workspace, "ncsf", 0))
    ):
        t_scan0 = time.perf_counter() if profile is not None else None
        ntiles = (int(workspace.ncsf) + int(workspace.j_tile) - 1) // int(workspace.j_tile)
        # Host-side scan avoids CuPy reduction JIT warmup costs in short-lived processes.
        x_h = np.asarray(cp.asnumpy(x))
        nnz_x = int(np.count_nonzero(x_h))
        if profile is not None:
            profile["x_tile_skip_policy_active"] = 1.0
            profile["x_nnz"] = float(nnz_x)
            profile["x_tile_total"] = float(ntiles)
        if nnz_x <= 0:
            tile_active_mask = np.zeros((int(ntiles),), dtype=np.bool_)
            skip_two_body_tiles = True
        elif nnz_x < int(workspace.ncsf):
            nz_tiles_h = np.unique(np.flatnonzero(x_h) // int(workspace.j_tile)).astype(np.int64, copy=False)
            tile_active_mask = np.zeros((int(ntiles),), dtype=np.bool_)
            if int(nz_tiles_h.size) > 0:
                tile_active_mask[nz_tiles_h] = True
        if profile is not None:
            if tile_active_mask is None:
                profile["x_tile_skip_mask_active"] = 0.0
                profile["x_tile_skipped"] = 0.0
            else:
                skipped = int(int(ntiles) - int(np.count_nonzero(tile_active_mask)))
                profile["x_tile_skip_mask_active"] = 1.0
                profile["x_tile_skipped"] = float(skipped)
                profile["x_tile_active"] = float(int(ntiles) - int(skipped))
            if t_scan0 is not None:
                profile["x_tile_scan_s"] = profile.get("x_tile_scan_s", 0.0) + (time.perf_counter() - t_scan0)
    elif profile is not None:
        profile["x_tile_skip_policy_active"] = 0.0

    return {
        "tile_active_mask": tile_active_mask,
        "skip_two_body_tiles": bool(skip_two_body_tiles),
    }
