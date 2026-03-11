from __future__ import annotations

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
    use_df = False
    if eri_mat_use is not None:
        if eri_mat_use.shape != (int(workspace.nops), int(workspace.nops)):
            raise ValueError("eri_mat must have shape (nops,nops)")
    else:
        l_full_use = workspace.l_full
        if l_full_use is None:
            raise ValueError("eri_mat or l_full must be provided (workspace has neither)")
        l_full_use = cp.ascontiguousarray(cp.asarray(l_full_use, dtype=workspace._dtype))
        if l_full_use.ndim != 2 or tuple(l_full_use.shape)[0] != int(workspace.nops):
            raise ValueError("l_full must have shape (norb*norb, naux)")
        use_df = True

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
        "use_df": bool(use_df),
        "h_eff_flat": h_eff_flat,
        "use_epq_streaming": bool(use_epq_streaming),
        "use_epq_streaming_tiles": bool(use_epq_streaming_tiles),
        "epq_stream_panic_requested": bool(epq_stream_panic_requested),
        "epq_stream_panic_active": bool(epq_stream_panic_active),
    }


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
