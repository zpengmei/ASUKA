from __future__ import annotations

import time
from typing import Any

from asuka.cuda._backend_hop_runtime import (
    finalize_hop_profile,
    finalize_one_body_phase,
    increment_profile_counter,
    resolve_epq_streaming_runtime,
    resolve_epq_table_for_tile,
    resolve_one_body_mode,
    resolve_tile_runtime_flags,
    resolve_x_tile_skip_policy,
    run_one_body_epq_streaming,
    run_one_body_single_scatter,
    set_matvec_path_profile,
    stamp_csr_prefilter_profile,
    stamp_csr_prefilter_tile_profile,
)


def _prepare_blocked_hop_state(
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
    use_fused_hop: bool,
    use_aggregate_offdiag: bool,
    use_epq_streaming: bool,
    epq_stream_panic_requested: bool,
    epq_stream_panic_active: bool,
    path_mode: str,
    epq_table: Any,
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
) -> dict[str, Any]:
    """Prepare common state for the blocked hop path.

    Starts the total timer, runs the one-body phase, and resolves the
    EPQ-blocked aggregate guard.
    """
    t_total0 = time.perf_counter() if profile is not None else None

    _ob = run_one_body_phase(
        workspace,
        cp=cp,
        x=x,
        y=y,
        h_eff_flat=h_eff_flat,
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        profile=profile,
        use_fused_hop=bool(use_fused_hop),
        use_aggregate_offdiag=bool(use_aggregate_offdiag),
        use_epq_streaming=bool(use_epq_streaming),
        epq_stream_panic_requested=bool(epq_stream_panic_requested),
        epq_stream_panic_active=bool(epq_stream_panic_active),
        path_mode=str(path_mode),
        epq_table=epq_table,
        build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device_fn,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn,
    )
    one_body_event = _ob["one_body_event"]

    # EPQ-blocked aggregate guard
    use_epq_agg_blocked = (
        bool(use_aggregate_offdiag)
        and epq_table is not None
        and workspace._w_offdiag is None
    )
    if use_epq_agg_blocked and (not workspace._should_use_blocked_epq_transpose(epq_table, profile=profile)):
        use_epq_agg_blocked = False
        if profile is not None:
            profile["epq_transpose_guard_fallback"] = profile.get("epq_transpose_guard_fallback", 0.0) + 1.0

    return {
        "t_total0": t_total0,
        "one_body_event": one_body_event,
        "use_epq_agg_blocked": bool(use_epq_agg_blocked),
    }


def run_one_body_phase(
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
    use_fused_hop: bool,
    use_aggregate_offdiag: bool,
    use_epq_streaming: bool,
    epq_stream_panic_requested: bool,
    epq_stream_panic_active: bool,
    path_mode: str,
    epq_table: Any,
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
) -> dict[str, Any]:
    """Resolve EPQ streaming config, dispatch one-body mode, return one_body_event.

    Extracted from GugaMatvecEriMatWorkspace.hop() to make the one-body phase
    a standalone unit shared by both the EPQ-blocked and per-tile CSR paths.
    """
    set_matvec_path_profile(
        profile=profile,
        path_mode=path_mode,
        effective_mode="epq_blocked",
        fallback_reason=str(getattr(workspace, "path_mode_fallback_reason", "") or ""),
    )

    _stream_runtime = resolve_epq_streaming_runtime(
        workspace,
        use_epq_streaming=bool(use_epq_streaming),
        check_overflow=bool(check_overflow),
        profile=profile,
        epq_stream_panic_requested=bool(epq_stream_panic_requested),
        epq_stream_panic_active=bool(epq_stream_panic_active),
    )
    epq_stream_j_tile = int(_stream_runtime["epq_stream_j_tile"])
    epq_stream_db_active = bool(_stream_runtime["epq_stream_db_active"])

    _one_body_overlap = False
    _stream_onebody = None
    one_body_mode = resolve_one_body_mode(
        use_fused_hop=bool(use_fused_hop),
        use_epq_streaming=bool(use_epq_streaming),
        epq_stream_panic_active=bool(epq_stream_panic_active),
        epq_stream_db_active=bool(epq_stream_db_active),
    )
    if profile is not None:
        profile["one_body_mode"] = str(one_body_mode)
    t0 = time.perf_counter() if profile is not None else None
    if one_body_mode == "fused_zero":
        cp.cuda.runtime.memsetAsync(
            int(y.data.ptr),
            0,
            int(y.size) * int(y.itemsize),
            int(stream.ptr),
        )
    elif one_body_mode in {"epq_panic", "epq_double_buffer", "epq_single_buffer"}:
        run_one_body_epq_streaming(
            workspace,
            cp=cp,
            one_body_mode=str(one_body_mode),
            x=x,
            y=y,
            h_eff_flat=h_eff_flat,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            epq_stream_j_tile=int(epq_stream_j_tile),
            build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
            apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device_fn,
            apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn,
        )
    else:
        _single = run_one_body_single_scatter(
            workspace,
            cp=cp,
            x=x,
            y=y,
            h_eff_flat=h_eff_flat,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            use_aggregate_offdiag=bool(use_aggregate_offdiag),
            epq_table=epq_table,
            apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device_fn,
        )
        _one_body_overlap = bool(_single["one_body_overlap"])
        _stream_onebody = _single["stream_onebody"]

    _one_body_event = finalize_one_body_phase(
        cp=cp,
        stream=stream,
        one_body_overlap=bool(_one_body_overlap),
        stream_onebody=_stream_onebody,
        profile=profile,
        t0=t0,
    )

    return {
        "one_body_event": _one_body_event,
        "epq_stream_j_tile": int(epq_stream_j_tile),
        "epq_stream_db_active": bool(epq_stream_db_active),
    }


def run_epq_blocked_aggregate_path(
    workspace: Any,
    *,
    cp: Any,
    ext: Any,
    x: Any,
    y: Any,
    eri_mat_use: Any,
    l_full_use: Any,
    direct_op_use: Any,
    use_df: bool,
    use_direct: bool,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    t_total0: float | None,
    one_body_event: Any,
    epq_table: Any,
    build_epq_action_table_transpose_device_fn: Any,
    build_w_from_epq_transpose_range_inplace_device_fn: Any,
    apply_g_flat_gather_epq_transpose_range_inplace_device_fn: Any,
    sym_pair_pack_inplace_device_fn: Any,
    sym_pair_unpack_inplace_device_fn: Any,
    has_sym_pair_fused_kernels_fn: Any,
    Kernel3BuildGDFWorkspace_cls: Any,
) -> Any:
    """Run the blocked EPQ-table aggregate path.

    Builds EPQ transpose, inits GEMM workspace (dense/DF), optional sym-pair setup,
    k-blocking loop (zero W -> diag W -> offdiag W -> GEMM -> apply g),
    profile finalization, return y.

    Extracted from GugaMatvecEriMatWorkspace.hop().
    """
    import os as _os

    _one_body_event = one_body_event

    nrows_block_max = int(getattr(workspace._g_buf, "shape", (0, 0))[0])
    if nrows_block_max <= 0:
        raise RuntimeError("internal error: invalid g_buf block size for aggregate_offdiag_k")
    epq_table_t = build_epq_action_table_transpose_device_fn(
        workspace.drt,
        epq_table,
        dtype=workspace._dtype,
        indptr_dtype=workspace.epq_indptr_dtype,
        use_cache=True,
    )

    # Dense ERI_mat blocked path needs a per-block W buffer (cannot GEMM in-place).
    offdiag_gemm_ws = None
    w_block_buf = None
    if not use_df:
        offdiag_gemm_ws = workspace._offdiag_gemm_ws
        if offdiag_gemm_ws is None:
            raise RuntimeError("internal error: offdiag_gemm_ws is not initialized for aggregate_offdiag_k")
        w_block_buf = workspace._w_block
        if (
            w_block_buf is None
            or not hasattr(w_block_buf, "shape")
            or tuple(w_block_buf.shape) != (int(nrows_block_max), int(workspace.nops))
        ):
            workspace._w_block = cp.empty((int(nrows_block_max), int(workspace.nops)), dtype=workspace._dtype)
            w_block_buf = workspace._w_block
        if profile is not None:
            profile["offdiag_cublas_workspace_bytes"] = float(int(workspace._offdiag_cublas_workspace_bytes))

    # DF (L_full) blocked path uses g_buf as W scratch.
    gdf_ws = None
    if use_df:
        if l_full_use is None:  # pragma: no cover
            raise RuntimeError("internal error: l_full_use is not set for DF path")
        naux = int(l_full_use.shape[1])
        gdf_ws = workspace._gdf_ws
        if (
            gdf_ws is None
            or int(getattr(gdf_ws, "naux", 0)) != int(naux)
            or int(getattr(gdf_ws, "max_nrows", 0)) < int(nrows_block_max)
        ):
            workspace._gdf_ws = Kernel3BuildGDFWorkspace_cls(
                int(workspace.nops),
                int(naux),
                max_nrows=int(nrows_block_max),
            )
            gdf_ws = workspace._gdf_ws

        if gdf_ws is None:  # pragma: no cover
            raise RuntimeError("internal error: failed to initialize Kernel3BuildGDFWorkspace")

        if workspace.offdiag_enable_fp64_emulation:
            gdf_ws.set_gemm_backend("gemmex_emulated_fixedpoint")
            gdf_ws.set_cublas_math_mode("fp64_emulated_fixedpoint")
            if workspace.offdiag_emulation_strategy:
                strategy = str(workspace.offdiag_emulation_strategy).strip().lower()
                if strategy == "eager":
                    allow = str(_os.getenv("CUGUGA_ALLOW_CUBLAS_EAGER", "")).strip().lower()
                    if allow not in ("1", "true", "yes"):
                        raise RuntimeError(
                            "offdiag_emulation_strategy='eager' is experimental and may crash for some shapes/GPUs; "
                            "set CUGUGA_ALLOW_CUBLAS_EAGER=1 to enable anyway"
                        )
                gdf_ws.set_cublas_emulation_strategy(strategy)
            # Configure workspace (one-time; capped).
            try:
                gdf_ws.autoset_cublas_workspace_bytes(
                    nrows=int(nrows_block_max),
                    cap_mb=int(workspace.offdiag_cublas_workspace_cap_mb),
                )
            except Exception:
                pass
        else:
            gdf_ws.set_gemm_backend(str(workspace.gemm_backend))
            gdf_ws.set_cublas_math_mode("default")

        if profile is not None:
            profile["eri_mat_used"] = float(0.0)
            profile["df_l_full_used"] = float(1.0)
            profile["df_cublas_workspace_bytes"] = float(int(gdf_ws.cublas_workspace_bytes()))

    # Symmetric-pair GEMM setup (Opt 2): compress nops -> npair = norb*(norb+1)/2.
    use_sym_pair = False
    eri_pair = None
    l_pair = None
    l_pair_t = None
    npair = workspace._sym_pair_npair
    if (not use_direct) and npair > 0 and workspace._sym_pair_pair_pq is not None:
        if use_df:
            # DF path: sym-pair compression only reduces one GEMM dimension
            # (npair vs nops) while adding pack/unpack overhead -- not beneficial.
            pass
        else:
            eri_pair = workspace._sym_pair_get_eri_pair(eri_mat_use)
            if eri_pair is not None:
                use_sym_pair = True
        if use_sym_pair:
            workspace._sym_pair_ensure_buffers(nrows_block_max)
            if profile is not None:
                profile["sym_pair_active"] = 1.0
                profile["sym_pair_npair"] = float(npair)
                profile["sym_pair_nops"] = float(int(workspace.nops))

    # Detect if fused pair-indexed kernels are available (avoids pack/unpack overhead).
    use_pair_fused = bool(use_sym_pair and has_sym_pair_fused_kernels_fn()
                          and workspace._dtype == cp.float64)
    _ftp_dev = workspace._sym_pair_full_to_pair if use_pair_fused else None

    for k0 in range(0, int(workspace.ncsf), int(nrows_block_max)):
        k1 = min(int(workspace.ncsf), int(k0 + int(nrows_block_max)))
        k_count = int(k1 - k0)

        # Select W buffer: pair-indexed (k_count, npair) or full (k_count, nops).
        if use_pair_fused:
            w_pair_buf = workspace._sym_pair_w_pair[:k_count]
            w_target = w_pair_buf  # W-build/W-diag write directly to pair buffer
        else:
            if use_df:
                w_block = workspace._g_buf[:k_count]
            else:
                if w_block_buf is None:  # pragma: no cover
                    raise RuntimeError("internal error: missing w_block_buf for dense blocked aggregate")
                w_block = w_block_buf[:k_count]
            w_target = w_block

        # Zero W buffer.
        t0 = time.perf_counter() if profile is not None else None
        cp.cuda.runtime.memsetAsync(
            int(w_target.data.ptr),
            0,
            int(w_target.size) * int(w_target.itemsize),
            int(stream.ptr),
        )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["offdiag_w_zero_s"] = profile.get("offdiag_w_zero_s", 0.0) + (time.perf_counter() - t0)

        if workspace.include_diagonal_rs:
            # Fill diagonal rs (r==s) entries for this k-block.
            t0 = time.perf_counter() if profile is not None else None
            if use_pair_fused:
                ext.build_w_diag_pair_from_steps_inplace_device(
                    workspace.state_dev,
                    int(k0),
                    int(k_count),
                    x,
                    w_pair_buf,
                    _ftp_dev,
                    256,  # threads
                    int(stream.ptr),
                    bool(sync),
                    True,  # relative_w
                )
            else:
                workspace._build_w_diag_from_steps_inplace(
                    x=x,
                    w_out=w_target,
                    j_start=int(k0),
                    j_count=int(k_count),
                    stream=stream,
                    sync=bool(sync),
                    relative_w=True,
                )
            if profile is not None and t0 is not None:
                stream.synchronize()
                profile["diag_w_build_s"] = profile.get("diag_w_build_s", 0.0) + (time.perf_counter() - t0)

        # Off-diagonal W from epq_table.
        t0 = time.perf_counter() if profile is not None else None
        if use_pair_fused:
            t_indptr, t_source, t_pq, t_data = epq_table_t
            ext.build_w_pair_from_epq_transpose_range_inplace_device(
                workspace.state_dev,
                t_indptr,
                t_source,
                t_pq,
                t_data,
                x,
                w_pair_buf,
                _ftp_dev,
                int(npair),
                workspace._overflow_w,
                int(workspace.threads_w),
                int(stream.ptr),
                bool(sync),
                bool(check_overflow),
                int(k0),
                int(k_count),
            )
        else:
            build_w_from_epq_transpose_range_inplace_device_fn(
                workspace.drt,
                workspace.state_dev,
                epq_table_t,
                x,
                w_out=w_target,
                overflow=workspace._overflow_w,
                threads=int(workspace.threads_w),
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                k_start=int(k0),
                k_count=int(k_count),
                dtype=workspace._dtype,
            )
        if profile is not None and t0 is not None:
            stream.synchronize()
            dt = time.perf_counter() - t0
            profile["offdiag_w_build_s"] = profile.get("offdiag_w_build_s", 0.0) + dt
            profile["blocked_w_build_s"] = profile.get("blocked_w_build_s", 0.0) + dt

        # Contract W against ERIs to build g_block.
        direct_applied = False
        if use_pair_fused:
            # Pair-fused path: W is already pair-indexed, GEMM directly, Apply reads pair-indexed g.
            t0 = time.perf_counter() if profile is not None else None
            g_pair_buf = workspace._sym_pair_g_pair[:k_count]
            if use_df:
                naux = int(l_full_use.shape[1])
                df_t_buf = workspace._offdiag_df_t
                if (
                    df_t_buf is None
                    or not hasattr(df_t_buf, "shape")
                    or tuple(df_t_buf.shape) != (int(nrows_block_max), int(naux))
                ):
                    workspace._offdiag_df_t = cp.empty((int(nrows_block_max), int(naux)), dtype=workspace._dtype)
                    df_t_buf = workspace._offdiag_df_t
                z_block = df_t_buf[:k_count]
                cp.matmul(w_pair_buf, l_pair, out=z_block)
                z_block *= 0.5
                cp.matmul(z_block, l_pair_t, out=g_pair_buf)
                gemm_flops = float(4.0 * float(k_count) * float(npair) * float(naux))
            else:
                sp_ws = workspace._sym_pair_gemm_ws
                if sp_ws is not None and workspace._dtype == cp.float64:
                    sp_ws.gemm_w_eri_mat_inplace_device(
                        w_pair_buf, eri_pair, g_out=g_pair_buf,
                        dtype=workspace._dtype, half=0.5, stream=stream, sync=False,
                    )
                else:
                    cp.matmul(w_pair_buf, eri_pair, out=g_pair_buf)
                    g_pair_buf *= 0.5
                gemm_flops = float(2.0 * float(k_count) * float(npair) * float(npair))
            # g_pair_buf IS the g_block for the pair-indexed Apply kernel.
            g_block = g_pair_buf
            if profile is not None and t0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t0
                gemm_key = "offdiag_df_gemm_s" if use_df else "offdiag_gemm_s"
                flops_key = "offdiag_df_gemm_flops" if use_df else "offdiag_gemm_flops"
                profile[gemm_key] = profile.get(gemm_key, 0.0) + dt
                profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
                profile[flops_key] = profile.get(flops_key, 0.0) + gemm_flops
        elif use_sym_pair:
            # Legacy pack/unpack path (fallback when fused kernels unavailable).
            t0 = time.perf_counter() if profile is not None else None
            w_pair_buf = workspace._sym_pair_w_pair[:k_count]
            g_pair_buf = workspace._sym_pair_g_pair[:k_count]
            is_f32 = bool(workspace._dtype == cp.float32)
            sym_pair_pack_inplace_device_fn(
                w_block, w_pair_buf,
                workspace._sym_pair_pair_pq, workspace._sym_pair_pair_qp,
                nrows=k_count, nops=int(workspace.nops), npair=npair,
                is_f32=is_f32, stream=stream, sync=False,
            )
            if use_df:
                naux = int(l_full_use.shape[1])
                df_t_buf = workspace._offdiag_df_t
                if (
                    df_t_buf is None
                    or not hasattr(df_t_buf, "shape")
                    or tuple(df_t_buf.shape) != (int(nrows_block_max), int(naux))
                ):
                    workspace._offdiag_df_t = cp.empty((int(nrows_block_max), int(naux)), dtype=workspace._dtype)
                    df_t_buf = workspace._offdiag_df_t
                z_block = df_t_buf[:k_count]
                cp.matmul(w_pair_buf, l_pair, out=z_block)
                z_block *= 0.5
                cp.matmul(z_block, l_pair_t, out=g_pair_buf)
                gemm_flops = float(4.0 * float(k_count) * float(npair) * float(naux))
            else:
                sp_ws = workspace._sym_pair_gemm_ws
                if sp_ws is not None and workspace._dtype == cp.float64:
                    sp_ws.gemm_w_eri_mat_inplace_device(
                        w_pair_buf, eri_pair, g_out=g_pair_buf,
                        dtype=workspace._dtype, half=0.5, stream=stream, sync=False,
                    )
                else:
                    cp.matmul(w_pair_buf, eri_pair, out=g_pair_buf)
                    g_pair_buf *= 0.5
                gemm_flops = float(2.0 * float(k_count) * float(npair) * float(npair))
            g_block = workspace._g_buf[:k_count]
            sym_pair_unpack_inplace_device_fn(
                g_pair_buf, g_block, workspace._sym_pair_full_to_pair,
                nrows=k_count, nops=int(workspace.nops), npair=npair,
                is_f32=is_f32, stream=stream, sync=False,
            )
            if profile is not None and t0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t0
                gemm_key = "offdiag_df_gemm_s" if use_df else "offdiag_gemm_s"
                flops_key = "offdiag_df_gemm_flops" if use_df else "offdiag_gemm_flops"
                profile[gemm_key] = profile.get(gemm_key, 0.0) + dt
                profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
                profile[flops_key] = profile.get(flops_key, 0.0) + gemm_flops
        elif use_direct:
            if direct_op_use is None:  # pragma: no cover
                raise RuntimeError("internal error: direct_op is not initialized")
            if callable(getattr(direct_op_use, "contract_apply_w_block_device", None)):
                if _one_body_event is not None:
                    stream.wait_event(_one_body_event)
                    _one_body_event = None
                t0 = time.perf_counter() if profile is not None else None
                direct_op_use.contract_apply_w_block_device(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    epq_table_t,
                    w_block,
                    k_start=int(k0),
                    y=y,
                    half=0.5,
                    overflow=workspace.overflow_apply,
                    threads_apply=int(workspace.threads_apply),
                    add=True,
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    use_kahan=bool(workspace.kahan_compensation),
                    profile=profile,
                )
                if profile is not None and t0 is not None:
                    stream.synchronize()
                    dt = time.perf_counter() - t0
                    profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
                    profile["blocked_w_apply_s"] = profile.get("blocked_w_apply_s", 0.0) + dt
                    profile["offdiag_apply_s"] = profile.get("offdiag_apply_s", 0.0) + dt
                direct_applied = True
            else:
                g_block = workspace._g_buf[:k_count]
                t0 = time.perf_counter() if profile is not None else None
                direct_op_use.contract_w_block_device(
                    w_block,
                    half=0.5,
                    out=g_block,
                )
                if profile is not None and t0 is not None:
                    stream.synchronize()
                    dt = time.perf_counter() - t0
                    profile["offdiag_direct_cuda_s"] = profile.get("offdiag_direct_cuda_s", 0.0) + dt
                    profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
        elif use_df:
            t0 = time.perf_counter() if profile is not None else None
            naux = int(l_full_use.shape[1])
            l_full_t = workspace._l_full_t
            if (
                l_full_t is None
                or not hasattr(l_full_t, "shape")
                or tuple(l_full_t.shape) != (int(naux), int(workspace.nops))
            ):
                workspace._l_full_t = l_full_use.T.copy()
                l_full_t = workspace._l_full_t
            df_t_buf = workspace._offdiag_df_t
            if (
                df_t_buf is None
                or not hasattr(df_t_buf, "shape")
                or tuple(df_t_buf.shape) != (int(nrows_block_max), int(naux))
            ):
                workspace._offdiag_df_t = cp.empty((int(nrows_block_max), int(naux)), dtype=workspace._dtype)
                df_t_buf = workspace._offdiag_df_t

            if df_t_buf is None or l_full_t is None:  # pragma: no cover
                raise RuntimeError("internal error: DF aggregate buffers are not initialized")
            t_block = df_t_buf[:k_count]

            try:
                if gdf_ws is None:  # pragma: no cover
                    raise RuntimeError("internal error: gdf_ws is not initialized")
                gdf_ws.gemm_w_l_full_inplace_device(
                    w_block,
                    l_full_use,
                    g_out=w_block,
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
            except Exception:
                # Fallback path: 2-step DF GEMM via CuPy (kept for compatibility and as a
                # safety net for unsupported dtype/backend combos).
                cp.dot(w_block, l_full_use, out=t_block)  # type: ignore[arg-type]
                t_block *= 0.5
                cp.dot(t_block, l_full_t, out=w_block)  # type: ignore[arg-type]
            if profile is not None and t0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t0
                profile["offdiag_df_gemm_s"] = profile.get("offdiag_df_gemm_s", 0.0) + dt
                profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
                profile["offdiag_df_gemm_flops"] = profile.get("offdiag_df_gemm_flops", 0.0) + float(
                    4.0 * float(int(k_count)) * float(int(workspace.nops)) * float(int(naux))
                )
            g_block = w_block
        else:
            if offdiag_gemm_ws is None:  # pragma: no cover
                raise RuntimeError("internal error: offdiag_gemm_ws is not initialized")
            # 10.16.8: Pad k_count to improve cuBLAS tile utilization for skinny GEMMs.
            k_count_padded = workspace._pad_k_count(k_count, nrows_max=nrows_block_max)
            gemm_rows = int(k_count)
            if k_count_padded > k_count:
                # Use padded views for GEMM: w_block_padded and g_block_padded.
                # The padding rows are already zeroed from the memsetAsync at the start of the loop.
                w_block_padded = w_block_buf[:k_count_padded]
                g_block_padded = workspace._g_buf[:k_count_padded]
                gemm_rows = int(k_count_padded)
                t0 = time.perf_counter() if profile is not None else None
                offdiag_gemm_ws.gemm_w_eri_mat_inplace_device(
                    w_block_padded,
                    eri_mat_use,
                    g_out=g_block_padded,
                    dtype=workspace._dtype,
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
                # Slice back to original k_count for the apply step.
                g_block = g_block_padded[:k_count]
            else:
                g_block = workspace._g_buf[:k_count]
                t0 = time.perf_counter() if profile is not None else None
                offdiag_gemm_ws.gemm_w_eri_mat_inplace_device(
                    w_block,
                    eri_mat_use,
                    g_out=g_block,
                    dtype=workspace._dtype,
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
            if profile is not None and t0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t0
                profile["offdiag_gemm_s"] = profile.get("offdiag_gemm_s", 0.0) + dt
                profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + dt
                profile["offdiag_gemm_flops"] = profile.get("offdiag_gemm_flops", 0.0) + float(
                    2.0 * float(int(gemm_rows)) * float(int(workspace.nops)) * float(int(workspace.nops))
                )

        if not (use_direct and direct_applied):
            # Wait for one_body to finish writing y before Apply accumulates into it.
            if _one_body_event is not None:
                stream.wait_event(_one_body_event)
                _one_body_event = None  # only need to wait once

            # Apply g_block to y via destination-major EPQ transpose gather.
            t0 = time.perf_counter() if profile is not None else None
            if use_pair_fused:
                t_indptr, t_source, t_pq, t_data = epq_table_t
                ext.apply_g_pair_gather_epq_transpose_range_inplace_device(
                    workspace.drt_dev,
                    workspace.state_dev,
                    t_indptr,
                    t_source,
                    t_pq,
                    t_data,
                    g_block,  # g_pair_buf (pair-indexed)
                    int(k0),
                    int(k_count),
                    _ftp_dev,
                    y,
                    workspace.overflow_apply,
                    int(workspace.threads_apply),
                    True,  # add
                    int(stream.ptr),
                    bool(sync),
                    bool(check_overflow),
                )
            else:
                apply_g_flat_gather_epq_transpose_range_inplace_device_fn(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    epq_table_t,
                    g_block,
                    k_start=int(k0),
                    k_count=int(k_count),
                    y=y,
                    overflow=workspace.overflow_apply,
                    threads=int(workspace.threads_apply),
                    add=True,
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    dtype=workspace._dtype,
                    use_kahan=bool(workspace.kahan_compensation),
                )
            if profile is not None and t0 is not None:
                stream.synchronize()
                dt = time.perf_counter() - t0
                profile["offdiag_apply_s"] = profile.get("offdiag_apply_s", 0.0) + dt
                profile["blocked_w_apply_s"] = profile.get("blocked_w_apply_s", 0.0) + dt

    if profile is not None:
        offdiag_gemm_s = float(profile.get("offdiag_gemm_s", 0.0))
        offdiag_gemm_flops = float(profile.get("offdiag_gemm_flops", 0.0))
        if offdiag_gemm_s > 0.0 and offdiag_gemm_flops > 0.0:
            profile["offdiag_gemm_tflops"] = float(offdiag_gemm_flops / offdiag_gemm_s / 1.0e12)
        offdiag_df_gemm_s = float(profile.get("offdiag_df_gemm_s", 0.0))
        offdiag_df_gemm_flops = float(profile.get("offdiag_df_gemm_flops", 0.0))
        if offdiag_df_gemm_s > 0.0 and offdiag_df_gemm_flops > 0.0:
            profile["offdiag_df_gemm_tflops"] = float(offdiag_df_gemm_flops / offdiag_df_gemm_s / 1.0e12)
    if profile is not None and t_total0 is not None:
        stream.synchronize()
        profile["total_s"] = profile.get("total_s", 0.0) + (time.perf_counter() - t_total0)
    return y


def run_per_tile_csr_path(
    workspace: Any,
    *,
    cp: Any,
    ext: Any,
    x: Any,
    y: Any,
    eri_mat_use: Any,
    eri_mat_t: Any,
    l_full_use: Any,
    direct_op_use: Any,
    use_df: bool,
    use_direct: bool,
    use_aggregate_offdiag: bool,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    t_total0: float | None,
    one_body_event: Any,
    use_fused_hop: bool,
    use_epq_streaming: bool,
    use_epq_streaming_tiles: bool,
    epq_stream_panic_active: bool,
    epq_table: Any,
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
    build_w_from_epq_table_inplace_device_fn: Any,
    build_occ_block_from_steps_inplace_device_fn: Any,
    kernel4_build_w_from_csr_unitnnz_inplace_device_fn: Any,
    kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn: Any,
    kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn: Any,
    kernel4_apply_csr_l_full_device_csr_inplace_device_fn: Any,
    kernel25_build_csr_from_tasks_deterministic_inplace_device_fn: Any,
    kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn: Any,
    Kernel3BuildGDFWorkspace_cls: Any,
) -> Any:
    """Run the per-tile CSR path.

    DF workspace setup, W_offdiag init, fast EPQ W build, CSR pipeline setup,
    x-sparsity tile skip, main j-tiling loop (diagonal rs, CSR build with retry,
    bucketed dispatch / kernel4), post-loop aggregate offdiag apply, pipeline sync,
    profile finalization, return y.

    Extracted from GugaMatvecEriMatWorkspace.hop().
    """
    import os as _os

    _one_body_event = one_body_event

    gdf_ws = None
    if use_df:
        if l_full_use is None:  # pragma: no cover
            raise RuntimeError("internal error: l_full_use is not set for DF path")
        want_naux = int(l_full_use.shape[1])
        want_max_nrows = int(getattr(workspace._g_buf, "shape", (0, 0))[0])
        if want_max_nrows < 1:
            raise RuntimeError("internal error: invalid g_buf block size for DF path")
        gdf_ws = workspace._gdf_ws
        if (
            gdf_ws is None
            or int(getattr(gdf_ws, "naux", 0)) != want_naux
            or int(getattr(gdf_ws, "max_nrows", 0)) < want_max_nrows
        ):
            workspace._gdf_ws = Kernel3BuildGDFWorkspace_cls(
                int(workspace.nops),
                int(want_naux),
                max_nrows=int(want_max_nrows),
            )
        gdf_ws = workspace._gdf_ws
        if gdf_ws is None:  # pragma: no cover
            raise RuntimeError("internal error: failed to initialize Kernel3BuildGDFWorkspace")
        if workspace.offdiag_enable_fp64_emulation:
            gdf_ws.set_gemm_backend("gemmex_emulated_fixedpoint")
            gdf_ws.set_cublas_math_mode("fp64_emulated_fixedpoint")
            if workspace.offdiag_emulation_strategy:
                strategy = str(workspace.offdiag_emulation_strategy).strip().lower()
                if strategy == "eager":
                    allow = str(_os.getenv("CUGUGA_ALLOW_CUBLAS_EAGER", "")).strip().lower()
                    if allow not in ("1", "true", "yes"):
                        raise RuntimeError(
                            "offdiag_emulation_strategy='eager' is experimental and may crash for some shapes/GPUs; "
                            "set CUGUGA_ALLOW_CUBLAS_EAGER=1 to enable anyway"
                        )
                gdf_ws.set_cublas_emulation_strategy(strategy)
            try:
                gdf_ws.autoset_cublas_workspace_bytes(
                    nrows=int(want_max_nrows),
                    cap_mb=int(workspace.offdiag_cublas_workspace_cap_mb),
                )
            except Exception:
                pass
        else:
            gdf_ws.set_gemm_backend(str(workspace.gemm_backend))
            gdf_ws.set_cublas_math_mode("default")
        if profile is not None:
            profile["eri_mat_used"] = float(0.0)
            profile["df_l_full_used"] = float(1.0)
            profile["df_cublas_workspace_bytes"] = float(int(gdf_ws.cublas_workspace_bytes()))

    w_offdiag = None
    offdiag_gemm_ws = None
    if bool(use_aggregate_offdiag):
        w_offdiag = workspace._w_offdiag
        offdiag_gemm_ws = workspace._offdiag_gemm_ws
        if offdiag_gemm_ws is None:
            raise RuntimeError("internal error: aggregate_offdiag_k gemm workspace is not initialized")
        if w_offdiag is not None:
            t0 = time.perf_counter() if profile is not None else None
            cp.cuda.runtime.memsetAsync(
                int(w_offdiag.data.ptr),
                0,
                int(w_offdiag.size) * int(w_offdiag.itemsize),
                int(stream.ptr),
            )
            if profile is not None and t0 is not None:
                stream.synchronize()
                profile["offdiag_w_zero_s"] = profile.get("offdiag_w_zero_s", 0.0) + (time.perf_counter() - t0)

    diag_in_w = bool(w_offdiag is not None) and bool(workspace.include_diagonal_rs)
    if diag_in_w:
        t0 = time.perf_counter() if profile is not None else None
        workspace._build_w_diag_from_steps_inplace(
            x=x,
            w_out=w_offdiag,
            j_start=0,
            j_count=int(workspace.ncsf),
            stream=stream,
            sync=bool(sync),
        )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["diag_w_build_s"] = profile.get("diag_w_build_s", 0.0) + (time.perf_counter() - t0)

    # Optional fast path: build off-diagonal W directly from EPQ table.
    skip_two_body_tiles = False
    if (
        w_offdiag is not None
        and (epq_table is not None or use_epq_streaming_tiles)
        and (
            workspace._dtype == cp.float32
            or (int(workspace.j_tile) < int(workspace.ncsf) and not bool(workspace.cache_csr_tiles))
        )
    ):
        t0 = time.perf_counter() if profile is not None else None
        if use_epq_streaming_tiles:
            stream_j_tile = int(workspace.epq_stream_j_tile)
            for j0 in range(0, int(workspace.ncsf), stream_j_tile):
                j1 = min(int(workspace.ncsf), int(j0 + stream_j_tile))
                epq_table_tile = resolve_epq_table_for_tile(
                    workspace,
                    use_epq_streaming_tiles=bool(use_epq_streaming_tiles),
                    j_start=int(j0),
                    j_count=int(j1 - j0),
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    profile=profile,
                    build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
                )
                build_w_from_epq_table_inplace_device_fn(
                    workspace.drt,
                    workspace.state_dev,
                    epq_table_tile,
                    x,
                    w_out=w_offdiag,
                    overflow=workspace._overflow_w,
                    threads=int(workspace.threads_w),
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    dtype=workspace._dtype,
                )
        else:
            build_w_from_epq_table_inplace_device_fn(
                workspace.drt,
                workspace.state_dev,
                epq_table,
                x,
                w_out=w_offdiag,
                overflow=workspace._overflow_w,
                threads=int(workspace.threads_w),
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                dtype=workspace._dtype,
            )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["offdiag_w_build_s"] = profile.get("offdiag_w_build_s", 0.0) + (time.perf_counter() - t0)
        skip_two_body_tiles = True

    use_csr_pipeline = bool(
        bool(getattr(workspace, "csr_pipeline_enabled", False))
        and profile is None
        and int(workspace.j_tile) < int(workspace.ncsf)
        and not bool(workspace.cache_csr_tiles)
        and not bool(getattr(workspace, "csr_host_cache_enabled", False))
        and not bool(use_epq_streaming_tiles)
        and workspace._csr_pipeline_apply_stream is not None
        and int(len(workspace._csr_pipeline_slots)) >= 2
    )
    pipeline_slots = workspace._csr_pipeline_slots if use_csr_pipeline else []
    pipeline_nslots = int(len(pipeline_slots)) if use_csr_pipeline else 0
    pipeline_apply_stream = workspace._csr_pipeline_apply_stream if use_csr_pipeline else None
    if use_csr_pipeline and pipeline_apply_stream is not None:
        y_ready_evt = cp.cuda.Event(disable_timing=True)
        y_ready_evt.record(stream)
        pipeline_apply_stream.wait_event(y_ready_evt)

    use_prefilter_trivial = bool(
        bool(getattr(workspace, "prefilter_trivial_tasks_enabled", False))
        and int(workspace.j_tile) < int(workspace.ncsf)
        and int(workspace._rs_n_pairs) > 0
        and not bool(use_epq_streaming_tiles)
    )
    stamp_csr_prefilter_profile(
        profile=profile,
        use_csr_pipeline=bool(use_csr_pipeline),
        pipeline_nslots=int(pipeline_nslots),
        use_prefilter_trivial=bool(use_prefilter_trivial),
    )

    _x_tile_skip = resolve_x_tile_skip_policy(
        workspace,
        cp=cp,
        x=x,
        use_fused_hop=bool(use_fused_hop),
        profile=profile,
    )
    tile_active_mask = _x_tile_skip["tile_active_mask"]
    if bool(_x_tile_skip["skip_two_body_tiles"]):
        skip_two_body_tiles = True

    # Two-body product term: process ket columns j in tiles.
    for tile_idx, j0 in enumerate(range(0, int(workspace.ncsf), int(workspace.j_tile))):
        if skip_two_body_tiles:
            break
        if tile_active_mask is not None and (not bool(tile_active_mask[int(tile_idx)])):
            continue
        j1 = min(int(workspace.ncsf), int(j0 + int(workspace.j_tile)))
        j_d = workspace.task_csf_all[int(j0) : int(j1)]
        j_count = int(j1 - j0)
        _tile_runtime_flags = resolve_tile_runtime_flags(
            workspace,
            j0=int(j0),
            check_overflow=bool(check_overflow),
            sync=bool(sync),
            use_csr_pipeline=bool(use_csr_pipeline),
        )
        check_overflow_tile = bool(_tile_runtime_flags["check_overflow_tile"])
        check_overflow_mode_tile = int(_tile_runtime_flags["check_overflow_mode_tile"])
        check_overflow_apply_tile = bool(_tile_runtime_flags["check_overflow_apply_tile"])
        tile_sync_apply = bool(_tile_runtime_flags["tile_sync_apply"])

        tile_slot = None
        stream_build = stream
        stream_apply = stream
        if use_csr_pipeline and pipeline_apply_stream is not None and pipeline_nslots >= 2:
            tile_slot = pipeline_slots[int(tile_idx) % int(pipeline_nslots)]
            stream_build = tile_slot["stream"]
            stream_apply = pipeline_apply_stream
            inflight_evt = tile_slot.get("inflight_event")
            if inflight_evt is not None:
                stream_build.wait_event(inflight_evt)

        # Fused hop kernel path: fused W/ERI/apply path replaces CSR build + dense GEMMs.
        if use_fused_hop:
            workspace._fused_hop_tile(
                j_start=int(j0),
                j_count=int(j_count),
                x=x,
                eri_mat=eri_mat_use,
                h_eff_flat=h_eff_flat,
                y=y,
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow_tile),
                profile=profile,
            )
            continue

        epq_table_j = resolve_epq_table_for_tile(
            workspace,
            use_epq_streaming_tiles=bool(use_epq_streaming_tiles),
            j_start=int(j0),
            j_count=int(j_count),
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        )

        # Diagonal rs terms: r==s, where E_rr is a diagonal number operator.
        if workspace.include_diagonal_rs and not diag_in_w:
            use_diag_w_gemm_fallback = bool(
                (workspace.eri_mat is not None)
                and (workspace._offdiag_gemm_ws is not None)
                and (
                    (
                        (workspace._dtype == cp.float32)
                        and (not bool(workspace.use_epq_table))
                        and int(workspace.ncsf) >= 1_000_000
                    )
                    or (
                        bool(workspace.use_epq_table)
                        and int(workspace.ncsf) >= 1_000_000
                    )
                )
            )
            g_diag = workspace._diag_g_cache.get(int(j0))
            if g_diag is None:
                if workspace._g_diag_buf is None:
                    raise RuntimeError("internal error: _g_diag_buf is not initialized")
                g_diag = workspace._g_diag_buf[:j_count]

                if use_diag_w_gemm_fallback:
                    w_diag_buf = workspace._diag_w_buf
                    if (
                        w_diag_buf is None
                        or not hasattr(w_diag_buf, "shape")
                        or tuple(w_diag_buf.shape) != (int(workspace.j_tile), int(workspace.nops))
                    ):
                        workspace._diag_w_buf = cp.empty((int(workspace.j_tile), int(workspace.nops)), dtype=workspace._dtype)
                        w_diag_buf = workspace._diag_w_buf
                    if w_diag_buf is None:  # pragma: no cover
                        raise RuntimeError("internal error: _diag_w_buf is not initialized")
                    w_diag = w_diag_buf[:j_count]
                    t0 = time.perf_counter() if profile is not None else None
                    cp.cuda.runtime.memsetAsync(
                        int(w_diag.data.ptr),
                        0,
                        int(w_diag.size) * int(w_diag.itemsize),
                        int(stream_apply.ptr),
                    )
                    workspace._build_w_diag_from_steps_inplace(
                        x=x,
                        w_out=w_diag,
                        j_start=int(j0),
                        j_count=int(j_count),
                        stream=stream_apply,
                        sync=False,
                        relative_w=True,
                    )
                    if profile is not None and t0 is not None:
                        stream_apply.synchronize()
                        profile["diag_w_build_s"] = profile.get("diag_w_build_s", 0.0) + (
                            time.perf_counter() - t0
                        )
                    t0 = time.perf_counter() if profile is not None else None
                    workspace._offdiag_gemm_ws.gemm_w_eri_mat_inplace_device(
                        w_diag,
                        eri_mat_use,
                        g_out=g_diag,
                        dtype=workspace._dtype,
                        half=0.5,
                        stream=stream_apply,
                        sync=False,
                    )
                    if profile is not None and t0 is not None:
                        stream_apply.synchronize()
                        profile["diag_gemm_s"] = profile.get("diag_gemm_s", 0.0) + (time.perf_counter() - t0)
                else:
                    occ_d = workspace._occ_buf[:j_count]
                    t0 = time.perf_counter() if profile is not None else None
                    build_occ_block_from_steps_inplace_device_fn(
                        workspace.state_dev,
                        j_start=int(j0),
                        j_count=j_count,
                        occ_out=occ_d,
                        threads=256,
                        stream=stream_apply,
                        sync=False,
                    )
                    if profile is not None and t0 is not None:
                        stream_apply.synchronize()
                        profile["diag_occ_s"] = profile.get("diag_occ_s", 0.0) + (time.perf_counter() - t0)
                    # g_diag[j,pq] = 0.5 * sum_r occ[j,r] * eri[pq, rr]
                    t0 = time.perf_counter() if profile is not None else None
                    with stream_apply:
                        if workspace._dtype == cp.float64:
                            occ_use = occ_d
                        elif workspace._occ_buf_dtype is not None:
                            occ_use = workspace._occ_buf_dtype[:j_count]
                            cp.copyto(occ_use, occ_d)
                        else:
                            occ_use = occ_d.astype(workspace._dtype)
                        cp.dot(occ_use, workspace._eri_diag_t, out=g_diag)  # type: ignore[arg-type]
                        g_diag *= 0.5
                    if profile is not None and t0 is not None:
                        stream_apply.synchronize()
                        profile["diag_gemm_s"] = profile.get("diag_gemm_s", 0.0) + (time.perf_counter() - t0)

            t0 = time.perf_counter() if profile is not None else None
            task_scale_diag = None
            if not use_diag_w_gemm_fallback:
                with stream_apply:
                    task_scale_diag = cp.take(x, j_d, out=workspace._task_scale_j[:j_count])
            apply_g_flat_scatter_atomic_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                j_d,
                g_diag,
                task_scale=task_scale_diag,
                epq_table=epq_table_j,
                apply_mode=str(workspace.apply_mode),
                y=y,
                overflow=workspace.overflow_apply,
                threads=int(workspace.threads_apply),
                zero_y=False,
                stream=stream_apply,
                sync=bool(tile_sync_apply),
                check_overflow=bool(check_overflow_apply_tile),
                dtype=workspace._dtype,
                use_kahan=bool(workspace.kahan_compensation),
            )
            if profile is not None and t0 is not None:
                stream_apply.synchronize()
                profile["diag_apply_s"] = profile.get("diag_apply_s", 0.0) + (time.perf_counter() - t0)

        # Build task arrays for all (r,s) pairs.
        # Kernel 2B+2.5: build CSR into reusable buffers (may retry with larger capacity).
        cached_tile = workspace._csr_tile_cache.get(int(j0)) if workspace.cache_csr_tiles else None
        if cached_tile is not None:
            row_j_d, row_k_d, indptr_d, indices_d, data_d, nrows, nnz = cached_tile
            increment_profile_counter(profile=profile, key="csr_dev_cache_hits", delta=1.0)
        else:
            cached_single = workspace._csr_single_tile_cache
            if (
                cached_single is not None
                and int(workspace.j_tile) >= int(workspace.ncsf)
                and int(j0) == 0
                and int(j_count) == int(workspace.ncsf)
            ):
                row_j_d, row_k_d, indptr_d, indices_d, data_d, nrows, nnz = cached_single
                increment_profile_counter(profile=profile, key="csr_dev_cache_hits", delta=1.0)
            else:
                host_cached = None
                if bool(getattr(workspace, "csr_host_cache_enabled", False)):
                    host_cached = workspace._csr_host_cache_load_tile(j0=int(j0), stream=stream, profile=profile)
                if host_cached is not None:
                    row_j_d, row_k_d, indptr_d, indices_d, data_d, nrows, nnz = host_cached
                else:
                    build_from_prefiltered_tasks = False
                    pref_task_csf_d = None
                    pref_task_p_d = None
                    pref_task_q_d = None
                    if use_prefilter_trivial:
                        t_pref0 = time.perf_counter() if profile is not None else None
                        pref_host = workspace._prefilter_nontrivial_tasks_host(j0=int(j0), j_count=int(j_count))
                        if profile is not None and t_pref0 is not None:
                            profile["csr_prefilter_s"] = profile.get("csr_prefilter_s", 0.0) + (
                                time.perf_counter() - t_pref0
                            )
                        if pref_host is not None:
                            task_csf_h, task_p_h, task_q_h, total_tasks_h, kept_tasks_h = pref_host
                            _pref_stats = stamp_csr_prefilter_tile_profile(
                                profile=profile,
                                total_tasks=int(total_tasks_h),
                                kept_tasks=int(kept_tasks_h),
                            )
                            if bool(_pref_stats["is_zero_tile"]):
                                continue
                            with stream_build:
                                pref_task_csf_d = cp.asarray(task_csf_h, dtype=cp.int32)
                                pref_task_p_d = cp.asarray(task_p_h, dtype=cp.int32)
                                pref_task_q_d = cp.asarray(task_q_h, dtype=cp.int32)
                            build_from_prefiltered_tasks = True
                            increment_profile_counter(
                                profile=profile,
                                key="csr_prefilter_dispatch_tiles",
                                delta=1.0,
                            )

                    if workspace.cache_csr_tiles and int(workspace.j_tile) < int(workspace.ncsf):
                        cap = int(workspace._tile_csr_capacity) if workspace._tile_csr_capacity > 0 else 0
                        if cap <= 0:
                            n_pairs = int(workspace._rs_n_pairs)
                            cap = int(max(1.0, float(workspace.csr_capacity_mult)) * float(int(j_count) * int(n_pairs)))
                            row_j_buf = cp.empty((cap,), dtype=cp.int32)
                            row_k_buf = cp.empty((cap,), dtype=cp.int32)
                            indptr_buf = cp.empty((cap + 1,), dtype=cp.int64)
                            indices_buf = cp.empty((cap,), dtype=cp.int32)
                            data_buf = cp.empty((cap,), dtype=workspace._csr_data_dtype)
                            overflow_buf = cp.empty((1,), dtype=cp.int32)
                        else:
                            row_j_buf = workspace._tile_csr_row_j
                            row_k_buf = workspace._tile_csr_row_k
                            indptr_buf = workspace._tile_csr_indptr
                            indices_buf = workspace._tile_csr_indices
                            data_buf = workspace._tile_csr_data
                            overflow_buf = workspace._tile_csr_overflow
                    else:
                        if tile_slot is not None:
                            cap = int(tile_slot["cap"])
                            row_j_buf = tile_slot["row_j"]
                            row_k_buf = tile_slot["row_k"]
                            indptr_buf = tile_slot["indptr"]
                            indices_buf = tile_slot["indices"]
                            data_buf = tile_slot["data"]
                            overflow_buf = tile_slot["overflow"]
                        else:
                            cap = int(workspace._csr_capacity)
                            row_j_buf = workspace._csr_row_j
                            row_k_buf = workspace._csr_row_k
                            indptr_buf = workspace._csr_indptr
                            indices_buf = workspace._csr_indices
                            data_buf = workspace._csr_data
                            overflow_buf = workspace._csr_overflow

                    last_err = None
                    for _ in range(3):
                        try:
                            t0 = time.perf_counter() if profile is not None else None
                            tile_profile = {} if (profile is not None and (not build_from_prefiltered_tasks)) else None
                            ws = tile_slot.get("ws") if tile_slot is not None else workspace._k25_ws
                            if build_from_prefiltered_tasks:
                                if pref_task_csf_d is None or pref_task_p_d is None or pref_task_q_d is None:
                                    raise RuntimeError("internal error: prefiltered task arrays are not initialized")
                                if ws is not None:
                                    if tile_slot is None and int(getattr(ws, "max_nnz_in", 0)) < int(cap):
                                        workspace._ensure_kernel25_workspace(max_nnz_in=cap)
                                        ws = workspace._k25_ws
                                    if ws is None:
                                        raise RuntimeError("Kernel25Workspace is unavailable")
                                    nrows, nnz, _nnz_in = ws.build_from_tasks_deterministic_inplace_device(
                                        workspace.drt_dev,
                                        workspace.state_dev,
                                        pref_task_csf_d,
                                        pref_task_p_d,
                                        pref_task_q_d,
                                        row_j_buf,
                                        row_k_buf,
                                        indptr_buf,
                                        indices_buf,
                                        data_buf,
                                        overflow_buf,
                                        int(workspace.threads_enum),
                                        bool(workspace.coalesce),
                                        int(stream_build.ptr),
                                        True,
                                        bool(check_overflow_tile),
                                    )
                                    nrows = int(nrows)
                                    nnz = int(nnz)
                                    _overflow_csr = overflow_buf
                                    row_j_d = row_j_buf[:nrows]
                                    row_k_d = row_k_buf[:nrows]
                                    indptr_d = indptr_buf[: nrows + 1]
                                    indices_d = indices_buf[:nnz]
                                    data_d = data_buf[:nnz]
                                else:
                                    (
                                        row_j_d,
                                        row_k_d,
                                        indptr_d,
                                        indices_d,
                                        data_d,
                                        _overflow_csr,
                                        nrows,
                                        nnz,
                                        _nnz_in,
                                    ) = kernel25_build_csr_from_tasks_deterministic_inplace_device_fn(
                                        workspace.drt,
                                        workspace.drt_dev,
                                        workspace.state_dev,
                                        pref_task_csf_d,
                                        pref_task_p_d,
                                        pref_task_q_d,
                                        capacity=cap,
                                        row_j=row_j_buf,
                                        row_k=row_k_buf,
                                        indptr=indptr_buf,
                                        indices=indices_buf,
                                        data=data_buf,
                                        overflow=overflow_buf,
                                        threads=int(workspace.threads_enum),
                                        coalesce=bool(workspace.coalesce),
                                        stream=stream_build,
                                        sync=True,
                                        check_overflow=bool(check_overflow_tile),
                                    )
                            else:
                                if ws is not None:
                                    if tile_slot is None and int(getattr(ws, "max_nnz_in", 0)) < int(cap):
                                        workspace._ensure_kernel25_workspace(max_nnz_in=cap)
                                        ws = workspace._k25_ws
                                    if ws is None:
                                        raise RuntimeError("Kernel25Workspace is unavailable")

                                    nrows, nnz, _nnz_in = ws.build_from_jrs_allpairs_deterministic_inplace_device(
                                        workspace.drt_dev,
                                        workspace.state_dev,
                                        int(j0),
                                        int(j_count),
                                        row_j_buf,
                                        row_k_buf,
                                        indptr_buf,
                                        indices_buf,
                                        data_buf,
                                        overflow_buf,
                                        int(workspace.threads_enum),
                                        bool(workspace.coalesce),
                                        int(stream_build.ptr),
                                        True,
                                        bool(check_overflow_tile),
                                        int(check_overflow_mode_tile),
                                        bool(workspace.fuse_count_write),
                                        tile_profile,
                                    )
                                    nrows = int(nrows)
                                    nnz = int(nnz)
                                    _overflow_csr = overflow_buf
                                    row_j_d = row_j_buf[:nrows]
                                    row_k_d = row_k_buf[:nrows]
                                    indptr_d = indptr_buf[: nrows + 1]
                                    indices_d = indices_buf[:nnz]
                                    data_d = data_buf[:nnz]
                                else:
                                    (
                                        row_j_d,
                                        row_k_d,
                                        indptr_d,
                                        indices_d,
                                        data_d,
                                        _overflow_csr,
                                        nrows,
                                        nnz,
                                        _nnz_in,
                                    ) = kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn(
                                        workspace.drt,
                                        workspace.drt_dev,
                                        workspace.state_dev,
                                        int(j0),
                                        int(j_count),
                                        capacity=cap,
                                        row_j=row_j_buf,
                                        row_k=row_k_buf,
                                        indptr=indptr_buf,
                                        indices=indices_buf,
                                        data=data_buf,
                                        overflow=overflow_buf,
                                        threads=int(workspace.threads_enum),
                                        coalesce=bool(workspace.coalesce),
                                        stream=stream_build,
                                        sync=True,
                                        check_overflow=bool(check_overflow_tile),
                                    )
                            if profile is not None and t0 is not None:
                                profile["csr_build_s"] = profile.get("csr_build_s", 0.0) + (time.perf_counter() - t0)
                                if tile_profile is not None and workspace._k25_ws is not None:
                                    stage_map = (
                                        ("count_ms", "csr_k25_count_s"),
                                        ("prefix_sum_ms", "csr_k25_prefix_sum_s"),
                                        ("write_ms", "csr_k25_write_s"),
                                        ("pack_ms", "csr_k25_pack_s"),
                                        ("sort_ms", "csr_k25_sort_s"),
                                        ("reduce_ms", "csr_k25_reduce_s"),
                                        ("rle_ms", "csr_k25_rle_s"),
                                        ("indptr_ms", "csr_k25_indptr_s"),
                                        ("unpack_ms", "csr_k25_unpack_s"),
                                        ("sync_overhead_ms", "csr_k25_sync_overhead_s"),
                                        ("bucket_ms", "csr_k25_bucket_s"),
                                    )
                                    for src_key, dst_key in stage_map:
                                        ms_val = float(tile_profile.get(src_key, 0.0))
                                        if ms_val != 0.0:
                                            profile[dst_key] = profile.get(dst_key, 0.0) + (ms_val * 1.0e-3)
                                    count_map = (
                                        ("nnz_in", "csr_k25_nnz_in"),
                                        ("nnz_out", "csr_k25_nnz_out"),
                                        ("nrows", "csr_k25_nrows"),
                                    )
                                    for src_key, dst_key in count_map:
                                        v = tile_profile.get(src_key)
                                        if v is not None:
                                            profile[dst_key] = profile.get(dst_key, 0.0) + float(v)
                            break
                        except RuntimeError as e:
                            last_err = e
                            err_s = str(e).lower()
                            if (
                                "exceeds output buffer capacity" in err_s
                                or "output buffer capacity" in err_s
                                or "capacity exceeds workspace max_nnz_in" in err_s
                            ):
                                cap *= 2
                                if tile_slot is not None:
                                    workspace._grow_csr_pipeline_slot(int(tile_idx) % int(pipeline_nslots), cap)
                                    cap = int(tile_slot["cap"])
                                    row_j_buf = tile_slot["row_j"]
                                    row_k_buf = tile_slot["row_k"]
                                    indptr_buf = tile_slot["indptr"]
                                    indices_buf = tile_slot["indices"]
                                    data_buf = tile_slot["data"]
                                    overflow_buf = tile_slot["overflow"]
                                    continue
                                workspace._ensure_kernel25_workspace(max_nnz_in=cap)
                                if workspace.cache_csr_tiles and int(workspace.j_tile) < int(workspace.ncsf):
                                    workspace._tile_csr_row_j = cp.empty((cap,), dtype=cp.int32)
                                    workspace._tile_csr_row_k = cp.empty((cap,), dtype=cp.int32)
                                    workspace._tile_csr_indptr = cp.empty((cap + 1,), dtype=cp.int64)
                                    workspace._tile_csr_indices = cp.empty((cap,), dtype=cp.int32)
                                    workspace._tile_csr_data = cp.empty((cap,), dtype=workspace._csr_data_dtype)
                                    workspace._tile_csr_overflow = cp.empty((1,), dtype=cp.int32)
                                    workspace._tile_csr_capacity = cap
                                    row_j_buf = workspace._tile_csr_row_j
                                    row_k_buf = workspace._tile_csr_row_k
                                    indptr_buf = workspace._tile_csr_indptr
                                    indices_buf = workspace._tile_csr_indices
                                    data_buf = workspace._tile_csr_data
                                    overflow_buf = workspace._tile_csr_overflow
                                    continue
                                workspace._alloc_csr_buffers(capacity=cap)
                                row_j_buf = workspace._csr_row_j
                                row_k_buf = workspace._csr_row_k
                                indptr_buf = workspace._csr_indptr
                                indices_buf = workspace._csr_indices
                                data_buf = workspace._csr_data
                                overflow_buf = workspace._csr_overflow
                                continue
                            raise
                    else:
                        raise last_err  # type: ignore[misc]

                    if (
                        workspace.cache_csr_tiles
                        and int(workspace.j_tile) < int(workspace.ncsf)
                        and int(indptr_d.size) > 0
                    ):
                        cache_data = data_d
                        if bool(getattr(workspace, "fp32_csr_cache", False)) and data_d.dtype != workspace._dtype:
                            cache_data = data_d.astype(workspace._dtype)
                        else:
                            cache_data = cp.array(cache_data, copy=True)
                        workspace._csr_tile_cache[int(j0)] = (
                            cp.array(row_j_d, copy=True),
                            cp.array(row_k_d, copy=True),
                            cp.array(indptr_d, copy=True),
                            cp.array(indices_d, copy=True),
                            cache_data,
                            int(nrows),
                            int(nnz),
                        )
                    elif int(workspace.j_tile) >= int(workspace.ncsf) and int(j0) == 0 and int(j_count) == int(workspace.ncsf):
                        cache_data = data_d
                        if bool(getattr(workspace, "fp32_csr_cache", False)) and data_d.dtype != workspace._dtype:
                            cache_data = data_d.astype(workspace._dtype)
                        else:
                            cache_data = cp.array(cache_data, copy=True)
                        workspace._csr_single_tile_cache = (
                            cp.array(row_j_d, copy=True),
                            cp.array(row_k_d, copy=True),
                            cp.array(indptr_d, copy=True),
                            cp.array(indices_d, copy=True),
                            cache_data,
                            int(nrows),
                            int(nnz),
                        )
                    elif (
                        bool(getattr(workspace, "csr_host_cache_enabled", False))
                        and int(workspace.j_tile) < int(workspace.ncsf)
                        and int(indptr_d.size) > 0
                    ):
                        workspace._csr_host_cache_store_tile(
                            j0=int(j0),
                            row_j_d=row_j_d,
                            row_k_d=row_k_d,
                            indptr_d=indptr_d,
                            indices_d=indices_d,
                            data_d=data_d,
                            nrows=int(nrows),
                            nnz=int(nnz),
                            stream=stream,
                            profile=profile,
                        )

        if int(nrows) == 0:
            continue

        if w_offdiag is not None:
            # --- Width-aware hybrid dispatch (bucketed) ---
            _bucket_info = None
            _ws_for_bucket = getattr(workspace, "_k25_ws", None)
            if _ws_for_bucket is not None:
                try:
                    _bucket_info = _ws_for_bucket.get_bucket_info()
                except Exception:
                    _bucket_info = None

            if _bucket_info is not None and bool(_bucket_info[0]):
                _bvalid, _boffsets, _bcounts, _bperm_ptr = _bucket_info
                _n_b0 = int(_boffsets[1])         # bucket 0: width 1
                _n_b1 = int(_boffsets[2]) - _n_b0  # bucket 1: width 2-4
                _n_b23 = int(nrows) - int(_boffsets[2])  # buckets 2+3: width 5+
                _has_remaining = (_n_b1 > 0 or _n_b23 > 0)

                t0 = time.perf_counter() if profile is not None else None
                if _n_b0 > 0:
                    ext.kernel4_build_w_from_csr_bucket0_rawptr(
                        int(_bperm_ptr),
                        _n_b0,
                        row_j_d,
                        row_k_d,
                        indptr_d,
                        indices_d,
                        data_d,
                        x,
                        w_offdiag,
                        workspace._overflow_w,
                        int(workspace.drt.ncsf),
                        int(workspace.drt.norb) * int(workspace.drt.norb),
                        int(workspace.threads_w),
                        int(stream_apply.ptr),
                        bool(tile_sync_apply) and (not _has_remaining),
                        bool(check_overflow_apply_tile) and (not _has_remaining),
                    )
                if _has_remaining and epq_table is not None and eri_mat_t is not None:
                    _epq_indptr_d, _epq_indices_d, _epq_pq_d, _epq_data_d = epq_table
                    # Bucket 1 (width 2-4): narrow kernel (no shared memory).
                    if _n_b1 > 0:
                        _perm_b1_ptr = int(_bperm_ptr) + _n_b0 * 4
                        ext.kernel4_apply_csr_narrow_fused_perm_rawptr(
                            _perm_b1_ptr,
                            _n_b1,
                            workspace.state_dev,
                            _epq_indptr_d,
                            _epq_indices_d,
                            _epq_pq_d,
                            _epq_data_d,
                            row_j_d,
                            row_k_d,
                            indptr_d,
                            indices_d,
                            data_d,
                            eri_mat_t,
                            x,
                            y,
                            workspace.overflow_apply,
                            int(workspace.drt.norb) * int(workspace.drt.norb),
                            int(workspace.threads_apply),
                            int(stream_apply.ptr),
                            bool(tile_sync_apply) and (_n_b23 == 0),
                            bool(check_overflow_apply_tile) and (_n_b23 == 0),
                        )
                    # Buckets 2+3 (width 5+): general fused warp kernel.
                    if _n_b23 > 0:
                        _perm_b23_ptr = int(_bperm_ptr) + int(_boffsets[2]) * 4
                        ext.kernel4_apply_csr_fused_perm_rawptr(
                            _perm_b23_ptr,
                            _n_b23,
                            workspace.state_dev,
                            _epq_indptr_d,
                            _epq_indices_d,
                            _epq_pq_d,
                            _epq_data_d,
                            row_j_d,
                            row_k_d,
                            indptr_d,
                            indices_d,
                            data_d,
                            eri_mat_t,
                            x,
                            y,
                            workspace.overflow_apply,
                            int(workspace.drt.norb) * int(workspace.drt.norb),
                            int(workspace.threads_apply),
                            int(stream_apply.ptr),
                            bool(tile_sync_apply),
                            bool(check_overflow_apply_tile),
                        )
                if profile is not None and t0 is not None:
                    stream_apply.synchronize()
                    profile["offdiag_w_build_s"] = profile.get("offdiag_w_build_s", 0.0) + (time.perf_counter() - t0)
            else:
                # Fallback: old unit-nnz path (requires nnz==nrows).
                if int(nnz) != int(nrows):
                    raise RuntimeError(
                        "aggregate_offdiag_k requires nnz==nrows (unit-nnz CSR rows); "
                        "try coalesce=False or disable aggregate_offdiag_k"
                    )
                t0 = time.perf_counter() if profile is not None else None
                kernel4_build_w_from_csr_unitnnz_inplace_device_fn(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    row_j_d,
                    row_k_d,
                    indices_d,
                    data_d,
                    x,
                    w_out=w_offdiag,
                    overflow=workspace._overflow_w,
                    threads=int(workspace.threads_w),
                    stream=stream_apply,
                    sync=bool(tile_sync_apply),
                    check_overflow=bool(check_overflow_apply_tile),
                )
                if profile is not None and t0 is not None:
                    stream_apply.synchronize()
                    profile["offdiag_w_build_s"] = profile.get("offdiag_w_build_s", 0.0) + (time.perf_counter() - t0)

            if use_csr_pipeline and tile_slot is not None and (not tile_sync_apply):
                evt = tile_slot.get("inflight_event")
                if evt is None:
                    evt = cp.cuda.Event(disable_timing=True)
                    tile_slot["inflight_event"] = evt
                evt.record(stream_apply)
            continue

        # Kernel 4: build g from CSR and apply/scatter into y (accumulate into existing y).
        t0 = time.perf_counter() if profile is not None else None
        if use_df:
            if l_full_use is None:  # pragma: no cover
                raise RuntimeError("internal error: l_full_use is not set for DF path")
            kernel4_apply_csr_l_full_device_csr_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                row_j_d,
                row_k_d,
                indptr_d,
                indices_d,
                data_d,
                l_full_use,
                x,
                epq_table=epq_table,
                gdf_workspace=workspace._gdf_ws,
                g_buf=workspace._g_buf,
                task_scale_buf=workspace._task_scale_rows,
                y=y,
                overflow=workspace.overflow_apply,
                max_g_bytes=int(workspace.max_g_bytes),
                threads_g=int(workspace.threads_g),
                threads_apply=int(workspace.threads_apply),
                zero_y=False,
                stream=stream_apply,
                sync=bool(tile_sync_apply),
                check_overflow=bool(check_overflow_apply_tile),
                profile=profile,
            )
        elif epq_table is not None:
            if eri_mat_t is None:
                raise RuntimeError("internal error: eri_mat_t is not initialized for fused epq_table kernel4")
            kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                epq_table,
                row_j_d,
                row_k_d,
                indptr_d,
                indices_d,
                data_d,
                eri_mat_t,
                x,
                row_start=0,
                nrows=int(nrows),
                y=y,
                overflow=workspace.overflow_apply,
                threads=int(workspace.threads_apply),
                zero_y=False,
                stream=stream_apply,
                sync=bool(tile_sync_apply),
                check_overflow=bool(check_overflow_apply_tile),
                use_kahan=bool(workspace.kahan_compensation),
            )
        else:
            kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn(
                workspace.drt,
                workspace.drt_dev,
                workspace.state_dev,
                row_j_d,
                row_k_d,
                indptr_d,
                indices_d,
                data_d,
                eri_mat_use,
                x,
                epq_table=None,
                g_buf=workspace._g_buf,
                task_scale_buf=workspace._task_scale_rows,
                y=y,
                overflow=workspace.overflow_apply,
                max_g_bytes=int(workspace.max_g_bytes),
                threads_g=int(workspace.threads_g),
                threads_apply=int(workspace.threads_apply),
                zero_y=False,
                stream=stream_apply,
                sync=bool(tile_sync_apply),
                check_overflow=bool(check_overflow_apply_tile),
                profile=profile,
            )
        if profile is not None and t0 is not None:
            stream_apply.synchronize()
            profile["kernel4_s"] = profile.get("kernel4_s", 0.0) + (time.perf_counter() - t0)
        if use_csr_pipeline and tile_slot is not None and (not tile_sync_apply):
            evt = tile_slot.get("inflight_event")
            if evt is None:
                evt = cp.cuda.Event(disable_timing=True)
                tile_slot["inflight_event"] = evt
            evt.record(stream_apply)

    if (
        bool(workspace.cache_csr_tiles)
        and int(workspace.j_tile) < int(workspace.ncsf)
        and int(workspace._tile_csr_capacity) > 0
        and workspace._csr_cache_ready()
    ):
        workspace._release_tile_csr_scratch()

    if w_offdiag is not None:
        if not use_df and offdiag_gemm_ws is None:
            raise RuntimeError("internal error: offdiag_gemm_ws is not initialized for aggregate_offdiag_k")
        if use_df and gdf_ws is None:
            raise RuntimeError("internal error: gdf_ws is not initialized for DF aggregate_offdiag_k")

        nrows_block_max = int(getattr(workspace._g_buf, "shape", (0, 0))[0])
        if nrows_block_max <= 0:
            raise RuntimeError("internal error: invalid g_buf block size")

        if profile is not None:
            if use_df:
                profile["offdiag_df_cublas_workspace_bytes"] = float(int(gdf_ws.cublas_workspace_bytes()))
            else:
                profile["offdiag_cublas_workspace_bytes"] = float(int(workspace._offdiag_cublas_workspace_bytes))

        for k0 in range(0, int(workspace.ncsf), int(nrows_block_max)):
            k1 = min(int(workspace.ncsf), int(k0 + int(nrows_block_max)))
            k_count = int(k1 - k0)
            task_csf_block = workspace.task_csf_all[int(k0) : int(k1)]
            w_block = w_offdiag[int(k0) : int(k1)]
            g_block = workspace._g_buf[:k_count]

            t0 = time.perf_counter() if profile is not None else None
            if use_df:
                gdf_ws.gemm_w_l_full_inplace_device(
                    w_block,
                    l_full_use,
                    g_out=g_block,
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
            else:
                offdiag_gemm_ws.gemm_w_eri_mat_inplace_device(
                    w_block,
                    eri_mat_use,
                    g_out=g_block,
                    dtype=workspace._dtype,
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
            if profile is not None and t0 is not None:
                stream.synchronize()
                if use_df:
                    naux = int(l_full_use.shape[1])
                    profile["offdiag_df_gemm_s"] = profile.get("offdiag_df_gemm_s", 0.0) + (time.perf_counter() - t0)
                    profile["offdiag_df_gemm_flops"] = profile.get("offdiag_df_gemm_flops", 0.0) + float(
                        2.0 * float(int(k_count)) * float(int(workspace.nops)) * float(naux)
                        + 2.0 * float(int(k_count)) * float(naux) * float(int(workspace.nops))
                    )
                else:
                    profile["offdiag_gemm_s"] = profile.get("offdiag_gemm_s", 0.0) + (time.perf_counter() - t0)
                    profile["offdiag_gemm_flops"] = profile.get("offdiag_gemm_flops", 0.0) + float(
                        2.0 * float(int(k_count)) * float(int(workspace.nops)) * float(int(workspace.nops))
                    )

            # Apply g_block to y.  Fast path: use EPQ tile cache to avoid DFS walks.
            _use_epq_tile = bool(getattr(workspace, "epq_apply_cache_enabled", False))
            _epq_tile = None
            if _use_epq_tile:
                _epq_tile = workspace._epq_apply_cache_load(k0=k0, stream=stream)

            if _epq_tile is None and _use_epq_tile and epq_table is None:
                t0_build = time.perf_counter() if profile is not None else None
                _tile_raw = build_epq_action_table_tile_device_fn(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    j_start=k0,
                    j_count=k_count,
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                    global_indptr=False,
                    dtype=workspace._dtype,
                )
                _tile_indptr, _tile_indices, _tile_pq_ids, _tile_data = _tile_raw
                _tile_nnz = int(_tile_indices.shape[0])
                workspace._epq_apply_cache_store(
                    k0=k0, indptr_d=_tile_indptr, indices_d=_tile_indices,
                    pq_ids_d=_tile_pq_ids, data_d=_tile_data,
                    j_count=k_count, nnz=_tile_nnz, stream=stream,
                )
                stream.synchronize()  # ensure D2H completes before device buffers reused
                _epq_tile = (_tile_indptr, _tile_indices, _tile_pq_ids, _tile_data)
                if profile is not None and t0_build is not None:
                    profile["epq_apply_build_s"] = profile.get("epq_apply_build_s", 0.0) + (
                        time.perf_counter() - t0_build
                    )

            t0 = time.perf_counter() if profile is not None else None
            if _epq_tile is not None:
                _ep_indptr, _ep_indices, _ep_pq_ids, _ep_data = _epq_tile
                apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    local_indptr=_ep_indptr,
                    indices=_ep_indices,
                    pq_ids=_ep_pq_ids,
                    epq_data=_ep_data,
                    task_g=g_block,
                    task_scale=None,
                    j_start=k0,
                    j_count=k_count,
                    y=y,
                    overflow=workspace.overflow_apply,
                    threads=int(workspace.threads_apply),
                    zero_y=False,
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    dtype=workspace._dtype,
                    use_kahan=bool(workspace.kahan_compensation),
                )
            else:
                apply_g_flat_scatter_atomic_inplace_device_fn(
                    workspace.drt,
                    workspace.drt_dev,
                    workspace.state_dev,
                    task_csf_block,
                    g_block,
                    task_scale=None,
                    epq_table=epq_table,
                    apply_mode=str(workspace.apply_mode),
                    y=y,
                    overflow=workspace.overflow_apply,
                    threads=int(workspace.threads_apply),
                    zero_y=False,
                    stream=stream,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    dtype=workspace._dtype,
                    use_kahan=bool(workspace.kahan_compensation),
                )
            if profile is not None and t0 is not None:
                stream.synchronize()
                profile["offdiag_apply_s"] = profile.get("offdiag_apply_s", 0.0) + (time.perf_counter() - t0)

    if use_csr_pipeline and pipeline_apply_stream is not None and bool(sync):
        pipeline_apply_stream.synchronize()

    finalize_hop_profile(
        profile=profile,
        stream=stream,
        t_total0=t_total0,
        csr_host_cache_hits=int(workspace._csr_host_cache_hits),
        csr_host_cache_misses=int(workspace._csr_host_cache_misses),
        epq_apply_cache_hits=int(workspace._epq_apply_cache_hits),
        epq_apply_cache_misses=int(workspace._epq_apply_cache_misses),
        epq_apply_cache_bytes=int(workspace._epq_apply_cache_bytes),
    )
    return y


def run_blocked_hop_path(
    workspace: Any,
    *,
    cp: Any,
    ext: Any,
    x: Any,
    y: Any,
    eri_mat_use: Any,
    eri_mat_t: Any,
    l_full_use: Any,
    direct_op_use: Any,
    use_df: bool,
    use_direct: bool,
    use_aggregate_offdiag: bool,
    h_eff_flat: Any,
    stream: Any,
    sync: bool,
    check_overflow: bool,
    profile: dict[str, float] | None,
    use_fused_hop: bool,
    use_epq_streaming: bool,
    use_epq_streaming_tiles: bool,
    epq_stream_panic_requested: bool,
    epq_stream_panic_active: bool,
    path_mode: str,
    epq_table: Any,
    # One-body function injections
    build_epq_action_table_tile_device_fn: Any,
    apply_g_flat_scatter_atomic_inplace_device_fn: Any,
    apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn: Any,
    # EPQ-blocked path function injections
    build_epq_action_table_transpose_device_fn: Any,
    build_w_from_epq_transpose_range_inplace_device_fn: Any,
    apply_g_flat_gather_epq_transpose_range_inplace_device_fn: Any,
    sym_pair_pack_inplace_device_fn: Any,
    sym_pair_unpack_inplace_device_fn: Any,
    has_sym_pair_fused_kernels_fn: Any,
    # Per-tile CSR path function injections
    build_w_from_epq_table_inplace_device_fn: Any,
    build_occ_block_from_steps_inplace_device_fn: Any,
    kernel4_build_w_from_csr_unitnnz_inplace_device_fn: Any,
    kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn: Any,
    kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn: Any,
    kernel4_apply_csr_l_full_device_csr_inplace_device_fn: Any,
    kernel25_build_csr_from_tasks_deterministic_inplace_device_fn: Any,
    kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn: Any,
    # Shared
    Kernel3BuildGDFWorkspace_cls: Any,
) -> Any:
    """Run the non-fused blocked hop path.

    Entry point for all non-fused hop execution: one-body phase,
    EPQ-blocked aggregate guard, then dispatch to either the
    EPQ-blocked aggregate path or the per-tile CSR path.
    """
    state = _prepare_blocked_hop_state(
        workspace,
        cp=cp,
        x=x,
        y=y,
        h_eff_flat=h_eff_flat,
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        profile=profile,
        use_fused_hop=bool(use_fused_hop),
        use_aggregate_offdiag=bool(use_aggregate_offdiag),
        use_epq_streaming=bool(use_epq_streaming),
        epq_stream_panic_requested=bool(epq_stream_panic_requested),
        epq_stream_panic_active=bool(epq_stream_panic_active),
        path_mode=str(path_mode),
        epq_table=epq_table,
        build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device_fn,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn,
    )

    if state["use_epq_agg_blocked"]:
        return run_epq_blocked_aggregate_path(
            workspace,
            cp=cp,
            ext=ext,
            x=x,
            y=y,
            eri_mat_use=eri_mat_use,
            l_full_use=l_full_use,
            direct_op_use=direct_op_use,
            use_df=bool(use_df),
            use_direct=bool(use_direct),
            h_eff_flat=h_eff_flat,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            t_total0=state["t_total0"],
            one_body_event=state["one_body_event"],
            epq_table=epq_table,
            build_epq_action_table_transpose_device_fn=build_epq_action_table_transpose_device_fn,
            build_w_from_epq_transpose_range_inplace_device_fn=build_w_from_epq_transpose_range_inplace_device_fn,
            apply_g_flat_gather_epq_transpose_range_inplace_device_fn=apply_g_flat_gather_epq_transpose_range_inplace_device_fn,
            sym_pair_pack_inplace_device_fn=sym_pair_pack_inplace_device_fn,
            sym_pair_unpack_inplace_device_fn=sym_pair_unpack_inplace_device_fn,
            has_sym_pair_fused_kernels_fn=has_sym_pair_fused_kernels_fn,
            Kernel3BuildGDFWorkspace_cls=Kernel3BuildGDFWorkspace_cls,
        )

    return run_per_tile_csr_path(
        workspace,
        cp=cp,
        ext=ext,
        x=x,
        y=y,
        eri_mat_use=eri_mat_use,
        eri_mat_t=eri_mat_t,
        l_full_use=l_full_use,
        direct_op_use=direct_op_use,
        use_df=bool(use_df),
        use_direct=bool(use_direct),
        use_aggregate_offdiag=bool(use_aggregate_offdiag),
        h_eff_flat=h_eff_flat,
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        profile=profile,
        t_total0=state["t_total0"],
        one_body_event=state["one_body_event"],
        use_fused_hop=bool(use_fused_hop),
        use_epq_streaming=bool(use_epq_streaming),
        use_epq_streaming_tiles=bool(use_epq_streaming_tiles),
        epq_stream_panic_active=bool(epq_stream_panic_active),
        epq_table=epq_table,
        build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device_fn,
        apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device_fn,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn,
        build_w_from_epq_table_inplace_device_fn=build_w_from_epq_table_inplace_device_fn,
        build_occ_block_from_steps_inplace_device_fn=build_occ_block_from_steps_inplace_device_fn,
        kernel4_build_w_from_csr_unitnnz_inplace_device_fn=kernel4_build_w_from_csr_unitnnz_inplace_device_fn,
        kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn=kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn,
        kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn=kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn,
        kernel4_apply_csr_l_full_device_csr_inplace_device_fn=kernel4_apply_csr_l_full_device_csr_inplace_device_fn,
        kernel25_build_csr_from_tasks_deterministic_inplace_device_fn=kernel25_build_csr_from_tasks_deterministic_inplace_device_fn,
        kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn=kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn,
        Kernel3BuildGDFWorkspace_cls=Kernel3BuildGDFWorkspace_cls,
    )
