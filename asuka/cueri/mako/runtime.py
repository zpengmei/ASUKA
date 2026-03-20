"""Mako J/K pipeline dispatch (Phase 7).

Replicates the slab/group loop from ``direct_jk.py`` but dispatches
ERI tile evaluation to Mako GEMM kernels for supported classes.
Unsupported classes fall back to the legacy Rys ``run_kernel_batch_spd``.

Both paths feed into the same ``contract_jk_tiles_ordered_inplace_device``
contraction kernel, so J/K output is identical.
"""
from __future__ import annotations

import logging
import time
import os
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def direct_JK_mako(
    ctx,
    D,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
    stats: dict | None = None,
    tile_dtype: str | None = None,
    mixed_precision: bool | None = None,
    f32_accum: bool | None = None,
    schedule: Any | None = None,
) -> tuple[Any, Any]:
    """Build J and K using the Mako MMD/GEMM engine with legacy fallback.

    Implements the full slab/group dispatch loop from ``direct_JK()``
    but routes supported ERI classes through Mako GEMM kernels.
    """
    if not want_J and not want_K:
        return None, None

    import cupy as cp  # noqa: PLC0415

    from asuka.kernels import cueri as cueri_kernels  # noqa: PLC0415
    from asuka.kernels.cueri_mako import require_ext as require_mako_ext  # noqa: PLC0415
    from asuka.cueri.eri_dispatch import KernelBatch, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList, decode_eri_class_id  # noqa: PLC0415
    from asuka.cueri.mako.tile_evaluator import mako_evaluate_tiles  # noqa: PLC0415

    _ext = cueri_kernels.require_ext()
    _mako_ext = require_mako_ext()

    # Precision defaults
    if tile_dtype is None:
        tile_dtype = "float32" if os.environ.get("ASUKA_ERI_TILE_F32", "") == "1" else "float64"
    if mixed_precision is None:
        mixed_precision = os.environ.get("ASUKA_ERI_MIXED_PRECISION", "1") != "0"
    if f32_accum is None:
        f32_accum = os.environ.get("ASUKA_ERI_F32_ACCUM", "") == "1"

    nao = ctx.nao
    eval_threads = int(getattr(ctx, "eval_threads", ctx.threads))
    contract_threads = int(getattr(ctx, "contract_threads", ctx.threads))

    D_gpu = cp.asarray(D, dtype=cp.float64)
    if D_gpu.ndim != 2 or D_gpu.shape != (nao, nao):
        raise ValueError(f"D must be ({nao}, {nao}), got {tuple(D_gpu.shape)}")

    # Schedule metadata for profiling
    quant_mode = os.environ.get("ASUKA_MAKO_QUANT_MODE", "fp64")
    if quant_mode == "fp64" and schedule is not None:
        if schedule.force_fp64:
            quant_mode = "fp64"
        elif hasattr(schedule, "tau_quant") and schedule.tau_quant is not None:
            quant_mode = "tf32"

    if profile is not None and schedule is not None:
        mako_prof = profile.setdefault("mako", {})
        mako_prof["schedule_label"] = schedule.label
        mako_prof["force_fp64"] = schedule.force_fp64
        mako_prof["rebase"] = schedule.rebase_this_iter

    # Density prescreening
    eps_density_env = os.environ.get("ASUKA_DIRECT_JK_EPS_DENSITY", "")
    eps_density = float(eps_density_env) if eps_density_env.strip() else max(ctx.eps_schwarz, 1e-10)
    Q_sp = getattr(ctx, "Q_sp", None)
    shell_l_dev = getattr(ctx, "shell_l_dev", None)
    D_sp_dev = None
    if Q_sp is not None and shell_l_dev is not None and eps_density > 0:
        from asuka.hf.direct_jk import _compute_D_sp_max  # noqa: PLC0415
        D_sp_dev = _compute_D_sp_max(
            D_gpu, ctx.sp_A_dev, ctx.sp_B_dev, ctx.shell_ao_start_dev,
            shell_l_dev, ctx.nsp, cp,
        )

    compute_stream = cp.cuda.get_current_stream()
    stream_ptr = int(compute_stream.ptr)
    t0 = time.perf_counter()
    n_kernel_calls = 0
    n_mako_calls = 0

    # Multi-buffer J/K accumulation (matches legacy _choose_n_bufs pattern)
    # Distributes atomicAdd contention across multiple buffers.
    from asuka.hf.direct_jk import _choose_n_bufs  # noqa: PLC0415
    n_bufs = _choose_n_bufs(nao)

    D_flat = cp.ascontiguousarray(D_gpu.ravel())
    J_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_J else None
    K_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_K else None

    if n_bufs > 1:
        J_bufs = cp.zeros((n_bufs, nao * nao), dtype=cp.float64) if want_J else None
        K_bufs = cp.zeros((n_bufs, nao * nao), dtype=cp.float64) if want_K else None
    else:
        J_bufs = None
        K_bufs = None

    sp_pair_start_fused = ctx.dsp.sp_pair_start
    if int(getattr(sp_pair_start_fused, "shape", (0,))[0]) == int(getattr(ctx.dsp.sp_npair, "shape", (0,))[0]) + 1:
        sp_pair_start_fused = sp_pair_start_fused[:-1]

    if stats is not None:
        stats["direct_jk_ntasks"] = int(ctx.ntasks)
        stats.setdefault("n_eval_calls", 0)
        stats.setdefault("n_contract_calls", 0)
        stats.setdefault("n_mako_calls", 0)
        stats.setdefault("classes", {})

    mako_supported = ctx.mako_supported_classes

    # ----- Slab/group loop (mirrors direct_jk.py lines 1833-2194) -----
    for slab_i, slab in enumerate(ctx.slabs):
        slab_i = int(slab_i)

        # Get task arrays for this slab
        if slab.gpu_resident:
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
        else:
            ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
            cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))

        slab_plan = ctx.plans[slab_i]
        for g, gp in enumerate(slab_plan):
            orig_cid = int(gp.orig_cid)
            cls = str(gp.class_label)
            klabel = str(gp.kernel_label).lower()
            j0, j1 = int(gp.j0), int(gp.j1)
            if j1 <= j0:
                continue

            kernel_cid = int(gp.kernel_cid)
            transpose = bool(gp.transpose)
            nA = int(gp.nA)
            nB = int(gp.nB)
            nC = int(gp.nC)
            nD = int(gp.nD)
            chunk_ntasks = int(gp.chunk_ntasks)

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]
            class_ntasks = j1 - j0

            # Density prescreening (CSAM)
            if D_sp_dev is not None and Q_sp is not None and class_ntasks > 0:
                _q_ab = Q_sp[kernel_spAB_full]
                _q_cd = Q_sp[kernel_spCD_full]
                _d_ab = D_sp_dev[kernel_spAB_full]
                _d_cd = D_sp_dev[kernel_spCD_full]
                _screen_val = _q_ab * _q_cd * cp.maximum(_d_ab, _d_cd)
                _mask = _screen_val >= eps_density
                n_survive = int(_mask.sum())
                if n_survive < class_ntasks:
                    if n_survive == 0:
                        continue
                    _idx = cp.nonzero(_mask)[0]
                    kernel_spAB_full = kernel_spAB_full[_idx]
                    kernel_spCD_full = kernel_spCD_full[_idx]
                    class_ntasks = n_survive

            # Decode kernel class angular momenta
            la, lb, lc, ld = decode_eri_class_id(kernel_cid)

            # ------ Dispatch: Mako GEMM or legacy Rys ------
            # Use Mako GEMM when ncomp >= 100 (roughly total_am >= 6).
            # Below that, the legacy 1-thread-per-task Rys kernel is faster
            # because the GEMM's 1-block-per-task overhead exceeds the
            # per-task compute savings.  Benchmark-derived threshold:
            #   ncomp >= 100: mako wins 3-17x (f/g shells, high-AM d-shell)
            #   ncomp < 100:  legacy wins (ssss, psss, ppss, ppps, pppp, etc.)
            ncomp_class = nA * nB * nC * nD
            # The mako GEMM kernel works for any (la,lb,lc,ld) — it takes
            # angular momenta explicitly, not class IDs.  Dispatch based on
            # ncomp threshold only (the supported_classes check is overly
            # restrictive since it only lists one canonical AM ordering).
            use_mako = ncomp_class >= 100

            # Resolve J/K target buffers for contraction
            _J_contract = J_bufs.ravel() if (n_bufs > 1 and J_bufs is not None) else J_flat
            _K_contract = K_bufs.ravel() if (n_bufs > 1 and K_bufs is not None) else K_flat

            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue

                chunk_spAB = kernel_spAB_full[c0:c1]
                chunk_spCD = kernel_spCD_full[c0:c1]

                if use_mako:
                    # Mako GEMM tile evaluation.
                    # The kernel uses (la,lb,lc,ld) from kernel_cid, so tiles
                    # are in kernel-class order regardless of transpose.
                    mako_la, mako_lb, mako_lc, mako_ld = la, lb, lc, ld
                    tiles = mako_evaluate_tiles(
                        _mako_ext,
                        chunk_spAB, chunk_spCD,
                        mako_la, mako_lb, mako_lc, mako_ld,
                        ctx.dsp, ctx.dbasis, ctx.pair_tables,
                        threads=eval_threads,
                        stream_ptr=stream_ptr,
                        quant_mode=quant_mode,
                    )
                    n_mako_calls += 1
                else:
                    sub_batch = KernelBatch(
                        task_idx=np.empty(0, dtype=np.int32),
                        kernel_tasks=TaskList(
                            task_spAB=chunk_spAB,
                            task_spCD=chunk_spCD,
                        ),
                        kernel_class_id=np.int32(kernel_cid),
                        transpose=transpose,
                    )
                    tiles = run_kernel_batch_spd(
                        sub_batch,
                        dbasis=ctx.dbasis,
                        dsp=ctx.dsp,
                        pt=ctx.pair_tables,
                        stream=None,
                        threads=eval_threads,
                        mode="auto",
                        profile=profile,
                        skip_transpose=True,
                        tile_dtype=tile_dtype,
                        mixed_precision=mixed_precision,
                        f32_accum=f32_accum,
                    )

                n_kernel_calls += 1
                if stats is not None:
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1

                # ------ Contract tiles to J/K ------
                contract_fn = _ext.contract_jk_tiles_ordered_inplace_device

                if use_mako:
                    # Mako tiles are always in kernel-class order (la,lb,lc,ld).
                    # The spAB/spCD arrays are already swapped for transpose,
                    # so pass kernel-class component counts directly.
                    _nA_c = _ncart(la)
                    _nB_c = _ncart(lb)
                    _nC_c = _ncart(lc)
                    _nD_c = _ncart(ld)
                    contract_fn(
                        chunk_spAB, chunk_spCD,
                        ctx.sp_A_dev, ctx.sp_B_dev,
                        ctx.shell_ao_start_dev, int(nao),
                        int(_nA_c), int(_nB_c), int(_nC_c), int(_nD_c),
                        cp.ascontiguousarray(tiles).ravel(),
                        D_flat,
                        _J_contract, _K_contract,
                        int(contract_threads), int(stream_ptr),
                        False,
                        n_bufs=int(n_bufs),
                    )
                elif transpose:
                    contract_fn(
                        kernel_spAB_full[c0:c1],
                        kernel_spCD_full[c0:c1],
                        ctx.sp_A_dev, ctx.sp_B_dev,
                        ctx.shell_ao_start_dev, int(nao),
                        int(nC), int(nD), int(nA), int(nB),
                        tiles.ravel(),
                        D_flat,
                        _J_contract, _K_contract,
                        int(contract_threads), int(stream_ptr),
                        False,
                        n_bufs=int(n_bufs),
                    )
                else:
                    contract_fn(
                        kernel_spAB_full[c0:c1],
                        kernel_spCD_full[c0:c1],
                        ctx.sp_A_dev, ctx.sp_B_dev,
                        ctx.shell_ao_start_dev, int(nao),
                        int(nA), int(nB), int(nC), int(nD),
                        tiles.ravel(),
                        D_flat,
                        _J_contract, _K_contract,
                        int(contract_threads), int(stream_ptr),
                        False,
                        n_bufs=int(n_bufs),
                    )

                if stats is not None:
                    stats["n_contract_calls"] = int(stats.get("n_contract_calls", 0)) + 1

    # Reduce multi-buffer contributions
    if J_bufs is not None and J_flat is not None:
        J_flat += J_bufs.sum(axis=0)
    if K_bufs is not None and K_flat is not None:
        K_flat += K_bufs.sum(axis=0)

    # Symmetrize output
    J = None
    K = None
    if J_flat is not None:
        J = J_flat.reshape((nao, nao))
        J = 0.5 * (J + J.T)
    if K_flat is not None:
        K = K_flat.reshape((nao, nao))
        K = 0.5 * (K + K.T)

    if profile is not None:
        profile["direct_jk_t_s"] = float(time.perf_counter() - t0)
        profile["direct_jk_kernel_calls"] = int(n_kernel_calls)
        profile["direct_jk_mako_calls"] = int(n_mako_calls)
        profile["direct_jk_ntasks"] = int(ctx.ntasks)

    return J, K


def direct_fock_rhf_mako(
    ctx,
    D,
    hcore,
    *,
    profile: dict | None = None,
    stats: dict | None = None,
    tile_dtype: str | None = None,
    mixed_precision: bool | None = None,
    f32_accum: bool | None = None,
    schedule: Any | None = None,
) -> Any:
    """Build the RHF Fock matrix using the Mako engine."""
    J, K = direct_JK_mako(
        ctx, D,
        want_J=True, want_K=True,
        profile=profile, stats=stats,
        tile_dtype=tile_dtype,
        mixed_precision=mixed_precision,
        f32_accum=f32_accum,
        schedule=schedule,
    )
    import cupy as cp  # noqa: PLC0415

    hcore_gpu = cp.asarray(hcore, dtype=cp.float64)
    F = hcore_gpu + J - 0.5 * K
    return F


def direct_JK_multi_mako(
    ctx,
    Da,
    Db,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
    stats: dict | None = None,
    tile_dtype: str | None = None,
    mixed_precision: bool | None = None,
    f32_accum: bool | None = None,
    schedule: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build (Ja, Ka, Jb, Kb) using the Mako engine."""
    Ja, Ka = direct_JK_mako(
        ctx, Da,
        want_J=want_J, want_K=want_K,
        profile=profile, stats=stats,
        tile_dtype=tile_dtype,
        mixed_precision=mixed_precision,
        f32_accum=f32_accum,
        schedule=schedule,
    )
    Jb, Kb = direct_JK_mako(
        ctx, Db,
        want_J=want_J, want_K=want_K,
        profile=profile, stats=stats,
        tile_dtype=tile_dtype,
        mixed_precision=mixed_precision,
        f32_accum=f32_accum,
        schedule=schedule,
    )
    return Ja, Ka, Jb, Kb
