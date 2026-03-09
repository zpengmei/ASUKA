from __future__ import annotations

"""Integral-direct J/K builder for SCF.

Builds Coulomb (J) and exchange (K) matrices by evaluating 4-center integrals
on-the-fly and contracting with the density matrix, without materializing the
full ERI tensor.  Memory is O(nao^2) GPU + O(ntasks) CPU for large systems.

For large systems (e.g. 100 H2O / 6-31g*) the number of screened shell-quartet
tasks can reach 1-10 billion.  This module avoids storing all tasks on GPU by
using *presorted slabs*: tasks are generated in Q-rank slabs on the GPU, sorted
by ERI class, and stored as either GPU-resident CuPy arrays (small systems) or
CPU numpy arrays (large systems).  Group offsets are computed on-GPU to avoid
expensive D→H of large class-ID arrays.

Uses specialized CUDA kernels (ssss, psss, pppp, dsds, ...) via the
``eri_dispatch`` infrastructure for optimal throughput, with automatic
bra/ket swap and fallback to generic Rys for unsupported angular momenta.

Usage::

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=1e-12)
    J, K = direct_JK(ctx, D)
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _SortedSlab:
    """Precomputed sorted task slab (shell-pair index pairs grouped by ERI class).

    If ``gpu_resident=True`` the arrays live on GPU (CuPy); otherwise they are
    CPU numpy arrays that are uploaded to GPU on each J/K call.
    """

    # Task arrays — either CuPy (gpu_resident) or numpy (cpu)
    ab_sorted: Any   # (ntasks_slab,) int32
    cd_sorted: Any   # (ntasks_slab,) int32
    # Class group metadata (always CPU numpy — tiny)
    class_ids: np.ndarray  # (nclass,) int32
    offsets: np.ndarray    # (nclass+1,) int32
    gpu_resident: bool


@dataclass(frozen=True)
class DirectJKContext:
    """Pre-computed, reusable context for integral-direct J/K builds."""

    ao_basis: Any
    nao: int
    # GPU-resident basis data
    dbasis: Any
    dsp: Any
    pair_tables: Any
    sp_A_dev: Any           # cp.ndarray int32 (nsp,)
    sp_B_dev: Any           # cp.ndarray int32 (nsp,)
    shell_ao_start_dev: Any # cp.ndarray int32 (nshell,)
    sp_class_lo_dev: Any    # cp.ndarray int32 (nsp,)
    sp_class_lo_cpu: np.ndarray  # (nsp,) int32
    # Presorted task slabs
    slabs: tuple            # tuple[_SortedSlab, ...]
    nsp: int
    ntasks: int
    eps_schwarz: float
    threads: int
    max_tile_bytes: int


def _build_sorted_slab(
    perm32_dev,
    jmax_dev,
    sp_class_lo_dev,
    i0: int,
    i1: int,
    gpu_budget_bytes: int,
):
    """Generate, sort, and return a task slab for Q-ranks [i0, i1).

    Group offsets are computed on GPU (avoids D→H of large class-ID arrays).
    Task arrays are kept GPU-resident if they fit within ``gpu_budget_bytes``;
    otherwise they are transferred to CPU numpy.
    """
    import cupy as cp  # noqa: PLC0415

    n_slab = i1 - i0
    if n_slab <= 0:
        return None

    jmax_slab = jmax_dev[i0:i1]
    total = int(jmax_slab.sum())
    if total == 0:
        return None

    # --- Generate tasks on GPU ---
    group_offsets = cp.empty(n_slab + 1, dtype=cp.int64)
    group_offsets[0] = 0
    cp.cumsum(jmax_slab, out=group_offsets[1:])

    indices = cp.arange(total, dtype=cp.int64)
    k_arr = cp.searchsorted(group_offsets, indices, side="right") - 1
    j_arr = indices - group_offsets[k_arr]
    del group_offsets, jmax_slab

    ab = perm32_dev[k_arr + cp.int64(i0)]
    cd = perm32_dev[j_arr]
    del k_arr, j_arr, indices

    swap = cd > ab
    ab_out = cp.where(swap, cd, ab)
    cd_out = cp.where(swap, ab, cd)
    del ab, cd, swap

    # --- Sort by class ID on GPU ---
    class_id_dev = sp_class_lo_dev[ab_out] | (sp_class_lo_dev[cd_out] << cp.int32(16))
    perm_dev = cp.argsort(class_id_dev, kind="stable")
    ab_sorted_dev = cp.ascontiguousarray(ab_out[perm_dev])
    cd_sorted_dev = cp.ascontiguousarray(cd_out[perm_dev])
    class_id_sorted_dev = class_id_dev[perm_dev]
    del ab_out, cd_out, class_id_dev, perm_dev

    # --- Compute group offsets on GPU (avoids D→H of full class_id array) ---
    slab_nt = total
    if slab_nt > 1:
        diff = class_id_sorted_dev[1:] != class_id_sorted_dev[:-1]
        change_positions = cp.nonzero(diff)[0] + 1  # positions where class changes
        del diff
        # D→H only the small arrays: change_positions (nclass-1,) and first/last class IDs
        change_cpu = cp.asnumpy(change_positions).astype(np.int32)
        del change_positions
    else:
        change_cpu = np.empty((0,), dtype=np.int32)

    # class IDs at group starts: D→H only nclass values (tiny)
    if change_cpu.size == 0:
        group_start_positions = cp.zeros(1, dtype=cp.int64)
    else:
        group_start_positions = cp.concatenate([
            cp.zeros(1, dtype=cp.int64),
            cp.asarray(change_cpu.astype(np.int64)),
        ])
    class_ids_dev_small = class_id_sorted_dev[cp.asarray(group_start_positions, dtype=cp.int64)]
    class_ids_cpu = cp.asnumpy(class_ids_dev_small).astype(np.int32)
    del class_id_sorted_dev, class_ids_dev_small, group_start_positions

    offsets_cpu = np.concatenate(([0], change_cpu, [slab_nt])).astype(np.int32)
    del change_cpu

    # --- Decide storage: GPU or CPU ---
    task_bytes = total * 2 * 4  # ab + cd as int32
    if task_bytes <= gpu_budget_bytes:
        # GPU-resident: keep CuPy arrays (zero per-iteration H→D)
        return _SortedSlab(
            ab_sorted=ab_sorted_dev,
            cd_sorted=cd_sorted_dev,
            class_ids=class_ids_cpu,
            offsets=offsets_cpu,
            gpu_resident=True,
        )
    else:
        # CPU-resident: D→H task arrays (paid once per context build, not per iteration)
        ab_cpu = cp.asnumpy(ab_sorted_dev).astype(np.int32)
        cd_cpu = cp.asnumpy(cd_sorted_dev).astype(np.int32)
        del ab_sorted_dev, cd_sorted_dev
        return _SortedSlab(
            ab_sorted=ab_cpu,
            cd_sorted=cd_cpu,
            class_ids=class_ids_cpu,
            offsets=offsets_cpu,
            gpu_resident=False,
        )


def make_direct_jk_context(
    ao_basis,
    *,
    eps_schwarz: float = 1e-12,
    threads: int = 256,
    max_tile_bytes: int = 256 << 20,
    max_slab_tasks: int = 200_000_000,
    gpu_task_budget_bytes: int = 2 << 30,
) -> DirectJKContext:
    """One-time setup: build shell pairs, Schwarz bounds, and presorted task slabs.

    Parameters
    ----------
    ao_basis
        Packed AO basis object (Cartesian).
    eps_schwarz
        Schwarz screening threshold.
    threads
        CUDA threads per block.
    max_tile_bytes
        Maximum bytes for the integral tile buffer per chunk.
    max_slab_tasks
        Maximum tasks per slab (bounds peak GPU memory during context build).
        Default 200 M ≈ 3.2 GB peak GPU for task arrays + sort workspace.
    gpu_task_budget_bytes
        If a slab's task arrays fit within this budget (default 2 GB), they
        are kept GPU-resident for zero per-iteration H→D overhead.
    """

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri.gpu import (  # noqa: PLC0415
        CUDA_MAX_L,
        CUDA_MAX_NROOTS,
        build_pair_tables_ss_device,
        has_cuda_ext,
        to_device_basis_ss,
        to_device_shell_pairs,
    )
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415
    from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available")

    nao = int(nao_cart_from_basis(ao_basis))
    if nao <= 0:
        raise ValueError("AO basis has zero AO functions")

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        raise ValueError("AO basis has zero shells")
    if int(shell_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"CUDA direct J/K supports only l<={CUDA_MAX_L} (nroots<={CUDA_MAX_NROOTS})"
        )

    sp = build_shell_pairs_l_order(ao_basis)
    nsp = int(sp.sp_A.shape[0])
    if nsp == 0:
        raise ValueError("No shell pairs generated")

    eps_f = float(eps_schwarz)
    if eps_f > 0.0:
        from asuka.cueri.screening import schwarz_shellpairs_device  # noqa: PLC0415

        Q_dev = schwarz_shellpairs_device(
            ao_basis, sp,
            threads=int(threads),
            max_tiles_bytes=int(max_tile_bytes),
        )
        Q_np = cp.asnumpy(Q_dev)
        del Q_dev
    else:
        Q_np = np.ones((nsp,), dtype=np.float64)

    perm = np.argsort(-Q_np, kind="stable")
    Q_sorted = Q_np[perm]
    n_valid = int(np.searchsorted(-Q_sorted, 0.0, side="left"))
    if n_valid == 0:
        raise ValueError("All shell pairs have zero Schwarz bound")

    Q_sorted = Q_sorted[:n_valid]
    neg_Q_sorted = -Q_sorted
    perm32_cpu = perm[:n_valid].astype(np.int32)

    if eps_f > 0.0:
        thrs = eps_f / np.maximum(Q_sorted, 1e-300)
        jmax_uncapped = np.searchsorted(neg_Q_sorted, -thrs, side="right").astype(np.int64)
        jmax_cpu = np.minimum(jmax_uncapped, np.arange(n_valid, dtype=np.int64) + 1)
    else:
        jmax_cpu = np.arange(1, n_valid + 1, dtype=np.int64)

    ntasks = int(jmax_cpu.sum())
    cumtasks_cpu = np.concatenate(([np.int64(0)], np.cumsum(jmax_cpu)))

    sp_A_np = np.asarray(sp.sp_A, dtype=np.int32)
    sp_B_np = np.asarray(sp.sp_B, dtype=np.int32)
    la_sp = shell_l[sp_A_np]
    lb_sp = shell_l[sp_B_np]
    sp_class_lo_cpu = ((la_sp & 0xFF) | ((lb_sp & 0xFF) << 8)).astype(np.int32)

    sp_class_lo_dev = cp.ascontiguousarray(cp.asarray(sp_class_lo_cpu, dtype=cp.int32))
    perm32_dev = cp.ascontiguousarray(cp.asarray(perm32_cpu, dtype=cp.int32))
    jmax_dev = cp.ascontiguousarray(cp.asarray(jmax_cpu, dtype=cp.int64))

    # Build presorted slabs with cumulative GPU budget tracking
    slabs = []
    gpu_bytes_used = 0
    slab_i0 = 0
    while slab_i0 < n_valid:
        target = cumtasks_cpu[slab_i0] + max_slab_tasks
        slab_i1 = int(np.searchsorted(cumtasks_cpu, target, side="right")) - 1
        slab_i1 = max(slab_i0 + 1, slab_i1)
        slab_i1 = min(slab_i1, n_valid)

        remaining_gpu = max(0, int(gpu_task_budget_bytes) - gpu_bytes_used)
        slab = _build_sorted_slab(
            perm32_dev, jmax_dev, sp_class_lo_dev,
            slab_i0, slab_i1,
            gpu_budget_bytes=remaining_gpu,
        )
        if slab is not None:
            if slab.gpu_resident:
                gpu_bytes_used += slab.ab_sorted.nbytes + slab.cd_sorted.nbytes
            slabs.append(slab)

        slab_i0 = slab_i1

    del perm32_dev, jmax_dev
    # Release slab-build temporaries from pool to avoid fragmentation during J/K
    # (only non-resident blocks; GPU-resident slab arrays are kept)
    cp.get_default_memory_pool().free_all_blocks()

    dbasis = to_device_basis_ss(ao_basis)
    dsp = to_device_shell_pairs(sp)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, stream=None, threads=int(threads))

    sp_A_dev = cp.ascontiguousarray(cp.asarray(sp_A_np, dtype=cp.int32))
    sp_B_dev = cp.ascontiguousarray(cp.asarray(sp_B_np, dtype=cp.int32))
    shell_ao_start_np = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_np, dtype=cp.int32))

    return DirectJKContext(
        ao_basis=ao_basis,
        nao=int(nao),
        dbasis=dbasis,
        dsp=dsp,
        pair_tables=pair_tables,
        sp_A_dev=sp_A_dev,
        sp_B_dev=sp_B_dev,
        shell_ao_start_dev=shell_ao_start_dev,
        sp_class_lo_dev=sp_class_lo_dev,
        sp_class_lo_cpu=sp_class_lo_cpu,
        slabs=tuple(slabs),
        nsp=int(nsp),
        ntasks=int(ntasks),
        eps_schwarz=float(eps_f),
        threads=int(threads),
        max_tile_bytes=int(max_tile_bytes),
    )


def direct_JK(
    ctx: DirectJKContext,
    D,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
):
    """Build J and K via streaming integral-direct 4-center evaluation.

    GPU-resident slabs (small systems) incur zero H→D overhead.
    CPU-resident slabs (large systems) are streamed to GPU one at a time.

    Parameters
    ----------
    ctx
        Pre-built context from :func:`make_direct_jk_context`.
    D
        AO density matrix, shape ``(nao, nao)``, CuPy or NumPy array.
    want_J, want_K
        Which matrices to compute.
    profile
        Optional dict for timing statistics.  **Note**: profile mode inserts
        ``event.synchronize()`` after each contraction kernel call, serializing
        execution.  Reported timings are not representative of production
        throughput.

    Returns
    -------
    (J, K) : tuple
        Coulomb and exchange matrices, each ``(nao, nao)`` CuPy array, or None.
    """

    if not want_J and not want_K:
        return None, None

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.eri_dispatch import KernelBatch, resolve_kernel_class_id, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList, decode_eri_class_id  # noqa: PLC0415

    nao = ctx.nao
    threads = ctx.threads
    max_tile_bytes = ctx.max_tile_bytes

    D_gpu = cp.asarray(D, dtype=cp.float64)
    if D_gpu.ndim != 2 or D_gpu.shape != (nao, nao):
        raise ValueError(f"D must be ({nao}, {nao}), got {tuple(D_gpu.shape)}")
    D_flat = cp.ascontiguousarray(D_gpu.ravel())

    J_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_J else None
    K_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_K else None

    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    n_kernel_calls = 0
    t0 = time.perf_counter()

    for slab in ctx.slabs:
        if slab.gpu_resident:
            # Task arrays already on GPU — zero H→D cost
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
            owned = False
        else:
            # H→D: upload presorted task indices from CPU
            ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
            cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))
            owned = True

        for g in range(int(slab.class_ids.shape[0])):
            orig_cid = int(slab.class_ids[g])
            j0, j1 = int(slab.offsets[g]), int(slab.offsets[g + 1])
            if j1 <= j0:
                continue

            la, lb, lc, ld = decode_eri_class_id(orig_cid)
            kernel_cid, transpose = resolve_kernel_class_id(orig_cid)

            nA = int(ncart(la))
            nB = int(ncart(lb))
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            ncomp = nA * nB * nC * nD
            chunk_ntasks = max(1, max_tile_bytes // max(ncomp * 8, 1))
            use_warp_mode = ncomp <= 128

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]

            class_ntasks = j1 - j0
            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue

                sub_batch = KernelBatch(
                    task_idx=np.empty(0, dtype=np.int32),  # unused by run_kernel_batch_spd
                    kernel_tasks=TaskList(
                        task_spAB=kernel_spAB_full[c0:c1],
                        task_spCD=kernel_spCD_full[c0:c1],
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
                    threads=threads,
                    mode="warp" if use_warp_mode else "auto",
                    profile=profile,
                )
                n_kernel_calls += 1

                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()

                _ext.contract_jk_tiles_ordered_inplace_device(
                    ab_dev[j0 + c0: j0 + c1],
                    cd_dev[j0 + c0: j0 + c1],
                    ctx.sp_A_dev,
                    ctx.sp_B_dev,
                    ctx.shell_ao_start_dev,
                    int(nao),
                    int(nA),
                    int(nB),
                    int(nC),
                    int(nD),
                    tiles.ravel(),
                    D_flat,
                    J_flat,
                    K_flat,
                    int(threads),
                    int(stream_ptr),
                    False,
                )

                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(
                        cp.cuda.get_elapsed_time(_tc0, _tc1)
                    )

        if owned:
            del ab_dev, cd_dev

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
        profile["direct_jk_ntasks"] = int(ctx.ntasks)

    return J, K


def direct_JK_multi(
    ctx: DirectJKContext,
    Da,
    Db,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
):
    """Build (Ja, Ka, Jb, Kb) evaluating each ERI tile once.

    For UHF/ROHF this is ~2x faster than calling ``direct_JK`` twice because
    the ERI evaluation (94% of runtime) is shared.

    Parameters
    ----------
    profile
        Optional dict for timing statistics.  **Note**: profile mode inserts
        ``event.synchronize()`` after each contraction kernel call, serializing
        execution.  Reported timings are not representative of production
        throughput.

    Returns
    -------
    (Ja, Ka, Jb, Kb) : tuple
        Coulomb/exchange matrices for each density, each ``(nao, nao)`` or None.
    """

    if not want_J and not want_K:
        return None, None, None, None

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.eri_dispatch import KernelBatch, resolve_kernel_class_id, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList, decode_eri_class_id  # noqa: PLC0415

    nao = ctx.nao
    threads = ctx.threads
    max_tile_bytes = ctx.max_tile_bytes

    Da_gpu = cp.asarray(Da, dtype=cp.float64)
    if Da_gpu.ndim != 2 or Da_gpu.shape != (nao, nao):
        raise ValueError(f"Da must be ({nao}, {nao}), got {tuple(Da_gpu.shape)}")
    Db_gpu = cp.asarray(Db, dtype=cp.float64)
    if Db_gpu.ndim != 2 or Db_gpu.shape != (nao, nao):
        raise ValueError(f"Db must be ({nao}, {nao}), got {tuple(Db_gpu.shape)}")

    Da_flat = cp.ascontiguousarray(Da_gpu.ravel())
    Db_flat = cp.ascontiguousarray(Db_gpu.ravel())

    nao2 = nao * nao
    Ja_flat = cp.zeros((nao2,), dtype=cp.float64) if want_J else None
    Ka_flat = cp.zeros((nao2,), dtype=cp.float64) if want_K else None
    Jb_flat = cp.zeros((nao2,), dtype=cp.float64) if want_J else None
    Kb_flat = cp.zeros((nao2,), dtype=cp.float64) if want_K else None

    # Dummies for want_J=False (kernel needs a non-NULL J pointer)
    Ja_dummy = cp.zeros((nao2,), dtype=cp.float64) if Ja_flat is None else None
    Jb_dummy = cp.zeros((nao2,), dtype=cp.float64) if Jb_flat is None else None

    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    n_kernel_calls = 0
    t0 = time.perf_counter()

    for slab in ctx.slabs:
        if slab.gpu_resident:
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
            owned = False
        else:
            ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
            cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))
            owned = True

        for g in range(int(slab.class_ids.shape[0])):
            orig_cid = int(slab.class_ids[g])
            j0, j1 = int(slab.offsets[g]), int(slab.offsets[g + 1])
            if j1 <= j0:
                continue

            la, lb, lc, ld = decode_eri_class_id(orig_cid)
            kernel_cid, transpose = resolve_kernel_class_id(orig_cid)

            nA = int(ncart(la))
            nB = int(ncart(lb))
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            ncomp = nA * nB * nC * nD
            chunk_ntasks = max(1, max_tile_bytes // max(ncomp * 8, 1))
            use_warp_mode = ncomp <= 128

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]

            class_ntasks = j1 - j0
            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue

                sub_batch = KernelBatch(
                    task_idx=np.empty(0, dtype=np.int32),
                    kernel_tasks=TaskList(
                        task_spAB=kernel_spAB_full[c0:c1],
                        task_spCD=kernel_spCD_full[c0:c1],
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
                    threads=threads,
                    mode="warp" if use_warp_mode else "auto",
                    profile=profile,
                )
                n_kernel_calls += 1

                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()

                _ext.contract_jk_tiles_ordered_multi2_inplace_device(
                    ab_dev[j0 + c0: j0 + c1],
                    cd_dev[j0 + c0: j0 + c1],
                    ctx.sp_A_dev,
                    ctx.sp_B_dev,
                    ctx.shell_ao_start_dev,
                    int(nao),
                    int(nA),
                    int(nB),
                    int(nC),
                    int(nD),
                    tiles.ravel(),
                    Da_flat,
                    Db_flat,
                    Ja_flat if Ja_flat is not None else Ja_dummy,
                    Ka_flat,
                    Jb_flat if Jb_flat is not None else Jb_dummy,
                    Kb_flat,
                    int(threads),
                    int(stream_ptr),
                    False,
                )

                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(
                        cp.cuda.get_elapsed_time(_tc0, _tc1)
                    )

        if owned:
            del ab_dev, cd_dev

    Ja = Jb = Ka = Kb = None
    if Ja_flat is not None:
        Ja = 0.5 * (Ja_flat.reshape((nao, nao)) + Ja_flat.reshape((nao, nao)).T)
    if Ka_flat is not None:
        Ka = 0.5 * (Ka_flat.reshape((nao, nao)) + Ka_flat.reshape((nao, nao)).T)
    if Jb_flat is not None:
        Jb = 0.5 * (Jb_flat.reshape((nao, nao)) + Jb_flat.reshape((nao, nao)).T)
    if Kb_flat is not None:
        Kb = 0.5 * (Kb_flat.reshape((nao, nao)) + Kb_flat.reshape((nao, nao)).T)

    if profile is not None:
        profile["direct_jk_t_s"] = float(time.perf_counter() - t0)
        profile["direct_jk_kernel_calls"] = int(n_kernel_calls)
        profile["direct_jk_ntasks"] = int(ctx.ntasks)

    return Ja, Ka, Jb, Kb


__all__ = ["DirectJKContext", "direct_JK", "direct_JK_multi", "make_direct_jk_context"]
