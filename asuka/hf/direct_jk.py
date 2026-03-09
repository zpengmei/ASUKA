from __future__ import annotations

"""Integral-direct J/K builder for SCF.

Builds Coulomb (J) and exchange (K) matrices by evaluating 4-center integrals
on-the-fly and contracting with the density matrix, without materializing the
full ERI tensor.  Memory is O(nao^2) instead of O(nao^4).

Uses specialized CUDA kernels (ssss, psss, pppp, dsds, ...) via the
``eri_dispatch`` infrastructure for optimal throughput, with automatic
bra/ket swap and fallback to generic Rys for unsupported angular momenta.

Usage::

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=1e-12)
    J, K = direct_JK(ctx, D)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _BatchInfo:
    """Pre-computed data for one kernel batch (one specialized kernel dispatch)."""

    batch: Any  # KernelBatch from eri_dispatch
    # Device-resident original (un-swapped) task indices for J/K contraction
    orig_spAB_dev: Any  # cp.ndarray int32 (batch_ntasks,)
    orig_spCD_dev: Any  # cp.ndarray int32 (batch_ntasks,)
    # Original angular momenta (before any bra/ket swap)
    orig_la: int
    orig_lb: int
    orig_lc: int
    orig_ld: int


@dataclass(frozen=True)
class DirectJKContext:
    """Pre-computed, reusable context for integral-direct J/K builds."""

    ao_basis: Any
    nao: int
    # Kernel batch dispatch plan (specialized kernels with bra/ket swap)
    batch_infos: tuple  # tuple[_BatchInfo, ...]
    # Device-resident basis data
    dbasis: Any  # DeviceBasisSS
    dsp: Any  # DeviceShellPairs
    pair_tables: Any  # DevicePairTables
    sp_A_dev: Any  # cp.ndarray int32 (nsp,)
    sp_B_dev: Any  # cp.ndarray int32 (nsp,)
    shell_ao_start_dev: Any  # cp.ndarray int32 (nshell,)
    # Backward compat / diagnostics
    class_ids: np.ndarray  # (nclass,) int32
    # Config
    ntasks: int
    threads: int
    max_tile_bytes: int


def make_direct_jk_context(
    ao_basis,
    *,
    eps_schwarz: float = 1e-12,
    threads: int = 256,
    max_tile_bytes: int = 256 << 20,
) -> DirectJKContext:
    """One-time setup: build shell pairs, Schwarz bounds, screened task list.

    Parameters
    ----------
    ao_basis
        Packed AO basis object (Cartesian).
    eps_schwarz
        Schwarz screening threshold.  Quartets with ``Q_AB * Q_CD < eps``
        are skipped.  Use 0 to disable screening.
    threads
        CUDA threads per block.
    max_tile_bytes
        Maximum bytes for the integral tile buffer per chunk.
    """

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri.eri_dispatch import KernelBatch, resolve_kernel_class_id  # noqa: PLC0415
    from asuka.cueri.gpu import (  # noqa: PLC0415
        CUDA_MAX_L,
        CUDA_MAX_NROOTS,
        build_pair_tables_ss_device,
        has_cuda_ext,
        to_device_basis_ss,
        to_device_shell_pairs,
    )
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415
    from asuka.cueri.tasks import (  # noqa: PLC0415
        TaskList,
        build_tasks_screened,
        build_tasks_screened_sorted_q,
        decode_eri_class_id,
        group_tasks_by_class,
        with_task_class_id,
    )
    from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

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
            ao_basis,
            sp,
            threads=int(threads),
            max_tiles_bytes=int(max_tile_bytes),
        )
        Q_np = cp.asnumpy(Q_dev)
        tasks = build_tasks_screened_sorted_q(Q_np, eps=eps_f)
    else:
        Q_np = np.ones((nsp,), dtype=np.float64)
        tasks = build_tasks_screened(Q_np, eps=0.0)

    if tasks.ntasks == 0:
        raise ValueError("All shell quartets screened out")

    tasks = with_task_class_id(tasks, sp, shell_l)
    assert tasks.task_class_id is not None

    # Group tasks by class: O(N log N) sort — fast even for 46M tasks
    perm, class_ids, offsets = group_tasks_by_class(tasks.task_class_id)
    task_ab = tasks.task_spAB[perm]
    task_cd = tasks.task_spCD[perm]

    # Upload all task arrays in ONE transfer instead of one per batch
    task_ab_dev = cp.ascontiguousarray(cp.asarray(task_ab, dtype=cp.int32))
    task_cd_dev = cp.ascontiguousarray(cp.asarray(task_cd, dtype=cp.int32))

    # Build dispatch plan: O(1) per class (not O(57×ntasks) like plan_kernel_batches_spd).
    # resolve_kernel_class_id() checks if a native kernel exists, possibly with bra/ket swap.
    batch_infos = []
    for g in range(int(class_ids.shape[0])):
        orig_cid = int(class_ids[g])
        j0, j1 = int(offsets[g]), int(offsets[g + 1])
        if j1 <= j0:
            continue

        la, lb, lc, ld = decode_eri_class_id(orig_cid)
        kernel_cid, transpose = resolve_kernel_class_id(orig_cid)

        # Kernel tasks are CPU numpy arrays (slices, no copy)
        if transpose:
            kernel_spAB = task_cd[j0:j1]
            kernel_spCD = task_ab[j0:j1]
        else:
            kernel_spAB = task_ab[j0:j1]
            kernel_spCD = task_cd[j0:j1]

        batch = KernelBatch(
            task_idx=perm[j0:j1],
            kernel_tasks=TaskList(task_spAB=kernel_spAB, task_spCD=kernel_spCD),
            kernel_class_id=np.int32(kernel_cid),
            transpose=transpose,  # run_kernel_batch_spd transposes tiles back to (AB|CD) order
        )

        batch_infos.append(_BatchInfo(
            batch=batch,
            orig_spAB_dev=task_ab_dev[j0:j1],  # view — no extra copy
            orig_spCD_dev=task_cd_dev[j0:j1],  # view — no extra copy
            orig_la=int(la),
            orig_lb=int(lb),
            orig_lc=int(lc),
            orig_ld=int(ld),
        ))

    dbasis = to_device_basis_ss(ao_basis)
    dsp = to_device_shell_pairs(sp)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, stream=None, threads=int(threads))

    sp_A_dev = cp.ascontiguousarray(cp.asarray(sp.sp_A, dtype=cp.int32))
    sp_B_dev = cp.ascontiguousarray(cp.asarray(sp.sp_B, dtype=cp.int32))
    shell_ao_start_np = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_np, dtype=cp.int32))

    return DirectJKContext(
        ao_basis=ao_basis,
        nao=int(nao),
        batch_infos=tuple(batch_infos),
        dbasis=dbasis,
        dsp=dsp,
        pair_tables=pair_tables,
        sp_A_dev=sp_A_dev,
        sp_B_dev=sp_B_dev,
        shell_ao_start_dev=shell_ao_start_dev,
        class_ids=class_ids,
        ntasks=int(tasks.ntasks),
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
    """Build J and K via integral-direct 4-center evaluation.

    Uses specialized CUDA kernels (ssss, psss, pppp, dsds, ...) dispatched
    via ``eri_dispatch`` for each shell class, with automatic bra/ket swap
    and fallback to generic Rys for unsupported angular momenta.

    Parameters
    ----------
    ctx
        Pre-built context from :func:`make_direct_jk_context`.
    D
        AO density matrix, shape ``(nao, nao)``, CuPy array.
    want_J, want_K
        Which matrices to compute.
    profile
        Optional dict for timing statistics.

    Returns
    -------
    (J, K) : tuple
        Coulomb and exchange matrices, each ``(nao, nao)`` or ``None``.
    """

    if not want_J and not want_K:
        return None, None

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.eri_dispatch import KernelBatch, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList  # noqa: PLC0415

    nao = ctx.nao
    threads = ctx.threads
    max_tile_bytes = ctx.max_tile_bytes

    D_gpu = cp.asarray(D, dtype=cp.float64)
    D_flat = cp.ascontiguousarray(D_gpu.ravel())
    if D_flat.shape[0] != nao * nao:
        raise ValueError(f"D must have {nao * nao} elements, got {D_flat.shape[0]}")

    J_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_J else None
    K_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_K else None

    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    n_kernel_calls = 0
    t0 = time.perf_counter()

    for bi in ctx.batch_infos:
        batch = bi.batch
        la, lb, lc, ld = bi.orig_la, bi.orig_lb, bi.orig_lc, bi.orig_ld
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nC = int(ncart(lc))
        nD = int(ncart(ld))
        nAB = nA * nB
        nCD = nC * nD
        bytes_per_task = nAB * nCD * 8
        chunk_ntasks = max(1, max_tile_bytes // max(bytes_per_task, 1))

        batch_ntasks = int(batch.task_idx.shape[0])

        for c0 in range(0, batch_ntasks, chunk_ntasks):
            c1 = min(batch_ntasks, c0 + chunk_ntasks)
            nt = c1 - c0
            if nt == 0:
                continue

            # Create sub-batch for this chunk
            sub_batch = KernelBatch(
                task_idx=batch.task_idx[c0:c1],
                kernel_tasks=TaskList(
                    task_spAB=batch.kernel_tasks.task_spAB[c0:c1],
                    task_spCD=batch.kernel_tasks.task_spCD[c0:c1],
                ),
                kernel_class_id=batch.kernel_class_id,
                transpose=batch.transpose,
            )

            # Evaluate ERI tiles using specialized kernel dispatch
            tiles = run_kernel_batch_spd(
                sub_batch,
                dbasis=ctx.dbasis,
                dsp=ctx.dsp,
                pt=ctx.pair_tables,
                stream=None,
                threads=threads,
                profile=profile,
            )
            n_kernel_calls += 1

            # Contract with D and accumulate into J/K
            # tiles shape: (nt, nAB, nCD) in original bra/ket order
            if profile is not None:
                _t_contract0 = cp.cuda.Event()
                _t_contract1 = cp.cuda.Event()
                _t_contract0.record()
            _ext.contract_jk_tiles_ordered_inplace_device(
                bi.orig_spAB_dev[c0:c1],
                bi.orig_spCD_dev[c0:c1],
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
                J_flat if J_flat is not None else cp.zeros((nao * nao,), dtype=cp.float64),
                K_flat,  # None → py::none → nullptr in C++
                int(threads),
                int(stream_ptr),
                False,
            )
            if profile is not None:
                _t_contract1.record()
                _t_contract1.synchronize()
                profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(
                    cp.cuda.get_elapsed_time(_t_contract0, _t_contract1)
                )

    cp.cuda.get_current_stream().synchronize()

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


__all__ = ["DirectJKContext", "direct_JK", "make_direct_jk_context"]
