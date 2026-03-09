from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cart import ncart
from .eri_dispatch import KernelBatch, plan_kernel_batches_spd, resolve_kernel_class_id, run_kernel_batch_spd
from .tasks import TaskList, decode_eri_class_id


@dataclass(frozen=True)
class TileEvalBatch:
    """One evaluated tile batch corresponding to a subset of a TaskList."""

    task_idx: np.ndarray  # int32 indices into the original task list
    task_spAB: np.ndarray  # int32, shape (ntasks,)
    task_spCD: np.ndarray  # int32, shape (ntasks,)
    kernel_class_id: np.int32
    tiles: object  # CuPy array, shape (ntasks, nAB, nCD)
    kernel_transposed: bool = False  # True only when skip_transpose=True and batch.transpose=True


def iter_tile_batches_spd(
    tasks: TaskList,
    *,
    shell_pairs,
    shell_l: np.ndarray,
    dbasis,
    dsp,
    pt,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    max_tile_bytes: int | None = None,
    boys: str = "ref",
    skip_transpose: bool = False,
    task_class_id: int | None = None,
    preupload_tasks: bool = False,
):
    """Yield evaluated (tile) batches for the given SPD-class task list.

    This is a thin iterator wrapper around:
    - `eri_dispatch.plan_kernel_batches_spd`
    - `eri_dispatch.run_kernel_batch_spd`

    By default the yielded `tiles` are in the original task orientation (i.e. any internal
    bra/ket swaps are transposed back by the dispatcher). When `skip_transpose=True`, swapped
    classes are yielded in kernel-native orientation and `TileEvalBatch.kernel_transposed=True`.

    If `task_class_id` is provided, dispatch planning is bypassed and a single
    fixed-class batch is used (with `resolve_kernel_class_id(task_class_id)`).
    This is intended for callers that already grouped tasks by class.

    If `preupload_tasks=True`, task index arrays are copied to device once per
    batch and reused across chunked launches.
    """

    if tasks.ntasks == 0:
        return

    if task_class_id is None:
        batches = plan_kernel_batches_spd(tasks, shell_pairs=shell_pairs, shell_l=shell_l)
    else:
        kernel_cid, transpose = resolve_kernel_class_id(int(task_class_id))
        idx = np.arange(int(tasks.ntasks), dtype=np.int32)
        if bool(transpose):
            kernel_tasks = TaskList(task_spAB=np.asarray(tasks.task_spCD, dtype=np.int32), task_spCD=np.asarray(tasks.task_spAB, dtype=np.int32))
        else:
            kernel_tasks = TaskList(task_spAB=np.asarray(tasks.task_spAB, dtype=np.int32), task_spCD=np.asarray(tasks.task_spCD, dtype=np.int32))
        batches = [
            KernelBatch(
                task_idx=idx,
                kernel_tasks=kernel_tasks,
                kernel_class_id=np.int32(kernel_cid),
                transpose=bool(transpose),
            )
        ]
    for batch in batches:
        task_spab_src = batch.kernel_tasks.task_spAB
        task_spcd_src = batch.kernel_tasks.task_spCD
        if bool(preupload_tasks):
            import cupy as cp

            task_spab_src = cp.ascontiguousarray(cp.asarray(task_spab_src, dtype=cp.int32))
            task_spcd_src = cp.ascontiguousarray(cp.asarray(task_spcd_src, dtype=cp.int32))

        la, lb, lc, ld = decode_eri_class_id(int(batch.kernel_class_id))
        nAB = int(ncart(int(la))) * int(ncart(int(lb)))
        nCD = int(ncart(int(lc))) * int(ncart(int(ld)))
        bytes_per_task = nAB * nCD * 8
        if bytes_per_task <= 0:
            raise RuntimeError("invalid tile shape in iter_tile_batches_spd (bytes_per_task <= 0)")

        nt = int(batch.kernel_tasks.ntasks)
        if nt == 0:
            continue

        max_tile_bytes_i = None if max_tile_bytes is None else int(max_tile_bytes)
        if max_tile_bytes_i is not None:
            if max_tile_bytes_i <= 0:
                raise ValueError("max_tile_bytes must be > 0 when provided")
            max_tasks = max(1, max_tile_bytes_i // bytes_per_task)
        else:
            max_tasks = nt

        for i0 in range(0, nt, max_tasks):
            i1 = min(nt, i0 + max_tasks)
            sub_tasks = TaskList(
                task_spAB=task_spab_src[i0:i1],
                task_spCD=task_spcd_src[i0:i1],
            )
            sub_batch = KernelBatch(
                task_idx=batch.task_idx[i0:i1],
                kernel_tasks=sub_tasks,
                kernel_class_id=batch.kernel_class_id,
                transpose=bool(batch.transpose),
            )

            tile = run_kernel_batch_spd(
                sub_batch,
                dbasis=dbasis,
                dsp=dsp,
                pt=pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
                boys=boys,
                skip_transpose=bool(skip_transpose),
            )

            idx = np.asarray(sub_batch.task_idx, dtype=np.int32)
            yield TileEvalBatch(
                task_idx=idx,
                task_spAB=tasks.task_spAB[idx],
                task_spCD=tasks.task_spCD[idx],
                kernel_class_id=np.int32(sub_batch.kernel_class_id),
                tiles=tile,
                kernel_transposed=bool(skip_transpose) and bool(sub_batch.transpose),
            )


__all__ = ["KernelBatch", "TileEvalBatch", "iter_tile_batches_spd"]
