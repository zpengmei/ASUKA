from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cart import ncart
from .eri_dispatch import KernelBatch, plan_kernel_batches_spd, run_kernel_batch_spd
from .tasks import TaskList, decode_eri_class_id


@dataclass(frozen=True)
class TileEvalBatch:
    """One evaluated tile batch corresponding to a subset of a TaskList."""

    task_idx: np.ndarray  # int32 indices into the original task list
    task_spAB: np.ndarray  # int32, shape (ntasks,)
    task_spCD: np.ndarray  # int32, shape (ntasks,)
    kernel_class_id: np.int32
    tiles: object  # CuPy array, shape (ntasks, nAB, nCD)


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
):
    """Yield evaluated (tile) batches for the given SPD-class task list.

    This is a thin iterator wrapper around:
    - `eri_dispatch.plan_kernel_batches_spd`
    - `eri_dispatch.run_kernel_batch_spd`

    The yielded `tiles` are always in the original task orientation (i.e. any internal
    bra/ket swaps are transposed back by the dispatcher).
    """

    if tasks.ntasks == 0:
        return

    batches = plan_kernel_batches_spd(tasks, shell_pairs=shell_pairs, shell_l=shell_l)
    for batch in batches:
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
                task_spAB=batch.kernel_tasks.task_spAB[i0:i1],
                task_spCD=batch.kernel_tasks.task_spCD[i0:i1],
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
            )

            idx = np.asarray(sub_batch.task_idx, dtype=np.int32)
            yield TileEvalBatch(
                task_idx=idx,
                task_spAB=tasks.task_spAB[idx],
                task_spCD=tasks.task_spCD[idx],
                kernel_class_id=np.int32(sub_batch.kernel_class_id),
                tiles=tile,
            )


__all__ = ["KernelBatch", "TileEvalBatch", "iter_tile_batches_spd"]
