"""Derivative-oriented batching utilities for cuERI.

The existing iterator in :mod:`asuka.cueri.tile_eval` is value-tile oriented.
For analytic 4-center nuclear derivatives, the recommended interface is *contracted*:

    contract d(μν|λσ)/dR against bar_eri(μν,λσ) -> (4 centers × 3 coords) per task

In that setting, batching is still driven by the ERI class id (la,lb,lc,ld),
but the iterator should yield batches that make it easy to:

  1) build / supply `bar_eri` for that batch
  2) call the low-level CPU/CUDA contracted derivative kernels

This module provides a small batch descriptor for that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import TaskList, decode_eri_class_id


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


@dataclass(frozen=True)
class DerivKernelBatch:
    """A batch of tasks with a single ERI class (la,lb,lc,ld)."""

    task_idx: np.ndarray  # int64 indices into the original TaskList
    task_spAB: np.ndarray  # int32
    task_spCD: np.ndarray  # int32

    la: int
    lb: int
    lc: int
    ld: int

    nA: int
    nB: int
    nC: int
    nD: int
    nAB: int
    nCD: int
    nElem: int


def iter_deriv_kernel_batches_spd(tasks: TaskList, max_tasks: int | None = None):
    """Yield :class:`DerivKernelBatch` objects grouped by ERI class.

    Parameters
    ----------
    tasks
        A :class:`~asuka.cueri.tasks.TaskList` (typically from the shell-pair generator).
    max_tasks
        Optional maximum number of tasks per yielded batch. Useful to cap
        `bar_eri` memory when the caller materializes it per batch.

    Notes
    -----
    The low-level derivative kernels require that all tasks in a launch share
    the same (la,lb,lc,ld), because that fixes (nAB,nCD) and thus the `bar_eri`
    and output layout.
    """

    class_ids = np.asarray(tasks.task_class_id, dtype=np.int32)
    uniq = np.unique(class_ids)

    spAB_all = np.asarray(tasks.task_spAB, dtype=np.int32)
    spCD_all = np.asarray(tasks.task_spCD, dtype=np.int32)

    for cid in uniq:
        la, lb, lc, ld = decode_eri_class_id(int(cid))

        nA = _ncart(la)
        nB = _ncart(lb)
        nC = _ncart(lc)
        nD = _ncart(ld)
        nAB = nA * nB
        nCD = nC * nD
        nElem = nAB * nCD

        idx = np.nonzero(class_ids == cid)[0].astype(np.int64, copy=False)
        if idx.size == 0:
            continue

        if max_tasks is None or max_tasks <= 0 or idx.size <= max_tasks:
            yield DerivKernelBatch(
                task_idx=idx,
                task_spAB=spAB_all[idx],
                task_spCD=spCD_all[idx],
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                nA=nA,
                nB=nB,
                nC=nC,
                nD=nD,
                nAB=nAB,
                nCD=nCD,
                nElem=nElem,
            )
            continue

        # Chunk.
        for i0 in range(0, idx.size, max_tasks):
            j = idx[i0 : i0 + max_tasks]
            yield DerivKernelBatch(
                task_idx=j,
                task_spAB=spAB_all[j],
                task_spCD=spCD_all[j],
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                nA=nA,
                nB=nB,
                nC=nC,
                nD=nD,
                nAB=nAB,
                nCD=nCD,
                nElem=nElem,
            )
