from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import BasisSoA
from .shell_pairs import ShellPairs, build_shell_pairs
from .stream import stream_ctx
from .tasks import TaskList, build_tasks_screened, build_tasks_screened_sorted_q


@dataclass(frozen=True)
class SSSSDigestResult:
    basis: BasisSoA
    shell_pairs: ShellPairs
    Q: np.ndarray  # float64, shape (nSP,)
    tasks: TaskList
    out: object  # CuPy array (float64, shape (nSP,))


def digest_ssss_shellpairs(
    basis: BasisSoA,
    W,
    eps: float,
    *,
    tasks_method: str = "sorted_q",
    batch_ntasks: int = 200_000,
    batch_bytes: int | None = None,
    batch_mem_fraction: float | None = None,
    eri_mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
    threads: int = 256,
    stream=None,
) -> SSSSDigestResult:
    """End-to-end Step-1 pipeline (ssss) with a shell-pair keyed digest.

    This implements the Step-1 execution model:
      pack basis → build shell-pairs → pair tables → Schwarz Q → screened tasks →
      evaluate ERI per task → entry-CSR (two-entry trick) → atomic-free reduction.

    Parameters
    ----------
    basis
        Packed s-shell basis (see :class:`~cueri.basis.BasisSoA`).
    W
        Weights per shell-pair key, shape (nSP,). Can be NumPy or CuPy.
    eps
        Screening threshold; keep tasks where Q[ab]*Q[cd] >= eps.
    tasks_method
        "sorted_q" (Step 1.2) or "quadratic".
    batch_ntasks
        Max #tasks per batch (CSR is built per batch).
    eri_mode
        Passed to `cueri.gpu.eri_ssss_device`: "block", "warp", or "auto".
    work_small_max
        Work threshold for `eri_mode="auto"`.
    """

    from .gpu import (  # local import to keep CuPy optional at import time
        build_entry_csr_device,
        build_pair_tables_ss_device,
        eri_ssss_device,
        has_cuda_ext,
        reduce_from_entry_csr_device,
        schwarz_ssss_device,
        to_device_basis_ss,
        to_device_shell_pairs,
    )

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

    import cupy as cp

    if batch_ntasks <= 0:
        raise ValueError("batch_ntasks must be > 0")
    if batch_bytes is not None and batch_bytes <= 0:
        raise ValueError("batch_bytes must be > 0")
    if batch_mem_fraction is not None and not (0.0 < batch_mem_fraction <= 1.0):
        raise ValueError("batch_mem_fraction must be in (0,1]")
    tasks_method = tasks_method.lower().strip()
    if tasks_method not in ("sorted_q", "quadratic"):
        raise ValueError("tasks_method must be one of: 'sorted_q', 'quadratic'")

    with stream_ctx(stream):
        stream = None
        shell_pairs = build_shell_pairs(basis)
        nsp = int(shell_pairs.sp_A.shape[0])

        if nsp == 0:
            out = cp.zeros((0,), dtype=cp.float64)
            return SSSSDigestResult(
                basis=basis,
                shell_pairs=shell_pairs,
                Q=np.empty((0,), dtype=np.float64),
                tasks=TaskList(task_spAB=np.empty((0,), dtype=np.int32), task_spCD=np.empty((0,), dtype=np.int32)),
                out=out,
            )

        W_dev = cp.asarray(W, dtype=cp.float64).ravel()
        if W_dev.shape != (nsp,):
            raise ValueError(f"W must have shape (nSP,) = ({nsp},), got {tuple(W_dev.shape)}")

        dbasis = to_device_basis_ss(basis)
        dsp = to_device_shell_pairs(shell_pairs)
        pair_tables = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=threads)
        Q_dev = schwarz_ssss_device(dsp, pair_tables, stream=stream, threads=threads)
        Q = cp.asnumpy(Q_dev)

        if tasks_method == "sorted_q":
            tasks = build_tasks_screened_sorted_q(Q, eps)
        else:
            tasks = build_tasks_screened(Q, eps)

        if batch_bytes is None and batch_mem_fraction is not None:
            free_bytes, _total = cp.cuda.runtime.memGetInfo()
            batch_bytes = int(free_bytes * float(batch_mem_fraction))

        if batch_bytes is not None:
            # Conservative estimate for batch temporary allocations.
            # - task_spAB/task_spCD: 8 bytes per task (2×int32)
            # - eri_task: 8 bytes per task (float64)
            # - entry_task/entry_widx: up to 16 bytes per task (2 entries × 2×int32)
            # - multiblock partials (worst-case): 8*blocks_per_task bytes per task
            bytes_per_task = 32 + 8 * int(blocks_per_task)
            batch_ntasks = max(1, int(batch_bytes // bytes_per_task))

        out_total = cp.zeros((nsp,), dtype=cp.float64)
        ntasks = int(tasks.ntasks)
        for t0 in range(0, ntasks, batch_ntasks):
            t1 = min(t0 + batch_ntasks, ntasks)
            batch = TaskList(task_spAB=tasks.task_spAB[t0:t1], task_spCD=tasks.task_spCD[t0:t1])

            eri_task = eri_ssss_device(
                batch,
                dsp,
                pair_tables,
                stream=stream,
                threads=threads,
                mode=eri_mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
                boys=boys,
            )
            entry_offsets, entry_task, entry_widx = build_entry_csr_device(batch, nkey=nsp, stream=stream, threads=threads)
            out_batch = reduce_from_entry_csr_device(
                entry_offsets,
                entry_task,
                entry_widx,
                eri_task,
                W_dev,
                stream=stream,
                threads=threads,
            )
            out_total += out_batch

        return SSSSDigestResult(basis=basis, shell_pairs=shell_pairs, Q=Q, tasks=tasks, out=out_total)


__all__ = ["SSSSDigestResult", "digest_ssss_shellpairs"]
