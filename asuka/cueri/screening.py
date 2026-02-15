from __future__ import annotations

import numpy as np

from .cart import ncart
from .eri_dispatch import KernelBatch, plan_kernel_batches_spd, run_kernel_batch_spd
from .stream import stream_ctx
from .gpu import (
    CUDA_MAX_L,
    CUDA_MAX_NROOTS,
    build_pair_tables_ss_device,
    eri_pppp_device,
    eri_psps_device,
    eri_ssss_device,
    to_device_basis_ss,
    to_device_shell_pairs,
)
from .shell_pairs import ShellPairs, build_shell_pairs_l_order
from .tasks import TaskList, decode_eri_class_id


def _pair_key(sp_A: np.ndarray, sp_B: np.ndarray, *, n_shell: int) -> np.ndarray:
    hi = np.maximum(sp_A.astype(np.int64), sp_B.astype(np.int64))
    lo = np.minimum(sp_A.astype(np.int64), sp_B.astype(np.int64))
    return hi * int(n_shell) + lo


def schwarz_sp_device(
    basis,
    shell_pairs: ShellPairs,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
):
    """Rigorous Schwarz bounds for s/p-only shell pairs on GPU.

    Computes, for each shell pair AB:
      Q_AB = sqrt( max_{μ in A, ν in B} (μν|μν) )

    Notes
    - Uses Step-2 baseline kernels (ssss / psps / pppp) to evaluate (AB|AB) diagonals.
    - The returned Q is aligned to the input `shell_pairs` order (A/B orientation does not matter).
    """

    import cupy as cp

    with stream_ctx(stream):
        shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
        if shell_l.size == 0:
            return cp.empty((0,), dtype=cp.float64)
        if int(shell_l.max()) > 1:
            raise NotImplementedError("schwarz_sp_device currently supports only l<=1 (s/p) shells")

        # Evaluate Q on an l-ordered shell-pair list to match available class kernels.
        sp_l = build_shell_pairs_l_order(basis)
        nsp_l = int(sp_l.sp_A.shape[0])
        if nsp_l == 0:
            return cp.empty((0,), dtype=cp.float64)

        dbasis = to_device_basis_ss(basis)
        dsp_l = to_device_shell_pairs(sp_l)
        pt_l = build_pair_tables_ss_device(dbasis, dsp_l, stream=stream, threads=threads)

        la = shell_l[np.asarray(sp_l.sp_A, dtype=np.int32)]
        lb = shell_l[np.asarray(sp_l.sp_B, dtype=np.int32)]

        idx_ss = np.nonzero((la == 0) & (lb == 0))[0].astype(np.int32, copy=False)
        idx_ps = np.nonzero((la == 1) & (lb == 0))[0].astype(np.int32, copy=False)
        idx_pp = np.nonzero((la == 1) & (lb == 1))[0].astype(np.int32, copy=False)

        Q_l = cp.zeros((nsp_l,), dtype=cp.float64)

        if int(idx_ss.size) > 0:
            tasks = TaskList(task_spAB=idx_ss, task_spCD=idx_ss)
            val = eri_ssss_device(
                tasks,
                dsp_l,
                pt_l,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
                boys=boys,
            )
            Q_l[idx_ss] = cp.sqrt(cp.maximum(val, 0.0))

        if int(idx_ps.size) > 0:
            tasks = TaskList(task_spAB=idx_ps, task_spCD=idx_ps)
            tile = eri_psps_device(
                tasks,
                dbasis,
                dsp_l,
                pt_l,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            ).reshape((-1, 3, 3))
            diag = cp.diagonal(tile, axis1=1, axis2=2)
            Q_l[idx_ps] = cp.sqrt(cp.maximum(cp.max(diag, axis=1), 0.0))

        if int(idx_pp.size) > 0:
            tasks = TaskList(task_spAB=idx_pp, task_spCD=idx_pp)
            tile = eri_pppp_device(
                tasks,
                dbasis,
                dsp_l,
                pt_l,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            ).reshape((-1, 9, 9))
            diag = cp.diagonal(tile, axis1=1, axis2=2)
            Q_l[idx_pp] = cp.sqrt(cp.maximum(cp.max(diag, axis=1), 0.0))

        # Map Q back to the caller shell_pairs order (unordered shell pair key).
        n_shell = int(shell_l.shape[0])
        key_l = _pair_key(np.asarray(sp_l.sp_A), np.asarray(sp_l.sp_B), n_shell=n_shell)
        perm_l = np.argsort(key_l, kind="stable")
        key_l_sorted = key_l[perm_l]

        key_in = _pair_key(np.asarray(shell_pairs.sp_A), np.asarray(shell_pairs.sp_B), n_shell=n_shell)
        pos = np.searchsorted(key_l_sorted, key_in, side="left")
        if np.any(pos < 0) or np.any(pos >= key_l_sorted.size) or np.any(key_l_sorted[pos] != key_in):
            raise ValueError("shell_pairs does not match the basis shell set (missing unordered pair in l-ordered map)")
        idx_map = np.asarray(perm_l[pos], dtype=np.int32)
        return Q_l[idx_map]


def schwarz_shellpairs_device(
    basis,
    shell_pairs: ShellPairs,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
    max_tiles_bytes: int = 256 << 20,
):
    """Rigorous Schwarz bounds Q_AB for general shell pairs on GPU.

    Computes, for each shell pair AB:
      Q_AB = sqrt( max_{μ in A, ν in B} (μν|μν) )

    Implementation details
    - Evaluates (AB|AB) tiles and extracts max diagonal.
    - Uses the central dispatch to select existing class kernels when available,
      otherwise falls back to the generic Rys microkernel.
    - To control temporary memory, tasks are processed in chunks sized by `max_tiles_bytes`.

    Current limitation
    - CUDA backend supports per-shell angular momentum l<=CUDA_MAX_L
      (nroots<=CUDA_MAX_NROOTS).
    """

    import cupy as cp
    with stream_ctx(stream):
        shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
        if shell_l.size == 0:
            return cp.empty((0,), dtype=cp.float64)

        lmax = int(shell_l.max())
        if lmax > CUDA_MAX_L:
            raise NotImplementedError(f"schwarz_shellpairs_device currently supports only l<={CUDA_MAX_L} per shell")
        if max_tiles_bytes <= 0:
            raise ValueError("max_tiles_bytes must be > 0")

        # Evaluate Q on an l-ordered shell-pair list to maximize use of class-specialized kernels.
        sp_l = build_shell_pairs_l_order(basis)
        nsp_l = int(sp_l.sp_A.shape[0])
        if nsp_l == 0:
            return cp.empty((0,), dtype=cp.float64)

        dbasis = to_device_basis_ss(basis)
        dsp_l = to_device_shell_pairs(sp_l)
        pt_l = build_pair_tables_ss_device(dbasis, dsp_l, stream=stream, threads=threads)

        idx = np.arange(nsp_l, dtype=np.int32)
        tasks = TaskList(task_spAB=idx, task_spCD=idx)

        Q_l = cp.empty((nsp_l,), dtype=cp.float64)
        batches = plan_kernel_batches_spd(tasks, shell_pairs=sp_l, shell_l=shell_l)
        for batch in batches:
            la, lb, lc, ld = decode_eri_class_id(int(batch.kernel_class_id))
            nAB = int(ncart(int(la))) * int(ncart(int(lb)))
            nCD = int(ncart(int(lc))) * int(ncart(int(ld)))
            tile_elems = int(nAB * nCD)
            bytes_per_task = 8 * tile_elems
            chunk_ntasks = int(max(1, int(max_tiles_bytes) // max(bytes_per_task, 1)))

            nbt = int(batch.task_idx.shape[0])
            for i0 in range(0, nbt, chunk_ntasks):
                i1 = min(nbt, i0 + chunk_ntasks)
                sub = KernelBatch(
                    task_idx=np.asarray(batch.task_idx[i0:i1], dtype=np.int32),
                    kernel_tasks=TaskList(
                        task_spAB=np.asarray(batch.kernel_tasks.task_spAB[i0:i1], dtype=np.int32),
                        task_spCD=np.asarray(batch.kernel_tasks.task_spCD[i0:i1], dtype=np.int32),
                    ),
                    kernel_class_id=batch.kernel_class_id,
                    transpose=batch.transpose,
                )
                tile = run_kernel_batch_spd(
                    sub,
                    dbasis=dbasis,
                    dsp=dsp_l,
                    pt=pt_l,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                    boys=boys,
                )
                diag = cp.diagonal(tile, axis1=1, axis2=2)
                Q_l[sub.task_idx] = cp.sqrt(cp.maximum(cp.max(diag, axis=1), 0.0))

        # Map Q back to the caller shell_pairs order (unordered shell pair key).
        n_shell = int(shell_l.shape[0])
        key_l = _pair_key(np.asarray(sp_l.sp_A), np.asarray(sp_l.sp_B), n_shell=n_shell)
        perm_l = np.argsort(key_l, kind="stable")
        key_l_sorted = key_l[perm_l]

        key_in = _pair_key(np.asarray(shell_pairs.sp_A), np.asarray(shell_pairs.sp_B), n_shell=n_shell)
        pos = np.searchsorted(key_l_sorted, key_in, side="left")
        if np.any(pos < 0) or np.any(pos >= key_l_sorted.size) or np.any(key_l_sorted[pos] != key_in):
            raise ValueError("shell_pairs does not match the basis shell set (missing unordered pair in l-ordered map)")
        idx_map = np.asarray(perm_l[pos], dtype=np.int32)
        return Q_l[idx_map]


__all__ = ["schwarz_sp_device", "schwarz_shellpairs_device"]
