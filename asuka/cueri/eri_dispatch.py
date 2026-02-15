from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from .cart import ncart
from .gpu import (
    DeviceBasisSS,
    DevicePairTables,
    DeviceShellPairs,
    eri_dddd_device,
    eri_dpdd_device,
    eri_dpdp_device,
    eri_dpfs_device,
    eri_dpgs_device,
    eri_fdps_device,
    eri_fdss_device,
    eri_fdds_device,
    eri_fdgs_device,
    eri_fdfs_device,
    eri_ffps_device,
    eri_ffss_device,
    eri_ffds_device,
    eri_ffgs_device,
    eri_fffs_device,
    eri_dsdd_device,
    eri_dsdp_device,
    eri_dsds_device,
    eri_dsfs_device,
    eri_dsgs_device,
    eri_ddfs_device,
    eri_ddgs_device,
    eri_ddss_device,
    eri_dsss_device,
    eri_fpds_device,
    eri_fpfs_device,
    eri_fpgs_device,
    eri_fpps_device,
    eri_fpss_device,
    eri_fsfs_device,
    eri_fsgs_device,
    eri_ppdd_device,
    eri_ppfs_device,
    eri_ppgs_device,
    eri_ppdp_device,
    eri_ppds_device,
    eri_pppp_device,
    eri_ppps_device,
    eri_ppss_device,
    eri_psdd_device,
    eri_psfs_device,
    eri_psgs_device,
    eri_psdp_device,
    eri_psds_device,
    eri_psps_device,
    eri_psss_device,
    eri_rys_df_ld0_warp_device,
    eri_rys_generic_device,
    eri_rys_generic_warp_device,
    eri_ssfs_device,
    eri_ssgs_device,
    eri_ssdp_device,
    eri_ssss_device,
)
from .stream import stream_ctx
from .tasks import TaskList, decode_eri_class_id, eri_class_id, with_task_class_id


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


# Native ff* kernels are available, but generic remains faster on common
# cc-pVTZ DF workloads; keep this as an opt-in experiment switch.
_USE_NATIVE_FF_QUARTETS = _env_bool("ASUKA_CUERI_USE_NATIVE_FF_QUARTETS", False)
_GENERIC_WARP_CAP_SMALL = max(1, _env_int("ASUKA_CUERI_GENERIC_WARP_CAP_SMALL", 4))
_GENERIC_WARP_CAP_LARGE = max(1, _env_int("ASUKA_CUERI_GENERIC_WARP_CAP_LARGE", 4))
_GENERIC_WARP_CAP_THRESHOLD = max(1, _env_int("ASUKA_CUERI_GENERIC_WARP_CAP_THRESHOLD", 1024))


@dataclass(frozen=True)
class KernelBatch:
    task_idx: np.ndarray  # int32, indices into original task list
    kernel_tasks: TaskList
    kernel_class_id: np.int32
    transpose: bool


def _choose_threads_rys_generic(*, n_elem: int, threads: int) -> int:
    """Heuristic thread count for the generic Rys kernel.

    The generic kernel uses `__syncthreads()` heavily while only a subset of
    threads are active for small tiles. Using fewer threads (fewer warps)
    reduces barrier overhead and improves throughput for small nElem.
    """

    n_elem = int(n_elem)
    threads = int(threads)
    if n_elem <= 0:
        return max(32, threads)
    if threads <= 32:
        return max(32, threads)
    max_warps = max(1, threads // 32)
    need_warps = max(1, (n_elem + 31) // 32)
    # Empirically, the generic kernel is often sync/register limited for small
    # tiles, but larger tiles benefit from more active warps per block.
    warp_cap = _GENERIC_WARP_CAP_LARGE if n_elem >= _GENERIC_WARP_CAP_THRESHOLD else _GENERIC_WARP_CAP_SMALL
    warps = min(max_warps, need_warps, int(warp_cap))
    return int(32 * warps)


def _am_label(l: int) -> str:
    am = "spdfghijklm"
    li = int(l)
    if 0 <= li < len(am):
        return am[li]
    return f"l{li}"


def plan_kernel_batches_spd(tasks: TaskList, *, shell_pairs, shell_l: np.ndarray) -> list[KernelBatch]:
    """Organize tasks into optimized kernel batches for Step-2 execution.

    Analyses the input tasks and groups them into batches that can be executed
    efficiently by specific CUDA kernels. It handles mapping of general shell
    quartets to available specialized kernels (e.g., swapping bra/ket pairs
    to match available kernels) or falling back to generic kernels.

    Parameters
    ----------
    tasks : TaskList
        The list of shell quartet tasks to schedule.
    shell_pairs : ShellPairs | DeviceShellPairs
        Shell pair definitions.
    shell_l : np.ndarray
        Angular momenta for shells.

    Returns
    -------
    list[KernelBatch]
        A list of `KernelBatch` objects representing the execution plan, grouped
        by kernel type and transposition requirements.
    """

    if tasks.ntasks == 0:
        return []

    if tasks.task_class_id is None:
        tasks = with_task_class_id(tasks, shell_pairs, shell_l)

    base: dict[int, str] = {
        int(eri_class_id(0, 0, 0, 0)): "ssss",
        int(eri_class_id(1, 0, 0, 0)): "psss",
        int(eri_class_id(1, 1, 0, 0)): "ppss",
        int(eri_class_id(1, 0, 1, 0)): "psps",
        int(eri_class_id(1, 1, 1, 0)): "ppps",
        int(eri_class_id(1, 1, 1, 1)): "pppp",
        int(eri_class_id(2, 0, 0, 0)): "dsss",
        int(eri_class_id(2, 2, 0, 0)): "ddss",
        int(eri_class_id(0, 0, 2, 1)): "ssdp",
        int(eri_class_id(1, 0, 2, 0)): "psds",
        int(eri_class_id(1, 0, 2, 1)): "psdp",
        int(eri_class_id(1, 0, 2, 2)): "psdd",
        int(eri_class_id(1, 1, 2, 0)): "ppds",
        int(eri_class_id(1, 1, 2, 1)): "ppdp",
        int(eri_class_id(1, 1, 2, 2)): "ppdd",
        int(eri_class_id(2, 0, 2, 0)): "dsds",
        int(eri_class_id(2, 0, 2, 1)): "dsdp",
        int(eri_class_id(2, 0, 2, 2)): "dsdd",
        int(eri_class_id(3, 1, 0, 0)): "fpss",
        int(eri_class_id(3, 2, 0, 0)): "fdss",
        int(eri_class_id(3, 3, 0, 0)): "ffss",
        int(eri_class_id(3, 1, 1, 0)): "fpps",
        int(eri_class_id(3, 2, 1, 0)): "fdps",
        int(eri_class_id(3, 3, 1, 0)): "ffps",
        int(eri_class_id(3, 1, 2, 0)): "fpds",
        int(eri_class_id(3, 2, 2, 0)): "fdds",
        int(eri_class_id(3, 3, 2, 0)): "ffds",
        int(eri_class_id(0, 0, 3, 0)): "ssfs",
        int(eri_class_id(1, 0, 3, 0)): "psfs",
        int(eri_class_id(1, 1, 3, 0)): "ppfs",
        int(eri_class_id(2, 0, 3, 0)): "dsfs",
        int(eri_class_id(3, 0, 3, 0)): "fsfs",
        int(eri_class_id(2, 1, 3, 0)): "dpfs",
        int(eri_class_id(3, 1, 3, 0)): "fpfs",
        int(eri_class_id(2, 2, 3, 0)): "ddfs",
        int(eri_class_id(3, 2, 3, 0)): "fdfs",
        int(eri_class_id(3, 3, 3, 0)): "fffs",
        int(eri_class_id(0, 0, 4, 0)): "ssgs",
        int(eri_class_id(1, 0, 4, 0)): "psgs",
        int(eri_class_id(1, 1, 4, 0)): "ppgs",
        int(eri_class_id(2, 0, 4, 0)): "dsgs",
        int(eri_class_id(3, 0, 4, 0)): "fsgs",
        int(eri_class_id(2, 1, 4, 0)): "dpgs",
        int(eri_class_id(3, 1, 4, 0)): "fpgs",
        int(eri_class_id(2, 2, 4, 0)): "ddgs",
        int(eri_class_id(3, 2, 4, 0)): "fdgs",
        int(eri_class_id(3, 3, 4, 0)): "ffgs",
        int(eri_class_id(2, 1, 2, 1)): "dpdp",
        int(eri_class_id(2, 1, 2, 2)): "dpdd",
        int(eri_class_id(2, 2, 2, 2)): "dddd",
    }
    _ = base  # mapping is used only as a set

    cid = np.asarray(tasks.task_class_id, dtype=np.int32).ravel()
    nt = int(cid.shape[0])

    # Vectorized mapping from task class -> (kernel class, transpose flag).
    #
    # This is hot for large screened workloads; avoid per-task Python loops.
    base_ids = np.asarray(list(base.keys()), dtype=np.int32)

    in_base = np.zeros((nt,), dtype=bool)
    for b in base_ids:
        in_base |= cid == np.int32(b)

    cid_u32 = cid.astype(np.uint32, copy=False)
    la = cid_u32 & np.uint32(0xFF)
    lb = (cid_u32 >> np.uint32(8)) & np.uint32(0xFF)
    lc = (cid_u32 >> np.uint32(16)) & np.uint32(0xFF)
    ld = (cid_u32 >> np.uint32(24)) & np.uint32(0xFF)
    swap_cid = (lc | (ld << np.uint32(8)) | (la << np.uint32(16)) | (lb << np.uint32(24))).astype(np.int32, copy=False)

    in_swap_base = np.zeros((nt,), dtype=bool)
    for b in base_ids:
        in_swap_base |= swap_cid == np.int32(b)

    swap_mask = (~in_base) & in_swap_base

    kernel_cid = cid.astype(np.int32, copy=True)
    tr = np.zeros((nt,), dtype=np.int32)
    if bool(np.any(swap_mask)):
        kernel_cid[swap_mask] = swap_cid[swap_mask]
        tr[swap_mask] = 1

    # Pack (kernel_class_id, transpose_flag) into a single sortable key.
    #
    # Use uint32->uint64 to avoid sign-extension pitfalls if kernel_cid ever
    # becomes negative in int32 (high bit set).
    tag = kernel_cid.astype(np.uint32).astype(np.uint64) | (tr.astype(np.uint64) << 32)
    perm = np.asarray(np.argsort(tag, kind="stable"), dtype=np.int32)
    tag_sorted = tag[perm]
    changes = np.nonzero(tag_sorted[1:] != tag_sorted[:-1])[0] + 1
    offsets = np.concatenate(([0], changes, [nt])).astype(np.int32)

    batches: list[KernelBatch] = []
    for g in range(int(offsets.shape[0] - 1)):
        i0 = int(offsets[g])
        i1 = int(offsets[g + 1])
        if i1 <= i0:
            continue
        idx = perm[i0:i1]
        kcid = int(kernel_cid[int(idx[0])])
        transpose = bool(tr[int(idx[0])])
        if not transpose:
            ktasks = TaskList(task_spAB=tasks.task_spAB[idx], task_spCD=tasks.task_spCD[idx])
        else:
            ktasks = TaskList(task_spAB=tasks.task_spCD[idx], task_spCD=tasks.task_spAB[idx])
        batches.append(KernelBatch(task_idx=np.asarray(idx, dtype=np.int32), kernel_tasks=ktasks, kernel_class_id=np.int32(kcid), transpose=transpose))

    return batches


def run_kernel_batch_spd(
    batch: KernelBatch,
    *,
    dbasis: DeviceBasisSS,
    dsp: DeviceShellPairs,
    pt: DevicePairTables,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
    profile: dict | None = None,
):
    """Execute a single kernel batch on the GPU.

    Dispatches the appropriate CUDA kernel for the given batch of tasks. Handles
    native kernels (ssss, psss, etc.), generic Rys kernels, and necessary
    transpositions for swapped task classes.

    Parameters
    ----------
    batch : KernelBatch
        The batch of tasks to execute.
    dbasis : DeviceBasisSS
        Device basis set data.
    dsp : DeviceShellPairs
        Device shell pair data.
    pt : DevicePairTables
        Device pair tables.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream.
    threads : int, default=256
        Block size.
    mode : str, default='auto'
        Execution strategy ('block', 'warp', etc.).
    work_small_max : int, default=512
        Threshold for small work dispatch.
    work_large_min : int, default=200000
        Threshold for large work dispatch.
    blocks_per_task : int, default=8
        Blocks per task for multiblock kernels.
    boys : str, default='ref'
        Boys function implementation ('ref' or 'fast').
    profile : dict | None, optional
        Optional profile dictionary. When provided, records per-class kernel
        timing and task counts in this dict.

    Returns
    -------
    cupy.ndarray
        The computed integrals as a CuPy array of shape `(ntasks, nAB, nCD)`.
    """

    import cupy as cp

    with stream_ctx(stream):
        cid = int(batch.kernel_class_id)
        la, lb, lc, ld = decode_eri_class_id(cid)
        nAB = int(ncart(int(la))) * int(ncart(int(lb)))
        nCD = int(ncart(int(lc))) * int(ncart(int(ld)))
        dispatch_path = "generic"
        evt0 = cp.cuda.Event() if profile is not None else None
        evt1 = cp.cuda.Event() if profile is not None else None
        s0 = cp.cuda.get_current_stream() if profile is not None else None
        if evt0 is not None and s0 is not None:
            evt0.record(s0)

        def _run_generic_raw():
            nonlocal dispatch_path
            n_elem = int(nAB * nCD)
            mode_l = str(mode).lower().strip()
            use_ld0_warp = (
                int(ld) == 0
                and ((mode_l == "warp" and n_elem <= 256) or (mode_l == "auto" and n_elem <= 128))
                and int(threads) <= 256
                and (int(threads) % 32) == 0
            )
            if use_ld0_warp:
                dispatch_path = "generic_ld0_warp"
                threads_ld0 = int(min(int(threads), 128))
                threads_ld0 = int(max(32, (threads_ld0 // 32) * 32))
                return eri_rys_df_ld0_warp_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    la=int(la),
                    lb=int(lb),
                    lc=int(lc),
                    stream=stream,
                    threads=int(threads_ld0),
                )

            use_generic_warp = (
                (mode_l == "warp" or mode_l == "auto")
                and n_elem <= 128
                and int(threads) <= 256
                and (int(threads) % 32) == 0
            )
            if use_generic_warp:
                dispatch_path = "generic_warp"
                return eri_rys_generic_warp_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    la=int(la),
                    lb=int(lb),
                    lc=int(lc),
                    ld=int(ld),
                    stream=stream,
                    threads=int(threads),
                )

            dispatch_path = "generic_block"
            threads_generic = _choose_threads_rys_generic(n_elem=n_elem, threads=int(threads))
            return eri_rys_generic_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                la=int(la),
                lb=int(lb),
                lc=int(lc),
                ld=int(ld),
                stream=stream,
                threads=int(threads_generic),
            )

        if cid == int(eri_class_id(0, 0, 0, 0)):
            dispatch_path = "native_ssss"
            raw = eri_ssss_device(
                batch.kernel_tasks,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
                boys=boys,
            )
            tile = raw.reshape((-1, 1, 1))
        elif cid == int(eri_class_id(1, 0, 0, 0)):
            dispatch_path = "native_psss"
            raw = eri_psss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 0, 0)):
            dispatch_path = "native_ppss"
            raw = eri_ppss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 1, 0)):
            dispatch_path = "native_psps"
            raw = eri_psps_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 1, 0)):
            dispatch_path = "native_ppps"
            raw = eri_ppps_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 1, 1)):
            dispatch_path = "native_pppp"
            raw = eri_pppp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 0, 0)):
            dispatch_path = "native_dsss"
            raw = eri_dsss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 2, 0, 0)):
            dispatch_path = "native_ddss"
            raw = eri_ddss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(0, 0, 2, 1)):
            dispatch_path = "native_ssdp"
            raw = eri_ssdp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 2, 0)):
            dispatch_path = "native_psds"
            raw = eri_psds_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 2, 1)):
            dispatch_path = "native_psdp"
            raw = eri_psdp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 2, 2)):
            dispatch_path = "native_psdd"
            raw = eri_psdd_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 2, 0)):
            dispatch_path = "native_ppds"
            raw = eri_ppds_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 2, 1)):
            dispatch_path = "native_ppdp"
            raw = eri_ppdp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 2, 2)):
            dispatch_path = "native_ppdd"
            raw = eri_ppdd_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 2, 0)):
            dispatch_path = "native_dsds"
            raw = eri_dsds_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 2, 1)):
            dispatch_path = "native_dsdp"
            raw = eri_dsdp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 2, 2)):
            dispatch_path = "native_dsdd"
            raw = eri_dsdd_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 1, 0, 0)):
            dispatch_path = "native_fpss"
            raw = eri_fpss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 2, 0, 0)):
            dispatch_path = "native_fdss"
            raw = eri_fdss_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 3, 0, 0)):
            if _USE_NATIVE_FF_QUARTETS:
                dispatch_path = "native_ffss"
                raw = eri_ffss_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                )
            else:
                raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 1, 1, 0)):
            dispatch_path = "native_fpps"
            raw = eri_fpps_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 2, 1, 0)):
            dispatch_path = "native_fdps"
            raw = eri_fdps_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 3, 1, 0)):
            if _USE_NATIVE_FF_QUARTETS:
                dispatch_path = "native_ffps"
                raw = eri_ffps_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                )
            else:
                raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 1, 2, 0)):
            dispatch_path = "native_fpds"
            raw = eri_fpds_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 2, 2, 0)):
            dispatch_path = "native_fdds"
            raw = eri_fdds_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 3, 2, 0)):
            if _USE_NATIVE_FF_QUARTETS:
                dispatch_path = "native_ffds"
                raw = eri_ffds_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                )
            else:
                raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(0, 0, 3, 0)):
            dispatch_path = "native_ssfs"
            raw = eri_ssfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 3, 0)):
            dispatch_path = "native_psfs"
            raw = eri_psfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 3, 0)):
            dispatch_path = "native_ppfs"
            raw = eri_ppfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 3, 0)):
            dispatch_path = "native_dsfs"
            raw = eri_dsfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 0, 3, 0)):
            dispatch_path = "native_fsfs"
            raw = eri_fsfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 1, 3, 0)):
            dispatch_path = "native_dpfs"
            raw = eri_dpfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 1, 3, 0)):
            dispatch_path = "native_fpfs"
            raw = eri_fpfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 2, 3, 0)):
            dispatch_path = "native_ddfs"
            raw = eri_ddfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 2, 3, 0)):
            dispatch_path = "native_fdfs"
            raw = eri_fdfs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 3, 3, 0)):
            if _USE_NATIVE_FF_QUARTETS:
                dispatch_path = "native_fffs"
                raw = eri_fffs_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                )
            else:
                raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(0, 0, 4, 0)):
            dispatch_path = "native_ssgs"
            raw = eri_ssgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 0, 4, 0)):
            dispatch_path = "native_psgs"
            raw = eri_psgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(1, 1, 4, 0)):
            dispatch_path = "native_ppgs"
            raw = eri_ppgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 0, 4, 0)):
            dispatch_path = "native_dsgs"
            raw = eri_dsgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 0, 4, 0)):
            dispatch_path = "native_fsgs"
            raw = eri_fsgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 1, 4, 0)):
            dispatch_path = "native_dpgs"
            raw = eri_dpgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 1, 4, 0)):
            dispatch_path = "native_fpgs"
            raw = eri_fpgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 2, 4, 0)):
            dispatch_path = "native_ddgs"
            raw = eri_ddgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 2, 4, 0)):
            dispatch_path = "native_fdgs"
            raw = eri_fdgs_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(3, 3, 4, 0)):
            if _USE_NATIVE_FF_QUARTETS:
                dispatch_path = "native_ffgs"
                raw = eri_ffgs_device(
                    batch.kernel_tasks,
                    dbasis,
                    dsp,
                    pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                )
            else:
                raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 1, 2, 1)):
            dispatch_path = "native_dpdp"
            raw = eri_dpdp_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 1, 2, 2)):
            dispatch_path = "native_dpdd"
            raw = eri_dpdd_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        elif cid == int(eri_class_id(2, 2, 2, 2)):
            dispatch_path = "native_dddd"
            raw = eri_dddd_device(
                batch.kernel_tasks,
                dbasis,
                dsp,
                pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            tile = raw.reshape((-1, nAB, nCD))
        else:
            raw = _run_generic_raw()
            tile = raw.reshape((-1, nAB, nCD))

        if batch.transpose:
            tile = tile.transpose((0, 2, 1))
        tile = cp.ascontiguousarray(tile)

        if evt0 is not None and evt1 is not None and s0 is not None:
            evt1.record(s0)
            evt1.synchronize()
            dt_ms = float(cp.cuda.get_elapsed_time(evt0, evt1))
            profile["kernel_ms"] = float(profile.get("kernel_ms", 0.0)) + dt_ms
            cls = f"{_am_label(int(la))}{_am_label(int(lb))}{_am_label(int(lc))}{_am_label(int(ld))}"
            key = (
                f"{cls}|la={int(la)}|lb={int(lb)}|lc={int(lc)}|ld={int(ld)}|"
                f"tr={int(batch.transpose)}|path={dispatch_path}|nAB={int(nAB)}|nCD={int(nCD)}"
            )
            row = profile.setdefault("classes", {}).setdefault(key, {"ms": 0.0, "calls": 0, "ntasks": 0})
            row["ms"] = float(row.get("ms", 0.0)) + dt_ms
            row["calls"] = int(row.get("calls", 0)) + 1
            row["ntasks"] = int(row.get("ntasks", 0)) + int(tile.shape[0])

        return tile


__all__ = ["KernelBatch", "plan_kernel_batches_spd", "run_kernel_batch_spd"]
