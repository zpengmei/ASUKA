from __future__ import annotations

import time
from typing import Any

import numpy as np


def csr_host_entry_bytes(*, nrows: int, nnz: int, data_itemsize: int) -> int:
    return int(nrows) * 4 + int(nrows) * 4 + (int(nrows) + 1) * 8 + int(nnz) * 4 + int(nnz) * int(data_itemsize)


def csr_host_cache_try_admit(workspace: Any, *, tile_bytes: int, score: float) -> bool:
    budget = int(getattr(workspace, "csr_host_cache_budget_bytes", 0))
    if int(tile_bytes) <= 0 or budget <= 0:
        return False
    if int(tile_bytes) > int(budget):
        return False
    if int(workspace._csr_host_cache_bytes) + int(tile_bytes) <= int(budget):
        return True

    entries = sorted(
        workspace._csr_host_tile_cache.items(),
        key=lambda kv: (float(kv[1].get("score", 0.0)), int(kv[1].get("bytes", 0))),
    )
    victims: list[int] = []
    freed = 0
    min_victim_score = float("inf")
    for j0, ent in entries:
        victims.append(int(j0))
        ent_bytes = int(ent.get("bytes", 0))
        freed += max(0, ent_bytes)
        min_victim_score = min(min_victim_score, float(ent.get("score", 0.0)))
        if int(workspace._csr_host_cache_bytes) - int(freed) + int(tile_bytes) <= int(budget):
            break

    if int(workspace._csr_host_cache_bytes) - int(freed) + int(tile_bytes) > int(budget):
        return False
    if victims and float(score) <= float(min_victim_score):
        return False

    for j0 in victims:
        old = workspace._csr_host_tile_cache.pop(int(j0), None)
        if old is not None:
            workspace._csr_host_cache_bytes -= int(old.get("bytes", 0))
            workspace._csr_host_cache_bytes = max(0, int(workspace._csr_host_cache_bytes))
            workspace._csr_host_cache_evictions += 1
    return True


def csr_host_cache_store_tile(
    workspace: Any,
    *,
    j0: int,
    row_j_d,
    row_k_d,
    indptr_d,
    indices_d,
    data_d,
    nrows: int,
    nnz: int,
    stream,
    profile: dict[str, float] | None,
) -> None:
    if not bool(getattr(workspace, "csr_host_cache_enabled", False)):
        return
    if int(nrows) <= 0:
        return
    j0_i = int(j0)
    if j0_i in workspace._csr_host_tile_cache:
        return

    workspace._csr_host_cache_store_attempts += 1
    data_itemsize = int(np.dtype(getattr(data_d, "dtype", np.float64)).itemsize)
    tile_bytes = csr_host_entry_bytes(nrows=int(nrows), nnz=int(nnz), data_itemsize=data_itemsize)
    score = float(nnz)
    if not csr_host_cache_try_admit(workspace, tile_bytes=int(tile_bytes), score=float(score)):
        return

    t0 = time.perf_counter() if profile is not None else None
    row_j_mem, row_j_h = workspace._alloc_pinned_np((int(nrows),), np.int32)
    row_k_mem, row_k_h = workspace._alloc_pinned_np((int(nrows),), np.int32)
    indptr_mem, indptr_h = workspace._alloc_pinned_np((int(nrows) + 1,), np.int64)
    indices_mem, indices_h = workspace._alloc_pinned_np((int(nnz),), np.int32)
    data_mem, data_h = workspace._alloc_pinned_np((int(nnz),), np.dtype(getattr(data_d, "dtype", np.float64)))

    import cupy as cp

    cp.cuda.runtime.memcpyAsync(
        int(row_j_h.ctypes.data),
        int(row_j_d.data.ptr),
        int(row_j_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(row_k_h.ctypes.data),
        int(row_k_d.data.ptr),
        int(row_k_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indptr_h.ctypes.data),
        int(indptr_d.data.ptr),
        int(indptr_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indices_h.ctypes.data),
        int(indices_d.data.ptr),
        int(indices_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(data_h.ctypes.data),
        int(data_d.data.ptr),
        int(data_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )

    workspace._csr_host_tile_cache[j0_i] = {
        "row_j_mem": row_j_mem,
        "row_j": row_j_h,
        "row_k_mem": row_k_mem,
        "row_k": row_k_h,
        "indptr_mem": indptr_mem,
        "indptr": indptr_h,
        "indices_mem": indices_mem,
        "indices": indices_h,
        "data_mem": data_mem,
        "data": data_h,
        "nrows": int(nrows),
        "nnz": int(nnz),
        "bytes": int(tile_bytes),
        "score": float(score),
    }
    workspace._csr_host_cache_bytes += int(tile_bytes)
    workspace._csr_host_cache_store_accepts += 1
    if profile is not None and t0 is not None:
        stream.synchronize()
        profile["csr_host_cache_store_s"] = profile.get("csr_host_cache_store_s", 0.0) + (time.perf_counter() - t0)


def csr_host_cache_load_tile(workspace: Any, *, j0: int, stream, profile: dict[str, float] | None):
    ent = workspace._csr_host_tile_cache.get(int(j0))
    if ent is None:
        workspace._csr_host_cache_misses += 1
        if profile is not None:
            profile["csr_host_cache_misses"] = profile.get("csr_host_cache_misses", 0.0) + 1.0
        return None

    nrows = int(ent["nrows"])
    nnz = int(ent["nnz"])
    workspace._ensure_csr_staging_capacity(max(int(nnz), int(nrows)))
    row_j_d = workspace._csr_row_j[:nrows]
    row_k_d = workspace._csr_row_k[:nrows]
    indptr_d = workspace._csr_indptr[: nrows + 1]
    indices_d = workspace._csr_indices[:nnz]
    data_d = workspace._csr_data[:nnz]

    import cupy as cp

    t0 = time.perf_counter() if profile is not None else None
    cp.cuda.runtime.memcpyAsync(
        int(row_j_d.data.ptr),
        int(ent["row_j"].ctypes.data),
        int(ent["row_j"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(row_k_d.data.ptr),
        int(ent["row_k"].ctypes.data),
        int(ent["row_k"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indptr_d.data.ptr),
        int(ent["indptr"].ctypes.data),
        int(ent["indptr"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indices_d.data.ptr),
        int(ent["indices"].ctypes.data),
        int(ent["indices"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(data_d.data.ptr),
        int(ent["data"].ctypes.data),
        int(ent["data"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    workspace._csr_host_cache_hits += 1
    if profile is not None:
        profile["csr_host_cache_hits"] = profile.get("csr_host_cache_hits", 0.0) + 1.0
    if profile is not None and t0 is not None:
        stream.synchronize()
        profile["csr_host_cache_load_s"] = profile.get("csr_host_cache_load_s", 0.0) + (time.perf_counter() - t0)

    return row_j_d, row_k_d, indptr_d, indices_d, data_d, int(nrows), int(nnz)


def epq_apply_cache_store(
    workspace: Any,
    *,
    k0: int,
    indptr_d,
    indices_d,
    pq_ids_d,
    data_d,
    j_count: int,
    nnz: int,
    stream,
) -> None:
    if not bool(getattr(workspace, "epq_apply_cache_enabled", False)):
        return
    k0_i = int(k0)
    if k0_i in workspace._epq_apply_tile_cache:
        return
    if int(nnz) <= 0:
        return

    import cupy as cp

    data_itemsize = int(np.dtype(getattr(data_d, "dtype", np.float64)).itemsize)
    pq_itemsize = int(np.dtype(getattr(pq_ids_d, "dtype", np.uint8)).itemsize)
    tile_bytes = (int(j_count) + 1) * 8 + int(nnz) * 4 + int(nnz) * pq_itemsize + int(nnz) * data_itemsize
    budget = int(getattr(workspace, "epq_apply_cache_budget_bytes", 0))
    if budget <= 0 or int(tile_bytes) > int(budget):
        return
    if int(workspace._epq_apply_cache_bytes) + int(tile_bytes) > int(budget):
        return

    indptr_mem, indptr_h = workspace._alloc_pinned_np((int(j_count) + 1,), np.int64)
    indices_mem, indices_h = workspace._alloc_pinned_np((int(nnz),), np.int32)
    pq_dtype_np = np.dtype(getattr(pq_ids_d, "dtype", np.uint8))
    pq_mem, pq_h = workspace._alloc_pinned_np((int(nnz),), pq_dtype_np)
    data_dtype_np = np.dtype(getattr(data_d, "dtype", np.float64))
    data_mem, data_h = workspace._alloc_pinned_np((int(nnz),), data_dtype_np)

    cp.cuda.runtime.memcpyAsync(
        int(indptr_h.ctypes.data),
        int(indptr_d.data.ptr),
        int(indptr_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indices_h.ctypes.data),
        int(indices_d.data.ptr),
        int(indices_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(pq_h.ctypes.data),
        int(pq_ids_d.data.ptr),
        int(pq_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(data_h.ctypes.data),
        int(data_d.data.ptr),
        int(data_h.nbytes),
        int(cp.cuda.runtime.memcpyDeviceToHost),
        int(stream.ptr),
    )

    workspace._epq_apply_tile_cache[k0_i] = {
        "indptr_mem": indptr_mem,
        "indptr": indptr_h,
        "indices_mem": indices_mem,
        "indices": indices_h,
        "pq_mem": pq_mem,
        "pq_ids": pq_h,
        "data_mem": data_mem,
        "data": data_h,
        "j_count": int(j_count),
        "nnz": int(nnz),
        "bytes": int(tile_bytes),
    }
    workspace._epq_apply_cache_bytes += int(tile_bytes)


def epq_apply_ensure_staging(
    workspace: Any,
    *,
    j_count: int,
    nnz: int,
    epq_pq_dtype_for_norb_fn: Any,
) -> None:
    import cupy as cp

    needed = max(int(nnz), 1)
    indptr_needed = int(j_count) + 1
    if int(workspace._epq_apply_staging_capacity) >= int(needed) and workspace._epq_apply_staging_indptr is not None:
        if workspace._epq_apply_staging_indptr.shape[0] >= indptr_needed:
            return

    pq_dtype = epq_pq_dtype_for_norb_fn(cp, int(workspace.norb))
    workspace._epq_apply_staging_indptr = cp.empty((int(indptr_needed),), dtype=cp.int64)
    workspace._epq_apply_staging_indices = cp.empty((int(needed),), dtype=cp.int32)
    workspace._epq_apply_staging_pq_ids = cp.empty((int(needed),), dtype=pq_dtype)
    workspace._epq_apply_staging_data = cp.empty((int(needed),), dtype=workspace._dtype)
    workspace._epq_apply_staging_capacity = int(needed)


def epq_apply_cache_load(
    workspace: Any,
    *,
    k0: int,
    stream,
    epq_pq_dtype_for_norb_fn: Any,
):
    ent = workspace._epq_apply_tile_cache.get(int(k0))
    if ent is None:
        workspace._epq_apply_cache_misses += 1
        return None

    import cupy as cp

    j_count = int(ent["j_count"])
    nnz = int(ent["nnz"])
    epq_apply_ensure_staging(
        workspace,
        j_count=j_count,
        nnz=nnz,
        epq_pq_dtype_for_norb_fn=epq_pq_dtype_for_norb_fn,
    )

    indptr_d = workspace._epq_apply_staging_indptr[: j_count + 1]
    indices_d = workspace._epq_apply_staging_indices[:nnz]
    pq_ids_d = workspace._epq_apply_staging_pq_ids[:nnz]
    data_d = workspace._epq_apply_staging_data[:nnz]

    cp.cuda.runtime.memcpyAsync(
        int(indptr_d.data.ptr),
        int(ent["indptr"].ctypes.data),
        int(ent["indptr"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(indices_d.data.ptr),
        int(ent["indices"].ctypes.data),
        int(ent["indices"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(pq_ids_d.data.ptr),
        int(ent["pq_ids"].ctypes.data),
        int(ent["pq_ids"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    cp.cuda.runtime.memcpyAsync(
        int(data_d.data.ptr),
        int(ent["data"].ctypes.data),
        int(ent["data"].nbytes),
        int(cp.cuda.runtime.memcpyHostToDevice),
        int(stream.ptr),
    )
    workspace._epq_apply_cache_hits += 1
    return indptr_d, indices_d, pq_ids_d, data_d
