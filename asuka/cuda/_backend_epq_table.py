from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


def init_workspace_epq_table(
    *,
    cp,
    drt,
    drt_dev,
    state_dev,
    use_epq_table: bool,
    epq_streaming: bool,
    epq_build_device: bool,
    epq_build_j_tile: int,
    j_tile: int,
    norb: int,
    ncsf: int,
    threads_enum: int,
    epq_recompute_warp_coop: bool,
    dtype: Any,
    epq_indptr_dtype: str,
    epq_build_nthreads: int,
    ext: Any,
    build_device_tiled_fn: Any,
    build_device_fn: Any,
    build_host_fn: Any,
    indptr_dtype_resolver_fn: Any,
    as_indptr_array_fn: Any,
    as_pq_array_fn: Any,
    epq_i32_max_nnz: int,
) -> dict[str, Any]:
    """Build and normalize EPQ table for workspace init; preserves legacy behavior."""
    out_epq_table = None
    out_epq_build_device = bool(epq_build_device)
    out_epq_table_build_s = 0.0

    if not bool(use_epq_table):
        return {
            "epq_table": None,
            "epq_build_device": bool(out_epq_build_device),
            "epq_table_build_s": float(out_epq_table_build_s),
        }

    t0 = time.perf_counter()
    if bool(epq_streaming):
        out_epq_table = None
    elif bool(out_epq_build_device):
        # If extension lacks device-build entrypoints, fall back to host path.
        if ext is None or not hasattr(ext, "epq_contribs_many_count_allpairs_inplace_device"):
            out_epq_build_device = False

    if (not bool(epq_streaming)) and bool(out_epq_build_device):
        jt = int(epq_build_j_tile)
        if jt <= 0:
            jt = int(j_tile)
        n_pairs_epq = int(norb) * max(0, int(norb) - 1)
        counts_bytes_est = int(ncsf) * int(n_pairs_epq) * int(np.dtype(np.int32).itemsize)
        use_tiled_epq_build = bool(counts_bytes_est >= (512 * 1024 * 1024))
        try:
            if bool(use_tiled_epq_build):
                out_epq_table = build_device_tiled_fn(
                    drt,
                    drt_dev,
                    state_dev,
                    j_tile=int(jt),
                    build_tile=int(jt),
                    threads=int(threads_enum),
                    sync=True,
                    check_overflow=True,
                    use_cache=True,
                    recompute_warp_coop=bool(epq_recompute_warp_coop),
                    dtype=dtype,
                    indptr_dtype=str(epq_indptr_dtype),
                )
            else:
                out_epq_table = build_device_fn(
                    drt,
                    drt_dev,
                    state_dev,
                    j_tile=int(jt),
                    threads=int(threads_enum),
                    sync=True,
                    check_overflow=True,
                    use_cache=True,
                    recompute_warp_coop=bool(epq_recompute_warp_coop),
                    dtype=dtype,
                    indptr_dtype=str(epq_indptr_dtype),
                )
        except AttributeError:
            # Likely an out-of-date extension build; fall back to host path.
            out_epq_build_device = False

    if (not bool(epq_streaming)) and (not bool(out_epq_build_device)):
        nt = int(epq_build_nthreads)
        if nt <= 0:
            # Heuristic: keep modest defaults on shared login nodes.
            try:
                nt = max(1, min(8, int(os.cpu_count() or 1)))
            except Exception:  # pragma: no cover
                nt = 1
        indptr_h, indices_h, pq_ids_h, data_h = build_host_fn(drt, precompute_nthreads=int(nt))
        indptr_cp_dtype = indptr_dtype_resolver_fn(
            cp,
            mode=str(epq_indptr_dtype),
            total_nnz=int(indptr_h[-1]) if int(indptr_h.size) > 0 else 0,
        )
        out_epq_table = (
            cp.ascontiguousarray(cp.asarray(indptr_h, dtype=indptr_cp_dtype)),
            cp.ascontiguousarray(cp.asarray(indices_h, dtype=cp.int32)),
            cp.ascontiguousarray(cp.asarray(pq_ids_h, dtype=cp.int32)),
            cp.ascontiguousarray(cp.asarray(data_h, dtype=dtype)),
        )

    if out_epq_table is not None:
        epq_indptr, epq_indices, epq_pq, epq_data = out_epq_table
        epq_indptr = as_indptr_array_fn(cp, epq_indptr, ncsf=int(ncsf), name="epq_indptr")
        if str(epq_indptr_dtype) != "auto":
            want_dtype = cp.int32 if str(epq_indptr_dtype) == "int32" else cp.int64
            if cp.dtype(epq_indptr.dtype) != cp.dtype(want_dtype):
                total_nnz = int(cp.asnumpy(epq_indptr[-1])) if int(epq_indptr.size) > 0 else 0
                if cp.dtype(want_dtype) == cp.dtype(cp.int32) and total_nnz > int(epq_i32_max_nnz):
                    raise RuntimeError(
                        f"Cannot cast EPQ indptr to int32: total_nnz={total_nnz} exceeds {int(epq_i32_max_nnz)}"
                    )
                epq_indptr = cp.ascontiguousarray(epq_indptr.astype(want_dtype, copy=False))
        out_epq_table = (
            epq_indptr,
            cp.ascontiguousarray(cp.asarray(epq_indices, dtype=cp.int32)),
            as_pq_array_fn(cp, epq_pq, name="epq_pq"),
            cp.ascontiguousarray(cp.asarray(epq_data, dtype=dtype)),
        )

    if not bool(epq_streaming):
        cp.cuda.get_current_stream().synchronize()
        out_epq_table_build_s = float(time.perf_counter() - t0)

    return {
        "epq_table": out_epq_table,
        "epq_build_device": bool(out_epq_build_device),
        "epq_table_build_s": float(out_epq_table_build_s),
    }
