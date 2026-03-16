from __future__ import annotations

import os
import time
from typing import Callable, Sequence

import numpy as np

from asuka.cuda._backend_caps import (
    device_info as _device_info_runtime,
    has_build_w_from_epq_transpose_range_mm as _has_build_w_from_epq_transpose_range_mm_runtime,
    has_build_w_from_epq_transpose_range_mm_scaled as _has_build_w_from_epq_transpose_range_mm_scaled_runtime,
    has_cuda_ext as _has_cuda_ext_runtime,
    has_epq_table_device_build as _has_epq_table_device_build_runtime,
    has_epq_table_device_build_recompute as _has_epq_table_device_build_recompute_runtime,
    has_epq_table_gather_apply_device as _has_epq_table_gather_apply_device_runtime,
    has_t_from_epq_table_device_build as _has_t_from_epq_table_device_build_runtime,
    mem_info as _mem_info_runtime,
)
from asuka.cuda._backend_epq_config import (
    EPQ_I32_MAX_NNZ as _EPQ_I32_MAX_NNZ,
    epq_indptr_cp_dtype_for_total_nnz as _epq_indptr_cp_dtype_for_total_nnz,
    normalize_epq_blocked_transpose_mode as _normalize_epq_blocked_transpose_mode,
    normalize_epq_indptr_mode as _normalize_epq_indptr_mode,
    normalize_matvec_cuda_path_mode as _normalize_matvec_cuda_path_mode,
    resolve_epq_blocked_transpose_mode_with_env as _resolve_epq_blocked_transpose_mode_with_env,
    resolve_epq_blocked_transpose_reserve_mib_with_env as _resolve_epq_blocked_transpose_reserve_mib_with_env,
)
from asuka.cuda._backend_epq_table import (
    init_workspace_epq_table as _init_workspace_epq_table,
)
from asuka.cuda._backend_workspace_config import (
    resolve_apply_warp_coop as _resolve_apply_warp_coop,
    resolve_check_overflow_mode as _resolve_check_overflow_mode,
    resolve_epq_stream_double_buffer_mode as _resolve_epq_stream_double_buffer_mode,
    resolve_epq_stream_panic_mode as _resolve_epq_stream_panic_mode,
    resolve_epq_stream_use_recompute as _resolve_epq_stream_use_recompute,
)
from asuka.cuda._backend_workspace_policy import (
    resolve_cache_csr_tiles as _resolve_cache_csr_tiles,
    resolve_csr_host_cache_enabled as _resolve_csr_host_cache_enabled,
    resolve_csr_host_cache_mode as _resolve_csr_host_cache_mode,
    resolve_csr_pipeline_enabled as _resolve_csr_pipeline_enabled,
    resolve_csr_pipeline_streams as _resolve_csr_pipeline_streams,
    resolve_epq_apply_cache_budget_gib as _resolve_epq_apply_cache_budget_gib,
    resolve_epq_apply_cache_enabled as _resolve_epq_apply_cache_enabled,
    resolve_epq_apply_cache_mode as _resolve_epq_apply_cache_mode,
    resolve_fp32_csr_cache as _resolve_fp32_csr_cache,
    resolve_prefilter_trivial_tasks_enabled as _resolve_prefilter_trivial_tasks_enabled,
    resolve_prefilter_trivial_tasks_mode as _resolve_prefilter_trivial_tasks_mode,
    resolve_skip_zero_x_tiles_enabled as _resolve_skip_zero_x_tiles_enabled,
    resolve_skip_zero_x_tiles_mode as _resolve_skip_zero_x_tiles_mode,
)
from asuka.cuda._backend_workspace_alloc import (
    build_workspace_cache_state as _build_workspace_cache_state,
    make_occ_buf_dtype as _make_occ_buf_dtype,
    resolve_workspace_nrows_block as _resolve_workspace_nrows_block,
)
from asuka.cuda._backend_csr_pipeline import (
    grow_csr_pipeline_slot as _grow_csr_pipeline_slot_runtime,
    init_csr_pipeline_slots as _init_csr_pipeline_slots_runtime,
)
from asuka.cuda._backend_workspace_utils import (
    csr_cache_ready as _csr_cache_ready_runtime,
    estimate_object_nbytes as _estimate_object_nbytes_runtime,
    release_tile_csr_scratch as _release_tile_csr_scratch_runtime,
    release_workspace_resources as _release_workspace_resources_runtime,
    workspace_nbytes_estimate_from_dict as _workspace_nbytes_estimate_from_dict_runtime,
)
from asuka.cuda._backend_cache_runtime import (
    csr_host_cache_load_tile as _csr_host_cache_load_tile_runtime,
    csr_host_cache_store_tile as _csr_host_cache_store_tile_runtime,
    csr_host_cache_try_admit as _csr_host_cache_try_admit_runtime,
    csr_host_entry_bytes as _csr_host_entry_bytes_runtime,
    epq_apply_cache_load as _epq_apply_cache_load_runtime,
    epq_apply_cache_store as _epq_apply_cache_store_runtime,
    epq_apply_ensure_staging as _epq_apply_ensure_staging_runtime,
)
from asuka.cuda._backend_offdiag_setup import (
    init_sym_pair_setup as _init_sym_pair_setup,
    resolve_w_offdiag_prefer_blocked as _resolve_w_offdiag_prefer_blocked,
)
from asuka.cuda._backend_offdiag_workspace import (
    autoset_offdiag_cublas_workspace_with_backoff as _autoset_offdiag_cublas_workspace_with_backoff,
    configure_offdiag_gemm_workspace as _configure_offdiag_gemm_workspace,
)
from asuka.cuda._backend_hop_runtime import (
    build_epq_stream_tile_profiled as _build_epq_stream_tile_profiled_runtime,
    finalize_one_body_phase as _finalize_one_body_phase_runtime,
    finalize_hop_profile as _finalize_hop_profile_runtime,
    increment_profile_counter as _increment_profile_counter_runtime,
    normalize_hop_x as _normalize_hop_x_runtime,
    prepare_hop_runtime_inputs as _prepare_hop_runtime_inputs_runtime,
    run_fused_hop_path as _run_fused_hop_path_runtime,
    resolve_epq_table_for_tile as _resolve_epq_table_for_tile_runtime,
    resolve_epq_streaming_runtime as _resolve_epq_streaming_runtime,
    resolve_hop_runtime_flags as _resolve_hop_runtime_flags_runtime,
    resolve_one_body_mode as _resolve_one_body_mode_runtime,
    resolve_tile_runtime_flags as _resolve_tile_runtime_flags_runtime,
    run_one_body_epq_streaming as _run_one_body_epq_streaming_runtime,
    run_one_body_single_scatter as _run_one_body_single_scatter_runtime,
    stamp_csr_prefilter_profile as _stamp_csr_prefilter_profile_runtime,
    stamp_csr_prefilter_tile_profile as _stamp_csr_prefilter_tile_profile_runtime,
    resolve_x_tile_skip_policy as _resolve_x_tile_skip_policy_runtime,
    set_matvec_path_profile as _set_matvec_path_profile_runtime,
    try_cuda_graph_fast_path as _try_cuda_graph_fast_path_runtime,
)
from asuka.cuda._backend_hop_blocked_runtime import (
    run_blocked_hop_path as _run_blocked_hop_path_runtime,
)
from asuka.cuguga.drt import DRT
from asuka.cuguga.epq.action import epq_apply_g, epq_contribs_one, path_nodes
from asuka.cuguga.oracle import _child_prefix_walks
from asuka.cuguga.state_cache import DRTStateCache, get_state_cache
from asuka.kernels import guga as guga_kernels

_ext = guga_kernels.load_ext()


def has_cuda_ext() -> bool:
    return _has_cuda_ext_runtime(_ext)


def has_epq_table_device_build() -> bool:
    """Return True if the CUDA extension exposes the epq-table device-build entrypoints."""
    return _has_epq_table_device_build_runtime(_ext)


def has_epq_table_device_build_recompute() -> bool:
    """Return True if the CUDA extension exposes on-the-fly state recompute EPQ build entrypoints."""
    return _has_epq_table_device_build_recompute_runtime(_ext)


def has_t_from_epq_table_device_build() -> bool:
    """Return True if the CUDA extension exposes the T-from-epq-table entrypoint."""
    return _has_t_from_epq_table_device_build_runtime(_ext)


def has_epq_table_gather_apply_device() -> bool:
    """Return True if the CUDA extension exposes the EPQ-table destination-gather apply entrypoint."""
    return _has_epq_table_gather_apply_device_runtime(_ext)


def has_build_w_from_epq_transpose_range_mm_scaled() -> bool:
    """Return True if the CUDA extension exposes the mm-scaled EPQ transpose-range W builder."""
    return _has_build_w_from_epq_transpose_range_mm_scaled_runtime(_ext)


def has_build_w_from_epq_transpose_range_mm() -> bool:
    """Return True if the CUDA extension exposes the mm EPQ transpose-range W builder."""
    return _has_build_w_from_epq_transpose_range_mm_runtime(_ext)


def device_info() -> dict[str, object]:
    return _device_info_runtime(_ext)


def mem_info() -> dict[str, int]:
    return _mem_info_runtime(_ext)


def ell_spmv_f64_inplace_device(
    col_idx,
    val,
    x,
    *,
    y=None,
    threads: int = 128,
    add: bool = False,
    stream=None,
    sync: bool = True,
):
    """Compute y <- A*x for an ELL-format sparse matrix on the GPU.

    Parameters
    ----------
    col_idx
        Device array int32 with shape (nrows,width), using col=-1 for padding.
    val
        Device array float64 with shape (nrows,width).
    x
        Device array float64 with shape (nrows,).
    y
        Optional device output array float64 with shape (nrows,). Allocated if None.
    add
        If True, accumulate into existing y; otherwise overwrite y.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    col_idx = cp.asarray(col_idx, dtype=cp.int32)
    col_idx = cp.ascontiguousarray(col_idx)
    val = cp.asarray(val, dtype=cp.float64)
    val = cp.ascontiguousarray(val)
    if col_idx.ndim != 2 or val.ndim != 2:
        raise ValueError("col_idx and val must be 2D arrays")
    if col_idx.shape != val.shape:
        raise ValueError("col_idx and val must have the same shape")

    x = cp.asarray(x, dtype=cp.float64).ravel()
    x = cp.ascontiguousarray(x)
    nrows = int(col_idx.shape[0])
    if x.shape != (nrows,):
        raise ValueError("x must have shape (nrows,)")

    if y is None:
        y = cp.empty((nrows,), dtype=cp.float64)
        if add:
            y.fill(0.0)
    else:
        y = cp.asarray(y, dtype=cp.float64).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (nrows,):
            raise ValueError("y must have shape (nrows,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.ell_spmv_f64_inplace_device(
        col_idx,
        val,
        x,
        y,
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
    )
    return y


def ell_spmm_f64_inplace_device(
    col_idx,
    val,
    x,
    *,
    y=None,
    threads: int = 128,
    add: bool = False,
    stream=None,
    sync: bool = True,
):
    """Compute Y <- A*X for an ELL-format sparse matrix on the GPU.

    Parameters
    ----------
    col_idx
        Device array int32 with shape (nrows,width), using col=-1 for padding.
    val
        Device array float64 with shape (nrows,width).
    x
        Device array float64 with shape (nrows,nvec), C-contiguous.
    y
        Optional device output array float64 with shape (nrows,nvec), C-contiguous. Allocated if None.
    add
        If True, accumulate into existing y; otherwise overwrite y.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    col_idx = cp.asarray(col_idx, dtype=cp.int32)
    col_idx = cp.ascontiguousarray(col_idx)
    val = cp.asarray(val, dtype=cp.float64)
    val = cp.ascontiguousarray(val)
    if col_idx.ndim != 2 or val.ndim != 2:
        raise ValueError("col_idx and val must be 2D arrays")
    if col_idx.shape != val.shape:
        raise ValueError("col_idx and val must have the same shape")

    x = cp.asarray(x, dtype=cp.float64)
    x = cp.ascontiguousarray(x)
    nrows = int(col_idx.shape[0])
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (nrows,nvec)")
    if x.shape[0] != nrows:
        raise ValueError("x must have shape (nrows,nvec) with nrows matching col_idx")
    nvec = int(x.shape[1])

    if y is None:
        y = cp.empty((nrows, nvec), dtype=cp.float64)
        if add:
            y.fill(0.0)
    else:
        y = cp.asarray(y, dtype=cp.float64)
        y = cp.ascontiguousarray(y)
        if y.shape != (nrows, nvec):
            raise ValueError("y must have the same shape as x")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.ell_spmm_f64_inplace_device(
        col_idx,
        val,
        x,
        y,
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
    )
    return y


def sell_spmv_f64_inplace_device(
    slice_ptr,
    slice_width,
    col_idx,
    val,
    x,
    *,
    y=None,
    slice_height: int = 32,
    threads: int = 128,
    add: bool = False,
    stream=None,
    sync: bool = True,
):
    """Compute y <- A*x for a SELL-C sparse matrix on the GPU.

    Parameters
    ----------
    slice_ptr
        Device array int64 with shape (nslices+1,).
    slice_width
        Device array int32 with shape (nslices,).
    col_idx
        Device array int32 with shape (nelems,), using col=-1 for padding.
    val
        Device array float64 with shape (nelems,).
    x
        Device array float64 with shape (nrows,).
    y
        Optional device output array float64 with shape (nrows,). Allocated if None.
    slice_height
        SELL slice height C.
    add
        If True, accumulate into existing y; otherwise overwrite y.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    slice_height = int(slice_height)
    if slice_height <= 0:
        raise ValueError("slice_height must be > 0")

    slice_ptr = cp.asarray(slice_ptr, dtype=cp.int64).ravel()
    slice_ptr = cp.ascontiguousarray(slice_ptr)
    slice_width = cp.asarray(slice_width, dtype=cp.int32).ravel()
    slice_width = cp.ascontiguousarray(slice_width)
    col_idx = cp.asarray(col_idx, dtype=cp.int32).ravel()
    col_idx = cp.ascontiguousarray(col_idx)
    val = cp.asarray(val, dtype=cp.float64).ravel()
    val = cp.ascontiguousarray(val)

    if slice_ptr.ndim != 1 or slice_width.ndim != 1:
        raise ValueError("slice_ptr and slice_width must be 1D arrays")
    if col_idx.ndim != 1 or val.ndim != 1:
        raise ValueError("col_idx and val must be 1D arrays")
    if col_idx.shape != val.shape:
        raise ValueError("col_idx and val must have the same shape")
    if slice_ptr.shape[0] != slice_width.shape[0] + 1:
        raise ValueError("slice_ptr must have shape (nslices+1,)")

    x = cp.asarray(x, dtype=cp.float64).ravel()
    x = cp.ascontiguousarray(x)
    nrows = int(x.shape[0])
    nslices = int(slice_width.shape[0])
    if nrows > nslices * slice_height:
        raise ValueError("x length exceeds nslices*slice_height")

    if y is None:
        y = cp.empty((nrows,), dtype=cp.float64)
        if add:
            y.fill(0.0)
    else:
        y = cp.asarray(y, dtype=cp.float64).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (nrows,):
            raise ValueError("y must have shape (nrows,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.sell_spmv_f64_inplace_device(
        slice_ptr,
        slice_width,
        col_idx,
        val,
        x,
        y,
        int(slice_height),
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
    )
    return y


def sell_spmm_f64_inplace_device(
    slice_ptr,
    slice_width,
    col_idx,
    val,
    x,
    *,
    y=None,
    slice_height: int = 32,
    threads: int = 128,
    add: bool = False,
    stream=None,
    sync: bool = True,
):
    """Compute Y <- A*X for a SELL-C sparse matrix on the GPU.

    Parameters
    ----------
    slice_ptr
        Device array int64 with shape (nslices+1,).
    slice_width
        Device array int32 with shape (nslices,).
    col_idx
        Device array int32 with shape (nelems,), using col=-1 for padding.
    val
        Device array float64 with shape (nelems,).
    x
        Device array float64 with shape (nrows,nvec), C-contiguous.
    y
        Optional device output array float64 with shape (nrows,nvec), C-contiguous. Allocated if None.
    slice_height
        SELL slice height C.
    add
        If True, accumulate into existing y; otherwise overwrite y.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    slice_height = int(slice_height)
    if slice_height <= 0:
        raise ValueError("slice_height must be > 0")

    slice_ptr = cp.asarray(slice_ptr, dtype=cp.int64).ravel()
    slice_ptr = cp.ascontiguousarray(slice_ptr)
    slice_width = cp.asarray(slice_width, dtype=cp.int32).ravel()
    slice_width = cp.ascontiguousarray(slice_width)
    col_idx = cp.asarray(col_idx, dtype=cp.int32).ravel()
    col_idx = cp.ascontiguousarray(col_idx)
    val = cp.asarray(val, dtype=cp.float64).ravel()
    val = cp.ascontiguousarray(val)
    if col_idx.shape != val.shape:
        raise ValueError("col_idx and val must have the same shape")
    if slice_ptr.shape[0] != slice_width.shape[0] + 1:
        raise ValueError("slice_ptr must have shape (nslices+1,)")

    x = cp.asarray(x, dtype=cp.float64)
    x = cp.ascontiguousarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (nrows,nvec)")
    nrows = int(x.shape[0])
    nvec = int(x.shape[1])
    nslices = int(slice_width.shape[0])
    if nrows > nslices * slice_height:
        raise ValueError("x rows exceed nslices*slice_height")

    if y is None:
        y = cp.empty((nrows, nvec), dtype=cp.float64)
        if add:
            y.fill(0.0)
    else:
        y = cp.asarray(y, dtype=cp.float64)
        y = cp.ascontiguousarray(y)
        if y.shape != (nrows, nvec):
            raise ValueError("y must have the same shape as x")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.sell_spmm_f64_inplace_device(
        slice_ptr,
        slice_width,
        col_idx,
        val,
        x,
        y,
        int(slice_height),
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
    )
    return y


def _csr_to_ell_host(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, *, nrows: int
) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(indptr, dtype=np.int64).ravel()
    indices = np.asarray(indices, dtype=np.int32).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    nrows = int(nrows)
    if nrows < 0:
        raise ValueError("nrows must be >= 0")
    if indptr.shape != (nrows + 1,):
        raise ValueError("indptr must have shape (nrows+1,)")
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")
    widths = np.diff(indptr)
    width = int(widths.max()) if widths.size else 0
    col_idx = -np.ones((nrows, width), dtype=np.int32)
    val = np.zeros((nrows, width), dtype=np.float64)
    for row in range(nrows):
        a = int(indptr[row])
        b = int(indptr[row + 1])
        if a == b:
            continue
        row_cols = indices[a:b]
        row_vals = data[a:b]
        col_idx[row, : row_cols.size] = row_cols
        val[row, : row_vals.size] = row_vals
    return col_idx, val


def _csr_to_sell_host(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    nrows: int,
    slice_height: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.asarray(indptr, dtype=np.int64).ravel()
    indices = np.asarray(indices, dtype=np.int32).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    nrows = int(nrows)
    slice_height = int(slice_height)
    if nrows < 0:
        raise ValueError("nrows must be >= 0")
    if slice_height <= 0:
        raise ValueError("slice_height must be > 0")
    if indptr.shape != (nrows + 1,):
        raise ValueError("indptr must have shape (nrows+1,)")
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")

    widths = np.diff(indptr)
    nslices = (nrows + slice_height - 1) // slice_height
    slice_width = np.zeros((nslices,), dtype=np.int32)
    slice_ptr = np.zeros((nslices + 1,), dtype=np.int64)
    for s in range(nslices):
        r0 = s * slice_height
        r1 = min(nrows, r0 + slice_height)
        w = int(widths[r0:r1].max()) if r1 > r0 and widths.size else 0
        slice_width[s] = int(w)
        slice_ptr[s + 1] = slice_ptr[s] + int(slice_height) * int(w)

    nelems = int(slice_ptr[-1])
    col_idx = -np.ones((nelems,), dtype=np.int32)
    val = np.zeros((nelems,), dtype=np.float64)
    for s in range(nslices):
        w = int(slice_width[s])
        if w <= 0:
            continue
        base0 = int(slice_ptr[s])
        for r in range(slice_height):
            row = s * slice_height + r
            if row >= nrows:
                continue
            a = int(indptr[row])
            b = int(indptr[row + 1])
            m = int(b - a)
            if m <= 0:
                continue
            off = base0 + r * w
            col_idx[off : off + m] = indices[a:b]
            val[off : off + m] = data[a:b]

    return slice_ptr, slice_width, col_idx, val


def build_h_csr_from_row_oracle_host(
    drt: DRT,
    h1e,
    eri,
    *,
    max_out: int = 200_000,
    row_oracle: str | Callable[[int], tuple[np.ndarray, np.ndarray]] = "sparse",
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build the full Hamiltonian H in CSR format on host (NumPy) for a fixed DRT.

    Returns (indptr, indices, data, stats) where:
      - indptr: int64 (ncsf+1,)
      - indices: int32 (nnz,)
      - data: float64 (nnz,)
      - stats: dict with keys like nnz_total, width_max
    """

    ncsf = int(drt.ncsf)
    if ncsf <= 0:
        raise ValueError("invalid DRT.ncsf")
    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")

    if state_cache is None:
        state_cache = get_state_cache(drt)

    if callable(row_oracle):
        _row = row_oracle
    else:
        row_oracle_s = str(row_oracle).strip().lower()
        if row_oracle_s in ("sparse", "row_oracle_sparse"):
            from asuka.cuguga.oracle.sparse import connected_row_sparse  # noqa: PLC0415

            def _row(j: int) -> tuple[np.ndarray, np.ndarray]:
                return connected_row_sparse(drt, h1e, eri, int(j), max_out=max_out, state_cache=state_cache)

        elif row_oracle_s in ("reference", "connected_row"):
            from asuka.cuguga.oracle import connected_row  # noqa: PLC0415

            def _row(j: int) -> tuple[np.ndarray, np.ndarray]:
                return connected_row(drt, h1e, eri, int(j), max_out=max_out)

        else:
            raise ValueError(f"unsupported row_oracle={row_oracle!r}")

    indptr = np.empty((ncsf + 1,), dtype=np.int64)
    indptr[0] = 0
    idx_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    width_max = 0
    nnz_total = 0
    for j in range(ncsf):
        i_idx, hij = _row(int(j))
        i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
        hij = np.asarray(hij, dtype=np.float64).ravel()
        if i_idx.shape != hij.shape:
            raise RuntimeError("row oracle returned mismatched (i_idx,hij) shapes")
        if i_idx.size == 0:
            raise RuntimeError("row oracle returned an empty row (expected at least the diagonal)")
        width_max = max(int(width_max), int(i_idx.size))
        idx_list.append(i_idx)
        val_list.append(hij)
        nnz_total += int(i_idx.size)
        indptr[j + 1] = int(nnz_total)

    indices = np.concatenate(idx_list).astype(np.int32, copy=False)
    data = np.concatenate(val_list).astype(np.float64, copy=False)
    stats = {"ncsf": ncsf, "nnz_total": int(nnz_total), "width_max": int(width_max)}
    return indptr, indices, data, stats


def build_h_csr_from_row_oracle_host_selected(
    drt: DRT,
    h1e,
    eri,
    sel_idx: Sequence[int] | np.ndarray,
    *,
    max_out: int = 200_000,
    row_oracle: str | Callable[[int], tuple[np.ndarray, np.ndarray]] = "sparse",
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build the selected-subspace Hamiltonian H[sel,sel] in CSR format on host (NumPy).

    Parameters
    ----------
    sel_idx
        Global CSF indices (basis order) inside the parent DRT. Duplicates are not allowed.

    Returns
    -------
    indptr, indices, data, stats
        CSR arrays for the selected subspace with shape (nsel,nsel), using *local*
        column indices in `indices`.
    """

    ncsf = int(drt.ncsf)
    if ncsf <= 0:
        raise ValueError("invalid DRT.ncsf")

    sel = np.asarray(sel_idx, dtype=np.int32).ravel()
    nsel = int(sel.size)
    if nsel <= 0:
        raise ValueError("sel_idx must be non-empty")
    if np.any(sel < 0) or np.any(sel >= ncsf):
        raise ValueError("sel_idx contains out-of-range CSF indices")

    uniq = np.unique(sel)
    if int(uniq.size) != nsel:
        raise ValueError("sel_idx contains duplicates")

    # Build a sorted view for O(log nsel) membership + mapping global->local (basis order).
    sort_order = np.argsort(sel, kind="stable").astype(np.int32, copy=False)
    sel_sorted = np.asarray(sel[sort_order], dtype=np.int32, order="C")
    sorted_to_local = np.asarray(sort_order, dtype=np.int32, order="C")

    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")

    if state_cache is None:
        state_cache = get_state_cache(drt)

    if callable(row_oracle):
        _row = row_oracle
    else:
        row_oracle_s = str(row_oracle).strip().lower()
        if row_oracle_s in ("sparse", "row_oracle_sparse"):
            from asuka.cuguga.oracle.sparse import connected_row_sparse  # noqa: PLC0415

            def _row(j: int) -> tuple[np.ndarray, np.ndarray]:
                return connected_row_sparse(drt, h1e, eri, int(j), max_out=max_out, state_cache=state_cache)

        elif row_oracle_s in ("reference", "connected_row"):
            from asuka.cuguga.oracle import connected_row  # noqa: PLC0415

            def _row(j: int) -> tuple[np.ndarray, np.ndarray]:
                return connected_row(drt, h1e, eri, int(j), max_out=max_out)

        else:
            raise ValueError(f"unsupported row_oracle={row_oracle!r}")

    indptr = np.empty((nsel + 1,), dtype=np.int64)
    indptr[0] = 0
    idx_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    width_max = 0
    nnz_total = 0
    nnz_kept = 0

    for r in range(nsel):
        j = int(sel[r])
        i_idx, hij = _row(j)
        i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
        hij = np.asarray(hij, dtype=np.float64).ravel()
        if i_idx.shape != hij.shape:
            raise RuntimeError("row oracle returned mismatched (i_idx,hij) shapes")
        if i_idx.size == 0:
            raise RuntimeError("row oracle returned an empty row (expected at least the diagonal)")

        nnz_total += int(i_idx.size)

        pos = np.searchsorted(sel_sorted, i_idx)
        pos_clip = np.minimum(pos, nsel - 1)
        mask = (pos < nsel) & (sel_sorted[pos_clip] == i_idx)
        cols = sorted_to_local[pos_clip[mask]]
        vals = hij[mask]
        if cols.size == 0:
            raise RuntimeError("selected-row filtering produced an empty row (expected at least the diagonal)")

        idx_list.append(np.asarray(cols, dtype=np.int32, order="C"))
        val_list.append(np.asarray(vals, dtype=np.float64, order="C"))
        width_max = max(int(width_max), int(cols.size))
        nnz_kept += int(cols.size)
        indptr[r + 1] = int(nnz_kept)

    indices = np.concatenate(idx_list).astype(np.int32, copy=False)
    data = np.concatenate(val_list).astype(np.float64, copy=False)
    stats = {
        "ncsf_parent": int(ncsf),
        "nsel": int(nsel),
        "nnz_total": int(nnz_total),
        "nnz_kept": int(nnz_kept),
        "width_max": int(width_max),
    }
    return indptr, indices, data, stats


def build_h_ell_from_row_oracle_host(
    drt: DRT,
    h1e,
    eri,
    *,
    max_out: int = 200_000,
    row_oracle: str | Callable[[int], tuple[np.ndarray, np.ndarray]] = "sparse",
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the full Hamiltonian H in ELL format on host (NumPy) for a fixed DRT.

    This is intended for small CSF spaces where H is sparse enough to store and reuse.
    Returns (col_idx, val, stats) where:
      - col_idx: int32 (ncsf,width) with -1 padding
      - val: float64 (ncsf,width)
      - stats: dict with keys like nnz_total, width_max
    """

    indptr, indices, data, stats = build_h_csr_from_row_oracle_host(
        drt,
        h1e,
        eri,
        max_out=int(max_out),
        row_oracle=row_oracle,
        state_cache=state_cache,
    )
    ncsf = int(stats.get("ncsf", int(drt.ncsf)))
    col_idx, val = _csr_to_ell_host(indptr, indices, data, nrows=ncsf)
    return col_idx, val, stats


def build_h_sell_from_row_oracle_host(
    drt: DRT,
    h1e,
    eri,
    *,
    slice_height: int = 32,
    max_out: int = 200_000,
    row_oracle: str | Callable[[int], tuple[np.ndarray, np.ndarray]] = "sparse",
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build the full Hamiltonian H in SELL-C format on host (NumPy) for a fixed DRT.

    Returns (slice_ptr, slice_width, col_idx, val, stats) where:
      - slice_ptr: int64 (nslices+1,)
      - slice_width: int32 (nslices,)
      - col_idx: int32 (nelems,) with -1 padding
      - val: float64 (nelems,)
    """
    slice_height = int(slice_height)
    if slice_height <= 0:
        raise ValueError("slice_height must be > 0")

    indptr, indices, data, stats0 = build_h_csr_from_row_oracle_host(
        drt,
        h1e,
        eri,
        max_out=int(max_out),
        row_oracle=row_oracle,
        state_cache=state_cache,
    )
    ncsf = int(stats0.get("ncsf", int(drt.ncsf)))
    slice_ptr, slice_width, col_idx, val = _csr_to_sell_host(
        indptr, indices, data, nrows=ncsf, slice_height=int(slice_height)
    )
    stats = dict(stats0)
    stats.update(
        {
            "slice_height": int(slice_height),
            "nslices": int(slice_width.size),
            "nelems": int(col_idx.size),
        }
    )
    return slice_ptr, slice_width, col_idx, val, stats


class GugaMatvecFixedEllWorkspace:
    """Fixed sparse matvec workspace: y = Hx using ELL SpMV/SpMM kernels."""

    def __init__(self, col_idx_d, val_d, *, threads_spmv: int = 128, threads_spmm: int = 128) -> None:
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the fixed-ELL workspace") from e

        self.col_idx = cp.ascontiguousarray(cp.asarray(col_idx_d, dtype=cp.int32))
        self.val = cp.ascontiguousarray(cp.asarray(val_d, dtype=cp.float64))
        if self.col_idx.ndim != 2 or self.val.ndim != 2 or self.col_idx.shape != self.val.shape:
            raise ValueError("col_idx and val must be 2D arrays with the same shape")
        self.ncsf = int(self.col_idx.shape[0])
        self.width = int(self.col_idx.shape[1])
        self.threads_spmv = int(threads_spmv)
        self.threads_spmm = int(threads_spmm)

    def hop(self, x_d, *, y=None, stream=None, sync: bool = False):
        return ell_spmv_f64_inplace_device(
            self.col_idx,
            self.val,
            x_d,
            y=y,
            threads=int(self.threads_spmv),
            add=False,
            stream=stream,
            sync=bool(sync),
        )

    def hop_many(self, x_d, *, y=None, stream=None, sync: bool = False):
        return ell_spmm_f64_inplace_device(
            self.col_idx,
            self.val,
            x_d,
            y=y,
            threads=int(self.threads_spmm),
            add=False,
            stream=stream,
            sync=bool(sync),
        )


class GugaMatvecFixedSellWorkspace:
    """Fixed sparse matvec workspace: y = Hx using SELL SpMV/SpMM kernels."""

    def __init__(
        self,
        slice_ptr_d,
        slice_width_d,
        col_idx_d,
        val_d,
        *,
        nrows: int,
        slice_height: int = 32,
        threads_spmv: int = 128,
        threads_spmm: int = 128,
    ) -> None:
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the fixed-SELL workspace") from e

        nrows = int(nrows)
        slice_height = int(slice_height)
        if nrows < 0:
            raise ValueError("nrows must be >= 0")
        if slice_height <= 0:
            raise ValueError("slice_height must be > 0")

        self.slice_ptr = cp.ascontiguousarray(cp.asarray(slice_ptr_d, dtype=cp.int64)).ravel()
        self.slice_width = cp.ascontiguousarray(cp.asarray(slice_width_d, dtype=cp.int32)).ravel()
        self.col_idx = cp.ascontiguousarray(cp.asarray(col_idx_d, dtype=cp.int32)).ravel()
        self.val = cp.ascontiguousarray(cp.asarray(val_d, dtype=cp.float64)).ravel()

        if self.slice_ptr.ndim != 1 or self.slice_width.ndim != 1:
            raise ValueError("slice_ptr and slice_width must be 1D arrays")
        if self.col_idx.ndim != 1 or self.val.ndim != 1 or self.col_idx.shape != self.val.shape:
            raise ValueError("col_idx and val must be 1D arrays with the same shape")
        if int(self.slice_ptr.size) != int(self.slice_width.size) + 1:
            raise ValueError("slice_ptr must have shape (nslices+1,)")

        nslices = int(self.slice_width.size)
        if nrows > nslices * slice_height:
            raise ValueError("nrows exceeds nslices*slice_height")

        self.ncsf = int(nrows)
        self.slice_height = int(slice_height)
        self.threads_spmv = int(threads_spmv)
        self.threads_spmm = int(threads_spmm)

    def hop(self, x_d, *, y=None, stream=None, sync: bool = False):
        return sell_spmv_f64_inplace_device(
            self.slice_ptr,
            self.slice_width,
            self.col_idx,
            self.val,
            x_d,
            y=y,
            slice_height=int(self.slice_height),
            threads=int(self.threads_spmv),
            add=False,
            stream=stream,
            sync=bool(sync),
        )

    def hop_many(self, x_d, *, y=None, stream=None, sync: bool = False):
        return sell_spmm_f64_inplace_device(
            self.slice_ptr,
            self.slice_width,
            self.col_idx,
            self.val,
            x_d,
            y=y,
            slice_height=int(self.slice_height),
            threads=int(self.threads_spmm),
            add=False,
            stream=stream,
            sync=bool(sync),
        )


def make_rdm_gram_workspace(*, nops: int):
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    return _ext.RDMGramWorkspace(int(nops))


def make_device_drt(drt: DRT):
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    child_prefix = _child_prefix_walks(drt)
    return _ext.make_device_drt(int(drt.norb), np.asarray(drt.child), np.asarray(drt.node_twos), np.asarray(child_prefix))


def make_device_state_cache(drt: DRT, drt_dev, cache: DRTStateCache | None = None):
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    if cache is None:
        cache = get_state_cache(drt)
    steps = np.asarray(cache.steps, dtype=np.int8, order="C")
    nodes = np.asarray(cache.nodes, dtype=np.int32, order="C")
    return _ext.make_device_state_cache(drt_dev, steps, nodes)


def _get_epq_action_table_combined_host(
    drt: DRT, *, precompute_nthreads: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a combined (off-diagonal) E_pq action table grouped by CSF.

    The combined table is a CSR-like structure over CSF columns:
      indptr[j]:indptr[j+1] gives entries (indices[t], pq_ids[t], data[t])
    where `pq_ids[t] == p*norb + q` indexes into `g_flat[pq]` (with `p!=q`).
    """

    cached = getattr(drt, "_epq_action_table_combined_host", None)
    if cached is not None:
        return cached

    from asuka.cuguga.oracle import _get_epq_action_cache, precompute_epq_actions  # noqa: PLC0415

    precompute_epq_actions(drt, nthreads=int(precompute_nthreads))
    cache = _get_epq_action_cache(drt)
    ncsf = int(drt.ncsf)

    nnz_per_csf = np.zeros((ncsf,), dtype=np.int64)
    for csr in cache.by_pair.values():
        nnz_per_csf += np.diff(csr.indptr).astype(np.int64, copy=False)

    indptr = np.empty((ncsf + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(nnz_per_csf, out=indptr[1:])
    nnz_total = int(indptr[-1])

    indices = np.empty((nnz_total,), dtype=np.int32)
    pq_ids = np.empty((nnz_total,), dtype=np.int32)
    data = np.empty((nnz_total,), dtype=np.float64)

    pos = indptr[:-1].copy()
    # Most rows have very small nnz for E_pq actions. Vectorize fixed small row lengths
    # to avoid a Python loop over millions of rows.
    LMAX = 19
    for pair_id, csr in cache.by_pair.items():
        indptr_pq = csr.indptr
        row_lens = np.diff(indptr_pq)

        if not row_lens.size:
            continue

        for L in range(1, LMAX + 1):
            rows = np.nonzero(row_lens == L)[0]
            if rows.size == 0:
                continue
            src0 = indptr_pq[rows].astype(np.int64, copy=False)
            dst0 = pos[rows].astype(np.int64, copy=False)
            offs = np.arange(L, dtype=np.int64)
            src = (src0[:, None] + offs[None, :]).ravel()
            dst = (dst0[:, None] + offs[None, :]).ravel()
            indices[dst] = csr.indices[src]
            pq_ids[dst] = int(pair_id)
            data[dst] = csr.data[src]
            pos[rows] += L

        rows_big = np.nonzero(row_lens > LMAX)[0]
        for row in rows_big.tolist():
            s = int(indptr_pq[row])
            e = int(indptr_pq[row + 1])
            dst0 = int(pos[row])
            m = int(e - s)
            indices[dst0 : dst0 + m] = csr.indices[s:e]
            pq_ids[dst0 : dst0 + m] = int(pair_id)
            data[dst0 : dst0 + m] = csr.data[s:e]
            pos[row] += m

    if not np.array_equal(pos, indptr[1:]):  # pragma: no cover
        raise RuntimeError("internal error building combined E_pq action table")

    out = (indptr, indices, pq_ids, data)
    setattr(drt, "_epq_action_table_combined_host", out)
    return out


def build_occ_block_from_steps_inplace_device(
    state_dev,
    *,
    j_start: int,
    j_count: int,
    occ_out=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Build an occupancy block from `DeviceStateCache.steps` on the GPU.

    Computes `occ_out[j_local,k] = step_to_occ(steps_table[j_start+j_local,k])` for `j_local=0..j_count-1`.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start and j_count must be >= 0")

    norb = int(getattr(state_dev, "norb", 0))
    if norb <= 0:
        raise ValueError("invalid state_dev.norb")

    if occ_out is None:
        occ_out = cp.empty((j_count, norb), dtype=cp.float64)
    else:
        occ_out = cp.asarray(occ_out)
        if occ_out.dtype != cp.float64:
            raise ValueError("occ_out must have dtype float64")
        if not getattr(occ_out, "flags", None) or not occ_out.flags.c_contiguous:
            raise ValueError("occ_out must be C-contiguous")
        if occ_out.shape != (j_count, norb):
            raise ValueError("occ_out must have shape (j_count,norb)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_occ_block_from_steps_inplace_device(
        state_dev,
        int(j_start),
        int(j_count),
        occ_out,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return occ_out


def build_w_diag_from_steps_inplace_device(
    state_dev,
    *,
    j_start: int,
    j_count: int,
    x,
    w_out,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    relative_w: bool = False,
):
    """Write diagonal rs (r==s) contributions directly into `W[j, rr]` on the GPU.

    Computes for `j in [j_start, j_start+j_count)` and `r in [0,norb)`:
        w_out[j, r*norb+r] = x[j] * occ(j,r)

    This is intended for the k-aggregated matvec path so the same `W @ ERIᵀ` GEMM covers both
    diagonal and off-diagonal rs contributions.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start and j_count must be >= 0")

    norb = int(getattr(state_dev, "norb", 0))
    ncsf = int(getattr(state_dev, "ncsf", 0))
    if norb <= 0 or ncsf <= 0:
        raise ValueError("invalid state_dev.norb/state_dev.ncsf")
    if j_start + j_count > ncsf:
        raise ValueError("j_start/j_count out of range")

    x_in = cp.asarray(x)
    w_in = cp.asarray(w_out)
    x_dtype = cp.dtype(x_in.dtype)
    w_dtype = cp.dtype(w_in.dtype)
    allowed = (cp.dtype(cp.float32), cp.dtype(cp.float64))
    if x_dtype not in allowed and w_dtype not in allowed:
        fp_dtype = cp.float64
    elif x_dtype in allowed and w_dtype in allowed and x_dtype != w_dtype:
        raise ValueError("x and w_out must use the same dtype (float32 or float64)")
    else:
        fp_dtype = x_dtype if x_dtype in allowed else w_dtype

    x = cp.asarray(x_in, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (ncsf,):
        raise ValueError("x must have shape (ncsf,)")

    w_out = cp.asarray(w_in, dtype=fp_dtype)
    if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
        raise ValueError("w_out must be C-contiguous")
    nops = norb * norb
    expected_rows = int(j_count) if bool(relative_w) else int(ncsf)
    if w_out.ndim == 1:
        if w_out.shape != (expected_rows * nops,):
            raise ValueError("w_out (1D) must have shape (expected_rows*nops,)")
    elif w_out.ndim == 2:
        if w_out.shape != (expected_rows, nops):
            raise ValueError("w_out (2D) must have shape (expected_rows,nops)")
    else:
        raise ValueError("w_out must be 1D or 2D")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    if not hasattr(_ext, "build_w_diag_from_steps_inplace_device"):
        raise RuntimeError("CUDA extension is missing build_w_diag_from_steps_inplace_device; rebuild the extension")

    _ext.build_w_diag_from_steps_inplace_device(
        state_dev,
        int(j_start),
        int(j_count),
        x,
        w_out,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(relative_w),
    )
    return w_out


def build_hdiag_det_guess_from_steps_inplace_device(
    state_dev,
    *,
    neleca_det: int,
    h1e_diag,
    eri_ppqq,
    eri_pqqp,
    hdiag_out=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Compute the determinant-style diagonal guess hdiag on GPU from the cached steps table.

    This matches the logic used by `GUGAFCISolver.make_hdiag` for dense ERIs:
    - infer (doubly,single,empty) from the DRT `steps_table`
    - assign the first `alpha_need = neleca_det - ndoubly` single-occupied orbitals to alpha
    - compute the diagonal guess using `(pp|qq)` and `(pq|qp)` slices

    Intended for large-CSF GPU solves where the CPU `make_hdiag` becomes a bottleneck.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    norb = int(getattr(state_dev, "norb", 0))
    ncsf = int(getattr(state_dev, "ncsf", 0))
    if norb <= 0 or ncsf <= 0:
        raise ValueError("invalid state_dev.norb/state_dev.ncsf")

    neleca_det = int(neleca_det)
    if neleca_det < 0:
        raise ValueError("neleca_det must be >= 0")

    h1e_diag = cp.asarray(h1e_diag, dtype=cp.float64).ravel()
    h1e_diag = cp.ascontiguousarray(h1e_diag)
    if h1e_diag.shape != (norb,):
        raise ValueError("h1e_diag must have shape (norb,)")

    eri_ppqq = cp.asarray(eri_ppqq, dtype=cp.float64)
    eri_ppqq = cp.ascontiguousarray(eri_ppqq)
    if eri_ppqq.shape != (norb, norb):
        raise ValueError("eri_ppqq must have shape (norb,norb)")

    eri_pqqp = cp.asarray(eri_pqqp, dtype=cp.float64)
    eri_pqqp = cp.ascontiguousarray(eri_pqqp)
    if eri_pqqp.shape != (norb, norb):
        raise ValueError("eri_pqqp must have shape (norb,norb)")

    if hdiag_out is None:
        hdiag_out = cp.empty((ncsf,), dtype=cp.float64)
    else:
        hdiag_out = cp.asarray(hdiag_out, dtype=cp.float64).ravel()
        hdiag_out = cp.ascontiguousarray(hdiag_out)
        if hdiag_out.shape != (ncsf,):
            raise ValueError("hdiag_out must have shape (ncsf,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    if not hasattr(_ext, "build_hdiag_det_guess_from_steps_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing build_hdiag_det_guess_from_steps_inplace_device; rebuild the extension"
        )

    _ext.build_hdiag_det_guess_from_steps_inplace_device(
        state_dev,
        int(neleca_det),
        h1e_diag,
        eri_ppqq,
        eri_pqqp,
        hdiag_out,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return hdiag_out


def epq_contribs_one_debug(
    drt: DRT,
    drt_dev,
    csf_idx: int,
    p: int,
    q: int,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
    max_out: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    csf_idx = int(csf_idx)
    p = int(p)
    q = int(q)

    if steps is None:
        steps = drt.index_to_path(csf_idx).astype(np.int8, copy=False)
    if nodes is None:
        nodes = path_nodes(drt, steps)

    steps = np.asarray(steps, dtype=np.int8).ravel()
    nodes = np.asarray(nodes, dtype=np.int32).ravel()

    i_idx, coeff = _ext.epq_contribs_one_debug(drt_dev, csf_idx, steps, nodes, p, q, int(max_out))
    return np.asarray(i_idx, dtype=np.int32), np.asarray(coeff, dtype=np.float64)


def validate_epq_contribs_one_debug(
    drt: DRT,
    csf_idx: int,
    p: int,
    q: int,
    *,
    drt_dev=None,
    max_out: int = 100_000,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    csf_idx = int(csf_idx)
    p = int(p)
    q = int(q)

    steps = drt.index_to_path(csf_idx).astype(np.int8, copy=False)
    nodes = path_nodes(drt, steps)

    i_cpu, v_cpu = epq_contribs_one(drt, csf_idx, p, q, steps=steps, nodes=nodes)

    if drt_dev is None:
        drt_dev = make_device_drt(drt)
    i_gpu, v_gpu = epq_contribs_one_debug(
        drt, drt_dev, csf_idx, p, q, steps=steps, nodes=nodes, max_out=max_out
    )

    order_cpu = np.argsort(i_cpu, kind="mergesort")
    order_gpu = np.argsort(i_gpu, kind="mergesort")
    i_cpu = np.asarray(i_cpu[order_cpu], dtype=np.int32)
    v_cpu = np.asarray(v_cpu[order_cpu], dtype=np.float64)
    i_gpu = np.asarray(i_gpu[order_gpu], dtype=np.int32)
    v_gpu = np.asarray(v_gpu[order_gpu], dtype=np.float64)

    if i_cpu.shape != i_gpu.shape or not np.array_equal(i_cpu, i_gpu):
        raise AssertionError(f"idx mismatch: cpu={i_cpu.shape} gpu={i_gpu.shape}")

    if not np.allclose(v_cpu, v_gpu, atol=float(atol), rtol=float(rtol)):
        diff = np.max(np.abs(v_cpu - v_gpu)) if v_cpu.size else 0.0
        raise AssertionError(f"coeff mismatch (max |Δ|={diff})")


def epq_apply_g_debug(
    drt: DRT,
    drt_dev,
    csf_idx: int,
    g_flat: np.ndarray,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
    thresh_gpq: float = 0.0,
    thresh_contrib: float = 0.0,
    max_out: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    csf_idx = int(csf_idx)
    g_flat = np.asarray(g_flat, dtype=np.float64).ravel(order="C")
    if steps is None:
        steps = drt.index_to_path(csf_idx).astype(np.int8, copy=False)
    if nodes is None:
        nodes = path_nodes(drt, steps)

    steps = np.asarray(steps, dtype=np.int8).ravel()
    nodes = np.asarray(nodes, dtype=np.int32).ravel()

    i_idx, val, n_pairs = _ext.epq_apply_g_debug(
        drt_dev,
        csf_idx,
        g_flat,
        steps,
        nodes,
        float(thresh_gpq),
        float(thresh_contrib),
        int(max_out),
    )
    return np.asarray(i_idx, dtype=np.int32), np.asarray(val, dtype=np.float64), int(n_pairs)


def _coalesce_by_index(i_idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
    val = np.asarray(val, dtype=np.float64).ravel()
    if i_idx.size == 0:
        return i_idx, val
    order = np.argsort(i_idx, kind="mergesort")
    i_sorted = i_idx[order]
    v_sorted = val[order]
    change = np.nonzero(i_sorted[1:] != i_sorted[:-1])[0] + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int32), change.astype(np.int32, copy=False)))
    i_uniq = i_sorted[starts]
    v_uniq = np.add.reduceat(v_sorted, starts)
    return np.asarray(i_uniq, dtype=np.int32), np.asarray(v_uniq, dtype=np.float64)


def validate_epq_apply_g_debug(
    drt: DRT,
    csf_idx: int,
    g_flat: np.ndarray,
    *,
    drt_dev=None,
    max_out: int = 200_000,
    thresh_gpq: float = 0.0,
    thresh_contrib: float = 0.0,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    csf_idx = int(csf_idx)
    g_flat = np.asarray(g_flat, dtype=np.float64).ravel(order="C")

    steps = drt.index_to_path(csf_idx).astype(np.int8, copy=False)
    nodes = path_nodes(drt, steps)

    i_cpu, v_cpu, n_pairs_cpu = epq_apply_g(
        drt,
        csf_idx,
        g_flat,
        steps=steps,
        nodes=nodes,
        thresh_gpq=float(thresh_gpq),
        thresh_contrib=float(thresh_contrib),
    )

    if drt_dev is None:
        drt_dev = make_device_drt(drt)
    i_gpu, v_gpu, n_pairs_gpu = epq_apply_g_debug(
        drt,
        drt_dev,
        csf_idx,
        g_flat,
        steps=steps,
        nodes=nodes,
        thresh_gpq=float(thresh_gpq),
        thresh_contrib=float(thresh_contrib),
        max_out=max_out,
    )

    if int(n_pairs_cpu) != int(n_pairs_gpu):
        raise AssertionError(f"n_pairs mismatch: cpu={int(n_pairs_cpu)} gpu={int(n_pairs_gpu)}")

    i_cpu_c, v_cpu_c = _coalesce_by_index(i_cpu, v_cpu)
    i_gpu_c, v_gpu_c = _coalesce_by_index(i_gpu, v_gpu)

    if i_cpu_c.shape != i_gpu_c.shape or not np.array_equal(i_cpu_c, i_gpu_c):
        raise AssertionError(f"idx mismatch after coalesce: cpu={i_cpu_c.shape} gpu={i_gpu_c.shape}")

    if not np.allclose(v_cpu_c, v_gpu_c, atol=float(atol), rtol=float(rtol)):
        diff = np.max(np.abs(v_cpu_c - v_gpu_c)) if v_cpu_c.size else 0.0
        raise AssertionError(f"val mismatch after coalesce (max |Δ|={diff})")


def epq_contribs_many_deterministic(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf: np.ndarray,
    task_p: np.ndarray,
    task_q: np.ndarray,
    *,
    threads: int = 128,
    max_total_out: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_p = np.asarray(task_p, dtype=np.int32).ravel()
    task_q = np.asarray(task_q, dtype=np.int32).ravel()
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")

    out_i, out_v, offsets = _ext.epq_contribs_many_deterministic(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        int(threads),
        int(max_total_out),
    )
    return np.asarray(out_i, dtype=np.int32), np.asarray(out_v, dtype=np.float64), np.asarray(offsets, dtype=np.int64)


def validate_epq_contribs_many_deterministic(
    drt: DRT,
    task_csf: np.ndarray,
    task_p: np.ndarray,
    task_q: np.ndarray,
    *,
    drt_dev=None,
    state_dev=None,
    cache: DRTStateCache | None = None,
    ncheck: int = 50,
    seed: int = 0,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_p = np.asarray(task_p, dtype=np.int32).ravel()
    task_q = np.asarray(task_q, dtype=np.int32).ravel()
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")

    if drt_dev is None:
        drt_dev = make_device_drt(drt)
    if cache is None:
        cache = get_state_cache(drt)
    if state_dev is None:
        state_dev = make_device_state_cache(drt, drt_dev, cache)

    out_i, out_v, offsets = epq_contribs_many_deterministic(
        drt,
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
    )

    rng = np.random.default_rng(int(seed))
    ntasks = int(task_csf.size)
    ncheck = max(0, min(int(ncheck), ntasks))
    if ncheck == 0:
        return

    picks = rng.choice(ntasks, size=ncheck, replace=False)
    for t in picks.tolist():
        csf_idx = int(task_csf[t])
        p = int(task_p[t])
        q = int(task_q[t])

        steps = np.asarray(cache.steps[csf_idx], dtype=np.int8, order="C")
        nodes = np.asarray(cache.nodes[csf_idx], dtype=np.int32, order="C")

        i_cpu, v_cpu = epq_contribs_one(drt, csf_idx, p, q, steps=steps, nodes=nodes)

        base = int(offsets[t])
        limit = int(offsets[t + 1])
        i_gpu = np.asarray(out_i[base:limit], dtype=np.int32)
        v_gpu = np.asarray(out_v[base:limit], dtype=np.float64)

        i_cpu_c, v_cpu_c = _coalesce_by_index(i_cpu, v_cpu)
        i_gpu_c, v_gpu_c = _coalesce_by_index(i_gpu, v_gpu)

        if i_cpu_c.shape != i_gpu_c.shape or not np.array_equal(i_cpu_c, i_gpu_c):
            raise AssertionError(f"idx mismatch after coalesce (task {t})")

        if not np.allclose(v_cpu_c, v_gpu_c, atol=float(atol), rtol=float(rtol)):
            diff = np.max(np.abs(v_cpu_c - v_gpu_c)) if v_cpu_c.size else 0.0
            raise AssertionError(f"coeff mismatch after coalesce (task {t}, max |Δ|={diff})")


def epq_apply_weighted_many_atomic(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf: np.ndarray,
    task_p: np.ndarray,
    task_q: np.ndarray,
    task_wgt: np.ndarray,
    *,
    task_scale: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    threads: int = 128,
    return_y: bool = True,
):
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_p = np.asarray(task_p, dtype=np.int32).ravel()
    task_q = np.asarray(task_q, dtype=np.int32).ravel()
    task_wgt = np.asarray(task_wgt, dtype=np.float64).ravel()
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape or task_wgt.shape != task_csf.shape:
        raise ValueError("task arrays must have the same shape")

    if task_scale is not None:
        task_scale = np.asarray(task_scale, dtype=np.float64).ravel()
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    if y0 is not None:
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        if y0.shape != (int(drt.ncsf),):
            raise ValueError("y0 must have shape (ncsf,)")

    out = _ext.epq_apply_weighted_many_atomic(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        task_wgt,
        task_scale if task_scale is not None else None,
        y0 if y0 is not None else None,
        int(threads),
        bool(return_y),
    )
    if not return_y:
        return None
    return np.asarray(out, dtype=np.float64)


def _step_to_occ(step: int) -> int:
    if step == 0:
        return 0
    if step == 3:
        return 2
    return 1


def validate_epq_apply_weighted_many_atomic(
    drt: DRT,
    task_csf: np.ndarray,
    task_p: np.ndarray,
    task_q: np.ndarray,
    task_wgt: np.ndarray,
    *,
    drt_dev=None,
    state_dev=None,
    cache: DRTStateCache | None = None,
    task_scale: np.ndarray | None = None,
    atol: float = 1e-11,
    rtol: float = 1e-11,
) -> None:
    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_p = np.asarray(task_p, dtype=np.int32).ravel()
    task_q = np.asarray(task_q, dtype=np.int32).ravel()
    task_wgt = np.asarray(task_wgt, dtype=np.float64).ravel()
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape or task_wgt.shape != task_csf.shape:
        raise ValueError("task arrays must have the same shape")

    if task_scale is not None:
        task_scale = np.asarray(task_scale, dtype=np.float64).ravel()
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    if drt_dev is None:
        drt_dev = make_device_drt(drt)
    if cache is None:
        cache = get_state_cache(drt)
    if state_dev is None:
        state_dev = make_device_state_cache(drt, drt_dev, cache)

    y_gpu = epq_apply_weighted_many_atomic(
        drt,
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        task_wgt,
        task_scale=task_scale,
        threads=128,
        return_y=True,
    )

    y_cpu = np.zeros(int(drt.ncsf), dtype=np.float64)
    for t in range(int(task_csf.size)):
        csf_idx = int(task_csf[t])
        p = int(task_p[t])
        q = int(task_q[t])
        wgt = float(task_wgt[t])
        scale = float(task_scale[t]) if task_scale is not None else 1.0
        if wgt == 0.0 or scale == 0.0:
            continue

        steps = cache.steps[csf_idx]
        nodes = cache.nodes[csf_idx]

        if p == q:
            occ = _step_to_occ(int(steps[p]))
            if occ:
                y_cpu[csf_idx] += scale * wgt * float(occ)
            continue

        i_idx, coeff = epq_contribs_one(drt, csf_idx, p, q, steps=steps, nodes=nodes)
        if i_idx.size:
            y_cpu[i_idx] += scale * wgt * coeff

    if not np.allclose(y_cpu, y_gpu, atol=float(atol), rtol=float(rtol)):
        diff = np.max(np.abs(y_cpu - y_gpu))
        raise AssertionError(f"y mismatch (max |Δ|={diff})")


def epq_apply_gather_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    p: int,
    q: int,
    x,
    *,
    y=None,
    overflow=None,
    alpha: float = 1.0,
    threads: int = 256,
    add: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """In-place device-array apply of a single generator using destination-gather DP.

    Computes `y <- y + alpha * E_pq * x` (if `add=True`), otherwise `y <- alpha * E_pq * x`.

    Notes
    -----
    Current implementation supports `x.shape == (ncsf, nvec)` with `nvec <= 32` (warp lanes map to vector lanes).
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    p = int(p)
    q = int(q)
    if p < 0 or p >= int(drt.norb) or q < 0 or q >= int(drt.norb):
        raise ValueError("orbital indices out of range")

    x = cp.asarray(x, dtype=cp.float64)
    if x.ndim != 2 or x.shape[0] != int(drt.ncsf):
        raise ValueError("x must have shape (ncsf,nvec)")
    nvec = int(x.shape[1])
    if nvec <= 0 or nvec > 32:
        raise ValueError("epq_apply_gather_inplace_device requires 1 <= nvec <= 32")

    # Allow pitched 2D arrays: stride1=itemsize, stride0>=nvec*itemsize.
    if (
        x.strides[1] != x.itemsize
        or x.strides[0] < nvec * x.itemsize
        or (x.strides[0] % x.itemsize) != 0
        or x.strides[0] <= 0
        or x.strides[1] <= 0
    ):
        x = cp.ascontiguousarray(x)

    if y is None:
        y = cp.empty((int(drt.ncsf), nvec), dtype=cp.float64)
        if add:
            y.fill(0.0)
    else:
        y = cp.asarray(y, dtype=cp.float64)
        if y.shape != (int(drt.ncsf), nvec):
            raise ValueError("y must have the same shape as x")
        if (
            y.strides[1] != y.itemsize
            or y.strides[0] < nvec * y.itemsize
            or (y.strides[0] % y.itemsize) != 0
            or y.strides[0] <= 0
            or y.strides[1] <= 0
        ):
            y = cp.ascontiguousarray(y)

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_apply_gather_inplace_device(
        drt_dev,
        state_dev,
        int(p),
        int(q),
        x,
        y,
        overflow,
        float(alpha),
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return y, overflow


def kernel4_build_w_from_csr_unitnnz_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    row_j,
    row_k,
    csr_rs,
    csr_c,
    x,
    *,
    w_out=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Build `W[k,rs] += x[j] * c(j->k,rs)` from a unit-nnz CSR-like edge list on GPU.

    Each input row corresponds to one nonzero `(j,k,rs,c)` contribution.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
    row_j = cp.ascontiguousarray(row_j)
    row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
    row_k = cp.ascontiguousarray(row_k)
    csr_rs = cp.asarray(csr_rs, dtype=cp.int32).ravel()
    csr_rs = cp.ascontiguousarray(csr_rs)
    csr_c = cp.asarray(csr_c, dtype=cp.float64).ravel()
    csr_c = cp.ascontiguousarray(csr_c)

    if row_j.shape != row_k.shape or row_j.shape != csr_rs.shape or row_j.shape != csr_c.shape:
        raise ValueError("row_j,row_k,csr_rs,csr_c must have the same shape (nrows,)")

    x = cp.asarray(x, dtype=cp.float64).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    if w_out is None:
        w_out = cp.zeros((int(drt.ncsf), int(nops)), dtype=cp.float64)
    else:
        w_out = cp.asarray(w_out, dtype=cp.float64)
        if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
            raise ValueError("w_out must be C-contiguous")
        if w_out.ndim == 1:
            if w_out.shape != (int(drt.ncsf) * int(nops),):
                raise ValueError("w_out (1D) must have shape (ncsf*nops,)")
        elif w_out.ndim == 2:
            if w_out.shape != (int(drt.ncsf), int(nops)):
                raise ValueError("w_out (2D) must have shape (ncsf,nops)")
        else:
            raise ValueError("w_out must be 1D or 2D")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.kernel4_build_w_from_csr_unitnnz_inplace_device(
        drt_dev,
        state_dev,
        row_j,
        row_k,
        csr_rs,
        csr_c,
        x,
        w_out,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return w_out, overflow


def scatter_embed_inplace_device(
    x_sub,
    sub_to_full,
    x_full,
    *,
    threads: int = 128,
    stream=None,
    sync: bool = False,
):
    """Embed a subspace vector into a full-space vector on GPU: x_full[idx[i]] = x_sub[i]."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    try:
        import cupy as cp
    except Exception:  # pragma: no cover
        raise RuntimeError("CuPy is required")

    x_sub = cp.asarray(x_sub, dtype=cp.float64).ravel()
    sub_to_full = cp.asarray(sub_to_full, dtype=cp.int64).ravel()
    x_full = cp.asarray(x_full, dtype=cp.float64).ravel()

    if x_sub.shape != sub_to_full.shape:
        raise ValueError("x_sub and sub_to_full must have the same shape")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.scatter_embed_inplace_device(
        x_sub, sub_to_full, x_full, int(stream_ptr), int(threads)
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def gather_project_inplace_device(
    y_full,
    sub_to_full,
    y_sub,
    *,
    threads: int = 128,
    stream=None,
    sync: bool = False,
):
    """Project a full-space vector into a subspace vector on GPU: y_sub[i] = y_full[idx[i]]."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    try:
        import cupy as cp
    except Exception:  # pragma: no cover
        raise RuntimeError("CuPy is required")

    y_full = cp.asarray(y_full, dtype=cp.float64).ravel()
    sub_to_full = cp.asarray(sub_to_full, dtype=cp.int64).ravel()
    y_sub = cp.asarray(y_sub, dtype=cp.float64).ravel()

    if y_sub.shape != sub_to_full.shape:
        raise ValueError("y_sub and sub_to_full must have the same shape")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.gather_project_inplace_device(
        y_full, sub_to_full, y_sub, int(stream_ptr), int(threads)
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()



def build_w_from_epq_table_inplace_device(
    drt: DRT,
    state_dev,
    epq_table,
    x,
    *,
    dtype=None,
    w_out=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    k_start: int = 0,
    k_count: int = 0,
):
    """Build `W[k,pq] += x[j] * c(j->k,pq)` from the combined off-diagonal E_pq table on GPU.

    `dtype` can be `cp.float64` (default) or `cp.float32`.
    For `dtype=float64`, mixed coefficient storage (`epq_data=float32`) is supported.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    if epq_table is None or len(epq_table) != 4:
        raise ValueError("epq_table must be a (indptr, indices, pq_ids, data) tuple")
    epq_indptr, epq_indices, epq_pq, epq_data = epq_table

    epq_indptr = _as_epq_indptr_array(cp, epq_indptr, ncsf=int(drt.ncsf), name="epq_indptr")
    epq_indices = cp.asarray(epq_indices, dtype=cp.int32).ravel()
    epq_indices = cp.ascontiguousarray(epq_indices)
    epq_pq = _as_epq_pq_array(cp, epq_pq, name="epq_pq")
    _validate_epq_pq_capacity(cp, epq_pq, norb=int(drt.norb), name="epq_pq")
    epq_data = cp.asarray(epq_data).ravel()
    epq_dt = cp.dtype(epq_data.dtype)
    if epq_dt not in (cp.float32, cp.float64):
        epq_data = cp.asarray(epq_data, dtype=fp_dtype).ravel()
        epq_dt = cp.dtype(epq_data.dtype)
    if fp_dtype == cp.float32 and epq_dt != cp.float32:
        epq_data = cp.asarray(epq_data, dtype=cp.float32).ravel()
    epq_data = cp.ascontiguousarray(epq_data)

    if epq_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("epq_indptr must have shape (ncsf+1,)")
    if epq_indices.shape != epq_pq.shape or epq_indices.shape != epq_data.shape:
        raise ValueError("epq_indices,epq_pq,epq_data must have the same shape (nnz,)")

    x = cp.asarray(x, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    expected_rows = int(k_count) if k_count > 0 else int(drt.ncsf)

    if w_out is None:
        w_out = cp.zeros((expected_rows, int(nops)), dtype=fp_dtype)
    else:
        w_out = cp.asarray(w_out, dtype=fp_dtype)
        if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
            raise ValueError("w_out must be C-contiguous")
        if w_out.ndim == 1:
            if w_out.shape != (expected_rows * int(nops),):
                raise ValueError(f"w_out (1D) must have shape ({expected_rows}*nops,)")
        elif w_out.ndim == 2:
            if w_out.shape != (expected_rows, int(nops)):
                raise ValueError(f"w_out (2D) must have shape ({expected_rows},nops)")
        else:
            raise ValueError("w_out must be 1D or 2D")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_from_epq_table_inplace_device(
        state_dev,
        epq_indptr,
        epq_indices,
        epq_pq,
        epq_data,
        x,
        w_out,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        int(k_start),
        int(k_count),
    )
    return w_out, overflow


def build_w_from_epq_transpose_range_inplace_device(
    drt: DRT,
    state_dev,
    epq_table_t,
    x,
    *,
    dtype=None,
    w_out=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    k_start: int = 0,
    k_count: int = 0,
):
    """Build `W_block` from a destination-major EPQ transpose table for `[k_start, k_start+k_count)`."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    if epq_table_t is None or len(epq_table_t) != 4:
        raise ValueError("epq_table_t must be a (t_indptr, t_source, t_pq, t_data) tuple")
    t_indptr, t_source, t_pq, t_data = epq_table_t
    t_indptr = _as_epq_indptr_array(cp, t_indptr, ncsf=int(drt.ncsf), name="t_indptr")
    t_source = cp.ascontiguousarray(cp.asarray(t_source, dtype=cp.int32).ravel())
    t_pq = _as_epq_pq_array(cp, t_pq, name="t_pq")
    _validate_epq_pq_capacity(cp, t_pq, norb=int(drt.norb), name="t_pq")
    t_data = cp.asarray(t_data).ravel()
    t_dt = cp.dtype(t_data.dtype)
    if t_dt not in (cp.float32, cp.float64):
        t_data = cp.asarray(t_data, dtype=fp_dtype).ravel()
        t_dt = cp.dtype(t_data.dtype)
    if fp_dtype == cp.float32 and t_dt != cp.float32:
        t_data = cp.asarray(t_data, dtype=cp.float32).ravel()
    t_data = cp.ascontiguousarray(t_data)

    if t_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("t_indptr must have shape (ncsf+1,)")
    if t_source.shape != t_pq.shape or t_source.shape != t_data.shape:
        raise ValueError("t_source,t_pq,t_data must have the same shape (nnz,)")

    x = cp.ascontiguousarray(cp.asarray(x, dtype=fp_dtype).ravel())
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    k_start = int(k_start)
    if k_start < 0 or k_start >= int(drt.ncsf):
        raise ValueError("k_start out of range")
    if int(k_count) <= 0:
        k_count = int(drt.ncsf) - int(k_start)
    else:
        k_count = int(k_count)
    if k_count <= 0 or k_start + k_count > int(drt.ncsf):
        raise ValueError("k_count out of range")

    if w_out is None:
        w_out = cp.zeros((int(k_count), int(nops)), dtype=fp_dtype)
    else:
        w_out = cp.asarray(w_out, dtype=fp_dtype)
        if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
            raise ValueError("w_out must be C-contiguous")
        if w_out.ndim == 1:
            if w_out.shape != (int(k_count) * int(nops),):
                raise ValueError("w_out (1D) must have shape (k_count*nops,)")
        elif w_out.ndim == 2:
            if w_out.shape[0] != int(k_count) or w_out.shape[1] < int(nops):
                raise ValueError("w_out (2D) must have shape (k_count,nops_or_more)")
        else:
            raise ValueError("w_out must be 1D or 2D")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_from_epq_transpose_range_inplace_device(
        state_dev,
        t_indptr,
        t_source,
        t_pq,
        t_data,
        x,
        w_out,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        int(k_start),
        int(k_count),
    )
    return w_out, overflow


def build_w_from_epq_transpose_range_mm_inplace_device(
    drt: DRT,
    state_dev,
    epq_table_t,
    x,
    *,
    w_out=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    k_start: int = 0,
    k_count: int = 0,
):
    """Build a W_block from a destination-major EPQ transpose table for multiple vectors.

    Computes, for k in [k_start, k_start+k_count) and for each vector lane v:
      W[k_local, pq*nvec+v] = (E_pq |x[:,v]>)[k]

    The diagonal E_pp contributions are included internally using the device steps table.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    if epq_table_t is None or len(epq_table_t) != 4:
        raise ValueError("epq_table_t must be a (t_indptr, t_source, t_pq, t_data) tuple")
    t_indptr, t_source, t_pq, t_data = epq_table_t
    t_indptr = _as_epq_indptr_array(cp, t_indptr, ncsf=int(drt.ncsf), name="t_indptr")
    t_source = cp.ascontiguousarray(cp.asarray(t_source, dtype=cp.int32).ravel())
    t_pq = _as_epq_pq_array(cp, t_pq, name="t_pq")
    _validate_epq_pq_capacity(cp, t_pq, norb=int(drt.norb), name="t_pq")
    t_data = cp.asarray(t_data).ravel()
    t_dt = cp.dtype(t_data.dtype)
    if t_dt not in (cp.float32, cp.float64):
        t_data = cp.asarray(t_data, dtype=cp.float64).ravel()
        t_dt = cp.dtype(t_data.dtype)
    t_data = cp.ascontiguousarray(t_data)

    if t_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("t_indptr must have shape (ncsf+1,)")
    if t_source.shape != t_pq.shape or t_source.shape != t_data.shape:
        raise ValueError("t_source,t_pq,t_data must have the same shape (nnz,)")

    x = cp.asarray(x, dtype=cp.float64)
    if x.ndim != 2 or x.shape[0] != int(drt.ncsf):
        raise ValueError("x must have shape (ncsf,nvec)")
    nvec = int(x.shape[1])
    if nvec <= 0 or nvec > 32:
        raise ValueError("x must satisfy 1 <= nvec <= 32")

    # Require dense row-major layout: stride1==itemsize and stride0>=nvec*itemsize.
    if x.strides[1] != x.itemsize or x.strides[0] < nvec * x.itemsize or (x.strides[0] % x.itemsize) != 0:
        x = cp.ascontiguousarray(x)

    nops = int(drt.norb) * int(drt.norb)
    out_cols = int(nops) * int(nvec)

    k_start = int(k_start)
    if k_start < 0 or k_start >= int(drt.ncsf):
        raise ValueError("k_start out of range")
    if int(k_count) <= 0:
        k_count = int(drt.ncsf) - int(k_start)
    else:
        k_count = int(k_count)
    if k_count <= 0 or k_start + k_count > int(drt.ncsf):
        raise ValueError("k_count out of range")

    if w_out is None:
        w_out = cp.empty((int(k_count), int(out_cols)), dtype=cp.float64)
    else:
        w_out = cp.asarray(w_out, dtype=cp.float64)
        if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
            raise ValueError("w_out must be C-contiguous")
        if w_out.ndim == 1:
            if w_out.shape != (int(k_count) * int(out_cols),):
                raise ValueError("w_out (1D) must have shape (k_count*nops*nvec,)")
        elif w_out.ndim == 2:
            if w_out.shape[0] != int(k_count) or w_out.shape[1] < int(out_cols):
                raise ValueError("w_out (2D) must have shape (k_count, nops*nvec_or_more)")
        else:
            raise ValueError("w_out must be 1D or 2D")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_from_epq_transpose_range_mm_inplace_device(
        state_dev,
        t_indptr,
        t_source,
        t_pq,
        t_data,
        x,
        w_out,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        int(k_start),
        int(k_count),
    )
    return w_out, overflow


def build_w_from_epq_transpose_range_mm_scaled_inplace_device(
    drt: DRT,
    state_dev,
    epq_table_t,
    x,
    hdiag,
    epsa,
    *,
    w_out=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    k_start: int = 0,
    k_count: int = 0,
):
    """Build a (scaled) W_block from a destination-major EPQ transpose table for multiple vectors.

    Computes, for k in [k_start, k_start+k_count) and for each vector lane v:
      W_scaled[k_local, pq*nvec+v] = (E_pq |x[:,v]>)[k] * (hdiag[k] - epsa[p])

    The diagonal E_pp contributions are included internally using the device steps table.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    if epq_table_t is None or len(epq_table_t) != 4:
        raise ValueError("epq_table_t must be a (t_indptr, t_source, t_pq, t_data) tuple")
    t_indptr, t_source, t_pq, t_data = epq_table_t
    t_indptr = _as_epq_indptr_array(cp, t_indptr, ncsf=int(drt.ncsf), name="t_indptr")
    t_source = cp.ascontiguousarray(cp.asarray(t_source, dtype=cp.int32).ravel())
    t_pq = _as_epq_pq_array(cp, t_pq, name="t_pq")
    _validate_epq_pq_capacity(cp, t_pq, norb=int(drt.norb), name="t_pq")
    t_data = cp.asarray(t_data).ravel()
    t_dt = cp.dtype(t_data.dtype)
    if t_dt not in (cp.float32, cp.float64):
        t_data = cp.asarray(t_data, dtype=cp.float64).ravel()
        t_dt = cp.dtype(t_data.dtype)
    t_data = cp.ascontiguousarray(t_data)

    if t_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("t_indptr must have shape (ncsf+1,)")
    if t_source.shape != t_pq.shape or t_source.shape != t_data.shape:
        raise ValueError("t_source,t_pq,t_data must have the same shape (nnz,)")

    x = cp.asarray(x, dtype=cp.float64)
    if x.ndim != 2 or x.shape[0] != int(drt.ncsf):
        raise ValueError("x must have shape (ncsf,nvec)")
    nvec = int(x.shape[1])
    if nvec <= 0 or nvec > 32:
        raise ValueError("x must satisfy 1 <= nvec <= 32")

    # Require dense row-major layout: stride1==itemsize and stride0>=nvec*itemsize.
    if x.strides[1] != x.itemsize or x.strides[0] < nvec * x.itemsize or (x.strides[0] % x.itemsize) != 0:
        x = cp.ascontiguousarray(x)

    hdiag = cp.ascontiguousarray(cp.asarray(hdiag, dtype=cp.float64).ravel())
    if hdiag.shape != (int(drt.ncsf),):
        raise ValueError("hdiag must have shape (ncsf,)")

    epsa = cp.ascontiguousarray(cp.asarray(epsa, dtype=cp.float64).ravel())
    if epsa.shape != (int(drt.norb),):
        raise ValueError("epsa must have shape (norb,)")

    nops = int(drt.norb) * int(drt.norb)
    out_cols = int(nops) * int(nvec)

    k_start = int(k_start)
    if k_start < 0 or k_start >= int(drt.ncsf):
        raise ValueError("k_start out of range")
    if int(k_count) <= 0:
        k_count = int(drt.ncsf) - int(k_start)
    else:
        k_count = int(k_count)
    if k_count <= 0 or k_start + k_count > int(drt.ncsf):
        raise ValueError("k_count out of range")

    if w_out is None:
        w_out = cp.empty((int(k_count), int(out_cols)), dtype=cp.float64)
    else:
        w_out = cp.asarray(w_out, dtype=cp.float64)
        if not getattr(w_out, "flags", None) or not w_out.flags.c_contiguous:
            raise ValueError("w_out must be C-contiguous")
        if w_out.ndim == 1:
            if w_out.shape != (int(k_count) * int(out_cols),):
                raise ValueError("w_out (1D) must have shape (k_count*nops*nvec,)")
        elif w_out.ndim == 2:
            if w_out.shape[0] != int(k_count) or w_out.shape[1] < int(out_cols):
                raise ValueError("w_out (2D) must have shape (k_count, nops*nvec_or_more)")
        else:
            raise ValueError("w_out must be 1D or 2D")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_from_epq_transpose_range_mm_scaled_inplace_device(
        state_dev,
        t_indptr,
        t_source,
        t_pq,
        t_data,
        x,
        hdiag,
        epsa,
        w_out,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        int(k_start),
        int(k_count),
    )
    return w_out, overflow


def apply_g_flat_gather_epq_transpose_range_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    epq_table_t,
    g_block,
    *,
    k_start: int,
    k_count: int,
    y=None,
    overflow=None,
    threads: int = 256,
    add: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    dtype=None,
    use_kahan: bool = False,
):
    """Apply a `g_block[k,pq]` over an EPQ transpose table for source range `[k_start, k_start+k_count)`."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    if epq_table_t is None or len(epq_table_t) != 4:
        raise ValueError("epq_table_t must be a (t_indptr, t_source, t_pq, t_data) tuple")
    t_indptr, t_source, t_pq, t_data = epq_table_t
    t_indptr = _as_epq_indptr_array(cp, t_indptr, ncsf=int(drt.ncsf), name="t_indptr")
    t_source = cp.ascontiguousarray(cp.asarray(t_source, dtype=cp.int32).ravel())
    t_pq = _as_epq_pq_array(cp, t_pq, name="t_pq")
    _validate_epq_pq_capacity(cp, t_pq, norb=int(drt.norb), name="t_pq")
    t_data = cp.asarray(t_data).ravel()
    t_dt = cp.dtype(t_data.dtype)
    if t_dt not in (cp.float32, cp.float64):
        t_data = cp.asarray(t_data, dtype=fp_dtype).ravel()
        t_dt = cp.dtype(t_data.dtype)
    if fp_dtype == cp.float32 and t_dt != cp.float32:
        t_data = cp.asarray(t_data, dtype=cp.float32).ravel()
    t_data = cp.ascontiguousarray(t_data)
    if t_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("t_indptr must have shape (ncsf+1,)")
    if t_source.shape != t_pq.shape or t_source.shape != t_data.shape:
        raise ValueError("t_source,t_pq,t_data must have the same shape (nnz,)")

    k_start = int(k_start)
    k_count = int(k_count)
    if k_start < 0 or k_count <= 0 or (k_start + k_count) > int(drt.ncsf):
        raise ValueError("k_start/k_count out of range")

    nops = int(drt.norb) * int(drt.norb)
    g_block = cp.asarray(g_block, dtype=fp_dtype)
    if not getattr(g_block, "flags", None) or not g_block.flags.c_contiguous:
        raise ValueError("g_block must be C-contiguous")
    if g_block.ndim == 1:
        if g_block.shape != (int(k_count) * int(nops),):
            raise ValueError("g_block (1D) must have shape (k_count*nops,)")
    elif g_block.ndim == 2:
        if g_block.shape[0] != int(k_count) or g_block.shape[1] < int(nops):
            raise ValueError("g_block (2D) must have shape (k_count,nops_or_more)")
    else:
        raise ValueError("g_block must be 1D or 2D")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.ascontiguousarray(cp.asarray(y, dtype=fp_dtype).ravel())
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.apply_g_flat_gather_epq_transpose_range_inplace_device(
        drt_dev,
        state_dev,
        t_indptr,
        t_source,
        t_pq,
        t_data,
        g_block,
        int(k_start),
        int(k_count),
        y,
        overflow,
        int(threads),
        bool(add),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        bool(use_kahan),
    )
    return y, overflow


def build_t_from_epq_table_inplace_device(
    drt: DRT,
    state_dev,
    epq_table,
    c,
    *,
    dtype=None,
    t_out=None,
    overflow=None,
    threads: int = 256,
    zero_out: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Build `T[pq, i] = (E_pq |c>)[i]` from the combined off-diagonal E_pq table on GPU.

    Notes
    -----
    - Off-diagonal rows are accumulated from `epq_table` with dtype-matched atomics.
    - Diagonal rows (p==q) are written from the CSF occupation implied by the DRT step table.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device RDM path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    if epq_table is None or len(epq_table) != 4:
        raise ValueError("epq_table must be a (indptr, indices, pq_ids, data) tuple")
    epq_indptr, epq_indices, epq_pq, epq_data = epq_table

    epq_indptr = cp.asarray(epq_indptr, dtype=cp.int64).ravel()
    epq_indptr = cp.ascontiguousarray(epq_indptr)
    epq_indices = cp.asarray(epq_indices, dtype=cp.int32).ravel()
    epq_indices = cp.ascontiguousarray(epq_indices)
    epq_pq = _as_epq_pq_array(cp, epq_pq, name="epq_pq")
    _validate_epq_pq_capacity(cp, epq_pq, norb=int(drt.norb), name="epq_pq")
    epq_data = cp.asarray(epq_data, dtype=fp_dtype).ravel()
    epq_data = cp.ascontiguousarray(epq_data)

    if epq_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("epq_indptr must have shape (ncsf+1,)")
    if epq_indices.shape != epq_pq.shape or epq_indices.shape != epq_data.shape:
        raise ValueError("epq_indices,epq_pq,epq_data must have the same shape (nnz,)")

    c = cp.asarray(c, dtype=fp_dtype).ravel()
    c = cp.ascontiguousarray(c)
    if c.shape != (int(drt.ncsf),):
        raise ValueError("c must have shape (ncsf,)")

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    if t_out is None:
        t_out = cp.empty((int(nops), int(drt.ncsf)), dtype=fp_dtype)
    else:
        t_out = cp.asarray(t_out, dtype=fp_dtype)
        if not getattr(t_out, "flags", None) or not t_out.flags.c_contiguous:
            raise ValueError("t_out must be C-contiguous")
        if t_out.shape != (int(nops), int(drt.ncsf)):
            raise ValueError("t_out must have shape (nops,ncsf)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_t_from_epq_table_inplace_device(
        state_dev,
        epq_indptr,
        epq_indices,
        epq_pq,
        epq_data,
        c,
        t_out,
        overflow,
        int(threads),
        bool(zero_out),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return t_out, overflow


def epq_contribs_many_count_allpairs_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    *,
    j_start: int,
    j_count: int,
    counts=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Count E_pq off-diagonal contributions for all ordered (p,q) pairs with p!=q.

    Computes counts for tasks in the fixed order:
      tid = j_local * (norb*(norb-1)) + pair
    where pair enumerates all ordered (p,q) with p!=q as used by the CUDA kernels.

    Returns (counts, overflow) device arrays.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start and j_count must be >= 0")
    if j_start > int(drt.ncsf) or j_start + j_count > int(drt.ncsf):
        raise ValueError("j_start/j_count out of range")

    norb = int(drt.norb)
    n_pairs = norb * (norb - 1)
    ntasks = int(j_count) * int(n_pairs)

    if counts is None:
        counts = cp.empty((ntasks,), dtype=cp.int32)
    else:
        counts = cp.asarray(counts, dtype=cp.int32).ravel()
        counts = cp.ascontiguousarray(counts)
        if counts.shape != (ntasks,):
            raise ValueError("counts must have shape (j_count*n_pairs,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_count_allpairs_inplace_device(
        drt_dev,
        state_dev,
        int(j_start),
        int(j_count),
        counts,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return counts, overflow


def epq_contribs_many_write_allpairs_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    *,
    j_start: int,
    j_count: int,
    offsets,
    out_idx,
    out_coeff,
    out_task_pq,
    out_task_csf=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Write E_pq off-diagonal contributions for all ordered (p,q) pairs with p!=q.

    The `offsets` array (int64, shape (ntasks+1,)) defines a packed output layout over tasks.
    For each output entry t, writes:
      out_idx[t]   = csf_i
      out_coeff[t] = <i|E_pq|j>
      out_task_pq[t]  = p*norb+q   (if provided)
      out_task_csf[t] = j          (if provided)

    Returns (out_idx, out_coeff, out_task_pq, out_task_csf, overflow) as device arrays.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start and j_count must be >= 0")
    if j_start > int(drt.ncsf) or j_start + j_count > int(drt.ncsf):
        raise ValueError("j_start/j_count out of range")

    norb = int(drt.norb)
    n_pairs = norb * (norb - 1)
    ntasks = int(j_count) * int(n_pairs)

    offsets = cp.asarray(offsets, dtype=cp.int64).ravel()
    offsets = cp.ascontiguousarray(offsets)
    if offsets.shape != (ntasks + 1,):
        raise ValueError("offsets must have shape (ntasks+1,)")

    out_idx = cp.asarray(out_idx, dtype=cp.int32).ravel()
    out_idx = cp.ascontiguousarray(out_idx)
    out_coeff = cp.asarray(out_coeff).ravel()
    if out_coeff.dtype not in (cp.float32, cp.float64):
        out_coeff = cp.asarray(out_coeff, dtype=cp.float64).ravel()
    out_coeff = cp.ascontiguousarray(out_coeff)
    out_task_pq = _as_epq_pq_array(cp, out_task_pq, name="out_task_pq")
    _validate_epq_pq_capacity(cp, out_task_pq, norb=norb, name="out_task_pq")

    if out_idx.shape != out_coeff.shape or out_idx.shape != out_task_pq.shape:
        raise ValueError("out_idx/out_coeff/out_task_pq must have the same shape")

    if out_task_csf is None:
        out_task_csf_d = None
    else:
        out_task_csf_d = cp.asarray(out_task_csf, dtype=cp.int32).ravel()
        out_task_csf_d = cp.ascontiguousarray(out_task_csf_d)
        if out_task_csf_d.shape != out_idx.shape:
            raise ValueError("out_task_csf must have shape (nnz,) matching out_idx")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_write_allpairs_inplace_device(
        drt_dev,
        state_dev,
        int(j_start),
        int(j_count),
        offsets,
        out_idx,
        out_coeff,
        out_task_csf_d if out_task_csf_d is not None else None,
        out_task_pq,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return out_idx, out_coeff, out_task_pq, out_task_csf_d, overflow


def epq_contribs_many_count_tasks_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    *,
    task_csf,
    task_p,
    task_q,
    counts=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Count E_pq off-diagonal contributions for explicit task lists."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    task_csf = cp.ascontiguousarray(cp.asarray(task_csf, dtype=cp.int32).ravel())
    task_p = cp.ascontiguousarray(cp.asarray(task_p, dtype=cp.int32).ravel())
    task_q = cp.ascontiguousarray(cp.asarray(task_q, dtype=cp.int32).ravel())
    if task_csf.shape != task_p.shape or task_csf.shape != task_q.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")
    ntasks = int(task_csf.size)

    if counts is None:
        counts = cp.empty((ntasks,), dtype=cp.int32)
    else:
        counts = cp.ascontiguousarray(cp.asarray(counts, dtype=cp.int32).ravel())
        if counts.shape != (ntasks,):
            raise ValueError("counts must have shape (ntasks,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_count_tasks_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        counts,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return counts, overflow


def epq_contribs_many_write_tasks_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    *,
    task_csf,
    task_p,
    task_q,
    offsets,
    out_idx,
    out_coeff,
    out_task_pq=None,
    out_task_csf=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Write E_pq off-diagonal contributions for explicit task lists.

    Notes
    -----
    This entrypoint currently requires:
    - `out_coeff.dtype == float64`
    - `out_task_pq.dtype == int32` (if provided)
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    task_csf = cp.ascontiguousarray(cp.asarray(task_csf, dtype=cp.int32).ravel())
    task_p = cp.ascontiguousarray(cp.asarray(task_p, dtype=cp.int32).ravel())
    task_q = cp.ascontiguousarray(cp.asarray(task_q, dtype=cp.int32).ravel())
    if task_csf.shape != task_p.shape or task_csf.shape != task_q.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")
    ntasks = int(task_csf.size)

    offsets = cp.ascontiguousarray(cp.asarray(offsets, dtype=cp.int64).ravel())
    if offsets.shape != (ntasks + 1,):
        raise ValueError("offsets must have shape (ntasks+1,)")

    out_idx = cp.ascontiguousarray(cp.asarray(out_idx, dtype=cp.int32).ravel())
    out_coeff = cp.ascontiguousarray(cp.asarray(out_coeff, dtype=cp.float64).ravel())
    if out_idx.shape != out_coeff.shape:
        raise ValueError("out_idx/out_coeff must have the same shape (nnz,)")

    if out_task_pq is None:
        out_task_pq_d = None
    else:
        out_task_pq_d = cp.ascontiguousarray(cp.asarray(out_task_pq, dtype=cp.int32).ravel())
        if out_task_pq_d.shape != out_idx.shape:
            raise ValueError("out_task_pq must have shape (nnz,) matching out_idx")

    if out_task_csf is None:
        out_task_csf_d = None
    else:
        out_task_csf_d = cp.ascontiguousarray(cp.asarray(out_task_csf, dtype=cp.int32).ravel())
        if out_task_csf_d.shape != out_idx.shape:
            raise ValueError("out_task_csf must have shape (nnz,) matching out_idx")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_write_tasks_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf_d if out_task_csf_d is not None else None,
        out_task_pq_d if out_task_pq_d is not None else None,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return out_idx, out_coeff, out_task_pq_d, out_task_csf_d, overflow


def epq_contribs_many_count_allpairs_recompute_inplace_device(
    drt: DRT,
    drt_dev,
    *,
    ncsf: int | None = None,
    j_start: int,
    j_count: int,
    counts=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    warp_coop: bool = False,
):
    """Count E_pq off-diagonal contributions for all ordered (p,q), reconstructing CSF path on the fly."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    if not has_epq_table_device_build_recompute():
        raise RuntimeError("CUDA extension missing recompute EPQ build entrypoints; rebuild extension")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    ncsf = int(drt.ncsf if ncsf is None else ncsf)
    j_start = int(j_start)
    j_count = int(j_count)
    if ncsf < 0 or j_start < 0 or j_count < 0:
        raise ValueError("ncsf/j_start/j_count must be >= 0")
    if j_start > ncsf or j_start + j_count > ncsf:
        raise ValueError("j_start/j_count out of range")

    norb = int(drt.norb)
    n_pairs = norb * (norb - 1)
    ntasks = int(j_count) * int(n_pairs)

    if counts is None:
        counts = cp.empty((ntasks,), dtype=cp.int32)
    else:
        counts = cp.asarray(counts, dtype=cp.int32).ravel()
        counts = cp.ascontiguousarray(counts)
        if counts.shape != (ntasks,):
            raise ValueError("counts must have shape (j_count*n_pairs,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_count_allpairs_recompute_inplace_device(
        drt_dev,
        int(ncsf),
        int(j_start),
        int(j_count),
        counts,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        bool(warp_coop),
    )
    return counts, overflow


def epq_contribs_many_write_allpairs_recompute_inplace_device(
    drt: DRT,
    drt_dev,
    *,
    ncsf: int | None = None,
    j_start: int,
    j_count: int,
    offsets,
    out_idx,
    out_coeff,
    out_task_pq,
    out_task_csf=None,
    overflow=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    warp_coop: bool = False,
):
    """Write E_pq off-diagonal contributions for all ordered (p,q), reconstructing CSF path on the fly."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    if not has_epq_table_device_build_recompute():
        raise RuntimeError("CUDA extension missing recompute EPQ build entrypoints; rebuild extension")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    ncsf = int(drt.ncsf if ncsf is None else ncsf)
    j_start = int(j_start)
    j_count = int(j_count)
    if ncsf < 0 or j_start < 0 or j_count < 0:
        raise ValueError("ncsf/j_start/j_count must be >= 0")
    if j_start > ncsf or j_start + j_count > ncsf:
        raise ValueError("j_start/j_count out of range")

    norb = int(drt.norb)
    n_pairs = norb * (norb - 1)
    ntasks = int(j_count) * int(n_pairs)

    offsets = cp.asarray(offsets, dtype=cp.int64).ravel()
    offsets = cp.ascontiguousarray(offsets)
    if offsets.shape != (ntasks + 1,):
        raise ValueError("offsets must have shape (ntasks+1,)")

    out_idx = cp.asarray(out_idx, dtype=cp.int32).ravel()
    out_idx = cp.ascontiguousarray(out_idx)
    out_coeff = cp.asarray(out_coeff).ravel()
    if out_coeff.dtype not in (cp.float32, cp.float64):
        out_coeff = cp.asarray(out_coeff, dtype=cp.float64).ravel()
    out_coeff = cp.ascontiguousarray(out_coeff)
    out_task_pq = _as_epq_pq_array(cp, out_task_pq, name="out_task_pq")
    _validate_epq_pq_capacity(cp, out_task_pq, norb=norb, name="out_task_pq")
    if out_idx.shape != out_coeff.shape or out_idx.shape != out_task_pq.shape:
        raise ValueError("out_idx/out_coeff/out_task_pq must have the same shape")

    if out_task_csf is None:
        out_task_csf_d = None
    else:
        out_task_csf_d = cp.asarray(out_task_csf, dtype=cp.int32).ravel()
        out_task_csf_d = cp.ascontiguousarray(out_task_csf_d)
        if out_task_csf_d.shape != out_idx.shape:
            raise ValueError("out_task_csf must have shape (nnz,) matching out_idx")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.epq_contribs_many_write_allpairs_recompute_inplace_device(
        drt_dev,
        int(ncsf),
        int(j_start),
        int(j_count),
        offsets,
        out_idx,
        out_coeff,
        out_task_csf_d if out_task_csf_d is not None else None,
        out_task_pq,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        bool(warp_coop),
    )
    return out_idx, out_coeff, out_task_pq, out_task_csf_d, overflow


def _epq_pq_dtype_for_norb(cp, norb: int):
    """Return compact dtype for pq_id = p*norb + q."""
    nops = int(norb) * int(norb)
    if nops <= 256:
        return cp.uint8
    if nops <= 65535:
        return cp.uint16
    return cp.int32


def _as_epq_pq_array(cp, arr, *, name: str = "epq_pq"):
    """Normalize EPQ pq-id array while preserving compact integer dtype."""
    out = cp.asarray(arr).ravel()
    out = cp.ascontiguousarray(out)
    dt = cp.dtype(out.dtype)
    allowed = (cp.dtype(cp.uint8), cp.dtype(cp.uint16), cp.dtype(cp.int32))
    if dt not in allowed:
        raise ValueError(f"{name} must have dtype uint8, uint16, or int32")
    return out


def _validate_epq_pq_capacity(cp, arr, *, norb: int, name: str = "epq_pq") -> None:
    """Ensure the chosen pq dtype can represent all `p*norb+q` ids."""
    nops = int(norb) * int(norb)
    dt = cp.dtype(arr.dtype)
    if dt == cp.dtype(cp.uint8) and nops > 256:
        raise ValueError(f"{name} dtype uint8 is too small for norb={int(norb)}")
    if dt == cp.dtype(cp.uint16) and nops > 65535:
        raise ValueError(f"{name} dtype uint16 is too small for norb={int(norb)}")


def _as_epq_indptr_array(cp, arr, *, ncsf: int, name: str = "epq_indptr"):
    """Normalize EPQ indptr array preserving int32/int64 storage."""
    out = cp.asarray(arr).ravel()
    out = cp.ascontiguousarray(out)
    dt = cp.dtype(out.dtype)
    if dt not in (cp.dtype(cp.int32), cp.dtype(cp.int64)):
        raise ValueError(f"{name} must have dtype int32 or int64")
    if out.shape != (int(ncsf) + 1,):
        raise ValueError(f"{name} must have shape (ncsf+1,)")
    return out


def build_epq_action_table_combined_device(
    drt: DRT,
    drt_dev,
    state_dev=None,
    *,
    j_tile: int = 0,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    use_cache: bool = True,
    use_recompute: bool | str = "auto",
    recompute_warp_coop: bool = False,
    dtype=None,
    indptr_dtype: str | None = "auto",
):
    """Build the combined off-diagonal E_pq action table on GPU.

    Returns a CSR-like epq_table tuple `(indptr, indices, pq_ids, data)` on the device:
      indptr[j]:indptr[j+1] gives entries (indices[t], pq_ids[t], data[t]).

    Notes
    -----
    This is intended to remove the CPU bottleneck of building `epq_table` for the CUDA matvec path.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64 for EPQ table build")
    indptr_mode = _normalize_epq_indptr_mode(indptr_dtype)

    cache_attr = (
        "_epq_action_table_combined_device"
        if fp_dtype == cp.float64
        else "_epq_action_table_combined_device_f32"
    )
    if bool(use_cache):
        cached = getattr(drt, cache_attr, None)
        if cached is not None:
            try:
                cached_indptr = cp.asarray(cached[0])
                cached_dt = cp.dtype(cached_indptr.dtype)
                if indptr_mode == "auto" or (
                    indptr_mode == "int32" and cached_dt == cp.dtype(cp.int32)
                ) or (
                    indptr_mode == "int64" and cached_dt == cp.dtype(cp.int64)
                ):
                    return cached
            except Exception:
                pass

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    if ncsf <= 0 or norb <= 0:
        raise ValueError("invalid DRT sizes")

    n_pairs = norb * (norb - 1)
    pq_dtype = _epq_pq_dtype_for_norb(cp, norb)
    if n_pairs <= 0:
        # No off-diagonal operators.
        indptr_dt = _epq_indptr_cp_dtype_for_total_nnz(cp, mode=indptr_mode, total_nnz=0)
        indptr = cp.zeros((ncsf + 1,), dtype=indptr_dt)
        indices = cp.empty((0,), dtype=cp.int32)
        pq_ids = cp.empty((0,), dtype=pq_dtype)
        data = cp.empty((0,), dtype=fp_dtype)
        out = (indptr, indices, pq_ids, data)
        if bool(use_cache):
            setattr(drt, cache_attr, out)
        return out

    ntasks = int(ncsf) * int(n_pairs)
    if ntasks <= 0:
        raise ValueError("invalid ntasks for epq_table build")

    if j_tile <= 0:
        # Target a few million tasks per tile to amortize kernel launch overhead while
        # keeping the temporary (ntasks_tile+1) offsets array modest.
        target_tasks = 2_000_000
        j_tile = max(1, int((target_tasks + n_pairs - 1) // n_pairs))
        j_tile_align = 256
        if j_tile_align > 1 and j_tile < ncsf:
            j_tile = min(ncsf, ((j_tile + j_tile_align - 1) // j_tile_align) * j_tile_align)
    j_tile = min(ncsf, int(j_tile))

    if stream is None:
        stream_obj = cp.cuda.get_current_stream()
        stream_ptr = int(stream_obj.ptr)
    else:
        stream_obj = stream if hasattr(stream, "synchronize") else None
        stream_ptr = int(getattr(stream, "ptr", stream))

    if isinstance(use_recompute, str):
        if use_recompute != "auto":
            raise ValueError("use_recompute must be bool or 'auto'")
        state_bytes = int(ncsf) * int(norb) * 1 + int(ncsf) * int(norb + 1) * 4
        use_recompute = bool(has_epq_table_device_build_recompute() and state_bytes >= (64 << 20))
    else:
        use_recompute = bool(use_recompute)

    if use_recompute and not has_epq_table_device_build_recompute():
        raise RuntimeError("Requested use_recompute=True but extension lacks recompute entrypoints")
    if (not use_recompute) and state_dev is None:
        raise ValueError("state_dev is required when use_recompute=False")
    recompute_warp_coop = bool(recompute_warp_coop)

    # 1) Count for all (j,p,q) tasks. We keep counts to avoid a second DP pass.
    counts = cp.empty((ntasks,), dtype=cp.int32)
    overflow = cp.empty((1,), dtype=cp.int32)
    if use_recompute:
        epq_contribs_many_count_allpairs_recompute_inplace_device(
            drt,
            drt_dev,
            ncsf=ncsf,
            j_start=0,
            j_count=ncsf,
            counts=counts,
            overflow=overflow,
            threads=int(threads),
            stream=stream_ptr,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            warp_coop=bool(recompute_warp_coop),
        )
    else:
        epq_contribs_many_count_allpairs_inplace_device(
            drt,
            drt_dev,
            state_dev,
            j_start=0,
            j_count=ncsf,
            counts=counts,
            overflow=overflow,
            threads=int(threads),
            stream=stream_ptr,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
        )

    # 2) Reduce counts per CSF and build indptr.
    counts2 = counts.reshape(ncsf, n_pairs)
    nnz_per_csf = counts2.sum(axis=1, dtype=cp.int64)
    indptr = cp.empty((ncsf + 1,), dtype=cp.int64)
    indptr[0] = 0
    cp.cumsum(nnz_per_csf, out=indptr[1:])
    if bool(sync) and stream_obj is not None:
        stream_obj.synchronize()

    indptr_h = np.asarray(cp.asnumpy(indptr), dtype=np.int64)
    total_nnz = int(indptr_h[-1])
    if total_nnz < 0:
        raise RuntimeError("invalid total nnz for epq_table build")
    indptr_cp_dtype = _epq_indptr_cp_dtype_for_total_nnz(
        cp, mode=indptr_mode, total_nnz=total_nnz
    )

    indices = cp.empty((total_nnz,), dtype=cp.int32)
    pq_ids = cp.empty((total_nnz,), dtype=pq_dtype)
    data = cp.empty((total_nnz,), dtype=fp_dtype)

    # 3) Write per tile. Offsets are computed per tile to avoid a huge global offsets array.
    for j0 in range(0, ncsf, j_tile):
        jc = min(j_tile, ncsf - j0)
        base = int(indptr_h[j0])
        end = int(indptr_h[j0 + jc])
        tile_nnz = int(end - base)
        if tile_nnz <= 0:
            continue

        counts_tile = counts[(int(j0) * int(n_pairs)) : (int(j0 + jc) * int(n_pairs))]
        offsets = cp.empty((int(jc) * int(n_pairs) + 1,), dtype=cp.int64)
        offsets[0] = 0
        cp.cumsum(counts_tile, dtype=cp.int64, out=offsets[1:])

        out_idx = indices[base:end]
        out_pq = pq_ids[base:end]
        out_coeff = data[base:end]

        if use_recompute:
            epq_contribs_many_write_allpairs_recompute_inplace_device(
                drt,
                drt_dev,
                ncsf=ncsf,
                j_start=int(j0),
                j_count=int(jc),
                offsets=offsets,
                out_idx=out_idx,
                out_coeff=out_coeff,
                out_task_pq=out_pq,
                out_task_csf=None,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                warp_coop=bool(recompute_warp_coop),
            )
        else:
            epq_contribs_many_write_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=int(j0),
                j_count=int(jc),
                offsets=offsets,
                out_idx=out_idx,
                out_coeff=out_coeff,
                out_task_pq=out_pq,
                out_task_csf=None,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
            )

        if bool(sync) and stream_obj is not None:
            stream_obj.synchronize()
            wrote = int(cp.asnumpy(offsets[-1]))
            if wrote != tile_nnz:
                raise RuntimeError(f"epq_table write mismatch: tile nnz {tile_nnz} vs offsets[-1] {wrote}")

    if cp.dtype(indptr_cp_dtype) == cp.dtype(cp.int64):
        out_indptr = indptr
    else:
        out_indptr = cp.ascontiguousarray(indptr.astype(cp.int32, copy=False))
    out = (out_indptr, indices, pq_ids, data)
    if bool(use_cache):
        setattr(drt, cache_attr, out)
    return out


def build_epq_action_table_combined_device_tiled(
    drt: DRT,
    drt_dev,
    state_dev=None,
    *,
    j_tile: int = 0,
    build_tile: int = 0,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    use_cache: bool = True,
    use_recompute: bool | str = "auto",
    recompute_warp_coop: bool = False,
    dtype=None,
    indptr_dtype: str | None = "auto",
):
    """Build EPQ table with O(build_tile*n_pairs) scratch instead of O(ncsf*n_pairs)."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64 for EPQ table build")
    indptr_mode = _normalize_epq_indptr_mode(indptr_dtype)

    cache_attr = (
        "_epq_action_table_combined_device"
        if fp_dtype == cp.float64
        else "_epq_action_table_combined_device_f32"
    )
    if bool(use_cache):
        cached = getattr(drt, cache_attr, None)
        if cached is not None:
            try:
                cached_indptr = cp.asarray(cached[0])
                cached_dt = cp.dtype(cached_indptr.dtype)
                if indptr_mode == "auto" or (
                    indptr_mode == "int32" and cached_dt == cp.dtype(cp.int32)
                ) or (
                    indptr_mode == "int64" and cached_dt == cp.dtype(cp.int64)
                ):
                    return cached
            except Exception:
                pass

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    if ncsf <= 0 or norb <= 0:
        raise ValueError("invalid DRT sizes")

    n_pairs = norb * (norb - 1)
    pq_dtype = _epq_pq_dtype_for_norb(cp, norb)
    if n_pairs <= 0:
        indptr_dt = _epq_indptr_cp_dtype_for_total_nnz(cp, mode=indptr_mode, total_nnz=0)
        indptr = cp.zeros((ncsf + 1,), dtype=indptr_dt)
        indices = cp.empty((0,), dtype=cp.int32)
        pq_ids = cp.empty((0,), dtype=pq_dtype)
        data = cp.empty((0,), dtype=fp_dtype)
        out = (indptr, indices, pq_ids, data)
        if bool(use_cache):
            setattr(drt, cache_attr, out)
        return out

    if stream is None:
        stream_obj = cp.cuda.get_current_stream()
        stream_ptr = int(stream_obj.ptr)
    else:
        stream_obj = stream if hasattr(stream, "synchronize") else None
        stream_ptr = int(getattr(stream, "ptr", stream))

    if isinstance(use_recompute, str):
        if use_recompute != "auto":
            raise ValueError("use_recompute must be bool or 'auto'")
        state_bytes = int(ncsf) * int(norb) * 1 + int(ncsf) * int(norb + 1) * 4
        use_recompute = bool(has_epq_table_device_build_recompute() and state_bytes >= (64 << 20))
    else:
        use_recompute = bool(use_recompute)

    if use_recompute and not has_epq_table_device_build_recompute():
        raise RuntimeError("Requested use_recompute=True but extension lacks recompute entrypoints")
    if (not use_recompute) and state_dev is None:
        raise ValueError("state_dev is required when use_recompute=False")
    recompute_warp_coop = bool(recompute_warp_coop)

    if build_tile <= 0:
        auto_bt = max(8192, int((256 * 1024) // max(1, int(n_pairs))))
        build_tile = min(int(ncsf), int(auto_bt))
    build_tile = max(1, min(int(ncsf), int(build_tile)))
    if j_tile <= 0:
        j_tile = int(build_tile)
    j_tile = max(1, min(int(ncsf), int(j_tile)))

    overflow = cp.empty((1,), dtype=cp.int32)
    scratch_cap = int(build_tile) * int(n_pairs)
    counts_scratch = cp.empty((scratch_cap,), dtype=cp.int32)
    nnz_per_csf_host = np.zeros((int(ncsf),), dtype=np.int64)

    # Pass 1: per-tile count -> host nnz_per_csf.
    for j0 in range(0, int(ncsf), int(build_tile)):
        j1 = min(int(ncsf), int(j0 + int(build_tile)))
        jc = int(j1 - j0)
        counts_tile = counts_scratch[: int(jc) * int(n_pairs)]
        if use_recompute:
            epq_contribs_many_count_allpairs_recompute_inplace_device(
                drt,
                drt_dev,
                ncsf=ncsf,
                j_start=int(j0),
                j_count=int(jc),
                counts=counts_tile,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
                warp_coop=bool(recompute_warp_coop),
            )
        else:
            epq_contribs_many_count_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=int(j0),
                j_count=int(jc),
                counts=counts_tile,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
            )
        nnz_tile = counts_tile.reshape(int(jc), int(n_pairs)).sum(axis=1, dtype=cp.int64)
        nnz_per_csf_host[int(j0) : int(j1)] = np.asarray(cp.asnumpy(nnz_tile), dtype=np.int64)

    indptr_host = np.zeros((int(ncsf) + 1,), dtype=np.int64)
    np.cumsum(nnz_per_csf_host, out=indptr_host[1:])
    total_nnz = int(indptr_host[-1])
    if total_nnz < 0:
        raise RuntimeError("invalid total nnz for epq_table build")
    indptr_cp_dtype = _epq_indptr_cp_dtype_for_total_nnz(
        cp, mode=indptr_mode, total_nnz=total_nnz
    )

    indices = cp.empty((int(total_nnz),), dtype=cp.int32)
    pq_ids = cp.empty((int(total_nnz),), dtype=pq_dtype)
    data = cp.empty((int(total_nnz),), dtype=fp_dtype)

    # Pass 2: recount tile + write directly to global outputs.
    for j0 in range(0, int(ncsf), int(j_tile)):
        j1 = min(int(ncsf), int(j0 + int(j_tile)))
        jc = int(j1 - j0)
        base = int(indptr_host[int(j0)])
        end = int(indptr_host[int(j1)])
        tile_nnz = int(end - base)
        if tile_nnz <= 0:
            continue

        counts_tile = counts_scratch[: int(jc) * int(n_pairs)]
        if use_recompute:
            epq_contribs_many_count_allpairs_recompute_inplace_device(
                drt,
                drt_dev,
                ncsf=ncsf,
                j_start=int(j0),
                j_count=int(jc),
                counts=counts_tile,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
                warp_coop=bool(recompute_warp_coop),
            )
        else:
            epq_contribs_many_count_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=int(j0),
                j_count=int(jc),
                counts=counts_tile,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
            )

        offsets = cp.empty((int(jc) * int(n_pairs) + 1,), dtype=cp.int64)
        offsets[0] = 0
        cp.cumsum(counts_tile, dtype=cp.int64, out=offsets[1:])

        out_idx = indices[base:end]
        out_pq = pq_ids[base:end]
        out_coeff = data[base:end]

        if use_recompute:
            epq_contribs_many_write_allpairs_recompute_inplace_device(
                drt,
                drt_dev,
                ncsf=ncsf,
                j_start=int(j0),
                j_count=int(jc),
                offsets=offsets,
                out_idx=out_idx,
                out_coeff=out_coeff,
                out_task_pq=out_pq,
                out_task_csf=None,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
                warp_coop=bool(recompute_warp_coop),
            )
        else:
            epq_contribs_many_write_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=int(j0),
                j_count=int(jc),
                offsets=offsets,
                out_idx=out_idx,
                out_coeff=out_coeff,
                out_task_pq=out_pq,
                out_task_csf=None,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
            )

        wrote = int(cp.asnumpy(offsets[-1]))
        if wrote != int(tile_nnz):
            raise RuntimeError(f"epq_table write mismatch: tile nnz {tile_nnz} vs offsets[-1] {wrote}")

    indptr = cp.ascontiguousarray(cp.asarray(indptr_host, dtype=indptr_cp_dtype))
    out = (indptr, indices, pq_ids, data)
    if bool(use_cache):
        setattr(drt, cache_attr, out)
    return out


def build_epq_action_table_tile_device(
    drt: DRT,
    drt_dev,
    state_dev=None,
    *,
    j_start: int,
    j_count: int,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    use_recompute: bool | str = "auto",
    recompute_warp_coop: bool = False,
    global_indptr: bool = True,
    pq_block: int = 0,
    dtype=None,
):
    """Build a j-range EPQ table tile on device.

    Parameters
    ----------
    j_start, j_count
        Source-row interval `[j_start, j_start + j_count)`.
    global_indptr
        If True, returns an `indptr` of shape `(ncsf+1,)` with only the selected
        row range populated (other rows are empty). This is convenient for existing
        kernels that expect global CSF row indices.
    pq_block
        Optional 2D tiling width over flattened `pq_id=p*norb+q` space. When >0 and
        smaller than `norb*norb`, build uses explicit task-list count/write kernels
        per pq-block to reduce temporary scratch for large `norb`.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64 for EPQ tile build")
    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start and j_count must be >= 0")
    if j_start > ncsf or j_start + j_count > ncsf:
        raise ValueError("j_start/j_count out of range")

    n_pairs = int(norb) * int(norb - 1)
    pq_dtype = _epq_pq_dtype_for_norb(cp, norb)
    if n_pairs <= 0:
        indptr_len = ncsf + 1 if bool(global_indptr) else j_count + 1
        indptr = cp.zeros((int(indptr_len),), dtype=cp.int64)
        indices = cp.empty((0,), dtype=cp.int32)
        pq_ids = cp.empty((0,), dtype=pq_dtype)
        data = cp.empty((0,), dtype=fp_dtype)
        return indptr, indices, pq_ids, data

    if stream is None:
        stream_obj = cp.cuda.get_current_stream()
        stream_ptr = int(stream_obj.ptr)
    else:
        stream_obj = stream if hasattr(stream, "synchronize") else None
        stream_ptr = int(getattr(stream, "ptr", stream))

    if isinstance(use_recompute, str):
        if use_recompute != "auto":
            raise ValueError("use_recompute must be bool or 'auto'")
        state_bytes = int(ncsf) * int(norb) * 1 + int(ncsf) * int(norb + 1) * 4
        use_recompute = bool(has_epq_table_device_build_recompute() and state_bytes >= (64 << 20))
    else:
        use_recompute = bool(use_recompute)

    if use_recompute and not has_epq_table_device_build_recompute():
        raise RuntimeError("Requested use_recompute=True but extension lacks recompute entrypoints")
    if (not use_recompute) and state_dev is None:
        raise ValueError("state_dev is required when use_recompute=False")
    recompute_warp_coop = bool(recompute_warp_coop)

    nops_total = int(norb) * int(norb)
    pq_block = int(pq_block)
    if pq_block < 0:
        raise ValueError("pq_block must be >= 0")
    use_pq_block = bool(pq_block > 0 and pq_block < nops_total)
    if use_pq_block and bool(use_recompute):
        raise ValueError("pq_block tiling currently supports use_recompute=False only")
    if use_pq_block and not bool(sync):
        raise ValueError("pq_block tiling currently requires sync=True")
    if use_pq_block:
        if _ext is None or not hasattr(_ext, "epq_contribs_many_count_tasks_inplace_device"):
            raise RuntimeError("pq_block tiling requires task-list EPQ count/write extension entrypoints")

    overflow = cp.empty((1,), dtype=cp.int32)

    row_ids = cp.arange(int(j_start), int(j_start + j_count), dtype=cp.int32)

    if use_pq_block:
        nnz_per_row = cp.zeros((j_count,), dtype=cp.int64)
        for pq0 in range(0, int(nops_total), int(pq_block)):
            pq1 = min(int(nops_total), int(pq0 + int(pq_block)))
            pq_ids_blk = cp.arange(int(pq0), int(pq1), dtype=cp.int32)
            p_blk = pq_ids_blk // int(norb)
            q_blk = pq_ids_blk - p_blk * int(norb)
            keep = p_blk != q_blk
            p_blk = cp.ascontiguousarray(p_blk[keep])
            q_blk = cp.ascontiguousarray(q_blk[keep])
            n_pairs_blk = int(p_blk.size)
            if n_pairs_blk <= 0:
                continue
            task_csf_blk = cp.repeat(row_ids, int(n_pairs_blk))
            task_p_blk = cp.tile(p_blk, int(j_count))
            task_q_blk = cp.tile(q_blk, int(j_count))
            ntasks_blk = int(task_csf_blk.size)
            if ntasks_blk <= 0:
                continue

            counts_blk = cp.empty((ntasks_blk,), dtype=cp.int32)
            epq_contribs_many_count_tasks_inplace_device(
                drt,
                drt_dev,
                state_dev,
                task_csf=task_csf_blk,
                task_p=task_p_blk,
                task_q=task_q_blk,
                counts=counts_blk,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=True,
                check_overflow=bool(check_overflow),
            )
            counts2_blk = counts_blk.reshape(int(j_count), int(n_pairs_blk))
            nnz_per_row += counts2_blk.sum(axis=1, dtype=cp.int64)
    else:
        ntasks = int(j_count) * int(n_pairs)
        counts = cp.empty((ntasks,), dtype=cp.int32)

        if use_recompute:
            epq_contribs_many_count_allpairs_recompute_inplace_device(
                drt,
                drt_dev,
                ncsf=ncsf,
                j_start=j_start,
                j_count=j_count,
                counts=counts,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                warp_coop=bool(recompute_warp_coop),
            )
        else:
            epq_contribs_many_count_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=j_start,
                j_count=j_count,
                counts=counts,
                overflow=overflow,
                threads=int(threads),
                stream=stream_ptr,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
            )

        counts2 = counts.reshape(j_count, n_pairs)
        nnz_per_row = counts2.sum(axis=1, dtype=cp.int64)

    local_indptr = cp.empty((j_count + 1,), dtype=cp.int64)
    local_indptr[0] = 0
    cp.cumsum(nnz_per_row, out=local_indptr[1:])
    if bool(sync) and stream_obj is not None:
        stream_obj.synchronize()

    tile_nnz = int(cp.asnumpy(local_indptr[-1])) if j_count > 0 else 0
    indices = cp.empty((tile_nnz,), dtype=cp.int32)
    pq_ids = cp.empty((tile_nnz,), dtype=pq_dtype)
    data = cp.empty((tile_nnz,), dtype=fp_dtype)

    if use_pq_block:
        if tile_nnz > 0:
            pack_k = _get_epq_tile_row_pack_kernel(cp, pq_dtype=pq_dtype, fp_dtype=fp_dtype)
            row_cursor = local_indptr[:-1].copy()
            for pq0 in range(0, int(nops_total), int(pq_block)):
                pq1 = min(int(nops_total), int(pq0 + int(pq_block)))
                pq_ids_blk = cp.arange(int(pq0), int(pq1), dtype=cp.int32)
                p_blk = pq_ids_blk // int(norb)
                q_blk = pq_ids_blk - p_blk * int(norb)
                keep = p_blk != q_blk
                p_blk = cp.ascontiguousarray(p_blk[keep])
                q_blk = cp.ascontiguousarray(q_blk[keep])
                n_pairs_blk = int(p_blk.size)
                if n_pairs_blk <= 0:
                    continue

                task_csf_blk = cp.repeat(row_ids, int(n_pairs_blk))
                task_p_blk = cp.tile(p_blk, int(j_count))
                task_q_blk = cp.tile(q_blk, int(j_count))
                ntasks_blk = int(task_csf_blk.size)
                if ntasks_blk <= 0:
                    continue

                counts_blk = cp.empty((ntasks_blk,), dtype=cp.int32)
                epq_contribs_many_count_tasks_inplace_device(
                    drt,
                    drt_dev,
                    state_dev,
                    task_csf=task_csf_blk,
                    task_p=task_p_blk,
                    task_q=task_q_blk,
                    counts=counts_blk,
                    overflow=overflow,
                    threads=int(threads),
                    stream=stream_ptr,
                    sync=True,
                    check_overflow=bool(check_overflow),
                )
                counts2_blk = counts_blk.reshape(int(j_count), int(n_pairs_blk))
                row_lens = counts2_blk.sum(axis=1, dtype=cp.int64)
                nnz_blk = int(cp.asnumpy(row_lens.sum()))
                if nnz_blk <= 0:
                    continue

                offsets_blk = cp.empty((ntasks_blk + 1,), dtype=cp.int64)
                offsets_blk[0] = 0
                cp.cumsum(counts_blk, dtype=cp.int64, out=offsets_blk[1:])
                wrote = int(cp.asnumpy(offsets_blk[-1]))
                if wrote != int(nnz_blk):
                    raise RuntimeError(f"epq tile write mismatch in pq-block: nnz {nnz_blk} vs offsets[-1] {wrote}")

                tmp_idx = cp.empty((int(nnz_blk),), dtype=cp.int32)
                tmp_pq32 = cp.empty((int(nnz_blk),), dtype=cp.int32)
                tmp_coeff64 = cp.empty((int(nnz_blk),), dtype=cp.float64)
                epq_contribs_many_write_tasks_inplace_device(
                    drt,
                    drt_dev,
                    state_dev,
                    task_csf=task_csf_blk,
                    task_p=task_p_blk,
                    task_q=task_q_blk,
                    offsets=offsets_blk,
                    out_idx=tmp_idx,
                    out_coeff=tmp_coeff64,
                    out_task_pq=tmp_pq32,
                    out_task_csf=None,
                    overflow=overflow,
                    threads=int(threads),
                    stream=stream_ptr,
                    sync=True,
                    check_overflow=bool(check_overflow),
                )

                row_src_ptr = cp.empty((int(j_count) + 1,), dtype=cp.int64)
                row_src_ptr[0] = 0
                cp.cumsum(row_lens, out=row_src_ptr[1:])
                row_src_start = row_src_ptr[:-1]
                pack_k(
                    (int(j_count),),
                    (256,),
                    (
                        row_src_start,
                        row_cursor,
                        row_lens,
                        tmp_idx,
                        tmp_pq32,
                        tmp_coeff64,
                        indices,
                        pq_ids,
                        data,
                        int(j_count),
                    ),
                )
                row_cursor = row_cursor + row_lens

            if bool(sync):
                done = bool(cp.all(row_cursor == local_indptr[1:]))
                if not done:
                    raise RuntimeError("epq tile pq-block packing produced inconsistent row cursors")
    else:
        ntasks = int(j_count) * int(n_pairs)
        if ntasks > 0:
            offsets = cp.empty((ntasks + 1,), dtype=cp.int64)
            offsets[0] = 0
            cp.cumsum(counts, dtype=cp.int64, out=offsets[1:])
            out_coeff = data
            if use_recompute:
                epq_contribs_many_write_allpairs_recompute_inplace_device(
                    drt,
                    drt_dev,
                    ncsf=ncsf,
                    j_start=j_start,
                    j_count=j_count,
                    offsets=offsets,
                    out_idx=indices,
                    out_coeff=out_coeff,
                    out_task_pq=pq_ids,
                    out_task_csf=None,
                    overflow=overflow,
                    threads=int(threads),
                    stream=stream_ptr,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                    warp_coop=bool(recompute_warp_coop),
                )
            else:
                epq_contribs_many_write_allpairs_inplace_device(
                    drt,
                    drt_dev,
                    state_dev,
                    j_start=j_start,
                    j_count=j_count,
                    offsets=offsets,
                    out_idx=indices,
                    out_coeff=out_coeff,
                    out_task_pq=pq_ids,
                    out_task_csf=None,
                    overflow=overflow,
                    threads=int(threads),
                    stream=stream_ptr,
                    sync=bool(sync),
                    check_overflow=bool(check_overflow),
                )
            if bool(sync) and stream_obj is not None:
                stream_obj.synchronize()
                wrote = int(cp.asnumpy(offsets[-1]))
                if wrote != tile_nnz:
                    raise RuntimeError(f"epq tile write mismatch: tile nnz {tile_nnz} vs offsets[-1] {wrote}")

    if not bool(global_indptr):
        return local_indptr, indices, pq_ids, data

    indptr = cp.zeros((ncsf + 1,), dtype=cp.int64)
    if j_count > 0:
        indptr[int(j_start) : int(j_start + j_count + 1)] = local_indptr
    if int(j_start + j_count) < ncsf:
        indptr[int(j_start + j_count + 1) :] = np.int64(tile_nnz)
    return indptr, indices, pq_ids, data


_EPQ_TRANSPOSE_ATOMIC_FILL_KERNELS = {}


def _get_epq_transpose_atomic_fill_kernel(cp, *, pq_dtype, fp_dtype, indptr_dtype):
    key = (str(cp.dtype(pq_dtype)), str(cp.dtype(fp_dtype)), str(cp.dtype(indptr_dtype)))
    k = _EPQ_TRANSPOSE_ATOMIC_FILL_KERNELS.get(key)
    if k is not None:
        return k

    pq_dtype = cp.dtype(pq_dtype)
    fp_dtype = cp.dtype(fp_dtype)

    if pq_dtype == cp.uint8:
        pq_c = "unsigned char"
        pq_tag = "u1"
    elif pq_dtype == cp.uint16:
        pq_c = "unsigned short"
        pq_tag = "u2"
    elif pq_dtype == cp.int32:
        pq_c = "int"
        pq_tag = "i4"
    else:
        raise ValueError(f"Unsupported pq_dtype for EPQ transpose: {pq_dtype!r}")

    if fp_dtype == cp.float32:
        fp_c = "float"
        fp_tag = "f4"
    elif fp_dtype == cp.float64:
        fp_c = "double"
        fp_tag = "f8"
    else:
        raise ValueError(f"Unsupported fp_dtype for EPQ transpose: {fp_dtype!r}")

    indptr_dtype = cp.dtype(indptr_dtype)
    if indptr_dtype == cp.int32:
        indptr_c = "int"
        indptr_tag = "i4"
        atomic_add = "int pos = atomicAdd((int*)&write_ptr[i], 1);"
    elif indptr_dtype == cp.int64:
        indptr_c = "long long"
        indptr_tag = "i8"
        atomic_add = "unsigned long long pos = atomicAdd((unsigned long long*)&write_ptr[i], 1ULL);"
    else:
        raise ValueError(f"Unsupported indptr_dtype for EPQ transpose: {indptr_dtype!r}")

    name = f"epq_transpose_atomic_fill_{pq_tag}_{fp_tag}_{indptr_tag}"
    code = f"""
extern "C" __global__ void {name}(
    const {indptr_c}* epq_indptr,
    const int* epq_indices,
    const {pq_c}* epq_pq,
    const {fp_c}* epq_data,
    int ncsf,
    {indptr_c}* write_ptr,
    int* t_source,
    {pq_c}* t_pq,
    {fp_c}* t_data) {{
  int j = (int)blockIdx.x;
  if (j >= ncsf) return;
  long long start = (long long)epq_indptr[j];
  long long end = (long long)epq_indptr[j + 1];
  for (long long t = start + (long long)threadIdx.x; t < end; t += (long long)blockDim.x) {{
    int i = epq_indices[t];
    if ((unsigned)i >= (unsigned)ncsf) continue;
    {atomic_add}
    t_source[pos] = j;
    t_pq[pos] = epq_pq[t];
    t_data[pos] = epq_data[t];
  }}
}}
"""
    k = cp.RawKernel(code, name, options=("--std=c++11",))
    _EPQ_TRANSPOSE_ATOMIC_FILL_KERNELS[key] = k
    return k


def _epq_action_table_transpose_atomic_fill(
    cp,
    epq_indptr,
    epq_indices,
    epq_pq,
    epq_data,
    t_indptr,
    *,
    validate: bool = False,
    threads: int = 256,
):
    """Argsort-free EPQ transpose using count + atomic fill.

    Note: ordering within each destination row is non-deterministic.
    """

    ncsf = int(t_indptr.size) - 1
    nnz = int(epq_indices.size)
    write_ptr = t_indptr[:-1].copy()

    t_source = cp.empty((nnz,), dtype=cp.int32)
    t_pq = cp.empty((nnz,), dtype=epq_pq.dtype)
    t_data = cp.empty((nnz,), dtype=epq_data.dtype)

    k = _get_epq_transpose_atomic_fill_kernel(
        cp,
        pq_dtype=epq_pq.dtype,
        fp_dtype=epq_data.dtype,
        indptr_dtype=t_indptr.dtype,
    )
    k((ncsf,), (int(threads),), (epq_indptr, epq_indices, epq_pq, epq_data, ncsf, write_ptr, t_source, t_pq, t_data))

    if validate:
        ok = bool(cp.all(write_ptr == t_indptr[1:]))
        if not ok:
            raise RuntimeError("EPQ transpose atomic fill produced inconsistent write_ptr offsets")

    return t_source, t_pq, t_data


_EPQ_TILE_ROW_PACK_KERNELS = {}


def _get_epq_tile_row_pack_kernel(cp, *, pq_dtype, fp_dtype):
    key = (str(cp.dtype(pq_dtype)), str(cp.dtype(fp_dtype)))
    k = _EPQ_TILE_ROW_PACK_KERNELS.get(key)
    if k is not None:
        return k

    pq_dtype = cp.dtype(pq_dtype)
    fp_dtype = cp.dtype(fp_dtype)

    if pq_dtype == cp.uint8:
        pq_c = "unsigned char"
        pq_tag = "u1"
    elif pq_dtype == cp.uint16:
        pq_c = "unsigned short"
        pq_tag = "u2"
    elif pq_dtype == cp.int32:
        pq_c = "int"
        pq_tag = "i4"
    else:
        raise ValueError(f"Unsupported pq_dtype for EPQ tile pack: {pq_dtype!r}")

    if fp_dtype == cp.float32:
        fp_c = "float"
        fp_tag = "f4"
    elif fp_dtype == cp.float64:
        fp_c = "double"
        fp_tag = "f8"
    else:
        raise ValueError(f"Unsupported fp_dtype for EPQ tile pack: {fp_dtype!r}")

    name = f"epq_tile_pack_rows_{pq_tag}_{fp_tag}"
    code = f"""
extern "C" __global__ void {name}(
    const long long* row_src_start,   // [nrows]
    const long long* row_dst_start,   // [nrows]
    const long long* row_len,         // [nrows]
    const int* src_idx,               // [nnz_block]
    const int* src_pq,                // [nnz_block] int32
    const double* src_data,           // [nnz_block] float64
    int* dst_idx,                     // [nnz_total]
    {pq_c}* dst_pq,                   // [nnz_total]
    {fp_c}* dst_data,                 // [nnz_total]
    int nrows) {{
  int row = (int)blockIdx.x;
  if (row >= nrows) return;
  long long src0 = row_src_start[row];
  long long dst0 = row_dst_start[row];
  long long n = row_len[row];
  for (long long t = (long long)threadIdx.x; t < n; t += (long long)blockDim.x) {{
    long long s = src0 + t;
    long long d = dst0 + t;
    dst_idx[d] = src_idx[s];
    dst_pq[d] = ({pq_c})src_pq[s];
    dst_data[d] = ({fp_c})src_data[s];
  }}
}}
"""
    k = cp.RawKernel(code, name, options=("--std=c++11",))
    _EPQ_TILE_ROW_PACK_KERNELS[key] = k
    return k


def build_epq_action_table_transpose_device(
    drt: DRT,
    epq_table,
    *,
    dtype=None,
    indptr_dtype: str | None = "auto",
    use_cache: bool = True,
    argsort_free: bool | None = None,
    validate: bool = False,
):
    """Build (or fetch cached) destination-major transpose of an EPQ action table on device.

    Input `epq_table` is `(indptr, indices, pq_ids, data)` with rows by source `j`.
    Returns `(t_indptr, t_source, t_pq, t_data)` with rows by destination `i`.
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64 for EPQ transpose")
    indptr_mode = _normalize_epq_indptr_mode(indptr_dtype)

    if not isinstance(epq_table, tuple) or len(epq_table) != 4:
        raise TypeError("epq_table must be a 4-tuple (indptr, indices, pq_ids, data)")

    epq_indptr, epq_indices, epq_pq, epq_data = epq_table
    ncsf = int(drt.ncsf)
    epq_indptr = _as_epq_indptr_array(cp, epq_indptr, ncsf=ncsf, name="epq_indptr")
    epq_indices = cp.ascontiguousarray(cp.asarray(epq_indices, dtype=cp.int32).ravel())
    epq_pq = _as_epq_pq_array(cp, epq_pq, name="epq_pq")
    _validate_epq_pq_capacity(cp, epq_pq, norb=int(drt.norb), name="epq_pq")
    epq_data = cp.ascontiguousarray(cp.asarray(epq_data, dtype=fp_dtype).ravel())

    if epq_indptr.shape != (ncsf + 1,):
        raise ValueError("epq_indptr must have shape (ncsf+1,)")
    if epq_indices.shape != epq_pq.shape or epq_indices.shape != epq_data.shape:
        raise ValueError("epq_indices/epq_pq/epq_data must have identical shapes")

    nnz = int(epq_indices.size)
    if argsort_free is None:
        # Heuristic: avoid allocating an `nnz`-scale argsort permutation buffer when it would be large.
        perm_bytes_est = nnz * 8  # conservative: argsort indices are typically int64.
        argsort_free = bool(perm_bytes_est >= (256 << 20))

    cache_key = (
        int(epq_indptr.data.ptr),
        int(epq_indices.data.ptr),
        int(epq_pq.data.ptr),
        int(epq_data.data.ptr),
        str(indptr_mode),
        int(bool(argsort_free)),
    )
    if bool(use_cache):
        cached = getattr(drt, "_epq_action_table_transpose_device", None)
        cached_key = getattr(drt, "_epq_action_table_transpose_device_key", None)
        if cached is not None and cached_key == cache_key:
            return cached

    if nnz == 0:
        t_indptr_dt = _epq_indptr_cp_dtype_for_total_nnz(
            cp, mode=indptr_mode, total_nnz=0
        )
        t_indptr = cp.zeros((ncsf + 1,), dtype=t_indptr_dt)
        t_source = cp.empty((0,), dtype=cp.int32)
        t_pq = cp.empty((0,), dtype=epq_pq.dtype)
        t_data = cp.empty((0,), dtype=fp_dtype)
        out = (t_indptr, t_source, t_pq, t_data)
        if bool(use_cache):
            setattr(drt, "_epq_action_table_transpose_device_key", cache_key)
            setattr(drt, "_epq_action_table_transpose_device", out)
        return out

    row_counts = cp.bincount(epq_indices, minlength=ncsf).astype(cp.int64, copy=False)
    t_indptr = cp.empty((ncsf + 1,), dtype=cp.int64)
    t_indptr[0] = 0
    cp.cumsum(row_counts, out=t_indptr[1:])
    t_total_nnz = int(cp.asnumpy(t_indptr[-1])) if ncsf > 0 else 0
    t_indptr_dt = _epq_indptr_cp_dtype_for_total_nnz(
        cp, mode=indptr_mode, total_nnz=t_total_nnz
    )
    if cp.dtype(t_indptr_dt) == cp.dtype(cp.int32):
        t_indptr = cp.ascontiguousarray(t_indptr.astype(cp.int32, copy=False))

    if argsort_free:
        epq_indptr_fill = epq_indptr
        if cp.dtype(epq_indptr_fill.dtype) != cp.dtype(t_indptr.dtype):
            epq_indptr_fill = cp.ascontiguousarray(
                epq_indptr_fill.astype(t_indptr.dtype, copy=False)
            )
        t_source, t_pq, t_data = _epq_action_table_transpose_atomic_fill(
            cp,
            epq_indptr_fill,
            epq_indices,
            epq_pq,
            epq_data,
            t_indptr,
            validate=bool(validate),
        )
    else:
        source_rows = cp.searchsorted(
            epq_indptr[1:],
            cp.arange(nnz, dtype=cp.int64),
            side="right",
        ).astype(cp.int32, copy=False)
        if source_rows.size != nnz:
            raise RuntimeError("invalid epq_table structure: source row expansion mismatch")

        perm = cp.argsort(epq_indices, kind="stable")
        t_source = cp.ascontiguousarray(source_rows[perm])
        t_pq = cp.ascontiguousarray(epq_pq[perm])
        t_data = cp.ascontiguousarray(epq_data[perm])

    # Sort entries within each row by t_source (destination CSF k) so that the
    # apply kernel can use binary search to find the k-block range, eliminating
    # redundant scans (Opt 1: ~49x speedup on the apply kernel for large CI).
    if t_total_nnz > 0:
        # Expand row indices from CSR indptr: row_ids[t] = row of entry t (vectorized).
        t_indptr_i64 = t_indptr if cp.dtype(t_indptr.dtype) == cp.dtype(cp.int64) else t_indptr.astype(cp.int64)
        row_ids = cp.searchsorted(t_indptr_i64[1:], cp.arange(t_total_nnz, dtype=cp.int64), side="right")
        # Build composite key: (row_id << 32) | t_source — sorts first by row, then by source within row.
        composite_key = (row_ids.astype(cp.uint64) << np.uint64(32)) | t_source.astype(cp.uint64)
        perm_sort = cp.argsort(composite_key, kind="stable")
        t_source = cp.ascontiguousarray(t_source[perm_sort])
        t_pq = cp.ascontiguousarray(t_pq[perm_sort])
        t_data = cp.ascontiguousarray(t_data[perm_sort])

    out = (t_indptr, t_source, t_pq, t_data)
    if bool(use_cache):
        setattr(drt, "_epq_action_table_transpose_device_key", cache_key)
        setattr(drt, "_epq_action_table_transpose_device", out)
    return out


def _prepare_epq_gather_task_maps(task_csf, task_g, task_scale, ncsf: int, dtype=None):
    """Prepare destination-gather maps: row-by-csf and optional scale-by-csf."""
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64 for gather maps")

    task_csf = cp.ascontiguousarray(cp.asarray(task_csf, dtype=cp.int32).ravel())
    ncsf = int(ncsf)
    ntasks = int(task_csf.size)

    if task_scale is not None:
        task_scale = cp.ascontiguousarray(cp.asarray(task_scale, dtype=fp_dtype).ravel())
        if task_scale.shape != (ntasks,):
            raise ValueError("task_scale must have shape (ntasks,)")

    task_row_by_csf = cp.full((ncsf,), -1, dtype=cp.int32)
    if task_g.ndim == 1:
        # Shared g over tasks: duplicates are allowed; accumulate scale into per-source weights.
        if task_scale is None:
            weights = cp.ones((ntasks,), dtype=fp_dtype)
        else:
            weights = task_scale
        task_scale_by_csf = cp.zeros((ncsf,), dtype=fp_dtype)
        cp.add.at(task_scale_by_csf, task_csf, weights)
        active = task_scale_by_csf != 0.0
        task_row_by_csf[active] = 0
        return task_row_by_csf, task_scale_by_csf

    # Task-specific g rows: require unique source indices.
    uniq = cp.unique(task_csf)
    if int(uniq.size) != ntasks:
        raise ValueError("gather mode with task-specific task_g requires unique task_csf")
    task_row_by_csf[task_csf] = cp.arange(ntasks, dtype=cp.int32)
    if task_scale is None:
        return task_row_by_csf, None
    task_scale_by_csf = cp.zeros((ncsf,), dtype=fp_dtype)
    task_scale_by_csf[task_csf] = task_scale
    return task_row_by_csf, task_scale_by_csf


def validate_epq_apply_gather_inplace_device(
    drt: DRT,
    p: int,
    q: int,
    *,
    drt_dev=None,
    state_dev=None,
    cache: DRTStateCache | None = None,
    nvec: int = 8,
    seed: int = 0,
    alpha: float = 1.0,
    atol: float = 1e-11,
    rtol: float = 1e-11,
) -> None:
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    p = int(p)
    q = int(q)
    if p < 0 or p >= int(drt.norb) or q < 0 or q >= int(drt.norb):
        raise ValueError("orbital indices out of range")

    nvec = int(nvec)
    if nvec <= 0 or nvec > 32:
        raise ValueError("nvec must be in 1..32")

    if drt_dev is None:
        drt_dev = make_device_drt(drt)
    if cache is None:
        cache = get_state_cache(drt)
    if state_dev is None:
        state_dev = make_device_state_cache(drt, drt_dev, cache)

    rng = np.random.default_rng(int(seed))
    x_h = rng.standard_normal((int(drt.ncsf), nvec), dtype=np.float64)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    y_d = cp.empty((int(drt.ncsf), nvec), dtype=cp.float64)
    overflow = cp.empty((1,), dtype=cp.int32)
    epq_apply_gather_inplace_device(
        drt,
        drt_dev,
        state_dev,
        p,
        q,
        x_d,
        y=y_d,
        overflow=overflow,
        alpha=float(alpha),
        threads=256,
        add=False,
        sync=True,
        check_overflow=True,
    )
    y_gpu = np.asarray(cp.asnumpy(y_d), dtype=np.float64)

    y_cpu = np.zeros((int(drt.ncsf), nvec), dtype=np.float64)
    for j in range(int(drt.ncsf)):
        steps = cache.steps[j]
        nodes = cache.nodes[j]
        i_idx, coeff = epq_contribs_one(drt, j, p, q, steps=steps, nodes=nodes)
        if i_idx.size:
            y_cpu[i_idx] += (float(alpha) * coeff[:, None]) * x_h[j][None, :]
        elif p == q:
            occ = _step_to_occ(int(steps[p]))
            if occ:
                y_cpu[j] += float(alpha) * float(occ) * x_h[j]

    if not np.allclose(y_cpu, y_gpu, atol=float(atol), rtol=float(rtol)):
        diff = np.max(np.abs(y_cpu - y_gpu))
        raise AssertionError(f"y mismatch (max |Δ|={diff})")


def apply_g_flat_scatter_atomic(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf: np.ndarray,
    task_g: np.ndarray,
    *,
    task_scale: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    threads: int = 32,
    return_y: bool = True,
):
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_g = np.asarray(task_g, dtype=np.float64, order="C")
    if task_g.ndim not in (1, 2):
        raise ValueError("task_g must be 1D (norb*norb,) or 2D (ntasks,norb*norb)")
    if task_g.ndim == 1:
        if task_g.shape != (int(drt.norb) * int(drt.norb),):
            raise ValueError("task_g (1D) must have shape (norb*norb,)")
    else:
        if task_g.shape != (int(task_csf.size), int(drt.norb) * int(drt.norb)):
            raise ValueError("task_g (2D) must have shape (ntasks,norb*norb)")

    if task_scale is not None:
        task_scale = np.asarray(task_scale, dtype=np.float64).ravel()
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    if y0 is not None:
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        if y0.shape != (int(drt.ncsf),):
            raise ValueError("y0 must have shape (ncsf,)")

    out = _ext.apply_g_flat_scatter_atomic(
        drt_dev,
        state_dev,
        task_csf,
        task_g,
        task_scale if task_scale is not None else None,
        y0 if y0 is not None else None,
        int(threads),
        bool(return_y),
    )
    if not return_y:
        return None
    return np.asarray(out, dtype=np.float64)


def apply_g_flat_scatter_atomic_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf,
    task_g,
    *,
    epq_table=None,
    epq_table_t=None,
    apply_mode: str = "scatter",
    dtype=None,
    task_scale=None,
    y=None,
    overflow=None,
    threads: int = 256,
    zero_y: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    warp_coop: bool = False,
    use_kahan: bool = False,
):
    """Apply G(pq) to CSF vector using Kernel1A (atomic scatter) or EPQ-table accelerated paths on GPU.

    `dtype` controls the floating-point type (`float32` or `float64`) for both EPQ and non-EPQ apply paths.
    
    If `warp_coop=True`, uses the warp-cooperative segment walk kernel (10.16.3/10.18 Option A) which
    eliminates per-thread DFS stack spills by distributing the segment walk across warp lanes.
    This is beneficial for large norb where local memory traffic becomes a bottleneck.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    task_csf = cp.asarray(task_csf, dtype=cp.int32).ravel()
    task_g = cp.asarray(task_g, dtype=fp_dtype)
    task_g = cp.ascontiguousarray(task_g)
    if task_g.ndim not in (1, 2):
        raise ValueError("task_g must be 1D (norb*norb,) or 2D (ntasks,norb*norb)")

    if task_scale is not None:
        task_scale = cp.asarray(task_scale, dtype=fp_dtype).ravel()
        task_scale = cp.ascontiguousarray(task_scale)
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.asarray(y, dtype=fp_dtype).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    if epq_table is None:
        # 10.16.3/10.18 Option A: Warp-cooperative segment walk kernel
        if bool(warp_coop):
            if not hasattr(_ext, "apply_g_flat_scatter_atomic_warp_coop_inplace_device"):
                raise RuntimeError(
                    "warp_coop=True requires extension with warp-cooperative kernel; rebuild with CUDA support"
                )
            # Warp-cooperative kernel requires threads to be a multiple of 32
            threads_warp = int(threads)
            if threads_warp < 32 or (threads_warp % 32) != 0:
                threads_warp = 256  # default to 256 threads for warp-coop
            _ext.apply_g_flat_scatter_atomic_warp_coop_inplace_device(
                drt_dev,
                state_dev,
                task_csf,
                task_g,
                task_scale if task_scale is not None else None,
                y,
                overflow,
                int(threads_warp),
                bool(zero_y),
                int(stream_ptr),
                bool(sync),
                bool(check_overflow),
            )
        else:
            _ext.apply_g_flat_scatter_atomic_inplace_device(
                drt_dev,
                state_dev,
                task_csf,
                task_g,
                task_scale if task_scale is not None else None,
                y,
                overflow,
                int(threads),
                bool(zero_y),
                int(stream_ptr),
                bool(sync),
                bool(check_overflow),
            )
        return y, overflow
    else:
        if not isinstance(epq_table, tuple) or len(epq_table) != 4:
            raise TypeError("epq_table must be a 4-tuple (indptr, indices, pq_ids, data)")
        epq_indptr, epq_indices, epq_pq, epq_data = epq_table
        epq_indptr = _as_epq_indptr_array(cp, epq_indptr, ncsf=int(drt.ncsf), name="epq_indptr")
        epq_indices = cp.asarray(epq_indices, dtype=cp.int32).ravel()
        epq_indices = cp.ascontiguousarray(epq_indices)
        epq_pq = _as_epq_pq_array(cp, epq_pq, name="epq_pq")
        _validate_epq_pq_capacity(cp, epq_pq, norb=int(drt.norb), name="epq_pq")
        epq_data = cp.asarray(epq_data).ravel()
        # Keep native dtype (f32 or f64). Only force to fp_dtype if neither.
        if epq_data.dtype not in (cp.float32, cp.float64):
            epq_data = cp.asarray(epq_data, dtype=fp_dtype).ravel()
        # If output is f32 but coefficients are f64, downcast to f32.
        if fp_dtype == cp.float32 and epq_data.dtype == cp.float64:
            epq_data = cp.asarray(epq_data, dtype=cp.float32).ravel()
        epq_data = cp.ascontiguousarray(epq_data)
        if epq_indices.shape != epq_pq.shape or epq_indices.shape != epq_data.shape:
            raise ValueError("epq_table arrays must have the same shape for (indices, pq_ids, data)")
        if epq_indptr.shape != (int(drt.ncsf) + 1,):
            raise ValueError("epq_table indptr must have shape (ncsf+1,)")
        mode = str(apply_mode).lower()
        if mode not in ("auto", "scatter", "gather"):
            raise ValueError("apply_mode must be one of: 'auto', 'scatter', 'gather'")
        use_gather = mode == "gather"
        if mode == "auto":
            use_gather = False
            if bool(has_epq_table_gather_apply_device() and task_g.ndim == 1):
                import os

                auto_env = str(os.getenv("ASUKA_CUGUGA_EPQ_APPLY_AUTO", "heuristic")).strip().lower()
                if auto_env in ("1", "true", "yes", "on", "gather"):
                    use_gather = True
                elif auto_env in ("0", "false", "no", "off", "scatter"):
                    use_gather = False
                elif auto_env in ("", "heuristic", "auto"):
                    # Conservative default tuned for RTX 4090: gather stayed slower up to
                    # ncsf~=2.76M for practical j-tiles. Keep gather opt-in unless the CI
                    # space is substantially larger and task granularity is non-trivial.
                    ncsf = int(drt.ncsf)
                    ntasks = int(task_csf.size)
                    use_gather = bool(ncsf >= 4_000_000 and ntasks >= 8_192)
                else:
                    raise ValueError(
                        "ASUKA_CUGUGA_EPQ_APPLY_AUTO must be one of: auto|heuristic|scatter|gather|0|1"
                    )

        if use_gather and not has_epq_table_gather_apply_device():
            raise RuntimeError("Requested gather mode but extension lacks gather EPQ apply entrypoint")

        if use_gather:
            if epq_table_t is None:
                transpose_dtype = fp_dtype
                if fp_dtype == cp.float64 and cp.dtype(epq_data.dtype) == cp.float32:
                    transpose_dtype = cp.float32
                epq_table_t = build_epq_action_table_transpose_device(
                    drt,
                    epq_table,
                    dtype=transpose_dtype,
                    use_cache=True,
                )
            if not isinstance(epq_table_t, tuple) or len(epq_table_t) != 4:
                raise TypeError("epq_table_t must be a 4-tuple (t_indptr, t_source, t_pq, t_data)")

            epq_t_indptr, epq_t_source, epq_t_pq, epq_t_data = epq_table_t
            epq_t_indptr = _as_epq_indptr_array(cp, epq_t_indptr, ncsf=int(drt.ncsf), name="epq_t_indptr")
            epq_t_source = cp.ascontiguousarray(cp.asarray(epq_t_source, dtype=cp.int32).ravel())
            epq_t_pq = _as_epq_pq_array(cp, epq_t_pq, name="epq_t_pq")
            _validate_epq_pq_capacity(cp, epq_t_pq, norb=int(drt.norb), name="epq_t_pq")
            epq_t_data = cp.asarray(epq_t_data).ravel()
            epq_t_dt = cp.dtype(epq_t_data.dtype)
            if epq_t_dt not in (cp.float32, cp.float64):
                epq_t_data = cp.asarray(epq_t_data, dtype=fp_dtype).ravel()
                epq_t_dt = cp.dtype(epq_t_data.dtype)
            if fp_dtype == cp.float32 and epq_t_dt != cp.float32:
                epq_t_data = cp.asarray(epq_t_data, dtype=cp.float32).ravel()
            epq_t_data = cp.ascontiguousarray(epq_t_data)
            if epq_t_indptr.shape != (int(drt.ncsf) + 1,):
                raise ValueError("epq_table_t indptr must have shape (ncsf+1,)")
            if epq_t_source.shape != epq_t_pq.shape or epq_t_source.shape != epq_t_data.shape:
                raise ValueError("epq_table_t arrays must have the same shape for (source, pq_ids, data)")

            task_row_by_csf, task_scale_by_csf = _prepare_epq_gather_task_maps(
                task_csf, task_g, task_scale, int(drt.ncsf), dtype=fp_dtype
            )

            _ext.apply_g_flat_gather_epq_table_inplace_device(
                drt_dev,
                state_dev,
                epq_t_indptr,
                epq_t_source,
                epq_t_pq,
                epq_t_data,
                task_row_by_csf,
                task_scale_by_csf if task_scale_by_csf is not None else None,
                task_g,
                y,
                overflow,
                int(threads),
                bool(zero_y),
                int(stream_ptr),
                bool(sync),
                bool(check_overflow),
                bool(use_kahan),
            )
            return y, overflow

        _ext.apply_g_flat_scatter_atomic_epq_table_inplace_device(
            drt_dev,
            state_dev,
            epq_indptr,
            epq_indices,
            epq_pq,
            epq_data,
            task_csf,
            task_g,
            task_scale if task_scale is not None else None,
            y,
            overflow,
            int(threads),
            bool(zero_y),
            int(stream_ptr),
            bool(sync),
            bool(check_overflow),
        )

    return y, overflow


def has_cipsi_frontier_hash_device() -> bool:
    """Return True if the CUDA extension exposes the CIPSI frontier-hash entrypoints."""

    return _ext is not None and hasattr(_ext, "cipsi_frontier_hash_clear_inplace_device")


def cipsi_frontier_hash_clear_inplace_device(
    keys,
    vals_root_major,
    *,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    if _ext is None or not hasattr(_ext, "cipsi_frontier_hash_clear_inplace_device"):
        raise RuntimeError("CUDA extension is missing CIPSI frontier-hash kernels; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    keys = cp.asarray(keys, dtype=cp.int32).ravel()
    keys = cp.ascontiguousarray(keys)
    vals_root_major = cp.asarray(vals_root_major, dtype=cp.float64)
    vals_root_major = cp.ascontiguousarray(vals_root_major)
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, cap)")
    if vals_root_major.shape[1] != keys.shape[0]:
        raise ValueError("vals_root_major must have shape (nroots, cap) with cap matching keys")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cipsi_frontier_hash_clear_inplace_device(
        keys,
        vals_root_major,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return keys, vals_root_major


def cipsi_frontier_hash_extract_inplace_device(
    keys,
    vals_root_major,
    *,
    out_idx=None,
    out_vals_root_major=None,
    out_nnz=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    if _ext is None or not hasattr(_ext, "cipsi_frontier_hash_extract_inplace_device"):
        raise RuntimeError("CUDA extension is missing CIPSI frontier-hash kernels; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    keys = cp.asarray(keys, dtype=cp.int32).ravel()
    keys = cp.ascontiguousarray(keys)
    vals_root_major = cp.asarray(vals_root_major, dtype=cp.float64)
    vals_root_major = cp.ascontiguousarray(vals_root_major)
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, cap)")
    nroots = int(vals_root_major.shape[0])
    cap = int(keys.size)
    if int(vals_root_major.shape[1]) != cap:
        raise ValueError("vals_root_major must have shape (nroots, cap) with cap matching keys")

    if out_idx is None:
        out_idx = cp.empty((cap,), dtype=cp.int32)
    else:
        out_idx = cp.asarray(out_idx, dtype=cp.int32).ravel()
        out_idx = cp.ascontiguousarray(out_idx)
        if out_idx.shape != (cap,):
            raise ValueError("out_idx must have shape (cap,)")

    if out_vals_root_major is None:
        out_vals_root_major = cp.empty((nroots, cap), dtype=cp.float64)
    else:
        out_vals_root_major = cp.asarray(out_vals_root_major, dtype=cp.float64)
        out_vals_root_major = cp.ascontiguousarray(out_vals_root_major)
        if out_vals_root_major.shape != (nroots, cap):
            raise ValueError("out_vals_root_major must have shape (nroots, cap)")

    if out_nnz is None:
        out_nnz = cp.empty((1,), dtype=cp.int32)
    else:
        out_nnz = cp.asarray(out_nnz, dtype=cp.int32).ravel()
        out_nnz = cp.ascontiguousarray(out_nnz)
        if out_nnz.shape != (1,):
            raise ValueError("out_nnz must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cipsi_frontier_hash_extract_inplace_device(
        keys,
        vals_root_major,
        out_idx,
        out_vals_root_major,
        out_nnz,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_idx, out_vals_root_major, out_nnz


def cipsi_score_and_select_topk_inplace_device(
    idx,
    vals_root_major,
    *,
    nnz: int,
    e_var,
    hdiag,
    selected_mask,
    denom_floor: float,
    out_new_idx,
    out_new_n,
    out_pt2,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    if _ext is None or not hasattr(_ext, "cipsi_score_and_select_topk_inplace_device"):
        raise RuntimeError("CUDA extension is missing CIPSI frontier-hash kernels; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    idx = cp.asarray(idx, dtype=cp.int32).ravel()
    idx = cp.ascontiguousarray(idx)
    vals_root_major = cp.asarray(vals_root_major, dtype=cp.float64)
    vals_root_major = cp.ascontiguousarray(vals_root_major)
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, stride)")
    nroots = int(vals_root_major.shape[0])

    e_var = cp.asarray(e_var, dtype=cp.float64).ravel()
    e_var = cp.ascontiguousarray(e_var)
    if e_var.shape != (nroots,):
        raise ValueError("e_var must have shape (nroots,)")
    hdiag = cp.asarray(hdiag, dtype=cp.float64).ravel()
    hdiag = cp.ascontiguousarray(hdiag)
    ncsf = int(hdiag.size)

    if selected_mask is not None:
        selected_mask = cp.asarray(selected_mask, dtype=cp.uint8).ravel()
        selected_mask = cp.ascontiguousarray(selected_mask)
        if selected_mask.shape != (ncsf,):
            raise ValueError("selected_mask must have shape (ncsf,)")

    out_new_idx = cp.asarray(out_new_idx, dtype=cp.int32).ravel()
    out_new_idx = cp.ascontiguousarray(out_new_idx)
    out_new_n = cp.asarray(out_new_n, dtype=cp.int32).ravel()
    out_new_n = cp.ascontiguousarray(out_new_n)
    if out_new_n.shape != (1,):
        raise ValueError("out_new_n must have shape (1,)")
    out_pt2 = cp.asarray(out_pt2, dtype=cp.float64).ravel()
    out_pt2 = cp.ascontiguousarray(out_pt2)
    if out_pt2.shape != (nroots,):
        raise ValueError("out_pt2 must have shape (nroots,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cipsi_score_and_select_topk_inplace_device(
        idx,
        vals_root_major,
        int(nnz),
        e_var,
        hdiag,
        selected_mask if selected_mask is not None else None,
        float(denom_floor),
        out_new_idx,
        out_new_n,
        out_pt2,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_new_idx, out_new_n, out_pt2


def cipsi_score_and_select_topk_from_hash_slots_inplace_device(
    keys,
    vals_root_major,
    *,
    e_var,
    hdiag,
    selected_mask,
    denom_floor: float,
    out_new_idx,
    out_new_n,
    out_pt2,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    if _ext is None or not hasattr(_ext, "cipsi_score_and_select_topk_from_hash_slots_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing hash-slot score/select kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    keys = cp.asarray(keys, dtype=cp.int32).ravel()
    keys = cp.ascontiguousarray(keys)
    vals_root_major = cp.asarray(vals_root_major, dtype=cp.float64)
    vals_root_major = cp.ascontiguousarray(vals_root_major)
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, cap)")
    nroots = int(vals_root_major.shape[0])
    cap = int(keys.size)
    if int(vals_root_major.shape[1]) != cap:
        raise ValueError("vals_root_major must have shape (nroots, cap) with cap matching keys")

    e_var = cp.asarray(e_var, dtype=cp.float64).ravel()
    e_var = cp.ascontiguousarray(e_var)
    if e_var.shape != (nroots,):
        raise ValueError("e_var must have shape (nroots,)")
    hdiag = cp.asarray(hdiag, dtype=cp.float64).ravel()
    hdiag = cp.ascontiguousarray(hdiag)
    ncsf = int(hdiag.size)

    if selected_mask is not None:
        selected_mask = cp.asarray(selected_mask, dtype=cp.uint8).ravel()
        selected_mask = cp.ascontiguousarray(selected_mask)
        if selected_mask.shape != (ncsf,):
            raise ValueError("selected_mask must have shape (ncsf,)")

    out_new_idx = cp.asarray(out_new_idx, dtype=cp.int32).ravel()
    out_new_idx = cp.ascontiguousarray(out_new_idx)
    out_new_n = cp.asarray(out_new_n, dtype=cp.int32).ravel()
    out_new_n = cp.ascontiguousarray(out_new_n)
    if out_new_n.shape != (1,):
        raise ValueError("out_new_n must have shape (1,)")
    out_pt2 = cp.asarray(out_pt2, dtype=cp.float64).ravel()
    out_pt2 = cp.ascontiguousarray(out_pt2)
    if out_pt2.shape != (nroots,):
        raise ValueError("out_pt2 must have shape (nroots,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cipsi_score_and_select_topk_from_hash_slots_inplace_device(
        keys,
        vals_root_major,
        e_var,
        hdiag,
        selected_mask if selected_mask is not None else None,
        float(denom_floor),
        out_new_idx,
        out_new_n,
        out_pt2,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_new_idx, out_new_n, out_pt2


def cipsi_score_and_select_topk_from_hash_slots_v2_inplace_device(
    keys,
    vals_root_major,
    *,
    e_var,
    hdiag,
    selected_mask,
    denom_floor: float,
    out_new_idx,
    out_new_n,
    out_pt2,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    if _ext is None or not hasattr(_ext, "cipsi_score_and_select_topk_from_hash_slots_v2_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing hash-slot score/select v2 kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array path") from e

    keys = cp.asarray(keys, dtype=cp.int32).ravel()
    keys = cp.ascontiguousarray(keys)
    vals_root_major = cp.asarray(vals_root_major, dtype=cp.float64)
    vals_root_major = cp.ascontiguousarray(vals_root_major)
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, cap)")
    nroots = int(vals_root_major.shape[0])
    cap = int(keys.size)
    if int(vals_root_major.shape[1]) != cap:
        raise ValueError("vals_root_major must have shape (nroots, cap) with cap matching keys")

    e_var = cp.asarray(e_var, dtype=cp.float64).ravel()
    e_var = cp.ascontiguousarray(e_var)
    if e_var.shape != (nroots,):
        raise ValueError("e_var must have shape (nroots,)")
    hdiag = cp.asarray(hdiag, dtype=cp.float64).ravel()
    hdiag = cp.ascontiguousarray(hdiag)
    ncsf = int(hdiag.size)

    if selected_mask is not None:
        selected_mask = cp.asarray(selected_mask, dtype=cp.uint8).ravel()
        selected_mask = cp.ascontiguousarray(selected_mask)
        if selected_mask.shape != (ncsf,):
            raise ValueError("selected_mask must have shape (ncsf,)")

    out_new_idx = cp.asarray(out_new_idx, dtype=cp.int32).ravel()
    out_new_idx = cp.ascontiguousarray(out_new_idx)
    out_new_n = cp.asarray(out_new_n, dtype=cp.int32).ravel()
    out_new_n = cp.ascontiguousarray(out_new_n)
    if out_new_n.shape != (1,):
        raise ValueError("out_new_n must have shape (1,)")
    out_pt2 = cp.asarray(out_pt2, dtype=cp.float64).ravel()
    out_pt2 = cp.ascontiguousarray(out_pt2)
    if out_pt2.shape != (nroots,):
        raise ValueError("out_pt2 must have shape (nroots,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cipsi_score_and_select_topk_from_hash_slots_v2_inplace_device(
        keys,
        vals_root_major,
        e_var,
        hdiag,
        selected_mask if selected_mask is not None else None,
        float(denom_floor),
        out_new_idx,
        out_new_n,
        out_pt2,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_new_idx, out_new_n, out_pt2


def has_cas36_hb_screen_and_apply_u64_device() -> bool:
    """Return True if the CUDA extension exposes CAS(36,36)-style u64 HB-SCI builders."""
    return _ext is not None and hasattr(_ext, "cas36_hb_screen_and_apply_u64_inplace_device")


def has_cas36_hb_screen_and_apply_many_roots_u64_device() -> bool:
    """Return True if the CUDA extension exposes the multiroot CAS(36,36) u64 HB-SCI builder."""
    return _ext is not None and hasattr(_ext, "cas36_hb_screen_and_apply_many_roots_u64_inplace_device")


def has_cas36_hb_emit_tuples_u64_device() -> bool:
    """Return True if the CUDA extension exposes CAS(36,36)-style u64 tuple emitters."""
    return _ext is not None and hasattr(_ext, "cas36_hb_emit_tuples_u64_inplace_device")


def has_cas36_exact_selected_emit_tuples_u64_device() -> bool:
    """Return True if the CUDA extension exposes the dedicated exact-selected u64 tuple emitter."""
    return _ext is not None and hasattr(_ext, "cas36_exact_selected_emit_tuples_u64_inplace_device")


def has_cas36_exact_selected_emit_tuples_dense_u64_device() -> bool:
    """Return True if the CUDA extension exposes the dense exact-selected u64 tuple emitter."""
    return _ext is not None and hasattr(_ext, "cas36_exact_selected_emit_tuples_dense_u64_inplace_device")


def has_cas36_exact_selected_build_rowhash_dense_u64_device() -> bool:
    """Return True if the CUDA extension exposes the dense exact-selected rowhash emitter."""
    return _ext is not None and hasattr(_ext, "cas36_exact_selected_build_rowhash_dense_u64_inplace_device")


def has_cas36_exact_external_emit_tuples_dense_u64_device() -> bool:
    """Return True if the CUDA extension exposes the dense exact-external u64 tuple emitter."""
    return _ext is not None and hasattr(_ext, "cas36_exact_external_emit_tuples_dense_u64_inplace_device")


def has_cas36_exact_external_apply_dense_many_roots_u64_device() -> bool:
    """Return True if the CUDA extension exposes the dense exact-external multiroot apply kernel."""
    return _ext is not None and hasattr(_ext, "cas36_exact_external_apply_dense_many_roots_u64_inplace_device")


def has_cas36_diag_guess_candidates_u64_dense_device() -> bool:
    """Return True if compact candidate diagonal-guess u64 kernel is available."""
    return _ext is not None and hasattr(_ext, "cas36_diag_guess_candidates_u64_dense_inplace_device")


def has_cas36_cipsi_score_pt2_compact_u64_device() -> bool:
    """Return True if compact candidate score/PT2 u64 kernel is available."""
    return _ext is not None and hasattr(_ext, "cas36_cipsi_score_pt2_compact_u64_inplace_device")


def cas36_hb_screen_and_apply_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_root,
    *,
    nsel: int,
    root: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    eps: float,
    hash_keys_u64,
    hash_vals,
    label_lo: int = 0,
    label_hi: int | None = None,
    selected_idx_sorted_u64=None,
    target_mode: str | int = "external_only",
    overflow=None,
    sym_pq_allowed=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the u64 heat-bath screen+apply kernel (large-space CUDA SCI path)."""
    if _ext is None or not hasattr(_ext, "cas36_hb_screen_and_apply_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing CAS36 HB-SCI u64 kernels; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_root = cp.ascontiguousarray(cp.asarray(c_root, dtype=cp.float64).ravel())
    h1_pq = cp.ascontiguousarray(cp.asarray(h1_pq, dtype=cp.int32))
    h1_abs = cp.ascontiguousarray(cp.asarray(h1_abs, dtype=cp.float64).ravel())
    h1_signed = cp.ascontiguousarray(cp.asarray(h1_signed, dtype=cp.float64).ravel())
    pq_ptr = cp.ascontiguousarray(cp.asarray(pq_ptr, dtype=cp.int64).ravel())
    rs_idx = cp.ascontiguousarray(cp.asarray(rs_idx, dtype=cp.int32).ravel())
    v_abs = cp.ascontiguousarray(cp.asarray(v_abs, dtype=cp.float64).ravel())
    v_signed = cp.ascontiguousarray(cp.asarray(v_signed, dtype=cp.float64).ravel())
    pq_max_v = cp.ascontiguousarray(cp.asarray(pq_max_v, dtype=cp.float64).ravel())
    hash_keys_u64 = cp.ascontiguousarray(cp.asarray(hash_keys_u64, dtype=cp.uint64).ravel())
    hash_vals = cp.ascontiguousarray(cp.asarray(hash_vals, dtype=cp.float64))
    if hash_vals.ndim != 2:
        raise ValueError("hash_vals must have shape (nroots, cap)")
    if int(hash_vals.shape[1]) != int(hash_keys_u64.size):
        raise ValueError("hash_vals must have shape (nroots, cap) with cap matching hash_keys_u64")
    label_lo = int(label_lo)
    label_hi = int(drt.ncsf) if label_hi is None else int(label_hi)
    if label_lo < 0 or label_hi < label_lo or label_hi > int(drt.ncsf):
        raise ValueError("label window must satisfy 0 <= label_lo <= label_hi <= drt.ncsf")
    if isinstance(target_mode, str):
        target_mode_s = str(target_mode).strip().lower()
        if target_mode_s in ("external", "external_only", "exclude_selected"):
            target_mode_i = 0
        elif target_mode_s in ("selected", "selected_only", "keep_selected"):
            target_mode_i = 1
        else:
            raise ValueError("target_mode must be 'external_only' or 'selected_only'")
    else:
        target_mode_i = int(target_mode)
        if target_mode_i not in (0, 1):
            raise ValueError("target_mode must be 0 or 1")

    if selected_idx_sorted_u64 is not None:
        selected_idx_sorted_u64 = cp.ascontiguousarray(cp.asarray(selected_idx_sorted_u64, dtype=cp.uint64).ravel())
    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_hb_screen_and_apply_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_root,
        int(nsel),
        int(root),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        float(eps),
        hash_keys_u64,
        hash_vals,
        int(label_lo),
        int(label_hi),
        selected_idx_sorted_u64 if selected_idx_sorted_u64 is not None else None,
        int(target_mode_i),
        overflow,
        sym_pq_allowed if sym_pq_allowed is not None else None,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return hash_keys_u64, hash_vals, overflow


def cas36_hb_screen_and_apply_many_roots_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_sel,
    *,
    nsel: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    eps: float,
    hash_keys_u64,
    hash_vals,
    label_lo: int = 0,
    label_hi: int | None = None,
    selected_idx_sorted_u64=None,
    target_mode: str | int = "external_only",
    overflow=None,
    sym_pq_allowed=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the multiroot u64 heat-bath screen+apply kernel."""
    if _ext is None or not hasattr(_ext, "cas36_hb_screen_and_apply_many_roots_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing CAS36 HB-SCI multiroot u64 kernels; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_sel = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
    if c_sel.ndim != 2:
        raise ValueError("c_sel must have shape (nsel, nroots)")
    if int(c_sel.shape[0]) < int(nsel):
        raise ValueError("c_sel must have leading dimension >= nsel")
    h1_pq = cp.ascontiguousarray(cp.asarray(h1_pq, dtype=cp.int32))
    h1_abs = cp.ascontiguousarray(cp.asarray(h1_abs, dtype=cp.float64).ravel())
    h1_signed = cp.ascontiguousarray(cp.asarray(h1_signed, dtype=cp.float64).ravel())
    pq_ptr = cp.ascontiguousarray(cp.asarray(pq_ptr, dtype=cp.int64).ravel())
    rs_idx = cp.ascontiguousarray(cp.asarray(rs_idx, dtype=cp.int32).ravel())
    v_abs = cp.ascontiguousarray(cp.asarray(v_abs, dtype=cp.float64).ravel())
    v_signed = cp.ascontiguousarray(cp.asarray(v_signed, dtype=cp.float64).ravel())
    pq_max_v = cp.ascontiguousarray(cp.asarray(pq_max_v, dtype=cp.float64).ravel())
    hash_keys_u64 = cp.ascontiguousarray(cp.asarray(hash_keys_u64, dtype=cp.uint64).ravel())
    hash_vals = cp.ascontiguousarray(cp.asarray(hash_vals, dtype=cp.float64))
    if hash_vals.ndim != 2:
        raise ValueError("hash_vals must have shape (nroots, cap)")
    if int(hash_vals.shape[0]) != int(c_sel.shape[1]) or int(hash_vals.shape[1]) != int(hash_keys_u64.size):
        raise ValueError("hash_vals must have shape (nroots, cap) matching c_sel and hash_keys_u64")
    label_lo = int(label_lo)
    label_hi = int(drt.ncsf) if label_hi is None else int(label_hi)
    if label_lo < 0 or label_hi < label_lo or label_hi > int(drt.ncsf):
        raise ValueError("label window must satisfy 0 <= label_lo <= label_hi <= drt.ncsf")
    if isinstance(target_mode, str):
        target_mode_s = str(target_mode).strip().lower()
        if target_mode_s in ("external", "external_only", "exclude_selected"):
            target_mode_i = 0
        elif target_mode_s in ("selected", "selected_only", "keep_selected"):
            target_mode_i = 1
        else:
            raise ValueError("target_mode must be 'external_only' or 'selected_only'")
    else:
        target_mode_i = int(target_mode)
        if target_mode_i not in (0, 1):
            raise ValueError("target_mode must be 0 or 1")

    if selected_idx_sorted_u64 is not None:
        selected_idx_sorted_u64 = cp.ascontiguousarray(cp.asarray(selected_idx_sorted_u64, dtype=cp.uint64).ravel())
    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_hb_screen_and_apply_many_roots_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_sel,
        int(nsel),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        float(eps),
        hash_keys_u64,
        hash_vals,
        int(label_lo),
        int(label_hi),
        selected_idx_sorted_u64 if selected_idx_sorted_u64 is not None else None,
        int(target_mode_i),
        overflow,
        sym_pq_allowed if sym_pq_allowed is not None else None,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return hash_keys_u64, hash_vals, overflow


def cas36_hb_emit_tuples_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_bound,
    *,
    nsel: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    eps: float,
    out_keys_u64,
    out_src,
    out_hij,
    cap: int,
    label_lo: int = 0,
    label_hi: int | None = None,
    selected_idx_sorted_u64=None,
    target_mode: str | int = "external_only",
    out_n=None,
    overflow=None,
    sym_pq_allowed=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the u64 heat-bath SCI tuple-emission kernel (large-space CUDA SCI path)."""
    if _ext is None or not hasattr(_ext, "cas36_hb_emit_tuples_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing CAS36 HB-SCI u64 tuple emitters; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_bound = cp.ascontiguousarray(cp.asarray(c_bound, dtype=cp.float64).ravel())
    h1_pq = cp.ascontiguousarray(cp.asarray(h1_pq, dtype=cp.int32))
    h1_abs = cp.ascontiguousarray(cp.asarray(h1_abs, dtype=cp.float64).ravel())
    h1_signed = cp.ascontiguousarray(cp.asarray(h1_signed, dtype=cp.float64).ravel())
    pq_ptr = cp.ascontiguousarray(cp.asarray(pq_ptr, dtype=cp.int64).ravel())
    rs_idx = cp.ascontiguousarray(cp.asarray(rs_idx, dtype=cp.int32).ravel())
    v_abs = cp.ascontiguousarray(cp.asarray(v_abs, dtype=cp.float64).ravel())
    v_signed = cp.ascontiguousarray(cp.asarray(v_signed, dtype=cp.float64).ravel())
    pq_max_v = cp.ascontiguousarray(cp.asarray(pq_max_v, dtype=cp.float64).ravel())
    out_keys_u64 = cp.ascontiguousarray(cp.asarray(out_keys_u64, dtype=cp.uint64).ravel())
    out_src = cp.ascontiguousarray(cp.asarray(out_src, dtype=cp.int32).ravel())
    out_hij = cp.ascontiguousarray(cp.asarray(out_hij, dtype=cp.float64).ravel())
    if int(out_keys_u64.size) < int(cap) or int(out_src.size) < int(cap) or int(out_hij.size) < int(cap):
        raise ValueError("out_keys_u64, out_src, and out_hij must have length >= cap")
    label_lo = int(label_lo)
    label_hi = int(drt.ncsf) if label_hi is None else int(label_hi)
    if label_lo < 0 or label_hi < label_lo or label_hi > int(drt.ncsf):
        raise ValueError("label window must satisfy 0 <= label_lo <= label_hi <= drt.ncsf")
    if isinstance(target_mode, str):
        target_mode_s = str(target_mode).strip().lower()
        if target_mode_s in ("external", "external_only", "exclude_selected"):
            target_mode_i = 0
        elif target_mode_s in ("selected", "selected_only", "keep_selected"):
            target_mode_i = 1
        else:
            raise ValueError("target_mode must be 'external_only' or 'selected_only'")
    else:
        target_mode_i = int(target_mode)
        if target_mode_i not in (0, 1):
            raise ValueError("target_mode must be 0 or 1")

    if selected_idx_sorted_u64 is not None:
        selected_idx_sorted_u64 = cp.ascontiguousarray(cp.asarray(selected_idx_sorted_u64, dtype=cp.uint64).ravel())
    if out_n is None:
        out_n = cp.zeros((1,), dtype=cp.int32)
    else:
        out_n = cp.ascontiguousarray(cp.asarray(out_n, dtype=cp.int32).ravel())
        if out_n.shape != (1,):
            raise ValueError("out_n must have shape (1,)")
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_hb_emit_tuples_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_bound,
        int(nsel),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        float(eps),
        out_keys_u64,
        out_src,
        out_hij,
        int(cap),
        int(label_lo),
        int(label_hi),
        selected_idx_sorted_u64 if selected_idx_sorted_u64 is not None else None,
        int(target_mode_i),
        out_n,
        overflow,
        sym_pq_allowed if sym_pq_allowed is not None else None,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_keys_u64, out_src, out_hij, out_n, overflow


def cas36_exact_selected_emit_tuples_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_bound,
    *,
    nsel: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    out_keys_u64,
    out_src,
    out_hij,
    cap: int,
    selected_idx_sorted_u64,
    out_n=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the dedicated exact-selected u64 tuple-emission surface."""
    if _ext is None or not hasattr(_ext, "cas36_exact_selected_emit_tuples_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the exact-selected CAS36 u64 tuple emitter; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_bound = cp.ascontiguousarray(cp.asarray(c_bound, dtype=cp.float64).ravel())
    h1_pq = cp.ascontiguousarray(cp.asarray(h1_pq, dtype=cp.int32))
    h1_abs = cp.ascontiguousarray(cp.asarray(h1_abs, dtype=cp.float64).ravel())
    h1_signed = cp.ascontiguousarray(cp.asarray(h1_signed, dtype=cp.float64).ravel())
    pq_ptr = cp.ascontiguousarray(cp.asarray(pq_ptr, dtype=cp.int64).ravel())
    rs_idx = cp.ascontiguousarray(cp.asarray(rs_idx, dtype=cp.int32).ravel())
    v_abs = cp.ascontiguousarray(cp.asarray(v_abs, dtype=cp.float64).ravel())
    v_signed = cp.ascontiguousarray(cp.asarray(v_signed, dtype=cp.float64).ravel())
    pq_max_v = cp.ascontiguousarray(cp.asarray(pq_max_v, dtype=cp.float64).ravel())
    out_keys_u64 = cp.ascontiguousarray(cp.asarray(out_keys_u64, dtype=cp.uint64).ravel())
    out_src = cp.ascontiguousarray(cp.asarray(out_src, dtype=cp.int32).ravel())
    out_hij = cp.ascontiguousarray(cp.asarray(out_hij, dtype=cp.float64).ravel())
    selected_idx_sorted_u64 = cp.ascontiguousarray(cp.asarray(selected_idx_sorted_u64, dtype=cp.uint64).ravel())
    if int(out_keys_u64.size) < int(cap) or int(out_src.size) < int(cap) or int(out_hij.size) < int(cap):
        raise ValueError("out_keys_u64, out_src, and out_hij must have length >= cap")
    if out_n is None:
        out_n = cp.zeros((1,), dtype=cp.int32)
    else:
        out_n = cp.ascontiguousarray(cp.asarray(out_n, dtype=cp.int32).ravel())
        if out_n.shape != (1,):
            raise ValueError("out_n must have shape (1,)")
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_exact_selected_emit_tuples_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_bound,
        int(nsel),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        out_keys_u64,
        out_src,
        out_hij,
        int(cap),
        selected_idx_sorted_u64,
        out_n,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_keys_u64, out_src, out_hij, out_n, overflow


def build_selected_membership_hash(sorted_keys_d, cp):
    """Build a GPU hash table for O(1) membership lookup from sorted u64 keys.

    Returns (hash_keys_d, hash_cap) where hash_keys_d is a CuPy uint64 array
    filled with 0xFF (empty sentinel) and populated via the CUDA build kernel.
    """
    if _ext is None or not hasattr(_ext, "cas36_sci_build_membership_hash_u64_inplace_device"):
        return None, 0
    nkeys = int(sorted_keys_d.size)
    if nkeys <= 0:
        return None, 0
    # Power-of-2 capacity at ~50% load factor
    cap = 1
    while cap < 2 * nkeys:
        cap <<= 1
    hash_keys_d = cp.full((cap,), 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)
    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    _ext.cas36_sci_build_membership_hash_u64_inplace_device(
        sorted_keys_d,
        nkeys,
        hash_keys_d,
        cap,
        stream_ptr,
        False,
    )
    return hash_keys_d, cap


def has_cas36_sym_row_graph_spmm_device() -> bool:
    """Return True if the CUDA extension exposes the selected symmetric row-graph SpMM kernel."""
    return _ext is not None and hasattr(_ext, "cas36_sym_row_graph_spmm_inplace_device")


def cas36_sym_row_graph_spmm_device(
    row_ptr,
    col_idx,
    hij,
    diag,
    x,
    cp,
    *,
    out=None,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """Apply a strict-lower symmetric selected graph to one or more vectors."""
    if _ext is None or not hasattr(_ext, "cas36_sym_row_graph_spmm_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the selected symmetric row-graph SpMM kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    row_ptr = cp.ascontiguousarray(cp.asarray(row_ptr, dtype=cp.int64).ravel())
    col_idx = cp.ascontiguousarray(cp.asarray(col_idx, dtype=cp.int32).ravel())
    hij = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())
    diag = cp.ascontiguousarray(cp.asarray(diag, dtype=cp.float64).ravel())
    x_arr = cp.asarray(x, dtype=cp.float64)
    squeeze_out = False
    if int(getattr(x_arr, "ndim", 1)) == 1:
        x_arr = cp.ascontiguousarray(x_arr.reshape((-1, 1)))
        squeeze_out = True
    else:
        x_arr = cp.ascontiguousarray(x_arr)
    if int(x_arr.ndim) != 2:
        raise ValueError("x must be 1D or 2D")
    if int(x_arr.shape[0]) != int(diag.size):
        raise ValueError("x leading dimension must match diag length")
    if out is None:
        y_arr = cp.zeros_like(x_arr)
    else:
        y_arr = cp.ascontiguousarray(cp.asarray(out, dtype=cp.float64))
        if y_arr.shape != x_arr.shape:
            raise ValueError("out must have the same shape as x")
        y_arr.fill(0.0)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_sym_row_graph_spmm_inplace_device(
        row_ptr,
        col_idx,
        hij,
        diag,
        x_arr,
        y_arr,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    if squeeze_out:
        return cp.ascontiguousarray(y_arr[:, 0].ravel())
    return y_arr


def cas36_exact_selected_emit_tuples_dense_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_bound,
    *,
    nsel: int,
    h_base,
    eri4,
    out_keys_u64,
    out_src,
    out_hij,
    cap: int,
    membership_hash_keys=None,
    membership_hash_cap: int = 0,
    out_diag=None,
    out_n=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the dense exact-selected u64 tuple-emission surface."""
    if _ext is None or not hasattr(_ext, "cas36_exact_selected_emit_tuples_dense_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the dense exact-selected CAS36 u64 tuple emitter; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_bound = cp.ascontiguousarray(cp.asarray(c_bound, dtype=cp.float64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    out_keys_u64 = cp.ascontiguousarray(cp.asarray(out_keys_u64, dtype=cp.uint64).ravel())
    out_src = cp.ascontiguousarray(cp.asarray(out_src, dtype=cp.int32).ravel())
    out_hij = cp.ascontiguousarray(cp.asarray(out_hij, dtype=cp.float64).ravel())
    if out_diag is not None:
        out_diag = cp.ascontiguousarray(cp.asarray(out_diag, dtype=cp.float64).ravel())
        if int(out_diag.size) < int(nsel):
            raise ValueError("out_diag must have length >= nsel")
    if int(out_keys_u64.size) < int(cap) or int(out_src.size) < int(cap) or int(out_hij.size) < int(cap):
        raise ValueError("out_keys_u64, out_src, and out_hij must have length >= cap")
    if out_n is None:
        out_n = cp.zeros((1,), dtype=cp.int32)
    else:
        out_n = cp.ascontiguousarray(cp.asarray(out_n, dtype=cp.int32).ravel())
        if out_n.shape != (1,):
            raise ValueError("out_n must have shape (1,)")
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_exact_selected_emit_tuples_dense_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_bound,
        int(nsel),
        h_base,
        eri4,
        out_keys_u64,
        out_src,
        out_hij,
        int(cap),
        membership_hash_keys,
        int(membership_hash_cap),
        out_diag,
        out_n,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_keys_u64, out_src, out_hij, out_diag, out_n, overflow


def cas36_exact_external_emit_tuples_dense_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_bound,
    *,
    nsel: int,
    h_base,
    eri4,
    out_keys_u64,
    out_src,
    out_hij,
    cap: int,
    label_lo: int = 0,
    label_hi: int | None = None,
    membership_hash_keys=None,
    membership_hash_cap: int = 0,
    out_n=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the dense exact-external u64 tuple-emission surface."""
    if _ext is None or not hasattr(_ext, "cas36_exact_external_emit_tuples_dense_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the dense exact-external CAS36 u64 tuple emitter; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_bound = cp.ascontiguousarray(cp.asarray(c_bound, dtype=cp.float64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    out_keys_u64 = cp.ascontiguousarray(cp.asarray(out_keys_u64, dtype=cp.uint64).ravel())
    out_src = cp.ascontiguousarray(cp.asarray(out_src, dtype=cp.int32).ravel())
    out_hij = cp.ascontiguousarray(cp.asarray(out_hij, dtype=cp.float64).ravel())
    if membership_hash_keys is None:
        raise ValueError("membership_hash_keys must not be None for exact external dense tuple emission")
    membership_hash_keys = cp.ascontiguousarray(cp.asarray(membership_hash_keys, dtype=cp.uint64).ravel())
    if int(out_keys_u64.size) < int(cap) or int(out_src.size) < int(cap) or int(out_hij.size) < int(cap):
        raise ValueError("out_keys_u64, out_src, and out_hij must have length >= cap")
    if out_n is None:
        out_n = cp.zeros((1,), dtype=cp.int32)
    else:
        out_n = cp.ascontiguousarray(cp.asarray(out_n, dtype=cp.int32).ravel())
        if out_n.shape != (1,):
            raise ValueError("out_n must have shape (1,)")
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_exact_external_emit_tuples_dense_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_bound,
        int(nsel),
        h_base,
        eri4,
        out_keys_u64,
        out_src,
        out_hij,
        int(cap),
        int(label_lo),
        int(drt.ncsf) if label_hi is None else int(label_hi),
        membership_hash_keys,
        int(membership_hash_cap),
        out_n,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return out_keys_u64, out_src, out_hij, out_n, overflow


def cas36_exact_selected_build_rowhash_dense_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_bound,
    *,
    nsel: int,
    h_base,
    eri4,
    row_hash_keys,
    row_hash_vals,
    row_cap: int,
    membership_hash_keys=None,
    membership_hash_cap: int = 0,
    out_diag=None,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the dense exact-selected rowhash emitter."""
    if _ext is None or not hasattr(_ext, "cas36_exact_selected_build_rowhash_dense_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the dense exact-selected rowhash emitter; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_bound = cp.ascontiguousarray(cp.asarray(c_bound, dtype=cp.float64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    row_hash_keys = cp.ascontiguousarray(cp.asarray(row_hash_keys, dtype=cp.uint64).ravel())
    row_hash_vals = cp.ascontiguousarray(cp.asarray(row_hash_vals, dtype=cp.float64).ravel())
    if out_diag is not None:
        out_diag = cp.ascontiguousarray(cp.asarray(out_diag, dtype=cp.float64).ravel())
        if int(out_diag.size) < int(nsel):
            raise ValueError("out_diag must have length >= nsel")
    if membership_hash_keys is not None:
        membership_hash_keys = cp.ascontiguousarray(cp.asarray(membership_hash_keys, dtype=cp.uint64).ravel())
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")
    total_cap = int(nsel) * int(row_cap)
    if int(row_hash_keys.size) < total_cap or int(row_hash_vals.size) < total_cap:
        raise ValueError("row_hash_keys/row_hash_vals must have length >= nsel * row_cap")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_exact_selected_build_rowhash_dense_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_bound,
        int(nsel),
        h_base,
        eri4,
        row_hash_keys,
        row_hash_vals,
        int(row_cap),
        membership_hash_keys if membership_hash_keys is not None else None,
        int(membership_hash_cap),
        out_diag if out_diag is not None else None,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return row_hash_keys, row_hash_vals, out_diag, overflow


def cas36_exact_external_apply_dense_many_roots_u64_inplace_device(
    drt: DRT,
    drt_dev,
    sel_idx_u64,
    c_sel,
    *,
    nsel: int,
    h_base,
    eri4,
    hash_keys_u64,
    hash_vals,
    cap: int,
    label_lo: int = 0,
    label_hi: int | None = None,
    membership_hash_keys=None,
    membership_hash_cap: int = 0,
    overflow=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the dense exact-external multiroot apply surface into a frontier hash table."""
    if _ext is None or not hasattr(_ext, "cas36_exact_external_apply_dense_many_roots_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing the dense exact-external CAS36 u64 apply kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    c_sel = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    hash_keys_u64 = cp.ascontiguousarray(cp.asarray(hash_keys_u64, dtype=cp.uint64).ravel())
    hash_vals = cp.ascontiguousarray(cp.asarray(hash_vals, dtype=cp.float64))
    if membership_hash_keys is None:
        raise ValueError("membership_hash_keys must not be None for exact external dense apply")
    membership_hash_keys = cp.ascontiguousarray(cp.asarray(membership_hash_keys, dtype=cp.uint64).ravel())
    if hash_vals.ndim != 2:
        raise ValueError("hash_vals must have shape (nroots, cap)")
    if int(hash_vals.shape[1]) < int(cap) or int(hash_keys_u64.size) < int(cap):
        raise ValueError("hash_keys_u64/hash_vals must have trailing dimension >= cap")
    if overflow is None:
        overflow = cp.zeros((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_exact_external_apply_dense_many_roots_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        c_sel,
        int(nsel),
        h_base,
        eri4,
        hash_keys_u64,
        hash_vals,
        int(cap),
        int(label_lo),
        int(drt.ncsf) if label_hi is None else int(label_hi),
        membership_hash_keys,
        int(membership_hash_cap),
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return hash_keys_u64, hash_vals, overflow


def has_pairwise_hij_u64_device() -> bool:
    """Return True if the CUDA extension exposes the pair-wise H[i,j] kernels."""
    return _ext is not None and hasattr(_ext, "pairwise_hij_u64_inplace_device")


def has_pairwise_materialize_u64_device() -> bool:
    """Return True if the CUDA extension exposes the pairwise materialize kernel."""
    return _ext is not None and hasattr(_ext, "pairwise_materialize_u64_inplace_device")


def has_pairwise_hij_bucketed_u64_device() -> bool:
    """Return True if the CUDA extension exposes the bucketed pairwise H builder."""
    return _ext is not None and hasattr(_ext, "pairwise_hij_bucketed_u64_inplace_device")


def has_pairwise_sigma_bucketed_u64_device() -> bool:
    """Return True if the CUDA extension exposes the matrix-free bucketed sigma kernel."""
    return _ext is not None and hasattr(_ext, "pairwise_sigma_bucketed_u64_inplace_device")


def has_pairwise_count_bucketed_u64_device() -> bool:
    """Return True if the CUDA extension exposes the bucketed sparse-count kernel."""
    return _ext is not None and hasattr(_ext, "pairwise_count_bucketed_u64_inplace_device")


def pairwise_materialize_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    cp,
    *,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Materialize step vectors, nodes, occ, b-values for selected CSFs.

    Returns (steps_all, nodes_all, occ_all, b_all) as cupy device arrays.
    """
    if _ext is None or not hasattr(_ext, "pairwise_materialize_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing pairwise materialize kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    norb = int(drt.norb)
    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())

    steps_all = cp.zeros((nsel, norb), dtype=cp.int8)
    nodes_all = cp.zeros((nsel, norb + 1), dtype=cp.int32)
    occ_all = cp.zeros((nsel, norb), dtype=cp.int8)
    b_all = cp.zeros((nsel, norb), dtype=cp.int16)
    overflow = cp.zeros((1,), dtype=cp.int32)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_materialize_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )

    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_materialize_u64 overflow")

    return steps_all, nodes_all, occ_all, b_all


def pairwise_hij_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    h_base,
    eri4,
    materialized,
    cp,
    *,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Build the dense H[nsel, nsel] matrix using pair-wise evaluation.

    Parameters
    ----------
    materialized : tuple
        (steps_all, nodes_all, occ_all, b_all) from pairwise_materialize_u64_device.

    Returns
    -------
    H_d : cupy array, shape (nsel, nsel), dtype float64
        Dense Hamiltonian matrix (symmetrized).
    diag_d : cupy array, shape (nsel,), dtype float64
        Diagonal elements.
    """
    if _ext is None or not hasattr(_ext, "pairwise_hij_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing pairwise hij kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())

    steps_all, nodes_all, occ_all, b_all = materialized

    H_d = cp.zeros((nsel, nsel), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_hij_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        h_base,
        eri4,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        H_d.ravel(),
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )

    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_hij_u64 overflow")

    # Kernel computes upper triangle and mirrors to lower in-place (no temp allocation).
    diag_d = cp.ascontiguousarray(cp.diag(H_d))

    return H_d, diag_d


def pairwise_hij_bucketed_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    h_base,
    eri4,
    materialized,
    bucket_data,
    cp,
    *,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Build dense H[nsel, nsel] using bucketed pair-wise evaluation.

    The materialized arrays and sel_idx_u64 must be pre-sorted by occupation key
    (using bucket_data["sort_perm"]). Output H is in sorted order.

    Parameters
    ----------
    materialized : tuple
        (steps_all, nodes_all, occ_all, b_all) — already sorted by sort_perm.
    bucket_data : dict
        From pairwise_build_bucket_data(). Must contain: csf_to_bucket,
        bucket_starts, bucket_sizes, neighbor_offsets, neighbor_list.

    Returns
    -------
    H_d : cupy array, shape (nsel, nsel), dtype float64
        Dense H in SORTED order. Caller must unpermute if original order needed.
    diag_d : cupy array, shape (nsel,), dtype float64
        Diagonal in sorted order.
    """
    if _ext is None or not hasattr(_ext, "pairwise_hij_bucketed_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension missing bucketed pairwise kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())

    steps_all, nodes_all, occ_all, b_all = materialized

    H_d = cp.zeros((nsel, nsel), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)
    mean_bucket_size = float(nsel) / max(int(bucket_data["nbuckets"]), 1)
    phase2_mode = os.environ.get("ASUKA_PAIRWISE_PHASE2_MODE", "flat").strip().lower()
    if phase2_mode == "row_union":
        phase2_mode_code = 2
    elif phase2_mode == "compaction":
        phase2_mode_code = 1
    elif phase2_mode == "auto":
        # Neither experimental Phase 2 variant currently beats the flat
        # packed-key path on the large benchmark suite.
        phase2_mode_code = 0
    else:
        phase2_mode_code = 0

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_hij_bucketed_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        h_base,
        eri4,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        cp.ascontiguousarray(bucket_data["occ_keys_sorted"]),
        cp.ascontiguousarray(bucket_data["bucket_keys"]),
        cp.ascontiguousarray(bucket_data["bucket_starts"]),
        cp.ascontiguousarray(bucket_data["bucket_sizes"]),
        cp.ascontiguousarray(bucket_data["neighbor_offsets"]),
        cp.ascontiguousarray(bucket_data["neighbor_list"]),
        cp.ascontiguousarray(bucket_data["csf_to_bucket"]),
        cp.ascontiguousarray(bucket_data["target_offsets"]),
        cp.ascontiguousarray(bucket_data["target_list"]),
        cp.ascontiguousarray(bucket_data["target_offsets_1b"]),
        cp.ascontiguousarray(bucket_data["target_list_1b"]),
        int(phase2_mode_code),
        H_d.ravel(),
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )

    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_hij_bucketed_u64 overflow")

    diag_d = cp.ascontiguousarray(cp.diag(H_d))
    return H_d, diag_d


def pairwise_build_selected_graph_bucketed_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    h_base,
    eri4,
    materialized,
    bucket_data,
    cp,
    *,
    threads: int = 256,
    grid_blocks: int | None = None,
    table_cap: int | None = None,
    stream=None,
    sync: bool = True,
):
    """Build exact selected-space off-diagonal local tuples + diagonal via bucketed pairwise traversal.

    This is the sparse/matrix-free companion to `pairwise_hij_bucketed_u64_device`.
    It avoids materializing dense `H[nsel,nsel]` by accumulating unique `(target_local, src_local)`
    edges per source row in a row-local hash table and returning compressed local COO tuples.
    """
    if _ext is None or not hasattr(_ext, "pairwise_count_bucketed_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension missing bucketed sparse pairwise graph builder; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    steps_all, nodes_all, occ_all, b_all = materialized

    if grid_blocks is None:
        grid_blocks = min(int(nsel), int(os.environ.get("ASUKA_PAIRWISE_GRAPH_BLOCKS", "120")))
    grid_blocks = max(1, int(grid_blocks))

    if table_cap is None:
        tgt_offsets = cp.ascontiguousarray(bucket_data["target_offsets"]).astype(cp.int32, copy=False)
        max_targets = int(cp.max(tgt_offsets[1:] - tgt_offsets[:-1]).item()) if int(tgt_offsets.size) > 1 else 0
        cap = 1
        need = max(256, 2 * max(1, max_targets))
        while cap < need:
            cap <<= 1
        table_cap = cap
    table_cap = int(table_cap)

    workspace_keys = cp.full((grid_blocks * table_cap,), -1, dtype=cp.int32)
    workspace_vals = cp.zeros((grid_blocks * table_cap,), dtype=cp.float64)
    row_counts = cp.zeros((nsel,), dtype=cp.int32)
    diag_d = cp.zeros((nsel,), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    common_args = (
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        h_base,
        eri4,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        cp.ascontiguousarray(bucket_data["occ_keys_sorted"]),
        cp.ascontiguousarray(bucket_data["bucket_keys"]),
        cp.ascontiguousarray(bucket_data["bucket_starts"]),
        cp.ascontiguousarray(bucket_data["bucket_sizes"]),
        cp.ascontiguousarray(bucket_data["neighbor_offsets"]),
        cp.ascontiguousarray(bucket_data["neighbor_list"]),
        cp.ascontiguousarray(bucket_data["csf_to_bucket"]),
        cp.ascontiguousarray(bucket_data["target_offsets"]),
        cp.ascontiguousarray(bucket_data["target_list"]),
        cp.ascontiguousarray(bucket_data["target_offsets_1b"]),
        cp.ascontiguousarray(bucket_data["target_list_1b"]),
        int(table_cap),
        workspace_keys,
        workspace_vals,
        int(grid_blocks),
    )

    _ext.pairwise_count_bucketed_u64_inplace_device(
        *common_args,
        row_counts,
        diag_d,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_count_bucketed_u64 overflow")

    row_offsets = cp.zeros((nsel + 1,), dtype=cp.int64)
    if int(nsel) > 0:
        row_offsets[1:] = cp.cumsum(row_counts.astype(cp.int64, copy=False), dtype=cp.int64)
    nnz = int(row_offsets[-1].item())
    if nnz == 0:
        return (
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.int32),
            cp.zeros((0,), dtype=cp.float64),
            diag_d,
            row_counts,
        )

    workspace_keys.fill(-1)
    workspace_vals.fill(0.0)
    overflow.fill(0)
    out_target_local = cp.empty((nnz,), dtype=cp.int32)
    out_src_pos = cp.empty((nnz,), dtype=cp.int32)
    out_hij = cp.empty((nnz,), dtype=cp.float64)

    _ext.pairwise_fill_bucketed_u64_inplace_device(
        *common_args,
        row_offsets,
        row_counts,
        out_target_local,
        out_src_pos,
        out_hij,
        diag_d,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_fill_bucketed_u64 overflow")

    return out_target_local, out_src_pos, out_hij, diag_d, row_counts


def pairwise_diag_bucketed_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    h_base,
    eri4,
    materialized,
    bucket_data,
    cp,
    *,
    threads: int = 256,
    grid_blocks: int | None = None,
    table_cap: int | None = None,
    stream=None,
    sync: bool = True,
):
    """Build the exact selected-space diagonal via the matrix-free pairwise path.

    Inputs must already be sorted consistently with the materialized arrays.
    ``bucket_data`` is accepted for API symmetry with the sigma builder path,
    but the exact diagonal kernel does not need the bucket metadata.
    """
    if _ext is None or not hasattr(_ext, "pairwise_diag_bucketed_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension missing bucketed pairwise diagonal kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    steps_all, nodes_all, occ_all, b_all = materialized
    diag_d = cp.zeros((nsel,), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_diag_bucketed_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        h_base,
        eri4,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        diag_d,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_diag_bucketed_u64 overflow")
    return diag_d


def pairwise_sigma_bucketed_u64_device(
    drt,
    drt_dev,
    sel_idx_u64,
    nsel: int,
    h_base,
    eri4,
    materialized,
    bucket_data,
    x,
    cp,
    *,
    out=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Apply y = P_S H P_S x using the exact bucketed pairwise traversal.

    This path is matrix-free: it reuses the bucketed pairwise traversal but
    accumulates directly into y instead of materializing dense H.
    Inputs must already be sorted by bucket_data["sort_perm"].
    """
    if _ext is None or not hasattr(_ext, "pairwise_sigma_bucketed_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension missing bucketed pairwise sigma kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    sel_idx_u64 = cp.ascontiguousarray(cp.asarray(sel_idx_u64, dtype=cp.uint64).ravel())
    h_base = cp.ascontiguousarray(cp.asarray(h_base, dtype=cp.float64).ravel())
    eri4 = cp.ascontiguousarray(cp.asarray(eri4, dtype=cp.float64).ravel())
    steps_all, nodes_all, occ_all, b_all = materialized

    x_arr = cp.asarray(x, dtype=cp.float64)
    squeeze_out = False
    if int(getattr(x_arr, "ndim", 1)) == 1:
        x_arr = cp.ascontiguousarray(x_arr.reshape((int(nsel), 1)))
        squeeze_out = True
    else:
        x_arr = cp.ascontiguousarray(x_arr)
    if int(x_arr.ndim) != 2:
        raise ValueError("x must be 1D or 2D")
    if int(x_arr.shape[0]) != int(nsel):
        raise ValueError("x leading dimension must match nsel")

    if out is None:
        y_arr = cp.zeros_like(x_arr)
    else:
        y_arr = cp.ascontiguousarray(cp.asarray(out, dtype=cp.float64))
        if y_arr.shape != x_arr.shape:
            raise ValueError("out must have the same shape as x")
        y_arr.fill(0.0)

    overflow = cp.zeros((1,), dtype=cp.int32)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_sigma_bucketed_u64_inplace_device(
        drt_dev,
        int(drt.ncsf),
        sel_idx_u64,
        int(nsel),
        h_base,
        eri4,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        cp.ascontiguousarray(bucket_data["occ_keys_sorted"]),
        cp.ascontiguousarray(bucket_data["csf_to_bucket"]),
        cp.ascontiguousarray(bucket_data["target_offsets"]),
        cp.ascontiguousarray(bucket_data["target_list"]),
        cp.ascontiguousarray(bucket_data["target_offsets_1b"]),
        cp.ascontiguousarray(bucket_data["target_list_1b"]),
        x_arr,
        y_arr,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )

    if int(overflow.item()) != 0:
        raise RuntimeError("pairwise_sigma_bucketed_u64 overflow")
    if squeeze_out:
        return cp.ascontiguousarray(y_arr[:, 0].ravel())
    return y_arr


def pairwise_T_matrix_bucketed_u64_device(
    drt,
    drt_dev,
    materialized_sorted,
    bucket_data,
    ci_sel,
    nsel: int,
    cp,
    *,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Compute T[pq, i] = sum_j E_pq[i,j] * c[j] for all orbital pairs (p,q).

    T has shape (norb*norb, nsel) on GPU. Use T @ c for dm1 and T @ T.T for dm2.
    Inputs must already be sorted by bucket_data["sort_perm"].

    Parameters
    ----------
    materialized_sorted : tuple
        (steps_all, nodes_all, occ_all, b_all) sorted by occ key.
    bucket_data : dict
        Bucket data from pairwise_build_bucket_data (sorted version).
    ci_sel : cupy array, shape (nsel,)
        CI coefficients in sorted order.

    Returns
    -------
    T_d : cupy array, shape (norb*norb, nsel), float64
    """
    if _ext is None or not hasattr(_ext, "pairwise_T_matrix_bucketed_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension missing T-matrix kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    norb = int(drt.norb)
    steps_all, nodes_all, occ_all, b_all = materialized_sorted

    ci_arr = cp.ascontiguousarray(cp.asarray(ci_sel, dtype=cp.float64).ravel())
    T_d = cp.zeros((norb * norb, nsel), dtype=cp.float64)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.pairwise_T_matrix_bucketed_u64_inplace_device(
        drt_dev,
        steps_all.ravel(),
        nodes_all.ravel(),
        occ_all.ravel(),
        b_all.ravel(),
        cp.ascontiguousarray(bucket_data["csf_to_bucket"]),
        cp.ascontiguousarray(bucket_data["target_offsets_1b"]),
        cp.ascontiguousarray(bucket_data["target_list_1b"]),
        ci_arr,
        T_d.ravel(),
        int(nsel),
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return T_d


def pairwise_compute_occ_keys(occ_all, norb: int, cp):
    """Pack occupation vectors into uint64 keys (2 bits per orbital).

    Parameters
    ----------
    occ_all : cupy array, shape (nsel, norb), dtype int8
        Occupation numbers (0, 1, 2) from pairwise_materialize_u64_device.
    norb : int
        Number of orbitals (must be <= 32 for uint64 packing).
    cp : cupy module

    Returns
    -------
    keys : cupy array, shape (nsel,), dtype uint64
        Packed occupation keys.
    """
    assert norb <= 32, f"occ key packing requires norb <= 32, got {norb}"
    shifts = cp.arange(norb, dtype=cp.uint64) * 2  # [norb]
    occ_u64 = occ_all.astype(cp.uint64)  # [nsel, norb]
    keys = (occ_u64 << shifts[None, :]).sum(axis=1)  # [nsel]
    return keys


def pairwise_build_occ_buckets(occ_keys, cp):
    """Sort CSFs by occupation key and compute bucket boundaries.

    Parameters
    ----------
    occ_keys : cupy array, shape (nsel,), dtype uint64
        Packed occupation keys from pairwise_compute_occ_keys.

    Returns
    -------
    sort_perm : cupy array, shape (nsel,), dtype int64
        Permutation that sorts CSFs by occupation key.
    sorted_keys : cupy array, shape (nsel,), dtype uint64
        Sorted occupation keys.
    bucket_starts : cupy array, shape (nbuckets,), dtype int64
        Start index of each bucket in the sorted order.
    bucket_keys : cupy array, shape (nbuckets,), dtype uint64
        Unique occupation key for each bucket.
    """
    sort_perm = cp.argsort(occ_keys)
    sorted_keys = occ_keys[sort_perm]

    # Find bucket boundaries: where sorted_keys changes value
    if len(sorted_keys) == 0:
        return sort_perm, sorted_keys, cp.zeros((0,), dtype=cp.int64), cp.zeros((0,), dtype=cp.uint64)

    changes = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1],
    ])
    bucket_starts = cp.nonzero(changes)[0].astype(cp.int64)
    bucket_keys = sorted_keys[bucket_starts]

    return sort_perm, sorted_keys, bucket_starts, bucket_keys


def pairwise_build_neighbor_csr(bucket_keys, norb, cp, *, max_occ_diff=4):
    """Build CSR neighbor list: for each bucket, which other buckets are within max_occ_diff.

    Parameters
    ----------
    bucket_keys : cupy array, shape (nbuckets,), dtype uint64
        Unique occupation keys (from pairwise_build_occ_buckets).
    norb : int
        Number of orbitals.
    max_occ_diff : int
        Maximum number of occupation positions that can differ (4 for two-body).

    Returns
    -------
    neighbor_offsets : cupy array, shape (nbuckets+1,), dtype int32
        CSR offsets: bucket b's neighbors are neighbor_list[offsets[b]:offsets[b+1]].
    neighbor_list : cupy array, shape (total_neighbors,), dtype int32
        Flat list of neighbor bucket indices.
    """
    import numpy as np

    nbuckets = len(bucket_keys)
    if nbuckets == 0:
        return cp.zeros((1,), dtype=cp.int32), cp.zeros((0,), dtype=cp.int32)

    bucket_keys_h = cp.asnumpy(bucket_keys)

    xor = np.bitwise_xor(bucket_keys_h[:, None], bucket_keys_h[None, :])
    nonzero = np.bitwise_and(np.bitwise_or(xor, np.right_shift(xor, 1)), np.uint64(0x5555555555555555))
    diff_matrix = np.bitwise_count(nonzero).astype(np.int32, copy=False)

    mask = diff_matrix <= max_occ_diff
    counts = mask.sum(axis=1, dtype=np.int32)
    offsets = np.zeros(nbuckets + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    neighbor_list = np.nonzero(mask)[1].astype(np.int32, copy=False)

    return cp.asarray(offsets), cp.asarray(neighbor_list)


def _build_flat_target_list(neighbor_offsets_h, neighbor_list_h, bucket_starts_h, bucket_sizes_h, nbuckets):
    """Build flat target list from neighbor CSR. Returns (offsets, list) as numpy arrays."""
    import numpy as np

    if nbuckets == 0 or int(neighbor_list_h.size) == 0:
        return np.zeros((nbuckets + 1,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    pair_sizes = bucket_sizes_h[neighbor_list_h].astype(np.int64, copy=False)
    pair_prefix = np.zeros((int(pair_sizes.size) + 1,), dtype=np.int64)
    np.cumsum(pair_sizes, out=pair_prefix[1:])

    offsets = pair_prefix[neighbor_offsets_h.astype(np.int64, copy=False)].astype(np.int32, copy=False)
    total = int(pair_prefix[-1])
    if total == 0:
        return offsets, np.zeros((0,), dtype=np.int32)

    pair_starts = bucket_starts_h[neighbor_list_h].astype(np.int64, copy=False)
    pair_delta = pair_starts - pair_prefix[:-1]
    tlist = (np.repeat(pair_delta, pair_sizes) + np.arange(total, dtype=np.int64)).astype(np.int32, copy=False)
    return offsets, tlist


def pairwise_build_bucket_data(occ_all, norb, cp):
    """Full pipeline: occ keys → buckets → neighbor CSR → flat target lists.

    Returns a dict with all data needed for bucketed kernel dispatch.
    Builds two target lists:
      - target_offsets/target_list: occ_diff <= 4 (two-body fallback)
      - target_offsets_1b/target_list_1b: occ_diff <= 2 (one-body + k-relative)
    """
    import numpy as np

    nsel = occ_all.shape[0]

    keys = pairwise_compute_occ_keys(occ_all, norb, cp)
    sort_perm, sorted_keys, bucket_starts, bucket_keys = pairwise_build_occ_buckets(keys, cp)

    # Build neighbor CSRs for both <=4 and <=2 (share diff_matrix computation)
    nbuckets = len(bucket_keys)
    if nbuckets > 0:
        bucket_keys_h = cp.asnumpy(bucket_keys)
        xor = np.bitwise_xor(bucket_keys_h[:, None], bucket_keys_h[None, :])
        nonzero = np.bitwise_and(np.bitwise_or(xor, np.right_shift(xor, 1)), np.uint64(0x5555555555555555))
        diff_matrix = np.bitwise_count(nonzero).astype(np.int32, copy=False)

        def _diff_to_csr(max_diff):
            mask = diff_matrix <= max_diff
            counts = mask.sum(axis=1, dtype=np.int32)
            off = np.zeros(nbuckets + 1, dtype=np.int32)
            np.cumsum(counts, out=off[1:])
            nlist = np.nonzero(mask)[1].astype(np.int32, copy=False)
            return off, nlist

        neighbor_offsets_h, neighbor_list_h = _diff_to_csr(4)
        neighbor_offsets_1b_h, neighbor_list_1b_h = _diff_to_csr(2)
    else:
        neighbor_offsets_h = np.zeros((1,), dtype=np.int32)
        neighbor_list_h = np.zeros((0,), dtype=np.int32)
        neighbor_offsets_1b_h = np.zeros((1,), dtype=np.int32)
        neighbor_list_1b_h = np.zeros((0,), dtype=np.int32)

    neighbor_offsets = cp.asarray(neighbor_offsets_h)
    neighbor_list = cp.asarray(neighbor_list_h)

    # Compute bucket sizes
    bucket_starts_ext = cp.concatenate([bucket_starts, cp.array([nsel], dtype=cp.int64)])
    bucket_sizes = (bucket_starts_ext[1:] - bucket_starts_ext[:-1]).astype(cp.int32)

    # Build inverse permutation: inv_perm[original_idx] = sorted_position
    inv_perm = cp.zeros(nsel, dtype=cp.int32)
    inv_perm[sort_perm] = cp.arange(nsel, dtype=cp.int32)

    bucket_starts_h = cp.asnumpy(bucket_starts)
    bucket_sizes_h = cp.asnumpy(bucket_sizes)
    # Map each sorted CSF to its bucket index.
    csf_to_bucket = cp.asarray(
        np.repeat(np.arange(nbuckets, dtype=np.int32), bucket_sizes_h.astype(np.int64, copy=False)),
        dtype=cp.int32,
    ) if nbuckets > 0 else cp.zeros((0,), dtype=cp.int32)

    # Build flat target lists for both <=4 and <=2
    target_offsets, target_list = _build_flat_target_list(
        neighbor_offsets_h, neighbor_list_h, bucket_starts_h, bucket_sizes_h, nbuckets)
    target_offsets_1b, target_list_1b = _build_flat_target_list(
        neighbor_offsets_1b_h, neighbor_list_1b_h, bucket_starts_h, bucket_sizes_h, nbuckets)

    return {
        "sort_perm": sort_perm.astype(cp.int32),
        "inv_perm": inv_perm,
        "bucket_starts": bucket_starts.astype(cp.int32),
        "bucket_sizes": bucket_sizes,
        "bucket_keys": bucket_keys,
        "nbuckets": nbuckets,
        "neighbor_offsets": neighbor_offsets,
        "neighbor_list": neighbor_list,
        "csf_to_bucket": csf_to_bucket,
        "occ_keys": keys,
        "occ_keys_sorted": sorted_keys,
        "target_offsets": cp.asarray(target_offsets),
        "target_list": cp.asarray(target_list),
        "target_offsets_1b": cp.asarray(target_offsets_1b),
        "target_list_1b": cp.asarray(target_list_1b),
    }


def cas36_diag_guess_candidates_u64_dense_inplace_device(
    drt: DRT,
    drt_dev,
    cand_idx_u64,
    *,
    h1_diag,
    eri_ppqq,
    eri_pqqp,
    neleca: int,
    nelecb: int,
    hdiag_out=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Compute compact diagonal guesses on GPU for u64 candidate labels."""
    if _ext is None or not hasattr(_ext, "cas36_diag_guess_candidates_u64_dense_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing CAS36 compact diagonal-guess kernels; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    cand_idx_u64 = cp.ascontiguousarray(cp.asarray(cand_idx_u64, dtype=cp.uint64).ravel())
    h1_diag = cp.ascontiguousarray(cp.asarray(h1_diag, dtype=cp.float64).ravel())
    eri_ppqq = cp.ascontiguousarray(cp.asarray(eri_ppqq, dtype=cp.float64).ravel())
    eri_pqqp = cp.ascontiguousarray(cp.asarray(eri_pqqp, dtype=cp.float64).ravel())
    if hdiag_out is None:
        hdiag_out = cp.empty((int(cand_idx_u64.size),), dtype=cp.float64)
    else:
        hdiag_out = cp.ascontiguousarray(cp.asarray(hdiag_out, dtype=cp.float64).ravel())
        if int(hdiag_out.size) != int(cand_idx_u64.size):
            raise ValueError("hdiag_out must have the same length as cand_idx_u64")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_diag_guess_candidates_u64_dense_inplace_device(
        drt_dev,
        int(drt.ncsf),
        cand_idx_u64,
        h1_diag,
        eri_ppqq,
        eri_pqqp,
        int(neleca),
        int(nelecb),
        hdiag_out,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return hdiag_out


def cas36_cipsi_score_pt2_compact_u64_inplace_device(
    idx_u64,
    vals_root_major,
    *,
    e_var,
    cand_hdiag,
    selected_idx_sorted_u64=None,
    denom_floor: float,
    score_bits_out=None,
    pt2_out=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Compute compact candidate scores + deterministic PT2 using u64 labels on GPU."""
    if _ext is None or not hasattr(_ext, "cas36_cipsi_score_pt2_compact_u64_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing CAS36 compact score/PT2 kernels; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    idx_u64 = cp.ascontiguousarray(cp.asarray(idx_u64, dtype=cp.uint64).ravel())
    vals_root_major = cp.ascontiguousarray(cp.asarray(vals_root_major, dtype=cp.float64))
    if vals_root_major.ndim != 2:
        raise ValueError("vals_root_major must have shape (nroots, ncand)")
    nroots = int(vals_root_major.shape[0])
    ncand = int(idx_u64.size)
    if int(vals_root_major.shape[1]) < ncand:
        raise ValueError("vals_root_major second dimension must be >= number of candidates")
    e_var = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())
    cand_hdiag = cp.ascontiguousarray(cp.asarray(cand_hdiag, dtype=cp.float64).ravel())
    if int(e_var.size) != nroots:
        raise ValueError("e_var must have shape (nroots,)")
    if int(cand_hdiag.size) != ncand:
        raise ValueError("cand_hdiag must have the same length as idx_u64")
    if selected_idx_sorted_u64 is not None:
        selected_idx_sorted_u64 = cp.ascontiguousarray(cp.asarray(selected_idx_sorted_u64, dtype=cp.uint64).ravel())

    if score_bits_out is None:
        score_bits_out = cp.empty((ncand,), dtype=cp.uint64)
    else:
        score_bits_out = cp.ascontiguousarray(cp.asarray(score_bits_out, dtype=cp.uint64).ravel())
        if int(score_bits_out.size) != ncand:
            raise ValueError("score_bits_out must have same length as idx_u64")
    if pt2_out is None:
        pt2_out = cp.zeros((nroots,), dtype=cp.float64)
    else:
        pt2_out = cp.ascontiguousarray(cp.asarray(pt2_out, dtype=cp.float64).ravel())
        if int(pt2_out.size) != nroots:
            raise ValueError("pt2_out must have shape (nroots,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.cas36_cipsi_score_pt2_compact_u64_inplace_device(
        idx_u64,
        vals_root_major,
        e_var,
        cand_hdiag,
        selected_idx_sorted_u64 if selected_idx_sorted_u64 is not None else None,
        float(denom_floor),
        score_bits_out,
        pt2_out,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )
    return score_bits_out, pt2_out


def has_hb_screen_and_apply_device() -> bool:
    """Return True if the CUDA extension exposes the HB screen-and-apply kernel."""
    return _ext is not None and hasattr(_ext, "hb_screen_and_apply_inplace_device")


def has_hb_screen_and_apply_many_roots_device() -> bool:
    """Return True if the CUDA extension exposes the HB many-roots kernel."""
    return _ext is not None and hasattr(_ext, "hb_screen_and_apply_many_roots_inplace_device")


def hb_screen_and_apply_inplace_device(
    drt,
    drt_dev,
    state_dev,
    sel_idx,
    c_root,
    *,
    nsel: int,
    nroots: int,
    root: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    eps: float,
    hash_keys,
    hash_vals,
    selected_mask=None,
    overflow=None,
    sym_pq_allowed=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the fused heat-bath screen + apply kernel."""
    if _ext is None or not hasattr(_ext, "hb_screen_and_apply_inplace_device"):
        raise RuntimeError("CUDA extension is missing HB-SCI kernels; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    sel_idx = cp.ascontiguousarray(cp.asarray(sel_idx, dtype=cp.int32).ravel())
    c_root = cp.ascontiguousarray(cp.asarray(c_root, dtype=cp.float64).ravel())
    if selected_mask is not None:
        selected_mask = cp.ascontiguousarray(cp.asarray(selected_mask, dtype=cp.uint8).ravel())
    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    _ext.hb_screen_and_apply_inplace_device(
        drt_dev,
        state_dev,
        sel_idx,
        c_root,
        int(nsel),
        int(nroots),
        int(root),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        float(eps),
        hash_keys,
        hash_vals,
        selected_mask if selected_mask is not None else None,
        overflow,
        sym_pq_allowed if sym_pq_allowed is not None else None,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )


def hb_screen_and_apply_many_roots_inplace_device(
    drt,
    drt_dev,
    state_dev,
    sel_idx,
    c_sel,
    *,
    nsel: int,
    nroots: int,
    h1_pq,
    h1_abs,
    h1_signed,
    n_h1: int,
    pq_ptr,
    rs_idx,
    v_abs,
    v_signed,
    pq_max_v,
    eps: float,
    hash_keys,
    hash_vals,
    selected_mask=None,
    overflow=None,
    sym_pq_allowed=None,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Launch the fused heat-bath screen+apply kernel for all roots."""
    if _ext is None or not hasattr(_ext, "hb_screen_and_apply_many_roots_inplace_device"):
        raise RuntimeError("CUDA extension is missing HB-SCI many-roots kernels; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required") from e

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    sel_idx = cp.ascontiguousarray(cp.asarray(sel_idx, dtype=cp.int32).ravel())
    c_sel = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
    if c_sel.ndim != 2:
        raise ValueError("c_sel must have shape (nsel, nroots)")
    if int(c_sel.shape[0]) != int(nsel) or int(c_sel.shape[1]) != int(nroots):
        raise ValueError("c_sel must have shape (nsel, nroots)")
    if selected_mask is not None:
        selected_mask = cp.ascontiguousarray(cp.asarray(selected_mask, dtype=cp.uint8).ravel())
    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    _ext.hb_screen_and_apply_many_roots_inplace_device(
        drt_dev,
        state_dev,
        sel_idx,
        c_sel,
        int(nsel),
        int(nroots),
        h1_pq,
        h1_abs,
        h1_signed,
        int(n_h1),
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        float(eps),
        hash_keys,
        hash_vals,
        selected_mask if selected_mask is not None else None,
        overflow,
        sym_pq_allowed if sym_pq_allowed is not None else None,
        int(threads),
        int(stream_ptr),
        bool(sync),
    )


def apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf,
    task_g,
    *,
    task_scale=None,
    hash_keys,
    hash_vals,
    root: int,
    selected_mask=None,
    overflow=None,
    clear_overflow: bool = True,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Apply G(pq) to a list of CSFs and accumulate results into a frontier hash map on GPU."""
    if _ext is None or not hasattr(_ext, "apply_g_flat_scatter_atomic_frontier_hash_inplace_device"):
        raise RuntimeError("CUDA extension is missing frontier-hash apply kernel; rebuild with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    task_csf = cp.asarray(task_csf, dtype=cp.int32).ravel()
    task_csf = cp.ascontiguousarray(task_csf)
    task_g = cp.asarray(task_g, dtype=cp.float64)
    task_g = cp.ascontiguousarray(task_g)
    if task_g.ndim not in (1, 2):
        raise ValueError("task_g must be 1D (norb*norb,) or 2D (ntasks,norb*norb)")
    nops = int(drt.norb) * int(drt.norb)
    if task_g.ndim == 1 and task_g.shape != (nops,):
        raise ValueError("task_g (1D) must have shape (norb*norb,)")
    if task_g.ndim == 2 and task_g.shape != (int(task_csf.size), nops):
        raise ValueError("task_g (2D) must have shape (ntasks,norb*norb)")

    if task_scale is not None:
        task_scale = cp.asarray(task_scale, dtype=cp.float64).ravel()
        task_scale = cp.ascontiguousarray(task_scale)
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    hash_keys = cp.asarray(hash_keys, dtype=cp.int32).ravel()
    hash_keys = cp.ascontiguousarray(hash_keys)
    hash_vals = cp.asarray(hash_vals, dtype=cp.float64)
    hash_vals = cp.ascontiguousarray(hash_vals)
    if hash_vals.ndim != 2:
        raise ValueError("hash_vals must have shape (nroots, cap)")
    if hash_vals.shape[1] != hash_keys.shape[0]:
        raise ValueError("hash_vals must have shape (nroots, cap) with cap matching hash_keys")
    if selected_mask is not None:
        selected_mask = cp.asarray(selected_mask, dtype=cp.uint8).ravel()
        selected_mask = cp.ascontiguousarray(selected_mask)
        if selected_mask.shape != (int(drt.ncsf),):
            raise ValueError("selected_mask must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    if bool(clear_overflow):
        cp.cuda.runtime.memsetAsync(int(overflow.data.ptr), 0, 4, int(stream_ptr))

    _ext.apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_g,
        task_scale if task_scale is not None else None,
        hash_keys,
        hash_vals,
        int(root),
        selected_mask if selected_mask is not None else None,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return hash_keys, hash_vals, overflow


def apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf,
    task_g,
    *,
    task_scale_task_major,
    hash_keys,
    hash_vals,
    selected_mask=None,
    overflow=None,
    clear_overflow: bool = True,
    threads: int = 256,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Apply frontier-hash scatter for all roots with task-major coefficients.

    Parameters
    ----------
    task_scale_task_major : array, shape (ntasks, nroots)
        Task-major coefficient matrix where column `r` is used as `task_scale`
        for `root=r`. A broadcast row shape `(1, nroots)` is also accepted.
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    if _ext is None or not hasattr(_ext, "apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device"):
        raise RuntimeError(
            "CUDA extension is missing many-roots frontier-hash apply kernel; rebuild with python -m asuka.build.guga_cuda_ext"
        )

    task_csf = cp.ascontiguousarray(cp.asarray(task_csf, dtype=cp.int32).ravel())
    task_g = cp.asarray(task_g, dtype=cp.float64)
    task_g = cp.ascontiguousarray(task_g)
    if task_g.ndim not in (1, 2):
        raise ValueError("task_g must be 1D (norb*norb,) or 2D (ntasks,norb*norb)")
    nops = int(drt.norb) * int(drt.norb)
    if task_g.ndim == 1 and task_g.shape != (nops,):
        raise ValueError("task_g (1D) must have shape (norb*norb,)")
    if task_g.ndim == 2 and task_g.shape != (int(task_csf.size), nops):
        raise ValueError("task_g (2D) must have shape (ntasks,norb*norb)")

    task_scale_task_major = cp.ascontiguousarray(cp.asarray(task_scale_task_major, dtype=cp.float64))
    hash_keys = cp.ascontiguousarray(cp.asarray(hash_keys, dtype=cp.int32).ravel())
    hash_vals = cp.ascontiguousarray(cp.asarray(hash_vals, dtype=cp.float64))
    if task_scale_task_major.ndim != 2:
        raise ValueError("task_scale_task_major must have shape (ntasks, nroots)")
    if int(task_scale_task_major.shape[0]) not in (1, int(task_csf.size)):
        raise ValueError("task_scale_task_major first dimension must be 1 or match ntasks")
    if hash_vals.ndim != 2:
        raise ValueError("hash_vals must have shape (nroots, cap)")
    if int(hash_vals.shape[1]) != int(hash_keys.shape[0]):
        raise ValueError("hash_vals must have shape (nroots, cap) with cap matching hash_keys")
    if int(task_scale_task_major.shape[1]) != int(hash_vals.shape[0]):
        raise ValueError("task_scale_task_major second dimension must match hash_vals.shape[0]")
    if selected_mask is not None:
        selected_mask = cp.asarray(selected_mask, dtype=cp.uint8).ravel()
        selected_mask = cp.ascontiguousarray(selected_mask)
        if selected_mask.shape != (int(drt.ncsf),):
            raise ValueError("selected_mask must have shape (ncsf,)")
    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")
    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    if bool(clear_overflow):
        cp.cuda.runtime.memsetAsync(int(overflow.data.ptr), 0, 4, int(stream_ptr))

    _ext.apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_g,
        task_scale_task_major,
        hash_keys,
        hash_vals,
        selected_mask if selected_mask is not None else None,
        overflow,
        int(threads),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )
    return hash_keys, hash_vals, overflow


def apply_g_flat_scatter_atomic_epq_table_tile_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    local_indptr,
    indices,
    pq_ids,
    epq_data,
    task_g,
    *,
    task_scale=None,
    j_start: int,
    j_count: int,
    y=None,
    overflow=None,
    threads: int = 256,
    zero_y: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    dtype=None,
    use_kahan: bool = False,
):
    """Apply a local-indptr EPQ tile in-place.

    The tile rows correspond to global source CSFs `[j_start, j_start + j_count)`.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0 or (j_start + j_count) > int(drt.ncsf):
        raise ValueError("j_start/j_count out of range")

    local_indptr = cp.ascontiguousarray(cp.asarray(local_indptr, dtype=cp.int64).ravel())
    indices = cp.ascontiguousarray(cp.asarray(indices, dtype=cp.int32).ravel())
    pq_ids = _as_epq_pq_array(cp, pq_ids, name="pq_ids")
    _validate_epq_pq_capacity(cp, pq_ids, norb=int(drt.norb), name="pq_ids")
    epq_data = cp.asarray(epq_data).ravel()
    if epq_data.dtype not in (cp.float32, cp.float64):
        epq_data = cp.asarray(epq_data, dtype=fp_dtype).ravel()
    if fp_dtype == cp.float32 and epq_data.dtype == cp.float64:
        epq_data = cp.asarray(epq_data, dtype=cp.float32).ravel()
    epq_data = cp.ascontiguousarray(epq_data)
    if local_indptr.shape != (int(j_count) + 1,):
        raise ValueError("local_indptr must have shape (j_count+1,)")
    if indices.shape != pq_ids.shape or indices.shape != epq_data.shape:
        raise ValueError("indices, pq_ids, epq_data must share shape (tile_nnz,)")

    task_g = cp.asarray(task_g, dtype=fp_dtype)
    task_g = cp.ascontiguousarray(task_g)
    nops = int(drt.norb) * int(drt.norb)
    if task_g.ndim == 1:
        if task_g.shape != (nops,):
            raise ValueError("task_g (1D) must have shape (norb*norb,)")
    elif task_g.ndim == 2:
        if task_g.shape != (int(j_count), nops):
            raise ValueError("task_g (2D) must have shape (j_count,norb*norb)")
    else:
        raise ValueError("task_g must be 1D or 2D")

    if task_scale is not None:
        task_scale = cp.ascontiguousarray(cp.asarray(task_scale, dtype=fp_dtype).ravel())
        if task_scale.shape != (int(j_count),):
            raise ValueError("task_scale must have shape (j_count,)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.ascontiguousarray(cp.asarray(y, dtype=fp_dtype).ravel())
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.ascontiguousarray(cp.asarray(overflow, dtype=cp.int32).ravel())
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.apply_g_flat_scatter_atomic_epq_table_tile_inplace_device(
        drt_dev,
        state_dev,
        local_indptr,
        indices,
        pq_ids,
        epq_data,
        task_g,
        task_scale if task_scale is not None else None,
        int(j_start),
        int(j_count),
        y,
        overflow,
        int(threads),
        bool(zero_y),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        bool(use_kahan),
    )
    return y, overflow


def apply_g_flat_task_sums_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf,
    task_g,
    *,
    task_scale=None,
    out_sum=None,
    overflow=None,
    threads: int = 32,
    zero_out: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Compute-only Kernel1A variant: sums all generated contributions per task.

    This is intended for profiling/benchmarking the segment-walk cost without the `y[i]` atomic scatters.
    All array-like inputs/outputs must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
    Returns `(out_sum, overflow)` where both are device arrays.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array matvec path") from e

    task_csf = cp.asarray(task_csf, dtype=cp.int32).ravel()
    task_g = cp.asarray(task_g, dtype=cp.float64)
    task_g = cp.ascontiguousarray(task_g)
    if task_g.ndim not in (1, 2):
        raise ValueError("task_g must be 1D (norb*norb,) or 2D (ntasks,norb*norb)")

    if task_scale is not None:
        task_scale = cp.asarray(task_scale, dtype=cp.float64).ravel()
        task_scale = cp.ascontiguousarray(task_scale)
        if task_scale.shape != task_csf.shape:
            raise ValueError("task_scale must have shape (ntasks,)")

    if out_sum is None:
        out_sum = cp.empty((int(task_csf.size),), dtype=cp.float64)
    else:
        out_sum = cp.asarray(out_sum, dtype=cp.float64).ravel()
        out_sum = cp.ascontiguousarray(out_sum)
        if out_sum.shape != (int(task_csf.size),):
            raise ValueError("out_sum must have shape (ntasks,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.apply_g_flat_task_sums_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_g,
        task_scale if task_scale is not None else None,
        out_sum,
        overflow,
        int(threads),
        bool(zero_out),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )

    return out_sum, overflow


def build_t_block_epq_atomic_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    c,
    p_list,
    q_list,
    *,
    out=None,
    overflow=None,
    threads: int = 256,
    zero_out: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Build `T_block[op, i] = (E_{p_op q_op} |c>)[i]` on the GPU (FP64 atomics).

    All array-like inputs/outputs must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
    Returns `(out, overflow)` where both are device arrays (so the caller can reuse them across calls).
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array RDM path") from e

    c = cp.asarray(c, dtype=cp.float64).ravel()
    c = cp.ascontiguousarray(c)
    if c.shape != (int(drt.ncsf),):
        raise ValueError("c must have shape (ncsf,)")

    p_list = cp.asarray(p_list, dtype=cp.int32).ravel()
    p_list = cp.ascontiguousarray(p_list)
    q_list = cp.asarray(q_list, dtype=cp.int32).ravel()
    q_list = cp.ascontiguousarray(q_list)
    if p_list.shape != q_list.shape:
        raise ValueError("p_list and q_list must have the same shape")
    nops_block = int(p_list.size)

    if out is None:
        out = cp.empty((nops_block, int(drt.ncsf)), dtype=cp.float64)
    else:
        out = cp.asarray(out, dtype=cp.float64)
        out = cp.ascontiguousarray(out)
        if out.shape != (nops_block, int(drt.ncsf)):
            raise ValueError("out must have shape (len(p_list), ncsf)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_t_block_epq_atomic_inplace_device(
        drt_dev,
        state_dev,
        c,
        p_list,
        q_list,
        out,
        overflow,
        int(threads),
        bool(zero_out),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )

    return out, overflow


def rdm_gram_and_dm1_inplace_device(
    ws,
    t,
    c,
    *,
    dm1_out=None,
    gram_out=None,
    stream=None,
    sync: bool = True,
):
    """Compute dm1_pq and Gram0 from device-resident T using cuBLAS.

    Parameters
    ----------
    ws
        `_guga_cuda_ext.RDMGramWorkspace` object.
    t
        Device array `float64` with shape `(nops,ncsf)` (C-contiguous rows).
    c
        Device array `float64` with shape `(ncsf,)`.
    dm1_out
        Optional device output array `float64` with shape `(nops,)`.
    gram_out
        Optional device output array `float64` with shape `(nops,nops)`.

    Returns
    -------
    (dm1_out, gram_out)
        Both are device arrays.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array RDM path") from e

    t = cp.asarray(t, dtype=cp.float64)
    t = cp.ascontiguousarray(t)
    if t.ndim != 2:
        raise ValueError("t must be 2D (nops,ncsf)")

    c = cp.asarray(c, dtype=cp.float64).ravel()
    c = cp.ascontiguousarray(c)
    if c.ndim != 1:
        raise ValueError("c must be 1D (ncsf,)")

    nops = int(t.shape[0])
    if dm1_out is None:
        dm1_out = cp.empty((nops,), dtype=cp.float64)
    else:
        dm1_out = cp.asarray(dm1_out, dtype=cp.float64).ravel()
        dm1_out = cp.ascontiguousarray(dm1_out)
        if dm1_out.shape != (nops,):
            raise ValueError("dm1_out must have shape (nops,)")

    if gram_out is None:
        gram_out = cp.empty((nops, nops), dtype=cp.float64)
    else:
        gram_out = cp.asarray(gram_out, dtype=cp.float64)
        gram_out = cp.ascontiguousarray(gram_out)
        if gram_out.shape != (nops, nops):
            raise ValueError("gram_out must have shape (nops,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    ws.compute(t, c, dm1_out, gram_out, int(stream_ptr), bool(sync))
    return dm1_out, gram_out


def rdm_gram_and_dm1_csf_major_inplace_device(
    ws,
    t,
    c,
    *,
    dm1_out=None,
    gram_out=None,
    stream=None,
    sync: bool = True,
    accumulate: bool = False,
):
    """Compute dm1_pq and Gram0 from CSF-major device-resident T using cuBLAS.

    Parameters
    ----------
    ws
        `_guga_cuda_ext.RDMGramWorkspace` object.
    t
        Device array `float64` with shape `(ncsf,nops)` (C-contiguous rows).
    c
        Device array `float64` with shape `(ncsf,)`.
    dm1_out
        Optional device output array `float64` with shape `(nops,)`.
    gram_out
        Optional device output array `float64` with shape `(nops,nops)`.
    accumulate
        If True, accumulate into dm1_out and gram_out (beta=1) instead of overwriting (beta=0).
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array RDM path") from e

    t = cp.asarray(t, dtype=cp.float64)
    t = cp.ascontiguousarray(t)
    if t.ndim != 2:
        raise ValueError("t must be 2D (ncsf,nops)")

    c = cp.asarray(c, dtype=cp.float64).ravel()
    c = cp.ascontiguousarray(c)
    if c.ndim != 1:
        raise ValueError("c must be 1D (ncsf,)")

    nops = int(t.shape[1])
    if dm1_out is None:
        dm1_out = cp.empty((nops,), dtype=cp.float64)
    else:
        dm1_out = cp.asarray(dm1_out, dtype=cp.float64).ravel()
        dm1_out = cp.ascontiguousarray(dm1_out)
        if dm1_out.shape != (nops,):
            raise ValueError("dm1_out must have shape (nops,)")

    if gram_out is None:
        gram_out = cp.empty((nops, nops), dtype=cp.float64)
    else:
        gram_out = cp.asarray(gram_out, dtype=cp.float64)
        gram_out = cp.ascontiguousarray(gram_out)
        if gram_out.shape != (nops, nops):
            raise ValueError("gram_out must have shape (nops,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    ws.compute_csf_major(t, c, dm1_out, gram_out, int(stream_ptr), bool(sync), bool(accumulate))
    return dm1_out, gram_out


def rdm_cross_gram_and_dm1_inplace_device(
    ws,
    t_bra,
    t_ket,
    c_bra,
    *,
    dm1_out=None,
    gram_out=None,
    stream=None,
    sync: bool = True,
):
    """Compute transition dm1_pq and cross Gram0 from device-resident T via cuBLAS.

    Parameters
    ----------
    ws
        `_guga_cuda_ext.RDMGramWorkspace` object.
    t_bra
        Device array `float64` with shape `(nops,ncsf)` for the bra vector.
    t_ket
        Device array `float64` with shape `(nops,ncsf)` for the ket vector.
    c_bra
        Device array `float64` with shape `(ncsf,)`.
    dm1_out
        Optional device output array `float64` with shape `(nops,)`.
    gram_out
        Optional device output array `float64` with shape `(nops,nops)`.

    Returns
    -------
    (dm1_out, gram_out)
        Both are device arrays.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array RDM path") from e

    t_bra = cp.asarray(t_bra, dtype=cp.float64)
    t_bra = cp.ascontiguousarray(t_bra)
    t_ket = cp.asarray(t_ket, dtype=cp.float64)
    t_ket = cp.ascontiguousarray(t_ket)
    if t_bra.ndim != 2 or t_ket.ndim != 2:
        raise ValueError("t_bra/t_ket must be 2D (nops,ncsf)")
    if t_bra.shape != t_ket.shape:
        raise ValueError("t_bra and t_ket must have the same shape")

    c_bra = cp.asarray(c_bra, dtype=cp.float64).ravel()
    c_bra = cp.ascontiguousarray(c_bra)

    nops = int(t_ket.shape[0])
    if dm1_out is None:
        dm1_out = cp.empty((nops,), dtype=cp.float64)
    else:
        dm1_out = cp.asarray(dm1_out, dtype=cp.float64).ravel()
        dm1_out = cp.ascontiguousarray(dm1_out)
        if dm1_out.shape != (nops,):
            raise ValueError("dm1_out must have shape (nops,)")

    if gram_out is None:
        gram_out = cp.empty((nops, nops), dtype=cp.float64)
    else:
        gram_out = cp.asarray(gram_out, dtype=cp.float64)
        gram_out = cp.ascontiguousarray(gram_out)
        if gram_out.shape != (nops, nops):
            raise ValueError("gram_out must have shape (nops,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    ws.compute_cross(t_bra, t_ket, c_bra, dm1_out, gram_out, int(stream_ptr), bool(sync))
    return dm1_out, gram_out


def rdm_cross_gram_and_dm1_csf_major_inplace_device(
    ws,
    t_bra,
    t_ket,
    c_bra,
    *,
    dm1_out=None,
    gram_out=None,
    stream=None,
    sync: bool = True,
    accumulate: bool = False,
):
    """Compute transition dm1_pq and cross Gram0 from CSF-major device-resident T via cuBLAS.

    Parameters
    ----------
    ws
        `_guga_cuda_ext.RDMGramWorkspace` object.
    t_bra
        Device array `float64` with shape `(ncsf,nops)` for the bra vector.
    t_ket
        Device array `float64` with shape `(ncsf,nops)` for the ket vector.
    c_bra
        Device array `float64` with shape `(ncsf,)`.
    dm1_out
        Optional device output array `float64` with shape `(nops,)`.
    gram_out
        Optional device output array `float64` with shape `(nops,nops)`.
    accumulate
        If True, accumulate into dm1_out and gram_out (beta=1) instead of overwriting (beta=0).
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array RDM path") from e

    t_bra = cp.asarray(t_bra, dtype=cp.float64)
    t_bra = cp.ascontiguousarray(t_bra)
    t_ket = cp.asarray(t_ket, dtype=cp.float64)
    t_ket = cp.ascontiguousarray(t_ket)
    if t_bra.ndim != 2 or t_ket.ndim != 2:
        raise ValueError("t_bra/t_ket must be 2D (ncsf,nops)")
    if t_bra.shape != t_ket.shape:
        raise ValueError("t_bra and t_ket must have the same shape")

    c_bra = cp.asarray(c_bra, dtype=cp.float64).ravel()
    c_bra = cp.ascontiguousarray(c_bra)

    nops = int(t_ket.shape[1])
    if dm1_out is None:
        dm1_out = cp.empty((nops,), dtype=cp.float64)
    else:
        dm1_out = cp.asarray(dm1_out, dtype=cp.float64).ravel()
        dm1_out = cp.ascontiguousarray(dm1_out)
        if dm1_out.shape != (nops,):
            raise ValueError("dm1_out must have shape (nops,)")

    if gram_out is None:
        gram_out = cp.empty((nops, nops), dtype=cp.float64)
    else:
        gram_out = cp.asarray(gram_out, dtype=cp.float64)
        gram_out = cp.ascontiguousarray(gram_out)
        if gram_out.shape != (nops, nops):
            raise ValueError("gram_out must have shape (nops,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    ws.compute_cross_csf_major(t_bra, t_ket, c_bra, dm1_out, gram_out, int(stream_ptr), bool(sync), bool(accumulate))
    return dm1_out, gram_out


def rs_enum_allpairs_deterministic(
    drt: DRT,
    drt_dev,
    state_dev,
    j_list: np.ndarray,
    *,
    threads: int = 128,
    max_total_out: int = -1,
    coalesce: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic baseline for Kernel 2B: enumerate k = E_rs |j> for all r!=s.

    Returns COO-style arrays (j_out, k_out, rs_id_out, c_rs_out) plus task offsets.
    """

    j_list = np.asarray(j_list, dtype=np.int32).ravel()
    norb = int(drt.norb)
    if norb <= 0:
        raise ValueError("invalid norb")
    if j_list.size == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_v = np.zeros(0, dtype=np.float64)
        empty_off = np.zeros(1, dtype=np.int64)
        return empty_i, empty_i, empty_i, empty_v, empty_off

    # All off-diagonal (r,s) pairs once, then tile per j.
    r_all = np.repeat(np.arange(norb, dtype=np.int32), norb)
    s_all = np.tile(np.arange(norb, dtype=np.int32), norb)
    mask = r_all != s_all
    r_all = r_all[mask]
    s_all = s_all[mask]
    n_pairs = int(r_all.size)

    task_csf = np.repeat(j_list, n_pairs).astype(np.int32, copy=False)
    task_r = np.tile(r_all, int(j_list.size)).astype(np.int32, copy=False)
    task_s = np.tile(s_all, int(j_list.size)).astype(np.int32, copy=False)

    k_out, c_out, offsets = epq_contribs_many_deterministic(
        drt,
        drt_dev,
        state_dev,
        task_csf,
        task_r,
        task_s,
        threads=int(threads),
        max_total_out=int(max_total_out),
    )

    counts = np.diff(offsets).astype(np.int32, copy=False)
    if int(k_out.size) != int(np.sum(counts, dtype=np.int64)):
        raise RuntimeError("internal error: offsets do not sum to output length")

    j_out = np.repeat(task_csf, counts)
    rs_id = (task_r.astype(np.int64) * norb + task_s.astype(np.int64)).astype(np.int32, copy=False)
    rs_out = np.repeat(rs_id, counts)

    if not coalesce or k_out.size == 0:
        return (
            np.asarray(j_out, dtype=np.int32),
            np.asarray(k_out, dtype=np.int32),
            np.asarray(rs_out, dtype=np.int32),
            np.asarray(c_out, dtype=np.float64),
            np.asarray(offsets, dtype=np.int64),
        )

    # Coalesce duplicates in (j,k,rs_id) by summing coefficients (CPU, deterministic).
    order = np.lexsort((rs_out, k_out, j_out))
    j_s = np.asarray(j_out[order], dtype=np.int32)
    k_s = np.asarray(k_out[order], dtype=np.int32)
    rs_s = np.asarray(rs_out[order], dtype=np.int32)
    c_s = np.asarray(c_out[order], dtype=np.float64)

    if j_s.size <= 1:
        return j_s, k_s, rs_s, c_s, np.asarray(offsets, dtype=np.int64)

    change = (j_s[1:] != j_s[:-1]) | (k_s[1:] != k_s[:-1]) | (rs_s[1:] != rs_s[:-1])
    if np.any(change):
        starts = np.concatenate(([0], np.nonzero(change)[0] + 1)).astype(np.int32, copy=False)
        j_s = j_s[starts]
        k_s = k_s[starts]
        rs_s = rs_s[starts]
        c_s = np.add.reduceat(c_s, starts)
    return j_s, k_s, rs_s, c_s, np.asarray(offsets, dtype=np.int64)


def kernel25_build_csr_from_triples(
    j_out: np.ndarray,
    k_out: np.ndarray,
    rs_out: np.ndarray,
    c_out: np.ndarray,
    *,
    nops: int,
    coalesce: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kernel 2.5 (host baseline): coalesce and build CSR C[row,rs_id] grouped by (j,k).

    Returns
    -------
    row_j, row_k, indptr, indices, data
        CSR representation of C with rows keyed by unique (j,k) pairs (sorted),
        and columns in rs_id space (0..nops-1).
    """

    j_out = np.asarray(j_out, dtype=np.int32).ravel()
    k_out = np.asarray(k_out, dtype=np.int32).ravel()
    rs_out = np.asarray(rs_out, dtype=np.int32).ravel()
    c_out = np.asarray(c_out, dtype=np.float64).ravel()
    if j_out.shape != k_out.shape or j_out.shape != rs_out.shape or j_out.shape != c_out.shape:
        raise ValueError("j_out/k_out/rs_out/c_out must have the same shape")

    nops = int(nops)
    if nops <= 0:
        raise ValueError("nops must be > 0")

    nnz = int(j_out.size)
    if nnz == 0:
        row_j = np.zeros(0, dtype=np.int32)
        row_k = np.zeros(0, dtype=np.int32)
        indptr = np.zeros(1, dtype=np.int64)
        indices = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float64)
        return row_j, row_k, indptr, indices, data

    # Sort by (j,k,rs) for deterministic grouping and optional coalescing.
    order = np.lexsort((rs_out, k_out, j_out))
    j_s = j_out[order]
    k_s = k_out[order]
    rs_s = rs_out[order]
    c_s = c_out[order]

    if coalesce and nnz > 1:
        change = (j_s[1:] != j_s[:-1]) | (k_s[1:] != k_s[:-1]) | (rs_s[1:] != rs_s[:-1])
        if np.any(change):
            starts = np.concatenate(([0], np.nonzero(change)[0] + 1)).astype(np.int32, copy=False)
            j_s = j_s[starts]
            k_s = k_s[starts]
            rs_s = rs_s[starts]
            c_s = np.add.reduceat(c_s, starts)

    if np.any(rs_s < 0) or np.any(rs_s >= nops):
        raise ValueError("rs_out contains out-of-range column indices for nops")

    # Build CSR rows keyed by (j,k).
    jk = (j_s.astype(np.uint64) << 32) | k_s.astype(np.uint32)
    row_starts = np.concatenate(([0], np.nonzero(jk[1:] != jk[:-1])[0] + 1)).astype(np.int64, copy=False)
    nrows = int(row_starts.size)
    indptr = np.concatenate((row_starts, np.asarray([j_s.size], dtype=np.int64)))

    row_j = np.asarray(j_s[row_starts], dtype=np.int32)
    row_k = np.asarray(k_s[row_starts], dtype=np.int32)
    indices = np.asarray(rs_s, dtype=np.int32)
    data = np.asarray(c_s, dtype=np.float64)
    return row_j, row_k, indptr, indices, data


def kernel25_build_csr_from_triples_cuda(
    j_out: np.ndarray,
    k_out: np.ndarray,
    rs_out: np.ndarray,
    c_out: np.ndarray,
    *,
    nops: int,
    coalesce: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kernel 2.5 (CUDA baseline): build CSR C[row,rs_id] grouped by (j,k) on the GPU.

    Notes
    -----
    - Deterministic ordering is enforced by sorting by (j,k,rs_id).
    - Coalescing duplicates sums coefficients on the GPU; due to parallel reduction this is not guaranteed to be
      bitwise-reproducible, but should be numerically stable for validation with tolerances.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    j_out = np.asarray(j_out, dtype=np.int32).ravel()
    k_out = np.asarray(k_out, dtype=np.int32).ravel()
    rs_out = np.asarray(rs_out, dtype=np.int32).ravel()
    c_out = np.asarray(c_out, dtype=np.float64).ravel()
    if j_out.shape != k_out.shape or j_out.shape != rs_out.shape or j_out.shape != c_out.shape:
        raise ValueError("j_out/k_out/rs_out/c_out must have the same shape")

    nops = int(nops)
    if nops <= 0:
        raise ValueError("nops must be > 0")

    row_j, row_k, indptr, indices, data = _ext.kernel25_build_csr_from_triples_cuda(
        j_out, k_out, rs_out, c_out, int(nops), bool(coalesce)
    )
    return (
        np.asarray(row_j, dtype=np.int32),
        np.asarray(row_k, dtype=np.int32),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
    )


def kernel25_build_csr_from_tasks_deterministic_cuda(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf: np.ndarray,
    task_p: np.ndarray,
    task_q: np.ndarray,
    *,
    threads: int = 128,
    max_total_out: int = -1,
    coalesce: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kernel 2B + 2.5 fused baseline: enumerate `E_pq|j>` and build CSR `C[(j,k), pq_id]` on GPU.

    This avoids materializing `(j_out,k_out,pq_id_out,c_out)` on the host.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    task_csf = np.asarray(task_csf, dtype=np.int32).ravel()
    task_p = np.asarray(task_p, dtype=np.int32).ravel()
    task_q = np.asarray(task_q, dtype=np.int32).ravel()
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")

    row_j, row_k, indptr, indices, data = _ext.kernel25_build_csr_from_tasks_deterministic_cuda(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        int(threads),
        int(max_total_out),
        bool(coalesce),
    )
    return (
        np.asarray(row_j, dtype=np.int32),
        np.asarray(row_k, dtype=np.int32),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
    )


def kernel25_build_csr_from_tasks_deterministic_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    task_csf,
    task_p,
    task_q,
    *,
    capacity: int | None = None,
    row_j=None,
    row_k=None,
    indptr=None,
    indices=None,
    data=None,
    overflow=None,
    threads: int = 128,
    coalesce: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Kernel 2B + 2.5: build CSR on the GPU into caller-provided device buffers.

    This path avoids copying the CSR back to host. The caller provides preallocated output buffers with a chosen
    `capacity` (upper bound for `nnz_in`, `nnz_out`, and `nrows`), and the function returns `(nrows, nnz, nnz_in)`.

    Notes
    -----
    - All arrays must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
    - Output buffer shapes must be:
      - `row_j, row_k, indices, data`: (capacity,)
      - `indptr`: (capacity+1,)
      - `overflow`: (1,)
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device CSR builder path") from e

    task_csf = cp.asarray(task_csf, dtype=cp.int32).ravel()
    task_csf = cp.ascontiguousarray(task_csf)
    task_p = cp.asarray(task_p, dtype=cp.int32).ravel()
    task_p = cp.ascontiguousarray(task_p)
    task_q = cp.asarray(task_q, dtype=cp.int32).ravel()
    task_q = cp.ascontiguousarray(task_q)
    if task_p.shape != task_csf.shape or task_q.shape != task_csf.shape:
        raise ValueError("task_csf/task_p/task_q must have the same shape")

    ntasks = int(task_csf.size)
    if ntasks == 0:
        if capacity is None:
            capacity = 1
        if row_j is None:
            row_j = cp.empty((int(capacity),), dtype=cp.int32)
        if row_k is None:
            row_k = cp.empty((int(capacity),), dtype=cp.int32)
        if indptr is None:
            indptr = cp.empty((int(capacity) + 1,), dtype=cp.int64)
        if indices is None:
            indices = cp.empty((int(capacity),), dtype=cp.int32)
        if data is None:
            data = cp.empty((int(capacity),), dtype=cp.float64)
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)
        return row_j[:0], row_k[:0], indptr[:1], indices[:0], data[:0], overflow, 0, 0, 0

    if capacity is None:
        capacity = max(1, 2 * ntasks)
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")

    if row_j is None:
        row_j = cp.empty((capacity,), dtype=cp.int32)
    else:
        row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
        row_j = cp.ascontiguousarray(row_j)

    if row_k is None:
        row_k = cp.empty((capacity,), dtype=cp.int32)
    else:
        row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
        row_k = cp.ascontiguousarray(row_k)

    if indptr is None:
        indptr = cp.empty((capacity + 1,), dtype=cp.int64)
    else:
        indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
        indptr = cp.ascontiguousarray(indptr)

    if indices is None:
        indices = cp.empty((capacity,), dtype=cp.int32)
    else:
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        indices = cp.ascontiguousarray(indices)

    if data is None:
        data = cp.empty((capacity,), dtype=cp.float64)
    else:
        data = cp.asarray(data).ravel()
        if data.dtype not in (cp.float32, cp.float64):
            data = cp.asarray(data, dtype=cp.float64).ravel()
        data = cp.ascontiguousarray(data)

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    nrows, nnz, nnz_in = _ext.kernel25_build_csr_from_tasks_deterministic_inplace_device(
        drt_dev,
        state_dev,
        task_csf,
        task_p,
        task_q,
        row_j,
        row_k,
        indptr,
        indices,
        data,
        overflow,
        int(threads),
        bool(coalesce),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )

    nrows = int(nrows)
    nnz = int(nnz)
    nnz_in = int(nnz_in)

    return (
        row_j[:nrows],
        row_k[:nrows],
        indptr[: nrows + 1],
        indices[:nnz],
        data[:nnz],
        overflow,
        nrows,
        nnz,
        nnz_in,
    )


def kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    j_start: int,
    j_count: int,
    *,
    capacity: int | None = None,
    row_j=None,
    row_k=None,
    indptr=None,
    indices=None,
    data=None,
    overflow=None,
    threads: int = 128,
    coalesce: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Kernel 2B (implicit all r!=s) + 2.5: build CSR on GPU into caller-provided device buffers.

    This variant avoids materializing `(task_csf, task_r, task_s)` arrays; instead it enumerates all off-diagonal
    (r,s) pairs implicitly for a contiguous j-block `[j_start, j_start+j_count)`.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device CSR builder path") from e

    j_start = int(j_start)
    j_count = int(j_count)
    if j_start < 0 or j_count < 0:
        raise ValueError("j_start/j_count must be >= 0")
    if j_count == 0:
        if capacity is None:
            capacity = 1
        if row_j is None:
            row_j = cp.empty((int(capacity),), dtype=cp.int32)
        if row_k is None:
            row_k = cp.empty((int(capacity),), dtype=cp.int32)
        if indptr is None:
            indptr = cp.empty((int(capacity) + 1,), dtype=cp.int64)
        if indices is None:
            indices = cp.empty((int(capacity),), dtype=cp.int32)
        if data is None:
            data = cp.empty((int(capacity),), dtype=cp.float64)
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)
        return row_j[:0], row_k[:0], indptr[:1], indices[:0], data[:0], overflow, 0, 0, 0

    if capacity is None:
        norb = int(drt.norb)
        n_pairs = max(0, norb * (norb - 1))
        ntasks = j_count * n_pairs
        capacity = max(1, 2 * int(ntasks))
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")

    if row_j is None:
        row_j = cp.empty((capacity,), dtype=cp.int32)
    else:
        row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
        row_j = cp.ascontiguousarray(row_j)

    if row_k is None:
        row_k = cp.empty((capacity,), dtype=cp.int32)
    else:
        row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
        row_k = cp.ascontiguousarray(row_k)

    if indptr is None:
        indptr = cp.empty((capacity + 1,), dtype=cp.int64)
    else:
        indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
        indptr = cp.ascontiguousarray(indptr)

    if indices is None:
        indices = cp.empty((capacity,), dtype=cp.int32)
    else:
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        indices = cp.ascontiguousarray(indices)

    if data is None:
        data = cp.empty((capacity,), dtype=cp.float64)
    else:
        data = cp.asarray(data, dtype=cp.float64).ravel()
        data = cp.ascontiguousarray(data)

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    nrows, nnz, nnz_in = _ext.kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device(
        drt_dev,
        state_dev,
        int(j_start),
        int(j_count),
        row_j,
        row_k,
        indptr,
        indices,
        data,
        overflow,
        int(threads),
        bool(coalesce),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
    )

    nrows = int(nrows)
    nnz = int(nnz)
    nnz_in = int(nnz_in)

    return (
        row_j[:nrows],
        row_k[:nrows],
        indptr[: nrows + 1],
        indices[:nnz],
        data[:nnz],
        overflow,
        nrows,
        nnz,
        nnz_in,
    )


def kernel25_rs_enum_allpairs_to_csr_deterministic_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    j_list: np.ndarray,
    *,
    threads: int = 128,
    coalesce: bool = True,
    include_diagonal: bool = False,
    capacity: int | None = None,
    max_retries: int = 2,
):
    """Kernel 2B (all r,s) + Kernel 2.5 fused: build CSR on the GPU (device outputs).

    Returns device arrays `(row_j, row_k, indptr, indices, data, overflow, nrows, nnz, nnz_in)`.

    Notes
    -----
    By default, this enumerates only off-diagonal generator actions (`r!=s`) to match the original baseline path.
    Set `include_diagonal=True` to include `r==s` terms (diagonal number operators), which is required for a full
    two-body matvec.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device CSR builder path") from e

    j_list = np.asarray(j_list, dtype=np.int32).ravel()
    norb = int(drt.norb)
    if norb <= 0:
        raise ValueError("invalid norb")
    if j_list.size == 0:
        row_j = cp.zeros((0,), dtype=cp.int32)
        row_k = cp.zeros((0,), dtype=cp.int32)
        indptr = cp.zeros((1,), dtype=cp.int64)
        indices = cp.zeros((0,), dtype=cp.int32)
        data = cp.zeros((0,), dtype=cp.float64)
        overflow = cp.zeros((1,), dtype=cp.int32)
        return row_j, row_k, indptr, indices, data, overflow, 0, 0, 0

    r_all = np.repeat(np.arange(norb, dtype=np.int32), norb)
    s_all = np.tile(np.arange(norb, dtype=np.int32), norb)
    if not bool(include_diagonal):
        mask = r_all != s_all
        r_all = r_all[mask]
        s_all = s_all[mask]
    n_pairs = int(r_all.size)

    j_d = cp.asarray(j_list, dtype=cp.int32)
    r_d = cp.asarray(r_all, dtype=cp.int32)
    s_d = cp.asarray(s_all, dtype=cp.int32)

    task_csf = cp.repeat(j_d, n_pairs)
    task_r = cp.tile(r_d, int(j_d.size))
    task_s = cp.tile(s_d, int(j_d.size))

    ntasks = int(task_csf.size)
    if capacity is None:
        capacity = max(1, 2 * ntasks)
    cap = int(capacity)
    last_err = None
    for _ in range(max(1, int(max_retries))):
        try:
            return kernel25_build_csr_from_tasks_deterministic_inplace_device(
                drt,
                drt_dev,
                state_dev,
                task_csf,
                task_r,
                task_s,
                capacity=cap,
                threads=int(threads),
                coalesce=bool(coalesce),
            )
        except RuntimeError as e:
            last_err = e
            if "exceeds output buffer capacity" in str(e):
                cap *= 2
                continue
            raise
    assert last_err is not None
    raise last_err


def kernel25_rs_enum_allpairs_to_csr_deterministic_cuda(
    drt: DRT,
    drt_dev,
    state_dev,
    j_list: np.ndarray,
    *,
    threads: int = 128,
    max_total_out: int = -1,
    coalesce: bool = True,
    include_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kernel 2B (all r,s) + Kernel 2.5 fused: build CSR C[row=(j,k), rs_id] for a batch of `j`.

    Returns `(row_j,row_k,indptr,indices,data)` with `indices` in `rs_id = r*norb+s`.

    Notes
    -----
    By default, this enumerates only off-diagonal generator actions (`r!=s`) to match the original baseline path.
    Set `include_diagonal=True` to include `r==s` terms (diagonal number operators), which is required for a full
    two-body matvec.
    """

    j_list = np.asarray(j_list, dtype=np.int32).ravel()
    norb = int(drt.norb)
    if norb <= 0:
        raise ValueError("invalid norb")
    if j_list.size == 0:
        row_j = np.zeros(0, dtype=np.int32)
        row_k = np.zeros(0, dtype=np.int32)
        indptr = np.zeros(1, dtype=np.int64)
        indices = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float64)
        return row_j, row_k, indptr, indices, data

    r_all = np.repeat(np.arange(norb, dtype=np.int32), norb)
    s_all = np.tile(np.arange(norb, dtype=np.int32), norb)
    if not bool(include_diagonal):
        mask = r_all != s_all
        r_all = r_all[mask]
        s_all = s_all[mask]
    n_pairs = int(r_all.size)

    task_csf = np.repeat(j_list, n_pairs).astype(np.int32, copy=False)
    task_r = np.tile(r_all, int(j_list.size)).astype(np.int32, copy=False)
    task_s = np.tile(s_all, int(j_list.size)).astype(np.int32, copy=False)

    return kernel25_build_csr_from_tasks_deterministic_cuda(
        drt,
        drt_dev,
        state_dev,
        task_csf,
        task_r,
        task_s,
        threads=int(threads),
        max_total_out=int(max_total_out),
        coalesce=bool(coalesce),
    )


class GugaMatvecEriMatWorkspace:
    """GPU-resident matvec workspace for the ERI_mat two-body path.

    This is a Python-side workspace that caches device-resident state tables and common buffers to reduce
    per-matvec allocations. It is intended as a stepping stone toward a fully C++/CUDA-resident pipeline.
    """

    def __init__(
        self,
        drt: DRT,
        *,
        drt_dev=None,
        state_dev=None,
        state_cache: DRTStateCache | None = None,
        eri_mat=None,
        l_full=None,
        direct_op=None,
        h_eff=None,
        j_tile: int = 1024,
        csr_capacity_mult: float = 2.0,
        cache_csr_tiles: bool | str = "auto",
        check_overflow_first_tile_only: bool = True,
        check_overflow_mode: str | int = "deferred",
        fuse_count_write: bool = False,
        fp32_coeff_data: bool = False,
        threads_enum: int = 128,
        threads_g: int = 256,
        threads_w: int = 256,
        threads_apply: int = 32,
        max_g_bytes: int = 256 * 1024 * 1024,
        coalesce: bool | str = "auto",
        include_diagonal_rs: bool = True,
        path_mode: str = "auto",
        use_fused_hop: bool = True,
        use_epq_table: bool = False,
        aggregate_offdiag_k: bool = False,
        offdiag_enable_fp64_emulation: bool = False,
        offdiag_emulation_strategy: str = "performant",
        offdiag_cublas_workspace_cap_mb: int = 2048,
        gemm_backend: str = "gemmex_fp64",
        gemm_k_pad_align: int = 64,
        dtype=None,
        epq_build_nthreads: int = 0,
        epq_build_device: bool = False,
        epq_build_j_tile: int = 0,
        epq_indptr_dtype: str | None = "auto",
        epq_blocked_transpose: bool | str | None = "auto",
        epq_blocked_transpose_reserve_mib: int = 512,
        epq_streaming: bool | None = None,
        epq_stream_j_tile: int = 0,
        epq_stream_pq_block: int = 0,
        epq_stream_double_buffer: bool | str = "auto",
        epq_stream_panic_mode: bool | str = "auto",
        epq_stream_use_recompute: bool | str | None = None,
        epq_recompute_warp_coop: bool | None = None,
        apply_mode: str = "auto",
        apply_warp_coop: bool | str | None = None,
        use_cuda_graph: bool = False,
        cuda_graph_mode: int | None = None,
        fp32_csr_cache: bool | str = "auto",
        csr_host_cache: bool | str = "auto",
        csr_host_cache_budget_gib: float = 4.0,
        csr_host_cache_min_ncsf: int = 1_000_000,
        csr_pipeline_streams: int | str = "auto",
        csr_pipeline_min_ncsf: int = 1_000_000,
        prefilter_trivial_tasks: bool | str = "auto",
        prefilter_trivial_tasks_min_ncsf: int = 1_000_000,
        skip_zero_x_tiles: bool | str = "auto",
        skip_zero_x_tiles_min_ncsf: int = 100_000,
        kahan_compensation: bool | str = "auto",
        epq_apply_cache: bool | str = "auto",
        epq_apply_cache_budget_gib: float = 4.0,
    ) -> None:
        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the CUDA matvec workspace") from e

        self.drt = drt
        self.norb = int(drt.norb)
        self.ncsf = int(drt.ncsf)
        self.nops = int(self.norb) * int(self.norb)
        if self.norb <= 0:
            raise ValueError("invalid norb")
        if self.ncsf <= 0:
            raise ValueError("invalid ncsf")

        self._dtype = cp.dtype(cp.float64 if dtype is None else dtype)
        if self._dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
            raise ValueError("dtype must be float32 or float64")
        self._itemsize = int(self._dtype.itemsize)

        self.threads_enum = int(threads_enum)
        self.threads_g = int(threads_g)
        self.threads_w = int(threads_w)
        self.threads_apply = int(threads_apply)
        self.max_g_bytes = int(max_g_bytes)
        if isinstance(coalesce, str):
            _coalesce_mode = coalesce.strip().lower()
            if _coalesce_mode in ("", "auto"):
                self.coalesce_mode = "auto"
                # Coalesce disabled: the GUGA oracle produces unique (j,k,rs) triples
                # (nnz_in == nnz_out), so ReduceByKey is a no-op. Disabling enables
                # Phase 4 rowkey-only sort (fewer radix bits, int32 index values →
                # 1.1-1.5x faster sort) and removes the coalesce sync point.
                self.coalesce = False
            elif _coalesce_mode in ("1", "true", "yes", "on", "enabled"):
                self.coalesce_mode = "on"
                self.coalesce = True
            elif _coalesce_mode in ("0", "false", "no", "off", "disabled"):
                self.coalesce_mode = "off"
                self.coalesce = False
            else:
                raise ValueError("coalesce must be bool or one of: auto/on/off")
        else:
            self.coalesce_mode = "manual"
            self.coalesce = bool(coalesce)
        self.include_diagonal_rs = bool(include_diagonal_rs)
        self.path_mode_requested = _normalize_matvec_cuda_path_mode(path_mode)
        self.use_fused_hop = bool(use_fused_hop and direct_op is None)
        self.use_epq_table = bool(use_epq_table)
        self.aggregate_offdiag_k = bool(aggregate_offdiag_k)
        self._fused_hop_kernel_available = bool(_ext is not None and hasattr(_ext, "fused_hop_device"))
        self._fused_hop_workspace_eligible = bool(
            self.use_fused_hop
            and self._fused_hop_kernel_available
            and int(self.norb) <= 20
            and (eri_mat is not None or l_full is not None)
        )
        if direct_op is not None:
            self._fused_hop_workspace_eligible = False
        # Slow-path policy: fused_coo is marked no-go for production runs.
        self.path_mode_fallback_reason: str | None = None
        mode_eff = str(self.path_mode_requested)
        if mode_eff == "auto":
            mode_eff = "epq_blocked"
        elif mode_eff == "fused_coo":
            raise ValueError(
                "matvec_cuda_path_mode='fused_coo' is disabled (no-go path due to performance). "
                "Use 'fused_epq_hybrid' or 'epq_blocked'."
            )
        elif mode_eff == "fused_epq_hybrid":
            if not self._fused_hop_workspace_eligible:
                mode_eff = "epq_blocked"
                self.path_mode_fallback_reason = "fused_hop_not_eligible"
            elif not self.use_epq_table:
                mode_eff = "epq_blocked"
                self.path_mode_fallback_reason = "use_epq_table_disabled"
            elif not hasattr(_ext, "fused_hop_phase1_device"):
                mode_eff = "epq_blocked"
                self.path_mode_fallback_reason = "missing_fused_hop_phase1_device"
        self.path_mode = str(mode_eff)
        if self.path_mode_fallback_reason:
            import warnings

            warnings.warn(
                f"matvec_cuda_path_mode='{self.path_mode_requested}' fell back to '{self.path_mode}' "
                f"({self.path_mode_fallback_reason})",
                RuntimeWarning,
            )

        if self.path_mode in ("fused_coo", "fused_epq_hybrid"):
            if self.aggregate_offdiag_k:
                import warnings

                warnings.warn(
                    "aggregate_offdiag_k is ignored for fused matvec modes",
                    RuntimeWarning,
                )
            self.aggregate_offdiag_k = False
        if self.path_mode == "fused_coo" and self.use_epq_table:
            import warnings

            warnings.warn(
                "use_epq_table is ignored in fused_coo mode",
                RuntimeWarning,
            )
            self.use_epq_table = False
        self.offdiag_enable_fp64_emulation = bool(offdiag_enable_fp64_emulation)
        self.offdiag_emulation_strategy = str(offdiag_emulation_strategy)
        self.offdiag_cublas_workspace_cap_mb = int(offdiag_cublas_workspace_cap_mb)
        self.gemm_backend = str(gemm_backend).strip()
        if self._dtype == cp.float32 and self.gemm_backend in ("gemmex_fp64", "cublaslt_fp64"):
            # Default FP32 policy: prefer TF32 tensor-core GEMM.
            self.gemm_backend = "gemmex_tf32"
        # 10.16.8: Pad k_count to this alignment for cuBLAS tile efficiency (0 or 1 to disable).
        self.gemm_k_pad_align = max(0, int(gemm_k_pad_align))
        if self.offdiag_enable_fp64_emulation and self._dtype != cp.float64:
            raise ValueError("offdiag_enable_fp64_emulation requires dtype=float64")
        if not self.offdiag_enable_fp64_emulation:
            if self._dtype == cp.float64:
                allowed = ("gemmex_fp64", "cublaslt_fp64")
            else:
                # 10.20.4: Added gemmex_fp32_acc64 for mixed-precision GEMM support.
                allowed = ("gemmex_fp32", "gemmex_tf32", "cublaslt_fp32", "cublaslt_tf32", "gemmex_fp32_acc64")
            if self.gemm_backend not in allowed:
                raise ValueError(f"gemm_backend must be one of: {', '.join(allowed)}")
        self.epq_build_nthreads = int(epq_build_nthreads)
        self.epq_build_device = bool(epq_build_device)
        self.epq_build_j_tile = int(epq_build_j_tile)
        import os

        env_epq_indptr_dtype = str(os.getenv("ASUKA_CUGUGA_EPQ_INDPTR_DTYPE", "")).strip()
        if (epq_indptr_dtype is None or str(epq_indptr_dtype).strip() == "") and env_epq_indptr_dtype:
            epq_indptr_dtype = env_epq_indptr_dtype
        self.epq_indptr_dtype = _normalize_epq_indptr_mode(epq_indptr_dtype)

        env_epq_blocked_transpose = os.getenv("ASUKA_CUGUGA_EPQ_BLOCKED_TRANSPOSE", "")
        self.epq_blocked_transpose_mode = _resolve_epq_blocked_transpose_mode_with_env(
            epq_blocked_transpose, env_epq_blocked_transpose
        )

        env_epq_blocked_transpose_reserve_mib = os.getenv("ASUKA_CUGUGA_EPQ_BLOCKED_TRANSPOSE_RESERVE_MIB", "")
        reserve_mib = _resolve_epq_blocked_transpose_reserve_mib_with_env(
            epq_blocked_transpose_reserve_mib, env_epq_blocked_transpose_reserve_mib
        )
        self.epq_blocked_transpose_reserve_bytes = int(reserve_mib) * 1024 * 1024

        # Adaptive thread tuning for blocked aggregate path:
        # For large active spaces (ncsf > 10k), benchmarking shows threads_apply=64
        # outperforms 32 due to better warp utilization in the gather kernel.
        _auto_tune_threads = os.getenv("CUGUGA_AUTO_TUNE_THREADS", "1").strip().lower()
        if _auto_tune_threads in ("1", "true", "yes") and self.threads_apply == 32 and self.threads_w == 256:
            try:
                _ncsf_est = int(getattr(drt, "ncsf", 0))
                if _ncsf_est > 10000:
                    self.threads_apply = 64
                    self.threads_w = 128
            except Exception:
                pass

        env_epq_streaming = str(os.getenv("ASUKA_CUGUGA_EPQ_STREAMING", "")).strip().lower()
        if epq_streaming is None:
            self.epq_streaming = env_epq_streaming in ("1", "true", "yes", "on")
        else:
            self.epq_streaming = bool(epq_streaming)
        if self.path_mode in ("fused_coo", "fused_epq_hybrid") and self.epq_streaming:
            import warnings

            warnings.warn(
                "epq_streaming is ignored for fused matvec modes",
                RuntimeWarning,
            )
            self.epq_streaming = False

        env_epq_stream_j_tile = str(os.getenv("ASUKA_CUGUGA_EPQ_STREAM_J_TILE", "")).strip()
        if int(epq_stream_j_tile) <= 0 and env_epq_stream_j_tile:
            try:
                epq_stream_j_tile = int(env_epq_stream_j_tile)
            except Exception as e:
                raise ValueError("ASUKA_CUGUGA_EPQ_STREAM_J_TILE must be an integer") from e
        self.epq_stream_j_tile = int(epq_stream_j_tile)

        env_epq_stream_pq_block = str(os.getenv("ASUKA_CUGUGA_EPQ_STREAM_PQ_BLOCK", "")).strip()
        if int(epq_stream_pq_block) <= 0 and env_epq_stream_pq_block:
            try:
                epq_stream_pq_block = int(env_epq_stream_pq_block)
            except Exception as e:
                raise ValueError("ASUKA_CUGUGA_EPQ_STREAM_PQ_BLOCK must be an integer") from e
        self.epq_stream_pq_block = int(epq_stream_pq_block)
        if self.epq_stream_pq_block < 0:
            raise ValueError("epq_stream_pq_block must be >= 0")

        self.epq_stream_double_buffer_mode = _resolve_epq_stream_double_buffer_mode(
            epq_stream_double_buffer,
            os.getenv("ASUKA_CUGUGA_EPQ_STREAM_DOUBLE_BUFFER", ""),
        )

        self.epq_stream_panic_mode = _resolve_epq_stream_panic_mode(
            epq_stream_panic_mode,
            os.getenv("ASUKA_CUGUGA_EPQ_STREAM_PANIC_MODE", ""),
        )

        self.epq_stream_use_recompute = _resolve_epq_stream_use_recompute(
            epq_stream_use_recompute,
            os.getenv("ASUKA_CUGUGA_EPQ_STREAM_RECOMPUTE", ""),
        )

        if epq_recompute_warp_coop is None:
            env_warp_coop = str(os.getenv("ASUKA_CUGUGA_EPQ_RECOMPUTE_WARP_COOP", "")).strip().lower()
            self.epq_recompute_warp_coop = env_warp_coop in ("1", "true", "yes", "on")
        else:
            self.epq_recompute_warp_coop = bool(epq_recompute_warp_coop)
        self.apply_mode = str(apply_mode).strip().lower()
        if self.apply_mode not in ("auto", "scatter", "gather"):
            raise ValueError("apply_mode must be one of: 'auto', 'scatter', 'gather'")
        # Keep apply_mode="auto" for FP32 so EPQ apply can select scatter/gather via
        # runtime heuristics in apply_g_flat_scatter_atomic_inplace_device().

        # Kahan-compensated accumulation for FP32 gather/fused kernels.
        # "auto" enables Kahan for FP32 (where it matters) and disables for FP64.
        if kahan_compensation == "auto":
            self.kahan_compensation: bool = (self._dtype == cp.float32)
        else:
            self.kahan_compensation = bool(kahan_compensation)

        # 10.16.3/10.18 Option A: Warp-cooperative segment walk kernel for apply path.
        # This eliminates per-thread DFS stack spills by distributing the segment walk across warp lanes.
        self.apply_warp_coop = _resolve_apply_warp_coop(
            apply_warp_coop,
            os.getenv("ASUKA_CUGUGA_APPLY_WARP_COOP", ""),
        )

        self.epq_table_build_s = 0.0
        self.use_cuda_graph = bool(use_cuda_graph)
        # 10.20.4 Item #3: Relax FP32 CUDA Graph restriction for non-GEMM paths.
        # CUDA Graph works for FP32 when aggregate_offdiag_k=False (pure EPQ-table path).
        # The original FP64-only restriction was due to cuBLAS stream capture issues,
        # but the EPQ-table apply path uses only custom CUDA kernels.
        if self.use_cuda_graph and self._dtype != cp.float64 and bool(aggregate_offdiag_k):
            raise ValueError(
                "use_cuda_graph with dtype=float32 requires aggregate_offdiag_k=False "
                "(CUDA Graph cannot capture cuBLAS calls used in the GEMM path)"
            )
        self._cuda_graph_mode = (
            int(cp.cuda.runtime.streamCaptureModeRelaxed) if cuda_graph_mode is None else int(cuda_graph_mode)
        )
        self._cuda_graph = None
        self._cuda_graph_x = None
        self._cuda_graph_y = None

        self.check_overflow_first_tile_only = bool(check_overflow_first_tile_only)
        self.check_overflow_mode = _resolve_check_overflow_mode(check_overflow_mode)
        # Phase 2A (optional): fuse kernel2b count+write into one allpairs pass.
        self.fuse_count_write = bool(fuse_count_write)
        # Phase 2B (optional): use FP32 CSR coefficient buffers in FP32 mode.
        if self._dtype == cp.float32:
            fp32_coeff_data = True if (fp32_coeff_data is False) else bool(fp32_coeff_data)
        self.fp32_coeff_data = bool(fp32_coeff_data) and (self._dtype == cp.float32)
        self._csr_data_dtype = cp.float32 if bool(self.fp32_coeff_data) else cp.float64

        requested_j_tile = int(j_tile)
        if requested_j_tile < 1:
            raise ValueError("j_tile must be >= 1")
        self.j_tile = int(requested_j_tile)
        self.j_tile_auto_promoted_2048 = False

        # Phase 1B auto-policy:
        # For large-CAS no-EPQ runs, j_tile=2048 usually reduces per-tile fixed cost.
        # Apply only to the default j_tile=1024 case and only when a conservative
        # workspace+G estimate fits comfortably in current free GPU memory.
        fused_hop_eligible = bool(
            self.use_fused_hop
            and
            self._fused_hop_kernel_available
            and int(self.norb) <= 20
            and (eri_mat is not None or l_full is not None)
        )
        if direct_op is not None:
            fused_hop_eligible = False
        if (
            int(requested_j_tile) == 1024
            and int(self.ncsf) >= 1_000_000
            and int(self.nops) <= 256
            and ((not bool(self.use_epq_table)) or fused_hop_eligible)
        ):
            n_pairs_est = int(self.norb) * max(0, int(self.norb) - 1)
            j_tile_target = min(int(self.ncsf), 2048)
            cap_mult = max(1.0, float(csr_capacity_mult))
            max_nnz_in_est = int(cap_mult * float(int(j_tile_target) * int(max(1, n_pairs_est))))
            # Conservative workspace estimate: packed/sort/reduce buffers are the dominant term.
            ws_bytes_est = int(max_nnz_in_est) * 160
            total_est = int(ws_bytes_est) + int(max(0, int(self.max_g_bytes)))
            can_promote = False
            try:
                free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
                can_promote = int(total_est) <= int(0.70 * int(free_bytes))
            except Exception:
                # Fallback: allow promotion only for small absolute estimate.
                can_promote = int(total_est) <= 2 * 1024 * 1024 * 1024
            if can_promote and int(j_tile_target) > int(requested_j_tile):
                self.j_tile = int(j_tile_target)
                self.j_tile_auto_promoted_2048 = True

        if self.epq_stream_j_tile <= 0:
            self.epq_stream_j_tile = int(self.j_tile)
        if self.epq_stream_j_tile < 1:
            raise ValueError("epq_stream_j_tile must be >= 1")
        self.csr_capacity_mult = float(csr_capacity_mult)
        if self.csr_capacity_mult <= 0.0:
            raise ValueError("csr_capacity_mult must be > 0")
        self.cache_csr_tiles = _resolve_cache_csr_tiles(
            cache_csr_tiles,
            j_tile=int(self.j_tile),
            ncsf=int(drt.ncsf),
            norb=int(drt.norb),
            csr_capacity_mult=float(self.csr_capacity_mult),
        )

        # 10.20.4 Item #2: Store cached CSR tile data in FP32 for FP32 workspaces.
        # This halves the memory for cached tiles. The scratch buffer remains FP64
        # (for kernel25 precision), but we cast to FP32 before caching.
        self.fp32_csr_cache = _resolve_fp32_csr_cache(
            fp32_csr_cache,
            dtype_is_float32=bool(self._dtype == cp.float32),
        )

        # Phase 3B (host-side CSR caching):
        # Keep selected CSR tiles in pinned host memory and upload on later hops.
        # This is primarily useful for large multi-hop solves where device-side full
        # caching is disabled by memory limits.
        self.csr_host_cache_budget_gib = max(0.0, float(csr_host_cache_budget_gib))
        self.csr_host_cache_budget_bytes = int(self.csr_host_cache_budget_gib * 1024 * 1024 * 1024)
        self.csr_host_cache_min_ncsf = max(1, int(csr_host_cache_min_ncsf))
        self.csr_host_cache_mode = _resolve_csr_host_cache_mode(csr_host_cache)

        self.csr_host_cache_enabled = _resolve_csr_host_cache_enabled(
            mode=str(self.csr_host_cache_mode),
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            budget_bytes=int(self.csr_host_cache_budget_bytes),
            min_ncsf=int(self.csr_host_cache_min_ncsf),
            cache_csr_tiles=bool(self.cache_csr_tiles),
            aggregate_offdiag_k=bool(self.aggregate_offdiag_k),
            use_epq_table=bool(self.use_epq_table),
        )

        # EPQ apply tile cache: host-pinned cache for streaming EPQ tiles used by
        # the aggregate post-loop apply.  On the first hop, tiles are built on device
        # and copied to host; on subsequent hops they are replayed from host, avoiding
        # costly DFS walks.
        self.epq_apply_cache_budget_gib = _resolve_epq_apply_cache_budget_gib(
            epq_apply_cache_budget_gib,
            os.getenv("ASUKA_CUGUGA_EPQ_APPLY_CACHE_BUDGET_GIB", ""),
        )
        self.epq_apply_cache_budget_bytes = int(self.epq_apply_cache_budget_gib * 1024 * 1024 * 1024)
        self.epq_apply_cache_mode = _resolve_epq_apply_cache_mode(
            epq_apply_cache,
            os.getenv("ASUKA_CUGUGA_EPQ_APPLY_CACHE", ""),
        )
        self.epq_apply_cache_enabled = _resolve_epq_apply_cache_enabled(
            mode=str(self.epq_apply_cache_mode),
            aggregate_offdiag_k=bool(self.aggregate_offdiag_k),
            use_epq_table=bool(self.use_epq_table),
            budget_bytes=int(self.epq_apply_cache_budget_bytes),
            has_epq_table_device_build=bool(has_epq_table_device_build()),
            csr_host_cache_mode=str(getattr(self, "csr_host_cache_mode", "off")),
        )
        # Auto arbitration: if EPQ apply cache is active, disable CSR host-cache auto path
        # so both caches do not compete for large pinned-host budgets.
        if bool(self.epq_apply_cache_enabled) and str(getattr(self, "csr_host_cache_mode", "off")) == "auto":
            self.csr_host_cache_enabled = False

        # Phase 3A (multi-stream pipelined tile processing):
        # Build CSR on one stream while applying the previous tile on another stream.
        # This is only useful for multi-tile uncached CSR builds.
        self.csr_pipeline_min_ncsf = max(1, int(csr_pipeline_min_ncsf))
        self.csr_pipeline_streams_mode, self.csr_pipeline_streams = _resolve_csr_pipeline_streams(
            csr_pipeline_streams
        )
        self.csr_pipeline_enabled = _resolve_csr_pipeline_enabled(
            streams_mode=str(self.csr_pipeline_streams_mode),
            streams=int(self.csr_pipeline_streams),
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            min_ncsf=int(self.csr_pipeline_min_ncsf),
            cache_csr_tiles=bool(self.cache_csr_tiles),
            csr_host_cache_enabled=bool(self.csr_host_cache_enabled),
            aggregate_offdiag_k=bool(self.aggregate_offdiag_k),
            use_epq_table=bool(self.use_epq_table),
        )

        # Phase 2C (optional): pre-filter trivial (j,r,s) tasks using host occupancy checks.
        # This can reduce kernel2b DRT walks by dispatching only non-trivial tasks to the
        # tasks-based CSR builder path.
        self.prefilter_trivial_tasks_min_ncsf = max(1, int(prefilter_trivial_tasks_min_ncsf))
        self.prefilter_trivial_tasks_mode = _resolve_prefilter_trivial_tasks_mode(prefilter_trivial_tasks)
        self.prefilter_trivial_tasks_enabled = _resolve_prefilter_trivial_tasks_enabled(
            mode=str(self.prefilter_trivial_tasks_mode),
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            min_ncsf=int(self.prefilter_trivial_tasks_min_ncsf),
            use_epq_table=bool(self.use_epq_table),
            aggregate_offdiag_k=bool(self.aggregate_offdiag_k),
        )

        # Skip two-body j-tiles whose task scales are exactly zero.
        # This is especially valuable for early Davidson vectors (often sparse)
        # while adding minimal overhead for dense vectors (single count_nonzero pass).
        self.skip_zero_x_tiles_min_ncsf = max(1, int(skip_zero_x_tiles_min_ncsf))
        self.skip_zero_x_tiles_mode = _resolve_skip_zero_x_tiles_mode(skip_zero_x_tiles)
        self.skip_zero_x_tiles_enabled = _resolve_skip_zero_x_tiles_enabled(
            mode=str(self.skip_zero_x_tiles_mode),
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            min_ncsf=int(self.skip_zero_x_tiles_min_ncsf),
        )

        if self.epq_streaming:
            if not self.use_epq_table:
                raise ValueError("epq_streaming=True requires use_epq_table=True")
            if self.use_cuda_graph:
                raise ValueError("epq_streaming is not supported with use_cuda_graph=True")
            if not has_epq_table_device_build():
                raise RuntimeError(
                    "epq_streaming requires epq-table device-build kernels; rebuild extension with CUDA entrypoints"
                )
            self.epq_build_device = True

        if state_cache is None:
            state_cache = get_state_cache(drt)
        self._state_cache = state_cache

        if drt_dev is None:
            drt_dev = make_device_drt(drt)
        if state_dev is None:
            state_dev = make_device_state_cache(drt, drt_dev, cache=state_cache)
        self.drt_dev = drt_dev
        self.state_dev = state_dev

        _epq_tbl = _init_workspace_epq_table(
            cp=cp,
            drt=drt,
            drt_dev=drt_dev,
            state_dev=state_dev,
            use_epq_table=bool(self.use_epq_table),
            epq_streaming=bool(self.epq_streaming),
            epq_build_device=bool(self.epq_build_device),
            epq_build_j_tile=int(self.epq_build_j_tile),
            j_tile=int(self.j_tile),
            norb=int(self.norb),
            ncsf=int(self.ncsf),
            threads_enum=int(self.threads_enum),
            epq_recompute_warp_coop=bool(self.epq_recompute_warp_coop),
            dtype=self._dtype,
            epq_indptr_dtype=str(self.epq_indptr_dtype),
            epq_build_nthreads=int(self.epq_build_nthreads),
            ext=_ext,
            build_device_tiled_fn=build_epq_action_table_combined_device_tiled,
            build_device_fn=build_epq_action_table_combined_device,
            build_host_fn=_get_epq_action_table_combined_host,
            indptr_dtype_resolver_fn=_epq_indptr_cp_dtype_for_total_nnz,
            as_indptr_array_fn=_as_epq_indptr_array,
            as_pq_array_fn=_as_epq_pq_array,
            epq_i32_max_nnz=int(_EPQ_I32_MAX_NNZ),
        )
        self._epq_table = _epq_tbl["epq_table"]
        self.epq_build_device = bool(_epq_tbl["epq_build_device"])
        self.epq_table_build_s = float(_epq_tbl["epq_table_build_s"])

        self.eri_mat = None if eri_mat is None else cp.ascontiguousarray(cp.asarray(eri_mat, dtype=self._dtype))
        self.l_full = None if l_full is None else cp.ascontiguousarray(cp.asarray(l_full, dtype=self._dtype))
        self.direct_op = direct_op
        if self.l_full is not None:
            if self.l_full.ndim != 2 or tuple(self.l_full.shape)[0] != int(self.nops):
                raise ValueError("l_full must have shape (norb*norb, naux)")
        self.naux = 0 if self.l_full is None else int(self.l_full.shape[1])
        if self.direct_op is not None and self._dtype != cp.float64:
            raise ValueError("direct_op currently requires dtype=float64")
        if self._dtype == cp.float32:
            if self.eri_mat is None and self.l_full is None and self.direct_op is None:
                raise ValueError("dtype=float32 requires eri_mat, l_full, or direct_op")
            if (not bool(self.use_epq_table)) and self.eri_mat is None:
                raise ValueError("dtype=float32 no-EPQ path currently requires dense eri_mat (DF/direct no-EPQ unsupported)")
        self._gdf_ws: Kernel3BuildGDFWorkspace | None = None
        self.h_eff_flat = None if h_eff is None else self._as_h_eff_flat(h_eff)
        self._eri_diag_t = None
        self._eri_mat_t = None
        # 10.16.5: Cache for per-call ERI transpose keyed on device pointer to avoid recomputation.
        self._eri_mat_t_cache: dict[int, object] = {}

        # One-body tasks: apply g_flat to every CSF index (scale = x[j]).
        self.task_csf_all = cp.arange(int(self.ncsf), dtype=cp.int32)
        self.overflow_apply = cp.empty((1,), dtype=cp.int32)
        self._overflow_w = cp.empty((1,), dtype=cp.int32)
        self._task_scale_j = cp.empty((int(self.j_tile),), dtype=self._dtype)

        # Reusable buffers for diagonal rs terms.
        # `build_occ_block_from_steps_inplace_device` is float64-only today.
        self._occ_buf = cp.empty((int(self.j_tile), int(self.norb)), dtype=cp.float64)
        self._g_diag_buf = None
        self._diag_w_buf = None
        if self.include_diagonal_rs:
            self._g_diag_buf = cp.empty((int(self.j_tile), int(self.nops)), dtype=self._dtype)

        # Precompute off-diagonal rs pair lists (host + device copies for tiling).
        r_all = np.repeat(np.arange(self.norb, dtype=np.int32), self.norb)
        s_all = np.tile(np.arange(self.norb, dtype=np.int32), self.norb)
        mask = r_all != s_all
        r_all = r_all[mask]
        s_all = s_all[mask]
        self._rs_r_h = np.asarray(r_all, dtype=np.int32)
        self._rs_s_h = np.asarray(s_all, dtype=np.int32)
        self._rs_r_d = cp.asarray(r_all, dtype=cp.int32)
        self._rs_s_d = cp.asarray(s_all, dtype=cp.int32)
        self._rs_n_pairs = int(r_all.size)
        self._step_occ_lut = np.zeros((256,), dtype=np.int8)
        self._step_occ_lut[1] = 1
        self._step_occ_lut[2] = 1
        self._step_occ_lut[3] = 2

        # Optional persistent workspace to eliminate hot-path alloc/free in Kernel2.5 + scans.
        self._k25_ws = None

        # CSR buffers (reused); allocate to the initial capacity for the configured j_tile.
        self._alloc_csr_buffers(capacity=self._initial_csr_capacity())
        _cache_state = _build_workspace_cache_state(
            cp=cp,
            cache_csr_tiles=bool(self.cache_csr_tiles),
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            csr_capacity_mult=float(self.csr_capacity_mult),
            rs_n_pairs=int(self._rs_n_pairs) if hasattr(self, "_rs_n_pairs") else None,
            norb=int(self.norb),
            csr_data_dtype=self._csr_data_dtype,
        )
        self._csr_single_tile_cache = _cache_state["_csr_single_tile_cache"]
        self._csr_tile_cache = _cache_state["_csr_tile_cache"]
        self._csr_host_tile_cache = _cache_state["_csr_host_tile_cache"]
        self._csr_host_cache_bytes = _cache_state["_csr_host_cache_bytes"]
        self._csr_host_cache_hits = _cache_state["_csr_host_cache_hits"]
        self._csr_host_cache_misses = _cache_state["_csr_host_cache_misses"]
        self._csr_host_cache_store_attempts = _cache_state["_csr_host_cache_store_attempts"]
        self._csr_host_cache_store_accepts = _cache_state["_csr_host_cache_store_accepts"]
        self._csr_host_cache_evictions = _cache_state["_csr_host_cache_evictions"]
        self._epq_apply_tile_cache = _cache_state["_epq_apply_tile_cache"]
        self._epq_apply_cache_bytes = _cache_state["_epq_apply_cache_bytes"]
        self._epq_apply_cache_hits = _cache_state["_epq_apply_cache_hits"]
        self._epq_apply_cache_misses = _cache_state["_epq_apply_cache_misses"]
        self._epq_apply_staging_capacity = _cache_state["_epq_apply_staging_capacity"]
        self._epq_apply_staging_indptr = _cache_state["_epq_apply_staging_indptr"]
        self._epq_apply_staging_indices = _cache_state["_epq_apply_staging_indices"]
        self._epq_apply_staging_pq_ids = _cache_state["_epq_apply_staging_pq_ids"]
        self._epq_apply_staging_data = _cache_state["_epq_apply_staging_data"]
        self._csr_pipeline_slots = _cache_state["_csr_pipeline_slots"]
        self._csr_pipeline_apply_stream = _cache_state["_csr_pipeline_apply_stream"]
        self._tile_csr_row_j = _cache_state["_tile_csr_row_j"]
        self._tile_csr_row_k = _cache_state["_tile_csr_row_k"]
        self._tile_csr_indptr = _cache_state["_tile_csr_indptr"]
        self._tile_csr_indices = _cache_state["_tile_csr_indices"]
        self._tile_csr_data = _cache_state["_tile_csr_data"]
        self._tile_csr_overflow = _cache_state["_tile_csr_overflow"]
        self._tile_csr_capacity = _cache_state["_tile_csr_capacity"]

        self._init_csr_pipeline_slots()

        # 10.16.4: Pre-convert occupancy buffer to workspace dtype to avoid per-tile astype() allocations.
        self._occ_buf_dtype = _make_occ_buf_dtype(
            cp=cp,
            dtype=self._dtype,
            j_tile=int(self.j_tile),
            norb=int(self.norb),
        )

        # Temporary g buffer for Kernel4 row blocks.
        nrows_block = _resolve_workspace_nrows_block(
            max_g_bytes=int(self.max_g_bytes),
            nops=int(self.nops),
            itemsize=int(self._itemsize),
            ncsf=int(self.ncsf),
            eri_mat_present=bool(self.eri_mat is not None),
            l_full_present=bool(self.l_full is not None),
            naux=int(self.naux),
        )
        self._g_buf = cp.empty((int(nrows_block), int(self.nops)), dtype=self._dtype)
        self._task_scale_rows = cp.empty((int(nrows_block),), dtype=self._dtype)
        self._diag_g_cache: dict[int, object] = {}

        _sym_pair = _init_sym_pair_setup(
            cp=cp,
            aggregate_offdiag_k=bool(self.aggregate_offdiag_k),
            has_sym_pair_pack_device=bool(has_sym_pair_pack_device()),
            norb=int(self.norb),
            nops=int(self.nops),
            dtype=self._dtype,
            kernel3_workspace_ctor=Kernel3BuildGWorkspace,
        )
        self._sym_pair_pair_pq = _sym_pair["_sym_pair_pair_pq"]
        self._sym_pair_pair_qp = _sym_pair["_sym_pair_pair_qp"]
        self._sym_pair_full_to_pair = _sym_pair["_sym_pair_full_to_pair"]
        self._sym_pair_npair = _sym_pair["_sym_pair_npair"]
        self._sym_pair_eri_pair = _sym_pair["_sym_pair_eri_pair"]
        self._sym_pair_eri_pair_src_id = _sym_pair["_sym_pair_eri_pair_src_id"]
        self._sym_pair_l_pair = _sym_pair["_sym_pair_l_pair"]
        self._sym_pair_l_pair_src_id = _sym_pair["_sym_pair_l_pair_src_id"]
        self._sym_pair_w_pair = _sym_pair["_sym_pair_w_pair"]
        self._sym_pair_g_pair = _sym_pair["_sym_pair_g_pair"]
        self._sym_pair_gemm_ws = _sym_pair["_sym_pair_gemm_ws"]

        # Optional k-aggregated off-diagonal buffers:
        #   W[k,rs] = sum_j x[j] * c(j->k,rs)
        #   g_block[k,pq] = 0.5 * W_block @ ERI_mat
        self._w_offdiag = None
        self._w_block = None
        self._l_full_t = None
        self._offdiag_df_t = None
        self._offdiag_gemm_ws: Kernel3BuildGWorkspace | None = None
        self._offdiag_cublas_workspace_bytes = 0
        self._offdiag_cublas_info: dict[str, object] | None = None
        if bool(self.aggregate_offdiag_k):
            self._offdiag_gemm_ws = Kernel3BuildGWorkspace(
                int(self.nops),
                max_nrows=1,
                dtype=self._dtype,
                gemm_backend=str(self.gemm_backend),
            )
            _configure_offdiag_gemm_workspace(
                ws=self._offdiag_gemm_ws,
                gemm_backend=str(self.gemm_backend),
                offdiag_enable_fp64_emulation=bool(self.offdiag_enable_fp64_emulation),
                offdiag_emulation_strategy=str(self.offdiag_emulation_strategy),
            )

            from asuka.cuda.cublas_workspace import recommend_cublas_workspace_bytes_for_emulated_fp64_gemm

            nrows_eff = min(int(self.ncsf), int(getattr(self._g_buf, "shape", (0,))[0]))
            _autoset_offdiag_cublas_workspace_with_backoff(
                ws=self._offdiag_gemm_ws,
                nops=int(self.nops),
                nrows_eff=int(nrows_eff),
                cap_mb=int(self.offdiag_cublas_workspace_cap_mb),
                recommend_fn=recommend_cublas_workspace_bytes_for_emulated_fp64_gemm,
            )
            self._offdiag_cublas_workspace_bytes = int(self._offdiag_gemm_ws.cublas_workspace_bytes())
            self._offdiag_cublas_info = self._offdiag_gemm_ws.cublas_emulation_info()

            # Full W buffer (O(ncsf*nops)) is only needed for the ERI_mat aggregate path.
            # For DF (L_full) we avoid allocating it to prevent GPU OOM.
            # Allocation is deferred to first use in hop() to avoid wasting VRAM
            # when fused hop mode is active (which bypasses the aggregate path).
            self._w_offdiag_prefer_blocked = _resolve_w_offdiag_prefer_blocked(
                dtype_is_float32=bool(self._dtype == cp.float32),
                ncsf=int(self.ncsf),
                epq_table_present=bool(self._epq_table is not None),
                use_epq_table=bool(self.use_epq_table),
                eri_mat_present=bool(self.eri_mat is not None),
            )

    def _sym_pair_get_eri_pair(self, eri_mat):
        """Build or retrieve cached compressed ERI pair matrix: (npair, npair) from (nops, nops)."""
        import cupy as cp

        src_id = id(eri_mat)
        if self._sym_pair_eri_pair is not None and self._sym_pair_eri_pair_src_id == src_id:
            return self._sym_pair_eri_pair

        pair_pq = self._sym_pair_pair_pq
        pair_qp = self._sym_pair_pair_qp
        if pair_pq is None or pair_qp is None:
            return None

        eri_mat = cp.asarray(eri_mat, dtype=self._dtype)
        eri_mat = cp.ascontiguousarray(eri_mat)
        # Symmetrize: eri_pair[u,v] = 0.5*(eri[pq,·]+eri[qp,·]) rows, then same on columns.
        rows = 0.5 * (cp.take(eri_mat, pair_pq, axis=0) + cp.take(eri_mat, pair_qp, axis=0))
        eri_pair = 0.5 * (cp.take(rows, pair_pq, axis=1) + cp.take(rows, pair_qp, axis=1))
        eri_pair = cp.ascontiguousarray(eri_pair)

        self._sym_pair_eri_pair_src_id = src_id
        self._sym_pair_eri_pair = eri_pair
        return eri_pair

    def _sym_pair_get_l_pair(self, l_full):
        """Build or retrieve cached compressed L_pair: (npair, naux) from (nops, naux)."""
        import cupy as cp

        src_id = id(l_full)
        if self._sym_pair_l_pair is not None and self._sym_pair_l_pair_src_id == src_id:
            return self._sym_pair_l_pair

        pair_pq = self._sym_pair_pair_pq
        pair_qp = self._sym_pair_pair_qp
        if pair_pq is None or pair_qp is None:
            return None

        l_full = cp.asarray(l_full, dtype=self._dtype)
        l_full = cp.ascontiguousarray(l_full)
        l_pair = 0.5 * (cp.take(l_full, pair_pq, axis=0) + cp.take(l_full, pair_qp, axis=0))
        l_pair = cp.ascontiguousarray(l_pair)

        self._sym_pair_l_pair_src_id = src_id
        self._sym_pair_l_pair = l_pair
        return l_pair

    def _sym_pair_ensure_buffers(self, nrows_block_max):
        """Ensure W_pair/G_pair buffers for symmetric-pair GEMM.

        To minimize VRAM overhead, ``w_pair`` aliases the flat memory of
        ``_w_block`` (which has shape ``(nrows_block_max, nops)`` and is no
        longer needed once packing is done).  Only ``g_pair`` gets a
        dedicated allocation — a single ``(nrows_block_max, npair)`` buffer.
        """
        import cupy as cp

        npair = self._sym_pair_npair
        if npair <= 0:
            return
        nrows = int(nrows_block_max)
        shape = (nrows, int(npair))
        # w_pair: alias into _w_block flat memory (npair <= nops, so it fits).
        w_block = self._w_block
        if w_block is not None and w_block.size >= nrows * npair:
            self._sym_pair_w_pair = cp.ndarray(
                shape, dtype=self._dtype,
                memptr=w_block.data,
            )
        elif self._sym_pair_w_pair is None or tuple(self._sym_pair_w_pair.shape) != shape:
            self._sym_pair_w_pair = cp.empty(shape, dtype=self._dtype)
        # g_pair: needs its own allocation (used simultaneously with g_block).
        if self._sym_pair_g_pair is None or tuple(self._sym_pair_g_pair.shape) != shape:
            self._sym_pair_g_pair = cp.empty(shape, dtype=self._dtype)

    def _build_diag_g_cache(self) -> None:
        """Precompute the diagonal (r==s) g_diag blocks for all j-tiles.

        This is intended for CUDA Graph capture, because CuPy's cuBLAS wrappers currently
        do not support stream capture. The cached blocks allow `hop()` to avoid cuBLAS
        calls during capture by reusing the precomputed `g_diag[j,pq]` values.
        """

        import cupy as cp

        if not bool(self.include_diagonal_rs):
            return
        if self.eri_mat is None:
            raise ValueError("eri_mat must be provided (workspace eri_mat is None)")

        # Safety: caching g_diag for all CSFs costs O(ncsf*nops*dtype).
        est_bytes = int(self.ncsf) * int(self.nops) * int(self._itemsize)
        if est_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError(
                f"diag g cache too large ({est_bytes/1024/1024:.1f} MiB); disable graph or reduce problem size"
            )

        if self._eri_diag_t is None:
            diag_ids = cp.asarray([int(r) * int(self.norb) + int(r) for r in range(int(self.norb))], dtype=cp.int32)
            self._eri_diag_t = self.eri_mat[:, diag_ids].T.copy()

        stream = cp.cuda.get_current_stream()
        for j0 in range(0, int(self.ncsf), int(self.j_tile)):
            j1 = min(int(self.ncsf), int(j0 + int(self.j_tile)))
            j_count = int(j1 - j0)
            occ_d = self._occ_buf[:j_count]
            build_occ_block_from_steps_inplace_device(
                self.state_dev,
                j_start=int(j0),
                j_count=int(j_count),
                occ_out=occ_d,
                threads=256,
                stream=stream,
                sync=False,
            )
            g_diag = self._diag_g_cache.get(int(j0))
            if g_diag is None:
                if getattr(self, "_cuda_graph", None) is not None:
                    raise RuntimeError("diag g cache is missing while CUDA Graph is active")
                g_diag = cp.empty((int(j_count), int(self.nops)), dtype=self._dtype)
                self._diag_g_cache[int(j0)] = g_diag
            else:
                # Keep pointers stable (CUDA Graph capture). If the shape changes while a graph exists,
                # do not silently reallocate: it would invalidate the captured graph.
                if getattr(self, "_cuda_graph", None) is not None and tuple(getattr(g_diag, "shape", ())) != (
                    int(j_count),
                    int(self.nops),
                ):
                    raise RuntimeError("diag g cache shape mismatch while CUDA Graph is active")
            # 10.16.4: Use pre-allocated dtype buffer for FP32 mode.
            if self._dtype == cp.float64:
                occ_use = occ_d
            elif self._occ_buf_dtype is not None:
                occ_use = self._occ_buf_dtype[:j_count]
                cp.copyto(occ_use, occ_d)
            else:
                occ_use = occ_d.astype(self._dtype)
            cp.dot(occ_use, self._eri_diag_t, out=g_diag)  # type: ignore[arg-type]
            g_diag *= 0.5
        stream.synchronize()

    @property
    def dtype(self):
        return self._dtype

    def _iter_j_tile_starts(self):
        jt = int(self.j_tile)
        for j0 in range(0, int(self.ncsf), jt):
            yield int(j0)

    def _csr_cache_ready(self) -> bool:
        return _csr_cache_ready_runtime(
            j_tile=int(self.j_tile),
            ncsf=int(self.ncsf),
            cache_csr_tiles=bool(self.cache_csr_tiles),
            csr_single_tile_cache=self._csr_single_tile_cache,
            csr_tile_cache=self._csr_tile_cache,
        )

    def _release_tile_csr_scratch(self) -> None:
        """Release multi-tile CSR build scratch once cache is fully populated."""
        _release_tile_csr_scratch_runtime(self)

    @staticmethod
    def _estimate_object_nbytes(obj, seen: set[int]) -> int:
        return _estimate_object_nbytes_runtime(obj, seen)

    def workspace_nbytes_estimate(self) -> int:
        """Best-effort estimate of live device/pinned workspace bytes."""
        return _workspace_nbytes_estimate_from_dict_runtime(self.__dict__)

    def release(self) -> int:
        """Release large retained buffers and cache structures.

        Returns
        -------
        int
            Best-effort estimated bytes held before release.
        """
        return _release_workspace_resources_runtime(self)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def _init_csr_pipeline_slots(self) -> None:
        import cupy as cp

        slots, apply_stream = _init_csr_pipeline_slots_runtime(
            cp=cp,
            ext=_ext,
            enabled=bool(getattr(self, "csr_pipeline_enabled", False)),
            n_slots=int(getattr(self, "csr_pipeline_streams", 0)),
            initial_capacity=int(self._initial_csr_capacity()),
            j_tile=int(self.j_tile),
            rs_n_pairs=int(self._rs_n_pairs),
            csr_data_dtype=self._csr_data_dtype,
        )
        self._csr_pipeline_slots = slots
        self._csr_pipeline_apply_stream = apply_stream

    def _grow_csr_pipeline_slot(self, slot_idx: int, new_cap: int) -> None:
        import cupy as cp

        _grow_csr_pipeline_slot_runtime(
            cp=cp,
            ext=_ext,
            slots=self._csr_pipeline_slots,
            slot_idx=int(slot_idx),
            new_cap=int(new_cap),
            j_tile=int(self.j_tile),
            rs_n_pairs=int(self._rs_n_pairs),
            csr_data_dtype=self._csr_data_dtype,
        )

    def _prefilter_nontrivial_tasks_host(
        self,
        *,
        j0: int,
        j_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
        """Build compact host task arrays for non-trivial (j,r,s) tasks.

        A task is trivial when `occ_q <= 0` or `occ_p >= 2`, matching the
        early-exit condition in kernel2b. Returning `None` means "do not
        dispatch the task-compacted path for this tile".
        """

        j0 = int(j0)
        j_count = int(j_count)
        n_pairs = int(self._rs_n_pairs)
        if j_count <= 0 or n_pairs <= 0:
            return None

        steps_all = getattr(self._state_cache, "steps", None)
        if steps_all is None:
            return None

        steps_tile = np.asarray(steps_all[j0 : j0 + j_count], dtype=np.int8, order="C")
        if steps_tile.shape != (j_count, int(self.norb)):
            return None

        # step codes map to occupancy via LUT:
        # E/U/L/D -> 0/1/1/2.
        occ_tile = self._step_occ_lut[steps_tile]
        pair_r = self._rs_r_h
        pair_s = self._rs_s_h
        keep_mask = (occ_tile[:, pair_s] > 0) & (occ_tile[:, pair_r] < 2)

        total_tasks = int(j_count) * int(n_pairs)
        kept_tasks = int(np.count_nonzero(keep_mask))
        skipped_tasks = int(total_tasks - kept_tasks)
        if skipped_tasks <= 0:
            return None

        # Auto-mode guard: avoid paying compaction overhead when very few tasks
        # are filtered out.
        if str(getattr(self, "prefilter_trivial_tasks_mode", "auto")) == "auto":
            min_skipped = max(4096, int(total_tasks // 20))  # ~5% or at least 4K tasks
            if skipped_tasks < min_skipped:
                return None

        if kept_tasks <= 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty, empty, int(total_tasks), 0

        j_rel, pair_idx = np.nonzero(keep_mask)
        task_csf_h = np.ascontiguousarray(j_rel.astype(np.int64, copy=False) + int(j0), dtype=np.int32)
        task_p_h = np.ascontiguousarray(pair_r[pair_idx], dtype=np.int32)
        task_q_h = np.ascontiguousarray(pair_s[pair_idx], dtype=np.int32)
        return task_csf_h, task_p_h, task_q_h, int(total_tasks), int(kept_tasks)

    def _ensure_csr_staging_capacity(self, min_cap: int) -> None:
        cap = int(max(1, min_cap))
        if int(getattr(self, "_csr_capacity", 0)) >= cap:
            return
        self._alloc_csr_buffers(capacity=cap)

    @staticmethod
    def _alloc_pinned_np(shape: tuple[int, ...], dtype) -> tuple[object, np.ndarray]:
        import cupy as cp

        dt = np.dtype(dtype)
        size = int(np.prod(shape, dtype=np.int64))
        nbytes = int(size) * int(dt.itemsize)
        mem = cp.cuda.alloc_pinned_memory(nbytes)
        arr = np.frombuffer(mem, dtype=dt, count=size).reshape(shape)
        return mem, arr

    @staticmethod
    def _csr_host_entry_bytes(*, nrows: int, nnz: int, data_itemsize: int) -> int:
        return _csr_host_entry_bytes_runtime(
            nrows=int(nrows),
            nnz=int(nnz),
            data_itemsize=int(data_itemsize),
        )

    def _csr_host_cache_try_admit(self, *, tile_bytes: int, score: float) -> bool:
        return _csr_host_cache_try_admit_runtime(
            self,
            tile_bytes=int(tile_bytes),
            score=float(score),
        )

    def _csr_host_cache_store_tile(
        self,
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
        _csr_host_cache_store_tile_runtime(
            self,
            j0=int(j0),
            row_j_d=row_j_d,
            row_k_d=row_k_d,
            indptr_d=indptr_d,
            indices_d=indices_d,
            data_d=data_d,
            nrows=int(nrows),
            nnz=int(nnz),
            stream=stream,
            profile=profile,
        )

    def _csr_host_cache_load_tile(self, *, j0: int, stream, profile: dict[str, float] | None):
        return _csr_host_cache_load_tile_runtime(
            self,
            j0=int(j0),
            stream=stream,
            profile=profile,
        )

    # ----- EPQ apply tile cache (host-pinned) -----

    def _epq_apply_cache_store(
        self,
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
        """Copy an EPQ tile from device to pinned host memory for later replay."""
        _epq_apply_cache_store_runtime(
            self,
            k0=int(k0),
            indptr_d=indptr_d,
            indices_d=indices_d,
            pq_ids_d=pq_ids_d,
            data_d=data_d,
            j_count=int(j_count),
            nnz=int(nnz),
            stream=stream,
        )

    def _epq_apply_ensure_staging(self, *, j_count: int, nnz: int) -> None:
        """Ensure device staging buffers are large enough for H2D upload."""
        _epq_apply_ensure_staging_runtime(
            self,
            j_count=int(j_count),
            nnz=int(nnz),
            epq_pq_dtype_for_norb_fn=_epq_pq_dtype_for_norb,
        )

    def _epq_apply_cache_load(
        self,
        *,
        k0: int,
        stream,
    ):
        """Load a cached EPQ tile from host to device.  Returns (indptr, indices, pq_ids, data) or None."""
        return _epq_apply_cache_load_runtime(
            self,
            k0=int(k0),
            stream=stream,
            epq_pq_dtype_for_norb_fn=_epq_pq_dtype_for_norb,
        )

    def _diag_g_cache_ready(self) -> bool:
        if not bool(self.include_diagonal_rs):
            return True
        jt = int(self.j_tile)
        ntiles = (int(self.ncsf) + jt - 1) // jt
        if int(len(self._diag_g_cache)) != int(ntiles):
            return False
        for j0 in self._iter_j_tile_starts():
            if int(j0) not in self._diag_g_cache:
                return False
        return True

    def _resolve_apply_warp_coop(self) -> bool:
        """Resolve apply_warp_coop to a concrete boolean.
        
        Auto mode enables warp-cooperative segment walk for norb >= 16 where local memory
        traffic from per-thread DFS stacks becomes a bottleneck. The warp-coop kernel
        distributes the segment walk across warp lanes, eliminating stack spills.
        """
        if isinstance(self.apply_warp_coop, str) and self.apply_warp_coop == "auto":
            # Heuristic: enable warp-coop for norb >= 16 where local memory is significant
            # Also require extension support
            has_warp_coop = _ext is not None and hasattr(_ext, "apply_g_flat_scatter_atomic_warp_coop_inplace_device")
            norb_threshold = 16
            return bool(has_warp_coop and int(self.norb) >= norb_threshold)
        return bool(self.apply_warp_coop)


    def enable_cuda_graph(self, *, mode: int | None = None, warmup: bool = True) -> None:
        """Capture this workspace's `hop()` into a CUDA Graph (prototype).

        Limitations (current):
        - Requires fixed `eri_mat` and `h_eff` in the workspace (no per-call overrides).
        - Only used when `check_overflow=False` and `profile is None`.
        """

        import cupy as cp

        if bool(getattr(self, "aggregate_offdiag_k", False)):
            raise RuntimeError("enable_cuda_graph is not supported with aggregate_offdiag_k=True (uses cuBLAS in hop())")

        if self._cuda_graph is not None:
            self.use_cuda_graph = True
            return

        if self.eri_mat is None:
            raise ValueError("enable_cuda_graph requires workspace eri_mat to be set")
        if self.h_eff_flat is None:
            raise ValueError("enable_cuda_graph requires workspace h_eff to be set")

        if int(self.j_tile) < int(self.ncsf):
            # Stream capture cannot include host reads of variable CSR sizes, so we require caching.
            self.cache_csr_tiles = True

        self._cuda_graph_x = cp.zeros((int(self.ncsf),), dtype=self._dtype)
        self._cuda_graph_y = cp.empty((int(self.ncsf),), dtype=self._dtype)

        if bool(warmup):
            # Prime CSR caches (required for capture) without cuBLAS calls.
            if not self._csr_cache_ready():
                saved_diag = bool(self.include_diagonal_rs)
                if saved_diag:
                    self.include_diagonal_rs = False
                try:
                    _ = self.hop(self._cuda_graph_x, y=self._cuda_graph_y, sync=True, check_overflow=False)
                finally:
                    self.include_diagonal_rs = bool(saved_diag)

            # Precompute diagonal rs blocks so that the captured hop contains no cuBLAS calls.
            if bool(self.include_diagonal_rs) and not self._diag_g_cache_ready():
                self._build_diag_g_cache()
        else:
            # Without warmup, require that the workspace has already populated all caches.
            if not self._csr_cache_ready():
                raise RuntimeError(
                    "CUDA graph capture requires cached CSR tiles; call ws.hop(...) once or enable warmup"
                )
            if bool(self.include_diagonal_rs) and not self._diag_g_cache_ready():
                raise RuntimeError(
                    "CUDA graph capture requires cached diagonal g blocks; enable warmup or call ws.enable_cuda_graph(warmup=True)"
                )

        cap_stream = cp.cuda.Stream(non_blocking=True)
        mode_eff = int(self._cuda_graph_mode) if mode is None else int(mode)
        with cap_stream:
            cap_stream.begin_capture(mode=mode_eff)
            _ = self.hop(self._cuda_graph_x, y=self._cuda_graph_y, sync=False, check_overflow=False)
            self._cuda_graph = cap_stream.end_capture()

        self.use_cuda_graph = True

    def _as_h_eff_flat(self, h_eff) -> "object":
        import cupy as cp

        h_eff = cp.asarray(h_eff, dtype=self._dtype)
        h_eff = cp.ascontiguousarray(h_eff)
        if h_eff.ndim != 2 or h_eff.shape != (int(self.norb), int(self.norb)):
            raise ValueError("h_eff must have shape (norb,norb)")
        return h_eff.reshape(int(self.nops))

    def _pad_k_count(self, k_count: int, *, nrows_max: int) -> int:
        """10.16.8: Pad k_count to the nearest multiple of gemm_k_pad_align for cuBLAS tile efficiency.

        Returns the padded k_count, capped at nrows_max to avoid exceeding the buffer size.
        When gemm_k_pad_align is 0 or 1, returns k_count unchanged.
        """
        align = int(self.gemm_k_pad_align)
        if align <= 1:
            return int(k_count)
        k = int(k_count)
        padded = ((k + align - 1) // align) * align
        # Cap at buffer size to avoid out-of-bounds access
        return min(padded, int(nrows_max))

    def _estimate_epq_blocked_transpose_bytes(self, epq_table) -> int:
        """Estimate incremental bytes needed to materialize EPQ transpose."""
        import cupy as cp

        if epq_table is None or len(epq_table) != 4:
            return 0
        epq_indptr, _epq_indices, epq_pq, epq_data = epq_table
        epq_indptr = cp.asarray(epq_indptr).ravel()
        if int(epq_indptr.size) <= 0:
            return 0
        nnz = int(cp.asnumpy(epq_indptr[-1]))
        if nnz <= 0:
            return 0

        pq_itemsize = int(cp.dtype(cp.asarray(epq_pq).dtype).itemsize)
        data_itemsize = int(cp.dtype(cp.asarray(epq_data).dtype).itemsize)
        if self.epq_indptr_dtype == "int32":
            indptr_itemsize = 4
        elif self.epq_indptr_dtype == "int64":
            indptr_itemsize = 8
        else:
            indptr_itemsize = 4 if nnz <= _EPQ_I32_MAX_NNZ else 8

        # Output transpose payload + auxiliary arrays used during transpose build.
        out_payload = int(nnz) * int(4 + pq_itemsize + data_itemsize)
        aux_bytes = int(self.ncsf) * int(8 + indptr_itemsize) + int(self.ncsf + 1) * int(indptr_itemsize)
        return int(out_payload + aux_bytes)

    def _should_use_blocked_epq_transpose(self, epq_table, *, profile: dict[str, float] | None = None) -> bool:
        """Return whether blocked EPQ aggregate should use full transpose materialization."""
        import cupy as cp

        mode = str(getattr(self, "epq_blocked_transpose_mode", "auto")).strip().lower()
        if mode == "on":
            if profile is not None:
                profile["epq_transpose_guard_active"] = 1.0
                profile["epq_transpose_guard_allow"] = 1.0
                profile["epq_transpose_guard_mode_on"] = 1.0
            return True
        if mode == "off":
            if profile is not None:
                profile["epq_transpose_guard_active"] = 1.0
                profile["epq_transpose_guard_allow"] = 0.0
                profile["epq_transpose_guard_mode_off"] = 1.0
            return False

        if profile is not None:
            profile["epq_transpose_guard_active"] = 1.0

        try:
            est_bytes = int(self._estimate_epq_blocked_transpose_bytes(epq_table))
            pool = cp.get_default_memory_pool()
            used_bytes = int(pool.used_bytes())
            limit_bytes = int(pool.get_limit()) if hasattr(pool, "get_limit") else 0
            free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
            headroom_bytes = int(free_bytes)
            if limit_bytes > 0:
                headroom_bytes = min(headroom_bytes, max(0, int(limit_bytes) - int(used_bytes)))
            reserve_bytes = int(getattr(self, "epq_blocked_transpose_reserve_bytes", 0))
            allow = bool(est_bytes + reserve_bytes <= headroom_bytes)
            if profile is not None:
                profile["epq_transpose_guard_est_bytes"] = float(est_bytes)
                profile["epq_transpose_guard_headroom_bytes"] = float(headroom_bytes)
                profile["epq_transpose_guard_reserve_bytes"] = float(reserve_bytes)
                profile["epq_transpose_guard_allow"] = float(1.0 if allow else 0.0)
            return allow
        except Exception:
            # If telemetry is unavailable, keep legacy behavior.
            if profile is not None:
                profile["epq_transpose_guard_allow"] = 1.0
                profile["epq_transpose_guard_probe_failed"] = 1.0
            return True

    def _build_w_diag_from_steps_inplace(
        self,
        *,
        x,
        w_out,
        j_start: int,
        j_count: int,
        stream,
        sync: bool,
        relative_w: bool = False,
    ) -> None:
        """Build diagonal `r==s` W entries in workspace dtype."""

        build_w_diag_from_steps_inplace_device(
            self.state_dev,
            j_start=int(j_start),
            j_count=int(j_count),
            x=x,
            w_out=w_out,
            threads=256,
            stream=stream,
            sync=bool(sync),
            relative_w=bool(relative_w),
        )

    def _initial_csr_capacity(self) -> int:
        ntasks = int(self.j_tile) * int(self._rs_n_pairs)
        # Tiny active spaces can yield zero implicit (r,s) tasks. Keep capacity
        # at least 1 so workspace allocation remains valid and downstream kernels
        # can operate on empty views.
        return max(1, int(max(1.0, float(self.csr_capacity_mult)) * float(ntasks)))

    def _ensure_kernel25_workspace(self, *, max_nnz_in: int | None = None) -> None:
        if _ext is None or not hasattr(_ext, "Kernel25Workspace"):
            self._k25_ws = None
            return

        max_tasks = int(self.j_tile) * int(self._rs_n_pairs)
        cap = int(self._csr_capacity) if max_nnz_in is None else int(max_nnz_in)
        if cap <= 0:
            cap = int(self._csr_capacity)

        ws = self._k25_ws
        if ws is not None:
            try:
                if int(ws.max_tasks) >= max_tasks and int(ws.max_nnz_in) >= cap:
                    return
            except Exception:
                pass

        try:
            if ws is not None:
                try:
                    ws.release()
                except Exception:
                    pass
            self._k25_ws = _ext.Kernel25Workspace(int(max_tasks), int(cap))
        except Exception:
            self._k25_ws = None

    def _alloc_csr_buffers(self, *, capacity: int) -> None:
        import cupy as cp

        cap = int(capacity)
        if cap <= 0:
            raise ValueError("capacity must be > 0")
        self._csr_capacity = cap
        self._csr_row_j = cp.empty((cap,), dtype=cp.int32)
        self._csr_row_k = cp.empty((cap,), dtype=cp.int32)
        self._csr_indptr = cp.empty((cap + 1,), dtype=cp.int64)
        self._csr_indices = cp.empty((cap,), dtype=cp.int32)
        self._csr_data = cp.empty((cap,), dtype=self._csr_data_dtype)
        self._csr_overflow = cp.empty((1,), dtype=cp.int32)
        self._ensure_kernel25_workspace(max_nnz_in=cap)

    def _fused_hop_tile(
        self,
        j_start: int,
        j_count: int,
        x,
        eri_mat,
        h_eff_flat,
        y,
        *,
        stream=None,
        sync: bool = True,
        check_overflow: bool = True,
        profile=None,
    ):
        """Fused hop kernel: W-build + one-body + ERI contraction + second E_pq apply."""
        import cupy as cp

        if _ext is None:
            raise RuntimeError("CUDA extension not available")

        overflow = self._csr_overflow
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)

        t0 = time.perf_counter() if profile is not None else None
        stream_ptr = 0
        if stream is not None:
            stream_ptr = int(stream.ptr)

        _ext.fused_hop_device(
            self.drt_dev,
            self.state_dev,
            int(j_start),
            int(j_count),
            x,
            eri_mat,
            h_eff_flat,
            y,
            overflow,
            stream=stream_ptr,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
        )

        if profile is not None and t0 is not None:
            if sync and stream is not None:
                stream.synchronize()
            dt = time.perf_counter() - t0
            profile["fused_hop_s"] = profile.get("fused_hop_s", 0.0) + dt
            profile["fused_hop_tiles"] = profile.get("fused_hop_tiles", 0.0) + 1.0
            profile["fused_hop_j_count"] = profile.get("fused_hop_j_count", 0.0) + float(j_count)

    def _fused_hop_phase1_tile(
        self,
        j_start: int,
        j_count: int,
        x,
        eri_mat,
        h_eff_flat,
        y,
        g_out,
        *,
        stream=None,
        sync: bool = False,
        check_overflow: bool = True,
        profile: dict | None = None,
    ):
        """Phase-1-only fused hop: W-build + ERI contraction → writes G to g_out.

        One-body + diagonal two-body contributions are accumulated into y.
        Off-diagonal two-body G[j_count, nops] is written to g_out for
        external scatter via the EPQ table kernel.
        """
        import cupy as cp

        if _ext is None:
            raise RuntimeError("CUDA extension not available")

        overflow = self._csr_overflow
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)

        t0 = time.perf_counter() if profile is not None else None
        stream_ptr = 0
        if stream is not None:
            stream_ptr = int(stream.ptr)

        _ext.fused_hop_phase1_device(
            self.drt_dev,
            self.state_dev,
            int(j_start),
            int(j_count),
            x,
            eri_mat,
            h_eff_flat,
            y,
            g_out,
            overflow,
            stream=stream_ptr,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
        )

        if profile is not None and t0 is not None:
            if sync and stream is not None:
                stream.synchronize()
            dt = time.perf_counter() - t0
            profile["fused_hop_phase1_s"] = profile.get("fused_hop_phase1_s", 0.0) + dt
            profile["fused_hop_phase1_tiles"] = profile.get("fused_hop_phase1_tiles", 0.0) + 1.0
            profile["fused_hop_phase1_j_count"] = profile.get("fused_hop_phase1_j_count", 0.0) + float(j_count)

    # ------------------------------------------------------------------
    # COO-based Phase 1+2 hybrid path
    # ------------------------------------------------------------------

    def _ensure_coo_buffers(self, max_coo: int, force: bool = False):
        """Lazily allocate (or grow) the rotating COO buffers on device.

        If *force* is True, reallocate even if the current buffer is larger
        (used by adaptive calibration to shrink oversized buffers).
        """
        import cupy as cp

        if (
            not force
            and hasattr(self, "_coo_max")
            and self._coo_max >= max_coo
            and self._coo_j_local is not None
        ):
            return  # already big enough

        self._coo_max = int(max_coo)
        self._coo_calibrated = False
        self._coo_nnz_counter = cp.zeros((1,), dtype=cp.int32)
        self._coo_j_local = cp.empty((max_coo,), dtype=cp.int32)
        self._coo_k = cp.empty((max_coo,), dtype=cp.int32)
        self._coo_pq = cp.empty((max_coo,), dtype=cp.int16)
        self._coo_w2 = cp.empty((max_coo,), dtype=self._dtype)

    def _fused_hop_phase1_coo_tile(
        self,
        j_start: int,
        j_count: int,
        x,
        eri_mat,
        h_eff_flat,
        y,
        g_out,
        *,
        stream=None,
        sync: bool = False,
        check_overflow: bool = True,
        profile: dict | None = None,
    ):
        """Phase-1 + COO output: W-build + ERI contraction + COO connectivity.

        Returns the number of COO entries written (nnz).
        """
        import cupy as cp

        if _ext is None:
            raise RuntimeError("CUDA extension not available")

        overflow = self._csr_overflow
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)

        # Reset COO counter to 0
        cp.cuda.runtime.memsetAsync(
            int(self._coo_nnz_counter.data.ptr), 0, 4,
            int(stream.ptr) if stream is not None else 0,
        )

        t0 = time.perf_counter() if profile is not None else None
        stream_ptr = 0
        if stream is not None:
            stream_ptr = int(stream.ptr)

        _use_merged = getattr(self, "use_merged_dfs", False) and hasattr(_ext, "fused_hop_phase1_coo_merged_device")
        _phase1_fn = _ext.fused_hop_phase1_coo_merged_device if _use_merged else _ext.fused_hop_phase1_coo_device

        _phase1_fn(
            self.drt_dev,
            self.state_dev,
            int(j_start),
            int(j_count),
            x,
            eri_mat,
            h_eff_flat,
            y,
            g_out,
            overflow,
            self._coo_nnz_counter,
            self._coo_j_local,
            self._coo_k,
            self._coo_pq,
            self._coo_w2,
            int(self._coo_max),
            stream=stream_ptr,
            sync=False,
            check_overflow=bool(check_overflow),
        )

        # Read back nnz (requires sync to get the counter value)
        nnz_host = int(self._coo_nnz_counter.get()[0])

        if profile is not None and t0 is not None:
            dt = time.perf_counter() - t0
            profile["fused_hop_phase1_coo_s"] = profile.get("fused_hop_phase1_coo_s", 0.0) + dt
            profile["fused_hop_phase1_coo_tiles"] = profile.get("fused_hop_phase1_coo_tiles", 0.0) + 1.0

        return nnz_host

    def _coo_scatter_tile(
        self,
        g_tile,
        nops: int,
        nnz: int,
        y,
        *,
        stream=None,
        sync: bool = False,
        profile: dict | None = None,
    ):
        """Phase-2 COO scatter: reads COO triples and scatters G into y."""
        if _ext is None:
            raise RuntimeError("CUDA extension not available")
        if nnz <= 0:
            return

        t0 = time.perf_counter() if profile is not None else None
        stream_ptr = 0
        if stream is not None:
            stream_ptr = int(stream.ptr)

        _ext.coo_scatter_device(
            self._coo_j_local,
            self._coo_k,
            self._coo_pq,
            self._coo_w2,
            g_tile,
            int(nops),
            int(nnz),
            y,
            stream=stream_ptr,
            sync=bool(sync),
        )

        if profile is not None and t0 is not None:
            if sync and stream is not None:
                stream.synchronize()
            dt = time.perf_counter() - t0
            profile["coo_scatter_s"] = profile.get("coo_scatter_s", 0.0) + dt

    def _hop_via_controller(
        self,
        x,
        *,
        y=None,
        stream=None,
        sync: bool = True,
        check_overflow: bool = True,
        profile: dict[str, float] | None = None,
    ):
        """Delegate hop() to the GugaMatvecController + CudaGugaExecutor split.

        Lazily creates the controller/executor pair on first call.  This path
        is the minimal blocked-EPQ trunk — no fused-hop, no CSR tiling, no
        cache layers.  It is intended as the first runnable backend for ROCm.
        """
        ctrl = getattr(self, "_controller", None)
        # Invalidate if workspace integrals changed since controller was built.
        if ctrl is not None:
            _key = (id(self.eri_mat), id(self.l_full), id(self.h_eff_flat))
            if _key != getattr(self, "_controller_integral_key", None):
                ctrl = None
        if ctrl is None:
            from asuka.cuguga.matvec.controller import GugaMatvecController
            from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor

            executor = CudaGugaExecutor(
                self.drt,
                drt_dev=self.drt_dev,
                state_dev=self.state_dev,
                state_cache=self._state_cache,
                eri_mat=self.eri_mat,
                l_full=self.l_full,
                h_eff=self.h_eff_flat,
                dtype=self._dtype,
                max_g_bytes=self.max_g_bytes,
                threads_w=self.threads_w,
                threads_apply=self.threads_apply,
                epq_build_device=False,  # reuse existing table
                gemm_backend=self.gemm_backend,
                kahan_compensation=bool(getattr(self, "kahan_compensation", False)),
            )
            # Reuse the workspace's pre-built EPQ table instead of rebuilding.
            executor._epq_table = self._epq_table
            ctrl = GugaMatvecController(
                executor, include_diagonal_rs=self.include_diagonal_rs,
            )
            self._controller = ctrl
            self._controller_integral_key = (id(self.eri_mat), id(self.l_full), id(self.h_eff_flat))

        return ctrl.hop(
            x, y=y, stream=stream, sync=sync,
            check_overflow=check_overflow, profile=profile,
        )

    def hop(
        self,
        x,
        *,
        eri_mat=None,
        h_eff=None,
        y=None,
        stream=None,
        sync: bool = True,
        check_overflow: bool = True,
        profile: dict[str, float] | None = None,
    ):
        """Compute y = Hx on GPU (ERI_mat two-body + one-body h_eff).

        Parameters
        ----------
        x
            Device array (CuPy) with shape (ncsf,).
        eri_mat
            Optional device array with shape (nops,nops). If omitted, uses the workspace `eri_mat`.
        h_eff
            Optional host/device array with shape (norb,norb). If omitted, uses the workspace `h_eff_flat`.
        y
            Optional device output array with shape (ncsf,).
        """
        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the CUDA matvec path") from e

        # Resolve runtime path mode from constructor policy.
        path_mode = str(getattr(self, "path_mode", "auto"))
        _hop_flags = _resolve_hop_runtime_flags_runtime(
            self,
            path_mode=str(path_mode),
            eri_mat=eri_mat,
        )
        use_fused_hop = bool(_hop_flags["use_fused_hop"])
        # Fused hop already includes one-body + full two-body apply. Keep aggregate-offdiag disabled
        # in this mode to avoid redundant work and double counting.
        use_aggregate_offdiag = bool(_hop_flags["use_aggregate_offdiag"])

        x = _normalize_hop_x_runtime(self, cp=cp, x=x)

        # --- Controller delegation (opt-in) ---
        # When ASUKA_MATVEC_USE_CONTROLLER=1, delegate the blocked-EPQ aggregate
        # path to the new GugaMatvecController + CudaGugaExecutor split.  This is
        # the minimal trunk with no fused-hop, no CSR, no cache layers — intended
        # as the first runnable backend for ROCm / alternative accelerators.
        if (
            str(os.getenv("ASUKA_MATVEC_USE_CONTROLLER", "")).strip().lower() in ("1", "true", "yes")
            and not use_fused_hop
            and bool(use_aggregate_offdiag)
            and self._epq_table is not None
            and eri_mat is None  # no per-call ERI override
            and h_eff is None   # no per-call h_eff override
        ):
            return self._hop_via_controller(x, y=y, stream=stream, sync=sync,
                                            check_overflow=check_overflow, profile=profile)

        _graph = _try_cuda_graph_fast_path_runtime(
            self,
            cp=cp,
            x=x,
            y=y,
            eri_mat=eri_mat,
            h_eff=h_eff,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
        )
        if bool(_graph["executed"]):
            return _graph["y"]
        stream = _graph["stream"]

        _inputs = _prepare_hop_runtime_inputs_runtime(
            self,
            cp=cp,
            y=y,
            eri_mat=eri_mat,
            h_eff=h_eff,
            use_fused_hop=bool(use_fused_hop),
        )
        y = _inputs["y"]
        eri_mat_use = _inputs["eri_mat_use"]
        l_full_use = _inputs["l_full_use"]
        direct_op_use = _inputs["direct_op_use"]
        use_df = bool(_inputs["use_df"])
        use_direct = bool(_inputs["use_direct"])
        h_eff_flat = _inputs["h_eff_flat"]
        use_epq_streaming = bool(_inputs["use_epq_streaming"])
        use_epq_streaming_tiles = bool(_inputs["use_epq_streaming_tiles"])
        epq_stream_panic_requested = bool(_inputs["epq_stream_panic_requested"])
        epq_stream_panic_active = bool(_inputs["epq_stream_panic_active"])
        if use_direct:
            use_fused_hop = False

        eri_mat_t = None
        if (not use_df) and (not use_direct) and self._epq_table is not None:
            if eri_mat is None:
                if self._eri_mat_t is None:
                    self._eri_mat_t = eri_mat_use.T.copy()
                eri_mat_t = self._eri_mat_t
            else:
                # 10.16.5: Cache per-call ERI transpose keyed on device pointer.
                eri_ptr = int(eri_mat_use.data.ptr)
                cached = self._eri_mat_t_cache.get(eri_ptr)
                if cached is not None:
                    eri_mat_t = cached
                else:
                    eri_mat_t = eri_mat_use.T.copy()
                    # Keep cache bounded (e.g., last 4 unique eri_mat pointers)
                    if len(self._eri_mat_t_cache) >= 4:
                        oldest = next(iter(self._eri_mat_t_cache))
                        del self._eri_mat_t_cache[oldest]
                    self._eri_mat_t_cache[eri_ptr] = eri_mat_t

        if self.include_diagonal_rs and (not use_direct) and not bool(use_aggregate_offdiag) and self._eri_diag_t is None:
            # Cache the transpose of the diagonal rs columns:
            #   eri_diag_t[r,pq] = eri_mat[pq, rr]  with rr = r*norb+r
            diag_ids = cp.asarray([int(r) * int(self.norb) + int(r) for r in range(int(self.norb))], dtype=cp.int32)
            if use_df:
                if l_full_use is None:  # pragma: no cover
                    raise RuntimeError("internal error: l_full_use is not set for DF path")
                l_diag = l_full_use[diag_ids]
                self._eri_diag_t = cp.ascontiguousarray(cp.dot(l_diag, l_full_use.T))
            else:
                self._eri_diag_t = eri_mat_use[:, diag_ids].T.copy()

        if stream is None:
            stream = cp.cuda.get_current_stream()

        if use_fused_hop:
            return _run_fused_hop_path_runtime(
                self,
                cp=cp,
                ext=_ext,
                x=x,
                y=y,
                eri_mat_use=eri_mat_use,
                h_eff_flat=h_eff_flat,
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                profile=profile,
                path_mode=str(path_mode),
                build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device,
                apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device,
            )

        return _run_blocked_hop_path_runtime(
            self,
            cp=cp,
            ext=_ext,
            x=x,
            y=y,
            eri_mat_use=eri_mat_use,
            eri_mat_t=eri_mat_t,
            l_full_use=l_full_use,
            direct_op_use=direct_op_use,
            use_df=bool(use_df),
            use_direct=bool(use_direct),
            use_aggregate_offdiag=bool(use_aggregate_offdiag),
            h_eff_flat=h_eff_flat,
            stream=stream,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile=profile,
            use_fused_hop=bool(use_fused_hop),
            use_epq_streaming=bool(use_epq_streaming),
            use_epq_streaming_tiles=bool(use_epq_streaming_tiles),
            epq_stream_panic_requested=bool(epq_stream_panic_requested),
            epq_stream_panic_active=bool(epq_stream_panic_active),
            path_mode=str(path_mode),
            epq_table=self._epq_table,
            # One-body function injections
            build_epq_action_table_tile_device_fn=build_epq_action_table_tile_device,
            apply_g_flat_scatter_atomic_inplace_device_fn=apply_g_flat_scatter_atomic_inplace_device,
            apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=apply_g_flat_scatter_atomic_epq_table_tile_inplace_device,
            # EPQ-blocked path function injections
            build_epq_action_table_transpose_device_fn=build_epq_action_table_transpose_device,
            build_w_from_epq_transpose_range_inplace_device_fn=build_w_from_epq_transpose_range_inplace_device,
            apply_g_flat_gather_epq_transpose_range_inplace_device_fn=apply_g_flat_gather_epq_transpose_range_inplace_device,
            sym_pair_pack_inplace_device_fn=sym_pair_pack_inplace_device,
            sym_pair_unpack_inplace_device_fn=sym_pair_unpack_inplace_device,
            has_sym_pair_fused_kernels_fn=has_sym_pair_fused_kernels,
            # Per-tile CSR path function injections
            build_w_from_epq_table_inplace_device_fn=build_w_from_epq_table_inplace_device,
            build_occ_block_from_steps_inplace_device_fn=build_occ_block_from_steps_inplace_device,
            kernel4_build_w_from_csr_unitnnz_inplace_device_fn=kernel4_build_w_from_csr_unitnnz_inplace_device,
            kernel4_apply_csr_eri_mat_device_csr_inplace_device_fn=kernel4_apply_csr_eri_mat_device_csr_inplace_device,
            kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device_fn=kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device,
            kernel4_apply_csr_l_full_device_csr_inplace_device_fn=kernel4_apply_csr_l_full_device_csr_inplace_device,
            kernel25_build_csr_from_tasks_deterministic_inplace_device_fn=kernel25_build_csr_from_tasks_deterministic_inplace_device,
            kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_fn=kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device,
            # Shared
            Kernel3BuildGDFWorkspace_cls=Kernel3BuildGDFWorkspace,
        )


def kernel3_build_g_from_csr_eri_mat_cuda(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    eri_mat: np.ndarray,
    *,
    row_start: int = 0,
    nrows: int = -1,
    half: float = 0.5,
    threads: int = 256,
) -> np.ndarray:
    """Kernel 3 (CUDA baseline): build dense `g_out[row,pq]` from CSR `C[row,rs]` and dense `ERI_mat[pq,rs]`."""

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    indptr = np.asarray(indptr, dtype=np.int64).ravel()
    indices = np.asarray(indices, dtype=np.int32).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")

    eri_mat = np.asarray(eri_mat, dtype=np.float64, order="C")
    if eri_mat.ndim != 2 or eri_mat.shape[0] != eri_mat.shape[1]:
        raise ValueError("eri_mat must have shape (nops,nops)")

    out = _ext.kernel3_build_g_from_csr_eri_mat_cuda(
        indptr,
        indices,
        data,
        eri_mat,
        int(row_start),
        int(nrows),
        float(half),
        int(threads),
    )
    return np.asarray(out, dtype=np.float64)


def kernel3_build_g_from_csr_eri_mat_inplace_device(
    indptr,
    indices,
    data,
    eri_mat,
    *,
    g_out=None,
    threads: int = 256,
    half: float = 0.5,
    stream=None,
    sync: bool = True,
):
    """In-place device-array path for Kernel 3 build-g.

    All array-like inputs/outputs must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
    Returns `g_out` (device array) so callers can reuse it across calls.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array Kernel 3 path") from e

    indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
    indptr = cp.ascontiguousarray(indptr)
    indices = cp.asarray(indices, dtype=cp.int32).ravel()
    indices = cp.ascontiguousarray(indices)
    data = cp.asarray(data).ravel()
    if data.dtype not in (cp.float32, cp.float64):
        data = cp.asarray(data, dtype=cp.float64).ravel()
    data = cp.ascontiguousarray(data)

    eri_mat = cp.asarray(eri_mat)
    fp_dtype = cp.float32 if eri_mat.dtype == cp.float32 else cp.float64
    if data.dtype != fp_dtype:
        data = cp.asarray(data, dtype=fp_dtype).ravel()
        data = cp.ascontiguousarray(data)

    eri_mat = cp.asarray(eri_mat, dtype=fp_dtype)
    eri_mat = cp.ascontiguousarray(eri_mat)
    if eri_mat.ndim != 2 or eri_mat.shape[0] != eri_mat.shape[1]:
        raise ValueError("eri_mat must have shape (nops,nops)")

    nrows = int(indptr.size) - 1
    nops = int(eri_mat.shape[0])
    if nrows < 0:
        raise ValueError("indptr must have shape (nrows+1,)")

    if g_out is None:
        g_out = cp.empty((nrows, nops), dtype=fp_dtype)
    else:
        g_out = cp.asarray(g_out, dtype=fp_dtype)
        g_out = cp.ascontiguousarray(g_out)
        if g_out.shape != (nrows, nops):
            raise ValueError("g_out must have shape (nrows,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.kernel3_build_g_from_csr_eri_mat_inplace_device(
        indptr,
        indices,
        data,
        eri_mat,
        g_out,
        int(threads),
        float(half),
        int(stream_ptr),
        bool(sync),
    )

    return g_out


def kernel3_build_g_from_csr_eri_mat_range_inplace_device(
    indptr,
    indices,
    data,
    *,
    row_start: int,
    nrows: int,
    eri_mat,
    g_out=None,
    threads: int = 256,
    half: float = 0.5,
    stream=None,
    sync: bool = True,
):
    """Range-based device-array path for Kernel 3 build-g (avoids slicing indices/data in Python).

    Computes `g_out[row_local,pq] = half * sum_rs ERI_mat[pq,rs] * C[row_start+row_local,rs]` for `row_local=0..nrows-1`,
    where `C` is given in CSR by global arrays `(indptr, indices, data)`.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device-array Kernel 3 range path") from e

    row_start = int(row_start)
    nrows = int(nrows)
    if row_start < 0 or nrows < 0:
        raise ValueError("row_start and nrows must be >= 0")

    indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
    indptr = cp.ascontiguousarray(indptr)
    indices = cp.asarray(indices, dtype=cp.int32).ravel()
    indices = cp.ascontiguousarray(indices)
    data = cp.asarray(data).ravel()
    if data.dtype not in (cp.float32, cp.float64):
        data = cp.asarray(data, dtype=cp.float64).ravel()
    data = cp.ascontiguousarray(data)

    eri_mat = cp.asarray(eri_mat)
    fp_dtype = cp.float32 if eri_mat.dtype == cp.float32 else cp.float64
    if data.dtype != fp_dtype:
        data = cp.asarray(data, dtype=fp_dtype).ravel()
        data = cp.ascontiguousarray(data)

    eri_mat = cp.asarray(eri_mat, dtype=fp_dtype)
    eri_mat = cp.ascontiguousarray(eri_mat)
    if eri_mat.ndim != 2 or eri_mat.shape[0] != eri_mat.shape[1]:
        raise ValueError("eri_mat must have shape (nops,nops)")

    nops = int(eri_mat.shape[0])
    if g_out is None:
        g_out = cp.empty((nrows, nops), dtype=fp_dtype)
    else:
        g_out = cp.asarray(g_out, dtype=fp_dtype)
        g_out = cp.ascontiguousarray(g_out)
        if g_out.shape != (nrows, nops):
            raise ValueError("g_out must have shape (nrows,nops)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.kernel3_build_g_from_csr_eri_mat_range_inplace_device(
        indptr,
        indices,
        data,
        int(row_start),
        int(nrows),
        eri_mat,
        g_out,
        int(threads),
        float(half),
        int(stream_ptr),
        bool(sync),
    )
    return g_out


class Kernel3BuildGWorkspace:
    """Persistent cuBLAS workspace for Kernel 3 build-g (CSR -> dense -> GEMMEx).

    This is intended for experimenting with CUDA 13.x FP64 fixed-point emulation on the
    dense ERI_mat contraction step.
    """

    def __init__(self, nops: int, *, max_nrows: int, dtype=None, gemm_backend: str | None = None) -> None:
        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array Kernel 3 GEMMEx path") from e

        fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
        if fp_dtype not in (cp.float32, cp.float64):
            raise ValueError("dtype must be float32 or float64")

        self._ws = _ext.Kernel3BuildGWorkspace(int(nops), int(max_nrows))
        self._dtype = fp_dtype
        backend = str(gemm_backend).strip() if gemm_backend is not None else ("gemmex_fp64" if fp_dtype == cp.float64 else "gemmex_fp32")
        self._ws.set_gemm_backend(backend)

    @property
    def nops(self) -> int:
        return int(self._ws.nops)

    @property
    def max_nrows(self) -> int:
        return int(self._ws.max_nrows)

    @property
    def dtype(self):
        return self._dtype

    def cublas_emulation_info(self) -> dict[str, object]:
        return dict(self._ws.cublas_emulation_info())

    def set_cublas_math_mode(self, mode: str) -> None:
        self._ws.set_cublas_math_mode(str(mode))

    def set_cublas_workspace_bytes(self, bytes_: int) -> None:
        self._ws.set_cublas_workspace_bytes(int(bytes_))

    def cublas_workspace_bytes(self) -> int:
        return int(self._ws.cublas_workspace_bytes())

    def autoset_cublas_workspace_bytes(
        self,
        *,
        nrows: int | None = None,
        cap_mb: int = 2048,
    ) -> int:
        """Auto-size cuBLAS workspace for emulated-FP64 GEMMEx (safe bound; capped).

        This is primarily intended for experimenting with fixed-point emulation performance.
        """

        from asuka.cuda.cublas_workspace import recommend_cublas_workspace_bytes_for_emulated_fp64_gemm

        nrows_eff = int(self.max_nrows) if nrows is None else int(nrows)
        if nrows_eff < 0 or nrows_eff > int(self.max_nrows):
            raise ValueError("nrows must be in [0, max_nrows]")
        cap_bytes = int(cap_mb) * 1024 * 1024
        ws_info = self.cublas_emulation_info()
        rec = recommend_cublas_workspace_bytes_for_emulated_fp64_gemm(
            ws_info=ws_info,
            gemm_shapes=[(int(self.nops), int(nrows_eff), int(self.nops))],
            batch_count=1,
            is_complex=False,
            cap_bytes=int(cap_bytes),
        )
        self.set_cublas_workspace_bytes(int(rec))
        return int(rec)

    def gemm_backend(self) -> str:
        return str(self._ws.gemm_backend())

    def set_gemm_backend(self, backend: str) -> None:
        self._ws.set_gemm_backend(str(backend))

    def gemm_algo(self) -> int:
        return int(self._ws.gemm_algo())

    def set_gemm_algo(self, algo: int) -> None:
        self._ws.set_gemm_algo(int(algo))

    def set_cublas_emulation_strategy(self, strategy: str) -> None:
        self._ws.set_cublas_emulation_strategy(str(strategy))

    def set_cublas_emulation_special_values_support(self, mask: int) -> None:
        self._ws.set_cublas_emulation_special_values_support(int(mask))

    def set_cublas_fixed_point_mantissa_control(self, control: str) -> None:
        self._ws.set_cublas_fixed_point_mantissa_control(str(control))

    def set_cublas_fixed_point_max_mantissa_bits(self, max_bits: int) -> None:
        self._ws.set_cublas_fixed_point_max_mantissa_bits(int(max_bits))

    def set_cublas_fixed_point_mantissa_bit_offset(self, bit_offset: int) -> None:
        self._ws.set_cublas_fixed_point_mantissa_bit_offset(int(bit_offset))

    def build_g_from_csr_eri_mat_inplace_device(
        self,
        indptr,
        indices,
        data,
        eri_mat,
        *,
        g_out=None,
        dtype=None,
        threads: int = 256,
        half: float = 0.5,
        stream=None,
        sync: bool = True,
    ):
        """Build `g_out[row,pq] = half * sum_rs ERI_mat[pq,rs] * C[row,rs]` via GEMMEx.

        All array-like inputs/outputs must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
        Returns `g_out` (device array) so callers can reuse it across calls.
        """

        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array Kernel 3 GEMMEx path") from e

        fp_dtype = self._dtype if dtype is None else cp.dtype(dtype)
        if fp_dtype not in (cp.float32, cp.float64):
            raise ValueError("dtype must be float32 or float64")
        backend = self.gemm_backend()
        if fp_dtype == cp.float32 and backend in ("gemmex_fp64", "cublaslt_fp64", "gemmex_emulated_fixedpoint"):
            raise ValueError("float32 inputs require gemm_backend in {gemmex_fp32, gemmex_tf32, cublaslt_fp32, cublaslt_tf32}")
        if fp_dtype == cp.float64 and backend in ("gemmex_fp32", "gemmex_tf32", "cublaslt_fp32", "cublaslt_tf32"):
            raise ValueError("float64 inputs require gemm_backend in {gemmex_fp64, cublaslt_fp64, gemmex_emulated_fixedpoint}")

        indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
        indptr = cp.ascontiguousarray(indptr)
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        indices = cp.ascontiguousarray(indices)
        data = cp.asarray(data, dtype=fp_dtype).ravel()
        data = cp.ascontiguousarray(data)

        eri_mat = cp.asarray(eri_mat, dtype=fp_dtype)
        eri_mat = cp.ascontiguousarray(eri_mat)
        if eri_mat.ndim != 2 or eri_mat.shape[0] != eri_mat.shape[1]:
            raise ValueError("eri_mat must have shape (nops,nops)")
        if int(eri_mat.shape[0]) != int(self.nops):
            raise ValueError("eri_mat has wrong nops for this workspace")

        nrows = int(indptr.size) - 1
        nops = int(self.nops)
        if nrows < 0:
            raise ValueError("indptr must have shape (nrows+1,)")
        if nrows > int(self.max_nrows):
            raise ValueError("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows")

        if g_out is None:
            g_out = cp.empty((nrows, nops), dtype=fp_dtype)
        else:
            g_out = cp.asarray(g_out, dtype=fp_dtype)
            g_out = cp.ascontiguousarray(g_out)
            if g_out.shape != (nrows, nops):
                raise ValueError("g_out must have shape (nrows,nops)")

        if stream is None:
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
        else:
            stream_ptr = int(getattr(stream, "ptr", stream))

        self._ws.build(
            indptr,
            indices,
            data,
            eri_mat,
            g_out,
            int(threads),
            float(half),
            int(stream_ptr),
            bool(sync),
        )
        return g_out

    def gemm_w_eri_mat_inplace_device(
        self,
        w_dense,
        eri_mat,
        *,
        g_out=None,
        dtype=None,
        half: float = 0.5,
        stream=None,
        sync: bool = True,
    ):
        """Compute `g_out = half * w_dense @ eri_mat` via GEMMEx (optionally emulated FP64).

        This is a dense-device helper to reuse the same cuBLAS handle/configuration as the Kernel 3 build-g workspace.
        All arrays must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
        """

        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array GEMMEx path") from e

        fp_dtype = self._dtype if dtype is None else cp.dtype(dtype)
        if fp_dtype not in (cp.float32, cp.float64):
            raise ValueError("dtype must be float32 or float64")
        backend = self.gemm_backend()
        if fp_dtype == cp.float32 and backend in ("gemmex_fp64", "cublaslt_fp64", "gemmex_emulated_fixedpoint"):
            raise ValueError("float32 inputs require gemm_backend in {gemmex_fp32, gemmex_tf32, cublaslt_fp32, cublaslt_tf32}")
        if fp_dtype == cp.float64 and backend in ("gemmex_fp32", "gemmex_tf32", "cublaslt_fp32", "cublaslt_tf32"):
            raise ValueError("float64 inputs require gemm_backend in {gemmex_fp64, cublaslt_fp64, gemmex_emulated_fixedpoint}")

        w_dense = cp.asarray(w_dense, dtype=fp_dtype)
        w_dense = cp.ascontiguousarray(w_dense)
        if w_dense.ndim != 2 or int(w_dense.shape[1]) != int(self.nops):
            raise ValueError("w_dense must have shape (nrows,nops)")
        nrows = int(w_dense.shape[0])

        eri_mat = cp.asarray(eri_mat, dtype=fp_dtype)
        eri_mat = cp.ascontiguousarray(eri_mat)
        if eri_mat.ndim != 2 or eri_mat.shape[0] != eri_mat.shape[1]:
            raise ValueError("eri_mat must have shape (nops,nops)")
        if int(eri_mat.shape[0]) != int(self.nops):
            raise ValueError("eri_mat has wrong nops for this workspace")

        if g_out is None:
            g_out = cp.empty((nrows, int(self.nops)), dtype=fp_dtype)
        else:
            g_out = cp.asarray(g_out, dtype=fp_dtype)
            g_out = cp.ascontiguousarray(g_out)
            if g_out.shape != (nrows, int(self.nops)):
                raise ValueError("g_out must have shape (nrows,nops)")

        if stream is None:
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
        else:
            stream_ptr = int(getattr(stream, "ptr", stream))

        if not hasattr(self._ws, "gemm_dense"):
            raise RuntimeError("CUDA extension is missing Kernel3BuildGWorkspace.gemm_dense; rebuild the extension")

        self._ws.gemm_dense(
            eri_mat,
            w_dense,
            g_out,
            float(half),
            int(stream_ptr),
            bool(sync),
        )
        return g_out


class Kernel3BuildGDFWorkspace:
    """Persistent cuBLAS workspace for Kernel 3 DF build-g (CSR -> dense -> GEMMEx).

    Given CSR C[row,rs] and DF factors L_full[rs,L], build:

        W[row,L] = sum_rs C[row,rs] * L_full[rs,L]
        g[row,pq] = half * sum_L W[row,L] * L_full[pq,L]

    This is intended for experimenting with CUDA 13.x FP64 fixed-point emulation on the
    GEMM-heavy DF contraction steps.
    """

    def __init__(self, nops: int, naux: int, *, max_nrows: int) -> None:
        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
        self._ws = _ext.Kernel3BuildGDFWorkspace(int(nops), int(naux), int(max_nrows))

    @property
    def nops(self) -> int:
        return int(self._ws.nops)

    @property
    def naux(self) -> int:
        return int(self._ws.naux)

    @property
    def max_nrows(self) -> int:
        return int(self._ws.max_nrows)

    def cublas_emulation_info(self) -> dict[str, object]:
        return dict(self._ws.cublas_emulation_info())

    def set_cublas_math_mode(self, mode: str) -> None:
        self._ws.set_cublas_math_mode(str(mode))

    def set_cublas_workspace_bytes(self, bytes_: int) -> None:
        self._ws.set_cublas_workspace_bytes(int(bytes_))

    def cublas_workspace_bytes(self) -> int:
        return int(self._ws.cublas_workspace_bytes())

    def autoset_cublas_workspace_bytes(
        self,
        *,
        nrows: int | None = None,
        cap_mb: int = 2048,
    ) -> int:
        """Auto-size cuBLAS workspace for emulated-FP64 GEMMEx (safe bound; capped)."""

        from asuka.cuda.cublas_workspace import recommend_cublas_workspace_bytes_for_emulated_fp64_gemm

        nrows_eff = int(self.max_nrows) if nrows is None else int(nrows)
        if nrows_eff < 0 or nrows_eff > int(self.max_nrows):
            raise ValueError("nrows must be in [0, max_nrows]")
        cap_bytes = int(cap_mb) * 1024 * 1024
        ws_info = self.cublas_emulation_info()
        rec = recommend_cublas_workspace_bytes_for_emulated_fp64_gemm(
            ws_info=ws_info,
            gemm_shapes=[
                (int(self.nops), int(nrows_eff), int(self.naux)),  # G^T = L @ W^T
            ],
            batch_count=1,
            is_complex=False,
            cap_bytes=int(cap_bytes),
        )
        self.set_cublas_workspace_bytes(int(rec))
        return int(rec)

    def gemm_backend(self) -> str:
        return str(self._ws.gemm_backend())

    def set_gemm_backend(self, backend: str) -> None:
        self._ws.set_gemm_backend(str(backend))

    def gemm_algo(self) -> int:
        return int(self._ws.gemm_algo())

    def set_gemm_algo(self, algo: int) -> None:
        self._ws.set_gemm_algo(int(algo))

    def set_cublas_emulation_strategy(self, strategy: str) -> None:
        self._ws.set_cublas_emulation_strategy(str(strategy))

    def set_cublas_emulation_special_values_support(self, mask: int) -> None:
        self._ws.set_cublas_emulation_special_values_support(int(mask))

    def set_cublas_fixed_point_mantissa_control(self, control: str) -> None:
        self._ws.set_cublas_fixed_point_mantissa_control(str(control))

    def set_cublas_fixed_point_max_mantissa_bits(self, max_bits: int) -> None:
        self._ws.set_cublas_fixed_point_max_mantissa_bits(int(max_bits))

    def set_cublas_fixed_point_mantissa_bit_offset(self, bit_offset: int) -> None:
        self._ws.set_cublas_fixed_point_mantissa_bit_offset(int(bit_offset))

    def gemm_w_l_full_inplace_device(
        self,
        w_dense,
        l_full,
        *,
        g_out=None,
        half: float = 0.5,
        stream=None,
        sync: bool = True,
    ):
        """Compute `g_out = half * (w_dense @ l_full) @ l_full^T` via GEMMEx (optionally emulated FP64).

        All arrays must live on the GPU and support `__cuda_array_interface__` (e.g. CuPy arrays).
        """

        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array DF GEMMEx path") from e

        w_dense = cp.asarray(w_dense)
        fp_dtype = cp.float32 if w_dense.dtype == cp.float32 else cp.float64
        w_dense = cp.ascontiguousarray(cp.asarray(w_dense, dtype=fp_dtype))
        if w_dense.ndim != 2 or int(w_dense.shape[1]) != int(self.nops):
            raise ValueError("w_dense must have shape (nrows,nops)")
        nrows = int(w_dense.shape[0])

        l_full = cp.ascontiguousarray(cp.asarray(l_full, dtype=fp_dtype))
        if l_full.ndim != 2:
            raise ValueError("l_full must be 2D with shape (nops,naux)")
        if (int(l_full.shape[0]), int(l_full.shape[1])) != (int(self.nops), int(self.naux)):
            raise ValueError("l_full has wrong shape for this workspace")

        if g_out is None:
            g_out = cp.empty((nrows, int(self.nops)), dtype=fp_dtype)
        else:
            g_out = cp.ascontiguousarray(cp.asarray(g_out, dtype=fp_dtype))
            if g_out.shape != (nrows, int(self.nops)):
                raise ValueError("g_out must have shape (nrows,nops)")

        if stream is None:
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
        else:
            stream_ptr = int(getattr(stream, "ptr", stream))

        if not hasattr(self._ws, "gemm_dense"):
            raise RuntimeError("CUDA extension is missing Kernel3BuildGDFWorkspace.gemm_dense; rebuild the extension")

        self._ws.gemm_dense(
            l_full,
            w_dense,
            g_out,
            float(half),
            int(stream_ptr),
            bool(sync),
        )
        return g_out

    def build_g_from_csr_l_full_inplace_device(
        self,
        indptr,
        indices,
        data,
        l_full,
        *,
        g_out=None,
        threads: int = 256,
        half: float = 0.5,
        stream=None,
        sync: bool = True,
    ):
        """Build `g_out[row,pq]` from CSR `C[row,rs]` and DF factors `L_full[rs,L]` via GEMMEx."""

        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array Kernel 3 DF GEMMEx path") from e

        indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
        indptr = cp.ascontiguousarray(indptr)
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        indices = cp.ascontiguousarray(indices)
        data = cp.asarray(data).ravel()
        if data.dtype not in (cp.float32, cp.float64):
            data = cp.asarray(data, dtype=cp.float64).ravel()
        data = cp.ascontiguousarray(data)

        l_full = cp.asarray(l_full)
        fp_dtype = cp.float32 if l_full.dtype == cp.float32 else cp.float64
        l_full = cp.ascontiguousarray(cp.asarray(l_full, dtype=fp_dtype))
        if l_full.ndim != 2:
            raise ValueError("l_full must be 2D with shape (nops,naux)")
        if (int(l_full.shape[0]), int(l_full.shape[1])) != (int(self.nops), int(self.naux)):
            raise ValueError("l_full has wrong shape for this workspace")

        nrows = int(indptr.size) - 1
        if nrows < 0:
            raise ValueError("indptr must have shape (nrows+1,)")
        if nrows > int(self.max_nrows):
            raise ValueError("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows")

        if g_out is None:
            g_out = cp.empty((nrows, int(self.nops)), dtype=fp_dtype)
        else:
            g_out = cp.ascontiguousarray(cp.asarray(g_out, dtype=fp_dtype))
            if g_out.shape != (nrows, int(self.nops)):
                raise ValueError("g_out must have shape (nrows,nops)")

        if stream is None:
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
        else:
            stream_ptr = int(getattr(stream, "ptr", stream))

        self._ws.build(
            indptr,
            indices,
            data,
            l_full,
            g_out,
            int(threads),
            float(half),
            int(stream_ptr),
            bool(sync),
        )
        return g_out

    def build_g_from_csr_l_full_range_inplace_device(
        self,
        indptr,
        indices,
        data,
        *,
        row_start: int,
        nrows: int,
        l_full,
        g_out=None,
        threads: int = 256,
        half: float = 0.5,
        stream=None,
        sync: bool = True,
    ):
        """Range-based DF build-g (avoids slicing indices/data in Python).

        Computes `g_out[row_local,pq] = half * sum_L (sum_rs C[row,rs] * L_full[rs,L]) * L_full[pq,L]` for
        `row = row_start + row_local` and `row_local=0..nrows-1`, where the CSR matrix `C` is given by global arrays
        `(indptr, indices, data)`.
        """

        if _ext is None:
            raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
        if not hasattr(self._ws, "build_range"):
            raise RuntimeError(
                "CUDA extension is missing Kernel3BuildGDFWorkspace.build_range; rebuild with python -m asuka.build.guga_cuda_ext"
            )

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for the device-array Kernel 3 DF range path") from e

        row_start = int(row_start)
        nrows = int(nrows)
        if row_start < 0 or nrows < 0:
            raise ValueError("row_start and nrows must be >= 0")
        if nrows > int(self.max_nrows):
            raise ValueError("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows")

        indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
        indptr = cp.ascontiguousarray(indptr)
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        indices = cp.ascontiguousarray(indices)
        data = cp.asarray(data).ravel()
        if data.dtype not in (cp.float32, cp.float64):
            data = cp.asarray(data, dtype=cp.float64).ravel()
        data = cp.ascontiguousarray(data)

        l_full = cp.asarray(l_full)
        fp_dtype = cp.float32 if l_full.dtype == cp.float32 else cp.float64
        l_full = cp.ascontiguousarray(cp.asarray(l_full, dtype=fp_dtype))
        if l_full.ndim != 2:
            raise ValueError("l_full must be 2D with shape (nops,naux)")
        if (int(l_full.shape[0]), int(l_full.shape[1])) != (int(self.nops), int(self.naux)):
            raise ValueError("l_full has wrong shape for this workspace")

        if g_out is None:
            g_out = cp.empty((nrows, int(self.nops)), dtype=fp_dtype)
        else:
            g_out = cp.ascontiguousarray(cp.asarray(g_out, dtype=fp_dtype))
            if g_out.shape != (nrows, int(self.nops)):
                raise ValueError("g_out must have shape (nrows,nops)")

        if stream is None:
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
        else:
            stream_ptr = int(getattr(stream, "ptr", stream))

        self._ws.build_range(
            indptr,
            indices,
            data,
            int(row_start),
            int(nrows),
            l_full,
            g_out,
            int(threads),
            float(half),
            int(stream_ptr),
            bool(sync),
        )
        return g_out


def kernel4_apply_csr_eri_mat_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    row_j,
    row_k,
    indptr,
    indices,
    data,
    eri_mat,
    x,
    *,
    g_workspace=None,
    y=None,
    overflow=None,
    max_g_bytes: int = 256 * 1024 * 1024,
    half: float = 0.5,
    threads_g: int = 256,
    threads_apply: int = 32,
    zero_y: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
):
    """Kernel 4 baseline: `CSR C[row,rs] -> g_out[row,pq] -> apply/scatter` on the GPU (ERI_mat path).

    This consumes the CSR output of Kernel 2.5 (`row_j,row_k,indptr,indices,data`) and applies the 2-body product
    contribution:

      for each row=(j,k): y += x[j] * (sum_pq g[pq] E_pq |k>)

    Notes
    -----
    - This is a baseline / integration helper; it still expects the CSR arrays on the host (NumPy) and uploads them.
    - `ERI_mat` and `x` may be NumPy or CuPy arrays; they are converted to contiguous device arrays once.
    - If `g_workspace` is provided, it must be a `Kernel3BuildGWorkspace` and is used for the build-g stage (CSR->dense->GEMMEx).
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    row_j = np.asarray(row_j, dtype=np.int32).ravel()
    row_k = np.asarray(row_k, dtype=np.int32).ravel()
    indptr = np.asarray(indptr, dtype=np.int64).ravel()
    indices = np.asarray(indices, dtype=np.int32).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()

    if row_j.shape != row_k.shape:
        raise ValueError("row_j and row_k must have the same shape")
    nrows = int(row_j.size)
    if indptr.shape != (nrows + 1,):
        raise ValueError("indptr must have shape (nrows+1,)")
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")
    if nrows == 0:
        if y is None:
            y = cp.zeros((int(drt.ncsf),), dtype=cp.float64)
        else:
            y = cp.asarray(y, dtype=cp.float64).ravel()
            y = cp.ascontiguousarray(y)
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)
        return y, overflow

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    eri_mat = cp.asarray(eri_mat, dtype=cp.float64)
    eri_mat = cp.ascontiguousarray(eri_mat)
    if eri_mat.shape != (nops, nops):
        raise ValueError("eri_mat must have shape (norb*norb, norb*norb)")

    x = cp.asarray(x, dtype=cp.float64).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=cp.float64)
    else:
        y = cp.asarray(y, dtype=cp.float64).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream = cp.cuda.get_current_stream()

    max_g_bytes = int(max_g_bytes)
    if max_g_bytes <= 0:
        raise ValueError("max_g_bytes must be > 0")
    bytes_per_row = nops * np.dtype(np.float64).itemsize
    nrows_block = max(1, int(max_g_bytes // max(1, bytes_per_row)))
    nrows_block = min(nrows_block, nrows)

    g_buf = cp.empty((nrows_block, nops), dtype=cp.float64)

    # Upload CSR + row metadata once; per-block we slice on device.
    row_j_d = cp.asarray(row_j, dtype=cp.int32)
    row_k_d = cp.asarray(row_k, dtype=cp.int32)
    indptr_d = cp.asarray(indptr, dtype=cp.int64)
    indices_d = cp.asarray(indices, dtype=cp.int32)
    data_d = cp.asarray(data, dtype=cp.float64)

    is_first_block = True
    for row_start in range(0, nrows, nrows_block):
        row_stop = min(nrows, row_start + nrows_block)
        nb = int(row_stop - row_start)

        base = int(indptr[row_start])
        end = int(indptr[row_stop])

        indptr_b = indptr_d[row_start : row_stop + 1] - int(base)
        indices_b = indices_d[base:end]
        data_b = data_d[base:end]

        row_j_b = row_j_d[row_start:row_stop]
        row_k_b = row_k_d[row_start:row_stop]

        g_b = g_buf[:nb]
        if g_workspace is None:
            kernel3_build_g_from_csr_eri_mat_inplace_device(
                indptr_b,
                indices_b,
                data_b,
                eri_mat,
                g_out=g_b,
                threads=int(threads_g),
                half=float(half),
                stream=stream,
                sync=False,
            )
        else:
            g_workspace.build_g_from_csr_eri_mat_inplace_device(
                indptr_b,
                indices_b,
                data_b,
                eri_mat,
                g_out=g_b,
                threads=int(threads_g),
                half=float(half),
                stream=stream,
                sync=False,
            )

        task_scale = cp.ascontiguousarray(x[row_j_b])
        apply_g_flat_scatter_atomic_inplace_device(
            drt,
            drt_dev,
            state_dev,
            row_k_b,
            g_b,
            task_scale=task_scale,
            y=y,
            overflow=overflow,
            threads=int(threads_apply),
            zero_y=bool(zero_y and is_first_block),
            stream=stream,
            sync=bool(check_overflow),
            check_overflow=bool(check_overflow),
        )
        is_first_block = False

    if sync:
        stream.synchronize()

    return y, overflow


def kernel4_apply_csr_eri_mat_device_csr_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    row_j,
    row_k,
    indptr,
    indices,
    data,
    eri_mat,
    x,
    *,
    epq_table=None,
    g_workspace=None,
    g_buf=None,
    task_scale_buf=None,
    y=None,
    overflow=None,
    max_g_bytes: int = 256 * 1024 * 1024,
    half: float = 0.5,
    threads_g: int = 256,
    threads_apply: int = 32,
    zero_y: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    profile: dict[str, float] | None = None,
):
    """Kernel 4 baseline: device CSR -> g_out -> apply/scatter on the GPU (ERI_mat path).

    This is the fully device-resident variant of `kernel4_apply_csr_eri_mat_inplace_device`:
    it expects `row_j,row_k,indptr,indices,data` to already live on the GPU.
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")
    if g_workspace is not None and not sync:
        raise ValueError("g_workspace requires sync=True (uses host scalar reads for CSR slicing)")

    row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
    row_j = cp.ascontiguousarray(row_j)
    row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
    row_k = cp.ascontiguousarray(row_k)
    indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
    indptr = cp.ascontiguousarray(indptr)
    indices = cp.asarray(indices, dtype=cp.int32).ravel()
    indices = cp.ascontiguousarray(indices)
    data = cp.asarray(data).ravel()
    if data.dtype not in (cp.float32, cp.float64):
        data = cp.asarray(data, dtype=cp.float64).ravel()
    data = cp.ascontiguousarray(data)

    eri_mat = cp.asarray(eri_mat)
    fp_dtype = cp.float32 if eri_mat.dtype == cp.float32 else cp.float64
    if data.dtype != fp_dtype:
        data = cp.asarray(data, dtype=fp_dtype).ravel()
        data = cp.ascontiguousarray(data)

    if row_j.shape != row_k.shape:
        raise ValueError("row_j and row_k must have the same shape")
    nrows = int(row_j.size)
    if indptr.shape != (nrows + 1,):
        raise ValueError("indptr must have shape (nrows+1,)")
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")
    if nrows == 0:
        if y is None:
            y = cp.zeros((int(drt.ncsf),), dtype=fp_dtype)
        else:
            y = cp.asarray(y, dtype=fp_dtype).ravel()
            y = cp.ascontiguousarray(y)
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)
        return y, overflow

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    eri_mat = cp.asarray(eri_mat, dtype=fp_dtype)
    eri_mat = cp.ascontiguousarray(eri_mat)
    if eri_mat.shape != (nops, nops):
        raise ValueError("eri_mat must have shape (norb*norb, norb*norb)")

    x = cp.asarray(x, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.asarray(y, dtype=fp_dtype).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream = cp.cuda.get_current_stream()

    max_g_bytes = int(max_g_bytes)
    if max_g_bytes <= 0:
        raise ValueError("max_g_bytes must be > 0")
    bytes_per_row = nops * np.dtype(fp_dtype).itemsize
    nrows_block_req = max(1, int(max_g_bytes // max(1, bytes_per_row)))
    nrows_block_req = min(nrows_block_req, nrows)

    if g_buf is not None:
        g_buf = cp.asarray(g_buf, dtype=fp_dtype)
        if not getattr(g_buf, "flags", None) or not g_buf.flags.c_contiguous:
            g_buf = cp.ascontiguousarray(g_buf)
        if g_buf.ndim != 2 or int(g_buf.shape[1]) != nops:
            raise ValueError("g_buf must have shape (>=1,nops)")
        nrows_block = min(int(g_buf.shape[0]), nrows_block_req)
        if nrows_block <= 0:
            raise ValueError("g_buf must have at least 1 row")
    else:
        nrows_block = nrows_block_req
        g_buf = cp.empty((nrows_block, nops), dtype=fp_dtype)

    if task_scale_buf is not None:
        task_scale_buf = cp.asarray(task_scale_buf, dtype=fp_dtype).ravel()
        task_scale_buf = cp.ascontiguousarray(task_scale_buf)
        if int(task_scale_buf.size) < int(nrows_block):
            raise ValueError("task_scale_buf must have at least nrows_block elements")

    is_first_block = True
    for row_start in range(0, nrows, nrows_block):
        row_stop = min(nrows, row_start + nrows_block)
        nb = int(row_stop - row_start)

        row_j_b = row_j[row_start:row_stop]
        row_k_b = row_k[row_start:row_stop]

        g_b = g_buf[:nb]
        if g_workspace is None:
            t0 = time.perf_counter() if profile is not None else None
            kernel3_build_g_from_csr_eri_mat_range_inplace_device(
                indptr,
                indices,
                data,
                row_start=int(row_start),
                nrows=int(nb),
                eri_mat=eri_mat,
                g_out=g_b,
                threads=int(threads_g),
                half=float(half),
                stream=stream,
                sync=False,
            )
            if profile is not None and t0 is not None:
                stream.synchronize()
                profile["kernel3_s"] = profile.get("kernel3_s", 0.0) + (time.perf_counter() - t0)
        else:
            base = int(indptr[int(row_start)].get())
            end = int(indptr[int(row_stop)].get())
            indptr_b = indptr[int(row_start) : int(row_stop) + 1] - int(base)
            indices_b = indices[int(base) : int(end)]
            data_b = data[int(base) : int(end)]
            t0 = time.perf_counter() if profile is not None else None
            g_workspace.build_g_from_csr_eri_mat_inplace_device(
                indptr_b,
                indices_b,
                data_b,
                eri_mat,
                g_out=g_b,
                dtype=fp_dtype,
                threads=int(threads_g),
                half=float(half),
                stream=stream,
                sync=False,
            )
            if profile is not None and t0 is not None:
                stream.synchronize()
                profile["kernel3_s"] = profile.get("kernel3_s", 0.0) + (time.perf_counter() - t0)

        if task_scale_buf is None:
            task_scale = cp.ascontiguousarray(x[row_j_b])
        else:
            task_scale = cp.take(x, row_j_b, out=task_scale_buf[:nb])
        t0 = time.perf_counter() if profile is not None else None
        apply_g_flat_scatter_atomic_inplace_device(
            drt,
            drt_dev,
            state_dev,
            row_k_b,
            g_b,
            task_scale=task_scale,
            epq_table=epq_table,
            dtype=fp_dtype,
            y=y,
            overflow=overflow,
            threads=int(threads_apply),
            zero_y=bool(zero_y and is_first_block),
            stream=stream,
            sync=bool(check_overflow),
            check_overflow=bool(check_overflow),
        )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["kernel1a_s"] = profile.get("kernel1a_s", 0.0) + (time.perf_counter() - t0)
        is_first_block = False

    if sync:
        stream.synchronize()

    return y, overflow


def kernel4_apply_csr_l_full_device_csr_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    row_j,
    row_k,
    indptr,
    indices,
    data,
    l_full,
    x,
    *,
    epq_table=None,
    gdf_workspace=None,
    g_buf=None,
    task_scale_buf=None,
    y=None,
    overflow=None,
    max_g_bytes: int = 256 * 1024 * 1024,
    half: float = 0.5,
    threads_g: int = 256,
    threads_apply: int = 32,
    zero_y: bool = True,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    profile: dict[str, float] | None = None,
):
    """Kernel 4 DF path: device CSR -> g_out via DF GEMM -> apply/scatter.

    This mirrors :func:`kernel4_apply_csr_eri_mat_device_csr_inplace_device`, but replaces the dense
    `eri_mat` build-g stage with a DF contraction using `L_full`:

        W[row,L] = sum_rs C[row,rs] * L_full[rs,L]
        g[row,pq] = half * sum_L W[row,L] * L_full[pq,L]
    """

    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the device matvec path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
    row_j = cp.ascontiguousarray(row_j)
    row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
    row_k = cp.ascontiguousarray(row_k)
    indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
    indptr = cp.ascontiguousarray(indptr)
    indices = cp.asarray(indices, dtype=cp.int32).ravel()
    indices = cp.ascontiguousarray(indices)
    l_full = cp.asarray(l_full)
    fp_dtype = cp.float32 if l_full.dtype == cp.float32 else cp.float64
    data = cp.asarray(data).ravel()
    if fp_dtype == cp.float64:
        data = cp.asarray(data, dtype=cp.float64).ravel()
    elif data.dtype not in (cp.float32, cp.float64):
        data = cp.asarray(data, dtype=cp.float64).ravel()
    data = cp.ascontiguousarray(data)

    if row_j.shape != row_k.shape:
        raise ValueError("row_j and row_k must have the same shape")
    nrows = int(row_j.size)
    if indptr.shape != (nrows + 1,):
        raise ValueError("indptr must have shape (nrows+1,)")
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")
    if nrows == 0:
        if y is None:
            y = cp.zeros((int(drt.ncsf),), dtype=fp_dtype)
        else:
            y = cp.asarray(y, dtype=fp_dtype).ravel()
            y = cp.ascontiguousarray(y)
        if overflow is None:
            overflow = cp.zeros((1,), dtype=cp.int32)
        return y, overflow

    nops = int(drt.norb) * int(drt.norb)
    if nops <= 0:
        raise ValueError("invalid nops")

    l_full = cp.ascontiguousarray(cp.asarray(l_full, dtype=fp_dtype))
    if l_full.ndim != 2 or int(l_full.shape[0]) != nops:
        raise ValueError("l_full must have shape (norb*norb, naux)")
    naux = int(l_full.shape[1])

    x = cp.asarray(x, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.asarray(y, dtype=fp_dtype).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream = cp.cuda.get_current_stream()

    max_g_bytes = int(max_g_bytes)
    if max_g_bytes <= 0:
        raise ValueError("max_g_bytes must be > 0")
    # Estimate scratch usage per CSR row in the DF path:
    # - g_buf: [nrows_block, nops]
    # - Kernel3BuildGDFWorkspace: W^T [naux, nrows_block] (stored in a [nrows_block, naux] allocation)
    bytes_per_row = (nops + naux) * int(np.dtype(fp_dtype).itemsize)
    nrows_block_req = max(1, int(max_g_bytes // max(1, bytes_per_row)))
    nrows_block_req = min(nrows_block_req, nrows)

    if g_buf is not None:
        g_buf = cp.asarray(g_buf, dtype=fp_dtype)
        if not getattr(g_buf, "flags", None) or not g_buf.flags.c_contiguous:
            g_buf = cp.ascontiguousarray(g_buf)
        if g_buf.ndim != 2 or int(g_buf.shape[1]) != nops:
            raise ValueError("g_buf must have shape (>=1,nops)")
        nrows_block = min(int(g_buf.shape[0]), nrows_block_req)
        if nrows_block <= 0:
            raise ValueError("g_buf must have at least 1 row")
    else:
        nrows_block = nrows_block_req
        g_buf = cp.empty((nrows_block, nops), dtype=fp_dtype)

    if task_scale_buf is not None:
        task_scale_buf = cp.asarray(task_scale_buf, dtype=fp_dtype).ravel()
        task_scale_buf = cp.ascontiguousarray(task_scale_buf)
        if int(task_scale_buf.size) < int(nrows_block):
            raise ValueError("task_scale_buf must have at least nrows_block elements")

    if gdf_workspace is None:
        gdf_workspace = Kernel3BuildGDFWorkspace(int(nops), int(naux), max_nrows=int(nrows_block))
        if fp_dtype == cp.float32:
            gdf_workspace.set_gemm_backend("gemmex_fp32")
            gdf_workspace.set_cublas_math_mode("default")
    else:
        if not isinstance(gdf_workspace, Kernel3BuildGDFWorkspace):
            raise TypeError("gdf_workspace must be a Kernel3BuildGDFWorkspace")
        if int(getattr(gdf_workspace, "nops", -1)) != int(nops) or int(getattr(gdf_workspace, "naux", -1)) != int(naux):
            raise ValueError("gdf_workspace has wrong (nops,naux) for this call")
        if int(getattr(gdf_workspace, "max_nrows", 0)) < int(nrows_block):
            raise ValueError("gdf_workspace.max_nrows is too small; recreate with a larger max_nrows")

    is_first_block = True
    for row_start in range(0, nrows, nrows_block):
        row_stop = min(nrows, row_start + nrows_block)
        nb = int(row_stop - row_start)

        row_j_b = row_j[row_start:row_stop]
        row_k_b = row_k[row_start:row_stop]

        g_b = g_buf[:nb]
        t0 = time.perf_counter() if profile is not None else None
        gdf_workspace.build_g_from_csr_l_full_range_inplace_device(
            indptr,
            indices,
            data,
            row_start=int(row_start),
            nrows=int(nb),
            l_full=l_full,
            g_out=g_b,
            threads=int(threads_g),
            half=float(half),
            stream=stream,
            sync=False,
        )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["kernel3_s"] = profile.get("kernel3_s", 0.0) + (time.perf_counter() - t0)
            profile["kernel3_df_s"] = profile.get("kernel3_df_s", 0.0) + (time.perf_counter() - t0)

        if task_scale_buf is None:
            task_scale = cp.ascontiguousarray(x[row_j_b])
        else:
            task_scale = cp.take(x, row_j_b, out=task_scale_buf[:nb])
        t0 = time.perf_counter() if profile is not None else None
        apply_g_flat_scatter_atomic_inplace_device(
            drt,
            drt_dev,
            state_dev,
            row_k_b,
            g_b,
            task_scale=task_scale,
            epq_table=epq_table,
            y=y,
            overflow=overflow,
            threads=int(threads_apply),
            zero_y=bool(zero_y and is_first_block),
            stream=stream,
            sync=bool(check_overflow),
            check_overflow=bool(check_overflow),
            dtype=fp_dtype,
        )
        if profile is not None and t0 is not None:
            stream.synchronize()
            profile["kernel1a_s"] = profile.get("kernel1a_s", 0.0) + (time.perf_counter() - t0)
        is_first_block = False

    if sync:
        stream.synchronize()

    return y, overflow


def kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device(
    drt: DRT,
    drt_dev,
    state_dev,
    epq_table,
    row_j,
    row_k,
    indptr,
    indices,
    data,
    eri_mat_t,
    x,
    *,
    row_start: int = 0,
    nrows: int = -1,
    y=None,
    overflow=None,
    threads: int = 32,
    zero_y: bool = True,
    half: float = 0.5,
    stream=None,
    sync: bool = True,
    check_overflow: bool = True,
    use_kahan: bool = False,
):
    """Fused Kernel4 (ERI_mat + epq_table): `CSR -> g_flat(shared) -> apply/scatter`.

    This eliminates the intermediate `g_buf` traffic by building `g_flat` in shared memory per CSR row and applying it
    immediately using the combined `E_pq` action table.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the fused Kernel4 path") from e

    if check_overflow and not sync:
        raise ValueError("check_overflow=True requires sync=True")

    if not isinstance(epq_table, tuple) or len(epq_table) != 4:
        raise TypeError("epq_table must be a 4-tuple (indptr, indices, pq_ids, data)")
    epq_indptr, epq_indices, epq_pq, epq_data = epq_table
    epq_indptr = cp.asarray(epq_indptr, dtype=cp.int64).ravel()
    epq_indptr = cp.ascontiguousarray(epq_indptr)
    epq_indices = cp.asarray(epq_indices, dtype=cp.int32).ravel()
    epq_indices = cp.ascontiguousarray(epq_indices)
    epq_pq = _as_epq_pq_array(cp, epq_pq, name="epq_pq")
    _validate_epq_pq_capacity(cp, epq_pq, norb=int(drt.norb), name="epq_pq")
    if epq_indices.shape != epq_pq.shape:
        raise ValueError("epq_table arrays must have the same shape for (indices, pq_ids)")
    if epq_indptr.shape != (int(drt.ncsf) + 1,):
        raise ValueError("epq_table indptr must have shape (ncsf+1,)")

    row_j = cp.asarray(row_j, dtype=cp.int32).ravel()
    row_j = cp.ascontiguousarray(row_j)
    row_k = cp.asarray(row_k, dtype=cp.int32).ravel()
    row_k = cp.ascontiguousarray(row_k)
    indptr = cp.asarray(indptr, dtype=cp.int64).ravel()
    indptr = cp.ascontiguousarray(indptr)
    indices = cp.asarray(indices, dtype=cp.int32).ravel()
    indices = cp.ascontiguousarray(indices)
    # Note: CSR data dtype is determined by x dtype (fp_dtype) below; defer cast until fp_dtype is known.

    if row_j.shape != row_k.shape:
        raise ValueError("row_j and row_k must have the same shape")
    nrows_total = int(row_j.size)
    if indptr.shape != (nrows_total + 1,):
        raise ValueError("indptr must have shape (nrows_total+1,)")

    row_start = int(row_start)
    nrows = int(nrows)
    if row_start < 0:
        raise ValueError("row_start must be >= 0")
    if nrows < 0:
        nrows = nrows_total - row_start
    if row_start + nrows > nrows_total:
        raise ValueError("row_start+nrows exceeds number of CSR rows")

    nops = int(drt.norb) * int(drt.norb)
    x = cp.asarray(x).ravel()
    fp_dtype = cp.float32 if x.dtype == cp.float32 else cp.float64
    x = cp.asarray(x, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)
    if x.shape != (int(drt.ncsf),):
        raise ValueError("x must have shape (ncsf,)")

    # Cast CSR data to match workspace dtype (fp_dtype).
    data = cp.asarray(data, dtype=fp_dtype).ravel()
    data = cp.ascontiguousarray(data)
    if indices.shape != data.shape:
        raise ValueError("indices and data must have the same shape")

    epq_data = cp.asarray(epq_data, dtype=fp_dtype).ravel()
    epq_data = cp.ascontiguousarray(epq_data)
    if epq_indices.shape != epq_pq.shape or epq_indices.shape != epq_data.shape:
        raise ValueError("epq_table arrays must have the same shape for (indices, pq_ids, data)")

    eri_mat_t = cp.asarray(eri_mat_t, dtype=fp_dtype)
    eri_mat_t = cp.ascontiguousarray(eri_mat_t)
    if eri_mat_t.shape != (nops, nops):
        raise ValueError("eri_mat_t must have shape (norb*norb, norb*norb)")

    if y is None:
        y = cp.empty((int(drt.ncsf),), dtype=fp_dtype)
    else:
        y = cp.asarray(y, dtype=fp_dtype).ravel()
        y = cp.ascontiguousarray(y)
        if y.shape != (int(drt.ncsf),):
            raise ValueError("y must have shape (ncsf,)")

    if overflow is None:
        overflow = cp.empty((1,), dtype=cp.int32)
    else:
        overflow = cp.asarray(overflow, dtype=cp.int32).ravel()
        overflow = cp.ascontiguousarray(overflow)
        if overflow.shape != (1,):
            raise ValueError("overflow must have shape (1,)")

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device(
        drt_dev,
        state_dev,
        epq_indptr,
        epq_indices,
        epq_pq,
        epq_data,
        row_j,
        row_k,
        indptr,
        indices,
        data,
        int(row_start),
        int(nrows),
        eri_mat_t,
        x,
        y,
        overflow,
        int(threads),
        bool(zero_y),
        float(half),
        int(stream_ptr),
        bool(sync),
        bool(check_overflow),
        bool(use_kahan),
    )
    return y, overflow


# ============================================================================
# Fused MRCI kernels (P1C, P2, P4)
# ============================================================================


def has_sym_pair_pack_device() -> bool:
    return _ext is not None and hasattr(_ext, "sym_pair_pack_inplace_device")


def sym_pair_pack_inplace_device(
    W,
    W_pair,
    pair_pq,
    pair_qp,
    *,
    nrows: int,
    nops: int,
    npair: int,
    is_f32: bool = False,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """Pack W into symmetric-pair layout on GPU.

    W_pair[k, u] = W[k, pair_pq[u]] + W[k, pair_qp[u]], halved on diagonal.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.sym_pair_pack_inplace_device(
        W, W_pair, pair_pq, pair_qp,
        int(nrows), int(nops), int(npair),
        bool(is_f32), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def sym_pair_unpack_inplace_device(
    G_pair,
    G,
    full_to_pair,
    *,
    nrows: int,
    nops: int,
    npair: int,
    is_f32: bool = False,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """Unpack symmetric-pair G into full layout on GPU.

    G[k, pq] = G_pair[k, full_to_pair[pq]].
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.sym_pair_unpack_inplace_device(
        G_pair, G, full_to_pair,
        int(nrows), int(nops), int(npair),
        bool(is_f32), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def has_sym_pair_fused_kernels() -> bool:
    """Check if pair-indexed W-build/Apply/W-diag kernels are available."""
    return (
        _ext is not None
        and hasattr(_ext, "build_w_pair_from_epq_transpose_range_inplace_device")
        and hasattr(_ext, "apply_g_pair_gather_epq_transpose_range_inplace_device")
        and hasattr(_ext, "build_w_diag_pair_from_steps_inplace_device")
    )


def rdm12_reorder_delta_inplace_device(
    dm1_pq,
    gram0,
    *,
    dm1_out=None,
    dm2_out=None,
    norb: int,
    threads: int = 256,
    stream=None,
    sync: bool = True,
):
    """Reorder dm1/dm2 from EPQ layout + apply delta correction, all on GPU.

    dm1_out[q,p] = dm1_pq[p*norb+q]
    dm2_out[p,q,r,s] = gram0[r*norb+s, p*norb+q] - delta(q,r)*dm1_pq[s*norb+p]
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    nops = int(norb) * int(norb)
    dm1_pq = cp.ascontiguousarray(cp.asarray(dm1_pq, dtype=cp.float64).ravel())
    gram0 = cp.ascontiguousarray(cp.asarray(gram0, dtype=cp.float64).reshape(nops, nops))

    if dm1_out is None:
        dm1_out = cp.empty((norb, norb), dtype=cp.float64)
    if dm2_out is None:
        dm2_out = cp.empty((norb, norb, norb, norb), dtype=cp.float64)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.rdm12_reorder_delta_inplace_device(
        dm1_pq, gram0, dm1_out, dm2_out,
        int(norb), int(stream_ptr), int(threads), bool(sync),
    )
    return dm1_out, dm2_out


def scatter_embed_batched_inplace_device(
    x_sub,
    sub_to_full,
    x_full,
    *,
    threads: int = 128,
    stream=None,
    sync: bool = False,
):
    """Batched scatter embed: x_full[sub_to_full[i], :] = x_sub[i, :] for all vectors."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    x_sub = cp.ascontiguousarray(cp.asarray(x_sub, dtype=cp.float64))
    sub_to_full = cp.asarray(sub_to_full, dtype=cp.int64).ravel()
    x_full = cp.asarray(x_full, dtype=cp.float64)

    if x_sub.ndim != 2:
        raise ValueError("x_sub must be 2D (nsub, nvec)")
    nsub, nvec = x_sub.shape
    sub_stride = int(x_sub.strides[0] // 8)
    full_stride = int(x_full.strides[0] // 8) if x_full.ndim >= 2 else nvec

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.scatter_embed_batched_inplace_device(
        x_sub, sub_to_full, x_full,
        int(nvec), int(sub_stride), int(full_stride),
        int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def gather_project_batched_inplace_device(
    y_full,
    sub_to_full,
    y_sub,
    *,
    threads: int = 128,
    stream=None,
    sync: bool = False,
):
    """Batched gather project: y_sub[i, :] = y_full[sub_to_full[i], :] for all vectors."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    y_full = cp.asarray(y_full, dtype=cp.float64)
    sub_to_full = cp.asarray(sub_to_full, dtype=cp.int64).ravel()
    y_sub = cp.ascontiguousarray(cp.asarray(y_sub, dtype=cp.float64))

    if y_sub.ndim != 2:
        raise ValueError("y_sub must be 2D (nsub, nvec)")
    nsub, nvec = y_sub.shape
    full_stride = int(y_full.strides[0] // 8) if y_full.ndim >= 2 else nvec
    sub_stride = int(y_sub.strides[0] // 8)

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.gather_project_batched_inplace_device(
        y_full, sub_to_full, y_sub,
        int(nvec), int(full_stride), int(sub_stride),
        int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


# ============================================================================
# P1A: Batched W-build + diagonal
# ============================================================================


def has_build_w_batched_device() -> bool:
    return _ext is not None and hasattr(_ext, "build_w_from_epq_table_batched_inplace_device")


def build_w_from_epq_table_batched_inplace_device(
    epq_table,
    x,
    *,
    ncsf: int,
    norb: int,
    w_out,
    nvec: int,
    v_start: int,
    k_start: int,
    k_count: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """Batched off-diagonal W-build from EPQ table for nvec vectors.

    W_out[v*k_count + k_local, pq] += coef * x[j, v_start+v] for all EPQ entries.
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    epq_indptr, epq_indices, epq_pq, epq_data = epq_table

    nops = int(norb) * int(norb)

    # Ensure int64 indptr for the batched kernel
    epq_indptr = cp.asarray(epq_indptr, dtype=cp.int64).ravel()
    epq_indptr = cp.ascontiguousarray(epq_indptr)
    epq_indices = cp.ascontiguousarray(cp.asarray(epq_indices, dtype=cp.int32).ravel())
    epq_pq = cp.ascontiguousarray(cp.asarray(epq_pq, dtype=cp.int32).ravel())
    epq_data = cp.ascontiguousarray(cp.asarray(epq_data, dtype=cp.float64).ravel())

    # x stride (number of columns in the full multivector array)
    if x.ndim == 2:
        x_stride = int(x.strides[0] // 8)
    else:
        x_stride = 1

    # w_out stride
    if w_out.ndim == 2:
        w_stride = int(w_out.strides[0] // 8)
    else:
        w_stride = nops

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_from_epq_table_batched_inplace_device(
        epq_indptr, epq_indices, epq_pq, epq_data,
        x, int(x_stride),
        w_out, int(w_stride),
        int(ncsf), int(nops),
        int(nvec), int(v_start),
        int(k_start), int(k_count),
        int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def build_w_diag_batched_inplace_device(
    state_dev,
    x,
    *,
    w_out,
    ncsf: int,
    norb: int,
    j_start: int,
    j_count: int,
    nvec: int,
    v_start: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """Batched diagonal W-build for nvec vectors.

    W_out[v*j_count + j_local, r*norb+r] = occ(j,r) * x[j, v_start+v].
    """
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    steps_table = state_dev.steps_table_dev
    nops = int(norb) * int(norb)

    if x.ndim == 2:
        x_stride = int(x.strides[0] // 8)
    else:
        x_stride = 1

    if w_out.ndim == 2:
        w_stride = int(w_out.strides[0] // 8)
    else:
        w_stride = nops

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.build_w_diag_batched_inplace_device(
        steps_table, x, int(x_stride),
        w_out, int(w_stride),
        int(ncsf), int(norb),
        int(j_start), int(j_count),
        int(nvec), int(v_start),
        int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


# ============================================================================
# P5: RDM123 delta correction + symmetry
# ============================================================================


def has_rdm123_reorder_device() -> bool:
    return _ext is not None and hasattr(_ext, "dm2_delta_correction_inplace_device")


def dm2_delta_correction_inplace_device(
    dm2,
    dm1,
    *,
    n: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """In-place dm2 delta: dm2[:, k, k, :] -= dm1 for all k."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.dm2_delta_correction_inplace_device(
        dm2, dm1, int(n), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def dm2_4way_symmetry_inplace_device(
    dm2_in,
    dm2_out,
    *,
    n: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """4-way dm2 symmetry: dm2_out = 0.25*(dm2 + 3 permutations)."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.dm2_4way_symmetry_inplace_device(
        dm2_in, dm2_out, int(n), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def dm3_delta_correction_inplace_device(
    dm3,
    dm2,
    dm1,
    *,
    n: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """In-place dm3 delta: 4 correction terms using post-symmetry dm2."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.dm3_delta_correction_inplace_device(
        dm3, dm2, dm1, int(n), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()


def dm3_6way_symmetry_inplace_device(
    dm3_in,
    dm3_out,
    *,
    n: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
):
    """6-way dm3 pair-permutation symmetry."""
    if _ext is None:
        raise RuntimeError("CUDA extension not available")

    import cupy as cp

    if stream is None:
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
    else:
        stream_ptr = int(getattr(stream, "ptr", stream))

    _ext.dm3_6way_symmetry_inplace_device(
        dm3_in, dm3_out, int(n), int(stream_ptr), int(threads),
    )
    if sync:
        cp.cuda.get_current_stream().synchronize()
