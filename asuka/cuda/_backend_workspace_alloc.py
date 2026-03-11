from __future__ import annotations


def build_workspace_cache_state(
    *,
    cp,
    cache_csr_tiles: bool,
    j_tile: int,
    ncsf: int,
    csr_capacity_mult: float,
    rs_n_pairs: int | None,
    norb: int,
    csr_data_dtype,
) -> dict[str, object]:
    """Initialize reusable cache-state containers and optional tile scratch buffers."""
    out: dict[str, object] = {
        "_csr_single_tile_cache": None,
        "_csr_tile_cache": {},
        "_csr_host_tile_cache": {},
        "_csr_host_cache_bytes": 0,
        "_csr_host_cache_hits": 0,
        "_csr_host_cache_misses": 0,
        "_csr_host_cache_store_attempts": 0,
        "_csr_host_cache_store_accepts": 0,
        "_csr_host_cache_evictions": 0,
        "_epq_apply_tile_cache": {},
        "_epq_apply_cache_bytes": 0,
        "_epq_apply_cache_hits": 0,
        "_epq_apply_cache_misses": 0,
        "_epq_apply_staging_capacity": 0,
        "_epq_apply_staging_indptr": None,
        "_epq_apply_staging_indices": None,
        "_epq_apply_staging_pq_ids": None,
        "_epq_apply_staging_data": None,
        "_csr_pipeline_slots": [],
        "_csr_pipeline_apply_stream": None,
        "_tile_csr_row_j": None,
        "_tile_csr_row_k": None,
        "_tile_csr_indptr": None,
        "_tile_csr_indices": None,
        "_tile_csr_data": None,
        "_tile_csr_overflow": None,
        "_tile_csr_capacity": 0,
    }

    if bool(cache_csr_tiles) and int(j_tile) < int(ncsf):
        n_pairs = int(rs_n_pairs) if rs_n_pairs is not None else int(norb) * (int(norb) - 1)
        tile_cap = int(max(1.0, float(csr_capacity_mult)) * float(int(j_tile)) * float(n_pairs))
        out["_tile_csr_row_j"] = cp.empty((int(tile_cap),), dtype=cp.int32)
        out["_tile_csr_row_k"] = cp.empty((int(tile_cap),), dtype=cp.int32)
        out["_tile_csr_indptr"] = cp.empty((int(tile_cap) + 1,), dtype=cp.int64)
        out["_tile_csr_indices"] = cp.empty((int(tile_cap),), dtype=cp.int32)
        out["_tile_csr_data"] = cp.empty((int(tile_cap),), dtype=csr_data_dtype)
        out["_tile_csr_overflow"] = cp.empty((1,), dtype=cp.int32)
        out["_tile_csr_capacity"] = int(tile_cap)

    return out


def make_occ_buf_dtype(*, cp, dtype, j_tile: int, norb: int):
    """Pre-allocate occupancy buffer in workspace dtype when not float64."""
    if cp.dtype(dtype) == cp.dtype(cp.float64):
        return None
    return cp.empty((int(j_tile), int(norb)), dtype=dtype)


def resolve_workspace_nrows_block(
    *,
    max_g_bytes: int,
    nops: int,
    itemsize: int,
    ncsf: int,
    eri_mat_present: bool,
    l_full_present: bool,
    naux: int,
) -> int:
    """Estimate row-block size for temporary g buffer, capped by CI dimension."""
    bytes_per_row = int(nops) * int(itemsize)
    if (not bool(eri_mat_present)) and bool(l_full_present):
        bytes_per_row += int(naux) * int(itemsize)
    nrows_block = max(1, int(max_g_bytes) // max(1, int(bytes_per_row)))
    try:
        return max(1, min(int(nrows_block), int(ncsf)))
    except Exception:
        return int(max(1, int(nrows_block)))
