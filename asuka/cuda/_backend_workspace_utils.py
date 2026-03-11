from __future__ import annotations

from typing import Any


def csr_cache_ready(
    *,
    j_tile: int,
    ncsf: int,
    cache_csr_tiles: bool,
    csr_single_tile_cache: Any,
    csr_tile_cache: dict[int, Any],
) -> bool:
    if int(j_tile) >= int(ncsf):
        return csr_single_tile_cache is not None
    if not bool(cache_csr_tiles):
        return False
    jt = int(j_tile)
    ntiles = (int(ncsf) + jt - 1) // jt
    if int(len(csr_tile_cache)) != int(ntiles):
        return False
    for j0 in range(0, int(ncsf), int(j_tile)):
        if int(j0) not in csr_tile_cache:
            return False
    return True


def release_tile_csr_scratch(workspace: Any) -> None:
    workspace._tile_csr_row_j = None
    workspace._tile_csr_row_k = None
    workspace._tile_csr_indptr = None
    workspace._tile_csr_indices = None
    workspace._tile_csr_data = None
    workspace._tile_csr_overflow = None
    workspace._tile_csr_capacity = 0


def estimate_object_nbytes(obj: Any, seen: set[int]) -> int:
    if obj is None:
        return 0
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    nbytes = 0
    arr_nbytes = getattr(obj, "nbytes", None)
    if arr_nbytes is not None:
        try:
            return int(arr_nbytes)
        except Exception:
            return 0
    if isinstance(obj, dict):
        for v in obj.values():
            nbytes += estimate_object_nbytes(v, seen)
        return int(nbytes)
    if isinstance(obj, (list, tuple)):
        for v in obj:
            nbytes += estimate_object_nbytes(v, seen)
        return int(nbytes)
    return 0


def workspace_nbytes_estimate_from_dict(state: dict[str, Any]) -> int:
    seen: set[int] = set()
    total = 0
    for name, value in state.items():
        if str(name).startswith("__"):
            continue
        total += estimate_object_nbytes(value, seen)
    return int(max(0, total))


def release_workspace_resources(workspace: Any) -> int:
    """Release retained buffers/caches for a workspace object; idempotent."""
    if bool(getattr(workspace, "_released", False)):
        return 0

    freed_est = int(workspace.workspace_nbytes_estimate())

    workspace._cuda_graph = None
    workspace._cuda_graph_x = None
    workspace._cuda_graph_y = None

    for slot in list(getattr(workspace, "_csr_pipeline_slots", [])):
        try:
            ws = slot.get("ws", None)
            if ws is not None and hasattr(ws, "release"):
                ws.release()
        except Exception:
            pass
    workspace._csr_pipeline_slots = []
    workspace._csr_pipeline_apply_stream = None

    for ws_name in ("_k25_ws", "_offdiag_gemm_ws", "_gdf_ws"):
        try:
            ws = getattr(workspace, ws_name, None)
            if ws is not None and hasattr(ws, "release"):
                ws.release()
        except Exception:
            pass
        setattr(workspace, ws_name, None)

    for name in (
        "_g_buf",
        "_task_scale_rows",
        "_diag_g_cache",
        "_g_diag_buf",
        "_diag_w_buf",
        "_occ_buf",
        "_occ_buf_dtype",
        "_w_offdiag",
        "_w_block",
        "_l_full_t",
        "_offdiag_df_t",
        "_eri_diag_t",
        "_eri_mat_t",
        "_eri_mat_t_cache",
        "_task_scale_j",
        "_overflow_w",
        "overflow_apply",
        "_csr_row_j",
        "_csr_row_k",
        "_csr_indptr",
        "_csr_indices",
        "_csr_data",
        "_csr_overflow",
        "_csr_single_tile_cache",
        "_csr_tile_cache",
        "_csr_host_tile_cache",
        "_tile_csr_row_j",
        "_tile_csr_row_k",
        "_tile_csr_indptr",
        "_tile_csr_indices",
        "_tile_csr_data",
        "_tile_csr_overflow",
        "_epq_table",
        "_epq_apply_tile_cache",
        "_epq_apply_staging_indptr",
        "_epq_apply_staging_indices",
        "_epq_apply_staging_pq_ids",
        "_epq_apply_staging_data",
        "task_csf_all",
        "eri_mat",
        "l_full",
        "h_eff_flat",
        "_rs_r_d",
        "_rs_s_d",
    ):
        if hasattr(workspace, name):
            setattr(workspace, name, None)

    workspace._csr_host_cache_bytes = 0
    workspace._epq_apply_cache_bytes = 0
    workspace._epq_apply_staging_capacity = 0
    workspace._tile_csr_capacity = 0
    workspace._released = True
    return int(freed_est)
