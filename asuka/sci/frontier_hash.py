from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuda.cuda_backend import (
    GugaMatvecEriMatWorkspace,
    Kernel3BuildGDFWorkspace,
    apply_g_flat_scatter_atomic_frontier_hash_inplace_device,
    apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device,
    build_occ_block_from_steps_inplace_device,
    cipsi_frontier_hash_clear_inplace_device,
    cipsi_score_and_select_topk_from_hash_slots_inplace_device,
    cipsi_score_and_select_topk_from_hash_slots_v2_inplace_device,
    kernel3_build_g_from_csr_eri_mat_range_inplace_device,
)


def _require_cupy():
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GPU frontier-hash selection requires cupy") from e
    return cp


@contextmanager
def _nvtx_range(name: str, *, enabled: bool):
    if not enabled:
        yield
        return
    cp = _require_cupy()
    cp.cuda.nvtx.RangePush(str(name))
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


@contextmanager
def _timed_phase(name: str, *, enabled: bool, timings_ms: dict[str, float], stream):
    if not enabled:
        yield
        return
    cp = _require_cupy()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record(stream)
    try:
        yield
    finally:
        ev1.record(stream)
        ev1.synchronize()
        dt = float(cp.cuda.get_elapsed_time(ev0, ev1))
        timings_ms[str(name)] = float(timings_ms.get(str(name), 0.0) + dt)


def _mem_snapshot(cp: Any) -> dict[str, int]:
    cp.cuda.runtime.deviceSynchronize()
    pool = cp.get_default_memory_pool()
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return {
        "driver_used_bytes": int(total_b - free_b),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
    }


def _round_up_pow2(x: int) -> int:
    x = int(x)
    if x <= 1:
        return 1
    out = 1
    while out < x:
        out <<= 1
    return int(out)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be an integer") from e


@dataclass
class FrontierHashStats:
    hash_cap: int
    nnz_out: int
    overflow_retries: int
    timings_ms: dict[str, float]
    memory: dict[str, Any]


class FrontierHashSelector:
    """Build a sparse external frontier on GPU via a hash map, then score/PT2 + select top-k.

    This bypasses any EPQ table materialization and avoids building dense H|c blocks.
    """

    def __init__(
        self,
        drt: DRT,
        ws: GugaMatvecEriMatWorkspace,
        *,
        nroots: int,
        hdiag,
        denom_floor: float,
        tile: int = 1024,
        rs_block: int = 0,
        g_rows: int | None = 0,
        hash_cap: int | None = None,
        max_retries: int = 8,
        offdiag_kernel_mode: str | None = None,
        g_rows_cap_mb: int | None = None,
        nvtx: bool | None = None,
        profile: bool = False,
    ) -> None:
        cp = _require_cupy()
        self.cp = cp
        self.drt = drt
        self.ws = ws
        self.nroots = int(nroots)
        if self.nroots < 1:
            raise ValueError("nroots must be >= 1")
        self.ncsf = int(drt.ncsf)
        self.norb = int(drt.norb)
        self.nops = int(self.norb) * int(self.norb)
        self.denom_floor = float(denom_floor)
        self.max_retries = int(max_retries)
        if self.max_retries < 1:
            self.max_retries = 1

        if bool(getattr(ws, "use_epq_table", False)):
            raise RuntimeError("FrontierHashSelector requires a workspace with use_epq_table=False")

        # NVTX is opt-in via env var or explicit arg.
        if nvtx is None:
            env_nvtx = str(os.environ.get("ASUKA_NVTX", "0")).strip().lower()
            self.nvtx_enabled = env_nvtx in ("1", "true", "yes", "on")
        else:
            self.nvtx_enabled = bool(nvtx)
        self.profile_enabled = bool(profile)

        mode_in = str(
            offdiag_kernel_mode
            if offdiag_kernel_mode is not None
            else os.environ.get("ASUKA_FH_OFFDIAG_KERNEL", "auto")
        ).strip().lower()
        if mode_in in ("", "default"):
            mode_in = "auto"
        if mode_in not in ("auto", "legacy", "warp_coop", "fused_eri"):
            raise ValueError("offdiag_kernel_mode must be one of: auto, legacy, warp_coop, fused_eri")
        # For now, warp_coop/fused_eri share the same high-level scheduling policy.
        self.offdiag_kernel_mode = "legacy" if mode_in == "legacy" else "auto"
        self.offdiag_kernel_mode_raw = mode_in
        grouping_in = str(os.environ.get("ASUKA_FH_OFFDIAG_GROUPING", "auto")).strip().lower()
        if grouping_in in ("", "auto"):
            grouping_in = "tile" if self.offdiag_kernel_mode == "legacy" else "chunk"
        if grouping_in not in ("tile", "chunk"):
            raise ValueError("ASUKA_FH_OFFDIAG_GROUPING must be one of: auto, tile, chunk")
        self.offdiag_grouping = grouping_in
        score_impl = str(os.environ.get("ASUKA_CIPSI_HASH_SELECT_IMPL", "v1")).strip().lower()
        if score_impl in ("", "default", "auto"):
            score_impl = "v1"
        if score_impl not in ("v1", "v2"):
            raise ValueError("ASUKA_CIPSI_HASH_SELECT_IMPL must be one of: v1, v2")
        self.hash_select_impl = score_impl
        agg_single_root = str(os.environ.get("ASUKA_FH_OFFDIAG_AGG_SINGLE_ROOT", "1")).strip().lower()
        self.offdiag_single_root_aggregate = agg_single_root in ("1", "true", "yes", "on")
        self.offdiag_agg_min_rows = max(1, int(_env_int("ASUKA_FH_OFFDIAG_AGG_MIN_ROWS", 256)))

        # Tile sizing: ensure (tile * rs_block) fits Kernel25Workspace.max_tasks.
        tile_i = int(tile)
        if tile_i <= 0:
            tile_i = 1024
        tile_i = min(tile_i, int(getattr(ws, "j_tile", tile_i)))
        tile_i = max(1, tile_i)
        rs_block_env = _env_int("ASUKA_FH_RS_BLOCK", int(rs_block))
        rs_block_i = int(rs_block_env)

        k25_ws = getattr(ws, "_k25_ws", None)
        if k25_ws is None:
            raise RuntimeError("Kernel25Workspace is unavailable on CUDA workspace (required for frontier-hash)")
        max_tasks = int(getattr(k25_ws, "max_tasks", 0))
        if max_tasks <= 0:
            raise RuntimeError("Kernel25Workspace.max_tasks is unavailable (required for frontier-hash)")
        self.max_tasks = int(max_tasks)

        rs_r_d = getattr(ws, "_rs_r_d", None)
        rs_s_d = getattr(ws, "_rs_s_d", None)
        if rs_r_d is None or rs_s_d is None:
            raise RuntimeError("internal error: workspace missing rs pair tables")
        n_pairs = int(rs_r_d.size)
        if rs_block_i <= 0:
            rs_block_i = 128 if self.offdiag_kernel_mode == "legacy" else max(1, n_pairs)
        rs_block_i = min(rs_block_i, max(1, n_pairs))
        if int(tile_i) * int(rs_block_i) > int(max_tasks):
            tile_i = max(1, int(max_tasks // max(1, rs_block_i)))

        threads_apply_offdiag = int(getattr(ws, "threads_apply", 256))
        if self.offdiag_kernel_mode != "legacy":
            threads_apply_cap = _env_int("ASUKA_FH_THREADS_APPLY_MAX", 64)
            if threads_apply_cap > 0:
                threads_apply_offdiag = min(threads_apply_offdiag, int(threads_apply_cap))

        self.tile = int(tile_i)
        self.rs_block = int(rs_block_i)
        self.rs_r_d = rs_r_d
        self.rs_s_d = rs_s_d
        self.n_pairs = int(n_pairs)
        self.threads_apply_offdiag = int(max(1, threads_apply_offdiag))
        j_chunk_cap = max(1, int(self.max_tasks // max(1, self.rs_block)))
        j_chunk_env = _env_int("ASUKA_FH_OFFDIAG_J_CHUNK", 0)
        if j_chunk_env > 0:
            j_chunk_cap = min(int(j_chunk_cap), int(j_chunk_env))
        self.offdiag_j_chunk = int(max(1, j_chunk_cap))

        # Selected mask for skipping internal entries during PT2/selection.
        self.selected_mask_d = cp.zeros((self.ncsf,), dtype=cp.uint8)

        # Diagonal Hamiltonian (device).
        hdiag_d = cp.asarray(hdiag, dtype=cp.float64).ravel()
        hdiag_d = cp.ascontiguousarray(hdiag_d)
        if hdiag_d.shape != (self.ncsf,):
            raise ValueError("hdiag must have shape (ncsf,)")
        self.hdiag_d = hdiag_d

        # Two-body diagonal rs (r==s) ERI columns: eri_diag_t[r,pq] = eri[pq, rr].
        diag_ids = cp.asarray([int(r) * int(self.norb) + int(r) for r in range(int(self.norb))], dtype=cp.int32)
        l_full = getattr(ws, "l_full", None)
        eri_mat = getattr(ws, "eri_mat", None)
        if l_full is not None:
            l_diag = l_full[diag_ids]
            eri_diag_t = cp.ascontiguousarray(cp.dot(l_diag, l_full.T))
        elif eri_mat is not None:
            eri_diag_t = cp.ascontiguousarray(eri_mat[:, diag_ids].T.copy())
        else:
            raise RuntimeError("internal error: workspace missing both eri_mat and l_full")
        self.eri_diag_t = eri_diag_t

        # Scratch buffers reused across calls.
        self.occ_buf = cp.empty((int(self.tile), int(self.norb)), dtype=cp.float64)
        self.x_full = cp.empty((int(self.ncsf),), dtype=cp.float64)
        self.task_csf_buf = cp.empty((int(self.max_tasks),), dtype=cp.int32)
        self.task_p_buf = cp.empty((int(self.max_tasks),), dtype=cp.int32)
        self.task_q_buf = cp.empty((int(self.max_tasks),), dtype=cp.int32)

        # Offdiag build-g staging buffer.
        g_rows_env = _env_int("ASUKA_FH_G_ROWS", int(0 if g_rows is None else g_rows))
        g_rows_eff = int(g_rows_env)
        if g_rows_eff <= 0:
            if self.offdiag_kernel_mode == "legacy":
                g_rows_eff = 4096
            else:
                cap_mb_eff = int(g_rows_cap_mb) if g_rows_cap_mb is not None else _env_int("ASUKA_FH_G_ROWS_CAP_MB", 128)
                cap_mb_eff = max(1, cap_mb_eff)
                bytes_per_row = max(1, int(self.nops) * 8)
                rows_from_cap = (int(cap_mb_eff) * (1 << 20)) // bytes_per_row
                g_rows_eff = int(rows_from_cap)
                g_rows_eff = max(4096, g_rows_eff)
                g_rows_eff = min(16384, g_rows_eff)
                # Align row blocks to the same granularity used by CSR tile helpers.
                g_rows_eff = max(256, (int(g_rows_eff) // 256) * 256)
        g_rows_eff = max(256, int(g_rows_eff))
        # Stay within workspace CSR capacity if available.
        g_rows_eff = min(g_rows_eff, int(getattr(ws, "_csr_capacity", g_rows_eff)))
        self.g_rows = int(g_rows_eff)
        self.g_rows_cap_mb = int(g_rows_cap_mb) if g_rows_cap_mb is not None else int(_env_int("ASUKA_FH_G_ROWS_CAP_MB", 128))
        self.g_buf = cp.empty((int(self.g_rows), int(self.nops)), dtype=cp.float64)
        self.g_scaled_buf = cp.empty((int(self.g_rows), int(self.nops)), dtype=cp.float64)
        # Reusable per-row coefficient staging for offdiag many-roots apply.
        self.task_scale_rows_buf = cp.empty((int(self.g_rows), int(self.nroots)), dtype=cp.float64)
        # Reusable per-row coefficient staging for nroots==1 single-root apply.
        self.task_scale_rows1_buf = cp.empty((int(self.g_rows),), dtype=cp.float64)

        self.gdf_ws: Kernel3BuildGDFWorkspace | None = None
        if l_full is not None:
            naux = int(l_full.shape[1])
            self.gdf_ws = Kernel3BuildGDFWorkspace(int(self.nops), int(naux), max_nrows=int(self.g_rows))

        # Hash buffers (allocated lazily / resized on demand).
        self._hash_cap_hint = int(hash_cap) if hash_cap is not None else 0
        self.hash_cap = 0
        self.hash_keys = None
        self.hash_vals = None
        self.hash_overflow = None
        self.out_idx = None
        self.out_vals = None
        self.out_nnz = None

    def reset_selected_mask(self, sel_idx: np.ndarray) -> None:
        cp = self.cp
        sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32).ravel()
        cp.cuda.runtime.memsetAsync(
            int(self.selected_mask_d.data.ptr),
            0,
            int(self.selected_mask_d.size),
            int(cp.cuda.get_current_stream().ptr),
        )
        if sel_idx_i32.size:
            self.selected_mask_d[cp.asarray(sel_idx_i32, dtype=cp.int32)] = np.uint8(1)

    def mark_selected(self, new_idx: list[int] | np.ndarray) -> None:
        cp = self.cp
        if new_idx is None:
            return
        new_i32 = np.asarray(new_idx, dtype=np.int32).ravel()
        if new_i32.size:
            self.selected_mask_d[cp.asarray(new_i32, dtype=cp.int32)] = np.uint8(1)

    def _aggregate_single_root_offdiag_rows(self, row_k, g_rows, task_scale_rows):
        """Scale rows by coefficients and merge duplicate target CSFs (single-root path)."""
        cp = self.cp
        nb = int(row_k.shape[0])
        if nb <= 0:
            return row_k, g_rows

        g_scaled = self.g_scaled_buf[:nb]
        g_scaled[...] = g_rows
        g_scaled *= task_scale_rows[:, None]

        if nb < int(self.offdiag_agg_min_rows):
            return row_k, g_scaled

        order = cp.argsort(row_k)
        row_k_sorted = row_k[order]
        g_sorted = g_scaled[order]
        if nb <= 1:
            return row_k_sorted, g_sorted

        heads = cp.empty((nb,), dtype=cp.bool_)
        heads[0] = True
        heads[1:] = row_k_sorted[1:] != row_k_sorted[:-1]
        starts = cp.nonzero(heads)[0]
        if int(starts.size) == nb:
            return row_k_sorted, g_sorted

        row_k_agg = row_k_sorted[starts]
        g_agg = cp.add.reduceat(g_sorted, starts, axis=0)
        return row_k_agg, cp.ascontiguousarray(g_agg)

    def _choose_hash_cap(self, *, nsel: int) -> int:
        cap_in = int(self._hash_cap_hint or self.hash_cap or 0)
        if cap_in <= 0:
            # Heuristic: scale with selected count; cap at ncsf.
            mult = min(256, max(32, int(self.nops) // 4))
            target = int(min(self.ncsf, max(1 << 20, int(mult) * int(nsel))))
            cap = _round_up_pow2(int(target))
        else:
            cap = _round_up_pow2(int(cap_in))
        cap = max(1024, int(cap))
        # Cap at ncsf (round down to pow2).
        while cap > int(self.ncsf) and cap > 1:
            cap >>= 1
        return int(cap)

    def _ensure_hash_buffers(self, cap: int) -> None:
        cp = self.cp
        cap = int(cap)
        if cap <= 0:
            raise ValueError("cap must be > 0")
        if int(self.hash_cap) == cap and self.hash_keys is not None:
            return
        self.hash_cap = int(cap)
        self.hash_keys = cp.empty((int(cap),), dtype=cp.int32)
        self.hash_vals = cp.empty((int(self.nroots), int(cap)), dtype=cp.float64)
        self.hash_overflow = cp.empty((1,), dtype=cp.int32)
        self.out_idx = cp.empty((int(cap),), dtype=cp.int32)
        self.out_vals = cp.empty((int(self.nroots), int(cap)), dtype=cp.float64)
        self.out_nnz = cp.empty((1,), dtype=cp.int32)

    def build_and_score(
        self,
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        e_var: np.ndarray,
        max_add: int,
        profile: bool | None = None,
        nvtx: bool | None = None,
    ) -> tuple[list[int], np.ndarray, FrontierHashStats]:
        cp = self.cp
        stream = cp.cuda.get_current_stream()
        profile_enabled = bool(self.profile_enabled if profile is None else profile)
        nvtx_enabled = bool(self.nvtx_enabled if nvtx is None else nvtx)
        timings: dict[str, float] = {}
        mem: dict[str, Any] = {}

        sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32).ravel()
        nsel = int(sel_idx_i32.size)
        if nsel <= 0:
            stats = FrontierHashStats(hash_cap=0, nnz_out=0, overflow_retries=0, timings_ms=timings, memory=mem)
            return [], np.zeros((int(self.nroots),), dtype=np.float64), stats

        # Device inputs.
        sel_idx_d = cp.asarray(sel_idx_i32, dtype=cp.int32)
        c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
        if c_sel_d.ndim != 2 or int(c_sel_d.shape[0]) != nsel or int(c_sel_d.shape[1]) != int(self.nroots):
            raise ValueError("c_sel must have shape (nsel, nroots)")
        e_var_d = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())
        if e_var_d.shape != (int(self.nroots),):
            raise ValueError("e_var must have shape (nroots,)")
        single_root = bool(int(self.nroots) == 1)
        # One global order for offdiag processing so scattered selections do not
        # force many tiny tile-id groups.
        sel_order_h = np.argsort(sel_idx_i32, kind="stable").astype(np.int32, copy=False)
        sel_idx_sorted_h = np.asarray(sel_idx_i32[sel_order_h], dtype=np.int32)
        sel_order_d = cp.asarray(sel_order_h, dtype=cp.int32)
        c_sel_sorted_d = cp.ascontiguousarray(c_sel_d[sel_order_d, :])

        h_eff_flat = getattr(self.ws, "h_eff_flat", None)
        if h_eff_flat is None:
            raise RuntimeError("internal error: workspace h_eff_flat is missing")

        # Tile groups for diagonal rs term (contiguous occ build).
        tile_size = int(self.tile)
        tile_id = (sel_idx_i32 // int(tile_size)).astype(np.int64, copy=False)
        uniq_tiles, inv = np.unique(tile_id, return_inverse=True)
        tile_groups: list[tuple[int, np.ndarray]] = []
        for t_i, tval in enumerate(uniq_tiles.tolist()):
            pos = np.nonzero(inv == int(t_i))[0].astype(np.int32, copy=False)
            if pos.size:
                tile_groups.append((int(tval), pos))

        cap = self._choose_hash_cap(nsel=nsel)
        overflow_retries = 0

        if profile_enabled:
            mem["before"] = _mem_snapshot(cp)
        peak_driver_used = int(mem.get("before", {}).get("driver_used_bytes", 0)) if isinstance(mem.get("before"), dict) else 0

        # Workspace handles.
        k25_ws = getattr(self.ws, "_k25_ws", None)
        if k25_ws is None:
            raise RuntimeError("Kernel25Workspace is unavailable on CUDA workspace")
        row_j_buf = getattr(self.ws, "_csr_row_j", None)
        row_k_buf = getattr(self.ws, "_csr_row_k", None)
        indptr_buf = getattr(self.ws, "_csr_indptr", None)
        indices_buf = getattr(self.ws, "_csr_indices", None)
        data_buf = getattr(self.ws, "_csr_data", None)
        overflow_buf = getattr(self.ws, "_csr_overflow", None)
        if row_j_buf is None or row_k_buf is None or indptr_buf is None or indices_buf is None or data_buf is None or overflow_buf is None:
            raise RuntimeError("internal error: missing Kernel25 staging buffers on workspace")

        l_full = getattr(self.ws, "l_full", None)
        eri_mat = getattr(self.ws, "eri_mat", None)

        for attempt in range(int(self.max_retries)):
            self._ensure_hash_buffers(cap)
            assert self.hash_keys is not None
            assert self.hash_vals is not None
            assert self.hash_overflow is not None
            assert self.out_idx is not None
            assert self.out_vals is not None
            assert self.out_nnz is not None

            with _nvtx_range("cipsi_hash_clear", enabled=nvtx_enabled), _timed_phase(
                "hash_clear", enabled=profile_enabled, timings_ms=timings, stream=stream
            ):
                cipsi_frontier_hash_clear_inplace_device(self.hash_keys, self.hash_vals, threads=256, stream=stream, sync=False)
                cp.cuda.runtime.memsetAsync(int(self.hash_overflow.data.ptr), 0, 4, int(stream.ptr))

            with _nvtx_range("cipsi_one_body", enabled=nvtx_enabled), _timed_phase(
                "one_body_apply", enabled=profile_enabled, timings_ms=timings, stream=stream
            ):
                if single_root:
                    apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                        self.drt,
                        self.ws.drt_dev,
                        self.ws.state_dev,
                        sel_idx_d,
                        h_eff_flat,
                        task_scale=c_sel_d[:, 0],
                        hash_keys=self.hash_keys,
                        hash_vals=self.hash_vals,
                        root=0,
                        selected_mask=self.selected_mask_d,
                        overflow=self.hash_overflow,
                        clear_overflow=False,
                        threads=256,
                        stream=stream,
                        sync=False,
                        check_overflow=False,
                    )
                else:
                    apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                        self.drt,
                        self.ws.drt_dev,
                        self.ws.state_dev,
                        sel_idx_d,
                        h_eff_flat,
                        task_scale_task_major=c_sel_d,
                        hash_keys=self.hash_keys,
                        hash_vals=self.hash_vals,
                        selected_mask=self.selected_mask_d,
                        overflow=self.hash_overflow,
                        clear_overflow=False,
                        threads=256,
                        stream=stream,
                        sync=False,
                        check_overflow=False,
                    )

            # Two-body diagonal rs (r==s): build occ blocks once per tile and apply across roots.
            with _nvtx_range("cipsi_two_body_diag", enabled=nvtx_enabled), _timed_phase(
                "two_body_diag_apply", enabled=profile_enabled, timings_ms=timings, stream=stream
            ):
                for tile_id_val, pos in tile_groups:
                    j_start = int(tile_id_val) * int(tile_size)
                    j_count = min(int(tile_size), int(self.ncsf) - int(j_start))
                    if j_count <= 0:
                        continue
                    build_occ_block_from_steps_inplace_device(
                        self.ws.state_dev,
                        j_start=int(j_start),
                        j_count=int(j_count),
                        occ_out=self.occ_buf[:j_count],
                        threads=256,
                        stream=stream,
                        sync=False,
                    )
                    rel = cp.asarray(sel_idx_i32[pos] - int(j_start), dtype=cp.int32)
                    occ_sel = cp.ascontiguousarray(self.occ_buf[rel])
                    g_diag_sel = cp.dot(occ_sel, self.eri_diag_t)
                    g_diag_sel *= 0.5
                    task_csf_tile = cp.asarray(sel_idx_i32[pos], dtype=cp.int32)
                    c_tile = cp.ascontiguousarray(c_sel_d[pos, :])
                    if single_root:
                        apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                            self.drt,
                            self.ws.drt_dev,
                            self.ws.state_dev,
                            task_csf_tile,
                            g_diag_sel,
                            task_scale=c_tile[:, 0],
                            hash_keys=self.hash_keys,
                            hash_vals=self.hash_vals,
                            root=0,
                            selected_mask=self.selected_mask_d,
                            overflow=self.hash_overflow,
                            clear_overflow=False,
                            threads=256,
                            stream=stream,
                            sync=False,
                            check_overflow=False,
                        )
                    else:
                        apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                            self.drt,
                            self.ws.drt_dev,
                            self.ws.state_dev,
                            task_csf_tile,
                            g_diag_sel,
                            task_scale_task_major=c_tile,
                            hash_keys=self.hash_keys,
                            hash_vals=self.hash_vals,
                            selected_mask=self.selected_mask_d,
                            overflow=self.hash_overflow,
                            clear_overflow=False,
                            threads=256,
                            stream=stream,
                            sync=False,
                            check_overflow=False,
                        )

            # Two-body off-diagonal rs (r!=s): build CSR/build-g once per tile/pair block,
            # then apply across roots.
            if int(self.rs_block) > 0 and int(self.n_pairs) > 0:
                with _nvtx_range("cipsi_two_body_offdiag", enabled=nvtx_enabled), _timed_phase(
                    "two_body_offdiag_apply", enabled=profile_enabled, timings_ms=timings, stream=stream
                ):
                    def _run_offdiag_group(task_csf_tile_h: np.ndarray, c_tile):
                        nsel_tile = int(task_csf_tile_h.size)
                        if nsel_tile <= 0:
                            return
                        task_csf_tile = cp.asarray(task_csf_tile_h, dtype=cp.int32)
                        for p0 in range(0, int(self.n_pairs), int(self.rs_block)):
                            p1 = min(int(self.n_pairs), int(p0 + int(self.rs_block)))
                            blk = int(p1 - p0)
                            if blk <= 0:
                                continue
                            ntasks = int(nsel_tile) * int(blk)
                            if ntasks <= 0:
                                continue
                            if ntasks > int(self.max_tasks):
                                raise RuntimeError("internal error: offdiag task batch exceeds Kernel25 workspace max_tasks")

                            with _nvtx_range("cipsi_k25_csr_build", enabled=nvtx_enabled), _timed_phase(
                                "offdiag_k25_build", enabled=profile_enabled, timings_ms=timings, stream=stream
                            ):
                                task_csf_2d = self.task_csf_buf[:ntasks].reshape(int(nsel_tile), int(blk))
                                task_p_2d = self.task_p_buf[:ntasks].reshape(int(nsel_tile), int(blk))
                                task_q_2d = self.task_q_buf[:ntasks].reshape(int(nsel_tile), int(blk))
                                task_csf_2d[...] = task_csf_tile[:, None]
                                task_p_2d[...] = self.rs_r_d[int(p0) : int(p1)][None, :]
                                task_q_2d[...] = self.rs_s_d[int(p0) : int(p1)][None, :]
                                task_csf = self.task_csf_buf[:ntasks]
                                task_p = self.task_p_buf[:ntasks]
                                task_q = self.task_q_buf[:ntasks]
                                nrows, nnz, _nnz_in = k25_ws.build_from_tasks_deterministic_inplace_device(
                                    self.ws.drt_dev,
                                    self.ws.state_dev,
                                    task_csf,
                                    task_p,
                                    task_q,
                                    row_j_buf,
                                    row_k_buf,
                                    indptr_buf,
                                    indices_buf,
                                    data_buf,
                                    overflow_buf,
                                    int(getattr(self.ws, "threads_enum", 128)),
                                    bool(getattr(self.ws, "coalesce", False)),
                                    int(stream.ptr),
                                    True,
                                    True,
                                )
                            nrows = int(nrows)
                            nnz = int(nnz)
                            if nrows <= 0 or nnz <= 0:
                                continue

                            row_j_d = row_j_buf[:nrows]
                            row_k_d = row_k_buf[:nrows]
                            indptr_d = indptr_buf[: nrows + 1]
                            indices_d = indices_buf[:nnz]
                            data_d = data_buf[:nnz]
                            # Map each CSR row's source-j to local selected index once per CSR build.
                            row_local_all = cp.searchsorted(task_csf_tile, row_j_d)

                            for row_start in range(0, nrows, int(self.g_rows)):
                                row_stop = min(nrows, int(row_start + int(self.g_rows)))
                                nb = int(row_stop - row_start)
                                if nb <= 0:
                                    continue
                                g_b = self.g_buf[:nb]

                                with _nvtx_range("cipsi_kernel3_build_g", enabled=nvtx_enabled), _timed_phase(
                                    "offdiag_kernel3_build_g", enabled=profile_enabled, timings_ms=timings, stream=stream
                                ):
                                    if l_full is not None:
                                        if self.gdf_ws is None:
                                            raise RuntimeError("internal error: gdf_ws is missing for DF path")
                                        self.gdf_ws.build_g_from_csr_l_full_range_inplace_device(
                                            indptr_d,
                                            indices_d,
                                            data_d,
                                            row_start=int(row_start),
                                            nrows=int(nb),
                                            l_full=l_full,
                                            g_out=g_b,
                                            threads=int(getattr(self.ws, "threads_g", 256)),
                                            half=0.5,
                                            stream=stream,
                                            sync=False,
                                        )
                                    else:
                                        if eri_mat is None:
                                            raise RuntimeError("internal error: eri_mat is missing for non-DF path")
                                        kernel3_build_g_from_csr_eri_mat_range_inplace_device(
                                            indptr_d,
                                            indices_d,
                                            data_d,
                                            row_start=int(row_start),
                                            nrows=int(nb),
                                            eri_mat=eri_mat,
                                            g_out=g_b,
                                            threads=int(getattr(self.ws, "threads_g", 256)),
                                            half=0.5,
                                            stream=stream,
                                            sync=False,
                                        )

                                row_k_b = row_k_d[int(row_start) : int(row_stop)]
                                row_local = row_local_all[int(row_start) : int(row_stop)]
                                if single_root:
                                    task_scale_rows1 = self.task_scale_rows1_buf[:nb]
                                    task_scale_rows1[...] = c_tile[row_local, 0]
                                    row_k_apply = row_k_b
                                    g_apply = g_b
                                    use_task_scale = True
                                    if bool(self.offdiag_single_root_aggregate):
                                        with _timed_phase(
                                            "offdiag_row_aggregate",
                                            enabled=profile_enabled,
                                            timings_ms=timings,
                                            stream=stream,
                                        ):
                                            row_k_apply, g_apply = self._aggregate_single_root_offdiag_rows(
                                                row_k_b,
                                                g_b,
                                                task_scale_rows1,
                                            )
                                        use_task_scale = False

                                    with _nvtx_range("cipsi_apply_to_hash_offdiag", enabled=nvtx_enabled), _timed_phase(
                                        "offdiag_apply_hash", enabled=profile_enabled, timings_ms=timings, stream=stream
                                    ):
                                        apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                                            self.drt,
                                            self.ws.drt_dev,
                                            self.ws.state_dev,
                                            row_k_apply,
                                            g_apply,
                                            task_scale=task_scale_rows1 if use_task_scale else None,
                                            hash_keys=self.hash_keys,
                                            hash_vals=self.hash_vals,
                                            root=0,
                                            selected_mask=self.selected_mask_d,
                                            overflow=self.hash_overflow,
                                            clear_overflow=False,
                                            threads=int(self.threads_apply_offdiag),
                                            stream=stream,
                                            sync=False,
                                            check_overflow=False,
                                        )
                                else:
                                    task_scale_rows = self.task_scale_rows_buf[:nb, :]
                                    task_scale_rows[...] = c_tile[row_local, :]
                                    with _nvtx_range("cipsi_apply_to_hash_offdiag", enabled=nvtx_enabled), _timed_phase(
                                        "offdiag_apply_hash", enabled=profile_enabled, timings_ms=timings, stream=stream
                                    ):
                                        apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                                            self.drt,
                                            self.ws.drt_dev,
                                            self.ws.state_dev,
                                            row_k_b,
                                            g_b,
                                            task_scale_task_major=task_scale_rows,
                                            hash_keys=self.hash_keys,
                                            hash_vals=self.hash_vals,
                                            selected_mask=self.selected_mask_d,
                                            overflow=self.hash_overflow,
                                            clear_overflow=False,
                                            threads=int(self.threads_apply_offdiag),
                                            stream=stream,
                                            sync=False,
                                            check_overflow=False,
                                        )

                    if self.offdiag_grouping == "tile":
                        for _tile_id_val, pos in tile_groups:
                            if int(pos.size) <= 0:
                                continue
                            pos_h = np.asarray(pos, dtype=np.int32)
                            order = np.argsort(sel_idx_i32[pos_h], kind="stable")
                            pos_sorted = pos_h[order]
                            task_csf_tile_h = np.asarray(sel_idx_i32[pos_sorted], dtype=np.int32)
                            c_tile = cp.ascontiguousarray(c_sel_d[pos_sorted, :])
                            _run_offdiag_group(task_csf_tile_h, c_tile)
                    else:
                        for s0 in range(0, int(nsel), int(self.offdiag_j_chunk)):
                            s1 = min(int(nsel), int(s0 + int(self.offdiag_j_chunk)))
                            nsel_tile = int(s1 - s0)
                            if nsel_tile <= 0:
                                continue
                            task_csf_tile_h = np.asarray(sel_idx_sorted_h[int(s0) : int(s1)], dtype=np.int32)
                            c_tile = c_sel_sorted_d[int(s0) : int(s1), :]
                            _run_offdiag_group(task_csf_tile_h, c_tile)

            # Overflow check: if set, grow cap and retry.
            if int(self.hash_overflow.get()[0]) != 0:
                cap = min(int(cap) << 1, 1 << 30)
                overflow_retries += 1
                continue

            out_new_idx = cp.empty((int(max_add),), dtype=cp.int32) if int(max_add) > 0 else cp.empty((0,), dtype=cp.int32)
            out_new_n = cp.empty((1,), dtype=cp.int32)
            out_pt2 = cp.empty((int(self.nroots),), dtype=cp.float64)

            with _nvtx_range("cipsi_score_select", enabled=nvtx_enabled), _timed_phase(
                "score_pt2_select", enabled=profile_enabled, timings_ms=timings, stream=stream
            ):
                if self.hash_select_impl == "v2":
                    cipsi_score_and_select_topk_from_hash_slots_v2_inplace_device(
                        self.hash_keys,
                        self.hash_vals,
                        e_var=e_var_d,
                        hdiag=self.hdiag_d,
                        selected_mask=self.selected_mask_d,
                        denom_floor=float(self.denom_floor),
                        out_new_idx=out_new_idx,
                        out_new_n=out_new_n,
                        out_pt2=out_pt2,
                        threads=256,
                        stream=stream,
                        sync=True,
                    )
                else:
                    cipsi_score_and_select_topk_from_hash_slots_inplace_device(
                        self.hash_keys,
                        self.hash_vals,
                        e_var=e_var_d,
                        hdiag=self.hdiag_d,
                        selected_mask=self.selected_mask_d,
                        denom_floor=float(self.denom_floor),
                        out_new_idx=out_new_idx,
                        out_new_n=out_new_n,
                        out_pt2=out_pt2,
                        threads=256,
                        stream=stream,
                        sync=True,
                    )
            with _timed_phase("hash_count_nnz", enabled=profile_enabled, timings_ms=timings, stream=stream):
                nnz_out = int(cp.count_nonzero(self.hash_keys >= 0).get())

            e_pt2_h = cp.asnumpy(out_pt2)
            n_new = int(out_new_n.get()[0])
            new_idx_h: list[int] = []
            if int(max_add) > 0 and n_new > 0:
                new_idx_h = cp.asnumpy(out_new_idx[:n_new]).astype(np.int64, copy=False).tolist()
                new_idx_h = [int(ii) for ii in new_idx_h]

            if profile_enabled:
                mem_after = _mem_snapshot(cp)
                mem["after"] = mem_after
                peak_driver_used = max(int(peak_driver_used), int(mem_after.get("driver_used_bytes", 0)))
                mem["peak_driver_used_bytes"] = int(peak_driver_used)

            stats = FrontierHashStats(
                hash_cap=int(cap),
                nnz_out=int(nnz_out),
                overflow_retries=int(overflow_retries),
                timings_ms=timings,
                memory=mem,
            )
            return new_idx_h, np.asarray(e_pt2_h, dtype=np.float64), stats

        raise RuntimeError("frontier-hash overflow: increase hash_cap or reduce selection size")
