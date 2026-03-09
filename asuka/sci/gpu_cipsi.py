from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.cuguga.oracle import _restore_eri_4d
from asuka.cuda.cuda_backend import (
    GugaMatvecEriMatWorkspace,
    Kernel3BuildGDFWorkspace,
    apply_g_flat_scatter_atomic_frontier_hash_inplace_device,
    build_occ_block_from_steps_inplace_device,
    cipsi_frontier_hash_clear_inplace_device,
    cipsi_frontier_hash_extract_inplace_device,
    cipsi_score_and_select_topk_inplace_device,
    gather_project_batched_inplace_device,
    has_cipsi_frontier_hash_device,
    kernel3_build_g_from_csr_eri_mat_range_inplace_device,
    make_device_drt,
    make_device_state_cache,
    scatter_embed_batched_inplace_device,
)
from asuka.cuda.cuda_davidson import davidson_sym_gpu
from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.mcscf.casci import CASCIResult, _build_casci_df_integrals
from asuka.qmc.sparse import SparseVector
from asuka.sci.selected_ci import _initial_selection, _make_hdiag_guess, _normalize_ci0


def _require_cupy():
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GPU CIPSI requires cupy and the CUDA backend") from e
    return cp


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CIPSITrialSpaceResult:
    e_var: np.ndarray
    e_pt2: np.ndarray
    e_tot: np.ndarray
    sel_idx: np.ndarray
    ci_sel: np.ndarray
    roots: list[SparseVector]
    history: list[dict[str, Any]]
    profile: dict[str, Any]
    epq_mode: str
    ncsf: int

    def to_qmc_x0(self, root: int | None = None):
        if root is None:
            return [(rv.idx.copy(), rv.val.copy()) for rv in self.roots]
        r = int(root)
        if r < 0 or r >= len(self.roots):
            raise IndexError("root out of range")
        rv = self.roots[r]
        return rv.idx.copy(), rv.val.copy()

    def to_rsi_columns(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(rv.idx.copy(), rv.val.copy()) for rv in self.roots]

    def save(self, path: str | Path) -> str:
        path_s = str(path)
        outdir = str(Path(path_s).parent)
        if outdir and outdir != ".":
            Path(outdir).mkdir(parents=True, exist_ok=True)
        payload = {
            "e_var": np.asarray(self.e_var, dtype=np.float64),
            "e_pt2": np.asarray(self.e_pt2, dtype=np.float64),
            "e_tot": np.asarray(self.e_tot, dtype=np.float64),
            "sel_idx": np.asarray(self.sel_idx, dtype=np.int32),
            "ci_sel": np.asarray(self.ci_sel, dtype=np.float64),
            "meta_json": np.asarray(
                json.dumps(
                    {
                        "epq_mode": str(self.epq_mode),
                        "ncsf": int(self.ncsf),
                        "history": _jsonify(self.history),
                        "profile": _jsonify(self.profile),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                dtype=np.str_,
            ),
        }
        with open(path_s, "wb") as fobj:
            np.savez_compressed(fobj, **payload)
        return path_s

    @classmethod
    def load(cls, path: str | Path) -> "CIPSITrialSpaceResult":
        with np.load(str(path), allow_pickle=False) as data:
            meta = json.loads(str(np.asarray(data["meta_json"]).item()))
            sel_idx = np.asarray(data["sel_idx"], dtype=np.int32)
            ci_sel = np.asarray(data["ci_sel"], dtype=np.float64)
            roots = _sparse_roots_from_selected(sel_idx, ci_sel)
            return cls(
                e_var=np.asarray(data["e_var"], dtype=np.float64),
                e_pt2=np.asarray(data["e_pt2"], dtype=np.float64),
                e_tot=np.asarray(data["e_tot"], dtype=np.float64),
                sel_idx=sel_idx,
                ci_sel=ci_sel,
                roots=roots,
                history=list(meta.get("history", [])),
                profile=dict(meta.get("profile", {})),
                epq_mode=str(meta.get("epq_mode", "unknown")),
                ncsf=int(meta.get("ncsf", int(sel_idx.max()) + 1 if sel_idx.size else 0)),
            )


class _ProjectedCudaSubspaceHop:
    def __init__(self, workspace: GugaMatvecEriMatWorkspace, sel_idx: np.ndarray):
        cp = _require_cupy()
        self.cp = cp
        self.workspace = workspace
        self.sel_idx = np.asarray(sel_idx, dtype=np.int64)
        self.sel_idx_d = cp.asarray(self.sel_idx, dtype=cp.int64)
        self.ncsf = int(workspace.ncsf)
        self.nsel = int(self.sel_idx.size)
        self._x_full = None
        self._y_full = None
        self._y_sub = None

    def _ensure_buffers(self, nvec: int):
        cp = self.cp
        nvec = int(nvec)
        if nvec <= 0:
            raise ValueError("nvec must be > 0")
        if self._x_full is not None and int(self._x_full.shape[1]) >= nvec:
            return
        self._x_full = cp.empty((self.ncsf, nvec), dtype=cp.float64, order="C")
        self._y_full = cp.empty((self.ncsf, nvec), dtype=cp.float64, order="C")
        self._y_sub = cp.empty((self.nsel, nvec), dtype=cp.float64, order="C")

    def _apply_block(self, x_sub):
        cp = self.cp
        x_sub = cp.asarray(x_sub, dtype=cp.float64)
        if x_sub.ndim != 2 or int(x_sub.shape[0]) != self.nsel:
            raise ValueError("x_sub must have shape (nsel, nvec)")
        nvec = int(x_sub.shape[1])
        self._ensure_buffers(nvec)
        x_full = self._x_full[:, :nvec]
        y_full = self._y_full[:, :nvec]
        y_sub = self._y_sub[:, :nvec]
        cp.cuda.runtime.memsetAsync(int(x_full.data.ptr), 0, int(x_full.nbytes), int(cp.cuda.get_current_stream().ptr))
        scatter_embed_batched_inplace_device(x_sub, self.sel_idx_d, x_full, sync=False)
        for i in range(nvec):
            self.workspace.hop(x_full[:, i], y=y_full[:, i], sync=False, check_overflow=False)
        gather_project_batched_inplace_device(y_full, self.sel_idx_d, y_sub, sync=False)
        return y_sub

    def apply_full_block(self, x_sub):
        cp = self.cp
        x_sub = cp.asarray(x_sub, dtype=cp.float64)
        if x_sub.ndim != 2 or int(x_sub.shape[0]) != self.nsel:
            raise ValueError("x_sub must have shape (nsel, nvec)")
        nvec = int(x_sub.shape[1])
        self._ensure_buffers(nvec)
        x_full = self._x_full[:, :nvec]
        y_full = self._y_full[:, :nvec]
        cp.cuda.runtime.memsetAsync(int(x_full.data.ptr), 0, int(x_full.nbytes), int(cp.cuda.get_current_stream().ptr))
        scatter_embed_batched_inplace_device(x_sub, self.sel_idx_d, x_full, sync=False)
        for i in range(nvec):
            self.workspace.hop(x_full[:, i], y=y_full[:, i], sync=False, check_overflow=False)
        return y_full

    def __call__(self, x_sub):
        cp = self.cp
        x_sub = cp.asarray(x_sub, dtype=cp.float64)
        if x_sub.ndim == 1:
            y_sub = self._apply_block(x_sub.reshape(self.nsel, 1))
            return y_sub[:, 0]
        if x_sub.ndim == 2:
            return self._apply_block(x_sub)
        raise ValueError("x_sub must be 1D or 2D")


def _normalize_epq_mode(epq_mode: str) -> str:
    mode = str(epq_mode).strip().lower()
    aliases = {
        "materialized": "materialized_epq",
        "epq": "materialized_epq",
        "streamed": "streamed_epq",
        "streaming": "streamed_epq",
        "no_epq": "no_epq_support_aware",
        "none": "no_epq_support_aware",
        "support_aware": "no_epq_support_aware",
    }
    mode = aliases.get(mode, mode)
    allowed = {"materialized_epq", "streamed_epq", "no_epq_support_aware"}
    if mode not in allowed:
        raise ValueError(f"epq_mode must be one of {sorted(allowed)}")
    return mode


def _dense_eri_mat_from_input(eri: Any, *, norb: int) -> np.ndarray:
    if isinstance(eri, np.ndarray) and eri.ndim == 2:
        return np.asarray(eri, dtype=np.float64, order="C")
    eri4 = _restore_eri_4d(eri, int(norb)).astype(np.float64, copy=False)
    return np.asarray(eri4.reshape(int(norb) * int(norb), int(norb) * int(norb)), dtype=np.float64, order="C")


def _build_cuda_workspace(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    epq_mode: str,
    workspace_kwargs: dict[str, Any] | None = None,
) -> tuple[GugaMatvecEriMatWorkspace, dict[str, Any]]:
    cp = _require_cupy()
    nops = int(drt.norb) * int(drt.norb)
    drt_dev = make_device_drt(drt)
    state_dev = make_device_state_cache(drt, drt_dev)
    h1e = np.asarray(h1e, dtype=np.float64, order="C")

    eri_mat_d = None
    l_full_d = None
    j_ps = None
    if isinstance(eri, DeviceDFMOIntegrals):
        j_ps = cp.asarray(eri.j_ps, dtype=cp.float64)
        if eri.eri_mat is not None:
            eri_mat_d = cp.ascontiguousarray(cp.asarray(eri.eri_mat, dtype=cp.float64))
        if eri.l_full is not None:
            l_full_d = cp.ascontiguousarray(cp.asarray(eri.l_full, dtype=cp.float64))
    elif isinstance(eri, DFMOIntegrals):
        j_ps = np.asarray(eri.j_ps, dtype=np.float64, order="C")
        l_full_d = cp.ascontiguousarray(cp.asarray(eri.l_full, dtype=cp.float64))
    else:
        eri_mat_h = _dense_eri_mat_from_input(eri, norb=int(drt.norb))
        eri_mat_d = cp.ascontiguousarray(cp.asarray(eri_mat_h, dtype=cp.float64))
        eri4 = np.asarray(eri_mat_h, dtype=np.float64).reshape(int(drt.norb), int(drt.norb), int(drt.norb), int(drt.norb))
        j_ps = np.einsum("pqqs->ps", eri4, optimize=True).astype(np.float64, copy=False)

    if j_ps is None:
        raise RuntimeError("failed to resolve J_ps for CUDA workspace")
    h_eff = cp.asarray(h1e, dtype=cp.float64) - 0.5 * cp.asarray(j_ps, dtype=cp.float64)

    mode = _normalize_epq_mode(epq_mode)
    ws_opts: dict[str, Any] = {
        "path_mode": "epq_blocked",
        "use_fused_hop": False,
        "use_cuda_graph": False,
        "skip_zero_x_tiles": True,
        "epq_build_device": True,
        "threads_enum": 128,
        "threads_g": 256,
        "threads_w": 256,
        "threads_apply": 64,
        "dtype": cp.float64,
    }
    if mode == "materialized_epq":
        ws_opts.update(
            {
                "use_epq_table": True,
                "epq_streaming": False,
                "aggregate_offdiag_k": True,
            }
        )
    elif mode == "streamed_epq":
        ws_opts.update(
            {
                "use_epq_table": True,
                "epq_streaming": True,
                "aggregate_offdiag_k": True,
            }
        )
    else:
        ws_opts.update(
            {
                "use_epq_table": False,
                "epq_streaming": False,
                "aggregate_offdiag_k": False,
                "prefilter_trivial_tasks": True,
            }
        )
    if workspace_kwargs:
        ws_opts.update(dict(workspace_kwargs))

    try:
        ws = GugaMatvecEriMatWorkspace(
            drt,
            drt_dev=drt_dev,
            state_dev=state_dev,
            eri_mat=eri_mat_d,
            l_full=l_full_d,
            h_eff=h_eff,
            **ws_opts,
        )
    except Exception as e:
        raise RuntimeError(f"failed to initialize GPU CIPSI workspace in mode={mode!r}: {e}") from e

    meta = {
        "epq_mode": mode,
        "norb": int(drt.norb),
        "ncsf": int(drt.ncsf),
        "nops": int(nops),
        "use_epq_table": bool(getattr(ws, "use_epq_table", False)),
        "epq_streaming": bool(getattr(ws, "epq_streaming", False)),
        "aggregate_offdiag_k": bool(getattr(ws, "aggregate_offdiag_k", False)),
        "path_mode": str(getattr(ws, "path_mode", "unknown")),
    }
    return ws, meta


def _match_roots_by_overlap(prev_c_sel: np.ndarray | None, cur_c_sel: np.ndarray, e_var: np.ndarray) -> np.ndarray:
    nroots = int(cur_c_sel.shape[1])
    if prev_c_sel is None or int(prev_c_sel.shape[1]) != nroots:
        return np.arange(nroots, dtype=np.int32)
    nprev = int(prev_c_sel.shape[0])
    overlap = np.abs(np.asarray(prev_c_sel, dtype=np.float64).T @ np.asarray(cur_c_sel[:nprev, :], dtype=np.float64))
    if nroots <= 7:
        best_perm = tuple(range(nroots))
        best_key = None
        for perm in permutations(range(nroots)):
            score = float(sum(overlap[r, perm[r]] for r in range(nroots)))
            key = (score, tuple(-float(e_var[p]) for p in perm))
            if best_key is None or key > best_key:
                best_key = key
                best_perm = perm
        return np.asarray(best_perm, dtype=np.int32)

    picked: list[int] = []
    for r in range(nroots):
        order = np.argsort(overlap[r])[::-1]
        chosen = None
        for c in order.tolist():
            if int(c) not in picked:
                chosen = int(c)
                break
        if chosen is None:
            for c in range(nroots):
                if c not in picked:
                    chosen = int(c)
                    break
        assert chosen is not None
        picked.append(chosen)
    return np.asarray(picked, dtype=np.int32)


def _choose_new_indices(
    score: np.ndarray,
    *,
    max_add: int,
    select_threshold: float | None,
) -> list[int]:
    ncsf, nroots = score.shape
    if max_add <= 0 or ncsf <= 0:
        return []
    global_score = np.max(score, axis=1)
    owner = np.argmax(score, axis=1)
    picked: list[int] = []
    picked_set: set[int] = set()

    for r in range(nroots):
        order = np.argsort(score[:, r])[::-1]
        for ii in order.tolist():
            val = float(score[ii, r])
            if not np.isfinite(val) or val <= 0.0:
                break
            if ii in picked_set:
                continue
            picked.append(int(ii))
            picked_set.add(int(ii))
            break
        if len(picked) >= int(max_add):
            return picked[:max_add]

    if select_threshold is not None:
        mask = np.asarray(global_score >= float(select_threshold), dtype=np.bool_)
        cand = np.nonzero(mask)[0]
    else:
        cand = np.nonzero(np.isfinite(global_score) & (global_score > 0.0))[0]
    if cand.size == 0:
        return picked[:max_add]

    order = sorted(
        (int(ii) for ii in cand.tolist() if int(ii) not in picked_set),
        key=lambda ii: (-float(global_score[ii]), int(owner[ii]), int(ii)),
    )
    for ii in order:
        picked.append(ii)
        picked_set.add(ii)
        if len(picked) >= int(max_add):
            break
    return picked[:max_add]


def _sparse_roots_from_selected(sel_idx: np.ndarray, ci_sel: np.ndarray) -> list[SparseVector]:
    sel_idx = np.asarray(sel_idx, dtype=np.int32).ravel()
    ci_sel = np.asarray(ci_sel, dtype=np.float64)
    if ci_sel.ndim != 2:
        raise ValueError("ci_sel must be 2D")
    roots: list[SparseVector] = []
    for r in range(int(ci_sel.shape[1])):
        col = np.asarray(ci_sel[:, r], dtype=np.float64)
        mask = np.abs(col) > 0.0
        idx = np.asarray(sel_idx[mask], dtype=np.int32)
        val = np.asarray(col[mask], dtype=np.float64)
        if idx.size > 1:
            order = np.argsort(idx, kind="stable")
            idx = np.asarray(idx[order], dtype=np.int32)
            val = np.asarray(val[order], dtype=np.float64)
        roots.append(SparseVector(idx, val))
    return roots


def _build_ci0_subspace(
    *,
    sel_idx: np.ndarray,
    nroots: int,
    ci0_list: list[np.ndarray] | None,
    prev_c_sel: np.ndarray | None,
) -> list[np.ndarray]:
    nsel = int(sel_idx.size)
    if prev_c_sel is not None:
        out = []
        old_nsel = int(prev_c_sel.shape[0])
        for r in range(int(nroots)):
            v = np.zeros((nsel,), dtype=np.float64)
            v[:old_nsel] = np.asarray(prev_c_sel[:, r], dtype=np.float64)
            out.append(v)
        return out
    if ci0_list is not None:
        return [np.asarray(ci0_list[r][sel_idx], dtype=np.float64) for r in range(int(nroots))]
    out = []
    for r in range(int(nroots)):
        v = np.zeros((nsel,), dtype=np.float64)
        v[min(r, nsel - 1)] = 1.0
        out.append(v)
    return out


def run_cipsi_trials(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    hdiag: np.ndarray | None = None,
    ci0: Any = None,
    ecore: float = 0.0,
    nroots: int = 1,
    init_ncsf: int = 256,
    max_ncsf: int = 50_000,
    grow_by: int = 2_000,
    max_iter: int = 20,
    select_threshold: float | None = None,
    denom_floor: float = 1e-12,
    epq_mode: str = "materialized_epq",
    workspace_kwargs: dict[str, Any] | None = None,
    davidson_max_cycle: int = 40,
    davidson_max_space: int = 12,
    davidson_tol: float = 1e-8,
    selection_mode: str = "dense",
    frontier_hash_cap: int | None = None,
    frontier_hash_tile: int = 1024,
    frontier_hash_rs_block: int = 128,
    frontier_hash_csr_capacity_mult: float = 2.0,
    frontier_hash_max_retries: int = 8,
    verbose: int = 0,
) -> CIPSITrialSpaceResult:
    cp = _require_cupy()

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    ncsf = int(drt.ncsf)
    if nroots > ncsf:
        raise ValueError("nroots must be <= drt.ncsf")
    max_ncsf = min(int(max_ncsf), ncsf)
    if max_ncsf < nroots:
        raise ValueError("max_ncsf must be >= nroots")

    if hdiag is None:
        from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415

        state_cache = get_state_cache(drt)
        hdiag = _make_hdiag_guess(drt, h1e, eri, state_cache=state_cache)
    else:
        hdiag = np.asarray(hdiag, dtype=np.float64).ravel()
    if int(hdiag.size) != ncsf:
        raise ValueError("hdiag has wrong length")

    ci0_list = _normalize_ci0(ci0, nroots=nroots, ncsf=ncsf)
    sel_seed = _initial_selection(
        ncsf=ncsf,
        nroots=nroots,
        init_ncsf=int(init_ncsf),
        hdiag=hdiag,
        ci0_list=ci0_list,
    )
    sel: list[int] = []
    loc = -np.ones((ncsf,), dtype=np.int32)
    for ii in sel_seed:
        jj = int(ii)
        if jj < 0 or jj >= ncsf or int(loc[jj]) >= 0:
            continue
        loc[jj] = int(len(sel))
        sel.append(jj)
        if len(sel) >= int(max_ncsf):
            break

    ws, profile = _build_cuda_workspace(drt, h1e, eri, epq_mode=epq_mode, workspace_kwargs=workspace_kwargs)
    history: list[dict[str, Any]] = []
    prev_c_sel: np.ndarray | None = None

    selection_mode_s = str(selection_mode).strip().lower()
    if selection_mode_s in ("dense", "full", "hc_full"):
        selection_mode_s = "dense"
    elif selection_mode_s in ("frontier_hash", "hash", "frontier-hash"):
        selection_mode_s = "frontier_hash"
    else:
        raise ValueError("selection_mode must be 'dense' or 'frontier_hash'")
    profile["selection_mode"] = str(selection_mode_s)

    if selection_mode_s == "frontier_hash":
        if select_threshold is not None:
            raise NotImplementedError("selection_mode='frontier_hash' does not currently support select_threshold")
        if not bool(has_cipsi_frontier_hash_device()):
            raise RuntimeError("selection_mode='frontier_hash' requires CUDA extension with frontier-hash kernels")
        if bool(getattr(ws, "use_epq_table", False)):
            raise RuntimeError("selection_mode='frontier_hash' currently requires epq_mode='no_epq_support_aware' (no EPQ table)")

    frontier = None
    if selection_mode_s == "frontier_hash":
        norb = int(drt.norb)
        nops = norb * norb
        tile = int(frontier_hash_tile)
        if tile <= 0:
            tile = 1024
        # Ensure (tile * rs_block) <= workspace Kernel25Workspace.max_tasks.
        tile = min(tile, int(getattr(ws, "j_tile", tile)))
        tile = max(1, tile)
        rs_block = int(frontier_hash_rs_block)
        if rs_block <= 0:
            rs_block = 128
        rs_r_d = getattr(ws, "_rs_r_d", None)
        rs_s_d = getattr(ws, "_rs_s_d", None)
        if rs_r_d is None or rs_s_d is None:
            raise RuntimeError("internal error: workspace missing rs pair tables")
        n_pairs = int(rs_r_d.size)
        rs_block = min(rs_block, max(1, n_pairs))
        if int(tile) * int(rs_block) > int(getattr(getattr(ws, "_k25_ws", None), "max_tasks", 0)):
            # Conservative fallback: shrink tile until it fits.
            max_tasks = int(getattr(getattr(ws, "_k25_ws", None), "max_tasks", 0))
            if max_tasks <= 0:
                raise RuntimeError("Kernel25Workspace is unavailable on CUDA workspace")
            tile = max(1, int(max_tasks // max(1, rs_block)))

        # Selected mask for skipping internal CI-space entries during PT2/selection.
        selected_mask_d = cp.zeros((ncsf,), dtype=cp.uint8)
        selected_mask_d[cp.asarray(np.asarray(sel, dtype=np.int32), dtype=cp.int32)] = np.uint8(1)
        hdiag_d = cp.ascontiguousarray(cp.asarray(np.asarray(hdiag, dtype=np.float64), dtype=cp.float64))

        # Two-body diagonal rs (r==s) ERI columns: eri_diag_t[r,pq] = eri[pq, rr].
        diag_ids = cp.asarray([int(r) * int(norb) + int(r) for r in range(int(norb))], dtype=cp.int32)
        l_full = getattr(ws, "l_full", None)
        eri_mat = getattr(ws, "eri_mat", None)
        if l_full is not None:
            l_diag = l_full[diag_ids]
            eri_diag_t = cp.ascontiguousarray(cp.dot(l_diag, l_full.T))
        elif eri_mat is not None:
            eri_diag_t = cp.ascontiguousarray(eri_mat[:, diag_ids].T.copy())
        else:
            raise RuntimeError("internal error: workspace missing both eri_mat and l_full")

        # Scratch buffers reused across selection iterations.
        occ_buf = cp.empty((tile, norb), dtype=cp.float64)
        x_full = cp.empty((ncsf,), dtype=cp.float64)
        g_rows = 4096
        g_rows = max(256, min(int(g_rows), int(getattr(ws, "_csr_capacity", g_rows))))
        g_buf = cp.empty((g_rows, nops), dtype=cp.float64)
        gdf_ws = None
        if l_full is not None:
            naux = int(l_full.shape[1])
            gdf_ws = Kernel3BuildGDFWorkspace(int(nops), int(naux), max_nrows=int(g_rows))

        frontier = {
            "tile": int(tile),
            "rs_block": int(rs_block),
            "rs_r_d": rs_r_d,
            "rs_s_d": rs_s_d,
            "n_pairs": int(n_pairs),
            "selected_mask_d": selected_mask_d,
            "hdiag_d": hdiag_d,
            "eri_diag_t": eri_diag_t,
            "x_full": x_full,
            "occ_buf": occ_buf,
            "g_rows": int(g_rows),
            "g_buf": g_buf,
            "gdf_ws": gdf_ws,
            "hash_cap": 0,
            "hash_keys": None,
            "hash_vals": None,
            "hash_overflow": None,
            "out_idx": None,
            "out_vals": None,
            "out_nnz": None,
        }

        def _frontier_hash_select_and_pt2(
            *,
            sel_idx: np.ndarray,
            c_sel: np.ndarray,
            e_var: np.ndarray,
            max_add: int,
        ) -> tuple[list[int], np.ndarray]:
            if frontier is None:  # pragma: no cover
                raise RuntimeError("internal error: frontier workspace is missing")

            stream = cp.cuda.get_current_stream()
            norb_i = int(drt.norb)
            nops_i = int(norb_i) * int(norb_i)
            nsel = int(sel_idx.size)
            if nsel <= 0:
                return [], np.zeros((int(nroots),), dtype=np.float64)

            sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32)
            sel_idx_d = cp.asarray(sel_idx_i32, dtype=cp.int32)
            c_sel_d = cp.asarray(c_sel, dtype=cp.float64)
            c_sel_d = cp.ascontiguousarray(c_sel_d)
            e_var_d = cp.asarray(e_var, dtype=cp.float64).ravel()
            e_var_d = cp.ascontiguousarray(e_var_d)
            if e_var_d.shape != (int(nroots),):
                raise RuntimeError("internal error: e_var has wrong shape")

            h_eff_flat = getattr(ws, "h_eff_flat", None)
            if h_eff_flat is None:
                raise RuntimeError("internal error: workspace h_eff_flat is missing")

            # Tile groups for diagonal rs term (contiguous occ build) and for off-diagonal tasks.
            tile_size = int(frontier["tile"])
            tile_id = (sel_idx_i32 // int(tile_size)).astype(np.int64, copy=False)
            uniq_tiles, inv = np.unique(tile_id, return_inverse=True)
            tile_groups: list[tuple[int, np.ndarray]] = []
            for t_i, tval in enumerate(uniq_tiles.tolist()):
                pos = np.nonzero(inv == int(t_i))[0].astype(np.int32, copy=False)
                if pos.size:
                    tile_groups.append((int(tval), pos))

            # Hash capacity (power of two), grow on overflow.
            cap_in = int(frontier_hash_cap) if frontier_hash_cap is not None else int(frontier.get("hash_cap", 0) or 0)
            cap = int(cap_in)
            if cap <= 0:
                mult = min(256, max(32, nops_i // 4))
                target = int(min(ncsf, max(1 << 20, int(mult) * int(nsel))))
                cap = 1
                while cap < target:
                    cap <<= 1
            else:
                # Round up to a power-of-two (hash kernels require this).
                cap = 1
                while cap < int(cap_in):
                    cap <<= 1
            cap = max(1024, int(cap))
            # Cap at ncsf (round down to pow2).
            while cap > ncsf and cap > 1:
                cap >>= 1

            rs_r_d_loc = frontier["rs_r_d"]
            rs_s_d_loc = frontier["rs_s_d"]
            n_pairs_loc = int(frontier["n_pairs"])
            rs_block_loc = int(frontier["rs_block"])
            rs_block_loc = min(rs_block_loc, n_pairs_loc) if n_pairs_loc > 0 else 0
            x_full = frontier["x_full"]
            occ_buf = frontier["occ_buf"]
            eri_diag_t = frontier["eri_diag_t"]
            g_buf = frontier["g_buf"]
            g_rows_loc = int(frontier["g_rows"])
            selected_mask_d_loc = frontier["selected_mask_d"]
            hdiag_d_loc = frontier["hdiag_d"]

            # CSR staging buffers and Kernel25Workspace from the matvec workspace.
            k25_ws = getattr(ws, "_k25_ws", None)
            if k25_ws is None:
                raise RuntimeError("Kernel25Workspace is unavailable on CUDA workspace")
            row_j_buf = getattr(ws, "_csr_row_j", None)
            row_k_buf = getattr(ws, "_csr_row_k", None)
            indptr_buf = getattr(ws, "_csr_indptr", None)
            indices_buf = getattr(ws, "_csr_indices", None)
            data_buf = getattr(ws, "_csr_data", None)
            overflow_buf = getattr(ws, "_csr_overflow", None)
            if row_j_buf is None or row_k_buf is None or indptr_buf is None or indices_buf is None or data_buf is None or overflow_buf is None:
                raise RuntimeError("internal error: missing Kernel25 staging buffers on workspace")

            l_full_loc = getattr(ws, "l_full", None)
            eri_mat_loc = getattr(ws, "eri_mat", None)
            gdf_ws_loc = frontier.get("gdf_ws")

            for attempt in range(int(frontier_hash_max_retries)):
                # (Re)allocate hash buffers when cap changes.
                if int(frontier.get("hash_cap", 0)) != int(cap) or frontier.get("hash_keys") is None:
                    frontier["hash_cap"] = int(cap)
                    frontier["hash_keys"] = cp.empty((int(cap),), dtype=cp.int32)
                    frontier["hash_vals"] = cp.empty((int(nroots), int(cap)), dtype=cp.float64)
                    frontier["hash_overflow"] = cp.empty((1,), dtype=cp.int32)
                    frontier["out_idx"] = cp.empty((int(cap),), dtype=cp.int32)
                    frontier["out_vals"] = cp.empty((int(nroots), int(cap)), dtype=cp.float64)
                    frontier["out_nnz"] = cp.empty((1,), dtype=cp.int32)

                hash_keys = frontier["hash_keys"]
                hash_vals = frontier["hash_vals"]
                hash_overflow = frontier["hash_overflow"]
                out_idx = frontier["out_idx"]
                out_vals = frontier["out_vals"]
                out_nnz = frontier["out_nnz"]
                if hash_keys is None or hash_vals is None or hash_overflow is None or out_idx is None or out_vals is None or out_nnz is None:
                    raise RuntimeError("internal error: hash buffers missing after allocation")

                cipsi_frontier_hash_clear_inplace_device(hash_keys, hash_vals, threads=256, stream=stream, sync=False)
                cp.cuda.runtime.memsetAsync(int(hash_overflow.data.ptr), 0, 4, int(stream.ptr))

                for r in range(int(nroots)):
                    # x_full <- scatter selected coefficients for this root.
                    cp.cuda.runtime.memsetAsync(int(x_full.data.ptr), 0, int(x_full.nbytes), int(stream.ptr))
                    x_full[sel_idx_d] = c_sel_d[:, r]

                    # One-body: g=h_eff (shared), apply on selected kets.
                    apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                        drt,
                        ws.drt_dev,
                        ws.state_dev,
                        sel_idx_d,
                        h_eff_flat,
                        task_scale=cp.ascontiguousarray(c_sel_d[:, r]),
                        hash_keys=hash_keys,
                        hash_vals=hash_vals,
                        root=int(r),
                        overflow=hash_overflow,
                        clear_overflow=False,
                        threads=256,
                        stream=stream,
                        sync=False,
                        check_overflow=False,
                    )

                    # Two-body diagonal rs (r==s): build occ blocks per active tile, then apply g_diag.
                    for tile_id_val, pos in tile_groups:
                        j_start = int(tile_id_val) * int(tile_size)
                        j_count = min(int(tile_size), int(ncsf) - int(j_start))
                        if j_count <= 0:
                            continue
                        build_occ_block_from_steps_inplace_device(
                            ws.state_dev,
                            j_start=int(j_start),
                            j_count=int(j_count),
                            occ_out=occ_buf[:j_count],
                            threads=256,
                            stream=stream,
                            sync=False,
                        )
                        rel = cp.asarray(sel_idx_i32[pos] - int(j_start), dtype=cp.int32)
                        occ_sel = cp.ascontiguousarray(occ_buf[rel])
                        g_diag_sel = cp.dot(occ_sel, eri_diag_t)
                        g_diag_sel *= 0.5
                        task_csf_tile = cp.asarray(sel_idx_i32[pos], dtype=cp.int32)
                        task_scale_tile = cp.ascontiguousarray(c_sel_d[pos, r])
                        apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                            drt,
                            ws.drt_dev,
                            ws.state_dev,
                            task_csf_tile,
                            g_diag_sel,
                            task_scale=task_scale_tile,
                            hash_keys=hash_keys,
                            hash_vals=hash_vals,
                            root=int(r),
                            overflow=hash_overflow,
                            clear_overflow=False,
                            threads=256,
                            stream=stream,
                            sync=False,
                            check_overflow=False,
                        )

                    # Two-body off-diagonal rs (r!=s): CSR build from tasks, build-g, apply to frontier hash.
                    if rs_block_loc > 0 and n_pairs_loc > 0:
                        for _tile_id_val, pos in tile_groups:
                            task_csf_tile = cp.asarray(sel_idx_i32[pos], dtype=cp.int32)
                            nsel_tile = int(task_csf_tile.size)
                            if nsel_tile <= 0:
                                continue
                            for p0 in range(0, n_pairs_loc, rs_block_loc):
                                p1 = min(n_pairs_loc, int(p0 + rs_block_loc))
                                blk = int(p1 - p0)
                                if blk <= 0:
                                    continue

                                task_csf = cp.ascontiguousarray(cp.repeat(task_csf_tile, blk))
                                task_p = cp.ascontiguousarray(cp.tile(rs_r_d_loc[p0:p1], nsel_tile))
                                task_q = cp.ascontiguousarray(cp.tile(rs_s_d_loc[p0:p1], nsel_tile))

                                nrows, nnz, _nnz_in = k25_ws.build_from_tasks_deterministic_inplace_device(
                                    ws.drt_dev,
                                    ws.state_dev,
                                    task_csf,
                                    task_p,
                                    task_q,
                                    row_j_buf,
                                    row_k_buf,
                                    indptr_buf,
                                    indices_buf,
                                    data_buf,
                                    overflow_buf,
                                    int(getattr(ws, "threads_enum", 128)),
                                    bool(getattr(ws, "coalesce", False)),
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

                                for row_start in range(0, nrows, g_rows_loc):
                                    row_stop = min(nrows, int(row_start + g_rows_loc))
                                    nb = int(row_stop - row_start)
                                    if nb <= 0:
                                        continue
                                    g_b = g_buf[:nb]
                                    if l_full_loc is not None:
                                        if gdf_ws_loc is None:
                                            raise RuntimeError("internal error: gdf_ws is missing for DF path")
                                        gdf_ws_loc.build_g_from_csr_l_full_range_inplace_device(
                                            indptr_d,
                                            indices_d,
                                            data_d,
                                            row_start=int(row_start),
                                            nrows=int(nb),
                                            l_full=l_full_loc,
                                            g_out=g_b,
                                            threads=int(getattr(ws, "threads_g", 256)),
                                            half=0.5,
                                            stream=stream,
                                            sync=False,
                                        )
                                    else:
                                        if eri_mat_loc is None:
                                            raise RuntimeError("internal error: eri_mat is missing for non-DF path")
                                        kernel3_build_g_from_csr_eri_mat_range_inplace_device(
                                            indptr_d,
                                            indices_d,
                                            data_d,
                                            row_start=int(row_start),
                                            nrows=int(nb),
                                            eri_mat=eri_mat_loc,
                                            g_out=g_b,
                                            threads=int(getattr(ws, "threads_g", 256)),
                                            half=0.5,
                                            stream=stream,
                                            sync=False,
                                        )

                                    row_j_b = row_j_d[int(row_start) : int(row_stop)]
                                    row_k_b = row_k_d[int(row_start) : int(row_stop)]
                                    task_scale_row = cp.ascontiguousarray(cp.take(x_full, row_j_b))
                                    apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                                        drt,
                                        ws.drt_dev,
                                        ws.state_dev,
                                        row_k_b,
                                        g_b,
                                        task_scale=task_scale_row,
                                        hash_keys=hash_keys,
                                        hash_vals=hash_vals,
                                        root=int(r),
                                        overflow=hash_overflow,
                                        clear_overflow=False,
                                        threads=int(getattr(ws, "threads_apply", 256)),
                                        stream=stream,
                                        sync=False,
                                        check_overflow=False,
                                    )

                # Overflow check: if set, grow cap and retry.
                if int(hash_overflow.get()[0]) != 0:
                    cap = min(int(cap) << 1, 1 << 30)
                    continue

                # Extract compact frontier for scoring.
                cipsi_frontier_hash_extract_inplace_device(
                    hash_keys,
                    hash_vals,
                    out_idx=out_idx,
                    out_vals_root_major=out_vals,
                    out_nnz=out_nnz,
                    threads=256,
                    stream=stream,
                    sync=True,
                )
                nnz_out = int(out_nnz.get()[0])
                out_new_idx = cp.empty((int(max_add),), dtype=cp.int32)
                out_new_n = cp.empty((1,), dtype=cp.int32)
                out_pt2 = cp.empty((int(nroots),), dtype=cp.float64)
                cipsi_score_and_select_topk_inplace_device(
                    out_idx,
                    out_vals,
                    nnz=int(nnz_out),
                    e_var=e_var_d,
                    hdiag=hdiag_d_loc,
                    selected_mask=selected_mask_d_loc,
                    denom_floor=float(denom_floor),
                    out_new_idx=out_new_idx,
                    out_new_n=out_new_n,
                    out_pt2=out_pt2,
                    threads=256,
                    stream=stream,
                    sync=True,
                )
                e_pt2_h = cp.asnumpy(out_pt2)
                n_new = int(out_new_n.get()[0])
                if n_new > 0:
                    new_idx_h = cp.asnumpy(out_new_idx[:n_new]).astype(np.int64, copy=False).tolist()
                    return [int(ii) for ii in new_idx_h], np.asarray(e_pt2_h, dtype=np.float64)
                return [], np.asarray(e_pt2_h, dtype=np.float64)

            raise RuntimeError("frontier-hash overflow: increase frontier_hash_cap or reduce grow_by")

    for it in range(1, int(max_iter) + 1):
        sel_idx = np.asarray(sel, dtype=np.int64)
        hop = _ProjectedCudaSubspaceHop(ws, sel_idx)
        ci0_sub = _build_ci0_subspace(
            sel_idx=np.asarray(sel_idx, dtype=np.int64),
            nroots=nroots,
            ci0_list=ci0_list,
            prev_c_sel=prev_c_sel,
        )
        dres = davidson_sym_gpu(
            hop,
            x0=ci0_sub,
            hdiag=np.asarray(hdiag[sel_idx], dtype=np.float64),
            nroots=nroots,
            max_cycle=int(davidson_max_cycle),
            max_space=int(davidson_max_space),
            tol=float(davidson_tol),
            subspace_eigh_cpu=True,
            subspace_eigh_cpu_max_m=64,
        )
        e_var = np.asarray(dres.e, dtype=np.float64)
        c_sel = np.ascontiguousarray(np.column_stack(dres.x), dtype=np.float64)
        perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
        if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
            e_var = np.asarray(e_var[perm], dtype=np.float64)
            c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)

        max_add = min(int(grow_by), int(max_ncsf) - int(sel_idx.size))
        if selection_mode_s == "frontier_hash":
            if frontier is None:  # pragma: no cover
                raise RuntimeError("internal error: frontier workspace is missing for selection_mode='frontier_hash'")
            new_idx, e_pt2 = _frontier_hash_select_and_pt2(sel_idx=sel_idx, c_sel=c_sel, e_var=e_var, max_add=int(max_add))
        else:
            c_sel_d = cp.asarray(c_sel, dtype=cp.float64)
            hc_full_d = hop.apply_full_block(c_sel_d)
            hc_full_d[hop.sel_idx_d, :] = 0.0
            denom_d = cp.asarray(np.asarray(e_var, dtype=np.float64).reshape(1, nroots), dtype=cp.float64)
            hdiag_d = cp.asarray(np.asarray(hdiag, dtype=np.float64).reshape(ncsf, 1), dtype=cp.float64)
            denom_d = denom_d - hdiag_d
            if float(denom_floor) > 0.0:
                small = cp.abs(denom_d) < float(denom_floor)
                denom_d = cp.where(small, cp.where(denom_d >= 0.0, float(denom_floor), -float(denom_floor)), denom_d)
            score_d = cp.abs(hc_full_d / denom_d)
            e_pt2 = cp.asnumpy(cp.sum((hc_full_d * hc_full_d) / denom_d, axis=0))
            score = cp.asnumpy(score_d)
            new_idx = _choose_new_indices(score, max_add=int(max_add), select_threshold=select_threshold)
        for ii in new_idx:
            if int(loc[int(ii)]) >= 0:
                continue
            loc[int(ii)] = int(len(sel))
            sel.append(int(ii))
            if len(sel) >= int(max_ncsf):
                break
        if selection_mode_s == "frontier_hash" and frontier is not None and len(new_idx) > 0:
            selected_mask_d = frontier.get("selected_mask_d")
            if selected_mask_d is not None:
                new_i32 = np.asarray(new_idx, dtype=np.int32)
                selected_mask_d[cp.asarray(new_i32, dtype=cp.int32)] = np.uint8(1)

        active_mask = np.any(np.abs(c_sel) > 0.0, axis=1)
        active_sources = np.asarray(sel_idx[active_mask], dtype=np.int64)
        j_tile = int(getattr(ws, "j_tile", int(ncsf)))
        active_tiles = 0
        if active_sources.size:
            active_tiles = int(np.unique(active_sources // max(1, j_tile)).size)

        rec = {
            "iter": int(it),
            "nsel": int(c_sel.shape[0]),
            "nadd": int(len(new_idx)),
            "e_var": np.asarray(e_var + float(ecore), dtype=np.float64),
            "e_pt2": np.asarray(e_pt2, dtype=np.float64),
            "e_tot": np.asarray(e_var + e_pt2 + float(ecore), dtype=np.float64),
            "active_sources": int(active_sources.size),
            "active_tiles": int(active_tiles),
            "epq_mode": str(profile.get("epq_mode", epq_mode)),
        }
        history.append(rec)
        prev_c_sel = np.asarray(c_sel, dtype=np.float64)
        if verbose:
            print(f"[GPU-CIPSI] iter={it} nsel={c_sel.shape[0]} nadd={len(new_idx)} e0={float(e_var[0] + ecore):.12f}")
        if len(new_idx) == 0 or len(sel) >= int(max_ncsf):
            break

    sel_idx = np.asarray(sel, dtype=np.int32)
    hop = _ProjectedCudaSubspaceHop(ws, sel_idx)
    ci0_sub = _build_ci0_subspace(
        sel_idx=np.asarray(sel_idx, dtype=np.int64),
        nroots=nroots,
        ci0_list=ci0_list,
        prev_c_sel=prev_c_sel,
    )
    dres = davidson_sym_gpu(
        hop,
        x0=ci0_sub,
        hdiag=np.asarray(hdiag[sel_idx], dtype=np.float64),
        nroots=nroots,
        max_cycle=int(davidson_max_cycle),
        max_space=int(davidson_max_space),
        tol=float(davidson_tol),
        subspace_eigh_cpu=True,
        subspace_eigh_cpu_max_m=64,
    )
    e_var = np.asarray(dres.e, dtype=np.float64)
    c_sel = np.ascontiguousarray(np.column_stack(dres.x), dtype=np.float64)
    perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
    if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
        e_var = np.asarray(e_var[perm], dtype=np.float64)
        c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)
    if selection_mode_s == "frontier_hash":
        if frontier is None:  # pragma: no cover
            raise RuntimeError("internal error: frontier workspace is missing for selection_mode='frontier_hash'")
        _new_idx, e_pt2 = _frontier_hash_select_and_pt2(sel_idx=np.asarray(sel_idx, dtype=np.int64), c_sel=c_sel, e_var=e_var, max_add=0)
        assert len(_new_idx) == 0
    else:
        c_sel_d = cp.asarray(c_sel, dtype=cp.float64)
        hc_full_d = hop.apply_full_block(c_sel_d)
        hc_full_d[hop.sel_idx_d, :] = 0.0
        denom_d = cp.asarray(np.asarray(e_var, dtype=np.float64).reshape(1, nroots), dtype=cp.float64) - cp.asarray(
            np.asarray(hdiag, dtype=np.float64).reshape(ncsf, 1), dtype=cp.float64
        )
        if float(denom_floor) > 0.0:
            small = cp.abs(denom_d) < float(denom_floor)
            denom_d = cp.where(small, cp.where(denom_d >= 0.0, float(denom_floor), -float(denom_floor)), denom_d)
        e_pt2 = cp.asnumpy(cp.sum((hc_full_d * hc_full_d) / denom_d, axis=0))
    roots = _sparse_roots_from_selected(sel_idx, c_sel)
    return CIPSITrialSpaceResult(
        e_var=np.asarray(e_var + float(ecore), dtype=np.float64),
        e_pt2=np.asarray(e_pt2, dtype=np.float64),
        e_tot=np.asarray(e_var + e_pt2 + float(ecore), dtype=np.float64),
        sel_idx=sel_idx,
        ci_sel=np.asarray(c_sel, dtype=np.float64),
        roots=roots,
        history=history,
        profile=dict(profile),
        epq_mode=str(profile.get("epq_mode", _normalize_epq_mode(epq_mode))),
        ncsf=int(ncsf),
    )


def build_cipsi_trials_from_scf(
    scf_or_casci: Any,
    *,
    ncore: int | None = None,
    ncas: int | None = None,
    nelecas: int | tuple[int, int] | None = None,
    nroots: int = 1,
    df: bool = True,
    backend: str = "cuda",
    epq_mode: str = "materialized_epq",
    ci0: Any = None,
    twos: int | None = None,
    **kwargs,
) -> CIPSITrialSpaceResult:
    backend_s = str(backend).lower()
    if backend_s != "cuda":
        raise ValueError("build_cipsi_trials_from_scf currently requires backend='cuda'")

    if isinstance(scf_or_casci, CASCIResult):
        cas = scf_or_casci
        norb = int(cas.ncas)
        nelecas_i = cas.nelecas if nelecas is None else nelecas
        if twos is None:
            if isinstance(nelecas_i, tuple):
                twos = int(nelecas_i[0]) - int(nelecas_i[1])
            else:
                twos = int(getattr(cas.mol, "spin", 0))
        drt = build_drt(norb=norb, nelec=int(sum(nelecas_i)) if isinstance(nelecas_i, tuple) else int(nelecas_i), twos_target=int(twos))
        return run_cipsi_trials(
            drt,
            cas.h1eff,
            cas.eri,
            ecore=float(cas.ecore),
            nroots=int(nroots),
            ci0=ci0,
            epq_mode=epq_mode,
            **kwargs,
        )

    if ncore is None or ncas is None or nelecas is None:
        raise ValueError("ncore, ncas, and nelecas are required when building from an SCF result")
    if twos is None:
        if isinstance(nelecas, tuple):
            twos = int(nelecas[0]) - int(nelecas[1])
        else:
            twos = int(getattr(getattr(scf_or_casci, "mol", None), "spin", 0))

    if bool(df):
        ints = _build_casci_df_integrals(
            scf_or_casci,
            ncore=int(ncore),
            ncas=int(ncas),
            want_eri_mat=False,
        )
        h1eff = ints.h1eff
        eri = ints.eri
        ecore = float(ints.ecore)
    else:
        raise NotImplementedError(
            "build_cipsi_trials_from_scf(..., df=False) currently requires a CASCIResult input with prebuilt dense integrals"
        )

    drt = build_drt(
        norb=int(ncas),
        nelec=int(sum(nelecas)) if isinstance(nelecas, tuple) else int(nelecas),
        twos_target=int(twos),
    )
    return run_cipsi_trials(
        drt,
        h1eff,
        eri,
        ecore=float(ecore),
        nroots=int(nroots),
        ci0=ci0,
        epq_mode=epq_mode,
        **kwargs,
    )


__all__ = ["CIPSITrialSpaceResult", "build_cipsi_trials_from_scf", "run_cipsi_trials"]
