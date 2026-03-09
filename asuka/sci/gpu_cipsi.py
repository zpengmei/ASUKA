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
    gather_project_batched_inplace_device,
    has_cipsi_frontier_hash_device,
    make_device_drt,
    make_device_state_cache,
    scatter_embed_batched_inplace_device,
)
from asuka.cuda.cuda_davidson import davidson_sym_gpu
from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.mcscf.casci import CASCIResult, _build_casci_df_integrals
from asuka.qmc.sparse import SparseVector
from asuka.sci.frontier_hash import FrontierHashSelector
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
    hb_epsilon: float = 1e-4,
    hb_eps_schedule: str = "fixed",
    hb_eps_init: float = 1e-3,
    hb_eps_final: float = 1e-6,
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

    selection_mode_s = str(selection_mode).strip().lower()
    if selection_mode_s in ("dense", "full", "hc_full"):
        selection_mode_s = "dense"
    elif selection_mode_s in ("frontier_hash", "hash", "frontier-hash"):
        selection_mode_s = "frontier_hash"
    elif selection_mode_s in ("heat_bath", "hb", "heatbath", "hb_sci", "heat-bath"):
        selection_mode_s = "heat_bath"
    else:
        raise ValueError("selection_mode must be 'dense', 'frontier_hash', or 'heat_bath'")

    # Resolve EPQ mode before building the CUDA workspace so we can enforce constraints
    # (frontier-hash and heat-bath selection require the no-EPQ path).
    epq_mode_s = _normalize_epq_mode(epq_mode)
    ws_kwargs = dict(workspace_kwargs or {})
    if selection_mode_s in ("frontier_hash", "heat_bath"):
        if select_threshold is not None:
            raise NotImplementedError(f"selection_mode='{selection_mode_s}' does not currently support select_threshold")
        if not bool(has_cipsi_frontier_hash_device()):
            raise RuntimeError(f"selection_mode='{selection_mode_s}' requires CUDA extension with frontier-hash kernels")
        if epq_mode_s != "no_epq_support_aware":
            raise RuntimeError(f"selection_mode='{selection_mode_s}' requires epq_mode='no_epq_support_aware' (no EPQ table)")
        ws_kwargs.setdefault("csr_capacity_mult", float(frontier_hash_csr_capacity_mult))
        # Force no-EPQ even if the caller passed overrides.
        ws_kwargs["use_epq_table"] = False
        ws_kwargs["aggregate_offdiag_k"] = False
        ws_kwargs["epq_streaming"] = False

    ws, profile = _build_cuda_workspace(drt, h1e, eri, epq_mode=epq_mode_s, workspace_kwargs=ws_kwargs)
    profile["selection_mode"] = str(selection_mode_s)
    history: list[dict[str, Any]] = []
    prev_c_sel: np.ndarray | None = None

    frontier_selector: FrontierHashSelector | None = None
    hb_index_obj = None
    hb_frontier = None
    if selection_mode_s == "heat_bath":
        from asuka.sci.hb_integrals import build_hb_index, build_hb_index_from_df  # noqa: PLC0415
        from asuka.sci.hb_selection import adaptive_epsilon as _adaptive_eps  # noqa: PLC0415

        norb = int(drt.norb)
        nops = norb * norb

        # Build the sorted integral index (one-time cost)
        l_full_ws = getattr(ws, "l_full", None)
        eri_mat_ws = getattr(ws, "eri_mat", None)
        h_eff_flat = getattr(ws, "h_eff_flat", None)
        if h_eff_flat is None:
            raise RuntimeError("internal error: workspace h_eff_flat is missing")
        h_eff_np = cp.asnumpy(h_eff_flat).reshape(norb, norb)

        if l_full_ws is not None:
            l_full_np = cp.asnumpy(l_full_ws)
            hb_index_obj = build_hb_index_from_df(h_eff_np, l_full_np, norb)
        elif eri_mat_ws is not None:
            eri_mat_np = cp.asnumpy(eri_mat_ws).reshape(norb, norb, norb, norb)
            hb_index_obj = build_hb_index(h_eff_np, eri_mat_np, norb)
        else:
            raise RuntimeError("internal error: workspace missing both eri_mat and l_full for HB-SCI")

        # Set up frontier hash buffers for heat-bath mode
        selected_mask_d = cp.zeros((ncsf,), dtype=cp.uint8)
        selected_mask_d[cp.asarray(np.asarray(sel, dtype=np.int32), dtype=cp.int32)] = np.uint8(1)
        hdiag_d = cp.ascontiguousarray(cp.asarray(np.asarray(hdiag, dtype=np.float64), dtype=cp.float64))

        hb_frontier = {
            "selected_mask_d": selected_mask_d,
            "hdiag_d": hdiag_d,
            "hash_cap": 0,
            "hash_keys": None,
            "hash_vals": None,
            "hash_overflow": None,
            "out_idx": None,
            "out_vals": None,
            "out_nnz": None,
            "max_retries": int(frontier_hash_max_retries),
            "hb_dev": None,
        }

    if selection_mode_s == "frontier_hash":
        frontier_selector = FrontierHashSelector(
            drt,
            ws,
            nroots=int(nroots),
            hdiag=hdiag,
            denom_floor=float(denom_floor),
            tile=int(frontier_hash_tile),
            rs_block=int(frontier_hash_rs_block),
            g_rows=None,
            hash_cap=frontier_hash_cap,
            max_retries=int(frontier_hash_max_retries),
            nvtx=None,
            profile=False,
        )
        frontier_selector.reset_selected_mask(np.asarray(sel, dtype=np.int64))

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
            if frontier_selector is None:  # pragma: no cover
                raise RuntimeError("internal error: frontier selector is missing for selection_mode='frontier_hash'")
            new_idx, e_pt2, _stats = frontier_selector.build_and_score(
                sel_idx=sel_idx,
                c_sel=c_sel,
                e_var=e_var,
                max_add=int(max_add),
            )
        elif selection_mode_s == "heat_bath":
            from asuka.sci.hb_selection import heat_bath_select_and_pt2 as _hb_select  # noqa: PLC0415
            if hb_index_obj is None or hb_frontier is None:  # pragma: no cover
                raise RuntimeError("internal error: HB-SCI index/buffers missing")
            # Compute epsilon for this iteration
            if str(hb_eps_schedule).lower() == "adaptive":
                _eps = _adaptive_eps(it, int(sel_idx.size), int(max_ncsf), eps_init=float(hb_eps_init), eps_final=float(hb_eps_final))
            else:
                _eps = float(hb_epsilon)
            new_idx, e_pt2 = _hb_select(
                hb_index_obj, sel_idx, c_sel, e_var, int(max_add), _eps,
                ws, drt, hb_frontier, nroots, ncsf, float(denom_floor),
                verbose=verbose,
            )
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
        if selection_mode_s == "frontier_hash" and frontier_selector is not None and len(new_idx) > 0:
            frontier_selector.mark_selected(new_idx)
        if selection_mode_s == "heat_bath" and hb_frontier is not None and len(new_idx) > 0:
            selected_mask_d = hb_frontier.get("selected_mask_d")
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
        if frontier_selector is None:  # pragma: no cover
            raise RuntimeError("internal error: frontier selector is missing for selection_mode='frontier_hash'")
        _new_idx, e_pt2, _stats = frontier_selector.build_and_score(
            sel_idx=np.asarray(sel_idx, dtype=np.int64),
            c_sel=c_sel,
            e_var=e_var,
            max_add=0,
        )
        assert len(_new_idx) == 0
    elif selection_mode_s == "heat_bath":
        from asuka.sci.hb_selection import heat_bath_select_and_pt2 as _hb_select  # noqa: PLC0415
        if hb_index_obj is None or hb_frontier is None:  # pragma: no cover
            raise RuntimeError("internal error: HB-SCI index/buffers missing")
        _eps_final = float(hb_eps_final) if str(hb_eps_schedule).lower() == "adaptive" else float(hb_epsilon)
        _new_idx, e_pt2 = _hb_select(
            hb_index_obj, np.asarray(sel_idx, dtype=np.int64), c_sel, e_var, 0, _eps_final,
            ws, drt, hb_frontier, nroots, ncsf, float(denom_floor),
            verbose=verbose,
        )
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
