from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.cuguga.oracle import _restore_eri_4d
from asuka.cuguga.state_cache import get_state_cache
from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.mcscf.casci import CASCIResult, _build_casci_df_integrals
from asuka.qmc.labels import normalize_state_rep
from asuka.qmc.sparse import SparseVector
from asuka.sci.frontier_hash import SparseFrontierSelector
from asuka.sci.hb_integrals import build_hb_index, build_hb_index_from_df, upload_hb_index
from asuka.sci.hb_selection import heat_bath_select_and_pt2_sparse
from asuka.sci.streaming_pt2 import StreamingPT2Result, semistochastic_pt2, streaming_pt2_deterministic
from asuka.sci.sparse_support import (
    DiagonalGuessLookup,
    SELECTOR_BUCKET_EDGE_THRESHOLD,
    SELECTOR_BUCKET_MAX_BUCKETS,
    _initial_selection_sparse,
    IncrementalVariationalHamiltonianBuilder,
    _maybe_split_bucket_range,
    _normalize_ci0_sparse,
    _plan_selector_buckets,
    _solver_reorder_perm,
    _solve_subspace,
)


_INT32_MAX = int(np.iinfo(np.int32).max)


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
    sel_key_u64: np.ndarray | None = None
    label_kind: str = "csf_idx"

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
            "sel_idx": np.asarray(self.sel_idx, dtype=np.int64),
            "sel_key_u64": np.asarray(
                np.zeros((0,), dtype=np.uint64) if self.sel_key_u64 is None else self.sel_key_u64,
                dtype=np.uint64,
            ),
            "ci_sel": np.asarray(self.ci_sel, dtype=np.float64),
            "meta_json": np.asarray(
                json.dumps(
                    {
                        "epq_mode": str(self.epq_mode),
                        "label_kind": str(self.label_kind),
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
            sel_idx = np.asarray(data["sel_idx"], dtype=np.int64)
            sel_key_u64 = None
            if "sel_key_u64" in data:
                sel_key_arr = np.asarray(data["sel_key_u64"], dtype=np.uint64).ravel()
                if sel_key_arr.size > 0:
                    sel_key_u64 = np.asarray(sel_key_arr, dtype=np.uint64)
            ci_sel = np.asarray(data["ci_sel"], dtype=np.float64)
            roots = _sparse_roots_from_selected(sel_idx, ci_sel)
            return cls(
                e_var=np.asarray(data["e_var"], dtype=np.float64),
                e_pt2=np.asarray(data["e_pt2"], dtype=np.float64),
                e_tot=np.asarray(data["e_tot"], dtype=np.float64),
                sel_idx=sel_idx,
                sel_key_u64=sel_key_u64,
                label_kind=str(meta.get("label_kind", "csf_idx")),
                ci_sel=ci_sel,
                roots=roots,
                history=list(meta.get("history", [])),
                profile=dict(meta.get("profile", {})),
                epq_mode=str(meta.get("epq_mode", "unknown")),
                ncsf=int(meta.get("ncsf", int(sel_idx.max()) + 1 if sel_idx.size else 0)),
            )


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


def _normalize_cipsi_backend(backend: str) -> str:
    mode = str(backend).strip().lower()
    allowed = {"auto", "cpu_sparse", "cuda_key64", "cuda_idx64"}
    if mode not in allowed:
        raise ValueError("backend must be 'auto', 'cpu_sparse', 'cuda_key64', or 'cuda_idx64'")
    return mode


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


def _sparse_roots_from_selected(sel_idx: np.ndarray, ci_sel: np.ndarray) -> list[SparseVector]:
    sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
    ci_sel = np.asarray(ci_sel, dtype=np.float64)
    if ci_sel.ndim != 2:
        raise ValueError("ci_sel must be 2D")
    roots: list[SparseVector] = []
    for r in range(int(ci_sel.shape[1])):
        col = np.asarray(ci_sel[:, r], dtype=np.float64)
        mask = np.abs(col) > 0.0
        idx = np.asarray(sel_idx[mask], dtype=np.int64)
        val = np.asarray(col[mask], dtype=np.float64)
        if idx.size > 1:
            order = np.argsort(idx, kind="stable")
            idx = np.asarray(idx[order], dtype=np.int64)
            val = np.asarray(val[order], dtype=np.float64)
        roots.append(SparseVector(idx, val))
    return roots


def _plan_cuda_selector_buckets_fast(
    *,
    drt: DRT,
    sel_size: int,
    prev_ncand_hint: int,
    max_add: int,
) -> dict[str, Any]:
    sel_n = max(1, int(sel_size))
    hint = int(prev_ncand_hint)
    if hint <= 0:
        hint = max(sel_n * max(32, int(drt.norb) * int(drt.norb)), sel_n * max(8, int(max_add)))
    if hint < int(SELECTOR_BUCKET_EDGE_THRESHOLD):
        return {
            "selector_bucketed": False,
            "selector_nbuckets": 1,
            "selector_bucket_kind": "cuda_fast_label_range",
            "selector_active_frontier_edges": int(hint),
            "bucket_bounds": ((0, int(drt.ncsf)),),
        }
    nbuckets = max(1, int(np.ceil(float(hint) / float(int(SELECTOR_BUCKET_EDGE_THRESHOLD)))))
    nbuckets = min(int(SELECTOR_BUCKET_MAX_BUCKETS), nbuckets)
    label_hi = int(drt.ncsf)
    bucket_span = max(1, int(np.ceil(float(label_hi) / float(nbuckets))))
    bounds: list[tuple[int, int]] = []
    lo = 0
    while lo < label_hi:
        hi = min(label_hi, lo + bucket_span)
        bounds.append((int(lo), int(hi)))
        lo = hi
    return {
        "selector_bucketed": len(bounds) > 1,
        "selector_nbuckets": int(len(bounds)),
        "selector_bucket_kind": "cuda_fast_label_range",
        "selector_active_frontier_edges": int(hint),
        "bucket_bounds": tuple(bounds),
    }


def _build_ci0_subspace_sparse(
    *,
    sel_idx: np.ndarray,
    loc_map: dict[int, int],
    nroots: int,
    ci0_sparse: list[SparseVector] | None,
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
    if ci0_sparse is not None:
        out: list[np.ndarray] = []
        for r in range(int(nroots)):
            v = np.zeros((nsel,), dtype=np.float64)
            rv = ci0_sparse[r]
            for ii, vv in zip(np.asarray(rv.idx, dtype=np.int64).tolist(), np.asarray(rv.val, dtype=np.float64).tolist()):
                pos = loc_map.get(int(ii))
                if pos is not None:
                    v[int(pos)] = float(vv)
            if not np.any(v) and nsel > 0:
                v[min(r, nsel - 1)] = 1.0
            out.append(v)
        return out
    out = []
    for r in range(int(nroots)):
        v = np.zeros((nsel,), dtype=np.float64)
        if nsel > 0:
            v[min(r, nsel - 1)] = 1.0
        out.append(v)
    return out


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None  # type: ignore[assignment]
    if cp is not None and isinstance(a, cp.ndarray):
        return np.asarray(cp.asnumpy(a), dtype=np.float64, order="C")
    return np.asarray(a, dtype=np.float64, order="C")


def _next_pow2(x: int) -> int:
    x_i = int(x)
    if x_i <= 1:
        return 1
    return 1 << (x_i - 1).bit_length()


def _build_hb_index_and_diag_inputs(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray] | None:
    norb = int(drt.norb)
    h1e_f64 = np.asarray(h1e, dtype=np.float64, order="C")
    h1_diag = np.asarray(np.diag(h1e_f64), dtype=np.float64, order="C")

    _orbsym = getattr(drt, "orbsym", None)
    if isinstance(eri, DeviceDFMOIntegrals):
        if eri.l_full is None:
            return None
        l_full = _asnumpy_f64(eri.l_full)
        j_ps = _asnumpy_f64(eri.j_ps)
        if l_full.ndim != 2 or int(l_full.shape[0]) != norb * norb:
            return None
        h_eff = np.asarray(h1e_f64 - 0.5 * j_ps, dtype=np.float64, order="C")
        hb_index = build_hb_index_from_df(h_eff, l_full, norb, orbsym=_orbsym)
        eri_2d = np.asarray(l_full @ l_full.T, dtype=np.float64, order="C")
    elif isinstance(eri, DFMOIntegrals):
        l_full = np.asarray(eri.l_full, dtype=np.float64, order="C")
        if l_full.ndim != 2 or int(l_full.shape[0]) != norb * norb:
            return None
        h_eff = np.asarray(h1e_f64 - 0.5 * np.asarray(eri.j_ps, dtype=np.float64), dtype=np.float64, order="C")
        hb_index = build_hb_index_from_df(h_eff, l_full, norb, orbsym=_orbsym)
        eri_2d = np.asarray(l_full @ l_full.T, dtype=np.float64, order="C")
    else:
        eri_4d = np.asarray(_restore_eri_4d(eri, norb), dtype=np.float64, order="C")
        h_eff = np.asarray(h1e_f64 - 0.5 * np.einsum("pqqs->ps", eri_4d), dtype=np.float64, order="C")
        hb_index = build_hb_index(h_eff, eri_4d, norb, orbsym=_orbsym)
        eri_2d = np.asarray(eri_4d.reshape(norb * norb, norb * norb), dtype=np.float64, order="C")

    diag_ids = np.arange(norb, dtype=np.int64) * (norb + 1)
    eri_ppqq = np.asarray(eri_2d[np.ix_(diag_ids, diag_ids)], dtype=np.float64, order="C")
    pq_ids = np.arange(norb * norb, dtype=np.int64).reshape(norb, norb)
    eri_pqqp = np.asarray(eri_2d[pq_ids, pq_ids.T], dtype=np.float64, order="C")
    return hb_index, h1_diag, eri_ppqq, eri_pqqp


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
    epq_mode: str = "no_epq_support_aware",
    workspace_kwargs: dict[str, Any] | None = None,
    davidson_max_cycle: int = 40,
    davidson_max_space: int = 12,
    davidson_tol: float = 1e-8,
    selection_mode: str = "frontier_hash",
    frontier_hash_cap: int | None = None,
    frontier_hash_tile: int = 1024,
    frontier_hash_rs_block: int = 0,
    frontier_hash_g_rows: int = 0,
    frontier_hash_offdiag_kernel_mode: str | None = None,
    frontier_hash_csr_capacity_mult: float = 2.0,
    frontier_hash_max_retries: int = 8,
    hb_epsilon: float = 1e-4,
    hb_eps_schedule: str = "fixed",
    hb_eps_init: float = 1e-3,
    hb_eps_final: float = 1e-6,
    backend: str = "auto",
    state_rep: str = "auto",
    verbose: int = 0,
    pt2_mode: str = "exact",
    pt2_bucket_size: int = 500_000,
    pt2_n_det_sources: int | None = None,
    pt2_n_stoch_samples: int = 1000,
    pt2_n_stoch_batches: int = 10,
    pt2_seed: int | None = None,
) -> CIPSITrialSpaceResult:
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    ncsf = int(drt.ncsf)
    if ncsf < 1:
        raise ValueError("drt.ncsf must be >= 1")
    if nroots > ncsf:
        raise ValueError("nroots must be <= drt.ncsf")
    max_ncsf = min(int(max_ncsf), ncsf)
    if max_ncsf < nroots:
        raise ValueError("max_ncsf must be >= nroots")
    backend_requested = _normalize_cipsi_backend(backend)
    state_rep_s = normalize_state_rep(state_rep)
    if state_rep_s == "key64" and int(drt.norb) > 32:
        raise ValueError("state_rep='key64' requires drt.norb <= 32")

    if backend_requested == "cuda_key64" and int(drt.norb) > 32:
        raise ValueError("backend='cuda_key64' requires drt.norb <= 32")
    if backend_requested == "cuda_key64" and state_rep_s in ("i32", "i64"):
        raise ValueError("backend='cuda_key64' is incompatible with state_rep='i32' and 'i64'/'idx64'")
    if backend_requested == "cuda_idx64" and state_rep_s == "key64":
        raise ValueError("backend='cuda_idx64' is incompatible with state_rep='key64'")
    if backend_requested == "cuda_idx64" and int(drt.ncsf) > np.iinfo(np.int64).max:
        raise ValueError("backend='cuda_idx64' requires drt.ncsf <= int64 max")

    if backend_requested == "auto":
        if state_rep_s == "key64":
            backend_effective = "cuda_key64"
        elif state_rep_s == "i64":
            backend_effective = "cuda_idx64"
        elif int(drt.ncsf) > _INT32_MAX:
            backend_effective = "cuda_key64" if int(drt.norb) <= 32 else "cuda_idx64"
        else:
            backend_effective = "cpu_sparse"
    else:
        backend_effective = backend_requested
    if backend_effective == "cuda_key64":
        state_rep_s = "key64"
    elif backend_effective == "cuda_idx64":
        state_rep_s = "i64"

    selection_mode_s = str(selection_mode).strip().lower()
    if selection_mode_s in ("frontier_hash", "hash", "frontier-hash"):
        selection_mode_s = "frontier_hash"
    elif selection_mode_s in ("heat_bath", "hb", "heatbath", "hb_sci", "heat-bath"):
        selection_mode_s = "heat_bath"
    elif selection_mode_s in ("dense", "full", "hc_full"):
        raise ValueError(
            "selection_mode='dense' has been removed from the scalable CIPSI path; "
            "use selection_mode='frontier_hash' or selection_mode='heat_bath'"
        )
    else:
        raise ValueError("selection_mode must be 'frontier_hash' or 'heat_bath'")
    requested_epq_mode = _normalize_epq_mode(epq_mode)
    epq_mode_s = str(requested_epq_mode)
    if epq_mode_s == "streamed_epq":
        allow_streamed = str(os.environ.get("ASUKA_ALLOW_STREAMED_EPQ_EXPERIMENTAL", "")).strip().lower()
        if allow_streamed not in ("1", "true", "yes", "on"):
            epq_mode_s = "materialized_epq"
            if verbose:
                print(
                    "[GPU-CIPSI] streamed_epq requested but currently disabled for stability; "
                    "falling back to materialized_epq (set ASUKA_ALLOW_STREAMED_EPQ_EXPERIMENTAL=1 to force)"
                )
    if epq_mode_s != "no_epq_support_aware":
        raise RuntimeError(f"selection_mode='{selection_mode_s}' requires epq_mode='no_epq_support_aware' (no EPQ table)")

    workspace_cfg = {} if workspace_kwargs is None else dict(workspace_kwargs)
    macro_growth_steps_raw = workspace_cfg.pop("macro_growth_steps", None)
    macro_growth_steps = 1 if macro_growth_steps_raw is None else int(macro_growth_steps_raw)
    macro_growth_resync_frac = float(workspace_cfg.pop("macro_growth_resync_frac", 0.25))
    projected_solver_gpu = bool(workspace_cfg.pop("projected_solver_gpu", True))
    projected_solver_matrix_free = bool(workspace_cfg.pop("projected_solver_matrix_free", False))
    exact_external_projected_selector_requested = bool(workspace_cfg.pop("exact_external_projected_selector", False))
    dense_key64_fast_path = str(workspace_cfg.pop("dense_key64_fast_path", "auto")).strip().lower()
    dense_key64_fast_debug_parity = bool(workspace_cfg.pop("dense_key64_fast_debug_parity", False))
    external_emit_streams = int(workspace_cfg.pop("external_emit_streams", 0))
    external_emit_chunk_min_nsel = int(workspace_cfg.pop("external_emit_chunk_min_nsel", 128))
    row_max_out = int(workspace_cfg.pop("row_max_out", 200_000))
    if macro_growth_steps_raw is None and backend_effective in ("cuda_key64", "cuda_idx64"):
        macro_growth_steps = 4
    if macro_growth_steps < 1:
        raise ValueError("workspace_kwargs['macro_growth_steps'] must be >= 1")
    if macro_growth_resync_frac <= 0.0:
        raise ValueError("workspace_kwargs['macro_growth_resync_frac'] must be > 0")
    if external_emit_streams < 0:
        raise ValueError("workspace_kwargs['external_emit_streams'] must be >= 0")
    if external_emit_chunk_min_nsel < 1:
        raise ValueError("workspace_kwargs['external_emit_chunk_min_nsel'] must be >= 1")
    if row_max_out < 1:
        raise ValueError("workspace_kwargs['row_max_out'] must be >= 1")
    if dense_key64_fast_path not in {"auto", "on", "off"}:
        raise ValueError("workspace_kwargs['dense_key64_fast_path'] must be one of: 'auto', 'on', 'off'")
    macro_schedule_enabled = bool(macro_growth_steps > 1)

    need_sparse_state = int(ncsf) > _INT32_MAX or state_rep_s in ("key64", "i64")
    state_cache = None if need_sparse_state else get_state_cache(drt)
    hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri, hdiag=None if hdiag is None else np.asarray(hdiag, dtype=np.float64))
    ci0_sparse = _normalize_ci0_sparse(ci0, nroots=nroots, ncsf=ncsf)
    sel_seed = _initial_selection_sparse(
        ncsf=ncsf,
        nroots=nroots,
        init_ncsf=int(init_ncsf),
        hdiag_lookup=hdiag_lookup,
        ci0_sparse=ci0_sparse,
    )
    sel: list[int] = []
    loc_map: dict[int, int] = {}
    for ii in sel_seed:
        jj = int(ii)
        if jj < 0 or jj >= ncsf or jj in loc_map:
            continue
        loc_map[jj] = int(len(sel))
        sel.append(jj)
        if len(sel) >= int(max_ncsf):
            break

    profile: dict[str, Any] = {
        "selection_mode": str(selection_mode_s),
        "state_rep": str(state_rep_s),
        "backend_requested": str(backend_requested),
        "backend_effective": str(backend_effective),
        "epq_mode": str(epq_mode_s),
        "driver": "sparse_row_oracle",
        "workspace_kwargs_ignored": bool(workspace_cfg),
        "workspace_kwargs_unused_keys": sorted(str(k) for k in workspace_cfg.keys()),
        "macro_schedule_enabled": bool(macro_schedule_enabled),
        "macro_growth_steps": int(macro_growth_steps),
        "macro_growth_steps_requested": None if macro_growth_steps_raw is None else int(macro_growth_steps_raw),
        "macro_growth_resync_frac": float(macro_growth_resync_frac),
        "projected_solver_gpu_requested": bool(projected_solver_gpu),
        "projected_solver_matrix_free_requested": bool(projected_solver_matrix_free),
        "exact_external_selector_requested": bool(exact_external_projected_selector_requested),
        "exact_external_selector_effective": False,
        "dense_key64_fast_path_requested": str(dense_key64_fast_path),
        "dense_key64_fast_debug_parity": bool(dense_key64_fast_debug_parity),
        "frontier_hash_cap": None if frontier_hash_cap is None else int(frontier_hash_cap),
        "frontier_hash_tile": int(frontier_hash_tile),
        "frontier_hash_rs_block": int(frontier_hash_rs_block),
        "frontier_hash_g_rows": int(frontier_hash_g_rows),
        "frontier_hash_offdiag_kernel_mode": frontier_hash_offdiag_kernel_mode,
        "frontier_hash_csr_capacity_mult": float(frontier_hash_csr_capacity_mult),
        "frontier_hash_max_retries": int(frontier_hash_max_retries),
        "selector_backend_history": [],
        "selection_policy_history": [],
        "threshold_tau_history": [],
        "macro_trim_sizes": [],
        "projected_tuple_build_history": [],
        "projected_tuple_incremental_build_count": 0,
        "external_emit_streams_requested": int(external_emit_streams),
        "external_emit_chunk_min_nsel": int(external_emit_chunk_min_nsel),
        "row_max_out": int(row_max_out),
        "timings_s": {
            "projected_solver_total": 0.0,
            "projected_tuple_build": 0.0,
            "exact_external_select_total": 0.0,
            "macro_commit_total": 0.0,
        },
    }
    row_cache_env = str(os.environ.get("ASUKA_CIPSI_ROW_CACHE", "1")).strip().lower()
    use_row_cache = row_cache_env not in ("0", "false", "off", "no")
    profile["row_cache_enabled"] = bool(use_row_cache)
    if str(requested_epq_mode) != str(epq_mode_s):
        profile["epq_mode_requested"] = str(requested_epq_mode)
        profile["epq_mode_effective"] = str(epq_mode_s)
        profile["epq_mode_fallback_reason"] = "streamed_epq_temporarily_disabled"
    else:
        profile["epq_mode_effective"] = str(epq_mode_s)
    history: list[dict[str, Any]] = []
    prev_c_sel: np.ndarray | None = None
    persistent_row_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if use_row_cache else None
    final_sel_idx_cache: np.ndarray | None = None
    final_e_var_cache: np.ndarray | None = None
    final_c_sel_cache: np.ndarray | None = None
    final_e_pt2_cache: np.ndarray | None = None
    final_cuda_stats_cache: dict[str, Any] | None = None
    final_solver_reordered_cache: bool | None = None
    frontier_selector = SparseFrontierSelector(
        drt,
        h1e,
        eri,
        hdiag_lookup=hdiag_lookup,
        denom_floor=float(denom_floor),
        max_out=int(row_max_out),
        screening=None,
        state_cache=state_cache,
        select_screen_contrib=0.0,
    )
    frontier_selector.reset_selected_mask(np.asarray(sel, dtype=np.int64))
    h_builder: IncrementalVariationalHamiltonianBuilder | None = None
    _sel_dev_cache_labels: np.ndarray | None = None
    _sel_dev_cache_u64 = None
    _sel_dev_cache_sorted = None
    _sel_dev_cache_hash_keys = None
    _sel_dev_cache_hash_cap = 0

    dense_key64_fast_eligible = bool(
        projected_solver_gpu
        and macro_schedule_enabled
        and backend_effective == "cuda_key64"
        and selection_mode_s == "frontier_hash"
        and int(nroots) == 1
        and isinstance(eri, np.ndarray)
        and int(drt.norb) <= 32
    )
    dense_key64_fast_active = bool(
        dense_key64_fast_eligible if dense_key64_fast_path == "auto" else (dense_key64_fast_path == "on")
    )
    if dense_key64_fast_active and macro_growth_steps_raw is None:
        macro_growth_steps = max(int(macro_growth_steps), 8)
        macro_schedule_enabled = bool(macro_growth_steps > 1)
        profile["macro_growth_steps"] = int(macro_growth_steps)
    profile["dense_key64_fast_path_effective"] = bool(dense_key64_fast_active)
    profile["lazy_selected_growth_enabled"] = bool(dense_key64_fast_active)
    selected_diag_cache: dict[int, float] = {}
    projected_tuple_hop_cache = None

    def _ensure_h_builder(
        *,
        sel_seed_cur: list[int] | None = None,
        loc_map_cur: dict[int, int] | None = None,
    ) -> IncrementalVariationalHamiltonianBuilder:
        nonlocal h_builder, sel, loc_map
        if h_builder is None:
            h_builder = IncrementalVariationalHamiltonianBuilder(
                drt,
                h1e,
                eri,
                sel=sel if sel_seed_cur is None else sel_seed_cur,
                loc_map=loc_map if loc_map_cur is None else loc_map_cur,
                max_out=int(row_max_out),
                screening=None,
                state_cache=state_cache,
                row_cache=persistent_row_cache,
            )
            sel = h_builder.sel
            loc_map = h_builder.loc_map
        return h_builder

    def _get_connected_row_exact(idx: int) -> tuple[np.ndarray, np.ndarray]:
        from asuka.sci.sparse_support import _connected_row_cached  # noqa: PLC0415

        return _connected_row_cached(
            drt,
            h1e,
            eri,
            int(idx),
            max_out=int(row_max_out),
            screening=None,
            state_cache=state_cache,
            row_cache=persistent_row_cache,
        )

    def _ensure_selected_diag(labels: list[int]) -> None:
        for jj in labels:
            kk = int(jj)
            if kk in selected_diag_cache:
                continue
            i_idx, hij = _get_connected_row_exact(kk)
            i_idx = np.asarray(i_idx, dtype=np.int64).ravel()
            hij = np.asarray(hij, dtype=np.float64).ravel()
            mask = i_idx == kk
            if not bool(np.any(mask)):
                raise RuntimeError(f"selected row {kk} is missing its diagonal entry")
            selected_diag_cache[kk] = float(hij[mask][0])

    def _get_sel_u64_device_cache(sel_idx_arr: np.ndarray):
        nonlocal _sel_dev_cache_labels, _sel_dev_cache_u64, _sel_dev_cache_sorted
        nonlocal _sel_dev_cache_hash_keys, _sel_dev_cache_hash_cap
        assert _cp is not None
        sel_idx_arr = np.asarray(sel_idx_arr, dtype=np.int64).ravel()
        if (
            _sel_dev_cache_labels is not None
            and int(_sel_dev_cache_labels.size) == int(sel_idx_arr.size)
            and np.array_equal(_sel_dev_cache_labels, sel_idx_arr)
        ):
            return _sel_dev_cache_u64, _sel_dev_cache_sorted, _sel_dev_cache_hash_keys, _sel_dev_cache_hash_cap
        sel_u64_d = _cp.ascontiguousarray(_cp.asarray(sel_idx_arr.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel())
        sel_sorted_d = _cp.ascontiguousarray(_cp.sort(sel_u64_d))
        _sel_dev_cache_labels = np.asarray(sel_idx_arr, dtype=np.int64, order="C")
        _sel_dev_cache_u64 = sel_u64_d
        _sel_dev_cache_sorted = sel_sorted_d
        # Build membership hash table for O(1) lookup (C1 optimization)
        from asuka.cuda.cuda_backend import build_selected_membership_hash  # noqa: PLC0415
        _sel_dev_cache_hash_keys, _sel_dev_cache_hash_cap = build_selected_membership_hash(sel_sorted_d, _cp)
        return _sel_dev_cache_u64, _sel_dev_cache_sorted, _sel_dev_cache_hash_keys, _sel_dev_cache_hash_cap

    cuda_selector_enabled = False
    cuda_selector_reason = ""
    _cp = None
    _cuda_threads = int(max(64, min(256, int(frontier_hash_tile))))
    _hash_cap = 0
    _hash_keys_d = None
    _hash_vals_d = None
    _overflow_d = None
    _drt_dev = None
    _hb_dev = None
    _h1_diag_d = None
    _eri_ppqq_d = None
    _eri_pqqp_d = None
    _hb_effectively_zero = False
    _neleca = 0
    _nelecb = 0
    _cas36_hb_apply = None
    _cas36_hb_emit_tuples = None
    _cas36_diag_guess = None
    _cas36_score_pt2 = None
    prev_cuda_ncand_hint = 0
    exact_external_selector_enabled = bool(
        exact_external_projected_selector_requested or backend_effective in ("cuda_key64", "cuda_idx64")
    )
    _tuple_cap = 0
    _tuple_keys_d = None
    _tuple_src_d = None
    _tuple_hij_d = None
    _tuple_n_d = None

    if backend_effective in ("cuda_key64", "cuda_idx64"):
        try:
            import cupy as _cp  # type: ignore[import-not-found]
            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                cas36_cipsi_score_pt2_compact_u64_inplace_device as _cas36_score_pt2,
                cas36_diag_guess_candidates_u64_dense_inplace_device as _cas36_diag_guess,
                cas36_hb_emit_tuples_u64_inplace_device as _cas36_hb_emit_tuples,
                cas36_hb_screen_and_apply_u64_inplace_device as _cas36_hb_apply,
                has_cas36_hb_emit_tuples_u64_device,
                has_cas36_cipsi_score_pt2_compact_u64_device,
                has_cas36_diag_guess_candidates_u64_dense_device,
                has_cas36_hb_screen_and_apply_u64_device,
                make_device_drt,
            )

            if int(drt.norb) > 64:
                cuda_selector_reason = "norb_gt_64"
            elif int(_cp.cuda.runtime.getDeviceCount()) <= 0:
                cuda_selector_reason = "no_cuda_device"
            elif not (
                bool(has_cas36_hb_screen_and_apply_u64_device())
                and bool(has_cas36_hb_emit_tuples_u64_device())
                and bool(has_cas36_diag_guess_candidates_u64_dense_device())
                and bool(has_cas36_cipsi_score_pt2_compact_u64_device())
            ):
                cuda_selector_reason = "missing_cas36_sci_kernels"
            else:
                hb_pack = _build_hb_index_and_diag_inputs(drt, h1e, eri)
                if hb_pack is None:
                    cuda_selector_reason = "unsupported_integrals_for_cuda_selector"
                else:
                    hb_index, h1_diag_h, eri_ppqq_h, eri_pqqp_h = hb_pack
                    _drt_dev = make_device_drt(drt)
                    _hb_dev = upload_hb_index(hb_index, _cp)
                    _hb_effectively_zero = int(hb_index.n_h1) == 0 and int(hb_index.nnz_2e) == 0
                    # The C++ launch helper currently requires non-null pointers even when
                    # corresponding logical lengths are zero; provide tiny dummy buffers.
                    if int(_hb_dev["h1_abs"].size) == 0:
                        _hb_dev["h1_pq"] = _cp.zeros((1, 2), dtype=_cp.int32)
                        _hb_dev["h1_abs"] = _cp.zeros((1,), dtype=_cp.float64)
                        _hb_dev["h1_signed"] = _cp.zeros((1,), dtype=_cp.float64)
                    if int(_hb_dev["rs_idx"].size) == 0:
                        _hb_dev["rs_idx"] = _cp.zeros((1,), dtype=_cp.int32)
                        _hb_dev["v_abs"] = _cp.zeros((1,), dtype=_cp.float64)
                        _hb_dev["v_signed"] = _cp.zeros((1,), dtype=_cp.float64)
                    _h1_diag_d = _cp.asarray(h1_diag_h, dtype=_cp.float64).ravel()
                    _eri_ppqq_d = _cp.asarray(eri_ppqq_h, dtype=_cp.float64).ravel()
                    _eri_pqqp_d = _cp.asarray(eri_pqqp_h, dtype=_cp.float64).ravel()
                    nelec_tot = int(drt.nelec)
                    twos_t = int(drt.twos_target)
                    if ((nelec_tot + twos_t) & 1) != 0:
                        cuda_selector_reason = "invalid_spin_parity"
                    else:
                        _neleca = (nelec_tot + twos_t) // 2
                        _nelecb = nelec_tot - _neleca
                        if _neleca < 0 or _nelecb < 0:
                            cuda_selector_reason = "invalid_alpha_beta_counts"
                        else:
                            if frontier_hash_cap is None:
                                cap_guess = max(
                                    4096,
                                    8 * max(1, int(init_ncsf)),
                                    4 * max(1, int(grow_by)),
                                )
                            else:
                                cap_guess = int(frontier_hash_cap)
                            _hash_cap = _next_pow2(max(256, int(cap_guess)))
                            _hash_keys_d = _cp.empty((_hash_cap,), dtype=_cp.uint64)
                            _hash_vals_d = _cp.empty((int(nroots), _hash_cap), dtype=_cp.float64)
                            _overflow_d = _cp.zeros((1,), dtype=_cp.int32)
                            _tuple_cap = _next_pow2(max(256, int(cap_guess) * 8))
                            _tuple_keys_d = _cp.empty((_tuple_cap,), dtype=_cp.uint64)
                            _tuple_src_d = _cp.empty((_tuple_cap,), dtype=_cp.int32)
                            _tuple_hij_d = _cp.empty((_tuple_cap,), dtype=_cp.float64)
                            _tuple_n_d = _cp.zeros((1,), dtype=_cp.int32)
                            cuda_selector_enabled = True
        except Exception as _cuda_e:
            cuda_selector_reason = f"cuda_init_failed:{type(_cuda_e).__name__}"

    if cuda_selector_enabled:
        profile["driver"] = "cuda_cas36_hb_compact_u64"
        profile["cuda_selector_enabled"] = True
        profile["cuda_selector_hash_cap_init"] = int(_hash_cap)
        profile["cuda_tuple_cap_init"] = int(_tuple_cap)
        profile["cuda_selector_threads"] = int(_cuda_threads)
    else:
        profile["cuda_selector_enabled"] = False
        if backend_effective in ("cuda_key64", "cuda_idx64"):
            profile["cuda_selector_disabled_reason"] = str(cuda_selector_reason or "unknown")

    def _set_empty_seeds(seeds_out: dict[str, Any] | None) -> None:
        if seeds_out is not None:
            seeds_out["seed_idx"] = np.zeros((0,), dtype=np.int64)
            seeds_out["seed_c1"] = np.zeros((0, int(nroots)), dtype=np.float64)
            seeds_out["seed_w"] = np.zeros((0,), dtype=np.float64)

    exact_external_apply = None

    def _ensure_exact_external_apply():
        nonlocal exact_external_apply
        if exact_external_apply is None:
            from asuka.sci.projected_apply import ExactExternalProjectedApply  # noqa: PLC0415

            exact_external_apply = ExactExternalProjectedApply(
                drt=drt,
                h1e=np.asarray(h1e, dtype=np.float64),
                eri=eri,
                max_out=int(row_max_out),
                screening=None,
                state_cache=state_cache,
                row_cache=persistent_row_cache,
            )
        return exact_external_apply

    def _emit_exact_external_tuples_device(
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        label_lo: int,
        label_hi: int | None,
        screen_contrib: float,
    ):
        nonlocal _tuple_cap, _tuple_keys_d, _tuple_src_d, _tuple_hij_d, _tuple_n_d, _overflow_d
        assert _cp is not None
        assert _drt_dev is not None
        assert _hb_dev is not None
        assert _cas36_hb_emit_tuples is not None
        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if int(sel_idx.size) == 0:
            return (
                _cp.zeros((0,), dtype=_cp.uint64),
                _cp.zeros((0,), dtype=_cp.int32),
                _cp.zeros((0,), dtype=_cp.float64),
            )
        sel_idx_u64_d, sel_idx_sorted_d, _, _ = _get_sel_u64_device_cache(sel_idx)
        c_bound_h = np.asarray(np.max(np.abs(c_sel), axis=1), dtype=np.float64) if int(c_sel.shape[0]) > 0 else np.zeros((0,), dtype=np.float64)
        c_bound_d = _cp.ascontiguousarray(_cp.asarray(c_bound_h, dtype=_cp.float64).ravel())
        eps_val = 0.0 if selection_mode_s == "frontier_hash" else float(screen_contrib)

        nstreams_eff = int(external_emit_streams)
        if nstreams_eff <= 0 and backend_effective == "cuda_key64":
            if int(sel_idx.size) >= int(external_emit_chunk_min_nsel):
                nstreams_eff = min(4, max(2, int(sel_idx.size) // int(external_emit_chunk_min_nsel)))
            else:
                nstreams_eff = 1
        nstreams_eff = max(1, min(int(nstreams_eff), int(sel_idx.size)))
        if nstreams_eff > 1:
            chunks = np.array_split(np.arange(int(sel_idx.size), dtype=np.int32), int(nstreams_eff))
            streams = [_cp.cuda.Stream(non_blocking=True) for _ in range(int(len(chunks)))]
            chunk_buffers: list[tuple[Any, Any, Any, Any, Any, int, int]] = []
            overflow_any = False
            for chunk_idx, src_chunk in enumerate(chunks):
                if int(src_chunk.size) == 0:
                    continue
                src_lo = int(src_chunk[0])
                src_hi = int(src_chunk[-1]) + 1
                sel_chunk_d = _cp.ascontiguousarray(sel_idx_u64_d[src_lo:src_hi].ravel())
                c_bound_chunk_d = _cp.ascontiguousarray(c_bound_d[src_lo:src_hi].ravel())
                cap_guess = max(256, int(np.ceil(float(_tuple_cap) * float(sel_chunk_d.size) / max(1.0, float(sel_idx_u64_d.size)))))
                keys_d = _cp.empty((_next_pow2(int(cap_guess)),), dtype=_cp.uint64)
                src_d = _cp.empty((_next_pow2(int(cap_guess)),), dtype=_cp.int32)
                hij_d = _cp.empty((_next_pow2(int(cap_guess)),), dtype=_cp.float64)
                tuple_n_d = _cp.zeros((1,), dtype=_cp.int32)
                overflow_d = _cp.zeros((1,), dtype=_cp.int32)
                stream_obj = streams[chunk_idx]
                with stream_obj:
                    _cas36_hb_emit_tuples(
                        drt,
                        _drt_dev,
                        sel_chunk_d,
                        c_bound_chunk_d,
                        nsel=int(sel_chunk_d.size),
                        h1_pq=_hb_dev["h1_pq"],
                        h1_abs=_hb_dev["h1_abs"],
                        h1_signed=_hb_dev["h1_signed"],
                        n_h1=int(_hb_dev["h1_abs"].size),
                        pq_ptr=_hb_dev["pq_ptr"],
                        rs_idx=_hb_dev["rs_idx"],
                        v_abs=_hb_dev["v_abs"],
                        v_signed=_hb_dev["v_signed"],
                        pq_max_v=_hb_dev["pq_max_v"],
                        eps=float(eps_val),
                        out_keys_u64=keys_d,
                        out_src=src_d,
                        out_hij=hij_d,
                        cap=int(keys_d.size),
                        label_lo=int(label_lo),
                        label_hi=int(drt.ncsf) if label_hi is None else int(label_hi),
                        selected_idx_sorted_u64=sel_idx_sorted_d,
                        target_mode="external_only",
                        out_n=tuple_n_d,
                        overflow=overflow_d,
                        sym_pq_allowed=_hb_dev.get("sym_pq_allowed"),
                        threads=int(_cuda_threads),
                        stream=int(stream_obj.ptr),
                        sync=False,
                    )
                chunk_buffers.append((keys_d, src_d, hij_d, tuple_n_d, overflow_d, src_lo, chunk_idx))
            for stream_obj in streams:
                stream_obj.synchronize()
            out_parts: list[tuple[Any, Any, Any]] = []
            max_seen_cap = int(_tuple_cap)
            for keys_d, src_d, hij_d, tuple_n_d, overflow_d, src_lo, chunk_idx in chunk_buffers:
                overflow_h = int(_cp.asnumpy(overflow_d)[0])
                nnz_h = int(_cp.asnumpy(tuple_n_d)[0])
                max_seen_cap = max(max_seen_cap, int(keys_d.size))
                if overflow_h != 0 or nnz_h > int(keys_d.size):
                    overflow_any = True
                    break
                if nnz_h <= 0:
                    continue
                src_out_d = _cp.ascontiguousarray(src_d[:nnz_h].ravel())
                if int(src_lo) != 0:
                    src_out_d = _cp.ascontiguousarray(src_out_d + int(src_lo))
                out_parts.append(
                    (
                        _cp.ascontiguousarray(keys_d[:nnz_h].ravel()),
                        src_out_d,
                        _cp.ascontiguousarray(hij_d[:nnz_h].ravel()),
                    )
                )
            profile["exact_external_emit_streams_effective"] = int(nstreams_eff)
            if not overflow_any:
                _tuple_cap = max(int(_tuple_cap), int(max_seen_cap))
                if not out_parts:
                    return (
                        _cp.zeros((0,), dtype=_cp.uint64),
                        _cp.zeros((0,), dtype=_cp.int32),
                        _cp.zeros((0,), dtype=_cp.float64),
                    )
                return (
                    _cp.ascontiguousarray(_cp.concatenate([part[0] for part in out_parts]).ravel()),
                    _cp.ascontiguousarray(_cp.concatenate([part[1] for part in out_parts]).ravel()),
                    _cp.ascontiguousarray(_cp.concatenate([part[2] for part in out_parts]).ravel()),
                )
            profile["exact_external_emit_streams_fallback_reason"] = "parallel_emit_overflow_retry"

        stream_u = int(_cp.cuda.get_current_stream().ptr)
        retries = 0
        while True:
            _tuple_n_d.fill(0)
            _overflow_d.fill(0)
            _cas36_hb_emit_tuples(
                drt,
                _drt_dev,
                sel_idx_u64_d,
                c_bound_d,
                nsel=int(sel_idx_u64_d.size),
                h1_pq=_hb_dev["h1_pq"],
                h1_abs=_hb_dev["h1_abs"],
                h1_signed=_hb_dev["h1_signed"],
                n_h1=int(_hb_dev["h1_abs"].size),
                pq_ptr=_hb_dev["pq_ptr"],
                rs_idx=_hb_dev["rs_idx"],
                v_abs=_hb_dev["v_abs"],
                v_signed=_hb_dev["v_signed"],
                pq_max_v=_hb_dev["pq_max_v"],
                eps=float(eps_val),
                out_keys_u64=_tuple_keys_d,
                out_src=_tuple_src_d,
                out_hij=_tuple_hij_d,
                cap=int(_tuple_cap),
                label_lo=int(label_lo),
                label_hi=int(drt.ncsf) if label_hi is None else int(label_hi),
                selected_idx_sorted_u64=sel_idx_sorted_d,
                target_mode="external_only",
                out_n=_tuple_n_d,
                overflow=_overflow_d,
                sym_pq_allowed=_hb_dev.get("sym_pq_allowed"),
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            overflow_h = int(_cp.asnumpy(_overflow_d)[0])
            nnz_h = int(_cp.asnumpy(_tuple_n_d)[0])
            if overflow_h == 0 and nnz_h <= int(_tuple_cap):
                if nnz_h <= 0:
                    return (
                        _cp.zeros((0,), dtype=_cp.uint64),
                        _cp.zeros((0,), dtype=_cp.int32),
                        _cp.zeros((0,), dtype=_cp.float64),
                    )
                return (
                    _cp.ascontiguousarray(_tuple_keys_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_src_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_hij_d[:nnz_h].ravel()),
                )
            retries += 1
            if retries > int(frontier_hash_max_retries):
                raise RuntimeError(
                    f"exact tuple emitter overflow after {retries} retries (cap={int(_tuple_cap)})"
                )
            grow_cap = max(int(_tuple_cap) * 2, max(256, int(nnz_h) * 2))
            _tuple_cap = _next_pow2(int(grow_cap))
            _tuple_keys_d = _cp.empty((_tuple_cap,), dtype=_cp.uint64)
            _tuple_src_d = _cp.empty((_tuple_cap,), dtype=_cp.int32)
            _tuple_hij_d = _cp.empty((_tuple_cap,), dtype=_cp.float64)
            _tuple_n_d = _cp.zeros((1,), dtype=_cp.int32)

    def _emit_exact_selected_tuples_device(
        *,
        sel_idx: np.ndarray,
    ):
        nonlocal _tuple_cap, _tuple_keys_d, _tuple_src_d, _tuple_hij_d, _tuple_n_d, _overflow_d
        assert _cp is not None
        assert _drt_dev is not None
        assert _hb_dev is not None
        assert _cas36_hb_emit_tuples is not None
        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        if int(sel_idx.size) == 0:
            return (
                _cp.zeros((0,), dtype=_cp.uint64),
                _cp.zeros((0,), dtype=_cp.int32),
                _cp.zeros((0,), dtype=_cp.float64),
            )
        sel_idx_u64_d = _cp.ascontiguousarray(_cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel())
        sel_idx_sorted_d = _cp.ascontiguousarray(_cp.sort(sel_idx_u64_d))
        c_bound_d = _cp.ones((int(sel_idx_u64_d.size),), dtype=_cp.float64)
        stream_u = int(_cp.cuda.get_current_stream().ptr)
        retries = 0
        while True:
            _tuple_n_d.fill(0)
            _overflow_d.fill(0)
            _cas36_hb_emit_tuples(
                drt,
                _drt_dev,
                sel_idx_u64_d,
                c_bound_d,
                nsel=int(sel_idx_u64_d.size),
                h1_pq=_hb_dev["h1_pq"],
                h1_abs=_hb_dev["h1_abs"],
                h1_signed=_hb_dev["h1_signed"],
                n_h1=int(_hb_dev["h1_abs"].size),
                pq_ptr=_hb_dev["pq_ptr"],
                rs_idx=_hb_dev["rs_idx"],
                v_abs=_hb_dev["v_abs"],
                v_signed=_hb_dev["v_signed"],
                pq_max_v=_hb_dev["pq_max_v"],
                eps=0.0,
                out_keys_u64=_tuple_keys_d,
                out_src=_tuple_src_d,
                out_hij=_tuple_hij_d,
                cap=int(_tuple_cap),
                label_lo=0,
                label_hi=int(drt.ncsf),
                selected_idx_sorted_u64=sel_idx_sorted_d,
                target_mode="selected_only",
                out_n=_tuple_n_d,
                overflow=_overflow_d,
                sym_pq_allowed=_hb_dev.get("sym_pq_allowed"),
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            overflow_h = int(_cp.asnumpy(_overflow_d)[0])
            nnz_h = int(_cp.asnumpy(_tuple_n_d)[0])
            if overflow_h == 0 and nnz_h <= int(_tuple_cap):
                if nnz_h <= 0:
                    return (
                        _cp.zeros((0,), dtype=_cp.uint64),
                        _cp.zeros((0,), dtype=_cp.int32),
                        _cp.zeros((0,), dtype=_cp.float64),
                    )
                return (
                    _cp.ascontiguousarray(_tuple_keys_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_src_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_hij_d[:nnz_h].ravel()),
                )
            retries += 1
            if retries > int(frontier_hash_max_retries):
                raise RuntimeError(
                    f"selected tuple emitter overflow after {retries} retries (cap={int(_tuple_cap)})"
                )
            grow_cap = max(int(_tuple_cap) * 2, max(256, int(nnz_h) * 2))
            _tuple_cap = _next_pow2(int(grow_cap))
            _tuple_keys_d = _cp.empty((_tuple_cap,), dtype=_cp.uint64)
            _tuple_src_d = _cp.empty((_tuple_cap,), dtype=_cp.int32)
            _tuple_hij_d = _cp.empty((_tuple_cap,), dtype=_cp.float64)
            _tuple_n_d = _cp.zeros((1,), dtype=_cp.int32)

    _exact_selected_dense_inputs_d = None

    def _ensure_exact_selected_dense_inputs():
        nonlocal _exact_selected_dense_inputs_d
        if _exact_selected_dense_inputs_d is not None:
            return _exact_selected_dense_inputs_d
        if _cp is None:
            raise RuntimeError("cupy is required for the dense exact-selected device emitter")
        from asuka.cuguga.oracle import _restore_eri_4d  # noqa: PLC0415

        if not isinstance(eri, np.ndarray):
            eri_arr = np.asarray(eri, dtype=np.float64)
        else:
            eri_arr = eri
        eri4_h = np.asarray(_restore_eri_4d(eri_arr, int(drt.norb)), dtype=np.float64, order="C")
        # Match the path-native dense oracle used for exact key64 semantics.
        h_base_h = np.asarray(
            np.asarray(h1e, dtype=np.float64, order="C") - 0.5 * np.einsum("pqqs->ps", eri4_h, optimize=True),
            dtype=np.float64,
            order="C",
        )
        _exact_selected_dense_inputs_d = {
            "h_base": _cp.ascontiguousarray(_cp.asarray(h_base_h, dtype=_cp.float64).ravel()),
            "eri4": _cp.ascontiguousarray(_cp.asarray(eri4_h, dtype=_cp.float64).ravel()),
        }
        return _exact_selected_dense_inputs_d

    def _emit_exact_selected_tuples_device_dense(
        *,
        sel_idx: np.ndarray,
        selected_target_idx: np.ndarray | None = None,
        return_diag: bool = False,
    ):
        nonlocal _tuple_cap, _tuple_keys_d, _tuple_src_d, _tuple_hij_d, _tuple_n_d, _overflow_d
        assert _cp is not None
        assert _drt_dev is not None
        from asuka.cuda.cuda_backend import cas36_exact_selected_emit_tuples_dense_u64_inplace_device  # noqa: PLC0415

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        if int(sel_idx.size) == 0:
            empty_ret = (
                _cp.zeros((0,), dtype=_cp.uint64),
                _cp.zeros((0,), dtype=_cp.int32),
                _cp.zeros((0,), dtype=_cp.float64),
            )
            if bool(return_diag):
                return (*empty_ret, _cp.zeros((0,), dtype=_cp.float64))
            return empty_ret
        dense_dev = _ensure_exact_selected_dense_inputs()
        sel_idx_u64_d, sel_idx_sorted_base_d, sel_hash_keys_d, sel_hash_cap = _get_sel_u64_device_cache(sel_idx)
        if selected_target_idx is None:
            sel_idx_sorted_d = sel_idx_sorted_base_d
        else:
            selected_target_idx = np.asarray(selected_target_idx, dtype=np.int64).ravel()
            sel_idx_sorted_d = _cp.ascontiguousarray(
                _cp.sort(_cp.asarray(selected_target_idx.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel())
            )
        c_bound_d = _cp.ones((int(sel_idx_u64_d.size),), dtype=_cp.float64)
        out_diag_d = _cp.zeros((int(sel_idx_u64_d.size),), dtype=_cp.float64) if bool(return_diag) else None
        stream_u = int(_cp.cuda.get_current_stream().ptr)
        retries = 0
        while True:
            _tuple_n_d.fill(0)
            _overflow_d.fill(0)
            cas36_exact_selected_emit_tuples_dense_u64_inplace_device(
                drt,
                _drt_dev,
                sel_idx_u64_d,
                c_bound_d,
                nsel=int(sel_idx_u64_d.size),
                h_base=dense_dev["h_base"],
                eri4=dense_dev["eri4"],
                out_keys_u64=_tuple_keys_d,
                out_src=_tuple_src_d,
                out_hij=_tuple_hij_d,
                cap=int(_tuple_cap),
                out_diag=out_diag_d,
                out_n=_tuple_n_d,
                overflow=_overflow_d,
                threads=min(256, int(_cuda_threads)),
                membership_hash_keys=sel_hash_keys_d,
                membership_hash_cap=sel_hash_cap,
                stream=stream_u,
                sync=False,
            )
            overflow_h = int(_cp.asnumpy(_overflow_d)[0])
            nnz_h = int(_cp.asnumpy(_tuple_n_d)[0])
            if overflow_h == 0 and nnz_h <= int(_tuple_cap):
                if nnz_h <= 0:
                    empty_ret = (
                        _cp.zeros((0,), dtype=_cp.uint64),
                        _cp.zeros((0,), dtype=_cp.int32),
                        _cp.zeros((0,), dtype=_cp.float64),
                    )
                    if bool(return_diag):
                        return (*empty_ret, _cp.ascontiguousarray(out_diag_d.ravel()))
                    return empty_ret
                ret = (
                    _cp.ascontiguousarray(_tuple_keys_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_src_d[:nnz_h].ravel()),
                    _cp.ascontiguousarray(_tuple_hij_d[:nnz_h].ravel()),
                )
                if bool(return_diag):
                    return (*ret, _cp.ascontiguousarray(out_diag_d.ravel()))
                return ret
            retries += 1
            if retries > int(frontier_hash_max_retries):
                raise RuntimeError(
                    f"dense exact selected tuple emitter overflow after {retries} retries (cap={int(_tuple_cap)})"
                )
            grow_cap = max(int(_tuple_cap) * 2, max(256, int(nnz_h) * 2))
            _tuple_cap = _next_pow2(int(grow_cap))
            _tuple_keys_d = _cp.empty((_tuple_cap,), dtype=_cp.uint64)
            _tuple_src_d = _cp.empty((_tuple_cap,), dtype=_cp.int32)
            _tuple_hij_d = _cp.empty((_tuple_cap,), dtype=_cp.float64)
            _tuple_n_d = _cp.zeros((1,), dtype=_cp.int32)

    def _exact_external_select(
        *,
        sel_idx_i64: np.ndarray,
        c_sel_arr: np.ndarray,
        e_var_arr: np.ndarray,
        max_add_i: int,
        eps_val: float,
        seeds_out: dict[str, Any] | None = None,
        selection_policy: str = "exact_topk",
    ) -> tuple[list[int], np.ndarray, dict[str, Any]]:
        select_t0 = time.perf_counter()
        op = _ensure_exact_external_apply()
        sel_idx_i64 = np.asarray(sel_idx_i64, dtype=np.int64).ravel()
        c_sel_arr = np.asarray(c_sel_arr, dtype=np.float64, order="C")
        e_var_arr = np.asarray(e_var_arr, dtype=np.float64).ravel()
        if sel_idx_i64.size == 0:
            _set_empty_seeds(seeds_out)
            return [], np.zeros((int(nroots),), dtype=np.float64), {"ncand": 0, "selector_backend": "exact_external_host"}

        stats: dict[str, Any] = {
            "ncand": 0,
            "selector_backend": "exact_external_host",
            "selection_policy": str(selection_policy),
            "selector_requested_cuda_compact": bool(
                _cp is not None
                and _drt_dev is not None
                and _cas36_diag_guess is not None
                and _cas36_score_pt2 is not None
                and select_threshold is None
            ),
        }

        use_cuda_compact = bool(
            _cp is not None
            and _drt_dev is not None
            and _cas36_hb_emit_tuples is not None
            and _cas36_diag_guess is not None
            and _cas36_score_pt2 is not None
            and select_threshold is None
        )
        if use_cuda_compact:
            cand_idx_u64_d, vals_ncand_root_d = op.accumulate_gpu(
                sel_idx=sel_idx_i64,
                c_sel=c_sel_arr,
                label_lo=0,
                label_hi=int(drt.ncsf),
                screen_contrib=0.0 if selection_mode_s == "frontier_hash" else float(eps_val),
                tuple_emitter=(
                    _emit_exact_external_tuples_device
                    if backend_effective == "cuda_key64"
                    else None
                ),
            )
            ncand = int(cand_idx_u64_d.size)
            stats["ncand"] = int(ncand)
            stats["selector_backend"] = (
                "exact_external_device_emit_cuda_compact"
                if backend_effective == "cuda_key64"
                else "exact_external_gpu_reduce_cuda_compact"
            )
            if ncand <= 0:
                _set_empty_seeds(seeds_out)
                profile["timings_s"]["exact_external_select_total"] = float(
                    profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
                )
                return [], np.zeros((int(nroots),), dtype=np.float64), stats
            _, sel_idx_sorted_d, _, _ = _get_sel_u64_device_cache(sel_idx_i64)
            vals_root_major_d = _cp.ascontiguousarray(vals_ncand_root_d.T)
            e_var_d = _cp.ascontiguousarray(_cp.asarray(e_var_arr, dtype=_cp.float64).ravel())
            stream_u = int(_cp.cuda.get_current_stream().ptr)
            cand_hdiag = _cas36_diag_guess(
                drt,
                _drt_dev,
                cand_idx_u64_d,
                h1_diag=_h1_diag_d,
                eri_ppqq=_eri_ppqq_d,
                eri_pqqp=_eri_pqqp_d,
                neleca=int(_neleca),
                nelecb=int(_nelecb),
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            denom_d = e_var_d[:, None] - cand_hdiag[None, :]
            if float(denom_floor) > 0.0:
                small = _cp.abs(denom_d) < float(denom_floor)
                if bool(_cp.any(small).item()):
                    denom_d = denom_d.copy()
                    denom_d[small] = _cp.where(
                        denom_d[small] >= 0.0,
                        float(denom_floor),
                        -float(denom_floor),
                    )
            c1_root_major_d = vals_root_major_d / denom_d
            w_d = _cp.max(_cp.abs(c1_root_major_d), axis=0)
            score_bits_d = _cp.empty((ncand,), dtype=_cp.uint64)
            pt2_d = _cp.zeros((int(nroots),), dtype=_cp.float64)
            _cas36_score_pt2(
                cand_idx_u64_d,
                vals_root_major_d,
                e_var=e_var_d,
                cand_hdiag=cand_hdiag,
                selected_idx_sorted_u64=sel_idx_sorted_d,
                denom_floor=float(denom_floor),
                score_bits_out=score_bits_d,
                pt2_out=pt2_d,
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            e_pt2_h = np.asarray(_cp.asnumpy(pt2_d), dtype=np.float64)
            max_add_i = int(max_add_i)
            if max_add_i <= 0:
                _set_empty_seeds(seeds_out)
                profile["timings_s"]["exact_external_select_total"] = float(
                    profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
                )
                return [], e_pt2_h, stats
            valid = score_bits_d > 0
            nvalid = int(_cp.count_nonzero(valid).get())
            stats["nvalid"] = int(nvalid)
            if nvalid <= 0:
                _set_empty_seeds(seeds_out)
                profile["timings_s"]["exact_external_select_total"] = float(
                    profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
                )
                return [], e_pt2_h, stats
            keep = min(int(max_add_i), int(nvalid))
            valid_pos = _cp.nonzero(valid)[0]
            if str(selection_policy) == "threshold" and int(valid_pos.size) > int(keep) and int(keep) > 0:
                tau = float(_cp.asnumpy(_cp.min(w_d[valid_pos][_cp.argpartition(w_d[valid_pos], -int(keep))[-int(keep):]])).item())
                chosen = valid_pos[w_d[valid_pos] >= tau]
                stats["threshold_tau"] = float(tau)
            else:
                if int(valid_pos.size) > int(keep):
                    part = _cp.argpartition(score_bits_d[valid_pos], -int(keep))[-int(keep):]
                    chosen = valid_pos[part]
                else:
                    chosen = valid_pos
            keep_score = np.asarray(_cp.asnumpy(_cp.ascontiguousarray(score_bits_d[chosen].ravel())), dtype=np.uint64)
            keep_idx = np.asarray(_cp.asnumpy(_cp.ascontiguousarray(cand_idx_u64_d[chosen].ravel())), dtype=np.uint64)
            keep_c1 = np.asarray(
                _cp.asnumpy(_cp.ascontiguousarray(c1_root_major_d[:, chosen].T)),
                dtype=np.float64,
            )
            keep_w = np.asarray(_cp.asnumpy(_cp.ascontiguousarray(w_d[chosen].ravel())), dtype=np.float64)
            order = np.lexsort((keep_idx, -keep_score.astype(np.int64, copy=False)))
            if seeds_out is not None:
                seeds_out["seed_idx"] = np.asarray(keep_idx[order], dtype=np.int64)
                seeds_out["seed_c1"] = np.asarray(keep_c1[order, :], dtype=np.float64)
                seeds_out["seed_w"] = np.asarray(keep_w[order], dtype=np.float64)
            profile["timings_s"]["exact_external_select_total"] = float(
                profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
            )
            return [int(x) for x in keep_idx[order].tolist()], e_pt2_h, stats

        idx_h, vals_h = op.accumulate_host(
            sel_idx=sel_idx_i64,
            c_sel=c_sel_arr,
            selected_set=set(int(x) for x in sel_idx_i64.tolist()),
            label_lo=0,
            label_hi=int(drt.ncsf),
            screen_contrib=0.0 if selection_mode_s == "frontier_hash" else float(eps_val),
        )
        ncand = int(idx_h.size)
        stats["ncand"] = int(ncand)
        c1_h, w_h, e_pt2_h = op.score_host(
            idx=idx_h,
            vals_root_major=vals_h,
            e_var=e_var_arr,
            hdiag_lookup=hdiag_lookup,
            denom_floor=float(denom_floor),
        )
        max_add_i = int(max_add_i)
        if max_add_i <= 0:
            _set_empty_seeds(seeds_out)
            profile["timings_s"]["exact_external_select_total"] = float(
                profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
            )
            return [], np.asarray(e_pt2_h, dtype=np.float64), stats
        if select_threshold is not None:
            keep = np.nonzero(w_h >= float(select_threshold))[0]
            if int(keep.size) == 0:
                _set_empty_seeds(seeds_out)
                return [], np.asarray(e_pt2_h, dtype=np.float64), stats
            if int(keep.size) > int(max_add_i):
                keep = keep[np.argpartition(w_h[keep], -int(max_add_i))[-int(max_add_i):]]
        else:
            keep_n = min(int(max_add_i), int(idx_h.size))
            if str(selection_policy) == "threshold" and int(idx_h.size) > int(keep_n) and int(keep_n) > 0:
                tau = float(np.min(w_h[np.argpartition(w_h, -keep_n)[-keep_n:]]))
                keep = np.nonzero(w_h >= tau)[0]
                stats["threshold_tau"] = float(tau)
            else:
                keep = np.argpartition(w_h, -keep_n)[-keep_n:]
        keep = keep[np.argsort(w_h[keep])[::-1]]
        idx_keep = np.asarray(idx_h[keep], dtype=np.int64)
        if seeds_out is not None:
            seeds_out["seed_idx"] = idx_keep.copy()
            seeds_out["seed_c1"] = np.asarray(c1_h[keep, :], dtype=np.float64)
            seeds_out["seed_w"] = np.asarray(w_h[keep], dtype=np.float64)
        profile["timings_s"]["exact_external_select_total"] = float(
            profile["timings_s"].get("exact_external_select_total", 0.0) + (time.perf_counter() - select_t0)
        )
        return [int(x) for x in idx_keep.tolist()], np.asarray(e_pt2_h, dtype=np.float64), stats

    def _cuda_select_external(
        *,
        sel_idx_i64: np.ndarray,
        c_sel_arr: np.ndarray,
        e_var_arr: np.ndarray,
        max_add_i: int,
        eps_val: float,
        bucket_bounds: tuple[tuple[int, int], ...] | None = None,
        seeds_out: dict[str, Any] | None = None,
    ) -> tuple[list[int], np.ndarray, dict[str, Any]]:
        nonlocal _hash_cap, _hash_keys_d, _hash_vals_d, _overflow_d
        assert _cp is not None
        assert _drt_dev is not None
        assert _hb_dev is not None
        assert _h1_diag_d is not None and _eri_ppqq_d is not None and _eri_pqqp_d is not None
        assert _cas36_hb_apply is not None and _cas36_diag_guess is not None and _cas36_score_pt2 is not None

        sel_idx_i64 = np.asarray(sel_idx_i64, dtype=np.int64).ravel()
        if sel_idx_i64.size == 0:
            _set_empty_seeds(seeds_out)
            return [], np.zeros((int(nroots),), dtype=np.float64), {"ncand": 0, "overflow_retries": 0, "hash_cap": int(_hash_cap)}
        if bool(_hb_effectively_zero):
            _set_empty_seeds(seeds_out)
            return [], np.zeros((int(nroots),), dtype=np.float64), {"ncand": 0, "overflow_retries": 0, "hash_cap": int(_hash_cap)}
        if int(np.min(sel_idx_i64)) < 0:
            raise ValueError("selected indices must be non-negative for CUDA selector")

        sel_idx_u64_d = _cp.asarray(sel_idx_i64.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel()
        sel_idx_u64_d = _cp.ascontiguousarray(sel_idx_u64_d)
        sel_idx_sorted_d = _cp.sort(sel_idx_u64_d)
        c_sel_d = _cp.ascontiguousarray(_cp.asarray(c_sel_arr, dtype=_cp.float64))
        e_var_d = _cp.ascontiguousarray(_cp.asarray(e_var_arr, dtype=_cp.float64).ravel())

        bucket_bounds_t = tuple(bucket_bounds or ((0, int(drt.ncsf)),))
        retries_total = 0
        bucket_overflow_retries = 0
        bucket_hash_cap_max = int(_hash_cap)
        pt2_total_h = np.zeros((int(nroots),), dtype=np.float64)
        chosen_idx_parts_d: list[Any] = []
        chosen_score_parts_d: list[Any] = []
        chosen_c1_parts_d: list[Any] = []
        ncand_total = 0
        nvalid_total = 0
        bucket_split_count = 0
        empty_u64 = np.uint64(0xFFFFFFFFFFFFFFFF)
        stream_u = int(_cp.cuda.get_current_stream().ptr)
        pending_bounds = list(bucket_bounds_t)
        while pending_bounds:
            label_lo, label_hi = pending_bounds.pop(0)
            retries = 0
            while True:
                _hash_keys_d.fill(empty_u64)
                _hash_vals_d.fill(0.0)
                _overflow_d.fill(0)

                for root in range(int(nroots)):
                    _cas36_hb_apply(
                        drt,
                        _drt_dev,
                        sel_idx_u64_d,
                        _cp.ascontiguousarray(c_sel_d[:, int(root)].ravel()),
                        nsel=int(sel_idx_u64_d.size),
                        root=int(root),
                        h1_pq=_hb_dev["h1_pq"],
                        h1_abs=_hb_dev["h1_abs"],
                        h1_signed=_hb_dev["h1_signed"],
                        n_h1=int(_hb_dev["h1_abs"].size),
                        pq_ptr=_hb_dev["pq_ptr"],
                        rs_idx=_hb_dev["rs_idx"],
                        v_abs=_hb_dev["v_abs"],
                        v_signed=_hb_dev["v_signed"],
                        pq_max_v=_hb_dev["pq_max_v"],
                        eps=float(eps_val),
                        hash_keys_u64=_hash_keys_d,
                        hash_vals=_hash_vals_d,
                        label_lo=int(label_lo),
                        label_hi=int(label_hi),
                        selected_idx_sorted_u64=sel_idx_sorted_d,
                        overflow=_overflow_d,
                        sym_pq_allowed=_hb_dev.get("sym_pq_allowed"),
                        threads=int(_cuda_threads),
                        stream=stream_u,
                        sync=False,
                    )
                overflow_h = int(_cp.asnumpy(_overflow_d)[0])
                if overflow_h == 0:
                    break
                retries += 1
                retries_total += 1
                bucket_overflow_retries = max(bucket_overflow_retries, int(retries))
                if retries > int(frontier_hash_max_retries):
                    raise RuntimeError(
                        f"CUDA selector hash overflow after {retries} retries (cap={int(_hash_cap)}); "
                        "increase frontier_hash_cap or reduce growth"
                    )
                _hash_cap = int(_hash_cap) * 2
                bucket_hash_cap_max = max(bucket_hash_cap_max, int(_hash_cap))
                _hash_keys_d = _cp.empty((int(_hash_cap),), dtype=_cp.uint64)
                _hash_vals_d = _cp.empty((int(nroots), int(_hash_cap)), dtype=_cp.float64)
                _overflow_d = _cp.zeros((1,), dtype=_cp.int32)

            mask = _hash_keys_d != empty_u64
            ncand = int(_cp.count_nonzero(mask).get())
            split_bounds = _maybe_split_bucket_range(
                int(label_lo),
                int(label_hi),
                cand_count=int(ncand),
                max_add=int(max_add_i),
            )
            if len(split_bounds) > 1:
                bucket_split_count += int(len(split_bounds) - 1)
                pending_bounds = list(split_bounds) + pending_bounds
                continue
            ncand_total += int(ncand)
            if ncand <= 0:
                continue

            cand_idx_u64 = _cp.ascontiguousarray(_hash_keys_d[mask].ravel())
            vals_root_major = _cp.ascontiguousarray(_hash_vals_d[:, mask])

            cand_hdiag = _cas36_diag_guess(
                drt,
                _drt_dev,
                cand_idx_u64,
                h1_diag=_h1_diag_d,
                eri_ppqq=_eri_ppqq_d,
                eri_pqqp=_eri_pqqp_d,
                neleca=int(_neleca),
                nelecb=int(_nelecb),
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            denom_d = e_var_d[:, None] - cand_hdiag[None, :]
            if float(denom_floor) > 0.0:
                small = _cp.abs(denom_d) < float(denom_floor)
                if bool(_cp.any(small).item()):
                    denom_d = denom_d.copy()
                    denom_d[small] = _cp.where(
                        denom_d[small] >= 0.0,
                        float(denom_floor),
                        -float(denom_floor),
                    )
            c1_root_major_d = vals_root_major / denom_d
            score_bits_d = _cp.empty((ncand,), dtype=_cp.uint64)
            pt2_d = _cp.zeros((int(nroots),), dtype=_cp.float64)
            _cas36_score_pt2(
                cand_idx_u64,
                vals_root_major,
                e_var=e_var_d,
                cand_hdiag=cand_hdiag,
                selected_idx_sorted_u64=sel_idx_sorted_d,
                denom_floor=float(denom_floor),
                score_bits_out=score_bits_d,
                pt2_out=pt2_d,
                threads=int(_cuda_threads),
                stream=stream_u,
                sync=False,
            )
            pt2_total_h += np.asarray(_cp.asnumpy(pt2_d), dtype=np.float64)

            max_add_i = int(max_add_i)
            if max_add_i <= 0:
                continue

            valid = score_bits_d > 0
            nvalid = int(_cp.count_nonzero(valid).get())
            nvalid_total += int(nvalid)
            if nvalid <= 0:
                continue

            keep = min(int(max_add_i), int(nvalid))
            valid_pos = _cp.nonzero(valid)[0]
            if int(valid_pos.size) > int(keep):
                part = _cp.argpartition(score_bits_d[valid_pos], -int(keep))[-int(keep):]
                chosen = valid_pos[part]
            else:
                chosen = valid_pos

            chosen_score_parts_d.append(_cp.ascontiguousarray(score_bits_d[chosen].ravel()))
            chosen_idx_parts_d.append(_cp.ascontiguousarray(cand_idx_u64[chosen].ravel()))
            if seeds_out is not None:
                chosen_c1_parts_d.append(_cp.ascontiguousarray(c1_root_major_d[:, chosen].T))

        if int(max_add_i) <= 0:
            _set_empty_seeds(seeds_out)
            return [], pt2_total_h, {
                "ncand": int(ncand_total),
                "overflow_retries": int(retries_total),
                "bucket_overflow_retries": int(bucket_overflow_retries),
                "hash_cap": int(_hash_cap),
                "bucket_hash_cap_max": int(bucket_hash_cap_max),
                "bucketed": bool(len(bucket_bounds_t) > 1),
                "nbuckets": int(len(bucket_bounds_t)),
            }

        if not chosen_idx_parts_d:
            _set_empty_seeds(seeds_out)
            return [], pt2_total_h, {
                "ncand": int(ncand_total),
                "nvalid": int(nvalid_total),
                "overflow_retries": int(retries_total),
                "bucket_overflow_retries": int(bucket_overflow_retries),
                "hash_cap": int(_hash_cap),
                "bucket_hash_cap_max": int(bucket_hash_cap_max),
                "bucketed": bool(len(bucket_bounds_t) > 1),
                "nbuckets": int(len(bucket_bounds_t)),
                "bucket_splits": int(bucket_split_count),
            }

        all_idx_d = chosen_idx_parts_d[0] if len(chosen_idx_parts_d) == 1 else _cp.concatenate(chosen_idx_parts_d)
        all_score_d = chosen_score_parts_d[0] if len(chosen_score_parts_d) == 1 else _cp.concatenate(chosen_score_parts_d)
        all_c1_d = None
        if seeds_out is not None and chosen_c1_parts_d:
            all_c1_d = chosen_c1_parts_d[0] if len(chosen_c1_parts_d) == 1 else _cp.concatenate(chosen_c1_parts_d, axis=0)
        keep = min(int(max_add_i), int(all_idx_d.size))
        keep_pos = _cp.argpartition(all_score_d, -keep)[-keep:]
        keep_score_d = _cp.ascontiguousarray(all_score_d[keep_pos].ravel())
        keep_idx_d = _cp.ascontiguousarray(all_idx_d[keep_pos].ravel())
        keep_score = np.asarray(_cp.asnumpy(keep_score_d), dtype=np.uint64)
        keep_idx = np.asarray(_cp.asnumpy(keep_idx_d), dtype=np.uint64)
        keep_c1 = None if all_c1_d is None else np.asarray(_cp.asnumpy(_cp.ascontiguousarray(all_c1_d[keep_pos, :])), dtype=np.float64)
        order = np.lexsort((keep_idx, -keep_score.astype(np.int64, copy=False)))
        if seeds_out is not None:
            seeds_out["seed_idx"] = np.asarray(keep_idx[order], dtype=np.int64)
            seeds_out["seed_c1"] = (
                np.zeros((0, int(nroots)), dtype=np.float64)
                if keep_c1 is None
                else np.asarray(keep_c1[order, :], dtype=np.float64)
            )
        new_idx_h = [int(x) for x in keep_idx[order].tolist()]
        return new_idx_h, pt2_total_h, {
            "ncand": int(ncand_total),
            "nvalid": int(nvalid_total),
            "overflow_retries": int(retries_total),
            "bucket_overflow_retries": int(bucket_overflow_retries),
            "hash_cap": int(_hash_cap),
            "bucket_hash_cap_max": int(bucket_hash_cap_max),
            "bucketed": bool(len(bucket_bounds_t) > 1),
            "nbuckets": int(len(bucket_bounds_t)),
            "bucket_splits": int(bucket_split_count),
        }

    def _compute_hb_eps_iter(*, sel_size: int) -> float:
        if selection_mode_s != "heat_bath":
            return 0.0
        if str(hb_eps_schedule).lower() == "adaptive":
            frac = 0.0 if int(max_ncsf) <= int(init_ncsf) else (int(sel_size) - int(init_ncsf)) / max(
                1, int(max_ncsf) - int(init_ncsf)
            )
            frac = float(np.clip(frac, 0.0, 1.0))
            return float(hb_eps_init) * (float(hb_eps_final) / float(hb_eps_init)) ** frac
        return float(hb_epsilon)

    def _solve_selected_subspace(
        *,
        sel_idx_arr: np.ndarray,
        loc_map_cur: dict[int, int],
        prev_guess: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        solve_t0 = time.perf_counter()
        h_sub = None
        nsel_cur = int(sel_idx_arr.size)

        def _ensure_h_sub():
            nonlocal h_sub
            if h_sub is None:
                h_sub = _ensure_h_builder().to_csr()
            return h_sub

        use_gpu_projected = bool(
            projected_solver_gpu
            and macro_schedule_enabled
            and _cp is not None
            and backend_effective in ("cuda_key64", "cuda_idx64")
        )
        ci0_sub = _build_ci0_subspace_sparse(
            sel_idx=sel_idx_arr,
            loc_map=loc_map_cur,
            nroots=int(nroots),
            ci0_sparse=ci0_sparse,
            prev_c_sel=prev_guess,
        )
        if use_gpu_projected:
            try:
                from asuka.cuda.cuda_davidson import davidson_sym_gpu  # noqa: PLC0415
                from asuka.sci.projected_apply import ExactSelectedProjectedHop, ExactSelectedTupleProjectedHop  # noqa: PLC0415

                guess_list: list[np.ndarray] = []
                if ci0_sub:
                    for v in ci0_sub:
                        arr = np.asarray(v, dtype=np.float64).ravel()
                        if int(arr.size) != int(nsel_cur):
                            continue
                        if float(np.linalg.norm(arr)) > 0.0:
                            guess_list.append(np.asarray(arr, dtype=np.float64))
                if not guess_list:
                    for i in range(min(int(nroots), int(nsel_cur))):
                        vec = np.zeros((int(nsel_cur),), dtype=np.float64)
                        vec[i] = 1.0
                        guess_list.append(vec)

                solver_backend = "cuda_davidson_projected_exact_sell"
                exact_projected_hop = None

                def _build_sell_hop():
                    nonlocal exact_projected_hop
                    if exact_projected_hop is None:
                        exact_projected_hop = ExactSelectedProjectedHop.from_csr(
                            np.asarray(sel_idx_arr, dtype=np.int64),
                            _ensure_h_sub(),
                        )

                    def _hop(v_d):
                        assert exact_projected_hop is not None
                        return exact_projected_hop.hop_gpu(v_d)

                    return _hop, np.asarray(exact_projected_hop.hdiag, dtype=np.float64)

                def _build_projected_tuple_hop():
                    nonlocal projected_tuple_hop_cache
                    tuple_build_t0 = time.perf_counter()
                    incremental_tuple_build = False
                    sel_labels = [int(j) for j in np.asarray(sel_idx_arr, dtype=np.int64).ravel().tolist()]

                    from asuka.cuda.cuda_backend import has_cas36_exact_selected_emit_tuples_dense_u64_device  # noqa: PLC0415

                    use_dense_device_emitter = bool(
                        backend_effective == "cuda_key64"
                        and _cp is not None
                        and isinstance(eri, np.ndarray)
                        and int(drt.norb) <= 32
                        and bool(has_cas36_exact_selected_emit_tuples_dense_u64_device())
                    )

                    # Check if pair-wise H[i,j] kernel should be used instead.
                    # Preferred for large half-filled active spaces where the DFS-based
                    # dense emitter is prohibitively slow.
                    use_pairwise_hij = False
                    _pairwise_norb_threshold = int(os.environ.get("ASUKA_PAIRWISE_HIJ_NORB_THRESHOLD", "14"))
                    _pairwise_nsel_max = int(os.environ.get("ASUKA_PAIRWISE_HIJ_NSEL_MAX", "25000"))
                    if (
                        backend_effective == "cuda_key64"
                        and _cp is not None
                        and isinstance(eri, np.ndarray)
                        and int(drt.norb) >= _pairwise_norb_threshold
                        and int(nsel_cur) <= _pairwise_nsel_max
                    ):
                        try:
                            from asuka.cuda.cuda_backend import has_pairwise_hij_u64_device  # noqa: PLC0415

                            if bool(has_pairwise_hij_u64_device()):
                                use_pairwise_hij = True
                        except Exception:
                            pass

                    if use_dense_device_emitter or use_pairwise_hij:
                        diag_sel_h = None
                    elif bool(dense_key64_fast_active):
                        _ensure_selected_diag(sel_labels)
                        diag_sel_h = np.asarray([float(selected_diag_cache[int(j)]) for j in sel_labels], dtype=np.float64, order="C")
                    else:
                        diag_sel_h = np.empty((nsel_cur,), dtype=np.float64)
                        for pos, j in enumerate(sel_labels):
                            i_idx, hij = _get_connected_row_exact(int(j))
                            i_idx = np.asarray(i_idx, dtype=np.int64).ravel()
                            hij = np.asarray(hij, dtype=np.float64).ravel()
                            mask = i_idx == int(j)
                            if not bool(np.any(mask)):
                                raise RuntimeError(f"selected row {int(j)} is missing its diagonal entry")
                            diag_sel_h[pos] = float(hij[mask][0])

                    used_dense_device_emitter = False
                    used_pairwise_hij = False
                    if use_pairwise_hij:
                        try:
                            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                                pairwise_materialize_u64_device,
                                pairwise_hij_u64_device,
                            )
                            from asuka.sci.projected_apply import DenseMatrixProjectedHop  # noqa: PLC0415

                            sel_idx_full = np.asarray(sel_idx_arr, dtype=np.int64, order="C")
                            sel_u64_d = _cp.ascontiguousarray(
                                _cp.asarray(sel_idx_full.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel()
                            )
                            # Reuse the same h_base/eri4 cached inputs as the dense emitter
                            dense_dev = _ensure_exact_selected_dense_inputs()
                            h_base_d = dense_dev["h_base"]
                            eri4_d = dense_dev["eri4"]

                            materialized = pairwise_materialize_u64_device(
                                drt, _drt_dev, sel_u64_d, int(nsel_cur), _cp,
                            )
                            H_d, diag_d = pairwise_hij_u64_device(
                                drt, _drt_dev, sel_u64_d, int(nsel_cur),
                                h_base_d, eri4_d, materialized, _cp,
                            )
                            diag_sel_h = np.asarray(_cp.asnumpy(diag_d), dtype=np.float64, order="C")
                            for pos_diag, jj in enumerate(sel_idx_full.tolist()):
                                selected_diag_cache[int(jj)] = float(diag_sel_h[int(pos_diag)])
                            tuple_hop = DenseMatrixProjectedHop(
                                sel_idx=np.asarray(sel_idx_full, dtype=np.int64, order="C"),
                                H_d=H_d,
                                hdiag_d=diag_d,
                            )
                            projected_tuple_hop_cache = None  # no incremental for dense matrix
                            used_pairwise_hij = True
                            used_dense_device_emitter = True
                        except Exception as pairwise_e:
                            profile["pairwise_hij_fallback_reason"] = (
                                f"{type(pairwise_e).__name__}: {pairwise_e}"
                            )
                            use_pairwise_hij = False

                    if not used_pairwise_hij and use_dense_device_emitter:
                        try:
                            sel_idx_full = np.asarray(sel_idx_arr, dtype=np.int64, order="C")
                            tuple_hop = None
                            if (
                                bool(dense_key64_fast_active)
                                and int(sel_idx_full.size) >= 64
                                and projected_tuple_hop_cache is not None
                                and int(projected_tuple_hop_cache.sel_idx.size) <= int(sel_idx_full.size)
                                and np.array_equal(
                                    np.asarray(sel_idx_full[: int(projected_tuple_hop_cache.sel_idx.size)], dtype=np.int64),
                                    np.asarray(projected_tuple_hop_cache.sel_idx, dtype=np.int64),
                                )
                            ):
                                old_n = int(projected_tuple_hop_cache.sel_idx.size)
                                new_sel = np.asarray(sel_idx_full[old_n:], dtype=np.int64, order="C")
                                incremental_tuple_build = True
                                if int(new_sel.size) == 0:
                                    diag_sel_h = np.asarray(
                                        [float(selected_diag_cache[int(j)]) for j in sel_idx_full.tolist()],
                                        dtype=np.float64,
                                        order="C",
                                    )
                                    tuple_hop = projected_tuple_hop_cache.with_hdiag(
                                        hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                                    )
                                else:
                                    new_to_all_labels_d, new_to_all_src_d, new_to_all_hij_d, new_diag_d = _emit_exact_selected_tuples_device_dense(
                                        sel_idx=np.asarray(new_sel, dtype=np.int64),
                                        selected_target_idx=np.asarray(sel_idx_full, dtype=np.int64),
                                        return_diag=True,
                                    )
                                    new_diag_h = np.asarray(_cp.asnumpy(new_diag_d), dtype=np.float64)
                                    for pos_new, jj in enumerate(new_sel.tolist()):
                                        selected_diag_cache[int(jj)] = float(new_diag_h[int(pos_new)])
                                    diag_sel_h = np.asarray(
                                        [float(selected_diag_cache[int(j)]) for j in sel_idx_full.tolist()],
                                        dtype=np.float64,
                                        order="C",
                                    )
                                    if int(new_to_all_src_d.size) > 0:
                                        new_to_all_src_d = _cp.ascontiguousarray(
                                            new_to_all_src_d + np.int32(old_n)
                                        )
                                    if int(new_to_all_labels_d.size) > 0:
                                        delta_hop = ExactSelectedTupleProjectedHop.from_tuples(
                                            sel_idx=np.asarray(sel_idx_full, dtype=np.int64),
                                            labels=new_to_all_labels_d,
                                            src_pos=new_to_all_src_d,
                                            hij=new_to_all_hij_d,
                                            hdiag=None,
                                        )
                                        add_target_d, add_src_d, add_hij_d = delta_hop.to_local_tuples()
                                        mirror_mask = add_target_d < np.int32(old_n)
                                        if bool(_cp.any(mirror_mask).item()):
                                            base_target_d = _cp.ascontiguousarray(add_target_d.ravel())
                                            base_src_d = _cp.ascontiguousarray(add_src_d.ravel())
                                            base_hij_d = _cp.ascontiguousarray(add_hij_d.ravel())
                                            add_target_d = _cp.ascontiguousarray(
                                                _cp.concatenate(
                                                    (
                                                        base_target_d,
                                                        base_src_d[mirror_mask],
                                                    )
                                                ).ravel()
                                            )
                                            add_src_d = _cp.ascontiguousarray(
                                                _cp.concatenate(
                                                    (
                                                        base_src_d,
                                                        base_target_d[mirror_mask],
                                                    )
                                                ).ravel()
                                            )
                                            add_hij_d = _cp.ascontiguousarray(
                                                _cp.concatenate(
                                                    (
                                                        base_hij_d,
                                                        base_hij_d[mirror_mask],
                                                    )
                                                ).ravel()
                                            )
                                        delta_hop = ExactSelectedTupleProjectedHop.from_local_tuples(
                                            sel_idx=np.asarray(sel_idx_full, dtype=np.int64),
                                            target_local=add_target_d,
                                            src_pos=add_src_d,
                                            hij=add_hij_d,
                                            hdiag=None,
                                        )
                                        tuple_hop = projected_tuple_hop_cache.with_merged_hop(
                                            other=delta_hop,
                                            sel_idx=np.asarray(sel_idx_full, dtype=np.int64),
                                            hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                                        )
                                    else:
                                        tuple_hop = ExactSelectedTupleProjectedHop.from_local_tuples(
                                            sel_idx=np.asarray(sel_idx_full, dtype=np.int64),
                                            target_local=projected_tuple_hop_cache.coo_target_local_d,
                                            src_pos=projected_tuple_hop_cache.coo_src_pos_d,
                                            hij=projected_tuple_hop_cache.coo_hij_d,
                                            hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                                        )
                            if tuple_hop is None:
                                labels_d, src_d, hij_d, diag_d = _emit_exact_selected_tuples_device_dense(
                                    sel_idx=np.asarray(sel_idx_arr, dtype=np.int64),
                                    return_diag=True,
                                )
                                diag_sel_h = np.asarray(_cp.asnumpy(diag_d), dtype=np.float64, order="C")
                                for pos_diag, jj in enumerate(sel_idx_full.tolist()):
                                    selected_diag_cache[int(jj)] = float(diag_sel_h[int(pos_diag)])
                                tuple_hop = ExactSelectedTupleProjectedHop.from_tuples(
                                    sel_idx=np.asarray(sel_idx_arr, dtype=np.int64),
                                    labels=labels_d,
                                    src_pos=src_d,
                                    hij=hij_d,
                                    hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                                )
                            projected_tuple_hop_cache = tuple_hop
                            if incremental_tuple_build:
                                profile["projected_tuple_incremental_build_count"] = int(
                                    profile.get("projected_tuple_incremental_build_count", 0)
                                ) + 1
                            used_dense_device_emitter = True
                        except Exception as dense_emit_e:
                            profile["projected_solver_exact_tuple_device_fallback_reason"] = (
                                f"{type(dense_emit_e).__name__}: {dense_emit_e}"
                            )
                            use_dense_device_emitter = False
                            if diag_sel_h is None:
                                _ensure_selected_diag(sel_labels)
                                diag_sel_h = np.asarray(
                                    [float(selected_diag_cache[int(j)]) for j in sel_labels],
                                    dtype=np.float64,
                                    order="C",
                                )
                    if not use_dense_device_emitter:
                        label_parts: list[np.ndarray] = []
                        src_parts: list[np.ndarray] = []
                        hij_parts: list[np.ndarray] = []
                        for pos, j in enumerate(np.asarray(sel_idx_arr, dtype=np.int64).ravel().tolist()):
                            i_idx, hij = _get_connected_row_exact(int(j))
                            i_idx = np.asarray(i_idx, dtype=np.int64).ravel()
                            hij = np.asarray(hij, dtype=np.float64).ravel()
                            keep_labels: list[int] = []
                            keep_vals: list[float] = []
                            for ii, vv in zip(i_idx.tolist(), hij.tolist(), strict=False):
                                if int(ii) == int(j):
                                    continue
                                if int(ii) in loc_map_cur:
                                    keep_labels.append(int(ii))
                                    keep_vals.append(float(vv))
                            if keep_labels:
                                label_parts.append(np.asarray(keep_labels, dtype=np.int64))
                                src_parts.append(np.full((len(keep_labels),), int(pos), dtype=np.int32))
                                hij_parts.append(np.asarray(keep_vals, dtype=np.float64))
                        if label_parts:
                            tuple_hop = ExactSelectedTupleProjectedHop.from_tuples(
                                sel_idx=np.asarray(sel_idx_arr, dtype=np.int64),
                                labels=np.asarray(np.concatenate(label_parts), dtype=np.int64, order="C"),
                                src_pos=np.asarray(np.concatenate(src_parts), dtype=np.int32, order="C"),
                                hij=np.asarray(np.concatenate(hij_parts), dtype=np.float64, order="C"),
                                hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                            )
                        else:
                            tuple_hop = ExactSelectedTupleProjectedHop.from_tuples(
                                sel_idx=np.asarray(sel_idx_arr, dtype=np.int64),
                                labels=np.zeros((0,), dtype=np.int64),
                                src_pos=np.zeros((0,), dtype=np.int32),
                                hij=np.zeros((0,), dtype=np.float64),
                                hdiag=np.asarray(diag_sel_h, dtype=np.float64, order="C"),
                            )
                        projected_tuple_hop_cache = None

                    ncheck = min(int(nsel_cur), 4)
                    if ncheck > 0 and (not bool(dense_key64_fast_active) or bool(dense_key64_fast_debug_parity)):
                        v_check_h = np.eye(int(nsel_cur), int(ncheck), dtype=np.float64)
                        w_proj_h = np.asarray(
                            _cp.asnumpy(tuple_hop.hop_gpu(_cp.asarray(v_check_h, dtype=_cp.float64))),
                            dtype=np.float64,
                        )
                        w_ref_h = np.zeros((int(nsel_cur), int(ncheck)), dtype=np.float64)
                        for col in range(int(ncheck)):
                            j = int(np.asarray(sel_idx_arr, dtype=np.int64).ravel()[col])
                            i_idx, hij = _get_connected_row_exact(j)
                            for ii, vv in zip(np.asarray(i_idx, dtype=np.int64).tolist(), np.asarray(hij, dtype=np.float64).tolist(), strict=False):
                                loc_i = loc_map_cur.get(int(ii))
                                if loc_i is not None:
                                    w_ref_h[int(loc_i), int(col)] += float(vv)
                        mismatch = float(np.max(np.abs(w_proj_h - w_ref_h))) if w_ref_h.size else 0.0
                        if mismatch > 1e-8:
                            raise RuntimeError(
                                f"selected tuple projected-hop parity mismatch against exact selected rows (maxabs={mismatch:.3e})"
                            )

                    def _hop(v_d):
                        return tuple_hop.hop_gpu(v_d)

                    tuple_build_dt = float(time.perf_counter() - tuple_build_t0)
                    profile["timings_s"]["projected_tuple_build"] = float(
                        profile["timings_s"].get("projected_tuple_build", 0.0) + tuple_build_dt
                    )
                    profile["projected_tuple_build_history"].append(
                        {
                            "nsel": int(nsel_cur),
                            "dt_s": float(tuple_build_dt),
                            "dense_device": bool(used_dense_device_emitter),
                            "pairwise_hij": bool(used_pairwise_hij),
                            "incremental": bool(incremental_tuple_build),
                        }
                    )
                    return _hop, diag_sel_h, bool(used_dense_device_emitter)

                try:
                    if backend_effective in ("cuda_key64", "cuda_idx64"):
                        _hop, solver_hdiag, used_dense_device_emitter = _build_projected_tuple_hop()
                        solver_backend = (
                            "cuda_davidson_projected_exact_tuples_device"
                            if bool(used_dense_device_emitter)
                            else "cuda_davidson_projected_exact_tuples"
                        )
                    elif bool(projected_solver_matrix_free):
                        raise RuntimeError("matrix-free projected solver is only supported on the cuda_key64 path")
                    else:
                        _hop, solver_hdiag = _build_sell_hop()
                except Exception as projected_hop_e:
                    if bool(projected_solver_matrix_free):
                        profile["projected_solver_matrix_free_fallback_reason"] = (
                            f"{type(projected_hop_e).__name__}: {projected_hop_e}"
                        )
                    _hop, solver_hdiag = _build_sell_hop()

                dav_res = davidson_sym_gpu(
                    _hop,
                    x0=guess_list,
                    hdiag=np.asarray(solver_hdiag, dtype=np.float64),
                    nroots=int(nroots),
                    max_cycle=max(4, int(davidson_max_cycle)),
                    max_space=max(int(davidson_max_space), int(nroots) + 2),
                    tol=float(davidson_tol),
                    subspace_eigh_cpu=False,
                    batch_convergence_transfer=True,
                )
                e_var_cur = np.asarray(dav_res.e, dtype=np.float64)
                c_sel_cur = np.column_stack([np.asarray(x, dtype=np.float64) for x in dav_res.x])
                profile["projected_solver_gpu_effective"] = True
                profile["projected_solver_backend"] = str(solver_backend)
                profile["projected_solver_last_stats"] = dict(dav_res.stats or {})
                profile["timings_s"]["projected_solver_total"] = float(
                    profile["timings_s"].get("projected_solver_total", 0.0) + (time.perf_counter() - solve_t0)
                )
                perm = _match_roots_by_overlap(prev_guess, c_sel_cur, e_var_cur)
                if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
                    e_var_cur = np.asarray(e_var_cur[perm], dtype=np.float64)
                    c_sel_cur = np.asarray(c_sel_cur[:, perm], dtype=np.float64)
                return np.asarray(e_var_cur, dtype=np.float64), np.asarray(c_sel_cur, dtype=np.float64, order="C"), False
            except Exception as projected_e:
                profile["projected_solver_gpu_effective"] = False
                profile["projected_solver_fallback_reason"] = f"{type(projected_e).__name__}: {projected_e}"

        h_sub = _ensure_h_sub()
        solve_perm = _solver_reorder_perm(sel_idx_arr, h_sub)
        if np.array_equal(solve_perm, np.arange(int(sel_idx_arr.size), dtype=np.int32)):
            h_solve = h_sub
            inv_solve_perm = None
        else:
            h_solve = h_sub[solve_perm, :][:, solve_perm].tocsr()
            inv_solve_perm = np.empty_like(solve_perm)
            inv_solve_perm[solve_perm] = np.arange(int(solve_perm.size), dtype=np.int32)
        if ci0_sub and inv_solve_perm is not None:
            ci0_sub = [np.asarray(v, dtype=np.float64)[solve_perm] for v in ci0_sub]
        v0 = None if not ci0_sub else np.asarray(ci0_sub[0], dtype=np.float64)
        e_var_cur, c_sel_cur = _solve_subspace(
            h_solve,
            nroots=int(nroots),
            dense_limit=max(64, int(davidson_max_space) * 8),
            eigsh_tol=float(davidson_tol),
            v0=v0,
        )
        c_sel_cur = np.asarray(c_sel_cur, dtype=np.float64, order="C")
        if inv_solve_perm is not None:
            c_sel_cur = np.asarray(c_sel_cur[inv_solve_perm, :], dtype=np.float64, order="C")
        perm = _match_roots_by_overlap(prev_guess, c_sel_cur, e_var_cur)
        if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
            e_var_cur = np.asarray(e_var_cur[perm], dtype=np.float64)
            c_sel_cur = np.asarray(c_sel_cur[:, perm], dtype=np.float64)
        profile["projected_solver_gpu_effective"] = False
        profile["projected_solver_backend"] = "cpu_sparse_eigsh"
        profile["timings_s"]["projected_solver_total"] = float(
            profile["timings_s"].get("projected_solver_total", 0.0) + (time.perf_counter() - solve_t0)
        )
        return np.asarray(e_var_cur, dtype=np.float64), np.asarray(c_sel_cur, dtype=np.float64, order="C"), bool(inv_solve_perm is not None)

    def _normalize_coeff_block(c_block: np.ndarray) -> np.ndarray:
        c_block = np.asarray(c_block, dtype=np.float64, order="C")
        if c_block.size == 0:
            return c_block
        norms = np.linalg.norm(c_block, axis=0)
        norms = np.where(norms > 0.0, norms, 1.0)
        return np.asarray(c_block / norms[None, :], dtype=np.float64, order="C")

    def _apply_seed_expansion(
        *,
        c_sel_cur: np.ndarray,
        added_idx: list[int],
        seeds_out: dict[str, Any],
    ) -> np.ndarray:
        if not added_idx:
            return np.asarray(c_sel_cur, dtype=np.float64, order="C")
        seed_idx = np.asarray(seeds_out.get("seed_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).ravel()
        seed_c1 = np.asarray(seeds_out.get("seed_c1", np.zeros((0, int(nroots)), dtype=np.float64)), dtype=np.float64)
        seed_map: dict[int, np.ndarray] = {}
        for pos, ii in enumerate(seed_idx.tolist()):
            seed_map[int(ii)] = np.asarray(seed_c1[pos], dtype=np.float64)
        add_block = np.zeros((int(len(added_idx)), int(nroots)), dtype=np.float64)
        for row, ii in enumerate(added_idx):
            add_block[row, :] = seed_map.get(int(ii), np.zeros((int(nroots),), dtype=np.float64))
        return _normalize_coeff_block(np.vstack((np.asarray(c_sel_cur, dtype=np.float64), add_block)))

    def _run_bulk_growth_loop() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal sel, loc_map, prev_c_sel, cuda_selector_enabled, prev_cuda_ncand_hint, exact_external_selector_enabled
        profile["solve_count"] = 0
        macro_iter = 0
        e_var_last = np.zeros((int(nroots),), dtype=np.float64)
        e_pt2_last = np.zeros((int(nroots),), dtype=np.float64)
        c_sel_last = np.zeros((int(len(sel)), int(nroots)), dtype=np.float64)
        final_solver_reordered = False
        while macro_iter < int(max_iter):
            macro_iter += 1
            sel_idx = np.asarray(sel, dtype=np.int64)
            e_var_last, c_sel_last, final_solver_reordered = _solve_selected_subspace(
                sel_idx_arr=sel_idx,
                loc_map_cur=loc_map,
                prev_guess=prev_c_sel,
            )
            prev_c_sel = np.asarray(c_sel_last, dtype=np.float64, order="C")
            profile["solve_count"] = int(profile.get("solve_count", 0)) + 1

            sel_work = list(int(x) for x in sel)
            loc_work = dict(loc_map)
            c_sel_work = np.asarray(c_sel_last, dtype=np.float64, order="C")
            added_in_macro: list[int] = []
            added_score_map: dict[int, float] = {}
            macro_growth_cap = max(256, int(np.ceil(float(macro_growth_resync_frac) * max(1, len(sel_work)))))

            for micro_iter in range(1, int(macro_growth_steps) + 1):
                remaining = max(0, int(max_ncsf) - int(len(sel_work)))
                max_add = min(int(grow_by), int(remaining))
                selector_iter_stats: dict[str, Any] = {}
                cuda_iter_stats: dict[str, Any] = {}
                seeds_out: dict[str, Any] = {}
                hb_eps_iter = _compute_hb_eps_iter(sel_size=int(len(sel_work)))
                if int(max_add) > 0:
                    if exact_external_selector_enabled:
                        selector_iter_stats.update(
                            {
                                "selector_bucketed": False,
                                "selector_nbuckets": 1,
                                "selector_bucket_kind": "exact_external_projected_apply",
                                "selector_active_frontier_edges": 0,
                            }
                        )
                    elif cuda_selector_enabled:
                        selector_plan_fast = _plan_cuda_selector_buckets_fast(
                            drt=drt,
                            sel_size=int(len(sel_work)),
                            prev_ncand_hint=int(prev_cuda_ncand_hint),
                            max_add=int(max_add),
                        )
                        selector_iter_stats.update(
                            {
                                "selector_bucketed": bool(selector_plan_fast["selector_bucketed"]),
                                "selector_nbuckets": int(selector_plan_fast["selector_nbuckets"]),
                                "selector_bucket_kind": str(selector_plan_fast["selector_bucket_kind"]),
                                "selector_active_frontier_edges": int(selector_plan_fast["selector_active_frontier_edges"]),
                            }
                        )
                    else:
                        selector_plan = _plan_selector_buckets(
                            drt,
                            h1e,
                            eri,
                            sel=np.asarray(sel_work, dtype=np.int64).tolist(),
                            c_sel=c_sel_work,
                            max_out=int(row_max_out),
                            screening=None,
                            state_cache=state_cache,
                            row_cache=persistent_row_cache,
                        )
                        selector_iter_stats.update(
                            {
                                "selector_bucketed": bool(selector_plan.bucketed),
                                "selector_nbuckets": int(selector_plan.nbuckets),
                                "selector_bucket_kind": str(selector_plan.bucket_kind),
                                "selector_active_frontier_edges": int(selector_plan.active_frontier_edges),
                            }
                        )
                if exact_external_selector_enabled:
                    try:
                        new_idx, e_pt2_last, exact_iter_stats = _exact_external_select(
                            sel_idx_i64=np.asarray(sel_work, dtype=np.int64),
                            c_sel_arr=c_sel_work,
                            e_var_arr=e_var_last,
                            max_add_i=int(max_add),
                            eps_val=float(hb_eps_iter),
                            seeds_out=seeds_out,
                            selection_policy=(
                                "threshold"
                                if backend_effective == "cuda_key64"
                                else "exact_topk"
                            ),
                        )
                        selector_iter_stats.update(dict(exact_iter_stats))
                        profile["exact_external_selector_effective"] = True
                    except Exception as exact_step_e:
                        exact_external_selector_enabled = False
                        profile["exact_external_selector_effective"] = False
                        profile["exact_external_selector_fallback_iter"] = int(macro_iter)
                        profile["exact_external_selector_fallback_reason"] = (
                            f"{type(exact_step_e).__name__}: {exact_step_e}"
                        )
                if cuda_selector_enabled and not exact_external_selector_enabled:
                    try:
                        eps_run = 0.0 if selection_mode_s == "frontier_hash" else float(hb_eps_iter)
                        new_idx, e_pt2_last, cuda_iter_stats = _cuda_select_external(
                            sel_idx_i64=np.asarray(sel_work, dtype=np.int64),
                            c_sel_arr=c_sel_work,
                            e_var_arr=e_var_last,
                            max_add_i=int(max_add),
                            eps_val=float(eps_run),
                            bucket_bounds=None if int(max_add) <= 0 else tuple(selector_plan_fast["bucket_bounds"]),
                            seeds_out=seeds_out,
                        )
                        prev_cuda_ncand_hint = int(cuda_iter_stats.get("ncand", prev_cuda_ncand_hint))
                    except Exception as cuda_step_e:
                        cuda_selector_enabled = False
                        profile["cuda_selector_step_fallback_iter"] = int(macro_iter)
                        profile["cuda_selector_step_fallback_reason"] = f"{type(cuda_step_e).__name__}: {cuda_step_e}"
                        profile["driver"] = "sparse_row_oracle"
                if not cuda_selector_enabled and not exact_external_selector_enabled:
                    if selection_mode_s == "frontier_hash":
                        new_idx, e_pt2_last, _stats = frontier_selector.build_and_score(
                            sel_idx=np.asarray(sel_work, dtype=np.int64),
                            c_sel=c_sel_work,
                            e_var=e_var_last,
                            max_add=int(max_add),
                            select_threshold=select_threshold,
                            row_cache=persistent_row_cache,
                            stats_out=selector_iter_stats,
                            seeds_out=seeds_out,
                        )
                    else:
                        new_idx, e_pt2_last = heat_bath_select_and_pt2_sparse(
                            drt,
                            h1e,
                            eri,
                            sel_idx=np.asarray(sel_work, dtype=np.int64),
                            c_sel=c_sel_work,
                            e_var=e_var_last,
                            max_add=int(max_add),
                            epsilon=float(hb_eps_iter),
                            denom_floor=float(denom_floor),
                            hdiag_lookup=hdiag_lookup,
                            max_out=int(row_max_out),
                            screening=None,
                            state_cache=state_cache,
                            row_cache=persistent_row_cache,
                            stats_out=selector_iter_stats,
                            seeds_out=seeds_out,
                        )
                added_idx: list[int] = []
                seed_w = np.asarray(seeds_out.get("seed_w", np.zeros((0,), dtype=np.float64)), dtype=np.float64).ravel()
                seed_idx = np.asarray(seeds_out.get("seed_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).ravel()
                seed_w_map = {int(ii): float(seed_w[pos]) for pos, ii in enumerate(seed_idx.tolist()) if pos < int(seed_w.size)}
                for ii in new_idx:
                    jj = int(ii)
                    if jj in loc_work:
                        continue
                    loc_work[jj] = int(len(sel_work))
                    sel_work.append(jj)
                    added_idx.append(jj)
                    added_score_map[jj] = max(float(added_score_map.get(jj, 0.0)), float(seed_w_map.get(jj, 0.0)))
                    if backend_effective != "cuda_key64" and len(added_idx) >= int(max_add):
                        break
                if added_idx:
                    frontier_selector.mark_selected(added_idx)
                    c_sel_work = _apply_seed_expansion(c_sel_cur=c_sel_work, added_idx=added_idx, seeds_out=seeds_out)
                    added_in_macro.extend(int(x) for x in added_idx)
                active_mask = np.any(np.abs(c_sel_work) > 0.0, axis=1)
                active_sources = int(np.count_nonzero(active_mask))
                rec = {
                    "iter": int(len(history) + 1),
                    "macro_iter": int(macro_iter),
                    "micro_iter": int(micro_iter),
                    "nsel": int(c_sel_work.shape[0]),
                    "nadd": int(len(added_idx)),
                    "e_var": np.asarray(e_var_last + float(ecore), dtype=np.float64),
                    "e_pt2": np.asarray(e_pt2_last, dtype=np.float64),
                    "e_tot": np.asarray(e_var_last + e_pt2_last + float(ecore), dtype=np.float64),
                    "active_sources": int(active_sources),
                    "active_tiles": int(active_sources),
                    "epq_mode": str(profile.get("epq_mode", epq_mode)),
                    "davidson_niter": -1,
                    "davidson_retry_count": 0,
                    "davidson_attempts": [],
                    "cpu_subspace_refined": False,
                    "solver_reordered": bool(final_solver_reordered),
                    "solve_refreshed": bool(micro_iter == 1),
                }
                if cuda_iter_stats:
                    rec["cuda_selector"] = dict(cuda_iter_stats)
                if selector_iter_stats:
                    rec["selector"] = dict(selector_iter_stats)
                    profile["selector_bucketed_any"] = bool(
                        profile.get("selector_bucketed_any", False) or selector_iter_stats.get("selector_bucketed", False)
                    )
                    selector_backend = selector_iter_stats.get("selector_backend")
                    if selector_backend is not None:
                        profile["selector_backend_history"].append(str(selector_backend))
                    selection_policy = selector_iter_stats.get("selection_policy")
                    if selection_policy is not None:
                        profile["selection_policy_history"].append(str(selection_policy))
                    threshold_tau = selector_iter_stats.get("threshold_tau")
                    if threshold_tau is not None:
                        profile["threshold_tau_history"].append(float(threshold_tau))
                history.append(rec)
                if not added_idx or len(sel_work) >= int(max_ncsf):
                    break
                if len(added_in_macro) >= int(macro_growth_cap):
                    break

            if added_in_macro:
                keep_added_all = list(dict.fromkeys(int(x) for x in added_in_macro))
                keep_added = list(keep_added_all)
                if backend_effective == "cuda_key64" and len(keep_added) > int(macro_growth_cap):
                    score_arr = np.asarray([float(added_score_map.get(int(ii), 0.0)) for ii in keep_added], dtype=np.float64)
                    idx_arr = np.asarray(keep_added, dtype=np.int64)
                    keep_n = min(int(macro_growth_cap), int(idx_arr.size))
                    part = np.argpartition(score_arr, -keep_n)[-keep_n:]
                    order = np.lexsort((idx_arr[part], -score_arr[part]))
                    keep_added = [int(x) for x in idx_arr[part][order].tolist()]
                profile["macro_trim_sizes"].append(max(0, int(len(keep_added_all)) - int(len(keep_added))))
                if keep_added != keep_added_all:
                    keep_rows = np.asarray(
                        list(range(int(len(sel)))) + [int(loc_work[int(ii)]) for ii in keep_added],
                        dtype=np.int64,
                    )
                    c_sel_work = _normalize_coeff_block(np.asarray(c_sel_work[keep_rows, :], dtype=np.float64, order="C"))
                if h_builder is not None or not bool(dense_key64_fast_active):
                    _ensure_h_builder().extend(keep_added)
                    sel = h_builder.sel  # type: ignore[union-attr]
                    loc_map = h_builder.loc_map  # type: ignore[union-attr]
                else:
                    for jj in keep_added:
                        kk = int(jj)
                        if kk in loc_map:
                            continue
                        loc_map[kk] = int(len(sel))
                        sel.append(kk)
                    if keep_added and not bool(dense_key64_fast_active):
                        _ensure_selected_diag([int(x) for x in keep_added])
                prev_c_sel = np.asarray(c_sel_work, dtype=np.float64, order="C")
            if not added_in_macro or len(sel) >= int(max_ncsf):
                break

        sel_idx = np.asarray(sel, dtype=np.int64)
        e_var_last, c_sel_last, final_solver_reordered = _solve_selected_subspace(
            sel_idx_arr=sel_idx,
            loc_map_cur=loc_map,
            prev_guess=prev_c_sel,
        )
        profile["solve_count"] = int(profile.get("solve_count", 0)) + 1
        profile["solver_reordered_final"] = bool(final_solver_reordered)
        if exact_external_selector_enabled:
            _new_idx, e_pt2_last, exact_final_stats = _exact_external_select(
                sel_idx_i64=np.asarray(sel_idx, dtype=np.int64),
                c_sel_arr=c_sel_last,
                e_var_arr=e_var_last,
                max_add_i=0,
                eps_val=float(_compute_hb_eps_iter(sel_size=int(sel_idx.size))),
            )
            profile["exact_external_selector_final_stats"] = dict(exact_final_stats)
            assert len(_new_idx) == 0
        elif cuda_selector_enabled:
            eps_final = 0.0 if selection_mode_s == "frontier_hash" else _compute_hb_eps_iter(sel_size=int(sel_idx.size))
            selector_plan_final = _plan_cuda_selector_buckets_fast(
                drt=drt,
                sel_size=int(sel_idx.size),
                prev_ncand_hint=int(prev_cuda_ncand_hint),
                max_add=0,
            )
            _new_idx, e_pt2_last, cuda_final_stats = _cuda_select_external(
                sel_idx_i64=np.asarray(sel_idx, dtype=np.int64),
                c_sel_arr=c_sel_last,
                e_var_arr=e_var_last,
                max_add_i=0,
                eps_val=float(eps_final),
                bucket_bounds=tuple(selector_plan_final["bucket_bounds"]),
            )
            profile["cuda_selector_final_stats"] = dict(cuda_final_stats)
            assert len(_new_idx) == 0
        elif selection_mode_s == "frontier_hash":
            _new_idx, e_pt2_last, _stats = frontier_selector.build_and_score(
                sel_idx=np.asarray(sel_idx, dtype=np.int64),
                c_sel=c_sel_last,
                e_var=e_var_last,
                max_add=0,
                select_threshold=select_threshold,
                row_cache=persistent_row_cache,
            )
            assert len(_new_idx) == 0
        else:
            if str(pt2_mode) in ("streaming", "semistochastic"):
                _sel_set = {int(s) for s in np.asarray(sel_idx, dtype=np.int64).ravel().tolist()}
                _pt2_func = streaming_pt2_deterministic if str(pt2_mode) == "streaming" else semistochastic_pt2
                _pt2_kw: dict[str, Any] = dict(
                    drt=drt, h1e=h1e, eri=eri,
                    sel=np.asarray(sel_idx, dtype=np.int64).ravel().tolist(),
                    selected_set=_sel_set,
                    c_sel=c_sel_last,
                    e_var=e_var_last,
                    hdiag_lookup=hdiag_lookup,
                    denom_floor=float(denom_floor),
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                    bucket_size=int(pt2_bucket_size),
                )
                if str(pt2_mode) == "semistochastic":
                    _pt2_kw.update(n_det_sources=pt2_n_det_sources, n_stoch_samples=pt2_n_stoch_samples,
                                   n_stoch_batches=pt2_n_stoch_batches, seed=pt2_seed)
                _pt2_res = _pt2_func(**_pt2_kw)
                e_pt2_last = _pt2_res.e_pt2
                _new_idx = []
            else:
                _new_idx, e_pt2_last = heat_bath_select_and_pt2_sparse(
                    drt,
                    h1e,
                    eri,
                    sel_idx=np.asarray(sel_idx, dtype=np.int64),
                    c_sel=c_sel_last,
                    e_var=e_var_last,
                    max_add=0,
                    epsilon=float(_compute_hb_eps_iter(sel_size=int(sel_idx.size))),
                    denom_floor=float(denom_floor),
                    hdiag_lookup=hdiag_lookup,
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                )
            assert len(_new_idx) == 0
        return sel_idx, np.asarray(e_var_last, dtype=np.float64), np.asarray(c_sel_last, dtype=np.float64, order="C"), np.asarray(e_pt2_last, dtype=np.float64)

    if macro_schedule_enabled:
        sel_idx, e_var, c_sel, e_pt2 = _run_bulk_growth_loop()
        roots = _sparse_roots_from_selected(sel_idx, c_sel)
        sel_key_u64 = None
        label_kind = "csf_idx"
        if state_rep_s == "key64":
            from asuka.qmc.cuda_backend import csf_idx_to_key64_host  # noqa: PLC0415

            sel_key_u64 = np.asarray(csf_idx_to_key64_host(drt, sel_idx, state_cache=None), dtype=np.uint64, order="C")
            label_kind = "key64"
        elif state_rep_s == "i64":
            sel_idx_i64 = np.asarray(sel_idx, dtype=np.int64).ravel()
            if sel_idx_i64.size:
                if int(np.min(sel_idx_i64)) < 0:
                    raise ValueError("selected indices must be non-negative for idx64 label mode")
                if int(np.max(sel_idx_i64)) >= int(ncsf):
                    raise ValueError("selected indices must be < drt.ncsf for idx64 label mode")
            sel_key_u64 = np.asarray(sel_idx_i64, dtype=np.uint64, order="C")
            label_kind = "idx64"
        return CIPSITrialSpaceResult(
            e_var=np.asarray(e_var + float(ecore), dtype=np.float64),
            e_pt2=np.asarray(e_pt2, dtype=np.float64),
            e_tot=np.asarray(e_var + e_pt2 + float(ecore), dtype=np.float64),
            sel_idx=sel_idx,
            sel_key_u64=sel_key_u64,
            label_kind=label_kind,
            ci_sel=np.asarray(c_sel, dtype=np.float64),
            roots=roots,
            history=history,
            profile=dict(profile),
            epq_mode=str(profile.get("epq_mode", epq_mode_s)),
            ncsf=int(ncsf),
        )

    for it in range(1, int(max_iter) + 1):
        sel_idx = np.asarray(sel, dtype=np.int64)
        h_sub = _ensure_h_builder().to_csr()
        solve_perm = _solver_reorder_perm(sel_idx, h_sub)
        if np.array_equal(solve_perm, np.arange(int(sel_idx.size), dtype=np.int32)):
            h_solve = h_sub
            inv_solve_perm = None
        else:
            h_solve = h_sub[solve_perm, :][:, solve_perm].tocsr()
            inv_solve_perm = np.empty_like(solve_perm)
            inv_solve_perm[solve_perm] = np.arange(int(solve_perm.size), dtype=np.int32)
        ci0_sub = _build_ci0_subspace_sparse(
            sel_idx=sel_idx,
            loc_map=loc_map,
            nroots=int(nroots),
            ci0_sparse=ci0_sparse,
            prev_c_sel=prev_c_sel,
        )
        if ci0_sub and inv_solve_perm is not None:
            ci0_sub = [np.asarray(v, dtype=np.float64)[solve_perm] for v in ci0_sub]
        v0 = None if not ci0_sub else np.asarray(ci0_sub[0], dtype=np.float64)
        e_var, c_sel = _solve_subspace(
            h_solve,
            nroots=int(nroots),
            dense_limit=max(64, int(davidson_max_space) * 8),
            eigsh_tol=float(davidson_tol),
            v0=v0,
        )
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if inv_solve_perm is not None:
            c_sel = np.asarray(c_sel[inv_solve_perm, :], dtype=np.float64, order="C")
        perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
        if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
            e_var = np.asarray(e_var[perm], dtype=np.float64)
            c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)

        max_add = min(int(grow_by), int(max_ncsf) - int(sel_idx.size))
        cuda_iter_stats: dict[str, Any] = {}
        selector_iter_stats: dict[str, Any] = {}
        selector_plan = None
        hb_eps_iter = 0.0
        if selection_mode_s == "heat_bath":
            if str(hb_eps_schedule).lower() == "adaptive":
                frac = 0.0 if int(max_ncsf) <= int(init_ncsf) else (int(sel_idx.size) - int(init_ncsf)) / max(
                    1, int(max_ncsf) - int(init_ncsf)
                )
                frac = float(np.clip(frac, 0.0, 1.0))
                hb_eps_iter = float(hb_eps_init) * (float(hb_eps_final) / float(hb_eps_init)) ** frac
            else:
                hb_eps_iter = float(hb_epsilon)

        if int(max_add) > 0:
            if exact_external_selector_enabled:
                selector_iter_stats.update(
                    {
                        "selector_bucketed": False,
                        "selector_nbuckets": 1,
                        "selector_bucket_kind": "exact_external_projected_apply",
                        "selector_active_frontier_edges": 0,
                    }
                )
            elif cuda_selector_enabled:
                selector_plan_fast = _plan_cuda_selector_buckets_fast(
                    drt=drt,
                    sel_size=int(sel_idx.size),
                    prev_ncand_hint=int(prev_cuda_ncand_hint),
                    max_add=int(max_add),
                )
                selector_iter_stats.update(
                    {
                        "selector_bucketed": bool(selector_plan_fast["selector_bucketed"]),
                        "selector_nbuckets": int(selector_plan_fast["selector_nbuckets"]),
                        "selector_bucket_kind": str(selector_plan_fast["selector_bucket_kind"]),
                        "selector_active_frontier_edges": int(selector_plan_fast["selector_active_frontier_edges"]),
                    }
                )
            else:
                selector_plan = _plan_selector_buckets(
                    drt,
                    h1e,
                    eri,
                    sel=sel_idx.tolist(),
                    c_sel=c_sel,
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                )
                selector_iter_stats.update(
                    {
                        "selector_bucketed": bool(selector_plan.bucketed),
                        "selector_nbuckets": int(selector_plan.nbuckets),
                        "selector_bucket_kind": str(selector_plan.bucket_kind),
                        "selector_active_frontier_edges": int(selector_plan.active_frontier_edges),
                    }
                )

        if exact_external_selector_enabled:
            try:
                new_idx, e_pt2, exact_iter_stats = _exact_external_select(
                    sel_idx_i64=sel_idx,
                    c_sel_arr=c_sel,
                    e_var_arr=e_var,
                    max_add_i=int(max_add),
                    eps_val=float(hb_eps_iter),
                )
                selector_iter_stats.update(dict(exact_iter_stats))
                profile["exact_external_selector_effective"] = True
            except Exception as exact_step_e:
                exact_external_selector_enabled = False
                profile["exact_external_selector_effective"] = False
                profile["exact_external_selector_fallback_iter"] = int(it)
                profile["exact_external_selector_fallback_reason"] = (
                    f"{type(exact_step_e).__name__}: {exact_step_e}"
                )

        if cuda_selector_enabled and not exact_external_selector_enabled:
            try:
                eps_run = 0.0 if selection_mode_s == "frontier_hash" else float(hb_eps_iter)
                new_idx, e_pt2, cuda_iter_stats = _cuda_select_external(
                    sel_idx_i64=sel_idx,
                    c_sel_arr=c_sel,
                    e_var_arr=e_var,
                    max_add_i=int(max_add),
                    eps_val=float(eps_run),
                    bucket_bounds=None if int(max_add) <= 0 else tuple(selector_plan_fast["bucket_bounds"]),
                )
                prev_cuda_ncand_hint = int(cuda_iter_stats.get("ncand", prev_cuda_ncand_hint))
            except Exception as cuda_step_e:
                cuda_selector_enabled = False
                profile["cuda_selector_step_fallback_iter"] = int(it)
                profile["cuda_selector_step_fallback_reason"] = f"{type(cuda_step_e).__name__}: {cuda_step_e}"
                profile["driver"] = "sparse_row_oracle"

        if not cuda_selector_enabled and not exact_external_selector_enabled:
            if selection_mode_s == "frontier_hash":
                new_idx, e_pt2, _stats = frontier_selector.build_and_score(
                    sel_idx=sel_idx,
                    c_sel=c_sel,
                    e_var=e_var,
                    max_add=int(max_add),
                    select_threshold=select_threshold,
                    row_cache=persistent_row_cache,
                    stats_out=selector_iter_stats,
                )
            else:
                new_idx, e_pt2 = heat_bath_select_and_pt2_sparse(
                    drt,
                    h1e,
                    eri,
                    sel_idx=sel_idx,
                    c_sel=c_sel,
                    e_var=e_var,
                    max_add=int(max_add),
                    epsilon=float(hb_eps_iter),
                    denom_floor=float(denom_floor),
                    hdiag_lookup=hdiag_lookup,
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                    stats_out=selector_iter_stats,
                )

        added_idx: list[int] = []
        remaining = max(0, int(max_ncsf) - int(len(sel)))
        if remaining > 0:
            for ii in new_idx:
                jj = int(ii)
                if jj in loc_map:
                    continue
                added_idx.append(jj)
                if len(added_idx) >= remaining:
                    break
        if added_idx:
            _ensure_h_builder().extend(added_idx)
            sel = h_builder.sel  # type: ignore[union-attr]
            loc_map = h_builder.loc_map  # type: ignore[union-attr]
            frontier_selector.mark_selected(added_idx)

        active_mask = np.any(np.abs(c_sel) > 0.0, axis=1)
        active_sources = np.asarray(sel_idx[active_mask], dtype=np.int64)
        active_tiles = int(active_sources.size)

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
            "davidson_niter": -1,
            "davidson_retry_count": 0,
            "davidson_attempts": [],
            "cpu_subspace_refined": False,
            "solver_reordered": bool(inv_solve_perm is not None),
        }
        if cuda_iter_stats:
            rec["cuda_selector"] = dict(cuda_iter_stats)
        if selector_iter_stats:
            rec["selector"] = dict(selector_iter_stats)
            profile["selector_bucketed_any"] = bool(profile.get("selector_bucketed_any", False) or selector_iter_stats.get("selector_bucketed", False))
            selector_backend = selector_iter_stats.get("selector_backend")
            if selector_backend is not None:
                profile["selector_backend_history"].append(str(selector_backend))
            selection_policy = selector_iter_stats.get("selection_policy")
            if selection_policy is not None:
                profile["selection_policy_history"].append(str(selection_policy))
            threshold_tau = selector_iter_stats.get("threshold_tau")
            if threshold_tau is not None:
                profile["threshold_tau_history"].append(float(threshold_tau))
        history.append(rec)
        prev_c_sel = np.asarray(c_sel, dtype=np.float64)
        if len(new_idx) == 0:
            final_sel_idx_cache = np.asarray(sel_idx, dtype=np.int64)
            final_e_var_cache = np.asarray(e_var, dtype=np.float64)
            final_c_sel_cache = np.asarray(c_sel, dtype=np.float64, order="C")
            final_e_pt2_cache = np.asarray(e_pt2, dtype=np.float64)
            final_cuda_stats_cache = dict(cuda_iter_stats) if cuda_iter_stats else None
            final_solver_reordered_cache = bool(inv_solve_perm is not None)
        if verbose:
            print(f"[GPU-CIPSI] iter={it} nsel={c_sel.shape[0]} nadd={len(new_idx)} e0={float(e_var[0] + ecore):.12f}")
        if len(new_idx) == 0 or len(sel) >= int(max_ncsf):
            break

    if (
        final_sel_idx_cache is not None
        and final_e_var_cache is not None
        and final_c_sel_cache is not None
        and final_e_pt2_cache is not None
        and int(final_sel_idx_cache.size) == int(len(sel))
        and np.array_equal(final_sel_idx_cache, np.asarray(sel, dtype=np.int64))
    ):
        sel_idx = np.asarray(final_sel_idx_cache, dtype=np.int64)
        e_var = np.asarray(final_e_var_cache, dtype=np.float64)
        c_sel = np.asarray(final_c_sel_cache, dtype=np.float64, order="C")
        e_pt2 = np.asarray(final_e_pt2_cache, dtype=np.float64)
        profile["davidson_final_retry_count"] = 0
        profile["davidson_final_attempts"] = []
        profile["davidson_final_niter"] = -1
        profile["cpu_subspace_refined_final"] = 0.0
        profile["solver_reordered_final"] = bool(final_solver_reordered_cache)
        if final_cuda_stats_cache is not None:
            profile["cuda_selector_final_stats"] = dict(final_cuda_stats_cache)
    else:
        sel_idx = np.asarray(sel, dtype=np.int64)
        h_sub = _ensure_h_builder().to_csr()
        solve_perm = _solver_reorder_perm(sel_idx, h_sub)
        if np.array_equal(solve_perm, np.arange(int(sel_idx.size), dtype=np.int32)):
            h_solve = h_sub
            inv_solve_perm = None
        else:
            h_solve = h_sub[solve_perm, :][:, solve_perm].tocsr()
            inv_solve_perm = np.empty_like(solve_perm)
            inv_solve_perm[solve_perm] = np.arange(int(solve_perm.size), dtype=np.int32)
        ci0_sub = _build_ci0_subspace_sparse(
            sel_idx=sel_idx,
            loc_map=loc_map,
            nroots=int(nroots),
            ci0_sparse=ci0_sparse,
            prev_c_sel=prev_c_sel,
        )
        if ci0_sub and inv_solve_perm is not None:
            ci0_sub = [np.asarray(v, dtype=np.float64)[solve_perm] for v in ci0_sub]
        v0 = None if not ci0_sub else np.asarray(ci0_sub[0], dtype=np.float64)
        e_var, c_sel = _solve_subspace(
            h_solve,
            nroots=int(nroots),
            dense_limit=max(64, int(davidson_max_space) * 8),
            eigsh_tol=float(davidson_tol),
            v0=v0,
        )
        profile["davidson_final_retry_count"] = 0
        profile["davidson_final_attempts"] = []
        profile["davidson_final_niter"] = -1
        profile["cpu_subspace_refined_final"] = 0.0
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if inv_solve_perm is not None:
            c_sel = np.asarray(c_sel[inv_solve_perm, :], dtype=np.float64, order="C")
            profile["solver_reordered_final"] = True
        else:
            profile["solver_reordered_final"] = False
        perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
        if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
            e_var = np.asarray(e_var[perm], dtype=np.float64)
            c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)
        _eps_final = 0.0
        if selection_mode_s == "heat_bath":
            _eps_final = float(hb_eps_final) if str(hb_eps_schedule).lower() == "adaptive" else float(hb_epsilon)
        if exact_external_selector_enabled:
            _new_idx, e_pt2, exact_final_stats = _exact_external_select(
                sel_idx_i64=np.asarray(sel_idx, dtype=np.int64),
                c_sel_arr=c_sel,
                e_var_arr=e_var,
                max_add_i=0,
                eps_val=float(_eps_final),
            )
            profile["exact_external_selector_final_stats"] = dict(exact_final_stats)
            assert len(_new_idx) == 0
        elif cuda_selector_enabled:
            eps_final = 0.0
            if selection_mode_s == "heat_bath":
                eps_final = float(hb_eps_final) if str(hb_eps_schedule).lower() == "adaptive" else float(hb_epsilon)
            selector_plan_final = _plan_cuda_selector_buckets_fast(
                drt=drt,
                sel_size=int(sel_idx.size),
                prev_ncand_hint=int(prev_cuda_ncand_hint),
                max_add=0,
            )
            _new_idx, e_pt2, cuda_final_stats = _cuda_select_external(
                sel_idx_i64=np.asarray(sel_idx, dtype=np.int64),
                c_sel_arr=c_sel,
                e_var_arr=e_var,
                max_add_i=0,
                eps_val=float(eps_final),
                bucket_bounds=tuple(selector_plan_final["bucket_bounds"]),
            )
            profile["cuda_selector_final_stats"] = dict(cuda_final_stats)
            assert len(_new_idx) == 0
        elif selection_mode_s == "frontier_hash":
            _new_idx, e_pt2, _stats = frontier_selector.build_and_score(
                sel_idx=np.asarray(sel_idx, dtype=np.int64),
                c_sel=c_sel,
                e_var=e_var,
                max_add=0,
                select_threshold=select_threshold,
                row_cache=persistent_row_cache,
            )
            assert len(_new_idx) == 0
        else:
            if str(pt2_mode) in ("streaming", "semistochastic"):
                _sel_set = {int(s) for s in np.asarray(sel_idx, dtype=np.int64).ravel().tolist()}
                _pt2_func = streaming_pt2_deterministic if str(pt2_mode) == "streaming" else semistochastic_pt2
                _pt2_kw2: dict[str, Any] = dict(
                    drt=drt, h1e=h1e, eri=eri,
                    sel=np.asarray(sel_idx, dtype=np.int64).ravel().tolist(),
                    selected_set=_sel_set,
                    c_sel=c_sel,
                    e_var=e_var,
                    hdiag_lookup=hdiag_lookup,
                    denom_floor=float(denom_floor),
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                    bucket_size=int(pt2_bucket_size),
                )
                if str(pt2_mode) == "semistochastic":
                    _pt2_kw2.update(n_det_sources=pt2_n_det_sources, n_stoch_samples=pt2_n_stoch_samples,
                                    n_stoch_batches=pt2_n_stoch_batches, seed=pt2_seed)
                _pt2_res2 = _pt2_func(**_pt2_kw2)
                e_pt2 = _pt2_res2.e_pt2
            else:
                _eps_final = float(hb_eps_final) if str(hb_eps_schedule).lower() == "adaptive" else float(hb_epsilon)
                _new_idx, e_pt2 = heat_bath_select_and_pt2_sparse(
                    drt,
                    h1e,
                    eri,
                    sel_idx=np.asarray(sel_idx, dtype=np.int64),
                    c_sel=c_sel,
                    e_var=e_var,
                    max_add=0,
                    epsilon=float(_eps_final),
                    denom_floor=float(denom_floor),
                    hdiag_lookup=hdiag_lookup,
                    max_out=int(row_max_out),
                    screening=None,
                    state_cache=state_cache,
                    row_cache=persistent_row_cache,
                )
    roots = _sparse_roots_from_selected(sel_idx, c_sel)
    sel_key_u64 = None
    label_kind = "csf_idx"
    if state_rep_s == "key64":
        from asuka.qmc.cuda_backend import csf_idx_to_key64_host  # noqa: PLC0415

        sel_key_u64 = np.asarray(csf_idx_to_key64_host(drt, sel_idx, state_cache=None), dtype=np.uint64, order="C")
        label_kind = "key64"
    elif state_rep_s == "i64":
        sel_idx_i64 = np.asarray(sel_idx, dtype=np.int64).ravel()
        if sel_idx_i64.size:
            if int(np.min(sel_idx_i64)) < 0:
                raise ValueError("selected indices must be non-negative for idx64 label mode")
            if int(np.max(sel_idx_i64)) >= int(ncsf):
                raise ValueError("selected indices must be < drt.ncsf for idx64 label mode")
        sel_key_u64 = np.asarray(sel_idx_i64, dtype=np.uint64, order="C")
        label_kind = "idx64"
    return CIPSITrialSpaceResult(
        e_var=np.asarray(e_var + float(ecore), dtype=np.float64),
        e_pt2=np.asarray(e_pt2, dtype=np.float64),
        e_tot=np.asarray(e_var + e_pt2 + float(ecore), dtype=np.float64),
        sel_idx=sel_idx,
        sel_key_u64=sel_key_u64,
        label_kind=label_kind,
        ci_sel=np.asarray(c_sel, dtype=np.float64),
        roots=roots,
        history=history,
        profile=dict(profile),
        epq_mode=str(profile.get("epq_mode", epq_mode_s)),
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
    backend: str = "auto",
    epq_mode: str = "no_epq_support_aware",
    ci0: Any = None,
    twos: int | None = None,
    orbsym: np.ndarray | None = None,
    wfnsym: int | None = None,
    **kwargs,
) -> CIPSITrialSpaceResult:
    backend_s = _normalize_cipsi_backend(backend)

    if isinstance(scf_or_casci, CASCIResult):
        cas = scf_or_casci
        norb = int(cas.ncas)
        nelecas_i = cas.nelecas if nelecas is None else nelecas
        if twos is None:
            if isinstance(nelecas_i, tuple):
                twos = int(nelecas_i[0]) - int(nelecas_i[1])
            else:
                twos = int(getattr(cas.mol, "spin", 0))
        drt = build_drt(norb=norb, nelec=int(sum(nelecas_i)) if isinstance(nelecas_i, tuple) else int(nelecas_i), twos_target=int(twos), orbsym=orbsym, wfnsym=wfnsym)
        return run_cipsi_trials(
            drt,
            cas.h1eff,
            cas.eri,
            ecore=float(cas.ecore),
            nroots=int(nroots),
            ci0=ci0,
            epq_mode=epq_mode,
            backend=backend_s,
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
        orbsym=orbsym,
        wfnsym=wfnsym,
    )
    return run_cipsi_trials(
        drt,
        h1eff,
        eri,
        ecore=float(ecore),
        nroots=int(nroots),
        ci0=ci0,
        epq_mode=epq_mode,
        backend=backend_s,
        **kwargs,
    )


__all__ = ["CIPSITrialSpaceResult", "build_cipsi_trials_from_scf", "run_cipsi_trials"]
