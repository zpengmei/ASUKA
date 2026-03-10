from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.cuguga.state_cache import get_state_cache
from asuka.mcscf.casci import CASCIResult, _build_casci_df_integrals
from asuka.qmc.labels import normalize_state_rep
from asuka.qmc.sparse import SparseVector
from asuka.sci.frontier_hash import SparseFrontierSelector
from asuka.sci.hb_selection import heat_bath_select_and_pt2_sparse
from asuka.sci.selected_ci import (
    DiagonalGuessLookup,
    _build_variational_hamiltonian_sparse,
    _initial_selection_sparse,
    _normalize_ci0_sparse,
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
    allowed = {"auto", "cpu_sparse", "cuda_key64"}
    if mode not in allowed:
        raise ValueError("backend must be 'auto', 'cpu_sparse', or 'cuda_key64'")
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
    if backend_requested == "cuda_key64" and state_rep_s == "i32":
        raise ValueError("backend='cuda_key64' is incompatible with state_rep='i32'")

    if backend_requested == "auto":
        use_key64 = state_rep_s == "key64" or (state_rep_s == "auto" and int(drt.ncsf) > _INT32_MAX and int(drt.norb) <= 32)
        backend_effective = "cuda_key64" if use_key64 else "cpu_sparse"
    else:
        backend_effective = backend_requested
        use_key64 = backend_effective == "cuda_key64"
    if use_key64:
        state_rep_s = "key64"

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

    need_sparse_state = int(ncsf) > _INT32_MAX or state_rep_s == "key64"
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
        "workspace_kwargs_ignored": bool(workspace_kwargs),
        "frontier_hash_cap_ignored": frontier_hash_cap is not None,
        "frontier_hash_tile_ignored": int(frontier_hash_tile),
        "frontier_hash_rs_block_ignored": int(frontier_hash_rs_block),
        "frontier_hash_g_rows_ignored": int(frontier_hash_g_rows),
        "frontier_hash_offdiag_kernel_mode_ignored": frontier_hash_offdiag_kernel_mode,
        "frontier_hash_csr_capacity_mult_ignored": float(frontier_hash_csr_capacity_mult),
        "frontier_hash_max_retries_ignored": int(frontier_hash_max_retries),
    }
    if str(requested_epq_mode) != str(epq_mode_s):
        profile["epq_mode_requested"] = str(requested_epq_mode)
        profile["epq_mode_effective"] = str(epq_mode_s)
        profile["epq_mode_fallback_reason"] = "streamed_epq_temporarily_disabled"
    else:
        profile["epq_mode_effective"] = str(epq_mode_s)
    history: list[dict[str, Any]] = []
    prev_c_sel: np.ndarray | None = None
    frontier_selector = SparseFrontierSelector(
        drt,
        h1e,
        eri,
        hdiag_lookup=hdiag_lookup,
        denom_floor=float(denom_floor),
        max_out=200_000,
        screening=None,
        state_cache=state_cache,
        select_screen_contrib=0.0,
    )
    frontier_selector.reset_selected_mask(np.asarray(sel, dtype=np.int64))

    for it in range(1, int(max_iter) + 1):
        sel_idx = np.asarray(sel, dtype=np.int64)
        h_sub = _build_variational_hamiltonian_sparse(
            drt,
            h1e,
            eri,
            sel=sel,
            loc_map=loc_map,
            max_out=200_000,
            screening=None,
            state_cache=state_cache,
        )
        ci0_sub = _build_ci0_subspace_sparse(
            sel_idx=sel_idx,
            loc_map=loc_map,
            nroots=int(nroots),
            ci0_sparse=ci0_sparse,
            prev_c_sel=prev_c_sel,
        )
        v0 = None if not ci0_sub else np.asarray(ci0_sub[0], dtype=np.float64)
        e_var, c_sel = _solve_subspace(
            h_sub,
            nroots=int(nroots),
            dense_limit=max(64, int(davidson_max_space) * 8),
            eigsh_tol=float(davidson_tol),
            v0=v0,
        )
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
        if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
            e_var = np.asarray(e_var[perm], dtype=np.float64)
            c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)

        max_add = min(int(grow_by), int(max_ncsf) - int(sel_idx.size))
        if selection_mode_s == "frontier_hash":
            new_idx, e_pt2, _stats = frontier_selector.build_and_score(
                sel_idx=sel_idx,
                c_sel=c_sel,
                e_var=e_var,
                max_add=int(max_add),
                select_threshold=select_threshold,
            )
        else:
            if str(hb_eps_schedule).lower() == "adaptive":
                frac = 0.0 if int(max_ncsf) <= int(init_ncsf) else (int(sel_idx.size) - int(init_ncsf)) / max(
                    1, int(max_ncsf) - int(init_ncsf)
                )
                frac = float(np.clip(frac, 0.0, 1.0))
                _eps = float(hb_eps_init) * (float(hb_eps_final) / float(hb_eps_init)) ** frac
            else:
                _eps = float(hb_epsilon)
            new_idx, e_pt2 = heat_bath_select_and_pt2_sparse(
                drt,
                h1e,
                eri,
                sel_idx=sel_idx,
                c_sel=c_sel,
                e_var=e_var,
                max_add=int(max_add),
                epsilon=float(_eps),
                denom_floor=float(denom_floor),
                hdiag_lookup=hdiag_lookup,
                max_out=200_000,
                screening=None,
                state_cache=state_cache,
            )
        for ii in new_idx:
            jj = int(ii)
            if jj in loc_map:
                continue
            loc_map[jj] = int(len(sel))
            sel.append(jj)
            if len(sel) >= int(max_ncsf):
                break
        if len(new_idx) > 0:
            frontier_selector.mark_selected(new_idx)

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
        }
        history.append(rec)
        prev_c_sel = np.asarray(c_sel, dtype=np.float64)
        if verbose:
            print(f"[GPU-CIPSI] iter={it} nsel={c_sel.shape[0]} nadd={len(new_idx)} e0={float(e_var[0] + ecore):.12f}")
        if len(new_idx) == 0 or len(sel) >= int(max_ncsf):
            break

    sel_idx = np.asarray(sel, dtype=np.int64)
    h_sub = _build_variational_hamiltonian_sparse(
        drt,
        h1e,
        eri,
        sel=sel,
        loc_map=loc_map,
        max_out=200_000,
        screening=None,
        state_cache=state_cache,
    )
    ci0_sub = _build_ci0_subspace_sparse(
        sel_idx=sel_idx,
        loc_map=loc_map,
        nroots=int(nroots),
        ci0_sparse=ci0_sparse,
        prev_c_sel=prev_c_sel,
    )
    v0 = None if not ci0_sub else np.asarray(ci0_sub[0], dtype=np.float64)
    e_var, c_sel = _solve_subspace(
        h_sub,
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
    perm = _match_roots_by_overlap(prev_c_sel, c_sel, e_var)
    if np.any(perm != np.arange(int(nroots), dtype=np.int32)):
        e_var = np.asarray(e_var[perm], dtype=np.float64)
        c_sel = np.asarray(c_sel[:, perm], dtype=np.float64)
    if selection_mode_s == "frontier_hash":
        _new_idx, e_pt2, _stats = frontier_selector.build_and_score(
            sel_idx=np.asarray(sel_idx, dtype=np.int64),
            c_sel=c_sel,
            e_var=e_var,
            max_add=0,
            select_threshold=select_threshold,
        )
        assert len(_new_idx) == 0
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
            max_out=200_000,
            screening=None,
            state_cache=state_cache,
        )
    roots = _sparse_roots_from_selected(sel_idx, c_sel)
    sel_key_u64 = None
    label_kind = "csf_idx"
    if state_rep_s == "key64":
        from asuka.qmc.cuda_backend import csf_idx_to_key64_host  # noqa: PLC0415

        sel_key_u64 = np.asarray(csf_idx_to_key64_host(drt, sel_idx, state_cache=None), dtype=np.uint64, order="C")
        label_kind = "key64"
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
        drt = build_drt(norb=norb, nelec=int(sum(nelecas_i)) if isinstance(nelecas_i, tuple) else int(nelecas_i), twos_target=int(twos))
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
