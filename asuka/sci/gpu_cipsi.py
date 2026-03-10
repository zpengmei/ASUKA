from __future__ import annotations

import json
import os
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

    if isinstance(eri, DeviceDFMOIntegrals):
        if eri.l_full is None:
            return None
        l_full = _asnumpy_f64(eri.l_full)
        j_ps = _asnumpy_f64(eri.j_ps)
        if l_full.ndim != 2 or int(l_full.shape[0]) != norb * norb:
            return None
        h_eff = np.asarray(h1e_f64 - 0.5 * j_ps, dtype=np.float64, order="C")
        hb_index = build_hb_index_from_df(h_eff, l_full, norb)
        eri_2d = np.asarray(l_full @ l_full.T, dtype=np.float64, order="C")
    elif isinstance(eri, DFMOIntegrals):
        l_full = np.asarray(eri.l_full, dtype=np.float64, order="C")
        if l_full.ndim != 2 or int(l_full.shape[0]) != norb * norb:
            return None
        h_eff = np.asarray(h1e_f64 - 0.5 * np.asarray(eri.j_ps, dtype=np.float64), dtype=np.float64, order="C")
        hb_index = build_hb_index_from_df(h_eff, l_full, norb)
        eri_2d = np.asarray(l_full @ l_full.T, dtype=np.float64, order="C")
    else:
        eri_4d = np.asarray(_restore_eri_4d(eri, norb), dtype=np.float64, order="C")
        h_eff = np.asarray(h1e_f64 - 0.5 * np.einsum("pqqs->ps", eri_4d), dtype=np.float64, order="C")
        hb_index = build_hb_index(h_eff, eri_4d, norb)
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
        "workspace_kwargs_ignored": bool(workspace_kwargs),
        "frontier_hash_cap": None if frontier_hash_cap is None else int(frontier_hash_cap),
        "frontier_hash_tile": int(frontier_hash_tile),
        "frontier_hash_rs_block": int(frontier_hash_rs_block),
        "frontier_hash_g_rows": int(frontier_hash_g_rows),
        "frontier_hash_offdiag_kernel_mode": frontier_hash_offdiag_kernel_mode,
        "frontier_hash_csr_capacity_mult": float(frontier_hash_csr_capacity_mult),
        "frontier_hash_max_retries": int(frontier_hash_max_retries),
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
    _cas36_diag_guess = None
    _cas36_score_pt2 = None

    if backend_effective in ("cuda_key64", "cuda_idx64"):
        try:
            import cupy as _cp  # type: ignore[import-not-found]
            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                cas36_cipsi_score_pt2_compact_u64_inplace_device as _cas36_score_pt2,
                cas36_diag_guess_candidates_u64_dense_inplace_device as _cas36_diag_guess,
                cas36_hb_screen_and_apply_u64_inplace_device as _cas36_hb_apply,
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
                            cuda_selector_enabled = True
        except Exception as _cuda_e:
            cuda_selector_reason = f"cuda_init_failed:{type(_cuda_e).__name__}"

    if cuda_selector_enabled:
        profile["driver"] = "cuda_cas36_hb_compact_u64"
        profile["cuda_selector_enabled"] = True
        profile["cuda_selector_hash_cap_init"] = int(_hash_cap)
        profile["cuda_selector_threads"] = int(_cuda_threads)
    else:
        profile["cuda_selector_enabled"] = False
        if backend_effective in ("cuda_key64", "cuda_idx64"):
            profile["cuda_selector_disabled_reason"] = str(cuda_selector_reason or "unknown")

    def _cuda_select_external(
        *,
        sel_idx_i64: np.ndarray,
        c_sel_arr: np.ndarray,
        e_var_arr: np.ndarray,
        max_add_i: int,
        eps_val: float,
    ) -> tuple[list[int], np.ndarray, dict[str, Any]]:
        nonlocal _hash_cap, _hash_keys_d, _hash_vals_d, _overflow_d
        assert _cp is not None
        assert _drt_dev is not None
        assert _hb_dev is not None
        assert _h1_diag_d is not None and _eri_ppqq_d is not None and _eri_pqqp_d is not None
        assert _cas36_hb_apply is not None and _cas36_diag_guess is not None and _cas36_score_pt2 is not None

        sel_idx_i64 = np.asarray(sel_idx_i64, dtype=np.int64).ravel()
        if sel_idx_i64.size == 0:
            return [], np.zeros((int(nroots),), dtype=np.float64), {"ncand": 0, "overflow_retries": 0, "hash_cap": int(_hash_cap)}
        if bool(_hb_effectively_zero):
            return [], np.zeros((int(nroots),), dtype=np.float64), {"ncand": 0, "overflow_retries": 0, "hash_cap": int(_hash_cap)}
        if int(np.min(sel_idx_i64)) < 0:
            raise ValueError("selected indices must be non-negative for CUDA selector")

        sel_idx_u64_d = _cp.asarray(sel_idx_i64.astype(np.uint64, copy=False), dtype=_cp.uint64).ravel()
        sel_idx_u64_d = _cp.ascontiguousarray(sel_idx_u64_d)
        sel_idx_sorted_d = _cp.sort(sel_idx_u64_d)
        c_sel_d = _cp.ascontiguousarray(_cp.asarray(c_sel_arr, dtype=_cp.float64))
        e_var_d = _cp.ascontiguousarray(_cp.asarray(e_var_arr, dtype=_cp.float64).ravel())

        retries = 0
        empty_u64 = np.uint64(0xFFFFFFFFFFFFFFFF)
        stream_u = int(_cp.cuda.get_current_stream().ptr)
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
                    selected_idx_sorted_u64=sel_idx_sorted_d,
                    overflow=_overflow_d,
                    threads=int(_cuda_threads),
                    stream=stream_u,
                    sync=True,
                )
            overflow_h = int(_cp.asnumpy(_overflow_d)[0])
            if overflow_h == 0:
                break
            retries += 1
            if retries > int(frontier_hash_max_retries):
                raise RuntimeError(
                    f"CUDA selector hash overflow after {retries} retries (cap={int(_hash_cap)}); "
                    "increase frontier_hash_cap or reduce growth"
                )
            _hash_cap = int(_hash_cap) * 2
            _hash_keys_d = _cp.empty((int(_hash_cap),), dtype=_cp.uint64)
            _hash_vals_d = _cp.empty((int(nroots), int(_hash_cap)), dtype=_cp.float64)
            _overflow_d = _cp.zeros((1,), dtype=_cp.int32)

        mask = _hash_keys_d != empty_u64
        ncand = int(_cp.count_nonzero(mask).get())
        if ncand <= 0:
            return [], np.zeros((int(nroots),), dtype=np.float64), {
                "ncand": 0,
                "overflow_retries": int(retries),
                "hash_cap": int(_hash_cap),
            }

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
            sync=True,
        )
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
            sync=True,
        )
        e_pt2_h = np.asarray(_cp.asnumpy(pt2_d), dtype=np.float64)

        max_add_i = int(max_add_i)
        if max_add_i <= 0:
            return [], e_pt2_h, {
                "ncand": int(ncand),
                "overflow_retries": int(retries),
                "hash_cap": int(_hash_cap),
            }

        valid = score_bits_d > 0
        nvalid = int(_cp.count_nonzero(valid).get())
        if nvalid <= 0:
            return [], e_pt2_h, {
                "ncand": int(ncand),
                "nvalid": 0,
                "overflow_retries": int(retries),
                "hash_cap": int(_hash_cap),
            }

        keep = min(int(max_add_i), int(nvalid))
        valid_pos = _cp.nonzero(valid)[0]
        if int(valid_pos.size) > int(keep):
            part = _cp.argpartition(score_bits_d[valid_pos], -int(keep))[-int(keep):]
            chosen = valid_pos[part]
        else:
            chosen = valid_pos

        chosen_score = np.asarray(_cp.asnumpy(score_bits_d[chosen]), dtype=np.uint64)
        chosen_idx_u64 = np.asarray(_cp.asnumpy(cand_idx_u64[chosen]), dtype=np.uint64)
        order = np.lexsort((chosen_idx_u64, -chosen_score.astype(np.int64, copy=False)))
        new_idx_h = [int(x) for x in chosen_idx_u64[order].tolist()]
        return new_idx_h, e_pt2_h, {
            "ncand": int(ncand),
            "nvalid": int(nvalid),
            "overflow_retries": int(retries),
            "hash_cap": int(_hash_cap),
        }

    for it in range(1, int(max_iter) + 1):
        sel_idx = np.asarray(sel, dtype=np.int64)
        row_cache_iter: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if use_row_cache else None
        h_sub = _build_variational_hamiltonian_sparse(
            drt,
            h1e,
            eri,
            sel=sel,
            loc_map=loc_map,
            max_out=200_000,
            screening=None,
            state_cache=state_cache,
            row_cache=row_cache_iter,
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
        cuda_iter_stats: dict[str, Any] = {}
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

        if cuda_selector_enabled:
            try:
                eps_run = 0.0 if selection_mode_s == "frontier_hash" else float(hb_eps_iter)
                new_idx, e_pt2, cuda_iter_stats = _cuda_select_external(
                    sel_idx_i64=sel_idx,
                    c_sel_arr=c_sel,
                    e_var_arr=e_var,
                    max_add_i=int(max_add),
                    eps_val=float(eps_run),
                )
            except Exception as cuda_step_e:
                cuda_selector_enabled = False
                profile["cuda_selector_step_fallback_iter"] = int(it)
                profile["cuda_selector_step_fallback_reason"] = f"{type(cuda_step_e).__name__}: {cuda_step_e}"
                profile["driver"] = "sparse_row_oracle"

        if not cuda_selector_enabled:
            if selection_mode_s == "frontier_hash":
                new_idx, e_pt2, _stats = frontier_selector.build_and_score(
                    sel_idx=sel_idx,
                    c_sel=c_sel,
                    e_var=e_var,
                    max_add=int(max_add),
                    select_threshold=select_threshold,
                    row_cache=row_cache_iter,
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
                    max_out=200_000,
                    screening=None,
                    state_cache=state_cache,
                    row_cache=row_cache_iter,
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
        if cuda_iter_stats:
            rec["cuda_selector"] = dict(cuda_iter_stats)
        history.append(rec)
        prev_c_sel = np.asarray(c_sel, dtype=np.float64)
        if verbose:
            print(f"[GPU-CIPSI] iter={it} nsel={c_sel.shape[0]} nadd={len(new_idx)} e0={float(e_var[0] + ecore):.12f}")
        if len(new_idx) == 0 or len(sel) >= int(max_ncsf):
            break

    sel_idx = np.asarray(sel, dtype=np.int64)
    row_cache_final: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if use_row_cache else None
    h_sub = _build_variational_hamiltonian_sparse(
        drt,
        h1e,
        eri,
        sel=sel,
        loc_map=loc_map,
        max_out=200_000,
        screening=None,
        state_cache=state_cache,
        row_cache=row_cache_final,
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
    if cuda_selector_enabled:
        eps_final = 0.0
        if selection_mode_s == "heat_bath":
            eps_final = float(hb_eps_final) if str(hb_eps_schedule).lower() == "adaptive" else float(hb_epsilon)
        _new_idx, e_pt2, cuda_final_stats = _cuda_select_external(
            sel_idx_i64=np.asarray(sel_idx, dtype=np.int64),
            c_sel_arr=c_sel,
            e_var_arr=e_var,
            max_add_i=0,
            eps_val=float(eps_final),
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
            row_cache=row_cache_final,
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
            row_cache=row_cache_final,
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
