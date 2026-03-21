from __future__ import annotations

"""Sparse helper utilities shared by scalable SCI/CIPSI paths."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as _e:  # pragma: no cover
    sp = None  # type: ignore[assignment]
    spla = None  # type: ignore[assignment]

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _STEP_TO_OCC, _restore_eri_4d, diagonal_element_det_guess
from asuka.cuguga.oracle.sparse import connected_row_sparse, connected_row_sparse_df
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache
from asuka.integrals.df_diag import diagonal_element_det_guess_df
from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.sci._sparse_vector import SparseVector

ConnectedRowCache = dict[int, tuple[np.ndarray, np.ndarray]]
SELECTOR_BUCKET_EDGE_THRESHOLD = 1_000_000
SELECTOR_BUCKET_MAX_BUCKETS = 64
SELECTOR_BUCKET_SPLIT_MIN_WIDTH = 1024
SELECTOR_BUCKET_SPLIT_CAND_MULT = 16
SOLVER_REORDER_MIN_NSEL = 256


def _as_numpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None  # type: ignore[assignment]

    if cp is not None and isinstance(a, cp.ndarray):
        return np.asarray(cp.asnumpy(a), dtype=np.float64, order="C")
    return np.asarray(a, dtype=np.float64, order="C")


def _topk_desc_indices(values: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(values)
    kk = int(k)
    if kk <= 0 or arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if kk >= int(arr.size):
        return np.asarray(np.argsort(arr)[::-1], dtype=np.int64)
    keep = np.argpartition(arr, -kk)[-kk:]
    return np.asarray(keep[np.argsort(arr[keep])[::-1]], dtype=np.int64)


def _bottomk_asc_indices(values: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(values)
    kk = int(k)
    if kk <= 0 or arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if kk >= int(arr.size):
        return np.asarray(np.argsort(arr), dtype=np.int64)
    keep = np.argpartition(arr, kk - 1)[:kk]
    return np.asarray(keep[np.argsort(arr[keep])], dtype=np.int64)


class DiagonalGuessLookup:
    """Lazy diagonal-element provider for scalable selected-CI/CIPSI paths."""

    def __init__(self, drt: DRT, h1e: np.ndarray, eri: Any, *, hdiag: np.ndarray | None = None) -> None:
        self.drt = drt
        self.h1e = np.asarray(h1e, dtype=np.float64)
        self.ncsf = int(drt.ncsf)
        self._dense = None if hdiag is None else np.asarray(hdiag, dtype=np.float64).ravel()
        if self._dense is not None and int(self._dense.size) != self.ncsf:
            raise ValueError("hdiag has wrong length")
        self._cache: dict[int, float] = {}
        self._eri_dense = None
        self._eri_ppqq = None
        self._eri_pqqp = None
        self._df_eri = None

        if isinstance(eri, DeviceDFMOIntegrals):
            l_full = _as_numpy_f64(eri.l_full) if eri.l_full is not None else None
            pair_norm = _as_numpy_f64(eri.pair_norm) if eri.pair_norm is not None else None
            eri_mat = _as_numpy_f64(eri.eri_mat) if eri.eri_mat is not None else None
            if l_full is None and eri_mat is None:
                raise ValueError("DeviceDFMOIntegrals must provide l_full or eri_mat for diagonal lookup")
            if l_full is not None:
                if pair_norm is None:
                    pair_norm = np.linalg.norm(l_full, axis=1)
                self._df_eri = DFMOIntegrals(
                    norb=int(eri.norb),
                    l_full=np.asarray(l_full, dtype=np.float64, order="C"),
                    j_ps=np.asarray(_as_numpy_f64(eri.j_ps), dtype=np.float64, order="C"),
                    pair_norm=np.asarray(pair_norm, dtype=np.float64, order="C"),
                    _eri_mat=None if eri_mat is None else np.asarray(eri_mat, dtype=np.float64, order="C"),
                )
            else:
                self._eri_dense = np.asarray(
                    eri_mat.reshape(int(drt.norb), int(drt.norb), int(drt.norb), int(drt.norb)),
                    dtype=np.float64,
                    order="C",
                )
        elif isinstance(eri, DFMOIntegrals):
            self._df_eri = eri
        else:
            self._eri_dense = _restore_eri_4d(eri, int(drt.norb)).astype(np.float64, copy=False)
            self._eri_ppqq = np.einsum("iijj->ij", self._eri_dense)
            self._eri_pqqp = np.einsum("ijji->ij", self._eri_dense)

    @property
    def has_dense(self) -> bool:
        return self._dense is not None

    def get(self, idx: int) -> float:
        ii = int(idx)
        if ii < 0 or ii >= self.ncsf:
            raise IndexError(f"CSF index out of range: {ii}")
        if self._dense is not None:
            return float(self._dense[ii])
        cached = self._cache.get(ii)
        if cached is not None:
            return float(cached)
        if self._df_eri is not None:
            val = diagonal_element_det_guess_df(self.drt, self.h1e, self._df_eri, ii)
        else:
            val = diagonal_element_det_guess(
                self.drt,
                self.h1e,
                self._eri_dense,
                ii,
                eri_ppqq=self._eri_ppqq,
                eri_pqqp=self._eri_pqqp,
            )
        self._cache[ii] = float(val)
        return float(val)

    def get_many(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        idx_i64 = np.asarray(idx, dtype=np.int64).ravel()
        if idx_i64.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if self._dense is not None:
            return np.asarray(self._dense[idx_i64], dtype=np.float64, order="C")
        return np.asarray([self.get(int(ii)) for ii in idx_i64.tolist()], dtype=np.float64, order="C")


@dataclass(frozen=True)
class SelectorBucketPlan:
    bucketed: bool
    nbuckets: int
    bucket_bounds: tuple[tuple[int, int], ...]
    active_frontier_edges: int
    bucket_kind: str = "csf_idx_range"


def _maybe_split_bucket_range(
    label_lo: int,
    label_hi: int,
    *,
    cand_count: int,
    max_add: int,
) -> tuple[tuple[int, int], ...]:
    width = int(label_hi) - int(label_lo)
    if width < int(SELECTOR_BUCKET_SPLIT_MIN_WIDTH):
        return ((int(label_lo), int(label_hi)),)
    split_threshold = max(int(SELECTOR_BUCKET_SPLIT_MIN_WIDTH), int(max_add) * int(SELECTOR_BUCKET_SPLIT_CAND_MULT))
    if int(cand_count) <= int(split_threshold):
        return ((int(label_lo), int(label_hi)),)
    mid = int(label_lo) + width // 2
    if mid <= int(label_lo) or mid >= int(label_hi):
        return ((int(label_lo), int(label_hi)),)
    return ((int(label_lo), int(mid)), (int(mid), int(label_hi)))


def _solver_reorder_perm(sel_idx: np.ndarray, h: "sp.csr_matrix") -> np.ndarray:
    nsel = int(h.shape[0])
    if sp is None or nsel < int(SOLVER_REORDER_MIN_NSEL):
        return np.arange(nsel, dtype=np.int32)
    try:
        perm = sp.csgraph.reverse_cuthill_mckee(h, symmetric_mode=True)
        perm = np.asarray(perm, dtype=np.int32)
        if perm.shape != (nsel,):
            raise ValueError("invalid permutation shape")
        if not np.array_equal(np.sort(perm.astype(np.int64, copy=False)), np.arange(nsel, dtype=np.int64)):
            raise ValueError("invalid permutation contents")
        return perm
    except Exception:
        row_nnz = np.diff(h.indptr).astype(np.int64, copy=False)
        return np.asarray(np.lexsort((np.asarray(sel_idx, dtype=np.int64), -row_nnz)), dtype=np.int32)


def _connected_row_cached(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    j: int,
    *,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    row_cache: ConnectedRowCache | None,
) -> tuple[np.ndarray, np.ndarray]:
    jj = int(j)
    cached = None if row_cache is None else row_cache.get(jj)
    if cached is not None:
        return cached
    i_idx, hij = _connected_row(
        drt,
        h1e,
        eri,
        jj,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
    )
    if row_cache is not None:
        row_cache[jj] = (i_idx, hij)
    return i_idx, hij


def _plan_selector_buckets(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    c_sel: np.ndarray,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    row_cache: ConnectedRowCache | None,
) -> SelectorBucketPlan:
    active_frontier_edges = 0
    for col, j in enumerate(sel):
        cj = np.asarray(c_sel[col, :], dtype=np.float64)
        if float(np.max(np.abs(cj))) == 0.0:
            continue
        i_idx, _hij = _connected_row_cached(
            drt,
            h1e,
            eri,
            int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )
        active_frontier_edges += int(i_idx.size)

    if active_frontier_edges < int(SELECTOR_BUCKET_EDGE_THRESHOLD):
        return SelectorBucketPlan(
            bucketed=False,
            nbuckets=1,
            bucket_bounds=((0, int(drt.ncsf)),),
            active_frontier_edges=int(active_frontier_edges),
        )

    nbuckets = max(1, int(np.ceil(float(active_frontier_edges) / float(SELECTOR_BUCKET_EDGE_THRESHOLD))))
    nbuckets = min(int(SELECTOR_BUCKET_MAX_BUCKETS), nbuckets)
    label_hi = int(drt.ncsf)
    bucket_span = max(1, int(np.ceil(float(label_hi) / float(nbuckets))))
    bounds: list[tuple[int, int]] = []
    lo = 0
    while lo < label_hi:
        hi = min(label_hi, lo + bucket_span)
        bounds.append((int(lo), int(hi)))
        lo = hi
    if not bounds:
        bounds = [(0, label_hi)]
    return SelectorBucketPlan(
        bucketed=len(bounds) > 1,
        nbuckets=int(len(bounds)),
        bucket_bounds=tuple(bounds),
        active_frontier_edges=int(active_frontier_edges),
    )


class IncrementalVariationalHamiltonianBuilder:
    """Incrementally maintain the selected-space sparse Hamiltonian.

    This avoids re-querying connected rows for the already-selected columns each
    time the variational space grows. The CSR matrix is rebuilt from cached COO
    triplets only when new columns/rows are appended.
    """

    def __init__(
        self,
        drt: DRT,
        h1e: np.ndarray,
        eri: Any,
        *,
        sel: Sequence[int],
        loc_map: dict[int, int],
        max_out: int,
        screening: RowScreening | None,
        state_cache: DRTStateCache | None,
        row_cache: ConnectedRowCache | None = None,
    ) -> None:
        if sp is None:  # pragma: no cover
            raise RuntimeError("scipy is required for scalable SCI support")
        self.drt = drt
        self.h1e = np.asarray(h1e, dtype=np.float64)
        self.eri = eri
        self.max_out = int(max_out)
        self.screening = screening
        self.state_cache = state_cache
        self.row_cache = row_cache
        _ = loc_map
        self.sel: list[int] = []
        self.loc_map: dict[int, int] = {}
        self._csr_cache = sp.csr_matrix((0, 0), dtype=np.float64)
        self.extend(sel)

    def _get_connected_row(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return _connected_row_cached(
            self.drt,
            self.h1e,
            self.eri,
            int(idx),
            max_out=int(self.max_out),
            screening=self.screening,
            state_cache=self.state_cache,
            row_cache=self.row_cache,
        )

    def extend(self, new_idx: Sequence[int]) -> None:
        appended: list[int] = []
        old_nsel = int(len(self.sel))
        for ii in new_idx:
            jj = int(ii)
            if jj in self.loc_map:
                continue
            self.loc_map[jj] = int(len(self.sel))
            self.sel.append(jj)
            appended.append(jj)
        if old_nsel == 0 and not appended:
            for jj in [int(ii) for ii in new_idx]:
                self.loc_map[jj] = int(len(self.sel))
                self.sel.append(jj)
                appended.append(jj)
        if not appended:
            return
        total_nsel = int(len(self.sel))
        new_nsel = int(len(appended))
        appended_set = {int(ii) for ii in appended}
        old_rows: list[int] = []
        old_cols: list[int] = []
        old_data: list[float] = []
        upper_rows: list[int] = []
        upper_cols: list[int] = []
        upper_data: list[float] = []
        lower_rows: list[int] = []
        lower_cols: list[int] = []
        lower_data: list[float] = []

        for col in range(old_nsel):
            jj = int(self.sel[col])
            i_idx, hij = self._get_connected_row(jj)
            for ii, vv in zip(i_idx.tolist(), hij.tolist(), strict=False):
                if int(ii) not in appended_set:
                    continue
                old_rows.append(int(self.loc_map[int(ii)] - old_nsel))
                old_cols.append(int(col))
                old_data.append(float(vv))

        for rel_col, col in enumerate(range(old_nsel, total_nsel)):
            jj = int(self.sel[col])
            i_idx, hij = self._get_connected_row(jj)
            for ii, vv in zip(i_idx.tolist(), hij.tolist(), strict=False):
                row = self.loc_map.get(int(ii))
                if row is None:
                    continue
                row_i = int(row)
                if row_i < old_nsel:
                    upper_rows.append(row_i)
                    upper_cols.append(int(rel_col))
                    upper_data.append(float(vv))
                else:
                    lower_rows.append(row_i - old_nsel)
                    lower_cols.append(int(rel_col))
                    lower_data.append(float(vv))

        old_to_new = sp.coo_matrix(
            (
                np.asarray(old_data, dtype=np.float64),
                (
                    np.asarray(old_rows, dtype=np.int32),
                    np.asarray(old_cols, dtype=np.int32),
                ),
            ),
            shape=(new_nsel, old_nsel),
        ).tocsr()
        upper_right_direct = sp.coo_matrix(
            (
                np.asarray(upper_data, dtype=np.float64),
                (
                    np.asarray(upper_rows, dtype=np.int32),
                    np.asarray(upper_cols, dtype=np.int32),
                ),
            ),
            shape=(old_nsel, new_nsel),
        ).tocsr()
        upper_right = 0.5 * (upper_right_direct + old_to_new.T)
        lower_right = sp.coo_matrix(
            (
                np.asarray(lower_data, dtype=np.float64),
                (
                    np.asarray(lower_rows, dtype=np.int32),
                    np.asarray(lower_cols, dtype=np.int32),
                ),
            ),
            shape=(new_nsel, new_nsel),
        ).tocsr()
        lower_right = (lower_right + lower_right.T) * 0.5
        lower_right.eliminate_zeros()

        if old_nsel == 0:
            self._csr_cache = lower_right.tocsr()
        else:
            self._csr_cache = sp.bmat(
                [
                    [self._csr_cache, upper_right],
                    [upper_right.T, lower_right],
                ],
                format="csr",
            )

    def to_csr(self) -> "sp.csr_matrix":
        return self._csr_cache


def _normalize_ci0_sparse(ci0: Any, *, nroots: int, ncsf: int) -> list[SparseVector] | None:
    if ci0 is None:
        return None

    if hasattr(ci0, "to_qmc_x0"):
        ci0 = ci0.to_qmc_x0()

    def _from_sparse_pair(idx_like: Any, val_like: Any) -> SparseVector:
        idx = np.asarray(idx_like)
        if idx.dtype.kind not in ("i", "u"):
            raise ValueError("sparse ci0 indices must be integers")
        idx = np.asarray(idx, dtype=np.int64).ravel()
        val = np.asarray(val_like, dtype=np.float64).ravel()
        if idx.size != val.size:
            raise ValueError("sparse ci0 index/value size mismatch")
        if idx.size and (int(np.min(idx)) < 0 or int(np.max(idx)) >= int(ncsf)):
            raise ValueError("sparse ci0 contains out-of-range indices")
        if idx.size > 1:
            order = np.argsort(idx, kind="stable")
            idx = np.asarray(idx[order], dtype=np.int64, order="C")
            val = np.asarray(val[order], dtype=np.float64, order="C")
            change = np.ones(idx.shape, dtype=np.bool_)
            change[1:] = idx[1:] != idx[:-1]
            if not np.all(change):
                starts = np.flatnonzero(change)
                idx = np.asarray(idx[starts], dtype=np.int64, order="C")
                val = np.asarray(np.add.reduceat(val, starts), dtype=np.float64, order="C")
        return SparseVector(idx, val)

    def _from_dense(x: Any) -> SparseVector:
        v = np.asarray(x, dtype=np.float64).ravel()
        if int(v.size) != int(ncsf):
            raise ValueError("ci0 has wrong length")
        mask = np.abs(v) > 0.0
        return SparseVector(np.asarray(np.nonzero(mask)[0], dtype=np.int64), np.asarray(v[mask], dtype=np.float64))

    if isinstance(ci0, SparseVector):
        out = [SparseVector(np.asarray(ci0.idx, dtype=np.int64), np.asarray(ci0.val, dtype=np.float64))]
    elif isinstance(ci0, tuple) and len(ci0) == 2 and not np.isscalar(ci0[0]):
        out = [_from_sparse_pair(ci0[0], ci0[1])]
    elif isinstance(ci0, list):
        out = []
        for x in ci0[:nroots]:
            if isinstance(x, SparseVector):
                out.append(SparseVector(np.asarray(x.idx, dtype=np.int64), np.asarray(x.val, dtype=np.float64)))
            elif isinstance(x, tuple) and len(x) == 2:
                out.append(_from_sparse_pair(x[0], x[1]))
            else:
                out.append(_from_dense(x))
    else:
        v = np.asarray(ci0, dtype=np.float64)
        if v.ndim == 2:
            out = [_from_dense(v[i]) for i in range(min(int(v.shape[0]), int(nroots)))]
        else:
            out = [_from_dense(v)]

    if not out:
        return None
    while len(out) < int(nroots):
        out.append(SparseVector(np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)))
    return out


def _initial_selection_sparse(
    *,
    ncsf: int,
    nroots: int,
    init_ncsf: int,
    hdiag_lookup: DiagonalGuessLookup,
    ci0_sparse: list[SparseVector] | None,
) -> list[int]:
    init_ncsf = int(init_ncsf)
    if init_ncsf <= 0:
        init_ncsf = max(1, int(nroots))
    init_ncsf = min(init_ncsf, int(ncsf))

    sel: list[int] = []
    seen: set[int] = set()

    if ci0_sparse is not None:
        per_root = max(1, init_ncsf // max(1, nroots))
        per_root = max(per_root, min(64, init_ncsf))
        for rv in ci0_sparse[: int(nroots)]:
            idx = np.asarray(rv.idx, dtype=np.int64).ravel()
            val = np.asarray(rv.val, dtype=np.float64).ravel()
            if idx.size == 0:
                continue
            order = _topk_desc_indices(np.abs(val), per_root)
            for ii in idx[order].tolist():
                jj = int(ii)
                if jj in seen:
                    continue
                sel.append(jj)
                seen.add(jj)
                if len(sel) >= init_ncsf:
                    return sel

    if len(sel) < init_ncsf:
        need = int(init_ncsf) - len(sel)
        if hdiag_lookup.has_dense:
            hdiag_dense = hdiag_lookup.get_many(np.arange(int(ncsf), dtype=np.int64))
            idx = _bottomk_asc_indices(hdiag_dense, need)
            for ii in np.asarray(idx, dtype=np.int64).tolist():
                jj = int(ii)
                if jj in seen:
                    continue
                sel.append(jj)
                seen.add(jj)
                if len(sel) >= init_ncsf:
                    break
        else:
            probe = min(int(ncsf), max(64, 8 * int(init_ncsf)))
            cand = np.arange(probe, dtype=np.int64)
            h_probe = hdiag_lookup.get_many(cand)
            order = _bottomk_asc_indices(h_probe, need)
            for ii in cand[order].tolist():
                jj = int(ii)
                if jj in seen:
                    continue
                sel.append(jj)
                seen.add(jj)
                if len(sel) >= init_ncsf:
                    break

    if len(sel) < nroots:
        for ii in range(int(nroots)):
            if ii >= int(ncsf):
                break
            if ii in seen:
                continue
            sel.append(ii)
            seen.add(ii)
            if len(sel) >= int(nroots):
                break
    return sel


def _connected_row(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    j: int,
    *,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(eri, DFMOIntegrals):
        return connected_row_sparse_df(
            drt,
            h1e,
            eri,
            int(j),
            max_out=int(max_out),
            screening=screening,
            state_cache=state_cache,
        )
    if isinstance(eri, DeviceDFMOIntegrals):
        if eri.l_full is None:
            raise ValueError("DeviceDFMOIntegrals.l_full is required for sparse connected-row queries")
        l_full = _as_numpy_f64(eri.l_full)
        j_ps = _as_numpy_f64(eri.j_ps)
        pair_norm = _as_numpy_f64(eri.pair_norm) if eri.pair_norm is not None else np.linalg.norm(l_full, axis=1)
        eri_host = DFMOIntegrals(
            norb=int(eri.norb),
            l_full=np.asarray(l_full, dtype=np.float64, order="C"),
            j_ps=np.asarray(j_ps, dtype=np.float64, order="C"),
            pair_norm=np.asarray(pair_norm, dtype=np.float64, order="C"),
        )
        return connected_row_sparse_df(
            drt,
            h1e,
            eri_host,
            int(j),
            max_out=int(max_out),
            screening=screening,
            state_cache=state_cache,
        )
    return connected_row_sparse(
        drt,
        h1e,
        eri,
        int(j),
        max_out=int(max_out),
        screening=screening,
        state_cache=state_cache,
    )


def _build_variational_hamiltonian_sparse(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    loc_map: dict[int, int],
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    row_cache: ConnectedRowCache | None = None,
) -> "sp.csr_matrix":
    if sp is None:  # pragma: no cover
        raise RuntimeError("scipy is required for scalable SCI support")

    nsel = int(len(sel))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for col, j in enumerate(sel):
        i_idx, hij = _connected_row_cached(
            drt,
            h1e,
            eri,
            int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )
        for i, v in zip(i_idx.tolist(), hij.tolist(), strict=False):
            row = loc_map.get(int(i))
            if row is None:
                continue
            rows.append(int(row))
            cols.append(int(col))
            data.append(float(v))

    h = sp.coo_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(nsel, nsel)).tocsr()
    h = (h + h.T) * 0.5
    h.eliminate_zeros()
    return h


def _accumulate_and_score_external_sparse(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    selected_set: set[int],
    c_sel: np.ndarray,
    e_var: np.ndarray,
    hdiag_lookup: DiagonalGuessLookup,
    max_add: int,
    select_threshold: float | None,
    denom_floor: float,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    select_screen_contrib: float,
    row_cache: ConnectedRowCache | None,
    label_lo: int,
    label_hi: int,
    return_c1: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None]:
    nroots = int(e_var.size)
    ext: dict[int, np.ndarray] = {}
    for col, j in enumerate(sel):
        cj = np.asarray(c_sel[col, :], dtype=np.float64)
        max_cj = float(np.max(np.abs(cj)))
        if max_cj == 0.0:
            continue

        i_idx, hij = _connected_row_cached(
            drt,
            h1e,
            eri,
            int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )
        for i, v in zip(i_idx.tolist(), hij.tolist(), strict=False):
            ii = int(i)
            if ii < int(label_lo) or ii >= int(label_hi) or ii in selected_set:
                continue
            vv = float(v)
            if select_screen_contrib > 0.0 and abs(vv) * max_cj < float(select_screen_contrib):
                continue
            acc = ext.get(ii)
            if acc is None:
                ext[ii] = vv * cj
            else:
                acc += vv * cj

    if not ext:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((nroots,), dtype=np.float64),
            0,
            None if not bool(return_c1) else np.zeros((0, nroots), dtype=np.float64),
        )

    cand_i: list[int] = []
    cand_w: list[float] = []
    cand_c1: list[np.ndarray] = []
    e_pt2 = np.zeros((nroots,), dtype=np.float64)
    for ii, p in ext.items():
        denom = np.asarray(e_var - float(hdiag_lookup.get(ii)), dtype=np.float64)
        if denom_floor > 0.0:
            small = np.abs(denom) < denom_floor
            if np.any(small):
                denom = denom.copy()
                denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)
        c1 = p / denom
        w = float(np.max(np.abs(c1)))
        if w == 0.0 or not np.isfinite(w):
            continue
        cand_i.append(int(ii))
        cand_w.append(w)
        if bool(return_c1):
            cand_c1.append(np.asarray(c1, dtype=np.float64))
        e_pt2 += (p * p) / denom

    if not cand_i:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            e_pt2,
            int(len(ext)),
            None if not bool(return_c1) else np.zeros((0, nroots), dtype=np.float64),
        )

    w_arr = np.asarray(cand_w, dtype=np.float64)
    i_arr = np.asarray(cand_i, dtype=np.int64)
    c1_arr = None if not bool(return_c1) else np.asarray(cand_c1, dtype=np.float64)
    max_add = int(max_add)
    if max_add <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            e_pt2,
            int(len(ext)),
            None if c1_arr is None else np.zeros((0, nroots), dtype=np.float64),
        )
    max_add = min(max_add, int(i_arr.size))

    if select_threshold is not None:
        thr = float(select_threshold)
        if thr < 0.0:
            raise ValueError("select_threshold must be >= 0")
        keep = np.nonzero(w_arr >= thr)[0]
        if int(keep.size) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
                e_pt2,
                int(len(ext)),
                None if c1_arr is None else np.zeros((0, nroots), dtype=np.float64),
            )
        if int(keep.size) > max_add:
            keep = keep[np.argpartition(w_arr[keep], -max_add)[-max_add:]]
    else:
        keep = np.argpartition(w_arr, -max_add)[-max_add:]

    keep = keep[np.argsort(w_arr[keep])[::-1]]
    return (
        np.asarray(i_arr[keep], dtype=np.int64),
        np.asarray(w_arr[keep], dtype=np.float64),
        e_pt2,
        int(len(ext)),
        None if c1_arr is None else np.asarray(c1_arr[keep], dtype=np.float64),
    )


def _select_external_sparse(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    selected_set: set[int],
    c_sel: np.ndarray,
    e_var: np.ndarray,
    hdiag_lookup: DiagonalGuessLookup,
    max_add: int,
    select_threshold: float | None,
    denom_floor: float,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    select_screen_contrib: float,
    row_cache: ConnectedRowCache | None = None,
    bucket_plan: SelectorBucketPlan | None = None,
    stats_out: dict[str, Any] | None = None,
    seeds_out: dict[str, Any] | None = None,
) -> tuple[list[int], np.ndarray]:
    nroots = int(e_var.size)
    denom_floor = float(denom_floor)
    if denom_floor < 0.0:
        raise ValueError("denom_floor must be >= 0")
    if bucket_plan is None:
        bucket_plan = _plan_selector_buckets(
            drt,
            h1e,
            eri,
            sel=sel,
            c_sel=c_sel,
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )

    bucket_candidate_sum = 0
    bucket_candidate_max = 0
    bucket_split_count = 0
    e_pt2_total = np.zeros((nroots,), dtype=np.float64)
    final_idx_parts: list[np.ndarray] = []
    final_w_parts: list[np.ndarray] = []
    final_c1_parts: list[np.ndarray] = []
    pending_bounds = list(bucket_plan.bucket_bounds)
    while pending_bounds:
        label_lo, label_hi = pending_bounds.pop(0)
        idx_part, w_part, e_pt2_part, cand_count, c1_part = _accumulate_and_score_external_sparse(
            drt,
            h1e,
            eri,
            sel=sel,
            selected_set=selected_set,
            c_sel=c_sel,
            e_var=e_var,
            hdiag_lookup=hdiag_lookup,
            max_add=max_add,
            select_threshold=select_threshold,
            denom_floor=denom_floor,
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            select_screen_contrib=select_screen_contrib,
            row_cache=row_cache,
            label_lo=int(label_lo),
            label_hi=int(label_hi),
            return_c1=seeds_out is not None,
        )
        split_bounds = _maybe_split_bucket_range(
            int(label_lo),
            int(label_hi),
            cand_count=int(cand_count),
            max_add=int(max_add),
        )
        if len(split_bounds) > 1:
            bucket_split_count += int(len(split_bounds) - 1)
            pending_bounds = list(split_bounds) + pending_bounds
            continue
        bucket_candidate_sum += int(cand_count)
        bucket_candidate_max = max(bucket_candidate_max, int(cand_count))
        e_pt2_total += np.asarray(e_pt2_part, dtype=np.float64)
        if int(idx_part.size) > 0:
            final_idx_parts.append(np.asarray(idx_part, dtype=np.int64))
            final_w_parts.append(np.asarray(w_part, dtype=np.float64))
            if c1_part is not None:
                final_c1_parts.append(np.asarray(c1_part, dtype=np.float64))

    if stats_out is not None:
        stats_out["selector_bucketed"] = bool(bucket_plan.bucketed)
        stats_out["selector_nbuckets"] = int(bucket_plan.nbuckets)
        stats_out["selector_bucket_kind"] = str(bucket_plan.bucket_kind)
        stats_out["selector_active_frontier_edges"] = int(bucket_plan.active_frontier_edges)
        stats_out["selector_bucket_candidate_sum"] = int(bucket_candidate_sum)
        stats_out["selector_bucket_candidate_max"] = int(bucket_candidate_max)
        stats_out["selector_bucket_splits"] = int(bucket_split_count)

    if not final_idx_parts:
        if seeds_out is not None:
            seeds_out["seed_idx"] = np.zeros((0,), dtype=np.int64)
            seeds_out["seed_c1"] = np.zeros((0, nroots), dtype=np.float64)
        return [], e_pt2_total

    idx_all = np.concatenate(final_idx_parts)
    w_all = np.concatenate(final_w_parts)
    c1_all = None if not final_c1_parts else np.concatenate(final_c1_parts, axis=0)
    if int(idx_all.size) == 0:
        if seeds_out is not None:
            seeds_out["seed_idx"] = np.zeros((0,), dtype=np.int64)
            seeds_out["seed_c1"] = np.zeros((0, nroots), dtype=np.float64)
        return [], e_pt2_total

    max_add = int(max_add)
    if max_add <= 0:
        if seeds_out is not None:
            seeds_out["seed_idx"] = np.zeros((0,), dtype=np.int64)
            seeds_out["seed_c1"] = np.zeros((0, nroots), dtype=np.float64)
        return [], e_pt2_total
    keep_n = min(max_add, int(idx_all.size))
    keep = np.argpartition(w_all, -keep_n)[-keep_n:]
    keep = keep[np.argsort(w_all[keep])[::-1]]
    idx_keep = np.asarray(idx_all[keep], dtype=np.int64)
    if seeds_out is not None:
        seeds_out["seed_idx"] = idx_keep.copy()
        seeds_out["seed_c1"] = (
            np.zeros((0, nroots), dtype=np.float64)
            if c1_all is None
            else np.asarray(c1_all[keep], dtype=np.float64)
        )
    return [int(x) for x in idx_keep.tolist()], e_pt2_total


def _make_hdiag_guess(drt: DRT, h1e: np.ndarray, eri: Any, *, state_cache: DRTStateCache) -> np.ndarray:
    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if ncsf <= 0:
        return np.zeros((0,), dtype=np.float64)

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")
    h1e_diag = np.diag(h1e)

    if isinstance(eri, (DFMOIntegrals, DeviceDFMOIntegrals)):
        diag_ids = (np.arange(norb, dtype=np.int32) * (norb + 1)).astype(np.int32, copy=False)
        l_full_obj = getattr(eri, "l_full", None)
        pair_norm_obj = getattr(eri, "pair_norm", None)
        eri_mat_obj = getattr(eri, "eri_mat", None)

        if l_full_obj is not None:
            l_full = _as_numpy_f64(l_full_obj)
            if pair_norm_obj is None:
                pair_norm = np.linalg.norm(l_full, axis=1)
            else:
                pair_norm = _as_numpy_f64(pair_norm_obj)
            l_diag = np.asarray(l_full[diag_ids], dtype=np.float64, order="C")
            eri_ppqq = l_diag @ l_diag.T
            eri_pqqp = np.square(pair_norm.reshape(norb, norb))
        elif eri_mat_obj is not None:
            eri_mat = _as_numpy_f64(eri_mat_obj)
            eri_ppqq = np.asarray(eri_mat[np.ix_(diag_ids, diag_ids)], dtype=np.float64, order="C")
            pq = np.arange(norb * norb, dtype=np.int32).reshape(norb, norb)
            qp = pq.T.reshape(norb, norb)
            eri_pqqp = np.asarray(
                eri_mat[pq.reshape(-1), qp.reshape(-1)].reshape(norb, norb),
                dtype=np.float64,
                order="C",
            )
        else:
            raise ValueError("DF integrals must provide either l_full or eri_mat for hdiag construction")
    else:
        eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)
        eri_ppqq = np.einsum("iijj->ij", eri4)
        eri_pqqp = np.einsum("ijji->ij", eri4)

    steps = np.asarray(state_cache.steps, dtype=np.int8)
    if steps.shape != (ncsf, norb):
        raise RuntimeError("internal error: invalid cached steps table shape")
    occ = _STEP_TO_OCC[steps]

    doubly = occ == 2
    singles = occ == 1

    neleca_det = (int(drt.nelec) + int(drt.twos_target)) // 2
    ndoubly = np.sum(doubly, axis=1, dtype=np.int32)
    alpha_need = np.asarray(neleca_det, dtype=np.int32) - ndoubly

    single_prefix = np.cumsum(singles, axis=1, dtype=np.int32)
    alpha_single = singles & (single_prefix <= alpha_need[:, None])
    beta_single = singles & (~alpha_single)

    alpha = (doubly | alpha_single).astype(np.float64)
    beta = (doubly | beta_single).astype(np.float64)
    n = alpha + beta

    hdiag = n @ h1e_diag
    tmp = n @ eri_ppqq
    hdiag += 0.5 * np.sum(tmp * n, axis=1)
    tmp_a = alpha @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_a * alpha, axis=1)
    tmp_b = beta @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_b * beta, axis=1)
    return np.asarray(hdiag, dtype=np.float64)


def _solve_subspace(
    h: "sp.csr_matrix",
    *,
    nroots: int,
    dense_limit: int,
    eigsh_tol: float,
    v0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if sp is None or spla is None:  # pragma: no cover
        raise RuntimeError("scipy is required for scalable SCI support")

    nsel = int(h.shape[0])
    nroots = int(nroots)
    if nsel <= 0:
        raise ValueError("empty variational space")
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    if nroots > nsel:
        raise ValueError("nroots must be <= len(sel)")

    dense_limit = int(dense_limit)
    if dense_limit <= 0:
        dense_limit = 1024

    use_dense = nsel <= dense_limit or nsel <= max(4 * nroots + 16, nroots + 2)
    if use_dense:
        hd = np.asarray(h.toarray(), dtype=np.float64)
        evals, evecs = np.linalg.eigh(hd)
        return (
            np.asarray(evals[:nroots], dtype=np.float64),
            np.asarray(evecs[:, :nroots], dtype=np.float64),
        )

    evals, evecs = spla.eigsh(
        h,
        k=int(nroots),
        which="SA",
        tol=float(eigsh_tol),
        v0=None if v0 is None else np.asarray(v0, dtype=np.float64).ravel(),
    )
    order = np.argsort(evals)
    return (
        np.asarray(evals[order], dtype=np.float64),
        np.asarray(evecs[:, order], dtype=np.float64),
    )
