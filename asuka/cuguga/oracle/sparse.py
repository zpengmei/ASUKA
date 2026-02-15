from __future__ import annotations

import time

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.epq.action import epq_apply_g, epq_apply_weighted_many, epq_contribs_one, path_nodes
from asuka.cuguga.oracle import _STEP_TO_OCC, _restore_eri_4d
from asuka.cuguga.row_stats import RowStats
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache

try:  # optional compiled fast path for dense accumulator updates
    from asuka._epq_cy import dense_row_mask_add_many_cy as _dense_row_mask_add_many_cy
except Exception:  # pragma: no cover
    _dense_row_mask_add_many_cy = None

try:  # optional compiled fast path: apply g[p,q] directly into dense accumulator
    from asuka._epq_cy import epq_apply_g_accum_cy as _epq_apply_g_accum_cy
except Exception:  # pragma: no cover
    _epq_apply_g_accum_cy = None


_DENSE_ACC_NCSF_MAX = 10_000_000

_DENSE_MASK_ACC_CACHE: dict[int, "_DenseRowMaskAccumulator"] = {}


class _DenseRowMaskAccumulator:
    __slots__ = ("row", "mask", "_dirty")

    def __init__(self, ncsf: int) -> None:
        ncsf = int(ncsf)
        if ncsf < 1:
            raise ValueError("ncsf must be >= 1")
        self.row = np.zeros(ncsf, dtype=np.float64)
        self.mask = np.zeros(ncsf, dtype=np.bool_)
        self._dirty = False

    @classmethod
    def get_cached(cls, ncsf: int) -> "_DenseRowMaskAccumulator":
        ncsf = int(ncsf)
        acc = _DENSE_MASK_ACC_CACHE.get(ncsf)
        if acc is None:
            acc = cls(ncsf)
            _DENSE_MASK_ACC_CACHE[ncsf] = acc
        return acc

    def add_scalar(self, i: int, v: float) -> None:
        ii = int(i)
        vv = float(v)
        self.row[ii] += vv
        self.mask[ii] = True
        self._dirty = True

    def add_many(self, idx: np.ndarray, val: np.ndarray) -> None:
        idx_i32 = np.asarray(idx, dtype=np.int32).ravel()
        if idx_i32.size == 0:
            return
        val_f64 = np.asarray(val, dtype=np.float64).ravel()
        if val_f64.size != idx_i32.size:
            raise ValueError("idx and val must have the same size")
        if _dense_row_mask_add_many_cy is not None:
            _dense_row_mask_add_many_cy(self.row, self.mask, idx_i32, val_f64)
        else:
            np.add.at(self.row, idx_i32, val_f64)
            self.mask[idx_i32] = True
        self._dirty = True

    def extract_coalesced(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (idx_u:int32[], val_u:float64[]) and reset internal scratch state."""
        if not self._dirty:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
        idx_i64 = np.flatnonzero(self.mask)
        val_u = np.asarray(self.row[idx_i64], dtype=np.float64).copy()
        # Reset scratch for reuse.
        self.row[idx_i64] = 0.0
        self.mask[idx_i64] = False
        self._dirty = False
        return idx_i64.astype(np.int32), val_u

    def clear(self) -> None:
        """Reset any previously touched entries (for safety after exceptions)."""
        if not self._dirty:
            return
        idx_i64 = np.flatnonzero(self.mask)
        self.row[idx_i64] = 0.0
        self.mask[idx_i64] = False
        self._dirty = False


def _as_f64_square(a: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.shape != (int(n), int(n)):
        raise ValueError(f"{name} has wrong shape: {arr.shape} (expected {(int(n), int(n))})")
    return arr


def _select_offdiag_pq(
    mat: np.ndarray,
    occ: np.ndarray,
    *,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select off-diagonal (p,q) pairs allowed by `occ` with |mat[p,q]| > thresh.

    Parameters
    ----------
    mat
        Square matrix (norb,norb) of coefficients (e.g. g[p,q]).
    occ
        Occupations (length norb) with values in {0,1,2}.
    thresh
        Absolute threshold on |mat[p,q]|. If 0, selects entries with mat[p,q] != 0.

    Returns
    -------
    p_idx, q_idx, val
        Arrays of orbital indices and corresponding matrix elements, with p!=q and
        satisfying the occupancy constraint for E_pq (occ[q]>0 and occ[p]<2).
    """

    occ_i8 = np.asarray(occ, dtype=np.int8).ravel()
    norb = int(occ_i8.size)
    g = np.asarray(mat, dtype=np.float64).reshape(norb, norb)

    thresh_f = float(thresh)
    if thresh_f > 0.0:
        mask = np.abs(g) > thresh_f
    else:
        mask = g != 0.0

    if not np.any(mask):
        empty_i = np.zeros(0, dtype=np.int32)
        empty_v = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i, empty_v

    # Occupancy feasibility: E_pq moves one electron q->p.
    mask &= (occ_i8 < 2)[:, None]
    mask &= (occ_i8 > 0)[None, :]
    # Exclude diagonal p==q.
    diag = np.arange(norb)
    mask[diag, diag] = False

    p_idx, q_idx = np.nonzero(mask)
    val = g[p_idx, q_idx]
    return (
        np.asarray(p_idx, dtype=np.int32, order="C"),
        np.asarray(q_idx, dtype=np.int32, order="C"),
        np.asarray(val, dtype=np.float64, order="C"),
    )


def _coalesce_coo_i32_f64(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx_i32 = np.asarray(idx, dtype=np.int32).ravel()
    val_f64 = np.asarray(val, dtype=np.float64).ravel()
    if idx_i32.size != val_f64.size:
        raise ValueError("idx and val must have the same size")
    if idx_i32.size <= 1:
        return np.ascontiguousarray(idx_i32), np.ascontiguousarray(val_f64)

    order = np.argsort(idx_i32, kind="stable")
    idx_s = idx_i32[order]
    val_s = val_f64[order]
    change = idx_s[1:] != idx_s[:-1]
    ngrp = int(np.count_nonzero(change)) + 1
    starts = np.empty(ngrp, dtype=np.int64)
    starts[0] = 0
    if ngrp > 1:
        starts[1:] = np.flatnonzero(change) + 1

    idx_u = np.asarray(idx_s[starts], dtype=np.int32, order="C")
    val_u = np.asarray(np.add.reduceat(val_s, starts), dtype=np.float64, order="C")
    return idx_u, val_u


def _finalize_coalesced_row(
    j: int, idx_u: np.ndarray, val_u: np.ndarray, *, max_out: int
) -> tuple[np.ndarray, np.ndarray]:
    j = int(j)
    idx_u = np.asarray(idx_u, dtype=np.int32).ravel()
    val_u = np.asarray(val_u, dtype=np.float64).ravel()
    if idx_u.size != val_u.size:
        raise ValueError("idx_u and val_u must have the same size")

    if idx_u.size == 0:
        i_idx_arr = np.asarray([j], dtype=np.int32)
        hij_arr = np.asarray([0.0], dtype=np.float64)
    else:
        pos = int(np.searchsorted(idx_u, j))
        has_j = pos < int(idx_u.size) and int(idx_u[pos]) == j
        if has_j:
            hij_j = float(val_u[pos])
            if idx_u.size == 1:
                i_idx_arr = np.asarray([j], dtype=np.int32)
                hij_arr = np.asarray([hij_j], dtype=np.float64)
            else:
                i_idx_arr = np.empty(int(idx_u.size), dtype=np.int32)
                hij_arr = np.empty(int(val_u.size), dtype=np.float64)
                i_idx_arr[0] = np.int32(j)
                hij_arr[0] = np.float64(hij_j)
                if pos:
                    i_idx_arr[1 : pos + 1] = idx_u[:pos]
                    hij_arr[1 : pos + 1] = val_u[:pos]
                if pos + 1 < int(idx_u.size):
                    i_idx_arr[pos + 1 :] = idx_u[pos + 1 :]
                    hij_arr[pos + 1 :] = val_u[pos + 1 :]
        else:
            i_idx_arr = np.empty(int(idx_u.size) + 1, dtype=np.int32)
            hij_arr = np.empty(int(val_u.size) + 1, dtype=np.float64)
            i_idx_arr[0] = np.int32(j)
            hij_arr[0] = np.float64(0.0)
            i_idx_arr[1:] = idx_u
            hij_arr[1:] = val_u

    if i_idx_arr.size > int(max_out):
        raise ValueError(f"oracle produced {i_idx_arr.size} entries > max_out={int(max_out)}")
    return np.ascontiguousarray(i_idx_arr), np.ascontiguousarray(hij_arr)
def connected_row_sparse(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    j: int,
    *,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    state_cache: DRTStateCache | None = None,
    stats: RowStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (i_idx, hij) for CSF j without global E_pq caches.

    Notes
    -----
    It matches the same Hamiltonian convention as `oracle.connected_row`:

      H = Σ_pq h_pq E_pq + 1/2 Σ_pqrs (pq|rs) (E_pq E_rs - δ_qr E_ps)

    The contraction term and the r==s slice are folded into `h_eff` exactly as
    in the reference row oracle.
    """

    # DF/RI dispatch: allow callers (notably QMC) to pass DF integrals through the
    # same API used for dense integrals.
    if isinstance(eri, DFMOIntegrals):
        return connected_row_sparse_df(
            drt,
            h1e,
            eri,
            j,
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            stats=stats,
        )

    j = int(j)
    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")
    screen = RowScreening() if screening is None else screening

    norb = int(drt.norb)
    h1e = _as_f64_square(h1e, norb, "h1e")
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    t0 = time.perf_counter() if stats is not None else 0.0
    if state_cache is None:
        steps_j = drt.index_to_path(j).astype(np.int8, copy=False)
        nodes_j = path_nodes(drt, steps_j)
    else:
        steps_j = state_cache.steps[j]
        nodes_j = state_cache.nodes[j]
    occ_j = _STEP_TO_OCC[steps_j].astype(np.int8, copy=False)
    if stats is not None:
        stats.add_time("state_j", time.perf_counter() - t0)

    t0 = time.perf_counter() if stats is not None else 0.0
    # h_eff = h1e - 0.5*Σ_q(p q| q s) + 0.5*Σ_r(p q| r r) occ_r(j)
    h_eff = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)
    h_eff = h_eff + 0.5 * np.einsum("pqrr,r->pq", eri4, occ_j, optimize=True)
    if stats is not None:
        stats.add_time("h_eff", time.perf_counter() - t0)

    ncsf = int(drt.ncsf)
    use_dense_acc = ncsf <= int(_DENSE_ACC_NCSF_MAX)
    dense_acc = _DenseRowMaskAccumulator.get_cached(ncsf) if use_dense_acc else None
    dense_apply_g = dense_acc is not None and _epq_apply_g_accum_cy is not None

    diag_j = float(np.dot(h_eff.diagonal(), occ_j))
    if dense_apply_g:
        dense_acc.clear()
        dense_acc.add_scalar(j, diag_j)
        coo_i: list[np.ndarray] = []
        coo_v: list[np.ndarray] = []
    else:
        # Accumulate as COO and coalesce at the end.
        coo_i = [np.asarray([j], dtype=np.int32)]
        coo_v = [np.asarray([diag_j], dtype=np.float64)]

    # One-body off-diagonal: restrict to feasible moves E_pq on |j>.
    t0 = time.perf_counter() if stats is not None else 0.0
    src1 = np.nonzero(occ_j > 0)[0].tolist()
    dst1 = np.nonzero(occ_j < 2)[0].tolist()
    p_list: list[int] = []
    q_list: list[int] = []
    w_list: list[float] = []
    thresh_h1 = float(screen.thresh_h1e)
    thresh_rs_coeff = float(screen.thresh_rs_coeff)
    thresh_gpq = float(screen.thresh_gpq)
    thresh_contrib = float(screen.thresh_contrib)
    for q in src1:
        for p in dst1:
            if p == q:
                continue
            hpq = float(h_eff[int(p), int(q)])
            if hpq == 0.0 or abs(hpq) <= thresh_h1:
                continue
            p_list.append(int(p))
            q_list.append(int(q))
            w_list.append(hpq)

    epq_calls_h1 = int(len(p_list))
    nnz_h1 = 0
    if epq_calls_h1:
        i_idx, val = epq_apply_weighted_many(
            drt,
            j,
            np.asarray(p_list, dtype=np.int32),
            np.asarray(q_list, dtype=np.int32),
            np.asarray(w_list, dtype=np.float64),
            steps=steps_j,
            nodes=nodes_j,
            thresh_contrib=thresh_contrib,
            trusted=True,
        )
        nnz_h1 = int(i_idx.size)
        if i_idx.size:
            if dense_apply_g:
                assert dense_acc is not None
                dense_acc.add_many(i_idx, val)
            else:
                coo_i.append(i_idx)
                coo_v.append(val)
    if stats is not None:
        stats.add_time("h1_offdiag", time.perf_counter() - t0)
        stats.inc("epq_calls_h1", epq_calls_h1)
        stats.inc("nnz_h1", nnz_h1)

    # Two-body product terms with r!=s via intermediate states k = E_rs |j>.
    nops = norb * norb
    eri_mat = eri4.reshape(nops, nops)
    g_buf = np.empty(nops, dtype=np.float64)

    t0 = time.perf_counter() if stats is not None else 0.0
    epq_calls_rs = 0
    nnz_rs = 0
    by_k: dict[int, list[tuple[int, float]]] = {}
    for s in src1:
        for r in dst1:
            if r == s:
                continue
            epq_calls_rs += 1
            k_idx, c_rs = epq_contribs_one(drt, j, int(r), int(s), steps=steps_j, nodes=nodes_j)
            nnz_rs += int(k_idx.size)
            if k_idx.size == 0:
                continue
            rs_id = int(r) * norb + int(s)
            for kk, cc in zip(k_idx, c_rs):
                by_k.setdefault(int(kk), []).append((rs_id, float(cc)))
    if stats is not None:
        stats.add_time("rs_enum", time.perf_counter() - t0)
        stats.inc("epq_calls_rs", epq_calls_rs)
        stats.inc("nnz_rs", nnz_rs)
        stats.inc("n_k", int(len(by_k)))

    t_apply_total = time.perf_counter() if stats is not None else 0.0
    t_gbuild_total = 0.0
    epq_calls_apply = 0
    nnz_apply = 0
    for k, rs_terms in by_k.items():
        t1 = time.perf_counter() if stats is not None else 0.0
        if len(rs_terms) == 1:
            rs_id, rs_c = rs_terms[0]
            if thresh_rs_coeff > 0.0 and abs(float(rs_c)) <= thresh_rs_coeff:
                continue
            np.multiply(
                eri_mat[:, int(rs_id)],
                0.5 * float(rs_c),
                out=g_buf,
            )
            g_flat = g_buf
        else:
            rs_ids = np.asarray([t[0] for t in rs_terms], dtype=np.int32)
            rs_coeff = np.asarray([t[1] for t in rs_terms], dtype=np.float64)
            if rs_ids.size > 1:
                order = np.argsort(rs_ids, kind="mergesort")
                rs_ids = rs_ids[order]
                rs_coeff = rs_coeff[order]
                change = np.nonzero(rs_ids[1:] != rs_ids[:-1])[0] + 1
                if change.size:
                    starts = np.concatenate(([0], change)).astype(np.int32, copy=False)
                    rs_ids = rs_ids[starts]
                    rs_coeff = np.add.reduceat(rs_coeff, starts)
            if thresh_rs_coeff > 0.0:
                keep = np.abs(rs_coeff) > thresh_rs_coeff
                rs_ids = rs_ids[keep]
                rs_coeff = rs_coeff[keep]
                if rs_ids.size == 0:
                    continue
            np.matmul(eri_mat[:, rs_ids], rs_coeff, out=g_buf)
            g_buf *= 0.5
            g_flat = g_buf
        if stats is not None:
            t_gbuild_total += time.perf_counter() - t1

        if state_cache is None:
            steps_k = drt.index_to_path(int(k)).astype(np.int8, copy=False)
            nodes_k = path_nodes(drt, steps_k)
        else:
            steps_k = state_cache.steps[int(k)]
            nodes_k = state_cache.nodes[int(k)]

        t_epq = time.perf_counter() if stats is not None else 0.0
        if dense_apply_g:
            assert dense_acc is not None
            assert _epq_apply_g_accum_cy is not None
            n_pairs, n_out = _epq_apply_g_accum_cy(
                drt,
                int(k),
                g_flat,
                steps_k,
                nodes_k,
                dense_acc.row,
                dense_acc.mask,
                thresh_gpq,
                thresh_contrib,
            )
            dense_acc._dirty = True
        else:
            i_idx, val, n_pairs = epq_apply_g(
                drt,
                int(k),
                g_flat,
                steps=steps_k,
                nodes=nodes_k,
                thresh_gpq=thresh_gpq,
                thresh_contrib=thresh_contrib,
            )
        epq_calls_apply += int(n_pairs)
        if stats is not None:
            stats.add_time("apply_epq", time.perf_counter() - t_epq)
        if dense_apply_g:
            nnz_apply += int(n_out)
        else:
            nnz_apply += int(i_idx.size)
            if i_idx.size:
                coo_i.append(i_idx)
                coo_v.append(val)
    if stats is not None:
        stats.add_time("g_build", float(t_gbuild_total))
        stats.add_time("apply_g", time.perf_counter() - t_apply_total - float(t_gbuild_total))
        stats.inc("epq_calls_apply", epq_calls_apply)
        stats.inc("nnz_apply", nnz_apply)

    # Materialize sparse row output.
    t0 = time.perf_counter() if stats is not None else 0.0
    if dense_apply_g:
        assert dense_acc is not None
        idx_u, val_u = dense_acc.extract_coalesced()
    else:
        idx_all = np.concatenate(coo_i)
        val_all = np.concatenate(coo_v)
        if dense_acc is not None:
            dense_acc.clear()
            dense_acc.add_many(idx_all, val_all)
            idx_u, val_u = dense_acc.extract_coalesced()
        else:
            idx_u, val_u = _coalesce_coo_i32_f64(idx_all, val_all)
    if stats is not None:
        stats.add_time("apply_accum", time.perf_counter() - t0)

    t0 = time.perf_counter() if stats is not None else 0.0
    i_idx_arr, hij_arr = _finalize_coalesced_row(j, idx_u, val_u, max_out=max_out)
    if stats is not None:
        stats.add_time("finalize", time.perf_counter() - t0)
    return i_idx_arr, hij_arr


def connected_row_sparse_df(
    drt: DRT,
    h1e: np.ndarray,
    df_eri: DFMOIntegrals,
    j: int,
    *,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    state_cache: DRTStateCache | None = None,
    stats: RowStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """DF-backed sparse row oracle (no global E_pq caches)."""

    j = int(j)
    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")
    screen = RowScreening() if screening is None else screening

    norb = int(drt.norb)
    if int(df_eri.norb) != norb:
        raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")

    h1e = _as_f64_square(h1e, norb, "h1e")

    t0 = time.perf_counter() if stats is not None else 0.0
    if state_cache is None:
        steps_j = drt.index_to_path(j).astype(np.int8, copy=False)
        nodes_j = path_nodes(drt, steps_j)
    else:
        steps_j = state_cache.steps[j]
        nodes_j = state_cache.nodes[j]
    occ_j = _STEP_TO_OCC[steps_j].astype(np.int8, copy=False)
    if stats is not None:
        stats.add_time("state_j", time.perf_counter() - t0)

    t0 = time.perf_counter() if stats is not None else 0.0
    # h_eff = h1e - 0.5*J_ps + 0.5*Σ_r(p q| r r) occ_r(j)
    h_eff = h1e - 0.5 * np.asarray(df_eri.j_ps, dtype=np.float64)
    h_eff = h_eff + df_eri.rr_slice_h_eff(occ_j, half=0.5, eri_mat_max_bytes=int(screen.df_eri_mat_max_bytes))
    if stats is not None:
        stats.add_time("h_eff", time.perf_counter() - t0)

    ncsf = int(drt.ncsf)
    use_dense_acc = ncsf <= int(_DENSE_ACC_NCSF_MAX)
    dense_acc = _DenseRowMaskAccumulator.get_cached(ncsf) if use_dense_acc else None
    dense_apply_g = dense_acc is not None and _epq_apply_g_accum_cy is not None

    diag_j = float(np.dot(h_eff.diagonal(), occ_j))
    if dense_apply_g:
        dense_acc.clear()
        dense_acc.add_scalar(j, diag_j)
        coo_i: list[np.ndarray] = []
        coo_v: list[np.ndarray] = []
    else:
        # Accumulate as COO and coalesce at the end.
        coo_i = [np.asarray([j], dtype=np.int32)]
        coo_v = [np.asarray([diag_j], dtype=np.float64)]

    t0 = time.perf_counter() if stats is not None else 0.0
    src1 = np.nonzero(occ_j > 0)[0].tolist()
    dst1 = np.nonzero(occ_j < 2)[0].tolist()
    p_list: list[int] = []
    q_list: list[int] = []
    w_list: list[float] = []
    thresh_h1 = float(screen.thresh_h1e)
    thresh_gpq = float(screen.thresh_gpq)
    thresh_contrib = float(screen.thresh_contrib)
    for q in src1:
        for p in dst1:
            if p == q:
                continue
            hpq = float(h_eff[int(p), int(q)])
            if hpq == 0.0 or abs(hpq) <= thresh_h1:
                continue
            p_list.append(int(p))
            q_list.append(int(q))
            w_list.append(hpq)

    epq_calls_h1 = int(len(p_list))
    nnz_h1 = 0
    if epq_calls_h1:
        i_idx, val = epq_apply_weighted_many(
            drt,
            j,
            np.asarray(p_list, dtype=np.int32),
            np.asarray(q_list, dtype=np.int32),
            np.asarray(w_list, dtype=np.float64),
            steps=steps_j,
            nodes=nodes_j,
            thresh_contrib=thresh_contrib,
            trusted=True,
        )
        nnz_h1 = int(i_idx.size)
        if i_idx.size:
            if dense_apply_g:
                assert dense_acc is not None
                dense_acc.add_many(i_idx, val)
            else:
                coo_i.append(i_idx)
                coo_v.append(val)
    if stats is not None:
        stats.add_time("h1_offdiag", time.perf_counter() - t0)
        stats.inc("epq_calls_h1", epq_calls_h1)
        stats.inc("nnz_h1", nnz_h1)

    # 2-body product terms with r!=s (r==s already absorbed into h_eff above).
    # Collect sparse coefficients for all intermediates k = E_rs|j⟩ in COO form, then
    # build g_k[pq] in a batched DF contraction.
    coo_k: list[int] = []
    coo_rs: list[int] = []
    coo_c: list[float] = []

    t0 = time.perf_counter() if stats is not None else 0.0
    epq_calls_rs = 0
    nnz_rs = 0
    for s in src1:
        for r in dst1:
            if r == s:
                continue
            epq_calls_rs += 1
            k_idx, c_rs = epq_contribs_one(drt, j, int(r), int(s), steps=steps_j, nodes=nodes_j)
            nnz_rs += int(k_idx.size)
            if k_idx.size == 0:
                continue
            rs_id = int(r) * norb + int(s)
            for kk, cc in zip(k_idx, c_rs):
                coo_k.append(int(kk))
                coo_rs.append(rs_id)
                coo_c.append(float(cc))
    if stats is not None:
        stats.add_time("rs_enum", time.perf_counter() - t0)
        stats.inc("epq_calls_rs", epq_calls_rs)
        stats.inc("nnz_rs", nnz_rs)
        stats.inc("coo_entries", int(len(coo_k)))

    if coo_k:
        from scipy import sparse

        nops = norb * norb
        k_idx_all = np.asarray(coo_k, dtype=np.int32)
        rs_id_all = np.asarray(coo_rs, dtype=np.int32)
        coeff_all = np.asarray(coo_c, dtype=np.float64)

        # Coalesce duplicates in (k, rs_id) and apply screening on the summed coefficient,
        # matching the previous per-k coalesce semantics.
        if k_idx_all.size > 1:
            t0 = time.perf_counter() if stats is not None else 0.0
            order = np.lexsort((rs_id_all, k_idx_all))
            k_idx_all = k_idx_all[order]
            rs_id_all = rs_id_all[order]
            coeff_all = coeff_all[order]
            change = (k_idx_all[1:] != k_idx_all[:-1]) | (rs_id_all[1:] != rs_id_all[:-1])
            if np.any(change):
                starts = np.concatenate(([0], np.nonzero(change)[0] + 1)).astype(np.int32, copy=False)
                k_idx_all = k_idx_all[starts]
                rs_id_all = rs_id_all[starts]
                coeff_all = np.add.reduceat(coeff_all, starts)
            if stats is not None:
                stats.add_time("rs_coalesce", time.perf_counter() - t0)

        t0 = time.perf_counter() if stats is not None else 0.0
        if float(screen.thresh_rs_coeff) > 0.0:
            keep = np.abs(coeff_all) > float(screen.thresh_rs_coeff)
            k_idx_all = k_idx_all[keep]
            rs_id_all = rs_id_all[keep]
            coeff_all = coeff_all[keep]

        if float(screen.thresh_rs_pairnorm) > 0.0 and k_idx_all.size:
            keep = np.abs(coeff_all) * df_eri.pair_norm[rs_id_all] > float(screen.thresh_rs_pairnorm)
            k_idx_all = k_idx_all[keep]
            rs_id_all = rs_id_all[keep]
            coeff_all = coeff_all[keep]
        if stats is not None:
            stats.add_time("rs_screen", time.perf_counter() - t0)

        if k_idx_all.size:
            t0 = time.perf_counter() if stats is not None else 0.0
            k_list, row_ids = np.unique(k_idx_all, return_inverse=True)

            # C[row, rs_id] = <k(row)|E_rs|j>
            C = sparse.coo_matrix((coeff_all, (row_ids, rs_id_all)), shape=(k_list.size, nops)).tocsr()
            if stats is not None:
                stats.add_time("C_build", time.perf_counter() - t0)
                stats.inc("n_k", int(k_list.size))
                stats.inc("C_nnz", int(C.nnz))

            eri_mat = df_eri._maybe_build_eri_mat(int(screen.df_eri_mat_max_bytes))
            use_eri_mat = False
            if eri_mat is not None:
                # Decide between:
                #   (A) pair-space matmul:   g = 0.5 * C @ ERI_mat
                #   (B) 2-step DF:           g = 0.5 * (C @ L) @ L.T
                #
                # (B) has a dense GEMM cost O(nk*naux*nops) regardless of sparsity, so it can
                # be slower when the number of intermediates nk is large and the rs pattern
                # is sparse. Use a simple flop-based heuristic to pick the cheaper path.
                nnz = int(C.nnz)
                nk = int(C.shape[0])
                naux = int(df_eri.naux)
                cost_pair = nnz * nops
                cost_two_step = nnz * naux + nk * naux * nops
                use_eri_mat = cost_pair <= cost_two_step
            if stats is not None:
                stats.inc("df_path_eri_mat", int(use_eri_mat))
                stats.inc("df_path_two_step", int(not use_eri_mat))

            if use_eri_mat:
                # Batched sparse-dense matmul in pair space:
                #   g_mat = 0.5 * C @ ERI_mat
                def build_g_block(C_block: sparse.csr_matrix) -> np.ndarray:
                    return 0.5 * (C_block @ eri_mat)
            else:
                # Two-step DF contraction, but batched over all k:
                #   W = C @ L_full   (W[k,L] = sum_rs c_rs(k)*d[L,rs])
                #   g = 0.5 * W @ L_full.T
                l_full = np.asarray(df_eri.l_full, dtype=np.float64, order="C")

                def build_g_block(C_block: sparse.csr_matrix) -> np.ndarray:
                    w_block = C_block @ l_full
                    # Avoid materializing a full transpose copy of `l_full` per row;
                    # NumPy can call BLAS GEMM with `transB='T'` for this transpose-view.
                    return 0.5 * (w_block @ l_full.T)

            # Process k in blocks to cap peak memory for g_block (nk_block * nops).
            g_block_max_bytes = 256 * 1024 * 1024  # 256 MiB
            bytes_per_k = nops * np.dtype(np.float64).itemsize
            k_block = max(1, int(g_block_max_bytes // max(1, bytes_per_k)))

            for start in range(0, int(k_list.size), k_block):
                stop = min(int(k_list.size), start + k_block)
                t0 = time.perf_counter() if stats is not None else 0.0
                g_block = np.asarray(build_g_block(C[start:stop]), dtype=np.float64, order="C")
                if stats is not None:
                    stats.add_time("g_build", time.perf_counter() - t0)

                t0 = time.perf_counter() if stats is not None else 0.0
                epq_calls_apply = 0
                nnz_apply = 0
                for local_row, k in enumerate(k_list[start:stop]):
                    g_flat = g_block[local_row]

                    if state_cache is None:
                        steps_k = drt.index_to_path(int(k)).astype(np.int8, copy=False)
                        nodes_k = path_nodes(drt, steps_k)
                    else:
                        steps_k = state_cache.steps[int(k)]
                        nodes_k = state_cache.nodes[int(k)]

                    t_epq = time.perf_counter() if stats is not None else 0.0
                    if dense_apply_g:
                        assert dense_acc is not None
                        assert _epq_apply_g_accum_cy is not None
                        n_pairs, n_out = _epq_apply_g_accum_cy(
                            drt,
                            int(k),
                            g_flat,
                            steps_k,
                            nodes_k,
                            dense_acc.row,
                            dense_acc.mask,
                            thresh_gpq,
                            thresh_contrib,
                        )
                        dense_acc._dirty = True
                    else:
                        i_idx, val, n_pairs = epq_apply_g(
                            drt,
                            int(k),
                            g_flat,
                            steps=steps_k,
                            nodes=nodes_k,
                            thresh_gpq=thresh_gpq,
                            thresh_contrib=thresh_contrib,
                        )
                    epq_calls_apply += int(n_pairs)
                    if stats is not None:
                        stats.add_time("apply_epq", time.perf_counter() - t_epq)
                    if dense_apply_g:
                        nnz_apply += int(n_out)
                    else:
                        nnz_apply += int(i_idx.size)
                        if i_idx.size:
                            coo_i.append(i_idx)
                            coo_v.append(val)
                if stats is not None:
                    stats.add_time("apply_g", time.perf_counter() - t0)
                    stats.inc("epq_calls_apply", epq_calls_apply)
                    stats.inc("nnz_apply", nnz_apply)

    t0 = time.perf_counter() if stats is not None else 0.0
    if dense_apply_g:
        assert dense_acc is not None
        idx_u, val_u = dense_acc.extract_coalesced()
    else:
        idx_all = np.concatenate(coo_i)
        val_all = np.concatenate(coo_v)
        if dense_acc is not None:
            dense_acc.clear()
            dense_acc.add_many(idx_all, val_all)
            idx_u, val_u = dense_acc.extract_coalesced()
        else:
            idx_u, val_u = _coalesce_coo_i32_f64(idx_all, val_all)
    if stats is not None:
        stats.add_time("apply_accum", time.perf_counter() - t0)

    t0 = time.perf_counter() if stats is not None else 0.0
    i_idx_arr, hij_arr = _finalize_coalesced_row(j, idx_u, val_u, max_out=max_out)
    if stats is not None:
        stats.add_time("finalize", time.perf_counter() - t0)
    return i_idx_arr, hij_arr

