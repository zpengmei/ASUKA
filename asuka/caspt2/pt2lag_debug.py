"""Debug, alignment, and Molcas parity utilities for pt2lag."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from itertools import permutations

import numpy as np


def _asnumpy_f64(a: Any) -> np.ndarray:
    """Convert NumPy/CuPy-like inputs to a NumPy float64 array."""
    if hasattr(a, "get"):
        try:
            a = a.get()
        except Exception:
            pass
    return np.asarray(a, dtype=np.float64)


def _read_molcas_dump_matrix(path: Path) -> np.ndarray:
    """Read a Molcas GRAD_DUMP 2D matrix file.

    Format:
      line 1: ``nrow ncol``
      remaining: ``nrow*ncol`` values written in Fortran order.
    """
    toks = str(path.read_text(encoding="utf-8")).split()
    if len(toks) < 2:
        raise ValueError(f"{path}: invalid dump header")
    nrow = int(toks[0])
    ncol = int(toks[1])
    nval = int(nrow * ncol)
    if len(toks) < 2 + nval:
        raise ValueError(
            f"{path}: insufficient values ({len(toks) - 2} < {nval})"
        )
    vals = [
        float(str(x).replace("D", "E").replace("d", "E"))
        for x in toks[2 : 2 + nval]
    ]
    arr = np.asarray(vals, dtype=np.float64).reshape((nrow, ncol), order="F")
    return arr


def _infer_active_row_signs_from_dump(
    trans_asuka: np.ndarray,
    trans_molcas_dump: np.ndarray,
) -> np.ndarray | None:
    """Infer per-row active-basis sign flips from TRANS overlap with dump.

    Returns a sign vector `s` such that `diag(s) @ trans_asuka` better matches
    the dumped Molcas `TRANS` rows. Returns *None* when no non-trivial sign
    adjustment is detected or when shapes are incompatible.
    """
    ta = np.asarray(trans_asuka, dtype=np.float64)
    tm = np.asarray(trans_molcas_dump, dtype=np.float64)
    if ta.ndim != 2 or tm.ndim != 2 or ta.shape != tm.shape:
        return None
    nrow = int(ta.shape[0])
    if nrow <= 0:
        return None
    signs = np.ones((nrow,), dtype=np.float64)
    changed = False
    for i in range(nrow):
        da = np.asarray(ta[i, :], dtype=np.float64).ravel()
        dm = np.asarray(tm[i, :], dtype=np.float64).ravel()
        dot = float(np.dot(da, dm))
        if dot < 0.0:
            signs[i] = -1.0
            changed = True
    if not changed:
        return None
    return signs


def _best_col_signed_perm_2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_ncol: int = 8,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Best signed column permutation to match ``a`` to ``b``.

    Returns ``(perm, signs)`` where ``a[:, perm] * signs`` best matches ``b``
    in max-abs sense. Intended for small SR bases (`nIN` up to ~8).
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape != bb.shape:
        return None
    ncol = int(aa.shape[1])
    if ncol < 1 or ncol > int(max_ncol):
        return None

    best_metric: float | None = None
    best_perm: np.ndarray | None = None
    best_signs: np.ndarray | None = None
    for perm_tup in permutations(range(ncol)):
        idx = np.asarray(perm_tup, dtype=np.int64)
        ap = np.asarray(aa[:, idx], dtype=np.float64)
        for mask in range(1 << ncol):
            signs = np.ones((ncol,), dtype=np.float64)
            for j in range(ncol):
                if (mask >> j) & 1:
                    signs[j] = -1.0
            cand = np.asarray(ap * signs[None, :], dtype=np.float64)
            m = float(np.max(np.abs(cand - bb)))
            if best_metric is None or m < best_metric:
                best_metric = m
                best_perm = idx.copy()
                best_signs = signs.copy()
    if best_perm is None or best_signs is None:
        return None
    return best_perm, best_signs


def _best_col_signed_perm_assign_2d(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assignment-based signed column map from ``a`` to ``b``.

    Returns ``(perm, signs)`` such that
      ``a[:, perm] * signs[None, :]``
    best matches ``b`` under one-to-one column assignment.

    This scales to larger external-index spaces (e.g. case E/G channels) where
    exhaustive permutation search is impractical.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape != bb.shape:
        return None
    ncol = int(aa.shape[1])
    if ncol <= 0:
        return None

    # Cost for mapping source col i -> target col j, allowing sign flip.
    cost = np.empty((ncol, ncol), dtype=np.float64)
    sign_mat = np.ones((ncol, ncol), dtype=np.float64)
    for i in range(ncol):
        ai = np.asarray(aa[:, i], dtype=np.float64).reshape(-1)
        for j in range(ncol):
            bj = np.asarray(bb[:, j], dtype=np.float64).reshape(-1)
            d_pos = float(np.linalg.norm(ai - bj))
            d_neg = float(np.linalg.norm(ai + bj))
            if d_neg < d_pos:
                cost[i, j] = d_neg
                sign_mat[i, j] = -1.0
            else:
                cost[i, j] = d_pos
                sign_mat[i, j] = 1.0

    row_ind: np.ndarray
    col_ind: np.ndarray
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception:
        # Greedy fallback when SciPy is unavailable.
        used_rows: set[int] = set()
        cols: list[int] = []
        rows: list[int] = []
        for j in range(ncol):
            best_i: int | None = None
            best_c: float | None = None
            for i in range(ncol):
                if i in used_rows:
                    continue
                cij = float(cost[i, j])
                if best_c is None or cij < best_c:
                    best_c = cij
                    best_i = i
            if best_i is None:
                return None
            used_rows.add(int(best_i))
            rows.append(int(best_i))
            cols.append(int(j))
        row_ind = np.asarray(rows, dtype=np.int64)
        col_ind = np.asarray(cols, dtype=np.int64)

    if row_ind.size != ncol or col_ind.size != ncol:
        return None

    perm = np.empty((ncol,), dtype=np.int64)
    signs = np.ones((ncol,), dtype=np.float64)
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        perm[int(j)] = int(i)
        signs[int(j)] = float(sign_mat[int(i), int(j)])
    return perm, signs


def _best_signed_perm_from_overlap_square(
    overlap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Best signed column permutation from an orbital-overlap square block.

    Given overlap ``O = C_left^T S C_right`` (rows: left, cols: right),
    returns ``(perm, signs)`` such that
      ``C_left[:, perm] * signs ~= C_right``.
    """
    o = np.asarray(overlap, dtype=np.float64)
    if o.ndim != 2 or o.shape[0] != o.shape[1]:
        return None
    n = int(o.shape[0])
    if n <= 0:
        return None

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return None

    cost = -np.abs(o)
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.size != n or col_ind.size != n:
        return None

    perm = np.empty((n,), dtype=np.int64)
    signs = np.ones((n,), dtype=np.float64)
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        perm[int(j)] = int(i)
        signs[int(j)] = -1.0 if float(o[int(i), int(j)]) < 0.0 else 1.0
    return perm, signs


def _infer_inact_virt_orbital_maps_from_dump(
    *,
    cmo_asuka: np.ndarray,
    s_ao: np.ndarray,
    dump_dir: Path,
    nish: int,
    nash: int,
    nssh: int,
) -> dict[str, np.ndarray] | None:
    """Infer inactive/virtual orbital signed permutations from CMOPT2 dump."""
    p = Path(dump_dir) / "GRAD_DUMP_CMOPT2.dat"
    if not p.exists():
        return None
    try:
        cmo_dump = _read_molcas_dump_matrix(p)
    except Exception:
        return None

    c_asu = np.asarray(cmo_asuka, dtype=np.float64)
    s = np.asarray(s_ao, dtype=np.float64)
    c_mol = np.asarray(cmo_dump, dtype=np.float64)
    if (
        c_asu.ndim != 2
        or c_mol.ndim != 2
        or s.ndim != 2
        or c_asu.shape != c_mol.shape
        or s.shape[0] != s.shape[1]
        or s.shape[0] != c_asu.shape[0]
    ):
        return None

    ov = np.asarray(c_asu.T @ s @ c_mol, dtype=np.float64)
    nmo = int(c_asu.shape[1])
    if int(nish + nash + nssh) > nmo:
        return None

    out: dict[str, np.ndarray] = {}
    if nish > 0:
        blk_i = np.asarray(ov[:nish, :nish], dtype=np.float64)
        m_i = _best_signed_perm_from_overlap_square(blk_i)
        if m_i is None:
            return None
        perm_i, sign_i = m_i
        out["perm_i"] = np.asarray(perm_i, dtype=np.int64)
        out["sign_i"] = np.asarray(sign_i, dtype=np.float64)
    else:
        out["perm_i"] = np.zeros((0,), dtype=np.int64)
        out["sign_i"] = np.zeros((0,), dtype=np.float64)

    if nssh > 0:
        vs = int(nish + nash)
        ve = int(vs + nssh)
        blk_v = np.asarray(ov[vs:ve, vs:ve], dtype=np.float64)
        m_v = _best_signed_perm_from_overlap_square(blk_v)
        if m_v is None:
            return None
        perm_v, sign_v = m_v
        out["perm_v"] = np.asarray(perm_v, dtype=np.int64)
        out["sign_v"] = np.asarray(sign_v, dtype=np.float64)
    else:
        out["perm_v"] = np.zeros((0,), dtype=np.int64)
        out["sign_v"] = np.zeros((0,), dtype=np.float64)
    return out


def _case_external_map_from_orbital_perms(
    *,
    case: int,
    nish: int,
    nssh: int,
    perm_i: np.ndarray,
    sign_i: np.ndarray,
    perm_v: np.ndarray,
    sign_v: np.ndarray,
    target_order: str = "molcas",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build case external index map/signs from orbital permutation/sign maps.

    Returns ``(perm, signs)`` where
      ``X_new[:, j] = signs[j] * X_old[:, perm[j]]``.
    """
    c = int(case)
    tgt_u = str(target_order).strip().lower()
    use_mol_tgt = tgt_u not in {"", "asuka", "c", "c_order"}

    def _sym_pairs_asuka(n: int) -> list[tuple[int, int]]:
        return [(p, q) for p in range(n) for q in range(p + 1)]

    def _sym_pairs_mol(n: int) -> list[tuple[int, int]]:
        return [(p, q) for q in range(n) for p in range(q, n)]

    def _asym_pairs_asuka(n: int) -> list[tuple[int, int]]:
        return [(p, q) for p in range(n) for q in range(p)]

    def _asym_pairs_mol(n: int) -> list[tuple[int, int]]:
        return [(p, q) for q in range(n) for p in range(q + 1, n)]

    pi = np.asarray(perm_i, dtype=np.int64).reshape(-1)
    si = np.asarray(sign_i, dtype=np.float64).reshape(-1)
    pv = np.asarray(perm_v, dtype=np.int64).reshape(-1)
    sv = np.asarray(sign_v, dtype=np.float64).reshape(-1)
    if int(pi.size) != int(nish) or int(si.size) != int(nish):
        return None
    if int(pv.size) != int(nssh) or int(sv.size) != int(nssh):
        return None

    # Cases with single external index.
    if c == 1 and nish > 0:
        return pi.copy(), si.copy()
    if c == 4 and nssh > 0:
        return pv.copy(), sv.copy()

    # Inactive/virtual pair-only cases.
    if c in (2, 12) and nish > 0:
        a_pairs = _sym_pairs_asuka(nish)
        t_pairs = _sym_pairs_mol(nish) if use_mol_tgt else _sym_pairs_asuka(nish)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        perm = np.empty((len(t_pairs),), dtype=np.int64)
        sign = np.ones((len(t_pairs),), dtype=np.float64)
        for j, (i_t, l_t) in enumerate(t_pairs):
            i_a = int(pi[i_t])
            l_a = int(pi[l_t])
            s = float(si[i_t] * si[l_t])
            if i_a < l_a:
                i_a, l_a = l_a, i_a
            perm[j] = int(idx_a[(i_a, l_a)])
            sign[j] = s
        return perm, sign
    if c in (3, 13) and nish > 1:
        a_pairs = _asym_pairs_asuka(nish)
        t_pairs = _asym_pairs_mol(nish) if use_mol_tgt else _asym_pairs_asuka(nish)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        perm = np.empty((len(t_pairs),), dtype=np.int64)
        sign = np.ones((len(t_pairs),), dtype=np.float64)
        for j, (i_t, l_t) in enumerate(t_pairs):
            i_a = int(pi[i_t])
            l_a = int(pi[l_t])
            s = float(si[i_t] * si[l_t])
            if i_a < l_a:
                i_a, l_a = l_a, i_a
                s *= -1.0
            perm[j] = int(idx_a[(i_a, l_a)])
            sign[j] = s
        return perm, sign
    if c == 8 and nssh > 0:
        a_pairs = _sym_pairs_asuka(nssh)
        t_pairs = _sym_pairs_mol(nssh) if use_mol_tgt else _sym_pairs_asuka(nssh)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        perm = np.empty((len(t_pairs),), dtype=np.int64)
        sign = np.ones((len(t_pairs),), dtype=np.float64)
        for j, (a_t, b_t) in enumerate(t_pairs):
            a_a = int(pv[a_t])
            b_a = int(pv[b_t])
            s = float(sv[a_t] * sv[b_t])
            if a_a < b_a:
                a_a, b_a = b_a, a_a
            perm[j] = int(idx_a[(a_a, b_a)])
            sign[j] = s
        return perm, sign
    if c == 9 and nssh > 1:
        a_pairs = _asym_pairs_asuka(nssh)
        t_pairs = _asym_pairs_mol(nssh) if use_mol_tgt else _asym_pairs_asuka(nssh)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        perm = np.empty((len(t_pairs),), dtype=np.int64)
        sign = np.ones((len(t_pairs),), dtype=np.float64)
        for j, (a_t, b_t) in enumerate(t_pairs):
            a_a = int(pv[a_t])
            b_a = int(pv[b_t])
            s = float(sv[a_t] * sv[b_t])
            if a_a < b_a:
                a_a, b_a = b_a, a_a
                s *= -1.0
            perm[j] = int(idx_a[(a_a, b_a)])
            sign[j] = s
        return perm, sign

    # Mixed inactive/virtual cases.
    if c == 5 and nish > 0 and nssh > 0:
        perm = np.empty((nssh * nish,), dtype=np.int64)
        sign = np.ones((nssh * nish,), dtype=np.float64)
        for a_m in range(nssh):
            for i_m in range(nish):
                new = int(a_m * nish + i_m)
                old = int(pv[a_m] * nish + pi[i_m])
                perm[new] = old
                sign[new] = float(sv[a_m] * si[i_m])
        return perm, sign
    if c == 6 and nish > 0 and nssh > 0:
        a_pairs = _sym_pairs_asuka(nish)
        t_pairs = _sym_pairs_mol(nish) if use_mol_tgt else _sym_pairs_asuka(nish)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        ncol = int(nssh * len(t_pairs))
        perm = np.empty((ncol,), dtype=np.int64)
        sign = np.ones((ncol,), dtype=np.float64)
        for q_t, (i_t, l_t) in enumerate(t_pairs):
            i_a = int(pi[i_t])
            l_a = int(pi[l_t])
            s_ij = float(si[i_t] * si[l_t])
            if i_a < l_a:
                i_a, l_a = l_a, i_a
            q_a = int(idx_a[(i_a, l_a)])
            for a_t in range(nssh):
                new = int(a_t + nssh * q_t)
                old = int(pv[a_t] + nssh * q_a)
                perm[new] = old
                sign[new] = float(sv[a_t] * s_ij)
        return perm, sign
    if c == 7 and nish > 1 and nssh > 0:
        a_pairs = _asym_pairs_asuka(nish)
        t_pairs = _asym_pairs_mol(nish) if use_mol_tgt else _asym_pairs_asuka(nish)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        ncol = int(nssh * len(t_pairs))
        perm = np.empty((ncol,), dtype=np.int64)
        sign = np.ones((ncol,), dtype=np.float64)
        for q_t, (i_t, l_t) in enumerate(t_pairs):
            i_a = int(pi[i_t])
            l_a = int(pi[l_t])
            s_ij = float(si[i_t] * si[l_t])
            if i_a < l_a:
                i_a, l_a = l_a, i_a
                s_ij *= -1.0
            q_a = int(idx_a[(i_a, l_a)])
            for a_t in range(nssh):
                new = int(a_t + nssh * q_t)
                old = int(pv[a_t] + nssh * q_a)
                perm[new] = old
                sign[new] = float(sv[a_t] * s_ij)
        return perm, sign
    if c == 10 and nish > 0 and nssh > 0:
        a_pairs = _sym_pairs_asuka(nssh)
        t_pairs = _sym_pairs_mol(nssh) if use_mol_tgt else _sym_pairs_asuka(nssh)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        ncol = int(nish * len(t_pairs))
        perm = np.empty((ncol,), dtype=np.int64)
        sign = np.ones((ncol,), dtype=np.float64)
        for q_t, (a_t, b_t) in enumerate(t_pairs):
            a_a = int(pv[a_t])
            b_a = int(pv[b_t])
            s_ab = float(sv[a_t] * sv[b_t])
            if a_a < b_a:
                a_a, b_a = b_a, a_a
            q_a = int(idx_a[(a_a, b_a)])
            for i_t in range(nish):
                new = int(i_t + nish * q_t)
                old = int(pi[i_t] + nish * q_a)
                perm[new] = old
                sign[new] = float(si[i_t] * s_ab)
        return perm, sign
    if c == 11 and nish > 0 and nssh > 1:
        a_pairs = _asym_pairs_asuka(nssh)
        t_pairs = _asym_pairs_mol(nssh) if use_mol_tgt else _asym_pairs_asuka(nssh)
        idx_a = {pq: i for i, pq in enumerate(a_pairs)}
        ncol = int(nish * len(t_pairs))
        perm = np.empty((ncol,), dtype=np.int64)
        sign = np.ones((ncol,), dtype=np.float64)
        for q_t, (a_t, b_t) in enumerate(t_pairs):
            a_a = int(pv[a_t])
            b_a = int(pv[b_t])
            s_ab = float(sv[a_t] * sv[b_t])
            if a_a < b_a:
                a_a, b_a = b_a, a_a
                s_ab *= -1.0
            q_a = int(idx_a[(a_a, b_a)])
            for i_t in range(nish):
                new = int(i_t + nish * q_t)
                old = int(pi[i_t] + nish * q_a)
                perm[new] = old
                sign[new] = float(si[i_t] * s_ab)
        return perm, sign

    return None


def _collect_case_offdiag_from_breakdown(
    bd: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Collect optional per-case offdiag matrices from ASUKA breakdown.

    Breakdown-provided OFFDIAG is interpreted in ASUKA superindex ordering.
    """
    out: dict[int, np.ndarray] = {}
    for case in range(1, 14):
        keys = (
            f"offdiag_case{case:02d}",
            f"OFFDIAG_case{case:02d}",
            f"offdiag_case{case}",
            f"OFFDIAG_case{case}",
        )
        for k in keys:
            if k not in bd:
                continue
            try:
                out[int(case)] = np.asarray(bd[k], dtype=np.float64)
            except Exception:
                pass
            break
    return out


def _collect_case_offdiag_from_dump_dir(
    dump_dir: Path,
) -> dict[int, np.ndarray]:
    """Collect per-case OFFDIAG matrices from Molcas GRAD_DUMP files.

    Dump-dir OFFDIAG is interpreted in Molcas superindex ordering.
    """
    out: dict[int, np.ndarray] = {}
    for case in range(1, 14):
        p = dump_dir / f"GRAD_DUMP_OFFDIAG_case{case:02d}.dat"
        if not p.exists():
            continue
        try:
            out[int(case)] = _read_molcas_dump_matrix(p)
        except Exception:
            continue
    return out
