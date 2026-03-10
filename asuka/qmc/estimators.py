from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache
from asuka.cuguga.oracle.sparse import connected_row_sparse


def _get_sparse_value(idx: np.ndarray, val: np.ndarray, key: int) -> float:
    idx = np.asarray(idx, dtype=np.int32).ravel()
    val = np.asarray(val, dtype=np.float64).ravel()
    if idx.size != val.size:
        raise ValueError("idx and val must have the same size")
    key = int(key)
    pos = int(np.searchsorted(idx, key))
    if pos < int(idx.size) and int(idx[pos]) == key:
        return float(val[pos])
    return 0.0


def choose_reference_index(
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    preferred: int | None = None,
    allow_fallback: bool = True,
) -> int:
    """Choose a reference CSF index for projected estimators.

    Preference order:
    1) `preferred` if present with nonzero amplitude
    2) index of maximum |x|
    """

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")
    if x_idx_i32.size == 0:
        raise ValueError("x is empty")

    if preferred is not None:
        x_ref = _get_sparse_value(x_idx_i32, x_val_f64, int(preferred))
        if x_ref != 0.0:
            return int(preferred)
        if not bool(allow_fallback):
            raise ValueError("preferred reference amplitude is zero")

    k = int(np.argmax(np.abs(x_val_f64)))
    return int(x_idx_i32[k])


def sparse_dot_sorted(
    idx_a: np.ndarray,
    val_a: np.ndarray,
    idx_b: np.ndarray,
    val_b: np.ndarray,
) -> float:
    """Sparse dot product for sorted COO vectors."""

    idx_a_i32 = np.asarray(idx_a, dtype=np.int32).ravel()
    val_a_f64 = np.asarray(val_a, dtype=np.float64).ravel()
    idx_b_i32 = np.asarray(idx_b, dtype=np.int32).ravel()
    val_b_f64 = np.asarray(val_b, dtype=np.float64).ravel()
    if idx_a_i32.size != val_a_f64.size or idx_b_i32.size != val_b_f64.size:
        raise ValueError("idx and val arrays must have matching sizes")

    ia = 0
    ib = 0
    na = int(idx_a_i32.size)
    nb = int(idx_b_i32.size)
    dot = 0.0
    while ia < na and ib < nb:
        a = int(idx_a_i32[ia])
        b = int(idx_b_i32[ib])
        if a == b:
            dot += float(val_a_f64[ia]) * float(val_b_f64[ib])
            ia += 1
            ib += 1
        elif a < b:
            ia += 1
        else:
            ib += 1
    return float(dot)


def sparse_abs_l1_on_support(
    x_idx: np.ndarray,
    x_val: np.ndarray,
    support_idx: np.ndarray,
) -> float:
    """L1 mass of ``x`` on the provided sorted support indices."""

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    support_i32 = np.asarray(support_idx, dtype=np.int32).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")
    if support_i32.size == 0 or x_idx_i32.size == 0:
        return 0.0

    pos = np.searchsorted(x_idx_i32, support_i32)
    in_range = pos < int(x_idx_i32.size)
    if not np.any(in_range):
        return 0.0
    pos2 = pos[in_range]
    hit = x_idx_i32[pos2] == support_i32[in_range]
    if not np.any(hit):
        return 0.0
    return float(np.sum(np.abs(x_val_f64[pos2[hit]]), dtype=np.float64))


def projected_energy_ref_status(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    ref_idx: int,
    max_out: int = 200_000,
    state_cache: DRTStateCache | None = None,
) -> tuple[float, float, float, bool]:
    """Projected energy with a liveness flag for strict fixed-reference evaluation."""

    ref_idx_i32 = int(ref_idx)
    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")

    den = _get_sparse_value(x_idx_i32, x_val_f64, ref_idx_i32)
    if den == 0.0:
        return float("nan"), 0.0, 0.0, False

    e, num, den2 = projected_energy_ref(
        drt,
        h1e,
        eri,
        x_idx_i32,
        x_val_f64,
        ref_idx=ref_idx_i32,
        max_out=int(max_out),
        state_cache=state_cache,
    )
    return float(e), float(num), float(den2), True


def projected_energy_ref(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    ref_idx: int,
    max_out: int = 200_000,
    state_cache: DRTStateCache | None = None,
) -> tuple[float, float, float]:
    """Projected energy E = (⟨ref|H|x⟩) / (⟨ref|x⟩) using a deterministic row oracle."""

    ref_idx = int(ref_idx)
    x_idx = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx.size != x_val.size:
        raise ValueError("x_idx and x_val must have the same size")

    x_ref = _get_sparse_value(x_idx, x_val, ref_idx)
    if x_ref == 0.0:
        raise ValueError("reference amplitude is zero (choose a different ref_idx)")

    i_idx, hij = connected_row_sparse(drt, h1e, eri, ref_idx, max_out=int(max_out), state_cache=state_cache)

    # Note: `connected_row_sparse` is not guaranteed to return a globally sorted row
    # (e.g. diagonal insertion at a fixed position). Use `searchsorted` against
    # the (sorted) sparse support of x.
    pos = np.searchsorted(x_idx, i_idx)
    in_range = pos < int(x_idx.size)
    if not np.any(in_range):
        num = 0.0
    else:
        pos2 = pos[in_range]
        i2 = i_idx[in_range]
        hit = x_idx[pos2] == i2
        if not np.any(hit):
            num = 0.0
        else:
            pos3 = pos2[hit]
            hij3 = hij[in_range][hit]
            num = float(np.dot(hij3, x_val[pos3]))
    den = float(x_ref)
    return float(num / den), float(num), float(den)


def rayleigh_energy_ref(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    max_out: int = 200_000,
    state_cache: DRTStateCache | None = None,
) -> tuple[float, float, float]:
    """Rayleigh quotient E = (x^T H x) / (x^T x) using deterministic row oracles.

    Notes
    -----
    - This is an O(nnz(x)) row-oracle evaluation and is intended for validation
      and debugging (not the production estimator for large problems).
    - The oracle returns a (mostly) sorted row with the diagonal inserted at
      position 0; we therefore use `searchsorted`-based sparse dot products
      rather than relying on a strict merge walk.
    """

    x_idx = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx.size != x_val.size:
        raise ValueError("x_idx and x_val must have the same size")
    if x_idx.size == 0:
        raise ValueError("x is empty")

    den = float(np.dot(x_val, x_val))
    if den == 0.0:
        raise ValueError("x has zero norm")

    num = 0.0
    for j, xj in zip(x_idx.tolist(), x_val.tolist()):
        if xj == 0.0:
            continue
        i_idx, hij = connected_row_sparse(
            drt,
            h1e,
            eri,
            int(j),
            max_out=int(max_out),
            state_cache=state_cache,
        )
        if i_idx.size == 0:
            continue
        pos = np.searchsorted(x_idx, i_idx)
        in_range = pos < int(x_idx.size)
        if not np.any(in_range):
            continue
        pos2 = pos[in_range]
        i2 = i_idx[in_range]
        hit = x_idx[pos2] == i2
        if not np.any(hit):
            continue
        row_dot_x = float(np.dot(hij[in_range][hit], x_val[pos2[hit]]))
        num += float(xj) * row_dot_x

    return float(num / den), float(num), float(den)
