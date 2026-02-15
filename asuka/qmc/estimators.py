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

    k = int(np.argmax(np.abs(x_val_f64)))
    return int(x_idx_i32[k])


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
        mask = (pos < int(x_idx.size)) & (x_idx[pos] == i_idx)
        if not np.any(mask):
            continue
        row_dot_x = float(np.dot(hij[mask], x_val[pos[mask]]))
        num += float(xj) * row_dot_x

    return float(num / den), float(num), float(den)
