from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize_sparse_idx(idx: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = np.asarray(idx).ravel()
    if arr.dtype.kind not in ("i", "u"):
        raise ValueError("idx must have an integer dtype")
    if dtype is None:
        out = np.asarray(arr)
    else:
        out = np.asarray(arr, dtype=dtype)
    return np.ascontiguousarray(out)


def _coalesce_coo_f64(idx: np.ndarray, val: np.ndarray, *, dtype: np.dtype | type) -> tuple[np.ndarray, np.ndarray]:
    idx_u = _normalize_sparse_idx(idx, dtype=dtype)
    val_f64 = np.asarray(val, dtype=np.float64).ravel()
    if idx_u.size != val_f64.size:
        raise ValueError("idx and val must have the same size")
    if idx_u.size <= 1:
        if idx_u.size == 0 or float(val_f64[0]) != 0.0:
            return np.ascontiguousarray(idx_u), np.ascontiguousarray(val_f64)
        return np.zeros(0, dtype=idx_u.dtype), np.zeros(0, dtype=np.float64)

    order = np.argsort(idx_u, kind="stable")
    idx_s = idx_u[order]
    val_s = val_f64[order]

    change = np.nonzero(idx_s[1:] != idx_s[:-1])[0] + 1
    if change.size == 0:
        s = float(val_s.sum())
        if s == 0.0:
            return np.zeros(0, dtype=idx_s.dtype), np.zeros(0, dtype=np.float64)
        return np.asarray(idx_s[:1], dtype=idx_s.dtype), np.asarray([s], dtype=np.float64)

    starts = np.concatenate(([0], change)).astype(np.int64, copy=False)
    idx_out = idx_s[starts]
    val_out = np.add.reduceat(val_s, starts)
    nz = val_out != 0.0
    if not bool(np.any(nz)):
        return np.zeros(0, dtype=idx_s.dtype), np.zeros(0, dtype=np.float64)
    if not bool(np.all(nz)):
        idx_out = idx_out[nz]
        val_out = val_out[nz]
    return np.ascontiguousarray(idx_out), np.asarray(val_out, dtype=np.float64, order="C")


def coalesce_coo_i32_f64(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce COO pairs by index: sort by idx, sum duplicates.

    Returns
    -------
    idx_u, val_u
        Sorted unique indices (int32) and summed values (float64).

    Notes
    -----
    - Exact zeros are pruned from the reduced output. This is important for QMC
      initiator gating, which checks sparse support membership via indices.
    """
    return _coalesce_coo_f64(idx, val, dtype=np.int32)


def coalesce_coo_i64_f64(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce COO pairs by index with int64 labels."""

    return _coalesce_coo_f64(idx, val, dtype=np.int64)


def coalesce_coo_u64_f64(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce COO pairs by index with uint64 labels."""

    return _coalesce_coo_f64(idx, val, dtype=np.uint64)


def coalesce_coo_auto_f64(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    prefer_unsigned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce COO pairs choosing int32/int64/uint64 based on the labels."""

    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind == "u":
        return coalesce_coo_u64_f64(idx_arr, val)
    if idx_arr.dtype.kind != "i":
        raise ValueError("idx must have an integer dtype")
    if idx_arr.size == 0:
        dtype = np.uint64 if prefer_unsigned else np.int64
        return np.zeros(0, dtype=dtype), np.zeros(0, dtype=np.float64)
    if bool(prefer_unsigned):
        if np.any(idx_arr < 0):
            raise ValueError("prefer_unsigned=True requires non-negative idx")
        return coalesce_coo_u64_f64(idx_arr, val)
    if int(np.min(idx_arr)) >= np.iinfo(np.int32).min and int(np.max(idx_arr)) <= np.iinfo(np.int32).max:
        return coalesce_coo_i32_f64(idx_arr, val)
    return coalesce_coo_i64_f64(idx_arr, val)


def sparse_lookup_value(idx: np.ndarray, val: np.ndarray, key: int | np.integer) -> float:
    idx_arr = _normalize_sparse_idx(idx)
    val_arr = np.asarray(val, dtype=np.float64).ravel()
    if idx_arr.size != val_arr.size:
        raise ValueError("idx and val must have the same size")
    if idx_arr.size == 0:
        return 0.0
    key_obj = np.asarray([key], dtype=idx_arr.dtype)[0]
    pos = int(np.searchsorted(idx_arr, key_obj))
    if pos < int(idx_arr.size) and idx_arr[pos] == key_obj:
        return float(val_arr[pos])
    return 0.0


def sparse_dot_sorted(
    idx_a: np.ndarray,
    val_a: np.ndarray,
    idx_b: np.ndarray,
    val_b: np.ndarray,
) -> float:
    idx_a_arr = _normalize_sparse_idx(idx_a)
    idx_b_arr = _normalize_sparse_idx(idx_b, dtype=idx_a_arr.dtype)
    val_a_arr = np.asarray(val_a, dtype=np.float64).ravel()
    val_b_arr = np.asarray(val_b, dtype=np.float64).ravel()
    if idx_a_arr.size != val_a_arr.size or idx_b_arr.size != val_b_arr.size:
        raise ValueError("idx and val arrays must have matching sizes")

    ia = 0
    ib = 0
    na = int(idx_a_arr.size)
    nb = int(idx_b_arr.size)
    dot = 0.0
    while ia < na and ib < nb:
        a = idx_a_arr[ia]
        b = idx_b_arr[ib]
        if a == b:
            dot += float(val_a_arr[ia]) * float(val_b_arr[ib])
            ia += 1
            ib += 1
        elif a < b:
            ia += 1
        else:
            ib += 1
    return float(dot)


def sparse_abs_l1_on_support(x_idx: np.ndarray, x_val: np.ndarray, support_idx: np.ndarray) -> float:
    x_idx_arr = _normalize_sparse_idx(x_idx)
    x_val_arr = np.asarray(x_val, dtype=np.float64).ravel()
    support_arr = _normalize_sparse_idx(support_idx, dtype=x_idx_arr.dtype)
    if x_idx_arr.size != x_val_arr.size:
        raise ValueError("x_idx and x_val must have the same size")
    if support_arr.size == 0 or x_idx_arr.size == 0:
        return 0.0

    pos = np.searchsorted(x_idx_arr, support_arr)
    in_range = pos < int(x_idx_arr.size)
    if not np.any(in_range):
        return 0.0
    pos2 = pos[in_range]
    hit = x_idx_arr[pos2] == support_arr[in_range]
    if not np.any(hit):
        return 0.0
    return float(np.sum(np.abs(x_val_arr[pos2[hit]]), dtype=np.float64))


@dataclass(frozen=True)
class SparseVector:
    """Sorted sparse vector in the CSF index basis."""

    idx: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        idx = _normalize_sparse_idx(self.idx)
        val = np.asarray(self.val, dtype=np.float64).ravel()
        if idx.size != val.size:
            raise ValueError("idx and val must have the same size")
        if idx.size > 1 and np.any(idx[1:] <= idx[:-1]):
            raise ValueError("idx must be strictly increasing (sorted unique)")
        object.__setattr__(self, "idx", np.ascontiguousarray(idx))
        object.__setattr__(self, "val", np.ascontiguousarray(val))

    @property
    def nnz(self) -> int:
        return int(self.idx.size)

    def l1_norm(self) -> float:
        return float(np.sum(np.abs(self.val)))

    def dot_dense(self, dense: np.ndarray) -> float:
        dense = np.asarray(dense, dtype=np.float64).ravel()
        if self.idx.size == 0:
            return 0.0
        if np.any(self.idx < 0) or np.any(self.idx >= dense.size):
            raise ValueError("index out of range for dense vector")
        return float(np.dot(dense[self.idx], self.val))
