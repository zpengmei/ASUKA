"""Sorted sparse vector in the CSF index basis (internalized from qmc.sparse)."""

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
