from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def coalesce_coo_i32_f64(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce COO pairs by index: sort by idx, sum duplicates.

    Returns
    -------
    idx_u, val_u
        Sorted unique indices (int32) and summed values (float64).
    """

    idx_i32 = np.asarray(idx, dtype=np.int32).ravel()
    val_f64 = np.asarray(val, dtype=np.float64).ravel()
    if idx_i32.size != val_f64.size:
        raise ValueError("idx and val must have the same size")
    if idx_i32.size <= 1:
        return np.ascontiguousarray(idx_i32), np.ascontiguousarray(val_f64)

    order = np.argsort(idx_i32, kind="stable")
    idx_s = idx_i32[order]
    val_s = val_f64[order]

    change = np.nonzero(idx_s[1:] != idx_s[:-1])[0] + 1
    if change.size == 0:
        return idx_s[:1].astype(np.int32, copy=False), np.asarray([float(val_s.sum())], dtype=np.float64)

    starts = np.concatenate(([0], change)).astype(np.int32, copy=False)
    idx_u = idx_s[starts]
    val_u = np.add.reduceat(val_s, starts)
    return (
        np.asarray(idx_u, dtype=np.int32, order="C"),
        np.asarray(val_u, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SparseVector:
    """Sorted sparse vector in the CSF index basis."""

    idx: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        idx = np.asarray(self.idx, dtype=np.int32).ravel()
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

