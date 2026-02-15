from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_DENSE_ACC_CACHE: dict[int, "DenseRowAccumulator"] = {}


@dataclass
class DenseRowAccumulator:
    """Dense scratch accumulator for row-oracle assembly.

    This is a performance-oriented alternative to Python dict accumulation:
    - accumulate into `row[i]` directly (float64[ncsf])
    - track touched indices and materialize a sparse row via `np.unique`

    Intended for the indexed-CSF regime where `ncsf` is not astronomically large.
    """

    row: np.ndarray
    _touched_arrays: list[np.ndarray] = field(default_factory=list)
    _touched_scalars: list[int] = field(default_factory=list)

    @classmethod
    def get_cached(cls, ncsf: int) -> "DenseRowAccumulator":
        ncsf = int(ncsf)
        if ncsf < 1:
            raise ValueError("ncsf must be >= 1")
        acc = _DENSE_ACC_CACHE.get(ncsf)
        if acc is None:
            acc = cls(row=np.zeros(ncsf, dtype=np.float64))
            _DENSE_ACC_CACHE[ncsf] = acc
        return acc

    def add_scalar(self, i: int, v: float) -> None:
        ii = int(i)
        vv = float(v)
        if vv == 0.0:
            return
        self.row[ii] += vv
        self._touched_scalars.append(ii)

    def add_many(self, idx: np.ndarray, val: np.ndarray) -> None:
        idx_i32 = np.asarray(idx, dtype=np.int32).ravel()
        if idx_i32.size == 0:
            return
        val_f64 = np.asarray(val, dtype=np.float64).ravel()
        if val_f64.size != idx_i32.size:
            raise ValueError("idx and val must have the same size")
        np.add.at(self.row, idx_i32, val_f64)
        self._touched_arrays.append(idx_i32)

    def clear(self) -> None:
        """Reset any previously touched entries (for safety after exceptions)."""
        parts = list(self._touched_arrays)
        if self._touched_scalars:
            parts.append(np.asarray(self._touched_scalars, dtype=np.int32))
        if parts:
            idx_all = np.concatenate(parts)
            idx_u = np.unique(idx_all)
            self.row[idx_u] = 0.0
        self._touched_arrays.clear()
        self._touched_scalars.clear()

    def finalize_and_reset(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (i_idx, hij) with `j` first, then reset internal scratch state."""
        j = int(j)

        parts = list(self._touched_arrays)
        if self._touched_scalars:
            parts.append(np.asarray(self._touched_scalars, dtype=np.int32))

        if not parts:
            i_idx_arr = np.asarray([j], dtype=np.int32)
            hij_arr = np.asarray([0.0], dtype=np.float64)
            return i_idx_arr, hij_arr

        idx_all = np.concatenate(parts)
        idx_u = np.unique(idx_all)

        # Keep existing convention: j first, others sorted.
        if idx_u.size and idx_u[0] == j:
            i_idx_arr = idx_u
        else:
            others = idx_u[idx_u != j]
            i_idx_arr = np.concatenate((np.asarray([j], dtype=np.int32), others))

        hij_arr = np.asarray(self.row[i_idx_arr], dtype=np.float64).copy()
        self.row[i_idx_arr] = 0.0
        self._touched_arrays.clear()
        self._touched_scalars.clear()

        return i_idx_arr, hij_arr
