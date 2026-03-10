from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .compress import compress_phi_pivot_resample
from .sparse import coalesce_coo_i32_f64


_INT32_MAX = int(np.iinfo(np.int32).max)
_INT32_MIN = int(np.iinfo(np.int32).min)


def _coerce_i32_sparse_labels(idx: np.ndarray, *, caller: str) -> np.ndarray:
    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"{caller} indices must have an integer dtype")
    if idx_arr.size > 0:
        if idx_arr.dtype.kind == "u":
            if int(np.max(idx_arr)) > _INT32_MAX:
                raise NotImplementedError(f"{caller} only supports int32-addressable labels")
        else:
            if int(np.min(idx_arr)) < _INT32_MIN or int(np.max(idx_arr)) > _INT32_MAX:
                raise NotImplementedError(f"{caller} only supports int32-addressable labels")
    return np.asarray(idx_arr, dtype=np.int32).ravel()


def dot_sparse(
    a_idx: np.ndarray,
    a_val: np.ndarray,
    b_idx: np.ndarray,
    b_val: np.ndarray,
) -> float:
    """Dot product of two sorted sparse vectors."""

    a_idx = _coerce_i32_sparse_labels(a_idx, caller="dot_sparse")
    a_val = np.asarray(a_val, dtype=np.float64).ravel()
    b_idx = _coerce_i32_sparse_labels(b_idx, caller="dot_sparse")
    b_val = np.asarray(b_val, dtype=np.float64).ravel()
    if a_idx.size != a_val.size or b_idx.size != b_val.size:
        raise ValueError("idx and val must have matching sizes")

    i = 0
    j = 0
    acc = 0.0
    na = int(a_idx.size)
    nb = int(b_idx.size)
    while i < na and j < nb:
        ai = int(a_idx[i])
        bj = int(b_idx[j])
        if ai == bj:
            acc += float(a_val[i]) * float(b_val[j])
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return float(acc)


def axpy_sparse(
    x_idx: np.ndarray,
    x_val: np.ndarray,
    y_idx: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y + alpha * x as a coalesced sparse vector."""

    x_idx = _coerce_i32_sparse_labels(x_idx, caller="axpy_sparse")
    x_val = np.asarray(x_val, dtype=np.float64).ravel()
    y_idx = _coerce_i32_sparse_labels(y_idx, caller="axpy_sparse")
    y_val = np.asarray(y_val, dtype=np.float64).ravel()
    if x_idx.size != x_val.size or y_idx.size != y_val.size:
        raise ValueError("idx and val must have matching sizes")

    if float(alpha) == 0.0 or x_idx.size == 0:
        return np.ascontiguousarray(y_idx), np.ascontiguousarray(y_val)
    if y_idx.size == 0:
        return np.ascontiguousarray(x_idx), np.ascontiguousarray(float(alpha) * x_val)

    idx = np.concatenate((y_idx, x_idx))
    val = np.concatenate((y_val, float(alpha) * x_val))
    return coalesce_coo_i32_f64(idx, val)


def normalize_sparse(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize a sparse vector to unit 2-norm."""

    idx = _coerce_i32_sparse_labels(idx, caller="normalize_sparse")
    val = np.asarray(val, dtype=np.float64).ravel()
    if idx.size != val.size:
        raise ValueError("idx and val must have the same size")
    n2 = float(np.dot(val, val))
    n = float(np.sqrt(n2))
    if n <= float(eps):
        raise ValueError("vector norm is too small to normalize")
    return np.ascontiguousarray(idx), np.ascontiguousarray(val / n), n


def orthonormalize_mgs(
    cols: list[tuple[np.ndarray, np.ndarray]],
    *,
    m: int,
    pivot: int,
    rng: np.random.Generator,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Modified Gram-Schmidt orthonormalization with Φ compression after updates.

    Parameters
    ----------
    compressor
        Optional custom compression operator. If a sequence is given, the k-th
        compressor is used for the k-th column.
    """

    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (idx_k, val_k) in enumerate(cols):
        idx_k, val_k = coalesce_coo_i32_f64(idx_k, val_k)
        if idx_k.size == 0:
            raise ValueError(f"column {k} is empty")

        for i, (idx_i, val_i) in enumerate(out):
            ov = dot_sparse(idx_i, val_i, idx_k, val_k)
            if ov != 0.0:
                idx_k, val_k = axpy_sparse(idx_i, val_i, idx_k, val_k, alpha=-ov)

                comp = compressor
                if isinstance(compressor, (list, tuple)):
                    if k >= len(compressor):
                        raise ValueError("compressor sequence is shorter than number of columns")
                    comp = compressor[k]

                if comp is None:
                    idx_k, val_k = compress_phi_pivot_resample(idx_k, val_k, m=m, pivot=int(pivot), rng=rng)
                else:
                    idx_k, val_k = comp(idx_k, val_k, m=m, pivot=int(pivot), rng=rng)
                if idx_k.size == 0:
                    raise ValueError(f"column {k} annihilated during orthonormalization (proj on {i})")

        idx_k, val_k, _n = normalize_sparse(idx_k, val_k)
        out.append((idx_k, val_k))

    return out


def apply_right_matrix(
    cols: list[tuple[np.ndarray, np.ndarray]],
    mat: np.ndarray,
    *,
    m: int,
    pivot: int,
    rng: np.random.Generator,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return new columns `Y = X @ mat` with Φ compression to budget `m` per column.

    Parameters
    ----------
    compressor
        Optional custom compression operator. If a sequence is given, the k-th
        compressor is used for the output column k.
    """

    mat = np.asarray(mat, dtype=np.float64)
    n = int(len(cols))
    if mat.shape != (n, n):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(n, n)})")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n):
        idx_chunks: list[np.ndarray] = []
        val_chunks: list[np.ndarray] = []
        for j in range(n):
            w = float(mat[j, k])
            if w == 0.0:
                continue
            idx_j, val_j = cols[j]
            if idx_j.size == 0:
                continue
            idx_chunks.append(_coerce_i32_sparse_labels(idx_j, caller="apply_right_matrix"))
            val_chunks.append(w * np.asarray(val_j, dtype=np.float64, order="C"))

        if not idx_chunks:
            raise ValueError("matrix application produced an empty column")
        idx = np.concatenate(idx_chunks)
        val = np.concatenate(val_chunks)
        idx, val = coalesce_coo_i32_f64(idx, val)

        comp = compressor
        if isinstance(compressor, (list, tuple)):
            if k >= len(compressor):
                raise ValueError("compressor sequence is shorter than number of columns")
            comp = compressor[k]

        if comp is None:
            idx, val = compress_phi_pivot_resample(idx, val, m=int(m), pivot=int(pivot), rng=rng)
        else:
            idx, val = comp(idx, val, m=int(m), pivot=int(pivot), rng=rng)
        out.append(normalize_sparse(idx, val)[0:2])

    return out
