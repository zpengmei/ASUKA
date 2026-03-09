from __future__ import annotations

"""Self-contained triangular-solve helpers for ASUKA.

These replace ``scipy.linalg.solve_triangular`` and
``cupyx.scipy.linalg.solve_triangular`` throughout the DF (density fitting)
and DF-adjoint code, making those paths free of SciPy/CuPy-SciPy dependencies.

CPU path
--------
Uses ``numpy.linalg.solve`` on small systems and a direct LAPACK ``dtrtrs``
call via ``numpy.linalg.lapack_lite`` (if available) or a pure-NumPy
column-substitution fallback for lower-triangular solves.

GPU path
--------
Wraps cuBLAS ``dtrsm`` directly so there is no ``cupyx.scipy`` import.
"""

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CPU (NumPy) triangular solves
# ---------------------------------------------------------------------------


def _dtrtrs_numpy(L: np.ndarray, B: np.ndarray, *, trans: str) -> np.ndarray:
    """Lower-triangular solve via LAPACK dtrtrs if available, else fallback.

    Solves ``L X = B`` (trans="N") or ``L^T X = B`` (trans="T") where *L* is
    lower-triangular (naux, naux) and *B* is (naux, k).

    Returns a new array *X* with the same shape as *B*.
    """

    L = np.asarray(L, dtype=np.float64, order="C")
    X = np.array(B, dtype=np.float64, order="C", copy=True)
    n = int(L.shape[0])
    if n == 0:
        return X

    # Pure NumPy triangular solve with BLAS-backed dot products.
    #
    # This avoids SciPy dependencies while remaining fast enough for the
    # moderate problem sizes encountered in ASUKA's CPU DF paths.
    if trans == "N":
        # Forward substitution: L X = B
        for i in range(n):
            if i:
                X[i] -= L[i, :i] @ X[:i]
            X[i] /= L[i, i]
        return X

    # Back substitution for L^T: L^T X = B (upper-triangular system)
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            X[i] -= L[i + 1 :, i] @ X[i + 1 :]
        X[i] /= L[i, i]
    return X


def solve_lower_triangular_numpy(
    L: np.ndarray,
    B: np.ndarray,
    *,
    trans: str = "N",
    overwrite_b: bool = False,
) -> np.ndarray:
    """Solve ``L X = B`` or ``L^T X = B`` on CPU with no SciPy dependency.

    Parameters
    ----------
    L : ndarray, shape (n, n)
        Lower-triangular matrix (only the lower triangle is read).
    B : ndarray, shape (n, k) or (n,)
        Right-hand side(s).
    trans : {"N", "T"}
        ``"N"`` for ``L X = B``, ``"T"`` for ``L^T X = B``.
    overwrite_b : bool
        Ignored (kept for API compatibility with scipy).  A new array is
        always returned.

    Returns
    -------
    X : ndarray, same shape as *B*
    """

    L = np.asarray(L, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")
    n = int(L.shape[0])
    if B.ndim == 1:
        if B.shape[0] != n:
            raise ValueError(f"B length {B.shape[0]} != L size {n}")
        return _dtrtrs_numpy(L, B.reshape(n, 1), trans=trans).ravel()
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B shape {B.shape} incompatible with L size {n}")
    return _dtrtrs_numpy(L, B, trans=trans)


# ---------------------------------------------------------------------------
# GPU (CuPy + cuBLAS) triangular solves
# ---------------------------------------------------------------------------


def solve_lower_triangular_cupy(
    L: Any,
    B: Any,
    *,
    trans: str = "N",
) -> Any:
    """Solve ``L X = B`` or ``L^T X = B`` on GPU via cuBLAS dtrsm.

    Parameters
    ----------
    L : cupy.ndarray, shape (n, n)
        Lower-triangular matrix.
    B : cupy.ndarray, shape (n, k)
        Right-hand side(s).  A **new** array is returned (B is not overwritten).
    trans : {"N", "T"}

    Returns
    -------
    X : cupy.ndarray, same shape as *B*
    """

    import cupy as cp  # noqa: PLC0415
    from cupy.cuda import cublas  # noqa: PLC0415

    L = cp.asarray(L, dtype=cp.float64)
    B = cp.asarray(B, dtype=cp.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")
    n = int(L.shape[0])
    squeeze = False
    if B.ndim == 1:
        B = B.reshape(n, 1)
        squeeze = True
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B shape {B.shape} incompatible with L size {n}")
    k = int(B.shape[1])
    if n == 0:
        return B.copy()

    # cuBLAS dtrsm expects column-major (Fortran) layout.
    Lf = cp.asfortranarray(L)
    Xf = cp.asfortranarray(B.copy())

    handle = cp.cuda.get_cublas_handle()
    prev_stream = cublas.getStream(handle)
    prev_mode = cublas.getPointerMode(handle)
    try:
        cublas.setStream(handle, int(cp.cuda.get_current_stream().ptr))
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        alpha = np.array(1.0, dtype=np.float64)
        # dtrsm: op(A) X = alpha B  with A on the LEFT, LOWER, trans flag
        op = cublas.CUBLAS_OP_N if trans == "N" else cublas.CUBLAS_OP_T
        cublas.dtrsm(
            handle,
            cublas.CUBLAS_SIDE_LEFT,      # A on left
            cublas.CUBLAS_FILL_MODE_LOWER, # lower triangular
            op,                            # N or T
            cublas.CUBLAS_DIAG_NON_UNIT,
            n,                             # m (rows)
            k,                             # n (cols of B)
            int(alpha.ctypes.data),
            int(Lf.data.ptr),
            n,                             # leading dim of L
            int(Xf.data.ptr),
            n,                             # leading dim of X
        )
    finally:
        try:
            cublas.setStream(handle, int(prev_stream))
        except Exception:
            pass
        try:
            cublas.setPointerMode(handle, int(prev_mode))
        except Exception:
            pass

    X = cp.ascontiguousarray(Xf)
    if squeeze:
        X = X.ravel()
    return X


# ---------------------------------------------------------------------------
# Dispatching helper (NumPy / CuPy)
# ---------------------------------------------------------------------------


def solve_triangular(
    L: Any,
    B: Any,
    *,
    lower: bool = True,
    trans: str = "N",
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
    check_finite: bool = True,
) -> Any:
    """Drop-in replacement for ``scipy.linalg.solve_triangular``.

    Only **lower-triangular** matrices are supported (``lower=True``).
    Dispatches automatically between CPU (NumPy) and GPU (CuPy) paths.

    Parameters
    ----------
    L, B : array-like
        Triangular matrix and right-hand side(s).
    lower : bool
        Must be True (upper-triangular not implemented).
    trans : {"N", "T"}
    overwrite_b : bool
        Ignored on CPU; on GPU a copy is always made for safety.
    unit_diagonal : bool
        Ignored (kept for API compatibility with SciPy).
    check_finite : bool
        Ignored (kept for API compatibility with SciPy).

    Returns
    -------
    X : same type as *B*
    """

    if not lower:
        raise NotImplementedError("Only lower-triangular solves are supported")

    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(L, cp.ndarray) or isinstance(B, cp.ndarray):
            return solve_lower_triangular_cupy(L, B, trans=trans)
    except ImportError:
        pass

    return solve_lower_triangular_numpy(
        np.asarray(L), np.asarray(B), trans=trans, overwrite_b=overwrite_b
    )


__all__ = [
    "solve_lower_triangular_cupy",
    "solve_lower_triangular_numpy",
    "solve_triangular",
]
