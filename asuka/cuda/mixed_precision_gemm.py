"""Low-level Ozaki splitting and refined GEMM building blocks.

These are the foundational mixed-precision GEMM primitives that the
higher-level mixed_precision module builds upon.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_xp(*arrays: Any):
    try:
        import cupy as cp
    except Exception:
        cp = None
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp, True
    return np, False


def ozaki_split_3(A_fp64: Any) -> tuple[Any, Any, Any]:
    """Split FP64 matrix into 3 FP32 components for higher accuracy.

    A = A_hi + A_mid + A_lo

    Each component is exactly representable in FP32. When used with
    pedantic FP32 GEMM (9 cross-products), gives near-FP64 accuracy.

    Parameters
    ----------
    A_fp64 : ndarray
        FP64 matrix.

    Returns
    -------
    (A_hi, A_mid, A_lo) : tuple of FP32 ndarrays
    """
    xp, _ = _get_xp(A_fp64)
    A_fp64 = xp.asarray(A_fp64, dtype=xp.float64)

    A_hi = A_fp64.astype(xp.float32)
    residual_1 = A_fp64 - A_hi.astype(xp.float64)
    A_mid = residual_1.astype(xp.float32)
    A_lo = (residual_1 - A_mid.astype(xp.float64)).astype(xp.float32)

    return A_hi, A_mid, A_lo


def gemm_ozaki3(
    A: Any,
    B: Any,
    *,
    out: Any | None = None,
) -> Any:
    """3-way Ozaki GEMM: 9 pedantic FP32 GEMMs for near-FP64 accuracy.

    For the highest-sensitivity operations where 2-way Ozaki is insufficient.

    Cost: 9 FP32 GEMMs with FP64 external accumulation.
    Accuracy: near-FP64 (~69-bit effective mantissa from 3x23-bit FP32 splits).
    """
    xp, is_gpu = _get_xp(A, B)

    if not is_gpu:
        C = np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)
        if out is not None:
            out[...] = C
            return out
        return C

    from asuka.cuda.mixed_precision import _pedantic_math_mode

    A = xp.asarray(A, dtype=xp.float64)
    B = xp.asarray(B, dtype=xp.float64)

    A_hi, A_mid, A_lo = ozaki_split_3(A)
    B_hi, B_mid, B_lo = ozaki_split_3(B)

    m, k = A.shape
    n = B.shape[1]

    if out is None:
        C = xp.zeros((m, n), dtype=xp.float64)
    else:
        C = out
        C[...] = 0.0

    with _pedantic_math_mode(xp):
        for Ai in (A_hi, A_mid, A_lo):
            for Bj in (B_hi, B_mid, B_lo):
                C += (Ai @ Bj).astype(xp.float64)

    return C


def syrk_ozaki2(
    A: Any,
    *,
    out: Any | None = None,
) -> Any:
    """Ozaki-2 symmetric rank-k update: C = A @ A.T with ~1e-7 accuracy.

    Exploits symmetry: A_lo @ A_hi.T = (A_hi @ A_lo.T).T, so we compute
    the cross-term once and add its transpose.

    Cost: 3 FP32 GEMMs (vs 4 for asymmetric Ozaki-2).
    """
    xp, is_gpu = _get_xp(A)

    if not is_gpu:
        A = np.asarray(A, dtype=np.float64)
        C = A @ A.T
        if out is not None:
            out[...] = C
            return out
        return C

    from asuka.cuda.mixed_precision import ozaki_split_2, _pedantic_math_mode

    A = xp.asarray(A, dtype=xp.float64)
    A_hi, A_lo = ozaki_split_2(A)

    m = A.shape[0]
    if out is None:
        C = xp.zeros((m, m), dtype=xp.float64)
    else:
        C = out
        C[...] = 0.0

    with _pedantic_math_mode(xp):
        # A_hi @ A_hi.T
        C += (A_hi @ A_hi.T).astype(xp.float64)
        # 2 * A_hi @ A_lo.T (symmetry: A_lo @ A_hi.T = (A_hi @ A_lo.T).T)
        cross = (A_hi @ A_lo.T).astype(xp.float64)
        C += cross + cross.T
        # A_lo @ A_lo.T
        C += (A_lo @ A_lo.T).astype(xp.float64)

    return C


def gemv_tf32_refined(
    A: Any,
    x: Any,
    *,
    out: Any | None = None,
) -> Any:
    """FP32-data GEMV with pedantic math for ~1e-7 accuracy.

    For matrix-vector products (e.g., DF J projection).

    Parameters
    ----------
    A : ndarray
        Matrix (m, k), float64.
    x : ndarray
        Vector (k,), float64.

    Returns
    -------
    ndarray
        y[m] = A @ x with ~1e-7 accuracy.
    """
    xp, is_gpu = _get_xp(A, x)

    if not is_gpu:
        y = np.asarray(A, dtype=np.float64) @ np.asarray(x, dtype=np.float64)
        if out is not None:
            out[...] = y
            return out
        return y

    from asuka.cuda.mixed_precision import _pedantic_math_mode

    A = xp.asarray(A, dtype=xp.float64)
    x = xp.asarray(x, dtype=xp.float64).ravel()

    A_f32 = A.astype(xp.float32)
    x_f32 = x.astype(xp.float32)

    with _pedantic_math_mode(xp):
        result = (A_f32 @ x_f32).astype(xp.float64)

    if out is not None:
        out[...] = result
        return out
    return result


__all__ = [
    "ozaki_split_3",
    "gemm_ozaki3",
    "syrk_ozaki2",
    "gemv_tf32_refined",
]
