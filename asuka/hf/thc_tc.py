from __future__ import annotations

"""Utilities for THC tensor-core-friendly caches.

This module does not change the represented THC tensor. It only exploits the
exact per-point gauge freedom:

  X' = S X
  Y' = S^{-2} Y

with diagonal S, which leaves the AO ERI approximation invariant while
balancing magnitudes prior to low-precision storage.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _get_xp(*arrays: Any):
    try:
        import cupy as cp  # type: ignore
    except Exception:  # pragma: no cover
        cp = None  # type: ignore
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


@dataclass(frozen=True)
class THCTCCache:
    """Balanced low-precision copies of THC factors."""

    X_tc: Any
    Y_tc: Any
    scale: Any  # (npt,) point scaling used for balancing
    meta: dict[str, Any] | None = None


def balance_thc_xy(
    X: Any,
    Y: Any,
    *,
    eps: float = 1e-30,
    s_min: float = 1e-4,
    s_max: float = 1e4,
):
    """Return balanced (Xb, Yb, s) using the exact THC point-gauge freedom.

    Heuristic:
      s[P] = ( ||Y[P]|| / (||X[P]|| + eps) )^(1/3), with optional clipping.
    """

    xp, _is_gpu = _get_xp(X, Y)
    X = xp.asarray(X)
    Y = xp.asarray(Y)
    if int(getattr(X, "ndim", 0)) != 2 or int(getattr(Y, "ndim", 0)) != 2:
        raise ValueError("X and Y must be 2D arrays")
    if int(X.shape[0]) != int(Y.shape[0]):
        raise ValueError("X/Y npt mismatch")

    # Row 2-norms.
    x_norm = xp.sqrt(xp.sum(X * X, axis=1))
    y_norm = xp.sqrt(xp.sum(Y * Y, axis=1))

    # Balance magnitudes before casting. Any positive scaling is exact.
    denom = x_norm + float(eps)
    s = (y_norm / denom) ** (1.0 / 3.0)
    if s_min is not None or s_max is not None:
        s = xp.clip(s, float(s_min), float(s_max))

    Xb = X * s[:, None]
    Yb = Y / (s[:, None] * s[:, None])
    return Xb, Yb, s


def make_thc_tc_cache(
    thc_factors: Any,
    *,
    compute_dtype=np.float32,
    balance: bool = True,
    eps: float = 1e-30,
    s_min: float = 1e-4,
    s_max: float = 1e4,
) -> THCTCCache:
    """Build a tensor-core-friendly cache from a THCFactors-like object.

    The cache stores balanced X/Y as `compute_dtype` (typically float32). The
    represented THC tensor is unchanged (exact gauge transform).
    """

    X = getattr(thc_factors, "X", None)
    Y = getattr(thc_factors, "Y", None)
    if X is None or Y is None:
        raise ValueError("thc_factors must provide .X and .Y")

    xp, _is_gpu = _get_xp(X, Y)
    if bool(balance):
        Xb, Yb, s = balance_thc_xy(X, Y, eps=float(eps), s_min=float(s_min), s_max=float(s_max))
    else:
        Xb = xp.asarray(X)
        Yb = xp.asarray(Y)
        s = xp.ones((int(Xb.shape[0]),), dtype=getattr(Xb, "dtype", np.float64))

    X_tc = xp.ascontiguousarray(xp.asarray(Xb, dtype=compute_dtype))
    Y_tc = xp.ascontiguousarray(xp.asarray(Yb, dtype=compute_dtype))
    meta = {
        "compute_dtype": str(np.dtype(compute_dtype)),
        "balanced": bool(balance),
        "eps": float(eps),
        "s_min": float(s_min),
        "s_max": float(s_max),
    }
    return THCTCCache(X_tc=X_tc, Y_tc=Y_tc, scale=s, meta=meta)


def make_local_thc_tc_cache(
    lthc: Any,
    *,
    compute_dtype=np.float32,
    balance: bool = True,
    eps: float = 1e-30,
    s_min: float = 1e-4,
    s_max: float = 1e4,
) -> dict[int, THCTCCache]:
    """Build per-block tensor-core caches for a LocalTHCFactors-like object."""

    blocks = getattr(lthc, "blocks", None)
    if blocks is None:
        raise ValueError("lthc must provide .blocks")
    out: dict[int, THCTCCache] = {}
    for blk in blocks:
        bid = int(getattr(blk, "block_id"))
        X = getattr(blk, "X", None)
        Y = getattr(blk, "Y", None)
        if X is None or Y is None:
            raise ValueError("LocalTHCBlock must provide .X and .Y")
        xp, _is_gpu = _get_xp(X, Y)
        if bool(balance):
            Xb, Yb, s = balance_thc_xy(X, Y, eps=float(eps), s_min=float(s_min), s_max=float(s_max))
        else:
            Xb = xp.asarray(X)
            Yb = xp.asarray(Y)
            s = xp.ones((int(Xb.shape[0]),), dtype=getattr(Xb, "dtype", np.float64))

        X_tc = xp.ascontiguousarray(xp.asarray(Xb, dtype=compute_dtype))
        Y_tc = xp.ascontiguousarray(xp.asarray(Yb, dtype=compute_dtype))
        out[int(bid)] = THCTCCache(
            X_tc=X_tc,
            Y_tc=Y_tc,
            scale=s,
            meta={
                "compute_dtype": str(np.dtype(compute_dtype)),
                "balanced": bool(balance),
                "eps": float(eps),
                "s_min": float(s_min),
                "s_max": float(s_max),
            },
        )
    return out


__all__ = [
    "THCTCCache",
    "balance_thc_xy",
    "make_thc_tc_cache",
    "make_local_thc_tc_cache",
]

