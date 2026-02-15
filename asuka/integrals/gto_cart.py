from __future__ import annotations

"""Gaussian primitive normalization helpers (Cartesian, libcint conventions).

These utilities are used by cuGUGA's various
"basis packers" that construct cuERI packed basis objects.

Notes
-----
cuERI's Step-2 (general-l) basis containers assume *unnormalized* primitives
exp(-a r^2) with coefficients that already include the primitive normalization
factors used by PySCF/libcint for `cart=True`.
"""

from math import gamma, pi

import numpy as np


def ncart(l: int) -> int:
    """Number of Cartesian components for angular momentum l."""

    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    return (l + 1) * (l + 2) // 2


def _gaussian_int(n: int, alpha: np.ndarray) -> np.ndarray:
    """Compute ∫_0^∞ x^n exp(-alpha x^2) dx for vector alpha (float64)."""

    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1:
        raise ValueError("alpha must be 1D")
    n1 = 0.5 * float(n + 1)
    return (gamma(n1) / 2.0) / np.power(alpha, n1)


def gto_norm_radial(l: int, exp: np.ndarray) -> np.ndarray:
    """Radial normalization matching PySCF's `gto_norm(l, exp)` for l>=2."""

    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    exp = np.asarray(exp, dtype=np.float64)
    if exp.ndim != 1:
        raise ValueError("exp must be 1D")
    return 1.0 / np.sqrt(_gaussian_int(l * 2 + 2, 2.0 * exp))


def primitive_norm_cart_like_pyscf(l: int, exp: np.ndarray) -> np.ndarray:
    """Primitive coefficient scale matching PySCF/libcint cart=True conventions."""

    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    exp = np.asarray(exp, dtype=np.float64)
    if exp.ndim != 1:
        raise ValueError("exp must be 1D")
    if l <= 1:
        # N_l = (2a/pi)^(3/4) * (4a)^(l/2)
        return (2.0 * exp / pi) ** 0.75 * (4.0 * exp) ** (0.5 * l)
    return gto_norm_radial(l, exp)


__all__ = ["gto_norm_radial", "ncart", "primitive_norm_cart_like_pyscf"]

