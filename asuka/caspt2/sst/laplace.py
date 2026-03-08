"""Laplace-quadrature helpers for future reduced-scaling SST.

The original reduced-scaling SST formulation combines:

  * a Schur-complement / supporting-subspace split of the CASPT2 equations
  * Laplace quadrature to separate energy denominators

so that the expensive pieces can be expressed in terms of MP2-like energy
computations and Fock builds.

ASUKA's initial SST backend does not yet rely on Laplace quadrature (it uses a
straightforward denominator evaluation for the H± sector).  This module
provides a small, self-contained Laplace grid implementation that can be
plugged in later without changing the SST public API.

We provide a practical exponential-sum approximation based on a trapezoidal
rule in the log-time variable:

    1/x = \int_0^\infty e^{-x t} dt
        = \int_{-\infty}^{\infty} e^{u} e^{-x e^{u}} du,   t = e^{u}

Discretizing the u-integral with equally spaced points u_k yields:

    1/x \approx \sum_k w_k e^{-t_k x},

with t_k = exp(u_k) and w_k = h * exp(u_k) (endpoints half-weighted).

This grid is simple and robust for x >= x_min > 0.  It is **not** a minimax
grid and is not guaranteed to be optimal, but it is sufficient for prototyping
and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LaplaceGrid",
    "make_log_trap_laplace_grid",
    "inv_denom_laplace",
]


@dataclass(frozen=True)
class LaplaceGrid:
    """Exponential-sum Laplace grid for 1/x approximation."""

    t: np.ndarray  # (npts,) positive nodes
    w: np.ndarray  # (npts,) positive weights


def make_log_trap_laplace_grid(
    *,
    x_min: float,
    npts: int,
    tol: float = 1e-12,
) -> LaplaceGrid:
    """Build a simple exponential-sum Laplace grid for x >= x_min.

    Parameters
    ----------
    x_min
        Minimum denominator value to be approximated (must be > 0). The
        approximation is intended for x in [x_min, +inf).
    npts
        Number of grid points (>= 2 recommended).
    tol
        Heuristic truncation tolerance used to select the u-range.

    Returns
    -------
    LaplaceGrid
        Nodes ``t`` and weights ``w`` such that 1/x ≈ Σ w_g exp(-t_g x).
    """
    x_min = float(x_min)
    if not np.isfinite(x_min) or x_min <= 0.0:
        raise ValueError("x_min must be finite and > 0")
    npts = int(npts)
    if npts <= 0:
        raise ValueError("npts must be positive")

    tol = float(tol)
    if tol <= 0.0 or tol >= 1.0:
        raise ValueError("tol must be in (0,1)")

    # Truncation heuristics:
    #  u -> -inf (t->0): integrand ~ exp(u), tail ~ exp(u_min)
    #  u -> +inf (t->inf): for worst-case x_min, require x_min * exp(u_max) ~ log(1/tol)
    u_min = np.log(tol)
    u_max = np.log(np.log(1.0 / tol) / x_min)

    if not np.isfinite(u_max):
        raise ValueError("failed to construct Laplace u_max; check x_min/tol")
    if u_max <= u_min:
        # extremely small ranges; fall back to a tiny span
        u_max = u_min + 1.0

    if npts == 1:
        u = np.array([0.5 * (u_min + u_max)], dtype=np.float64)
        h = float(u_max - u_min)
        t = np.exp(u)
        w = np.exp(u) * h
        return LaplaceGrid(t=np.asarray(t, order="C"), w=np.asarray(w, order="C"))

    u = np.linspace(u_min, u_max, npts, dtype=np.float64)
    h = float(u[1] - u[0])

    t = np.exp(u)
    w = np.exp(u) * h
    # trapezoidal endpoint half-weights
    w[0] *= 0.5
    w[-1] *= 0.5

    return LaplaceGrid(t=np.asarray(t, dtype=np.float64, order="C"), w=np.asarray(w, dtype=np.float64, order="C"))


def inv_denom_laplace(x: np.ndarray, grid: LaplaceGrid) -> np.ndarray:
    """Approximate 1/x for x>0 using an exponential-sum Laplace grid."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(grid.t, dtype=np.float64)
    w = np.asarray(grid.w, dtype=np.float64)

    if np.any(x <= 0.0):
        raise ValueError("inv_denom_laplace requires x>0")

    # result = sum_g w_g * exp(-t_g * x)
    # Use broadcasting: (G,1) * (1,N) -> (G,N)
    ex = np.exp(-t[:, None] * x[None, ...])
    return np.tensordot(w, ex, axes=(0, 0))
