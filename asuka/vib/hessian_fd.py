from __future__ import annotations

"""Finite-difference Cartesian Hessians from analytic gradients.

This module provides a method-agnostic finite-difference Hessian builder:

    H_{ij} = d^2E / (dR_i dR_j)

computed as finite differences of analytic gradients:

    H[:, k] ≈ (g(R + h e_k) - g(R - h e_k)) / (2h)

The goal is to support frequency analysis (harmonic normal modes) for methods
that already have nuclear gradients in ASUKA/PySCF.

Design constraints
------------------
- No import-time dependency on PySCF.
- The core FD routine works with *any* callable that returns gradients.

See also
--------
- :mod:`asuka.vib.frequency` for mass-weighted normal-mode analysis.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


GradFn = Callable[[np.ndarray], np.ndarray | tuple[float, np.ndarray]]


@dataclass(frozen=True)
class HessianFDResult:
    """Result of finite-difference Hessian construction."""

    coords_bohr: np.ndarray
    hessian: np.ndarray
    grad0: np.ndarray
    e0: float | None
    step_bohr: float
    method: str
    symmetrized: bool


def _unpack_grad_return(ret: np.ndarray | tuple[float, np.ndarray]) -> tuple[float | None, np.ndarray]:
    if isinstance(ret, tuple) and len(ret) == 2:
        e, g = ret
        return float(e), np.asarray(g, dtype=np.float64)
    return None, np.asarray(ret, dtype=np.float64)


def fd_cartesian_hessian(
    grad_fn: GradFn,
    coords_bohr: np.ndarray,
    *,
    step_bohr: float = 1e-3,
    method: str = "central",
    symmetrize: bool = True,
    verbose: int = 0,
) -> HessianFDResult:
    """Compute Cartesian Hessian by finite differences of analytic gradients."""

    method_s = str(method).strip().lower()
    if method_s not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    coords0 = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords0.shape[0])
    n = 3 * natm
    h = float(step_bohr)
    if h <= 0.0:
        raise ValueError("step_bohr must be positive")

    e0, g0 = _unpack_grad_return(grad_fn(coords0))
    g0 = np.asarray(g0, dtype=np.float64).reshape((natm, 3))

    H = np.zeros((n, n), dtype=np.float64)

    g0_flat = g0.reshape((n,))

    for k in range(n):
        ia = k // 3
        xyz = k % 3
        if verbose:
            print(f"FD Hessian column {k+1}/{n} (atom={ia}, xyz={xyz})")

        if method_s == "central":
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, xyz] += h
            coords_m[ia, xyz] -= h

            _ep, gp = _unpack_grad_return(grad_fn(coords_p))
            _em, gm = _unpack_grad_return(grad_fn(coords_m))

            gp = np.asarray(gp, dtype=np.float64).reshape((n,))
            gm = np.asarray(gm, dtype=np.float64).reshape((n,))
            H[:, k] = (gp - gm) / (2.0 * h)
        else:
            coords_p = coords0.copy()
            coords_p[ia, xyz] += h

            _ep, gp = _unpack_grad_return(grad_fn(coords_p))
            gp = np.asarray(gp, dtype=np.float64).reshape((n,))
            H[:, k] = (gp - g0_flat) / h

    if bool(symmetrize):
        H = 0.5 * (H + H.T)

    return HessianFDResult(
        coords_bohr=np.asarray(coords0, dtype=np.float64),
        hessian=np.asarray(H, dtype=np.float64),
        grad0=np.asarray(g0, dtype=np.float64),
        e0=None if e0 is None else float(e0),
        step_bohr=float(h),
        method=str(method_s),
        symmetrized=bool(symmetrize),
    )


__all__ = [
    "HessianFDResult",
    "fd_cartesian_hessian",
]

