from __future__ import annotations

"""Internal-coordinate constraints for relaxed scans.

This module provides *single-constraint* objects that can be passed to
:func:`asuka.geomopt.optimizer.optimize_cartesian` to perform constrained
minimum optimization (projected L-BFGS) and to drive 1D relaxed scans.

Only two constraint types are implemented for now:

- **distance** between atoms (i, j)
- **angle** between atoms (i, j, k) (angle at j)

Coordinates are in **Bohr**. Distance values are in **Bohr**. Angle values are
in **radians**.

Indexing convention
-------------------
Atom indices are **0-based** (Python convention).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class InternalCoordinateConstraint(Protocol):
    """Protocol for a scalar internal-coordinate constraint q(R)."""

    def value(self, coords_bohr: np.ndarray) -> float:
        """Return q(coords)."""

    def jacobian(self, coords_bohr: np.ndarray) -> np.ndarray:
        """Return dq/dR as an array of shape (natm,3)."""

    def project(
        self,
        coords_bohr: np.ndarray,
        target: float,
        *,
        tol: float = 1.0e-10,
        max_iter: int = 25,
        eps: float = 1.0e-20,
        max_corr_bohr: float | None = None,
    ) -> np.ndarray:
        """Project coords onto q(coords)=target using a scalar Newton step.

        The projection uses the first-order model:

            q(x + t dq) ≈ q(x) + t (dq·dq)

        and chooses:

            t = (q(x) - target) / (dq·dq)

        Parameters
        ----------
        coords_bohr
            Geometry, shape (natm,3).
        target
            Desired internal coordinate value (Bohr for distance, rad for angle).
        tol
            Absolute tolerance on |q-target|.
        max_iter
            Maximum Newton projection iterations.
        eps
            Small number to protect divisions.
        max_corr_bohr
            Optional cap on the *Cartesian* correction RMS per iteration.
            If provided, the Newton step is scaled down to respect this cap.

        Returns
        -------
        coords_proj
            Projected geometry.
        """


def _as_coords(coords_bohr: np.ndarray) -> np.ndarray:
    c = np.asarray(coords_bohr, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (natm,3)")
    return c


def _project_newton_scalar(
    constraint: InternalCoordinateConstraint,
    coords_bohr: np.ndarray,
    target: float,
    *,
    tol: float,
    max_iter: int,
    eps: float,
    max_corr_bohr: float | None,
) -> np.ndarray:
    coords = _as_coords(coords_bohr).copy()
    natm = int(coords.shape[0])

    for _it in range(int(max_iter)):
        q = float(constraint.value(coords))
        dq = np.asarray(constraint.jacobian(coords), dtype=np.float64)
        if dq.shape != (natm, 3):
            raise ValueError("constraint.jacobian must return shape (natm,3)")

        resid = q - float(target)
        if abs(resid) <= float(tol):
            return coords

        dq_vec = dq.reshape(-1)
        denom = float(np.dot(dq_vec, dq_vec))
        if denom <= float(eps):
            return coords

        step = dq_vec * (resid / denom)

        if max_corr_bohr is not None:
            rms = float(np.sqrt(np.mean(step * step))) if step.size else 0.0
            if rms > float(max_corr_bohr) > 0.0:
                step = step * (float(max_corr_bohr) / rms)

        coords = (coords.reshape(-1) - step).reshape((natm, 3))

    return coords


@dataclass(frozen=True)
class DistanceConstraint:
    """Distance constraint between atoms i and j."""

    i: int
    j: int

    def value(self, coords_bohr: np.ndarray) -> float:
        coords = _as_coords(coords_bohr)
        natm = int(coords.shape[0])
        i = int(self.i)
        j = int(self.j)
        if not (0 <= i < natm and 0 <= j < natm):
            raise IndexError("atom index out of range")
        if i == j:
            raise ValueError("distance constraint requires i != j")
        d = coords[i] - coords[j]
        r = float(np.linalg.norm(d))
        if r <= 0.0:
            raise ValueError("coincident atoms in distance constraint")
        return r

    def jacobian(self, coords_bohr: np.ndarray) -> np.ndarray:
        coords = _as_coords(coords_bohr)
        natm = int(coords.shape[0])
        i = int(self.i)
        j = int(self.j)
        d = coords[i] - coords[j]
        r = float(np.linalg.norm(d))
        if r <= 0.0:
            raise ValueError("coincident atoms in distance constraint")

        dq = np.zeros((natm, 3), dtype=np.float64)
        dq[i] = d / r
        dq[j] = -d / r
        return dq

    def project(
        self,
        coords_bohr: np.ndarray,
        target: float,
        *,
        tol: float = 1.0e-10,
        max_iter: int = 25,
        eps: float = 1.0e-20,
        max_corr_bohr: float | None = None,
    ) -> np.ndarray:
        return _project_newton_scalar(
            self,
            coords_bohr,
            float(target),
            tol=float(tol),
            max_iter=int(max_iter),
            eps=float(eps),
            max_corr_bohr=max_corr_bohr,
        )


@dataclass(frozen=True)
class AngleConstraint:
    """Angle constraint for atoms (i, j, k), i-j-k with the angle at j."""

    i: int
    j: int
    k: int

    def value(self, coords_bohr: np.ndarray) -> float:
        coords = _as_coords(coords_bohr)
        natm = int(coords.shape[0])
        i = int(self.i)
        j = int(self.j)
        k = int(self.k)
        if not (0 <= i < natm and 0 <= j < natm and 0 <= k < natm):
            raise IndexError("atom index out of range")
        if len({i, j, k}) != 3:
            raise ValueError("angle constraint requires distinct i, j, k")

        u = coords[i] - coords[j]
        v = coords[k] - coords[j]
        ru = float(np.linalg.norm(u))
        rv = float(np.linalg.norm(v))
        if ru <= 0.0 or rv <= 0.0:
            raise ValueError("degenerate angle constraint (coincident atoms)")

        c = float(np.dot(u, v) / (ru * rv))
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.arccos(c))

    def jacobian(self, coords_bohr: np.ndarray) -> np.ndarray:
        coords = _as_coords(coords_bohr)
        natm = int(coords.shape[0])
        i = int(self.i)
        j = int(self.j)
        k = int(self.k)

        u = coords[i] - coords[j]
        v = coords[k] - coords[j]
        ru = float(np.linalg.norm(u))
        rv = float(np.linalg.norm(v))
        if ru <= 0.0 or rv <= 0.0:
            raise ValueError("degenerate angle constraint (coincident atoms)")

        c = float(np.dot(u, v) / (ru * rv))
        c = float(np.clip(c, -1.0, 1.0))
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        s = max(s, 1.0e-12)

        dc_du = (v / (ru * rv)) - (c * u / (ru * ru))
        dc_dv = (u / (ru * rv)) - (c * v / (rv * rv))

        dtheta_du = -(1.0 / s) * dc_du
        dtheta_dv = -(1.0 / s) * dc_dv

        dq = np.zeros((natm, 3), dtype=np.float64)
        dq[i] = dtheta_du
        dq[k] = dtheta_dv
        dq[j] = -(dtheta_du + dtheta_dv)
        return dq

    def project(
        self,
        coords_bohr: np.ndarray,
        target: float,
        *,
        tol: float = 1.0e-10,
        max_iter: int = 25,
        eps: float = 1.0e-20,
        max_corr_bohr: float | None = None,
    ) -> np.ndarray:
        if max_corr_bohr is None:
            max_corr_bohr = 0.10

        return _project_newton_scalar(
            self,
            coords_bohr,
            float(target),
            tol=float(tol),
            max_iter=int(max_iter),
            eps=float(eps),
            max_corr_bohr=max_corr_bohr,
        )


__all__ = [
    "InternalCoordinateConstraint",
    "DistanceConstraint",
    "AngleConstraint",
]
