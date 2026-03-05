from __future__ import annotations

"""Numerical integration grids for density.

ASUKA does not currently ship a full DFT grid stack, so density includes a
self-contained implementation of:

- atom-centered grids (radial × angular)
- Becke partitioning to form molecular integration weights

This is sufficient for integrating local energy densities of the form

    E = \int f(\rho, \Pi, \nabla\rho, \tau, ...) \, d\mathbf{r}

and for training neural-network functionals against reference energies.

Notes
-----
- Coordinates are in Bohr.
- We use a simple Gauss-Legendre radial grid on [0, rmax]. This is not the
  same as the Treutler-Ahlrichs / Mura-Knowles grids used in many QC codes,
  but it is robust and has no external dependencies.
- For the angular grid we support:
    * Lebedev–Laikov (recommended; limited to specific point counts)
    * Fibonacci sphere (approximate; any point count)
  By default ("auto"), we use Lebedev when available for the requested
  ``angular_n`` and fall back to Fibonacci otherwise.
"""

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from asuka.frontend.molecule import Molecule
from asuka.orbitals.eval_cart import CubeGrid, make_cube_grid_from_atoms

from .lebedev import LEBEDEV_ORDERS, lebedev_sphere


@dataclass(frozen=True)
class GridSpec:
    """Specification for a numerical integration grid."""

    kind: str = "becke"  # 'becke' or 'cube'

    # Becke/atom-centered grid params
    radial_n: int = 50
    angular_n: int = 302
    angular_kind: str = "auto"  # 'auto'|'lebedev'|'fibonacci'
    rmax: float = 20.0
    becke_n: int = 3  # smoothing iterations
    block_size: int = 20000
    prune_tol: float = 1e-16

    # Cube grid params (debug / visualization)
    spacing: float = 0.25
    padding: float = 4.0


def _coords_bohr(mol_or_coords: Molecule | Any) -> np.ndarray:
    if isinstance(mol_or_coords, Molecule):
        return np.asarray(mol_or_coords.coords_bohr, dtype=np.float64).reshape((-1, 3))
    arr = np.asarray(mol_or_coords, dtype=np.float64)
    return arr.reshape((-1, 3))


def fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return approximately-uniform points on the unit sphere.

    Returns
    -------
    dirs : (n,3)
        Unit vectors.
    w : (n,)
        Equal weights that sum to 4π.
    """

    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")

    i = np.arange(n, dtype=np.float64) + 0.5
    phi = (1.0 + 5.0 ** 0.5) * 0.5

    z = 1.0 - 2.0 * i / float(n)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))

    theta = 2.0 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    dirs = np.stack([x, y, z], axis=1)
    # weights integrate over the sphere
    w = np.full((n,), 4.0 * np.pi / float(n), dtype=np.float64)
    return dirs, w


def angular_grid(n: int, *, kind: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Return an angular grid on the unit sphere.

    Parameters
    ----------
    n
        Requested number of angular points.
    kind
        - ``"lebedev"``: use a Lebedev–Laikov grid; requires ``n`` in
          :data:`asuka.density.lebedev.LEBEDEV_ORDERS`.
        - ``"fibonacci"``: Fibonacci sphere; supports any ``n``.
        - ``"auto"``: use Lebedev when available for ``n``, otherwise Fibonacci.

    Returns
    -------
    dirs : (n,3)
        Unit vectors.
    w : (n,)
        Weights that sum to 4π.
    """

    k = str(kind).lower()
    if k in ("auto", "default"):
        if int(n) in LEBEDEV_ORDERS:
            return lebedev_sphere(int(n))
        return fibonacci_sphere(int(n))
    if k in ("lebedev", "lebedev-laikov", "ll", "laikov"):
        return lebedev_sphere(int(n))
    if k in ("fibonacci", "fib", "fibo"):
        return fibonacci_sphere(int(n))
    raise ValueError("angular kind must be 'auto', 'lebedev', or 'fibonacci'")


def radial_grid_leggauss(n: int, *, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre radial grid on [0, rmax] including r^2 Jacobian.

    Returns
    -------
    r : (n,)
        Radial coordinates.
    wr : (n,)
        Weights for \int_0^{rmax} f(r) r^2 dr.
    """

    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    rmax = float(rmax)
    if rmax <= 0.0:
        raise ValueError("rmax must be > 0")

    # nodes/weights on [-1,1]
    x, w = np.polynomial.legendre.leggauss(n)
    # map to [0,rmax]
    r = 0.5 * (x + 1.0) * rmax
    drdx = 0.5 * rmax

    wr = w * drdx * (r * r)
    return r.astype(np.float64, copy=False), wr.astype(np.float64, copy=False)


def becke_partition_weights(
    points: np.ndarray,
    atom_coords: np.ndarray,
    *,
    becke_n: int = 3,
    RAB: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Becke partition weights w_A(r) for each atom A.

    Parameters
    ----------
    points
        Grid points, shape (npt,3).
    atom_coords
        Atomic coordinates, shape (natm,3).
    becke_n
        Number of smoothing iterations (Becke's recursive polynomial). Typical
        values are 3 or 4.
    RAB
        Optional precomputed inter-atomic distances, shape (natm,natm).

    Returns
    -------
    w : (npt,natm)
        Partition weights that sum to 1 across atoms for each point.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    R = np.asarray(atom_coords, dtype=np.float64).reshape((-1, 3))
    natm = int(R.shape[0])
    if natm == 0:
        raise ValueError("no atoms")

    becke_n = int(becke_n)
    if becke_n < 0:
        raise ValueError("becke_n must be >= 0")

    # Distances from each point to each atom: (npt,natm)
    d = pts[:, None, :] - R[None, :, :]
    rA = np.linalg.norm(d, axis=2)

    if RAB is None:
        dAB = R[:, None, :] - R[None, :, :]
        RAB = np.linalg.norm(dAB, axis=2)
    RAB = np.asarray(RAB, dtype=np.float64)
    if RAB.shape != (natm, natm):
        raise ValueError("RAB has wrong shape")

    w_raw = np.ones((pts.shape[0], natm), dtype=np.float64)

    # Pairwise updates (A<B) to build products efficiently.
    for A in range(natm):
        for B in range(A + 1, natm):
            Rab = float(RAB[A, B])
            if Rab <= 1e-14:
                continue

            mu = (rA[:, A] - rA[:, B]) / Rab
            # Numerically clamp.
            mu = np.clip(mu, -1.0, 1.0)

            p = mu
            for _ in range(becke_n):
                p = 0.5 * (3.0 * p - p * p * p)

            sA = 0.5 * (1.0 - p)  # weight for atom A vs B
            sB = 1.0 - sA

            w_raw[:, A] *= sA
            w_raw[:, B] *= sB

    w_sum = np.sum(w_raw, axis=1, keepdims=True)
    # Avoid division by zero (should not happen unless natm==0).
    w_sum = np.where(w_sum == 0.0, 1.0, w_sum)
    return w_raw / w_sum


def iter_becke_grid(
    mol_or_coords: Molecule | Any,
    *,
    radial_n: int = 50,
    angular_n: int = 302,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    block_size: int = 20000,
    prune_tol: float = 1e-16,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (points, weights) blocks for an atom-centered Becke grid."""

    R = _coords_bohr(mol_or_coords)
    natm = int(R.shape[0])
    if natm == 0:
        raise ValueError("no atoms")

    radial_n = int(radial_n)
    angular_n = int(angular_n)
    rmax = float(rmax)
    becke_n = int(becke_n)
    block_size = max(1, int(block_size))
    prune_tol = float(prune_tol)

    r, wr = radial_grid_leggauss(radial_n, rmax=rmax)
    dirs, w_ang = angular_grid(angular_n, kind=angular_kind)

    # Precompute atom-atom distances once.
    dAB = R[:, None, :] - R[None, :, :]
    RAB = np.linalg.norm(dAB, axis=2)

    # Per-atom blocks.
    for ia in range(natm):
        # Build all points for this atom: (nrad*nang,3)
        pts = R[ia][None, None, :] + r[:, None, None] * dirs[None, :, :]
        pts = pts.reshape((-1, 3))

        w = (wr[:, None] * w_ang[None, :]).reshape((-1,))

        # Becke partition weights for these points; take the column for this atom.
        part = becke_partition_weights(pts, R, becke_n=becke_n, RAB=RAB)[:, ia]
        w = w * part

        if prune_tol > 0.0:
            mask = w > prune_tol
            pts = pts[mask]
            w = w[mask]

        # Yield in chunks.
        for p0 in range(0, int(w.size), block_size):
            p1 = min(int(w.size), p0 + block_size)
            yield pts[p0:p1], w[p0:p1]


def make_becke_grid(
    mol_or_coords: Molecule | Any,
    *,
    radial_n: int = 50,
    angular_n: int = 302,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a full Becke grid (points, weights).

    This is convenient for small systems / debugging. For production, prefer
    :func:`iter_becke_grid` to avoid large allocations.
    """

    pts_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    for pts, w in iter_becke_grid(
        mol_or_coords,
        radial_n=radial_n,
        angular_n=angular_n,
        angular_kind=angular_kind,
        rmax=rmax,
        becke_n=becke_n,
        block_size=10**9,
        prune_tol=prune_tol,
    ):
        pts_list.append(pts)
        w_list.append(w)

    if len(pts_list) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    pts_all = np.concatenate(pts_list, axis=0)
    w_all = np.concatenate(w_list, axis=0)
    return pts_all, w_all


def iter_cube_grid(
    mol_or_coords: Molecule | Any,
    *,
    spacing: float = 0.25,
    padding: float = 4.0,
    block_size: int = 20000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (points, weights) blocks for a uniform cube grid.

    This is mainly intended for visualization and quick prototyping; it is not
    an efficient/accurate replacement for atom-centered grids.
    """

    R = _coords_bohr(mol_or_coords)
    if int(R.shape[0]) == 0:
        raise ValueError("no atoms")

    spacing = float(spacing)
    padding = float(padding)
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")
    if padding < 0.0:
        raise ValueError("padding must be >= 0")

    grid: CubeGrid = make_cube_grid_from_atoms(R, spacing=spacing, padding=padding)

    origin = np.asarray(grid.origin, dtype=np.float64).reshape((3,))
    axes = np.asarray(grid.axes, dtype=np.float64).reshape((3, 3))
    nx, ny, nz = map(int, grid.shape)

    # Volume element (assumes orthogonal grid in make_cube_grid_from_atoms)
    dV = float(abs(np.linalg.det(axes)))

    # We'll stream along z, generating xy slices.
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    ixf, iyf = np.meshgrid(ix, iy, indexing="ij")
    ixf = ixf.reshape((-1, 1))
    iyf = iyf.reshape((-1, 1))

    axis0 = axes[0].reshape((1, 3))
    axis1 = axes[1].reshape((1, 3))
    axis2 = axes[2].reshape((1, 3))

    # Accumulate blocks across multiple z-slices if needed.
    buf_pts: list[np.ndarray] = []
    buf_w: list[np.ndarray] = []
    buf_n = 0

    def flush():
        nonlocal buf_pts, buf_w, buf_n
        if buf_n == 0:
            return
        pts_out = np.concatenate(buf_pts, axis=0)
        w_out = np.concatenate(buf_w, axis=0)
        buf_pts = []
        buf_w = []
        buf_n = 0
        return pts_out, w_out

    for kz in range(nz):
        base = origin[None, :] + float(kz) * axis2
        pts = base + ixf * axis0 + iyf * axis1  # (nx*ny,3)
        w = np.full((pts.shape[0],), dV, dtype=np.float64)

        buf_pts.append(pts)
        buf_w.append(w)
        buf_n += int(w.size)

        if buf_n >= int(block_size):
            out = flush()
            if out is not None:
                yield out

    out = flush()
    if out is not None:
        yield out


__all__ = [
    "GridSpec",
    "fibonacci_sphere",
    "angular_grid",
    "radial_grid_leggauss",
    "becke_partition_weights",
    "iter_becke_grid",
    "make_becke_grid",
    "iter_cube_grid",
]
