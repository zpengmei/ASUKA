from __future__ import annotations

"""Numerical integration grids for density.

ASUKA does not currently ship a full DFT grid stack, so density includes a
self-contained implementation of:

- atom-centered grids (radial × angular)
- Becke partitioning to form molecular integration weights

This is sufficient for integrating local energy densities of the form

    E = integral f(rho, Pi, nabla_rho, tau, ...) dr

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
from .types import GridBatch


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
    # Radial grid scheme: 'leggauss' (fixed [0,rmax] GL) or 'treutler' (Treutler-Ahlrichs M4,
    # per-atom Bragg-Slater scaling — recommended for DFT with meta-GGA functionals).
    radial_scheme: str = "treutler"

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


def radial_grid_treutler_ahlrichs(n: int, r0: float, *, alpha: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Treutler-Ahlrichs M4 radial grid mapped to [0, ∞).

    Transformation: r = (r0 / ln 2) * (1+x)^alpha * ln(2/(1-x))

    Reference: Treutler & Ahlrichs, JCP 102, 346 (1995), Eq. (19) method M4.

    Parameters
    ----------
    n : int
        Number of radial points.
    r0 : float
        Atomic radial scale (Bragg-Slater radius in Bohr).
    alpha : float
        Acceleration parameter (default 0.6 = M4).

    Returns
    -------
    r : (n,)
        Radial coordinates in Bohr, strictly positive, clustered near the nucleus.
    wr : (n,)
        Weights for ∫₀^∞ f(r) r² dr.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    r0 = float(r0)
    if r0 <= 0.0:
        raise ValueError("r0 must be > 0")
    alpha = float(alpha)

    x, w = np.polynomial.legendre.leggauss(n)
    # Clamp away from the singularity at x = +1.
    eps = float(np.finfo(np.float64).eps)
    x = np.clip(x, -1.0 + eps, 1.0 - eps)

    log2 = np.log(2.0)
    ln_term = np.log(2.0 / (1.0 - x))
    r = (r0 / log2) * (1.0 + x) ** alpha * ln_term

    # Jacobian dr/dx
    drdx = (r0 / log2) * (
        alpha * (1.0 + x) ** (alpha - 1.0) * ln_term + (1.0 + x) ** alpha / (1.0 - x)
    )

    wr = w * drdx * r * r
    return r.astype(np.float64, copy=False), wr.astype(np.float64, copy=False)


def _radial_grid_for_atom(n: int, Z: int | None, *, scheme: str, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Select and compute the radial grid for a single atom.

    Parameters
    ----------
    n
        Number of radial points.
    Z
        Atomic number (required for ``'treutler'``; ignored for ``'leggauss'``).
    scheme
        ``'leggauss'``: fixed-[0,rmax] GL grid.
        ``'treutler'``: Treutler-Ahlrichs M4 grid scaled by Bragg-Slater radius.
    rmax
        Upper bound for GL grid (ignored for Treutler).
    """
    s = str(scheme).strip().lower()
    if s in ("leggauss", "gl", "gauss-legendre"):
        return radial_grid_leggauss(int(n), rmax=float(rmax))
    if s in ("treutler", "ta", "treutler-ahlrichs"):
        if Z is None:
            raise ValueError("radial_scheme='treutler' requires atomic number Z")
        from .dvr_grids import _BRAGG_SLATER_RADII_BOHR  # noqa: PLC0415

        z = int(Z)
        if z < 1 or z >= int(_BRAGG_SLATER_RADII_BOHR.size):
            raise ValueError(f"unsupported atomic number Z={z} for Treutler-Ahlrichs grid")
        r0 = float(_BRAGG_SLATER_RADII_BOHR[z])
        return radial_grid_treutler_ahlrichs(int(n), r0)
    raise ValueError(f"unknown radial_scheme: {scheme!r}. Choose 'leggauss' or 'treutler'.")


def radial_grid_leggauss(n: int, *, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre radial grid on [0, rmax] including r^2 Jacobian.

    Returns
    -------
    r : (n,)
        Radial coordinates.
    wr : (n,)
        Weights for int_0^{rmax} f(r) r^2 dr.
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


def _plan_becke_angular_orders(
    radii: np.ndarray,
    *,
    angular_n_max: int,
    angular_kind: str,
    angular_prune: bool,
    atom_Z: int | None,
) -> np.ndarray:
    """Choose a Lebedev point count per radial node.

    This reuses the Parrish-style pruning envelope already used in R-DVR.
    """

    radii = np.asarray(radii, dtype=np.float64).ravel()
    if int(radii.size) == 0:
        return np.zeros((0,), dtype=np.int32)
    if not bool(angular_prune):
        return np.full((int(radii.size),), int(angular_n_max), dtype=np.int32)

    from .dvr_grids import (  # noqa: PLC0415
        _BRAGG_SLATER_RADII_BOHR,
        _lebedev_lmax_from_npts,
        _lebedev_npts_for_lreq,
        _pruned_lreq_parrish_2013,
    )

    angular_kind_s = str(angular_kind).strip().lower()
    if angular_kind_s in {"fibonacci", "fib", "fibo"}:
        raise ValueError("angular_prune=True requires Lebedev grids (use angular_kind='auto' or 'lebedev')")

    lmax_max = _lebedev_lmax_from_npts(int(angular_n_max))
    if lmax_max is None:
        raise ValueError(
            f"angular_prune=True requires angular_n to be a supported Lebedev point count. Got angular_n={int(angular_n_max)}."
        )
    if atom_Z is None:
        raise ValueError("angular_prune=True for Becke grids requires atom_Z (or a Molecule with atomic numbers).")

    z = int(atom_Z)
    if z < 1 or z >= int(_BRAGG_SLATER_RADII_BOHR.size):
        raise ValueError(f"unsupported atomic number for Bragg-Slater pruning: Z={z}")

    rho_bs = float(_BRAGG_SLATER_RADII_BOHR[z])
    out = np.empty((int(radii.size),), dtype=np.int32)
    for i, rnode in enumerate(radii.tolist()):
        lreq = _pruned_lreq_parrish_2013(float(rnode), rho_bs=rho_bs, lmax_max=int(lmax_max))
        out[i] = int(_lebedev_npts_for_lreq(int(lreq), lmax_max=int(lmax_max)))
    return out


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
    angular_prune: bool = False,
    atom_Z: Any | None = None,
    return_batch: bool = False,
    radial_scheme: str = "leggauss",
) -> Iterator[Any]:
    """Yield (points, weights) blocks for an atom-centered Becke grid.

    Parameters
    ----------
    radial_scheme
        ``'leggauss'``: uniform GL grid on [0, rmax] (default, backward-compatible).
        ``'treutler'``: Treutler-Ahlrichs M4 grid with per-atom Bragg-Slater scaling.
        Recommended for DFT: clusters radial points near the nucleus per atom's size.
        Requires atomic numbers (pass a Molecule or ``atom_Z``).
    angular_prune
        Enable Parrish-2013 angular pruning. Requires Lebedev angular grid.
        Works correctly with ``radial_scheme='treutler'`` (uses actual r coordinates
        for the pruning envelope). With ``'leggauss'`` the pruning is less effective
        because points are not clustered near the nucleus.

    If ``return_batch=True``, yields :class:`~asuka.density.types.GridBatch` with
    per-point provenance arrays.
    """

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
    radial_scheme = str(radial_scheme).strip().lower()

    # Precompute atom-atom distances once.
    dAB = R[:, None, :] - R[None, :, :]
    RAB = np.linalg.norm(dAB, axis=2)

    angular_prune = bool(angular_prune)
    return_batch = bool(return_batch)

    # Resolve atomic numbers when needed (Treutler scaling or angular pruning).
    needs_Z = angular_prune or radial_scheme not in ("leggauss", "gl", "gauss-legendre")
    Z_all: np.ndarray | None = None
    if needs_Z:
        if atom_Z is None:
            from .dvr_grids import _atomic_numbers_or_none  # noqa: PLC0415

            Z_all = _atomic_numbers_or_none(mol_or_coords)
        else:
            Z_all = np.asarray(atom_Z, dtype=np.int32).ravel()

        if Z_all is None:
            what = "angular_prune=True" if angular_prune else f"radial_scheme={radial_scheme!r}"
            raise ValueError(
                f"{what} requires atomic numbers. Pass a Molecule with atoms_bohr or provide atom_Z (len=natm)."
            )
        Z_all = np.asarray(Z_all, dtype=np.int32).ravel()
        if int(Z_all.size) == 1 and natm > 1:
            Z_all = np.full((natm,), int(Z_all[0]), dtype=np.int32)
        if int(Z_all.size) != int(natm):
            raise ValueError("atom_Z must have length natm when provided")

    def _emit_blocks(
        ia: int,
        pts: np.ndarray,
        w: np.ndarray,
        rid: np.ndarray | None,
        nang: np.ndarray | None,
    ):
        if int(w.size) == 0:
            return
        for p0 in range(0, int(w.size), int(block_size)):
            p1 = min(int(w.size), p0 + int(block_size))
            pts_blk = pts[p0:p1]
            w_blk = w[p0:p1]
            if return_batch:
                rid_blk = None if rid is None else rid[p0:p1]
                nang_blk = None if nang is None else nang[p0:p1]
                yield GridBatch(
                    points=np.ascontiguousarray(pts_blk),
                    weights=np.ascontiguousarray(w_blk).ravel(),
                    point_atom=np.full((int(w_blk.size),), int(ia), dtype=np.int32),
                    point_radial_index=None if rid_blk is None else np.ascontiguousarray(rid_blk),
                    point_angular_n=None if nang_blk is None else np.ascontiguousarray(nang_blk),
                    meta={"grid_kind": "becke", "atom": int(ia), "angular_prune": bool(angular_prune)},
                )
            else:
                yield np.ascontiguousarray(pts_blk), np.ascontiguousarray(w_blk).ravel()

    # Unpruned fast path: fixed angular order everywhere.
    if not angular_prune:
        dirs, w_ang = angular_grid(angular_n, kind=angular_kind)

        # For leggauss, all atoms share the same radial grid; precompute once.
        _r_shared: np.ndarray | None = None
        _wr_shared: np.ndarray | None = None
        if radial_scheme in ("leggauss", "gl", "gauss-legendre"):
            _r_shared, _wr_shared = radial_grid_leggauss(radial_n, rmax=rmax)
            rid_base_shared = np.repeat(np.arange(int(radial_n), dtype=np.int32), int(angular_n))
            nang_base_shared = np.full((int(rid_base_shared.size),), int(angular_n), dtype=np.int32)

        for ia in range(natm):
            if _r_shared is not None:
                r, wr = _r_shared, _wr_shared
                rid_base = rid_base_shared
                nang_base = nang_base_shared
            else:
                Z_ia = int(Z_all[int(ia)]) if Z_all is not None else None  # type: ignore[index]
                r, wr = _radial_grid_for_atom(radial_n, Z_ia, scheme=radial_scheme, rmax=rmax)
                rid_base = np.repeat(np.arange(int(radial_n), dtype=np.int32), int(angular_n))
                nang_base = np.full((int(rid_base.size),), int(angular_n), dtype=np.int32)

            pts = R[ia][None, None, :] + r[:, None, None] * dirs[None, :, :]
            pts = np.asarray(pts, dtype=np.float64).reshape((-1, 3))
            w = (wr[:, None] * w_ang[None, :]).reshape((-1,))

            part = becke_partition_weights(pts, R, becke_n=becke_n, RAB=RAB)[:, ia]
            w = w * part

            rid = rid_base.copy() if return_batch else None
            nang = nang_base.copy() if return_batch else None

            if prune_tol > 0.0:
                mask = w > prune_tol
                pts = pts[mask]
                w = w[mask]
                if rid is not None:
                    rid = rid[mask]
                if nang is not None:
                    nang = nang[mask]

            yield from _emit_blocks(int(ia), pts, w, rid, nang)
        return

    # Per-atom blocks.
    for ia in range(natm):
        assert Z_all is not None  # for type-checkers
        Z_ia = int(Z_all[int(ia)])
        r, wr = _radial_grid_for_atom(radial_n, Z_ia, scheme=radial_scheme, rmax=rmax)
        nang_nodes = _plan_becke_angular_orders(
            r,
            angular_n_max=int(angular_n),
            angular_kind=str(angular_kind),
            angular_prune=True,
            atom_Z=Z_ia,
        )

        pts_buf: list[np.ndarray] = []
        w_buf: list[np.ndarray] = []
        rid_buf: list[np.ndarray] = []
        nang_buf: list[np.ndarray] = []
        n_buf = 0

        i0 = 0
        while i0 < int(r.size):
            nang = int(nang_nodes[i0])
            i1 = i0 + 1
            while i1 < int(r.size) and int(nang_nodes[i1]) == nang:
                i1 += 1

            dirs_i, wang_i = angular_grid(int(nang), kind=str(angular_kind))
            r_seg = r[i0:i1]
            wr_seg = wr[i0:i1]

            pts_seg = R[int(ia)][None, None, :] + r_seg[:, None, None] * dirs_i[None, :, :]
            pts_seg = np.asarray(pts_seg, dtype=np.float64).reshape((-1, 3))
            w_seg = (wr_seg[:, None] * wang_i[None, :]).reshape((-1,))

            rid_seg = np.repeat(np.arange(int(i0), int(i1), dtype=np.int32), int(nang))
            nang_seg = np.full((int(w_seg.size),), int(nang), dtype=np.int32)

            pts_buf.append(np.ascontiguousarray(pts_seg))
            w_buf.append(np.ascontiguousarray(w_seg))
            rid_buf.append(np.ascontiguousarray(rid_seg))
            nang_buf.append(np.ascontiguousarray(nang_seg))
            n_buf += int(w_seg.size)
            i0 = i1

            if n_buf >= int(block_size):
                pts_blk = np.concatenate(pts_buf, axis=0)
                w_blk = np.concatenate(w_buf, axis=0)
                rid_blk = np.concatenate(rid_buf, axis=0)
                nang_blk = np.concatenate(nang_buf, axis=0)
                pts_buf.clear()
                w_buf.clear()
                rid_buf.clear()
                nang_buf.clear()
                n_buf = 0

                wpart = becke_partition_weights(pts_blk, R, becke_n=int(becke_n), RAB=RAB)
                w_mol = w_blk * wpart[:, int(ia)]
                if prune_tol > 0.0:
                    mask = w_mol > float(prune_tol)
                    pts_blk = pts_blk[mask]
                    w_mol = w_mol[mask]
                    rid_blk = rid_blk[mask]
                    nang_blk = nang_blk[mask]

                yield from _emit_blocks(int(ia), pts_blk, w_mol, rid_blk, nang_blk)

        if n_buf:
            pts_blk = np.concatenate(pts_buf, axis=0)
            w_blk = np.concatenate(w_buf, axis=0)
            rid_blk = np.concatenate(rid_buf, axis=0)
            nang_blk = np.concatenate(nang_buf, axis=0)

            wpart = becke_partition_weights(pts_blk, R, becke_n=int(becke_n), RAB=RAB)
            w_mol = w_blk * wpart[:, int(ia)]
            if prune_tol > 0.0:
                mask = w_mol > float(prune_tol)
                pts_blk = pts_blk[mask]
                w_mol = w_mol[mask]
                rid_blk = rid_blk[mask]
                nang_blk = nang_blk[mask]

            yield from _emit_blocks(int(ia), pts_blk, w_mol, rid_blk, nang_blk)


def make_becke_grid(
    mol_or_coords: Molecule | Any,
    *,
    radial_n: int = 50,
    angular_n: int = 302,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1e-16,
    radial_scheme: str = "leggauss",
    atom_Z: Any | None = None,
    return_point_atom: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a full Becke grid (points, weights[, point_atom]).

    This is convenient for small systems / debugging. For production, prefer
    :func:`iter_becke_grid` to avoid large allocations.
    """

    pts_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    atom_list: list[np.ndarray] = []
    if bool(return_point_atom):
        for batch in iter_becke_grid(
            mol_or_coords,
            radial_n=radial_n,
            angular_n=angular_n,
            angular_kind=angular_kind,
            rmax=rmax,
            becke_n=becke_n,
            block_size=10**9,
            prune_tol=prune_tol,
            radial_scheme=radial_scheme,
            atom_Z=atom_Z,
            return_batch=True,
        ):
            pts_list.append(batch.points)
            w_list.append(batch.weights)
            atom_list.append(batch.point_atom)
    else:
        for pts, w in iter_becke_grid(
            mol_or_coords,
            radial_n=radial_n,
            angular_n=angular_n,
            angular_kind=angular_kind,
            rmax=rmax,
            becke_n=becke_n,
            block_size=10**9,
            prune_tol=prune_tol,
            radial_scheme=radial_scheme,
            atom_Z=atom_Z,
        ):
            pts_list.append(pts)
            w_list.append(w)

    if len(pts_list) == 0:
        if bool(return_point_atom):
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
            )
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    pts_all = np.concatenate(pts_list, axis=0)
    w_all = np.concatenate(w_list, axis=0)
    if bool(return_point_atom):
        atom_all = np.concatenate(atom_list, axis=0)
        return pts_all, w_all, atom_all
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
    "radial_grid_treutler_ahlrichs",
    "becke_partition_weights",
    "iter_becke_grid",
    "make_becke_grid",
    "iter_cube_grid",
]
