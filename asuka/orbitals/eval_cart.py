from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.cart import cartesian_components, ncart


def _asnumpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class CubeGrid:
    origin: np.ndarray
    axes: np.ndarray
    shape: tuple[int, int, int]


def make_cube_grid_from_atoms(
    atom_coords_bohr: np.ndarray,
    *,
    spacing: float = 0.25,
    padding: float = 4.0,
) -> CubeGrid:
    """Create an orthogonal cube grid covering atoms with padding (Bohr units)."""

    R = np.asarray(atom_coords_bohr, dtype=np.float64).reshape((-1, 3))
    if R.size == 0:
        raise ValueError("no atoms")
    spacing = float(spacing)
    padding = float(padding)
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")
    if padding < 0.0:
        raise ValueError("padding must be >= 0")

    rmin = np.min(R, axis=0) - padding
    rmax = np.max(R, axis=0) + padding
    L = rmax - rmin

    nx = int(np.ceil(L[0] / spacing)) + 1
    ny = int(np.ceil(L[1] / spacing)) + 1
    nz = int(np.ceil(L[2] / spacing)) + 1

    origin = np.asarray(rmin, dtype=np.float64)
    axes = np.asarray(
        [
            [spacing, 0.0, 0.0],
            [0.0, spacing, 0.0],
            [0.0, 0.0, spacing],
        ],
        dtype=np.float64,
    )
    return CubeGrid(origin=origin, axes=axes, shape=(nx, ny, nz))


def eval_shell_cart(ao_basis: Any, shell: int, points: np.ndarray) -> np.ndarray:
    """Evaluate a single contracted cartesian shell on points.

    Returns array (npt, ncart(l)) in PySCF cart=True ordering.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    sh = int(shell)

    cxyz = np.asarray(ao_basis.shell_cxyz[sh], dtype=np.float64).reshape((3,))
    l = int(ao_basis.shell_l[sh])
    p0 = int(ao_basis.shell_prim_start[sh])
    npg = int(ao_basis.shell_nprim[sh])

    exps = np.asarray(ao_basis.prim_exp[p0 : p0 + npg], dtype=np.float64)
    coefs = np.asarray(ao_basis.prim_coef[p0 : p0 + npg], dtype=np.float64)

    d = pts - cxyz[None, :]
    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]
    r2 = dx * dx + dy * dy + dz * dz

    rad = np.zeros((pts.shape[0],), dtype=np.float64)
    for a, c in zip(exps, coefs):
        rad += float(c) * np.exp(-float(a) * r2)

    px = [np.ones_like(dx)]
    py = [np.ones_like(dy)]
    pz = [np.ones_like(dz)]
    for _ in range(l):
        px.append(px[-1] * dx)
        py.append(py[-1] * dy)
        pz.append(pz[-1] * dz)

    comps = cartesian_components(l)
    out = np.empty((pts.shape[0], len(comps)), dtype=np.float64)
    for ic, (lx, ly, lz) in enumerate(comps):
        out[:, ic] = rad * px[lx] * py[ly] * pz[lz]
    return out


def eval_mos_cart_on_points(
    ao_basis: Any,
    C: Any,
    points: np.ndarray,
    mo_list: list[int],
    *,
    sph_map: tuple[np.ndarray, int, int] | None = None,
) -> np.ndarray:
    """Evaluate selected MOs on points (streamed by shell).

    Parameters
    ----------
    sph_map : tuple | None
        If not None, ``(T, nao_cart, nao_sph)`` from SCF result.
        C is assumed to be in spherical AO basis and will be back-transformed
        to Cartesian: ``C_cart = T @ C_sph``.

    Returns
    -------
    psi : (npt, nsel)
    """

    Cn = _asnumpy(C)

    # Back-transform spherical MO coefficients to Cartesian for grid evaluation
    if sph_map is not None:
        T_c2s = np.asarray(sph_map[0], dtype=np.float64)
        Cn = T_c2s @ Cn

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    mo_list = [int(i) for i in mo_list]
    if Cn.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    nao, nmo = map(int, Cn.shape)
    if any(i < 0 or i >= nmo for i in mo_list):
        raise IndexError("mo index out of range")

    nshell = int(np.asarray(ao_basis.shell_l).size)
    psi = np.zeros((pts.shape[0], len(mo_list)), dtype=np.float64)

    for sh in range(nshell):
        ao0 = int(ao_basis.shell_ao_start[sh])
        l = int(ao_basis.shell_l[sh])
        nbf = int(ncart(l))
        block = eval_shell_cart(ao_basis, sh, pts)  # (npt, nbf)
        Cblk = Cn[ao0 : ao0 + nbf, :][:, mo_list]  # (nbf, nsel)
        psi += block @ Cblk

    return psi


__all__ = [
    "CubeGrid",
    "eval_mos_cart_on_points",
    "eval_shell_cart",
    "make_cube_grid_from_atoms",
]

