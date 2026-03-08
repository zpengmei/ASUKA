from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.cart import cartesian_components, ncart
from asuka.integrals.cart2sph import AOSphericalTransform, coerce_sph_map


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


def _shell_radial_terms(
    ao_basis: Any, shell: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shifted coordinates and radial prefactors for one shell.

    The returned radial arrays are

    ``rad0 = Σ c exp(-a r²)``
    ``rad1 = Σ c a exp(-a r²)``
    ``rad2 = Σ c a² exp(-a r²)``

    which are sufficient to build AO values, gradients, and Hessians.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    sh = int(shell)

    cxyz = np.asarray(ao_basis.shell_cxyz[sh], dtype=np.float64).reshape((3,))
    p0 = int(ao_basis.shell_prim_start[sh])
    npg = int(ao_basis.shell_nprim[sh])

    exps = np.asarray(ao_basis.prim_exp[p0 : p0 + npg], dtype=np.float64)
    coefs = np.asarray(ao_basis.prim_coef[p0 : p0 + npg], dtype=np.float64)
    if int(exps.size) == 0:
        raise ValueError("shell has no primitives")

    d = pts - cxyz[None, :]
    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]
    r2 = dx * dx + dy * dy + dz * dz

    exp_arg = np.exp(-r2[:, None] * exps[None, :])
    weighted = exp_arg * coefs[None, :]
    rad0 = np.sum(weighted, axis=1)
    rad1 = np.sum(weighted * exps[None, :], axis=1)
    rad2 = np.sum(weighted * (exps[None, :] ** 2), axis=1)
    return pts, dx, dy, dz, rad0, rad1, rad2


def eval_shell_cart(ao_basis: Any, shell: int, points: np.ndarray) -> np.ndarray:
    """Evaluate a single contracted cartesian shell on points.

    Returns array (npt, ncart(l)) in PySCF cart=True ordering.
    """

    pts, dx, dy, dz, rad0, _rad1, _rad2 = _shell_radial_terms(ao_basis, int(shell), points)
    l = int(ao_basis.shell_l[int(shell)])

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
        out[:, ic] = rad0 * px[lx] * py[ly] * pz[lz]
    return out


def eval_shell_cart_value_grad_hess(
    ao_basis: Any,
    shell: int,
    points: np.ndarray,
    *,
    want_hess: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate one cartesian shell and its first/second derivatives on points.

    Parameters
    ----------
    want_hess
        When ``False``, skip the Hessian and return ``None`` in its place.

    Returns
    -------
    value : ndarray
        Shape ``(npt, ncart(l))``.
    grad : ndarray
        Shape ``(npt, ncart(l), 3)``.
    hess : ndarray | None
        Shape ``(npt, ncart(l), 6)`` in ``(xx, xy, xz, yy, yz, zz)`` order,
        or ``None`` when ``want_hess=False``.
    """

    sh = int(shell)
    l = int(ao_basis.shell_l[sh])
    pts, dx, dy, dz, rad0, rad1, rad2 = _shell_radial_terms(ao_basis, sh, points)

    px = [np.ones_like(dx)]
    py = [np.ones_like(dy)]
    pz = [np.ones_like(dz)]
    for _ in range(l + 2):
        px.append(px[-1] * dx)
        py.append(py[-1] * dy)
        pz.append(pz[-1] * dz)

    comps = cartesian_components(l)
    npt = int(pts.shape[0])
    nbf = int(len(comps))

    value = np.empty((npt, nbf), dtype=np.float64)
    grad = np.empty((npt, nbf, 3), dtype=np.float64)
    hess = np.empty((npt, nbf, 6), dtype=np.float64) if bool(want_hess) else None

    for ic, (lx, ly, lz) in enumerate(comps):
        yz = py[ly] * pz[lz]
        xz = px[lx] * pz[lz]
        xy = px[lx] * py[ly]
        xyz = px[lx] * yz

        value[:, ic] = rad0 * xyz

        ddx = -2.0 * rad1 * px[lx + 1] * yz
        if lx > 0:
            ddx = ddx + float(lx) * rad0 * px[lx - 1] * yz
        grad[:, ic, 0] = ddx

        ddy = -2.0 * rad1 * py[ly + 1] * xz
        if ly > 0:
            ddy = ddy + float(ly) * rad0 * py[ly - 1] * xz
        grad[:, ic, 1] = ddy

        ddz = -2.0 * rad1 * pz[lz + 1] * xy
        if lz > 0:
            ddz = ddz + float(lz) * rad0 * pz[lz - 1] * xy
        grad[:, ic, 2] = ddz

        if hess is None:
            continue

        dxx = 4.0 * rad2 * px[lx + 2] * yz
        dxx = dxx + (-2.0 * rad1) * (1.0 + 2.0 * float(lx)) * px[lx] * yz
        if lx >= 2:
            dxx = dxx + float(lx * (lx - 1)) * rad0 * px[lx - 2] * yz
        hess[:, ic, 0] = dxx

        dyy = 4.0 * rad2 * py[ly + 2] * xz
        dyy = dyy + (-2.0 * rad1) * (1.0 + 2.0 * float(ly)) * py[ly] * xz
        if ly >= 2:
            dyy = dyy + float(ly * (ly - 1)) * rad0 * py[ly - 2] * xz
        hess[:, ic, 3] = dyy

        dzz = 4.0 * rad2 * pz[lz + 2] * xy
        dzz = dzz + (-2.0 * rad1) * (1.0 + 2.0 * float(lz)) * pz[lz] * xy
        if lz >= 2:
            dzz = dzz + float(lz * (lz - 1)) * rad0 * pz[lz - 2] * xy
        hess[:, ic, 5] = dzz

        dxy = 4.0 * rad2 * px[lx + 1] * py[ly + 1] * pz[lz]
        if ly > 0:
            dxy = dxy - 2.0 * float(ly) * rad1 * px[lx + 1] * py[ly - 1] * pz[lz]
        if lx > 0:
            dxy = dxy - 2.0 * float(lx) * rad1 * px[lx - 1] * py[ly + 1] * pz[lz]
        if lx > 0 and ly > 0:
            dxy = dxy + float(lx * ly) * rad0 * px[lx - 1] * py[ly - 1] * pz[lz]
        hess[:, ic, 1] = dxy

        dxz = 4.0 * rad2 * px[lx + 1] * py[ly] * pz[lz + 1]
        if lz > 0:
            dxz = dxz - 2.0 * float(lz) * rad1 * px[lx + 1] * py[ly] * pz[lz - 1]
        if lx > 0:
            dxz = dxz - 2.0 * float(lx) * rad1 * px[lx - 1] * py[ly] * pz[lz + 1]
        if lx > 0 and lz > 0:
            dxz = dxz + float(lx * lz) * rad0 * px[lx - 1] * py[ly] * pz[lz - 1]
        hess[:, ic, 2] = dxz

        dyz = 4.0 * rad2 * px[lx] * py[ly + 1] * pz[lz + 1]
        if lz > 0:
            dyz = dyz - 2.0 * float(lz) * rad1 * px[lx] * py[ly + 1] * pz[lz - 1]
        if ly > 0:
            dyz = dyz - 2.0 * float(ly) * rad1 * px[lx] * py[ly - 1] * pz[lz + 1]
        if ly > 0 and lz > 0:
            dyz = dyz + float(ly * lz) * rad0 * px[lx] * py[ly - 1] * pz[lz - 1]
        hess[:, ic, 4] = dyz

    return value, grad, hess


def eval_basis_cart_value_grad_hess_on_points(
    ao_basis: Any,
    points: np.ndarray,
    *,
    want_hess: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate all contracted Cartesian AOs and derivatives on points (CPU).

    Returns
    -------
    phi : ndarray
        Shape ``(npt, nao_cart)``.
    dphi : ndarray
        Shape ``(npt, nao_cart, 3)``.
    hphi : ndarray | None
        Shape ``(npt, nao_cart, 6)`` in ``(xx, xy, xz, yy, yz, zz)`` order,
        or ``None`` when ``want_hess=False``.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    shell_ao_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()
    nshell = int(shell_l.size)

    nao = 0
    for sh in range(nshell):
        ao0 = int(shell_ao_start[sh])
        nao = max(int(nao), int(ao0 + ncart(int(shell_l[sh]))))

    phi = np.zeros((int(pts.shape[0]), int(nao)), dtype=np.float64)
    dphi = np.zeros((int(pts.shape[0]), int(nao), 3), dtype=np.float64)
    hphi = np.zeros((int(pts.shape[0]), int(nao), 6), dtype=np.float64) if bool(want_hess) else None

    for sh in range(nshell):
        ao0 = int(shell_ao_start[sh])
        nbf = int(ncart(int(shell_l[sh])))
        v_blk, g_blk, h_blk = eval_shell_cart_value_grad_hess(
            ao_basis,
            int(sh),
            pts,
            want_hess=bool(want_hess),
        )
        phi[:, ao0 : ao0 + nbf] = v_blk
        dphi[:, ao0 : ao0 + nbf, :] = g_blk
        if hphi is not None:
            assert h_blk is not None
            hphi[:, ao0 : ao0 + nbf, :] = h_blk

    return phi, dphi, hphi


def eval_basis_cart_value_on_points(ao_basis: Any, points: np.ndarray) -> np.ndarray:
    """Evaluate all contracted Cartesian basis functions on points (CPU).

    Returns
    -------
    phi : (npt, nao_cart)
        AO values in PySCF cart=True ordering.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    shell_ao_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()
    nshell = int(shell_l.size)

    nao = 0
    for sh in range(nshell):
        ao0 = int(shell_ao_start[sh])
        l = int(shell_l[sh])
        nao = max(int(nao), int(ao0 + ncart(l)))

    phi = np.zeros((int(pts.shape[0]), int(nao)), dtype=np.float64)
    for sh in range(nshell):
        ao0 = int(shell_ao_start[sh])
        l = int(shell_l[sh])
        nbf = int(ncart(l))
        phi[:, ao0 : ao0 + nbf] = eval_shell_cart(ao_basis, int(sh), pts)
    return phi


def eval_mos_cart_on_points(
    ao_basis: Any,
    C: Any,
    points: np.ndarray,
    mo_list: list[int],
    *,
    sph_map: AOSphericalTransform | tuple[np.ndarray, int, int] | None = None,
) -> np.ndarray:
    """Evaluate selected MOs on points (streamed by shell).

    Parameters
    ----------
    sph_map : AOSphericalTransform | tuple | None
        If not None, spherical AO transform metadata (dataclass or legacy tuple).
        C is assumed to be in spherical AO basis and will be back-transformed
        to Cartesian: ``C_cart = T @ C_sph``.

    Returns
    -------
    psi : (npt, nsel)
    """

    Cn = _asnumpy(C)

    # Back-transform spherical MO coefficients to Cartesian for grid evaluation
    if sph_map is not None:
        T_c2s = np.asarray(coerce_sph_map(sph_map).T_c2s, dtype=np.float64)
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
    "eval_basis_cart_value_grad_hess_on_points",
    "eval_basis_cart_value_on_points",
    "eval_mos_cart_on_points",
    "eval_shell_cart",
    "eval_shell_cart_value_grad_hess",
    "make_cube_grid_from_atoms",
]
