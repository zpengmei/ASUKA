from __future__ import annotations

"""Finite-difference XC nuclear gradients for the numerical-integration module.

This module provides a reference finite-difference geometry gradient for the
CPU :func:`asuka.xc.nuc_grad._build_vxc_numpy` helper. It is primarily intended
for cross-validation of the analytical implementation in :mod:`asuka.xc.nuc_grad`.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from asuka.density.grids import becke_partition_weights, make_becke_grid
from asuka.integrals.int1e_cart import shell_to_atom_map

from .functional import FunctionalSpec, get_functional
from .nuc_grad import _build_vxc_numpy, _coords_bohr_like, _recover_local_becke_weights


@dataclass(frozen=True)
class XCNucGradFDResult:
    """Result container for finite-difference XC gradients."""

    e_xc: float
    grad_xc: np.ndarray
    v_xc: np.ndarray | None = None
    grad_vxc: np.ndarray | None = None
    moving_grid: bool = True
    backend: str = "fd"


def _normalize_spec(spec: FunctionalSpec | str) -> FunctionalSpec:
    if isinstance(spec, FunctionalSpec):
        return spec
    return get_functional(str(spec))


def _shift_basis_centers(
    ao_basis: Any,
    *,
    shell_atom: np.ndarray,
    atom_index: int,
    displacement: np.ndarray,
) -> Any:
    if not hasattr(ao_basis, "shell_cxyz"):
        raise TypeError("ao_basis must provide shell_cxyz")
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if int(shell_atom.size) != int(np.asarray(getattr(ao_basis, "shell_cxyz")).shape[0]):
        raise ValueError("shell_atom must have one entry per shell")

    disp = np.asarray(displacement, dtype=np.float64).reshape((3,))
    shell_cxyz = np.asarray(getattr(ao_basis, "shell_cxyz"), dtype=np.float64).copy()
    shell_cxyz[shell_atom == int(atom_index), :] += disp[None, :]

    if hasattr(ao_basis, "__dataclass_fields__"):
        return replace(ao_basis, shell_cxyz=np.ascontiguousarray(shell_cxyz))
    raise TypeError("ao_basis must be a dataclass (e.g., BasisCartSoA) for finite-difference shifting")


def _fd_geom_gradient_from_forward(
    *,
    ao_basis: Any,
    atom_coords: np.ndarray,
    grid_points: np.ndarray,
    grid_weights: np.ndarray,
    shell_atom: np.ndarray,
    point_atom: np.ndarray | None,
    displacement: float,
    moving_grid: bool,
    becke_n: int,
    forward: Callable[[Any, np.ndarray, np.ndarray], tuple[np.ndarray, float]],
    return_vxc_grad: bool,
    partition_weights_fn: Callable[..., np.ndarray] = becke_partition_weights,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Central-difference geometry gradient for a forward Vxc/Exc routine.

    The ``forward`` callback must return ``(V_xc, E_xc)`` for the provided basis,
    points, and weights.
    """

    atom_coords = np.asarray(atom_coords, dtype=np.float64).reshape((-1, 3))
    if int(atom_coords.shape[0]) == 0:
        raise ValueError("atom_coords must be non-empty")

    pts0 = np.asarray(grid_points, dtype=np.float64).reshape((-1, 3))
    w0 = np.asarray(grid_weights, dtype=np.float64).ravel()
    if int(pts0.shape[0]) != int(w0.shape[0]):
        raise ValueError("grid_points and grid_weights must contain the same number of points")

    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if int(shell_atom.size) != int(np.asarray(getattr(ao_basis, "shell_cxyz")).shape[0]):
        raise ValueError("shell_atom must have one entry per shell")

    moving_grid = bool(moving_grid)
    displacement = float(displacement)
    if displacement <= 0.0:
        raise ValueError("displacement must be > 0")

    if moving_grid:
        if point_atom is None:
            raise ValueError("point_atom is required when moving_grid=True")
        point_atom = np.asarray(point_atom, dtype=np.int32).ravel()
        if point_atom.shape != (int(pts0.shape[0]),):
            raise ValueError("point_atom must have one entry per grid point")

        part0 = partition_weights_fn(pts0, atom_coords, becke_n=int(becke_n))
        owner_part0 = part0[np.arange(int(point_atom.size)), point_atom]
        w_local = _recover_local_becke_weights(w0, owner_part0)
    else:
        w_local = None

    natm = int(atom_coords.shape[0])
    grad = np.zeros((natm, 3), dtype=np.float64)
    vxc_grad: np.ndarray | None = None
    if bool(return_vxc_grad):
        raise NotImplementedError("Finite-difference grad_vxc is not implemented yet")

    for ia in range(natm):
        for xyz in range(3):
            disp_vec = np.zeros((3,), dtype=np.float64)
            disp_vec[xyz] = displacement

            coords_p = atom_coords.copy()
            coords_m = atom_coords.copy()
            coords_p[ia, :] += disp_vec
            coords_m[ia, :] -= disp_vec

            basis_p = _shift_basis_centers(ao_basis, shell_atom=shell_atom, atom_index=ia, displacement=disp_vec)
            basis_m = _shift_basis_centers(ao_basis, shell_atom=shell_atom, atom_index=ia, displacement=-disp_vec)

            if moving_grid:
                assert point_atom is not None
                assert w_local is not None
                pts_p = pts0.copy()
                pts_m = pts0.copy()
                mask = point_atom == int(ia)
                pts_p[mask, :] += disp_vec[None, :]
                pts_m[mask, :] -= disp_vec[None, :]

                part_p = partition_weights_fn(pts_p, coords_p, becke_n=int(becke_n))
                part_m = partition_weights_fn(pts_m, coords_m, becke_n=int(becke_n))
                owner_part_p = part_p[np.arange(int(point_atom.size)), point_atom]
                owner_part_m = part_m[np.arange(int(point_atom.size)), point_atom]
                w_p = w_local * owner_part_p
                w_m = w_local * owner_part_m
            else:
                pts_p = pts0
                pts_m = pts0
                w_p = w0
                w_m = w0

            _V_p, E_p = forward(basis_p, pts_p, w_p)
            _V_m, E_m = forward(basis_m, pts_m, w_m)
            grad[ia, xyz] = (float(E_p) - float(E_m)) / (2.0 * displacement)

    return np.ascontiguousarray(grad), vxc_grad


def build_vxc_nuc_grad_fd(
    spec: FunctionalSpec | str,
    D: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    atom_coords: Any,
    point_atom: Any | None = None,
    shell_atom: Any | None = None,
    becke_n: int = 3,
    moving_grid: bool = True,
    displacement: float = 1.0e-5,
    batch_size: int = 4096,
    sph_transform: Any | None = None,
    return_vxc: bool = False,
    return_vxc_grad: bool = False,
) -> XCNucGradFDResult:
    """Finite-difference XC nuclear gradient on an explicit quadrature grid."""

    if bool(return_vxc_grad):
        raise NotImplementedError("Finite-difference grad_vxc is not implemented yet")

    spec_use = _normalize_spec(spec)
    atom_coords_np = _coords_bohr_like(atom_coords)

    pts_np = np.asarray(grid_coords, dtype=np.float64).reshape((-1, 3))
    w_np = np.asarray(grid_weights, dtype=np.float64).ravel()
    if int(pts_np.shape[0]) != int(w_np.shape[0]):
        raise ValueError("grid_coords and grid_weights must contain the same number of points")

    if shell_atom is None:
        shell_atom_np = shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords_np)
    else:
        shell_atom_np = np.asarray(shell_atom, dtype=np.int32).ravel()

    if point_atom is None:
        point_atom_np: np.ndarray | None = None
    else:
        point_atom_np = np.asarray(point_atom, dtype=np.int32).ravel()

    V_ref: np.ndarray | None = None
    E_ref: float = 0.0
    if bool(return_vxc):
        V_ref, E_ref = _build_vxc_numpy(
            spec_use,
            D,
            ao_basis,
            pts_np,
            w_np,
            batch_size=int(batch_size),
            sph_transform=sph_transform,
        )

    def forward(basis, pts: np.ndarray, wts: np.ndarray) -> tuple[np.ndarray, float]:
        return _build_vxc_numpy(
            spec_use,
            D,
            basis,
            pts,
            wts,
            batch_size=int(batch_size),
            sph_transform=sph_transform,
        )

    grad, _ = _fd_geom_gradient_from_forward(
        ao_basis=ao_basis,
        atom_coords=atom_coords_np,
        grid_points=pts_np,
        grid_weights=w_np,
        shell_atom=shell_atom_np,
        point_atom=point_atom_np,
        displacement=float(displacement),
        moving_grid=bool(moving_grid),
        becke_n=int(becke_n),
        forward=forward,
        return_vxc_grad=False,
        partition_weights_fn=becke_partition_weights,
    )

    if not bool(return_vxc):
        _V, E_ref = _build_vxc_numpy(
            spec_use,
            D,
            ao_basis,
            pts_np,
            w_np,
            batch_size=int(batch_size),
            sph_transform=sph_transform,
        )
    return XCNucGradFDResult(
        e_xc=float(E_ref),
        grad_xc=np.ascontiguousarray(grad),
        v_xc=V_ref,
        grad_vxc=None,
        moving_grid=bool(moving_grid),
        backend="fd",
    )


def build_vxc_nuc_grad_fd_from_mol(
    spec: FunctionalSpec | str,
    D: Any,
    ao_basis: Any,
    mol_or_coords: Any,
    *,
    radial_n: int = 75,
    angular_n: int = 590,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1.0e-16,
    radial_scheme: str = "treutler",
    atom_Z: Any | None = None,
    moving_grid: bool = True,
    displacement: float = 1.0e-5,
    batch_size: int = 4096,
    sph_transform: Any | None = None,
    return_vxc: bool = False,
    return_vxc_grad: bool = False,
) -> XCNucGradFDResult:
    """Finite-difference XC nuclear gradient with an internally built Becke grid."""

    atom_coords_np = _coords_bohr_like(mol_or_coords)
    grid_coords, grid_weights, point_atom = make_becke_grid(
        mol_or_coords,
        radial_n=int(radial_n),
        angular_n=int(angular_n),
        angular_kind=str(angular_kind),
        rmax=float(rmax),
        becke_n=int(becke_n),
        prune_tol=float(prune_tol),
        radial_scheme=str(radial_scheme),
        atom_Z=atom_Z,
        return_point_atom=True,
    )
    return build_vxc_nuc_grad_fd(
        spec,
        D,
        ao_basis,
        grid_coords,
        grid_weights,
        atom_coords=atom_coords_np,
        point_atom=point_atom,
        becke_n=int(becke_n),
        moving_grid=bool(moving_grid),
        displacement=float(displacement),
        batch_size=int(batch_size),
        sph_transform=sph_transform,
        return_vxc=bool(return_vxc),
        return_vxc_grad=bool(return_vxc_grad),
    )


__all__ = [
    "XCNucGradFDResult",
    "_fd_geom_gradient_from_forward",
    "build_vxc_nuc_grad_fd",
    "build_vxc_nuc_grad_fd_from_mol",
]

