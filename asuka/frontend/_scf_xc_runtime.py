from __future__ import annotations

from typing import Any


def coerce_or_build_xc_grid(
    mol: Any,
    *,
    xc_grid_coords: Any | None,
    xc_grid_weights: Any | None,
    grid_radial_n: int,
    grid_angular_n: int,
):
    from asuka.density.grids_device import make_becke_grid_device
    import cupy as _cp_grid

    if xc_grid_coords is not None and xc_grid_weights is not None:
        return (
            _cp_grid.asarray(xc_grid_coords, dtype=_cp_grid.float64),
            _cp_grid.asarray(xc_grid_weights, dtype=_cp_grid.float64),
        )
    return make_becke_grid_device(
        mol,
        radial_n=int(grid_radial_n),
        angular_n=int(grid_angular_n),
        radial_scheme="treutler",
    )


def xc_sph_transform_from_map(mol: Any, sph_map: Any):
    if bool(getattr(mol, "cart", False)) or sph_map is None:
        return None
    import cupy as _cp_xc

    if hasattr(sph_map, "T_c2s"):
        return _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
    if hasattr(sph_map, "T_matrix"):
        return _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
    if isinstance(sph_map, tuple) and len(sph_map) >= 1:
        return _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)
    return None


def resolve_xc_runtime(
    *,
    functional: str | None,
    mol: Any,
    sph_map: Any,
    grid_radial_n: int,
    grid_angular_n: int,
    xc_grid_coords: Any | None = None,
    xc_grid_weights: Any | None = None,
):
    """Resolve optional XC spec + grid + AO-transform objects for DFT paths."""

    if functional is None:
        return None, None, None, None

    from asuka.xc.functional import get_functional

    xc_spec = get_functional(functional)
    grid_coords, grid_weights = coerce_or_build_xc_grid(
        mol,
        xc_grid_coords=xc_grid_coords,
        xc_grid_weights=xc_grid_weights,
        grid_radial_n=int(grid_radial_n),
        grid_angular_n=int(grid_angular_n),
    )
    xc_sph_transform = xc_sph_transform_from_map(mol, sph_map)
    return xc_spec, grid_coords, grid_weights, xc_sph_transform


__all__ = [
    "coerce_or_build_xc_grid",
    "resolve_xc_runtime",
    "xc_sph_transform_from_map",
]
