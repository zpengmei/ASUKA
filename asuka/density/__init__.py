from __future__ import annotations

"""Numerical grid utilities (ported from the density work).

This package currently provides a small, self-contained atom-centered Becke
grid stack (CPU + optional CUDA) used by THC-SCF factor construction.

Notes
-----
- Coordinates are in Bohr.
- CUDA utilities require CuPy and the optional `asuka._orbitals_cuda_ext`.
"""

from .grids import GridSpec, iter_becke_grid, iter_cube_grid, make_becke_grid

try:  # optional CUDA backend
    from .grids_device import DeviceGridSpec, iter_becke_grid_device, make_becke_grid_device

    _CUDA_GRID_IMPORT_OK = True
except Exception:  # pragma: no cover
    _CUDA_GRID_IMPORT_OK = False

try:  # optional CUDA R-DVR backend (depends on the same orbitals CUDA ext)
    from .dvr_grids_device import iter_rdvr_grid_device, make_rdvr_grid_device

    _CUDA_DVR_IMPORT_OK = True
except Exception:  # pragma: no cover
    _CUDA_DVR_IMPORT_OK = False

__all__ = [
    "GridSpec",
    "iter_becke_grid",
    "make_becke_grid",
    "iter_cube_grid",
]

if _CUDA_GRID_IMPORT_OK:  # pragma: no cover
    __all__ += [
        "DeviceGridSpec",
        "iter_becke_grid_device",
        "make_becke_grid_device",
    ]

if _CUDA_DVR_IMPORT_OK:  # pragma: no cover
    __all__ += [
        "iter_rdvr_grid_device",
        "make_rdvr_grid_device",
    ]
