# asuka.density

Numerical grid construction utilities (CPU and optional CUDA paths).

## Purpose

Provides atom-centered Becke grids, cube grids, and grid-collection helpers for
DFT, THC, and DVR workflows.

## Public API

- Grid core: `GridSpec`, `GridBatch`, `GridRequest`, `iter_grid`, `collect_grid`
- CPU grids: `iter_becke_grid`, `make_becke_grid`, `iter_cube_grid`
- Optional CUDA grids (when available):
  `DeviceGridSpec`, `iter_becke_grid_device`, `make_becke_grid_device`,
  `iter_rdvr_grid_device`, `make_rdvr_grid_device`

## Workflows

Used by XC integration, THC factor generation, and DVR/grid-based analysis.

## Optional Dependencies

CUDA grid paths require CuPy and orbitals CUDA extension access via
`asuka.kernels.orbitals`.

## Test Status

Covered by `tests/density`, `tests/integrals` grid-moment checks, and
non-CUDA blocking lanes.
