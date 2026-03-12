# asuka.qmc

Stochastic projector methods in CSF (GUGA/DRT) space.

## Purpose

Implements scalable FCIQMC/FCI-FRI style sparse-state propagation,
coalescing, estimators, and projector infrastructure.

## Public API

- Driver/results: `run_fciqmc`, `FCIQMCRun`, `run_fcifri_ground`,
  `run_fcifri_block`, `FCIFRIRun`
- Sparse/core helpers: `SparseVector`, `sparse_dot_sorted`, coalescers,
  `initiator_threshold`, shift/estimator helpers
- The old int32-only CPU spawn/projector/subspace helper stack is no longer
  part of the supported package surface.
- Optional debug exports are re-exported when debug module is present.

## Workflows

Used for stochastic CI simulation and projector diagnostics.

## Optional Dependencies

CUDA backends require CuPy and cuGUGA CUDA extension.

## Test Status

Covered by `tests/qmc` plus non-CUDA blocking suites.
