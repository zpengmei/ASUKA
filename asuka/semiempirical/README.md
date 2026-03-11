# asuka.semiempirical

Semiempirical NDDO methods (AM1 active, PM7 scaffold).

## Purpose

Provides user-facing semiempirical energy/gradient and SCF workflows built on
shared NDDO primitives.

## Public API

- Energies/gradients: `am1_energy`, `am1_gradient`, `am1_energy_gradient`,
  `pm7_energy`
- SCF: `am1_scf`, `SCFResult`
- High-level wrapper: `SemiempiricalCalculator`

## Workflows

Used for lightweight semiempirical geometry/energy scans and as a fast
approximate backend.

## Optional Dependencies

GPU helper paths are optional and require CuPy/native CUDA support.

## Test Status

Covered by semiempirical unit/integration tests and downstream workflow checks.
