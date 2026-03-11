# asuka.linalg

Dependency-light linear algebra helper package.

## Purpose

Provides focused linear algebra utilities used by DF and adjoint paths without
requiring SciPy/CuPy SciPy wrappers in core call sites.

## Public API

Package is module-oriented; import specific helpers (for example triangular
solves) from submodules.

## Workflows

Used by DF/adjoint and numerical kernels that need consistent CPU/GPU math
helpers.

## Optional Dependencies

GPU branches may use CuPy when present.

## Test Status

Covered indirectly via DF/adjoint regression suites.
