# asuka._solver

Internal decomposition package for `asuka.solver`.

## Purpose

This package contains private runtime/config/helper modules extracted from
`asuka/solver.py` to reduce monolithic complexity while preserving the public
`GUGAFCISolver` API.

## Public API

No stable public API is provided. Treat all symbols here as internal.

## Workflows

Used only through `asuka.solver` and internal test seams.

## Optional Dependencies

Some helper paths use optional CUDA/CuPy when solver backends request GPU.

## Test Status

Covered by `tests/infra/test_solver_internal_modules.py` and solver runtime
regression suites.
