# asuka.nddo_core

Shared NDDO primitives backing semiempirical workflows.

## Purpose

Provides lazily-exported NDDO basis/pair/multipole/RI/core-H/Fock/energy helper
functions and datatypes used by semiempirical drivers.

## Public API

Package root lazily exports symbols listed in `_EXPORT_MAP` in `__init__.py`
(for example `build_pair_list`, `build_core_hamiltonian`, `build_fock`,
`core_core_repulsion`, `AtomicData`, `PairData`).

## Workflows

Consumed by `asuka.semiempirical` CPU/GPU paths.

## Optional Dependencies

None at package import; some downstream workflows may require CUDA stacks.

## Test Status

Covered through semiempirical SCF/gradient regression tests.
