# asuka.kernels

Shared native-extension loader, probe, and registry layer.

## Purpose

Centralizes optional extension imports and symbol probing so runtime modules do
not depend on ad hoc private-module import patterns.

## Public API

- Top-level reporting: `kernel_report`, `print_kernel_report`
- Extension-specific loader/probe modules (for example `guga`, `cueri`,
  `hf_df_jk`, `hf_thc`, `orbitals`, `caspt2`):
  - `load_ext()`
  - `require_ext()`
  - `probe()`

## Workflows

Used by runtime modules for optional extension loading and by diagnostics/CLI
for capability reporting.

## Optional Dependencies

Each registry module handles optional extension presence independently.

## Test Status

Covered by `tests/infra/test_kernels_registry_report.py` and package-level
runtime integration tests.
