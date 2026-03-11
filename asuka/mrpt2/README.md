# asuka.mrpt2

Multi-reference perturbation theory package (currently NEVPT2-focused).

## Purpose

Implements NEVPT2 SC-DF energy/gradient and related DF/CUDA helper paths on top
of ASUKA CAS references.

## Public API

- Driver/data: `DFPairBlock`, `build_df_pair_block`
- NEVPT2 SC-DF drivers/results:
  - `nevpt2_sc_df_from_ref`, `NEVPT2SCDFResult`
  - `nevpt2_sc_df_grad_from_ref`, `NEVPT2SCDFGradResult`
  - `nevpt2_sc_total_energy_df_adjoint`, `NEVPT2SCDFAdjointTotalResult`
- Term-level energy helpers (CPU/CUDA variants) exported from package root.

## Workflows

Used for NEVPT2 SC-DF energy and gradient workflows after CASSCF/CASCI.

## Optional Dependencies

CUDA term evaluators require CuPy and native CUDA extension availability.

## Test Status

Validated by MRPT2/NEVPT2 targeted regression tests and integration runs.
