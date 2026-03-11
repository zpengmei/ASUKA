# asuka.soc

Spin-orbit coupling state-interaction and response workflows.

## Purpose

Provides SOC state-interaction Hamiltonian builders, triplet operator
contractions, and z-vector/gradient response utilities.

## Public API

The stable public surface is defined in `asuka.soc.api` and re-exported by
`asuka.soc` package root.

Major exports include:
- SI/state interaction: `soc_state_interaction`, `soc_state_interaction_rassi`,
  `solve_spinfree_state_interaction`
- Triplet/RDM paths: `apply_contracted_triplet_all_m`,
  `trans_trdm1_triplet_streaming`
- Response/grad: `build_soc_ci_rhs_for_zvector`,
  `solve_soc_ci_zvector_response`, `soc_lagrange_response_nuc_grad`
- Optional CUDA helpers: `has_soc_cuda`, `apply_contracted_triplet_all_m_cuda`

## Workflows

Used for SOC coupling analysis and SOC-aware response/gradient calculations.

## Optional Dependencies

CUDA SOC paths require CuPy and cuGUGA CUDA extension via `asuka.kernels.guga`.

## Test Status

Covered by SOC API-surface tests and SOC-related integration checks.
