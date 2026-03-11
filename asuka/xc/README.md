# asuka.xc

Exchange-correlation (XC) functional and numerical integration package.

## Purpose

Provides functional lookup/evaluation, XC potential builders, and XC nuclear
gradient paths used by RKS/UKS and related workflows.

## Public API

- Functional surface: `FunctionalSpec`, `get_functional`,
  `eval_xc`, `eval_xc_sp`, `eval_xc_u`
- Numerical integration: `build_vxc`, `build_vxc_u`
- XC gradients: `build_vxc_nuc_grad`, `build_vxc_nuc_grad_from_mol`,
  `build_vxc_nuc_grad_fd`, `build_vxc_nuc_grad_fd_from_mol`,
  `XCNucGradResult`, `XCNucGradFDResult`

## Workflows

Used by KS-DFT energy/potential builds and analytic/FD XC gradient paths.

## Optional Dependencies

GPU acceleration paths require CuPy and optional CUDA kernels where available.

## Test Status

Covered by `tests/xc` in both CPU and CUDA-marked lanes.
