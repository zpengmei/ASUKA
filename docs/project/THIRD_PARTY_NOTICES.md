# Third-Party Notices

ASUKA interoperates with, benchmarks against, or optionally depends on third-party
software and datasets. This file provides attribution and points to canonical citations.

## Optional Runtime Dependencies

- PySCF (`pyscf`)
- Basis Set Exchange (`basis_set_exchange`)
- CuPy (`cupy-cuda13x` / `cupy-cuda12x`)
- PyTorch (`torch`)
- SciPy (`scipy`)

## CUDA Toolchain Dependencies

- NVIDIA CUDA Toolkit (`nvcc`, runtime libraries)
- pybind11
- CMake

## Citation Policy

When publishing results produced with ASUKA, cite:

1. ASUKA software (`CITATION.cff`)
2. Method references (`docs/project/REFERENCES.md`)
3. Third-party frameworks used in your workflow (for example PySCF, OpenMolcas, BSE)

## License Notes

- ASUKA is distributed under `LICENSE` at the repository root.
- Research-only trees (for example `asuka/research/cueri/tests` and `asuka/research/cueri/tools`)
  are excluded from release source distributions.
