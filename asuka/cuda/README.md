# asuka.cuda

CUDA backends for ASUKA GUGA/DRT workflows: matvec/hop kernels, Davidson,
RDM builders, MRCI CUDA hops, and active-space DF integration helpers.

## API Surface

- `asuka.cuda` package root intentionally exports no stable top-level API.
- Main usage is indirect through solver/driver backends.
- The `active_space_df` subpackage provides explicit helper APIs for active-space
  DF/dense integral construction.

## Quick Usage

### Run CI with CUDA matvec backend

```python
from asuka.frontend import Molecule
from asuka.frontend.scf import run_hf_df
from asuka.mcscf import run_casscf

mol = Molecule.from_atoms("N 0 0 0; N 0 0 1.0977", unit="Angstrom", basis="cc-pvdz", cart=True, spin=0)
scf_out = run_hf_df(mol, method="rhf", backend="cuda", df=True, auxbasis="autoaux")
casscf = run_casscf(scf_out, ncore=2, ncas=6, nelecas=6, nroots=1, backend="cuda", df=True)
print(casscf.e_tot)
```

### Use active-space DF helpers

```python
from asuka.cuda.active_space_df import (
    ActiveSpaceDFBuilder,
)

builder = ActiveSpaceDFBuilder(mol)
df_data = builder.build(c_cas=mo_cas)
```

## CUDA Submodules

| Module | Purpose |
| --- | --- |
| `cuda_backend.py` | Core CUDA matvec/hop kernels, EPQ-table builders, sparse formats, and kernel4 apply paths |
| `cuda_davidson.py` | GPU Davidson solver (`davidson_sym_gpu`) |
| `rdm_gpu.py` | CUDA RDM builders (`make_rdm12_cuda`, transition RDM helpers) |
| `mrci_hop.py` / `mrci_hop_tiled.py` | CUDA hop backends for MRCI kernels |
| `cuda_linalg_backend.py` | Optional CUDA linear-algebra extension wrappers |
| `cublas_workspace.py` | cuBLAS FP64 fixed-point emulation workspace estimators |
| `autotune.py` | CUDA kernel/thread autotuning helpers and cache |
| `active_space_df/` | GPU active-space DF/dense integral builders |
| `ext/` | Source/build files for `_guga_cuda_ext` |
| `linalg_ext/` | Source/build files for `_guga_cuda_linalg_ext` |

## Build Requirements

- Runtime: CuPy + CUDA-capable GPU.
- ASUKA CUDA extension: `python -m asuka.build.guga_cuda_ext`
- Optional CUDA linalg extension: `python -m asuka.build.guga_cuda_linalg_ext`
- For `active_space_df` cuERI paths: build cuERI CUDA extension too
  (`python -m asuka.cueri.build_cuda_ext`).
