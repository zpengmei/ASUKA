# asuka.cueri

cuERI integral backend in ASUKA: packed basis utilities, dense active-space ERI
builders (CPU/GPU), and streamed DF builders.

## API Layers

- Stable integration boundary: `asuka.cueri.api`
  - `ActiveDFResult`
  - `build_active_df(...)`
- Consolidated public surface: `asuka.cueri` (re-export of `asuka.cueri.core`)
- Advanced controls: import directly from submodules (for cached builders,
  profiling, and lower-level kernels)

## Quick Usage

### Build dense active-space ERIs on CPU

```python
from asuka.cueri import pack_cart_shells_from_mol, build_active_eri_mat_dense_cpu

ao_basis = pack_cart_shells_from_mol(mol)
eri_mat = build_active_eri_mat_dense_cpu(
    ao_basis,
    C_active,  # shape (nao, norb)
    eps_ao=0.0,
    eps_mo=0.0,
)
```

### Reuse cached CPU preprocessing across repeated builds

```python
from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder

builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis, threads=0)
eri_mat = builder.build_eri_mat(C_active, eps_ao=1e-12, eps_mo=1e-12)
```

### Build active-space DF factors

```python
from asuka.cueri import build_active_df

res = build_active_df(
    ao_basis,
    aux_basis,
    C_active,   # shape (nao, norb)
    backend="gpu_rys",
    streamed=True,
)
L_full = res.l_full  # shape (norb*norb, naux), row id pq = p*norb + q
```

## Module Map

| Module | Purpose |
| --- | --- |
| `core.py` | Consolidated public API exported by `asuka.cueri` |
| `api.py` | Stable minimal cuERI boundary (`ActiveDFResult`, `build_active_df`) |
| `mol_basis.py` | PySCF mol -> `BasisCartSoA` packing and cached basis helpers |
| `dense_cpu.py` | CPU dense ERI builds and CPU Schwarz shell-pair bounds |
| `active_space_dense_cpu.py` | Reusable cached CPU builder (`CuERIActiveSpaceDenseCPUBuilder`) |
| `dense.py` | GPU dense ERI builders (`build_active_eri_mat_dense_rys`, etc.) |
| `active_space_dense_gpu.py` | Reusable cached GPU builder (`CuERIActiveSpaceDenseGPUBuilder`) |
| `df.py` | Streamed DF pipeline (`active_Lfull_streamed_basis`, metric/cholesky helpers) |
| `screening.py` | GPU Schwarz bounds (`schwarz_sp_device`, `schwarz_shellpairs_device`) |
| `tasks.py` / `eri_dispatch.py` | Task construction/grouping and class-based kernel dispatch |
| `ao2mo.py` | Native AO->MO transform helpers compatible with PySCF-style workflows |

## Notes

- Dense builders and basis packers use cartesian shell layout; use
  `pack_cart_shells_from_mol(...)` for PySCF mol-like objects.
- GPU paths require CuPy and the CUDA extension (`asuka.cueri._cueri_cuda_ext`).
- CPU dense paths use Cython extensions (`asuka.cueri._eri_rys_cpu`,
  `asuka.cueri._pair_coeff_cpu`).
- CUDA angular-momentum/root limits are set by compiled kernels; see
  `asuka.cueri.gpu.CUDA_MAX_L` and `asuka.cueri.gpu.CUDA_MAX_NROOTS`.
- PySCF patch/context helpers are not part of `asuka.cueri`; they live in
  higher-level integration modules (for example `asuka/cuda/active_space_df/`).

## Build Helpers

- CPU extensions: `python -m asuka.cueri.build_cpu_ext build_ext --inplace`
- CUDA extension: `python -m asuka.cueri.build_cuda_ext --clean`
