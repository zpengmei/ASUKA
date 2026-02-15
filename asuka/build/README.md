# asuka.build

Build helpers for optional native extensions (Cython/CUDA). These modules are
invoked via `python -m ...` and build extension artifacts in-place for the
checked-out repository.

## Public Entry Points

- `python -m asuka.build.epq_ext build_ext --inplace`
- `python -m asuka.build.int1e_ext build_ext --inplace`
- `python -m asuka.build.guga_cuda_ext`
- `python -m asuka.build.guga_cuda_linalg_ext`

## Quick Usage

### Build Cython EPQ helper

```bash
python -m asuka.build.epq_ext build_ext --inplace
```

### Build CUDA matvec extension

```bash
python -m asuka.build.guga_cuda_ext
```

### Build optional CUDA linalg extension

```bash
python -m asuka.build.guga_cuda_linalg_ext
```

## Build Environment Knobs

- `GUGA_USE_OPENMP=1`: enable OpenMP flags in `epq_ext` builds.
- `GUGA_CUDA_NVCC` / `CUDACXX` / `NVCC`: explicit `nvcc` path override.
- `GUGA_CUDA_ROOT` / `CUDAToolkit_ROOT` / `CUDA_HOME` / `CUDA_PATH`:
  explicit CUDA toolkit root override.
- `GUGA_CUDA_ARCH`: override `CMAKE_CUDA_ARCHITECTURES` (for example `80`,
  `89`, or `80;89`).
- `GUGA_CUDA_CMAKE_CONFIGURE_ARGS`: extra args appended to `cmake -S ... -B ...`.
- `GUGA_CUDA_CMAKE_BUILD_ARGS`: extra args appended to `cmake --build ...`.
- `GUGA_CUDA_EXT_OUTPUT_DIR`: output directory for built CUDA extension modules.

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Package-level description of supported build helpers |
| `epq_ext.py` | Cython build helper for `asuka._epq_cy` |
| `int1e_ext.py` | Cython build helper for `asuka.integrals._int1e_cart_cy` |
| `guga_cuda_ext.py` | CMake/pybind11 build helper for `asuka.cuda._guga_cuda_ext` |
| `guga_cuda_linalg_ext.py` | CMake/pybind11 build helper for `asuka.cuda._guga_cuda_linalg_ext` |

## Notes

- Cython helpers require `numpy`, `setuptools`, and `Cython`.
- CUDA helpers require `nvcc`, `cmake`, and `pybind11` available in the active
  Python environment.
- CUDA scripts auto-detect architecture via `nvidia-smi` when possible and
  fall back to `80` if detection is unavailable.
