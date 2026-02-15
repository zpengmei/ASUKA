# asuka.cuguga

Core GUGA/DRT building blocks for ASUKA CSF solvers: DRT construction, EPQ
actions, Hamiltonian row oracles, screening knobs, and CUDA autotuning helpers.

## Public Package Exports

`asuka.cuguga` re-exports:

- `DRT`
- `STEP_ORDER`
- `build_drt`
- `autotune`
- `detect_cuda_device_info`
- `list_gpu_profile_presets`
- `AutoTuneTrial`
- `AutoTuneResult`

Most row-oracle and utility APIs are intentionally imported from submodules.

## Quick Usage

### Build a DRT

```python
from asuka.cuguga import build_drt

drt = build_drt(
    norb=8,
    nelec=10,
    twos_target=0,
    # Optional:
    # orbsym=[...],
    # wfnsym=...,
    # ne_constraints={k: (ne_min, ne_max)},
)
print(drt.ncsf, drt.nnodes)
```

### Evaluate one DF sparse Hamiltonian row

```python
from asuka.cuguga.oracle.sparse import connected_row_sparse_df
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import get_state_cache

screen = RowScreening()  # all thresholds disabled -> exact reference behavior
state_cache = get_state_cache(drt)
i_idx, hij = connected_row_sparse_df(
    drt,
    h1e,
    df_eri,
    j=0,
    screening=screen,
    state_cache=state_cache,
)
```

### Autotune CUDA solver settings

```python
from asuka.cuguga import autotune, detect_cuda_device_info

gpu = detect_cuda_device_info()
result = autotune(
    solver,
    h1e,
    eri,
    norb,
    nelec,
    max_cycle=2,
    gpu_info_override=gpu,
)
print(result.best.config)
print(result.recommended_solver_kwargs)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `drt.py` | `DRT` representation and `build_drt`/`build_drt_symm` constructors |
| `oracle/` | Cached connected-row path (`_segment`, `_cache`, `_connected`) and compatibility re-exports |
| `oracle/sparse.py` | Index-based sparse row oracles (`connected_row_sparse*`, dense and DF-backed) |
| `oracle/sparse_path.py` | Path-key sparse row oracles for workflows that avoid global CSF indices |
| `oracle/matvec.py` | Reference `y = Hx` via DF sparse row oracle column-scan |
| `epq/action.py` | EPQ stepping/action kernels and weighted apply helpers |
| `overlap/` | Overlap and matrix-element utilities |
| `davidson.py` | Matrix-free symmetric Davidson eigensolver (NumPy) |
| `autotune.py` | CUDA matvec autotuner and GPU-aware config autofill |
| `eri.py` | ERI format restoration helpers (sym=1/4/8 inputs) |
| `state_cache.py` | `DRTStateCache`, `get_state_cache`, `clear_state_cache` |
| `screening.py` | `RowScreening` thresholds for sparse-row pruning controls |
| `blas_threads.py` | CPU thread-limit context managers (BLAS/OpenMP/PySCF/env) |
| `record.py` | DRT snapshot/reference-count helpers for regression/debugging |

## Notes

- `RowScreening` defaults to zero thresholds. With zeros, sparse-row routines are
  exact up to floating-point roundoff.
- `RowScreening.df_eri_mat_max_bytes` optionally allows DF routines to
  materialize pair-space ERI matrices when profitable.
- Use `asuka.cuguga.blas_threads.asuka_thread_limit(...)` to cap thread pools and
  avoid CPU oversubscription in Python-parallel regions.
