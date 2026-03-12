# asuka.sci

Scalable SCI/CIPSI utilities in the native GUGA/DRT CSF basis.

## Public Package Exports

`asuka.sci` re-exports:

- `run_cipsi_trials`
- `build_cipsi_trials_from_scf`
- `heat_bath_select_and_pt2_sparse`

## Quick Usage

### Scalable Trial-Space API

```python
from asuka.sci import run_cipsi_trials

res = run_cipsi_trials(
    drt,
    h1e,
    eri,
    backend="cuda_key64",
    nroots=1,
    init_ncsf=256,
    max_ncsf=20_000,
    selection_mode="heat_bath",
    epq_mode="no_epq_support_aware",
)
print(res.e_var[0], res.e_pt2[0], res.e_tot[0])
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public API re-export hub |
| `gpu_cipsi.py` | Scalable CIPSI trial-space driver |
| `sparse_support.py` | Shared sparse Hamiltonian / selection / subspace helpers |
| `hb_selection.py` | Scalable heat-bath selection helpers |
| `frontier_hash.py` | Scalable sparse frontier selector |

## Notes

- `run_cipsi_trials(...)` supports `backend="auto|cpu_sparse|cuda_key64"`.
- `selection_mode="dense"` is removed from the scalable path; use
  `selection_mode="frontier_hash"` or `selection_mode="heat_bath"`.
- SciPy is required for sparse matrix assembly / eigensolvers in the helper layer.
- Dense `selected_ci(...)` and `GUGASelectedCISolver` are removed; the supported SCI surface is sparse trial-space construction via `run_cipsi_trials(...)`.
