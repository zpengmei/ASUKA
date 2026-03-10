# asuka.sci

Selected CI utilities in the native GUGA/DRT CSF basis.

## Public Package Exports

`asuka.sci` re-exports:

- `run_cipsi_trials`
- `build_cipsi_trials_from_scf`
- `heat_bath_select_and_pt2_sparse`

## Quick Usage

### Functional SCI API (Small/Medium Spaces)

```python
from asuka.cuguga import build_drt
from asuka.sci.selected_ci import selected_ci

drt = build_drt(norb=8, nelec=10, twos_target=0)
res = selected_ci(drt, h1e, eri, nroots=1, init_ncsf=256, max_ncsf=20_000)
print(res.e_var[0], res.e_pt2[0], res.e_tot[0])
```

`selected_ci(...)` is a small-space reference driver. It is not the supported
path for very large CSF spaces.

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

### Solver-style API (kernel-style signature)

```python
from asuka.sci.selected_ci import GUGASelectedCISolver

solver = GUGASelectedCISolver(twos=0)
e_var, ci = solver.kernel(h1e, eri, norb=8, nelec=10, nroots=1)
print(e_var)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public API re-export hub |
| `selected_ci.py` | SCI loop, external selection/PT2 estimates, and solver wrapper |

## Notes

- `selected_ci(...)` works from an explicit `DRT` and integral tensors/objects;
  for `DFMOIntegrals` inputs it uses DF row-oracle paths automatically.
- Large-space / `key64` workflows should use `run_cipsi_trials(...)` rather than
  `selected_ci(...)`.
- `run_cipsi_trials(...)` supports `backend="auto|cpu_sparse|cuda_key64"`.
- `selection_mode="dense"` is removed from the scalable path; use
  `selection_mode="frontier_hash"` or `selection_mode="heat_bath"`.
- SciPy is required for sparse matrix assembly / eigensolvers in this module.
- `GUGASelectedCISolver.kernel(...)` returns the variational energy (`e_var`);
  PT2 diagnostics are kept in `solver.last_sci_result`.
