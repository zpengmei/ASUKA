# asuka.sci

Selected CI utilities in the native GUGA/DRT CSF basis.

## Public Package Exports

`asuka.sci` re-exports:

- `SCIResult`
- `selected_ci`
- `GUGASelectedCISolver`

## Quick Usage

### Functional SCI API

```python
from asuka.cuguga import build_drt
from asuka.sci import selected_ci

drt = build_drt(norb=8, nelec=10, twos_target=0)
res = selected_ci(drt, h1e, eri, nroots=1, init_ncsf=256, max_ncsf=20_000)
print(res.e_var[0], res.e_pt2[0], res.e_tot[0])
```

### Solver-style API (PySCF-like kernel signature)

```python
from asuka.sci import GUGASelectedCISolver

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
- SciPy is required for sparse matrix assembly / eigensolvers in this module.
- `GUGASelectedCISolver.kernel(...)` returns the variational energy (`e_var`);
  PT2 diagnostics are kept in `solver.last_sci_result`.
