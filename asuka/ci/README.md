# asuka.ci

Spin-adapted single-reference CI utilities in the CSF/DRT basis (currently
focused on CISD).

## Public Package Exports

`asuka.ci` re-exports:

- `CISDResult`
- `CISDResultMulti`
- `GUGACISDSolver`
- `build_drt_cisd`
- `cisd`
- `cisd_kernel`

## Quick Usage

### Integrals-first functional API

```python
from asuka.ci import cisd

res = cisd(
    h1e,
    eri,
    n_occ=5,
    n_virt=8,
    nelec=10,
    twos=0,
)
print(res.e_cisd)
```

### Solver-style API

```python
from asuka.ci import GUGACISDSolver

solver = GUGACISDSolver(n_occ=5, n_virt=8, nelec=10, twos=0)
res = solver.kernel(h1e, eri)
print(res.e_cisd)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public API re-export hub |
| `cisd.py` | CISD result types, DRT builder, functional API, and solver wrapper |

## Notes

- `cisd_kernel` expects MO integrals in correlated orbital ordering
  `[occupied][virtual]`.
- Default reference inference covers closed-shell RHF (`D...D`) and high-spin
  ROHF-like (`D...D U...U`) occupied references; provide `ci_ref_occ` or
  `ref_steps_occ` for other cases.
- Internally, CISD is implemented by calling uncontracted MRCISD with
  `n_act = n_occ`.
