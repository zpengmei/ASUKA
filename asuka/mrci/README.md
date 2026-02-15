# asuka.mrci

Multi-reference CI stack for ASUKA: uncontracted MRCISD, internally contracted
ic-MRCISD, and supporting helpers.

## Public Package Exports

Top-level `asuka.mrci` exports include:

- Uncontracted solver APIs: `mrcisd_kernel`, `mrcisd_plus_q`,
  `mrcisd_virtual_weights`, `build_drt_mrcisd`, `embed_cas_ci_into_mrcisd`
- Contracted solver APIs: `ic_mrcisd_kernel`, `ICMRCISDResult`
- Frozen-core helpers: `frozen_core_from_eri4`, `FrozenCoreMOIntegrals`
- Generalized Davidson: `generalized_davidson1`, `GeneralizedDavidsonResult`
- IC basis / overlap / sigma helpers re-exported from `ic_basis`, `ic_overlap`,
  `ic_sigma_rdm`, and `ic_sigma_semidirect`

Driver helpers (`mrci_from_mc`, gradients, multi-state utilities) remain in
submodules (`driver.py`, `grad_driver.py`) and are intentionally not re-exported
at package root.

## Quick Usage

### Run low-level MRCISD kernel

```python
from asuka.mrci import mrcisd_kernel
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Package-level re-export hub |
| `driver.py` | Driver helpers (`mrci_from_mc`, multi-state variants) |
| `grad_driver.py` | Gradient drivers (FD + analytic paths, state/root handling) |
| `mrcisd.py` | Uncontracted MRCISD kernels and +Q utilities |
| `ic_mrcisd.py` | Internally contracted MRCISD solver |
| `frozen_core.py` | Frozen-core effective integrals and shifts |
| `generalized_davidson.py` | Generalized Davidson for non-orthogonal eigenproblems |
| `ic_basis.py` | IC basis enumeration/filtering helpers |
| `ic_overlap.py` | Overlap application in contracted bases |
| `ic_sigma_rdm.py` / `ic_sigma_semidirect.py` | Contracted sigma builders (RDM/semi-direct backends) |
| `grad_analytic.py` | Analytic MRCISD gradient implementation details |

## Notes

- Correlated orbital ordering for kernels is `[internal/active][external]`.
- `integrals_backend="cueri_df"` in `driver.py::mrci_from_mc` currently supports only
  uncontracted `method="mrcisd"` and requires `hop_backend="cuda"`.
- `integrals_backend="pyscf_eri4"` is maintained mainly for parity/benchmark
  workflows.
- `+Q` correction support is currently tied to uncontracted `mrcisd`.

## Related Docs

- `../../docs/cleanup/mrci.md`
- `../../docs/cisd/CISD_IMPLEMENTATION.md`
