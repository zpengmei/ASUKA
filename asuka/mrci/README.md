# asuka.mrci

Native ASUKA multi-reference CI stack:

- uncontracted `mrcisd`
- internally contracted `ic_mrcisd`
- analytic and FD nuclear gradients on top of ASUKA SCF/CASSCF references

PySCF-facing MRCI entrypoints are intentionally not part of this package.

## Native Driver Surface

Preferred public entrypoints:

- `mrci_from_ref`
- `mrci_states_from_ref`
- `mrci_grad_from_ref`
- `mrci_grad_states_from_ref`
- `mrci_states_from_ref_soc`

Supported native energy backends:

- `integrals_backend="df_B"`
- `integrals_backend="thc"`
  Local vs global THC is inferred from `scf_out.thc_factors`.

Supported native contracted backends:

- `backend="semi_direct"`
- `backend="rdm"`

Gradient backend selector:

- `backend="analytic"`
- `backend="fd"`

When calling gradients for contracted MRCI, use `mrci_backend=...` to select
the contracted solver backend without colliding with the gradient backend.

## Current Scope

Energy / solver support:

- native-only `mrcisd` and `ic_mrcisd`
- DF, global THC, and local THC integral backends
- contracted `semi_direct` and `rdm`
- multi-root native `ic_mrcisd` in the shared-basis solver path

Analytic gradient implementation:

- native `mrcisd`
- native `ic_mrcisd`
- DF, global THC, and local THC target backends

Current hard limits:

- THC MRCI remains CUDA-only
- THC analytic gradients require gradient-capable THC metadata, supported solve
  modes, and no point downselect
- SOC remains uncontracted-only
- `+Q` remains uncontracted-only

## Validation Status

The repo now contains FD-replay tests that lock the displaced-geometry rebuild
path for SCF and CASSCF settings, plus slow CUDA analytic-vs-FD full-gradient
checks for native MRCI gradients on tiny systems.

Validated matrix on tiny LiH/STO-3G cases:

- uncontracted `mrcisd` with DF
- contracted `ic_mrcisd` with DF + `semi_direct`
- contracted `ic_mrcisd` with DF + `rdm`
- uncontracted `mrcisd` with global THC
- contracted `ic_mrcisd` with global THC + `semi_direct`
- contracted `ic_mrcisd` with global THC + `rdm`
- uncontracted `mrcisd` with local THC
- contracted `ic_mrcisd` with local THC + `semi_direct`
- contracted `ic_mrcisd` with local THC + `rdm`

Test split:

- `tests/test_mrci_grad_fd_replay_configs.py` locks the SCF/CASSCF FD replay settings.
- `tests/test_mrci_analytic_grad_fd_cuda_smoke.py` checks full analytic-vs-FD gradients for the native matrix on a CUDA runner.

The slow CUDA file is intentionally guarded by `ASUKA_RUN_SLOW_TESTS=1` and a
conservative CUDA-availability probe. In environments where `pytest` cannot
open a CUDA context cleanly, it will self-skip instead of producing false
failures.

## Module Map

| Module | Purpose |
| --- | --- |
| `driver_asuka.py` | Native end-to-end MRCI drivers |
| `grad_driver.py` | Native analytic/FD gradient driver surface |
| `grad_analytic.py` | Native analytic MRCI gradient implementation |
| `grad_fd.py` | Native finite-difference MRCI gradient replay path |
| `result.py` | Native driver/gradient result dataclasses |
| `mrcisd.py` | Uncontracted MRCISD kernels and `+Q` helpers |
| `ic_mrcisd.py` | Internally contracted MRCISD solvers |
| `ic_sigma_rdm.py` / `ic_sigma_semidirect.py` | Contracted sigma backends |
| `generalized_davidson.py` | Generalized Davidson solvers |

## Notes

- Correlated orbital ordering is `[internal/active][external]`.
- For native workflows, prefer `driver_asuka.py` / `grad_driver.py` over
  low-level kernels.

## Related Docs

- `../../docs/cleanup/mrci.md`
- `../../docs/cisd/CISD_IMPLEMENTATION.md`
