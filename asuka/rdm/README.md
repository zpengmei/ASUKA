# asuka.rdm

Streaming reduced-density-matrix (RDM) helpers for large CSF spaces. The main
APIs avoid storing large intermediate tensors in RAM and can spill to memmap
workspace when needed.

## Public Package Exports

`asuka.rdm` re-exports:

- `make_rdm12_streaming(...)`
- `trans_rdm12_streaming(...)`

Additional specialized APIs (for example dm1-only and batched transition dm1)
live in `asuka.rdm.stream`.

## Quick Usage

### Build state `(dm1, dm2)` from one CI vector

```python
from asuka.rdm import make_rdm12_streaming

dm1, dm2 = make_rdm12_streaming(
    drt,
    civec,
    max_memory_mb=4000.0,
    block_nops=8,
)
```

### Build transition `(dm1, dm2)` between bra/ket vectors

```python
from asuka.rdm import trans_rdm12_streaming

tdm1, tdm2 = trans_rdm12_streaming(drt, ci_bra, ci_ket, reorder=True)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public package export surface |
| `stream.py` | Streaming builders for state/transition dm1/dm2 and dm1-only batched paths |
| `rdm123.py` | Internal reorder/convention helpers for dm1/dm2/dm3 tensors |
| `accumulator.py` | Dense row-accumulator scratch helper (`DenseRowAccumulator`) |
| `contract4pdm.py` | Internal 4-PDM contraction helpers used by higher-level workflows |

## Notes

- `make_rdm12_streaming` and `trans_rdm12_streaming` use the same spin-free
  conventions as the surrounding cuguga/contract workflows.
- For large problems, set `max_memory_mb` and optionally `tmpdir` to control
  memmap workspace behavior.
- `trans_rdm12_streaming(..., reorder=False)` returns raw `<E_pq E_rs>` ordering
  without delta correction.

