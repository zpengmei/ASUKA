# AM1 CUDA Path (Semiempirical)

This document covers the polished AM1 CUDA path in `asuka/semiempirical`,
including the AM1 Cartesian gradient API.

## Scope and support

- Method: `AM1` only (PM7 remains scaffold-only in this tree).
- Supported element set for this cycle: `H`, `C`, `N`, `O`.
- Execution model: hybrid SCF
  - CPU precompute: overlap + reduced RI payload and core terms
  - CUDA: one-center/two-center Fock updates (`ri`/`w`) and one-time RI->W
    block materialization for `fock_mode="w"`

## Runtime requirements

1. Python dependencies for ASUKA + semiempirical stack (`numpy`, etc.)
2. CuPy installed for your CUDA runtime
3. Visible CUDA device (`cupy.cuda.runtime.getDeviceCount() > 0`)
4. Packaged CUDA kernel source files present:
   `asuka/semiempirical/gpu/cuda/fock_kernels.cu`
   `asuka/semiempirical/gpu/cuda/gradient_kernels.cu`
   `asuka/semiempirical/gpu/cuda/pair_math.cuh`
5. `asuka.nddo_core` available with required AM1 symbols

The runtime preflight now raises clear errors when any of these checks fail.

## Public API and `fock_mode`

Public API is unchanged:

- `am1_energy(..., device="cuda", fock_mode="ri|w|auto")`
- `SemiempiricalCalculator(...).energy(..., fock_mode="ri|w|auto")`
- `am1_gradient(..., device="cpu|cuda", fock_mode="ri|w|auto")`
- `am1_energy_gradient(..., device="cpu|cuda", fock_mode="ri|w|auto")`
- `SemiempiricalCalculator(...).gradient(...)`
- `SemiempiricalCalculator(...).energy_gradient(...)`
- Gradient backend selector on gradient entrypoints:
  `gradient_backend="auto|cuda_analytic|cpu_frozen"`

`fock_mode` values:

- `ri` (default): reconstructs pair tensors in-kernel from rotational invariants
- `w`: materializes full two-center W blocks on device once, then reuses them
- `auto`: selects `w` by default and falls back to `ri` when packed-W memory
  would exceed a conservative threshold

`fock_mode` is validated at all AM1 entrypoints and accepts `ri`, `w`, or `auto`.

## AM1 Cartesian gradients

- Units: gradients are always returned in `Hartree/Bohr`.
- Backends:
  - `cpu_frozen`: frozen-density pair-energy finite differences (reference)
  - `cuda_analytic`: CUDA pairwise analytical gradient kernels
  - `auto` (default): selects `cuda_analytic` on `device="cuda"` and falls
    back to `cpu_frozen` with a runtime warning on CUDA gradient failures
- For `device="cpu"`, `auto` resolves to `cpu_frozen`.

## Known limits

- Open-shell systems are not supported in this AM1 RHF driver.
- PM7 GPU corrections are not implemented.
- Gradient path supports AM1 RHF only (same H/C/N/O scope as energy path).
- Performance tracking is non-blocking in this cycle; correctness gates are primary.

## Local validation gate (required before merge)

Run all commands from repository root.

```bash
# 1) CPU/API + packaging smoke
python3 -m pytest -q \
  asuka/semiempirical/tests/test_api.py \
  asuka/semiempirical/tests/test_packaging_smoke.py \
  asuka/semiempirical/tests/test_reduced_core_payload.py

# 2) CUDA preflight + CPU<->CUDA parity + RI/W parity
python3 -m pytest -q -m cuda \
  asuka/semiempirical/tests/test_cuda_preflight.py \
  asuka/semiempirical/tests/test_cuda_parity.py

# 2b) Gradient correctness (CPU)
python3 -m pytest -q \
  asuka/semiempirical/tests/test_gradient_api.py \
  asuka/semiempirical/tests/test_gradient_fd_consistency.py

# 2c) Gradient CUDA parity
python3 -m pytest -q -m cuda \
  asuka/semiempirical/tests/test_gradient_cuda_parity.py \
  asuka/semiempirical/tests/test_gradient_cuda_analytic_parity.py

# 2d) Optional OpenMOPAC reference parity (requires `mopac` on PATH)
python3 -m pytest -q \
  asuka/semiempirical/tests/test_openmopac_parity.py \
  asuka/semiempirical/tests/test_openmopac_gradient_parity.py

# 3) Benchmark report (non-blocking perf tracking)
python3 bench/am1_cuda_bench.py \
  --output docs/bench/am1_cuda_baseline.json \
  --repeat 3 --warmup 1 --max-iter 120 --conv-tol 1e-9

# 4) Gradient benchmark report (non-blocking perf tracking)
python3 bench/am1_gradient_bench.py \
  --output docs/bench/am1_gradient_baseline.json \
  --gradient-backend auto \
  --cases water_16,water_24,water_32 \
  --repeat 3 --warmup 1 --max-iter 120 --conv-tol 1e-9
```

Benchmark output includes both:

- `cold_sample`: first-run timing (includes JIT/runtime startup)
- `warm_summary`: steady-state timing after discarded warmup runs

Current benchmark matrix includes small/medium molecules and larger water
clusters (`water_16`, `water_24`, `water_32`) for CUDA scaling in the
analytical gradient acceleration gate.

## Numerical acceptance targets

- CPU vs CUDA energy parity:
  - `|E_total| <= 1e-6 Ha`
  - `|E_electronic| <= 1e-6 Ha`
  - `|E_core| <= 1e-6 Ha`
- Matrix parity:
  - `max_abs(F/P diff) <= 1e-5`
  - `fro_norm(F/P diff) <= 1e-4`
- Convergence:
  - both converged
  - iteration count delta `<= 3`
- CUDA run-to-run stability:
  - total-energy spread `<= 1e-6 Ha`
