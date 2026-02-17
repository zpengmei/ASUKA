# Release Notes

## 2026-02-17

### AM1 CUDA polish (semiempirical)

- Added AM1 CUDA runtime preflight checks for:
  - CuPy availability
  - visible CUDA device
  - packaged kernel source presence
  - `asuka.nddo_core` contract validation
- Kept `fock_mode` public (`ri` default, `w` optional) and enforced consistent
  validation at AM1 entrypoints.
- Added `fock_mode="auto"` for CUDA mode selection based on packed-W memory.
- Hardened packaging metadata to include semiempirical runtime assets:
  - `asuka/semiempirical/data/*.json`
  - `asuka/semiempirical/gpu/cuda/*.cu`
- Added tests for:
  - CPU<->CUDA parity for `ri` and `w`
  - CUDA `ri` vs `w` internal parity
  - preflight failure modes and invalid `fock_mode`
  - packaging smoke checks for runtime assets
  - optional OpenMOPAC parity (`test_openmopac_parity.py`) using the local
    `mopac` executable as an external reference
- Added AM1 benchmark script:
  - `bench/am1_cuda_bench.py`
  - baseline artifact: `docs/bench/am1_cuda_baseline.json`
  - reports now split cold-start and warm steady-state timings
  - benchmark matrix now includes larger scaling cases:
    - `water_16`, `water_24`, `water_32`, `water_48`
- Added AM1 CUDA user/developer guide:
  - `docs/project/AM1_CUDA.md`
- Added AM1 CUDA deep-offload acceleration path:
  - new reduced pair payload export: `build_pair_ri_payload`
  - new reduced core builders:
    - `build_core_hamiltonian_from_pair_terms`
    - `core_core_repulsion_from_gamma_ss`
  - CUDA `w` setup now materializes W blocks once on device from RI payload
    (`build_wblocks_from_ri_kernel`) instead of requiring host full-`W_list`
- Optimized legacy CPU two-center integral assembly by caching per-bucket
  `np.einsum` contraction paths (`11/14/41/44`).

### AM1 analytical gradients (OpenMOPAC-aligned, CUDA hybrid)

- Added public AM1 gradient APIs:
  - `SemiempiricalCalculator.gradient(...)`
  - `SemiempiricalCalculator.energy_gradient(...)`
  - `am1_gradient(...)`
  - `am1_energy_gradient(...)`
- Gradient units are fixed to Hartree/Bohr.
- Added frozen-density pair-energy Cartesian gradient engine:
  - `asuka/semiempirical/gradient.py`
  - high-level helper: `am1_energy_gradient_scf(...)`
- Added pair helper wrappers to reduce duplicate pair assembly logic:
  - `build_pair_overlap_block(...)` in `asuka/semiempirical/overlap.py`
  - `build_pair_two_center_tensor(...)` in `asuka/semiempirical/nddo_integrals.py`
- Initial CUDA gradient behavior in this phase was correctness-first hybrid:
  - SCF on CUDA
  - gradient evaluation on CPU from converged CUDA density/matrices
  - superseded later in this release by the CUDA analytical gradient kernels
- Added test coverage:
  - API and failure-mode tests: `test_gradient_api.py`
  - directional finite-difference consistency: `test_gradient_fd_consistency.py`
  - CPU↔CUDA and CUDA `ri`↔`w` gradient parity: `test_gradient_cuda_parity.py`
  - optional OpenMOPAC gradient parity: `test_openmopac_gradient_parity.py`
- Added gradient benchmark script and baseline artifact:
  - `bench/am1_gradient_bench.py`
  - `docs/bench/am1_gradient_baseline.json`

### AM1 CUDA analytical gradient acceleration (10x program)

- Added CUDA analytical AM1 gradient backend and dispatcher:
  - `gradient_backend="auto|cuda_analytic|cpu_frozen"` on gradient APIs
  - `auto` now selects `cuda_analytic` on CUDA and falls back to
    `cpu_frozen` with actionable runtime warning
- Added CUDA analytical gradient implementation:
  - host launcher: `asuka/semiempirical/gpu/gradient_gpu.py`
  - kernels: `asuka/semiempirical/gpu/cuda/gradient_kernels.cu`
  - shared dual-number math/integral header:
    `asuka/semiempirical/gpu/cuda/pair_math.cuh`
- Added kernel loader wiring:
  - `get_gradient_kernels()`
  - `ensure_gradient_kernel_sources_available()`
- Added gradient metadata in AM1 energy+gradient details:
  - `gradient_backend_used`
  - `gradient_pack_time_s`
  - `gradient_kernel_time_s`
  - `gradient_post_time_s`
- Added/updated tests:
  - `test_gradient_cuda_analytic_parity.py`
  - `test_gradient_cuda_backend_dispatch.py`
  - CUDA parity tests now exercise explicit `cuda_analytic` vs `cpu_frozen`
- Updated gradient benchmark tooling:
  - `bench/am1_gradient_bench.py` now supports:
    - `--gradient-backend`
    - `--cases`
    - gradient sub-timer reporting
- Added pre-acceleration baseline snapshot:
  - `docs/bench/am1_gradient_cuda_baseline_pre10x.json`
