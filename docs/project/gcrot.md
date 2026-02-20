# ASUKA Z-Vector: GPU GCROTMK Consolidated Plan

## 1) Current Baseline in ASUKA

This plan reflects the current implementation state in the repository:

- `solve_mcscf_zvector(...)` supports `method="gmres"` and `method="gcrotmk"` in `asuka/mcscf/zvector.py`.
- `MCSCFHessianOp` already exposes `gpu_mode` and the Newton Hessian matvec path can operate on CuPy vectors when the operator is device-native.
- `contract_2e(...)` and `trans_rdm12(...)` in `asuka/solver.py` already accept CuPy inputs in CUDA backends and support device outputs via `return_cupy`.
- GPU-native GMRES already exists (`_gmres_solve_gpu`), while GCROTMK is currently CPU/SciPy-only.
- `nuc_grad_df.py` currently defaults to GMRES for GPU Hessian workflows and previously forced explicit `gcrotmk` requests back to GMRES.

## 2) Gap Definition

The primary missing capability is:

- A GPU-capable GCROTMK Krylov implementation with recycle-space support, integrated into `solve_mcscf_zvector(...)` when `hessian_op.gpu_mode=True`.

Not in scope:

- New public API arguments for backend selection.
- Changes to `MCSCFHessianOp` surface.
- Re-architecting Newton Hessian or CI kernels (already largely xp-aware for this use case).

## 3) Final Design Decisions

### Public API stability

- Keep `solve_mcscf_zvector(...)` signature unchanged.
- Keep `MCSCFHessianOp` unchanged.
- Keep `recycle_space` call contract unchanged; in GPU mode it may hold CuPy vectors internally.

### Backend dispatch policy

- `method="gcrotmk"` uses GPU GCROTMK when `op.gpu_mode` is `True`.
- Otherwise, it uses the existing SciPy CPU GCROTMK path.
- If GPU GCROTMK is selected but CuPy is unavailable, fall back to CPU GCROTMK and annotate:
  - `info["backend"] = "cpu"`
  - `info["fallback_reason"] = "cupy_unavailable"`

### Solver info normalization

For GCROTMK results:

- CPU path:
  - `info["solver"] = "gcrotmk"`
  - `info["backend"] = "cpu"`
- GPU path:
  - `info["solver"] = "gcrotmk_gpu"`
  - `info["backend"] = "cuda"`

## 4) File-by-File Implementation Plan

### A. `asuka/cuda/krylov_gcrotmk.py` (new)

Add a backend-agnostic GCROTMK implementation:

- Entry point:
  - `gcrotmk_xp(matvec, b, x0=None, rtol=..., atol=..., maxiter=..., M=None, callback=None, m=20, k=None, CU=None, discard_C=False, truncate="oldest", xp=None) -> (x, info)`
- Design:
  - Vector operations in `xp` (`numpy` or `cupy`).
  - Small Hessenberg/Givens linear algebra in NumPy.
  - In-place `CU` mutation for recycling across solves.
  - Stable residual checks with explicit residual recomputation near convergence.
- Required behavior:
  - `truncate="oldest"` fully supported (current ASUKA usage).

### B. `asuka/mcscf/zvector.py`

1. Keep existing SciPy GCROTMK path and add standardized CPU metadata (`backend="cpu"`).
2. Add `_gcrotmk_solve_gpu(...)` wrapper:
   - Converts RHS / initial guess / recycle vectors to CuPy.
   - Builds CuPy preconditioner from `diag_precond` when provided.
   - Uses user preconditioner if provided (accepting GPU-native or host-returning implementations).
   - Calls `asuka.cuda.krylov_gcrotmk.gcrotmk_xp(..., xp=cp)`.
   - Returns NumPy solution and solver info with `solver="gcrotmk_gpu"`, `backend="cuda"`.
3. In `solve_mcscf_zvector(...)`, route GCROTMK via a dispatch helper:
   - Use GPU GCROTMK when `op.gpu_mode=True`.
   - Otherwise use CPU GCROTMK.
   - On CuPy import failure in GPU branch, fall back to CPU GCROTMK and annotate `fallback_reason`.
4. Keep GMRES logic unchanged.

### C. `asuka/mcscf/nuc_grad_df.py`

Update method-selection policy:

- Preserve conservative `auto` behavior:
  - GPU Hessian mode still defaults to GMRES under `ASUKA_ZVECTOR_METHOD=auto`.
- Remove hard override that rewrote explicit `gcrotmk` requests to GMRES.
- Result: explicit `ASUKA_ZVECTOR_METHOD=gcrotmk` is now respected in GPU mode.

### D. `docs/project/gcrot.md`

Replace long mixed design/code dump with this consolidated implementation plan:

1. Current baseline
2. Gap definition
3. Final design decisions
4. File-by-file implementation plan
5. Validation matrix
6. Rollout and risk controls

## 5) Validation Matrix

### Unit tests (CPU)

- `gcrotmk_xp(xp=np)` converges for nonsymmetric random systems.
- Solution and/or residual quality is close to SciPy `gcrotmk` under matched tolerances.
- Recycle space (`CU`) is reused/mutated across sequential solves.
- `truncate="oldest"` keeps recycle space size bounded by `k`.

### Unit tests (CUDA)

- `gcrotmk_xp(xp=cp)` converges with CuPy vectors and CuPy matvec.
- GPU recycle vectors remain device arrays.
- `solve_mcscf_zvector(..., method="gcrotmk")` with `hessian_op.gpu_mode=True` reports `solver="gcrotmk_gpu"`, `backend="cuda"` and converges.

### Integration checks

- Existing GMRES and CPU GCROTMK workflows continue to run unchanged.
- In `nuc_grad_df`, explicit `ASUKA_ZVECTOR_METHOD=gcrotmk` is no longer forcibly rewritten to GMRES.

## 6) Rollout and Risk Controls

### Rollout strategy

1. Land `asuka/cuda/krylov_gcrotmk.py` + unit tests.
2. Land `zvector.py` dispatch and fallback.
3. Land `nuc_grad_df.py` policy update.
4. Keep defaults conservative (`auto` still prefers GMRES on GPU).

### Key risks and mitigations

- **Risk:** CuPy unavailable in environments with `gpu_mode=True`.
  - **Mitigation:** explicit CPU fallback with metadata (`fallback_reason`).
- **Risk:** recycle-space type drift (NumPy vs CuPy).
  - **Mitigation:** convert/mutate recycle vectors consistently at GPU solver entry.
- **Risk:** regression in existing CPU workflows.
  - **Mitigation:** no public API changes; CPU GCROTMK path retained.

### Acceptance criteria

- No public Z-vector signature changes.
- Existing tests continue to pass.
- New CPU/CUDA GCROTMK tests pass.
- GPU GCROTMK converges with residual behavior comparable to CPU for matched tolerances.
- Recycle space is functional across repeated solves.
