# `asuka.caspt2` — Internally Contracted CASPT2

This module implements **internally contracted CASPT2** (Complete Active Space Second-Order Perturbation Theory) following the 13-case OpenMolcas formalism (C1 symmetry). It supports single-state (SS), multi-state (MS), and extended multi-state (XMS) variants, with both CPU (NumPy) and GPU (CuPy/CUDA) backends.

## Theory Overview

IC-CASPT2 expands the first-order wavefunction in an internally contracted basis built from excitation operators acting on the CASSCF reference. The basis functions are classified into **13 cases** (A through H±) depending on which orbital subspaces (inactive `i,j`, active `t,u,v`, virtual `a,b`) are involved:

| Case | Label | Active superindex | External superindex | Excitation type |
|------|-------|-------------------|---------------------|-----------------|
| 1 | A (VJTU) | `tuv` (triples) | `i` | core → active |
| 2 | B+ (VJTIP) | `t≥u` (sym pairs) | `i≥j` (sym pairs) | core → active |
| 3 | B− (VJTIM) | `t>u` (asym pairs) | `i>j` (asym pairs) | core → active |
| 4 | C (ATVX) | `tuv` (triples) | `a` | active → virtual |
| 5 | D (AIVX) | `[tu₁; tu₂]` (2×ntu block) | `a·nish+i` | core → virtual (mixed) |
| 6 | E+ (VJAIP) | `t` | `igej·nssh+a` | core → active+virtual |
| 7 | E− (VJAIM) | `t` | `igtj·nssh+a` | core → active+virtual |
| 8 | F+ (BVATP) | `t≥u` (sym pairs) | `a≥b` (sym pairs) | active → virtual |
| 9 | F− (BVATM) | `t>u` (asym pairs) | `a>b` (asym pairs) | active → virtual |
| 10 | G+ (BJATQ) | `t` | `ageb·nish+i` | core+active → virtual |
| 11 | G− (BJATM) | `t` | `agtb·nish+i` | core+active → virtual |
| 12 | H+ (BJAIP) | `a≥b` (sym pairs) | `i≥j` (sym pairs) | core → virtual (pure) |
| 13 | H− (BJAIM) | `a>b` (asym pairs) | `i>j` (asym pairs) | core → virtual (pure) |

For each case, the CASPT2 equations are solved independently after a joint S/B diagonalization that removes linear dependencies in the IC basis.

## Computational Workflow

### Single-State (SS) CASPT2

For a single root, the algorithm proceeds as:

```
1. build_superindex()          → SuperindexMap (index tables for all 13 cases)
2. build_caspt2_fock()         → CASPT2Fock (fimo, famo, fifa, epsa, e_core)
3. For each case c = 1..13:
   a. build_smat(c, ...)       → S_c   (active overlap matrix, from dm1/dm2/dm3)
   b. build_bmat(c, ...)       → B_c   (H0 active matrix, from Fock + dm1/dm2/dm3 + F3)
   c. sbdiag(S_c, B_c)         → SBDecomposition (transform T, b_diag, nindep)
   d. build_rhs(c, ...)        → V_c   (coupling vector <P|H|0>, from eri_mo)
   e. Transform to SR basis:     V_SR = T^T S V_c
   f. Build denominator:         denom = b_diag + ext_energies  (±orbital ε shifts)
4. Apply level shifts           (IPEA / imaginary / real)
5. Solve (H0 - E0)|T⟩ = -|V⟩  via direct divide or iterative PCG
6. E_PT2 = Σ_c ⟨V_c|T_c⟩     (+ shift correction if applicable)
```

Cases A and C (triples-indexed) require **F3 contractions** — Fock-weighted 4-body
quantities computed without an explicit 4-RDM, using CI-driven `E_pq` operator
applications through `F3ContractionEngine`.

### Solver Strategy

When the Fock matrix has non-zero off-diagonal blocks between orbital subspaces
(active–inactive `F_ti`, active–virtual `F_ta`, inactive–virtual `F_ia`), different
IC cases become coupled through the sigma vector. The solver automatically selects:

- **Direct divide** (`pcg_solve`): Used when all off-diagonal Fock blocks are zero
  (quasi-canonical orbitals). Each case is solved independently as `T = -V / denom`.
- **Iterative PCG** (`pcg_solve_iterative`): Used when inter-case couplings exist.
  The sigma operator (`SigmaC1CaseCoupling`) implements all 24 KOD coupling channels
  from OpenMolcas `sgm.f`, working in the covariant/contravariant `C` representation
  and transforming between `C` and `SR` (diagonalized) bases.

### Multi-State (MS) and Extended Multi-State (XMS) CASPT2

For multiple states, the workflow extends to:

```
MS-CASPT2:
  1. Run SS-CASPT2 for each state (state-specific Fock operators)
  2. Build Heff[I,J]:
     - Diagonal: Heff[I,I] = E_CASPT2(I)
     - Off-diagonal: Heff[I,J] = Σ_c hcoup_case_contribution(c, row_dots, TG1/TG2/TG3)
       where TG1/TG2/TG3 are transition RDMs between states I and J,
       and row_dots[ias,jas] = dot(RHS_raw[I,ias,:], T_raw[J,jas,:])
  3. Diagonalize Heff → MS-CASPT2 energies

XMS-CASPT2:
  1. xms_rotate_states(): diagonalize ⟨I|F_SA|J⟩ in model space → rotated CI vectors
  2. Run MS-CASPT2 with a common state-averaged Fock operator
```

## Module Structure

### Entry Point

- **`driver_asuka.py`** — Top-level ASUKA-native CASPT2 driver `run_caspt2(...)` (accepts ASUKA CASCI/CASSCF results) for SS/MS/XMS-CASPT2 end-to-end with DF + CUDA (C1, FP64).
- **`asuka/pipeline/caspt2_cuda.py`** — Runnable end-to-end pipeline (GPU DF-SCF → GPU (SA-)CASCI/(SA-)CASSCF → GPU CASPT2).

  | Parameter | Options | Controls |
  |-----------|---------|----------|
  | `pt2_backend` | `"cpu"`, `"cuda"` | S/B diag, RHS, solver, sigma |
  | `integrals_backend` | `"fulleri"`, `"df"` | MO ERI source (4-index vs DF) |
  | `fock_backend` | `"fulleri"`, `"df"` | Fock matrix construction |
  | `rdm_backend` | `"cpu"`, `"cuda"` | RDM construction |
  | `heff_backend` | `"cpu"`, `"cuda"` | MS/XMS Heff build |
  | `cuda_e2e` | `True`/`False` | Convenience: sets all to CUDA+DF |

### Core Equation Modules (CPU)

- **`superindex.py`** — `CASOrbitals`, `SuperindexMap`, and `build_superindex()`. Precomputes index mappings (active triples, pair indices with sym/asym variants, per-case dimensions `nasup`/`nisup`) needed by all 13 IC cases. Ports OpenMolcas `superindex.f` (SUPINI).

- **`fock.py`** — `CASPT2Fock` dataclass and `build_caspt2_fock()`. Constructs the MO-basis Fock matrices from full 4-index ERIs:
  - `fimo`: inactive Fock `h + J_core - 0.5·K_core`
  - `famo`: active Fock `Σ_{tu} D_{tu} [(pq|tu) - 0.5·(pt|qu)]`
  - `fifa`: full Fock `fimo + famo`
  - `epsa`: diagonal of `fifa` in the active block
  - `e_core`: `Tr[h·D_core] + 0.5·Tr[V_core·D_core] + E_nuc`

- **`fock_df.py`** — `build_caspt2_fock_df()`. DF/Cholesky-vector alternative that builds the same `CASPT2Fock` from 3-index MO pair factors (`CASPT2DFBlocks`) instead of the full `(nmo)^4` ERI tensor. The J and K contractions are performed as `L^T L` products in the auxiliary basis. Supports CuPy device execution with automatic CPU transfer of results.

- **`overlap.py`** — `SBDecomposition`, `build_smat()`, `sbdiag()`. Per-case overlap (S) matrix builders from active-space RDMs (porting `mksmat.f`), followed by joint S/B diagonalization with Molcas-style linear-dependence removal:
  1. Diagonal pre-screen by `S_ii` (threshold `THRSHN`)
  2. Diagonalize scaled S, remove small eigenvalues (threshold `THRSHS`)
  3. Form orthonormal transform `C` such that `C^T S C = I`
  4. Diagonalize `C^T B C` to get H0 eigenvalues in the orthonormal basis

- **`hzero.py`** — `build_bmat()`, `compute_e0()`. Constructs the zeroth-order Hamiltonian (H0) matrix in the IC basis. Uses Fock-weighted RDM intermediates:
  - `EASUM = Σ_w ε_w · D_{ww}` (active Fock energy, = E0)
  - `FD[t,x] = Σ_w ε_w · Γ_{tx,ww}` (Fock-weighted 2-RDM trace)
  - `FP[p,q,r,s] = 0.5 · Σ_w ε_w · Γ_{pqrs,ww}` (Fock-weighted 3-RDM half-trace)

  Cases A and C additionally require F3 (DELTA3) contractions via `F3ContractionEngine`.

- **`rhs.py`** — `build_rhs()`. Builds the RHS coupling vector `V_P = ⟨P|H|0⟩` between IC basis functions and the reference, for each of the 13 cases. Uses full MO ERIs in chemists' notation `(pq|rs)`. Ports OpenMolcas `mkrhs.f` (MKRHSA through MKRHSH). Cases A, C, D include extra one-electron corrections proportional to `fimo/nactel`.

- **`sigma.py`** — Sigma-vector operators for the CASPT2 linear system `(H0 - E0)|T⟩ = -|V⟩`:
  - `sigma_caspt2_diagonal()`: element-wise multiply in the diagonalized basis (no inter-case coupling)
  - `SigmaC1ActiveVirtualCoupling`: includes KOD=9,10 couplings (C↔F± via active–virtual Fock `F_ta`). Used when `nish=0`.
  - `SigmaC1CaseCoupling`: full 24-channel IFCOUP table from `eqsolv.F90`, handling all inter-case couplings via the `_mltsca`, `_mltmv`, `_mltdxp`, `_mltr1` tensor contraction primitives (ports of `mltsca.f`, `mltmv.f`, `mltdxp.f`, `mltr1.f`). Uses precomputed MKLIST coupling lists 1–17.

- **`solver.py`** — `pcg_solve()` (direct), `pcg_solve_iterative()` (PCG). Solves `(H0 − E0)|T⟩ = −|V⟩` for the amplitudes. The direct solver handles the diagonal case; the iterative PCG solver handles non-diagonal H0 using the diagonal as a preconditioner.

- **`f3.py`** — `CASPT2CIContext`, `F3ContractionEngine`. Computes Fock-contracted 4-body quantities (DELTA3) needed for cases A and C without explicit 4-RDM construction. The engine precomputes `|fc⟩ = (Σ_w ε_w E_{ww})|c⟩` and the first-order intermediates `T1[pq] = E_{pq}|c⟩`, then evaluates F3 elements on demand via `⟨c|E_{tu} (H_{diag} - ε_v) E_{vx} E_{yz}|c⟩` with Molcas DELTA3 correction terms applied. Results are cached per (y,z) pair.

- **`energy.py`** — `caspt2_energy_ss()`. Single-state CASPT2 energy driver that orchestrates the full per-case loop (S/B build → diag → RHS → transform → solve), applies level shifts, and computes `E_PT2 = Σ_P V_P · T_P`. Dispatches to `caspt2_energy_ss_cuda()` when `pt2_backend="cuda"`.

- **`shifts.py`** — Level-shift implementations for intruder-state removal:
  - `apply_ipea_shift()`: orbital-occupation-dependent shift to the zeroth-order Hamiltonian
  - `apply_imaginary_shift()`: `d → d + σ²/d` regularization (equivalent to `1/(d+iσ) → d/(d²+σ²)`)
  - `apply_real_shift()`: constant shift `d → d + ε`
  - `compute_shift_correction()`: amplitude-weighted energy correction `Σ_P |T_P|² · Δ_P`

### Multi-State Modules

- **`multistate.py`** — `build_heff()`, `build_heff_coupling()`, `diagonalize_heff()`. Constructs the MS-CASPT2 effective Hamiltonian:
  - Diagonal: `Heff[I,I] = E_CASPT2(I)` (SS-CASPT2 energy)
  - Off-diagonal: `Heff[I,J] = ⟨I|H|Ω_J⟩` via per-case HCOUP kernels contracted against transition RDMs (TG1/TG2/TG3) computed from CI vectors of states I and J

- **`hcoup.py`** — `hcoup_case_contribution()`. Per-case HCOUP contraction kernels porting `hcoup.f`. These are *not* symmetric for transition densities (unlike the S-metric builders in `overlap.py`). Contracts transition-density-weighted active overlap against precomputed `row_dots[ias,jas] = V₁[ias,:]·V₂[jas,:]` to yield each case's contribution to the coupling element.

- **`xms.py`** — `xms_rotate_states()`. XMS state rotation: builds the model-space H0 matrix `H0[I,J] = ⟨I|F_SA|J⟩` using transition 1-RDMs, diagonalizes it, and rotates the SA-CASSCF CI vectors. Ports OpenMolcas `xdwinit.f`.

### Data Structures

- **`result.py`** — Frozen dataclasses for results:
  - `CASPT2EnergyResult`: single-state output (`e_ref`, `e_pt2`, `e_tot`, `amplitudes`, `breakdown`)
  - `CASPT2Result`: multi-state output (lists of energies, `heff`, `ueff`, `method`)
  - `CASPT2GradResult`: gradient output (adds `grad`, Lagrangian objects `clag`/`olag`/`slag`/`wlag`)

### CUDA Backend (`cuda/`)

GPU-accelerated counterparts of the CPU modules, using CuPy and custom CUDA kernels:

- **`energy_cuda.py`** — `caspt2_energy_ss_cuda()`. End-to-end SS-CASPT2 on GPU: S/B build on CPU, RHS via DF on GPU, solver with GPU sigma operator. Optionally stores `row_dots` per case for subsequent Heff builds.
- **`rhs_df_cuda.py`** — `CASPT2DFBlocks`, `build_rhs_df_cuda()`, `build_all_rhs_df_cuda()`. DF-based RHS construction on GPU via `L^T L` contractions, avoiding the `O(nmo^4)` full ERI tensor.
- **`sigma_cuda.py`** — `SigmaC1CaseCouplingCuda`. GPU sigma-vector operator with full inter-case couplings.
- **`sb_cuda.py`** — `SBDecompositionDevice`, `precompute_fock_quantities_cuda()`. Device-side Fock-weighted quantity precomputation (`EASUM`, `FD`, `FP`).
- **`f3_cuda.py`** — `F3CudaWorkspace`. GPU-accelerated F3 contractions via EPQ-table device builds (requires GUGA CUDA extension).
- **`hcoup_cuda.py`** — `hcoup_case_contribution_cuda()`. GPU HCOUP contractions operating on device-resident `row_dots` and transition RDMs.
- **`multistate_cuda.py`** — `build_heff_cuda()`. Full GPU Heff build: transition RDMs via GPU EPQ apply-all, HCOUP contractions on device.
- **`xms_cuda.py`** — `xms_rotate_states_cuda()`. GPU-backed XMS rotation using CUDA transition dm1 construction.
- **`kernels.py`** — Low-level CUDA kernel wrappers: `apply_h0diag_sr` (denominator application), `apply_precond_sr` (PCG preconditioner), `mltsca`/`mltmv`/`mltdxp`/`mltr1` (coupling tensor contractions).

## Usage

```python
	from asuka.frontend import Molecule
	from asuka.frontend.scf import run_hf_df
	from asuka.mcscf import run_casscf
	from asuka.caspt2 import run_caspt2

mol = Molecule.from_atoms("N 0 0 0; N 0 0 1.1", unit="Angstrom", basis="cc-pvdz", cart=True, spin=0)
scf_out = run_hf_df(mol, method="rhf", backend="cuda", df=True, auxbasis="autoaux")
	casscf = run_casscf(scf_out, ncore=2, ncas=8, nelecas=8, nroots=2, root_weights=(0.5, 0.5),
	                    backend="cuda", df=True)

	# Single-state CASPT2 (SS)
	result_ss = run_caspt2(casscf, method="SS", nstates=1, iroot=0, device=0)
	print(result_ss.e_tot)  # Total SS-CASPT2 energy

	# Multi-state CASPT2 (MS)
	result_ms = run_caspt2(casscf, method="MS", nstates=2, device=0)
	print(result_ms.e_tot)  # List of MS-CASPT2 energies

	# XMS-CASPT2 (XMS)
	result_xms = run_caspt2(casscf, method="XMS", nstates=2, device=0)

	# With intruder-state removal (imaginary shift)
	result = run_caspt2(casscf, method="SS", nstates=1, device=0, imag_shift=0.2)
```

## RDM Conventions

All active-space RDMs use the **E-operator (Molcas) convention**:
- `dm1[p,q]` = ⟨E_pq⟩
- `dm2[p,q,r,s]` = ⟨E_pq E_rs⟩  (normal-ordered, *not* cumulant)
- `dm3[p,q,r,s,t,u]` = ⟨E_pq E_rs E_tu⟩

where `E_pq = Σ_σ a†_{pσ} a_{qσ}` is the spin-free unitary group generator.

Two-electron integrals use **chemists' notation**: `eri_mo[p,q,r,s]` = (pq|rs).

## References

- The 13-case IC formalism follows **OpenMolcas** (`src/caspt2/`), particularly:
  - Superindex setup: `superindex.f` (SUPINI)
  - S/B matrices: `mksmat.f` (MKSA–MKSG), `mkbmat.f` (MKBA–MKBH), `sbdiag.f`
  - F3/DELTA3: `mkfg3.f`
  - RHS vectors: `mkrhs.f` (MKRHSA–MKRHSH)
  - Sigma operator: `sigma.f`, `sgm.f`, `eqsolv.F90` (IFCOUP table), `pcg.f`
  - MS couplings: `hcoup.f`, `hefval.F90`, `mltctl.f`
  - XMS rotation: `xdwinit.f`
  - Tensor contractions: `mltsca.f`, `mltmv.f`, `mltdxp.f`, `mltr1.f`, `mklist.f`
