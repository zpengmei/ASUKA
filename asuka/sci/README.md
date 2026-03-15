# asuka.sci

Scalable SCI/CIPSI in the native GUGA/DRT CSF basis with GPU-resident algorithms throughout.

## Public API

- `run_cipsi_trials` — main CIPSI driver
- `build_cipsi_trials_from_scf` — convenience wrapper from an SCF result
- `heat_bath_select_and_pt2_sparse` — standalone HB selection + PT2

```python
from asuka.sci import run_cipsi_trials

res = run_cipsi_trials(
    drt, h1e, eri,
    backend="cuda_key64",
    nroots=1,
    init_ncsf=256,
    max_ncsf=20_000,
    selection_mode="heat_bath",
    epq_mode="no_epq_support_aware",
)
print(res.e_var[0], res.e_pt2[0], res.e_tot[0])
```

---

## Algorithm

### Basis: GUGA/DRT Configuration State Functions

The entire algorithm works in the **CSF (Configuration State Function)** basis via the
Graphical Unitary Group Approach (GUGA) and Distinct Row Table (DRT).  CSFs are exact
spin eigenfunctions (S² eigenstates), so the variational space is always spin-pure and
smaller than the determinant space by a factor that grows with the number of open shells.

A CSF is addressed by an integer index.  Its internal structure — the step vector
`(d₀, d₁, …, d_{norb-1})` where each step is Empty/Single-α/Single-β/Double — and
the associated b-values (running 2S quantum numbers at each node) are recovered from
the DRT on demand via a path-reconstruction walk.  The **connected-row oracle**
(`connected_row_sparse`, `connected_row_sparse_df`) generates all non-zero off-diagonal
Hamiltonian matrix elements `⟨J|H|I⟩` for a given CSF `|I⟩` by walking the DRT and
evaluating GUGA segment values, without ever materializing the full Hamiltonian.

The CIPSI macro-loop structure is:

```
Seed → [Variational solve → Select externals → Commit] × max_iter → PT2 correction
```

---

### Innovation 1: Matrix-Free GPU Projected Davidson

Standard CIPSI implementations build `H_sel` as an explicit `nsel × nsel` sparse matrix
and perform sparse matrix–vector products (SpMV) during Davidson.  As `nsel` grows the
matrix storage alone becomes limiting and the SpMV dominates wall time.

**Our approach eliminates the matrix entirely.**  The σ-vector `σ_i = Σ_j H_ij c_j` is
computed on the fly by directly evaluating pairwise GUGA coupling integrals for every
(i, j) pair in the selected space.

#### Pairwise CSF Materialization

Before the Davidson solve, a single CUDA kernel (`pairwise_materialize_u64_kernel`)
decodes all `nsel` selected CSFs in parallel.  Each thread reconstructs one CSF's path
from its index, and writes the step vector, DRT node path, occupation string, and
b-values into contiguous GPU arrays:

```
steps_all  [nsel, norb]      int8  — step type per orbital
nodes_all  [nsel, norb+1]    int32 — DRT node indices along the path
occ_all    [nsel, norb]      int8  — orbital occupation (0/1/2)
b_all      [nsel, norb]      int16 — 2×S at each DRT node
```

This materializes once per selected-space update; subsequent Davidson micro-iterations
reuse these arrays at no extra cost.

#### On-the-Fly H_ij Evaluation (`pairwise_compute_epq_coupling`)

For a given bra `|i⟩` and ket `|j⟩`, the GUGA coupling `⟨i|E_pq|j⟩` is evaluated
directly from the two materialized step/b vectors:

1. The steps must agree outside the orbital segment `[min(p,q), max(p,q)]`; a mismatch
   anywhere outside this window immediately returns 0.
2. Inside the window each orbital contributes a segment value computed analytically from
   the step types `(d'_k, d_k)`, the generator direction (raising/lowering), and the
   b-value of the ket at the child node.
3. The segment values are multiplied together; a zero at any orbital short-circuits the
   product.

The full two-electron matrix element `H_ij = h_1e + ½ h_2e` involves:
- **1e part**: one call to `pairwise_compute_epq_coupling` per (p,q) pair in `[start, end]`
- **2e part**: intermediate sums over a third CSF `|k⟩` that differs from `|j⟩` by at
  most one excitation (occupation-difference ≤ 4), evaluated via nested coupling lookups

#### Bucketed Target Lists

The key performance optimisation is that only CSFs within a bounded occupation-difference
window can have non-zero coupling:
- **1e (E_pq)**: `occ_diff(i, j) ≤ 2` (one creation + one annihilation)
- **2e (e_pqrs)**: `occ_diff(i, k) ≤ 2` for the intermediate `|k⟩` used in Phase 2

All `nsel` selected CSFs are pre-sorted by an occupation key, partitioning them into
buckets by occupation string.  For a given source CSF `|j⟩` the valid target range
reduces to a binary-search slice of the sorted list (typically 7–9× smaller than the
naïve `occ_diff ≤ 4` list), directly cutting Phase 2 scan volume.

The kernel `pairwise_hij_u64_kernel` assigns one CUDA block to each source `j`, computes
the **upper triangle only** (`i ≥ j`, halving the work by symmetry), and atomically
accumulates into a flat σ-vector.  A mirror kernel copies the upper triangle to the lower
triangle in a single pass.

**Scaling**: `O(nsel × (norb² + n_interm × n_1b_targets × norb))` vs the naïve oracle
approach `O(nsel × ncsf_connected × norb)`, where `ncsf_connected >> nsel` for
half-filled spaces.  No H→D transfer and no nsel² dense allocation.

#### GPU Projected Solve with Sliced-ELL SpMV

As an intermediate representation, the selected-space COO edge list can be cached on GPU
as a Sliced-ELL (SELL) matrix (`ExactSelectedTupleProjectedHop`).  The SELL format groups
rows into fixed-height slices (32 rows), zero-pads within each slice to a uniform row
width, and executes as a coalesced memory access pattern with one warp per row.  Both
single-vector (`hop`) and multi-vector (`hop_many` for multi-root Davidson) are supported.

The Davidson eigensolver runs entirely on the GPU with a shifted spectral block
preconditioner: H_sel is partitioned into contiguous blocks of ~64 CSFs, each block is
diagonalized on the CPU once per selected-space update, and the preconditioner applies
`(H_sel − θI)⁻¹ r` in the eigenbasis via two GPU GEMVs per block.

---

### Innovation 2: GPU Hash Table for External CSF Accumulation

During selection, the contribution of the variational wavefunction to every external
CSF `|J⟩ ∉ S` must be accumulated:

```
w_J += Σ_{I ∈ S} H_IJ × c_I
```

Naïvely this requires a dense array of size `ncsf` (up to 10¹¹ elements).  Instead we
use a **GPU-resident open-addressing hash table** keyed on the compact `uint64` CSF key.

- **Layout**: Two flat CuPy arrays `_hash_keys_d[cap]` (uint64) and
  `_hash_vals_d[nroots, cap]` (float64), capacity a power of 2.
- **Insertion**: Each warp processes one source CSF `|I⟩`, calls the GUGA DFS walk to
  enumerate connected `|J⟩` keys, and accumulates `H_IJ × c_I` with `atomicAdd` at the
  hash slot `(key × 6364136223846793005 + 1442695040888963407) % cap` (Knuth
  multiplicative hash), using linear probing on collision.
- **Overflow detection**: A single `_overflow_d[1]` int32 flag is set atomically by any
  thread that exhausts the probe limit.  On overflow the driver doubles the capacity and
  re-runs the kernel (`frontier_hash_max_retries` attempts).
- **Scoring**: After accumulation, `cas36_cipsi_score_pt2_compact_u64_inplace_device`
  scores every slot with the Epstein–Nesbet PT2 importance measure:
  `score(J) = |w_J|² / |E_var − H_JJ|`
  The top-`grow_by` candidates by score are transferred to CPU (only their keys and
  scores — typically O(grow_by) entries, not O(cap)) and merged into `sel`.

This approach keeps the entire accumulation on GPU with no CSF-space-sized host
allocation and no D→H transfer of the full candidate list.

---

### Innovation 3: Heat-Bath SCI with DF-Cholesky Screening

Heat-bath SCI (HB-SCI) adds a pre-screening layer that prunes the external search space
before evaluating any GUGA coupling integrals.

#### Heat-Bath Index

For each ordered pair of orbital indices (p,q) define:

```
g_pq(occ) = |h_pq^eff|  +  Σ_{rs} |⟨pq||rs⟩| × n_rs(occ)
```

where `h_pq^eff = h_pq − ½ Σ_r ⟨pr|rq⟩ n_r` is an effective one-body operator and
`n_rs = n_r × n_s` for the occupation of `|I⟩`.  The key inequality is:

```
|H_IJ × c_I|  ≤  g_pq(occ_I) × |c_I|
```

for any CSF `|J⟩` reachable from `|I⟩` by the (p→r, q→s) substitution.  Therefore:

- Any (p,q) with `g_pq × |c_max| < ε` can be skipped entirely (no matrix elements
  needed).
- For surviving (p,q) pairs, the (r,s) list is pre-sorted by `|⟨pq||rs⟩|` so the inner
  loop breaks early once `|⟨pq||rs⟩| × |c_max| < ε`.

The index stores `h1_abs[k]`, `h1_pq[k,2]`, `v_abs[k]`, `v_signed[k]`, `rs_idx[k]`,
and per-(p,q) CSR pointers `pq_ptr[pq]` — all sorted by magnitude so both loops break
as early as possible.

#### GPU-Side Index Build from DF Cholesky Factors

When `DeviceDFMOIntegrals` is available (DF integral object with GPU-resident Cholesky
factors `L[pq, P]`), the heat-bath index is built entirely on GPU:

```
⟨pq|rs⟩ = Σ_P L_{pq,P} L_{rs,P}          (standard DF reconstruction)
pq_max_v[pq] = ‖L_{pq,:}‖₂ × max_{rs} ‖L_{rs,:}‖₂   (Cauchy–Schwarz upper bound)
```

This avoids materializing the O(norb⁴) full ERI tensor.  The (r,s) list for each (p,q)
is sorted on GPU directly; the result is uploaded to the device HB index used by the
CUDA selector kernel.

#### GUGA DFS Walk Inside the Selector Kernel

The CUDA selection kernel `cas36_hb_screen_and_apply_u64_inplace_device` fuses the
outer HB screening loop and the inner GUGA CSF enumeration into a single kernel:

1. Load the source CSF `|I⟩` occupation string from the pre-materialized GPU arrays.
2. Iterate over h1 entries in descending `h1_abs` order; break when
   `h1_abs[k] × |c_max| < ε`.
3. Iterate over (p,q) pairs in descending `pq_max_v` order; break when
   `pq_max_v[pq] × |c_max| < ε`.
4. For each surviving (r,s), call the GUGA DFS walk (shared-memory DRT traversal) to
   enumerate all target CSF keys `|J⟩` and atomically insert `H_IJ × c_I` into the
   hash table.

All of this runs inside a single kernel launch; no intermediate list of (I,J) pairs is
written to global memory.

---

### Innovation 4: Sparse GPU RDM via T-Matrix + cuBLAS GEMM

For large active spaces the full dense CI vector may not fit in memory (or the memory
bandwidth to fill it dominates).  The RDM computation operates **entirely on the sparse
selected-space CI** without ever constructing the dense vector.

#### T-Matrix Formulation

Define the **T-matrix**:

```
T[pq, i]  =  Σ_j ⟨i|E_pq|j⟩ c_j      shape (norb², nsel)
```

where the sum runs only over selected CSFs `j`.  Then:

```
dm1[p, q]       =  Σ_i T[qp, i] c_i           =  (T @ c)[qp]     (cuBLAS GEMV)
dm2[p, q, r, s] =  Σ_i T[qp,i] T[rs,i]  −  δ_{qr} dm1[p,s]
                 =  (T @ T.T)[qp, rs]    −  correction            (cuBLAS GEMM)
```

This reduces a naïve O(nsel²×norb²) loop to a single `(norb², nsel) @ (norb², nsel).T`
GEMM, which saturates GPU tensor cores.

#### Bucketed Pairwise T-Matrix Kernel

The T-matrix is accumulated using the same pairwise materialized CSF data.  CSFs are
**sorted by occupation key** before accumulation so that target CSFs with small
occupation difference from the source (the majority of non-zero entries) land in
contiguous memory regions, maximising L2 cache reuse.  Bucket offsets are precomputed to
enable binary-search range reduction at the per-block level (same bucketed target list
strategy as the projected solver).

The T-matrix and CI coefficient arrays remain on GPU throughout:

```
pairwise_materialize_u64_device  →  steps/nodes/occ/b arrays on GPU
pairwise_build_bucket_data       →  sort permutation + bucket offsets on GPU
pairwise_T_matrix_bucketed_u64_device  →  T[norb², nsel] on GPU
T_d @ ci_sel_d   →  dm1  (cuBLAS GEMV, GPU)
T_d @ T_d.T      →  gram matrix → dm2  (cuBLAS GEMM, GPU)
```

No H→D or D→H transfers occur between these steps.

---

### Innovation 5: Compact key64 State Representation

For `norb ≤ 32`, each CSF's spin-up (`α`) and spin-down (`β`) occupation strings each
fit in a 32-bit word, so the full CSF is packed as a single `uint64`:

```
key64 = (α_occ_bitstring << 32) | β_occ_bitstring
```

All GPU-side operations — hash table lookup, sorting, set-membership binary search — work
on this single 64-bit word.  Compared to using the raw CSF index (`int64`):

- **Sorting**: GPU radix sort on `uint64` is native; no index-to-string decode needed.
- **Membership test**: O(1) binary search in a sorted `uint64` array; no hash collision
  for valid CSF keys (keys are unique by construction).
- **Hash key**: The key directly encodes occupation, so the HB screening kernel can read
  the occupation of an external candidate from its key without a DRT decode.

For `norb > 32` the `cuda_idx64` backend falls back to raw `int64` CSF indices with a
GPU-side hash table for membership.

---

### Innovation 6: GUGASCISolver — Sparse-First CASSCF Integration

`GUGASCISolver` (in `solver.py`) plugs SCI into the SA-CASSCF Newton driver with two
architectural choices that avoid the performance bottlenecks of a naïve integration.

#### Stub CI Vector

After `kernel()` completes, the CASSCF driver would receive the full dense CI vector
`c ∈ ℝ^ncsf` and:
1. Copy it (`c.copy()`) — triggering O(ncsf) OS page faults.
2. Upload it to GPU on every CASSCF micro-iteration via PCIe.

For CAS(22,22) with `ncsf ≈ 80B`, this would be ~640 GB of data per micro-iteration.
Instead, `kernel()` stores the sparse arrays `(_sci_sel_idx, _sci_ci_sel)` on the solver
and returns a **1-element stub** `np.ones(1)` to the CASSCF driver.  All subsequent
`make_rdm12` calls ignore the CI vector argument and use the stored sparse arrays
directly.

#### approx_kernel — Fixed-Subspace Warm-Start

During CASSCF macro-iterations the selected space is held fixed; only the integrals
change.  `approx_kernel()` calls `run_cipsi_trials` with `max_iter=0, grow_by=0`,
which skips the selection loop entirely and runs a single warm-started Davidson solve in
the existing subspace.  This replaces the O(nsel) HB-SCI growth steps with a single
short eigensolver call, typically 10–50× faster than a full `kernel()`.

---

### Streaming PT2 Correction

After the final variational solve, the Epstein–Nesbet PT2 correction is:

```
E_PT2 = Σ_{J ∉ S}  |⟨J|H|ψ_var⟩|²  /  (E_var − H_JJ)
```

`streaming_pt2_deterministic` partitions the full external space `[0, ncsf)` into
buckets of `pt2_bucket_size` CSFs.  For each bucket, contributions from all source CSFs
are accumulated via the connected-row oracle, the PT2 contribution is computed and
aggregated, and the bucket data is discarded.  Memory usage is bounded at
`O(nsel × pt2_bucket_size)` regardless of `ncsf`.

When the CUDA selector is active, `_cas36_score_pt2` scores all candidates in the hash
table in a single GPU kernel pass immediately after accumulation, avoiding a separate
CPU-side PT2 sweep.

---

## Backend Selection

| Backend | CSF label | norb limit | σ-vector | Notes |
| --- | --- | --- | --- | --- |
| `cpu_sparse` | `int64` index | unlimited | SciPy CSR SpMV | ARPACK Davidson on CPU |
| `cuda_key64` | `uint64` (α‖β) | ≤ 32 | on-the-fly pairwise GUGA | Full GPU pipeline; default for norb≤32 |
| `cuda_idx64` | `int64` index | ≤ int64 max | on-the-fly pairwise GUGA | GPU pipeline for large active spaces |

`backend="auto"` promotes to `cuda_key64` when `norb ≤ 32` and CuPy is available,
`cuda_idx64` for `norb > 32`, and `cpu_sparse` as final fallback.

---

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public API re-export hub |
| `gpu_cipsi.py` | CIPSI macro-loop driver (`run_cipsi_trials`) |
| `solver.py` | `GUGASCISolver` — drop-in fcisolver for CASSCF |
| `sparse_support.py` | Sparse oracle helpers, Davidson utilities, incremental H builder |
| `hb_selection.py` | Heat-bath selection entry point |
| `hb_integrals.py` | Heat-bath index build (CPU + GPU-DF paths) |
| `frontier_hash.py` | `SparseFrontierSelector` — row-oracle accumulation selector |
| `streaming_pt2.py` | Memory-bounded streaming PT2 / semistochastic PT2 |
| `gpu_rdm.py` | Sparse GPU RDM: T-matrix + cuBLAS GEMM |
| `sparse_rdm.py` | CPU sparse RDM fallback |
| `projected_apply.py` | Projected σ-vector operators (CSR/SELL/tuple-cached) |
