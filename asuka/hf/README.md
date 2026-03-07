# asuka.hf

HF SCF helpers for standalone workflows (DF and dense AO-ERI paths).

## Entry points
- `rhf_df(...)`, `rohf_df(...)`, `uhf_df(...)`
- `rhf_dense(...)`, `rohf_dense(...)`, `uhf_dense(...)`

## THC (LS-THC-Style Build) and Local (LS-)THC (Equations + Implementation Map)

ASUKA supports:
- **global THC (LS-THC-style factor build)**: one set of factors `(X,Y[,Z])` for the full AO space
- **local (LS-)THC / local THC**: many blocks with local AO subsets and local point sets, assembled with ownership masking

Key capabilities shared across both backends:
- **Y-first** contractions where the central metric `Z` is represented as `Z = Y @ Y.T` and `Z` never needs to be materialized.
- **Low-rank / factored densities** `D = U @ V.T` to avoid dense AO densities when the rank is small (common in SCF, gradients, and pair-density batches).
- A **mixed-precision** (TF32/FP32) path based on caching balanced `X/Y` and running tensor-core GEMMs on `Delta D` in SCF, with periodic **rebasing**.

Literature background (what the baseline math/algorithms come from)
- THC-DF / THC ansatz for ERIs: Hohenstein, Parrish, and Martinez (2012) [1]
- LS-THC (least-squares renormalization / grid-based THC factor construction): Parrish et al. (2012) [2]
- Local (LS-)THC block partitioning and ownership masking ideas: Song and Martinez (2017) [3]

ASUKA engineering additions beyond the baseline literature
- **Y-first compute graph**: treat `Z` as optional, apply the metric as `Y(Y^T m)` and build `Z[:,q]` blocks as `Y @ Yq.T` on demand.
- **`store_Z=False`**: allow factor builds that never materialize the dense `O(npt^2)` `Z` matrix.
- **Low-rank density APIs**: accept `D = U V^T` for both global and local THC, so exchange avoids the dense `D @ Xq.T` multiply.
- **No-AO-batches local kernels**: batched MO contractions for Coulomb-like and exchange-like local THC paths without forming `(nbatch,nao,nao)` intermediates.
- **Mixed precision + SCF control loop**: TF32/FP32 hot GEMMs on cached, balanced `X/Y`, applied to `DeltaD` with periodic FP64 rebasing.
- **Exact point-gauge balancing** before casting `X/Y` to FP32 (exact invariance of the represented ERIs).

Notation
- `nao`: number of AOs
- `npt`: number of THC grid points
- `naux`: number of auxiliary basis functions used for the metric build
- Shapes:
  - `X`: (npt, nao) weighted AO collocation in point-major order
  - `Y`: (npt, naux) such that `Z = Y @ Y.T`
  - `Z`: optional dense (npt, npt) central metric
  - `D`: (nao, nao) AO density (typically symmetric)
  - `U, V`: (nao, r) low-rank density factors with `D = U @ V.T`

### THC ERI Approximation

Global THC approximates AO ERIs as:

```text
(mu nu | la si) ~= sum_{P,Q} X[P,mu] X[P,nu] Z[P,Q] X[Q,la] X[Q,si]
```

This is the THC-DF / THC ansatz introduced in [1] and refined into LS-THC in [2].

ASUKA stores `Y` such that:

```text
Z[P,Q] = sum_L Y[P,L] Y[Q,L]   (i.e. Z = Y @ Y.T)
```

### Coulomb J[D]

Define the pointwise scalar:

```text
m[P] = sum_{mu,nu} X[P,mu] D[mu,nu] X[P,nu] = (X D X^T)[P,P]
```

Central-metric action (two equivalent options):

```text
Z-first:   n = Z @ m
Y-first:   n = Y @ (Y.T @ m)     (since Z = Y Y^T)
```

Then:

```text
J[mu,nu] = sum_P X[P,mu] n[P] X[P,nu]
        = X.T @ diag(n) @ X
```

**Factored density (low-rank) `D = U V^T`:**

```text
P = X @ U   (npt,r)
Q = X @ V   (npt,r)
m[P] = sum_k P[P,k] Q[P,k]
n = Z@m   or   Y(Y^T m)
J = X.T @ diag(n) @ X
```

Implementation
- Global J: `asuka/hf/thc_jk.py::thc_J(...)`, `thc_J_factored(...)`
- Local J assembly: `asuka/hf/local_thc_jk.py::local_thc_J(...)`, `local_thc_J_factored(...)`
- Local MO-batched low-rank J contraction (no dense AO matrices): `asuka/hf/local_thc_jk.py::local_thc_J_factored_mo_batched(...)`

### Exchange K[D] (Blocked Form)

Let:

```text
M = X D X^T   (npt,npt)
K = X^T ( (Z ⊙ M) X )
```

ASUKA uses a point-blocked algorithm over `q` (to avoid materializing `M`):

```text
for q in blocks:
  Xq = X[q0:q1,:]                    (qb,nao)
  Aq = D @ Xq.T                      (nao,qb)
  Mq = X @ Aq                        (npt,qb)   # == M[:, q0:q1]

  Zblk = Z[:, q0:q1]                 (npt,qb)   # Z-first
  Zblk = Y @ Yq.T , Yq = Y[q0:q1,:]  (npt,qb)   # Y-first (no dense Z load)

  Tq  = Zblk ⊙ Mq                    (npt,qb)
  Fq  = X.T @ Tq                     (nao,qb)
  K  += Fq @ Xq                      (nao,nao)
```

**Factored density (low-rank) `D = U V^T` removes the dense `D @ Xq.T` multiply:**

```text
P = X @ U  (npt,r)
Q = X @ V  (npt,r)
Mq = P @ Qq.T    where Qq = Q[q0:q1,:]   (qb,r)
```

Implementation
- Global K: `asuka/hf/thc_jk.py::thc_K_blocked(...)`, `thc_K_blocked_factored(...)`
- Local K assembly: `asuka/hf/local_thc_jk.py::local_thc_K_blocked(...)`, `local_thc_K_blocked_factored(...)`
- Local MO-batched low-rank K contraction (no dense AO matrices): `asuka/hf/local_thc_jk.py::local_thc_K_factored_mo_batched(...)`
- Convenience wrapper for rank-2 pair densities: `asuka/hf/local_thc_jk.py::local_thc_K_pairs_mo_batched(...)`

### Exact Point-Gauge Freedom (Used for Balancing Before FP32/TF32)

THC has an exact per-point row-scaling freedom:

```text
X' = S X
Y' = S^{-2} Y
```

with diagonal `S = diag(s[P])`. This transformation is exact (it does not change the represented ERIs):

```text
X'[P,mu] X'[P,nu] Z'[P,Q] X'[Q,la] X'[Q,si]
= X[P,mu]  X[P,nu]  Z[P,Q]  X[Q,la]  X[Q,si]
```

ASUKA uses this to **balance magnitudes** before casting cached `X/Y` to FP32:

```text
s[P] ~= ( ||Y[P,:]|| / (||X[P,:]|| + eps) )^(1/3)    (with clipping)
```

Implementation
- Cache + balancing: `asuka/hf/thc_tc.py`

### Local (LS-THC) Block Ownership Masking

Each local block defines a local AO ordering:

```text
[ early secondary | primary | late secondary ]
```

This follows the local LS-THC block/ownership idea in [3], adapted to ASUKA's
global AO-matrix assembly (mask locally, then scatter-add).

Ownership masking rules (applied to each block contribution before scatter-add):
- Zero any output involving **early secondary** indices.
- Zero the **late-late** output sub-block (owned by later blocks).

This keeps only:
- primary-primary
- primary-late
- late-primary

Implementation
- Mask helper: `asuka/hf/local_thc_jk.py::_mask_owned_outputs_inplace(...)`

### "Don't Form AO Batches" (MO-Contracted Local Apply)

For Coulomb-like operators of the form:

```text
V = X^T diag(n) X
```

MO contraction can be written without forming `V`:

```text
C_L^T V C_R = (X C_L)^T diag(n) (X C_R)
```

For **rank-2 pair densities** (common in gradient batches),

```text
D_b = 0.5 * (cL_b cR_b^T + cR_b cL_b^T)
m_b = (X cL_b) ⊙ (X cR_b)
n_b = Y(Y^T m_b)   (or Z m_b)
```

Local LS-THC adds the block-ownership mask; ASUKA computes the masked MO contraction
directly in point space (per block) rather than materializing:
- `D_batch` with shape (nbatch, nao, nao)
- `V_batch` with shape (nbatch, nao, nao)

Implementation
- Pair Coulomb MO-batched (no dense AO batches): `asuka/hf/local_thc_jk.py::local_thc_eri_apply_pairs_mo_batched(...)`
- Factored Coulomb MO-batched (general rank): `asuka/hf/local_thc_jk.py::local_thc_J_factored_mo_batched(...)`
- Factored exchange MO-batched (general rank): `asuka/hf/local_thc_jk.py::local_thc_K_factored_mo_batched(...)`

### Mixed Precision (TF32/FP32) Policy and SCF Delta-D Rebasing

ASUKA's mixed-precision THC uses:
- Factor build/validation in FP64
- Cached balanced `X/Y` in FP32, with TF32 GEMMs enabled where possible
- Output accumulation in FP64

In SCF, the mixed-precision path is applied to:

```text
DeltaD = D - D_ref
J(D) ~= J_ref + J(DeltaD)
K(D) ~= K_ref + K(DeltaD)
```

To keep the hot contractions well-conditioned, ASUKA **rebases** periodically:

```text
if ||DeltaD||_F / ||D||_F > rebase_dD_rel_tol and cycle >= rebase_min_cycle:
  (dJ,dK) = THC_JK_fp64(DeltaD)
  J_ref += dJ; K_ref += dK; D_ref = D
else:
  (dJ,dK) = THC_JK_tf32(DeltaD)   # cached X_tc/Y_tc
  J = J_ref + dJ; K = K_ref + dK
```

Implementation
- Global THC SCF: `asuka/hf/thc_scf.py` (`mp_mode="tf32"`, `rebase_*`)
- Local THC SCF: `asuka/hf/local_thc_scf.py`

### Benchmarks / Tests

Benchmarks
- Global Y-first + TF32 cache microbench: `benchmark/scripts/bench_thc_yfirst_tf32.py`
- Local low-rank + MO-batched kernels: `benchmark/scripts/bench_local_thc_lowrank.py`
- SCF comparison fp64 vs tf32 modes: `benchmark/scripts/compare_thc_scf_mp_modes.py`

Tests
- Global: `tests/test_thc_jk_factored.py`
- Local MO-batched kernels: `tests/test_local_thc_pair_mo_apply.py`

### References

[1] E. G. Hohenstein, R. M. Parrish, and T. J. Martinez, "Tensor hypercontraction density fitting. I. Quartic scaling second- and third-order Moller-Plesset perturbation theory," J. Chem. Phys. 137, 044103 (2012). DOI: 10.1063/1.4732310.

[2] R. M. Parrish, E. G. Hohenstein, T. J. Martinez, and C. D. Sherrill, "Tensor hypercontraction. II. Least-squares renormalization," J. Chem. Phys. 137, 224106 (2012). DOI: 10.1063/1.4768233.

[3] C. Song and T. J. Martinez, "Atomic orbital-based SOS-MP2 with tensor hypercontraction. II. Local tensor hypercontraction," J. Chem. Phys. 146, 034104 (2017). DOI: 10.1063/1.4973840.
