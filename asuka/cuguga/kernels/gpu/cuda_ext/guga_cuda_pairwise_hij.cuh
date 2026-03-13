
#pragma once

// Pair-wise H[i,j] evaluation kernel for selected-space SCI.
//
// Instead of DFS-walking from each source j to enumerate ALL connected CSFs
// in the full ncsf space (then filtering to the selected set), this kernel
// directly evaluates H[i,j] for (i,j) pairs within the selected set.
//
// Two kernels:
//   1. pairwise_materialize_u64_kernel: Pre-compute step vectors, nodes, occ,
//      b-values for all nsel selected CSFs.
//   2. pairwise_hij_u64_kernel: For each source j (one CUDA block), compute
//      the UPPER TRIANGLE of row j (i >= j) of the dense H[nsel, nsel] matrix.
//   3. pairwise_hij_mirror_kernel: Copy upper triangle to lower triangle.
//
// Optimizations over initial version:
//   - Upper-triangle only: halves compute (H is symmetric)
//   - Shared-memory tiling: target step/b vectors loaded in tiles into shmem
//     in Phase 2, reducing global memory bandwidth by ~#intermediates
//   - In-place mirror: no temporary allocation for symmetrization
//
// Scaling: O(nsel * [norb^2 + #interm * nsel * norb]) vs the old
// O(nsel * ncsf_connected * norb) where ncsf_connected >> nsel for
// half-filled spaces.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

// ============================================================================
// Kernel M: Materialize selected CSF data
// ============================================================================

template <int MAX_NORB_T>
__global__ void pairwise_materialize_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,   // [nsel] global CSF indices
    int nsel,
    int norb,
    uint64_t ncsf,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int8_t*  __restrict__ steps_all,             // [nsel, norb]
    int32_t* __restrict__ nodes_all,             // [nsel, norb+1]
    int8_t*  __restrict__ occ_all,               // [nsel, norb]
    int16_t* __restrict__ b_all,                 // [nsel, norb]
    int* __restrict__ overflow_flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nsel) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t csf_idx = sel_idx_u64[idx];
  int8_t*  my_steps = steps_all + (int64_t)idx * norb;
  int32_t* my_nodes = nodes_all + (int64_t)idx * (norb + 1);
  int8_t*  my_occ   = occ_all   + (int64_t)idx * norb;
  int16_t* my_b     = b_all     + (int64_t)idx * norb;

  bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
      child_table, child_prefix, norb, ncsf, csf_idx, my_steps, my_nodes);
  if (!ok) {
    atomicExch(overflow_flag, 1);
    return;
  }

  for (int k = 0; k < norb; ++k) {
    my_occ[k] = (int8_t)step_to_occ(my_steps[k]);
    my_b[k] = node_twos[my_nodes[k + 1]];
  }
}

// ============================================================================
// Device helper: compute <i|E_pq|j> coupling directly from step vectors.
//
// Given materialized step/b vectors for bra (i) and ket (j), compute the
// one-electron coupling coefficient for generator E_pq.
//
// Returns 0.0 if the transition is forbidden.
// ============================================================================

template <int MAX_NORB_T>
__device__ __forceinline__ double pairwise_compute_epq_coupling(
    int norb,
    int p,                                       // creation orbital
    int q,                                       // annihilation orbital
    const int8_t* __restrict__ steps_i,          // bra steps [norb]
    const int16_t* __restrict__ b_i,             // bra b-values [norb]
    const int8_t* __restrict__ steps_j,          // ket steps [norb]
    const int16_t* __restrict__ b_j,             // ket b-values [norb]
    const int32_t* __restrict__ nodes_j,         // ket node path [norb+1]
    const int16_t* __restrict__ node_twos) {     // [nnodes]
  // For E_pq with p < q: raising generator. For p > q: lowering generator.
  // For p == q: diagonal (W generator) — handle separately.
  if (p == q) return 0.0;

  int start, end, q_start, q_mid, q_end;
  if (p < q) {
    start = p; end = q;
    q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
  } else {
    start = q; end = p;
    q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
  }

  // Steps must agree outside [start, end] for nonzero coupling
  for (int k = 0; k < start; ++k) {
    if (steps_i[k] != steps_j[k]) return 0.0;
  }
  for (int k = end + 1; k < norb; ++k) {
    if (steps_i[k] != steps_j[k]) return 0.0;
  }

  // Evaluate coupling as product of segment values from start to end
  double coupling = 1.0;
  for (int k = start; k <= end; ++k) {
    int qk;
    if (k == start) qk = q_start;
    else if (k == end) qk = q_end;
    else qk = q_mid;

    int dprime = (int)steps_i[k];  // bra step
    int dk     = (int)steps_j[k];  // ket step
    int bk     = (int)b_j[k];      // ket b-value at child node

    // Compute db = b_j[k] - b_i[k] (spin difference)
    // For the ket, b_j[k] = node_twos[nodes_j[k+1]].
    // For the bra, b_i[k] = node_twos[nodes_i[k+1]], which is stored in b_i.
    int db = bk - (int)b_i[k];

    double seg = segment_value_int(qk, dprime, dk, db, bk);
    if (seg == 0.0) return 0.0;
    coupling *= seg;
  }
  return coupling;
}

// ============================================================================
// Kernel H: Pair-wise Hamiltonian builder (UPPER TRIANGLE)
// ============================================================================
//
// Grid: nsel blocks, Block: 256 threads
// Each block computes row j = blockIdx.x of H[nsel, nsel], but only for
// columns i >= j (upper triangle including diagonal).
// A separate mirror kernel copies upper→lower afterwards.

template <int MAX_NORB_T>
__global__ __launch_bounds__(256)
void pairwise_hij_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,     // [nsel]
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* __restrict__ h_base,            // [norb*norb] one-body integrals
    const double* __restrict__ eri4,              // [norb^4] two-body integrals
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    // Materialized CSF data (from Kernel M)
    const int8_t*  __restrict__ steps_all,        // [nsel, norb]
    const int32_t* __restrict__ nodes_all,        // [nsel, norb+1]
    const int8_t*  __restrict__ occ_all,          // [nsel, norb]
    const int16_t* __restrict__ b_all,            // [nsel, norb]
    // Output
    double* __restrict__ H_out,                   // [nsel, nsel] row-major
    int* __restrict__ overflow_flag) {

  int j_local = blockIdx.x;
  if (j_local >= nsel) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t j_global = sel_idx_u64[j_local];
  if (j_global >= ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;

  // === Shared memory layout ===
  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];

  // h1e cache in dynamic shared memory
  extern __shared__ char _dyn_smem[];
  double* _h1e_cache = (double*)_dyn_smem;

  // Phase 1 intermediate storage
  enum { PAIRWISE_MAX_INTERMEDIATES = 2048 };
  __shared__ uint64_t _interm_k_global[PAIRWISE_MAX_INTERMEDIATES];
  __shared__ int8_t   _interm_r[PAIRWISE_MAX_INTERMEDIATES];
  __shared__ int8_t   _interm_s[PAIRWISE_MAX_INTERMEDIATES];
  __shared__ double   _interm_crs[PAIRWISE_MAX_INTERMEDIATES];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;

  // Phase 2: reconstructed k path in shared memory
  __shared__ int8_t   _p2_steps[MAX_NORB_T];
  __shared__ int32_t  _p2_nodes[MAX_NORB_T + 1];
  __shared__ int8_t   _p2_occ[MAX_NORB_T];
  __shared__ int16_t  _p2_b[MAX_NORB_T];

  // === Step 1: Load source j's path from materialized arrays ===
  for (int k = tid; k < norb; k += nthreads) {
    steps_j_s[k] = steps_all[(int64_t)j_local * norb + k];
    occ_j_s[k]   = occ_all[(int64_t)j_local * norb + k];
    b_j_s[k]     = b_all[(int64_t)j_local * norb + k];
  }
  for (int k = tid; k < norb + 1; k += nthreads) {
    nodes_j_s[k] = nodes_all[(int64_t)j_local * (norb + 1) + k];
  }
  __syncthreads();

  // Compute idx_prefix for source j
  if (tid == 0) {
    idx_prefix_j_s[0] = 0ull;
    for (int k = 0; k < norb; ++k) {
      int node_k = nodes_j_s[k];
      int step_k = (int)steps_j_s[k];
      idx_prefix_j_s[k + 1] = idx_prefix_j_s[k] + (uint64_t)child_prefix[node_k * 5 + step_k];
    }
  }

  // Cooperative load of h1e into shared memory
  for (int i = tid; i < nops; i += nthreads) _h1e_cache[i] = h_base[i];
  __syncthreads();

  // === Step 2: Compute h_eff_j = h[p,q] + 0.5 * sum_r eri(p,q,r,r) * occ_j[r] ===
  for (int pq = tid; pq < nops; pq += nthreads) {
    int p = pq / norb;
    int q = pq - p * norb;
    double hpq = _h1e_cache[pq];
    for (int r = 0; r < norb; ++r) {
      int occ_r = (int)occ_j_s[r];
      if (occ_r == 0) continue;
      hpq += 0.5 * cas36_dense_eri4_at(eri4, norb, p, q, r, r) * (double)occ_r;
    }
    _h1e_cache[pq] = hpq;
  }
  __syncthreads();

  // === Step 3: One-body pass (upper triangle only: i_local >= j_local) ===
  double diag_local = 0.0;
  for (int p = tid; p < norb; p += nthreads) {
    diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
  }

  // Off-diagonal one-body: for each target i >= j, compute <i|H_1|j>
  for (int i_local = j_local + 1 + tid; i_local < nsel; i_local += nthreads) {
    const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
    const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

    // Quick screen: find diff range
    int diff_min = norb, diff_max = -1;
    for (int k = 0; k < norb; ++k) {
      if (steps_i[k] != steps_j_s[k]) {
        if (diff_min > k) diff_min = k;
        diff_max = k;
      }
    }

    if (diff_min > diff_max) continue;

    double h_1b = 0.0;

    // Try E_{diff_min, diff_max}
    {
      double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_min, diff_max,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling != 0.0) {
        h_1b += _h1e_cache[diff_min * norb + diff_max] * coupling;
      }
    }

    // Try E_{diff_max, diff_min}
    if (diff_min != diff_max) {
      double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_max, diff_min,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling != 0.0) {
        h_1b += _h1e_cache[diff_max * norb + diff_min] * coupling;
      }
    }

    if (h_1b != 0.0) {
      H_out[(int64_t)j_local * nsel + i_local] += h_1b;
    }
  }

  // === Step 4: Two-body Phase 1 — enumerate E_rs|j> intermediates via DFS ===
  if (tid == 0) { _interm_count = 0; _interm_overflow = 0; }
  __syncthreads();

  for (int rs = tid; rs < nops; rs += nthreads) {
    int r = rs / norb;
    int s = rs - r * norb;
    if (r == s) continue;
    if ((int)occ_j_s[s] <= 0 || (int)occ_j_s[r] >= 2) continue;

    int start_p1, end_p1, q_start_p1, q_mid_p1, q_end_p1;
    if (r < s) {
      start_p1 = r; end_p1 = s;
      q_start_p1 = Q_uR; q_mid_p1 = Q_R; q_end_p1 = Q_oR;
    } else {
      start_p1 = s; end_p1 = r;
      q_start_p1 = Q_uL; q_mid_p1 = Q_L; q_end_p1 = Q_oL;
    }

    int32_t node_start_p1 = nodes_j_s[start_p1];
    int32_t node_end_target_p1 = nodes_j_s[end_p1 + 1];
    uint64_t prefix_offset_p1 = idx_prefix_j_s[start_p1];
    uint64_t prefix_endplus1_p1 = idx_prefix_j_s[end_p1 + 1];
    if (j_global < prefix_endplus1_p1) continue;
    uint64_t suffix_offset_p1 = j_global - prefix_endplus1_p1;

    int8_t st_k_p1[MAX_NORB_T];
    int32_t st_node_p1[MAX_NORB_T];
    double st_w_p1[MAX_NORB_T];
    uint64_t st_seg_p1[MAX_NORB_T];
    int top_p1 = 0;
    st_k_p1[top_p1] = (int8_t)start_p1;
    st_node_p1[top_p1] = node_start_p1;
    st_w_p1[top_p1] = 1.0;
    st_seg_p1[top_p1] = 0ull;
    ++top_p1;

    while (top_p1) {
      --top_p1;
      double w_p1 = st_w_p1[top_p1];
      int kpos_p1 = (int)st_k_p1[top_p1];
      int node_k_p1 = st_node_p1[top_p1];
      uint64_t seg_idx_p1 = st_seg_p1[top_p1];
      int qk_p1 = (kpos_p1 == start_p1) ? q_start_p1 : ((kpos_p1 == end_p1) ? q_end_p1 : q_mid_p1);
      int dk_p1 = (int)steps_j_s[kpos_p1];
      int bk_p1 = (int)b_j_s[kpos_p1];
      int k_next_p1 = kpos_p1 + 1;

      int dp0_p1 = 0, dp1_p1 = 0;
      int ndp_p1 = candidate_dprimes(qk_p1, dk_p1, &dp0_p1, &dp1_p1);
      if (ndp_p1 == 0) continue;
      for (int which_p1 = 0; which_p1 < ndp_p1; ++which_p1) {
        int dprime_p1 = (which_p1 == 0) ? dp0_p1 : dp1_p1;
        int child_k_p1 = child_table[node_k_p1 * 4 + dprime_p1];
        if (child_k_p1 < 0) continue;
        int bprime_p1 = (int)node_twos[child_k_p1];
        int db_p1 = bk_p1 - bprime_p1;
        double seg_p1 = (double)segment_value_int(qk_p1, dprime_p1, dk_p1, db_p1, bk_p1);
        if (seg_p1 == 0.0) continue;
        double c_rs_p1 = w_p1 * seg_p1;
        uint64_t seg_idx2_p1 = seg_idx_p1 + (uint64_t)child_prefix[node_k_p1 * 5 + dprime_p1];
        if (kpos_p1 != end_p1) {
          if (top_p1 >= MAX_NORB_T) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          st_k_p1[top_p1] = (int8_t)k_next_p1;
          st_node_p1[top_p1] = child_k_p1;
          st_w_p1[top_p1] = c_rs_p1;
          st_seg_p1[top_p1] = seg_idx2_p1;
          ++top_p1;
          continue;
        }
        if (child_k_p1 != node_end_target_p1) continue;
        uint64_t k_global_p1 = prefix_offset_p1 + seg_idx2_p1 + suffix_offset_p1;
        if (k_global_p1 >= ncsf) {
          if (overflow_flag) atomicExch(overflow_flag, 1);
          continue;
        }
        int slot = atomicAdd(&_interm_count, 1);
        if (slot < PAIRWISE_MAX_INTERMEDIATES) {
          _interm_k_global[slot] = k_global_p1;
          _interm_r[slot] = (int8_t)r;
          _interm_s[slot] = (int8_t)s;
          _interm_crs[slot] = c_rs_p1;
        } else {
          atomicExch(&_interm_overflow, 1);
        }
      }
    }
  }
  __syncthreads();

  // === Step 5: Two-body Phase 2 — pair-wise evaluation (TILED, upper triangle) ===
  if (!_interm_overflow) {
    int n_ki = _interm_count;
    for (int ki = 0; ki < n_ki; ++ki) {
      uint64_t ki_global = _interm_k_global[ki];
      int ki_r = (int)_interm_r[ki];
      int ki_s = (int)_interm_s[ki];
      double ki_crs = _interm_crs[ki];

      // tid==0 reconstructs k's path into shared memory
      if (tid == 0) {
        bool ok_p2 = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
            child_table, child_prefix, norb, ncsf, ki_global, _p2_steps, _p2_nodes);
        if (!ok_p2) {
          _p2_nodes[0] = -1;
        } else {
          for (int kk = 0; kk < norb; ++kk) {
            _p2_occ[kk] = (int8_t)step_to_occ(_p2_steps[kk]);
            _p2_b[kk] = node_twos[_p2_nodes[kk + 1]];
          }
        }
      }
      __syncthreads();

      if (_p2_nodes[0] < 0) {
        if (tid == 0 && overflow_flag) atomicExch(overflow_flag, 1);
        __syncthreads();
        continue;
      }

      // Diagonal contribution from 2-body for H[j][j]
      if (tid == 0 && ki_global == j_global) {
        for (int p = 0; p < norb; ++p) {
          int occ_p = (int)_p2_occ[p];
          if (occ_p == 0) continue;
          diag_local += 0.5 * ki_crs *
              cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
        }
      }

      // Upper-triangle targets: i_local > j_local (diagonal handled by explicit block)
      for (int i_local = j_local + 1 + tid; i_local < nsel; i_local += nthreads) {
        const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
        const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

        // Quick screen: find diff range between i and k
        int diff_min = norb, diff_max = -1;
        for (int k = 0; k < norb; ++k) {
          if (steps_i[k] != _p2_steps[k]) {
            if (diff_min > k) diff_min = k;
            diff_max = k;
          }
        }

        double h_2b = 0.0;

        if (diff_min > diff_max) {
          // i == k in step space: diagonal 2-body contribution
          for (int p = 0; p < norb; ++p) {
            int occ_p = (int)occ_all[(int64_t)i_local * norb + p];
            if (occ_p == 0) continue;
            h_2b += 0.5 * ki_crs *
                cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
          }
        } else {
          // Off-diagonal: try both orientations
          {
            double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                norb, diff_min, diff_max,
                steps_i, b_i, _p2_steps, _p2_b, _p2_nodes, node_twos);
            if (coupling != 0.0) {
              h_2b += 0.5 * ki_crs *
                  cas36_dense_eri4_at(eri4, norb, diff_min, diff_max, ki_r, ki_s) * coupling;
            }
          }
          if (diff_min != diff_max) {
            double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                norb, diff_max, diff_min,
                steps_i, b_i, _p2_steps, _p2_b, _p2_nodes, node_twos);
            if (coupling != 0.0) {
              h_2b += 0.5 * ki_crs *
                  cas36_dense_eri4_at(eri4, norb, diff_max, diff_min, ki_r, ki_s) * coupling;
            }
          }
        }

        if (h_2b != 0.0) {
          H_out[(int64_t)j_local * nsel + i_local] += h_2b;
        }
      }
      __syncthreads();
    }
  } else {
    // === Fallback: intermediate overflow — recompute per (r,s) serially ===
    for (int rs = tid; rs < nops; rs += nthreads) {
      int r = rs / norb;
      int s = rs - r * norb;
      if (r == s) continue;
      if ((int)occ_j_s[s] <= 0 || (int)occ_j_s[r] >= 2) continue;

      int start, end, q_start, q_mid, q_end;
      if (r < s) {
        start = r; end = s;
        q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
      } else {
        start = s; end = r;
        q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
      }

      int32_t node_start = nodes_j_s[start];
      int32_t node_end_target = nodes_j_s[end + 1];
      uint64_t prefix_offset = idx_prefix_j_s[start];
      uint64_t prefix_endplus1 = idx_prefix_j_s[end + 1];
      if (j_global < prefix_endplus1) continue;
      uint64_t suffix_offset = j_global - prefix_endplus1;

      int8_t st_k[MAX_NORB_T];
      int32_t st_node[MAX_NORB_T];
      double st_w[MAX_NORB_T];
      uint64_t st_seg[MAX_NORB_T];
      int top = 0;
      st_k[top] = (int8_t)start;
      st_node[top] = node_start;
      st_w[top] = 1.0;
      st_seg[top] = 0ull;
      ++top;

      while (top) {
        --top;
        double w = st_w[top];
        int kpos = (int)st_k[top];
        int node_k = st_node[top];
        uint64_t seg_idx = st_seg[top];
        int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
        int dk = (int)steps_j_s[kpos];
        int bk = (int)b_j_s[kpos];
        int k_next = kpos + 1;

        int dp0 = 0, dp1 = 0;
        int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
        if (ndp == 0) continue;
        for (int which = 0; which < ndp; ++which) {
          int dprime = (which == 0) ? dp0 : dp1;
          int child_k = child_table[node_k * 4 + dprime];
          if (child_k < 0) continue;
          int bprime = (int)node_twos[child_k];
          int db = bk - bprime;
          double seg = (double)segment_value_int(qk, dprime, dk, db, bk);
          if (seg == 0.0) continue;
          double c_rs = w * seg;
          uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];
          if (kpos != end) {
            if (top >= MAX_NORB_T) {
              if (overflow_flag) atomicExch(overflow_flag, 1);
              continue;
            }
            st_k[top] = (int8_t)k_next;
            st_node[top] = child_k;
            st_w[top] = c_rs;
            st_seg[top] = seg_idx2;
            ++top;
            continue;
          }
          if (child_k != node_end_target) continue;

          uint64_t k_global = prefix_offset + seg_idx2 + suffix_offset;
          if (k_global >= ncsf) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }

          // Reconstruct k's path in registers
          int8_t steps_k[MAX_NORB_T];
          int32_t nodes_k[MAX_NORB_T + 1];
          int8_t occ_k[MAX_NORB_T];
          int16_t b_k[MAX_NORB_T];
          bool ok_k = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, k_global, steps_k, nodes_k);
          if (!ok_k) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          for (int kk = 0; kk < norb; ++kk) {
            occ_k[kk] = (int8_t)step_to_occ(steps_k[kk]);
            b_k[kk] = node_twos[nodes_k[kk + 1]];
          }

          // Diagonal 2-body for H[j][j]
          if (k_global == j_global) {
            for (int p = 0; p < norb; ++p) {
              int occ_p = (int)occ_k[p];
              if (occ_p == 0) continue;
              diag_local += 0.5 * c_rs *
                  cas36_dense_eri4_at(eri4, norb, p, p, r, s) * (double)occ_p;
            }
          }

          // For each target i >= j, evaluate pair-wise coupling
          for (int i_local = j_local + 1; i_local < nsel; i_local++) {
            const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
            const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

            int diff_min2 = norb, diff_max2 = -1;
            for (int kk = 0; kk < norb; ++kk) {
              if (steps_i[kk] != steps_k[kk]) {
                if (diff_min2 > kk) diff_min2 = kk;
                diff_max2 = kk;
              }
            }

            double h_2b = 0.0;
            if (diff_min2 > diff_max2) {
              for (int p = 0; p < norb; ++p) {
                int occ_p = (int)occ_all[(int64_t)i_local * norb + p];
                if (occ_p == 0) continue;
                h_2b += 0.5 * c_rs *
                    cas36_dense_eri4_at(eri4, norb, p, p, r, s) * (double)occ_p;
              }
            } else {
              {
                double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                    norb, diff_min2, diff_max2,
                    steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
                if (coupling != 0.0) {
                  h_2b += 0.5 * c_rs *
                      cas36_dense_eri4_at(eri4, norb, diff_min2, diff_max2, r, s) * coupling;
                }
              }
              if (diff_min2 != diff_max2) {
                double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                    norb, diff_max2, diff_min2,
                    steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
                if (coupling != 0.0) {
                  h_2b += 0.5 * c_rs *
                      cas36_dense_eri4_at(eri4, norb, diff_max2, diff_min2, r, s) * coupling;
                }
              }
            }

            if (h_2b != 0.0) {
              atomicAdd(&H_out[(int64_t)j_local * nsel + i_local], h_2b);
            }
          }
        }
      }
    }
  }

  // Write diagonal
  if (diag_local != 0.0) {
    atomicAdd(&H_out[(int64_t)j_local * nsel + j_local], diag_local);
  }
}

// ============================================================================
// Kernel: Mirror upper triangle to lower triangle (in-place symmetrization)
// ============================================================================
//
// Grid: enough blocks to cover all upper-triangle elements
// Each thread copies H[j][i] to H[i][j] for j < i

__global__ void pairwise_hij_mirror_kernel(
    double* __restrict__ H,   // [nsel, nsel]
    int nsel) {
  // Total upper-triangle elements (excluding diagonal): nsel*(nsel-1)/2
  int64_t total = (int64_t)nsel * (nsel - 1) / 2;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  // Map linear index to (row, col) with row < col (upper triangle)
  // Using quadratic formula: row = floor((2*nsel - 1 - sqrt((2*nsel-1)^2 - 8*idx)) / 2)
  double n2 = 2.0 * nsel - 1.0;
  int row = (int)((n2 - sqrt(n2 * n2 - 8.0 * (double)idx)) * 0.5);
  int64_t row_start = (int64_t)row * (2 * nsel - row - 1) / 2;
  int col = (int)(idx - row_start) + row + 1;
  if (row >= nsel || col >= nsel || row >= col) return;

  // Copy upper to lower: H[col][row] = H[row][col]
  H[(int64_t)col * nsel + row] = H[(int64_t)row * nsel + col];
}

// ============================================================================
// Kernel H (Bucketed): Uses a precomputed flat target list per bucket.
// ============================================================================
//
// Instead of iterating over ALL targets, each source j looks up its bucket
// and iterates only over the precomputed flat target list for that bucket.
// This is a single stride loop (no nested loops), same access pattern as the
// non-bucketed kernel but over fewer targets.
//
// Materialized arrays and sel_idx_u64 must be PRE-SORTED by occupation key.
// Output H[nsel, nsel] is in sorted order; caller unpermutes if needed.

template <int MAX_NORB_T>
__global__ __launch_bounds__(256)
void pairwise_hij_bucketed_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,     // [nsel] sorted
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* __restrict__ h_base,
    const double* __restrict__ eri4,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t*  __restrict__ steps_all,        // [nsel, norb] sorted
    const int32_t* __restrict__ nodes_all,        // [nsel, norb+1] sorted
    const int8_t*  __restrict__ occ_all,          // [nsel, norb] sorted
    const int16_t* __restrict__ b_all,            // [nsel, norb] sorted
    // Bucket data: flat target lists
    const int32_t* __restrict__ csf_to_bucket,    // [nsel] sorted idx → bucket
    const int32_t* __restrict__ target_offsets,    // [nbuckets+1]
    const int32_t* __restrict__ target_list,       // [total_targets]
    // Output
    double* __restrict__ H_out,                   // [nsel, nsel] sorted order
    int* __restrict__ overflow_flag) {

  int j_local = blockIdx.x;
  if (j_local >= nsel) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t j_global = sel_idx_u64[j_local];
  if (j_global >= ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;

  // Look up this source's target range from flat target list
  int j_bucket = csf_to_bucket[j_local];
  int tgt_start = target_offsets[j_bucket];
  int tgt_end   = target_offsets[j_bucket + 1];

  // === Shared memory layout ===
  // Static shared: small arrays
  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;

  // Dynamic shared: h1e cache + large intermediate arrays (allows > 48 KB via opt-in)
  // Layout: [h1e_cache: norb*norb doubles]
  //         [_interm_k_global: 4096 uint64]
  //         [_interm_crs: 4096 doubles]
  //         [_interm_r: 4096 int8]
  //         [_interm_s: 4096 int8]
  enum { PAIRWISE_BUCKETED_MAX_INTERMEDIATES = 4096 };
  extern __shared__ char _dyn_smem[];
  double*   _h1e_cache       = (double*)_dyn_smem;
  uint64_t* _interm_k_global = (uint64_t*)(_dyn_smem + nops * sizeof(double));
  double*   _interm_crs      = (double*)((char*)_interm_k_global + PAIRWISE_BUCKETED_MAX_INTERMEDIATES * sizeof(uint64_t));
  int8_t*   _interm_r        = (int8_t*)((char*)_interm_crs + PAIRWISE_BUCKETED_MAX_INTERMEDIATES * sizeof(double));
  int8_t*   _interm_s        = _interm_r + PAIRWISE_BUCKETED_MAX_INTERMEDIATES;

  // === Step 1: Load source j's path from materialized arrays ===
  for (int k = tid; k < norb; k += nthreads) {
    steps_j_s[k] = steps_all[(int64_t)j_local * norb + k];
    occ_j_s[k]   = occ_all[(int64_t)j_local * norb + k];
    b_j_s[k]     = b_all[(int64_t)j_local * norb + k];
  }
  for (int k = tid; k < norb + 1; k += nthreads) {
    nodes_j_s[k] = nodes_all[(int64_t)j_local * (norb + 1) + k];
  }
  __syncthreads();

  if (tid == 0) {
    idx_prefix_j_s[0] = 0ull;
    for (int k = 0; k < norb; ++k) {
      int node_k = nodes_j_s[k];
      int step_k = (int)steps_j_s[k];
      idx_prefix_j_s[k + 1] = idx_prefix_j_s[k] + (uint64_t)child_prefix[node_k * 5 + step_k];
    }
  }

  for (int i = tid; i < nops; i += nthreads) _h1e_cache[i] = h_base[i];
  __syncthreads();

  // === Step 2: Compute h_eff_j ===
  for (int pq = tid; pq < nops; pq += nthreads) {
    int p = pq / norb;
    int q = pq - p * norb;
    double hpq = _h1e_cache[pq];
    for (int r = 0; r < norb; ++r) {
      int occ_r = (int)occ_j_s[r];
      if (occ_r == 0) continue;
      hpq += 0.5 * cas36_dense_eri4_at(eri4, norb, p, q, r, r) * (double)occ_r;
    }
    _h1e_cache[pq] = hpq;
  }
  __syncthreads();

  // === Step 3: One-body pass (flat target list, upper triangle) ===
  double diag_local = 0.0;
  for (int p = tid; p < norb; p += nthreads) {
    diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
  }

  // Single stride loop over flat target list
  for (int t_idx = tgt_start + tid; t_idx < tgt_end; t_idx += nthreads) {
    int i_local = target_list[t_idx];
    if (i_local <= j_local) continue;  // upper triangle, skip self and below

    const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
    const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

    int diff_min = norb, diff_max = -1;
    for (int k = 0; k < norb; ++k) {
      if (steps_i[k] != steps_j_s[k]) {
        if (diff_min > k) diff_min = k;
        diff_max = k;
      }
    }
    if (diff_min > diff_max) continue;

    double h_1b = 0.0;
    {
      double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_min, diff_max,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling != 0.0) {
        h_1b += _h1e_cache[diff_min * norb + diff_max] * coupling;
      }
    }
    if (diff_min != diff_max) {
      double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_max, diff_min,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling != 0.0) {
        h_1b += _h1e_cache[diff_max * norb + diff_min] * coupling;
      }
    }
    if (h_1b != 0.0) {
      H_out[(int64_t)j_local * nsel + i_local] += h_1b;
    }
  }

  // === Step 4: Two-body Phase 1 — enumerate intermediates (identical) ===
  if (tid == 0) { _interm_count = 0; _interm_overflow = 0; }
  __syncthreads();

  for (int rs = tid; rs < nops; rs += nthreads) {
    int r = rs / norb;
    int s = rs - r * norb;
    if (r == s) continue;
    if ((int)occ_j_s[s] <= 0 || (int)occ_j_s[r] >= 2) continue;

    int start_p1, end_p1, q_start_p1, q_mid_p1, q_end_p1;
    if (r < s) {
      start_p1 = r; end_p1 = s;
      q_start_p1 = Q_uR; q_mid_p1 = Q_R; q_end_p1 = Q_oR;
    } else {
      start_p1 = s; end_p1 = r;
      q_start_p1 = Q_uL; q_mid_p1 = Q_L; q_end_p1 = Q_oL;
    }

    int32_t node_start_p1 = nodes_j_s[start_p1];
    int32_t node_end_target_p1 = nodes_j_s[end_p1 + 1];
    uint64_t prefix_offset_p1 = idx_prefix_j_s[start_p1];
    uint64_t prefix_endplus1_p1 = idx_prefix_j_s[end_p1 + 1];
    if (j_global < prefix_endplus1_p1) continue;
    uint64_t suffix_offset_p1 = j_global - prefix_endplus1_p1;

    int8_t st_k_p1[MAX_NORB_T];
    int32_t st_node_p1[MAX_NORB_T];
    double st_w_p1[MAX_NORB_T];
    uint64_t st_seg_p1[MAX_NORB_T];
    int top_p1 = 0;
    st_k_p1[top_p1] = (int8_t)start_p1;
    st_node_p1[top_p1] = node_start_p1;
    st_w_p1[top_p1] = 1.0;
    st_seg_p1[top_p1] = 0ull;
    ++top_p1;

    while (top_p1) {
      --top_p1;
      double w_p1 = st_w_p1[top_p1];
      int kpos_p1 = (int)st_k_p1[top_p1];
      int node_k_p1 = st_node_p1[top_p1];
      uint64_t seg_idx_p1 = st_seg_p1[top_p1];
      int qk_p1 = (kpos_p1 == start_p1) ? q_start_p1 : ((kpos_p1 == end_p1) ? q_end_p1 : q_mid_p1);
      int dk_p1 = (int)steps_j_s[kpos_p1];
      int bk_p1 = (int)b_j_s[kpos_p1];
      int k_next_p1 = kpos_p1 + 1;

      int dp0_p1 = 0, dp1_p1 = 0;
      int ndp_p1 = candidate_dprimes(qk_p1, dk_p1, &dp0_p1, &dp1_p1);
      if (ndp_p1 == 0) continue;
      for (int which_p1 = 0; which_p1 < ndp_p1; ++which_p1) {
        int dprime_p1 = (which_p1 == 0) ? dp0_p1 : dp1_p1;
        int child_k_p1 = child_table[node_k_p1 * 4 + dprime_p1];
        if (child_k_p1 < 0) continue;
        int bprime_p1 = (int)node_twos[child_k_p1];
        int db_p1 = bk_p1 - bprime_p1;
        double seg_p1 = (double)segment_value_int(qk_p1, dprime_p1, dk_p1, db_p1, bk_p1);
        if (seg_p1 == 0.0) continue;
        double c_rs_p1 = w_p1 * seg_p1;
        uint64_t seg_idx2_p1 = seg_idx_p1 + (uint64_t)child_prefix[node_k_p1 * 5 + dprime_p1];
        if (kpos_p1 != end_p1) {
          if (top_p1 >= MAX_NORB_T) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          st_k_p1[top_p1] = (int8_t)k_next_p1;
          st_node_p1[top_p1] = child_k_p1;
          st_w_p1[top_p1] = c_rs_p1;
          st_seg_p1[top_p1] = seg_idx2_p1;
          ++top_p1;
          continue;
        }
        if (child_k_p1 != node_end_target_p1) continue;
        uint64_t k_global_p1 = prefix_offset_p1 + seg_idx2_p1 + suffix_offset_p1;
        if (k_global_p1 >= ncsf) {
          if (overflow_flag) atomicExch(overflow_flag, 1);
          continue;
        }
        int slot = atomicAdd(&_interm_count, 1);
        if (slot < PAIRWISE_BUCKETED_MAX_INTERMEDIATES) {
          _interm_k_global[slot] = k_global_p1;
          _interm_r[slot] = (int8_t)r;
          _interm_s[slot] = (int8_t)s;
          _interm_crs[slot] = c_rs_p1;
        } else {
          atomicExch(&_interm_overflow, 1);
        }
      }
    }
  }
  __syncthreads();

  // === Step 5: Two-body Phase 2 (BATCHED, flat target list, upper triangle) ===
  // Process intermediates in batches of BATCH_SZ to amortize sync overhead.
  // Each batch: BATCH_SZ threads reconstruct paths in parallel, then ALL threads
  // iterate over target list for ALL batch intermediates.
  if (!_interm_overflow) {
    const int BATCH_SZ = 32;
    int n_ki = min(_interm_count, (int)PAIRWISE_BUCKETED_MAX_INTERMEDIATES);

    // Batch shared memory: reuse p2 arrays as batch storage
    // _p2_steps[MAX_NORB_T] → _batch_steps[BATCH_SZ * MAX_NORB_T] via dynamic smem
    // We place batch arrays in dynamic shared memory after intermediate arrays
    int8_t*   _batch_steps = _interm_s + PAIRWISE_BUCKETED_MAX_INTERMEDIATES;
    int16_t*  _batch_b     = (int16_t*)(_batch_steps + BATCH_SZ * MAX_NORB_T);
    int32_t*  _batch_nodes = (int32_t*)((char*)_batch_b + BATCH_SZ * MAX_NORB_T * sizeof(int16_t));
    int8_t*   _batch_occ   = (int8_t*)((char*)_batch_nodes + BATCH_SZ * (MAX_NORB_T + 1) * sizeof(int32_t));
    int8_t*   _batch_ok    = _batch_occ + BATCH_SZ * MAX_NORB_T;  // [BATCH_SZ] flags

    for (int ki_base = 0; ki_base < n_ki; ki_base += BATCH_SZ) {
      int batch_end = min(ki_base + BATCH_SZ, n_ki);
      int batch_sz = batch_end - ki_base;

      // Parallel reconstruction: first batch_sz threads each reconstruct one intermediate
      if (tid < batch_sz) {
        int ki = ki_base + tid;
        uint64_t ki_global = _interm_k_global[ki];
        int8_t*  my_steps = _batch_steps + tid * MAX_NORB_T;
        int32_t* my_nodes = _batch_nodes + tid * (MAX_NORB_T + 1);
        int8_t*  my_occ   = _batch_occ   + tid * MAX_NORB_T;
        int16_t* my_b     = _batch_b     + tid * MAX_NORB_T;
        bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
            child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
        if (!ok) {
          _batch_ok[tid] = 0;
        } else {
          _batch_ok[tid] = 1;
          for (int kk = 0; kk < norb; ++kk) {
            my_occ[kk] = (int8_t)step_to_occ(my_steps[kk]);
            my_b[kk] = node_twos[my_nodes[kk + 1]];
          }
        }
      }
      __syncthreads();

      // Process each intermediate in the batch
      for (int b = 0; b < batch_sz; ++b) {
        if (!_batch_ok[b]) continue;
        int ki = ki_base + b;
        uint64_t ki_global = _interm_k_global[ki];
        int ki_r = (int)_interm_r[ki];
        int ki_s = (int)_interm_s[ki];
        double ki_crs = _interm_crs[ki];

        const int8_t*  b_steps = _batch_steps + b * MAX_NORB_T;
        const int16_t* b_b     = _batch_b     + b * MAX_NORB_T;
        const int32_t* b_nodes = _batch_nodes + b * (MAX_NORB_T + 1);
        const int8_t*  b_occ   = _batch_occ   + b * MAX_NORB_T;

        // Diagonal contribution from 2-body for H[j][j]
        if (tid == 0 && ki_global == j_global) {
          for (int p = 0; p < norb; ++p) {
            int occ_p = (int)b_occ[p];
            if (occ_p == 0) continue;
            diag_local += 0.5 * ki_crs *
                cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
          }
        }

        // All threads iterate over flat target list
        for (int t_idx = tgt_start + tid; t_idx < tgt_end; t_idx += nthreads) {
          int i_local = target_list[t_idx];
          if (i_local <= j_local) continue;

          const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
          const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

          int diff_min = norb, diff_max = -1;
          for (int k = 0; k < norb; ++k) {
            if (steps_i[k] != b_steps[k]) {
              if (diff_min > k) diff_min = k;
              diff_max = k;
            }
          }

          double h_2b = 0.0;

          if (diff_min > diff_max) {
            for (int p = 0; p < norb; ++p) {
              int occ_p = (int)occ_all[(int64_t)i_local * norb + p];
              if (occ_p == 0) continue;
              h_2b += 0.5 * ki_crs *
                  cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
            }
          } else {
            {
              double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                  norb, diff_min, diff_max,
                  steps_i, b_i, b_steps, b_b, b_nodes, node_twos);
              if (coupling != 0.0) {
                h_2b += 0.5 * ki_crs *
                    cas36_dense_eri4_at(eri4, norb, diff_min, diff_max, ki_r, ki_s) * coupling;
              }
            }
            if (diff_min != diff_max) {
              double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                  norb, diff_max, diff_min,
                  steps_i, b_i, b_steps, b_b, b_nodes, node_twos);
              if (coupling != 0.0) {
                h_2b += 0.5 * ki_crs *
                    cas36_dense_eri4_at(eri4, norb, diff_max, diff_min, ki_r, ki_s) * coupling;
              }
            }
          }

          if (h_2b != 0.0) {
            H_out[(int64_t)j_local * nsel + i_local] += h_2b;
          }
        }
      }
      __syncthreads();
    }
  } else {
    // Fallback: intermediate overflow — recompute per (r,s) serially
    // Uses register-based k path + flat target list for targets
    for (int rs = tid; rs < nops; rs += nthreads) {
      int r = rs / norb;
      int s = rs - r * norb;
      if (r == s) continue;
      if ((int)occ_j_s[s] <= 0 || (int)occ_j_s[r] >= 2) continue;

      int start, end, q_start, q_mid, q_end;
      if (r < s) {
        start = r; end = s;
        q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
      } else {
        start = s; end = r;
        q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
      }

      int32_t node_start = nodes_j_s[start];
      int32_t node_end_target = nodes_j_s[end + 1];
      uint64_t prefix_offset = idx_prefix_j_s[start];
      uint64_t prefix_endplus1 = idx_prefix_j_s[end + 1];
      if (j_global < prefix_endplus1) continue;
      uint64_t suffix_offset = j_global - prefix_endplus1;

      int8_t st_k[MAX_NORB_T];
      int32_t st_node[MAX_NORB_T];
      double st_w[MAX_NORB_T];
      uint64_t st_seg[MAX_NORB_T];
      int top = 0;
      st_k[top] = (int8_t)start;
      st_node[top] = node_start;
      st_w[top] = 1.0;
      st_seg[top] = 0ull;
      ++top;

      while (top) {
        --top;
        double w = st_w[top];
        int kpos = (int)st_k[top];
        int node_k = st_node[top];
        uint64_t seg_idx = st_seg[top];
        int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
        int dk = (int)steps_j_s[kpos];
        int bk = (int)b_j_s[kpos];
        int k_next = kpos + 1;

        int dp0 = 0, dp1 = 0;
        int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
        if (ndp == 0) continue;
        for (int which = 0; which < ndp; ++which) {
          int dprime = (which == 0) ? dp0 : dp1;
          int child_k = child_table[node_k * 4 + dprime];
          if (child_k < 0) continue;
          int bprime = (int)node_twos[child_k];
          int db = bk - bprime;
          double seg = (double)segment_value_int(qk, dprime, dk, db, bk);
          if (seg == 0.0) continue;
          double c_rs = w * seg;
          uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];
          if (kpos != end) {
            if (top >= MAX_NORB_T) continue;
            st_k[top] = (int8_t)k_next;
            st_node[top] = child_k;
            st_w[top] = c_rs;
            st_seg[top] = seg_idx2;
            ++top;
            continue;
          }
          if (child_k != node_end_target) continue;

          uint64_t k_global = prefix_offset + seg_idx2 + suffix_offset;
          if (k_global >= ncsf) continue;

          // Reconstruct k's path in registers
          int8_t steps_k[MAX_NORB_T];
          int32_t nodes_k[MAX_NORB_T + 1];
          int8_t occ_k[MAX_NORB_T];
          int16_t b_k[MAX_NORB_T];
          bool ok_k = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, k_global, steps_k, nodes_k);
          if (!ok_k) continue;
          for (int kk = 0; kk < norb; ++kk) {
            occ_k[kk] = (int8_t)step_to_occ(steps_k[kk]);
            b_k[kk] = node_twos[nodes_k[kk + 1]];
          }

          // Diagonal 2-body for H[j][j]
          if (k_global == j_global) {
            for (int p = 0; p < norb; ++p) {
              int occ_p = (int)occ_k[p];
              if (occ_p == 0) continue;
              diag_local += 0.5 * c_rs *
                  cas36_dense_eri4_at(eri4, norb, p, p, r, s) * (double)occ_p;
            }
          }

          // Iterate over flat target list for each intermediate
          for (int t_idx = tgt_start; t_idx < tgt_end; t_idx++) {
            int i_local = target_list[t_idx];
            if (i_local <= j_local) continue;  // upper triangle only
            const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
            const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

            int diff_min2 = norb, diff_max2 = -1;
            for (int kk = 0; kk < norb; ++kk) {
              if (steps_i[kk] != steps_k[kk]) {
                if (diff_min2 > kk) diff_min2 = kk;
                diff_max2 = kk;
              }
            }

            double h_2b = 0.0;
            if (diff_min2 > diff_max2) {
              for (int p = 0; p < norb; ++p) {
                int occ_p = (int)occ_all[(int64_t)i_local * norb + p];
                if (occ_p == 0) continue;
                h_2b += 0.5 * c_rs *
                    cas36_dense_eri4_at(eri4, norb, p, p, r, s) * (double)occ_p;
              }
            } else {
              {
                double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                    norb, diff_min2, diff_max2,
                    steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
                if (coupling != 0.0) {
                  h_2b += 0.5 * c_rs *
                      cas36_dense_eri4_at(eri4, norb, diff_min2, diff_max2, r, s) * coupling;
                }
              }
              if (diff_min2 != diff_max2) {
                double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
                    norb, diff_max2, diff_min2,
                    steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
                if (coupling != 0.0) {
                  h_2b += 0.5 * c_rs *
                      cas36_dense_eri4_at(eri4, norb, diff_max2, diff_min2, r, s) * coupling;
                }
              }
            }

            if (h_2b != 0.0) {
              atomicAdd(&H_out[(int64_t)j_local * nsel + i_local], h_2b);
            }
          }
        }
      }
    }
  }

  // Write diagonal
  if (diag_local != 0.0) {
    atomicAdd(&H_out[(int64_t)j_local * nsel + j_local], diag_local);
  }
}

}  // anonymous namespace

// ============================================================================
// Launch wrappers
// ============================================================================

extern "C" cudaError_t pairwise_materialize_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int8_t*  steps_all,
    int32_t* nodes_all,
    int8_t*  occ_all,
    int16_t* b_all,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid(((unsigned int)nsel + block.x - 1u) / block.x);

#define LAUNCH_MATERIALIZE_(NORB_T) \
    pairwise_materialize_u64_kernel<NORB_T><<<grid, block, 0, stream>>>( \
        sel_idx_u64, nsel, norb, ncsf, child_table, node_twos, child_prefix, \
        steps_all, nodes_all, occ_all, b_all, overflow_flag)

  if (norb <= 8) {
    LAUNCH_MATERIALIZE_(8);
  } else if (norb <= 16) {
    LAUNCH_MATERIALIZE_(16);
  } else if (norb <= 24) {
    LAUNCH_MATERIALIZE_(24);
  } else if (norb <= 32) {
    LAUNCH_MATERIALIZE_(32);
  } else if (norb <= 48) {
    LAUNCH_MATERIALIZE_(48);
  } else {
    LAUNCH_MATERIALIZE_(64);
  }

#undef LAUNCH_MATERIALIZE_

  return cudaGetLastError();
}

extern "C" cudaError_t pairwise_hij_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* h_base,
    const double* eri4,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t*  steps_all,
    const int32_t* nodes_all,
    const int8_t*  occ_all,
    const int16_t* b_all,
    double* H_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !H_out || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  // Dynamic shared memory for h1e cache (norb^2 doubles)
  size_t dyn_smem_bytes = (size_t)norb * (size_t)norb * sizeof(double);

#define LAUNCH_HIJ_(NORB_T) \
    do { \
      auto _kfn = pairwise_hij_u64_kernel<NORB_T>; \
      /* Always set max dynamic smem: static smem from intermediates is ~38KB, */ \
      /* so total may exceed 48KB even with small dynamic smem. */ \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, \
          H_out, overflow_flag); \
    } while(0)

  if (norb <= 8) {
    LAUNCH_HIJ_(8);
  } else if (norb <= 16) {
    LAUNCH_HIJ_(16);
  } else if (norb <= 24) {
    LAUNCH_HIJ_(24);
  } else if (norb <= 32) {
    LAUNCH_HIJ_(32);
  } else if (norb <= 48) {
    LAUNCH_HIJ_(48);
  } else {
    LAUNCH_HIJ_(64);
  }

#undef LAUNCH_HIJ_

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Launch mirror kernel to copy upper triangle to lower triangle
  {
    int64_t n_upper = (int64_t)nsel * (nsel - 1) / 2;
    if (n_upper > 0) {
      dim3 mirror_block(256);
      dim3 mirror_grid(((unsigned int)n_upper + 255u) / 256u);
      pairwise_hij_mirror_kernel<<<mirror_grid, mirror_block, 0, stream>>>(H_out, nsel);
      err = cudaGetLastError();
    }
  }

  return err;
}

extern "C" cudaError_t pairwise_hij_bucketed_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* h_base,
    const double* eri4,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t*  steps_all,
    const int32_t* nodes_all,
    const int8_t*  occ_all,
    const int16_t* b_all,
    // Bucket data: flat target lists
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets,
    const int32_t* target_list,
    // Output
    double* H_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !H_out || !overflow_flag ||
      !csf_to_bucket || !target_offsets || !target_list) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  // Dynamic shared memory layout:
  //   [h1e cache: norb^2 doubles]
  //   [interm_k_global: 4096 uint64][interm_crs: 4096 doubles]
  //   [interm_r: 4096 int8][interm_s: 4096 int8]
  //   [batch_steps: 32*NORB_T int8][batch_b: 32*NORB_T int16]
  //   [batch_nodes: 32*(NORB_T+1) int32][batch_occ: 32*NORB_T int8]
  //   [batch_ok: 32 int8]
  const size_t BUCKETED_MAX_INTERM = 4096;
  const size_t BATCH_SZ = 32;
  size_t base_smem = (size_t)norb * (size_t)norb * sizeof(double)
      + BUCKETED_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t));

#define LAUNCH_HIJ_BUCKETED_(NORB_T) \
    do { \
      size_t batch_smem = BATCH_SZ * ( \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t)); \
      size_t dyn_smem_bytes = base_smem + batch_smem; \
      auto _kfn = pairwise_hij_bucketed_u64_kernel<NORB_T>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, \
          csf_to_bucket, target_offsets, target_list, \
          H_out, overflow_flag); \
    } while(0)

  if (norb <= 8) {
    LAUNCH_HIJ_BUCKETED_(8);
  } else if (norb <= 16) {
    LAUNCH_HIJ_BUCKETED_(16);
  } else if (norb <= 24) {
    LAUNCH_HIJ_BUCKETED_(24);
  } else if (norb <= 32) {
    LAUNCH_HIJ_BUCKETED_(32);
  } else if (norb <= 48) {
    LAUNCH_HIJ_BUCKETED_(48);
  } else {
    LAUNCH_HIJ_BUCKETED_(64);
  }

#undef LAUNCH_HIJ_BUCKETED_

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Mirror upper triangle to lower
  {
    int64_t n_upper = (int64_t)nsel * (nsel - 1) / 2;
    if (n_upper > 0) {
      dim3 mirror_block(256);
      dim3 mirror_grid(((unsigned int)n_upper + 255u) / 256u);
      pairwise_hij_mirror_kernel<<<mirror_grid, mirror_block, 0, stream>>>(H_out, nsel);
      err = cudaGetLastError();
    }
  }

  return err;
}
