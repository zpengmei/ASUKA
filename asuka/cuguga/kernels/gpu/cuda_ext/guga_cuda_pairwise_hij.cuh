
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
//   - Bucketed target lists: separate occ_diff<=2 lists for one-body (Step 3)
//     and k-relative binary-search of occ_diff<=2 targets per intermediate in
//     Phase 2, reducing scan volume by ~7-9x vs the coarse occ_diff<=4 list
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

template <int MAX_NORB_T>
__device__ __forceinline__ double pairwise_phase2_eval_target(
    int norb,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    const int8_t* __restrict__ steps_i,
    const int16_t* __restrict__ b_i,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos) {
  int diff_min = norb;
  int diff_max = -1;
  for (int k = 0; k < norb; ++k) {
    if (steps_i[k] != steps_k[k]) {
      if (diff_min > k) diff_min = k;
      diff_max = k;
    }
  }

  double h_2b = 0.0;
  if (diff_min > diff_max) {
    for (int p = 0; p < norb; ++p) {
      int occ_p = (int)occ_k[p];
      if (occ_p == 0) continue;
      h_2b += 0.5 * ki_crs *
          cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
    }
  } else {
    double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
        norb, diff_min, diff_max,
        steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
    if (coupling != 0.0) {
      h_2b += 0.5 * ki_crs *
          cas36_dense_eri4_at(eri4, norb, diff_min, diff_max, ki_r, ki_s) * coupling;
    }
    if (diff_min != diff_max) {
      coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_max, diff_min,
          steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
      if (coupling != 0.0) {
        h_2b += 0.5 * ki_crs *
            cas36_dense_eri4_at(eri4, norb, diff_max, diff_min, ki_r, ki_s) * coupling;
      }
    }
  }
  return h_2b;
}

template <int MAX_NORB_T>
__device__ __forceinline__ double pairwise_phase2_eval_source_diag(
    int norb,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_j,
    const int16_t* __restrict__ b_j,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos) {
  return pairwise_phase2_eval_target<MAX_NORB_T>(
      norb, ki_r, ki_s, ki_crs,
      steps_k, b_k, nodes_k, occ_k,
      steps_j, b_j, eri4, node_twos);
}

template <int MAX_NORB_T>
__device__ __forceinline__ void pairwise_phase2_process_target(
    int norb,
    int nsel,
    int j_local,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    int i_local,
    const int8_t* __restrict__ steps_all,
    const int16_t* __restrict__ b_all,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos,
    double* __restrict__ H_out) {
  if (i_local <= j_local) return;
  const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
  const int16_t* b_i     = b_all     + (int64_t)i_local * norb;
  double h_2b = pairwise_phase2_eval_target<MAX_NORB_T>(
      norb, ki_r, ki_s, ki_crs,
      steps_k, b_k, nodes_k, occ_k,
      steps_i, b_i, eri4, node_twos);
  if (h_2b != 0.0) {
    H_out[(int64_t)j_local * nsel + i_local] += h_2b;
  }
}

template <int MAX_NORB_T>
__device__ __forceinline__ void pairwise_phase2_process_bucket(
    int norb,
    int nsel,
    int j_local,
    int tid,
    int nthreads,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    int tgt_bucket,
    const int32_t* __restrict__ bucket_starts,
    const int32_t* __restrict__ bucket_sizes,
    const int8_t* __restrict__ steps_all,
    const int16_t* __restrict__ b_all,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos,
    double* __restrict__ H_out) {
  if (tgt_bucket < 0) return;
  int tgt_start = bucket_starts[tgt_bucket];
  int tgt_end = tgt_start + bucket_sizes[tgt_bucket];
  for (int i_local = tgt_start + tid; i_local < tgt_end; i_local += nthreads) {
    pairwise_phase2_process_target<MAX_NORB_T>(
        norb, nsel, j_local,
        ki_r, ki_s, ki_crs,
        steps_k, b_k, nodes_k, occ_k,
        i_local,
        steps_all, b_all, eri4, node_twos, H_out);
  }
}

template <int MAX_NORB_T>
__device__ __forceinline__ void pairwise_phase2_process_target_tiled(
    int norb,
    int nsel,
    int j_local,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int8_t* __restrict__ occ_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    int i_local,
    const int8_t* __restrict__ steps_i,
    const int16_t* __restrict__ b_i,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos,
    double* __restrict__ H_out) {
  if (i_local <= j_local) return;

  int diff_min = norb;
  int diff_max = -1;
  for (int k = 0; k < norb; ++k) {
    if (steps_i[k] != steps_k[k]) {
      if (diff_min > k) diff_min = k;
      diff_max = k;
    }
  }

  double h_2b = 0.0;
  if (diff_min > diff_max) {
    for (int p = 0; p < norb; ++p) {
      int occ_p = (int)occ_k[p];
      if (occ_p == 0) continue;
      h_2b += 0.5 * ki_crs *
          cas36_dense_eri4_at(eri4, norb, p, p, ki_r, ki_s) * (double)occ_p;
    }
  } else {
    double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
        norb, diff_min, diff_max,
        steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
    if (coupling != 0.0) {
      h_2b += 0.5 * ki_crs *
          cas36_dense_eri4_at(eri4, norb, diff_min, diff_max, ki_r, ki_s) * coupling;
    }
    if (diff_min != diff_max) {
      coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_max, diff_min,
          steps_i, b_i, steps_k, b_k, nodes_k, node_twos);
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

__device__ __forceinline__ uint32_t pairwise_rowhash_mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ void pairwise_rowhash_clear(
    int32_t* __restrict__ keys,
    double* __restrict__ vals,
    int cap,
    int tid,
    int nthreads) {
  for (int idx = tid; idx < cap; idx += nthreads) {
    keys[idx] = -1;
    vals[idx] = 0.0;
  }
}

__device__ __forceinline__ void pairwise_rowhash_add(
    int32_t* __restrict__ keys,
    double* __restrict__ vals,
    int cap,
    int key,
    double val,
    int* __restrict__ overflow_flag) {
  if (key < 0 || val == 0.0 || cap <= 0) return;
  uint32_t slot = pairwise_rowhash_mix((uint32_t)key) & (uint32_t)(cap - 1);
  for (int probe = 0; probe < cap; ++probe) {
    int32_t prev = atomicCAS(&keys[slot], -1, key);
    if (prev == -1 || prev == key) {
      atomicAdd(&vals[slot], val);
      return;
    }
    slot = (slot + 1) & (uint32_t)(cap - 1);
  }
  if (overflow_flag) atomicExch(overflow_flag, 1);
}

template <int MAX_NORB_T>
__device__ __forceinline__ void pairwise_phase2_sparse_accumulate_target(
    int norb,
    int j_local,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    int i_local,
    const int8_t* __restrict__ steps_all,
    const int16_t* __restrict__ b_all,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos,
    int32_t* __restrict__ row_keys,
    double* __restrict__ row_vals,
    int table_cap,
    int* __restrict__ overflow_flag) {
  if (i_local <= j_local) return;
  const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
  const int16_t* b_i     = b_all     + (int64_t)i_local * norb;
  double h_2b = pairwise_phase2_eval_target<MAX_NORB_T>(
      norb, ki_r, ki_s, ki_crs,
      steps_k, b_k, nodes_k, occ_k,
      steps_i, b_i, eri4, node_twos);
  if (h_2b != 0.0) {
    pairwise_rowhash_add(row_keys, row_vals, table_cap, i_local, h_2b, overflow_flag);
  }
}

template <int MAX_NORB_T>
__device__ __forceinline__ void pairwise_phase2_sparse_accumulate_bucket(
    int norb,
    int j_local,
    int tid,
    int nthreads,
    int ki_r,
    int ki_s,
    double ki_crs,
    const int8_t* __restrict__ steps_k,
    const int16_t* __restrict__ b_k,
    const int32_t* __restrict__ nodes_k,
    const int8_t* __restrict__ occ_k,
    int tgt_bucket,
    const int32_t* __restrict__ bucket_starts,
    const int32_t* __restrict__ bucket_sizes,
    const int8_t* __restrict__ steps_all,
    const int16_t* __restrict__ b_all,
    const double* __restrict__ eri4,
    const int16_t* __restrict__ node_twos,
    int32_t* __restrict__ row_keys,
    double* __restrict__ row_vals,
    int table_cap,
    int* __restrict__ overflow_flag) {
  if (tgt_bucket < 0) return;
  int tgt_start = bucket_starts[tgt_bucket];
  int tgt_end = tgt_start + bucket_sizes[tgt_bucket];
  for (int i_local = tgt_start + tid; i_local < tgt_end; i_local += nthreads) {
    pairwise_phase2_sparse_accumulate_target<MAX_NORB_T>(
        norb, j_local,
        ki_r, ki_s, ki_crs,
        steps_k, b_k, nodes_k, occ_k,
        i_local,
        steps_all, b_all, eri4, node_twos,
        row_keys, row_vals, table_cap, overflow_flag);
  }
}

template <int MAX_NORB_T, bool FILL_PASS>
__global__ __launch_bounds__(256)
void pairwise_emit_bucketed_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* __restrict__ h_base,
    const double* __restrict__ eri4,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t*  __restrict__ steps_all,
    const int32_t* __restrict__ nodes_all,
    const int8_t*  __restrict__ occ_all,
    const int16_t* __restrict__ b_all,
    const uint64_t* __restrict__ occ_keys_sorted,
    const uint64_t* __restrict__ bucket_keys,
    const int32_t* __restrict__ bucket_starts,
    const int32_t* __restrict__ bucket_sizes,
    const int32_t* __restrict__ neighbor_offsets,
    const int32_t* __restrict__ neighbor_list,
    const int32_t* __restrict__ csf_to_bucket,
    const int32_t* __restrict__ target_offsets,
    const int32_t* __restrict__ target_list,
    const int32_t* __restrict__ target_offsets_1b,
    const int32_t* __restrict__ target_list_1b,
    int table_cap,
    int32_t* __restrict__ workspace_keys,
    double* __restrict__ workspace_vals,
    const int64_t* __restrict__ row_offsets,
    int32_t* __restrict__ row_counts,
    int32_t* __restrict__ out_target_local,
    int32_t* __restrict__ out_src_pos,
    double* __restrict__ out_hij,
    double* __restrict__ diag_out,
    int* __restrict__ overflow_flag) {
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }
  if (table_cap <= 0 || (table_cap & (table_cap - 1)) != 0) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;
  int32_t* row_keys = workspace_keys + (int64_t)blockIdx.x * table_cap;
  double* row_vals = workspace_vals + (int64_t)blockIdx.x * table_cap;

  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;
  __shared__ int      _row_write_count;

  enum { FLAT_MAX_INTERMEDIATES = 4096 };
  enum { MEMBER_BATCH_SZ = 32 };
  extern __shared__ char _dyn_smem[];
  double* _h1e_cache = (double*)_dyn_smem;
  char* _smem_ptr = _dyn_smem + nops * sizeof(double);
  uint64_t* _interm_k_global = (uint64_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(uint64_t);
  double* _interm_crs = (double*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(double);
  int8_t* _interm_r = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);
  int8_t* _interm_s = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);
  int32_t* _member_slots = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_slots + MEMBER_BATCH_SZ);
  int8_t* _member_steps = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
  int16_t* _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
  _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
  int32_t* _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
  int8_t* _member_occ = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
  int8_t* _member_ok = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_ok + MEMBER_BATCH_SZ);
  uint64_t* _member_occ_key = (uint64_t*)(((uintptr_t)_smem_ptr + 7) & ~(uintptr_t)7);

  for (int j_local = blockIdx.x; j_local < nsel; j_local += gridDim.x) {
    if (tid == 0) _row_write_count = 0;
    pairwise_rowhash_clear(row_keys, row_vals, table_cap, tid, nthreads);
    if (tid == 0) diag_out[j_local] = 0.0;
    __syncthreads();

    uint64_t j_global = sel_idx_u64[j_local];
    if (j_global >= ncsf) {
      if (tid == 0) atomicExch(overflow_flag, 1);
      __syncthreads();
      continue;
    }

    int j_bucket = csf_to_bucket[j_local];
    int tgt_start = target_offsets[j_bucket];
    int tgt_end   = target_offsets[j_bucket + 1];

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

    double diag_local = 0.0;
    for (int p = tid; p < norb; p += nthreads) {
      diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
    }

    int tgt_start_1b = target_offsets_1b[j_bucket];
    int tgt_end_1b   = target_offsets_1b[j_bucket + 1];
    for (int t_idx = tgt_start_1b + tid; t_idx < tgt_end_1b; t_idx += nthreads) {
      int i_local = target_list_1b[t_idx];
      if (i_local <= j_local) continue;

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
      double coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_min, diff_max,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling != 0.0) {
        h_1b += _h1e_cache[diff_min * norb + diff_max] * coupling;
      }
      if (diff_min != diff_max) {
        coupling = pairwise_compute_epq_coupling<MAX_NORB_T>(
            norb, diff_max, diff_min,
            steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
        if (coupling != 0.0) {
          h_1b += _h1e_cache[diff_max * norb + diff_min] * coupling;
        }
      }
      if (h_1b != 0.0) {
        pairwise_rowhash_add(row_keys, row_vals, table_cap, i_local, h_1b, overflow_flag);
      }
    }

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
          if (slot < FLAT_MAX_INTERMEDIATES) {
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

    if (!_interm_overflow) {
      int n_ki = min(_interm_count, (int)FLAT_MAX_INTERMEDIATES);
      for (int ki_base = 0; ki_base < n_ki; ki_base += MEMBER_BATCH_SZ) {
        int batch_end = min(ki_base + MEMBER_BATCH_SZ, n_ki);
        int batch_sz = batch_end - ki_base;

        if (tid < batch_sz) {
          int ki = ki_base + tid;
          uint64_t ki_global = _interm_k_global[ki];
          int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
          int16_t* my_b     = _member_b     + tid * MAX_NORB_T;
          int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
          int8_t*  my_occ   = _member_occ   + tid * MAX_NORB_T;
          bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
          if (!ok) {
            _member_ok[tid] = 0;
            if (overflow_flag) atomicExch(overflow_flag, 1);
          } else {
            _member_ok[tid] = 1;
            uint64_t okey = 0ull;
            for (int kk = 0; kk < norb; ++kk) {
              int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
              my_occ[kk] = occ_kk;
              my_b[kk] = node_twos[my_nodes[kk + 1]];
              okey |= ((uint64_t)(unsigned char)occ_kk) << (2 * kk);
            }
            _member_occ_key[tid] = okey;
          }
        }
        if (tid >= batch_sz && tid < MEMBER_BATCH_SZ) {
          _member_ok[tid] = 0;
        }
        __syncthreads();

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          if (tid == 0) {
            diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
                norb,
                (int)_interm_r[ki],
                (int)_interm_s[ki],
                _interm_crs[ki],
                steps_j_s, b_j_s,
                _member_steps + b * MAX_NORB_T,
                _member_b     + b * MAX_NORB_T,
                _member_nodes + b * (MAX_NORB_T + 1),
                _member_occ   + b * MAX_NORB_T,
                eri4, node_twos);
          }
        }

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          int ki_r = (int)_interm_r[ki];
          int ki_s = (int)_interm_s[ki];
          double ki_crs = _interm_crs[ki];
          uint64_t b_key = _member_occ_key[b];
          const int8_t*  b_steps = _member_steps + b * MAX_NORB_T;
          const int16_t* b_b     = _member_b     + b * MAX_NORB_T;
          const int32_t* b_nodes = _member_nodes + b * (MAX_NORB_T + 1);
          const int8_t*  b_occ   = _member_occ   + b * MAX_NORB_T;
          for (int t_idx = tgt_start + tid; t_idx < tgt_end; t_idx += nthreads) {
            int i_local = target_list[t_idx];
            uint64_t xor_key = occ_keys_sorted[i_local] ^ b_key;
            uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
            if (__popcll(nonzero) > 2) continue;
            pairwise_phase2_sparse_accumulate_target<MAX_NORB_T>(
                norb, j_local,
                ki_r, ki_s, ki_crs,
                b_steps, b_b, b_nodes, b_occ,
                i_local,
                steps_all, b_all, eri4, node_twos,
                row_keys, row_vals, table_cap, overflow_flag);
          }
        }
        __syncthreads();
      }
    } else {
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

            diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
                norb, r, s, c_rs,
                steps_j_s, b_j_s,
                steps_k, b_k, nodes_k, occ_k,
                eri4, node_twos);
            for (int t_idx = tgt_start; t_idx < tgt_end; ++t_idx) {
              int i_local = target_list[t_idx];
              if (i_local <= j_local) continue;
              pairwise_phase2_sparse_accumulate_target<MAX_NORB_T>(
                  norb, j_local,
                  r, s, c_rs,
                  steps_k, b_k, nodes_k, occ_k,
                  i_local,
                  steps_all, b_all, eri4, node_twos,
                  row_keys, row_vals, table_cap, overflow_flag);
            }
          }
        }
      }
    }
    __syncthreads();

    if constexpr (!FILL_PASS) {
      int local_count = 0;
      for (int idx = tid; idx < table_cap; idx += nthreads) {
        int32_t key = row_keys[idx];
        if (key >= 0) {
          local_count += 1;
        }
      }
      if (local_count != 0) {
        atomicAdd(&row_counts[j_local], local_count);
      }
      if (diag_local != 0.0) {
        atomicAdd(&diag_out[j_local], diag_local);
      }
    } else {
      int64_t row_base = row_offsets[j_local];
      if (diag_local != 0.0) {
        atomicAdd(&diag_out[j_local], diag_local);
      }
      __syncthreads();
      for (int idx = tid; idx < table_cap; idx += nthreads) {
        int32_t key = row_keys[idx];
        double val = row_vals[idx];
        if (key >= 0) {
          int out_idx = atomicAdd(&_row_write_count, 1);
          out_target_local[row_base + out_idx] = key;
          out_src_pos[row_base + out_idx] = j_local;
          out_hij[row_base + out_idx] = val;
        }
      }
      __syncthreads();
      if (tid == 0 && _row_write_count != row_counts[j_local]) {
        atomicExch(overflow_flag, 1);
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ uint64_t pairwise_move_occ_key(uint64_t src_key, int p, int q) {
  const uint64_t mask_p = 0x3ULL << (2 * p);
  const uint64_t mask_q = 0x3ULL << (2 * q);
  const uint64_t occ_p = (src_key & mask_p) >> (2 * p);
  const uint64_t occ_q = (src_key & mask_q) >> (2 * q);
  return (src_key & ~(mask_p | mask_q))
      | ((occ_p + 1ULL) << (2 * p))
      | ((occ_q - 1ULL) << (2 * q));
}

__device__ __forceinline__ void pairwise_sigma_accumulate_diag(
    int row_local,
    double hdiag,
    const double* __restrict__ x,
    int nvec,
    double* __restrict__ y) {
  if (hdiag == 0.0) return;
  const int64_t row_off = (int64_t)row_local * (int64_t)nvec;
  for (int v = 0; v < nvec; ++v) {
    atomicAdd(&y[row_off + v], hdiag * x[row_off + v]);
  }
}

__device__ __forceinline__ void pairwise_sigma_accumulate_pair(
    int j_local,
    int i_local,
    double hij,
    const double* __restrict__ x,
    int nvec,
    double* __restrict__ y) {
  if (hij == 0.0 || i_local <= j_local) return;
  const int64_t j_off = (int64_t)j_local * (int64_t)nvec;
  const int64_t i_off = (int64_t)i_local * (int64_t)nvec;
  for (int v = 0; v < nvec; ++v) {
    atomicAdd(&y[j_off + v], hij * x[i_off + v]);
    atomicAdd(&y[i_off + v], hij * x[j_off + v]);
  }
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

      // Diagonal contribution from 2-body for H[j][j].
      // This includes both the k == j case and k != j self-return terms.
      if (tid == 0) {
        diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
            norb, ki_r, ki_s, ki_crs,
            steps_j_s, b_j_s,
            _p2_steps, _p2_b, _p2_nodes, _p2_occ,
            eri4, node_twos);
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

          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb, r, s, c_rs,
              steps_j_s, b_j_s,
              steps_k, b_k, nodes_k, occ_k,
              eri4, node_twos);
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
//
// Step 3 (one-body) uses the tight occ_diff<=2 target list.
// Phase 2 (two-body) uses the occ_diff<=4 list with a packed uint64 occ_diff
// filter (XOR + popcount, O(1)) that rejects ~85% of candidates before the
// more expensive step-diff + coupling path.
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
    const uint64_t* __restrict__ occ_keys_sorted, // [nsel] packed occ keys, sorted order
    const uint64_t* __restrict__ bucket_keys,     // [nbuckets] sorted unique occ keys
    const int32_t* __restrict__ bucket_starts,    // [nbuckets]
    const int32_t* __restrict__ bucket_sizes,     // [nbuckets]
    const int32_t* __restrict__ neighbor_offsets, // [nbuckets+1] j-relative occ_diff<=4 buckets
    const int32_t* __restrict__ neighbor_list,    // [total_bucket_neighbors]
    // Bucket data: flat target lists (overflow fallback + Step 3 source bucket)
    const int32_t* __restrict__ csf_to_bucket,    // [nsel] sorted idx → bucket
    const int32_t* __restrict__ target_offsets,    // [nbuckets+1] occ_diff<=4
    const int32_t* __restrict__ target_list,       // [total_targets_4]
    // One-body target lists (<=2)
    const int32_t* __restrict__ target_offsets_1b, // [nbuckets+1] occ_diff<=2
    const int32_t* __restrict__ target_list_1b,    // [total_targets_2]
    int use_bucket_filter_phase2,
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
  __shared__ int8_t  occ_group_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;
  __shared__ int      _surv_bucket_count;
  __shared__ int      _surv_bucket_overflow;
  __shared__ int      _group_target_count;
  __shared__ int      _group_target_overflow;
  __shared__ int      _member_batch_count;
  __shared__ int      _member_cursor;

  // Dynamic shared: h1e cache + intermediate arrays. The large grouped path
  // and the small flat path use different layouts so the flat path can keep a
  // larger intermediate cap without paying for the grouped scratch.
  enum { FLAT_MAX_INTERMEDIATES = 4096 };
  enum { GROUP_MAX_INTERMEDIATES = 2048 };
  enum { MEMBER_BATCH_SZ = 32 };
  enum { PHASE2_MAX_SURV_BUCKETS = 1024 };
  enum { PHASE2_MAX_COMPACT_TARGETS = 1024 };
  enum { PHASE2_MAX_GROUP_TARGETS = 4096 };
  enum { PHASE2_TARGET_TILE_SZ = 128 };
  int phase2_mode = use_bucket_filter_phase2;
  bool use_compaction = (phase2_mode == 1);
  bool use_row_union = (phase2_mode == 2);
  int phase2_max_intermediates = use_row_union
      ? GROUP_MAX_INTERMEDIATES
      : FLAT_MAX_INTERMEDIATES;
  extern __shared__ char _dyn_smem[];
  double*   _h1e_cache       = (double*)_dyn_smem;
  char* _smem_ptr = _dyn_smem + nops * sizeof(double);
  uint64_t* _interm_k_global = (uint64_t*)_smem_ptr;
  _smem_ptr += phase2_max_intermediates * sizeof(uint64_t);
  double* _interm_crs = (double*)_smem_ptr;
  _smem_ptr += phase2_max_intermediates * sizeof(double);
  int8_t* _interm_r = (int8_t*)_smem_ptr;
  _smem_ptr += phase2_max_intermediates * sizeof(int8_t);
  int8_t* _interm_s = (int8_t*)_smem_ptr;
  _smem_ptr += phase2_max_intermediates * sizeof(int8_t);

  int32_t* _interm_next = nullptr;
  int32_t* _rs_heads = nullptr;
  int32_t* _rs_sizes = nullptr;
  int32_t* _surv_bucket_ids = nullptr;
  int32_t* _group_target_ids = nullptr;
  int32_t* _member_slots = nullptr;
  int8_t* _member_steps = nullptr;
  int16_t* _member_b = nullptr;
  int32_t* _member_nodes = nullptr;
  int8_t* _member_occ = nullptr;
  int8_t* _member_ok = nullptr;
  uint64_t* _member_occ_key = nullptr;
  int32_t* _tile_ids = nullptr;
  int8_t* _tile_steps = nullptr;
  int16_t* _tile_b = nullptr;

  if (use_row_union) {
    _interm_next = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_interm_next + phase2_max_intermediates);
    _rs_heads = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_rs_heads + nops);
    _rs_sizes = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_rs_sizes + nops);
    _surv_bucket_ids = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_surv_bucket_ids + PHASE2_MAX_SURV_BUCKETS);
    _group_target_ids = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_group_target_ids + PHASE2_MAX_GROUP_TARGETS);
    _member_slots = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_slots + MEMBER_BATCH_SZ);
    _member_steps = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
    _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
    _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
    _member_occ = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_ok = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_ok + MEMBER_BATCH_SZ);
    _member_occ_key = (uint64_t*)(((uintptr_t)_smem_ptr + 7) & ~(uintptr_t)7);
    _smem_ptr = (char*)(_member_occ_key + MEMBER_BATCH_SZ);
    _tile_ids = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_tile_ids + PHASE2_TARGET_TILE_SZ);
    _tile_steps = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_tile_steps + PHASE2_TARGET_TILE_SZ * MAX_NORB_T);
    _tile_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
  } else if (use_compaction) {
    _surv_bucket_ids = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_surv_bucket_ids + PHASE2_MAX_SURV_BUCKETS);
    _group_target_ids = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_group_target_ids + PHASE2_MAX_COMPACT_TARGETS);
    _member_slots = (int32_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_slots + MEMBER_BATCH_SZ);
    _member_steps = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
    _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
    _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
    _member_occ = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_ok = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_ok + MEMBER_BATCH_SZ);
    _member_occ_key = (uint64_t*)(((uintptr_t)_smem_ptr + 7) & ~(uintptr_t)7);
  } else {
    _member_slots = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_member_slots + MEMBER_BATCH_SZ);
    _member_steps = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
    _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
    _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
    _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
    _member_occ = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
    _member_ok = (int8_t*)_smem_ptr;
    _smem_ptr = (char*)(_member_ok + MEMBER_BATCH_SZ);
    _member_occ_key = (uint64_t*)(((uintptr_t)_smem_ptr + 7) & ~(uintptr_t)7);
  }

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

  // === Step 3: One-body pass (occ_diff<=2 target list, upper triangle) ===
  double diag_local = 0.0;
  for (int p = tid; p < norb; p += nthreads) {
    diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
  }

  // Use the tight occ_diff<=2 list — one-body E_pq changes at most 2 orbitals
  int tgt_start_1b = target_offsets_1b[j_bucket];
  int tgt_end_1b   = target_offsets_1b[j_bucket + 1];
  for (int t_idx = tgt_start_1b + tid; t_idx < tgt_end_1b; t_idx += nthreads) {
    int i_local = target_list_1b[t_idx];
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
  if (use_row_union) {
    for (int rs = tid; rs < nops; rs += nthreads) {
      _rs_heads[rs] = -1;
      _rs_sizes[rs] = 0;
    }
  }
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
        if (slot < phase2_max_intermediates) {
          _interm_k_global[slot] = k_global_p1;
          _interm_r[slot] = (int8_t)r;
          _interm_s[slot] = (int8_t)s;
          _interm_crs[slot] = c_rs_p1;
          if (use_row_union) {
            _interm_next[slot] = _rs_heads[rs];
            _rs_heads[rs] = slot;
            _rs_sizes[rs] += 1;
          }
        } else {
          atomicExch(&_interm_overflow, 1);
        }
      }
    }
  }
  __syncthreads();

  // === Step 5: Two-body Phase 2 ===
  // Large-bucket path: group intermediates by their (r,s) move, which fixes the
  // intermediate occupation exactly. Build the target union once per group, then
  // reuse target tiles across the group's member batch.
  // Small-bucket path: keep the flat packed-key filter.
  if (!_interm_overflow) {
    int n_ki = min(_interm_count, phase2_max_intermediates);
    int nb_start = neighbor_offsets[j_bucket];
    int nb_end = neighbor_offsets[j_bucket + 1];
    uint64_t j_occ_key = bucket_keys[j_bucket];

    if (use_row_union) {
      for (int rs = 0; rs < nops; ++rs) {
        int group_size = _rs_sizes[rs];
        if (group_size == 0) continue;
        int r = rs / norb;
        int s = rs - r * norb;
        uint64_t group_occ_key = pairwise_move_occ_key(j_occ_key, r, s);

        for (int kk = tid; kk < norb; kk += nthreads) {
          occ_group_s[kk] = occ_j_s[kk];
        }
        __syncthreads();
        if (tid == 0) {
          occ_group_s[r] += 1;
          occ_group_s[s] -= 1;
          _surv_bucket_count = 0;
          _surv_bucket_overflow = 0;
          _group_target_count = 0;
          _group_target_overflow = 0;
          _member_cursor = _rs_heads[rs];
        }
        __syncthreads();

        for (int nb_idx = nb_start + tid; nb_idx < nb_end; nb_idx += nthreads) {
          int tgt_bucket = neighbor_list[nb_idx];
          uint64_t xor_key = bucket_keys[tgt_bucket] ^ group_occ_key;
          uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
          if (__popcll(nonzero) > 2) continue;
          int slot = atomicAdd(&_surv_bucket_count, 1);
          if (slot < PHASE2_MAX_SURV_BUCKETS) {
            _surv_bucket_ids[slot] = tgt_bucket;
          } else {
            atomicExch(&_surv_bucket_overflow, 1);
          }
        }
        __syncthreads();

        if (!_surv_bucket_overflow) {
          for (int sb = tid; sb < _surv_bucket_count; sb += nthreads) {
            int tgt_bucket = _surv_bucket_ids[sb];
            int tgt_start_b = bucket_starts[tgt_bucket];
            int tgt_end_b = tgt_start_b + bucket_sizes[tgt_bucket];
            for (int i_local = tgt_start_b; i_local < tgt_end_b; ++i_local) {
              if (i_local <= j_local) continue;
              int slot = atomicAdd(&_group_target_count, 1);
              if (slot < PHASE2_MAX_GROUP_TARGETS) {
                _group_target_ids[slot] = i_local;
              } else {
                atomicExch(&_group_target_overflow, 1);
              }
            }
          }
        }
        __syncthreads();

        while (true) {
          if (tid == 0) {
            int cursor = _member_cursor;
            int batch_count = 0;
            while (batch_count < MEMBER_BATCH_SZ && cursor >= 0) {
              _member_slots[batch_count++] = cursor;
              cursor = _interm_next[cursor];
            }
            _member_batch_count = batch_count;
            _member_cursor = cursor;
          }
          __syncthreads();

          int member_batch_count = _member_batch_count;
          if (member_batch_count == 0) break;

          if (tid < member_batch_count) {
            int slot = _member_slots[tid];
            uint64_t ki_global = _interm_k_global[slot];
            int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
            int16_t* my_b     = _member_b     + tid * MAX_NORB_T;
            int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
            int8_t*  my_occ   = _member_occ   + tid * MAX_NORB_T;
            bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
                child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
            if (!ok) {
              _member_ok[tid] = 0;
              if (overflow_flag) atomicExch(overflow_flag, 1);
            } else {
              _member_ok[tid] = 1;
              for (int kk = 0; kk < norb; ++kk) {
                int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
                my_occ[kk] = occ_kk;
                my_b[kk] = node_twos[my_nodes[kk + 1]];
              }
            }
          }
          if (tid >= member_batch_count && tid < MEMBER_BATCH_SZ) {
            _member_ok[tid] = 0;
          }
          __syncthreads();

          for (int m = 0; m < member_batch_count; ++m) {
            if (!_member_ok[m]) continue;
            int slot = _member_slots[m];
            if (tid == 0) {
              diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
                  norb,
                  (int)_interm_r[slot],
                  (int)_interm_s[slot],
                  _interm_crs[slot],
                  steps_j_s, b_j_s,
                  _member_steps + m * MAX_NORB_T,
                  _member_b     + m * MAX_NORB_T,
                  _member_nodes + m * (MAX_NORB_T + 1),
                  occ_group_s,
                  eri4, node_twos);
            }
          }

          if (_surv_bucket_overflow || _group_target_overflow) {
            for (int m = 0; m < member_batch_count; ++m) {
              if (!_member_ok[m]) continue;
              int slot = _member_slots[m];
              int ki_r = (int)_interm_r[slot];
              int ki_s = (int)_interm_s[slot];
              double ki_crs = _interm_crs[slot];
              const int8_t*  k_steps = _member_steps + m * MAX_NORB_T;
              const int16_t* k_b     = _member_b     + m * MAX_NORB_T;
              const int32_t* k_nodes = _member_nodes + m * (MAX_NORB_T + 1);
              if (_surv_bucket_overflow) {
                for (int nb_idx = nb_start; nb_idx < nb_end; ++nb_idx) {
                  int tgt_bucket = neighbor_list[nb_idx];
                  uint64_t xor_key = bucket_keys[tgt_bucket] ^ group_occ_key;
                  uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
                  if (__popcll(nonzero) > 2) continue;
                  pairwise_phase2_process_bucket<MAX_NORB_T>(
                      norb, nsel, j_local, tid, nthreads,
                      ki_r, ki_s, ki_crs,
                      k_steps, k_b, k_nodes, occ_group_s,
                      tgt_bucket,
                      bucket_starts, bucket_sizes,
                      steps_all, b_all, eri4, node_twos, H_out);
                }
              } else {
                for (int sb = 0; sb < _surv_bucket_count; ++sb) {
                  int tgt_bucket = _surv_bucket_ids[sb];
                  pairwise_phase2_process_bucket<MAX_NORB_T>(
                      norb, nsel, j_local, tid, nthreads,
                      ki_r, ki_s, ki_crs,
                      k_steps, k_b, k_nodes, occ_group_s,
                      tgt_bucket,
                      bucket_starts, bucket_sizes,
                      steps_all, b_all, eri4, node_twos, H_out);
                }
              }
            }
          } else {
            for (int tile_base = 0; tile_base < _group_target_count; tile_base += PHASE2_TARGET_TILE_SZ) {
              int tile_n = min(PHASE2_TARGET_TILE_SZ, _group_target_count - tile_base);
              for (int t = tid; t < tile_n; t += nthreads) {
                _tile_ids[t] = _group_target_ids[tile_base + t];
              }
              __syncthreads();
              for (int idx = tid; idx < tile_n * norb; idx += nthreads) {
                int t = idx / norb;
                int kk = idx - t * norb;
                int i_local = _tile_ids[t];
                _tile_steps[t * MAX_NORB_T + kk] = steps_all[(int64_t)i_local * norb + kk];
                _tile_b[t * MAX_NORB_T + kk] = b_all[(int64_t)i_local * norb + kk];
              }
              __syncthreads();

              for (int t = tid; t < tile_n; t += nthreads) {
                int i_local = _tile_ids[t];
                const int8_t*  steps_i = _tile_steps + t * MAX_NORB_T;
                const int16_t* b_i     = _tile_b     + t * MAX_NORB_T;
                double h_sum = 0.0;
                for (int m = 0; m < member_batch_count; ++m) {
                  if (!_member_ok[m]) continue;
                  int slot = _member_slots[m];
                  h_sum += pairwise_phase2_eval_target<MAX_NORB_T>(
                      norb,
                      (int)_interm_r[slot],
                      (int)_interm_s[slot],
                      _interm_crs[slot],
                      _member_steps + m * MAX_NORB_T,
                      _member_b     + m * MAX_NORB_T,
                      _member_nodes + m * (MAX_NORB_T + 1),
                      occ_group_s,
                      steps_i,
                      b_i,
                      eri4,
                      node_twos);
                }
                if (h_sum != 0.0) {
                  H_out[(int64_t)j_local * nsel + i_local] += h_sum;
                }
              }
              __syncthreads();
            }
          }
          __syncthreads();
        }
        __syncthreads();
      }
    } else if (use_compaction) {
      for (int ki_base = 0; ki_base < n_ki; ki_base += MEMBER_BATCH_SZ) {
        int batch_end = min(ki_base + MEMBER_BATCH_SZ, n_ki);
        int batch_sz = batch_end - ki_base;

        if (tid < batch_sz) {
          int ki = ki_base + tid;
          uint64_t ki_global = _interm_k_global[ki];
          int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
          int16_t* my_b     = _member_b     + tid * MAX_NORB_T;
          int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
          int8_t*  my_occ   = _member_occ   + tid * MAX_NORB_T;
          bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
          if (!ok) {
            _member_ok[tid] = 0;
            if (overflow_flag) atomicExch(overflow_flag, 1);
          } else {
            _member_ok[tid] = 1;
            uint64_t okey = 0ull;
            for (int kk = 0; kk < norb; ++kk) {
              int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
              my_occ[kk] = occ_kk;
              my_b[kk] = node_twos[my_nodes[kk + 1]];
              okey |= ((uint64_t)(unsigned char)occ_kk) << (2 * kk);
            }
            _member_occ_key[tid] = okey;
          }
        }
        if (tid >= batch_sz && tid < MEMBER_BATCH_SZ) {
          _member_ok[tid] = 0;
        }
        __syncthreads();

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          if (tid == 0) {
            diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
                norb,
                (int)_interm_r[ki],
                (int)_interm_s[ki],
                _interm_crs[ki],
                steps_j_s, b_j_s,
                _member_steps + b * MAX_NORB_T,
                _member_b     + b * MAX_NORB_T,
                _member_nodes + b * (MAX_NORB_T + 1),
                _member_occ   + b * MAX_NORB_T,
                eri4, node_twos);
          }
        }

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          int ki_r = (int)_interm_r[ki];
          int ki_s = (int)_interm_s[ki];
          double ki_crs = _interm_crs[ki];
          uint64_t b_key = _member_occ_key[b];
          const int8_t*  b_steps = _member_steps + b * MAX_NORB_T;
          const int16_t* b_b     = _member_b     + b * MAX_NORB_T;
          const int32_t* b_nodes = _member_nodes + b * (MAX_NORB_T + 1);
          const int8_t*  b_occ   = _member_occ   + b * MAX_NORB_T;

          if (tid == 0) {
            _surv_bucket_count = 0;
            _surv_bucket_overflow = 0;
            _group_target_count = 0;
            _group_target_overflow = 0;
          }
          __syncthreads();

          for (int nb_idx = nb_start + tid; nb_idx < nb_end; nb_idx += nthreads) {
            int tgt_bucket = neighbor_list[nb_idx];
            uint64_t xor_key = bucket_keys[tgt_bucket] ^ b_key;
            uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
            if (__popcll(nonzero) > 2) continue;
            int slot = atomicAdd(&_surv_bucket_count, 1);
            if (slot < PHASE2_MAX_SURV_BUCKETS) {
              _surv_bucket_ids[slot] = tgt_bucket;
            } else {
              atomicExch(&_surv_bucket_overflow, 1);
            }
          }
          __syncthreads();

          if (!_surv_bucket_overflow) {
            for (int sb = tid; sb < _surv_bucket_count; sb += nthreads) {
              int tgt_bucket = _surv_bucket_ids[sb];
              int tgt_start_b = bucket_starts[tgt_bucket];
              int tgt_end_b = tgt_start_b + bucket_sizes[tgt_bucket];
              for (int i_local = tgt_start_b; i_local < tgt_end_b; ++i_local) {
                if (i_local <= j_local) continue;
                int slot = atomicAdd(&_group_target_count, 1);
                if (slot < PHASE2_MAX_COMPACT_TARGETS) {
                  _group_target_ids[slot] = i_local;
                } else {
                  atomicExch(&_group_target_overflow, 1);
                }
              }
            }
          }
          __syncthreads();

          if (_surv_bucket_overflow || _group_target_overflow) {
            if (_surv_bucket_overflow) {
              for (int nb_idx = nb_start; nb_idx < nb_end; ++nb_idx) {
                int tgt_bucket = neighbor_list[nb_idx];
                uint64_t xor_key = bucket_keys[tgt_bucket] ^ b_key;
                uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
                if (__popcll(nonzero) > 2) continue;
                pairwise_phase2_process_bucket<MAX_NORB_T>(
                    norb, nsel, j_local, tid, nthreads,
                    ki_r, ki_s, ki_crs,
                    b_steps, b_b, b_nodes, b_occ,
                    tgt_bucket,
                    bucket_starts, bucket_sizes,
                    steps_all, b_all, eri4, node_twos, H_out);
              }
            } else {
              for (int sb = 0; sb < _surv_bucket_count; ++sb) {
                int tgt_bucket = _surv_bucket_ids[sb];
                pairwise_phase2_process_bucket<MAX_NORB_T>(
                    norb, nsel, j_local, tid, nthreads,
                    ki_r, ki_s, ki_crs,
                    b_steps, b_b, b_nodes, b_occ,
                    tgt_bucket,
                    bucket_starts, bucket_sizes,
                    steps_all, b_all, eri4, node_twos, H_out);
              }
            }
          } else {
            for (int st = tid; st < _group_target_count; st += nthreads) {
              int i_local = _group_target_ids[st];
              pairwise_phase2_process_target<MAX_NORB_T>(
                  norb, nsel, j_local,
                  ki_r, ki_s, ki_crs,
                  b_steps, b_b, b_nodes, b_occ,
                  i_local,
                  steps_all, b_all, eri4, node_twos, H_out);
            }
          }
          __syncthreads();
        }
      }
    } else {
      for (int ki_base = 0; ki_base < n_ki; ki_base += MEMBER_BATCH_SZ) {
        int batch_end = min(ki_base + MEMBER_BATCH_SZ, n_ki);
        int batch_sz = batch_end - ki_base;

        if (tid < batch_sz) {
          int ki = ki_base + tid;
          uint64_t ki_global = _interm_k_global[ki];
          int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
          int16_t* my_b     = _member_b     + tid * MAX_NORB_T;
          int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
          int8_t*  my_occ   = _member_occ   + tid * MAX_NORB_T;
          bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
          if (!ok) {
            _member_ok[tid] = 0;
            if (overflow_flag) atomicExch(overflow_flag, 1);
          } else {
            _member_ok[tid] = 1;
            uint64_t okey = 0ull;
            for (int kk = 0; kk < norb; ++kk) {
              int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
              my_occ[kk] = occ_kk;
              my_b[kk] = node_twos[my_nodes[kk + 1]];
              okey |= ((uint64_t)(unsigned char)occ_kk) << (2 * kk);
            }
            _member_occ_key[tid] = okey;
          }
        }
        if (tid >= batch_sz && tid < MEMBER_BATCH_SZ) {
          _member_ok[tid] = 0;
        }
        __syncthreads();

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          if (tid == 0) {
            diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
                norb,
                (int)_interm_r[ki],
                (int)_interm_s[ki],
                _interm_crs[ki],
                steps_j_s, b_j_s,
                _member_steps + b * MAX_NORB_T,
                _member_b     + b * MAX_NORB_T,
                _member_nodes + b * (MAX_NORB_T + 1),
                _member_occ   + b * MAX_NORB_T,
                eri4, node_twos);
          }
        }

        for (int b = 0; b < batch_sz; ++b) {
          if (!_member_ok[b]) continue;
          int ki = ki_base + b;
          int ki_r = (int)_interm_r[ki];
          int ki_s = (int)_interm_s[ki];
          double ki_crs = _interm_crs[ki];
          uint64_t b_key = _member_occ_key[b];
          const int8_t*  b_steps = _member_steps + b * MAX_NORB_T;
          const int16_t* b_b     = _member_b     + b * MAX_NORB_T;
          const int32_t* b_nodes = _member_nodes + b * (MAX_NORB_T + 1);
          const int8_t*  b_occ   = _member_occ   + b * MAX_NORB_T;
          for (int t_idx = tgt_start + tid; t_idx < tgt_end; t_idx += nthreads) {
            int i_local = target_list[t_idx];
            uint64_t xor_key = occ_keys_sorted[i_local] ^ b_key;
            uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
            if (__popcll(nonzero) > 2) continue;
            pairwise_phase2_process_target<MAX_NORB_T>(
                norb, nsel, j_local,
                ki_r, ki_s, ki_crs,
                b_steps, b_b, b_nodes, b_occ,
                i_local,
                steps_all, b_all, eri4, node_twos, H_out);
          }
        }
        __syncthreads();
      }
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

          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb, r, s, c_rs,
              steps_j_s, b_j_s,
              steps_k, b_k, nodes_k, occ_k,
              eri4, node_twos);
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

template <int MAX_NORB_T>
__global__ __launch_bounds__(256)
void pairwise_sigma_bucketed_u64_kernel(
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
    const uint64_t* __restrict__ occ_keys_sorted, // [nsel] packed occ keys
    const int32_t* __restrict__ csf_to_bucket,    // [nsel]
    const int32_t* __restrict__ target_offsets,   // [nbuckets+1] occ_diff<=4
    const int32_t* __restrict__ target_list,      // [total_targets_4]
    const int32_t* __restrict__ target_offsets_1b,// [nbuckets+1] occ_diff<=2
    const int32_t* __restrict__ target_list_1b,   // [total_targets_2]
    const double* __restrict__ x,                 // [nsel, nvec]
    int nvec,
    double* __restrict__ y,                       // [nsel, nvec]
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

  int j_bucket = csf_to_bucket[j_local];
  int tgt_start = target_offsets[j_bucket];
  int tgt_end = target_offsets[j_bucket + 1];

  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;

  enum { FLAT_MAX_INTERMEDIATES = 4096 };
  enum { MEMBER_BATCH_SZ = 32 };
  extern __shared__ char _dyn_smem[];
  double* _h1e_cache = (double*)_dyn_smem;
  char* _smem_ptr = _dyn_smem + nops * sizeof(double);
  uint64_t* _interm_k_global = (uint64_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(uint64_t);
  double* _interm_crs = (double*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(double);
  int8_t* _interm_r = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);
  int8_t* _interm_s = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);

  int32_t* _member_slots = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_slots + MEMBER_BATCH_SZ);
  int8_t* _member_steps = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
  int16_t* _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
  _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
  int32_t* _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
  int8_t* _member_occ = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
  int8_t* _member_ok = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_ok + MEMBER_BATCH_SZ);
  uint64_t* _member_occ_key = (uint64_t*)(((uintptr_t)_smem_ptr + 7) & ~(uintptr_t)7);

  for (int k = tid; k < norb; k += nthreads) {
    steps_j_s[k] = steps_all[(int64_t)j_local * norb + k];
    occ_j_s[k] = occ_all[(int64_t)j_local * norb + k];
    b_j_s[k] = b_all[(int64_t)j_local * norb + k];
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

  double diag_local = 0.0;
  for (int p = tid; p < norb; p += nthreads) {
    diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
  }

  int tgt_start_1b = target_offsets_1b[j_bucket];
  int tgt_end_1b = target_offsets_1b[j_bucket + 1];
  for (int t_idx = tgt_start_1b + tid; t_idx < tgt_end_1b; t_idx += nthreads) {
    int i_local = target_list_1b[t_idx];
    if (i_local <= j_local) continue;
    const int8_t* steps_i = steps_all + (int64_t)i_local * norb;
    const int16_t* b_i = b_all + (int64_t)i_local * norb;
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
    pairwise_sigma_accumulate_pair(j_local, i_local, h_1b, x, nvec, y);
  }

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
        if (slot < FLAT_MAX_INTERMEDIATES) {
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

  if (!_interm_overflow) {
    int n_ki = _interm_count;
    for (int ki_base = 0; ki_base < n_ki; ki_base += MEMBER_BATCH_SZ) {
      int batch_end = min(ki_base + MEMBER_BATCH_SZ, n_ki);
      int batch_sz = batch_end - ki_base;

      if (tid < batch_sz) {
        int ki = ki_base + tid;
        uint64_t ki_global = _interm_k_global[ki];
        int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
        int16_t* my_b = _member_b + tid * MAX_NORB_T;
        int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
        int8_t*  my_occ = _member_occ + tid * MAX_NORB_T;
        bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
            child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
        if (!ok) {
          _member_ok[tid] = 0;
          if (overflow_flag) atomicExch(overflow_flag, 1);
        } else {
          _member_ok[tid] = 1;
          uint64_t okey = 0ull;
          for (int kk = 0; kk < norb; ++kk) {
            int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
            my_occ[kk] = occ_kk;
            my_b[kk] = node_twos[my_nodes[kk + 1]];
            okey |= ((uint64_t)(unsigned char)occ_kk) << (2 * kk);
          }
          _member_occ_key[tid] = okey;
        }
      }
      if (tid >= batch_sz && tid < MEMBER_BATCH_SZ) {
        _member_ok[tid] = 0;
      }
      __syncthreads();

      for (int b = 0; b < batch_sz; ++b) {
        if (!_member_ok[b]) continue;
        int ki = ki_base + b;
        if (tid == 0) {
          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb,
              (int)_interm_r[ki],
              (int)_interm_s[ki],
              _interm_crs[ki],
              steps_j_s, b_j_s,
              _member_steps + b * MAX_NORB_T,
              _member_b + b * MAX_NORB_T,
              _member_nodes + b * (MAX_NORB_T + 1),
              _member_occ + b * MAX_NORB_T,
              eri4, node_twos);
        }
      }

      for (int b = 0; b < batch_sz; ++b) {
        if (!_member_ok[b]) continue;
        int ki = ki_base + b;
        int ki_r = (int)_interm_r[ki];
        int ki_s = (int)_interm_s[ki];
        double ki_crs = _interm_crs[ki];
        uint64_t b_key = _member_occ_key[b];
        const int8_t* b_steps = _member_steps + b * MAX_NORB_T;
        const int16_t* b_b = _member_b + b * MAX_NORB_T;
        const int32_t* b_nodes = _member_nodes + b * (MAX_NORB_T + 1);
        const int8_t* b_occ = _member_occ + b * MAX_NORB_T;
        for (int t_idx = tgt_start + tid; t_idx < tgt_end; t_idx += nthreads) {
          int i_local = target_list[t_idx];
          if (i_local <= j_local) continue;
          uint64_t xor_key = occ_keys_sorted[i_local] ^ b_key;
          uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
          if (__popcll(nonzero) > 2) continue;
          const int8_t* steps_i = steps_all + (int64_t)i_local * norb;
          const int16_t* b_i = b_all + (int64_t)i_local * norb;
          double h_2b = pairwise_phase2_eval_target<MAX_NORB_T>(
              norb, ki_r, ki_s, ki_crs,
              b_steps, b_b, b_nodes, b_occ,
              steps_i, b_i,
              eri4, node_twos);
          pairwise_sigma_accumulate_pair(j_local, i_local, h_2b, x, nvec, y);
        }
      }
      __syncthreads();
    }
  } else {
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

          int8_t steps_k[MAX_NORB_T];
          int32_t nodes_k[MAX_NORB_T + 1];
          int8_t occ_k[MAX_NORB_T];
          int16_t b_k[MAX_NORB_T];
          bool ok_k = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child_table, child_prefix, norb, ncsf, k_global, steps_k, nodes_k);
          if (!ok_k) continue;
          uint64_t occ_key_k = 0ull;
          for (int kk = 0; kk < norb; ++kk) {
            occ_k[kk] = (int8_t)step_to_occ(steps_k[kk]);
            b_k[kk] = node_twos[nodes_k[kk + 1]];
            occ_key_k |= ((uint64_t)(unsigned char)occ_k[kk]) << (2 * kk);
          }

          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb, r, s, c_rs,
              steps_j_s, b_j_s,
              steps_k, b_k, nodes_k, occ_k,
              eri4, node_twos);
          for (int t_idx = tgt_start; t_idx < tgt_end; ++t_idx) {
            int i_local = target_list[t_idx];
            if (i_local <= j_local) continue;
            uint64_t xor_key = occ_keys_sorted[i_local] ^ occ_key_k;
            uint64_t nonzero = (xor_key | (xor_key >> 1)) & 0x5555555555555555ULL;
            if (__popcll(nonzero) > 2) continue;
            const int8_t* steps_i = steps_all + (int64_t)i_local * norb;
            const int16_t* b_i = b_all + (int64_t)i_local * norb;
            double h_2b = pairwise_phase2_eval_target<MAX_NORB_T>(
                norb, r, s, c_rs,
                steps_k, b_k, nodes_k, occ_k,
                steps_i, b_i,
                eri4, node_twos);
            pairwise_sigma_accumulate_pair(j_local, i_local, h_2b, x, nvec, y);
          }
        }
      }
    }
  }

  pairwise_sigma_accumulate_diag(j_local, diag_local, x, nvec, y);
}

template <int MAX_NORB_T>
__global__ __launch_bounds__(256)
void pairwise_diag_bucketed_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,     // [nsel]
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* __restrict__ h_base,
    const double* __restrict__ eri4,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t*  __restrict__ steps_all,        // [nsel, norb]
    const int32_t* __restrict__ nodes_all,        // [nsel, norb+1]
    const int8_t*  __restrict__ occ_all,          // [nsel, norb]
    const int16_t* __restrict__ b_all,            // [nsel, norb]
    double* __restrict__ diag_out,                // [nsel]
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

  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];
  __shared__ int      _interm_count;
  __shared__ int      _interm_overflow;

  enum { FLAT_MAX_INTERMEDIATES = 4096 };
  enum { MEMBER_BATCH_SZ = 32 };
  extern __shared__ char _dyn_smem[];
  double* _h1e_cache = (double*)_dyn_smem;
  char* _smem_ptr = _dyn_smem + nops * sizeof(double);
  uint64_t* _interm_k_global = (uint64_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(uint64_t);
  double* _interm_crs = (double*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(double);
  int8_t* _interm_r = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);
  int8_t* _interm_s = (int8_t*)_smem_ptr;
  _smem_ptr += FLAT_MAX_INTERMEDIATES * sizeof(int8_t);

  int8_t* _member_steps = (int8_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_steps + MEMBER_BATCH_SZ * MAX_NORB_T);
  int16_t* _member_b = (int16_t*)(((uintptr_t)_smem_ptr + 1) & ~(uintptr_t)1);
  _smem_ptr = (char*)_member_b + MEMBER_BATCH_SZ * MAX_NORB_T * sizeof(int16_t);
  int32_t* _member_nodes = (int32_t*)(((uintptr_t)_smem_ptr + 3) & ~(uintptr_t)3);
  _smem_ptr = (char*)(_member_nodes + MEMBER_BATCH_SZ * (MAX_NORB_T + 1));
  int8_t* _member_occ = (int8_t*)_smem_ptr;
  _smem_ptr = (char*)(_member_occ + MEMBER_BATCH_SZ * MAX_NORB_T);
  int8_t* _member_ok = (int8_t*)_smem_ptr;

  for (int k = tid; k < norb; k += nthreads) {
    steps_j_s[k] = steps_all[(int64_t)j_local * norb + k];
    occ_j_s[k] = occ_all[(int64_t)j_local * norb + k];
    b_j_s[k] = b_all[(int64_t)j_local * norb + k];
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

  double diag_local = 0.0;
  for (int p = tid; p < norb; p += nthreads) {
    diag_local += _h1e_cache[p * norb + p] * (double)occ_j_s[p];
  }

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
        if (slot < FLAT_MAX_INTERMEDIATES) {
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

  if (!_interm_overflow) {
    int n_ki = _interm_count;
    for (int ki_base = 0; ki_base < n_ki; ki_base += MEMBER_BATCH_SZ) {
      int batch_end = min(ki_base + MEMBER_BATCH_SZ, n_ki);
      int batch_sz = batch_end - ki_base;

      if (tid < batch_sz) {
        int ki = ki_base + tid;
        uint64_t ki_global = _interm_k_global[ki];
        int8_t*  my_steps = _member_steps + tid * MAX_NORB_T;
        int16_t* my_b = _member_b + tid * MAX_NORB_T;
        int32_t* my_nodes = _member_nodes + tid * (MAX_NORB_T + 1);
        int8_t*  my_occ = _member_occ + tid * MAX_NORB_T;
        bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
            child_table, child_prefix, norb, ncsf, ki_global, my_steps, my_nodes);
        if (!ok) {
          _member_ok[tid] = 0;
          if (overflow_flag) atomicExch(overflow_flag, 1);
        } else {
          _member_ok[tid] = 1;
          for (int kk = 0; kk < norb; ++kk) {
            int8_t occ_kk = (int8_t)step_to_occ(my_steps[kk]);
            my_occ[kk] = occ_kk;
            my_b[kk] = node_twos[my_nodes[kk + 1]];
          }
        }
      }
      if (tid >= batch_sz && tid < MEMBER_BATCH_SZ) {
        _member_ok[tid] = 0;
      }
      __syncthreads();

      for (int b = 0; b < batch_sz; ++b) {
        if (!_member_ok[b]) continue;
        if (tid == 0) {
          int ki = ki_base + b;
          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb,
              (int)_interm_r[ki],
              (int)_interm_s[ki],
              _interm_crs[ki],
              steps_j_s, b_j_s,
              _member_steps + b * MAX_NORB_T,
              _member_b + b * MAX_NORB_T,
              _member_nodes + b * (MAX_NORB_T + 1),
              _member_occ + b * MAX_NORB_T,
              eri4, node_twos);
        }
      }
      __syncthreads();
    }
  } else {
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

          diag_local += pairwise_phase2_eval_source_diag<MAX_NORB_T>(
              norb, r, s, c_rs,
              steps_j_s, b_j_s,
              steps_k, b_k, nodes_k, occ_k,
              eri4, node_twos);
        }
      }
    }
  }

  if (diag_local != 0.0) {
    atomicAdd(&diag_out[j_local], diag_local);
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
    const uint64_t* occ_keys_sorted,
    const uint64_t* bucket_keys,
    const int32_t* bucket_starts,
    const int32_t* bucket_sizes,
    const int32_t* neighbor_offsets,
    const int32_t* neighbor_list,
    // Bucket data: flat target lists
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets,
    const int32_t* target_list,
    // One-body target lists (<=2)
    const int32_t* target_offsets_1b,
    const int32_t* target_list_1b,
    int use_bucket_filter_phase2,
    // Output
    double* H_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !occ_keys_sorted ||
      !bucket_keys || !bucket_starts || !bucket_sizes || !neighbor_offsets || !neighbor_list ||
      !H_out || !overflow_flag ||
      !csf_to_bucket || !target_offsets || !target_list ||
      !target_offsets_1b || !target_list_1b) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  // Dynamic shared memory layout depends on the Phase 2 mode:
  // flat mode keeps a 4096-intermediate cap;
  // grouped mode uses a 2048 cap and spends the saved memory on group scratch.
  const size_t FLAT_MAX_INTERM = 4096;
  const size_t GROUP_MAX_INTERM = 2048;
  const size_t MEMBER_BATCH_SZ = 32;
  const size_t PHASE2_MAX_SURV_BUCKETS = 1024;
  const size_t PHASE2_MAX_COMPACT_TARGETS = 1024;
  const size_t PHASE2_MAX_GROUP_TARGETS = 4096;
  const size_t PHASE2_TARGET_TILE_SZ = 128;

#define LAUNCH_HIJ_BUCKETED_(NORB_T) \
    do { \
      size_t flat_base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)); \
      size_t flat_member_smem = 16 + MEMBER_BATCH_SZ * ( \
          sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t) + \
          sizeof(uint64_t)); \
      size_t compact_base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)) \
          + (PHASE2_MAX_SURV_BUCKETS + PHASE2_MAX_COMPACT_TARGETS) * sizeof(int32_t); \
      size_t compact_member_smem = flat_member_smem; \
      size_t grouped_base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + GROUP_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t) + sizeof(int32_t)) \
          + 2 * (size_t)norb * (size_t)norb * sizeof(int32_t) \
          + PHASE2_MAX_SURV_BUCKETS * sizeof(int32_t) \
          + PHASE2_MAX_GROUP_TARGETS * sizeof(int32_t) \
          + MEMBER_BATCH_SZ * sizeof(int32_t); \
      size_t grouped_member_smem = 16 + MEMBER_BATCH_SZ * ( \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t) + \
          sizeof(uint64_t)); \
      size_t grouped_tile_smem = PHASE2_TARGET_TILE_SZ * ( \
          sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t)); \
      size_t dyn_smem_bytes = (use_bucket_filter_phase2 == 2) \
          ? (grouped_base_smem + grouped_member_smem + grouped_tile_smem) \
          : ((use_bucket_filter_phase2 == 1) \
              ? (compact_base_smem + compact_member_smem) \
              : (flat_base_smem + flat_member_smem)); \
      auto _kfn = pairwise_hij_bucketed_u64_kernel<NORB_T>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, occ_keys_sorted, \
          bucket_keys, bucket_starts, bucket_sizes, neighbor_offsets, neighbor_list, \
          csf_to_bucket, target_offsets, target_list, \
          target_offsets_1b, target_list_1b, \
          use_bucket_filter_phase2, \
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

extern "C" cudaError_t pairwise_diag_bucketed_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* h_base,
    const double* eri4,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_all,
    const int32_t* nodes_all,
    const int8_t* occ_all,
    const int16_t* b_all,
    double* diag_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !diag_out || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  const size_t FLAT_MAX_INTERM = 4096;
  const size_t MEMBER_BATCH_SZ = 32;

#define LAUNCH_DIAG_BUCKETED_(NORB_T) \
    do { \
      size_t flat_base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)); \
      size_t flat_member_smem = 16 + MEMBER_BATCH_SZ * ( \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t)); \
      size_t dyn_smem_bytes = flat_base_smem + flat_member_smem; \
      auto _kfn = pairwise_diag_bucketed_u64_kernel<NORB_T>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, diag_out, overflow_flag); \
    } while (0)

  if (norb <= 8) {
    LAUNCH_DIAG_BUCKETED_(8);
  } else if (norb <= 16) {
    LAUNCH_DIAG_BUCKETED_(16);
  } else if (norb <= 24) {
    LAUNCH_DIAG_BUCKETED_(24);
  } else if (norb <= 32) {
    LAUNCH_DIAG_BUCKETED_(32);
  } else if (norb <= 48) {
    LAUNCH_DIAG_BUCKETED_(48);
  } else {
    LAUNCH_DIAG_BUCKETED_(64);
  }

#undef LAUNCH_DIAG_BUCKETED_

  return cudaGetLastError();
}

extern "C" cudaError_t pairwise_count_bucketed_u64_launch_stream(
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
    const uint64_t* occ_keys_sorted,
    const uint64_t* bucket_keys,
    const int32_t* bucket_starts,
    const int32_t* bucket_sizes,
    const int32_t* neighbor_offsets,
    const int32_t* neighbor_list,
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets,
    const int32_t* target_list,
    const int32_t* target_offsets_1b,
    const int32_t* target_list_1b,
    int table_cap,
    int32_t* workspace_keys,
    double* workspace_vals,
    int grid_blocks,
    int32_t* row_counts,
    double* diag_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !occ_keys_sorted ||
      !bucket_keys || !bucket_starts || !bucket_sizes || !neighbor_offsets || !neighbor_list ||
      !csf_to_bucket || !target_offsets || !target_list || !target_offsets_1b || !target_list_1b ||
      !workspace_keys || !workspace_vals || !row_counts || !diag_out || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024 || grid_blocks <= 0) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)grid_blocks);
  const size_t FLAT_MAX_INTERM = 4096;
  const size_t MEMBER_BATCH_SZ = 32;

#define LAUNCH_COUNT_BUCKETED_(NORB_T) \
    do { \
      size_t base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)); \
      size_t member_smem = 16 + MEMBER_BATCH_SZ * ( \
          sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t) + \
          sizeof(uint64_t)); \
      size_t dyn_smem_bytes = base_smem + member_smem; \
      auto _kfn = pairwise_emit_bucketed_u64_kernel<NORB_T, false>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, occ_keys_sorted, \
          bucket_keys, bucket_starts, bucket_sizes, neighbor_offsets, neighbor_list, \
          csf_to_bucket, target_offsets, target_list, target_offsets_1b, target_list_1b, \
          table_cap, workspace_keys, workspace_vals, \
          nullptr, row_counts, nullptr, nullptr, nullptr, diag_out, overflow_flag); \
    } while(0)

  if (norb <= 8) {
    LAUNCH_COUNT_BUCKETED_(8);
  } else if (norb <= 16) {
    LAUNCH_COUNT_BUCKETED_(16);
  } else if (norb <= 24) {
    LAUNCH_COUNT_BUCKETED_(24);
  } else if (norb <= 32) {
    LAUNCH_COUNT_BUCKETED_(32);
  } else if (norb <= 48) {
    LAUNCH_COUNT_BUCKETED_(48);
  } else {
    LAUNCH_COUNT_BUCKETED_(64);
  }

#undef LAUNCH_COUNT_BUCKETED_

  return cudaGetLastError();
}

extern "C" cudaError_t pairwise_fill_bucketed_u64_launch_stream(
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
    const uint64_t* occ_keys_sorted,
    const uint64_t* bucket_keys,
    const int32_t* bucket_starts,
    const int32_t* bucket_sizes,
    const int32_t* neighbor_offsets,
    const int32_t* neighbor_list,
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets,
    const int32_t* target_list,
    const int32_t* target_offsets_1b,
    const int32_t* target_list_1b,
    int table_cap,
    int32_t* workspace_keys,
    double* workspace_vals,
    int grid_blocks,
    const int64_t* row_offsets,
    const int32_t* row_counts,
    int32_t* out_target_local,
    int32_t* out_src_pos,
    double* out_hij,
    double* diag_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !occ_keys_sorted ||
      !bucket_keys || !bucket_starts || !bucket_sizes || !neighbor_offsets || !neighbor_list ||
      !csf_to_bucket || !target_offsets || !target_list || !target_offsets_1b || !target_list_1b ||
      !workspace_keys || !workspace_vals || !row_offsets || !row_counts ||
      !out_target_local || !out_src_pos || !out_hij || !diag_out || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024 || grid_blocks <= 0) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)grid_blocks);
  const size_t FLAT_MAX_INTERM = 4096;
  const size_t MEMBER_BATCH_SZ = 32;

#define LAUNCH_FILL_BUCKETED_(NORB_T) \
    do { \
      size_t base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)); \
      size_t member_smem = 16 + MEMBER_BATCH_SZ * ( \
          sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t) + \
          sizeof(uint64_t)); \
      size_t dyn_smem_bytes = base_smem + member_smem; \
      auto _kfn = pairwise_emit_bucketed_u64_kernel<NORB_T, true>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, occ_keys_sorted, \
          bucket_keys, bucket_starts, bucket_sizes, neighbor_offsets, neighbor_list, \
          csf_to_bucket, target_offsets, target_list, target_offsets_1b, target_list_1b, \
          table_cap, workspace_keys, workspace_vals, \
          row_offsets, const_cast<int32_t*>(row_counts), out_target_local, out_src_pos, out_hij, diag_out, overflow_flag); \
    } while(0)

  if (norb <= 8) {
    LAUNCH_FILL_BUCKETED_(8);
  } else if (norb <= 16) {
    LAUNCH_FILL_BUCKETED_(16);
  } else if (norb <= 24) {
    LAUNCH_FILL_BUCKETED_(24);
  } else if (norb <= 32) {
    LAUNCH_FILL_BUCKETED_(32);
  } else if (norb <= 48) {
    LAUNCH_FILL_BUCKETED_(48);
  } else {
    LAUNCH_FILL_BUCKETED_(64);
  }

#undef LAUNCH_FILL_BUCKETED_

  return cudaGetLastError();
}

extern "C" cudaError_t pairwise_sigma_bucketed_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* h_base,
    const double* eri4,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_all,
    const int32_t* nodes_all,
    const int8_t* occ_all,
    const int16_t* b_all,
    const uint64_t* occ_keys_sorted,
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets,
    const int32_t* target_list,
    const int32_t* target_offsets_1b,
    const int32_t* target_list_1b,
    const double* x,
    int nvec,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !h_base || !eri4 || !child_table || !node_twos || !child_prefix ||
      !steps_all || !nodes_all || !occ_all || !b_all || !occ_keys_sorted || !csf_to_bucket ||
      !target_offsets || !target_list || !target_offsets_1b || !target_list_1b ||
      !x || !y || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || nvec <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  const size_t FLAT_MAX_INTERM = 4096;
  const size_t MEMBER_BATCH_SZ = 32;

#define LAUNCH_SIGMA_BUCKETED_(NORB_T) \
    do { \
      size_t flat_base_smem = (size_t)norb * (size_t)norb * sizeof(double) \
          + FLAT_MAX_INTERM * (sizeof(uint64_t) + sizeof(double) + 2 * sizeof(int8_t)); \
      size_t flat_member_smem = 16 + MEMBER_BATCH_SZ * ( \
          sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          (NORB_T) * sizeof(int16_t) + \
          ((NORB_T) + 1) * sizeof(int32_t) + \
          (NORB_T) * sizeof(int8_t) + \
          sizeof(int8_t) + \
          sizeof(uint64_t)); \
      size_t dyn_smem_bytes = flat_base_smem + flat_member_smem; \
      auto _kfn = pairwise_sigma_bucketed_u64_kernel<NORB_T>; \
      cudaFuncSetAttribute(_kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes); \
      _kfn<<<grid, block, dyn_smem_bytes, stream>>>( \
          sel_idx_u64, nsel, norb, ncsf, h_base, eri4, \
          child_table, node_twos, child_prefix, \
          steps_all, nodes_all, occ_all, b_all, occ_keys_sorted, \
          csf_to_bucket, target_offsets, target_list, target_offsets_1b, target_list_1b, \
          x, nvec, y, overflow_flag); \
    } while (0)

  if (norb <= 8) {
    LAUNCH_SIGMA_BUCKETED_(8);
  } else if (norb <= 16) {
    LAUNCH_SIGMA_BUCKETED_(16);
  } else if (norb <= 24) {
    LAUNCH_SIGMA_BUCKETED_(24);
  } else if (norb <= 32) {
    LAUNCH_SIGMA_BUCKETED_(32);
  } else if (norb <= 48) {
    LAUNCH_SIGMA_BUCKETED_(48);
  } else {
    LAUNCH_SIGMA_BUCKETED_(64);
  }

#undef LAUNCH_SIGMA_BUCKETED_

  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// T-matrix kernel: computes T[pq, i] = sum_j E_pq[i,j] * c[j]
// for all orbital pairs (p,q) and all selected CSFs i, using the same
// bucketed occupation-key approach as the sigma kernel (Phase 1 only).
// No h_base or eri4 needed — raw GUGA coupling coefficients only.
// ---------------------------------------------------------------------------
template <int MAX_NORB_T>
__global__ void pairwise_T_matrix_bucketed_u64_kernel(
    int nsel,
    int norb,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int8_t*  __restrict__ steps_all,         // [nsel, norb] sorted by occ key
    const int32_t* __restrict__ nodes_all,         // [nsel, norb+1] sorted
    const int8_t*  __restrict__ occ_all,           // [nsel, norb] sorted
    const int16_t* __restrict__ b_all,             // [nsel, norb] sorted
    const int32_t* __restrict__ csf_to_bucket,     // [nsel]
    const int32_t* __restrict__ target_offsets_1b, // [nbuckets+1]
    const int32_t* __restrict__ target_list_1b,    // [total_targets_1b]
    const double*  __restrict__ ci_sel,            // [nsel] CI coefficients (sorted)
    double*        __restrict__ T_out) {            // [norb*norb, nsel] output

  int j_local = blockIdx.x;
  if (j_local >= nsel || norb > MAX_NORB_T) return;

  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Load source CSF j data into shared memory
  __shared__ int8_t  steps_j_s[MAX_NORB_T];
  __shared__ int8_t  occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ double  c_j_s;

  for (int k = tid; k < norb; k += nthreads) {
    steps_j_s[k] = steps_all[(int64_t)j_local * norb + k];
    occ_j_s[k]   = occ_all[(int64_t)j_local * norb + k];
    b_j_s[k]     = b_all[(int64_t)j_local * norb + k];
  }
  for (int k = tid; k < norb + 1; k += nthreads) {
    nodes_j_s[k] = nodes_all[(int64_t)j_local * (norb + 1) + k];
  }
  if (tid == 0) c_j_s = ci_sel[j_local];
  __syncthreads();

  double c_j = c_j_s;

  // Diagonal: T[pp, j] += occ_p[j] * c_j for all p
  for (int p = tid; p < norb; p += nthreads) {
    int occ_p = (int)occ_j_s[p];
    if (occ_p != 0) {
      atomicAdd(&T_out[(int64_t)(p * norb + p) * nsel + j_local], (double)occ_p * c_j);
    }
  }

  // Off-diagonal phase 1: 1-body targets (occ_diff == 2 in packed encoding)
  int j_bucket = csf_to_bucket[j_local];
  int tgt_start_1b = target_offsets_1b[j_bucket];
  int tgt_end_1b   = target_offsets_1b[j_bucket + 1];

  for (int t_idx = tgt_start_1b + tid; t_idx < tgt_end_1b; t_idx += nthreads) {
    int i_local = target_list_1b[t_idx];
    if (i_local <= j_local) continue;  // process each pair once; apply symmetry below

    const int8_t*  steps_i = steps_all + (int64_t)i_local * norb;
    const int16_t* b_i     = b_all     + (int64_t)i_local * norb;

    // Identify the loop segment [diff_min, diff_max] where steps differ
    int diff_min = norb, diff_max = -1;
    for (int k = 0; k < norb; ++k) {
      if (steps_i[k] != steps_j_s[k]) {
        if (diff_min > k) diff_min = k;
        diff_max = k;
      }
    }
    if (diff_min > diff_max) continue;

    double c_i = ci_sel[i_local];
    int pm_flat = diff_min * norb + diff_max;
    int mp_flat = diff_max * norb + diff_min;

    // E_{diff_min, diff_max}[i,j]: creation at diff_min, annihilation at diff_max
    double coupling_pm = pairwise_compute_epq_coupling<MAX_NORB_T>(
        norb, diff_min, diff_max,
        steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
    if (coupling_pm != 0.0) {
      // T[pm, i] += E_pm[i,j] * c_j
      atomicAdd(&T_out[(int64_t)pm_flat * nsel + i_local], coupling_pm * c_j);
      // T[mp, j] += E_pm[j,i] * c_i  (E_pm[j,i] = E_mp[i,j] for real wfn)
      atomicAdd(&T_out[(int64_t)mp_flat * nsel + j_local], coupling_pm * c_i);
    }

    if (diff_min != diff_max) {
      // E_{diff_max, diff_min}[i,j]: creation at diff_max, annihilation at diff_min
      double coupling_mp = pairwise_compute_epq_coupling<MAX_NORB_T>(
          norb, diff_max, diff_min,
          steps_i, b_i, steps_j_s, b_j_s, nodes_j_s, node_twos);
      if (coupling_mp != 0.0) {
        // T[mp, i] += E_mp[i,j] * c_j
        atomicAdd(&T_out[(int64_t)mp_flat * nsel + i_local], coupling_mp * c_j);
        // T[pm, j] += E_mp[j,i] * c_i  (E_mp[j,i] = E_pm[i,j] for real wfn)
        atomicAdd(&T_out[(int64_t)pm_flat * nsel + j_local], coupling_mp * c_i);
      }
    }
  }
}

extern "C" cudaError_t pairwise_T_matrix_bucketed_u64_launch_stream(
    int nsel,
    int norb,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int8_t*  steps_all,
    const int32_t* nodes_all,
    const int8_t*  occ_all,
    const int16_t* b_all,
    const int32_t* csf_to_bucket,
    const int32_t* target_offsets_1b,
    const int32_t* target_list_1b,
    const double*  ci_sel,
    double*        T_out,
    cudaStream_t stream,
    int threads) {
  if (!child_table || !node_twos || !steps_all || !nodes_all || !occ_all || !b_all ||
      !csf_to_bucket || !target_offsets_1b || !target_list_1b || !ci_sel || !T_out) {
    return cudaErrorInvalidValue;
  }
  if (nsel <= 0 || norb <= 0 || norb > 64 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);

#define LAUNCH_T_MATRIX_(NORB_T) \
    pairwise_T_matrix_bucketed_u64_kernel<NORB_T><<<grid, block, 0, stream>>>( \
        nsel, norb, child_table, node_twos, \
        steps_all, nodes_all, occ_all, b_all, \
        csf_to_bucket, target_offsets_1b, target_list_1b, \
        ci_sel, T_out)

  if      (norb <= 8)  { LAUNCH_T_MATRIX_(8);  }
  else if (norb <= 16) { LAUNCH_T_MATRIX_(16); }
  else if (norb <= 24) { LAUNCH_T_MATRIX_(24); }
  else if (norb <= 32) { LAUNCH_T_MATRIX_(32); }
  else if (norb <= 48) { LAUNCH_T_MATRIX_(48); }
  else                 { LAUNCH_T_MATRIX_(64); }

#undef LAUNCH_T_MATRIX_

  return cudaGetLastError();
}
