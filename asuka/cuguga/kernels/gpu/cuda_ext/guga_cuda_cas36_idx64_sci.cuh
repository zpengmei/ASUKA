
#pragma once

// CAS(36,36) / idx64 SCI helper kernels for ASUKA.
//
// This file provides the large-space pieces that the old CUDA SCI path was
// missing:
//   - u64 frontier hash (external labels are uint64 global CSF indices)
//   - on-demand source-path reconstruction from (child, child_prefix, idx64)
//   - one-root frontier-hash CIPSI builder
//   - one-root heat-bath builder
//   - compact candidate diagonal-guess kernel
//   - compact candidate score/PT2 kernel
//
// This is intentionally a "core kernel bundle", not a full pybind patch.
// The host wrapper integration is spelled out in cas36_unblock_impl.md.
//
// Semantics:
//   * device labels are uint64_t, representing nonnegative global CSF indices
//   * selected-set membership is via a sorted uint64 list, not a dense mask
//   * hdiag is compact/candidate-sized, not dense over ncsf
//
// Dependencies:
//   Expects the usual GUGA EPQ helpers/constants to be visible from the
//   aggregating TU (same as the existing CUDA SCI/QMC code path).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ bool cas36_sci_contains_sorted_u64(
    const uint64_t* __restrict__ a,
    int n,
    uint64_t key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    uint64_t v = a[mid];
    if (v < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < n) && (a[lo] == key);
}

__device__ __forceinline__ uint64_t cas36_sci_hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}

__device__ __forceinline__ void cas36_exact_emit_tuple_u64(
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_src,
    double* __restrict__ out_hij,
    int cap,
    uint64_t key,
    int src_local,
    double hij,
    int* __restrict__ out_n,
    int* __restrict__ overflow_flag) {
  if (hij == 0.0) return;
  int slot = atomicAdd(out_n, 1);
  if (slot < 0 || slot >= cap) {
    if (overflow_flag) atomicExch(overflow_flag, 1);
    return;
  }
  out_keys[slot] = key;
  out_src[slot] = src_local;
  out_hij[slot] = hij;
}

__device__ __forceinline__ double cas36_dense_eri4_at(
    const double* __restrict__ eri4,
    int norb,
    int p,
    int q,
    int r,
    int s) {
  int64_t idx = ((((int64_t)p * (int64_t)norb) + (int64_t)q) * (int64_t)norb + (int64_t)r) * (int64_t)norb +
      (int64_t)s;
  return eri4[idx];
}

template <int MAX_NORB_T>
__device__ __forceinline__ bool cas36_sci_reconstruct_path_from_index_u64(
    const int32_t* __restrict__ child,
    const int64_t* __restrict__ child_prefix,
    int norb,
    uint64_t ncsf,
    uint64_t csf_idx,
    int8_t* __restrict__ steps_out,
    int32_t* __restrict__ nodes_out) {
  if (norb > MAX_NORB_T) return false;
  if (csf_idx >= ncsf) return false;

  int32_t node = 0;
  uint64_t idx = csf_idx;
  nodes_out[0] = node;

  for (int k = 0; k < norb; ++k) {
    int step = -1;
    int32_t next = -1;
    for (int s = 0; s < 4; ++s) {
      int32_t child_s = child[node * 4 + s];
      if (child_s < 0) continue;
      uint64_t lo = (uint64_t)child_prefix[node * 5 + s];
      uint64_t hi = (uint64_t)child_prefix[node * 5 + (s + 1)];
      if (idx >= lo && idx < hi) {
        step = s;
        next = child_s;
        idx -= lo;
        break;
      }
    }
    if (step < 0 || next < 0) return false;
    steps_out[k] = (int8_t)step;
    node = next;
    nodes_out[k + 1] = node;
  }
  return true;
}

template <int MAX_NORB_T>
__device__ __forceinline__ void cas36_exact_emit_weighted_epq_selected_u64(
    uint64_t source_global_exclude,
    int source_local,
    uint64_t state_global,
    int norb,
    uint64_t ncsf,
    const int8_t* __restrict__ steps_s,
    const int32_t* __restrict__ nodes_s,
    const int8_t* __restrict__ occ_s,
    const int16_t* __restrict__ b_s,
    const uint64_t* __restrict__ idx_prefix_s,
    int p,
    int q,
    double weight,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_src,
    double* __restrict__ out_hij,
    int cap,
    int* __restrict__ out_n,
    int* __restrict__ overflow_flag) {
  if (weight == 0.0) return;
  int occ_p = (int)occ_s[p];
  int occ_q = (int)occ_s[q];
  if (occ_q <= 0 || occ_p >= 2) return;

  int start, end, q_start, q_mid, q_end;
  if (p < q) {
    start = p;
    end = q;
    q_start = Q_uR;
    q_mid = Q_R;
    q_end = Q_oR;
  } else {
    start = q;
    end = p;
    q_start = Q_uL;
    q_mid = Q_L;
    q_end = Q_oL;
  }

  int32_t node_start = nodes_s[start];
  int32_t node_end_target = nodes_s[end + 1];
  uint64_t prefix_offset = idx_prefix_s[start];
  uint64_t prefix_endplus1 = idx_prefix_s[end + 1];
  if (state_global < prefix_endplus1) return;
  uint64_t suffix_offset = state_global - prefix_endplus1;

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
    int dk = (int)steps_s[kpos];
    int bk = (int)b_s[kpos];
    int k_next = kpos + 1;

    int dp0 = 0;
    int dp1 = 0;
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
      double w2 = w * seg;
      uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];
      if (kpos == end) {
        if (child_k != node_end_target) continue;
        uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
        if (csf_i >= ncsf) {
          if (overflow_flag) atomicExch(overflow_flag, 1);
          continue;
        }
        if (csf_i == source_global_exclude) continue;
        if (!selected_idx_sorted_u64 ||
            !cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, csf_i)) {
          continue;
        }
        cas36_exact_emit_tuple_u64(
            out_keys, out_src, out_hij, cap, csf_i, source_local, weight * w2, out_n, overflow_flag);
      } else {
        if (top >= MAX_NORB_T) {
          if (overflow_flag) atomicExch(overflow_flag, 1);
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        ++top;
      }
    }
  }
}

template <int MAX_NORB_T>
__device__ __forceinline__ double cas36_exact_accumulate_weighted_epq_self_u64(
    uint64_t target_global_match,
    uint64_t state_global,
    int norb,
    uint64_t ncsf,
    const int8_t* __restrict__ steps_s,
    const int32_t* __restrict__ nodes_s,
    const int8_t* __restrict__ occ_s,
    const int16_t* __restrict__ b_s,
    const uint64_t* __restrict__ idx_prefix_s,
    int p,
    int q,
    double weight,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix) {
  if (weight == 0.0) return 0.0;
  int occ_p = (int)occ_s[p];
  int occ_q = (int)occ_s[q];
  if (occ_q <= 0 || occ_p >= 2) return 0.0;

  int start, end, q_start, q_mid, q_end;
  if (p < q) {
    start = p;
    end = q;
    q_start = Q_uR;
    q_mid = Q_R;
    q_end = Q_oR;
  } else {
    start = q;
    end = p;
    q_start = Q_uL;
    q_mid = Q_L;
    q_end = Q_oL;
  }

  int32_t node_start = nodes_s[start];
  int32_t node_end_target = nodes_s[end + 1];
  uint64_t prefix_offset = idx_prefix_s[start];
  uint64_t prefix_endplus1 = idx_prefix_s[end + 1];
  if (state_global < prefix_endplus1) return 0.0;
  uint64_t suffix_offset = state_global - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  uint64_t st_seg[MAX_NORB_T];
  int top = 0;
  double accum = 0.0;
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
    int dk = (int)steps_s[kpos];
    int bk = (int)b_s[kpos];
    int k_next = kpos + 1;

    int dp0 = 0;
    int dp1 = 0;
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
      double w2 = w * seg;
      uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];
      if (kpos == end) {
        if (child_k != node_end_target) continue;
        uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
        if (csf_i >= ncsf) continue;
        if (csf_i == target_global_match) accum += weight * w2;
      } else {
        if (top >= MAX_NORB_T) continue;
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        ++top;
      }
    }
  }
  return accum;
}

template <int MAX_PROBES_T = 256>
__device__ __forceinline__ void cas36_frontier_hash_insert_add_f64_u64(
    uint64_t* __restrict__ keys,
    double* __restrict__ vals_root_major,
    int cap,
    int root,
    uint64_t idx_u64,
    double v,
    int* __restrict__ overflow_flag) {
  if (v == 0.0) return;
  if (!keys || !vals_root_major || cap <= 0) {
    if (overflow_flag) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t mask = (uint64_t)(cap - 1);
  uint64_t slot = cas36_sci_hash_u64(idx_u64) & mask;
  constexpr uint64_t EMPTY = 0xffffffffffffffffull;

  for (int probe = 0; probe < MAX_PROBES_T; ++probe) {
    uint64_t cur = keys[slot];
    if (cur == idx_u64) {
      atomicAdd(&vals_root_major[(int64_t)root * (int64_t)cap + (int64_t)slot], v);
      return;
    }
    if (cur == EMPTY) {
      unsigned long long* ptr = reinterpret_cast<unsigned long long*>(&keys[slot]);
      unsigned long long prev = atomicCAS(ptr, (unsigned long long)EMPTY, (unsigned long long)idx_u64);
      if (prev == (unsigned long long)EMPTY || prev == (unsigned long long)idx_u64) {
        atomicAdd(&vals_root_major[(int64_t)root * (int64_t)cap + (int64_t)slot], v);
        return;
      }
    }
    slot = (slot + 1ull) & mask;
  }

  if (overflow_flag) atomicExch(overflow_flag, 1);
}

__global__ void cas36_frontier_hash_clear_keys_u64_kernel(uint64_t* keys, int cap) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  keys[i] = 0xffffffffffffffffull;
}

__global__ void cas36_frontier_hash_clear_vals_kernel(double* vals, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= n) return;
  vals[i] = 0.0;
}

__global__ void cas36_frontier_hash_flags_u64_kernel(
    const uint64_t* keys,
    int cap,
    int* flags) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  flags[i] = (keys[i] != 0xffffffffffffffffull) ? 1 : 0;
}

__global__ void cas36_frontier_hash_scatter_extract_u64_kernel(
    const uint64_t* __restrict__ keys,
    const double* __restrict__ vals_root_major,
    int cap,
    int nroots,
    const int* __restrict__ offsets,
    const int* __restrict__ flags,
    uint64_t* __restrict__ out_idx_u64,
    double* __restrict__ out_vals_root_major) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  if (!flags[i]) return;

  int pos = offsets[i];
  out_idx_u64[pos] = keys[i];
  for (int r = 0; r < nroots; ++r) {
    out_vals_root_major[(int64_t)r * (int64_t)cap + (int64_t)pos] =
        vals_root_major[(int64_t)r * (int64_t)cap + (int64_t)i];
  }
}

// ---------------------------------------------------------------------------
// Frontier-hash builder (one root)
// ---------------------------------------------------------------------------

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void cas36_apply_g_flat_scatter_atomic_frontier_hash_u64_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* __restrict__ task_csf_u64,   // [ntasks]
    const double* __restrict__ task_scale,       // [ntasks] or NULL
    const double* __restrict__ task_g,           // [nops] or [ntasks,nops]
    int64_t g_stride,                            // 0 or nops
    int ntasks,
    uint64_t* __restrict__ hash_keys,            // [cap]
    double* __restrict__ hash_vals,              // [nroots*cap]
    int cap,
    int root,
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    int* __restrict__ overflow_flag) {
  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t csf_idx = task_csf_u64[task_id];
  if (csf_idx >= ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double scale = task_scale ? task_scale[task_id] : 1.0;
  if (scale == 0.0) return;

  int nops = norb * norb;
  const double* g_flat = task_g + (int64_t)task_id * g_stride;

  __shared__ int8_t steps_s[MAX_NORB_T];
  __shared__ int32_t nodes_s[MAX_NORB_T + 1];
  __shared__ int8_t occ_s[MAX_NORB_T];
  __shared__ int16_t b_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_s[MAX_NORB_T + 1];

  bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
      child, child_prefix, norb, ncsf, csf_idx, steps_s, nodes_s);
  if (!ok) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int k = threadIdx.x; k < norb; k += blockDim.x) {
    int8_t st = steps_s[k];
    occ_s[k] = (int8_t)step_to_occ(st);
    b_s[k] = node_twos[nodes_s[k + 1]];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    idx_prefix_s[0] = 0ull;
    for (int k = 0; k < norb; ++k) {
      int node_k = nodes_s[k];
      int step_k = (int)steps_s[k];
      idx_prefix_s[k + 1] = idx_prefix_s[k] + (uint64_t)child_prefix[node_k * 5 + step_k];
    }
  }
  __syncthreads();

  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    double wgt = g_flat[pq];
    if (wgt == 0.0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start, end, q_start, q_mid, q_end;
    if (p < q) {
      start = p; end = q;
      q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
    } else {
      start = q; end = p;
      q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];
    uint64_t prefix_offset = idx_prefix_s[start];
    uint64_t prefix_endplus1 = idx_prefix_s[end + 1];
    if (csf_idx < prefix_endplus1) continue;
    uint64_t suffix_offset = csf_idx - prefix_endplus1;

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
      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
      int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
      if (ndp == 0) continue;

      for (int which = 0; which < ndp; ++which) {
        int dprime = (which == 0) ? dp0 : dp1;
        int child_k = child[node_k * 4 + dprime];
        if (child_k < 0) continue;
        int bprime = (int)node_twos[child_k];
        int db = bk - bprime;
        double seg = (double)segment_value_int(qk, dprime, dk, db, bk);
        if (seg == 0.0) continue;
        double w2 = w * seg;
        uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];

        if (kpos == end) {
          if (child_k != node_end_target) continue;
          uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if (csf_i >= ncsf) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          if (csf_i == csf_idx) continue;
          if (selected_idx_sorted_u64 &&
              cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, csf_i)) {
            continue;
          }
          if (w2 == 0.0) continue;
          cas36_frontier_hash_insert_add_f64_u64(
              hash_keys, hash_vals, cap, root, csf_i, scale * wgt * w2, overflow_flag);
        } else {
          if (top >= MAX_NORB_T) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          ++top;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Heat-bath SCI builder (one root)
// ---------------------------------------------------------------------------

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void cas36_hb_screen_and_apply_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,   // [nsel]
    const double* __restrict__ c_root,          // [nsel]
    int nsel,
    int root,
    int norb,
    uint64_t ncsf,
    const int32_t* __restrict__ h1_pq,
    const double* __restrict__ h1_abs,
    const double* __restrict__ h1_signed,
    int n_h1,
    const int64_t* __restrict__ pq_ptr,
    const int32_t* __restrict__ rs_idx,
    const double* __restrict__ v_abs,
    const double* __restrict__ v_signed,
    const double* __restrict__ pq_max_v,
    double eps,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    uint64_t* __restrict__ hash_keys,
    double* __restrict__ hash_vals,
    int cap,
    uint64_t label_lo,
    uint64_t label_hi,
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    int target_mode,
    int* __restrict__ overflow_flag) {
  int j_local = blockIdx.x;
  if (j_local >= nsel) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t j_global = sel_idx_u64[j_local];
  double cj = c_root[j_local];
  double abs_cj = fabs(cj);
  if (abs_cj == 0.0) return;
  if (j_global >= ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double cutoff = eps / abs_cj;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;

  extern __shared__ char smem_raw[];
  auto align_up_ptr = [](uintptr_t x, uintptr_t a) -> uintptr_t {
    return (x + (a - 1u)) & ~(a - 1u);
  };
  char* smem_p = smem_raw;

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(double)));
  double* g_flat_s = reinterpret_cast<double*>(smem_p);
  smem_p += (size_t)nops * sizeof(double);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int8_t)));
  int8_t* steps_s = reinterpret_cast<int8_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int8_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int32_t)));
  int32_t* nodes_s = reinterpret_cast<int32_t*>(smem_p);
  smem_p += (size_t)(norb + 1) * sizeof(int32_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int8_t)));
  int8_t* occ_s = reinterpret_cast<int8_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int8_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int16_t)));
  int16_t* b_s = reinterpret_cast<int16_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int16_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(uint64_t)));
  uint64_t* idx_prefix_s = reinterpret_cast<uint64_t*>(smem_p);

  bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
      child_table, child_prefix, norb, ncsf, j_global, steps_s, nodes_s);
  if (!ok) {
    if (tid == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int k = tid; k < norb; k += nthreads) {
    int8_t st = steps_s[k];
    occ_s[k] = (int8_t)step_to_occ(st);
    b_s[k] = node_twos[nodes_s[k + 1]];
  }
  __syncthreads();

  if (tid == 0) {
    idx_prefix_s[0] = 0ull;
    for (int k = 0; k < norb; ++k) {
      int node_k = nodes_s[k];
      int step_k = (int)steps_s[k];
      idx_prefix_s[k + 1] = idx_prefix_s[k] + (uint64_t)child_prefix[node_k * 5 + step_k];
    }
  }
  __syncthreads();

  for (int i = tid; i < nops; i += nthreads) g_flat_s[i] = 0.0;
  __syncthreads();

  for (int k = tid; k < n_h1; k += nthreads) {
    if (h1_abs[k] < cutoff) break;
    int p = h1_pq[k * 2];
    int q = h1_pq[k * 2 + 1];
    g_flat_s[p * norb + q] = h1_signed[k];
  }
  __syncthreads();

  for (int pq = tid; pq < nops; pq += nthreads) {
    if (pq_max_v[pq] < cutoff) continue;
    int64_t lo = pq_ptr[pq];
    int64_t hi = pq_ptr[pq + 1];
    if (lo >= hi) continue;

    double g_acc = 0.0;
    for (int64_t k = lo; k < hi; ++k) {
      if (v_abs[k] < cutoff) break;
      int rs_flat = rs_idx[k];
      int r = rs_flat / norb;
      int s = rs_flat - r * norb;
      double v = v_signed[k];
      if (r == s) {
        g_acc += 0.5 * (double)occ_s[r] * v;
      } else {
        g_acc += 0.5 * v;
      }
    }
    g_flat_s[pq] += g_acc;
  }
  __syncthreads();

  for (int pq = tid; pq < nops; pq += nthreads) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    double wgt = g_flat_s[pq];
    if (wgt == 0.0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start, end, q_start, q_mid, q_end;
    if (p < q) {
      start = p; end = q;
      q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
    } else {
      start = q; end = p;
      q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];
    uint64_t prefix_offset = idx_prefix_s[start];
    uint64_t prefix_endplus1 = idx_prefix_s[end + 1];
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
      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
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
        double w2 = w * seg;
        uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];

        if (kpos == end) {
          if (child_k != node_end_target) continue;
          uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if (csf_i >= ncsf) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          if (csf_i < label_lo || csf_i >= label_hi) continue;
          if (target_mode == 0 && csf_i == j_global) continue;
          bool in_selected = false;
          if (selected_idx_sorted_u64) {
            in_selected = cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, csf_i);
          }
          if (target_mode == 0) {
            if (in_selected) continue;
          } else if (target_mode == 1) {
            if (!in_selected) continue;
          }
          if (w2 == 0.0) continue;

          cas36_frontier_hash_insert_add_f64_u64(
              hash_keys, hash_vals, cap, root, csf_i, cj * wgt * w2, overflow_flag);
        } else {
          if (top >= MAX_NORB_T) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          ++top;
        }
      }
    }
  }
}

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void cas36_hb_emit_tuples_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,   // [nsel]
    const double* __restrict__ c_bound,         // [nsel]
    int nsel,
    int norb,
    uint64_t ncsf,
    const int32_t* __restrict__ h1_pq,
    const double* __restrict__ h1_abs,
    const double* __restrict__ h1_signed,
    int n_h1,
    const int64_t* __restrict__ pq_ptr,
    const int32_t* __restrict__ rs_idx,
    const double* __restrict__ v_abs,
    const double* __restrict__ v_signed,
    const double* __restrict__ pq_max_v,
    double eps,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_src,
    double* __restrict__ out_hij,
    int cap,
    uint64_t label_lo,
    uint64_t label_hi,
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    int target_mode,
    int* __restrict__ out_n,
    int* __restrict__ overflow_flag) {
  int j_local = blockIdx.x;
  if (j_local >= nsel) return;
  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  uint64_t j_global = sel_idx_u64[j_local];
  double cj = c_bound[j_local];
  double abs_cj = fabs(cj);
  if (abs_cj == 0.0) return;
  if (j_global >= ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double cutoff = eps / abs_cj;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;

  extern __shared__ char smem_raw[];
  auto align_up_ptr = [](uintptr_t x, uintptr_t a) -> uintptr_t {
    return (x + (a - 1u)) & ~(a - 1u);
  };
  char* smem_p = smem_raw;

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(double)));
  double* g_flat_s = reinterpret_cast<double*>(smem_p);
  smem_p += (size_t)nops * sizeof(double);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int8_t)));
  int8_t* steps_s = reinterpret_cast<int8_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int8_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int32_t)));
  int32_t* nodes_s = reinterpret_cast<int32_t*>(smem_p);
  smem_p += (size_t)(norb + 1) * sizeof(int32_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int8_t)));
  int8_t* occ_s = reinterpret_cast<int8_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int8_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(int16_t)));
  int16_t* b_s = reinterpret_cast<int16_t*>(smem_p);
  smem_p += (size_t)norb * sizeof(int16_t);

  smem_p = reinterpret_cast<char*>(align_up_ptr((uintptr_t)smem_p, (uintptr_t)alignof(uint64_t)));
  uint64_t* idx_prefix_s = reinterpret_cast<uint64_t*>(smem_p);

  bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
      child_table, child_prefix, norb, ncsf, j_global, steps_s, nodes_s);
  if (!ok) {
    if (tid == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int k = tid; k < norb; k += nthreads) {
    int8_t st = steps_s[k];
    occ_s[k] = (int8_t)step_to_occ(st);
    b_s[k] = node_twos[nodes_s[k + 1]];
  }
  __syncthreads();

  if (tid == 0) {
    idx_prefix_s[0] = 0ull;
    for (int k = 0; k < norb; ++k) {
      int node_k = nodes_s[k];
      int step_k = (int)steps_s[k];
      idx_prefix_s[k + 1] = idx_prefix_s[k] + (uint64_t)child_prefix[node_k * 5 + step_k];
    }
  }
  __syncthreads();

  for (int i = tid; i < nops; i += nthreads) g_flat_s[i] = 0.0;
  __syncthreads();

  for (int k = tid; k < n_h1; k += nthreads) {
    if (h1_abs[k] < cutoff) break;
    int p = h1_pq[k * 2];
    int q = h1_pq[k * 2 + 1];
    g_flat_s[p * norb + q] = h1_signed[k];
  }
  __syncthreads();

  for (int pq = tid; pq < nops; pq += nthreads) {
    if (pq_max_v[pq] < cutoff) continue;
    int64_t lo = pq_ptr[pq];
    int64_t hi = pq_ptr[pq + 1];
    if (lo >= hi) continue;

    double g_acc = 0.0;
    for (int64_t k = lo; k < hi; ++k) {
      if (v_abs[k] < cutoff) break;
      int rs_flat = rs_idx[k];
      int r = rs_flat / norb;
      int s = rs_flat - r * norb;
      double v = v_signed[k];
      if (r == s) {
        g_acc += 0.5 * (double)occ_s[r] * v;
      } else {
        g_acc += 0.5 * v;
      }
    }
    g_flat_s[pq] += g_acc;
  }
  __syncthreads();

  for (int pq = tid; pq < nops; pq += nthreads) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    double wgt = g_flat_s[pq];
    if (wgt == 0.0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start, end, q_start, q_mid, q_end;
    if (p < q) {
      start = p; end = q;
      q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
    } else {
      start = q; end = p;
      q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];
    uint64_t prefix_offset = idx_prefix_s[start];
    uint64_t prefix_endplus1 = idx_prefix_s[end + 1];
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
      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
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
        double w2 = w * seg;
        uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];

        if (kpos == end) {
          if (child_k != node_end_target) continue;
          uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if (csf_i >= ncsf) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          if (csf_i < label_lo || csf_i >= label_hi) continue;
          if (target_mode == 0 && csf_i == j_global) continue;
          bool in_selected = false;
          if (selected_idx_sorted_u64) {
            in_selected = cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, csf_i);
          }
          if (target_mode == 0) {
            if (in_selected) continue;
          } else if (target_mode == 1) {
            if (!in_selected) continue;
          }
          if (w2 == 0.0) continue;

          int slot = atomicAdd(out_n, 1);
          if (slot < 0 || slot >= cap) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          out_keys[slot] = csf_i;
          out_src[slot] = j_local;
          out_hij[slot] = wgt * w2;
        } else {
          if (top >= MAX_NORB_T) {
            if (overflow_flag) atomicExch(overflow_flag, 1);
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          ++top;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Compact diagonal guess on candidate list (dense-ERI path)
// ---------------------------------------------------------------------------

template <int MAX_NORB_T>
__global__ void cas36_diag_guess_candidates_u64_dense_kernel(
    const int32_t* __restrict__ child,
    const int64_t* __restrict__ child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* __restrict__ cand_idx_u64,  // [ncand]
    int ncand,
    const double* __restrict__ h1_diag,         // [norb]
    const double* __restrict__ eri_ppqq,        // [norb*norb]
    const double* __restrict__ eri_pqqp,        // [norb*norb]
    int neleca,
    int nelecb,
    double* __restrict__ hdiag_out) {           // [ncand]
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= ncand) return;
  if (norb > MAX_NORB_T) return;

  uint64_t idx = cand_idx_u64[i];
  if (idx >= ncsf) {
    hdiag_out[i] = 0.0;
    return;
  }

  int8_t steps[MAX_NORB_T];
  int32_t nodes[MAX_NORB_T + 1];
  if (!cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
          child, child_prefix, norb, ncsf, idx, steps, nodes)) {
    hdiag_out[i] = 0.0;
    return;
  }

  int occ[MAX_NORB_T];
  int alpha[MAX_NORB_T];
  int beta[MAX_NORB_T];

  int ndoubly = 0;
  int nsingle = 0;
  for (int p = 0; p < norb; ++p) {
    int o = step_to_occ(steps[p]);
    occ[p] = o;
    alpha[p] = 0;
    beta[p] = 0;
    if (o == 2) ++ndoubly;
    else if (o == 1) ++nsingle;
  }

  int alpha_need = neleca - ndoubly;
  int beta_need = nelecb - ndoubly;
  if (alpha_need < 0 || beta_need < 0 || alpha_need + beta_need != nsingle) {
    hdiag_out[i] = 0.0;
    return;
  }

  int seen_single = 0;
  for (int p = 0; p < norb; ++p) {
    if (occ[p] == 2) {
      alpha[p] = 1;
      beta[p] = 1;
    } else if (occ[p] == 1) {
      if (seen_single < alpha_need) alpha[p] = 1;
      else beta[p] = 1;
      ++seen_single;
    }
  }

  double e1 = 0.0;
  for (int p = 0; p < norb; ++p) {
    double n_p = (double)(alpha[p] + beta[p]);
    e1 += h1_diag[p] * n_p;
  }

  double ecoul = 0.0;
  double exa = 0.0;
  double exb = 0.0;
  for (int p = 0; p < norb; ++p) {
    double n_p = (double)(alpha[p] + beta[p]);
    double a_p = (double)alpha[p];
    double b_p = (double)beta[p];
    for (int q = 0; q < norb; ++q) {
      double n_q = (double)(alpha[q] + beta[q]);
      double a_q = (double)alpha[q];
      double b_q = (double)beta[q];
      ecoul += 0.5 * n_p * eri_ppqq[p * norb + q] * n_q;
      exa += -0.5 * a_p * eri_pqqp[p * norb + q] * a_q;
      exb += -0.5 * b_p * eri_pqqp[p * norb + q] * b_q;
    }
  }

  hdiag_out[i] = e1 + ecoul + exa + exb;
}

// ---------------------------------------------------------------------------
// Compact score/PT2 kernel
// ---------------------------------------------------------------------------

__global__ void cas36_cipsi_score_pt2_compact_u64_kernel(
    const uint64_t* __restrict__ idx_u64,            // [ncand]
    const double* __restrict__ vals_root_major,      // [nroots*stride]
    int64_t vals_stride,
    int ncand,
    int nroots,
    const double* __restrict__ e_var,                // [nroots]
    const double* __restrict__ cand_hdiag,           // [ncand]
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    double denom_floor,
    uint64_t* __restrict__ score_bits_out,           // [ncand]
    double* __restrict__ pt2_out) {                  // [nroots]
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= ncand) return;

  uint64_t csf = idx_u64[i];
  if (selected_idx_sorted_u64 &&
      cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, csf)) {
    score_bits_out[i] = 0ull;
    return;
  }

  double h = cand_hdiag[i];
  double best = 0.0;

  for (int r = 0; r < nroots; ++r) {
    double p = vals_root_major[(int64_t)r * vals_stride + (int64_t)i];
    double denom = e_var[r] - h;
    if (denom_floor > 0.0) {
      double ad = fabs(denom);
      if (ad < denom_floor) denom = (denom >= 0.0) ? denom_floor : -denom_floor;
    }
    double s = 0.0;
    if (denom != 0.0) s = fabs(p / denom);
    if (s > best && isfinite(s)) best = s;
    if (denom != 0.0) {
      double contrib = (p * p) / denom;
      if (contrib != 0.0) atomicAdd(&pt2_out[r], contrib);
    }
  }

  score_bits_out[i] = (best > 0.0 && isfinite(best))
      ? (uint64_t)__double_as_longlong(best)
      : 0ull;
}

// ---------------------------------------------------------------------------
// Minimal launch helpers for the two compact kernels above
// ---------------------------------------------------------------------------

}  // namespace

extern "C" cudaError_t cas36_diag_guess_candidates_u64_dense_launch_stream(
    const int32_t* child,
    const int64_t* child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* cand_idx_u64,
    int ncand,
    const double* h1_diag,
    const double* eri_ppqq,
    const double* eri_pqqp,
    int neleca,
    int nelecb,
    double* hdiag_out,
    cudaStream_t stream,
    int threads) {
  if (!child || !child_prefix || !cand_idx_u64 || !h1_diag || !eri_ppqq || !eri_pqqp || !hdiag_out) return cudaErrorInvalidValue;
  if (ncand < 0 || norb <= 0 || norb > 64) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ncand == 0) return cudaSuccess;

  int blocks = (ncand + threads - 1) / threads;
  if (norb <= 16) {
    cas36_diag_guess_candidates_u64_dense_kernel<16><<<blocks, threads, 0, stream>>>(
        child, child_prefix, ncsf, norb, cand_idx_u64, ncand,
        h1_diag, eri_ppqq, eri_pqqp, neleca, nelecb, hdiag_out);
  } else if (norb <= 24) {
    cas36_diag_guess_candidates_u64_dense_kernel<24><<<blocks, threads, 0, stream>>>(
        child, child_prefix, ncsf, norb, cand_idx_u64, ncand,
        h1_diag, eri_ppqq, eri_pqqp, neleca, nelecb, hdiag_out);
  } else if (norb <= 32) {
    cas36_diag_guess_candidates_u64_dense_kernel<32><<<blocks, threads, 0, stream>>>(
        child, child_prefix, ncsf, norb, cand_idx_u64, ncand,
        h1_diag, eri_ppqq, eri_pqqp, neleca, nelecb, hdiag_out);
  } else if (norb <= 48) {
    cas36_diag_guess_candidates_u64_dense_kernel<48><<<blocks, threads, 0, stream>>>(
        child, child_prefix, ncsf, norb, cand_idx_u64, ncand,
        h1_diag, eri_ppqq, eri_pqqp, neleca, nelecb, hdiag_out);
  } else {
    cas36_diag_guess_candidates_u64_dense_kernel<64><<<blocks, threads, 0, stream>>>(
        child, child_prefix, ncsf, norb, cand_idx_u64, ncand,
        h1_diag, eri_ppqq, eri_pqqp, neleca, nelecb, hdiag_out);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t cas36_cipsi_score_pt2_compact_u64_launch_stream(
    const uint64_t* idx_u64,
    const double* vals_root_major,
    int64_t vals_stride,
    int ncand,
    int nroots,
    const double* e_var,
    const double* cand_hdiag,
    const uint64_t* selected_idx_sorted_u64,
    int nselected,
    double denom_floor,
    uint64_t* score_bits_out,
    double* pt2_out,
    cudaStream_t stream,
    int threads) {
  if (!idx_u64 || !vals_root_major || !e_var || !cand_hdiag || !score_bits_out || !pt2_out) return cudaErrorInvalidValue;
  if (ncand < 0 || nroots <= 0 || vals_stride < ncand) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ncand == 0) return cudaSuccess;

  int blocks = (ncand + threads - 1) / threads;
  cas36_cipsi_score_pt2_compact_u64_kernel<<<blocks, threads, 0, stream>>>(
      idx_u64, vals_root_major, vals_stride, ncand, nroots, e_var, cand_hdiag,
      selected_idx_sorted_u64, nselected, denom_floor, score_bits_out, pt2_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cas36_hb_screen_and_apply_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    const double* c_root,
    int nsel,
    int root,
    int norb,
    uint64_t ncsf,
    const int32_t* h1_pq,
    const double* h1_abs,
    const double* h1_signed,
    int n_h1,
    const int64_t* pq_ptr,
    const int32_t* rs_idx,
    const double* v_abs,
    const double* v_signed,
    const double* pq_max_v,
    double eps,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t* hash_keys,
    double* hash_vals,
    int cap,
    uint64_t label_lo,
    uint64_t label_hi,
    const uint64_t* selected_idx_sorted_u64,
    int nselected,
    int target_mode,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !c_root || !h1_pq || !h1_abs || !h1_signed || !pq_ptr || !rs_idx || !v_abs || !v_signed ||
      !pq_max_v || !child_table || !node_twos || !child_prefix || !hash_keys || !hash_vals || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel < 0 || root < 0 || norb <= 0 || norb > 64 || n_h1 < 0 || cap <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (target_mode < 0 || target_mode > 1) {
    return cudaErrorInvalidValue;
  }
  if ((cap & (cap - 1)) != 0) {
    return cudaErrorInvalidValue;
  }
  if (nsel == 0) {
    return cudaSuccess;
  }
  if (label_hi <= label_lo) {
    return cudaSuccess;
  }

  auto align_up = [](size_t x, size_t a) -> size_t {
    return (x + (a - 1u)) & ~(a - 1u);
  };
  const int nops = norb * norb;
  size_t smem_bytes = 0u;
  smem_bytes = align_up(smem_bytes, alignof(double));
  smem_bytes += (size_t)nops * sizeof(double);
  smem_bytes = align_up(smem_bytes, alignof(int8_t));
  smem_bytes += (size_t)norb * sizeof(int8_t);
  smem_bytes = align_up(smem_bytes, alignof(int32_t));
  smem_bytes += (size_t)(norb + 1) * sizeof(int32_t);
  smem_bytes = align_up(smem_bytes, alignof(int8_t));
  smem_bytes += (size_t)norb * sizeof(int8_t);
  smem_bytes = align_up(smem_bytes, alignof(int16_t));
  smem_bytes += (size_t)norb * sizeof(int16_t);
  smem_bytes = align_up(smem_bytes, alignof(uint64_t));
  smem_bytes += (size_t)(norb + 1) * sizeof(uint64_t);

  dim3 grid((unsigned)nsel);
  dim3 block((unsigned)threads);
  if (norb <= 16) {
    cas36_hb_screen_and_apply_u64_kernel<16><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64,
        c_root,
        nsel,
        root,
        norb,
        ncsf,
        h1_pq,
        h1_abs,
        h1_signed,
        n_h1,
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        eps,
        child_table,
        node_twos,
        child_prefix,
        hash_keys,
        hash_vals,
        cap,
        label_lo,
        label_hi,
        selected_idx_sorted_u64,
        nselected,
        target_mode,
        overflow_flag);
  } else if (norb <= 24) {
    cas36_hb_screen_and_apply_u64_kernel<24><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64,
        c_root,
        nsel,
        root,
        norb,
        ncsf,
        h1_pq,
        h1_abs,
        h1_signed,
        n_h1,
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        eps,
        child_table,
        node_twos,
        child_prefix,
        hash_keys,
        hash_vals,
        cap,
        label_lo,
        label_hi,
        selected_idx_sorted_u64,
        nselected,
        target_mode,
        overflow_flag);
  } else if (norb <= 32) {
    cas36_hb_screen_and_apply_u64_kernel<32><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64,
        c_root,
        nsel,
        root,
        norb,
        ncsf,
        h1_pq,
        h1_abs,
        h1_signed,
        n_h1,
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        eps,
        child_table,
        node_twos,
        child_prefix,
        hash_keys,
        hash_vals,
        cap,
        label_lo,
        label_hi,
        selected_idx_sorted_u64,
        nselected,
        target_mode,
        overflow_flag);
  } else if (norb <= 48) {
    cas36_hb_screen_and_apply_u64_kernel<48><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64,
        c_root,
        nsel,
        root,
        norb,
        ncsf,
        h1_pq,
        h1_abs,
        h1_signed,
        n_h1,
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        eps,
        child_table,
        node_twos,
        child_prefix,
        hash_keys,
        hash_vals,
        cap,
        label_lo,
        label_hi,
        selected_idx_sorted_u64,
        nselected,
        target_mode,
        overflow_flag);
  } else {
    cas36_hb_screen_and_apply_u64_kernel<64><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64,
        c_root,
        nsel,
        root,
        norb,
        ncsf,
        h1_pq,
        h1_abs,
        h1_signed,
        n_h1,
        pq_ptr,
        rs_idx,
        v_abs,
        v_signed,
        pq_max_v,
        eps,
        child_table,
        node_twos,
        child_prefix,
        hash_keys,
        hash_vals,
        cap,
        label_lo,
        label_hi,
        selected_idx_sorted_u64,
        nselected,
        target_mode,
        overflow_flag);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t cas36_hb_emit_tuples_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    const double* c_bound,
    int nsel,
    int norb,
    uint64_t ncsf,
    const int32_t* h1_pq,
    const double* h1_abs,
    const double* h1_signed,
    int n_h1,
    const int64_t* pq_ptr,
    const int32_t* rs_idx,
    const double* v_abs,
    const double* v_signed,
    const double* pq_max_v,
    double eps,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t* out_keys,
    int* out_src,
    double* out_hij,
    int cap,
    uint64_t label_lo,
    uint64_t label_hi,
    const uint64_t* selected_idx_sorted_u64,
    int nselected,
    int target_mode,
    int* out_n,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !c_bound || !h1_pq || !h1_abs || !h1_signed || !pq_ptr || !rs_idx || !v_abs || !v_signed ||
      !pq_max_v || !child_table || !node_twos || !child_prefix || !out_keys || !out_src || !out_hij || !out_n ||
      !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel < 0 || norb <= 0 || norb > 64 || n_h1 < 0 || cap <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (target_mode < 0 || target_mode > 1) {
    return cudaErrorInvalidValue;
  }
  if (nsel == 0) {
    return cudaSuccess;
  }
  if (label_hi <= label_lo) {
    return cudaSuccess;
  }

  auto align_up = [](size_t x, size_t a) -> size_t {
    return (x + (a - 1u)) & ~(a - 1u);
  };
  const int nops = norb * norb;
  size_t smem_bytes = 0u;
  smem_bytes = align_up(smem_bytes, alignof(double));
  smem_bytes += (size_t)nops * sizeof(double);
  smem_bytes = align_up(smem_bytes, alignof(int8_t));
  smem_bytes += (size_t)norb * sizeof(int8_t);
  smem_bytes = align_up(smem_bytes, alignof(int32_t));
  smem_bytes += (size_t)(norb + 1) * sizeof(int32_t);
  smem_bytes = align_up(smem_bytes, alignof(int8_t));
  smem_bytes += (size_t)norb * sizeof(int8_t);
  smem_bytes = align_up(smem_bytes, alignof(int16_t));
  smem_bytes += (size_t)norb * sizeof(int16_t);
  smem_bytes = align_up(smem_bytes, alignof(uint64_t));
  smem_bytes += (size_t)(norb + 1) * sizeof(uint64_t);

  dim3 grid((unsigned)nsel);
  dim3 block((unsigned)threads);
  if (norb <= 16) {
    cas36_hb_emit_tuples_u64_kernel<16><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64, c_bound, nsel, norb, ncsf, h1_pq, h1_abs, h1_signed, n_h1, pq_ptr, rs_idx, v_abs, v_signed,
        pq_max_v, eps, child_table, node_twos, child_prefix, out_keys, out_src, out_hij, cap, label_lo, label_hi,
        selected_idx_sorted_u64, nselected, target_mode, out_n, overflow_flag);
  } else if (norb <= 24) {
    cas36_hb_emit_tuples_u64_kernel<24><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64, c_bound, nsel, norb, ncsf, h1_pq, h1_abs, h1_signed, n_h1, pq_ptr, rs_idx, v_abs, v_signed,
        pq_max_v, eps, child_table, node_twos, child_prefix, out_keys, out_src, out_hij, cap, label_lo, label_hi,
        selected_idx_sorted_u64, nselected, target_mode, out_n, overflow_flag);
  } else if (norb <= 32) {
    cas36_hb_emit_tuples_u64_kernel<32><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64, c_bound, nsel, norb, ncsf, h1_pq, h1_abs, h1_signed, n_h1, pq_ptr, rs_idx, v_abs, v_signed,
        pq_max_v, eps, child_table, node_twos, child_prefix, out_keys, out_src, out_hij, cap, label_lo, label_hi,
        selected_idx_sorted_u64, nselected, target_mode, out_n, overflow_flag);
  } else if (norb <= 48) {
    cas36_hb_emit_tuples_u64_kernel<48><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64, c_bound, nsel, norb, ncsf, h1_pq, h1_abs, h1_signed, n_h1, pq_ptr, rs_idx, v_abs, v_signed,
        pq_max_v, eps, child_table, node_twos, child_prefix, out_keys, out_src, out_hij, cap, label_lo, label_hi,
        selected_idx_sorted_u64, nselected, target_mode, out_n, overflow_flag);
  } else {
    cas36_hb_emit_tuples_u64_kernel<64><<<grid, block, smem_bytes, stream>>>(
        sel_idx_u64, c_bound, nsel, norb, ncsf, h1_pq, h1_abs, h1_signed, n_h1, pq_ptr, rs_idx, v_abs, v_signed,
        pq_max_v, eps, child_table, node_twos, child_prefix, out_keys, out_src, out_hij, cap, label_lo, label_hi,
        selected_idx_sorted_u64, nselected, target_mode, out_n, overflow_flag);
  }
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Exact selected-hop tuple emission (off-diagonal part only)
//
// This launch surface is reserved for the exact projected selected-space hop
//
//   Y = P_S H P_S X
//
// with diagonal handled separately by the caller as
//
//   Y[i,:] += H_ii * X[i,:]
//
// and this kernel family responsible only for emitting exact off-diagonal
// selected-to-selected tuples
//
//   (i_local, j_local, H_ij),  i != j,  i in S, j in S.
//
// For now the dedicated surface delegates to the existing u64 tuple emitter
// with eps=0 and selected-only target filtering so the solver path can evolve
// independently of the HB-SCI naming. The runtime path should not treat this
// as solver-exact until parity checks pass.
// ---------------------------------------------------------------------------

extern "C" cudaError_t cas36_exact_selected_emit_tuples_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    const double* c_bound,
    int nsel,
    int norb,
    uint64_t ncsf,
    const int32_t* h1_pq,
    const double* h1_abs,
    const double* h1_signed,
    int n_h1,
    const int64_t* pq_ptr,
    const int32_t* rs_idx,
    const double* v_abs,
    const double* v_signed,
    const double* pq_max_v,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t* out_keys,
    int* out_src,
    double* out_hij,
    int cap,
    const uint64_t* selected_idx_sorted_u64,
    int nselected,
    int* out_n,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !c_bound || !h1_pq || !h1_abs || !h1_signed || !pq_ptr || !rs_idx || !v_abs || !v_signed ||
      !pq_max_v || !child_table || !node_twos || !child_prefix || !out_keys || !out_src || !out_hij || !out_n ||
      !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel < 0 || norb <= 0 || norb > 64 || n_h1 < 0 || cap <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (nsel == 0) {
    return cudaSuccess;
  }

  // No screening for the exact selected projected hop; diagonal is handled
  // separately by the caller.
  return cas36_hb_emit_tuples_u64_launch_stream(
      sel_idx_u64,
      c_bound,
      nsel,
      norb,
      ncsf,
      h1_pq,
      h1_abs,
      h1_signed,
      n_h1,
      pq_ptr,
      rs_idx,
      v_abs,
      v_signed,
      pq_max_v,
      0.0,
      child_table,
      node_twos,
      child_prefix,
      out_keys,
      out_src,
      out_hij,
      cap,
      0ull,
      ncsf,
      selected_idx_sorted_u64,
      nselected,
      1,
      out_n,
      overflow_flag,
      stream,
      threads);
}

template <int MAX_NORB_T>
__global__ __launch_bounds__(128, 2)
void cas36_exact_selected_emit_tuples_dense_u64_kernel(
    const uint64_t* __restrict__ sel_idx_u64,
    const double* __restrict__ c_bound,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* __restrict__ h_base,
    const double* __restrict__ eri4,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_src,
    double* __restrict__ out_hij,
    int cap,
    const uint64_t* __restrict__ selected_idx_sorted_u64,
    int nselected,
    double* __restrict__ out_diag,
    int* __restrict__ out_n,
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
  if (c_bound && c_bound[j_local] == 0.0) return;

  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int nops = norb * norb;

  __shared__ int8_t steps_j_s[MAX_NORB_T];
  __shared__ int32_t nodes_j_s[MAX_NORB_T + 1];
  __shared__ int8_t occ_j_s[MAX_NORB_T];
  __shared__ int16_t b_j_s[MAX_NORB_T];
  __shared__ uint64_t idx_prefix_j_s[MAX_NORB_T + 1];

  bool ok = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
      child_table, child_prefix, norb, ncsf, j_global, steps_j_s, nodes_j_s);
  if (!ok) {
    if (tid == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int k = tid; k < norb; k += nthreads) {
    int8_t st = steps_j_s[k];
    occ_j_s[k] = (int8_t)step_to_occ(st);
    b_j_s[k] = node_twos[nodes_j_s[k + 1]];
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
  __syncthreads();

  if (tid == 0 && out_diag) out_diag[j_local] = 0.0;
  __syncthreads();
  double diag_local = 0.0;

  // Exact one-body off-diagonal from h_eff(j).
  for (int pq = tid; pq < nops; pq += nthreads) {
    int p = pq / norb;
    int q = pq - p * norb;
    double hpq = h_base[pq];
    for (int r = 0; r < norb; ++r) {
      int occ_r = (int)occ_j_s[r];
      if (occ_r == 0) continue;
      hpq += 0.5 * cas36_dense_eri4_at(eri4, norb, p, q, r, r) * (double)occ_r;
    }
    if (hpq == 0.0) continue;
    if (p == q) {
      diag_local += hpq * (double)occ_j_s[p];
      continue;
    }
    cas36_exact_emit_weighted_epq_selected_u64<MAX_NORB_T>(
        j_global,
        j_local,
        j_global,
        norb,
        ncsf,
        steps_j_s,
        nodes_j_s,
        occ_j_s,
        b_j_s,
        idx_prefix_j_s,
        p,
        q,
        hpq,
        child_table,
        node_twos,
        child_prefix,
        selected_idx_sorted_u64,
        nselected,
        out_keys,
        out_src,
        out_hij,
        cap,
        out_n,
        overflow_flag);
  }

  // Exact two-body product terms with r != s via k = E_rs |j>.
  for (int rs = tid; rs < nops; rs += nthreads) {
    int r = rs / norb;
    int s = rs - r * norb;
    if (r == s) continue;
    if ((int)occ_j_s[s] <= 0 || (int)occ_j_s[r] >= 2) continue;

    int start, end, q_start, q_mid, q_end;
    if (r < s) {
      start = r;
      end = s;
      q_start = Q_uR;
      q_mid = Q_R;
      q_end = Q_oR;
    } else {
      start = s;
      end = r;
      q_start = Q_uL;
      q_mid = Q_L;
      q_end = Q_oL;
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

      int dp0 = 0;
      int dp1 = 0;
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
        uint64_t idx_prefix_k[MAX_NORB_T + 1];
        bool ok_k = cas36_sci_reconstruct_path_from_index_u64<MAX_NORB_T>(
            child_table, child_prefix, norb, ncsf, k_global, steps_k, nodes_k);
        if (!ok_k) {
          if (overflow_flag) atomicExch(overflow_flag, 1);
          continue;
        }
        idx_prefix_k[0] = 0ull;
        for (int kk = 0; kk < norb; ++kk) {
          occ_k[kk] = (int8_t)step_to_occ(steps_k[kk]);
          b_k[kk] = node_twos[nodes_k[kk + 1]];
          idx_prefix_k[kk + 1] = idx_prefix_k[kk] + (uint64_t)child_prefix[nodes_k[kk] * 5 + (int)steps_k[kk]];
        }

        double diag_contrib = 0.0;
        for (int p = 0; p < norb; ++p) {
          int occ_p = (int)occ_k[p];
          if (occ_p == 0) continue;
          diag_contrib += 0.5 * c_rs * cas36_dense_eri4_at(eri4, norb, p, p, r, s) * (double)occ_p;
        }
        if (diag_contrib != 0.0 && k_global != j_global && selected_idx_sorted_u64 &&
            cas36_sci_contains_sorted_u64(selected_idx_sorted_u64, nselected, k_global)) {
          cas36_exact_emit_tuple_u64(
              out_keys, out_src, out_hij, cap, k_global, j_local, diag_contrib, out_n, overflow_flag);
        }

        for (int pq2 = 0; pq2 < nops; ++pq2) {
          int p = pq2 / norb;
          int q = pq2 - p * norb;
          if (p == q) continue;
          double gpq = 0.5 * c_rs * cas36_dense_eri4_at(eri4, norb, p, q, r, s);
          if (gpq == 0.0) continue;
          diag_local += cas36_exact_accumulate_weighted_epq_self_u64<MAX_NORB_T>(
              j_global,
              k_global,
              norb,
              ncsf,
              steps_k,
              nodes_k,
              occ_k,
              b_k,
              idx_prefix_k,
              p,
              q,
              gpq,
              child_table,
              node_twos,
              child_prefix);
          cas36_exact_emit_weighted_epq_selected_u64<MAX_NORB_T>(
              j_global,
              j_local,
              k_global,
              norb,
              ncsf,
              steps_k,
              nodes_k,
              occ_k,
              b_k,
              idx_prefix_k,
              p,
              q,
              gpq,
              child_table,
              node_twos,
              child_prefix,
              selected_idx_sorted_u64,
              nselected,
              out_keys,
              out_src,
              out_hij,
              cap,
              out_n,
              overflow_flag);
        }
      }
    }
  }
  if (out_diag && diag_local != 0.0) atomicAdd(&out_diag[j_local], diag_local);
}

extern "C" cudaError_t cas36_exact_selected_emit_tuples_dense_u64_launch_stream(
    const uint64_t* sel_idx_u64,
    const double* c_bound,
    int nsel,
    int norb,
    uint64_t ncsf,
    const double* h_base,
    const double* eri4,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t* out_keys,
    int* out_src,
    double* out_hij,
    int cap,
    const uint64_t* selected_idx_sorted_u64,
    int nselected,
    double* out_diag,
    int* out_n,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!sel_idx_u64 || !c_bound || !h_base || !eri4 || !child_table || !node_twos || !child_prefix || !out_keys ||
      !out_src || !out_hij || !selected_idx_sorted_u64 || !out_n || !overflow_flag) {
    return cudaErrorInvalidValue;
  }
  if (nsel < 0 || norb <= 0 || norb > 32 || cap <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (nsel == 0) {
    return cudaSuccess;
  }

  dim3 block((unsigned int)threads);
  dim3 grid((unsigned int)nsel);
  if (norb <= 8) {
    cas36_exact_selected_emit_tuples_dense_u64_kernel<8><<<grid, block, 0, stream>>>(
        sel_idx_u64,
        c_bound,
        nsel,
        norb,
        ncsf,
        h_base,
        eri4,
        child_table,
        node_twos,
        child_prefix,
        out_keys,
        out_src,
        out_hij,
        cap,
        selected_idx_sorted_u64,
        nselected,
        out_diag,
        out_n,
        overflow_flag);
  } else if (norb <= 16) {
    cas36_exact_selected_emit_tuples_dense_u64_kernel<16><<<grid, block, 0, stream>>>(
        sel_idx_u64,
        c_bound,
        nsel,
        norb,
        ncsf,
        h_base,
        eri4,
        child_table,
        node_twos,
        child_prefix,
        out_keys,
        out_src,
        out_hij,
        cap,
        selected_idx_sorted_u64,
        nselected,
        out_diag,
        out_n,
        overflow_flag);
  } else if (norb <= 24) {
    cas36_exact_selected_emit_tuples_dense_u64_kernel<24><<<grid, block, 0, stream>>>(
        sel_idx_u64,
        c_bound,
        nsel,
        norb,
        ncsf,
        h_base,
        eri4,
        child_table,
        node_twos,
        child_prefix,
        out_keys,
        out_src,
        out_hij,
        cap,
        selected_idx_sorted_u64,
        nselected,
        out_diag,
        out_n,
        overflow_flag);
  } else {
    cas36_exact_selected_emit_tuples_dense_u64_kernel<32><<<grid, block, 0, stream>>>(
        sel_idx_u64,
        c_bound,
        nsel,
        norb,
        ncsf,
        h_base,
        eri4,
        child_table,
        node_twos,
        child_prefix,
        out_keys,
        out_src,
        out_hij,
        cap,
        selected_idx_sorted_u64,
        nselected,
        out_diag,
        out_n,
        overflow_flag);
  }
  return cudaGetLastError();
}
