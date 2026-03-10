
#pragma once

// CAS(36,36) / idx64 QMC spawn kernels for ASUKA.
//
// Device label type:
//   uint64_t  -- semantic value is a nonnegative global CSF index in [0, ncsf).
//
// This path exists to remove the current Key64 / norb<=32 limitation without
// introducing a new workspace type. It is designed to plug into the existing
// QmcWorkspaceU64 + coalesce_coo_u64_f64 + phi_pivot_resample_u64_f64 flow.
//
// Integration notes:
//   - keep using QmcWorkspaceU64 unchanged
//   - bind new launchers parallel to the current Key64 wrappers
//   - Python/C++ can continue to name the device arrays x_key/key_u/etc. to
//     keep the diff small; in this backend they simply store global CSF idx64
//     values rather than packed paths.
//
// Dependency:
//   This header expects the usual EPQ helpers/constants from the existing CUDA
//   tree, in particular:
//     * step_to_occ
//     * candidate_dprimes
//     * segment_value_int
//     * Q_uR / Q_R / Q_oR / Q_uL / Q_L / Q_oL
//
// Include it from the same aggregation TU that already includes the existing
// EPQ support header (or include that header before this one).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ uint64_t cas36_splitmix64_next(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ double cas36_rand_u01(uint64_t* state) {
  return (double)(cas36_splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ uint32_t cas36_rand_below_u32(uint64_t* state, uint32_t n) {
  if (n == 0) return 0;
  uint64_t nn = (uint64_t)n;
  uint64_t threshold = (uint64_t)(-nn) % nn;
  while (true) {
    uint64_t x = cas36_splitmix64_next(state);
    if (x >= threshold) return (uint32_t)(x % nn);
  }
}

__device__ __forceinline__ int cas36_alias_sample_i32(
    const float* __restrict__ prob,
    const int32_t* __restrict__ alias,
    int n,
    uint64_t* state) {
  if (n <= 0) return 0;
  int i = (int)cas36_rand_below_u32(state, (uint32_t)n);
  float u = (float)cas36_rand_u01(state);
  return (u < prob[i]) ? i : (int)alias[i];
}

__device__ __forceinline__ bool cas36_contains_sorted_u64(
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

template <int MAX_NORB_T>
__device__ __forceinline__ bool cas36_reconstruct_path_from_index_u64(
    const int32_t* __restrict__ child,        // [nnodes,4] flattened
    const int64_t* __restrict__ child_prefix, // [nnodes,5] flattened
    int norb,
    uint64_t ncsf,
    uint64_t csf_idx,
    int8_t* __restrict__ steps_out,           // [norb]
    int32_t* __restrict__ nodes_out) {        // [norb+1]
  if (norb > MAX_NORB_T) return false;
  if (csf_idx >= ncsf) return false;

  int32_t node = 0;  // ASUKA DRT root convention
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
__device__ __forceinline__ bool cas36_epq_sample_one_reservoir_idx64_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int norb,
    uint64_t ncsf,
    uint64_t csf_idx,
    const int8_t* __restrict__ steps,
    const int32_t* __restrict__ nodes,
    int p,
    int q,
    uint64_t* rng_state,
    uint64_t* out_child_idx,
    double* out_coeff,
    double* out_inv_p) {
  if (norb > MAX_NORB_T) return false;
  if (p == q) {
    int occ = step_to_occ(steps[p]);
    if (occ <= 0) return false;
    *out_child_idx = csf_idx;
    *out_coeff = (double)occ;
    *out_inv_p = 1.0;
    return true;
  }

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return false;

  int start;
  int end;
  int q_start;
  int q_mid;
  int q_end;
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

  int32_t node_start = nodes[start];
  int32_t node_end_target = nodes[end + 1];

  uint64_t idx_prefix[MAX_NORB_T + 1];
  idx_prefix[0] = 0;
  for (int kk = 0; kk < norb; ++kk) {
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx_prefix[kk + 1] = idx_prefix[kk] + (uint64_t)child_prefix[node_kk * 5 + step_kk];
  }

  uint64_t prefix_offset = idx_prefix[start];
  uint64_t prefix_endplus1 = idx_prefix[end + 1];
  if (csf_idx < prefix_endplus1) return false;  // sanity
  uint64_t suffix_offset = csf_idx - prefix_endplus1;

  int d_ref[MAX_NORB_T];
  int b_ref[MAX_NORB_T];
  for (int kk = 0; kk < norb; ++kk) {
    d_ref[kk] = (int)steps[kk];
    b_ref[kk] = (int)node_twos[nodes[kk + 1]];
  }

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  uint64_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  ++top;

  double sum_abs = 0.0;
  uint64_t selected_child = 0ull;
  double selected_coeff = 0.0;
  double selected_abs = 0.0;
  bool have_selected = false;

  while (top) {
    --top;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    uint64_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = d_ref[kpos];
    int bk = b_ref[kpos];
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
      double seg = segment_value_int(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      double w2 = w * seg;
      uint64_t seg_idx2 = seg_idx + (uint64_t)child_prefix[node_k * 5 + dprime];

      if (is_last) {
        if (child_k != node_end_target) continue;
        uint64_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
        if (csf_i >= ncsf) continue;
        if (csf_i == csf_idx) continue;

        double wabs = fabs(w2);
        if (wabs == 0.0) continue;
        sum_abs += wabs;
        if (cas36_rand_u01(rng_state) * sum_abs < wabs) {
          selected_child = csf_i;
          selected_coeff = w2;
          selected_abs = wabs;
          have_selected = true;
        }
      } else {
        if (top >= MAX_NORB_T) return false;
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        ++top;
      }
    }
  }

  if (!have_selected || sum_abs == 0.0 || selected_abs == 0.0) return false;
  *out_child_idx = selected_child;
  *out_coeff = selected_coeff;
  *out_inv_p = sum_abs / selected_abs;
  return true;
}

template <int MAX_NORB_T>
__global__ void cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* __restrict__ x_idx_u64,  // [m], sorted unique if initiator_t > 0
    const double* __restrict__ x_val,        // [m]
    int m,
    const double* __restrict__ h_base_flat,  // [norb*norb]
    const double* __restrict__ eri_mat,      // [nops*nops]
    const float* __restrict__ pair_alias_prob,
    const int32_t* __restrict__ pair_alias_idx,
    const double* __restrict__ pair_norm,
    double pair_norm_sum,
    int pair_sampling_mode,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    const double* __restrict__ initiator_t_dev,
    uint64_t* __restrict__ out_idx_u64,      // [m*(nspawn_one+nspawn_two)]
    double* __restrict__ out_val) {          // [m*(nspawn_one+nspawn_two)]
  int parent_pos = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (parent_pos >= m) return;
  if (norb > MAX_NORB_T) return;

  uint64_t parent_idx = x_idx_u64[parent_pos];
  double xj = x_val[parent_pos];
  if (xj == 0.0) return;

  int8_t steps_j[MAX_NORB_T];
  int32_t nodes_j[MAX_NORB_T + 1];
  if (!cas36_reconstruct_path_from_index_u64<MAX_NORB_T>(
          child, child_prefix, norb, ncsf, parent_idx, steps_j, nodes_j)) {
    return;
  }

  int src[MAX_NORB_T];
  int dst[MAX_NORB_T];
  int ns = 0;
  int nd = 0;
  for (int orb = 0; orb < norb; ++orb) {
    int8_t stp = steps_j[orb];
    if (stp != 0) src[ns++] = orb;
    if (stp != 3) dst[nd++] = orb;
  }
  if (ns == 0) return;

  int rr_id_src[MAX_NORB_T];
  double occ_src[MAX_NORB_T];
  for (int t = 0; t < ns; ++t) {
    int r = src[t];
    rr_id_src[t] = r * norb + r;
    occ_src[t] = (steps_j[r] == 3) ? 2.0 : 1.0;
  }

  if (initiator_t_dev) initiator_t = *initiator_t_dev;
  bool allow_new = true;
  if (initiator_t > 0.0 && fabs(xj) < initiator_t) allow_new = false;

  uint32_t idx_mix = (uint32_t)(parent_idx ^ (parent_idx >> 32));
  uint64_t st = seed
              ^ ((uint64_t)idx_mix * 0xD1B54A32D192ED03ull)
              ^ ((uint64_t)(uint32_t)parent_pos * 0x94D049BB133111EBull);

  int nspawn_total = nspawn_one + nspawn_two;
  int64_t base = (int64_t)parent_pos * (int64_t)nspawn_total;
  int nops = norb * norb;

  if (nspawn_one > 0) {
    double scale_one = -eps / (double)nspawn_one;
    double inv_p_pair_one = (double)(ns * norb);

    for (int s = 0; s < nspawn_one; ++s) {
      int q_orb = src[(int)cas36_rand_below_u32(&st, (uint32_t)ns)];
      int p_orb = (int)cas36_rand_below_u32(&st, (uint32_t)norb);
      int pq_id = p_orb * norb + q_orb;

      double w_eff = h_base_flat[pq_id];
      double sum_rr = 0.0;
      for (int t = 0; t < ns; ++t) {
        sum_rr += eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rr_id_src[t]] * occ_src[t];
      }
      w_eff += 0.5 * sum_rr;
      if (w_eff == 0.0) continue;

      uint64_t child_idx = 0ull;
      double coeff_out = 0.0;
      double inv_p_out = 0.0;
      bool ok = cas36_epq_sample_one_reservoir_idx64_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, ncsf, parent_idx, steps_j, nodes_j,
          p_orb, q_orb, &st, &child_idx, &coeff_out, &inv_p_out);
      if (!ok) continue;

      if (!allow_new && !cas36_contains_sorted_u64(x_idx_u64, m, child_idx)) continue;

      out_idx_u64[base + s] = child_idx;
      out_val[base + s] = scale_one * xj * w_eff * coeff_out * inv_p_out * inv_p_pair_one;
    }
  }

  if (nspawn_two > 0) {
    if (norb <= 1) return;
    double scale_two = -eps / (double)nspawn_two;

    bool use_pair_norm =
        (pair_sampling_mode == 1) &&
        (pair_alias_prob != nullptr) &&
        (pair_alias_idx != nullptr) &&
        (pair_norm != nullptr) &&
        (pair_norm_sum > 0.0);

    for (int s = 0; s < nspawn_two; ++s) {
      int r_orb = -1;
      int s_orb = -1;
      int rs_id = -1;
      double inv_p_pair_rs = 0.0;

      if (use_pair_norm) {
        rs_id = cas36_alias_sample_i32(pair_alias_prob, pair_alias_idx, nops, &st);
        r_orb = rs_id / norb;
        s_orb = rs_id - r_orb * norb;
        if (r_orb == s_orb) continue;
        if (steps_j[s_orb] == 0) continue;
        if (steps_j[r_orb] == 3) continue;
        double pn_rs = pair_norm[rs_id];
        if (!(pn_rs > 0.0)) continue;
        inv_p_pair_rs = pair_norm_sum / pn_rs;
      } else {
        if (nd == 0) return;
        s_orb = src[(int)cas36_rand_below_u32(&st, (uint32_t)ns)];
        int occ_s = (steps_j[s_orb] == 3) ? 2 : 1;
        if (occ_s == 1) {
          if (nd <= 1) continue;
          while (true) {
            r_orb = dst[(int)cas36_rand_below_u32(&st, (uint32_t)nd)];
            if (r_orb != s_orb) break;
          }
          inv_p_pair_rs = (double)(ns * (nd - 1));
        } else {
          r_orb = dst[(int)cas36_rand_below_u32(&st, (uint32_t)nd)];
          inv_p_pair_rs = (double)(ns * nd);
        }
        rs_id = r_orb * norb + s_orb;
      }

      uint64_t k_idx = 0ull;
      double coeff_rs = 0.0;
      double inv_p_rs_epq = 0.0;
      bool ok_rs = cas36_epq_sample_one_reservoir_idx64_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, ncsf, parent_idx, steps_j, nodes_j,
          r_orb, s_orb, &st, &k_idx, &coeff_rs, &inv_p_rs_epq);
      if (!ok_rs) continue;

      int8_t steps_k[MAX_NORB_T];
      int32_t nodes_k[MAX_NORB_T + 1];
      if (!cas36_reconstruct_path_from_index_u64<MAX_NORB_T>(
              child, child_prefix, norb, ncsf, k_idx, steps_k, nodes_k)) {
        continue;
      }

      int p_orb = -1;
      int q_orb = -1;
      int pq_id = -1;
      double inv_p_pair_pq = 0.0;
      if (use_pair_norm) {
        pq_id = cas36_alias_sample_i32(pair_alias_prob, pair_alias_idx, nops, &st);
        p_orb = pq_id / norb;
        q_orb = pq_id - p_orb * norb;
        double pn_pq = pair_norm[pq_id];
        if (!(pn_pq > 0.0)) continue;
        inv_p_pair_pq = pair_norm_sum / pn_pq;
      } else {
        int src_k[MAX_NORB_T];
        int ns_k = 0;
        for (int orb = 0; orb < norb; ++orb) {
          if (steps_k[orb] != 0) src_k[ns_k++] = orb;
        }
        if (ns_k == 0) continue;
        q_orb = src_k[(int)cas36_rand_below_u32(&st, (uint32_t)ns_k)];
        p_orb = (int)cas36_rand_below_u32(&st, (uint32_t)norb);
        pq_id = p_orb * norb + q_orb;
        inv_p_pair_pq = (double)(ns_k * norb);
      }

      double v_pqrs = 0.5 * eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rs_id];
      if (v_pqrs == 0.0) continue;

      uint64_t i_idx = 0ull;
      double coeff_pq = 0.0;
      double inv_p_pq_epq = 0.0;
      bool ok_pq = cas36_epq_sample_one_reservoir_idx64_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, ncsf, k_idx, steps_k, nodes_k,
          p_orb, q_orb, &st, &i_idx, &coeff_pq, &inv_p_pq_epq);
      if (!ok_pq) continue;

      if (!allow_new && !cas36_contains_sorted_u64(x_idx_u64, m, i_idx)) continue;

      out_idx_u64[base + (int64_t)nspawn_one + s] = i_idx;
      out_val[base + (int64_t)nspawn_one + s] =
          scale_two * xj * v_pqrs
        * coeff_rs * inv_p_rs_epq * inv_p_pair_rs
        * coeff_pq * inv_p_pq_epq * inv_p_pair_pq;
    }
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_idx64_u64_f64_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* x_idx_u64,
    const double* x_val,
    int m,
    const double* h_base_flat,
    const double* eri_mat,
    const float* pair_alias_prob,
    const int32_t* pair_alias_idx,
    const double* pair_norm,
    double pair_norm_sum,
    int pair_sampling_mode,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    uint64_t* out_idx_u64,
    double* out_val,
    cudaStream_t stream,
    int threads) {
  if (!child || !node_twos || !child_prefix) return cudaErrorInvalidValue;
  if (!x_idx_u64 || !x_val || !h_base_flat || !eri_mat || !out_idx_u64 || !out_val) return cudaErrorInvalidValue;
  if (ncsf == 0ull || norb <= 0 || norb > 64) return cudaErrorInvalidValue;
  if (m < 0) return cudaErrorInvalidValue;
  if (nspawn_one < 0 || nspawn_two < 0) return cudaErrorInvalidValue;
  if (nspawn_one == 0 && nspawn_two == 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  int blocks = (m + threads - 1) / threads;
  if (blocks == 0) return cudaSuccess;

  if (norb <= 16) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<16><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, initiator_t, nullptr,
        out_idx_u64, out_val);
  } else if (norb <= 24) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<24><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, initiator_t, nullptr,
        out_idx_u64, out_val);
  } else if (norb <= 32) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, initiator_t, nullptr,
        out_idx_u64, out_val);
  } else if (norb <= 48) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<48><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, initiator_t, nullptr,
        out_idx_u64, out_val);
  } else {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, initiator_t, nullptr,
        out_idx_u64, out_val);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_idx64_u64_f64_initiator_dev_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    uint64_t ncsf,
    int norb,
    const uint64_t* x_idx_u64,
    const double* x_val,
    int m,
    const double* h_base_flat,
    const double* eri_mat,
    const float* pair_alias_prob,
    const int32_t* pair_alias_idx,
    const double* pair_norm,
    double pair_norm_sum,
    int pair_sampling_mode,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    const double* initiator_t_dev,
    uint64_t* out_idx_u64,
    double* out_val,
    cudaStream_t stream,
    int threads) {
  if (!initiator_t_dev) return cudaErrorInvalidValue;
  if (!child || !node_twos || !child_prefix) return cudaErrorInvalidValue;
  if (!x_idx_u64 || !x_val || !h_base_flat || !eri_mat || !out_idx_u64 || !out_val) return cudaErrorInvalidValue;
  if (ncsf == 0ull || norb <= 0 || norb > 64) return cudaErrorInvalidValue;
  if (m < 0) return cudaErrorInvalidValue;
  if (nspawn_one < 0 || nspawn_two < 0) return cudaErrorInvalidValue;
  if (nspawn_one == 0 && nspawn_two == 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  int blocks = (m + threads - 1) / threads;
  if (blocks == 0) return cudaSuccess;

  if (norb <= 16) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<16><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, 0.0, initiator_t_dev,
        out_idx_u64, out_val);
  } else if (norb <= 24) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<24><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, 0.0, initiator_t_dev,
        out_idx_u64, out_val);
  } else if (norb <= 32) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, 0.0, initiator_t_dev,
        out_idx_u64, out_val);
  } else if (norb <= 48) {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<48><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, 0.0, initiator_t_dev,
        out_idx_u64, out_val);
  } else {
    cas36_qmc_spawn_hamiltonian_idx64_u64_f64_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, x_idx_u64, x_val, m,
        h_base_flat, eri_mat,
        pair_alias_prob, pair_alias_idx, pair_norm, pair_norm_sum, pair_sampling_mode,
        eps, nspawn_one, nspawn_two, seed, 0.0, initiator_t_dev,
        out_idx_u64, out_val);
  }
  return cudaGetLastError();
}
