#pragma once

// -----------------------------------------------------------------------------
// QMC: one-body stochastic spawn (events-only)
// -----------------------------------------------------------------------------

namespace {

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ double rand_u01(uint64_t* state) {
  // Use the top 53 bits to create a uniform double in [0,1).
  return (double)(splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ uint32_t rand_below_u32(uint64_t* state, uint32_t n) {
  if (n == 0) return 0;
  uint64_t nn = (uint64_t)n;
  uint64_t threshold = (uint64_t)(-nn) % nn;
  while (true) {
    uint64_t x = splitmix64_next(state);
    if (x >= threshold) return (uint32_t)(x % nn);
  }
}

__device__ __forceinline__ int alias_sample_i32(const float* __restrict__ prob, const int32_t* __restrict__ alias, int n, uint64_t* state) {
  if (n <= 0) return 0;
  int i = (int)rand_below_u32(state, (uint32_t)n);
  float u = (float)rand_u01(state);
  return (u < prob[i]) ? i : (int)alias[i];
}

__device__ __forceinline__ bool contains_sorted_i32(const int32_t* __restrict__ a, int n, int32_t key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    int32_t v = a[mid];
    if (v < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < n) && (a[lo] == key);
}

__device__ __forceinline__ bool contains_sorted_u64(const uint64_t* __restrict__ a, int n, uint64_t key) {
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
__device__ __forceinline__ bool key64_to_steps_nodes(
    const int32_t* __restrict__ child,  // [nnodes,4] flattened
    int norb,
    uint64_t key,
    int8_t* __restrict__ steps,    // [norb]
    int32_t* __restrict__ nodes) { // [norb+1]
  if (norb > MAX_NORB_T) return false;
  int32_t node = 0;  // DRT root is always 0 in the current ASUKA DRT builder.
  nodes[0] = node;
  for (int k = 0; k < norb; k++) {
    int8_t step = (int8_t)((key >> (2 * k)) & 0x3ull);
    steps[k] = step;
    int32_t nxt = child[node * 4 + (int)step];
    if (nxt < 0) return false;
    node = nxt;
    nodes[k + 1] = node;
  }
  return true;
}

__device__ __forceinline__ uint64_t key64_set_step(uint64_t key, int k, int step) {
  uint64_t mask = 0x3ull << (2 * k);
  return (key & ~mask) | (uint64_t(step & 3) << (2 * k));
}

template <int MAX_NORB_T>
__device__ __forceinline__ bool epq_sample_one_reservoir_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int norb,
    int csf_idx,
    const int8_t* __restrict__ steps,
    const int32_t* __restrict__ nodes,
    int p,
    int q,
    uint64_t* rng_state,
    int* out_child,
    double* out_coeff,
    double* out_inv_p) {
  if (norb > MAX_NORB_T) return false;
  if (p == q) {
    int occ = step_to_occ(steps[p]);
    if (occ <= 0) return false;
    *out_child = csf_idx;
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

  int64_t idx_prefix[MAX_NORB_T + 1];
  idx_prefix[0] = 0;
  for (int kk = 0; kk < norb; kk++) {
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx_prefix[kk + 1] = idx_prefix[kk] + child_prefix[node_kk * 5 + step_kk];
  }

  int64_t prefix_offset = idx_prefix[start];
  int64_t prefix_endplus1 = idx_prefix[end + 1];
  int64_t suffix_offset = (int64_t)csf_idx - prefix_endplus1;

  int d_ref[MAX_NORB_T];
  int b_ref[MAX_NORB_T];
  for (int kk = 0; kk < norb; kk++) {
    d_ref[kk] = (int)steps[kk];
    b_ref[kk] = (int)node_twos[nodes[kk + 1]];
  }

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int64_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  double sum_abs = 0.0;
  int selected_child = -1;
  double selected_coeff = 0.0;
  double selected_abs = 0.0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int64_t seg_idx = st_seg[top];

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

    for (int which = 0; which < ndp; which++) {
      int dprime = (which == 0) ? dp0 : dp1;
      int child_k = child[node_k * 4 + dprime];
      if (child_k < 0) continue;
      int bprime = (int)node_twos[child_k];
      int db = bk - bprime;
      double seg = segment_value_int(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      double w2 = w * seg;
      int64_t seg_idx2 = seg_idx + child_prefix[node_k * 5 + dprime];

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = prefix_offset + seg_idx2 + suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;

        double wabs = fabs(w2);
        if (wabs == 0.0) continue;
        sum_abs += wabs;
        if (rand_u01(rng_state) * sum_abs < wabs) {
          selected_child = csf_i;
          selected_coeff = w2;
          selected_abs = wabs;
        }
      } else {
        if (top >= MAX_NORB_T) {
          // Stack overflow: abort sampling (should be extremely rare for typical CAS sizes).
          return false;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (sum_abs == 0.0 || selected_child < 0 || selected_abs == 0.0) return false;
  *out_child = selected_child;
  *out_coeff = selected_coeff;
  *out_inv_p = sum_abs / selected_abs;
  return true;
}

template <int MAX_NORB_T>
__device__ __forceinline__ bool epq_sample_one_reservoir_key64_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    int norb,
    uint64_t parent_key,
    const int8_t* __restrict__ steps,
    const int32_t* __restrict__ nodes,
    int p,
    int q,
    uint64_t* rng_state,
    uint64_t* out_child_key,
    double* out_coeff,
    double* out_inv_p) {
  if (norb > MAX_NORB_T) return false;
  if (p == q) {
    int occ = step_to_occ(steps[p]);
    if (occ <= 0) return false;
    *out_child_key = parent_key;
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

  int d_ref[MAX_NORB_T];
  int b_ref[MAX_NORB_T];
  for (int kk = 0; kk < norb; kk++) {
    d_ref[kk] = (int)steps[kk];
    b_ref[kk] = (int)node_twos[nodes[kk + 1]];
  }

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  uint64_t st_key[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_key[top] = parent_key;
  top++;

  double sum_abs = 0.0;
  uint64_t selected_key = 0;
  double selected_coeff = 0.0;
  double selected_abs = 0.0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    uint64_t cur_key = st_key[top];

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

    for (int which = 0; which < ndp; which++) {
      int dprime = (which == 0) ? dp0 : dp1;
      int child_k = child[node_k * 4 + dprime];
      if (child_k < 0) continue;
      int bprime = (int)node_twos[child_k];
      int db = bk - bprime;
      double seg = segment_value_int(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      double w2 = w * seg;
      uint64_t key2 = key64_set_step(cur_key, kpos, dprime);

      if (is_last) {
        if (child_k != node_end_target) continue;
        if (key2 == parent_key) continue;

        double wabs = fabs(w2);
        if (wabs == 0.0) continue;
        sum_abs += wabs;
        if (rand_u01(rng_state) * sum_abs < wabs) {
          selected_key = key2;
          selected_coeff = w2;
          selected_abs = wabs;
        }
      } else {
        if (top >= MAX_NORB_T) {
          // Stack overflow: abort sampling (should be extremely rare for typical CAS sizes).
          return false;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_key[top] = key2;
        top++;
      }
    }
  }

  if (sum_abs == 0.0 || selected_abs == 0.0) return false;
  *out_child_key = selected_key;
  *out_coeff = selected_coeff;
  *out_inv_p = sum_abs / selected_abs;
  return true;
}

template <int MAX_NORB_T>
__global__ void qmc_spawn_one_body_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ x_idx,  // [m] sorted unique
    const double* __restrict__ x_val,   // [m]
    int m,
    const double* __restrict__ h_eff_flat,  // [norb*norb] (p*norb+q)
    double eps,
    int nspawn,
    uint64_t seed,
    double initiator_t,
    int32_t* __restrict__ out_idx,  // [m*nspawn]
    double* __restrict__ out_val) { // [m*nspawn]
  int parent_pos = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (parent_pos >= m) return;
  if (norb > MAX_NORB_T) return;

  int j = (int)x_idx[parent_pos];
  double xj = x_val[parent_pos];

  int64_t base = (int64_t)parent_pos * (int64_t)nspawn;

  if (xj == 0.0) return;
  if ((unsigned)j >= (unsigned)ncsf) return;

  const int8_t* steps = steps_table + (int64_t)j * (int64_t)norb;
  const int32_t* nodes = nodes_table + (int64_t)j * (int64_t)(norb + 1);

  int src[MAX_NORB_T];
  int ns = 0;
  for (int orb = 0; orb < norb; orb++) {
    if (steps[orb] != 0) {
      src[ns] = orb;
      ns++;
    }
  }
  if (ns == 0) return;

  double inv_p_pair = (double)(ns * norb);
  double scale = -eps / (double)nspawn;

  bool allow_new = true;
  if (initiator_t > 0.0 && fabs(xj) < initiator_t) allow_new = false;

  uint64_t st = seed ^ ((uint64_t)(uint32_t)j * 0xD1B54A32D192ED03ull) ^ ((uint64_t)(uint32_t)parent_pos * 0x94D049BB133111EBull);

  for (int s = 0; s < nspawn; s++) {
    int q = src[(int)rand_below_u32(&st, (uint32_t)ns)];
    int p = (int)rand_below_u32(&st, (uint32_t)norb);

    double w = h_eff_flat[p * norb + q];
    if (w == 0.0) continue;

    int child_out = -1;
    double coeff_out = 0.0;
    double inv_p_out = 0.0;
    bool ok = epq_sample_one_reservoir_t<MAX_NORB_T>(
        child, node_twos, child_prefix, norb, j, steps, nodes, p, q, &st, &child_out, &coeff_out, &inv_p_out);
    if (!ok) continue;

    if (!allow_new) {
      if (!contains_sorted_i32(x_idx, m, (int32_t)child_out)) continue;
    }

    out_idx[base + s] = (int32_t)child_out;
    out_val[base + s] = scale * xj * w * coeff_out * inv_p_out * inv_p_pair;
  }
}

template <int MAX_NORB_T>
__global__ void qmc_spawn_hamiltonian_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ x_idx,  // [m] sorted unique
    const double* __restrict__ x_val,   // [m]
    int m,
    const double* __restrict__ h_base_flat,  // [norb*norb] (p*norb+q)
    const double* __restrict__ eri_mat,      // [nops*nops], nops=norb*norb (row-major)
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    int32_t* __restrict__ out_idx,  // [m*(nspawn_one+nspawn_two)]
    double* __restrict__ out_val) { // [m*(nspawn_one+nspawn_two)]
  int parent_pos = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (parent_pos >= m) return;
  if (norb > MAX_NORB_T) return;

  int j = (int)x_idx[parent_pos];
  double xj = x_val[parent_pos];
  if (xj == 0.0) return;
  if ((unsigned)j >= (unsigned)ncsf) return;

  const int8_t* steps_j = steps_table + (int64_t)j * (int64_t)norb;
  const int32_t* nodes_j = nodes_table + (int64_t)j * (int64_t)(norb + 1);

  int src[MAX_NORB_T];
  int dst[MAX_NORB_T];
  int ns = 0;
  int nd = 0;
  for (int orb = 0; orb < norb; orb++) {
    int8_t st = steps_j[orb];
    if (st != 0) {
      src[ns] = orb;
      ns++;
    }
    if (st != 3) {
      dst[nd] = orb;
      nd++;
    }
  }
  if (ns == 0) return;

  bool allow_new = true;
  if (initiator_t > 0.0 && fabs(xj) < initiator_t) allow_new = false;

  uint64_t st = seed ^ ((uint64_t)(uint32_t)j * 0xD1B54A32D192ED03ull) ^ ((uint64_t)(uint32_t)parent_pos * 0x94D049BB133111EBull);

  int nspawn_total = nspawn_one + nspawn_two;
  int64_t base = (int64_t)parent_pos * (int64_t)nspawn_total;

  int nops = norb * norb;

  // One-body effective part: h_eff(j)[p,q] = h_base[p,q] + 0.5 * Σ_r (p q | r r) occ_j[r]
  if (nspawn_one > 0) {
    double scale_one = -eps / (double)nspawn_one;
    double inv_p_pair_one = (double)(ns * norb);
    for (int s = 0; s < nspawn_one; s++) {
      int q = src[(int)rand_below_u32(&st, (uint32_t)ns)];
      int p = (int)rand_below_u32(&st, (uint32_t)norb);
      int pq_id = p * norb + q;

      double w_eff = h_base_flat[pq_id];
      double sum_rr = 0.0;
      for (int t = 0; t < ns; t++) {
        int r = src[t];
        int occ_r = (steps_j[r] == 3) ? 2 : 1;
        int rr_id = r * norb + r;
        sum_rr += eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rr_id] * (double)occ_r;
      }
      w_eff += 0.5 * sum_rr;
      if (w_eff == 0.0) continue;

      int child_out = -1;
      double coeff_out = 0.0;
      double inv_p_out = 0.0;
      bool ok = epq_sample_one_reservoir_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, j, steps_j, nodes_j, p, q, &st, &child_out, &coeff_out, &inv_p_out);
      if (!ok) continue;

      if (!allow_new) {
        if (!contains_sorted_i32(x_idx, m, (int32_t)child_out)) continue;
      }

      out_idx[base + s] = (int32_t)child_out;
      out_val[base + s] = scale_one * xj * w_eff * coeff_out * inv_p_out * inv_p_pair_one;
    }
  }

  // Two-body product term, restricted to r!=s: (1/2) Σ_{pqrs, r!=s} (pq|rs) E_pq E_rs
  if (nspawn_two > 0) {
    if (norb <= 1 || nd == 0) return;
    double scale_two = -eps / (double)nspawn_two;
    for (int s = 0; s < nspawn_two; s++) {
      int s_orb = src[(int)rand_below_u32(&st, (uint32_t)ns)];
      int occ_s = (steps_j[s_orb] == 3) ? 2 : 1;

      int r_orb = -1;
      double inv_p_pair_rs = 0.0;
      if (occ_s == 1) {
        if (nd <= 1) continue;
        while (true) {
          r_orb = dst[(int)rand_below_u32(&st, (uint32_t)nd)];
          if (r_orb != s_orb) break;
        }
        inv_p_pair_rs = (double)(ns * (nd - 1));
      } else {
        // occ_s == 2: s is not in dst.
        r_orb = dst[(int)rand_below_u32(&st, (uint32_t)nd)];
        inv_p_pair_rs = (double)(ns * nd);
      }

      int rs_id = r_orb * norb + s_orb;

      int k_csf = -1;
      double coeff_rs = 0.0;
      double inv_p_rs_epq = 0.0;
      bool ok_rs = epq_sample_one_reservoir_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, j, steps_j, nodes_j, r_orb, s_orb, &st, &k_csf, &coeff_rs, &inv_p_rs_epq);
      if (!ok_rs) continue;
      if ((unsigned)k_csf >= (unsigned)ncsf) continue;

      const int8_t* steps_k = steps_table + (int64_t)k_csf * (int64_t)norb;
      const int32_t* nodes_k = nodes_table + (int64_t)k_csf * (int64_t)(norb + 1);

      int src_k[MAX_NORB_T];
      int ns_k = 0;
      for (int orb = 0; orb < norb; orb++) {
        if (steps_k[orb] != 0) {
          src_k[ns_k] = orb;
          ns_k++;
        }
      }
      if (ns_k == 0) continue;

      int q_orb = src_k[(int)rand_below_u32(&st, (uint32_t)ns_k)];
      int p_orb = (int)rand_below_u32(&st, (uint32_t)norb);
      int pq_id = p_orb * norb + q_orb;

      double v_pqrs = 0.5 * eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rs_id];
      if (v_pqrs == 0.0) continue;

      int i_csf = -1;
      double coeff_pq = 0.0;
      double inv_p_pq_epq = 0.0;
      bool ok_pq = epq_sample_one_reservoir_t<MAX_NORB_T>(
          child, node_twos, child_prefix, norb, k_csf, steps_k, nodes_k, p_orb, q_orb, &st, &i_csf, &coeff_pq, &inv_p_pq_epq);
      if (!ok_pq) continue;

      if (!allow_new) {
        if (!contains_sorted_i32(x_idx, m, (int32_t)i_csf)) continue;
      }

      double inv_p_pair_pq = (double)(ns_k * norb);
      out_idx[base + (int64_t)nspawn_one + s] = (int32_t)i_csf;
      out_val[base + (int64_t)nspawn_one + s] = scale_two * xj * v_pqrs * coeff_rs * inv_p_rs_epq * inv_p_pair_rs * coeff_pq * inv_p_pq_epq * inv_p_pair_pq;
    }
  }
}

template <int MAX_NORB_T>
__global__ void qmc_spawn_hamiltonian_u64_f64_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    int norb,
    const uint64_t* __restrict__ x_key, // [m] (sorted unique if initiator_t > 0)
    const double* __restrict__ x_val,   // [m]
    int m,
    const double* __restrict__ h_base_flat, // [norb*norb] (p*norb+q)
    const double* __restrict__ eri_mat,     // [nops*nops], nops=norb*norb (row-major)
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    const float* __restrict__ pair_alias_prob, // [nops] (optional)
    const int32_t* __restrict__ pair_alias_idx, // [nops] (optional)
    const double* __restrict__ pair_norm,       // [nops] (optional, weights used for alias)
    double pair_norm_sum,
    int pair_sampling_mode,
    uint64_t* __restrict__ out_key, // [m*(nspawn_one+nspawn_two)]
    double* __restrict__ out_val) { // [m*(nspawn_one+nspawn_two)]
  int parent_pos = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (parent_pos >= m) return;
  if (norb > MAX_NORB_T) return;

  uint64_t parent_key = x_key[parent_pos];
  double xj = x_val[parent_pos];
  if (xj == 0.0) return;

  int8_t steps_j[MAX_NORB_T];
  int32_t nodes_j[MAX_NORB_T + 1];
  if (!key64_to_steps_nodes<MAX_NORB_T>(child, norb, parent_key, steps_j, nodes_j)) return;

  int src[MAX_NORB_T];
  int dst[MAX_NORB_T];
  int ns = 0;
  int nd = 0;
  for (int orb = 0; orb < norb; orb++) {
    int8_t stp = steps_j[orb];
    if (stp != 0) {
      src[ns] = orb;
      ns++;
    }
    if (stp != 3) {
      dst[nd] = orb;
      nd++;
    }
  }
  if (ns == 0) return;

  bool allow_new = true;
  if (initiator_t > 0.0 && fabs(xj) < initiator_t) allow_new = false;

  uint32_t key_mix = (uint32_t)(parent_key ^ (parent_key >> 32));
  uint64_t st = seed ^ ((uint64_t)key_mix * 0xD1B54A32D192ED03ull) ^ ((uint64_t)(uint32_t)parent_pos * 0x94D049BB133111EBull);

  int nspawn_total = nspawn_one + nspawn_two;
  int64_t base = (int64_t)parent_pos * (int64_t)nspawn_total;

  int nops = norb * norb;

  // One-body effective part: h_eff(j)[p,q] = h_base[p,q] + 0.5 * Σ_r (p q | r r) occ_j[r]
  if (nspawn_one > 0) {
    double scale_one = -eps / (double)nspawn_one;
    double inv_p_pair_one = (double)(ns * norb);
    for (int s = 0; s < nspawn_one; s++) {
      int q_orb = src[(int)rand_below_u32(&st, (uint32_t)ns)];
      int p_orb = (int)rand_below_u32(&st, (uint32_t)norb);
      int pq_id = p_orb * norb + q_orb;

      double w_eff = h_base_flat[pq_id];
      double sum_rr = 0.0;
      for (int t = 0; t < ns; t++) {
        int r = src[t];
        int occ_r = (steps_j[r] == 3) ? 2 : 1;
        int rr_id = r * norb + r;
        sum_rr += eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rr_id] * (double)occ_r;
      }
      w_eff += 0.5 * sum_rr;
      if (w_eff == 0.0) continue;

      uint64_t child_key = 0;
      double coeff_out = 0.0;
      double inv_p_out = 0.0;
      bool ok = epq_sample_one_reservoir_key64_t<MAX_NORB_T>(
          child, node_twos, norb, parent_key, steps_j, nodes_j, p_orb, q_orb, &st, &child_key, &coeff_out, &inv_p_out);
      if (!ok) continue;

      if (!allow_new) {
        if (!contains_sorted_u64(x_key, m, child_key)) continue;
      }

      out_key[base + s] = child_key;
      out_val[base + s] = scale_one * xj * w_eff * coeff_out * inv_p_out * inv_p_pair_one;
    }
  }

  // Two-body product term, restricted to r!=s: (1/2) Σ_{pqrs, r!=s} (pq|rs) E_pq E_rs
  if (nspawn_two > 0) {
    if (norb <= 1) return;
    double scale_two = -eps / (double)nspawn_two;

    bool use_pair_norm =
        (pair_sampling_mode == 1) && (pair_alias_prob != nullptr) && (pair_alias_idx != nullptr) && (pair_norm != nullptr) && (pair_norm_sum > 0.0);

    for (int s = 0; s < nspawn_two; s++) {
      int r_orb = -1;
      int s_orb = -1;
      int rs_id = -1;
      double inv_p_pair_rs = 0.0;

      if (use_pair_norm) {
        rs_id = alias_sample_i32(pair_alias_prob, pair_alias_idx, nops, &st);
        r_orb = rs_id / norb;
        s_orb = rs_id - r_orb * norb;
        if (r_orb == s_orb) continue;
        int8_t st_s = steps_j[s_orb];
        if (st_s == 0) continue;
        if (steps_j[r_orb] == 3) continue;
        double pn_rs = pair_norm[rs_id];
        if (!(pn_rs > 0.0)) continue;
        inv_p_pair_rs = pair_norm_sum / pn_rs;
      } else {
        if (nd == 0) return;
        s_orb = src[(int)rand_below_u32(&st, (uint32_t)ns)];
        int occ_s = (steps_j[s_orb] == 3) ? 2 : 1;
        if (occ_s == 1) {
          if (nd <= 1) continue;
          while (true) {
            r_orb = dst[(int)rand_below_u32(&st, (uint32_t)nd)];
            if (r_orb != s_orb) break;
          }
          inv_p_pair_rs = (double)(ns * (nd - 1));
        } else {
          // occ_s == 2: s is not in dst.
          r_orb = dst[(int)rand_below_u32(&st, (uint32_t)nd)];
          inv_p_pair_rs = (double)(ns * nd);
        }
        rs_id = r_orb * norb + s_orb;
      }

      uint64_t k_key = 0;
      double coeff_rs = 0.0;
      double inv_p_rs_epq = 0.0;
      bool ok_rs = epq_sample_one_reservoir_key64_t<MAX_NORB_T>(
          child, node_twos, norb, parent_key, steps_j, nodes_j, r_orb, s_orb, &st, &k_key, &coeff_rs, &inv_p_rs_epq);
      if (!ok_rs) continue;

      int8_t steps_k[MAX_NORB_T];
      int32_t nodes_k[MAX_NORB_T + 1];
      if (!key64_to_steps_nodes<MAX_NORB_T>(child, norb, k_key, steps_k, nodes_k)) continue;

      int p_orb = -1;
      int q_orb = -1;
      int pq_id = -1;
      double inv_p_pair_pq = 0.0;
      if (use_pair_norm) {
        pq_id = alias_sample_i32(pair_alias_prob, pair_alias_idx, nops, &st);
        p_orb = pq_id / norb;
        q_orb = pq_id - p_orb * norb;
        double pn_pq = pair_norm[pq_id];
        if (!(pn_pq > 0.0)) continue;
        inv_p_pair_pq = pair_norm_sum / pn_pq;
      } else {
        int src_k[MAX_NORB_T];
        int ns_k = 0;
        for (int orb = 0; orb < norb; orb++) {
          if (steps_k[orb] != 0) {
            src_k[ns_k] = orb;
            ns_k++;
          }
        }
        if (ns_k == 0) continue;

        q_orb = src_k[(int)rand_below_u32(&st, (uint32_t)ns_k)];
        p_orb = (int)rand_below_u32(&st, (uint32_t)norb);
        pq_id = p_orb * norb + q_orb;
        inv_p_pair_pq = (double)(ns_k * norb);
      }

      double v_pqrs = 0.5 * eri_mat[(int64_t)pq_id * (int64_t)nops + (int64_t)rs_id];
      if (v_pqrs == 0.0) continue;

      uint64_t i_key = 0;
      double coeff_pq = 0.0;
      double inv_p_pq_epq = 0.0;
      bool ok_pq = epq_sample_one_reservoir_key64_t<MAX_NORB_T>(
          child, node_twos, norb, k_key, steps_k, nodes_k, p_orb, q_orb, &st, &i_key, &coeff_pq, &inv_p_pq_epq);
      if (!ok_pq) continue;

      if (!allow_new) {
        if (!contains_sorted_u64(x_key, m, i_key)) continue;
      }

      out_key[base + (int64_t)nspawn_one + s] = i_key;
      out_val[base + (int64_t)nspawn_one + s] =
          scale_two * xj * v_pqrs * coeff_rs * inv_p_rs_epq * inv_p_pair_rs * coeff_pq * inv_p_pq_epq * inv_p_pair_pq;
    }
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_spawn_one_body_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* x_idx,
    const double* x_val,
    int m,
    const double* h_eff_flat,
    double eps,
    int nspawn,
    uint64_t seed,
    double initiator_t,
    int32_t* out_idx,
    double* out_val,
    cudaStream_t stream,
    int threads) {
  if (!child || !node_twos || !child_prefix) return cudaErrorInvalidValue;
  if (!steps_table || !nodes_table) return cudaErrorInvalidValue;
  if (!x_idx || !x_val) return cudaErrorInvalidValue;
  if (!h_eff_flat || !out_idx || !out_val) return cudaErrorInvalidValue;
  if (m < 0 || ncsf < 0 || norb <= 0 || norb > MAX_NORB) return cudaErrorInvalidValue;
  if (nspawn <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  int blocks = (m + threads - 1) / threads;
  if (norb <= 32) {
    qmc_spawn_one_body_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        x_idx,
        x_val,
        m,
        h_eff_flat,
        eps,
        nspawn,
        seed,
        initiator_t,
        out_idx,
        out_val);
  } else {
    qmc_spawn_one_body_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        x_idx,
        x_val,
        m,
        h_eff_flat,
        eps,
        nspawn,
        seed,
        initiator_t,
        out_idx,
        out_val);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t guga_qmc_spawn_one_body_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* x_idx,
    const double* x_val,
    int m,
    const double* h_eff_flat,
    double eps,
    int nspawn,
    uint64_t seed,
    double initiator_t,
    int32_t* out_idx,
    double* out_val,
    int threads) {
  return guga_qmc_spawn_one_body_launch_stream(
      child,
      node_twos,
      child_prefix,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      x_idx,
      x_val,
      m,
      h_eff_flat,
      eps,
      nspawn,
      seed,
      initiator_t,
      out_idx,
      out_val,
      /*stream=*/0,
      threads);
}

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* x_idx,
    const double* x_val,
    int m,
    const double* h_base_flat,
    const double* eri_mat,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    int32_t* out_idx,
    double* out_val,
    cudaStream_t stream,
    int threads) {
  if (!child || !node_twos || !child_prefix) return cudaErrorInvalidValue;
  if (!steps_table || !nodes_table) return cudaErrorInvalidValue;
  if (!x_idx || !x_val) return cudaErrorInvalidValue;
  if (!h_base_flat || !eri_mat || !out_idx || !out_val) return cudaErrorInvalidValue;
  if (m < 0 || ncsf < 0 || norb <= 0 || norb > MAX_NORB) return cudaErrorInvalidValue;
  if (nspawn_one < 0 || nspawn_two < 0) return cudaErrorInvalidValue;
  if (nspawn_one == 0 && nspawn_two == 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  int blocks = (m + threads - 1) / threads;
  if (norb <= 32) {
    qmc_spawn_hamiltonian_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        x_idx,
        x_val,
        m,
        h_base_flat,
        eri_mat,
        eps,
        nspawn_one,
        nspawn_two,
        seed,
        initiator_t,
        out_idx,
        out_val);
  } else {
    qmc_spawn_hamiltonian_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        x_idx,
        x_val,
        m,
        h_base_flat,
        eri_mat,
        eps,
        nspawn_one,
        nspawn_two,
        seed,
        initiator_t,
        out_idx,
        out_val);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* x_idx,
    const double* x_val,
    int m,
    const double* h_base_flat,
    const double* eri_mat,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    int32_t* out_idx,
    double* out_val,
    int threads) {
  return guga_qmc_spawn_hamiltonian_launch_stream(
      child,
      node_twos,
      child_prefix,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      x_idx,
      x_val,
      m,
      h_base_flat,
      eri_mat,
      eps,
      nspawn_one,
      nspawn_two,
      seed,
      initiator_t,
      out_idx,
      out_val,
      /*stream=*/0,
      threads);
}

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_u64_f64_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    int norb,
    const uint64_t* x_key,
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
    uint64_t* out_key,
    double* out_val,
    cudaStream_t stream,
    int threads) {
  if (!child || !node_twos) return cudaErrorInvalidValue;
  if (!x_key || !x_val) return cudaErrorInvalidValue;
  if (!h_base_flat || !eri_mat || !out_key || !out_val) return cudaErrorInvalidValue;
  if (m < 0 || norb <= 0 || norb > 32) return cudaErrorInvalidValue;
  if (nspawn_one < 0 || nspawn_two < 0) return cudaErrorInvalidValue;
  if (nspawn_one == 0 && nspawn_two == 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  int blocks = (m + threads - 1) / threads;
  qmc_spawn_hamiltonian_u64_f64_kernel_t<32><<<blocks, threads, 0, stream>>>(
      child,
      node_twos,
      norb,
      x_key,
      x_val,
      m,
      h_base_flat,
      eri_mat,
      eps,
      nspawn_one,
      nspawn_two,
      seed,
      initiator_t,
      pair_alias_prob,
      pair_alias_idx,
      pair_norm,
      pair_norm_sum,
      pair_sampling_mode,
      out_key,
      out_val);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_qmc_spawn_hamiltonian_u64_f64_launch(
    const int32_t* child,
    const int16_t* node_twos,
    int norb,
    const uint64_t* x_key,
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
    uint64_t* out_key,
    double* out_val,
    int threads) {
  return guga_qmc_spawn_hamiltonian_u64_f64_launch_stream(
      child,
      node_twos,
      norb,
      x_key,
      x_val,
      m,
      h_base_flat,
      eri_mat,
      pair_alias_prob,
      pair_alias_idx,
      pair_norm,
      pair_norm_sum,
      pair_sampling_mode,
      eps,
      nspawn_one,
      nspawn_two,
      seed,
      initiator_t,
      out_key,
      out_val,
      /*stream=*/0,
      threads);
}

// -----------------------------------------------------------------------------
// QMC: COO event coalesce (sort + reduce-by-key)
// -----------------------------------------------------------------------------

namespace {

struct QmcF64Sum {
  __host__ __device__ __forceinline__ double operator()(double a, double b) const { return a + b; }
};

__global__ void qmc_sanitize_invalid_idx_kernel(int32_t* idx, double* val, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  int32_t k = idx[i];
  if (k < 0) {
    idx[i] = (int32_t)-1;
    val[i] = 0.0;
  }
}

__global__ void qmc_strip_invalid_prefix_kernel(int32_t* idx, double* val, int* nnz) {
  int n = nnz ? nnz[0] : 0;
  if (n <= 0) return;
  if (idx[0] >= 0) return;
  int out_n = n - 1;
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < out_n) {
    idx[i] = idx[i + 1];
    val[i] = val[i + 1];
  }
  if (i == 0) nnz[0] = out_n;
}

}  // namespace

extern "C" cudaError_t guga_qmc_coalesce_coo_i32_f64_launch_stream(
    const int32_t* idx_in,
    const double* val_in,
    int n,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    cudaStream_t stream,
    int threads) {
  if (!idx_in || !val_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  // Scratch buffers (input copy) so we never mutate idx_in/val_in.
  int32_t* d_idx_tmp_raw = nullptr;
  double* d_val_tmp_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_idx_tmp_raw, (size_t)n * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_val_tmp_raw, (size_t)n * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_idx_tmp(d_idx_tmp_raw, CudaFreeStreamDeleter<int32_t>{stream});
  cuda_unique_ptr_stream<double> d_val_tmp(d_val_tmp_raw, CudaFreeStreamDeleter<double>{stream});

  err = cudaMemcpyAsync(d_idx_tmp.get(), idx_in, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) return err;
  err = cudaMemcpyAsync(d_val_tmp.get(), val_in, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) return err;

  // Ensure invalid entries contribute exactly zero under key -1.
  {
    int blocks = (n + threads - 1) / threads;
    qmc_sanitize_invalid_idx_kernel<<<blocks, threads, 0, stream>>>(d_idx_tmp.get(), d_val_tmp.get(), n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<int32_t> keys(d_idx_tmp.get(), idx_out);
  cub::DoubleBuffer<double> vals(d_val_tmp.get(), val_out);

  size_t temp_sort = 0;
  err = cub::DeviceRadixSort::SortPairs(nullptr, temp_sort, keys, vals, n, 0, 32, stream);
  if (err != cudaSuccess) return err;

  // Reduce output must not alias the sorted input. Prefer writing into idx_out/val_out;
  // if the sort output ends up in those buffers, reduce into the tmp buffers and copy back.
  size_t temp_reduce = 0;
  const int32_t* d_key_sorted = keys.Current();
  const double* d_val_sorted = vals.Current();
  int32_t* d_key_out = (d_key_sorted == idx_out) ? d_idx_tmp.get() : idx_out;
  double* d_val_out = (d_val_sorted == val_out) ? d_val_tmp.get() : val_out;

  err = cub::DeviceReduce::ReduceByKey(
      nullptr, temp_reduce, d_key_sorted, d_key_out, d_val_sorted, d_val_out, out_nnz, QmcF64Sum(), n, stream);
  if (err != cudaSuccess) return err;

  size_t temp_bytes = (temp_sort > temp_reduce) ? temp_sort : temp_reduce;
  void* d_temp_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp((void*)d_temp_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceRadixSort::SortPairs(d_temp.get(), temp_sort, keys, vals, n, 0, 32, stream);
  if (err != cudaSuccess) return err;

  d_key_sorted = keys.Current();
  d_val_sorted = vals.Current();
  d_key_out = (d_key_sorted == idx_out) ? d_idx_tmp.get() : idx_out;
  d_val_out = (d_val_sorted == val_out) ? d_val_tmp.get() : val_out;

  err = cub::DeviceReduce::ReduceByKey(
      d_temp.get(), temp_reduce, d_key_sorted, d_key_out, d_val_sorted, d_val_out, out_nnz, QmcF64Sum(), n, stream);
  if (err != cudaSuccess) return err;

  if (d_key_out != idx_out) {
    err = cudaMemcpyAsync(idx_out, d_key_out, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
  }
  if (d_val_out != val_out) {
    err = cudaMemcpyAsync(val_out, d_val_out, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
  }

  // Drop the (-1,0) sentinel group if present (spawn buffers use idx==-1 for unused slots).
  {
    int t = (threads > 0 ? threads : 256);
    int blocks = (n + t - 1) / t;
    qmc_strip_invalid_prefix_kernel<<<blocks, t, 0, stream>>>(idx_out, val_out, out_nnz);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return cudaSuccess;
}

extern "C" cudaError_t guga_qmc_coalesce_coo_i32_f64_launch(
    const int32_t* idx_in,
    const double* val_in,
    int n,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int threads) {
  return guga_qmc_coalesce_coo_i32_f64_launch_stream(idx_in, val_in, n, idx_out, val_out, out_nnz, /*stream=*/0, threads);
}


// -----------------------------------------------------------------------------
// QMC: guided accept/thin for COO event buffers (idx==-1 sentinel)
// -----------------------------------------------------------------------------

namespace {

__global__ void qmc_guided_thin_events_kernel(
    const int32_t* __restrict__ idx_in,
    const double* __restrict__ val_in,
    const double* __restrict__ q_in,
    int n,
    double alpha,
    double q_scale,
    double min_accept,
    uint64_t seed,
    int32_t* __restrict__ idx_out,
    double* __restrict__ val_out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  int32_t idx = idx_in[i];
  double v = val_in[i];
  if (idx < 0 || v == 0.0) {
    idx_out[i] = (int32_t)-1;
    val_out[i] = 0.0;
    return;
  }

  double qi = q_in ? q_in[i] : 1.0;
  double a = 1.0;
  if (alpha > 0.0) {
    double t = qi * q_scale;
    if (t <= 0.0) {
      a = min_accept;
    } else {
      a = pow(t, alpha);
      if (a < min_accept) a = min_accept;
      if (a > 1.0) a = 1.0;
    }
  } else {
    a = 1.0;
  }

  // Deterministic per-entry RNG stream.
  uint64_t st = seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(i + 1));
  double u = rand_u01(&st);

  if (u < a) {
    idx_out[i] = idx;
    val_out[i] = v / a;
  } else {
    idx_out[i] = (int32_t)-1;
    val_out[i] = 0.0;
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_guided_thin_events_i32_f64_launch_stream(
    const int32_t* idx_in,
    const double* val_in,
    const double* q_in,
    int n,
    int32_t* idx_out,
    double* val_out,
    double alpha,
    double q_scale,
    double min_accept,
    uint64_t seed,
    cudaStream_t stream,
    int threads) {
  if (!idx_in || !val_in || !q_in || !idx_out || !val_out) return cudaErrorInvalidValue;
  if (n < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (min_accept < 0.0 || min_accept > 1.0) return cudaErrorInvalidValue;

  if (n == 0) return cudaSuccess;

  int blocks = (n + threads - 1) / threads;
  qmc_guided_thin_events_kernel<<<blocks, threads, 0, stream>>>(idx_in, val_in, q_in, n, alpha, q_scale, min_accept, seed, idx_out, val_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_qmc_guided_thin_events_i32_f64_launch(
    const int32_t* idx_in,
    const double* val_in,
    const double* q_in,
    int n,
    int32_t* idx_out,
    double* val_out,
    double alpha,
    double q_scale,
    double min_accept,
    uint64_t seed,
    int threads) {
  return guga_qmc_guided_thin_events_i32_f64_launch_stream(
      idx_in, val_in, q_in, n, idx_out, val_out, alpha, q_scale, min_accept, seed, /*stream=*/0, threads);
}


// -----------------------------------------------------------------------------
// QMC: Φ compression (pivot + systematic resampling)
// -----------------------------------------------------------------------------

namespace {

struct QmcIdxVal {
  int32_t idx;
  double val;
};

__global__ void qmc_pack_abs_and_pairs_kernel(const int32_t* idx, const double* val, double* abs_out, QmcIdxVal* pair_out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  int32_t k = idx[i];
  double v = val[i];
  abs_out[i] = fabs(v);
  pair_out[i].idx = k;
  pair_out[i].val = v;
}

__device__ __forceinline__ int lower_bound_cdf_f64(const double* __restrict__ cdf, int n, double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    double v = cdf[mid];
    if (v < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void qmc_systematic_resample_kernel(
    const QmcIdxVal* __restrict__ pairs_r,
    const double* __restrict__ cdf_r,
    int r_len,
    int k_samp,
    uint64_t seed,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= k_samp) return;
  if (r_len <= 0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double W = cdf_r[r_len - 1];
  if (W <= 0.0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double step = W / (double)k_samp;
  uint64_t st = seed;
  double u0 = rand_u01(&st) * step;
  double u = u0 + (double)k * step;
  int pos = lower_bound_cdf_f64(cdf_r, r_len, u);
  if (pos < 0) pos = 0;
  if (pos >= r_len) pos = r_len - 1;

  QmcIdxVal it = pairs_r[pos];
  out_idx[k] = it.idx;
  out_val[k] = (it.val >= 0.0 ? 1.0 : -1.0) * step;
}

__global__ void qmc_build_phi_buffer_kernel(
    const QmcIdxVal* __restrict__ pairs_sorted,
    int n_in,
    int p_piv,
    const int32_t* __restrict__ samp_idx,
    const double* __restrict__ samp_val,
    int k_samp,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int n_out = p_piv + k_samp;
  if (i >= n_out) return;
  if (i < p_piv) {
    QmcIdxVal it = pairs_sorted[i];
    out_idx[i] = it.idx;
    out_val[i] = it.val;
  } else {
    int k = i - p_piv;
    out_idx[i] = samp_idx[k];
    out_val[i] = samp_val[k];
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_phi_pivot_resample_i32_f64_launch_stream(
    const int32_t* idx_in,
    const double* val_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    uint64_t seed,
    cudaStream_t stream,
    int threads) {
  if (!idx_in || !val_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n_in < 0 || m < 0 || pivot < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n_in == 0 || m == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  if (n_in <= m) {
    cudaError_t err = cudaMemcpyAsync(idx_out, idx_in, (size_t)n_in * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(val_out, val_in, (size_t)n_in * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out_nnz, &n_in, sizeof(int), cudaMemcpyHostToDevice, stream);
    return err;
  }

  int p = pivot;
  if (p > m) p = m;
  if (p > n_in) p = n_in;
  int k_samp = m - p;
  int r_len = n_in - p;

  // Sort by |val| descending.
  double* d_abs_a_raw = nullptr;
  double* d_abs_b_raw = nullptr;
  QmcIdxVal* d_pair_a_raw = nullptr;
  QmcIdxVal* d_pair_b_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_abs_a_raw, (size_t)n_in * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_abs_b_raw, (size_t)n_in * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_pair_a_raw, (size_t)n_in * sizeof(QmcIdxVal), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_pair_b_raw, (size_t)n_in * sizeof(QmcIdxVal), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<double> d_abs_a(d_abs_a_raw, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<double> d_abs_b(d_abs_b_raw, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<QmcIdxVal> d_pair_a(d_pair_a_raw, CudaFreeStreamDeleter<QmcIdxVal>{stream});
  cuda_unique_ptr_stream<QmcIdxVal> d_pair_b(d_pair_b_raw, CudaFreeStreamDeleter<QmcIdxVal>{stream});

  {
    int blocks = (n_in + threads - 1) / threads;
    qmc_pack_abs_and_pairs_kernel<<<blocks, threads, 0, stream>>>(idx_in, val_in, d_abs_a.get(), d_pair_a.get(), n_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<double> keys(d_abs_a.get(), d_abs_b.get());
  cub::DoubleBuffer<QmcIdxVal> vals(d_pair_a.get(), d_pair_b.get());

  size_t temp_sort = 0;
  err = cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_sort, keys, vals, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_sort_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_sort_raw, temp_sort, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_sort((void*)d_temp_sort_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceRadixSort::SortPairsDescending(d_temp_sort.get(), temp_sort, keys, vals, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;

  const QmcIdxVal* d_pairs_sorted = vals.Current();
  const double* d_abs_sorted = keys.Current();

  // If we have no sampling mass, fall back to pivots-only.
  if (k_samp <= 0 || r_len <= 0) {
    // Build pivot buffer of length p and coalesce (sort by idx).
    int32_t* d_idx_buf_raw = nullptr;
    double* d_val_buf_raw = nullptr;
    err = guga_cuda_malloc(&d_idx_buf_raw, (size_t)p * sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;
    err = guga_cuda_malloc(&d_val_buf_raw, (size_t)p * sizeof(double), stream);
    if (err != cudaSuccess) return err;
    cuda_unique_ptr_stream<int32_t> d_idx_buf(d_idx_buf_raw, CudaFreeStreamDeleter<int32_t>{stream});
    cuda_unique_ptr_stream<double> d_val_buf(d_val_buf_raw, CudaFreeStreamDeleter<double>{stream});

    {
      int blocks = (p + threads - 1) / threads;
      qmc_build_phi_buffer_kernel<<<blocks, threads, 0, stream>>>(d_pairs_sorted, n_in, p, nullptr, nullptr, 0, d_idx_buf.get(), d_val_buf.get());
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
    }
    return guga_qmc_coalesce_coo_i32_f64_launch_stream(
        d_idx_buf.get(), d_val_buf.get(), p, idx_out, val_out, out_nnz, stream, threads);
  }

  // Remainder CDF via inclusive scan on |val| for entries [p, n_in).
  double* d_cdf_raw = nullptr;
  err = guga_cuda_malloc(&d_cdf_raw, (size_t)r_len * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<double> d_cdf(d_cdf_raw, CudaFreeStreamDeleter<double>{stream});

  size_t temp_scan = 0;
  err = cub::DeviceScan::InclusiveSum(nullptr, temp_scan, d_abs_sorted + p, d_cdf.get(), r_len, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_scan_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_scan_raw, temp_scan, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_scan((void*)d_temp_scan_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceScan::InclusiveSum(d_temp_scan.get(), temp_scan, d_abs_sorted + p, d_cdf.get(), r_len, stream);
  if (err != cudaSuccess) return err;

  // Sample K indices/values.
  int32_t* d_samp_idx_raw = nullptr;
  double* d_samp_val_raw = nullptr;
  err = guga_cuda_malloc(&d_samp_idx_raw, (size_t)k_samp * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_samp_val_raw, (size_t)k_samp * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_samp_idx(d_samp_idx_raw, CudaFreeStreamDeleter<int32_t>{stream});
  cuda_unique_ptr_stream<double> d_samp_val(d_samp_val_raw, CudaFreeStreamDeleter<double>{stream});

  {
    int blocks = (k_samp + threads - 1) / threads;
    qmc_systematic_resample_kernel<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted + p, d_cdf.get(), r_len, k_samp, seed, d_samp_idx.get(), d_samp_val.get());
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  // Build (pivot + samples) buffer of length m, then coalesce by index.
  int32_t* d_idx_buf_raw = nullptr;
  double* d_val_buf_raw = nullptr;
  err = guga_cuda_malloc(&d_idx_buf_raw, (size_t)m * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_val_buf_raw, (size_t)m * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_idx_buf(d_idx_buf_raw, CudaFreeStreamDeleter<int32_t>{stream});
  cuda_unique_ptr_stream<double> d_val_buf(d_val_buf_raw, CudaFreeStreamDeleter<double>{stream});

  {
    int blocks = (m + threads - 1) / threads;
    qmc_build_phi_buffer_kernel<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted, n_in, p, d_samp_idx.get(), d_samp_val.get(), k_samp, d_idx_buf.get(), d_val_buf.get());
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return guga_qmc_coalesce_coo_i32_f64_launch_stream(d_idx_buf.get(), d_val_buf.get(), m, idx_out, val_out, out_nnz, stream, threads);
}

extern "C" cudaError_t guga_qmc_phi_pivot_resample_i32_f64_launch(
    const int32_t* idx_in,
    const double* val_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    uint64_t seed,
    int threads) {
  return guga_qmc_phi_pivot_resample_i32_f64_launch_stream(
      idx_in, val_in, n_in, idx_out, val_out, out_nnz, m, pivot, seed, /*stream=*/0, threads);
}


// -----------------------------------------------------------------------------
// QMC: guided Φ compression (pivot + systematic resampling)
// -----------------------------------------------------------------------------

namespace {

struct QmcIdxValQ {
  int32_t idx;
  double val;
  double q;
};

__global__ void qmc_pack_abs_and_pairs_q_kernel(
    const int32_t* idx,
    const double* val,
    const double* q,
    double* abs_out,
    QmcIdxValQ* pair_out,
    int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  int32_t k = idx[i];
  double v = val[i];
  double qq = q ? q[i] : 0.0;
  abs_out[i] = fabs(v);
  pair_out[i].idx = k;
  pair_out[i].val = v;
  pair_out[i].q = qq;
}

__global__ void qmc_compute_guided_weights_kernel(
    const double* __restrict__ abs_r,
    const QmcIdxValQ* __restrict__ pairs_r,
    int r_len,
    double alpha,
    double q_floor,
    double* __restrict__ w_out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= r_len) return;
  double a = abs_r[i];
  double qv = pairs_r[i].q;
  if (qv < q_floor) qv = q_floor;

  double w = 0.0;
  if (alpha <= 0.0) {
    w = a;
  } else if (alpha >= 1.0) {
    w = qv;
  } else {
    if (a > 0.0 && qv > 0.0) {
      w = pow(a, 1.0 - alpha) * pow(qv, alpha);
    } else {
      w = 0.0;
    }
  }
  w_out[i] = w;
}

__global__ void qmc_systematic_resample_guided_kernel(
    const QmcIdxValQ* __restrict__ pairs_r,
    const double* __restrict__ abs_r,
    const double* __restrict__ w_r,
    const double* __restrict__ cdf_r,
    int r_len,
    int k_samp,
    uint64_t seed,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= k_samp) return;
  if (r_len <= 0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double W = cdf_r[r_len - 1];
  if (W <= 0.0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double step = W / (double)k_samp;
  uint64_t st = seed;
  double u0 = rand_u01(&st) * step;
  double u = u0 + (double)k * step;
  int pos = lower_bound_cdf_f64(cdf_r, r_len, u);
  if (pos < 0) pos = 0;
  if (pos >= r_len) pos = r_len - 1;

  QmcIdxValQ it = pairs_r[pos];
  double a = abs_r[pos];
  double w = w_r[pos];
  if (w <= 0.0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  out_idx[k] = it.idx;
  out_val[k] = (it.val >= 0.0 ? 1.0 : -1.0) * step * (a / w);
}

__global__ void qmc_build_phi_buffer_kernel_q(
    const QmcIdxValQ* __restrict__ pairs_sorted,
    int n_in,
    int p_piv,
    const int32_t* __restrict__ samp_idx,
    const double* __restrict__ samp_val,
    int k_samp,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int n_out = p_piv + k_samp;
  if (i >= n_out) return;
  if (i < p_piv) {
    QmcIdxValQ it = pairs_sorted[i];
    out_idx[i] = it.idx;
    out_val[i] = it.val;
  } else {
    int k = i - p_piv;
    out_idx[i] = samp_idx[k];
    out_val[i] = samp_val[k];
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_phi_pivot_resample_guided_i32_f64_launch_stream(
    const int32_t* idx_in,
    const double* val_in,
    const double* q_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    double alpha,
    double q_floor,
    uint64_t seed,
    cudaStream_t stream,
    int threads) {
  if (!idx_in || !val_in || !q_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n_in < 0 || m < 0 || pivot < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n_in == 0 || m == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  if (n_in <= m) {
    cudaError_t err = cudaMemcpyAsync(idx_out, idx_in, (size_t)n_in * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(val_out, val_in, (size_t)n_in * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out_nnz, &n_in, sizeof(int), cudaMemcpyHostToDevice, stream);
    return err;
  }

  int p = pivot;
  if (p > m) p = m;
  if (p > n_in) p = n_in;
  int k_samp = m - p;
  int r_len = n_in - p;

  // Sort by |val| descending (same pivot selection as the unguided kernel).
  double* d_abs_a_raw = nullptr;
  double* d_abs_b_raw = nullptr;
  QmcIdxValQ* d_pair_a_raw = nullptr;
  QmcIdxValQ* d_pair_b_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_abs_a_raw, (size_t)n_in * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_abs_b_raw, (size_t)n_in * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_pair_a_raw, (size_t)n_in * sizeof(QmcIdxValQ), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_pair_b_raw, (size_t)n_in * sizeof(QmcIdxValQ), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<double> d_abs_a(d_abs_a_raw, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<double> d_abs_b(d_abs_b_raw, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<QmcIdxValQ> d_pair_a(d_pair_a_raw, CudaFreeStreamDeleter<QmcIdxValQ>{stream});
  cuda_unique_ptr_stream<QmcIdxValQ> d_pair_b(d_pair_b_raw, CudaFreeStreamDeleter<QmcIdxValQ>{stream});

  {
    int blocks = (n_in + threads - 1) / threads;
    qmc_pack_abs_and_pairs_q_kernel<<<blocks, threads, 0, stream>>>(idx_in, val_in, q_in, d_abs_a.get(), d_pair_a.get(), n_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<double> keys(d_abs_a.get(), d_abs_b.get());
  cub::DoubleBuffer<QmcIdxValQ> vals(d_pair_a.get(), d_pair_b.get());

  size_t temp_sort = 0;
  err = cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_sort, keys, vals, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_sort_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_sort_raw, temp_sort, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_sort((void*)d_temp_sort_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceRadixSort::SortPairsDescending(d_temp_sort.get(), temp_sort, keys, vals, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;

  const QmcIdxValQ* d_pairs_sorted = vals.Current();
  const double* d_abs_sorted = keys.Current();

  // If we have no sampling mass, fall back to pivots-only.
  if (k_samp <= 0 || r_len <= 0) {
    int32_t* d_idx_buf_raw = nullptr;
    double* d_val_buf_raw = nullptr;
    err = guga_cuda_malloc(&d_idx_buf_raw, (size_t)p * sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;
    err = guga_cuda_malloc(&d_val_buf_raw, (size_t)p * sizeof(double), stream);
    if (err != cudaSuccess) return err;
    cuda_unique_ptr_stream<int32_t> d_idx_buf(d_idx_buf_raw, CudaFreeStreamDeleter<int32_t>{stream});
    cuda_unique_ptr_stream<double> d_val_buf(d_val_buf_raw, CudaFreeStreamDeleter<double>{stream});

    {
      int blocks = (p + threads - 1) / threads;
      qmc_build_phi_buffer_kernel_q<<<blocks, threads, 0, stream>>>(d_pairs_sorted, n_in, p, nullptr, nullptr, 0, d_idx_buf.get(), d_val_buf.get());
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
    }
    return guga_qmc_coalesce_coo_i32_f64_launch_stream(d_idx_buf.get(), d_val_buf.get(), p, idx_out, val_out, out_nnz, stream, threads);
  }

  // Compute guided weights for the remainder segment [p, n_in).
  double* d_w_raw = nullptr;
  double* d_cdf_raw = nullptr;
  err = guga_cuda_malloc(&d_w_raw, (size_t)r_len * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_cdf_raw, (size_t)r_len * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<double> d_w(d_w_raw, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<double> d_cdf(d_cdf_raw, CudaFreeStreamDeleter<double>{stream});

  {
    int blocks = (r_len + threads - 1) / threads;
    qmc_compute_guided_weights_kernel<<<blocks, threads, 0, stream>>>(d_abs_sorted + p, d_pairs_sorted + p, r_len, alpha, q_floor, d_w.get());
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  // Inclusive scan on weights.
  size_t temp_scan = 0;
  err = cub::DeviceScan::InclusiveSum(nullptr, temp_scan, d_w.get(), d_cdf.get(), r_len, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_scan_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_scan_raw, temp_scan, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_scan((void*)d_temp_scan_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceScan::InclusiveSum(d_temp_scan.get(), temp_scan, d_w.get(), d_cdf.get(), r_len, stream);
  if (err != cudaSuccess) return err;

  // Sample K indices/values.
  int32_t* d_samp_idx_raw = nullptr;
  double* d_samp_val_raw = nullptr;
  err = guga_cuda_malloc(&d_samp_idx_raw, (size_t)k_samp * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_samp_val_raw, (size_t)k_samp * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_samp_idx(d_samp_idx_raw, CudaFreeStreamDeleter<int32_t>{stream});
  cuda_unique_ptr_stream<double> d_samp_val(d_samp_val_raw, CudaFreeStreamDeleter<double>{stream});

  {
    int blocks = (k_samp + threads - 1) / threads;
    qmc_systematic_resample_guided_kernel<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted + p, d_abs_sorted + p, d_w.get(), d_cdf.get(), r_len, k_samp, seed, d_samp_idx.get(), d_samp_val.get());
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  // Build (pivot + samples) buffer of length m, then coalesce by index.
  int32_t* d_idx_buf_raw = nullptr;
  double* d_val_buf_raw = nullptr;
  err = guga_cuda_malloc(&d_idx_buf_raw, (size_t)m * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_val_buf_raw, (size_t)m * sizeof(double), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_idx_buf(d_idx_buf_raw, CudaFreeStreamDeleter<int32_t>{stream});
  cuda_unique_ptr_stream<double> d_val_buf(d_val_buf_raw, CudaFreeStreamDeleter<double>{stream});

  {
    int blocks = (m + threads - 1) / threads;
    qmc_build_phi_buffer_kernel_q<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted, n_in, p, d_samp_idx.get(), d_samp_val.get(), k_samp, d_idx_buf.get(), d_val_buf.get());
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return guga_qmc_coalesce_coo_i32_f64_launch_stream(d_idx_buf.get(), d_val_buf.get(), m, idx_out, val_out, out_nnz, stream, threads);
}

extern "C" cudaError_t guga_qmc_phi_pivot_resample_guided_i32_f64_launch(
    const int32_t* idx_in,
    const double* val_in,
    const double* q_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    double alpha,
    double q_floor,
    uint64_t seed,
    int threads) {
  return guga_qmc_phi_pivot_resample_guided_i32_f64_launch_stream(
      idx_in, val_in, q_in, n_in, idx_out, val_out, out_nnz, m, pivot, alpha, q_floor, seed, /*stream=*/0, threads);
}


// -----------------------------------------------------------------------------
// QMC: persistent workspace (allocation-free steady state)
// -----------------------------------------------------------------------------

namespace {

inline void qmc_throw_on_cuda_error_ws(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct QmcWorkspace {
  int device = 0;
  int max_n = 0;
  int max_m = 0;

  // Coalesce scratch (length max_n).
  int32_t* d_idx_tmp = nullptr;
  int32_t* d_idx_alt = nullptr;
  double* d_val_tmp = nullptr;
  double* d_val_alt = nullptr;
  int* d_nnz_tmp = nullptr;

  // Φ scratch (length max_n or max_m).
  double* d_abs_a = nullptr;
  double* d_abs_b = nullptr;
  QmcIdxVal* d_pair_a = nullptr;
  QmcIdxVal* d_pair_b = nullptr;
  double* d_cdf = nullptr;
  int32_t* d_samp_idx = nullptr;
  double* d_samp_val = nullptr;
  int32_t* d_phi_idx = nullptr;
  double* d_phi_val = nullptr;

  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  QmcWorkspace(int max_n_, int max_m_) : max_n(max_n_), max_m(max_m_) {
    qmc_throw_on_cuda_error_ws(cudaGetDevice(&device), "cudaGetDevice");
    if (max_n_ <= 0) throw std::invalid_argument("max_n must be >= 1");
    if (max_m_ <= 0) throw std::invalid_argument("max_m must be >= 1");
    if (max_n < max_m) max_n = max_m;

    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_idx_tmp, (size_t)max_n * sizeof(int32_t)), "cudaMalloc(qmc idx_tmp)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_idx_alt, (size_t)max_n * sizeof(int32_t)), "cudaMalloc(qmc idx_alt)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_val_tmp, (size_t)max_n * sizeof(double)), "cudaMalloc(qmc val_tmp)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_val_alt, (size_t)max_n * sizeof(double)), "cudaMalloc(qmc val_alt)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_nnz_tmp, sizeof(int)), "cudaMalloc(qmc nnz_tmp)");

    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_abs_a, (size_t)max_n * sizeof(double)), "cudaMalloc(qmc abs_a)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_abs_b, (size_t)max_n * sizeof(double)), "cudaMalloc(qmc abs_b)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_pair_a, (size_t)max_n * sizeof(QmcIdxVal)), "cudaMalloc(qmc pair_a)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_pair_b, (size_t)max_n * sizeof(QmcIdxVal)), "cudaMalloc(qmc pair_b)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_cdf, (size_t)max_n * sizeof(double)), "cudaMalloc(qmc cdf)");

    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_samp_idx, (size_t)max_m * sizeof(int32_t)), "cudaMalloc(qmc samp_idx)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_samp_val, (size_t)max_m * sizeof(double)), "cudaMalloc(qmc samp_val)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_phi_idx, (size_t)max_m * sizeof(int32_t)), "cudaMalloc(qmc phi_idx)");
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_phi_val, (size_t)max_m * sizeof(double)), "cudaMalloc(qmc phi_val)");

    // Query CUB temp storage sizes for the maximum problem sizes.
    size_t tmp = 0;

    cub::DoubleBuffer<int32_t> keys_i32(d_idx_tmp, d_idx_alt);
    cub::DoubleBuffer<double> vals_f64(d_val_tmp, d_val_alt);
    qmc_throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys_i32, vals_f64, max_n, 0, 32, /*stream=*/0), "cub::SortPairs(i32) query");
    temp_bytes = std::max(temp_bytes, tmp);

    qmc_throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(
            nullptr, tmp, d_idx_tmp, d_idx_alt, d_val_tmp, d_val_alt, d_nnz_tmp, QmcF64Sum(), max_n, /*stream=*/0),
        "cub::ReduceByKey(i32) query");
    temp_bytes = std::max(temp_bytes, tmp);

    cub::DoubleBuffer<double> keys_abs(d_abs_a, d_abs_b);
    cub::DoubleBuffer<QmcIdxVal> vals_abs(d_pair_a, d_pair_b);
    qmc_throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp, keys_abs, vals_abs, max_n, 0, 64, /*stream=*/0),
        "cub::SortPairsDescending(abs) query");
    temp_bytes = std::max(temp_bytes, tmp);

    qmc_throw_on_cuda_error_ws(
        cub::DeviceScan::InclusiveSum(nullptr, tmp, d_abs_a, d_cdf, max_n, /*stream=*/0), "cub::InclusiveSum(abs) query");
    temp_bytes = std::max(temp_bytes, tmp);

    if (temp_bytes == 0) temp_bytes = 1;
    qmc_throw_on_cuda_error_ws(cudaMalloc((void**)&d_temp, temp_bytes), "cudaMalloc(qmc temp)");
  }

  ~QmcWorkspace() { release(); }

  QmcWorkspace(const QmcWorkspace&) = delete;
  QmcWorkspace& operator=(const QmcWorkspace&) = delete;

  void release() noexcept {
    if (d_idx_tmp) cudaFree(d_idx_tmp);
    if (d_idx_alt) cudaFree(d_idx_alt);
    if (d_val_tmp) cudaFree(d_val_tmp);
    if (d_val_alt) cudaFree(d_val_alt);
    if (d_nnz_tmp) cudaFree(d_nnz_tmp);

    if (d_abs_a) cudaFree(d_abs_a);
    if (d_abs_b) cudaFree(d_abs_b);
    if (d_pair_a) cudaFree(d_pair_a);
    if (d_pair_b) cudaFree(d_pair_b);
    if (d_cdf) cudaFree(d_cdf);
    if (d_samp_idx) cudaFree(d_samp_idx);
    if (d_samp_val) cudaFree(d_samp_val);
    if (d_phi_idx) cudaFree(d_phi_idx);
    if (d_phi_val) cudaFree(d_phi_val);

    if (d_temp) cudaFree(d_temp);

    d_idx_tmp = nullptr;
    d_idx_alt = nullptr;
    d_val_tmp = nullptr;
    d_val_alt = nullptr;
    d_nnz_tmp = nullptr;
    d_abs_a = nullptr;
    d_abs_b = nullptr;
    d_pair_a = nullptr;
    d_pair_b = nullptr;
    d_cdf = nullptr;
    d_samp_idx = nullptr;
    d_samp_val = nullptr;
    d_phi_idx = nullptr;
    d_phi_val = nullptr;
    d_temp = nullptr;
    temp_bytes = 0;
  }
};

inline QmcWorkspace* qmc_ws_from_handle(void* ws_handle) {
  if (!ws_handle) throw std::invalid_argument("QmcWorkspace handle is null");
  return reinterpret_cast<QmcWorkspace*>(ws_handle);
}

inline void qmc_ensure_ws_device(QmcWorkspace* ws) {
  int dev = 0;
  qmc_throw_on_cuda_error_ws(cudaGetDevice(&dev), "cudaGetDevice");
  if (dev != ws->device) {
    throw std::runtime_error("QmcWorkspace was created on a different CUDA device");
  }
}

}  // namespace

extern "C" void* guga_qmc_workspace_create(int max_n, int max_m) {
  return reinterpret_cast<void*>(new QmcWorkspace(int(max_n), int(max_m)));
}

extern "C" void guga_qmc_workspace_destroy(void* ws_handle) {
  auto* ws = reinterpret_cast<QmcWorkspace*>(ws_handle);
  delete ws;
}

extern "C" cudaError_t guga_qmc_coalesce_coo_i32_f64_ws_launch_stream(
    void* ws_handle,
    const int32_t* idx_in,
    const double* val_in,
    int n,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    cudaStream_t stream,
    int threads) {
  QmcWorkspace* ws = qmc_ws_from_handle(ws_handle);
  qmc_ensure_ws_device(ws);
  if (!idx_in || !val_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n < 0 || n > ws->max_n) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  cudaError_t err = cudaMemcpyAsync(ws->d_idx_tmp, idx_in, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) return err;
  err = cudaMemcpyAsync(ws->d_val_tmp, val_in, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) return err;

  {
    int blocks = (n + threads - 1) / threads;
    qmc_sanitize_invalid_idx_kernel<<<blocks, threads, 0, stream>>>(ws->d_idx_tmp, ws->d_val_tmp, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<int32_t> keys(ws->d_idx_tmp, ws->d_idx_alt);
  cub::DoubleBuffer<double> vals(ws->d_val_tmp, ws->d_val_alt);

  err = cub::DeviceRadixSort::SortPairs(ws->d_temp, ws->temp_bytes, keys, vals, n, 0, 32, stream);
  if (err != cudaSuccess) return err;

  err = cub::DeviceReduce::ReduceByKey(
      ws->d_temp, ws->temp_bytes, keys.Current(), idx_out, vals.Current(), val_out, out_nnz, QmcF64Sum(), n, stream);
  if (err != cudaSuccess) return err;

  {
    int blocks = (n + threads - 1) / threads;
    qmc_strip_invalid_prefix_kernel<<<blocks, threads, 0, stream>>>(idx_out, val_out, out_nnz);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return cudaSuccess;
}

extern "C" cudaError_t guga_qmc_phi_pivot_resample_i32_f64_ws_launch_stream(
    void* ws_handle,
    const int32_t* idx_in,
    const double* val_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    uint64_t seed,
    cudaStream_t stream,
    int threads) {
  QmcWorkspace* ws = qmc_ws_from_handle(ws_handle);
  qmc_ensure_ws_device(ws);
  if (!idx_in || !val_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n_in < 0 || n_in > ws->max_n) return cudaErrorInvalidValue;
  if (m < 0 || m > ws->max_m) return cudaErrorInvalidValue;
  if (pivot < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n_in == 0 || m == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  if (n_in <= m) {
    cudaError_t err = cudaMemcpyAsync(idx_out, idx_in, (size_t)n_in * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(val_out, val_in, (size_t)n_in * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out_nnz, &n_in, sizeof(int), cudaMemcpyHostToDevice, stream);
    return err;
  }

  int p = pivot;
  if (p > m) p = m;
  if (p > n_in) p = n_in;
  int k_samp = m - p;
  int r_len = n_in - p;

  // Sort by |val| descending.
  {
    int blocks = (n_in + threads - 1) / threads;
    qmc_pack_abs_and_pairs_kernel<<<blocks, threads, 0, stream>>>(idx_in, val_in, ws->d_abs_a, ws->d_pair_a, n_in);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<double> keys_abs(ws->d_abs_a, ws->d_abs_b);
  cub::DoubleBuffer<QmcIdxVal> vals_abs(ws->d_pair_a, ws->d_pair_b);

  cudaError_t err = cub::DeviceRadixSort::SortPairsDescending(ws->d_temp, ws->temp_bytes, keys_abs, vals_abs, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;

  const QmcIdxVal* d_pairs_sorted = vals_abs.Current();
  const double* d_abs_sorted = keys_abs.Current();

  if (k_samp > 0 && r_len > 0) {
    err = cub::DeviceScan::InclusiveSum(ws->d_temp, ws->temp_bytes, d_abs_sorted + p, ws->d_cdf, r_len, stream);
    if (err != cudaSuccess) return err;

    int blocks = (k_samp + threads - 1) / threads;
    qmc_systematic_resample_kernel<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted + p, ws->d_cdf, r_len, k_samp, seed, ws->d_samp_idx, ws->d_samp_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    blocks = (m + threads - 1) / threads;
    qmc_build_phi_buffer_kernel<<<blocks, threads, 0, stream>>>(
        d_pairs_sorted, n_in, p, ws->d_samp_idx, ws->d_samp_val, k_samp, ws->d_phi_idx, ws->d_phi_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    int blocks = (p + threads - 1) / threads;
    qmc_build_phi_buffer_kernel<<<blocks, threads, 0, stream>>>(d_pairs_sorted, n_in, p, nullptr, nullptr, 0, ws->d_phi_idx, ws->d_phi_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    m = p;
  }

  return guga_qmc_coalesce_coo_i32_f64_ws_launch_stream(ws_handle, ws->d_phi_idx, ws->d_phi_val, m, idx_out, val_out, out_nnz, stream, threads);
}


// -----------------------------------------------------------------------------
// QMC: guided Φ compression with persistent workspace
// -----------------------------------------------------------------------------

namespace {

__global__ void qmc_pack_abs_only_kernel(const double* __restrict__ val, double* __restrict__ abs_out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  abs_out[i] = fabs(val[i]);
}

__global__ void qmc_fill_iota_i32_kernel(int32_t* __restrict__ out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = (int32_t)i;
}

__global__ void qmc_compute_guided_weights_from_pos_kernel(
    const double* __restrict__ abs_r,
    const int32_t* __restrict__ pos_r,
    const double* __restrict__ q_in,
    int r_len,
    double alpha,
    double q_floor,
    double* __restrict__ w_out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= r_len) return;
  double a = abs_r[i];
  int32_t pos = pos_r[i];
  double qv = q_in[pos];
  if (qv < q_floor) qv = q_floor;

  double w = 0.0;
  if (alpha <= 0.0) {
    w = a;
  } else if (alpha >= 1.0) {
    w = qv;
  } else {
    if (a > 0.0 && qv > 0.0) {
      w = pow(a, 1.0 - alpha) * pow(qv, alpha);
    } else {
      w = 0.0;
    }
  }
  w_out[i] = w;
}

__global__ void qmc_systematic_resample_guided_from_pos_kernel(
    const int32_t* __restrict__ pos_r,
    const double* __restrict__ abs_r,
    const double* __restrict__ w_r,
    const double* __restrict__ cdf_r,
    int r_len,
    int k_samp,
    uint64_t seed,
    const int32_t* __restrict__ idx_in,
    const double* __restrict__ val_in,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= k_samp) return;

  if (r_len <= 0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double W = cdf_r[r_len - 1];
  if (W <= 0.0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  double step = W / (double)k_samp;
  uint64_t st = seed;
  double u0 = rand_u01(&st) * step;
  double u = u0 + (double)k * step;

  int posi = lower_bound_cdf_f64(cdf_r, r_len, u);
  if (posi < 0) posi = 0;
  if (posi >= r_len) posi = r_len - 1;

  int32_t pos = pos_r[posi];
  int32_t outk = idx_in[pos];
  double v = val_in[pos];
  double a = abs_r[posi];
  double w = w_r[posi];

  if (w <= 0.0 || outk < 0) {
    out_idx[k] = (int32_t)-1;
    out_val[k] = 0.0;
    return;
  }

  out_idx[k] = outk;
  out_val[k] = (v >= 0.0 ? 1.0 : -1.0) * step * (a / w);
}

__global__ void qmc_build_phi_buffer_gather_kernel(
    const int32_t* __restrict__ pos_sorted,
    const int32_t* __restrict__ idx_in,
    const double* __restrict__ val_in,
    int p_piv,
    const int32_t* __restrict__ samp_idx,
    const double* __restrict__ samp_val,
    int k_samp,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int n_out = p_piv + k_samp;
  if (i >= n_out) return;
  if (i < p_piv) {
    int32_t pos = pos_sorted[i];
    out_idx[i] = idx_in[pos];
    out_val[i] = val_in[pos];
  } else {
    int k = i - p_piv;
    out_idx[i] = samp_idx[k];
    out_val[i] = samp_val[k];
  }
}

}  // namespace

extern "C" cudaError_t guga_qmc_phi_pivot_resample_guided_i32_f64_ws_launch_stream(
    void* ws_handle,
    const int32_t* idx_in,
    const double* val_in,
    const double* q_in,
    int n_in,
    int32_t* idx_out,
    double* val_out,
    int* out_nnz,
    int m,
    int pivot,
    double alpha,
    double q_floor,
    uint64_t seed,
    cudaStream_t stream,
    int threads) {
  QmcWorkspace* ws = qmc_ws_from_handle(ws_handle);
  qmc_ensure_ws_device(ws);
  if (!idx_in || !val_in || !q_in || !idx_out || !val_out || !out_nnz) return cudaErrorInvalidValue;
  if (n_in < 0 || n_in > ws->max_n) return cudaErrorInvalidValue;
  if (m < 0 || m > ws->max_m) return cudaErrorInvalidValue;
  if (pivot < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  if (n_in == 0 || m == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }

  if (n_in <= m) {
    cudaError_t err = cudaMemcpyAsync(idx_out, idx_in, (size_t)n_in * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(val_out, val_in, (size_t)n_in * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out_nnz, &n_in, sizeof(int), cudaMemcpyHostToDevice, stream);
    return err;
  }

  int p = pivot;
  if (p > m) p = m;
  if (p > n_in) p = n_in;
  int k_samp = m - p;
  int r_len = n_in - p;

  // Compute |val| and initialize positions 0..n_in-1
  {
    int blocks = (n_in + threads - 1) / threads;
    qmc_pack_abs_only_kernel<<<blocks, threads, 0, stream>>>(val_in, ws->d_abs_a, n_in);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    int32_t* d_pos_a = reinterpret_cast<int32_t*>(ws->d_pair_a);
    qmc_fill_iota_i32_kernel<<<blocks, threads, 0, stream>>>(d_pos_a, n_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  cub::DoubleBuffer<double> keys_abs(ws->d_abs_a, ws->d_abs_b);
  int32_t* d_pos_a = reinterpret_cast<int32_t*>(ws->d_pair_a);
  int32_t* d_pos_b = reinterpret_cast<int32_t*>(ws->d_pair_b);
  cub::DoubleBuffer<int32_t> vals_pos(d_pos_a, d_pos_b);

  cudaError_t err = cub::DeviceRadixSort::SortPairsDescending(ws->d_temp, ws->temp_bytes, keys_abs, vals_pos, n_in, 0, 64, stream);
  if (err != cudaSuccess) return err;

  const double* d_abs_sorted = keys_abs.Current();
  const int32_t* d_pos_sorted = vals_pos.Current();

  // If we have no sampling budget, only gather pivots.
  if (k_samp <= 0 || r_len <= 0) {
    int blocks = (p + threads - 1) / threads;
    qmc_build_phi_buffer_gather_kernel<<<blocks, threads, 0, stream>>>(d_pos_sorted, idx_in, val_in, p, nullptr, nullptr, 0, ws->d_phi_idx, ws->d_phi_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    m = p;
    return guga_qmc_coalesce_coo_i32_f64_ws_launch_stream(ws_handle, ws->d_phi_idx, ws->d_phi_val, m, idx_out, val_out, out_nnz, stream, threads);
  }

  // Compute guided weights for remainder into the alternate abs buffer (safe scratch).
  double* d_w = keys_abs.Alternate();
  {
    int blocks = (r_len + threads - 1) / threads;
    qmc_compute_guided_weights_from_pos_kernel<<<blocks, threads, 0, stream>>>(
        d_abs_sorted + p, d_pos_sorted + p, q_in, r_len, alpha, q_floor, d_w);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  err = cub::DeviceScan::InclusiveSum(ws->d_temp, ws->temp_bytes, d_w, ws->d_cdf, r_len, stream);
  if (err != cudaSuccess) return err;

  {
    int blocks = (k_samp + threads - 1) / threads;
    qmc_systematic_resample_guided_from_pos_kernel<<<blocks, threads, 0, stream>>>(
        d_pos_sorted + p, d_abs_sorted + p, d_w, ws->d_cdf, r_len, k_samp, seed, idx_in, val_in, ws->d_samp_idx, ws->d_samp_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  {
    int blocks = (m + threads - 1) / threads;
    qmc_build_phi_buffer_gather_kernel<<<blocks, threads, 0, stream>>>(
        d_pos_sorted, idx_in, val_in, p, ws->d_samp_idx, ws->d_samp_val, k_samp, ws->d_phi_idx, ws->d_phi_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return guga_qmc_coalesce_coo_i32_f64_ws_launch_stream(ws_handle, ws->d_phi_idx, ws->d_phi_val, m, idx_out, val_out, out_nnz, stream, threads);
}
