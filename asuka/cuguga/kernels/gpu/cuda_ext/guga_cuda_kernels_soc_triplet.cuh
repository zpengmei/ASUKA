#pragma once

namespace {

constexpr int SOC_TRIPLET_MAX_NORB = 64;

__device__ __forceinline__ int soc_triplet_step_to_occ(int8_t step) {
  if (step == 0) return 0;
  if (step == 3) return 2;
  return 1;
}

__device__ __forceinline__ int soc_triplet_tri_ok_twos(int tj1, int tj2, int tj3) {
  if (tj1 < 0 || tj2 < 0 || tj3 < 0) return 0;
  if ((tj1 + tj2 + tj3) & 1) return 0;
  int lo = tj1 - tj2;
  if (lo < 0) lo = -lo;
  int hi = tj1 + tj2;
  if (tj3 < lo || tj3 > hi) return 0;
  return 1;
}

__device__ __forceinline__ double soc_triplet_phase_from_twos_sum(int twos_sum) {
  if (twos_sum & 1) return 0.0;
  int half = twos_sum / 2;
  return (half & 1) ? -1.0 : 1.0;
}

__device__ __forceinline__ float soc_triplet_sixj_211_lookup(
    const float* sixj_211,
    int twos_max,
    int d,
    int e,
    int f) {
  if ((unsigned)d > (unsigned)twos_max) return 0.0f;
  if ((unsigned)e > (unsigned)twos_max) return 0.0f;
  if ((unsigned)f > (unsigned)twos_max) return 0.0f;
  int n = twos_max + 1;
  int idx = (d * n + e) * n + f;
  return sixj_211[idx];
}

__device__ __forceinline__ float soc_triplet_t_factor_lookup(
    const float* t_factor,
    int twos_max,
    int twos_opline,
    int skm1,
    int sk,
    int skm1_p,
    int sk_p) {
  int op_idx = (twos_opline == 1) ? 0 : ((twos_opline == 2) ? 1 : -1);
  if (op_idx < 0) return 0.0f;
  int dk = sk - skm1;
  int db = sk_p - skm1_p;
  int dk_idx = (dk == +1) ? 0 : ((dk == -1) ? 1 : -1);
  int db_idx = (db == +1) ? 0 : ((db == -1) ? 1 : -1);
  if (dk_idx < 0 || db_idx < 0) return 0.0f;
  if ((unsigned)skm1 > (unsigned)twos_max) return 0.0f;
  if ((unsigned)sk > (unsigned)twos_max) return 0.0f;
  if ((unsigned)skm1_p > (unsigned)twos_max) return 0.0f;
  if ((unsigned)sk_p > (unsigned)twos_max) return 0.0f;

  int n = twos_max + 1;
  int idx = ((((op_idx * n + skm1) * n + skm1_p) * 2 + dk_idx) * 2 + db_idx);
  return t_factor[idx];
}

constexpr int SOC_TRIPLET_OCC_INVALID = 0;
constexpr int SOC_TRIPLET_OCC_SOMO = 1;
constexpr int SOC_TRIPLET_OCC_DOMO = 2;

__device__ __forceinline__ int soc_triplet_occ_case_from_occ(int occ) {
  if (occ == 1) return SOC_TRIPLET_OCC_SOMO;
  if (occ == 2) return SOC_TRIPLET_OCC_DOMO;
  return SOC_TRIPLET_OCC_INVALID;
}

__device__ __forceinline__ double soc_triplet_A_factor_dev(
    int twos_km1,
    int twos_k,
    int twos_km1_p,
    int twos_k_p,
    int occ_case) {
  if (occ_case == SOC_TRIPLET_OCC_DOMO) {
    if (twos_k != twos_km1) return 0.0;
    return sqrt((double)twos_k_p + 1.0);
  }
  if (occ_case == SOC_TRIPLET_OCC_SOMO) {
    if (twos_k_p != twos_km1_p) return 0.0;
    double ph = soc_triplet_phase_from_twos_sum(twos_km1_p - twos_k - 1);
    if (ph == 0.0) return 0.0;
    return ph * sqrt((double)twos_k + 1.0);
  }
  return 0.0;
}

__device__ __forceinline__ double soc_triplet_B_factor_dev(
    int twos_km1,
    int twos_k,
    int twos_km1_p,
    int twos_k_p,
    int occ_case) {
  if (occ_case == SOC_TRIPLET_OCC_SOMO) {
    if (twos_k != twos_km1) return 0.0;
    return sqrt((double)twos_k_p + 1.0);
  }
  if (occ_case == SOC_TRIPLET_OCC_DOMO) {
    if (twos_k_p != twos_km1_p) return 0.0;
    double ph = soc_triplet_phase_from_twos_sum(twos_km1_p - twos_k - 1);
    if (ph == 0.0) return 0.0;
    return ph * sqrt((double)twos_k + 1.0);
  }
  return 0.0;
}

__device__ __forceinline__ double soc_triplet_Atilde_factor_dev(
    const float* sixj_211,
    int twos_max,
    int twos_km1,
    int twos_k,
    int twos_km1_p,
    int twos_k_p,
    int occ_case) {
  constexpr int twos_opline = 2;  // k=1 for triplet RMEs
  if (occ_case == SOC_TRIPLET_OCC_DOMO) {
    if (twos_k != twos_km1) return 0.0;
    double ph = soc_triplet_phase_from_twos_sum(twos_k_p + twos_k - twos_opline - 2);
    if (ph == 0.0) return 0.0;
    float sixj = soc_triplet_sixj_211_lookup(sixj_211, twos_max, twos_km1_p, twos_k, twos_k_p);
    return ph * sqrt((double)twos_k_p + 1.0) * (double)sixj;
  }
  if (occ_case == SOC_TRIPLET_OCC_SOMO) {
    if (twos_k_p != twos_km1_p) return 0.0;
    double ph = soc_triplet_phase_from_twos_sum(twos_km1_p + twos_km1 - twos_opline + 1);
    if (ph == 0.0) return 0.0;
    float sixj = soc_triplet_sixj_211_lookup(sixj_211, twos_max, twos_km1, twos_km1_p, twos_k);
    return ph * sqrt((double)twos_k + 1.0) * (double)sixj;
  }
  return 0.0;
}

__device__ __forceinline__ double soc_triplet_Btilde_factor_dev(
    const float* sixj_211,
    int twos_max,
    int twos_km1,
    int twos_k,
    int twos_km1_p,
    int twos_k_p,
    int occ_case) {
  int mapped = SOC_TRIPLET_OCC_INVALID;
  if (occ_case == SOC_TRIPLET_OCC_SOMO) mapped = SOC_TRIPLET_OCC_DOMO;
  if (occ_case == SOC_TRIPLET_OCC_DOMO) mapped = SOC_TRIPLET_OCC_SOMO;
  return soc_triplet_Atilde_factor_dev(sixj_211, twos_max, twos_km1, twos_k, twos_km1_p, twos_k_p, mapped);
}

__device__ __forceinline__ int soc_triplet_prefix_twos(
    const int16_t* node_twos,
    const int32_t* nodes,
    int t) {
  return (int)node_twos[nodes[t]];
}

__device__ __forceinline__ double soc_triplet_rme_single_excitation_dev_general(
    const int16_t* node_twos_bra,
    const int16_t* node_twos_ket,
    const int8_t* bra_steps,
    const int8_t* ket_steps,
    const int32_t* bra_nodes,
    const int32_t* ket_nodes,
    int norb,
    int p,
    int q,
    const float* sixj_211,
    const float* t_factor,
    int twos_max) {
  if (p == q) return 0.0;

  int twos_bra_total = (int)node_twos_bra[bra_nodes[norb]];
  int twos_ket_total = (int)node_twos_ket[ket_nodes[norb]];
  if (!soc_triplet_tri_ok_twos(twos_bra_total, 2, twos_ket_total)) return 0.0;

  int occ_ket_p = soc_triplet_step_to_occ(ket_steps[p]);
  int occ_ket_q = soc_triplet_step_to_occ(ket_steps[q]);
  int occ_bra_p = soc_triplet_step_to_occ(bra_steps[p]);
  int occ_bra_q = soc_triplet_step_to_occ(bra_steps[q]);

  if (occ_bra_p != occ_ket_p + 1) return 0.0;
  if (occ_ket_q != occ_bra_q + 1) return 0.0;

  int diff_count = 0;
  int d0 = -1;
  int d1 = -1;
  for (int t = 0; t < norb; ++t) {
    int ok = soc_triplet_step_to_occ(ket_steps[t]);
    int ob = soc_triplet_step_to_occ(bra_steps[t]);
    if (ok != ob) {
      if (diff_count == 0) d0 = t;
      else if (diff_count == 1) d1 = t;
      diff_count++;
      if (diff_count > 2) return 0.0;
      int delta = ok - ob;
      if (delta < 0) delta = -delta;
      if (delta != 1) return 0.0;
    }
  }
  if (diff_count != 2) return 0.0;
  int lo = (p < q) ? p : q;
  int hi = (p < q) ? q : p;
  if (!(d0 == lo && d1 == hi)) return 0.0;

  double coef = 1.2247448713915890491 / sqrt((double)twos_bra_total + 1.0);

  int n_between = 0;
  for (int t = lo + 1; t < hi; ++t) {
    n_between += soc_triplet_step_to_occ(ket_steps[t]);
  }
  if (p < q) {
    if (occ_bra_p == 2) n_between += 1;
    if (occ_ket_q == 2) n_between += 1;
  }
  if (n_between & 1) coef = -coef;

  for (int t = 1; t < lo; ++t) {
    int tw_k = soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t);
    int tw_b = soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t);
    if (tw_k != tw_b) return 0.0;
  }

  if (p < q) {
    int p_case = soc_triplet_occ_case_from_occ(occ_bra_p);
    if (p_case == SOC_TRIPLET_OCC_INVALID) return 0.0;
    coef *= soc_triplet_B_factor_dev(
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, p),
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, p + 1),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, p),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, p + 1),
        p_case);
    if (coef == 0.0) return 0.0;

    for (int t = p + 1; t < q; ++t) {
      if (soc_triplet_step_to_occ(ket_steps[t]) != 1) continue;
      float tf = soc_triplet_t_factor_lookup(
          t_factor,
          twos_max,
          1,
          soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t),
          soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t + 1),
          soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t),
          soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t + 1));
      if (tf == 0.0f) return 0.0;
      coef *= (double)tf;
    }

    int q_case = soc_triplet_occ_case_from_occ(occ_ket_q);
    if (q_case == SOC_TRIPLET_OCC_INVALID) return 0.0;
    coef *= soc_triplet_Atilde_factor_dev(
        sixj_211,
        twos_max,
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, q),
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, q + 1),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, q),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, q + 1),
        q_case);
    if (coef == 0.0) return 0.0;

    for (int t = q + 1; t < norb; ++t) {
      if (soc_triplet_step_to_occ(ket_steps[t]) != 1) continue;
      float tf = soc_triplet_t_factor_lookup(
          t_factor,
          twos_max,
          2,
          soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t),
          soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t + 1),
          soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t),
          soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t + 1));
      if (tf == 0.0f) return 0.0;
      coef *= (double)tf;
    }
    return coef;
  }

  int q_case = soc_triplet_occ_case_from_occ(occ_ket_q);
  if (q_case == SOC_TRIPLET_OCC_INVALID) return 0.0;
  coef *= soc_triplet_A_factor_dev(
      soc_triplet_prefix_twos(node_twos_ket, ket_nodes, q),
      soc_triplet_prefix_twos(node_twos_ket, ket_nodes, q + 1),
      soc_triplet_prefix_twos(node_twos_bra, bra_nodes, q),
      soc_triplet_prefix_twos(node_twos_bra, bra_nodes, q + 1),
      q_case);
  if (coef == 0.0) return 0.0;

  for (int t = q + 1; t < p; ++t) {
    if (soc_triplet_step_to_occ(ket_steps[t]) != 1) continue;
    float tf = soc_triplet_t_factor_lookup(
        t_factor,
        twos_max,
        1,
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t),
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t + 1),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t + 1));
    if (tf == 0.0f) return 0.0;
    coef *= (double)tf;
  }

  int p_case = soc_triplet_occ_case_from_occ(occ_bra_p);
  if (p_case == SOC_TRIPLET_OCC_INVALID) return 0.0;
  coef *= soc_triplet_Btilde_factor_dev(
      sixj_211,
      twos_max,
      soc_triplet_prefix_twos(node_twos_ket, ket_nodes, p),
      soc_triplet_prefix_twos(node_twos_ket, ket_nodes, p + 1),
      soc_triplet_prefix_twos(node_twos_bra, bra_nodes, p),
      soc_triplet_prefix_twos(node_twos_bra, bra_nodes, p + 1),
      p_case);
  if (coef == 0.0) return 0.0;

  for (int t = p + 1; t < norb; ++t) {
    if (soc_triplet_step_to_occ(ket_steps[t]) != 1) continue;
    float tf = soc_triplet_t_factor_lookup(
        t_factor,
        twos_max,
        2,
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t),
        soc_triplet_prefix_twos(node_twos_ket, ket_nodes, t + 1),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t),
        soc_triplet_prefix_twos(node_twos_bra, bra_nodes, t + 1));
    if (tf == 0.0f) return 0.0;
    coef *= (double)tf;
  }
  return coef;
}

__global__ void guga_triplet_apply_contracted_epq_table_kernel(
    const int16_t* __restrict__ node_twos,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    const int64_t* __restrict__ epq_indptr,
    const int32_t* __restrict__ epq_indices,
    const int32_t* __restrict__ epq_pq,
    const double* __restrict__ x,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ y_re,
    double* __restrict__ y_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf) return;

  double xj = x[j];
  if (xj == 0.0) return;

  const int8_t* ket_steps = &steps_table[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table[(int64_t)j * (int64_t)(norb + 1)];

  int64_t start = epq_indptr[j];
  int64_t end = epq_indptr[j + 1];
  int64_t h_stride = (int64_t)norb * (int64_t)norb;

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int i = epq_indices[t];
    if ((unsigned)i >= (unsigned)ncsf) continue;
    int pq = epq_pq[t];
    int p = pq / norb;
    int q = pq - p * norb;
    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || p == q) continue;

    const int8_t* bra_steps = &steps_table[(int64_t)i * (int64_t)norb];
    const int32_t* bra_nodes = &nodes_table[(int64_t)i * (int64_t)(norb + 1)];

    double cij = soc_triplet_rme_single_excitation_dev_general(
        node_twos,
        node_twos,
        bra_steps,
        ket_steps,
        bra_nodes,
        ket_nodes,
        norb,
        p,
        q,
        sixj_211,
        t_factor,
        twos_max);
    if (cij == 0.0) continue;

    double scale = xj * cij;
    int64_t h_idx = (int64_t)p * (int64_t)norb + (int64_t)q;
    for (int m = 0; m < 3; ++m) {
      int64_t hm_idx = (int64_t)m * h_stride + h_idx;
      double hr = h_re[hm_idx];
      double hi = h_im[hm_idx];
      if (hr != 0.0) atomicAdd(&y_re[(int64_t)m * (int64_t)ncsf + (int64_t)i], scale * hr);
      if (hi != 0.0) atomicAdd(&y_im[(int64_t)m * (int64_t)ncsf + (int64_t)i], scale * hi);
    }
  }
}

__global__ void guga_triplet_apply_contracted_dfs_kernel(
    const int32_t* __restrict__ child_bra,
    const int16_t* __restrict__ node_twos_bra,
    const int64_t* __restrict__ child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* __restrict__ node_twos_ket,
    const int8_t* __restrict__ steps_table_ket,
    const int32_t* __restrict__ nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* __restrict__ x,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ y_re,
    double* __restrict__ y_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf_ket) return;
  if (!soc_triplet_tri_ok_twos(twos_bra_total, 2, twos_ket_total)) return;

  double xj = x[j];
  if (xj == 0.0) return;

  const int8_t* ket_steps = &steps_table_ket[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table_ket[(int64_t)j * (int64_t)(norb + 1)];
  int64_t h_stride = (int64_t)norb * (int64_t)norb;
  int nops = norb * norb;

  for (int pq_lin = (int)threadIdx.x; pq_lin < nops; pq_lin += (int)blockDim.x) {
    int p = pq_lin / norb;
    int q = pq_lin - p * norb;
    if (p == q) continue;

    int occ_p = soc_triplet_step_to_occ(ket_steps[p]);
    int occ_q = soc_triplet_step_to_occ(ket_steps[q]);
    if (occ_q <= 0 || occ_p >= 2) continue;

    int occ_p_target = occ_p + 1;
    int occ_q_target = occ_q - 1;
    if (occ_p_target > 2 || occ_q_target < 0) continue;

    int lo = (p < q) ? p : q;
    int hi = (p < q) ? q : p;

    int8_t bra_steps_local[SOC_TRIPLET_MAX_NORB];
    int32_t bra_nodes_local[SOC_TRIPLET_MAX_NORB + 1];
    for (int t = 0; t < norb; ++t) bra_steps_local[t] = ket_steps[t];
    bra_nodes_local[0] = root_bra;

    int32_t node0 = root_bra;
    int32_t idx0 = 0;
    bool prefix_ok = true;
    for (int t = 0; t < lo; ++t) {
      int step = (int)ket_steps[t];
      int32_t node1 = child_bra[(int64_t)node0 * 4 + step];
      if (node1 < 0) {
        prefix_ok = false;
        break;
      }
      idx0 += (int32_t)child_prefix_bra[(int64_t)node0 * 5 + step];
      node0 = node1;
      bra_nodes_local[t + 1] = node0;
    }
    if (!prefix_ok) continue;

    int stack_t[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_node[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_idx[SOC_TRIPLET_MAX_NORB + 2];
    int8_t stack_choice[SOC_TRIPLET_MAX_NORB + 2];

    int top = 0;
    stack_t[0] = lo;
    stack_node[0] = node0;
    stack_idx[0] = idx0;
    stack_choice[0] = 0;

    while (top >= 0) {
      int t = stack_t[top];
      if (t == hi + 1) {
        int32_t node_s = stack_node[top];
        int32_t idx_s = stack_idx[top];
        bool suffix_ok = true;
        for (int u = hi + 1; u < norb; ++u) {
          int step_u = (int)ket_steps[u];
          int32_t node1 = child_bra[(int64_t)node_s * 4 + step_u];
          if (node1 < 0) {
            suffix_ok = false;
            break;
          }
          idx_s += (int32_t)child_prefix_bra[(int64_t)node_s * 5 + step_u];
          node_s = node1;
          bra_nodes_local[u + 1] = node_s;
        }
        if (suffix_ok && node_s == leaf_bra && idx_s >= 0 && idx_s < ncsf_bra) {
          double cij = soc_triplet_rme_single_excitation_dev_general(
              node_twos_bra,
              node_twos_ket,
              bra_steps_local,
              ket_steps,
              bra_nodes_local,
              ket_nodes,
              norb,
              p,
              q,
              sixj_211,
              t_factor,
              twos_max);
          if (cij != 0.0) {
            double scale = xj * cij;
            int64_t h_idx = (int64_t)p * (int64_t)norb + (int64_t)q;
            for (int m = 0; m < 3; ++m) {
              int64_t hm_idx = (int64_t)m * h_stride + h_idx;
              double hr = h_re[hm_idx];
              double hi = h_im[hm_idx];
              if (hr != 0.0) atomicAdd(&y_re[(int64_t)m * (int64_t)ncsf_bra + (int64_t)idx_s], scale * hr);
              if (hi != 0.0) atomicAdd(&y_im[(int64_t)m * (int64_t)ncsf_bra + (int64_t)idx_s], scale * hi);
            }
          }
        }
        top--;
        continue;
      }

      int occ_target = soc_triplet_step_to_occ(ket_steps[t]);
      if (t == p) occ_target = occ_p_target;
      else if (t == q) occ_target = occ_q_target;

      int opt0 = -1;
      int opt1 = -1;
      int nopt = 0;
      if (occ_target == 0) {
        opt0 = 0;
        nopt = 1;
      } else if (occ_target == 2) {
        opt0 = 3;
        nopt = 1;
      } else if (occ_target == 1) {
        opt0 = 1;
        opt1 = 2;
        nopt = 2;
      } else {
        top--;
        continue;
      }

      int choice = (int)stack_choice[top];
      if (choice >= nopt) {
        top--;
        continue;
      }

      int step = (choice == 0) ? opt0 : opt1;
      stack_choice[top] = (int8_t)(choice + 1);

      int32_t node = stack_node[top];
      int32_t node1 = child_bra[(int64_t)node * 4 + step];
      if (node1 < 0) continue;

      bra_steps_local[t] = (int8_t)step;
      bra_nodes_local[t + 1] = node1;
      int32_t idx1 = stack_idx[top] + (int32_t)child_prefix_bra[(int64_t)node * 5 + step];

      top++;
      stack_t[top] = t + 1;
      stack_node[top] = node1;
      stack_idx[top] = idx1;
      stack_choice[top] = 0;
    }
  }
}

__device__ __forceinline__ void soc_triplet_accumulate_rho_edge_dev(
    int i,
    int j,
    int p,
    int q,
    int norb,
    double cij,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* eta_re,
    const double* eta_im,
    double* rho_re,
    double* rho_im) {
  if (cij == 0.0) return;
  const double* cbi = c_bra + (int64_t)i * (int64_t)nb;
  const double* ckj = c_ket + (int64_t)j * (int64_t)nk;
  int64_t pq_idx = (int64_t)p * (int64_t)norb + (int64_t)q;
  int64_t eta_m_stride = (int64_t)nb * (int64_t)nk;
  int64_t rho_m_stride = (int64_t)norb * (int64_t)norb;

  for (int m = 0; m < 3; ++m) {
    double s_re = 0.0;
    double s_im = 0.0;
    int64_t eta_base = (int64_t)m * eta_m_stride;
    for (int b = 0; b < nb; ++b) {
      double cb = cbi[b];
      if (cb == 0.0) continue;
      int64_t eta_row = eta_base + (int64_t)b * (int64_t)nk;
      for (int k = 0; k < nk; ++k) {
        double coeff = cb * ckj[k];
        if (coeff == 0.0) continue;
        s_re += coeff * eta_re[eta_row + (int64_t)k];
        s_im += coeff * eta_im[eta_row + (int64_t)k];
      }
    }
    if (s_re != 0.0) atomicAdd(&rho_re[(int64_t)m * rho_m_stride + pq_idx], cij * s_re);
    if (s_im != 0.0) atomicAdd(&rho_im[(int64_t)m * rho_m_stride + pq_idx], cij * s_im);
  }
}

__device__ __forceinline__ void soc_triplet_accumulate_gm_edge_dev(
    int i,
    int j,
    int p,
    int q,
    int norb,
    double cij,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* h_re,
    const double* h_im,
    double* gm_re,
    double* gm_im) {
  if (cij == 0.0) return;
  const double* cbi = c_bra + (int64_t)i * (int64_t)nb;
  const double* ckj = c_ket + (int64_t)j * (int64_t)nk;
  int64_t pq_idx = (int64_t)p * (int64_t)norb + (int64_t)q;
  int64_t gm_m_stride = (int64_t)nb * (int64_t)nk;
  int64_t h_m_stride = (int64_t)norb * (int64_t)norb;

  for (int m = 0; m < 3; ++m) {
    int64_t hm_idx = (int64_t)m * h_m_stride + pq_idx;
    double hr = h_re[hm_idx];
    double hi = h_im[hm_idx];
    if (hr == 0.0 && hi == 0.0) continue;
    int64_t gm_base = (int64_t)m * gm_m_stride;
    for (int b = 0; b < nb; ++b) {
      double cb = cbi[b];
      if (cb == 0.0) continue;
      int64_t gm_row = gm_base + (int64_t)b * (int64_t)nk;
      for (int k = 0; k < nk; ++k) {
        double coeff = cij * cb * ckj[k];
        if (coeff == 0.0) continue;
        if (hr != 0.0) atomicAdd(&gm_re[gm_row + (int64_t)k], coeff * hr);
        if (hi != 0.0) atomicAdd(&gm_im[gm_row + (int64_t)k], coeff * hi);
      }
    }
  }
}

__global__ void guga_triplet_build_rho_epq_table_kernel(
    const int16_t* __restrict__ node_twos,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    const int64_t* __restrict__ epq_indptr,
    const int32_t* __restrict__ epq_indices,
    const int32_t* __restrict__ epq_pq,
    const double* __restrict__ c_bra,
    int nb,
    const double* __restrict__ c_ket,
    int nk,
    const double* __restrict__ eta_re,
    const double* __restrict__ eta_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ rho_re,
    double* __restrict__ rho_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf) return;

  const int8_t* ket_steps = &steps_table[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table[(int64_t)j * (int64_t)(norb + 1)];

  int64_t start = epq_indptr[j];
  int64_t end = epq_indptr[j + 1];

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int i = epq_indices[t];
    if ((unsigned)i >= (unsigned)ncsf) continue;
    int pq = epq_pq[t];
    int p = pq / norb;
    int q = pq - p * norb;
    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || p == q) continue;

    const int8_t* bra_steps = &steps_table[(int64_t)i * (int64_t)norb];
    const int32_t* bra_nodes = &nodes_table[(int64_t)i * (int64_t)(norb + 1)];

    double cij = soc_triplet_rme_single_excitation_dev_general(
        node_twos,
        node_twos,
        bra_steps,
        ket_steps,
        bra_nodes,
        ket_nodes,
        norb,
        p,
        q,
        sixj_211,
        t_factor,
        twos_max);
    soc_triplet_accumulate_rho_edge_dev(
        i,
        j,
        p,
        q,
        norb,
        cij,
        c_bra,
        nb,
        c_ket,
        nk,
        eta_re,
        eta_im,
        rho_re,
        rho_im);
  }
}

__global__ void guga_triplet_build_rho_dfs_kernel(
    const int32_t* __restrict__ child_bra,
    const int16_t* __restrict__ node_twos_bra,
    const int64_t* __restrict__ child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* __restrict__ node_twos_ket,
    const int8_t* __restrict__ steps_table_ket,
    const int32_t* __restrict__ nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* __restrict__ c_bra,
    int nb,
    const double* __restrict__ c_ket,
    int nk,
    const double* __restrict__ eta_re,
    const double* __restrict__ eta_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ rho_re,
    double* __restrict__ rho_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf_ket) return;
  if (!soc_triplet_tri_ok_twos(twos_bra_total, 2, twos_ket_total)) return;

  const int8_t* ket_steps = &steps_table_ket[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table_ket[(int64_t)j * (int64_t)(norb + 1)];
  int nops = norb * norb;

  for (int pq_lin = (int)threadIdx.x; pq_lin < nops; pq_lin += (int)blockDim.x) {
    int p = pq_lin / norb;
    int q = pq_lin - p * norb;
    if (p == q) continue;

    int occ_p = soc_triplet_step_to_occ(ket_steps[p]);
    int occ_q = soc_triplet_step_to_occ(ket_steps[q]);
    if (occ_q <= 0 || occ_p >= 2) continue;

    int occ_p_target = occ_p + 1;
    int occ_q_target = occ_q - 1;
    if (occ_p_target > 2 || occ_q_target < 0) continue;

    int lo = (p < q) ? p : q;
    int hi = (p < q) ? q : p;

    int8_t bra_steps_local[SOC_TRIPLET_MAX_NORB];
    int32_t bra_nodes_local[SOC_TRIPLET_MAX_NORB + 1];
    for (int t = 0; t < norb; ++t) bra_steps_local[t] = ket_steps[t];
    bra_nodes_local[0] = root_bra;

    int32_t node0 = root_bra;
    int32_t idx0 = 0;
    bool prefix_ok = true;
    for (int t = 0; t < lo; ++t) {
      int step = (int)ket_steps[t];
      int32_t node1 = child_bra[(int64_t)node0 * 4 + step];
      if (node1 < 0) {
        prefix_ok = false;
        break;
      }
      idx0 += (int32_t)child_prefix_bra[(int64_t)node0 * 5 + step];
      node0 = node1;
      bra_nodes_local[t + 1] = node0;
    }
    if (!prefix_ok) continue;

    int stack_t[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_node[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_idx[SOC_TRIPLET_MAX_NORB + 2];
    int8_t stack_choice[SOC_TRIPLET_MAX_NORB + 2];

    int top = 0;
    stack_t[0] = lo;
    stack_node[0] = node0;
    stack_idx[0] = idx0;
    stack_choice[0] = 0;

    while (top >= 0) {
      int t = stack_t[top];
      if (t == hi + 1) {
        int32_t node_s = stack_node[top];
        int32_t idx_s = stack_idx[top];
        bool suffix_ok = true;
        for (int u = hi + 1; u < norb; ++u) {
          int step_u = (int)ket_steps[u];
          int32_t node1 = child_bra[(int64_t)node_s * 4 + step_u];
          if (node1 < 0) {
            suffix_ok = false;
            break;
          }
          idx_s += (int32_t)child_prefix_bra[(int64_t)node_s * 5 + step_u];
          node_s = node1;
          bra_nodes_local[u + 1] = node_s;
        }
        if (suffix_ok && node_s == leaf_bra && idx_s >= 0 && idx_s < ncsf_bra) {
          double cij = soc_triplet_rme_single_excitation_dev_general(
              node_twos_bra,
              node_twos_ket,
              bra_steps_local,
              ket_steps,
              bra_nodes_local,
              ket_nodes,
              norb,
              p,
              q,
              sixj_211,
              t_factor,
              twos_max);
          soc_triplet_accumulate_rho_edge_dev(
              idx_s,
              j,
              p,
              q,
              norb,
              cij,
              c_bra,
              nb,
              c_ket,
              nk,
              eta_re,
              eta_im,
              rho_re,
              rho_im);
        }
        top--;
        continue;
      }

      int occ_target = soc_triplet_step_to_occ(ket_steps[t]);
      if (t == p) occ_target = occ_p_target;
      else if (t == q) occ_target = occ_q_target;

      int opt0 = -1;
      int opt1 = -1;
      int nopt = 0;
      if (occ_target == 0) {
        opt0 = 0;
        nopt = 1;
      } else if (occ_target == 2) {
        opt0 = 3;
        nopt = 1;
      } else if (occ_target == 1) {
        opt0 = 1;
        opt1 = 2;
        nopt = 2;
      } else {
        top--;
        continue;
      }

      int choice = (int)stack_choice[top];
      if (choice >= nopt) {
        top--;
        continue;
      }

      int step = (choice == 0) ? opt0 : opt1;
      stack_choice[top] = (int8_t)(choice + 1);

      int32_t node = stack_node[top];
      int32_t node1 = child_bra[(int64_t)node * 4 + step];
      if (node1 < 0) continue;

      bra_steps_local[t] = (int8_t)step;
      bra_nodes_local[t + 1] = node1;
      int32_t idx1 = stack_idx[top] + (int32_t)child_prefix_bra[(int64_t)node * 5 + step];

      top++;
      stack_t[top] = t + 1;
      stack_node[top] = node1;
      stack_idx[top] = idx1;
      stack_choice[top] = 0;
    }
  }
}

__global__ void guga_triplet_build_gm_epq_table_kernel(
    const int16_t* __restrict__ node_twos,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    const int64_t* __restrict__ epq_indptr,
    const int32_t* __restrict__ epq_indices,
    const int32_t* __restrict__ epq_pq,
    const double* __restrict__ c_bra,
    int nb,
    const double* __restrict__ c_ket,
    int nk,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ gm_re,
    double* __restrict__ gm_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf) return;

  const int8_t* ket_steps = &steps_table[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table[(int64_t)j * (int64_t)(norb + 1)];

  int64_t start = epq_indptr[j];
  int64_t end = epq_indptr[j + 1];

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int i = epq_indices[t];
    if ((unsigned)i >= (unsigned)ncsf) continue;
    int pq = epq_pq[t];
    int p = pq / norb;
    int q = pq - p * norb;
    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || p == q) continue;

    const int8_t* bra_steps = &steps_table[(int64_t)i * (int64_t)norb];
    const int32_t* bra_nodes = &nodes_table[(int64_t)i * (int64_t)(norb + 1)];

    double cij = soc_triplet_rme_single_excitation_dev_general(
        node_twos,
        node_twos,
        bra_steps,
        ket_steps,
        bra_nodes,
        ket_nodes,
        norb,
        p,
        q,
        sixj_211,
        t_factor,
        twos_max);
    soc_triplet_accumulate_gm_edge_dev(
        i,
        j,
        p,
        q,
        norb,
        cij,
        c_bra,
        nb,
        c_ket,
        nk,
        h_re,
        h_im,
        gm_re,
        gm_im);
  }
}

__global__ void guga_triplet_build_gm_dfs_kernel(
    const int32_t* __restrict__ child_bra,
    const int16_t* __restrict__ node_twos_bra,
    const int64_t* __restrict__ child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* __restrict__ node_twos_ket,
    const int8_t* __restrict__ steps_table_ket,
    const int32_t* __restrict__ nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* __restrict__ c_bra,
    int nb,
    const double* __restrict__ c_ket,
    int nk,
    const double* __restrict__ h_re,
    const double* __restrict__ h_im,
    const float* __restrict__ sixj_211,
    const float* __restrict__ t_factor,
    int twos_max,
    double* __restrict__ gm_re,
    double* __restrict__ gm_im) {
  int j = (int)blockIdx.x;
  if (j >= ncsf_ket) return;
  if (!soc_triplet_tri_ok_twos(twos_bra_total, 2, twos_ket_total)) return;

  const int8_t* ket_steps = &steps_table_ket[(int64_t)j * (int64_t)norb];
  const int32_t* ket_nodes = &nodes_table_ket[(int64_t)j * (int64_t)(norb + 1)];
  int nops = norb * norb;

  for (int pq_lin = (int)threadIdx.x; pq_lin < nops; pq_lin += (int)blockDim.x) {
    int p = pq_lin / norb;
    int q = pq_lin - p * norb;
    if (p == q) continue;

    int occ_p = soc_triplet_step_to_occ(ket_steps[p]);
    int occ_q = soc_triplet_step_to_occ(ket_steps[q]);
    if (occ_q <= 0 || occ_p >= 2) continue;

    int occ_p_target = occ_p + 1;
    int occ_q_target = occ_q - 1;
    if (occ_p_target > 2 || occ_q_target < 0) continue;

    int lo = (p < q) ? p : q;
    int hi = (p < q) ? q : p;

    int8_t bra_steps_local[SOC_TRIPLET_MAX_NORB];
    int32_t bra_nodes_local[SOC_TRIPLET_MAX_NORB + 1];
    for (int t = 0; t < norb; ++t) bra_steps_local[t] = ket_steps[t];
    bra_nodes_local[0] = root_bra;

    int32_t node0 = root_bra;
    int32_t idx0 = 0;
    bool prefix_ok = true;
    for (int t = 0; t < lo; ++t) {
      int step = (int)ket_steps[t];
      int32_t node1 = child_bra[(int64_t)node0 * 4 + step];
      if (node1 < 0) {
        prefix_ok = false;
        break;
      }
      idx0 += (int32_t)child_prefix_bra[(int64_t)node0 * 5 + step];
      node0 = node1;
      bra_nodes_local[t + 1] = node0;
    }
    if (!prefix_ok) continue;

    int stack_t[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_node[SOC_TRIPLET_MAX_NORB + 2];
    int32_t stack_idx[SOC_TRIPLET_MAX_NORB + 2];
    int8_t stack_choice[SOC_TRIPLET_MAX_NORB + 2];

    int top = 0;
    stack_t[0] = lo;
    stack_node[0] = node0;
    stack_idx[0] = idx0;
    stack_choice[0] = 0;

    while (top >= 0) {
      int t = stack_t[top];
      if (t == hi + 1) {
        int32_t node_s = stack_node[top];
        int32_t idx_s = stack_idx[top];
        bool suffix_ok = true;
        for (int u = hi + 1; u < norb; ++u) {
          int step_u = (int)ket_steps[u];
          int32_t node1 = child_bra[(int64_t)node_s * 4 + step_u];
          if (node1 < 0) {
            suffix_ok = false;
            break;
          }
          idx_s += (int32_t)child_prefix_bra[(int64_t)node_s * 5 + step_u];
          node_s = node1;
          bra_nodes_local[u + 1] = node_s;
        }
        if (suffix_ok && node_s == leaf_bra && idx_s >= 0 && idx_s < ncsf_bra) {
          double cij = soc_triplet_rme_single_excitation_dev_general(
              node_twos_bra,
              node_twos_ket,
              bra_steps_local,
              ket_steps,
              bra_nodes_local,
              ket_nodes,
              norb,
              p,
              q,
              sixj_211,
              t_factor,
              twos_max);
          soc_triplet_accumulate_gm_edge_dev(
              idx_s,
              j,
              p,
              q,
              norb,
              cij,
              c_bra,
              nb,
              c_ket,
              nk,
              h_re,
              h_im,
              gm_re,
              gm_im);
        }
        top--;
        continue;
      }

      int occ_target = soc_triplet_step_to_occ(ket_steps[t]);
      if (t == p) occ_target = occ_p_target;
      else if (t == q) occ_target = occ_q_target;

      int opt0 = -1;
      int opt1 = -1;
      int nopt = 0;
      if (occ_target == 0) {
        opt0 = 0;
        nopt = 1;
      } else if (occ_target == 2) {
        opt0 = 3;
        nopt = 1;
      } else if (occ_target == 1) {
        opt0 = 1;
        opt1 = 2;
        nopt = 2;
      } else {
        top--;
        continue;
      }

      int choice = (int)stack_choice[top];
      if (choice >= nopt) {
        top--;
        continue;
      }

      int step = (choice == 0) ? opt0 : opt1;
      stack_choice[top] = (int8_t)(choice + 1);

      int32_t node = stack_node[top];
      int32_t node1 = child_bra[(int64_t)node * 4 + step];
      if (node1 < 0) continue;

      bra_steps_local[t] = (int8_t)step;
      bra_nodes_local[t + 1] = node1;
      int32_t idx1 = stack_idx[top] + (int32_t)child_prefix_bra[(int64_t)node * 5 + step];

      top++;
      stack_t[top] = t + 1;
      stack_node[top] = node1;
      stack_idx[top] = idx1;
      stack_choice[top] = 0;
    }
  }
}

}  // namespace

extern "C" void guga_triplet_apply_contracted_all_m_from_epq_table_launch_stream(
    const int16_t* node_twos,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const int32_t* epq_pq,
    const double* x,
    const double* h_re,
    const double* h_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* y_re,
    double* y_im,
    cudaStream_t stream,
    int threads) {
  if (!node_twos || !steps_table || !nodes_table || !epq_indptr || !epq_indices || !epq_pq || !x || !h_re || !h_im ||
      !sixj_211 || !t_factor || !y_re || !y_im) {
    return;
  }
  if (ncsf <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf, 1, 1);
  guga_triplet_apply_contracted_epq_table_kernel<<<grid, block, 0, stream>>>(
      node_twos,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      epq_indptr,
      epq_indices,
      epq_pq,
      x,
      h_re,
      h_im,
      sixj_211,
      t_factor,
      twos_max,
      y_re,
      y_im);
}

extern "C" void guga_triplet_apply_contracted_all_m_dfs_launch_stream(
    const int32_t* child_bra,
    const int16_t* node_twos_bra,
    const int64_t* child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* node_twos_ket,
    const int8_t* steps_table_ket,
    const int32_t* nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* x,
    const double* h_re,
    const double* h_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* y_re,
    double* y_im,
    cudaStream_t stream,
    int threads) {
  if (!child_bra || !node_twos_bra || !child_prefix_bra || !node_twos_ket || !steps_table_ket || !nodes_table_ket ||
      !x || !h_re || !h_im || !sixj_211 || !t_factor || !y_re || !y_im) {
    return;
  }
  if (ncsf_bra <= 0 || ncsf_ket <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB) return;
  if (root_bra < 0 || leaf_bra < 0) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf_ket, 1, 1);
  guga_triplet_apply_contracted_dfs_kernel<<<grid, block, 0, stream>>>(
      child_bra,
      node_twos_bra,
      child_prefix_bra,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      node_twos_ket,
      steps_table_ket,
      nodes_table_ket,
      ncsf_ket,
      norb,
      twos_ket_total,
      x,
      h_re,
      h_im,
      sixj_211,
      t_factor,
      twos_max,
      y_re,
      y_im);
}

extern "C" void guga_triplet_build_rho_all_m_from_epq_table_launch_stream(
    const int16_t* node_twos,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const int32_t* epq_pq,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* eta_re,
    const double* eta_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* rho_re,
    double* rho_im,
    cudaStream_t stream,
    int threads) {
  if (!node_twos || !steps_table || !nodes_table || !epq_indptr || !epq_indices || !epq_pq || !c_bra || !c_ket || !eta_re ||
      !eta_im || !sixj_211 || !t_factor || !rho_re || !rho_im) {
    return;
  }
  if (ncsf <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB || nb <= 0 || nk <= 0) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf, 1, 1);
  guga_triplet_build_rho_epq_table_kernel<<<grid, block, 0, stream>>>(
      node_twos,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      epq_indptr,
      epq_indices,
      epq_pq,
      c_bra,
      nb,
      c_ket,
      nk,
      eta_re,
      eta_im,
      sixj_211,
      t_factor,
      twos_max,
      rho_re,
      rho_im);
}

extern "C" void guga_triplet_build_rho_all_m_dfs_launch_stream(
    const int32_t* child_bra,
    const int16_t* node_twos_bra,
    const int64_t* child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* node_twos_ket,
    const int8_t* steps_table_ket,
    const int32_t* nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* eta_re,
    const double* eta_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* rho_re,
    double* rho_im,
    cudaStream_t stream,
    int threads) {
  if (!child_bra || !node_twos_bra || !child_prefix_bra || !node_twos_ket || !steps_table_ket || !nodes_table_ket ||
      !c_bra || !c_ket || !eta_re || !eta_im || !sixj_211 || !t_factor || !rho_re || !rho_im) {
    return;
  }
  if (ncsf_bra <= 0 || ncsf_ket <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB || nb <= 0 || nk <= 0) return;
  if (root_bra < 0 || leaf_bra < 0) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf_ket, 1, 1);
  guga_triplet_build_rho_dfs_kernel<<<grid, block, 0, stream>>>(
      child_bra,
      node_twos_bra,
      child_prefix_bra,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      node_twos_ket,
      steps_table_ket,
      nodes_table_ket,
      ncsf_ket,
      norb,
      twos_ket_total,
      c_bra,
      nb,
      c_ket,
      nk,
      eta_re,
      eta_im,
      sixj_211,
      t_factor,
      twos_max,
      rho_re,
      rho_im);
}

extern "C" void guga_triplet_build_gm_all_m_from_epq_table_launch_stream(
    const int16_t* node_twos,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const int32_t* epq_pq,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* h_re,
    const double* h_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* gm_re,
    double* gm_im,
    cudaStream_t stream,
    int threads) {
  if (!node_twos || !steps_table || !nodes_table || !epq_indptr || !epq_indices || !epq_pq || !c_bra || !c_ket || !h_re ||
      !h_im || !sixj_211 || !t_factor || !gm_re || !gm_im) {
    return;
  }
  if (ncsf <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB || nb <= 0 || nk <= 0) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf, 1, 1);
  guga_triplet_build_gm_epq_table_kernel<<<grid, block, 0, stream>>>(
      node_twos,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      epq_indptr,
      epq_indices,
      epq_pq,
      c_bra,
      nb,
      c_ket,
      nk,
      h_re,
      h_im,
      sixj_211,
      t_factor,
      twos_max,
      gm_re,
      gm_im);
}

extern "C" void guga_triplet_build_gm_all_m_dfs_launch_stream(
    const int32_t* child_bra,
    const int16_t* node_twos_bra,
    const int64_t* child_prefix_bra,
    int root_bra,
    int leaf_bra,
    int ncsf_bra,
    int twos_bra_total,
    const int16_t* node_twos_ket,
    const int8_t* steps_table_ket,
    const int32_t* nodes_table_ket,
    int ncsf_ket,
    int norb,
    int twos_ket_total,
    const double* c_bra,
    int nb,
    const double* c_ket,
    int nk,
    const double* h_re,
    const double* h_im,
    const float* sixj_211,
    const float* t_factor,
    int twos_max,
    double* gm_re,
    double* gm_im,
    cudaStream_t stream,
    int threads) {
  if (!child_bra || !node_twos_bra || !child_prefix_bra || !node_twos_ket || !steps_table_ket || !nodes_table_ket ||
      !c_bra || !c_ket || !h_re || !h_im || !sixj_211 || !t_factor || !gm_re || !gm_im) {
    return;
  }
  if (ncsf_bra <= 0 || ncsf_ket <= 0 || norb <= 0 || norb > SOC_TRIPLET_MAX_NORB || nb <= 0 || nk <= 0) return;
  if (root_bra < 0 || leaf_bra < 0) return;
  if (threads <= 0) threads = 128;
  if (threads > 1024) threads = 1024;

  dim3 block((unsigned)threads, 1, 1);
  dim3 grid((unsigned)ncsf_ket, 1, 1);
  guga_triplet_build_gm_dfs_kernel<<<grid, block, 0, stream>>>(
      child_bra,
      node_twos_bra,
      child_prefix_bra,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      node_twos_ket,
      steps_table_ket,
      nodes_table_ket,
      ncsf_ket,
      norb,
      twos_ket_total,
      c_bra,
      nb,
      c_ket,
      nk,
      h_re,
      h_im,
      sixj_211,
      t_factor,
      twos_max,
      gm_re,
      gm_im);
}
