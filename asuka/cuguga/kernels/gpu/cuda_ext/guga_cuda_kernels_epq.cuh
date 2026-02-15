namespace {

constexpr int Q_W = 0;
constexpr int Q_uR = 1;
constexpr int Q_R = 2;
constexpr int Q_oR = 3;
constexpr int Q_uL = 4;
constexpr int Q_L = 5;
constexpr int Q_oL = 6;

constexpr int MAX_NORB = 64;
constexpr int MAX_B = 2 * MAX_NORB + 2;  // ensure we cover (b+2) when b<=2*norb

__device__ __forceinline__ int step_to_occ(int8_t step) {
  if (step == 0) return 0;
  if (step == 3) return 2;
  return 1;
}

__device__ __forceinline__ void guga_maybe_enable_smem_spilling() {
#if defined(GUGA_CUDA_ENABLE_SMEM_SPILLING) && GUGA_CUDA_ENABLE_SMEM_SPILLING
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
  asm volatile(".pragma \"enable_smem_spilling\";");
#endif
#endif
}

// Precomputed per-b constants to reduce sqrt/div work in the inner segment loop.
__device__ __constant__ double kA_1_0[MAX_B + 1];  // sqrt((b+1)/(b+0))
__device__ __constant__ double kA_1_2[MAX_B + 1];  // sqrt((b+1)/(b+2))
__device__ __constant__ double kA_0_1[MAX_B + 1];  // sqrt((b+0)/(b+1))
__device__ __constant__ double kA_2_1[MAX_B + 1];  // sqrt((b+2)/(b+1))
__device__ __constant__ double kC_0[MAX_B + 1];    // C_int(b,0)
__device__ __constant__ double kC_1[MAX_B + 1];    // C_int(b,1)
__device__ __constant__ double kC_2[MAX_B + 1];    // C_int(b,2)
__device__ __constant__ double kInv_b[MAX_B + 1];  // 1/b (kInv_b[0]=0)
__device__ __constant__ double kInv_b1[MAX_B + 1]; // 1/(b+1)
__device__ __constant__ double kInv_b2[MAX_B + 1]; // 1/(b+2)

// Float versions of the same LUTs for FP32 EPQ kernels.
// Note: keep these in global device memory (not constant memory) to avoid
// constant-space pressure/regressions on some toolchain/driver combinations.
__device__ float kA_1_0_f[MAX_B + 1];
__device__ float kA_1_2_f[MAX_B + 1];
__device__ float kA_0_1_f[MAX_B + 1];
__device__ float kA_2_1_f[MAX_B + 1];
__device__ float kC_0_f[MAX_B + 1];
__device__ float kC_1_f[MAX_B + 1];
__device__ float kC_2_f[MAX_B + 1];
__device__ float kInv_b_f[MAX_B + 1];
__device__ float kInv_b1_f[MAX_B + 1];
__device__ float kInv_b2_f[MAX_B + 1];
__device__ __constant__ int kSegmentLutReady = 0;

__device__ __forceinline__ double A_int(int b, int x, int y) {
  return sqrt(((double)b + (double)x) / ((double)b + (double)y));
}

__device__ __forceinline__ double C_int(int b, int x) {
  double bp = (double)b + (double)x;
  return sqrt((bp - 1.0) * (bp + 1.0)) / bp;
}

__device__ __forceinline__ double segment_value_int_fallback(int q, int dprime, int d, int db, int b) {
  if (q == Q_W) {
    if (dprime != d) return 0.0;
    if (d == 0) return 0.0;
    if (d == 1 || d == 2) return 1.0;
    if (d == 3) return 2.0;
    return 0.0;
  }

  if (q == Q_uR) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0;
    if (dprime == 3 && d == 1) return A_int(b, 1, 0);
    if (dprime == 3 && d == 2) return A_int(b, 1, 2);
    return 0.0;
  }

  if (q == Q_oR) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0;
    if (dprime == 1 && d == 3) return A_int(b, 0, 1);
    if (dprime == 2 && d == 3) return A_int(b, 2, 1);
    return 0.0;
  }

  if (q == Q_uL) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0;
    if (dprime == 1 && d == 3) return A_int(b, 2, 1);
    if (dprime == 2 && d == 3) return A_int(b, 0, 1);
    return 0.0;
  }

  if (q == Q_oL) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0;
    if (dprime == 3 && d == 1) return A_int(b, 0, 1);
    if (dprime == 3 && d == 2) return A_int(b, 2, 1);
    return 0.0;
  }

  if (q == Q_R) {
    if (db != -1 && db != +1) return 0.0;
    if (dprime == 0 && d == 0) return 1.0;
    if (dprime == 1 && d == 1) return (db == -1) ? -1.0 : C_int(b, 0);
    if (dprime == 1 && d == 2) return (db == -1) ? (-1.0 / ((double)b + 2.0)) : 0.0;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0 : ((b != 0) ? (1.0 / (double)b) : 0.0);
    if (dprime == 2 && d == 2) return (db == -1) ? C_int(b, 2) : -1.0;
    if (dprime == 3 && d == 3) return -1.0;
    return 0.0;
  }

  if (q == Q_L) {
    if (db != -1 && db != +1) return 0.0;
    if (dprime == 0 && d == 0) return 1.0;
    if (dprime == 1 && d == 1) return (db == -1) ? C_int(b, 1) : -1.0;
    if (dprime == 1 && d == 2) return (db == -1) ? (1.0 / ((double)b + 1.0)) : 0.0;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0 : (-1.0 / ((double)b + 1.0));
    if (dprime == 2 && d == 2) return (db == -1) ? -1.0 : C_int(b, 1);
    if (dprime == 3 && d == 3) return -1.0;
    return 0.0;
  }

  return 0.0;
}

__device__ __forceinline__ double segment_value_int(int q, int dprime, int d, int db, int b) {
  // Fast path using precomputed per-b constants; fall back if LUT is unavailable or b is out of range.
  if (!kSegmentLutReady || (unsigned)b > (unsigned)MAX_B) return segment_value_int_fallback(q, dprime, d, db, b);

  if (q == Q_W) {
    if (dprime != d) return 0.0;
    if (d == 0) return 0.0;
    if (d == 1 || d == 2) return 1.0;
    if (d == 3) return 2.0;
    return 0.0;
  }

  if (q == Q_uR) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0;
    if (dprime == 3 && d == 1) return kA_1_0[b];
    if (dprime == 3 && d == 2) return kA_1_2[b];
    return 0.0;
  }

  if (q == Q_oR) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0;
    if (dprime == 1 && d == 3) return kA_0_1[b];
    if (dprime == 2 && d == 3) return kA_2_1[b];
    return 0.0;
  }

  if (q == Q_uL) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0;
    if (dprime == 1 && d == 3) return kA_2_1[b];
    if (dprime == 2 && d == 3) return kA_0_1[b];
    return 0.0;
  }

  if (q == Q_oL) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0;
    if (dprime == 3 && d == 1) return kA_0_1[b];
    if (dprime == 3 && d == 2) return kA_2_1[b];
    return 0.0;
  }

  if (q == Q_R) {
    if (db != -1 && db != +1) return 0.0;
    if (dprime == 0 && d == 0) return 1.0;
    if (dprime == 1 && d == 1) return (db == -1) ? -1.0 : kC_0[b];
    if (dprime == 1 && d == 2) return (db == -1) ? (-kInv_b2[b]) : 0.0;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0 : kInv_b[b];
    if (dprime == 2 && d == 2) return (db == -1) ? kC_2[b] : -1.0;
    if (dprime == 3 && d == 3) return -1.0;
    return 0.0;
  }

  if (q == Q_L) {
    if (db != -1 && db != +1) return 0.0;
    if (dprime == 0 && d == 0) return 1.0;
    if (dprime == 1 && d == 1) return (db == -1) ? kC_1[b] : -1.0;
    if (dprime == 1 && d == 2) return (db == -1) ? kInv_b1[b] : 0.0;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0 : (-kInv_b1[b]);
    if (dprime == 2 && d == 2) return (db == -1) ? -1.0 : kC_1[b];
    if (dprime == 3 && d == 3) return -1.0;
    return 0.0;
  }

  return 0.0;
}

// Float version of segment_value_int using float LUTs for true FP32 EPQ kernels.
__device__ __forceinline__ float segment_value_int_f32(int q, int dprime, int d, int db, int b) {
  if (!kSegmentLutReady || (unsigned)b > (unsigned)MAX_B) {
    return (float)segment_value_int_fallback(q, dprime, d, db, b);
  }

  if (q == Q_W) {
    if (dprime != d) return 0.0f;
    if (d == 0) return 0.0f;
    if (d == 1 || d == 2) return 1.0f;
    if (d == 3) return 2.0f;
    return 0.0f;
  }

  if (q == Q_uR) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0f;
    if (dprime == 3 && d == 1) return kA_1_0_f[b];
    if (dprime == 3 && d == 2) return kA_1_2_f[b];
    return 0.0f;
  }

  if (q == Q_oR) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0f;
    if (dprime == 1 && d == 3) return kA_0_1_f[b];
    if (dprime == 2 && d == 3) return kA_2_1_f[b];
    return 0.0f;
  }

  if (q == Q_uL) {
    if (dprime == 0 && (d == 1 || d == 2)) return 1.0f;
    if (dprime == 1 && d == 3) return kA_2_1_f[b];
    if (dprime == 2 && d == 3) return kA_0_1_f[b];
    return 0.0f;
  }

  if (q == Q_oL) {
    if (d == 0 && (dprime == 1 || dprime == 2)) return 1.0f;
    if (dprime == 3 && d == 1) return kA_0_1_f[b];
    if (dprime == 3 && d == 2) return kA_2_1_f[b];
    return 0.0f;
  }

  if (q == Q_R) {
    if (db != -1 && db != +1) return 0.0f;
    if (dprime == 0 && d == 0) return 1.0f;
    if (dprime == 1 && d == 1) return (db == -1) ? -1.0f : kC_0_f[b];
    if (dprime == 1 && d == 2) return (db == -1) ? (-kInv_b2_f[b]) : 0.0f;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0f : kInv_b_f[b];
    if (dprime == 2 && d == 2) return (db == -1) ? kC_2_f[b] : -1.0f;
    if (dprime == 3 && d == 3) return -1.0f;
    return 0.0f;
  }

  if (q == Q_L) {
    if (db != -1 && db != +1) return 0.0f;
    if (dprime == 0 && d == 0) return 1.0f;
    if (dprime == 1 && d == 1) return (db == -1) ? kC_1_f[b] : -1.0f;
    if (dprime == 1 && d == 2) return (db == -1) ? kInv_b1_f[b] : 0.0f;
    if (dprime == 2 && d == 1) return (db == -1) ? 0.0f : (-kInv_b1_f[b]);
    if (dprime == 2 && d == 2) return (db == -1) ? -1.0f : kC_1_f[b];
    if (dprime == 3 && d == 3) return -1.0f;
    return 0.0f;
  }

  return 0.0f;
}

// Template dispatcher: selects FP32 or FP64 segment_value_int based on type.
template <typename T>
__device__ __forceinline__ T segment_value_int_t(int q, int dprime, int d, int db, int b);

template <>
__device__ __forceinline__ double segment_value_int_t<double>(int q, int dprime, int d, int db, int b) {
  return segment_value_int(q, dprime, d, db, b);
}

template <>
__device__ __forceinline__ float segment_value_int_t<float>(int q, int dprime, int d, int db, int b) {
  return segment_value_int_f32(q, dprime, d, db, b);
}

__device__ __forceinline__ int candidate_dprimes(int qk, int d_k, int* dp0, int* dp1) {
  if (qk == Q_W) {
    dp0[0] = d_k;
    return 1;
  }

  if (qk == Q_uR || qk == Q_oL) {
    if (d_k == 0) {
      dp0[0] = 1;
      dp1[0] = 2;
      return 2;
    }
    if (d_k == 1 || d_k == 2) {
      dp0[0] = 3;
      return 1;
    }
    return 0;
  }

  if (qk == Q_oR || qk == Q_uL) {
    if (d_k == 1 || d_k == 2) {
      dp0[0] = 0;
      return 1;
    }
    if (d_k == 3) {
      dp0[0] = 1;
      dp1[0] = 2;
      return 2;
    }
    return 0;
  }

  if (qk == Q_R) {
    if (d_k == 1 || d_k == 2) {
      dp0[0] = 1;
      dp1[0] = 2;
      return 2;
    }
    dp0[0] = d_k;
    return 1;
  }

  if (qk == Q_L) {
    if (d_k == 0) {
      dp0[0] = 0;
      return 1;
    }
    if (d_k == 3) {
      dp0[0] = 3;
      return 1;
    }
    dp0[0] = 1;
    dp1[0] = 2;
    return 2;
  }

  return 0;
}

template <int MAX_NORB_T>
__device__ __forceinline__ bool epq_reconstruct_path_from_index(
    const int32_t* __restrict__ child,
    const int64_t* __restrict__ child_prefix,
    int norb,
    int ncsf,
    int csf_idx,
    int8_t* __restrict__ steps_out,
    int32_t* __restrict__ nodes_out) {
  if ((unsigned)csf_idx >= (unsigned)ncsf) return false;
  if (norb > MAX_NORB_T) return false;

  int32_t node = 0;
  int64_t idx = (int64_t)csf_idx;
  nodes_out[0] = node;

  for (int k = 0; k < norb; k++) {
    int step = -1;
    int32_t next = -1;

    // child_prefix[node, s] stores cumulative walks for steps prior to s.
    for (int s = 0; s < 4; s++) {
      int32_t child_s = child[node * 4 + s];
      if (child_s < 0) continue;
      int64_t lo = child_prefix[node * 5 + s];
      int64_t hi = child_prefix[node * 5 + (s + 1)];
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

__global__ void epq_contribs_one_debug_kernel(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int norb,
    int csf_idx,
    const int8_t* __restrict__ steps,
    const int32_t* __restrict__ nodes,
    int p,
    int q,
    int max_out,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_coeff,
    int* __restrict__ out_count,
    int* __restrict__ out_overflow) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  if (p == q) {
    *out_count = 0;
    *out_overflow = 0;
    return;
  }

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) {
    *out_count = 0;
    *out_overflow = 0;
    return;
  }

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

  int64_t idx_prefix[MAX_NORB + 1];
  idx_prefix[0] = 0;
  for (int kk = 0; kk < norb; kk++) {
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx_prefix[kk + 1] = idx_prefix[kk] + child_prefix[node_kk * 5 + step_kk];
  }

  int64_t prefix_offset = idx_prefix[start];
  int64_t prefix_endplus1 = idx_prefix[end + 1];
  int64_t suffix_offset = (int64_t)csf_idx - prefix_endplus1;

  int d_ref[MAX_NORB];
  int b_ref[MAX_NORB];
  for (int kk = 0; kk < norb; kk++) {
    d_ref[kk] = (int)steps[kk];
    b_ref[kk] = (int)node_twos[nodes[kk + 1]];
  }

  int st_k[MAX_NORB];
  int st_node[MAX_NORB];
  double st_w[MAX_NORB];
  int64_t st_seg[MAX_NORB];
  int top = 0;

  st_k[top] = start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = st_k[top];
    int node_k = st_node[top];
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
        if (out >= max_out) {
          overflow = 1;
          continue;
        }
        out_idx[out] = csf_i;
        out_coeff[out] = w2;
        out++;
      } else {
        if (top >= MAX_NORB) {
          overflow = 1;
          continue;
        }
        st_k[top] = k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  *out_count = out;
  *out_overflow = overflow;
}

__global__ void epq_apply_g_debug_kernel(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int norb,
    int csf_idx,
    const int8_t* __restrict__ steps,
    const int32_t* __restrict__ nodes,
    const double* __restrict__ g_flat,
    double thresh_gpq,
    double thresh_contrib,
    int max_out,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_val,
    int* __restrict__ out_count,
    int* __restrict__ out_overflow,
    int* __restrict__ out_n_pairs) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int64_t idx_prefix[MAX_NORB + 1];
  idx_prefix[0] = 0;

  int d_ref[MAX_NORB];
  int b_ref[MAX_NORB];
  int occ_ref[MAX_NORB];
  for (int kk = 0; kk < norb; kk++) {
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx_prefix[kk + 1] = idx_prefix[kk] + child_prefix[node_kk * 5 + step_kk];
    d_ref[kk] = step_kk;
    b_ref[kk] = (int)node_twos[nodes[kk + 1]];
    occ_ref[kk] = step_to_occ((int8_t)step_kk);
  }

  int out = 0;
  int overflow = 0;
  int n_pairs = 0;

  // Diagonal contribution: Σ_p g[p,p] * occ[p]
  double diag_contrib = 0.0;
  for (int p = 0; p < norb; p++) {
    double gpp = g_flat[p * norb + p];
    if (gpp == 0.0) continue;
    if (thresh_gpq > 0.0 && fabs(gpp) <= thresh_gpq) continue;
    int occ_p = occ_ref[p];
    if (occ_p) diag_contrib += gpp * (double)occ_p;
  }
  if (diag_contrib != 0.0) {
    if (!(thresh_contrib > 0.0 && fabs(diag_contrib) <= thresh_contrib)) {
      if (out >= max_out) {
        overflow = 1;
      } else {
        out_idx[out] = csf_idx;
        out_val[out] = diag_contrib;
        out++;
      }
    }
  }

  int st_k[MAX_NORB];
  int st_node[MAX_NORB];
  double st_w[MAX_NORB];
  int64_t st_seg[MAX_NORB];

  for (int p = 0; p < norb; p++) {
    int occ_p = occ_ref[p];
    if (occ_p >= 2) continue;
    for (int q = 0; q < norb; q++) {
      if (q == p) continue;
      int occ_q = occ_ref[q];
      if (occ_q <= 0) continue;

      double wgt = g_flat[p * norb + q];
      if (wgt == 0.0) continue;
      if (thresh_gpq > 0.0 && fabs(wgt) <= thresh_gpq) continue;

      n_pairs += 1;

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

      int64_t prefix_offset = idx_prefix[start];
      int64_t prefix_endplus1 = idx_prefix[end + 1];
      int64_t suffix_offset = (int64_t)csf_idx - prefix_endplus1;

      int top = 0;
      st_k[top] = start;
      st_node[top] = node_start;
      st_w[top] = 1.0;
      st_seg[top] = 0;
      top++;

      while (top) {
        top--;
        int kpos = st_k[top];
        int node_k = st_node[top];
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
            double val = wgt * w2;
            if (thresh_contrib > 0.0 && fabs(val) <= thresh_contrib) continue;
            if (val == 0.0) continue;
            if (out >= max_out) {
              overflow = 1;
              continue;
            }
            out_idx[out] = csf_i;
            out_val[out] = val;
            out++;
          } else {
            if (top >= MAX_NORB) {
              overflow = 1;
              continue;
            }
            st_k[top] = k_next;
            st_node[top] = child_k;
            st_w[top] = w2;
            st_seg[top] = seg_idx2;
            top++;
          }
        }
      }
    }
  }

  *out_count = out;
  *out_overflow = overflow;
  *out_n_pairs = n_pairs;
}

}  // namespace

extern "C" cudaError_t guga_init_segment_lut() {
  static bool initialized = false;
  if (initialized) return cudaSuccess;

  constexpr int n = MAX_B + 1;
  double hA_1_0[n];
  double hA_1_2[n];
  double hA_0_1[n];
  double hA_2_1[n];
  double hC_0[n];
  double hC_1[n];
  double hC_2[n];
  double hInv_b[n];
  double hInv_b1[n];
  double hInv_b2[n];

  for (int b = 0; b <= MAX_B; b++) {
    double bd = (double)b;
    hA_1_0[b] = std::sqrt(((double)b + 1.0) / ((double)b + 0.0));
    hA_1_2[b] = std::sqrt(((double)b + 1.0) / ((double)b + 2.0));
    hA_0_1[b] = std::sqrt(((double)b + 0.0) / ((double)b + 1.0));
    hA_2_1[b] = std::sqrt(((double)b + 2.0) / ((double)b + 1.0));

    auto C_host = [&](double x) -> double {
      double bp = bd + x;
      return std::sqrt((bp - 1.0) * (bp + 1.0)) / bp;
    };
    hC_0[b] = C_host(0.0);
    hC_1[b] = C_host(1.0);
    hC_2[b] = C_host(2.0);

    hInv_b[b] = (b == 0) ? 0.0 : (1.0 / bd);
    hInv_b1[b] = 1.0 / (bd + 1.0);
    hInv_b2[b] = 1.0 / (bd + 2.0);
  }

  cudaError_t err = cudaSuccess;
  err = cudaMemcpyToSymbol(kA_1_0, hA_1_0, sizeof(hA_1_0));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_1_2, hA_1_2, sizeof(hA_1_2));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_0_1, hA_0_1, sizeof(hA_0_1));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_2_1, hA_2_1, sizeof(hA_2_1));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_0, hC_0, sizeof(hC_0));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_1, hC_1, sizeof(hC_1));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_2, hC_2, sizeof(hC_2));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b, hInv_b, sizeof(hInv_b));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b1, hInv_b1, sizeof(hInv_b1));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b2, hInv_b2, sizeof(hInv_b2));
  if (err != cudaSuccess) return err;

  // Initialize float LUTs by casting from the double host arrays.
  float hA_1_0_f[n], hA_1_2_f[n], hA_0_1_f[n], hA_2_1_f[n];
  float hC_0_f[n], hC_1_f[n], hC_2_f[n];
  float hInv_b_f[n], hInv_b1_f[n], hInv_b2_f[n];
  for (int b = 0; b <= MAX_B; b++) {
    hA_1_0_f[b] = (float)hA_1_0[b];
    hA_1_2_f[b] = (float)hA_1_2[b];
    hA_0_1_f[b] = (float)hA_0_1[b];
    hA_2_1_f[b] = (float)hA_2_1[b];
    hC_0_f[b]   = (float)hC_0[b];
    hC_1_f[b]   = (float)hC_1[b];
    hC_2_f[b]   = (float)hC_2[b];
    hInv_b_f[b] = (float)hInv_b[b];
    hInv_b1_f[b] = (float)hInv_b1[b];
    hInv_b2_f[b] = (float)hInv_b2[b];
  }
  err = cudaMemcpyToSymbol(kA_1_0_f, hA_1_0_f, sizeof(hA_1_0_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_1_2_f, hA_1_2_f, sizeof(hA_1_2_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_0_1_f, hA_0_1_f, sizeof(hA_0_1_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kA_2_1_f, hA_2_1_f, sizeof(hA_2_1_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_0_f, hC_0_f, sizeof(hC_0_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_1_f, hC_1_f, sizeof(hC_1_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kC_2_f, hC_2_f, sizeof(hC_2_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b_f, hInv_b_f, sizeof(hInv_b_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b1_f, hInv_b1_f, sizeof(hInv_b1_f));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(kInv_b2_f, hInv_b2_f, sizeof(hInv_b2_f));
  if (err != cudaSuccess) return err;

  int ready = 1;
  err = cudaMemcpyToSymbol(kSegmentLutReady, &ready, sizeof(ready));
  if (err != cudaSuccess) return err;

  initialized = true;
  return cudaSuccess;
}

extern "C" void guga_epq_contribs_one_debug_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int norb,
    int csf_idx,
    const int8_t* steps,
    const int32_t* nodes,
    int p,
    int q,
    int max_out,
    int32_t* out_idx,
    double* out_coeff,
    int* out_count,
    int* out_overflow) {
  epq_contribs_one_debug_kernel<<<1, 1>>>(
      child,
      node_twos,
      child_prefix,
      norb,
      csf_idx,
      steps,
      nodes,
      p,
      q,
      max_out,
      out_idx,
      out_coeff,
      out_count,
      out_overflow);
}

extern "C" void guga_epq_apply_g_debug_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int norb,
    int csf_idx,
    const int8_t* steps,
    const int32_t* nodes,
    const double* g_flat,
    double thresh_gpq,
    double thresh_contrib,
    int max_out,
    int32_t* out_idx,
    double* out_val,
    int* out_count,
    int* out_overflow,
    int* out_n_pairs) {
  epq_apply_g_debug_kernel<<<1, 1>>>(
      child,
      node_twos,
      child_prefix,
      norb,
      csf_idx,
      steps,
      nodes,
      g_flat,
      thresh_gpq,
      thresh_contrib,
      max_out,
      out_idx,
      out_val,
      out_count,
      out_overflow,
      out_n_pairs);
}

template <int MAX_NORB_T>
__global__ void epq_contribs_many_count_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,
    const int32_t* __restrict__ task_p,
    const int32_t* __restrict__ task_q,
    int ntasks,
    int32_t* __restrict__ counts,
    int* __restrict__ overflow_flag) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    counts[tid] = 0;
    return;
  }

  int csf_idx = task_csf[tid];
  int p = task_p[tid];
  int q = task_q[tid];

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    counts[tid] = 0;
    atomicExch(overflow_flag, 1);
    return;
  }

  if (p == q) {
    counts[tid] = 0;
    return;
  }

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) {
    counts[tid] = 0;
    return;
  }

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

  // Compute prefix offsets for csf index mapping (scan only up to end).
  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 != 0.0) out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
  counts[tid] = out;
}

template <int MAX_NORB_T>
__global__ void epq_contribs_many_write_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,
    const int32_t* __restrict__ task_p,
    const int32_t* __restrict__ task_q,
    int ntasks,
    const int64_t* __restrict__ offsets,  // [ntasks+1]
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_coeff,
    int32_t* __restrict__ out_task_csf,  // may be null, [total_out]
    int32_t* __restrict__ out_task_pq,   // may be null, [total_out] pq_id = p*norb+q
    int* __restrict__ overflow_flag) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_idx = task_csf[tid];
  int p = task_p[tid];
  int q = task_q[tid];

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    atomicExch(overflow_flag, 1);
    return;
  }

  if (p == q) return;

  int32_t pq_id = (int32_t)(p * norb + q);
  int64_t base = offsets[tid];
  int64_t limit = offsets[tid + 1];
  int64_t max_out = limit - base;

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return;

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int64_t out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 == 0.0) continue;
        if (out >= max_out) {
          overflow = 1;
          continue;
        }
        out_idx[base + out] = csf_i;
        out_coeff[base + out] = w2;
        if (out_task_csf) out_task_csf[base + out] = csf_idx;
        if (out_task_pq) out_task_pq[base + out] = pq_id;
        out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (out != max_out) overflow = 1;
  if (overflow) atomicExch(overflow_flag, 1);
}

template <int MAX_NORB_T>
__global__ void epq_contribs_many_count_allpairs_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* __restrict__ counts,
    int* __restrict__ overflow_flag) {
  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    counts[tid] = 0;
    return;
  }

  int j_local = tid / n_pairs;
  int pair = tid - j_local * n_pairs;
  int csf_idx = j_start + j_local;

  // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
  // each CSR row is already sorted by pq_id when concatenating per-task outputs.
  int p = pair / (norb - 1);
  int q_adj = pair - p * (norb - 1);
  int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    counts[tid] = 0;
    atomicExch(overflow_flag, 1);
    return;
  }

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) {
    counts[tid] = 0;
    return;
  }

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 != 0.0) out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
  counts[tid] = out;
}

template <int MAX_NORB_T, typename PQ_T, typename CoeffT>
__global__ void epq_contribs_many_write_allpairs_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* __restrict__ offsets,  // [ntasks+1]
    int32_t* __restrict__ out_idx,
    CoeffT* __restrict__ out_coeff,
    int32_t* __restrict__ out_task_csf,  // may be null, [total_out]
    PQ_T* __restrict__ out_task_pq,   // may be null, [total_out] pq_id = p*norb+q
    int* __restrict__ overflow_flag) {
  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int j_local = tid / n_pairs;
  int pair = tid - j_local * n_pairs;
  int csf_idx = j_start + j_local;

  // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
  // each CSR row is already sorted by pq_id when concatenating per-task outputs.
  int p = pair / (norb - 1);
  int q_adj = pair - p * (norb - 1);
  int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    atomicExch(overflow_flag, 1);
    return;
  }

  int32_t pq_id = (int32_t)(p * norb + q);
  int64_t base = offsets[tid];
  int64_t limit = offsets[tid + 1];
  int64_t max_out = limit - base;

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return;

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  CoeffT st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = CoeffT(1.0);
  st_seg[top] = 0;
  top++;

  int64_t out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    CoeffT w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      CoeffT seg = segment_value_int_t<CoeffT>(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      CoeffT w2 = w * seg;
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 == 0.0) continue;
        if (out >= max_out) {
          overflow = 1;
          continue;
        }
        out_idx[base + out] = csf_i;
        out_coeff[base + out] = w2;
        if (out_task_csf) out_task_csf[base + out] = csf_idx;
        if (out_task_pq) out_task_pq[base + out] = (PQ_T)pq_id;
        out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (out != max_out) overflow = 1;
  if (overflow) atomicExch(overflow_flag, 1);
}

template <int MAX_NORB_T, typename PQ_T, typename CoeffT>
__global__ void epq_contribs_many_fused_allpairs_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int max_out,
    int32_t* __restrict__ out_idx,
    CoeffT* __restrict__ out_coeff,
    int32_t* __restrict__ out_task_csf,  // may be null, [max_out]
    PQ_T* __restrict__ out_task_pq,      // may be null, [max_out] pq_id = p*norb+q
    int* __restrict__ nnz_counter,
    int* __restrict__ overflow_flag) {
  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int j_local = tid / n_pairs;
  int pair = tid - j_local * n_pairs;
  int csf_idx = j_start + j_local;

  // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
  // each CSR row is already sorted by pq_id when concatenating per-task outputs.
  int p = pair / (norb - 1);
  int q_adj = pair - p * (norb - 1);
  int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    atomicExch(overflow_flag, 1);
    return;
  }

  int32_t pq_id = (int32_t)(p * norb + q);
  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return;

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  CoeffT st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = CoeffT(1.0);
  st_seg[top] = 0;
  top++;

  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    CoeffT w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      CoeffT seg = segment_value_int_t<CoeffT>(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      CoeffT w2 = w * seg;
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 == 0.0) continue;

        int slot = atomicAdd(nnz_counter, 1);
        if (slot < 0 || slot >= max_out) {
          overflow = 1;
          continue;
        }
        out_idx[slot] = csf_i;
        out_coeff[slot] = w2;
        if (out_task_csf) out_task_csf[slot] = csf_idx;
        if (out_task_pq) out_task_pq[slot] = (PQ_T)pq_id;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
}

extern "C" void guga_epq_contribs_many_count_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    int ntasks,
    int32_t* counts,
    int* overflow_flag,
    int threads) {
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_count_kernel_t<32><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        counts,
        overflow_flag);
  } else {
    epq_contribs_many_count_kernel_t<64><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        counts,
        overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_count_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    int ntasks,
    int32_t* counts,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_count_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        counts,
        overflow_flag);
  } else {
    epq_contribs_many_count_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        counts,
        overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_write_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    int ntasks,
    const int64_t* offsets,
    int32_t* out_idx,
    double* out_coeff,
    int32_t* out_task_csf,
    int32_t* out_task_pq,
    int* overflow_flag,
    int threads) {
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_write_kernel_t<32><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        out_task_pq,
        overflow_flag);
  } else {
    epq_contribs_many_write_kernel_t<64><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        out_task_pq,
        overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_write_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    int ntasks,
    const int64_t* offsets,
    int32_t* out_idx,
    double* out_coeff,
    int32_t* out_task_csf,
    int32_t* out_task_pq,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_write_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        out_task_pq,
        overflow_flag);
  } else {
    epq_contribs_many_write_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        ntasks,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        out_task_pq,
        overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_count_allpairs_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* counts,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_count_allpairs_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, j_start, j_count, counts, overflow_flag);
  } else {
    epq_contribs_many_count_allpairs_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, j_start, j_count, counts, overflow_flag);
  }
}

template <int MAX_NORB_T, typename CoeffT>
inline void launch_write_allpairs_pq_typed_(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* offsets,
    int32_t* out_idx,
    CoeffT* out_coeff,
    int32_t* out_task_csf,
    void* out_task_pq,
    int pq_type,
    int* overflow_flag,
    int blocks,
    int threads,
    cudaStream_t stream) {
  if (pq_type == 1) {
    epq_contribs_many_write_allpairs_kernel_t<MAX_NORB_T, uint8_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<uint8_t*>(out_task_pq),
        overflow_flag);
  } else if (pq_type == 2) {
    epq_contribs_many_write_allpairs_kernel_t<MAX_NORB_T, uint16_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<uint16_t*>(out_task_pq),
        overflow_flag);
  } else {
    epq_contribs_many_write_allpairs_kernel_t<MAX_NORB_T, int32_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        offsets,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<int32_t*>(out_task_pq),
        overflow_flag);
  }
}

template <int MAX_NORB_T, typename CoeffT>
inline void launch_fused_allpairs_pq_typed_(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int max_out,
    int32_t* out_idx,
    CoeffT* out_coeff,
    int32_t* out_task_csf,
    void* out_task_pq,
    int pq_type,
    int* nnz_counter,
    int* overflow_flag,
    int blocks,
    int threads,
    cudaStream_t stream) {
  if (pq_type == 1) {
    epq_contribs_many_fused_allpairs_kernel_t<MAX_NORB_T, uint8_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        max_out,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<uint8_t*>(out_task_pq),
        nnz_counter,
        overflow_flag);
  } else if (pq_type == 2) {
    epq_contribs_many_fused_allpairs_kernel_t<MAX_NORB_T, uint16_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        max_out,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<uint16_t*>(out_task_pq),
        nnz_counter,
        overflow_flag);
  } else {
    epq_contribs_many_fused_allpairs_kernel_t<MAX_NORB_T, int32_t, CoeffT><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        max_out,
        out_idx,
        out_coeff,
        out_task_csf,
        reinterpret_cast<int32_t*>(out_task_pq),
        nnz_counter,
        overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_write_allpairs_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* offsets,
    int32_t* out_idx,
    void* out_coeff,
    int out_coeff_type,
    int32_t* out_task_csf,
    void* out_task_pq,
    int out_task_pq_type,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int pq_type = out_task_pq ? out_task_pq_type : 4;
  if (pq_type != 1 && pq_type != 2 && pq_type != 4) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (out_coeff_type != 4 && out_coeff_type != 8) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int blocks = (ntasks + threads - 1) / threads;
  if (out_coeff_type == 4) {
    float* out_coeff_f32 = reinterpret_cast<float*>(out_coeff);
    if (norb <= 32) {
      launch_write_allpairs_pq_typed_<32, float>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f32,
          out_task_csf,
          out_task_pq,
          pq_type,
          overflow_flag,
          blocks,
          threads,
          stream);
    } else {
      launch_write_allpairs_pq_typed_<64, float>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f32,
          out_task_csf,
          out_task_pq,
          pq_type,
          overflow_flag,
          blocks,
          threads,
          stream);
    }
    return;
  }

  double* out_coeff_f64 = reinterpret_cast<double*>(out_coeff);
  if (norb <= 32) {
    launch_write_allpairs_pq_typed_<32, double>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        offsets,
        out_idx,
        out_coeff_f64,
        out_task_csf,
        out_task_pq,
        pq_type,
        overflow_flag,
        blocks,
        threads,
        stream);
  } else {
    launch_write_allpairs_pq_typed_<64, double>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        offsets,
        out_idx,
        out_coeff_f64,
        out_task_csf,
        out_task_pq,
        pq_type,
        overflow_flag,
        blocks,
        threads,
        stream);
  }
}

extern "C" void guga_epq_contribs_many_fused_allpairs_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int max_out,
    int32_t* out_idx,
    void* out_coeff,
    int out_coeff_type,
    int32_t* out_task_csf,
    void* out_task_pq,
    int out_task_pq_type,
    int* nnz_counter,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (!nnz_counter || max_out <= 0) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  cudaMemsetAsync(nnz_counter, 0, sizeof(int), stream);
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int pq_type = out_task_pq ? out_task_pq_type : 4;
  if (pq_type != 1 && pq_type != 2 && pq_type != 4) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (out_coeff_type != 4 && out_coeff_type != 8) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int blocks = (ntasks + threads - 1) / threads;
  if (out_coeff_type == 4) {
    float* out_coeff_f32 = reinterpret_cast<float*>(out_coeff);
    if (norb <= 32) {
      launch_fused_allpairs_pq_typed_<32, float>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          j_start,
          j_count,
          max_out,
          out_idx,
          out_coeff_f32,
          out_task_csf,
          out_task_pq,
          pq_type,
          nnz_counter,
          overflow_flag,
          blocks,
          threads,
          stream);
    } else {
      launch_fused_allpairs_pq_typed_<64, float>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          j_start,
          j_count,
          max_out,
          out_idx,
          out_coeff_f32,
          out_task_csf,
          out_task_pq,
          pq_type,
          nnz_counter,
          overflow_flag,
          blocks,
          threads,
          stream);
    }
    return;
  }

  double* out_coeff_f64 = reinterpret_cast<double*>(out_coeff);
  if (norb <= 32) {
    launch_fused_allpairs_pq_typed_<32, double>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        max_out,
        out_idx,
        out_coeff_f64,
        out_task_csf,
        out_task_pq,
        pq_type,
        nnz_counter,
        overflow_flag,
        blocks,
        threads,
        stream);
  } else {
    launch_fused_allpairs_pq_typed_<64, double>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        j_start,
        j_count,
        max_out,
        out_idx,
        out_coeff_f64,
        out_task_csf,
        out_task_pq,
        pq_type,
        nnz_counter,
        overflow_flag,
        blocks,
        threads,
        stream);
  }
}

template <int MAX_NORB_T>
__global__ void epq_contribs_many_count_allpairs_recompute_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* __restrict__ counts,
    int* __restrict__ overflow_flag) {
  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    counts[tid] = 0;
    return;
  }

  int j_local = tid / n_pairs;
  int pair = tid - j_local * n_pairs;
  int csf_idx = j_start + j_local;

  // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
  // each CSR row is already sorted by pq_id when concatenating per-task outputs.
  int p = pair / (norb - 1);
  int q_adj = pair - p * (norb - 1);
  int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    counts[tid] = 0;
    atomicExch(overflow_flag, 1);
    return;
  }

  int8_t steps[MAX_NORB_T];
  int32_t nodes[MAX_NORB_T + 1];
  if (!epq_reconstruct_path_from_index<MAX_NORB_T>(child, child_prefix, norb, ncsf, csf_idx, steps, nodes)) {
    counts[tid] = 0;
    atomicExch(overflow_flag, 1);
    return;
  }

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) {
    counts[tid] = 0;
    return;
  }

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 != 0.0) out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
  counts[tid] = out;
}

template <int MAX_NORB_T, typename PQ_T, typename CoeffT>
__global__ void epq_contribs_many_write_allpairs_recompute_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* __restrict__ offsets,  // [ntasks+1]
    int32_t* __restrict__ out_idx,
    CoeffT* __restrict__ out_coeff,
    int32_t* __restrict__ out_task_csf,  // may be null, [total_out]
    PQ_T* __restrict__ out_task_pq,   // may be null, [total_out] pq_id = p*norb+q
    int* __restrict__ overflow_flag) {
  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int j_local = tid / n_pairs;
  int pair = tid - j_local * n_pairs;
  int csf_idx = j_start + j_local;

  // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
  // each CSR row is already sorted by pq_id when concatenating per-task outputs.
  int p = pair / (norb - 1);
  int q_adj = pair - p * (norb - 1);
  int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    atomicExch(overflow_flag, 1);
    return;
  }

  int32_t pq_id = (int32_t)(p * norb + q);
  int64_t base = offsets[tid];
  int64_t limit = offsets[tid + 1];
  int64_t max_out = limit - base;

  int8_t steps[MAX_NORB_T];
  int32_t nodes[MAX_NORB_T + 1];
  if (!epq_reconstruct_path_from_index<MAX_NORB_T>(child, child_prefix, norb, ncsf, csf_idx, steps, nodes)) {
    atomicExch(overflow_flag, 1);
    return;
  }

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return;

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  CoeffT st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = CoeffT(1.0);
  st_seg[top] = 0;
  top++;

  int64_t out = 0;
  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    CoeffT w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      CoeffT seg = segment_value_int_t<CoeffT>(qk, dprime, dk, db, bk);
      if (seg == 0.0) continue;
      CoeffT w2 = w * seg;
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if (csf_i == csf_idx) continue;
        if (w2 == 0.0) continue;
        if (out >= max_out) {
          overflow = 1;
          continue;
        }
        out_idx[base + out] = csf_i;
        out_coeff[base + out] = w2;
        if (out_task_csf) out_task_csf[base + out] = csf_idx;
        if (out_task_pq) out_task_pq[base + out] = (PQ_T)pq_id;
        out++;
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (out != max_out) overflow = 1;
  if (overflow) atomicExch(overflow_flag, 1);
}

// 10.16.6: __launch_bounds__ for hot warp-cooperative EPQ count kernel.
template <int MAX_NORB_T, int SPLIT_LEVELS_T = 2>
__global__ __launch_bounds__(256, 2)
void epq_contribs_many_count_allpairs_recompute_warp_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* __restrict__ counts,
    int* __restrict__ overflow_flag) {
  constexpr int kWarpSize = 32;
  constexpr int kMaxWarpsPerBlock = 32;
  constexpr int kMaxSeeds = 32;

  __shared__ int8_t sh_steps[kMaxWarpsPerBlock][MAX_NORB_T];
  __shared__ int32_t sh_nodes[kMaxWarpsPerBlock][MAX_NORB_T + 1];
  __shared__ int8_t sh_seed_k[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_node[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ double sh_seed_w[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_seg[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int sh_seed_count[kMaxWarpsPerBlock];
  __shared__ int sh_status[kMaxWarpsPerBlock];  // 0=ok, 1=overflow, 2=zero count
  __shared__ int sh_end[kMaxWarpsPerBlock];
  __shared__ int sh_start[kMaxWarpsPerBlock];
  __shared__ int sh_q_start[kMaxWarpsPerBlock];
  __shared__ int sh_q_mid[kMaxWarpsPerBlock];
  __shared__ int sh_q_end[kMaxWarpsPerBlock];
  __shared__ int32_t sh_node_end_target[kMaxWarpsPerBlock];
  __shared__ int32_t sh_prefix_offset[kMaxWarpsPerBlock];
  __shared__ int32_t sh_suffix_offset[kMaxWarpsPerBlock];
  __shared__ int sh_csf_idx[kMaxWarpsPerBlock];

  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int lane = (int)(threadIdx.x & (kWarpSize - 1));
  int warp_local = (int)(threadIdx.x >> 5);
  int warps_per_block = (int)(blockDim.x >> 5);
  if (warps_per_block <= 0) return;

  int task_id = (int)(blockIdx.x * warps_per_block + warp_local);
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (lane == 0) {
      counts[task_id] = 0;
      atomicExch(overflow_flag, 1);
    }
    return;
  }

  if (lane == 0) {
    sh_status[warp_local] = 0;
    sh_seed_count[warp_local] = 0;

    int j_local = task_id / n_pairs;
    int pair = task_id - j_local * n_pairs;
    int csf_idx = j_start + j_local;

    // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
    // each CSR row is already sorted by pq_id when concatenating per-task outputs.
    int p = pair / (norb - 1);
    int q_adj = pair - p * (norb - 1);
    int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
      counts[task_id] = 0;
      sh_status[warp_local] = 1;
      atomicExch(overflow_flag, 1);
    } else {
      if (!epq_reconstruct_path_from_index<MAX_NORB_T>(
              child, child_prefix, norb, ncsf, csf_idx, sh_steps[warp_local], sh_nodes[warp_local])) {
        counts[task_id] = 0;
        sh_status[warp_local] = 1;
        atomicExch(overflow_flag, 1);
      } else {
        int occ_p = step_to_occ(sh_steps[warp_local][p]);
        int occ_q = step_to_occ(sh_steps[warp_local][q]);
        if (occ_q <= 0 || occ_p >= 2) {
          counts[task_id] = 0;
          sh_status[warp_local] = 2;
        } else {
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

          int32_t node_start = sh_nodes[warp_local][start];
          int32_t node_end_target = sh_nodes[warp_local][end + 1];

          int32_t idx = 0;
          int32_t prefix_offset = 0;
          int32_t prefix_endplus1 = 0;
          for (int kk = 0; kk <= end; kk++) {
            if (kk == start) prefix_offset = idx;
            int node_kk = sh_nodes[warp_local][kk];
            int step_kk = (int)sh_steps[warp_local][kk];
            idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
            if (kk == end) prefix_endplus1 = idx;
          }
          int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

          sh_end[warp_local] = end;
          sh_start[warp_local] = start;
          sh_q_start[warp_local] = q_start;
          sh_q_mid[warp_local] = q_mid;
          sh_q_end[warp_local] = q_end;
          sh_node_end_target[warp_local] = node_end_target;
          sh_prefix_offset[warp_local] = prefix_offset;
          sh_suffix_offset[warp_local] = suffix_offset;
          sh_csf_idx[warp_local] = csf_idx;

          // Expand a short prefix serially (lane 0) so lanes can process independent suffix subtrees.
          int split_levels = end - start;
          if (split_levels > SPLIT_LEVELS_T) split_levels = SPLIT_LEVELS_T;
          if (split_levels < 0) split_levels = 0;

          int8_t cur_k[kMaxSeeds];
          int32_t cur_node[kMaxSeeds];
          double cur_w[kMaxSeeds];
          int32_t cur_seg[kMaxSeeds];
          int cur_count = 1;
          cur_k[0] = (int8_t)start;
          cur_node[0] = node_start;
          cur_w[0] = 1.0;
          cur_seg[0] = 0;

          for (int depth = 0; depth < split_levels; depth++) {
            int8_t next_k[kMaxSeeds];
            int32_t next_node[kMaxSeeds];
            double next_w[kMaxSeeds];
            int32_t next_seg[kMaxSeeds];
            int next_count = 0;

            for (int si = 0; si < cur_count; si++) {
              int kpos = (int)cur_k[si];
              int node_k = (int)cur_node[si];
              double w = cur_w[si];
              int32_t seg_idx = cur_seg[si];

              int is_first = (kpos == start);
              int is_last = (kpos == end);
              int qk = is_first ? q_start : (is_last ? q_end : q_mid);
              int dk = (int)sh_steps[warp_local][kpos];
              int bk = (int)node_twos[sh_nodes[warp_local][kpos + 1]];
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
                if (next_count >= kMaxSeeds) {
                  sh_status[warp_local] = 1;
                  atomicExch(overflow_flag, 1);
                  break;
                }
                next_k[next_count] = (int8_t)k_next;
                next_node[next_count] = child_k;
                next_w[next_count] = w * seg;
                next_seg[next_count] = (int32_t)((int64_t)seg_idx + child_prefix[node_k * 5 + dprime]);
                next_count++;
              }
              if (sh_status[warp_local] == 1) break;
            }

            if (sh_status[warp_local] == 1) break;
            cur_count = next_count;
            for (int si = 0; si < cur_count; si++) {
              cur_k[si] = next_k[si];
              cur_node[si] = next_node[si];
              cur_w[si] = next_w[si];
              cur_seg[si] = next_seg[si];
            }
            if (cur_count == 0) break;
          }

          if (sh_status[warp_local] == 0) {
            sh_seed_count[warp_local] = cur_count;
            if (cur_count == 0) {
              counts[task_id] = 0;
              sh_status[warp_local] = 2;
            } else {
              for (int si = 0; si < cur_count; si++) {
                sh_seed_k[warp_local][si] = cur_k[si];
                sh_seed_node[warp_local][si] = cur_node[si];
                sh_seed_w[warp_local][si] = cur_w[si];
                sh_seed_seg[warp_local][si] = cur_seg[si];
              }
            }
          }
        }
      }
    }
  }
  __syncwarp();

  int status = sh_status[warp_local];
  if (status != 0) return;

  int end = sh_end[warp_local];
  int start = sh_start[warp_local];
  int q_start = sh_q_start[warp_local];
  int q_mid = sh_q_mid[warp_local];
  int q_end = sh_q_end[warp_local];
  int32_t node_end_target = sh_node_end_target[warp_local];
  int32_t prefix_offset = sh_prefix_offset[warp_local];
  int32_t suffix_offset = sh_suffix_offset[warp_local];
  int csf_idx = sh_csf_idx[warp_local];

  int local_count = 0;
  int local_overflow = 0;
  int nseed = sh_seed_count[warp_local];
  if (lane < nseed) {
    int8_t st_k[MAX_NORB_T];
    int32_t st_node[MAX_NORB_T];
    double st_w[MAX_NORB_T];
    int32_t st_seg[MAX_NORB_T];
    int top = 0;

    st_k[top] = sh_seed_k[warp_local][lane];
    st_node[top] = sh_seed_node[warp_local][lane];
    st_w[top] = sh_seed_w[warp_local][lane];
    st_seg[top] = sh_seed_seg[warp_local][lane];
    top++;

    while (top) {
      top--;
      int kpos = (int)st_k[top];
      int node_k = (int)st_node[top];
      double w = st_w[top];
      int32_t seg_idx = st_seg[top];

      int is_last = (kpos == end);
      int qk = (kpos == start) ? q_start : (is_last ? q_end : q_mid);
      int dk = (int)sh_steps[warp_local][kpos];
      int bk = (int)node_twos[sh_nodes[warp_local][kpos + 1]];
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
        int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
        int32_t seg_idx2 = (int32_t)seg_idx2_64;

        if (is_last) {
          if (child_k != node_end_target) continue;
          int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
          int csf_i = (int)csf_i_ll;
          if (csf_i == csf_idx) continue;
          if (w2 != 0.0) local_count++;
        } else {
          if (top >= MAX_NORB_T) {
            local_overflow = 1;
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          top++;
        }
      }
    }
  }

  unsigned mask = __activemask();
  for (int off = 16; off > 0; off >>= 1) {
    local_count += __shfl_down_sync(mask, local_count, off);
  }
  int any_overflow = __any_sync(mask, local_overflow != 0);
  if (lane == 0) {
    counts[task_id] = local_count;
    if (any_overflow) atomicExch(overflow_flag, 1);
  }
}

// 10.16.6: __launch_bounds__ for hot warp-cooperative EPQ write kernel.
template <int MAX_NORB_T, typename PQ_T, typename CoeffT, int SPLIT_LEVELS_T = 2>
__global__ __launch_bounds__(256, 2)
void epq_contribs_many_write_allpairs_recompute_warp_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* __restrict__ offsets,
    int32_t* __restrict__ out_idx,
    CoeffT* __restrict__ out_coeff,
    int32_t* __restrict__ out_task_csf,
    PQ_T* __restrict__ out_task_pq,
    int* __restrict__ overflow_flag) {
  constexpr int kWarpSize = 32;
  constexpr int kMaxWarpsPerBlock = 32;
  constexpr int kMaxSeeds = 32;

  __shared__ int8_t sh_steps[kMaxWarpsPerBlock][MAX_NORB_T];
  __shared__ int32_t sh_nodes[kMaxWarpsPerBlock][MAX_NORB_T + 1];
  __shared__ int8_t sh_seed_k[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_node[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ CoeffT sh_seed_w[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_seg[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int sh_seed_count[kMaxWarpsPerBlock];
  __shared__ int sh_status[kMaxWarpsPerBlock];  // 0=ok, 1=overflow, 2=empty
  __shared__ int sh_end[kMaxWarpsPerBlock];
  __shared__ int sh_start[kMaxWarpsPerBlock];
  __shared__ int sh_q_start[kMaxWarpsPerBlock];
  __shared__ int sh_q_mid[kMaxWarpsPerBlock];
  __shared__ int sh_q_end[kMaxWarpsPerBlock];
  __shared__ int32_t sh_node_end_target[kMaxWarpsPerBlock];
  __shared__ int32_t sh_prefix_offset[kMaxWarpsPerBlock];
  __shared__ int32_t sh_suffix_offset[kMaxWarpsPerBlock];
  __shared__ int sh_csf_idx[kMaxWarpsPerBlock];
  __shared__ int32_t sh_pq_id[kMaxWarpsPerBlock];
  __shared__ int64_t sh_base[kMaxWarpsPerBlock];
  __shared__ int64_t sh_max_out[kMaxWarpsPerBlock];
  __shared__ unsigned long long sh_out_count[kMaxWarpsPerBlock];

  int n_pairs = norb * (norb - 1);
  int ntasks = j_count * n_pairs;

  int lane = (int)(threadIdx.x & (kWarpSize - 1));
  int warp_local = (int)(threadIdx.x >> 5);
  int warps_per_block = (int)(blockDim.x >> 5);
  if (warps_per_block <= 0) return;

  int task_id = (int)(blockIdx.x * warps_per_block + warp_local);
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  if (lane == 0) {
    sh_status[warp_local] = 0;
    sh_seed_count[warp_local] = 0;
    sh_out_count[warp_local] = 0ULL;

    int j_local = task_id / n_pairs;
    int pair = task_id - j_local * n_pairs;
    int csf_idx = j_start + j_local;

    // Keep pair IDs in monotonically increasing pq order (diagonal excluded) so
    // each CSR row is already sorted by pq_id when concatenating per-task outputs.
    int p = pair / (norb - 1);
    int q_adj = pair - p * (norb - 1);
    int q = (q_adj >= p) ? (q_adj + 1) : q_adj;

    int32_t pq_id = (int32_t)(p * norb + q);
    int64_t base = offsets[task_id];
    int64_t limit = offsets[task_id + 1];
    int64_t max_out = limit - base;
    sh_pq_id[warp_local] = pq_id;
    sh_base[warp_local] = base;
    sh_max_out[warp_local] = max_out;

    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
      sh_status[warp_local] = 1;
      atomicExch(overflow_flag, 1);
    } else {
      if (!epq_reconstruct_path_from_index<MAX_NORB_T>(
              child, child_prefix, norb, ncsf, csf_idx, sh_steps[warp_local], sh_nodes[warp_local])) {
        sh_status[warp_local] = 1;
        atomicExch(overflow_flag, 1);
      } else {
        int occ_p = step_to_occ(sh_steps[warp_local][p]);
        int occ_q = step_to_occ(sh_steps[warp_local][q]);
        if (occ_q <= 0 || occ_p >= 2) {
          sh_status[warp_local] = 2;
        } else {
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

          int32_t node_start = sh_nodes[warp_local][start];
          int32_t node_end_target = sh_nodes[warp_local][end + 1];

          int32_t idx = 0;
          int32_t prefix_offset = 0;
          int32_t prefix_endplus1 = 0;
          for (int kk = 0; kk <= end; kk++) {
            if (kk == start) prefix_offset = idx;
            int node_kk = sh_nodes[warp_local][kk];
            int step_kk = (int)sh_steps[warp_local][kk];
            idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
            if (kk == end) prefix_endplus1 = idx;
          }
          int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

          sh_end[warp_local] = end;
          sh_start[warp_local] = start;
          sh_q_start[warp_local] = q_start;
          sh_q_mid[warp_local] = q_mid;
          sh_q_end[warp_local] = q_end;
          sh_node_end_target[warp_local] = node_end_target;
          sh_prefix_offset[warp_local] = prefix_offset;
          sh_suffix_offset[warp_local] = suffix_offset;
          sh_csf_idx[warp_local] = csf_idx;

          int split_levels = end - start;
          if (split_levels > SPLIT_LEVELS_T) split_levels = SPLIT_LEVELS_T;
          if (split_levels < 0) split_levels = 0;

          int8_t cur_k[kMaxSeeds];
          int32_t cur_node[kMaxSeeds];
          CoeffT cur_w[kMaxSeeds];
          int32_t cur_seg[kMaxSeeds];
          int cur_count = 1;
          cur_k[0] = (int8_t)start;
          cur_node[0] = node_start;
          cur_w[0] = CoeffT(1.0);
          cur_seg[0] = 0;

          for (int depth = 0; depth < split_levels; depth++) {
            int8_t next_k[kMaxSeeds];
            int32_t next_node[kMaxSeeds];
            CoeffT next_w[kMaxSeeds];
            int32_t next_seg[kMaxSeeds];
            int next_count = 0;

            for (int si = 0; si < cur_count; si++) {
              int kpos = (int)cur_k[si];
              int node_k = (int)cur_node[si];
              CoeffT w = cur_w[si];
              int32_t seg_idx = cur_seg[si];

              int is_first = (kpos == start);
              int is_last = (kpos == end);
              int qk = is_first ? q_start : (is_last ? q_end : q_mid);
              int dk = (int)sh_steps[warp_local][kpos];
              int bk = (int)node_twos[sh_nodes[warp_local][kpos + 1]];
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
                CoeffT seg = segment_value_int_t<CoeffT>(qk, dprime, dk, db, bk);
                if (seg == 0.0) continue;
                if (next_count >= kMaxSeeds) {
                  sh_status[warp_local] = 1;
                  atomicExch(overflow_flag, 1);
                  break;
                }
                next_k[next_count] = (int8_t)k_next;
                next_node[next_count] = child_k;
                next_w[next_count] = w * seg;
                next_seg[next_count] = (int32_t)((int64_t)seg_idx + child_prefix[node_k * 5 + dprime]);
                next_count++;
              }
              if (sh_status[warp_local] == 1) break;
            }

            if (sh_status[warp_local] == 1) break;
            cur_count = next_count;
            for (int si = 0; si < cur_count; si++) {
              cur_k[si] = next_k[si];
              cur_node[si] = next_node[si];
              cur_w[si] = next_w[si];
              cur_seg[si] = next_seg[si];
            }
            if (cur_count == 0) break;
          }

          if (sh_status[warp_local] == 0) {
            sh_seed_count[warp_local] = cur_count;
            if (cur_count == 0) {
              sh_status[warp_local] = 2;
            } else {
              for (int si = 0; si < cur_count; si++) {
                sh_seed_k[warp_local][si] = cur_k[si];
                sh_seed_node[warp_local][si] = cur_node[si];
                sh_seed_w[warp_local][si] = cur_w[si];
                sh_seed_seg[warp_local][si] = cur_seg[si];
              }
            }
          }
        }
      }
    }
  }
  __syncwarp();

  int status = sh_status[warp_local];
  int end = sh_end[warp_local];
  int start = sh_start[warp_local];
  int q_start = sh_q_start[warp_local];
  int q_mid = sh_q_mid[warp_local];
  int q_end = sh_q_end[warp_local];
  int32_t node_end_target = sh_node_end_target[warp_local];
  int32_t prefix_offset = sh_prefix_offset[warp_local];
  int32_t suffix_offset = sh_suffix_offset[warp_local];
  int csf_idx = sh_csf_idx[warp_local];
  int32_t pq_id = sh_pq_id[warp_local];
  int64_t base = sh_base[warp_local];
  int64_t max_out = sh_max_out[warp_local];

  int local_overflow = (status == 1) ? 1 : 0;
  int nseed = sh_seed_count[warp_local];
  if (status == 0 && lane < nseed) {
    int8_t st_k[MAX_NORB_T];
    int32_t st_node[MAX_NORB_T];
    CoeffT st_w[MAX_NORB_T];
    int32_t st_seg[MAX_NORB_T];
    int top = 0;

    st_k[top] = sh_seed_k[warp_local][lane];
    st_node[top] = sh_seed_node[warp_local][lane];
    st_w[top] = sh_seed_w[warp_local][lane];
    st_seg[top] = sh_seed_seg[warp_local][lane];
    top++;

    while (top) {
      top--;
      int kpos = (int)st_k[top];
      int node_k = (int)st_node[top];
      CoeffT w = st_w[top];
      int32_t seg_idx = st_seg[top];

      int is_last = (kpos == end);
      int qk = (kpos == start) ? q_start : (is_last ? q_end : q_mid);

      int dk = (int)sh_steps[warp_local][kpos];
      int bk = (int)node_twos[sh_nodes[warp_local][kpos + 1]];
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
        CoeffT seg = segment_value_int_t<CoeffT>(qk, dprime, dk, db, bk);
        if (seg == 0.0) continue;
        CoeffT w2 = w * seg;
        int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
        int32_t seg_idx2 = (int32_t)seg_idx2_64;

        if (is_last) {
          if (child_k != node_end_target) continue;
          int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
          int csf_i = (int)csf_i_ll;
          if (csf_i == csf_idx) continue;
          if (w2 == 0.0) continue;

          unsigned long long slot = atomicAdd(&sh_out_count[warp_local], 1ULL);
          if ((int64_t)slot >= max_out) {
            local_overflow = 1;
            continue;
          }
          int64_t out_pos = base + (int64_t)slot;
          out_idx[out_pos] = csf_i;
          out_coeff[out_pos] = w2;
          if (out_task_csf) out_task_csf[out_pos] = csf_idx;
          if (out_task_pq) out_task_pq[out_pos] = (PQ_T)pq_id;
        } else {
          if (top >= MAX_NORB_T) {
            local_overflow = 1;
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          top++;
        }
      }
    }
  }

  unsigned mask = __activemask();
  int any_overflow = __any_sync(mask, local_overflow != 0);
  __syncwarp(mask);
  if (lane == 0) {
    if (any_overflow) atomicExch(overflow_flag, 1);
  }
}

extern "C" void guga_epq_contribs_many_count_allpairs_recompute_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* counts,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_contribs_many_count_allpairs_recompute_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, j_start, j_count, counts, overflow_flag);
  } else {
    epq_contribs_many_count_allpairs_recompute_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, j_start, j_count, counts, overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_write_allpairs_recompute_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* offsets,
    int32_t* out_idx,
    void* out_coeff,
    int out_coeff_type,
    int32_t* out_task_csf,
    void* out_task_pq,
    int out_task_pq_type,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int pq_type = out_task_pq ? out_task_pq_type : 4;
  if (pq_type != 1 && pq_type != 2 && pq_type != 4) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (out_coeff_type != 4 && out_coeff_type != 8) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int blocks = (ntasks + threads - 1) / threads;
  if (out_coeff_type == 4) {
    float* out_coeff_f32 = reinterpret_cast<float*>(out_coeff);
    if (norb <= 32) {
      if (pq_type == 1) {
        epq_contribs_many_write_allpairs_recompute_kernel_t<32, uint8_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint8_t*>(out_task_pq),
            overflow_flag);
      } else if (pq_type == 2) {
        epq_contribs_many_write_allpairs_recompute_kernel_t<32, uint16_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint16_t*>(out_task_pq),
            overflow_flag);
      } else {
        epq_contribs_many_write_allpairs_recompute_kernel_t<32, int32_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<int32_t*>(out_task_pq),
            overflow_flag);
      }
    } else {
      if (pq_type == 1) {
        epq_contribs_many_write_allpairs_recompute_kernel_t<64, uint8_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint8_t*>(out_task_pq),
            overflow_flag);
      } else if (pq_type == 2) {
        epq_contribs_many_write_allpairs_recompute_kernel_t<64, uint16_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint16_t*>(out_task_pq),
            overflow_flag);
      } else {
        epq_contribs_many_write_allpairs_recompute_kernel_t<64, int32_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<int32_t*>(out_task_pq),
            overflow_flag);
      }
    }
    return;
  }

  double* out_coeff_f64 = reinterpret_cast<double*>(out_coeff);
  if (norb <= 32) {
    if (pq_type == 1) {
      epq_contribs_many_write_allpairs_recompute_kernel_t<32, uint8_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint8_t*>(out_task_pq),
          overflow_flag);
    } else if (pq_type == 2) {
      epq_contribs_many_write_allpairs_recompute_kernel_t<32, uint16_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint16_t*>(out_task_pq),
          overflow_flag);
    } else {
      epq_contribs_many_write_allpairs_recompute_kernel_t<32, int32_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<int32_t*>(out_task_pq),
          overflow_flag);
    }
  } else {
    if (pq_type == 1) {
      epq_contribs_many_write_allpairs_recompute_kernel_t<64, uint8_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint8_t*>(out_task_pq),
          overflow_flag);
    } else if (pq_type == 2) {
      epq_contribs_many_write_allpairs_recompute_kernel_t<64, uint16_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint16_t*>(out_task_pq),
          overflow_flag);
    } else {
      epq_contribs_many_write_allpairs_recompute_kernel_t<64, int32_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<int32_t*>(out_task_pq),
          overflow_flag);
    }
  }
}

extern "C" void guga_epq_contribs_many_count_allpairs_recompute_warp_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* counts,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (threads < 32 || (threads % 32) != 0) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int warps_per_block = threads / 32;
  int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  if (norb <= 32) {
    epq_contribs_many_count_allpairs_recompute_warp_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, j_start, j_count, counts, overflow_flag);
  } else {
    epq_contribs_many_count_allpairs_recompute_warp_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, ncsf, norb, j_start, j_count, counts, overflow_flag);
  }
}

extern "C" void guga_epq_contribs_many_write_allpairs_recompute_warp_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const int64_t* offsets,
    int32_t* out_idx,
    void* out_coeff,
    int out_coeff_type,
    int32_t* out_task_csf,
    void* out_task_pq,
    int out_task_pq_type,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) return;
  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll <= 0) return;
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (threads < 32 || (threads % 32) != 0) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int pq_type = out_task_pq ? out_task_pq_type : 4;
  if (pq_type != 1 && pq_type != 2 && pq_type != 4) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  if (out_coeff_type != 4 && out_coeff_type != 8) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int ntasks = (int)ntasks_ll;
  int warps_per_block = threads / 32;
  int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  if (out_coeff_type == 4) {
    float* out_coeff_f32 = reinterpret_cast<float*>(out_coeff);
    if (norb <= 32) {
      if (pq_type == 1) {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, uint8_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint8_t*>(out_task_pq),
            overflow_flag);
      } else if (pq_type == 2) {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, uint16_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint16_t*>(out_task_pq),
            overflow_flag);
      } else {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, int32_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<int32_t*>(out_task_pq),
            overflow_flag);
      }
    } else {
      if (pq_type == 1) {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, uint8_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint8_t*>(out_task_pq),
            overflow_flag);
      } else if (pq_type == 2) {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, uint16_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<uint16_t*>(out_task_pq),
            overflow_flag);
      } else {
        epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, int32_t, float><<<blocks, threads, 0, stream>>>(
            child,
            node_twos,
            child_prefix,
            ncsf,
            norb,
            j_start,
            j_count,
            offsets,
            out_idx,
            out_coeff_f32,
            out_task_csf,
            reinterpret_cast<int32_t*>(out_task_pq),
            overflow_flag);
      }
    }
    return;
  }

  double* out_coeff_f64 = reinterpret_cast<double*>(out_coeff);
  if (norb <= 32) {
    if (pq_type == 1) {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, uint8_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint8_t*>(out_task_pq),
          overflow_flag);
    } else if (pq_type == 2) {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, uint16_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint16_t*>(out_task_pq),
          overflow_flag);
    } else {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<32, int32_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<int32_t*>(out_task_pq),
          overflow_flag);
    }
  } else {
    if (pq_type == 1) {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, uint8_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint8_t*>(out_task_pq),
          overflow_flag);
    } else if (pq_type == 2) {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, uint16_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<uint16_t*>(out_task_pq),
          overflow_flag);
    } else {
      epq_contribs_many_write_allpairs_recompute_warp_kernel_t<64, int32_t, double><<<blocks, threads, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          ncsf,
          norb,
          j_start,
          j_count,
          offsets,
          out_idx,
          out_coeff_f64,
          out_task_csf,
          reinterpret_cast<int32_t*>(out_task_pq),
          overflow_flag);
    }
  }
}

template <int MAX_NORB_T>
__global__ void epq_apply_weighted_many_atomic_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,
    const int32_t* __restrict__ task_p,
    const int32_t* __restrict__ task_q,
    const double* __restrict__ task_wgt,
    const double* __restrict__ task_scale,  // may be null
    int ntasks,
    double* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_idx = task_csf[tid];
  int p = task_p[tid];
  int q = task_q[tid];

  if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb || (unsigned)csf_idx >= (unsigned)ncsf) {
    atomicExch(overflow_flag, 1);
    return;
  }

  double wgt = task_wgt[tid];
  if (wgt == 0.0) return;
  double scale = task_scale ? task_scale[tid] : 1.0;
  if (scale == 0.0) return;

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  if (p == q) {
    int occ_p = step_to_occ(steps[p]);
    if (occ_p) atomicAdd(&y[csf_idx], scale * wgt * (double)occ_p);
    return;
  }

  int occ_p = step_to_occ(steps[p]);
  int occ_q = step_to_occ(steps[q]);
  if (occ_q <= 0 || occ_p >= 2) return;

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

  int32_t idx = 0;
  int32_t prefix_offset = 0;
  int32_t prefix_endplus1 = 0;
  for (int kk = 0; kk <= end; kk++) {
    if (kk == start) prefix_offset = idx;
    int node_kk = nodes[kk];
    int step_kk = (int)steps[kk];
    idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
    if (kk == end) prefix_endplus1 = idx;
  }
  int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];
  int top = 0;

  st_k[top] = (int8_t)start;
  st_node[top] = node_start;
  st_w[top] = 1.0;
  st_seg[top] = 0;
  top++;

  int overflow = 0;

  while (top) {
    top--;
    int kpos = (int)st_k[top];
    int node_k = (int)st_node[top];
    double w = st_w[top];
    int32_t seg_idx = st_seg[top];

    int is_first = (kpos == start);
    int is_last = (kpos == end);
    int qk = is_first ? q_start : (is_last ? q_end : q_mid);

    int dk = (int)steps[kpos];
    int bk = (int)node_twos[nodes[kpos + 1]];
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
      int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
      int32_t seg_idx2 = (int32_t)seg_idx2_64;

      if (is_last) {
        if (child_k != node_end_target) continue;
        int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
        int csf_i = (int)csf_i_ll;
        if ((unsigned)csf_i >= (unsigned)ncsf) {
          overflow = 1;
          continue;
        }
        if (csf_i == csf_idx) continue;
        if (w2 == 0.0) continue;
        atomicAdd(&y[csf_i], scale * wgt * w2);
      } else {
        if (top >= MAX_NORB_T) {
          overflow = 1;
          continue;
        }
        st_k[top] = (int8_t)k_next;
        st_node[top] = child_k;
        st_w[top] = w2;
        st_seg[top] = seg_idx2;
        top++;
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
}

// Atomics-free destination-gather apply of a single generator:
//   Y[i,:] += alpha * sum_j <j|E_qp|i> * X[j,:]
// which equals applying E_pq in the usual y = E_pq x sense for real CSFs.
//
// Notes
// -----
// - We enumerate reachable `j` from each destination `i` (apply E_qp to |i>) and then gather X[j,:].
// - This kernel is tuned for small RHS blocks (nvec<=32): 1 warp per row, lane=vector index.
// - To avoid per-thread stack/register blowup, lane0 enumerates into per-warp shared buffers.
template <int MAX_NORB_T>
__global__ void epq_apply_gather_inplace_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    int p,  // apply E_pq via gather enumeration of E_qp
    int q,
    const double* __restrict__ x,  // [ncsf,nvec] row-major
    int nvec,
    double alpha,
    double* __restrict__ y,  // [ncsf,nvec] row-major
    bool add,
    int* __restrict__ overflow_flag) {
  constexpr int MAX_WARPS_PER_BLOCK = 8;  // requires threads<=256
  constexpr int MAX_OUT_PER_ROW = 256;    // cap on # of emitted (j,coeff) per row; overflow triggers error

  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_global = tid >> 5;
  int lane = tid & 31;
  int warp_in_block = (int)(threadIdx.x >> 5);
  if (warp_global >= ncsf) return;
  if (warp_in_block >= MAX_WARPS_PER_BLOCK) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }
  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_i = warp_global;

  // Each lane maps to a vector index. Keep all lanes active until after the
  // warp sync point to avoid deadlocking `__syncwarp()` when nvec<32.
  bool lane_active = ((unsigned)lane < (unsigned)nvec);

  const int8_t* steps = steps_table + ((int64_t)csf_i * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_i * (int64_t)(norb + 1));

  // Per-warp shared buffers: DP stack (size MAX_NORB_T) + output list (size MAX_OUT_PER_ROW).
  __shared__ int8_t st_k[MAX_WARPS_PER_BLOCK][MAX_NORB_T];
  __shared__ int32_t st_node[MAX_WARPS_PER_BLOCK][MAX_NORB_T];
  __shared__ double st_w[MAX_WARPS_PER_BLOCK][MAX_NORB_T];
  __shared__ int32_t st_seg[MAX_WARPS_PER_BLOCK][MAX_NORB_T];
  __shared__ int32_t out_idx[MAX_WARPS_PER_BLOCK][MAX_OUT_PER_ROW];
  __shared__ double out_coeff[MAX_WARPS_PER_BLOCK][MAX_OUT_PER_ROW];
  __shared__ int out_count[MAX_WARPS_PER_BLOCK];
  __shared__ int out_overflow[MAX_WARPS_PER_BLOCK];

  if (lane == 0) {
    out_count[warp_in_block] = 0;
    out_overflow[warp_in_block] = 0;

    // Enumerate E_qp on |i> so we can gather <j|E_qp|i> == <i|E_pq|j>.
    int p2 = q;
    int q2 = p;

    if ((unsigned)p2 >= (unsigned)norb || (unsigned)q2 >= (unsigned)norb) {
      out_overflow[warp_in_block] = 1;
    } else if (p2 == q2) {
      int occ_p = step_to_occ(steps[p2]);
      if (occ_p) {
        out_idx[warp_in_block][0] = csf_i;
        out_coeff[warp_in_block][0] = (double)occ_p;
        out_count[warp_in_block] = 1;
      }
    } else {
      int occ_p = step_to_occ(steps[p2]);
      int occ_q = step_to_occ(steps[q2]);
      if (!(occ_q <= 0 || occ_p >= 2)) {
        int start;
        int end;
        int q_start;
        int q_mid;
        int q_end;
        if (p2 < q2) {
          start = p2;
          end = q2;
          q_start = Q_uR;
          q_mid = Q_R;
          q_end = Q_oR;
        } else {
          start = q2;
          end = p2;
          q_start = Q_uL;
          q_mid = Q_L;
          q_end = Q_oL;
        }

        int32_t node_start = nodes[start];
        int32_t node_end_target = nodes[end + 1];

        int32_t idx = 0;
        int32_t prefix_offset = 0;
        int32_t prefix_endplus1 = 0;
        for (int kk = 0; kk <= end; kk++) {
          if (kk == start) prefix_offset = idx;
          int node_kk = nodes[kk];
          int step_kk = (int)steps[kk];
          idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
          if (kk == end) prefix_endplus1 = idx;
        }
        int32_t suffix_offset = (int32_t)csf_i - prefix_endplus1;

        int top = 0;
        st_k[warp_in_block][top] = (int8_t)start;
        st_node[warp_in_block][top] = node_start;
        st_w[warp_in_block][top] = 1.0;
        st_seg[warp_in_block][top] = 0;
        top++;

        while (top) {
          top--;
          int kpos = (int)st_k[warp_in_block][top];
          int node_k = (int)st_node[warp_in_block][top];
          double w = st_w[warp_in_block][top];
          int32_t seg_idx = st_seg[warp_in_block][top];

          int is_first = (kpos == start);
          int is_last = (kpos == end);
          int qk = is_first ? q_start : (is_last ? q_end : q_mid);

          int dk = (int)steps[kpos];
          int bk = (int)node_twos[nodes[kpos + 1]];
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
            int64_t seg_idx2_64 = (int64_t)seg_idx + child_prefix[node_k * 5 + dprime];
            int32_t seg_idx2 = (int32_t)seg_idx2_64;

            if (is_last) {
              if (child_k != node_end_target) continue;
              int64_t csf_j_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
              int csf_j = (int)csf_j_ll;
              if ((unsigned)csf_j >= (unsigned)ncsf) {
                out_overflow[warp_in_block] = 1;
                continue;
              }
              if (csf_j == csf_i) continue;
              if (w2 == 0.0) continue;
              int pos = out_count[warp_in_block];
              if (pos >= MAX_OUT_PER_ROW) {
                out_overflow[warp_in_block] = 1;
                break;
              }
              out_idx[warp_in_block][pos] = csf_j;
              out_coeff[warp_in_block][pos] = w2;
              out_count[warp_in_block] = pos + 1;
            } else {
              if (top >= MAX_NORB_T) {
                out_overflow[warp_in_block] = 1;
                continue;
              }
              st_k[warp_in_block][top] = (int8_t)k_next;
              st_node[warp_in_block][top] = child_k;
              st_w[warp_in_block][top] = w2;
              st_seg[warp_in_block][top] = seg_idx2;
              top++;
            }
          }
          if (out_overflow[warp_in_block]) break;
        }
      }
    }
  }

  __syncwarp();

  if (out_overflow[warp_in_block]) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  if (!lane_active) return;

  double acc = 0.0;
  int cnt = out_count[warp_in_block];
  for (int t = 0; t < cnt; t++) {
    int j = out_idx[warp_in_block][t];
    double c = out_coeff[warp_in_block][t];
    acc += c * x[(int64_t)j * (int64_t)nvec + (int64_t)lane];
  }

  double contrib = alpha * acc;
  int64_t y_off = (int64_t)csf_i * (int64_t)nvec + (int64_t)lane;
  if (add) {
    y[y_off] += contrib;
  } else {
    y[y_off] = contrib;
  }
}

extern "C" void guga_epq_apply_gather_inplace_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int p,
    int q,
    const double* x,
    int nvec,
    double alpha,
    double* y,
    int add,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (ncsf <= 0 || nvec <= 0) return;
  if (threads <= 0) return;
  if ((threads & 31) != 0) return;
  if (threads > 256) {
    if (overflow_flag) cudaMemsetAsync(overflow_flag, 1, sizeof(int), stream);
    return;
  }
  int warps_per_block = threads >> 5;
  if (warps_per_block <= 0) return;
  int blocks = (ncsf + warps_per_block - 1) / warps_per_block;
  if (norb <= 32) {
    epq_apply_gather_inplace_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, p, q, x, nvec, alpha, y, (bool)add,
        overflow_flag);
  } else {
    epq_apply_gather_inplace_kernel_t<64><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, p, q, x, nvec, alpha, y, (bool)add,
        overflow_flag);
  }
}

extern "C" void guga_epq_apply_weighted_many_atomic_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    const double* task_wgt,
    const double* task_scale,
    int ntasks,
    double* y,
    int* overflow_flag,
    int threads) {
  int blocks = (ntasks + threads - 1) / threads;
  if (norb <= 32) {
    epq_apply_weighted_many_atomic_kernel_t<32><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        task_wgt,
        task_scale,
        ntasks,
        y,
        overflow_flag);
  } else {
    epq_apply_weighted_many_atomic_kernel_t<64><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_p,
        task_q,
        task_wgt,
        task_scale,
        ntasks,
        y,
        overflow_flag);
  }
}

template <int MAX_NORB_T>
__global__ void build_t_block_epq_atomic_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const double* __restrict__ c,         // [ncsf]
    const int32_t* __restrict__ p_list,   // [nops_block]
    const int32_t* __restrict__ q_list,   // [nops_block]
    int nops_block,
    double* __restrict__ out,  // [nops_block, out_stride] row-major
    int64_t out_stride,        // elements
    int* __restrict__ overflow_flag) {
  int csf_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (csf_idx >= ncsf) return;

  if (norb > MAX_NORB_T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double cj = c[csf_idx];
  if (cj == 0.0) return;

  const int8_t* steps = steps_table + ((int64_t)csf_idx * (int64_t)norb);
  const int32_t* nodes = nodes_table + ((int64_t)csf_idx * (int64_t)(norb + 1));

  int8_t st_k[MAX_NORB_T];
  int32_t st_node[MAX_NORB_T];
  double st_w[MAX_NORB_T];
  int32_t st_seg[MAX_NORB_T];

  int overflow = 0;

  for (int op = 0; op < nops_block; op++) {
    int p = p_list[op];
    int q = q_list[op];

    if ((unsigned)p >= (unsigned)norb || (unsigned)q >= (unsigned)norb) {
      overflow = 1;
      continue;
    }

    double* out_row = out + (int64_t)op * out_stride;

    if (p == q) {
      int occ_p = step_to_occ(steps[p]);
      if (occ_p) out_row[csf_idx] += (double)occ_p * cj;
      continue;
    }

    int occ_p = step_to_occ(steps[p]);
    int occ_q = step_to_occ(steps[q]);
    if (occ_q <= 0 || occ_p >= 2) continue;

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

    int32_t idx = 0;
    int32_t prefix_offset = 0;
    int32_t prefix_endplus1 = 0;
    for (int kk = 0; kk <= end; kk++) {
      if (kk == start) prefix_offset = idx;
      int node_kk = nodes[kk];
      int step_kk = (int)steps[kk];
      idx += (int32_t)child_prefix[node_kk * 5 + step_kk];
      if (kk == end) prefix_endplus1 = idx;
    }
    int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

    int top = 0;
    st_k[top] = (int8_t)start;
    st_node[top] = node_start;
    st_w[top] = 1.0;
    st_seg[top] = 0;
    top++;

    while (top) {
      top--;
      int kpos = (int)st_k[top];
      int node_k = (int)st_node[top];
      double w = st_w[top];
      int64_t seg_idx = (int64_t)st_seg[top];

      int is_first = (kpos == start);
      int is_last = (kpos == end);
      int qk = is_first ? q_start : (is_last ? q_end : q_mid);

      int dk = (int)steps[kpos];
      int bk = (int)node_twos[nodes[kpos + 1]];
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
        int64_t seg_idx2_ll = seg_idx + child_prefix[node_k * 5 + dprime];
        if (seg_idx2_ll > (int64_t)0x7fffffffLL) {
          overflow = 1;
          continue;
        }
        int32_t seg_idx2 = (int32_t)seg_idx2_ll;

        if (is_last) {
          if (child_k != node_end_target) continue;
          int64_t csf_i_ll = (int64_t)prefix_offset + (int64_t)seg_idx2 + (int64_t)suffix_offset;
          int csf_i = (int)csf_i_ll;
          if ((unsigned)csf_i >= (unsigned)ncsf) {
            overflow = 1;
            continue;
          }
          if (csf_i == csf_idx) continue;
          if (w2 == 0.0) continue;
          atomicAdd(&out_row[csf_i], cj * w2);
        } else {
          if (top >= MAX_NORB_T) {
            overflow = 1;
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          top++;
        }
      }
    }
  }

  if (overflow) atomicExch(overflow_flag, 1);
}

extern "C" void guga_build_t_block_epq_atomic_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const double* c,
    const int32_t* p_list,
    const int32_t* q_list,
    int nops_block,
    double* out,
    int64_t out_stride,
    int* overflow_flag,
    int threads) {
  if (ncsf <= 0 || nops_block <= 0) return;
  int blocks = (ncsf + threads - 1) / threads;
  if (norb <= 32) {
    build_t_block_epq_atomic_kernel_t<32><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        c,
        p_list,
        q_list,
        nops_block,
        out,
        out_stride,
        overflow_flag);
  } else {
    build_t_block_epq_atomic_kernel_t<MAX_NORB><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        c,
        p_list,
        q_list,
        nops_block,
        out,
        out_stride,
        overflow_flag);
  }
}

extern "C" void guga_build_t_block_epq_atomic_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const double* c,
    const int32_t* p_list,
    const int32_t* q_list,
    int nops_block,
    double* out,
    int64_t out_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (ncsf <= 0 || nops_block <= 0) return;
  int blocks = (ncsf + threads - 1) / threads;
  if (norb <= 32) {
    build_t_block_epq_atomic_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        c,
        p_list,
        q_list,
        nops_block,
        out,
        out_stride,
        overflow_flag);
  } else {
    build_t_block_epq_atomic_kernel_t<MAX_NORB><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        c,
        p_list,
        q_list,
        nops_block,
        out,
        out_stride,
        overflow_flag);
  }
}
