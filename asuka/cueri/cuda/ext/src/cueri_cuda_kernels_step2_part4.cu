// Auto-split from cueri_cuda_kernels_step2.cu (part 4/4: KernelERI_ppps_packed_exact_2phase..KernelERI_pppp_multiblock_component_reduce)
// Do not edit — regenerate with split_large_kernels.py

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cueri_cuda_kernels_api.h"
#include "cueri_cuda_contract_fock_warp.cuh"
#include "cueri_cuda_contract_jk_warp.cuh"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPiToFiveHalves = 2.0 * kPi * kPi * 1.772453850905516027298167483341145182;  // 2*pi^(5/2)

// Bridge: gap code from previous part(s) (types/helpers needed here).

__device__ __forceinline__ double warp_reduce_sum(double x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

__device__ __forceinline__ double subwarp8_reduce_sum(double x) {
  x += __shfl_down_sync(0xffffffff, x, 4, 8);
  x += __shfl_down_sync(0xffffffff, x, 2, 8);
  x += __shfl_down_sync(0xffffffff, x, 1, 8);
  return x;
}

__device__ __forceinline__ double block_reduce_sum(double x) {
  __shared__ double shared[32];  // up to 1024 threads
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  x = warp_reduce_sum(x);
  if (lane == 0) shared[wid] = x;
  __syncthreads();

  x = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0;
  if (wid == 0) x = warp_reduce_sum(x);
  return x;
}

__device__ __forceinline__ void accumulate_jk_single_value(
    double val,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    int a,
    int b,
    int c,
    int d,
    bool ab_neq,
    bool cd_neq,
    bool bk_swap,
    double f_ab,
    double f_cd,
    int64_t N) {
  if (val == 0.0) return;
  if (J_mat != nullptr) {
    const double Dcd = D_mat[c * N + d];
    atomicAdd(&J_mat[a * N + b], f_cd * val * Dcd);
    if (ab_neq) atomicAdd(&J_mat[b * N + a], f_cd * val * Dcd);
    if (bk_swap) {
      const double Dab = D_mat[a * N + b];
      atomicAdd(&J_mat[c * N + d], f_ab * val * Dab);
      if (cd_neq) atomicAdd(&J_mat[d * N + c], f_ab * val * Dab);
    }
  }
  if (K_mat != nullptr) {
    atomicAdd(&K_mat[a * N + c], val * D_mat[b * N + d]);
    if (cd_neq) atomicAdd(&K_mat[a * N + d], val * D_mat[b * N + c]);
    if (ab_neq) atomicAdd(&K_mat[b * N + c], val * D_mat[a * N + d]);
    if (ab_neq && cd_neq) atomicAdd(&K_mat[b * N + d], val * D_mat[a * N + c]);
    if (bk_swap) {
      atomicAdd(&K_mat[c * N + a], val * D_mat[d * N + b]);
      if (cd_neq) atomicAdd(&K_mat[d * N + a], val * D_mat[c * N + b]);
      if (ab_neq) atomicAdd(&K_mat[c * N + b], val * D_mat[d * N + a]);
      if (ab_neq && cd_neq) atomicAdd(&K_mat[d * N + b], val * D_mat[c * N + a]);
    }
  }
}

__device__ __forceinline__ void accumulate_fock_single_value(
    double val,
    const double* D_mat,
    double* F_mat,
    int a,
    int b,
    int c,
    int d,
    bool ab_neq,
    bool cd_neq,
    bool bk_swap,
    double f_ab,
    double f_cd,
    int64_t N) {
  if (val == 0.0 || F_mat == nullptr) return;
  const double Dcd = D_mat[c * N + d];
  atomicAdd(&F_mat[a * N + b], f_cd * val * Dcd);
  if (ab_neq) atomicAdd(&F_mat[b * N + a], f_cd * val * Dcd);
  if (bk_swap) {
    const double Dab = D_mat[a * N + b];
    atomicAdd(&F_mat[c * N + d], f_ab * val * Dab);
    if (cd_neq) atomicAdd(&F_mat[d * N + c], f_ab * val * Dab);
  }

  constexpr double alpha = -0.5;
  atomicAdd(&F_mat[a * N + c], alpha * val * D_mat[b * N + d]);
  if (cd_neq) atomicAdd(&F_mat[a * N + d], alpha * val * D_mat[b * N + c]);
  if (ab_neq) atomicAdd(&F_mat[b * N + c], alpha * val * D_mat[a * N + d]);
  if (ab_neq && cd_neq) atomicAdd(&F_mat[b * N + d], alpha * val * D_mat[a * N + c]);
  if (bk_swap) {
    atomicAdd(&F_mat[c * N + a], alpha * val * D_mat[d * N + b]);
    if (cd_neq) atomicAdd(&F_mat[d * N + a], alpha * val * D_mat[c * N + b]);
    if (ab_neq) atomicAdd(&F_mat[c * N + b], alpha * val * D_mat[d * N + a]);
    if (ab_neq && cd_neq) atomicAdd(&F_mat[d * N + b], alpha * val * D_mat[c * N + a]);
  }
}

__device__ __forceinline__ void boys_f0_f1_f2(double T, double& F0, double& F1, double& F2) {
  // Robust for small T: evaluate F2 by series, then get F1/F0 by downward recursion:
  //   F_{m-1} = (2T*F_m + exp(-T)) / (2m-1)
  if (T < 1.0) {
    double term = 1.0;
    double f2 = 0.0;
    // F2(T) = sum_{k>=0} (-T)^k / (k!(2k+5))
    constexpr int kMaxSeries = 60;
    constexpr double kTermTol = 1e-22;
    for (int k = 0; k < kMaxSeries; ++k) {
      f2 += term / static_cast<double>(2 * k + 5);
      term *= -T / static_cast<double>(k + 1);
      if (::fabs(term) < kTermTol) break;
    }
    const double expT = ::exp(-T);
    F2 = f2;
    F1 = (2.0 * T * f2 + expT) / 3.0;
    F0 = 2.0 * T * F1 + expT;
    return;
  }

  const double expT = ::exp(-T);
  F0 = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
  F1 = (F0 - expT) / (2.0 * T);
  F2 = (3.0 * F1 - expT) / (2.0 * T);
}

__device__ __forceinline__ void boys_f0_f1_f2_f3_f4(double T, double& F0, double& F1, double& F2, double& F3, double& F4) {
  // Robust for small T: evaluate F4 by series, then get F3..F0 by downward recursion:
  //   F_{m-1} = (2T*F_m + exp(-T)) / (2m-1)
  if (T < 1.0) {
    double term = 1.0;
    double f4 = 0.0;
    // F4(T) = sum_{k>=0} (-T)^k / (k!(2k+9))
    constexpr int kMaxSeries = 60;
    constexpr double kTermTol = 1e-22;
    for (int k = 0; k < kMaxSeries; ++k) {
      f4 += term / static_cast<double>(2 * k + 9);
      term *= -T / static_cast<double>(k + 1);
      if (::fabs(term) < kTermTol) break;
    }
    const double expT = ::exp(-T);
    F4 = f4;
    F3 = (2.0 * T * F4 + expT) / 7.0;
    F2 = (2.0 * T * F3 + expT) / 5.0;
    F1 = (2.0 * T * F2 + expT) / 3.0;
    F0 = 2.0 * T * F1 + expT;
    return;
  }

  const double expT = ::exp(-T);
  F0 = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
  F1 = (F0 - expT) / (2.0 * T);
  F2 = (3.0 * F1 - expT) / (2.0 * T);
  F3 = (5.0 * F2 - expT) / (2.0 * T);
  F4 = (7.0 * F3 - expT) / (2.0 * T);
}

__device__ __forceinline__ void boys_f0_f1(double T, double& F0, double& F1) {
  if (T < 1.0) {
    double term = 1.0;
    double f1 = 0.0;
    constexpr int kMaxSeries = 60;
    constexpr double kTermTol = 1e-22;
    for (int k = 0; k < kMaxSeries; ++k) {
      f1 += term / static_cast<double>(2 * k + 3);
      term *= -T / static_cast<double>(k + 1);
      if (::fabs(term) < kTermTol) break;
    }
    const double expT = ::exp(-T);
    F1 = f1;
    F0 = 2.0 * T * F1 + expT;
    return;
  }
  const double expT = ::exp(-T);
  F0 = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
  F1 = (F0 - expT) / (2.0 * T);
}

__device__ inline double t3_component(int i, int j, int k, const double* d, double term_f2, double term_f3) {
  const double di = d[i];
  const double dj = d[j];
  const double dk = d[k];
  const double t_f2 = term_f2 * ((i == k ? dj : 0.0) + (j == k ? di : 0.0) + (i == j ? dk : 0.0));
  const double t_f3 = term_f3 * (di * dj * dk);
  return t_f2 + t_f3;
}

__device__ inline double t4_component(
    int i, int j, int k, int l, const double* d, double term_f2, double term_f3, double term_f4) {
  const double di = d[i];
  const double dj = d[j];
  const double dk = d[k];
  const double dl = d[l];

  const double t_f2 = term_f2 * ((i == j && k == l) + (i == k && j == l) + (i == l && j == k));
  const double t_f3 = term_f3 *
                      ((i == j ? dk * dl : 0.0) + (i == k ? dj * dl : 0.0) + (i == l ? dj * dk : 0.0) +
                       (j == k ? di * dl : 0.0) + (j == l ? di * dk : 0.0) + (k == l ? di * dj : 0.0));
  const double t_f4 = term_f4 * (di * dj * dk * dl);
  return t_f2 + t_f3 + t_f4;
}



























// ---------------------------------------------------------------------------
// Fused ERI->Fock kernels for dominant SPD classes.
//
// These kernels evaluate contracted ERIs in-register (or via warp reductions),
// write the compact tile into shared memory, then immediately contract into
// RHF Fock (F += J - 0.5*K) using the warp-reduced contraction routine.
// Eliminates the global-memory tile round-trip and the extra contraction launch.
// ---------------------------------------------------------------------------








// ---------------------------------------------------------------------------
// Flat kernels: 1 thread per task, no shared memory / block reduction.
// For low-ncomp classes (ncomp <= 9), this is much faster than block kernels
// because it avoids wasting 112/128 threads per task.
// ---------------------------------------------------------------------------








// ---------------------------------------------------------------------------
// Component-warp helpers for ppps / pppp (analytical Boys classes)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void decode_ppps_comp_cw(int comp, int& ia, int& ib, int& ic) {
  ia = comp / 9; comp -= ia * 9; ib = comp / 3; ic = comp - ib * 3;
}

__device__ __forceinline__ void decode_pppp_comp_cw(int comp, int& ia, int& ib, int& ic, int& id) {
  ia = comp / 27; comp -= ia * 27;
  ib = comp / 9;  comp -= ib * 9;
  ic = comp / 3;  id = comp - ic * 3;
}

__device__ __forceinline__ double lane0_bcast_cw(double x) {
  return __shfl_sync(0xffffffffu, x, 0);
}

__device__ __forceinline__ double lane0_bcast_hw16(double x, unsigned int mask) {
  return __shfl_sync(mask, x, 0, 16);
}

__device__ __forceinline__ void lane0_bcast3_cw(double (&v)[3]) {
#pragma unroll
  for (int i = 0; i < 3; ++i) v[i] = __shfl_sync(0xffffffffu, v[i], 0);
}

__device__ __forceinline__ void lane0_bcast3_hw16(double (&v)[3], unsigned int mask) {
#pragma unroll
  for (int i = 0; i < 3; ++i) v[i] = __shfl_sync(mask, v[i], 0, 16);
}

__device__ __forceinline__ void lane0_bcast33_cw(double (&m)[3][3]) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      m[i][j] = __shfl_sync(0xffffffffu, m[i][j], 0);
}

__device__ __forceinline__ void lane0_bcast33_hw16(double (&m)[3][3], unsigned int mask) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      m[i][j] = __shfl_sync(mask, m[i][j], 0, 16);
}

struct PPPSCommon_cw {
  double I, p, q;
  double Jp[3], Jq[3];
  double Kp[3][3], L[3][3];
  double PA[3], PB[3], QC[3], dvec[3];
  double term_t3_f2, term_t3_f3;
};

struct PPPPCommon_cw {
  double I, p, q;
  double Jp[3], Jq[3];
  double Kp[3][3], Kq[3][3], L[3][3];
  double PA[3], PB[3], QC[3], QD[3], dvec[3];
  double term_t3_f2, term_t3_f3;
  double term_t4_f2, term_t4_f3, term_t4_f4;
};

struct PairTileEntry_cw {
  double eta;
  double Px;
  double Py;
  double Pz;
  double cK;
};

constexpr int kPairTileAB_cw = 8;
constexpr int kPairTileCD_cw = 8;

// Two tasks per warp; each half-warp covers all 27 outputs in 2 phases (16 + 11).
// Lane 0 in each half-warp computes pair-product invariants and broadcasts within width 16.

template <int WARPS_PER_BLOCK, int NPAIR_AB, int NPAIR_CD>
__global__ void KernelERI_ppps_packed_exact_2phase(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ out_task_idx,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  static_assert(WARPS_PER_BLOCK >= 1, "");
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int half = lane >> 4;
  const int lane16 = lane & 15;
  const int t = static_cast<int>(blockIdx.x) * (WARPS_PER_BLOCK * 2) + warp * 2 + half;
  if (t >= ntasks) return;

  (void)sp_npair;

  const unsigned int mask = (half == 0) ? 0x0000ffffu : 0xffff0000u;
  const int comp0 = lane16;
  const int comp1 = 16 + lane16;
  const bool active0 = comp0 < 27;
  const bool active1 = comp1 < 27;
  int ia0 = 0, ib0 = 0, ic0 = 0;
  int ia1 = 0, ib1 = 0, ic1 = 0;
  if (active0) decode_ppps_comp_cw(comp0, ia0, ib0, ic0);
  if (active1) decode_ppps_comp_cw(comp1, ia1, ib1, ic1);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);

  double acc0 = 0.0;
  double acc1 = 0.0;

#pragma unroll
  for (int ii = 0; ii < NPAIR_AB; ++ii) {
    const int ki = baseAB + ii;
#pragma unroll
    for (int jj = 0; jj < NPAIR_CD; ++jj) {
      const int kj = baseCD + jj;
      PPPSCommon_cw c;

      if (lane16 == 0) {
        c.p = pair_eta[ki];
        c.q = pair_eta[kj];
        const double Px = pair_Px[ki], Py = pair_Py[ki], Pz = pair_Pz[ki];
        const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
        c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
        c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
        c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
        c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
        const double PQ2 = c.dvec[0] * c.dvec[0] + c.dvec[1] * c.dvec[1] + c.dvec[2] * c.dvec[2];
        const double denom = c.p + c.q;
        const double omega = c.p * c.q / denom;
        const double T = omega * PQ2;
        const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
        const double base = pref * pair_cK[ki] * pair_cK[kj];
        double F0, F1, F2, F3, F4;
        boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
        (void)F4;
        c.I = base * F0;
        const double omega_over_p = omega / c.p;
        const double omega_over_q = omega / c.q;
        c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
        c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
        c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
        c.Jq[0] = omega_over_q * base * F1 * c.dvec[0];
        c.Jq[1] = omega_over_q * base * F1 * c.dvec[1];
        c.Jq[2] = omega_over_q * base * F1 * c.dvec[2];
        const double w2 = omega * omega;
        const double w3 = w2 * omega;
        const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
        const double inv4pq = 1.0 / (4.0 * c.p * c.q);
        const double t4 = 4.0 * w2 * F2;
        const double t2 = 2.0 * omega * F1;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
#pragma unroll
          for (int b = 0; b < 3; ++b) {
            const double dij = (a == b) ? 1.0 : 0.0;
            const double H = base * (t4 * c.dvec[a] * c.dvec[b] - (a == b ? t2 : 0.0));
            c.Kp[a][b] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
            c.L[a][b] = -H * inv4pq;
          }
        }
        c.term_t3_f2 = 4.0 * w2 * base * F2;
        c.term_t3_f3 = -8.0 * w3 * base * F3;
      }

      c.I = lane0_bcast_hw16(c.I, mask);
      c.p = lane0_bcast_hw16(c.p, mask);
      c.q = lane0_bcast_hw16(c.q, mask);
      lane0_bcast3_hw16(c.Jp, mask);
      lane0_bcast3_hw16(c.Jq, mask);
      lane0_bcast3_hw16(c.PA, mask);
      lane0_bcast3_hw16(c.PB, mask);
      lane0_bcast3_hw16(c.QC, mask);
      lane0_bcast3_hw16(c.dvec, mask);
      lane0_bcast33_hw16(c.Kp, mask);
      lane0_bcast33_hw16(c.L, mask);
      c.term_t3_f2 = lane0_bcast_hw16(c.term_t3_f2, mask);
      c.term_t3_f3 = lane0_bcast_hw16(c.term_t3_f3, mask);

      if (active0) {
        const double a = c.PA[ia0];
        const double b = c.PB[ib0];
        const double cc = c.QC[ic0];
        const double dij = (ia0 == ib0) ? 1.0 : 0.0;
        const double T3 = t3_component(ia0, ib0, ic0, c.dvec, c.term_t3_f2, c.term_t3_f3);
        const double M_ijk = (-T3 + 4.0 * c.p * c.q * dij * c.Jq[ic0]) / (8.0 * c.p * c.p * c.q);
        acc0 += M_ijk + cc * c.Kp[ia0][ib0]
              + b * c.L[ia0][ic0] + b * cc * c.Jp[ia0]
              + a * c.L[ib0][ic0] + a * cc * c.Jp[ib0]
              + a * b * c.Jq[ic0] + a * b * cc * c.I;
      }

      if (active1) {
        const double a = c.PA[ia1];
        const double b = c.PB[ib1];
        const double cc = c.QC[ic1];
        const double dij = (ia1 == ib1) ? 1.0 : 0.0;
        const double T3 = t3_component(ia1, ib1, ic1, c.dvec, c.term_t3_f2, c.term_t3_f3);
        const double M_ijk = (-T3 + 4.0 * c.p * c.q * dij * c.Jq[ic1]) / (8.0 * c.p * c.p * c.q);
        acc1 += M_ijk + cc * c.Kp[ia1][ib1]
              + b * c.L[ia1][ic1] + b * cc * c.Jp[ia1]
              + a * c.L[ib1][ic1] + a * cc * c.Jp[ib1]
              + a * b * c.Jq[ic1] + a * b * cc * c.I;
      }
    }
  }

  const int out_t = out_task_idx ? static_cast<int>(out_task_idx[t]) : t;
  const int64_t base_out = static_cast<int64_t>(out_t) * 27;
  if (active0) eri_out[base_out + comp0] = acc0;
  if (active1) eri_out[base_out + comp1] = acc1;
}

__device__ __forceinline__ void accumulate_ppps_components_exact(
    double* __restrict__ acc,
    const double (&PA)[3],
    const double (&PB)[3],
    const double (&QC)[3],
    const double (&dvec)[3],
    double p,
    double q,
    double I,
    const double (&Jp)[3],
    const double (&Jq)[3],
    const double (&Kp)[3][3],
    const double (&L)[3][3],
    double term_t3_f2,
    double term_t3_f3) {
#pragma unroll
  for (int ia = 0; ia < 3; ++ia) {
#pragma unroll
    for (int ib = 0; ib < 3; ++ib) {
      const double a = PA[ia];
      const double b = PB[ib];
      const double dij = (ia == ib) ? 1.0 : 0.0;
#pragma unroll
      for (int ic = 0; ic < 3; ++ic) {
        const double cc = QC[ic];
        const double T3 = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
        const double M_ijk = (-T3 + 4.0 * p * q * dij * Jq[ic]) / (8.0 * p * p * q);
        acc[ia * 9 + ib * 3 + ic] += M_ijk + cc * Kp[ia][ib]
          + b * L[ia][ic] + b * cc * Jp[ia]
          + a * L[ib][ic] + a * cc * Jp[ib]
          + a * b * Jq[ic] + a * b * cc * I;
      }
    }
  }
}

template <int NPAIR_CD>
struct PPPSExactCDPreload {
  double q[NPAIR_CD];
  double Qx[NPAIR_CD];
  double Qy[NPAIR_CD];
  double Qz[NPAIR_CD];
  double cK[NPAIR_CD];
  double QC[NPAIR_CD][3];
};

template <int NPAIR_AB>
struct PPPSExactABPreload {
  double p[NPAIR_AB];
  double Px[NPAIR_AB];
  double Py[NPAIR_AB];
  double Pz[NPAIR_AB];
  double cK[NPAIR_AB];
  double PA[NPAIR_AB][3];
  double PB[NPAIR_AB][3];
};

template <int NPAIR_AB, int NPAIR_CD>
__global__ void KernelERI_ppps_packed_exact_cdpreload(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ out_task_idx,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  (void)sp_npair;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);

  PPPSExactCDPreload<NPAIR_CD> cd;
#pragma unroll
  for (int jj = 0; jj < NPAIR_CD; ++jj) {
    const int kj = baseCD + jj;
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    cd.q[jj] = pair_eta[kj];
    cd.Qx[jj] = Qx;
    cd.Qy[jj] = Qy;
    cd.Qz[jj] = Qz;
    cd.cK[jj] = pair_cK[kj];
    cd.QC[jj][0] = Qx - Cx;
    cd.QC[jj][1] = Qy - Cy;
    cd.QC[jj][2] = Qz - Cz;
  }

  double acc[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) acc[i] = 0.0;

#pragma unroll
  for (int ii = 0; ii < NPAIR_AB; ++ii) {
    const int ki = baseAB + ii;
    const double p = pair_eta[ki];
    const double cKab = pair_cK[ki];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double PA[3] = {Px - Ax, Py - Ay, Pz - Az};
    const double PB[3] = {Px - Bx, Py - By, Pz - Bz};

#pragma unroll
    for (int jj = 0; jj < NPAIR_CD; ++jj) {
      const double q = cd.q[jj];
      const double dx = Px - cd.Qx[jj];
      const double dy = Py - cd.Qy[jj];
      const double dz = Pz - cd.Qz[jj];
      const double dvec[3] = {dx, dy, dz};
      const double PQ2 = dx * dx + dy * dy + dz * dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKab * cd.cK[jj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {
          -omega_over_p * base * F1 * dx,
          -omega_over_p * base * F1 * dy,
          -omega_over_p * base * F1 * dz,
      };
      const double Jq[3] = {
          omega_over_q * base * F1 * dx,
          omega_over_q * base * F1 * dy,
          omega_over_q * base * F1 * dz,
      };
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double Kp[3][3];
      double L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          const double H = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -H * inv4pq;
        }
      }
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
      accumulate_ppps_components_exact(
          acc, PA, PB, cd.QC[jj], dvec, p, q, I, Jp, Jq, Kp, L, term_t3_f2, term_t3_f3);
    }
  }

  const int out_t = out_task_idx ? static_cast<int>(out_task_idx[t]) : t;
  const int64_t base_out = static_cast<int64_t>(out_t) * 27;
#pragma unroll
  for (int i = 0; i < 27; ++i) eri_out[base_out + i] = acc[i];
}

template <int NPAIR_AB, int NPAIR_CD>
__global__ void KernelERI_ppps_packed_exact_abpreload(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ out_task_idx,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  (void)sp_npair;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);

  PPPSExactABPreload<NPAIR_AB> ab;
#pragma unroll
  for (int ii = 0; ii < NPAIR_AB; ++ii) {
    const int ki = baseAB + ii;
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    ab.p[ii] = pair_eta[ki];
    ab.Px[ii] = Px;
    ab.Py[ii] = Py;
    ab.Pz[ii] = Pz;
    ab.cK[ii] = pair_cK[ki];
    ab.PA[ii][0] = Px - Ax;
    ab.PA[ii][1] = Py - Ay;
    ab.PA[ii][2] = Pz - Az;
    ab.PB[ii][0] = Px - Bx;
    ab.PB[ii][1] = Py - By;
    ab.PB[ii][2] = Pz - Bz;
  }

  double acc[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) acc[i] = 0.0;

#pragma unroll
  for (int jj = 0; jj < NPAIR_CD; ++jj) {
    const int kj = baseCD + jj;
    const double q = pair_eta[kj];
    const double cKcd = pair_cK[kj];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double QC[3] = {Qx - Cx, Qy - Cy, Qz - Cz};

#pragma unroll
    for (int ii = 0; ii < NPAIR_AB; ++ii) {
      const double p = ab.p[ii];
      const double dx = ab.Px[ii] - Qx;
      const double dy = ab.Py[ii] - Qy;
      const double dz = ab.Pz[ii] - Qz;
      const double dvec[3] = {dx, dy, dz};
      const double PQ2 = dx * dx + dy * dy + dz * dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * ab.cK[ii] * cKcd;
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {
          -omega_over_p * base * F1 * dx,
          -omega_over_p * base * F1 * dy,
          -omega_over_p * base * F1 * dz,
      };
      const double Jq[3] = {
          omega_over_q * base * F1 * dx,
          omega_over_q * base * F1 * dy,
          omega_over_q * base * F1 * dz,
      };
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double Kp[3][3];
      double L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          const double H = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -H * inv4pq;
        }
      }
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
      accumulate_ppps_components_exact(
          acc, ab.PA[ii], ab.PB[ii], QC, dvec, p, q, I, Jp, Jq, Kp, L, term_t3_f2, term_t3_f3);
    }
  }

  const int out_t = out_task_idx ? static_cast<int>(out_task_idx[t]) : t;
  const int64_t base_out = static_cast<int64_t>(out_t) * 27;
#pragma unroll
  for (int i = 0; i < 27; ++i) eri_out[base_out + i] = acc[i];
}

template <int NPAIR_AB, int NPAIR_CD>
__global__ void KernelERI_ppps_packed_exact_scalar(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ out_task_idx,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  (void)sp_npair;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);

  double acc[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) acc[i] = 0.0;

#pragma unroll
  for (int ii = 0; ii < NPAIR_AB; ++ii) {
    const int ki = baseAB + ii;
#pragma unroll
    for (int jj = 0; jj < NPAIR_CD; ++jj) {
      const int kj = baseCD + jj;
      const double p = pair_eta[ki];
      const double q = pair_eta[kj];
      const double Px = pair_Px[ki], Py = pair_Py[ki], Pz = pair_Pz[ki];
      const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
      const double PA[3] = {Px - Ax, Py - Ay, Pz - Az};
      const double PB[3] = {Px - Bx, Py - By, Pz - Bz};
      const double QC[3] = {Qx - Cx, Qy - Cy, Qz - Cz};
      const double dvec[3] = {Px - Qx, Py - Qy, Pz - Qz};
      const double PQ2 = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * pair_cK[ki] * pair_cK[kj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {
          -omega_over_p * base * F1 * dvec[0],
          -omega_over_p * base * F1 * dvec[1],
          -omega_over_p * base * F1 * dvec[2],
      };
      const double Jq[3] = {
          omega_over_q * base * F1 * dvec[0],
          omega_over_q * base * F1 * dvec[1],
          omega_over_q * base * F1 * dvec[2],
      };
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double Kp[3][3];
      double L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          const double H = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -H * inv4pq;
        }
      }
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;

#pragma unroll
      for (int ia = 0; ia < 3; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 3; ++ib) {
          const double a = PA[ia];
          const double b = PB[ib];
          const double dij = (ia == ib) ? 1.0 : 0.0;
#pragma unroll
          for (int ic = 0; ic < 3; ++ic) {
            const double cc = QC[ic];
            const double T3 = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
            const double M_ijk = (-T3 + 4.0 * p * q * dij * Jq[ic]) / (8.0 * p * p * q);
            acc[ia * 9 + ib * 3 + ic] += M_ijk + cc * Kp[ia][ib]
              + b * L[ia][ic] + b * cc * Jp[ia]
              + a * L[ib][ic] + a * cc * Jp[ib]
              + a * b * Jq[ic] + a * b * cc * I;
          }
        }
      }
    }
  }

  const int out_t = out_task_idx ? static_cast<int>(out_task_idx[t]) : t;
  const int64_t base_out = static_cast<int64_t>(out_t) * 27;
#pragma unroll
  for (int i = 0; i < 27; ++i) {
    eri_out[base_out + i] = acc[i];
  }
}

template <int NPAIR_AB, int NPAIR_CD>
__global__ void KernelERI_ppps_prepacked_exact_scalar(
    const double* __restrict__ Ax,
    const double* __restrict__ Ay,
    const double* __restrict__ Az,
    const double* __restrict__ Bx,
    const double* __restrict__ By,
    const double* __restrict__ Bz,
    const double* __restrict__ Cx,
    const double* __restrict__ Cy,
    const double* __restrict__ Cz,
    const double* __restrict__ ab_eta,
    const double* __restrict__ ab_Px,
    const double* __restrict__ ab_Py,
    const double* __restrict__ ab_Pz,
    const double* __restrict__ ab_cK,
    const double* __restrict__ cd_eta,
    const double* __restrict__ cd_Qx,
    const double* __restrict__ cd_Qy,
    const double* __restrict__ cd_Qz,
    const double* __restrict__ cd_cK,
    int ntasks,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  const double ax = Ax[t], ay = Ay[t], az = Az[t];
  const double bx = Bx[t], by = By[t], bz = Bz[t];
  const double cx = Cx[t], cy = Cy[t], cz = Cz[t];
  const int ab0 = t * NPAIR_AB;
  const int cd0 = t * NPAIR_CD;

  double acc[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) acc[i] = 0.0;

#pragma unroll
  for (int ii = 0; ii < NPAIR_AB; ++ii) {
    const int abi = ab0 + ii;
    const double p = ab_eta[abi];
    const double Px = ab_Px[abi];
    const double Py = ab_Py[abi];
    const double Pz = ab_Pz[abi];
    const double PA[3] = {Px - ax, Py - ay, Pz - az};
    const double PB[3] = {Px - bx, Py - by, Pz - bz};
    const double cKab = ab_cK[abi];
#pragma unroll
    for (int jj = 0; jj < NPAIR_CD; ++jj) {
      const int cdj = cd0 + jj;
      const double q = cd_eta[cdj];
      const double Qx = cd_Qx[cdj];
      const double Qy = cd_Qy[cdj];
      const double Qz = cd_Qz[cdj];
      const double QC[3] = {Qx - cx, Qy - cy, Qz - cz};
      const double dvec[3] = {Px - Qx, Py - Qy, Pz - Qz};
      const double PQ2 = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKab * cd_cK[cdj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {
          -omega_over_p * base * F1 * dvec[0],
          -omega_over_p * base * F1 * dvec[1],
          -omega_over_p * base * F1 * dvec[2],
      };
      const double Jq[3] = {
          omega_over_q * base * F1 * dvec[0],
          omega_over_q * base * F1 * dvec[1],
          omega_over_q * base * F1 * dvec[2],
      };
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double Kp[3][3];
      double L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          const double H = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -H * inv4pq;
        }
      }
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
#pragma unroll
      for (int ia = 0; ia < 3; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 3; ++ib) {
          const double a = PA[ia];
          const double b = PB[ib];
          const double dij = (ia == ib) ? 1.0 : 0.0;
#pragma unroll
          for (int ic = 0; ic < 3; ++ic) {
            const double cc = QC[ic];
            const double T3 = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
            const double M_ijk = (-T3 + 4.0 * p * q * dij * Jq[ic]) / (8.0 * p * p * q);
            acc[ia * 9 + ib * 3 + ic] += M_ijk + cc * Kp[ia][ib]
              + b * L[ia][ic] + b * cc * Jp[ia]
              + a * L[ib][ic] + a * cc * Jp[ib]
              + a * b * Jq[ic] + a * b * cc * I;
          }
        }
      }
    }
  }

  const int64_t base_out = static_cast<int64_t>(t) * 27;
#pragma unroll
  for (int i = 0; i < 27; ++i) {
    eri_out[base_out + i] = acc[i];
  }
}

// One warp per task; lane i owns output component i (27 active lanes).
// Lane 0 computes all pair-product invariants and broadcasts to the warp.
template <int WARPS_PER_BLOCK>
__global__ void KernelERI_ppps_warp_component(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ out_task_idx,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  static_assert(WARPS_PER_BLOCK >= 1, "");
  const int lane   = static_cast<int>(threadIdx.x) & 31;
  const int warp   = static_cast<int>(threadIdx.x) >> 5;
  const int t      = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (t >= ntasks) return;

  const int comp   = lane;
  const bool active = comp < 27;
  int ia = 0, ib = 0, ic = 0;
  if (active) decode_ppps_comp_cw(comp, ia, ib, ic);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB    = static_cast<int>(sp_npair[spAB]);
  const int nCD    = static_cast<int>(sp_npair[spCD]);
  __shared__ PairTileEntry_cw sh_ab[WARPS_PER_BLOCK][kPairTileAB_cw];
  __shared__ PairTileEntry_cw sh_cd[WARPS_PER_BLOCK][kPairTileCD_cw];
  PairTileEntry_cw* const ab_tile = sh_ab[warp];
  PairTileEntry_cw* const cd_tile = sh_cd[warp];

  double acc = 0.0;

  for (int ab0 = 0; ab0 < nAB; ab0 += kPairTileAB_cw) {
    const int ab_count = (nAB - ab0 < kPairTileAB_cw) ? (nAB - ab0) : kPairTileAB_cw;
    if (lane < ab_count) {
      const int ki = baseAB + ab0 + lane;
      ab_tile[lane].eta = pair_eta[ki];
      ab_tile[lane].Px = pair_Px[ki];
      ab_tile[lane].Py = pair_Py[ki];
      ab_tile[lane].Pz = pair_Pz[ki];
      ab_tile[lane].cK = pair_cK[ki];
    }
    __syncwarp();

    for (int cd0 = 0; cd0 < nCD; cd0 += kPairTileCD_cw) {
      const int cd_count = (nCD - cd0 < kPairTileCD_cw) ? (nCD - cd0) : kPairTileCD_cw;
      if (lane < cd_count) {
        const int kj = baseCD + cd0 + lane;
        cd_tile[lane].eta = pair_eta[kj];
        cd_tile[lane].Px = pair_Px[kj];
        cd_tile[lane].Py = pair_Py[kj];
        cd_tile[lane].Pz = pair_Pz[kj];
        cd_tile[lane].cK = pair_cK[kj];
      }
      __syncwarp();

      for (int ii = 0; ii < ab_count; ++ii) {
        const PairTileEntry_cw ab = ab_tile[ii];
        for (int jj = 0; jj < cd_count; ++jj) {
          const PairTileEntry_cw cd = cd_tile[jj];
      PPPSCommon_cw c;

      if (lane == 0) {
        c.p = ab.eta;
        c.q = cd.eta;
        const double Px = ab.Px, Py = ab.Py, Pz = ab.Pz;
        const double Qx = cd.Px, Qy = cd.Py, Qz = cd.Pz;
        c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
        c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
        c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
        c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
        const double PQ2 = c.dvec[0]*c.dvec[0] + c.dvec[1]*c.dvec[1] + c.dvec[2]*c.dvec[2];
        const double denom = c.p + c.q;
        const double omega = c.p * c.q / denom;
        const double T = omega * PQ2;
        const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
        const double base = pref * ab.cK * cd.cK;
        double F0, F1, F2, F3, F4;
        boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
        (void)F4;
        c.I = base * F0;
        const double omega_over_p = omega / c.p;
        const double omega_over_q = omega / c.q;
        c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
        c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
        c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
        c.Jq[0] =  omega_over_q * base * F1 * c.dvec[0];
        c.Jq[1] =  omega_over_q * base * F1 * c.dvec[1];
        c.Jq[2] =  omega_over_q * base * F1 * c.dvec[2];
        const double w2 = omega * omega;
        const double w3 = w2 * omega;
        const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
        const double inv4pq = 1.0 / (4.0 * c.p * c.q);
        const double t4 = 4.0 * w2 * F2;
        const double t2 = 2.0 * omega * F1;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
#pragma unroll
          for (int b = 0; b < 3; ++b) {
            const double dij = (a == b) ? 1.0 : 0.0;
            const double H = base * (t4 * c.dvec[a] * c.dvec[b] - (a == b ? t2 : 0.0));
            c.Kp[a][b] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
            c.L [a][b] = -H * inv4pq;
          }
        }
        c.term_t3_f2 = 4.0 * w2 * base * F2;
        c.term_t3_f3 = -8.0 * w3 * base * F3;
      }

      c.I          = lane0_bcast_cw(c.I);
      c.p          = lane0_bcast_cw(c.p);
      c.q          = lane0_bcast_cw(c.q);
      lane0_bcast3_cw(c.Jp);
      lane0_bcast3_cw(c.Jq);
      lane0_bcast3_cw(c.PA);
      lane0_bcast3_cw(c.PB);
      lane0_bcast3_cw(c.QC);
      lane0_bcast3_cw(c.dvec);
      lane0_bcast33_cw(c.Kp);
      lane0_bcast33_cw(c.L);
      c.term_t3_f2 = lane0_bcast_cw(c.term_t3_f2);
      c.term_t3_f3 = lane0_bcast_cw(c.term_t3_f3);

      if (active) {
        const double a  = c.PA[ia];
        const double b  = c.PB[ib];
        const double cc = c.QC[ic];
        const double dij = (ia == ib) ? 1.0 : 0.0;
        const double T3   = t3_component(ia, ib, ic, c.dvec, c.term_t3_f2, c.term_t3_f3);
        const double M_ijk = (-T3 + 4.0 * c.p * c.q * dij * c.Jq[ic]) / (8.0 * c.p * c.p * c.q);
        acc += M_ijk + cc * c.Kp[ia][ib]
             + b  * c.L[ia][ic] + b  * cc * c.Jp[ia]
             + a  * c.L[ib][ic] + a  * cc * c.Jp[ib]
             + a  * b  * c.Jq[ic] + a  * b  * cc * c.I;
      }
    }
      }
      __syncwarp();
    }
  }

  if (active) {
    const int out_t = out_task_idx ? static_cast<int>(out_task_idx[t]) : t;
    eri_out[static_cast<int64_t>(out_t) * 27 + comp] = acc;
  }
}

template <int WARPS_PER_BLOCK, bool kToFock>
__global__ void KernelFused_ppps_warp_component(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    const int32_t* __restrict__ shell_ao_start,
    int nao,
    const double* __restrict__ D_mat,
    double* __restrict__ out0_mat,
    double* __restrict__ out1_mat,
    int n_bufs) {
  static_assert(WARPS_PER_BLOCK >= 1, "");
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (t >= ntasks) return;

  const int comp = lane;
  const bool active = comp < 27;
  int ia = 0, ib = 0, ic = 0;
  if (active) decode_ppps_comp_cw(comp, ia, ib, ic);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  __shared__ PairTileEntry_cw sh_ab[WARPS_PER_BLOCK][kPairTileAB_cw];
  __shared__ PairTileEntry_cw sh_cd[WARPS_PER_BLOCK][kPairTileCD_cw];
  PairTileEntry_cw* const ab_tile = sh_ab[warp];
  PairTileEntry_cw* const cd_tile = sh_cd[warp];

  double acc = 0.0;

  for (int ab0 = 0; ab0 < nAB; ab0 += kPairTileAB_cw) {
    const int ab_count = (nAB - ab0 < kPairTileAB_cw) ? (nAB - ab0) : kPairTileAB_cw;
    if (lane < ab_count) {
      const int ki = baseAB + ab0 + lane;
      ab_tile[lane].eta = pair_eta[ki];
      ab_tile[lane].Px = pair_Px[ki];
      ab_tile[lane].Py = pair_Py[ki];
      ab_tile[lane].Pz = pair_Pz[ki];
      ab_tile[lane].cK = pair_cK[ki];
    }
    __syncwarp();

    for (int cd0 = 0; cd0 < nCD; cd0 += kPairTileCD_cw) {
      const int cd_count = (nCD - cd0 < kPairTileCD_cw) ? (nCD - cd0) : kPairTileCD_cw;
      if (lane < cd_count) {
        const int kj = baseCD + cd0 + lane;
        cd_tile[lane].eta = pair_eta[kj];
        cd_tile[lane].Px = pair_Px[kj];
        cd_tile[lane].Py = pair_Py[kj];
        cd_tile[lane].Pz = pair_Pz[kj];
        cd_tile[lane].cK = pair_cK[kj];
      }
      __syncwarp();

      for (int ii = 0; ii < ab_count; ++ii) {
        const PairTileEntry_cw ab = ab_tile[ii];
        for (int jj = 0; jj < cd_count; ++jj) {
          const PairTileEntry_cw cd = cd_tile[jj];
          PPPSCommon_cw c;

          if (lane == 0) {
            c.p = ab.eta;
            c.q = cd.eta;
            const double Px = ab.Px, Py = ab.Py, Pz = ab.Pz;
            const double Qx = cd.Px, Qy = cd.Py, Qz = cd.Pz;
            c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
            c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
            c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
            c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
            const double PQ2 = c.dvec[0] * c.dvec[0] + c.dvec[1] * c.dvec[1] + c.dvec[2] * c.dvec[2];
            const double denom = c.p + c.q;
            const double omega = c.p * c.q / denom;
            const double T = omega * PQ2;
            const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
            const double base = pref * ab.cK * cd.cK;
            double F0, F1, F2, F3, F4;
            boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
            (void)F4;
            c.I = base * F0;
            const double omega_over_p = omega / c.p;
            const double omega_over_q = omega / c.q;
            c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
            c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
            c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
            c.Jq[0] = omega_over_q * base * F1 * c.dvec[0];
            c.Jq[1] = omega_over_q * base * F1 * c.dvec[1];
            c.Jq[2] = omega_over_q * base * F1 * c.dvec[2];
            const double w2 = omega * omega;
            const double w3 = w2 * omega;
            const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
            const double inv4pq = 1.0 / (4.0 * c.p * c.q);
            const double t4 = 4.0 * w2 * F2;
            const double t2 = 2.0 * omega * F1;
#pragma unroll
            for (int a = 0; a < 3; ++a) {
#pragma unroll
              for (int b = 0; b < 3; ++b) {
                const double dij = (a == b) ? 1.0 : 0.0;
                const double H = base * (t4 * c.dvec[a] * c.dvec[b] - (a == b ? t2 : 0.0));
                c.Kp[a][b] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
                c.L[a][b] = -H * inv4pq;
              }
            }
            c.term_t3_f2 = 4.0 * w2 * base * F2;
            c.term_t3_f3 = -8.0 * w3 * base * F3;
          }

          c.I = lane0_bcast_cw(c.I);
          c.p = lane0_bcast_cw(c.p);
          c.q = lane0_bcast_cw(c.q);
          lane0_bcast3_cw(c.Jp);
          lane0_bcast3_cw(c.Jq);
          lane0_bcast3_cw(c.PA);
          lane0_bcast3_cw(c.PB);
          lane0_bcast3_cw(c.QC);
          lane0_bcast3_cw(c.dvec);
          lane0_bcast33_cw(c.Kp);
          lane0_bcast33_cw(c.L);
          c.term_t3_f2 = lane0_bcast_cw(c.term_t3_f2);
          c.term_t3_f3 = lane0_bcast_cw(c.term_t3_f3);

          if (active) {
            const double a = c.PA[ia];
            const double b = c.PB[ib];
            const double cc = c.QC[ic];
            const double dij = (ia == ib) ? 1.0 : 0.0;
            const double T3 = t3_component(ia, ib, ic, c.dvec, c.term_t3_f2, c.term_t3_f3);
            const double M_ijk = (-T3 + 4.0 * c.p * c.q * dij * c.Jq[ic]) / (8.0 * c.p * c.p * c.q);
            acc += M_ijk + cc * c.Kp[ia][ib]
                + b * c.L[ia][ic] + b * cc * c.Jp[ia]
                + a * c.L[ib][ic] + a * cc * c.Jp[ib]
                + a * b * c.Jq[ic] + a * b * cc * c.I;
          }
        }
      }
      __syncwarp();
    }
  }

  if (active) {
    const int A_sh = static_cast<int>(sp_A[spAB]);
    const int B_sh = static_cast<int>(sp_B[spAB]);
    const int C_sh = static_cast<int>(sp_A[spCD]);
    const int D_sh = static_cast<int>(sp_B[spCD]);
    const int a0 = static_cast<int>(shell_ao_start[A_sh]);
    const int b0 = static_cast<int>(shell_ao_start[B_sh]);
    const int c0 = static_cast<int>(shell_ao_start[C_sh]);
    const int d0 = static_cast<int>(shell_ao_start[D_sh]);
    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);
    const bool bk_swap = (spAB != spCD);
    const double f_ab = ab_neq ? 2.0 : 1.0;
    const double f_cd = cd_neq ? 2.0 : 1.0;
    const int64_t N = static_cast<int64_t>(nao);
    const int64_t buf_off = static_cast<int64_t>(t % n_bufs) * N * N;
    const int a = a0 + ia;
    const int b = b0 + ib;
    const int c = c0 + ic;

    if constexpr (kToFock) {
      double* F_mat = out0_mat + buf_off;
      accumulate_fock_single_value(acc, D_mat, F_mat, a, b, c, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    } else {
      double* J_mat = (out0_mat != nullptr) ? out0_mat + buf_off : nullptr;
      double* K_mat = (out1_mat != nullptr) ? out1_mat + buf_off : nullptr;
      accumulate_jk_single_value(acc, D_mat, J_mat, K_mat, a, b, c, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    }
  }
}

// One warp per task; 3 phases cover all 81 pppp components (32 + 32 + 17 active).
// acc0/acc1/acc2 hold components [0..31], [32..63], [64..80] respectively.
template <int WARPS_PER_BLOCK>
__global__ void KernelERI_pppp_warp_component_3phase(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    double* __restrict__ eri_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int t    = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const int D = static_cast<int>(sp_B[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];
  const double Dx = shell_cx[D], Dy = shell_cy[D], Dz = shell_cz[D];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB    = static_cast<int>(sp_npair[spAB]);
  const int nCD    = static_cast<int>(sp_npair[spCD]);
  __shared__ PairTileEntry_cw sh_ab[WARPS_PER_BLOCK][kPairTileAB_cw];
  __shared__ PairTileEntry_cw sh_cd[WARPS_PER_BLOCK][kPairTileCD_cw];
  PairTileEntry_cw* const ab_tile = sh_ab[warp];
  PairTileEntry_cw* const cd_tile = sh_cd[warp];

  double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;

  for (int ab0 = 0; ab0 < nAB; ab0 += kPairTileAB_cw) {
    const int ab_count = (nAB - ab0 < kPairTileAB_cw) ? (nAB - ab0) : kPairTileAB_cw;
    if (lane < ab_count) {
      const int ki = baseAB + ab0 + lane;
      ab_tile[lane].eta = pair_eta[ki];
      ab_tile[lane].Px = pair_Px[ki];
      ab_tile[lane].Py = pair_Py[ki];
      ab_tile[lane].Pz = pair_Pz[ki];
      ab_tile[lane].cK = pair_cK[ki];
    }
    __syncwarp();

    for (int cd0 = 0; cd0 < nCD; cd0 += kPairTileCD_cw) {
      const int cd_count = (nCD - cd0 < kPairTileCD_cw) ? (nCD - cd0) : kPairTileCD_cw;
      if (lane < cd_count) {
        const int kj = baseCD + cd0 + lane;
        cd_tile[lane].eta = pair_eta[kj];
        cd_tile[lane].Px = pair_Px[kj];
        cd_tile[lane].Py = pair_Py[kj];
        cd_tile[lane].Pz = pair_Pz[kj];
        cd_tile[lane].cK = pair_cK[kj];
      }
      __syncwarp();

      for (int ii = 0; ii < ab_count; ++ii) {
        const PairTileEntry_cw ab = ab_tile[ii];
        for (int jj = 0; jj < cd_count; ++jj) {
          const PairTileEntry_cw cd = cd_tile[jj];
      PPPPCommon_cw c;

      if (lane == 0) {
        c.p = ab.eta;
        c.q = cd.eta;
        const double Px = ab.Px, Py = ab.Py, Pz = ab.Pz;
        const double Qx = cd.Px, Qy = cd.Py, Qz = cd.Pz;
        c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
        c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
        c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
        c.QD[0] = Qx - Dx; c.QD[1] = Qy - Dy; c.QD[2] = Qz - Dz;
        c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
        const double PQ2 = c.dvec[0]*c.dvec[0] + c.dvec[1]*c.dvec[1] + c.dvec[2]*c.dvec[2];
        const double denom = c.p + c.q;
        const double omega = c.p * c.q / denom;
        const double T = omega * PQ2;
        const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
        const double base = pref * ab.cK * cd.cK;
        double F0, F1, F2, F3, F4;
        boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
        c.I = base * F0;
        const double omega_over_p = omega / c.p;
        const double omega_over_q = omega / c.q;
        c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
        c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
        c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
        c.Jq[0] =  omega_over_q * base * F1 * c.dvec[0];
        c.Jq[1] =  omega_over_q * base * F1 * c.dvec[1];
        c.Jq[2] =  omega_over_q * base * F1 * c.dvec[2];
        const double w2 = omega * omega;
        const double w3 = w2 * omega;
        const double w4 = w2 * w2;
        const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
        const double inv4q2 = 1.0 / (4.0 * c.q * c.q);
        const double inv4pq = 1.0 / (4.0 * c.p * c.q);
        const double t4 = 4.0 * w2 * F2;
        const double t2 = 2.0 * omega * F1;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
#pragma unroll
          for (int b = 0; b < 3; ++b) {
            const double dij = (a == b) ? 1.0 : 0.0;
            const double H = base * (t4 * c.dvec[a] * c.dvec[b] - (a == b ? t2 : 0.0));
            c.Kp[a][b] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
            c.Kq[a][b] = (H + 2.0 * c.q * c.I * dij) * inv4q2;
            c.L [a][b] = -H * inv4pq;
          }
        }
        c.term_t3_f2 = 4.0 * w2 * base * F2;
        c.term_t3_f3 = -8.0 * w3 * base * F3;
        c.term_t4_f2 = 4.0 * w2 * base * F2;
        c.term_t4_f3 = -8.0 * w3 * base * F3;
        c.term_t4_f4 = 16.0 * w4 * base * F4;
      }

      c.I          = lane0_bcast_cw(c.I);
      c.p          = lane0_bcast_cw(c.p);
      c.q          = lane0_bcast_cw(c.q);
      lane0_bcast3_cw(c.Jp); lane0_bcast3_cw(c.Jq);
      lane0_bcast3_cw(c.PA); lane0_bcast3_cw(c.PB);
      lane0_bcast3_cw(c.QC); lane0_bcast3_cw(c.QD);
      lane0_bcast3_cw(c.dvec);
      lane0_bcast33_cw(c.Kp); lane0_bcast33_cw(c.Kq); lane0_bcast33_cw(c.L);
      c.term_t3_f2 = lane0_bcast_cw(c.term_t3_f2);
      c.term_t3_f3 = lane0_bcast_cw(c.term_t3_f3);
      c.term_t4_f2 = lane0_bcast_cw(c.term_t4_f2);
      c.term_t4_f3 = lane0_bcast_cw(c.term_t4_f3);
      c.term_t4_f4 = lane0_bcast_cw(c.term_t4_f4);

#pragma unroll
      for (int phase = 0; phase < 3; ++phase) {
        const int comp = phase * 32 + lane;
        if (comp < 81) {
          int ia, ib, ic, id;
          decode_pppp_comp_cw(comp, ia, ib, ic, id);
          const double a  = c.PA[ia];
          const double b  = c.PB[ib];
          const double cc = c.QC[ic];
          const double d  = c.QD[id];
          const double dij = (ia == ib) ? 1.0 : 0.0;
          const double dkl = (ic == id) ? 1.0 : 0.0;
          const double T3_ijk  = t3_component(ia, ib, ic, c.dvec, c.term_t3_f2, c.term_t3_f3);
          const double T3_ijl  = t3_component(ia, ib, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
          const double T3_i_kl = t3_component(ia, ic, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
          const double T3_j_kl = t3_component(ib, ic, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
          const double M_ijk  = (-T3_ijk  + 4.0*c.p*c.q*dij*c.Jq[ic]) / (8.0*c.p*c.p*c.q);
          const double M_ijl  = (-T3_ijl  + 4.0*c.p*c.q*dij*c.Jq[id]) / (8.0*c.p*c.p*c.q);
          const double N_i_kl = ( T3_i_kl + 4.0*c.p*c.q*dkl*c.Jp[ia]) / (8.0*c.p*c.q*c.q);
          const double N_j_kl = ( T3_j_kl + 4.0*c.p*c.q*dkl*c.Jp[ib]) / (8.0*c.p*c.q*c.q);
          const double T4 = t4_component(ia, ib, ic, id, c.dvec, c.term_t4_f2, c.term_t4_f3, c.term_t4_f4);
          const double M4 = (T4 + 8.0*c.p*c.p*c.q*dkl*c.Kp[ia][ib]
                                + 8.0*c.p*c.q*c.q*dij*c.Kq[ic][id]
                                - 4.0*c.p*c.q*dij*dkl*c.I)
                          / (16.0*c.p*c.p*c.q*c.q);
          const double val = M4
            + d * M_ijk + cc * M_ijl + cc * d * c.Kp[ia][ib]
            + b * N_i_kl + b * d * c.L[ia][ic] + b * cc * c.L[ia][id] + b * cc * d * c.Jp[ia]
            + a * N_j_kl + a * d * c.L[ib][ic] + a * cc * c.L[ib][id] + a * cc * d * c.Jp[ib]
            + a * b * c.Kq[ic][id] + a * b * d * c.Jq[ic] + a * b * cc * c.Jq[id]
            + a * b * cc * d * c.I;
          if (phase == 0) acc0 += val;
          else if (phase == 1) acc1 += val;
          else acc2 += val;
        }
      }
    }
      }
      __syncwarp();
    }
  }

  const int64_t base_out = static_cast<int64_t>(t) * 81;
  eri_out[base_out + lane] = acc0;
  if (lane + 32 < 81) eri_out[base_out + lane + 32] = acc1;
  if (lane + 64 < 81) eri_out[base_out + lane + 64] = acc2;
}

__global__ void KernelERI_ppps_multiblock_component_partial(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    int blocks_per_task,
    double* __restrict__ partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (t >= ntasks || b >= blocks_per_task || lane >= 32) return;

  const int comp = lane;
  const bool active = comp < 27;
  int ia = 0, ib = 0, ic = 0;
  if (active) decode_ppps_comp_cw(comp, ia, ib, ic);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  __shared__ PairTileEntry_cw ab_tile[kPairTileAB_cw];
  __shared__ PairTileEntry_cw cd_tile[kPairTileCD_cw];
  const int nAB_tiles = (nAB + kPairTileAB_cw - 1) / kPairTileAB_cw;
  const int nCD_tiles = (nCD + kPairTileCD_cw - 1) / kPairTileCD_cw;
  const int ntiles = nAB_tiles * nCD_tiles;

  double acc = 0.0;

  for (int tile = b; tile < ntiles; tile += blocks_per_task) {
    const int ab_tile_idx = tile / nCD_tiles;
    const int cd_tile_idx = tile - ab_tile_idx * nCD_tiles;
    const int ab0 = ab_tile_idx * kPairTileAB_cw;
    const int cd0 = cd_tile_idx * kPairTileCD_cw;
    const int ab_count = (nAB - ab0 < kPairTileAB_cw) ? (nAB - ab0) : kPairTileAB_cw;
    const int cd_count = (nCD - cd0 < kPairTileCD_cw) ? (nCD - cd0) : kPairTileCD_cw;

    if (lane < ab_count) {
      const int ki = baseAB + ab0 + lane;
      ab_tile[lane].eta = pair_eta[ki];
      ab_tile[lane].Px = pair_Px[ki];
      ab_tile[lane].Py = pair_Py[ki];
      ab_tile[lane].Pz = pair_Pz[ki];
      ab_tile[lane].cK = pair_cK[ki];
    }
    if (lane < cd_count) {
      const int kj = baseCD + cd0 + lane;
      cd_tile[lane].eta = pair_eta[kj];
      cd_tile[lane].Px = pair_Px[kj];
      cd_tile[lane].Py = pair_Py[kj];
      cd_tile[lane].Pz = pair_Pz[kj];
      cd_tile[lane].cK = pair_cK[kj];
    }
    __syncwarp();

    for (int ii = 0; ii < ab_count; ++ii) {
      const PairTileEntry_cw ab = ab_tile[ii];
      for (int jj = 0; jj < cd_count; ++jj) {
        const PairTileEntry_cw cd = cd_tile[jj];
        PPPSCommon_cw c;

        if (lane == 0) {
          c.p = ab.eta;
          c.q = cd.eta;
          const double Px = ab.Px, Py = ab.Py, Pz = ab.Pz;
          const double Qx = cd.Px, Qy = cd.Py, Qz = cd.Pz;
          c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
          c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
          c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
          c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
          const double PQ2 = c.dvec[0] * c.dvec[0] + c.dvec[1] * c.dvec[1] + c.dvec[2] * c.dvec[2];
          const double denom = c.p + c.q;
          const double omega = c.p * c.q / denom;
          const double T = omega * PQ2;
          const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
          const double base = pref * ab.cK * cd.cK;
          double F0, F1, F2, F3, F4;
          boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
          (void)F4;
          c.I = base * F0;
          const double omega_over_p = omega / c.p;
          const double omega_over_q = omega / c.q;
          c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
          c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
          c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
          c.Jq[0] = omega_over_q * base * F1 * c.dvec[0];
          c.Jq[1] = omega_over_q * base * F1 * c.dvec[1];
          c.Jq[2] = omega_over_q * base * F1 * c.dvec[2];
          const double w2 = omega * omega;
          const double w3 = w2 * omega;
          const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
          const double inv4pq = 1.0 / (4.0 * c.p * c.q);
          const double t4 = 4.0 * w2 * F2;
          const double t2 = 2.0 * omega * F1;
#pragma unroll
          for (int a = 0; a < 3; ++a) {
#pragma unroll
            for (int bb = 0; bb < 3; ++bb) {
              const double dij = (a == bb) ? 1.0 : 0.0;
              const double H = base * (t4 * c.dvec[a] * c.dvec[bb] - (a == bb ? t2 : 0.0));
              c.Kp[a][bb] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
              c.L[a][bb] = -H * inv4pq;
            }
          }
          c.term_t3_f2 = 4.0 * w2 * base * F2;
          c.term_t3_f3 = -8.0 * w3 * base * F3;
        }

        c.I = lane0_bcast_cw(c.I);
        c.p = lane0_bcast_cw(c.p);
        c.q = lane0_bcast_cw(c.q);
        lane0_bcast3_cw(c.Jp);
        lane0_bcast3_cw(c.Jq);
        lane0_bcast3_cw(c.PA);
        lane0_bcast3_cw(c.PB);
        lane0_bcast3_cw(c.QC);
        lane0_bcast3_cw(c.dvec);
        lane0_bcast33_cw(c.Kp);
        lane0_bcast33_cw(c.L);
        c.term_t3_f2 = lane0_bcast_cw(c.term_t3_f2);
        c.term_t3_f3 = lane0_bcast_cw(c.term_t3_f3);

        if (active) {
          const double a = c.PA[ia];
          const double bb = c.PB[ib];
          const double cc = c.QC[ic];
          const double dij = (ia == ib) ? 1.0 : 0.0;
          const double T3 = t3_component(ia, ib, ic, c.dvec, c.term_t3_f2, c.term_t3_f3);
          const double M_ijk = (-T3 + 4.0 * c.p * c.q * dij * c.Jq[ic]) / (8.0 * c.p * c.p * c.q);
          acc += M_ijk + cc * c.Kp[ia][ib]
               + bb * c.L[ia][ic] + bb * cc * c.Jp[ia]
               + a * c.L[ib][ic] + a * cc * c.Jp[ib]
               + a * bb * c.Jq[ic] + a * bb * cc * c.I;
        }
      }
    }
    __syncwarp();
  }

  if (active) {
    const int64_t out = (static_cast<int64_t>(t) * blocks_per_task + b) * 27 + comp;
    partial_sums[out] = acc;
  }
  (void)sp_B;
}

__global__ void KernelERI_ppps_multiblock_component_reduce(
    const double* __restrict__ partial_sums,
    int blocks_per_task,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  const int e = static_cast<int>(threadIdx.x);
  if (e >= 27) return;
  double s = 0.0;
  for (int b = 0; b < blocks_per_task; ++b) {
    const int64_t in = (static_cast<int64_t>(t) * blocks_per_task + b) * 27 + e;
    s += partial_sums[in];
  }
  eri_out[static_cast<int64_t>(t) * 27 + e] = s;
}

__global__ void KernelERI_pppp_multiblock_component_partial(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    int blocks_per_task,
    double* __restrict__ partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (t >= ntasks || b >= blocks_per_task || lane >= 32) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const int D = static_cast<int>(sp_B[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];
  const double Dx = shell_cx[D], Dy = shell_cy[D], Dz = shell_cz[D];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  __shared__ PairTileEntry_cw ab_tile[kPairTileAB_cw];
  __shared__ PairTileEntry_cw cd_tile[kPairTileCD_cw];
  const int nAB_tiles = (nAB + kPairTileAB_cw - 1) / kPairTileAB_cw;
  const int nCD_tiles = (nCD + kPairTileCD_cw - 1) / kPairTileCD_cw;
  const int ntiles = nAB_tiles * nCD_tiles;

  double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;

  for (int tile = b; tile < ntiles; tile += blocks_per_task) {
    const int ab_tile_idx = tile / nCD_tiles;
    const int cd_tile_idx = tile - ab_tile_idx * nCD_tiles;
    const int ab0 = ab_tile_idx * kPairTileAB_cw;
    const int cd0 = cd_tile_idx * kPairTileCD_cw;
    const int ab_count = (nAB - ab0 < kPairTileAB_cw) ? (nAB - ab0) : kPairTileAB_cw;
    const int cd_count = (nCD - cd0 < kPairTileCD_cw) ? (nCD - cd0) : kPairTileCD_cw;

    if (lane < ab_count) {
      const int ki = baseAB + ab0 + lane;
      ab_tile[lane].eta = pair_eta[ki];
      ab_tile[lane].Px = pair_Px[ki];
      ab_tile[lane].Py = pair_Py[ki];
      ab_tile[lane].Pz = pair_Pz[ki];
      ab_tile[lane].cK = pair_cK[ki];
    }
    if (lane < cd_count) {
      const int kj = baseCD + cd0 + lane;
      cd_tile[lane].eta = pair_eta[kj];
      cd_tile[lane].Px = pair_Px[kj];
      cd_tile[lane].Py = pair_Py[kj];
      cd_tile[lane].Pz = pair_Pz[kj];
      cd_tile[lane].cK = pair_cK[kj];
    }
    __syncwarp();

    for (int ii = 0; ii < ab_count; ++ii) {
      const PairTileEntry_cw ab = ab_tile[ii];
      for (int jj = 0; jj < cd_count; ++jj) {
        const PairTileEntry_cw cd = cd_tile[jj];
        PPPPCommon_cw c;

        if (lane == 0) {
          c.p = ab.eta;
          c.q = cd.eta;
          const double Px = ab.Px, Py = ab.Py, Pz = ab.Pz;
          const double Qx = cd.Px, Qy = cd.Py, Qz = cd.Pz;
          c.PA[0] = Px - Ax; c.PA[1] = Py - Ay; c.PA[2] = Pz - Az;
          c.PB[0] = Px - Bx; c.PB[1] = Py - By; c.PB[2] = Pz - Bz;
          c.QC[0] = Qx - Cx; c.QC[1] = Qy - Cy; c.QC[2] = Qz - Cz;
          c.QD[0] = Qx - Dx; c.QD[1] = Qy - Dy; c.QD[2] = Qz - Dz;
          c.dvec[0] = Px - Qx; c.dvec[1] = Py - Qy; c.dvec[2] = Pz - Qz;
          const double PQ2 = c.dvec[0] * c.dvec[0] + c.dvec[1] * c.dvec[1] + c.dvec[2] * c.dvec[2];
          const double denom = c.p + c.q;
          const double omega = c.p * c.q / denom;
          const double T = omega * PQ2;
          const double pref = kTwoPiToFiveHalves / (c.p * c.q * ::sqrt(denom));
          const double base = pref * ab.cK * cd.cK;
          double F0, F1, F2, F3, F4;
          boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
          c.I = base * F0;
          const double omega_over_p = omega / c.p;
          const double omega_over_q = omega / c.q;
          c.Jp[0] = -omega_over_p * base * F1 * c.dvec[0];
          c.Jp[1] = -omega_over_p * base * F1 * c.dvec[1];
          c.Jp[2] = -omega_over_p * base * F1 * c.dvec[2];
          c.Jq[0] = omega_over_q * base * F1 * c.dvec[0];
          c.Jq[1] = omega_over_q * base * F1 * c.dvec[1];
          c.Jq[2] = omega_over_q * base * F1 * c.dvec[2];
          const double w2 = omega * omega;
          const double w3 = w2 * omega;
          const double w4 = w2 * w2;
          const double inv4p2 = 1.0 / (4.0 * c.p * c.p);
          const double inv4q2 = 1.0 / (4.0 * c.q * c.q);
          const double inv4pq = 1.0 / (4.0 * c.p * c.q);
          const double t4 = 4.0 * w2 * F2;
          const double t2 = 2.0 * omega * F1;
#pragma unroll
          for (int a = 0; a < 3; ++a) {
#pragma unroll
            for (int bb = 0; bb < 3; ++bb) {
              const double dij = (a == bb) ? 1.0 : 0.0;
              const double H = base * (t4 * c.dvec[a] * c.dvec[bb] - (a == bb ? t2 : 0.0));
              c.Kp[a][bb] = (H + 2.0 * c.p * c.I * dij) * inv4p2;
              c.Kq[a][bb] = (H + 2.0 * c.q * c.I * dij) * inv4q2;
              c.L[a][bb] = -H * inv4pq;
            }
          }
          c.term_t3_f2 = 4.0 * w2 * base * F2;
          c.term_t3_f3 = -8.0 * w3 * base * F3;
          c.term_t4_f2 = 4.0 * w2 * base * F2;
          c.term_t4_f3 = -8.0 * w3 * base * F3;
          c.term_t4_f4 = 16.0 * w4 * base * F4;
        }

        c.I = lane0_bcast_cw(c.I);
        c.p = lane0_bcast_cw(c.p);
        c.q = lane0_bcast_cw(c.q);
        lane0_bcast3_cw(c.Jp);
        lane0_bcast3_cw(c.Jq);
        lane0_bcast3_cw(c.PA);
        lane0_bcast3_cw(c.PB);
        lane0_bcast3_cw(c.QC);
        lane0_bcast3_cw(c.QD);
        lane0_bcast3_cw(c.dvec);
        lane0_bcast33_cw(c.Kp);
        lane0_bcast33_cw(c.Kq);
        lane0_bcast33_cw(c.L);
        c.term_t3_f2 = lane0_bcast_cw(c.term_t3_f2);
        c.term_t3_f3 = lane0_bcast_cw(c.term_t3_f3);
        c.term_t4_f2 = lane0_bcast_cw(c.term_t4_f2);
        c.term_t4_f3 = lane0_bcast_cw(c.term_t4_f3);
        c.term_t4_f4 = lane0_bcast_cw(c.term_t4_f4);

#pragma unroll
        for (int phase = 0; phase < 3; ++phase) {
          const int comp = phase * 32 + lane;
          if (comp < 81) {
            int ia, ib, ic, id;
            decode_pppp_comp_cw(comp, ia, ib, ic, id);
            const double a = c.PA[ia];
            const double bb = c.PB[ib];
            const double cc = c.QC[ic];
            const double d = c.QD[id];
            const double dij = (ia == ib) ? 1.0 : 0.0;
            const double dkl = (ic == id) ? 1.0 : 0.0;
            const double T3_ijk = t3_component(ia, ib, ic, c.dvec, c.term_t3_f2, c.term_t3_f3);
            const double T3_ijl = t3_component(ia, ib, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
            const double T3_i_kl = t3_component(ia, ic, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
            const double T3_j_kl = t3_component(ib, ic, id, c.dvec, c.term_t3_f2, c.term_t3_f3);
            const double M_ijk = (-T3_ijk + 4.0 * c.p * c.q * dij * c.Jq[ic]) / (8.0 * c.p * c.p * c.q);
            const double M_ijl = (-T3_ijl + 4.0 * c.p * c.q * dij * c.Jq[id]) / (8.0 * c.p * c.p * c.q);
            const double N_i_kl = (T3_i_kl + 4.0 * c.p * c.q * dkl * c.Jp[ia]) / (8.0 * c.p * c.q * c.q);
            const double N_j_kl = (T3_j_kl + 4.0 * c.p * c.q * dkl * c.Jp[ib]) / (8.0 * c.p * c.q * c.q);
            const double T4 = t4_component(ia, ib, ic, id, c.dvec, c.term_t4_f2, c.term_t4_f3, c.term_t4_f4);
            const double M4 = (T4 + 8.0 * c.p * c.p * c.q * dkl * c.Kp[ia][ib]
                                  + 8.0 * c.p * c.q * c.q * dij * c.Kq[ic][id]
                                  - 4.0 * c.p * c.q * dij * dkl * c.I)
                            / (16.0 * c.p * c.p * c.q * c.q);
            const double val = M4
              + d * M_ijk + cc * M_ijl + cc * d * c.Kp[ia][ib]
              + bb * N_i_kl + bb * d * c.L[ia][ic] + bb * cc * c.L[ia][id] + bb * cc * d * c.Jp[ia]
              + a * N_j_kl + a * d * c.L[ib][ic] + a * cc * c.L[ib][id] + a * cc * d * c.Jp[ib]
              + a * bb * c.Kq[ic][id] + a * bb * d * c.Jq[ic] + a * bb * cc * c.Jq[id]
              + a * bb * cc * d * c.I;
            if (phase == 0) acc0 += val;
            else if (phase == 1) acc1 += val;
            else acc2 += val;
          }
        }
      }
    }
    __syncwarp();
  }

  const int64_t out = (static_cast<int64_t>(t) * blocks_per_task + b) * 81;
  partial_sums[out + lane] = acc0;
  if (lane + 32 < 81) partial_sums[out + lane + 32] = acc1;
  if (lane + 64 < 81) partial_sums[out + lane + 64] = acc2;
  (void)sp_B;
}

__global__ void KernelERI_pppp_multiblock_component_reduce(
    const double* __restrict__ partial_sums,
    int blocks_per_task,
    double* __restrict__ eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  for (int e = static_cast<int>(threadIdx.x); e < 81; e += static_cast<int>(blockDim.x)) {
    double s = 0.0;
    for (int b = 0; b < blocks_per_task; ++b) {
      const int64_t in = (static_cast<int64_t>(t) * blocks_per_task + b) * 81 + e;
      s += partial_sums[in];
    }
    eri_out[static_cast<int64_t>(t) * 81 + e] = s;
  }
}

inline int sanitize_component_warp_threads(int threads) {
  int t = threads;
  if (t <= 0) t = 32;
  if (t > 256) t = 256;
  t = (t / 32) * 32;
  return t < 32 ? 32 : t;
}

inline int sanitize_halfwarp_launch_threads(int threads) {
  int t = threads;
  if (t <= 0) t = 128;
  if (t > 256) t = 256;
  t = (t / 32) * 32;
  return t < 32 ? 32 : t;
}

inline int sanitize_subwarp8_launch_threads(int threads) {
  int t = threads;
  if (t <= 0) t = 128;
  if (t > 256) t = 256;
  t = (t / 32) * 32;
  return t < 32 ? 32 : t;
}

inline cudaError_t launch_ppps_packed_exact_shape(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* out_task_idx,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    int npair_ab_expected,
    int npair_cd_expected,
    cudaStream_t stream,
    int threads) {
#define LAUNCH_PPPS_PACKED_EXACT_2PHASE_CASE(NAB, NCD)                                    \
  do {                                                                                     \
    const int threads_eff = sanitize_halfwarp_launch_threads(threads);                     \
    const int warps_per_block = threads_eff >> 5;                                          \
    const int tasks_per_block = warps_per_block * 2;                                       \
    const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;                   \
    switch (warps_per_block) {                                                             \
      case 1:                                                                              \
        KernelERI_ppps_packed_exact_2phase<1, NAB, NCD><<<blocks, 32, 0, stream>>>(       \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 2:                                                                              \
        KernelERI_ppps_packed_exact_2phase<2, NAB, NCD><<<blocks, 64, 0, stream>>>(       \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 3:                                                                              \
        KernelERI_ppps_packed_exact_2phase<3, NAB, NCD><<<blocks, 96, 0, stream>>>(       \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 4:                                                                              \
        KernelERI_ppps_packed_exact_2phase<4, NAB, NCD><<<blocks, 128, 0, stream>>>(      \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 5:                                                                              \
        KernelERI_ppps_packed_exact_2phase<5, NAB, NCD><<<blocks, 160, 0, stream>>>(      \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 6:                                                                              \
        KernelERI_ppps_packed_exact_2phase<6, NAB, NCD><<<blocks, 192, 0, stream>>>(      \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 7:                                                                              \
        KernelERI_ppps_packed_exact_2phase<7, NAB, NCD><<<blocks, 224, 0, stream>>>(      \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      case 8:                                                                              \
        KernelERI_ppps_packed_exact_2phase<8, NAB, NCD><<<blocks, 256, 0, stream>>>(      \
            task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
            shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,   \
            eri_out);                                                                      \
        break;                                                                             \
      default:                                                                             \
        return cudaErrorInvalidConfiguration;                                               \
    }                                                                                      \
    return cudaGetLastError();                                                             \
  } while (0)

#define LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(NAB, NCD)                                     \
  do {                                                                                      \
    int threads_eff = 32;                                                                   \
    const int blocks = (ntasks + threads_eff - 1) / threads_eff;                            \
    KernelERI_ppps_packed_exact_scalar<NAB, NCD><<<blocks, threads_eff, 0, stream>>>(      \
        task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,  \
        shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,        \
        eri_out);                                                                           \
    return cudaGetLastError();                                                              \
  } while (0)

#define LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(NAB, NCD)                                  \
  do {                                                                                      \
    int threads_eff = 32;                                                                   \
    const int blocks = (ntasks + threads_eff - 1) / threads_eff;                            \
    KernelERI_ppps_packed_exact_cdpreload<NAB, NCD><<<blocks, threads_eff, 0, stream>>>(   \
        task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,  \
        shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,        \
        eri_out);                                                                           \
    return cudaGetLastError();                                                              \
  } while (0)

#define LAUNCH_PPPS_PACKED_EXACT_ABPRELOAD_CASE(NAB, NCD)                                  \
  do {                                                                                      \
    int threads_eff = 32;                                                                   \
    const int blocks = (ntasks + threads_eff - 1) / threads_eff;                            \
    KernelERI_ppps_packed_exact_abpreload<NAB, NCD><<<blocks, threads_eff, 0, stream>>>(   \
        task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,  \
        shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,        \
        eri_out);                                                                           \
    return cudaGetLastError();                                                              \
  } while (0)

  if (npair_ab_expected == 1 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(1, 1);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(1, 3);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(1, 6);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PACKED_EXACT_ABPRELOAD_CASE(3, 1);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(3, 3);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(3, 6);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(6, 1);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(6, 3);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(6, 6);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(6, 9);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(9, 3);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE(9, 1);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(9, 6);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE(9, 9);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PACKED_EXACT_ABPRELOAD_CASE(3, 9);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PACKED_EXACT_ABPRELOAD_CASE(1, 9);
  }
  #undef LAUNCH_PPPS_PACKED_EXACT_ABPRELOAD_CASE
  #undef LAUNCH_PPPS_PACKED_EXACT_CDPRELOAD_CASE
  #undef LAUNCH_PPPS_PACKED_EXACT_SCALAR_CASE
  #undef LAUNCH_PPPS_PACKED_EXACT_2PHASE_CASE
  return cudaErrorInvalidValue;
}

inline cudaError_t launch_ppps_prepacked_exact_shape(
    const double* Ax,
    const double* Ay,
    const double* Az,
    const double* Bx,
    const double* By,
    const double* Bz,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* ab_eta,
    const double* ab_Px,
    const double* ab_Py,
    const double* ab_Pz,
    const double* ab_cK,
    const double* cd_eta,
    const double* cd_Qx,
    const double* cd_Qy,
    const double* cd_Qz,
    const double* cd_cK,
    int ntasks,
    double* eri_out,
    int npair_ab_expected,
    int npair_cd_expected,
    cudaStream_t stream,
    int threads) {
#define LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(NAB, NCD)                                  \
  do {                                                                                      \
    int threads_eff = 32;                                                                   \
    const int blocks = (ntasks + threads_eff - 1) / threads_eff;                            \
    KernelERI_ppps_prepacked_exact_scalar<NAB, NCD><<<blocks, threads_eff, 0, stream>>>(   \
        Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,                                                 \
        ab_eta, ab_Px, ab_Py, ab_Pz, ab_cK,                                                 \
        cd_eta, cd_Qx, cd_Qy, cd_Qz, cd_cK,                                                 \
        ntasks, eri_out);                                                                   \
    return cudaGetLastError();                                                              \
  } while (0)

  if (npair_ab_expected == 1 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(1, 1);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(1, 3);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(1, 6);
  }
  if (npair_ab_expected == 1 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(1, 9);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(3, 1);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(3, 3);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(3, 6);
  }
  if (npair_ab_expected == 3 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(3, 9);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(6, 1);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(6, 3);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(6, 6);
  }
  if (npair_ab_expected == 6 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(6, 9);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 1) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(9, 1);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 3) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(9, 3);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 6) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(9, 6);
  }
  if (npair_ab_expected == 9 && npair_cd_expected == 9) {
    LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE(9, 9);
  }
#undef LAUNCH_PPPS_PREPACKED_EXACT_SCALAR_CASE
  return cudaErrorInvalidValue;
}

inline cudaError_t launch_ppps_component_warp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* out_task_idx,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  const int threads_eff = sanitize_component_warp_threads(threads);
  const int warps_per_block = threads_eff >> 5;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  switch (warps_per_block) {
    case 1:
      KernelERI_ppps_warp_component<1><<<blocks, 32, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 2:
      KernelERI_ppps_warp_component<2><<<blocks, 64, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 3:
      KernelERI_ppps_warp_component<3><<<blocks, 96, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 4:
      KernelERI_ppps_warp_component<4><<<blocks, 128, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 5:
      KernelERI_ppps_warp_component<5><<<blocks, 160, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 6:
      KernelERI_ppps_warp_component<6><<<blocks, 192, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 7:
      KernelERI_ppps_warp_component<7><<<blocks, 224, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 8:
      KernelERI_ppps_warp_component<8><<<blocks, 256, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    default:
      return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

inline cudaError_t launch_pppp_component_warp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  const int threads_eff = sanitize_component_warp_threads(threads);
  const int warps_per_block = threads_eff >> 5;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  switch (warps_per_block) {
    case 1:
      KernelERI_pppp_warp_component_3phase<1><<<blocks, 32, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 2:
      KernelERI_pppp_warp_component_3phase<2><<<blocks, 64, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 3:
      KernelERI_pppp_warp_component_3phase<3><<<blocks, 96, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 4:
      KernelERI_pppp_warp_component_3phase<4><<<blocks, 128, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 5:
      KernelERI_pppp_warp_component_3phase<5><<<blocks, 160, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 6:
      KernelERI_pppp_warp_component_3phase<6><<<blocks, 192, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 7:
      KernelERI_pppp_warp_component_3phase<7><<<blocks, 224, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 8:
      KernelERI_pppp_warp_component_3phase<8><<<blocks, 256, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    default:
      return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t cueri_eri_ppps_warp_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  return launch_ppps_component_warp(
      task_spAB, task_spCD, nullptr, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_exact_shape_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    int npair_ab_expected,
    int npair_cd_expected,
    cudaStream_t stream,
    int threads) {
  return launch_ppps_packed_exact_shape(
      task_spAB, task_spCD, nullptr, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      eri_out, npair_ab_expected, npair_cd_expected, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_warp_indexed_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* out_task_idx,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  return launch_ppps_component_warp(
      task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_exact_shape_indexed_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* out_task_idx,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    int npair_ab_expected,
    int npair_cd_expected,
    cudaStream_t stream,
    int threads) {
  return launch_ppps_packed_exact_shape(
      task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      eri_out, npair_ab_expected, npair_cd_expected, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_prepacked_exact_launch_stream(
    const double* Ax,
    const double* Ay,
    const double* Az,
    const double* Bx,
    const double* By,
    const double* Bz,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* ab_eta,
    const double* ab_Px,
    const double* ab_Py,
    const double* ab_Pz,
    const double* ab_cK,
    const double* cd_eta,
    const double* cd_Qx,
    const double* cd_Qy,
    const double* cd_Qz,
    const double* cd_cK,
    int ntasks,
    double* eri_out,
    int npair_ab_expected,
    int npair_cd_expected,
    cudaStream_t stream,
    int threads) {
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  return launch_ppps_prepacked_exact_shape(
      Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
      ab_eta, ab_Px, ab_Py, ab_Pz, ab_cK,
      cd_eta, cd_Qx, cd_Qy, cd_Qz, cd_cK,
      ntasks, eri_out, npair_ab_expected, npair_cd_expected, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_multiblock_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* partial_sums,
    int blocks_per_task,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  (void)threads;
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  if (blocks_per_task <= 0) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(blocks_per_task), 1u);
  KernelERI_ppps_multiblock_component_partial<<<grid, 32, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, blocks_per_task, partial_sums);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  KernelERI_ppps_multiblock_component_reduce<<<static_cast<unsigned int>(ntasks), 32, 0, stream>>>(
      partial_sums, blocks_per_task, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_pppp_warp_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  return launch_pppp_component_warp(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_pppp_multiblock_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* partial_sums,
    int blocks_per_task,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  (void)threads;
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  if (blocks_per_task <= 0) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(blocks_per_task), 1u);
  KernelERI_pppp_multiblock_component_partial<<<grid, 32, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, blocks_per_task, partial_sums);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  KernelERI_pppp_multiblock_component_reduce<<<static_cast<unsigned int>(ntasks), 32, 0, stream>>>(
      partial_sums, blocks_per_task, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_fock_ppps_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* F_mat,
    cudaStream_t stream,
    int threads,
    int n_bufs,
    bool mixed_prec) {
  (void)mixed_prec;  // ppps uses warp_component kernels, mixed_prec not yet implemented
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int launch_threads = sanitize_component_warp_threads(threads);
  const int warps_per_block = launch_threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  switch (warps_per_block) {
    case 1:
      KernelFused_ppps_warp_component<1, true><<<static_cast<unsigned int>(blocks), 32, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 2:
      KernelFused_ppps_warp_component<2, true><<<static_cast<unsigned int>(blocks), 64, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 3:
      KernelFused_ppps_warp_component<3, true><<<static_cast<unsigned int>(blocks), 96, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 4:
      KernelFused_ppps_warp_component<4, true><<<static_cast<unsigned int>(blocks), 128, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 5:
      KernelFused_ppps_warp_component<5, true><<<static_cast<unsigned int>(blocks), 160, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 6:
      KernelFused_ppps_warp_component<6, true><<<static_cast<unsigned int>(blocks), 192, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    case 7:
      KernelFused_ppps_warp_component<7, true><<<static_cast<unsigned int>(blocks), 224, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
    default:
      KernelFused_ppps_warp_component<8, true><<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
      break;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_jk_ppps_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    cudaStream_t stream,
    int threads,
    int n_bufs,
    bool mixed_prec) {
  (void)mixed_prec;  // ppps uses warp_component kernels, mixed_prec not yet implemented
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int launch_threads = sanitize_component_warp_threads(threads);
  const int warps_per_block = launch_threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  switch (warps_per_block) {
    case 1:
      KernelFused_ppps_warp_component<1, false><<<static_cast<unsigned int>(blocks), 32, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 2:
      KernelFused_ppps_warp_component<2, false><<<static_cast<unsigned int>(blocks), 64, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 3:
      KernelFused_ppps_warp_component<3, false><<<static_cast<unsigned int>(blocks), 96, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 4:
      KernelFused_ppps_warp_component<4, false><<<static_cast<unsigned int>(blocks), 128, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 5:
      KernelFused_ppps_warp_component<5, false><<<static_cast<unsigned int>(blocks), 160, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 6:
      KernelFused_ppps_warp_component<6, false><<<static_cast<unsigned int>(blocks), 192, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    case 7:
      KernelFused_ppps_warp_component<7, false><<<static_cast<unsigned int>(blocks), 224, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
    default:
      KernelFused_ppps_warp_component<8, false><<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
          task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
      break;
  }
  return cudaGetLastError();
}

