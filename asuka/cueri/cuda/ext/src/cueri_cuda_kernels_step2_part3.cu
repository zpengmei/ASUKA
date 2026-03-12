// Auto-split from cueri_cuda_kernels_step2.cu (part 3/4: KernelFusedJK_dsss_warp..KernelERI_ppps_warp_tiny_2phase)
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



__global__ void KernelFusedJK_dsss_warp(
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
    int n_bufs) {
  constexpr int nA = 6, nB = 1, nC = 1, nD = 1;
  constexpr int nAB = nA * nB;
  constexpr int nCD = nC * nD;
  constexpr int kNComp = nAB * nCD;  // 6

  extern __shared__ double sh_tile[];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;
  const int buf_id = static_cast<int>(blockIdx.x) % n_bufs;

  double* tile = sh_tile + static_cast<int64_t>(warp_id) * static_cast<int64_t>(kNComp);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nPairAB) * static_cast<int64_t>(nPairCD);

  // Match KernelERI_dsss_warp exactly: outputs are (xx,xy,xz,yy,yz,zz).
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yy = 0.0;
  double s_yz = 0.0;
  double s_zz = 0.0;
  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nPairCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nPairCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];

    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = p + q;
    const double omega = p * q / denom;
    const double T = omega * PQ2;

    const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
    const double base = pref * pair_cK[ki] * pair_cK[kj];

    double F0, F1, F2;
    boys_f0_f1_f2(T, F0, F1, F2);

    const double I = base * F0;
    const double omega_over_p = omega / p;
    const double Jx = -omega_over_p * base * F1 * dx;
    const double Jy = -omega_over_p * base * F1 * dy;
    const double Jz = -omega_over_p * base * F1 * dz;

    const double inv4p2 = 1.0 / (4.0 * p * p);
    const double w2 = omega * omega;
    const double t4 = 4.0 * w2 * F2;
    const double t2 = 2.0 * omega * F1;
    const double Kxx = (base * (t4 * dx * dx - t2) + 2.0 * p * I) * inv4p2;
    const double Kyy = (base * (t4 * dy * dy - t2) + 2.0 * p * I) * inv4p2;
    const double Kzz = (base * (t4 * dz * dz - t2) + 2.0 * p * I) * inv4p2;
    const double Kxy = (base * (t4 * dx * dy)) * inv4p2;
    const double Kxz = (base * (t4 * dx * dz)) * inv4p2;
    const double Kyz = (base * (t4 * dy * dz)) * inv4p2;

    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;

    s_xx += Kxx + 2.0 * PAx * Jx + (PAx * PAx) * I;
    s_xy += Kxy + PAx * Jy + PAy * Jx + (PAx * PAy) * I;
    s_xz += Kxz + PAx * Jz + PAz * Jx + (PAx * PAz) * I;
    s_yy += Kyy + 2.0 * PAy * Jy + (PAy * PAy) * I;
    s_yz += Kyz + PAy * Jz + PAz * Jy + (PAy * PAz) * I;
    s_zz += Kzz + 2.0 * PAz * Jz + (PAz * PAz) * I;
  }

  s_xx = warp_reduce_sum(s_xx);
  s_xy = warp_reduce_sum(s_xy);
  s_xz = warp_reduce_sum(s_xz);
  s_yy = warp_reduce_sum(s_yy);
  s_yz = warp_reduce_sum(s_yz);
  s_zz = warp_reduce_sum(s_zz);

  if (lane == 0) {
    tile[0] = s_xx;
    tile[1] = s_xy;
    tile[2] = s_xz;
    tile[3] = s_yy;
    tile[4] = s_yz;
    tile[5] = s_zz;
  }
  __syncwarp();

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

  cueri_contract_jk_warp_single(
      tile, D_mat, J_mat, K_mat, lane,
      nAB, nCD, nA, nB, nC, nD,
      a0, b0, c0, d0,
      ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);
  (void)sp_B;
}

template <bool kToFock>
__global__ void KernelFused_ppss_warp(
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
    double* out0_mat,
    double* out1_mat,
    int n_bufs) {
  constexpr int nA = 3, nB = 3, nC = 1, nD = 1;
  constexpr int nAB = nA * nB;
  constexpr int nCD = nC * nD;
  constexpr int kNComp = nAB * nCD;  // 9

  extern __shared__ double sh_tile[];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;
  const int buf_id = static_cast<int>(blockIdx.x) % n_bufs;

  double* tile = sh_tile + static_cast<int64_t>(warp_id) * static_cast<int64_t>(kNComp);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];
  const double Bx = shell_cx[B];
  const double By = shell_cy[B];
  const double Bz = shell_cz[B];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nPairAB) * static_cast<int64_t>(nPairCD);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nPairCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nPairCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];

    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = p + q;
    const double omega = p * q / denom;
    const double T = omega * PQ2;

    const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
    const double base = pref * pair_cK[ki] * pair_cK[kj];

    double F0, F1, F2;
    boys_f0_f1_f2(T, F0, F1, F2);

    const double I = base * F0;

    const double omega_over_p = omega / p;
    const double omega_over_q = omega / q;
    const double Jpx = -omega_over_p * base * F1 * dx;
    const double Jpy = -omega_over_p * base * F1 * dy;
    const double Jpz = -omega_over_p * base * F1 * dz;
    const double Jqx = omega_over_q * base * F1 * dx;
    const double Jqy = omega_over_q * base * F1 * dy;
    const double Jqz = omega_over_q * base * F1 * dz;

    const double w2 = omega * omega;
    const double t4 = 4.0 * w2 * F2;
    const double t2 = 2.0 * omega * F1;
    const double inv4pq = 1.0 / (4.0 * p * q);

    const double Hxx = base * (t4 * dx * dx - t2);
    const double Hyy = base * (t4 * dy * dy - t2);
    const double Hzz = base * (t4 * dz * dz - t2);
    const double Hxy = base * (t4 * dx * dy);
    const double Hxz = base * (t4 * dx * dz);
    const double Hyz = base * (t4 * dy * dz);

    const double Lxx = -(Hxx) * inv4pq;
    const double Lyy = -(Hyy) * inv4pq;
    const double Lzz = -(Hzz) * inv4pq;
    const double Lxy = -(Hxy) * inv4pq;
    const double Lxz = -(Hxz) * inv4pq;
    const double Lyz = -(Hyz) * inv4pq;

    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;
    const double QBx = Qx - Bx;
    const double QBy = Qy - By;
    const double QBz = Qz - Bz;

    s00 += Lxx + QBx * Jpx + PAx * Jqx + PAx * QBx * I;
    s01 += Lxy + QBy * Jpx + PAx * Jqy + PAx * QBy * I;
    s02 += Lxz + QBz * Jpx + PAx * Jqz + PAx * QBz * I;

    s10 += Lxy + QBx * Jpy + PAy * Jqx + PAy * QBx * I;
    s11 += Lyy + QBy * Jpy + PAy * Jqy + PAy * QBy * I;
    s12 += Lyz + QBz * Jpy + PAy * Jqz + PAy * QBz * I;

    s20 += Lxz + QBx * Jpz + PAz * Jqx + PAz * QBx * I;
    s21 += Lyz + QBy * Jpz + PAz * Jqy + PAz * QBy * I;
    s22 += Lzz + QBz * Jpz + PAz * Jqz + PAz * QBz * I;
  }

  s00 = warp_reduce_sum(s00);
  s01 = warp_reduce_sum(s01);
  s02 = warp_reduce_sum(s02);
  s10 = warp_reduce_sum(s10);
  s11 = warp_reduce_sum(s11);
  s12 = warp_reduce_sum(s12);
  s20 = warp_reduce_sum(s20);
  s21 = warp_reduce_sum(s21);
  s22 = warp_reduce_sum(s22);

  if (lane == 0) {
    tile[0] = s00;
    tile[1] = s01;
    tile[2] = s02;
    tile[3] = s10;
    tile[4] = s11;
    tile[5] = s12;
    tile[6] = s20;
    tile[7] = s21;
    tile[8] = s22;
  }
  __syncwarp();

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

  if constexpr (kToFock) {
    cueri_contract_fock_warp_single(
        tile, D_mat, out0_mat, lane,
        nAB, nCD, nA, nB, nC, nD,
        a0, b0, c0, d0,
        ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);
  } else {
    cueri_contract_jk_warp_single(
        tile, D_mat, out0_mat, out1_mat, lane,
        nAB, nCD, nA, nB, nC, nD,
        a0, b0, c0, d0,
        ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);
  }
}

__global__ void KernelFusedJK_ppss_warp(
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
    int n_bufs) {
  constexpr int nA = 3, nB = 3, nC = 1, nD = 1;
  constexpr int nAB = nA * nB;
  constexpr int nCD = nC * nD;
  constexpr int kNComp = nAB * nCD;  // 9

  extern __shared__ double sh_tile[];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;
  const int buf_id = static_cast<int>(blockIdx.x) % n_bufs;

  double* tile = sh_tile + static_cast<int64_t>(warp_id) * static_cast<int64_t>(kNComp);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];
  const double Bx = shell_cx[B];
  const double By = shell_cy[B];
  const double Bz = shell_cz[B];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nPairAB) * static_cast<int64_t>(nPairCD);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nPairCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nPairCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];

    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = p + q;
    const double omega = p * q / denom;
    const double T = omega * PQ2;

    const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
    const double base = pref * pair_cK[ki] * pair_cK[kj];

    double F0, F1, F2;
    boys_f0_f1_f2(T, F0, F1, F2);

    const double I = base * F0;

    const double omega_over_p = omega / p;
    const double omega_over_q = omega / q;
    const double Jpx = -omega_over_p * base * F1 * dx;
    const double Jpy = -omega_over_p * base * F1 * dy;
    const double Jpz = -omega_over_p * base * F1 * dz;
    const double Jqx = omega_over_q * base * F1 * dx;
    const double Jqy = omega_over_q * base * F1 * dy;
    const double Jqz = omega_over_q * base * F1 * dz;

    const double w2 = omega * omega;
    const double t4 = 4.0 * w2 * F2;
    const double t2 = 2.0 * omega * F1;
    const double inv4pq = 1.0 / (4.0 * p * q);

    const double Hxx = base * (t4 * dx * dx - t2);
    const double Hyy = base * (t4 * dy * dy - t2);
    const double Hzz = base * (t4 * dz * dz - t2);
    const double Hxy = base * (t4 * dx * dy);
    const double Hxz = base * (t4 * dx * dz);
    const double Hyz = base * (t4 * dy * dz);

    const double Lxx = -(Hxx) * inv4pq;
    const double Lyy = -(Hyy) * inv4pq;
    const double Lzz = -(Hzz) * inv4pq;
    const double Lxy = -(Hxy) * inv4pq;
    const double Lxz = -(Hxz) * inv4pq;
    const double Lyz = -(Hyz) * inv4pq;

    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;
    const double QBx = Qx - Bx;
    const double QBy = Qy - By;
    const double QBz = Qz - Bz;

    s00 += Lxx + QBx * Jpx + PAx * Jqx + PAx * QBx * I;
    s01 += Lxy + QBy * Jpx + PAx * Jqy + PAx * QBy * I;
    s02 += Lxz + QBz * Jpx + PAx * Jqz + PAx * QBz * I;

    s10 += Lxy + QBx * Jpy + PAy * Jqx + PAy * QBx * I;
    s11 += Lyy + QBy * Jpy + PAy * Jqy + PAy * QBy * I;
    s12 += Lyz + QBz * Jpy + PAy * Jqz + PAy * QBz * I;

    s20 += Lxz + QBx * Jpz + PAz * Jqx + PAz * QBx * I;
    s21 += Lyz + QBy * Jpz + PAz * Jqy + PAz * QBy * I;
    s22 += Lzz + QBz * Jpz + PAz * Jqz + PAz * QBz * I;
  }

  s00 = warp_reduce_sum(s00);
  s01 = warp_reduce_sum(s01);
  s02 = warp_reduce_sum(s02);
  s10 = warp_reduce_sum(s10);
  s11 = warp_reduce_sum(s11);
  s12 = warp_reduce_sum(s12);
  s20 = warp_reduce_sum(s20);
  s21 = warp_reduce_sum(s21);
  s22 = warp_reduce_sum(s22);

  if (lane == 0) {
    tile[0] = s00;
    tile[1] = s01;
    tile[2] = s02;
    tile[3] = s10;
    tile[4] = s11;
    tile[5] = s12;
    tile[6] = s20;
    tile[7] = s21;
    tile[8] = s22;
  }
  __syncwarp();

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

  cueri_contract_jk_warp_single(
      tile, D_mat, J_mat, K_mat, lane,
      nAB, nCD, nA, nB, nC, nD,
      a0, b0, c0, d0,
      ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);
}

template <bool kToFock>
__global__ void KernelFused_psps_subwarp8(
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
    double* out0_mat,
    double* out1_mat,
    int n_bufs) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  const int subwarp = lane >> 3;
  const int lane8 = lane & 7;
  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];
  const double Cx = shell_cx[C];
  const double Cy = shell_cy[C];
  const double Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nPairAB) * static_cast<int64_t>(nPairCD);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int64_t u = static_cast<int64_t>(lane8); u < nTot; u += 8) {
    const int i = static_cast<int>(u / nPairCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nPairCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];

    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = p + q;
    const double omega = p * q / denom;
    const double T = omega * PQ2;

    const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
    const double base = pref * pair_cK[ki] * pair_cK[kj];

    double F0, F1, F2;
    boys_f0_f1_f2(T, F0, F1, F2);

    const double I = base * F0;
    const double omega_over_p = omega / p;
    const double omega_over_q = omega / q;
    const double Jpx = -omega_over_p * base * F1 * dx;
    const double Jpy = -omega_over_p * base * F1 * dy;
    const double Jpz = -omega_over_p * base * F1 * dz;
    const double Jqx = omega_over_q * base * F1 * dx;
    const double Jqy = omega_over_q * base * F1 * dy;
    const double Jqz = omega_over_q * base * F1 * dz;

    const double w2 = omega * omega;
    const double t4 = 4.0 * w2 * F2;
    const double t2 = 2.0 * omega * F1;
    const double inv4pq = 1.0 / (4.0 * p * q);

    const double Hxx = base * (t4 * dx * dx - t2);
    const double Hyy = base * (t4 * dy * dy - t2);
    const double Hzz = base * (t4 * dz * dz - t2);
    const double Hxy = base * (t4 * dx * dy);
    const double Hxz = base * (t4 * dx * dz);
    const double Hyz = base * (t4 * dy * dz);

    const double Lxx = -(Hxx) * inv4pq;
    const double Lyy = -(Hyy) * inv4pq;
    const double Lzz = -(Hzz) * inv4pq;
    const double Lxy = -(Hxy) * inv4pq;
    const double Lxz = -(Hxz) * inv4pq;
    const double Lyz = -(Hyz) * inv4pq;

    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;
    const double QCx = Qx - Cx;
    const double QCy = Qy - Cy;
    const double QCz = Qz - Cz;

    s00 += Lxx + QCx * Jpx + PAx * Jqx + PAx * QCx * I;
    s01 += Lxy + QCy * Jpx + PAx * Jqy + PAx * QCy * I;
    s02 += Lxz + QCz * Jpx + PAx * Jqz + PAx * QCz * I;

    s10 += Lxy + QCx * Jpy + PAy * Jqx + PAy * QCx * I;
    s11 += Lyy + QCy * Jpy + PAy * Jqy + PAy * QCy * I;
    s12 += Lyz + QCz * Jpy + PAy * Jqz + PAy * QCz * I;

    s20 += Lxz + QCx * Jpz + PAz * Jqx + PAz * QCx * I;
    s21 += Lyz + QCy * Jpz + PAz * Jqy + PAz * QCy * I;
    s22 += Lzz + QCz * Jpz + PAz * Jqz + PAz * QCz * I;
  }

  s00 = subwarp8_reduce_sum(s00);
  s01 = subwarp8_reduce_sum(s01);
  s02 = subwarp8_reduce_sum(s02);
  s10 = subwarp8_reduce_sum(s10);
  s11 = subwarp8_reduce_sum(s11);
  s12 = subwarp8_reduce_sum(s12);
  s20 = subwarp8_reduce_sum(s20);
  s21 = subwarp8_reduce_sum(s21);
  s22 = subwarp8_reduce_sum(s22);
  if (lane8 != 0) return;

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

  if constexpr (kToFock) {
    double* F_mat = out0_mat + buf_off;
    accumulate_fock_single_value(s00, D_mat, F_mat, a0 + 0, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s01, D_mat, F_mat, a0 + 0, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s02, D_mat, F_mat, a0 + 0, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s10, D_mat, F_mat, a0 + 1, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s11, D_mat, F_mat, a0 + 1, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s12, D_mat, F_mat, a0 + 1, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s20, D_mat, F_mat, a0 + 2, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s21, D_mat, F_mat, a0 + 2, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_fock_single_value(s22, D_mat, F_mat, a0 + 2, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
  } else {
    double* J_mat = out0_mat + buf_off;
    double* K_mat = out1_mat + buf_off;
    accumulate_jk_single_value(s00, D_mat, J_mat, K_mat, a0 + 0, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s01, D_mat, J_mat, K_mat, a0 + 0, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s02, D_mat, J_mat, K_mat, a0 + 0, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s10, D_mat, J_mat, K_mat, a0 + 1, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s11, D_mat, J_mat, K_mat, a0 + 1, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s12, D_mat, J_mat, K_mat, a0 + 1, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s20, D_mat, J_mat, K_mat, a0 + 2, b0, c0 + 0, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s21, D_mat, J_mat, K_mat, a0 + 2, b0, c0 + 1, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
    accumulate_jk_single_value(s22, D_mat, J_mat, K_mat, a0 + 2, b0, c0 + 2, d0, ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
  }
}


// ---------------------------------------------------------------------------
// Flat kernels: 1 thread per task, no shared memory / block reduction.
// For low-ncomp classes (ncomp <= 9), this is much faster than block kernels
// because it avoids wasting 112/128 threads per task.
// ---------------------------------------------------------------------------

__global__ void KernelERI_psss_flat(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
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

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    for (int j = 0; j < nCD; ++j) {
      const int kj = baseCD + j;
      const double q = pair_eta[kj];
      const double dx = Pxi - pair_Px[kj];
      const double dy = Pyi - pair_Py[kj];
      const double dz = Pzi - pair_Pz[kj];
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1;
      boys_f0_f1(T, F0, F1);
      const double q_over = q / denom;
      sx += base * (-(Ax - Pxi) * F0 - q_over * dx * F1);
      sy += base * (-(Ay - Pyi) * F0 - q_over * dy * F1);
      sz += base * (-(Az - Pzi) * F0 - q_over * dz * F1);
    }
  }
  const int out = t * 3;
  eri_out[out + 0] = sx;
  eri_out[out + 1] = sy;
  eri_out[out + 2] = sz;
}

__global__ void KernelERI_ppss_flat(
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
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double s00=0, s01=0, s02=0, s10=0, s11=0, s12=0, s20=0, s21=0, s22=0;
  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double PAx = Pxi - Ax, PAy = Pyi - Ay, PAz = Pzi - Az;
    const double PBx = Pxi - Bx, PBy = Pyi - By, PBz = Pzi - Bz;
    for (int j = 0; j < nCD; ++j) {
      const int kj = baseCD + j;
      const double q = pair_eta[kj];
      const double dx = Pxi - pair_Px[kj];
      const double dy = Pyi - pair_Py[kj];
      const double dz = Pzi - pair_Pz[kj];
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1, F2;
      boys_f0_f1_f2(T, F0, F1, F2);
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double Jx = -omega_over_p * base * F1 * dx;
      const double Jy = -omega_over_p * base * F1 * dy;
      const double Jz = -omega_over_p * base * F1 * dz;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double w2 = omega * omega;
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      const double Kxx = (base * (t4*dx*dx - t2) + 2.0*p*I) * inv4p2;
      const double Kyy = (base * (t4*dy*dy - t2) + 2.0*p*I) * inv4p2;
      const double Kzz = (base * (t4*dz*dz - t2) + 2.0*p*I) * inv4p2;
      const double Kxy = (base * (t4*dx*dy)) * inv4p2;
      const double Kxz = (base * (t4*dx*dz)) * inv4p2;
      const double Kyz = (base * (t4*dy*dz)) * inv4p2;
      s00 += Kxx + PAx*Jx + PBx*Jx + PAx*PBx*I;
      s01 += Kxy + PAx*Jy + PBy*Jx + PAx*PBy*I;
      s02 += Kxz + PAx*Jz + PBz*Jx + PAx*PBz*I;
      s10 += Kxy + PAy*Jx + PBx*Jy + PAy*PBx*I;
      s11 += Kyy + PAy*Jy + PBy*Jy + PAy*PBy*I;
      s12 += Kyz + PAy*Jz + PBz*Jy + PAy*PBz*I;
      s20 += Kxz + PAz*Jx + PBx*Jz + PAz*PBx*I;
      s21 += Kyz + PAz*Jy + PBy*Jz + PAz*PBy*I;
      s22 += Kzz + PAz*Jz + PBz*Jz + PAz*PBz*I;
    }
  }
  const int out = t * 9;
  eri_out[out+0]=s00; eri_out[out+1]=s01; eri_out[out+2]=s02;
  eri_out[out+3]=s10; eri_out[out+4]=s11; eri_out[out+5]=s12;
  eri_out[out+6]=s20; eri_out[out+7]=s21; eri_out[out+8]=s22;
}

__global__ void KernelERI_psps_flat(
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
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
  const double Cx = shell_cx[C], Cy = shell_cy[C], Cz = shell_cz[C];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double s00=0, s01=0, s02=0, s10=0, s11=0, s12=0, s20=0, s21=0, s22=0;
  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double PAx = Pxi - Ax, PAy = Pyi - Ay, PAz = Pzi - Az;
    for (int j = 0; j < nCD; ++j) {
      const int kj = baseCD + j;
      const double q = pair_eta[kj];
      const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
      const double dx = Pxi - Qx, dy = Pyi - Qy, dz = Pzi - Qz;
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1, F2;
      boys_f0_f1_f2(T, F0, F1, F2);
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jpx = -omega_over_p * base * F1 * dx;
      const double Jpy = -omega_over_p * base * F1 * dy;
      const double Jpz = -omega_over_p * base * F1 * dz;
      const double Jqx = omega_over_q * base * F1 * dx;
      const double Jqy = omega_over_q * base * F1 * dy;
      const double Jqz = omega_over_q * base * F1 * dz;
      const double w2 = omega * omega;
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double Lxx = -(base * (t4*dx*dx - t2)) * inv4pq;
      const double Lyy = -(base * (t4*dy*dy - t2)) * inv4pq;
      const double Lzz = -(base * (t4*dz*dz - t2)) * inv4pq;
      const double Lxy = -(base * (t4*dx*dy)) * inv4pq;
      const double Lxz = -(base * (t4*dx*dz)) * inv4pq;
      const double Lyz = -(base * (t4*dy*dz)) * inv4pq;
      const double QCx = Qx - Cx, QCy = Qy - Cy, QCz = Qz - Cz;
      s00 += Lxx + QCx*Jpx + PAx*Jqx + PAx*QCx*I;
      s01 += Lxy + QCy*Jpx + PAx*Jqy + PAx*QCy*I;
      s02 += Lxz + QCz*Jpx + PAx*Jqz + PAx*QCz*I;
      s10 += Lxy + QCx*Jpy + PAy*Jqx + PAy*QCx*I;
      s11 += Lyy + QCy*Jpy + PAy*Jqy + PAy*QCy*I;
      s12 += Lyz + QCz*Jpy + PAy*Jqz + PAy*QCz*I;
      s20 += Lxz + QCx*Jpz + PAz*Jqx + PAz*QCx*I;
      s21 += Lyz + QCy*Jpz + PAz*Jqy + PAz*QCy*I;
      s22 += Lzz + QCz*Jpz + PAz*Jqz + PAz*QCz*I;
    }
  }
  const int out = t * 9;
  eri_out[out+0]=s00; eri_out[out+1]=s01; eri_out[out+2]=s02;
  eri_out[out+3]=s10; eri_out[out+4]=s11; eri_out[out+5]=s12;
  eri_out[out+6]=s20; eri_out[out+7]=s21; eri_out[out+8]=s22;
}

__global__ void KernelERI_dsss_flat(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_A,
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

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double s_xx=0, s_xy=0, s_xz=0, s_yy=0, s_yz=0, s_zz=0;
  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double PAx = Pxi - Ax, PAy = Pyi - Ay, PAz = Pzi - Az;
    for (int j = 0; j < nCD; ++j) {
      const int kj = baseCD + j;
      const double q = pair_eta[kj];
      const double dx = Pxi - pair_Px[kj];
      const double dy = Pyi - pair_Py[kj];
      const double dz = Pzi - pair_Pz[kj];
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1, F2;
      boys_f0_f1_f2(T, F0, F1, F2);
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double Jx = -omega_over_p * base * F1 * dx;
      const double Jy = -omega_over_p * base * F1 * dy;
      const double Jz = -omega_over_p * base * F1 * dz;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double w2 = omega * omega;
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      const double Kxx = (base * (t4*dx*dx - t2) + 2.0*p*I) * inv4p2;
      const double Kyy = (base * (t4*dy*dy - t2) + 2.0*p*I) * inv4p2;
      const double Kzz = (base * (t4*dz*dz - t2) + 2.0*p*I) * inv4p2;
      const double Kxy = (base * (t4*dx*dy)) * inv4p2;
      const double Kxz = (base * (t4*dx*dz)) * inv4p2;
      const double Kyz = (base * (t4*dy*dz)) * inv4p2;
      s_xx += Kxx + 2.0*PAx*Jx + PAx*PAx*I;
      s_xy += Kxy + PAx*Jy + PAy*Jx + PAx*PAy*I;
      s_xz += Kxz + PAx*Jz + PAz*Jx + PAx*PAz*I;
      s_yy += Kyy + 2.0*PAy*Jy + PAy*PAy*I;
      s_yz += Kyz + PAy*Jz + PAz*Jy + PAy*PAz*I;
      s_zz += Kzz + 2.0*PAz*Jz + PAz*PAz*I;
    }
  }
  const int out = t * 6;
  eri_out[out+0]=s_xx; eri_out[out+1]=s_xy; eri_out[out+2]=s_xz;
  eri_out[out+3]=s_yy; eri_out[out+4]=s_yz; eri_out[out+5]=s_zz;
}

__global__ void KernelERI_ppps_flat(
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
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;

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

  double s[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) s[i] = 0.0;

  for (int ii = 0; ii < nAB; ++ii) {
    const int ki = baseAB + ii;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double PA[3] = {Pxi - Ax, Pyi - Ay, Pzi - Az};
    const double PB[3] = {Pxi - Bx, Pyi - By, Pzi - Bz};
    for (int jj = 0; jj < nCD; ++jj) {
      const int kj = baseCD + jj;
      const double q = pair_eta[kj];
      const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
      const double dx = Pxi - Qx, dy = Pyi - Qy, dz = Pzi - Qz;
      const double dvec[3] = {dx, dy, dz};
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {-omega_over_p*base*F1*dx, -omega_over_p*base*F1*dy, -omega_over_p*base*F1*dz};
      const double Jq[3] = {omega_over_q*base*F1*dx, omega_over_q*base*F1*dy, omega_over_q*base*F1*dz};
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double H[3][3], Kp[3][3], L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          H[a][b] = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H[a][b] + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -(H[a][b]) * inv4pq;
        }
      }
      const double QC[3] = {Qx - Cx, Qy - Cy, Qz - Cz};
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
#pragma unroll
      for (int ia = 0; ia < 3; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 3; ++ib) {
          const double a_ = PA[ia], b_ = PB[ib];
          const double dij = (ia == ib) ? 1.0 : 0.0;
          const double Kp_ij = Kp[ia][ib];
#pragma unroll
          for (int ic = 0; ic < 3; ++ic) {
            const double c_ = QC[ic];
            const double T3_ijk = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
            const double M_ijk = (-T3_ijk + 4.0*p*q*dij*Jq[ic]) / (8.0*p*p*q);
            s[ia*9 + ib*3 + ic] += M_ijk + c_*Kp_ij + b_*L[ia][ic] + b_*c_*Jp[ia]
                                 + a_*L[ib][ic] + a_*c_*Jp[ib] + a_*b_*Jq[ic] + a_*b_*c_*I;
          }
        }
      }
    }
  }
  const int out = t * 27;
#pragma unroll
  for (int i = 0; i < 27; ++i) eri_out[out + i] = s[i];
}

__global__ void KernelERI_pppp_warp(
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
    double* eri_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double s[81];
#pragma unroll
  for (int i = 0; i < 81; ++i) s[i] = 0.0;

  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int iab = static_cast<int>(u / nCD);
    const int icd = static_cast<int>(u - static_cast<int64_t>(iab) * nCD);
    const int ki = baseAB + iab;
    const int kj = baseCD + icd;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];

    const double dx = Pxi - Qx, dy = Pyi - Qy, dz = Pzi - Qz;
    const double dvec[3] = {dx, dy, dz};
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = p + q;
    const double omega = p * q / denom;
    const double T = omega * PQ2;

    const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
    const double base = pref * pair_cK[ki] * pair_cK[kj];

    double F0, F1, F2, F3, F4;
    boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);

    const double I = base * F0;

    const double omega_over_p = omega / p;
    const double omega_over_q = omega / q;
    const double Jp[3] = {-omega_over_p*base*F1*dx, -omega_over_p*base*F1*dy, -omega_over_p*base*F1*dz};
    const double Jq[3] = {omega_over_q*base*F1*dx, omega_over_q*base*F1*dy, omega_over_q*base*F1*dz};

    const double w2 = omega * omega;
    const double w3 = w2 * omega;
    const double w4 = w2 * w2;
    const double inv4p2 = 1.0 / (4.0 * p * p);
    const double inv4q2 = 1.0 / (4.0 * q * q);
    const double inv4pq = 1.0 / (4.0 * p * q);

    const double t4 = 4.0 * w2 * F2;
    const double t2 = 2.0 * omega * F1;
    double H[3][3], Kp[3][3], Kq[3][3], L[3][3];
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        const double dij = (a == b) ? 1.0 : 0.0;
        H[a][b] = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
        Kp[a][b] = (H[a][b] + 2.0 * p * I * dij) * inv4p2;
        Kq[a][b] = (H[a][b] + 2.0 * q * I * dij) * inv4q2;
        L[a][b] = -(H[a][b]) * inv4pq;
      }
    }

    const double PA[3] = {Pxi - Ax, Pyi - Ay, Pzi - Az};
    const double PB[3] = {Pxi - Bx, Pyi - By, Pzi - Bz};
    const double QC[3] = {Qx - Cx, Qy - Cy, Qz - Cz};
    const double QD[3] = {Qx - Dx, Qy - Dy, Qz - Dz};

    const double term_t3_f2 = 4.0 * w2 * base * F2;
    const double term_t3_f3 = -8.0 * w3 * base * F3;
    const double term_t4_f2 = 4.0 * w2 * base * F2;
    const double term_t4_f3 = -8.0 * w3 * base * F3;
    const double term_t4_f4 = 16.0 * w4 * base * F4;

#pragma unroll
    for (int ia = 0; ia < 3; ++ia) {
#pragma unroll
      for (int ib = 0; ib < 3; ++ib) {
        const double a_ = PA[ia], b_ = PB[ib];
        const double dij = (ia == ib) ? 1.0 : 0.0;
        const double Kp_ij = Kp[ia][ib];
#pragma unroll
        for (int ic = 0; ic < 3; ++ic) {
#pragma unroll
          for (int id = 0; id < 3; ++id) {
            const double c_ = QC[ic], d_ = QD[id];
            const double dkl = (ic == id) ? 1.0 : 0.0;
            const double Kq_kl = Kq[ic][id];
            const double T3_ijk = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
            const double T3_ijl = t3_component(ia, ib, id, dvec, term_t3_f2, term_t3_f3);
            const double T3_i_kl = t3_component(ia, ic, id, dvec, term_t3_f2, term_t3_f3);
            const double T3_j_kl = t3_component(ib, ic, id, dvec, term_t3_f2, term_t3_f3);
            const double M_ijk = (-T3_ijk + 4.0*p*q*dij*Jq[ic]) / (8.0*p*p*q);
            const double M_ijl = (-T3_ijl + 4.0*p*q*dij*Jq[id]) / (8.0*p*p*q);
            const double N_i_kl = (T3_i_kl + 4.0*p*q*dkl*Jp[ia]) / (8.0*p*q*q);
            const double N_j_kl = (T3_j_kl + 4.0*p*q*dkl*Jp[ib]) / (8.0*p*q*q);
            const double T4_ijkl = t4_component(ia, ib, ic, id, dvec, term_t4_f2, term_t4_f3, term_t4_f4);
            const double M4_ij_kl = (T4_ijkl + 8.0*p*p*q*dkl*Kp_ij + 8.0*p*q*q*dij*Kq_kl - 4.0*p*q*dij*dkl*I) / (16.0*p*p*q*q);
            s[ia*27 + ib*9 + ic*3 + id] += M4_ij_kl + d_*M_ijk + c_*M_ijl + c_*d_*Kp_ij
              + b_*N_i_kl + b_*d_*L[ia][ic] + b_*c_*L[ia][id] + b_*c_*d_*Jp[ia]
              + a_*N_j_kl + a_*d_*L[ib][ic] + a_*c_*L[ib][id] + a_*c_*d_*Jp[ib]
              + a_*b_*Kq_kl + a_*b_*d_*Jq[ic] + a_*b_*c_*Jq[id] + a_*b_*c_*d_*I;
          }
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 81; ++i) s[i] = warp_reduce_sum(s[i]);
  if (lane == 0) {
    const int out = t * 81;
#pragma unroll
    for (int i = 0; i < 81; ++i) eri_out[out + i] = s[i];
  }
}

__global__ void KernelERI_pppp_flat(
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
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double s[81];
#pragma unroll 1
  for (int i = 0; i < 81; ++i) s[i] = 0.0;

  for (int ii = 0; ii < nAB; ++ii) {
    const int ki = baseAB + ii;
    const double p = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki], Pyi = pair_Py[ki], Pzi = pair_Pz[ki];
    const double PA[3] = {Pxi - Ax, Pyi - Ay, Pzi - Az};
    const double PB[3] = {Pxi - Bx, Pyi - By, Pzi - Bz};
    for (int jj = 0; jj < nCD; ++jj) {
      const int kj = baseCD + jj;
      const double q = pair_eta[kj];
      const double Qx = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
      const double dx = Pxi - Qx, dy = Pyi - Qy, dz = Pzi - Qz;
      const double dvec[3] = {dx, dy, dz};
      const double PQ2 = dx*dx + dy*dy + dz*dz;
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (p * q * ::sqrt(denom));
      const double base = pref * cKi * pair_cK[kj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {-omega_over_p*base*F1*dx, -omega_over_p*base*F1*dy, -omega_over_p*base*F1*dz};
      const double Jq[3] = {omega_over_q*base*F1*dx, omega_over_q*base*F1*dy, omega_over_q*base*F1*dz};
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double w4 = w2 * w2;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4q2 = 1.0 / (4.0 * q * q);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double H[3][3], Kp[3][3], Kq[3][3], L[3][3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          H[a][b] = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H[a][b] + 2.0 * p * I * dij) * inv4p2;
          Kq[a][b] = (H[a][b] + 2.0 * q * I * dij) * inv4q2;
          L[a][b] = -(H[a][b]) * inv4pq;
        }
      }
      const double QC[3] = {Qx - Cx, Qy - Cy, Qz - Cz};
      const double QD[3] = {Qx - Dx, Qy - Dy, Qz - Dz};
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
      const double term_t4_f2 = 4.0 * w2 * base * F2;
      const double term_t4_f3 = -8.0 * w3 * base * F3;
      const double term_t4_f4 = 16.0 * w4 * base * F4;
#pragma unroll 1
      for (int ia = 0; ia < 3; ++ia) {
#pragma unroll 1
        for (int ib = 0; ib < 3; ++ib) {
          const double a_ = PA[ia], b_ = PB[ib];
          const double dij = (ia == ib) ? 1.0 : 0.0;
          const double Kp_ij = Kp[ia][ib];
#pragma unroll 1
          for (int ic = 0; ic < 3; ++ic) {
#pragma unroll 1
            for (int id = 0; id < 3; ++id) {
              const double c_ = QC[ic], d_ = QD[id];
              const double dkl = (ic == id) ? 1.0 : 0.0;
              const double Kq_kl = Kq[ic][id];
              const double T3_ijk = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
              const double T3_ijl = t3_component(ia, ib, id, dvec, term_t3_f2, term_t3_f3);
              const double T3_i_kl = t3_component(ia, ic, id, dvec, term_t3_f2, term_t3_f3);
              const double T3_j_kl = t3_component(ib, ic, id, dvec, term_t3_f2, term_t3_f3);
              const double M_ijk = (-T3_ijk + 4.0*p*q*dij*Jq[ic]) / (8.0*p*p*q);
              const double M_ijl = (-T3_ijl + 4.0*p*q*dij*Jq[id]) / (8.0*p*p*q);
              const double N_i_kl = (T3_i_kl + 4.0*p*q*dkl*Jp[ia]) / (8.0*p*q*q);
              const double N_j_kl = (T3_j_kl + 4.0*p*q*dkl*Jp[ib]) / (8.0*p*q*q);
              const double T4_ijkl = t4_component(ia, ib, ic, id, dvec, term_t4_f2, term_t4_f3, term_t4_f4);
              const double M4_ij_kl = (T4_ijkl + 8.0*p*p*q*dkl*Kp_ij + 8.0*p*q*q*dij*Kq_kl - 4.0*p*q*dij*dkl*I) / (16.0*p*p*q*q);
              s[ia*27 + ib*9 + ic*3 + id] += M4_ij_kl + d_*M_ijk + c_*M_ijl + c_*d_*Kp_ij
                + b_*N_i_kl + b_*d_*L[ia][ic] + b_*c_*L[ia][id] + b_*c_*d_*Jp[ia]
                + a_*N_j_kl + a_*d_*L[ib][ic] + a_*c_*L[ib][id] + a_*c_*d_*Jp[ib]
                + a_*b_*Kq_kl + a_*b_*d_*Jq[ic] + a_*b_*c_*Jq[id] + a_*b_*c_*d_*I;
            }
          }
        }
      }
    }
  }
  const int out = t * 81;
#pragma unroll 1
  for (int i = 0; i < 81; ++i) eri_out[out + i] = s[i];
}

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
template <int WARPS_PER_BLOCK>
__global__ void KernelERI_ppps_warp_tiny_2phase(
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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double acc0 = 0.0;
  double acc1 = 0.0;

  for (int ii = 0; ii < nAB; ++ii) {
    const int ki = baseAB + ii;
    for (int jj = 0; jj < nCD; ++jj) {
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

inline cudaError_t launch_ppps_tiny_warp(
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
  const int threads_eff = sanitize_halfwarp_launch_threads(threads);
  const int warps_per_block = threads_eff >> 5;
  const int tasks_per_block = warps_per_block * 2;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;
  switch (warps_per_block) {
    case 1:
      KernelERI_ppps_warp_tiny_2phase<1><<<blocks, 32, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 2:
      KernelERI_ppps_warp_tiny_2phase<2><<<blocks, 64, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 3:
      KernelERI_ppps_warp_tiny_2phase<3><<<blocks, 96, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 4:
      KernelERI_ppps_warp_tiny_2phase<4><<<blocks, 128, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 5:
      KernelERI_ppps_warp_tiny_2phase<5><<<blocks, 160, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 6:
      KernelERI_ppps_warp_tiny_2phase<6><<<blocks, 192, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 7:
      KernelERI_ppps_warp_tiny_2phase<7><<<blocks, 224, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    case 8:
      KernelERI_ppps_warp_tiny_2phase<8><<<blocks, 256, 0, stream>>>(
          task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
      break;
    default:
      return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t cueri_eri_psss_launch_stream(
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
  (void)sp_B;
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_psss_flat<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_psss_multiblock_launch_stream(
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
  (void)partial_sums; (void)blocks_per_task;
  return cueri_eri_psss_launch_stream(task_spAB, task_spCD, ntasks, sp_A, sp_B,
      sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppss_launch_stream(
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
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_ppss_flat<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_ppss_multiblock_launch_stream(
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
  (void)partial_sums; (void)blocks_per_task;
  return cueri_eri_ppss_launch_stream(task_spAB, task_spCD, ntasks, sp_A, sp_B,
      sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_psps_launch_stream(
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
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_psps_flat<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_psps_multiblock_launch_stream(
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
  (void)partial_sums; (void)blocks_per_task;
  return cueri_eri_psps_launch_stream(task_spAB, task_spCD, ntasks, sp_A, sp_B,
      sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_dsss_launch_stream(
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
  (void)sp_B;
  if (ntasks <= 0) return ntasks == 0 ? cudaSuccess : cudaErrorInvalidValue;
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_dsss_flat<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_dsss_multiblock_launch_stream(
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
  (void)partial_sums; (void)blocks_per_task;
  return cueri_eri_dsss_launch_stream(task_spAB, task_spCD, ntasks, sp_A, sp_B,
      sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_launch_stream(
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
  if (ntasks <= 0) return cudaSuccess;
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_ppps_flat<<<blocks, threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_ppps_tiny_warp_launch_stream(
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
  return launch_ppps_tiny_warp(
      task_spAB, task_spCD, nullptr, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_ppps_tiny_warp_indexed_launch_stream(
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
  return launch_ppps_tiny_warp(
      task_spAB, task_spCD, out_task_idx, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);
}

extern "C" cudaError_t cueri_eri_pppp_launch_stream(
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
  if (ntasks <= 0) return cudaSuccess;
  const int blocks = (ntasks + threads - 1) / threads;
  KernelERI_pppp_flat<<<blocks, threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_jk_dsss_launch_stream(
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
    int n_bufs) {
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int warps_per_block = threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  const size_t shmem = static_cast<size_t>(warps_per_block) * 6u * sizeof(double);
  KernelFusedJK_dsss_warp<<<static_cast<unsigned int>(blocks), threads, shmem, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_fock_ppss_launch_stream(
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
    int n_bufs) {
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int warps_per_block = threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  const size_t shmem = static_cast<size_t>(warps_per_block) * 9u * sizeof(double);
  KernelFused_ppss_warp<true><<<static_cast<unsigned int>(blocks), threads, shmem, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_jk_ppss_launch_stream(
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
    int n_bufs) {
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int warps_per_block = threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  const size_t shmem = static_cast<size_t>(warps_per_block) * 9u * sizeof(double);
  KernelFusedJK_ppss_warp<<<static_cast<unsigned int>(blocks), threads, shmem, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_fock_psps_launch_stream(
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
    int n_bufs) {
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int launch_threads = sanitize_subwarp8_launch_threads(threads);
  const int warps_per_block = launch_threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int tasks_per_block = warps_per_block << 2;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;
  KernelFused_psps_subwarp8<true><<<static_cast<unsigned int>(blocks), launch_threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      shell_ao_start, nao, D_mat, F_mat, nullptr, n_bufs);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fused_jk_psps_launch_stream(
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
    int n_bufs) {
  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int launch_threads = sanitize_subwarp8_launch_threads(threads);
  const int warps_per_block = launch_threads >> 5;
  if (warps_per_block <= 0) return cudaErrorInvalidValue;
  const int tasks_per_block = warps_per_block << 2;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;
  KernelFused_psps_subwarp8<false><<<static_cast<unsigned int>(blocks), launch_threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);
  return cudaGetLastError();
}

