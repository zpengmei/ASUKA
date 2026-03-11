// Auto-split from cueri_cuda_kernels_step2.cu (part 2/4: KernelERI_ppss_warp..KernelERI_dsss_multiblock_reduce)
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











__global__ void KernelERI_ppss_warp(
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

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];
  const double Bx = shell_cx[B];
  const double By = shell_cy[B];
  const double Bz = shell_cz[B];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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
    const double PBx = Px - Bx;
    const double PBy = Py - By;
    const double PBz = Pz - Bz;

    s00 += Kxx + PAx * Jx + PBx * Jx + (PAx * PBx) * I;
    s01 += Kxy + PAx * Jy + PBy * Jx + (PAx * PBy) * I;
    s02 += Kxz + PAx * Jz + PBz * Jx + (PAx * PBz) * I;

    s10 += Kxy + PAy * Jx + PBx * Jy + (PAy * PBx) * I;
    s11 += Kyy + PAy * Jy + PBy * Jy + (PAy * PBy) * I;
    s12 += Kyz + PAy * Jz + PBz * Jy + (PAy * PBz) * I;

    s20 += Kxz + PAz * Jx + PBx * Jz + (PAz * PBx) * I;
    s21 += Kyz + PAz * Jy + PBy * Jz + (PAz * PBy) * I;
    s22 += Kzz + PAz * Jz + PBz * Jz + (PAz * PBz) * I;
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
    const int out = t * 9;
    eri_out[out + 0] = s00;
    eri_out[out + 1] = s01;
    eri_out[out + 2] = s02;
    eri_out[out + 3] = s10;
    eri_out[out + 4] = s11;
    eri_out[out + 5] = s12;
    eri_out[out + 6] = s20;
    eri_out[out + 7] = s21;
    eri_out[out + 8] = s22;
  }
}

__global__ void KernelERI_ppss_subwarp8(
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
  // 4 tasks per warp, 8 lanes per task (subwarp8).
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;

  const int subwarp = lane >> 3;  // 0..3
  const int lane8 = lane & 7;

  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int64_t u = static_cast<int64_t>(lane8); u < nTot; u += 8) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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
    const double PBx = Px - Bx;
    const double PBy = Py - By;
    const double PBz = Pz - Bz;

    s00 += Kxx + PAx * Jx + PBx * Jx + (PAx * PBx) * I;
    s01 += Kxy + PAx * Jy + PBy * Jx + (PAx * PBy) * I;
    s02 += Kxz + PAx * Jz + PBz * Jx + (PAx * PBz) * I;

    s10 += Kxy + PAy * Jx + PBx * Jy + (PAy * PBx) * I;
    s11 += Kyy + PAy * Jy + PBy * Jy + (PAy * PBy) * I;
    s12 += Kyz + PAy * Jz + PBz * Jy + (PAy * PBz) * I;

    s20 += Kxz + PAz * Jx + PBx * Jz + (PAz * PBx) * I;
    s21 += Kyz + PAz * Jy + PBy * Jz + (PAz * PBy) * I;
    s22 += Kzz + PAz * Jz + PBz * Jz + (PAz * PBz) * I;
  }

  s00 += __shfl_down_sync(0xffffffff, s00, 4, 8);
  s00 += __shfl_down_sync(0xffffffff, s00, 2, 8);
  s00 += __shfl_down_sync(0xffffffff, s00, 1, 8);
  s01 += __shfl_down_sync(0xffffffff, s01, 4, 8);
  s01 += __shfl_down_sync(0xffffffff, s01, 2, 8);
  s01 += __shfl_down_sync(0xffffffff, s01, 1, 8);
  s02 += __shfl_down_sync(0xffffffff, s02, 4, 8);
  s02 += __shfl_down_sync(0xffffffff, s02, 2, 8);
  s02 += __shfl_down_sync(0xffffffff, s02, 1, 8);

  s10 += __shfl_down_sync(0xffffffff, s10, 4, 8);
  s10 += __shfl_down_sync(0xffffffff, s10, 2, 8);
  s10 += __shfl_down_sync(0xffffffff, s10, 1, 8);
  s11 += __shfl_down_sync(0xffffffff, s11, 4, 8);
  s11 += __shfl_down_sync(0xffffffff, s11, 2, 8);
  s11 += __shfl_down_sync(0xffffffff, s11, 1, 8);
  s12 += __shfl_down_sync(0xffffffff, s12, 4, 8);
  s12 += __shfl_down_sync(0xffffffff, s12, 2, 8);
  s12 += __shfl_down_sync(0xffffffff, s12, 1, 8);

  s20 += __shfl_down_sync(0xffffffff, s20, 4, 8);
  s20 += __shfl_down_sync(0xffffffff, s20, 2, 8);
  s20 += __shfl_down_sync(0xffffffff, s20, 1, 8);
  s21 += __shfl_down_sync(0xffffffff, s21, 4, 8);
  s21 += __shfl_down_sync(0xffffffff, s21, 2, 8);
  s21 += __shfl_down_sync(0xffffffff, s21, 1, 8);
  s22 += __shfl_down_sync(0xffffffff, s22, 4, 8);
  s22 += __shfl_down_sync(0xffffffff, s22, 2, 8);
  s22 += __shfl_down_sync(0xffffffff, s22, 1, 8);

  if (lane8 == 0) {
    const int out = t * 9;
    eri_out[out + 0] = s00;
    eri_out[out + 1] = s01;
    eri_out[out + 2] = s02;
    eri_out[out + 3] = s10;
    eri_out[out + 4] = s11;
    eri_out[out + 5] = s12;
    eri_out[out + 6] = s20;
    eri_out[out + 7] = s21;
    eri_out[out + 8] = s22;
  }
}

__global__ void KernelERI_dsss_warp(
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

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yy = 0.0;
  double s_yz = 0.0;
  double s_zz = 0.0;

  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    for (int j = lane; j < nCD; j += 32) {
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
  }

  s_xx = warp_reduce_sum(s_xx);
  s_xy = warp_reduce_sum(s_xy);
  s_xz = warp_reduce_sum(s_xz);
  s_yy = warp_reduce_sum(s_yy);
  s_yz = warp_reduce_sum(s_yz);
  s_zz = warp_reduce_sum(s_zz);

  if (lane == 0) {
    const int out = t * 6;
    eri_out[out + 0] = s_xx;
    eri_out[out + 1] = s_xy;
    eri_out[out + 2] = s_xz;
    eri_out[out + 3] = s_yy;
    eri_out[out + 4] = s_yz;
    eri_out[out + 5] = s_zz;
  }
  (void)sp_B;
}

__global__ void KernelERI_dsss_subwarp8(
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
  // 4 tasks per warp, 8 lanes per task (subwarp8).
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;

  const int subwarp = lane >> 3;  // 0..3
  const int lane8 = lane & 7;

  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yy = 0.0;
  double s_yz = 0.0;
  double s_zz = 0.0;

  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    for (int j = lane8; j < nCD; j += 8) {
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
  }

  s_xx += __shfl_down_sync(0xffffffff, s_xx, 4, 8);
  s_xx += __shfl_down_sync(0xffffffff, s_xx, 2, 8);
  s_xx += __shfl_down_sync(0xffffffff, s_xx, 1, 8);

  s_xy += __shfl_down_sync(0xffffffff, s_xy, 4, 8);
  s_xy += __shfl_down_sync(0xffffffff, s_xy, 2, 8);
  s_xy += __shfl_down_sync(0xffffffff, s_xy, 1, 8);

  s_xz += __shfl_down_sync(0xffffffff, s_xz, 4, 8);
  s_xz += __shfl_down_sync(0xffffffff, s_xz, 2, 8);
  s_xz += __shfl_down_sync(0xffffffff, s_xz, 1, 8);

  s_yy += __shfl_down_sync(0xffffffff, s_yy, 4, 8);
  s_yy += __shfl_down_sync(0xffffffff, s_yy, 2, 8);
  s_yy += __shfl_down_sync(0xffffffff, s_yy, 1, 8);

  s_yz += __shfl_down_sync(0xffffffff, s_yz, 4, 8);
  s_yz += __shfl_down_sync(0xffffffff, s_yz, 2, 8);
  s_yz += __shfl_down_sync(0xffffffff, s_yz, 1, 8);

  s_zz += __shfl_down_sync(0xffffffff, s_zz, 4, 8);
  s_zz += __shfl_down_sync(0xffffffff, s_zz, 2, 8);
  s_zz += __shfl_down_sync(0xffffffff, s_zz, 1, 8);

  if (lane8 == 0) {
    const int out = t * 6;
    eri_out[out + 0] = s_xx;
    eri_out[out + 1] = s_xy;
    eri_out[out + 2] = s_xz;
    eri_out[out + 3] = s_yy;
    eri_out[out + 4] = s_yz;
    eri_out[out + 5] = s_zz;
  }
  (void)sp_B;
}

__global__ void KernelERI_psss_multiblock_partial(
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
    int blocks_per_task,
    double* partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  if (t >= ntasks || b >= blocks_per_task) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blocks_per_task);
  int64_t u = static_cast<int64_t>(b) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (; u < nTot; u += stride) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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

    double F0, F1;
    boys_f0_f1(T, F0, F1);

    const double q_over = q / denom;  // omega/p
    sx += base * (-(Ax - Px) * F0 - q_over * dx * F1);
    sy += base * (-(Ay - Py) * F0 - q_over * dy * F1);
    sz += base * (-(Az - Pz) * F0 - q_over * dz * F1);
  }

  sx = block_reduce_sum(sx);
  sy = block_reduce_sum(sy);
  sz = block_reduce_sum(sz);
  if (threadIdx.x == 0) {
    const int out = (t * blocks_per_task + b) * 3;
    partial_sums[out + 0] = sx;
    partial_sums[out + 1] = sy;
    partial_sums[out + 2] = sz;
  }
  (void)sp_B;
}

__global__ void KernelERI_psss_multiblock_reduce(const double* partial_sums, int blocks_per_task, double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (int b = static_cast<int>(threadIdx.x); b < blocks_per_task; b += static_cast<int>(blockDim.x)) {
    const int in = (t * blocks_per_task + b) * 3;
    sx += partial_sums[in + 0];
    sy += partial_sums[in + 1];
    sz += partial_sums[in + 2];
  }
  sx = block_reduce_sum(sx);
  sy = block_reduce_sum(sy);
  sz = block_reduce_sum(sz);
  if (threadIdx.x == 0) {
    const int out = t * 3;
    eri_out[out + 0] = sx;
    eri_out[out + 1] = sy;
    eri_out[out + 2] = sz;
  }
}

__global__ void KernelERI_ppss_multiblock_partial(
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
    int blocks_per_task,
    double* partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  if (t >= ntasks || b >= blocks_per_task) return;

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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blocks_per_task);
  int64_t u = static_cast<int64_t>(b) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (; u < nTot; u += stride) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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
    const double PBx = Px - Bx;
    const double PBy = Py - By;
    const double PBz = Pz - Bz;

    s00 += Kxx + PAx * Jx + PBx * Jx + (PAx * PBx) * I;
    s01 += Kxy + PAx * Jy + PBy * Jx + (PAx * PBy) * I;
    s02 += Kxz + PAx * Jz + PBz * Jx + (PAx * PBz) * I;

    s10 += Kxy + PAy * Jx + PBx * Jy + (PAy * PBx) * I;
    s11 += Kyy + PAy * Jy + PBy * Jy + (PAy * PBy) * I;
    s12 += Kyz + PAy * Jz + PBz * Jy + (PAy * PBz) * I;

    s20 += Kxz + PAz * Jx + PBx * Jz + (PAz * PBx) * I;
    s21 += Kyz + PAz * Jy + PBy * Jz + (PAz * PBy) * I;
    s22 += Kzz + PAz * Jz + PBz * Jz + (PAz * PBz) * I;
  }

  s00 = block_reduce_sum(s00);
  s01 = block_reduce_sum(s01);
  s02 = block_reduce_sum(s02);
  s10 = block_reduce_sum(s10);
  s11 = block_reduce_sum(s11);
  s12 = block_reduce_sum(s12);
  s20 = block_reduce_sum(s20);
  s21 = block_reduce_sum(s21);
  s22 = block_reduce_sum(s22);

  if (threadIdx.x == 0) {
    const int out = (t * blocks_per_task + b) * 9;
    partial_sums[out + 0] = s00;
    partial_sums[out + 1] = s01;
    partial_sums[out + 2] = s02;
    partial_sums[out + 3] = s10;
    partial_sums[out + 4] = s11;
    partial_sums[out + 5] = s12;
    partial_sums[out + 6] = s20;
    partial_sums[out + 7] = s21;
    partial_sums[out + 8] = s22;
  }
}

__global__ void KernelERI_ppss_multiblock_reduce(const double* partial_sums, int blocks_per_task, double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (int b = static_cast<int>(threadIdx.x); b < blocks_per_task; b += static_cast<int>(blockDim.x)) {
    const int in = (t * blocks_per_task + b) * 9;
    s00 += partial_sums[in + 0];
    s01 += partial_sums[in + 1];
    s02 += partial_sums[in + 2];
    s10 += partial_sums[in + 3];
    s11 += partial_sums[in + 4];
    s12 += partial_sums[in + 5];
    s20 += partial_sums[in + 6];
    s21 += partial_sums[in + 7];
    s22 += partial_sums[in + 8];
  }

  s00 = block_reduce_sum(s00);
  s01 = block_reduce_sum(s01);
  s02 = block_reduce_sum(s02);
  s10 = block_reduce_sum(s10);
  s11 = block_reduce_sum(s11);
  s12 = block_reduce_sum(s12);
  s20 = block_reduce_sum(s20);
  s21 = block_reduce_sum(s21);
  s22 = block_reduce_sum(s22);

  if (threadIdx.x == 0) {
    const int out = t * 9;
    eri_out[out + 0] = s00;
    eri_out[out + 1] = s01;
    eri_out[out + 2] = s02;
    eri_out[out + 3] = s10;
    eri_out[out + 4] = s11;
    eri_out[out + 5] = s12;
    eri_out[out + 6] = s20;
    eri_out[out + 7] = s21;
    eri_out[out + 8] = s22;
  }
}

__global__ void KernelERI_psps_multiblock_partial(
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
    int blocks_per_task,
    double* partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  if (t >= ntasks || b >= blocks_per_task) return;

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
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blocks_per_task);
  int64_t u = static_cast<int64_t>(b) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  for (; u < nTot; u += stride) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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

  s00 = block_reduce_sum(s00);
  s01 = block_reduce_sum(s01);
  s02 = block_reduce_sum(s02);
  s10 = block_reduce_sum(s10);
  s11 = block_reduce_sum(s11);
  s12 = block_reduce_sum(s12);
  s20 = block_reduce_sum(s20);
  s21 = block_reduce_sum(s21);
  s22 = block_reduce_sum(s22);

  if (threadIdx.x == 0) {
    const int out = (t * blocks_per_task + b) * 9;
    partial_sums[out + 0] = s00;
    partial_sums[out + 1] = s01;
    partial_sums[out + 2] = s02;
    partial_sums[out + 3] = s10;
    partial_sums[out + 4] = s11;
    partial_sums[out + 5] = s12;
    partial_sums[out + 6] = s20;
    partial_sums[out + 7] = s21;
    partial_sums[out + 8] = s22;
  }
  (void)sp_B;
}

__global__ void KernelERI_psps_multiblock_reduce(const double* partial_sums, int blocks_per_task, double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  for (int e = static_cast<int>(threadIdx.x); e < 9; e += static_cast<int>(blockDim.x)) {
    double s = 0.0;
    for (int b = 0; b < blocks_per_task; ++b) {
      const int in = (t * blocks_per_task + b) * 9 + e;
      s += partial_sums[in];
    }
    eri_out[t * 9 + e] = s;
  }
}

__global__ void KernelERI_dsss_multiblock_partial(
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
    int blocks_per_task,
    double* partial_sums) {
  const int t = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  if (t >= ntasks || b >= blocks_per_task) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blocks_per_task);
  int64_t u = static_cast<int64_t>(b) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yy = 0.0;
  double s_yz = 0.0;
  double s_zz = 0.0;

  for (; u < nTot; u += stride) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
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

  s_xx = block_reduce_sum(s_xx);
  s_xy = block_reduce_sum(s_xy);
  s_xz = block_reduce_sum(s_xz);
  s_yy = block_reduce_sum(s_yy);
  s_yz = block_reduce_sum(s_yz);
  s_zz = block_reduce_sum(s_zz);

  if (threadIdx.x == 0) {
    const int out = (t * blocks_per_task + b) * 6;
    partial_sums[out + 0] = s_xx;
    partial_sums[out + 1] = s_xy;
    partial_sums[out + 2] = s_xz;
    partial_sums[out + 3] = s_yy;
    partial_sums[out + 4] = s_yz;
    partial_sums[out + 5] = s_zz;
  }
  (void)sp_B;
}

__global__ void KernelERI_dsss_multiblock_reduce(const double* partial_sums, int blocks_per_task, double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yy = 0.0;
  double s_yz = 0.0;
  double s_zz = 0.0;

  for (int b = static_cast<int>(threadIdx.x); b < blocks_per_task; b += static_cast<int>(blockDim.x)) {
    const int in = (t * blocks_per_task + b) * 6;
    s_xx += partial_sums[in + 0];
    s_xy += partial_sums[in + 1];
    s_xz += partial_sums[in + 2];
    s_yy += partial_sums[in + 3];
    s_yz += partial_sums[in + 4];
    s_zz += partial_sums[in + 5];
  }

  s_xx = block_reduce_sum(s_xx);
  s_xy = block_reduce_sum(s_xy);
  s_xz = block_reduce_sum(s_xz);
  s_yy = block_reduce_sum(s_yy);
  s_yz = block_reduce_sum(s_yz);
  s_zz = block_reduce_sum(s_zz);

  if (threadIdx.x == 0) {
    const int out = t * 6;
    eri_out[out + 0] = s_xx;
    eri_out[out + 1] = s_xy;
    eri_out[out + 2] = s_xz;
    eri_out[out + 3] = s_yy;
    eri_out[out + 4] = s_yz;
    eri_out[out + 5] = s_zz;
  }
}

}  // namespace

