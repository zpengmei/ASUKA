#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cueri_cuda_kernels_api.h"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtPi = 1.772453850905516027298167483341145182;
constexpr double kTwoPiToFiveHalves = 2.0 * kPi * kPi * kSqrtPi;  // 2*pi^(5/2)

__device__ inline double boys_f0_ref(double T) {
  if (T < 1e-12) {
    return 1.0 - (T / 3.0) + (T * T / 10.0);
  }
  return 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
}

__device__ inline double boys_f0_fast(double T) {
  if (T < 1e-12) {
    return 1.0 - (T / 3.0) + (T * T / 10.0);
  }
  const float Tf = static_cast<float>(T);
  const float u = ::sqrtf(Tf);
  const float ef = ::erff(u);
  return 0.5 * ::sqrt(kPi / T) * static_cast<double>(ef);
}

template <bool kFastBoys>
__device__ inline double boys_f0(double T) {
  if constexpr (kFastBoys) {
    return boys_f0_fast(T);
  } else {
    return boys_f0_ref(T);
  }
}

__device__ inline double warp_reduce_sum(double x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

__device__ inline double block_reduce_sum(double x) {
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

__global__ void KernelBuildPairTables_ss(
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* prim_coef,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    double* pair_eta,
    double* pair_Px,
    double* pair_Py,
    double* pair_Pz,
    double* pair_cK) {
  const int sp = static_cast<int>(blockIdx.x);
  const int A = static_cast<int>(sp_A[sp]);
  const int B = static_cast<int>(sp_B[sp]);
  const int base = static_cast<int>(sp_pair_start[sp]);
  const int nB = static_cast<int>(shell_nprim[B]);
  const int startA = static_cast<int>(shell_prim_start[A]);
  const int startB = static_cast<int>(shell_prim_start[B]);

  const double Ax = shell_cx[A];
  const double Ay = shell_cy[A];
  const double Az = shell_cz[A];
  const double Bx = shell_cx[B];
  const double By = shell_cy[B];
  const double Bz = shell_cz[B];
  const double dx = Ax - Bx;
  const double dy = Ay - By;
  const double dz = Az - Bz;
  const double AB2 = dx * dx + dy * dy + dz * dz;

  const int npair = static_cast<int>(sp_npair[sp]);
  for (int idx = static_cast<int>(threadIdx.x); idx < npair; idx += static_cast<int>(blockDim.x)) {
    const int ia = idx / nB;
    const int ib = idx - ia * nB;
    const int pA = startA + ia;
    const int pB = startB + ib;

    const double alpha = prim_exp[pA];
    const double beta = prim_exp[pB];
    const double eta = alpha + beta;
    const double inv_eta = 1.0 / eta;
    const double Px = (alpha * Ax + beta * Bx) * inv_eta;
    const double Py = (alpha * Ay + beta * By) * inv_eta;
    const double Pz = (alpha * Az + beta * Bz) * inv_eta;

    const double mu = (alpha * beta) * inv_eta;
    const double Kab = ::exp(-mu * AB2);
    const double cK = prim_coef[pA] * prim_coef[pB] * Kab;

    const int k = base + idx;
    pair_eta[k] = eta;
    pair_Px[k] = Px;
    pair_Py[k] = Py;
    pair_Pz[k] = Pz;
    pair_cK[k] = cK;
  }
}

template <bool kFastBoys>
__global__ void KernelSchwarz_ssss_t(
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* sp_Q) {
  const int sp = static_cast<int>(blockIdx.x);
  const int base = static_cast<int>(sp_pair_start[sp]);
  const int nP = static_cast<int>(sp_npair[sp]);
  const int64_t nTot = static_cast<int64_t>(nP) * static_cast<int64_t>(nP);

  double sum = 0.0;
  for (int64_t t = static_cast<int64_t>(threadIdx.x); t < nTot; t += static_cast<int64_t>(blockDim.x)) {
    const int i = static_cast<int>(t / nP);
    const int j = static_cast<int>(t - static_cast<int64_t>(i) * nP);
    const int ki = base + i;
    const int kj = base + j;
    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];

    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) sp_Q[sp] = ::sqrt(fmax(sum, 0.0));
}

template <bool kFastBoys>
__global__ void KernelERI_ssss(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(threadIdx.x); u < nTot; u += static_cast<int64_t>(blockDim.x)) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];

    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) eri_out[t] = sum;
}

// Fused ssss+JK kernel: evaluates (ss|ss) ERI scalar inline and immediately
// accumulates J and K, eliminating the tile global-memory round-trip.
// Uses 1 warp per task (ntasks blocks × 32 threads).  sp_A/sp_B/shell_ao_start
// are the same arrays passed to cueri_contract_jk_warp_launch_stream.
template <bool kFastBoys>
__global__ void KernelFusedJKSsss(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    int warps_per_block,
    int n_bufs) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  // Compute ERI scalar via warp reduction (identical to KernelERI_ssss_warp)
  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;
    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];
    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }
  sum = warp_reduce_sum(sum);

  // Only lane 0 has the correct reduction result
  if (lane == 0 && sum != 0.0) {
    const int A_sh = static_cast<int>(sp_A[spAB]);
    const int B_sh = static_cast<int>(sp_B[spAB]);
    const int C_sh = static_cast<int>(sp_A[spCD]);
    const int D_sh = static_cast<int>(sp_B[spCD]);
    const int a = static_cast<int>(shell_ao_start[A_sh]);
    const int b = static_cast<int>(shell_ao_start[B_sh]);
    const int c = static_cast<int>(shell_ao_start[C_sh]);
    const int d = static_cast<int>(shell_ao_start[D_sh]);
    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);
    const bool bk_swap = (spAB != spCD);
    const double f_ab = ab_neq ? 2.0 : 1.0;
    const double f_cd = cd_neq ? 2.0 : 1.0;
    const int64_t N = static_cast<int64_t>(nao);

    // Spread accumulation buffers per task rather than per block so warps in
    // the same block do not contend on the same J/K backing matrix.
    const int buf_id = t % n_bufs;
    const int64_t buf_off = static_cast<int64_t>(buf_id) * N * N;
    if (J_mat != nullptr) { J_mat = J_mat + buf_off; }
    if (K_mat != nullptr) { K_mat = K_mat + buf_off; }

    if (J_mat != nullptr) {
      const double Dcd = D_mat[c * N + d];
      atomicAdd(&J_mat[a * N + b], f_cd * sum * Dcd);
      if (ab_neq) atomicAdd(&J_mat[b * N + a], f_cd * sum * Dcd);
      if (bk_swap) {
        const double Dab = D_mat[a * N + b];
        atomicAdd(&J_mat[c * N + d], f_ab * sum * Dab);
        if (cd_neq) atomicAdd(&J_mat[d * N + c], f_ab * sum * Dab);
      }
    }
    if (K_mat != nullptr) {
      atomicAdd(&K_mat[a * N + c], sum * D_mat[b * N + d]);
      if (cd_neq) atomicAdd(&K_mat[a * N + d], sum * D_mat[b * N + c]);
      if (ab_neq) atomicAdd(&K_mat[b * N + c], sum * D_mat[a * N + d]);
      if (ab_neq && cd_neq) atomicAdd(&K_mat[b * N + d], sum * D_mat[a * N + c]);
      if (bk_swap) {
        atomicAdd(&K_mat[c * N + a], sum * D_mat[d * N + b]);
        if (cd_neq) atomicAdd(&K_mat[d * N + a], sum * D_mat[c * N + b]);
        if (ab_neq) atomicAdd(&K_mat[c * N + b], sum * D_mat[d * N + a]);
        if (ab_neq && cd_neq) atomicAdd(&K_mat[d * N + b], sum * D_mat[c * N + a]);
      }
    }
  }
}

template <bool kFastBoys>
__global__ void KernelFusedJKSsssSubwarp8(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    int warps_per_block,
    int n_bufs) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  const int subwarp = lane >> 3;
  const int lane8 = lane & 7;
  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(lane8); u < nTot; u += 8) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;
    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];
    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

  if (lane8 == 0 && sum != 0.0) {
    const int A_sh = static_cast<int>(sp_A[spAB]);
    const int B_sh = static_cast<int>(sp_B[spAB]);
    const int C_sh = static_cast<int>(sp_A[spCD]);
    const int D_sh = static_cast<int>(sp_B[spCD]);
    const int a = static_cast<int>(shell_ao_start[A_sh]);
    const int b = static_cast<int>(shell_ao_start[B_sh]);
    const int c = static_cast<int>(shell_ao_start[C_sh]);
    const int d = static_cast<int>(shell_ao_start[D_sh]);
    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);
    const bool bk_swap = (spAB != spCD);
    const double f_ab = ab_neq ? 2.0 : 1.0;
    const double f_cd = cd_neq ? 2.0 : 1.0;
    const int64_t N = static_cast<int64_t>(nao);

    const int buf_id = t % n_bufs;
    const int64_t buf_off = static_cast<int64_t>(buf_id) * N * N;
    if (J_mat != nullptr) J_mat += buf_off;
    if (K_mat != nullptr) K_mat += buf_off;

    if (J_mat != nullptr) {
      const double Dcd = D_mat[c * N + d];
      atomicAdd(&J_mat[a * N + b], f_cd * sum * Dcd);
      if (ab_neq) atomicAdd(&J_mat[b * N + a], f_cd * sum * Dcd);
      if (bk_swap) {
        const double Dab = D_mat[a * N + b];
        atomicAdd(&J_mat[c * N + d], f_ab * sum * Dab);
        if (cd_neq) atomicAdd(&J_mat[d * N + c], f_ab * sum * Dab);
      }
    }
    if (K_mat != nullptr) {
      atomicAdd(&K_mat[a * N + c], sum * D_mat[b * N + d]);
      if (cd_neq) atomicAdd(&K_mat[a * N + d], sum * D_mat[b * N + c]);
      if (ab_neq) atomicAdd(&K_mat[b * N + c], sum * D_mat[a * N + d]);
      if (ab_neq && cd_neq) atomicAdd(&K_mat[b * N + d], sum * D_mat[a * N + c]);
      if (bk_swap) {
        atomicAdd(&K_mat[c * N + a], sum * D_mat[d * N + b]);
        if (cd_neq) atomicAdd(&K_mat[d * N + a], sum * D_mat[c * N + b]);
        if (ab_neq) atomicAdd(&K_mat[c * N + b], sum * D_mat[d * N + a]);
        if (ab_neq && cd_neq) atomicAdd(&K_mat[d * N + b], sum * D_mat[c * N + a]);
      }
    }
  }
}

// Fused ssss->Fock: evaluates (ss|ss) ERI and accumulates F = J - 0.5*K.
// Same warp-per-task pattern as KernelFusedJKSsss.
template <bool kFastBoys>
__global__ void KernelFusedFock_ssss_warp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* F_mat,
    int warps_per_block,
    int n_bufs) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;
    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];
    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }
  sum = warp_reduce_sum(sum);

  if (lane == 0 && sum != 0.0) {
    const int A_sh = static_cast<int>(sp_A[spAB]);
    const int B_sh = static_cast<int>(sp_B[spAB]);
    const int C_sh = static_cast<int>(sp_A[spCD]);
    const int D_sh = static_cast<int>(sp_B[spCD]);
    const int a = static_cast<int>(shell_ao_start[A_sh]);
    const int b = static_cast<int>(shell_ao_start[B_sh]);
    const int c = static_cast<int>(shell_ao_start[C_sh]);
    const int d = static_cast<int>(shell_ao_start[D_sh]);
    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);
    const bool bk_swap = (spAB != spCD);
    const double f_cd = cd_neq ? 2.0 : 1.0;
    const double f_ab = ab_neq ? 2.0 : 1.0;
    const int64_t N = static_cast<int64_t>(nao);

    // Spread accumulation buffers per task rather than per block so warps in
    // the same block do not contend on the same Fock backing matrix.
    const int buf_id = t % n_bufs;
    F_mat = F_mat + static_cast<int64_t>(buf_id) * N * N;

    // J contribution: F[a,b] += f_cd * sum * D[c,d]
    const double Dcd = D_mat[c * N + d];
    atomicAdd(&F_mat[a * N + b], f_cd * sum * Dcd);
    if (ab_neq) atomicAdd(&F_mat[b * N + a], f_cd * sum * Dcd);
    if (bk_swap) {
      const double Dab = D_mat[a * N + b];
      atomicAdd(&F_mat[c * N + d], f_ab * sum * Dab);
      if (cd_neq) atomicAdd(&F_mat[d * N + c], f_ab * sum * Dab);
    }

    // -0.5*K contribution
    const double alpha = -0.5;
    atomicAdd(&F_mat[a * N + c], alpha * sum * D_mat[b * N + d]);
    if (cd_neq) atomicAdd(&F_mat[a * N + d], alpha * sum * D_mat[b * N + c]);
    if (ab_neq) atomicAdd(&F_mat[b * N + c], alpha * sum * D_mat[a * N + d]);
    if (ab_neq && cd_neq) atomicAdd(&F_mat[b * N + d], alpha * sum * D_mat[a * N + c]);
    if (bk_swap) {
      atomicAdd(&F_mat[c * N + a], alpha * sum * D_mat[d * N + b]);
      if (cd_neq) atomicAdd(&F_mat[d * N + a], alpha * sum * D_mat[c * N + b]);
      if (ab_neq) atomicAdd(&F_mat[c * N + b], alpha * sum * D_mat[d * N + a]);
      if (ab_neq && cd_neq) atomicAdd(&F_mat[d * N + b], alpha * sum * D_mat[c * N + a]);
    }
  }
}

template <bool kFastBoys>
__global__ void KernelERI_ssss_warp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
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
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(lane); u < nTot; u += 32) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];

    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum = warp_reduce_sum(sum);
  if (lane == 0) eri_out[t] = sum;
}

template <bool kFastBoys>
__global__ void KernelERI_ssss_subwarp8(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
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
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  double sum = 0.0;
  for (int64_t u = static_cast<int64_t>(lane8); u < nTot; u += 8) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];

    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 1, 8);
  if (lane8 == 0) eri_out[t] = sum;
}

template <bool kFastBoys>
__global__ void KernelERI_ssss_multiblock_partial(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
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
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);
  const int64_t nTot = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blocks_per_task);
  int64_t u = static_cast<int64_t>(b) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  double sum = 0.0;
  for (; u < nTot; u += stride) {
    const int i = static_cast<int>(u / nCD);
    const int j = static_cast<int>(u - static_cast<int64_t>(i) * nCD);
    const int ki = baseAB + i;
    const int kj = baseCD + j;

    const double eta = pair_eta[ki];
    const double zeta = pair_eta[kj];

    const double dx = pair_Px[ki] - pair_Px[kj];
    const double dy = pair_Py[ki] - pair_Py[kj];
    const double dz = pair_Pz[ki] - pair_Pz[kj];
    const double PQ2 = dx * dx + dy * dy + dz * dz;

    const double denom = eta + zeta;
    const double omega = eta * zeta / denom;
    const double T = omega * PQ2;
    const double pref = kTwoPiToFiveHalves / (eta * zeta * ::sqrt(denom));
    sum += pref * pair_cK[ki] * pair_cK[kj] * boys_f0<kFastBoys>(T);
  }

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) partial_sums[t * blocks_per_task + b] = sum;
}

__global__ void KernelERI_ssss_multiblock_reduce(const double* partial_sums, int blocks_per_task, double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  double sum = 0.0;
  for (int b = static_cast<int>(threadIdx.x); b < blocks_per_task; b += static_cast<int>(blockDim.x)) {
    sum += partial_sums[t * blocks_per_task + b];
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) eri_out[t] = sum;
}

__global__ void KernelCountEntriesPerKey(
    const int32_t* task_spAB, const int32_t* task_spCD, int ntasks, int32_t* counts) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;
  const int32_t ab = task_spAB[t];
  const int32_t cd = task_spCD[t];
  atomicAdd(&counts[ab], 1);
  if (ab != cd) atomicAdd(&counts[cd], 1);
}

__global__ void KernelFillEntryCSR(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* entry_offsets,
    int32_t* cursor,
    int32_t* entry_task,
    int32_t* entry_widx) {
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= ntasks) return;
  const int32_t ab = task_spAB[t];
  const int32_t cd = task_spCD[t];

  int32_t pos = atomicAdd(&cursor[ab], 1);
  int32_t out = entry_offsets[ab] + pos;
  entry_task[out] = t;
  entry_widx[out] = cd;

  if (ab != cd) {
    pos = atomicAdd(&cursor[cd], 1);
    out = entry_offsets[cd] + pos;
    entry_task[out] = t;
    entry_widx[out] = ab;
  }
}

__global__ void KernelReduceFromEntryCSR(
    const int32_t* entry_offsets,
    const int32_t* entry_task,
    const int32_t* entry_widx,
    const double* eri_task,
    const double* W,
    double* Out) {
  const int key = static_cast<int>(blockIdx.x);
  const int begin = static_cast<int>(entry_offsets[key]);
  const int end = static_cast<int>(entry_offsets[key + 1]);

  double sum = 0.0;
  for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += static_cast<int>(blockDim.x)) {
    sum += W[entry_widx[i]] * eri_task[entry_task[i]];
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) Out[key] = sum;
}

}  // namespace

extern "C" cudaError_t cueri_build_pair_tables_ss_launch_stream(
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* prim_coef,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    int nsp,
    double* pair_eta,
    double* pair_Px,
    double* pair_Py,
    double* pair_Pz,
    double* pair_cK,
    cudaStream_t stream,
    int threads) {
  KernelBuildPairTables_ss<<<static_cast<unsigned int>(nsp), threads, 0, stream>>>(
      shell_cx,
      shell_cy,
      shell_cz,
      shell_prim_start,
      shell_nprim,
      prim_exp,
      prim_coef,
      sp_A,
      sp_B,
      sp_pair_start,
      sp_npair,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_schwarz_ssss_launch_stream(
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    int nsp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* sp_Q,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys) {
  if (use_fast_boys) {
    KernelSchwarz_ssss_t<true><<<static_cast<unsigned int>(nsp), threads, 0, stream>>>(
        sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, sp_Q);
  } else {
    KernelSchwarz_ssss_t<false><<<static_cast<unsigned int>(nsp), threads, 0, stream>>>(
        sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, sp_Q);
  }
  return cudaGetLastError();
}

// Flat ssss kernel: 1 thread per task, no shared memory / reduction.
template <bool kFastBoys>
__global__ void KernelERI_ssss_flat(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
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
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nAB = static_cast<int>(sp_npair[spAB]);
  const int nCD = static_cast<int>(sp_npair[spCD]);

  double sum = 0.0;
  for (int i = 0; i < nAB; ++i) {
    const int ki = baseAB + i;
    const double eta_i = pair_eta[ki];
    const double cKi = pair_cK[ki];
    const double Pxi = pair_Px[ki];
    const double Pyi = pair_Py[ki];
    const double Pzi = pair_Pz[ki];
    for (int j = 0; j < nCD; ++j) {
      const int kj = baseCD + j;
      const double zeta = pair_eta[kj];
      const double dx = Pxi - pair_Px[kj];
      const double dy = Pyi - pair_Py[kj];
      const double dz = Pzi - pair_Pz[kj];
      const double PQ2 = dx * dx + dy * dy + dz * dz;
      const double denom = eta_i + zeta;
      const double omega = eta_i * zeta / denom;
      const double T = omega * PQ2;
      const double pref = kTwoPiToFiveHalves / (eta_i * zeta * ::sqrt(denom));
      sum += pref * cKi * pair_cK[kj] * boys_f0<kFastBoys>(T);
    }
  }
  eri_out[t] = sum;
}

extern "C" cudaError_t cueri_eri_ssss_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys) {
  if (ntasks <= 0) return cudaSuccess;
  const int blocks = (ntasks + threads - 1) / threads;
  if (use_fast_boys) {
    KernelERI_ssss_flat<true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB, task_spCD, ntasks, sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  } else {
    KernelERI_ssss_flat<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB, task_spCD, ntasks, sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_ssss_warp_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys) {
  return cueri_eri_ssss_launch_stream(
      task_spAB, task_spCD, ntasks, sp_pair_start, sp_npair,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      eri_out, stream, threads, use_fast_boys);
}

extern "C" cudaError_t cueri_eri_ssss_multiblock_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* partial_sums,
    int blocks_per_task,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys) {
  (void)partial_sums;
  (void)blocks_per_task;
  return cueri_eri_ssss_launch_stream(
      task_spAB, task_spCD, ntasks, sp_pair_start, sp_npair,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      eri_out, stream, threads, use_fast_boys);
}

extern "C" cudaError_t cueri_fused_fock_ssss_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* /*shell_cx*/,
    const double* /*shell_cy*/,
    const double* /*shell_cz*/,
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
  if (ntasks <= 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int warps_per_block = threads >> 5;
  const int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  KernelFusedFock_ssss_warp<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks,
      sp_pair_start, sp_npair,
      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      sp_A, sp_B, shell_ao_start, nao,
      D_mat, F_mat, warps_per_block, n_bufs);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_fused_jk_ssss_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys,
    int n_bufs) {
  if (ntasks <= 0) return cudaSuccess;
  if (threads < 32 || (threads & 31) != 0) return cudaErrorInvalidValue;
  const int warps_per_block = threads >> 5;
  const int tasks_per_block = warps_per_block * 4;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;
  if (use_fast_boys) {
    KernelFusedJKSsssSubwarp8<true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB, task_spCD, ntasks,
        sp_pair_start, sp_npair,
        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
        sp_A, sp_B, shell_ao_start, nao,
        D_mat, J_mat, K_mat, warps_per_block, n_bufs);
  } else {
    KernelFusedJKSsssSubwarp8<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB, task_spCD, ntasks,
        sp_pair_start, sp_npair,
        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
        sp_A, sp_B, shell_ao_start, nao,
        D_mat, J_mat, K_mat, warps_per_block, n_bufs);
  }
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_count_entries_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    int32_t* counts,
    cudaStream_t stream,
    int threads) {
  const int blocks = (ntasks + threads - 1) / threads;
  KernelCountEntriesPerKey<<<blocks, threads, 0, stream>>>(task_spAB, task_spCD, ntasks, counts);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_fill_entry_csr_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* entry_offsets,
    int32_t* cursor,
    int32_t* entry_task,
    int32_t* entry_widx,
    cudaStream_t stream,
    int threads) {
  const int blocks = (ntasks + threads - 1) / threads;
  KernelFillEntryCSR<<<blocks, threads, 0, stream>>>(
      task_spAB, task_spCD, ntasks, entry_offsets, cursor, entry_task, entry_widx);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_reduce_from_entry_csr_launch_stream(
    const int32_t* entry_offsets,
    int nkey,
    const int32_t* entry_task,
    const int32_t* entry_widx,
    const double* eri_task,
    const double* W,
    double* Out,
    cudaStream_t stream,
    int threads) {
  KernelReduceFromEntryCSR<<<static_cast<unsigned int>(nkey), threads, 0, stream>>>(
      entry_offsets, entry_task, entry_widx, eri_task, W, Out);
  return cudaGetLastError();
}

namespace {

__global__ void KernelScatterDFMetricTiles(
    const double* tile,  // [ntasks, nP, nQ]
    const int32_t* p0,   // [ntasks]
    const int32_t* q0,   // [ntasks]
    int ntasks,
    int naux,
    int nP,
    int nQ,
    double* V) {  // [naux, naux]
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nP) * static_cast<int64_t>(nQ);
  for (; tid < n; tid += stride) {
    const int64_t t = tid / (static_cast<int64_t>(nP) * static_cast<int64_t>(nQ));
    const int64_t rem = tid - t * static_cast<int64_t>(nP) * static_cast<int64_t>(nQ);
    const int i = static_cast<int>(rem / static_cast<int64_t>(nQ));
    const int j = static_cast<int>(rem - static_cast<int64_t>(i) * static_cast<int64_t>(nQ));
    const int P = static_cast<int>(p0[t]) + i;
    const int Q = static_cast<int>(q0[t]) + j;
    const double v = tile[tid];
    V[static_cast<int64_t>(P) * static_cast<int64_t>(naux) + static_cast<int64_t>(Q)] = v;
    V[static_cast<int64_t>(Q) * static_cast<int64_t>(naux) + static_cast<int64_t>(P)] = v;
  }
}

__global__ void KernelScatterDFInt3c2eTiles(
    const double* tile,  // [ntasks, nAB, nP]
    const int32_t* a0,   // [ntasks]
    const int32_t* b0,   // [ntasks]
    const int32_t* p0,   // [ntasks] (relative to current aux block)
    int ntasks,
    int nao,
    int naux,
    int nAB,
    int nB,
    int nP,
    double* X) {  // [nao, nao, naux]
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nAB) * static_cast<int64_t>(nP);
  for (; tid < n; tid += stride) {
    const int64_t t = tid / (static_cast<int64_t>(nAB) * static_cast<int64_t>(nP));
    const int64_t rem = tid - t * static_cast<int64_t>(nAB) * static_cast<int64_t>(nP);
    const int ab = static_cast<int>(rem / static_cast<int64_t>(nP));
    const int P = static_cast<int>(rem - static_cast<int64_t>(ab) * static_cast<int64_t>(nP));

    const int i = ab / nB;
    const int j = ab - i * nB;

    const int a = static_cast<int>(a0[t]) + i;
    const int b = static_cast<int>(b0[t]) + j;
    const int p = static_cast<int>(p0[t]) + P;
    const double v = tile[tid];

    const int64_t idx_abp =
        (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(p);
    const int64_t idx_bap =
        (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(p);
    X[idx_abp] = v;
    X[idx_bap] = v;
  }
}

__global__ void KernelScatterAddDFYTTiles(
    const double* tile,  // [ntasks, nops, nP]
    const int32_t* p0,   // [ntasks] (absolute aux AO start index)
    int ntasks,
    int naux,
    int nops,
    int nP,
    double* YT) {  // [naux, nops]
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nops) * static_cast<int64_t>(nP);
  for (; tid < n; tid += stride) {
    const int64_t t = tid / (static_cast<int64_t>(nops) * static_cast<int64_t>(nP));
    const int64_t rem = tid - t * static_cast<int64_t>(nops) * static_cast<int64_t>(nP);
    const int pq = static_cast<int>(rem / static_cast<int64_t>(nP));
    const int P = static_cast<int>(rem - static_cast<int64_t>(pq) * static_cast<int64_t>(nP));

    const int row = static_cast<int>(p0[t]) + P;
    if (row < 0 || row >= naux) continue;
    const int64_t idx = static_cast<int64_t>(row) * static_cast<int64_t>(nops) + static_cast<int64_t>(pq);
    YT[idx] += tile[tid];
  }
}

__global__ void KernelDFSymmetrizeMnQInplace(
    double* arr_mnQ,  // shape (nao, nao, naux), C-order
    int nao,
    int naux) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid % static_cast<int64_t>(naux));
    const int64_t mn = tid / static_cast<int64_t>(naux);
    const int n = static_cast<int>(mn % static_cast<int64_t>(nao));
    const int m = static_cast<int>(mn / static_cast<int64_t>(nao));
    if (m >= n) continue;  // process each unordered AO pair exactly once

    const int64_t idx_mn = (static_cast<int64_t>(m) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                               static_cast<int64_t>(naux) +
                           static_cast<int64_t>(q);
    const int64_t idx_nm = (static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                               static_cast<int64_t>(naux) +
                           static_cast<int64_t>(q);
    const double avg = 0.5 * (arr_mnQ[idx_mn] + arr_mnQ[idx_nm]);
    arr_mnQ[idx_mn] = avg;
    arr_mnQ[idx_nm] = avg;
  }
}

__global__ void KernelDFSymmetrizeMnQToF32(
    const double* in_mnQ,  // shape (nao, nao, naux), C-order
    float* out_mnQ,        // shape (nao, nao, naux), C-order
    int nao,
    int naux) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid % static_cast<int64_t>(naux));
    const int64_t mn = tid / static_cast<int64_t>(naux);
    const int n = static_cast<int>(mn % static_cast<int64_t>(nao));
    const int m = static_cast<int>(mn / static_cast<int64_t>(nao));
    if (m > n) continue;  // process each unordered AO pair exactly once

    const int64_t idx_mn = (static_cast<int64_t>(m) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                               static_cast<int64_t>(naux) +
                           static_cast<int64_t>(q);
    if (m == n) {
      out_mnQ[idx_mn] = static_cast<float>(in_mnQ[idx_mn]);
      continue;
    }

    const int64_t idx_nm = (static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                               static_cast<int64_t>(naux) +
                           static_cast<int64_t>(q);
    const double avg = 0.5 * (in_mnQ[idx_mn] + in_mnQ[idx_nm]);
    const float favg = static_cast<float>(avg);
    out_mnQ[idx_mn] = favg;
    out_mnQ[idx_nm] = favg;
  }
}

// Fused Qmn -> mnQ transpose + symmetrize.
__global__ void KernelDFSymmetrizeQmnToMnQ(
    const double* in_Qmn,  // shape (naux, nao, nao), C-order
    double* out_mnQ,       // shape (nao, nao, naux), C-order
    int naux,
    int nao) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid % static_cast<int64_t>(naux));
    const int64_t mn = tid / static_cast<int64_t>(naux);
    const int n = static_cast<int>(mn % static_cast<int64_t>(nao));
    const int m = static_cast<int>(mn / static_cast<int64_t>(nao));
    if (m > n) continue;  // process each unordered AO pair exactly once

    const int64_t idx_in_mn = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                                  static_cast<int64_t>(nao) +
                              static_cast<int64_t>(n);
    const int64_t idx_in_nm = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                                  static_cast<int64_t>(nao) +
                              static_cast<int64_t>(m);
    const double avg = 0.5 * (in_Qmn[idx_in_mn] + in_Qmn[idx_in_nm]);

    const int64_t idx_out_mn = tid;  // ((m*nao + n)*naux + q)
    const int64_t idx_out_nm = (static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                                   static_cast<int64_t>(naux) +
                               static_cast<int64_t>(q);
    out_mnQ[idx_out_mn] = avg;
    out_mnQ[idx_out_nm] = avg;
  }
}

// Fused Qmn -> mnQ transpose + symmetrize + cast to float32.
__global__ void KernelDFSymmetrizeQmnToMnQToF32(
    const double* in_Qmn,  // shape (naux, nao, nao), C-order
    float* out_mnQ,        // shape (nao, nao, naux), C-order
    int naux,
    int nao) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid % static_cast<int64_t>(naux));
    const int64_t mn = tid / static_cast<int64_t>(naux);
    const int n = static_cast<int>(mn % static_cast<int64_t>(nao));
    const int m = static_cast<int>(mn / static_cast<int64_t>(nao));
    if (m > n) continue;  // process each unordered AO pair exactly once

    const int64_t idx_in_mn = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                                  static_cast<int64_t>(nao) +
                              static_cast<int64_t>(n);
    const int64_t idx_out_mn = tid;  // ((m*nao + n)*naux + q)
    if (m == n) {
      out_mnQ[idx_out_mn] = static_cast<float>(in_Qmn[idx_in_mn]);
      continue;
    }

    const int64_t idx_in_nm = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                                  static_cast<int64_t>(nao) +
                              static_cast<int64_t>(m);
    const double avg = 0.5 * (in_Qmn[idx_in_mn] + in_Qmn[idx_in_nm]);
    const float favg = static_cast<float>(avg);
    const int64_t idx_out_nm = (static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                                   static_cast<int64_t>(naux) +
                               static_cast<int64_t>(q);
    out_mnQ[idx_out_mn] = favg;
    out_mnQ[idx_out_nm] = favg;
  }
}

__global__ void KernelDFSymmetrizeQmnInplace(
    double* arr_Qmn,  // shape (naux, nao, nao), C-order
    int naux,
    int nao) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  for (; tid < total; tid += stride) {
    const int n = static_cast<int>(tid % static_cast<int64_t>(nao));
    const int64_t qm = tid / static_cast<int64_t>(nao);
    const int m = static_cast<int>(qm % static_cast<int64_t>(nao));
    if (m >= n) continue;  // process each unordered AO pair exactly once
    const int q = static_cast<int>(qm / static_cast<int64_t>(nao));

    const int64_t idx_mn = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                               static_cast<int64_t>(nao) +
                           static_cast<int64_t>(n);
    const int64_t idx_nm = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                               static_cast<int64_t>(nao) +
                           static_cast<int64_t>(m);
    const double avg = 0.5 * (arr_Qmn[idx_mn] + arr_Qmn[idx_nm]);
    arr_Qmn[idx_mn] = avg;
    arr_Qmn[idx_nm] = avg;
  }
}

__device__ __forceinline__ void TriUnpackS2Index(int64_t p, int& m, int& n) {
  // Invert p = m*(m+1)/2 + n with m>=n>=0.
  // Use sqrt approximation with robust fix-up to avoid rare off-by-one from FP rounding.
  if (p <= 0) {
    m = 0;
    n = static_cast<int>(p);
    return;
  }
  const double x = sqrt(8.0 * static_cast<double>(p) + 1.0);
  int64_t mm = static_cast<int64_t>((x - 1.0) * 0.5);
  int64_t t = mm * (mm + 1) / 2;
  while (t > p) {
    --mm;
    t = mm * (mm + 1) / 2;
  }
  while (true) {
    const int64_t mm1 = mm + 1;
    const int64_t tnext = mm1 * (mm1 + 1) / 2;
    if (tnext <= p) {
      mm = mm1;
      t = tnext;
    } else {
      break;
    }
  }
  m = static_cast<int>(mm);
  n = static_cast<int>(p - t);
}

__global__ void KernelDFPackMnQToQp(
    const double* in_mnQ,  // (nao, nao, naux), C-order ((m*nao+n)*naux+q)
    double* out_Qp,        // (naux, ntri), C-order (q*ntri+p)
    int nao,
    int naux) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(naux) * ntri;
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid / ntri);
    const int64_t p = tid - static_cast<int64_t>(q) * ntri;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const int64_t idx_in = (static_cast<int64_t>(m) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) *
                               static_cast<int64_t>(naux) +
                           static_cast<int64_t>(q);
    out_Qp[tid] = in_mnQ[idx_in];
  }
}

__global__ void KernelDFPackQmnToQp(
    const double* in_Qmn,  // (naux, nao, nao), C-order ((q*nao+m)*nao+n)
    double* out_Qp,        // (naux, ntri), C-order (q*ntri+p)
    int naux,
    int nao) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(naux) * ntri;
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid / ntri);
    const int64_t p = tid - static_cast<int64_t>(q) * ntri;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const int64_t idx_in = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                               static_cast<int64_t>(nao) +
                           static_cast<int64_t>(n);
    out_Qp[tid] = in_Qmn[idx_in];
  }
}

__global__ void KernelDFPackQmnBlockToQp(
    const double* in_Qmn_block,  // (q_count, nao, nao), C-order ((q*nao+m)*nao+n)
    double* out_Qp_block,        // (q_count, ntri), C-order (q*ntri+p)
    int q_count,
    int nao) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(q_count) * ntri;
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid / ntri);
    const int64_t p = tid - static_cast<int64_t>(q) * ntri;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const int64_t idx_in = (static_cast<int64_t>(q) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) *
                               static_cast<int64_t>(nao) +
                           static_cast<int64_t>(n);
    out_Qp_block[tid] = in_Qmn_block[idx_in];
  }
}

__global__ void KernelDFPackLfBlockToQp(
    const double* in_Lf_block,  // (nao, q_count*nao), C-order (m*(q_count*nao) + (q*nao+n))
    double* out_Qp,             // (naux, ntri), C-order (Q*ntri + p)
    int nao,
    int naux,
    int q0,
    int q_count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(q_count) * ntri;
  const int64_t ld = static_cast<int64_t>(q_count) * static_cast<int64_t>(nao);
  for (; tid < total; tid += stride) {
    const int q_local = static_cast<int>(tid / ntri);
    const int q = static_cast<int>(q0 + q_local);
    if (q < 0 || q >= naux) continue;
    const int64_t p = tid - static_cast<int64_t>(q_local) * ntri;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const int64_t idx_in = static_cast<int64_t>(m) * ld + static_cast<int64_t>(q_local) * static_cast<int64_t>(nao) +
                           static_cast<int64_t>(n);
    const int64_t idx_out = static_cast<int64_t>(q) * ntri + p;
    out_Qp[idx_out] = in_Lf_block[idx_in];
  }
}

__global__ void KernelDFUnpackQpToQmnBlock(
    const double* in_Qp,   // (naux, ntri), C-order (q*ntri+p)
    double* out_Qmn_block, // (q_count, nao, nao), C-order ((q*nao+m)*nao+n)
    int nao,
    int naux,
    int q0,
    int q_count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(q_count) * ntri;
  for (; tid < total; tid += stride) {
    const int q_local = static_cast<int>(tid / ntri);
    const int64_t p = tid - static_cast<int64_t>(q_local) * ntri;
    const int q = int(q0 + q_local);
    if (q < 0 || q >= naux) continue;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const double v = in_Qp[static_cast<int64_t>(q) * ntri + p];
    const int64_t base = static_cast<int64_t>(q_local) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
    out_Qmn_block[base + static_cast<int64_t>(m) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)] = v;
    out_Qmn_block[base + static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)] = v;
  }
}

__global__ void KernelDFUnpackQpToMnQ(
    const double* in_Qp,  // (naux, ntri), C-order (q*ntri+p)
    double* out_mnQ,      // (nao, nao, naux), C-order ((m*nao+n)*naux+q)
    int nao,
    int naux) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t total = static_cast<int64_t>(naux) * ntri;
  for (; tid < total; tid += stride) {
    const int q = static_cast<int>(tid / ntri);
    const int64_t p = tid - static_cast<int64_t>(q) * ntri;
    int m, n;
    TriUnpackS2Index(p, m, n);
    const double v = in_Qp[tid];
    const int64_t idx_mn =
        (static_cast<int64_t>(m) * static_cast<int64_t>(nao) + static_cast<int64_t>(n)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const int64_t idx_nm =
        (static_cast<int64_t>(n) * static_cast<int64_t>(nao) + static_cast<int64_t>(m)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    out_mnQ[idx_mn] = v;
    out_mnQ[idx_nm] = v;
  }
}

// ---------------------------------------------------------------------------
// KernelFusedQpLact
// Computes L_act[q,u,v] = sum_{mu,nu} C_act[mu,u] * B[q,mu,nu] * C_act[nu,v]
// directly from packed Qp storage with no (q,nao,nao) or (q,nao,ncas) intermediate.
//
// Grid:  (q_count, ceil(ncas/blockDim.x), ceil(ncas/blockDim.y))
// Block: (tile, tile)  — tile=16 is the default; one thread per (q, u, v) output.
// ---------------------------------------------------------------------------
__global__ void KernelFusedQpLact(
    const double* __restrict__ B_Qp,   // (naux, ntri) packed lower triangle
    const double* __restrict__ C_act,  // (nao, ncas) row-major
    double* __restrict__ L_act,        // (q_count, ncas, ncas) output
    int naux,
    int nao,
    int ncas,
    int ntri,
    int q0,
    int q_count) {
  const int q_local = blockIdx.x;
  const int u = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int v = static_cast<int>(blockIdx.z) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
  if (q_local >= q_count) return;
  const int q = q0 + q_local;
  if (q < 0 || q >= naux) return;
  if (u >= ncas || v >= ncas) return;

  const double* B_q = B_Qp + static_cast<int64_t>(q) * static_cast<int64_t>(ntri);
  double acc = 0.0;

  // L[q,u,v] = sum_{mu,nu} B[q,mu,nu] * C[mu,u] * C[nu,v]
  // Iterate lower triangle: p = mu*(mu+1)/2 + nu with mu >= nu.
  // Diagonal (mu==nu): contributes once.
  // Off-diagonal (mu>nu): B is symmetric → contributes (mu,nu) AND (nu,mu) directions.
  for (int mu = 0; mu < nao; ++mu) {
    const double c_mu_u = C_act[static_cast<int64_t>(mu) * ncas + u];
    const double c_mu_v = C_act[static_cast<int64_t>(mu) * ncas + v];
    const int p_base = mu * (mu + 1) / 2;
    // diagonal element
    acc += B_q[p_base + mu] * c_mu_u * c_mu_v;
    // off-diagonal elements
    for (int nu = 0; nu < mu; ++nu) {
      const double bval = B_q[p_base + nu];
      acc += bval * (c_mu_u * C_act[static_cast<int64_t>(nu) * ncas + v] +
                     C_act[static_cast<int64_t>(nu) * ncas + u] * c_mu_v);
    }
  }
  L_act[static_cast<int64_t>(q_local) * static_cast<int64_t>(ncas) * static_cast<int64_t>(ncas) +
        static_cast<int64_t>(u) * ncas + v] = acc;
}

// ---------------------------------------------------------------------------
// KernelFusedQpExchangeSym
// Computes out_Qp[q,p] += alpha * (F[mu,nu] + F[nu,mu])
// where F = D1 @ B_q @ D2 and B_q is read from packed Qp storage.
// The result is symmetric, so written back in packed Qp format.
//
// Grid:  ceil(q_count * ntri / threads) 1-D blocks
// Block: threads (typically 256)
// ---------------------------------------------------------------------------
__global__ void KernelFusedQpExchangeSym(
    const double* __restrict__ B_Qp,  // (naux, ntri) input
    const double* __restrict__ D1,    // (nao, nao) density matrix
    const double* __restrict__ D2,    // (nao, nao) density matrix
    double* __restrict__ out_Qp,      // (naux, ntri) accumulated output
    int naux,
    int nao,
    int ntri,
    int q0,
    int q_count,
    double alpha) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(q_count) * static_cast<int64_t>(ntri);

  for (int64_t idx = tid; idx < total; idx += stride) {
    const int q_local = static_cast<int>(idx / static_cast<int64_t>(ntri));
    const int64_t p = idx - static_cast<int64_t>(q_local) * static_cast<int64_t>(ntri);
    const int q = q0 + q_local;
    if (q < 0 || q >= naux) continue;

    int mu, nu;
    TriUnpackS2Index(p, mu, nu);  // mu >= nu

    const double* B_q = B_Qp + static_cast<int64_t>(q) * static_cast<int64_t>(ntri);

    // F[mu,nu] + F[nu,mu] where F = D1 @ B_q @ D2.
    // Exploit B_q symmetry: iterate lower triangle of B_q.
    // Diagonal (k==l): D1[mu,k]*B[k,k]*D2[k,nu] + D1[nu,k]*B[k,k]*D2[k,mu]
    // Off-diag (k>l):  B[k,l] contributes to pairs (k,l) and (l,k) in double sum.
    double val = 0.0;
    for (int k = 0; k < nao; ++k) {
      const double D1_mu_k = D1[static_cast<int64_t>(mu) * nao + k];
      const double D1_nu_k = D1[static_cast<int64_t>(nu) * nao + k];
      const int k_base = k * (k + 1) / 2;
      // diagonal l == k
      {
        const double b_kk = B_q[k_base + k];
        val += b_kk * (D1_mu_k * D2[static_cast<int64_t>(k) * nao + nu] +
                       D1_nu_k * D2[static_cast<int64_t>(k) * nao + mu]);
      }
      // off-diagonal l < k
      for (int l = 0; l < k; ++l) {
        const double b_kl = B_q[k_base + l];
        const double D1_mu_l = D1[static_cast<int64_t>(l) * nao + mu];
        const double D1_nu_l = D1[static_cast<int64_t>(l) * nao + nu];
        val += b_kl * (D1_mu_k * D2[static_cast<int64_t>(l) * nao + nu] +
                       D1_nu_k * D2[static_cast<int64_t>(l) * nao + mu] +
                       D1_mu_l * D2[static_cast<int64_t>(k) * nao + nu] +
                       D1_nu_l * D2[static_cast<int64_t>(k) * nao + mu]);
      }
    }
    out_Qp[static_cast<int64_t>(q) * static_cast<int64_t>(ntri) + p] += alpha * val;
  }
}

}  // namespace

extern "C" cudaError_t cueri_scatter_df_metric_tiles_launch_stream(
    const double* tile,
    const int32_t* p0,
    const int32_t* q0,
    int ntasks,
    int naux,
    int nP,
    int nQ,
    double* V_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || naux < 0 || nP < 0 || nQ < 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nP) * static_cast<int64_t>(nQ);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  KernelScatterDFMetricTiles<<<blocks, threads, 0, stream>>>(tile, p0, q0, ntasks, naux, nP, nQ, V_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_scatter_df_int3c2e_tiles_launch_stream(
    const double* tile,
    const int32_t* a0,
    const int32_t* b0,
    const int32_t* p0,
    int ntasks,
    int nao,
    int naux,
    int nAB,
    int nB,
    int nP,
    double* X_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao < 0 || naux < 0 || nAB < 0 || nB < 0 || nP < 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nAB) * static_cast<int64_t>(nP);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  KernelScatterDFInt3c2eTiles<<<blocks, threads, 0, stream>>>(tile, a0, b0, p0, ntasks, nao, naux, nAB, nB, nP, X_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_scatter_add_df_yt_tiles_launch_stream(
    const double* tile,
    const int32_t* p0,
    int ntasks,
    int naux,
    int nops,
    int nP,
    double* YT_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || naux < 0 || nops < 0 || nP < 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nops) * static_cast<int64_t>(nP);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  KernelScatterAddDFYTTiles<<<blocks, threads, 0, stream>>>(tile, p0, ntasks, naux, nops, nP, YT_out);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_symmetrize_mnq_inplace_launch_stream(
    double* arr_mnQ,
    int nao,
    int naux,
    cudaStream_t stream,
    int threads) {
  if (nao < 0 || naux < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFSymmetrizeMnQInplace<<<blocks, threads, 0, stream>>>(arr_mnQ, nao, naux);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_symmetrize_mnq_to_f32_launch_stream(
    const double* in_mnQ,
    float* out_mnQ,
    int nao,
    int naux,
    cudaStream_t stream,
    int threads) {
  if (nao < 0 || naux < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFSymmetrizeMnQToF32<<<blocks, threads, 0, stream>>>(in_mnQ, out_mnQ, nao, naux);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_symmetrize_qmn_inplace_launch_stream(
    double* arr_Qmn,
    int naux,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFSymmetrizeQmnInplace<<<blocks, threads, 0, stream>>>(arr_Qmn, naux, nao);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_symmetrize_qmn_to_mnq_launch_stream(
    const double* in_Qmn,
    double* out_mnQ,
    int naux,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFSymmetrizeQmnToMnQ<<<blocks, threads, 0, stream>>>(in_Qmn, out_mnQ, naux, nao);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_symmetrize_qmn_to_mnq_to_f32_launch_stream(
    const double* in_Qmn,
    float* out_mnQ,
    int naux,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFSymmetrizeQmnToMnQToF32<<<blocks, threads, 0, stream>>>(in_Qmn, out_mnQ, naux, nao);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_pack_mnq_to_qp_launch_stream(
    const double* in_mnQ,
    double* out_Qp,
    int nao,
    int naux,
    cudaStream_t stream,
    int threads) {
  if (nao < 0 || naux < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(naux) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFPackMnQToQp<<<blocks, threads, 0, stream>>>(in_mnQ, out_Qp, nao, naux);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_pack_qmn_to_qp_launch_stream(
    const double* in_Qmn,
    double* out_Qp,
    int naux,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(naux) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFPackQmnToQp<<<blocks, threads, 0, stream>>>(in_Qmn, out_Qp, naux, nao);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_pack_qmn_block_to_qp_launch_stream(
    const double* in_Qmn_block,
    double* out_Qp_block,
    int q_count,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (q_count < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(q_count) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFPackQmnBlockToQp<<<blocks, threads, 0, stream>>>(in_Qmn_block, out_Qp_block, q_count, nao);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_pack_lf_block_to_qp_launch_stream(
    const double* in_Lf_block,
    double* out_Qp,
    int naux,
    int nao,
    int q0,
    int q_count,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || q0 < 0 || q_count < 0 || threads <= 0) return cudaErrorInvalidValue;
  if (q0 > naux) return cudaErrorInvalidValue;
  if (q_count > naux - q0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(q_count) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFPackLfBlockToQp<<<blocks, threads, 0, stream>>>(in_Lf_block, out_Qp, nao, naux, q0, q_count);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_unpack_qp_to_qmn_block_launch_stream(
    const double* in_Qp,
    double* out_Qmn_block,
    int naux,
    int nao,
    int q0,
    int q_count,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || q0 < 0 || q_count < 0 || threads <= 0) return cudaErrorInvalidValue;
  if (q0 > naux) return cudaErrorInvalidValue;
  if (q_count > naux - q0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(q_count) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFUnpackQpToQmnBlock<<<blocks, threads, 0, stream>>>(in_Qp, out_Qmn_block, nao, naux, q0, q_count);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_unpack_qp_to_mnq_launch_stream(
    const double* in_Qp,
    double* out_mnQ,
    int naux,
    int nao,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || threads <= 0) return cudaErrorInvalidValue;
  const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
  const int64_t n = static_cast<int64_t>(naux) * ntri;
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelDFUnpackQpToMnQ<<<blocks, threads, 0, stream>>>(in_Qp, out_mnQ, nao, naux);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_fused_qp_l_act_launch_stream(
    const double* B_Qp,
    const double* C_act,
    double* L_act,
    int naux,
    int nao,
    int ncas,
    int ntri,
    int q0,
    int q_count,
    cudaStream_t stream,
    int tile) {
  if (naux < 0 || nao < 0 || ncas < 0 || ntri < 0 || q0 < 0 || q_count < 0 || tile <= 0)
    return cudaErrorInvalidValue;
  if (ncas == 0 || q_count == 0) return cudaSuccess;
  const dim3 block(static_cast<unsigned>(tile), static_cast<unsigned>(tile), 1);
  const dim3 grid(
      static_cast<unsigned>(q_count),
      static_cast<unsigned>((ncas + tile - 1) / tile),
      static_cast<unsigned>((ncas + tile - 1) / tile));
  KernelFusedQpLact<<<grid, block, 0, stream>>>(
      B_Qp, C_act, L_act, naux, nao, ncas, ntri, q0, q_count);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_fused_qp_exchange_sym_launch_stream(
    const double* B_Qp,
    const double* D1,
    const double* D2,
    double* out_Qp,
    int naux,
    int nao,
    int ntri,
    int q0,
    int q_count,
    double alpha,
    cudaStream_t stream,
    int threads) {
  if (naux < 0 || nao < 0 || ntri < 0 || q0 < 0 || q_count < 0 || threads <= 0)
    return cudaErrorInvalidValue;
  if (q_count == 0) return cudaSuccess;
  const int64_t n = static_cast<int64_t>(q_count) * static_cast<int64_t>(ntri);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelFusedQpExchangeSym<<<blocks, threads, 0, stream>>>(
      B_Qp, D1, D2, out_Qp, naux, nao, ntri, q0, q_count, alpha);
  return cudaGetLastError();
}
