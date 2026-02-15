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
  if (use_fast_boys) {
    KernelERI_ssss<true><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(
        task_spAB, task_spCD, sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
  } else {
    KernelERI_ssss<false><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(
        task_spAB, task_spCD, sp_pair_start, sp_npair, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);
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
  const int warps_per_block = threads >> 5;
  const int tasks_per_block = warps_per_block * 4;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;
  if (use_fast_boys) {
    KernelERI_ssss_subwarp8<true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_pair_start,
        sp_npair,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        eri_out);
  } else {
    KernelERI_ssss_subwarp8<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_pair_start,
        sp_npair,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        eri_out);
  }
  return cudaGetLastError();
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
  if (blocks_per_task <= 0) return cudaErrorInvalidValue;
  dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(blocks_per_task), 1);
  if (use_fast_boys) {
    KernelERI_ssss_multiblock_partial<true><<<grid, threads, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_pair_start,
        sp_npair,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        blocks_per_task,
        partial_sums);
  } else {
    KernelERI_ssss_multiblock_partial<false><<<grid, threads, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_pair_start,
        sp_npair,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        blocks_per_task,
        partial_sums);
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  KernelERI_ssss_multiblock_reduce<<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(
      partial_sums, blocks_per_task, eri_out);
  return cudaGetLastError();
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
