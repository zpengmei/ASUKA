#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cuda_kernels_api.h"

namespace {

__global__ void KernelScatterERITilesOrdered(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* eri_mat,
    int64_t total,
    int64_t nPairAO) {
  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const int64_t tile_stride = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride;
    const int t = static_cast<int>(t64);
    const int64_t rem = idx - t64 * tile_stride;
    const int iab = static_cast<int>(rem / nCD);
    const int icd = static_cast<int>(rem - static_cast<int64_t>(iab) * nCD);

    const int ia = iab / nB;
    const int ib = iab - ia * nB;
    const int ic = icd / nD;
    const int id = icd - ic * nD;

    const int spab = static_cast<int>(task_spAB[t]);
    const int spcd = static_cast<int>(task_spCD[t]);
    const int A = static_cast<int>(sp_A[spab]);
    const int B = static_cast<int>(sp_B[spab]);
    const int C = static_cast<int>(sp_A[spcd]);
    const int D = static_cast<int>(sp_B[spcd]);

    const int a = static_cast<int>(shell_ao_start[A]) + ia;
    const int b = static_cast<int>(shell_ao_start[B]) + ib;
    const int c = static_cast<int>(shell_ao_start[C]) + ic;
    const int d = static_cast<int>(shell_ao_start[D]) + id;

    const int64_t ab_p = static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b);
    const int64_t ab_s = static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a);
    const int64_t cd_p = static_cast<int64_t>(c) * static_cast<int64_t>(nao) + static_cast<int64_t>(d);
    const int64_t cd_s = static_cast<int64_t>(d) * static_cast<int64_t>(nao) + static_cast<int64_t>(c);

    const double v = tile_vals[idx];
    if (v == 0.0) continue;

    atomicAdd(&eri_mat[ab_p * nPairAO + cd_p], v);
    if (C != D) atomicAdd(&eri_mat[ab_p * nPairAO + cd_s], v);
    if (A != B) atomicAdd(&eri_mat[ab_s * nPairAO + cd_p], v);
    if (A != B && C != D) atomicAdd(&eri_mat[ab_s * nPairAO + cd_s], v);

    if (spab != spcd) {
      atomicAdd(&eri_mat[cd_p * nPairAO + ab_p], v);
      if (A != B) atomicAdd(&eri_mat[cd_p * nPairAO + ab_s], v);
      if (C != D) atomicAdd(&eri_mat[cd_s * nPairAO + ab_p], v);
      if (A != B && C != D) atomicAdd(&eri_mat[cd_s * nPairAO + ab_s], v);
    }
  }
}

}  // namespace

extern "C" cudaError_t cueri_scatter_eri_tiles_ordered_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* eri_mat,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (ntasks == 0) return cudaSuccess;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  const int64_t nPairAO = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  if (total <= 0 || nPairAO <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelScatterERITilesOrdered<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start,
      nao,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      eri_mat,
      total,
      nPairAO);
  return cudaPeekAtLastError();
}
