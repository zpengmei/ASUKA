#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cart2sph_tables.cuh"
#include "cueri_cuda_kernels_api.h"

namespace {

__device__ __forceinline__ int ncart(int l) { return ((l + 1) * (l + 2)) >> 1; }
__device__ __forceinline__ int nsph(int l) { return (l << 1) + 1; }

__global__ void KernelCart2SphEriRight(
    const double* __restrict__ tile_cart,  // (nt, nAB_cart, nCD_cart)
    double* __restrict__ tile_tmp,         // (nt, nAB_cart, nCD_sph)
    int ntasks,
    int la,
    int lb,
    int lc,
    int ld) {
  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nC_cart = ncart(lc);
  const int nD_cart = ncart(ld);
  const int nAB_cart = nA_cart * nB_cart;
  const int nCD_cart = nC_cart * nD_cart;

  const int nC_sph = nsph(lc);
  const int nD_sph = nsph(ld);
  const int nCD_sph = nC_sph * nD_sph;

  const int64_t tile_stride_in = static_cast<int64_t>(nAB_cart) * static_cast<int64_t>(nCD_cart);
  const int64_t tile_stride_out = static_cast<int64_t>(nAB_cart) * static_cast<int64_t>(nCD_sph);

  const int64_t total = static_cast<int64_t>(ntasks) * tile_stride_out;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                     static_cast<int64_t>(threadIdx.x);
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride_out;
    const int64_t rem = idx - t64 * tile_stride_out;
    const int iab = static_cast<int>(rem / static_cast<int64_t>(nCD_sph));
    const int scd = static_cast<int>(rem - static_cast<int64_t>(iab) * static_cast<int64_t>(nCD_sph));

    const int isC = scd / nD_sph;
    const int isD = scd - isC * nD_sph;

    const double* __restrict__ in_t = tile_cart + t64 * tile_stride_in +
                                      static_cast<int64_t>(iab) * static_cast<int64_t>(nCD_cart);

    double acc = 0.0;
#pragma unroll
    for (int ic = 0; ic < 21; ++ic) {
      if (ic >= nC_cart) break;
      const double tc = cart2sph_coeff(lc, ic, isC);
      if (tc == 0.0) continue;
#pragma unroll
      for (int id = 0; id < 21; ++id) {
        if (id >= nD_cart) break;
        const double td = cart2sph_coeff(ld, id, isD);
        if (td == 0.0) continue;
        const int icd = ic * nD_cart + id;
        acc += (tc * td) * in_t[icd];
      }
    }

    tile_tmp[t64 * tile_stride_out + static_cast<int64_t>(iab) * static_cast<int64_t>(nCD_sph) + scd] = acc;
  }
}

__global__ void KernelCart2SphEriLeft(
    const double* __restrict__ tile_tmp,  // (nt, nAB_cart, nCD_sph)
    double* __restrict__ tile_sph,        // (nt, nAB_sph, nCD_sph)
    int ntasks,
    int la,
    int lb,
    int lc,
    int ld) {
  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nAB_cart = nA_cart * nB_cart;

  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);
  const int nAB_sph = nA_sph * nB_sph;

  const int nC_sph = nsph(lc);
  const int nD_sph = nsph(ld);
  const int nCD_sph = nC_sph * nD_sph;

  const int64_t tile_stride_in = static_cast<int64_t>(nAB_cart) * static_cast<int64_t>(nCD_sph);
  const int64_t tile_stride_out = static_cast<int64_t>(nAB_sph) * static_cast<int64_t>(nCD_sph);

  const int64_t total = static_cast<int64_t>(ntasks) * tile_stride_out;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                     static_cast<int64_t>(threadIdx.x);
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride_out;
    const int64_t rem = idx - t64 * tile_stride_out;

    const int sab = static_cast<int>(rem / static_cast<int64_t>(nCD_sph));
    const int scd = static_cast<int>(rem - static_cast<int64_t>(sab) * static_cast<int64_t>(nCD_sph));

    const int isA = sab / nB_sph;
    const int isB = sab - isA * nB_sph;

    const double* __restrict__ in_t = tile_tmp + t64 * tile_stride_in;

    double acc = 0.0;
#pragma unroll
    for (int ia = 0; ia < 21; ++ia) {
      if (ia >= nA_cart) break;
      const double ta = cart2sph_coeff(la, ia, isA);
      if (ta == 0.0) continue;
#pragma unroll
      for (int ib = 0; ib < 21; ++ib) {
        if (ib >= nB_cart) break;
        const double tb = cart2sph_coeff(lb, ib, isB);
        if (tb == 0.0) continue;
        const int iab = ia * nB_cart + ib;
        acc += (ta * tb) * in_t[static_cast<int64_t>(iab) * static_cast<int64_t>(nCD_sph) + scd];
      }
    }

    tile_sph[t64 * tile_stride_out + static_cast<int64_t>(sab) * static_cast<int64_t>(nCD_sph) + scd] = acc;
  }
}

__device__ __forceinline__ int64_t pair_index(int i, int j) {
  int hi = i;
  int lo = j;
  if (lo > hi) {
    hi = j;
    lo = i;
  }
  return (static_cast<int64_t>(hi) * static_cast<int64_t>(hi + 1)) / 2 + static_cast<int64_t>(lo);
}

__device__ __forceinline__ int64_t pairpair_index(int64_t p, int64_t q) {
  int64_t hi = p;
  int64_t lo = q;
  if (lo > hi) {
    hi = q;
    lo = p;
  }
  return (hi * (hi + 1)) / 2 + lo;
}

__global__ void KernelScatterEriTilesSphS8(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* __restrict__ tile_vals,  // (nt, nAB, nCD)
    double* __restrict__ out_s8,
    int64_t total) {
  (void)nao_sph;
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

    if (A == B && ia < ib) continue;
    if (C == D && ic < id) continue;

    const int a = static_cast<int>(shell_ao_start_sph[A]) + ia;
    const int b = static_cast<int>(shell_ao_start_sph[B]) + ib;
    const int c = static_cast<int>(shell_ao_start_sph[C]) + ic;
    const int d = static_cast<int>(shell_ao_start_sph[D]) + id;

    const int64_t p = pair_index(a, b);
    const int64_t q = pair_index(c, d);

    if (spab == spcd && p < q) continue;

    const int64_t out_idx = pairpair_index(p, q);
    const double v = tile_vals[idx];
    if (v == 0.0) continue;
    out_s8[out_idx] = v;
  }
}

__global__ void KernelScatterEriTilesSphS4(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* __restrict__ tile_vals,  // (nt, nAB, nCD)
    double* __restrict__ out_s4,
    int64_t total,
    int64_t nao_pair) {
  (void)nao_sph;
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

    if (A == B && ia < ib) continue;
    if (C == D && ic < id) continue;

    const int a = static_cast<int>(shell_ao_start_sph[A]) + ia;
    const int b = static_cast<int>(shell_ao_start_sph[B]) + ib;
    const int c = static_cast<int>(shell_ao_start_sph[C]) + ic;
    const int d = static_cast<int>(shell_ao_start_sph[D]) + id;

    const int64_t p = pair_index(a, b);
    const int64_t q = pair_index(c, d);

    if (spab == spcd && p < q) continue;

    const double v = tile_vals[idx];
    if (v == 0.0) continue;
    out_s4[p * nao_pair + q] = v;
    out_s4[q * nao_pair + p] = v;
  }
}

}  // namespace

extern "C" cudaError_t cueri_cart2sph_eri_right_launch_stream(
    const double* tile_cart,
    double* tile_tmp,
    int ntasks,
    int la,
    int lb,
    int lc,
    int ld,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (la < 0 || la > 5 || lb < 0 || lb > 5 || lc < 0 || lc > 5 || ld < 0 || ld > 5) {
    return cudaErrorInvalidValue;
  }

  const int nA_cart = ((la + 1) * (la + 2)) >> 1;
  const int nB_cart = ((lb + 1) * (lb + 2)) >> 1;
  const int nAB_cart = nA_cart * nB_cart;
  const int nCD_sph = (2 * lc + 1) * (2 * ld + 1);
  const int64_t total = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nAB_cart) * static_cast<int64_t>(nCD_sph);

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelCart2SphEriRight<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      tile_cart, tile_tmp, ntasks, la, lb, lc, ld);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_cart2sph_eri_left_launch_stream(
    const double* tile_tmp,
    double* tile_sph,
    int ntasks,
    int la,
    int lb,
    int lc,
    int ld,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;
  if (la < 0 || la > 5 || lb < 0 || lb > 5 || lc < 0 || lc > 5 || ld < 0 || ld > 5) {
    return cudaErrorInvalidValue;
  }

  const int nAB_sph = (2 * la + 1) * (2 * lb + 1);
  const int nCD_sph = (2 * lc + 1) * (2 * ld + 1);
  const int64_t total = static_cast<int64_t>(ntasks) * static_cast<int64_t>(nAB_sph) * static_cast<int64_t>(nCD_sph);

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelCart2SphEriLeft<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      tile_tmp, tile_sph, ntasks, la, lb, lc, ld);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_scatter_eri_tiles_sph_s8_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* out_s8,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao_sph < 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  if (total <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelScatterEriTilesSphS8<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start_sph,
      nao_sph,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      out_s8,
      total);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_scatter_eri_tiles_sph_s4_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* out_s4,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao_sph < 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ntasks == 0) return cudaSuccess;

  const int64_t nao_pair = (static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph + 1)) / 2;
  if (nao_pair <= 0) return cudaErrorInvalidValue;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  if (total <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelScatterEriTilesSphS4<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start_sph,
      nao_sph,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      out_s4,
      total,
      nao_pair);
  return cudaPeekAtLastError();
}
