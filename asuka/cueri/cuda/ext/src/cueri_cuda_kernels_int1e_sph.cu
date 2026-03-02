#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cuda_kernels_api.h"

namespace {

__device__ __forceinline__ double warp_reduce_sum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ double block_reduce_sum(double value) {
  __shared__ double shared[32];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp_id] = value;
  __syncthreads();
  const int nwarp = static_cast<int>(blockDim.x) >> 5;
  double out = 0.0;
  if (warp_id == 0) {
    out = (lane < nwarp) ? shared[lane] : 0.0;
    out = warp_reduce_sum(out);
  }
  return out;
}

__global__ void KernelInt1eDSDerivContractSph(
    const double* __restrict__ dS_sph_flat,
    const double* __restrict__ M_sph_flat,
    int nao_sph,
    int nax,
    double* __restrict__ grad_flat) {
  const int ax = static_cast<int>(blockIdx.y);
  if (ax >= nax) return;

  const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  const double* d = dS_sph_flat + static_cast<int64_t>(ax) * nao2;
  double local = 0.0;
  for (int64_t idx = tid; idx < nao2; idx += stride) {
    local += d[idx] * M_sph_flat[idx];
  }

  const double block_sum = block_reduce_sum(local);
  if (threadIdx.x == 0) {
    atomicAdd(&grad_flat[ax], block_sum);
  }
}

__global__ void KernelInt1eDHcoreDerivContractSph(
    const double* __restrict__ dT_sph_flat,
    const double* __restrict__ dV_sph_flat,
    const double* __restrict__ M_sph_flat,
    int nao_sph,
    int nax,
    double* __restrict__ grad_flat) {
  const int ax = static_cast<int>(blockIdx.y);
  if (ax >= nax) return;

  const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

  const double* dT = dT_sph_flat + static_cast<int64_t>(ax) * nao2;
  const double* dV = dV_sph_flat + static_cast<int64_t>(ax) * nao2;
  double local = 0.0;
  for (int64_t idx = tid; idx < nao2; idx += stride) {
    local += (dT[idx] + dV[idx]) * M_sph_flat[idx];
  }

  const double block_sum = block_reduce_sum(local);
  if (threadIdx.x == 0) {
    atomicAdd(&grad_flat[ax], block_sum);
  }
}

}  // namespace

extern "C" cudaError_t cueri_int1e_dS_deriv_contracted_sph_launch_stream(
    const double* dS_sph_flat,
    const double* M_sph_flat,
    int natm,
    int nao_sph,
    double* grad_flat,
    cudaStream_t stream,
    int threads) {
  if (dS_sph_flat == nullptr || M_sph_flat == nullptr || grad_flat == nullptr) return cudaErrorInvalidValue;
  if (natm <= 0 || nao_sph <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256 || (threads & 31) != 0) return cudaErrorInvalidValue;

  const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
  const int max_blocks_x = 256;
  int blocks_x = static_cast<int>((nao2 + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  if (blocks_x < 1) blocks_x = 1;
  if (blocks_x > max_blocks_x) blocks_x = max_blocks_x;
  const int nax = natm * 3;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(nax), 1u);

  KernelInt1eDSDerivContractSph<<<grid, static_cast<unsigned int>(threads), 0, stream>>>(
      dS_sph_flat,
      M_sph_flat,
      nao_sph,
      nax,
      grad_flat);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_int1e_dhcore_deriv_contracted_sph_launch_stream(
    const double* dT_sph_flat,
    const double* dV_sph_flat,
    const double* M_sph_flat,
    int natm,
    int nao_sph,
    double* grad_flat,
    cudaStream_t stream,
    int threads) {
  if (dT_sph_flat == nullptr || dV_sph_flat == nullptr || M_sph_flat == nullptr || grad_flat == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (natm <= 0 || nao_sph <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256 || (threads & 31) != 0) return cudaErrorInvalidValue;

  const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
  const int max_blocks_x = 256;
  int blocks_x = static_cast<int>((nao2 + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  if (blocks_x < 1) blocks_x = 1;
  if (blocks_x > max_blocks_x) blocks_x = max_blocks_x;
  const int nax = natm * 3;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(nax), 1u);

  KernelInt1eDHcoreDerivContractSph<<<grid, static_cast<unsigned int>(threads), 0, stream>>>(
      dT_sph_flat,
      dV_sph_flat,
      M_sph_flat,
      nao_sph,
      nax,
      grad_flat);
  return cudaGetLastError();
}
