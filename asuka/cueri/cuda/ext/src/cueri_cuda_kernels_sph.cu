#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cart2sph_tables.cuh"
#include "cueri_cuda_kernels_api.h"

namespace {

__global__ void KernelSphCoeffSphToCart(
    const double* __restrict__ C_sph,
    double* __restrict__ C_cart,
    int norb,
    const int32_t* __restrict__ ao2shell_cart,
    const int32_t* __restrict__ ao2local_cart,
    const int32_t* __restrict__ shell_ao_start_sph,
    const int32_t* __restrict__ shell_l,
    int nao_cart) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
  const int64_t n = static_cast<int64_t>(nao_cart) * static_cast<int64_t>(norb);
  if (idx >= n) return;

  const int a = static_cast<int>(idx / static_cast<int64_t>(norb));
  const int p = static_cast<int>(idx - static_cast<int64_t>(a) * static_cast<int64_t>(norb));

  const int sh = static_cast<int>(ao2shell_cart[a]);
  const int ic = static_cast<int>(ao2local_cart[a]);
  const int l = static_cast<int>(shell_l[sh]);
  if (l < 0 || l > 5) {
    C_cart[idx] = 0.0;
    return;
  }

  const int ns = 2 * l + 1;
  const int sph0 = static_cast<int>(shell_ao_start_sph[sh]);

  double acc = 0.0;
#pragma unroll
  for (int is = 0; is < 11; ++is) {
    if (is >= ns) break;
    const double coef = cart2sph_coeff(l, ic, is);
    const int64_t row = static_cast<int64_t>(sph0 + is) * static_cast<int64_t>(norb);
    acc += coef * C_sph[row + p];
  }
  C_cart[idx] = acc;
}

}  // namespace

extern "C" cudaError_t cueri_sph_coeff_sph_to_cart_launch_stream(
    const double* C_sph,
    double* C_cart,
    int nao_cart,
    int norb,
    const int32_t* ao2shell_cart,
    const int32_t* ao2local_cart,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_l,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || norb < 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(nao_cart) * static_cast<int64_t>(norb);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelSphCoeffSphToCart<<<blocks, threads, 0, stream>>>(
      C_sph,
      C_cart,
      norb,
      ao2shell_cart,
      ao2local_cart,
      shell_ao_start_sph,
      shell_l,
      nao_cart);
  return cudaGetLastError();
}
