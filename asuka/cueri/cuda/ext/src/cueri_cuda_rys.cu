#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cuda_kernels_api.h"
#include "cueri_cuda_rys_device.cuh"

namespace {

template <int NROOTS>
__global__ void KernelRysRootsWeights(const double* T, int nT, double* roots_out, double* weights_out) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (i >= nT) return;
  double r[NROOTS];
  double w[NROOTS];
  cueri_rys::rys_roots_weights<NROOTS>(T[i], r, w);
  const int base = i * NROOTS;
  #pragma unroll
  for (int k = 0; k < NROOTS; ++k) {
    roots_out[base + k] = r[k];
    weights_out[base + k] = w[k];
  }
}

}  // namespace

extern "C" cudaError_t cueri_rys_roots_weights_launch_stream(
    const double* T,
    int nT,
    int nroots,
    double* roots_out,
    double* weights_out,
    cudaStream_t stream,
    int threads) {
  if (nT < 0) return cudaErrorInvalidValue;
  if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

  const int blocks = (nT + threads - 1) / threads;
  switch (nroots) {
    case 1:
      KernelRysRootsWeights<1><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 2:
      KernelRysRootsWeights<2><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 3:
      KernelRysRootsWeights<3><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 4:
      KernelRysRootsWeights<4><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 5:
      KernelRysRootsWeights<5><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 6:
      KernelRysRootsWeights<6><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 7:
      KernelRysRootsWeights<7><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 8:
      KernelRysRootsWeights<8><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 9:
      KernelRysRootsWeights<9><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 10:
      KernelRysRootsWeights<10><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    case 11:
      KernelRysRootsWeights<11><<<blocks, threads, 0, stream>>>(T, nT, roots_out, weights_out);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
