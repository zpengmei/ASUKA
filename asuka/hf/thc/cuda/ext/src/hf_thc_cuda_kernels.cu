#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Fixed block size keeps the kernels simple and predictable.
constexpr int32_t kBlock = 256;

__global__ void rowwise_dot_f64_kernel(const double* a, const double* x, int32_t nao, double* out_m) {
  const int32_t p = int32_t(blockIdx.x);
  const int32_t tid = int32_t(threadIdx.x);
  if (!a || !x || !out_m) return;
  if (nao <= 0) {
    if (tid == 0) out_m[p] = 0.0;
    return;
  }

  const double* a_row = a + int64_t(p) * int64_t(nao);
  const double* x_row = x + int64_t(p) * int64_t(nao);

  double sum = 0.0;
  for (int32_t mu = tid; mu < nao; mu += int32_t(blockDim.x)) {
    sum += a_row[mu] * x_row[mu];
  }

  __shared__ double sh[kBlock];
  sh[tid] = sum;
  __syncthreads();

  // Standard power-of-two reduction (kBlock is 256).
  for (int32_t stride = int32_t(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sh[tid] += sh[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out_m[p] = sh[0];
}

__global__ void scale_rows_f64_kernel(const double* x, const double* n, int64_t nelem, int32_t nao, double* out) {
  const int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (idx >= nelem) return;
  const int32_t p = int32_t(idx / int64_t(nao));
  out[idx] = x[idx] * n[p];
}

__global__ void hadamard_inplace_f64_strided_kernel(
    double* m, int64_t ld_m, const double* z, int64_t ld_z, int32_t nb, int64_t nelem) {
  const int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (idx >= nelem) return;
  const int32_t k = int32_t(idx % int64_t(nb));
  const int32_t p = int32_t(idx / int64_t(nb));
  m[int64_t(p) * ld_m + int64_t(k)] *= z[int64_t(p) * ld_z + int64_t(k)];
}

}  // namespace

extern "C" {

void hf_thc_rowwise_dot_f64(
    const double* a, const double* x, int32_t npt, int32_t nao, double* out_m, cudaStream_t stream) {
  if (!a || !x || !out_m) return;
  if (npt <= 0) return;
  const dim3 block(kBlock, 1, 1);
  const dim3 grid(uint32_t(npt), 1, 1);
  rowwise_dot_f64_kernel<<<grid, block, 0, stream>>>(a, x, nao, out_m);
}

void hf_thc_scale_rows_f64(
    const double* x, const double* n, int32_t npt, int32_t nao, double* out, cudaStream_t stream) {
  if (!x || !n || !out) return;
  if (npt <= 0 || nao <= 0) return;
  const int64_t nelem = int64_t(npt) * int64_t(nao);
  const int32_t block = kBlock;
  const int32_t grid = int32_t((nelem + int64_t(block) - 1) / int64_t(block));
  scale_rows_f64_kernel<<<dim3(uint32_t(grid), 1, 1), dim3(uint32_t(block), 1, 1), 0, stream>>>(
      x, n, nelem, nao, out);
}

void hf_thc_hadamard_inplace_f64(
    double* m,
    int64_t ld_m,
    const double* z,
    int64_t ld_z,
    int32_t npt,
    int32_t nb,
    cudaStream_t stream) {
  if (!m || !z) return;
  if (npt <= 0 || nb <= 0) return;
  const int64_t nelem = int64_t(npt) * int64_t(nb);
  const int32_t block = kBlock;
  const int32_t grid = int32_t((nelem + int64_t(block) - 1) / int64_t(block));
  hadamard_inplace_f64_strided_kernel<<<dim3(uint32_t(grid), 1, 1), dim3(uint32_t(block), 1, 1), 0, stream>>>(
      m, ld_m, z, ld_z, nb, nelem);
}

}  // extern "C"
