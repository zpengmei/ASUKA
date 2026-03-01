#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

__global__ void fill_lower_from_upper_f64_kernel(double* __restrict__ a, int n) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= n || j >= n) return;
  if (i <= j) return;
  a[static_cast<int64_t>(i) * static_cast<int64_t>(n) + static_cast<int64_t>(j)] =
      a[static_cast<int64_t>(j) * static_cast<int64_t>(n) + static_cast<int64_t>(i)];
}

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

__global__ void pack_bmnq_to_bq_f64_kernel(const double* __restrict__ b_mnq,
                                          int64_t p,
                                          int naux,
                                          int q0,
                                          int q,
                                          double* __restrict__ bq) {
  __shared__ double tile[kTileDim][kTileDim + 1];

  const int x = static_cast<int>(blockIdx.x * kTileDim + threadIdx.x);  // q
  const int y = static_cast<int>(blockIdx.y * kTileDim + threadIdx.y);  // p

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = y + j;
    if (x < q && yy < p) {
      tile[threadIdx.y + j][threadIdx.x] = b_mnq[static_cast<int64_t>(yy) * static_cast<int64_t>(naux) +
                                                 static_cast<int64_t>(q0 + x)];
    }
  }

  __syncthreads();

  const int out_x = static_cast<int>(blockIdx.y * kTileDim + threadIdx.x);  // p
  const int out_y = static_cast<int>(blockIdx.x * kTileDim + threadIdx.y);  // q

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    const int yy = out_y + j;
    if (out_x < p && yy < q) {
      bq[static_cast<int64_t>(yy) * p + static_cast<int64_t>(out_x)] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

}  // namespace

extern "C" void hf_df_jk_fill_lower_from_upper_f64(double* a, int n, cudaStream_t stream) {
  if (!a || n <= 0) return;
  const dim3 block(16, 16, 1);
  const dim3 grid((static_cast<uint32_t>(n) + block.x - 1) / block.x,
                  (static_cast<uint32_t>(n) + block.y - 1) / block.y,
                  1);
  fill_lower_from_upper_f64_kernel<<<grid, block, 0, stream>>>(a, n);
  throw_on_cuda_error(cudaGetLastError(), "fill_lower_from_upper_f64_kernel launch");
}

extern "C" void hf_df_jk_pack_bmnq_to_bq_f64(
    const double* b_mnq, int nao, int naux, int q0, int q, double* out_bq, cudaStream_t stream) {
  if (!b_mnq || !out_bq || nao <= 0 || naux <= 0 || q <= 0) return;
  const int64_t p = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  const dim3 block(kTileDim, kBlockRows, 1);
  const dim3 grid((static_cast<uint32_t>(q) + kTileDim - 1) / kTileDim,
                  (static_cast<uint32_t>(p) + kTileDim - 1) / kTileDim,
                  1);
  pack_bmnq_to_bq_f64_kernel<<<grid, block, 0, stream>>>(b_mnq, p, naux, q0, q, out_bq);
  throw_on_cuda_error(cudaGetLastError(), "pack_bmnq_to_bq_f64_kernel launch");
}
