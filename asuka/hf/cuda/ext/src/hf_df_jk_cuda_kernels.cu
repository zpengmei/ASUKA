#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Packed AO-pair ("s2") index:
//   p(m,n) = m*(m+1)/2 + n, with m>=n>=0.
__device__ __forceinline__ int64_t tri_index_s2(int a, int b) {
  // Use int64_t to avoid overflow when nao is large.
  return (static_cast<int64_t>(a) * static_cast<int64_t>(a + 1)) / 2 + static_cast<int64_t>(b);
}

__global__ void fill_lower_from_upper_f64_kernel(double* __restrict__ a, int n) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= n || j >= n) return;
  if (i <= j) return;
  a[static_cast<int64_t>(i) * static_cast<int64_t>(n) + static_cast<int64_t>(j)] =
      a[static_cast<int64_t>(j) * static_cast<int64_t>(n) + static_cast<int64_t>(i)];
}

__global__ void symmetrize_inplace_f64_kernel(double* __restrict__ a, int n) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= n || j >= n) return;
  if (i <= j) return;
  const int64_t idx0 = static_cast<int64_t>(i) * static_cast<int64_t>(n) + static_cast<int64_t>(j);
  const int64_t idx1 = static_cast<int64_t>(j) * static_cast<int64_t>(n) + static_cast<int64_t>(i);
  const double v0 = a[idx0];
  const double v1 = a[idx1];
  const double v = 0.5 * (v0 + v1);
  a[idx0] = v;
  a[idx1] = v;
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

__global__ void unpack_qp_to_bq_f64_kernel(
    const double* __restrict__ b_qp,  // (naux, ntri) packed, row-major
    int nao,
    int naux,
    int q0,
    int q,
    double* __restrict__ out_bq) {  // (q, nao, nao) row-major
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t nao_i64 = static_cast<int64_t>(nao);
  const int64_t ntri = nao_i64 * (nao_i64 + 1) / 2;
  const int64_t total = static_cast<int64_t>(q) * nao_i64 * nao_i64;
  for (; tid < total; tid += stride) {
    int64_t t = tid;
    const int n = static_cast<int>(t % nao_i64);
    t /= nao_i64;
    const int m = static_cast<int>(t % nao_i64);
    t /= nao_i64;
    const int q_local = static_cast<int>(t);
    const int q_abs = q0 + q_local;
    if (q_abs < 0 || q_abs >= naux) continue;

    int a = m;
    int b = n;
    if (b > a) {
      const int tmp = a;
      a = b;
      b = tmp;
    }
    const int64_t p = tri_index_s2(a, b);
    const double v = b_qp[static_cast<int64_t>(q_abs) * ntri + p];
    out_bq[tid] = v;
  }
}

__global__ void extract_qp_rows_fullcols_f64_kernel(
    const double* __restrict__ b_qp,  // (naux, ntri) packed, row-major
    int nao,
    int naux,
    int q0,
    int q,
    int row0,
    int row_count,
    double* __restrict__ out_rows) {  // (q*row_count, nao) row-major
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t nao_i64 = static_cast<int64_t>(nao);
  const int64_t ntri = nao_i64 * (nao_i64 + 1) / 2;
  const int64_t total = static_cast<int64_t>(q) * static_cast<int64_t>(row_count) * nao_i64;
  for (; tid < total; tid += stride) {
    int64_t t = tid;
    const int mu = static_cast<int>(t % nao_i64);
    t /= nao_i64;
    const int r = static_cast<int>(t % static_cast<int64_t>(row_count));
    t /= static_cast<int64_t>(row_count);
    const int q_local = static_cast<int>(t);
    const int q_abs = q0 + q_local;
    if (q_abs < 0 || q_abs >= naux) continue;
    const int row = row0 + r;
    if (row < 0 || row >= nao) continue;

    int a = row;
    int b = mu;
    if (b > a) {
      const int tmp = a;
      a = b;
      b = tmp;
    }
    const int64_t p = tri_index_s2(a, b);
    out_rows[tid] = b_qp[static_cast<int64_t>(q_abs) * ntri + p];
  }
}

__global__ void repack_y2d_to_yflat_f64_kernel(
    const double* __restrict__ y2d,  // (q*row_count, nao) row-major
    int q,
    int row_count,
    int nao,
    double* __restrict__ yflat) {  // (row_count, q*nao) row-major
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t nao_i64 = static_cast<int64_t>(nao);
  const int64_t q_i64 = static_cast<int64_t>(q);
  const int64_t k = q_i64 * nao_i64;
  const int64_t total = static_cast<int64_t>(row_count) * k;
  for (; tid < total; tid += stride) {
    int64_t t = tid;
    const int64_t col = t % k;
    t /= k;
    const int r = static_cast<int>(t);
    const int q_local = static_cast<int>(col / nao_i64);
    const int mu = static_cast<int>(col - static_cast<int64_t>(q_local) * nao_i64);
    const int64_t src = (static_cast<int64_t>(q_local) * static_cast<int64_t>(row_count) + static_cast<int64_t>(r)) *
                            nao_i64 +
                        static_cast<int64_t>(mu);
    yflat[tid] = y2d[src];
  }
}

__global__ void extract_qp_cols_to_zflat_t_f64_kernel(
    const double* __restrict__ b_qp,  // (naux, ntri) packed, row-major
    int nao,
    int naux,
    int q0,
    int q,
    int col0,
    int col_count,
    double* __restrict__ zflat_t) {  // (q*nao, col_count) row-major
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t nao_i64 = static_cast<int64_t>(nao);
  const int64_t ntri = nao_i64 * (nao_i64 + 1) / 2;
  const int64_t total = static_cast<int64_t>(q) * nao_i64 * static_cast<int64_t>(col_count);
  for (; tid < total; tid += stride) {
    int64_t t = tid;
    const int j = static_cast<int>(t % static_cast<int64_t>(col_count));
    t /= static_cast<int64_t>(col_count);
    const int mu = static_cast<int>(t % nao_i64);
    t /= nao_i64;
    const int q_local = static_cast<int>(t);
    const int q_abs = q0 + q_local;
    if (q_abs < 0 || q_abs >= naux) continue;
    const int row = col0 + j;
    if (row < 0 || row >= nao) continue;

    int a = row;
    int b = mu;
    if (b > a) {
      const int tmp = a;
      a = b;
      b = tmp;
    }
    const int64_t p = tri_index_s2(a, b);
    zflat_t[tid] = b_qp[static_cast<int64_t>(q_abs) * ntri + p];
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

extern "C" void hf_df_jk_symmetrize_inplace_f64(double* a, int n, cudaStream_t stream) {
  if (!a || n <= 0) return;
  const dim3 block(16, 16, 1);
  const dim3 grid((static_cast<uint32_t>(n) + block.x - 1) / block.x,
                  (static_cast<uint32_t>(n) + block.y - 1) / block.y,
                  1);
  symmetrize_inplace_f64_kernel<<<grid, block, 0, stream>>>(a, n);
  throw_on_cuda_error(cudaGetLastError(), "symmetrize_inplace_f64_kernel launch");
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

extern "C" void hf_df_jk_unpack_qp_to_bq_f64(
    const double* b_qp, int nao, int naux, int q0, int q, double* out_bq, cudaStream_t stream) {
  if (!b_qp || !out_bq || nao <= 0 || naux <= 0 || q <= 0) return;
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(q) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  unpack_qp_to_bq_f64_kernel<<<blocks, threads, 0, stream>>>(b_qp, nao, naux, q0, q, out_bq);
  throw_on_cuda_error(cudaGetLastError(), "unpack_qp_to_bq_f64_kernel launch");
}

extern "C" void hf_df_jk_extract_qp_rows_fullcols_f64(
    const double* b_qp, int nao, int naux, int q0, int q, int row0, int row_count, double* out_rows, cudaStream_t stream) {
  if (!b_qp || !out_rows || nao <= 0 || naux <= 0 || q <= 0 || row_count <= 0) return;
  const int threads = 256;
  const int64_t total =
      static_cast<int64_t>(q) * static_cast<int64_t>(row_count) * static_cast<int64_t>(nao);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  extract_qp_rows_fullcols_f64_kernel<<<blocks, threads, 0, stream>>>(b_qp, nao, naux, q0, q, row0, row_count, out_rows);
  throw_on_cuda_error(cudaGetLastError(), "extract_qp_rows_fullcols_f64_kernel launch");
}

extern "C" void hf_df_jk_repack_y2d_to_yflat_f64(
    const double* y2d, int q, int row_count, int nao, double* out_yflat, cudaStream_t stream) {
  if (!y2d || !out_yflat || q <= 0 || row_count <= 0 || nao <= 0) return;
  const int threads = 256;
  const int64_t total =
      static_cast<int64_t>(q) * static_cast<int64_t>(row_count) * static_cast<int64_t>(nao);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  repack_y2d_to_yflat_f64_kernel<<<blocks, threads, 0, stream>>>(y2d, q, row_count, nao, out_yflat);
  throw_on_cuda_error(cudaGetLastError(), "repack_y2d_to_yflat_f64_kernel launch");
}

extern "C" void hf_df_jk_extract_qp_cols_to_zflat_t_f64(
    const double* b_qp, int nao, int naux, int q0, int q, int col0, int col_count, double* out_zflat_t, cudaStream_t stream) {
  if (!b_qp || !out_zflat_t || nao <= 0 || naux <= 0 || q <= 0 || col_count <= 0) return;
  const int threads = 256;
  const int64_t total =
      static_cast<int64_t>(q) * static_cast<int64_t>(nao) * static_cast<int64_t>(col_count);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  extract_qp_cols_to_zflat_t_f64_kernel<<<blocks, threads, 0, stream>>>(b_qp, nao, naux, q0, q, col0, col_count, out_zflat_t);
  throw_on_cuda_error(cudaGetLastError(), "extract_qp_cols_to_zflat_t_f64_kernel launch");
}
