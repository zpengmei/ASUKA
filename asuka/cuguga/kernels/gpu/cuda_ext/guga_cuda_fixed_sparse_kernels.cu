#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ double warp_reduce_sum(double v) {
  // Warp-wide sum reduction.
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__global__ void ell_spmv_f64_warp_per_row_kernel(
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ val,
    int nrows,
    int width,
    const double* __restrict__ x,
    double* __restrict__ y,
    int add) {
  int lane = (int)(threadIdx.x & 31);
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_in_block = (int)(threadIdx.x >> 5);
  int warp_id = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (warp_id >= nrows) return;

  int row = warp_id;
  int base = row * width;

  double sum = 0.0;
  for (int t = lane; t < width; t += 32) {
    int col = col_idx[base + t];
    if (col >= 0) {
      sum += val[base + t] * x[col];
    }
  }

  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    if (add) {
      y[row] += sum;
    } else {
      y[row] = sum;
    }
  }
}

__global__ void sell_spmv_f64_warp_per_row_kernel(
    const int64_t* __restrict__ slice_ptr,
    const int32_t* __restrict__ slice_width,
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ val,
    int nrows,
    int slice_height,
    const double* __restrict__ x,
    double* __restrict__ y,
    int add) {
  int lane = (int)(threadIdx.x & 31);
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_in_block = (int)(threadIdx.x >> 5);
  int warp_id = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (warp_id >= nrows) return;

  int row = warp_id;
  int slice = row / slice_height;
  int row_in_slice = row - slice * slice_height;
  int width = slice_width[slice];
  int64_t base = slice_ptr[slice] + (int64_t)row_in_slice * (int64_t)width;

  double sum = 0.0;
  for (int t = lane; t < width; t += 32) {
    int col = col_idx[base + (int64_t)t];
    if (col >= 0) {
      sum += val[base + (int64_t)t] * x[col];
    }
  }

  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    if (add) {
      y[row] += sum;
    } else {
      y[row] = sum;
    }
  }
}

template <int VEC>
__global__ void ell_spmm_f64_warp_per_row_kernel(
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ val,
    int nrows,
    int width,
    const double* __restrict__ x,
    int ldx,
    double* __restrict__ y,
    int ldy,
    int add) {
  int lane = (int)(threadIdx.x & 31);
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_in_block = (int)(threadIdx.x >> 5);
  int warp_id = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (warp_id >= nrows) return;

  int row = warp_id;
  int base = row * width;

  double sum[VEC];
#pragma unroll
  for (int k = 0; k < VEC; k++) sum[k] = 0.0;

  for (int t = lane; t < width; t += 32) {
    int col = col_idx[base + t];
    if (col >= 0) {
      double w = val[base + t];
      const double* x_row = x + (int64_t)col * (int64_t)ldx;
#pragma unroll
      for (int k = 0; k < VEC; k++) {
        sum[k] += w * x_row[k];
      }
    }
  }

#pragma unroll
  for (int k = 0; k < VEC; k++) {
    sum[k] = warp_reduce_sum(sum[k]);
  }

  if (lane == 0) {
    double* y_row = y + (int64_t)row * (int64_t)ldy;
    if (add) {
#pragma unroll
      for (int k = 0; k < VEC; k++) y_row[k] += sum[k];
    } else {
#pragma unroll
      for (int k = 0; k < VEC; k++) y_row[k] = sum[k];
    }
  }
}

template <int VEC>
__global__ void sell_spmm_f64_warp_per_row_kernel(
    const int64_t* __restrict__ slice_ptr,
    const int32_t* __restrict__ slice_width,
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ val,
    int nrows,
    int slice_height,
    const double* __restrict__ x,
    int ldx,
    double* __restrict__ y,
    int ldy,
    int add) {
  int lane = (int)(threadIdx.x & 31);
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_in_block = (int)(threadIdx.x >> 5);
  int warp_id = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (warp_id >= nrows) return;

  int row = warp_id;
  int slice = row / slice_height;
  int row_in_slice = row - slice * slice_height;
  int width = slice_width[slice];
  int64_t base = slice_ptr[slice] + (int64_t)row_in_slice * (int64_t)width;

  double sum[VEC];
#pragma unroll
  for (int k = 0; k < VEC; k++) sum[k] = 0.0;

  for (int t = lane; t < width; t += 32) {
    int col = col_idx[base + (int64_t)t];
    if (col >= 0) {
      double w = val[base + (int64_t)t];
      const double* x_row = x + (int64_t)col * (int64_t)ldx;
#pragma unroll
      for (int k = 0; k < VEC; k++) {
        sum[k] += w * x_row[k];
      }
    }
  }

#pragma unroll
  for (int k = 0; k < VEC; k++) {
    sum[k] = warp_reduce_sum(sum[k]);
  }

  if (lane == 0) {
    double* y_row = y + (int64_t)row * (int64_t)ldy;
    if (add) {
#pragma unroll
      for (int k = 0; k < VEC; k++) y_row[k] += sum[k];
    } else {
#pragma unroll
      for (int k = 0; k < VEC; k++) y_row[k] = sum[k];
    }
  }
}

__global__ void ell_spmm_f64_lanevec_warp_per_row_kernel(
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ val,
    int nrows,
    int width,
    const double* __restrict__ x,
    int ldx,
    double* __restrict__ y,
    int ldy,
    int nvec,
    int add) {
  int lane = (int)(threadIdx.x & 31);
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_in_block = (int)(threadIdx.x >> 5);
  int warp_id = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (warp_id >= nrows) return;
  if ((unsigned)lane >= (unsigned)nvec) return;

  int row = warp_id;
  int base = row * width;

  double sum = 0.0;
  for (int t = 0; t < width; t++) {
    int col = col_idx[base + t];
    if (col >= 0) {
      sum += val[base + t] * x[(int64_t)col * (int64_t)ldx + (int64_t)lane];
    }
  }

  double* y_row = y + (int64_t)row * (int64_t)ldy;
  if (add) {
    y_row[lane] += sum;
  } else {
    y_row[lane] = sum;
  }
}

}  // namespace

extern "C" cudaError_t guga_ell_spmv_f64_launch_stream(
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int width,
    const double* x,
    double* y,
    int add,
    cudaStream_t stream,
    int threads) {
  if (nrows < 0 || width < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (nrows == 0) return cudaSuccess;
  if (width == 0) {
    if (add) return cudaSuccess;
    return cudaMemsetAsync(y, 0, (size_t)nrows * sizeof(double), stream);
  }
  if (!col_idx || !val || !x || !y) return cudaErrorInvalidValue;

  int warps_per_block = threads / 32;
  int blocks = (nrows + warps_per_block - 1) / warps_per_block;
  ell_spmv_f64_warp_per_row_kernel<<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x, y, add);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_sell_spmv_f64_launch_stream(
    const int64_t* slice_ptr,
    const int32_t* slice_width,
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int slice_height,
    const double* x,
    double* y,
    int add,
    cudaStream_t stream,
    int threads) {
  if (nrows < 0 || slice_height <= 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (nrows == 0) return cudaSuccess;
  // Empty matrix (nelems==0) is represented by null `col_idx/val` pointers (zero-sized device arrays).
  // In that case, the product is zero.
  if (!col_idx || !val) {
    if (add) return cudaSuccess;
    return cudaMemsetAsync(y, 0, (size_t)nrows * sizeof(double), stream);
  }
  if (!slice_ptr || !slice_width || !x || !y) return cudaErrorInvalidValue;

  int warps_per_block = threads / 32;
  int blocks = (nrows + warps_per_block - 1) / warps_per_block;
  sell_spmv_f64_warp_per_row_kernel<<<blocks, threads, 0, stream>>>(
      slice_ptr, slice_width, col_idx, val, nrows, slice_height, x, y, add);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_ell_spmv_f64_launch(
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int width,
    const double* x,
    double* y,
    int add,
    int threads) {
  return guga_ell_spmv_f64_launch_stream(col_idx, val, nrows, width, x, y, add, /*stream=*/0, threads);
}

extern "C" cudaError_t guga_sell_spmv_f64_launch(
    const int64_t* slice_ptr,
    const int32_t* slice_width,
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int slice_height,
    const double* x,
    double* y,
    int add,
    int threads) {
  return guga_sell_spmv_f64_launch_stream(
      slice_ptr, slice_width, col_idx, val, nrows, slice_height, x, y, add, /*stream=*/0, threads);
}

extern "C" cudaError_t guga_ell_spmm_f64_launch_stream(
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int width,
    const double* x,
    int ldx,
    double* y,
    int ldy,
    int nvec,
    int add,
    cudaStream_t stream,
    int threads) {
  if (nrows < 0 || width < 0 || nvec < 0) return cudaErrorInvalidValue;
  if (ldx < 0 || ldy < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (nrows == 0 || nvec == 0) return cudaSuccess;
  if (ldx < nvec || ldy < nvec) return cudaErrorInvalidValue;
  if (width == 0) {
    if (add) return cudaSuccess;
    return cudaMemsetAsync(y, 0, (size_t)nrows * (size_t)ldy * sizeof(double), stream);
  }
  if (!col_idx || !val || !x || !y) return cudaErrorInvalidValue;

  int warps_per_block = threads / 32;
  int blocks = (nrows + warps_per_block - 1) / warps_per_block;

  // For very small widths, the usual "lane iterates nonzeros + warp reduction" mapping
  // underutilizes the warp (many lanes do no work, but reductions still happen).
  // Use a specialized mapping where lane=vector index for nvec<=32.
  constexpr int LANE_VEC_MAX = 32;
  constexpr int LANE_VEC_WIDTH_MAX = 4;
  if (width <= LANE_VEC_WIDTH_MAX && nvec <= LANE_VEC_MAX) {
    ell_spmm_f64_lanevec_warp_per_row_kernel<<<blocks, threads, 0, stream>>>(
        col_idx, val, nrows, width, x, ldx, y, ldy, nvec, add);
    return cudaGetLastError();
  }

  int v0 = 0;
  while (v0 < nvec) {
    int rem = nvec - v0;
    int vec = (rem >= 16) ? 16 : (rem >= 8) ? 8 : (rem >= 4) ? 4 : (rem >= 2) ? 2 : 1;
    const double* x0 = x + v0;
    double* y0 = y + v0;
    switch (vec) {
      case 16:
        ell_spmm_f64_warp_per_row_kernel<16>
            <<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x0, ldx, y0, ldy, add);
        break;
      case 8:
        ell_spmm_f64_warp_per_row_kernel<8>
            <<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x0, ldx, y0, ldy, add);
        break;
      case 4:
        ell_spmm_f64_warp_per_row_kernel<4>
            <<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x0, ldx, y0, ldy, add);
        break;
      case 2:
        ell_spmm_f64_warp_per_row_kernel<2>
            <<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x0, ldx, y0, ldy, add);
        break;
      case 1:
        ell_spmm_f64_warp_per_row_kernel<1>
            <<<blocks, threads, 0, stream>>>(col_idx, val, nrows, width, x0, ldx, y0, ldy, add);
        break;
      default:
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    v0 += vec;
  }

  return cudaSuccess;
}

extern "C" cudaError_t guga_sell_spmm_f64_launch_stream(
    const int64_t* slice_ptr,
    const int32_t* slice_width,
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int slice_height,
    const double* x,
    int ldx,
    double* y,
    int ldy,
    int nvec,
    int add,
    cudaStream_t stream,
    int threads) {
  if (nrows < 0 || slice_height <= 0 || nvec < 0) return cudaErrorInvalidValue;
  if (ldx < 0 || ldy < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (nrows == 0 || nvec == 0) return cudaSuccess;
  if (ldx < nvec || ldy < nvec) return cudaErrorInvalidValue;
  // Empty matrix (nelems==0) is represented by null `col_idx/val` pointers (zero-sized device arrays).
  if (!col_idx || !val) {
    if (add) return cudaSuccess;
    return cudaMemsetAsync(y, 0, (size_t)nrows * (size_t)ldy * sizeof(double), stream);
  }
  if (!slice_ptr || !slice_width || !x || !y) return cudaErrorInvalidValue;

  int warps_per_block = threads / 32;
  int blocks = (nrows + warps_per_block - 1) / warps_per_block;

  int v0 = 0;
  while (v0 < nvec) {
    int rem = nvec - v0;
    int vec = (rem >= 16) ? 16 : (rem >= 8) ? 8 : (rem >= 4) ? 4 : (rem >= 2) ? 2 : 1;
    const double* x0 = x + v0;
    double* y0 = y + v0;
    switch (vec) {
      case 16:
        sell_spmm_f64_warp_per_row_kernel<16><<<blocks, threads, 0, stream>>>(
            slice_ptr, slice_width, col_idx, val, nrows, slice_height, x0, ldx, y0, ldy, add);
        break;
      case 8:
        sell_spmm_f64_warp_per_row_kernel<8><<<blocks, threads, 0, stream>>>(
            slice_ptr, slice_width, col_idx, val, nrows, slice_height, x0, ldx, y0, ldy, add);
        break;
      case 4:
        sell_spmm_f64_warp_per_row_kernel<4><<<blocks, threads, 0, stream>>>(
            slice_ptr, slice_width, col_idx, val, nrows, slice_height, x0, ldx, y0, ldy, add);
        break;
      case 2:
        sell_spmm_f64_warp_per_row_kernel<2><<<blocks, threads, 0, stream>>>(
            slice_ptr, slice_width, col_idx, val, nrows, slice_height, x0, ldx, y0, ldy, add);
        break;
      case 1:
        sell_spmm_f64_warp_per_row_kernel<1><<<blocks, threads, 0, stream>>>(
            slice_ptr, slice_width, col_idx, val, nrows, slice_height, x0, ldx, y0, ldy, add);
        break;
      default:
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    v0 += vec;
  }

  return cudaSuccess;
}

extern "C" cudaError_t guga_ell_spmm_f64_launch(
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int width,
    const double* x,
    int ldx,
    double* y,
    int ldy,
    int nvec,
    int add,
    int threads) {
  return guga_ell_spmm_f64_launch_stream(col_idx, val, nrows, width, x, ldx, y, ldy, nvec, add, /*stream=*/0, threads);
}

extern "C" cudaError_t guga_sell_spmm_f64_launch(
    const int64_t* slice_ptr,
    const int32_t* slice_width,
    const int32_t* col_idx,
    const double* val,
    int nrows,
    int slice_height,
    const double* x,
    int ldx,
    double* y,
    int ldy,
    int nvec,
    int add,
    int threads) {
  return guga_sell_spmm_f64_launch_stream(
      slice_ptr,
      slice_width,
      col_idx,
      val,
      nrows,
      slice_height,
      x,
      ldx,
      y,
      ldy,
      nvec,
      add,
      /*stream=*/0,
      threads);
}
