#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace {

__device__ __forceinline__ double caspt2_abs(double x) { return fabs(x); }

__inline__ __device__ double warp_reduce_sum(double val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ double block_reduce_sum(double val) {
  static __shared__ double shared[32];  // max 1024 threads / 32 = 32 warps
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;
}

__global__ void caspt2_apply_h0diag_sr_f64_kernel(
    const double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    const double* __restrict__ bd,
    const double* __restrict__ id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double alpha,
    double beta,
    double denom_tol,
    double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1) {
  const int64_t n = (int64_t)nin * (int64_t)nis;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    int i = (int)(idx / (int64_t)nis);
    int j = (int)(idx - (int64_t)i * (int64_t)nis);

    double d0 = bd[i] + id[j];
    double d = d0;
    if (imag_shift != 0.0) {
      if (caspt2_abs(d0) > denom_tol) {
        d = d0 + (imag_shift * imag_shift) / d0;
      }
    }
    d += real_shift;

    double xv = x[(int64_t)i * x_s0 + (int64_t)j * x_s1];
    double yv = y[(int64_t)i * y_s0 + (int64_t)j * y_s1];
    y[(int64_t)i * y_s0 + (int64_t)j * y_s1] = beta * yv + alpha * d * xv;
  }
}

__global__ void caspt2_apply_precond_sr_f64_kernel(
    const double* __restrict__ r,
    int64_t r_s0,
    int64_t r_s1,
    const double* __restrict__ bd,
    const double* __restrict__ id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double scale,
    double denom_tol,
    double* __restrict__ out,
    int64_t out_s0,
    int64_t out_s1) {
  const int64_t n = (int64_t)nin * (int64_t)nis;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    int i = (int)(idx / (int64_t)nis);
    int j = (int)(idx - (int64_t)i * (int64_t)nis);

    double d0 = bd[i] + id[j];
    double d = d0;
    if (imag_shift != 0.0) {
      if (caspt2_abs(d0) > denom_tol) {
        d = d0 + (imag_shift * imag_shift) / d0;
      }
    }
    d += real_shift;

    double rv = r[(int64_t)i * r_s0 + (int64_t)j * r_s1];
    double zv = 0.0;
    if (caspt2_abs(d) > denom_tol) {
      zv = scale * rv / d;
    }
    out[(int64_t)i * out_s0 + (int64_t)j * out_s1] = zv;
  }
}

__global__ void caspt2_mltsca_f64_kernel(
    int imltop,
    const int32_t* __restrict__ lst1,
    int n1,
    const int32_t* __restrict__ lst2,
    int n2,
    double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    double v10,
    double v11,
    double v20,
    double v21) {
  const int64_t total = (int64_t)n1 * (int64_t)n2;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; k < total; k += stride) {
    const int i1 = (int)(k / (int64_t)n2);
    const int i2 = (int)(k - (int64_t)i1 * (int64_t)n2);

    const int l11 = lst1[0 * n1 + i1];
    const int l12 = lst1[1 * n1 + i1];
    const int l13 = lst1[2 * n1 + i1];
    const int c1 = lst1[3 * n1 + i1] & 1;

    const int l21 = lst2[0 * n2 + i2];
    const int l22 = lst2[1 * n2 + i2];
    const int l23 = lst2[2 * n2 + i2];
    const int c2 = lst2[3 * n2 + i2] & 1;

    const double a1 = (c1 == 0) ? v10 : v11;
    const double a2 = (c2 == 0) ? v20 : v21;
    const double fac = a1 * a2 * f[(int64_t)l12 * f_s0 + (int64_t)l22 * f_s1];

    if (imltop == 0) {
      const double yv = y[(int64_t)l13 * y_s0 + (int64_t)l23 * y_s1];
      atomicAdd(&x[(int64_t)l11 * x_s0 + (int64_t)l21 * x_s1], fac * yv);
    } else if (imltop == 1) {
      const double xv = x[(int64_t)l11 * x_s0 + (int64_t)l21 * x_s1];
      atomicAdd(&y[(int64_t)l13 * y_s0 + (int64_t)l23 * y_s1], fac * xv);
    }
  }
}

__global__ void caspt2_mltdxp_f64_kernel(
    int imltop,
    const int32_t* __restrict__ lst1,
    int n1,
    const int32_t* __restrict__ lst2,
    int n2,
    double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_a,
    double v10,
    double v11,
    double v20,
    double v21) {
  const int64_t total_pairs = (int64_t)n1 * (int64_t)n2;
  const int64_t total = total_pairs * (int64_t)len_a;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int a = (int)(t % (int64_t)len_a);
    const int64_t k = t / (int64_t)len_a;

    const int i1 = (int)(k / (int64_t)n2);
    const int i2 = (int)(k - (int64_t)i1 * (int64_t)n2);

    const int l11 = lst1[0 * n1 + i1];
    const int l12 = lst1[1 * n1 + i1];
    const int l13 = lst1[2 * n1 + i1];
    const int c1 = lst1[3 * n1 + i1] & 1;

    const int l21 = lst2[0 * n2 + i2];
    const int l22 = lst2[1 * n2 + i2];
    const int l23 = lst2[2 * n2 + i2];
    const int c2 = lst2[3 * n2 + i2] & 1;

    const double a1 = (c1 == 0) ? v10 : v11;
    const double a2 = (c2 == 0) ? v20 : v21;
    const double fac = a1 * a2 * f[(int64_t)l12 * f_s0 + (int64_t)l22 * f_s1];

    const int64_t x_off = (int64_t)l11 * x_s0 + (int64_t)l21 * x_s1 + (int64_t)a * x_s2;
    const int64_t y_off = (int64_t)l13 * y_s0 + (int64_t)l23 * y_s1 + (int64_t)a * y_s2;

    if (imltop == 0) {
      const double yv = y[y_off];
      atomicAdd(&x[x_off], fac * yv);
    } else if (imltop == 1) {
      const double xv = x[x_off];
      atomicAdd(&y[y_off], fac * xv);
    }
  }
}

__global__ void caspt2_mltmv_f64_kernel_imltop0(
    const int32_t* __restrict__ lst1,
    int n1,
    double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    const double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_i,
    int len_a,
    double v10,
    double v11) {
  const int64_t total = (int64_t)n1 * (int64_t)len_i;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int i = (int)(t % (int64_t)len_i);
    const int e = (int)(t / (int64_t)len_i);

    const int l1 = lst1[0 * n1 + e];
    const int l2 = lst1[1 * n1 + e];
    const int l3 = lst1[2 * n1 + e];
    const int c = lst1[3 * n1 + e] & 1;
    const double a = (c == 0) ? v10 : v11;

    double sum = 0.0;
    const int64_t y_base = (int64_t)l3 * y_s0 + (int64_t)i * y_s1;
    const int64_t f_base = (int64_t)l2 * f_s0;
    for (int aa = 0; aa < len_a; ++aa) {
      const double yv = y[y_base + (int64_t)aa * y_s2];
      const double fv = f[f_base + (int64_t)aa * f_s1];
      sum += yv * fv;
    }
    atomicAdd(&x[(int64_t)l1 * x_s0 + (int64_t)i * x_s1], a * sum);
  }
}

__global__ void caspt2_mltmv_f64_kernel_imltop1(
    const int32_t* __restrict__ lst1,
    int n1,
    const double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_i,
    int len_a,
    double v10,
    double v11) {
  const int64_t total = (int64_t)n1 * (int64_t)len_i * (int64_t)len_a;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int aa = (int)(t % (int64_t)len_a);
    const int64_t tmp = t / (int64_t)len_a;
    const int i = (int)(tmp % (int64_t)len_i);
    const int e = (int)(tmp / (int64_t)len_i);

    const int l1 = lst1[0 * n1 + e];
    const int l2 = lst1[1 * n1 + e];
    const int l3 = lst1[2 * n1 + e];
    const int c = lst1[3 * n1 + e] & 1;
    const double a = (c == 0) ? v10 : v11;

    const double xv = x[(int64_t)l1 * x_s0 + (int64_t)i * x_s1];
    const double fv = f[(int64_t)l2 * f_s0 + (int64_t)aa * f_s1];
    atomicAdd(&y[(int64_t)l3 * y_s0 + (int64_t)i * y_s1 + (int64_t)aa * y_s2], a * xv * fv);
  }
}

__global__ void caspt2_mltr1_f64_kernel_imltop0(
    const int32_t* __restrict__ lst1,
    int n1,
    double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    const double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    int len_p,
    int len_q,
    double v10,
    double v11) {
  const int64_t total = (int64_t)n1 * (int64_t)len_p * (int64_t)len_q;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int q = (int)(t % (int64_t)len_q);
    const int64_t tmp = t / (int64_t)len_q;
    const int p = (int)(tmp % (int64_t)len_p);
    const int e = (int)(tmp / (int64_t)len_p);

    const int l1 = lst1[0 * n1 + e];
    const int l2 = lst1[1 * n1 + e];
    const int l3 = lst1[2 * n1 + e];
    const int c = lst1[3 * n1 + e] & 1;
    const double a = (c == 0) ? v10 : v11;

    const double fv = f[(int64_t)l2 * f_s0 + (int64_t)p * f_s1];
    const double yv = y[(int64_t)l3 * y_s0 + (int64_t)q * y_s1];
    atomicAdd(&x[(int64_t)l1 * x_s0 + (int64_t)p * x_s1 + (int64_t)q * x_s2], a * fv * yv);
  }
}

__global__ void caspt2_mltr1_f64_kernel_imltop1(
    const int32_t* __restrict__ lst1,
    int n1,
    const double* __restrict__ x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* __restrict__ f,
    int64_t f_s0,
    int64_t f_s1,
    double* __restrict__ y,
    int64_t y_s0,
    int64_t y_s1,
    int len_p,
    int len_q,
    double v10,
    double v11) {
  const int64_t total = (int64_t)n1 * (int64_t)len_q;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int q = (int)(t % (int64_t)len_q);
    const int e = (int)(t / (int64_t)len_q);

    const int l1 = lst1[0 * n1 + e];
    const int l2 = lst1[1 * n1 + e];
    const int l3 = lst1[2 * n1 + e];
    const int c = lst1[3 * n1 + e] & 1;
    const double a = (c == 0) ? v10 : v11;

    double sum = 0.0;
    const int64_t x_base = (int64_t)l1 * x_s0 + (int64_t)q * x_s2;
    const int64_t f_base = (int64_t)l2 * f_s0;
    for (int p = 0; p < len_p; ++p) {
      const double fv = f[f_base + (int64_t)p * f_s1];
      const double xv = x[x_base + (int64_t)p * x_s1];
      sum += fv * xv;
    }
    atomicAdd(&y[(int64_t)l3 * y_s0 + (int64_t)q * y_s1], a * sum);
  }
}

__global__ void caspt2_ddot_f64_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n,
    double* __restrict__ out) {
  double sum = 0.0;
  const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    sum += x[idx] * y[idx];
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(out, sum);
  }
}

inline int64_t clamp_grid(int64_t grid) {
  // Keep grid size bounded; kernels use grid-stride loops.
  const int64_t max_grid = 65535;
  return (grid > max_grid) ? max_grid : grid;
}

}  // namespace

extern "C" void caspt2_apply_h0diag_sr_f64_launch(
    const double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* bd,
    const double* id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double alpha,
    double beta,
    double denom_tol,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    cudaStream_t stream) {
  if (nin <= 0 || nis <= 0) return;
  const int threads = 256;
  const int64_t total = (int64_t)nin * (int64_t)nis;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = (int)clamp_grid(blocks64);
  caspt2_apply_h0diag_sr_f64_kernel<<<blocks, threads, 0, stream>>>(
      x, x_s0, x_s1, bd, id, nin, nis, real_shift, imag_shift, alpha, beta, denom_tol, y, y_s0, y_s1);
}

extern "C" void caspt2_apply_precond_sr_f64_launch(
    const double* r,
    int64_t r_s0,
    int64_t r_s1,
    const double* bd,
    const double* id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double scale,
    double denom_tol,
    double* out,
    int64_t out_s0,
    int64_t out_s1,
    cudaStream_t stream) {
  if (nin <= 0 || nis <= 0) return;
  const int threads = 256;
  const int64_t total = (int64_t)nin * (int64_t)nis;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = (int)clamp_grid(blocks64);
  caspt2_apply_precond_sr_f64_kernel<<<blocks, threads, 0, stream>>>(
      r, r_s0, r_s1, bd, id, nin, nis, real_shift, imag_shift, scale, denom_tol, out, out_s0, out_s1);
}

extern "C" void caspt2_mltsca_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    const int32_t* lst2,
    int n2,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    double v10,
    double v11,
    double v20,
    double v21,
    cudaStream_t stream) {
  if (n1 <= 0 || n2 <= 0) return;
  const int threads = 256;
  const int64_t total = (int64_t)n1 * (int64_t)n2;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = (int)clamp_grid(blocks64);
  caspt2_mltsca_f64_kernel<<<blocks, threads, 0, stream>>>(
      imltop, lst1, n1, lst2, n2, x, x_s0, x_s1, f, f_s0, f_s1, y, y_s0, y_s1, v10, v11, v20, v21);
}

extern "C" void caspt2_mltdxp_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    const int32_t* lst2,
    int n2,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_a,
    double v10,
    double v11,
    double v20,
    double v21,
    cudaStream_t stream) {
  if (n1 <= 0 || n2 <= 0 || len_a <= 0) return;
  const int threads = 256;
  const int64_t total = (int64_t)n1 * (int64_t)n2 * (int64_t)len_a;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = (int)clamp_grid(blocks64);
  caspt2_mltdxp_f64_kernel<<<blocks, threads, 0, stream>>>(
      imltop,
      lst1,
      n1,
      lst2,
      n2,
      x,
      x_s0,
      x_s1,
      x_s2,
      f,
      f_s0,
      f_s1,
      y,
      y_s0,
      y_s1,
      y_s2,
      len_a,
      v10,
      v11,
      v20,
      v21);
}

extern "C" void caspt2_mltmv_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_i,
    int len_a,
    double v10,
    double v11,
    cudaStream_t stream) {
  if (n1 <= 0 || len_i <= 0 || len_a <= 0) return;
  const int threads = 256;
  if (imltop == 0) {
    const int64_t total = (int64_t)n1 * (int64_t)len_i;
    const int64_t blocks64 = (total + threads - 1) / threads;
    const int blocks = (int)clamp_grid(blocks64);
    caspt2_mltmv_f64_kernel_imltop0<<<blocks, threads, 0, stream>>>(
        lst1, n1, x, x_s0, x_s1, f, f_s0, f_s1, (const double*)y, y_s0, y_s1, y_s2, len_i, len_a, v10, v11);
  } else if (imltop == 1) {
    const int64_t total = (int64_t)n1 * (int64_t)len_i * (int64_t)len_a;
    const int64_t blocks64 = (total + threads - 1) / threads;
    const int blocks = (int)clamp_grid(blocks64);
    caspt2_mltmv_f64_kernel_imltop1<<<blocks, threads, 0, stream>>>(
        lst1, n1, (const double*)x, x_s0, x_s1, f, f_s0, f_s1, y, y_s0, y_s1, y_s2, len_i, len_a, v10, v11);
  }
}

extern "C" void caspt2_mltr1_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int len_p,
    int len_q,
    double v10,
    double v11,
    cudaStream_t stream) {
  if (n1 <= 0 || len_p <= 0 || len_q <= 0) return;
  const int threads = 256;
  if (imltop == 0) {
    const int64_t total = (int64_t)n1 * (int64_t)len_p * (int64_t)len_q;
    const int64_t blocks64 = (total + threads - 1) / threads;
    const int blocks = (int)clamp_grid(blocks64);
    caspt2_mltr1_f64_kernel_imltop0<<<blocks, threads, 0, stream>>>(
        lst1, n1, x, x_s0, x_s1, x_s2, f, f_s0, f_s1, (const double*)y, y_s0, y_s1, len_p, len_q, v10, v11);
  } else if (imltop == 1) {
    const int64_t total = (int64_t)n1 * (int64_t)len_q;
    const int64_t blocks64 = (total + threads - 1) / threads;
    const int blocks = (int)clamp_grid(blocks64);
    caspt2_mltr1_f64_kernel_imltop1<<<blocks, threads, 0, stream>>>(
        lst1, n1, (const double*)x, x_s0, x_s1, x_s2, f, f_s0, f_s1, y, y_s0, y_s1, len_p, len_q, v10, v11);
  }
}

extern "C" void caspt2_ddot_f64_launch(
    const double* x,
    const double* y,
    int64_t n,
    double* out,
    cudaStream_t stream) {
  if (n <= 0) return;
  const int threads = 256;
  const int64_t blocks64 = (n + threads - 1) / threads;
  const int blocks = (int)clamp_grid(blocks64);
  caspt2_ddot_f64_kernel<<<blocks, threads, 0, stream>>>(x, y, n, out);
}

