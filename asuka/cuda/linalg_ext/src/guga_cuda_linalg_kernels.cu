#include <cuda_runtime.h>

#include <cmath>

namespace {

__global__ void residual_kernel(const double* __restrict__ ax,
                                const double* __restrict__ x,
                                const double* __restrict__ evals,
                                int n,
                                int nroots,
                                double* __restrict__ out_r) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * nroots;
  if (idx >= total) return;
  int r = idx / n;
  out_r[idx] = ax[idx] - evals[r] * x[idx];
}

__global__ void precond_kernel(const double* __restrict__ r,
                               const double* __restrict__ diag,
                               const double* __restrict__ evals,
                               int n,
                               int nroots,
                               double denom_tol,
                               double* __restrict__ out_t) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * nroots;
  if (idx >= total) return;
  int i = idx % n;
  // Match PySCF davidson1 multi-root behavior: use the lowest root energy (e[0])
  // for preconditioning all roots.
  double denom = diag[i] - evals[0];
  double adenom = fabs(denom);
  if (adenom < denom_tol) {
    denom = (denom >= 0.0) ? denom_tol : -denom_tol;
  }
  out_t[idx] = r[idx] / denom;
}

}  // namespace

extern "C" void guga_cuda_linalg_residual_launch(
    const double* ax,
    const double* x,
    const double* evals,
    int n,
    int nroots,
    double* out_r) {
  int total = n * nroots;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  residual_kernel<<<blocks, threads>>>(ax, x, evals, n, nroots, out_r);
}

extern "C" void guga_cuda_linalg_precond_launch(
    const double* r,
    const double* diag,
    const double* evals,
    int n,
    int nroots,
    double denom_tol,
    double* out_t) {
  int total = n * nroots;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  precond_kernel<<<blocks, threads>>>(r, diag, evals, n, nroots, denom_tol, out_t);
}
