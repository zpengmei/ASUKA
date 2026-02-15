#ifndef CUERI_CUDA_RYS_DEVICE_CUH
#define CUERI_CUDA_RYS_DEVICE_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace cueri_rys {

constexpr double kPi = 3.141592653589793238462643383279502884;

__device__ inline double clamp_double(double x, double lo, double hi) { return fmin(fmax(x, lo), hi); }

template <int MMAX>
__device__ inline void boys_fm_ref(double T, double* F) {
  // Computes F[0..MMAX] for Boys function:
  //   F_m(T) = ∫_0^1 t^{2m} exp(-T t^2) dt
  // Strategy:
  // - T tiny: compute F_MMAX by series, recurse downward.
  // - else: compute F0 via erf, recurse upward.
  if (T < 5.0) {
    // series for F_MMAX(T) = sum_{k>=0} (-T)^k / (k!(2MMAX+2k+1))
    double term = 1.0;
    double Fm = 0.0;
    // Upward recursion can lose many digits for moderate T and large m due to cancellation
    // against exp(-T). Prefer a robust top-series + downward recursion in this regime.
    // Use an early-exit tolerance to avoid paying for many tiny terms when T is small.
    // (This routine is performance-critical for the generic Rys microkernel.)
    constexpr int kMaxSeries = 80;
    constexpr double kTermTol = 1e-22;
    for (int k = 0; k < kMaxSeries; ++k) {
      Fm += term / static_cast<double>(2 * MMAX + 2 * k + 1);
      term *= -T / static_cast<double>(k + 1);
      if (::fabs(term) < kTermTol) break;
    }
    F[MMAX] = Fm;
    const double e = ::exp(-T);
    for (int m = MMAX; m >= 1; --m) {
      F[m - 1] = (2.0 * T * F[m] + e) / static_cast<double>(2 * m - 1);
    }
    return;
  }

  const double e = ::exp(-T);
  // F0
  F[0] = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
  for (int m = 1; m <= MMAX; ++m) {
    F[m] = ((2 * m - 1) * F[m - 1] - e) / (2.0 * T);
  }
}

template <int MMAX>
__device__ inline void boys_fm_fast(double T, double* F) {
  // Faster Boys moments for small MMAX (intended for MMAX<=5) using a cheaper
  // switching threshold than boys_fm_ref:
  // - For small/moderate T, use a short series for F_MMAX and recurse downward
  //   (stable near T=0).
  // - For larger T, compute F0 via erf and recurse upward.
  if (T <= 0.0) {
#pragma unroll
    for (int m = 0; m <= MMAX; ++m) F[m] = 1.0 / static_cast<double>(2 * m + 1);
    return;
  }

  // Empirically, upward recurrence is accurate for these small MMAX once T is
  // O(1), while the series converges quickly for T<1.
  constexpr double kSwitchT = 1.0;
  if (T < kSwitchT) {
    // series for F_MMAX(T) = sum_{k>=0} (-T)^k / (k!(2MMAX+2k+1))
    double term = 1.0;
    double Fm = 0.0;
    constexpr int kMaxSeries = 48;
    constexpr double kTermTol = 1e-22;
    for (int k = 0; k < kMaxSeries; ++k) {
      Fm += term / static_cast<double>(2 * MMAX + 2 * k + 1);
      term *= -T / static_cast<double>(k + 1);
      if (::fabs(term) < kTermTol) break;
    }
    F[MMAX] = Fm;
    const double e = ::exp(-T);
    for (int m = MMAX; m >= 1; --m) {
      F[m - 1] = (2.0 * T * F[m] + e) / static_cast<double>(2 * m - 1);
    }
    return;
  }

  const double invT = 1.0 / T;
  const double e = ::exp(-T);
  F[0] = 0.5 * ::sqrt(kPi * invT) * ::erf(::sqrt(T));
#pragma unroll
  for (int m = 1; m <= MMAX; ++m) {
    F[m] = ((2 * m - 1) * F[m - 1] - e) * (0.5 * invT);
  }
}

template <int NROOTS>
__device__ inline void tridiag_eigh_ql(double* d, double* e, double* z0) {
  // Symmetric tridiagonal eigen decomposition via implicit QL iterations.
  // Inputs:
  //   d[0..NROOTS-1] : diagonal
  //   e[0..NROOTS-2] : off-diagonal (super/sub)
  // Outputs:
  //   d[] : eigenvalues
  //   z0[]: first component of each eigenvector (orthonormal columns)
  //
  // Note: we update only the first row of the eigenvector matrix to get weights
  //   w_i = mu0 * z0[i]^2
  //
  // This is adapted from the classic tqli routine (Numerical Recipes), rewritten for 0-based indexing.

  constexpr int kMaxIter = 32;
  constexpr double kEps = 1e-14;

  // Extend e to length NROOTS with a trailing zero (the algorithm accesses e[m]).
  e[NROOTS - 1] = 0.0;

  // Initialize z0 as the first row of the identity matrix.
#pragma unroll
  for (int i = 0; i < NROOTS; ++i) {
    z0[i] = (i == 0) ? 1.0 : 0.0;
  }

  for (int l = 0; l < NROOTS; ++l) {
    int iter = 0;
    while (true) {
      int m = l;
      for (; m < (NROOTS - 1); ++m) {
        const double dd = ::fabs(d[m]) + ::fabs(d[m + 1]);
        if (::fabs(e[m]) <= kEps * dd) break;
      }
      if (m == l) break;
      if (++iter >= kMaxIter) break;

      double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      double r = ::sqrt(g * g + 1.0);
      // g = d[m] - d[l] + e[l] / (g + sign(r,g))
      g = d[m] - d[l] + e[l] / (g + ::copysign(r, g));

      double s = 1.0;
      double c = 1.0;
      double p = 0.0;

      int i = m - 1;
      for (; i >= l; --i) {
        const double f = s * e[i];
        const double b = c * e[i];
        r = ::sqrt(f * f + g * g);
        e[i + 1] = r;
        if (r == 0.0) {
          d[i + 1] -= p;
          e[m] = 0.0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        const double rr = (d[i] - g) * s + 2.0 * c * b;
        p = s * rr;
        d[i + 1] = g + p;
        g = c * rr - b;

        // Update first row of eigenvector matrix.
        const double z_ip1 = z0[i + 1];
        const double z_i = z0[i];
        z0[i + 1] = s * z_i + c * z_ip1;
        z0[i] = c * z_i - s * z_ip1;
      }

      if (r == 0.0 && i >= l) continue;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }
  }

  // Sort by ascending eigenvalue, swapping z0 accordingly.
  for (int i = 0; i < (NROOTS - 1); ++i) {
    int k = i;
    double p = d[i];
    for (int j = i + 1; j < NROOTS; ++j) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      const double td = d[i];
      d[i] = d[k];
      d[k] = td;
      const double tz = z0[i];
      z0[i] = z0[k];
      z0[k] = tz;
    }
  }
}

template <int NROOTS>
__device__ inline double poly_inner_moments(const double* a, int deg_a, const double* b, int deg_b, const double* mu, int shift) {
  // <x^shift * a(x), b(x)> = sum_{i<=deg_a} sum_{j<=deg_b} a_i b_j mu_{i+j+shift}
  // Note: our callers ensure deg_a/deg_b <= NROOTS-1, so mu indexing stays in bounds.
  double s = 0.0;
#pragma unroll
  for (int i = 0; i <= NROOTS; ++i) {
    if (i > deg_a) break;
    const double ai = a[i];
    if (ai == 0.0) continue;
#pragma unroll
    for (int j = 0; j <= NROOTS; ++j) {
      if (j > deg_b) break;
      s += ai * b[j] * mu[i + j + shift];
    }
  }
  return s;
}

template <int NROOTS>
__device__ inline void rys_roots_weights(double T, double* r, double* w) {
  static_assert(NROOTS >= 1 && NROOTS <= 11, "rys_roots_weights supports NROOTS in [1,11]");

  // Moments mu[m] = F_m(T), m=0..2*NROOTS-1
  double mu[2 * NROOTS];
  boys_fm_ref<2 * NROOTS - 1>(T, mu);

  const double mu0 = mu[0];
  if (!(mu0 > 0.0)) {
#pragma unroll
    for (int i = 0; i < NROOTS; ++i) {
      r[i] = 0.0;
      w[i] = 0.0;
    }
    return;
  }

  // Stieltjes procedure (monic orthogonal polynomials) to build Jacobi matrix.
  double alpha[NROOTS];
  double beta[NROOTS];  // beta[0] unused; beta[k] = h_k / h_{k-1} for k>=1

  double p_prev[NROOTS + 1];
  double p_curr[NROOTS + 1];
  double p_next[NROOTS + 1];

#pragma unroll
  for (int i = 0; i <= NROOTS; ++i) {
    p_prev[i] = 0.0;
    p_curr[i] = 0.0;
    p_next[i] = 0.0;
  }
  p_curr[0] = 1.0;  // P0(x) = 1

  double h_curr = mu0;
  alpha[0] = mu[1] / mu0;
  beta[0] = 0.0;

  // Build alpha[1..NROOTS-1] and beta[1..NROOTS-1].
  for (int k = 0; k < (NROOTS - 1); ++k) {
    const int deg_curr = k;
    const int deg_prev = k - 1;
    const double beta_k = (k >= 1 && h_curr > 0.0) ? beta[k] : 0.0;

    // p_next = (x - alpha[k]) p_curr - beta_k p_prev
#pragma unroll
    for (int i = 0; i <= NROOTS; ++i) p_next[i] = 0.0;

#pragma unroll
    for (int i = 0; i <= NROOTS; ++i) {
      if (i > deg_curr) break;
      const double ci = p_curr[i];
      // x * p_curr
      p_next[i + 1] += ci;
      // -alpha * p_curr
      p_next[i] -= alpha[k] * ci;
    }
    if (k >= 1) {
#pragma unroll
      for (int i = 0; i <= NROOTS; ++i) {
        if (i > deg_prev) break;
        p_next[i] -= beta_k * p_prev[i];
      }
    }

    const int deg_next = k + 1;
    double h_next = poly_inner_moments<NROOTS>(p_next, deg_next, p_next, deg_next, mu, 0);
    h_next = fmax(h_next, 0.0);
    beta[k + 1] = (h_curr > 0.0) ? (h_next / h_curr) : 0.0;
    const double a_num = poly_inner_moments<NROOTS>(p_next, deg_next, p_next, deg_next, mu, 1);
    alpha[k + 1] = (h_next > 0.0) ? (a_num / h_next) : alpha[k];

    // Shift (P_{k-1}, P_k) <- (P_k, P_{k+1})
#pragma unroll
    for (int i = 0; i <= NROOTS; ++i) {
      p_prev[i] = p_curr[i];
      p_curr[i] = p_next[i];
    }
    h_curr = h_next;
  }

  // Jacobi matrix eigenproblem -> roots/weights.
  double d[NROOTS];
  double e[NROOTS];
  double z0[NROOTS];

#pragma unroll
  for (int i = 0; i < NROOTS; ++i) {
    d[i] = alpha[i];
    e[i] = 0.0;
  }
#pragma unroll
  for (int i = 0; i < (NROOTS - 1); ++i) {
    e[i] = ::sqrt(fmax(beta[i + 1], 0.0));
  }
  tridiag_eigh_ql<NROOTS>(d, e, z0);

#pragma unroll
  for (int i = 0; i < NROOTS; ++i) {
    r[i] = clamp_double(d[i], 0.0, 1.0);
    const double v0 = z0[i];
    w[i] = fmax(mu0 * (v0 * v0), 0.0);
  }
}

template <>
__device__ inline void rys_roots_weights<1>(double T, double* r, double* w) {
  double F[2];
#ifdef CUERI_FAST_BOYS
  boys_fm_fast<1>(T, F);
#else
  boys_fm_ref<1>(T, F);
#endif
  const double F0 = F[0];
  const double F1 = F[1];
  w[0] = F0;
  r[0] = (F0 > 0.0) ? (F1 / F0) : 0.0;
  r[0] = clamp_double(r[0], 0.0, 1.0);
}

template <>
__device__ inline void rys_roots_weights<2>(double T, double* r, double* w) {
  double mu[4];
#ifdef CUERI_FAST_BOYS
  boys_fm_fast<3>(T, mu);  // mu[0..3] = F0..F3
#else
  boys_fm_ref<3>(T, mu);  // mu[0..3] = F0..F3
#endif

  const double mu0 = mu[0];
  const double mu1 = mu[1];
  const double mu2 = mu[2];
  const double mu3 = mu[3];

  const double a0 = (mu0 > 0.0) ? (mu1 / mu0) : 0.0;
  const double h1 = fmax(mu2 - (mu1 * mu1) / fmax(mu0, 1e-300), 0.0);
  const double b1 = (mu0 > 0.0) ? (h1 / mu0) : 0.0;
  const double a1_num = mu3 - 2.0 * a0 * mu2 + a0 * a0 * mu1;
  const double a1 = (h1 > 0.0) ? (a1_num / h1) : a0;
  const double s1 = ::sqrt(fmax(b1, 0.0));

  // eigenvalues of [[a0,s1],[s1,a1]]
  const double tr = a0 + a1;
  const double det = a0 * a1 - b1;
  const double disc = fmax(tr * tr - 4.0 * det, 0.0);
  const double root = ::sqrt(disc);
  double l0 = 0.5 * (tr - root);
  double l1 = 0.5 * (tr + root);
  // sort
  if (l0 > l1) {
    const double tmp = l0;
    l0 = l1;
    l1 = tmp;
  }
  r[0] = clamp_double(l0, 0.0, 1.0);
  r[1] = clamp_double(l1, 0.0, 1.0);

  // weights: w_i = mu0 * v0^2 where v is normalized eigenvector.
  // For eigenvalue l: (a0-l)v0 + s1 v1 = 0. Choose v = [s1, l-a0].
  const double v00 = s1;
  const double v01 = (l0 - a0);
  const double n0 = v00 * v00 + v01 * v01;
  w[0] = (n0 > 0.0) ? (mu0 * (v00 * v00) / n0) : 0.0;

  const double v10 = s1;
  const double v11 = (l1 - a0);
  const double n1 = v10 * v10 + v11 * v11;
  w[1] = (n1 > 0.0) ? (mu0 * (v10 * v10) / n1) : 0.0;
}

template <>
__device__ inline void rys_roots_weights<3>(double T, double* r, double* w) {
  double mu[6];
#ifdef CUERI_FAST_BOYS
  boys_fm_fast<5>(T, mu);  // mu[0..5] = F0..F5
#else
  boys_fm_ref<5>(T, mu);  // mu[0..5] = F0..F5
#endif

  const double mu0 = mu[0];
  const double mu1 = mu[1];
  const double mu2 = mu[2];
  const double mu3 = mu[3];
  const double mu4 = mu[4];
  const double mu5 = mu[5];

  const double a0 = (mu0 > 0.0) ? (mu1 / mu0) : 0.0;
  const double h1 = fmax(mu2 - (mu1 * mu1) / fmax(mu0, 1e-300), 0.0);
  const double b1 = (mu0 > 0.0) ? (h1 / mu0) : 0.0;
  const double a1_num = mu3 - 2.0 * a0 * mu2 + a0 * a0 * mu1;
  const double a1 = (h1 > 0.0) ? (a1_num / h1) : a0;

  // P2(x) = (x-a1)(x-a0) - b1 = x^2 + c1 x + c0
  const double c1 = -(a0 + a1);
  const double c0 = a0 * a1 - b1;

  // h2 = <P2,P2>
  const double h2 = fmax(
      mu4 + 2.0 * c1 * mu3 + 2.0 * c0 * mu2 + (c1 * c1) * mu2 + 2.0 * c1 * c0 * mu1 + (c0 * c0) * mu0, 0.0);
  const double b2 = (h1 > 0.0) ? (h2 / h1) : 0.0;

  // a2 = <x P2, P2> / h2
  const double a2_num =
      mu5 + 2.0 * c1 * mu4 + 2.0 * c0 * mu3 + (c1 * c1) * mu3 + 2.0 * c1 * c0 * mu2 + (c0 * c0) * mu1;
  const double a2 = (h2 > 0.0) ? (a2_num / h2) : a1;

  const double s1 = ::sqrt(fmax(b1, 0.0));
  const double s2 = ::sqrt(fmax(b2, 0.0));

  // Symmetric 3x3 eigenvalues (stable closed form).
  const double a11 = a0, a22 = a1, a33 = a2;
  const double a12 = s1, a13 = 0.0, a23 = s2;
  const double p1 = a12 * a12 + a13 * a13 + a23 * a23;
  double l0 = a11, l1 = a22, l2 = a33;
  if (p1 > 0.0) {
    const double q = (a11 + a22 + a33) / 3.0;
    const double p2 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + (a33 - q) * (a33 - q) + 2.0 * p1;
    const double p = ::sqrt(p2 / 6.0);

    // B = (A - qI)/p
    const double b11 = (a11 - q) / p;
    const double b22 = (a22 - q) / p;
    const double b33 = (a33 - q) / p;
    const double b12 = a12 / p;
    const double b13 = a13 / p;
    const double b23 = a23 / p;

    const double detB =
        b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13);
    const double rr = clamp_double(detB * 0.5, -1.0, 1.0);
    const double phi = ::acos(rr) / 3.0;

    l2 = q + 2.0 * p * ::cos(phi);
    l0 = q + 2.0 * p * ::cos(phi + (2.0 * kPi / 3.0));
    l1 = 3.0 * q - l0 - l2;
  }

  // sort ascending
  double e0 = l0, e1 = l1, e2 = l2;
  if (e0 > e1) {
    const double tmp = e0;
    e0 = e1;
    e1 = tmp;
  }
  if (e1 > e2) {
    const double tmp = e1;
    e1 = e2;
    e2 = tmp;
  }
  if (e0 > e1) {
    const double tmp = e0;
    e0 = e1;
    e1 = tmp;
  }

  r[0] = clamp_double(e0, 0.0, 1.0);
  r[1] = clamp_double(e1, 0.0, 1.0);
  r[2] = clamp_double(e2, 0.0, 1.0);

  // Weights via eigenvector first component:
  // v ∝ cross(row0, row2) for (J - λI).
  // row0 = [a0-λ, s1, 0], row2 = [0, s2, a2-λ]
  {
    const double r0x = a0 - e0;
    const double r2z = a2 - e0;
    const double v0 = s1 * r2z;
    const double v1 = -r0x * r2z;
    const double v2 = r0x * s2;
    const double n2 = v0 * v0 + v1 * v1 + v2 * v2;
    w[0] = (n2 > 0.0) ? (mu0 * (v0 * v0) / n2) : 0.0;
  }
  {
    const double r0x = a0 - e1;
    const double r2z = a2 - e1;
    const double v0 = s1 * r2z;
    const double v1 = -r0x * r2z;
    const double v2 = r0x * s2;
    const double n2 = v0 * v0 + v1 * v1 + v2 * v2;
    w[1] = (n2 > 0.0) ? (mu0 * (v0 * v0) / n2) : 0.0;
  }
  {
    const double r0x = a0 - e2;
    const double r2z = a2 - e2;
    const double v0 = s1 * r2z;
    const double v1 = -r0x * r2z;
    const double v2 = r0x * s2;
    const double n2 = v0 * v0 + v1 * v1 + v2 * v2;
    w[2] = (n2 > 0.0) ? (mu0 * (v0 * v0) / n2) : 0.0;
  }
}

}  // namespace cueri_rys

#endif  // CUERI_CUDA_RYS_DEVICE_CUH
