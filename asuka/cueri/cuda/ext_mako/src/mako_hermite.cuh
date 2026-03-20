#pragma once
// Mako MMD/Hermite shared device functions.
//
// Provides:
//   - Boys function evaluation for arbitrary order (with Chebyshev LUT fast path)
//   - R-integral (auxiliary Hermite integral) recurrence
//   - E-coefficient computation for Hermite-to-Cartesian transforms
//   - Generic ERI tile evaluator

#include <cmath>
#include <cstdint>

#ifdef MAKO_HAS_BOYS_LUT
#include "mako_boys_lut.cuh"
#endif

namespace mako {

static constexpr double kPi = 3.141592653589793238462643383279502884;
static constexpr double kTwoPiToFiveHalves =
    2.0 * kPi * kPi * 1.772453850905516027298167483341145182;
static constexpr int kMaxL = 4;       // max angular momentum per shell (Phase 6: d)
static constexpr int kMaxTotalL = 16; // max la+lb+lc+ld (Phase 11: gggg)
static constexpr int kMaxL_fg = 4;    // max angular momentum for f/g shells

// ---------------------------------------------------------------------------
// Boys function F_m(T) for m = 0, 1, ..., max_order.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void boys_f0_f1(double T, double& F0, double& F1) {
    if (T < 1.0) {
        double term = 1.0;
        double f1 = 0.0;
        for (int k = 0; k < 60; ++k) {
            f1 += term / static_cast<double>(2 * k + 3);
            term *= -T / static_cast<double>(k + 1);
            if (::fabs(term) < 1e-22) break;
        }
        const double expT = ::exp(-T);
        F1 = f1;
        F0 = 2.0 * T * F1 + expT;
        return;
    }
    const double expT = ::exp(-T);
    F0 = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
    F1 = (F0 - expT) / (2.0 * T);
}

__device__ inline void boys_function(double T, int max_order, double* F) {
    // Compute F_m(T) for m = 0..max_order.
    if (max_order <= 1) {
        boys_f0_f1(T, F[0], F[1]);
        return;
    }
    if (T < 1e-14) {
        for (int m = 0; m <= max_order; ++m)
            F[m] = 1.0 / static_cast<double>(2 * m + 1);
        return;
    }
    const double expT = ::exp(-T);

    // For small T: series for F_{max_order}, then downward recursion.
    // For large T: compute F_0 and F_1 via the stable erf/exp path,
    //              then use F_0 for the asymptotic starting point and
    //              downward recursion from a high enough order.
    // Two-branch approach:
    // Small T: alternating series for F_{max_order}, then downward recursion.
    // Large T: compute F_0 via erf, then upward recursion (stable for m << T).

    // Use series+downward ONLY for very small T.  The alternating series
    // suffers catastrophic cancellation starting around T ≈ 5-7 for all
    // max_order values (partial sums oscillate up to T^k/k!).
    // Upward recursion from F_0(erf) is stable for all T > 0 when
    // max_order < T (recurrence coefficient (2m+1)/(2T) < 1).
    // Conservative threshold: T < 5 uses series, T >= 5 uses upward.
    if (T < 5.0) {
        double term = 1.0;
        double fm = 0.0;
        for (int k = 0; k < 400; ++k) {
            fm += term / static_cast<double>(2 * (max_order + k) + 1);
            term *= -T / static_cast<double>(k + 1);
            if (::fabs(term) < 1e-22) break;
        }
        F[max_order] = fm;
        for (int m = max_order - 1; m >= 0; --m) {
            F[m] = (2.0 * T * F[m + 1] + expT) / static_cast<double>(2 * m + 1);
        }
    } else {
        // Large T: F_0 via erf, then upward recursion.
        F[0] = 0.5 * ::sqrt(kPi / T) * ::erf(::sqrt(T));
        for (int m = 0; m < max_order; ++m) {
            F[m + 1] = (static_cast<double>(2 * m + 1) * F[m] - expT) / (2.0 * T);
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Boys dispatch: uses Chebyshev LUT when available, falls back to
// series+recursion otherwise.
// ---------------------------------------------------------------------------
__device__ inline void boys_function_fast(double T, int max_order, double* F) {
#ifdef MAKO_HAS_BOYS_LUT
    // Chebyshev LUT: constant-time per m-value via 7 FMA ops.
    // Runtime dispatch on max_order (template MMAX must be a compile-time
    // constant, so we use a switch over the supported range).
    switch (max_order) {
        case  0: boys_fm_lut< 0>(T, F); return;
        case  1: boys_fm_lut< 1>(T, F); return;
        case  2: boys_fm_lut< 2>(T, F); return;
        case  3: boys_fm_lut< 3>(T, F); return;
        case  4: boys_fm_lut< 4>(T, F); return;
        case  5: boys_fm_lut< 5>(T, F); return;
        case  6: boys_fm_lut< 6>(T, F); return;
        case  7: boys_fm_lut< 7>(T, F); return;
        case  8: boys_fm_lut< 8>(T, F); return;
        case  9: boys_fm_lut< 9>(T, F); return;
        case 10: boys_fm_lut<10>(T, F); return;
        case 11: boys_fm_lut<11>(T, F); return;
        case 12: boys_fm_lut<12>(T, F); return;
        case 13: boys_fm_lut<13>(T, F); return;
        case 14: boys_fm_lut<14>(T, F); return;
        case 15: boys_fm_lut<15>(T, F); return;
        case 16: boys_fm_lut<16>(T, F); return;
        default: break;  // fall through to series for max_order > 16
    }
#endif
    boys_function(T, max_order, F);
}

// ---------------------------------------------------------------------------
// Cartesian component indexing.
//   ncart(l) = (l+1)*(l+2)/2 components for angular momentum l.
//   The components are ordered as in ASUKA: lexicographic with decreasing lx.
//   cart_index(lx, ly, lz) maps to a linear index.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int ncart(int l) {
    return (l + 1) * (l + 2) / 2;
}

// Linear index for Cartesian component (lx, ly, lz) with lx+ly+lz = l.
// Ordered: (l,0,0), (l-1,1,0), (l-1,0,1), (l-2,2,0), ...
__device__ __forceinline__ int cart_index(int lx, int ly, int /*lz*/) {
    int l = lx + ly;  // lz is determined
    // Position within shell of angular momentum lx+ly+lz
    // For a given lx, the index offset is Sum_{lx'=l..lx+1} (l-lx'+1) = ...
    // Simpler: index = (l-lx)*(l-lx+1)/2 + ly  (where l = total AM - lz... no)
    // Actually for ordering (lx desc, ly desc):
    int ll = lx + ly;  // total for this x-level... hmm
    // Let me use the standard ordering from ASUKA:
    // For total l, the index of (lx, ly, lz) is:
    //   idx = l*(l+1)/2 - lx*(lx+1)/2 + ly  ... not quite
    // Actually, let me just use the simple formula:
    // For the standard "downward lx" ordering:
    int total_l = lx + ly;  // = l - lz
    (void)total_l;
    // The correct formula for the ASUKA ordering is:
    // For (lx, ly, lz) with lx+ly+lz = l, the position is:
    //   (l - lx) * (l - lx + 1) / 2 + ly
    int remaining = ly + (lx + ly);  // This is l itself... hmm
    // Let me hardcode for l <= 4 using a lookup approach in the caller
    // For now, use a safe but slower computation
    int l_total = lx + ly;  // + lz
    // Position: count all components with lx' > lx, then within same lx, count ly' > ly
    // Components with higher lx: Sum_{lx'=l..lx+1} (l - lx' + 1) = Sum_{k=0}^{l-lx-1} (k+1) = (l-lx)(l-lx+1)/2... no
    // The standard formula: for (lx, ly, lz) in the ordering
    //   [(l,0,0), (l-1,1,0), (l-1,0,1), (l-2,2,0), (l-2,1,1), (l-2,0,2), ...]
    // the index is: Sum_{lx'=l}^{lx+1} (l - lx' + 1) + (l - lx - ly)
    // Wait, that's not right either. Let me just use:
    // idx = l*(l+1)/2 - lx*(lx+1)/2 + ... argh.
    // The simplest correct formula for "reverse-lex on lx, then lex on ly" is:
    //   idx = (l - lx) * (l - lx + 1) / 2 + ly   WHERE l = lx+ly+lz
    // But I don't have l here (only lx+ly).  The caller needs to pass total l.
    // For now this function is unused -- the generic kernel uses explicit loops.
    return 0;  // placeholder
}

// ---------------------------------------------------------------------------
// R-integral helpers.
//
// We use two arrays:
//   Rt[t][m]  — 1D R-integrals R_{t,0,0}^{(m)} after x-recurrence
//   Rtuv[t][u][v] — final R_{t,u,v}^{(0)} after y,z recurrence
//
// This avoids the 4D R array that would be (L+1)^4 = 6561 for dddd.
// Instead we need (L+1)^2 + (L+1)^3 = 81 + 729 = 810 doubles for L=8.
// ---------------------------------------------------------------------------
static constexpr int R_MAX_LP1 = 9;  // L+1 for L=8
static constexpr int RT_SIZE = R_MAX_LP1 * R_MAX_LP1;               // 81
static constexpr int RTUV_SIZE = R_MAX_LP1 * R_MAX_LP1 * R_MAX_LP1; // 729
// Smaller variants for lower L to reduce per-thread stack usage:
static constexpr int RTUV_SIZE_L4 = 5 * 5 * 5;   // 125, for L<=4
static constexpr int RT_SIZE_L4 = 5 * 5;           // 25

// Rt[t * Lp1 + m]
__device__ __forceinline__ int Rt_idx(int t, int m, int Lp1) {
    return t * Lp1 + m;
}
// Rtu[t * Lp1 + u] for a given m (2D working array during u-recurrence)
__device__ __forceinline__ int Rtu_idx(int t, int u, int Lp1) {
    return t * Lp1 + u;
}
// Rtuv[((t) * Lp1 + u) * Lp1 + v]  — final output
__device__ __forceinline__ int Rtuv_idx(int t, int u, int v, int Lp1) {
    return (t * Lp1 + u) * Lp1 + v;
}

// Compute all R_{tuv}^{(0)} needed for ERI evaluation.
// Output: Rtuv[(t*Lp1+u)*Lp1+v] for t+u+v <= L.
// Working storage: Rt[t*Lp1+m], Rtu[t*Lp1+u] (caller-allocated).
__device__ inline void compute_R_integrals(
    double XPQ, double YPQ, double ZPQ,
    double rho, double prefactor,
    const double* Fm, int L,
    double* Rtuv, double* Rt, int Lp1)
{
    // Step 1: R_{0,0,0}^{(m)} = prefactor * (-2*rho)^m * F_m(T)
    double neg2rho_pow = 1.0;
    for (int m = 0; m <= L; ++m) {
        Rt[Rt_idx(0, m, Lp1)] = prefactor * neg2rho_pow * Fm[m];
        neg2rho_pow *= (-2.0 * rho);
    }

    // Step 2: x-recurrence to get R_{t,0,0}^{(m)} for all t,m with t+m<=L
    for (int t = 0; t < L; ++t) {
        for (int m = 0; m <= L - t - 1; ++m) {
            double val = XPQ * Rt[Rt_idx(t, m + 1, Lp1)];
            if (t > 0)
                val += static_cast<double>(t) * Rt[Rt_idx(t - 1, m + 1, Lp1)];
            Rt[Rt_idx(t + 1, m, Lp1)] = val;
        }
    }

    // Step 3: y-recurrence and z-recurrence to get R_{t,u,v}^{(0)}
    // For each v-target, we do y-recurrence from Rt[t,m] -> Rtu[t,u] at various m,
    // then z-recurrence from Rtu -> Rtuv.
    // But for efficiency, let's build R_{t,u,0}^{(m)} for all t,u,m first,
    // then z-recurrence only for m=0.

    // Actually, the simplest correct approach: for each (t), build R_{t,u,0}^{(m)}
    // from Rt using y-recurrence, then for each (t,u) build R_{t,u,v}^{(0)} from
    // R_{t,u,0}^{(m)} using z-recurrence.

    // Working array for R_{t,u,0}^{(m)}: need [t][u][m] with t+u+m<=L.
    // Use Rtuv as scratch: Rtuv[(t*Lp1+u)*Lp1+m] = R_{t,u,0}^{(m)}
    // This reuses the output array as intermediate storage.

    // Initialize: R_{t,0,0}^{(m)} = Rt[t,m]
    for (int t = 0; t <= L; ++t)
        for (int m = 0; m <= L - t; ++m)
            Rtuv[Rtuv_idx(t, 0, m, Lp1)] = Rt[Rt_idx(t, m, Lp1)];

    // y-recurrence: R_{t,u+1,0}^{(m)} = u * R_{t,u-1,0}^{(m+1)} + YPQ * R_{t,u,0}^{(m+1)}
    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u < L - t; ++u) {
            for (int m = 0; m <= L - t - u - 1; ++m) {
                double val = YPQ * Rtuv[Rtuv_idx(t, u, m + 1, Lp1)];
                if (u > 0)
                    val += static_cast<double>(u) * Rtuv[Rtuv_idx(t, u - 1, m + 1, Lp1)];
                Rtuv[Rtuv_idx(t, u + 1, m, Lp1)] = val;
            }
        }
    }

    // Now Rtuv[t,u,m] = R_{t,u,0}^{(m)} for t+u+m<=L.
    // z-recurrence: for each (t,u), build R_{t,u,v}^{(0)} for v=0..L-t-u.
    //
    // Use three local buffers for a clean double-buffer approach:
    //   buf_vm1[m] = R_{t,u,v-1}^{(m)}  ("two steps ago")
    //   buf_v[m]   = R_{t,u,v}^{(m)}    ("previous step")
    //   buf_vp1[m] = R_{t,u,v+1}^{(m)}  ("current, being computed")
    //
    // After computing v+1, rotate: vm1 <- v, v <- vp1.

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u <= L - t; ++u) {
            int vmax = L - t - u;
            // v=0 is already in place (m=0 slot holds R_{t,u,0}^{(0)})
            if (vmax == 0) continue;

            double buf_vm1[R_MAX_LP1];
            double buf_v[R_MAX_LP1];
            double buf_vp1[R_MAX_LP1];

            for (int m = 0; m <= vmax; ++m)
                buf_v[m] = Rtuv[Rtuv_idx(t, u, m, Lp1)];

            for (int v = 0; v < vmax; ++v) {
                for (int m = 0; m <= vmax - v - 1; ++m) {
                    double val = ZPQ * buf_v[m + 1];
                    if (v > 0)
                        val += static_cast<double>(v) * buf_vm1[m + 1];
                    buf_vp1[m] = val;
                }
                Rtuv[Rtuv_idx(t, u, v + 1, Lp1)] = buf_vp1[0];
                for (int m = 0; m <= vmax - v - 1; ++m) {
                    buf_vm1[m] = buf_v[m];
                    buf_v[m] = buf_vp1[m];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1D E-coefficient computation.
// E_t^{i,j}(X_PA, X_PB, p) for t = 0..i+j, using the McMurchie-Davidson
// recurrence.  Output stored in E[i+1][j+1][i+j+1] indexed as E[ii][jj][t].
// ---------------------------------------------------------------------------
__device__ inline void compute_E_1d(
    int la, int lb,
    double X_PA, double X_PB, double inv_2p,
    double* E, int stride_i, int stride_j)
{
    // E[ii * stride_i + jj * stride_j + t]
    // Initialize all to zero
    int max_t = la + lb;
    for (int ii = 0; ii <= la; ++ii)
        for (int jj = 0; jj <= lb; ++jj)
            for (int t = 0; t <= max_t; ++t)
                E[ii * stride_i + jj * stride_j + t] = 0.0;
    E[0] = 1.0;  // E_0^{0,0}

    // Build up in interleaved order: for each target (ii, jj), first
    // increment i from (ii-1, jj), then increment j from (ii, jj-1).
    // This ensures all dependencies are satisfied.

    // First build the i-column with j=0:
    // E_t^{i+1,0} = inv_2p * E_{t-1}^{i,0} + X_PA * E_t^{i,0} + (t+1) * E_{t+1}^{i,0}
    for (int ii = 0; ii < la; ++ii) {
        for (int t = 0; t <= ii + 1; ++t) {
            double val = X_PA * E[ii * stride_i + t];
            if (t > 0)
                val += inv_2p * E[ii * stride_i + (t - 1)];
            if (t + 1 <= ii)
                val += static_cast<double>(t + 1) * E[ii * stride_i + (t + 1)];
            E[(ii + 1) * stride_i + t] = val;
        }
    }

    // Now for each i, build up j:
    // E_t^{i,j+1} = inv_2p * E_{t-1}^{i,j} + X_PB * E_t^{i,j} + (t+1) * E_{t+1}^{i,j}
    for (int ii = 0; ii <= la; ++ii) {
        for (int jj = 0; jj < lb; ++jj) {
            for (int t = 0; t <= ii + jj + 1; ++t) {
                double val = X_PB * E[ii * stride_i + jj * stride_j + t];
                if (t > 0)
                    val += inv_2p * E[ii * stride_i + jj * stride_j + (t - 1)];
                if (t + 1 <= ii + jj)
                    val += static_cast<double>(t + 1) * E[ii * stride_i + jj * stride_j + (t + 1)];
                E[ii * stride_i + (jj + 1) * stride_j + t] = val;
            }
        }
    }
}

// Templated version for FP32-native computation (TF32 tensor core path)
template <typename T>
__device__ inline void compute_E_1d_t(
    int la, int lb,
    T X_PA, T X_PB, T inv_2p,
    T* E, int stride_i, int stride_j)
{
    int max_t = la + lb;
    for (int ii = 0; ii <= la; ++ii)
        for (int jj = 0; jj <= lb; ++jj)
            for (int t = 0; t <= max_t; ++t)
                E[ii * stride_i + jj * stride_j + t] = T(0);
    E[0] = T(1);

    for (int ii = 0; ii < la; ++ii) {
        for (int t = 0; t <= ii + 1; ++t) {
            T val = X_PA * E[ii * stride_i + t];
            if (t > 0)     val += inv_2p * E[ii * stride_i + (t - 1)];
            if (t + 1 <= ii) val += T(t + 1) * E[ii * stride_i + (t + 1)];
            E[(ii + 1) * stride_i + t] = val;
        }
    }

    for (int ii = 0; ii <= la; ++ii) {
        for (int jj = 0; jj < lb; ++jj) {
            for (int t = 0; t <= ii + jj + 1; ++t) {
                T val = X_PB * E[ii * stride_i + jj * stride_j + t];
                if (t > 0)          val += inv_2p * E[ii * stride_i + jj * stride_j + (t - 1)];
                if (t + 1 <= ii + jj) val += T(t + 1) * E[ii * stride_i + jj * stride_j + (t + 1)];
                E[ii * stride_i + (jj + 1) * stride_j + t] = val;
            }
        }
    }
}

}  // namespace mako
