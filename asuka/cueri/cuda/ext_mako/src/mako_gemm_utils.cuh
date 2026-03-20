#pragma once
// Mako GEMM kernel utilities.
//
// Provides cooperative device functions for shared-memory ERI evaluation:
//   - Hermite index enumeration and linear indexing
//   - Cartesian exponent enumeration
//   - Small shared-memory GEMM (optimized for bank conflicts + no int div)
//   - Cooperative R-integral computation
//   - Shared-memory zeroing

#include <cstdint>
#include "mako_hermite.cuh"

namespace mako {

// ---------------------------------------------------------------------------
// Hermite function count: C(l+3, 3) = (l+1)(l+2)(l+3)/6
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int n_hermite_count(int l) {
    return (l + 1) * (l + 2) * (l + 3) / 6;
}

// ---------------------------------------------------------------------------
// Enumerate all (t,u,v) with t+u+v <= L into shared-memory arrays.
// ---------------------------------------------------------------------------
__device__ inline int enumerate_hermite_indices(
    int L, int* __restrict__ ht, int* __restrict__ hu, int* __restrict__ hv)
{
    int idx = 0;
    for (int n = 0; n <= L; ++n)
        for (int t = n; t >= 0; --t)
            for (int u = n - t; u >= 0; --u) {
                ht[idx] = t;
                hu[idx] = u;
                hv[idx] = n - t - u;
                idx++;
            }
    return idx;
}

// ---------------------------------------------------------------------------
// Enumerate Cartesian exponents for angular momentum l.
// ---------------------------------------------------------------------------
__device__ inline int enumerate_cart_exponents(
    int l, int* __restrict__ cx, int* __restrict__ cy, int* __restrict__ cz)
{
    int idx = 0;
    for (int ix = l; ix >= 0; --ix)
        for (int iy = l - ix; iy >= 0; --iy) {
            cx[idx] = ix;
            cy[idx] = iy;
            cz[idx] = l - ix - iy;
            idx++;
        }
    return idx;
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory GEMM: C[M×N] = A[M×K] × B[K×N]
// Row-major storage.  Uses row-based thread assignment to avoid
// integer division in the hot inner loop.
//
// ldb: leading dimension (stride) of B.  Pass ldb=N for dense layout,
//      or ldb=N+1 for padded layout that avoids shared memory bank conflicts.
// ---------------------------------------------------------------------------
template <typename T = double>
__device__ inline void smem_gemm_AB(
    T* __restrict__ C,
    const T* __restrict__ A,
    const T* __restrict__ B,
    int M, int N, int K,
    int tid, int nthreads,
    int ldb = 0)
{
    if (ldb <= 0) ldb = N;
    for (int row = tid; row < M; row += nthreads) {
        const T* A_row = A + row * K;
        T* C_row = C + row * N;
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            for (int k = 0; k < K; ++k)
                sum += A_row[k] * B[k * ldb + j];
            C_row[j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory GEMM with accumulation:
//   C[M×N] += A[M×K] × B[K×N]
// Row-major storage, non-transposed, accumulates into C.
//
// ldb: leading dimension (stride) of B (see smem_gemm_AB).
// ---------------------------------------------------------------------------
template <typename T = double>
__device__ inline void smem_gemm_AB_accum(
    T* __restrict__ C,
    const T* __restrict__ A,
    const T* __restrict__ B,
    int M, int N, int K,
    int tid, int nthreads,
    int ldb = 0)
{
    if (ldb <= 0) ldb = N;
    for (int row = tid; row < M; row += nthreads) {
        const T* A_row = A + row * K;
        T* C_row = C + row * N;
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            for (int k = 0; k < K; ++k)
                sum += A_row[k] * B[k * ldb + j];
            C_row[j] += sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Element-distributed GEMM with K-unrolled accumulation (Gap 3a):
//   C[M×N] += A[M×K] × B[K×N]
//
// Unlike the row-based version, each thread owns one (i,j) output element,
// achieving full thread utilization when M*N > nthreads.  Falls back to
// the row-based version for tiny matrices where overhead isn't justified.
//
// ldb: leading dimension (stride) of B.
// ---------------------------------------------------------------------------
template <typename T = double>
__device__ inline void smem_gemm_AB_accum_v2(
    T* __restrict__ C,
    const T* __restrict__ A,
    const T* __restrict__ B,
    int M, int N, int K,
    int tid, int nthreads,
    int ldb = 0)
{
    if (ldb <= 0) ldb = N;
    // Element-distributed path: each thread owns one (i,j) output element.
    // Beneficial when M is small relative to nthreads (many idle threads in
    // row-based).  But idx/N and idx%N are expensive for non-power-of-2 N,
    // so only use when M < nthreads/4 (row-based would waste 75%+ threads).
    if (M < nthreads / 4 && M * N > nthreads) {
        for (int idx = tid; idx < M * N; idx += nthreads) {
            const int i = idx / N, j = idx % N;
            const T* A_row = A + i * K;
            T sum = T(0);
            #pragma unroll 4
            for (int k = 0; k < K; ++k)
                sum += A_row[k] * B[k * ldb + j];
            C[i * N + j] += sum;
        }
    } else {
        // Row-based: each thread owns one or more rows, no integer division.
        for (int row = tid; row < M; row += nthreads) {
            const T* A_row = A + row * K;
            T* C_row = C + row * N;
            for (int j = 0; j < N; ++j) {
                T sum = T(0);
                for (int k = 0; k < K; ++k)
                    sum += A_row[k] * B[k * ldb + j];
                C_row[j] += sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory GEMM with accumulation:
//   C[M×N] += A[M×K] × B[N×K]^T
// A row-major [M×K], B row-major [N×K] (transposed in multiply).
// ---------------------------------------------------------------------------
template <typename T = double>
__device__ inline void smem_gemm_ABt_accum(
    T* __restrict__ C,
    const T* __restrict__ A,
    const T* __restrict__ B,
    int M, int N, int K,
    int tid, int nthreads)
{
    for (int row = tid; row < M; row += nthreads) {
        const T* A_row = A + row * K;
        T* C_row = C + row * N;
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            const T* B_row = B + j * K;
            for (int k = 0; k < K; ++k)
                sum += A_row[k] * B_row[k];
            C_row[j] += sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory zero-fill.
// ---------------------------------------------------------------------------
template <typename T = double>
__device__ __forceinline__ void smem_zero(
    T* __restrict__ buf, int count, int tid, int nthreads)
{
    for (int i = tid; i < count; i += nthreads)
        buf[i] = T(0);
}

// ---------------------------------------------------------------------------
// Cooperative R-integral computation.
//
// Distributes the Boys function + R-integral recurrence across all
// threads in the block.  For L=8, the total work is ~1000 FLOPs;
// parallelizing the per-level iterations speeds up the sequential part.
//
// Phase A is split:
//   1. Thread 0 computes Boys function F_m(T) → smem_Fm
//   2. __syncthreads()
//   3. Thread 0 initializes R_{0,0,0}^{(m)} in smem_Rt
//   4. All threads cooperatively run x-recurrence (parallel over m)
//   5. __syncthreads()
//   6. All threads cooperatively run y-recurrence
//   7. __syncthreads()
//   8. All threads cooperatively run z-recurrence → smem_Rtuv
// ---------------------------------------------------------------------------
__device__ inline void cooperative_compute_R_integrals(
    double XPQ, double YPQ, double ZPQ,
    double rho, double prefactor,
    int L,
    double* __restrict__ smem_Fm,
    double* __restrict__ smem_Rtuv,
    double* __restrict__ smem_Rt,
    int Lp1,
    int tid, int nthreads)
{
    // Step 1: Boys function (single thread — numerically fragile)
    if (tid == 0) {
        double T = rho * (XPQ * XPQ + YPQ * YPQ + ZPQ * ZPQ);
        boys_function_fast(T, L, smem_Fm);

        // Step 2: Initialize R_{0,0,0}^{(m)} = prefactor * (-2*rho)^m * F_m(T)
        double neg2rho_pow = 1.0;
        for (int m = 0; m <= L; ++m) {
            smem_Rt[0 * Lp1 + m] = prefactor * neg2rho_pow * smem_Fm[m];
            neg2rho_pow *= (-2.0 * rho);
        }
    }
    __syncthreads();

    // Step 3: x-recurrence — R_{t+1,0,0}^{(m)} for each t level.
    // Each level depends on the previous, but within a level the m values
    // are independent.  Parallelize over m.
    for (int t = 0; t < L; ++t) {
        int n_m = L - t;  // number of m values at this level
        for (int m = tid; m < n_m; m += nthreads) {
            double val = XPQ * smem_Rt[t * Lp1 + (m + 1)];
            if (t > 0)
                val += double(t) * smem_Rt[(t - 1) * Lp1 + (m + 1)];
            smem_Rt[(t + 1) * Lp1 + m] = val;
        }
        __syncthreads();
    }

    // Step 4: Copy Rt to Rtuv for y-recurrence.
    // Rtuv[(t*Lp1+u)*Lp1+m] = R_{t,u,0}^{(m)}
    // Initialize u=0 slice from Rt.
    for (int idx = tid; idx < Lp1 * Lp1; idx += nthreads) {
        int t = idx / Lp1, m = idx % Lp1;
        if (t + m <= L)
            smem_Rtuv[(t * Lp1 + 0) * Lp1 + m] = smem_Rt[t * Lp1 + m];
        else
            smem_Rtuv[(t * Lp1 + 0) * Lp1 + m] = 0.0;
    }
    __syncthreads();

    // Step 5: y-recurrence — sequential in u, parallel over (t, m).
    for (int u = 0; u < L; ++u) {
        for (int t = 0; t <= L - u - 1; ++t) {
            for (int m = tid; m <= L - t - u - 1; m += nthreads) {
                double val = YPQ * smem_Rtuv[(t * Lp1 + u) * Lp1 + (m + 1)];
                if (u > 0)
                    val += double(u) * smem_Rtuv[(t * Lp1 + (u - 1)) * Lp1 + (m + 1)];
                smem_Rtuv[(t * Lp1 + (u + 1)) * Lp1 + m] = val;
            }
        }
        __syncthreads();
    }

    // Step 6: z-recurrence.
    // Rtuv[(t*Lp1+u)*Lp1+v] starts as R_{t,u,0}^{(m)} with v=m.
    // We overwrite in-place: v slot replaces m slot.
    // Sequential in v, parallel over (t, u).
    // After this, Rtuv[(t*Lp1+u)*Lp1+v] = R_{t,u,v}^{(0)}.

    // For each (t,u) with t+u <= L, we need to transform the m-dimension
    // into the v-dimension.  This is done per-(t,u) pair independently.
    for (int idx = tid; idx < Lp1 * Lp1; idx += nthreads) {
        int t = idx / Lp1, u = idx % Lp1;
        if (t + u > L) continue;
        int vmax = L - t - u;
        if (vmax == 0) continue;

        // In-place z-recurrence for this (t, u):
        // R_{t,u,v+1}^{(m)} = ZPQ * R_{t,u,v}^{(m+1)} + v * R_{t,u,v-1}^{(m+1)}
        // We use the fact that Rtuv[(t*Lp1+u)*Lp1+m] currently holds
        // R_{t,u,0}^{(m)}.  We need to produce R_{t,u,v}^{(0)} for v=0..vmax.

        double buf_vm1[17];  // kMaxTotalL+1 = 17
        double buf_v[17];
        double buf_vp1[17];

        // Load v=0: buf_v[m] = R_{t,u,0}^{(m)} for m=0..vmax
        for (int m = 0; m <= vmax; ++m)
            buf_v[m] = smem_Rtuv[(t * Lp1 + u) * Lp1 + m];

        for (int v = 0; v < vmax; ++v) {
            for (int m = 0; m <= vmax - v - 1; ++m) {
                double val = ZPQ * buf_v[m + 1];
                if (v > 0)
                    val += double(v) * buf_vm1[m + 1];
                buf_vp1[m] = val;
            }
            // Store R_{t,u,v+1}^{(0)}
            smem_Rtuv[(t * Lp1 + u) * Lp1 + (v + 1)] = buf_vp1[0];
            // Rotate buffers
            for (int m = 0; m <= vmax - v - 1; ++m) {
                buf_vm1[m] = buf_v[m];
                buf_v[m] = buf_vp1[m];
            }
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Cooperative 1D E-coefficient computation.
// Distributes the 6 direction×pair arrays across threads.
// ---------------------------------------------------------------------------
__device__ inline void cooperative_compute_E_coefficients(
    int la, int lb, int lc, int ld,
    double XPA, double YPA, double ZPA, double XPB, double YPB, double ZPB, double inv_2p,
    double XQC, double YQC, double ZQC, double XQD, double YQD, double ZQD, double inv_2q,
    double* __restrict__ smem_1d_bra, int bra_1d_size,
    double* __restrict__ smem_1d_ket, int ket_1d_size,
    int e_bra_stride_i, int e_bra_stride_j,
    int e_ket_stride_i, int e_ket_stride_j,
    int tid, int nthreads)
{
    double* E_bra_x = smem_1d_bra;
    double* E_bra_y = smem_1d_bra + bra_1d_size;
    double* E_bra_z = smem_1d_bra + 2 * bra_1d_size;
    double* E_ket_x = smem_1d_ket;
    double* E_ket_y = smem_1d_ket + ket_1d_size;
    double* E_ket_z = smem_1d_ket + 2 * ket_1d_size;

    // Distribute 6 arrays across threads.  Each compute_E_1d is fast
    // (~50 FLOPs for d-shells), so 6 threads handle all arrays while
    // the remaining threads proceed to cooperative R-integral computation.
    if (tid == 0) compute_E_1d(la, lb, XPA, XPB, inv_2p, E_bra_x, e_bra_stride_i, e_bra_stride_j);
    if (tid == 1) compute_E_1d(la, lb, YPA, YPB, inv_2p, E_bra_y, e_bra_stride_i, e_bra_stride_j);
    if (tid == 2) compute_E_1d(la, lb, ZPA, ZPB, inv_2p, E_bra_z, e_bra_stride_i, e_bra_stride_j);
    if (tid == 3) compute_E_1d(lc, ld, XQC, XQD, inv_2q, E_ket_x, e_ket_stride_i, e_ket_stride_j);
    if (tid == 4) compute_E_1d(lc, ld, YQC, YQD, inv_2q, E_ket_y, e_ket_stride_i, e_ket_stride_j);
    if (tid == 5) compute_E_1d(lc, ld, ZQC, ZQD, inv_2q, E_ket_z, e_ket_stride_i, e_ket_stride_j);
    // All threads must reach this barrier before entering
    // cooperative_compute_R_integrals (which has internal syncs).
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Compute total dynamic shared memory bytes for the GEMM ERI kernel.
//
// The [p|q] matrix (smem_R) uses padded stride to avoid shared memory bank
// conflicts.  Only even nhcd values are padded (+1 → odd): even strides cause
// severe bank conflicts (gcd(stride*2, 32) >= 4).  Odd strides already have
// gcd(stride*2, 32) <= 2, so padding them would actually worsen conflicts.
// ---------------------------------------------------------------------------
__host__ inline int gemm_smem_bytes(int la, int lb, int lc, int ld) {
    const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
    const int nab = na * nb, ncd = nc * nd;
    const int lab = la + lb, lcd = lc + ld, L = lab + lcd;
    const int Lp1 = L + 1;
    const int nhab = n_hermite_count(lab);
    const int nhcd = n_hermite_count(lcd);
    const int R_stride = (nhcd % 2 == 0) ? nhcd + 1 : nhcd;  // pad even strides only

    int E_size  = (nab * nhab > ncd * nhcd) ? nab * nhab : ncd * nhcd;
    int R_size  = nhab * R_stride;  // padded R layout
    int M_size  = nab * nhcd;
    int Acc_sz  = nab * ncd;
    int Rtuv_sz = Lp1 * Lp1 * Lp1;
    int bra_1d  = 3 * (la + 1) * (lb + 1) * (lab + 1);
    int ket_1d  = 3 * (lc + 1) * (ld + 1) * (lcd + 1);
    int Fm_sz   = L + 1;
    int Rt_sz   = Lp1 * Lp1;

    int n_doubles = E_size + R_size + M_size + Acc_sz + Rtuv_sz
                  + bra_1d + ket_1d + Fm_sz + Rt_sz;

    int n_ints = 3 * nhab + 3 * nhcd + 3 * (na + nb + nc + nd);

    int byte_doubles = n_doubles * static_cast<int>(sizeof(double));
    int byte_ints = n_ints * static_cast<int>(sizeof(int));
    int pad = (byte_doubles % 8 == 0) ? 0 : (8 - byte_doubles % 8);

    return byte_doubles + pad + byte_ints;
}

// ---------------------------------------------------------------------------
// Templated R-integrals: Boys function in FP64, recurrence in T (float/double)
// ---------------------------------------------------------------------------
template <typename T>
__device__ inline void cooperative_compute_R_integrals_t(
    double XPQ, double YPQ, double ZPQ,
    double rho, double prefactor,
    int L,
    double* __restrict__ smem_Fm,   // FP64 Boys values (temporary)
    T* __restrict__ smem_Rtuv,      // R-integral output in T
    T* __restrict__ smem_Rt,        // scratch for x-recurrence in T
    int Lp1,
    int tid, int nthreads)
{
    // Step 1: Boys function in FP64 (numerically fragile)
    if (tid == 0) {
        double T_val = rho * (XPQ * XPQ + YPQ * YPQ + ZPQ * ZPQ);
        boys_function_fast(T_val, L, smem_Fm);

        // Step 2: Initialize R_{0,0,0}^{(m)} = prefactor * (-2*rho)^m * F_m(T) → convert to T
        double neg2rho_pow = 1.0;
        for (int m = 0; m <= L; ++m) {
            smem_Rt[0 * Lp1 + m] = static_cast<T>(prefactor * neg2rho_pow * smem_Fm[m]);
            neg2rho_pow *= (-2.0 * rho);
        }
    }
    __syncthreads();

    // Steps 3-6: recurrences in T (float for TF32 path, double for FP64 path)
    T XPQ_t = static_cast<T>(XPQ), YPQ_t = static_cast<T>(YPQ), ZPQ_t = static_cast<T>(ZPQ);

    // x-recurrence
    for (int t = 0; t < L; ++t) {
        int n_m = L - t;
        for (int m = tid; m < n_m; m += nthreads) {
            T val = XPQ_t * smem_Rt[t * Lp1 + (m + 1)];
            if (t > 0) val += T(t) * smem_Rt[(t - 1) * Lp1 + (m + 1)];
            smem_Rt[(t + 1) * Lp1 + m] = val;
        }
        __syncthreads();
    }

    // Copy Rt → Rtuv (u=0 slice)
    for (int idx = tid; idx < Lp1 * Lp1; idx += nthreads) {
        int t = idx / Lp1, m = idx % Lp1;
        smem_Rtuv[(t * Lp1 + 0) * Lp1 + m] = (t + m <= L) ? smem_Rt[t * Lp1 + m] : T(0);
    }
    __syncthreads();

    // y-recurrence
    for (int u = 0; u < L; ++u) {
        for (int t = 0; t <= L - u - 1; ++t) {
            for (int m = tid; m <= L - t - u - 1; m += nthreads) {
                T val = YPQ_t * smem_Rtuv[(t * Lp1 + u) * Lp1 + (m + 1)];
                if (u > 0) val += T(u) * smem_Rtuv[(t * Lp1 + (u - 1)) * Lp1 + (m + 1)];
                smem_Rtuv[(t * Lp1 + (u + 1)) * Lp1 + m] = val;
            }
        }
        __syncthreads();
    }

    // z-recurrence (per-(t,u) pair, independent)
    for (int idx = tid; idx < Lp1 * Lp1; idx += nthreads) {
        int t = idx / Lp1, u = idx % Lp1;
        if (t + u > L) continue;
        int vmax = L - t - u;
        if (vmax == 0) continue;

        T buf_vm1[17], buf_v[17], buf_vp1[17];
        for (int m = 0; m <= vmax; ++m)
            buf_v[m] = smem_Rtuv[(t * Lp1 + u) * Lp1 + m];

        for (int v = 0; v < vmax; ++v) {
            for (int m = 0; m <= vmax - v - 1; ++m) {
                T val = ZPQ_t * buf_v[m + 1];
                if (v > 0) val += T(v) * buf_vm1[m + 1];
                buf_vp1[m] = val;
            }
            smem_Rtuv[(t * Lp1 + u) * Lp1 + (v + 1)] = buf_vp1[0];
            for (int m = 0; m <= vmax - v - 1; ++m) {
                buf_vm1[m] = buf_v[m];
                buf_v[m] = buf_vp1[m];
            }
        }
    }
    __syncthreads();
}

}  // namespace mako
