// Analytic 4-center ERI nuclear derivatives (contracted) using GPU generic Rys.
//
// This implements a *contracted* derivative API: it contracts d(μν|λσ)/dR against a
// user-provided adjoint/weight tile bar(μν,λσ) and outputs 12 numbers per task
// (4 centers × 3 Cartesian directions).
//
// The derivative formula used is the standard Cartesian Gaussian center-derivative
// relation (per primitive):
//   d/dA_x (a_x) = 2*alpha * (a_x+1) - a_x * (a_x-1)
// (and analogously for other axes and other centers).
//
// This kernel is reference-oriented. It reuses the existing generic Rys machinery
// and calls shift_from_G for the required +/-1 shifts.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#include "cueri_cuda_kernels_api.h"
#include "cueri_cuda_rys_device.cuh"

namespace {

constexpr double kPi = 3.1415926535897932384626433832795;
// 2*pi^2*sqrt(pi) (avoid non-constexpr sqrt in device compilation)
constexpr double kTwoPiToFiveHalves = 34.98683665524972;

constexpr int kLMax = 5;
constexpr int kLMaxD = kLMax + 1; // derivatives require l+1 intermediate
constexpr int kStrideD = 2 * kLMaxD + 1; // 13
constexpr int kGSizeD = kStrideD * kStrideD; // 169
constexpr int kNcartMax = (kLMax + 1) * (kLMax + 2) / 2; // 21

// Cached bar-tile size (in doubles) in shared memory.
constexpr int kBarCacheMax = 1024;

__device__ __forceinline__ constexpr int ncart(int l) {
    return (l + 1) * (l + 2) / 2;
}

// Binomial coefficients for n<=6.
__device__ __forceinline__ constexpr int binom_d(int n, int k) {
    if (k < 0 || k > n) return 0;
    switch (n) {
        case 0:
            return (k == 0) ? 1 : 0;
        case 1: {
            constexpr int row[] = {1, 1};
            return row[k];
        }
        case 2: {
            constexpr int row[] = {1, 2, 1};
            return row[k];
        }
        case 3: {
            constexpr int row[] = {1, 3, 3, 1};
            return row[k];
        }
        case 4: {
            constexpr int row[] = {1, 4, 6, 4, 1};
            return row[k];
        }
        case 5: {
            constexpr int row[] = {1, 5, 10, 10, 5, 1};
            return row[k];
        }
        case 6: {
            constexpr int row[] = {1, 6, 15, 20, 15, 6, 1};
            return row[k];
        }
        default:
            return 0;
    }
}

__device__ __forceinline__ void fill_cart_comp(int l, int8_t* lx, int8_t* ly, int8_t* lz) {
    // Cartesian component ordering matches the CPU implementation.
    int idx = 0;
    for (int ix = l; ix >= 0; --ix) {
        for (int iy = l - ix; iy >= 0; --iy) {
            const int iz = l - ix - iy;
            lx[idx] = static_cast<int8_t>(ix);
            ly[idx] = static_cast<int8_t>(iy);
            lz[idx] = static_cast<int8_t>(iz);
            ++idx;
        }
    }
}

template <int STRIDE>
__device__ __forceinline__ void compute_G_stride(
    double* G,
    int nmax,
    int mmax,
    double C,
    double Cp,
    double B0,
    double B1,
    double B1p
) {
    // Compute the 2D recurrence table G[a,b] for a<=nmax, b<=mmax.
    G[0 * STRIDE + 0] = 1.0;
    if (nmax >= 1) G[1 * STRIDE + 0] = C;
    if (mmax >= 1) G[0 * STRIDE + 1] = Cp;
    if (nmax >= 1 && mmax >= 1) G[1 * STRIDE + 1] = Cp * G[1 * STRIDE + 0] + B0;

    for (int a = 2; a <= nmax; ++a) {
        G[a * STRIDE + 0] = C * G[(a - 1) * STRIDE + 0] + (a - 1) * B1 * G[(a - 2) * STRIDE + 0];
    }
    for (int b = 2; b <= mmax; ++b) {
        G[0 * STRIDE + b] = Cp * G[0 * STRIDE + (b - 1)] + (b - 1) * B1p * G[0 * STRIDE + (b - 2)];
    }

    for (int a = 2; a <= nmax; ++a) {
        G[a * STRIDE + 1] = Cp * G[a * STRIDE + 0] + a * B0 * G[(a - 1) * STRIDE + 0];
    }
    for (int b = 2; b <= mmax; ++b) {
        G[1 * STRIDE + b] = C * G[0 * STRIDE + b] + b * B0 * G[0 * STRIDE + (b - 1)];
    }

    for (int a = 2; a <= nmax; ++a) {
        for (int b = 2; b <= mmax; ++b) {
            G[a * STRIDE + b] = C * G[(a - 1) * STRIDE + b] + (a - 1) * B1 * G[(a - 2) * STRIDE + b] +
                                b * B0 * G[(a - 1) * STRIDE + (b - 1)];
        }
    }
}

template <int STRIDE>
__device__ __forceinline__ double shift_from_G_stride_d(
    const double* G,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow
) {
    // General shift (supports j,l up to 6).
    double ijkl = 0.0;
    for (int m = 0; m <= l; ++m) {
        double ijm0 = 0.0;
        for (int n = 0; n <= j; ++n) {
            ijm0 += static_cast<double>(binom_d(j, n)) * xij_pow[j - n] * G[(n + i) * STRIDE + (m + k)];
        }
        ijkl += static_cast<double>(binom_d(l, m)) * xkl_pow[l - m] * ijm0;
    }
    return ijkl;
}

__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int N>
__device__ __forceinline__ void warp_reduce_sum_arr(double (&arr)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        arr[i] = warp_reduce_sum(arr[i]);
    }
}

__device__ __forceinline__ void rys_roots_weights_dispatch(int nroots, double T, double* r, double* w) {
    // Dispatch to `cueri_rys::rys_roots_weights<NROOTS>` without templating the entire kernel.
    // This keeps compile time manageable.
    switch (nroots) {
        case 1: cueri_rys::rys_roots_weights<1>(T, r, w); break;
        case 2: cueri_rys::rys_roots_weights<2>(T, r, w); break;
        case 3: cueri_rys::rys_roots_weights<3>(T, r, w); break;
        case 4: cueri_rys::rys_roots_weights<4>(T, r, w); break;
        case 5: cueri_rys::rys_roots_weights<5>(T, r, w); break;
        case 6: cueri_rys::rys_roots_weights<6>(T, r, w); break;
        case 7: cueri_rys::rys_roots_weights<7>(T, r, w); break;
        case 8: cueri_rys::rys_roots_weights<8>(T, r, w); break;
        case 9: cueri_rys::rys_roots_weights<9>(T, r, w); break;
        case 10: cueri_rys::rys_roots_weights<10>(T, r, w); break;
        case 11: cueri_rys::rys_roots_weights<11>(T, r, w); break;
        default: break;
    }
}

} // namespace

// ---- Kernel ----

__global__ void KernelERI_RysGenericDerivContracted(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int32_t ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    int ld,
    int nroots,
    const double* bar_tile,
    double* out
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (static_cast<int>(blockDim.x) + 31) >> 5;

    const int t = static_cast<int>(blockIdx.x);
    if (t >= ntasks) return;

    const int spAB = static_cast<int>(task_spAB[t]);
    const int spCD = static_cast<int>(task_spCD[t]);

    const int A = static_cast<int>(sp_A[spAB]);
    const int B = static_cast<int>(sp_B[spAB]);
    const int C = static_cast<int>(sp_A[spCD]);
    const int D = static_cast<int>(sp_B[spCD]);

    const int nA = ncart(la);
    const int nB = ncart(lb);
    const int nC = ncart(lc);
    const int nD = ncart(ld);
    const int nAB = nA * nB;
    const int nCD = nC * nD;
    const int nElem = nAB * nCD;

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Bx = shell_cx[B];
    const double By = shell_cy[B];
    const double Bz = shell_cz[B];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];
    const double Dx = shell_cx[D];
    const double Dy = shell_cy[D];
    const double Dz = shell_cz[D];

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    const double CDx = Cx - Dx;
    const double CDy = Cy - Dy;
    const double CDz = Cz - Dz;

    __shared__ double sh_xij_pow[kLMaxD + 1];
    __shared__ double sh_yij_pow[kLMaxD + 1];
    __shared__ double sh_zij_pow[kLMaxD + 1];
    __shared__ double sh_xkl_pow[kLMaxD + 1];
    __shared__ double sh_ykl_pow[kLMaxD + 1];
    __shared__ double sh_zkl_pow[kLMaxD + 1];

    __shared__ int8_t shA_lx[kNcartMax];
    __shared__ int8_t shA_ly[kNcartMax];
    __shared__ int8_t shA_lz[kNcartMax];
    __shared__ int8_t shB_lx[kNcartMax];
    __shared__ int8_t shB_ly[kNcartMax];
    __shared__ int8_t shB_lz[kNcartMax];
    __shared__ int8_t shC_lx[kNcartMax];
    __shared__ int8_t shC_ly[kNcartMax];
    __shared__ int8_t shC_lz[kNcartMax];
    __shared__ int8_t shD_lx[kNcartMax];
    __shared__ int8_t shD_ly[kNcartMax];
    __shared__ int8_t shD_lz[kNcartMax];

    __shared__ double sh_bar[kBarCacheMax];

    if (tid == 0) {
        sh_xij_pow[0] = sh_yij_pow[0] = sh_zij_pow[0] = 1.0;
        sh_xkl_pow[0] = sh_ykl_pow[0] = sh_zkl_pow[0] = 1.0;
        for (int m = 1; m <= kLMaxD; ++m) {
            sh_xij_pow[m] = sh_xij_pow[m - 1] * ABx;
            sh_yij_pow[m] = sh_yij_pow[m - 1] * ABy;
            sh_zij_pow[m] = sh_zij_pow[m - 1] * ABz;
            sh_xkl_pow[m] = sh_xkl_pow[m - 1] * CDx;
            sh_ykl_pow[m] = sh_ykl_pow[m - 1] * CDy;
            sh_zkl_pow[m] = sh_zkl_pow[m - 1] * CDz;
        }
        fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
        fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
        fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);
        fill_cart_comp(ld, shD_lx, shD_ly, shD_lz);
    }

    if (nElem <= kBarCacheMax) {
        for (int i = tid; i < nElem; i += static_cast<int>(blockDim.x)) {
            sh_bar[i] = bar_tile[static_cast<size_t>(t) * static_cast<size_t>(nElem) + static_cast<size_t>(i)];
        }
    }
    __syncthreads();

    constexpr int kMaxWarps = 8; // threads<=256
    __shared__ double sh_roots[kMaxWarps][11];
    __shared__ double sh_weights[kMaxWarps][11];
    __shared__ double sh_Gx[kMaxWarps][kGSizeD];
    __shared__ double sh_Gy[kMaxWarps][kGSizeD];
    __shared__ double sh_Gz[kMaxWarps][kGSizeD];

    const int baseAB = static_cast<int>(sp_pair_start[spAB]);
    const int baseCD = static_cast<int>(sp_pair_start[spCD]);
    const int nPairAB = static_cast<int>(sp_npair[spAB]);
    const int nPairCD = static_cast<int>(sp_npair[spCD]);
    const int nTot = nPairAB * nPairCD;

    const int sA = static_cast<int>(shell_prim_start[A]);
    const int sB = static_cast<int>(shell_prim_start[B]);
    const int sC = static_cast<int>(shell_prim_start[C]);
    const int sD = static_cast<int>(shell_prim_start[D]);
    const int nprimB = static_cast<int>(shell_nprim[B]);
    const int nprimD = static_cast<int>(shell_nprim[D]);

    const int nmax = la + lb + 1;
    const int mmax = lc + ld + 1;

    double acc[12];
    #pragma unroll
    for (int i = 0; i < 12; ++i) acc[i] = 0.0;

    for (int uPair = warp; uPair < nTot; uPair += nwarps) {
        const int iPairAB = uPair / nPairCD;
        const int iPairCD = uPair - iPairAB * nPairCD;

        const double p = pair_eta[baseAB + iPairAB];
        const double Px = pair_Px[baseAB + iPairAB];
        const double Py = pair_Py[baseAB + iPairAB];
        const double Pz = pair_Pz[baseAB + iPairAB];
        const double cKab = pair_cK[baseAB + iPairAB];

        const int ipA = iPairAB / nprimB;
        const int ipB = iPairAB - ipA * nprimB;
        const double aexp = prim_exp[sA + ipA];
        const double bexp = prim_exp[sB + ipB];

        const double q = pair_eta[baseCD + iPairCD];
        const double Qx = pair_Px[baseCD + iPairCD];
        const double Qy = pair_Py[baseCD + iPairCD];
        const double Qz = pair_Pz[baseCD + iPairCD];
        const double cKcd = pair_cK[baseCD + iPairCD];

        const int ipC = iPairCD / nprimD;
        const int ipD = iPairCD - ipC * nprimD;
        const double cexp = prim_exp[sC + ipC];
        const double dexp = prim_exp[sD + ipD];

        const double denom = p + q;
        const double inv_denom = 1.0 / denom;
        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;
        const double omega = p * q * inv_denom;
        const double T = omega * PQ2;
        const double base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd;

        if (lane == 0) {
            rys_roots_weights_dispatch(nroots, T, sh_roots[warp], sh_weights[warp]);
        }
        __syncwarp();

        #pragma unroll
        for (int u = 0; u < 11; ++u) {
            if (u >= nroots) break;

            if (lane == 0) {
                const double x = sh_roots[warp][u];
                const double B0 = x * 0.5 * inv_denom;
                const double B1 = (1.0 - x) * 0.5 / p + B0;
                const double B1p = (1.0 - x) * 0.5 / q + B0;

                const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
                const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
                const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

                const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
                const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
                const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

                compute_G_stride<kStrideD>(sh_Gx[warp], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
                compute_G_stride<kStrideD>(sh_Gy[warp], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
                compute_G_stride<kStrideD>(sh_Gz[warp], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);
            }
            __syncwarp();

            const double scale = base * sh_weights[warp][u];

            for (int idxElem = lane; idxElem < nElem; idxElem += 32) {
                const double bar = (nElem <= kBarCacheMax)
                                       ? sh_bar[idxElem]
                                       : bar_tile[static_cast<size_t>(t) * static_cast<size_t>(nElem) + static_cast<size_t>(idxElem)];
                if (bar == 0.0) continue;

                const int row = idxElem / nCD;
                const int col = idxElem - row * nCD;
                const int ia = row / nB;
                const int ib = row - ia * nB;
                const int ic = col / nD;
                const int id = col - ic * nD;

                const int iax = static_cast<int>(shA_lx[ia]);
                const int iay = static_cast<int>(shA_ly[ia]);
                const int iaz = static_cast<int>(shA_lz[ia]);
                const int ibx = static_cast<int>(shB_lx[ib]);
                const int iby = static_cast<int>(shB_ly[ib]);
                const int ibz = static_cast<int>(shB_lz[ib]);
                const int icx = static_cast<int>(shC_lx[ic]);
                const int icy = static_cast<int>(shC_ly[ic]);
                const int icz = static_cast<int>(shC_lz[ic]);
                const int idx_d = static_cast<int>(shD_lx[id]);
                const int idy_d = static_cast<int>(shD_ly[id]);
                const int idz_d = static_cast<int>(shD_lz[id]);

                const double Ix = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double Iy = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double Iz = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow);

                const double bar_scale = bar * scale;

                // ---- center A ----
                const double IxA_m = (iax > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax - 1, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxA_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax + 1, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxA = (-static_cast<double>(iax)) * IxA_m + (2.0 * aexp) * IxA_p;
                acc[0] += bar_scale * (dIxA * Iy * Iz);

                const double IyA_m = (iay > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay - 1, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyA_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay + 1, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyA = (-static_cast<double>(iay)) * IyA_m + (2.0 * aexp) * IyA_p;
                acc[1] += bar_scale * (Ix * dIyA * Iz);

                const double IzA_m = (iaz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz - 1, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzA_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz + 1, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzA = (-static_cast<double>(iaz)) * IzA_m + (2.0 * aexp) * IzA_p;
                acc[2] += bar_scale * (Ix * Iy * dIzA);

                // ---- center B ----
                const double IxB_m = (ibx > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx - 1, icx, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxB_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx + 1, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxB = (-static_cast<double>(ibx)) * IxB_m + (2.0 * bexp) * IxB_p;
                acc[3] += bar_scale * (dIxB * Iy * Iz);

                const double IyB_m = (iby > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby - 1, icy, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyB_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby + 1, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyB = (-static_cast<double>(iby)) * IyB_m + (2.0 * bexp) * IyB_p;
                acc[4] += bar_scale * (Ix * dIyB * Iz);

                const double IzB_m = (ibz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz - 1, icz, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzB_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz + 1, icz, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzB = (-static_cast<double>(ibz)) * IzB_m + (2.0 * bexp) * IzB_p;
                acc[5] += bar_scale * (Ix * Iy * dIzB);

                // ---- center C ----
                const double IxC_m = (icx > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx - 1, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxC_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx + 1, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxC = (-static_cast<double>(icx)) * IxC_m + (2.0 * cexp) * IxC_p;
                acc[6] += bar_scale * (dIxC * Iy * Iz);

                const double IyC_m = (icy > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy - 1, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyC_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy + 1, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyC = (-static_cast<double>(icy)) * IyC_m + (2.0 * cexp) * IyC_p;
                acc[7] += bar_scale * (Ix * dIyC * Iz);

                const double IzC_m = (icz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz - 1, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzC_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz + 1, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzC = (-static_cast<double>(icz)) * IzC_m + (2.0 * cexp) * IzC_p;
                acc[8] += bar_scale * (Ix * Iy * dIzC);

                // ---- center D ----
                const double IxD_m = (idx_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d - 1, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxD_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d + 1, sh_xij_pow, sh_xkl_pow);
                const double dIxD = (-static_cast<double>(idx_d)) * IxD_m + (2.0 * dexp) * IxD_p;
                acc[9] += bar_scale * (dIxD * Iy * Iz);

                const double IyD_m = (idy_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d - 1, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyD_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d + 1, sh_yij_pow, sh_ykl_pow);
                const double dIyD = (-static_cast<double>(idy_d)) * IyD_m + (2.0 * dexp) * IyD_p;
                acc[10] += bar_scale * (Ix * dIyD * Iz);

                const double IzD_m = (idz_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d - 1, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzD_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d + 1, sh_zij_pow, sh_zkl_pow);
                const double dIzD = (-static_cast<double>(idz_d)) * IzD_m + (2.0 * dexp) * IzD_p;
                acc[11] += bar_scale * (Ix * Iy * dIzD);
            }
            __syncwarp(); // ensure this warp is done before lane0 overwrites sh_G*
        }
    }

    warp_reduce_sum_arr<12>(acc);

    __shared__ double sh_warp_sum[8][12];
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            sh_warp_sum[warp][i] = acc[i];
        }
    }
    __syncthreads();

    if (tid == 0) {
        double sum[12];
        #pragma unroll
        for (int i = 0; i < 12; ++i) sum[i] = 0.0;

        for (int w = 0; w < nwarps; ++w) {
            #pragma unroll
            for (int i = 0; i < 12; ++i) {
                sum[i] += sh_warp_sum[w][i];
            }
        }

        double* out_task = out + static_cast<size_t>(t) * 12;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            out_task[i] = sum[i];
        }
    }
}

// ---- C API ----

extern "C" cudaError_t cueri_eri_rys_generic_deriv_contracted_cart_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int32_t ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int32_t la,
    int32_t lb,
    int32_t lc,
    int32_t ld,
    const double* bar_tile,
    double* out,
    cudaStream_t stream,
    int32_t threads
) {
    if (ntasks <= 0) return cudaSuccess;
    if (threads <= 0) return cudaErrorInvalidValue;

    const int L_total = static_cast<int>(la + lb + lc + ld);
    const int nroots = ((L_total + 1) / 2) + 1;
    if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

    dim3 blocks(static_cast<unsigned int>(ntasks));
    dim3 tpb(static_cast<unsigned int>(threads));

    KernelERI_RysGenericDerivContracted<<<blocks, tpb, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_A,
        sp_B,
        sp_pair_start,
        sp_npair,
        shell_cx,
        shell_cy,
        shell_cz,
        shell_prim_start,
        shell_nprim,
        prim_exp,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        la,
        lb,
        lc,
        ld,
        nroots,
        bar_tile,
        out
    );

    return cudaGetLastError();
}

__global__ void KernelERI_RysGenericDerivContractedAtomGrad(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int32_t ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    int ld,
    int nroots,
    const double* bar_tile,
    const int32_t* shell_atom,
    double* grad_out
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (static_cast<int>(blockDim.x) + 31) >> 5;

    const int t = static_cast<int>(blockIdx.x);
    if (t >= ntasks) return;

    const int spAB = static_cast<int>(task_spAB[t]);
    const int spCD = static_cast<int>(task_spCD[t]);

    const int A = static_cast<int>(sp_A[spAB]);
    const int B = static_cast<int>(sp_B[spAB]);
    const int C = static_cast<int>(sp_A[spCD]);
    const int D = static_cast<int>(sp_B[spCD]);

    const int nA = ncart(la);
    const int nB = ncart(lb);
    const int nC = ncart(lc);
    const int nD = ncart(ld);
    const int nAB = nA * nB;
    const int nCD = nC * nD;
    const int nElem = nAB * nCD;

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Bx = shell_cx[B];
    const double By = shell_cy[B];
    const double Bz = shell_cz[B];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];
    const double Dx = shell_cx[D];
    const double Dy = shell_cy[D];
    const double Dz = shell_cz[D];

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    const double CDx = Cx - Dx;
    const double CDy = Cy - Dy;
    const double CDz = Cz - Dz;

    __shared__ double sh_xij_pow[kLMaxD + 1];
    __shared__ double sh_yij_pow[kLMaxD + 1];
    __shared__ double sh_zij_pow[kLMaxD + 1];
    __shared__ double sh_xkl_pow[kLMaxD + 1];
    __shared__ double sh_ykl_pow[kLMaxD + 1];
    __shared__ double sh_zkl_pow[kLMaxD + 1];

    __shared__ int8_t shA_lx[kNcartMax];
    __shared__ int8_t shA_ly[kNcartMax];
    __shared__ int8_t shA_lz[kNcartMax];
    __shared__ int8_t shB_lx[kNcartMax];
    __shared__ int8_t shB_ly[kNcartMax];
    __shared__ int8_t shB_lz[kNcartMax];
    __shared__ int8_t shC_lx[kNcartMax];
    __shared__ int8_t shC_ly[kNcartMax];
    __shared__ int8_t shC_lz[kNcartMax];
    __shared__ int8_t shD_lx[kNcartMax];
    __shared__ int8_t shD_ly[kNcartMax];
    __shared__ int8_t shD_lz[kNcartMax];

    __shared__ double sh_bar[kBarCacheMax];

    if (tid == 0) {
        sh_xij_pow[0] = sh_yij_pow[0] = sh_zij_pow[0] = 1.0;
        sh_xkl_pow[0] = sh_ykl_pow[0] = sh_zkl_pow[0] = 1.0;
        for (int m = 1; m <= kLMaxD; ++m) {
            sh_xij_pow[m] = sh_xij_pow[m - 1] * ABx;
            sh_yij_pow[m] = sh_yij_pow[m - 1] * ABy;
            sh_zij_pow[m] = sh_zij_pow[m - 1] * ABz;
            sh_xkl_pow[m] = sh_xkl_pow[m - 1] * CDx;
            sh_ykl_pow[m] = sh_ykl_pow[m - 1] * CDy;
            sh_zkl_pow[m] = sh_zkl_pow[m - 1] * CDz;
        }
        fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
        fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
        fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);
        fill_cart_comp(ld, shD_lx, shD_ly, shD_lz);
    }

    if (nElem <= kBarCacheMax) {
        for (int i = tid; i < nElem; i += static_cast<int>(blockDim.x)) {
            sh_bar[i] = bar_tile[static_cast<size_t>(t) * static_cast<size_t>(nElem) + static_cast<size_t>(i)];
        }
    }
    __syncthreads();

    constexpr int kMaxWarps = 8; // threads<=256
    __shared__ double sh_roots[kMaxWarps][11];
    __shared__ double sh_weights[kMaxWarps][11];
    __shared__ double sh_Gx[kMaxWarps][kGSizeD];
    __shared__ double sh_Gy[kMaxWarps][kGSizeD];
    __shared__ double sh_Gz[kMaxWarps][kGSizeD];

    const int baseAB = static_cast<int>(sp_pair_start[spAB]);
    const int baseCD = static_cast<int>(sp_pair_start[spCD]);
    const int nPairAB = static_cast<int>(sp_npair[spAB]);
    const int nPairCD = static_cast<int>(sp_npair[spCD]);
    const int nTot = nPairAB * nPairCD;

    const int sA = static_cast<int>(shell_prim_start[A]);
    const int sB = static_cast<int>(shell_prim_start[B]);
    const int sC = static_cast<int>(shell_prim_start[C]);
    const int sD = static_cast<int>(shell_prim_start[D]);
    const int nprimB = static_cast<int>(shell_nprim[B]);
    const int nprimD = static_cast<int>(shell_nprim[D]);

    const int nmax = la + lb + 1;
    const int mmax = lc + ld + 1;

    double acc[12];
    #pragma unroll
    for (int i = 0; i < 12; ++i) acc[i] = 0.0;

    for (int uPair = warp; uPair < nTot; uPair += nwarps) {
        const int iPairAB = uPair / nPairCD;
        const int iPairCD = uPair - iPairAB * nPairCD;

        const double p = pair_eta[baseAB + iPairAB];
        const double Px = pair_Px[baseAB + iPairAB];
        const double Py = pair_Py[baseAB + iPairAB];
        const double Pz = pair_Pz[baseAB + iPairAB];
        const double cKab = pair_cK[baseAB + iPairAB];

        const int ipA = iPairAB / nprimB;
        const int ipB = iPairAB - ipA * nprimB;
        const double aexp = prim_exp[sA + ipA];
        const double bexp = prim_exp[sB + ipB];

        const double q = pair_eta[baseCD + iPairCD];
        const double Qx = pair_Px[baseCD + iPairCD];
        const double Qy = pair_Py[baseCD + iPairCD];
        const double Qz = pair_Pz[baseCD + iPairCD];
        const double cKcd = pair_cK[baseCD + iPairCD];

        const int ipC = iPairCD / nprimD;
        const int ipD = iPairCD - ipC * nprimD;
        const double cexp = prim_exp[sC + ipC];
        const double dexp = prim_exp[sD + ipD];

        const double denom = p + q;
        const double inv_denom = 1.0 / denom;
        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;
        const double omega = p * q * inv_denom;
        const double T = omega * PQ2;
        const double base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd;

        if (lane == 0) {
            rys_roots_weights_dispatch(nroots, T, sh_roots[warp], sh_weights[warp]);
        }
        __syncwarp();

        #pragma unroll
        for (int u = 0; u < 11; ++u) {
            if (u >= nroots) break;

            if (lane == 0) {
                const double x = sh_roots[warp][u];
                const double B0 = x * 0.5 * inv_denom;
                const double B1 = (1.0 - x) * 0.5 / p + B0;
                const double B1p = (1.0 - x) * 0.5 / q + B0;

                const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
                const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
                const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

                const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
                const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
                const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

                compute_G_stride<kStrideD>(sh_Gx[warp], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
                compute_G_stride<kStrideD>(sh_Gy[warp], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
                compute_G_stride<kStrideD>(sh_Gz[warp], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);
            }
            __syncwarp();

            const double scale = base * sh_weights[warp][u];

            for (int idxElem = lane; idxElem < nElem; idxElem += 32) {
                const double bar = (nElem <= kBarCacheMax)
                                       ? sh_bar[idxElem]
                                       : bar_tile[static_cast<size_t>(t) * static_cast<size_t>(nElem) + static_cast<size_t>(idxElem)];
                if (bar == 0.0) continue;

                const int row = idxElem / nCD;
                const int col = idxElem - row * nCD;
                const int ia = row / nB;
                const int ib = row - ia * nB;
                const int ic = col / nD;
                const int id = col - ic * nD;

                const int iax = static_cast<int>(shA_lx[ia]);
                const int iay = static_cast<int>(shA_ly[ia]);
                const int iaz = static_cast<int>(shA_lz[ia]);
                const int ibx = static_cast<int>(shB_lx[ib]);
                const int iby = static_cast<int>(shB_ly[ib]);
                const int ibz = static_cast<int>(shB_lz[ib]);
                const int icx = static_cast<int>(shC_lx[ic]);
                const int icy = static_cast<int>(shC_ly[ic]);
                const int icz = static_cast<int>(shC_lz[ic]);
                const int idx_d = static_cast<int>(shD_lx[id]);
                const int idy_d = static_cast<int>(shD_ly[id]);
                const int idz_d = static_cast<int>(shD_lz[id]);

                const double Ix = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double Iy = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double Iz = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow);

                const double bar_scale = bar * scale;

                const double IxA_m = (iax > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax - 1, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxA_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax + 1, ibx, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxA = (-static_cast<double>(iax)) * IxA_m + (2.0 * aexp) * IxA_p;
                acc[0] += bar_scale * (dIxA * Iy * Iz);

                const double IyA_m = (iay > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay - 1, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyA_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay + 1, iby, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyA = (-static_cast<double>(iay)) * IyA_m + (2.0 * aexp) * IyA_p;
                acc[1] += bar_scale * (Ix * dIyA * Iz);

                const double IzA_m = (iaz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz - 1, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzA_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz + 1, ibz, icz, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzA = (-static_cast<double>(iaz)) * IzA_m + (2.0 * aexp) * IzA_p;
                acc[2] += bar_scale * (Ix * Iy * dIzA);

                const double IxB_m = (ibx > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx - 1, icx, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxB_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx + 1, icx, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxB = (-static_cast<double>(ibx)) * IxB_m + (2.0 * bexp) * IxB_p;
                acc[3] += bar_scale * (dIxB * Iy * Iz);

                const double IyB_m = (iby > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby - 1, icy, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyB_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby + 1, icy, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyB = (-static_cast<double>(iby)) * IyB_m + (2.0 * bexp) * IyB_p;
                acc[4] += bar_scale * (Ix * dIyB * Iz);

                const double IzB_m = (ibz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz - 1, icz, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzB_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz + 1, icz, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzB = (-static_cast<double>(ibz)) * IzB_m + (2.0 * bexp) * IzB_p;
                acc[5] += bar_scale * (Ix * Iy * dIzB);

                const double IxC_m = (icx > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx - 1, idx_d, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxC_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx + 1, idx_d, sh_xij_pow, sh_xkl_pow);
                const double dIxC = (-static_cast<double>(icx)) * IxC_m + (2.0 * cexp) * IxC_p;
                acc[6] += bar_scale * (dIxC * Iy * Iz);

                const double IyC_m = (icy > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy - 1, idy_d, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyC_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy + 1, idy_d, sh_yij_pow, sh_ykl_pow);
                const double dIyC = (-static_cast<double>(icy)) * IyC_m + (2.0 * cexp) * IyC_p;
                acc[7] += bar_scale * (Ix * dIyC * Iz);

                const double IzC_m = (icz > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz - 1, idz_d, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzC_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz + 1, idz_d, sh_zij_pow, sh_zkl_pow);
                const double dIzC = (-static_cast<double>(icz)) * IzC_m + (2.0 * cexp) * IzC_p;
                acc[8] += bar_scale * (Ix * Iy * dIzC);

                const double IxD_m = (idx_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d - 1, sh_xij_pow, sh_xkl_pow) : 0.0;
                const double IxD_p = shift_from_G_stride_d<kStrideD>(sh_Gx[warp], iax, ibx, icx, idx_d + 1, sh_xij_pow, sh_xkl_pow);
                const double dIxD = (-static_cast<double>(idx_d)) * IxD_m + (2.0 * dexp) * IxD_p;
                acc[9] += bar_scale * (dIxD * Iy * Iz);

                const double IyD_m = (idy_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d - 1, sh_yij_pow, sh_ykl_pow) : 0.0;
                const double IyD_p = shift_from_G_stride_d<kStrideD>(sh_Gy[warp], iay, iby, icy, idy_d + 1, sh_yij_pow, sh_ykl_pow);
                const double dIyD = (-static_cast<double>(idy_d)) * IyD_m + (2.0 * dexp) * IyD_p;
                acc[10] += bar_scale * (Ix * dIyD * Iz);

                const double IzD_m = (idz_d > 0) ? shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d - 1, sh_zij_pow, sh_zkl_pow) : 0.0;
                const double IzD_p = shift_from_G_stride_d<kStrideD>(sh_Gz[warp], iaz, ibz, icz, idz_d + 1, sh_zij_pow, sh_zkl_pow);
                const double dIzD = (-static_cast<double>(idz_d)) * IzD_m + (2.0 * dexp) * IzD_p;
                acc[11] += bar_scale * (Ix * Iy * dIzD);
            }
            __syncwarp();
        }
    }

    warp_reduce_sum_arr<12>(acc);

    __shared__ double sh_warp_sum[8][12];
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            sh_warp_sum[warp][i] = acc[i];
        }
    }
    __syncthreads();

    if (tid == 0) {
        double sum[12];
        #pragma unroll
        for (int i = 0; i < 12; ++i) sum[i] = 0.0;
        for (int w = 0; w < nwarps; ++w) {
            #pragma unroll
            for (int i = 0; i < 12; ++i) {
                sum[i] += sh_warp_sum[w][i];
            }
        }

        const int atomA = static_cast<int>(shell_atom[A]);
        const int atomB = static_cast<int>(shell_atom[B]);
        const int atomC = static_cast<int>(shell_atom[C]);
        const int atomD = static_cast<int>(shell_atom[D]);

        atomicAdd(&grad_out[static_cast<size_t>(atomA) * 3 + 0], sum[0]);
        atomicAdd(&grad_out[static_cast<size_t>(atomA) * 3 + 1], sum[1]);
        atomicAdd(&grad_out[static_cast<size_t>(atomA) * 3 + 2], sum[2]);

        atomicAdd(&grad_out[static_cast<size_t>(atomB) * 3 + 0], sum[3]);
        atomicAdd(&grad_out[static_cast<size_t>(atomB) * 3 + 1], sum[4]);
        atomicAdd(&grad_out[static_cast<size_t>(atomB) * 3 + 2], sum[5]);

        atomicAdd(&grad_out[static_cast<size_t>(atomC) * 3 + 0], sum[6]);
        atomicAdd(&grad_out[static_cast<size_t>(atomC) * 3 + 1], sum[7]);
        atomicAdd(&grad_out[static_cast<size_t>(atomC) * 3 + 2], sum[8]);

        atomicAdd(&grad_out[static_cast<size_t>(atomD) * 3 + 0], sum[9]);
        atomicAdd(&grad_out[static_cast<size_t>(atomD) * 3 + 1], sum[10]);
        atomicAdd(&grad_out[static_cast<size_t>(atomD) * 3 + 2], sum[11]);
    }
}

extern "C" cudaError_t cueri_eri_rys_generic_deriv_contracted_atom_grad_inplace_cart_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int32_t la,
    int32_t lb,
    int32_t lc,
    int32_t ld,
    const double* bar_eri,
    const int32_t* shell_atom,
    double* grad_out,
    cudaStream_t stream,
    int32_t threads
) {
    if (ntasks <= 0) return cudaSuccess;
    if (threads <= 0) return cudaErrorInvalidValue;

    const int L_total = static_cast<int>(la + lb + lc + ld);
    const int nroots = ((L_total + 1) / 2) + 1;
    if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

    dim3 blocks(static_cast<unsigned int>(ntasks));
    dim3 tpb(static_cast<unsigned int>(threads));
    KernelERI_RysGenericDerivContractedAtomGrad<<<blocks, tpb, 0, stream>>>(
        task_spAB,
        task_spCD,
        ntasks,
        sp_A,
        sp_B,
        sp_pair_start,
        sp_npair,
        shell_cx,
        shell_cy,
        shell_cz,
        shell_prim_start,
        shell_nprim,
        prim_exp,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        la,
        lb,
        lc,
        ld,
        nroots,
        bar_eri,
        shell_atom,
        grad_out
    );

    return cudaGetLastError();
}
