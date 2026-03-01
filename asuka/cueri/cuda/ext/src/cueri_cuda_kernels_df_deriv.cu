#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cueri_cuda_kernels_api.h"
#include "cueri_cart2sph_tables.cuh"
#include "cueri_cuda_rys_device.cuh"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPiToFiveHalves = 2.0 * kPi * kPi * 1.772453850905516027298167483341145182;  // 2*pi^(5/2)

// Integral engine limits (match CPU Step-2 implementation).
constexpr int kLMax = 5;
// DF derivative contractions need component exponents up to (l+1) == 6.
constexpr int kLMaxD = 6;

// For ld=0 DF kernels, the G(a,b) table is only needed for:
//   a = 0..(la+lb+1) <= 11  (derivatives require +1)
//   b = 0..(lc+1)     <= 6  (derivatives require +1)
// Store as a compact (a,b) rectangle with a-major layout and fixed stride in b.
constexpr int kNMaxD = 2 * kLMax + 1;          // 11
constexpr int kMMaxD = kLMax + 1;              // 6
constexpr int kGStrideD = kMMaxD + 1;          // 7
constexpr int kGSizeD = (kNMaxD + 1) * kGStrideD;  // 12*7 = 84

constexpr int kNcartMax = (kLMax + 1) * (kLMax + 2) / 2;  // 21 for l=5
constexpr int kMaxWarpsPerBlock = 8;  // threads <= 256

// Cache small DF tiles of bar_X / bar_V in shared memory when nElem is modest.
// This substantially reduces redundant global loads across primitive+root loops.
constexpr int kBarCacheMax = 1024;
// Aggregate C-center atom updates per block before global atomic flush.
constexpr int kAtomAggMax = 64;

__device__ __forceinline__ int ncart(int l) { return ((l + 1) * (l + 2)) >> 1; }
__device__ __forceinline__ int nsph(int l) { return (l << 1) + 1; }

// Transform spherical adjoint bar_X(Q,m_sph,n_sph) into a single Cartesian AO-pair element
// bar_X_cart(m_cart,n_cart,Q) for the given (shellA,shellB) local AO indices.
__device__ __forceinline__ double df_bar_cart_from_sph_qmn(
    const double* __restrict__ bar_X_sph_Qmn,  // [naux, nao_sph, nao_sph] C-order
    int q,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);
  double acc = 0.0;
#pragma unroll
  for (int isA = 0; isA < 11; ++isA) {
    if (isA >= nA_sph) break;
    const double ta = cart2sph_coeff(la, ia_cart, isA);
    if (ta == 0.0) continue;
    const int i = a0_sph + isA;
    const int64_t base = (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) *
                         static_cast<int64_t>(nao_sph);
#pragma unroll
    for (int isB = 0; isB < 11; ++isB) {
      if (isB >= nB_sph) break;
      const double tb = cart2sph_coeff(lb, ib_cart, isB);
      if (tb == 0.0) continue;
      const int j = b0_sph + isB;
      acc += (ta * tb) * bar_X_sph_Qmn[base + static_cast<int64_t>(j)];
    }
  }
  return acc;
}

__device__ __forceinline__ void fill_cart_comp(int l, int8_t* lx, int8_t* ly, int8_t* lz) {
  int idx = 0;
  for (int x = l; x >= 0; --x) {
    const int rest = l - x;
    for (int y = rest; y >= 0; --y) {
      const int z = rest - y;
      lx[idx] = static_cast<int8_t>(x);
      ly[idx] = static_cast<int8_t>(y);
      lz[idx] = static_cast<int8_t>(z);
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
    double B1p) {
  // Normalized PyQuante2/GAMESS recurrence. Only b<=mmax and a<=nmax are used.
  G[0] = 1.0;
  if (nmax > 0) G[1 * STRIDE + 0] = C;
  if (mmax > 0) G[0 * STRIDE + 1] = Cp;

#pragma unroll
  for (int a = 2; a <= nmax; ++a) {
    G[a * STRIDE + 0] = B1 * static_cast<double>(a - 1) * G[(a - 2) * STRIDE + 0] + C * G[(a - 1) * STRIDE + 0];
  }
#pragma unroll
  for (int b = 2; b <= mmax; ++b) {
    G[0 * STRIDE + b] = B1p * static_cast<double>(b - 1) * G[0 * STRIDE + (b - 2)] + Cp * G[0 * STRIDE + (b - 1)];
  }

  if (mmax == 0 || nmax == 0) return;

#pragma unroll
  for (int a = 1; a <= nmax; ++a) {
    G[a * STRIDE + 1] = static_cast<double>(a) * B0 * G[(a - 1) * STRIDE + 0] + Cp * G[a * STRIDE + 0];
#pragma unroll
    for (int b = 2; b <= mmax; ++b) {
      G[a * STRIDE + b] = B1p * static_cast<double>(b - 1) * G[a * STRIDE + (b - 2)] +
                          static_cast<double>(a) * B0 * G[(a - 1) * STRIDE + (b - 1)] + Cp * G[a * STRIDE + (b - 1)];
    }
  }
}

__device__ __forceinline__ void compute_G_d(double* G, int nmax, int mmax, double C, double Cp, double B0, double B1, double B1p) {
  compute_G_stride<kGStrideD>(G, nmax, mmax, C, Cp, B0, B1, B1p);
}

// Specialized shift for ld=0 (DF 3c2e / 2c2e metric): l=0 => outer sum collapses.
template <int STRIDE>
__device__ __forceinline__ double shift_from_G_ld0_stride(const double* G, int i, int j, int k, const double* xij_pow) {
  switch (j) {
    case 0:
      return G[(i + 0) * STRIDE + k];
    case 1:
      return xij_pow[1] * G[(i + 0) * STRIDE + k] + G[(i + 1) * STRIDE + k];
    case 2:
      return xij_pow[2] * G[(i + 0) * STRIDE + k] + 2.0 * xij_pow[1] * G[(i + 1) * STRIDE + k] + G[(i + 2) * STRIDE + k];
    case 3:
      return xij_pow[3] * G[(i + 0) * STRIDE + k] + 3.0 * xij_pow[2] * G[(i + 1) * STRIDE + k] +
             3.0 * xij_pow[1] * G[(i + 2) * STRIDE + k] + G[(i + 3) * STRIDE + k];
    case 4:
      return xij_pow[4] * G[(i + 0) * STRIDE + k] + 4.0 * xij_pow[3] * G[(i + 1) * STRIDE + k] +
             6.0 * xij_pow[2] * G[(i + 2) * STRIDE + k] + 4.0 * xij_pow[1] * G[(i + 3) * STRIDE + k] +
             G[(i + 4) * STRIDE + k];
    case 5:
      return xij_pow[5] * G[(i + 0) * STRIDE + k] + 5.0 * xij_pow[4] * G[(i + 1) * STRIDE + k] +
             10.0 * xij_pow[3] * G[(i + 2) * STRIDE + k] + 10.0 * xij_pow[2] * G[(i + 3) * STRIDE + k] +
             5.0 * xij_pow[1] * G[(i + 4) * STRIDE + k] + G[(i + 5) * STRIDE + k];
    case 6:
      return xij_pow[6] * G[(i + 0) * STRIDE + k] + 6.0 * xij_pow[5] * G[(i + 1) * STRIDE + k] +
             15.0 * xij_pow[4] * G[(i + 2) * STRIDE + k] + 20.0 * xij_pow[3] * G[(i + 3) * STRIDE + k] +
             15.0 * xij_pow[2] * G[(i + 4) * STRIDE + k] + 6.0 * xij_pow[1] * G[(i + 5) * STRIDE + k] +
             G[(i + 6) * STRIDE + k];
    default:
      break;
  }
  constexpr double kBinom[kLMaxD + 1][kLMaxD + 1] = {
      {1, 0, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 0},
      {1, 2, 1, 0, 0, 0, 0},
      {1, 3, 3, 1, 0, 0, 0},
      {1, 4, 6, 4, 1, 0, 0},
      {1, 5, 10, 10, 5, 1, 0},
      {1, 6, 15, 20, 15, 6, 1},
  };
  const int jj = (j < 0) ? 0 : ((j > kLMaxD) ? kLMaxD : j);
  double out = 0.0;
#pragma unroll
  for (int n = 0; n <= kLMaxD; ++n) {
    if (n > jj) break;
    out += kBinom[jj][n] * xij_pow[jj - n] * G[(n + i) * STRIDE + k];
  }
  return out;
}

__device__ __forceinline__ double shift_from_G_ld0_d(const double* G, int i, int j, int k, const double* xij_pow) {
  return shift_from_G_ld0_stride<kGStrideD>(G, i, j, k, xij_pow);
}

__device__ __forceinline__ double warp_reduce_sum(double x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

template <int N>
__device__ __forceinline__ void warp_reduce_sum_arr(double (&v)[N]) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] += __shfl_down_sync(0xffffffff, v[i], offset);
    }
  }
}

template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartBatch(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    double* out) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;

  // Shared per-block (task) data.
  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  // Per-warp G tables.
  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  // Per-warp partial sums (lane 0 only) and optional bar tile cache.
  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ double sh_bar[kBarCacheMax];

  const int spCD_i = static_cast<int>(spCD[t]);

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]);
  const int b0 = static_cast<int>(shell_ao_start[shellB]);
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }

  // Optional shared cache of bar tile for this (spAB, shellC) pair.
  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / (nB * nC);
      const int rem = idx - ia * (nB * nC);
      const int ib = rem / nC;
      const int ic = rem - ib * nC;
      const int row_idx = (a0 + ia) * nao + (b0 + ib);
      const int col_idx = c0 + ic;
      sh_bar[idx] = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) acc[i] = 0.0;

  const int nmax = la + lb + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const int ia_prim = iAB / nprimB;
    const int ib_prim = iAB - ia_prim * nprimB;
    const double aexp = prim_exp[sA + ia_prim];
    const double bexp = prim_exp[sB + ib_prim];
    const double cexp = prim_exp[sC + iCD];  // ld=0 => nprimD=1

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    // Roots/weights are computed once per (iAB,iCD) and broadcast from lane 0.
    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
          else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        double bar = 0.0;
        if (cache_bar) {
          bar = sh_bar[idx];
        } else {
          const int ia = idx / (nB * nC);
          const int rem = idx - ia * (nB * nC);
          const int ib = rem / nC;
          const int ic = rem - ib * nC;
          const int row_idx = (a0 + ia) * nao + (b0 + ib);
          const int col_idx = c0 + ic;
          bar = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        }
        if (bar == 0.0) continue;

        const int ia = idx / (nB * nC);
        const int rem = idx - ia * (nB * nC);
        const int ib = rem / nC;
        const int ic = rem - ib * nC;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int ibx = static_cast<int>(shB_lx[ib]);
        const int iby = static_cast<int>(shB_ly[ib]);
        const int ibz = static_cast<int>(shB_lz[ib]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);

        const double bar_scale = bar * scale;

        // Center A derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center B derivatives.
        const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
        const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
        const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
        acc[3] += bar_scale * (dIx_B * Iy * Iz);

        const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
        const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
        const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
        acc[4] += bar_scale * (Ix * dIy_B * Iz);

        const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
        const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
        const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
        acc[5] += bar_scale * (Ix * Iy * dIz_B);

        // Center C derivatives (aux).
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[6] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[7] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[8] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  // Warp reduction then block reduction over warps (lane 0 only).
  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const int out0 = t * 9;
#pragma unroll
    for (int i = 0; i < 9; ++i) out[out0 + i] = sum[i];
  }
}

// Batched variant: processes ALL AO shell pairs in one (la,lb) class × all spCD in one lq class
// in a single 2D kernel launch.  Results are accumulated directly into grad_dev via atomicAdd.
// Grid: dim3(ntasks_cd, n_spAB).  Block: same thread layout as KernelDFInt3c2eDerivContractedCartBatch.
template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartAllSPAtomGrad(
    const int32_t* spAB_arr,   // [n_spAB] AO shell-pair indices in this (la,lb) class
    int n_spAB,
    const int32_t* spCD,       // [ntasks] aux shell-pair indices
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,  // combined AO+aux shell->atom map (length: nAOshells+nAuxShells)
    double* grad_dev) {         // [natm*3] gradient accumulator (atomicAdd target)
  const int t   = static_cast<int>(blockIdx.x);  // CD task index
  const int iAB = static_cast<int>(blockIdx.y);  // AB class index
  if (t >= ntasks || iAB >= n_spAB) return;

  const int spAB   = static_cast<int>(spAB_arr[iAB]);
  const int spCD_i = static_cast<int>(spCD[t]);

  // --- Shared memory (identical layout to KernelDFInt3c2eDerivContractedCartBatch) ---
  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ double sh_bar[kBarCacheMax];

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]);
  const int b0 = static_cast<int>(shell_ao_start[shellB]);
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / (nB * nC);
      const int rem = idx - ia * (nB * nC);
      const int ib = rem / nC;
      const int ic = rem - ib * nC;
      const int row_idx = (a0 + ia) * nao + (b0 + ib);
      const int col_idx = c0 + ic;
      sh_bar[idx] = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) acc[i] = 0.0;

  const int nmax = la + lb + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB_prim;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const int ia_prim = iAB_prim / nprimB;
    const int ib_prim = iAB_prim - ia_prim * nprimB;
    const double aexp = prim_exp[sA + ia_prim];
    const double bexp = prim_exp[sB + ib_prim];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
          else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        double bar = 0.0;
        if (cache_bar) {
          bar = sh_bar[idx];
        } else {
          const int ia = idx / (nB * nC);
          const int rem = idx - ia * (nB * nC);
          const int ib = rem / nC;
          const int ic = rem - ib * nC;
          const int row_idx = (a0 + ia) * nao + (b0 + ib);
          const int col_idx = c0 + ic;
          bar = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        }
        if (bar == 0.0) continue;

        const int ia = idx / (nB * nC);
        const int rem = idx - ia * (nB * nC);
        const int ib = rem / nC;
        const int ic = rem - ib * nC;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int ibx = static_cast<int>(shB_lx[ib]);
        const int iby = static_cast<int>(shB_ly[ib]);
        const int ibz = static_cast<int>(shB_lz[ib]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);

        const double bar_scale = bar * scale;

        // Center A derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center B derivatives.
        const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
        const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
        const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
        acc[3] += bar_scale * (dIx_B * Iy * Iz);

        const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
        const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
        const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
        acc[4] += bar_scale * (Ix * dIy_B * Iz);

        const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
        const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
        const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
        acc[5] += bar_scale * (Ix * Iy * dIz_B);

        // Center C derivatives (aux).
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[6] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[7] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[8] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  // Warp reduction then block reduction over warps (lane 0 only).
  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const double fac = (shellA != shellB) ? 2.0 : 1.0;
    const int atomA = static_cast<int>(shell_atom[shellA]);
    const int atomB = static_cast<int>(shell_atom[shellB]);
    const int atomC = static_cast<int>(shell_atom[shellC]);
    atomicAdd(&grad_dev[atomA * 3 + 0], fac * sum[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], fac * sum[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], fac * sum[2]);
    atomicAdd(&grad_dev[atomB * 3 + 0], fac * sum[3]);
    atomicAdd(&grad_dev[atomB * 3 + 1], fac * sum[4]);
    atomicAdd(&grad_dev[atomB * 3 + 2], fac * sum[5]);
    atomicAdd(&grad_dev[atomC * 3 + 0], fac * sum[6]);
    atomicAdd(&grad_dev[atomC * 3 + 1], fac * sum[7]);
    atomicAdd(&grad_dev[atomC * 3 + 2], fac * sum[8]);
  }
}

// Spherical bar_X variant: consumes bar_X in spherical AO basis in Qmn layout and applies the
// cart<-sph transforms inside the contraction kernel to avoid materializing the full
// Cartesian bar_X tensor (nao_cart^2 * naux).
template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn(
    const int32_t* spAB_arr,   // [n_spAB] AO shell-pair indices in this (la,lb) class
    int n_spAB,
    const int32_t* spCD,       // [ntasks] aux shell-pair indices
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,       // [naux*nao_sph*nao_sph] (Q,m,n) C-order
    const int32_t* shell_ao_start_sph, // spherical AO start offset per AO shell
    const int32_t* shell_atom,         // combined AO+aux shell->atom map (length: nAOshells+nAuxShells)
    double* grad_dev) {                // [natm*3] gradient accumulator (atomicAdd target)
  const int t   = static_cast<int>(blockIdx.x);  // CD task index
  const int iAB = static_cast<int>(blockIdx.y);  // AB class index
  if (t >= ntasks || iAB >= n_spAB) return;

  const int spAB   = static_cast<int>(spAB_arr[iAB]);
  const int spCD_i = static_cast<int>(spCD[t]);

  // --- Shared memory (identical layout to KernelDFInt3c2eDerivContractedCartBatch) ---
  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ double sh_bar[kBarCacheMax];

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;

  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / (nB * nC);
      const int rem = idx - ia * (nB * nC);
      const int ib = rem / nC;
      const int ic = rem - ib * nC;
      const int q = c0 + ic;
      sh_bar[idx] = df_bar_cart_from_sph_qmn(bar_X_sph_Qmn, q, ia, ib, la, lb, a0_sph, b0_sph, nao_sph);
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) acc[i] = 0.0;

  const int nmax = la + lb + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB_prim;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const int ia_prim = iAB_prim / nprimB;
    const int ib_prim = iAB_prim - ia_prim * nprimB;
    const double aexp = prim_exp[sA + ia_prim];
    const double bexp = prim_exp[sB + ib_prim];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
          else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        const int ia = idx / (nB * nC);
        const int rem = idx - ia * (nB * nC);
        const int ib = rem / nC;
        const int ic = rem - ib * nC;

        double bar = 0.0;
        if (cache_bar) {
          bar = sh_bar[idx];
        } else {
          const int q_idx = c0 + ic;
          bar = df_bar_cart_from_sph_qmn(bar_X_sph_Qmn, q_idx, ia, ib, la, lb, a0_sph, b0_sph, nao_sph);
        }
        if (bar == 0.0) continue;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int ibx = static_cast<int>(shB_lx[ib]);
        const int iby = static_cast<int>(shB_ly[ib]);
        const int ibz = static_cast<int>(shB_lz[ib]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);

        const double bar_scale = bar * scale;

        // Center A derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center B derivatives.
        const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
        const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
        const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
        acc[3] += bar_scale * (dIx_B * Iy * Iz);

        const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
        const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
        const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
        acc[4] += bar_scale * (Ix * dIy_B * Iz);

        const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
        const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
        const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
        acc[5] += bar_scale * (Ix * Iy * dIz_B);

        // Center C derivatives (aux).
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[6] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[7] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[8] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  // Warp reduction then block reduction over warps (lane 0 only).
  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const double fac = (shellA != shellB) ? 2.0 : 1.0;
    const int atomA = static_cast<int>(shell_atom[shellA]);
    const int atomB = static_cast<int>(shell_atom[shellB]);
    const int atomC = static_cast<int>(shell_atom[shellC]);
    atomicAdd(&grad_dev[atomA * 3 + 0], fac * sum[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], fac * sum[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], fac * sum[2]);
    atomicAdd(&grad_dev[atomB * 3 + 0], fac * sum[3]);
    atomicAdd(&grad_dev[atomB * 3 + 1], fac * sum[4]);
    atomicAdd(&grad_dev[atomB * 3 + 2], fac * sum[5]);
    atomicAdd(&grad_dev[atomC * 3 + 0], fac * sum[6]);
    atomicAdd(&grad_dev[atomC * 3 + 1], fac * sum[7]);
    atomicAdd(&grad_dev[atomC * 3 + 2], fac * sum[8]);
  }
}

// AB-tiled variant: each block processes one spAB and up to `cd_tile` spCD tasks.
// A/B-center derivatives are accumulated across tile tasks and atomically flushed once,
// while C-center derivatives are still emitted per task.
template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartAllSPAtomGradABTile(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev) {
  const int t0 = static_cast<int>(blockIdx.x) * cd_tile;
  const int iAB = static_cast<int>(blockIdx.y);
  if (iAB >= n_spAB || cd_tile <= 0 || t0 >= ntasks) return;

  const int spAB = static_cast<int>(spAB_arr[iAB]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];
  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ double sh_bar[kBarCacheMax];
  __shared__ int sh_atomC_agg[kAtomAggMax];
  __shared__ double sh_sumC_agg[kAtomAggMax][3];
  __shared__ int sh_n_atomC_agg;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;
  const int a0 = static_cast<int>(shell_ao_start[shellA]);
  const int b0 = static_cast<int>(shell_ao_start[shellB]);
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double fac = (shellA != shellB) ? 2.0 : 1.0;
  const int atomA = static_cast<int>(shell_atom[shellA]);
  const int atomB = static_cast<int>(shell_atom[shellB]);

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);
    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }
  __syncthreads();

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  const int nmax = la + lb + 1;
  const int mmax = lc + 1;

  double sumA_blk[3];
  double sumB_blk[3];
  if (threadIdx.x == 0) {
    sumA_blk[0] = 0.0;
    sumA_blk[1] = 0.0;
    sumA_blk[2] = 0.0;
    sumB_blk[0] = 0.0;
    sumB_blk[1] = 0.0;
    sumB_blk[2] = 0.0;
    sh_n_atomC_agg = 0;
  }
  __syncthreads();

  for (int tt = 0; tt < cd_tile; ++tt) {
    const int t = t0 + tt;
    const bool active = (t < ntasks);
    const int spCD_i = active ? static_cast<int>(spCD[t]) : 0;
    const int shellC = active ? static_cast<int>(sp_A[spCD_i]) : 0;
    const int c0 = active ? (static_cast<int>(shell_ao_start[shellC]) - nao) : 0;
    const int baseCD = active ? static_cast<int>(sp_pair_start[spCD_i]) : 0;
    const int nprimCD = active ? static_cast<int>(sp_npair[spCD_i]) : 0;
    const int sC = active ? static_cast<int>(shell_prim_start[shellC]) : 0;
    const double Cx = active ? shell_cx[shellC] : 0.0;
    const double Cy = active ? shell_cy[shellC] : 0.0;
    const double Cz = active ? shell_cz[shellC] : 0.0;

    if (cache_bar) {
      for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
        if (active) {
          const int ia = idx / (nB * nC);
          const int rem = idx - ia * (nB * nC);
          const int ib = rem / nC;
          const int ic = rem - ib * nC;
          const int row_idx = (a0 + ia) * nao + (b0 + ib);
          const int col_idx = c0 + ic;
          sh_bar[idx] = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        } else {
          sh_bar[idx] = 0.0;
        }
      }
    }
    __syncthreads();

    // Block-level bar_X early-exit: skip tile if max|sh_bar| < threshold.
    if (cache_bar) {
      double tmax = 0.0;
      for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
        double v = static_cast<double>(sh_bar[idx]);
        if (v < 0.0) v = -v;
        if (v > tmax) tmax = v;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xffffffff, tmax, offset);
        if (other > tmax) tmax = other;
      }
      {
        const int lane_ = static_cast<int>(threadIdx.x) & 31;
        const int warp_ = static_cast<int>(threadIdx.x) >> 5;
        if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        double bmax = sh_warp_sum[0][0];
        const int nw = static_cast<int>(blockDim.x) >> 5;
        for (int w = 1; w < nw; ++w) {
          if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
        }
        sh_warp_sum[0][0] = bmax;
      }
      __syncthreads();
      if (sh_warp_sum[0][0] < 1e-14) continue;
    }

    double acc[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) acc[i] = 0.0;

    if (active) {
      const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);
      for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
        const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
        const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
        const int ki = baseAB + iAB_prim;
        const int kj = baseCD + iCD;

        const double p = pair_eta[ki];
        const double q = pair_eta[kj];
        const double Px = pair_Px[ki];
        const double Py = pair_Py[ki];
        const double Pz = pair_Pz[ki];
        const double Qx = pair_Px[kj];
        const double Qy = pair_Py[kj];
        const double Qz = pair_Pz[kj];
        const double cKab = pair_cK[ki];
        const double cKcd = pair_cK[kj];

        const int ia_prim = iAB_prim / nprimB;
        const int ib_prim = iAB_prim - ia_prim * nprimB;
        const double aexp = prim_exp[sA + ia_prim];
        const double bexp = prim_exp[sB + ib_prim];
        const double cexp = prim_exp[sC + iCD];

        const double denom = p + q;
        const double inv_denom = 1.0 / denom;
        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;
        const double omega = p * q * inv_denom;
        const double T = omega * PQ2;
        const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        double roots[NROOTS];
        double weights[NROOTS];
        if (lane == 0) {
          cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
        }

        for (int r = 0; r < NROOTS; ++r) {
          double x = (lane == 0) ? roots[r] : 0.0;
          double w = (lane == 0) ? weights[r] : 0.0;
          x = __shfl_sync(0xffffffff, x, 0);
          w = __shfl_sync(0xffffffff, w, 0);

          {
            const double B0 = x * 0.5 * inv_denom;
            const double B1 = (1.0 - x) * 0.5 / p + B0;
            const double B1p = (1.0 - x) * 0.5 / q + B0;
            const double q_over = q * inv_denom;
            const double p_over = p * inv_denom;
            if (lane < 3) {
              double PA, QC, PQd;
              double* G_target;
              if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
              else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
              else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
              compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
            }
          }

          const double scale = base * w;
          __syncwarp();

          for (int idx = lane; idx < nElem; idx += 32) {
            double bar = 0.0;
            if (cache_bar) {
              bar = sh_bar[idx];
            } else {
              const int ia = idx / (nB * nC);
              const int rem = idx - ia * (nB * nC);
              const int ib = rem / nC;
              const int ic = rem - ib * nC;
              const int row_idx = (a0 + ia) * nao + (b0 + ib);
              const int col_idx = c0 + ic;
              bar = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
            }
            if (bar == 0.0) continue;

            const int ia = idx / (nB * nC);
            const int rem = idx - ia * (nB * nC);
            const int ib = rem / nC;
            const int ic = rem - ib * nC;
            const int iax = static_cast<int>(shA_lx[ia]);
            const int iay = static_cast<int>(shA_ly[ia]);
            const int iaz = static_cast<int>(shA_lz[ia]);
            const int ibx = static_cast<int>(shB_lx[ib]);
            const int iby = static_cast<int>(shB_ly[ib]);
            const int ibz = static_cast<int>(shB_lz[ib]);
            const int icx = static_cast<int>(shC_lx[ic]);
            const int icy = static_cast<int>(shC_ly[ic]);
            const int icz = static_cast<int>(shC_lz[ic]);

            const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
            const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
            const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);
            const double bar_scale = bar * scale;

            const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
            const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
            const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
            acc[0] += bar_scale * (dIx_A * Iy * Iz);

            const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
            const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
            const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
            acc[1] += bar_scale * (Ix * dIy_A * Iz);

            const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
            const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
            const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
            acc[2] += bar_scale * (Ix * Iy * dIz_A);

            const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
            const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
            const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
            acc[3] += bar_scale * (dIx_B * Iy * Iz);

            const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
            const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
            const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
            acc[4] += bar_scale * (Ix * dIy_B * Iz);

            const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
            const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
            const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
            acc[5] += bar_scale * (Ix * Iy * dIz_B);

            const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
            const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
            const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
            acc[6] += bar_scale * (dIx_C * Iy * Iz);

            const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
            const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
            const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
            acc[7] += bar_scale * (Ix * dIy_C * Iz);

            const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
            const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
            const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
            acc[8] += bar_scale * (Ix * Iy * dIz_C);
          }
          __syncwarp();
        }
      }
    }

    warp_reduce_sum_arr(acc);
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
    }
    __syncthreads();

    if (threadIdx.x == 0 && active) {
      double sum[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] = 0.0;
      for (int w = 0; w < warps; ++w) {
#pragma unroll
        for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
      }
      sumA_blk[0] += fac * sum[0];
      sumA_blk[1] += fac * sum[1];
      sumA_blk[2] += fac * sum[2];
      sumB_blk[0] += fac * sum[3];
      sumB_blk[1] += fac * sum[4];
      sumB_blk[2] += fac * sum[5];
      const int atomC = static_cast<int>(shell_atom[shellC]);
      int slot = -1;
#pragma unroll
      for (int u = 0; u < kAtomAggMax; ++u) {
        if (u >= sh_n_atomC_agg) break;
        if (sh_atomC_agg[u] == atomC) {
          slot = u;
          break;
        }
      }
      if (slot < 0 && sh_n_atomC_agg < kAtomAggMax) {
        slot = sh_n_atomC_agg;
        sh_atomC_agg[slot] = atomC;
        sh_sumC_agg[slot][0] = 0.0;
        sh_sumC_agg[slot][1] = 0.0;
        sh_sumC_agg[slot][2] = 0.0;
        ++sh_n_atomC_agg;
      }
      if (slot >= 0) {
        sh_sumC_agg[slot][0] += fac * sum[6];
        sh_sumC_agg[slot][1] += fac * sum[7];
        sh_sumC_agg[slot][2] += fac * sum[8];
      } else {
        atomicAdd(&grad_dev[atomC * 3 + 0], fac * sum[6]);
        atomicAdd(&grad_dev[atomC * 3 + 1], fac * sum[7]);
        atomicAdd(&grad_dev[atomC * 3 + 2], fac * sum[8]);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&grad_dev[atomA * 3 + 0], sumA_blk[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], sumA_blk[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], sumA_blk[2]);
    atomicAdd(&grad_dev[atomB * 3 + 0], sumB_blk[0]);
    atomicAdd(&grad_dev[atomB * 3 + 1], sumB_blk[1]);
    atomicAdd(&grad_dev[atomB * 3 + 2], sumB_blk[2]);
    for (int u = 0; u < sh_n_atomC_agg; ++u) {
      const int atomC = sh_atomC_agg[u];
      atomicAdd(&grad_dev[atomC * 3 + 0], sh_sumC_agg[u][0]);
      atomicAdd(&grad_dev[atomC * 3 + 1], sh_sumC_agg[u][1]);
      atomicAdd(&grad_dev[atomC * 3 + 2], sh_sumC_agg[u][2]);
    }
  }
}

template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmnABTile(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev) {
  const int t0 = static_cast<int>(blockIdx.x) * cd_tile;
  const int iAB = static_cast<int>(blockIdx.y);
  if (iAB >= n_spAB || cd_tile <= 0 || t0 >= ntasks) return;

  const int spAB = static_cast<int>(spAB_arr[iAB]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];
  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ double sh_bar[kBarCacheMax];
  __shared__ int sh_atomC_agg[kAtomAggMax];
  __shared__ double sh_sumC_agg[kAtomAggMax][3];
  __shared__ int sh_n_atomC_agg;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;
  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);
  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double fac = (shellA != shellB) ? 2.0 : 1.0;
  const int atomA = static_cast<int>(shell_atom[shellA]);
  const int atomB = static_cast<int>(shell_atom[shellB]);

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);
    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }
  __syncthreads();

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  const int nmax = la + lb + 1;
  const int mmax = lc + 1;

  double sumA_blk[3];
  double sumB_blk[3];
  if (threadIdx.x == 0) {
    sumA_blk[0] = 0.0;
    sumA_blk[1] = 0.0;
    sumA_blk[2] = 0.0;
    sumB_blk[0] = 0.0;
    sumB_blk[1] = 0.0;
    sumB_blk[2] = 0.0;
    sh_n_atomC_agg = 0;
  }
  __syncthreads();

  for (int tt = 0; tt < cd_tile; ++tt) {
    const int t = t0 + tt;
    const bool active = (t < ntasks);
    const int spCD_i = active ? static_cast<int>(spCD[t]) : 0;
    const int shellC = active ? static_cast<int>(sp_A[spCD_i]) : 0;
    const int c0 = active ? (static_cast<int>(shell_ao_start[shellC]) - nao) : 0;
    const int baseCD = active ? static_cast<int>(sp_pair_start[spCD_i]) : 0;
    const int nprimCD = active ? static_cast<int>(sp_npair[spCD_i]) : 0;
    const int sC = active ? static_cast<int>(shell_prim_start[shellC]) : 0;
    const double Cx = active ? shell_cx[shellC] : 0.0;
    const double Cy = active ? shell_cy[shellC] : 0.0;
    const double Cz = active ? shell_cz[shellC] : 0.0;

    if (cache_bar) {
      for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
        if (active) {
          const int ia = idx / (nB * nC);
          const int rem = idx - ia * (nB * nC);
          const int ib = rem / nC;
          const int ic = rem - ib * nC;
          const int q = c0 + ic;
          sh_bar[idx] = df_bar_cart_from_sph_qmn(bar_X_sph_Qmn, q, ia, ib, la, lb, a0_sph, b0_sph, nao_sph);
        } else {
          sh_bar[idx] = 0.0;
        }
      }
    }
    __syncthreads();

    // Block-level bar_X early-exit: skip tile if max|sh_bar| < threshold.
    if (cache_bar) {
      double tmax = 0.0;
      for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
        double v = static_cast<double>(sh_bar[idx]);
        if (v < 0.0) v = -v;
        if (v > tmax) tmax = v;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xffffffff, tmax, offset);
        if (other > tmax) tmax = other;
      }
      {
        const int lane_ = static_cast<int>(threadIdx.x) & 31;
        const int warp_ = static_cast<int>(threadIdx.x) >> 5;
        if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        double bmax = sh_warp_sum[0][0];
        const int nw = static_cast<int>(blockDim.x) >> 5;
        for (int w = 1; w < nw; ++w) {
          if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
        }
        sh_warp_sum[0][0] = bmax;
      }
      __syncthreads();
      if (sh_warp_sum[0][0] < 1e-14) continue;
    }

    double acc[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) acc[i] = 0.0;

    if (active) {
      const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);
      for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
        const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
        const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
        const int ki = baseAB + iAB_prim;
        const int kj = baseCD + iCD;

        const double p = pair_eta[ki];
        const double q = pair_eta[kj];
        const double Px = pair_Px[ki];
        const double Py = pair_Py[ki];
        const double Pz = pair_Pz[ki];
        const double Qx = pair_Px[kj];
        const double Qy = pair_Py[kj];
        const double Qz = pair_Pz[kj];
        const double cKab = pair_cK[ki];
        const double cKcd = pair_cK[kj];

        const int ia_prim = iAB_prim / nprimB;
        const int ib_prim = iAB_prim - ia_prim * nprimB;
        const double aexp = prim_exp[sA + ia_prim];
        const double bexp = prim_exp[sB + ib_prim];
        const double cexp = prim_exp[sC + iCD];

        const double denom = p + q;
        const double inv_denom = 1.0 / denom;
        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;
        const double omega = p * q * inv_denom;
        const double T = omega * PQ2;
        const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        double roots[NROOTS];
        double weights[NROOTS];
        if (lane == 0) {
          cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
        }

        for (int r = 0; r < NROOTS; ++r) {
          double x = (lane == 0) ? roots[r] : 0.0;
          double w = (lane == 0) ? weights[r] : 0.0;
          x = __shfl_sync(0xffffffff, x, 0);
          w = __shfl_sync(0xffffffff, w, 0);

          {
            const double B0 = x * 0.5 * inv_denom;
            const double B1 = (1.0 - x) * 0.5 / p + B0;
            const double B1p = (1.0 - x) * 0.5 / q + B0;
            const double q_over = q * inv_denom;
            const double p_over = p * inv_denom;
            if (lane < 3) {
              double PA, QC, PQd;
              double* G_target;
              if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
              else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
              else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
              compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
            }
          }

          const double scale = base * w;
          __syncwarp();

          for (int idx = lane; idx < nElem; idx += 32) {
            const int ia = idx / (nB * nC);
            const int rem = idx - ia * (nB * nC);
            const int ib = rem / nC;
            const int ic = rem - ib * nC;

            double bar = 0.0;
            if (cache_bar) {
              bar = sh_bar[idx];
            } else {
              const int q_idx = c0 + ic;
              bar = df_bar_cart_from_sph_qmn(bar_X_sph_Qmn, q_idx, ia, ib, la, lb, a0_sph, b0_sph, nao_sph);
            }
            if (bar == 0.0) continue;

            const int iax = static_cast<int>(shA_lx[ia]);
            const int iay = static_cast<int>(shA_ly[ia]);
            const int iaz = static_cast<int>(shA_lz[ia]);
            const int ibx = static_cast<int>(shB_lx[ib]);
            const int iby = static_cast<int>(shB_ly[ib]);
            const int ibz = static_cast<int>(shB_lz[ib]);
            const int icx = static_cast<int>(shC_lx[ic]);
            const int icy = static_cast<int>(shC_ly[ic]);
            const int icz = static_cast<int>(shC_lz[ic]);

            const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
            const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
            const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);
            const double bar_scale = bar * scale;

            const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
            const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
            const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
            acc[0] += bar_scale * (dIx_A * Iy * Iz);

            const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
            const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
            const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
            acc[1] += bar_scale * (Ix * dIy_A * Iz);

            const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
            const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
            const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
            acc[2] += bar_scale * (Ix * Iy * dIz_A);

            const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
            const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
            const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
            acc[3] += bar_scale * (dIx_B * Iy * Iz);

            const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
            const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
            const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
            acc[4] += bar_scale * (Ix * dIy_B * Iz);

            const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
            const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
            const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
            acc[5] += bar_scale * (Ix * Iy * dIz_B);

            const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
            const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
            const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
            acc[6] += bar_scale * (dIx_C * Iy * Iz);

            const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
            const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
            const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
            acc[7] += bar_scale * (Ix * dIy_C * Iz);

            const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
            const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
            const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
            acc[8] += bar_scale * (Ix * Iy * dIz_C);
          }
          __syncwarp();
        }
      }
    }

    warp_reduce_sum_arr(acc);
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
    }
    __syncthreads();

    if (threadIdx.x == 0 && active) {
      double sum[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] = 0.0;
      for (int w = 0; w < warps; ++w) {
#pragma unroll
        for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
      }
      sumA_blk[0] += fac * sum[0];
      sumA_blk[1] += fac * sum[1];
      sumA_blk[2] += fac * sum[2];
      sumB_blk[0] += fac * sum[3];
      sumB_blk[1] += fac * sum[4];
      sumB_blk[2] += fac * sum[5];
      const int atomC = static_cast<int>(shell_atom[shellC]);
      int slot = -1;
#pragma unroll
      for (int u = 0; u < kAtomAggMax; ++u) {
        if (u >= sh_n_atomC_agg) break;
        if (sh_atomC_agg[u] == atomC) {
          slot = u;
          break;
        }
      }
      if (slot < 0 && sh_n_atomC_agg < kAtomAggMax) {
        slot = sh_n_atomC_agg;
        sh_atomC_agg[slot] = atomC;
        sh_sumC_agg[slot][0] = 0.0;
        sh_sumC_agg[slot][1] = 0.0;
        sh_sumC_agg[slot][2] = 0.0;
        ++sh_n_atomC_agg;
      }
      if (slot >= 0) {
        sh_sumC_agg[slot][0] += fac * sum[6];
        sh_sumC_agg[slot][1] += fac * sum[7];
        sh_sumC_agg[slot][2] += fac * sum[8];
      } else {
        atomicAdd(&grad_dev[atomC * 3 + 0], fac * sum[6]);
        atomicAdd(&grad_dev[atomC * 3 + 1], fac * sum[7]);
        atomicAdd(&grad_dev[atomC * 3 + 2], fac * sum[8]);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&grad_dev[atomA * 3 + 0], sumA_blk[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], sumA_blk[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], sumA_blk[2]);
    atomicAdd(&grad_dev[atomB * 3 + 0], sumB_blk[0]);
    atomicAdd(&grad_dev[atomB * 3 + 1], sumB_blk[1]);
    atomicAdd(&grad_dev[atomB * 3 + 2], sumB_blk[2]);
    for (int u = 0; u < sh_n_atomC_agg; ++u) {
      const int atomC = sh_atomC_agg[u];
      atomicAdd(&grad_dev[atomC * 3 + 0], sh_sumC_agg[u][0]);
      atomicAdd(&grad_dev[atomC * 3 + 1], sh_sumC_agg[u][1]);
      atomicAdd(&grad_dev[atomC * 3 + 2], sh_sumC_agg[u][2]);
    }
  }
}

// Float32 bar_X variant: reduces global memory bandwidth for the dominant DF 3c2e derivative contraction.
// Accumulation is still in FP64 for stability.
template <int NROOTS>
__global__ void KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar(
    const int32_t* spAB_arr,   // [n_spAB] AO shell-pair indices in this (la,lb) class
    int n_spAB,
    const int32_t* spCD,       // [ntasks] aux shell-pair indices
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const float* bar_X_flat,
    const int32_t* shell_atom,  // combined AO+aux shell->atom map (length: nAOshells+nAuxShells)
    double* grad_dev) {         // [natm*3] gradient accumulator (atomicAdd target)
  const int t   = static_cast<int>(blockIdx.x);  // CD task index
  const int iAB = static_cast<int>(blockIdx.y);  // AB class index
  if (t >= ntasks || iAB >= n_spAB) return;

  const int spAB   = static_cast<int>(spAB_arr[iAB]);
  const int spCD_i = static_cast<int>(spCD[t]);

  // --- Shared memory (identical layout to KernelDFInt3c2eDerivContractedCartBatch) ---
  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shB_lx[kNcartMax], shB_ly[kNcartMax], shB_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][9];
  __shared__ float sh_bar[kBarCacheMax];

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]);
  const int b0 = static_cast<int>(shell_ao_start[shellB]);
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int nprimB = static_cast<int>(shell_nprim[shellB]);
  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sB = static_cast<int>(shell_prim_start[shellB]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Bx = shell_cx[shellB];
  const double By = shell_cy[shellB];
  const double Bz = shell_cz[shellB];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lb, shB_lx, shB_ly, shB_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * ABx;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * ABy;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * ABz;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / (nB * nC);
      const int rem = idx - ia * (nB * nC);
      const int ib = rem / nC;
      const int ic = rem - ib * nC;
      const int row_idx = (a0 + ia) * nao + (b0 + ib);
      const int col_idx = c0 + ic;
      sh_bar[idx] = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) acc[i] = 0.0;

  const int nmax = la + lb + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB_prim;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const int ia_prim = iAB_prim / nprimB;
    const int ib_prim = iAB_prim - ia_prim * nprimB;
    const double aexp = prim_exp[sA + ia_prim];
    const double bexp = prim_exp[sB + ib_prim];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
          else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        float bar_f = 0.0f;
        if (cache_bar) {
          bar_f = sh_bar[idx];
        } else {
          const int ia = idx / (nB * nC);
          const int rem = idx - ia * (nB * nC);
          const int ib = rem / nC;
          const int ic = rem - ib * nC;
          const int row_idx = (a0 + ia) * nao + (b0 + ib);
          const int col_idx = c0 + ic;
          bar_f = bar_X_flat[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        }
        if (bar_f == 0.0f) continue;

        const int ia = idx / (nB * nC);
        const int rem = idx - ia * (nB * nC);
        const int ib = rem / nC;
        const int ic = rem - ib * nC;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int ibx = static_cast<int>(shB_lx[ib]);
        const int iby = static_cast<int>(shB_ly[ib]);
        const int ibz = static_cast<int>(shB_lz[ib]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz, sh_zij_pow);

        const double bar_scale = static_cast<double>(bar_f) * scale;

        // Center A derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, ibx, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, ibx, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, iby, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, iby, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, ibz, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, ibz, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center B derivatives.
        const double Ix_m_B = (ibx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx - 1, icx, sh_xij_pow) : 0.0;
        const double Ix_p_B = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx + 1, icx, sh_xij_pow);
        const double dIx_B = (-static_cast<double>(ibx)) * Ix_m_B + (2.0 * bexp) * Ix_p_B;
        acc[3] += bar_scale * (dIx_B * Iy * Iz);

        const double Iy_m_B = (iby > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby - 1, icy, sh_yij_pow) : 0.0;
        const double Iy_p_B = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby + 1, icy, sh_yij_pow);
        const double dIy_B = (-static_cast<double>(iby)) * Iy_m_B + (2.0 * bexp) * Iy_p_B;
        acc[4] += bar_scale * (Ix * dIy_B * Iz);

        const double Iz_m_B = (ibz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz - 1, icz, sh_zij_pow) : 0.0;
        const double Iz_p_B = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz + 1, icz, sh_zij_pow);
        const double dIz_B = (-static_cast<double>(ibz)) * Iz_m_B + (2.0 * bexp) * Iz_p_B;
        acc[5] += bar_scale * (Ix * Iy * dIz_B);

        // Center C derivatives (aux).
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, ibx, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[6] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, iby, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[7] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, ibz, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[8] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  // Warp reduction then block reduction over warps (lane 0 only).
  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 9; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const double fac = (shellA != shellB) ? 2.0 : 1.0;
    const int atomA = static_cast<int>(shell_atom[shellA]);
    const int atomB = static_cast<int>(shell_atom[shellB]);
    const int atomC = static_cast<int>(shell_atom[shellC]);
    atomicAdd(&grad_dev[atomA * 3 + 0], fac * sum[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], fac * sum[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], fac * sum[2]);
    atomicAdd(&grad_dev[atomB * 3 + 0], fac * sum[3]);
    atomicAdd(&grad_dev[atomB * 3 + 1], fac * sum[4]);
    atomicAdd(&grad_dev[atomB * 3 + 2], fac * sum[5]);
    atomicAdd(&grad_dev[atomC * 3 + 0], fac * sum[6]);
    atomicAdd(&grad_dev[atomC * 3 + 1], fac * sum[7]);
    atomicAdd(&grad_dev[atomC * 3 + 2], fac * sum[8]);
  }
}

template <int NROOTS>
__global__ void KernelDFMetric2c2eDerivContractedCartBatch(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    double* out) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;

  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][6];
  __shared__ double sh_bar[kBarCacheMax];

  const int spCD_i = static_cast<int>(spCD[t]);

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);  // dummy s at origin
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  (void)shellB;

  const int nA = ncart(la);
  const int nC = ncart(lc);
  const int nElem = nA * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]) - nao;
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);  // == nprimA (nprimB=1)
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);  // == nprimC (nprimD=1)

  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    // Dummy shell at origin => AB vector == A.
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * Ax;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * Ay;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * Az;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / nC;
      const int ic = idx - ia * nC;
      const int row_idx = a0 + ia;
      const int col_idx = c0 + ic;
      sh_bar[idx] = bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) acc[i] = 0.0;

  const int nmax = la + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB = static_cast<int>(u / static_cast<int64_t>(nprimCD));  // primitive index for A
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB) * static_cast<int64_t>(nprimCD));  // primitive index for C
    const int ki = baseAB + iAB;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px = pair_Px[ki];
    const double Py = pair_Py[ki];
    const double Pz = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const double aexp = prim_exp[sA + iAB];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px - Qx;
    const double dy = Py - Qy;
    const double dz = Pz - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px - Ax; QC = Qx - Cx; PQd = Qx - Px; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py - Ay; QC = Qy - Cy; PQd = Qy - Py; G_target = sh_Gy[warp_id]; }
          else { PA = Pz - Az; QC = Qz - Cz; PQd = Qz - Pz; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        double bar = 0.0;
        if (cache_bar) {
          bar = sh_bar[idx];
        } else {
          const int ia = idx / nC;
          const int ic = idx - ia * nC;
          const int row_idx = a0 + ia;
          const int col_idx = c0 + ic;
          bar = bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        }
        if (bar == 0.0) continue;

        const int ia = idx / nC;
        const int ic = idx - ia * nC;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz, sh_zij_pow);

        const double bar_scale = bar * scale;

        // Center A (P) derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, 0, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, 0, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, 0, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, 0, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, 0, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, 0, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center C (Q) derivatives.
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[3] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[4] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[5] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 6; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 6; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const int out0 = t * 6;
#pragma unroll
    for (int i = 0; i < 6; ++i) out[out0 + i] = sum[i];
  }
}

// ──────────────────────────────────────────────────────────────────────
// KernelDFMetric2c2eDerivContractedCartAllSPAtomGrad
//
// Like KernelDFMetric2c2eDerivContractedCartBatch but:
//   • takes an array spAB_arr[n_spAB] instead of a single spAB
//   • uses a 2D grid: blockIdx.x = CD task,  blockIdx.y = AB index
//   • writes results directly to grad_dev[natm*3] via atomicAdd
//   • processes the FULL matrix (no upper-triangle restriction);
//     the caller is responsible for supplying the correct adjoint bar_V
//     that already accounts for symmetry (or passing the full matrix).
// ──────────────────────────────────────────────────────────────────────
template <int NROOTS>
__global__ void KernelDFMetric2c2eDerivContractedCartAllSPAtomGrad(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev) {
  const int t   = static_cast<int>(blockIdx.x);   // CD task index
  const int iAB = static_cast<int>(blockIdx.y);   // AB class index
  if (t >= ntasks || iAB >= n_spAB) return;

  const int spAB   = static_cast<int>(spAB_arr[iAB]);
  const int spCD_i = static_cast<int>(spCD[t]);

  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][6];
  __shared__ double sh_bar[kBarCacheMax];

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);

  const int nA = ncart(la);
  const int nC = ncart(lc);
  const int nElem = nA * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]) - nao;
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * Ax;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * Ay;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * Az;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / nC;
      const int ic = idx - ia * nC;
      const int row_idx = a0 + ia;
      const int col_idx = c0 + ic;
      sh_bar[idx] = bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) acc[i] = 0.0;

  const int nmax = la + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB_prim;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px_ = pair_Px[ki];
    const double Py_ = pair_Py[ki];
    const double Pz_ = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const double aexp = prim_exp[sA + iAB_prim];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px_ - Qx;
    const double dy = Py_ - Qy;
    const double dz = Pz_ - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

    double roots[NROOTS];
    double weights[NROOTS];
    if (lane == 0) {
      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
    }

    for (int r = 0; r < NROOTS; ++r) {
      double x = (lane == 0) ? roots[r] : 0.0;
      double w = (lane == 0) ? weights[r] : 0.0;
      x = __shfl_sync(0xffffffff, x, 0);
      w = __shfl_sync(0xffffffff, w, 0);

      {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;
        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;
        if (lane < 3) {
          double PA, QC, PQd;
          double* G_target;
          if (lane == 0) { PA = Px_ - Ax; QC = Qx - Cx; PQd = Qx - Px_; G_target = sh_Gx[warp_id]; }
          else if (lane == 1) { PA = Py_ - Ay; QC = Qy - Cy; PQd = Qy - Py_; G_target = sh_Gy[warp_id]; }
          else { PA = Pz_ - Az; QC = Qz - Cz; PQd = Qz - Pz_; G_target = sh_Gz[warp_id]; }
          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
        }
      }

      const double scale = base * w;
      __syncwarp();

      for (int idx = lane; idx < nElem; idx += 32) {
        double bar = 0.0;
        if (cache_bar) {
          bar = sh_bar[idx];
        } else {
          const int ia = idx / nC;
          const int ic = idx - ia * nC;
          const int row_idx = a0 + ia;
          const int col_idx = c0 + ic;
          bar = bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
        }
        if (bar == 0.0) continue;

        const int ia = idx / nC;
        const int ic = idx - ia * nC;

        const int iax = static_cast<int>(shA_lx[ia]);
        const int iay = static_cast<int>(shA_ly[ia]);
        const int iaz = static_cast<int>(shA_lz[ia]);
        const int icx = static_cast<int>(shC_lx[ic]);
        const int icy = static_cast<int>(shC_ly[ic]);
        const int icz = static_cast<int>(shC_lz[ic]);

        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx, sh_xij_pow);
        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy, sh_yij_pow);
        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz, sh_zij_pow);

        const double bar_scale = bar * scale;

        // Center A (P) derivatives.
        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, 0, icx, sh_xij_pow) : 0.0;
        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, 0, icx, sh_xij_pow);
        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
        acc[0] += bar_scale * (dIx_A * Iy * Iz);

        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, 0, icy, sh_yij_pow) : 0.0;
        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, 0, icy, sh_yij_pow);
        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
        acc[1] += bar_scale * (Ix * dIy_A * Iz);

        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, 0, icz, sh_zij_pow) : 0.0;
        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, 0, icz, sh_zij_pow);
        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
        acc[2] += bar_scale * (Ix * Iy * dIz_A);

        // Center C (Q) derivatives.
        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx - 1, sh_xij_pow) : 0.0;
        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx + 1, sh_xij_pow);
        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
        acc[3] += bar_scale * (dIx_C * Iy * Iz);

        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy - 1, sh_yij_pow) : 0.0;
        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy + 1, sh_yij_pow);
        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
        acc[4] += bar_scale * (Ix * dIy_C * Iz);

        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz - 1, sh_zij_pow) : 0.0;
        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz + 1, sh_zij_pow);
        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
        acc[5] += bar_scale * (Ix * Iy * dIz_C);
      }

      __syncwarp();
    }
  }

  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 6; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 6; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const int atomA = static_cast<int>(shell_atom[shellA]);
    const int atomC = static_cast<int>(shell_atom[shellC]);
    atomicAdd(&grad_dev[atomA * 3 + 0], sum[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], sum[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], sum[2]);
    atomicAdd(&grad_dev[atomC * 3 + 0], sum[3]);
    atomicAdd(&grad_dev[atomC * 3 + 1], sum[4]);
    atomicAdd(&grad_dev[atomC * 3 + 2], sum[5]);
  }
}

// ──────────────────────────────────────────────────────────────────────
// KernelDFMetric2c2eDerivContractedCartAllSPAtomGradTril
//
// Variant of KernelDFMetric2c2eDerivContractedCartAllSPAtomGrad that:
// - skips upper-triangle shell pairs (shellC > shellA)
// - scales off-diagonal shell pairs by 2.0 to account for symmetry
//
// This matches the common "compute lower triangle + factor 2" contraction
// pattern without requiring per-shell task lists on the host.
// ──────────────────────────────────────────────────────────────────────
template <int NROOTS>
__global__ void KernelDFMetric2c2eDerivContractedCartAllSPAtomGradTril(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev) {
  const int t   = static_cast<int>(blockIdx.x);   // CD task index
  const int iAB = static_cast<int>(blockIdx.y);   // AB class index
  if (t >= ntasks || iAB >= n_spAB) return;

  const int spAB   = static_cast<int>(spAB_arr[iAB]);
  const int spCD_i = static_cast<int>(spCD[t]);

  __shared__ int8_t shA_lx[kNcartMax], shA_ly[kNcartMax], shA_lz[kNcartMax];
  __shared__ int8_t shC_lx[kNcartMax], shC_ly[kNcartMax], shC_lz[kNcartMax];
  __shared__ double sh_xij_pow[kLMaxD + 1], sh_yij_pow[kLMaxD + 1], sh_zij_pow[kLMaxD + 1];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][kGSizeD];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][kGSizeD];

  __shared__ double sh_warp_sum[kMaxWarpsPerBlock][6];
  __shared__ double sh_bar[kBarCacheMax];

  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellC = static_cast<int>(sp_A[spCD_i]);
  if (shellC > shellA) return;
  const double fac = (shellC == shellA) ? 1.0 : 2.0;

  const int nA = ncart(la);
  const int nC = ncart(lc);
  const int nElem = nA * nC;

  const int a0 = static_cast<int>(shell_ao_start[shellA]) - nao;
  const int c0 = static_cast<int>(shell_ao_start[shellC]) - nao;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD_i]);
  const int nprimAB = static_cast<int>(sp_npair[spAB]);
  const int nprimCD = static_cast<int>(sp_npair[spCD_i]);

  const int sA = static_cast<int>(shell_prim_start[shellA]);
  const int sC = static_cast<int>(shell_prim_start[shellC]);

  const double Ax = shell_cx[shellA];
  const double Ay = shell_cy[shellA];
  const double Az = shell_cz[shellA];
  const double Cx = shell_cx[shellC];
  const double Cy = shell_cy[shellC];
  const double Cz = shell_cz[shellC];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, shA_lx, shA_ly, shA_lz);
    fill_cart_comp(lc, shC_lx, shC_ly, shC_lz);

    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int p = 1; p <= kLMaxD; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * Ax;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * Ay;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * Az;
    }
  }

  const bool cache_bar = (nElem > 0 && nElem <= kBarCacheMax);
  if (cache_bar) {
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      const int ia = idx / nC;
      const int ic = idx - ia * nC;
      const int row_idx = a0 + ia;
      const int col_idx = c0 + ic;
      sh_bar[idx] = fac * bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
    }
  }

  __syncthreads();

  // Block-level bar_X early-exit: skip block if max|sh_bar| < threshold.
  if (cache_bar) {
    double tmax = 0.0;
    for (int idx = static_cast<int>(threadIdx.x); idx < nElem; idx += static_cast<int>(blockDim.x)) {
      double v = static_cast<double>(sh_bar[idx]);
      if (v < 0.0) v = -v;
      if (v > tmax) tmax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      double other = __shfl_down_sync(0xffffffff, tmax, offset);
      if (other > tmax) tmax = other;
    }
    {
      const int lane_ = static_cast<int>(threadIdx.x) & 31;
      const int warp_ = static_cast<int>(threadIdx.x) >> 5;
      if (lane_ == 0) sh_warp_sum[warp_][0] = tmax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double bmax = sh_warp_sum[0][0];
      const int nw = static_cast<int>(blockDim.x) >> 5;
      for (int w = 1; w < nw; ++w) {
        if (sh_warp_sum[w][0] > bmax) bmax = sh_warp_sum[w][0];
      }
      sh_warp_sum[0][0] = bmax;
    }
    __syncthreads();
    if (sh_warp_sum[0][0] < 1e-14) return;
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;

  double acc[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) acc[i] = 0.0;

  const int nmax = la + 1;
  const int mmax = lc + 1;
  const int64_t nTot = static_cast<int64_t>(nprimAB) * static_cast<int64_t>(nprimCD);

  for (int64_t u = static_cast<int64_t>(warp_id); u < nTot; u += static_cast<int64_t>(warps)) {
    const int iAB_prim = static_cast<int>(u / static_cast<int64_t>(nprimCD));
    const int iCD = static_cast<int>(u - static_cast<int64_t>(iAB_prim) * static_cast<int64_t>(nprimCD));
    const int ki = baseAB + iAB_prim;
    const int kj = baseCD + iCD;

    const double p = pair_eta[ki];
    const double q = pair_eta[kj];
    const double Px_ = pair_Px[ki];
    const double Py_ = pair_Py[ki];
    const double Pz_ = pair_Pz[ki];
    const double Qx = pair_Px[kj];
    const double Qy = pair_Py[kj];
    const double Qz = pair_Pz[kj];
    const double cKab = pair_cK[ki];
    const double cKcd = pair_cK[kj];

    const double aexp = prim_exp[sA + iAB_prim];
    const double cexp = prim_exp[sC + iCD];

    const double denom = p + q;
    const double inv_denom = 1.0 / denom;
    const double dx = Px_ - Qx;
    const double dy = Py_ - Qy;
    const double dz = Pz_ - Qz;
    const double PQ2 = dx * dx + dy * dy + dz * dz;
    const double omega = p * q * inv_denom;
    const double T = omega * PQ2;

	    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

	    double roots[NROOTS];
	    double weights[NROOTS];
	    if (lane == 0) {
	      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);
	    }
	    for (int r = 0; r < NROOTS; ++r) {
	      double x = (lane == 0) ? roots[r] : 0.0;
	      double w = (lane == 0) ? weights[r] : 0.0;
	      x = __shfl_sync(0xffffffff, x, 0);
	      w = __shfl_sync(0xffffffff, w, 0);

	      {
	        const double B0 = x * 0.5 * inv_denom;
	        const double B1 = (1.0 - x) * 0.5 / p + B0;
	        const double B1p = (1.0 - x) * 0.5 / q + B0;
	        const double q_over = q * inv_denom;
	        const double p_over = p * inv_denom;
	        if (lane < 3) {
	          double PA, QC, PQd;
	          double* G_target;
	          if (lane == 0) { PA = Px_ - Ax; QC = Qx - Cx; PQd = Qx - Px_; G_target = sh_Gx[warp_id]; }
	          else if (lane == 1) { PA = Py_ - Ay; QC = Qy - Cy; PQd = Qy - Py_; G_target = sh_Gy[warp_id]; }
	          else { PA = Pz_ - Az; QC = Qz - Cz; PQd = Qz - Pz_; G_target = sh_Gz[warp_id]; }
	          compute_G_d(G_target, nmax, mmax, PA + q_over * x * PQd, QC - p_over * x * PQd, B0, B1, B1p);
	        }
	      }

	      const double scale = base * w;
	      __syncwarp();

	      for (int idx = lane; idx < nElem; idx += 32) {
	        double bar = 0.0;
	        if (cache_bar) {
	          bar = sh_bar[idx];
	        } else {
	          const int ia = idx / nC;
	          const int ic = idx - ia * nC;
	          const int row_idx = a0 + ia;
	          const int col_idx = c0 + ic;
	          bar = fac * bar_V[static_cast<int64_t>(row_idx) * static_cast<int64_t>(naux) + static_cast<int64_t>(col_idx)];
	        }
	        if (bar == 0.0) continue;

	        const int ia = idx / nC;
	        const int ic = idx - ia * nC;

	        const int iax = static_cast<int>(shA_lx[ia]);
	        const int iay = static_cast<int>(shA_ly[ia]);
	        const int iaz = static_cast<int>(shA_lz[ia]);
	        const int icx = static_cast<int>(shC_lx[ic]);
	        const int icy = static_cast<int>(shC_ly[ic]);
	        const int icz = static_cast<int>(shC_lz[ic]);

	        const double Ix = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx, sh_xij_pow);
	        const double Iy = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy, sh_yij_pow);
	        const double Iz = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz, sh_zij_pow);

	        const double bar_scale = bar * scale;

	        // Center A (P) derivatives.
	        const double Ix_m_A = (iax > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax - 1, 0, icx, sh_xij_pow) : 0.0;
	        const double Ix_p_A = shift_from_G_ld0_d(sh_Gx[warp_id], iax + 1, 0, icx, sh_xij_pow);
	        const double dIx_A = (-static_cast<double>(iax)) * Ix_m_A + (2.0 * aexp) * Ix_p_A;
	        acc[0] += bar_scale * (dIx_A * Iy * Iz);

	        const double Iy_m_A = (iay > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay - 1, 0, icy, sh_yij_pow) : 0.0;
	        const double Iy_p_A = shift_from_G_ld0_d(sh_Gy[warp_id], iay + 1, 0, icy, sh_yij_pow);
	        const double dIy_A = (-static_cast<double>(iay)) * Iy_m_A + (2.0 * aexp) * Iy_p_A;
	        acc[1] += bar_scale * (Ix * dIy_A * Iz);

	        const double Iz_m_A = (iaz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz - 1, 0, icz, sh_zij_pow) : 0.0;
	        const double Iz_p_A = shift_from_G_ld0_d(sh_Gz[warp_id], iaz + 1, 0, icz, sh_zij_pow);
	        const double dIz_A = (-static_cast<double>(iaz)) * Iz_m_A + (2.0 * aexp) * Iz_p_A;
	        acc[2] += bar_scale * (Ix * Iy * dIz_A);

	        // Center C (Q) derivatives.
	        const double Ix_m_C = (icx > 0) ? shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx - 1, sh_xij_pow) : 0.0;
	        const double Ix_p_C = shift_from_G_ld0_d(sh_Gx[warp_id], iax, 0, icx + 1, sh_xij_pow);
	        const double dIx_C = (-static_cast<double>(icx)) * Ix_m_C + (2.0 * cexp) * Ix_p_C;
	        acc[3] += bar_scale * (dIx_C * Iy * Iz);

	        const double Iy_m_C = (icy > 0) ? shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy - 1, sh_yij_pow) : 0.0;
	        const double Iy_p_C = shift_from_G_ld0_d(sh_Gy[warp_id], iay, 0, icy + 1, sh_yij_pow);
	        const double dIy_C = (-static_cast<double>(icy)) * Iy_m_C + (2.0 * cexp) * Iy_p_C;
	        acc[4] += bar_scale * (Ix * dIy_C * Iz);

	        const double Iz_m_C = (icz > 0) ? shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz - 1, sh_zij_pow) : 0.0;
	        const double Iz_p_C = shift_from_G_ld0_d(sh_Gz[warp_id], iaz, 0, icz + 1, sh_zij_pow);
	        const double dIz_C = (-static_cast<double>(icz)) * Iz_m_C + (2.0 * cexp) * Iz_p_C;
	        acc[5] += bar_scale * (Ix * Iy * dIz_C);
	      }

	      __syncwarp();
	    }
	  }

  warp_reduce_sum_arr(acc);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 6; ++i) sh_warp_sum[warp_id][i] = acc[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double sum[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) sum[i] = 0.0;
    for (int w = 0; w < warps; ++w) {
#pragma unroll
      for (int i = 0; i < 6; ++i) sum[i] += sh_warp_sum[w][i];
    }
    const int atomA = static_cast<int>(shell_atom[shellA]);
    const int atomC = static_cast<int>(shell_atom[shellC]);
    atomicAdd(&grad_dev[atomA * 3 + 0], sum[0]);
    atomicAdd(&grad_dev[atomA * 3 + 1], sum[1]);
    atomicAdd(&grad_dev[atomA * 3 + 2], sum[2]);
    atomicAdd(&grad_dev[atomC * 3 + 0], sum[3]);
    atomicAdd(&grad_dev[atomC * 3 + 1], sum[4]);
    atomicAdd(&grad_dev[atomC * 3 + 2], sum[5]);
  }
}

static inline int df_nroots_from_L(int L_total) {
  return ((L_total + 1) / 2) + 1;
}

static inline cudaError_t launch_df_int3c2e_deriv_cart(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    double* out,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lb + lc);
  switch (nroots) {
    case 1:
      KernelDFInt3c2eDerivContractedCartBatch<1><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(
          spAB,
          spCD,
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
          shell_ao_start,
          prim_exp,
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          nao,
          naux,
          la,
          lb,
          lc,
          bar_X_flat,
          out);
      break;
    case 2:
      KernelDFInt3c2eDerivContractedCartBatch<2><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 3:
      KernelDFInt3c2eDerivContractedCartBatch<3><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 4:
      KernelDFInt3c2eDerivContractedCartBatch<4><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 5:
      KernelDFInt3c2eDerivContractedCartBatch<5><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 6:
      KernelDFInt3c2eDerivContractedCartBatch<6><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 7:
      KernelDFInt3c2eDerivContractedCartBatch<7><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 8:
      KernelDFInt3c2eDerivContractedCartBatch<8><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 9:
      KernelDFInt3c2eDerivContractedCartBatch<9><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                           spCD,
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
                                                                                                           shell_ao_start,
                                                                                                           prim_exp,
                                                                                                           pair_eta,
                                                                                                           pair_Px,
                                                                                                           pair_Py,
                                                                                                           pair_Pz,
                                                                                                           pair_cK,
                                                                                                           nao,
                                                                                                           naux,
                                                                                                           la,
                                                                                                           lb,
                                                                                                           lc,
                                                                                                           bar_X_flat,
                                                                                                           out);
      break;
    case 10:
      KernelDFInt3c2eDerivContractedCartBatch<10><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                            spCD,
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
                                                                                                            shell_ao_start,
                                                                                                            prim_exp,
                                                                                                            pair_eta,
                                                                                                            pair_Px,
                                                                                                            pair_Py,
                                                                                                            pair_Pz,
                                                                                                            pair_cK,
                                                                                                            nao,
                                                                                                            naux,
                                                                                                            la,
                                                                                                            lb,
                                                                                                            lc,
                                                                                                            bar_X_flat,
                                                                                                            out);
      break;
    case 11:
      KernelDFInt3c2eDerivContractedCartBatch<11><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                            spCD,
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
                                                                                                            shell_ao_start,
                                                                                                            prim_exp,
                                                                                                            pair_eta,
                                                                                                            pair_Px,
                                                                                                            pair_Py,
                                                                                                            pair_Pz,
                                                                                                            pair_cK,
                                                                                                            nao,
                                                                                                            naux,
                                                                                                            la,
                                                                                                            lb,
                                                                                                            lc,
                                                                                                            bar_X_flat,
                                                                                                            out);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_metric_2c2e_deriv_cart(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    double* out,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lc);
  switch (nroots) {
    case 1:
      KernelDFMetric2c2eDerivContractedCartBatch<1><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(
          spAB,
          spCD,
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
          shell_ao_start,
          prim_exp,
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          nao,
          naux,
          la,
          lc,
          bar_V,
          out);
      break;
    case 2:
      KernelDFMetric2c2eDerivContractedCartBatch<2><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 3:
      KernelDFMetric2c2eDerivContractedCartBatch<3><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 4:
      KernelDFMetric2c2eDerivContractedCartBatch<4><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 5:
      KernelDFMetric2c2eDerivContractedCartBatch<5><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 6:
      KernelDFMetric2c2eDerivContractedCartBatch<6><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 7:
      KernelDFMetric2c2eDerivContractedCartBatch<7><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 8:
      KernelDFMetric2c2eDerivContractedCartBatch<8><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 9:
      KernelDFMetric2c2eDerivContractedCartBatch<9><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                              spCD,
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
                                                                                                              shell_ao_start,
                                                                                                              prim_exp,
                                                                                                              pair_eta,
                                                                                                              pair_Px,
                                                                                                              pair_Py,
                                                                                                              pair_Pz,
                                                                                                              pair_cK,
                                                                                                              nao,
                                                                                                              naux,
                                                                                                              la,
                                                                                                              lc,
                                                                                                              bar_V,
                                                                                                              out);
      break;
    case 10:
      KernelDFMetric2c2eDerivContractedCartBatch<10><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                               spCD,
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
                                                                                                               shell_ao_start,
                                                                                                               prim_exp,
                                                                                                               pair_eta,
                                                                                                               pair_Px,
                                                                                                               pair_Py,
                                                                                                               pair_Pz,
                                                                                                               pair_cK,
                                                                                                               nao,
                                                                                                               naux,
                                                                                                               la,
                                                                                                               lc,
                                                                                                               bar_V,
                                                                                                               out);
      break;
    case 11:
      KernelDFMetric2c2eDerivContractedCartBatch<11><<<static_cast<unsigned int>(ntasks), threads, 0, stream>>>(spAB,
                                                                                                               spCD,
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
                                                                                                               shell_ao_start,
                                                                                                               prim_exp,
                                                                                                               pair_eta,
                                                                                                               pair_Px,
                                                                                                               pair_Py,
                                                                                                               pair_Pz,
                                                                                                               pair_cK,
                                                                                                               nao,
                                                                                                               naux,
                                                                                                               la,
                                                                                                               lc,
                                                                                                               bar_V,
                                                                                                               out);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_int3c2e_deriv_cart_allsp_atomgrad(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lb + lc);
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
    case 1:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<1><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 2:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<2><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 3:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<3><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 4:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<4><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 5:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<5><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 6:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<6><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 7:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<7><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 8:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<8><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 9:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<9><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 10:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<10><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 11:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGrad<11><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_int3c2e_deriv_cart_allsp_atomgrad_sphbar_qmn(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lb + lc);
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
    case 1:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<1><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 2:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<2><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 3:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<3><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 4:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<4><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 5:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<5><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 6:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<6><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 7:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<7><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 8:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<8><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 9:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<9><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 10:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<10><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    case 11:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmn<11><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_int3c2e_deriv_cart_allsp_atomgrad_abtile(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  if (cd_tile <= 0) return cudaErrorInvalidValue;
  const int nroots = df_nroots_from_L(la + lb + lc);
  const int ntiles = (ntasks + cd_tile - 1) / cd_tile;
  const dim3 grid(static_cast<unsigned int>(ntiles), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
#define LAUNCH_3C_ABTILE(NR) \
    case NR: \
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradABTile<NR><<<grid, threads, 0, stream>>>( \
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start, \
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, \
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, cd_tile, grad_dev); \
      break;
    LAUNCH_3C_ABTILE(1)
    LAUNCH_3C_ABTILE(2)
    LAUNCH_3C_ABTILE(3)
    LAUNCH_3C_ABTILE(4)
    LAUNCH_3C_ABTILE(5)
    LAUNCH_3C_ABTILE(6)
    LAUNCH_3C_ABTILE(7)
    LAUNCH_3C_ABTILE(8)
    LAUNCH_3C_ABTILE(9)
    LAUNCH_3C_ABTILE(10)
    LAUNCH_3C_ABTILE(11)
#undef LAUNCH_3C_ABTILE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_int3c2e_deriv_cart_allsp_atomgrad_sphbar_qmn_abtile(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  if (cd_tile <= 0) return cudaErrorInvalidValue;
  const int nroots = df_nroots_from_L(la + lb + lc);
  const int ntiles = (ntasks + cd_tile - 1) / cd_tile;
  const dim3 grid(static_cast<unsigned int>(ntiles), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
#define LAUNCH_3C_SPH_ABTILE(NR) \
    case NR: \
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradSphBarQmnABTile<NR><<<grid, threads, 0, stream>>>( \
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start, \
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, \
          nao, naux, nao_sph, la, lb, lc, bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, cd_tile, grad_dev); \
      break;
    LAUNCH_3C_SPH_ABTILE(1)
    LAUNCH_3C_SPH_ABTILE(2)
    LAUNCH_3C_SPH_ABTILE(3)
    LAUNCH_3C_SPH_ABTILE(4)
    LAUNCH_3C_SPH_ABTILE(5)
    LAUNCH_3C_SPH_ABTILE(6)
    LAUNCH_3C_SPH_ABTILE(7)
    LAUNCH_3C_SPH_ABTILE(8)
    LAUNCH_3C_SPH_ABTILE(9)
    LAUNCH_3C_SPH_ABTILE(10)
    LAUNCH_3C_SPH_ABTILE(11)
#undef LAUNCH_3C_SPH_ABTILE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_int3c2e_deriv_cart_allsp_atomgrad_f32bar(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const float* bar_X_flat,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lb + lc);
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
    case 1:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<1><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 2:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<2><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 3:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<3><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 4:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<4><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 5:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<5><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 6:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<6><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 7:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<7><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 8:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<8><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 9:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<9><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 10:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<10><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    case 11:
      KernelDFInt3c2eDerivContractedCartAllSPAtomGradF32Bar<11><<<grid, threads, 0, stream>>>(
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start,
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
          nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_metric_2c2e_deriv_cart_allsp_atomgrad(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lc);
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
    // Aux basis angular momentum ≤ 5 → la+lc ≤ 10 → nroots ≤ 6.
    // Instantiate only 1–6 to keep compile time manageable.
#define LAUNCH_2C_ALLSP(NR) \
    case NR: \
      KernelDFMetric2c2eDerivContractedCartAllSPAtomGrad<NR><<<grid, threads, 0, stream>>>( \
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start, \
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, \
          nao, naux, la, lc, bar_V, shell_atom, grad_dev); \
      break;
    LAUNCH_2C_ALLSP(1)
    LAUNCH_2C_ALLSP(2)
    LAUNCH_2C_ALLSP(3)
    LAUNCH_2C_ALLSP(4)
    LAUNCH_2C_ALLSP(5)
    LAUNCH_2C_ALLSP(6)
#undef LAUNCH_2C_ALLSP
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

static inline cudaError_t launch_df_metric_2c2e_deriv_cart_allsp_atomgrad_tril(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  const int nroots = df_nroots_from_L(la + lc);
  const dim3 grid(static_cast<unsigned int>(ntasks), static_cast<unsigned int>(n_spAB));
  switch (nroots) {
#define LAUNCH_2C_ALLSP_TRIL(NR) \
    case NR: \
      KernelDFMetric2c2eDerivContractedCartAllSPAtomGradTril<NR><<<grid, threads, 0, stream>>>( \
          spAB_arr, n_spAB, spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, \
          shell_cx, shell_cy, shell_cz, shell_prim_start, shell_nprim, shell_ao_start, \
          prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, \
          nao, naux, la, lc, bar_V, shell_atom, grad_dev); \
      break;
    LAUNCH_2C_ALLSP_TRIL(1)
    LAUNCH_2C_ALLSP_TRIL(2)
    LAUNCH_2C_ALLSP_TRIL(3)
    LAUNCH_2C_ALLSP_TRIL(4)
    LAUNCH_2C_ALLSP_TRIL(5)
    LAUNCH_2C_ALLSP_TRIL(6)
#undef LAUNCH_2C_ALLSP_TRIL
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_launch_stream(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    double* out,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart(
      spAB,
      spCD,
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
      shell_ao_start,
      prim_exp,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      nao,
      naux,
      la,
      lb,
      lc,
      bar_X_flat,
      out,
      stream,
      threads);
}

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart_allsp_atomgrad(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart_allsp_atomgrad_sphbar_qmn(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, nao_sph, la, lb, lc,
      bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart_allsp_atomgrad_abtile(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, la, lb, lc, bar_X_flat, shell_atom, cd_tile, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int nao_sph,
    int la,
    int lb,
    int lc,
    const double* bar_X_sph_Qmn,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_atom,
    int cd_tile,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart_allsp_atomgrad_sphbar_qmn_abtile(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, nao_sph, la, lb, lc,
      bar_X_sph_Qmn, shell_ao_start_sph, shell_atom, cd_tile, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_f32bar_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const float* bar_X_flat,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_int3c2e_deriv_cart_allsp_atomgrad_f32bar(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, la, lb, lc, bar_X_flat, shell_atom, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_metric_2c2e_deriv_contracted_cart_launch_stream(
    int32_t spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    double* out,
    cudaStream_t stream,
    int threads) {
  return launch_df_metric_2c2e_deriv_cart(
      spAB,
      spCD,
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
      shell_ao_start,
      prim_exp,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      nao,
      naux,
      la,
      lc,
      bar_V,
      out,
      stream,
      threads);
}

extern "C" cudaError_t cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_metric_2c2e_deriv_cart_allsp_atomgrad(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, la, lc, bar_V, shell_atom, grad_dev,
      stream, threads);
}

extern "C" cudaError_t cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* spCD,
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
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,
    const int32_t* shell_atom,
    double* grad_dev,
    cudaStream_t stream,
    int threads) {
  return launch_df_metric_2c2e_deriv_cart_allsp_atomgrad_tril(
      spAB_arr, n_spAB, spCD, ntasks,
      sp_A, sp_B, sp_pair_start, sp_npair,
      shell_cx, shell_cy, shell_cz,
      shell_prim_start, shell_nprim, shell_ao_start,
      prim_exp, pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
      nao, naux, la, lc, bar_V, shell_atom, grad_dev,
      stream, threads);
}
