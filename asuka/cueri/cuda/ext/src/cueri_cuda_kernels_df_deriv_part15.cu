// Auto-split from cueri_cuda_kernels_df_deriv.cu (part 15/17: KernelDFMetric2c2eDerivContractedCartBatch..KernelDFMetric2c2eDerivContractedCartBatch)
// Do not edit — regenerate with split_large_kernels.py

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
template <typename T>
__device__ __forceinline__ double df_bar_cart_from_sph_qmn_t(
    const T* __restrict__ bar_X_sph_Qmn,  // [naux, nao_sph, nao_sph] C-order
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
      acc += (ta * tb) * static_cast<double>(bar_X_sph_Qmn[base + static_cast<int64_t>(j)]);
    }
  }
  return acc;
}

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
  return df_bar_cart_from_sph_qmn_t<double>(bar_X_sph_Qmn, q, ia_cart, ib_cart, la, lb, a0_sph, b0_sph, nao_sph);
}

__device__ __forceinline__ double df_bar_cart_from_sph_qmn_f32(
    const float* __restrict__ bar_X_sph_Qmn,  // [naux, nao_sph, nao_sph] C-order
    int q,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  return df_bar_cart_from_sph_qmn_t<float>(bar_X_sph_Qmn, q, ia_cart, ib_cart, la, lb, a0_sph, b0_sph, nao_sph);
}

template <typename T>
__device__ __forceinline__ double df_bar_cart_from_sph_qmn_stream_t(
    const T* __restrict__ bar_X_sph_Qmn,  // [q_count, nao_sph, nao_sph] C-order
    int q_abs,
    int q_offset,
    int q_count,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  const int q_rel = q_abs - q_offset;
  if (q_rel < 0 || q_rel >= q_count) return 0.0;
  return df_bar_cart_from_sph_qmn_t<T>(bar_X_sph_Qmn, q_rel, ia_cart, ib_cart, la, lb, a0_sph, b0_sph, nao_sph);
}

__device__ __forceinline__ double df_bar_cart_from_sph_qmn_stream(
    const double* __restrict__ bar_X_sph_Qmn,  // [q_count, nao_sph, nao_sph] C-order
    int q_abs,
    int q_offset,
    int q_count,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  return df_bar_cart_from_sph_qmn_stream_t<double>(
      bar_X_sph_Qmn,
      q_abs,
      q_offset,
      q_count,
      ia_cart,
      ib_cart,
      la,
      lb,
      a0_sph,
      b0_sph,
      nao_sph);
}

// Packed-Qp bar_X variant: bar_X stored in packed AO-pair ("s2") layout per Q:
//   bar_X_sph_Qp[q, p] where p indexes (i>=j) pairs in row-major lower triangle.
__device__ __forceinline__ int64_t tri_index_s2(int i, int j) {
  if (j > i) {
    const int tmp = i;
    i = j;
    j = tmp;
  }
  return (static_cast<int64_t>(i) * (static_cast<int64_t>(i) + 1)) / 2 + static_cast<int64_t>(j);
}

template <typename T>
__device__ __forceinline__ double df_bar_cart_from_sph_qp_t(
    const T* __restrict__ bar_X_sph_Qp,  // [naux, ntri] C-order
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
  const int64_t ntri = (static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph + 1)) / 2;
  const int64_t base_q = static_cast<int64_t>(q) * ntri;
  double acc = 0.0;
#pragma unroll
  for (int isA = 0; isA < 11; ++isA) {
    if (isA >= nA_sph) break;
    const double ta = cart2sph_coeff(la, ia_cart, isA);
    if (ta == 0.0) continue;
    const int i = a0_sph + isA;
#pragma unroll
    for (int isB = 0; isB < 11; ++isB) {
      if (isB >= nB_sph) break;
      const double tb = cart2sph_coeff(lb, ib_cart, isB);
      if (tb == 0.0) continue;
      const int j = b0_sph + isB;
      const int64_t p = tri_index_s2(i, j);
      acc += (ta * tb) * static_cast<double>(bar_X_sph_Qp[base_q + p]);
    }
  }
  return acc;
}

__device__ __forceinline__ double df_bar_cart_from_sph_qp(
    const double* __restrict__ bar_X_sph_Qp,  // [naux, ntri] C-order
    int q,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  return df_bar_cart_from_sph_qp_t<double>(bar_X_sph_Qp, q, ia_cart, ib_cart, la, lb, a0_sph, b0_sph, nao_sph);
}

template <typename T>
__device__ __forceinline__ double df_bar_cart_from_sph_qp_stream(
    const T* __restrict__ bar_X_sph_Qp,  // [naux, ntri] C-order
    int q_abs,
    int q_offset,
    int q_count,
    int ia_cart,
    int ib_cart,
    int la,
    int lb,
    int a0_sph,
    int b0_sph,
    int nao_sph) {
  const int q_rel = q_abs - q_offset;
  if (q_rel < 0 || q_rel >= q_count) return 0.0;
  // bar_X_sph_Qp is stored as a full [naux, ntri] tensor; keep absolute indexing (q_abs).
  return df_bar_cart_from_sph_qp_t<T>(bar_X_sph_Qp, q_abs, ia_cart, ib_cart, la, lb, a0_sph, b0_sph, nao_sph);
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

// Pointer overload for cases where the reduced segment is a slice of a larger
// array and `N` cannot be deduced from the argument type.
template <int N>
__device__ __forceinline__ void warp_reduce_sum_arr(double* v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] += __shfl_down_sync(0xffffffff, v[i], offset);
    }
  }
}

// Bridge: gap code from previous part (types needed here).

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
static inline int df_nroots_from_L(int L_total) {
  return ((L_total + 1) / 2) + 1;
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

}  // namespace

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

