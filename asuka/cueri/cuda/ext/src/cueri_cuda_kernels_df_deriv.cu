#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cueri_cuda_kernels_api.h"
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

__device__ __forceinline__ int ncart(int l) { return ((l + 1) * (l + 2)) >> 1; }

__device__ __forceinline__ double binom_d(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  if (n < 0 || n > kLMaxD) return 0.0;
  constexpr int kBinom[kLMaxD + 1][kLMaxD + 1] = {
      {1, 0, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 0},
      {1, 2, 1, 0, 0, 0, 0},
      {1, 3, 3, 1, 0, 0, 0},
      {1, 4, 6, 4, 1, 0, 0},
      {1, 5, 10, 10, 5, 1, 0},
      {1, 6, 15, 20, 15, 6, 1},
  };
  return static_cast<double>(kBinom[n][k]);
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

  for (int a = 2; a <= nmax; ++a) {
    G[a * STRIDE + 0] = B1 * static_cast<double>(a - 1) * G[(a - 2) * STRIDE + 0] + C * G[(a - 1) * STRIDE + 0];
  }
  for (int b = 2; b <= mmax; ++b) {
    G[0 * STRIDE + b] = B1p * static_cast<double>(b - 1) * G[0 * STRIDE + (b - 2)] + Cp * G[0 * STRIDE + (b - 1)];
  }

  if (mmax == 0 || nmax == 0) return;

  for (int a = 1; a <= nmax; ++a) {
    G[a * STRIDE + 1] = static_cast<double>(a) * B0 * G[(a - 1) * STRIDE + 0] + Cp * G[a * STRIDE + 0];
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
  double out = 0.0;
  for (int n = 0; n <= j; ++n) {
    out += binom_d(j, n) * xij_pow[j - n] * G[(n + i) * STRIDE + k];
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

      if (lane == 0) {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;

        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;

        const double Cx_ = (Px - Ax) + q_over * x * (Qx - Px);
        const double Cy_ = (Py - Ay) + q_over * x * (Qy - Py);
        const double Cz_ = (Pz - Az) + q_over * x * (Qz - Pz);

        const double Cpx_ = (Qx - Cx) + p_over * x * (Px - Qx);
        const double Cpy_ = (Qy - Cy) + p_over * x * (Py - Qy);
        const double Cpz_ = (Qz - Cz) + p_over * x * (Pz - Qz);

        compute_G_d(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
        compute_G_d(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
        compute_G_d(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);
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

      if (lane == 0) {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;

        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;

        const double Cx_ = (Px - Ax) + q_over * x * (Qx - Px);
        const double Cy_ = (Py - Ay) + q_over * x * (Qy - Py);
        const double Cz_ = (Pz - Az) + q_over * x * (Qz - Pz);

        const double Cpx_ = (Qx - Cx) + p_over * x * (Px - Qx);
        const double Cpy_ = (Qy - Cy) + p_over * x * (Py - Qy);
        const double Cpz_ = (Qz - Cz) + p_over * x * (Pz - Qz);

        compute_G_d(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
        compute_G_d(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
        compute_G_d(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);
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

      if (lane == 0) {
        const double B0 = x * 0.5 * inv_denom;
        const double B1 = (1.0 - x) * 0.5 / p + B0;
        const double B1p = (1.0 - x) * 0.5 / q + B0;

        const double q_over = q * inv_denom;
        const double p_over = p * inv_denom;

        const double Cx_ = (Px - Ax) + q_over * x * (Qx - Px);
        const double Cy_ = (Py - Ay) + q_over * x * (Qy - Py);
        const double Cz_ = (Pz - Az) + q_over * x * (Qz - Pz);

        const double Cpx_ = (Qx - Cx) + p_over * x * (Px - Qx);
        const double Cpy_ = (Qy - Cy) + p_over * x * (Py - Qy);
        const double Cpz_ = (Qz - Cz) + p_over * x * (Pz - Qz);

        compute_G_d(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
        compute_G_d(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
        compute_G_d(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);
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
