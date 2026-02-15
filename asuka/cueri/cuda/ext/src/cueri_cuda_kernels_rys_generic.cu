#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cueri_cuda_kernels_api.h"
#include "cueri_cuda_rys_device.cuh"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPiToFiveHalves = 2.0 * kPi * kPi * 1.772453850905516027298167483341145182;  // 2*pi^(5/2)

constexpr int kLMax = 5;
constexpr int kStride = 2 * kLMax + 1;  // la+lb <= 10, lc+ld <= 10
constexpr int kGSize = kStride * kStride;
constexpr int kNcartMax = (kLMax + 1) * (kLMax + 2) / 2;  // 21 for l=5
constexpr int kMaxWarpsPerBlock = 8;  // threads <= 256

__device__ __forceinline__ int ncart(int l) { return ((l + 1) * (l + 2)) >> 1; }

__device__ __forceinline__ double binom(int n, int k) {
  // Binomials are only ever needed for component exponents j/l, which satisfy j,l<=kLMax.
  if (k < 0 || k > n) return 0.0;
  if (n < 0 || n > kLMax) return 0.0;
  constexpr int kBinom[kLMax + 1][kLMax + 1] = {
      {1, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0},
      {1, 2, 1, 0, 0, 0},
      {1, 3, 3, 1, 0, 0},
      {1, 4, 6, 4, 1, 0},
      {1, 5, 10, 10, 5, 1},
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
  // Normalized version of PyQuante2 / GAMESS recurrence:
  // G[0,0]=1 and all other entries are dimensionless polynomials.

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

__device__ __forceinline__ void compute_G(
    double* G,
    int nmax,
    int mmax,
    double C,
    double Cp,
    double B0,
    double B1,
    double B1p) {
  compute_G_stride<kStride>(G, nmax, mmax, C, Cp, B0, B1, B1p);
}

template <int STRIDE>
__device__ __forceinline__ double shift_from_G_stride(
    const double* G,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow) {
  // Compute I(i,j,k,l) from G(a,b)=I(a,0,b,0).
  // Equivalent to PyQuante2 Shift(G,i,k,xij,xkl) but with explicit (j,l).

  double ijkl = 0.0;
  for (int m = 0; m <= l; ++m) {
    double ijm0 = 0.0;
    for (int n = 0; n <= j; ++n) {
      ijm0 += binom(j, n) * xij_pow[j - n] * G[(n + i) * STRIDE + (m + k)];
    }
    ijkl += binom(l, m) * xkl_pow[l - m] * ijm0;
  }
  return ijkl;
}

__device__ __forceinline__ double shift_from_G(
    const double* G,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow) {
  return shift_from_G_stride<kStride>(G, i, j, k, l, xij_pow, xkl_pow);
}

template <int NROOTS>
__global__ void KernelERI_RysGeneric(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    int ld,
    double* eri_out) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0 || ld < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax || ld > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nD = ncart(ld);
  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const int nElem = nAB * nCD;

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);
  const int D = static_cast<int>(sp_B[spCD]);

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

  const double xij = Ax - Bx;
  const double yij = Ay - By;
  const double zij = Az - Bz;
  const double xkl = Cx - Dx;
  const double ykl = Cy - Dy;
  const double zkl = Cz - Dz;

  const int nmax = la + lb;
  const int mmax = lc + ld;
  if (nmax >= kStride || mmax >= kStride) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  __shared__ int8_t sh_Ax[kNcartMax], sh_Ay[kNcartMax], sh_Az[kNcartMax];
  __shared__ int8_t sh_Bx[kNcartMax], sh_By[kNcartMax], sh_Bz[kNcartMax];
  __shared__ int8_t sh_Cx[kNcartMax], sh_Cy[kNcartMax], sh_Cz[kNcartMax];
  __shared__ int8_t sh_Dx[kNcartMax], sh_Dy[kNcartMax], sh_Dz[kNcartMax];

  __shared__ double sh_Gx[kGSize];
  __shared__ double sh_Gy[kGSize];
  __shared__ double sh_Gz[kGSize];
  __shared__ double sh_scale;
  __shared__ double sh_roots[NROOTS];
  __shared__ double sh_weights[NROOTS];

  __shared__ double sh_xij_pow[kLMax + 1], sh_xkl_pow[kLMax + 1];
  __shared__ double sh_yij_pow[kLMax + 1], sh_ykl_pow[kLMax + 1];
  __shared__ double sh_zij_pow[kLMax + 1], sh_zkl_pow[kLMax + 1];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, sh_Ax, sh_Ay, sh_Az);
    fill_cart_comp(lb, sh_Bx, sh_By, sh_Bz);
    fill_cart_comp(lc, sh_Cx, sh_Cy, sh_Cz);
    fill_cart_comp(ld, sh_Dx, sh_Dy, sh_Dz);

    sh_xij_pow[0] = 1.0;
    sh_xkl_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_ykl_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    sh_zkl_pow[0] = 1.0;
    for (int p = 1; p <= kLMax; ++p) {
      sh_xij_pow[p] = sh_xij_pow[p - 1] * xij;
      sh_xkl_pow[p] = sh_xkl_pow[p - 1] * xkl;
      sh_yij_pow[p] = sh_yij_pow[p - 1] * yij;
      sh_ykl_pow[p] = sh_ykl_pow[p - 1] * ykl;
      sh_zij_pow[p] = sh_zij_pow[p - 1] * zij;
      sh_zkl_pow[p] = sh_zkl_pow[p - 1] * zkl;
    }
  }
  __syncthreads();

  for (int e_base = 0; e_base < nElem; e_base += static_cast<int>(blockDim.x)) {
    const int e = e_base + static_cast<int>(threadIdx.x);
    const bool active = (e < nElem);

    int ia = 0, ib = 0, ic = 0, id = 0;
    int iax = 0, iay = 0, iaz = 0;
    int ibx = 0, iby = 0, ibz = 0;
    int icx = 0, icy = 0, icz = 0;
    int idx = 0, idy = 0, idz = 0;

    if (active) {
      const int ab = e / nCD;
      const int cd = e - ab * nCD;
      ia = ab / nB;
      ib = ab - ia * nB;
      ic = cd / nD;
      id = cd - ic * nD;

      iax = static_cast<int>(sh_Ax[ia]);
      iay = static_cast<int>(sh_Ay[ia]);
      iaz = static_cast<int>(sh_Az[ia]);
      ibx = static_cast<int>(sh_Bx[ib]);
      iby = static_cast<int>(sh_By[ib]);
      ibz = static_cast<int>(sh_Bz[ib]);
      icx = static_cast<int>(sh_Cx[ic]);
      icy = static_cast<int>(sh_Cy[ic]);
      icz = static_cast<int>(sh_Cz[ic]);
      idx = static_cast<int>(sh_Dx[id]);
      idy = static_cast<int>(sh_Dy[id]);
      idz = static_cast<int>(sh_Dz[id]);
    }

    double val = 0.0;

    for (int ip = 0; ip < nPairAB; ++ip) {
      const int ki = baseAB + ip;
      double p = 0.0;
      double Px = 0.0;
      double Py = 0.0;
      double Pz = 0.0;
      double cKab = 0.0;
      if (threadIdx.x == 0) {
        p = pair_eta[ki];
        Px = pair_Px[ki];
        Py = pair_Py[ki];
        Pz = pair_Pz[ki];
        cKab = pair_cK[ki];
      }

      for (int jp = 0; jp < nPairCD; ++jp) {
        const int kj = baseCD + jp;
        double q = 0.0;
        double Qx = 0.0;
        double Qy = 0.0;
        double Qz = 0.0;
        double cKcd = 0.0;
        double denom = 0.0;
        double base = 0.0;
        if (threadIdx.x == 0) {
          q = pair_eta[kj];
          Qx = pair_Px[kj];
          Qy = pair_Py[kj];
          Qz = pair_Pz[kj];
          cKcd = pair_cK[kj];

          const double dx = Px - Qx;
          const double dy = Py - Qy;
          const double dz = Pz - Qz;
          const double PQ2 = dx * dx + dy * dy + dz * dz;

          denom = p + q;
          const double omega = p * q / denom;
          const double T = omega * PQ2;

          base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

          cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);
        }

        for (int u = 0; u < NROOTS; ++u) {
          if (threadIdx.x == 0) {
            const double x = sh_roots[u];
            const double w = sh_weights[u];

            const double inv_denom = 1.0 / denom;
            const double B0 = x * 0.5 * inv_denom;
            const double B1 = (1.0 - x) * 0.5 / p + B0;
            const double B1p = (1.0 - x) * 0.5 / q + B0;

            const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
            const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
            const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

            const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
            const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
            const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

            compute_G(sh_Gx, nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
            compute_G(sh_Gy, nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
            compute_G(sh_Gz, nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

            sh_scale = base * w;
          }
          __syncthreads();

          if (active) {
            const double Ix = shift_from_G(sh_Gx, iax, ibx, icx, idx, sh_xij_pow, sh_xkl_pow);
            const double Iy = shift_from_G(sh_Gy, iay, iby, icy, idy, sh_yij_pow, sh_ykl_pow);
            const double Iz = shift_from_G(sh_Gz, iaz, ibz, icz, idz, sh_zij_pow, sh_zkl_pow);
            val += sh_scale * (Ix * Iy * Iz);
          }
          __syncthreads();
        }
      }
    }

    if (active) {
      eri_out[static_cast<int64_t>(t) * static_cast<int64_t>(nElem) + static_cast<int64_t>(e)] = val;
    }
  }
}

template <int NROOTS, int STRIDE>
struct Ld0SubwarpScratch {
  int8_t Ax[kNcartMax], Ay[kNcartMax], Az[kNcartMax];
  int8_t Bx[kNcartMax], By[kNcartMax], Bz[kNcartMax];
  int8_t Cx[kNcartMax], Cy[kNcartMax], Cz[kNcartMax];

  double Gx[STRIDE * STRIDE];
  double Gy[STRIDE * STRIDE];
  double Gz[STRIDE * STRIDE];

  double scale;
  double roots[NROOTS];
  double weights[NROOTS];

  double xij_pow[kLMax + 1], xkl_pow[kLMax + 1];
  double yij_pow[kLMax + 1], ykl_pow[kLMax + 1];
  double zij_pow[kLMax + 1], zkl_pow[kLMax + 1];

  double Axyz[3];
  double Cxyz[3];
};

template <int NROOTS, int STRIDE>
__global__ void KernelERI_RysDF_Ld0Subwarp16(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out) {
  // 2 tasks per warp, 16 lanes per task.
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;

  const int subwarp = lane >> 4;  // 0 or 1
  const int lane16 = lane & 15;
  const unsigned mask = 0xFFFFu << (16 * subwarp);

  const int t = warp_global * 2 + subwarp;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nAB = nA * nB;
  const int nElem = nAB * nC;
  if (nElem <= 0 || nElem > 64) return;
  const bool lb_is_zero = (lb == 0);

  // Shared scratch: one entry per task in this block.
  extern __shared__ unsigned char smem[];
  auto* scratch = reinterpret_cast<Ld0SubwarpScratch<NROOTS, STRIDE>*>(smem);
  const int task_local = warp_id * 2 + subwarp;
  Ld0SubwarpScratch<NROOTS, STRIDE>* s = scratch + task_local;

  const bool leader = (lane16 == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  if (leader) {
    fill_cart_comp(la, s->Ax, s->Ay, s->Az);
    if (!lb_is_zero) fill_cart_comp(lb, s->Bx, s->By, s->Bz);
    fill_cart_comp(lc, s->Cx, s->Cy, s->Cz);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    s->Axyz[0] = Ax;
    s->Axyz[1] = Ay;
    s->Axyz[2] = Az;
    s->Cxyz[0] = Cx;
    s->Cxyz[1] = Cy;
    s->Cxyz[2] = Cz;

    s->xij_pow[0] = 1.0;
    s->yij_pow[0] = 1.0;
    s->zij_pow[0] = 1.0;
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    s->xkl_pow[0] = 1.0;
    s->ykl_pow[0] = 1.0;
    s->zkl_pow[0] = 1.0;
    if (!lb_is_zero) {
      const double Bx = shell_cx[B];
      const double By = shell_cy[B];
      const double Bz = shell_cz[B];
      const double xij = Ax - Bx;
      const double yij = Ay - By;
      const double zij = Az - Bz;
      for (int p = 1; p <= kLMax; ++p) {
        s->xij_pow[p] = s->xij_pow[p - 1] * xij;
        s->yij_pow[p] = s->yij_pow[p - 1] * yij;
        s->zij_pow[p] = s->zij_pow[p - 1] * zij;
      }
    }
  }
  __syncwarp(mask);

  // Each lane owns up to four elements: e = lane16 + pass*16, pass=0..3 (supports up to nElem<=64).
  const int e0 = lane16;
  const int e1 = lane16 + 16;
  const int e2 = lane16 + 32;
  const int e3 = lane16 + 48;
  const bool has0 = (e0 < nElem);
  const bool has1 = (e1 < nElem);
  const bool has2 = (e2 < nElem);
  const bool has3 = (e3 < nElem);

  int iax0 = 0, iay0 = 0, iaz0 = 0, ibx0 = 0, iby0 = 0, ibz0 = 0, icx0 = 0, icy0 = 0, icz0 = 0;
  int iax1 = 0, iay1 = 0, iaz1 = 0, ibx1 = 0, iby1 = 0, ibz1 = 0, icx1 = 0, icy1 = 0, icz1 = 0;
  int iax2 = 0, iay2 = 0, iaz2 = 0, ibx2 = 0, iby2 = 0, ibz2 = 0, icx2 = 0, icy2 = 0, icz2 = 0;
  int iax3 = 0, iay3 = 0, iaz3 = 0, ibx3 = 0, iby3 = 0, ibz3 = 0, icx3 = 0, icy3 = 0, icz3 = 0;

  if (has0) {
    const int ab = e0 / nC;
    const int ic = e0 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax0 = static_cast<int>(s->Ax[ia]);
    iay0 = static_cast<int>(s->Ay[ia]);
    iaz0 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx0 = static_cast<int>(s->Bx[ib]);
      iby0 = static_cast<int>(s->By[ib]);
      ibz0 = static_cast<int>(s->Bz[ib]);
    }
    icx0 = static_cast<int>(s->Cx[ic]);
    icy0 = static_cast<int>(s->Cy[ic]);
    icz0 = static_cast<int>(s->Cz[ic]);
  }
  if (has1) {
    const int ab = e1 / nC;
    const int ic = e1 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax1 = static_cast<int>(s->Ax[ia]);
    iay1 = static_cast<int>(s->Ay[ia]);
    iaz1 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx1 = static_cast<int>(s->Bx[ib]);
      iby1 = static_cast<int>(s->By[ib]);
      ibz1 = static_cast<int>(s->Bz[ib]);
    }
    icx1 = static_cast<int>(s->Cx[ic]);
    icy1 = static_cast<int>(s->Cy[ic]);
    icz1 = static_cast<int>(s->Cz[ic]);
  }
  if (has2) {
    const int ab = e2 / nC;
    const int ic = e2 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax2 = static_cast<int>(s->Ax[ia]);
    iay2 = static_cast<int>(s->Ay[ia]);
    iaz2 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx2 = static_cast<int>(s->Bx[ib]);
      iby2 = static_cast<int>(s->By[ib]);
      ibz2 = static_cast<int>(s->Bz[ib]);
    }
    icx2 = static_cast<int>(s->Cx[ic]);
    icy2 = static_cast<int>(s->Cy[ic]);
    icz2 = static_cast<int>(s->Cz[ic]);
  }
  if (has3) {
    const int ab = e3 / nC;
    const int ic = e3 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax3 = static_cast<int>(s->Ax[ia]);
    iay3 = static_cast<int>(s->Ay[ia]);
    iaz3 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx3 = static_cast<int>(s->Bx[ib]);
      iby3 = static_cast<int>(s->By[ib]);
      ibz3 = static_cast<int>(s->Bz[ib]);
    }
    icx3 = static_cast<int>(s->Cx[ic]);
    icy3 = static_cast<int>(s->Cy[ic]);
    icz3 = static_cast<int>(s->Cz[ic]);
  }

  double acc0 = 0.0;
  double acc1 = 0.0;
  double acc2 = 0.0;
  double acc3 = 0.0;

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double cKab = 0.0;
    if (leader) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      cKab = pair_cK[ki];
    }

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (leader) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, s->roots, s->weights);
      }

      for (int u = 0; u < NROOTS; ++u) {
        if (leader) {
          const double x = s->roots[u];
          const double w = s->weights[u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = s->Axyz[0];
          const double Ay = s->Axyz[1];
          const double Az = s->Axyz[2];
          const double Cx = s->Cxyz[0];
          const double Cy = s->Cxyz[1];
          const double Cz = s->Cxyz[2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(s->Gx, nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gy, nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gz, nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          s->scale = base * w;
        }
        __syncwarp(mask);

        const double scale = s->scale;
        if (has0) {
          const double Ix = lb_is_zero ? s->Gx[iax0 * STRIDE + icx0]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax0, ibx0, icx0, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay0 * STRIDE + icy0]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay0, iby0, icy0, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz0 * STRIDE + icz0]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz0, ibz0, icz0, 0, s->zij_pow, s->zkl_pow);
          acc0 += scale * (Ix * Iy * Iz);
        }
        if (has1) {
          const double Ix = lb_is_zero ? s->Gx[iax1 * STRIDE + icx1]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax1, ibx1, icx1, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay1 * STRIDE + icy1]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay1, iby1, icy1, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz1 * STRIDE + icz1]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz1, ibz1, icz1, 0, s->zij_pow, s->zkl_pow);
          acc1 += scale * (Ix * Iy * Iz);
        }
        if (has2) {
          const double Ix = lb_is_zero ? s->Gx[iax2 * STRIDE + icx2]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax2, ibx2, icx2, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay2 * STRIDE + icy2]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay2, iby2, icy2, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz2 * STRIDE + icz2]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz2, ibz2, icz2, 0, s->zij_pow, s->zkl_pow);
          acc2 += scale * (Ix * Iy * Iz);
        }
        if (has3) {
          const double Ix = lb_is_zero ? s->Gx[iax3 * STRIDE + icx3]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax3, ibx3, icx3, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay3 * STRIDE + icy3]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay3, iby3, icy3, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz3 * STRIDE + icz3]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz3, ibz3, icz3, 0, s->zij_pow, s->zkl_pow);
          acc3 += scale * (Ix * Iy * Iz);
        }
        __syncwarp(mask);
      }
    }
  }

  double* out = eri_out + static_cast<int64_t>(t) * static_cast<int64_t>(nElem);
  if (has0) out[e0] = acc0;
  if (has1) out[e1] = acc1;
  if (has2) out[e2] = acc2;
  if (has3) out[e3] = acc3;
}

template <int NROOTS, int STRIDE>
__global__ void KernelERI_RysDF_Ld0Subwarp8(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out) {
  // 4 tasks per warp, 8 lanes per task (subwarp8). Each lane computes up to 4 elements:
  //   e = lane8 + pass*8, pass=0..3 -> supports up to nElem<=32.
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;

  const int subwarp = lane >> 3;  // 0..3
  const int lane8 = lane & 7;
  const unsigned mask = 0xFFu << (8 * subwarp);

  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nAB = nA * nB;
  const int nElem = nAB * nC;
  if (nElem <= 0 || nElem > 32) return;
  const bool lb_is_zero = (lb == 0);

  // Shared scratch: one entry per task in this block.
  extern __shared__ unsigned char smem[];
  auto* scratch = reinterpret_cast<Ld0SubwarpScratch<NROOTS, STRIDE>*>(smem);
  const int task_local = warp_id * 4 + subwarp;
  Ld0SubwarpScratch<NROOTS, STRIDE>* s = scratch + task_local;

  const bool leader = (lane8 == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  if (leader) {
    fill_cart_comp(la, s->Ax, s->Ay, s->Az);
    if (!lb_is_zero) fill_cart_comp(lb, s->Bx, s->By, s->Bz);
    fill_cart_comp(lc, s->Cx, s->Cy, s->Cz);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    s->Axyz[0] = Ax;
    s->Axyz[1] = Ay;
    s->Axyz[2] = Az;
    s->Cxyz[0] = Cx;
    s->Cxyz[1] = Cy;
    s->Cxyz[2] = Cz;

    s->xij_pow[0] = 1.0;
    s->yij_pow[0] = 1.0;
    s->zij_pow[0] = 1.0;
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    s->xkl_pow[0] = 1.0;
    s->ykl_pow[0] = 1.0;
    s->zkl_pow[0] = 1.0;
    if (!lb_is_zero) {
      const double Bx = shell_cx[B];
      const double By = shell_cy[B];
      const double Bz = shell_cz[B];
      const double xij = Ax - Bx;
      const double yij = Ay - By;
      const double zij = Az - Bz;
      for (int p = 1; p <= kLMax; ++p) {
        s->xij_pow[p] = s->xij_pow[p - 1] * xij;
        s->yij_pow[p] = s->yij_pow[p - 1] * yij;
        s->zij_pow[p] = s->zij_pow[p - 1] * zij;
      }
    }
  }
  __syncwarp(mask);

  // Each lane owns up to 4 elements: e0..e3.
  const int e0 = lane8;
  const int e1 = lane8 + 8;
  const int e2 = lane8 + 16;
  const int e3 = lane8 + 24;

  const bool has0 = (e0 < nElem);
  const bool has1 = (e1 < nElem);
  const bool has2 = (e2 < nElem);
  const bool has3 = (e3 < nElem);

  int iax0 = 0, iay0 = 0, iaz0 = 0, ibx0 = 0, iby0 = 0, ibz0 = 0, icx0 = 0, icy0 = 0, icz0 = 0;
  int iax1 = 0, iay1 = 0, iaz1 = 0, ibx1 = 0, iby1 = 0, ibz1 = 0, icx1 = 0, icy1 = 0, icz1 = 0;
  int iax2 = 0, iay2 = 0, iaz2 = 0, ibx2 = 0, iby2 = 0, ibz2 = 0, icx2 = 0, icy2 = 0, icz2 = 0;
  int iax3 = 0, iay3 = 0, iaz3 = 0, ibx3 = 0, iby3 = 0, ibz3 = 0, icx3 = 0, icy3 = 0, icz3 = 0;

  if (has0) {
    const int ab = e0 / nC;
    const int ic = e0 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax0 = static_cast<int>(s->Ax[ia]);
    iay0 = static_cast<int>(s->Ay[ia]);
    iaz0 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx0 = static_cast<int>(s->Bx[ib]);
      iby0 = static_cast<int>(s->By[ib]);
      ibz0 = static_cast<int>(s->Bz[ib]);
    }
    icx0 = static_cast<int>(s->Cx[ic]);
    icy0 = static_cast<int>(s->Cy[ic]);
    icz0 = static_cast<int>(s->Cz[ic]);
  }
  if (has1) {
    const int ab = e1 / nC;
    const int ic = e1 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax1 = static_cast<int>(s->Ax[ia]);
    iay1 = static_cast<int>(s->Ay[ia]);
    iaz1 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx1 = static_cast<int>(s->Bx[ib]);
      iby1 = static_cast<int>(s->By[ib]);
      ibz1 = static_cast<int>(s->Bz[ib]);
    }
    icx1 = static_cast<int>(s->Cx[ic]);
    icy1 = static_cast<int>(s->Cy[ic]);
    icz1 = static_cast<int>(s->Cz[ic]);
  }
  if (has2) {
    const int ab = e2 / nC;
    const int ic = e2 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax2 = static_cast<int>(s->Ax[ia]);
    iay2 = static_cast<int>(s->Ay[ia]);
    iaz2 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx2 = static_cast<int>(s->Bx[ib]);
      iby2 = static_cast<int>(s->By[ib]);
      ibz2 = static_cast<int>(s->Bz[ib]);
    }
    icx2 = static_cast<int>(s->Cx[ic]);
    icy2 = static_cast<int>(s->Cy[ic]);
    icz2 = static_cast<int>(s->Cz[ic]);
  }
  if (has3) {
    const int ab = e3 / nC;
    const int ic = e3 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax3 = static_cast<int>(s->Ax[ia]);
    iay3 = static_cast<int>(s->Ay[ia]);
    iaz3 = static_cast<int>(s->Az[ia]);
    if (!lb_is_zero) {
      ibx3 = static_cast<int>(s->Bx[ib]);
      iby3 = static_cast<int>(s->By[ib]);
      ibz3 = static_cast<int>(s->Bz[ib]);
    }
    icx3 = static_cast<int>(s->Cx[ic]);
    icy3 = static_cast<int>(s->Cy[ic]);
    icz3 = static_cast<int>(s->Cz[ic]);
  }

  double acc0 = 0.0;
  double acc1 = 0.0;
  double acc2 = 0.0;
  double acc3 = 0.0;

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double cKab = 0.0;
    if (leader) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      cKab = pair_cK[ki];
    }

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (leader) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, s->roots, s->weights);
      }

      for (int u = 0; u < NROOTS; ++u) {
        if (leader) {
          const double x = s->roots[u];
          const double w = s->weights[u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = s->Axyz[0];
          const double Ay = s->Axyz[1];
          const double Az = s->Axyz[2];
          const double Cx = s->Cxyz[0];
          const double Cy = s->Cxyz[1];
          const double Cz = s->Cxyz[2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(s->Gx, nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gy, nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gz, nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          s->scale = base * w;
        }
        __syncwarp(mask);

        const double scale = s->scale;
        if (has0) {
          const double Ix = lb_is_zero ? s->Gx[iax0 * STRIDE + icx0]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax0, ibx0, icx0, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay0 * STRIDE + icy0]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay0, iby0, icy0, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz0 * STRIDE + icz0]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz0, ibz0, icz0, 0, s->zij_pow, s->zkl_pow);
          acc0 += scale * (Ix * Iy * Iz);
        }
        if (has1) {
          const double Ix = lb_is_zero ? s->Gx[iax1 * STRIDE + icx1]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax1, ibx1, icx1, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay1 * STRIDE + icy1]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay1, iby1, icy1, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz1 * STRIDE + icz1]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz1, ibz1, icz1, 0, s->zij_pow, s->zkl_pow);
          acc1 += scale * (Ix * Iy * Iz);
        }
        if (has2) {
          const double Ix = lb_is_zero ? s->Gx[iax2 * STRIDE + icx2]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax2, ibx2, icx2, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay2 * STRIDE + icy2]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay2, iby2, icy2, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz2 * STRIDE + icz2]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz2, ibz2, icz2, 0, s->zij_pow, s->zkl_pow);
          acc2 += scale * (Ix * Iy * Iz);
        }
        if (has3) {
          const double Ix = lb_is_zero ? s->Gx[iax3 * STRIDE + icx3]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax3, ibx3, icx3, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay3 * STRIDE + icy3]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay3, iby3, icy3, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz3 * STRIDE + icz3]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz3, ibz3, icz3, 0, s->zij_pow, s->zkl_pow);
          acc3 += scale * (Ix * Iy * Iz);
        }
        __syncwarp(mask);
      }
    }
  }

  double* out = eri_out + static_cast<int64_t>(t) * static_cast<int64_t>(nElem);
  if (has0) out[e0] = acc0;
  if (has1) out[e1] = acc1;
  if (has2) out[e2] = acc2;
  if (has3) out[e3] = acc3;
}

// Warp-per-task ld=0 kernel for nElem <= 128.
//
// Keep this as a scalar/unrolled implementation (4 elements per lane) to
// minimize register pressure in the common DF regime (many small tiles).
template <int NROOTS, int STRIDE>
__global__ void KernelERI_RysDF_Ld0Warp128(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  if (warp_id >= kMaxWarpsPerBlock) return;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nAB = nA * nB;
  const int nElem = nAB * nC;
  const bool lb_is_zero = (lb == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  __shared__ int8_t sh_Ax[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Ay[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Az[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_By[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bz[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cy[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cz[kMaxWarpsPerBlock][kNcartMax];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_scale[kMaxWarpsPerBlock];
  __shared__ double sh_roots[kMaxWarpsPerBlock][NROOTS];
  __shared__ double sh_weights[kMaxWarpsPerBlock][NROOTS];

  __shared__ double sh_Axyz[kMaxWarpsPerBlock][3];
  __shared__ double sh_Cxyz[kMaxWarpsPerBlock][3];

  __shared__ double sh_xij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_xkl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_yij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_ykl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zkl_pow[kMaxWarpsPerBlock][kLMax + 1];

  if (lane == 0) {
    fill_cart_comp(la, sh_Ax[warp_id], sh_Ay[warp_id], sh_Az[warp_id]);
    if (!lb_is_zero) fill_cart_comp(lb, sh_Bx[warp_id], sh_By[warp_id], sh_Bz[warp_id]);
    fill_cart_comp(lc, sh_Cx[warp_id], sh_Cy[warp_id], sh_Cz[warp_id]);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    sh_Axyz[warp_id][0] = Ax;
    sh_Axyz[warp_id][1] = Ay;
    sh_Axyz[warp_id][2] = Az;
    sh_Cxyz[warp_id][0] = Cx;
    sh_Cxyz[warp_id][1] = Cy;
    sh_Cxyz[warp_id][2] = Cz;

    sh_xij_pow[warp_id][0] = 1.0;
    sh_yij_pow[warp_id][0] = 1.0;
    sh_zij_pow[warp_id][0] = 1.0;
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    sh_xkl_pow[warp_id][0] = 1.0;
    sh_ykl_pow[warp_id][0] = 1.0;
    sh_zkl_pow[warp_id][0] = 1.0;
    if (!lb_is_zero) {
      const double Bx = shell_cx[B];
      const double By = shell_cy[B];
      const double Bz = shell_cz[B];
      const double xij = Ax - Bx;
      const double yij = Ay - By;
      const double zij = Az - Bz;
      for (int p = 1; p <= kLMax; ++p) {
        sh_xij_pow[warp_id][p] = sh_xij_pow[warp_id][p - 1] * xij;
        sh_yij_pow[warp_id][p] = sh_yij_pow[warp_id][p - 1] * yij;
        sh_zij_pow[warp_id][p] = sh_zij_pow[warp_id][p - 1] * zij;
      }
    }
  }
  __syncwarp();

  // Each lane owns up to 4 elements (nElem <= 128 enforced by launcher).
  const int e0 = lane;
  const int e1 = lane + 32;
  const int e2 = lane + 64;
  const int e3 = lane + 96;

  const bool has0 = (e0 < nElem);
  const bool has1 = (e1 < nElem);
  const bool has2 = (e2 < nElem);
  const bool has3 = (e3 < nElem);

  int iax0 = 0, iay0 = 0, iaz0 = 0, ibx0 = 0, iby0 = 0, ibz0 = 0, icx0 = 0, icy0 = 0, icz0 = 0;
  int iax1 = 0, iay1 = 0, iaz1 = 0, ibx1 = 0, iby1 = 0, ibz1 = 0, icx1 = 0, icy1 = 0, icz1 = 0;
  int iax2 = 0, iay2 = 0, iaz2 = 0, ibx2 = 0, iby2 = 0, ibz2 = 0, icx2 = 0, icy2 = 0, icz2 = 0;
  int iax3 = 0, iay3 = 0, iaz3 = 0, ibx3 = 0, iby3 = 0, ibz3 = 0, icx3 = 0, icy3 = 0, icz3 = 0;

  if (has0) {
    const int ab = e0 / nC;
    const int ic = e0 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax0 = static_cast<int>(sh_Ax[warp_id][ia]);
    iay0 = static_cast<int>(sh_Ay[warp_id][ia]);
    iaz0 = static_cast<int>(sh_Az[warp_id][ia]);
    if (!lb_is_zero) {
      ibx0 = static_cast<int>(sh_Bx[warp_id][ib]);
      iby0 = static_cast<int>(sh_By[warp_id][ib]);
      ibz0 = static_cast<int>(sh_Bz[warp_id][ib]);
    }
    icx0 = static_cast<int>(sh_Cx[warp_id][ic]);
    icy0 = static_cast<int>(sh_Cy[warp_id][ic]);
    icz0 = static_cast<int>(sh_Cz[warp_id][ic]);
  }
  if (has1) {
    const int ab = e1 / nC;
    const int ic = e1 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax1 = static_cast<int>(sh_Ax[warp_id][ia]);
    iay1 = static_cast<int>(sh_Ay[warp_id][ia]);
    iaz1 = static_cast<int>(sh_Az[warp_id][ia]);
    if (!lb_is_zero) {
      ibx1 = static_cast<int>(sh_Bx[warp_id][ib]);
      iby1 = static_cast<int>(sh_By[warp_id][ib]);
      ibz1 = static_cast<int>(sh_Bz[warp_id][ib]);
    }
    icx1 = static_cast<int>(sh_Cx[warp_id][ic]);
    icy1 = static_cast<int>(sh_Cy[warp_id][ic]);
    icz1 = static_cast<int>(sh_Cz[warp_id][ic]);
  }
  if (has2) {
    const int ab = e2 / nC;
    const int ic = e2 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax2 = static_cast<int>(sh_Ax[warp_id][ia]);
    iay2 = static_cast<int>(sh_Ay[warp_id][ia]);
    iaz2 = static_cast<int>(sh_Az[warp_id][ia]);
    if (!lb_is_zero) {
      ibx2 = static_cast<int>(sh_Bx[warp_id][ib]);
      iby2 = static_cast<int>(sh_By[warp_id][ib]);
      ibz2 = static_cast<int>(sh_Bz[warp_id][ib]);
    }
    icx2 = static_cast<int>(sh_Cx[warp_id][ic]);
    icy2 = static_cast<int>(sh_Cy[warp_id][ic]);
    icz2 = static_cast<int>(sh_Cz[warp_id][ic]);
  }
  if (has3) {
    const int ab = e3 / nC;
    const int ic = e3 - ab * nC;
    const int ia = ab / nB;
    const int ib = ab - ia * nB;
    iax3 = static_cast<int>(sh_Ax[warp_id][ia]);
    iay3 = static_cast<int>(sh_Ay[warp_id][ia]);
    iaz3 = static_cast<int>(sh_Az[warp_id][ia]);
    if (!lb_is_zero) {
      ibx3 = static_cast<int>(sh_Bx[warp_id][ib]);
      iby3 = static_cast<int>(sh_By[warp_id][ib]);
      ibz3 = static_cast<int>(sh_Bz[warp_id][ib]);
    }
    icx3 = static_cast<int>(sh_Cx[warp_id][ic]);
    icy3 = static_cast<int>(sh_Cy[warp_id][ic]);
    icz3 = static_cast<int>(sh_Cz[warp_id][ic]);
  }

  double acc0 = 0.0;
  double acc1 = 0.0;
  double acc2 = 0.0;
  double acc3 = 0.0;

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double cKab = 0.0;
    if (lane == 0) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      cKab = pair_cK[ki];
    }

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (lane == 0) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots[warp_id], sh_weights[warp_id]);
      }

      for (int u = 0; u < NROOTS; ++u) {
        if (lane == 0) {
          const double x = sh_roots[warp_id][u];
          const double w = sh_weights[warp_id][u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = sh_Axyz[warp_id][0];
          const double Ay = sh_Axyz[warp_id][1];
          const double Az = sh_Axyz[warp_id][2];
          const double Cx = sh_Cxyz[warp_id][0];
          const double Cy = sh_Cxyz[warp_id][1];
          const double Cz = sh_Cxyz[warp_id][2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          sh_scale[warp_id] = base * w;
        }
        __syncwarp();

        const double scale = sh_scale[warp_id];

        if (has0) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax0 * STRIDE + icx0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax0, ibx0, icx0, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay0 * STRIDE + icy0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay0, iby0, icy0, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz0 * STRIDE + icz0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz0, ibz0, icz0, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          acc0 += scale * (Ix * Iy * Iz);
        }
        if (has1) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax1 * STRIDE + icx1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax1, ibx1, icx1, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay1 * STRIDE + icy1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay1, iby1, icy1, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz1 * STRIDE + icz1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz1, ibz1, icz1, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          acc1 += scale * (Ix * Iy * Iz);
        }
        if (has2) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax2 * STRIDE + icx2]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax2, ibx2, icx2, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay2 * STRIDE + icy2]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay2, iby2, icy2, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz2 * STRIDE + icz2]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz2, ibz2, icz2, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          acc2 += scale * (Ix * Iy * Iz);
        }
        if (has3) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax3 * STRIDE + icx3]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax3, ibx3, icx3, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay3 * STRIDE + icy3]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay3, iby3, icy3, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz3 * STRIDE + icz3]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz3, ibz3, icz3, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          acc3 += scale * (Ix * Iy * Iz);
        }
        __syncwarp();
      }
    }
  }

  double* out = eri_out + static_cast<int64_t>(t) * static_cast<int64_t>(nElem);
  if (has0) out[e0] = acc0;
  if (has1) out[e1] = acc1;
  if (has2) out[e2] = acc2;
  if (has3) out[e3] = acc3;
}

template <int NROOTS, int STRIDE, int MAX_PASS>
__global__ void KernelERI_RysDF_Ld0Warp(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  if (warp_id >= kMaxWarpsPerBlock) return;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nAB = nA * nB;
  const int nElem = nAB * nC;
  const bool lb_is_zero = (lb == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  __shared__ int8_t sh_Ax[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Ay[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Az[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_By[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bz[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cy[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cz[kMaxWarpsPerBlock][kNcartMax];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_scale[kMaxWarpsPerBlock];
  __shared__ double sh_roots[kMaxWarpsPerBlock][NROOTS];
  __shared__ double sh_weights[kMaxWarpsPerBlock][NROOTS];

  __shared__ double sh_Axyz[kMaxWarpsPerBlock][3];
  __shared__ double sh_Cxyz[kMaxWarpsPerBlock][3];

  __shared__ double sh_xij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_xkl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_yij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_ykl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zkl_pow[kMaxWarpsPerBlock][kLMax + 1];

  if (lane == 0) {
    fill_cart_comp(la, sh_Ax[warp_id], sh_Ay[warp_id], sh_Az[warp_id]);
    if (!lb_is_zero) fill_cart_comp(lb, sh_Bx[warp_id], sh_By[warp_id], sh_Bz[warp_id]);
    fill_cart_comp(lc, sh_Cx[warp_id], sh_Cy[warp_id], sh_Cz[warp_id]);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    sh_Axyz[warp_id][0] = Ax;
    sh_Axyz[warp_id][1] = Ay;
    sh_Axyz[warp_id][2] = Az;
    sh_Cxyz[warp_id][0] = Cx;
    sh_Cxyz[warp_id][1] = Cy;
    sh_Cxyz[warp_id][2] = Cz;

    sh_xij_pow[warp_id][0] = 1.0;
    sh_yij_pow[warp_id][0] = 1.0;
    sh_zij_pow[warp_id][0] = 1.0;
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    sh_xkl_pow[warp_id][0] = 1.0;
    sh_ykl_pow[warp_id][0] = 1.0;
    sh_zkl_pow[warp_id][0] = 1.0;
    if (!lb_is_zero) {
      const double Bx = shell_cx[B];
      const double By = shell_cy[B];
      const double Bz = shell_cz[B];
      const double xij = Ax - Bx;
      const double yij = Ay - By;
      const double zij = Az - Bz;
      for (int p = 1; p <= kLMax; ++p) {
        sh_xij_pow[warp_id][p] = sh_xij_pow[warp_id][p - 1] * xij;
        sh_yij_pow[warp_id][p] = sh_yij_pow[warp_id][p - 1] * yij;
        sh_zij_pow[warp_id][p] = sh_zij_pow[warp_id][p - 1] * zij;
      }
    }
  }
  __syncwarp();

  // Each lane owns up to MAX_PASS elements (4 for nElem<=128, 8 for nElem<=256).
  constexpr int kMaxPass = MAX_PASS;
  int e[kMaxPass];
  bool has[kMaxPass];
  int iax[kMaxPass], iay[kMaxPass], iaz[kMaxPass];
  int ibx[kMaxPass], iby[kMaxPass], ibz[kMaxPass];
  int icx[kMaxPass], icy[kMaxPass], icz[kMaxPass];
  double acc[kMaxPass];

#pragma unroll
  for (int pass = 0; pass < kMaxPass; ++pass) {
    e[pass] = lane + (pass << 5);
    has[pass] = (e[pass] < nElem);
    iax[pass] = iay[pass] = iaz[pass] = 0;
    ibx[pass] = iby[pass] = ibz[pass] = 0;
    icx[pass] = icy[pass] = icz[pass] = 0;
    acc[pass] = 0.0;
    if (has[pass]) {
      const int ab = e[pass] / nC;
      const int ic = e[pass] - ab * nC;
      const int ia = ab / nB;
      const int ib = ab - ia * nB;
      iax[pass] = static_cast<int>(sh_Ax[warp_id][ia]);
      iay[pass] = static_cast<int>(sh_Ay[warp_id][ia]);
      iaz[pass] = static_cast<int>(sh_Az[warp_id][ia]);
      if (!lb_is_zero) {
        ibx[pass] = static_cast<int>(sh_Bx[warp_id][ib]);
        iby[pass] = static_cast<int>(sh_By[warp_id][ib]);
        ibz[pass] = static_cast<int>(sh_Bz[warp_id][ib]);
      }
      icx[pass] = static_cast<int>(sh_Cx[warp_id][ic]);
      icy[pass] = static_cast<int>(sh_Cy[warp_id][ic]);
      icz[pass] = static_cast<int>(sh_Cz[warp_id][ic]);
    }
  }

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double cKab = 0.0;
    if (lane == 0) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      cKab = pair_cK[ki];
    }

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (lane == 0) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots[warp_id], sh_weights[warp_id]);
      }

      for (int u = 0; u < NROOTS; ++u) {
        if (lane == 0) {
          const double x = sh_roots[warp_id][u];
          const double w = sh_weights[warp_id][u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = sh_Axyz[warp_id][0];
          const double Ay = sh_Axyz[warp_id][1];
          const double Az = sh_Axyz[warp_id][2];
          const double Cx = sh_Cxyz[warp_id][0];
          const double Cy = sh_Cxyz[warp_id][1];
          const double Cz = sh_Cxyz[warp_id][2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          sh_scale[warp_id] = base * w;
        }
        __syncwarp();

        const double scale = sh_scale[warp_id];
        // Each lane may own multiple output elements (passes). For ld=0, `l=0` in shift().
#pragma unroll
        for (int pass = 0; pass < kMaxPass; ++pass) {
          if (!has[pass]) continue;
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax[pass] * STRIDE + icx[pass]]
                                       : shift_from_G_stride<STRIDE>(sh_Gx[warp_id],
                                                                    iax[pass],
                                                                    ibx[pass],
                                                                    icx[pass],
                                                                    0,
                                                                    sh_xij_pow[warp_id],
                                                                    sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay[pass] * STRIDE + icy[pass]]
                                       : shift_from_G_stride<STRIDE>(sh_Gy[warp_id],
                                                                    iay[pass],
                                                                    iby[pass],
                                                                    icy[pass],
                                                                    0,
                                                                    sh_yij_pow[warp_id],
                                                                    sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz[pass] * STRIDE + icz[pass]]
                                       : shift_from_G_stride<STRIDE>(sh_Gz[warp_id],
                                                                    iaz[pass],
                                                                    ibz[pass],
                                                                    icz[pass],
                                                                    0,
                                                                    sh_zij_pow[warp_id],
                                                                    sh_zkl_pow[warp_id]);
          acc[pass] += scale * (Ix * Iy * Iz);
        }
        __syncwarp();
      }
    }
  }

  double* out = eri_out + static_cast<int64_t>(t) * static_cast<int64_t>(nElem);
#pragma unroll
  for (int pass = 0; pass < kMaxPass; ++pass) {
    if (has[pass]) out[e[pass]] = acc[pass];
  }
}

enum class Ld0WarpMode : int { kWarp = 0, kSubwarp16 = 1, kSubwarp8 = 2 };

template <int NROOTS, int STRIDE>
inline void launch_eri_rys_df_ld0(
    Ld0WarpMode mode,
    int blocks,
    int threads,
    int tasks_per_block,
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out,
    cudaStream_t stream) {
  if (mode == Ld0WarpMode::kSubwarp8) {
    const int shared_bytes = tasks_per_block * static_cast<int>(sizeof(Ld0SubwarpScratch<NROOTS, STRIDE>));
    KernelERI_RysDF_Ld0Subwarp8<NROOTS, STRIDE><<<blocks, threads, shared_bytes, stream>>>(
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
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        la,
        lb,
        lc,
        eri_out);
    return;
  }
  if (mode == Ld0WarpMode::kSubwarp16) {
    const int shared_bytes = tasks_per_block * static_cast<int>(sizeof(Ld0SubwarpScratch<NROOTS, STRIDE>));
    KernelERI_RysDF_Ld0Subwarp16<NROOTS, STRIDE><<<blocks, threads, shared_bytes, stream>>>(
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
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        la,
        lb,
        lc,
        eri_out);
    return;
  }
  const int nA = ((la + 1) * (la + 2)) >> 1;
  const int nB = ((lb + 1) * (lb + 2)) >> 1;
  const int nC = ((lc + 1) * (lc + 2)) >> 1;
  const int nElem = nA * nB * nC;
  if (nElem <= 128) {
    KernelERI_RysDF_Ld0Warp128<NROOTS, STRIDE><<<blocks, threads, 0, stream>>>(
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
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        la,
        lb,
        lc,
        eri_out);
    return;
  }
  KernelERI_RysDF_Ld0Warp<NROOTS, STRIDE, 8><<<blocks, threads, 0, stream>>>(
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
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      la,
      lb,
      lc,
      eri_out);
}

// ---- DF int3c2e with non-expanded AO contractions (AO-side only) ----
//
// This is a DF-specific kernel family that:
// - assumes ld=0 (dummy s shell),
// - consumes the same (spAB, spCD) task encoding as the tile kernels,
// - writes directly to X(nao,nao,naux) with symmetric fill,
// - supports AO general contractions via per-shell coefficient matrices.
//
// Current scope:
// - AO contractions only (aux shells are still treated as expanded shells).
// - nctr per AO shell is limited to kCtrMax for now.
constexpr int kCtrMax = 5;
[[maybe_unused]] constexpr int kCtrPairMax = kCtrMax * kCtrMax;  // 25

template <int NROOTS, int STRIDE, int CTR_PAIR_MAX>
struct DFInt3c2eContractedLd0SubwarpScratch {
  int8_t Ax[kNcartMax], Ay[kNcartMax], Az[kNcartMax];
  int8_t Bx[kNcartMax], By[kNcartMax], Bz[kNcartMax];
  int8_t Cx[kNcartMax], Cy[kNcartMax], Cz[kNcartMax];

  double Gx[STRIDE * STRIDE];
  double Gy[STRIDE * STRIDE];
  double Gz[STRIDE * STRIDE];

  double scale;
  double roots[NROOTS];
  double weights[NROOTS];

  double xij_pow[kLMax + 1], xkl_pow[kLMax + 1];
  double yij_pow[kLMax + 1], ykl_pow[kLMax + 1];
  double zij_pow[kLMax + 1], zkl_pow[kLMax + 1];

  double Axyz[3];
  double Cxyz[3];

  double wab[CTR_PAIR_MAX];
};

template <int NROOTS, int STRIDE, int CTR_MAX>
__global__ void KernelDFInt3c2e_RysContractedLd0Subwarp8(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out) {
  // 4 tasks per warp (subwarp8). Each lane computes up to 4 elements:
  //   e = lane8 + pass*8, pass=0..3 -> supports up to nElem<=32.
  constexpr int CTR_PAIR_MAX = CTR_MAX * CTR_MAX;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  if (warp_id >= kMaxWarpsPerBlock) return;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int warp_global = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;

  const int subwarp = lane >> 3;  // 0..3
  const int lane8 = lane & 7;
  const unsigned mask = 0xFFu << (8 * subwarp);

  const int t = warp_global * 4 + subwarp;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nAB = nA * nB;
  const int nElem = nAB * nC;
  if (nElem <= 0 || nElem > 32) return;
  const bool lb_is_zero = (lb == 0);

  // Shared scratch: one entry per task in this block.
  extern __shared__ unsigned char smem[];
  using ScratchT = DFInt3c2eContractedLd0SubwarpScratch<NROOTS, STRIDE, CTR_PAIR_MAX>;
  auto* scratch = reinterpret_cast<ScratchT*>(smem);
  const int task_local = warp_id * 4 + subwarp;
  ScratchT* s = scratch + task_local;

  const bool leader = (lane8 == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);  // combined-basis aux shell index
  const int Psh = C - n_shell_ao;
  if (Psh < 0) return;

  const int a0_shell = static_cast<int>(ao_shell_ao_start[A]);
  const int b0_shell = static_cast<int>(ao_shell_ao_start[B]);
  const int nctrA = static_cast<int>(ao_shell_nctr[A]);
  const int nctrB = static_cast<int>(ao_shell_nctr[B]);
  if (nctrA <= 0 || nctrB <= 0) return;
  if (nctrA > CTR_MAX || nctrB > CTR_MAX) return;
  const int nctrAB = nctrA * nctrB;

  const int coefA0 = static_cast<int>(ao_shell_coef_start[A]);
  const int coefB0 = static_cast<int>(ao_shell_coef_start[B]);
  const int nprimB = static_cast<int>(shell_nprim[B]);
  if (nprimB <= 0) return;

  const int p0_shell = static_cast<int>(aux_shell_ao_start[Psh]) - aux_p0_block;

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  if (leader) {
    fill_cart_comp(la, s->Ax, s->Ay, s->Az);
    if (!lb_is_zero) fill_cart_comp(lb, s->Bx, s->By, s->Bz);
    fill_cart_comp(lc, s->Cx, s->Cy, s->Cz);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    s->Axyz[0] = Ax;
    s->Axyz[1] = Ay;
    s->Axyz[2] = Az;
    s->Cxyz[0] = Cx;
    s->Cxyz[1] = Cy;
    s->Cxyz[2] = Cz;

    s->xij_pow[0] = 1.0;
    s->yij_pow[0] = 1.0;
    s->zij_pow[0] = 1.0;
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    s->xkl_pow[0] = 1.0;
    s->ykl_pow[0] = 1.0;
    s->zkl_pow[0] = 1.0;
    for (int p = 1; p <= kLMax; ++p) {
      s->xkl_pow[p] = 0.0;
      s->ykl_pow[p] = 0.0;
      s->zkl_pow[p] = 0.0;
    }

    if (!lb_is_zero) {
      const double Bx = shell_cx[B];
      const double By = shell_cy[B];
      const double Bz = shell_cz[B];
      const double xij = Ax - Bx;
      const double yij = Ay - By;
      const double zij = Az - Bz;
      for (int p = 1; p <= kLMax; ++p) {
        s->xij_pow[p] = s->xij_pow[p - 1] * xij;
        s->yij_pow[p] = s->yij_pow[p - 1] * yij;
        s->zij_pow[p] = s->zij_pow[p - 1] * zij;
      }
    }
  }
  __syncwarp(mask);

  // Each lane owns up to 4 elements: e0..e3.
  const int e0 = lane8;
  const int e1 = lane8 + 8;
  const int e2 = lane8 + 16;
  const int e3 = lane8 + 24;

  const bool has0 = (e0 < nElem);
  const bool has1 = (e1 < nElem);
  const bool has2 = (e2 < nElem);
  const bool has3 = (e3 < nElem);

  int ia0 = 0, ib0 = 0, ic0 = 0;
  int ia1 = 0, ib1 = 0, ic1 = 0;
  int ia2 = 0, ib2 = 0, ic2 = 0;
  int ia3 = 0, ib3 = 0, ic3 = 0;

  int iax0 = 0, iay0 = 0, iaz0 = 0, ibx0 = 0, iby0 = 0, ibz0 = 0, icx0 = 0, icy0 = 0, icz0 = 0;
  int iax1 = 0, iay1 = 0, iaz1 = 0, ibx1 = 0, iby1 = 0, ibz1 = 0, icx1 = 0, icy1 = 0, icz1 = 0;
  int iax2 = 0, iay2 = 0, iaz2 = 0, ibx2 = 0, iby2 = 0, ibz2 = 0, icx2 = 0, icy2 = 0, icz2 = 0;
  int iax3 = 0, iay3 = 0, iaz3 = 0, ibx3 = 0, iby3 = 0, ibz3 = 0, icx3 = 0, icy3 = 0, icz3 = 0;

  if (has0) {
    const int ab = e0 / nC;
    ic0 = e0 - ab * nC;
    ia0 = ab / nB;
    ib0 = ab - ia0 * nB;
    iax0 = static_cast<int>(s->Ax[ia0]);
    iay0 = static_cast<int>(s->Ay[ia0]);
    iaz0 = static_cast<int>(s->Az[ia0]);
    if (!lb_is_zero) {
      ibx0 = static_cast<int>(s->Bx[ib0]);
      iby0 = static_cast<int>(s->By[ib0]);
      ibz0 = static_cast<int>(s->Bz[ib0]);
    }
    icx0 = static_cast<int>(s->Cx[ic0]);
    icy0 = static_cast<int>(s->Cy[ic0]);
    icz0 = static_cast<int>(s->Cz[ic0]);
  }
  if (has1) {
    const int ab = e1 / nC;
    ic1 = e1 - ab * nC;
    ia1 = ab / nB;
    ib1 = ab - ia1 * nB;
    iax1 = static_cast<int>(s->Ax[ia1]);
    iay1 = static_cast<int>(s->Ay[ia1]);
    iaz1 = static_cast<int>(s->Az[ia1]);
    if (!lb_is_zero) {
      ibx1 = static_cast<int>(s->Bx[ib1]);
      iby1 = static_cast<int>(s->By[ib1]);
      ibz1 = static_cast<int>(s->Bz[ib1]);
    }
    icx1 = static_cast<int>(s->Cx[ic1]);
    icy1 = static_cast<int>(s->Cy[ic1]);
    icz1 = static_cast<int>(s->Cz[ic1]);
  }
  if (has2) {
    const int ab = e2 / nC;
    ic2 = e2 - ab * nC;
    ia2 = ab / nB;
    ib2 = ab - ia2 * nB;
    iax2 = static_cast<int>(s->Ax[ia2]);
    iay2 = static_cast<int>(s->Ay[ia2]);
    iaz2 = static_cast<int>(s->Az[ia2]);
    if (!lb_is_zero) {
      ibx2 = static_cast<int>(s->Bx[ib2]);
      iby2 = static_cast<int>(s->By[ib2]);
      ibz2 = static_cast<int>(s->Bz[ib2]);
    }
    icx2 = static_cast<int>(s->Cx[ic2]);
    icy2 = static_cast<int>(s->Cy[ic2]);
    icz2 = static_cast<int>(s->Cz[ic2]);
  }
  if (has3) {
    const int ab = e3 / nC;
    ic3 = e3 - ab * nC;
    ia3 = ab / nB;
    ib3 = ab - ia3 * nB;
    iax3 = static_cast<int>(s->Ax[ia3]);
    iay3 = static_cast<int>(s->Ay[ia3]);
    iaz3 = static_cast<int>(s->Az[ia3]);
    if (!lb_is_zero) {
      ibx3 = static_cast<int>(s->Bx[ib3]);
      iby3 = static_cast<int>(s->By[ib3]);
      ibz3 = static_cast<int>(s->Bz[ib3]);
    }
    icx3 = static_cast<int>(s->Cx[ic3]);
    icy3 = static_cast<int>(s->Cy[ic3]);
    icz3 = static_cast<int>(s->Cz[ic3]);
  }

  double acc0[CTR_PAIR_MAX];
  double acc1[CTR_PAIR_MAX];
  double acc2[CTR_PAIR_MAX];
  double acc3[CTR_PAIR_MAX];
  for (int c = 0; c < CTR_PAIR_MAX; ++c) {
    acc0[c] = 0.0;
    acc1[c] = 0.0;
    acc2[c] = 0.0;
    acc3[c] = 0.0;
  }

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double Kab = 0.0;
    if (leader) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      Kab = pair_cK[ki];  // AO: Kab_geom (prim coef=1)

      const int ipA = ip / nprimB;
      const int ipB = ip - ipA * nprimB;
      for (int ca = 0; ca < nctrA; ++ca) {
        const double cA = ao_prim_coef[coefA0 + ipA * nctrA + ca];
        for (int cb = 0; cb < nctrB; ++cb) {
          const double cB = ao_prim_coef[coefB0 + ipB * nctrB + cb];
          s->wab[ca * nctrB + cb] = cA * cB;
        }
      }
    }
    __syncwarp(mask);

    double val0 = 0.0;
    double val1 = 0.0;
    double val2 = 0.0;
    double val3 = 0.0;

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (leader) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];  // aux coef (dummy exp=0 => Kab=1)

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * Kab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, s->roots, s->weights);
      }
      __syncwarp(mask);

      for (int u = 0; u < NROOTS; ++u) {
        if (leader) {
          const double x = s->roots[u];
          const double w = s->weights[u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = s->Axyz[0];
          const double Ay = s->Axyz[1];
          const double Az = s->Axyz[2];
          const double Cx = s->Cxyz[0];
          const double Cy = s->Cxyz[1];
          const double Cz = s->Cxyz[2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(s->Gx, nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gy, nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(s->Gz, nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          s->scale = base * w;
        }
        __syncwarp(mask);

        const double scale = s->scale;
        if (has0) {
          const double Ix = lb_is_zero ? s->Gx[iax0 * STRIDE + icx0]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax0, ibx0, icx0, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay0 * STRIDE + icy0]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay0, iby0, icy0, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz0 * STRIDE + icz0]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz0, ibz0, icz0, 0, s->zij_pow, s->zkl_pow);
          val0 += scale * (Ix * Iy * Iz);
        }
        if (has1) {
          const double Ix = lb_is_zero ? s->Gx[iax1 * STRIDE + icx1]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax1, ibx1, icx1, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay1 * STRIDE + icy1]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay1, iby1, icy1, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz1 * STRIDE + icz1]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz1, ibz1, icz1, 0, s->zij_pow, s->zkl_pow);
          val1 += scale * (Ix * Iy * Iz);
        }
        if (has2) {
          const double Ix = lb_is_zero ? s->Gx[iax2 * STRIDE + icx2]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax2, ibx2, icx2, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay2 * STRIDE + icy2]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay2, iby2, icy2, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz2 * STRIDE + icz2]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz2, ibz2, icz2, 0, s->zij_pow, s->zkl_pow);
          val2 += scale * (Ix * Iy * Iz);
        }
        if (has3) {
          const double Ix = lb_is_zero ? s->Gx[iax3 * STRIDE + icx3]
                                       : shift_from_G_stride<STRIDE>(s->Gx, iax3, ibx3, icx3, 0, s->xij_pow, s->xkl_pow);
          const double Iy = lb_is_zero ? s->Gy[iay3 * STRIDE + icy3]
                                       : shift_from_G_stride<STRIDE>(s->Gy, iay3, iby3, icy3, 0, s->yij_pow, s->ykl_pow);
          const double Iz = lb_is_zero ? s->Gz[iaz3 * STRIDE + icz3]
                                       : shift_from_G_stride<STRIDE>(s->Gz, iaz3, ibz3, icz3, 0, s->zij_pow, s->zkl_pow);
          val3 += scale * (Ix * Iy * Iz);
        }
        __syncwarp(mask);
      }
    }

    if (has0) {
      for (int c = 0; c < nctrAB; ++c) acc0[c] += s->wab[c] * val0;
    }
    if (has1) {
      for (int c = 0; c < nctrAB; ++c) acc1[c] += s->wab[c] * val1;
    }
    if (has2) {
      for (int c = 0; c < nctrAB; ++c) acc2[c] += s->wab[c] * val2;
    }
    if (has3) {
      for (int c = 0; c < nctrAB; ++c) acc3[c] += s->wab[c] * val3;
    }
    __syncwarp(mask);
  }

  if (has0) {
    const int p = p0_shell + ic0;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia0;
      const int b = b0_shell + cb * nB + ib0;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc0[c];
      if (a != b) X_out[idx_bap] = acc0[c];
    }
  }
  if (has1) {
    const int p = p0_shell + ic1;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia1;
      const int b = b0_shell + cb * nB + ib1;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc1[c];
      if (a != b) X_out[idx_bap] = acc1[c];
    }
  }
  if (has2) {
    const int p = p0_shell + ic2;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia2;
      const int b = b0_shell + cb * nB + ib2;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc2[c];
      if (a != b) X_out[idx_bap] = acc2[c];
    }
  }
  if (has3) {
    const int p = p0_shell + ic3;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia3;
      const int b = b0_shell + cb * nB + ib3;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc3[c];
      if (a != b) X_out[idx_bap] = acc3[c];
    }
  }
}

template <int NROOTS, int CTR_MAX>
static inline void launch_df_int3c2e_rys_contracted_ld0_subwarp8_stride5(
    int blocks,
    int threads,
    int tasks_per_block,
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,
    cudaStream_t stream) {
  using ScratchT = DFInt3c2eContractedLd0SubwarpScratch<NROOTS, 5, CTR_MAX * CTR_MAX>;
  const size_t shared_bytes = static_cast<size_t>(tasks_per_block) * sizeof(ScratchT);
  KernelDFInt3c2e_RysContractedLd0Subwarp8<NROOTS, 5, CTR_MAX><<<blocks, threads, shared_bytes, stream>>>(
      task_spAB,
      task_spCD,
      ntasks,
      sp_A,
      sp_B,
      sp_pair_start,
      sp_npair,
      shell_nprim,
      shell_cx,
      shell_cy,
      shell_cz,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      ao_shell_ao_start,
      ao_shell_nctr,
      ao_shell_coef_start,
      ao_prim_coef,
      aux_shell_ao_start,
      n_shell_ao,
      nao,
      naux,
      aux_p0_block,
      la,
      lb,
      lc,
      X_out);
}

template <int NROOTS, int STRIDE, int CTR_MAX>
__global__ void KernelDFInt3c2e_RysContractedLd0Warp64(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out) {
  constexpr int CTR_PAIR_MAX = CTR_MAX * CTR_MAX;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  if (warp_id >= kMaxWarpsPerBlock) return;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;
  if (nElem <= 0 || nElem > 64) return;
  const bool lb_is_zero = (lb == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);  // combined-basis aux shell index
  const int Psh = C - n_shell_ao;
  if (Psh < 0) return;

  const int a0_shell = static_cast<int>(ao_shell_ao_start[A]);
  const int b0_shell = static_cast<int>(ao_shell_ao_start[B]);
  const int nctrA = static_cast<int>(ao_shell_nctr[A]);
  const int nctrB = static_cast<int>(ao_shell_nctr[B]);
  if (nctrA <= 0 || nctrB <= 0) return;
  if (nctrA > CTR_MAX || nctrB > CTR_MAX) return;
  const int nctrAB = nctrA * nctrB;

  const int coefA0 = static_cast<int>(ao_shell_coef_start[A]);
  const int coefB0 = static_cast<int>(ao_shell_coef_start[B]);
  const int nprimB = static_cast<int>(shell_nprim[B]);
  if (nprimB <= 0) return;

  const int p0_shell = static_cast<int>(aux_shell_ao_start[Psh]) - aux_p0_block;

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  __shared__ int8_t sh_Ax[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Ay[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Az[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_By[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Bz[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cx[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cy[kMaxWarpsPerBlock][kNcartMax];
  __shared__ int8_t sh_Cz[kMaxWarpsPerBlock][kNcartMax];

  __shared__ double sh_Gx[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gy[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_Gz[kMaxWarpsPerBlock][STRIDE * STRIDE];
  __shared__ double sh_scale[kMaxWarpsPerBlock];
  __shared__ double sh_roots[kMaxWarpsPerBlock][NROOTS];
  __shared__ double sh_weights[kMaxWarpsPerBlock][NROOTS];

  __shared__ double sh_Axyz[kMaxWarpsPerBlock][3];
  __shared__ double sh_Cxyz[kMaxWarpsPerBlock][3];

  __shared__ double sh_xij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_xkl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_yij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_ykl_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zij_pow[kMaxWarpsPerBlock][kLMax + 1];
  __shared__ double sh_zkl_pow[kMaxWarpsPerBlock][kLMax + 1];

  __shared__ double sh_wab[kMaxWarpsPerBlock][CTR_PAIR_MAX];

  if (lane == 0) {
    fill_cart_comp(la, sh_Ax[warp_id], sh_Ay[warp_id], sh_Az[warp_id]);
    if (!lb_is_zero) fill_cart_comp(lb, sh_Bx[warp_id], sh_By[warp_id], sh_Bz[warp_id]);
    fill_cart_comp(lc, sh_Cx[warp_id], sh_Cy[warp_id], sh_Cz[warp_id]);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Bx = shell_cx[B];
    const double By = shell_cy[B];
    const double Bz = shell_cz[B];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    sh_Axyz[warp_id][0] = Ax;
    sh_Axyz[warp_id][1] = Ay;
    sh_Axyz[warp_id][2] = Az;
    sh_Cxyz[warp_id][0] = Cx;
    sh_Cxyz[warp_id][1] = Cy;
    sh_Cxyz[warp_id][2] = Cz;

    const double xij = Ax - Bx;
    const double yij = Ay - By;
    const double zij = Az - Bz;
    sh_xij_pow[warp_id][0] = 1.0;
    sh_yij_pow[warp_id][0] = 1.0;
    sh_zij_pow[warp_id][0] = 1.0;
    for (int l = 1; l <= kLMax; ++l) {
      sh_xij_pow[warp_id][l] = sh_xij_pow[warp_id][l - 1] * xij;
      sh_yij_pow[warp_id][l] = sh_yij_pow[warp_id][l - 1] * yij;
      sh_zij_pow[warp_id][l] = sh_zij_pow[warp_id][l - 1] * zij;
    }
    // ld=0 => D is an s-shell: shift() reads only xkl_pow[0]=1.
    sh_xkl_pow[warp_id][0] = 1.0;
    sh_ykl_pow[warp_id][0] = 1.0;
    sh_zkl_pow[warp_id][0] = 1.0;
    for (int l = 1; l <= kLMax; ++l) {
      sh_xkl_pow[warp_id][l] = 0.0;
      sh_ykl_pow[warp_id][l] = 0.0;
      sh_zkl_pow[warp_id][l] = 0.0;
    }
  }
  __syncwarp();

  // Lane owns up to 2 elements (since nElem<=64).
  const int e0 = lane;
  const int e1 = lane + 32;
  const bool has0 = (e0 < nElem);
  const bool has1 = (e1 < nElem);

  int ia0 = 0, ib0 = 0, ic0 = 0;
  int ia1 = 0, ib1 = 0, ic1 = 0;
  int iax0 = 0, iay0 = 0, iaz0 = 0, ibx0 = 0, iby0 = 0, ibz0 = 0, icx0 = 0, icy0 = 0, icz0 = 0;
  int iax1 = 0, iay1 = 0, iaz1 = 0, ibx1 = 0, iby1 = 0, ibz1 = 0, icx1 = 0, icy1 = 0, icz1 = 0;
  if (has0) {
    const int iab = e0 / nC;
    ic0 = e0 - iab * nC;
    ia0 = iab / nB;
    ib0 = iab - ia0 * nB;
    iax0 = static_cast<int>(sh_Ax[warp_id][ia0]);
    iay0 = static_cast<int>(sh_Ay[warp_id][ia0]);
    iaz0 = static_cast<int>(sh_Az[warp_id][ia0]);
    if (!lb_is_zero) {
      ibx0 = static_cast<int>(sh_Bx[warp_id][ib0]);
      iby0 = static_cast<int>(sh_By[warp_id][ib0]);
      ibz0 = static_cast<int>(sh_Bz[warp_id][ib0]);
    }
    icx0 = static_cast<int>(sh_Cx[warp_id][ic0]);
    icy0 = static_cast<int>(sh_Cy[warp_id][ic0]);
    icz0 = static_cast<int>(sh_Cz[warp_id][ic0]);
  }
  if (has1) {
    const int iab = e1 / nC;
    ic1 = e1 - iab * nC;
    ia1 = iab / nB;
    ib1 = iab - ia1 * nB;
    iax1 = static_cast<int>(sh_Ax[warp_id][ia1]);
    iay1 = static_cast<int>(sh_Ay[warp_id][ia1]);
    iaz1 = static_cast<int>(sh_Az[warp_id][ia1]);
    if (!lb_is_zero) {
      ibx1 = static_cast<int>(sh_Bx[warp_id][ib1]);
      iby1 = static_cast<int>(sh_By[warp_id][ib1]);
      ibz1 = static_cast<int>(sh_Bz[warp_id][ib1]);
    }
    icx1 = static_cast<int>(sh_Cx[warp_id][ic1]);
    icy1 = static_cast<int>(sh_Cy[warp_id][ic1]);
    icz1 = static_cast<int>(sh_Cz[warp_id][ic1]);
  }

  double acc0[CTR_PAIR_MAX];
  double acc1[CTR_PAIR_MAX];
  for (int c = 0; c < CTR_PAIR_MAX; ++c) {
    acc0[c] = 0.0;
    acc1[c] = 0.0;
  }

  for (int ip = 0; ip < nPairAB; ++ip) {
    const int ki = baseAB + ip;
    double p = 0.0;
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double Kab = 0.0;
    if (lane == 0) {
      p = pair_eta[ki];
      Px = pair_Px[ki];
      Py = pair_Py[ki];
      Pz = pair_Pz[ki];
      Kab = pair_cK[ki];  // AO: Kab_geom (prim coef=1)

      const int ipA = ip / nprimB;
      const int ipB = ip - ipA * nprimB;
      for (int ca = 0; ca < nctrA; ++ca) {
        const double cA = ao_prim_coef[coefA0 + ipA * nctrA + ca];
        for (int cb = 0; cb < nctrB; ++cb) {
          const double cB = ao_prim_coef[coefB0 + ipB * nctrB + cb];
          sh_wab[warp_id][ca * nctrB + cb] = cA * cB;
        }
      }
    }
    __syncwarp();

    double val0 = 0.0;
    double val1 = 0.0;

    for (int jp = 0; jp < nPairCD; ++jp) {
      const int kj = baseCD + jp;
      double q = 0.0;
      double Qx = 0.0;
      double Qy = 0.0;
      double Qz = 0.0;
      double cKcd = 0.0;
      double denom = 0.0;
      double base = 0.0;
      if (lane == 0) {
        q = pair_eta[kj];
        Qx = pair_Px[kj];
        Qy = pair_Py[kj];
        Qz = pair_Pz[kj];
        cKcd = pair_cK[kj];  // aux coef (dummy exp=0 => Kab=1)

        const double dx = Px - Qx;
        const double dy = Py - Qy;
        const double dz = Pz - Qz;
        const double PQ2 = dx * dx + dy * dy + dz * dz;

        denom = p + q;
        const double omega = p * q / denom;
        const double T = omega * PQ2;

        base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * Kab * cKcd;

        cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots[warp_id], sh_weights[warp_id]);
      }
      __syncwarp();

      for (int u = 0; u < NROOTS; ++u) {
        if (lane == 0) {
          const double x = sh_roots[warp_id][u];
          const double w = sh_weights[warp_id][u];

          const double inv_denom = 1.0 / denom;
          const double B0 = x * 0.5 * inv_denom;
          const double B1 = (1.0 - x) * 0.5 / p + B0;
          const double B1p = (1.0 - x) * 0.5 / q + B0;

          const double Ax = sh_Axyz[warp_id][0];
          const double Ay = sh_Axyz[warp_id][1];
          const double Az = sh_Axyz[warp_id][2];
          const double Cx = sh_Cxyz[warp_id][0];
          const double Cy = sh_Cxyz[warp_id][1];
          const double Cz = sh_Cxyz[warp_id][2];

          const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
          const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
          const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

          const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
          const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
          const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

          compute_G_stride<STRIDE>(sh_Gx[warp_id], nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gy[warp_id], nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
          compute_G_stride<STRIDE>(sh_Gz[warp_id], nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

          sh_scale[warp_id] = base * w;
        }
        __syncwarp();

        const double scale = sh_scale[warp_id];

        if (has0) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax0 * STRIDE + icx0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax0, ibx0, icx0, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay0 * STRIDE + icy0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay0, iby0, icy0, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz0 * STRIDE + icz0]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz0, ibz0, icz0, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          val0 += scale * (Ix * Iy * Iz);
        }
        if (has1) {
          const double Ix = lb_is_zero ? sh_Gx[warp_id][iax1 * STRIDE + icx1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gx[warp_id], iax1, ibx1, icx1, 0, sh_xij_pow[warp_id], sh_xkl_pow[warp_id]);
          const double Iy = lb_is_zero ? sh_Gy[warp_id][iay1 * STRIDE + icy1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gy[warp_id], iay1, iby1, icy1, 0, sh_yij_pow[warp_id], sh_ykl_pow[warp_id]);
          const double Iz = lb_is_zero ? sh_Gz[warp_id][iaz1 * STRIDE + icz1]
                                       : shift_from_G_stride<STRIDE>(
                                             sh_Gz[warp_id], iaz1, ibz1, icz1, 0, sh_zij_pow[warp_id], sh_zkl_pow[warp_id]);
          val1 += scale * (Ix * Iy * Iz);
        }
        __syncwarp();
      }
    }

    if (has0) {
      for (int c = 0; c < nctrAB; ++c) {
        acc0[c] += sh_wab[warp_id][c] * val0;
      }
    }
    if (has1) {
      for (int c = 0; c < nctrAB; ++c) {
        acc1[c] += sh_wab[warp_id][c] * val1;
      }
    }
    __syncwarp();
  }

  if (has0) {
    const int p = p0_shell + ic0;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia0;
      const int b = b0_shell + cb * nB + ib0;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc0[c];
      if (a != b) X_out[idx_bap] = acc0[c];
    }
  }
  if (has1) {
    const int p = p0_shell + ic1;
    for (int c = 0; c < nctrAB; ++c) {
      const int ca = c / nctrB;
      const int cb = c - ca * nctrB;
      const int a = a0_shell + ca * nA + ia1;
      const int b = b0_shell + cb * nB + ib1;
      const int64_t idx_abp =
          (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      const int64_t idx_bap =
          (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
          static_cast<int64_t>(p);
      X_out[idx_abp] = acc1[c];
      if (a != b) X_out[idx_bap] = acc1[c];
    }
  }
}

template <int NROOTS, int STRIDE, int CTR_MAX>
__global__ void KernelDFInt3c2e_RysContractedLd0Block(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out) {
  constexpr int CTR_PAIR_MAX = CTR_MAX * CTR_MAX;
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;

  if (la < 0 || lb < 0 || lc < 0) return;
  if (la > kLMax || lb > kLMax || lc > kLMax) return;

  const int nA = ncart(la);
  const int nB = ncart(lb);
  const int nC = ncart(lc);
  const int nElem = nA * nB * nC;
  const bool lb_is_zero = (lb == 0);

  const int spAB = static_cast<int>(task_spAB[t]);
  const int spCD = static_cast<int>(task_spCD[t]);
  const int A = static_cast<int>(sp_A[spAB]);
  const int B = static_cast<int>(sp_B[spAB]);
  const int C = static_cast<int>(sp_A[spCD]);  // combined-basis aux shell index
  const int Psh = C - n_shell_ao;
  if (Psh < 0) return;

  const int a0_shell = static_cast<int>(ao_shell_ao_start[A]);
  const int b0_shell = static_cast<int>(ao_shell_ao_start[B]);
  const int nctrA = static_cast<int>(ao_shell_nctr[A]);
  const int nctrB = static_cast<int>(ao_shell_nctr[B]);
  if (nctrA <= 0 || nctrB <= 0) return;
  if (nctrA > CTR_MAX || nctrB > CTR_MAX) return;
  const int nctrAB = nctrA * nctrB;

  const int coefA0 = static_cast<int>(ao_shell_coef_start[A]);
  const int coefB0 = static_cast<int>(ao_shell_coef_start[B]);
  const int nprimB = static_cast<int>(shell_nprim[B]);
  if (nprimB <= 0) return;

  const int p0_shell = static_cast<int>(aux_shell_ao_start[Psh]) - aux_p0_block;

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  if (nmax >= STRIDE || mmax >= STRIDE) return;

  const int baseAB = static_cast<int>(sp_pair_start[spAB]);
  const int baseCD = static_cast<int>(sp_pair_start[spCD]);
  const int nPairAB = static_cast<int>(sp_npair[spAB]);
  const int nPairCD = static_cast<int>(sp_npair[spCD]);

  __shared__ int8_t sh_Ax[kNcartMax];
  __shared__ int8_t sh_Ay[kNcartMax];
  __shared__ int8_t sh_Az[kNcartMax];
  __shared__ int8_t sh_Bx[kNcartMax];
  __shared__ int8_t sh_By[kNcartMax];
  __shared__ int8_t sh_Bz[kNcartMax];
  __shared__ int8_t sh_Cx[kNcartMax];
  __shared__ int8_t sh_Cy[kNcartMax];
  __shared__ int8_t sh_Cz[kNcartMax];

  __shared__ double sh_Gx[STRIDE * STRIDE];
  __shared__ double sh_Gy[STRIDE * STRIDE];
  __shared__ double sh_Gz[STRIDE * STRIDE];
  __shared__ double sh_scale;
  __shared__ double sh_roots[NROOTS];
  __shared__ double sh_weights[NROOTS];

  __shared__ double sh_xij_pow[kLMax + 1];
  __shared__ double sh_xkl_pow[kLMax + 1];
  __shared__ double sh_yij_pow[kLMax + 1];
  __shared__ double sh_ykl_pow[kLMax + 1];
  __shared__ double sh_zij_pow[kLMax + 1];
  __shared__ double sh_zkl_pow[kLMax + 1];

  __shared__ double sh_Axyz[3];
  __shared__ double sh_Cxyz[3];

  __shared__ double sh_wab[CTR_PAIR_MAX];

  if (threadIdx.x == 0) {
    fill_cart_comp(la, sh_Ax, sh_Ay, sh_Az);
    if (!lb_is_zero) fill_cart_comp(lb, sh_Bx, sh_By, sh_Bz);
    fill_cart_comp(lc, sh_Cx, sh_Cy, sh_Cz);

    const double Ax = shell_cx[A];
    const double Ay = shell_cy[A];
    const double Az = shell_cz[A];
    const double Bx = shell_cx[B];
    const double By = shell_cy[B];
    const double Bz = shell_cz[B];
    const double Cx = shell_cx[C];
    const double Cy = shell_cy[C];
    const double Cz = shell_cz[C];

    sh_Axyz[0] = Ax;
    sh_Axyz[1] = Ay;
    sh_Axyz[2] = Az;
    sh_Cxyz[0] = Cx;
    sh_Cxyz[1] = Cy;
    sh_Cxyz[2] = Cz;

    const double xij = Ax - Bx;
    const double yij = Ay - By;
    const double zij = Az - Bz;
    sh_xij_pow[0] = 1.0;
    sh_yij_pow[0] = 1.0;
    sh_zij_pow[0] = 1.0;
    for (int l = 1; l <= kLMax; ++l) {
      sh_xij_pow[l] = sh_xij_pow[l - 1] * xij;
      sh_yij_pow[l] = sh_yij_pow[l - 1] * yij;
      sh_zij_pow[l] = sh_zij_pow[l - 1] * zij;
    }
    sh_xkl_pow[0] = 1.0;
    sh_ykl_pow[0] = 1.0;
    sh_zkl_pow[0] = 1.0;
    for (int l = 1; l <= kLMax; ++l) {
      sh_xkl_pow[l] = 0.0;
      sh_ykl_pow[l] = 0.0;
      sh_zkl_pow[l] = 0.0;
    }
  }
  __syncthreads();

  for (int e_base = 0; e_base < nElem; e_base += static_cast<int>(blockDim.x)) {
    const int e = e_base + static_cast<int>(threadIdx.x);
    const bool active = (e < nElem);

    int ia = 0, ib = 0, ic = 0;
    int iax = 0, iay = 0, iaz = 0, ibx = 0, iby = 0, ibz = 0, icx = 0, icy = 0, icz = 0;
    if (active) {
      const int iab = e / nC;
      ic = e - iab * nC;
      ia = iab / nB;
      ib = iab - ia * nB;
      iax = static_cast<int>(sh_Ax[ia]);
      iay = static_cast<int>(sh_Ay[ia]);
      iaz = static_cast<int>(sh_Az[ia]);
      if (!lb_is_zero) {
        ibx = static_cast<int>(sh_Bx[ib]);
        iby = static_cast<int>(sh_By[ib]);
        ibz = static_cast<int>(sh_Bz[ib]);
      }
      icx = static_cast<int>(sh_Cx[ic]);
      icy = static_cast<int>(sh_Cy[ic]);
      icz = static_cast<int>(sh_Cz[ic]);
    }

    double acc[CTR_PAIR_MAX];
    for (int c = 0; c < CTR_PAIR_MAX; ++c) acc[c] = 0.0;

    for (int ip = 0; ip < nPairAB; ++ip) {
      const int ki = baseAB + ip;
      double p = 0.0;
      double Px = 0.0;
      double Py = 0.0;
      double Pz = 0.0;
      double Kab = 0.0;
      if (threadIdx.x == 0) {
        p = pair_eta[ki];
        Px = pair_Px[ki];
        Py = pair_Py[ki];
        Pz = pair_Pz[ki];
        Kab = pair_cK[ki];

        const int ipA = ip / nprimB;
        const int ipB = ip - ipA * nprimB;
        for (int ca = 0; ca < nctrA; ++ca) {
          const double cA = ao_prim_coef[coefA0 + ipA * nctrA + ca];
          for (int cb = 0; cb < nctrB; ++cb) {
            const double cB = ao_prim_coef[coefB0 + ipB * nctrB + cb];
            sh_wab[ca * nctrB + cb] = cA * cB;
          }
        }
      }
      __syncthreads();

      double val = 0.0;
      for (int jp = 0; jp < nPairCD; ++jp) {
        const int kj = baseCD + jp;
        double q = 0.0;
        double Qx = 0.0;
        double Qy = 0.0;
        double Qz = 0.0;
        double cKcd = 0.0;
        double denom = 0.0;
        double base = 0.0;
        if (threadIdx.x == 0) {
          q = pair_eta[kj];
          Qx = pair_Px[kj];
          Qy = pair_Py[kj];
          Qz = pair_Pz[kj];
          cKcd = pair_cK[kj];

          const double dx = Px - Qx;
          const double dy = Py - Qy;
          const double dz = Pz - Qz;
          const double PQ2 = dx * dx + dy * dy + dz * dz;

          denom = p + q;
          const double omega = p * q / denom;
          const double T = omega * PQ2;

          base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * Kab * cKcd;

          cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);
        }
        __syncthreads();

        for (int u = 0; u < NROOTS; ++u) {
          if (threadIdx.x == 0) {
            const double x = sh_roots[u];
            const double w = sh_weights[u];

            const double inv_denom = 1.0 / denom;
            const double B0 = x * 0.5 * inv_denom;
            const double B1 = (1.0 - x) * 0.5 / p + B0;
            const double B1p = (1.0 - x) * 0.5 / q + B0;

            const double Ax = sh_Axyz[0];
            const double Ay = sh_Axyz[1];
            const double Az = sh_Axyz[2];
            const double Cx = sh_Cxyz[0];
            const double Cy = sh_Cxyz[1];
            const double Cz = sh_Cxyz[2];

            const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);
            const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);
            const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);

            const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);
            const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);
            const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);

            compute_G_stride<STRIDE>(sh_Gx, nmax, mmax, Cx_, Cpx_, B0, B1, B1p);
            compute_G_stride<STRIDE>(sh_Gy, nmax, mmax, Cy_, Cpy_, B0, B1, B1p);
            compute_G_stride<STRIDE>(sh_Gz, nmax, mmax, Cz_, Cpz_, B0, B1, B1p);

            sh_scale = base * w;
          }
          __syncthreads();

          if (active) {
            const double Ix = lb_is_zero ? sh_Gx[iax * STRIDE + icx]
                                         : shift_from_G_stride<STRIDE>(sh_Gx, iax, ibx, icx, 0, sh_xij_pow, sh_xkl_pow);
            const double Iy = lb_is_zero ? sh_Gy[iay * STRIDE + icy]
                                         : shift_from_G_stride<STRIDE>(sh_Gy, iay, iby, icy, 0, sh_yij_pow, sh_ykl_pow);
            const double Iz = lb_is_zero ? sh_Gz[iaz * STRIDE + icz]
                                         : shift_from_G_stride<STRIDE>(sh_Gz, iaz, ibz, icz, 0, sh_zij_pow, sh_zkl_pow);
            val += sh_scale * (Ix * Iy * Iz);
          }
          __syncthreads();
        }
      }

      if (active) {
        for (int c = 0; c < nctrAB; ++c) {
          acc[c] += sh_wab[c] * val;
        }
      }
      __syncthreads();
    }

    if (active) {
      const int p = p0_shell + ic;
      for (int c = 0; c < nctrAB; ++c) {
        const int ca = c / nctrB;
        const int cb = c - ca * nctrB;
        const int a = a0_shell + ca * nA + ia;
        const int b = b0_shell + cb * nB + ib;
        const int64_t idx_abp =
            (static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b)) * static_cast<int64_t>(naux) +
            static_cast<int64_t>(p);
        const int64_t idx_bap =
            (static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a)) * static_cast<int64_t>(naux) +
            static_cast<int64_t>(p);
        X_out[idx_abp] = acc[c];
        if (a != b) X_out[idx_bap] = acc[c];
      }
    }
  }
}

}  // namespace

extern "C" cudaError_t cueri_eri_rys_generic_launch_stream(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    int ld,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || lc < 0 || ld < 0) return cudaErrorInvalidValue;
  if (la > kLMax || lb > kLMax || lc > kLMax || ld > kLMax) return cudaErrorInvalidValue;

  const int lsum = la + lb + lc + ld;
  const int nroots = (lsum >> 1) + 1;
  if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

  const int blocks = ntasks;
  switch (nroots) {
    case 1:
      KernelERI_RysGeneric<1><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 2:
      KernelERI_RysGeneric<2><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 3:
      KernelERI_RysGeneric<3><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 4:
      KernelERI_RysGeneric<4><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 5:
      KernelERI_RysGeneric<5><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 6:
      KernelERI_RysGeneric<6><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 7:
      KernelERI_RysGeneric<7><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 8:
      KernelERI_RysGeneric<8><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 9:
      KernelERI_RysGeneric<9><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 10:
      KernelERI_RysGeneric<10><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    case 11:
      KernelERI_RysGeneric<11><<<blocks, threads, 0, stream>>>(
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
          pair_eta,
          pair_Px,
          pair_Py,
          pair_Pz,
          pair_cK,
          la,
          lb,
          lc,
          ld,
          eri_out);
      break;
    default:
      return cudaErrorInvalidValue;
  }

  return cudaGetLastError();
}

extern "C" cudaError_t cueri_eri_rys_generic_warp_launch_stream(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    int ld,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || lc < 0 || ld < 0) return cudaErrorInvalidValue;
  if (la > kLMax || lb > kLMax || lc > kLMax || ld > kLMax) return cudaErrorInvalidValue;
  if (threads <= 0) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (threads > 256) return cudaErrorInvalidValue;

  const int nA = ((la + 1) * (la + 2)) >> 1;
  const int nB = ((lb + 1) * (lb + 2)) >> 1;
  const int nC = ((lc + 1) * (lc + 2)) >> 1;
  const int nD = ((ld + 1) * (ld + 2)) >> 1;
  const int nElem = nA * nB * nC * nD;
  if (nElem <= 0 || nElem > 128) return cudaErrorInvalidValue;

  // Keep the C API stable first: route through the reference-oriented generic launcher.
  // Dispatcher heuristics ensure this entrypoint is used only for small tiles.
  return cueri_eri_rys_generic_launch_stream(
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
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      la,
      lb,
      lc,
      ld,
      eri_out,
      stream,
      threads);
}

extern "C" cudaError_t cueri_eri_rys_df_ld0_warp_launch_stream(
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
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int la,
    int lb,
    int lc,
    double* eri_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || lc < 0) return cudaErrorInvalidValue;
  if (la > kLMax || lb > kLMax || lc > kLMax) return cudaErrorInvalidValue;
  if (threads <= 0) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (threads > 256) return cudaErrorInvalidValue;

  const int nA = ((la + 1) * (la + 2)) >> 1;
  const int nB = ((lb + 1) * (lb + 2)) >> 1;
  const int nC = ((lc + 1) * (lc + 2)) >> 1;
  const int nElem = nA * nB * nC;
  if (nElem <= 0) return cudaErrorInvalidValue;

  const int lsum = la + lb + lc;  // ld=0
  const int nroots = (lsum >> 1) + 1;
  if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  const bool use_stride5 = (nmax <= 4 && mmax <= 4);

  // Subwarp packing for small tiles:
  // - STRIDE=5 and nElem<=32: prefer subwarp8 (4 tasks/warp, 8 lanes/task) with multi-element lanes.
  // - otherwise, nElem<=64: use subwarp16 (2 tasks/warp, 16 lanes/task) with multi-element lanes.
  Ld0WarpMode mode = Ld0WarpMode::kWarp;
  if (nElem <= 32) mode = use_stride5 ? Ld0WarpMode::kSubwarp8 : Ld0WarpMode::kSubwarp16;
  if (nElem > 32 && nElem <= 64) mode = Ld0WarpMode::kSubwarp16;
  int tasks_per_warp = 1;
  if (mode == Ld0WarpMode::kSubwarp8) tasks_per_warp = 4;
  if (mode == Ld0WarpMode::kSubwarp16) tasks_per_warp = 2;

  int threads_eff = threads;
  if (mode == Ld0WarpMode::kSubwarp16 && !use_stride5 && threads_eff > 128) threads_eff = 128;  // large-stride scratch

  const int warps_per_block = threads_eff >> 5;
  const int tasks_per_block = warps_per_block * tasks_per_warp;
  const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;

  if (use_stride5) {
    switch (nroots) {
      case 1:
        launch_eri_rys_df_ld0<1, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 2:
        launch_eri_rys_df_ld0<2, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 3:
        launch_eri_rys_df_ld0<3, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 4:
        launch_eri_rys_df_ld0<4, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 5:
        launch_eri_rys_df_ld0<5, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 6:
        launch_eri_rys_df_ld0<6, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 7:
        launch_eri_rys_df_ld0<7, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 8:
        launch_eri_rys_df_ld0<8, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 9:
        launch_eri_rys_df_ld0<9, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 10:
        launch_eri_rys_df_ld0<10, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 11:
        launch_eri_rys_df_ld0<11, 5>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      default:
        return cudaErrorInvalidValue;
    }
  } else {
    switch (nroots) {
      case 1:
        launch_eri_rys_df_ld0<1, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 2:
        launch_eri_rys_df_ld0<2, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 3:
        launch_eri_rys_df_ld0<3, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 4:
        launch_eri_rys_df_ld0<4, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 5:
        launch_eri_rys_df_ld0<5, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 6:
        launch_eri_rys_df_ld0<6, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 7:
        launch_eri_rys_df_ld0<7, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 8:
        launch_eri_rys_df_ld0<8, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 9:
        launch_eri_rys_df_ld0<9, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 10:
        launch_eri_rys_df_ld0<10, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      case 11:
        launch_eri_rys_df_ld0<11, kStride>(
            mode,
            blocks,
            threads_eff,
            tasks_per_block,
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
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            la,
            lb,
            lc,
            eri_out,
            stream);
        break;
      default:
        return cudaErrorInvalidValue;
    }
  }

  return cudaGetLastError();
}

template <int CTR_MAX>
static cudaError_t cueri_df_int3c2e_rys_contracted_launch_stream_impl(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0) return cudaErrorInvalidValue;
  if (n_shell_ao < 0) return cudaErrorInvalidValue;
  if (nao < 0 || naux < 0) return cudaErrorInvalidValue;
  if (aux_p0_block < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || lc < 0) return cudaErrorInvalidValue;
  if (la > kLMax || lb > kLMax || lc > kLMax) return cudaErrorInvalidValue;
  if (threads <= 0) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (threads > 256) return cudaErrorInvalidValue;

  const int nA = ((la + 1) * (la + 2)) >> 1;
  const int nB = ((lb + 1) * (lb + 2)) >> 1;
  const int nC = ((lc + 1) * (lc + 2)) >> 1;
  const int nElem = nA * nB * nC;
  if (nElem <= 0) return cudaErrorInvalidValue;

  const int lsum = la + lb + lc;  // ld=0
  const int nroots = (lsum >> 1) + 1;
  if (nroots < 1 || nroots > 11) return cudaErrorInvalidValue;

  const int nmax = la + lb;
  const int mmax = lc;  // ld=0
  const bool use_stride5 = (nmax <= 4 && mmax <= 4);

  const bool use_warp = (nElem <= 64);

  // Subwarp packing for common contracted AO cases (CTR_MAX=2) in the STRIDE=5 bucket.
  //
  // Targets small DF tiles (nElem<=32) where the warp-per-task kernel suffers from
  // poor lane utilization and setup-heavy leader work.
  if constexpr (CTR_MAX == 2) {
    if (use_stride5 && use_warp && nElem <= 32) {
      const int threads_eff = threads;
      const int warps_per_block = threads_eff >> 5;
      if (warps_per_block <= 0 || warps_per_block > kMaxWarpsPerBlock) return cudaErrorInvalidValue;
      const int tasks_per_block = warps_per_block * 4;
      const int blocks = (ntasks + tasks_per_block - 1) / tasks_per_block;

#define CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(NR)                                                                         \
  launch_df_int3c2e_rys_contracted_ld0_subwarp8_stride5<NR, CTR_MAX>(                                                           \
      blocks,                                                                                                                  \
      threads_eff,                                                                                                             \
      tasks_per_block,                                                                                                         \
      task_spAB,                                                                                                               \
      task_spCD,                                                                                                               \
      ntasks,                                                                                                                  \
      sp_A,                                                                                                                    \
      sp_B,                                                                                                                    \
      sp_pair_start,                                                                                                           \
      sp_npair,                                                                                                                \
      shell_nprim,                                                                                                             \
      shell_cx,                                                                                                                \
      shell_cy,                                                                                                                \
      shell_cz,                                                                                                                \
      pair_eta,                                                                                                                \
      pair_Px,                                                                                                                 \
      pair_Py,                                                                                                                 \
      pair_Pz,                                                                                                                 \
      pair_cK,                                                                                                                 \
      ao_shell_ao_start,                                                                                                       \
      ao_shell_nctr,                                                                                                           \
      ao_shell_coef_start,                                                                                                     \
      ao_prim_coef,                                                                                                            \
      aux_shell_ao_start,                                                                                                      \
      n_shell_ao,                                                                                                              \
      nao,                                                                                                                     \
      naux,                                                                                                                    \
      aux_p0_block,                                                                                                            \
      la,                                                                                                                      \
      lb,                                                                                                                      \
      lc,                                                                                                                      \
      X_out,                                                                                                                   \
      stream)

      switch (nroots) {
        case 1:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(1);
          break;
        case 2:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(2);
          break;
        case 3:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(3);
          break;
        case 4:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(4);
          break;
        case 5:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(5);
          break;
        case 6:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(6);
          break;
        case 7:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(7);
          break;
        case 8:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(8);
          break;
        case 9:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(9);
          break;
        case 10:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(10);
          break;
        case 11:
          CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8(11);
          break;
        default:
#undef CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8
          return cudaErrorInvalidValue;
      }
#undef CUERI_LAUNCH_DF_INT3C2E_CONTRACTED_SUBWARP8
      return cudaGetLastError();
    }
  }

  int threads_eff = threads;
  int blocks = 0;
  if (use_warp) {
    const int warps_per_block = threads_eff >> 5;
    if (warps_per_block <= 0 || warps_per_block > kMaxWarpsPerBlock) return cudaErrorInvalidValue;
    blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  } else {
    // For block-per-task, avoid launching many idle threads.
    const int want = ((nElem + 31) >> 5) << 5;  // round up to warp multiple
    threads_eff = (threads_eff > want) ? want : threads_eff;
    if (threads_eff < 32) threads_eff = 32;
    blocks = ntasks;
  }

  if (use_stride5) {
    switch (nroots) {
      case 1:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<1, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(
              task_spAB,
              task_spCD,
              ntasks,
              sp_A,
              sp_B,
              sp_pair_start,
              sp_npair,
              shell_nprim,
              shell_cx,
              shell_cy,
              shell_cz,
              pair_eta,
              pair_Px,
              pair_Py,
              pair_Pz,
              pair_cK,
              ao_shell_ao_start,
              ao_shell_nctr,
              ao_shell_coef_start,
              ao_prim_coef,
              aux_shell_ao_start,
              n_shell_ao,
              nao,
              naux,
              aux_p0_block,
              la,
              lb,
              lc,
              X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<1, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(
              task_spAB,
              task_spCD,
              ntasks,
              sp_A,
              sp_B,
              sp_pair_start,
              sp_npair,
              shell_nprim,
              shell_cx,
              shell_cy,
              shell_cz,
              pair_eta,
              pair_Px,
              pair_Py,
              pair_Pz,
              pair_cK,
              ao_shell_ao_start,
              ao_shell_nctr,
              ao_shell_coef_start,
              ao_prim_coef,
              aux_shell_ao_start,
              n_shell_ao,
              nao,
              naux,
              aux_p0_block,
              la,
              lb,
              lc,
              X_out);
        }
        break;
      case 2:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<2, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<2, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 3:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<3, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<3, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 4:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<4, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<4, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 5:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<5, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<5, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 6:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<6, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<6, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 7:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<7, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<7, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 8:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<8, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<8, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 9:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<9, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy,
                                                                                           shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                           pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                           ao_shell_nctr, ao_shell_coef_start,
                                                                                           ao_prim_coef, aux_shell_ao_start,
                                                                                           n_shell_ao, nao, naux, aux_p0_block,
                                                                                           la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<9, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                          sp_B, sp_pair_start, sp_npair,
                                                                                          shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                          pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                          ao_shell_ao_start, ao_shell_nctr,
                                                                                          ao_shell_coef_start, ao_prim_coef,
                                                                                          aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                          aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 10:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<10, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                            sp_B, sp_pair_start, sp_npair,
                                                                                            shell_nprim, shell_cx, shell_cy,
                                                                                            shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                            pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                            ao_shell_nctr, ao_shell_coef_start,
                                                                                            ao_prim_coef, aux_shell_ao_start,
                                                                                            n_shell_ao, nao, naux, aux_p0_block,
                                                                                            la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<10, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                           pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                           ao_shell_ao_start, ao_shell_nctr,
                                                                                           ao_shell_coef_start, ao_prim_coef,
                                                                                           aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                           aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 11:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<11, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                            sp_B, sp_pair_start, sp_npair,
                                                                                            shell_nprim, shell_cx, shell_cy,
                                                                                            shell_cz, pair_eta, pair_Px, pair_Py,
                                                                                            pair_Pz, pair_cK, ao_shell_ao_start,
                                                                                            ao_shell_nctr, ao_shell_coef_start,
                                                                                            ao_prim_coef, aux_shell_ao_start,
                                                                                            n_shell_ao, nao, naux, aux_p0_block,
                                                                                            la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<11, 5, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                           sp_B, sp_pair_start, sp_npair,
                                                                                           shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                           pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                           ao_shell_ao_start, ao_shell_nctr,
                                                                                           ao_shell_coef_start, ao_prim_coef,
                                                                                           aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                           aux_p0_block, la, lb, lc, X_out);
        }
        break;
      default:
        return cudaErrorInvalidValue;
    }
  } else {
    switch (nroots) {
      case 1:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<1, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<1, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 2:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<2, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<2, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 3:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<3, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<3, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 4:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<4, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<4, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 5:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<5, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<5, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 6:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<6, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<6, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 7:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<7, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<7, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 8:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<8, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<8, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 9:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<9, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<9, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                               sp_B, sp_pair_start, sp_npair,
                                                                                               shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                               pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                               ao_shell_ao_start, ao_shell_nctr,
                                                                                               ao_shell_coef_start, ao_prim_coef,
                                                                                               aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                               aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 10:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<10, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                 sp_B, sp_pair_start, sp_npair,
                                                                                                 shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                 pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                 ao_shell_ao_start, ao_shell_nctr,
                                                                                                 ao_shell_coef_start, ao_prim_coef,
                                                                                                 aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                 aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<10, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        }
        break;
      case 11:
        if (use_warp) {
          KernelDFInt3c2e_RysContractedLd0Warp64<11, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                 sp_B, sp_pair_start, sp_npair,
                                                                                                 shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                 pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                 ao_shell_ao_start, ao_shell_nctr,
                                                                                                 ao_shell_coef_start, ao_prim_coef,
                                                                                                 aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                 aux_p0_block, la, lb, lc, X_out);
        } else {
          KernelDFInt3c2e_RysContractedLd0Block<11, kStride, CTR_MAX><<<blocks, threads_eff, 0, stream>>>(task_spAB, task_spCD, ntasks, sp_A,
                                                                                                sp_B, sp_pair_start, sp_npair,
                                                                                                shell_nprim, shell_cx, shell_cy, shell_cz,
                                                                                                pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
                                                                                                ao_shell_ao_start, ao_shell_nctr,
                                                                                                ao_shell_coef_start, ao_prim_coef,
                                                                                                aux_shell_ao_start, n_shell_ao, nao, naux,
                                                                                                aux_p0_block, la, lb, lc, X_out);
        }
        break;
      default:
        return cudaErrorInvalidValue;
    }
  }

  return cudaGetLastError();
}


extern "C" cudaError_t cueri_df_int3c2e_rys_contracted_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,
    cudaStream_t stream,
    int threads) {
  return cueri_df_int3c2e_rys_contracted_launch_stream_impl<kCtrMax>(
      task_spAB,
      task_spCD,
      ntasks,
      sp_A,
      sp_B,
      sp_pair_start,
      sp_npair,
      shell_nprim,
      shell_cx,
      shell_cy,
      shell_cz,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      ao_shell_ao_start,
      ao_shell_nctr,
      ao_shell_coef_start,
      ao_prim_coef,
      aux_shell_ao_start,
      n_shell_ao,
      nao,
      naux,
      aux_p0_block,
      la,
      lb,
      lc,
      X_out,
      stream,
      threads);
}

extern "C" cudaError_t cueri_df_int3c2e_rys_contracted_ctr2_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,
    cudaStream_t stream,
    int threads) {
  // CTR_MAX=2 specialization: significantly reduces register pressure for the
  // common case nctrA,nctrB<=2 (e.g., cc-pVXZ).
  return cueri_df_int3c2e_rys_contracted_launch_stream_impl<2>(
      task_spAB,
      task_spCD,
      ntasks,
      sp_A,
      sp_B,
      sp_pair_start,
      sp_npair,
      shell_nprim,
      shell_cx,
      shell_cy,
      shell_cz,
      pair_eta,
      pair_Px,
      pair_Py,
      pair_Pz,
      pair_cK,
      ao_shell_ao_start,
      ao_shell_nctr,
      ao_shell_coef_start,
      ao_prim_coef,
      aux_shell_ao_start,
      n_shell_ao,
      nao,
      naux,
      aux_p0_block,
      la,
      lb,
      lc,
      X_out,
      stream,
      threads);
}
