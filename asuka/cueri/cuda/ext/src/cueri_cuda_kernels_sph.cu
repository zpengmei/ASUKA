#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cart2sph_tables.cuh"
#include "cueri_cuda_kernels_api.h"

namespace {

__device__ __forceinline__ int ncart(int l) { return ((l + 1) * (l + 2)) >> 1; }
__device__ __forceinline__ int nsph(int l) { return (l << 1) + 1; }

__global__ void KernelSphCoeffSphToCart(
    const double* __restrict__ C_sph,
    double* __restrict__ C_cart,
    int norb,
    const int32_t* __restrict__ ao2shell_cart,
    const int32_t* __restrict__ ao2local_cart,
    const int32_t* __restrict__ shell_ao_start_sph,
    const int32_t* __restrict__ shell_l,
    int nao_cart) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
  const int64_t n = static_cast<int64_t>(nao_cart) * static_cast<int64_t>(norb);
  if (idx >= n) return;

  const int a = static_cast<int>(idx / static_cast<int64_t>(norb));
  const int p = static_cast<int>(idx - static_cast<int64_t>(a) * static_cast<int64_t>(norb));

  const int sh = static_cast<int>(ao2shell_cart[a]);
  const int ic = static_cast<int>(ao2local_cart[a]);
  const int l = static_cast<int>(shell_l[sh]);
  if (l < 0 || l > 5) {
    C_cart[idx] = 0.0;
    return;
  }

  const int ns = 2 * l + 1;
  const int sph0 = static_cast<int>(shell_ao_start_sph[sh]);

  double acc = 0.0;
#pragma unroll
  for (int is = 0; is < 11; ++is) {
    if (is >= ns) break;
    const double coef = cart2sph_coeff(l, ic, is);
    const int64_t row = static_cast<int64_t>(sph0 + is) * static_cast<int64_t>(norb);
    acc += coef * C_sph[row + p];
  }
  C_cart[idx] = acc;
}

template <typename Tout>
__global__ void KernelDFBarXSphToCartSym(
    const int32_t* __restrict__ spAB_arr,  // [n_spAB] AO shell-pair indices for this (la,lb)
    int n_spAB,
    const int32_t* __restrict__ sp_A,                // [nsp]
    const int32_t* __restrict__ sp_B,                // [nsp]
    const int32_t* __restrict__ shell_ao_start_cart,  // [nShellTotal]
    const int32_t* __restrict__ shell_ao_start_sph,   // [nShellAO]
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* __restrict__ in_sph_mnQ,  // (nao_sph, nao_sph, naux), C-order
    Tout* __restrict__ out_cart_mnQ) {      // (nao_cart, nao_cart, naux), C-order
  // Block geometry: blockDim.x must be 32 (one warp-lane per Q element), blockDim.y == nwarps.
  const int lane = static_cast<int>(threadIdx.x);
  const int wid = static_cast<int>(threadIdx.y);
  if (lane >= 32) return;

  const int spAB_i = static_cast<int>(blockIdx.x);
  if (spAB_i >= n_spAB) return;
  const int q0 = static_cast<int>(blockIdx.y) * 32;
  const int q = q0 + lane;

  const int spAB = static_cast<int>(spAB_arr[spAB_i]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  const int a0_cart = static_cast<int>(shell_ao_start_cart[shellA]);
  const int b0_cart = static_cast<int>(shell_ao_start_cart[shellB]);
  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);

  // Shared cache of spherical tile for this shell-pair and Q block:
  // sh[(isA*nB_sph + isB)*32 + lane] = in_sph[(a0_sph+isA, b0_sph+isB, q0+lane)]
  extern __shared__ double sh_sph[];
  const int nS = nA_sph * nB_sph;
  for (int idx = wid; idx < nS; idx += static_cast<int>(blockDim.y)) {
    const int isA = idx / nB_sph;
    const int isB = idx - isA * nB_sph;
    double v = 0.0;
    if (q < naux) {
      const int i = a0_sph + isA;
      const int j = b0_sph + isB;
      v = in_sph_mnQ[(static_cast<int64_t>(i) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) *
                         static_cast<int64_t>(naux) +
                     static_cast<int64_t>(q)];
    }
    sh_sph[idx * 32 + lane] = v;
  }

  __syncthreads();

  const bool diag_shell = (shellA == shellB);
  const int nElem = nA_cart * nB_cart;
  for (int elem = wid; elem < nElem; elem += static_cast<int>(blockDim.y)) {
    const int ia = elem / nB_cart;
    const int ib = elem - ia * nB_cart;
    if (diag_shell && ia < ib) continue;  // avoid duplicate writes on diagonal shell blocks

    double acc = 0.0;
#pragma unroll
    for (int isA = 0; isA < 11; ++isA) {
      if (isA >= nA_sph) break;
      const double ta = cart2sph_coeff(la, ia, isA);
      if (ta == 0.0) continue;
#pragma unroll
      for (int isB = 0; isB < 11; ++isB) {
        if (isB >= nB_sph) break;
        const double tb = cart2sph_coeff(lb, ib, isB);
        if (tb == 0.0) continue;
        acc += (ta * tb) * sh_sph[(isA * nB_sph + isB) * 32 + lane];
      }
    }

    if (q >= naux) continue;
    const int m = a0_cart + ia;
    const int n = b0_cart + ib;
    const int64_t idx_mn =
        (static_cast<int64_t>(m) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(n)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const int64_t idx_nm =
        (static_cast<int64_t>(n) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(m)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const Tout outv = static_cast<Tout>(acc);
    out_cart_mnQ[idx_mn] = outv;
    if (idx_nm != idx_mn) out_cart_mnQ[idx_nm] = outv;
  }
}

template <typename Tout>
__global__ void KernelDFBarXSphQmnToCartSym(
    const int32_t* __restrict__ spAB_arr,  // [n_spAB] AO shell-pair indices for this (la,lb)
    int n_spAB,
    const int32_t* __restrict__ sp_A,                // [nsp]
    const int32_t* __restrict__ sp_B,                // [nsp]
    const int32_t* __restrict__ shell_ao_start_cart,  // [nShellTotal]
    const int32_t* __restrict__ shell_ao_start_sph,   // [nShellAO]
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* __restrict__ in_sph_Qmn,  // (naux, nao_sph, nao_sph), C-order
    Tout* __restrict__ out_cart_mnQ) {      // (nao_cart, nao_cart, naux), C-order
  // Block geometry: blockDim.x must be 32 (one warp-lane per Q element), blockDim.y == nwarps.
  const int lane = static_cast<int>(threadIdx.x);
  const int wid = static_cast<int>(threadIdx.y);
  if (lane >= 32) return;

  const int spAB_i = static_cast<int>(blockIdx.x);
  if (spAB_i >= n_spAB) return;
  const int q0 = static_cast<int>(blockIdx.y) * 32;
  const int q = q0 + lane;

  const int spAB = static_cast<int>(spAB_arr[spAB_i]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  const int a0_cart = static_cast<int>(shell_ao_start_cart[shellA]);
  const int b0_cart = static_cast<int>(shell_ao_start_cart[shellB]);
  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);

  // Shared cache of symmetrized spherical tile for this shell-pair and Q block.
  extern __shared__ double sh_sph[];
  const int nS = nA_sph * nB_sph;
  for (int idx = wid; idx < nS; idx += static_cast<int>(blockDim.y)) {
    const int isA = idx / nB_sph;
    const int isB = idx - isA * nB_sph;
    double v = 0.0;
    if (q < naux) {
      const int i = a0_sph + isA;
      const int j = b0_sph + isB;
      const int64_t idx_ij =
          (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) * static_cast<int64_t>(nao_sph) +
          static_cast<int64_t>(j);
      const int64_t idx_ji =
          (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) * static_cast<int64_t>(nao_sph) +
          static_cast<int64_t>(i);
      v = 0.5 * (in_sph_Qmn[idx_ij] + in_sph_Qmn[idx_ji]);
    }
    sh_sph[idx * 32 + lane] = v;
  }

  __syncthreads();

  const bool diag_shell = (shellA == shellB);
  const int nElem = nA_cart * nB_cart;
  for (int elem = wid; elem < nElem; elem += static_cast<int>(blockDim.y)) {
    const int ia = elem / nB_cart;
    const int ib = elem - ia * nB_cart;
    if (diag_shell && ia < ib) continue;  // avoid duplicate writes on diagonal shell blocks

    double acc = 0.0;
#pragma unroll
    for (int isA = 0; isA < 11; ++isA) {
      if (isA >= nA_sph) break;
      const double ta = cart2sph_coeff(la, ia, isA);
      if (ta == 0.0) continue;
#pragma unroll
      for (int isB = 0; isB < 11; ++isB) {
        if (isB >= nB_sph) break;
        const double tb = cart2sph_coeff(lb, ib, isB);
        if (tb == 0.0) continue;
        acc += (ta * tb) * sh_sph[(isA * nB_sph + isB) * 32 + lane];
      }
    }

    if (q >= naux) continue;
    const int m = a0_cart + ia;
    const int n = b0_cart + ib;
    const int64_t idx_mn =
        (static_cast<int64_t>(m) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(n)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const int64_t idx_nm =
        (static_cast<int64_t>(n) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(m)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const Tout outv = static_cast<Tout>(acc);
    out_cart_mnQ[idx_mn] = outv;
    if (idx_nm != idx_mn) out_cart_mnQ[idx_nm] = outv;
  }
}

template <typename Tout>
__global__ void KernelDFBCartToSphSym(
    const int32_t* __restrict__ spAB_arr,  // [n_spAB] shell-pair indices for this (la,lb)
    int n_spAB,
    const int32_t* __restrict__ sp_A,                // [nsp]
    const int32_t* __restrict__ sp_B,                // [nsp]
    const int32_t* __restrict__ shell_ao_start_cart,  // [nShellAO]
    const int32_t* __restrict__ shell_ao_start_sph,   // [nShellAO]
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* __restrict__ in_cart_mnQ,  // (nao_cart, nao_cart, naux), C-order
    Tout* __restrict__ out_sph_mnQ) {        // (nao_sph, nao_sph, naux), C-order
  // Threading: x-dim is a 32-wide Q lane, y-dim is warp id.
  const int lane = static_cast<int>(threadIdx.x);
  const int wid = static_cast<int>(threadIdx.y);
  const int nwarps = static_cast<int>(blockDim.y);
  if (lane >= 32) return;

  const int spAB_i = static_cast<int>(blockIdx.x);
  if (spAB_i >= n_spAB) return;
  const int q0 = static_cast<int>(blockIdx.y) * 32;
  const int q = q0 + lane;

  const int spAB = static_cast<int>(spAB_arr[spAB_i]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  const int a0_cart = static_cast<int>(shell_ao_start_cart[shellA]);
  const int b0_cart = static_cast<int>(shell_ao_start_cart[shellB]);
  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);
  const int nElem = nA_sph * nB_sph;

  // Each warp computes 2 spherical output elements (elem0/elem1) for this shell-pair and Q block.
  const int base = (static_cast<int>(blockIdx.z) * nwarps + wid) * 2;
  int elem0 = base;
  int elem1 = base + 1;

  int isA0 = 0, isB0 = 0;
  int isA1 = 0, isB1 = 0;
  bool active0 = (elem0 < nElem);
  bool active1 = (elem1 < nElem);
  if (active0) {
    isA0 = elem0 / nB_sph;
    isB0 = elem0 - isA0 * nB_sph;
  }
  if (active1) {
    isA1 = elem1 / nB_sph;
    isB1 = elem1 - isA1 * nB_sph;
  }
  const bool diag_shell = (shellA == shellB);
  if (diag_shell) {
    if (active0 && isA0 < isB0) active0 = false;
    if (active1 && isA1 < isB1) active1 = false;
  }

  double acc0 = 0.0;
  double acc1 = 0.0;

  // Shared cache for one Cartesian row slice: sh_cart[icB,lane] = in_cart[(a0_cart+icA),(b0_cart+icB),q]
  extern __shared__ double sh_cart[];

#pragma unroll
  for (int icA = 0; icA < 21; ++icA) {
    if (icA >= nA_cart) break;

    // Load the Cartesian slice for this icA into shared memory.
    for (int icB = wid; icB < nB_cart; icB += nwarps) {
      double v = 0.0;
      if (q < naux) {
        const int m = a0_cart + icA;
        const int n = b0_cart + icB;
        v = in_cart_mnQ[(static_cast<int64_t>(m) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(n)) *
                            static_cast<int64_t>(naux) +
                        static_cast<int64_t>(q)];
      }
      sh_cart[icB * 32 + lane] = v;
    }

    __syncthreads();

    if (active0) {
      const double ta = cart2sph_coeff(la, icA, isA0);
      if (ta != 0.0) {
        double tmp = 0.0;
#pragma unroll
        for (int icB = 0; icB < 21; ++icB) {
          if (icB >= nB_cart) break;
          const double tb = cart2sph_coeff(lb, icB, isB0);
          if (tb == 0.0) continue;
          tmp += tb * sh_cart[icB * 32 + lane];
        }
        acc0 += ta * tmp;
      }
    }
    if (active1) {
      const double ta = cart2sph_coeff(la, icA, isA1);
      if (ta != 0.0) {
        double tmp = 0.0;
#pragma unroll
        for (int icB = 0; icB < 21; ++icB) {
          if (icB >= nB_cart) break;
          const double tb = cart2sph_coeff(lb, icB, isB1);
          if (tb == 0.0) continue;
          tmp += tb * sh_cart[icB * 32 + lane];
        }
        acc1 += ta * tmp;
      }
    }

    __syncthreads();
  }

  if (q >= naux) return;
  if (active0) {
    const int i = a0_sph + isA0;
    const int j = b0_sph + isB0;
    const int64_t idx_ij =
        (static_cast<int64_t>(i) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const int64_t idx_ji =
        (static_cast<int64_t>(j) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const Tout outv = static_cast<Tout>(acc0);
    out_sph_mnQ[idx_ij] = outv;
    if (idx_ji != idx_ij) out_sph_mnQ[idx_ji] = outv;
  }
  if (active1) {
    const int i = a0_sph + isA1;
    const int j = b0_sph + isB1;
    const int64_t idx_ij =
        (static_cast<int64_t>(i) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const int64_t idx_ji =
        (static_cast<int64_t>(j) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) * static_cast<int64_t>(naux) +
        static_cast<int64_t>(q);
    const Tout outv = static_cast<Tout>(acc1);
    out_sph_mnQ[idx_ij] = outv;
    if (idx_ji != idx_ij) out_sph_mnQ[idx_ji] = outv;
  }
}

template <typename Tout>
__global__ void KernelDFBCartToSphQmnSym(
    const int32_t* __restrict__ spAB_arr,  // [n_spAB] shell-pair indices for this (la,lb)
    int n_spAB,
    const int32_t* __restrict__ sp_A,                // [nsp]
    const int32_t* __restrict__ sp_B,                // [nsp]
    const int32_t* __restrict__ shell_ao_start_cart,  // [nShellAO]
    const int32_t* __restrict__ shell_ao_start_sph,   // [nShellAO]
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* __restrict__ in_cart_mnQ,  // (nao_cart, nao_cart, naux), C-order
    Tout* __restrict__ out_sph_Qmn) {        // (naux, nao_sph, nao_sph), C-order
  // Threading: x-dim is a 32-wide Q lane, y-dim is warp id.
  const int lane = static_cast<int>(threadIdx.x);
  const int wid = static_cast<int>(threadIdx.y);
  const int nwarps = static_cast<int>(blockDim.y);
  if (lane >= 32) return;

  const int spAB_i = static_cast<int>(blockIdx.x);
  if (spAB_i >= n_spAB) return;
  const int q0 = static_cast<int>(blockIdx.y) * 32;
  const int q = q0 + lane;

  const int spAB = static_cast<int>(spAB_arr[spAB_i]);
  const int shellA = static_cast<int>(sp_A[spAB]);
  const int shellB = static_cast<int>(sp_B[spAB]);

  const int a0_cart = static_cast<int>(shell_ao_start_cart[shellA]);
  const int b0_cart = static_cast<int>(shell_ao_start_cart[shellB]);
  const int a0_sph = static_cast<int>(shell_ao_start_sph[shellA]);
  const int b0_sph = static_cast<int>(shell_ao_start_sph[shellB]);

  const int nA_cart = ncart(la);
  const int nB_cart = ncart(lb);
  const int nA_sph = nsph(la);
  const int nB_sph = nsph(lb);
  const int nElem = nA_sph * nB_sph;

  // Each warp computes 2 spherical output elements (elem0/elem1) for this shell-pair and Q block.
  const int base = (static_cast<int>(blockIdx.z) * nwarps + wid) * 2;
  int elem0 = base;
  int elem1 = base + 1;

  int isA0 = 0, isB0 = 0;
  int isA1 = 0, isB1 = 0;
  bool active0 = (elem0 < nElem);
  bool active1 = (elem1 < nElem);
  if (active0) {
    isA0 = elem0 / nB_sph;
    isB0 = elem0 - isA0 * nB_sph;
  }
  if (active1) {
    isA1 = elem1 / nB_sph;
    isB1 = elem1 - isA1 * nB_sph;
  }
  const bool diag_shell = (shellA == shellB);
  if (diag_shell) {
    if (active0 && isA0 < isB0) active0 = false;
    if (active1 && isA1 < isB1) active1 = false;
  }

  double acc0 = 0.0;
  double acc1 = 0.0;

  // Shared cache for one Cartesian row slice: sh_cart[icB,lane] = in_cart[(a0_cart+icA),(b0_cart+icB),q]
  extern __shared__ double sh_cart[];

#pragma unroll
  for (int icA = 0; icA < 21; ++icA) {
    if (icA >= nA_cart) break;

    // Load the Cartesian slice for this icA into shared memory.
    for (int icB = wid; icB < nB_cart; icB += nwarps) {
      double v = 0.0;
      if (q < naux) {
        const int m = a0_cart + icA;
        const int n = b0_cart + icB;
        v = in_cart_mnQ[(static_cast<int64_t>(m) * static_cast<int64_t>(nao_cart) + static_cast<int64_t>(n)) *
                            static_cast<int64_t>(naux) +
                        static_cast<int64_t>(q)];
      }
      sh_cart[icB * 32 + lane] = v;
    }

    __syncthreads();

    if (active0) {
      const double ta = cart2sph_coeff(la, icA, isA0);
      if (ta != 0.0) {
        double tmp = 0.0;
#pragma unroll
        for (int icB = 0; icB < 21; ++icB) {
          if (icB >= nB_cart) break;
          const double tb = cart2sph_coeff(lb, icB, isB0);
          if (tb == 0.0) continue;
          tmp += tb * sh_cart[icB * 32 + lane];
        }
        acc0 += ta * tmp;
      }
    }
    if (active1) {
      const double ta = cart2sph_coeff(la, icA, isA1);
      if (ta != 0.0) {
        double tmp = 0.0;
#pragma unroll
        for (int icB = 0; icB < 21; ++icB) {
          if (icB >= nB_cart) break;
          const double tb = cart2sph_coeff(lb, icB, isB1);
          if (tb == 0.0) continue;
          tmp += tb * sh_cart[icB * 32 + lane];
        }
        acc1 += ta * tmp;
      }
    }

    __syncthreads();
  }

  if (q >= naux) return;
  if (active0) {
    const int i = a0_sph + isA0;
    const int j = b0_sph + isB0;
    const int64_t idx_ij =
        (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) * static_cast<int64_t>(nao_sph) +
        static_cast<int64_t>(j);
    const int64_t idx_ji =
        (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) * static_cast<int64_t>(nao_sph) +
        static_cast<int64_t>(i);
    const Tout outv = static_cast<Tout>(acc0);
    out_sph_Qmn[idx_ij] = outv;
    if (idx_ji != idx_ij) out_sph_Qmn[idx_ji] = outv;
  }
  if (active1) {
    const int i = a0_sph + isA1;
    const int j = b0_sph + isB1;
    const int64_t idx_ij =
        (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(i)) * static_cast<int64_t>(nao_sph) +
        static_cast<int64_t>(j);
    const int64_t idx_ji =
        (static_cast<int64_t>(q) * static_cast<int64_t>(nao_sph) + static_cast<int64_t>(j)) * static_cast<int64_t>(nao_sph) +
        static_cast<int64_t>(i);
    const Tout outv = static_cast<Tout>(acc1);
    out_sph_Qmn[idx_ij] = outv;
    if (idx_ji != idx_ij) out_sph_Qmn[idx_ji] = outv;
  }
}

}  // namespace

extern "C" cudaError_t cueri_sph_coeff_sph_to_cart_launch_stream(
    const double* C_sph,
    double* C_cart,
    int nao_cart,
    int norb,
    const int32_t* ao2shell_cart,
    const int32_t* ao2local_cart,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_l,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || norb < 0) return cudaErrorInvalidValue;
  const int64_t n = static_cast<int64_t>(nao_cart) * static_cast<int64_t>(norb);
  if (n == 0) return cudaSuccess;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads));
  KernelSphCoeffSphToCart<<<blocks, threads, 0, stream>>>(
      C_sph,
      C_cart,
      norb,
      ao2shell_cart,
      ao2local_cart,
      shell_ao_start_sph,
      shell_l,
      nao_cart);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_bar_x_sph_to_cart_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_mnQ,
    double* out_cart_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const dim3 block(32, static_cast<unsigned int>(threads >> 5), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), 1);
  const int sh_elems = ((la << 1) + 1) * ((lb << 1) + 1) * 32;
  const size_t sh_bytes = static_cast<size_t>(sh_elems) * sizeof(double);
  KernelDFBarXSphToCartSym<double><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_sph_mnQ,
      out_cart_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_bar_x_sph_to_cart_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_mnQ,
    float* out_cart_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const dim3 block(32, static_cast<unsigned int>(threads >> 5), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), 1);
  const int sh_elems = ((la << 1) + 1) * ((lb << 1) + 1) * 32;
  const size_t sh_bytes = static_cast<size_t>(sh_elems) * sizeof(double);
  KernelDFBarXSphToCartSym<float><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_sph_mnQ,
      out_cart_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_bar_x_sph_qmn_to_cart_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_Qmn,
    double* out_cart_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const dim3 block(32, static_cast<unsigned int>(threads >> 5), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), 1);
  const int sh_elems = ((la << 1) + 1) * ((lb << 1) + 1) * 32;
  const size_t sh_bytes = static_cast<size_t>(sh_elems) * sizeof(double);
  KernelDFBarXSphQmnToCartSym<double><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_sph_Qmn,
      out_cart_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_bar_x_sph_qmn_to_cart_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_Qmn,
    float* out_cart_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const dim3 block(32, static_cast<unsigned int>(threads >> 5), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), 1);
  const int sh_elems = ((la << 1) + 1) * ((lb << 1) + 1) * 32;
  const size_t sh_bytes = static_cast<size_t>(sh_elems) * sizeof(double);
  KernelDFBarXSphQmnToCartSym<float><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_sph_Qmn,
      out_cart_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_B_cart_to_sph_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    double* out_sph_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const int nwarps = threads >> 5;
  const int nElem = ((la << 1) + 1) * ((lb << 1) + 1);
  const int nElemBlocks = (nElem + (nwarps * 2) - 1) / (nwarps * 2);
  const dim3 block(32, static_cast<unsigned int>(nwarps), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), static_cast<unsigned int>(nElemBlocks));
  const int nB_cart = ((lb + 1) * (lb + 2)) >> 1;
  const size_t sh_bytes = static_cast<size_t>(nB_cart) * static_cast<size_t>(32) * sizeof(double);
  KernelDFBCartToSphSym<double><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_cart_mnQ,
      out_sph_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_B_cart_to_sph_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    float* out_sph_mnQ,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const int nwarps = threads >> 5;
  const int nElem = ((la << 1) + 1) * ((lb << 1) + 1);
  const int nElemBlocks = (nElem + (nwarps * 2) - 1) / (nwarps * 2);
  const dim3 block(32, static_cast<unsigned int>(nwarps), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), static_cast<unsigned int>(nElemBlocks));
  const int nB_cart = ((lb + 1) * (lb + 2)) >> 1;
  const size_t sh_bytes = static_cast<size_t>(nB_cart) * static_cast<size_t>(32) * sizeof(double);
  KernelDFBCartToSphSym<float><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_cart_mnQ,
      out_sph_mnQ);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_B_cart_to_sph_qmn_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    double* out_sph_Qmn,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const int nwarps = threads >> 5;
  const int nElem = ((la << 1) + 1) * ((lb << 1) + 1);
  const int nElemBlocks = (nElem + (nwarps * 2) - 1) / (nwarps * 2);
  const dim3 block(32, static_cast<unsigned int>(nwarps), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), static_cast<unsigned int>(nElemBlocks));
  const int nB_cart = ((lb + 1) * (lb + 2)) >> 1;
  const size_t sh_bytes = static_cast<size_t>(nB_cart) * static_cast<size_t>(32) * sizeof(double);
  KernelDFBCartToSphQmnSym<double><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_cart_mnQ,
      out_sph_Qmn);
  return cudaGetLastError();
}

extern "C" cudaError_t cueri_df_B_cart_to_sph_qmn_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    float* out_sph_Qmn,
    cudaStream_t stream,
    int threads) {
  if (nao_cart < 0 || nao_sph < 0 || naux < 0) return cudaErrorInvalidValue;
  if (la < 0 || lb < 0 || la > 5 || lb > 5) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 256) return cudaErrorInvalidValue;
  if ((threads & 31) != 0) return cudaErrorInvalidValue;
  if (n_spAB <= 0 || naux == 0) return cudaSuccess;

  const int nQBlocks = (naux + 31) >> 5;
  const int nwarps = threads >> 5;
  const int nElem = ((la << 1) + 1) * ((lb << 1) + 1);
  const int nElemBlocks = (nElem + (nwarps * 2) - 1) / (nwarps * 2);
  const dim3 block(32, static_cast<unsigned int>(nwarps), 1);
  const dim3 grid(static_cast<unsigned int>(n_spAB), static_cast<unsigned int>(nQBlocks), static_cast<unsigned int>(nElemBlocks));
  const int nB_cart = ((lb + 1) * (lb + 2)) >> 1;
  const size_t sh_bytes = static_cast<size_t>(nB_cart) * static_cast<size_t>(32) * sizeof(double);
  KernelDFBCartToSphQmnSym<float><<<grid, block, sh_bytes, stream>>>(
      spAB_arr,
      n_spAB,
      sp_A,
      sp_B,
      shell_ao_start_cart,
      shell_ao_start_sph,
      nao_cart,
      nao_sph,
      naux,
      la,
      lb,
      in_cart_mnQ,
      out_sph_Qmn);
  return cudaGetLastError();
}
