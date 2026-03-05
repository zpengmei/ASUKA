#include "orbitals_cuda_kernels_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int ORB_TILE = 64;
constexpr int PT_TILE = 4;

constexpr int LMAX = 10;
constexpr int NBF_MAX = (LMAX + 1) * (LMAX + 2) / 2;  // 66

constexpr int MAX_NCAS = 64;

__device__ __forceinline__ int ncart(int l) { return (l + 1) * (l + 2) / 2; }

__global__ void build_atom_centered_points_weights_kernel(
    const double* center_xyz,
    const double* radial_r,
    const double* radial_wr,
    int32_t nrad,
    const double* angular_dirs,
    const double* angular_w,
    int32_t nang,
    double* pts_local,
    double* w_base) {
  const int32_t q = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  const int32_t nloc = nrad * nang;
  if (q >= nloc) return;

  const int32_t ir = q / nang;
  const int32_t ia = q - ir * nang;

  const double rr = radial_r[ir];
  const double wr = radial_wr[ir];
  const double* dir = angular_dirs + int64_t(ia) * 3;

  pts_local[int64_t(q) * 3 + 0] = center_xyz[0] + rr * dir[0];
  pts_local[int64_t(q) * 3 + 1] = center_xyz[1] + rr * dir[1];
  pts_local[int64_t(q) * 3 + 2] = center_xyz[2] + rr * dir[2];
  w_base[q] = wr * angular_w[ia];
}

__global__ void becke_partition_atom_block_kernel(
    const double* pts_local,
    const double* w_base,
    int32_t nloc,
    const double* atom_coords,
    const double* RAB,
    int32_t natm,
    int32_t atom_index,
    int32_t becke_n,
    double* rA,
    double* w_raw,
    double* w_atom) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= nloc) return;

  const double x = pts_local[int64_t(p) * 3 + 0];
  const double y = pts_local[int64_t(p) * 3 + 1];
  const double z = pts_local[int64_t(p) * 3 + 2];

  double* rA_row = rA + int64_t(p) * int64_t(natm);
  double* w_row = w_raw + int64_t(p) * int64_t(natm);

  // Distances from local point p to each atom center.
  for (int32_t A = 0; A < natm; ++A) {
    const double ax = atom_coords[int64_t(A) * 3 + 0];
    const double ay = atom_coords[int64_t(A) * 3 + 1];
    const double az = atom_coords[int64_t(A) * 3 + 2];
    const double dx = x - ax;
    const double dy = y - ay;
    const double dz = z - az;
    rA_row[A] = sqrt(dx * dx + dy * dy + dz * dz);
    w_row[A] = 1.0;
  }

  // Pair updates in exact CPU order: A outer, B inner.
  for (int32_t A = 0; A < natm; ++A) {
    for (int32_t B = A + 1; B < natm; ++B) {
      const double Rab = RAB[int64_t(A) * int64_t(natm) + B];
      if (Rab <= 1e-14) continue;

      double mu = (rA_row[A] - rA_row[B]) / Rab;
      mu = mu < -1.0 ? -1.0 : (mu > 1.0 ? 1.0 : mu);

      double pp = mu;
      for (int32_t it = 0; it < becke_n; ++it) {
        pp = 0.5 * (3.0 * pp - pp * pp * pp);
      }

      const double sA = 0.5 * (1.0 - pp);
      const double sB = 1.0 - sA;
      w_row[A] *= sA;
      w_row[B] *= sB;
    }
  }

  double wsum = 0.0;
  for (int32_t A = 0; A < natm; ++A) {
    wsum += w_row[A];
  }
  if (wsum == 0.0) wsum = 1.0;

  w_atom[p] = w_base[p] * (w_row[atom_index] / wsum);
}

template <bool WANT_GRAD, bool WANT_LAPL>
__device__ __forceinline__ void eval_shell_cart_components(
    int l,
    double dx,
    double dy,
    double dz,
    double rad0,
    double rad1,
    double rad2,
    double* ao,
    double* gx,
    double* gy,
    double* gz,
    double* lapl) {
  double px[LMAX + 3];
  double py[LMAX + 3];
  double pz[LMAX + 3];

  const int max_pow = l + 2;
  px[0] = 1.0;
  py[0] = 1.0;
  pz[0] = 1.0;
  for (int k = 0; k < max_pow; ++k) {
    px[k + 1] = px[k] * dx;
    py[k + 1] = py[k] * dy;
    pz[k + 1] = pz[k] * dz;
  }

  int ic = 0;
  for (int lx = l; lx >= 0; --lx) {
    for (int ly = l - lx; ly >= 0; --ly) {
      const int lz = l - lx - ly;

      const double yz = py[ly] * pz[lz];
      const double xz = px[lx] * pz[lz];
      const double xy = px[lx] * py[ly];

      ao[ic] = rad0 * px[lx] * yz;

      if constexpr (WANT_GRAD) {
        double ddx = -2.0 * rad1 * px[lx + 1] * yz;
        if (lx > 0) ddx += rad0 * double(lx) * px[lx - 1] * yz;
        gx[ic] = ddx;

        double ddy = -2.0 * rad1 * py[ly + 1] * xz;
        if (ly > 0) ddy += rad0 * double(ly) * py[ly - 1] * xz;
        gy[ic] = ddy;

        double ddz = -2.0 * rad1 * pz[lz + 1] * xy;
        if (lz > 0) ddz += rad0 * double(lz) * pz[lz - 1] * xy;
        gz[ic] = ddz;
      }

      if constexpr (WANT_LAPL) {
        double dxx = 4.0 * rad2 * px[lx + 2] * yz;
        dxx += (-2.0 * rad1) * (1.0 + 2.0 * double(lx)) * px[lx] * yz;
        if (lx >= 2) dxx += rad0 * double(lx * (lx - 1)) * px[lx - 2] * yz;

        double dyy = 4.0 * rad2 * py[ly + 2] * xz;
        dyy += (-2.0 * rad1) * (1.0 + 2.0 * double(ly)) * py[ly] * xz;
        if (ly >= 2) dyy += rad0 * double(ly * (ly - 1)) * py[ly - 2] * xz;

        double dzz = 4.0 * rad2 * pz[lz + 2] * xy;
        dzz += (-2.0 * rad1) * (1.0 + 2.0 * double(lz)) * pz[lz] * xy;
        if (lz >= 2) dzz += rad0 * double(lz * (lz - 1)) * pz[lz - 2] * xy;

        lapl[ic] = dxx + dyy + dzz;
      }

      ++ic;
    }
  }
}

__global__ void aos_cart_value_kernel(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    double* ao_out) {
  const int32_t sh = int32_t(blockIdx.x);
  const int pt_lane = int(threadIdx.y);
  const int p = int(blockIdx.y) * PT_TILE + pt_lane;
  const int ic = int(threadIdx.x);

  if (sh >= nshell) return;
  if (p >= npt) return;

  const int l = int(shell_l[sh]);
  const int nbf = ncart(l);
  const int ao0 = int(shell_ao_start[sh]);

  if (l > LMAX || nbf > NBF_MAX) return;
  if (ao0 < 0 || ao0 + nbf > nao) return;

  __shared__ double sh_ao[PT_TILE][NBF_MAX];

  if (ic == 0) {
    const double cx = shell_cxyz[int(sh) * 3 + 0];
    const double cy = shell_cxyz[int(sh) * 3 + 1];
    const double cz = shell_cxyz[int(sh) * 3 + 2];

    const double x = points[int(p) * 3 + 0];
    const double y = points[int(p) * 3 + 1];
    const double z = points[int(p) * 3 + 2];

    const double dx = x - cx;
    const double dy = y - cy;
    const double dz = z - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;

    const int32_t p0 = shell_prim_start[sh];
    const int32_t npg = shell_nprim[sh];

    double rad0 = 0.0;
    for (int32_t ip = 0; ip < npg; ++ip) {
      const double aa = prim_exp[int(p0 + ip)];
      const double cc = prim_coef[int(p0 + ip)];
      const double e = exp(-aa * r2);
      rad0 += cc * e;
    }

    eval_shell_cart_components<false, false>(
        l,
        dx,
        dy,
        dz,
        rad0,
        0.0,
        0.0,
        sh_ao[pt_lane],
        nullptr,
        nullptr,
        nullptr,
        nullptr);
  }

  __syncthreads();

  if (ic < nbf) {
    ao_out[int64_t(p) * int64_t(nao) + int64_t(ao0 + ic)] = sh_ao[pt_lane][ic];
  }
}

__global__ void aos_cart_value_grad_kernel(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    double* ao_out,
    double* ao_grad_out) {
  const int32_t sh = int32_t(blockIdx.x);
  const int pt_lane = int(threadIdx.y);
  const int p = int(blockIdx.y) * PT_TILE + pt_lane;
  const int ic = int(threadIdx.x);

  if (sh >= nshell) return;
  if (p >= npt) return;

  const int l = int(shell_l[sh]);
  const int nbf = ncart(l);
  const int ao0 = int(shell_ao_start[sh]);

  if (l > LMAX || nbf > NBF_MAX) return;
  if (ao0 < 0 || ao0 + nbf > nao) return;

  __shared__ double sh_ao[PT_TILE][NBF_MAX];
  __shared__ double sh_gx[PT_TILE][NBF_MAX];
  __shared__ double sh_gy[PT_TILE][NBF_MAX];
  __shared__ double sh_gz[PT_TILE][NBF_MAX];

  if (ic == 0) {
    const double cx = shell_cxyz[int(sh) * 3 + 0];
    const double cy = shell_cxyz[int(sh) * 3 + 1];
    const double cz = shell_cxyz[int(sh) * 3 + 2];

    const double x = points[int(p) * 3 + 0];
    const double y = points[int(p) * 3 + 1];
    const double z = points[int(p) * 3 + 2];

    const double dx = x - cx;
    const double dy = y - cy;
    const double dz = z - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;

    const int32_t p0 = shell_prim_start[sh];
    const int32_t npg = shell_nprim[sh];

    double rad0 = 0.0;
    double rad1 = 0.0;
    for (int32_t ip = 0; ip < npg; ++ip) {
      const double aa = prim_exp[int(p0 + ip)];
      const double cc = prim_coef[int(p0 + ip)];
      const double e = exp(-aa * r2);
      const double t = cc * e;
      rad0 += t;
      rad1 += aa * t;
    }

    eval_shell_cart_components<true, false>(
        l,
        dx,
        dy,
        dz,
        rad0,
        rad1,
        0.0,
        sh_ao[pt_lane],
        sh_gx[pt_lane],
        sh_gy[pt_lane],
        sh_gz[pt_lane],
        nullptr);
  }

  __syncthreads();

  if (ic < nbf) {
    const int64_t idx = int64_t(p) * int64_t(nao) + int64_t(ao0 + ic);
    ao_out[idx] = sh_ao[pt_lane][ic];
    const int64_t gbase = idx * 3;
    ao_grad_out[gbase + 0] = sh_gx[pt_lane][ic];
    ao_grad_out[gbase + 1] = sh_gy[pt_lane][ic];
    ao_grad_out[gbase + 2] = sh_gz[pt_lane][ic];
  }
}

__global__ void contract_aos_cart_value_grad_vjp_atomgrad_kernel(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    const int32_t* point_atom,
    const double* w_pow,
    const double* bar_ao,
    const int32_t* shell_atom,
    int32_t natm,
    double* grad_out) {
  const int32_t sh = int32_t(blockIdx.x);
  const int pt_lane = int(threadIdx.y);
  const int p = int(blockIdx.y) * PT_TILE + pt_lane;
  const int ic = int(threadIdx.x);

  if (sh >= nshell) return;
  if (p >= npt) return;

  const int l = int(shell_l[sh]);
  const int nbf = ncart(l);
  const int ao0 = int(shell_ao_start[sh]);

  if (l > LMAX || nbf > NBF_MAX) return;
  if (ao0 < 0 || ao0 + nbf > nao) return;

  const int32_t ia_pt = point_atom ? point_atom[p] : -1;
  const int32_t ia_sh = shell_atom ? shell_atom[sh] : -1;
  if (ia_pt < 0 || ia_pt >= natm) return;
  if (ia_sh < 0 || ia_sh >= natm) return;

  __shared__ double sh_gx[PT_TILE][NBF_MAX];
  __shared__ double sh_gy[PT_TILE][NBF_MAX];
  __shared__ double sh_gz[PT_TILE][NBF_MAX];

  // Partial sums for this shell + point lane (per AO component).
  __shared__ double sh_sumx[PT_TILE][NBF_MAX];
  __shared__ double sh_sumy[PT_TILE][NBF_MAX];
  __shared__ double sh_sumz[PT_TILE][NBF_MAX];

  if (ic == 0) {
    const double cx = shell_cxyz[int(sh) * 3 + 0];
    const double cy = shell_cxyz[int(sh) * 3 + 1];
    const double cz = shell_cxyz[int(sh) * 3 + 2];

    const double x = points[int(p) * 3 + 0];
    const double y = points[int(p) * 3 + 1];
    const double z = points[int(p) * 3 + 2];

    const double dx = x - cx;
    const double dy = y - cy;
    const double dz = z - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;

    const int32_t p0 = shell_prim_start[sh];
    const int32_t npg = shell_nprim[sh];

    double rad0 = 0.0;
    double rad1 = 0.0;
    for (int32_t ip = 0; ip < npg; ++ip) {
      const double aa = prim_exp[int(p0 + ip)];
      const double cc = prim_coef[int(p0 + ip)];
      const double e = exp(-aa * r2);
      const double t = cc * e;
      rad0 += t;
      rad1 += aa * t;
    }

    eval_shell_cart_components<true, false>(
        l,
        dx,
        dy,
        dz,
        rad0,
        rad1,
        0.0,
        nullptr,
        sh_gx[pt_lane],
        sh_gy[pt_lane],
        sh_gz[pt_lane],
        nullptr);
  }

  // Ensure gradients ready before we use them.
  __syncthreads();

  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  if (ic < nbf) {
    const double bw = w_pow ? w_pow[p] : 1.0;
    const double b = bar_ao[int64_t(p) * int64_t(nao) + int64_t(ao0 + ic)];
    const double fac = bw * b;
    px = fac * sh_gx[pt_lane][ic];
    py = fac * sh_gy[pt_lane][ic];
    pz = fac * sh_gz[pt_lane][ic];
  }
  sh_sumx[pt_lane][ic] = px;
  sh_sumy[pt_lane][ic] = py;
  sh_sumz[pt_lane][ic] = pz;

  __syncthreads();

  // Reduce across ic within each point lane. Use a power-of-two reduction
  // starting from 64 (largest pow2 <= NBF_MAX) and guard by nbf so we handle
  // the non-power-of-two shell sizes correctly.
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (ic < stride) {
      const int j = ic + stride;
      if (j < nbf) {
        sh_sumx[pt_lane][ic] += sh_sumx[pt_lane][j];
        sh_sumy[pt_lane][ic] += sh_sumy[pt_lane][j];
        sh_sumz[pt_lane][ic] += sh_sumz[pt_lane][j];
      }
    }
    __syncthreads();
  }

  if (ic == 0) {
    const double gx = sh_sumx[pt_lane][0];
    const double gy = sh_sumy[pt_lane][0];
    const double gz = sh_sumz[pt_lane][0];

    // Point translation contribution (+) and basis-center contribution (-).
    const int64_t pbase = int64_t(ia_pt) * 3;
    atomicAdd(&grad_out[pbase + 0], gx);
    atomicAdd(&grad_out[pbase + 1], gy);
    atomicAdd(&grad_out[pbase + 2], gz);

    const int64_t sbase = int64_t(ia_sh) * 3;
    atomicAdd(&grad_out[sbase + 0], -gx);
    atomicAdd(&grad_out[sbase + 1], -gy);
    atomicAdd(&grad_out[sbase + 2], -gz);
  }
}

constexpr int BECKE_NATM_MAX = 128;

__global__ void becke_weight_vjp_atomgrad_kernel(
    const double* points,
    const double* weights,
    const double* bar_w,
    int32_t npt,
    const int32_t* point_atom,
    const double* atom_coords,
    const double* RAB,
    int32_t natm,
    int32_t becke_n,
    double* grad_out) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  if (natm <= 0 || natm > BECKE_NATM_MAX) return;

  const int32_t ia_pt = point_atom ? point_atom[p] : -1;
  if (ia_pt < 0 || ia_pt >= natm) return;

  const double bw = bar_w ? bar_w[p] : 0.0;
  if (bw == 0.0) return;

  const double x = points[int64_t(p) * 3 + 0];
  const double y = points[int64_t(p) * 3 + 1];
  const double z = points[int64_t(p) * 3 + 2];

  double rA[BECKE_NATM_MAX];
  double wx[BECKE_NATM_MAX];

  // Distances + initialize raw weights.
  for (int32_t A = 0; A < natm; ++A) {
    const double ax = atom_coords[int64_t(A) * 3 + 0];
    const double ay = atom_coords[int64_t(A) * 3 + 1];
    const double az = atom_coords[int64_t(A) * 3 + 2];
    const double dx = x - ax;
    const double dy = y - ay;
    const double dz = z - az;
    rA[A] = sqrt(dx * dx + dy * dy + dz * dz);
    wx[A] = 1.0;
  }

  // Forward Becke raw weight products.
  for (int32_t A = 0; A < natm; ++A) {
    for (int32_t B = A + 1; B < natm; ++B) {
      const double Rab = RAB[int64_t(A) * int64_t(natm) + int64_t(B)];
      if (Rab <= 1e-14) continue;
      double mu = (rA[A] - rA[B]) / Rab;
      if (mu < -1.0) mu = -1.0;
      if (mu > 1.0) mu = 1.0;

      double pp = mu;
      for (int32_t it = 0; it < becke_n; ++it) {
        pp = 0.5 * (3.0 * pp - pp * pp * pp);
      }

      const double sA = 0.5 * (1.0 - pp);
      const double sB = 1.0 - sA;
      wx[A] *= sA;
      wx[B] *= sB;
    }
  }

  double wsum = 0.0;
  for (int32_t A = 0; A < natm; ++A) wsum += wx[A];
  if (wsum == 0.0) wsum = 1.0;

  const double wi = wx[ia_pt];
  const double g = wi / wsum;
  if (g == 0.0) return;

  const double w_base = weights[p] / g;
  const double bar_g = bw * w_base;

  const double inv_wsum = 1.0 / wsum;
  const double inv_wsum2 = inv_wsum * inv_wsum;

  // Pairwise derivative contributions.
  for (int32_t A = 0; A < natm; ++A) {
    for (int32_t B = A + 1; B < natm; ++B) {
      const double Rab = RAB[int64_t(A) * int64_t(natm) + int64_t(B)];
      if (Rab <= 1e-14) continue;

      const double rAi = rA[A];
      const double rBi = rA[B];

      double mu_raw = (rAi - rBi) / Rab;
      double mu = mu_raw;
      if (mu < -1.0) mu = -1.0;
      if (mu > 1.0) mu = 1.0;

      // If clamped, derivative is zero (piecewise-constant outside [-1,1]).
      double dpp_dmu = (mu_raw == mu) ? 1.0 : 0.0;
      double pp = mu;
      for (int32_t it = 0; it < becke_n; ++it) {
        const double gprime = 1.5 * (1.0 - pp * pp);
        dpp_dmu *= gprime;
        pp = 0.5 * (3.0 * pp - pp * pp * pp);
      }

      const double sA = 0.5 * (1.0 - pp);
      const double sB = 1.0 - sA;

      const double wA = wx[A];
      const double wB = wx[B];

      double bar_wA = -bar_g * wi * inv_wsum2;
      double bar_wB = bar_wA;
      if (A == ia_pt) bar_wA += bar_g * inv_wsum;
      if (B == ia_pt) bar_wB += bar_g * inv_wsum;

      // Avoid division by zero on pathological points.
      const double inv_sA = (sA != 0.0) ? (1.0 / sA) : 0.0;
      const double inv_sB = (sB != 0.0) ? (1.0 / sB) : 0.0;

      const double bar_sA = bar_wA * wA * inv_sA;
      const double bar_sB = bar_wB * wB * inv_sB;

      // bar_mu = dE/dmu.
      const double bar_mu = 0.5 * dpp_dmu * (bar_sB - bar_sA);
      if (bar_mu == 0.0) continue;

      // Unit vectors uA, uB, vAB.
      double uAx = 0.0, uAy = 0.0, uAz = 0.0;
      if (rAi > 1e-16) {
        const double ax = atom_coords[int64_t(A) * 3 + 0];
        const double ay = atom_coords[int64_t(A) * 3 + 1];
        const double az = atom_coords[int64_t(A) * 3 + 2];
        uAx = (x - ax) / rAi;
        uAy = (y - ay) / rAi;
        uAz = (z - az) / rAi;
      }
      double uBx = 0.0, uBy = 0.0, uBz = 0.0;
      if (rBi > 1e-16) {
        const double bx = atom_coords[int64_t(B) * 3 + 0];
        const double by = atom_coords[int64_t(B) * 3 + 1];
        const double bz = atom_coords[int64_t(B) * 3 + 2];
        uBx = (x - bx) / rBi;
        uBy = (y - by) / rBi;
        uBz = (z - bz) / rBi;
      }

      const double Ax = atom_coords[int64_t(A) * 3 + 0];
      const double Ay = atom_coords[int64_t(A) * 3 + 1];
      const double Az = atom_coords[int64_t(A) * 3 + 2];
      const double Bx = atom_coords[int64_t(B) * 3 + 0];
      const double By = atom_coords[int64_t(B) * 3 + 1];
      const double Bz = atom_coords[int64_t(B) * 3 + 2];

      const double vABx = (Ax - Bx) / Rab;
      const double vABy = (Ay - By) / Rab;
      const double vABz = (Az - Bz) / Rab;

      const double coef = bar_mu / Rab;
      const double coef_rab = bar_mu * (rAi - rBi) / (Rab * Rab);

      // Point translation contribution: point moves with ia_pt.
      const int64_t pbase = int64_t(ia_pt) * 3;
      atomicAdd(&grad_out[pbase + 0], coef * (uAx - uBx));
      atomicAdd(&grad_out[pbase + 1], coef * (uAy - uBy));
      atomicAdd(&grad_out[pbase + 2], coef * (uAz - uBz));

      // Atom A contribution.
      const int64_t abase = int64_t(A) * 3;
      atomicAdd(&grad_out[abase + 0], -coef * uAx - coef_rab * vABx);
      atomicAdd(&grad_out[abase + 1], -coef * uAy - coef_rab * vABy);
      atomicAdd(&grad_out[abase + 2], -coef * uAz - coef_rab * vABz);

      // Atom B contribution.
      const int64_t bbase = int64_t(B) * 3;
      atomicAdd(&grad_out[bbase + 0], coef * uBx + coef_rab * vABx);
      atomicAdd(&grad_out[bbase + 1], coef * uBy + coef_rab * vABy);
      atomicAdd(&grad_out[bbase + 2], coef * uBz + coef_rab * vABz);
    }
  }
}

template <bool WANT_GRAD, bool WANT_LAPL>
__global__ void mos_cart_kernel(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    double* psi_grad,
    double* psi_lapl) {
  const int orb = int(blockIdx.x) * ORB_TILE + int(threadIdx.x);
  const int pt_lane = int(threadIdx.y);
  const int p = int(blockIdx.y) * PT_TILE + pt_lane;

  const bool valid_orb = (orb < nocc);
  const bool valid_p = (p < npt);

  double acc = 0.0;
  double acc_gx = 0.0;
  double acc_gy = 0.0;
  double acc_gz = 0.0;
  double acc_lapl = 0.0;

  __shared__ double sh_ao[PT_TILE][NBF_MAX];
  __shared__ double sh_gx[PT_TILE][NBF_MAX];
  __shared__ double sh_gy[PT_TILE][NBF_MAX];
  __shared__ double sh_gz[PT_TILE][NBF_MAX];
  __shared__ double sh_lapl[PT_TILE][NBF_MAX];

  for (int32_t sh = 0; sh < nshell; ++sh) {
    const int l = int(shell_l[sh]);
    const int nbf = ncart(l);
    const int ao0 = int(shell_ao_start[sh]);

    if (l > LMAX || nbf > NBF_MAX) {
      // Unsupported angular momentum. Bail out deterministically (zero output).
      acc = 0.0;
      acc_gx = acc_gy = acc_gz = 0.0;
      acc_lapl = 0.0;
      break;
    }

    if (int(threadIdx.x) == 0 && valid_p) {
      const double cx = shell_cxyz[int(sh) * 3 + 0];
      const double cy = shell_cxyz[int(sh) * 3 + 1];
      const double cz = shell_cxyz[int(sh) * 3 + 2];

      const double x = points[int(p) * 3 + 0];
      const double y = points[int(p) * 3 + 1];
      const double z = points[int(p) * 3 + 2];

      const double dx = x - cx;
      const double dy = y - cy;
      const double dz = z - cz;
      const double r2 = dx * dx + dy * dy + dz * dz;

      const int32_t p0 = shell_prim_start[sh];
      const int32_t npg = shell_nprim[sh];

      double rad0 = 0.0;
      double rad1 = 0.0;
      double rad2 = 0.0;
      for (int32_t ip = 0; ip < npg; ++ip) {
        const double aa = prim_exp[int(p0 + ip)];
        const double cc = prim_coef[int(p0 + ip)];
        const double e = exp(-aa * r2);
        const double t = cc * e;
        rad0 += t;
        if constexpr (WANT_GRAD || WANT_LAPL) rad1 += aa * t;
        if constexpr (WANT_LAPL) rad2 += (aa * aa) * t;
      }

      eval_shell_cart_components<WANT_GRAD, WANT_LAPL>(
          l,
          dx,
          dy,
          dz,
          rad0,
          rad1,
          rad2,
          sh_ao[pt_lane],
          sh_gx[pt_lane],
          sh_gy[pt_lane],
          sh_gz[pt_lane],
          sh_lapl[pt_lane]);
    }

    __syncthreads();

    if (valid_orb && valid_p) {
      if (ao0 + nbf > nao) {
        acc = 0.0;
        acc_gx = acc_gy = acc_gz = 0.0;
        acc_lapl = 0.0;
        break;
      }
      for (int ic = 0; ic < nbf; ++ic) {
        const double c = C_occ[(ao0 + ic) * nocc + orb];
        acc += sh_ao[pt_lane][ic] * c;
        if constexpr (WANT_GRAD) {
          acc_gx += sh_gx[pt_lane][ic] * c;
          acc_gy += sh_gy[pt_lane][ic] * c;
          acc_gz += sh_gz[pt_lane][ic] * c;
        }
        if constexpr (WANT_LAPL) {
          acc_lapl += sh_lapl[pt_lane][ic] * c;
        }
      }
    }

    __syncthreads();
  }

  if (valid_orb && valid_p) {
    psi[int(p) * nocc + orb] = acc;
    if constexpr (WANT_GRAD) {
      const int base = (int(p) * nocc + orb) * 3;
      psi_grad[base + 0] = acc_gx;
      psi_grad[base + 1] = acc_gy;
      psi_grad[base + 2] = acc_gz;
    }
    if constexpr (WANT_LAPL) {
      psi_lapl[int(p) * nocc + orb] = acc_lapl;
    }
  }
}

__global__ void rho_parts_kernel(
    const double* psi,
    const double* psi_grad,
    const double* psi_lapl,
    const double* dm1,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t need_grad,
    int32_t need_lapl,
    int32_t compute_tau,
    double* rho,
    double* rho_grad,
    double* tau,
    double* rho_lapl,
    double* rho_core,
    double* rho_act,
    double* rho_core_grad,
    double* rho_act_grad,
    double* rho_core_lapl,
    double* rho_act_lapl) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;

  if (ncas > MAX_NCAS) return;

  const double* psi_p = psi + int64_t(p) * int64_t(nocc);
  const double* grad_p = need_grad ? (psi_grad + int64_t(p) * int64_t(nocc) * 3) : nullptr;
  const double* lapl_p = need_lapl ? (psi_lapl + int64_t(p) * int64_t(nocc)) : nullptr;

  double rc = 0.0;
  for (int32_t i = 0; i < ncore; ++i) {
    const double x = psi_p[i];
    rc += x * x;
  }
  rc *= 2.0;

  double ra = 0.0;
  double s[MAX_NCAS];
  for (int32_t i = 0; i < ncas; ++i) {
    double sum = 0.0;
    for (int32_t j = 0; j < ncas; ++j) {
      const double dm = dm1[int64_t(i) * int64_t(ncas) + j];
      const double pj = psi_p[ncore + j];
      sum += dm * pj;
    }
    s[i] = sum;
    ra += psi_p[ncore + i] * sum;
  }

  rho_core[p] = rc;
  rho_act[p] = ra;
  rho[p] = rc + ra;

  if (need_grad) {
    double gc0 = 0.0, gc1 = 0.0, gc2 = 0.0;
    for (int32_t i = 0; i < ncore; ++i) {
      const double x = psi_p[i];
      const double gx = grad_p[i * 3 + 0];
      const double gy = grad_p[i * 3 + 1];
      const double gz = grad_p[i * 3 + 2];
      gc0 += x * gx;
      gc1 += x * gy;
      gc2 += x * gz;
    }
    gc0 *= 4.0;
    gc1 *= 4.0;
    gc2 *= 4.0;
    if (rho_core_grad) {
      rho_core_grad[int64_t(p) * 3 + 0] = gc0;
      rho_core_grad[int64_t(p) * 3 + 1] = gc1;
      rho_core_grad[int64_t(p) * 3 + 2] = gc2;
    }

    double ga0 = 0.0, ga1 = 0.0, ga2 = 0.0;
    for (int32_t i = 0; i < ncas; ++i) {
      const double ti = s[i];
      const double gx = grad_p[(ncore + i) * 3 + 0];
      const double gy = grad_p[(ncore + i) * 3 + 1];
      const double gz = grad_p[(ncore + i) * 3 + 2];
      ga0 += ti * gx;
      ga1 += ti * gy;
      ga2 += ti * gz;
    }
    ga0 *= 2.0;
    ga1 *= 2.0;
    ga2 *= 2.0;
    if (rho_act_grad) {
      rho_act_grad[int64_t(p) * 3 + 0] = ga0;
      rho_act_grad[int64_t(p) * 3 + 1] = ga1;
      rho_act_grad[int64_t(p) * 3 + 2] = ga2;
    }

    if (rho_grad) {
      rho_grad[int64_t(p) * 3 + 0] = gc0 + ga0;
      rho_grad[int64_t(p) * 3 + 1] = gc1 + ga1;
      rho_grad[int64_t(p) * 3 + 2] = gc2 + ga2;
    }

    if (compute_tau) {
      double tau_core = 0.0;
      for (int32_t i = 0; i < ncore; ++i) {
        const double gx = grad_p[i * 3 + 0];
        const double gy = grad_p[i * 3 + 1];
        const double gz = grad_p[i * 3 + 2];
        tau_core += gx * gx + gy * gy + gz * gz;
      }

      double grad_dm1_grad = 0.0;
      for (int c = 0; c < 3; ++c) {
        double u[MAX_NCAS];
        for (int32_t i = 0; i < ncas; ++i) {
          double sum = 0.0;
          for (int32_t j = 0; j < ncas; ++j) {
            const double dm = dm1[int64_t(i) * int64_t(ncas) + j];
            const double gj = grad_p[(ncore + j) * 3 + c];
            sum += dm * gj;
          }
          u[i] = sum;
        }
        for (int32_t i = 0; i < ncas; ++i) {
          const double gi = grad_p[(ncore + i) * 3 + c];
          grad_dm1_grad += u[i] * gi;
        }
      }

      const double tau_act = 0.5 * grad_dm1_grad;
      tau[p] = tau_core + tau_act;
    }
  }

  if (need_lapl) {
    double grad2_core = 0.0;
    double psi_lapl_core = 0.0;
    for (int32_t i = 0; i < ncore; ++i) {
      const double gx = grad_p[i * 3 + 0];
      const double gy = grad_p[i * 3 + 1];
      const double gz = grad_p[i * 3 + 2];
      grad2_core += gx * gx + gy * gy + gz * gz;
      psi_lapl_core += psi_p[i] * lapl_p[i];
    }
    const double rc_lapl = 4.0 * (grad2_core + psi_lapl_core);
    if (rho_core_lapl) rho_core_lapl[p] = rc_lapl;

    double grad_dm1_grad = 0.0;
    for (int c = 0; c < 3; ++c) {
      double u[MAX_NCAS];
      for (int32_t i = 0; i < ncas; ++i) {
        double sum = 0.0;
        for (int32_t j = 0; j < ncas; ++j) {
          const double dm = dm1[int64_t(i) * int64_t(ncas) + j];
          const double gj = grad_p[(ncore + j) * 3 + c];
          sum += dm * gj;
        }
        u[i] = sum;
      }
      for (int32_t i = 0; i < ncas; ++i) {
        const double gi = grad_p[(ncore + i) * 3 + c];
        grad_dm1_grad += u[i] * gi;
      }
    }

    double term2 = 0.0;
    for (int32_t i = 0; i < ncas; ++i) {
      term2 += lapl_p[ncore + i] * s[i];
    }
    term2 *= 2.0;

    const double ra_lapl = 2.0 * grad_dm1_grad + term2;
    if (rho_act_lapl) rho_act_lapl[p] = ra_lapl;

    if (rho_lapl) rho_lapl[p] = rc_lapl + ra_lapl;
  }
}

__global__ void build_pair_x_kernel(
    const double* psi,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pair_buf) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  const int32_t n2 = ncas * ncas;
  const double* psi_p = psi + int64_t(p) * int64_t(nocc) + ncore;
  double* out = pair_buf + int64_t(p) * int64_t(n2);
  for (int32_t i = 0; i < ncas; ++i) {
    const double pi = psi_p[i];
    for (int32_t j = 0; j < ncas; ++j) {
      out[i * ncas + j] = pi * psi_p[j];
    }
  }
}

__global__ void build_pair_y_kernel(
    const double* psi,
    const double* psi_grad,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t comp,
    double* pair_buf) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  const int32_t n2 = ncas * ncas;
  const double* psi_p = psi + int64_t(p) * int64_t(nocc) + ncore;
  const double* grad_p = psi_grad + int64_t(p) * int64_t(nocc) * 3 + int64_t(ncore) * 3;
  double* out = pair_buf + int64_t(p) * int64_t(n2);
  for (int32_t i = 0; i < ncas; ++i) {
    const double psi_i = psi_p[i];
    const double gi = grad_p[int64_t(i) * 3 + comp];
    for (int32_t j = 0; j < ncas; ++j) {
      const double psi_j = psi_p[j];
      const double gj = grad_p[int64_t(j) * 3 + comp];
      out[i * ncas + j] = gi * psi_j + psi_i * gj;
    }
  }
}

__global__ void pi_kernel(
    const double* rho_core,
    const double* rho_act,
    const double* pair_buf,
    const double* g_buf,
    int32_t npt,
    int32_t n2,
    double* pi) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  const double rc = rho_core[p];
  const double ra = rho_act[p];
  const double* x = pair_buf + int64_t(p) * int64_t(n2);
  const double* g = g_buf + int64_t(p) * int64_t(n2);
  double sum = 0.0;
  for (int32_t i = 0; i < n2; ++i) sum += x[i] * g[i];
  const double pi_act = 0.5 * sum;
  pi[p] = 0.25 * rc * rc + 0.5 * rc * ra + pi_act;
}

__global__ void pi_grad_kernel(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* psi,
    const double* psi_grad,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_grad) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  if (ncas > MAX_NCAS) return;

  const double rc = rho_core[p];
  const double ra = rho_act[p];

  const double* gc = rho_core_grad + int64_t(p) * 3;
  const double* ga = rho_act_grad + int64_t(p) * 3;

  double out0 = 0.5 * rc * gc[0] + 0.5 * (ra * gc[0] + rc * ga[0]);
  double out1 = 0.5 * rc * gc[1] + 0.5 * (ra * gc[1] + rc * ga[1]);
  double out2 = 0.5 * rc * gc[2] + 0.5 * (ra * gc[2] + rc * ga[2]);

  const double* psi_p = psi + int64_t(p) * int64_t(nocc) + ncore;
  const double* grad_p = psi_grad + int64_t(p) * int64_t(nocc) * 3 + int64_t(ncore) * 3;
  const double* g = g_buf + int64_t(p) * int64_t(ncas) * int64_t(ncas);

  for (int32_t i = 0; i < ncas; ++i) {
    double hi = 0.0;
    for (int32_t j = 0; j < ncas; ++j) {
      const double gij = g[int64_t(i) * ncas + j];
      const double gji = g[int64_t(j) * ncas + i];
      hi += (gij + gji) * psi_p[j];
    }
    out0 += hi * grad_p[int64_t(i) * 3 + 0];
    out1 += hi * grad_p[int64_t(i) * 3 + 1];
    out2 += hi * grad_p[int64_t(i) * 3 + 2];
  }

  pi_grad[int64_t(p) * 3 + 0] = out0;
  pi_grad[int64_t(p) * 3 + 1] = out1;
  pi_grad[int64_t(p) * 3 + 2] = out2;
}

__global__ void pi_lapl_base_kernel(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* rho_core_lapl,
    const double* rho_act_lapl,
    const double* psi,
    const double* psi_grad,
    const double* psi_lapl,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_lapl) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  if (ncas > MAX_NCAS) return;

  const double rc = rho_core[p];
  const double ra = rho_act[p];

  const double* gc = rho_core_grad + int64_t(p) * 3;
  const double* ga = rho_act_grad + int64_t(p) * 3;

  const double rc_lapl = rho_core_lapl[p];
  const double ra_lapl = rho_act_lapl[p];

  const double g2_core = gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2];
  const double pi_core_lapl = 0.5 * g2_core + 0.5 * rc * rc_lapl;

  const double dot_ca = gc[0] * ga[0] + gc[1] * ga[1] + gc[2] * ga[2];
  const double pi_cross_lapl = 0.5 * (ra * rc_lapl + 2.0 * dot_ca + rc * ra_lapl);

  const double* psi_p = psi + int64_t(p) * int64_t(nocc) + ncore;
  const double* grad_p = psi_grad + int64_t(p) * int64_t(nocc) * 3 + int64_t(ncore) * 3;
  const double* lapl_p = psi_lapl + int64_t(p) * int64_t(nocc) + ncore;
  const double* g = g_buf + int64_t(p) * int64_t(ncas) * int64_t(ncas);

  double h[MAX_NCAS];
  for (int32_t i = 0; i < ncas; ++i) {
    double hi = 0.0;
    for (int32_t j = 0; j < ncas; ++j) {
      const double gij = g[int64_t(i) * ncas + j];
      const double gji = g[int64_t(j) * ncas + i];
      hi += (gij + gji) * psi_p[j];
    }
    h[i] = hi;
  }

  double termA = 0.0;
  for (int32_t i = 0; i < ncas; ++i) termA += lapl_p[i] * h[i];

  for (int c = 0; c < 3; ++c) {
    for (int32_t i = 0; i < ncas; ++i) {
      double wi = 0.0;
      for (int32_t j = 0; j < ncas; ++j) {
        const double gij = g[int64_t(i) * ncas + j];
        const double gji = g[int64_t(j) * ncas + i];
        const double gsym = gij + gji;
        wi += gsym * grad_p[int64_t(j) * 3 + c];
      }
      termA += grad_p[int64_t(i) * 3 + c] * wi;
    }
  }

  pi_lapl[p] = pi_core_lapl + pi_cross_lapl + termA;
}

__global__ void pi_lapl_add_termB_kernel(
    const double* pair_buf,
    const double* gy_buf,
    int32_t npt,
    int32_t n2,
    double* pi_lapl) {
  const int32_t p = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (p >= npt) return;
  const double* y = pair_buf + int64_t(p) * int64_t(n2);
  const double* gy = gy_buf + int64_t(p) * int64_t(n2);
  double sum = 0.0;
  for (int32_t i = 0; i < n2; ++i) sum += y[i] * gy[i];
  pi_lapl[p] += sum;
}

}  // namespace

void orbitals_build_atom_centered_points_weights_f64(
    const double* center_xyz,
    const double* radial_r,
    const double* radial_wr,
    int32_t nrad,
    const double* angular_dirs,
    const double* angular_w,
    int32_t nang,
    double* pts_local,
    double* w_base,
    cudaStream_t stream,
    int32_t threads) {
  if (!center_xyz || !radial_r || !radial_wr || !angular_dirs || !angular_w || !pts_local || !w_base) return;
  if (nrad <= 0 || nang <= 0) return;
  const int32_t nloc = nrad * nang;
  const int32_t block = (threads > 0) ? threads : 256;
  const int32_t grid = (nloc + block - 1) / block;
  build_atom_centered_points_weights_kernel<<<grid, block, 0, stream>>>(
      center_xyz,
      radial_r,
      radial_wr,
      nrad,
      angular_dirs,
      angular_w,
      nang,
      pts_local,
      w_base);
}

void orbitals_becke_partition_atom_block_f64(
    const double* pts_local,
    const double* w_base,
    int32_t nloc,
    const double* atom_coords,
    const double* RAB,
    int32_t natm,
    int32_t atom_index,
    int32_t becke_n,
    double* rA,
    double* w_raw,
    double* w_atom,
    cudaStream_t stream,
    int32_t threads) {
  if (!pts_local || !w_base || !atom_coords || !RAB || !rA || !w_raw || !w_atom) return;
  if (nloc <= 0 || natm <= 0) return;
  if (atom_index < 0 || atom_index >= natm) return;
  const int32_t block = (threads > 0) ? threads : 256;
  const int32_t grid = (nloc + block - 1) / block;
  becke_partition_atom_block_kernel<<<grid, block, 0, stream>>>(
      pts_local,
      w_base,
      nloc,
      atom_coords,
      RAB,
      natm,
      atom_index,
      becke_n,
      rA,
      w_raw,
      w_atom);
}

extern "C" {

void orbitals_eval_aos_cart_value_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    double* ao,
    cudaStream_t stream) {
  if (!shell_cxyz || !shell_prim_start || !shell_nprim || !shell_l || !shell_ao_start || !prim_exp || !prim_coef || !points || !ao)
    return;
  if (nshell <= 0 || nao <= 0 || npt <= 0) return;

  dim3 block(NBF_MAX, PT_TILE, 1);
  dim3 grid(nshell, (npt + PT_TILE - 1) / PT_TILE, 1);
  aos_cart_value_kernel<<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      nao,
      points,
      npt,
      ao);
}

void orbitals_eval_aos_cart_value_grad_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    double* ao,
    double* ao_grad,
    cudaStream_t stream) {
  if (!shell_cxyz || !shell_prim_start || !shell_nprim || !shell_l || !shell_ao_start || !prim_exp || !prim_coef || !points || !ao ||
      !ao_grad)
    return;
  if (nshell <= 0 || nao <= 0 || npt <= 0) return;

  dim3 block(NBF_MAX, PT_TILE, 1);
  dim3 grid(nshell, (npt + PT_TILE - 1) / PT_TILE, 1);
  aos_cart_value_grad_kernel<<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      nao,
      points,
      npt,
      ao,
      ao_grad);
}

void orbitals_contract_aos_cart_value_grad_vjp_atomgrad_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    const int32_t* point_atom,
    const double* w_pow,
    const double* bar_ao,
    const int32_t* shell_atom,
    int32_t natm,
    double* grad_out,
    cudaStream_t stream) {
  if (!shell_cxyz || !shell_prim_start || !shell_nprim || !shell_l || !shell_ao_start || !prim_exp || !prim_coef || !points || !bar_ao ||
      !shell_atom || !grad_out)
    return;
  if (nshell <= 0 || nao <= 0 || npt <= 0 || natm <= 0) return;

  dim3 block(NBF_MAX, PT_TILE, 1);
  dim3 grid(nshell, (npt + PT_TILE - 1) / PT_TILE, 1);
  contract_aos_cart_value_grad_vjp_atomgrad_kernel<<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      nao,
      points,
      npt,
      point_atom,
      w_pow,
      bar_ao,
      shell_atom,
      natm,
      grad_out);
}

void orbitals_becke_weight_vjp_atomgrad_f64(
    const double* points,
    const double* weights,
    const double* bar_w,
    int32_t npt,
    const int32_t* point_atom,
    const double* atom_coords,
    const double* RAB,
    int32_t natm,
    int32_t becke_n,
    double* grad_out,
    cudaStream_t stream,
    int32_t threads) {
  if (!points || !weights || !bar_w || !point_atom || !atom_coords || !RAB || !grad_out) return;
  if (npt <= 0 || natm <= 0) return;
  const int32_t block = (threads > 0) ? threads : 256;
  const int32_t grid = (npt + block - 1) / block;
  becke_weight_vjp_atomgrad_kernel<<<grid, block, 0, stream>>>(
      points,
      weights,
      bar_w,
      npt,
      point_atom,
      atom_coords,
      RAB,
      natm,
      becke_n,
      grad_out);
}

void orbitals_eval_mos_cart_value_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    cudaStream_t stream) {
  dim3 block(ORB_TILE, PT_TILE, 1);
  dim3 grid((nocc + ORB_TILE - 1) / ORB_TILE, (npt + PT_TILE - 1) / PT_TILE, 1);
  mos_cart_kernel<false, false><<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      C_occ,
      nao,
      nocc,
      points,
      npt,
      psi,
      nullptr,
      nullptr);
}

void orbitals_eval_mos_cart_value_grad_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    double* psi_grad,
    cudaStream_t stream) {
  dim3 block(ORB_TILE, PT_TILE, 1);
  dim3 grid((nocc + ORB_TILE - 1) / ORB_TILE, (npt + PT_TILE - 1) / PT_TILE, 1);
  mos_cart_kernel<true, false><<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      C_occ,
      nao,
      nocc,
      points,
      npt,
      psi,
      psi_grad,
      nullptr);
}

void orbitals_eval_mos_cart_value_grad_lapl_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    double* psi_grad,
    double* psi_lapl,
    cudaStream_t stream) {
  dim3 block(ORB_TILE, PT_TILE, 1);
  dim3 grid((nocc + ORB_TILE - 1) / ORB_TILE, (npt + PT_TILE - 1) / PT_TILE, 1);
  mos_cart_kernel<true, true><<<grid, block, 0, stream>>>(
      shell_cxyz,
      shell_prim_start,
      shell_nprim,
      shell_l,
      shell_ao_start,
      prim_exp,
      prim_coef,
      nshell,
      C_occ,
      nao,
      nocc,
      points,
      npt,
      psi,
      psi_grad,
      psi_lapl);
}

void orbitals_eval_rho_parts_f64(
    const double* psi,
    const double* psi_grad,
    const double* psi_lapl,
    const double* dm1,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t need_grad,
    int32_t need_lapl,
    int32_t compute_tau,
    double* rho,
    double* rho_grad,
    double* tau,
    double* rho_lapl,
    double* rho_core,
    double* rho_act,
    double* rho_core_grad,
    double* rho_act_grad,
    double* rho_core_lapl,
    double* rho_act_lapl,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  rho_parts_kernel<<<grid, block, 0, stream>>>(
      psi,
      psi_grad,
      psi_lapl,
      dm1,
      ncore,
      ncas,
      npt,
      nocc,
      need_grad,
      need_lapl,
      compute_tau,
      rho,
      rho_grad,
      tau,
      rho_lapl,
      rho_core,
      rho_act,
      rho_core_grad,
      rho_act_grad,
      rho_core_lapl,
      rho_act_lapl);
}

void orbitals_build_pair_x_f64(
    const double* psi,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pair_buf,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  build_pair_x_kernel<<<grid, block, 0, stream>>>(psi, ncore, ncas, npt, nocc, pair_buf);
}

void orbitals_build_pair_y_f64(
    const double* psi,
    const double* psi_grad,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t comp,
    double* pair_buf,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  build_pair_y_kernel<<<grid, block, 0, stream>>>(psi, psi_grad, ncore, ncas, npt, nocc, comp, pair_buf);
}

void orbitals_eval_pi_f64(
    const double* rho_core,
    const double* rho_act,
    const double* pair_buf,
    const double* g_buf,
    int32_t npt,
    int32_t n2,
    double* pi,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  pi_kernel<<<grid, block, 0, stream>>>(rho_core, rho_act, pair_buf, g_buf, npt, n2, pi);
}

void orbitals_eval_pi_grad_f64(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* psi,
    const double* psi_grad,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_grad,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  pi_grad_kernel<<<grid, block, 0, stream>>>(
      rho_core,
      rho_act,
      rho_core_grad,
      rho_act_grad,
      psi,
      psi_grad,
      g_buf,
      ncore,
      ncas,
      npt,
      nocc,
      pi_grad);
}

void orbitals_eval_pi_lapl_base_f64(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* rho_core_lapl,
    const double* rho_act_lapl,
    const double* psi,
    const double* psi_grad,
    const double* psi_lapl,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_lapl,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  pi_lapl_base_kernel<<<grid, block, 0, stream>>>(
      rho_core,
      rho_act,
      rho_core_grad,
      rho_act_grad,
      rho_core_lapl,
      rho_act_lapl,
      psi,
      psi_grad,
      psi_lapl,
      g_buf,
      ncore,
      ncas,
      npt,
      nocc,
      pi_lapl);
}

void orbitals_pi_lapl_add_termB_f64(
    const double* pair_buf,
    const double* gy_buf,
    int32_t npt,
    int32_t n2,
    double* pi_lapl,
    cudaStream_t stream,
    int32_t threads) {
  const int32_t t = threads > 0 ? threads : 256;
  dim3 block(t, 1, 1);
  dim3 grid((npt + t - 1) / t, 1, 1);
  pi_lapl_add_termB_kernel<<<grid, block, 0, stream>>>(pair_buf, gy_buf, npt, n2, pi_lapl);
}

}  // extern "C"
