#pragma once

#include <cstdint>

// Shared device helpers for fused ERI->J/K kernels.
//
// NOTE: CUDA separable compilation is OFF in this project. Keep these helpers
// header-only with internal linkage so each TU can call them locally.

namespace {

__device__ __forceinline__ double cueri_warp_reduce_sum_jk(double x) {
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, off);
  }
  return x;
}

// Contract one ERI tile into Coulomb and exchange: J += J(D), K += K(D).
// Uses 1 warp per task; inner loops are warp-strided; lane-0 atomics only.
__device__ __forceinline__ void cueri_contract_jk_warp_single(
    const double* __restrict__ tile,
    const double* __restrict__ D_mat,
    double* J_mat,
    double* K_mat,
    int lane,
    int nAB,
    int nCD,
    int nA,
    int nB,
    int nC,
    int nD,
    int a0,
    int b0,
    int c0,
    int d0,
    bool ab_neq,
    bool cd_neq,
    bool bk_swap,
    double f_ab,
    double f_cd,
    int64_t N) {
  if (J_mat != nullptr) {
    for (int iab = 0; iab < nAB; iab++) {
      const int ia = iab / nB;
      const int ib = iab - ia * nB;
      const int a = a0 + ia;
      const int b = b0 + ib;
      double pj = 0.0;
      for (int icd = lane; icd < nCD; icd += 32) {
        const int ic = icd / nD;
        const int id = icd - ic * nD;
        pj += tile[iab * nCD + icd] * D_mat[(c0 + ic) * N + (d0 + id)];
      }
      pj = cueri_warp_reduce_sum_jk(pj);
      if (lane == 0 && pj != 0.0) {
        atomicAdd(&J_mat[a * N + b], f_cd * pj);
        if (ab_neq) atomicAdd(&J_mat[b * N + a], f_cd * pj);
      }
    }

    if (bk_swap) {
      for (int icd = 0; icd < nCD; icd++) {
        const int ic = icd / nD;
        const int id = icd - ic * nD;
        const int c = c0 + ic;
        const int d = d0 + id;
        double pj = 0.0;
        for (int iab = lane; iab < nAB; iab += 32) {
          const int ia = iab / nB;
          const int ib = iab - ia * nB;
          pj += tile[iab * nCD + icd] * D_mat[(a0 + ia) * N + (b0 + ib)];
        }
        pj = cueri_warp_reduce_sum_jk(pj);
        if (lane == 0 && pj != 0.0) {
          atomicAdd(&J_mat[c * N + d], f_ab * pj);
          if (cd_neq) atomicAdd(&J_mat[d * N + c], f_ab * pj);
        }
      }
    }
  }

  if (K_mat != nullptr) {
    for (int iac = 0; iac < nA * nC; iac++) {
      const int ia = iac / nC;
      const int ic = iac - ia * nC;
      const int a = a0 + ia;
      const int c = c0 + ic;
      double pk = 0.0;
      for (int ibd = lane; ibd < nB * nD; ibd += 32) {
        const int ib = ibd / nD;
        const int id = ibd - ib * nD;
        pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(b0 + ib) * N + (d0 + id)];
      }
      pk = cueri_warp_reduce_sum_jk(pk);
      if (lane == 0 && pk != 0.0) {
        atomicAdd(&K_mat[a * N + c], pk);
        if (bk_swap) atomicAdd(&K_mat[c * N + a], pk);
      }
    }

    if (cd_neq) {
      for (int iad = 0; iad < nA * nD; iad++) {
        const int ia = iad / nD;
        const int id = iad - ia * nD;
        const int a = a0 + ia;
        const int d = d0 + id;
        double pk = 0.0;
        for (int ibc = lane; ibc < nB * nC; ibc += 32) {
          const int ib = ibc / nC;
          const int ic = ibc - ib * nC;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(b0 + ib) * N + (c0 + ic)];
        }
        pk = cueri_warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[a * N + d], pk);
          if (bk_swap) atomicAdd(&K_mat[d * N + a], pk);
        }
      }
    }

    if (ab_neq) {
      for (int ibc = 0; ibc < nB * nC; ibc++) {
        const int ib = ibc / nC;
        const int ic = ibc - ib * nC;
        const int b = b0 + ib;
        const int c = c0 + ic;
        double pk = 0.0;
        for (int iad = lane; iad < nA * nD; iad += 32) {
          const int ia = iad / nD;
          const int id = iad - ia * nD;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(a0 + ia) * N + (d0 + id)];
        }
        pk = cueri_warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[b * N + c], pk);
          if (bk_swap) atomicAdd(&K_mat[c * N + b], pk);
        }
      }
    }

    if (ab_neq && cd_neq) {
      for (int ibd = 0; ibd < nB * nD; ibd++) {
        const int ib = ibd / nD;
        const int id = ibd - ib * nD;
        const int b = b0 + ib;
        const int d = d0 + id;
        double pk = 0.0;
        for (int iac = lane; iac < nA * nC; iac += 32) {
          const int ia = iac / nC;
          const int ic = iac - ia * nC;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(a0 + ia) * N + (c0 + ic)];
        }
        pk = cueri_warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[b * N + d], pk);
          if (bk_swap) atomicAdd(&K_mat[d * N + b], pk);
        }
      }
    }
  }
}

}  // namespace
