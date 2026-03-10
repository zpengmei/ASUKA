#include <cuda_runtime.h>

#include <cstdint>

#include "cueri_cuda_kernels_api.h"

namespace {

__global__ void KernelScatterERITilesOrdered(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* eri_mat,
    int64_t total,
    int64_t nPairAO) {
  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const int64_t tile_stride = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride;
    const int t = static_cast<int>(t64);
    const int64_t rem = idx - t64 * tile_stride;
    const int iab = static_cast<int>(rem / nCD);
    const int icd = static_cast<int>(rem - static_cast<int64_t>(iab) * nCD);

    const int ia = iab / nB;
    const int ib = iab - ia * nB;
    const int ic = icd / nD;
    const int id = icd - ic * nD;

    const int spab = static_cast<int>(task_spAB[t]);
    const int spcd = static_cast<int>(task_spCD[t]);
    const int A = static_cast<int>(sp_A[spab]);
    const int B = static_cast<int>(sp_B[spab]);
    const int C = static_cast<int>(sp_A[spcd]);
    const int D = static_cast<int>(sp_B[spcd]);

    const int a = static_cast<int>(shell_ao_start[A]) + ia;
    const int b = static_cast<int>(shell_ao_start[B]) + ib;
    const int c = static_cast<int>(shell_ao_start[C]) + ic;
    const int d = static_cast<int>(shell_ao_start[D]) + id;

    const int64_t ab_p = static_cast<int64_t>(a) * static_cast<int64_t>(nao) + static_cast<int64_t>(b);
    const int64_t ab_s = static_cast<int64_t>(b) * static_cast<int64_t>(nao) + static_cast<int64_t>(a);
    const int64_t cd_p = static_cast<int64_t>(c) * static_cast<int64_t>(nao) + static_cast<int64_t>(d);
    const int64_t cd_s = static_cast<int64_t>(d) * static_cast<int64_t>(nao) + static_cast<int64_t>(c);

    const double v = tile_vals[idx];
    if (v == 0.0) continue;

    atomicAdd(&eri_mat[ab_p * nPairAO + cd_p], v);
    if (C != D) atomicAdd(&eri_mat[ab_p * nPairAO + cd_s], v);
    if (A != B) atomicAdd(&eri_mat[ab_s * nPairAO + cd_p], v);
    if (A != B && C != D) atomicAdd(&eri_mat[ab_s * nPairAO + cd_s], v);

    if (spab != spcd) {
      atomicAdd(&eri_mat[cd_p * nPairAO + ab_p], v);
      if (A != B) atomicAdd(&eri_mat[cd_p * nPairAO + ab_s], v);
      if (C != D) atomicAdd(&eri_mat[cd_s * nPairAO + ab_p], v);
      if (A != B && C != D) atomicAdd(&eri_mat[cd_s * nPairAO + ab_s], v);
    }
  }
}

// ---------------------------------------------------------------------------
// Direct J/K contraction kernel: contracts ERI tiles with density matrix D
// and accumulates into Coulomb (J) and exchange (K) matrices.
//
// Handles 8-fold permutation symmetry of (μν|λσ):
//   (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ)
//           = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)
//
// J_{μν} = Σ_{λσ} (μν|λσ) D_{λσ}
// K_{μλ} = Σ_{νσ} (μν|λσ) D_{νσ}
//
// After this kernel, caller must symmetrize: J = 0.5*(J+J^T), K = 0.5*(K+K^T)
// ---------------------------------------------------------------------------
__global__ void KernelContractJKTilesOrdered(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* D_mat,   // (nao*nao,) density matrix, row-major
    double* J_mat,         // (nao*nao,) Coulomb output, row-major
    double* K_mat,         // (nao*nao,) exchange output, row-major (NULL to skip K)
    int64_t total) {
  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const int64_t tile_stride = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride;
    const int t = static_cast<int>(t64);
    const int64_t rem = idx - t64 * tile_stride;
    const int iab = static_cast<int>(rem / nCD);
    const int icd = static_cast<int>(rem - static_cast<int64_t>(iab) * nCD);

    const int ia = iab / nB;
    const int ib = iab - ia * nB;
    const int ic = icd / nD;
    const int id = icd - ic * nD;

    const int spab = static_cast<int>(task_spAB[t]);
    const int spcd = static_cast<int>(task_spCD[t]);
    const int A_sh = static_cast<int>(sp_A[spab]);
    const int B_sh = static_cast<int>(sp_B[spab]);
    const int C_sh = static_cast<int>(sp_A[spcd]);
    const int D_sh = static_cast<int>(sp_B[spcd]);

    const int a = static_cast<int>(shell_ao_start[A_sh]) + ia;
    const int b = static_cast<int>(shell_ao_start[B_sh]) + ib;
    const int c = static_cast<int>(shell_ao_start[C_sh]) + ic;
    const int d = static_cast<int>(shell_ao_start[D_sh]) + id;

    const double v = tile_vals[idx];
    if (v == 0.0) continue;

    const int64_t N = static_cast<int64_t>(nao);

    // Read density values (D is symmetric, D[p,q] = D[q,p])
    const double D_cd = D_mat[c * N + d];
    const double D_ab = D_mat[a * N + b];
    const double D_bd = D_mat[b * N + d];
    const double D_bc = D_mat[b * N + c];
    const double D_ad = D_mat[a * N + d];
    const double D_ac = D_mat[a * N + c];

    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);

    // --- J contributions ---
    if (J_mat != nullptr) {
      // D symmetric → D[c,d] = D[d,c], combine AB/CD swap pairs
      const double f_cd = cd_neq ? 2.0 : 1.0;
      const double f_ab = ab_neq ? 2.0 : 1.0;

      const double vJ_ab = f_cd * v * D_cd;
      atomicAdd(&J_mat[a * N + b], vJ_ab);
      if (ab_neq) atomicAdd(&J_mat[b * N + a], vJ_ab);

      if (spab != spcd) {
        const double vJ_cd = f_ab * v * D_ab;
        atomicAdd(&J_mat[c * N + d], vJ_cd);
        if (cd_neq) atomicAdd(&J_mat[d * N + c], vJ_cd);
      }
    }

    // --- K contributions ---
    if (K_mat != nullptr) {
      const double v_Dbd = v * D_bd;
      const double v_Dbc = v * D_bc;
      const double v_Dad = v * D_ad;
      const double v_Dac = v * D_ac;

      // Primary: eri[(a,b),(c,d)] → K[a,c] += v * D[b,d]
      atomicAdd(&K_mat[a * N + c], v_Dbd);
      if (cd_neq)           atomicAdd(&K_mat[a * N + d], v_Dbc);
      if (ab_neq)           atomicAdd(&K_mat[b * N + c], v_Dad);
      if (ab_neq && cd_neq) atomicAdd(&K_mat[b * N + d], v_Dac);

      // Bra-ket swap: eri[(c,d),(a,b)] → K[c,a] += v * D[d,b]
      if (spab != spcd) {
        atomicAdd(&K_mat[c * N + a], v_Dbd);
        if (ab_neq)           atomicAdd(&K_mat[c * N + b], v_Dad);
        if (cd_neq)           atomicAdd(&K_mat[d * N + a], v_Dbc);
        if (cd_neq && ab_neq) atomicAdd(&K_mat[d * N + b], v_Dac);
      }
    }
  }
}

}  // namespace

extern "C" cudaError_t cueri_contract_jk_tiles_ordered_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (ntasks == 0) return cudaSuccess;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  if (total <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelContractJKTilesOrdered<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start,
      nao,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      D_mat,
      J_mat,
      K_mat,
      total);
  return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// Multi-density variant: contracts ERI tiles with two density matrices (Da, Db)
// simultaneously, producing (Ja, Ka, Jb, Kb).  ERIs are read once.
// ---------------------------------------------------------------------------
namespace {
__global__ void KernelContractJKTilesOrderedMulti2(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* Da_mat,
    const double* Db_mat,
    double* Ja_mat,
    double* Ka_mat,
    double* Jb_mat,
    double* Kb_mat,
    int64_t total) {
  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const int64_t tile_stride = static_cast<int64_t>(nAB) * static_cast<int64_t>(nCD);

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
    const int64_t t64 = idx / tile_stride;
    const int t = static_cast<int>(t64);
    const int64_t rem = idx - t64 * tile_stride;
    const int iab = static_cast<int>(rem / nCD);
    const int icd = static_cast<int>(rem - static_cast<int64_t>(iab) * nCD);

    const int ia = iab / nB;
    const int ib = iab - ia * nB;
    const int ic = icd / nD;
    const int id = icd - ic * nD;

    const int spab = static_cast<int>(task_spAB[t]);
    const int spcd = static_cast<int>(task_spCD[t]);
    const int A_sh = static_cast<int>(sp_A[spab]);
    const int B_sh = static_cast<int>(sp_B[spab]);
    const int C_sh = static_cast<int>(sp_A[spcd]);
    const int D_sh = static_cast<int>(sp_B[spcd]);

    const int a = static_cast<int>(shell_ao_start[A_sh]) + ia;
    const int b = static_cast<int>(shell_ao_start[B_sh]) + ib;
    const int c = static_cast<int>(shell_ao_start[C_sh]) + ic;
    const int d = static_cast<int>(shell_ao_start[D_sh]) + id;

    const double v = tile_vals[idx];
    if (v == 0.0) continue;

    const int64_t N = static_cast<int64_t>(nao);
    const bool ab_neq = (A_sh != B_sh);
    const bool cd_neq = (C_sh != D_sh);
    const double f_cd = cd_neq ? 2.0 : 1.0;
    const double f_ab = ab_neq ? 2.0 : 1.0;

    // Process both densities with identical symmetry logic
    // Use a macro to avoid duplicating the J/K accumulation pattern
    #define MULTI2_ACCUM(D_ptr, J_ptr, K_ptr) do {                             \
      /* --- J contributions --- */                                            \
      if ((J_ptr) != nullptr) {                                                \
        const double D_cd_val = (D_ptr)[c * N + d];                            \
        const double vJ_ab = f_cd * v * D_cd_val;                              \
        atomicAdd(&(J_ptr)[a * N + b], vJ_ab);                                \
        if (ab_neq) atomicAdd(&(J_ptr)[b * N + a], vJ_ab);                    \
        if (spab != spcd) {                                                    \
          const double D_ab_val = (D_ptr)[a * N + b];                          \
          const double vJ_cd = f_ab * v * D_ab_val;                            \
          atomicAdd(&(J_ptr)[c * N + d], vJ_cd);                              \
          if (cd_neq) atomicAdd(&(J_ptr)[d * N + c], vJ_cd);                  \
        }                                                                      \
      }                                                                        \
      /* --- K contributions --- */                                            \
      if ((K_ptr) != nullptr) {                                                \
        const double v_Dbd = v * (D_ptr)[b * N + d];                           \
        const double v_Dbc = v * (D_ptr)[b * N + c];                           \
        const double v_Dad = v * (D_ptr)[a * N + d];                           \
        const double v_Dac = v * (D_ptr)[a * N + c];                           \
        atomicAdd(&(K_ptr)[a * N + c], v_Dbd);                                \
        if (cd_neq)           atomicAdd(&(K_ptr)[a * N + d], v_Dbc);           \
        if (ab_neq)           atomicAdd(&(K_ptr)[b * N + c], v_Dad);           \
        if (ab_neq && cd_neq) atomicAdd(&(K_ptr)[b * N + d], v_Dac);          \
        if (spab != spcd) {                                                    \
          atomicAdd(&(K_ptr)[c * N + a], v_Dbd);                              \
          if (ab_neq)           atomicAdd(&(K_ptr)[c * N + b], v_Dad);         \
          if (cd_neq)           atomicAdd(&(K_ptr)[d * N + a], v_Dbc);         \
          if (cd_neq && ab_neq) atomicAdd(&(K_ptr)[d * N + b], v_Dac);        \
        }                                                                      \
      }                                                                        \
    } while(0)

    MULTI2_ACCUM(Da_mat, Ja_mat, Ka_mat);
    MULTI2_ACCUM(Db_mat, Jb_mat, Kb_mat);

    #undef MULTI2_ACCUM
  }
}

}  // namespace

extern "C" cudaError_t cueri_contract_jk_tiles_ordered_multi2_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* Da_mat,
    const double* Db_mat,
    double* Ja_mat,
    double* Ka_mat,
    double* Jb_mat,
    double* Kb_mat,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (ntasks == 0) return cudaSuccess;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  if (total <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelContractJKTilesOrderedMulti2<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start,
      nao,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      Da_mat,
      Db_mat,
      Ja_mat,
      Ka_mat,
      Jb_mat,
      Kb_mat,
      total);
  return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// Warp-reduce J/K contraction (D9): 1 block per task, 32 threads (1 warp).
// Inner loops are warp-strided; warp_reduce_sum then single lane-0 atomicAdd.
// Reduces atomic contention: J has (nAB + nCD) writes vs nAB*nCD;
// K has nA*nC + nA*nD + nB*nC + nB*nD outer iters vs nAB*nCD * 4-6 atomics.
// ---------------------------------------------------------------------------
namespace {

// Local copy of warp_reduce_sum (defined in cueri_cuda_kernels.cu, not a shared header).
__device__ inline double warp_reduce_sum_jk(double x) {
  for (int off = 16; off > 0; off >>= 1)
    x += __shfl_down_sync(0xffffffff, x, off);
  return x;
}

__device__ __noinline__ void _contract_jk_warp_single(
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
  // --- J contributions ---
  if (J_mat != nullptr) {
    // Outer iab, inner warp-stride icd → J[a,b]
    for (int iab = 0; iab < nAB; iab++) {
      const int ia = iab / nB, ib = iab % nB;
      const int a = a0 + ia, b = b0 + ib;
      double pj = 0.0;
      for (int icd = lane; icd < nCD; icd += 32) {
        const int ic = icd / nD, id = icd % nD;
        pj += tile[iab * nCD + icd] * D_mat[(c0 + ic) * N + (d0 + id)];
      }
      pj = warp_reduce_sum_jk(pj);
      if (lane == 0 && pj != 0.0) {
        atomicAdd(&J_mat[a * N + b], f_cd * pj);
        if (ab_neq) atomicAdd(&J_mat[b * N + a], f_cd * pj);
      }
    }
    // Bra-ket swap: outer icd, inner warp-stride iab → J[c,d]
    if (bk_swap) {
      for (int icd = 0; icd < nCD; icd++) {
        const int ic = icd / nD, id = icd % nD;
        const int c = c0 + ic, d = d0 + id;
        double pj = 0.0;
        for (int iab = lane; iab < nAB; iab += 32) {
          const int ia = iab / nB, ib = iab % nB;
          pj += tile[iab * nCD + icd] * D_mat[(a0 + ia) * N + (b0 + ib)];
        }
        pj = warp_reduce_sum_jk(pj);
        if (lane == 0 && pj != 0.0) {
          atomicAdd(&J_mat[c * N + d], f_ab * pj);
          if (cd_neq) atomicAdd(&J_mat[d * N + c], f_ab * pj);
        }
      }
    }
  }

  // --- K contributions ---
  if (K_mat != nullptr) {
    // K[a,c]: outer (ia,ic), inner warp (ib,id)
    for (int iac = 0; iac < nA * nC; iac++) {
      const int ia = iac / nC, ic = iac % nC;
      const int a = a0 + ia, c = c0 + ic;
      double pk = 0.0;
      for (int ibd = lane; ibd < nB * nD; ibd += 32) {
        const int ib = ibd / nD, id = ibd % nD;
        pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(b0 + ib) * N + (d0 + id)];
      }
      pk = warp_reduce_sum_jk(pk);
      if (lane == 0 && pk != 0.0) {
        atomicAdd(&K_mat[a * N + c], pk);
        if (bk_swap) atomicAdd(&K_mat[c * N + a], pk);
      }
    }
    // K[a,d]: outer (ia,id), inner warp (ib,ic) — only if cd_neq
    if (cd_neq) {
      for (int iad = 0; iad < nA * nD; iad++) {
        const int ia = iad / nD, id = iad % nD;
        const int a = a0 + ia, d = d0 + id;
        double pk = 0.0;
        for (int ibc = lane; ibc < nB * nC; ibc += 32) {
          const int ib = ibc / nC, ic = ibc % nC;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(b0 + ib) * N + (c0 + ic)];
        }
        pk = warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[a * N + d], pk);
          if (bk_swap) atomicAdd(&K_mat[d * N + a], pk);
        }
      }
    }
    // K[b,c]: outer (ib,ic), inner warp (ia,id) — only if ab_neq
    if (ab_neq) {
      for (int ibc = 0; ibc < nB * nC; ibc++) {
        const int ib = ibc / nC, ic = ibc % nC;
        const int b = b0 + ib, c = c0 + ic;
        double pk = 0.0;
        for (int iad = lane; iad < nA * nD; iad += 32) {
          const int ia = iad / nD, id = iad % nD;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(a0 + ia) * N + (d0 + id)];
        }
        pk = warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[b * N + c], pk);
          if (bk_swap) atomicAdd(&K_mat[c * N + b], pk);
        }
      }
    }
    // K[b,d]: outer (ib,id), inner warp (ia,ic) — only if ab_neq && cd_neq
    if (ab_neq && cd_neq) {
      for (int ibd = 0; ibd < nB * nD; ibd++) {
        const int ib = ibd / nD, id = ibd % nD;
        const int b = b0 + ib, d = d0 + id;
        double pk = 0.0;
        for (int iac = lane; iac < nA * nC; iac += 32) {
          const int ia = iac / nC, ic = iac % nC;
          pk += tile[(ia * nB + ib) * nCD + ic * nD + id] * D_mat[(a0 + ia) * N + (c0 + ic)];
        }
        pk = warp_reduce_sum_jk(pk);
        if (lane == 0 && pk != 0.0) {
          atomicAdd(&K_mat[b * N + d], pk);
          if (bk_swap) atomicAdd(&K_mat[d * N + b], pk);
        }
      }
    }
  }
}

__global__ void KernelContractJKTilesOrderedWarp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    int ntasks) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;
  const int lane = static_cast<int>(threadIdx.x);  // 0..31

  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const double* tile = tile_vals + static_cast<int64_t>(t) * nAB * nCD;
  const int64_t N = static_cast<int64_t>(nao);

  const int spab = static_cast<int>(task_spAB[t]);
  const int spcd = static_cast<int>(task_spCD[t]);
  const int A_sh = static_cast<int>(sp_A[spab]);
  const int B_sh = static_cast<int>(sp_B[spab]);
  const int C_sh = static_cast<int>(sp_A[spcd]);
  const int D_sh = static_cast<int>(sp_B[spcd]);
  const int a0 = static_cast<int>(shell_ao_start[A_sh]);
  const int b0 = static_cast<int>(shell_ao_start[B_sh]);
  const int c0 = static_cast<int>(shell_ao_start[C_sh]);
  const int d0 = static_cast<int>(shell_ao_start[D_sh]);
  const bool ab_neq = (A_sh != B_sh);
  const bool cd_neq = (C_sh != D_sh);
  const bool bk_swap = (spab != spcd);
  const double f_ab = ab_neq ? 2.0 : 1.0;
  const double f_cd = cd_neq ? 2.0 : 1.0;

  _contract_jk_warp_single(tile, D_mat, J_mat, K_mat, lane,
                            nAB, nCD, nA, nB, nC, nD,
                            a0, b0, c0, d0,
                            ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
}

__global__ void KernelContractJKTilesOrderedMulti2Warp(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* Da_mat,
    const double* Db_mat,
    double* Ja_mat,
    double* Ka_mat,
    double* Jb_mat,
    double* Kb_mat,
    int ntasks) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= ntasks) return;
  const int lane = static_cast<int>(threadIdx.x);  // 0..31

  const int nAB = nA * nB;
  const int nCD = nC * nD;
  const double* tile = tile_vals + static_cast<int64_t>(t) * nAB * nCD;
  const int64_t N = static_cast<int64_t>(nao);

  const int spab = static_cast<int>(task_spAB[t]);
  const int spcd = static_cast<int>(task_spCD[t]);
  const int A_sh = static_cast<int>(sp_A[spab]);
  const int B_sh = static_cast<int>(sp_B[spab]);
  const int C_sh = static_cast<int>(sp_A[spcd]);
  const int D_sh = static_cast<int>(sp_B[spcd]);
  const int a0 = static_cast<int>(shell_ao_start[A_sh]);
  const int b0 = static_cast<int>(shell_ao_start[B_sh]);
  const int c0 = static_cast<int>(shell_ao_start[C_sh]);
  const int d0 = static_cast<int>(shell_ao_start[D_sh]);
  const bool ab_neq = (A_sh != B_sh);
  const bool cd_neq = (C_sh != D_sh);
  const bool bk_swap = (spab != spcd);
  const double f_ab = ab_neq ? 2.0 : 1.0;
  const double f_cd = cd_neq ? 2.0 : 1.0;

  _contract_jk_warp_single(tile, Da_mat, Ja_mat, Ka_mat, lane,
                            nAB, nCD, nA, nB, nC, nD,
                            a0, b0, c0, d0,
                            ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
  __syncwarp();  // ensure all threads complete first pass before second
  _contract_jk_warp_single(tile, Db_mat, Jb_mat, Kb_mat, lane,
                            nAB, nCD, nA, nB, nC, nD,
                            a0, b0, c0, d0,
                            ab_neq, cd_neq, bk_swap, f_ab, f_cd, N);
}

}  // namespace (warp JK)

extern "C" cudaError_t cueri_contract_jk_warp_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* D_mat,
    double* J_mat,
    double* K_mat,
    cudaStream_t stream) {
  if (ntasks <= 0) return cudaSuccess;
  if (nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) return cudaErrorInvalidValue;
  KernelContractJKTilesOrderedWarp<<<static_cast<unsigned int>(ntasks), 32, 0, stream>>>(
      task_spAB, task_spCD, sp_A, sp_B, shell_ao_start,
      nao, nA, nB, nC, nD,
      tile_vals, D_mat, J_mat, K_mat, ntasks);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_contract_jk_warp_multi2_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    const double* Da_mat,
    const double* Db_mat,
    double* Ja_mat,
    double* Ka_mat,
    double* Jb_mat,
    double* Kb_mat,
    cudaStream_t stream) {
  if (ntasks <= 0) return cudaSuccess;
  if (nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) return cudaErrorInvalidValue;
  KernelContractJKTilesOrderedMulti2Warp<<<static_cast<unsigned int>(ntasks), 32, 0, stream>>>(
      task_spAB, task_spCD, sp_A, sp_B, shell_ao_start,
      nao, nA, nB, nC, nD,
      tile_vals, Da_mat, Db_mat, Ja_mat, Ka_mat, Jb_mat, Kb_mat, ntasks);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t cueri_scatter_eri_tiles_ordered_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,
    double* eri_mat,
    cudaStream_t stream,
    int threads) {
  if (ntasks < 0 || nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0 || threads <= 0 || threads > 1024) {
    return cudaErrorInvalidValue;
  }
  if (ntasks == 0) return cudaSuccess;

  const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
  const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
  const int64_t total = static_cast<int64_t>(ntasks) * nAB * nCD;
  const int64_t nPairAO = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
  if (total <= 0 || nPairAO <= 0) return cudaErrorInvalidValue;

  int64_t blocks64 = (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;

  KernelScatterERITilesOrdered<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      task_spAB,
      task_spCD,
      sp_A,
      sp_B,
      shell_ao_start,
      nao,
      nA,
      nB,
      nC,
      nD,
      tile_vals,
      eri_mat,
      total,
      nPairAO);
  return cudaPeekAtLastError();
}
