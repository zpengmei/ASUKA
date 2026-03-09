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
      {                                                                        \
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
