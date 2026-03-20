#pragma once
// Mako TF32 Tensor Core GEMM via CuTe TiledMMA.
//
// Global FP32 staging (converted once from FP64) + direct CuTe MMA
// from the staging buffer with dynamic strides.
//
// Only enabled under #ifdef MAKO_HAS_CUTLASS.

#ifdef MAKO_HAS_CUTLASS

#include <cudaTypedefs.h>
#ifndef PFN_cuTensorMapEncodeTiled
using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;
#endif
#ifndef PFN_cuTensorMapEncodeIm2col
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;
#endif
#ifndef PFN_cuTensorMapReplaceAddress
using PFN_cuTensorMapReplaceAddress = PFN_cuTensorMapReplaceAddress_v12000;
#endif

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

namespace mako {

static constexpr int kMmaM = 16;
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 8;

static constexpr int kCutlassMinM = 16;
static constexpr int kCutlassMinN = 8;
static constexpr int kCutlassMinK = 8;

__device__ inline bool cutlass_gemm_eligible(int M, int N, int K) {
    return M >= kCutlassMinM && N >= kCutlassMinN && K >= kCutlassMinK;
}

__device__ __host__ __forceinline__ int pad8(int x) { return (x + 7) & ~7; }
__device__ __host__ __forceinline__ int pad16(int x) { return (x + 15) & ~15; }

// Staging: global A[Mp×Kp] + B[Np×Kp] + per-warp tile_C[16×8]
__host__ inline int tf32_staging_bytes(int M, int N, int K, int nwarps = 4) {
    int Mp = pad16(M), Kp = pad8(K), Np = pad8(N);
    int global = (Mp * Kp + Np * Kp) * (int)sizeof(float);
    int per_warp_out = nwarps * kMmaM * kMmaN * (int)sizeof(float);
    return global + per_warp_out + 16;  // +16 alignment
}

// =====================================================================
// FP64 warp-tiled GEMM (unchanged)
// =====================================================================
__device__ inline void smem_gemm_AB_accum_cute(
    double* __restrict__ C, const double* __restrict__ A, const double* __restrict__ B,
    int M, int N, int K, int tid, int nthreads, int ldb = 0)
{
    if (ldb <= 0) ldb = N;
    const int warp_id = tid / 32, lane_id = tid % 32, nwarps = nthreads / 32;
    const int tiles_m = (M + 7) / 8, tiles_n = (N + 7) / 8;
    for (int tile = warp_id; tile < tiles_m * tiles_n; tile += nwarps) {
        const int i0 = (tile / tiles_n) * 8, j0 = (tile % tiles_n) * 8;
        const int tr = min(8, M - i0), tc = min(8, N - j0);
        const int lr = lane_id / 4, lc = (lane_id % 4) * 2;
        double acc0 = 0, acc1 = 0;
        int k = 0;
        for (; k + 3 < K; k += 4) {
            double a0=0,a1=0,a2=0,a3=0;
            if(lr<tr){int gi=i0+lr;a0=A[gi*K+k];a1=A[gi*K+k+1];a2=A[gi*K+k+2];a3=A[gi*K+k+3];}
            if(lc<tc)acc0+=a0*B[k*ldb+j0+lc]+a1*B[(k+1)*ldb+j0+lc]+a2*B[(k+2)*ldb+j0+lc]+a3*B[(k+3)*ldb+j0+lc];
            if(lc+1<tc)acc1+=a0*B[k*ldb+j0+lc+1]+a1*B[(k+1)*ldb+j0+lc+1]+a2*B[(k+2)*ldb+j0+lc+1]+a3*B[(k+3)*ldb+j0+lc+1];
        }
        for (; k<K; ++k) {
            double a=(lr<tr)?A[(i0+lr)*K+k]:0;
            if(lc<tc)acc0+=a*B[k*ldb+j0+lc]; if(lc+1<tc)acc1+=a*B[k*ldb+j0+lc+1];
        }
        if(lr<tr){if(lc<tc)C[(i0+lr)*N+j0+lc]+=acc0;if(lc+1<tc)C[(i0+lr)*N+j0+lc+1]+=acc1;}
    }
}

// =====================================================================
// TF32 Tensor Core GEMM — global staging + direct CuTe MMA
// =====================================================================

using TF32Mma = cute::SM80_16x8x8_F32TF32TF32F32_TN;
using TF32TiledMma = cute::TiledMMA<cute::MMA_Atom<TF32Mma>,
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>>;

// --- Non-transposed: C[M×N] += A[M×K] × B[K×N] ---
__device__ inline void smem_gemm_AB_accum_tf32(
    double* __restrict__ C,
    const double* __restrict__ A,
    const double* __restrict__ B,
    int M, int N, int K,
    int tid, int nthreads,
    float* __restrict__ staging,
    int ldb = 0)
{
#if __CUDA_ARCH__ >= 800
    using namespace cute;
    using TF32 = cutlass::tfloat32_t;
    if (ldb <= 0) ldb = N;

    const int warp_id = tid / 32, lane_id = tid % 32, nwarps = nthreads / 32;
    const int Mp = pad16(M), Kp = pad8(K), Np = pad8(N);

    float* stg_A = staging;            // [Mp × Kp] row-major
    float* stg_B = staging + Mp * Kp;  // [Np × Kp] (N, K) K-contiguous

    // Phase 1: cooperative FP64→FP32 conversion (all threads, once)
    for (int i = tid; i < Mp * Kp; i += nthreads) {
        int r = i / Kp, c = i % Kp;
        stg_A[i] = (r < M && c < K) ? __double2float_rn(A[r * K + c]) : 0.0f;
    }
    // B[K×N] row-major → staging (N, K) K-contiguous
    for (int i = tid; i < Np * Kp; i += nthreads) {
        int n = i / Kp, k = i % Kp;
        stg_B[i] = (k < K && n < N) ? __double2float_rn(B[k * ldb + n]) : 0.0f;
    }
    __syncthreads();

    // Phase 2: per-warp MMA with CuTe (direct from global staging)
    TF32TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    // Per-warp output tile
    float* tile_C = stg_B + Np * Kp + warp_id * (kMmaM * kMmaN);
    auto sC = make_tensor(make_smem_ptr(tile_C), Layout<Shape<_16, _8>, Stride<_8, _1>>{});

    const int tiles_m = Mp / kMmaM, tiles_n = Np / kMmaN;
    for (int tile = warp_id; tile < tiles_m * tiles_n; tile += nwarps) {
        const int i0 = (tile / tiles_n) * kMmaM;
        const int j0 = (tile % tiles_n) * kMmaN;

        auto acc = thr_mma.partition_fragment_C(sC);
        clear(acc);

        for (int k0 = 0; k0 < Kp; k0 += kMmaK) {
            // Direct CuTe tensors from global staging with dynamic stride Kp
            auto sA_k = make_tensor(
                make_smem_ptr(reinterpret_cast<TF32*>(stg_A + i0 * Kp + k0)),
                make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Kp, Int<1>{})));
            auto sB_k = make_tensor(
                make_smem_ptr(reinterpret_cast<TF32*>(stg_B + j0 * Kp + k0)),
                make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Kp, Int<1>{})));

            auto tCsA = thr_mma.partition_A(sA_k);
            auto tCsB = thr_mma.partition_B(sB_k);
            auto tCrA = thr_mma.partition_fragment_A(sA_k);
            auto tCrB = thr_mma.partition_fragment_B(sB_k);

            copy(tCsA, tCrA);
            copy(tCsB, tCrB);
            gemm(tiled_mma, acc, tCrA, tCrB, acc);
        }

        auto tCsC = thr_mma.partition_C(sC);
        copy(acc, tCsC);
        __syncwarp();

        for (int i = lane_id; i < kMmaM * kMmaN; i += 32) {
            int r = i0 + i / kMmaN, c = j0 + i % kMmaN;
            if (r < M && c < N)
                C[r * N + c] += static_cast<double>(tile_C[i]);
        }
        __syncwarp();
    }
#endif
}

// --- Transposed: C[M×N] += A[M×K] × B_orig[N×K]^T ---
__device__ inline void smem_gemm_ABt_accum_tf32(
    double* __restrict__ C,
    const double* __restrict__ A,
    const double* __restrict__ B_orig,
    int M, int N, int K,
    int tid, int nthreads,
    float* __restrict__ staging)
{
#if __CUDA_ARCH__ >= 800
    using namespace cute;
    using TF32 = cutlass::tfloat32_t;

    const int warp_id = tid / 32, lane_id = tid % 32, nwarps = nthreads / 32;
    const int Mp = pad16(M), Kp = pad8(K), Np = pad8(N);

    float* stg_A = staging;
    float* stg_B = staging + Mp * Kp;

    for (int i = tid; i < Mp * Kp; i += nthreads) {
        int r = i / Kp, c = i % Kp;
        stg_A[i] = (r < M && c < K) ? __double2float_rn(A[r * K + c]) : 0.0f;
    }
    // B_orig[N×K] row-major → staging (N, K) K-contiguous (direct copy + pad)
    for (int i = tid; i < Np * Kp; i += nthreads) {
        int n = i / Kp, k = i % Kp;
        stg_B[i] = (n < N && k < K) ? __double2float_rn(B_orig[n * K + k]) : 0.0f;
    }
    __syncthreads();

    TF32TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    float* tile_C = stg_B + Np * Kp + warp_id * (kMmaM * kMmaN);
    auto sC = make_tensor(make_smem_ptr(tile_C), Layout<Shape<_16, _8>, Stride<_8, _1>>{});

    const int tiles_m = Mp / kMmaM, tiles_n = Np / kMmaN;
    for (int tile = warp_id; tile < tiles_m * tiles_n; tile += nwarps) {
        const int i0 = (tile / tiles_n) * kMmaM;
        const int j0 = (tile % tiles_n) * kMmaN;

        auto acc = thr_mma.partition_fragment_C(sC);
        clear(acc);

        for (int k0 = 0; k0 < Kp; k0 += kMmaK) {
            auto sA_k = make_tensor(
                make_smem_ptr(reinterpret_cast<TF32*>(stg_A + i0 * Kp + k0)),
                make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Kp, Int<1>{})));
            auto sB_k = make_tensor(
                make_smem_ptr(reinterpret_cast<TF32*>(stg_B + j0 * Kp + k0)),
                make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Kp, Int<1>{})));

            auto tCsA = thr_mma.partition_A(sA_k);
            auto tCsB = thr_mma.partition_B(sB_k);
            auto tCrA = thr_mma.partition_fragment_A(sA_k);
            auto tCrB = thr_mma.partition_fragment_B(sB_k);

            copy(tCsA, tCrA);
            copy(tCsB, tCrB);
            gemm(tiled_mma, acc, tCrA, tCrB, acc);
        }

        auto tCsC = thr_mma.partition_C(sC);
        copy(acc, tCsC);
        __syncwarp();

        for (int i = lane_id; i < kMmaM * kMmaN; i += 32) {
            int r = i0 + i / kMmaN, c = j0 + i % kMmaN;
            if (r < M && c < N)
                C[r * N + c] += static_cast<double>(tile_C[i]);
        }
        __syncwarp();
    }
#endif
}

}  // namespace mako

#endif  // MAKO_HAS_CUTLASS
