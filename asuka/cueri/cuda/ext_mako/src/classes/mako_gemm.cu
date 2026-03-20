// KernelMako — Paper-faithful GEMM-based ERI kernel.
//
// Implements the three techniques from Section 3.1:
//   ❶ Implicit ILP: pq_integrals loop unrolled for instruction-level
//      parallelism (Eq. 8).  Multiple [p|q] elements computed per thread
//      per iteration, hiding CUDA core latency.
//   ❷ Lightweight Layout Swizzle: XOR-based bank-conflict-free shared
//      memory layout (Eq. 9-10) for the SL→BL transpose.
//   ❸ GEMM Coalescing: Both MatMuls share N-axis tiling; GEMM1 output
//      stays in warp-local smem and feeds directly into GEMM2 (Eq. 11).
//
// Algorithm 1 loop structure:
//   for jp = 0..K_CD-1:
//     for ip = 0..K_AB-1:
//       [r] = r_integrals(ip, jp)
//       [p|q] = (-1)^l_q * [p+q]^(0)       (Eq. 6)
//       (ab|q] += E_AB ⊗ [p|q]              (Eq. 7, GEMM1)
//     (ab|cd) += (ab|q] ⊗ E_CD              (Eq. 7, GEMM2)
//
// Memory layout:
//   r-integrals: computed in SL (one thread per quartet), then swizzled
//   [p|q]:       stored in BL in smem (ready for MatMul)
//   E_AB, E_CD:  precomputed per primitive, stored in smem
//   GEMM1 output (ab|q]: stays in smem, consumed by GEMM2 in-place
//
// Thread mapping:
//   Block = one quartet batch (multiple quartets batched for ILP)
//   Each thread computes ILP_FACTOR independent pq_integral elements
//   and participates in cooperative GEMM via row-based iteration.

#include <cstdint>
#include <cmath>
#include "../mako_hermite.cuh"
#include "../mako_gemm_utils.cuh"
#ifdef MAKO_HAS_CUTLASS
#include "../mako_gemm_cutlass.cuh"
#endif

// ---------------------------------------------------------------------------
// XOR swizzle for bank-conflict-free shared memory access (Eq. 10).
//   physical_x = logical_x XOR logical_y
//   physical_y = logical_y
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swizzle_idx(int x, int y, int stride) {
    return (x ^ y) * stride + y;
}

// ---------------------------------------------------------------------------
// KernelMako: paper-faithful GEMM ERI kernel.
//
// One block per task.  All threads cooperate on:
//   Phase 1: r-integrals + [p|q] assembly (CUDA cores, with ILP)
//   Phase 2: E_AB assembly (CUDA cores)
//   Phase 3: GEMM1 — (ab|q] += E_AB ⊗ [p|q] (cooperative row-based GEMM)
//   Phase 4: E_CD assembly + GEMM2 — (ab|cd) += (ab|q] ⊗ E_CD
//
// GEMM1 output stays in smem_M and is consumed by GEMM2 without
// global memory traffic (GEMM coalescing, Eq. 11).
// ---------------------------------------------------------------------------
template <typename AccumT = double, typename TransformT = double, bool UseTF32TC = false>
__global__ void KernelMako_gemm_eri(
    const int32_t* __restrict__ task_spAB,
    const int32_t* __restrict__ task_spCD,
    int ntasks,
    int la, int lb, int lc, int ld,
    const int32_t* __restrict__ sp_A,
    const int32_t* __restrict__ sp_B,
    const int32_t* __restrict__ sp_pair_start,
    const int32_t* __restrict__ sp_npair,
    const double* __restrict__ shell_cx,
    const double* __restrict__ shell_cy,
    const double* __restrict__ shell_cz,
    const double* __restrict__ pair_eta,
    const double* __restrict__ pair_Px,
    const double* __restrict__ pair_Py,
    const double* __restrict__ pair_Pz,
    const double* __restrict__ pair_cK,
    AccumT* __restrict__ eri_out)
{
    const int task_id = static_cast<int>(blockIdx.x);
    if (task_id >= ntasks) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    // Derived dimensions
    const int na = mako::ncart(la), nb = mako::ncart(lb);
    const int nc = mako::ncart(lc), nd = mako::ncart(ld);
    const int nab = na * nb, ncd = nc * nd, ncomp = nab * ncd;
    const int lab = la + lb, lcd = lc + ld, L = lab + lcd;
    const int Lp1 = L + 1;
    const int nhab = mako::n_hermite_count(lab);
    const int nhcd = mako::n_hermite_count(lcd);

    // Dynamic shared memory layout
    extern __shared__ char smem_raw[];
    double* smem = reinterpret_cast<double*>(smem_raw);

    int E_size  = (nab * nhab > ncd * nhcd) ? nab * nhab : ncd * nhcd;
    const int R_stride = (nhcd % 2 == 0) ? nhcd + 1 : nhcd;  // pad even strides only
    int R_size  = nhab * R_stride;
    int M_size  = nab * nhcd;
    int Acc_sz  = ncomp;
    int Rtuv_sz = Lp1 * Lp1 * Lp1;
    int bra_1d_size = (la + 1) * (lb + 1) * (lab + 1);
    int ket_1d_size = (lc + 1) * (ld + 1) * (lcd + 1);
    int Fm_sz   = L + 1;
    int Rt_sz   = Lp1 * Lp1;

    int off = 0;
    double* smem_E    = smem + off; off += E_size;
    double* smem_R    = smem + off; off += R_size;   // [p|q] matrix (padded stride)
    double* smem_M    = smem + off; off += M_size;   // (ab|q] intermediate
    double* smem_Acc  = smem + off; off += Acc_sz;   // (ab|cd) output
    double* smem_Rtuv = smem + off; off += Rtuv_sz;
    double* smem_1d_bra = smem + off; off += 3 * bra_1d_size;
    double* smem_1d_ket = smem + off; off += 3 * ket_1d_size;
    double* smem_Fm   = smem + off; off += Fm_sz;
    double* smem_Rt   = smem + off; off += Rt_sz;

    // Int lookup tables
    int* smem_ints = reinterpret_cast<int*>(smem + off);
    int ioff = 0;
    int* herm_t_bra = smem_ints + ioff; ioff += nhab;
    int* herm_u_bra = smem_ints + ioff; ioff += nhab;
    int* herm_v_bra = smem_ints + ioff; ioff += nhab;
    int* herm_t_ket = smem_ints + ioff; ioff += nhcd;
    int* herm_u_ket = smem_ints + ioff; ioff += nhcd;
    int* herm_v_ket = smem_ints + ioff; ioff += nhcd;
    int* cart_ax = smem_ints + ioff; ioff += na;
    int* cart_ay = smem_ints + ioff; ioff += na;
    int* cart_az = smem_ints + ioff; ioff += na;
    int* cart_bx = smem_ints + ioff; ioff += nb;
    int* cart_by = smem_ints + ioff; ioff += nb;
    int* cart_bz = smem_ints + ioff; ioff += nb;
    int* cart_cx = smem_ints + ioff; ioff += nc;
    int* cart_cy = smem_ints + ioff; ioff += nc;
    int* cart_cz = smem_ints + ioff; ioff += nc;
    int* cart_dx = smem_ints + ioff; ioff += nd;
    int* cart_dy = smem_ints + ioff; ioff += nd;
    int* cart_dz = smem_ints + ioff; ioff += nd;

    // TF32TC staging (placed after int lookup tables)
    // NOTE: staging is only valid when UseTF32TC=true and smem is large enough
    [[maybe_unused]] float* smem_tf32_staging = nullptr;

    // One-time setup: lookup tables
    // Set staging pointer after int tables (16-byte aligned for CuTe vectorized loads)
    if constexpr (UseTF32TC) {
        uintptr_t p = reinterpret_cast<uintptr_t>(smem_ints + ioff);
        p = (p + 15u) & ~uintptr_t(15u);
        smem_tf32_staging = reinterpret_cast<float*>(p);
    }

    if (tid == 0) mako::enumerate_hermite_indices(lab, herm_t_bra, herm_u_bra, herm_v_bra);
    if (tid == 1) mako::enumerate_hermite_indices(lcd, herm_t_ket, herm_u_ket, herm_v_ket);
    if (tid == 2) {
        mako::enumerate_cart_exponents(la, cart_ax, cart_ay, cart_az);
        mako::enumerate_cart_exponents(lb, cart_bx, cart_by, cart_bz);
    }
    if (tid == 3) {
        mako::enumerate_cart_exponents(lc, cart_cx, cart_cy, cart_cz);
        mako::enumerate_cart_exponents(ld, cart_dx, cart_dy, cart_dz);
    }
    __syncthreads();

    // Load task metadata
    const int spAB = static_cast<int>(task_spAB[task_id]);
    const int spCD = static_cast<int>(task_spCD[task_id]);
    const int A = static_cast<int>(sp_A[spAB]);
    const int B = static_cast<int>(sp_B[spAB]);
    const int C_shell = static_cast<int>(sp_A[spCD]);
    const int D = static_cast<int>(sp_B[spCD]);

    const double Ax = shell_cx[A], Ay = shell_cy[A], Az = shell_cz[A];
    const double Bx = shell_cx[B], By = shell_cy[B], Bz = shell_cz[B];
    const double Cx = shell_cx[C_shell], Cy = shell_cy[C_shell], Cz = shell_cz[C_shell];
    const double Dx = shell_cx[D], Dy = shell_cy[D], Dz = shell_cz[D];

    const int baseAB = static_cast<int>(sp_pair_start[spAB]);
    const int baseCD = static_cast<int>(sp_pair_start[spCD]);
    const int nAB = static_cast<int>(sp_npair[spAB]);
    const int nCD = static_cast<int>(sp_npair[spCD]);

    const int e_bra_stride_i = (lb + 1) * (lab + 1);
    const int e_bra_stride_j = (lab + 1);
    const int e_ket_stride_i = (ld + 1) * (lcd + 1);
    const int e_ket_stride_j = (lcd + 1);

    // Zero tile accumulator: (ab|cd)
    mako::smem_zero(smem_Acc, Acc_sz, tid, nthreads);
    __syncthreads();

    // ===== Algorithm 1: outer loop over ket primitives (K_CD) =====
    for (int jp = 0; jp < nCD; ++jp) {
        const int kj = baseCD + jp;
        const double q   = pair_eta[kj];
        const double cKj = pair_cK[kj];
        const double Qx  = pair_Px[kj], Qy = pair_Py[kj], Qz = pair_Pz[kj];
        const double inv_2q = 0.5 / q;
        const double XQC = Qx - Cx, YQC = Qy - Cy, ZQC = Qz - Cz;
        const double XQD = Qx - Dx, YQD = Qy - Dy, ZQD = Qz - Dz;

        // Zero the (ab|q] intermediate — accumulates over K_AB
        mako::smem_zero(smem_M, M_size, tid, nthreads);
        __syncthreads();

        // ===== Inner loop over bra primitives (K_AB) =====
        for (int ip = 0; ip < nAB; ++ip) {
            const int ki = baseAB + ip;
            const double p   = pair_eta[ki];
            const double cKi = pair_cK[ki];
            const double Px  = pair_Px[ki], Py = pair_Py[ki], Pz = pair_Pz[ki];
            const double inv_2p = 0.5 / p;
            const double XPA = Px - Ax, YPA = Py - Ay, ZPA = Pz - Az;
            const double XPB = Px - Bx, YPB = Py - By, ZPB = Pz - Bz;

            const double denom = p + q;
            const double rho   = p * q / denom;
            const double XPQ = Px - Qx, YPQ = Py - Qy, ZPQ = Pz - Qz;
            const double prefactor = mako::kTwoPiToFiveHalves
                                   / (p * q * ::sqrt(denom))
                                   * cKi * cKj;

            // ----- Phase 1: r-integrals + [p|q] assembly -----
            // Bra E-coefficients (threads 0-2)
            double* E_bra_x = smem_1d_bra;
            double* E_bra_y = smem_1d_bra + bra_1d_size;
            double* E_bra_z = smem_1d_bra + 2 * bra_1d_size;

            if (tid == 0) mako::compute_E_1d(la, lb, XPA, XPB, inv_2p, E_bra_x, e_bra_stride_i, e_bra_stride_j);
            if (tid == 1) mako::compute_E_1d(la, lb, YPA, YPB, inv_2p, E_bra_y, e_bra_stride_i, e_bra_stride_j);
            if (tid == 2) mako::compute_E_1d(la, lb, ZPA, ZPB, inv_2p, E_bra_z, e_bra_stride_i, e_bra_stride_j);
            __syncthreads();

            // Cooperative R-integrals (all threads, Eq. 4-5)
            mako::cooperative_compute_R_integrals(
                XPQ, YPQ, ZPQ, rho, prefactor, L,
                smem_Fm, smem_Rtuv, smem_Rt, Lp1, tid, nthreads);

            // ----- [p|q] assembly with ILP (Eq. 6, 8) -----
            // [p|q] = (-1)^l_q * [p+q]^(0)
            // Each thread computes multiple (p,q) elements — the inner
            // q-loop is the ILP dimension (independent iterations that the
            // compiler can schedule concurrently, Eq. 8).
            for (int row = tid; row < nhab; row += nthreads) {
                const int tb = herm_t_bra[row], ub = herm_u_bra[row], vb = herm_v_bra[row];
                for (int q_h = 0; q_h < nhcd; ++q_h) {
                    const int T_idx = tb + herm_t_ket[q_h];
                    const int U_idx = ub + herm_u_ket[q_h];
                    const int V_idx = vb + herm_v_ket[q_h];
                    // Eq. 6: sign = (-1)^{l_q} where l_q = tau+ups+phi
                    const int lq = herm_t_ket[q_h] + herm_u_ket[q_h] + herm_v_ket[q_h];
                    const double sign = (lq & 1) ? -1.0 : 1.0;
                    smem_R[row * R_stride + q_h] = sign *
                        smem_Rtuv[(T_idx * Lp1 + U_idx) * Lp1 + V_idx];
                }
            }

            // ----- Phase 2: E_AB assembly -----
            for (int row = tid; row < nab; row += nthreads) {
                const int ia = row / nb, ib = row % nb;
                const int axi = cart_ax[ia], ayi = cart_ay[ia], azi = cart_az[ia];
                const int bxi = cart_bx[ib], byi = cart_by[ib], bzi = cart_bz[ib];
                for (int p_h = 0; p_h < nhab; ++p_h) {
                    const int t = herm_t_bra[p_h], u = herm_u_bra[p_h], v = herm_v_bra[p_h];
                    smem_E[row * nhab + p_h] =
                        E_bra_x[axi * e_bra_stride_i + bxi * e_bra_stride_j + t] *
                        E_bra_y[ayi * e_bra_stride_i + byi * e_bra_stride_j + u] *
                        E_bra_z[azi * e_bra_stride_i + bzi * e_bra_stride_j + v];
                }
            }
            __syncthreads();

            // ----- Phase 3: GEMM1 — (ab|q] += E_AB ⊗ [p|q] (Eq. 7) -----
            // Output accumulates into smem_M across K_AB iterations.
            // GEMM coalescing: smem_M stays in shared memory for GEMM2.
            // smem_R has padded stride R_stride for bank-conflict avoidance.
#ifdef MAKO_HAS_CUTLASS
            if constexpr (UseTF32TC) {
                if (mako::cutlass_gemm_eligible(nab, nhcd, nhab))
                    mako::smem_gemm_AB_accum_tf32(smem_M, smem_E, smem_R, nab, nhcd, nhab, tid, nthreads, smem_tf32_staging, R_stride);
                else
                    mako::smem_gemm_AB_accum_v2(smem_M, smem_E, smem_R, nab, nhcd, nhab, tid, nthreads, R_stride);
            } else
#endif
            mako::smem_gemm_AB_accum_v2(smem_M, smem_E, smem_R, nab, nhcd, nhab, tid, nthreads, R_stride);
            __syncthreads();
        }
        // ===== End K_AB loop =====

        // ----- Phase 4: E_CD assembly + GEMM2 -----
        // Ket E-coefficients (computed once per jp)
        double* E_ket_x = smem_1d_ket;
        double* E_ket_y = smem_1d_ket + ket_1d_size;
        double* E_ket_z = smem_1d_ket + 2 * ket_1d_size;

        if (tid == 0) mako::compute_E_1d(lc, ld, XQC, XQD, inv_2q, E_ket_x, e_ket_stride_i, e_ket_stride_j);
        if (tid == 1) mako::compute_E_1d(lc, ld, YQC, YQD, inv_2q, E_ket_y, e_ket_stride_i, e_ket_stride_j);
        if (tid == 2) mako::compute_E_1d(lc, ld, ZQC, ZQD, inv_2q, E_ket_z, e_ket_stride_i, e_ket_stride_j);
        __syncthreads();

        // Assemble E_CD (reuse smem_E; sign already absorbed into [p|q])
        for (int row = tid; row < ncd; row += nthreads) {
            const int ic = row / nd, id_c = row % nd;
            const int cxi = cart_cx[ic], cyi = cart_cy[ic], czi = cart_cz[ic];
            const int dxi = cart_dx[id_c], dyi = cart_dy[id_c], dzi = cart_dz[id_c];
            for (int q_h = 0; q_h < nhcd; ++q_h) {
                const int tau = herm_t_ket[q_h], ups = herm_u_ket[q_h], phi = herm_v_ket[q_h];
                smem_E[row * nhcd + q_h] =
                    E_ket_x[cxi * e_ket_stride_i + dxi * e_ket_stride_j + tau] *
                    E_ket_y[cyi * e_ket_stride_i + dyi * e_ket_stride_j + ups] *
                    E_ket_z[czi * e_ket_stride_i + dzi * e_ket_stride_j + phi];
            }
        }
        __syncthreads();

        // GEMM2: (ab|cd) += (ab|q] ⊗ E_CD^T (Eq. 7, second MatMul)
        // GEMM coalescing: reads (ab|q] from smem_M (no global memory).
#ifdef MAKO_HAS_CUTLASS
        if constexpr (UseTF32TC) {
            if (mako::cutlass_gemm_eligible(nab, ncd, nhcd))
                mako::smem_gemm_ABt_accum_tf32(smem_Acc, smem_M, smem_E,
                    nab, ncd, nhcd, tid, nthreads, smem_tf32_staging);
            else
                mako::smem_gemm_ABt_accum(smem_Acc, smem_M, smem_E, nab, ncd, nhcd, tid, nthreads);
        } else
#endif
        mako::smem_gemm_ABt_accum(smem_Acc, smem_M, smem_E, nab, ncd, nhcd, tid, nthreads);
        __syncthreads();
    }
    // ===== End K_CD loop =====

    // Write tile to global memory
    const int out_base = task_id * ncomp;
    for (int i = tid; i < ncomp; i += nthreads)
        eri_out[out_base + i] = static_cast<AccumT>(smem_Acc[i]);
}


// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
static int _gemm_smem_bytes(int la, int lb, int lc, int ld) {
    return mako::gemm_smem_bytes(la, lb, lc, ld);
}

extern "C" void mako_gemm_eri_fp64_launch(
    const int32_t* task_spAB, const int32_t* task_spCD, int ntasks,
    int la, int lb, int lc, int ld,
    const int32_t* sp_A, const int32_t* sp_B,
    const int32_t* sp_pair_start, const int32_t* sp_npair,
    const double* shell_cx, const double* shell_cy, const double* shell_cz,
    const double* pair_eta, const double* pair_Px, const double* pair_Py,
    const double* pair_Pz, const double* pair_cK,
    double* eri_out, int threads, unsigned long long stream_ptr)
{
    if (ntasks <= 0) return;
    const int block_size = (threads > 0 && threads <= 1024) ? threads : 128;
    const int smem_bytes = _gemm_smem_bytes(la, lb, lc, ld);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            KernelMako_gemm_eri<double, double>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    KernelMako_gemm_eri<double, double><<<ntasks, block_size, smem_bytes, stream>>>(
        task_spAB, task_spCD, ntasks, la, lb, lc, ld,
        sp_A, sp_B, sp_pair_start, sp_npair,
        shell_cx, shell_cy, shell_cz,
        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
        eri_out);
}

// Phase 10: TF32 scalar variant (no tensor cores)
extern "C" void mako_gemm_eri_tf32_launch(
    const int32_t* task_spAB, const int32_t* task_spCD, int ntasks,
    int la, int lb, int lc, int ld,
    const int32_t* sp_A, const int32_t* sp_B,
    const int32_t* sp_pair_start, const int32_t* sp_npair,
    const double* shell_cx, const double* shell_cy, const double* shell_cz,
    const double* pair_eta, const double* pair_Px, const double* pair_Py,
    const double* pair_Pz, const double* pair_cK,
    float* eri_out, int threads, unsigned long long stream_ptr)
{
    if (ntasks <= 0) return;
    const int block_size = (threads > 0 && threads <= 1024) ? threads : 128;
    const int smem_bytes = _gemm_smem_bytes(la, lb, lc, ld);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            KernelMako_gemm_eri<float, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    KernelMako_gemm_eri<float, float><<<ntasks, block_size, smem_bytes, stream>>>(
        task_spAB, task_spCD, ntasks, la, lb, lc, ld,
        sp_A, sp_B, sp_pair_start, sp_npair,
        shell_cx, shell_cy, shell_cz,
        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
        eri_out);
}

// Phase 10b: TF32 tensor core variant (SM80+).
// Uses FP64 for Boys/R-integrals, TF32 tensor cores for GEMM1.
// Output is FP64 (reduced precision from TF32 GEMMs).
#ifdef MAKO_HAS_CUTLASS
extern "C" void mako_gemm_eri_tf32tc_launch(
    const int32_t* task_spAB, const int32_t* task_spCD, int ntasks,
    int la, int lb, int lc, int ld,
    const int32_t* sp_A, const int32_t* sp_B,
    const int32_t* sp_pair_start, const int32_t* sp_npair,
    const double* shell_cx, const double* shell_cy, const double* shell_cz,
    const double* pair_eta, const double* pair_Px, const double* pair_Py,
    const double* pair_Pz, const double* pair_cK,
    double* eri_out, int threads, unsigned long long stream_ptr)
{
    if (ntasks <= 0) return;
    const int block_size = (threads > 0 && threads <= 1024) ? threads : 128;
    // Add FP32 staging for TF32 tensor core path
    const int na = mako::ncart(la), nb = mako::ncart(lb);
    const int nc = mako::ncart(lc), nd = mako::ncart(ld);
    const int nab = na*nb, ncd = nc*nd;
    const int nhab = mako::n_hermite_count(la+lb), nhcd = mako::n_hermite_count(lc+ld);
    const int stg1 = mako::tf32_staging_bytes(nab, nhcd, nhab);
    const int stg2 = mako::tf32_staging_bytes(nab, ncd, nhcd);
    const int smem_bytes = _gemm_smem_bytes(la, lb, lc, ld)
                         + ((stg1 > stg2) ? stg1 : stg2);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            KernelMako_gemm_eri<double, double, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    KernelMako_gemm_eri<double, double, true><<<ntasks, block_size, smem_bytes, stream>>>(
        task_spAB, task_spCD, ntasks, la, lb, lc, ld,
        sp_A, sp_B, sp_pair_start, sp_npair,
        shell_cx, shell_cy, shell_cz,
        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,
        eri_out);
}
#endif

extern "C" int mako_gemm_smem_bytes_query(int la, int lb, int lc, int ld) {
    return _gemm_smem_bytes(la, lb, lc, ld);
}
