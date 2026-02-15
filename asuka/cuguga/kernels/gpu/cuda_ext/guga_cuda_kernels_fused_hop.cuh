// Fused Warp-Cooperative Hop Kernel for CAS(14,14) Sub-10s
// Fuses W accumulation + one-body h_eff + ERI contraction + second E_pq apply.
// Eliminates CSR materialization and dense GEMMs.
//
// Thread mapping: 1 thread per (p,q) pair (strided), 1 block per CSF j.
// Each thread does serial DFS walks for its assigned (p,q) pairs:
//   1) build W[j,pq] and one-body local contribution
//   2) apply E_pq with g[j,pq]=0.5*(W@ERI)[j,pq]

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations from guga_cuda_kernels_epq.cuh
// (already included via guga_cuda_kernels.cu compilation unit)

template <int MAX_NORB_T, typename T>
__global__ __launch_bounds__(128, 4)
void guga_fused_hop_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf, norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf, norb+1]
    int ncsf,
    int norb,
    int j_start,                              // tile start CSF index
    int j_count,                              // number of CSFs in this tile
    const T* __restrict__ x,                  // [ncsf]
    const T* __restrict__ eri_mat,            // [nops, nops] row-major
    const T* __restrict__ h_eff_flat,         // [nops] flattened h_eff
    T* __restrict__ y,                        // [ncsf] output (atomicAdd)
    int* __restrict__ overflow_flag)
{
    guga_maybe_enable_smem_spilling();

    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;

    int warp_id = (int)threadIdx.x / kWarpSize;
    int lane_id = (int)threadIdx.x % kWarpSize;

    // Each warp processes one CSF j. Keep all warps alive until the shared h_eff
    // preload completes; partial tail blocks can otherwise leave sh_h_eff entries
    // uninitialized when some warps exit early.
    int j_local = (int)blockIdx.x * kWarpsPerBlock + warp_id;
    bool warp_in_tile = (j_local < j_count);
    int csf_j = j_start + j_local;
    bool warp_in_range = warp_in_tile && ((unsigned)csf_j < (unsigned)ncsf);

    int nops = norb * norb;
    if (norb > MAX_NORB_T) {
        if (lane_id == 0) atomicExch(overflow_flag, 1);
        return;
    }

    // Shared memory: per-warp CSF state + W accumulation buffer
    __shared__ int8_t sh_steps[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_nodes[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ int8_t sh_occ[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int16_t sh_b[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_idx_prefix[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ T sh_W[kWarpsPerBlock][MAX_NORB_T * MAX_NORB_T];
    __shared__ T sh_h_eff[MAX_NORB_T * MAX_NORB_T];

    // Cooperative load of h_eff into shared memory (once per block)
    for (int i = (int)threadIdx.x; i < nops; i += (int)blockDim.x) {
        sh_h_eff[i] = h_eff_flat[i];
    }

    __syncthreads();

    if (!warp_in_range) return;

    // Load CSF state for this warp's CSF j
    for (int k = lane_id; k < norb; k += kWarpSize) {
        int8_t step = steps_table[(int64_t)csf_j * (int64_t)norb + (int64_t)k];
        sh_steps[warp_id][k] = step;
        sh_occ[warp_id][k] = (int8_t)step_to_occ(step);
        int32_t node_next = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1) + (int64_t)(k + 1)];
        sh_nodes[warp_id][k + 1] = node_next;
        sh_b[warp_id][k] = node_twos[node_next];
    }
    __syncwarp();
    if (lane_id == 0) {
        sh_nodes[warp_id][0] = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1)];
    }
    __syncwarp();

    // Build idx_prefix via warp-wide inclusive scan
    {
        int32_t delta = 0;
        if (lane_id < norb) {
            int node_k = sh_nodes[warp_id][lane_id];
            int step_k = (int)sh_steps[warp_id][lane_id];
            delta = (int32_t)child_prefix[node_k * 5 + step_k];
        }
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
            if (lane_id >= off) delta += v;
        }
        if (lane_id == 0) sh_idx_prefix[warp_id][0] = 0;
        if (lane_id < norb) sh_idx_prefix[warp_id][lane_id + 1] = delta;
    }
    // For norb > 32, need second warp pass (rare for CAS14)
    if (norb > 32) {
        // Serial fallback for norb > 32
        if (lane_id == 0) {
            sh_idx_prefix[warp_id][0] = 0;
            int32_t acc = 0;
            for (int k = 0; k < norb; k++) {
                int node_k = sh_nodes[warp_id][k];
                int step_k = (int)sh_steps[warp_id][k];
                acc += (int32_t)child_prefix[node_k * 5 + step_k];
                sh_idx_prefix[warp_id][k + 1] = acc;
            }
        }
    }
    __syncwarp();

    // Zero W shared buffer for this warp
    for (int i = lane_id; i < nops; i += kWarpSize) {
        sh_W[warp_id][i] = (T)0;
    }
    __syncwarp();

    // Set diagonal W values: W[pp] = occ_p * x[j]
    // This ensures the ERI contraction includes E_rr * E_ss two-body terms.
    {
        T xj = __ldg(&x[csf_j]);
        for (int p = lane_id; p < norb; p += kWarpSize) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) {
                sh_W[warp_id][p * norb + p] = (T)occ_p * xj;
            }
        }
    }
    __syncwarp();

    // Phase 1: DFS walk for each (p,q) pair assigned to this lane.
    // To evaluate (E_pq x)[j] without destination atomics, walk the adjoint path E_qp on |j>:
    //   (E_pq x)[j] = sum_k <j|E_pq|k> x[k] = sum_k <k|E_qp|j> x[k].
    // We store this as W[j,pq] and reuse it for both one-body and two-body build.
    T y_one_body = (T)0;
    T y_diag = (T)0;

    // Diagonal (p==q) one-body contribution: h_eff[p,p] * occ[p] * x[j]
    if (lane_id == 0) {
        T xj = __ldg(&x[csf_j]);
        for (int p = 0; p < norb; p++) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (!occ_p) continue;
            T h_pp = sh_h_eff[p * norb + p];
            if (h_pp != (T)0) y_diag += h_pp * (T)occ_p * xj;
        }
    }

    // Off-diagonal (p!=q) DFS walks
    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) continue;

        // Adjoint walk: apply E_qp to |j>.
        int p_act = q;
        int q_act = p;
        int occ_p_act = (int)sh_occ[warp_id][p_act];
        int occ_q_act = (int)sh_occ[warp_id][q_act];
        if (occ_q_act <= 0 || occ_p_act >= 2) continue;

        T h_pq = sh_h_eff[pq];

        int start, end, q_start, q_mid, q_end;
        if (p_act < q_act) {
            start = p_act; end = q_act;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q_act; end = p_act;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start = sh_nodes[warp_id][start];
        int32_t node_end_target = sh_nodes[warp_id][end + 1];
        int32_t prefix_offset = sh_idx_prefix[warp_id][start];
        int32_t prefix_endplus1 = sh_idx_prefix[warp_id][end + 1];
        int32_t suffix_offset = (int32_t)csf_j - prefix_endplus1;

        // DFS stack in registers (same as apply_g_flat)
        int8_t st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        T st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = (T)1;
        top++;

        T W_pq = (T)0;

        while (top) {
            top--;
            T w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)sh_steps[warp_id][kpos];
            int bk = (int)sh_b[warp_id][kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == (T)0) continue;
                T w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_k = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_k >= (unsigned)ncsf) {
                        overflow = 1; continue;
                    }
                    if (w2 == (T)0) continue;
                    // Fused accumulation: W and one-body
                    // For p!=q, csf_k != csf_j always holds
                    T xk = __ldg(&x[csf_k]);
                    W_pq += w2 * xk;
                    y_one_body += h_pq * w2 * xk;
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) | ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }

        if (overflow) atomicExch(overflow_flag, 1);

        // Store W_pq into shared memory for ERI contraction
        sh_W[warp_id][pq] = W_pq;
    }
    __syncwarp();

    // Phase 2: full two-body apply.
    // Build g[j,pq] = 0.5 * sum_rs W[j,rs] * ERI[rs,pq], then apply:
    //   y[k] += sum_j sum_pq E_pq[k,j] * g[j,pq]
    T y_two_diag = (T)0;
    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        T g_pq = (T)0;
        {
            int rs = 0;
            for (; rs + 3 < nops; rs += 4) {
                g_pq += sh_W[warp_id][rs]   * __ldg(&eri_mat[rs       * nops + pq])
                      + sh_W[warp_id][rs+1] * __ldg(&eri_mat[(rs + 1) * nops + pq])
                      + sh_W[warp_id][rs+2] * __ldg(&eri_mat[(rs + 2) * nops + pq])
                      + sh_W[warp_id][rs+3] * __ldg(&eri_mat[(rs + 3) * nops + pq]);
            }
            for (; rs < nops; rs++) {
                g_pq += sh_W[warp_id][rs] * __ldg(&eri_mat[rs * nops + pq]);
            }
        }
        g_pq *= (T)0.5;
        if (g_pq == (T)0) continue;

        int p = pq / norb;
        int q = pq - p * norb;

        if (p == q) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) y_two_diag += g_pq * (T)occ_p;
            continue;
        }

        int occ_p = (int)sh_occ[warp_id][p];
        int occ_q = (int)sh_occ[warp_id][q];
        if (occ_q <= 0 || occ_p >= 2) continue;

        int start, end, q_start, q_mid, q_end;
        if (p < q) {
            start = p; end = q;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q; end = p;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start = sh_nodes[warp_id][start];
        int32_t node_end_target = sh_nodes[warp_id][end + 1];
        int32_t prefix_offset = sh_idx_prefix[warp_id][start];
        int32_t prefix_endplus1 = sh_idx_prefix[warp_id][end + 1];
        int32_t suffix_offset = (int32_t)csf_j - prefix_endplus1;

        int8_t st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        T st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = (T)1;
        top++;

        while (top) {
            top--;
            T w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)sh_steps[warp_id][kpos];
            int bk = (int)sh_b[warp_id][kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == (T)0) continue;
                T w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_k = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_k >= (unsigned)ncsf) {
                        overflow = 1; continue;
                    }
                    if (csf_k == csf_j) continue;
                    if (w2 == (T)0) continue;
                    atomicAdd(&y[csf_k], g_pq * w2);
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) | ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }

        if (overflow) atomicExch(overflow_flag, 1);
    }
    __syncwarp();

    // Warp-reduce local y[j] contributions.
    T y_total = y_one_body + y_diag + y_two_diag;
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        y_total += __shfl_xor_sync(0xffffffffu, y_total, off);
    }

    if (lane_id == 0 && y_total != (T)0) {
        atomicAdd(&y[csf_j], y_total);
    }
}

// =============================================================================
// Phase-1-only variant: W-build + ERI contraction, writes G to global memory.
// Phase 2 scatter is handled externally via the EPQ table scatter kernel.
// =============================================================================
template <int MAX_NORB_T, typename T>
__global__ __launch_bounds__(128, 4)
void guga_fused_hop_phase1_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* __restrict__ x,
    const T* __restrict__ eri_mat,
    const T* __restrict__ h_eff_flat,
    T* __restrict__ y,              // [ncsf] output for one-body + diagonal
    T* __restrict__ g_out,          // [j_count, nops] output for G tile
    int* __restrict__ overflow_flag)
{
    guga_maybe_enable_smem_spilling();

    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;

    int warp_id = (int)threadIdx.x / kWarpSize;
    int lane_id = (int)threadIdx.x % kWarpSize;

    int j_local = (int)blockIdx.x * kWarpsPerBlock + warp_id;
    bool warp_in_tile = (j_local < j_count);
    int csf_j = j_start + j_local;
    bool warp_in_range = warp_in_tile && ((unsigned)csf_j < (unsigned)ncsf);

    int nops = norb * norb;
    if (norb > MAX_NORB_T) {
        if (lane_id == 0) atomicExch(overflow_flag, 1);
        return;
    }

    __shared__ int8_t sh_steps[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_nodes[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ int8_t sh_occ[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int16_t sh_b[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_idx_prefix[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ T sh_W[kWarpsPerBlock][MAX_NORB_T * MAX_NORB_T];
    __shared__ T sh_h_eff[MAX_NORB_T * MAX_NORB_T];

    for (int i = (int)threadIdx.x; i < nops; i += (int)blockDim.x) {
        sh_h_eff[i] = h_eff_flat[i];
    }
    __syncthreads();

    if (!warp_in_range) return;

    // Load CSF state
    for (int k = lane_id; k < norb; k += kWarpSize) {
        int8_t step = steps_table[(int64_t)csf_j * (int64_t)norb + (int64_t)k];
        sh_steps[warp_id][k] = step;
        sh_occ[warp_id][k] = (int8_t)step_to_occ(step);
        int32_t node_next = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1) + (int64_t)(k + 1)];
        sh_nodes[warp_id][k + 1] = node_next;
        sh_b[warp_id][k] = node_twos[node_next];
    }
    __syncwarp();
    if (lane_id == 0) {
        sh_nodes[warp_id][0] = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1)];
    }
    __syncwarp();

    // idx_prefix scan
    {
        int32_t delta = 0;
        if (lane_id < norb) {
            int node_k = sh_nodes[warp_id][lane_id];
            int step_k = (int)sh_steps[warp_id][lane_id];
            delta = (int32_t)child_prefix[node_k * 5 + step_k];
        }
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
            if (lane_id >= off) delta += v;
        }
        if (lane_id == 0) sh_idx_prefix[warp_id][0] = 0;
        if (lane_id < norb) sh_idx_prefix[warp_id][lane_id + 1] = delta;
    }
    if (norb > 32) {
        if (lane_id == 0) {
            sh_idx_prefix[warp_id][0] = 0;
            int32_t acc = 0;
            for (int k = 0; k < norb; k++) {
                int node_k = sh_nodes[warp_id][k];
                int step_k = (int)sh_steps[warp_id][k];
                acc += (int32_t)child_prefix[node_k * 5 + step_k];
                sh_idx_prefix[warp_id][k + 1] = acc;
            }
        }
    }
    __syncwarp();

    // Zero W
    for (int i = lane_id; i < nops; i += kWarpSize) {
        sh_W[warp_id][i] = (T)0;
    }
    __syncwarp();

    // Diagonal W: W[pp] = occ_p * x[j]
    {
        T xj = __ldg(&x[csf_j]);
        for (int p = lane_id; p < norb; p += kWarpSize) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) {
                sh_W[warp_id][p * norb + p] = (T)occ_p * xj;
            }
        }
    }
    __syncwarp();

    // Phase 1: DFS walk for off-diagonal (p,q) pairs
    T y_one_body = (T)0;
    T y_diag = (T)0;

    if (lane_id == 0) {
        T xj = __ldg(&x[csf_j]);
        for (int p = 0; p < norb; p++) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (!occ_p) continue;
            T h_pp = sh_h_eff[p * norb + p];
            if (h_pp != (T)0) y_diag += h_pp * (T)occ_p * xj;
        }
    }

    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) continue;

        int p_act = q;
        int q_act = p;
        int occ_p_act = (int)sh_occ[warp_id][p_act];
        int occ_q_act = (int)sh_occ[warp_id][q_act];
        if (occ_q_act <= 0 || occ_p_act >= 2) continue;

        T h_pq = sh_h_eff[pq];

        int start, end, q_start, q_mid, q_end;
        if (p_act < q_act) {
            start = p_act; end = q_act;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q_act; end = p_act;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start = sh_nodes[warp_id][start];
        int32_t node_end_target = sh_nodes[warp_id][end + 1];
        int32_t prefix_offset = sh_idx_prefix[warp_id][start];
        int32_t prefix_endplus1 = sh_idx_prefix[warp_id][end + 1];
        int32_t suffix_offset = (int32_t)csf_j - prefix_endplus1;

        int8_t st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        T st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = (T)1;
        top++;

        T W_pq = (T)0;

        while (top) {
            top--;
            T w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)sh_steps[warp_id][kpos];
            int bk = (int)sh_b[warp_id][kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == (T)0) continue;
                T w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_k = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_k >= (unsigned)ncsf) {
                        overflow = 1; continue;
                    }
                    if (w2 == (T)0) continue;
                    T xk = __ldg(&x[csf_k]);
                    W_pq += w2 * xk;
                    y_one_body += h_pq * w2 * xk;
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) | ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }

        if (overflow) atomicExch(overflow_flag, 1);
        sh_W[warp_id][pq] = W_pq;
    }
    __syncwarp();

    // ERI contraction: g[pq] = 0.5 * sum_rs W[rs] * ERI[rs,pq]
    // Write G to g_out instead of doing Phase 2 DFS scatter.
    T y_two_diag = (T)0;
    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        T g_pq = (T)0;
        int rs = 0;
        for (; rs + 3 < nops; rs += 4) {
            g_pq += sh_W[warp_id][rs]   * __ldg(&eri_mat[rs       * nops + pq])
                  + sh_W[warp_id][rs+1] * __ldg(&eri_mat[(rs + 1) * nops + pq])
                  + sh_W[warp_id][rs+2] * __ldg(&eri_mat[(rs + 2) * nops + pq])
                  + sh_W[warp_id][rs+3] * __ldg(&eri_mat[(rs + 3) * nops + pq]);
        }
        for (; rs < nops; rs++) {
            g_pq += sh_W[warp_id][rs] * __ldg(&eri_mat[rs * nops + pq]);
        }
        g_pq *= (T)0.5;

        // Diagonal two-body: accumulate locally instead of scattering
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) y_two_diag += g_pq * (T)occ_p;
            // Zero diagonal in g_out — diagonal is handled by y_two_diag above,
            // and the EPQ scatter kernel must not double-count it.
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = (T)0;
        } else {
            // Write off-diagonal G tile for external EPQ scatter
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = g_pq;
        }
    }
    __syncwarp();

    // Warp-reduce one-body + diagonal contributions to y[j]
    T y_total = y_one_body + y_diag + y_two_diag;
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        y_total += __shfl_xor_sync(0xffffffffu, y_total, off);
    }
    if (lane_id == 0 && y_total != (T)0) {
        atomicAdd(&y[csf_j], y_total);
    }
}

// =============================================================================
// Phase-1-only + COO output variant: W-build + ERI contraction + COO connectivity.
// Writes (j_local, csf_k, pq, w2) to global COO buffer during Phase 1 DFS.
// Phase 2 scatter is handled by the COO scatter kernel above.
// =============================================================================
template <int MAX_NORB_T, typename T>
__global__ __launch_bounds__(128, 4)
void guga_fused_hop_phase1_coo_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* __restrict__ x,
    const T* __restrict__ eri_mat,
    const T* __restrict__ h_eff_flat,
    T* __restrict__ y,
    T* __restrict__ g_out,
    int* __restrict__ overflow_flag,
    // COO output buffers
    int* __restrict__ coo_nnz_counter,
    int32_t* __restrict__ coo_j_local,
    int32_t* __restrict__ coo_k,
    int16_t* __restrict__ coo_pq,
    T* __restrict__ coo_w2,
    int max_coo)
{
    guga_maybe_enable_smem_spilling();

    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;

    int warp_id = (int)threadIdx.x / kWarpSize;
    int lane_id = (int)threadIdx.x % kWarpSize;

    int j_local = (int)blockIdx.x * kWarpsPerBlock + warp_id;
    bool warp_in_tile = (j_local < j_count);
    int csf_j = j_start + j_local;
    bool warp_in_range = warp_in_tile && ((unsigned)csf_j < (unsigned)ncsf);

    int nops = norb * norb;
    if (norb > MAX_NORB_T) {
        if (lane_id == 0) atomicExch(overflow_flag, 1);
        return;
    }

    __shared__ int8_t sh_steps[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_nodes[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ int8_t sh_occ[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int16_t sh_b[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_idx_prefix[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ T sh_W[kWarpsPerBlock][MAX_NORB_T * MAX_NORB_T];
    __shared__ T sh_h_eff[MAX_NORB_T * MAX_NORB_T];

    for (int i = (int)threadIdx.x; i < nops; i += (int)blockDim.x) {
        sh_h_eff[i] = h_eff_flat[i];
    }
    __syncthreads();

    if (!warp_in_range) return;

    // Load CSF state
    for (int k = lane_id; k < norb; k += kWarpSize) {
        int8_t step = steps_table[(int64_t)csf_j * (int64_t)norb + (int64_t)k];
        sh_steps[warp_id][k] = step;
        sh_occ[warp_id][k] = (int8_t)step_to_occ(step);
        int32_t node_next = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1) + (int64_t)(k + 1)];
        sh_nodes[warp_id][k + 1] = node_next;
        sh_b[warp_id][k] = node_twos[node_next];
    }
    __syncwarp();
    if (lane_id == 0) {
        sh_nodes[warp_id][0] = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1)];
    }
    __syncwarp();

    // idx_prefix scan
    {
        int32_t delta = 0;
        if (lane_id < norb) {
            int node_k = sh_nodes[warp_id][lane_id];
            int step_k = (int)sh_steps[warp_id][lane_id];
            delta = (int32_t)child_prefix[node_k * 5 + step_k];
        }
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
            if (lane_id >= off) delta += v;
        }
        if (lane_id == 0) sh_idx_prefix[warp_id][0] = 0;
        if (lane_id < norb) sh_idx_prefix[warp_id][lane_id + 1] = delta;
    }
    if (norb > 32) {
        if (lane_id == 0) {
            sh_idx_prefix[warp_id][0] = 0;
            int32_t acc = 0;
            for (int k = 0; k < norb; k++) {
                int node_k = sh_nodes[warp_id][k];
                int step_k = (int)sh_steps[warp_id][k];
                acc += (int32_t)child_prefix[node_k * 5 + step_k];
                sh_idx_prefix[warp_id][k + 1] = acc;
            }
        }
    }
    __syncwarp();

    // Zero W
    for (int i = lane_id; i < nops; i += kWarpSize) {
        sh_W[warp_id][i] = (T)0;
    }
    __syncwarp();

    // Diagonal W: W[pp] = occ_p * x[j]
    {
        T xj = __ldg(&x[csf_j]);
        for (int p = lane_id; p < norb; p += kWarpSize) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) {
                sh_W[warp_id][p * norb + p] = (T)occ_p * xj;
            }
        }
    }
    __syncwarp();

    // Phase 1: DFS walk for off-diagonal (p,q) pairs + COO output
    T y_one_body = (T)0;
    T y_diag = (T)0;

    if (lane_id == 0) {
        T xj = __ldg(&x[csf_j]);
        for (int p = 0; p < norb; p++) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (!occ_p) continue;
            T h_pp = sh_h_eff[p * norb + p];
            if (h_pp != (T)0) y_diag += h_pp * (T)occ_p * xj;
        }
    }

    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) continue;

        int p_act = q;
        int q_act = p;
        int occ_p_act = (int)sh_occ[warp_id][p_act];
        int occ_q_act = (int)sh_occ[warp_id][q_act];
        if (occ_q_act <= 0 || occ_p_act >= 2) continue;

        T h_pq = sh_h_eff[pq];

        int start, end, q_start, q_mid, q_end;
        if (p_act < q_act) {
            start = p_act; end = q_act;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q_act; end = p_act;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start = sh_nodes[warp_id][start];
        int32_t node_end_target = sh_nodes[warp_id][end + 1];
        int32_t prefix_offset = sh_idx_prefix[warp_id][start];
        int32_t prefix_endplus1 = sh_idx_prefix[warp_id][end + 1];
        int32_t suffix_offset = (int32_t)csf_j - prefix_endplus1;

        int8_t st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        T st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = (T)1;
        top++;

        T W_pq = (T)0;

        while (top) {
            top--;
            T w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)sh_steps[warp_id][kpos];
            int bk = (int)sh_b[warp_id][kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == (T)0) continue;
                T w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_k = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_k >= (unsigned)ncsf) {
                        overflow = 1; continue;
                    }
                    if (w2 == (T)0) continue;
                    T xk = __ldg(&x[csf_k]);
                    W_pq += w2 * xk;
                    y_one_body += h_pq * w2 * xk;
                    // COO output: record (j_local, csf_k, pq, w2)
                    int slot = atomicAdd(coo_nnz_counter, 1);
                    if (slot < max_coo) {
                        coo_j_local[slot] = j_local;
                        coo_k[slot] = csf_k;
                        coo_pq[slot] = (int16_t)pq;
                        coo_w2[slot] = w2;
                    }
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) | ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }

        if (overflow) atomicExch(overflow_flag, 1);
        sh_W[warp_id][pq] = W_pq;
    }
    __syncwarp();

    // ERI contraction: g[pq] = 0.5 * sum_rs W[rs] * ERI[rs,pq]
    T y_two_diag = (T)0;
    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        T g_pq = (T)0;
        int rs = 0;
        for (; rs + 3 < nops; rs += 4) {
            g_pq += sh_W[warp_id][rs]   * __ldg(&eri_mat[rs       * nops + pq])
                  + sh_W[warp_id][rs+1] * __ldg(&eri_mat[(rs + 1) * nops + pq])
                  + sh_W[warp_id][rs+2] * __ldg(&eri_mat[(rs + 2) * nops + pq])
                  + sh_W[warp_id][rs+3] * __ldg(&eri_mat[(rs + 3) * nops + pq]);
        }
        for (; rs < nops; rs++) {
            g_pq += sh_W[warp_id][rs] * __ldg(&eri_mat[rs * nops + pq]);
        }
        g_pq *= (T)0.5;

        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) y_two_diag += g_pq * (T)occ_p;
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = (T)0;
        } else {
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = g_pq;
        }
    }
    __syncwarp();

    // Warp-reduce one-body + diagonal contributions to y[j]
    T y_total = y_one_body + y_diag + y_two_diag;
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        y_total += __shfl_xor_sync(0xffffffffu, y_total, off);
    }
    if (lane_id == 0 && y_total != (T)0) {
        atomicAdd(&y[csf_j], y_total);
    }
}

// =============================================================================
// Multi-Pair Merged DFS Kernel (WP14)
//
// Groups (p,q) pairs by (direction, start) and walks the shared DRT prefix
// once per group instead of once per pair. At peel-off levels where a pair
// ends, the q_end and q_mid candidate sets are disjoint, so we iterate over
// all 4 children and apply the correct segment value for each branch.
//
// Key insight: pq index, node_end_target, prefix/suffix offsets can all be
// recomputed from (direction, start, end) + shared CSF state, so per-end-slot
// data does NOT need shared memory — only the group header does.
//
// Theoretical speedup: ~5× fewer DFS steps for CAS(14,14), realistic 2-3×.
// =============================================================================

template <int MAX_NORB_T, typename T>
__global__ __launch_bounds__(128, 4)
void guga_fused_hop_phase1_coo_merged_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* __restrict__ x,
    const T* __restrict__ eri_mat,
    const T* __restrict__ h_eff_flat,
    T* __restrict__ y,
    T* __restrict__ g_out,
    int* __restrict__ overflow_flag,
    int* __restrict__ coo_nnz_counter,
    int32_t* __restrict__ coo_j_local,
    int32_t* __restrict__ coo_k,
    int16_t* __restrict__ coo_pq,
    T* __restrict__ coo_w2,
    int max_coo)
{
    guga_maybe_enable_smem_spilling();

    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;

    int warp_id = (int)threadIdx.x / kWarpSize;
    int lane_id = (int)threadIdx.x % kWarpSize;

    int j_local = (int)blockIdx.x * kWarpsPerBlock + warp_id;
    bool warp_in_tile = (j_local < j_count);
    int csf_j = j_start + j_local;
    bool warp_in_range = warp_in_tile && ((unsigned)csf_j < (unsigned)ncsf);

    int nops = norb * norb;
    if (norb > MAX_NORB_T) {
        if (lane_id == 0) atomicExch(overflow_flag, 1);
        return;
    }

    __shared__ int8_t sh_steps[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_nodes[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ int8_t sh_occ[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int16_t sh_b[kWarpsPerBlock][MAX_NORB_T];
    __shared__ int32_t sh_idx_prefix[kWarpsPerBlock][MAX_NORB_T + 1];
    __shared__ T sh_W[kWarpsPerBlock][MAX_NORB_T * MAX_NORB_T];
    __shared__ T sh_h_eff[MAX_NORB_T * MAX_NORB_T];

    // Group header: only 4 bytes per group × 64 groups × 4 warps = 1 KB
    __shared__ int sh_n_groups[kWarpsPerBlock];
    __shared__ int8_t sh_grp_start[kWarpsPerBlock][64];
    __shared__ int8_t sh_grp_max_end[kWarpsPerBlock][64];
    __shared__ int8_t sh_grp_dir[kWarpsPerBlock][64];
    __shared__ uint32_t sh_grp_end_mask[kWarpsPerBlock][64];

    // Cooperative load of h_eff
    for (int i = (int)threadIdx.x; i < nops; i += (int)blockDim.x) {
        sh_h_eff[i] = h_eff_flat[i];
    }
    __syncthreads();

    if (!warp_in_range) return;

    // Load CSF state
    for (int k = lane_id; k < norb; k += kWarpSize) {
        int8_t step = steps_table[(int64_t)csf_j * (int64_t)norb + (int64_t)k];
        sh_steps[warp_id][k] = step;
        sh_occ[warp_id][k] = (int8_t)step_to_occ(step);
        int32_t node_next = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1) + (int64_t)(k + 1)];
        sh_nodes[warp_id][k + 1] = node_next;
        sh_b[warp_id][k] = node_twos[node_next];
    }
    __syncwarp();
    if (lane_id == 0) {
        sh_nodes[warp_id][0] = nodes_table[(int64_t)csf_j * (int64_t)(norb + 1)];
    }
    __syncwarp();

    // idx_prefix scan
    {
        int32_t delta = 0;
        if (lane_id < norb) {
            int node_k = sh_nodes[warp_id][lane_id];
            int step_k = (int)sh_steps[warp_id][lane_id];
            delta = (int32_t)child_prefix[node_k * 5 + step_k];
        }
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
            if (lane_id >= off) delta += v;
        }
        if (lane_id == 0) sh_idx_prefix[warp_id][0] = 0;
        if (lane_id < norb) sh_idx_prefix[warp_id][lane_id + 1] = delta;
    }
    if (norb > 32) {
        if (lane_id == 0) {
            sh_idx_prefix[warp_id][0] = 0;
            int32_t acc = 0;
            for (int k = 0; k < norb; k++) {
                int node_k = sh_nodes[warp_id][k];
                int step_k = (int)sh_steps[warp_id][k];
                acc += (int32_t)child_prefix[node_k * 5 + step_k];
                sh_idx_prefix[warp_id][k + 1] = acc;
            }
        }
    }
    __syncwarp();

    // Zero W
    for (int i = lane_id; i < nops; i += kWarpSize) {
        sh_W[warp_id][i] = (T)0;
    }
    __syncwarp();

    // Diagonal W: W[pp] = occ_p * x[j]
    {
        T xj = __ldg(&x[csf_j]);
        for (int p = lane_id; p < norb; p += kWarpSize) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) {
                sh_W[warp_id][p * norb + p] = (T)occ_p * xj;
            }
        }
    }
    __syncwarp();

    // =========================================================================
    // Group construction (lane 0 builds groups in shared memory)
    // =========================================================================
    if (lane_id == 0) {
        // Group key: (direction, start). Use a simple linear scan.
        // grp_map[dir][start] = group index, -1 if not yet created
        int8_t grp_map[2][MAX_NORB_T];
        for (int d = 0; d < 2; d++)
            for (int s = 0; s < norb; s++)
                grp_map[d][s] = -1;

        int ng = 0;
        // Initialize all end_masks to 0
        for (int g = 0; g < 64; g++)
            sh_grp_end_mask[warp_id][g] = 0;

        for (int pq = 0; pq < nops; pq++) {
            int p = pq / norb;
            int q = pq - p * norb;
            if (p == q) continue;

            // Adjoint walk: apply E_qp to |j>
            int p_act = q;
            int q_act = p;
            int occ_p_act = (int)sh_occ[warp_id][p_act];
            int occ_q_act = (int)sh_occ[warp_id][q_act];
            if (occ_q_act <= 0 || occ_p_act >= 2) continue;

            int dir, start, end;
            if (p_act < q_act) {
                dir = 0; start = p_act; end = q_act;
            } else {
                dir = 1; start = q_act; end = p_act;
            }

            int gi = grp_map[dir][start];
            if (gi < 0) {
                gi = ng++;
                if (gi >= 64) { // overflow — shouldn't happen for norb<=20
                    atomicExch(overflow_flag, 1);
                    break;
                }
                grp_map[dir][start] = (int8_t)gi;
                sh_grp_start[warp_id][gi] = (int8_t)start;
                sh_grp_max_end[warp_id][gi] = (int8_t)end;
                sh_grp_dir[warp_id][gi] = (int8_t)dir;
            }
            // Update max_end
            if (end > (int)sh_grp_max_end[warp_id][gi])
                sh_grp_max_end[warp_id][gi] = (int8_t)end;

            // Set end_mask bit: bit (end - start - 1)
            int slot = end - start - 1;
            sh_grp_end_mask[warp_id][gi] |= (1u << slot);
        }
        sh_n_groups[warp_id] = ng;
    }
    __syncwarp();

    int n_groups = sh_n_groups[warp_id];

    // =========================================================================
    // Phase 1: Merged DFS walk + one-body + COO output
    // =========================================================================
    T y_one_body = (T)0;
    T y_diag = (T)0;

    // Diagonal one-body
    if (lane_id == 0) {
        T xj = __ldg(&x[csf_j]);
        for (int p = 0; p < norb; p++) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (!occ_p) continue;
            T h_pp = sh_h_eff[p * norb + p];
            if (h_pp != (T)0) y_diag += h_pp * (T)occ_p * xj;
        }
    }

    // Each lane processes one group (strided)
    for (int grp = lane_id; grp < n_groups; grp += kWarpSize) {
        int my_start = (int)sh_grp_start[warp_id][grp];
        int my_max_end = (int)sh_grp_max_end[warp_id][grp];
        int my_dir = (int)sh_grp_dir[warp_id][grp];
        uint32_t my_end_mask = sh_grp_end_mask[warp_id][grp];

        int q_start_type, q_mid_type, q_end_type;
        if (my_dir == 0) {
            q_start_type = Q_uR; q_mid_type = Q_R; q_end_type = Q_oR;
        } else {
            q_start_type = Q_uL; q_mid_type = Q_L; q_end_type = Q_oL;
        }

        int32_t node_start = sh_nodes[warp_id][my_start];

        // DFS stack — depth bounded by (max_end - start + 1)
        int8_t st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        T st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)my_start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0)
                         | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = (T)1;
        top++;

        while (top) {
            top--;
            T w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int dk = (int)sh_steps[warp_id][kpos];
            int bk = (int)sh_b[warp_id][kpos];
            int k_next = kpos + 1;

            // Check if any pair ends at this level (end = kpos)
            // slot = end - start - 1 = kpos - my_start - 1
            int slot = kpos - my_start - 1;
            bool is_peel = (kpos > my_start) && (slot >= 0)
                        && (my_end_mask & (1u << slot));
            bool is_last = (kpos == my_max_end);

            if (is_peel && !is_last) {
                // FORK: this level is q_end for one pair, q_mid for rest.
                // Compute pq/net/pfx/sfx for the ending pair from CSF state.
                int end_level = kpos;
                int pq_end;
                if (my_dir == 0) // raising: p=end, q=start
                    pq_end = end_level * norb + my_start;
                else // lowering: p=start, q=end
                    pq_end = my_start * norb + end_level;
                int32_t net_end = sh_nodes[warp_id][end_level + 1];
                int32_t pfx_end = sh_idx_prefix[warp_id][my_start];
                int32_t sfx_end = (int32_t)csf_j
                                - sh_idx_prefix[warp_id][end_level + 1];
                T h_pq_end = sh_h_eff[pq_end];

                // Process q_end candidates (terminate this pair)
                {
                    int dp0e = 0, dp1e = 0;
                    int ndpe = candidate_dprimes(q_end_type, dk, &dp0e, &dp1e);
                    for (int ww = 0; ww < ndpe; ww++) {
                        int dprime = (ww == 0) ? dp0e : dp1e;
                        int ck = __ldg(&child[node_k * 4 + dprime]);
                        if (ck < 0) continue;
                        if (ck != net_end) continue;
                        int bprime = (int)__ldg(&node_twos[ck]);
                        int db = bk - bprime;
                        T seg = (T)segment_value_int(
                            q_end_type, dprime, dk, db, bk);
                        if (seg == (T)0) continue;
                        T w2 = w * seg;
                        int32_t si2 = seg_idx
                            + (int32_t)__ldg(&child_prefix[node_k * 5 + dprime]);
                        int32_t csf_k = pfx_end + si2 + sfx_end;
                        if ((unsigned)csf_k >= (unsigned)ncsf) {
                            overflow = 1; continue;
                        }
                        T xk = __ldg(&x[csf_k]);
                        sh_W[warp_id][pq_end] += w2 * xk;
                        y_one_body += h_pq_end * w2 * xk;
                        int cs = atomicAdd(coo_nnz_counter, 1);
                        if (cs < max_coo) {
                            coo_j_local[cs] = j_local;
                            coo_k[cs] = csf_k;
                            coo_pq[cs] = (int16_t)pq_end;
                            coo_w2[cs] = w2;
                        }
                    }
                }
                // Process q_mid candidates (continue DFS)
                {
                    int dp0m = 0, dp1m = 0;
                    int ndpm = candidate_dprimes(q_mid_type, dk, &dp0m, &dp1m);
                    for (int ww = 0; ww < ndpm; ww++) {
                        int dprime = (ww == 0) ? dp0m : dp1m;
                        int ck = __ldg(&child[node_k * 4 + dprime]);
                        if (ck < 0) continue;
                        int bprime = (int)__ldg(&node_twos[ck]);
                        int db = bk - bprime;
                        T seg = (T)segment_value_int(
                            q_mid_type, dprime, dk, db, bk);
                        if (seg == (T)0) continue;
                        T w2 = w * seg;
                        int32_t si2 = seg_idx
                            + (int32_t)__ldg(&child_prefix[node_k * 5 + dprime]);
                        if (top >= MAX_NORB_T) { overflow = 1; continue; }
                        st_k[top] = (int8_t)k_next;
                        st_node_seg[top] = ((uint64_t)(uint32_t)si2)
                                         | ((uint64_t)(uint32_t)ck << 32);
                        st_w[top] = w2;
                        top++;
                    }
                }
            } else if (is_last) {
                // Last end level — terminate the last pair
                int pq_last;
                if (my_dir == 0)
                    pq_last = my_max_end * norb + my_start;
                else
                    pq_last = my_start * norb + my_max_end;
                int32_t net_last = sh_nodes[warp_id][my_max_end + 1];
                int32_t pfx_last = sh_idx_prefix[warp_id][my_start];
                int32_t sfx_last = (int32_t)csf_j
                                 - sh_idx_prefix[warp_id][my_max_end + 1];
                T h_pq_last = sh_h_eff[pq_last];

                int dp0e = 0, dp1e = 0;
                int ndpe = candidate_dprimes(q_end_type, dk, &dp0e, &dp1e);
                for (int ww = 0; ww < ndpe; ww++) {
                    int dprime = (ww == 0) ? dp0e : dp1e;
                    int ck = __ldg(&child[node_k * 4 + dprime]);
                    if (ck < 0) continue;
                    if (ck != net_last) continue;
                    int bprime = (int)__ldg(&node_twos[ck]);
                    int db = bk - bprime;
                    T seg = (T)segment_value_int(
                        q_end_type, dprime, dk, db, bk);
                    if (seg == (T)0) continue;
                    T w2 = w * seg;
                    int32_t si2 = seg_idx
                        + (int32_t)__ldg(&child_prefix[node_k * 5 + dprime]);
                    int32_t csf_k = pfx_last + si2 + sfx_last;
                    if ((unsigned)csf_k >= (unsigned)ncsf) {
                        overflow = 1; continue;
                    }
                    T xk = __ldg(&x[csf_k]);
                    sh_W[warp_id][pq_last] += w2 * xk;
                    y_one_body += h_pq_last * w2 * xk;
                    int cs = atomicAdd(coo_nnz_counter, 1);
                    if (cs < max_coo) {
                        coo_j_local[cs] = j_local;
                        coo_k[cs] = csf_k;
                        coo_pq[cs] = (int16_t)pq_last;
                        coo_w2[cs] = w2;
                    }
                }
            } else {
                // Normal level: start or middle, no peel-off
                int qk = (kpos == my_start) ? q_start_type : q_mid_type;
                int dp0 = 0, dp1 = 0;
                int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
                for (int ww = 0; ww < ndp; ww++) {
                    int dprime = (ww == 0) ? dp0 : dp1;
                    int ck = __ldg(&child[node_k * 4 + dprime]);
                    if (ck < 0) continue;
                    int bprime = (int)__ldg(&node_twos[ck]);
                    int db = bk - bprime;
                    T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
                    if (seg == (T)0) continue;
                    T w2 = w * seg;
                    int32_t si2 = seg_idx
                        + (int32_t)__ldg(&child_prefix[node_k * 5 + dprime]);
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)si2)
                                     | ((uint64_t)(uint32_t)ck << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        } // end while (top)

        if (overflow) atomicExch(overflow_flag, 1);
    } // end for (grp)
    __syncwarp();

    // ERI contraction: g[pq] = 0.5 * sum_rs W[rs] * ERI[rs,pq]
    T y_two_diag = (T)0;
    for (int pq = lane_id; pq < nops; pq += kWarpSize) {
        T g_pq = (T)0;
        int rs = 0;
        for (; rs + 3 < nops; rs += 4) {
            g_pq += sh_W[warp_id][rs]   * __ldg(&eri_mat[rs       * nops + pq])
                  + sh_W[warp_id][rs+1] * __ldg(&eri_mat[(rs + 1) * nops + pq])
                  + sh_W[warp_id][rs+2] * __ldg(&eri_mat[(rs + 2) * nops + pq])
                  + sh_W[warp_id][rs+3] * __ldg(&eri_mat[(rs + 3) * nops + pq]);
        }
        for (; rs < nops; rs++) {
            g_pq += sh_W[warp_id][rs] * __ldg(&eri_mat[rs * nops + pq]);
        }
        g_pq *= (T)0.5;

        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) {
            int occ_p = (int)sh_occ[warp_id][p];
            if (occ_p > 0) y_two_diag += g_pq * (T)occ_p;
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = (T)0;
        } else {
            g_out[(int64_t)j_local * (int64_t)nops + (int64_t)pq] = g_pq;
        }
    }
    __syncwarp();

    // Warp-reduce one-body + diagonal contributions to y[j]
    T y_total = y_one_body + y_diag + y_two_diag;
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        y_total += __shfl_xor_sync(0xffffffffu, y_total, off);
    }
    if (lane_id == 0 && y_total != (T)0) {
        atomicAdd(&y[csf_j], y_total);
    }
}

// =============================================================================
// COO Scatter Kernel: reads COO triples and scatters G into y.
// =============================================================================
template <typename T>
__global__ void guga_coo_scatter_kernel(
    const int32_t* __restrict__ coo_j_local,
    const int32_t* __restrict__ coo_k,
    const int16_t* __restrict__ coo_pq,
    const T* __restrict__ coo_w2,
    const T* __restrict__ g_tile,   // [j_count, nops]
    int nops,
    int nnz,
    T* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int j = coo_j_local[i];
    int k = coo_k[i];
    int pq = (int)coo_pq[i];
    T w = coo_w2[i];
    T g = __ldg(&g_tile[(int64_t)j * (int64_t)nops + (int64_t)pq]);
    atomicAdd(&y[k], g * w);
}

// COO scatter launch wrapper
template <typename T>
void guga_coo_scatter_launch_stream(
    const int32_t* coo_j_local,
    const int32_t* coo_k,
    const int16_t* coo_pq,
    const T* coo_w2,
    const T* g_tile,
    int nops,
    int nnz,
    T* y,
    cudaStream_t stream)
{
    if (nnz <= 0) return;
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    guga_coo_scatter_kernel<T><<<blocks, threads, 0, stream>>>(
        coo_j_local, coo_k, coo_pq, coo_w2,
        g_tile, nops, nnz, y);
}

template void guga_coo_scatter_launch_stream<float>(
    const int32_t*, const int32_t*, const int16_t*, const float*,
    const float*, int, int, float*, cudaStream_t);
template void guga_coo_scatter_launch_stream<double>(
    const int32_t*, const int32_t*, const int16_t*, const double*,
    const double*, int, int, double*, cudaStream_t);

// =============================================================================
// Launch wrappers
// =============================================================================
template <int MAX_NORB_T, typename T>
void guga_fused_hop_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* x,
    const T* eri_mat,
    const T* h_eff_flat,
    T* y,
    int* overflow_flag,
    cudaStream_t stream)
{
    if (j_count <= 0) return;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;
    int threads_per_block = kWarpsPerBlock * kWarpSize;  // 128
    int blocks = (j_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    guga_fused_hop_kernel_t<MAX_NORB_T, T><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix,
        steps_table, nodes_table,
        ncsf, norb,
        j_start, j_count,
        x, eri_mat, h_eff_flat,
        y, overflow_flag);
}

// Explicit instantiations
#define INSTANTIATE_FUSED_HOP(NORB, TYPE) \
    template void guga_fused_hop_launch_stream<NORB, TYPE>( \
        const int32_t*, const int16_t*, const int64_t*, \
        const int8_t*, const int32_t*, \
        int, int, int, int, \
        const TYPE*, const TYPE*, const TYPE*, \
        TYPE*, int*, cudaStream_t);

INSTANTIATE_FUSED_HOP(8, float)
INSTANTIATE_FUSED_HOP(8, double)
INSTANTIATE_FUSED_HOP(12, float)
INSTANTIATE_FUSED_HOP(12, double)
INSTANTIATE_FUSED_HOP(16, float)
INSTANTIATE_FUSED_HOP(16, double)
INSTANTIATE_FUSED_HOP(20, float)
INSTANTIATE_FUSED_HOP(20, double)

#undef INSTANTIATE_FUSED_HOP

// Phase-1-only launch wrapper
template <int MAX_NORB_T, typename T>
void guga_fused_hop_phase1_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* x,
    const T* eri_mat,
    const T* h_eff_flat,
    T* y,
    T* g_out,
    int* overflow_flag,
    cudaStream_t stream)
{
    if (j_count <= 0) return;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;
    int threads_per_block = kWarpsPerBlock * kWarpSize;
    int blocks = (j_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    guga_fused_hop_phase1_kernel_t<MAX_NORB_T, T><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix,
        steps_table, nodes_table,
        ncsf, norb,
        j_start, j_count,
        x, eri_mat, h_eff_flat,
        y, g_out, overflow_flag);
}

#define INSTANTIATE_FUSED_HOP_PHASE1(NORB, TYPE) \
    template void guga_fused_hop_phase1_launch_stream<NORB, TYPE>( \
        const int32_t*, const int16_t*, const int64_t*, \
        const int8_t*, const int32_t*, \
        int, int, int, int, \
        const TYPE*, const TYPE*, const TYPE*, \
        TYPE*, TYPE*, int*, cudaStream_t);

INSTANTIATE_FUSED_HOP_PHASE1(8, float)
INSTANTIATE_FUSED_HOP_PHASE1(8, double)
INSTANTIATE_FUSED_HOP_PHASE1(12, float)
INSTANTIATE_FUSED_HOP_PHASE1(12, double)
INSTANTIATE_FUSED_HOP_PHASE1(16, float)
INSTANTIATE_FUSED_HOP_PHASE1(16, double)
INSTANTIATE_FUSED_HOP_PHASE1(20, float)
INSTANTIATE_FUSED_HOP_PHASE1(20, double)

#undef INSTANTIATE_FUSED_HOP_PHASE1

// Phase-1 + COO output launch wrapper
template <int MAX_NORB_T, typename T>
void guga_fused_hop_phase1_coo_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* x,
    const T* eri_mat,
    const T* h_eff_flat,
    T* y,
    T* g_out,
    int* overflow_flag,
    int* coo_nnz_counter,
    int32_t* coo_j_local,
    int32_t* coo_k,
    int16_t* coo_pq,
    T* coo_w2,
    int max_coo,
    cudaStream_t stream)
{
    if (j_count <= 0) return;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;
    int threads_per_block = kWarpsPerBlock * kWarpSize;
    int blocks = (j_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    guga_fused_hop_phase1_coo_kernel_t<MAX_NORB_T, T>
        <<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix,
        steps_table, nodes_table,
        ncsf, norb,
        j_start, j_count,
        x, eri_mat, h_eff_flat,
        y, g_out, overflow_flag,
        coo_nnz_counter, coo_j_local, coo_k, coo_pq, coo_w2, max_coo);
}

#define INSTANTIATE_FUSED_HOP_PHASE1_COO(NORB, TYPE) \
    template void guga_fused_hop_phase1_coo_launch_stream<NORB, TYPE>( \
        const int32_t*, const int16_t*, const int64_t*, \
        const int8_t*, const int32_t*, \
        int, int, int, int, \
        const TYPE*, const TYPE*, const TYPE*, \
        TYPE*, TYPE*, int*, \
        int*, int32_t*, int32_t*, int16_t*, TYPE*, int, \
        cudaStream_t);

INSTANTIATE_FUSED_HOP_PHASE1_COO(8, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO(8, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO(12, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO(12, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO(16, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO(16, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO(20, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO(20, double)

#undef INSTANTIATE_FUSED_HOP_PHASE1_COO

// Phase-1 + COO output launch wrapper (merged DFS variant)
template <int MAX_NORB_T, typename T>
void guga_fused_hop_phase1_coo_merged_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* x,
    const T* eri_mat,
    const T* h_eff_flat,
    T* y,
    T* g_out,
    int* overflow_flag,
    int* coo_nnz_counter,
    int32_t* coo_j_local,
    int32_t* coo_k,
    int16_t* coo_pq,
    T* coo_w2,
    int max_coo,
    cudaStream_t stream)
{
    if (j_count <= 0) return;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kWarpSize = 32;
    int threads_per_block = kWarpsPerBlock * kWarpSize;
    int blocks = (j_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    guga_fused_hop_phase1_coo_merged_kernel_t<MAX_NORB_T, T>
        <<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix,
        steps_table, nodes_table,
        ncsf, norb,
        j_start, j_count,
        x, eri_mat, h_eff_flat,
        y, g_out, overflow_flag,
        coo_nnz_counter, coo_j_local, coo_k, coo_pq, coo_w2, max_coo);
}

#define INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(NORB, TYPE) \
    template void guga_fused_hop_phase1_coo_merged_launch_stream<NORB, TYPE>( \
        const int32_t*, const int16_t*, const int64_t*, \
        const int8_t*, const int32_t*, \
        int, int, int, int, \
        const TYPE*, const TYPE*, const TYPE*, \
        TYPE*, TYPE*, int*, \
        int*, int32_t*, int32_t*, int16_t*, TYPE*, int, \
        cudaStream_t);

INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(8, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(8, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(12, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(12, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(16, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(16, double)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(20, float)
INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED(20, double)

#undef INSTANTIATE_FUSED_HOP_PHASE1_COO_MERGED
