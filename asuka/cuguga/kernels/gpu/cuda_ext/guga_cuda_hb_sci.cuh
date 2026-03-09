// Heat-Bath SCI CUDA kernels: fused screening + apply + DFS walk.
//
// Phase 2: hb_screen_and_apply_kernel — one block per source CSF,
//   builds screened g_flat in shared memory, applies via DFS walk to hash.
// Phase 3: hb_fused_dfs_kernel — inline screening inside DFS walk.
// Phase 4: hb_stochastic_pt2_kernel — stochastic PT2 sampling kernel.
//
// This header is included by guga_cuda_kernels.cu (aggregator TU).
//
// Reference: Holmes, Sharma, Umrigar, JCTC 2017, 13, 1595.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#include "guga_cuda_frontier_hash.cuh"

namespace {

// ---------------------------------------------------------------------------
// Phase 2: Fused screen + g_flat build + apply kernel
// ---------------------------------------------------------------------------
// One block per source CSF j. Each block:
//   1. Loads steps/nodes/occ/b for CSF j from the state cache
//   2. Builds screened g_flat in shared memory from sorted integral index
//   3. Applies g_flat via the standard DFS walk → scatters to frontier hash
//
// Template parameter MAX_NORB_T controls shared memory sizing.
// For norb=40: g_flat = 40*40*8 = 12.8 KB, plus ~0.5 KB overhead = 13.3 KB,
// well under the 48 KB smem limit.

// Step-to-occupancy mapping (same as CPU-side _STEP_TO_OCC)
__device__ __constant__ int8_t hb_step_to_occ[4] = {0, 1, 1, 2};

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void hb_screen_and_apply_kernel(
    // Source CSF info
    const int32_t* __restrict__ sel_idx,       // [nsel] global CSF indices
    const double*  __restrict__ c_root,        // [nsel] CI coefficients for active root
    int nsel,
    int root,
    // State cache
    const int8_t*  __restrict__ steps_table,   // [ncsf * norb]
    const int32_t* __restrict__ nodes_table,   // [ncsf * (norb+1)]
    int norb,
    int ncsf,
    // Sorted integral index (device-resident)
    const int32_t* __restrict__ h1_pq,         // [n_h1, 2]
    const double*  __restrict__ h1_abs,        // [n_h1]
    const double*  __restrict__ h1_signed,     // [n_h1]
    int n_h1,
    const int64_t* __restrict__ pq_ptr,        // [norb^2 + 1]
    const int32_t* __restrict__ rs_idx,        // [nnz_2e]
    const double*  __restrict__ v_abs,         // [nnz_2e]
    const double*  __restrict__ v_signed,      // [nnz_2e]
    const double*  __restrict__ pq_max_v,      // [norb^2]
    double eps,
    // DRT tables (for DFS walk)
    const int32_t* __restrict__ child_table,   // [nnodes * 4]
    const int16_t* __restrict__ node_twos,     // [nnodes]
    const int64_t* __restrict__ child_prefix,  // [nnodes * 5]
    int nnodes,
    // Hash output
    int32_t* __restrict__ hash_keys,
    double*  __restrict__ hash_vals,
    int cap,
    const uint8_t* __restrict__ selected_mask,  // [ncsf] or NULL
    int* __restrict__ overflow_flag)
{
    // One block per source CSF
    int j_local = blockIdx.x;
    if (j_local >= nsel) return;

    int j_global = sel_idx[j_local];
    double cj = c_root[j_local];
    double abs_cj = fabs(cj);
    if (abs_cj == 0.0) return;

    double cutoff = eps / abs_cj;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int nops = norb * norb;

    // --- Shared memory layout ---
    // g_flat_s:              double[norb^2]       (8-byte aligned)
    // steps_s:               int8_t[norb]         padded to 8-byte boundary
    // nodes_s:               int32_t[norb+1]      (4-byte aligned after 8-byte pad)
    // occ_s:                 int8_t[norb]         padded to 2-byte boundary
    // b_s:                   int16_t[norb]        padded to 4-byte boundary
    // idx_prefix_s:          int32_t[norb+1]      (4-byte aligned after 4-byte pad)
    // idx_prefix_warp_sums:  int32_t[ceil(norb/32)]
    extern __shared__ char smem_raw[];
    double*  g_flat_s             = reinterpret_cast<double*>(smem_raw);
    int8_t*  steps_s              = reinterpret_cast<int8_t*>(g_flat_s + nops);
    int32_t* nodes_s              = reinterpret_cast<int32_t*>(steps_s + ((norb + 7) & ~7));
    int8_t*  occ_s                = reinterpret_cast<int8_t*>(nodes_s + norb + 1);
    // Pad occ_s to 4-byte boundary so b_s (int16_t) starts 4-byte-aligned,
    // which in turn ensures idx_prefix_s (int32_t) is 4-byte-aligned.
    int16_t* b_s                  = reinterpret_cast<int16_t*>(occ_s + ((norb + 3) & ~3));
    int32_t* idx_prefix_s         = reinterpret_cast<int32_t*>(
                                        reinterpret_cast<char*>(b_s) + ((norb * 2 + 3) & ~3));
    int32_t* idx_prefix_warp_sums = idx_prefix_s + (norb + 1);

    // --- Cooperative load of steps / nodes[k+1] / occ / b ---
    // (mirrors guga_apply_g_flat_scatter_atomic_frontier_hash_kernel_t pattern)
    for (int k = tid; k < norb; k += nthreads) {
        int8_t step = steps_table[(int64_t)j_global * norb + k];
        steps_s[k] = step;
        occ_s[k] = hb_step_to_occ[(int)(step & 3)];
        int32_t node_next = nodes_table[(int64_t)j_global * (norb + 1) + (k + 1)];
        nodes_s[k + 1] = node_next;
        b_s[k] = node_twos[node_next];
    }
    if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }
    if (tid == 0) {
        nodes_s[0] = nodes_table[(int64_t)j_global * (norb + 1)];
    }
    if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }

    // --- Build idx_prefix_s (inclusive prefix sum of child_prefix[node_k*5+step_k]) ---
    // Used by DFS walk to compute target CSF index from segment walks.
    if (norb <= 32) {
        int lane = tid;
        if (lane < 32) {
            int32_t delta = 0;
            if (lane < norb) {
                int node_k = nodes_s[lane];
                int step_k = (int)steps_s[lane];
                delta = (int32_t)child_prefix[node_k * 5 + step_k];
            }
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
                if (lane >= off) delta += v;
            }
            if (lane == 0) idx_prefix_s[0] = 0;
            if (lane < norb) idx_prefix_s[lane + 1] = delta;
        }
        if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }
    } else {
        if (nthreads >= 64) {
            int lane = tid & 31;
            int warp = tid >> 5;
            int warps_needed = (norb + 31) >> 5;
            int k = warp * 32 + lane;
            int32_t delta = 0;
            if (warp < warps_needed && k < norb) {
                int node_k = nodes_s[k];
                int step_k = (int)steps_s[k];
                delta = (int32_t)child_prefix[node_k * 5 + step_k];
            }
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
                if (lane >= off) delta += v;
            }
            if (warp < warps_needed && lane == 31) idx_prefix_warp_sums[warp] = delta;
            __syncthreads();
            int32_t warp_offset = 0;
            if (warp < warps_needed && warp > 0) warp_offset = idx_prefix_warp_sums[0];
            if (tid == 0) idx_prefix_s[0] = 0;
            if (warp < warps_needed && k < norb) idx_prefix_s[k + 1] = delta + warp_offset;
            __syncthreads();
        } else {
            if (tid == 0) {
                idx_prefix_s[0] = 0;
                int32_t acc = 0;
                for (int k = 0; k < norb; k++) {
                    int node_k = nodes_s[k];
                    int step_k = (int)steps_s[k];
                    acc += (int32_t)child_prefix[node_k * 5 + step_k];
                    idx_prefix_s[k + 1] = acc;
                }
            }
            __syncthreads();
        }
    }

    // --- Zero g_flat in shared memory ---
    for (int i = tid; i < nops; i += nthreads) {
        g_flat_s[i] = 0.0;
    }
    __syncthreads();

    // --- Build screened g_flat ---

    // One-body: scan sorted h1 entries until |h| < cutoff
    for (int k = tid; k < n_h1; k += nthreads) {
        if (h1_abs[k] < cutoff) break;
        int p = h1_pq[k * 2];
        int q = h1_pq[k * 2 + 1];
        // Each (p,q) pair appears at most once in h1, so no race condition
        // with strided access pattern across threads.
        g_flat_s[p * norb + q] = h1_signed[k];
    }
    __syncthreads();

    // Two-body: for each (p,q), scan sorted v entries
    for (int pq = tid; pq < nops; pq += nthreads) {
        if (pq_max_v[pq] < cutoff) continue;

        int64_t lo = pq_ptr[pq];
        int64_t hi = pq_ptr[pq + 1];
        if (lo >= hi) continue;

        double g_acc = 0.0;
        for (int64_t k = lo; k < hi; k++) {
            if (v_abs[k] < cutoff) break;
            int rs_flat = rs_idx[k];
            int r = rs_flat / norb;
            int s = rs_flat % norb;
            double v = v_signed[k];
            if (r == s) {
                // Diagonal: occupancy-weighted
                g_acc += 0.5 * (double)occ_s[r] * v;
            } else {
                // Off-diagonal: raw integral
                g_acc += 0.5 * v;
            }
        }
        g_flat_s[pq] += g_acc;
    }
    __syncthreads();

    // --- DFS walk: apply screened g_flat to frontier hash ---
    // Mirrors guga_apply_g_flat_scatter_atomic_frontier_hash_kernel_t exactly,
    // reading g_flat from shared memory (g_flat_s) instead of global memory.
    //
    // Diagonal (p==q): skipped — csf_idx is always in the selected support, and
    // its self-contribution does not affect external frontier amplitudes.

    for (int pq = tid; pq < nops; pq += nthreads) {
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) continue;

        double wgt = g_flat_s[pq];
        if (wgt == 0.0) continue;

        int occ_p = (int)occ_s[p];
        int occ_q = (int)occ_s[q];
        if (occ_q <= 0 || occ_p >= 2) continue;

        int start, end, q_start, q_mid, q_end;
        if (p < q) {
            start = p; end = q;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q; end = p;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start      = nodes_s[start];
        int32_t node_end_target = nodes_s[end + 1];
        int32_t prefix_offset   = idx_prefix_s[start];
        int32_t prefix_endplus1 = idx_prefix_s[end + 1];
        int32_t suffix_offset   = (int32_t)j_global - prefix_endplus1;

        int8_t   st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        double   st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = 1.0;
        top++;

        while (top) {
            top--;
            double w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)steps_s[kpos];
            int bk = (int)b_s[kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child_table[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                double seg = (double)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == 0.0) continue;
                double w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_i >= (unsigned)ncsf) { overflow = 1; continue; }
                    if (csf_i == j_global) continue;
                    if (selected_mask && selected_mask[csf_i]) continue;
                    if (w2 == 0.0) continue;
                    guga_frontier_hash_insert_add_f64(
                        hash_keys, hash_vals, cap, root, csf_i,
                        cj * wgt * w2, overflow_flag);
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) |
                                       ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }

        if (overflow) atomicExch(overflow_flag, 1);
    }
}


// ---------------------------------------------------------------------------
// Phase 3: Fused heat-bath DFS kernel (maximum throughput)
// ---------------------------------------------------------------------------
// Eliminates g_flat intermediate: screens inside the DFS walk itself.
// Multi-warp parallelism: each warp handles a subset of (p,q) pairs.

// This is a more advanced version that will be developed iteratively.
// The kernel signature is defined here for forward declaration.

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void hb_fused_dfs_kernel(
    const int32_t* __restrict__ sel_idx,
    const double*  __restrict__ c_root,
    int nsel,
    int root,
    const int8_t*  __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    int norb,
    int ncsf,
    const int32_t* __restrict__ h1_pq,
    const double*  __restrict__ h1_abs,
    const double*  __restrict__ h1_signed,
    int n_h1,
    const int64_t* __restrict__ pq_ptr,
    const int32_t* __restrict__ rs_idx,
    const double*  __restrict__ v_abs,
    const double*  __restrict__ v_signed,
    const double*  __restrict__ pq_max_v,
    double eps,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int nnodes,
    int32_t* __restrict__ hash_keys,
    double*  __restrict__ hash_vals,
    int cap,
    const uint8_t* __restrict__ selected_mask,
    int* __restrict__ overflow_flag)
{
    int j_local = blockIdx.x;
    if (j_local >= nsel) return;

    int j_global = sel_idx[j_local];
    if ((unsigned)j_global >= (unsigned)ncsf) {
        if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
        return;
    }
    double cj = c_root[j_local];
    double abs_cj = fabs(cj);
    if (abs_cj == 0.0) return;
    if (norb > MAX_NORB_T) {
        if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
        return;
    }
    (void)nnodes;

    double cutoff = eps / abs_cj;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int nops = norb * norb;

    __shared__ int8_t steps_s[MAX_NORB_T];
    __shared__ int32_t nodes_s[MAX_NORB_T + 1];
    __shared__ int8_t occ_s[MAX_NORB_T];
    __shared__ int16_t b_s[MAX_NORB_T];
    __shared__ int32_t idx_prefix_s[MAX_NORB_T + 1];
    __shared__ int32_t idx_prefix_warp_sums[(MAX_NORB_T + 31) / 32];

    for (int k = tid; k < norb; k += nthreads) {
        int8_t step = steps_table[(int64_t)j_global * norb + k];
        steps_s[k] = step;
        occ_s[k] = hb_step_to_occ[(int)(step & 3)];
        int32_t node_next = nodes_table[(int64_t)j_global * (norb + 1) + (k + 1)];
        nodes_s[k + 1] = node_next;
        b_s[k] = node_twos[node_next];
    }
    if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }
    if (tid == 0) {
        nodes_s[0] = nodes_table[(int64_t)j_global * (norb + 1)];
    }
    if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }

    if (norb <= 32) {
        int lane = tid;
        if (lane < 32) {
            int32_t delta = 0;
            if (lane < norb) {
                int node_k = nodes_s[lane];
                int step_k = (int)steps_s[lane];
                delta = (int32_t)child_prefix[node_k * 5 + step_k];
            }
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
                if (lane >= off) delta += v;
            }
            if (lane == 0) idx_prefix_s[0] = 0;
            if (lane < norb) idx_prefix_s[lane + 1] = delta;
        }
        if (nthreads > 32) { __syncthreads(); } else { __syncwarp(); }
    } else {
        if (nthreads >= 64) {
            int lane = tid & 31;
            int warp = tid >> 5;
            int warps_needed = (norb + 31) >> 5;
            int k = warp * 32 + lane;
            int32_t delta = 0;
            if (warp < warps_needed && k < norb) {
                int node_k = nodes_s[k];
                int step_k = (int)steps_s[k];
                delta = (int32_t)child_prefix[node_k * 5 + step_k];
            }
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int32_t v = __shfl_up_sync(0xffffffffu, delta, off);
                if (lane >= off) delta += v;
            }
            if (warp < warps_needed && lane == 31) idx_prefix_warp_sums[warp] = delta;
            __syncthreads();
            int32_t warp_offset = 0;
            if (warp < warps_needed && warp > 0) warp_offset = idx_prefix_warp_sums[0];
            if (tid == 0) idx_prefix_s[0] = 0;
            if (warp < warps_needed && k < norb) idx_prefix_s[k + 1] = delta + warp_offset;
            __syncthreads();
        } else {
            if (tid == 0) {
                idx_prefix_s[0] = 0;
                int32_t acc = 0;
                for (int k = 0; k < norb; k++) {
                    int node_k = nodes_s[k];
                    int step_k = (int)steps_s[k];
                    acc += (int32_t)child_prefix[node_k * 5 + step_k];
                    idx_prefix_s[k + 1] = acc;
                }
            }
            __syncthreads();
        }
    }

    for (int pq = tid; pq < nops; pq += nthreads) {
        int p = pq / norb;
        int q = pq - p * norb;
        if (p == q) continue;

        int occ_p = (int)occ_s[p];
        int occ_q = (int)occ_s[q];
        if (occ_q <= 0 || occ_p >= 2) continue;

        double wgt = 0.0;

        for (int k = 0; k < n_h1; k++) {
            if (h1_abs[k] < cutoff) break;
            int hp = h1_pq[k * 2];
            int hq = h1_pq[k * 2 + 1];
            if (hp == p && hq == q) {
                wgt += h1_signed[k];
                break;
            }
        }

        if (pq_max_v[pq] >= cutoff) {
            int64_t lo = pq_ptr[pq];
            int64_t hi = pq_ptr[pq + 1];
            for (int64_t k = lo; k < hi; k++) {
                if (v_abs[k] < cutoff) break;
                int rs_flat = rs_idx[k];
                int r = rs_flat / norb;
                int s = rs_flat % norb;
                double v = v_signed[k];
                if (r == s) {
                    wgt += 0.5 * (double)occ_s[r] * v;
                } else {
                    wgt += 0.5 * v;
                }
            }
        }

        if (wgt == 0.0) continue;

        int start, end, q_start, q_mid, q_end;
        if (p < q) {
            start = p; end = q;
            q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
        } else {
            start = q; end = p;
            q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
        }

        int32_t node_start      = nodes_s[start];
        int32_t node_end_target = nodes_s[end + 1];
        int32_t prefix_offset   = idx_prefix_s[start];
        int32_t prefix_endplus1 = idx_prefix_s[end + 1];
        int32_t suffix_offset   = (int32_t)j_global - prefix_endplus1;

        int8_t   st_k[MAX_NORB_T];
        uint64_t st_node_seg[MAX_NORB_T];
        double   st_w[MAX_NORB_T];
        int top = 0;
        int overflow = 0;

        st_k[top] = (int8_t)start;
        st_node_seg[top] = ((uint64_t)(uint32_t)0) | ((uint64_t)(uint32_t)node_start << 32);
        st_w[top] = 1.0;
        top++;

        while (top) {
            top--;
            double w = st_w[top];
            int kpos = (int)st_k[top];
            uint64_t node_seg = st_node_seg[top];
            int node_k = (int)(node_seg >> 32);
            int32_t seg_idx = (int32_t)(uint32_t)node_seg;

            int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);
            int dk = (int)steps_s[kpos];
            int bk = (int)b_s[kpos];
            int k_next = kpos + 1;

            int dp0 = 0, dp1 = 0;
            int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
            if (ndp == 0) continue;

            for (int which = 0; which < ndp; which++) {
                int dprime = (which == 0) ? dp0 : dp1;
                int child_k = child_table[node_k * 4 + dprime];
                if (child_k < 0) continue;
                int bprime = (int)node_twos[child_k];
                int db = bk - bprime;
                double seg = (double)segment_value_int(qk, dprime, dk, db, bk);
                if (seg == 0.0) continue;
                double w2 = w * seg;
                int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

                if (kpos == end) {
                    if (child_k != node_end_target) continue;
                    int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
                    if ((unsigned)csf_i >= (unsigned)ncsf) { overflow = 1; continue; }
                    if (csf_i == j_global) continue;
                    if (selected_mask && selected_mask[csf_i]) continue;
                    if (w2 == 0.0) continue;
                    guga_frontier_hash_insert_add_f64(
                        hash_keys, hash_vals, cap, root, csf_i,
                        cj * wgt * w2, overflow_flag);
                } else {
                    if (top >= MAX_NORB_T) { overflow = 1; continue; }
                    st_k[top] = (int8_t)k_next;
                    st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) |
                                       ((uint64_t)(uint32_t)child_k << 32);
                    st_w[top] = w2;
                    top++;
                }
            }
        }
        if (overflow) atomicExch(overflow_flag, 1);
    }
}


// ---------------------------------------------------------------------------
// Phase 4: Semi-stochastic PT2 kernel
// ---------------------------------------------------------------------------

template <int MAX_NORB_T>
__global__ __launch_bounds__(256, 2)
void hb_stochastic_pt2_kernel(
    // Sampled source CSFs (importance-sampled by |c_j|^2)
    const int32_t* __restrict__ sample_j_idx,   // [n_samples] indices into sel_idx
    const double*  __restrict__ sample_weight,   // [n_samples] importance weights
    int n_samples,
    // Variational space
    const int32_t* __restrict__ sel_idx,
    const double*  __restrict__ c_sel,           // [nsel * nroots]
    int nsel,
    int nroots,
    // Full integrals (no screening — exact for each sample)
    const double*  __restrict__ eri_flat,        // [norb^4] pre-materialized
    const double*  __restrict__ h1e_eff,         // [norb^2]
    // State cache + DRT
    const int8_t*  __restrict__ steps_table,
    const int32_t* __restrict__ nodes_table,
    const int32_t* __restrict__ child_table,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    int norb,
    int nnodes,
    int ncsf,
    // Variational eigenvalues + diagonal
    const double*  __restrict__ e_var,           // [nroots]
    const double*  __restrict__ hdiag,           // [ncsf]
    // Deterministic set (to exclude)
    const int32_t* __restrict__ det_set_sorted,  // [n_det_set]
    int n_det_set,
    // Output: per-root PT2 accumulators (atomic add)
    double*  __restrict__ pt2_sum,               // [nroots]
    double*  __restrict__ pt2_sumsq)             // [nroots] for variance
{
    // Phase 4 placeholder — one block per sampled source CSF.
    // To be implemented when Phase 3 DFS walk is complete.
}


}  // namespace


// ---------------------------------------------------------------------------
// Host launch wrappers (extern "C" for pybind11)
// ---------------------------------------------------------------------------

extern "C" cudaError_t guga_hb_screen_and_apply_launch_stream(
    const int32_t* sel_idx,
    const double*  c_root,
    int nsel,
    int root,
    const int8_t*  steps_table,
    const int32_t* nodes_table,
    int norb,
    int ncsf,
    const int32_t* h1_pq,
    const double*  h1_abs,
    const double*  h1_signed,
    int n_h1,
    const int64_t* pq_ptr,
    const int32_t* rs_idx,
    const double*  v_abs,
    const double*  v_signed,
    const double*  pq_max_v,
    double eps,
    const int32_t* child_table,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    int nnodes,
    int32_t* hash_keys,
    double*  hash_vals,
    int cap,
    const uint8_t* selected_mask,
    int* overflow_flag,
    cudaStream_t stream,
    int threads)
{
    if (!sel_idx || !c_root) return cudaErrorInvalidValue;
    if (nsel <= 0) return cudaSuccess;
    if (norb <= 0 || norb > 64) return cudaErrorInvalidValue;
    if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

    // 5-way dispatch matching apply_g_flat kernel specializations.
    // Phase-3 fused DFS path: no materialized g_flat shared buffer.
    if (norb <= 16) {
        hb_fused_dfs_kernel<16><<<nsel, threads, 0, stream>>>(
            sel_idx, c_root, nsel, root,
            steps_table, nodes_table, norb, ncsf,
            h1_pq, h1_abs, h1_signed, n_h1,
            pq_ptr, rs_idx, v_abs, v_signed, pq_max_v, eps,
            child_table, node_twos, child_prefix, nnodes,
            hash_keys, hash_vals, cap, selected_mask, overflow_flag);
    } else if (norb <= 24) {
        hb_fused_dfs_kernel<24><<<nsel, threads, 0, stream>>>(
            sel_idx, c_root, nsel, root,
            steps_table, nodes_table, norb, ncsf,
            h1_pq, h1_abs, h1_signed, n_h1,
            pq_ptr, rs_idx, v_abs, v_signed, pq_max_v, eps,
            child_table, node_twos, child_prefix, nnodes,
            hash_keys, hash_vals, cap, selected_mask, overflow_flag);
    } else if (norb <= 32) {
        hb_fused_dfs_kernel<32><<<nsel, threads, 0, stream>>>(
            sel_idx, c_root, nsel, root,
            steps_table, nodes_table, norb, ncsf,
            h1_pq, h1_abs, h1_signed, n_h1,
            pq_ptr, rs_idx, v_abs, v_signed, pq_max_v, eps,
            child_table, node_twos, child_prefix, nnodes,
            hash_keys, hash_vals, cap, selected_mask, overflow_flag);
    } else if (norb <= 48) {
        hb_fused_dfs_kernel<48><<<nsel, threads, 0, stream>>>(
            sel_idx, c_root, nsel, root,
            steps_table, nodes_table, norb, ncsf,
            h1_pq, h1_abs, h1_signed, n_h1,
            pq_ptr, rs_idx, v_abs, v_signed, pq_max_v, eps,
            child_table, node_twos, child_prefix, nnodes,
            hash_keys, hash_vals, cap, selected_mask, overflow_flag);
    } else {
        hb_fused_dfs_kernel<64><<<nsel, threads, 0, stream>>>(
            sel_idx, c_root, nsel, root,
            steps_table, nodes_table, norb, ncsf,
            h1_pq, h1_abs, h1_signed, n_h1,
            pq_ptr, rs_idx, v_abs, v_signed, pq_max_v, eps,
            child_table, node_twos, child_prefix, nnodes,
            hash_keys, hash_vals, cap, selected_mask, overflow_flag);
    }
    return cudaGetLastError();
}
