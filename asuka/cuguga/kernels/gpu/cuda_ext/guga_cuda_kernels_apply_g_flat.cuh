// 10.16.6: __launch_bounds__ hints the compiler for better register allocation.
template <int MAX_NORB_T, typename T>
__global__ __launch_bounds__(256, 2)
void guga_apply_g_flat_scatter_atomic_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,    // [ntasks]
    const T* __restrict__ task_scale,  // [ntasks] or NULL
    const T* __restrict__ task_g,      // [nops] or [ntasks,nops]
    int64_t g_stride,                       // 0 (shared) or nops
    int ntasks,
    T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  T scale = task_scale ? task_scale[task_id] : (T)1;
  if (scale == (T)0) return;

  int nops = norb * norb;
  const T* g_flat = task_g + (int64_t)task_id * g_stride;

  __shared__ int8_t steps_s[MAX_NORB_T];
  __shared__ int32_t nodes_s[MAX_NORB_T + 1];
  __shared__ int8_t occ_s[MAX_NORB_T];
  __shared__ int16_t b_s[MAX_NORB_T];
  __shared__ int32_t idx_prefix_s[MAX_NORB_T + 1];
  __shared__ int32_t idx_prefix_warp_sums[(MAX_NORB_T + 31) / 32];

  // Cooperative load of steps/nodes and derived occ/b arrays.
  for (int k = (int)threadIdx.x; k < norb; k += (int)blockDim.x) {
    int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)k];
    steps_s[k] = step;
    occ_s[k] = (int8_t)step_to_occ(step);
    int32_t node_next = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1) + (int64_t)(k + 1)];
    nodes_s[k + 1] = node_next;
    b_s[k] = node_twos[node_next];
  }
  // Ensure all shared writes are visible before thread 0 builds idx_prefix_s.
  // For norb <= 32, only the first warp writes; avoid a full-block barrier.
  if ((int)blockDim.x > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
  if ((int)threadIdx.x == 0) {
    nodes_s[0] = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1)];
  }
  if ((int)blockDim.x > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
  if (norb <= 32) {
    int lane = (int)threadIdx.x;
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
    if ((int)blockDim.x > 32) {
      __syncthreads();
    } else {
      __syncwarp();
    }
  } else {
    // Multi-warp inclusive scan (norb <= MAX_NORB_T <= 64). Prefer this when the block has >=2 full warps;
    // fall back to the serial thread-0 loop for smaller blocks.
    if ((int)blockDim.x >= 64) {
      int tid = (int)threadIdx.x;
      int lane = tid & 31;
      int warp = tid >> 5;
      int warps_needed = (norb + 31) >> 5;  // 2 for (32,64].
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
      if (warp < warps_needed && warp > 0) {
        // With MAX_NORB_T <= 64, we only need to offset by the first warp sum.
        warp_offset = idx_prefix_warp_sums[0];
      }
      if (tid == 0) idx_prefix_s[0] = 0;
      if (warp < warps_needed && k < norb) idx_prefix_s[k + 1] = delta + warp_offset;
      __syncthreads();
    } else {
      if ((int)threadIdx.x == 0) {
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

  // Diagonal (p==q) contribution: sum once per task to reduce atomics.
  if ((int)threadIdx.x == 0) {
    T diag = (T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = (int)occ_s[p];
      if (!occ_p) continue;
      T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != (T)0) diag += wgt_pp * (T)occ_p;
    }
    if (diag != (T)0) atomicAdd(&y[csf_idx], scale * diag);
  }

  // Each thread processes a strided subset of pq entries (skip diag; handled above).
  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    T wgt = g_flat[pq];
    if (wgt == (T)0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start;
    int end;
    int q_start;
    int q_mid;
    int q_end;
    if (p < q) {
      start = p;
      end = q;
      q_start = Q_uR;
      q_mid = Q_R;
      q_end = Q_oR;
    } else {
      start = q;
      end = p;
      q_start = Q_uL;
      q_mid = Q_L;
      q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];

    int32_t prefix_offset = idx_prefix_s[start];
    int32_t prefix_endplus1 = idx_prefix_s[end + 1];
    int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

    int8_t st_k[MAX_NORB_T];
    uint64_t st_node_seg[MAX_NORB_T];  // [node(32) | seg(32)]
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

      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
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
          int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if ((unsigned)csf_i >= (unsigned)ncsf) {
            overflow = 1;
            continue;
          }
          if (csf_i == csf_idx) continue;
          if (w2 == (T)0) continue;
          atomicAdd(&y[csf_i], scale * wgt * w2);
        } else {
          if (top >= MAX_NORB_T) {
            overflow = 1;
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node_seg[top] = ((uint64_t)(uint32_t)seg_idx2) | ((uint64_t)(uint32_t)child_k << 32);
          st_w[top] = w2;
          top++;
        }
      }
    }

    if (overflow) atomicExch(overflow_flag, 1);
  }
}

// =============================================================================
// 10.16.3 / 10.18: Warp-Cooperative Segment Walk Kernel
// =============================================================================
// This kernel eliminates per-thread DFS stacks by using warp-wide parallelism:
// - Lane 0 serially expands the first SPLIT_LEVELS to create up to 32 seeds
// - Seeds are distributed across warp lanes
// - Each lane processes its seed's subtree independently
// - Results are reduced via warp shuffle
//
// Key optimization: no per-thread local memory arrays - all state in registers
// and shared memory, shared via __shfl_sync().

template <int MAX_NORB_T, typename T, int SPLIT_LEVELS_T = 2>
__global__ __launch_bounds__(256, 2)
void guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,    // [ntasks]
    const T* __restrict__ task_scale,  // [ntasks] or NULL
    const T* __restrict__ task_g,      // [nops] or [ntasks,nops]
    int64_t g_stride,                       // 0 (shared) or nops
    int ntasks,
    T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  constexpr int kWarpSize = 32;
  constexpr int kMaxSeeds = 32;  // max seeds that can be distributed to lanes
  constexpr int kMaxWarpsPerBlock = 8;

  // Per-warp shared memory for seed distribution and CSF state
  __shared__ int8_t sh_steps[kMaxWarpsPerBlock][MAX_NORB_T];
  __shared__ int32_t sh_nodes[kMaxWarpsPerBlock][MAX_NORB_T + 1];
  __shared__ int8_t sh_occ[kMaxWarpsPerBlock][MAX_NORB_T];
  __shared__ int16_t sh_b[kMaxWarpsPerBlock][MAX_NORB_T];
  __shared__ int32_t sh_idx_prefix[kMaxWarpsPerBlock][MAX_NORB_T + 1];

  // Seed arrays for warp distribution
  __shared__ int8_t sh_seed_k[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_node[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ T sh_seed_w[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int32_t sh_seed_seg[kMaxWarpsPerBlock][kMaxSeeds];
  __shared__ int sh_seed_count[kMaxWarpsPerBlock];

  // Per-warp task context
  __shared__ int sh_task_csf[kMaxWarpsPerBlock];
  __shared__ T sh_task_scale[kMaxWarpsPerBlock];
  __shared__ int sh_status[kMaxWarpsPerBlock];  // 0=ok, 1=overflow, 2=skip

  int lane = (int)(threadIdx.x & (kWarpSize - 1));
  int warp_local = (int)(threadIdx.x >> 5);
  int warps_per_block = (int)(blockDim.x >> 5);
  if (warps_per_block <= 0 || warps_per_block > kMaxWarpsPerBlock) return;

  int task_id = (int)(blockIdx.x * warps_per_block + warp_local);
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int nops = norb * norb;

  // --- Phase 1: Lane 0 loads CSF state and prepares task ---
  int csf_idx = -1;
  T scale = (T)0;
  
  if (lane == 0) {
    sh_status[warp_local] = 0;
    csf_idx = task_csf[task_id];
    if ((unsigned)csf_idx >= (unsigned)ncsf) {
      sh_status[warp_local] = 1;
      atomicExch(overflow_flag, 1);
    } else {
      scale = task_scale ? task_scale[task_id] : (T)1;
      if (scale == (T)0) {
        sh_status[warp_local] = 2;  // skip this task
      } else {
        sh_task_csf[warp_local] = csf_idx;
        sh_task_scale[warp_local] = scale;
        
        // Load steps, nodes, and compute derived arrays
        for (int k = 0; k < norb; k++) {
          int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)k];
          sh_steps[warp_local][k] = step;
          sh_occ[warp_local][k] = (int8_t)step_to_occ(step);
          int32_t node_next = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1) + (int64_t)(k + 1)];
          sh_nodes[warp_local][k + 1] = node_next;
          sh_b[warp_local][k] = node_twos[node_next];
        }
        sh_nodes[warp_local][0] = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1)];
        
        // Build idx_prefix via inclusive scan
        sh_idx_prefix[warp_local][0] = 0;
        int32_t acc = 0;
        for (int k = 0; k < norb; k++) {
          int node_k = sh_nodes[warp_local][k];
          int step_k = (int)sh_steps[warp_local][k];
          acc += (int32_t)child_prefix[node_k * 5 + step_k];
          sh_idx_prefix[warp_local][k + 1] = acc;
        }
      }
    }
  }
  __syncwarp();

  int status = sh_status[warp_local];
  if (status != 0) return;

  csf_idx = sh_task_csf[warp_local];
  scale = sh_task_scale[warp_local];
  const T* g_flat = task_g + (int64_t)task_id * g_stride;

  // --- Phase 2: Handle diagonal contribution (lane 0 only) ---
  if (lane == 0) {
    T diag = (T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = (int)sh_occ[warp_local][p];
      if (!occ_p) continue;
      T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != (T)0) diag += wgt_pp * (T)occ_p;
    }
    if (diag != (T)0) atomicAdd(&y[csf_idx], scale * diag);
  }

  // --- Phase 3: Off-diagonal contributions via warp-cooperative walk ---
  // Each (p,q) pair with p != q is processed by the warp cooperatively.
  // Lane 0 expands SPLIT_LEVELS levels to create seeds, then lanes process subtrees.
  
  for (int pq = 0; pq < nops; pq++) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    T wgt = g_flat[pq];
    if (wgt == (T)0) continue;

    int occ_p = (int)sh_occ[warp_local][p];
    int occ_q = (int)sh_occ[warp_local][q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start, end, q_start, q_mid, q_end;
    if (p < q) {
      start = p; end = q;
      q_start = Q_uR; q_mid = Q_R; q_end = Q_oR;
    } else {
      start = q; end = p;
      q_start = Q_uL; q_mid = Q_L; q_end = Q_oL;
    }

    int32_t node_start = sh_nodes[warp_local][start];
    int32_t node_end_target = sh_nodes[warp_local][end + 1];
    int32_t prefix_offset = sh_idx_prefix[warp_local][start];
    int32_t prefix_endplus1 = sh_idx_prefix[warp_local][end + 1];
    int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

    // --- Seed expansion by lane 0 ---
    int seed_count = 0;
    int any_overflow = 0;
    
    if (lane == 0) {
      int8_t cur_k[kMaxSeeds];
      int32_t cur_node[kMaxSeeds];
      T cur_w[kMaxSeeds];
      int32_t cur_seg[kMaxSeeds];
      int cur_count = 1;
      cur_k[0] = (int8_t)start;
      cur_node[0] = node_start;
      cur_w[0] = (T)1;
      cur_seg[0] = 0;

      int path_length = end - start;
      int split_levels = (path_length < SPLIT_LEVELS_T) ? path_length : SPLIT_LEVELS_T;
      
      // Expand first split_levels levels serially
      for (int depth = 0; depth < split_levels && cur_count > 0 && cur_count <= kMaxSeeds; depth++) {
        int8_t next_k[kMaxSeeds];
        int32_t next_node[kMaxSeeds];
        T next_w[kMaxSeeds];
        int32_t next_seg[kMaxSeeds];
        int next_count = 0;

        for (int si = 0; si < cur_count && next_count < kMaxSeeds; si++) {
          int kpos = (int)cur_k[si];
          int node_k = (int)cur_node[si];
          T w = cur_w[si];
          int32_t seg_idx = cur_seg[si];

          int is_first = (kpos == start);
          int is_last = (kpos == end);
          int qk = is_first ? q_start : (is_last ? q_end : q_mid);
          int dk = (int)sh_steps[warp_local][kpos];
          int bk = (int)sh_b[warp_local][kpos];
          int k_next = kpos + 1;

          int dp0 = 0, dp1 = 0;
          int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
          if (ndp == 0) continue;

          for (int which = 0; which < ndp && next_count < kMaxSeeds; which++) {
            int dprime = (which == 0) ? dp0 : dp1;
            int child_k = child[node_k * 4 + dprime];
            if (child_k < 0) continue;
            int bprime = (int)node_twos[child_k];
            int db = bk - bprime;
            T seg = (T)segment_value_int(qk, dprime, dk, db, bk);
            if (seg == (T)0) continue;

            next_k[next_count] = (int8_t)k_next;
            next_node[next_count] = child_k;
            next_w[next_count] = w * seg;
            next_seg[next_count] = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];
            next_count++;
          }
        }

        cur_count = next_count;
        for (int si = 0; si < cur_count; si++) {
          cur_k[si] = next_k[si];
          cur_node[si] = next_node[si];
          cur_w[si] = next_w[si];
          cur_seg[si] = next_seg[si];
        }
      }

      // Store seeds to shared memory for warp distribution
      seed_count = cur_count;
      sh_seed_count[warp_local] = seed_count;
      for (int si = 0; si < seed_count; si++) {
        sh_seed_k[warp_local][si] = cur_k[si];
        sh_seed_node[warp_local][si] = cur_node[si];
        sh_seed_w[warp_local][si] = cur_w[si];
        sh_seed_seg[warp_local][si] = cur_seg[si];
      }
    }
    __syncwarp();

    seed_count = sh_seed_count[warp_local];
    if (seed_count == 0) continue;

    // --- Each lane processes its assigned seed(s) ---
    // For seed_count <= 32, each lane processes at most 1 seed
    int local_overflow = 0;
    
    if (lane < seed_count) {
      int8_t st_k[MAX_NORB_T];
      int32_t st_node[MAX_NORB_T];
      T st_w[MAX_NORB_T];
      int32_t st_seg[MAX_NORB_T];
      int top = 0;

      st_k[top] = sh_seed_k[warp_local][lane];
      st_node[top] = sh_seed_node[warp_local][lane];
      st_w[top] = sh_seed_w[warp_local][lane];
      st_seg[top] = sh_seed_seg[warp_local][lane];
      top++;

      while (top) {
        top--;
        int kpos = (int)st_k[top];
        int node_k = (int)st_node[top];
        T w = st_w[top];
        int32_t seg_idx = st_seg[top];

        int is_last = (kpos == end);
        int qk = (kpos == start) ? q_start : (is_last ? q_end : q_mid);
        int dk = (int)sh_steps[warp_local][kpos];
        int bk = (int)sh_b[warp_local][kpos];
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

          if (is_last) {
            if (child_k != node_end_target) continue;
            int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
            if ((unsigned)csf_i >= (unsigned)ncsf) {
              local_overflow = 1;
              continue;
            }
            if (csf_i == csf_idx) continue;
            if (w2 == (T)0) continue;
            atomicAdd(&y[csf_i], scale * wgt * w2);
          } else {
            if (top >= MAX_NORB_T) {
              local_overflow = 1;
              continue;
            }
            st_k[top] = (int8_t)k_next;
            st_node[top] = child_k;
            st_w[top] = w2;
            st_seg[top] = seg_idx2;
            top++;
          }
        }
      }
    }

    // Check for overflow across warp
    any_overflow = __any_sync(0xFFFFFFFF, local_overflow != 0);
    if (any_overflow && lane == 0) {
      atomicExch(overflow_flag, 1);
    }
  }
}

template <typename INDPTR_T, typename PQ_T>
__global__ void guga_apply_g_flat_scatter_atomic_epq_table_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_indptr,  // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [nnz]
    const PQ_T* __restrict__ epq_pq,          // [nnz] (pq_id = p*norb+q, p!=q)
    const double* __restrict__ epq_data,      // [nnz]
    const int32_t* __restrict__ task_csf,     // [ntasks]
    const double* __restrict__ task_scale,   // [ntasks] or NULL
    const double* __restrict__ task_g,        // [nops] or [ntasks,nops]
    int64_t g_stride,                         // 0 (shared) or nops
    int ntasks,
    double* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double scale = task_scale ? task_scale[task_id] : 1.0;
  if (scale == 0.0) return;

  int nops = norb * norb;
  const double* g_flat = task_g + (int64_t)task_id * g_stride;

  // Diagonal (p==q) contribution: sum once per task to reduce atomics.
  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_idx * (int64_t)norb;
    double diag = 0.0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      double wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != 0.0) diag += wgt_pp * (double)occ_p;
    }
    if (diag != 0.0) atomicAdd(&y[csf_idx], scale * diag);
  }

  int64_t start = epq_indptr[(int64_t)csf_idx];
  int64_t end = epq_indptr[(int64_t)csf_idx + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    double w = epq_data[t];
    if (w == 0.0) continue;
    double wgt = g_flat[pq];
    if (wgt == 0.0) continue;
    atomicAdd(&y[csf_i], scale * wgt * w);
  }
}

template <typename INDPTR_T, typename OUT_T, typename COEF_T, typename PQ_T>
__global__ void guga_apply_g_flat_gather_epq_table_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_t_indptr,  // [ncsf+1], rows by destination i
    const int32_t* __restrict__ epq_t_source,   // [nnz], source j
    const PQ_T* __restrict__ epq_t_pq,          // [nnz], pq_id = p*norb+q
    const COEF_T* __restrict__ epq_t_data,      // [nnz]
    const int32_t* __restrict__ task_row_by_csf,    // [ncsf], -1 if inactive; row index for task_g when g_stride>0
    const OUT_T* __restrict__ task_scale_by_csf,    // [ncsf] or NULL
    const OUT_T* __restrict__ task_g,               // [nops] or [ntasks,nops]
    int64_t g_stride,                               // 0 (shared) or nops
    OUT_T* __restrict__ y,
    int* __restrict__ overflow_flag,
    int add) {
  int csf_i = (int)blockIdx.x;
  if (csf_i >= ncsf) return;

  int nops = norb * norb;
  OUT_T sum = (OUT_T)0;

  // Diagonal contribution from active source j == i.
  int row_i = task_row_by_csf[csf_i];
  if (row_i >= 0) {
    OUT_T scale_i = task_scale_by_csf ? task_scale_by_csf[csf_i] : (OUT_T)1;
    if (scale_i != (OUT_T)0) {
      const OUT_T* g_flat_i = task_g + (int64_t)row_i * g_stride;
      const int8_t* steps_i = steps_table + (int64_t)csf_i * (int64_t)norb;
      for (int p = (int)threadIdx.x; p < norb; p += (int)blockDim.x) {
        int occ_p = step_to_occ(steps_i[p]);
        if (!occ_p) continue;
        int pq = p * norb + p;
        OUT_T wgt = g_flat_i[pq];
        if (wgt != (OUT_T)0) sum += scale_i * (OUT_T)occ_p * wgt;
      }
    }
  }

  int64_t start = epq_t_indptr[(int64_t)csf_i];
  int64_t end = epq_t_indptr[(int64_t)csf_i + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_j = epq_t_source[t];
    if ((unsigned)csf_j >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int row_j = task_row_by_csf[csf_j];
    if (row_j < 0) continue;

    int pq = (int)epq_t_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }

    OUT_T coef = (OUT_T)epq_t_data[t];
    if (coef == (OUT_T)0) continue;

    OUT_T scale_j = task_scale_by_csf ? task_scale_by_csf[csf_j] : (OUT_T)1;
    if (scale_j == (OUT_T)0) continue;

    const OUT_T* g_flat_j = task_g + (int64_t)row_j * g_stride;
    OUT_T wgt = g_flat_j[pq];
    if (wgt == (OUT_T)0) continue;
    sum += scale_j * wgt * coef;
  }

  // Force warp reconvergence after divergent EPQ loop (required on Volta+ with
  // Independent Thread Scheduling — threads may not reconverge automatically).
  __syncwarp();

  int lane = (int)threadIdx.x & 31;
  int warp = (int)threadIdx.x >> 5;
  int nwarps = ((int)blockDim.x + 31) >> 5;
  // Deterministic mask: all lanes in this warp that correspond to actual threads.
  int warp_threads = min(32, (int)blockDim.x - warp * 32);
  unsigned warp_mask = (warp_threads == 32) ? 0xFFFFFFFFu : ((1u << warp_threads) - 1u);
  OUT_T v = sum;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(warp_mask, v, off);
  }
  __shared__ OUT_T warp_sums[8];  // up to 256 threads -> up to 8 warps.
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  OUT_T block_sum = ((int)threadIdx.x < nwarps) ? warp_sums[(int)threadIdx.x] : (OUT_T)0;
  if (warp == 0) {
    // All of warp 0's 32 lanes are always present (blockDim.x >= 1 → nwarps >= 1 → warp 0 full if blockDim.x >= 32).
    unsigned warp0_threads = min(32, (int)blockDim.x);
    unsigned warp0_mask = (warp0_threads == 32) ? 0xFFFFFFFFu : ((1u << warp0_threads) - 1u);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block_sum += __shfl_down_sync(warp0_mask, block_sum, off);
    }
  }
  if ((int)threadIdx.x == 0) {
    OUT_T out = block_sum;
    if (add) y[csf_i] += out;
    else y[csf_i] = out;
  }
}

template <typename INDPTR_T, typename PQ_T>
__global__ void guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_indptr,  // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [nnz]
    const PQ_T* __restrict__ epq_pq,          // [nnz] (pq_id = p*norb+q, p!=q)
    const float* __restrict__ epq_data,       // [nnz]
    const int32_t* __restrict__ task_csf,     // [ntasks]
    const float* __restrict__ task_scale,     // [ntasks] or NULL
    const float* __restrict__ task_g,         // [nops] or [ntasks,nops]
    int64_t g_stride,                         // 0 (shared) or nops
    int ntasks,
    float* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  float scale = task_scale ? task_scale[task_id] : 1.0f;
  if (scale == 0.0f) return;

  int nops = norb * norb;
  const float* g_flat = task_g + (int64_t)task_id * g_stride;

  // Diagonal (p==q) contribution: sum once per task to reduce atomics.
  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_idx * (int64_t)norb;
    float diag = 0.0f;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      float wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != 0.0f) diag += wgt_pp * (float)occ_p;
    }
    if (diag != 0.0f) atomicAdd(&y[csf_idx], scale * diag);
  }

  int64_t start = epq_indptr[(int64_t)csf_idx];
  int64_t end = epq_indptr[(int64_t)csf_idx + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    float w = epq_data[t];
    if (w == 0.0f) continue;
    float wgt = g_flat[pq];
    if (wgt == 0.0f) continue;
    atomicAdd(&y[csf_i], scale * wgt * w);
  }
}

// Mixed-type scatter kernel: OUT_T for y/g/scale, COEF_T for epq_data.
// Supports (f64,f64), (f32,f32), and (f64,f32) via template instantiation.
template <typename INDPTR_T, typename OUT_T, typename COEF_T, typename PQ_T>
__global__ void guga_apply_g_flat_scatter_atomic_epq_table_mixed_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_indptr,  // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [nnz]
    const PQ_T* __restrict__ epq_pq,          // [nnz]
    const COEF_T* __restrict__ epq_data,      // [nnz]
    const int32_t* __restrict__ task_csf,     // [ntasks]
    const OUT_T* __restrict__ task_scale,     // [ntasks] or NULL
    const OUT_T* __restrict__ task_g,         // [nops] or [ntasks,nops]
    int64_t g_stride,                         // 0 (shared) or nops
    int ntasks,
    OUT_T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  OUT_T scale = task_scale ? task_scale[task_id] : (OUT_T)1.0;
  if (scale == (OUT_T)0.0) return;

  int nops = norb * norb;
  const OUT_T* g_flat = task_g + (int64_t)task_id * g_stride;

  // Diagonal (p==q) contribution
  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_idx * (int64_t)norb;
    OUT_T diag = (OUT_T)0.0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      OUT_T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != (OUT_T)0.0) diag += wgt_pp * (OUT_T)occ_p;
    }
    if (diag != (OUT_T)0.0) atomicAdd(&y[csf_idx], scale * diag);
  }

  int64_t start = epq_indptr[(int64_t)csf_idx];
  int64_t end = epq_indptr[(int64_t)csf_idx + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T w = (OUT_T)epq_data[t];
    if (w == (OUT_T)0.0) continue;
    OUT_T wgt = g_flat[pq];
    if (wgt == (OUT_T)0.0) continue;
    atomicAdd(&y[csf_i], scale * wgt * w);
  }
}

template <typename FP_T, typename PQ_T, bool USE_KAHAN = false>
__global__ __launch_bounds__(256, 2) void guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const int64_t* __restrict__ local_indptr,  // [j_count+1]
    const int32_t* __restrict__ epq_indices,   // [tile_nnz]
    const PQ_T* __restrict__ epq_pq,           // [tile_nnz]
    const FP_T* __restrict__ epq_data,         // [tile_nnz]
    const FP_T* __restrict__ task_g,           // [nops] or [j_count,nops]
    int64_t g_stride,                          // 0 (shared) or nops
    const FP_T* __restrict__ task_scale,       // [j_count] or NULL
    int j_start,
    int j_count,
    FP_T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int j_local = (int)blockIdx.x;
  if (j_local < 0 || j_local >= j_count) return;

  int csf_j = j_start + j_local;
  if ((unsigned)csf_j >= (unsigned)ncsf) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  FP_T scale = task_scale ? task_scale[j_local] : (FP_T)1;
  if (scale == (FP_T)0) return;

  int nops = norb * norb;
  const FP_T* g_flat = task_g + (int64_t)j_local * g_stride;

  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_j * (int64_t)norb;
    FP_T diag = (FP_T)0;
    FP_T diag_comp = (FP_T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      FP_T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp == (FP_T)0) continue;
      if constexpr (USE_KAHAN) {
        kahan_add(diag, diag_comp, wgt_pp * (FP_T)occ_p);
      } else {
        diag += wgt_pp * (FP_T)occ_p;
      }
    }
    FP_T diag_final = USE_KAHAN ? (diag + diag_comp) : diag;
    if (diag_final != (FP_T)0) atomicAdd(&y[csf_j], scale * diag_final);
  }

  int64_t start = local_indptr[(int64_t)j_local];
  int64_t end = local_indptr[(int64_t)j_local + 1];
  if (start < 0 || end < start) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    FP_T coef = epq_data[t];
    if (coef == (FP_T)0) continue;
    FP_T wgt = g_flat[pq];
    if (wgt == (FP_T)0) continue;
    atomicAdd(&y[csf_i], scale * wgt * coef);
  }
}

// Mixed-type tile scatter: OUT_T for y/g/scale, COEF_T for epq_data.
template <typename OUT_T, typename COEF_T, typename PQ_T>
__global__ __launch_bounds__(256, 2) void guga_apply_g_flat_scatter_atomic_epq_table_tile_mixed_kernel_t(
    const int8_t* __restrict__ steps_table,
    int ncsf,
    int norb,
    const int64_t* __restrict__ local_indptr,
    const int32_t* __restrict__ epq_indices,
    const PQ_T* __restrict__ epq_pq,
    const COEF_T* __restrict__ epq_data,
    const OUT_T* __restrict__ task_g,
    int64_t g_stride,
    const OUT_T* __restrict__ task_scale,
    int j_start,
    int j_count,
    OUT_T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  int j_local = (int)blockIdx.x;
  if (j_local < 0 || j_local >= j_count) return;

  int csf_j = j_start + j_local;
  if ((unsigned)csf_j >= (unsigned)ncsf) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  OUT_T scale = task_scale ? task_scale[j_local] : (OUT_T)1;
  if (scale == (OUT_T)0) return;

  int nops = norb * norb;
  const OUT_T* g_flat = task_g + (int64_t)j_local * g_stride;

  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_j * (int64_t)norb;
    OUT_T diag = (OUT_T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      OUT_T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp == (OUT_T)0) continue;
      diag += wgt_pp * (OUT_T)occ_p;
    }
    if (diag != (OUT_T)0) atomicAdd(&y[csf_j], scale * diag);
  }

  int64_t start = local_indptr[(int64_t)j_local];
  int64_t end = local_indptr[(int64_t)j_local + 1];
  if (start < 0 || end < start) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T coef = (OUT_T)epq_data[t];
    if (coef == (OUT_T)0) continue;
    OUT_T wgt = g_flat[pq];
    if (wgt == (OUT_T)0) continue;
    atomicAdd(&y[csf_i], scale * wgt * coef);
  }
}

template <typename INDPTR_T, typename PQ_T>
__global__ void guga_apply_g_flat_gather_epq_table_f32_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_t_indptr,  // [ncsf+1], rows by destination i
    const int32_t* __restrict__ epq_t_source,   // [nnz], source j
    const PQ_T* __restrict__ epq_t_pq,          // [nnz], pq_id = p*norb+q
    const float* __restrict__ epq_t_data,       // [nnz]
    const int32_t* __restrict__ task_row_by_csf,    // [ncsf], -1 if inactive; row index for task_g when g_stride>0
    const float* __restrict__ task_scale_by_csf,    // [ncsf] or NULL
    const float* __restrict__ task_g,               // [nops] or [ntasks,nops]
    int64_t g_stride,                               // 0 (shared) or nops
    float* __restrict__ y,
    int* __restrict__ overflow_flag,
    int add) {
  int csf_i = (int)blockIdx.x;
  if (csf_i >= ncsf) return;

  int nops = norb * norb;
  float sum = 0.0f;

  // Diagonal contribution from active source j == i.
  int row_i = task_row_by_csf[csf_i];
  if (row_i >= 0) {
    float scale_i = task_scale_by_csf ? task_scale_by_csf[csf_i] : 1.0f;
    if (scale_i != 0.0f) {
      const float* g_flat_i = task_g + (int64_t)row_i * g_stride;
      const int8_t* steps_i = steps_table + (int64_t)csf_i * (int64_t)norb;
      for (int p = (int)threadIdx.x; p < norb; p += (int)blockDim.x) {
        int occ_p = step_to_occ(steps_i[p]);
        if (!occ_p) continue;
        int pq = p * norb + p;
        float wgt = g_flat_i[pq];
        if (wgt != 0.0f) sum += scale_i * (float)occ_p * wgt;
      }
    }
  }

  int64_t start = epq_t_indptr[(int64_t)csf_i];
  int64_t end = epq_t_indptr[(int64_t)csf_i + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_j = epq_t_source[t];
    if ((unsigned)csf_j >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int row_j = task_row_by_csf[csf_j];
    if (row_j < 0) continue;

    int pq = (int)epq_t_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }

    float coef = epq_t_data[t];
    if (coef == 0.0f) continue;

    float scale_j = task_scale_by_csf ? task_scale_by_csf[csf_j] : 1.0f;
    if (scale_j == 0.0f) continue;

    const float* g_flat_j = task_g + (int64_t)row_j * g_stride;
    float wgt = g_flat_j[pq];
    if (wgt == 0.0f) continue;
    sum += scale_j * wgt * coef;
  }

  // Force warp reconvergence after divergent EPQ loop (Volta+ ITS).
  __syncwarp();

  int lane = (int)threadIdx.x & 31;
  int warp = (int)threadIdx.x >> 5;
  int nwarps = ((int)blockDim.x + 31) >> 5;
  int warp_threads = min(32, (int)blockDim.x - warp * 32);
  unsigned warp_mask = (warp_threads == 32) ? 0xFFFFFFFFu : ((1u << warp_threads) - 1u);
  double v = (double)sum;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(warp_mask, v, off);
  }
  __shared__ double warp_sums[8];  // up to 256 threads -> up to 8 warps.
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  double block_sum = ((int)threadIdx.x < nwarps) ? warp_sums[(int)threadIdx.x] : 0.0;
  if (warp == 0) {
    unsigned warp0_threads = min(32, (int)blockDim.x);
    unsigned warp0_mask = (warp0_threads == 32) ? 0xFFFFFFFFu : ((1u << warp0_threads) - 1u);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block_sum += __shfl_down_sync(warp0_mask, block_sum, off);
    }
  }
  if ((int)threadIdx.x == 0) {
    float out = (float)block_sum;
    if (add) y[csf_i] += out;
    else y[csf_i] = out;
  }
}

// Kahan-compensated variant of the FP32 gather kernel.
// Uses Neumaier compensated summation in the per-thread accumulation loop
// to reduce FP32 rounding error from O(N*ULP) to O(1*ULP).
template <typename INDPTR_T, typename PQ_T>
__global__ void guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_t_indptr,  // [ncsf+1], rows by destination i
    const int32_t* __restrict__ epq_t_source,   // [nnz], source j
    const PQ_T* __restrict__ epq_t_pq,          // [nnz], pq_id = p*norb+q
    const float* __restrict__ epq_t_data,       // [nnz]
    const int32_t* __restrict__ task_row_by_csf,    // [ncsf], -1 if inactive
    const float* __restrict__ task_scale_by_csf,    // [ncsf] or NULL
    const float* __restrict__ task_g,               // [nops] or [ntasks,nops]
    int64_t g_stride,                               // 0 (shared) or nops
    float* __restrict__ y,
    int* __restrict__ overflow_flag,
    int add) {
  int csf_i = (int)blockIdx.x;
  if (csf_i >= ncsf) return;

  int nops = norb * norb;
  float sum = 0.0f;
  float comp = 0.0f;  // Kahan compensation term

  // Diagonal contribution from active source j == i.
  int row_i = task_row_by_csf[csf_i];
  if (row_i >= 0) {
    float scale_i = task_scale_by_csf ? task_scale_by_csf[csf_i] : 1.0f;
    if (scale_i != 0.0f) {
      const float* g_flat_i = task_g + (int64_t)row_i * g_stride;
      const int8_t* steps_i = steps_table + (int64_t)csf_i * (int64_t)norb;
      for (int p = (int)threadIdx.x; p < norb; p += (int)blockDim.x) {
        int occ_p = step_to_occ(steps_i[p]);
        if (!occ_p) continue;
        int pq = p * norb + p;
        float wgt = g_flat_i[pq];
        if (wgt != 0.0f) kahan_add(sum, comp, scale_i * (float)occ_p * wgt);
      }
    }
  }

  int64_t start = epq_t_indptr[(int64_t)csf_i];
  int64_t end = epq_t_indptr[(int64_t)csf_i + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_j = epq_t_source[t];
    if ((unsigned)csf_j >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int row_j = task_row_by_csf[csf_j];
    if (row_j < 0) continue;

    int pq = (int)epq_t_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }

    float coef = epq_t_data[t];
    if (coef == 0.0f) continue;

    float scale_j = task_scale_by_csf ? task_scale_by_csf[csf_j] : 1.0f;
    if (scale_j == 0.0f) continue;

    const float* g_flat_j = task_g + (int64_t)row_j * g_stride;
    float wgt = g_flat_j[pq];
    if (wgt == 0.0f) continue;
    kahan_add(sum, comp, scale_j * wgt * coef);
  }

  // Fold compensation into sum before warp/block reduction.
  // The tree reduction has only ~8 steps so plain summation is fine there.
  sum += comp;

  // Force warp reconvergence after divergent EPQ loop (Volta+ ITS).
  __syncwarp();

  int lane = (int)threadIdx.x & 31;
  int warp = (int)threadIdx.x >> 5;
  int nwarps = ((int)blockDim.x + 31) >> 5;
  int warp_threads = min(32, (int)blockDim.x - warp * 32);
  unsigned warp_mask = (warp_threads == 32) ? 0xFFFFFFFFu : ((1u << warp_threads) - 1u);
  float v = sum;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(warp_mask, v, off);
  }
  __shared__ float warp_sums[8];  // up to 256 threads -> up to 8 warps.
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  float block_sum = ((int)threadIdx.x < nwarps) ? warp_sums[(int)threadIdx.x] : 0.0f;
  if (warp == 0) {
    unsigned warp0_threads = min(32, (int)blockDim.x);
    unsigned warp0_mask = (warp0_threads == 32) ? 0xFFFFFFFFu : ((1u << warp0_threads) - 1u);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block_sum += __shfl_down_sync(warp0_mask, block_sum, off);
    }
  }
  if ((int)threadIdx.x == 0) {
    float out = block_sum;
    if (add) y[csf_i] += out;
    else y[csf_i] = out;
  }
}

template <typename INDPTR_T, typename OUT_T, typename COEF_T, typename PQ_T>
__global__ void guga_build_w_from_epq_transpose_range_kernel_t(
    const INDPTR_T* __restrict__ epq_t_indptr,  // [ncsf+1], rows by destination k
    const int32_t* __restrict__ epq_t_source,   // [nnz], source j
    const PQ_T* __restrict__ epq_t_pq,          // [nnz]
    const COEF_T* __restrict__ epq_t_data,      // [nnz]
    const OUT_T* __restrict__ x,                // [ncsf]
    int ncsf,
    int nops,
    OUT_T* __restrict__ w_out,                  // [k_count,w_stride]
    int64_t w_stride,
    int* __restrict__ overflow_flag,
    int k_start,
    int k_count) {
  int k_local = (int)blockIdx.x;
  if (k_local < 0 || k_local >= k_count) return;
  int csf_k = k_start + k_local;
  if ((unsigned)csf_k >= (unsigned)ncsf) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  extern __shared__ unsigned char smem[];
  OUT_T* w_s = reinterpret_cast<OUT_T*>(smem);
  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    w_s[pq] = (OUT_T)0;
  }
  __syncthreads();

  int64_t start = epq_t_indptr[(int64_t)csf_k];
  int64_t end = epq_t_indptr[(int64_t)csf_k + 1];
  if (start < 0 || end < start) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_j = epq_t_source[t];
    if ((unsigned)csf_j >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_t_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T coef = (OUT_T)epq_t_data[t];
    if (coef == (OUT_T)0) continue;
    OUT_T xj = x[csf_j];
    if (xj == (OUT_T)0) continue;
    atomicAdd(&w_s[pq], xj * coef);
  }
  __syncthreads();

  OUT_T* w_row = w_out + (int64_t)k_local * w_stride;
  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    w_row[pq] += w_s[pq];
  }
}

template <typename INDPTR_T, typename OUT_T, typename COEF_T, typename PQ_T, bool USE_KAHAN = false>
__global__ void guga_apply_g_flat_gather_epq_transpose_range_kernel_t(
    const int8_t* __restrict__ steps_table,     // [ncsf,norb]
    int ncsf,
    int norb,
    const INDPTR_T* __restrict__ epq_t_indptr,  // [ncsf+1], rows by destination i
    const int32_t* __restrict__ epq_t_source,   // [nnz], source k
    const PQ_T* __restrict__ epq_t_pq,          // [nnz]
    const COEF_T* __restrict__ epq_t_data,      // [nnz]
    const OUT_T* __restrict__ g_block,          // [k_count,g_stride]
    int64_t g_stride,
    int k_start,
    int k_count,
    OUT_T* __restrict__ y,                      // [ncsf]
    int* __restrict__ overflow_flag,
    int add) {
  int csf_i = (int)blockIdx.x;
  if (csf_i < 0 || csf_i >= ncsf) return;

  int nops = norb * norb;
  OUT_T sum = (OUT_T)0;
  OUT_T comp = (OUT_T)0;

  if (csf_i >= k_start && csf_i < (k_start + k_count)) {
    int k_local = csf_i - k_start;
    const OUT_T* g_row = g_block + (int64_t)k_local * g_stride;
    const int8_t* steps_i = steps_table + (int64_t)csf_i * (int64_t)norb;
    for (int p = (int)threadIdx.x; p < norb; p += (int)blockDim.x) {
      int occ_p = step_to_occ(steps_i[p]);
      if (!occ_p) continue;
      OUT_T wgt = g_row[p * norb + p];
      if (wgt == (OUT_T)0) continue;
      if constexpr (USE_KAHAN) {
        kahan_add(sum, comp, (OUT_T)occ_p * wgt);
      } else {
        sum += (OUT_T)occ_p * wgt;
      }
    }
  }

  int64_t start = epq_t_indptr[(int64_t)csf_i];
  int64_t end = epq_t_indptr[(int64_t)csf_i + 1];
  if (start < 0 || end < start) {
    if ((int)threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_k = epq_t_source[t];
    if ((unsigned)csf_k >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    if (csf_k < k_start || csf_k >= (k_start + k_count)) continue;

    int pq = (int)epq_t_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T coef = (OUT_T)epq_t_data[t];
    if (coef == (OUT_T)0) continue;
    OUT_T wgt = g_block[(int64_t)(csf_k - k_start) * g_stride + (int64_t)pq];
    if (wgt == (OUT_T)0) continue;
    if constexpr (USE_KAHAN) {
      kahan_add(sum, comp, wgt * coef);
    } else {
      sum += wgt * coef;
    }
  }

  if constexpr (USE_KAHAN) {
    sum += comp;
  }

  __syncwarp();
  int lane = (int)threadIdx.x & 31;
  int warp = (int)threadIdx.x >> 5;
  int nwarps = ((int)blockDim.x + 31) >> 5;
  int warp_threads = min(32, (int)blockDim.x - warp * 32);
  unsigned warp_mask = (warp_threads == 32) ? 0xFFFFFFFFu : ((1u << warp_threads) - 1u);
  OUT_T v = sum;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(warp_mask, v, off);
  }
  __shared__ OUT_T warp_sums[8];  // up to 256 threads
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  OUT_T block_sum = ((int)threadIdx.x < nwarps) ? warp_sums[(int)threadIdx.x] : (OUT_T)0;
  if (warp == 0) {
    unsigned warp0_threads = min(32, (int)blockDim.x);
    unsigned warp0_mask = (warp0_threads == 32) ? 0xFFFFFFFFu : ((1u << warp0_threads) - 1u);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      block_sum += __shfl_down_sync(warp0_mask, block_sum, off);
    }
  }
  if ((int)threadIdx.x == 0) {
    if (add) y[csf_i] += block_sum;
    else y[csf_i] = block_sum;
  }
}

template <typename OUT_T, typename COEF_T, typename PQ_T, bool USE_KAHAN = false>
__global__ void guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const int64_t* __restrict__ epq_indptr,   // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [epq_nnz]
    const PQ_T* __restrict__ epq_pq,          // [epq_nnz] (pq_id = p*norb+q, p!=q)
    const COEF_T* __restrict__ epq_data,      // [epq_nnz]
    const int32_t* __restrict__ row_j,        // [nrows_total]
    const int32_t* __restrict__ row_k,        // [nrows_total]
    const int64_t* __restrict__ csr_indptr,   // [nrows_total+1]
    const int32_t* __restrict__ csr_indices,  // [csr_nnz_total]
    const OUT_T* __restrict__ csr_data,      // [csr_nnz_total]
    int row_start,
    int nrows,
    const OUT_T* __restrict__ eri_mat_t,  // [nops,nops] row-major, eri_mat_t[rs*nops + pq] = eri_mat[pq,rs]
    int nops,
    OUT_T half,
    const OUT_T* __restrict__ x,  // [ncsf]
    OUT_T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  int row_local = (int)blockIdx.x;
  if (row_local >= nrows) return;
  int row = row_start + row_local;
  if (row < 0) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_j = row_j[row];
  int csf_k = row_k[row];
  if ((unsigned)csf_j >= (unsigned)ncsf || (unsigned)csf_k >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  OUT_T scale = x[csf_j];
  if (scale == (OUT_T)0) return;
  // Fold the per-row `x[j]` scale into the shared `g[pq]` vector once, so the hot epq-table
  // scatter loop can avoid one multiply per atomicAdd.
  OUT_T scale_half = scale * half;

  int64_t csr_start = csr_indptr[row];
  int64_t csr_end = csr_indptr[row + 1];
  if (csr_start < 0 || csr_end < csr_start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  extern __shared__ unsigned char smem[];
  OUT_T* g_row_s = reinterpret_cast<OUT_T*>(smem);

  // Build g_flat[pq] for this row in shared memory:
  //   g[pq] = half * sum_rs C[rs] * ERI_mat[pq,rs] = half * sum_rs C[rs] * ERI_mat_t[rs,pq]
  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    OUT_T acc = (OUT_T)0;
    OUT_T acc_comp = (OUT_T)0;
    for (int64_t t = csr_start; t < csr_end; t++) {
      int32_t rs = csr_indices[t];
      OUT_T c = csr_data[t];
      if ((unsigned)rs >= (unsigned)nops) {
        atomicExch(overflow_flag, 1);
        continue;
      }
      if constexpr (USE_KAHAN) {
        kahan_add(acc, acc_comp, eri_mat_t[(int64_t)rs * (int64_t)nops + (int64_t)pq] * c);
      } else {
        acc += eri_mat_t[(int64_t)rs * (int64_t)nops + (int64_t)pq] * c;
      }
    }
    if constexpr (USE_KAHAN) {
      g_row_s[pq] = scale_half * (acc + acc_comp);
    } else {
      g_row_s[pq] = scale_half * acc;
    }
  }
  if ((int)blockDim.x > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }

  // Diagonal (p==q) contribution to y[csf_k].
  if ((int)threadIdx.x == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_k * (int64_t)norb;
    OUT_T diag = (OUT_T)0;
    OUT_T diag_comp = (OUT_T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      OUT_T wgt_pp = g_row_s[p * norb + p];
      if (wgt_pp != (OUT_T)0) {
        if constexpr (USE_KAHAN) {
          kahan_add(diag, diag_comp, wgt_pp * (OUT_T)occ_p);
        } else {
          diag += wgt_pp * (OUT_T)occ_p;
        }
      }
    }
    OUT_T diag_final = USE_KAHAN ? (diag + diag_comp) : diag;
    if (diag_final != (OUT_T)0) atomicAdd(&y[csf_k], diag_final);
  }

  int64_t start = epq_indptr[(int64_t)csf_k];
  int64_t end = epq_indptr[(int64_t)csf_k + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T w = (OUT_T)epq_data[t];
    if (w == (OUT_T)0) continue;
    OUT_T wgt = g_row_s[pq];
    if (wgt == (OUT_T)0) continue;
    atomicAdd(&y[csf_i], wgt * w);
  }
}

// 10.16.6: __launch_bounds__ for hot warp-cooperative kernel.
template <typename OUT_T, typename COEF_T, typename PQ_T, bool USE_KAHAN = false>
__global__ __launch_bounds__(256, 2)
void guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    const int64_t* __restrict__ epq_indptr,   // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [epq_nnz]
    const PQ_T* __restrict__ epq_pq,          // [epq_nnz] (pq_id = p*norb+q, p!=q)
    const COEF_T* __restrict__ epq_data,      // [epq_nnz]
    const int32_t* __restrict__ row_j,        // [nrows_total]
    const int32_t* __restrict__ row_k,        // [nrows_total]
    const int64_t* __restrict__ csr_indptr,   // [nrows_total+1]
    const int32_t* __restrict__ csr_indices,  // [csr_nnz_total]
    const OUT_T* __restrict__ csr_data,      // [csr_nnz_total]
    int row_start,
    int nrows,
    const OUT_T* __restrict__ eri_mat_t,  // [nops,nops] row-major, eri_mat_t[rs*nops + pq] = eri_mat[pq,rs]
    int nops,
    OUT_T half,
    const OUT_T* __restrict__ x,  // [ncsf]
    OUT_T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  // Warp-cooperative: each warp processes one CSR row, and each block processes `warps_per_block` rows.
  int lane = (int)threadIdx.x & 31;
  int warp_id = (int)threadIdx.x >> 5;
  int warps_per_block = (int)blockDim.x >> 5;
  if (warps_per_block <= 0) return;

  int row_local = (int)blockIdx.x * warps_per_block + warp_id;
  if (row_local >= nrows) return;

  int row = row_start + row_local;
  if (row < 0) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_j = row_j[row];
  int csf_k = row_k[row];
  if ((unsigned)csf_j >= (unsigned)ncsf || (unsigned)csf_k >= (unsigned)ncsf) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  OUT_T scale = x[csf_j];
  if (scale == (OUT_T)0) return;
  // Fold the per-row `x[j]` scale into g[pq] once.
  OUT_T scale_half = scale * half;

  int64_t csr_start = csr_indptr[row];
  int64_t csr_end = csr_indptr[row + 1];
  if (csr_start < 0 || csr_end < csr_start) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  extern __shared__ unsigned char smem[];
  OUT_T* g = reinterpret_cast<OUT_T*>(smem) + (int64_t)warp_id * (int64_t)nops;

  // Build g[pq] for this row.
  // Fast path: nnz==1 (common for small CAS).
  int64_t nnz = csr_end - csr_start;
  if (nnz == 1) {
    int32_t rs = csr_indices[csr_start];
    OUT_T c = csr_data[csr_start];
    if ((unsigned)rs >= (unsigned)nops) {
      if (lane == 0) atomicExch(overflow_flag, 1);
      return;
    }
    for (int pq = lane; pq < nops; pq += 32) {
      g[pq] = scale_half * (eri_mat_t[(int64_t)rs * (int64_t)nops + (int64_t)pq] * c);
    }
  } else {
    for (int pq = lane; pq < nops; pq += 32) {
      OUT_T acc = (OUT_T)0;
      OUT_T acc_comp = (OUT_T)0;
      for (int64_t t = csr_start; t < csr_end; t++) {
        int32_t rs = csr_indices[t];
        OUT_T c = csr_data[t];
        if ((unsigned)rs >= (unsigned)nops) {
          atomicExch(overflow_flag, 1);
          continue;
        }
        if constexpr (USE_KAHAN) {
          kahan_add(acc, acc_comp, eri_mat_t[(int64_t)rs * (int64_t)nops + (int64_t)pq] * c);
        } else {
          acc += eri_mat_t[(int64_t)rs * (int64_t)nops + (int64_t)pq] * c;
        }
      }
      if constexpr (USE_KAHAN) {
        g[pq] = scale_half * (acc + acc_comp);
      } else {
        g[pq] = scale_half * acc;
      }
    }
  }
  __syncwarp();

  // Diagonal (p==q) contribution to y[csf_k].
  if (lane == 0) {
    const int8_t* steps = steps_table + (int64_t)csf_k * (int64_t)norb;
    OUT_T diag = (OUT_T)0;
    OUT_T diag_comp = (OUT_T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = step_to_occ(steps[p]);
      if (!occ_p) continue;
      OUT_T wgt_pp = g[p * norb + p];
      if (wgt_pp != (OUT_T)0) {
        if constexpr (USE_KAHAN) {
          kahan_add(diag, diag_comp, wgt_pp * (OUT_T)occ_p);
        } else {
          diag += wgt_pp * (OUT_T)occ_p;
        }
      }
    }
    OUT_T diag_final = USE_KAHAN ? (diag + diag_comp) : diag;
    if (diag_final != (OUT_T)0) atomicAdd(&y[csf_k], diag_final);
  }

  int64_t start = epq_indptr[(int64_t)csf_k];
  int64_t end = epq_indptr[(int64_t)csf_k + 1];
  if (start < 0 || end < start) {
    if (lane == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)lane; t < end; t += 32) {
    int32_t csf_i = epq_indices[t];
    if ((unsigned)csf_i >= (unsigned)ncsf) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    int pq = (int)epq_pq[t];
    if ((unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    OUT_T w = (OUT_T)epq_data[t];
    if (w == (OUT_T)0) continue;
    OUT_T wgt = g[pq];
    if (wgt == (OUT_T)0) continue;
    atomicAdd(&y[csf_i], wgt * w);
  }
}

__global__ void guga_build_w_from_csr_unitnnz_kernel(
    const int32_t* __restrict__ row_j,       // [nrows]
    const int32_t* __restrict__ row_k,       // [nrows]
    const int32_t* __restrict__ csr_rs,      // [nrows]
    const double* __restrict__ csr_c,        // [nrows]
    int nrows,
    const double* __restrict__ x,  // [ncsf]
    int ncsf,
    int nops,
    double* __restrict__ w_out,  // [ncsf,w_stride]
    int64_t w_stride,
    int* __restrict__ overflow_flag) {
  int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx >= nrows) return;

  int csf_j = row_j[idx];
  int csf_k = row_k[idx];
  int rs = csr_rs[idx];
  double c = csr_c[idx];

  if ((unsigned)csf_j >= (unsigned)ncsf || (unsigned)csf_k >= (unsigned)ncsf || (unsigned)rs >= (unsigned)nops) {
    atomicExch(overflow_flag, 1);
    return;
  }

  double scale = x[csf_j];
  if (scale == 0.0) return;
  if (c == 0.0) return;

  atomicAdd(&w_out[(int64_t)csf_k * (int64_t)w_stride + (int64_t)rs], scale * c);
}

template <typename INDPTR_T, typename OUT_T, typename COEF_T, typename PQ_T>
__global__ void guga_build_w_from_epq_table_kernel_t(
    const INDPTR_T* __restrict__ epq_indptr,  // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [epq_nnz]
    const PQ_T* __restrict__ epq_pq,          // [epq_nnz]
    const COEF_T* __restrict__ epq_data,      // [epq_nnz]
    const OUT_T* __restrict__ x,              // [ncsf]
    int ncsf,
    int nops,
    OUT_T* __restrict__ w_out,  // [k_count,w_stride]
    int64_t w_stride,
    int* __restrict__ overflow_flag,
    int k_start,
    int k_count) {
  int csf_j = (int)blockIdx.x;
  if (csf_j >= ncsf) return;

  OUT_T scale = x[csf_j];
  if (scale == (OUT_T)0) return;

  int64_t start = epq_indptr[(int64_t)csf_j];
  int64_t end = epq_indptr[(int64_t)csf_j + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_k = epq_indices[t];
    int pq = (int)epq_pq[t];
    OUT_T c = (OUT_T)epq_data[t];
    if (c == (OUT_T)0) continue;
    if ((unsigned)csf_k >= (unsigned)ncsf || (unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }

    // Tiling check
    if (k_count > 0) {
      if (csf_k < k_start || csf_k >= k_start + k_count) continue;
      int64_t k_offset = (int64_t)csf_k - (int64_t)k_start;
      atomicAdd(&w_out[k_offset * (int64_t)w_stride + (int64_t)pq], scale * c);
    } else {
      atomicAdd(&w_out[(int64_t)csf_k * (int64_t)w_stride + (int64_t)pq], scale * c);
    }
  }
}

template <typename T, typename PQ_T>
__global__ void guga_build_t_from_epq_table_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    const int64_t* __restrict__ epq_indptr,   // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [epq_nnz]
    const PQ_T* __restrict__ epq_pq,          // [epq_nnz]
    const T* __restrict__ epq_data,           // [epq_nnz]
    const T* __restrict__ c_vec,              // [ncsf]
    int ncsf,
    int norb,
    int nops,
    T* __restrict__ t_out,  // [nops,t_stride]
    int64_t t_stride,
    int* __restrict__ overflow_flag) {
  int csf_j = (int)blockIdx.x;
  if (csf_j >= ncsf) return;

  T scale = c_vec[csf_j];
  if (scale == 0.0) return;

  // Diagonal operators: (E_pp |c>)[j] = occ_p(j) * c[j]
  for (int p = (int)threadIdx.x; p < norb; p += (int)blockDim.x) {
    int8_t step = steps_table[(int64_t)csf_j * (int64_t)norb + (int64_t)p];
    int occ = (int)step_to_occ(step);
    if (occ) {
      int pq = p * norb + p;
      t_out[(int64_t)pq * (int64_t)t_stride + (int64_t)csf_j] = (T)occ * scale;
    }
  }

  // Off-diagonal operators from the precomputed table:
  //   E_pq |j> = sum_t coeff[t] |indices[t]>  (p!=q)
  // so:
  //   T[pq, k] += c[j] * coeff(j->k,pq)
  int64_t start = epq_indptr[(int64_t)csf_j];
  int64_t end = epq_indptr[(int64_t)csf_j + 1];
  if (start < 0 || end < start) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t csf_k = epq_indices[t];
    int pq = (int)epq_pq[t];
    T coef = epq_data[t];
    if (coef == 0.0) continue;
    if ((unsigned)csf_k >= (unsigned)ncsf || (unsigned)pq >= (unsigned)nops) {
      atomicExch(overflow_flag, 1);
      continue;
    }
    atomicAdd(&t_out[(int64_t)pq * (int64_t)t_stride + (int64_t)csf_k], scale * coef);
  }
}

template <int MAX_NORB_T>
__global__ void guga_apply_g_flat_task_sums_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,    // [ntasks]
    const double* __restrict__ task_scale,  // [ntasks] or NULL
    const double* __restrict__ task_g,      // [nops] or [ntasks,nops]
    int64_t g_stride,                       // 0 (shared) or nops
    int ntasks,
    double* __restrict__ out_sum,  // [ntasks]
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  int threads = (int)blockDim.x;
  if (threads <= 0 || threads > 256 || (threads & (threads - 1)) != 0) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  double scale = task_scale ? task_scale[task_id] : 1.0;
  if (scale == 0.0) {
    if (threadIdx.x == 0) out_sum[task_id] = 0.0;
    return;
  }

  int nops = norb * norb;
  const double* g_flat = task_g + (int64_t)task_id * g_stride;

  __shared__ int8_t steps_s[MAX_NORB_T];
  __shared__ int32_t nodes_s[MAX_NORB_T + 1];
  __shared__ int8_t occ_s[MAX_NORB_T];
  __shared__ int16_t b_s[MAX_NORB_T];
  __shared__ int32_t idx_prefix_s[MAX_NORB_T + 1];
  __shared__ int32_t idx_prefix_warp_sums[(MAX_NORB_T + 31) / 32];

  for (int k = (int)threadIdx.x; k < norb; k += (int)blockDim.x) {
    int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)k];
    steps_s[k] = step;
    occ_s[k] = (int8_t)step_to_occ(step);
    int32_t node_next = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1) + (int64_t)(k + 1)];
    nodes_s[k + 1] = node_next;
    b_s[k] = node_twos[node_next];
  }
  if ((int)blockDim.x > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
  if ((int)threadIdx.x == 0) {
    nodes_s[0] = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1)];
  }
  if ((int)blockDim.x > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
  if (norb <= 32) {
    int lane = (int)threadIdx.x;
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
    if ((int)blockDim.x > 32) {
      __syncthreads();
    } else {
      __syncwarp();
    }
  } else {
    if ((int)blockDim.x >= 64) {
      int tid = (int)threadIdx.x;
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
      if (warp < warps_needed && warp > 0) {
        warp_offset = idx_prefix_warp_sums[0];
      }
      if (tid == 0) idx_prefix_s[0] = 0;
      if (warp < warps_needed && k < norb) idx_prefix_s[k + 1] = delta + warp_offset;
      __syncthreads();
    } else {
      if ((int)threadIdx.x == 0) {
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

  double sum = 0.0;

  if ((int)threadIdx.x == 0) {
    double diag = 0.0;
    for (int p = 0; p < norb; p++) {
      int occ_p = (int)occ_s[p];
      if (!occ_p) continue;
      double wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != 0.0) diag += wgt_pp * (double)occ_p;
    }
    sum += scale * diag;
  }

  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    double wgt = g_flat[pq];
    if (wgt == 0.0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start;
    int end;
    int q_start;
    int q_mid;
    int q_end;
    if (p < q) {
      start = p;
      end = q;
      q_start = Q_uR;
      q_mid = Q_R;
      q_end = Q_oR;
    } else {
      start = q;
      end = p;
      q_start = Q_uL;
      q_mid = Q_L;
      q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];

    int32_t prefix_offset = idx_prefix_s[start];
    int32_t prefix_endplus1 = idx_prefix_s[end + 1];
    int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

    int8_t st_k[MAX_NORB_T];
    int32_t st_node[MAX_NORB_T];
    double st_w[MAX_NORB_T];
    int32_t st_seg[MAX_NORB_T];
    int top = 0;

    st_k[top] = (int8_t)start;
    st_node[top] = node_start;
    st_w[top] = 1.0;
    st_seg[top] = 0;
    top++;

    int overflow = 0;

    while (top) {
      top--;
      int kpos = (int)st_k[top];
      int node_k = (int)st_node[top];
      double w = st_w[top];
      int32_t seg_idx = st_seg[top];

      int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);

      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
      int ndp = candidate_dprimes(qk, dk, &dp0, &dp1);
      if (ndp == 0) continue;

      for (int which = 0; which < ndp; which++) {
        int dprime = (which == 0) ? dp0 : dp1;
        int child_k = child[node_k * 4 + dprime];
        if (child_k < 0) continue;
        int bprime = (int)node_twos[child_k];
        int db = bk - bprime;
        double seg = segment_value_int(qk, dprime, dk, db, bk);
        if (seg == 0.0) continue;
        double w2 = w * seg;
        int32_t seg_idx2 = seg_idx + (int32_t)child_prefix[node_k * 5 + dprime];

        if (kpos == end) {
          if (child_k != node_end_target) continue;
          int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if ((unsigned)csf_i >= (unsigned)ncsf) {
            overflow = 1;
            continue;
          }
          if (csf_i == csf_idx) continue;
          if (w2 == 0.0) continue;
          sum += scale * wgt * w2;
        } else {
          if (top >= MAX_NORB_T) {
            overflow = 1;
            continue;
          }
          st_k[top] = (int8_t)k_next;
          st_node[top] = child_k;
          st_w[top] = w2;
          st_seg[top] = seg_idx2;
          top++;
        }
      }
    }

    if (overflow) atomicExch(overflow_flag, 1);
  }

  // Reduce per-thread sums within the block and emit one output value per task.
  if (threads <= 32) {
    int width = threads;  // power-of-two by check above
    unsigned mask = (width == 32) ? 0xffffffffu : ((1u << width) - 1u);
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset, width);
    }
    if ((int)threadIdx.x == 0) out_sum[task_id] = sum;
  } else {
    __shared__ double sum_s[256];
    sum_s[(int)threadIdx.x] = sum;
    __syncthreads();
    for (int offset = threads >> 1; offset > 0; offset >>= 1) {
      if ((int)threadIdx.x < offset) sum_s[(int)threadIdx.x] += sum_s[(int)threadIdx.x + offset];
      __syncthreads();
    }
    if ((int)threadIdx.x == 0) out_sum[task_id] = sum_s[0];
  }
}

template <int MAX_NORB_T, typename T>
__global__ void guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t(
    const int32_t* __restrict__ child,
    const int16_t* __restrict__ node_twos,
    const int64_t* __restrict__ child_prefix,
    const int8_t* __restrict__ steps_table,   // [ncsf,norb]
    const int32_t* __restrict__ nodes_table,  // [ncsf,norb+1]
    int ncsf,
    int norb,
    const int32_t* __restrict__ task_csf,    // [ntasks]
    const T* __restrict__ task_scale,  // [ntasks] or NULL
    const T* __restrict__ task_g,      // [nops] or [ntasks,nops]
    int64_t g_stride,                       // 0 (shared) or nops
    int ntasks,
    T* __restrict__ y,
    int* __restrict__ overflow_flag) {
  guga_maybe_enable_smem_spilling();

  static_assert(MAX_NORB_T <= 64, "MAX_NORB_T out of supported range");

  if ((int)blockDim.x != 32) return;

  int task_id = (int)blockIdx.x;
  if (task_id >= ntasks) return;

  if (norb > MAX_NORB_T) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  int csf_idx = task_csf[task_id];
  if ((unsigned)csf_idx >= (unsigned)ncsf) {
    if (threadIdx.x == 0) atomicExch(overflow_flag, 1);
    return;
  }

  T scale = task_scale ? task_scale[task_id] : (T)1;
  if (scale == (T)0) return;

  int nops = norb * norb;
  const T* g_flat = task_g + (int64_t)task_id * g_stride;

  __shared__ int8_t steps_s[MAX_NORB_T];
  __shared__ int32_t nodes_s[MAX_NORB_T + 1];
  __shared__ int8_t occ_s[MAX_NORB_T];
  __shared__ int16_t b_s[MAX_NORB_T];
  __shared__ int32_t idx_prefix_s[MAX_NORB_T + 1];

  constexpr int STACK_PITCH = MAX_NORB_T + 1;  // +1 padding avoids worst-case shared bank conflicts.
  __shared__ int8_t st_k_s[32 * STACK_PITCH];
  __shared__ int32_t st_node_s[32 * STACK_PITCH];
  __shared__ T st_w_s[32 * STACK_PITCH];
  __shared__ int32_t st_seg_s[32 * STACK_PITCH];

  int lane = (int)threadIdx.x;
  int base = lane * STACK_PITCH;

  for (int k = lane; k < norb; k += 32) {
    int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)k];
    steps_s[k] = step;
    occ_s[k] = (int8_t)step_to_occ(step);
    int32_t node_next = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1) + (int64_t)(k + 1)];
    nodes_s[k + 1] = node_next;
    b_s[k] = node_twos[node_next];
  }
  __syncwarp();
  if (lane == 0) {
    nodes_s[0] = nodes_table[(int64_t)csf_idx * (int64_t)(norb + 1)];
    idx_prefix_s[0] = 0;
  }
  __syncwarp();
  {
    // Warp-only inclusive scan over up to 64 elements:
    // - lanes 0..31 handle k=0..31
    // - lanes 0..31 also handle k=32..63 via a second scan
    int32_t delta0 = 0;
    if (lane < norb) {
      int node_k = nodes_s[lane];
      int step_k = (int)steps_s[lane];
      delta0 = (int32_t)child_prefix[node_k * 5 + step_k];
    }
    int32_t sum0 = delta0;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int32_t v = __shfl_up_sync(0xffffffffu, sum0, off);
      if (lane >= off) sum0 += v;
    }
    int32_t base = __shfl_sync(0xffffffffu, sum0, 31);

    int k1 = lane + 32;
    int32_t delta1 = 0;
    if (k1 < norb) {
      int node_k = nodes_s[k1];
      int step_k = (int)steps_s[k1];
      delta1 = (int32_t)child_prefix[node_k * 5 + step_k];
    }
    int32_t sum1 = delta1;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int32_t v = __shfl_up_sync(0xffffffffu, sum1, off);
      if (lane >= off) sum1 += v;
    }
    int32_t prefix1 = base + sum1;

    if (lane < norb) idx_prefix_s[lane + 1] = sum0;
    if (k1 < norb) idx_prefix_s[k1 + 1] = prefix1;
  }
  __syncwarp();

  // Diagonal (p==q): one atomicAdd per task.  Kahan-compensated to reduce FP32 error.
  if (lane == 0) {
    T diag = (T)0;
    T diag_c = (T)0;
    for (int p = 0; p < norb; p++) {
      int occ_p = (int)occ_s[p];
      if (!occ_p) continue;
      T wgt_pp = g_flat[p * norb + p];
      if (wgt_pp != (T)0) kahan_add(diag, diag_c, wgt_pp * (T)occ_p);
    }
    T diag_final = diag + diag_c;
    if (diag_final != (T)0) atomicAdd(&y[csf_idx], scale * diag_final);
  }

  for (int pq = lane; pq < nops; pq += 32) {
    int p = pq / norb;
    int q = pq - p * norb;
    if (p == q) continue;

    T wgt = g_flat[pq];
    if (wgt == (T)0) continue;

    int occ_p = (int)occ_s[p];
    int occ_q = (int)occ_s[q];
    if (occ_q <= 0 || occ_p >= 2) continue;

    int start;
    int end;
    int q_start;
    int q_mid;
    int q_end;
    if (p < q) {
      start = p;
      end = q;
      q_start = Q_uR;
      q_mid = Q_R;
      q_end = Q_oR;
    } else {
      start = q;
      end = p;
      q_start = Q_uL;
      q_mid = Q_L;
      q_end = Q_oL;
    }

    int32_t node_start = nodes_s[start];
    int32_t node_end_target = nodes_s[end + 1];

    int32_t prefix_offset = idx_prefix_s[start];
    int32_t prefix_endplus1 = idx_prefix_s[end + 1];
    int32_t suffix_offset = (int32_t)csf_idx - prefix_endplus1;

    int top = 0;
    st_k_s[base + top] = (int8_t)start;
    st_node_s[base + top] = node_start;
    st_w_s[base + top] = (T)1;
    st_seg_s[base + top] = 0;
    top++;

    int overflow = 0;

    while (top) {
      top--;
      int kpos = (int)st_k_s[base + top];
      int node_k = (int)st_node_s[base + top];
      T w = st_w_s[base + top];
      int32_t seg_idx = st_seg_s[base + top];

      int qk = (kpos == start) ? q_start : ((kpos == end) ? q_end : q_mid);

      int dk = (int)steps_s[kpos];
      int bk = (int)b_s[kpos];
      int k_next = kpos + 1;

      int dp0 = 0;
      int dp1 = 0;
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
          int32_t csf_i = prefix_offset + seg_idx2 + suffix_offset;
          if ((unsigned)csf_i >= (unsigned)ncsf) {
            overflow = 1;
            continue;
          }
          if (csf_i == csf_idx) continue;
          if (w2 == (T)0) continue;
          atomicAdd(&y[csf_i], scale * wgt * w2);
        } else {
          if (top >= MAX_NORB_T) {
            overflow = 1;
            continue;
          }
          st_k_s[base + top] = (int8_t)k_next;
          st_node_s[base + top] = child_k;
          st_w_s[base + top] = w2;
          st_seg_s[base + top] = seg_idx2;
          top++;
        }
      }
    }

    if (overflow) atomicExch(overflow_flag, 1);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_launch(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const int32_t* epq_pq,
    const double* epq_data,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
  int* overflow_flag,
  int threads) {
  int blocks = ntasks;
  guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int64_t, int32_t><<<blocks, threads>>>(
      steps_table, ncsf, norb, epq_indptr, epq_indices, epq_pq, epq_data, task_csf, task_scale, task_g, g_stride,
      ntasks, y, overflow_flag);
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const double* epq_data,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ntasks;
  if (epq_indptr_type == 4) {
    const int32_t* epq_indptr_i32 = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int32_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int32_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int32_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    }
  } else {
    const int64_t* epq_indptr_i64 = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int64_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int64_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else {
      guga_apply_g_flat_scatter_atomic_epq_table_kernel_t<int64_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    }
  }
}

extern "C" void guga_apply_g_flat_gather_epq_table_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const double* epq_t_data,
    const int32_t* task_row_by_csf,
    const double* task_scale_by_csf,
    const double* task_g,
    int64_t g_stride,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_kernel_t<int32_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_kernel_t<int32_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_table_kernel_t<int32_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_kernel_t<int64_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_kernel_t<int64_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_table_kernel_t<int64_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    }
  }
}

// Mixed-type launcher: OUT_T=double, COEF_T=float
extern "C" void guga_apply_g_flat_gather_epq_table_mixed_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const int32_t* task_row_by_csf,
    const double* task_scale_by_csf,
    const double* task_g,
    int64_t g_stride,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;

  #define LAUNCH_MIXED_GATHER(INDPTR_T, PQ_T, indptr_ptr, pq_ptr) \
    guga_apply_g_flat_gather_epq_table_kernel_t<INDPTR_T, double, float, PQ_T> \
        <<<blocks, threads, 0, stream>>>( \
            steps_table, ncsf, norb, indptr_ptr, epq_t_source, pq_ptr, epq_t_data, \
            task_row_by_csf, task_scale_by_csf, task_g, g_stride, y, overflow_flag, add)

  if (epq_t_indptr_type == 4) {
    const int32_t* ip = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      LAUNCH_MIXED_GATHER(int32_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_t_pq));
    } else if (epq_t_pq_type == 2) {
      LAUNCH_MIXED_GATHER(int32_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_t_pq));
    } else {
      LAUNCH_MIXED_GATHER(int32_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_t_pq));
    }
  } else {
    const int64_t* ip = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      LAUNCH_MIXED_GATHER(int64_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_t_pq));
    } else if (epq_t_pq_type == 2) {
      LAUNCH_MIXED_GATHER(int64_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_t_pq));
    } else {
      LAUNCH_MIXED_GATHER(int64_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_t_pq));
    }
  }

  #undef LAUNCH_MIXED_GATHER
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const int32_t* task_csf,
    const float* task_scale,
    const float* task_g,
    int64_t g_stride,
    int ntasks,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ntasks;
  if (epq_indptr_type == 4) {
    const int32_t* epq_indptr_i32 = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int32_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int32_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int32_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    }
  } else {
    const int64_t* epq_indptr_i64 = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int64_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int64_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    } else {
      guga_apply_g_flat_scatter_atomic_epq_table_f32_kernel_t<int64_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
    }
  }
}

// Mixed-type launcher: OUT_T=double, COEF_T=float
extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_mixed_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ntasks;

  #define LAUNCH_MIXED_SCATTER(INDPTR_T, PQ_T, indptr_ptr, pq_ptr) \
    guga_apply_g_flat_scatter_atomic_epq_table_mixed_kernel_t<INDPTR_T, double, float, PQ_T> \
        <<<blocks, threads, 0, stream>>>( \
        steps_table, ncsf, norb, indptr_ptr, epq_indices, pq_ptr, \
        epq_data, task_csf, task_scale, task_g, g_stride, ntasks, y, overflow_flag)

  if (epq_indptr_type == 4) {
    const int32_t* ip = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      LAUNCH_MIXED_SCATTER(int32_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_pq));
    } else if (epq_pq_type == 2) {
      LAUNCH_MIXED_SCATTER(int32_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_pq));
    } else {
      LAUNCH_MIXED_SCATTER(int32_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_pq));
    }
  } else {
    const int64_t* ip = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      LAUNCH_MIXED_SCATTER(int64_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_pq));
    } else if (epq_pq_type == 2) {
      LAUNCH_MIXED_SCATTER(int64_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_pq));
    } else {
      LAUNCH_MIXED_SCATTER(int64_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_pq));
    }
  }
  #undef LAUNCH_MIXED_SCATTER
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_tile_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* local_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const double* epq_data,
    const double* task_g,
    int64_t g_stride,
    const double* task_scale,
    int j_start,
    int j_count,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = j_count;
  if (epq_pq_type == 1) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<double, uint8_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint8_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else if (epq_pq_type == 2) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<double, uint16_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint16_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<double, int32_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const int32_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_tile_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* local_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const float* task_g,
    int64_t g_stride,
    const float* task_scale,
    int j_start,
    int j_count,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = j_count;
  if (epq_pq_type == 1) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, uint8_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint8_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else if (epq_pq_type == 2) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, uint16_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint16_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, int32_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const int32_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_tile_f32_kahan_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* local_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const float* task_g,
    int64_t g_stride,
    const float* task_scale,
    int j_start,
    int j_count,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = j_count;
  if (epq_pq_type == 1) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, uint8_t, true><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint8_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else if (epq_pq_type == 2) {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, uint16_t, true><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const uint16_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_epq_table_tile_kernel_t<float, int32_t, true><<<blocks, threads, 0, stream>>>(
        steps_table,
        ncsf,
        norb,
        local_indptr,
        epq_indices,
        reinterpret_cast<const int32_t*>(epq_pq),
        epq_data,
        task_g,
        g_stride,
        task_scale,
        j_start,
        j_count,
        y,
        overflow_flag);
  }
}

// Mixed-type tile launcher: OUT_T=double, COEF_T=float
extern "C" void guga_apply_g_flat_scatter_atomic_epq_table_tile_mixed_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* local_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const double* task_g,
    int64_t g_stride,
    const double* task_scale,
    int j_start,
    int j_count,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (j_count <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = j_count;

  #define LAUNCH_MIXED_TILE(PQ_T, pq_ptr) \
    guga_apply_g_flat_scatter_atomic_epq_table_tile_mixed_kernel_t<double, float, PQ_T> \
        <<<blocks, threads, 0, stream>>>( \
        steps_table, ncsf, norb, local_indptr, epq_indices, pq_ptr, \
        epq_data, task_g, g_stride, task_scale, j_start, j_count, y, overflow_flag)

  if (epq_pq_type == 1) {
    LAUNCH_MIXED_TILE(uint8_t, reinterpret_cast<const uint8_t*>(epq_pq));
  } else if (epq_pq_type == 2) {
    LAUNCH_MIXED_TILE(uint16_t, reinterpret_cast<const uint16_t*>(epq_pq));
  } else {
    LAUNCH_MIXED_TILE(int32_t, reinterpret_cast<const int32_t*>(epq_pq));
  }
  #undef LAUNCH_MIXED_TILE
}

extern "C" void guga_apply_g_flat_gather_epq_table_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const int32_t* task_row_by_csf,
    const float* task_scale_by_csf,
    const float* task_g,
    int64_t g_stride,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int32_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int32_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int32_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int64_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int64_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_table_f32_kernel_t<int64_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          task_row_by_csf,
          task_scale_by_csf,
          task_g,
          g_stride,
          y,
          overflow_flag,
          add);
    }
  }
}

extern "C" void guga_apply_g_flat_gather_epq_table_f32_kahan_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const int32_t* task_row_by_csf,
    const float* task_scale_by_csf,
    const float* task_g,
    int64_t g_stride,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int32_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i32, epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int32_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i32, epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    } else {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int32_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i32, epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int64_t, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i64, epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int64_t, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i64, epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    } else {
      guga_apply_g_flat_gather_epq_table_f32_kahan_kernel_t<int64_t, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table, ncsf, norb, epq_t_indptr_i64, epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq), epq_t_data,
          task_row_by_csf, task_scale_by_csf, task_g, g_stride,
          y, overflow_flag, add);
    }
  }
}

extern "C" void guga_apply_g_flat_gather_epq_transpose_range_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const double* epq_t_data,
    const double* g_block,
    int64_t g_stride,
    int k_start,
    int k_count,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (ncsf <= 0 || k_count <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  }
}

// Mixed-type launcher: OUT_T=double, COEF_T=float
extern "C" void guga_apply_g_flat_gather_epq_transpose_range_mixed_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const double* g_block,
    int64_t g_stride,
    int k_start,
    int k_count,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (ncsf <= 0 || k_count <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;

  #define LAUNCH_MIXED_TRANSPOSE(INDPTR_T, PQ_T, indptr_ptr, pq_ptr) \
    guga_apply_g_flat_gather_epq_transpose_range_kernel_t<INDPTR_T, double, float, PQ_T> \
        <<<blocks, threads, 0, stream>>>( \
            steps_table, ncsf, norb, indptr_ptr, epq_t_source, pq_ptr, epq_t_data, \
            g_block, g_stride, k_start, k_count, y, overflow_flag, add)

  if (epq_t_indptr_type == 4) {
    const int32_t* ip = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      LAUNCH_MIXED_TRANSPOSE(int32_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_t_pq));
    } else if (epq_t_pq_type == 2) {
      LAUNCH_MIXED_TRANSPOSE(int32_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_t_pq));
    } else {
      LAUNCH_MIXED_TRANSPOSE(int32_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_t_pq));
    }
  } else {
    const int64_t* ip = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      LAUNCH_MIXED_TRANSPOSE(int64_t, uint8_t, ip, reinterpret_cast<const uint8_t*>(epq_t_pq));
    } else if (epq_t_pq_type == 2) {
      LAUNCH_MIXED_TRANSPOSE(int64_t, uint16_t, ip, reinterpret_cast<const uint16_t*>(epq_t_pq));
    } else {
      LAUNCH_MIXED_TRANSPOSE(int64_t, int32_t, ip, reinterpret_cast<const int32_t*>(epq_t_pq));
    }
  }

  #undef LAUNCH_MIXED_TRANSPOSE
}

extern "C" void guga_apply_g_flat_gather_epq_transpose_range_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const float* g_block,
    int64_t g_stride,
    int k_start,
    int k_count,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (ncsf <= 0 || k_count <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, uint8_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, uint16_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, int32_t><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  }
}

extern "C" void guga_apply_g_flat_gather_epq_transpose_range_f32_kahan_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const float* g_block,
    int64_t g_stride,
    int k_start,
    int k_count,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int add) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (ncsf <= 0 || k_count <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, uint8_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, uint16_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int32_t, float, float, int32_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, uint8_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else if (epq_t_pq_type == 2) {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, uint16_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_kernel_t<int64_t, float, float, int32_t, true><<<blocks, threads, 0, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          g_block,
          g_stride,
          k_start,
          k_count,
          y,
          overflow_flag,
          add);
    }
  }
}

extern "C" void guga_apply_csr_eri_mat_fused_epq_table_range_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const double* epq_data,
    const int32_t* row_j,
    const int32_t* row_k,
    const int64_t* csr_indptr,
    const int32_t* csr_indices,
    const double* csr_data,
    int row_start,
    int nrows,
    const double* eri_mat_t,
    int nops,
    double half,
    const double* x,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (nrows <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  // Prefer the warp-per-row kernel when the block size is a multiple of 32. This reduces
  // block-level synchronization and amortizes launch/scheduling overhead by processing multiple
  // CSR rows per thread block.
  if (threads >= 32 && (threads % 32) == 0) {
    int warps = threads / 32;
    int blocks = (nrows + warps - 1) / warps;
    size_t smem_bytes = (size_t)warps * (size_t)nops * sizeof(double);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, double, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, double, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, double, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    }
  } else {
    int blocks = nrows;
    size_t smem_bytes = (size_t)nops * sizeof(double);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, double, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, double, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, double, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    }
  }
}

// Mixed-type launcher: OUT_T=double, COEF_T=float
extern "C" void guga_apply_csr_eri_mat_fused_epq_table_range_f64_out_f32_coeff_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const int32_t* row_j,
    const int32_t* row_k,
    const int64_t* csr_indptr,
    const int32_t* csr_indices,
    const double* csr_data,
    int row_start,
    int nrows,
    const double* eri_mat_t,
    int nops,
    double half,
    const double* x,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (nrows <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  if (threads >= 32 && (threads % 32) == 0) {
    int warps = threads / 32;
    int blocks = (nrows + warps - 1) / warps;
    size_t smem_bytes = (size_t)warps * (size_t)nops * sizeof(double);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<double, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    }
  } else {
    int blocks = nrows;
    size_t smem_bytes = (size_t)nops * sizeof(double);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<double, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    }
  }
}

extern "C" void guga_apply_csr_eri_mat_fused_epq_table_range_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const int32_t* row_j,
    const int32_t* row_k,
    const int64_t* csr_indptr,
    const int32_t* csr_indices,
    const float* csr_data,
    int row_start,
    int nrows,
    const float* eri_mat_t,
    int nops,
    float half,
    const float* x,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (nrows <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  if (threads >= 32 && (threads % 32) == 0) {
    int warps = threads / 32;
    int blocks = (nrows + warps - 1) / warps;
    size_t smem_bytes = (size_t)warps * (size_t)nops * sizeof(float);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    }
  } else {
    int blocks = nrows;
    size_t smem_bytes = (size_t)nops * sizeof(float);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          steps_table,
          ncsf,
          norb,
          epq_indptr,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          row_j,
          row_k,
          csr_indptr,
          csr_indices,
          csr_data,
          row_start,
          nrows,
          eri_mat_t,
          nops,
          half,
          x,
          y,
          overflow_flag);
    }
  }
}

extern "C" void guga_apply_csr_eri_mat_fused_epq_table_range_f32_kahan_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const int32_t* row_j,
    const int32_t* row_k,
    const int64_t* csr_indptr,
    const int32_t* csr_indices,
    const float* csr_data,
    int row_start,
    int nrows,
    const float* eri_mat_t,
    int nops,
    float half,
    const float* x,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (nrows <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  if (threads >= 32 && (threads % 32) == 0) {
    int warps = threads / 32;
    int blocks = (nrows + warps - 1) / warps;
    size_t smem_bytes = (size_t)warps * (size_t)nops * sizeof(float);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, uint8_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, uint16_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_warp_kernel_t<float, float, int32_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    }
  } else {
    int blocks = nrows;
    size_t smem_bytes = (size_t)nops * sizeof(float);
    if (epq_pq_type == 1) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, uint8_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else if (epq_pq_type == 2) {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, uint16_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    } else {
      guga_apply_csr_eri_mat_fused_epq_table_range_kernel_t<float, float, int32_t, true><<<blocks, threads, smem_bytes, stream>>>(
          steps_table, ncsf, norb, epq_indptr, epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq), epq_data,
          row_j, row_k, csr_indptr, csr_indices, csr_data,
          row_start, nrows, eri_mat_t, nops, half, x, y, overflow_flag);
    }
  }
}

extern "C" void guga_build_w_from_csr_unitnnz_launch_stream(
    const int32_t* row_j,
    const int32_t* row_k,
    const int32_t* csr_rs,
    const double* csr_c,
    int nrows,
    const double* x,
    int ncsf,
    int nops,
    double* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (nrows <= 0) return;
  if (ncsf <= 0) return;
  if (nops <= 0) return;
  int blocks = (nrows + threads - 1) / threads;
  guga_build_w_from_csr_unitnnz_kernel<<<blocks, threads, 0, stream>>>(
      row_j, row_k, csr_rs, csr_c, nrows, x, ncsf, nops, w_out, w_stride, overflow_flag);
}

extern "C" void guga_build_w_from_epq_transpose_range_launch_stream(
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const double* epq_t_data,
    const double* x,
    int ncsf,
    int nops,
    double* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (k_count <= 0 || ncsf <= 0 || nops <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = k_count;
  size_t smem_bytes = (size_t)nops * sizeof(double);
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, double, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, double, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, double, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, double, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, double, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, double, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_w_from_epq_transpose_range_f32_launch_stream(
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const float* x,
    int ncsf,
    int nops,
    float* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (k_count <= 0 || ncsf <= 0 || nops <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = k_count;
  size_t smem_bytes = (size_t)nops * sizeof(float);
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, float, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, float, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, float, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, float, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, float, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, float, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_w_from_epq_transpose_range_f64_out_f32_coeff_launch_stream(
    const void* epq_t_indptr,
    int epq_t_indptr_type,
    const int32_t* epq_t_source,
    const void* epq_t_pq,
    int epq_t_pq_type,
    const float* epq_t_data,
    const double* x,
    int ncsf,
    int nops,
    double* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_t_indptr_type != 4 && epq_t_indptr_type != 8) return;
  if (k_count <= 0 || ncsf <= 0 || nops <= 0) return;
  if (epq_t_pq_type != 1 && epq_t_pq_type != 2 && epq_t_pq_type != 4) return;
  int blocks = k_count;
  size_t smem_bytes = (size_t)nops * sizeof(double);
  if (epq_t_indptr_type == 4) {
    const int32_t* epq_t_indptr_i32 = reinterpret_cast<const int32_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int32_t, double, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i32,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_t_indptr_i64 = reinterpret_cast<const int64_t*>(epq_t_indptr);
    if (epq_t_pq_type == 1) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, float, uint8_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint8_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_t_pq_type == 2) {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, float, uint16_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const uint16_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_transpose_range_kernel_t<int64_t, double, float, int32_t><<<blocks, threads, smem_bytes, stream>>>(
          epq_t_indptr_i64,
          epq_t_source,
          reinterpret_cast<const int32_t*>(epq_t_pq),
          epq_t_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_w_from_epq_table_launch_stream(
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const double* epq_data,
    const double* x,
    int ncsf,
    int nops,
    double* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (ncsf <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_indptr_type == 4) {
    const int32_t* epq_indptr_i32 = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_indptr_i64 = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, double, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, double, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, double, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_w_from_epq_table_f64_out_f32_coeff_launch_stream(
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const double* x,
    int ncsf,
    int nops,
    double* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (ncsf <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_indptr_type == 4) {
    const int32_t* epq_indptr_i32 = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, float, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, float, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int32_t, double, float, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_indptr_i64 = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, float, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, float, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int64_t, double, float, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_t_from_epq_table_launch_stream(
    const int8_t* steps_table,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const double* epq_data,
    const double* c_vec,
    int ncsf,
    int norb,
    int nops,
    double* t_out,
    int64_t t_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (ncsf <= 0) return;
  if (norb <= 0) return;
  if (nops <= 0) return;
  if (!steps_table) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_pq_type == 1) {
    guga_build_t_from_epq_table_kernel_t<double, uint8_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const uint8_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  } else if (epq_pq_type == 2) {
    guga_build_t_from_epq_table_kernel_t<double, uint16_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const uint16_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  } else {
    guga_build_t_from_epq_table_kernel_t<double, int32_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const int32_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  }
}

extern "C" void guga_build_w_from_epq_table_f32_launch_stream(
    const void* epq_indptr,
    int epq_indptr_type,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const float* x,
    int ncsf,
    int nops,
    float* w_out,
    int64_t w_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int k_start,
    int k_count) {
  if (epq_indptr_type != 4 && epq_indptr_type != 8) return;
  if (ncsf <= 0) return;
  if (nops <= 0) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_indptr_type == 4) {
    const int32_t* epq_indptr_i32 = reinterpret_cast<const int32_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int32_t, float, float, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int32_t, float, float, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int32_t, float, float, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i32,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  } else {
    const int64_t* epq_indptr_i64 = reinterpret_cast<const int64_t*>(epq_indptr);
    if (epq_pq_type == 1) {
      guga_build_w_from_epq_table_kernel_t<int64_t, float, float, uint8_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint8_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else if (epq_pq_type == 2) {
      guga_build_w_from_epq_table_kernel_t<int64_t, float, float, uint16_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const uint16_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    } else {
      guga_build_w_from_epq_table_kernel_t<int64_t, float, float, int32_t><<<blocks, threads, 0, stream>>>(
          epq_indptr_i64,
          epq_indices,
          reinterpret_cast<const int32_t*>(epq_pq),
          epq_data,
          x,
          ncsf,
          nops,
          w_out,
          w_stride,
          overflow_flag,
          k_start,
          k_count);
    }
  }
}

extern "C" void guga_build_t_from_epq_table_f32_launch_stream(
    const int8_t* steps_table,
    const int64_t* epq_indptr,
    const int32_t* epq_indices,
    const void* epq_pq,
    int epq_pq_type,
    const float* epq_data,
    const float* c_vec,
    int ncsf,
    int norb,
    int nops,
    float* t_out,
    int64_t t_stride,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  if (ncsf <= 0) return;
  if (norb <= 0) return;
  if (nops <= 0) return;
  if (!steps_table) return;
  if (epq_pq_type != 1 && epq_pq_type != 2 && epq_pq_type != 4) return;
  int blocks = ncsf;
  if (epq_pq_type == 1) {
    guga_build_t_from_epq_table_kernel_t<float, uint8_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const uint8_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  } else if (epq_pq_type == 2) {
    guga_build_t_from_epq_table_kernel_t<float, uint16_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const uint16_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  } else {
    guga_build_t_from_epq_table_kernel_t<float, int32_t><<<blocks, threads, 0, stream>>>(
        steps_table,
        epq_indptr,
        epq_indices,
        reinterpret_cast<const int32_t*>(epq_pq),
        epq_data,
        c_vec,
        ncsf,
        norb,
        nops,
        t_out,
        t_stride,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_task_sums_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* out_sum,
    int* overflow_flag,
    int threads) {
  int blocks = ntasks;
  if (norb <= 16) {
    guga_apply_g_flat_task_sums_kernel_t<16><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_task_sums_kernel_t<24><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_task_sums_kernel_t<32><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_task_sums_kernel_t<48><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else {
    guga_apply_g_flat_task_sums_kernel_t<MAX_NORB><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_task_sums_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* out_sum,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  int blocks = ntasks;
  if (norb <= 16) {
    guga_apply_g_flat_task_sums_kernel_t<16><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_task_sums_kernel_t<24><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_task_sums_kernel_t<32><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_task_sums_kernel_t<48><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  } else {
    guga_apply_g_flat_task_sums_kernel_t<MAX_NORB><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, out_sum,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
    int* overflow_flag,
  int threads) {
  int blocks = ntasks;
  if (threads == 32) {
    if (norb <= 16) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<16, double><<<blocks, 32>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 24) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<24, double><<<blocks, 32>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 32) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<32, double><<<blocks, 32>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 48) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<48, double><<<blocks, 32>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<MAX_NORB, double><<<blocks, 32>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    }
  }
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_kernel_t<16, double><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_kernel_t<24, double><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_kernel_t<32, double><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_kernel_t<48, double><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_kernel_t<MAX_NORB, double><<<blocks, threads>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  int blocks = ntasks;
  if (threads == 32) {
    if (norb <= 16) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<16, double><<<blocks, 32, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 24) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<24, double><<<blocks, 32, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 32) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<32, double><<<blocks, 32, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else if (norb <= 48) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<48, double><<<blocks, 32, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    } else {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<MAX_NORB, double><<<blocks, 32, 0, stream>>>(
          child,
          node_twos,
          child_prefix,
          steps_table,
          nodes_table,
          ncsf,
          norb,
          task_csf,
          task_scale,
          task_g,
          g_stride,
          ntasks,
          y,
          overflow_flag);
      return;
    }
  }
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_kernel_t<16, double><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_kernel_t<24, double><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_kernel_t<32, double><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_kernel_t<48, double><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_kernel_t<MAX_NORB, double><<<blocks, threads, 0, stream>>>(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        task_csf,
        task_scale,
        task_g,
        g_stride,
        ntasks,
        y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_f32_launch(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const float* task_scale,
    const float* task_g,
    int64_t g_stride,
    int ntasks,
    float* y,
    int* overflow_flag,
    int threads) {
  int blocks = ntasks;
  if (threads == 32) {
    if (norb <= 16) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<16, float><<<blocks, 32>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 24) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<24, float><<<blocks, 32>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 32) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<32, float><<<blocks, 32>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 48) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<48, float><<<blocks, 32>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<MAX_NORB, float><<<blocks, 32>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    }
  }
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_kernel_t<16, float><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_kernel_t<24, float><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_kernel_t<32, float><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_kernel_t<48, float><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_kernel_t<MAX_NORB, float><<<blocks, threads>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_f32_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const float* task_scale,
    const float* task_g,
    int64_t g_stride,
    int ntasks,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  int blocks = ntasks;
  if (threads == 32) {
    if (norb <= 16) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<16, float><<<blocks, 32, 0, stream>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 24) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<24, float><<<blocks, 32, 0, stream>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 32) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<32, float><<<blocks, 32, 0, stream>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else if (norb <= 48) {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<48, float><<<blocks, 32, 0, stream>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    } else {
      guga_apply_g_flat_scatter_atomic_warp32_shstack_kernel_t<MAX_NORB, float><<<blocks, 32, 0, stream>>>(
          child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
          overflow_flag);
      return;
    }
  }
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_kernel_t<16, float><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_kernel_t<24, float><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_kernel_t<32, float><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_kernel_t<48, float><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_kernel_t<MAX_NORB, float><<<blocks, threads, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  }
}

// =============================================================================
// 10.16.3 / 10.18: Warp-Cooperative Kernel Launch Functions
// =============================================================================

extern "C" void guga_apply_g_flat_scatter_atomic_warp_coop_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const double* task_scale,
    const double* task_g,
    int64_t g_stride,
    int ntasks,
    double* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  // Warp-cooperative: each warp processes one task, 8 warps per block
  constexpr int kWarpsPerBlock = 8;
  int threads_per_block = threads > 0 ? threads : (kWarpsPerBlock * 32);
  int warps_per_block = threads_per_block >> 5;
  if (warps_per_block <= 0) warps_per_block = 1;
  int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<16, double><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<24, double><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<32, double><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<48, double><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<MAX_NORB, double><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  }
}

extern "C" void guga_apply_g_flat_scatter_atomic_warp_coop_f32_launch_stream(
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const float* task_scale,
    const float* task_g,
    int64_t g_stride,
    int ntasks,
    float* y,
    int* overflow_flag,
    cudaStream_t stream,
    int threads) {
  // Warp-cooperative: each warp processes one task, 8 warps per block
  constexpr int kWarpsPerBlock = 8;
  int threads_per_block = threads > 0 ? threads : (kWarpsPerBlock * 32);
  int warps_per_block = threads_per_block >> 5;
  if (warps_per_block <= 0) warps_per_block = 1;
  int blocks = (ntasks + warps_per_block - 1) / warps_per_block;
  
  if (norb <= 16) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<16, float><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 24) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<24, float><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 32) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<32, float><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else if (norb <= 48) {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<48, float><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  } else {
    guga_apply_g_flat_scatter_atomic_warp_coop_kernel_t<MAX_NORB, float><<<blocks, threads_per_block, 0, stream>>>(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_scale, task_g, g_stride, ntasks, y,
        overflow_flag);
  }
}
