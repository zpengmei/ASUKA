// guga_cuda_kernels_mrci_fused.cuh — Fused MRCI kernels
// P1C: sym-pair pack/unpack
// P1A: batched W-build + diagonal
// P1B: batched G apply scatter
// P2:  RDM12 reorder + delta
// P4:  batched scatter/gather embed/project

#pragma once

#include <cstdint>

namespace {

// ============================================================================
// P1C: Symmetric-pair pack/unpack kernels
// ============================================================================

// Pack: W_pair[k, u] = W[k, pair_pq[u]] + W[k, pair_qp[u]]
//       halved if pair_pq[u] == pair_qp[u]  (diagonal element)
__global__ void sym_pair_pack_kernel(
    const double* __restrict__ W,
    double* __restrict__ W_pair,
    const int32_t* __restrict__ pair_pq,
    const int32_t* __restrict__ pair_qp,
    int nrows, int nops, int npair, int64_t w_stride, int64_t wp_stride) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const double* w_row = W + (int64_t)row * w_stride;
  double* wp_row = W_pair + (int64_t)row * wp_stride;
  for (int u = threadIdx.x; u < npair; u += blockDim.x) {
    int pq = pair_pq[u];
    int qp = pair_qp[u];
    double val = w_row[pq] + w_row[qp];
    if (pq == qp) val *= 0.5;  // diagonal: halve
    wp_row[u] = val;
  }
}

// Unpack: G[k, pq] = G_pair[k, full_to_pair[pq]]
__global__ void sym_pair_unpack_kernel(
    const double* __restrict__ G_pair,
    double* __restrict__ G,
    const int32_t* __restrict__ full_to_pair,
    int nrows, int nops, int npair, int64_t gp_stride, int64_t g_stride) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const double* gp_row = G_pair + (int64_t)row * gp_stride;
  double* g_row = G + (int64_t)row * g_stride;
  for (int u = threadIdx.x; u < nops; u += blockDim.x) {
    g_row[u] = gp_row[full_to_pair[u]];
  }
}

// Float32 variants
__global__ void sym_pair_pack_kernel_f32(
    const float* __restrict__ W,
    float* __restrict__ W_pair,
    const int32_t* __restrict__ pair_pq,
    const int32_t* __restrict__ pair_qp,
    int nrows, int nops, int npair, int64_t w_stride, int64_t wp_stride) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const float* w_row = W + (int64_t)row * w_stride;
  float* wp_row = W_pair + (int64_t)row * wp_stride;
  for (int u = threadIdx.x; u < npair; u += blockDim.x) {
    int pq = pair_pq[u];
    int qp = pair_qp[u];
    float val = w_row[pq] + w_row[qp];
    if (pq == qp) val *= 0.5f;
    wp_row[u] = val;
  }
}

__global__ void sym_pair_unpack_kernel_f32(
    const float* __restrict__ G_pair,
    float* __restrict__ G,
    const int32_t* __restrict__ full_to_pair,
    int nrows, int nops, int npair, int64_t gp_stride, int64_t g_stride) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const float* gp_row = G_pair + (int64_t)row * gp_stride;
  float* g_row = G + (int64_t)row * g_stride;
  for (int u = threadIdx.x; u < nops; u += blockDim.x) {
    g_row[u] = gp_row[full_to_pair[u]];
  }
}

// ============================================================================
// P2: RDM12 reorder + delta kernel
// ============================================================================

// dm1_out[q, p] = dm1_pq[p * norb + q]   (transpose)
// dm2_out[p, q, r, s] = gram0[r * norb + s, p * norb + q]   (row-swap + transpose)
//     if q == r: dm2_out[p, q, r, s] -= dm1_pq[s * norb + p]
__global__ void rdm12_reorder_delta_kernel(
    const double* __restrict__ dm1_pq,
    const double* __restrict__ gram0,
    double* __restrict__ dm1_out,
    double* __restrict__ dm2_out,
    int norb) {
  int nops = norb * norb;
  int n4 = nops * nops;  // norb^4

  // Phase 1: dm1 transpose (first norb^2 threads do this)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nops;
       idx += gridDim.x * blockDim.x) {
    int p = idx / norb;
    int q = idx % norb;
    dm1_out[q * norb + p] = dm1_pq[p * norb + q];
  }

  // Phase 2: dm2 reorder + delta (all threads)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n4;
       idx += gridDim.x * blockDim.x) {
    int s = idx % norb;
    int r = (idx / norb) % norb;
    int q = (idx / (norb * norb)) % norb;
    int p = idx / (norb * norb * norb);

    // gram0 is [nops, nops] row-major
    // Reference: swap = arange(nops).reshape(norb,norb).T.ravel()
    //   gram = gram0[swap]  →  gram[pq,:] = gram0[qp,:]
    //   dm2[p,q,r,s] = gram[p*norb+q, r*norb+s] = gram0[q*norb+p, r*norb+s]
    double val = gram0[(int64_t)(q * norb + p) * nops + (r * norb + s)];

    // delta correction: dm2[p,q,q,s] -= dm1[s,p] = dm1_pq[p*norb+s]
    if (q == r) {
      val -= dm1_pq[p * norb + s];
    }

    dm2_out[idx] = val;
  }
}

// ============================================================================
// P4: Batched scatter embed / gather project
// ============================================================================

// Scatter: x_full[sub_to_full[i], v] = x_sub[i, v]  for all v in [0, nvec)
__global__ void scatter_embed_batched_kernel(
    const double* __restrict__ x_sub,
    const int64_t* __restrict__ sub_to_full,
    double* __restrict__ x_full,
    int nsub, int nvec, int64_t sub_stride, int64_t full_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nsub) return;
  int64_t dst = sub_to_full[i];
  const double* src_row = x_sub + (int64_t)i * sub_stride;
  double* dst_row = x_full + dst * full_stride;
  for (int v = 0; v < nvec; v++) {
    dst_row[v] = src_row[v];
  }
}

// Gather: y_sub[i, v] = y_full[sub_to_full[i], v]  for all v in [0, nvec)
__global__ void gather_project_batched_kernel(
    const double* __restrict__ y_full,
    const int64_t* __restrict__ sub_to_full,
    double* __restrict__ y_sub,
    int nsub, int nvec, int64_t full_stride, int64_t sub_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nsub) return;
  int64_t src = sub_to_full[i];
  const double* src_row = y_full + src * full_stride;
  double* dst_row = y_sub + (int64_t)i * sub_stride;
  for (int v = 0; v < nvec; v++) {
    dst_row[v] = src_row[v];
  }
}

// ============================================================================
// P1A: Batched source-major W-build + diagonal kernels
// ============================================================================

// Batched W-build: W_out[v*k_count + k_local, pq] += coef * x[j, v_start+v]
// One block per source CSF j.  Threads stride over EPQ entries.
// Inner loop accumulates for all nvec vectors.
// x values for source CSF j are cached in shared memory (one read per block).
__global__ void build_w_from_epq_table_batched_kernel(
    const int64_t* __restrict__ epq_indptr,   // [ncsf+1]
    const int32_t* __restrict__ epq_indices,  // [nnz] destination CSF
    const int32_t* __restrict__ epq_pq,       // [nnz] operator index
    const double* __restrict__ epq_data,      // [nnz] coefficient
    const double* __restrict__ x,             // [ncsf, x_stride]
    int64_t x_stride,
    double* __restrict__ w_out,               // [nvec * k_count, w_stride]
    int64_t w_stride,
    int ncsf, int nops,
    int nvec, int v_start,
    int k_start, int k_count) {
  int j = blockIdx.x;
  if (j >= ncsf) return;

  // Cache x[j, v_start:v_start+nvec] in shared memory (max 32 vectors)
  extern __shared__ double x_s[];
  if (threadIdx.x < nvec && threadIdx.x < 32) {
    x_s[threadIdx.x] = x[(int64_t)j * x_stride + (v_start + threadIdx.x)];
  }
  __syncthreads();

  int64_t start = epq_indptr[j];
  int64_t end = epq_indptr[j + 1];

  for (int64_t e = start + threadIdx.x; e < end; e += blockDim.x) {
    int k = epq_indices[e];
    if (k < k_start || k >= k_start + k_count) continue;
    int pq = epq_pq[e];
    double coef = epq_data[e];
    int k_local = k - k_start;

    for (int v = 0; v < nvec; v++) {
      atomicAdd(&w_out[(int64_t)(v * k_count + k_local) * w_stride + pq],
                coef * x_s[v]);
    }
  }
}

// Batched diagonal: W_out[v*j_count + j_local, r*norb+r] = occ(j,r) * x[j, v_start+v]
// Grid-stride over (j_local * norb + r).  Inner loop over nvec.
__global__ void build_w_diag_batched_kernel(
    const int8_t* __restrict__ steps_table,  // [ncsf, norb]
    const double* __restrict__ x,            // [ncsf, x_stride]
    int64_t x_stride,
    double* __restrict__ w_out,              // [nvec * j_count, w_stride]
    int64_t w_stride,
    int ncsf, int norb,
    int j_start, int j_count,
    int nvec, int v_start) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int j_local = t / norb;
  int r = t % norb;
  if (j_local >= j_count) return;

  int j = j_start + j_local;
  if (j >= ncsf) return;

  int8_t step = steps_table[(int64_t)j * norb + r];
  // step_to_occ: 0->0, 1->1, 2->1, 3->2
  int occ_r = (step == 0) ? 0 : ((step == 3) ? 2 : 1);
  if (occ_r == 0) return;

  int rr = r * norb + r;
  double occ_d = (double)occ_r;
  for (int v = 0; v < nvec; v++) {
    double x_val = x[(int64_t)j * x_stride + (v_start + v)];
    w_out[(int64_t)(v * j_count + j_local) * w_stride + rr] = occ_d * x_val;
  }
}

// ============================================================================
// P5: RDM123 delta correction + symmetry kernels
// ============================================================================

// P5a: dm2 delta correction — dm2[p, k, k, s] -= dm1[p, s] for all k
__global__ void dm2_delta_correction_kernel(
    double* __restrict__ dm2,          // [n, n, n, n] in-place
    const double* __restrict__ dm1,    // [n, n]
    int n) {
  int n2 = n * n;
  int n3 = n2 * n;
  // one thread per (p, k, s)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n3) return;
  int s = idx % n;
  int k = (idx / n) % n;
  int p = idx / n2;
  // dm2[p, k, k, s] at flat index p*n^3 + k*n^2 + k*n + s
  dm2[(int64_t)p * n3 + (int64_t)k * n2 + (int64_t)k * n + s] -= dm1[p * n + s];
}

// P5b: dm2 4-way symmetry — dm2_out = 0.25*(dm2 + dm2.T(2,3,0,1) + dm2.T(3,2,1,0) + dm2.T(1,0,3,2))
__global__ void dm2_4way_symmetry_kernel(
    const double* __restrict__ dm2_in,   // [n^4] read-only
    double* __restrict__ dm2_out,         // [n^4] output
    int n) {
  int n2 = n * n;
  int n3 = n2 * n;
  int n4 = n3 * n;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n4) return;
  int s = idx % n;
  int r = (idx / n) % n;
  int q = (idx / n2) % n;
  int p = idx / n3;
  double v0 = dm2_in[(int64_t)p * n3 + q * n2 + r * n + s];
  double v1 = dm2_in[(int64_t)r * n3 + s * n2 + p * n + q];  // T(2,3,0,1)
  double v2 = dm2_in[(int64_t)s * n3 + r * n2 + q * n + p];  // T(3,2,1,0)
  double v3 = dm2_in[(int64_t)q * n3 + p * n2 + s * n + r];  // T(1,0,3,2)
  dm2_out[idx] = 0.25 * (v0 + v1 + v2 + v3);
}

// P5c: dm3 delta correction — all 4 corrections in single pass
// Uses post-symmetry dm2 (already delta-corrected + 4-way averaged).
//   if b==c: dm3[a,b,c,d,e,f] -= dm2[a,d,e,f]
//   if d==e: dm3[a,b,c,d,e,f] -= dm2[a,b,c,f]
//   if b==e: dm3[a,b,c,d,e,f] -= dm2[c,d,a,f]
//   if b==c && d==e: dm3[a,b,c,d,e,f] -= dm1[a,f]
__global__ void dm3_delta_correction_kernel(
    double* __restrict__ dm3,          // [n^6] in-place
    const double* __restrict__ dm2,    // [n^4] post-symmetry
    const double* __restrict__ dm1,    // [n^2]
    int n) {
  int n2 = n * n;
  int n3 = n2 * n;
  int64_t n4 = (int64_t)n3 * n;
  int64_t n5 = n4 * n;
  int64_t n6 = n5 * n;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n6) return;

  int f = (int)(idx % n);
  int e = (int)((idx / n) % n);
  int d = (int)((idx / n2) % n);
  int c = (int)((idx / n3) % n);
  int b = (int)((idx / n4) % n);
  int a = (int)(idx / n5);

  double delta = 0.0;
  if (b == c) delta -= dm2[(int64_t)a * n3 + d * n2 + e * n + f];
  if (d == e) delta -= dm2[(int64_t)a * n3 + b * n2 + c * n + f];
  if (b == e) delta -= dm2[(int64_t)c * n3 + d * n2 + a * n + f];
  if (b == c && d == e) delta -= dm1[a * n + f];

  dm3[idx] += delta;
}

// P5d: dm3 6-way symmetry
// dm3_out = (1/6)*(dm3 + dm3.T(2,3,0,1,4,5) + dm3.T(4,5,2,3,0,1)
//           + dm3.T(0,1,4,5,2,3) + dm3.T(2,3,4,5,0,1) + dm3.T(4,5,0,1,2,3))
__global__ void dm3_6way_symmetry_kernel(
    const double* __restrict__ dm3_in,   // [n^6] read-only
    double* __restrict__ dm3_out,         // [n^6] output
    int n) {
  int n2 = n * n;
  int n3 = n2 * n;
  int64_t n4 = (int64_t)n3 * n;
  int64_t n5 = n4 * n;
  int64_t n6 = n5 * n;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n6) return;

  int f = (int)(idx % n);
  int e = (int)((idx / n) % n);
  int d = (int)((idx / n2) % n);
  int c = (int)((idx / n3) % n);
  int b = (int)((idx / n4) % n);
  int a = (int)(idx / n5);

  // Helper: flat index for dm3[i0,i1,i2,i3,i4,i5]
  #define DM3_IDX(i0,i1,i2,i3,i4,i5) \
    ((int64_t)(i0)*n5 + (int64_t)(i1)*n4 + (int64_t)(i2)*n3 + (int64_t)(i3)*n2 + (int64_t)(i4)*n + (i5))

  double v0 = dm3_in[DM3_IDX(a,b,c,d,e,f)];
  double v1 = dm3_in[DM3_IDX(c,d,a,b,e,f)];  // T(2,3,0,1,4,5)
  double v2 = dm3_in[DM3_IDX(e,f,c,d,a,b)];  // T(4,5,2,3,0,1)
  double v3 = dm3_in[DM3_IDX(a,b,e,f,c,d)];  // T(0,1,4,5,2,3)
  double v4 = dm3_in[DM3_IDX(c,d,e,f,a,b)];  // T(2,3,4,5,0,1)
  double v5 = dm3_in[DM3_IDX(e,f,a,b,c,d)];  // T(4,5,0,1,2,3)

  dm3_out[idx] = (v0 + v1 + v2 + v3 + v4 + v5) * (1.0 / 6.0);

  #undef DM3_IDX
}

}  // anonymous namespace

// ============================================================================
// Launch wrappers (extern "C")
// ============================================================================

extern "C" void sym_pair_pack_launch_stream(
    const double* W, double* W_pair,
    const int32_t* pair_pq, const int32_t* pair_qp,
    int nrows, int nops, int npair,
    int64_t w_stride, int64_t wp_stride,
    cudaStream_t stream, int threads) {
  if (nrows <= 0 || npair <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  sym_pair_pack_kernel<<<nrows, threads, 0, stream>>>(
      W, W_pair, pair_pq, pair_qp, nrows, nops, npair, w_stride, wp_stride);
}

extern "C" void sym_pair_unpack_launch_stream(
    const double* G_pair, double* G,
    const int32_t* full_to_pair,
    int nrows, int nops, int npair,
    int64_t gp_stride, int64_t g_stride,
    cudaStream_t stream, int threads) {
  if (nrows <= 0 || nops <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  sym_pair_unpack_kernel<<<nrows, threads, 0, stream>>>(
      G_pair, G, full_to_pair, nrows, nops, npair, gp_stride, g_stride);
}

extern "C" void sym_pair_pack_f32_launch_stream(
    const float* W, float* W_pair,
    const int32_t* pair_pq, const int32_t* pair_qp,
    int nrows, int nops, int npair,
    int64_t w_stride, int64_t wp_stride,
    cudaStream_t stream, int threads) {
  if (nrows <= 0 || npair <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  sym_pair_pack_kernel_f32<<<nrows, threads, 0, stream>>>(
      W, W_pair, pair_pq, pair_qp, nrows, nops, npair, w_stride, wp_stride);
}

extern "C" void sym_pair_unpack_f32_launch_stream(
    const float* G_pair, float* G,
    const int32_t* full_to_pair,
    int nrows, int nops, int npair,
    int64_t gp_stride, int64_t g_stride,
    cudaStream_t stream, int threads) {
  if (nrows <= 0 || nops <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  sym_pair_unpack_kernel_f32<<<nrows, threads, 0, stream>>>(
      G_pair, G, full_to_pair, nrows, nops, npair, gp_stride, g_stride);
}

extern "C" void rdm12_reorder_delta_launch_stream(
    const double* dm1_pq, const double* gram0,
    double* dm1_out, double* dm2_out,
    int norb,
    cudaStream_t stream, int threads) {
  if (norb <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int n4 = norb * norb * norb * norb;
  int blocks = (n4 + threads - 1) / threads;
  rdm12_reorder_delta_kernel<<<blocks, threads, 0, stream>>>(
      dm1_pq, gram0, dm1_out, dm2_out, norb);
}

extern "C" void scatter_embed_batched_launch_stream(
    const double* x_sub, const int64_t* sub_to_full, double* x_full,
    int nsub, int nvec, int64_t sub_stride, int64_t full_stride,
    cudaStream_t stream, int threads) {
  if (nsub <= 0 || nvec <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 128;
  int blocks = (nsub + threads - 1) / threads;
  scatter_embed_batched_kernel<<<blocks, threads, 0, stream>>>(
      x_sub, sub_to_full, x_full, nsub, nvec, sub_stride, full_stride);
}

extern "C" void gather_project_batched_launch_stream(
    const double* y_full, const int64_t* sub_to_full, double* y_sub,
    int nsub, int nvec, int64_t full_stride, int64_t sub_stride,
    cudaStream_t stream, int threads) {
  if (nsub <= 0 || nvec <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 128;
  int blocks = (nsub + threads - 1) / threads;
  gather_project_batched_kernel<<<blocks, threads, 0, stream>>>(
      y_full, sub_to_full, y_sub, nsub, nvec, full_stride, sub_stride);
}

// P1A launch wrappers

extern "C" void build_w_from_epq_table_batched_launch_stream(
    const int64_t* epq_indptr, const int32_t* epq_indices,
    const int32_t* epq_pq, const double* epq_data,
    const double* x, int64_t x_stride,
    double* w_out, int64_t w_stride,
    int ncsf, int nops,
    int nvec, int v_start,
    int k_start, int k_count,
    cudaStream_t stream, int threads) {
  if (ncsf <= 0 || nvec <= 0 || k_count <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int smem = (nvec < 32 ? nvec : 32) * (int)sizeof(double);
  build_w_from_epq_table_batched_kernel<<<ncsf, threads, smem, stream>>>(
      epq_indptr, epq_indices, epq_pq, epq_data,
      x, x_stride, w_out, w_stride,
      ncsf, nops, nvec, v_start, k_start, k_count);
}

extern "C" void build_w_diag_batched_launch_stream(
    const int8_t* steps_table, const double* x, int64_t x_stride,
    double* w_out, int64_t w_stride,
    int ncsf, int norb,
    int j_start, int j_count,
    int nvec, int v_start,
    cudaStream_t stream, int threads) {
  if (j_count <= 0 || nvec <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int total = j_count * norb;
  int blocks = (total + threads - 1) / threads;
  build_w_diag_batched_kernel<<<blocks, threads, 0, stream>>>(
      steps_table, x, x_stride, w_out, w_stride,
      ncsf, norb, j_start, j_count, nvec, v_start);
}

// P5 launch wrappers

extern "C" void dm2_delta_correction_launch_stream(
    double* dm2, const double* dm1, int n,
    cudaStream_t stream, int threads) {
  if (n <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int total = n * n * n;
  int blocks = (total + threads - 1) / threads;
  dm2_delta_correction_kernel<<<blocks, threads, 0, stream>>>(dm2, dm1, n);
}

extern "C" void dm2_4way_symmetry_launch_stream(
    const double* dm2_in, double* dm2_out, int n,
    cudaStream_t stream, int threads) {
  if (n <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int n4 = n * n * n * n;
  int blocks = (n4 + threads - 1) / threads;
  dm2_4way_symmetry_kernel<<<blocks, threads, 0, stream>>>(dm2_in, dm2_out, n);
}

extern "C" void dm3_delta_correction_launch_stream(
    double* dm3, const double* dm2, const double* dm1, int n,
    cudaStream_t stream, int threads) {
  if (n <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int64_t n6 = (int64_t)n * n * n * n * n * n;
  int64_t blocks = (n6 + threads - 1) / threads;
  if (blocks > (int64_t)INT32_MAX) blocks = INT32_MAX;
  dm3_delta_correction_kernel<<<(int)blocks, threads, 0, stream>>>(
      dm3, dm2, dm1, n);
}

extern "C" void dm3_6way_symmetry_launch_stream(
    const double* dm3_in, double* dm3_out, int n,
    cudaStream_t stream, int threads) {
  if (n <= 0) return;
  if (threads <= 0 || threads > 1024) threads = 256;
  int64_t n6 = (int64_t)n * n * n * n * n * n;
  int64_t blocks = (n6 + threads - 1) / threads;
  if (blocks > (int64_t)INT32_MAX) blocks = INT32_MAX;
  dm3_6way_symmetry_kernel<<<(int)blocks, threads, 0, stream>>>(
      dm3_in, dm3_out, n);
}
