// CIPSI frontier-hash kernels: clear, extract (compact), score+topk.
//
// This header is included by guga_cuda_kernels.cu (aggregator TU).

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cub/cub.cuh>

namespace {

__global__ void cipsi_frontier_hash_clear_keys_kernel(int32_t* keys, int cap) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  keys[i] = (int32_t)-1;
}

__global__ void cipsi_frontier_hash_clear_vals_kernel(double* vals, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= n) return;
  vals[i] = 0.0;
}

__global__ void cipsi_frontier_hash_flags_kernel(const int32_t* keys, int cap, int* flags) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  flags[i] = (keys[i] >= 0) ? 1 : 0;
}

__global__ void cipsi_frontier_hash_scatter_extract_kernel(
    const int32_t* __restrict__ keys,
    const double* __restrict__ vals_root_major,
    int cap,
    int nroots,
    const int* __restrict__ offsets,
    const int* __restrict__ flags,
    int32_t* __restrict__ out_idx,
    double* __restrict__ out_vals_root_major) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= cap) return;
  if (!flags[i]) return;
  int pos = offsets[i];
  out_idx[pos] = keys[i];
  for (int r = 0; r < nroots; r++) {
    out_vals_root_major[(int64_t)r * (int64_t)cap + (int64_t)pos] =
        vals_root_major[(int64_t)r * (int64_t)cap + (int64_t)i];
  }
}

__global__ void cipsi_frontier_hash_write_nnz_kernel(const int* offsets, const int* flags, int cap, int* out_nnz) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int nnz = 0;
  if (cap > 0) {
    nnz = offsets[cap - 1] + flags[cap - 1];
  }
  out_nnz[0] = nnz;
}

// Score/owner/pt2 kernel. Produces score_bits (uint64) and tie_key (uint64).
__global__ void cipsi_score_owner_pt2_kernel(
    const int32_t* __restrict__ idx,               // [nnz]
    const double* __restrict__ vals_root_major,    // [nroots*nnz]
    int64_t vals_stride,                          // elements between roots (>= nnz)
    int nnz,
    int nroots,
    const double* __restrict__ e_var,              // [nroots]
    const double* __restrict__ hdiag,              // [ncsf]
    int ncsf,
    const uint8_t* __restrict__ selected_mask,     // [ncsf] or NULL
    double denom_floor,
    uint64_t* __restrict__ score_bits_out,         // [nnz]
    uint64_t* __restrict__ tie_key_out,            // [nnz]
    double* __restrict__ pt2_out) {                // [nroots]
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= nnz) return;

  int32_t csf = idx[i];
  if ((unsigned)csf >= (unsigned)ncsf) {
    score_bits_out[i] = 0;
    tie_key_out[i] = 0;
    return;
  }
  if (selected_mask && selected_mask[csf]) {
    // Internal CI-space elements are ignored for PT2/selection.
    score_bits_out[i] = 0;
    tie_key_out[i] = 0;
    return;
  }

  double h = hdiag[csf];
  double best = 0.0;
  int best_r = 0;

  for (int r = 0; r < nroots; r++) {
    double p = vals_root_major[(int64_t)r * (int64_t)vals_stride + (int64_t)i];
    double denom = e_var[r] - h;
    if (denom_floor > 0.0) {
      double ad = fabs(denom);
      if (ad < denom_floor) denom = (denom >= 0.0) ? denom_floor : -denom_floor;
    }
    double s = 0.0;
    if (denom != 0.0) s = fabs(p / denom);
    if (!(s > best)) {  // handles NaN
      // no-op
    } else {
      best = s;
      best_r = r;
    }
    // PT2 accumulation.
    if (denom != 0.0) {
      double contrib = (p * p) / denom;
      if (contrib != 0.0) atomicAdd(&pt2_out[r], contrib);
    }
  }

  uint64_t bits = 0;
  if (best > 0.0 && isfinite(best)) {
    bits = reinterpret_cast<const uint64_t&>(best);
  }
  score_bits_out[i] = bits;
  tie_key_out[i] = (((uint64_t)(uint32_t)best_r) << 32) | (uint64_t)(uint32_t)csf;
}

__global__ void cipsi_pt2_only_kernel(
    const int32_t* __restrict__ idx,               // [nnz]
    const double* __restrict__ vals_root_major,    // [nroots*vals_stride]
    int64_t vals_stride,                          // elements between roots (>= nnz)
    int nnz,
    int nroots,
    const double* __restrict__ e_var,              // [nroots]
    const double* __restrict__ hdiag,              // [ncsf]
    int ncsf,
    const uint8_t* __restrict__ selected_mask,     // [ncsf] or NULL
    double denom_floor,
    double* __restrict__ pt2_out) {                // [nroots]
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= nnz) return;
  int32_t csf = idx[i];
  if ((unsigned)csf >= (unsigned)ncsf) return;
  if (selected_mask && selected_mask[csf]) return;

  double h = hdiag[csf];
  for (int r = 0; r < nroots; r++) {
    double p = vals_root_major[(int64_t)r * (int64_t)vals_stride + (int64_t)i];
    double denom = e_var[r] - h;
    if (denom_floor > 0.0) {
      double ad = fabs(denom);
      if (ad < denom_floor) denom = (denom >= 0.0) ? denom_floor : -denom_floor;
    }
    if (denom != 0.0) {
      double contrib = (p * p) / denom;
      if (contrib != 0.0) atomicAdd(&pt2_out[r], contrib);
    }
  }
}

__global__ void cipsi_take_topk_kernel(
    const uint64_t* __restrict__ score_bits_sorted,  // [nnz]
    const uint64_t* __restrict__ tie_key_sorted,     // [nnz]
    int nnz,
    int max_add,
    int32_t* __restrict__ out_new_idx,               // [max_add]
    int* __restrict__ out_new_n) {                   // [1]
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int take = (max_add < nnz) ? max_add : nnz;
  int out_n = 0;
  for (int i = 0; i < take; i++) {
    uint64_t sb = score_bits_sorted[i];
    if (sb == 0) break;
    uint64_t tk = tie_key_sorted[i];
    int32_t idx = (int32_t)(uint32_t)(tk & 0xffffffffu);
    out_new_idx[i] = idx;
    out_n++;
  }
  out_new_n[0] = out_n;
}

__global__ void cipsi_score_nonzero_flags_kernel(
    const uint64_t* __restrict__ score_bits,
    int n,
    int* __restrict__ flags) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  flags[i] = (score_bits[i] != 0) ? 1 : 0;
}

inline bool is_pow2_i32(int x) {
  return (x > 0) && ((x & (x - 1)) == 0);
}

}  // namespace

extern "C" cudaError_t guga_cipsi_frontier_hash_clear_launch_stream(
    int32_t* keys,
    double* vals_root_major,
    int cap,
    int nroots,
    cudaStream_t stream,
    int threads) {
  if (!keys || !vals_root_major) return cudaErrorInvalidValue;
  if (cap < 0 || nroots < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (cap == 0 || nroots == 0) return cudaSuccess;
  if (!is_pow2_i32(cap)) return cudaErrorInvalidValue;

  // keys are int32 with sentinel -1, so 0xFF byte pattern is correct.
  cudaError_t err = cudaMemsetAsync(keys, 0xFF, (size_t)cap * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  // vals are float64 and zero-initialized with normal memset.
  return cudaMemsetAsync(vals_root_major, 0, (size_t)cap * (size_t)nroots * sizeof(double), stream);
}

extern "C" cudaError_t guga_cipsi_frontier_hash_extract_launch_stream(
    const int32_t* keys,
    const double* vals_root_major,
    int cap,
    int nroots,
    int32_t* out_idx,             // [cap]
    double* out_vals_root_major,  // [nroots*cap]
    int* out_nnz,                 // [1]
    cudaStream_t stream,
    int threads) {
  if (!keys || !vals_root_major || !out_idx || !out_vals_root_major || !out_nnz) return cudaErrorInvalidValue;
  if (cap < 0 || nroots < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (cap == 0 || nroots == 0) {
    return cudaMemsetAsync(out_nnz, 0, sizeof(int), stream);
  }
  if (!is_pow2_i32(cap)) return cudaErrorInvalidValue;

  // flags/offsets scratch on device (allocated per-call; caller may wrap in a higher-level workspace).
  int* d_flags_raw = nullptr;
  int* d_offsets_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_flags_raw, (size_t)cap * sizeof(int), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_offsets_raw, (size_t)cap * sizeof(int), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int> d_flags(d_flags_raw, CudaFreeStreamDeleter<int>{stream});
  cuda_unique_ptr_stream<int> d_offsets(d_offsets_raw, CudaFreeStreamDeleter<int>{stream});

  int blocks = (cap + threads - 1) / threads;
  cipsi_frontier_hash_flags_kernel<<<blocks, threads, 0, stream>>>(keys, cap, d_flags.get());
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Exclusive scan flags -> offsets.
  size_t temp_bytes = 0;
  err = cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_flags.get(), d_offsets.get(), cap, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp((void*)d_temp_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceScan::ExclusiveSum(d_temp.get(), temp_bytes, d_flags.get(), d_offsets.get(), cap, stream);
  if (err != cudaSuccess) return err;

  // Scatter compacted idx/vals into out arrays; still sized to cap.
  cipsi_frontier_hash_scatter_extract_kernel<<<blocks, threads, 0, stream>>>(
      keys, vals_root_major, cap, nroots, d_offsets.get(), d_flags.get(), out_idx, out_vals_root_major);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Write out_nnz = offsets[last] + flags[last].
  cipsi_frontier_hash_write_nnz_kernel<<<1, 1, 0, stream>>>(d_offsets.get(), d_flags.get(), cap, out_nnz);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_cipsi_score_and_select_topk_launch_stream(
    const int32_t* idx,            // [nnz]
    const double* vals_root_major, // [nroots*nnz]
    int64_t vals_stride,           // elements between roots (>= nnz)
    int nnz,
    int nroots,
    const double* e_var,           // [nroots]
    const double* hdiag,           // [ncsf]
    int ncsf,
    const uint8_t* selected_mask,  // [ncsf] or NULL
    double denom_floor,
    int max_add,
    int32_t* out_new_idx,          // [max_add]
    int* out_new_n,                // [1]
    double* out_pt2,               // [nroots]
    cudaStream_t stream,
    int threads) {
  if (!idx || !vals_root_major || !e_var || !hdiag || !out_new_n || !out_pt2) return cudaErrorInvalidValue;
  if (max_add > 0 && !out_new_idx) return cudaErrorInvalidValue;
  if (nnz < 0 || nroots < 0 || ncsf < 0 || max_add < 0) return cudaErrorInvalidValue;
  if (vals_stride < (int64_t)nnz) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;

  // Trivial.
  if (nnz == 0 || nroots == 0) {
    cudaError_t err0 = cudaMemsetAsync(out_new_n, 0, sizeof(int), stream);
    if (err0 != cudaSuccess) return err0;
    return cudaMemsetAsync(out_pt2, 0, (size_t)std::max(0, nroots) * sizeof(double), stream);
  }

  // PT2-only mode (no selection/sort).
  if (max_add == 0) {
    cudaError_t err0 = cudaMemsetAsync(out_pt2, 0, (size_t)nroots * sizeof(double), stream);
    if (err0 != cudaSuccess) return err0;
    int blocks0 = (nnz + threads - 1) / threads;
    cipsi_pt2_only_kernel<<<blocks0, threads, 0, stream>>>(
        idx, vals_root_major, vals_stride, nnz, nroots, e_var, hdiag, ncsf, selected_mask, denom_floor, out_pt2);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) return err1;
    return cudaMemsetAsync(out_new_n, 0, sizeof(int), stream);
  }

  // score_bits + tie_key buffers.
  uint64_t* d_score_a_raw = nullptr;
  uint64_t* d_score_b_raw = nullptr;
  uint64_t* d_tie_a_raw = nullptr;
  uint64_t* d_tie_b_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_score_a_raw, (size_t)nnz * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_score_b_raw, (size_t)nnz * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_tie_a_raw, (size_t)nnz * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_tie_b_raw, (size_t)nnz * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<uint64_t> d_score_a(d_score_a_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_score_b(d_score_b_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_tie_a(d_tie_a_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_tie_b(d_tie_b_raw, CudaFreeStreamDeleter<uint64_t>{stream});

  // Zero pt2.
  err = cudaMemsetAsync(out_pt2, 0, (size_t)nroots * sizeof(double), stream);
  if (err != cudaSuccess) return err;

  int blocks = (nnz + threads - 1) / threads;
  cipsi_score_owner_pt2_kernel<<<blocks, threads, 0, stream>>>(
      idx, vals_root_major, vals_stride, nnz, nroots, e_var, hdiag, ncsf, selected_mask, denom_floor,
      d_score_a.get(), d_tie_a.get(), out_pt2);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Pass 1: stable sort by tie_key ascending (keys=tie, vals=score).
  cub::DoubleBuffer<uint64_t> keys_tie(d_tie_a.get(), d_tie_b.get());
  cub::DoubleBuffer<uint64_t> vals_score(d_score_a.get(), d_score_b.get());
  size_t temp_sort1 = 0;
  err = cub::DeviceRadixSort::SortPairs(nullptr, temp_sort1, keys_tie, vals_score, nnz, 0, 64, stream);
  if (err != cudaSuccess) return err;

  // Pass 2: stable sort by score descending (keys=score, vals=tie).
  cub::DoubleBuffer<uint64_t> keys_score(vals_score.Current(), vals_score.Alternate());
  cub::DoubleBuffer<uint64_t> vals_tie(keys_tie.Current(), keys_tie.Alternate());
  size_t temp_sort2 = 0;
  err = cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_sort2, keys_score, vals_tie, nnz, 0, 64, stream);
  if (err != cudaSuccess) return err;

  size_t temp_bytes = (temp_sort1 > temp_sort2) ? temp_sort1 : temp_sort2;
  void* d_temp_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp((void*)d_temp_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceRadixSort::SortPairs(d_temp.get(), temp_sort1, keys_tie, vals_score, nnz, 0, 64, stream);
  if (err != cudaSuccess) return err;

  // Recreate score/tie buffers after pass1.
  keys_score = cub::DoubleBuffer<uint64_t>(vals_score.Current(), vals_score.Alternate());
  vals_tie = cub::DoubleBuffer<uint64_t>(keys_tie.Current(), keys_tie.Alternate());

  err = cub::DeviceRadixSort::SortPairsDescending(d_temp.get(), temp_sort2, keys_score, vals_tie, nnz, 0, 64, stream);
  if (err != cudaSuccess) return err;

  // Take first max_add nonzero scores.
  cipsi_take_topk_kernel<<<1, 1, 0, stream>>>(
      keys_score.Current(), vals_tie.Current(), nnz, max_add, out_new_idx, out_new_n);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_cipsi_score_and_select_topk_from_hash_slots_launch_stream(
    const int32_t* keys,           // [cap]
    const double* vals_root_major, // [nroots*cap]
    int cap,
    int nroots,
    const double* e_var,           // [nroots]
    const double* hdiag,           // [ncsf]
    int ncsf,
    const uint8_t* selected_mask,  // [ncsf] or NULL
    double denom_floor,
    int max_add,
    int32_t* out_new_idx,          // [max_add]
    int* out_new_n,                // [1]
    double* out_pt2,               // [nroots]
    cudaStream_t stream,
    int threads) {
  if (!keys || !vals_root_major || !e_var || !hdiag || !out_new_n || !out_pt2) return cudaErrorInvalidValue;
  if (max_add > 0 && !out_new_idx) return cudaErrorInvalidValue;
  if (cap < 0 || nroots < 0 || ncsf < 0 || max_add < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (cap == 0 || nroots == 0) {
    cudaError_t err0 = cudaMemsetAsync(out_new_n, 0, sizeof(int), stream);
    if (err0 != cudaSuccess) return err0;
    return cudaMemsetAsync(out_pt2, 0, (size_t)std::max(0, nroots) * sizeof(double), stream);
  }

  // PT2-only mode: avoid sort/selection work.
  if (max_add == 0) {
    cudaError_t err0 = cudaMemsetAsync(out_pt2, 0, (size_t)nroots * sizeof(double), stream);
    if (err0 != cudaSuccess) return err0;
    int blocks0 = (cap + threads - 1) / threads;
    cipsi_pt2_only_kernel<<<blocks0, threads, 0, stream>>>(
        keys, vals_root_major, (int64_t)cap, cap, nroots, e_var, hdiag, ncsf, selected_mask, denom_floor, out_pt2);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) return err1;
    return cudaMemsetAsync(out_new_n, 0, sizeof(int), stream);
  }

  // score/tie buffers.
  uint64_t* d_score_a_raw = nullptr;
  uint64_t* d_score_b_raw = nullptr;
  uint64_t* d_tie_a_raw = nullptr;
  uint64_t* d_tie_b_raw = nullptr;
  cudaError_t err = guga_cuda_malloc(&d_score_a_raw, (size_t)cap * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_score_b_raw, (size_t)cap * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_tie_a_raw, (size_t)cap * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_tie_b_raw, (size_t)cap * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<uint64_t> d_score_a(d_score_a_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_score_b(d_score_b_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_tie_a(d_tie_a_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_tie_b(d_tie_b_raw, CudaFreeStreamDeleter<uint64_t>{stream});

  // flags + selected count for compaction.
  int* d_flags_raw = nullptr;
  int* d_nsel_raw = nullptr;
  err = guga_cuda_malloc(&d_flags_raw, (size_t)cap * sizeof(int), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_nsel_raw, sizeof(int), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int> d_flags(d_flags_raw, CudaFreeStreamDeleter<int>{stream});
  cuda_unique_ptr_stream<int> d_nsel(d_nsel_raw, CudaFreeStreamDeleter<int>{stream});

  // Zero pt2.
  err = cudaMemsetAsync(out_pt2, 0, (size_t)nroots * sizeof(double), stream);
  if (err != cudaSuccess) return err;

  // Score all hash slots in-place (invalid/empty slots produce score 0).
  int blocks = (cap + threads - 1) / threads;
  cipsi_score_owner_pt2_kernel<<<blocks, threads, 0, stream>>>(
      keys, vals_root_major, (int64_t)cap, cap, nroots, e_var, hdiag, ncsf, selected_mask, denom_floor,
      d_score_a.get(), d_tie_a.get(), out_pt2);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  cipsi_score_nonzero_flags_kernel<<<blocks, threads, 0, stream>>>(d_score_a.get(), cap, d_flags.get());
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  // Compact nonzero scores/ties so radix sorts run on nnz instead of full cap.
  size_t temp_select_score = 0;
  size_t temp_select_tie = 0;
  err = cub::DeviceSelect::Flagged(
      nullptr, temp_select_score, d_score_a.get(), d_flags.get(), d_score_b.get(), d_nsel.get(), cap, stream);
  if (err != cudaSuccess) return err;
  err = cub::DeviceSelect::Flagged(
      nullptr, temp_select_tie, d_tie_a.get(), d_flags.get(), d_tie_b.get(), d_nsel.get(), cap, stream);
  if (err != cudaSuccess) return err;

  size_t temp_select = (temp_select_score > temp_select_tie) ? temp_select_score : temp_select_tie;
  void* d_temp_select_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_select_raw, temp_select, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_select((void*)d_temp_select_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceSelect::Flagged(
      d_temp_select.get(), temp_select_score, d_score_a.get(), d_flags.get(), d_score_b.get(), d_nsel.get(), cap, stream);
  if (err != cudaSuccess) return err;
  err = cub::DeviceSelect::Flagged(
      d_temp_select.get(), temp_select_tie, d_tie_a.get(), d_flags.get(), d_tie_b.get(), d_nsel.get(), cap, stream);
  if (err != cudaSuccess) return err;

  int h_nsel = 0;
  err = cudaMemcpyAsync(&h_nsel, d_nsel.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) return err;
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;

  if (h_nsel <= 0) {
    return cudaMemsetAsync(out_new_n, 0, sizeof(int), stream);
  }
  if (h_nsel > cap) h_nsel = cap;

  // Pass 1: stable sort compacted entries by tie_key ascending.
  cub::DoubleBuffer<uint64_t> keys_tie(d_tie_b.get(), d_tie_a.get());
  cub::DoubleBuffer<uint64_t> vals_score(d_score_b.get(), d_score_a.get());
  size_t temp_sort1 = 0;
  err = cub::DeviceRadixSort::SortPairs(nullptr, temp_sort1, keys_tie, vals_score, h_nsel, 0, 64, stream);
  if (err != cudaSuccess) return err;

  // Pass 2: stable sort by score descending.
  cub::DoubleBuffer<uint64_t> keys_score(vals_score.Current(), vals_score.Alternate());
  cub::DoubleBuffer<uint64_t> vals_tie(keys_tie.Current(), keys_tie.Alternate());
  size_t temp_sort2 = 0;
  err = cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_sort2, keys_score, vals_tie, h_nsel, 0, 64, stream);
  if (err != cudaSuccess) return err;

  size_t temp_sort = (temp_sort1 > temp_sort2) ? temp_sort1 : temp_sort2;
  void* d_temp_sort_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_sort_raw, temp_sort, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp_sort((void*)d_temp_sort_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceRadixSort::SortPairs(
      d_temp_sort.get(), temp_sort1, keys_tie, vals_score, h_nsel, 0, 64, stream);
  if (err != cudaSuccess) return err;

  keys_score = cub::DoubleBuffer<uint64_t>(vals_score.Current(), vals_score.Alternate());
  vals_tie = cub::DoubleBuffer<uint64_t>(keys_tie.Current(), keys_tie.Alternate());

  err = cub::DeviceRadixSort::SortPairsDescending(
      d_temp_sort.get(), temp_sort2, keys_score, vals_tie, h_nsel, 0, 64, stream);
  if (err != cudaSuccess) return err;

  cipsi_take_topk_kernel<<<1, 1, 0, stream>>>(
      keys_score.Current(), vals_tie.Current(), h_nsel, max_add, out_new_idx, out_new_n);
  return cudaGetLastError();
}

// V2 path: deterministic full-cap sort without host synchronization.
// This avoids the DeviceSelect + D2H nsel synchronization point used by v1.
extern "C" cudaError_t guga_cipsi_score_and_select_topk_from_hash_slots_v2_launch_stream(
    const int32_t* keys,           // [cap]
    const double* vals_root_major, // [nroots*cap]
    int cap,
    int nroots,
    const double* e_var,           // [nroots]
    const double* hdiag,           // [ncsf]
    int ncsf,
    const uint8_t* selected_mask,  // [ncsf] or NULL
    double denom_floor,
    int max_add,
    int32_t* out_new_idx,          // [max_add]
    int* out_new_n,                // [1]
    double* out_pt2,               // [nroots]
    cudaStream_t stream,
    int threads) {
  return guga_cipsi_score_and_select_topk_launch_stream(
      keys,
      vals_root_major,
      (int64_t)cap,
      cap,
      nroots,
      e_var,
      hdiag,
      ncsf,
      selected_mask,
      denom_floor,
      max_add,
      out_new_idx,
      out_new_n,
      out_pt2,
      stream,
      threads);
}
