#include <chrono>

namespace {

inline void throw_on_cuda_error_ws(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct Kernel25StageTimer {
  bool enabled = false;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;

  explicit Kernel25StageTimer(bool enable) : enabled(enable) {
    if (!enabled) return;
    throw_on_cuda_error_ws(cudaEventCreate(&ev_start), "cudaEventCreate(k25 stage start)");
    throw_on_cuda_error_ws(cudaEventCreate(&ev_stop), "cudaEventCreate(k25 stage stop)");
  }

  ~Kernel25StageTimer() {
    if (ev_stop) cudaEventDestroy(ev_stop);
    if (ev_start) cudaEventDestroy(ev_start);
  }

  void start(cudaStream_t stream) const {
    if (!enabled) return;
    throw_on_cuda_error_ws(cudaEventRecord(ev_start, stream), "cudaEventRecord(k25 stage start)");
  }

  float stop(cudaStream_t stream) const {
    if (!enabled) return 0.0f;
    throw_on_cuda_error_ws(cudaEventRecord(ev_stop, stream), "cudaEventRecord(k25 stage stop)");
    throw_on_cuda_error_ws(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(k25 stage stop)");
    float ms = 0.0f;
    throw_on_cuda_error_ws(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime(k25 stage)");
    return ms;
  }
};

inline void kernel25_profile_zero(Kernel25Profile* profile) {
  if (!profile) return;
  profile->count_ms = 0.0f;
  profile->prefix_sum_ms = 0.0f;
  profile->write_ms = 0.0f;
  profile->pack_ms = 0.0f;
  profile->sort_ms = 0.0f;
  profile->reduce_ms = 0.0f;
  profile->rle_ms = 0.0f;
  profile->indptr_ms = 0.0f;
  profile->unpack_ms = 0.0f;
  profile->sync_overhead_ms = 0.0f;
  profile->nnz_in = 0;
  profile->nnz_out = 0;
  profile->nrows = 0;
}

template <typename Fn>
inline void kernel25_profile_measure_sync_overhead(Kernel25Profile* profile, const Fn& fn) {
  if (!profile) {
    fn();
    return;
  }
  auto t0 = std::chrono::steady_clock::now();
  fn();
  auto t1 = std::chrono::steady_clock::now();
  profile->sync_overhead_ms += static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

struct Kernel25Workspace {
  int device = 0;
  int max_tasks = 0;
  int max_nnz_in = 0;
  int max_counts64 = 0;

  // Kernel 2B scratch (counts/offsets).
  int32_t* d_counts = nullptr;   // [max_tasks]
  int64_t* d_offsets = nullptr;  // [max_tasks+1]
  int64_t* d_counts64 = nullptr; // [max_counts64]

  // Kernel 2B write outputs (triples for Kernel 2.5).
  int32_t* d_k = nullptr;   // [max_nnz_in]
  double* d_c = nullptr;    // [max_nnz_in] (f64 coeff scratch)
  float* d_c_f32 = nullptr; // [max_nnz_in] (f32 coeff scratch)
  int32_t* d_j = nullptr;   // [max_nnz_in]
  int32_t* d_rs = nullptr;  // [max_nnz_in] (rs_id = r*norb+s or pq_id)

  // Kernel 2.5 outputs/metadata.
  uint64_t* d_row_jk = nullptr; // [max_nnz_in]
  int* d_nrows = nullptr;       // [1]
  int* d_nnz = nullptr;         // [1]

  // Kernel 2.5 internal buffers.
  int32_t* d_rs_alt = nullptr; // [max_nnz_in]
  K25Val1* d_v1_a = nullptr;   // [max_nnz_in]
  K25Val1* d_v1_b = nullptr;   // [max_nnz_in]
  uint64_t* d_jk_a = nullptr;  // [max_nnz_in]
  uint64_t* d_jk_b = nullptr;  // [max_nnz_in]
  K25Val2* d_v2_a = nullptr;   // [max_nnz_in]
  K25Val2* d_v2_b = nullptr;   // [max_nnz_in]
  K25Key* d_key_in = nullptr;  // [max_nnz_in]
  K25Key* d_key_out = nullptr; // [max_nnz_in]
  double* d_val_in = nullptr;  // [max_nnz_in] (f64 sort/reduce scratch)
  float* d_val_in_f32 = nullptr; // [max_nnz_in] (f32 sort/reduce scratch)
  int32_t* d_row_counts = nullptr; // [max_nnz_in]

  // For CUB temp-size query (avoid relying on output pointers at creation time).
  int64_t* d_indptr_tmp = nullptr; // [max_nnz_in+1]

  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  Kernel25Workspace(int max_tasks_, int max_nnz_in_) : max_tasks(max_tasks_), max_nnz_in(max_nnz_in_) {
    throw_on_cuda_error_ws(cudaGetDevice(&device), "cudaGetDevice");
    if (max_tasks <= 0) throw std::invalid_argument("max_tasks must be >= 1");
    if (max_nnz_in <= 0) throw std::invalid_argument("max_nnz_in must be >= 1");
    max_counts64 = (max_tasks > max_nnz_in) ? max_tasks : max_nnz_in;

    throw_on_cuda_error_ws(cudaMalloc((void**)&d_counts, (size_t)max_tasks * sizeof(int32_t)), "cudaMalloc(k25 counts)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_offsets, (size_t)(max_tasks + 1) * sizeof(int64_t)), "cudaMalloc(k25 offsets)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_counts64, (size_t)max_counts64 * sizeof(int64_t)), "cudaMalloc(k25 counts64)");

    throw_on_cuda_error_ws(cudaMalloc((void**)&d_k, (size_t)max_nnz_in * sizeof(int32_t)), "cudaMalloc(k25 k)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_c, (size_t)max_nnz_in * sizeof(double)), "cudaMalloc(k25 c)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_c_f32, (size_t)max_nnz_in * sizeof(float)), "cudaMalloc(k25 c_f32)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_j, (size_t)max_nnz_in * sizeof(int32_t)), "cudaMalloc(k25 j)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_rs, (size_t)max_nnz_in * sizeof(int32_t)), "cudaMalloc(k25 rs)");

    throw_on_cuda_error_ws(cudaMalloc((void**)&d_row_jk, (size_t)max_nnz_in * sizeof(uint64_t)), "cudaMalloc(k25 row_jk)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_nrows, sizeof(int)), "cudaMalloc(k25 nrows)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_nnz, sizeof(int)), "cudaMalloc(k25 nnz)");

    throw_on_cuda_error_ws(cudaMalloc((void**)&d_rs_alt, (size_t)max_nnz_in * sizeof(int32_t)), "cudaMalloc(k25 rs_alt)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_v1_a, (size_t)max_nnz_in * sizeof(K25Val1)), "cudaMalloc(k25 v1_a)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_v1_b, (size_t)max_nnz_in * sizeof(K25Val1)), "cudaMalloc(k25 v1_b)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_jk_a, (size_t)max_nnz_in * sizeof(uint64_t)), "cudaMalloc(k25 jk_a)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_jk_b, (size_t)max_nnz_in * sizeof(uint64_t)), "cudaMalloc(k25 jk_b)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_v2_a, (size_t)max_nnz_in * sizeof(K25Val2)), "cudaMalloc(k25 v2_a)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_v2_b, (size_t)max_nnz_in * sizeof(K25Val2)), "cudaMalloc(k25 v2_b)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_key_in, (size_t)max_nnz_in * sizeof(K25Key)), "cudaMalloc(k25 key_in)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_key_out, (size_t)max_nnz_in * sizeof(K25Key)), "cudaMalloc(k25 key_out)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_val_in, (size_t)max_nnz_in * sizeof(double)), "cudaMalloc(k25 val_in)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_val_in_f32, (size_t)max_nnz_in * sizeof(float)), "cudaMalloc(k25 val_in_f32)");
    throw_on_cuda_error_ws(
        cudaMalloc((void**)&d_row_counts, (size_t)max_nnz_in * sizeof(int32_t)), "cudaMalloc(k25 row_counts)");
    throw_on_cuda_error_ws(cudaMalloc((void**)&d_indptr_tmp, (size_t)(max_nnz_in + 1) * sizeof(int64_t)), "cudaMalloc(k25 indptr_tmp)");

    // Precompute a worst-case CUB temp storage size for the workspace capacities, so steady-state calls do not allocate.
    size_t temp_max = 0;
    size_t tmp = 0;
    cudaStream_t stream = 0;

    cub::DoubleBuffer<int32_t> keys_rs(d_rs, d_rs_alt);
    cub::DoubleBuffer<K25Val1> vals1(d_v1_a, d_v1_b);
    throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys_rs, vals1, max_nnz_in, 0, 32, stream),
        "cub::DeviceRadixSort::SortPairs(rs) query");
    temp_max = std::max(temp_max, tmp);

    cub::DoubleBuffer<uint64_t> keys_jk(d_jk_a, d_jk_b);
    cub::DoubleBuffer<K25Val2> vals2(d_v2_a, d_v2_b);
    throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys_jk, vals2, max_nnz_in, 0, 64, stream),
        "cub::DeviceRadixSort::SortPairs(jk) query");
    temp_max = std::max(temp_max, tmp);

    // Packed-key fast path (allpairs): sort (rowkey<<bits_rs | rs) with double values.
    cub::DoubleBuffer<double> vals_packed(d_c, d_val_in);
    throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys_jk, vals_packed, max_nnz_in, 0, 64, stream),
        "cub::DeviceRadixSort::SortPairs(packed) query");
    temp_max = std::max(temp_max, tmp);

    cub::DoubleBuffer<float> vals_packed_f32(d_c_f32, d_val_in_f32);
    throw_on_cuda_error_ws(
        cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys_jk, vals_packed_f32, max_nnz_in, 0, 64, stream),
        "cub::DeviceRadixSort::SortPairs(packed_f32) query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(nullptr, tmp, d_key_in, d_key_out, d_val_in, d_c, d_nnz, K25Sum(), max_nnz_in, stream),
        "cub::DeviceReduce::ReduceByKey query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(
            nullptr, tmp, d_key_in, d_key_out, d_val_in_f32, d_c_f32, d_nnz, K25SumT<float>(), max_nnz_in, stream),
        "cub::DeviceReduce::ReduceByKey(f32) query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(nullptr, tmp, d_jk_a, d_jk_b, d_c, d_val_in, d_nnz, K25Sum(), max_nnz_in, stream),
        "cub::DeviceReduce::ReduceByKey(packed) query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(
            nullptr, tmp, d_jk_a, d_jk_b, d_c_f32, d_val_in_f32, d_nnz, K25SumT<float>(), max_nnz_in, stream),
        "cub::DeviceReduce::ReduceByKey(packed_f32) query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceRunLengthEncode::Encode(nullptr, tmp, d_jk_a, d_row_jk, d_row_counts, d_nrows, max_nnz_in, stream),
        "cub::DeviceRunLengthEncode::Encode query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_counts64, d_indptr_tmp, max_nnz_in, stream),
        "cub::DeviceScan::ExclusiveSum(row) query");
    temp_max = std::max(temp_max, tmp);

    throw_on_cuda_error_ws(
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_counts64, d_offsets, max_tasks, stream),
        "cub::DeviceScan::ExclusiveSum(tasks) query");
    temp_max = std::max(temp_max, tmp);

    if (temp_max == 0) temp_max = 1;
    temp_bytes = temp_max;
    throw_on_cuda_error_ws(cudaMalloc(&d_temp, temp_bytes), "cudaMalloc(k25 temp)");
  }

  ~Kernel25Workspace() { release(); }

  Kernel25Workspace(const Kernel25Workspace&) = delete;
  Kernel25Workspace& operator=(const Kernel25Workspace&) = delete;

  void release() noexcept {
    if (d_temp) cudaFree(d_temp);
    if (d_indptr_tmp) cudaFree(d_indptr_tmp);
    if (d_row_counts) cudaFree(d_row_counts);
    if (d_val_in_f32) cudaFree(d_val_in_f32);
    if (d_val_in) cudaFree(d_val_in);
    if (d_key_out) cudaFree(d_key_out);
    if (d_key_in) cudaFree(d_key_in);
    if (d_v2_b) cudaFree(d_v2_b);
    if (d_v2_a) cudaFree(d_v2_a);
    if (d_jk_b) cudaFree(d_jk_b);
    if (d_jk_a) cudaFree(d_jk_a);
    if (d_v1_b) cudaFree(d_v1_b);
    if (d_v1_a) cudaFree(d_v1_a);
    if (d_rs_alt) cudaFree(d_rs_alt);
    if (d_nnz) cudaFree(d_nnz);
    if (d_nrows) cudaFree(d_nrows);
    if (d_row_jk) cudaFree(d_row_jk);
    if (d_rs) cudaFree(d_rs);
    if (d_j) cudaFree(d_j);
    if (d_c_f32) cudaFree(d_c_f32);
    if (d_c) cudaFree(d_c);
    if (d_k) cudaFree(d_k);
    if (d_counts64) cudaFree(d_counts64);
    if (d_offsets) cudaFree(d_offsets);
    if (d_counts) cudaFree(d_counts);

    d_temp = nullptr;
    d_indptr_tmp = nullptr;
    d_row_counts = nullptr;
    d_val_in_f32 = nullptr;
    d_val_in = nullptr;
    d_key_out = nullptr;
    d_key_in = nullptr;
    d_v2_b = nullptr;
    d_v2_a = nullptr;
    d_jk_b = nullptr;
    d_jk_a = nullptr;
    d_v1_b = nullptr;
    d_v1_a = nullptr;
    d_rs_alt = nullptr;
    d_nnz = nullptr;
    d_nrows = nullptr;
    d_row_jk = nullptr;
    d_rs = nullptr;
    d_j = nullptr;
    d_c_f32 = nullptr;
    d_c = nullptr;
    d_k = nullptr;
    d_counts64 = nullptr;
    d_offsets = nullptr;
    d_counts = nullptr;
    temp_bytes = 0;
  }
};

inline Kernel25Workspace* ws_from_handle(void* ws_handle) {
  if (!ws_handle) throw std::invalid_argument("Kernel25Workspace handle is null");
  return reinterpret_cast<Kernel25Workspace*>(ws_handle);
}

inline int current_device_checked() {
  int dev = 0;
  throw_on_cuda_error_ws(cudaGetDevice(&dev), "cudaGetDevice");
  return dev;
}

inline void ensure_ws_device(Kernel25Workspace* ws) {
  int cur = current_device_checked();
  if (cur != ws->device) {
    throw std::runtime_error("Kernel25Workspace was created on a different CUDA device");
  }
}

inline int ceil_log2_u32(uint32_t x) {
  if (x <= 1u) return 0;
  return 32 - __builtin_clz(x - 1u);
}

inline int64_t counts_to_offsets_total_ws(Kernel25Workspace* ws, int ntasks, cudaStream_t stream, Kernel25Profile* profile = nullptr) {
  if (ntasks < 0 || ntasks > ws->max_tasks) {
    throw std::invalid_argument("ntasks exceeds Kernel25Workspace.max_tasks (recreate workspace)");
  }
  if (ntasks == 0) return 0;

  Kernel25StageTimer stage_timer(profile != nullptr);
  int threads = 256;
  int blocks = (ntasks + threads - 1) / threads;
  stage_timer.start(stream);
  cast_i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(ws->d_counts, ws->d_counts64, ntasks);
  throw_on_cuda_error_ws(cudaGetLastError(), "cast_i32_to_i64_kernel(counts)");

  throw_on_cuda_error_ws(
      cub::DeviceScan::ExclusiveSum(ws->d_temp, ws->temp_bytes, ws->d_counts64, ws->d_offsets, ntasks, stream),
      "cub::DeviceScan::ExclusiveSum(counts)");

  counts_scan_write_total_kernel<<<1, 1, 0, stream>>>(ws->d_counts, ntasks, ws->d_offsets);
  throw_on_cuda_error_ws(cudaGetLastError(), "counts_scan_write_total_kernel");
  if (profile) {
    profile->prefix_sum_ms += stage_timer.stop(stream);
  }

  int64_t total = 0;
  kernel25_profile_measure_sync_overhead(profile, [&]() {
    throw_on_cuda_error_ws(cudaMemcpyAsync(&total, ws->d_offsets + ntasks, sizeof(int64_t), cudaMemcpyDeviceToHost, stream), "D2H total");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(total)");
  });
  return total;
}

__global__ void k25_cast_f64_to_f32_kernel(const double* __restrict__ in, float* __restrict__ out, int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  out[tid] = static_cast<float>(in[tid]);
}

template <typename CoeffT>
inline void kernel25_build_csr_allpairs_packed_ws_typed(
    Kernel25Workspace* ws,
    int nnz_in,
    bool coalesce,
    int64_t* out_indptr,
    int32_t* out_indices,
    CoeffT* out_data,
    CoeffT* coeff_in,
    CoeffT* coeff_tmp,
    cudaStream_t stream,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    Kernel25Profile* profile = nullptr) {
  if (profile) {
    profile->nnz_in = nnz_in;
  }
  Kernel25StageTimer stage_timer(profile != nullptr);
  if (nnz_in <= 0) {
    int zero = 0;
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nrows, &zero, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nrows=0");
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nnz, &zero, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nnz=0");
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    if (profile) {
      profile->nnz_out = 0;
      profile->nrows = 0;
    }
    return;
  }
  if (nnz_in > ws->max_nnz_in) {
    throw std::invalid_argument("nnz_in exceeds Kernel25Workspace.max_nnz_in (recreate workspace)");
  }

  int nops = norb * norb;
  if (nops <= 0 || ncsf <= 0 || j_count <= 0) {
    throw std::invalid_argument("invalid dimensions for packed CSR build");
  }
  int bits_rs = ceil_log2_u32((uint32_t)nops);
  int bits_k = ceil_log2_u32((uint32_t)ncsf);
  int bits_j = ceil_log2_u32((uint32_t)j_count);
  int bits_total = bits_rs + bits_k + bits_j;
  if (bits_total <= 0 || bits_total > 64) {
    throw std::runtime_error("packed CSR key width exceeds 64 bits (fallback required)");
  }
  uint64_t rs_mask = (bits_rs >= 63) ? ~0ull : ((1ull << (uint32_t)bits_rs) - 1ull);

  int threads = 256;
  int blocks = (nnz_in + threads - 1) / threads;

  stage_timer.start(stream);
  k25_build_packed_key_kernel<<<blocks, threads, 0, stream>>>(
      ws->d_j, ws->d_k, ws->d_rs, ws->d_jk_a, nnz_in, j_start, bits_k, bits_rs);
  throw_on_cuda_error_ws(cudaGetLastError(), "k25_build_packed_key_kernel");
  if (profile) {
    profile->pack_ms += stage_timer.stop(stream);
  }

  cub::DoubleBuffer<uint64_t> keys(ws->d_jk_a, ws->d_jk_b);
  cub::DoubleBuffer<CoeffT> vals(coeff_in, coeff_tmp);
  stage_timer.start(stream);
  throw_on_cuda_error_ws(
      cub::DeviceRadixSort::SortPairs(ws->d_temp, ws->temp_bytes, keys, vals, nnz_in, 0, bits_total, stream),
      "cub::DeviceRadixSort::SortPairs(packed)");
  if (profile) {
    profile->sort_ms += stage_timer.stop(stream);
  }

  int nnz_out = nnz_in;
  uint64_t* rowkey_inout = keys.Current();
  const CoeffT* vals_in = vals.Current();

  if (coalesce) {
    uint64_t* keys_out = (rowkey_inout == ws->d_jk_a) ? ws->d_jk_b : ws->d_jk_a;
    stage_timer.start(stream);
    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(
            ws->d_temp,
            ws->temp_bytes,
            rowkey_inout,
            keys_out,
            vals_in,
            out_data,
            ws->d_nnz,
            K25SumT<CoeffT>(),
            nnz_in,
            stream),
        "cub::DeviceReduce::ReduceByKey(packed)");
    if (profile) {
      profile->reduce_ms += stage_timer.stop(stream);
    }

    kernel25_profile_measure_sync_overhead(profile, [&]() {
      throw_on_cuda_error_ws(cudaMemcpyAsync(&nnz_out, ws->d_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nnz_out");
      throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nnz_out)");
    });
    if (nnz_out < 0) nnz_out = 0;
    if (nnz_out == 0) {
      int64_t zero64 = 0;
      throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
      throw_on_cuda_error_ws(cudaMemsetAsync(ws->d_nrows, 0, sizeof(int), stream), "cudaMemsetAsync(nrows=0)");
      if (profile) {
        profile->nnz_out = 0;
        profile->nrows = 0;
      }
      return;
    }

    rowkey_inout = keys_out;
    vals_in = out_data;
  } else {
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nnz, &nnz_out, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nnz_out");
  }

  blocks = (nnz_out + threads - 1) / threads;
  stage_timer.start(stream);
  k25_unpack_packed_key_and_vals_kernel_t<CoeffT><<<blocks, threads, 0, stream>>>(
      rowkey_inout, vals_in, nnz_out, out_indices, out_data, bits_rs, rs_mask);
  throw_on_cuda_error_ws(cudaGetLastError(), "k25_unpack_packed_key_and_vals_kernel_t");
  if (profile) {
    profile->unpack_ms += stage_timer.stop(stream);
  }

  stage_timer.start(stream);
  throw_on_cuda_error_ws(
      cub::DeviceRunLengthEncode::Encode(
          ws->d_temp, ws->temp_bytes, rowkey_inout, ws->d_row_jk, ws->d_row_counts, ws->d_nrows, nnz_out, stream),
      "cub::DeviceRunLengthEncode::Encode(packed)");
  if (profile) {
    profile->rle_ms += stage_timer.stop(stream);
  }

  int nrows = 0;
  kernel25_profile_measure_sync_overhead(profile, [&]() {
    throw_on_cuda_error_ws(cudaMemcpyAsync(&nrows, ws->d_nrows, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nrows");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nrows)");
  });
  if (nrows < 0) nrows = 0;
  if (nrows == 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    if (profile) {
      profile->nnz_out = nnz_out;
      profile->nrows = 0;
    }
    return;
  }

  blocks = (nrows + threads - 1) / threads;
  stage_timer.start(stream);
  k25_rowkey_to_rowjk_kernel<<<blocks, threads, 0, stream>>>(ws->d_row_jk, nrows, j_start, bits_k);
  throw_on_cuda_error_ws(cudaGetLastError(), "k25_rowkey_to_rowjk_kernel");
  if (profile) {
    profile->unpack_ms += stage_timer.stop(stream);
  }

  stage_timer.start(stream);
  cast_i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(ws->d_row_counts, ws->d_counts64, nrows);
  throw_on_cuda_error_ws(cudaGetLastError(), "cast_i32_to_i64_kernel(row_counts)");

  throw_on_cuda_error_ws(
      cub::DeviceScan::ExclusiveSum(ws->d_temp, ws->temp_bytes, ws->d_counts64, out_indptr, nrows, stream),
      "cub::DeviceScan::ExclusiveSum(indptr)");

  int64_t nnz64 = (int64_t)nnz_out;
  throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr + nrows, &nnz64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[nrows]=nnz");
  if (profile) {
    profile->indptr_ms += stage_timer.stop(stream);
    profile->nnz_out = nnz_out;
    profile->nrows = nrows;
  }
}

inline void kernel25_build_csr_ws(
    Kernel25Workspace* ws,
    int nnz_in,
    bool coalesce,
    int64_t* out_indptr,
    int32_t* out_indices,
    double* out_data,
    cudaStream_t stream,
    Kernel25Profile* profile = nullptr) {
  if (profile) {
    profile->nnz_in = nnz_in;
  }
  Kernel25StageTimer stage_timer(profile != nullptr);
  if (nnz_in <= 0) {
    int zero = 0;
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nrows, &zero, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nrows=0");
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nnz, &zero, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nnz=0");
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    if (profile) {
      profile->nnz_out = 0;
      profile->nrows = 0;
    }
    return;
  }
  if (nnz_in > ws->max_nnz_in) {
    throw std::invalid_argument("nnz_in exceeds Kernel25Workspace.max_nnz_in (recreate workspace)");
  }

  int threads = 256;
  int blocks = (nnz_in + threads - 1) / threads;

  stage_timer.start(stream);
  k25_build_val1_kernel<<<blocks, threads, 0, stream>>>(ws->d_j, ws->d_k, ws->d_c, ws->d_v1_a, nnz_in);
  throw_on_cuda_error_ws(cudaGetLastError(), "k25_build_val1_kernel");
  if (profile) {
    profile->pack_ms += stage_timer.stop(stream);
  }

  cub::DoubleBuffer<int32_t> keys_rs(ws->d_rs, ws->d_rs_alt);
  cub::DoubleBuffer<K25Val1> vals1(ws->d_v1_a, ws->d_v1_b);
  stage_timer.start(stream);
  throw_on_cuda_error_ws(
      cub::DeviceRadixSort::SortPairs(ws->d_temp, ws->temp_bytes, keys_rs, vals1, nnz_in, 0, 32, stream),
      "cub::DeviceRadixSort::SortPairs(rs)");
  if (profile) {
    profile->sort_ms += stage_timer.stop(stream);
  }

  stage_timer.start(stream);
  k25_build_sort2_inputs_kernel<<<blocks, threads, 0, stream>>>(keys_rs.Current(), vals1.Current(), ws->d_jk_a, ws->d_v2_a, nnz_in);
  throw_on_cuda_error_ws(cudaGetLastError(), "k25_build_sort2_inputs_kernel");
  if (profile) {
    profile->pack_ms += stage_timer.stop(stream);
  }

  cub::DoubleBuffer<uint64_t> keys_jk(ws->d_jk_a, ws->d_jk_b);
  cub::DoubleBuffer<K25Val2> vals2(ws->d_v2_a, ws->d_v2_b);
  stage_timer.start(stream);
  throw_on_cuda_error_ws(
      cub::DeviceRadixSort::SortPairs(ws->d_temp, ws->temp_bytes, keys_jk, vals2, nnz_in, 0, 64, stream),
      "cub::DeviceRadixSort::SortPairs(jk)");
  if (profile) {
    profile->sort_ms += stage_timer.stop(stream);
  }

  int nnz_out = nnz_in;
  uint64_t* d_jk_unique = keys_jk.Current();
  const K25Val2* d_v2_sorted = vals2.Current();

  if (coalesce) {
    stage_timer.start(stream);
    k25_build_reduce_inputs_kernel<<<blocks, threads, 0, stream>>>(keys_jk.Current(), vals2.Current(), ws->d_key_in, ws->d_val_in, nnz_in);
    throw_on_cuda_error_ws(cudaGetLastError(), "k25_build_reduce_inputs_kernel");
    if (profile) {
      profile->pack_ms += stage_timer.stop(stream);
    }

    stage_timer.start(stream);
    throw_on_cuda_error_ws(
        cub::DeviceReduce::ReduceByKey(
            ws->d_temp, ws->temp_bytes, ws->d_key_in, ws->d_key_out, ws->d_val_in, out_data, ws->d_nnz, K25Sum(), nnz_in, stream),
        "cub::DeviceReduce::ReduceByKey");
    if (profile) {
      profile->reduce_ms += stage_timer.stop(stream);
    }

    kernel25_profile_measure_sync_overhead(profile, [&]() {
      throw_on_cuda_error_ws(cudaMemcpyAsync(&nnz_out, ws->d_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nnz_out");
      throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nnz_out)");
    });
    if (nnz_out < 0) nnz_out = 0;

    blocks = (nnz_out + threads - 1) / threads;
    stage_timer.start(stream);
    k25_extract_unique_kernel<<<blocks, threads, 0, stream>>>(ws->d_key_out, d_jk_unique, out_indices, nnz_out);
    throw_on_cuda_error_ws(cudaGetLastError(), "k25_extract_unique_kernel");
    if (profile) {
      profile->unpack_ms += stage_timer.stop(stream);
    }
  } else {
    throw_on_cuda_error_ws(cudaMemcpyAsync(ws->d_nnz, &nnz_out, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D nnz_out");
    stage_timer.start(stream);
    k25_extract_sorted_kernel<<<blocks, threads, 0, stream>>>(d_v2_sorted, out_indices, out_data, nnz_in);
    throw_on_cuda_error_ws(cudaGetLastError(), "k25_extract_sorted_kernel");
    if (profile) {
      profile->unpack_ms += stage_timer.stop(stream);
    }
  }

  stage_timer.start(stream);
  throw_on_cuda_error_ws(
      cub::DeviceRunLengthEncode::Encode(ws->d_temp, ws->temp_bytes, d_jk_unique, ws->d_row_jk, ws->d_row_counts, ws->d_nrows, nnz_out, stream),
      "cub::DeviceRunLengthEncode::Encode");
  if (profile) {
    profile->rle_ms += stage_timer.stop(stream);
  }

  int nrows = 0;
  kernel25_profile_measure_sync_overhead(profile, [&]() {
    throw_on_cuda_error_ws(cudaMemcpyAsync(&nrows, ws->d_nrows, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nrows");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nrows)");
  });
  if (nrows < 0) nrows = 0;
  if (nrows == 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    if (profile) {
      profile->nnz_out = nnz_out;
      profile->nrows = 0;
    }
    return;
  }

  blocks = (nrows + threads - 1) / threads;
  stage_timer.start(stream);
  cast_i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(ws->d_row_counts, ws->d_counts64, nrows);
  throw_on_cuda_error_ws(cudaGetLastError(), "cast_i32_to_i64_kernel(row_counts)");

  throw_on_cuda_error_ws(
      cub::DeviceScan::ExclusiveSum(ws->d_temp, ws->temp_bytes, ws->d_counts64, out_indptr, nrows, stream),
      "cub::DeviceScan::ExclusiveSum(indptr)");

  int64_t nnz64 = (int64_t)nnz_out;
  throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr + nrows, &nnz64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[nrows]=nnz");
  if (profile) {
    profile->indptr_ms += stage_timer.stop(stream);
    profile->nnz_out = nnz_out;
    profile->nrows = nrows;
  }
}

} // namespace

extern "C" void* guga_kernel25_workspace_create(int max_tasks, int max_nnz_in) {
  return reinterpret_cast<void*>(new Kernel25Workspace(int(max_tasks), int(max_nnz_in)));
}

extern "C" void guga_kernel25_workspace_destroy(void* ws_handle) {
  auto* ws = reinterpret_cast<Kernel25Workspace*>(ws_handle);
  delete ws;
}

extern "C" void guga_kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_ws(
    void* ws_handle,
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    int32_t* out_row_j,
    int32_t* out_row_k,
    int64_t* out_indptr,
    int32_t* out_indices,
    void* out_data,
    int out_data_type,
    int max_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int coalesce,
    int* out_nrows_host,
    int* out_nnz_host,
    int* out_nnz_in_host,
    int sync,
    int check_overflow_mode,
    int use_fused_count_write,
    Kernel25Profile* out_profile) {
  Kernel25Workspace* ws = ws_from_handle(ws_handle);
  ensure_ws_device(ws);
  kernel25_profile_zero(out_profile);
  Kernel25StageTimer stage_timer(out_profile != nullptr);
  if (!child || !node_twos || !child_prefix || !steps_table || !nodes_table) {
    throw std::invalid_argument("null DRT/state pointers");
  }
  if (!out_row_j || !out_row_k || !out_indptr || !out_indices || !out_data || !overflow_flag) {
    throw std::invalid_argument("null output pointers");
  }
  if (!out_nrows_host || !out_nnz_host || !out_nnz_in_host) {
    throw std::invalid_argument("null host output pointers");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow_mode < 0 || check_overflow_mode > 2) {
    throw std::invalid_argument("check_overflow_mode must be 0 (none), 1 (deferred), or 2 (per-stage)");
  }
  if (use_fused_count_write != 0 && use_fused_count_write != 1) {
    throw std::invalid_argument("use_fused_count_write must be 0 or 1");
  }
  if (out_data_type != 4 && out_data_type != 8) {
    throw std::invalid_argument("out_data_type must be 4 (<f4) or 8 (<f8)");
  }
  const bool check_overflow_deferred = (check_overflow_mode == 1);
  const bool check_overflow_per_stage = (check_overflow_mode == 2);
  const bool fused_count_write = (use_fused_count_write != 0);
  if (max_out <= 0 || max_out > ws->max_nnz_in) {
    throw std::invalid_argument("output capacity exceeds Kernel25Workspace.max_nnz_in (recreate workspace)");
  }
  if (j_count <= 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    *out_nrows_host = 0;
    *out_nnz_host = 0;
    *out_nnz_in_host = 0;
    return;
  }

  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    *out_nrows_host = 0;
    *out_nnz_host = 0;
    *out_nnz_in_host = 0;
    return;
  }

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range (batch the work)");
  }
  int ntasks = (int)ntasks_ll;
  if (ntasks > ws->max_tasks) {
    throw std::invalid_argument("ntasks exceeds Kernel25Workspace.max_tasks (recreate workspace)");
  }

  int nnz_in = 0;
  void* coeff_scratch = (out_data_type == 4) ? reinterpret_cast<void*>(ws->d_c_f32) : reinterpret_cast<void*>(ws->d_c);
  void* coeff_sort_tmp = (out_data_type == 4) ? reinterpret_cast<void*>(ws->d_val_in_f32) : reinterpret_cast<void*>(ws->d_val_in);
  if (!fused_count_write) {
    throw_on_cuda_error_ws(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream), "cudaMemsetAsync(overflow=0)");
    stage_timer.start(stream);
    guga_epq_contribs_many_count_allpairs_launch_stream(
        child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, int(j_start), int(j_count), ws->d_counts, overflow_flag, stream, threads);
    throw_on_cuda_error_ws(cudaGetLastError(), "kernel launch(kernel2b_count_allpairs)");
    if (out_profile) {
      out_profile->count_ms += stage_timer.stop(stream);
    }

    if (check_overflow_per_stage) {
      int h_overflow = 0;
      kernel25_profile_measure_sync_overhead(out_profile, [&]() {
        throw_on_cuda_error_ws(cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H overflow(count)");
        throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel2b_count_allpairs)");
      });
      if (h_overflow) {
        throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
      }
    }

    int64_t total_out_ll = counts_to_offsets_total_ws(ws, ntasks, stream, out_profile);
    if (check_overflow_deferred) {
      int h_overflow = 0;
      kernel25_profile_measure_sync_overhead(out_profile, [&]() {
        throw_on_cuda_error_ws(cudaMemcpy(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow(deferred_count)");
      });
      if (h_overflow) {
        throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
      }
    }
    if (total_out_ll <= 0) {
      int64_t zero64 = 0;
      throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
      *out_nrows_host = 0;
      *out_nnz_host = 0;
      *out_nnz_in_host = 0;
      return;
    }
    if (total_out_ll > (int64_t)max_out) {
      throw std::runtime_error("Kernel 2B total output exceeds output buffer capacity");
    }
    if (total_out_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::runtime_error("Kernel 2B total output too large (batch the work)");
    }
    nnz_in = (int)total_out_ll;
    *out_nnz_in_host = nnz_in;
    if (out_profile) {
      out_profile->nnz_in = nnz_in;
    }

    throw_on_cuda_error_ws(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream), "cudaMemsetAsync(overflow=0)");
    stage_timer.start(stream);
    guga_epq_contribs_many_write_allpairs_launch_stream(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        int(j_start),
        int(j_count),
        ws->d_offsets,
        ws->d_k,
        coeff_scratch,
        out_data_type,
        ws->d_j,
        ws->d_rs,
        4,
        overflow_flag,
        stream,
        threads);
    throw_on_cuda_error_ws(cudaGetLastError(), "kernel launch(kernel2b_write_allpairs)");
    if (out_profile) {
      out_profile->write_ms += stage_timer.stop(stream);
    }

    if (check_overflow_per_stage) {
      int h_overflow = 0;
      kernel25_profile_measure_sync_overhead(out_profile, [&]() {
        throw_on_cuda_error_ws(cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H overflow(write)");
        throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel2b_write_allpairs)");
      });
      if (h_overflow) {
        throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
      }
    }
  } else {
    throw_on_cuda_error_ws(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream), "cudaMemsetAsync(overflow=0)");
    stage_timer.start(stream);
    guga_epq_contribs_many_fused_allpairs_launch_stream(
        child,
        node_twos,
        child_prefix,
        steps_table,
        nodes_table,
        ncsf,
        norb,
        int(j_start),
        int(j_count),
        max_out,
        ws->d_k,
        coeff_scratch,
        out_data_type,
        ws->d_j,
        ws->d_rs,
        4,
        ws->d_nnz,
        overflow_flag,
        stream,
        threads);
    throw_on_cuda_error_ws(cudaGetLastError(), "kernel launch(kernel2b_fused_allpairs)");
    if (out_profile) {
      // Fused path reports the full DRT walk/write as count_ms for now.
      out_profile->count_ms += stage_timer.stop(stream);
    }

    if (check_overflow_per_stage) {
      int h_overflow = 0;
      kernel25_profile_measure_sync_overhead(out_profile, [&]() {
        throw_on_cuda_error_ws(cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H overflow(fused)");
        throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel2b_fused_allpairs)");
      });
      if (h_overflow) {
        throw std::runtime_error("Kernel 2B fused kernel overflow (invalid indices, stack overflow, or output overflow)");
      }
    }

    int h_nnz_in = 0;
    kernel25_profile_measure_sync_overhead(out_profile, [&]() {
      throw_on_cuda_error_ws(cudaMemcpyAsync(&h_nnz_in, ws->d_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nnz_in(fused)");
      throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nnz_in_fused)");
    });
    int64_t total_out_ll = (int64_t)h_nnz_in;
    if (total_out_ll <= 0) {
      int64_t zero64 = 0;
      throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
      *out_nrows_host = 0;
      *out_nnz_host = 0;
      *out_nnz_in_host = 0;
      return;
    }
    if (total_out_ll > (int64_t)max_out) {
      throw std::runtime_error("Kernel 2B fused total output exceeds output buffer capacity");
    }
    if (total_out_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::runtime_error("Kernel 2B fused total output too large (batch the work)");
    }
    nnz_in = (int)total_out_ll;
    *out_nnz_in_host = nnz_in;
    if (out_profile) {
      out_profile->nnz_in = nnz_in;
    }
  }

  throw_on_cuda_error_ws(cudaMemsetAsync(ws->d_nrows, 0, sizeof(int), stream), "cudaMemsetAsync(nrows=0)");
  throw_on_cuda_error_ws(cudaMemsetAsync(ws->d_nnz, 0, sizeof(int), stream), "cudaMemsetAsync(nnz=0)");

  {
    int nops = norb * norb;
    int bits_rs = (nops > 0) ? ceil_log2_u32((uint32_t)nops) : 0;
    int bits_k = (ncsf > 0) ? ceil_log2_u32((uint32_t)ncsf) : 0;
    int bits_j = (j_count > 0) ? ceil_log2_u32((uint32_t)j_count) : 0;
    int bits_total = bits_rs + bits_k + bits_j;
    bool can_pack = (bits_total > 0 && bits_total <= 64);
    if (can_pack) {
      if (out_data_type == 4) {
        kernel25_build_csr_allpairs_packed_ws_typed<float>(
            ws,
            nnz_in,
            bool(coalesce),
            out_indptr,
            out_indices,
            reinterpret_cast<float*>(out_data),
            reinterpret_cast<float*>(coeff_scratch),
            reinterpret_cast<float*>(coeff_sort_tmp),
            stream,
            ncsf,
            norb,
            j_start,
            j_count,
            out_profile);
      } else {
        kernel25_build_csr_allpairs_packed_ws_typed<double>(
            ws,
            nnz_in,
            bool(coalesce),
            out_indptr,
            out_indices,
            reinterpret_cast<double*>(out_data),
            reinterpret_cast<double*>(coeff_scratch),
            reinterpret_cast<double*>(coeff_sort_tmp),
            stream,
            ncsf,
            norb,
            j_start,
            j_count,
            out_profile);
      }
    } else {
      if (out_data_type == 4) {
        throw std::runtime_error("float32 out_data requires packed-key CSR path (bits_total must be <= 64)");
      }
      kernel25_build_csr_ws(
          ws, nnz_in, bool(coalesce), out_indptr, out_indices, reinterpret_cast<double*>(out_data), stream, out_profile);
    }
  }

  int h_nrows = 0;
  int h_nnz = 0;
  int h_overflow_deferred = 0;
  kernel25_profile_measure_sync_overhead(out_profile, [&]() {
    if (check_overflow_deferred) {
      throw_on_cuda_error_ws(
          cudaMemcpyAsync(&h_overflow_deferred, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream),
          "D2H overflow(deferred_final)");
    }
    throw_on_cuda_error_ws(cudaMemcpyAsync(&h_nrows, ws->d_nrows, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nrows");
    throw_on_cuda_error_ws(cudaMemcpyAsync(&h_nnz, ws->d_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nnz");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel25_ws)");
  });
  if (check_overflow_deferred && h_overflow_deferred) {
    if (fused_count_write) {
      throw std::runtime_error("Kernel 2B fused kernel overflow (invalid indices, stack overflow, or output overflow)");
    }
    throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
  }
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;
  if (h_nrows > nnz_in || h_nrows > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nrows out of range for output buffers");
  }
  if (h_nnz > nnz_in || h_nnz > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nnz out of range for output buffers");
  }

  stage_timer.start(stream);
  throw_on_cuda_error_ws(guga_unpack_row_jk_launch_stream(ws->d_row_jk, h_nrows, out_row_j, out_row_k, stream, /*threads=*/256), "guga_unpack_row_jk_launch_stream");
  if (out_profile) {
    out_profile->unpack_ms += stage_timer.stop(stream);
    out_profile->nrows = h_nrows;
    out_profile->nnz_out = h_nnz;
  }
  if (sync) {
    kernel25_profile_measure_sync_overhead(out_profile, [&]() {
      throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(unpack_row_jk)");
    });
  }

  *out_nrows_host = h_nrows;
  *out_nnz_host = h_nnz;
}

#include "guga_cuda_qmc_kernels.cuh"

extern "C" void guga_kernel25_build_csr_from_tasks_deterministic_inplace_device_ws(
    void* ws_handle,
    const int32_t* child,
    const int16_t* node_twos,
    const int64_t* child_prefix,
    const int8_t* steps_table,
    const int32_t* nodes_table,
    int ncsf,
    int norb,
    const int32_t* task_csf,
    const int32_t* task_p,
    const int32_t* task_q,
    int ntasks,
    int32_t* out_row_j,
    int32_t* out_row_k,
    int64_t* out_indptr,
    int32_t* out_indices,
    void* out_data,
    int out_data_type,
    int max_out,
    int* overflow_flag,
    cudaStream_t stream,
    int threads,
    int coalesce,
    int* out_nrows_host,
    int* out_nnz_host,
    int* out_nnz_in_host,
    int sync,
    int check_overflow) {
  Kernel25Workspace* ws = ws_from_handle(ws_handle);
  ensure_ws_device(ws);
  if (!child || !node_twos || !child_prefix || !steps_table || !nodes_table) {
    throw std::invalid_argument("null DRT/state pointers");
  }
  if (!task_csf || !task_p || !task_q) {
    throw std::invalid_argument("null task pointers");
  }
  if (!out_row_j || !out_row_k || !out_indptr || !out_indices || !out_data || !overflow_flag) {
    throw std::invalid_argument("null output pointers");
  }
  if (!out_nrows_host || !out_nnz_host || !out_nnz_in_host) {
    throw std::invalid_argument("null host output pointers");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (out_data_type != 4 && out_data_type != 8) {
    throw std::invalid_argument("out_data_type must be 4 (<f4) or 8 (<f8)");
  }
  if (ntasks < 0 || ntasks > ws->max_tasks) {
    throw std::invalid_argument("ntasks exceeds Kernel25Workspace.max_tasks (recreate workspace)");
  }
  if (max_out <= 0 || max_out > ws->max_nnz_in) {
    throw std::invalid_argument("output capacity exceeds Kernel25Workspace.max_nnz_in (recreate workspace)");
  }
  if (ntasks == 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    *out_nrows_host = 0;
    *out_nnz_host = 0;
    *out_nnz_in_host = 0;
    return;
  }

  throw_on_cuda_error_ws(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream), "cudaMemsetAsync(overflow=0)");
  guga_epq_contribs_many_count_launch_stream(
      child, node_twos, child_prefix, steps_table, nodes_table, ncsf, norb, task_csf, task_p, task_q, ntasks, ws->d_counts, overflow_flag, stream, threads);
  throw_on_cuda_error_ws(cudaGetLastError(), "kernel launch(kernel2b_count)");

  if (check_overflow) {
    int h_overflow = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H overflow(count)");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel2b_count)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
    }
  }

  int64_t total_out_ll = counts_to_offsets_total_ws(ws, ntasks, stream);
  if (total_out_ll <= 0) {
    int64_t zero64 = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream), "H2D indptr[0]=0");
    *out_nrows_host = 0;
    *out_nnz_host = 0;
    *out_nnz_in_host = 0;
    return;
  }
  if (total_out_ll > (int64_t)max_out) {
    throw std::runtime_error("Kernel 2B total output exceeds output buffer capacity");
  }
  if (total_out_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("Kernel 2B total output too large (batch the work)");
  }
  int nnz_in = (int)total_out_ll;
  *out_nnz_in_host = nnz_in;

  throw_on_cuda_error_ws(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream), "cudaMemsetAsync(overflow=0)");
  guga_epq_contribs_many_write_launch_stream(
      child,
      node_twos,
      child_prefix,
      steps_table,
      nodes_table,
      ncsf,
      norb,
      task_csf,
      task_p,
      task_q,
      ntasks,
      ws->d_offsets,
      ws->d_k,
      ws->d_c,
      ws->d_j,
      ws->d_rs,
      overflow_flag,
      stream,
      threads);
  throw_on_cuda_error_ws(cudaGetLastError(), "kernel launch(kernel2b_write)");

  if (check_overflow) {
    int h_overflow = 0;
    throw_on_cuda_error_ws(cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H overflow(write)");
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel2b_write)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
    }
  }

  throw_on_cuda_error_ws(cudaMemsetAsync(ws->d_nrows, 0, sizeof(int), stream), "cudaMemsetAsync(nrows=0)");
  throw_on_cuda_error_ws(cudaMemsetAsync(ws->d_nnz, 0, sizeof(int), stream), "cudaMemsetAsync(nnz=0)");

  if (out_data_type == 8) {
    kernel25_build_csr_ws(ws, nnz_in, bool(coalesce), out_indptr, out_indices, reinterpret_cast<double*>(out_data), stream);
  } else {
    // Build CSR coefficients in fp64 workspace buffers, then cast to fp32 output.
    kernel25_build_csr_ws(ws, nnz_in, bool(coalesce), out_indptr, out_indices, ws->d_c, stream);
  }

  int h_nrows = 0;
  int h_nnz = 0;
  throw_on_cuda_error_ws(cudaMemcpyAsync(&h_nrows, ws->d_nrows, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nrows");
  throw_on_cuda_error_ws(cudaMemcpyAsync(&h_nnz, ws->d_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H nnz");
  throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel25_ws)");
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;
  if (h_nrows > nnz_in || h_nrows > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nrows out of range for output buffers");
  }
  if (h_nnz > nnz_in || h_nnz > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nnz out of range for output buffers");
  }

  if (out_data_type == 4 && h_nnz > 0) {
    int threads_cast = 256;
    int blocks_cast = (h_nnz + threads_cast - 1) / threads_cast;
    k25_cast_f64_to_f32_kernel<<<blocks_cast, threads_cast, 0, stream>>>(
        ws->d_c,
        reinterpret_cast<float*>(out_data),
        h_nnz);
    throw_on_cuda_error_ws(cudaGetLastError(), "k25_cast_f64_to_f32_kernel");
  }

  throw_on_cuda_error_ws(guga_unpack_row_jk_launch_stream(ws->d_row_jk, h_nrows, out_row_j, out_row_k, stream, /*threads=*/256), "guga_unpack_row_jk_launch_stream");
  if (sync) {
    throw_on_cuda_error_ws(cudaStreamSynchronize(stream), "cudaStreamSynchronize(unpack_row_jk)");
  }

  *out_nrows_host = h_nrows;
  *out_nnz_host = h_nnz;
}

extern "C" __global__ void scatter_embed_kernel(
    const double* __restrict__ x_sub,
    const int64_t* __restrict__ sub_to_full,
    double* __restrict__ x_full,
    int nsub) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nsub) {
    x_full[sub_to_full[i]] = x_sub[i];
  }
}

extern "C" __global__ void gather_project_kernel(
    const double* __restrict__ y_full,
    const int64_t* __restrict__ sub_to_full,
    double* __restrict__ y_sub,
    int nsub) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nsub) {
    y_sub[i] = y_full[sub_to_full[i]];
  }
}

extern "C" void launch_scatter_embed(
    const double* x_sub,
    const int64_t* sub_to_full,
    double* x_full,
    int nsub,
    int threads,
    cudaStream_t stream) {
  if (nsub <= 0) return;
  int blocks = (nsub + threads - 1) / threads;
  scatter_embed_kernel<<<blocks, threads, 0, stream>>>(x_sub, sub_to_full, x_full, nsub);
}

extern "C" void launch_gather_project(
    const double* y_full,
    const int64_t* sub_to_full,
    double* y_sub,
    int nsub,
    int threads,
    cudaStream_t stream) {
  if (nsub <= 0) return;
  int blocks = (nsub + threads - 1) / threads;
  gather_project_kernel<<<blocks, threads, 0, stream>>>(y_full, sub_to_full, y_sub, nsub);
}
