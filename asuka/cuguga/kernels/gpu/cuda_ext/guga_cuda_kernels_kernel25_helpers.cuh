namespace {

template <typename T>
struct CudaFreeDeleter {
  void operator()(T* ptr) const noexcept {
    if (ptr) cudaFree(ptr);
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaFreeDeleter<T>>;

inline bool guga_cuda_async_alloc_supported() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
  static int cached = -1;
  if (cached >= 0) return cached != 0;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cached = 0;
    return false;
  }
  int supported = 0;
  if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev) != cudaSuccess) {
    cached = 0;
    return false;
  }
  cached = supported ? 1 : 0;
  return cached != 0;
#else
  return false;
#endif
}

template <typename T>
cudaError_t guga_cuda_malloc(T** ptr, size_t bytes, cudaStream_t stream) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
  if (guga_cuda_async_alloc_supported()) {
    return cudaMallocAsync(reinterpret_cast<void**>(ptr), bytes, stream);
  }
#endif
  (void)stream;
  return cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
}

template <typename T>
struct CudaFreeStreamDeleter {
  cudaStream_t stream = 0;
  void operator()(T* ptr) const noexcept {
    if (!ptr) return;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (guga_cuda_async_alloc_supported()) {
      cudaFreeAsync(ptr, stream);
      return;
    }
#endif
    (void)stream;
    cudaFree(ptr);
  }
};

template <typename T>
using cuda_unique_ptr_stream = std::unique_ptr<T, CudaFreeStreamDeleter<T>>;

struct K25Val1 {
  uint64_t jk;
  double c;
};

struct K25Val2 {
  int32_t rs;
  double c;
};

struct K25Key {
  uint64_t jk;
  int32_t rs;
  __host__ __device__ bool operator==(const K25Key& other) const { return (jk == other.jk) && (rs == other.rs); }
};

template <typename T>
struct K25SumT {
  __host__ __device__ T operator()(T a, T b) const { return a + b; }
};

using K25Sum = K25SumT<double>;

__global__ void k25_build_val1_kernel(
    const int32_t* __restrict__ j,
    const int32_t* __restrict__ k,
    const double* __restrict__ c,
    K25Val1* __restrict__ out,
    int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  uint32_t ju = (uint32_t)j[tid];
  uint32_t ku = (uint32_t)k[tid];
  out[tid].jk = (((uint64_t)ju) << 32) | (uint64_t)ku;
  out[tid].c = c[tid];
}

__global__ void k25_build_sort2_inputs_kernel(
    const int32_t* __restrict__ rs_sorted,
    const K25Val1* __restrict__ v1_sorted,
    uint64_t* __restrict__ jk_out,
    K25Val2* __restrict__ v2_out,
    int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  jk_out[tid] = v1_sorted[tid].jk;
  v2_out[tid].rs = rs_sorted[tid];
  v2_out[tid].c = v1_sorted[tid].c;
}

__global__ void k25_build_reduce_inputs_kernel(
    const uint64_t* __restrict__ jk_sorted,
    const K25Val2* __restrict__ v2_sorted,
    K25Key* __restrict__ key_out,
    double* __restrict__ val_out,
    int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  key_out[tid].jk = jk_sorted[tid];
  key_out[tid].rs = v2_sorted[tid].rs;
  val_out[tid] = v2_sorted[tid].c;
}

__global__ void k25_extract_unique_kernel(
    const K25Key* __restrict__ keys,
    uint64_t* __restrict__ jk_out,
    int32_t* __restrict__ rs_out,
    int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  jk_out[tid] = keys[tid].jk;
  rs_out[tid] = keys[tid].rs;
}

__global__ void k25_extract_sorted_kernel(
    const K25Val2* __restrict__ v2_sorted,
    int32_t* __restrict__ rs_out,
    double* __restrict__ c_out,
    int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  rs_out[tid] = v2_sorted[tid].rs;
  c_out[tid] = v2_sorted[tid].c;
}

__global__ void cast_i32_to_i64_kernel(const int32_t* __restrict__ in, int64_t* __restrict__ out, int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  out[tid] = (int64_t)in[tid];
}

__global__ void counts_scan_write_total_kernel(const int32_t* __restrict__ counts, int ntasks, int64_t* __restrict__ offsets) {
  if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;
  if (ntasks <= 0) return;
  offsets[ntasks] = offsets[ntasks - 1] + (int64_t)counts[ntasks - 1];
}

__global__ void unpack_row_jk_kernel(const uint64_t* __restrict__ row_jk, int nrows, int32_t* __restrict__ row_j,
                                     int32_t* __restrict__ row_k) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= nrows) return;
  uint64_t key = row_jk[tid];
  row_j[tid] = (int32_t)((uint32_t)(key >> 32));
  row_k[tid] = (int32_t)((uint32_t)key);
}

__global__ void k25_build_packed_key_kernel(
    const int32_t* __restrict__ j,
    const int32_t* __restrict__ k,
    const int32_t* __restrict__ rs,
    uint64_t* __restrict__ key_out,
    int n,
    int j_start,
    int bits_k,
    int bits_rs) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;

  uint32_t ju = (uint32_t)j[tid];
  uint32_t ku = (uint32_t)k[tid];
  uint32_t rsu = (uint32_t)rs[tid];

  uint32_t j_off = ju - (uint32_t)j_start;
  uint64_t rowkey = (((uint64_t)j_off) << (uint32_t)bits_k) | (uint64_t)ku;
  key_out[tid] = (rowkey << (uint32_t)bits_rs) | (uint64_t)rsu;
}

template <typename CoeffT>
__global__ void k25_unpack_packed_key_and_vals_kernel_t(
    uint64_t* __restrict__ keys_inout,
    const CoeffT* __restrict__ vals_in,
    int n,
    int32_t* __restrict__ out_indices,
    CoeffT* __restrict__ out_data,
    int bits_rs,
    uint64_t rs_mask) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  uint64_t key = keys_inout[tid];
  out_indices[tid] = (int32_t)(uint32_t)(key & rs_mask);
  out_data[tid] = vals_in[tid];
  keys_inout[tid] = key >> (uint32_t)bits_rs;
}

__global__ void k25_rowkey_to_rowjk_kernel(uint64_t* __restrict__ rowkey_inout, int nrows, int j_start, int bits_k) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= nrows) return;
  uint64_t rowkey = rowkey_inout[tid];
  uint64_t k_mask = (bits_k >= 63) ? ~0ull : ((1ull << (uint32_t)bits_k) - 1ull);
  uint32_t ku = (uint32_t)(rowkey & k_mask);
  uint32_t j_off = (uint32_t)(rowkey >> (uint32_t)bits_k);
  uint32_t ju = (uint32_t)j_start + j_off;
  rowkey_inout[tid] = (((uint64_t)ju) << 32) | (uint64_t)ku;
}

}  // namespace

extern "C" cudaError_t guga_unpack_row_jk_launch_stream(
    const uint64_t* row_jk,
    int nrows,
    int32_t* row_j,
    int32_t* row_k,
    cudaStream_t stream,
    int threads) {
  if (!row_jk || !row_j || !row_k) return cudaErrorInvalidValue;
  if (nrows < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0) return cudaSuccess;
  int blocks = (nrows + threads - 1) / threads;
  unpack_row_jk_kernel<<<blocks, threads, 0, stream>>>(row_jk, nrows, row_j, row_k);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_kernel25_build_csr_launch(
    const int32_t* j,
    const int32_t* k,
    const int32_t* rs,
    const double* c,
    int nnz_in,
    int coalesce,
    uint64_t* out_row_jk,
    int64_t* out_indptr,
    int32_t* out_indices,
    double* out_data,
    int* out_nrows,
    int* out_nnz);

extern "C" cudaError_t guga_kernel25_build_csr_launch_stream(
    const int32_t* j,
    const int32_t* k,
    const int32_t* rs,
    const double* c,
    int nnz_in,
    int coalesce,
    uint64_t* out_row_jk,
    int64_t* out_indptr,
    int32_t* out_indices,
    double* out_data,
    int* out_nrows,
    int* out_nnz,
    cudaStream_t stream) {
  if (nnz_in <= 0) {
    int zero = 0;
    int64_t zero64 = 0;
    cudaMemcpyAsync(out_nrows, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out_nnz, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out_indptr, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    return cudaSuccess;
  }

  int threads = 256;
  int blocks = (nnz_in + threads - 1) / threads;

  cudaError_t err = cudaSuccess;
  int32_t* d_rs_alt_raw = nullptr;
  err = guga_cuda_malloc(&d_rs_alt_raw, (size_t)nnz_in * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_rs_alt(d_rs_alt_raw, CudaFreeStreamDeleter<int32_t>{stream});

  K25Val1* d_v1_a_raw = nullptr;
  K25Val1* d_v1_b_raw = nullptr;
  err = guga_cuda_malloc(&d_v1_a_raw, (size_t)nnz_in * sizeof(K25Val1), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_v1_b_raw, (size_t)nnz_in * sizeof(K25Val1), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<K25Val1> d_v1_a(d_v1_a_raw, CudaFreeStreamDeleter<K25Val1>{stream});
  cuda_unique_ptr_stream<K25Val1> d_v1_b(d_v1_b_raw, CudaFreeStreamDeleter<K25Val1>{stream});

  k25_build_val1_kernel<<<blocks, threads, 0, stream>>>(j, k, c, d_v1_a.get(), nnz_in);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  cub::DoubleBuffer<int32_t> keys_rs(const_cast<int32_t*>(rs), d_rs_alt.get());
  cub::DoubleBuffer<K25Val1> vals1(d_v1_a.get(), d_v1_b.get());

  size_t temp_bytes = 0;
  err = cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_rs, vals1, nnz_in, 0, 32, stream);
  if (err != cudaSuccess) return err;
  void* d_temp_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp((void*)d_temp_raw, CudaFreeStreamDeleter<void>{stream});
  err = cub::DeviceRadixSort::SortPairs(d_temp.get(), temp_bytes, keys_rs, vals1, nnz_in, 0, 32, stream);
  if (err != cudaSuccess) return err;

  uint64_t* d_jk_a_raw = nullptr;
  uint64_t* d_jk_b_raw = nullptr;
  err = guga_cuda_malloc(&d_jk_a_raw, (size_t)nnz_in * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_jk_b_raw, (size_t)nnz_in * sizeof(uint64_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<uint64_t> d_jk_a(d_jk_a_raw, CudaFreeStreamDeleter<uint64_t>{stream});
  cuda_unique_ptr_stream<uint64_t> d_jk_b(d_jk_b_raw, CudaFreeStreamDeleter<uint64_t>{stream});

  K25Val2* d_v2_a_raw = nullptr;
  K25Val2* d_v2_b_raw = nullptr;
  err = guga_cuda_malloc(&d_v2_a_raw, (size_t)nnz_in * sizeof(K25Val2), stream);
  if (err != cudaSuccess) return err;
  err = guga_cuda_malloc(&d_v2_b_raw, (size_t)nnz_in * sizeof(K25Val2), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<K25Val2> d_v2_a(d_v2_a_raw, CudaFreeStreamDeleter<K25Val2>{stream});
  cuda_unique_ptr_stream<K25Val2> d_v2_b(d_v2_b_raw, CudaFreeStreamDeleter<K25Val2>{stream});

  k25_build_sort2_inputs_kernel<<<blocks, threads, 0, stream>>>(
      keys_rs.Current(), vals1.Current(), d_jk_a.get(), d_v2_a.get(), nnz_in);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  cub::DoubleBuffer<uint64_t> keys_jk(d_jk_a.get(), d_jk_b.get());
  cub::DoubleBuffer<K25Val2> vals2(d_v2_a.get(), d_v2_b.get());

  temp_bytes = 0;
  err = cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_jk, vals2, nnz_in, 0, 64, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset();
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset((void*)d_temp_raw);
  err = cub::DeviceRadixSort::SortPairs(d_temp.get(), temp_bytes, keys_jk, vals2, nnz_in, 0, 64, stream);
  if (err != cudaSuccess) return err;

  int nnz_out = nnz_in;
  uint64_t* d_jk_unique = keys_jk.Current();
  const K25Val2* d_v2_sorted = vals2.Current();

  cuda_unique_ptr_stream<K25Key> d_key_in(nullptr, CudaFreeStreamDeleter<K25Key>{stream});
  cuda_unique_ptr_stream<double> d_val_in(nullptr, CudaFreeStreamDeleter<double>{stream});
  cuda_unique_ptr_stream<K25Key> d_key_out(nullptr, CudaFreeStreamDeleter<K25Key>{stream});

  if (coalesce) {
    K25Key* d_key_in_raw = nullptr;
    K25Key* d_key_out_raw = nullptr;
    double* d_val_in_raw = nullptr;
    err = guga_cuda_malloc(&d_key_in_raw, (size_t)nnz_in * sizeof(K25Key), stream);
    if (err != cudaSuccess) return err;
    err = guga_cuda_malloc(&d_key_out_raw, (size_t)nnz_in * sizeof(K25Key), stream);
    if (err != cudaSuccess) return err;
    err = guga_cuda_malloc(&d_val_in_raw, (size_t)nnz_in * sizeof(double), stream);
    if (err != cudaSuccess) return err;
    d_key_in.reset(d_key_in_raw);
    d_key_out.reset(d_key_out_raw);
    d_val_in.reset(d_val_in_raw);

    k25_build_reduce_inputs_kernel<<<blocks, threads, 0, stream>>>(
        keys_jk.Current(), vals2.Current(), d_key_in.get(), d_val_in.get(), nnz_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    temp_bytes = 0;
    err = cub::DeviceReduce::ReduceByKey(
        nullptr,
        temp_bytes,
        d_key_in.get(),
        d_key_out.get(),
        d_val_in.get(),
        out_data,
        out_nnz,
        K25Sum(),
        nnz_in,
        stream);
    if (err != cudaSuccess) return err;
    d_temp.reset();
    err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
    if (err != cudaSuccess) return err;
    d_temp.reset((void*)d_temp_raw);
    err = cub::DeviceReduce::ReduceByKey(
        d_temp.get(),
        temp_bytes,
        d_key_in.get(),
        d_key_out.get(),
        d_val_in.get(),
        out_data,
        out_nnz,
        K25Sum(),
        nnz_in,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(&nnz_out, out_nnz, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;
    if (nnz_out < 0) nnz_out = 0;

    blocks = (nnz_out + threads - 1) / threads;
    k25_extract_unique_kernel<<<blocks, threads, 0, stream>>>(d_key_out.get(), d_jk_unique, out_indices, nnz_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    err = cudaMemcpyAsync(out_nnz, &nnz_out, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    k25_extract_sorted_kernel<<<blocks, threads, 0, stream>>>(d_v2_sorted, out_indices, out_data, nnz_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  int32_t* d_row_counts_raw = nullptr;
  err = guga_cuda_malloc(&d_row_counts_raw, (size_t)nnz_out * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int32_t> d_row_counts(d_row_counts_raw, CudaFreeStreamDeleter<int32_t>{stream});

  temp_bytes = 0;
  err = cub::DeviceRunLengthEncode::Encode(
      nullptr, temp_bytes, d_jk_unique, out_row_jk, d_row_counts.get(), out_nrows, nnz_out, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset();
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset((void*)d_temp_raw);
  err = cub::DeviceRunLengthEncode::Encode(
      d_temp.get(), temp_bytes, d_jk_unique, out_row_jk, d_row_counts.get(), out_nrows, nnz_out, stream);
  if (err != cudaSuccess) return err;

  int nrows = 0;
  err = cudaMemcpyAsync(&nrows, out_nrows, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) return err;
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  if (nrows < 0) nrows = 0;

  int64_t* d_counts64_raw = nullptr;
  err = guga_cuda_malloc(&d_counts64_raw, (size_t)nrows * sizeof(int64_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int64_t> d_counts64(d_counts64_raw, CudaFreeStreamDeleter<int64_t>{stream});

  blocks = (nrows + threads - 1) / threads;
  cast_i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(d_row_counts.get(), d_counts64.get(), nrows);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  temp_bytes = 0;
  err = cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts64.get(), out_indptr, nrows, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset();
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  d_temp.reset((void*)d_temp_raw);
  err = cub::DeviceScan::ExclusiveSum(d_temp.get(), temp_bytes, d_counts64.get(), out_indptr, nrows, stream);
  if (err != cudaSuccess) return err;

  int64_t nnz64 = (int64_t)nnz_out;
  err = cudaMemcpyAsync(out_indptr + nrows, &nnz64, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return err;

  return cudaSuccess;
}

extern "C" cudaError_t guga_counts_to_offsets_exclusive_scan_launch_stream(
    const int32_t* counts,
    int ntasks,
    int64_t* offsets,
    int64_t* total_host,
    cudaStream_t stream);

extern "C" cudaError_t guga_counts_to_offsets_exclusive_scan_launch(
    const int32_t* counts,
    int ntasks,
    int64_t* offsets,
    int64_t* total_host) {
  return guga_counts_to_offsets_exclusive_scan_launch_stream(counts, ntasks, offsets, total_host, /*stream=*/0);
}

extern "C" cudaError_t guga_counts_to_offsets_exclusive_scan_launch_stream(
    const int32_t* counts,
    int ntasks,
    int64_t* offsets,
    int64_t* total_host,
    cudaStream_t stream) {
  if (!counts || !offsets || !total_host) return cudaErrorInvalidValue;
  if (ntasks < 0) return cudaErrorInvalidValue;

  if (ntasks == 0) {
    int64_t zero = 0;
    cudaError_t err = cudaMemcpyAsync(offsets, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    *total_host = 0;
    return cudaSuccess;
  }

  cudaError_t err = cudaSuccess;
  size_t temp_bytes = 0;

  int64_t* d_counts64_raw = nullptr;
  err = guga_cuda_malloc(&d_counts64_raw, (size_t)ntasks * sizeof(int64_t), stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<int64_t> d_counts64(d_counts64_raw, CudaFreeStreamDeleter<int64_t>{stream});

  int threads = 256;
  int blocks = (ntasks + threads - 1) / threads;
  cast_i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(counts, d_counts64.get(), ntasks);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts64.get(), offsets, ntasks, stream);
  if (err != cudaSuccess) return err;

  void* d_temp_raw = nullptr;
  err = guga_cuda_malloc(&d_temp_raw, temp_bytes, stream);
  if (err != cudaSuccess) return err;
  cuda_unique_ptr_stream<void> d_temp((void*)d_temp_raw, CudaFreeStreamDeleter<void>{stream});

  err = cub::DeviceScan::ExclusiveSum(d_temp.get(), temp_bytes, d_counts64.get(), offsets, ntasks, stream);
  if (err != cudaSuccess) return err;

  counts_scan_write_total_kernel<<<1, 1, 0, stream>>>(counts, ntasks, offsets);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int64_t total = 0;
  err = cudaMemcpyAsync(&total, offsets + ntasks, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) return err;
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;

  *total_host = total;
  return cudaSuccess;
}

extern "C" cudaError_t guga_kernel25_build_csr_launch(
    const int32_t* j,
    const int32_t* k,
    const int32_t* rs,
    const double* c,
    int nnz_in,
    int coalesce,
    uint64_t* out_row_jk,
    int64_t* out_indptr,
    int32_t* out_indices,
    double* out_data,
    int* out_nrows,
    int* out_nnz) {
  return guga_kernel25_build_csr_launch_stream(
      j, k, rs, c, nnz_in, coalesce, out_row_jk, out_indptr, out_indices, out_data, out_nrows, out_nnz, /*stream=*/0);
}

// -----------------------------------------------------------------------------
// Kernel 2B + 2.5 persistent workspace (allocation-free steady state)
// -----------------------------------------------------------------------------
