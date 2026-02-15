#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "guga_cuda_kernels_api.h"

namespace py = pybind11;

namespace {

constexpr int MAX_NORB = 64;

struct CudaArrayView {
  void* ptr = nullptr;
  bool read_only = false;
  std::string typestr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;  // empty means "not provided"/contiguous
  uint64_t stream = 0;
};

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void throw_on_cublas_error(cublasStatus_t stat, const char* what) {
  if (stat == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string((int)stat));
}

struct CublasHandle {
  cublasHandle_t h = nullptr;
  CublasHandle() { throw_on_cublas_error(cublasCreate(&h), "cublasCreate"); }
  ~CublasHandle() {
    if (h) cublasDestroy(h);
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  CublasHandle(CublasHandle&& other) noexcept : h(other.h) { other.h = nullptr; }
  CublasHandle& operator=(CublasHandle&& other) noexcept {
    if (this != &other) {
      if (h) cublasDestroy(h);
      h = other.h;
      other.h = nullptr;
    }
    return *this;
  }
};

struct CublasLtHandle {
  cublasLtHandle_t h = nullptr;
  CublasLtHandle() { throw_on_cublas_error(cublasLtCreate(&h), "cublasLtCreate"); }
  ~CublasLtHandle() {
    if (h) cublasLtDestroy(h);
  }
  CublasLtHandle(const CublasLtHandle&) = delete;
  CublasLtHandle& operator=(const CublasLtHandle&) = delete;
  CublasLtHandle(CublasLtHandle&& other) noexcept : h(other.h) { other.h = nullptr; }
  CublasLtHandle& operator=(CublasLtHandle&& other) noexcept {
    if (this != &other) {
      if (h) cublasLtDestroy(h);
      h = other.h;
      other.h = nullptr;
    }
    return *this;
  }
};

inline void validate_cuda_device_pointer(const void* ptr, const char* what) {
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    // Clear sticky error if pointer is not a CUDA allocation.
    cudaGetLastError();
    throw std::invalid_argument(std::string(what) + " is not a valid CUDA device pointer");
  }
#if CUDART_VERSION >= 10000
  if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
    throw std::invalid_argument(std::string(what) + " must point to device/managed memory");
  }
#else
  if (attr.memoryType != cudaMemoryTypeDevice && attr.memoryType != cudaMemoryTypeManaged) {
    throw std::invalid_argument(std::string(what) + " must point to device/managed memory");
  }
#endif
}

inline CudaArrayView cuda_array_view_from_object(const py::object& obj, const char* name) {
  if (!py::hasattr(obj, "__cuda_array_interface__")) {
    throw std::invalid_argument(std::string(name) + " must support __cuda_array_interface__");
  }
  py::dict cai = obj.attr("__cuda_array_interface__").cast<py::dict>();

  CudaArrayView out;

  if (!cai.contains("data")) {
    throw std::invalid_argument(std::string(name) + " __cuda_array_interface__ missing 'data'");
  }
  py::tuple data = cai["data"].cast<py::tuple>();
  if (data.size() < 2) {
    throw std::invalid_argument(std::string(name) + " __cuda_array_interface__ 'data' must be (ptr, read_only)");
  }
  uintptr_t ptr_val = data[0].cast<uintptr_t>();
  out.ptr = reinterpret_cast<void*>(ptr_val);
  out.read_only = data[1].cast<bool>();

  if (!cai.contains("typestr")) {
    throw std::invalid_argument(std::string(name) + " __cuda_array_interface__ missing 'typestr'");
  }
  out.typestr = cai["typestr"].cast<std::string>();

  if (!cai.contains("shape")) {
    throw std::invalid_argument(std::string(name) + " __cuda_array_interface__ missing 'shape'");
  }
  py::tuple shape = cai["shape"].cast<py::tuple>();
  out.shape.reserve((size_t)shape.size());
  for (py::handle dim : shape) {
    int64_t d = py::cast<int64_t>(dim);
    if (d < 0) throw std::invalid_argument(std::string(name) + " has negative dimension in shape");
    out.shape.push_back(d);
  }

  if (cai.contains("strides")) {
    py::object strides_obj = cai["strides"];
    if (!strides_obj.is_none()) {
      py::tuple strides = strides_obj.cast<py::tuple>();
      out.strides_bytes.reserve((size_t)strides.size());
      for (py::handle s : strides) {
        int64_t sb = py::cast<int64_t>(s);
        out.strides_bytes.push_back(sb);
      }
    }
  }

  if (cai.contains("stream")) {
    py::object stream_obj = cai["stream"];
    if (!stream_obj.is_none()) {
      out.stream = stream_obj.cast<uint64_t>();
    }
  }

  bool is_empty = false;
  for (int64_t d : out.shape) {
    if (d == 0) {
      is_empty = true;
      break;
    }
  }
  if (out.ptr == nullptr) {
    if (!is_empty) {
      throw std::invalid_argument(std::string(name) + " has null device pointer");
    }
    // Allow null pointers for empty arrays (CuPy uses ptr=0 for size==0). Callers must ensure
    // they never dereference the pointer when the corresponding shape has zero elements.
    return out;
  }
  validate_cuda_device_pointer(out.ptr, name);
  return out;
}

inline void require_typestr(const CudaArrayView& a, const char* name, const char* expected) {
  if (a.typestr == expected) return;
  // Accept native-endian marker '=' as equivalent to little-endian '<' on typical CUDA platforms.
  if (a.typestr.size() == 3 && expected[0] == '<' && a.typestr[0] == '=' && a.typestr[1] == expected[1] &&
      a.typestr[2] == expected[2]) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must have typestr " + expected + " (got " + a.typestr + ")");
}

inline std::string normalize_typestr(std::string t) {
  if (t.size() == 3 && (t[0] == '=' || t[0] == '|')) t[0] = '<';
  return t;
}

inline int epq_pq_type_from_typestr(const CudaArrayView& a, const char* name) {
  const std::string t = normalize_typestr(a.typestr);
  if (t == "<u1") return 1;
  if (t == "<u2") return 2;
  if (t == "<i4") return 4;
  throw std::invalid_argument(std::string(name) + " must have typestr <u1, <u2, or <i4 (got " + a.typestr + ")");
}

inline int64_t epq_pq_itemsize_from_type(int pq_type) {
  if (pq_type == 1) return (int64_t)sizeof(uint8_t);
  if (pq_type == 2) return (int64_t)sizeof(uint16_t);
  if (pq_type == 4) return (int64_t)sizeof(int32_t);
  return -1;
}

inline int epq_indptr_type_from_typestr(const CudaArrayView& a, const char* name) {
  const std::string t = normalize_typestr(a.typestr);
  if (t == "<i4") return 4;
  if (t == "<i8") return 8;
  throw std::invalid_argument(std::string(name) + " must have typestr <i4 or <i8 (got " + a.typestr + ")");
}

inline int64_t epq_indptr_itemsize_from_type(int indptr_type) {
  if (indptr_type == 4) return (int64_t)sizeof(int32_t);
  if (indptr_type == 8) return (int64_t)sizeof(int64_t);
  return -1;
}

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

inline size_t align_up_size(size_t value, size_t alignment) {
  if (alignment <= 1) return value;
  return (value + alignment - 1) / alignment * alignment;
}

struct DeviceDRT {
  int norb = 0;
  int nnodes = 0;
  int32_t* child = nullptr;       // [nnodes,4]
  int16_t* node_twos = nullptr;   // [nnodes]
  int64_t* child_prefix = nullptr;  // [nnodes,5]
  void* table_blob = nullptr;     // contiguous backing store for child/node_twos/child_prefix
  size_t table_blob_bytes = 0;
  mutable uint64_t access_policy_stream = 0;
  mutable int access_policy_device = -1;

  DeviceDRT() = default;
  ~DeviceDRT() { release(); }

  DeviceDRT(const DeviceDRT&) = delete;
  DeviceDRT& operator=(const DeviceDRT&) = delete;

  DeviceDRT(DeviceDRT&& other) noexcept { *this = std::move(other); }
  DeviceDRT& operator=(DeviceDRT&& other) noexcept {
    if (this != &other) {
      release();
      norb = other.norb;
      nnodes = other.nnodes;
      child = other.child;
      node_twos = other.node_twos;
      child_prefix = other.child_prefix;
      table_blob = other.table_blob;
      table_blob_bytes = other.table_blob_bytes;
      access_policy_stream = other.access_policy_stream;
      access_policy_device = other.access_policy_device;
      other.norb = 0;
      other.nnodes = 0;
      other.child = nullptr;
      other.node_twos = nullptr;
      other.child_prefix = nullptr;
      other.table_blob = nullptr;
      other.table_blob_bytes = 0;
      other.access_policy_stream = 0;
      other.access_policy_device = -1;
    }
    return *this;
  }

  void release() noexcept {
    if (table_blob) {
      cudaFree(table_blob);
      table_blob = nullptr;
      table_blob_bytes = 0;
      child = nullptr;
      node_twos = nullptr;
      child_prefix = nullptr;
      access_policy_stream = 0;
      access_policy_device = -1;
      return;
    }
    if (child) cudaFree(child);
    if (node_twos) cudaFree(node_twos);
    if (child_prefix) cudaFree(child_prefix);
    child = nullptr;
    node_twos = nullptr;
    child_prefix = nullptr;
    table_blob_bytes = 0;
    access_policy_stream = 0;
    access_policy_device = -1;
  }
};

inline void maybe_set_drt_access_policy_window(const DeviceDRT& drt, cudaStream_t stream) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
  if (!stream) return;

  const char* env_val = std::getenv("ASUKA_CUGUGA_DRT_L2_WINDOW");
  if (env_val && (std::string(env_val) == "0" || std::string(env_val) == "false" || std::string(env_val) == "FALSE")) {
    return;
  }

  void* base_ptr = drt.table_blob ? drt.table_blob : static_cast<void*>(drt.child);
  size_t total_bytes = drt.table_blob_bytes;
  if (total_bytes == 0 && drt.nnodes > 0) {
    total_bytes =
        (size_t)drt.nnodes * (4 * sizeof(int32_t) + sizeof(int16_t) + 5 * sizeof(int64_t));
  }
  constexpr size_t kMinWindowBytes = 256 * 1024;
  if (!base_ptr || total_bytes < kMinWindowBytes) return;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }

  uint64_t stream_key = reinterpret_cast<uint64_t>(stream);
  if (drt.access_policy_stream == stream_key && drt.access_policy_device == dev) {
    return;
  }

  constexpr int kCacheMaxDevices = 16;
  static int s_max_window[kCacheMaxDevices] = {0};
  static int s_max_persisting_l2[kCacheMaxDevices] = {0};
  static bool s_ready[kCacheMaxDevices] = {false};

  int max_window = 0;
  int persisting_l2_bytes = 0;
  bool can_cache = (dev >= 0 && dev < kCacheMaxDevices);
  if (can_cache && s_ready[dev]) {
    max_window = s_max_window[dev];
    persisting_l2_bytes = s_max_persisting_l2[dev];
  } else {
    if (cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    if (cudaDeviceGetAttribute(&persisting_l2_bytes, cudaDevAttrMaxPersistingL2CacheSize, dev) != cudaSuccess) {
      persisting_l2_bytes = 0;
      (void)cudaGetLastError();
    }
    if (can_cache) {
      s_max_window[dev] = max_window;
      s_max_persisting_l2[dev] = persisting_l2_bytes;
      s_ready[dev] = true;
    }
  }
  if (max_window <= 0) return;

  size_t window_bytes = std::min(total_bytes, static_cast<size_t>(max_window));
  if (window_bytes == 0) return;

  if (persisting_l2_bytes > 0) {
    (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(persisting_l2_bytes));
    (void)cudaGetLastError();
  }

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = base_ptr;
  attr.accessPolicyWindow.num_bytes = window_bytes;
  attr.accessPolicyWindow.hitRatio = 1.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) == cudaSuccess) {
    drt.access_policy_stream = stream_key;
    drt.access_policy_device = dev;
  } else {
    (void)cudaGetLastError();
  }
#else
  (void)drt;
  (void)stream;
#endif
}

struct DeviceStateCache {
  int norb = 0;
  int ncsf = 0;
  int8_t* steps = nullptr;  // [ncsf,norb]
  int32_t* nodes = nullptr; // [ncsf,norb+1]

  DeviceStateCache() = default;
  ~DeviceStateCache() { release(); }

  DeviceStateCache(const DeviceStateCache&) = delete;
  DeviceStateCache& operator=(const DeviceStateCache&) = delete;

  DeviceStateCache(DeviceStateCache&& other) noexcept { *this = std::move(other); }
  DeviceStateCache& operator=(DeviceStateCache&& other) noexcept {
    if (this != &other) {
      release();
      norb = other.norb;
      ncsf = other.ncsf;
      steps = other.steps;
      nodes = other.nodes;
      other.norb = 0;
      other.ncsf = 0;
      other.steps = nullptr;
      other.nodes = nullptr;
    }
    return *this;
  }

  void release() noexcept {
    if (steps) cudaFree(steps);
    if (nodes) cudaFree(nodes);
    steps = nullptr;
    nodes = nullptr;
  }
};

struct TripletFactorsWorkspace {
  int twos_max = 0;
  float* sixj_211 = nullptr;
  float* t_factor = nullptr;
  void* blob = nullptr;
  size_t blob_bytes = 0;

  TripletFactorsWorkspace() = default;
  ~TripletFactorsWorkspace() { release(); }

  TripletFactorsWorkspace(const TripletFactorsWorkspace&) = delete;
  TripletFactorsWorkspace& operator=(const TripletFactorsWorkspace&) = delete;

  TripletFactorsWorkspace(TripletFactorsWorkspace&& other) noexcept { *this = std::move(other); }
  TripletFactorsWorkspace& operator=(TripletFactorsWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      twos_max = other.twos_max;
      sixj_211 = other.sixj_211;
      t_factor = other.t_factor;
      blob = other.blob;
      blob_bytes = other.blob_bytes;
      other.twos_max = 0;
      other.sixj_211 = nullptr;
      other.t_factor = nullptr;
      other.blob = nullptr;
      other.blob_bytes = 0;
    }
    return *this;
  }

  void release() noexcept {
    if (blob) cudaFree(blob);
    blob = nullptr;
    blob_bytes = 0;
    sixj_211 = nullptr;
    t_factor = nullptr;
    twos_max = 0;
  }
};

struct Kernel25Workspace {
  int max_tasks = 0;
  int max_nnz_in = 0;
  void* ws = nullptr;

  Kernel25Workspace(int max_tasks_, int max_nnz_in_) : max_tasks(std::max(0, max_tasks_)), max_nnz_in(std::max(0, max_nnz_in_)) {
    if (max_tasks <= 0) {
      throw std::invalid_argument("max_tasks must be >= 1");
    }
    if (max_nnz_in <= 0) {
      throw std::invalid_argument("max_nnz_in must be >= 1");
    }
    ws = guga_kernel25_workspace_create(max_tasks, max_nnz_in);
    if (!ws) {
      throw std::runtime_error("failed to create Kernel25Workspace");
    }
  }

  ~Kernel25Workspace() { release(); }

  Kernel25Workspace(const Kernel25Workspace&) = delete;
  Kernel25Workspace& operator=(const Kernel25Workspace&) = delete;

  Kernel25Workspace(Kernel25Workspace&& other) noexcept { *this = std::move(other); }
  Kernel25Workspace& operator=(Kernel25Workspace&& other) noexcept {
    if (this != &other) {
      release();
      max_tasks = other.max_tasks;
      max_nnz_in = other.max_nnz_in;
      ws = other.ws;
      other.max_tasks = 0;
      other.max_nnz_in = 0;
      other.ws = nullptr;
    }
    return *this;
  }

  void release() noexcept {
    if (ws) {
      guga_kernel25_workspace_destroy(ws);
    }
    ws = nullptr;
  }

  py::tuple build_from_jrs_allpairs_deterministic_inplace_device(
      const DeviceDRT& drt,
      const DeviceStateCache& state,
      int j_start,
      int j_count,
      py::object row_j,
      py::object row_k,
      py::object indptr,
      py::object indices,
      py::object data,
      py::object overflow,
      int threads,
      bool coalesce,
      uint64_t stream,
      bool sync,
      bool check_overflow,
      int check_overflow_mode,
      bool use_fused_count_write,
      py::object profile) const {
    if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
      throw std::runtime_error("DeviceDRT is not initialized");
    }
    if (state.steps == nullptr || state.nodes == nullptr) {
      throw std::runtime_error("DeviceStateCache is not initialized");
    }
    if (drt.norb != state.norb) {
      throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
    }
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    if (check_overflow_mode < 0 || check_overflow_mode > 2) {
      throw std::invalid_argument("check_overflow_mode must be 0 (none), 1 (deferred), or 2 (per-stage)");
    }
    int overflow_mode = int(check_overflow_mode);
    if (!check_overflow) overflow_mode = 0;
    if (overflow_mode != 0 && !sync) {
      throw std::invalid_argument("overflow checking requires sync=True");
    }
    py::dict profile_dict;
    Kernel25Profile native_profile{};
    Kernel25Profile* native_profile_ptr = nullptr;
    if (!profile.is_none()) {
      if (!py::isinstance<py::dict>(profile)) {
        throw std::invalid_argument("profile must be a dict or None");
      }
      profile_dict = py::reinterpret_borrow<py::dict>(profile);
      native_profile_ptr = &native_profile;
    }
    if (row_j.is_none() || row_k.is_none() || indptr.is_none() || indices.is_none() || data.is_none() || overflow.is_none()) {
      throw std::invalid_argument("output arrays must be device arrays (cannot be None)");
    }

    auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
    auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
    auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
    auto indices_dev = cuda_array_view_from_object(indices, "indices");
    auto data_dev = cuda_array_view_from_object(data, "data");
    auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");

    require_typestr(row_j_dev, "row_j", "<i4");
    require_typestr(row_k_dev, "row_k", "<i4");
    require_typestr(indptr_dev, "indptr", "<i8");
    require_typestr(indices_dev, "indices", "<i4");
    int out_data_type = 0;
    if (data_dev.typestr == "<f8") out_data_type = 8;
    else if (data_dev.typestr == "<f4") out_data_type = 4;
    else throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
    require_typestr(overflow_dev, "overflow", "<i4");

    if (row_j_dev.read_only || row_k_dev.read_only || indptr_dev.read_only || indices_dev.read_only || data_dev.read_only || overflow_dev.read_only) {
      throw std::invalid_argument("output arrays must be writable");
    }
    if (row_j_dev.shape.size() != 1 || row_k_dev.shape.size() != 1 || indptr_dev.shape.size() != 1 || indices_dev.shape.size() != 1 ||
        data_dev.shape.size() != 1) {
      throw std::invalid_argument("output arrays must be 1D device arrays");
    }
    if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
      throw std::invalid_argument("overflow must have shape (1,)");
    }
    if (indices_dev.shape[0] != data_dev.shape[0]) {
      throw std::invalid_argument("indices and data must have the same length");
    }

    int64_t capacity = indices_dev.shape[0];
    if (capacity <= 0 || capacity > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("capacity out of supported range");
    }
    int max_out = (int)capacity;
    if (max_out > max_nnz_in) {
      throw std::invalid_argument("capacity exceeds workspace max_nnz_in (recreate workspace)");
    }
    if (row_j_dev.shape[0] != capacity || row_k_dev.shape[0] != capacity) {
      throw std::invalid_argument("row_j/row_k must have shape (capacity,) matching indices/data length");
    }
    if (indptr_dev.shape[0] != capacity + 1) {
      throw std::invalid_argument("indptr must have shape (capacity+1,)");
    }

    int32_t* d_row_j = reinterpret_cast<int32_t*>(row_j_dev.ptr);
    int32_t* d_row_k = reinterpret_cast<int32_t*>(row_k_dev.ptr);
    int64_t* d_indptr = reinterpret_cast<int64_t*>(indptr_dev.ptr);
    int32_t* d_indices = reinterpret_cast<int32_t*>(indices_dev.ptr);
    void* d_data = reinterpret_cast<void*>(data_dev.ptr);
    int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (data_dev.stream) stream_u = data_dev.stream;
      else if (indices_dev.stream) stream_u = indices_dev.stream;
      else if (indptr_dev.stream) stream_u = indptr_dev.stream;
      else if (row_j_dev.stream) stream_u = row_j_dev.stream;
      else if (row_k_dev.stream) stream_u = row_k_dev.stream;
      else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    int h_nrows = 0;
    int h_nnz = 0;
    int h_nnz_in = 0;
    guga_kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device_ws(
        ws,
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        int(j_start),
        int(j_count),
        d_row_j,
        d_row_k,
        d_indptr,
        d_indices,
        d_data,
        out_data_type,
        max_out,
        d_overflow,
        stream_t,
        int(threads),
        coalesce ? 1 : 0,
        &h_nrows,
        &h_nnz,
        &h_nnz_in,
        sync ? 1 : 0,
        overflow_mode,
        use_fused_count_write ? 1 : 0,
        native_profile_ptr);

    if (native_profile_ptr != nullptr) {
      profile_dict["count_ms"] = py::float_(native_profile.count_ms);
      profile_dict["prefix_sum_ms"] = py::float_(native_profile.prefix_sum_ms);
      profile_dict["write_ms"] = py::float_(native_profile.write_ms);
      profile_dict["pack_ms"] = py::float_(native_profile.pack_ms);
      profile_dict["sort_ms"] = py::float_(native_profile.sort_ms);
      profile_dict["reduce_ms"] = py::float_(native_profile.reduce_ms);
      profile_dict["rle_ms"] = py::float_(native_profile.rle_ms);
      profile_dict["indptr_ms"] = py::float_(native_profile.indptr_ms);
      profile_dict["unpack_ms"] = py::float_(native_profile.unpack_ms);
      profile_dict["sync_overhead_ms"] = py::float_(native_profile.sync_overhead_ms);
      profile_dict["nnz_in"] = py::int_(native_profile.nnz_in);
      profile_dict["nnz_out"] = py::int_(native_profile.nnz_out);
      profile_dict["nrows"] = py::int_(native_profile.nrows);
    }

    return py::make_tuple(py::int_(h_nrows), py::int_(h_nnz), py::int_(h_nnz_in));
  }

  py::tuple build_from_tasks_deterministic_inplace_device(
      const DeviceDRT& drt,
      const DeviceStateCache& state,
      py::object task_csf,
      py::object task_p,
      py::object task_q,
      py::object row_j,
      py::object row_k,
      py::object indptr,
      py::object indices,
      py::object data,
      py::object overflow,
      int threads,
      bool coalesce,
      uint64_t stream,
      bool sync,
      bool check_overflow) const {
    if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
      throw std::runtime_error("DeviceDRT is not initialized");
    }
    if (state.steps == nullptr || state.nodes == nullptr) {
      throw std::runtime_error("DeviceStateCache is not initialized");
    }
    if (drt.norb != state.norb) {
      throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
    }
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    if (check_overflow && !sync) {
      throw std::invalid_argument("check_overflow=True requires sync=True");
    }
    if (task_csf.is_none() || task_p.is_none() || task_q.is_none()) {
      throw std::invalid_argument("task arrays must be device arrays (cannot be None)");
    }
    if (row_j.is_none() || row_k.is_none() || indptr.is_none() || indices.is_none() || data.is_none() || overflow.is_none()) {
      throw std::invalid_argument("output arrays must be device arrays (cannot be None)");
    }

    auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
    auto task_p_dev = cuda_array_view_from_object(task_p, "task_p");
    auto task_q_dev = cuda_array_view_from_object(task_q, "task_q");
    require_typestr(task_csf_dev, "task_csf", "<i4");
    require_typestr(task_p_dev, "task_p", "<i4");
    require_typestr(task_q_dev, "task_q", "<i4");
    if (task_csf_dev.shape.size() != 1 || task_p_dev.shape.size() != 1 || task_q_dev.shape.size() != 1) {
      throw std::invalid_argument("task arrays must be 1D device arrays");
    }
    if (task_csf_dev.shape[0] != task_p_dev.shape[0] || task_csf_dev.shape[0] != task_q_dev.shape[0]) {
      throw std::invalid_argument("task arrays must have the same length");
    }

    int64_t ntasks_ll = task_csf_dev.shape[0];
    if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("ntasks out of supported range (batch the work)");
    }
    int ntasks = (int)ntasks_ll;
    if (ntasks > max_tasks) {
      throw std::invalid_argument("ntasks exceeds workspace max_tasks (recreate workspace)");
    }

    auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
    auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
    auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
    auto indices_dev = cuda_array_view_from_object(indices, "indices");
    auto data_dev = cuda_array_view_from_object(data, "data");
    auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");

    require_typestr(row_j_dev, "row_j", "<i4");
    require_typestr(row_k_dev, "row_k", "<i4");
    require_typestr(indptr_dev, "indptr", "<i8");
    require_typestr(indices_dev, "indices", "<i4");
    int out_data_type = 0;
    if (data_dev.typestr == "<f8") out_data_type = 8;
    else if (data_dev.typestr == "<f4") out_data_type = 4;
    else throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
    require_typestr(overflow_dev, "overflow", "<i4");

    if (row_j_dev.read_only || row_k_dev.read_only || indptr_dev.read_only || indices_dev.read_only || data_dev.read_only || overflow_dev.read_only) {
      throw std::invalid_argument("output arrays must be writable");
    }
    if (row_j_dev.shape.size() != 1 || row_k_dev.shape.size() != 1 || indptr_dev.shape.size() != 1 || indices_dev.shape.size() != 1 ||
        data_dev.shape.size() != 1) {
      throw std::invalid_argument("output arrays must be 1D device arrays");
    }
    if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
      throw std::invalid_argument("overflow must have shape (1,)");
    }
    if (indices_dev.shape[0] != data_dev.shape[0]) {
      throw std::invalid_argument("indices and data must have the same length");
    }

    int64_t capacity = indices_dev.shape[0];
    if (capacity <= 0 || capacity > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("capacity out of supported range");
    }
    int max_out = (int)capacity;
    if (max_out > max_nnz_in) {
      throw std::invalid_argument("capacity exceeds workspace max_nnz_in (recreate workspace)");
    }
    if (row_j_dev.shape[0] != capacity || row_k_dev.shape[0] != capacity) {
      throw std::invalid_argument("row_j/row_k must have shape (capacity,) matching indices/data length");
    }
    if (indptr_dev.shape[0] != capacity + 1) {
      throw std::invalid_argument("indptr must have shape (capacity+1,)");
    }

    int32_t* d_task_csf = reinterpret_cast<int32_t*>(task_csf_dev.ptr);
    int32_t* d_task_p = reinterpret_cast<int32_t*>(task_p_dev.ptr);
    int32_t* d_task_q = reinterpret_cast<int32_t*>(task_q_dev.ptr);
    int32_t* d_row_j = reinterpret_cast<int32_t*>(row_j_dev.ptr);
    int32_t* d_row_k = reinterpret_cast<int32_t*>(row_k_dev.ptr);
    int64_t* d_indptr = reinterpret_cast<int64_t*>(indptr_dev.ptr);
    int32_t* d_indices = reinterpret_cast<int32_t*>(indices_dev.ptr);
    void* d_data = reinterpret_cast<void*>(data_dev.ptr);
    int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (data_dev.stream) stream_u = data_dev.stream;
      else if (indices_dev.stream) stream_u = indices_dev.stream;
      else if (indptr_dev.stream) stream_u = indptr_dev.stream;
      else if (row_j_dev.stream) stream_u = row_j_dev.stream;
      else if (row_k_dev.stream) stream_u = row_k_dev.stream;
      else if (task_q_dev.stream) stream_u = task_q_dev.stream;
      else if (task_p_dev.stream) stream_u = task_p_dev.stream;
      else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
      else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    int h_nrows = 0;
    int h_nnz = 0;
    int h_nnz_in = 0;
    guga_kernel25_build_csr_from_tasks_deterministic_inplace_device_ws(
        ws,
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        d_task_csf,
        d_task_p,
        d_task_q,
        ntasks,
        d_row_j,
        d_row_k,
        d_indptr,
        d_indices,
        d_data,
        out_data_type,
        max_out,
        d_overflow,
        stream_t,
        int(threads),
        coalesce ? 1 : 0,
        &h_nrows,
        &h_nnz,
        &h_nnz_in,
        sync ? 1 : 0,
        check_overflow ? 1 : 0);

    return py::make_tuple(py::int_(h_nrows), py::int_(h_nnz), py::int_(h_nnz_in));
  }
};

struct QmcWorkspace {
  int max_n = 0;
  int max_m = 0;
  void* ws = nullptr;

  QmcWorkspace(int max_n_, int max_m_) : max_n(std::max(0, max_n_)), max_m(std::max(0, max_m_)) {
    if (max_n <= 0) {
      throw std::invalid_argument("max_n must be >= 1");
    }
    if (max_m <= 0) {
      throw std::invalid_argument("max_m must be >= 1");
    }
    ws = guga_qmc_workspace_create(max_n, max_m);
    if (!ws) {
      throw std::runtime_error("failed to create QmcWorkspace");
    }
  }

  ~QmcWorkspace() { release(); }

  QmcWorkspace(const QmcWorkspace&) = delete;
  QmcWorkspace& operator=(const QmcWorkspace&) = delete;

  QmcWorkspace(QmcWorkspace&& other) noexcept { *this = std::move(other); }
  QmcWorkspace& operator=(QmcWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      max_n = other.max_n;
      max_m = other.max_m;
      ws = other.ws;
      other.max_n = 0;
      other.max_m = 0;
      other.ws = nullptr;
    }
    return *this;
  }

  void release() noexcept {
    if (ws) {
      guga_qmc_workspace_destroy(ws);
    }
    ws = nullptr;
  }

  void coalesce_coo_i32_f64_inplace_device(
      py::object idx_in,
      py::object val_in,
      py::object idx_out,
      py::object val_out,
      py::object out_nnz,
      int n,
      int threads,
      uint64_t stream,
      bool sync) const {
    if (!ws) throw std::runtime_error("QmcWorkspace has been released");
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    n = int(n);
    if (n < 0 || n > max_n) {
      throw std::invalid_argument("n out of range for workspace max_n");
    }
    if (idx_in.is_none() || val_in.is_none() || idx_out.is_none() || val_out.is_none() || out_nnz.is_none()) {
      throw std::invalid_argument("idx_in/val_in/idx_out/val_out/out_nnz must be device arrays (cannot be None)");
    }

    CudaArrayView idx_in_dev = cuda_array_view_from_object(idx_in, "idx_in");
    CudaArrayView val_in_dev = cuda_array_view_from_object(val_in, "val_in");
    CudaArrayView idx_out_dev = cuda_array_view_from_object(idx_out, "idx_out");
    CudaArrayView val_out_dev = cuda_array_view_from_object(val_out, "val_out");
    CudaArrayView nnz_dev = cuda_array_view_from_object(out_nnz, "out_nnz");

    require_typestr(idx_in_dev, "idx_in", "<i4");
    require_typestr(val_in_dev, "val_in", "<f8");
    require_typestr(idx_out_dev, "idx_out", "<i4");
    require_typestr(val_out_dev, "val_out", "<f8");
    require_typestr(nnz_dev, "out_nnz", "<i4");

    if (idx_out_dev.read_only || val_out_dev.read_only || nnz_dev.read_only) {
      throw std::invalid_argument("idx_out/val_out/out_nnz must be writable device arrays");
    }

    if (idx_in_dev.shape.size() != 1 || val_in_dev.shape.size() != 1) {
      throw std::invalid_argument("idx_in and val_in must be 1D device arrays");
    }
    if (idx_in_dev.shape[0] != val_in_dev.shape[0]) {
      throw std::invalid_argument("idx_in and val_in must have the same length");
    }
    if ((int64_t)n > idx_in_dev.shape[0]) {
      throw std::invalid_argument("n exceeds len(idx_in)");
    }

    if (idx_out_dev.shape.size() != 1 || val_out_dev.shape.size() != 1) {
      throw std::invalid_argument("idx_out and val_out must be 1D device arrays");
    }
    if (idx_out_dev.shape[0] < (int64_t)n || val_out_dev.shape[0] < (int64_t)n) {
      throw std::invalid_argument("idx_out/val_out must have length >= n");
    }

    if (!idx_in_dev.strides_bytes.empty()) {
      if (idx_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_in must be 1D with valid strides");
      if (idx_in_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_in must be contiguous");
    }
    if (!val_in_dev.strides_bytes.empty()) {
      if (val_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_in must be 1D with valid strides");
      if (val_in_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_in must be contiguous");
    }
    if (!idx_out_dev.strides_bytes.empty()) {
      if (idx_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_out must be 1D with valid strides");
      if (idx_out_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_out must be contiguous");
    }
    if (!val_out_dev.strides_bytes.empty()) {
      if (val_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_out must be 1D with valid strides");
      if (val_out_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_out must be contiguous");
    }

    if (nnz_dev.shape.size() == 1) {
      if (nnz_dev.shape[0] != 1) throw std::invalid_argument("out_nnz must have shape (1,) or ()");
      if (!nnz_dev.strides_bytes.empty()) {
        if (nnz_dev.strides_bytes.size() != 1) throw std::invalid_argument("out_nnz must be 1D with valid strides");
        if (nnz_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("out_nnz must be contiguous");
      }
    } else if (nnz_dev.shape.size() == 0) {
      // scalar: ok
    } else {
      throw std::invalid_argument("out_nnz must have shape (1,) or ()");
    }

    cudaStream_t stream_t = (cudaStream_t)stream;
    throw_on_cuda_error(
        guga_qmc_coalesce_coo_i32_f64_ws_launch_stream(
            ws,
            (const int32_t*)idx_in_dev.ptr,
            (const double*)val_in_dev.ptr,
            n,
            (int32_t*)idx_out_dev.ptr,
            (double*)val_out_dev.ptr,
            (int*)nnz_dev.ptr,
            stream_t,
            threads),
        "guga_qmc_coalesce_coo_i32_f64_ws_launch_stream");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_ws_coalesce)");
    }
  }

  void phi_pivot_resample_i32_f64_inplace_device(
      py::object idx_in,
      py::object val_in,
      py::object idx_out,
      py::object val_out,
      py::object out_nnz,
      int n_in,
      int m,
      int pivot,
      uint64_t seed,
      int threads,
      uint64_t stream,
      bool sync) const {
    if (!ws) throw std::runtime_error("QmcWorkspace has been released");
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    n_in = int(n_in);
    m = int(m);
    pivot = int(pivot);
    if (n_in < 0 || n_in > max_n) {
      throw std::invalid_argument("n_in out of range for workspace max_n");
    }
    if (m < 0 || m > max_m) {
      throw std::invalid_argument("m out of range for workspace max_m");
    }
    if (pivot < 0) {
      throw std::invalid_argument("pivot must be >= 0");
    }
    if (idx_in.is_none() || val_in.is_none() || idx_out.is_none() || val_out.is_none() || out_nnz.is_none()) {
      throw std::invalid_argument("idx_in/val_in/idx_out/val_out/out_nnz must be device arrays (cannot be None)");
    }

    CudaArrayView idx_in_dev = cuda_array_view_from_object(idx_in, "idx_in");
    CudaArrayView val_in_dev = cuda_array_view_from_object(val_in, "val_in");
    CudaArrayView idx_out_dev = cuda_array_view_from_object(idx_out, "idx_out");
    CudaArrayView val_out_dev = cuda_array_view_from_object(val_out, "val_out");
    CudaArrayView nnz_dev = cuda_array_view_from_object(out_nnz, "out_nnz");

    require_typestr(idx_in_dev, "idx_in", "<i4");
    require_typestr(val_in_dev, "val_in", "<f8");
    require_typestr(idx_out_dev, "idx_out", "<i4");
    require_typestr(val_out_dev, "val_out", "<f8");
    require_typestr(nnz_dev, "out_nnz", "<i4");

    if (idx_out_dev.read_only || val_out_dev.read_only || nnz_dev.read_only) {
      throw std::invalid_argument("idx_out/val_out/out_nnz must be writable device arrays");
    }

    if (idx_in_dev.shape.size() != 1 || val_in_dev.shape.size() != 1) {
      throw std::invalid_argument("idx_in and val_in must be 1D device arrays");
    }
    if (idx_in_dev.shape[0] != val_in_dev.shape[0]) {
      throw std::invalid_argument("idx_in and val_in must have the same length");
    }
    if ((int64_t)n_in > idx_in_dev.shape[0]) {
      throw std::invalid_argument("n_in exceeds len(idx_in)");
    }

    if (idx_out_dev.shape.size() != 1 || val_out_dev.shape.size() != 1) {
      throw std::invalid_argument("idx_out and val_out must be 1D device arrays");
    }
    if (idx_out_dev.shape[0] < (int64_t)m || val_out_dev.shape[0] < (int64_t)m) {
      throw std::invalid_argument("idx_out/val_out must have length >= m");
    }

    if (!idx_in_dev.strides_bytes.empty()) {
      if (idx_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_in must be 1D with valid strides");
      if (idx_in_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_in must be contiguous");
    }
    if (!val_in_dev.strides_bytes.empty()) {
      if (val_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_in must be 1D with valid strides");
      if (val_in_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_in must be contiguous");
    }
    if (!idx_out_dev.strides_bytes.empty()) {
      if (idx_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_out must be 1D with valid strides");
      if (idx_out_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_out must be contiguous");
    }
    if (!val_out_dev.strides_bytes.empty()) {
      if (val_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_out must be 1D with valid strides");
      if (val_out_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_out must be contiguous");
    }

    if (nnz_dev.shape.size() == 1) {
      if (nnz_dev.shape[0] != 1) throw std::invalid_argument("out_nnz must have shape (1,) or ()");
      if (!nnz_dev.strides_bytes.empty()) {
        if (nnz_dev.strides_bytes.size() != 1) throw std::invalid_argument("out_nnz must be 1D with valid strides");
        if (nnz_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("out_nnz must be contiguous");
      }
    } else if (nnz_dev.shape.size() == 0) {
      // scalar: ok
    } else {
      throw std::invalid_argument("out_nnz must have shape (1,) or ()");
    }

    cudaStream_t stream_t = (cudaStream_t)stream;
    throw_on_cuda_error(
        guga_qmc_phi_pivot_resample_i32_f64_ws_launch_stream(
            ws,
            (const int32_t*)idx_in_dev.ptr,
            (const double*)val_in_dev.ptr,
            n_in,
            (int32_t*)idx_out_dev.ptr,
            (double*)val_out_dev.ptr,
            (int*)nnz_dev.ptr,
            m,
            pivot,
            seed,
            stream_t,
            threads),
        "guga_qmc_phi_pivot_resample_i32_f64_ws_launch_stream");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_ws_phi)");
    }
  }
};

struct Kernel3BuildGGemmExWorkspace {
  int nops = 0;
  int max_nrows = 0;

  CublasHandle cublas;
  CublasLtHandle cublaslt;

  enum class GemmBackendKind { GEMMEX = 0, CUBLASLT = 1 };
  GemmBackendKind gemm_backend_kind = GemmBackendKind::GEMMEX;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_64F;
  cudaDataType_t gemm_data_type = CUDA_R_64F;
  bool gemm_tf32_prefer = false;
  cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;

  void* dCublasWorkspace = nullptr;
  size_t cublas_workspace_bytes = 0;

  void* dCsrDense = nullptr;  // [max_nrows, nops] row-major
  size_t csr_dense_itemsize = sizeof(double);

  struct LtAlgoCacheEntry {
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
  };
  std::unordered_map<int, LtAlgoCacheEntry> lt_cache_w_eri;  // keyed by nrows

  void ensure_csr_dense_buffer(size_t itemsize) {
    if (itemsize != sizeof(double) && itemsize != sizeof(float)) {
      throw std::invalid_argument("unsupported csr_dense itemsize");
    }
    if (dCsrDense && csr_dense_itemsize == itemsize) return;
    if (dCsrDense) {
      throw_on_cuda_error(cudaFree(dCsrDense), "cudaFree(kernel3 csr_dense)");
      dCsrDense = nullptr;
    }
    throw_on_cuda_error(
        cudaMalloc(&dCsrDense, (size_t)max_nrows * (size_t)nops * itemsize), "cudaMalloc(kernel3 csr_dense)");
    csr_dense_itemsize = itemsize;
    lt_cache_w_eri.clear();
  }

  Kernel3BuildGGemmExWorkspace(int nops_, int max_nrows_) : nops(std::max(0, nops_)), max_nrows(std::max(0, max_nrows_)) {
    if (nops <= 0) {
      throw std::invalid_argument("nops must be >= 1");
    }
    if (max_nrows <= 0) {
      throw std::invalid_argument("max_nrows must be >= 1");
    }
    ensure_csr_dense_buffer(sizeof(double));
  }

  ~Kernel3BuildGGemmExWorkspace() { release(); }

  Kernel3BuildGGemmExWorkspace(const Kernel3BuildGGemmExWorkspace&) = delete;
  Kernel3BuildGGemmExWorkspace& operator=(const Kernel3BuildGGemmExWorkspace&) = delete;

  Kernel3BuildGGemmExWorkspace(Kernel3BuildGGemmExWorkspace&& other) noexcept { *this = std::move(other); }
  Kernel3BuildGGemmExWorkspace& operator=(Kernel3BuildGGemmExWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      nops = other.nops;
      max_nrows = other.max_nrows;
      cublas = std::move(other.cublas);
      cublaslt = std::move(other.cublaslt);
      gemm_backend_kind = other.gemm_backend_kind;
      gemm_compute_type = other.gemm_compute_type;
      gemm_data_type = other.gemm_data_type;
      gemm_tf32_prefer = other.gemm_tf32_prefer;
      gemm_algo = other.gemm_algo;
      dCublasWorkspace = other.dCublasWorkspace;
      cublas_workspace_bytes = other.cublas_workspace_bytes;
      dCsrDense = other.dCsrDense;
      csr_dense_itemsize = other.csr_dense_itemsize;
      lt_cache_w_eri = std::move(other.lt_cache_w_eri);
      other.nops = 0;
      other.max_nrows = 0;
      other.gemm_backend_kind = GemmBackendKind::GEMMEX;
      other.gemm_compute_type = CUBLAS_COMPUTE_64F;
      other.gemm_data_type = CUDA_R_64F;
      other.gemm_tf32_prefer = false;
      other.gemm_algo = CUBLAS_GEMM_DEFAULT;
      other.dCublasWorkspace = nullptr;
      other.cublas_workspace_bytes = 0;
      other.dCsrDense = nullptr;
      other.csr_dense_itemsize = sizeof(double);
      other.lt_cache_w_eri.clear();
    }
    return *this;
  }

  void release() noexcept {
    if (dCublasWorkspace) {
      if (cublas.h) cublasSetWorkspace(cublas.h, nullptr, 0);
      cudaFree(dCublasWorkspace);
    }
    if (dCsrDense) cudaFree(dCsrDense);
    dCublasWorkspace = nullptr;
    cublas_workspace_bytes = 0;
    dCsrDense = nullptr;
    csr_dense_itemsize = sizeof(double);
    gemm_tf32_prefer = false;
    lt_cache_w_eri.clear();
  }

  std::string gemm_backend() const {
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "cublaslt_fp64";
      if (gemm_data_type == CUDA_R_32F && gemm_tf32_prefer) return "cublaslt_tf32";
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) return "cublaslt_fp32";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) return "cublaslt_tf32";
#endif
      return "cublaslt_unknown";
    }
    if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp64";
    // 10.20.4: Mixed-precision FP32 data with FP64 accumulation
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp32_acc64";
    if (gemm_data_type == CUDA_R_32F && gemm_tf32_prefer) return "gemmex_tf32";
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) return "gemmex_fp32";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) return "gemmex_tf32";
#endif
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) {
      return "gemmex_emulated_fixedpoint";
    }
#endif
    return "gemmex_unknown";
  }

  void set_gemm_backend(const std::string& backend) {
    if (backend == "cublaslt_fp64") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(double));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "cublaslt_fp32") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
      gemm_compute_type = CUBLAS_COMPUTE_32F;
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(float));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "cublaslt_tf32") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
      gemm_compute_type = CUBLAS_COMPUTE_32F;
#endif
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = true;
      ensure_csr_dense_buffer(sizeof(float));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "gemmex_fp64") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(double));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "gemmex_fp32") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_32F;
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(float));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "gemmex_tf32") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
      gemm_compute_type = CUBLAS_COMPUTE_32F;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(tf32)");
#else
      throw std::runtime_error(
          "gemmex_tf32 backend is unavailable: this CUDA toolkit exposes neither "
          "CUBLAS_COMPUTE_32F_FAST_TF32 nor CUBLAS_TF32_TENSOR_OP_MATH");
#endif
#endif
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = true;
      ensure_csr_dense_buffer(sizeof(float));
      lt_cache_w_eri.clear();
      return;
    }
    if (backend == "gemmex_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(double));
      lt_cache_w_eri.clear();
      return;
#else
      throw std::runtime_error("gemmex_emulated_fixedpoint requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    // 10.20.4 Recommendation #2: Mixed-precision GEMM with FP32 data and FP64 accumulation.
    // This provides better numerical accuracy than pure FP32 by accumulating in FP64,
    // while still benefiting from FP32 memory bandwidth for input matrices.
    if (backend == "gemmex_fp32_acc64") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F;  // FP64 accumulation
      gemm_data_type = CUDA_R_32F;             // FP32 input data
      gemm_tf32_prefer = false;
      ensure_csr_dense_buffer(sizeof(float));
      lt_cache_w_eri.clear();
      return;
    }
    throw std::invalid_argument("unknown gemm backend: " + backend);
  }

  int gemm_algo_int() const { return (int)gemm_algo; }

  void set_gemm_algo_int(int algo) { gemm_algo = (cublasGemmAlgo_t)algo; }

  void set_cublas_math_mode(const std::string& mode) {
    if (mode == "default") {
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) {
        gemm_tf32_prefer = false;
      }
      return;
    }
    if (mode == "tf32_tensor_op") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode");
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) {
        gemm_tf32_prefer = true;
      }
      return;
#else
      throw std::runtime_error("tf32_tensor_op math mode is unavailable in this CUDA toolkit");
#endif
    }
    if (mode == "fp64_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH), "cublasSetMathMode");
      return;
#else
      throw std::runtime_error("fp64_emulated_fixedpoint math mode requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    throw std::invalid_argument("unknown cuBLAS math mode: " + mode);
  }

  size_t get_cublas_workspace_bytes() const { return cublas_workspace_bytes; }

  void set_cublas_workspace_bytes(size_t bytes) {
    if (dCublasWorkspace) {
      throw_on_cuda_error(cudaFree(dCublasWorkspace), "cudaFree(cublas workspace)");
      dCublasWorkspace = nullptr;
      cublas_workspace_bytes = 0;
    }
    lt_cache_w_eri.clear();
    if (bytes > 0) {
      void* p = nullptr;
      throw_on_cuda_error(cudaMalloc(&p, bytes), "cudaMalloc(cublas workspace)");
      dCublasWorkspace = p;
      cublas_workspace_bytes = bytes;
    }
    throw_on_cublas_error(
        cublasSetWorkspace(cublas.h, dCublasWorkspace, cublas_workspace_bytes), "cublasSetWorkspace");
  }

  py::dict cublas_emulation_info() const {
    int ver = 0;
    throw_on_cublas_error(cublasGetVersion(cublas.h, &ver), "cublasGetVersion");
    py::dict out;
    out["cublas_version"] = ver;

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    throw_on_cublas_error(cublasGetMathMode(cublas.h, &math_mode), "cublasGetMathMode");
    const char* math_name = "unknown";
    switch (math_mode) {
      case CUBLAS_DEFAULT_MATH:
        math_name = "default";
        break;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      case CUBLAS_TF32_TENSOR_OP_MATH:
        math_name = "tf32_tensor_op";
        break;
#endif
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      case CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH:
        math_name = "fp64_emulated_fixedpoint";
        break;
#endif
      default:
        break;
    }
    out["math_mode"] = (int)math_mode;
    out["math_mode_name"] = std::string(math_name);

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    throw_on_cublas_error(cublasGetEmulationStrategy(cublas.h, &strategy), "cublasGetEmulationStrategy");

    cudaEmulationSpecialValuesSupport special_mask = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    throw_on_cublas_error(
        cublasGetEmulationSpecialValuesSupport(cublas.h, &special_mask), "cublasGetEmulationSpecialValuesSupport");

    cudaEmulationMantissaControl mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaControl(cublas.h, &mantissa_control),
        "cublasGetFixedPointEmulationMantissaControl");

    int max_bits = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMaxMantissaBitCount(cublas.h, &max_bits),
        "cublasGetFixedPointEmulationMaxMantissaBitCount");

    int bit_offset = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaBitOffset(cublas.h, &bit_offset),
        "cublasGetFixedPointEmulationMantissaBitOffset");

    const char* strategy_name = "unknown";
    switch (strategy) {
      case CUBLAS_EMULATION_STRATEGY_DEFAULT:
        strategy_name = "default";
        break;
      case CUBLAS_EMULATION_STRATEGY_PERFORMANT:
        strategy_name = "performant";
        break;
      case CUBLAS_EMULATION_STRATEGY_EAGER:
        strategy_name = "eager";
        break;
      default:
        break;
    }

    const char* mantissa_name = "unknown";
    switch (mantissa_control) {
      case CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC:
        mantissa_name = "dynamic";
        break;
      case CUDA_EMULATION_MANTISSA_CONTROL_FIXED:
        mantissa_name = "fixed";
        break;
      default:
        break;
    }

    out["emulation_supported"] = true;
    out["emulation_strategy"] = (int)strategy;
    out["emulation_strategy_name"] = std::string(strategy_name);
    out["emulation_special_values_support_mask"] = (int)special_mask;
    out["fixed_point_mantissa_control"] = (int)mantissa_control;
    out["fixed_point_mantissa_control_name"] = std::string(mantissa_name);
    out["fixed_point_max_mantissa_bits"] = max_bits;
    out["fixed_point_mantissa_bit_offset"] = bit_offset;
#else
    out["emulation_supported"] = false;
#endif

    out["gemm_backend"] = gemm_backend();
    out["gemm_compute_type"] = (int)gemm_compute_type;
    out["gemm_data_type"] = (int)gemm_data_type;
    out["gemm_tf32_prefer"] = gemm_tf32_prefer;
    out["gemm_algo"] = (int)gemm_algo;
    out["cublas_workspace_bytes"] = py::int_(cublas_workspace_bytes);
    return out;
  }

  void set_cublas_emulation_strategy(const std::string& strategy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    if (strategy == "default") {
      s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    } else if (strategy == "performant") {
      s = CUBLAS_EMULATION_STRATEGY_PERFORMANT;
    } else if (strategy == "eager") {
      s = CUBLAS_EMULATION_STRATEGY_EAGER;
    } else {
      throw std::invalid_argument("unknown emulation strategy: " + strategy);
    }
    throw_on_cublas_error(cublasSetEmulationStrategy(cublas.h, s), "cublasSetEmulationStrategy");
#else
    (void)strategy;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_emulation_special_values_support(int mask) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    auto m = (cudaEmulationSpecialValuesSupport)mask;
    throw_on_cublas_error(cublasSetEmulationSpecialValuesSupport(cublas.h, m), "cublasSetEmulationSpecialValuesSupport");
#else
    (void)mask;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_control(const std::string& control) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cudaEmulationMantissaControl c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    if (control == "dynamic") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    } else if (control == "fixed") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    } else {
      throw std::invalid_argument("unknown mantissa control: " + control);
    }
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaControl(cublas.h, c), "cublasSetFixedPointEmulationMantissaControl");
#else
    (void)control;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_max_mantissa_bits(int max_bits) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMaxMantissaBitCount(cublas.h, max_bits),
        "cublasSetFixedPointEmulationMaxMantissaBitCount");
#else
    (void)max_bits;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_bit_offset(int bit_offset) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaBitOffset(cublas.h, bit_offset),
        "cublasSetFixedPointEmulationMantissaBitOffset");
#else
    (void)bit_offset;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void gemm_w_eri_mat_inplace_device(
      py::object eri_mat,
      py::object w_dense,
      py::object g_out,
      double half,
      uint64_t stream,
      bool sync) {
    if (eri_mat.is_none() || w_dense.is_none() || g_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
    bool use_f32 = false;
    if (eri_dev.typestr == "<f8") {
      use_f32 = false;
    } else if (eri_dev.typestr == "<f4") {
      use_f32 = true;
    } else {
      throw std::invalid_argument("eri_mat must have typestr <f8 (float64) or <f4 (float32)");
    }
    if (use_f32 && gemm_data_type != CUDA_R_32F) {
      throw std::invalid_argument("eri_mat is float32 but workspace backend is not fp32/tf32");
    }
    if (!use_f32 && gemm_data_type != CUDA_R_64F) {
      throw std::invalid_argument("eri_mat is float64 but workspace backend is fp32/tf32");
    }
    const int64_t fp_itemsize = use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
      throw std::invalid_argument("eri_mat must have shape (nops,nops)");
    }
    if (eri_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("eri_mat has wrong nops for this workspace");
    }
    if (!eri_dev.strides_bytes.empty()) {
      if (eri_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("eri_mat strides must have length 2");
      }
      int64_t s0 = eri_dev.strides_bytes[0];
      int64_t s1 = eri_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
      }
    }

    auto w_dev = cuda_array_view_from_object(w_dense, "w_dense");
    if (use_f32) require_typestr(w_dev, "w_dense", "<f4");
    else require_typestr(w_dev, "w_dense", "<f8");
    if (w_dev.read_only) {
      throw std::invalid_argument("w_dense must be writable (we do not support read-only device buffers)");
    }
    if (w_dev.shape.size() != 2 || w_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("w_dense must have shape (nrows,nops)");
    }
    if (!w_dev.strides_bytes.empty()) {
      if (w_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_dense strides must have length 2");
      }
      int64_t s0 = w_dev.strides_bytes[0];
      int64_t s1 = w_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("w_dense must be C-contiguous with no padding");
      }
    }

    const int64_t nrows_ll = w_dev.shape[0];
    if (nrows_ll < 0) {
      throw std::invalid_argument("w_dense invalid nrows");
    }
    if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("nrows too large (batch the work)");
    }
    const int nrows = (int)nrows_ll;

    auto g_dev = cuda_array_view_from_object(g_out, "g_out");
    if (use_f32) require_typestr(g_dev, "g_out", "<f4");
    else require_typestr(g_dev, "g_out", "<f8");
    if (g_dev.read_only) {
      throw std::invalid_argument("g_out must be writable");
    }
    if (g_dev.shape.size() != 2 || g_dev.shape[0] != nrows_ll || g_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("g_out must have shape (nrows,nops)");
    }
    if (!g_dev.strides_bytes.empty()) {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }
    if (g_dev.ptr == w_dev.ptr) {
      throw std::invalid_argument("g_out must not alias w_dense");
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (g_dev.stream) stream_u = g_dev.stream;
      else if (w_dev.stream) stream_u = w_dev.stream;
      else if (eri_dev.stream) stream_u = eri_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const void* d_eri = eri_dev.ptr;
    const void* d_w = w_dev.ptr;
    void* d_g = g_dev.ptr;

    if (nrows == 0 || nops == 0) return;

    // Layout trick: treat row-major W[nrows,nops] as column-major W^T[nops,nrows], and
    // treat row-major g_out[nrows,nops] as column-major G^T[nops,nrows].
    //
    // Compute: G^T = half * ERI_mat[nops,nops] @ W^T[nops,nrows]
    // so that g_out = (G^T)^T = half * W @ ERI_mat.
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      } else if (gemm_compute_type != CUBLAS_COMPUTE_64F) {
        throw std::runtime_error("float64 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_64F");
      }

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b_desc = nullptr;
      cublasLtMatrixLayout_t c_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;
      try {
        throw_on_cublas_error(
            cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, gemm_data_type), "cublasLtMatmulDescCreate");
        cublasOperation_t opn = CUBLAS_OP_N;
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transa)");
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transb)");

        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&a_desc, gemm_data_type, nops, nops, nops), "cublasLtMatrixLayoutCreate(A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(C)");
        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C)");

        throw_on_cublas_error(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        throw_on_cublas_error(
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");

        LtAlgoCacheEntry entry;
        bool have_algo = false;
        auto it = lt_cache_w_eri.find(nrows);
        if (it != lt_cache_w_eri.end()) {
          entry = it->second;
          have_algo = true;
        }
        if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
          cublasLtMatmulHeuristicResult_t heur{};
          int nret = 0;
          throw_on_cublas_error(
              cublasLtMatmulAlgoGetHeuristic(
                  cublaslt.h, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heur, &nret),
              "cublasLtMatmulAlgoGetHeuristic(w_eri)");
          if (nret <= 0) {
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (w_eri)");
          }
          entry.algo = heur.algo;
          entry.workspace_bytes = (size_t)heur.workspaceSize;
          lt_cache_w_eri[nrows] = entry;
        }

        const float alpha_f32 = (float)half;
        const float beta_f32 = 0.0f;
        const double alpha_f64 = half;
        const double beta_f64 = 0.0;
        const void* alpha = use_f32 ? static_cast<const void*>(&alpha_f32) : static_cast<const void*>(&alpha_f64);
        const void* beta = use_f32 ? static_cast<const void*>(&beta_f32) : static_cast<const void*>(&beta_f64);
        void* ws_ptr = dCublasWorkspace;
        size_t ws_bytes = cublas_workspace_bytes;
        if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
        if (ws_bytes == 0) ws_ptr = nullptr;
        throw_on_cublas_error(
            cublasLtMatmul(
                cublaslt.h,
                op_desc,
                alpha,
                d_eri,
                a_desc,
                d_w,
                b_desc,
                beta,
                d_g,
                c_desc,
                d_g,
                c_desc,
                &entry.algo,
                ws_ptr,
                ws_bytes,
                stream_t),
            "cublasLtMatmul(gemm_w_eri)");
      } catch (...) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        throw;
      }
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
      if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    } else {
      throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");
      const float alpha_f32 = (float)half;
      const float beta_f32 = 0.0f;
      const double alpha_f64 = half;
      const double beta_f64 = 0.0;
      const void* alpha = use_f32 ? static_cast<const void*>(&alpha_f32) : static_cast<const void*>(&alpha_f64);
      const void* beta = use_f32 ? static_cast<const void*>(&beta_f32) : static_cast<const void*>(&beta_f64);
      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              nops,
              nrows,
              nops,
              alpha,
              d_eri,
              gemm_data_type,
              nops,
              d_w,
              gemm_data_type,
              nops,
              beta,
              d_g,
              gemm_data_type,
              nops,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(gemm_w_eri)");
    }

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(gemm_w_eri)");
    }
  }

  void build_g_from_csr_eri_mat_inplace_device(
      py::object indptr,
      py::object indices,
      py::object data,
      py::object eri_mat,
      py::object g_out,
      int threads,
      double half,
      uint64_t stream,
      bool sync) {
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    if (indptr.is_none() || indices.is_none() || data.is_none() || eri_mat.is_none() || g_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
    require_typestr(indptr_dev, "indptr", "<i8");
    if (indptr_dev.shape.size() != 1 || indptr_dev.shape[0] < 1) {
      throw std::invalid_argument("indptr must be a 1D device array with shape (nrows+1,)");
    }
    if (!indptr_dev.strides_bytes.empty()) {
      if (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
        throw std::invalid_argument("indptr must be contiguous");
      }
    }

    int64_t nrows_ll = indptr_dev.shape[0] - 1;
    if (nrows_ll < 0) {
      throw std::invalid_argument("invalid indptr length");
    }
    if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("nrows too large (batch the work)");
    }
    int nrows = (int)nrows_ll;
    if (nrows > max_nrows) {
      throw std::invalid_argument("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows");
    }

    auto indices_dev = cuda_array_view_from_object(indices, "indices");
    require_typestr(indices_dev, "indices", "<i4");
    if (indices_dev.shape.size() != 1) {
      throw std::invalid_argument("indices must be a 1D device array");
    }
    if (!indices_dev.strides_bytes.empty()) {
      if (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("indices must be contiguous");
      }
    }

    auto data_dev = cuda_array_view_from_object(data, "data");
    bool use_f32 = false;
    if (data_dev.typestr == "<f8") {
      use_f32 = false;
    } else if (data_dev.typestr == "<f4") {
      use_f32 = true;
    } else {
      throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
    }
    if (use_f32 && gemm_data_type != CUDA_R_32F) {
      throw std::invalid_argument("data is float32 but workspace backend is not fp32/tf32");
    }
    if (!use_f32 && gemm_data_type != CUDA_R_64F) {
      throw std::invalid_argument("data is float64 but workspace backend is fp32/tf32");
    }
    const int64_t fp_itemsize = use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if (data_dev.shape.size() != 1) {
      throw std::invalid_argument("data must be a 1D device array");
    }
    if (!data_dev.strides_bytes.empty()) {
      if (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("data must be contiguous");
      }
    }
    if (data_dev.shape[0] != indices_dev.shape[0]) {
      throw std::invalid_argument("indices and data must have the same length");
    }

    auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
    if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
    else require_typestr(eri_dev, "eri_mat", "<f8");
    if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
      throw std::invalid_argument("eri_mat must have shape (nops,nops)");
    }
    if (eri_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("eri_mat has wrong nops for this workspace");
    }
    if (!eri_dev.strides_bytes.empty()) {
      if (eri_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("eri_mat strides must have length 2");
      }
      int64_t s0 = eri_dev.strides_bytes[0];
      int64_t s1 = eri_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
      }
    }

    auto g_dev = cuda_array_view_from_object(g_out, "g_out");
    if (use_f32) require_typestr(g_dev, "g_out", "<f4");
    else require_typestr(g_dev, "g_out", "<f8");
    if (g_dev.read_only) {
      throw std::invalid_argument("g_out must be writable");
    }
    if (g_dev.shape.size() != 2 || g_dev.shape[0] != nrows_ll || g_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("g_out must have shape (nrows,nops)");
    }
    if (!g_dev.strides_bytes.empty()) {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (g_dev.stream) stream_u = g_dev.stream;
      else if (eri_dev.stream) stream_u = eri_dev.stream;
      else if (data_dev.stream) stream_u = data_dev.stream;
      else if (indices_dev.stream) stream_u = indices_dev.stream;
      else if (indptr_dev.stream) stream_u = indptr_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const int64_t* d_indptr = reinterpret_cast<const int64_t*>(indptr_dev.ptr);
    const int32_t* d_indices = reinterpret_cast<const int32_t*>(indices_dev.ptr);
    const void* d_data = data_dev.ptr;
    const void* d_eri = eri_dev.ptr;
    void* d_g = g_dev.ptr;

    if (nrows == 0 || nops == 0) return;

    // Build dense A[row,rs] from CSR, then do GEMM: G = half * A * ERI_mat.
    ensure_csr_dense_buffer((size_t)fp_itemsize);
    throw_on_cuda_error(
        cudaMemsetAsync(dCsrDense, 0, (size_t)nrows * (size_t)nops * (size_t)fp_itemsize, stream_t),
        "cudaMemsetAsync(csr_dense)");

    if (use_f32) {
      throw_on_cuda_error(
          guga_csr_to_dense_f32_launch_stream(
              d_indptr,
              d_indices,
              reinterpret_cast<const float*>(d_data),
              nrows,
              nops,
              reinterpret_cast<float*>(dCsrDense),
              stream_t,
              threads),
          "guga_csr_to_dense_f32_launch_stream");
    } else {
      throw_on_cuda_error(
          guga_csr_to_dense_f64_launch_stream(
              d_indptr,
              d_indices,
              reinterpret_cast<const double*>(d_data),
              nrows,
              nops,
              reinterpret_cast<double*>(dCsrDense),
              stream_t,
              threads),
          "guga_csr_to_dense_f64_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(csr_to_dense)");

    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      } else if (gemm_compute_type != CUBLAS_COMPUTE_64F) {
        throw std::runtime_error("float64 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_64F");
      }

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b_desc = nullptr;
      cublasLtMatrixLayout_t c_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;
      try {
        throw_on_cublas_error(
            cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, gemm_data_type), "cublasLtMatmulDescCreate");
        cublasOperation_t opn = CUBLAS_OP_N;
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transa)");
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transb)");

        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&a_desc, gemm_data_type, nops, nops, nops), "cublasLtMatrixLayoutCreate(A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(C)");
        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C)");

        throw_on_cublas_error(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        throw_on_cublas_error(
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");

        LtAlgoCacheEntry entry;
        bool have_algo = false;
        auto it = lt_cache_w_eri.find(nrows);
        if (it != lt_cache_w_eri.end()) {
          entry = it->second;
          have_algo = true;
        }
        if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
          cublasLtMatmulHeuristicResult_t heur{};
          int nret = 0;
          throw_on_cublas_error(
              cublasLtMatmulAlgoGetHeuristic(
                  cublaslt.h, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heur, &nret),
              "cublasLtMatmulAlgoGetHeuristic(build_g)");
          if (nret <= 0) {
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (build_g)");
          }
          entry.algo = heur.algo;
          entry.workspace_bytes = (size_t)heur.workspaceSize;
          lt_cache_w_eri[nrows] = entry;
        }

        const float alpha_f32 = (float)half;
        const float beta_f32 = 0.0f;
        const double alpha_f64 = half;
        const double beta_f64 = 0.0;
        const void* alpha = use_f32 ? static_cast<const void*>(&alpha_f32) : static_cast<const void*>(&alpha_f64);
        const void* beta = use_f32 ? static_cast<const void*>(&beta_f32) : static_cast<const void*>(&beta_f64);
        void* ws_ptr = dCublasWorkspace;
        size_t ws_bytes = cublas_workspace_bytes;
        if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
        if (ws_bytes == 0) ws_ptr = nullptr;
        throw_on_cublas_error(
            cublasLtMatmul(
                cublaslt.h,
                op_desc,
                alpha,
                d_eri,
                a_desc,
                dCsrDense,
                b_desc,
                beta,
                d_g,
                c_desc,
                d_g,
                c_desc,
                &entry.algo,
                ws_ptr,
                ws_bytes,
                stream_t),
            "cublasLtMatmul(build_g)");
      } catch (...) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        throw;
      }
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
      if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    } else {
      throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");
      const float alpha_f32 = (float)half;
      const float beta_f32 = 0.0f;
      const double alpha_f64 = half;
      const double beta_f64 = 0.0;
      const void* alpha = use_f32 ? static_cast<const void*>(&alpha_f32) : static_cast<const void*>(&alpha_f64);
      const void* beta = use_f32 ? static_cast<const void*>(&beta_f32) : static_cast<const void*>(&beta_f64);
      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              nops,
              nrows,
              nops,
              alpha,
              d_eri,
              gemm_data_type,
              nops,
              dCsrDense,
              gemm_data_type,
              nops,
              beta,
              d_g,
              gemm_data_type,
              nops,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(build_g)");
    }

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel3_build_g_gemmex)");
    }
  }
};

struct Kernel3BuildGDFGemmExWorkspace {
  int nops = 0;
  int naux = 0;
  int max_nrows = 0;

  CublasHandle cublas;
  CublasLtHandle cublaslt;

  enum class GemmBackendKind { GEMMEX = 0, CUBLASLT = 1 };
  GemmBackendKind gemm_backend_kind = GemmBackendKind::GEMMEX;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_64F;
  cudaDataType_t gemm_data_type = CUDA_R_64F;
  bool gemm_tf32_prefer = false;
  cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;

  void* dCublasWorkspace = nullptr;
  size_t cublas_workspace_bytes = 0;

  // W^T buffer (column-major [naux, max_nrows] with leading dimension naux). The allocation is sized as
  // [max_nrows, naux] in row-major to match C-contiguous expectations from Python-side views.
  void* dW = nullptr;
  size_t w_itemsize = sizeof(double);

  struct LtAlgoCacheEntry {
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
  };
  std::unordered_map<int, LtAlgoCacheEntry> lt_cache_step1;  // keyed by nrows
  std::unordered_map<int, LtAlgoCacheEntry> lt_cache_step2;  // keyed by nrows

  void ensure_w_buffer(size_t itemsize) {
    if (itemsize != sizeof(double) && itemsize != sizeof(float)) {
      throw std::invalid_argument("unsupported dW itemsize");
    }
    if (dW && w_itemsize == itemsize) return;
    if (dW) {
      throw_on_cuda_error(cudaFree(dW), "cudaFree(kernel3 df W)");
      dW = nullptr;
    }
    throw_on_cuda_error(cudaMalloc(&dW, (size_t)max_nrows * (size_t)naux * itemsize), "cudaMalloc(kernel3 df W)");
    w_itemsize = itemsize;
    lt_cache_step1.clear();
    lt_cache_step2.clear();
  }

  Kernel3BuildGDFGemmExWorkspace(int nops_, int naux_, int max_nrows_)
      : nops(std::max(0, nops_)), naux(std::max(0, naux_)), max_nrows(std::max(0, max_nrows_)) {
    if (nops <= 0) {
      throw std::invalid_argument("nops must be >= 1");
    }
    if (naux <= 0) {
      throw std::invalid_argument("naux must be >= 1");
    }
    if (max_nrows <= 0) {
      throw std::invalid_argument("max_nrows must be >= 1");
    }
    ensure_w_buffer(sizeof(double));
  }

  ~Kernel3BuildGDFGemmExWorkspace() { release(); }

  Kernel3BuildGDFGemmExWorkspace(const Kernel3BuildGDFGemmExWorkspace&) = delete;
  Kernel3BuildGDFGemmExWorkspace& operator=(const Kernel3BuildGDFGemmExWorkspace&) = delete;

  Kernel3BuildGDFGemmExWorkspace(Kernel3BuildGDFGemmExWorkspace&& other) noexcept { *this = std::move(other); }
  Kernel3BuildGDFGemmExWorkspace& operator=(Kernel3BuildGDFGemmExWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      nops = other.nops;
      naux = other.naux;
      max_nrows = other.max_nrows;
      cublas = std::move(other.cublas);
      cublaslt = std::move(other.cublaslt);
      gemm_backend_kind = other.gemm_backend_kind;
      gemm_compute_type = other.gemm_compute_type;
      gemm_data_type = other.gemm_data_type;
      gemm_tf32_prefer = other.gemm_tf32_prefer;
      gemm_algo = other.gemm_algo;
      dCublasWorkspace = other.dCublasWorkspace;
      cublas_workspace_bytes = other.cublas_workspace_bytes;
      dW = other.dW;
      w_itemsize = other.w_itemsize;
      lt_cache_step1 = std::move(other.lt_cache_step1);
      lt_cache_step2 = std::move(other.lt_cache_step2);
      other.nops = 0;
      other.naux = 0;
      other.max_nrows = 0;
      other.gemm_backend_kind = GemmBackendKind::GEMMEX;
      other.gemm_compute_type = CUBLAS_COMPUTE_64F;
      other.gemm_data_type = CUDA_R_64F;
      other.gemm_tf32_prefer = false;
      other.gemm_algo = CUBLAS_GEMM_DEFAULT;
      other.dCublasWorkspace = nullptr;
      other.cublas_workspace_bytes = 0;
      other.dW = nullptr;
      other.w_itemsize = sizeof(double);
      other.lt_cache_step1.clear();
      other.lt_cache_step2.clear();
    }
    return *this;
  }

  void release() noexcept {
    if (dCublasWorkspace) {
      if (cublas.h) cublasSetWorkspace(cublas.h, nullptr, 0);
      cudaFree(dCublasWorkspace);
    }
    if (dW) cudaFree(dW);
    dCublasWorkspace = nullptr;
    cublas_workspace_bytes = 0;
    dW = nullptr;
    w_itemsize = sizeof(double);
    gemm_tf32_prefer = false;
    lt_cache_step1.clear();
    lt_cache_step2.clear();
  }

  std::string gemm_backend() const {
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "cublaslt_fp64";
      if (gemm_data_type == CUDA_R_32F && gemm_tf32_prefer) return "cublaslt_tf32";
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) return "cublaslt_fp32";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) return "cublaslt_tf32";
#endif
      return "cublaslt_unknown";
    }
    if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp64";
    // 10.20.4: Mixed-precision FP32 data with FP64 accumulation
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp32_acc64";
    if (gemm_data_type == CUDA_R_32F && gemm_tf32_prefer) return "gemmex_tf32";
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) return "gemmex_fp32";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) return "gemmex_tf32";
#endif
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    if (gemm_data_type == CUDA_R_64F && gemm_compute_type == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) {
      return "gemmex_emulated_fixedpoint";
    }
#endif
    return "gemmex_unknown";
  }

  void set_gemm_backend(const std::string& backend) {
    if (backend == "cublaslt_fp64") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(double));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "cublaslt_fp32") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
      gemm_compute_type = CUBLAS_COMPUTE_32F;
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(float));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "cublaslt_tf32") {
      gemm_backend_kind = GemmBackendKind::CUBLASLT;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
      gemm_compute_type = CUBLAS_COMPUTE_32F;
#endif
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = true;
      ensure_w_buffer(sizeof(float));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "gemmex_fp64") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(double));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "gemmex_fp32") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_32F;
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(float));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "gemmex_tf32") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
      gemm_compute_type = CUBLAS_COMPUTE_32F;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(tf32)");
#else
      throw std::runtime_error(
          "gemmex_tf32 backend is unavailable: this CUDA toolkit exposes neither "
          "CUBLAS_COMPUTE_32F_FAST_TF32 nor CUBLAS_TF32_TENSOR_OP_MATH");
#endif
#endif
      gemm_data_type = CUDA_R_32F;
      gemm_tf32_prefer = true;
      ensure_w_buffer(sizeof(float));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    if (backend == "gemmex_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
      gemm_data_type = CUDA_R_64F;
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(double));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
#else
      throw std::runtime_error("gemmex_emulated_fixedpoint requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    // 10.20.4 Recommendation #2: Mixed-precision GEMM with FP32 data and FP64 accumulation.
    if (backend == "gemmex_fp32_acc64") {
      gemm_backend_kind = GemmBackendKind::GEMMEX;
      gemm_compute_type = CUBLAS_COMPUTE_64F;  // FP64 accumulation
      gemm_data_type = CUDA_R_32F;             // FP32 input data
      gemm_tf32_prefer = false;
      ensure_w_buffer(sizeof(float));
      lt_cache_step1.clear();
      lt_cache_step2.clear();
      return;
    }
    throw std::invalid_argument("unknown gemm backend: " + backend);
  }

  int gemm_algo_int() const { return (int)gemm_algo; }

  void set_gemm_algo_int(int algo) { gemm_algo = (cublasGemmAlgo_t)algo; }

  void set_cublas_math_mode(const std::string& mode) {
    if (mode == "default") {
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) {
        gemm_tf32_prefer = false;
      }
      return;
    }
    if (mode == "tf32_tensor_op") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode");
      if (gemm_data_type == CUDA_R_32F && gemm_compute_type == CUBLAS_COMPUTE_32F) {
        gemm_tf32_prefer = true;
      }
      return;
#else
      throw std::runtime_error("tf32_tensor_op math mode is unavailable in this CUDA toolkit");
#endif
    }
    if (mode == "fp64_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH), "cublasSetMathMode");
      return;
#else
      throw std::runtime_error("fp64_emulated_fixedpoint math mode requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    throw std::invalid_argument("unknown cuBLAS math mode: " + mode);
  }

  size_t get_cublas_workspace_bytes() const { return cublas_workspace_bytes; }

  void set_cublas_workspace_bytes(size_t bytes) {
    if (dCublasWorkspace) {
      throw_on_cuda_error(cudaFree(dCublasWorkspace), "cudaFree(cublas workspace)");
      dCublasWorkspace = nullptr;
      cublas_workspace_bytes = 0;
    }
    lt_cache_step1.clear();
    lt_cache_step2.clear();
    if (bytes > 0) {
      void* p = nullptr;
      throw_on_cuda_error(cudaMalloc(&p, bytes), "cudaMalloc(cublas workspace)");
      dCublasWorkspace = p;
      cublas_workspace_bytes = bytes;
    }
    throw_on_cublas_error(
        cublasSetWorkspace(cublas.h, dCublasWorkspace, cublas_workspace_bytes), "cublasSetWorkspace");
  }

  py::dict cublas_emulation_info() const {
    int ver = 0;
    throw_on_cublas_error(cublasGetVersion(cublas.h, &ver), "cublasGetVersion");
    py::dict out;
    out["cublas_version"] = ver;

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    throw_on_cublas_error(cublasGetMathMode(cublas.h, &math_mode), "cublasGetMathMode");
    const char* math_name = "unknown";
    switch (math_mode) {
      case CUBLAS_DEFAULT_MATH:
        math_name = "default";
        break;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
      case CUBLAS_TF32_TENSOR_OP_MATH:
        math_name = "tf32_tensor_op";
        break;
#endif
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      case CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH:
        math_name = "fp64_emulated_fixedpoint";
        break;
#endif
      default:
        break;
    }
    out["math_mode"] = (int)math_mode;
    out["math_mode_name"] = std::string(math_name);

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    throw_on_cublas_error(cublasGetEmulationStrategy(cublas.h, &strategy), "cublasGetEmulationStrategy");

    cudaEmulationSpecialValuesSupport special_mask = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    throw_on_cublas_error(
        cublasGetEmulationSpecialValuesSupport(cublas.h, &special_mask), "cublasGetEmulationSpecialValuesSupport");

    cudaEmulationMantissaControl mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaControl(cublas.h, &mantissa_control),
        "cublasGetFixedPointEmulationMantissaControl");

    int max_bits = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMaxMantissaBitCount(cublas.h, &max_bits),
        "cublasGetFixedPointEmulationMaxMantissaBitCount");

    int bit_offset = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaBitOffset(cublas.h, &bit_offset),
        "cublasGetFixedPointEmulationMantissaBitOffset");

    const char* strategy_name = "unknown";
    switch (strategy) {
      case CUBLAS_EMULATION_STRATEGY_DEFAULT:
        strategy_name = "default";
        break;
      case CUBLAS_EMULATION_STRATEGY_PERFORMANT:
        strategy_name = "performant";
        break;
      case CUBLAS_EMULATION_STRATEGY_EAGER:
        strategy_name = "eager";
        break;
      default:
        break;
    }

    const char* mantissa_name = "unknown";
    switch (mantissa_control) {
      case CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC:
        mantissa_name = "dynamic";
        break;
      case CUDA_EMULATION_MANTISSA_CONTROL_FIXED:
        mantissa_name = "fixed";
        break;
      default:
        break;
    }

    out["emulation_supported"] = true;
    out["emulation_strategy"] = (int)strategy;
    out["emulation_strategy_name"] = std::string(strategy_name);
    out["emulation_special_values_support_mask"] = (int)special_mask;
    out["fixed_point_mantissa_control"] = (int)mantissa_control;
    out["fixed_point_mantissa_control_name"] = std::string(mantissa_name);
    out["fixed_point_max_mantissa_bits"] = max_bits;
    out["fixed_point_mantissa_bit_offset"] = bit_offset;
#else
    out["emulation_supported"] = false;
#endif

    out["gemm_backend"] = gemm_backend();
    out["gemm_algo"] = (int)gemm_algo;
    out["cublas_workspace_bytes"] = py::int_(cublas_workspace_bytes);
    return out;
  }

  void set_cublas_emulation_strategy(const std::string& strategy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    if (strategy == "default") {
      s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    } else if (strategy == "performant") {
      s = CUBLAS_EMULATION_STRATEGY_PERFORMANT;
    } else if (strategy == "eager") {
      s = CUBLAS_EMULATION_STRATEGY_EAGER;
    } else {
      throw std::invalid_argument("unknown emulation strategy: " + strategy);
    }
    throw_on_cublas_error(cublasSetEmulationStrategy(cublas.h, s), "cublasSetEmulationStrategy");
#else
    (void)strategy;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_emulation_special_values_support(int mask) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    auto m = (cudaEmulationSpecialValuesSupport)mask;
    throw_on_cublas_error(cublasSetEmulationSpecialValuesSupport(cublas.h, m), "cublasSetEmulationSpecialValuesSupport");
#else
    (void)mask;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_control(const std::string& control) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cudaEmulationMantissaControl c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    if (control == "dynamic") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    } else if (control == "fixed") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    } else {
      throw std::invalid_argument("unknown mantissa control: " + control);
    }
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaControl(cublas.h, c), "cublasSetFixedPointEmulationMantissaControl");
#else
    (void)control;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_max_mantissa_bits(int max_bits) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMaxMantissaBitCount(cublas.h, max_bits),
        "cublasSetFixedPointEmulationMaxMantissaBitCount");
#else
    (void)max_bits;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_bit_offset(int bit_offset) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaBitOffset(cublas.h, bit_offset),
        "cublasSetFixedPointEmulationMantissaBitOffset");
#else
    (void)bit_offset;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void gemm_w_l_full_inplace_device(
      py::object l_full,
      py::object w_dense,
      py::object g_out,
      double half,
      uint64_t stream,
      bool sync) {
    if (l_full.is_none() || w_dense.is_none() || g_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto l_full_dev = cuda_array_view_from_object(l_full, "l_full");
    std::string l_full_typestr = normalize_typestr(l_full_dev.typestr);
    bool use_f32 = false;
    if (l_full_typestr == "<f8") {
      use_f32 = false;
    } else if (l_full_typestr == "<f4") {
      use_f32 = true;
    } else {
      throw std::invalid_argument("l_full must have typestr <f8 (float64) or <f4 (float32)");
    }
    if (use_f32 && gemm_data_type != CUDA_R_32F) {
      throw std::runtime_error("float32 DF GEMM requires gemm_data_type=CUDA_R_32F (set gemm_backend to *_fp32/*_tf32)");
    }
    if (!use_f32 && gemm_data_type != CUDA_R_64F) {
      throw std::runtime_error("float64 DF GEMM requires gemm_data_type=CUDA_R_64F (set gemm_backend to *_fp64)");
    }
    const int64_t fp_itemsize = use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if ((int64_t)w_itemsize != fp_itemsize) {
      throw std::runtime_error("DF workspace dW dtype does not match input dtype; call set_gemm_backend first");
    }
    if (l_full_dev.shape.size() != 2) {
      throw std::invalid_argument("l_full must be a 2D device array");
    }
    if (l_full_dev.shape[0] != (int64_t)nops || l_full_dev.shape[1] != (int64_t)naux) {
      throw std::invalid_argument("l_full must have shape (nops,naux) matching this workspace");
    }
    if (!l_full_dev.strides_bytes.empty()) {
      if (l_full_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("l_full strides must have length 2");
      }
      int64_t s0 = l_full_dev.strides_bytes[0];
      int64_t s1 = l_full_dev.strides_bytes[1];
      if (s0 != (int64_t)naux * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("l_full must be C-contiguous with no padding");
      }
    }

    auto w_dev = cuda_array_view_from_object(w_dense, "w_dense");
    if (use_f32) require_typestr(w_dev, "w_dense", "<f4");
    else require_typestr(w_dev, "w_dense", "<f8");
    if (w_dev.shape.size() != 2 || w_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("w_dense must have shape (nrows,nops)");
    }
    if (!w_dev.strides_bytes.empty()) {
      if (w_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_dense strides must have length 2");
      }
      int64_t s0 = w_dev.strides_bytes[0];
      int64_t s1 = w_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("w_dense must be C-contiguous with no padding");
      }
    }

    int64_t nrows_ll = w_dev.shape[0];
    if (nrows_ll < 0) {
      throw std::invalid_argument("w_dense invalid nrows");
    }
    if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("nrows too large (batch the work)");
    }
    const int nrows = (int)nrows_ll;
    if (nrows > max_nrows) {
      throw std::invalid_argument("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows");
    }

    auto g_dev = cuda_array_view_from_object(g_out, "g_out");
    if (use_f32) require_typestr(g_dev, "g_out", "<f4");
    else require_typestr(g_dev, "g_out", "<f8");
    if (g_dev.read_only) {
      throw std::invalid_argument("g_out must be writable");
    }
    if (g_dev.shape.size() != 2 || g_dev.shape[0] != nrows_ll || g_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("g_out must have shape (nrows,nops)");
    }
    if (!g_dev.strides_bytes.empty()) {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (g_dev.stream) stream_u = g_dev.stream;
      else if (w_dev.stream) stream_u = w_dev.stream;
      else if (l_full_dev.stream) stream_u = l_full_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const void* d_lfull = l_full_dev.ptr;
    const void* d_w = w_dev.ptr;
    void* d_g = g_dev.ptr;

    if (nrows == 0 || nops == 0 || naux == 0) return;

    // Layout trick:
    // - Treat row-major W[nrows,nops] as column-major W^T[nops,nrows].
    // - Treat row-major g_out[nrows,nops] as column-major G^T[nops,nrows].
    //
    // Compute in two steps:
    //   1) T^T = L_full^T[naux,nops] @ W^T[nops,nrows]   (stored in dW as column-major [naux,nrows])
    //   2) G^T = half * L_full[nops,naux] @ T^T[naux,nrows]
    //
    // so that g_out = (G^T)^T = half * (W @ L_full) @ L_full^T.
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      } else if (gemm_compute_type != CUBLAS_COMPUTE_64F) {
        throw std::runtime_error("float64 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_64F");
      }

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b1_desc = nullptr;
      cublasLtMatrixLayout_t c1_desc = nullptr;
      cublasLtMatrixLayout_t b2_desc = nullptr;
      cublasLtMatrixLayout_t c2_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;
      try {
        throw_on_cublas_error(
            cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, gemm_data_type), "cublasLtMatmulDescCreate");

        // Treat row-major L_full[nops,naux] as column-major L_full^T[naux,nops].
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&a_desc, gemm_data_type, naux, nops, naux), "cublasLtMatrixLayoutCreate(A)");
        // Treat row-major W[nrows,nops] as column-major W^T[nops,nrows].
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b1_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(B1)");
        // dW is column-major [naux,nrows] with ld=naux.
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c1_desc, gemm_data_type, naux, nrows, naux), "cublasLtMatrixLayoutCreate(C1)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b2_desc, gemm_data_type, naux, nrows, naux), "cublasLtMatrixLayoutCreate(B2)");
        // Treat row-major g_out[nrows,nops] as column-major G^T[nops,nrows].
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c2_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(C2)");

        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b1_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B1)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c1_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C1)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b2_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B2)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c2_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C2)");

        throw_on_cublas_error(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        throw_on_cublas_error(
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");

        // Step 1: dW = L_full^T @ W^T
        {
          cublasOperation_t opn = CUBLAS_OP_N;
          throw_on_cublas_error(
              cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn)),
              "cublasLtMatmulDescSetAttribute(step1 transa)");
          throw_on_cublas_error(
              cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
              "cublasLtMatmulDescSetAttribute(step1 transb)");

          LtAlgoCacheEntry entry;
          bool have_algo = false;
          auto it = lt_cache_step1.find(nrows);
          if (it != lt_cache_step1.end()) {
            entry = it->second;
            have_algo = true;
          }
          if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
            cublasLtMatmulHeuristicResult_t heur{};
            int nret = 0;
            throw_on_cublas_error(
                cublasLtMatmulAlgoGetHeuristic(
                    cublaslt.h, op_desc, a_desc, b1_desc, c1_desc, c1_desc, pref, 1, &heur, &nret),
                "cublasLtMatmulAlgoGetHeuristic(df step1)");
            if (nret <= 0) {
              throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (df step1)");
            }
            entry.algo = heur.algo;
            entry.workspace_bytes = (size_t)heur.workspaceSize;
            lt_cache_step1[nrows] = entry;
          }

          const float alpha1_f32 = 1.0f;
          const float beta0_f32 = 0.0f;
          const double alpha1_f64 = 1.0;
          const double beta0_f64 = 0.0;
          const void* alpha1 = use_f32 ? static_cast<const void*>(&alpha1_f32) : static_cast<const void*>(&alpha1_f64);
          const void* beta0 = use_f32 ? static_cast<const void*>(&beta0_f32) : static_cast<const void*>(&beta0_f64);
          void* ws_ptr = dCublasWorkspace;
          size_t ws_bytes = cublas_workspace_bytes;
          if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
          if (ws_bytes == 0) ws_ptr = nullptr;
          throw_on_cublas_error(
              cublasLtMatmul(
                  cublaslt.h,
                  op_desc,
                  alpha1,
                  d_lfull,
                  a_desc,
                  d_w,
                  b1_desc,
                  beta0,
                  dW,
                  c1_desc,
                  dW,
                  c1_desc,
                  &entry.algo,
                  ws_ptr,
                  ws_bytes,
                  stream_t),
              "cublasLtMatmul(df step1)");
        }

        // Step 2: d_g = half * L_full @ dW  (opA=T on L_full^T)
        {
          cublasOperation_t opt = CUBLAS_OP_T;
          cublasOperation_t opn = CUBLAS_OP_N;
          throw_on_cublas_error(
              cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opt, sizeof(opt)),
              "cublasLtMatmulDescSetAttribute(step2 transa)");
          throw_on_cublas_error(
              cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
              "cublasLtMatmulDescSetAttribute(step2 transb)");

          LtAlgoCacheEntry entry;
          bool have_algo = false;
          auto it = lt_cache_step2.find(nrows);
          if (it != lt_cache_step2.end()) {
            entry = it->second;
            have_algo = true;
          }
          if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
            cublasLtMatmulHeuristicResult_t heur{};
            int nret = 0;
            throw_on_cublas_error(
                cublasLtMatmulAlgoGetHeuristic(
                    cublaslt.h, op_desc, a_desc, b2_desc, c2_desc, c2_desc, pref, 1, &heur, &nret),
                "cublasLtMatmulAlgoGetHeuristic(df step2)");
            if (nret <= 0) {
              throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (df step2)");
            }
            entry.algo = heur.algo;
            entry.workspace_bytes = (size_t)heur.workspaceSize;
            lt_cache_step2[nrows] = entry;
          }

          const float alpha2_f32 = (float)half;
          const float beta0_f32 = 0.0f;
          const double alpha2_f64 = half;
          const double beta0_f64 = 0.0;
          const void* alpha2 = use_f32 ? static_cast<const void*>(&alpha2_f32) : static_cast<const void*>(&alpha2_f64);
          const void* beta0 = use_f32 ? static_cast<const void*>(&beta0_f32) : static_cast<const void*>(&beta0_f64);
          void* ws_ptr = dCublasWorkspace;
          size_t ws_bytes = cublas_workspace_bytes;
          if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
          if (ws_bytes == 0) ws_ptr = nullptr;
          throw_on_cublas_error(
              cublasLtMatmul(
                  cublaslt.h,
                  op_desc,
                  alpha2,
                  d_lfull,
                  a_desc,
                  dW,
                  b2_desc,
                  beta0,
                  d_g,
                  c2_desc,
                  d_g,
                  c2_desc,
                  &entry.algo,
                  ws_ptr,
                  ws_bytes,
                  stream_t),
              "cublasLtMatmul(df step2)");
        }
      } catch (...) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c2_desc) cublasLtMatrixLayoutDestroy(c2_desc);
        if (b2_desc) cublasLtMatrixLayoutDestroy(b2_desc);
        if (c1_desc) cublasLtMatrixLayoutDestroy(c1_desc);
        if (b1_desc) cublasLtMatrixLayoutDestroy(b1_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        throw;
      }
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c2_desc) cublasLtMatrixLayoutDestroy(c2_desc);
      if (b2_desc) cublasLtMatrixLayoutDestroy(b2_desc);
      if (c1_desc) cublasLtMatrixLayoutDestroy(c1_desc);
      if (b1_desc) cublasLtMatrixLayoutDestroy(b1_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    } else {
      throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 gemm requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 gemm requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      }

      const float alpha1_f32 = 1.0f;
      const float beta0_f32 = 0.0f;
      const float alpha2_f32 = (float)half;
      const double alpha1_f64 = 1.0;
      const double beta0_f64 = 0.0;
      const double alpha2_f64 = half;
      const void* alpha1 = use_f32 ? static_cast<const void*>(&alpha1_f32) : static_cast<const void*>(&alpha1_f64);
      const void* beta0 = use_f32 ? static_cast<const void*>(&beta0_f32) : static_cast<const void*>(&beta0_f64);
      const void* alpha2 = use_f32 ? static_cast<const void*>(&alpha2_f32) : static_cast<const void*>(&alpha2_f64);

      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              naux,
              nrows,
              nops,
              alpha1,
              d_lfull,
              gemm_data_type,
              naux,
              d_w,
              gemm_data_type,
              nops,
              beta0,
              dW,
              gemm_data_type,
              naux,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(df_gemm_step1)");

      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              nops,
              nrows,
              naux,
              alpha2,
              d_lfull,
              gemm_data_type,
              naux,
              dW,
              gemm_data_type,
              naux,
              beta0,
              d_g,
              gemm_data_type,
              nops,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(df_gemm_step2)");
    }

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(df_gemm_dense)");
    }
  }

  void build_g_from_csr_l_full_inplace_device(
      py::object indptr,
      py::object indices,
      py::object data,
      py::object l_full,
      py::object g_out,
      int threads,
      double half,
      uint64_t stream,
      bool sync) {
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    if (indptr.is_none() || indices.is_none() || data.is_none() || l_full.is_none() || g_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
    require_typestr(indptr_dev, "indptr", "<i8");
    if (indptr_dev.shape.size() != 1 || indptr_dev.shape[0] < 1) {
      throw std::invalid_argument("indptr must be a 1D device array with shape (nrows+1,)");
    }
    if (!indptr_dev.strides_bytes.empty()) {
      if (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
        throw std::invalid_argument("indptr must be contiguous");
      }
    }

    int64_t nrows_ll = indptr_dev.shape[0] - 1;
    if (nrows_ll < 0) {
      throw std::invalid_argument("invalid indptr length");
    }
    if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("nrows too large (batch the work)");
    }
    int nrows = (int)nrows_ll;
    if (nrows > max_nrows) {
      throw std::invalid_argument("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows");
    }

    auto indices_dev = cuda_array_view_from_object(indices, "indices");
    require_typestr(indices_dev, "indices", "<i4");
    if (indices_dev.shape.size() != 1) {
      throw std::invalid_argument("indices must be 1D");
    }
    if (!indices_dev.strides_bytes.empty()) {
      if (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("indices must be contiguous");
      }
    }

    auto data_dev = cuda_array_view_from_object(data, "data");
    bool data_is_f32 = false;
    if (data_dev.typestr == "<f8") {
      data_is_f32 = false;
    } else if (data_dev.typestr == "<f4") {
      data_is_f32 = true;
    } else {
      throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
    }
    const int64_t data_itemsize = data_is_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if (data_dev.shape.size() != 1) {
      throw std::invalid_argument("data must be 1D");
    }
    if (data_dev.shape[0] != indices_dev.shape[0]) {
      throw std::invalid_argument("indices and data must have the same length");
    }
    if (!data_dev.strides_bytes.empty()) {
      if (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != data_itemsize) {
        throw std::invalid_argument("data must be contiguous");
      }
    }

    auto l_full_dev = cuda_array_view_from_object(l_full, "l_full");
    bool use_f32 = false;
    if (l_full_dev.typestr == "<f8") {
      use_f32 = false;
    } else if (l_full_dev.typestr == "<f4") {
      use_f32 = true;
    } else {
      throw std::invalid_argument("l_full must have typestr <f8 (float64) or <f4 (float32)");
    }
    if (use_f32 && gemm_data_type != CUDA_R_32F) {
      throw std::runtime_error("float32 DF build-g requires gemm_data_type=CUDA_R_32F (set gemm_backend to *_fp32/*_tf32)");
    }
    if (!use_f32 && gemm_data_type != CUDA_R_64F) {
      throw std::runtime_error("float64 DF build-g requires gemm_data_type=CUDA_R_64F (set gemm_backend to *_fp64)");
    }
    if (!use_f32 && data_is_f32) {
      throw std::invalid_argument("float64 DF build-g does not support float32 CSR data");
    }
    const int64_t fp_itemsize = use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if ((int64_t)w_itemsize != fp_itemsize) {
      throw std::runtime_error("DF workspace dW dtype does not match input dtype; call set_gemm_backend first");
    }
    if (l_full_dev.shape.size() != 2) {
      throw std::invalid_argument("l_full must be a 2D device array");
    }
    if (l_full_dev.shape[0] != (int64_t)nops || l_full_dev.shape[1] != (int64_t)naux) {
      throw std::invalid_argument("l_full must have shape (nops,naux) matching this workspace");
    }
    if (!l_full_dev.strides_bytes.empty()) {
      if (l_full_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("l_full strides must have length 2");
      }
      int64_t s0 = l_full_dev.strides_bytes[0];
      int64_t s1 = l_full_dev.strides_bytes[1];
      if (s0 != (int64_t)naux * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("l_full must be C-contiguous with no padding");
      }
    }

    auto g_dev = cuda_array_view_from_object(g_out, "g_out");
    if (use_f32) require_typestr(g_dev, "g_out", "<f4");
    else require_typestr(g_dev, "g_out", "<f8");
    if (g_dev.read_only) {
      throw std::invalid_argument("g_out must be writable");
    }
    if (g_dev.shape.size() != 2 || g_dev.shape[0] != nrows_ll || g_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("g_out must have shape (nrows,nops)");
    }
    if (!g_dev.strides_bytes.empty()) {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (g_dev.stream) stream_u = g_dev.stream;
      else if (l_full_dev.stream) stream_u = l_full_dev.stream;
      else if (data_dev.stream) stream_u = data_dev.stream;
      else if (indices_dev.stream) stream_u = indices_dev.stream;
      else if (indptr_dev.stream) stream_u = indptr_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const int64_t* d_indptr = reinterpret_cast<const int64_t*>(indptr_dev.ptr);
    const int32_t* d_indices = reinterpret_cast<const int32_t*>(indices_dev.ptr);
    const void* d_data = data_dev.ptr;
    const void* d_lfull = l_full_dev.ptr;
    void* d_g = g_dev.ptr;

    if (nrows == 0 || nops == 0 || naux == 0) return;

    // Build W^T[L,row] = sum_rs C[row,rs] * L_full[rs,L] directly from CSR (avoids dense csr matrix + first GEMM).
    if (use_f32) {
      if (data_is_f32) {
        throw_on_cuda_error(
            guga_csr_l_full_to_wt_f32_range_launch_stream(
                d_indptr,
                d_indices,
                reinterpret_cast<const float*>(d_data),
                /*row_start=*/0,
                nrows,
                reinterpret_cast<const float*>(d_lfull),
                naux,
                reinterpret_cast<float*>(dW),
                stream_t,
                threads),
            "guga_csr_l_full_to_wt_f32_range_launch_stream");
      } else {
        throw_on_cuda_error(
            guga_csr_l_full_to_wt_f32_from_f64_range_launch_stream(
                d_indptr,
                d_indices,
                reinterpret_cast<const double*>(d_data),
                /*row_start=*/0,
                nrows,
                reinterpret_cast<const float*>(d_lfull),
                naux,
                reinterpret_cast<float*>(dW),
                stream_t,
                threads),
            "guga_csr_l_full_to_wt_f32_from_f64_range_launch_stream");
      }
    } else {
      throw_on_cuda_error(
          guga_csr_l_full_to_wt_f64_range_launch_stream(
              d_indptr,
              d_indices,
              reinterpret_cast<const double*>(d_data),
              /*row_start=*/0,
              nrows,
              reinterpret_cast<const double*>(d_lfull),
              naux,
              reinterpret_cast<double*>(dW),
              stream_t,
              threads),
          "guga_csr_l_full_to_wt_f64_range_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(csr_l_full_to_wt_range)");

    const float beta0_f32 = 0.0f;
    const double beta0_f64 = 0.0;
    const void* beta0 = use_f32 ? static_cast<const void*>(&beta0_f32) : static_cast<const void*>(&beta0_f64);

    // Second GEMMEx: G^T = half * L_full @ W^T.
    // Use op(A)=L_full by transposing the column-major view (naux,nops) of the row-major L_full buffer.
    const float alpha2_f32 = (float)half;
    const double alpha2_f64 = half;
    const void* alpha2 = use_f32 ? static_cast<const void*>(&alpha2_f32) : static_cast<const void*>(&alpha2_f64);
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      } else if (gemm_compute_type != CUBLAS_COMPUTE_64F) {
        throw std::runtime_error("float64 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_64F");
      }

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b_desc = nullptr;
      cublasLtMatrixLayout_t c_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;
      try {
        throw_on_cublas_error(
            cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, gemm_data_type), "cublasLtMatmulDescCreate");
        cublasOperation_t opt = CUBLAS_OP_T;
        cublasOperation_t opn = CUBLAS_OP_N;
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opt, sizeof(opt)),
            "cublasLtMatmulDescSetAttribute(transa)");
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transb)");

        // A is L_full^T (col-major [naux,nops]), op(A)=T => L_full (nops,naux)
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&a_desc, gemm_data_type, naux, nops, naux), "cublasLtMatrixLayoutCreate(A)");
        // B is W^T (col-major [naux,nrows]) in dW
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b_desc, gemm_data_type, naux, nrows, naux), "cublasLtMatrixLayoutCreate(B)");
        // C is G^T (col-major [nops,nrows]) backed by row-major g_out
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(C)");
        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C)");

        throw_on_cublas_error(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        throw_on_cublas_error(
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");

        LtAlgoCacheEntry entry;
        bool have_algo = false;
        auto it = lt_cache_step2.find(nrows);
        if (it != lt_cache_step2.end()) {
          entry = it->second;
          have_algo = true;
        }
        if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
          cublasLtMatmulHeuristicResult_t heur{};
          int nret = 0;
          throw_on_cublas_error(
              cublasLtMatmulAlgoGetHeuristic(
                  cublaslt.h, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heur, &nret),
              "cublasLtMatmulAlgoGetHeuristic(build_g_df)");
          if (nret <= 0) {
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (build_g_df)");
          }
          entry.algo = heur.algo;
          entry.workspace_bytes = (size_t)heur.workspaceSize;
          lt_cache_step2[nrows] = entry;
        }

        void* ws_ptr = dCublasWorkspace;
        size_t ws_bytes = cublas_workspace_bytes;
        if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
        if (ws_bytes == 0) ws_ptr = nullptr;
        throw_on_cublas_error(
            cublasLtMatmul(
                cublaslt.h,
                op_desc,
                alpha2,
                d_lfull,
                a_desc,
                dW,
                b_desc,
                beta0,
                d_g,
                c_desc,
                d_g,
                c_desc,
                &entry.algo,
                ws_ptr,
                ws_bytes,
                stream_t),
            "cublasLtMatmul(build_g_df)");
      } catch (...) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        throw;
      }
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
      if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    } else {
      throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");
      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              nops,
              nrows,
              naux,
              alpha2,
              d_lfull,
              gemm_data_type,
              naux,
              dW,
              gemm_data_type,
              naux,
              beta0,
              d_g,
              gemm_data_type,
              nops,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(build_g_df)");
    }

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel3_build_g_df)");
    }
  }

  void build_g_from_csr_l_full_range_inplace_device(
      py::object indptr,
      py::object indices,
      py::object data,
      int row_start,
      int nrows,
      py::object l_full,
      py::object g_out,
      int threads,
      double half,
      uint64_t stream,
      bool sync) {
    if (threads <= 0 || threads > 1024) {
      throw std::invalid_argument("threads must be in 1..1024");
    }
    if (row_start < 0 || nrows < 0) {
      throw std::invalid_argument("row_start and nrows must be >= 0");
    }
    if (indptr.is_none() || indices.is_none() || data.is_none() || l_full.is_none() || g_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
    require_typestr(indptr_dev, "indptr", "<i8");
    if (indptr_dev.shape.size() != 1 || indptr_dev.shape[0] < 1) {
      throw std::invalid_argument("indptr must be a 1D device array with shape (nrows_total+1,)");
    }
    if (!indptr_dev.strides_bytes.empty()) {
      if (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
        throw std::invalid_argument("indptr must be contiguous");
      }
    }

    int64_t nrows_total_ll = indptr_dev.shape[0] - 1;
    if (nrows_total_ll < 0) {
      throw std::invalid_argument("invalid indptr length");
    }
    if (row_start > nrows_total_ll) {
      throw std::invalid_argument("row_start exceeds CSR row count");
    }
    if ((int64_t)row_start + (int64_t)nrows > nrows_total_ll) {
      throw std::invalid_argument("row_start+nrows exceeds CSR row count");
    }
    if (nrows > max_nrows) {
      throw std::invalid_argument("nrows exceeds workspace max_nrows; recreate workspace with a larger max_nrows");
    }

    auto indices_dev = cuda_array_view_from_object(indices, "indices");
    require_typestr(indices_dev, "indices", "<i4");
    if (indices_dev.shape.size() != 1) {
      throw std::invalid_argument("indices must be 1D");
    }
    if (!indices_dev.strides_bytes.empty()) {
      if (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("indices must be contiguous");
      }
    }

    auto data_dev = cuda_array_view_from_object(data, "data");
    bool data_is_f32 = false;
    if (data_dev.typestr == "<f8") {
      data_is_f32 = false;
    } else if (data_dev.typestr == "<f4") {
      data_is_f32 = true;
    } else {
      throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
    }
    const int64_t data_itemsize = data_is_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if (data_dev.shape.size() != 1) {
      throw std::invalid_argument("data must be 1D");
    }
    if (data_dev.shape[0] != indices_dev.shape[0]) {
      throw std::invalid_argument("indices and data must have the same length");
    }
    if (!data_dev.strides_bytes.empty()) {
      if (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != data_itemsize) {
        throw std::invalid_argument("data must be contiguous");
      }
    }

    auto l_full_dev = cuda_array_view_from_object(l_full, "l_full");
    bool use_f32 = false;
    if (l_full_dev.typestr == "<f8") {
      use_f32 = false;
    } else if (l_full_dev.typestr == "<f4") {
      use_f32 = true;
    } else {
      throw std::invalid_argument("l_full must have typestr <f8 (float64) or <f4 (float32)");
    }
    if (use_f32 && gemm_data_type != CUDA_R_32F) {
      throw std::runtime_error("float32 DF build-g requires gemm_data_type=CUDA_R_32F (set gemm_backend to *_fp32/*_tf32)");
    }
    if (!use_f32 && gemm_data_type != CUDA_R_64F) {
      throw std::runtime_error("float64 DF build-g requires gemm_data_type=CUDA_R_64F (set gemm_backend to *_fp64)");
    }
    if (!use_f32 && data_is_f32) {
      throw std::invalid_argument("float64 DF build-g does not support float32 CSR data");
    }
    const int64_t fp_itemsize = use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
    if ((int64_t)w_itemsize != fp_itemsize) {
      throw std::runtime_error("DF workspace dW dtype does not match input dtype; call set_gemm_backend first");
    }
    if (l_full_dev.shape.size() != 2) {
      throw std::invalid_argument("l_full must be a 2D device array");
    }
    if (l_full_dev.shape[0] != (int64_t)nops || l_full_dev.shape[1] != (int64_t)naux) {
      throw std::invalid_argument("l_full must have shape (nops,naux) matching this workspace");
    }
    if (!l_full_dev.strides_bytes.empty()) {
      if (l_full_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("l_full strides must have length 2");
      }
      int64_t s0 = l_full_dev.strides_bytes[0];
      int64_t s1 = l_full_dev.strides_bytes[1];
      if (s0 != (int64_t)naux * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("l_full must be C-contiguous with no padding");
      }
    }

    auto g_dev = cuda_array_view_from_object(g_out, "g_out");
    if (use_f32) require_typestr(g_dev, "g_out", "<f4");
    else require_typestr(g_dev, "g_out", "<f8");
    if (g_dev.read_only) {
      throw std::invalid_argument("g_out must be writable");
    }
    if (g_dev.shape.size() != 2 || g_dev.shape[0] != (int64_t)nrows || g_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("g_out must have shape (nrows,nops)");
    }
    if (!g_dev.strides_bytes.empty()) {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (g_dev.stream) stream_u = g_dev.stream;
      else if (l_full_dev.stream) stream_u = l_full_dev.stream;
      else if (data_dev.stream) stream_u = data_dev.stream;
      else if (indices_dev.stream) stream_u = indices_dev.stream;
      else if (indptr_dev.stream) stream_u = indptr_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const int64_t* d_indptr = reinterpret_cast<const int64_t*>(indptr_dev.ptr);
    const int32_t* d_indices = reinterpret_cast<const int32_t*>(indices_dev.ptr);
    const void* d_data = data_dev.ptr;
    const void* d_lfull = l_full_dev.ptr;
    void* d_g = g_dev.ptr;

    if (nrows == 0 || nops == 0 || naux == 0) return;

    if (use_f32) {
      if (data_is_f32) {
        throw_on_cuda_error(
            guga_csr_l_full_to_wt_f32_range_launch_stream(
                d_indptr,
                d_indices,
                reinterpret_cast<const float*>(d_data),
                row_start,
                nrows,
                reinterpret_cast<const float*>(d_lfull),
                naux,
                reinterpret_cast<float*>(dW),
                stream_t,
                threads),
            "guga_csr_l_full_to_wt_f32_range_launch_stream");
      } else {
        throw_on_cuda_error(
            guga_csr_l_full_to_wt_f32_from_f64_range_launch_stream(
                d_indptr,
                d_indices,
                reinterpret_cast<const double*>(d_data),
                row_start,
                nrows,
                reinterpret_cast<const float*>(d_lfull),
                naux,
                reinterpret_cast<float*>(dW),
                stream_t,
                threads),
            "guga_csr_l_full_to_wt_f32_from_f64_range_launch_stream");
      }
    } else {
      throw_on_cuda_error(
          guga_csr_l_full_to_wt_f64_range_launch_stream(
              d_indptr,
              d_indices,
              reinterpret_cast<const double*>(d_data),
              row_start,
              nrows,
              reinterpret_cast<const double*>(d_lfull),
              naux,
              reinterpret_cast<double*>(dW),
              stream_t,
              threads),
          "guga_csr_l_full_to_wt_f64_range_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(csr_l_full_to_wt_range)");

    const float beta0_f32 = 0.0f;
    const double beta0_f64 = 0.0;
    const void* beta0 = use_f32 ? static_cast<const void*>(&beta0_f32) : static_cast<const void*>(&beta0_f64);
    const float alpha2_f32 = (float)half;
    const double alpha2_f64 = half;
    const void* alpha2 = use_f32 ? static_cast<const void*>(&alpha2_f32) : static_cast<const void*>(&alpha2_f64);
    if (gemm_backend_kind == GemmBackendKind::CUBLASLT) {
      if (use_f32) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
        if (gemm_compute_type != CUBLAS_COMPUTE_32F && gemm_compute_type != CUBLAS_COMPUTE_32F_FAST_TF32) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F or FAST_TF32");
        }
#else
        if (gemm_compute_type != CUBLAS_COMPUTE_32F) {
          throw std::runtime_error("float32 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_32F");
        }
#endif
      } else if (gemm_compute_type != CUBLAS_COMPUTE_64F) {
        throw std::runtime_error("float64 cublasLt requires gemm_compute_type=CUBLAS_COMPUTE_64F");
      }

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b_desc = nullptr;
      cublasLtMatrixLayout_t c_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;
      try {
        throw_on_cublas_error(
            cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, gemm_data_type), "cublasLtMatmulDescCreate");
        cublasOperation_t opt = CUBLAS_OP_T;
        cublasOperation_t opn = CUBLAS_OP_N;
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opt, sizeof(opt)),
            "cublasLtMatmulDescSetAttribute(transa)");
        throw_on_cublas_error(
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)),
            "cublasLtMatmulDescSetAttribute(transb)");

        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&a_desc, gemm_data_type, naux, nops, naux), "cublasLtMatrixLayoutCreate(A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&b_desc, gemm_data_type, naux, nrows, naux), "cublasLtMatrixLayoutCreate(B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutCreate(&c_desc, gemm_data_type, nops, nrows, nops), "cublasLtMatrixLayoutCreate(C)");
        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order A)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order B)");
        throw_on_cublas_error(
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(order C)");

        throw_on_cublas_error(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        throw_on_cublas_error(
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");

        LtAlgoCacheEntry entry;
        bool have_algo = false;
        auto it = lt_cache_step2.find(nrows);
        if (it != lt_cache_step2.end()) {
          entry = it->second;
          have_algo = true;
        }
        if (!have_algo || entry.workspace_bytes > cublas_workspace_bytes) {
          cublasLtMatmulHeuristicResult_t heur{};
          int nret = 0;
          throw_on_cublas_error(
              cublasLtMatmulAlgoGetHeuristic(
                  cublaslt.h, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heur, &nret),
              "cublasLtMatmulAlgoGetHeuristic(build_g_df_range)");
          if (nret <= 0) {
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms (build_g_df_range)");
          }
          entry.algo = heur.algo;
          entry.workspace_bytes = (size_t)heur.workspaceSize;
          lt_cache_step2[nrows] = entry;
        }

        void* ws_ptr = dCublasWorkspace;
        size_t ws_bytes = cublas_workspace_bytes;
        if (entry.workspace_bytes < ws_bytes) ws_bytes = entry.workspace_bytes;
        if (ws_bytes == 0) ws_ptr = nullptr;
        throw_on_cublas_error(
            cublasLtMatmul(
                cublaslt.h,
                op_desc,
                alpha2,
                d_lfull,
                a_desc,
                dW,
                b_desc,
                beta0,
                d_g,
                c_desc,
                d_g,
                c_desc,
                &entry.algo,
                ws_ptr,
                ws_bytes,
                stream_t),
            "cublasLtMatmul(build_g_df_range)");
      } catch (...) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        throw;
      }
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
      if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    } else {
      throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");
      throw_on_cublas_error(
          cublasGemmEx(
              cublas.h,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              nops,
              nrows,
              naux,
              alpha2,
              d_lfull,
              gemm_data_type,
              naux,
              dW,
              gemm_data_type,
              naux,
              beta0,
              d_g,
              gemm_data_type,
              nops,
              gemm_compute_type,
              gemm_algo),
          "cublasGemmEx(build_g_df_range)");
    }

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel3_build_g_df_range)");
    }
  }
};

struct RDMGramGemmExWorkspace {
  int nops = 0;

  CublasHandle cublas;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_64F;
  cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;

  void* dCublasWorkspace = nullptr;
  size_t cublas_workspace_bytes = 0;

  explicit RDMGramGemmExWorkspace(int nops_) : nops(std::max(0, nops_)) {
    if (nops <= 0) {
      throw std::invalid_argument("nops must be >= 1");
    }
  }

  ~RDMGramGemmExWorkspace() { release(); }

  RDMGramGemmExWorkspace(const RDMGramGemmExWorkspace&) = delete;
  RDMGramGemmExWorkspace& operator=(const RDMGramGemmExWorkspace&) = delete;

  RDMGramGemmExWorkspace(RDMGramGemmExWorkspace&& other) noexcept { *this = std::move(other); }
  RDMGramGemmExWorkspace& operator=(RDMGramGemmExWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      nops = other.nops;
      cublas = std::move(other.cublas);
      gemm_compute_type = other.gemm_compute_type;
      gemm_algo = other.gemm_algo;
      dCublasWorkspace = other.dCublasWorkspace;
      cublas_workspace_bytes = other.cublas_workspace_bytes;
      other.nops = 0;
      other.gemm_compute_type = CUBLAS_COMPUTE_64F;
      other.gemm_algo = CUBLAS_GEMM_DEFAULT;
      other.dCublasWorkspace = nullptr;
      other.cublas_workspace_bytes = 0;
    }
    return *this;
  }

  void release() noexcept {
    if (dCublasWorkspace) {
      if (cublas.h) cublasSetWorkspace(cublas.h, nullptr, 0);
      cudaFree(dCublasWorkspace);
    }
    dCublasWorkspace = nullptr;
    cublas_workspace_bytes = 0;
  }

  std::string gemm_backend() const {
    if (gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp64";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    if (gemm_compute_type == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) return "gemmex_emulated_fixedpoint";
#endif
    return "gemmex_unknown";
  }

  void set_gemm_backend(const std::string& backend) {
    if (backend == "gemmex_fp64") {
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      return;
    }
    if (backend == "gemmex_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      gemm_compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
      return;
#else
      throw std::runtime_error("gemmex_emulated_fixedpoint requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    throw std::invalid_argument("unknown gemm backend: " + backend);
  }

  int gemm_algo_int() const { return (int)gemm_algo; }

  void set_gemm_algo_int(int algo) { gemm_algo = (cublasGemmAlgo_t)algo; }

  void set_cublas_math_mode(const std::string& mode) {
    if (mode == "default") {
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
      return;
    }
    if (mode == "fp64_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      throw_on_cublas_error(cublasSetMathMode(cublas.h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH), "cublasSetMathMode");
      return;
#else
      throw std::runtime_error("fp64_emulated_fixedpoint math mode requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    throw std::invalid_argument("unknown cuBLAS math mode: " + mode);
  }

  size_t get_cublas_workspace_bytes() const { return cublas_workspace_bytes; }

  void set_cublas_workspace_bytes(size_t bytes) {
    if (bytes == 0) {
      if (dCublasWorkspace) {
        throw_on_cublas_error(cublasSetWorkspace(cublas.h, nullptr, 0), "cublasSetWorkspace");
        cudaFree(dCublasWorkspace);
      }
      dCublasWorkspace = nullptr;
      cublas_workspace_bytes = 0;
      return;
    }
    if (bytes == cublas_workspace_bytes && dCublasWorkspace) {
      throw_on_cublas_error(cublasSetWorkspace(cublas.h, dCublasWorkspace, cublas_workspace_bytes), "cublasSetWorkspace");
      return;
    }
    if (dCublasWorkspace) {
      throw_on_cublas_error(cublasSetWorkspace(cublas.h, nullptr, 0), "cublasSetWorkspace");
      cudaFree(dCublasWorkspace);
    }
    void* raw = nullptr;
    throw_on_cuda_error(cudaMalloc(&raw, bytes), "cudaMalloc(cublas_workspace)");
    dCublasWorkspace = raw;
    cublas_workspace_bytes = bytes;
    throw_on_cublas_error(
        cublasSetWorkspace(cublas.h, dCublasWorkspace, cublas_workspace_bytes), "cublasSetWorkspace");
  }

  py::dict cublas_emulation_info() const {
    int ver = 0;
    throw_on_cublas_error(cublasGetVersion(cublas.h, &ver), "cublasGetVersion");
    py::dict out;
    out["cublas_version"] = ver;

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    throw_on_cublas_error(cublasGetMathMode(cublas.h, &math_mode), "cublasGetMathMode");
    const char* math_name = "unknown";
    switch (math_mode) {
      case CUBLAS_DEFAULT_MATH:
        math_name = "default";
        break;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      case CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH:
        math_name = "fp64_emulated_fixedpoint";
        break;
#endif
      default:
        break;
    }
    out["math_mode"] = (int)math_mode;
    out["math_mode_name"] = std::string(math_name);

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    throw_on_cublas_error(cublasGetEmulationStrategy(cublas.h, &strategy), "cublasGetEmulationStrategy");

    cudaEmulationSpecialValuesSupport special_mask = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    throw_on_cublas_error(
        cublasGetEmulationSpecialValuesSupport(cublas.h, &special_mask), "cublasGetEmulationSpecialValuesSupport");

    cudaEmulationMantissaControl mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaControl(cublas.h, &mantissa_control),
        "cublasGetFixedPointEmulationMantissaControl");

    int max_bits = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMaxMantissaBitCount(cublas.h, &max_bits),
        "cublasGetFixedPointEmulationMaxMantissaBitCount");

    int bit_offset = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaBitOffset(cublas.h, &bit_offset),
        "cublasGetFixedPointEmulationMantissaBitOffset");

    const char* strategy_name = "unknown";
    switch (strategy) {
      case CUBLAS_EMULATION_STRATEGY_DEFAULT:
        strategy_name = "default";
        break;
      case CUBLAS_EMULATION_STRATEGY_PERFORMANT:
        strategy_name = "performant";
        break;
      case CUBLAS_EMULATION_STRATEGY_EAGER:
        strategy_name = "eager";
        break;
      default:
        break;
    }

    const char* mantissa_name = "unknown";
    switch (mantissa_control) {
      case CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC:
        mantissa_name = "dynamic";
        break;
      case CUDA_EMULATION_MANTISSA_CONTROL_FIXED:
        mantissa_name = "fixed";
        break;
      default:
        break;
    }

    out["emulation_supported"] = true;
    out["emulation_strategy"] = (int)strategy;
    out["emulation_strategy_name"] = std::string(strategy_name);
    out["emulation_special_values_support_mask"] = (int)special_mask;
    out["fixed_point_mantissa_control"] = (int)mantissa_control;
    out["fixed_point_mantissa_control_name"] = std::string(mantissa_name);
    out["fixed_point_max_mantissa_bits"] = max_bits;
    out["fixed_point_mantissa_bit_offset"] = bit_offset;
#else
    out["emulation_supported"] = false;
#endif

    out["gemm_backend"] = gemm_backend();
    out["gemm_algo"] = (int)gemm_algo;
    out["cublas_workspace_bytes"] = py::int_(cublas_workspace_bytes);
    return out;
  }

  void set_cublas_emulation_strategy(const std::string& strategy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cublasEmulationStrategy_t s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    if (strategy == "default") {
      s = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    } else if (strategy == "performant") {
      s = CUBLAS_EMULATION_STRATEGY_PERFORMANT;
    } else if (strategy == "eager") {
      s = CUBLAS_EMULATION_STRATEGY_EAGER;
    } else {
      throw std::invalid_argument("unknown emulation strategy: " + strategy);
    }
    throw_on_cublas_error(cublasSetEmulationStrategy(cublas.h, s), "cublasSetEmulationStrategy");
#else
    (void)strategy;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_emulation_special_values_support(int mask) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    auto m = (cudaEmulationSpecialValuesSupport)mask;
    throw_on_cublas_error(cublasSetEmulationSpecialValuesSupport(cublas.h, m), "cublasSetEmulationSpecialValuesSupport");
#else
    (void)mask;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_control(const std::string& control) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    cudaEmulationMantissaControl c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    if (control == "dynamic") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    } else if (control == "fixed") {
      c = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    } else {
      throw std::invalid_argument("unknown mantissa control: " + control);
    }
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaControl(cublas.h, c), "cublasSetFixedPointEmulationMantissaControl");
#else
    (void)control;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_max_mantissa_bits(int max_bits) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMaxMantissaBitCount(cublas.h, max_bits),
        "cublasSetFixedPointEmulationMaxMantissaBitCount");
#else
    (void)max_bits;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void set_cublas_fixed_point_mantissa_bit_offset(int bit_offset) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    throw_on_cublas_error(
        cublasSetFixedPointEmulationMantissaBitOffset(cublas.h, bit_offset),
        "cublasSetFixedPointEmulationMantissaBitOffset");
#else
    (void)bit_offset;
    throw std::runtime_error("cuBLAS emulation APIs require CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
  }

  void gram_and_dm1_inplace_device(py::object t, py::object c, py::object dm1_out, py::object gram_out, uint64_t stream, bool sync) {
    if (t.is_none() || c.is_none() || dm1_out.is_none() || gram_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto t_dev = cuda_array_view_from_object(t, "t");
    require_typestr(t_dev, "t", "<f8");
    if (t_dev.shape.size() != 2 || t_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("t must have shape (nops,ncsf)");
    }
    if (!t_dev.strides_bytes.empty()) {
      if (t_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t strides must have length 2");
      }
      if (t_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t must be C-contiguous along last dimension");
      }
    }

    int64_t ncsf_ll = t_dev.shape[1];
    if (ncsf_ll < 0 || ncsf_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("ncsf too large");
    }
    int ncsf = (int)ncsf_ll;

    int64_t lda = ncsf_ll;
    if (!t_dev.strides_bytes.empty()) {
      int64_t s0 = t_dev.strides_bytes[0];
      if (s0 < ncsf_ll * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t row stride must be a multiple of itemsize");
      }
      lda = s0 / (int64_t)sizeof(double);
    }

    auto c_dev = cuda_array_view_from_object(c, "c");
    require_typestr(c_dev, "c", "<f8");
    if (c_dev.shape.size() != 1 || c_dev.shape[0] != ncsf_ll) {
      throw std::invalid_argument("c must have shape (ncsf,)");
    }
    if (!c_dev.strides_bytes.empty()) {
      if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("c must be contiguous");
      }
    }

    auto dm1_dev = cuda_array_view_from_object(dm1_out, "dm1_out");
    require_typestr(dm1_dev, "dm1_out", "<f8");
    if (dm1_dev.read_only) {
      throw std::invalid_argument("dm1_out must be writable");
    }
    if (dm1_dev.shape.size() != 1 || dm1_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("dm1_out must have shape (nops,)");
    }
    if (!dm1_dev.strides_bytes.empty()) {
      if (dm1_dev.strides_bytes.size() != 1 || dm1_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("dm1_out must be contiguous");
      }
    }

    auto gram_dev = cuda_array_view_from_object(gram_out, "gram_out");
    require_typestr(gram_dev, "gram_out", "<f8");
    if (gram_dev.read_only) {
      throw std::invalid_argument("gram_out must be writable");
    }
    if (gram_dev.shape.size() != 2 || gram_dev.shape[0] != (int64_t)nops || gram_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("gram_out must have shape (nops,nops)");
    }
    if (!gram_dev.strides_bytes.empty()) {
      if (gram_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("gram_out strides must have length 2");
      }
      if (gram_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out must be C-contiguous along last dimension");
      }
    }

    int64_t ldc = (int64_t)nops;
    if (!gram_dev.strides_bytes.empty()) {
      int64_t s0 = gram_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("gram_out row stride must be a multiple of itemsize");
      }
      ldc = s0 / (int64_t)sizeof(double);
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (gram_dev.stream) stream_u = gram_dev.stream;
      else if (dm1_dev.stream) stream_u = dm1_dev.stream;
      else if (t_dev.stream) stream_u = t_dev.stream;
      else if (c_dev.stream) stream_u = c_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const double* d_t = reinterpret_cast<const double*>(t_dev.ptr);
    const double* d_c = reinterpret_cast<const double*>(c_dev.ptr);
    double* d_dm1 = reinterpret_cast<double*>(dm1_dev.ptr);
    double* d_gram = reinterpret_cast<double*>(gram_dev.ptr);

    throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");

    const double alpha = 1.0;
    const double beta0 = 0.0;

    // dm1_pq[op] = dot(c, t[op,:]) = (T_cm^T @ c)[op], where T_cm is the column-major view (ncsf,nops) of t.
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            nops,
            1,
            ncsf,
            &alpha,
            d_t,
            CUDA_R_64F,
            (int)lda,
            d_c,
            CUDA_R_64F,
            ncsf,
            &beta0,
            d_dm1,
            CUDA_R_64F,
            nops,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_dm1)");

    // gram0 = T @ T^T. Use the same column-major view: gram0 = T_cm^T @ T_cm.
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            nops,
            nops,
            ncsf,
            &alpha,
            d_t,
            CUDA_R_64F,
            (int)lda,
            d_t,
            CUDA_R_64F,
            (int)lda,
            &beta0,
            d_gram,
            CUDA_R_64F,
            (int)ldc,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_gram)");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(rdm_gram_dm1)");
    }
  }

  void gram_and_dm1_csf_major_inplace_device(
      py::object t_csf_major,
      py::object c,
      py::object dm1_out,
      py::object gram_out,
      uint64_t stream,
      bool sync,
      bool accumulate = false) {
    if (t_csf_major.is_none() || c.is_none() || dm1_out.is_none() || gram_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto t_dev = cuda_array_view_from_object(t_csf_major, "t");
    require_typestr(t_dev, "t", "<f8");
    if (t_dev.shape.size() != 2 || t_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("t must have shape (ncsf,nops)");
    }
    if (!t_dev.strides_bytes.empty()) {
      if (t_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t strides must have length 2");
      }
      if (t_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t must be C-contiguous along last dimension");
      }
    }

    int64_t ncsf_ll = t_dev.shape[0];
    if (ncsf_ll < 0 || ncsf_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("ncsf too large");
    }
    int ncsf = (int)ncsf_ll;

    // Treat row-major t(ncsf,nops) as a column-major matrix A(nops,ncsf), where A[pq,i] = t[i,pq].
    int64_t lda = (int64_t)nops;
    if (!t_dev.strides_bytes.empty()) {
      int64_t s0 = t_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t row stride must be a multiple of itemsize");
      }
      lda = s0 / (int64_t)sizeof(double);
    }

    auto c_dev = cuda_array_view_from_object(c, "c");
    require_typestr(c_dev, "c", "<f8");
    if (c_dev.shape.size() != 1 || c_dev.shape[0] != ncsf_ll) {
      throw std::invalid_argument("c must have shape (ncsf,)");
    }
    if (!c_dev.strides_bytes.empty()) {
      if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("c must be contiguous");
      }
    }

    auto dm1_dev = cuda_array_view_from_object(dm1_out, "dm1_out");
    require_typestr(dm1_dev, "dm1_out", "<f8");
    if (dm1_dev.read_only) {
      throw std::invalid_argument("dm1_out must be writable");
    }
    if (dm1_dev.shape.size() != 1 || dm1_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("dm1_out must have shape (nops,)");
    }
    if (!dm1_dev.strides_bytes.empty()) {
      if (dm1_dev.strides_bytes.size() != 1 || dm1_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("dm1_out must be contiguous");
      }
    }

    auto gram_dev = cuda_array_view_from_object(gram_out, "gram_out");
    require_typestr(gram_dev, "gram_out", "<f8");
    if (gram_dev.read_only) {
      throw std::invalid_argument("gram_out must be writable");
    }
    if (gram_dev.shape.size() != 2 || gram_dev.shape[0] != (int64_t)nops || gram_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("gram_out must have shape (nops,nops)");
    }
    if (!gram_dev.strides_bytes.empty()) {
      if (gram_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("gram_out strides must have length 2");
      }
      if (gram_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out must be C-contiguous along last dimension");
      }
    }

    int64_t ldc = (int64_t)nops;
    if (!gram_dev.strides_bytes.empty()) {
      int64_t s0 = gram_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("gram_out row stride must be a multiple of itemsize");
      }
      ldc = s0 / (int64_t)sizeof(double);
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (gram_dev.stream) stream_u = gram_dev.stream;
      else if (dm1_dev.stream) stream_u = dm1_dev.stream;
      else if (t_dev.stream) stream_u = t_dev.stream;
      else if (c_dev.stream) stream_u = c_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const double* d_t = reinterpret_cast<const double*>(t_dev.ptr);
    const double* d_c = reinterpret_cast<const double*>(c_dev.ptr);
    double* d_dm1 = reinterpret_cast<double*>(dm1_dev.ptr);
    double* d_gram = reinterpret_cast<double*>(gram_dev.ptr);

    throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");

    const double alpha = 1.0;
    const double beta_val = accumulate ? 1.0 : 0.0;

    // dm1_pq[pq] = dot(c, t[:,pq]) = (A @ c)[pq], where A is the column-major view (nops,ncsf) of t_csf_major.
    // When accumulate=true, dm1_out += A @ c (beta=1).
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            nops,
            1,
            ncsf,
            &alpha,
            d_t,
            CUDA_R_64F,
            (int)lda,
            d_c,
            CUDA_R_64F,
            ncsf,
            &beta_val,
            d_dm1,
            CUDA_R_64F,
            nops,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_dm1_csf)");

    // gram0 = A @ A^T.  When accumulate=true, gram_out += A @ A^T (beta=1).
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            nops,
            nops,
            ncsf,
            &alpha,
            d_t,
            CUDA_R_64F,
            (int)lda,
            d_t,
            CUDA_R_64F,
            (int)lda,
            &beta_val,
            d_gram,
            CUDA_R_64F,
            (int)ldc,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_gram_csf)");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(rdm_gram_dm1_csf)");
    }
  }

  void gram_cross_and_dm1_inplace_device(
      py::object t_bra,
      py::object t_ket,
      py::object c_bra,
      py::object dm1_out,
      py::object gram_out,
      uint64_t stream,
      bool sync) {
    if (t_bra.is_none() || t_ket.is_none() || c_bra.is_none() || dm1_out.is_none() || gram_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto tbra_dev = cuda_array_view_from_object(t_bra, "t_bra");
    require_typestr(tbra_dev, "t_bra", "<f8");
    if (tbra_dev.shape.size() != 2 || tbra_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("t_bra must have shape (nops,ncsf)");
    }
    if (!tbra_dev.strides_bytes.empty()) {
      if (tbra_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t_bra strides must have length 2");
      }
      if (tbra_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_bra must be C-contiguous along last dimension");
      }
    }

    auto tket_dev = cuda_array_view_from_object(t_ket, "t_ket");
    require_typestr(tket_dev, "t_ket", "<f8");
    if (tket_dev.shape.size() != 2 || tket_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("t_ket must have shape (nops,ncsf)");
    }
    if (!tket_dev.strides_bytes.empty()) {
      if (tket_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t_ket strides must have length 2");
      }
      if (tket_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_ket must be C-contiguous along last dimension");
      }
    }

    int64_t ncsf_ll = tket_dev.shape[1];
    if (tbra_dev.shape[1] != ncsf_ll) {
      throw std::invalid_argument("t_bra and t_ket must have the same ncsf");
    }
    if (ncsf_ll < 0 || ncsf_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("ncsf too large");
    }
    int ncsf = (int)ncsf_ll;

    int64_t lda_bra = ncsf_ll;
    if (!tbra_dev.strides_bytes.empty()) {
      int64_t s0 = tbra_dev.strides_bytes[0];
      if (s0 < ncsf_ll * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_bra row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t_bra row stride must be a multiple of itemsize");
      }
      lda_bra = s0 / (int64_t)sizeof(double);
    }

    int64_t lda_ket = ncsf_ll;
    if (!tket_dev.strides_bytes.empty()) {
      int64_t s0 = tket_dev.strides_bytes[0];
      if (s0 < ncsf_ll * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_ket row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t_ket row stride must be a multiple of itemsize");
      }
      lda_ket = s0 / (int64_t)sizeof(double);
    }

    auto c_dev = cuda_array_view_from_object(c_bra, "c_bra");
    require_typestr(c_dev, "c_bra", "<f8");
    if (c_dev.shape.size() != 1 || c_dev.shape[0] != ncsf_ll) {
      throw std::invalid_argument("c_bra must have shape (ncsf,)");
    }
    if (!c_dev.strides_bytes.empty()) {
      if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("c_bra must be contiguous");
      }
    }

    auto dm1_dev = cuda_array_view_from_object(dm1_out, "dm1_out");
    require_typestr(dm1_dev, "dm1_out", "<f8");
    if (dm1_dev.read_only) {
      throw std::invalid_argument("dm1_out must be writable");
    }
    if (dm1_dev.shape.size() != 1 || dm1_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("dm1_out must have shape (nops,)");
    }
    if (!dm1_dev.strides_bytes.empty()) {
      if (dm1_dev.strides_bytes.size() != 1 || dm1_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("dm1_out must be contiguous");
      }
    }

    auto gram_dev = cuda_array_view_from_object(gram_out, "gram_out");
    require_typestr(gram_dev, "gram_out", "<f8");
    if (gram_dev.read_only) {
      throw std::invalid_argument("gram_out must be writable");
    }
    if (gram_dev.shape.size() != 2 || gram_dev.shape[0] != (int64_t)nops || gram_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("gram_out must have shape (nops,nops)");
    }
    if (!gram_dev.strides_bytes.empty()) {
      if (gram_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("gram_out strides must have length 2");
      }
      if (gram_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out must be C-contiguous along last dimension");
      }
    }

    int64_t ldc = (int64_t)nops;
    if (!gram_dev.strides_bytes.empty()) {
      int64_t s0 = gram_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("gram_out row stride must be a multiple of itemsize");
      }
      ldc = s0 / (int64_t)sizeof(double);
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (gram_dev.stream) stream_u = gram_dev.stream;
      else if (dm1_dev.stream) stream_u = dm1_dev.stream;
      else if (tket_dev.stream) stream_u = tket_dev.stream;
      else if (tbra_dev.stream) stream_u = tbra_dev.stream;
      else if (c_dev.stream) stream_u = c_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const double* d_t_bra = reinterpret_cast<const double*>(tbra_dev.ptr);
    const double* d_t_ket = reinterpret_cast<const double*>(tket_dev.ptr);
    const double* d_c_bra = reinterpret_cast<const double*>(c_dev.ptr);
    double* d_dm1 = reinterpret_cast<double*>(dm1_dev.ptr);
    double* d_gram = reinterpret_cast<double*>(gram_dev.ptr);

    throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");

    const double alpha = 1.0;
    const double beta0 = 0.0;

    // dm1_pq[op] = dot(c_bra, t_ket[op,:]) = (T_ket_cm^T @ c_bra)[op]
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            nops,
            1,
            ncsf,
            &alpha,
            d_t_ket,
            CUDA_R_64F,
            (int)lda_ket,
            d_c_bra,
            CUDA_R_64F,
            ncsf,
            &beta0,
            d_dm1,
            CUDA_R_64F,
            nops,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_dm1_trans)");

    // gram0_pq = T_bra @ T_ket^T. Use the column-major views:
    // gram0_pq = T_bra_cm^T @ T_ket_cm.
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            nops,
            nops,
            ncsf,
            &alpha,
            d_t_bra,
            CUDA_R_64F,
            (int)lda_bra,
            d_t_ket,
            CUDA_R_64F,
            (int)lda_ket,
            &beta0,
            d_gram,
            CUDA_R_64F,
            (int)ldc,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_gram_trans)");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(rdm_gram_dm1_trans)");
    }
  }

  void gram_cross_and_dm1_csf_major_inplace_device(
      py::object t_bra_csf_major,
      py::object t_ket_csf_major,
      py::object c_bra,
      py::object dm1_out,
      py::object gram_out,
      uint64_t stream,
      bool sync,
      bool accumulate = false) {
    if (t_bra_csf_major.is_none() || t_ket_csf_major.is_none() || c_bra.is_none() || dm1_out.is_none() || gram_out.is_none()) {
      throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
    }

    auto tbra_dev = cuda_array_view_from_object(t_bra_csf_major, "t_bra");
    require_typestr(tbra_dev, "t_bra", "<f8");
    if (tbra_dev.shape.size() != 2 || tbra_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("t_bra must have shape (ncsf,nops)");
    }
    if (!tbra_dev.strides_bytes.empty()) {
      if (tbra_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t_bra strides must have length 2");
      }
      if (tbra_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_bra must be C-contiguous along last dimension");
      }
    }

    auto tket_dev = cuda_array_view_from_object(t_ket_csf_major, "t_ket");
    require_typestr(tket_dev, "t_ket", "<f8");
    if (tket_dev.shape.size() != 2 || tket_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("t_ket must have shape (ncsf,nops)");
    }
    if (!tket_dev.strides_bytes.empty()) {
      if (tket_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("t_ket strides must have length 2");
      }
      if (tket_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_ket must be C-contiguous along last dimension");
      }
    }

    int64_t ncsf_ll = tket_dev.shape[0];
    if (tbra_dev.shape[0] != ncsf_ll) {
      throw std::invalid_argument("t_bra and t_ket must have the same ncsf");
    }
    if (ncsf_ll < 0 || ncsf_ll > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("ncsf too large");
    }
    int ncsf = (int)ncsf_ll;

    int64_t lda_bra = (int64_t)nops;
    if (!tbra_dev.strides_bytes.empty()) {
      int64_t s0 = tbra_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_bra row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t_bra row stride must be a multiple of itemsize");
      }
      lda_bra = s0 / (int64_t)sizeof(double);
    }

    int64_t lda_ket = (int64_t)nops;
    if (!tket_dev.strides_bytes.empty()) {
      int64_t s0 = tket_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("t_ket row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("t_ket row stride must be a multiple of itemsize");
      }
      lda_ket = s0 / (int64_t)sizeof(double);
    }

    auto c_dev = cuda_array_view_from_object(c_bra, "c_bra");
    require_typestr(c_dev, "c_bra", "<f8");
    if (c_dev.shape.size() != 1 || c_dev.shape[0] != ncsf_ll) {
      throw std::invalid_argument("c_bra must have shape (ncsf,)");
    }
    if (!c_dev.strides_bytes.empty()) {
      if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("c_bra must be contiguous");
      }
    }

    auto dm1_dev = cuda_array_view_from_object(dm1_out, "dm1_out");
    require_typestr(dm1_dev, "dm1_out", "<f8");
    if (dm1_dev.read_only) {
      throw std::invalid_argument("dm1_out must be writable");
    }
    if (dm1_dev.shape.size() != 1 || dm1_dev.shape[0] != (int64_t)nops) {
      throw std::invalid_argument("dm1_out must have shape (nops,)");
    }
    if (!dm1_dev.strides_bytes.empty()) {
      if (dm1_dev.strides_bytes.size() != 1 || dm1_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("dm1_out must be contiguous");
      }
    }

    auto gram_dev = cuda_array_view_from_object(gram_out, "gram_out");
    require_typestr(gram_dev, "gram_out", "<f8");
    if (gram_dev.read_only) {
      throw std::invalid_argument("gram_out must be writable");
    }
    if (gram_dev.shape.size() != 2 || gram_dev.shape[0] != (int64_t)nops || gram_dev.shape[1] != (int64_t)nops) {
      throw std::invalid_argument("gram_out must have shape (nops,nops)");
    }
    if (!gram_dev.strides_bytes.empty()) {
      if (gram_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("gram_out strides must have length 2");
      }
      if (gram_dev.strides_bytes[1] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out must be C-contiguous along last dimension");
      }
    }

    int64_t ldc = (int64_t)nops;
    if (!gram_dev.strides_bytes.empty()) {
      int64_t s0 = gram_dev.strides_bytes[0];
      if (s0 < (int64_t)nops * (int64_t)sizeof(double)) {
        throw std::invalid_argument("gram_out row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("gram_out row stride must be a multiple of itemsize");
      }
      ldc = s0 / (int64_t)sizeof(double);
    }

    uint64_t stream_u = stream;
    if (stream_u == 0) {
      if (gram_dev.stream) stream_u = gram_dev.stream;
      else if (dm1_dev.stream) stream_u = dm1_dev.stream;
      else if (tket_dev.stream) stream_u = tket_dev.stream;
      else if (tbra_dev.stream) stream_u = tbra_dev.stream;
      else if (c_dev.stream) stream_u = c_dev.stream;
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

    const double* d_t_bra = reinterpret_cast<const double*>(tbra_dev.ptr);
    const double* d_t_ket = reinterpret_cast<const double*>(tket_dev.ptr);
    const double* d_c_bra = reinterpret_cast<const double*>(c_dev.ptr);
    double* d_dm1 = reinterpret_cast<double*>(dm1_dev.ptr);
    double* d_gram = reinterpret_cast<double*>(gram_dev.ptr);

    throw_on_cublas_error(cublasSetStream(cublas.h, stream_t), "cublasSetStream");

    const double alpha = 1.0;
    const double beta_val = accumulate ? 1.0 : 0.0;

    // dm1_pq[pq] = dot(c_bra, t_ket[:,pq]) = (A_ket @ c_bra)[pq]
    // When accumulate=true, dm1_out += A_ket @ c_bra (beta=1).
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            nops,
            1,
            ncsf,
            &alpha,
            d_t_ket,
            CUDA_R_64F,
            (int)lda_ket,
            d_c_bra,
            CUDA_R_64F,
            ncsf,
            &beta_val,
            d_dm1,
            CUDA_R_64F,
            nops,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_dm1_trans_csf)");

    // gram0 = A_bra @ A_ket^T.  When accumulate=true, gram_out += A_bra @ A_ket^T (beta=1).
    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            nops,
            nops,
            ncsf,
            &alpha,
            d_t_bra,
            CUDA_R_64F,
            (int)lda_bra,
            d_t_ket,
            CUDA_R_64F,
            (int)lda_ket,
            &beta_val,
            d_gram,
            CUDA_R_64F,
            (int)ldc,
            gemm_compute_type,
            gemm_algo),
        "cublasGemmEx(rdm_gram_trans_csf)");

    if (sync) {
      throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(rdm_gram_dm1_trans_csf)");
    }
  }
};

py::dict device_info() {
  int device = 0;
  throw_on_cuda_error(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp prop{};
  throw_on_cuda_error(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  py::dict out;
  out["device"] = device;
  out["name"] = std::string(prop.name);
  out["cc_major"] = prop.major;
  out["cc_minor"] = prop.minor;
  out["total_global_mem"] = py::int_(prop.totalGlobalMem);
  out["multi_processor_count"] = prop.multiProcessorCount;
  return out;
}

py::dict mem_info() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  throw_on_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

  py::dict out;
  out["free_bytes"] = py::int_(free_bytes);
  out["total_bytes"] = py::int_(total_bytes);
  out["used_bytes"] = py::int_(total_bytes - free_bytes);
  return out;
}

DeviceDRT make_device_drt(
    int norb,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> child,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> node_twos,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> child_prefix) {
  cudaError_t lut_err = guga_init_segment_lut();
  if (lut_err != cudaSuccess) {
    bool lut_soft_fail = (lut_err == cudaErrorOperatingSystem || lut_err == cudaErrorNotSupported);
#ifdef cudaErrorSystemNotReady
    lut_soft_fail = lut_soft_fail || (lut_err == cudaErrorSystemNotReady);
#endif
    if (!lut_soft_fail) {
      throw_on_cuda_error(lut_err, "guga_init_segment_lut");
    }
    (void)cudaGetLastError();  // clear sticky error for soft-fail environments
  }

  if (norb < 1 || norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }

  auto child_buf = child.request();
  auto twos_buf = node_twos.request();
  auto pref_buf = child_prefix.request();

  if (child_buf.ndim != 2 || child_buf.shape[1] != 4) {
    throw std::invalid_argument("child must have shape (nnodes,4)");
  }
  if (twos_buf.ndim != 1) {
    throw std::invalid_argument("node_twos must be 1D");
  }
  if (pref_buf.ndim != 2 || pref_buf.shape[1] != 5) {
    throw std::invalid_argument("child_prefix must have shape (nnodes,5)");
  }

  int nnodes = (int)child_buf.shape[0];
  if ((int)twos_buf.shape[0] != nnodes || (int)pref_buf.shape[0] != nnodes) {
    throw std::invalid_argument("child/node_twos/child_prefix have inconsistent nnodes");
  }

  DeviceDRT out;
  out.norb = norb;
  out.nnodes = nnodes;

  size_t child_bytes = (size_t)nnodes * 4 * sizeof(int32_t);
  size_t twos_bytes = (size_t)nnodes * sizeof(int16_t);
  size_t pref_bytes = (size_t)nnodes * 5 * sizeof(int64_t);

  size_t off_child = 0;
  size_t off_twos = align_up_size(off_child + child_bytes, alignof(int16_t));
  size_t off_pref = align_up_size(off_twos + twos_bytes, alignof(int64_t));
  size_t total_bytes = off_pref + pref_bytes;

  throw_on_cuda_error(cudaMalloc(&out.table_blob, total_bytes), "cudaMalloc(drt_table_blob)");
  out.table_blob_bytes = total_bytes;
  char* base = reinterpret_cast<char*>(out.table_blob);
  out.child = reinterpret_cast<int32_t*>(base + off_child);
  out.node_twos = reinterpret_cast<int16_t*>(base + off_twos);
  out.child_prefix = reinterpret_cast<int64_t*>(base + off_pref);

  throw_on_cuda_error(
      cudaMemcpy(out.child, child_buf.ptr, child_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(child)");
  throw_on_cuda_error(
      cudaMemcpy(out.node_twos, twos_buf.ptr, twos_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(node_twos)");
  throw_on_cuda_error(
      cudaMemcpy(out.child_prefix, pref_buf.ptr, pref_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(child_prefix)");

  return out;
}

DeviceStateCache make_device_state_cache(
    const DeviceDRT& drt,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> steps,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> nodes) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }

  auto steps_buf = steps.request();
  auto nodes_buf = nodes.request();
  if (steps_buf.ndim != 2 || (int)steps_buf.shape[1] != drt.norb) {
    throw std::invalid_argument("steps must have shape (ncsf,norb)");
  }
  int ncsf = (int)steps_buf.shape[0];
  if (ncsf < 0) {
    throw std::invalid_argument("ncsf must be >= 0");
  }
  if (nodes_buf.ndim != 2 || (int)nodes_buf.shape[0] != ncsf || (int)nodes_buf.shape[1] != drt.norb + 1) {
    throw std::invalid_argument("nodes must have shape (ncsf,norb+1)");
  }

  DeviceStateCache out;
  out.norb = drt.norb;
  out.ncsf = ncsf;

  size_t steps_bytes = (size_t)ncsf * (size_t)drt.norb * sizeof(int8_t);
  size_t nodes_bytes = (size_t)ncsf * (size_t)(drt.norb + 1) * sizeof(int32_t);

  throw_on_cuda_error(cudaMalloc((void**)&out.steps, steps_bytes), "cudaMalloc(steps_table)");
  throw_on_cuda_error(cudaMalloc((void**)&out.nodes, nodes_bytes), "cudaMalloc(nodes_table)");

  throw_on_cuda_error(cudaMemcpy(out.steps, steps_buf.ptr, steps_bytes, cudaMemcpyHostToDevice), "H2D steps_table");
  throw_on_cuda_error(cudaMemcpy(out.nodes, nodes_buf.ptr, nodes_bytes, cudaMemcpyHostToDevice), "H2D nodes_table");

  return out;
}

TripletFactorsWorkspace make_triplet_factors_workspace(
    int twos_max,
    py::array_t<float, py::array::c_style | py::array::forcecast> sixj_211_host,
    py::array_t<float, py::array::c_style | py::array::forcecast> t_factor_host) {
  if (twos_max < 0) throw std::invalid_argument("twos_max must be >= 0");
  int n = twos_max + 1;

  auto b1 = sixj_211_host.request();
  auto b2 = t_factor_host.request();
  if (b1.ndim != 3 || (int)b1.shape[0] != n || (int)b1.shape[1] != n || (int)b1.shape[2] != n) {
    throw std::invalid_argument("sixj_211 must have shape (twos_max+1, twos_max+1, twos_max+1)");
  }
  if (b2.ndim != 5 || (int)b2.shape[0] != 2 || (int)b2.shape[1] != n || (int)b2.shape[2] != n || (int)b2.shape[3] != 2 ||
      (int)b2.shape[4] != 2) {
    throw std::invalid_argument("t_factor must have shape (2, twos_max+1, twos_max+1, 2, 2)");
  }

  size_t bytes_sixj = (size_t)n * (size_t)n * (size_t)n * sizeof(float);
  size_t bytes_t = (size_t)2 * (size_t)n * (size_t)n * (size_t)2 * (size_t)2 * sizeof(float);
  size_t total = bytes_sixj + bytes_t;

  TripletFactorsWorkspace ws;
  ws.twos_max = twos_max;
  ws.blob_bytes = total;
  throw_on_cuda_error(cudaMalloc(&ws.blob, total), "cudaMalloc(triplet_factors_blob)");
  ws.sixj_211 = reinterpret_cast<float*>(ws.blob);
  ws.t_factor = reinterpret_cast<float*>(reinterpret_cast<char*>(ws.blob) + bytes_sixj);

  throw_on_cuda_error(cudaMemcpy(ws.sixj_211, b1.ptr, bytes_sixj, cudaMemcpyHostToDevice), "cudaMemcpy(sixj_211)");
  throw_on_cuda_error(cudaMemcpy(ws.t_factor, b2.ptr, bytes_t, cudaMemcpyHostToDevice), "cudaMemcpy(t_factor)");
  return ws;
}

void triplet_apply_contracted_all_m_from_epq_table_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object x,
    py::object h_re,
    py::object h_im,
    const TripletFactorsWorkspace& tf,
    py::object y_re,
    py::object y_im,
    int threads,
    uint64_t stream,
    bool sync) {
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.node_twos == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  require_typestr(epq_indptr_dev, "epq_indptr", "<i8");
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  require_typestr(epq_pq_dev, "epq_pq", "<i4");
  if (epq_pq_dev.shape.size() != 1 || epq_pq_dev.shape[0] != epq_indices_dev.shape[0]) {
    throw std::invalid_argument("epq_pq must have shape (nnz,) matching epq_indices");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  int64_t nops3 = (int64_t)3 * (int64_t)drt.norb * (int64_t)drt.norb;
  auto h_re_dev = cuda_array_view_from_object(h_re, "h_re");
  auto h_im_dev = cuda_array_view_from_object(h_im, "h_im");
  require_typestr(h_re_dev, "h_re", "<f8");
  require_typestr(h_im_dev, "h_im", "<f8");

  auto check_h_layout = [&](const CudaArrayView& h_dev, const char* name) {
    if (h_dev.shape.size() == 1) {
      if (h_dev.shape[0] != nops3) throw std::invalid_argument(std::string(name) + " must have size 3*norb*norb");
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 1 || h_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (h_dev.shape.size() == 3) {
      if (h_dev.shape[0] != 3 || h_dev.shape[1] != (int64_t)drt.norb || h_dev.shape[2] != (int64_t)drt.norb) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,norb,norb)");
      }
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 3 || h_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            h_dev.strides_bytes[1] != (int64_t)drt.norb * (int64_t)sizeof(double) ||
            h_dev.strides_bytes[0] != (int64_t)drt.norb * (int64_t)drt.norb * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_h_layout(h_re_dev, "h_re");
  check_h_layout(h_im_dev, "h_im");

  auto y_re_dev = cuda_array_view_from_object(y_re, "y_re");
  auto y_im_dev = cuda_array_view_from_object(y_im, "y_im");
  require_typestr(y_re_dev, "y_re", "<f8");
  require_typestr(y_im_dev, "y_im", "<f8");
  if (y_re_dev.read_only || y_im_dev.read_only) {
    throw std::invalid_argument("y_re/y_im must be writable");
  }
  if (y_re_dev.shape.size() != 2 || y_re_dev.shape[0] != 3 || y_re_dev.shape[1] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y_re must have shape (3,ncsf)");
  }
  if (y_im_dev.shape.size() != 2 || y_im_dev.shape[0] != 3 || y_im_dev.shape[1] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y_im must have shape (3,ncsf)");
  }
  if (!y_re_dev.strides_bytes.empty()) {
    if (y_re_dev.strides_bytes.size() != 2 || y_re_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        y_re_dev.strides_bytes[0] != (int64_t)state.ncsf * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y_re must be C-contiguous");
    }
  }
  if (!y_im_dev.strides_bytes.empty()) {
    if (y_im_dev.strides_bytes.size() != 2 || y_im_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        y_im_dev.strides_bytes[0] != (int64_t)state.ncsf * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_re_dev.stream) stream_u = y_re_dev.stream;
    else if (y_im_dev.stream) stream_u = y_im_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (h_re_dev.stream) stream_u = h_re_dev.stream;
    else if (h_im_dev.stream) stream_u = h_im_dev.stream;
    else if (epq_indptr_dev.stream) stream_u = epq_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  guga_triplet_apply_contracted_all_m_from_epq_table_launch_stream(
      drt.node_twos,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      reinterpret_cast<const int64_t*>(epq_indptr_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_indices_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_pq_dev.ptr),
      reinterpret_cast<const double*>(x_dev.ptr),
      reinterpret_cast<const double*>(h_re_dev.ptr),
      reinterpret_cast<const double*>(h_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(y_re_dev.ptr),
      reinterpret_cast<double*>(y_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_apply_contracted_all_m_from_epq_table)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_apply_contracted_all_m_from_epq_table)");
  }
}

void triplet_apply_contracted_all_m_dfs_inplace_device(
    const DeviceDRT& drt_bra,
    const DeviceDRT& drt_ket,
    const DeviceStateCache& ket_state,
    py::object x,
    py::object h_re,
    py::object h_im,
    const TripletFactorsWorkspace& tf,
    py::object y_re,
    py::object y_im,
    int threads,
    uint64_t stream,
    bool sync,
    int root_bra,
    int leaf_bra,
    int twos_bra_total,
    int twos_ket_total) {
  if (drt_bra.child == nullptr || drt_bra.node_twos == nullptr || drt_bra.child_prefix == nullptr) {
    throw std::runtime_error("drt_bra is not initialized");
  }
  if (drt_ket.node_twos == nullptr) {
    throw std::runtime_error("drt_ket is not initialized");
  }
  if (ket_state.steps == nullptr || ket_state.nodes == nullptr) {
    throw std::runtime_error("ket_state is not initialized");
  }
  if (drt_bra.norb != drt_ket.norb || drt_ket.norb != ket_state.norb) {
    throw std::invalid_argument("drt_bra/drt_ket/ket_state have inconsistent norb");
  }
  if (drt_bra.norb <= 0 || drt_bra.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (root_bra < 0 || root_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("root_bra is out of range for drt_bra");
  }
  if (leaf_bra < 0 || leaf_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("leaf_bra is out of range for drt_bra");
  }
  if (twos_bra_total < 0) {
    throw std::invalid_argument("twos_bra_total must be >= 0");
  }
  if (twos_ket_total < 0) {
    throw std::invalid_argument("twos_ket_total must be >= 0");
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)ket_state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf_ket,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  int64_t nops3 = (int64_t)3 * (int64_t)drt_ket.norb * (int64_t)drt_ket.norb;
  auto h_re_dev = cuda_array_view_from_object(h_re, "h_re");
  auto h_im_dev = cuda_array_view_from_object(h_im, "h_im");
  require_typestr(h_re_dev, "h_re", "<f8");
  require_typestr(h_im_dev, "h_im", "<f8");

  auto check_h_layout = [&](const CudaArrayView& h_dev, const char* name) {
    if (h_dev.shape.size() == 1) {
      if (h_dev.shape[0] != nops3) throw std::invalid_argument(std::string(name) + " must have size 3*norb*norb");
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 1 || h_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (h_dev.shape.size() == 3) {
      if (h_dev.shape[0] != 3 || h_dev.shape[1] != (int64_t)drt_ket.norb || h_dev.shape[2] != (int64_t)drt_ket.norb) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,norb,norb)");
      }
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 3 || h_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            h_dev.strides_bytes[1] != (int64_t)drt_ket.norb * (int64_t)sizeof(double) ||
            h_dev.strides_bytes[0] != (int64_t)drt_ket.norb * (int64_t)drt_ket.norb * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_h_layout(h_re_dev, "h_re");
  check_h_layout(h_im_dev, "h_im");

  auto y_re_dev = cuda_array_view_from_object(y_re, "y_re");
  auto y_im_dev = cuda_array_view_from_object(y_im, "y_im");
  require_typestr(y_re_dev, "y_re", "<f8");
  require_typestr(y_im_dev, "y_im", "<f8");
  if (y_re_dev.read_only || y_im_dev.read_only) {
    throw std::invalid_argument("y_re/y_im must be writable");
  }
  if (y_re_dev.shape.size() != 2 || y_re_dev.shape[0] != 3) {
    throw std::invalid_argument("y_re must have shape (3,ncsf_bra)");
  }
  if (y_im_dev.shape.size() != 2 || y_im_dev.shape[0] != 3) {
    throw std::invalid_argument("y_im must have shape (3,ncsf_bra)");
  }
  if (y_re_dev.shape[1] != y_im_dev.shape[1]) {
    throw std::invalid_argument("y_re and y_im must have the same ncsf_bra dimension");
  }
  int ncsf_bra = (int)y_re_dev.shape[1];
  if (ncsf_bra <= 0) {
    return;
  }
  if (!y_re_dev.strides_bytes.empty()) {
    if (y_re_dev.strides_bytes.size() != 2 || y_re_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        y_re_dev.strides_bytes[0] != (int64_t)ncsf_bra * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y_re must be C-contiguous");
    }
  }
  if (!y_im_dev.strides_bytes.empty()) {
    if (y_im_dev.strides_bytes.size() != 2 || y_im_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        y_im_dev.strides_bytes[0] != (int64_t)ncsf_bra * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_re_dev.stream) stream_u = y_re_dev.stream;
    else if (y_im_dev.stream) stream_u = y_im_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (h_re_dev.stream) stream_u = h_re_dev.stream;
    else if (h_im_dev.stream) stream_u = h_im_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt_bra, stream_t);
  maybe_set_drt_access_policy_window(drt_ket, stream_t);

  guga_triplet_apply_contracted_all_m_dfs_launch_stream(
      drt_bra.child,
      drt_bra.node_twos,
      drt_bra.child_prefix,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      drt_ket.node_twos,
      ket_state.steps,
      ket_state.nodes,
      ket_state.ncsf,
      drt_ket.norb,
      twos_ket_total,
      reinterpret_cast<const double*>(x_dev.ptr),
      reinterpret_cast<const double*>(h_re_dev.ptr),
      reinterpret_cast<const double*>(h_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(y_re_dev.ptr),
      reinterpret_cast<double*>(y_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_apply_contracted_all_m_dfs)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_apply_contracted_all_m_dfs)");
  }
}

void triplet_build_rho_all_m_from_epq_table_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object c_bra,
    py::object c_ket,
    py::object eta_re,
    py::object eta_im,
    const TripletFactorsWorkspace& tf,
    py::object rho_re,
    py::object rho_im,
    int threads,
    uint64_t stream,
    bool sync) {
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.node_twos == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  require_typestr(epq_indptr_dev, "epq_indptr", "<i8");
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  require_typestr(epq_pq_dev, "epq_pq", "<i4");
  if (epq_pq_dev.shape.size() != 1 || epq_pq_dev.shape[0] != epq_indices_dev.shape[0]) {
    throw std::invalid_argument("epq_pq must have shape (nnz,) matching epq_indices");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto c_bra_dev = cuda_array_view_from_object(c_bra, "c_bra");
  auto c_ket_dev = cuda_array_view_from_object(c_ket, "c_ket");
  require_typestr(c_bra_dev, "c_bra", "<f8");
  require_typestr(c_ket_dev, "c_ket", "<f8");
  if (c_bra_dev.shape.size() != 2 || c_bra_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c_bra must have shape (ncsf,nb)");
  }
  if (c_ket_dev.shape.size() != 2 || c_ket_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c_ket must have shape (ncsf,nk)");
  }
  int nb = (int)c_bra_dev.shape[1];
  int nk = (int)c_ket_dev.shape[1];
  if (nb <= 0 || nk <= 0) return;
  if (!c_bra_dev.strides_bytes.empty()) {
    if (c_bra_dev.strides_bytes.size() != 2 || c_bra_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_bra_dev.strides_bytes[0] != (int64_t)nb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_bra must be C-contiguous");
    }
  }
  if (!c_ket_dev.strides_bytes.empty()) {
    if (c_ket_dev.strides_bytes.size() != 2 || c_ket_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_ket_dev.strides_bytes[0] != (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_ket must be C-contiguous");
    }
  }

  auto eta_re_dev = cuda_array_view_from_object(eta_re, "eta_re");
  auto eta_im_dev = cuda_array_view_from_object(eta_im, "eta_im");
  require_typestr(eta_re_dev, "eta_re", "<f8");
  require_typestr(eta_im_dev, "eta_im", "<f8");
  auto check_eta_layout = [&](const CudaArrayView& eta_dev, const char* name) {
    int64_t n_eta = (int64_t)3 * (int64_t)nb * (int64_t)nk;
    if (eta_dev.shape.size() == 1) {
      if (eta_dev.shape[0] != n_eta) throw std::invalid_argument(std::string(name) + " must have size 3*nb*nk");
      if (!eta_dev.strides_bytes.empty()) {
        if (eta_dev.strides_bytes.size() != 1 || eta_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (eta_dev.shape.size() == 3) {
      if (eta_dev.shape[0] != 3 || eta_dev.shape[1] != (int64_t)nb || eta_dev.shape[2] != (int64_t)nk) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,nb,nk)");
      }
      if (!eta_dev.strides_bytes.empty()) {
        if (eta_dev.strides_bytes.size() != 3 || eta_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            eta_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
            eta_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_eta_layout(eta_re_dev, "eta_re");
  check_eta_layout(eta_im_dev, "eta_im");

  auto rho_re_dev = cuda_array_view_from_object(rho_re, "rho_re");
  auto rho_im_dev = cuda_array_view_from_object(rho_im, "rho_im");
  require_typestr(rho_re_dev, "rho_re", "<f8");
  require_typestr(rho_im_dev, "rho_im", "<f8");
  if (rho_re_dev.read_only || rho_im_dev.read_only) {
    throw std::invalid_argument("rho_re/rho_im must be writable");
  }
  if (rho_re_dev.shape.size() != 3 || rho_re_dev.shape[0] != 3 || rho_re_dev.shape[1] != (int64_t)drt.norb ||
      rho_re_dev.shape[2] != (int64_t)drt.norb) {
    throw std::invalid_argument("rho_re must have shape (3,norb,norb)");
  }
  if (rho_im_dev.shape.size() != 3 || rho_im_dev.shape[0] != 3 || rho_im_dev.shape[1] != (int64_t)drt.norb ||
      rho_im_dev.shape[2] != (int64_t)drt.norb) {
    throw std::invalid_argument("rho_im must have shape (3,norb,norb)");
  }
  if (!rho_re_dev.strides_bytes.empty()) {
    if (rho_re_dev.strides_bytes.size() != 3 || rho_re_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        rho_re_dev.strides_bytes[1] != (int64_t)drt.norb * (int64_t)sizeof(double) ||
        rho_re_dev.strides_bytes[0] != (int64_t)drt.norb * (int64_t)drt.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("rho_re must be C-contiguous");
    }
  }
  if (!rho_im_dev.strides_bytes.empty()) {
    if (rho_im_dev.strides_bytes.size() != 3 || rho_im_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        rho_im_dev.strides_bytes[1] != (int64_t)drt.norb * (int64_t)sizeof(double) ||
        rho_im_dev.strides_bytes[0] != (int64_t)drt.norb * (int64_t)drt.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("rho_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (rho_re_dev.stream) stream_u = rho_re_dev.stream;
    else if (rho_im_dev.stream) stream_u = rho_im_dev.stream;
    else if (c_bra_dev.stream) stream_u = c_bra_dev.stream;
    else if (c_ket_dev.stream) stream_u = c_ket_dev.stream;
    else if (eta_re_dev.stream) stream_u = eta_re_dev.stream;
    else if (eta_im_dev.stream) stream_u = eta_im_dev.stream;
    else if (epq_indptr_dev.stream) stream_u = epq_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  guga_triplet_build_rho_all_m_from_epq_table_launch_stream(
      drt.node_twos,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      reinterpret_cast<const int64_t*>(epq_indptr_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_indices_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_pq_dev.ptr),
      reinterpret_cast<const double*>(c_bra_dev.ptr),
      nb,
      reinterpret_cast<const double*>(c_ket_dev.ptr),
      nk,
      reinterpret_cast<const double*>(eta_re_dev.ptr),
      reinterpret_cast<const double*>(eta_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(rho_re_dev.ptr),
      reinterpret_cast<double*>(rho_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_build_rho_all_m_from_epq_table)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_build_rho_all_m_from_epq_table)");
  }
}

void triplet_build_rho_all_m_dfs_inplace_device(
    const DeviceDRT& drt_bra,
    const DeviceDRT& drt_ket,
    const DeviceStateCache& ket_state,
    py::object c_bra,
    py::object c_ket,
    py::object eta_re,
    py::object eta_im,
    const TripletFactorsWorkspace& tf,
    py::object rho_re,
    py::object rho_im,
    int threads,
    uint64_t stream,
    bool sync,
    int root_bra,
    int leaf_bra,
    int twos_bra_total,
    int twos_ket_total) {
  if (drt_bra.child == nullptr || drt_bra.node_twos == nullptr || drt_bra.child_prefix == nullptr) {
    throw std::runtime_error("drt_bra is not initialized");
  }
  if (drt_ket.node_twos == nullptr) {
    throw std::runtime_error("drt_ket is not initialized");
  }
  if (ket_state.steps == nullptr || ket_state.nodes == nullptr) {
    throw std::runtime_error("ket_state is not initialized");
  }
  if (drt_bra.norb != drt_ket.norb || drt_ket.norb != ket_state.norb) {
    throw std::invalid_argument("drt_bra/drt_ket/ket_state have inconsistent norb");
  }
  if (drt_bra.norb <= 0 || drt_bra.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (root_bra < 0 || root_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("root_bra is out of range for drt_bra");
  }
  if (leaf_bra < 0 || leaf_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("leaf_bra is out of range for drt_bra");
  }
  if (twos_bra_total < 0 || twos_ket_total < 0) {
    throw std::invalid_argument("twos totals must be >= 0");
  }

  auto c_bra_dev = cuda_array_view_from_object(c_bra, "c_bra");
  auto c_ket_dev = cuda_array_view_from_object(c_ket, "c_ket");
  require_typestr(c_bra_dev, "c_bra", "<f8");
  require_typestr(c_ket_dev, "c_ket", "<f8");
  if (c_bra_dev.shape.size() != 2) {
    throw std::invalid_argument("c_bra must have shape (ncsf_bra,nb)");
  }
  if (c_ket_dev.shape.size() != 2 || c_ket_dev.shape[0] != (int64_t)ket_state.ncsf) {
    throw std::invalid_argument("c_ket must have shape (ncsf_ket,nk)");
  }
  int ncsf_bra = (int)c_bra_dev.shape[0];
  int nb = (int)c_bra_dev.shape[1];
  int nk = (int)c_ket_dev.shape[1];
  if (ncsf_bra <= 0 || nb <= 0 || nk <= 0) return;
  if (!c_bra_dev.strides_bytes.empty()) {
    if (c_bra_dev.strides_bytes.size() != 2 || c_bra_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_bra_dev.strides_bytes[0] != (int64_t)nb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_bra must be C-contiguous");
    }
  }
  if (!c_ket_dev.strides_bytes.empty()) {
    if (c_ket_dev.strides_bytes.size() != 2 || c_ket_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_ket_dev.strides_bytes[0] != (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_ket must be C-contiguous");
    }
  }

  auto eta_re_dev = cuda_array_view_from_object(eta_re, "eta_re");
  auto eta_im_dev = cuda_array_view_from_object(eta_im, "eta_im");
  require_typestr(eta_re_dev, "eta_re", "<f8");
  require_typestr(eta_im_dev, "eta_im", "<f8");
  auto check_eta_layout = [&](const CudaArrayView& eta_dev, const char* name) {
    int64_t n_eta = (int64_t)3 * (int64_t)nb * (int64_t)nk;
    if (eta_dev.shape.size() == 1) {
      if (eta_dev.shape[0] != n_eta) throw std::invalid_argument(std::string(name) + " must have size 3*nb*nk");
      if (!eta_dev.strides_bytes.empty()) {
        if (eta_dev.strides_bytes.size() != 1 || eta_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (eta_dev.shape.size() == 3) {
      if (eta_dev.shape[0] != 3 || eta_dev.shape[1] != (int64_t)nb || eta_dev.shape[2] != (int64_t)nk) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,nb,nk)");
      }
      if (!eta_dev.strides_bytes.empty()) {
        if (eta_dev.strides_bytes.size() != 3 || eta_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            eta_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
            eta_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_eta_layout(eta_re_dev, "eta_re");
  check_eta_layout(eta_im_dev, "eta_im");

  auto rho_re_dev = cuda_array_view_from_object(rho_re, "rho_re");
  auto rho_im_dev = cuda_array_view_from_object(rho_im, "rho_im");
  require_typestr(rho_re_dev, "rho_re", "<f8");
  require_typestr(rho_im_dev, "rho_im", "<f8");
  if (rho_re_dev.read_only || rho_im_dev.read_only) {
    throw std::invalid_argument("rho_re/rho_im must be writable");
  }
  if (rho_re_dev.shape.size() != 3 || rho_re_dev.shape[0] != 3 || rho_re_dev.shape[1] != (int64_t)drt_ket.norb ||
      rho_re_dev.shape[2] != (int64_t)drt_ket.norb) {
    throw std::invalid_argument("rho_re must have shape (3,norb,norb)");
  }
  if (rho_im_dev.shape.size() != 3 || rho_im_dev.shape[0] != 3 || rho_im_dev.shape[1] != (int64_t)drt_ket.norb ||
      rho_im_dev.shape[2] != (int64_t)drt_ket.norb) {
    throw std::invalid_argument("rho_im must have shape (3,norb,norb)");
  }
  if (!rho_re_dev.strides_bytes.empty()) {
    if (rho_re_dev.strides_bytes.size() != 3 || rho_re_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        rho_re_dev.strides_bytes[1] != (int64_t)drt_ket.norb * (int64_t)sizeof(double) ||
        rho_re_dev.strides_bytes[0] != (int64_t)drt_ket.norb * (int64_t)drt_ket.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("rho_re must be C-contiguous");
    }
  }
  if (!rho_im_dev.strides_bytes.empty()) {
    if (rho_im_dev.strides_bytes.size() != 3 || rho_im_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        rho_im_dev.strides_bytes[1] != (int64_t)drt_ket.norb * (int64_t)sizeof(double) ||
        rho_im_dev.strides_bytes[0] != (int64_t)drt_ket.norb * (int64_t)drt_ket.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("rho_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (rho_re_dev.stream) stream_u = rho_re_dev.stream;
    else if (rho_im_dev.stream) stream_u = rho_im_dev.stream;
    else if (c_bra_dev.stream) stream_u = c_bra_dev.stream;
    else if (c_ket_dev.stream) stream_u = c_ket_dev.stream;
    else if (eta_re_dev.stream) stream_u = eta_re_dev.stream;
    else if (eta_im_dev.stream) stream_u = eta_im_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt_bra, stream_t);
  maybe_set_drt_access_policy_window(drt_ket, stream_t);

  guga_triplet_build_rho_all_m_dfs_launch_stream(
      drt_bra.child,
      drt_bra.node_twos,
      drt_bra.child_prefix,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      drt_ket.node_twos,
      ket_state.steps,
      ket_state.nodes,
      ket_state.ncsf,
      drt_ket.norb,
      twos_ket_total,
      reinterpret_cast<const double*>(c_bra_dev.ptr),
      nb,
      reinterpret_cast<const double*>(c_ket_dev.ptr),
      nk,
      reinterpret_cast<const double*>(eta_re_dev.ptr),
      reinterpret_cast<const double*>(eta_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(rho_re_dev.ptr),
      reinterpret_cast<double*>(rho_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_build_rho_all_m_dfs)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_build_rho_all_m_dfs)");
  }
}

void triplet_build_gm_all_m_from_epq_table_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object c_bra,
    py::object c_ket,
    py::object h_re,
    py::object h_im,
    const TripletFactorsWorkspace& tf,
    py::object gm_re,
    py::object gm_im,
    int threads,
    uint64_t stream,
    bool sync) {
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.node_twos == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  require_typestr(epq_indptr_dev, "epq_indptr", "<i8");
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  require_typestr(epq_pq_dev, "epq_pq", "<i4");
  if (epq_pq_dev.shape.size() != 1 || epq_pq_dev.shape[0] != epq_indices_dev.shape[0]) {
    throw std::invalid_argument("epq_pq must have shape (nnz,) matching epq_indices");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto c_bra_dev = cuda_array_view_from_object(c_bra, "c_bra");
  auto c_ket_dev = cuda_array_view_from_object(c_ket, "c_ket");
  require_typestr(c_bra_dev, "c_bra", "<f8");
  require_typestr(c_ket_dev, "c_ket", "<f8");
  if (c_bra_dev.shape.size() != 2 || c_bra_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c_bra must have shape (ncsf,nb)");
  }
  if (c_ket_dev.shape.size() != 2 || c_ket_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c_ket must have shape (ncsf,nk)");
  }
  int nb = (int)c_bra_dev.shape[1];
  int nk = (int)c_ket_dev.shape[1];
  if (nb <= 0 || nk <= 0) return;
  if (!c_bra_dev.strides_bytes.empty()) {
    if (c_bra_dev.strides_bytes.size() != 2 || c_bra_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_bra_dev.strides_bytes[0] != (int64_t)nb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_bra must be C-contiguous");
    }
  }
  if (!c_ket_dev.strides_bytes.empty()) {
    if (c_ket_dev.strides_bytes.size() != 2 || c_ket_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_ket_dev.strides_bytes[0] != (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_ket must be C-contiguous");
    }
  }

  int64_t nops3 = (int64_t)3 * (int64_t)drt.norb * (int64_t)drt.norb;
  auto h_re_dev = cuda_array_view_from_object(h_re, "h_re");
  auto h_im_dev = cuda_array_view_from_object(h_im, "h_im");
  require_typestr(h_re_dev, "h_re", "<f8");
  require_typestr(h_im_dev, "h_im", "<f8");
  auto check_h_layout = [&](const CudaArrayView& h_dev, const char* name) {
    if (h_dev.shape.size() == 1) {
      if (h_dev.shape[0] != nops3) throw std::invalid_argument(std::string(name) + " must have size 3*norb*norb");
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 1 || h_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (h_dev.shape.size() == 3) {
      if (h_dev.shape[0] != 3 || h_dev.shape[1] != (int64_t)drt.norb || h_dev.shape[2] != (int64_t)drt.norb) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,norb,norb)");
      }
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 3 || h_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            h_dev.strides_bytes[1] != (int64_t)drt.norb * (int64_t)sizeof(double) ||
            h_dev.strides_bytes[0] != (int64_t)drt.norb * (int64_t)drt.norb * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_h_layout(h_re_dev, "h_re");
  check_h_layout(h_im_dev, "h_im");

  auto gm_re_dev = cuda_array_view_from_object(gm_re, "gm_re");
  auto gm_im_dev = cuda_array_view_from_object(gm_im, "gm_im");
  require_typestr(gm_re_dev, "gm_re", "<f8");
  require_typestr(gm_im_dev, "gm_im", "<f8");
  if (gm_re_dev.read_only || gm_im_dev.read_only) {
    throw std::invalid_argument("gm_re/gm_im must be writable");
  }
  if (gm_re_dev.shape.size() != 3 || gm_re_dev.shape[0] != 3 || gm_re_dev.shape[1] != (int64_t)nb ||
      gm_re_dev.shape[2] != (int64_t)nk) {
    throw std::invalid_argument("gm_re must have shape (3,nb,nk)");
  }
  if (gm_im_dev.shape.size() != 3 || gm_im_dev.shape[0] != 3 || gm_im_dev.shape[1] != (int64_t)nb ||
      gm_im_dev.shape[2] != (int64_t)nk) {
    throw std::invalid_argument("gm_im must have shape (3,nb,nk)");
  }
  if (!gm_re_dev.strides_bytes.empty()) {
    if (gm_re_dev.strides_bytes.size() != 3 || gm_re_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        gm_re_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
        gm_re_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("gm_re must be C-contiguous");
    }
  }
  if (!gm_im_dev.strides_bytes.empty()) {
    if (gm_im_dev.strides_bytes.size() != 3 || gm_im_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        gm_im_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
        gm_im_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("gm_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (gm_re_dev.stream) stream_u = gm_re_dev.stream;
    else if (gm_im_dev.stream) stream_u = gm_im_dev.stream;
    else if (c_bra_dev.stream) stream_u = c_bra_dev.stream;
    else if (c_ket_dev.stream) stream_u = c_ket_dev.stream;
    else if (h_re_dev.stream) stream_u = h_re_dev.stream;
    else if (h_im_dev.stream) stream_u = h_im_dev.stream;
    else if (epq_indptr_dev.stream) stream_u = epq_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  guga_triplet_build_gm_all_m_from_epq_table_launch_stream(
      drt.node_twos,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      reinterpret_cast<const int64_t*>(epq_indptr_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_indices_dev.ptr),
      reinterpret_cast<const int32_t*>(epq_pq_dev.ptr),
      reinterpret_cast<const double*>(c_bra_dev.ptr),
      nb,
      reinterpret_cast<const double*>(c_ket_dev.ptr),
      nk,
      reinterpret_cast<const double*>(h_re_dev.ptr),
      reinterpret_cast<const double*>(h_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(gm_re_dev.ptr),
      reinterpret_cast<double*>(gm_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_build_gm_all_m_from_epq_table)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_build_gm_all_m_from_epq_table)");
  }
}

void triplet_build_gm_all_m_dfs_inplace_device(
    const DeviceDRT& drt_bra,
    const DeviceDRT& drt_ket,
    const DeviceStateCache& ket_state,
    py::object c_bra,
    py::object c_ket,
    py::object h_re,
    py::object h_im,
    const TripletFactorsWorkspace& tf,
    py::object gm_re,
    py::object gm_im,
    int threads,
    uint64_t stream,
    bool sync,
    int root_bra,
    int leaf_bra,
    int twos_bra_total,
    int twos_ket_total) {
  if (drt_bra.child == nullptr || drt_bra.node_twos == nullptr || drt_bra.child_prefix == nullptr) {
    throw std::runtime_error("drt_bra is not initialized");
  }
  if (drt_ket.node_twos == nullptr) {
    throw std::runtime_error("drt_ket is not initialized");
  }
  if (ket_state.steps == nullptr || ket_state.nodes == nullptr) {
    throw std::runtime_error("ket_state is not initialized");
  }
  if (drt_bra.norb != drt_ket.norb || drt_ket.norb != ket_state.norb) {
    throw std::invalid_argument("drt_bra/drt_ket/ket_state have inconsistent norb");
  }
  if (drt_bra.norb <= 0 || drt_bra.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range");
  }
  if (!tf.sixj_211 || !tf.t_factor) {
    throw std::invalid_argument("TripletFactorsWorkspace is not initialized");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (root_bra < 0 || root_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("root_bra is out of range for drt_bra");
  }
  if (leaf_bra < 0 || leaf_bra >= drt_bra.nnodes) {
    throw std::invalid_argument("leaf_bra is out of range for drt_bra");
  }
  if (twos_bra_total < 0 || twos_ket_total < 0) {
    throw std::invalid_argument("twos totals must be >= 0");
  }

  auto c_bra_dev = cuda_array_view_from_object(c_bra, "c_bra");
  auto c_ket_dev = cuda_array_view_from_object(c_ket, "c_ket");
  require_typestr(c_bra_dev, "c_bra", "<f8");
  require_typestr(c_ket_dev, "c_ket", "<f8");
  if (c_bra_dev.shape.size() != 2) {
    throw std::invalid_argument("c_bra must have shape (ncsf_bra,nb)");
  }
  if (c_ket_dev.shape.size() != 2 || c_ket_dev.shape[0] != (int64_t)ket_state.ncsf) {
    throw std::invalid_argument("c_ket must have shape (ncsf_ket,nk)");
  }
  int ncsf_bra = (int)c_bra_dev.shape[0];
  int nb = (int)c_bra_dev.shape[1];
  int nk = (int)c_ket_dev.shape[1];
  if (ncsf_bra <= 0 || nb <= 0 || nk <= 0) return;
  if (!c_bra_dev.strides_bytes.empty()) {
    if (c_bra_dev.strides_bytes.size() != 2 || c_bra_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_bra_dev.strides_bytes[0] != (int64_t)nb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_bra must be C-contiguous");
    }
  }
  if (!c_ket_dev.strides_bytes.empty()) {
    if (c_ket_dev.strides_bytes.size() != 2 || c_ket_dev.strides_bytes[1] != (int64_t)sizeof(double) ||
        c_ket_dev.strides_bytes[0] != (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("c_ket must be C-contiguous");
    }
  }

  int64_t nops3 = (int64_t)3 * (int64_t)drt_ket.norb * (int64_t)drt_ket.norb;
  auto h_re_dev = cuda_array_view_from_object(h_re, "h_re");
  auto h_im_dev = cuda_array_view_from_object(h_im, "h_im");
  require_typestr(h_re_dev, "h_re", "<f8");
  require_typestr(h_im_dev, "h_im", "<f8");
  auto check_h_layout = [&](const CudaArrayView& h_dev, const char* name) {
    if (h_dev.shape.size() == 1) {
      if (h_dev.shape[0] != nops3) throw std::invalid_argument(std::string(name) + " must have size 3*norb*norb");
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 1 || h_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be contiguous");
        }
      }
      return;
    }
    if (h_dev.shape.size() == 3) {
      if (h_dev.shape[0] != 3 || h_dev.shape[1] != (int64_t)drt_ket.norb || h_dev.shape[2] != (int64_t)drt_ket.norb) {
        throw std::invalid_argument(std::string(name) + " must have shape (3,norb,norb)");
      }
      if (!h_dev.strides_bytes.empty()) {
        if (h_dev.strides_bytes.size() != 3 || h_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
            h_dev.strides_bytes[1] != (int64_t)drt_ket.norb * (int64_t)sizeof(double) ||
            h_dev.strides_bytes[0] != (int64_t)drt_ket.norb * (int64_t)drt_ket.norb * (int64_t)sizeof(double)) {
          throw std::invalid_argument(std::string(name) + " must be C-contiguous");
        }
      }
      return;
    }
    throw std::invalid_argument(std::string(name) + " must be 1D or 3D");
  };
  check_h_layout(h_re_dev, "h_re");
  check_h_layout(h_im_dev, "h_im");

  auto gm_re_dev = cuda_array_view_from_object(gm_re, "gm_re");
  auto gm_im_dev = cuda_array_view_from_object(gm_im, "gm_im");
  require_typestr(gm_re_dev, "gm_re", "<f8");
  require_typestr(gm_im_dev, "gm_im", "<f8");
  if (gm_re_dev.read_only || gm_im_dev.read_only) {
    throw std::invalid_argument("gm_re/gm_im must be writable");
  }
  if (gm_re_dev.shape.size() != 3 || gm_re_dev.shape[0] != 3 || gm_re_dev.shape[1] != (int64_t)nb ||
      gm_re_dev.shape[2] != (int64_t)nk) {
    throw std::invalid_argument("gm_re must have shape (3,nb,nk)");
  }
  if (gm_im_dev.shape.size() != 3 || gm_im_dev.shape[0] != 3 || gm_im_dev.shape[1] != (int64_t)nb ||
      gm_im_dev.shape[2] != (int64_t)nk) {
    throw std::invalid_argument("gm_im must have shape (3,nb,nk)");
  }
  if (!gm_re_dev.strides_bytes.empty()) {
    if (gm_re_dev.strides_bytes.size() != 3 || gm_re_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        gm_re_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
        gm_re_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("gm_re must be C-contiguous");
    }
  }
  if (!gm_im_dev.strides_bytes.empty()) {
    if (gm_im_dev.strides_bytes.size() != 3 || gm_im_dev.strides_bytes[2] != (int64_t)sizeof(double) ||
        gm_im_dev.strides_bytes[1] != (int64_t)nk * (int64_t)sizeof(double) ||
        gm_im_dev.strides_bytes[0] != (int64_t)nb * (int64_t)nk * (int64_t)sizeof(double)) {
      throw std::invalid_argument("gm_im must be C-contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (gm_re_dev.stream) stream_u = gm_re_dev.stream;
    else if (gm_im_dev.stream) stream_u = gm_im_dev.stream;
    else if (c_bra_dev.stream) stream_u = c_bra_dev.stream;
    else if (c_ket_dev.stream) stream_u = c_ket_dev.stream;
    else if (h_re_dev.stream) stream_u = h_re_dev.stream;
    else if (h_im_dev.stream) stream_u = h_im_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt_bra, stream_t);
  maybe_set_drt_access_policy_window(drt_ket, stream_t);

  guga_triplet_build_gm_all_m_dfs_launch_stream(
      drt_bra.child,
      drt_bra.node_twos,
      drt_bra.child_prefix,
      root_bra,
      leaf_bra,
      ncsf_bra,
      twos_bra_total,
      drt_ket.node_twos,
      ket_state.steps,
      ket_state.nodes,
      ket_state.ncsf,
      drt_ket.norb,
      twos_ket_total,
      reinterpret_cast<const double*>(c_bra_dev.ptr),
      nb,
      reinterpret_cast<const double*>(c_ket_dev.ptr),
      nk,
      reinterpret_cast<const double*>(h_re_dev.ptr),
      reinterpret_cast<const double*>(h_im_dev.ptr),
      tf.sixj_211,
      tf.t_factor,
      tf.twos_max,
      reinterpret_cast<double*>(gm_re_dev.ptr),
      reinterpret_cast<double*>(gm_im_dev.ptr),
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(triplet_build_gm_all_m_dfs)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(triplet_build_gm_all_m_dfs)");
  }
}

py::tuple epq_contribs_many_deterministic(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_csf,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_p,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_q,
    int threads,
    int64_t max_total_out) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (state.ncsf < 0) {
    throw std::invalid_argument("DeviceStateCache has invalid ncsf");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto csf_buf = task_csf.request();
  auto p_buf = task_p.request();
  auto q_buf = task_q.request();

  if (csf_buf.ndim != 1 || p_buf.ndim != 1 || q_buf.ndim != 1) {
    throw std::invalid_argument("task arrays must be 1D");
  }
  if (p_buf.shape[0] != csf_buf.shape[0] || q_buf.shape[0] != csf_buf.shape[0]) {
    throw std::invalid_argument("task arrays must have the same length");
  }

  int ntasks = (int)csf_buf.shape[0];
  if (ntasks < 0) {
    throw std::invalid_argument("ntasks must be >= 0");
  }

  py::array_t<int64_t> out_offsets({(py::ssize_t)ntasks + 1});
  auto* out_offsets_ptr = out_offsets.mutable_data();
  out_offsets_ptr[0] = 0;

  if (ntasks == 0) {
    py::array_t<int32_t> out_i({0});
    py::array_t<double> out_v({0});
    return py::make_tuple(out_i, out_v, out_offsets);
  }

  size_t task_bytes = (size_t)ntasks * sizeof(int32_t);

  int32_t* d_task_csf_raw = nullptr;
  int32_t* d_task_p_raw = nullptr;
  int32_t* d_task_q_raw = nullptr;
  int32_t* d_counts_raw = nullptr;
  int* d_overflow_raw = nullptr;
  int64_t* d_offsets_raw = nullptr;
  int32_t* d_out_idx_raw = nullptr;
  double* d_out_coeff_raw = nullptr;

  cuda_unique_ptr<int32_t> d_task_csf;
  cuda_unique_ptr<int32_t> d_task_p;
  cuda_unique_ptr<int32_t> d_task_q;
  cuda_unique_ptr<int32_t> d_counts;
  cuda_unique_ptr<int> d_overflow;
  cuda_unique_ptr<int64_t> d_offsets;
  cuda_unique_ptr<int32_t> d_out_idx;
  cuda_unique_ptr<double> d_out_coeff;

  throw_on_cuda_error(cudaMalloc((void**)&d_task_csf_raw, task_bytes), "cudaMalloc(task_csf)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_p_raw, task_bytes), "cudaMalloc(task_p)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_q_raw, task_bytes), "cudaMalloc(task_q)");
  throw_on_cuda_error(cudaMalloc((void**)&d_counts_raw, task_bytes), "cudaMalloc(counts)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow_raw, sizeof(int)), "cudaMalloc(overflow)");

  d_task_csf.reset(d_task_csf_raw);
  d_task_p.reset(d_task_p_raw);
  d_task_q.reset(d_task_q_raw);
  d_counts.reset(d_counts_raw);
  d_overflow.reset(d_overflow_raw);

  throw_on_cuda_error(cudaMemcpy(d_task_csf.get(), csf_buf.ptr, task_bytes, cudaMemcpyHostToDevice), "H2D task_csf");
  throw_on_cuda_error(cudaMemcpy(d_task_p.get(), p_buf.ptr, task_bytes, cudaMemcpyHostToDevice), "H2D task_p");
  throw_on_cuda_error(cudaMemcpy(d_task_q.get(), q_buf.ptr, task_bytes, cudaMemcpyHostToDevice), "H2D task_q");

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_epq_contribs_many_count_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_p.get(),
      d_task_q.get(),
      ntasks,
      d_counts.get(),
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(count)");

  int h_overflow = 0;
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("epq_contribs_many_deterministic overflow in count kernel");
  }

  std::vector<int32_t> h_counts((size_t)ntasks);
  throw_on_cuda_error(
      cudaMemcpy(h_counts.data(), d_counts.get(), task_bytes, cudaMemcpyDeviceToHost), "D2H counts");

  int64_t total = 0;
  for (int i = 0; i < ntasks; i++) {
    int32_t c = h_counts[(size_t)i];
    if (c < 0) {
      throw std::runtime_error("negative count returned by GPU");
    }
    if (total > (std::numeric_limits<int64_t>::max() - (int64_t)c)) {
      throw std::runtime_error("total output size overflow");
    }
    total += (int64_t)c;
    out_offsets_ptr[i + 1] = total;
  }

  if (max_total_out >= 0 && total > max_total_out) {
    throw std::runtime_error("epq_contribs_many_deterministic total output exceeds max_total_out");
  }

  if (total <= 0) {
    py::array_t<int32_t> out_i({0});
    py::array_t<double> out_v({0});
    return py::make_tuple(out_i, out_v, out_offsets);
  }

  size_t offsets_bytes = (size_t)(ntasks + 1) * sizeof(int64_t);
  throw_on_cuda_error(cudaMalloc((void**)&d_offsets_raw, offsets_bytes), "cudaMalloc(offsets)");
  d_offsets.reset(d_offsets_raw);
  throw_on_cuda_error(cudaMemcpy(d_offsets.get(), out_offsets_ptr, offsets_bytes, cudaMemcpyHostToDevice), "H2D offsets");

  if ((uint64_t)total > (uint64_t)(std::numeric_limits<size_t>::max() / sizeof(int32_t))) {
    throw std::runtime_error("output too large for size_t");
  }
  size_t out_i_bytes = (size_t)total * sizeof(int32_t);
  size_t out_v_bytes = (size_t)total * sizeof(double);

  throw_on_cuda_error(cudaMalloc((void**)&d_out_idx_raw, out_i_bytes), "cudaMalloc(out_idx)");
  throw_on_cuda_error(cudaMalloc((void**)&d_out_coeff_raw, out_v_bytes), "cudaMalloc(out_coeff)");
  d_out_idx.reset(d_out_idx_raw);
  d_out_coeff.reset(d_out_coeff_raw);

  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_epq_contribs_many_write_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_p.get(),
      d_task_q.get(),
      ntasks,
      d_offsets.get(),
      d_out_idx.get(),
      d_out_coeff.get(),
      nullptr,
      nullptr,
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(write)");

  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("epq_contribs_many_deterministic overflow in write kernel");
  }

  py::array_t<int32_t> out_i({(py::ssize_t)total});
  py::array_t<double> out_v({(py::ssize_t)total});
  throw_on_cuda_error(cudaMemcpy(out_i.mutable_data(), d_out_idx.get(), out_i_bytes, cudaMemcpyDeviceToHost), "D2H out_idx");
  throw_on_cuda_error(
      cudaMemcpy(out_v.mutable_data(), d_out_coeff.get(), out_v_bytes, cudaMemcpyDeviceToHost), "D2H out_coeff");

  return py::make_tuple(out_i, out_v, out_offsets);
}

py::object epq_apply_weighted_many_atomic(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_csf,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_p,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_q,
    py::array_t<double, py::array::c_style | py::array::forcecast> task_wgt,
    py::object task_scale,
    py::object y0,
    int threads,
    bool return_y) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto csf_buf = task_csf.request();
  auto p_buf = task_p.request();
  auto q_buf = task_q.request();
  auto w_buf = task_wgt.request();

  if (csf_buf.ndim != 1 || p_buf.ndim != 1 || q_buf.ndim != 1 || w_buf.ndim != 1) {
    throw std::invalid_argument("task arrays must be 1D");
  }
  if (p_buf.shape[0] != csf_buf.shape[0] || q_buf.shape[0] != csf_buf.shape[0] || w_buf.shape[0] != csf_buf.shape[0]) {
    throw std::invalid_argument("task arrays must have the same length");
  }

  int ntasks = (int)csf_buf.shape[0];
  if (ntasks < 0) {
    throw std::invalid_argument("ntasks must be >= 0");
  }

  const double* h_scale_ptr = nullptr;
  py::array_t<double> scale_arr;
  if (!task_scale.is_none()) {
    scale_arr = task_scale.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto s_buf = scale_arr.request();
    if (s_buf.ndim != 1 || s_buf.shape[0] != csf_buf.shape[0]) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    h_scale_ptr = static_cast<const double*>(s_buf.ptr);
  }

  const double* h_y0_ptr = nullptr;
  py::array_t<double> y0_arr;
  if (!y0.is_none()) {
    y0_arr = y0.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto y_buf = y0_arr.request();
    if (y_buf.ndim != 1 || (int)y_buf.shape[0] != state.ncsf) {
      throw std::invalid_argument("y0 must have shape (ncsf,) when provided");
    }
    h_y0_ptr = static_cast<const double*>(y_buf.ptr);
  }

  if (ntasks == 0) {
    if (!return_y) return py::none();
    py::array_t<double> out_y({state.ncsf});
    std::fill_n(out_y.mutable_data(), (size_t)state.ncsf, 0.0);
    if (h_y0_ptr) {
      std::copy_n(h_y0_ptr, (size_t)state.ncsf, out_y.mutable_data());
    }
    return std::move(out_y);
  }

  int32_t* d_task_csf_raw = nullptr;
  int32_t* d_task_p_raw = nullptr;
  int32_t* d_task_q_raw = nullptr;
  double* d_task_w_raw = nullptr;
  double* d_task_scale_raw = nullptr;
  double* d_y_raw = nullptr;
  int* d_overflow_raw = nullptr;

  cuda_unique_ptr<int32_t> d_task_csf;
  cuda_unique_ptr<int32_t> d_task_p;
  cuda_unique_ptr<int32_t> d_task_q;
  cuda_unique_ptr<double> d_task_wgt;
  cuda_unique_ptr<double> d_task_scale_dev;
  cuda_unique_ptr<double> d_y;
  cuda_unique_ptr<int> d_overflow;

  size_t task_i_bytes = (size_t)ntasks * sizeof(int32_t);
  size_t task_w_bytes = (size_t)ntasks * sizeof(double);
  size_t y_bytes = (size_t)state.ncsf * sizeof(double);

  throw_on_cuda_error(cudaMalloc((void**)&d_task_csf_raw, task_i_bytes), "cudaMalloc(task_csf)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_p_raw, task_i_bytes), "cudaMalloc(task_p)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_q_raw, task_i_bytes), "cudaMalloc(task_q)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_w_raw, task_w_bytes), "cudaMalloc(task_wgt)");
  throw_on_cuda_error(cudaMalloc((void**)&d_y_raw, y_bytes), "cudaMalloc(y)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow_raw, sizeof(int)), "cudaMalloc(overflow)");

  d_task_csf.reset(d_task_csf_raw);
  d_task_p.reset(d_task_p_raw);
  d_task_q.reset(d_task_q_raw);
  d_task_wgt.reset(d_task_w_raw);
  d_y.reset(d_y_raw);
  d_overflow.reset(d_overflow_raw);

  throw_on_cuda_error(cudaMemcpy(d_task_csf.get(), csf_buf.ptr, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_csf");
  throw_on_cuda_error(cudaMemcpy(d_task_p.get(), p_buf.ptr, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_p");
  throw_on_cuda_error(cudaMemcpy(d_task_q.get(), q_buf.ptr, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_q");
  throw_on_cuda_error(cudaMemcpy(d_task_wgt.get(), w_buf.ptr, task_w_bytes, cudaMemcpyHostToDevice), "H2D task_wgt");

  if (h_scale_ptr) {
    throw_on_cuda_error(cudaMalloc((void**)&d_task_scale_raw, task_w_bytes), "cudaMalloc(task_scale)");
    d_task_scale_dev.reset(d_task_scale_raw);
    throw_on_cuda_error(
        cudaMemcpy(d_task_scale_dev.get(), h_scale_ptr, task_w_bytes, cudaMemcpyHostToDevice), "H2D task_scale");
  }

  if (h_y0_ptr) {
    throw_on_cuda_error(cudaMemcpy(d_y.get(), h_y0_ptr, y_bytes, cudaMemcpyHostToDevice), "H2D y0");
  } else {
    throw_on_cuda_error(cudaMemset(d_y.get(), 0, y_bytes), "cudaMemset(y=0)");
  }

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_epq_apply_weighted_many_atomic_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_p.get(),
      d_task_q.get(),
      d_task_wgt.get(),
      d_task_scale_dev ? d_task_scale_dev.get() : nullptr,
      ntasks,
      d_y.get(),
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(epq_atomic)");

  int h_overflow = 0;
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("epq_apply_weighted_many_atomic overflow (invalid indices or stack overflow)");
  }

  if (!return_y) return py::none();

  py::array_t<double> out_y({state.ncsf});
  throw_on_cuda_error(cudaMemcpy(out_y.mutable_data(), d_y.get(), y_bytes, cudaMemcpyDeviceToHost), "D2H y");
  return std::move(out_y);
}

void scatter_embed_inplace_device(
    py::object x_sub,
    py::object sub_to_full,
    py::object x_full,
    uint64_t stream,
    int threads) {
  CudaArrayView x_sub_dev = cuda_array_view_from_object(x_sub, "x_sub");
  CudaArrayView sub_to_full_dev = cuda_array_view_from_object(sub_to_full, "sub_to_full");
  CudaArrayView x_full_dev = cuda_array_view_from_object(x_full, "x_full");

  require_typestr(x_sub_dev, "x_sub", "<f8");
  require_typestr(sub_to_full_dev, "sub_to_full", "<i8");
  require_typestr(x_full_dev, "x_full", "<f8");

  if (x_full_dev.read_only) throw std::invalid_argument("x_full must be writable");

  int nsub = (int)x_sub_dev.shape[0];
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  launch_scatter_embed(
      (const double*)x_sub_dev.ptr,
      (const int64_t*)sub_to_full_dev.ptr,
      (double*)x_full_dev.ptr,
      nsub,
      threads,
      stream_t);
}

void gather_project_inplace_device(
    py::object y_full,
    py::object sub_to_full,
    py::object y_sub,
    uint64_t stream,
    int threads) {
  CudaArrayView y_full_dev = cuda_array_view_from_object(y_full, "y_full");
  CudaArrayView sub_to_full_dev = cuda_array_view_from_object(sub_to_full, "sub_to_full");
  CudaArrayView y_sub_dev = cuda_array_view_from_object(y_sub, "y_sub");

  require_typestr(y_full_dev, "y_full", "<f8");
  require_typestr(sub_to_full_dev, "sub_to_full", "<i8");
  require_typestr(y_sub_dev, "y_sub", "<f8");

  if (y_sub_dev.read_only) throw std::invalid_argument("y_sub must be writable");

  int nsub = (int)y_sub_dev.shape[0];
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  launch_gather_project(
      (const double*)y_full_dev.ptr,
      (const int64_t*)sub_to_full_dev.ptr,
      (double*)y_sub_dev.ptr,
      nsub,
      threads,
      stream_t);
}

void epq_apply_gather_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int p,
    int q,
    py::object x,
    py::object y,
    py::object overflow,
    double alpha,
    int threads,
    bool add,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 256) {
    throw std::invalid_argument("threads must be in 1..256 for epq_apply_gather");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument("threads must be a multiple of 32");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  if (x.is_none()) {
    throw std::invalid_argument("x must be a device array (cannot be None)");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  if ((unsigned)p >= (unsigned)drt.norb || (unsigned)q >= (unsigned)drt.norb) {
    throw std::invalid_argument("orbital indices out of range");
  }

  CudaArrayView x_dev = cuda_array_view_from_object(x, "x");
  CudaArrayView y_dev = cuda_array_view_from_object(y, "y");
  CudaArrayView overflow_dev = cuda_array_view_from_object(overflow, "overflow");

  require_typestr(x_dev, "x", "<f8");
  require_typestr(y_dev, "y", "<f8");
  require_typestr(overflow_dev, "overflow", "<i4");

  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }

  if (x_dev.shape.size() != 2 || y_dev.shape.size() != 2) {
    throw std::invalid_argument("x and y must be 2D arrays with shape (ncsf,nvec)");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }

  int64_t ncsf64 = x_dev.shape[0];
  int64_t nvec64 = x_dev.shape[1];
  if (ncsf64 < 0 || nvec64 < 0) {
    throw std::invalid_argument("invalid x shape");
  }
  if (y_dev.shape[0] != ncsf64 || y_dev.shape[1] != nvec64) {
    throw std::invalid_argument("y must have the same shape as x");
  }
  if (ncsf64 != (int64_t)state.ncsf) {
    throw std::invalid_argument("x/y ncsf must match DeviceStateCache.ncsf");
  }
  if (ncsf64 > (int64_t)std::numeric_limits<int>::max() || nvec64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ncsf/nvec too large for int32 kernel interface");
  }
  int ncsf = (int)ncsf64;
  int nvec = (int)nvec64;
  if (nvec <= 0) {
    throw std::invalid_argument("nvec must be >= 1");
  }
  if (nvec > 32) {
    throw std::invalid_argument("epq_apply_gather currently supports nvec<=32");
  }

  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 2) throw std::invalid_argument("x must be 2D with valid strides");
    if (x_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("x must be contiguous");
    if (x_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be C-contiguous");
    }
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 2) throw std::invalid_argument("y must be 2D with valid strides");
    if (y_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("y must be contiguous");
    if (y_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y must be C-contiguous");
    }
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  maybe_set_drt_access_policy_window(drt, stream_t);
  throw_on_cuda_error(cudaMemsetAsync(overflow_dev.ptr, 0, sizeof(int32_t), stream_t), "cudaMemsetAsync(overflow=0)");

  guga_epq_apply_gather_inplace_launch_stream(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      ncsf,
      drt.norb,
      p,
      q,
      (const double*)x_dev.ptr,
      nvec,
      alpha,
      (double*)y_dev.ptr,
      add ? 1 : 0,
      (int*)overflow_dev.ptr,
      stream_t,
      threads);

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_apply_gather)");
  }

  if (check_overflow) {
    int h_overflow = 0;
    throw_on_cuda_error(cudaMemcpy(&h_overflow, overflow_dev.ptr, sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
    if (h_overflow) {
      throw std::runtime_error("epq_apply_gather overflow (invalid indices or output list overflow)");
    }
  }
}

void epq_contribs_many_count_allpairs_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object counts,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  j_start = int(j_start);
  j_count = int(j_count);
  if (j_start < 0 || j_count < 0) {
    throw std::invalid_argument("j_start and j_count must be >= 0");
  }
  if (j_start > state.ncsf || j_start + j_count > state.ncsf) {
    throw std::invalid_argument("j_start/j_count out of range for DeviceStateCache.ncsf");
  }

  if (counts.is_none()) {
    throw std::invalid_argument("counts must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  int64_t n_pairs_ll = (int64_t)drt.norb * (int64_t)(drt.norb - 1);
  if (n_pairs_ll < 0 || n_pairs_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("n_pairs out of supported range");
  }
  int n_pairs = (int)n_pairs_ll;

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto counts_dev = cuda_array_view_from_object(counts, "counts");
  require_typestr(counts_dev, "counts", "<i4");
  if (counts_dev.read_only) {
    throw std::invalid_argument("counts must be writable");
  }
  if (counts_dev.shape.size() != 1 || counts_dev.shape[0] != (int64_t)ntasks) {
    throw std::invalid_argument("counts must have shape (j_count*n_pairs,)");
  }
  if (!counts_dev.strides_bytes.empty()) {
    if (counts_dev.strides_bytes.size() != 1 || counts_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("counts must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (counts_dev.stream) stream_u = counts_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  int32_t* d_counts = reinterpret_cast<int32_t*>(counts_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    guga_epq_contribs_many_count_allpairs_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        j_start,
        j_count,
        d_counts,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_count_allpairs)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(count_allpairs)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_count_allpairs)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_count_allpairs overflow (invalid indices or stack overflow)");
  }
}

void epq_contribs_many_count_tasks_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_p,
    py::object task_q,
    py::object counts,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (task_csf.is_none() || task_p.is_none() || task_q.is_none()) {
    throw std::invalid_argument("task_csf/task_p/task_q must be device arrays (cannot be None)");
  }
  if (counts.is_none()) {
    throw std::invalid_argument("counts must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  auto task_p_dev = cuda_array_view_from_object(task_p, "task_p");
  auto task_q_dev = cuda_array_view_from_object(task_q, "task_q");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  require_typestr(task_p_dev, "task_p", "<i4");
  require_typestr(task_q_dev, "task_q", "<i4");
  if (task_csf_dev.read_only || task_p_dev.read_only || task_q_dev.read_only) {
    throw std::invalid_argument("task arrays must be readable/writable device buffers");
  }
  if (task_csf_dev.shape.size() != 1 || task_p_dev.shape.size() != 1 || task_q_dev.shape.size() != 1) {
    throw std::invalid_argument("task arrays must be 1D");
  }
  if (task_csf_dev.shape[0] != task_p_dev.shape[0] || task_csf_dev.shape[0] != task_q_dev.shape[0]) {
    throw std::invalid_argument("task arrays must have the same shape");
  }
  if (!task_csf_dev.strides_bytes.empty() &&
      (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_csf must be contiguous");
  }
  if (!task_p_dev.strides_bytes.empty() &&
      (task_p_dev.strides_bytes.size() != 1 || task_p_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_p must be contiguous");
  }
  if (!task_q_dev.strides_bytes.empty() &&
      (task_q_dev.strides_bytes.size() != 1 || task_q_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_q must be contiguous");
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto counts_dev = cuda_array_view_from_object(counts, "counts");
  require_typestr(counts_dev, "counts", "<i4");
  if (counts_dev.read_only) {
    throw std::invalid_argument("counts must be writable");
  }
  if (counts_dev.shape.size() != 1 || counts_dev.shape[0] != ntasks_ll) {
    throw std::invalid_argument("counts must have shape (ntasks,)");
  }
  if (!counts_dev.strides_bytes.empty()) {
    if (counts_dev.strides_bytes.size() != 1 || counts_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("counts must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (counts_dev.stream) stream_u = counts_dev.stream;
    else if (task_q_dev.stream) stream_u = task_q_dev.stream;
    else if (task_p_dev.stream) stream_u = task_p_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const int32_t* d_task_p = reinterpret_cast<const int32_t*>(task_p_dev.ptr);
  const int32_t* d_task_q = reinterpret_cast<const int32_t*>(task_q_dev.ptr);
  int32_t* d_counts = reinterpret_cast<int32_t*>(counts_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    guga_epq_contribs_many_count_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        d_task_csf,
        d_task_p,
        d_task_q,
        ntasks,
        d_counts,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_count_tasks)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(count_tasks)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_count_tasks)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_count_tasks overflow (invalid indices or stack overflow)");
  }
}

void epq_contribs_many_write_tasks_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_p,
    py::object task_q,
    py::object offsets,
    py::object out_idx,
    py::object out_coeff,
    py::object out_task_csf,
    py::object out_task_pq,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (task_csf.is_none() || task_p.is_none() || task_q.is_none()) {
    throw std::invalid_argument("task_csf/task_p/task_q must be device arrays (cannot be None)");
  }
  if (offsets.is_none() || out_idx.is_none() || out_coeff.is_none() || overflow.is_none()) {
    throw std::invalid_argument("offsets/out_idx/out_coeff/overflow must be device arrays (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  auto task_p_dev = cuda_array_view_from_object(task_p, "task_p");
  auto task_q_dev = cuda_array_view_from_object(task_q, "task_q");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  require_typestr(task_p_dev, "task_p", "<i4");
  require_typestr(task_q_dev, "task_q", "<i4");
  if (task_csf_dev.shape.size() != 1 || task_p_dev.shape.size() != 1 || task_q_dev.shape.size() != 1) {
    throw std::invalid_argument("task arrays must be 1D");
  }
  if (task_csf_dev.shape[0] != task_p_dev.shape[0] || task_csf_dev.shape[0] != task_q_dev.shape[0]) {
    throw std::invalid_argument("task arrays must have the same shape");
  }
  if (!task_csf_dev.strides_bytes.empty() &&
      (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_csf must be contiguous");
  }
  if (!task_p_dev.strides_bytes.empty() &&
      (task_p_dev.strides_bytes.size() != 1 || task_p_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_p must be contiguous");
  }
  if (!task_q_dev.strides_bytes.empty() &&
      (task_q_dev.strides_bytes.size() != 1 || task_q_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_q must be contiguous");
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto offsets_dev = cuda_array_view_from_object(offsets, "offsets");
  require_typestr(offsets_dev, "offsets", "<i8");
  if (offsets_dev.read_only) {
    throw std::invalid_argument("offsets must be writable");
  }
  if (offsets_dev.shape.size() != 1 || offsets_dev.shape[0] != ntasks_ll + 1) {
    throw std::invalid_argument("offsets must have shape (ntasks+1,)");
  }
  if (!offsets_dev.strides_bytes.empty()) {
    if (offsets_dev.strides_bytes.size() != 1 || offsets_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("offsets must be contiguous");
    }
  }

  auto out_idx_dev = cuda_array_view_from_object(out_idx, "out_idx");
  require_typestr(out_idx_dev, "out_idx", "<i4");
  if (out_idx_dev.read_only) {
    throw std::invalid_argument("out_idx must be writable");
  }
  if (out_idx_dev.shape.size() != 1) {
    throw std::invalid_argument("out_idx must be a 1D device array");
  }
  if (!out_idx_dev.strides_bytes.empty()) {
    if (out_idx_dev.strides_bytes.size() != 1 || out_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("out_idx must be contiguous");
    }
  }

  auto out_coeff_dev = cuda_array_view_from_object(out_coeff, "out_coeff");
  require_typestr(out_coeff_dev, "out_coeff", "<f8");
  if (out_coeff_dev.read_only) {
    throw std::invalid_argument("out_coeff must be writable");
  }
  if (out_coeff_dev.shape.size() != 1 || out_coeff_dev.shape[0] != out_idx_dev.shape[0]) {
    throw std::invalid_argument("out_coeff must have shape (nnz,) matching out_idx");
  }
  if (!out_coeff_dev.strides_bytes.empty()) {
    if (out_coeff_dev.strides_bytes.size() != 1 || out_coeff_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out_coeff must be contiguous");
    }
  }

  int32_t* d_out_task_csf = nullptr;
  uint64_t out_task_csf_stream = 0;
  if (!out_task_csf.is_none()) {
    auto out_task_csf_dev = cuda_array_view_from_object(out_task_csf, "out_task_csf");
    require_typestr(out_task_csf_dev, "out_task_csf", "<i4");
    if (out_task_csf_dev.read_only) {
      throw std::invalid_argument("out_task_csf must be writable");
    }
    if (out_task_csf_dev.shape.size() != 1 || out_task_csf_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_csf must have shape (nnz,) matching out_idx");
    }
    if (!out_task_csf_dev.strides_bytes.empty()) {
      if (out_task_csf_dev.strides_bytes.size() != 1 || out_task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("out_task_csf must be contiguous");
      }
    }
    d_out_task_csf = reinterpret_cast<int32_t*>(out_task_csf_dev.ptr);
    out_task_csf_stream = out_task_csf_dev.stream;
  }

  int32_t* d_out_task_pq = nullptr;
  uint64_t out_task_pq_stream = 0;
  if (!out_task_pq.is_none()) {
    auto out_task_pq_dev = cuda_array_view_from_object(out_task_pq, "out_task_pq");
    require_typestr(out_task_pq_dev, "out_task_pq", "<i4");
    if (out_task_pq_dev.read_only) {
      throw std::invalid_argument("out_task_pq must be writable");
    }
    if (out_task_pq_dev.shape.size() != 1 || out_task_pq_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_pq must have shape (nnz,) matching out_idx");
    }
    if (!out_task_pq_dev.strides_bytes.empty()) {
      if (out_task_pq_dev.strides_bytes.size() != 1 || out_task_pq_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("out_task_pq must be contiguous");
      }
    }
    d_out_task_pq = reinterpret_cast<int32_t*>(out_task_pq_dev.ptr);
    out_task_pq_stream = out_task_pq_dev.stream;
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (out_coeff_dev.stream) stream_u = out_coeff_dev.stream;
    else if (out_idx_dev.stream) stream_u = out_idx_dev.stream;
    else if (offsets_dev.stream) stream_u = offsets_dev.stream;
    else if (out_task_pq_stream) stream_u = out_task_pq_stream;
    else if (out_task_csf_stream) stream_u = out_task_csf_stream;
    else if (task_q_dev.stream) stream_u = task_q_dev.stream;
    else if (task_p_dev.stream) stream_u = task_p_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const int32_t* d_task_p = reinterpret_cast<const int32_t*>(task_p_dev.ptr);
  const int32_t* d_task_q = reinterpret_cast<const int32_t*>(task_q_dev.ptr);
  const int64_t* d_offsets = reinterpret_cast<const int64_t*>(offsets_dev.ptr);
  int32_t* d_out_idx = reinterpret_cast<int32_t*>(out_idx_dev.ptr);
  double* d_out_coeff = reinterpret_cast<double*>(out_coeff_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    guga_epq_contribs_many_write_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        d_task_csf,
        d_task_p,
        d_task_q,
        ntasks,
        d_offsets,
        d_out_idx,
        d_out_coeff,
        d_out_task_csf,
        d_out_task_pq,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_write_tasks)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(write_tasks)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_write_tasks)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_write_tasks overflow (count/write mismatch or output overflow)");
  }
}

void epq_contribs_many_write_allpairs_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object offsets,
    py::object out_idx,
    py::object out_coeff,
    py::object out_task_csf,
    py::object out_task_pq,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  j_start = int(j_start);
  j_count = int(j_count);
  if (j_start < 0 || j_count < 0) {
    throw std::invalid_argument("j_start and j_count must be >= 0");
  }
  if (j_start > state.ncsf || j_start + j_count > state.ncsf) {
    throw std::invalid_argument("j_start/j_count out of range for DeviceStateCache.ncsf");
  }

  if (offsets.is_none()) {
    throw std::invalid_argument("offsets must be a device array (cannot be None)");
  }
  if (out_idx.is_none()) {
    throw std::invalid_argument("out_idx must be a device array (cannot be None)");
  }
  if (out_coeff.is_none()) {
    throw std::invalid_argument("out_coeff must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  int64_t n_pairs_ll = (int64_t)drt.norb * (int64_t)(drt.norb - 1);
  if (n_pairs_ll < 0 || n_pairs_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("n_pairs out of supported range");
  }
  int n_pairs = (int)n_pairs_ll;

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto offsets_dev = cuda_array_view_from_object(offsets, "offsets");
  require_typestr(offsets_dev, "offsets", "<i8");
  if (offsets_dev.read_only) {
    throw std::invalid_argument("offsets must be writable");
  }
  if (offsets_dev.shape.size() != 1 || offsets_dev.shape[0] != (int64_t)ntasks + 1) {
    throw std::invalid_argument("offsets must have shape (ntasks+1,)");
  }
  if (!offsets_dev.strides_bytes.empty()) {
    if (offsets_dev.strides_bytes.size() != 1 || offsets_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("offsets must be contiguous");
    }
  }

  auto out_idx_dev = cuda_array_view_from_object(out_idx, "out_idx");
  require_typestr(out_idx_dev, "out_idx", "<i4");
  if (out_idx_dev.read_only) {
    throw std::invalid_argument("out_idx must be writable");
  }
  if (out_idx_dev.shape.size() != 1) {
    throw std::invalid_argument("out_idx must be a 1D device array");
  }
  if (!out_idx_dev.strides_bytes.empty()) {
    if (out_idx_dev.strides_bytes.size() != 1 || out_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("out_idx must be contiguous");
    }
  }

  auto out_coeff_dev = cuda_array_view_from_object(out_coeff, "out_coeff");
  std::string out_coeff_typestr = normalize_typestr(out_coeff_dev.typestr);
  int out_coeff_type = 8;
  int64_t out_coeff_itemsize = 0;
  if (out_coeff_typestr == "<f8") {
    out_coeff_type = 8;
    out_coeff_itemsize = (int64_t)sizeof(double);
  } else if (out_coeff_typestr == "<f4") {
    out_coeff_type = 4;
    out_coeff_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("out_coeff must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (out_coeff_dev.read_only) {
    throw std::invalid_argument("out_coeff must be writable");
  }
  if (out_coeff_dev.shape.size() != 1 || out_coeff_dev.shape[0] != out_idx_dev.shape[0]) {
    throw std::invalid_argument("out_coeff must have shape (nnz,) matching out_idx");
  }
  if (!out_coeff_dev.strides_bytes.empty()) {
    if (out_coeff_dev.strides_bytes.size() != 1 || out_coeff_dev.strides_bytes[0] != out_coeff_itemsize) {
      throw std::invalid_argument("out_coeff must be contiguous");
    }
  }

  int32_t* d_out_task_csf = nullptr;
  uint64_t out_task_csf_stream = 0;
  if (!out_task_csf.is_none()) {
    auto out_task_csf_dev = cuda_array_view_from_object(out_task_csf, "out_task_csf");
    require_typestr(out_task_csf_dev, "out_task_csf", "<i4");
    if (out_task_csf_dev.read_only) {
      throw std::invalid_argument("out_task_csf must be writable");
    }
    if (out_task_csf_dev.shape.size() != 1 || out_task_csf_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_csf must have shape (nnz,) matching out_idx");
    }
    if (!out_task_csf_dev.strides_bytes.empty()) {
      if (out_task_csf_dev.strides_bytes.size() != 1 || out_task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("out_task_csf must be contiguous");
      }
    }
    d_out_task_csf = reinterpret_cast<int32_t*>(out_task_csf_dev.ptr);
    out_task_csf_stream = out_task_csf_dev.stream;
  }

  void* d_out_task_pq = nullptr;
  int out_task_pq_type = 4;
  uint64_t out_task_pq_stream = 0;
  if (!out_task_pq.is_none()) {
    auto out_task_pq_dev = cuda_array_view_from_object(out_task_pq, "out_task_pq");
    out_task_pq_type = epq_pq_type_from_typestr(out_task_pq_dev, "out_task_pq");
    int64_t pq_itemsize = epq_pq_itemsize_from_type(out_task_pq_type);
    if (out_task_pq_dev.read_only) {
      throw std::invalid_argument("out_task_pq must be writable");
    }
    if (out_task_pq_dev.shape.size() != 1 || out_task_pq_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_pq must have shape (nnz,) matching out_idx");
    }
    int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
    if (out_task_pq_type == 1 && nops_ll > 256) {
      throw std::invalid_argument("out_task_pq dtype <u1 is too small for current norb");
    }
    if (out_task_pq_type == 2 && nops_ll > 65535) {
      throw std::invalid_argument("out_task_pq dtype <u2 is too small for current norb");
    }
    if (!out_task_pq_dev.strides_bytes.empty()) {
      if (out_task_pq_dev.strides_bytes.size() != 1 || out_task_pq_dev.strides_bytes[0] != pq_itemsize) {
        throw std::invalid_argument("out_task_pq must be contiguous");
      }
    }
    d_out_task_pq = out_task_pq_dev.ptr;
    out_task_pq_stream = out_task_pq_dev.stream;
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (out_coeff_dev.stream) stream_u = out_coeff_dev.stream;
    else if (out_idx_dev.stream) stream_u = out_idx_dev.stream;
    else if (offsets_dev.stream) stream_u = offsets_dev.stream;
    else if (out_task_pq_stream) stream_u = out_task_pq_stream;
    else if (out_task_csf_stream) stream_u = out_task_csf_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int64_t* d_offsets = reinterpret_cast<const int64_t*>(offsets_dev.ptr);
  int32_t* d_out_idx = reinterpret_cast<int32_t*>(out_idx_dev.ptr);
  void* d_out_coeff = out_coeff_dev.ptr;
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    guga_epq_contribs_many_write_allpairs_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        j_start,
        j_count,
        d_offsets,
        d_out_idx,
        d_out_coeff,
        out_coeff_type,
        d_out_task_csf,
        d_out_task_pq,
        out_task_pq_type,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_write_allpairs)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(write_allpairs)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_write_allpairs)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_write_allpairs overflow (count/write mismatch or output overflow)");
  }
}

void epq_contribs_many_count_allpairs_recompute_inplace_device(
    const DeviceDRT& drt,
    int ncsf,
    int j_start,
    int j_count,
    py::object counts,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool warp_coop) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (warp_coop && (threads < 32 || (threads % 32) != 0)) {
    throw std::invalid_argument("warp_coop=True requires threads to be a multiple of 32 and >= 32");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  ncsf = int(ncsf);
  j_start = int(j_start);
  j_count = int(j_count);
  if (ncsf < 0 || j_start < 0 || j_count < 0) {
    throw std::invalid_argument("ncsf/j_start/j_count must be >= 0");
  }
  if (j_start > ncsf || j_start + j_count > ncsf) {
    throw std::invalid_argument("j_start/j_count out of range for ncsf");
  }

  if (counts.is_none()) {
    throw std::invalid_argument("counts must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  int64_t n_pairs_ll = (int64_t)drt.norb * (int64_t)(drt.norb - 1);
  if (n_pairs_ll < 0 || n_pairs_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("n_pairs out of supported range");
  }
  int n_pairs = (int)n_pairs_ll;

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto counts_dev = cuda_array_view_from_object(counts, "counts");
  require_typestr(counts_dev, "counts", "<i4");
  if (counts_dev.read_only) {
    throw std::invalid_argument("counts must be writable");
  }
  if (counts_dev.shape.size() != 1 || counts_dev.shape[0] != (int64_t)ntasks) {
    throw std::invalid_argument("counts must have shape (j_count*n_pairs,)");
  }
  if (!counts_dev.strides_bytes.empty()) {
    if (counts_dev.strides_bytes.size() != 1 || counts_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("counts must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (counts_dev.stream) stream_u = counts_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  int32_t* d_counts = reinterpret_cast<int32_t*>(counts_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    if (warp_coop) {
      guga_epq_contribs_many_count_allpairs_recompute_warp_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          ncsf,
          drt.norb,
          j_start,
          j_count,
          d_counts,
          d_overflow,
          stream_t,
          threads);
    } else {
      guga_epq_contribs_many_count_allpairs_recompute_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          ncsf,
          drt.norb,
          j_start,
          j_count,
          d_counts,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_count_allpairs_recompute)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(count_allpairs_recompute)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_count_allpairs_recompute)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_count_allpairs_recompute overflow");
  }
}

void epq_contribs_many_write_allpairs_recompute_inplace_device(
    const DeviceDRT& drt,
    int ncsf,
    int j_start,
    int j_count,
    py::object offsets,
    py::object out_idx,
    py::object out_coeff,
    py::object out_task_csf,
    py::object out_task_pq,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool warp_coop) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (warp_coop && (threads < 32 || (threads % 32) != 0)) {
    throw std::invalid_argument("warp_coop=True requires threads to be a multiple of 32 and >= 32");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  ncsf = int(ncsf);
  j_start = int(j_start);
  j_count = int(j_count);
  if (ncsf < 0 || j_start < 0 || j_count < 0) {
    throw std::invalid_argument("ncsf/j_start/j_count must be >= 0");
  }
  if (j_start > ncsf || j_start + j_count > ncsf) {
    throw std::invalid_argument("j_start/j_count out of range for ncsf");
  }

  if (offsets.is_none()) {
    throw std::invalid_argument("offsets must be a device array (cannot be None)");
  }
  if (out_idx.is_none()) {
    throw std::invalid_argument("out_idx must be a device array (cannot be None)");
  }
  if (out_coeff.is_none()) {
    throw std::invalid_argument("out_coeff must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  int64_t n_pairs_ll = (int64_t)drt.norb * (int64_t)(drt.norb - 1);
  if (n_pairs_ll < 0 || n_pairs_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("n_pairs out of supported range");
  }
  int n_pairs = (int)n_pairs_ll;

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range for current kernels (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto offsets_dev = cuda_array_view_from_object(offsets, "offsets");
  require_typestr(offsets_dev, "offsets", "<i8");
  if (offsets_dev.read_only) {
    throw std::invalid_argument("offsets must be writable");
  }
  if (offsets_dev.shape.size() != 1 || offsets_dev.shape[0] != (int64_t)ntasks + 1) {
    throw std::invalid_argument("offsets must have shape (ntasks+1,)");
  }
  if (!offsets_dev.strides_bytes.empty()) {
    if (offsets_dev.strides_bytes.size() != 1 || offsets_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("offsets must be contiguous");
    }
  }

  auto out_idx_dev = cuda_array_view_from_object(out_idx, "out_idx");
  require_typestr(out_idx_dev, "out_idx", "<i4");
  if (out_idx_dev.read_only) {
    throw std::invalid_argument("out_idx must be writable");
  }
  if (out_idx_dev.shape.size() != 1) {
    throw std::invalid_argument("out_idx must be a 1D device array");
  }
  if (!out_idx_dev.strides_bytes.empty()) {
    if (out_idx_dev.strides_bytes.size() != 1 || out_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("out_idx must be contiguous");
    }
  }

  auto out_coeff_dev = cuda_array_view_from_object(out_coeff, "out_coeff");
  std::string out_coeff_typestr = normalize_typestr(out_coeff_dev.typestr);
  int out_coeff_type = 8;
  int64_t out_coeff_itemsize = 0;
  if (out_coeff_typestr == "<f8") {
    out_coeff_type = 8;
    out_coeff_itemsize = (int64_t)sizeof(double);
  } else if (out_coeff_typestr == "<f4") {
    out_coeff_type = 4;
    out_coeff_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("out_coeff must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (out_coeff_dev.read_only) {
    throw std::invalid_argument("out_coeff must be writable");
  }
  if (out_coeff_dev.shape.size() != 1 || out_coeff_dev.shape[0] != out_idx_dev.shape[0]) {
    throw std::invalid_argument("out_coeff must have shape (nnz,) matching out_idx");
  }
  if (!out_coeff_dev.strides_bytes.empty()) {
    if (out_coeff_dev.strides_bytes.size() != 1 || out_coeff_dev.strides_bytes[0] != out_coeff_itemsize) {
      throw std::invalid_argument("out_coeff must be contiguous");
    }
  }

  int32_t* d_out_task_csf = nullptr;
  uint64_t out_task_csf_stream = 0;
  if (!out_task_csf.is_none()) {
    auto out_task_csf_dev = cuda_array_view_from_object(out_task_csf, "out_task_csf");
    require_typestr(out_task_csf_dev, "out_task_csf", "<i4");
    if (out_task_csf_dev.read_only) {
      throw std::invalid_argument("out_task_csf must be writable");
    }
    if (out_task_csf_dev.shape.size() != 1 || out_task_csf_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_csf must have shape (nnz,) matching out_idx");
    }
    if (!out_task_csf_dev.strides_bytes.empty()) {
      if (out_task_csf_dev.strides_bytes.size() != 1 || out_task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("out_task_csf must be contiguous");
      }
    }
    d_out_task_csf = reinterpret_cast<int32_t*>(out_task_csf_dev.ptr);
    out_task_csf_stream = out_task_csf_dev.stream;
  }

  void* d_out_task_pq = nullptr;
  int out_task_pq_type = 4;
  uint64_t out_task_pq_stream = 0;
  if (!out_task_pq.is_none()) {
    auto out_task_pq_dev = cuda_array_view_from_object(out_task_pq, "out_task_pq");
    out_task_pq_type = epq_pq_type_from_typestr(out_task_pq_dev, "out_task_pq");
    int64_t pq_itemsize = epq_pq_itemsize_from_type(out_task_pq_type);
    if (out_task_pq_dev.read_only) {
      throw std::invalid_argument("out_task_pq must be writable");
    }
    if (out_task_pq_dev.shape.size() != 1 || out_task_pq_dev.shape[0] != out_idx_dev.shape[0]) {
      throw std::invalid_argument("out_task_pq must have shape (nnz,) matching out_idx");
    }
    int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
    if (out_task_pq_type == 1 && nops_ll > 256) {
      throw std::invalid_argument("out_task_pq dtype <u1 is too small for current norb");
    }
    if (out_task_pq_type == 2 && nops_ll > 65535) {
      throw std::invalid_argument("out_task_pq dtype <u2 is too small for current norb");
    }
    if (!out_task_pq_dev.strides_bytes.empty()) {
      if (out_task_pq_dev.strides_bytes.size() != 1 || out_task_pq_dev.strides_bytes[0] != pq_itemsize) {
        throw std::invalid_argument("out_task_pq must be contiguous");
      }
    }
    d_out_task_pq = out_task_pq_dev.ptr;
    out_task_pq_stream = out_task_pq_dev.stream;
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (out_coeff_dev.stream) stream_u = out_coeff_dev.stream;
    else if (out_idx_dev.stream) stream_u = out_idx_dev.stream;
    else if (offsets_dev.stream) stream_u = offsets_dev.stream;
    else if (out_task_pq_stream) stream_u = out_task_pq_stream;
    else if (out_task_csf_stream) stream_u = out_task_csf_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int64_t* d_offsets = reinterpret_cast<const int64_t*>(offsets_dev.ptr);
  int32_t* d_out_idx = reinterpret_cast<int32_t*>(out_idx_dev.ptr);
  void* d_out_coeff = out_coeff_dev.ptr;
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (ntasks > 0) {
    if (warp_coop) {
      guga_epq_contribs_many_write_allpairs_recompute_warp_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          ncsf,
          drt.norb,
          j_start,
          j_count,
          d_offsets,
          d_out_idx,
          d_out_coeff,
          out_coeff_type,
          d_out_task_csf,
          d_out_task_pq,
          out_task_pq_type,
          d_overflow,
          stream_t,
          threads);
    } else {
      guga_epq_contribs_many_write_allpairs_recompute_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          ncsf,
          drt.norb,
          j_start,
          j_count,
          d_offsets,
          d_out_idx,
          d_out_coeff,
          out_coeff_type,
          d_out_task_csf,
          d_out_task_pq,
          out_task_pq_type,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(epq_write_allpairs_recompute)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow(write_allpairs_recompute)");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(epq_write_allpairs_recompute)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("epq_contribs_many_write_allpairs_recompute overflow");
  }
}

void apply_g_flat_scatter_atomic_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_g,
    py::object task_scale,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  if (task_csf_dev.shape.size() != 1) {
    throw std::invalid_argument("task_csf must be 1D device array");
  }
  if (!task_csf_dev.strides_bytes.empty()) {
    if (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("task_csf must be contiguous");
    }
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("task_csf too large");
  }
  int ntasks = (int)ntasks_ll;

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  std::string task_g_typestr = task_g_dev.typestr;
  if (!task_g_typestr.empty() && task_g_typestr[0] == '=') task_g_typestr[0] = '<';
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (task_g_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (task_g_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("task_g must have typestr <f8 (float64) or <f4 (float32)");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (nops_ll <= 0) {
    throw std::invalid_argument("invalid nops");
  }

  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;
    if (!task_g_dev.strides_bytes.empty()) {
      if (task_g_dev.strides_bytes.size() != 1 || task_g_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_g (1D) must be contiguous");
      }
    }
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[0] != ntasks_ll || task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (ntasks,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have length 2");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize) {
        throw std::invalid_argument("task_g (2D) row stride too small");
      }
      if (s0 % fp_itemsize != 0) {
        throw std::invalid_argument("task_g (2D) row stride must be a multiple of itemsize");
      }
      g_stride = s0 / fp_itemsize;
    }
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D device array");
  }

  uint64_t task_scale_stream = 0;
  const double* d_task_scale_f64 = nullptr;
  const float* d_task_scale_f32 = nullptr;
  if (!task_scale.is_none()) {
    auto task_scale_dev = cuda_array_view_from_object(task_scale, "task_scale");
    if (use_f32) require_typestr(task_scale_dev, "task_scale", "<f4");
    else require_typestr(task_scale_dev, "task_scale", "<f8");
    task_scale_stream = task_scale_dev.stream;
    if (task_scale_dev.shape.size() != 1 || task_scale_dev.shape[0] != ntasks_ll) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    if (!task_scale_dev.strides_bytes.empty()) {
      if (task_scale_dev.strides_bytes.size() != 1 || task_scale_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_scale must be contiguous");
      }
    }
    if (use_f32) d_task_scale_f32 = reinterpret_cast<const float*>(task_scale_dev.ptr);
    else d_task_scale_f64 = reinterpret_cast<const double*>(task_scale_dev.ptr);
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }
  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (task_scale_stream) stream_u = task_scale_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const double* d_task_g_f64 = nullptr;
  const float* d_task_g_f32 = nullptr;
  if (use_f32) d_task_g_f32 = reinterpret_cast<const float*>(task_g_dev.ptr);
  else d_task_g_f64 = reinterpret_cast<const double*>(task_g_dev.ptr);
  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize;
    void* d_y_void = use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (ntasks > 0) {
    if (use_f32) {
      guga_apply_g_flat_scatter_atomic_f32_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          d_task_csf,
          d_task_scale_f32,
          d_task_g_f32,
          g_stride,
          ntasks,
          d_y_f32,
          d_overflow,
          stream_t,
          threads);
    } else {
      guga_apply_g_flat_scatter_atomic_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          d_task_csf,
          d_task_scale_f64,
          d_task_g_f64,
          g_stride,
          ntasks,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_scatter_atomic)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(apply_g_flat_scatter_atomic)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_scatter_atomic overflow (invalid indices or stack overflow)");
  }
}

// =============================================================================
// 10.16.3 / 10.18: Warp-Cooperative Kernel Wrapper
// =============================================================================
// Forward declare the warp-cooperative launch functions from guga_cuda_kernels_apply_g_flat.cuh
extern "C" void guga_apply_g_flat_scatter_atomic_warp_coop_launch_stream(
    const int32_t* child, const int16_t* node_twos, const int64_t* child_prefix,
    const int8_t* steps_table, const int32_t* nodes_table, int ncsf, int norb,
    const int32_t* task_csf, const double* task_scale, const double* task_g,
    int64_t g_stride, int ntasks, double* y, int* overflow_flag, cudaStream_t stream, int threads);

extern "C" void guga_apply_g_flat_scatter_atomic_warp_coop_f32_launch_stream(
    const int32_t* child, const int16_t* node_twos, const int64_t* child_prefix,
    const int8_t* steps_table, const int32_t* nodes_table, int ncsf, int norb,
    const int32_t* task_csf, const float* task_scale, const float* task_g,
    int64_t g_stride, int ntasks, float* y, int* overflow_flag, cudaStream_t stream, int threads);

void apply_g_flat_scatter_atomic_warp_coop_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_g,
    py::object task_scale,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  // Validate inputs (same as apply_g_flat_scatter_atomic_inplace_device)
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  if (task_csf_dev.shape.size() != 1) {
    throw std::invalid_argument("task_csf must be 1D device array");
  }
  if (!task_csf_dev.strides_bytes.empty()) {
    if (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("task_csf must be contiguous");
    }
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("task_csf too large");
  }
  int ntasks = (int)ntasks_ll;

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  std::string task_g_typestr = task_g_dev.typestr;
  if (!task_g_typestr.empty() && task_g_typestr[0] == '=') task_g_typestr[0] = '<';
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (task_g_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (task_g_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("task_g must have typestr <f8 (float64) or <f4 (float32)");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (nops_ll <= 0) {
    throw std::invalid_argument("invalid nops");
  }

  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;
    if (!task_g_dev.strides_bytes.empty()) {
      if (task_g_dev.strides_bytes.size() != 1 || task_g_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_g (1D) must be contiguous");
      }
    }
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[0] != ntasks_ll || task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (ntasks,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have length 2");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize) {
        throw std::invalid_argument("task_g (2D) row stride too small");
      }
      if (s0 % fp_itemsize != 0) {
        throw std::invalid_argument("task_g (2D) row stride must be a multiple of itemsize");
      }
      g_stride = s0 / fp_itemsize;
    }
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D device array");
  }

  uint64_t task_scale_stream = 0;
  const double* d_task_scale_f64 = nullptr;
  const float* d_task_scale_f32 = nullptr;
  if (!task_scale.is_none()) {
    auto task_scale_dev = cuda_array_view_from_object(task_scale, "task_scale");
    if (use_f32) require_typestr(task_scale_dev, "task_scale", "<f4");
    else require_typestr(task_scale_dev, "task_scale", "<f8");
    task_scale_stream = task_scale_dev.stream;
    if (task_scale_dev.shape.size() != 1 || task_scale_dev.shape[0] != ntasks_ll) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    if (!task_scale_dev.strides_bytes.empty()) {
      if (task_scale_dev.strides_bytes.size() != 1 || task_scale_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_scale must be contiguous");
      }
    }
    if (use_f32) d_task_scale_f32 = reinterpret_cast<const float*>(task_scale_dev.ptr);
    else d_task_scale_f64 = reinterpret_cast<const double*>(task_scale_dev.ptr);
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }
  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (task_scale_stream) stream_u = task_scale_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const double* d_task_g_f64 = nullptr;
  const float* d_task_g_f32 = nullptr;
  if (use_f32) d_task_g_f32 = reinterpret_cast<const float*>(task_g_dev.ptr);
  else d_task_g_f64 = reinterpret_cast<const double*>(task_g_dev.ptr);
  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize;
    void* d_y_void = use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (ntasks > 0) {
    if (use_f32) {
      guga_apply_g_flat_scatter_atomic_warp_coop_f32_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          d_task_csf,
          d_task_scale_f32,
          d_task_g_f32,
          g_stride,
          ntasks,
          d_y_f32,
          d_overflow,
          stream_t,
          threads);
    } else {
      guga_apply_g_flat_scatter_atomic_warp_coop_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          d_task_csf,
          d_task_scale_f64,
          d_task_g_f64,
          g_stride,
          ntasks,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_scatter_atomic_warp_coop)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(apply_g_flat_scatter_atomic_warp_coop)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_scatter_atomic_warp_coop overflow (invalid indices or stack overflow)");
  }
}

void apply_g_flat_scatter_atomic_epq_table_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object epq_data,
    py::object task_csf,
    py::object task_g,
    py::object task_scale,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  int epq_indptr_type = epq_indptr_type_from_typestr(epq_indptr_dev, "epq_indptr");
  int64_t epq_indptr_itemsize = epq_indptr_itemsize_from_type(epq_indptr_type);
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != epq_indptr_itemsize) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D device array");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  int epq_pq_type = epq_pq_type_from_typestr(epq_pq_dev, "epq_pq");
  int64_t epq_pq_itemsize = epq_pq_itemsize_from_type(epq_pq_type);
  if (epq_pq_dev.shape.size() != 1 || epq_pq_dev.shape[0] != epq_indices_dev.shape[0]) {
    throw std::invalid_argument("epq_pq must have shape (nnz,) and match epq_indices");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (epq_pq_type == 1 && nops_ll > 256) {
    throw std::invalid_argument("epq_pq dtype <u1 is too small for current norb");
  }
  if (epq_pq_type == 2 && nops_ll > 65535) {
    throw std::invalid_argument("epq_pq dtype <u2 is too small for current norb");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != epq_pq_itemsize) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto epq_data_dev = cuda_array_view_from_object(epq_data, "epq_data");
  std::string epq_typestr = epq_data_dev.typestr;
  if (!epq_typestr.empty() && epq_typestr[0] == '=') epq_typestr[0] = '<';
  bool epq_use_f32 = false;
  int64_t epq_itemsize = 0;
  if (epq_typestr == "<f8") {
    epq_use_f32 = false;
    epq_itemsize = (int64_t)sizeof(double);
  } else if (epq_typestr == "<f4") {
    epq_use_f32 = true;
    epq_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_data_dev.shape.size() != 1 || epq_data_dev.shape[0] != epq_indices_dev.shape[0]) {
    throw std::invalid_argument("epq_data must have shape (nnz,) and match epq_indices");
  }
  if (!epq_data_dev.strides_bytes.empty()) {
    if (epq_data_dev.strides_bytes.size() != 1 || epq_data_dev.strides_bytes[0] != epq_itemsize) {
      throw std::invalid_argument("epq_data must be contiguous");
    }
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  if (task_csf_dev.shape.size() != 1) {
    throw std::invalid_argument("task_csf must be 1D device array");
  }
  if (!task_csf_dev.strides_bytes.empty()) {
    if (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("task_csf must be contiguous");
    }
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("task_csf too large");
  }
  int ntasks = (int)ntasks_ll;

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  // Determine output type from task_g (must match y).
  std::string g_typestr = task_g_dev.typestr;
  if (!g_typestr.empty() && g_typestr[0] == '=') g_typestr[0] = '<';
  bool out_use_f32 = false;
  int64_t fp_itemsize = 0;
  if (g_typestr == "<f8") {
    out_use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (g_typestr == "<f4") {
    out_use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("task_g must have typestr <f8 (float64) or <f4 (float32)");
  }
  // Reject f32 output with f64 coefficients (no use case).
  if (out_use_f32 && !epq_use_f32) {
    throw std::invalid_argument("Cannot use float32 output with float64 epq_data");
  }

  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;
    if (!task_g_dev.strides_bytes.empty()) {
      if (task_g_dev.strides_bytes.size() != 1 || task_g_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_g (1D) must be contiguous");
      }
    }
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[0] != ntasks_ll || task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (ntasks,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have length 2");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize) {
        throw std::invalid_argument("task_g (2D) row stride too small");
      }
      if (s0 % fp_itemsize != 0) {
        throw std::invalid_argument("task_g (2D) row stride must be a multiple of itemsize");
      }
      g_stride = s0 / fp_itemsize;
    }
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D device array");
  }

  uint64_t task_scale_stream = 0;
  const double* d_task_scale_f64 = nullptr;
  const float* d_task_scale_f32 = nullptr;
  if (!task_scale.is_none()) {
    auto task_scale_dev = cuda_array_view_from_object(task_scale, "task_scale");
    if (out_use_f32) require_typestr(task_scale_dev, "task_scale", "<f4");
    else require_typestr(task_scale_dev, "task_scale", "<f8");
    task_scale_stream = task_scale_dev.stream;
    if (task_scale_dev.shape.size() != 1 || task_scale_dev.shape[0] != ntasks_ll) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    if (!task_scale_dev.strides_bytes.empty()) {
      if (task_scale_dev.strides_bytes.size() != 1 || task_scale_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_scale must be contiguous");
      }
    }
    if (out_use_f32) d_task_scale_f32 = reinterpret_cast<const float*>(task_scale_dev.ptr);
    else d_task_scale_f64 = reinterpret_cast<const double*>(task_scale_dev.ptr);
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (out_use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (task_scale_stream) stream_u = task_scale_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    else if (epq_data_dev.stream) stream_u = epq_data_dev.stream;
    else if (epq_indices_dev.stream) stream_u = epq_indices_dev.stream;
    else if (epq_pq_dev.stream) stream_u = epq_pq_dev.stream;
    else if (epq_indptr_dev.stream) stream_u = epq_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const void* d_epq_indptr = epq_indptr_dev.ptr;
  const int32_t* d_epq_indices = reinterpret_cast<const int32_t*>(epq_indices_dev.ptr);
  const void* d_epq_pq = epq_pq_dev.ptr;
  const double* d_epq_data_f64 = nullptr;
  const float* d_epq_data_f32 = nullptr;
  if (epq_use_f32) d_epq_data_f32 = reinterpret_cast<const float*>(epq_data_dev.ptr);
  else d_epq_data_f64 = reinterpret_cast<const double*>(epq_data_dev.ptr);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const double* d_task_g_f64 = nullptr;
  const float* d_task_g_f32 = nullptr;
  if (out_use_f32) d_task_g_f32 = reinterpret_cast<const float*>(task_g_dev.ptr);
  else d_task_g_f64 = reinterpret_cast<const double*>(task_g_dev.ptr);

  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (out_use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize;
    void* d_y_void = out_use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (ntasks > 0) {
    if (out_use_f32 && epq_use_f32) {
      // Pure FP32 path
      guga_apply_g_flat_scatter_atomic_epq_table_f32_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_indptr,
          epq_indptr_type,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f32,
          d_task_csf,
          d_task_scale_f32,
          d_task_g_f32,
          g_stride,
          ntasks,
          d_y_f32,
          d_overflow,
          stream_t,
          threads);
    } else if (!out_use_f32 && epq_use_f32) {
      // Mixed path: FP64 output, FP32 coefficients
      guga_apply_g_flat_scatter_atomic_epq_table_mixed_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_indptr,
          epq_indptr_type,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f32,
          d_task_csf,
          d_task_scale_f64,
          d_task_g_f64,
          g_stride,
          ntasks,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    } else {
      // Pure FP64 path
      guga_apply_g_flat_scatter_atomic_epq_table_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_indptr,
          epq_indptr_type,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f64,
          d_task_csf,
          d_task_scale_f64,
          d_task_g_f64,
          g_stride,
          ntasks,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_scatter_atomic_epq_table)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(apply_g_flat_scatter_atomic_epq_table)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_scatter_atomic_epq_table overflow (invalid indices)");
  }
}

void apply_g_flat_scatter_atomic_epq_table_tile_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object local_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object epq_data,
    py::object task_g,
    py::object task_scale,
    int j_start,
    int j_count,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool use_kahan) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 256) {
    throw std::invalid_argument("threads must be in 1..256 for epq-table tile kernel");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (j_start < 0 || j_count < 0 || j_start + j_count > state.ncsf) {
    throw std::invalid_argument("j_start/j_count out of range");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto local_indptr_dev = cuda_array_view_from_object(local_indptr, "local_indptr");
  require_typestr(local_indptr_dev, "local_indptr", "<i8");
  if (local_indptr_dev.shape.size() != 1 || local_indptr_dev.shape[0] != (int64_t)j_count + 1) {
    throw std::invalid_argument("local_indptr must have shape (j_count+1,)");
  }
  if (!local_indptr_dev.strides_bytes.empty()) {
    if (local_indptr_dev.strides_bytes.size() != 1 ||
        local_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("local_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be a 1D device array");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 ||
        epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  int epq_pq_type = epq_pq_type_from_typestr(epq_pq_dev, "epq_pq");
  int64_t epq_pq_itemsize = epq_pq_itemsize_from_type(epq_pq_type);
  if (epq_pq_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_pq must have shape (nnz,) and match epq_indices");
  }
  int64_t nops_ll_cap = (int64_t)drt.norb * (int64_t)drt.norb;
  if (epq_pq_type == 1 && nops_ll_cap > 256) {
    throw std::invalid_argument("epq_pq dtype <u1 is too small for current norb");
  }
  if (epq_pq_type == 2 && nops_ll_cap > 65535) {
    throw std::invalid_argument("epq_pq dtype <u2 is too small for current norb");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != epq_pq_itemsize) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto epq_data_dev = cuda_array_view_from_object(epq_data, "epq_data");
  std::string epq_typestr = epq_data_dev.typestr;
  if (!epq_typestr.empty() && epq_typestr[0] == '=') epq_typestr[0] = '<';
  bool epq_use_f32 = false;
  int64_t fp_itemsize = 0;
  if (epq_typestr == "<f8") {
    epq_use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (epq_typestr == "<f4") {
    epq_use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_data_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_data must have shape (nnz,) and match epq_indices");
  }
  if (!epq_data_dev.strides_bytes.empty()) {
    if (epq_data_dev.strides_bytes.size() != 1 || epq_data_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("epq_data must be contiguous");
    }
  }

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  std::string g_typestr_t = task_g_dev.typestr;
  if (!g_typestr_t.empty() && g_typestr_t[0] == '=') g_typestr_t[0] = '<';
  bool out_use_f32 = (g_typestr_t == "<f4");
  int64_t fp_itemsize_out = out_use_f32 ? (int64_t)sizeof(float) : (int64_t)sizeof(double);
  if (g_typestr_t != "<f4" && g_typestr_t != "<f8") {
    throw std::invalid_argument("task_g must have typestr <f8 or <f4");
  }
  if (out_use_f32 && !epq_use_f32) {
    throw std::invalid_argument("Cannot use float32 output with float64 epq_data");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;
    if (!task_g_dev.strides_bytes.empty()) {
      if (task_g_dev.strides_bytes.size() != 1 || task_g_dev.strides_bytes[0] != fp_itemsize_out) {
        throw std::invalid_argument("task_g (1D) must be contiguous");
      }
    }
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[0] != (int64_t)j_count || task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (j_count,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have length 2");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != fp_itemsize_out) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize_out || (s0 % fp_itemsize_out != 0)) {
        throw std::invalid_argument("task_g (2D) row stride invalid");
      }
      g_stride = s0 / fp_itemsize_out;
    }
  } else {
    throw std::invalid_argument("task_g must be a 1D or 2D device array");
  }

  uint64_t task_scale_stream = 0;
  const double* d_task_scale_f64 = nullptr;
  const float* d_task_scale_f32 = nullptr;
  if (!task_scale.is_none()) {
    auto task_scale_dev = cuda_array_view_from_object(task_scale, "task_scale");
    if (out_use_f32) require_typestr(task_scale_dev, "task_scale", "<f4");
    else require_typestr(task_scale_dev, "task_scale", "<f8");
    task_scale_stream = task_scale_dev.stream;
    if (task_scale_dev.shape.size() != 1 || task_scale_dev.shape[0] != (int64_t)j_count) {
      throw std::invalid_argument("task_scale must have shape (j_count,) when provided");
    }
    if (!task_scale_dev.strides_bytes.empty()) {
      if (task_scale_dev.strides_bytes.size() != 1 || task_scale_dev.strides_bytes[0] != fp_itemsize_out) {
        throw std::invalid_argument("task_scale must be contiguous");
      }
    }
    if (out_use_f32) d_task_scale_f32 = reinterpret_cast<const float*>(task_scale_dev.ptr);
    else d_task_scale_f64 = reinterpret_cast<const double*>(task_scale_dev.ptr);
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (out_use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize_out) {
      throw std::invalid_argument("y must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_scale_stream) stream_u = task_scale_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    else if (epq_data_dev.stream) stream_u = epq_data_dev.stream;
    else if (epq_indices_dev.stream) stream_u = epq_indices_dev.stream;
    else if (epq_pq_dev.stream) stream_u = epq_pq_dev.stream;
    else if (local_indptr_dev.stream) stream_u = local_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);
  maybe_set_drt_access_policy_window(drt, stream_t);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const int64_t* d_local_indptr = reinterpret_cast<const int64_t*>(local_indptr_dev.ptr);
  const int32_t* d_epq_indices = reinterpret_cast<const int32_t*>(epq_indices_dev.ptr);
  const void* d_epq_pq = epq_pq_dev.ptr;
  const double* d_epq_data_f64 = nullptr;
  const float* d_epq_data_f32 = nullptr;
  if (epq_use_f32) d_epq_data_f32 = reinterpret_cast<const float*>(epq_data_dev.ptr);
  else d_epq_data_f64 = reinterpret_cast<const double*>(epq_data_dev.ptr);

  const double* d_task_g_f64 = nullptr;
  const float* d_task_g_f32 = nullptr;
  if (out_use_f32) d_task_g_f32 = reinterpret_cast<const float*>(task_g_dev.ptr);
  else d_task_g_f64 = reinterpret_cast<const double*>(task_g_dev.ptr);

  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (out_use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize_out;
    void* d_y_void = out_use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (j_count > 0) {
    if (out_use_f32 && epq_use_f32) {
      // Pure FP32 path
      if (use_kahan) {
        guga_apply_g_flat_scatter_atomic_epq_table_tile_f32_kahan_launch_stream(
            d_steps,
            state.ncsf,
            drt.norb,
            d_local_indptr,
            d_epq_indices,
            d_epq_pq,
            epq_pq_type,
            d_epq_data_f32,
            d_task_g_f32,
            g_stride,
            d_task_scale_f32,
            j_start,
            j_count,
            d_y_f32,
            d_overflow,
            stream_t,
            threads);
      } else {
        guga_apply_g_flat_scatter_atomic_epq_table_tile_f32_launch_stream(
            d_steps,
            state.ncsf,
            drt.norb,
            d_local_indptr,
            d_epq_indices,
            d_epq_pq,
            epq_pq_type,
            d_epq_data_f32,
            d_task_g_f32,
            g_stride,
            d_task_scale_f32,
            j_start,
            j_count,
            d_y_f32,
            d_overflow,
            stream_t,
            threads);
      }
    } else if (!out_use_f32 && epq_use_f32) {
      // Mixed path: FP64 output, FP32 coefficients
      guga_apply_g_flat_scatter_atomic_epq_table_tile_mixed_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_local_indptr,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f32,
          d_task_g_f64,
          g_stride,
          d_task_scale_f64,
          j_start,
          j_count,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    } else {
      // Pure FP64 path
      guga_apply_g_flat_scatter_atomic_epq_table_tile_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_local_indptr,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f64,
          d_task_g_f64,
          g_stride,
          d_task_scale_f64,
          j_start,
          j_count,
          d_y_f64,
          d_overflow,
          stream_t,
          threads);
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_scatter_atomic_epq_table_tile)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(
        cudaStreamSynchronize(stream_t),
        "cudaStreamSynchronize(apply_g_flat_scatter_atomic_epq_table_tile)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_scatter_atomic_epq_table_tile overflow (invalid indices)");
  }
}

void apply_g_flat_gather_epq_table_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_t_indptr,
    py::object epq_t_source,
    py::object epq_t_pq,
    py::object epq_t_data,
    py::object task_row_by_csf,
    py::object task_scale_by_csf,
    py::object task_g,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool use_kahan) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 256) {
    throw std::invalid_argument("threads must be in 1..256 for gather epq-table kernel");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto epq_t_indptr_dev = cuda_array_view_from_object(epq_t_indptr, "epq_t_indptr");
  int epq_t_indptr_type = epq_indptr_type_from_typestr(epq_t_indptr_dev, "epq_t_indptr");
  int64_t epq_t_indptr_itemsize = epq_indptr_itemsize_from_type(epq_t_indptr_type);
  if (epq_t_indptr_dev.shape.size() != 1 || epq_t_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_t_indptr must have shape (ncsf+1,)");
  }
  if (!epq_t_indptr_dev.strides_bytes.empty()) {
    if (epq_t_indptr_dev.strides_bytes.size() != 1 || epq_t_indptr_dev.strides_bytes[0] != epq_t_indptr_itemsize) {
      throw std::invalid_argument("epq_t_indptr must be contiguous");
    }
  }

  auto epq_t_source_dev = cuda_array_view_from_object(epq_t_source, "epq_t_source");
  require_typestr(epq_t_source_dev, "epq_t_source", "<i4");
  if (epq_t_source_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_t_source must be 1D device array");
  }
  if (!epq_t_source_dev.strides_bytes.empty()) {
    if (epq_t_source_dev.strides_bytes.size() != 1 || epq_t_source_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_t_source must be contiguous");
    }
  }

  auto epq_t_pq_dev = cuda_array_view_from_object(epq_t_pq, "epq_t_pq");
  int epq_t_pq_type = epq_pq_type_from_typestr(epq_t_pq_dev, "epq_t_pq");
  int64_t epq_t_pq_itemsize = epq_pq_itemsize_from_type(epq_t_pq_type);
  if (epq_t_pq_dev.shape.size() != 1 || epq_t_pq_dev.shape[0] != epq_t_source_dev.shape[0]) {
    throw std::invalid_argument("epq_t_pq must have shape (nnz,) and match epq_t_source");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (epq_t_pq_type == 1 && nops_ll > 256) {
    throw std::invalid_argument("epq_t_pq dtype <u1 is too small for current norb");
  }
  if (epq_t_pq_type == 2 && nops_ll > 65535) {
    throw std::invalid_argument("epq_t_pq dtype <u2 is too small for current norb");
  }
  if (!epq_t_pq_dev.strides_bytes.empty()) {
    if (epq_t_pq_dev.strides_bytes.size() != 1 || epq_t_pq_dev.strides_bytes[0] != epq_t_pq_itemsize) {
      throw std::invalid_argument("epq_t_pq must be contiguous");
    }
  }

  auto epq_t_data_dev = cuda_array_view_from_object(epq_t_data, "epq_t_data");
  std::string epq_t_typestr = normalize_typestr(epq_t_data_dev.typestr);
  bool epq_data_is_f32 = false;
  int64_t epq_data_itemsize = 0;
  if (epq_t_typestr == "<f8") {
    epq_data_is_f32 = false;
    epq_data_itemsize = (int64_t)sizeof(double);
  } else if (epq_t_typestr == "<f4") {
    epq_data_is_f32 = true;
    epq_data_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_t_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_t_data_dev.shape.size() != 1 || epq_t_data_dev.shape[0] != epq_t_source_dev.shape[0]) {
    throw std::invalid_argument("epq_t_data must have shape (nnz,) and match epq_t_source");
  }
  if (!epq_t_data_dev.strides_bytes.empty()) {
    if (epq_t_data_dev.strides_bytes.size() != 1 || epq_t_data_dev.strides_bytes[0] != epq_data_itemsize) {
      throw std::invalid_argument("epq_t_data must be contiguous");
    }
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  std::string y_typestr = normalize_typestr(y_dev.typestr);
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (y_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (y_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("y must have typestr <f8 (float64) or <f4 (float32)");
  }

  auto task_row_by_csf_dev = cuda_array_view_from_object(task_row_by_csf, "task_row_by_csf");
  require_typestr(task_row_by_csf_dev, "task_row_by_csf", "<i4");
  if (task_row_by_csf_dev.shape.size() != 1 || task_row_by_csf_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("task_row_by_csf must have shape (ncsf,)");
  }
  if (!task_row_by_csf_dev.strides_bytes.empty()) {
    if (task_row_by_csf_dev.strides_bytes.size() != 1 ||
        task_row_by_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("task_row_by_csf must be contiguous");
    }
  }

  const double* d_task_scale_by_csf_f64 = nullptr;
  const float* d_task_scale_by_csf_f32 = nullptr;
  uint64_t task_scale_by_csf_stream = 0;
  if (!task_scale_by_csf.is_none()) {
    auto task_scale_by_csf_dev = cuda_array_view_from_object(task_scale_by_csf, "task_scale_by_csf");
    if (use_f32) require_typestr(task_scale_by_csf_dev, "task_scale_by_csf", "<f4");
    else require_typestr(task_scale_by_csf_dev, "task_scale_by_csf", "<f8");
    if (task_scale_by_csf_dev.shape.size() != 1 || task_scale_by_csf_dev.shape[0] != (int64_t)state.ncsf) {
      throw std::invalid_argument("task_scale_by_csf must have shape (ncsf,)");
    }
    if (!task_scale_by_csf_dev.strides_bytes.empty()) {
      if (task_scale_by_csf_dev.strides_bytes.size() != 1 || task_scale_by_csf_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_scale_by_csf must be contiguous");
      }
    }
    if (use_f32) d_task_scale_by_csf_f32 = reinterpret_cast<const float*>(task_scale_by_csf_dev.ptr);
    else d_task_scale_by_csf_f64 = reinterpret_cast<const double*>(task_scale_by_csf_dev.ptr);
    task_scale_by_csf_stream = task_scale_by_csf_dev.stream;
  }

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  if (use_f32) require_typestr(task_g_dev, "task_g", "<f4");
  else require_typestr(task_g_dev, "task_g", "<f8");
  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;
    if (!task_g_dev.strides_bytes.empty()) {
      if (task_g_dev.strides_bytes.size() != 1 || task_g_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("task_g (1D) must be contiguous");
      }
    }
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (ntasks,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have length 2");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize || (s0 % fp_itemsize != 0)) {
        throw std::invalid_argument("task_g (2D) row stride invalid");
      }
      g_stride = s0 / fp_itemsize;
    }
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D device array");
  }
  if (use_f32 && !epq_data_is_f32) {
    throw std::invalid_argument("epq_t_data must be float32 when task_g/y are float32");
  }

  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_row_by_csf_dev.stream) stream_u = task_row_by_csf_dev.stream;
    else if (task_scale_by_csf_stream) stream_u = task_scale_by_csf_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    else if (epq_t_data_dev.stream) stream_u = epq_t_data_dev.stream;
    else if (epq_t_pq_dev.stream) stream_u = epq_t_pq_dev.stream;
    else if (epq_t_source_dev.stream) stream_u = epq_t_source_dev.stream;
    else if (epq_t_indptr_dev.stream) stream_u = epq_t_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const void* d_epq_t_indptr = epq_t_indptr_dev.ptr;
  const int32_t* d_epq_t_source = reinterpret_cast<const int32_t*>(epq_t_source_dev.ptr);
  const void* d_epq_t_pq = epq_t_pq_dev.ptr;
  const double* d_epq_t_data_f64 = nullptr;
  const float* d_epq_t_data_f32 = nullptr;
  if (epq_data_is_f32) d_epq_t_data_f32 = reinterpret_cast<const float*>(epq_t_data_dev.ptr);
  else d_epq_t_data_f64 = reinterpret_cast<const double*>(epq_t_data_dev.ptr);

  const int32_t* d_task_row_by_csf = reinterpret_cast<const int32_t*>(task_row_by_csf_dev.ptr);
  const double* d_task_g_f64 = nullptr;
  const float* d_task_g_f32 = nullptr;
  if (use_f32) d_task_g_f32 = reinterpret_cast<const float*>(task_g_dev.ptr);
  else d_task_g_f64 = reinterpret_cast<const double*>(task_g_dev.ptr);

  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize;
    void* d_y_void = use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (use_f32) {
    // The plain FP32 gather kernel is currently more robust across workloads than the
    // experimental Kahan path; keep `use_kahan` accepted at the API boundary but route
    // both modes through the stable kernel for now.
    (void)use_kahan;
    guga_apply_g_flat_gather_epq_table_f32_launch_stream(
        d_steps,
        state.ncsf,
        drt.norb,
        d_epq_t_indptr,
        epq_t_indptr_type,
        d_epq_t_source,
        d_epq_t_pq,
        epq_t_pq_type,
        d_epq_t_data_f32,
        d_task_row_by_csf,
        d_task_scale_by_csf_f32,
        d_task_g_f32,
        g_stride,
        d_y_f32,
        d_overflow,
        stream_t,
        threads,
        zero_y ? 0 : 1);
  } else {
    if (epq_data_is_f32) {
      guga_apply_g_flat_gather_epq_table_mixed_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f32,
          d_task_row_by_csf,
          d_task_scale_by_csf_f64,
          d_task_g_f64,
          g_stride,
          d_y_f64,
          d_overflow,
          stream_t,
          threads,
          zero_y ? 0 : 1);
    } else {
      guga_apply_g_flat_gather_epq_table_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f64,
          d_task_row_by_csf,
          d_task_scale_by_csf_f64,
          d_task_g_f64,
          g_stride,
          d_y_f64,
          d_overflow,
          stream_t,
          threads,
          zero_y ? 0 : 1);
    }
  }
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_gather_epq_table)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(apply_g_flat_gather_epq_table)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_gather_epq_table overflow (invalid indices)");
  }
}

void apply_g_flat_gather_epq_transpose_range_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_t_indptr,
    py::object epq_t_source,
    py::object epq_t_pq,
    py::object epq_t_data,
    py::object g_block,
    int k_start,
    int k_count,
    py::object y,
    py::object overflow,
    int threads,
    bool add,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool use_kahan) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 256) {
    throw std::invalid_argument("threads must be in 1..256 for gather epq-transpose range kernel");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (k_start < 0 || k_count <= 0 || k_start + k_count > state.ncsf) {
    throw std::invalid_argument("k_start/k_count out of range");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto epq_t_indptr_dev = cuda_array_view_from_object(epq_t_indptr, "epq_t_indptr");
  int epq_t_indptr_type = epq_indptr_type_from_typestr(epq_t_indptr_dev, "epq_t_indptr");
  int64_t epq_t_indptr_itemsize = epq_indptr_itemsize_from_type(epq_t_indptr_type);
  if (epq_t_indptr_dev.shape.size() != 1 || epq_t_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_t_indptr must have shape (ncsf+1,)");
  }
  if (!epq_t_indptr_dev.strides_bytes.empty()) {
    if (epq_t_indptr_dev.strides_bytes.size() != 1 ||
        epq_t_indptr_dev.strides_bytes[0] != epq_t_indptr_itemsize) {
      throw std::invalid_argument("epq_t_indptr must be contiguous");
    }
  }

  auto epq_t_source_dev = cuda_array_view_from_object(epq_t_source, "epq_t_source");
  require_typestr(epq_t_source_dev, "epq_t_source", "<i4");
  if (epq_t_source_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_t_source must be a 1D device array");
  }
  if (!epq_t_source_dev.strides_bytes.empty()) {
    if (epq_t_source_dev.strides_bytes.size() != 1 ||
        epq_t_source_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_t_source must be contiguous");
    }
  }

  auto epq_t_pq_dev = cuda_array_view_from_object(epq_t_pq, "epq_t_pq");
  int epq_t_pq_type = epq_pq_type_from_typestr(epq_t_pq_dev, "epq_t_pq");
  int64_t epq_t_pq_itemsize = epq_pq_itemsize_from_type(epq_t_pq_type);
  if (epq_t_pq_dev.shape != epq_t_source_dev.shape) {
    throw std::invalid_argument("epq_t_pq must have shape (nnz,) and match epq_t_source");
  }
  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (epq_t_pq_type == 1 && nops_ll > 256) {
    throw std::invalid_argument("epq_t_pq dtype <u1 is too small for current norb");
  }
  if (epq_t_pq_type == 2 && nops_ll > 65535) {
    throw std::invalid_argument("epq_t_pq dtype <u2 is too small for current norb");
  }
  if (!epq_t_pq_dev.strides_bytes.empty()) {
    if (epq_t_pq_dev.strides_bytes.size() != 1 || epq_t_pq_dev.strides_bytes[0] != epq_t_pq_itemsize) {
      throw std::invalid_argument("epq_t_pq must be contiguous");
    }
  }

  auto epq_t_data_dev = cuda_array_view_from_object(epq_t_data, "epq_t_data");
  std::string epq_t_typestr = normalize_typestr(epq_t_data_dev.typestr);
  bool epq_data_is_f32 = false;
  int64_t epq_data_itemsize = 0;
  if (epq_t_typestr == "<f8") {
    epq_data_is_f32 = false;
    epq_data_itemsize = (int64_t)sizeof(double);
  } else if (epq_t_typestr == "<f4") {
    epq_data_is_f32 = true;
    epq_data_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_t_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_t_data_dev.shape != epq_t_source_dev.shape) {
    throw std::invalid_argument("epq_t_data must have shape (nnz,) and match epq_t_source");
  }
  if (!epq_t_data_dev.strides_bytes.empty()) {
    if (epq_t_data_dev.strides_bytes.size() != 1 || epq_t_data_dev.strides_bytes[0] != epq_data_itemsize) {
      throw std::invalid_argument("epq_t_data must be contiguous");
    }
  }

  auto g_block_dev = cuda_array_view_from_object(g_block, "g_block");
  std::string g_typestr = normalize_typestr(g_block_dev.typestr);
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (g_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (g_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("g_block must have typestr <f8 (float64) or <f4 (float32)");
  }
  int64_t g_stride = 0;
  if (g_block_dev.shape.size() == 1) {
    if (g_block_dev.shape[0] != (int64_t)k_count * nops_ll) {
      throw std::invalid_argument("g_block (1D) must have shape (k_count*nops,)");
    }
    if (!g_block_dev.strides_bytes.empty()) {
      if (g_block_dev.strides_bytes.size() != 1 || g_block_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("g_block (1D) must be contiguous");
      }
    }
    g_stride = nops_ll;
  } else if (g_block_dev.shape.size() == 2) {
    if (g_block_dev.shape[0] != (int64_t)k_count || g_block_dev.shape[1] < nops_ll) {
      throw std::invalid_argument("g_block (2D) must have shape (k_count,nops_or_more)");
    }
    if (g_block_dev.strides_bytes.empty()) {
      g_stride = g_block_dev.shape[1];
    } else {
      if (g_block_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_block (2D) strides must have length 2");
      }
      int64_t s0 = g_block_dev.strides_bytes[0];
      int64_t s1 = g_block_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("g_block (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("g_block (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * fp_itemsize || (s0 % fp_itemsize != 0)) {
        throw std::invalid_argument("g_block (2D) row stride invalid");
      }
      g_stride = s0 / fp_itemsize;
    }
  } else {
    throw std::invalid_argument("g_block must be a 1D or 2D device array");
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }
  if (use_f32 && !epq_data_is_f32) {
    throw std::invalid_argument("epq_t_data must be float32 when g_block/y are float32");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (g_block_dev.stream) stream_u = g_block_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
    else if (epq_t_data_dev.stream) stream_u = epq_t_data_dev.stream;
    else if (epq_t_pq_dev.stream) stream_u = epq_t_pq_dev.stream;
    else if (epq_t_source_dev.stream) stream_u = epq_t_source_dev.stream;
    else if (epq_t_indptr_dev.stream) stream_u = epq_t_indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const void* d_epq_t_indptr = epq_t_indptr_dev.ptr;
  const int32_t* d_epq_t_source = reinterpret_cast<const int32_t*>(epq_t_source_dev.ptr);
  const void* d_epq_t_pq = epq_t_pq_dev.ptr;
  const double* d_epq_t_data_f64 = nullptr;
  const float* d_epq_t_data_f32 = nullptr;
  if (epq_data_is_f32) d_epq_t_data_f32 = reinterpret_cast<const float*>(epq_t_data_dev.ptr);
  else d_epq_t_data_f64 = reinterpret_cast<const double*>(epq_t_data_dev.ptr);

  const double* d_g_block_f64 = nullptr;
  const float* d_g_block_f32 = nullptr;
  if (use_f32) d_g_block_f32 = reinterpret_cast<const float*>(g_block_dev.ptr);
  else d_g_block_f64 = reinterpret_cast<const double*>(g_block_dev.ptr);

  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (use_f32) {
    if (use_kahan) {
      guga_apply_g_flat_gather_epq_transpose_range_f32_kahan_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f32,
          d_g_block_f32,
          g_stride,
          k_start,
          k_count,
          d_y_f32,
          d_overflow,
          stream_t,
          threads,
          add ? 1 : 0);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_f32_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f32,
          d_g_block_f32,
          g_stride,
          k_start,
          k_count,
          d_y_f32,
          d_overflow,
          stream_t,
          threads,
          add ? 1 : 0);
    }
  } else {
    if (epq_data_is_f32) {
      guga_apply_g_flat_gather_epq_transpose_range_mixed_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f32,
          d_g_block_f64,
          g_stride,
          k_start,
          k_count,
          d_y_f64,
          d_overflow,
          stream_t,
          threads,
          add ? 1 : 0);
    } else {
      guga_apply_g_flat_gather_epq_transpose_range_launch_stream(
          d_steps,
          state.ncsf,
          drt.norb,
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f64,
          d_g_block_f64,
          g_stride,
          k_start,
          k_count,
          d_y_f64,
          d_overflow,
          stream_t,
          threads,
          add ? 1 : 0);
    }
  }
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_gather_epq_transpose_range)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(
        cudaStreamSynchronize(stream_t),
        "cudaStreamSynchronize(apply_g_flat_gather_epq_transpose_range)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_gather_epq_transpose_range overflow (invalid indices)");
  }
}

void kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object epq_data,
    py::object row_j,
    py::object row_k,
    py::object csr_indptr,
    py::object csr_indices,
    py::object csr_data,
    int row_start,
    int nrows,
    py::object eri_mat_t,
    py::object x,
    py::object y,
    py::object overflow,
    int threads,
    bool zero_y,
    double half,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    bool use_kahan) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (row_start < 0 || nrows < 0) {
    throw std::invalid_argument("row_start/nrows must be >= 0");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (y.is_none()) {
    throw std::invalid_argument("y must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  require_typestr(epq_indptr_dev, "epq_indptr", "<i8");
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D device array");
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  int epq_pq_type = epq_pq_type_from_typestr(epq_pq_dev, "epq_pq");
  int64_t epq_pq_itemsize = epq_pq_itemsize_from_type(epq_pq_type);
  if (epq_pq_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_pq must have the same shape as epq_indices");
  }
  int64_t nops_ll_cap = (int64_t)state.norb * (int64_t)state.norb;
  if (epq_pq_type == 1 && nops_ll_cap > 256) {
    throw std::invalid_argument("epq_pq dtype <u1 is too small for current norb");
  }
  if (epq_pq_type == 2 && nops_ll_cap > 65535) {
    throw std::invalid_argument("epq_pq dtype <u2 is too small for current norb");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != epq_pq_itemsize) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto epq_data_dev = cuda_array_view_from_object(epq_data, "epq_data");
  std::string epq_data_typestr = normalize_typestr(epq_data_dev.typestr);
  bool epq_data_is_f32 = false;
  int64_t epq_data_itemsize = 0;
  if (epq_data_typestr == "<f8") {
    epq_data_is_f32 = false;
    epq_data_itemsize = (int64_t)sizeof(double);
  } else if (epq_data_typestr == "<f4") {
    epq_data_is_f32 = true;
    epq_data_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_data_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_data must have the same shape as epq_indices");
  }
  if (!epq_data_dev.strides_bytes.empty()) {
    if (epq_data_dev.strides_bytes.size() != 1 || epq_data_dev.strides_bytes[0] != epq_data_itemsize) {
      throw std::invalid_argument("epq_data must be contiguous");
    }
  }

  auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
  require_typestr(row_j_dev, "row_j", "<i4");
  if (row_j_dev.shape.size() != 1) {
    throw std::invalid_argument("row_j must be 1D device array");
  }
  if (!row_j_dev.strides_bytes.empty()) {
    if (row_j_dev.strides_bytes.size() != 1 || row_j_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("row_j must be contiguous");
    }
  }

  auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
  require_typestr(row_k_dev, "row_k", "<i4");
  if (row_k_dev.shape != row_j_dev.shape) {
    throw std::invalid_argument("row_k must have the same shape as row_j");
  }
  if (!row_k_dev.strides_bytes.empty()) {
    if (row_k_dev.strides_bytes.size() != 1 || row_k_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("row_k must be contiguous");
    }
  }

  int64_t nrows_total = row_j_dev.shape[0];
  if ((int64_t)row_start + (int64_t)nrows > nrows_total) {
    throw std::invalid_argument("row_start+nrows exceeds row_j length");
  }

  auto csr_indptr_dev = cuda_array_view_from_object(csr_indptr, "csr_indptr");
  require_typestr(csr_indptr_dev, "csr_indptr", "<i8");
  if (csr_indptr_dev.shape.size() != 1 || csr_indptr_dev.shape[0] != nrows_total + 1) {
    throw std::invalid_argument("csr_indptr must have shape (nrows_total+1,)");
  }
  if (!csr_indptr_dev.strides_bytes.empty()) {
    if (csr_indptr_dev.strides_bytes.size() != 1 || csr_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("csr_indptr must be contiguous");
    }
  }

  auto csr_indices_dev = cuda_array_view_from_object(csr_indices, "csr_indices");
  require_typestr(csr_indices_dev, "csr_indices", "<i4");
  if (csr_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("csr_indices must be 1D device array");
  }
  if (!csr_indices_dev.strides_bytes.empty()) {
    if (csr_indices_dev.strides_bytes.size() != 1 || csr_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("csr_indices must be contiguous");
    }
  }

  auto csr_data_dev = cuda_array_view_from_object(csr_data, "csr_data");
  std::string csr_data_typestr = normalize_typestr(csr_data_dev.typestr);
  bool csr_data_is_f32 = false;
  int64_t csr_data_itemsize = 0;
  if (csr_data_typestr == "<f8") {
    csr_data_is_f32 = false;
    csr_data_itemsize = (int64_t)sizeof(double);
  } else if (csr_data_typestr == "<f4") {
    csr_data_is_f32 = true;
    csr_data_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("csr_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (csr_data_dev.shape != csr_indices_dev.shape) {
    throw std::invalid_argument("csr_data must have the same shape as csr_indices");
  }
  if (!csr_data_dev.strides_bytes.empty()) {
    if (csr_data_dev.strides_bytes.size() != 1 || csr_data_dev.strides_bytes[0] != csr_data_itemsize) {
      throw std::invalid_argument("csr_data must be contiguous");
    }
  }

  auto eri_mat_t_dev = cuda_array_view_from_object(eri_mat_t, "eri_mat_t");
  std::string eri_mat_t_typestr = normalize_typestr(eri_mat_t_dev.typestr);
  bool eri_mat_t_is_f32 = false;
  int64_t eri_mat_t_itemsize = 0;
  if (eri_mat_t_typestr == "<f8") {
    eri_mat_t_is_f32 = false;
    eri_mat_t_itemsize = (int64_t)sizeof(double);
  } else if (eri_mat_t_typestr == "<f4") {
    eri_mat_t_is_f32 = true;
    eri_mat_t_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("eri_mat_t must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (eri_mat_t_dev.read_only) {
    // read-only is fine for inputs, but the view helper marks read-only if the exporting object says so; accept it.
  }
  if (eri_mat_t_dev.shape.size() != 2) {
    throw std::invalid_argument("eri_mat_t must be a 2D device array");
  }

  int nops = drt.norb * drt.norb;
  if (nops <= 0) {
    throw std::invalid_argument("invalid nops");
  }
  if ((int)eri_mat_t_dev.shape[0] != nops || (int)eri_mat_t_dev.shape[1] != nops) {
    throw std::invalid_argument("eri_mat_t must have shape (norb*norb, norb*norb)");
  }
  if (!eri_mat_t_dev.strides_bytes.empty()) {
    if (eri_mat_t_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("eri_mat_t strides must have length 2");
    }
    int64_t s0 = eri_mat_t_dev.strides_bytes[0];
    int64_t s1 = eri_mat_t_dev.strides_bytes[1];
    if (s0 != (int64_t)nops * eri_mat_t_itemsize || s1 != eri_mat_t_itemsize) {
      throw std::invalid_argument("eri_mat_t must be contiguous (C-order)");
    }
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  std::string x_typestr = normalize_typestr(x_dev.typestr);
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (x_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (x_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("x must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("x must be contiguous");
    }
  }
  if (use_f32 && !epq_data_is_f32) {
    throw std::invalid_argument("epq_data must be float32 when x/y are float32");
  }
  if (use_f32 != eri_mat_t_is_f32) {
    throw std::invalid_argument("eri_mat_t dtype must match x dtype");
  }
  if (use_f32 != csr_data_is_f32) {
    throw std::invalid_argument("csr_data dtype must match x dtype (use float32 csr_data for float32 mode)");
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("y must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (row_j_dev.stream) stream_u = row_j_dev.stream;
    else if (csr_data_dev.stream) stream_u = csr_data_dev.stream;
    else if (epq_data_dev.stream) stream_u = epq_data_dev.stream;
    else if (eri_mat_t_dev.stream) stream_u = eri_mat_t_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  double* d_y_f64 = nullptr;
  float* d_y_f32 = nullptr;
  if (use_f32) d_y_f32 = reinterpret_cast<float*>(y_dev.ptr);
  else d_y_f64 = reinterpret_cast<double*>(y_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_y) {
    size_t y_bytes = (size_t)state.ncsf * (size_t)fp_itemsize;
    void* d_y_void = use_f32 ? static_cast<void*>(d_y_f32) : static_cast<void*>(d_y_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_y_void, 0, y_bytes, stream_t), "cudaMemsetAsync(y=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (nrows > 0) {
    const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
    const int64_t* d_epq_indptr = reinterpret_cast<const int64_t*>(epq_indptr_dev.ptr);
    const int32_t* d_epq_indices = reinterpret_cast<const int32_t*>(epq_indices_dev.ptr);
    const void* d_epq_pq = epq_pq_dev.ptr;
    const int32_t* d_row_j = reinterpret_cast<const int32_t*>(row_j_dev.ptr);
    const int32_t* d_row_k = reinterpret_cast<const int32_t*>(row_k_dev.ptr);
    const int64_t* d_csr_indptr = reinterpret_cast<const int64_t*>(csr_indptr_dev.ptr);
    const int32_t* d_csr_indices = reinterpret_cast<const int32_t*>(csr_indices_dev.ptr);
    if (use_f32) {
      const float* d_epq_data = reinterpret_cast<const float*>(epq_data_dev.ptr);
      const float* d_eri_mat_t = reinterpret_cast<const float*>(eri_mat_t_dev.ptr);
      const float* d_x = reinterpret_cast<const float*>(x_dev.ptr);
      const float* d_csr_data = reinterpret_cast<const float*>(csr_data_dev.ptr);
      if (use_kahan) {
        guga_apply_csr_eri_mat_fused_epq_table_range_f32_kahan_launch_stream(
            d_steps, state.ncsf, drt.norb,
            d_epq_indptr, d_epq_indices, d_epq_pq, epq_pq_type, d_epq_data,
            d_row_j, d_row_k, d_csr_indptr, d_csr_indices, d_csr_data,
            int(row_start), int(nrows), d_eri_mat_t, int(nops),
            float(half), d_x, d_y_f32, d_overflow, stream_t, threads);
      } else {
        guga_apply_csr_eri_mat_fused_epq_table_range_f32_launch_stream(
            d_steps, state.ncsf, drt.norb,
            d_epq_indptr, d_epq_indices, d_epq_pq, epq_pq_type, d_epq_data,
            d_row_j, d_row_k, d_csr_indptr, d_csr_indices, d_csr_data,
            int(row_start), int(nrows), d_eri_mat_t, int(nops),
            float(half), d_x, d_y_f32, d_overflow, stream_t, threads);
      }
    } else {
      const float* d_epq_data_f32 = nullptr;
      const double* d_epq_data_f64 = nullptr;
      if (epq_data_is_f32) d_epq_data_f32 = reinterpret_cast<const float*>(epq_data_dev.ptr);
      else d_epq_data_f64 = reinterpret_cast<const double*>(epq_data_dev.ptr);
      const double* d_eri_mat_t = reinterpret_cast<const double*>(eri_mat_t_dev.ptr);
      const double* d_x = reinterpret_cast<const double*>(x_dev.ptr);
      const double* d_csr_data = reinterpret_cast<const double*>(csr_data_dev.ptr);
      if (epq_data_is_f32) {
        guga_apply_csr_eri_mat_fused_epq_table_range_f64_out_f32_coeff_launch_stream(
            d_steps,
            state.ncsf,
            drt.norb,
            d_epq_indptr,
            d_epq_indices,
            d_epq_pq,
            epq_pq_type,
            d_epq_data_f32,
            d_row_j,
            d_row_k,
            d_csr_indptr,
            d_csr_indices,
            d_csr_data,
            int(row_start),
            int(nrows),
            d_eri_mat_t,
            int(nops),
            double(half),
            d_x,
            d_y_f64,
            d_overflow,
            stream_t,
            threads);
      } else {
        guga_apply_csr_eri_mat_fused_epq_table_range_launch_stream(
            d_steps,
            state.ncsf,
            drt.norb,
            d_epq_indptr,
            d_epq_indices,
            d_epq_pq,
            epq_pq_type,
            d_epq_data_f64,
            d_row_j,
            d_row_k,
            d_csr_indptr,
            d_csr_indices,
            d_csr_data,
            int(row_start),
            int(nrows),
            d_eri_mat_t,
            int(nops),
            double(half),
            d_x,
            d_y_f64,
            d_overflow,
            stream_t,
            threads);
      }
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel4_apply_csr_eri_mat_fused_epq_table_range)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel4_apply_csr_eri_mat_fused_epq_table_range)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("kernel4_apply_csr_eri_mat_fused_epq_table_range overflow (invalid indices)");
  }
}

void kernel4_build_w_from_csr_unitnnz_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object row_j,
    py::object row_k,
    py::object csr_rs,
    py::object csr_c,
    py::object x,
    py::object w_out,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }
  if (w_out.is_none()) {
    throw std::invalid_argument("w_out must be a device array (cannot be None)");
  }

  auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
  require_typestr(row_j_dev, "row_j", "<i4");
  if (row_j_dev.shape.size() != 1) {
    throw std::invalid_argument("row_j must be 1D device array");
  }
  if (!row_j_dev.strides_bytes.empty()) {
    if (row_j_dev.strides_bytes.size() != 1 || row_j_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("row_j must be contiguous");
    }
  }

  auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
  require_typestr(row_k_dev, "row_k", "<i4");
  if (row_k_dev.shape != row_j_dev.shape) {
    throw std::invalid_argument("row_k must have the same shape as row_j");
  }
  if (!row_k_dev.strides_bytes.empty()) {
    if (row_k_dev.strides_bytes.size() != 1 || row_k_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("row_k must be contiguous");
    }
  }

  auto csr_rs_dev = cuda_array_view_from_object(csr_rs, "csr_rs");
  require_typestr(csr_rs_dev, "csr_rs", "<i4");
  if (csr_rs_dev.shape != row_j_dev.shape) {
    throw std::invalid_argument("csr_rs must have shape (nrows,) matching row_j");
  }
  if (!csr_rs_dev.strides_bytes.empty()) {
    if (csr_rs_dev.strides_bytes.size() != 1 || csr_rs_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("csr_rs must be contiguous");
    }
  }

  auto csr_c_dev = cuda_array_view_from_object(csr_c, "csr_c");
  require_typestr(csr_c_dev, "csr_c", "<f8");
  if (csr_c_dev.shape != row_j_dev.shape) {
    throw std::invalid_argument("csr_c must have shape (nrows,) matching row_j");
  }
  if (!csr_c_dev.strides_bytes.empty()) {
    if (csr_c_dev.strides_bytes.size() != 1 || csr_c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("csr_c must be contiguous");
    }
  }

  int64_t nrows_ll = row_j_dev.shape[0];
  if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("row_j too large");
  }
  int nrows = (int)nrows_ll;

  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  if (nops_ll <= 0) {
    throw std::invalid_argument("invalid nops");
  }
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops too large");
  }
  int nops = (int)nops_ll;

  auto x_dev = cuda_array_view_from_object(x, "x");
  require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  auto w_out_dev = cuda_array_view_from_object(w_out, "w_out");
  require_typestr(w_out_dev, "w_out", "<f8");
  if (w_out_dev.read_only) {
    throw std::invalid_argument("w_out must be writable");
  }

  int64_t w_stride = 0;
  if (w_out_dev.shape.size() == 1) {
    if (w_out_dev.shape[0] != (int64_t)state.ncsf * nops_ll) {
      throw std::invalid_argument("w_out (1D) must have shape (ncsf*nops,)");
    }
    if (!w_out_dev.strides_bytes.empty()) {
      if (w_out_dev.strides_bytes.size() != 1 || w_out_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("w_out (1D) must be contiguous");
      }
    }
    w_stride = nops_ll;
  } else if (w_out_dev.shape.size() == 2) {
    if (w_out_dev.shape[0] != (int64_t)state.ncsf) {
      throw std::invalid_argument("w_out (2D) must have shape (ncsf, nops_or_more)");
    }
    if (w_out_dev.shape[1] < nops_ll) {
      throw std::invalid_argument("w_out (2D) second dimension too small");
    }
    if (w_out_dev.strides_bytes.empty()) {
      w_stride = w_out_dev.shape[1];
    } else {
      if (w_out_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_out (2D) strides must have length 2");
      }
      int64_t s0 = w_out_dev.strides_bytes[0];
      int64_t s1 = w_out_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("w_out (2D) must have positive strides");
      }
      if (s1 != (int64_t)sizeof(double)) {
        throw std::invalid_argument("w_out (2D) must be C-contiguous along last dimension");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("w_out (2D) row stride must be a multiple of itemsize");
      }
      w_stride = s0 / (int64_t)sizeof(double);
    }
    if (w_stride < nops_ll) {
      throw std::invalid_argument("w_out row stride too small");
    }
  } else {
    throw std::invalid_argument("w_out must be 1D or 2D device array");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (w_out_dev.stream) stream_u = w_out_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (row_j_dev.stream) stream_u = row_j_dev.stream;
    else if (csr_c_dev.stream) stream_u = csr_c_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int32_t* d_row_j = reinterpret_cast<const int32_t*>(row_j_dev.ptr);
  const int32_t* d_row_k = reinterpret_cast<const int32_t*>(row_k_dev.ptr);
  const int32_t* d_csr_rs = reinterpret_cast<const int32_t*>(csr_rs_dev.ptr);
  const double* d_csr_c = reinterpret_cast<const double*>(csr_c_dev.ptr);
  const double* d_x = reinterpret_cast<const double*>(x_dev.ptr);
  double* d_w_out = reinterpret_cast<double*>(w_out_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (nrows > 0) {
    guga_build_w_from_csr_unitnnz_launch_stream(
        d_row_j,
        d_row_k,
        d_csr_rs,
        d_csr_c,
        int(nrows),
        d_x,
        int(state.ncsf),
        int(nops),
        d_w_out,
        int64_t(w_stride),
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_w_from_csr_unitnnz)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_w_from_csr_unitnnz)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("build_w_from_csr_unitnnz overflow (invalid indices)");
  }
}

void build_w_from_epq_table_inplace_device(
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object epq_data,
    py::object x,
    py::object w_out,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    int k_start,
    int k_count) {
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }
  if (w_out.is_none()) {
    throw std::invalid_argument("w_out must be a device array (cannot be None)");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  int epq_indptr_type = epq_indptr_type_from_typestr(epq_indptr_dev, "epq_indptr");
  int64_t epq_indptr_itemsize = epq_indptr_itemsize_from_type(epq_indptr_type);
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != epq_indptr_itemsize) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D device array");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  int epq_pq_type = epq_pq_type_from_typestr(epq_pq_dev, "epq_pq");
  int64_t epq_pq_itemsize = epq_pq_itemsize_from_type(epq_pq_type);
  if (epq_pq_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_pq must have the same shape as epq_indices");
  }
  int64_t nops_ll_cap = (int64_t)state.norb * (int64_t)state.norb;
  if (epq_pq_type == 1 && nops_ll_cap > 256) {
    throw std::invalid_argument("epq_pq dtype <u1 is too small for current norb");
  }
  if (epq_pq_type == 2 && nops_ll_cap > 65535) {
    throw std::invalid_argument("epq_pq dtype <u2 is too small for current norb");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != epq_pq_itemsize) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto epq_data_dev = cuda_array_view_from_object(epq_data, "epq_data");
  std::string epq_typestr = epq_data_dev.typestr;
  if (!epq_typestr.empty() && epq_typestr[0] == '=') epq_typestr[0] = '<';
  bool epq_use_f32 = false;
  int64_t epq_itemsize = 0;
  if (epq_typestr == "<f8") {
    epq_use_f32 = false;
    epq_itemsize = (int64_t)sizeof(double);
  } else if (epq_typestr == "<f4") {
    epq_use_f32 = true;
    epq_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_data_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_data must have the same shape as epq_indices");
  }
  if (!epq_data_dev.strides_bytes.empty()) {
    if (epq_data_dev.strides_bytes.size() != 1 || epq_data_dev.strides_bytes[0] != epq_itemsize) {
      throw std::invalid_argument("epq_data must be contiguous");
    }
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  std::string x_typestr = normalize_typestr(x_dev.typestr);
  bool out_use_f32 = false;
  int64_t out_itemsize = 0;
  if (x_typestr == "<f8") {
    out_use_f32 = false;
    out_itemsize = (int64_t)sizeof(double);
  } else if (x_typestr == "<f4") {
    out_use_f32 = true;
    out_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("x must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (out_use_f32 && !epq_use_f32) {
    throw std::invalid_argument("float32 output requires epq_data float32");
  }
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != out_itemsize) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  const int64_t nops_ll = (int64_t)state.norb * (int64_t)state.norb;
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops too large");
  }
  int nops = (int)nops_ll;

  auto w_out_dev = cuda_array_view_from_object(w_out, "w_out");
  if (out_use_f32) require_typestr(w_out_dev, "w_out", "<f4");
  else require_typestr(w_out_dev, "w_out", "<f8");
  if (w_out_dev.read_only) {
    throw std::invalid_argument("w_out must be writable");
  }

  int expected_rows = (k_count > 0) ? k_count : (int)state.ncsf;

  int64_t w_stride = 0;
  if (w_out_dev.shape.size() == 1) {
    if (w_out_dev.shape[0] != (int64_t)expected_rows * nops_ll) {
      throw std::invalid_argument("w_out (1D) must have shape (k_count*nops,)");
    }
    if (!w_out_dev.strides_bytes.empty()) {
      if (w_out_dev.strides_bytes.size() != 1 || w_out_dev.strides_bytes[0] != out_itemsize) {
        throw std::invalid_argument("w_out (1D) must be contiguous");
      }
    }
    w_stride = nops_ll;
  } else if (w_out_dev.shape.size() == 2) {
    if (w_out_dev.shape[0] != (int64_t)expected_rows) {
      throw std::invalid_argument("w_out (2D) must have shape (k_count, nops_or_more)");
    }
    if (w_out_dev.shape[1] < nops_ll) {
      throw std::invalid_argument("w_out (2D) second dimension too small");
    }
    if (w_out_dev.strides_bytes.empty()) {
      w_stride = w_out_dev.shape[1];
    } else {
      if (w_out_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_out (2D) strides must have length 2");
      }
      int64_t s0 = w_out_dev.strides_bytes[0];
      int64_t s1 = w_out_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("w_out (2D) must have positive strides");
      }
      if (s1 != out_itemsize) {
        throw std::invalid_argument("w_out (2D) must be C-contiguous along last dimension");
      }
      if (s0 % out_itemsize != 0) {
        throw std::invalid_argument("w_out (2D) row stride must be a multiple of itemsize");
      }
      w_stride = s0 / out_itemsize;
    }
    if (w_stride < nops_ll) {
      throw std::invalid_argument("w_out row stride too small");
    }
  } else {
    throw std::invalid_argument("w_out must be 1D or 2D device array");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (w_out_dev.stream) stream_u = w_out_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (epq_data_dev.stream) stream_u = epq_data_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const void* d_epq_indptr = epq_indptr_dev.ptr;
  const int32_t* d_epq_indices = reinterpret_cast<const int32_t*>(epq_indices_dev.ptr);
  const void* d_epq_pq = epq_pq_dev.ptr;
  const double* d_epq_data_f64 = nullptr;
  const float* d_epq_data_f32 = nullptr;
  if (epq_use_f32) d_epq_data_f32 = reinterpret_cast<const float*>(epq_data_dev.ptr);
  else d_epq_data_f64 = reinterpret_cast<const double*>(epq_data_dev.ptr);

  const double* d_x_f64 = nullptr;
  const float* d_x_f32 = nullptr;
  if (out_use_f32) d_x_f32 = reinterpret_cast<const float*>(x_dev.ptr);
  else d_x_f64 = reinterpret_cast<const double*>(x_dev.ptr);

  double* d_w_out_f64 = nullptr;
  float* d_w_out_f32 = nullptr;
  if (out_use_f32) d_w_out_f32 = reinterpret_cast<float*>(w_out_dev.ptr);
  else d_w_out_f64 = reinterpret_cast<double*>(w_out_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (out_use_f32) {
    guga_build_w_from_epq_table_f32_launch_stream(
        d_epq_indptr,
        epq_indptr_type,
        d_epq_indices,
        d_epq_pq,
        epq_pq_type,
        d_epq_data_f32,
        d_x_f32,
        int(state.ncsf),
        int(nops),
        d_w_out_f32,
        int64_t(w_stride),
        d_overflow,
        stream_t,
        threads,
        int(k_start),
        int(k_count));
  } else {
    if (epq_use_f32) {
      guga_build_w_from_epq_table_f64_out_f32_coeff_launch_stream(
          d_epq_indptr,
          epq_indptr_type,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f32,
          d_x_f64,
          int(state.ncsf),
          int(nops),
          d_w_out_f64,
          int64_t(w_stride),
          d_overflow,
          stream_t,
          threads,
          int(k_start),
          int(k_count));
    } else {
      guga_build_w_from_epq_table_launch_stream(
          d_epq_indptr,
          epq_indptr_type,
          d_epq_indices,
          d_epq_pq,
          epq_pq_type,
          d_epq_data_f64,
          d_x_f64,
          int(state.ncsf),
          int(nops),
          d_w_out_f64,
          int64_t(w_stride),
          d_overflow,
          stream_t,
          threads,
          int(k_start),
          int(k_count));
    }
  }
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_w_from_epq_table)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_w_from_epq_table)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("build_w_from_epq_table overflow (invalid indices)");
  }
}

void build_w_from_epq_transpose_range_inplace_device(
    const DeviceStateCache& state,
    py::object epq_t_indptr,
    py::object epq_t_source,
    py::object epq_t_pq,
    py::object epq_t_data,
    py::object x,
    py::object w_out,
    py::object overflow,
    int threads,
    uint64_t stream,
    bool sync,
    bool check_overflow,
    int k_start,
    int k_count) {
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }
  if (w_out.is_none()) {
    throw std::invalid_argument("w_out must be a device array (cannot be None)");
  }
  if (k_start < 0 || k_start >= state.ncsf) {
    throw std::invalid_argument("k_start out of range");
  }

  int eff_k_count = (k_count > 0) ? k_count : (state.ncsf - k_start);
  if (eff_k_count <= 0 || k_start + eff_k_count > state.ncsf) {
    throw std::invalid_argument("k_count out of range");
  }

  auto epq_t_indptr_dev = cuda_array_view_from_object(epq_t_indptr, "epq_t_indptr");
  int epq_t_indptr_type = epq_indptr_type_from_typestr(epq_t_indptr_dev, "epq_t_indptr");
  int64_t epq_t_indptr_itemsize = epq_indptr_itemsize_from_type(epq_t_indptr_type);
  if (epq_t_indptr_dev.shape.size() != 1 || epq_t_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_t_indptr must have shape (ncsf+1,)");
  }
  if (!epq_t_indptr_dev.strides_bytes.empty()) {
    if (epq_t_indptr_dev.strides_bytes.size() != 1 ||
        epq_t_indptr_dev.strides_bytes[0] != epq_t_indptr_itemsize) {
      throw std::invalid_argument("epq_t_indptr must be contiguous");
    }
  }

  auto epq_t_source_dev = cuda_array_view_from_object(epq_t_source, "epq_t_source");
  require_typestr(epq_t_source_dev, "epq_t_source", "<i4");
  if (epq_t_source_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_t_source must be 1D device array");
  }
  if (!epq_t_source_dev.strides_bytes.empty()) {
    if (epq_t_source_dev.strides_bytes.size() != 1 ||
        epq_t_source_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_t_source must be contiguous");
    }
  }

  auto epq_t_pq_dev = cuda_array_view_from_object(epq_t_pq, "epq_t_pq");
  int epq_t_pq_type = epq_pq_type_from_typestr(epq_t_pq_dev, "epq_t_pq");
  int64_t epq_t_pq_itemsize = epq_pq_itemsize_from_type(epq_t_pq_type);
  if (epq_t_pq_dev.shape != epq_t_source_dev.shape) {
    throw std::invalid_argument("epq_t_pq must have the same shape as epq_t_source");
  }
  int64_t nops_ll_cap = (int64_t)state.norb * (int64_t)state.norb;
  if (epq_t_pq_type == 1 && nops_ll_cap > 256) {
    throw std::invalid_argument("epq_t_pq dtype <u1 is too small for current norb");
  }
  if (epq_t_pq_type == 2 && nops_ll_cap > 65535) {
    throw std::invalid_argument("epq_t_pq dtype <u2 is too small for current norb");
  }
  if (!epq_t_pq_dev.strides_bytes.empty()) {
    if (epq_t_pq_dev.strides_bytes.size() != 1 || epq_t_pq_dev.strides_bytes[0] != epq_t_pq_itemsize) {
      throw std::invalid_argument("epq_t_pq must be contiguous");
    }
  }

  auto epq_t_data_dev = cuda_array_view_from_object(epq_t_data, "epq_t_data");
  std::string epq_t_typestr = epq_t_data_dev.typestr;
  if (!epq_t_typestr.empty() && epq_t_typestr[0] == '=') epq_t_typestr[0] = '<';
  bool epq_use_f32 = false;
  int64_t epq_itemsize = 0;
  if (epq_t_typestr == "<f8") {
    epq_use_f32 = false;
    epq_itemsize = (int64_t)sizeof(double);
  } else if (epq_t_typestr == "<f4") {
    epq_use_f32 = true;
    epq_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_t_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_t_data_dev.shape != epq_t_source_dev.shape) {
    throw std::invalid_argument("epq_t_data must have the same shape as epq_t_source");
  }
  if (!epq_t_data_dev.strides_bytes.empty()) {
    if (epq_t_data_dev.strides_bytes.size() != 1 || epq_t_data_dev.strides_bytes[0] != epq_itemsize) {
      throw std::invalid_argument("epq_t_data must be contiguous");
    }
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  std::string x_typestr = normalize_typestr(x_dev.typestr);
  bool out_use_f32 = false;
  int64_t out_itemsize = 0;
  if (x_typestr == "<f8") {
    out_use_f32 = false;
    out_itemsize = (int64_t)sizeof(double);
  } else if (x_typestr == "<f4") {
    out_use_f32 = true;
    out_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("x must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (out_use_f32 && !epq_use_f32) {
    throw std::invalid_argument("float32 output requires epq_t_data float32");
  }
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != out_itemsize) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  int64_t nops_ll = (int64_t)state.norb * (int64_t)state.norb;
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops too large");
  }
  int nops = (int)nops_ll;

  auto w_out_dev = cuda_array_view_from_object(w_out, "w_out");
  if (out_use_f32) require_typestr(w_out_dev, "w_out", "<f4");
  else require_typestr(w_out_dev, "w_out", "<f8");
  if (w_out_dev.read_only) {
    throw std::invalid_argument("w_out must be writable");
  }

  int64_t w_stride = 0;
  if (w_out_dev.shape.size() == 1) {
    if (w_out_dev.shape[0] != (int64_t)eff_k_count * nops_ll) {
      throw std::invalid_argument("w_out (1D) must have shape (k_count*nops,)");
    }
    if (!w_out_dev.strides_bytes.empty()) {
      if (w_out_dev.strides_bytes.size() != 1 || w_out_dev.strides_bytes[0] != out_itemsize) {
        throw std::invalid_argument("w_out (1D) must be contiguous");
      }
    }
    w_stride = nops_ll;
  } else if (w_out_dev.shape.size() == 2) {
    if (w_out_dev.shape[0] != (int64_t)eff_k_count) {
      throw std::invalid_argument("w_out (2D) must have shape (k_count, nops_or_more)");
    }
    if (w_out_dev.shape[1] < nops_ll) {
      throw std::invalid_argument("w_out (2D) second dimension too small");
    }
    if (w_out_dev.strides_bytes.empty()) {
      w_stride = w_out_dev.shape[1];
    } else {
      if (w_out_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_out (2D) strides must have length 2");
      }
      int64_t s0 = w_out_dev.strides_bytes[0];
      int64_t s1 = w_out_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("w_out (2D) must have positive strides");
      }
      if (s1 != out_itemsize) {
        throw std::invalid_argument("w_out (2D) must be C-contiguous along last dimension");
      }
      if (s0 % out_itemsize != 0) {
        throw std::invalid_argument("w_out (2D) row stride must be a multiple of itemsize");
      }
      w_stride = s0 / out_itemsize;
    }
    if (w_stride < nops_ll) {
      throw std::invalid_argument("w_out row stride too small");
    }
  } else {
    throw std::invalid_argument("w_out must be 1D or 2D device array");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (w_out_dev.stream) stream_u = w_out_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
    else if (epq_t_data_dev.stream) stream_u = epq_t_data_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const void* d_epq_t_indptr = epq_t_indptr_dev.ptr;
  const int32_t* d_epq_t_source = reinterpret_cast<const int32_t*>(epq_t_source_dev.ptr);
  const void* d_epq_t_pq = epq_t_pq_dev.ptr;
  const double* d_epq_t_data_f64 = nullptr;
  const float* d_epq_t_data_f32 = nullptr;
  if (epq_use_f32) d_epq_t_data_f32 = reinterpret_cast<const float*>(epq_t_data_dev.ptr);
  else d_epq_t_data_f64 = reinterpret_cast<const double*>(epq_t_data_dev.ptr);

  const double* d_x_f64 = nullptr;
  const float* d_x_f32 = nullptr;
  if (out_use_f32) d_x_f32 = reinterpret_cast<const float*>(x_dev.ptr);
  else d_x_f64 = reinterpret_cast<const double*>(x_dev.ptr);

  double* d_w_out_f64 = nullptr;
  float* d_w_out_f32 = nullptr;
  if (out_use_f32) d_w_out_f32 = reinterpret_cast<float*>(w_out_dev.ptr);
  else d_w_out_f64 = reinterpret_cast<double*>(w_out_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (out_use_f32) {
    guga_build_w_from_epq_transpose_range_f32_launch_stream(
        d_epq_t_indptr,
        epq_t_indptr_type,
        d_epq_t_source,
        d_epq_t_pq,
        epq_t_pq_type,
        d_epq_t_data_f32,
        d_x_f32,
        int(state.ncsf),
        int(nops),
        d_w_out_f32,
        int64_t(w_stride),
        d_overflow,
        stream_t,
        threads,
        int(k_start),
        int(eff_k_count));
  } else {
    if (epq_use_f32) {
      guga_build_w_from_epq_transpose_range_f64_out_f32_coeff_launch_stream(
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f32,
          d_x_f64,
          int(state.ncsf),
          int(nops),
          d_w_out_f64,
          int64_t(w_stride),
          d_overflow,
          stream_t,
          threads,
          int(k_start),
          int(eff_k_count));
    } else {
      guga_build_w_from_epq_transpose_range_launch_stream(
          d_epq_t_indptr,
          epq_t_indptr_type,
          d_epq_t_source,
          d_epq_t_pq,
          epq_t_pq_type,
          d_epq_t_data_f64,
          d_x_f64,
          int(state.ncsf),
          int(nops),
          d_w_out_f64,
          int64_t(w_stride),
          d_overflow,
          stream_t,
          threads,
          int(k_start),
          int(eff_k_count));
    }
  }
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_w_from_epq_transpose_range)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(
        cudaStreamSynchronize(stream_t),
        "cudaStreamSynchronize(build_w_from_epq_transpose_range)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("build_w_from_epq_transpose_range overflow (invalid indices)");
  }
}

void build_t_from_epq_table_inplace_device(
    const DeviceStateCache& state,
    py::object epq_indptr,
    py::object epq_indices,
    py::object epq_pq,
    py::object epq_data,
    py::object c_vec,
    py::object t_out,
    py::object overflow,
    int threads,
    bool zero_out,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }
  if (t_out.is_none()) {
    throw std::invalid_argument("t_out must be a device array (cannot be None)");
  }

  auto epq_indptr_dev = cuda_array_view_from_object(epq_indptr, "epq_indptr");
  require_typestr(epq_indptr_dev, "epq_indptr", "<i8");
  if (epq_indptr_dev.shape.size() != 1 || epq_indptr_dev.shape[0] != (int64_t)state.ncsf + 1) {
    throw std::invalid_argument("epq_indptr must have shape (ncsf+1,)");
  }
  if (!epq_indptr_dev.strides_bytes.empty()) {
    if (epq_indptr_dev.strides_bytes.size() != 1 || epq_indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("epq_indptr must be contiguous");
    }
  }

  auto epq_indices_dev = cuda_array_view_from_object(epq_indices, "epq_indices");
  require_typestr(epq_indices_dev, "epq_indices", "<i4");
  if (epq_indices_dev.shape.size() != 1) {
    throw std::invalid_argument("epq_indices must be 1D device array");
  }
  if (!epq_indices_dev.strides_bytes.empty()) {
    if (epq_indices_dev.strides_bytes.size() != 1 || epq_indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("epq_indices must be contiguous");
    }
  }

  auto epq_pq_dev = cuda_array_view_from_object(epq_pq, "epq_pq");
  int epq_pq_type = epq_pq_type_from_typestr(epq_pq_dev, "epq_pq");
  int64_t epq_pq_itemsize = epq_pq_itemsize_from_type(epq_pq_type);
  if (epq_pq_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_pq must have the same shape as epq_indices");
  }
  int64_t nops_ll_cap = (int64_t)state.norb * (int64_t)state.norb;
  if (epq_pq_type == 1 && nops_ll_cap > 256) {
    throw std::invalid_argument("epq_pq dtype <u1 is too small for current norb");
  }
  if (epq_pq_type == 2 && nops_ll_cap > 65535) {
    throw std::invalid_argument("epq_pq dtype <u2 is too small for current norb");
  }
  if (!epq_pq_dev.strides_bytes.empty()) {
    if (epq_pq_dev.strides_bytes.size() != 1 || epq_pq_dev.strides_bytes[0] != epq_pq_itemsize) {
      throw std::invalid_argument("epq_pq must be contiguous");
    }
  }

  auto epq_data_dev = cuda_array_view_from_object(epq_data, "epq_data");
  std::string epq_typestr = epq_data_dev.typestr;
  if (!epq_typestr.empty() && epq_typestr[0] == '=') epq_typestr[0] = '<';
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (epq_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (epq_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("epq_data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (epq_data_dev.shape != epq_indices_dev.shape) {
    throw std::invalid_argument("epq_data must have the same shape as epq_indices");
  }
  if (!epq_data_dev.strides_bytes.empty()) {
    if (epq_data_dev.strides_bytes.size() != 1 || epq_data_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("epq_data must be contiguous");
    }
  }

  auto c_dev = cuda_array_view_from_object(c_vec, "c_vec");
  if (use_f32) require_typestr(c_dev, "c_vec", "<f4");
  else require_typestr(c_dev, "c_vec", "<f8");
  if (c_dev.shape.size() != 1 || c_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c_vec must have shape (ncsf,)");
  }
  if (!c_dev.strides_bytes.empty()) {
    if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("c_vec must be contiguous");
    }
  }

  auto t_out_dev = cuda_array_view_from_object(t_out, "t_out");
  if (use_f32) require_typestr(t_out_dev, "t_out", "<f4");
  else require_typestr(t_out_dev, "t_out", "<f8");
  if (t_out_dev.read_only) {
    throw std::invalid_argument("t_out must be writable");
  }
  int64_t nops_ll = (int64_t)state.norb * (int64_t)state.norb;
  if (t_out_dev.shape.size() != 2) {
    throw std::invalid_argument("t_out must be a 2D device array");
  }
  if (t_out_dev.shape[0] != nops_ll || t_out_dev.shape[1] != (int64_t)state.ncsf) {
    throw std::invalid_argument("t_out must have shape (nops, ncsf)");
  }
  if (!t_out_dev.strides_bytes.empty()) {
    if (t_out_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("t_out strides must have length 2");
    }
    int64_t s0 = t_out_dev.strides_bytes[0];
    int64_t s1 = t_out_dev.strides_bytes[1];
    if (s0 <= 0 || s1 <= 0) {
      throw std::invalid_argument("t_out must have positive strides");
    }
    if (s1 != fp_itemsize) {
      throw std::invalid_argument("t_out must be C-contiguous along last dimension");
    }
    if (s0 != (int64_t)state.ncsf * fp_itemsize) {
      throw std::invalid_argument("t_out must be C-contiguous");
    }
  }
  int64_t t_stride = (int64_t)state.ncsf;

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (t_out_dev.stream) stream_u = t_out_dev.stream;
    else if (c_dev.stream) stream_u = c_dev.stream;
    else if (epq_data_dev.stream) stream_u = epq_data_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const int64_t* d_epq_indptr = reinterpret_cast<const int64_t*>(epq_indptr_dev.ptr);
  const int32_t* d_epq_indices = reinterpret_cast<const int32_t*>(epq_indices_dev.ptr);
  const void* d_epq_pq = epq_pq_dev.ptr;
  const double* d_epq_data_f64 = nullptr;
  const float* d_epq_data_f32 = nullptr;
  if (use_f32) d_epq_data_f32 = reinterpret_cast<const float*>(epq_data_dev.ptr);
  else d_epq_data_f64 = reinterpret_cast<const double*>(epq_data_dev.ptr);

  const double* d_c_f64 = nullptr;
  const float* d_c_f32 = nullptr;
  if (use_f32) d_c_f32 = reinterpret_cast<const float*>(c_dev.ptr);
  else d_c_f64 = reinterpret_cast<const double*>(c_dev.ptr);

  double* d_t_out_f64 = nullptr;
  float* d_t_out_f32 = nullptr;
  if (use_f32) d_t_out_f32 = reinterpret_cast<float*>(t_out_dev.ptr);
  else d_t_out_f64 = reinterpret_cast<double*>(t_out_dev.ptr);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  if (zero_out) {
    size_t bytes = (size_t)nops_ll * (size_t)t_stride * (size_t)fp_itemsize;
    void* d_t_out_void = use_f32 ? static_cast<void*>(d_t_out_f32) : static_cast<void*>(d_t_out_f64);
    throw_on_cuda_error(cudaMemsetAsync(d_t_out_void, 0, bytes, stream_t), "cudaMemsetAsync(t_out=0)");
  }

  if (use_f32) {
    guga_build_t_from_epq_table_f32_launch_stream(
        d_steps,
        d_epq_indptr,
        d_epq_indices,
        d_epq_pq,
        epq_pq_type,
        d_epq_data_f32,
        d_c_f32,
        int(state.ncsf),
        int(state.norb),
        int(nops_ll),
        d_t_out_f32,
        int64_t(t_stride),
        d_overflow,
        stream_t,
        threads);
  } else {
    guga_build_t_from_epq_table_launch_stream(
        d_steps,
        d_epq_indptr,
        d_epq_indices,
        d_epq_pq,
        epq_pq_type,
        d_epq_data_f64,
        d_c_f64,
        int(state.ncsf),
        int(state.norb),
        int(nops_ll),
        d_t_out_f64,
        int64_t(t_stride),
        d_overflow,
        stream_t,
        threads);
  }
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_t_from_epq_table)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_t_from_epq_table)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("build_t_from_epq_table overflow (invalid indices)");
  }
}

void build_occ_block_from_steps_inplace_device(
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object occ_out,
    int threads,
    uint64_t stream,
    bool sync) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (j_start < 0 || j_count < 0 || j_start + j_count > state.ncsf) {
    throw std::invalid_argument("invalid j_start/j_count for DeviceStateCache");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (occ_out.is_none()) {
    throw std::invalid_argument("occ_out must be a device array (cannot be None)");
  }

  auto occ_dev = cuda_array_view_from_object(occ_out, "occ_out");
  require_typestr(occ_dev, "occ_out", "<f8");
  if (occ_dev.read_only) {
    throw std::invalid_argument("occ_out must be writable");
  }
  if (occ_dev.shape.size() != 2) {
    throw std::invalid_argument("occ_out must be a 2D device array");
  }
  if ((int)occ_dev.shape[0] != j_count || (int)occ_dev.shape[1] != state.norb) {
    throw std::invalid_argument("occ_out must have shape (j_count,norb)");
  }
  if (!occ_dev.strides_bytes.empty()) {
    if (occ_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("occ_out strides must have length 2");
    }
    int64_t s0 = occ_dev.strides_bytes[0];
    int64_t s1 = occ_dev.strides_bytes[1];
    if (s0 <= 0 || s1 <= 0) {
      throw std::invalid_argument("occ_out must have positive strides");
    }
    if (s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("occ_out must be C-contiguous along last dimension");
    }
    if (s0 != (int64_t)state.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("occ_out must be C-contiguous with no padding");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (occ_dev.stream) stream_u = occ_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  double* d_occ = reinterpret_cast<double*>(occ_dev.ptr);

  throw_on_cuda_error(
      guga_build_occ_block_from_steps_launch_stream(state.steps, state.ncsf, state.norb, j_start, j_count, d_occ, stream_t, threads),
      "guga_build_occ_block_from_steps_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_occ_block)");
  }
}

void build_w_diag_from_steps_inplace_device(
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object x,
    py::object w_out,
    int threads,
    uint64_t stream,
    bool sync,
    bool relative_w) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (j_start < 0 || j_count < 0 || j_start + j_count > state.ncsf) {
    throw std::invalid_argument("invalid j_start/j_count for DeviceStateCache");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (x.is_none() || w_out.is_none()) {
    throw std::invalid_argument("x and w_out must be device arrays (cannot be None)");
  }

  auto x_dev = cuda_array_view_from_object(x, "x");
  std::string x_typestr = normalize_typestr(x_dev.typestr);
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  if (x_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (x_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("x must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("x must be contiguous");
    }
  }

  const int64_t nops_ll = (int64_t)state.norb * (int64_t)state.norb;
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops too large");
  }

  auto w_out_dev = cuda_array_view_from_object(w_out, "w_out");
  if (use_f32) require_typestr(w_out_dev, "w_out", "<f4");
  else require_typestr(w_out_dev, "w_out", "<f8");
  if (w_out_dev.read_only) {
    throw std::invalid_argument("w_out must be writable");
  }

  int expected_rows = relative_w ? j_count : (int)state.ncsf;

  int64_t w_stride = 0;
  if (w_out_dev.shape.size() == 1) {
    if (w_out_dev.shape[0] != (int64_t)expected_rows * nops_ll) {
      throw std::invalid_argument("w_out (1D) must have shape (expected_rows*nops,)");
    }
    if (!w_out_dev.strides_bytes.empty()) {
      if (w_out_dev.strides_bytes.size() != 1 || w_out_dev.strides_bytes[0] != fp_itemsize) {
        throw std::invalid_argument("w_out (1D) must be contiguous");
      }
    }
    w_stride = nops_ll;
  } else if (w_out_dev.shape.size() == 2) {
    if (w_out_dev.shape[0] != (int64_t)expected_rows) {
      throw std::invalid_argument("w_out (2D) must have shape (expected_rows, nops)");
    }
    if (w_out_dev.shape[1] < nops_ll) {
      throw std::invalid_argument("w_out (2D) second dimension too small");
    }
    if (w_out_dev.strides_bytes.empty()) {
      w_stride = w_out_dev.shape[1];
    } else {
      if (w_out_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("w_out (2D) strides must have length 2");
      }
      int64_t s0 = w_out_dev.strides_bytes[0];
      int64_t s1 = w_out_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("w_out (2D) must have positive strides");
      }
      if (s1 != fp_itemsize) {
        throw std::invalid_argument("w_out (2D) must be C-contiguous along last dimension");
      }
      if (s0 % fp_itemsize != 0) {
        throw std::invalid_argument("w_out (2D) row stride must be a multiple of itemsize");
      }
      w_stride = s0 / fp_itemsize;
    }
    if (w_stride < nops_ll) {
      throw std::invalid_argument("w_out row stride too small");
    }
  } else {
    throw std::invalid_argument("w_out must be 1D or 2D device array");
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (w_out_dev.stream) stream_u = w_out_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const double* d_x_f64 = nullptr;
  const float* d_x_f32 = nullptr;
  if (use_f32) d_x_f32 = reinterpret_cast<const float*>(x_dev.ptr);
  else d_x_f64 = reinterpret_cast<const double*>(x_dev.ptr);
  double* d_w_f64 = nullptr;
  float* d_w_f32 = nullptr;
  if (use_f32) d_w_f32 = reinterpret_cast<float*>(w_out_dev.ptr);
  else d_w_f64 = reinterpret_cast<double*>(w_out_dev.ptr);

  if (j_count > 0) {
    if (use_f32) {
      throw_on_cuda_error(
          guga_build_w_diag_from_steps_f32_launch_stream(
              d_steps,
              state.ncsf,
              state.norb,
              int(j_start),
              int(j_count),
              d_x_f32,
              int(nops_ll),
              d_w_f32,
              int64_t(w_stride),
              stream_t,
              threads,
              int(relative_w)),
          "guga_build_w_diag_from_steps_f32_launch_stream");
    } else {
      throw_on_cuda_error(
          guga_build_w_diag_from_steps_launch_stream(
              d_steps,
              state.ncsf,
              state.norb,
              int(j_start),
              int(j_count),
              d_x_f64,
              int(nops_ll),
              d_w_f64,
              int64_t(w_stride),
              stream_t,
              threads,
              int(relative_w)),
          "guga_build_w_diag_from_steps_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_w_diag_from_steps)");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_w_diag_from_steps)");
  }
}

void build_hdiag_det_guess_from_steps_inplace_device(
    const DeviceStateCache& state,
    int neleca_det,
    py::object h1e_diag,
    py::object eri_ppqq,
    py::object eri_pqqp,
    py::object hdiag_out,
    int threads,
    uint64_t stream,
    bool sync) {
  if (state.steps == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (state.norb <= 0 || state.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (neleca_det < 0) {
    throw std::invalid_argument("neleca_det must be >= 0");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (h1e_diag.is_none() || eri_ppqq.is_none() || eri_pqqp.is_none() || hdiag_out.is_none()) {
    throw std::invalid_argument("h1e_diag/eri_ppqq/eri_pqqp/hdiag_out must be device arrays (cannot be None)");
  }

  auto h1e_diag_dev = cuda_array_view_from_object(h1e_diag, "h1e_diag");
  require_typestr(h1e_diag_dev, "h1e_diag", "<f8");
  if (h1e_diag_dev.shape.size() != 1 || (int)h1e_diag_dev.shape[0] != state.norb) {
    throw std::invalid_argument("h1e_diag must have shape (norb,)");
  }
  if (!h1e_diag_dev.strides_bytes.empty()) {
    if (h1e_diag_dev.strides_bytes.size() != 1 || h1e_diag_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("h1e_diag must be contiguous");
    }
  }

  auto eri_ppqq_dev = cuda_array_view_from_object(eri_ppqq, "eri_ppqq");
  require_typestr(eri_ppqq_dev, "eri_ppqq", "<f8");
  if (eri_ppqq_dev.shape.size() != 2 || (int)eri_ppqq_dev.shape[0] != state.norb || (int)eri_ppqq_dev.shape[1] != state.norb) {
    throw std::invalid_argument("eri_ppqq must have shape (norb,norb)");
  }
  if (!eri_ppqq_dev.strides_bytes.empty()) {
    if (eri_ppqq_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("eri_ppqq strides must have length 2");
    }
    int64_t s0 = eri_ppqq_dev.strides_bytes[0];
    int64_t s1 = eri_ppqq_dev.strides_bytes[1];
    if (s0 <= 0 || s1 <= 0) {
      throw std::invalid_argument("eri_ppqq must have positive strides");
    }
    if (s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_ppqq must be C-contiguous along last dimension");
    }
    if (s0 != (int64_t)state.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_ppqq must be C-contiguous with no padding");
    }
  }

  auto eri_pqqp_dev = cuda_array_view_from_object(eri_pqqp, "eri_pqqp");
  require_typestr(eri_pqqp_dev, "eri_pqqp", "<f8");
  if (eri_pqqp_dev.shape.size() != 2 || (int)eri_pqqp_dev.shape[0] != state.norb || (int)eri_pqqp_dev.shape[1] != state.norb) {
    throw std::invalid_argument("eri_pqqp must have shape (norb,norb)");
  }
  if (!eri_pqqp_dev.strides_bytes.empty()) {
    if (eri_pqqp_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("eri_pqqp strides must have length 2");
    }
    int64_t s0 = eri_pqqp_dev.strides_bytes[0];
    int64_t s1 = eri_pqqp_dev.strides_bytes[1];
    if (s0 <= 0 || s1 <= 0) {
      throw std::invalid_argument("eri_pqqp must have positive strides");
    }
    if (s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_pqqp must be C-contiguous along last dimension");
    }
    if (s0 != (int64_t)state.norb * (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_pqqp must be C-contiguous with no padding");
    }
  }

  auto hdiag_dev = cuda_array_view_from_object(hdiag_out, "hdiag_out");
  require_typestr(hdiag_dev, "hdiag_out", "<f8");
  if (hdiag_dev.read_only) {
    throw std::invalid_argument("hdiag_out must be writable");
  }
  if (hdiag_dev.shape.size() != 1 || hdiag_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("hdiag_out must have shape (ncsf,)");
  }
  if (!hdiag_dev.strides_bytes.empty()) {
    if (hdiag_dev.strides_bytes.size() != 1 || hdiag_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("hdiag_out must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (hdiag_dev.stream) stream_u = hdiag_dev.stream;
    else if (h1e_diag_dev.stream) stream_u = h1e_diag_dev.stream;
    else if (eri_ppqq_dev.stream) stream_u = eri_ppqq_dev.stream;
    else if (eri_pqqp_dev.stream) stream_u = eri_pqqp_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int8_t* d_steps = reinterpret_cast<const int8_t*>(state.steps);
  const double* d_h1e_diag = reinterpret_cast<const double*>(h1e_diag_dev.ptr);
  const double* d_eri_ppqq = reinterpret_cast<const double*>(eri_ppqq_dev.ptr);
  const double* d_eri_pqqp = reinterpret_cast<const double*>(eri_pqqp_dev.ptr);
  double* d_hdiag = reinterpret_cast<double*>(hdiag_dev.ptr);

  throw_on_cuda_error(
      guga_build_hdiag_det_guess_from_steps_launch_stream(
          d_steps,
          state.ncsf,
          state.norb,
          int(neleca_det),
          d_h1e_diag,
          d_eri_ppqq,
          d_eri_pqqp,
          d_hdiag,
          stream_t,
          threads),
      "guga_build_hdiag_det_guess_from_steps_launch_stream");
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_hdiag_det_guess_from_steps)");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_hdiag_det_guess_from_steps)");
  }
}

void apply_g_flat_task_sums_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_g,
    py::object task_scale,
    py::object out_sum,
    py::object overflow,
    int threads,
    bool zero_out,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (out_sum.is_none()) {
    throw std::invalid_argument("out_sum must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  if (task_csf_dev.shape.size() != 1) {
    throw std::invalid_argument("task_csf must be 1D device array");
  }
  if (!task_csf_dev.strides_bytes.empty()) {
    if (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("task_csf must be contiguous");
    }
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("task_csf too large");
  }
  int ntasks = (int)ntasks_ll;

  int64_t nops_ll = (int64_t)drt.norb * (int64_t)drt.norb;
  int nops = (int)nops_ll;

  auto task_g_dev = cuda_array_view_from_object(task_g, "task_g");
  require_typestr(task_g_dev, "task_g", "<f8");

  int64_t g_stride = 0;
  if (task_g_dev.shape.size() == 1) {
    if (task_g_dev.shape[0] != nops_ll) {
      throw std::invalid_argument("task_g (1D) must have shape (norb*norb,)");
    }
    g_stride = 0;  // broadcast
  } else if (task_g_dev.shape.size() == 2) {
    if (task_g_dev.shape[0] != ntasks_ll || task_g_dev.shape[1] != nops_ll) {
      throw std::invalid_argument("task_g (2D) must have shape (ntasks,norb*norb)");
    }
    if (task_g_dev.strides_bytes.empty()) {
      g_stride = nops_ll;
    } else {
      if (task_g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("task_g (2D) strides must have 2 entries");
      }
      int64_t s0 = task_g_dev.strides_bytes[0];
      int64_t s1 = task_g_dev.strides_bytes[1];
      if (s0 <= 0 || s1 <= 0) {
        throw std::invalid_argument("task_g (2D) must have positive strides");
      }
      if (s1 != (int64_t)sizeof(double)) {
        throw std::invalid_argument("task_g (2D) must be C-contiguous along last dimension");
      }
      if (s0 < nops_ll * (int64_t)sizeof(double)) {
        throw std::invalid_argument("task_g (2D) row stride too small");
      }
      if (s0 % (int64_t)sizeof(double) != 0) {
        throw std::invalid_argument("task_g (2D) row stride must be a multiple of itemsize");
      }
      g_stride = s0 / (int64_t)sizeof(double);
    }
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D device array");
  }

  uint64_t task_scale_stream = 0;
  const double* d_task_scale = nullptr;
  if (!task_scale.is_none()) {
    auto task_scale_dev = cuda_array_view_from_object(task_scale, "task_scale");
    require_typestr(task_scale_dev, "task_scale", "<f8");
    task_scale_stream = task_scale_dev.stream;
    if (task_scale_dev.shape.size() != 1 || task_scale_dev.shape[0] != ntasks_ll) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    if (!task_scale_dev.strides_bytes.empty()) {
      if (task_scale_dev.strides_bytes.size() != 1 || task_scale_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("task_scale must be contiguous");
      }
    }
    d_task_scale = reinterpret_cast<const double*>(task_scale_dev.ptr);
  }

  auto out_sum_dev = cuda_array_view_from_object(out_sum, "out_sum");
  require_typestr(out_sum_dev, "out_sum", "<f8");
  if (out_sum_dev.read_only) {
    throw std::invalid_argument("out_sum must be writable");
  }
  if (out_sum_dev.shape.size() != 1 || out_sum_dev.shape[0] != ntasks_ll) {
    throw std::invalid_argument("out_sum must have shape (ntasks,)");
  }
  if (!out_sum_dev.strides_bytes.empty()) {
    if (out_sum_dev.strides_bytes.size() != 1 || out_sum_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out_sum must be contiguous");
    }
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (out_sum_dev.stream) stream_u = out_sum_dev.stream;
    else if (task_g_dev.stream) stream_u = task_g_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (task_scale_stream) stream_u = task_scale_stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const double* d_task_g = reinterpret_cast<const double*>(task_g_dev.ptr);
  double* d_out_sum = reinterpret_cast<double*>(out_sum_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_out) {
    size_t out_bytes = (size_t)ntasks * sizeof(double);
    throw_on_cuda_error(cudaMemsetAsync(d_out_sum, 0, out_bytes, stream_t), "cudaMemsetAsync(out_sum=0)");
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (ntasks > 0) {
    guga_apply_g_flat_task_sums_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        d_task_csf,
        d_task_scale,
        d_task_g,
        g_stride,
        ntasks,
        d_out_sum,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(apply_g_flat_task_sums)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(apply_g_flat_task_sums)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("apply_g_flat_task_sums overflow (invalid indices or stack overflow)");
  }
}

void build_t_block_epq_atomic_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object c,
    py::object p_list,
    py::object q_list,
    py::object out,
    py::object overflow,
    int threads,
    bool zero_out,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }
  if (out.is_none()) {
    throw std::invalid_argument("out must be a device array (cannot be None)");
  }
  if (overflow.is_none()) {
    throw std::invalid_argument("overflow must be a device array of shape (1,) (cannot be None)");
  }

  auto c_dev = cuda_array_view_from_object(c, "c");
  require_typestr(c_dev, "c", "<f8");
  if (c_dev.shape.size() != 1) {
    throw std::invalid_argument("c must be 1D device array");
  }
  if (c_dev.shape[0] != (int64_t)state.ncsf) {
    throw std::invalid_argument("c must have shape (ncsf,)");
  }
  if (!c_dev.strides_bytes.empty()) {
    if (c_dev.strides_bytes.size() != 1 || c_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("c must be contiguous");
    }
  }

  auto p_dev = cuda_array_view_from_object(p_list, "p_list");
  require_typestr(p_dev, "p_list", "<i4");
  if (p_dev.shape.size() != 1) {
    throw std::invalid_argument("p_list must be 1D device array");
  }
  if (!p_dev.strides_bytes.empty()) {
    if (p_dev.strides_bytes.size() != 1 || p_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("p_list must be contiguous");
    }
  }

  auto q_dev = cuda_array_view_from_object(q_list, "q_list");
  require_typestr(q_dev, "q_list", "<i4");
  if (q_dev.shape.size() != 1) {
    throw std::invalid_argument("q_list must be 1D device array");
  }
  if (!q_dev.strides_bytes.empty()) {
    if (q_dev.strides_bytes.size() != 1 || q_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("q_list must be contiguous");
    }
  }

  int64_t nops_block_ll = p_dev.shape[0];
  if (nops_block_ll != q_dev.shape[0]) {
    throw std::invalid_argument("p_list and q_list must have the same length");
  }
  if (nops_block_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("p_list too large");
  }
  int nops_block = (int)nops_block_ll;

  auto out_dev = cuda_array_view_from_object(out, "out");
  require_typestr(out_dev, "out", "<f8");
  if (out_dev.read_only) {
    throw std::invalid_argument("out must be writable");
  }
  if (out_dev.shape.size() != 2 || out_dev.shape[0] != nops_block_ll || out_dev.shape[1] != (int64_t)state.ncsf) {
    throw std::invalid_argument("out must have shape (nops_block,ncsf)");
  }

  int64_t out_stride = 0;
  if (out_dev.strides_bytes.empty()) {
    out_stride = (int64_t)state.ncsf;
  } else {
    if (out_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("out strides must have length 2");
    }
    int64_t s0 = out_dev.strides_bytes[0];
    int64_t s1 = out_dev.strides_bytes[1];
    if (s0 <= 0 || s1 <= 0) {
      throw std::invalid_argument("out must have positive strides");
    }
    if (s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out must be C-contiguous along last dimension");
    }
    int64_t min_s0 = (int64_t)state.ncsf * (int64_t)sizeof(double);
    if (s0 < min_s0) {
      throw std::invalid_argument("out row stride too small");
    }
    if (s0 % (int64_t)sizeof(double) != 0) {
      throw std::invalid_argument("out row stride must be a multiple of itemsize");
    }
    out_stride = s0 / (int64_t)sizeof(double);
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) {
    throw std::invalid_argument("overflow must be writable");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (!overflow_dev.strides_bytes.empty()) {
    if (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("overflow must be contiguous");
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (out_dev.stream) stream_u = out_dev.stream;
    else if (c_dev.stream) stream_u = c_dev.stream;
    else if (p_dev.stream) stream_u = p_dev.stream;
    else if (q_dev.stream) stream_u = q_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const double* d_c = reinterpret_cast<const double*>(c_dev.ptr);
  const int32_t* d_p_list = reinterpret_cast<const int32_t*>(p_dev.ptr);
  const int32_t* d_q_list = reinterpret_cast<const int32_t*>(q_dev.ptr);
  double* d_out = reinterpret_cast<double*>(out_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  if (zero_out) {
    size_t row_bytes = (size_t)state.ncsf * sizeof(double);
    for (int op = 0; op < nops_block; op++) {
      throw_on_cuda_error(cudaMemsetAsync(d_out + (int64_t)op * out_stride, 0, row_bytes, stream_t),
                          "cudaMemsetAsync(out_row=0)");
    }
  }
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  if (nops_block > 0 && state.ncsf > 0) {
    guga_build_t_block_epq_atomic_launch_stream(
        drt.child,
        drt.node_twos,
        drt.child_prefix,
        state.steps,
        state.nodes,
        state.ncsf,
        drt.norb,
        d_c,
        d_p_list,
        d_q_list,
        nops_block,
        d_out,
        out_stride,
        d_overflow,
        stream_t,
        threads);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(build_t_block_epq_atomic)");
  }

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t),
                        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(build_t_block_epq_atomic)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("build_t_block_epq_atomic overflow (invalid indices or stack overflow)");
  }
}

py::object apply_g_flat_scatter_atomic(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_csf,
    py::array_t<double, py::array::c_style | py::array::forcecast> task_g,
    py::object task_scale,
    py::object y0,
    int threads,
    bool return_y) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }

  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto csf_buf = task_csf.request();
  auto g_buf = task_g.request();
  if (csf_buf.ndim != 1) {
    throw std::invalid_argument("task_csf must be 1D");
  }

  int ntasks = (int)csf_buf.shape[0];
  if (ntasks < 0) {
    throw std::invalid_argument("ntasks must be >= 0");
  }

  int nops = drt.norb * drt.norb;
  int64_t g_stride = 0;
  if (g_buf.ndim == 1) {
    if ((int)g_buf.shape[0] != nops) {
      throw std::invalid_argument("task_g must have shape (norb*norb,) when 1D");
    }
    g_stride = 0;
  } else if (g_buf.ndim == 2) {
    if ((int)g_buf.shape[0] != ntasks || (int)g_buf.shape[1] != nops) {
      throw std::invalid_argument("task_g must have shape (ntasks,norb*norb) when 2D");
    }
    g_stride = (int64_t)nops;
  } else {
    throw std::invalid_argument("task_g must be 1D or 2D");
  }

  const double* h_scale_ptr = nullptr;
  py::array_t<double> scale_arr;
  if (!task_scale.is_none()) {
    scale_arr = task_scale.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto s_buf = scale_arr.request();
    if (s_buf.ndim != 1 || s_buf.shape[0] != csf_buf.shape[0]) {
      throw std::invalid_argument("task_scale must have shape (ntasks,) when provided");
    }
    h_scale_ptr = static_cast<const double*>(s_buf.ptr);
  }

  const double* h_y0_ptr = nullptr;
  py::array_t<double> y0_arr;
  if (!y0.is_none()) {
    y0_arr = y0.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto y_buf = y0_arr.request();
    if (y_buf.ndim != 1 || (int)y_buf.shape[0] != state.ncsf) {
      throw std::invalid_argument("y0 must have shape (ncsf,) when provided");
    }
    h_y0_ptr = static_cast<const double*>(y_buf.ptr);
  }

  if (ntasks == 0) {
    if (!return_y) return py::none();
    py::array_t<double> out_y({state.ncsf});
    std::fill_n(out_y.mutable_data(), (size_t)state.ncsf, 0.0);
    if (h_y0_ptr) {
      std::copy_n(h_y0_ptr, (size_t)state.ncsf, out_y.mutable_data());
    }
    return std::move(out_y);
  }

  int32_t* d_task_csf_raw = nullptr;
  double* d_task_scale_raw = nullptr;
  double* d_g_raw = nullptr;
  double* d_y_raw = nullptr;
  int* d_overflow_raw = nullptr;

  cuda_unique_ptr<int32_t> d_task_csf;
  cuda_unique_ptr<double> d_task_scale_dev;
  cuda_unique_ptr<double> d_g;
  cuda_unique_ptr<double> d_y;
  cuda_unique_ptr<int> d_overflow;

  size_t task_i_bytes = (size_t)ntasks * sizeof(int32_t);
  size_t task_scale_bytes = (size_t)ntasks * sizeof(double);
  size_t y_bytes = (size_t)state.ncsf * sizeof(double);

  size_t g_bytes = 0;
  if (g_buf.ndim == 1) {
    g_bytes = (size_t)nops * sizeof(double);
  } else {
    g_bytes = (size_t)ntasks * (size_t)nops * sizeof(double);
  }

  throw_on_cuda_error(cudaMalloc((void**)&d_task_csf_raw, task_i_bytes), "cudaMalloc(task_csf)");
  throw_on_cuda_error(cudaMalloc((void**)&d_g_raw, g_bytes), "cudaMalloc(task_g)");
  throw_on_cuda_error(cudaMalloc((void**)&d_y_raw, y_bytes), "cudaMalloc(y)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow_raw, sizeof(int)), "cudaMalloc(overflow)");

  d_task_csf.reset(d_task_csf_raw);
  d_g.reset(d_g_raw);
  d_y.reset(d_y_raw);
  d_overflow.reset(d_overflow_raw);

  throw_on_cuda_error(cudaMemcpy(d_task_csf.get(), csf_buf.ptr, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_csf");
  throw_on_cuda_error(cudaMemcpy(d_g.get(), g_buf.ptr, g_bytes, cudaMemcpyHostToDevice), "H2D task_g");

  if (h_scale_ptr) {
    throw_on_cuda_error(cudaMalloc((void**)&d_task_scale_raw, task_scale_bytes), "cudaMalloc(task_scale)");
    d_task_scale_dev.reset(d_task_scale_raw);
    throw_on_cuda_error(
        cudaMemcpy(d_task_scale_dev.get(), h_scale_ptr, task_scale_bytes, cudaMemcpyHostToDevice), "H2D task_scale");
  }

  if (h_y0_ptr) {
    throw_on_cuda_error(cudaMemcpy(d_y.get(), h_y0_ptr, y_bytes, cudaMemcpyHostToDevice), "H2D y0");
  } else {
    throw_on_cuda_error(cudaMemset(d_y.get(), 0, y_bytes), "cudaMemset(y=0)");
  }

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_apply_g_flat_scatter_atomic_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_scale_dev ? d_task_scale_dev.get() : nullptr,
      d_g.get(),
      g_stride,
      ntasks,
      d_y.get(),
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(apply_g_flat_atomic)");

  int h_overflow = 0;
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("apply_g_flat_scatter_atomic overflow (invalid indices or stack overflow)");
  }

  if (!return_y) return py::none();

  py::array_t<double> out_y({state.ncsf});
  throw_on_cuda_error(cudaMemcpy(out_y.mutable_data(), d_y.get(), y_bytes, cudaMemcpyDeviceToHost), "D2H y");
  return std::move(out_y);
}

py::tuple kernel25_build_csr_from_triples_cuda(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> j_out,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> k_out,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> rs_out,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_out,
    int nops,
    bool coalesce) {
  auto j_buf = j_out.request();
  auto k_buf = k_out.request();
  auto rs_buf = rs_out.request();
  auto c_buf = c_out.request();

  if (j_buf.ndim != 1 || k_buf.ndim != 1 || rs_buf.ndim != 1 || c_buf.ndim != 1) {
    throw std::invalid_argument("j_out/k_out/rs_out/c_out must be 1D arrays");
  }
  if (j_buf.size != k_buf.size || j_buf.size != rs_buf.size || j_buf.size != c_buf.size) {
    throw std::invalid_argument("j_out/k_out/rs_out/c_out must have the same length");
  }

  nops = int(nops);
  if (nops <= 0) {
    throw std::invalid_argument("nops must be > 0");
  }

  py::ssize_t nnz_py = j_buf.size;
  if (nnz_py <= 0) {
    py::array_t<int32_t> row_j({0});
    py::array_t<int32_t> row_k({0});
    py::array_t<int64_t> indptr({1});
    indptr.mutable_data()[0] = 0;
    py::array_t<int32_t> indices({0});
    py::array_t<double> data({0});
    return py::make_tuple(row_j, row_k, indptr, indices, data);
  }
  if (nnz_py > (py::ssize_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("nnz too large for current CUDA Kernel 2.5 builder");
  }
  int nnz = (int)nnz_py;

  const int32_t* h_j = static_cast<const int32_t*>(j_buf.ptr);
  const int32_t* h_k = static_cast<const int32_t*>(k_buf.ptr);
  const int32_t* h_rs = static_cast<const int32_t*>(rs_buf.ptr);
  const double* h_c = static_cast<const double*>(c_buf.ptr);

  // Basic validation: ensure rs indices are within [0,nops).
  for (py::ssize_t i = 0; i < nnz_py; i++) {
    int32_t rs_id = h_rs[i];
    if (rs_id < 0 || rs_id >= (int32_t)nops) {
      throw std::invalid_argument("rs_out contains out-of-range column indices for nops");
    }
  }

  int32_t* d_j_raw = nullptr;
  int32_t* d_k_raw = nullptr;
  int32_t* d_rs_raw = nullptr;
  double* d_c_raw = nullptr;
  uint64_t* d_row_jk_raw = nullptr;
  int64_t* d_indptr_raw = nullptr;
  int32_t* d_indices_raw = nullptr;
  double* d_data_raw = nullptr;
  int* d_nrows_raw = nullptr;
  int* d_nnz_raw = nullptr;

  cuda_unique_ptr<int32_t> d_j;
  cuda_unique_ptr<int32_t> d_k;
  cuda_unique_ptr<int32_t> d_rs;
  cuda_unique_ptr<double> d_c;
  cuda_unique_ptr<uint64_t> d_row_jk;
  cuda_unique_ptr<int64_t> d_indptr;
  cuda_unique_ptr<int32_t> d_indices;
  cuda_unique_ptr<double> d_data;
  cuda_unique_ptr<int> d_nrows;
  cuda_unique_ptr<int> d_nnz;

  size_t i_bytes = (size_t)nnz * sizeof(int32_t);
  size_t v_bytes = (size_t)nnz * sizeof(double);
  size_t row_jk_bytes = (size_t)nnz * sizeof(uint64_t);
  size_t indptr_bytes = (size_t)(nnz + 1) * sizeof(int64_t);

  throw_on_cuda_error(cudaMalloc((void**)&d_j_raw, i_bytes), "cudaMalloc(j_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_k_raw, i_bytes), "cudaMalloc(k_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_rs_raw, i_bytes), "cudaMalloc(rs_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_c_raw, v_bytes), "cudaMalloc(c_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_row_jk_raw, row_jk_bytes), "cudaMalloc(row_jk)");
  throw_on_cuda_error(cudaMalloc((void**)&d_indptr_raw, indptr_bytes), "cudaMalloc(indptr)");
  throw_on_cuda_error(cudaMalloc((void**)&d_indices_raw, i_bytes), "cudaMalloc(indices)");
  throw_on_cuda_error(cudaMalloc((void**)&d_data_raw, v_bytes), "cudaMalloc(data)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nrows_raw, sizeof(int)), "cudaMalloc(nrows)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nnz_raw, sizeof(int)), "cudaMalloc(nnz_out)");

  d_j.reset(d_j_raw);
  d_k.reset(d_k_raw);
  d_rs.reset(d_rs_raw);
  d_c.reset(d_c_raw);
  d_row_jk.reset(d_row_jk_raw);
  d_indptr.reset(d_indptr_raw);
  d_indices.reset(d_indices_raw);
  d_data.reset(d_data_raw);
  d_nrows.reset(d_nrows_raw);
  d_nnz.reset(d_nnz_raw);

  throw_on_cuda_error(cudaMemcpy(d_j.get(), h_j, i_bytes, cudaMemcpyHostToDevice), "H2D j_out");
  throw_on_cuda_error(cudaMemcpy(d_k.get(), h_k, i_bytes, cudaMemcpyHostToDevice), "H2D k_out");
  throw_on_cuda_error(cudaMemcpy(d_rs.get(), h_rs, i_bytes, cudaMemcpyHostToDevice), "H2D rs_out");
  throw_on_cuda_error(cudaMemcpy(d_c.get(), h_c, v_bytes, cudaMemcpyHostToDevice), "H2D c_out");

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_nrows.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D nrows=0");
  throw_on_cuda_error(cudaMemcpy(d_nnz.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D nnz=0");

  throw_on_cuda_error(
      guga_kernel25_build_csr_launch(
          d_j.get(),
          d_k.get(),
          d_rs.get(),
          d_c.get(),
          nnz,
          coalesce ? 1 : 0,
          d_row_jk.get(),
          d_indptr.get(),
          d_indices.get(),
          d_data.get(),
          d_nrows.get(),
          d_nnz.get()),
      "guga_kernel25_build_csr_launch");
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel25)");

  int h_nrows = 0;
  int h_nnz = 0;
  throw_on_cuda_error(cudaMemcpy(&h_nrows, d_nrows.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H nrows");
  throw_on_cuda_error(cudaMemcpy(&h_nnz, d_nnz.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H nnz");
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;

  std::vector<uint64_t> h_row_jk((size_t)h_nrows);
  if (h_nrows > 0) {
    throw_on_cuda_error(
        cudaMemcpy(h_row_jk.data(), d_row_jk.get(), (size_t)h_nrows * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "D2H row_jk");
  }

  py::array_t<int32_t> row_j({h_nrows});
  py::array_t<int32_t> row_k({h_nrows});
  int32_t* row_j_ptr = row_j.mutable_data();
  int32_t* row_k_ptr = row_k.mutable_data();
  for (int i = 0; i < h_nrows; i++) {
    uint64_t key = h_row_jk[(size_t)i];
    uint32_t ju = (uint32_t)(key >> 32);
    uint32_t ku = (uint32_t)key;
    row_j_ptr[i] = (int32_t)ju;
    row_k_ptr[i] = (int32_t)ku;
  }

  py::array_t<int64_t> indptr({(py::ssize_t)h_nrows + 1});
  if (h_nrows > 0) {
    throw_on_cuda_error(
        cudaMemcpy(indptr.mutable_data(), d_indptr.get(), (size_t)(h_nrows + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost),
        "D2H indptr");
  } else {
    indptr.mutable_data()[0] = 0;
  }

  py::array_t<int32_t> indices({h_nnz});
  py::array_t<double> data({h_nnz});
  if (h_nnz > 0) {
    throw_on_cuda_error(
        cudaMemcpy(indices.mutable_data(), d_indices.get(), (size_t)h_nnz * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "D2H indices");
    throw_on_cuda_error(
        cudaMemcpy(data.mutable_data(), d_data.get(), (size_t)h_nnz * sizeof(double), cudaMemcpyDeviceToHost),
        "D2H data");
  }

  return py::make_tuple(row_j, row_k, indptr, indices, data);
}

py::tuple kernel25_build_csr_from_tasks_deterministic_cuda(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_csf,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_p,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> task_q,
    int threads,
    int64_t max_total_out,
    bool coalesce) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto csf_buf = task_csf.request();
  auto p_buf = task_p.request();
  auto q_buf = task_q.request();
  if (csf_buf.ndim != 1 || p_buf.ndim != 1 || q_buf.ndim != 1) {
    throw std::invalid_argument("task arrays must be 1D");
  }
  if (csf_buf.size != p_buf.size || csf_buf.size != q_buf.size) {
    throw std::invalid_argument("task arrays must have the same length");
  }

  py::ssize_t ntasks_py = csf_buf.size;
  if (ntasks_py <= 0) {
    py::array_t<int32_t> row_j({0});
    py::array_t<int32_t> row_k({0});
    py::array_t<int64_t> indptr({1});
    indptr.mutable_data()[0] = 0;
    py::array_t<int32_t> indices({0});
    py::array_t<double> data({0});
    return py::make_tuple(row_j, row_k, indptr, indices, data);
  }
  if (ntasks_py > (py::ssize_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("ntasks too large for current CUDA Kernel 2B path (batch the tasks)");
  }
  int ntasks = (int)ntasks_py;

  const int32_t* h_task_csf = static_cast<const int32_t*>(csf_buf.ptr);
  const int32_t* h_task_p = static_cast<const int32_t*>(p_buf.ptr);
  const int32_t* h_task_q = static_cast<const int32_t*>(q_buf.ptr);

  size_t task_i_bytes = (size_t)ntasks * sizeof(int32_t);
  size_t offsets_bytes = (size_t)(ntasks + 1) * sizeof(int64_t);

  int32_t* d_task_csf_raw = nullptr;
  int32_t* d_task_p_raw = nullptr;
  int32_t* d_task_q_raw = nullptr;
  int32_t* d_counts_raw = nullptr;
  int64_t* d_offsets_raw = nullptr;
  int* d_overflow_raw = nullptr;

  cuda_unique_ptr<int32_t> d_task_csf;
  cuda_unique_ptr<int32_t> d_task_p;
  cuda_unique_ptr<int32_t> d_task_q;
  cuda_unique_ptr<int32_t> d_counts;
  cuda_unique_ptr<int64_t> d_offsets;
  cuda_unique_ptr<int> d_overflow;

  throw_on_cuda_error(cudaMalloc((void**)&d_task_csf_raw, task_i_bytes), "cudaMalloc(task_csf)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_p_raw, task_i_bytes), "cudaMalloc(task_p)");
  throw_on_cuda_error(cudaMalloc((void**)&d_task_q_raw, task_i_bytes), "cudaMalloc(task_q)");
  throw_on_cuda_error(cudaMalloc((void**)&d_counts_raw, task_i_bytes), "cudaMalloc(counts)");
  throw_on_cuda_error(cudaMalloc((void**)&d_offsets_raw, offsets_bytes), "cudaMalloc(offsets)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow_raw, sizeof(int)), "cudaMalloc(overflow)");

  d_task_csf.reset(d_task_csf_raw);
  d_task_p.reset(d_task_p_raw);
  d_task_q.reset(d_task_q_raw);
  d_counts.reset(d_counts_raw);
  d_offsets.reset(d_offsets_raw);
  d_overflow.reset(d_overflow_raw);

  throw_on_cuda_error(cudaMemcpy(d_task_csf.get(), h_task_csf, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_csf");
  throw_on_cuda_error(cudaMemcpy(d_task_p.get(), h_task_p, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_p");
  throw_on_cuda_error(cudaMemcpy(d_task_q.get(), h_task_q, task_i_bytes, cudaMemcpyHostToDevice), "H2D task_q");

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_epq_contribs_many_count_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_p.get(),
      d_task_q.get(),
      ntasks,
      d_counts.get(),
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(count)");

  int h_overflow = 0;
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
  }

  int64_t total_out = 0;
  throw_on_cuda_error(
      guga_counts_to_offsets_exclusive_scan_launch(d_counts.get(), ntasks, d_offsets.get(), &total_out),
      "guga_counts_to_offsets_exclusive_scan_launch");

  if (max_total_out >= 0 && total_out > max_total_out) {
    throw std::runtime_error("Kernel 2B total output exceeds max_total_out");
  }

  if (total_out <= 0) {
    py::array_t<int32_t> row_j({0});
    py::array_t<int32_t> row_k({0});
    py::array_t<int64_t> indptr({1});
    indptr.mutable_data()[0] = 0;
    py::array_t<int32_t> indices({0});
    py::array_t<double> data({0});
    return py::make_tuple(row_j, row_k, indptr, indices, data);
  }

  if (total_out > (int64_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("Kernel 2B output too large for current Kernel 2.5 builder (batch the tasks)");
  }
  int nnz_in = (int)total_out;

  int32_t* d_j_out_raw = nullptr;
  int32_t* d_k_out_raw = nullptr;
  int32_t* d_pq_out_raw = nullptr;
  double* d_c_out_raw = nullptr;
  throw_on_cuda_error(cudaMalloc((void**)&d_j_out_raw, (size_t)nnz_in * sizeof(int32_t)), "cudaMalloc(j_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_k_out_raw, (size_t)nnz_in * sizeof(int32_t)), "cudaMalloc(k_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_pq_out_raw, (size_t)nnz_in * sizeof(int32_t)), "cudaMalloc(pq_out)");
  throw_on_cuda_error(cudaMalloc((void**)&d_c_out_raw, (size_t)nnz_in * sizeof(double)), "cudaMalloc(c_out)");
  cuda_unique_ptr<int32_t> d_j_out(d_j_out_raw);
  cuda_unique_ptr<int32_t> d_k_out(d_k_out_raw);
  cuda_unique_ptr<int32_t> d_pq_out(d_pq_out_raw);
  cuda_unique_ptr<double> d_c_out(d_c_out_raw);

  throw_on_cuda_error(cudaMemcpy(d_overflow.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow=0");

  guga_epq_contribs_many_write_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf.get(),
      d_task_p.get(),
      d_task_q.get(),
      ntasks,
      d_offsets.get(),
      d_k_out.get(),
      d_c_out.get(),
      d_j_out.get(),
      d_pq_out.get(),
      d_overflow.get(),
      threads);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(write_triples)");

  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  if (h_overflow) {
    throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
  }

  uint64_t* d_row_jk_raw = nullptr;
  int64_t* d_indptr_raw = nullptr;
  int32_t* d_indices_raw = nullptr;
  double* d_data_raw = nullptr;
  int* d_nrows_raw = nullptr;
  int* d_nnz_raw = nullptr;

  cuda_unique_ptr<uint64_t> d_row_jk;
  cuda_unique_ptr<int64_t> d_indptr;
  cuda_unique_ptr<int32_t> d_indices;
  cuda_unique_ptr<double> d_data;
  cuda_unique_ptr<int> d_nrows;
  cuda_unique_ptr<int> d_nnz;

  throw_on_cuda_error(cudaMalloc((void**)&d_row_jk_raw, (size_t)nnz_in * sizeof(uint64_t)), "cudaMalloc(row_jk)");
  throw_on_cuda_error(cudaMalloc((void**)&d_indptr_raw, (size_t)(nnz_in + 1) * sizeof(int64_t)), "cudaMalloc(indptr)");
  throw_on_cuda_error(cudaMalloc((void**)&d_indices_raw, (size_t)nnz_in * sizeof(int32_t)), "cudaMalloc(indices)");
  throw_on_cuda_error(cudaMalloc((void**)&d_data_raw, (size_t)nnz_in * sizeof(double)), "cudaMalloc(data)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nrows_raw, sizeof(int)), "cudaMalloc(nrows)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nnz_raw, sizeof(int)), "cudaMalloc(nnz_out)");

  d_row_jk.reset(d_row_jk_raw);
  d_indptr.reset(d_indptr_raw);
  d_indices.reset(d_indices_raw);
  d_data.reset(d_data_raw);
  d_nrows.reset(d_nrows_raw);
  d_nnz.reset(d_nnz_raw);

  throw_on_cuda_error(cudaMemcpy(d_nrows.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D nrows=0");
  throw_on_cuda_error(cudaMemcpy(d_nnz.get(), &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D nnz=0");

  throw_on_cuda_error(
      guga_kernel25_build_csr_launch(
          d_j_out.get(),
          d_k_out.get(),
          d_pq_out.get(),
          d_c_out.get(),
          nnz_in,
          coalesce ? 1 : 0,
          d_row_jk.get(),
          d_indptr.get(),
          d_indices.get(),
          d_data.get(),
          d_nrows.get(),
          d_nnz.get()),
      "guga_kernel25_build_csr_launch(fused)");
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel25_fused)");

  int h_nrows = 0;
  int h_nnz = 0;
  throw_on_cuda_error(cudaMemcpy(&h_nrows, d_nrows.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H nrows");
  throw_on_cuda_error(cudaMemcpy(&h_nnz, d_nnz.get(), sizeof(int), cudaMemcpyDeviceToHost), "D2H nnz");
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;

  std::vector<uint64_t> h_row_jk((size_t)h_nrows);
  if (h_nrows > 0) {
    throw_on_cuda_error(
        cudaMemcpy(h_row_jk.data(), d_row_jk.get(), (size_t)h_nrows * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "D2H row_jk");
  }

  py::array_t<int32_t> row_j({h_nrows});
  py::array_t<int32_t> row_k({h_nrows});
  int32_t* row_j_ptr = row_j.mutable_data();
  int32_t* row_k_ptr = row_k.mutable_data();
  for (int i = 0; i < h_nrows; i++) {
    uint64_t key = h_row_jk[(size_t)i];
    row_j_ptr[i] = (int32_t)((uint32_t)(key >> 32));
    row_k_ptr[i] = (int32_t)((uint32_t)key);
  }

  py::array_t<int64_t> indptr({(py::ssize_t)h_nrows + 1});
  if (h_nrows > 0) {
    throw_on_cuda_error(
        cudaMemcpy(indptr.mutable_data(), d_indptr.get(), (size_t)(h_nrows + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost),
        "D2H indptr");
  } else {
    indptr.mutable_data()[0] = 0;
  }

  py::array_t<int32_t> indices({h_nnz});
  py::array_t<double> data({h_nnz});
  if (h_nnz > 0) {
    throw_on_cuda_error(
        cudaMemcpy(indices.mutable_data(), d_indices.get(), (size_t)h_nnz * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "D2H indices");
    throw_on_cuda_error(
        cudaMemcpy(data.mutable_data(), d_data.get(), (size_t)h_nnz * sizeof(double), cudaMemcpyDeviceToHost),
        "D2H data");
  }

  return py::make_tuple(row_j, row_k, indptr, indices, data);
}

py::tuple kernel25_build_csr_from_tasks_deterministic_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object task_csf,
    py::object task_p,
    py::object task_q,
    py::object row_j,
    py::object row_k,
    py::object indptr,
    py::object indices,
    py::object data,
    py::object overflow,
    int threads,
    bool coalesce,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  if (task_csf.is_none() || task_p.is_none() || task_q.is_none()) {
    throw std::invalid_argument("task arrays must be device arrays (cannot be None)");
  }
  if (row_j.is_none() || row_k.is_none() || indptr.is_none() || indices.is_none() || data.is_none() || overflow.is_none()) {
    throw std::invalid_argument("output arrays must be device arrays (cannot be None)");
  }

  auto task_csf_dev = cuda_array_view_from_object(task_csf, "task_csf");
  auto task_p_dev = cuda_array_view_from_object(task_p, "task_p");
  auto task_q_dev = cuda_array_view_from_object(task_q, "task_q");
  require_typestr(task_csf_dev, "task_csf", "<i4");
  require_typestr(task_p_dev, "task_p", "<i4");
  require_typestr(task_q_dev, "task_q", "<i4");
  if (task_csf_dev.shape.size() != 1 || task_p_dev.shape.size() != 1 || task_q_dev.shape.size() != 1) {
    throw std::invalid_argument("task arrays must be 1D device arrays");
  }
  if (task_csf_dev.shape[0] != task_p_dev.shape[0] || task_csf_dev.shape[0] != task_q_dev.shape[0]) {
    throw std::invalid_argument("task arrays must have the same length");
  }
  if (!task_csf_dev.strides_bytes.empty() &&
      (task_csf_dev.strides_bytes.size() != 1 || task_csf_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_csf must be contiguous");
  }
  if (!task_p_dev.strides_bytes.empty() &&
      (task_p_dev.strides_bytes.size() != 1 || task_p_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_p must be contiguous");
  }
  if (!task_q_dev.strides_bytes.empty() &&
      (task_q_dev.strides_bytes.size() != 1 || task_q_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("task_q must be contiguous");
  }

  int64_t ntasks_ll = task_csf_dev.shape[0];
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range (batch the work)");
  }
  int ntasks = (int)ntasks_ll;
  if (ntasks == 0) {
    return py::make_tuple(py::int_(0), py::int_(0), py::int_(0));
  }

  auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
  auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
  auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
  auto indices_dev = cuda_array_view_from_object(indices, "indices");
  auto data_dev = cuda_array_view_from_object(data, "data");
  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");

  require_typestr(row_j_dev, "row_j", "<i4");
  require_typestr(row_k_dev, "row_k", "<i4");
  require_typestr(indptr_dev, "indptr", "<i8");
  require_typestr(indices_dev, "indices", "<i4");
  int out_data_type = 0;
  int64_t out_data_itemsize = 0;
  if (data_dev.typestr == "<f8") {
    out_data_type = 8;
    out_data_itemsize = (int64_t)sizeof(double);
  } else if (data_dev.typestr == "<f4") {
    out_data_type = 4;
    out_data_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
  }
  require_typestr(overflow_dev, "overflow", "<i4");

  if (row_j_dev.read_only || row_k_dev.read_only || indptr_dev.read_only || indices_dev.read_only || data_dev.read_only ||
      overflow_dev.read_only) {
    throw std::invalid_argument("output arrays must be writable");
  }

  if (row_j_dev.shape.size() != 1 || row_k_dev.shape.size() != 1 || indptr_dev.shape.size() != 1 ||
      indices_dev.shape.size() != 1 || data_dev.shape.size() != 1) {
    throw std::invalid_argument("output arrays must be 1D device arrays");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }

  if (indices_dev.shape[0] != data_dev.shape[0]) {
    throw std::invalid_argument("indices and data must have the same length");
  }

  int64_t capacity = indices_dev.shape[0];
  if (capacity <= 0 || capacity > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("capacity out of supported range");
  }
  int max_out = (int)capacity;

  if (row_j_dev.shape[0] != capacity || row_k_dev.shape[0] != capacity) {
    throw std::invalid_argument("row_j/row_k must have shape (capacity,) matching indices/data length");
  }
  if (indptr_dev.shape[0] != capacity + 1) {
    throw std::invalid_argument("indptr must have shape (capacity+1,)");
  }

  if (!row_j_dev.strides_bytes.empty() &&
      (row_j_dev.strides_bytes.size() != 1 || row_j_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("row_j must be contiguous");
  }
  if (!row_k_dev.strides_bytes.empty() &&
      (row_k_dev.strides_bytes.size() != 1 || row_k_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("row_k must be contiguous");
  }
  if (!indptr_dev.strides_bytes.empty() &&
      (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t))) {
    throw std::invalid_argument("indptr must be contiguous");
  }
  if (!indices_dev.strides_bytes.empty() &&
      (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("indices must be contiguous");
  }
  if (!data_dev.strides_bytes.empty() &&
      (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != out_data_itemsize)) {
    throw std::invalid_argument("data must be contiguous");
  }
  if (!overflow_dev.strides_bytes.empty() &&
      (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("overflow must be contiguous");
  }

  const int32_t* d_task_csf = reinterpret_cast<const int32_t*>(task_csf_dev.ptr);
  const int32_t* d_task_p = reinterpret_cast<const int32_t*>(task_p_dev.ptr);
  const int32_t* d_task_q = reinterpret_cast<const int32_t*>(task_q_dev.ptr);

  int32_t* d_row_j = reinterpret_cast<int32_t*>(row_j_dev.ptr);
  int32_t* d_row_k = reinterpret_cast<int32_t*>(row_k_dev.ptr);
  int64_t* d_indptr = reinterpret_cast<int64_t*>(indptr_dev.ptr);
  int32_t* d_indices = reinterpret_cast<int32_t*>(indices_dev.ptr);
  void* d_data = reinterpret_cast<void*>(data_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (data_dev.stream) stream_u = data_dev.stream;
    else if (indices_dev.stream) stream_u = indices_dev.stream;
    else if (indptr_dev.stream) stream_u = indptr_dev.stream;
    else if (row_j_dev.stream) stream_u = row_j_dev.stream;
    else if (row_k_dev.stream) stream_u = row_k_dev.stream;
    else if (task_q_dev.stream) stream_u = task_q_dev.stream;
    else if (task_p_dev.stream) stream_u = task_p_dev.stream;
    else if (task_csf_dev.stream) stream_u = task_csf_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  if (out_data_type == 4) {
    void* ws_handle = guga_kernel25_workspace_create(ntasks, max_out);
    if (!ws_handle) {
      throw std::runtime_error("failed to create Kernel25Workspace for float32 task CSR build");
    }
    int h_nrows = 0;
    int h_nnz = 0;
    int h_nnz_in = 0;
    try {
      guga_kernel25_build_csr_from_tasks_deterministic_inplace_device_ws(
          ws_handle,
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          d_task_csf,
          d_task_p,
          d_task_q,
          ntasks,
          d_row_j,
          d_row_k,
          d_indptr,
          d_indices,
          d_data,
          out_data_type,
          max_out,
          d_overflow,
          stream_t,
          int(threads),
          coalesce ? 1 : 0,
          &h_nrows,
          &h_nnz,
          &h_nnz_in,
          sync ? 1 : 0,
          check_overflow ? 1 : 0);
    } catch (...) {
      guga_kernel25_workspace_destroy(ws_handle);
      throw;
    }
    guga_kernel25_workspace_destroy(ws_handle);
    return py::make_tuple(py::int_(h_nrows), py::int_(h_nnz), py::int_(h_nnz_in));
  }

  double* d_data_f64 = reinterpret_cast<double*>(d_data);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  // Count + offsets
  int32_t* d_counts_raw = nullptr;
  int64_t* d_offsets_raw = nullptr;
  throw_on_cuda_error(
      guga_cuda_malloc(&d_counts_raw, (size_t)ntasks * sizeof(int32_t), stream_t), "cudaMallocAsync(counts)");
  throw_on_cuda_error(
      guga_cuda_malloc(&d_offsets_raw, (size_t)(ntasks + 1) * sizeof(int64_t), stream_t), "cudaMallocAsync(offsets)");
  cuda_unique_ptr_stream<int32_t> d_counts(d_counts_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<int64_t> d_offsets(d_offsets_raw, CudaFreeStreamDeleter<int64_t>{stream_t});

  guga_epq_contribs_many_count_launch_stream(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf,
      d_task_p,
      d_task_q,
      ntasks,
      d_counts.get(),
      d_overflow,
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel2b_count)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H overflow(count)");
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel2b_count)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
    }
  }

  int64_t total_out_ll = 0;
  throw_on_cuda_error(
      guga_counts_to_offsets_exclusive_scan_launch_stream(d_counts.get(), ntasks, d_offsets.get(), &total_out_ll, stream_t),
      "guga_counts_to_offsets_exclusive_scan_launch_stream");
  if (total_out_ll <= 0) {
    return py::make_tuple(py::int_(0), py::int_(0), py::int_(0));
  }
  if (total_out_ll > (int64_t)max_out) {
    throw std::runtime_error("Kernel 2B total output exceeds output buffer capacity");
  }
  if (total_out_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("Kernel 2B total output too large (batch the work)");
  }
  int nnz_in = (int)total_out_ll;

  // Write triples into temporary arrays.
  int32_t* d_k_out_raw = nullptr;
  double* d_c_out_raw = nullptr;
  int32_t* d_j_out_raw = nullptr;
  int32_t* d_pq_out_raw = nullptr;
  throw_on_cuda_error(guga_cuda_malloc(&d_k_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(k_out)");
  throw_on_cuda_error(guga_cuda_malloc(&d_c_out_raw, (size_t)nnz_in * sizeof(double), stream_t), "cudaMallocAsync(c_out)");
  throw_on_cuda_error(guga_cuda_malloc(&d_j_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(j_out)");
  throw_on_cuda_error(
      guga_cuda_malloc(&d_pq_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(pq_out)");
  cuda_unique_ptr_stream<int32_t> d_k_out(d_k_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<double> d_c_out(d_c_out_raw, CudaFreeStreamDeleter<double>{stream_t});
  cuda_unique_ptr_stream<int32_t> d_j_out(d_j_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<int32_t> d_pq_out(d_pq_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  guga_epq_contribs_many_write_launch_stream(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      d_task_csf,
      d_task_p,
      d_task_q,
      ntasks,
      d_offsets.get(),
      d_k_out.get(),
      d_c_out.get(),
      d_j_out.get(),
      d_pq_out.get(),
      d_overflow,
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel2b_write)");

  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H overflow(write)");
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel2b_write)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
    }
  }

  // Kernel 2.5: build CSR in caller-provided output arrays.
  uint64_t* d_row_jk_raw = nullptr;
  int* d_nrows_raw = nullptr;
  int* d_nnz_raw = nullptr;
  throw_on_cuda_error(
      guga_cuda_malloc(&d_row_jk_raw, (size_t)nnz_in * sizeof(uint64_t), stream_t), "cudaMallocAsync(row_jk)");
  throw_on_cuda_error(guga_cuda_malloc(&d_nrows_raw, sizeof(int), stream_t), "cudaMallocAsync(nrows)");
  throw_on_cuda_error(guga_cuda_malloc(&d_nnz_raw, sizeof(int), stream_t), "cudaMallocAsync(nnz_out)");
  cuda_unique_ptr_stream<uint64_t> d_row_jk(d_row_jk_raw, CudaFreeStreamDeleter<uint64_t>{stream_t});
  cuda_unique_ptr_stream<int> d_nrows(d_nrows_raw, CudaFreeStreamDeleter<int>{stream_t});
  cuda_unique_ptr_stream<int> d_nnz(d_nnz_raw, CudaFreeStreamDeleter<int>{stream_t});

  throw_on_cuda_error(cudaMemsetAsync(d_nrows.get(), 0, sizeof(int), stream_t), "cudaMemsetAsync(nrows=0)");
  throw_on_cuda_error(cudaMemsetAsync(d_nnz.get(), 0, sizeof(int), stream_t), "cudaMemsetAsync(nnz=0)");

  throw_on_cuda_error(
      guga_kernel25_build_csr_launch_stream(
          d_j_out.get(),
          d_k_out.get(),
          d_pq_out.get(),
          d_c_out.get(),
          nnz_in,
          coalesce ? 1 : 0,
          d_row_jk.get(),
          d_indptr,
          d_indices,
          d_data_f64,
          d_nrows.get(),
          d_nnz.get(),
          stream_t),
      "guga_kernel25_build_csr_launch_stream(device)");

  int h_nrows = 0;
  int h_nnz = 0;
  throw_on_cuda_error(cudaMemcpyAsync(&h_nrows, d_nrows.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H nrows");
  throw_on_cuda_error(cudaMemcpyAsync(&h_nnz, d_nnz.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H nnz");
  throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel25_device)");
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;
  if (h_nrows > nnz_in || h_nrows > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nrows out of range for output buffers");
  }
  if (h_nnz > nnz_in || h_nnz > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nnz out of range for output buffers");
  }

  throw_on_cuda_error(
      guga_unpack_row_jk_launch_stream(d_row_jk.get(), h_nrows, d_row_j, d_row_k, stream_t, /*threads=*/256),
      "guga_unpack_row_jk_launch_stream");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(unpack_row_jk)");
  }

  return py::make_tuple(py::int_(h_nrows), py::int_(h_nnz), py::int_(nnz_in));
}

py::tuple kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object row_j,
    py::object row_k,
    py::object indptr,
    py::object indices,
    py::object data,
    py::object overflow,
    int threads,
    bool coalesce,
    uint64_t stream,
    bool sync,
    bool check_overflow) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (check_overflow && !sync) {
    throw std::invalid_argument("check_overflow=True requires sync=True");
  }

  if (row_j.is_none() || row_k.is_none() || indptr.is_none() || indices.is_none() || data.is_none() || overflow.is_none()) {
    throw std::invalid_argument("output arrays must be device arrays (cannot be None)");
  }

  int norb = int(drt.norb);
  int ncsf = int(state.ncsf);
  if (j_start < 0 || j_count < 0 || j_start > ncsf || j_start + j_count > ncsf) {
    throw std::invalid_argument("invalid (j_start, j_count) for this DRT");
  }
  if (j_count == 0) {
    return py::make_tuple(py::int_(0), py::int_(0), py::int_(0));
  }
  int n_pairs = norb * (norb - 1);
  if (n_pairs <= 0) {
    return py::make_tuple(py::int_(0), py::int_(0), py::int_(0));
  }

  int64_t ntasks_ll = (int64_t)j_count * (int64_t)n_pairs;
  if (ntasks_ll < 0 || ntasks_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("ntasks out of supported range (batch the work)");
  }
  int ntasks = (int)ntasks_ll;

  auto row_j_dev = cuda_array_view_from_object(row_j, "row_j");
  auto row_k_dev = cuda_array_view_from_object(row_k, "row_k");
  auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
  auto indices_dev = cuda_array_view_from_object(indices, "indices");
  auto data_dev = cuda_array_view_from_object(data, "data");
  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");

  require_typestr(row_j_dev, "row_j", "<i4");
  require_typestr(row_k_dev, "row_k", "<i4");
  require_typestr(indptr_dev, "indptr", "<i8");
  require_typestr(indices_dev, "indices", "<i4");
  require_typestr(data_dev, "data", "<f8");
  require_typestr(overflow_dev, "overflow", "<i4");

  if (row_j_dev.read_only || row_k_dev.read_only || indptr_dev.read_only || indices_dev.read_only || data_dev.read_only ||
      overflow_dev.read_only) {
    throw std::invalid_argument("output arrays must be writable");
  }

  if (row_j_dev.shape.size() != 1 || row_k_dev.shape.size() != 1 || indptr_dev.shape.size() != 1 ||
      indices_dev.shape.size() != 1 || data_dev.shape.size() != 1) {
    throw std::invalid_argument("output arrays must be 1D device arrays");
  }
  if (overflow_dev.shape.size() != 1 || overflow_dev.shape[0] != 1) {
    throw std::invalid_argument("overflow must have shape (1,)");
  }
  if (indices_dev.shape[0] != data_dev.shape[0]) {
    throw std::invalid_argument("indices and data must have the same length");
  }

  int64_t capacity = indices_dev.shape[0];
  if (capacity <= 0 || capacity > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("capacity out of supported range");
  }
  int max_out = (int)capacity;
  if (row_j_dev.shape[0] != capacity || row_k_dev.shape[0] != capacity) {
    throw std::invalid_argument("row_j/row_k must have shape (capacity,) matching indices/data length");
  }
  if (indptr_dev.shape[0] != capacity + 1) {
    throw std::invalid_argument("indptr must have shape (capacity+1,)");
  }

  if (!row_j_dev.strides_bytes.empty() &&
      (row_j_dev.strides_bytes.size() != 1 || row_j_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("row_j must be contiguous");
  }
  if (!row_k_dev.strides_bytes.empty() &&
      (row_k_dev.strides_bytes.size() != 1 || row_k_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("row_k must be contiguous");
  }
  if (!indptr_dev.strides_bytes.empty() &&
      (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t))) {
    throw std::invalid_argument("indptr must be contiguous");
  }
  if (!indices_dev.strides_bytes.empty() &&
      (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("indices must be contiguous");
  }
  if (!data_dev.strides_bytes.empty() && (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != (int64_t)sizeof(double))) {
    throw std::invalid_argument("data must be contiguous");
  }
  if (!overflow_dev.strides_bytes.empty() &&
      (overflow_dev.strides_bytes.size() != 1 || overflow_dev.strides_bytes[0] != (int64_t)sizeof(int32_t))) {
    throw std::invalid_argument("overflow must be contiguous");
  }

  int32_t* d_row_j = reinterpret_cast<int32_t*>(row_j_dev.ptr);
  int32_t* d_row_k = reinterpret_cast<int32_t*>(row_k_dev.ptr);
  int64_t* d_indptr = reinterpret_cast<int64_t*>(indptr_dev.ptr);
  int32_t* d_indices = reinterpret_cast<int32_t*>(indices_dev.ptr);
  double* d_data = reinterpret_cast<double*>(data_dev.ptr);
  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (data_dev.stream) stream_u = data_dev.stream;
    else if (indices_dev.stream) stream_u = indices_dev.stream;
    else if (indptr_dev.stream) stream_u = indptr_dev.stream;
    else if (row_j_dev.stream) stream_u = row_j_dev.stream;
    else if (row_k_dev.stream) stream_u = row_k_dev.stream;
    else if (overflow_dev.stream) stream_u = overflow_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");

  // Count + offsets
  int32_t* d_counts_raw = nullptr;
  int64_t* d_offsets_raw = nullptr;
  throw_on_cuda_error(
      guga_cuda_malloc(&d_counts_raw, (size_t)ntasks * sizeof(int32_t), stream_t), "cudaMallocAsync(counts)");
  throw_on_cuda_error(
      guga_cuda_malloc(&d_offsets_raw, (size_t)(ntasks + 1) * sizeof(int64_t), stream_t), "cudaMallocAsync(offsets)");
  cuda_unique_ptr_stream<int32_t> d_counts(d_counts_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<int64_t> d_offsets(d_offsets_raw, CudaFreeStreamDeleter<int64_t>{stream_t});

  guga_epq_contribs_many_count_allpairs_launch_stream(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      int(j_start),
      int(j_count),
      d_counts.get(),
      d_overflow,
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel2b_count_allpairs)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H overflow(count)");
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel2b_count)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B count kernel overflow (invalid indices or stack overflow)");
    }
  }

  int64_t total_out_ll = 0;
  throw_on_cuda_error(
      guga_counts_to_offsets_exclusive_scan_launch_stream(d_counts.get(), ntasks, d_offsets.get(), &total_out_ll, stream_t),
      "guga_counts_to_offsets_exclusive_scan_launch_stream");
  if (total_out_ll <= 0) {
    return py::make_tuple(py::int_(0), py::int_(0), py::int_(0));
  }
  if (total_out_ll > (int64_t)max_out) {
    throw std::runtime_error("Kernel 2B total output exceeds output buffer capacity");
  }
  if (total_out_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::runtime_error("Kernel 2B total output too large (batch the work)");
  }
  int nnz_in = (int)total_out_ll;

  // Write triples into temporary arrays.
  int32_t* d_k_out_raw = nullptr;
  double* d_c_out_raw = nullptr;
  int32_t* d_j_out_raw = nullptr;
  int32_t* d_pq_out_raw = nullptr;
  throw_on_cuda_error(guga_cuda_malloc(&d_k_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(k_out)");
  throw_on_cuda_error(guga_cuda_malloc(&d_c_out_raw, (size_t)nnz_in * sizeof(double), stream_t), "cudaMallocAsync(c_out)");
  throw_on_cuda_error(guga_cuda_malloc(&d_j_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(j_out)");
  throw_on_cuda_error(guga_cuda_malloc(&d_pq_out_raw, (size_t)nnz_in * sizeof(int32_t), stream_t), "cudaMallocAsync(pq_out)");
  cuda_unique_ptr_stream<int32_t> d_k_out(d_k_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<double> d_c_out(d_c_out_raw, CudaFreeStreamDeleter<double>{stream_t});
  cuda_unique_ptr_stream<int32_t> d_j_out(d_j_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});
  cuda_unique_ptr_stream<int32_t> d_pq_out(d_pq_out_raw, CudaFreeStreamDeleter<int32_t>{stream_t});

  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t), "cudaMemsetAsync(overflow=0)");
  guga_epq_contribs_many_write_allpairs_launch_stream(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      state.steps,
      state.nodes,
      state.ncsf,
      drt.norb,
      int(j_start),
      int(j_count),
      d_offsets.get(),
      d_k_out.get(),
      d_c_out.get(),
      8,
      d_j_out.get(),
      d_pq_out.get(),
      4,
      d_overflow,
      stream_t,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel2b_write_allpairs)");

  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H overflow(write)");
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel2b_write)");
    if (h_overflow) {
      throw std::runtime_error("Kernel 2B write kernel overflow (count/write mismatch or output overflow)");
    }
  }

  // Kernel 2.5: build CSR in caller-provided output arrays.
  uint64_t* d_row_jk_raw = nullptr;
  int* d_nrows_raw = nullptr;
  int* d_nnz_raw = nullptr;
  throw_on_cuda_error(guga_cuda_malloc(&d_row_jk_raw, (size_t)nnz_in * sizeof(uint64_t), stream_t), "cudaMallocAsync(row_jk)");
  throw_on_cuda_error(guga_cuda_malloc(&d_nrows_raw, sizeof(int), stream_t), "cudaMallocAsync(nrows)");
  throw_on_cuda_error(guga_cuda_malloc(&d_nnz_raw, sizeof(int), stream_t), "cudaMallocAsync(nnz_out)");
  cuda_unique_ptr_stream<uint64_t> d_row_jk(d_row_jk_raw, CudaFreeStreamDeleter<uint64_t>{stream_t});
  cuda_unique_ptr_stream<int> d_nrows(d_nrows_raw, CudaFreeStreamDeleter<int>{stream_t});
  cuda_unique_ptr_stream<int> d_nnz(d_nnz_raw, CudaFreeStreamDeleter<int>{stream_t});

  throw_on_cuda_error(cudaMemsetAsync(d_nrows.get(), 0, sizeof(int), stream_t), "cudaMemsetAsync(nrows=0)");
  throw_on_cuda_error(cudaMemsetAsync(d_nnz.get(), 0, sizeof(int), stream_t), "cudaMemsetAsync(nnz=0)");

  throw_on_cuda_error(
      guga_kernel25_build_csr_launch_stream(
          d_j_out.get(),
          d_k_out.get(),
          d_pq_out.get(),
          d_c_out.get(),
          nnz_in,
          coalesce ? 1 : 0,
          d_row_jk.get(),
          d_indptr,
          d_indices,
          d_data,
          d_nrows.get(),
          d_nnz.get(),
          stream_t),
      "guga_kernel25_build_csr_launch_stream(device)");

  int h_nrows = 0;
  int h_nnz = 0;
  throw_on_cuda_error(cudaMemcpyAsync(&h_nrows, d_nrows.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H nrows");
  throw_on_cuda_error(cudaMemcpyAsync(&h_nnz, d_nnz.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_t), "D2H nnz");
  throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel25_device)");
  if (h_nrows < 0) h_nrows = 0;
  if (h_nnz < 0) h_nnz = 0;
  if (h_nrows > nnz_in || h_nrows > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nrows out of range for output buffers");
  }
  if (h_nnz > nnz_in || h_nnz > max_out) {
    throw std::runtime_error("Kernel 2.5 produced nnz out of range for output buffers");
  }

  throw_on_cuda_error(
      guga_unpack_row_jk_launch_stream(d_row_jk.get(), h_nrows, d_row_j, d_row_k, stream_t, /*threads=*/256),
      "guga_unpack_row_jk_launch_stream");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(unpack_row_jk)");
  }

  return py::make_tuple(py::int_(h_nrows), py::int_(h_nnz), py::int_(nnz_in));
}

py::array_t<double> kernel3_build_g_from_csr_eri_mat_cuda(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    py::array_t<double, py::array::c_style | py::array::forcecast> eri_mat,
    int row_start,
    int nrows,
    double half,
    int threads) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }

  auto indptr_buf = indptr.request();
  auto indices_buf = indices.request();
  auto data_buf = data.request();
  auto eri_buf = eri_mat.request();

  if (indptr_buf.ndim != 1) {
    throw std::invalid_argument("indptr must be 1D (nrows+1,)");
  }
  if (indices_buf.ndim != 1 || data_buf.ndim != 1) {
    throw std::invalid_argument("indices and data must be 1D");
  }
  if (indices_buf.shape[0] != data_buf.shape[0]) {
    throw std::invalid_argument("indices and data must have the same length");
  }
  if (eri_buf.ndim != 2 || eri_buf.shape[0] != eri_buf.shape[1]) {
    throw std::invalid_argument("eri_mat must have shape (nops,nops)");
  }

  int64_t nops_ll = (int64_t)eri_buf.shape[0];
  if (nops_ll < 0 || nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops out of supported range");
  }
  int nops = (int)nops_ll;

  if (indptr_buf.shape[0] < 1) {
    throw std::invalid_argument("indptr must have length >= 1");
  }
  int64_t nrows_total_ll = (int64_t)indptr_buf.shape[0] - 1;
  if (nrows_total_ll < 0) {
    throw std::invalid_argument("invalid indptr length");
  }

  row_start = int(row_start);
  if (row_start < 0) {
    throw std::invalid_argument("row_start must be >= 0");
  }
  if ((int64_t)row_start > nrows_total_ll) {
    throw std::invalid_argument("row_start out of range for indptr");
  }

  int64_t nrows_out_ll = 0;
  if (nrows < 0) {
    nrows_out_ll = nrows_total_ll - (int64_t)row_start;
  } else {
    nrows_out_ll = (int64_t)nrows;
  }
  if (nrows_out_ll < 0) {
    throw std::invalid_argument("nrows must be >= 0 or -1");
  }
  if ((int64_t)row_start + nrows_out_ll > nrows_total_ll) {
    throw std::invalid_argument("requested row range exceeds indptr length");
  }
  if (nrows_out_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nrows too large (batch the work)");
  }
  int nrows_out = (int)nrows_out_ll;

  py::array_t<double> out_g({(py::ssize_t)nrows_out_ll, (py::ssize_t)nops_ll});
  if (nrows_out == 0 || nops == 0) {
    return out_g;
  }

  const int64_t* h_indptr = static_cast<const int64_t*>(indptr_buf.ptr);
  int64_t nnz_total_ll = (int64_t)indices_buf.shape[0];

  int64_t indptr_nnz = h_indptr[nrows_total_ll];
  if (indptr_nnz != nnz_total_ll) {
    throw std::invalid_argument("indptr[-1] must equal len(indices)");
  }

  int64_t base = h_indptr[(int64_t)row_start];
  int64_t end = h_indptr[(int64_t)row_start + nrows_out_ll];
  if (base < 0 || end < base || end > nnz_total_ll) {
    throw std::invalid_argument("indptr contains invalid nnz offsets for requested row range");
  }
  int64_t nnz_sub_ll = end - base;

  std::vector<int64_t> h_indptr_sub((size_t)nrows_out + 1);
  for (int i = 0; i <= nrows_out; i++) {
    h_indptr_sub[(size_t)i] = h_indptr[(int64_t)row_start + (int64_t)i] - base;
  }

  const int32_t* h_indices = static_cast<const int32_t*>(indices_buf.ptr);
  const double* h_data = static_cast<const double*>(data_buf.ptr);
  const double* h_eri = static_cast<const double*>(eri_buf.ptr);

  // Allocate/copy CSR (subrange) and ERI on device.
  int64_t nnz_alloc_ll = std::max<int64_t>(nnz_sub_ll, 1);
  size_t indptr_bytes = ((size_t)nrows_out + 1) * sizeof(int64_t);
  size_t indices_bytes = (size_t)nnz_alloc_ll * sizeof(int32_t);
  size_t data_bytes = (size_t)nnz_alloc_ll * sizeof(double);
  size_t eri_bytes = (size_t)nops * (size_t)nops * sizeof(double);
  size_t g_bytes = (size_t)nrows_out * (size_t)nops * sizeof(double);

  int64_t* d_indptr_raw = nullptr;
  int32_t* d_indices_raw = nullptr;
  double* d_data_raw = nullptr;
  double* d_eri_raw = nullptr;
  double* d_g_raw = nullptr;

  cuda_unique_ptr<int64_t> d_indptr;
  cuda_unique_ptr<int32_t> d_indices;
  cuda_unique_ptr<double> d_data;
  cuda_unique_ptr<double> d_eri;
  cuda_unique_ptr<double> d_g;

  throw_on_cuda_error(cudaMalloc((void**)&d_indptr_raw, indptr_bytes), "cudaMalloc(indptr)");
  throw_on_cuda_error(cudaMalloc((void**)&d_indices_raw, indices_bytes), "cudaMalloc(indices)");
  throw_on_cuda_error(cudaMalloc((void**)&d_data_raw, data_bytes), "cudaMalloc(data)");
  throw_on_cuda_error(cudaMalloc((void**)&d_eri_raw, eri_bytes), "cudaMalloc(eri_mat)");
  throw_on_cuda_error(cudaMalloc((void**)&d_g_raw, g_bytes), "cudaMalloc(g_out)");

  d_indptr.reset(d_indptr_raw);
  d_indices.reset(d_indices_raw);
  d_data.reset(d_data_raw);
  d_eri.reset(d_eri_raw);
  d_g.reset(d_g_raw);

  throw_on_cuda_error(cudaMemcpy(d_indptr.get(), h_indptr_sub.data(), indptr_bytes, cudaMemcpyHostToDevice), "H2D indptr");
  if (nnz_sub_ll > 0) {
    throw_on_cuda_error(
        cudaMemcpy(d_indices.get(), h_indices + base, (size_t)nnz_sub_ll * sizeof(int32_t), cudaMemcpyHostToDevice),
        "H2D indices");
    throw_on_cuda_error(
        cudaMemcpy(d_data.get(), h_data + base, (size_t)nnz_sub_ll * sizeof(double), cudaMemcpyHostToDevice), "H2D data");
  }
  throw_on_cuda_error(cudaMemcpy(d_eri.get(), h_eri, eri_bytes, cudaMemcpyHostToDevice), "H2D eri_mat");

  throw_on_cuda_error(
      guga_build_g_from_csr_eri_mat_launch(d_indptr.get(), d_indices.get(), d_data.get(), nrows_out, d_eri.get(), nops, half, d_g.get(), threads),
      "guga_build_g_from_csr_eri_mat_launch");
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel3_build_g)");

  throw_on_cuda_error(cudaMemcpy(out_g.mutable_data(), d_g.get(), g_bytes, cudaMemcpyDeviceToHost), "D2H g_out");
  return out_g;
}

void kernel3_build_g_from_csr_eri_mat_inplace_device(
    py::object indptr,
    py::object indices,
    py::object data,
    py::object eri_mat,
    py::object g_out,
    int threads,
    double half,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (indptr.is_none() || indices.is_none() || data.is_none() || eri_mat.is_none() || g_out.is_none()) {
    throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
  }

  auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
  require_typestr(indptr_dev, "indptr", "<i8");
  if (indptr_dev.shape.size() != 1 || indptr_dev.shape[0] < 1) {
    throw std::invalid_argument("indptr must be a 1D device array with shape (nrows+1,)");
  }
  if (!indptr_dev.strides_bytes.empty()) {
    if (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("indptr must be contiguous");
    }
  }

  int64_t nrows_ll = indptr_dev.shape[0] - 1;
  if (nrows_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nrows too large (batch the work)");
  }
  int nrows = (int)nrows_ll;

  auto indices_dev = cuda_array_view_from_object(indices, "indices");
  require_typestr(indices_dev, "indices", "<i4");
  if (indices_dev.shape.size() != 1) {
    throw std::invalid_argument("indices must be a 1D device array");
  }
  if (!indices_dev.strides_bytes.empty()) {
    if (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("indices must be contiguous");
    }
  }

  auto data_dev = cuda_array_view_from_object(data, "data");
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  std::string data_typestr = normalize_typestr(data_dev.typestr);
  if (data_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (data_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (data_dev.shape.size() != 1) {
    throw std::invalid_argument("data must be a 1D device array");
  }
  if (!data_dev.strides_bytes.empty()) {
    if (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("data must be contiguous");
    }
  }
  if (data_dev.shape[0] != indices_dev.shape[0]) {
    throw std::invalid_argument("indices and data must have the same length");
  }

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
    throw std::invalid_argument("eri_mat must have shape (nops,nops)");
  }
  int64_t nops_ll = eri_dev.shape[0];
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops out of supported range");
  }
  int nops = (int)nops_ll;
  if (nops < 0) {
    throw std::invalid_argument("invalid nops");
  }
  if (nops > 0) {
    if (eri_dev.strides_bytes.empty()) {
      // ok
    } else {
      if (eri_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("eri_mat strides must have length 2");
      }
      int64_t s0 = eri_dev.strides_bytes[0];
      int64_t s1 = eri_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
      }
    }
  }

  auto g_dev = cuda_array_view_from_object(g_out, "g_out");
  if (use_f32) require_typestr(g_dev, "g_out", "<f4");
  else require_typestr(g_dev, "g_out", "<f8");
  if (g_dev.read_only) {
    throw std::invalid_argument("g_out must be writable");
  }
  if (g_dev.shape.size() != 2 || g_dev.shape[0] != nrows_ll || g_dev.shape[1] != nops_ll) {
    throw std::invalid_argument("g_out must have shape (nrows,nops)");
  }
  if (nops > 0 && nrows > 0) {
    if (g_dev.strides_bytes.empty()) {
      // ok
    } else {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (g_dev.stream) stream_u = g_dev.stream;
    else if (eri_dev.stream) stream_u = eri_dev.stream;
    else if (data_dev.stream) stream_u = data_dev.stream;
    else if (indices_dev.stream) stream_u = indices_dev.stream;
    else if (indptr_dev.stream) stream_u = indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int64_t* d_indptr = reinterpret_cast<const int64_t*>(indptr_dev.ptr);
  const int32_t* d_indices = reinterpret_cast<const int32_t*>(indices_dev.ptr);

  if (nrows > 0 && nops > 0) {
    if (use_f32) {
      const float* d_data = reinterpret_cast<const float*>(data_dev.ptr);
      const float* d_eri = reinterpret_cast<const float*>(eri_dev.ptr);
      float* d_g = reinterpret_cast<float*>(g_dev.ptr);
      throw_on_cuda_error(
          guga_build_g_from_csr_eri_mat_f32_launch_stream(
              d_indptr, d_indices, d_data, nrows, d_eri, nops, (float)half, d_g, stream_t, threads),
          "guga_build_g_from_csr_eri_mat_f32_launch_stream");
    } else {
      const double* d_data = reinterpret_cast<const double*>(data_dev.ptr);
      const double* d_eri = reinterpret_cast<const double*>(eri_dev.ptr);
      double* d_g = reinterpret_cast<double*>(g_dev.ptr);
      throw_on_cuda_error(
          guga_build_g_from_csr_eri_mat_f64_launch_stream(
              d_indptr, d_indices, d_data, nrows, d_eri, nops, half, d_g, stream_t, threads),
          "guga_build_g_from_csr_eri_mat_f64_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel3_build_g)");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel3_build_g)");
  }
}

void kernel3_build_g_from_csr_eri_mat_range_inplace_device(
    py::object indptr,
    py::object indices,
    py::object data,
    int row_start,
    int nrows,
    py::object eri_mat,
    py::object g_out,
    int threads,
    double half,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (row_start < 0 || nrows < 0) {
    throw std::invalid_argument("row_start and nrows must be >= 0");
  }
  if (indptr.is_none() || indices.is_none() || data.is_none() || eri_mat.is_none() || g_out.is_none()) {
    throw std::invalid_argument("all inputs/outputs must be device arrays (cannot be None)");
  }

  auto indptr_dev = cuda_array_view_from_object(indptr, "indptr");
  require_typestr(indptr_dev, "indptr", "<i8");
  if (indptr_dev.shape.size() != 1 || indptr_dev.shape[0] < 1) {
    throw std::invalid_argument("indptr must be a 1D device array with shape (nrows_total+1,)");
  }
  if (!indptr_dev.strides_bytes.empty()) {
    if (indptr_dev.strides_bytes.size() != 1 || indptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) {
      throw std::invalid_argument("indptr must be contiguous");
    }
  }

  int64_t nrows_total_ll = indptr_dev.shape[0] - 1;
  if (nrows_total_ll < 0) {
    throw std::invalid_argument("invalid indptr length");
  }
  if ((int64_t)row_start + (int64_t)nrows > nrows_total_ll) {
    throw std::invalid_argument("requested row range exceeds indptr length");
  }

  auto indices_dev = cuda_array_view_from_object(indices, "indices");
  require_typestr(indices_dev, "indices", "<i4");
  if (indices_dev.shape.size() != 1) {
    throw std::invalid_argument("indices must be a 1D device array");
  }
  if (!indices_dev.strides_bytes.empty()) {
    if (indices_dev.strides_bytes.size() != 1 || indices_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("indices must be contiguous");
    }
  }

  auto data_dev = cuda_array_view_from_object(data, "data");
  bool use_f32 = false;
  int64_t fp_itemsize = 0;
  std::string data_typestr = normalize_typestr(data_dev.typestr);
  if (data_typestr == "<f8") {
    use_f32 = false;
    fp_itemsize = (int64_t)sizeof(double);
  } else if (data_typestr == "<f4") {
    use_f32 = true;
    fp_itemsize = (int64_t)sizeof(float);
  } else {
    throw std::invalid_argument("data must have typestr <f8 (float64) or <f4 (float32)");
  }
  if (data_dev.shape.size() != 1) {
    throw std::invalid_argument("data must be a 1D device array");
  }
  if (!data_dev.strides_bytes.empty()) {
    if (data_dev.strides_bytes.size() != 1 || data_dev.strides_bytes[0] != fp_itemsize) {
      throw std::invalid_argument("data must be contiguous");
    }
  }
  if (data_dev.shape[0] != indices_dev.shape[0]) {
    throw std::invalid_argument("indices and data must have the same length");
  }

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
    throw std::invalid_argument("eri_mat must have shape (nops,nops)");
  }
  int64_t nops_ll = eri_dev.shape[0];
  if (nops_ll > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nops out of supported range");
  }
  int nops = (int)nops_ll;

  if (nops > 0) {
    if (eri_dev.strides_bytes.empty()) {
      // ok
    } else {
      if (eri_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("eri_mat strides must have length 2");
      }
      int64_t s0 = eri_dev.strides_bytes[0];
      int64_t s1 = eri_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
      }
    }
  }

  auto g_dev = cuda_array_view_from_object(g_out, "g_out");
  if (use_f32) require_typestr(g_dev, "g_out", "<f4");
  else require_typestr(g_dev, "g_out", "<f8");
  if (g_dev.read_only) {
    throw std::invalid_argument("g_out must be writable");
  }
  if (g_dev.shape.size() != 2 || g_dev.shape[0] != (int64_t)nrows || g_dev.shape[1] != nops_ll) {
    throw std::invalid_argument("g_out must have shape (nrows,nops)");
  }
  if (nops > 0 && nrows > 0) {
    if (g_dev.strides_bytes.empty()) {
      // ok
    } else {
      if (g_dev.strides_bytes.size() != 2) {
        throw std::invalid_argument("g_out strides must have length 2");
      }
      int64_t s0 = g_dev.strides_bytes[0];
      int64_t s1 = g_dev.strides_bytes[1];
      if (s0 != (int64_t)nops * fp_itemsize || s1 != fp_itemsize) {
        throw std::invalid_argument("g_out must be C-contiguous with no padding");
      }
    }
  }

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (g_dev.stream) stream_u = g_dev.stream;
    else if (eri_dev.stream) stream_u = eri_dev.stream;
    else if (data_dev.stream) stream_u = data_dev.stream;
    else if (indices_dev.stream) stream_u = indices_dev.stream;
    else if (indptr_dev.stream) stream_u = indptr_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  const int64_t* d_indptr = reinterpret_cast<const int64_t*>(indptr_dev.ptr);
  const int32_t* d_indices = reinterpret_cast<const int32_t*>(indices_dev.ptr);

  if (nrows > 0 && nops > 0) {
    if (use_f32) {
      const float* d_data = reinterpret_cast<const float*>(data_dev.ptr);
      const float* d_eri = reinterpret_cast<const float*>(eri_dev.ptr);
      float* d_g = reinterpret_cast<float*>(g_dev.ptr);
      throw_on_cuda_error(
          guga_build_g_from_csr_eri_mat_range_f32_launch_stream(
              d_indptr, d_indices, d_data, row_start, nrows, d_eri, nops, (float)half, d_g, stream_t, threads),
          "guga_build_g_from_csr_eri_mat_range_f32_launch_stream");
    } else {
      const double* d_data = reinterpret_cast<const double*>(data_dev.ptr);
      const double* d_eri = reinterpret_cast<const double*>(eri_dev.ptr);
      double* d_g = reinterpret_cast<double*>(g_dev.ptr);
      throw_on_cuda_error(
          guga_build_g_from_csr_eri_mat_range_f64_launch_stream(
              d_indptr, d_indices, d_data, row_start, nrows, d_eri, nops, half, d_g, stream_t, threads),
          "guga_build_g_from_csr_eri_mat_range_f64_launch_stream");
    }
    throw_on_cuda_error(cudaGetLastError(), "kernel launch(kernel3_build_g_range)");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(kernel3_build_g_range)");
  }
}

py::tuple epq_contribs_one_debug(
    const DeviceDRT& drt,
    int csf_idx,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> steps,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> nodes,
    int p,
    int q,
    int max_out) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (max_out < 0) {
    throw std::invalid_argument("max_out must be >= 0");
  }

  auto steps_buf = steps.request();
  auto nodes_buf = nodes.request();
  if (steps_buf.ndim != 1 || (int)steps_buf.shape[0] != drt.norb) {
    throw std::invalid_argument("steps must have shape (norb,)");
  }
  if (nodes_buf.ndim != 1 || (int)nodes_buf.shape[0] != drt.norb + 1) {
    throw std::invalid_argument("nodes must have shape (norb+1,)");
  }

  int8_t* d_steps = nullptr;
  int32_t* d_nodes = nullptr;
  int32_t* d_out_idx = nullptr;
  double* d_out_coeff = nullptr;
  int* d_out_count = nullptr;
  int* d_overflow = nullptr;

  size_t steps_bytes = (size_t)drt.norb * sizeof(int8_t);
  size_t nodes_bytes = (size_t)(drt.norb + 1) * sizeof(int32_t);
  size_t out_i_bytes = (size_t)max_out * sizeof(int32_t);
  size_t out_v_bytes = (size_t)max_out * sizeof(double);

  throw_on_cuda_error(cudaMalloc((void**)&d_steps, steps_bytes), "cudaMalloc(steps)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nodes, nodes_bytes), "cudaMalloc(nodes)");
  if (max_out > 0) {
    throw_on_cuda_error(cudaMalloc((void**)&d_out_idx, out_i_bytes), "cudaMalloc(out_idx)");
    throw_on_cuda_error(cudaMalloc((void**)&d_out_coeff, out_v_bytes), "cudaMalloc(out_coeff)");
  }
  throw_on_cuda_error(cudaMalloc((void**)&d_out_count, sizeof(int)), "cudaMalloc(out_count)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow, sizeof(int)), "cudaMalloc(overflow)");

  throw_on_cuda_error(cudaMemcpy(d_steps, steps_buf.ptr, steps_bytes, cudaMemcpyHostToDevice), "H2D steps");
  throw_on_cuda_error(cudaMemcpy(d_nodes, nodes_buf.ptr, nodes_bytes, cudaMemcpyHostToDevice), "H2D nodes");

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_out_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D out_count");
  throw_on_cuda_error(cudaMemcpy(d_overflow, &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow");

  guga_epq_contribs_one_debug_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      drt.norb,
      csf_idx,
      d_steps,
      d_nodes,
      p,
      q,
      max_out,
      d_out_idx,
      d_out_coeff,
      d_out_count,
      d_overflow);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int h_count = 0;
  int h_overflow = 0;
  throw_on_cuda_error(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost), "D2H out_count");
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");

  if (h_overflow) {
    cudaFree(d_steps);
    cudaFree(d_nodes);
    if (d_out_idx) cudaFree(d_out_idx);
    if (d_out_coeff) cudaFree(d_out_coeff);
    cudaFree(d_out_count);
    cudaFree(d_overflow);
    throw std::runtime_error("epq_contribs_one_debug overflow: increase max_out or MAX_NORB");
  }

  h_count = std::max(0, std::min(h_count, max_out));
  py::array_t<int32_t> out_i({h_count});
  py::array_t<double> out_v({h_count});
  if (h_count > 0) {
    throw_on_cuda_error(
        cudaMemcpy(out_i.mutable_data(), d_out_idx, (size_t)h_count * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "D2H out_idx");
    throw_on_cuda_error(
        cudaMemcpy(out_v.mutable_data(), d_out_coeff, (size_t)h_count * sizeof(double), cudaMemcpyDeviceToHost),
        "D2H out_coeff");
  }

  cudaFree(d_steps);
  cudaFree(d_nodes);
  if (d_out_idx) cudaFree(d_out_idx);
  if (d_out_coeff) cudaFree(d_out_coeff);
  cudaFree(d_out_count);
  cudaFree(d_overflow);

  return py::make_tuple(out_i, out_v);
}

py::tuple epq_apply_g_debug(
    const DeviceDRT& drt,
    int csf_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> g_flat,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> steps,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> nodes,
    double thresh_gpq,
    double thresh_contrib,
    int max_out) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (max_out < 0) {
    throw std::invalid_argument("max_out must be >= 0");
  }

  auto g_buf = g_flat.request();
  auto steps_buf = steps.request();
  auto nodes_buf = nodes.request();

  if (g_buf.ndim != 1 || (int)g_buf.shape[0] != drt.norb * drt.norb) {
    throw std::invalid_argument("g_flat must have shape (norb*norb,)");
  }
  if (steps_buf.ndim != 1 || (int)steps_buf.shape[0] != drt.norb) {
    throw std::invalid_argument("steps must have shape (norb,)");
  }
  if (nodes_buf.ndim != 1 || (int)nodes_buf.shape[0] != drt.norb + 1) {
    throw std::invalid_argument("nodes must have shape (norb+1,)");
  }

  int8_t* d_steps = nullptr;
  int32_t* d_nodes = nullptr;
  double* d_g = nullptr;
  int32_t* d_out_idx = nullptr;
  double* d_out_val = nullptr;
  int* d_out_count = nullptr;
  int* d_overflow = nullptr;
  int* d_n_pairs = nullptr;

  size_t steps_bytes = (size_t)drt.norb * sizeof(int8_t);
  size_t nodes_bytes = (size_t)(drt.norb + 1) * sizeof(int32_t);
  size_t g_bytes = (size_t)(drt.norb * drt.norb) * sizeof(double);
  size_t out_i_bytes = (size_t)max_out * sizeof(int32_t);
  size_t out_v_bytes = (size_t)max_out * sizeof(double);

  throw_on_cuda_error(cudaMalloc((void**)&d_steps, steps_bytes), "cudaMalloc(steps)");
  throw_on_cuda_error(cudaMalloc((void**)&d_nodes, nodes_bytes), "cudaMalloc(nodes)");
  throw_on_cuda_error(cudaMalloc((void**)&d_g, g_bytes), "cudaMalloc(g_flat)");
  if (max_out > 0) {
    throw_on_cuda_error(cudaMalloc((void**)&d_out_idx, out_i_bytes), "cudaMalloc(out_idx)");
    throw_on_cuda_error(cudaMalloc((void**)&d_out_val, out_v_bytes), "cudaMalloc(out_val)");
  }
  throw_on_cuda_error(cudaMalloc((void**)&d_out_count, sizeof(int)), "cudaMalloc(out_count)");
  throw_on_cuda_error(cudaMalloc((void**)&d_overflow, sizeof(int)), "cudaMalloc(overflow)");
  throw_on_cuda_error(cudaMalloc((void**)&d_n_pairs, sizeof(int)), "cudaMalloc(n_pairs)");

  throw_on_cuda_error(cudaMemcpy(d_steps, steps_buf.ptr, steps_bytes, cudaMemcpyHostToDevice), "H2D steps");
  throw_on_cuda_error(cudaMemcpy(d_nodes, nodes_buf.ptr, nodes_bytes, cudaMemcpyHostToDevice), "H2D nodes");
  throw_on_cuda_error(cudaMemcpy(d_g, g_buf.ptr, g_bytes, cudaMemcpyHostToDevice), "H2D g_flat");

  int zero = 0;
  throw_on_cuda_error(cudaMemcpy(d_out_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D out_count");
  throw_on_cuda_error(cudaMemcpy(d_overflow, &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D overflow");
  throw_on_cuda_error(cudaMemcpy(d_n_pairs, &zero, sizeof(int), cudaMemcpyHostToDevice), "H2D n_pairs");

  guga_epq_apply_g_debug_launch(
      drt.child,
      drt.node_twos,
      drt.child_prefix,
      drt.norb,
      csf_idx,
      d_steps,
      d_nodes,
      d_g,
      thresh_gpq,
      thresh_contrib,
      max_out,
      d_out_idx,
      d_out_val,
      d_out_count,
      d_overflow,
      d_n_pairs);
  throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int h_count = 0;
  int h_overflow = 0;
  int h_pairs = 0;
  throw_on_cuda_error(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost), "D2H out_count");
  throw_on_cuda_error(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost), "D2H overflow");
  throw_on_cuda_error(cudaMemcpy(&h_pairs, d_n_pairs, sizeof(int), cudaMemcpyDeviceToHost), "D2H n_pairs");

  if (h_overflow) {
    cudaFree(d_steps);
    cudaFree(d_nodes);
    cudaFree(d_g);
    if (d_out_idx) cudaFree(d_out_idx);
    if (d_out_val) cudaFree(d_out_val);
    cudaFree(d_out_count);
    cudaFree(d_overflow);
    cudaFree(d_n_pairs);
    throw std::runtime_error("epq_apply_g_debug overflow: increase max_out or MAX_NORB");
  }

  h_count = std::max(0, std::min(h_count, max_out));
  py::array_t<int32_t> out_i({h_count});
  py::array_t<double> out_v({h_count});
  if (h_count > 0) {
    throw_on_cuda_error(
        cudaMemcpy(out_i.mutable_data(), d_out_idx, (size_t)h_count * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "D2H out_idx");
    throw_on_cuda_error(
        cudaMemcpy(out_v.mutable_data(), d_out_val, (size_t)h_count * sizeof(double), cudaMemcpyDeviceToHost),
        "D2H out_val");
  }

  cudaFree(d_steps);
  cudaFree(d_nodes);
  cudaFree(d_g);
  if (d_out_idx) cudaFree(d_out_idx);
  if (d_out_val) cudaFree(d_out_val);
  cudaFree(d_out_count);
  cudaFree(d_overflow);
  cudaFree(d_n_pairs);

  return py::make_tuple(out_i, out_v, h_pairs);
}

void ell_spmv_f64_inplace_device(
    py::object col_idx,
    py::object val,
    py::object x,
    py::object y,
    int threads,
    bool add,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument("threads must be a multiple of 32");
  }

  if (col_idx.is_none() || val.is_none() || x.is_none() || y.is_none()) {
    throw std::invalid_argument("col_idx/val/x/y must be device arrays (cannot be None)");
  }

  CudaArrayView col_dev = cuda_array_view_from_object(col_idx, "col_idx");
  CudaArrayView val_dev = cuda_array_view_from_object(val, "val");
  CudaArrayView x_dev = cuda_array_view_from_object(x, "x");
  CudaArrayView y_dev = cuda_array_view_from_object(y, "y");

  require_typestr(col_dev, "col_idx", "<i4");
  require_typestr(val_dev, "val", "<f8");
  require_typestr(x_dev, "x", "<f8");
  require_typestr(y_dev, "y", "<f8");

  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }

  if (col_dev.shape.size() != 2 || val_dev.shape.size() != 2) {
    throw std::invalid_argument("col_idx and val must be 2D arrays");
  }
  if (x_dev.shape.size() != 1 || y_dev.shape.size() != 1) {
    throw std::invalid_argument("x and y must be 1D arrays");
  }

  int64_t nrows64 = col_dev.shape[0];
  int64_t width64 = col_dev.shape[1];
  if (nrows64 < 0 || width64 < 0) {
    throw std::invalid_argument("invalid col_idx shape");
  }
  if (val_dev.shape[0] != nrows64 || val_dev.shape[1] != width64) {
    throw std::invalid_argument("val must have the same shape as col_idx");
  }
  if (x_dev.shape[0] != nrows64 || y_dev.shape[0] != nrows64) {
    throw std::invalid_argument("x/y must have length equal to nrows");
  }
  if (nrows64 > (int64_t)std::numeric_limits<int>::max() || width64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nrows/width too large for int32 kernel interface");
  }
  int nrows = (int)nrows64;
  int width = (int)width64;

  if (!col_dev.strides_bytes.empty()) {
    if (col_dev.strides_bytes.size() != 2) throw std::invalid_argument("col_idx must be 2D with valid strides");
    if (col_dev.strides_bytes[1] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("col_idx must be contiguous");
    if (col_dev.strides_bytes[0] != (int64_t)width * (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("col_idx must be C-contiguous");
    }
  }
  if (!val_dev.strides_bytes.empty()) {
    if (val_dev.strides_bytes.size() != 2) throw std::invalid_argument("val must be 2D with valid strides");
    if (val_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("val must be contiguous");
    if (val_dev.strides_bytes[0] != (int64_t)width * (int64_t)sizeof(double)) {
      throw std::invalid_argument("val must be C-contiguous");
    }
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1 || x_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be contiguous");
    }
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1 || y_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("y must be contiguous");
    }
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_ell_spmv_f64_launch_stream(
          (const int32_t*)col_dev.ptr,
          (const double*)val_dev.ptr,
          nrows,
          width,
          (const double*)x_dev.ptr,
          (double*)y_dev.ptr,
          add ? 1 : 0,
          stream_t,
          threads),
      "guga_ell_spmv_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(ell_spmv)");
  }
}

void ell_spmm_f64_inplace_device(
    py::object col_idx,
    py::object val,
    py::object x,
    py::object y,
    int threads,
    bool add,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument("threads must be a multiple of 32");
  }

  if (col_idx.is_none() || val.is_none() || x.is_none() || y.is_none()) {
    throw std::invalid_argument("col_idx/val/x/y must be device arrays (cannot be None)");
  }

  CudaArrayView col_dev = cuda_array_view_from_object(col_idx, "col_idx");
  CudaArrayView val_dev = cuda_array_view_from_object(val, "val");
  CudaArrayView x_dev = cuda_array_view_from_object(x, "x");
  CudaArrayView y_dev = cuda_array_view_from_object(y, "y");

  require_typestr(col_dev, "col_idx", "<i4");
  require_typestr(val_dev, "val", "<f8");
  require_typestr(x_dev, "x", "<f8");
  require_typestr(y_dev, "y", "<f8");

  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }

  if (col_dev.shape.size() != 2 || val_dev.shape.size() != 2) {
    throw std::invalid_argument("col_idx and val must be 2D arrays");
  }
  if (x_dev.shape.size() != 2 || y_dev.shape.size() != 2) {
    throw std::invalid_argument("x and y must be 2D arrays");
  }

  int64_t nrows64 = col_dev.shape[0];
  int64_t width64 = col_dev.shape[1];
  if (nrows64 < 0 || width64 < 0) {
    throw std::invalid_argument("invalid col_idx shape");
  }
  if (val_dev.shape[0] != nrows64 || val_dev.shape[1] != width64) {
    throw std::invalid_argument("val must have the same shape as col_idx");
  }

  if (x_dev.shape[0] != nrows64 || y_dev.shape[0] != nrows64) {
    throw std::invalid_argument("x/y must have shape (nrows,nvec)");
  }
  int64_t nvec64 = x_dev.shape[1];
  if (nvec64 < 0) {
    throw std::invalid_argument("invalid x shape");
  }
  if (y_dev.shape[1] != nvec64) {
    throw std::invalid_argument("y must have the same shape as x");
  }

  if (nrows64 > (int64_t)std::numeric_limits<int>::max() || width64 > (int64_t)std::numeric_limits<int>::max() ||
      nvec64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nrows/width/nvec too large for int32 kernel interface");
  }
  int nrows = (int)nrows64;
  int width = (int)width64;
  int nvec = (int)nvec64;

  if (!col_dev.strides_bytes.empty()) {
    if (col_dev.strides_bytes.size() != 2) throw std::invalid_argument("col_idx must be 2D with valid strides");
    if (col_dev.strides_bytes[1] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("col_idx must be contiguous");
    if (col_dev.strides_bytes[0] != (int64_t)width * (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("col_idx must be C-contiguous");
    }
  }
  if (!val_dev.strides_bytes.empty()) {
    if (val_dev.strides_bytes.size() != 2) throw std::invalid_argument("val must be 2D with valid strides");
    if (val_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("val must be contiguous");
    if (val_dev.strides_bytes[0] != (int64_t)width * (int64_t)sizeof(double)) {
      throw std::invalid_argument("val must be C-contiguous");
    }
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 2) throw std::invalid_argument("x must be 2D with valid strides");
    if (x_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("x must be contiguous");
    if (x_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be C-contiguous");
    }
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 2) throw std::invalid_argument("y must be 2D with valid strides");
    if (y_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("y must be contiguous");
    if (y_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y must be C-contiguous");
    }
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_ell_spmm_f64_launch_stream(
          (const int32_t*)col_dev.ptr,
          (const double*)val_dev.ptr,
          nrows,
          width,
          (const double*)x_dev.ptr,
          /*ldx=*/nvec,
          (double*)y_dev.ptr,
          /*ldy=*/nvec,
          nvec,
          add ? 1 : 0,
          stream_t,
          threads),
      "guga_ell_spmm_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(ell_spmm)");
  }
}

void sell_spmv_f64_inplace_device(
    py::object slice_ptr,
    py::object slice_width,
    py::object col_idx,
    py::object val,
    py::object x,
    py::object y,
    int slice_height,
    int threads,
    bool add,
    uint64_t stream,
    bool sync) {
  if (slice_height <= 0) {
    throw std::invalid_argument("slice_height must be > 0");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument("threads must be a multiple of 32");
  }

  if (slice_ptr.is_none() || slice_width.is_none() || col_idx.is_none() || val.is_none() || x.is_none() || y.is_none()) {
    throw std::invalid_argument("slice_ptr/slice_width/col_idx/val/x/y must be device arrays (cannot be None)");
  }

  CudaArrayView ptr_dev = cuda_array_view_from_object(slice_ptr, "slice_ptr");
  CudaArrayView width_dev = cuda_array_view_from_object(slice_width, "slice_width");
  CudaArrayView col_dev = cuda_array_view_from_object(col_idx, "col_idx");
  CudaArrayView val_dev = cuda_array_view_from_object(val, "val");
  CudaArrayView x_dev = cuda_array_view_from_object(x, "x");
  CudaArrayView y_dev = cuda_array_view_from_object(y, "y");

  require_typestr(ptr_dev, "slice_ptr", "<i8");
  require_typestr(width_dev, "slice_width", "<i4");
  require_typestr(col_dev, "col_idx", "<i4");
  require_typestr(val_dev, "val", "<f8");
  require_typestr(x_dev, "x", "<f8");
  require_typestr(y_dev, "y", "<f8");

  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }

  if (ptr_dev.shape.size() != 1 || width_dev.shape.size() != 1) {
    throw std::invalid_argument("slice_ptr and slice_width must be 1D arrays");
  }
  if (col_dev.shape.size() != 1 || val_dev.shape.size() != 1) {
    throw std::invalid_argument("col_idx and val must be 1D arrays");
  }
  if (x_dev.shape.size() != 1 || y_dev.shape.size() != 1) {
    throw std::invalid_argument("x and y must be 1D arrays");
  }

  int64_t nslices64 = width_dev.shape[0];
  if (nslices64 < 0) {
    throw std::invalid_argument("invalid slice_width shape");
  }
  if (ptr_dev.shape[0] != nslices64 + 1) {
    throw std::invalid_argument("slice_ptr must have shape (nslices+1,)");
  }

  int64_t nrows64 = x_dev.shape[0];
  if (y_dev.shape[0] != nrows64) {
    throw std::invalid_argument("x and y must have the same length");
  }
  if (nrows64 < 0) {
    throw std::invalid_argument("invalid x shape");
  }
  int64_t max_rows64 = nslices64 * (int64_t)slice_height;
  if (nrows64 > max_rows64) {
    throw std::invalid_argument("x length exceeds nslices*slice_height");
  }

  if (col_dev.shape[0] != val_dev.shape[0]) {
    throw std::invalid_argument("col_idx and val must have the same length");
  }

  if (nslices64 > (int64_t)std::numeric_limits<int>::max() || nrows64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nslices/nrows too large for int32 kernel interface");
  }
  int nrows = (int)nrows64;

  if (!ptr_dev.strides_bytes.empty()) {
    if (ptr_dev.strides_bytes.size() != 1) throw std::invalid_argument("slice_ptr must be 1D with valid strides");
    if (ptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) throw std::invalid_argument("slice_ptr must be contiguous");
  }
  if (!width_dev.strides_bytes.empty()) {
    if (width_dev.strides_bytes.size() != 1) throw std::invalid_argument("slice_width must be 1D with valid strides");
    if (width_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("slice_width must be contiguous");
  }
  if (!col_dev.strides_bytes.empty()) {
    if (col_dev.strides_bytes.size() != 1) throw std::invalid_argument("col_idx must be 1D with valid strides");
    if (col_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("col_idx must be contiguous");
  }
  if (!val_dev.strides_bytes.empty()) {
    if (val_dev.strides_bytes.size() != 1) throw std::invalid_argument("val must be 1D with valid strides");
    if (val_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val must be contiguous");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 1) throw std::invalid_argument("x must be 1D with valid strides");
    if (x_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("x must be contiguous");
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 1) throw std::invalid_argument("y must be 1D with valid strides");
    if (y_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("y must be contiguous");
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_sell_spmv_f64_launch_stream(
          (const int64_t*)ptr_dev.ptr,
          (const int32_t*)width_dev.ptr,
          (const int32_t*)col_dev.ptr,
          (const double*)val_dev.ptr,
          nrows,
          slice_height,
          (const double*)x_dev.ptr,
          (double*)y_dev.ptr,
          add ? 1 : 0,
          stream_t,
          threads),
      "guga_sell_spmv_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(sell_spmv)");
  }
}

void sell_spmm_f64_inplace_device(
    py::object slice_ptr,
    py::object slice_width,
    py::object col_idx,
    py::object val,
    py::object x,
    py::object y,
    int slice_height,
    int threads,
    bool add,
    uint64_t stream,
    bool sync) {
  if (slice_height <= 0) {
    throw std::invalid_argument("slice_height must be > 0");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument("threads must be a multiple of 32");
  }

  if (slice_ptr.is_none() || slice_width.is_none() || col_idx.is_none() || val.is_none() || x.is_none() || y.is_none()) {
    throw std::invalid_argument("slice_ptr/slice_width/col_idx/val/x/y must be device arrays (cannot be None)");
  }

  CudaArrayView ptr_dev = cuda_array_view_from_object(slice_ptr, "slice_ptr");
  CudaArrayView width_dev = cuda_array_view_from_object(slice_width, "slice_width");
  CudaArrayView col_dev = cuda_array_view_from_object(col_idx, "col_idx");
  CudaArrayView val_dev = cuda_array_view_from_object(val, "val");
  CudaArrayView x_dev = cuda_array_view_from_object(x, "x");
  CudaArrayView y_dev = cuda_array_view_from_object(y, "y");

  require_typestr(ptr_dev, "slice_ptr", "<i8");
  require_typestr(width_dev, "slice_width", "<i4");
  require_typestr(col_dev, "col_idx", "<i4");
  require_typestr(val_dev, "val", "<f8");
  require_typestr(x_dev, "x", "<f8");
  require_typestr(y_dev, "y", "<f8");

  if (y_dev.read_only) {
    throw std::invalid_argument("y must be writable");
  }

  if (ptr_dev.shape.size() != 1 || width_dev.shape.size() != 1) {
    throw std::invalid_argument("slice_ptr and slice_width must be 1D arrays");
  }
  if (col_dev.shape.size() != 1 || val_dev.shape.size() != 1) {
    throw std::invalid_argument("col_idx and val must be 1D arrays");
  }
  if (x_dev.shape.size() != 2 || y_dev.shape.size() != 2) {
    throw std::invalid_argument("x and y must be 2D arrays");
  }

  int64_t nslices64 = width_dev.shape[0];
  if (nslices64 < 0) {
    throw std::invalid_argument("invalid slice_width shape");
  }
  if (ptr_dev.shape[0] != nslices64 + 1) {
    throw std::invalid_argument("slice_ptr must have shape (nslices+1,)");
  }

  int64_t nrows64 = x_dev.shape[0];
  int64_t nvec64 = x_dev.shape[1];
  if (y_dev.shape[0] != nrows64 || y_dev.shape[1] != nvec64) {
    throw std::invalid_argument("y must have the same shape as x");
  }
  if (nrows64 < 0 || nvec64 < 0) {
    throw std::invalid_argument("invalid x shape");
  }
  int64_t max_rows64 = nslices64 * (int64_t)slice_height;
  if (nrows64 > max_rows64) {
    throw std::invalid_argument("x rows exceed nslices*slice_height");
  }

  if (col_dev.shape[0] != val_dev.shape[0]) {
    throw std::invalid_argument("col_idx and val must have the same length");
  }

  if (nrows64 > (int64_t)std::numeric_limits<int>::max() || nvec64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("nrows/nvec too large for int32 kernel interface");
  }
  int nrows = (int)nrows64;
  int nvec = (int)nvec64;

  if (!ptr_dev.strides_bytes.empty()) {
    if (ptr_dev.strides_bytes.size() != 1) throw std::invalid_argument("slice_ptr must be 1D with valid strides");
    if (ptr_dev.strides_bytes[0] != (int64_t)sizeof(int64_t)) throw std::invalid_argument("slice_ptr must be contiguous");
  }
  if (!width_dev.strides_bytes.empty()) {
    if (width_dev.strides_bytes.size() != 1) throw std::invalid_argument("slice_width must be 1D with valid strides");
    if (width_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("slice_width must be contiguous");
  }
  if (!col_dev.strides_bytes.empty()) {
    if (col_dev.strides_bytes.size() != 1) throw std::invalid_argument("col_idx must be 1D with valid strides");
    if (col_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("col_idx must be contiguous");
  }
  if (!val_dev.strides_bytes.empty()) {
    if (val_dev.strides_bytes.size() != 1) throw std::invalid_argument("val must be 1D with valid strides");
    if (val_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val must be contiguous");
  }
  if (!x_dev.strides_bytes.empty()) {
    if (x_dev.strides_bytes.size() != 2) throw std::invalid_argument("x must be 2D with valid strides");
    if (x_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("x must be contiguous");
    if (x_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("x must be C-contiguous");
    }
  }
  if (!y_dev.strides_bytes.empty()) {
    if (y_dev.strides_bytes.size() != 2) throw std::invalid_argument("y must be 2D with valid strides");
    if (y_dev.strides_bytes[1] != (int64_t)sizeof(double)) throw std::invalid_argument("y must be contiguous");
    if (y_dev.strides_bytes[0] != (int64_t)nvec * (int64_t)sizeof(double)) {
      throw std::invalid_argument("y must be C-contiguous");
    }
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_sell_spmm_f64_launch_stream(
          (const int64_t*)ptr_dev.ptr,
          (const int32_t*)width_dev.ptr,
          (const int32_t*)col_dev.ptr,
          (const double*)val_dev.ptr,
          nrows,
          slice_height,
          (const double*)x_dev.ptr,
          /*ldx=*/nvec,
          (double*)y_dev.ptr,
          /*ldy=*/nvec,
          nvec,
          add ? 1 : 0,
          stream_t,
          threads),
      "guga_sell_spmm_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(sell_spmm)");
  }
}

void qmc_spawn_one_body_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object x_idx,
    py::object x_val,
    py::object h_eff_flat,
    py::object out_idx,
    py::object out_val,
    double eps,
    int nspawn,
    uint64_t seed,
    double initiator_t,
    int threads,
    uint64_t stream,
    bool sync) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (nspawn <= 0) {
    throw std::invalid_argument("nspawn must be >= 1");
  }

  if (x_idx.is_none() || x_val.is_none() || h_eff_flat.is_none() || out_idx.is_none() || out_val.is_none()) {
    throw std::invalid_argument("x_idx/x_val/h_eff_flat/out_idx/out_val must be device arrays (cannot be None)");
  }

  CudaArrayView x_idx_dev = cuda_array_view_from_object(x_idx, "x_idx");
  CudaArrayView x_val_dev = cuda_array_view_from_object(x_val, "x_val");
  CudaArrayView h_dev = cuda_array_view_from_object(h_eff_flat, "h_eff_flat");
  CudaArrayView out_idx_dev = cuda_array_view_from_object(out_idx, "out_idx");
  CudaArrayView out_val_dev = cuda_array_view_from_object(out_val, "out_val");

  require_typestr(x_idx_dev, "x_idx", "<i4");
  require_typestr(x_val_dev, "x_val", "<f8");
  require_typestr(h_dev, "h_eff_flat", "<f8");
  require_typestr(out_idx_dev, "out_idx", "<i4");
  require_typestr(out_val_dev, "out_val", "<f8");

  if (out_idx_dev.read_only || out_val_dev.read_only) {
    throw std::invalid_argument("out_idx/out_val must be writable");
  }

  if (x_idx_dev.shape.size() != 1 || x_val_dev.shape.size() != 1) {
    throw std::invalid_argument("x_idx and x_val must be 1D arrays");
  }
  if (h_dev.shape.size() != 1) {
    throw std::invalid_argument("h_eff_flat must be a 1D array");
  }
  if (out_idx_dev.shape.size() != 1 || out_val_dev.shape.size() != 1) {
    throw std::invalid_argument("out_idx and out_val must be 1D arrays");
  }

  int64_t m64 = x_idx_dev.shape[0];
  if (m64 < 0) {
    throw std::invalid_argument("invalid x_idx shape");
  }
  if (x_val_dev.shape[0] != m64) {
    throw std::invalid_argument("x_val must have the same length as x_idx");
  }
  if (m64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("x_idx length too large for int32 kernel interface");
  }
  int m = (int)m64;

  int64_t nops64 = (int64_t)drt.norb * (int64_t)drt.norb;
  if (h_dev.shape[0] != nops64) {
    throw std::invalid_argument("h_eff_flat must have shape (norb*norb,)");
  }

  int64_t out_len64 = m64 * (int64_t)nspawn;
  if (out_len64 < 0) {
    throw std::invalid_argument("invalid output length");
  }
  if (out_idx_dev.shape[0] != out_len64 || out_val_dev.shape[0] != out_len64) {
    throw std::invalid_argument("out_idx/out_val must have shape (len(x_idx)*nspawn,)");
  }
  if (m == 0) return;

  if (!x_idx_dev.strides_bytes.empty()) {
    if (x_idx_dev.strides_bytes.size() != 1 || x_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("x_idx must be contiguous");
    }
  }
  if (!x_val_dev.strides_bytes.empty()) {
    if (x_val_dev.strides_bytes.size() != 1 || x_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x_val must be contiguous");
    }
  }
  if (!h_dev.strides_bytes.empty()) {
    if (h_dev.strides_bytes.size() != 1 || h_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("h_eff_flat must be contiguous");
    }
  }
  if (!out_idx_dev.strides_bytes.empty()) {
    if (out_idx_dev.strides_bytes.size() != 1 || out_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("out_idx must be contiguous");
    }
  }
  if (!out_val_dev.strides_bytes.empty()) {
    if (out_val_dev.strides_bytes.size() != 1 || out_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out_val must be contiguous");
    }
  }

  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  throw_on_cuda_error(cudaMemsetAsync(out_idx_dev.ptr, 0xFF, (size_t)out_len64 * sizeof(int32_t), stream_t), "cudaMemsetAsync(out_idx=-1)");
  throw_on_cuda_error(cudaMemsetAsync(out_val_dev.ptr, 0, (size_t)out_len64 * sizeof(double), stream_t), "cudaMemsetAsync(out_val=0)");

  throw_on_cuda_error(
      guga_qmc_spawn_one_body_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          (const int32_t*)x_idx_dev.ptr,
          (const double*)x_val_dev.ptr,
          m,
          (const double*)h_dev.ptr,
          eps,
          nspawn,
          seed,
          initiator_t,
          (int32_t*)out_idx_dev.ptr,
          (double*)out_val_dev.ptr,
          stream_t,
          threads),
      "guga_qmc_spawn_one_body_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_spawn_one_body)");
  }
}

void qmc_spawn_hamiltonian_inplace_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    py::object x_idx,
    py::object x_val,
    py::object h_base_flat,
    py::object eri_mat,
    py::object out_idx,
    py::object out_val,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    int threads,
    uint64_t stream,
    bool sync) {
  if (drt.child == nullptr || drt.node_twos == nullptr || drt.child_prefix == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (state.steps == nullptr || state.nodes == nullptr) {
    throw std::runtime_error("DeviceStateCache is not initialized");
  }
  if (drt.norb != state.norb) {
    throw std::invalid_argument("DeviceDRT and DeviceStateCache have inconsistent norb");
  }
  if (drt.norb <= 0 || drt.norb > MAX_NORB) {
    throw std::invalid_argument("norb out of supported range for current kernels");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (nspawn_one < 0 || nspawn_two < 0) {
    throw std::invalid_argument("nspawn_one/nspawn_two must be >= 0");
  }
  if (nspawn_one == 0 && nspawn_two == 0) {
    throw std::invalid_argument("at least one of nspawn_one or nspawn_two must be > 0");
  }

  if (x_idx.is_none() || x_val.is_none() || h_base_flat.is_none() || eri_mat.is_none() || out_idx.is_none() || out_val.is_none()) {
    throw std::invalid_argument("x_idx/x_val/h_base_flat/eri_mat/out_idx/out_val must be device arrays (cannot be None)");
  }

  CudaArrayView x_idx_dev = cuda_array_view_from_object(x_idx, "x_idx");
  CudaArrayView x_val_dev = cuda_array_view_from_object(x_val, "x_val");
  CudaArrayView h_base_dev = cuda_array_view_from_object(h_base_flat, "h_base_flat");
  CudaArrayView eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  CudaArrayView out_idx_dev = cuda_array_view_from_object(out_idx, "out_idx");
  CudaArrayView out_val_dev = cuda_array_view_from_object(out_val, "out_val");

  require_typestr(x_idx_dev, "x_idx", "<i4");
  require_typestr(x_val_dev, "x_val", "<f8");
  require_typestr(h_base_dev, "h_base_flat", "<f8");
  require_typestr(eri_dev, "eri_mat", "<f8");
  require_typestr(out_idx_dev, "out_idx", "<i4");
  require_typestr(out_val_dev, "out_val", "<f8");

  if (out_idx_dev.read_only || out_val_dev.read_only) {
    throw std::invalid_argument("out_idx/out_val must be writable");
  }

  if (x_idx_dev.shape.size() != 1 || x_val_dev.shape.size() != 1) {
    throw std::invalid_argument("x_idx and x_val must be 1D arrays");
  }
  if (h_base_dev.shape.size() != 1) {
    throw std::invalid_argument("h_base_flat must be a 1D array");
  }
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
    throw std::invalid_argument("eri_mat must have shape (nops,nops)");
  }
  if (out_idx_dev.shape.size() != 1 || out_val_dev.shape.size() != 1) {
    throw std::invalid_argument("out_idx and out_val must be 1D arrays");
  }

  int64_t m64 = x_idx_dev.shape[0];
  if (m64 < 0) {
    throw std::invalid_argument("invalid x_idx shape");
  }
  if (x_val_dev.shape[0] != m64) {
    throw std::invalid_argument("x_val must have the same length as x_idx");
  }
  if (m64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("x_idx length too large for int32 kernel interface");
  }
  int m = (int)m64;

  int64_t nops64 = (int64_t)drt.norb * (int64_t)drt.norb;
  if (h_base_dev.shape[0] != nops64) {
    throw std::invalid_argument("h_base_flat must have shape (norb*norb,)");
  }
  if (eri_dev.shape[0] != nops64) {
    throw std::invalid_argument("eri_mat must have shape (norb*norb, norb*norb)");
  }

  int nspawn_total = nspawn_one + nspawn_two;
  int64_t out_len64 = m64 * (int64_t)nspawn_total;
  if (out_len64 < 0) {
    throw std::invalid_argument("invalid output length");
  }
  if (out_idx_dev.shape[0] != out_len64 || out_val_dev.shape[0] != out_len64) {
    throw std::invalid_argument("out_idx/out_val must have shape (len(x_idx)*(nspawn_one+nspawn_two),)");
  }
  if (m == 0) return;

  if (!x_idx_dev.strides_bytes.empty()) {
    if (x_idx_dev.strides_bytes.size() != 1 || x_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("x_idx must be contiguous");
    }
  }
  if (!x_val_dev.strides_bytes.empty()) {
    if (x_val_dev.strides_bytes.size() != 1 || x_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x_val must be contiguous");
    }
  }
  if (!h_base_dev.strides_bytes.empty()) {
    if (h_base_dev.strides_bytes.size() != 1 || h_base_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("h_base_flat must be contiguous");
    }
  }
  if (!eri_dev.strides_bytes.empty()) {
    if (eri_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("eri_mat strides must have length 2");
    }
    int64_t s0 = eri_dev.strides_bytes[0];
    int64_t s1 = eri_dev.strides_bytes[1];
    if (s0 != (int64_t)nops64 * (int64_t)sizeof(double) || s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
    }
  }
  if (!out_idx_dev.strides_bytes.empty()) {
    if (out_idx_dev.strides_bytes.size() != 1 || out_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
      throw std::invalid_argument("out_idx must be contiguous");
    }
  }
  if (!out_val_dev.strides_bytes.empty()) {
    if (out_val_dev.strides_bytes.size() != 1 || out_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out_val must be contiguous");
    }
  }

  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  throw_on_cuda_error(cudaMemsetAsync(out_idx_dev.ptr, 0xFF, (size_t)out_len64 * sizeof(int32_t), stream_t), "cudaMemsetAsync(out_idx=-1)");
  throw_on_cuda_error(cudaMemsetAsync(out_val_dev.ptr, 0, (size_t)out_len64 * sizeof(double), stream_t), "cudaMemsetAsync(out_val=0)");

  throw_on_cuda_error(
      guga_qmc_spawn_hamiltonian_launch_stream(
          drt.child,
          drt.node_twos,
          drt.child_prefix,
          state.steps,
          state.nodes,
          state.ncsf,
          drt.norb,
          (const int32_t*)x_idx_dev.ptr,
          (const double*)x_val_dev.ptr,
          m,
          (const double*)h_base_dev.ptr,
          (const double*)eri_dev.ptr,
          eps,
          nspawn_one,
          nspawn_two,
          seed,
          initiator_t,
          (int32_t*)out_idx_dev.ptr,
          (double*)out_val_dev.ptr,
          stream_t,
          threads),
      "guga_qmc_spawn_hamiltonian_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_spawn_hamiltonian)");
  }
}

void qmc_spawn_hamiltonian_u64_inplace_device(
    const DeviceDRT& drt,
    py::object x_key,
    py::object x_val,
    py::object h_base_flat,
    py::object eri_mat,
    py::object out_key,
    py::object out_val,
    double eps,
    int nspawn_one,
    int nspawn_two,
    uint64_t seed,
    double initiator_t,
    int threads,
    uint64_t stream,
    bool sync,
    py::object pair_alias_prob,
    py::object pair_alias_idx,
    py::object pair_norm,
    double pair_norm_sum,
    int pair_sampling_mode) {
  if (drt.child == nullptr || drt.node_twos == nullptr) {
    throw std::runtime_error("DeviceDRT is not initialized");
  }
  if (drt.norb <= 0 || drt.norb > 32) {
    throw std::invalid_argument("Key64 QMC kernels require norb in 1..32");
  }
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (nspawn_one < 0 || nspawn_two < 0) {
    throw std::invalid_argument("nspawn_one/nspawn_two must be >= 0");
  }
  if (nspawn_one == 0 && nspawn_two == 0) {
    throw std::invalid_argument("at least one of nspawn_one or nspawn_two must be > 0");
  }

  if (x_key.is_none() || x_val.is_none() || h_base_flat.is_none() || eri_mat.is_none() || out_key.is_none() || out_val.is_none()) {
    throw std::invalid_argument("x_key/x_val/h_base_flat/eri_mat/out_key/out_val must be device arrays (cannot be None)");
  }

  CudaArrayView x_key_dev = cuda_array_view_from_object(x_key, "x_key");
  CudaArrayView x_val_dev = cuda_array_view_from_object(x_val, "x_val");
  CudaArrayView h_base_dev = cuda_array_view_from_object(h_base_flat, "h_base_flat");
  CudaArrayView eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  CudaArrayView out_key_dev = cuda_array_view_from_object(out_key, "out_key");
  CudaArrayView out_val_dev = cuda_array_view_from_object(out_val, "out_val");

  require_typestr(x_key_dev, "x_key", "<u8");
  require_typestr(x_val_dev, "x_val", "<f8");
  require_typestr(h_base_dev, "h_base_flat", "<f8");
  require_typestr(eri_dev, "eri_mat", "<f8");
  require_typestr(out_key_dev, "out_key", "<u8");
  require_typestr(out_val_dev, "out_val", "<f8");

  if (out_key_dev.read_only || out_val_dev.read_only) {
    throw std::invalid_argument("out_key/out_val must be writable");
  }

  if (x_key_dev.shape.size() != 1 || x_val_dev.shape.size() != 1) {
    throw std::invalid_argument("x_key and x_val must be 1D arrays");
  }
  if (h_base_dev.shape.size() != 1) {
    throw std::invalid_argument("h_base_flat must be a 1D array");
  }
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != eri_dev.shape[1]) {
    throw std::invalid_argument("eri_mat must have shape (nops,nops)");
  }
  if (out_key_dev.shape.size() != 1 || out_val_dev.shape.size() != 1) {
    throw std::invalid_argument("out_key and out_val must be 1D arrays");
  }

  int64_t m64 = x_key_dev.shape[0];
  if (m64 < 0) {
    throw std::invalid_argument("x_key length must be non-negative");
  }
  if (x_val_dev.shape[0] != m64) {
    throw std::invalid_argument("x_key and x_val must have the same length");
  }
  if (m64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("x_key length out of supported range for int32 kernel interface");
  }
  int m = (int)m64;

  int nops = drt.norb * drt.norb;
  int64_t nops64 = (int64_t)nops;
  if (h_base_dev.shape[0] != nops64) {
    throw std::invalid_argument("h_base_flat must have length norb*norb");
  }
  if (eri_dev.shape[0] != nops64) {
    throw std::invalid_argument("eri_mat must have shape (norb*norb, norb*norb)");
  }

  int64_t nspawn_total64 = (int64_t)nspawn_one + (int64_t)nspawn_two;
  if (nspawn_total64 < 0) {
    throw std::invalid_argument("nspawn_one+nspawn_two out of supported range");
  }
  if (m64 != 0 && nspawn_total64 > (std::numeric_limits<int64_t>::max() / m64)) {
    throw std::invalid_argument("output length out of supported range");
  }
  int64_t out_len64 = m64 * nspawn_total64;
  if (out_len64 < 0) {
    throw std::invalid_argument("output length out of supported range");
  }
  if (out_key_dev.shape[0] < out_len64 || out_val_dev.shape[0] < out_len64) {
    throw std::invalid_argument("out_key/out_val must have length >= m*(nspawn_one+nspawn_two)");
  }

  if (!x_key_dev.strides_bytes.empty()) {
    if (x_key_dev.strides_bytes.size() != 1 || x_key_dev.strides_bytes[0] != (int64_t)sizeof(uint64_t)) {
      throw std::invalid_argument("x_key must be contiguous");
    }
  }
  if (!x_val_dev.strides_bytes.empty()) {
    if (x_val_dev.strides_bytes.size() != 1 || x_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("x_val must be contiguous");
    }
  }
  if (!h_base_dev.strides_bytes.empty()) {
    if (h_base_dev.strides_bytes.size() != 1 || h_base_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("h_base_flat must be contiguous");
    }
  }
  if (!eri_dev.strides_bytes.empty()) {
    if (eri_dev.strides_bytes.size() != 2) {
      throw std::invalid_argument("eri_mat strides must have length 2");
    }
    int64_t s0 = eri_dev.strides_bytes[0];
    int64_t s1 = eri_dev.strides_bytes[1];
    if (s0 != (int64_t)nops64 * (int64_t)sizeof(double) || s1 != (int64_t)sizeof(double)) {
      throw std::invalid_argument("eri_mat must be C-contiguous with no padding");
    }
  }
  if (!out_key_dev.strides_bytes.empty()) {
    if (out_key_dev.strides_bytes.size() != 1 || out_key_dev.strides_bytes[0] != (int64_t)sizeof(uint64_t)) {
      throw std::invalid_argument("out_key must be contiguous");
    }
  }
  if (!out_val_dev.strides_bytes.empty()) {
    if (out_val_dev.strides_bytes.size() != 1 || out_val_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
      throw std::invalid_argument("out_val must be contiguous");
    }
  }

  const float* pair_alias_prob_ptr = nullptr;
  const int32_t* pair_alias_idx_ptr = nullptr;
  const double* pair_norm_ptr = nullptr;
  double pair_norm_sum_in = pair_norm_sum;
  int pair_sampling_mode_in = pair_sampling_mode;
  if (pair_sampling_mode_in != 0) {
    if (pair_sampling_mode_in != 1) {
      throw std::invalid_argument("pair_sampling_mode must be 0 (uniform) or 1 (pair_norm alias)");
    }
    if (pair_alias_prob.is_none() || pair_alias_idx.is_none() || pair_norm.is_none()) {
      throw std::invalid_argument("pair_alias_prob/pair_alias_idx/pair_norm must be provided when pair_sampling_mode!=0");
    }
    if (!std::isfinite(pair_norm_sum_in) || pair_norm_sum_in <= 0.0) {
      throw std::invalid_argument("pair_norm_sum must be finite and > 0 when pair_sampling_mode!=0");
    }

    CudaArrayView pair_alias_prob_dev = cuda_array_view_from_object(pair_alias_prob, "pair_alias_prob");
    CudaArrayView pair_alias_idx_dev = cuda_array_view_from_object(pair_alias_idx, "pair_alias_idx");
    CudaArrayView pair_norm_dev = cuda_array_view_from_object(pair_norm, "pair_norm");

    require_typestr(pair_alias_prob_dev, "pair_alias_prob", "<f4");
    require_typestr(pair_alias_idx_dev, "pair_alias_idx", "<i4");
    require_typestr(pair_norm_dev, "pair_norm", "<f8");

    if (pair_alias_prob_dev.shape.size() != 1 || pair_alias_idx_dev.shape.size() != 1 || pair_norm_dev.shape.size() != 1) {
      throw std::invalid_argument("pair_alias_prob/pair_alias_idx/pair_norm must be 1D arrays");
    }
    if (pair_alias_prob_dev.shape[0] != nops64 || pair_alias_idx_dev.shape[0] != nops64 || pair_norm_dev.shape[0] != nops64) {
      throw std::invalid_argument("pair_alias_prob/pair_alias_idx/pair_norm must have length norb*norb");
    }
    if (!pair_alias_prob_dev.strides_bytes.empty()) {
      if (pair_alias_prob_dev.strides_bytes.size() != 1 || pair_alias_prob_dev.strides_bytes[0] != (int64_t)sizeof(float)) {
        throw std::invalid_argument("pair_alias_prob must be contiguous");
      }
    }
    if (!pair_alias_idx_dev.strides_bytes.empty()) {
      if (pair_alias_idx_dev.strides_bytes.size() != 1 || pair_alias_idx_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) {
        throw std::invalid_argument("pair_alias_idx must be contiguous");
      }
    }
    if (!pair_norm_dev.strides_bytes.empty()) {
      if (pair_norm_dev.strides_bytes.size() != 1 || pair_norm_dev.strides_bytes[0] != (int64_t)sizeof(double)) {
        throw std::invalid_argument("pair_norm must be contiguous");
      }
    }

    pair_alias_prob_ptr = (const float*)pair_alias_prob_dev.ptr;
    pair_alias_idx_ptr = (const int32_t*)pair_alias_idx_dev.ptr;
    pair_norm_ptr = (const double*)pair_norm_dev.ptr;
  }

  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream);
  throw_on_cuda_error(cudaMemsetAsync(out_key_dev.ptr, 0xFF, (size_t)out_len64 * sizeof(uint64_t), stream_t), "cudaMemsetAsync(out_key=INVALID)");
  throw_on_cuda_error(cudaMemsetAsync(out_val_dev.ptr, 0, (size_t)out_len64 * sizeof(double), stream_t), "cudaMemsetAsync(out_val=0)");

  throw_on_cuda_error(
      guga_qmc_spawn_hamiltonian_u64_f64_launch_stream(
          drt.child,
          drt.node_twos,
          drt.norb,
          (const uint64_t*)x_key_dev.ptr,
          (const double*)x_val_dev.ptr,
          m,
          (const double*)h_base_dev.ptr,
          (const double*)eri_dev.ptr,
          pair_alias_prob_ptr,
          pair_alias_idx_ptr,
          pair_norm_ptr,
          pair_norm_sum_in,
          pair_sampling_mode_in,
          eps,
          nspawn_one,
          nspawn_two,
          seed,
          initiator_t,
          (uint64_t*)out_key_dev.ptr,
          (double*)out_val_dev.ptr,
          stream_t,
          threads),
      "guga_qmc_spawn_hamiltonian_u64_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_spawn_hamiltonian_u64)");
  }
}

void qmc_coalesce_coo_i32_f64_inplace_device(
    py::object idx_in,
    py::object val_in,
    py::object idx_out,
    py::object val_out,
    py::object out_nnz,
    int threads,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (idx_in.is_none() || val_in.is_none() || idx_out.is_none() || val_out.is_none() || out_nnz.is_none()) {
    throw std::invalid_argument("idx_in/val_in/idx_out/val_out/out_nnz must be device arrays (cannot be None)");
  }

  CudaArrayView idx_in_dev = cuda_array_view_from_object(idx_in, "idx_in");
  CudaArrayView val_in_dev = cuda_array_view_from_object(val_in, "val_in");
  CudaArrayView idx_out_dev = cuda_array_view_from_object(idx_out, "idx_out");
  CudaArrayView val_out_dev = cuda_array_view_from_object(val_out, "val_out");
  CudaArrayView nnz_dev = cuda_array_view_from_object(out_nnz, "out_nnz");

  require_typestr(idx_in_dev, "idx_in", "<i4");
  require_typestr(val_in_dev, "val_in", "<f8");
  require_typestr(idx_out_dev, "idx_out", "<i4");
  require_typestr(val_out_dev, "val_out", "<f8");
  require_typestr(nnz_dev, "out_nnz", "<i4");

  if (idx_out_dev.read_only || val_out_dev.read_only || nnz_dev.read_only) {
    throw std::invalid_argument("idx_out/val_out/out_nnz must be writable device arrays");
  }

  if (idx_in_dev.shape.size() != 1 || val_in_dev.shape.size() != 1) {
    throw std::invalid_argument("idx_in and val_in must be 1D device arrays");
  }
  if (idx_in_dev.shape[0] != val_in_dev.shape[0]) {
    throw std::invalid_argument("idx_in and val_in must have the same length");
  }

  int64_t n64 = idx_in_dev.shape[0];
  if (n64 < 0 || n64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("input length out of supported range for int32 kernel interface");
  }
  int n = (int)n64;

  if (idx_out_dev.shape.size() != 1 || val_out_dev.shape.size() != 1) {
    throw std::invalid_argument("idx_out and val_out must be 1D device arrays");
  }
  if (idx_out_dev.shape[0] < n64 || val_out_dev.shape[0] < n64) {
    throw std::invalid_argument("idx_out/val_out must have length >= len(idx_in)");
  }

  if (!idx_in_dev.strides_bytes.empty()) {
    if (idx_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_in must be 1D with valid strides");
    if (idx_in_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_in must be contiguous");
  }
  if (!val_in_dev.strides_bytes.empty()) {
    if (val_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_in must be 1D with valid strides");
    if (val_in_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_in must be contiguous");
  }
  if (!idx_out_dev.strides_bytes.empty()) {
    if (idx_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_out must be 1D with valid strides");
    if (idx_out_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_out must be contiguous");
  }
  if (!val_out_dev.strides_bytes.empty()) {
    if (val_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_out must be 1D with valid strides");
    if (val_out_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_out must be contiguous");
  }

  // out_nnz: allow either scalar (shape=[]) or 1D length-1.
  if (nnz_dev.shape.size() == 1) {
    if (nnz_dev.shape[0] != 1) throw std::invalid_argument("out_nnz must have shape (1,) or ()");
    if (!nnz_dev.strides_bytes.empty()) {
      if (nnz_dev.strides_bytes.size() != 1) throw std::invalid_argument("out_nnz must be 1D with valid strides");
      if (nnz_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("out_nnz must be contiguous");
    }
  } else if (nnz_dev.shape.size() == 0) {
    // scalar: ok
  } else {
    throw std::invalid_argument("out_nnz must have shape (1,) or ()");
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_qmc_coalesce_coo_i32_f64_launch_stream(
          (const int32_t*)idx_in_dev.ptr,
          (const double*)val_in_dev.ptr,
          n,
          (int32_t*)idx_out_dev.ptr,
          (double*)val_out_dev.ptr,
          (int*)nnz_dev.ptr,
          stream_t,
          threads),
      "guga_qmc_coalesce_coo_i32_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_coalesce_coo)");
  }
}

void qmc_phi_pivot_resample_i32_f64_inplace_device(
    py::object idx_in,
    py::object val_in,
    py::object idx_out,
    py::object val_out,
    py::object out_nnz,
    int m,
    int pivot,
    uint64_t seed,
    int threads,
    uint64_t stream,
    bool sync) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument("threads must be in 1..1024");
  }
  if (m < 0 || pivot < 0) {
    throw std::invalid_argument("m and pivot must be >= 0");
  }
  if (idx_in.is_none() || val_in.is_none() || idx_out.is_none() || val_out.is_none() || out_nnz.is_none()) {
    throw std::invalid_argument("idx_in/val_in/idx_out/val_out/out_nnz must be device arrays (cannot be None)");
  }

  CudaArrayView idx_in_dev = cuda_array_view_from_object(idx_in, "idx_in");
  CudaArrayView val_in_dev = cuda_array_view_from_object(val_in, "val_in");
  CudaArrayView idx_out_dev = cuda_array_view_from_object(idx_out, "idx_out");
  CudaArrayView val_out_dev = cuda_array_view_from_object(val_out, "val_out");
  CudaArrayView nnz_dev = cuda_array_view_from_object(out_nnz, "out_nnz");

  require_typestr(idx_in_dev, "idx_in", "<i4");
  require_typestr(val_in_dev, "val_in", "<f8");
  require_typestr(idx_out_dev, "idx_out", "<i4");
  require_typestr(val_out_dev, "val_out", "<f8");
  require_typestr(nnz_dev, "out_nnz", "<i4");

  if (idx_out_dev.read_only || val_out_dev.read_only || nnz_dev.read_only) {
    throw std::invalid_argument("idx_out/val_out/out_nnz must be writable device arrays");
  }

  if (idx_in_dev.shape.size() != 1 || val_in_dev.shape.size() != 1) {
    throw std::invalid_argument("idx_in and val_in must be 1D device arrays");
  }
  if (idx_in_dev.shape[0] != val_in_dev.shape[0]) {
    throw std::invalid_argument("idx_in and val_in must have the same length");
  }

  int64_t n_in64 = idx_in_dev.shape[0];
  if (n_in64 < 0 || n_in64 > (int64_t)std::numeric_limits<int>::max()) {
    throw std::invalid_argument("input length out of supported range for int32 kernel interface");
  }
  int n_in = (int)n_in64;

  if (idx_out_dev.shape.size() != 1 || val_out_dev.shape.size() != 1) {
    throw std::invalid_argument("idx_out and val_out must be 1D device arrays");
  }
  if (idx_out_dev.shape[0] < (int64_t)m || val_out_dev.shape[0] < (int64_t)m) {
    throw std::invalid_argument("idx_out/val_out must have length >= m");
  }

  if (!idx_in_dev.strides_bytes.empty()) {
    if (idx_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_in must be 1D with valid strides");
    if (idx_in_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_in must be contiguous");
  }
  if (!val_in_dev.strides_bytes.empty()) {
    if (val_in_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_in must be 1D with valid strides");
    if (val_in_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_in must be contiguous");
  }
  if (!idx_out_dev.strides_bytes.empty()) {
    if (idx_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("idx_out must be 1D with valid strides");
    if (idx_out_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("idx_out must be contiguous");
  }
  if (!val_out_dev.strides_bytes.empty()) {
    if (val_out_dev.strides_bytes.size() != 1) throw std::invalid_argument("val_out must be 1D with valid strides");
    if (val_out_dev.strides_bytes[0] != (int64_t)sizeof(double)) throw std::invalid_argument("val_out must be contiguous");
  }

  // out_nnz: allow either scalar (shape=[]) or 1D length-1.
  if (nnz_dev.shape.size() == 1) {
    if (nnz_dev.shape[0] != 1) throw std::invalid_argument("out_nnz must have shape (1,) or ()");
    if (!nnz_dev.strides_bytes.empty()) {
      if (nnz_dev.strides_bytes.size() != 1) throw std::invalid_argument("out_nnz must be 1D with valid strides");
      if (nnz_dev.strides_bytes[0] != (int64_t)sizeof(int32_t)) throw std::invalid_argument("out_nnz must be contiguous");
    }
  } else if (nnz_dev.shape.size() == 0) {
    // scalar: ok
  } else {
    throw std::invalid_argument("out_nnz must have shape (1,) or ()");
  }

  cudaStream_t stream_t = (cudaStream_t)stream;
  throw_on_cuda_error(
      guga_qmc_phi_pivot_resample_i32_f64_launch_stream(
          (const int32_t*)idx_in_dev.ptr,
          (const double*)val_in_dev.ptr,
          n_in,
          (int32_t*)idx_out_dev.ptr,
          (double*)val_out_dev.ptr,
          (int*)nnz_dev.ptr,
          m,
          pivot,
          seed,
          stream_t,
          threads),
      "guga_qmc_phi_pivot_resample_i32_f64_launch_stream");

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t), "cudaStreamSynchronize(qmc_phi_pivot_resample)");
  }
}

// =============================================================================
// Fused Hop Kernel Dispatch
// =============================================================================
void fused_hop_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object x,
    py::object eri_mat,
    py::object h_eff_flat,
    py::object y,
    py::object overflow,
    uint64_t stream,
    bool sync,
    bool check_overflow)
{
  int norb = drt.norb;
  int ncsf = state.ncsf;

  auto x_dev = cuda_array_view_from_object(x, "x");
  bool use_f32 = (x_dev.typestr == "<f4");
  if (!use_f32) require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");
  int nops = norb * norb;
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != nops || eri_dev.shape[1] != nops) {
    throw std::invalid_argument("eri_mat must have shape (norb*norb, norb*norb)");
  }

  auto h_dev = cuda_array_view_from_object(h_eff_flat, "h_eff_flat");
  if (use_f32) require_typestr(h_dev, "h_eff_flat", "<f4");
  else require_typestr(h_dev, "h_eff_flat", "<f8");
  if (h_dev.shape.size() != 1 || h_dev.shape[0] != nops) {
    throw std::invalid_argument("h_eff_flat must have shape (norb*norb,)");
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) throw std::invalid_argument("y must be writable");
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) throw std::invalid_argument("overflow must be writable");

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t),
                      "cudaMemsetAsync(overflow=0)");

  // Template dispatch on MAX_NORB and dtype
  int max_norb;
  if (norb <= 8) max_norb = 8;
  else if (norb <= 12) max_norb = 12;
  else if (norb <= 16) max_norb = 16;
  else if (norb <= 20) max_norb = 20;
  else throw std::invalid_argument("fused_hop: norb > 20 not supported");

  #define DISPATCH_FUSED_HOP(NORB, TYPE) \
    guga_fused_hop_launch_stream<NORB, TYPE>( \
        drt.child, drt.node_twos, drt.child_prefix, \
        state.steps, state.nodes, \
        ncsf, norb, j_start, j_count, \
        reinterpret_cast<const TYPE*>(x_dev.ptr), \
        reinterpret_cast<const TYPE*>(eri_dev.ptr), \
        reinterpret_cast<const TYPE*>(h_dev.ptr), \
        reinterpret_cast<TYPE*>(y_dev.ptr), \
        d_overflow, stream_t)

  if (use_f32) {
    switch (max_norb) {
      case 8:  DISPATCH_FUSED_HOP(8, float); break;
      case 12: DISPATCH_FUSED_HOP(12, float); break;
      case 16: DISPATCH_FUSED_HOP(16, float); break;
      case 20: DISPATCH_FUSED_HOP(20, float); break;
    }
  } else {
    switch (max_norb) {
      case 8:  DISPATCH_FUSED_HOP(8, double); break;
      case 12: DISPATCH_FUSED_HOP(12, double); break;
      case 16: DISPATCH_FUSED_HOP(16, double); break;
      case 20: DISPATCH_FUSED_HOP(20, double); break;
    }
  }
  #undef DISPATCH_FUSED_HOP

  throw_on_cuda_error(cudaGetLastError(), "kernel launch(fused_hop)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t),
                        "cudaStreamSynchronize(fused_hop)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("fused_hop overflow (invalid indices or stack overflow)");
  }
}

void fused_hop_phase1_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object x,
    py::object eri_mat,
    py::object h_eff_flat,
    py::object y,
    py::object g_out,
    py::object overflow,
    uint64_t stream,
    bool sync,
    bool check_overflow)
{
  int norb = drt.norb;
  int ncsf = state.ncsf;
  int nops = norb * norb;

  auto x_dev = cuda_array_view_from_object(x, "x");
  bool use_f32 = (x_dev.typestr == "<f4");
  if (!use_f32) require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)ncsf) {
    throw std::invalid_argument("x must have shape (ncsf,)");
  }

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");
  if (eri_dev.shape.size() != 2 || eri_dev.shape[0] != nops || eri_dev.shape[1] != nops) {
    throw std::invalid_argument("eri_mat must have shape (norb*norb, norb*norb)");
  }

  auto h_dev = cuda_array_view_from_object(h_eff_flat, "h_eff_flat");
  if (use_f32) require_typestr(h_dev, "h_eff_flat", "<f4");
  else require_typestr(h_dev, "h_eff_flat", "<f8");
  if (h_dev.shape.size() != 1 || h_dev.shape[0] != nops) {
    throw std::invalid_argument("h_eff_flat must have shape (norb*norb,)");
  }

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) throw std::invalid_argument("y must be writable");
  if (y_dev.shape.size() != 1 || y_dev.shape[0] != (int64_t)ncsf) {
    throw std::invalid_argument("y must have shape (ncsf,)");
  }

  auto g_dev = cuda_array_view_from_object(g_out, "g_out");
  if (use_f32) require_typestr(g_dev, "g_out", "<f4");
  else require_typestr(g_dev, "g_out", "<f8");
  if (g_dev.read_only) throw std::invalid_argument("g_out must be writable");
  if (g_dev.shape.size() != 2 || g_dev.shape[0] < (int64_t)j_count || g_dev.shape[1] != nops) {
    throw std::invalid_argument("g_out must have shape (>=j_count, nops)");
  }

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) throw std::invalid_argument("overflow must be writable");

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t),
                      "cudaMemsetAsync(overflow=0)");

  int max_norb;
  if (norb <= 8) max_norb = 8;
  else if (norb <= 12) max_norb = 12;
  else if (norb <= 16) max_norb = 16;
  else if (norb <= 20) max_norb = 20;
  else throw std::invalid_argument("fused_hop_phase1: norb > 20 not supported");

  #define DISPATCH_FUSED_HOP_P1(NORB, TYPE) \
    guga_fused_hop_phase1_launch_stream<NORB, TYPE>( \
        drt.child, drt.node_twos, drt.child_prefix, \
        state.steps, state.nodes, \
        ncsf, norb, j_start, j_count, \
        reinterpret_cast<const TYPE*>(x_dev.ptr), \
        reinterpret_cast<const TYPE*>(eri_dev.ptr), \
        reinterpret_cast<const TYPE*>(h_dev.ptr), \
        reinterpret_cast<TYPE*>(y_dev.ptr), \
        reinterpret_cast<TYPE*>(g_dev.ptr), \
        d_overflow, stream_t)

  if (use_f32) {
    switch (max_norb) {
      case 8:  DISPATCH_FUSED_HOP_P1(8, float); break;
      case 12: DISPATCH_FUSED_HOP_P1(12, float); break;
      case 16: DISPATCH_FUSED_HOP_P1(16, float); break;
      case 20: DISPATCH_FUSED_HOP_P1(20, float); break;
    }
  } else {
    switch (max_norb) {
      case 8:  DISPATCH_FUSED_HOP_P1(8, double); break;
      case 12: DISPATCH_FUSED_HOP_P1(12, double); break;
      case 16: DISPATCH_FUSED_HOP_P1(16, double); break;
      case 20: DISPATCH_FUSED_HOP_P1(20, double); break;
    }
  }
  #undef DISPATCH_FUSED_HOP_P1

  throw_on_cuda_error(cudaGetLastError(), "kernel launch(fused_hop_phase1)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t),
                        "cudaStreamSynchronize(fused_hop_phase1)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("fused_hop_phase1 overflow (invalid indices or stack overflow)");
  }
}

void fused_hop_phase1_coo_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object x,
    py::object eri_mat,
    py::object h_eff_flat,
    py::object y,
    py::object g_out,
    py::object overflow,
    py::object coo_nnz_counter,
    py::object coo_j_local,
    py::object coo_k,
    py::object coo_pq,
    py::object coo_w2,
    int max_coo,
    uint64_t stream,
    bool sync,
    bool check_overflow)
{
  int norb = drt.norb;
  int ncsf = state.ncsf;
  int nops = norb * norb;

  auto x_dev = cuda_array_view_from_object(x, "x");
  bool use_f32 = (x_dev.typestr == "<f4");
  if (!use_f32) require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)ncsf)
    throw std::invalid_argument("x must have shape (ncsf,)");

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");

  auto h_dev = cuda_array_view_from_object(h_eff_flat, "h_eff_flat");
  if (use_f32) require_typestr(h_dev, "h_eff_flat", "<f4");
  else require_typestr(h_dev, "h_eff_flat", "<f8");

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) throw std::invalid_argument("y must be writable");

  auto g_dev = cuda_array_view_from_object(g_out, "g_out");
  if (use_f32) require_typestr(g_dev, "g_out", "<f4");
  else require_typestr(g_dev, "g_out", "<f8");
  if (g_dev.read_only) throw std::invalid_argument("g_out must be writable");

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) throw std::invalid_argument("overflow must be writable");

  auto cnt_dev = cuda_array_view_from_object(coo_nnz_counter, "coo_nnz_counter");
  require_typestr(cnt_dev, "coo_nnz_counter", "<i4");
  if (cnt_dev.read_only) throw std::invalid_argument("coo_nnz_counter must be writable");

  auto jl_dev = cuda_array_view_from_object(coo_j_local, "coo_j_local");
  require_typestr(jl_dev, "coo_j_local", "<i4");
  if (jl_dev.read_only) throw std::invalid_argument("coo_j_local must be writable");

  auto ck_dev = cuda_array_view_from_object(coo_k, "coo_k");
  require_typestr(ck_dev, "coo_k", "<i4");
  if (ck_dev.read_only) throw std::invalid_argument("coo_k must be writable");

  auto pq_dev = cuda_array_view_from_object(coo_pq, "coo_pq");
  require_typestr(pq_dev, "coo_pq", "<i2");
  if (pq_dev.read_only) throw std::invalid_argument("coo_pq must be writable");

  auto w2_dev = cuda_array_view_from_object(coo_w2, "coo_w2");
  if (use_f32) require_typestr(w2_dev, "coo_w2", "<f4");
  else require_typestr(w2_dev, "coo_w2", "<f8");
  if (w2_dev.read_only) throw std::invalid_argument("coo_w2 must be writable");

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t),
                      "cudaMemsetAsync(overflow=0)");

  int max_norb;
  if (norb <= 8) max_norb = 8;
  else if (norb <= 12) max_norb = 12;
  else if (norb <= 16) max_norb = 16;
  else if (norb <= 20) max_norb = 20;
  else throw std::invalid_argument("fused_hop_phase1_coo: norb > 20 not supported");

  #define DISPATCH_P1COO(NORB, TYPE) \
    guga_fused_hop_phase1_coo_launch_stream<NORB, TYPE>( \
        drt.child, drt.node_twos, drt.child_prefix, \
        state.steps, state.nodes, \
        ncsf, norb, j_start, j_count, \
        reinterpret_cast<const TYPE*>(x_dev.ptr), \
        reinterpret_cast<const TYPE*>(eri_dev.ptr), \
        reinterpret_cast<const TYPE*>(h_dev.ptr), \
        reinterpret_cast<TYPE*>(y_dev.ptr), \
        reinterpret_cast<TYPE*>(g_dev.ptr), \
        d_overflow, \
        reinterpret_cast<int*>(cnt_dev.ptr), \
        reinterpret_cast<int32_t*>(jl_dev.ptr), \
        reinterpret_cast<int32_t*>(ck_dev.ptr), \
        reinterpret_cast<int16_t*>(pq_dev.ptr), \
        reinterpret_cast<TYPE*>(w2_dev.ptr), \
        max_coo, stream_t)

  if (use_f32) {
    switch (max_norb) {
      case 8:  DISPATCH_P1COO(8, float); break;
      case 12: DISPATCH_P1COO(12, float); break;
      case 16: DISPATCH_P1COO(16, float); break;
      case 20: DISPATCH_P1COO(20, float); break;
    }
  } else {
    switch (max_norb) {
      case 8:  DISPATCH_P1COO(8, double); break;
      case 12: DISPATCH_P1COO(12, double); break;
      case 16: DISPATCH_P1COO(16, double); break;
      case 20: DISPATCH_P1COO(20, double); break;
    }
  }
  #undef DISPATCH_P1COO

  throw_on_cuda_error(cudaGetLastError(), "kernel launch(fused_hop_phase1_coo)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t),
                        "cudaStreamSynchronize(fused_hop_phase1_coo)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("fused_hop_phase1_coo overflow");
  }
}

void fused_hop_phase1_coo_merged_device(
    const DeviceDRT& drt,
    const DeviceStateCache& state,
    int j_start,
    int j_count,
    py::object x,
    py::object eri_mat,
    py::object h_eff_flat,
    py::object y,
    py::object g_out,
    py::object overflow,
    py::object coo_nnz_counter,
    py::object coo_j_local,
    py::object coo_k,
    py::object coo_pq,
    py::object coo_w2,
    int max_coo,
    uint64_t stream,
    bool sync,
    bool check_overflow)
{
  int norb = drt.norb;
  int ncsf = state.ncsf;

  auto x_dev = cuda_array_view_from_object(x, "x");
  bool use_f32 = (x_dev.typestr == "<f4");
  if (!use_f32) require_typestr(x_dev, "x", "<f8");
  if (x_dev.shape.size() != 1 || x_dev.shape[0] != (int64_t)ncsf)
    throw std::invalid_argument("x must have shape (ncsf,)");

  auto eri_dev = cuda_array_view_from_object(eri_mat, "eri_mat");
  if (use_f32) require_typestr(eri_dev, "eri_mat", "<f4");
  else require_typestr(eri_dev, "eri_mat", "<f8");

  auto h_dev = cuda_array_view_from_object(h_eff_flat, "h_eff_flat");
  if (use_f32) require_typestr(h_dev, "h_eff_flat", "<f4");
  else require_typestr(h_dev, "h_eff_flat", "<f8");

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) throw std::invalid_argument("y must be writable");

  auto g_dev = cuda_array_view_from_object(g_out, "g_out");
  if (use_f32) require_typestr(g_dev, "g_out", "<f4");
  else require_typestr(g_dev, "g_out", "<f8");
  if (g_dev.read_only) throw std::invalid_argument("g_out must be writable");

  auto overflow_dev = cuda_array_view_from_object(overflow, "overflow");
  require_typestr(overflow_dev, "overflow", "<i4");
  if (overflow_dev.read_only) throw std::invalid_argument("overflow must be writable");

  auto cnt_dev = cuda_array_view_from_object(coo_nnz_counter, "coo_nnz_counter");
  require_typestr(cnt_dev, "coo_nnz_counter", "<i4");
  if (cnt_dev.read_only) throw std::invalid_argument("coo_nnz_counter must be writable");

  auto jl_dev = cuda_array_view_from_object(coo_j_local, "coo_j_local");
  require_typestr(jl_dev, "coo_j_local", "<i4");
  if (jl_dev.read_only) throw std::invalid_argument("coo_j_local must be writable");

  auto ck_dev = cuda_array_view_from_object(coo_k, "coo_k");
  require_typestr(ck_dev, "coo_k", "<i4");
  if (ck_dev.read_only) throw std::invalid_argument("coo_k must be writable");

  auto pq_dev = cuda_array_view_from_object(coo_pq, "coo_pq");
  require_typestr(pq_dev, "coo_pq", "<i2");
  if (pq_dev.read_only) throw std::invalid_argument("coo_pq must be writable");

  auto w2_dev = cuda_array_view_from_object(coo_w2, "coo_w2");
  if (use_f32) require_typestr(w2_dev, "coo_w2", "<f4");
  else require_typestr(w2_dev, "coo_w2", "<f8");
  if (w2_dev.read_only) throw std::invalid_argument("coo_w2 must be writable");

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
    else if (x_dev.stream) stream_u = x_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  int* d_overflow = reinterpret_cast<int*>(overflow_dev.ptr);
  throw_on_cuda_error(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream_t),
                      "cudaMemsetAsync(overflow=0)");

  int max_norb;
  if (norb <= 8) max_norb = 8;
  else if (norb <= 12) max_norb = 12;
  else if (norb <= 16) max_norb = 16;
  else if (norb <= 20) max_norb = 20;
  else throw std::invalid_argument("fused_hop_phase1_coo_merged: norb > 20");

  #define DISPATCH_P1COO_M(NORB, TYPE) \
    guga_fused_hop_phase1_coo_merged_launch_stream<NORB, TYPE>( \
        drt.child, drt.node_twos, drt.child_prefix, \
        state.steps, state.nodes, \
        ncsf, norb, j_start, j_count, \
        reinterpret_cast<const TYPE*>(x_dev.ptr), \
        reinterpret_cast<const TYPE*>(eri_dev.ptr), \
        reinterpret_cast<const TYPE*>(h_dev.ptr), \
        reinterpret_cast<TYPE*>(y_dev.ptr), \
        reinterpret_cast<TYPE*>(g_dev.ptr), \
        d_overflow, \
        reinterpret_cast<int*>(cnt_dev.ptr), \
        reinterpret_cast<int32_t*>(jl_dev.ptr), \
        reinterpret_cast<int32_t*>(ck_dev.ptr), \
        reinterpret_cast<int16_t*>(pq_dev.ptr), \
        reinterpret_cast<TYPE*>(w2_dev.ptr), \
        max_coo, stream_t)

  if (use_f32) {
    switch (max_norb) {
      case 8:  DISPATCH_P1COO_M(8, float); break;
      case 12: DISPATCH_P1COO_M(12, float); break;
      case 16: DISPATCH_P1COO_M(16, float); break;
      case 20: DISPATCH_P1COO_M(20, float); break;
    }
  } else {
    switch (max_norb) {
      case 8:  DISPATCH_P1COO_M(8, double); break;
      case 12: DISPATCH_P1COO_M(12, double); break;
      case 16: DISPATCH_P1COO_M(16, double); break;
      case 20: DISPATCH_P1COO_M(20, double); break;
    }
  }
  #undef DISPATCH_P1COO_M

  throw_on_cuda_error(cudaGetLastError(),
                      "kernel launch(fused_hop_phase1_coo_merged)");

  int h_overflow = 0;
  if (check_overflow) {
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_overflow, d_overflow, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_t),
        "D2H overflow");
  }
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t),
                        "cudaStreamSynchronize(fused_hop_phase1_coo_merged)");
  }
  if (check_overflow && h_overflow) {
    throw std::runtime_error("fused_hop_phase1_coo_merged overflow");
  }
}

void coo_scatter_device(
    py::object coo_j_local,
    py::object coo_k,
    py::object coo_pq,
    py::object coo_w2,
    py::object g_tile,
    int nops,
    int nnz,
    py::object y,
    uint64_t stream,
    bool sync)
{
  auto jl_dev = cuda_array_view_from_object(coo_j_local, "coo_j_local");
  require_typestr(jl_dev, "coo_j_local", "<i4");

  auto ck_dev = cuda_array_view_from_object(coo_k, "coo_k");
  require_typestr(ck_dev, "coo_k", "<i4");

  auto pq_dev = cuda_array_view_from_object(coo_pq, "coo_pq");
  require_typestr(pq_dev, "coo_pq", "<i2");

  auto w2_dev = cuda_array_view_from_object(coo_w2, "coo_w2");
  bool use_f32 = (w2_dev.typestr == "<f4");
  if (!use_f32) require_typestr(w2_dev, "coo_w2", "<f8");

  auto g_dev = cuda_array_view_from_object(g_tile, "g_tile");
  if (use_f32) require_typestr(g_dev, "g_tile", "<f4");
  else require_typestr(g_dev, "g_tile", "<f8");

  auto y_dev = cuda_array_view_from_object(y, "y");
  if (use_f32) require_typestr(y_dev, "y", "<f4");
  else require_typestr(y_dev, "y", "<f8");
  if (y_dev.read_only) throw std::invalid_argument("y must be writable");

  uint64_t stream_u = stream;
  if (stream_u == 0) {
    if (y_dev.stream) stream_u = y_dev.stream;
  }
  cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream_u);

  if (use_f32) {
    guga_coo_scatter_launch_stream<float>(
        reinterpret_cast<const int32_t*>(jl_dev.ptr),
        reinterpret_cast<const int32_t*>(ck_dev.ptr),
        reinterpret_cast<const int16_t*>(pq_dev.ptr),
        reinterpret_cast<const float*>(w2_dev.ptr),
        reinterpret_cast<const float*>(g_dev.ptr),
        nops, nnz,
        reinterpret_cast<float*>(y_dev.ptr),
        stream_t);
  } else {
    guga_coo_scatter_launch_stream<double>(
        reinterpret_cast<const int32_t*>(jl_dev.ptr),
        reinterpret_cast<const int32_t*>(ck_dev.ptr),
        reinterpret_cast<const int16_t*>(pq_dev.ptr),
        reinterpret_cast<const double*>(w2_dev.ptr),
        reinterpret_cast<const double*>(g_dev.ptr),
        nops, nnz,
        reinterpret_cast<double*>(y_dev.ptr),
        stream_t);
  }

  throw_on_cuda_error(cudaGetLastError(), "kernel launch(coo_scatter)");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream_t),
                        "cudaStreamSynchronize(coo_scatter)");
  }
}

}  // namespace

PYBIND11_MODULE(_guga_cuda_ext, m) {
  m.doc() = "CUDA kernels for ASUKA GUGA/DRT CI acceleration.";

  m.def("device_info", &device_info);
  m.def("mem_info", &mem_info);

  m.def(
      "ell_spmv_f64_inplace_device",
      &ell_spmv_f64_inplace_device,
      py::arg("col_idx"),
      py::arg("val"),
      py::arg("x"),
      py::arg("y"),
      py::arg("threads") = 128,
      py::arg("add") = false,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "ell_spmm_f64_inplace_device",
      &ell_spmm_f64_inplace_device,
      py::arg("col_idx"),
      py::arg("val"),
      py::arg("x"),
      py::arg("y"),
      py::arg("threads") = 128,
      py::arg("add") = false,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "sell_spmv_f64_inplace_device",
      &sell_spmv_f64_inplace_device,
      py::arg("slice_ptr"),
      py::arg("slice_width"),
      py::arg("col_idx"),
      py::arg("val"),
      py::arg("x"),
      py::arg("y"),
      py::arg("slice_height") = 32,
      py::arg("threads") = 128,
      py::arg("add") = false,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "sell_spmm_f64_inplace_device",
      &sell_spmm_f64_inplace_device,
      py::arg("slice_ptr"),
      py::arg("slice_width"),
      py::arg("col_idx"),
      py::arg("val"),
      py::arg("x"),
      py::arg("y"),
      py::arg("slice_height") = 32,
      py::arg("threads") = 128,
      py::arg("add") = false,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  py::class_<DeviceDRT>(m, "DeviceDRT")
      .def(py::init<>())
      .def("release", &DeviceDRT::release)
      .def_readonly("norb", &DeviceDRT::norb)
      .def_readonly("nnodes", &DeviceDRT::nnodes);

  py::class_<DeviceStateCache>(m, "DeviceStateCache")
      .def(py::init<>())
      .def("release", &DeviceStateCache::release)
      .def_readonly("norb", &DeviceStateCache::norb)
      .def_readonly("ncsf", &DeviceStateCache::ncsf);

  py::class_<TripletFactorsWorkspace>(m, "TripletFactorsWorkspace")
      .def(py::init<>())
      .def("release", &TripletFactorsWorkspace::release)
      .def_readonly("twos_max", &TripletFactorsWorkspace::twos_max);

  m.def(
      "make_triplet_factors_workspace",
      &make_triplet_factors_workspace,
      py::arg("twos_max"),
      py::arg("sixj_211_host"),
      py::arg("t_factor_host"));

  m.def(
      "triplet_apply_contracted_all_m_from_epq_table_inplace_device",
      &triplet_apply_contracted_all_m_from_epq_table_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("x"),
      py::arg("h_re"),
      py::arg("h_im"),
      py::arg("triplet_factors"),
      py::arg("y_re"),
      py::arg("y_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "triplet_apply_contracted_all_m_dfs_inplace_device",
      &triplet_apply_contracted_all_m_dfs_inplace_device,
      py::arg("drt_bra"),
      py::arg("drt_ket"),
      py::arg("ket_state"),
      py::arg("x"),
      py::arg("h_re"),
      py::arg("h_im"),
      py::arg("triplet_factors"),
      py::arg("y_re"),
      py::arg("y_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("root_bra") = 0,
      py::arg("leaf_bra") = -1,
      py::arg("twos_bra_total") = -1,
      py::arg("twos_ket_total") = -1);

  m.def(
      "triplet_build_rho_all_m_from_epq_table_inplace_device",
      &triplet_build_rho_all_m_from_epq_table_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("c_bra"),
      py::arg("c_ket"),
      py::arg("eta_re"),
      py::arg("eta_im"),
      py::arg("triplet_factors"),
      py::arg("rho_re"),
      py::arg("rho_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "triplet_build_rho_all_m_dfs_inplace_device",
      &triplet_build_rho_all_m_dfs_inplace_device,
      py::arg("drt_bra"),
      py::arg("drt_ket"),
      py::arg("ket_state"),
      py::arg("c_bra"),
      py::arg("c_ket"),
      py::arg("eta_re"),
      py::arg("eta_im"),
      py::arg("triplet_factors"),
      py::arg("rho_re"),
      py::arg("rho_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("root_bra") = 0,
      py::arg("leaf_bra") = -1,
      py::arg("twos_bra_total") = -1,
      py::arg("twos_ket_total") = -1);

  m.def(
      "triplet_build_gm_all_m_from_epq_table_inplace_device",
      &triplet_build_gm_all_m_from_epq_table_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("c_bra"),
      py::arg("c_ket"),
      py::arg("h_re"),
      py::arg("h_im"),
      py::arg("triplet_factors"),
      py::arg("gm_re"),
      py::arg("gm_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "triplet_build_gm_all_m_dfs_inplace_device",
      &triplet_build_gm_all_m_dfs_inplace_device,
      py::arg("drt_bra"),
      py::arg("drt_ket"),
      py::arg("ket_state"),
      py::arg("c_bra"),
      py::arg("c_ket"),
      py::arg("h_re"),
      py::arg("h_im"),
      py::arg("triplet_factors"),
      py::arg("gm_re"),
      py::arg("gm_im"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("root_bra") = 0,
      py::arg("leaf_bra") = -1,
      py::arg("twos_bra_total") = -1,
      py::arg("twos_ket_total") = -1);

  m.def(
      "qmc_spawn_one_body_inplace_device",
      &qmc_spawn_one_body_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("x_idx"),
      py::arg("x_val"),
      py::arg("h_eff_flat"),
      py::arg("out_idx"),
      py::arg("out_val"),
      py::arg("eps"),
      py::arg("nspawn"),
      py::arg("seed"),
      py::arg("initiator_t") = 0.0,
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "qmc_spawn_hamiltonian_inplace_device",
      &qmc_spawn_hamiltonian_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("x_idx"),
      py::arg("x_val"),
      py::arg("h_base_flat"),
      py::arg("eri_mat"),
      py::arg("out_idx"),
      py::arg("out_val"),
      py::arg("eps"),
      py::arg("nspawn_one"),
      py::arg("nspawn_two"),
      py::arg("seed"),
      py::arg("initiator_t") = 0.0,
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "qmc_spawn_hamiltonian_u64_inplace_device",
      &qmc_spawn_hamiltonian_u64_inplace_device,
      py::arg("drt"),
      py::arg("x_key"),
      py::arg("x_val"),
      py::arg("h_base_flat"),
      py::arg("eri_mat"),
      py::arg("out_key"),
      py::arg("out_val"),
      py::arg("eps"),
      py::arg("nspawn_one"),
      py::arg("nspawn_two"),
      py::arg("seed"),
      py::arg("initiator_t") = 0.0,
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("pair_alias_prob") = py::none(),
      py::arg("pair_alias_idx") = py::none(),
      py::arg("pair_norm") = py::none(),
      py::arg("pair_norm_sum") = 0.0,
      py::arg("pair_sampling_mode") = 0);

  m.def(
      "qmc_coalesce_coo_i32_f64_inplace_device",
      &qmc_coalesce_coo_i32_f64_inplace_device,
      py::arg("idx_in"),
      py::arg("val_in"),
      py::arg("idx_out"),
      py::arg("val_out"),
      py::arg("out_nnz"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "qmc_phi_pivot_resample_i32_f64_inplace_device",
      &qmc_phi_pivot_resample_i32_f64_inplace_device,
      py::arg("idx_in"),
      py::arg("val_in"),
      py::arg("idx_out"),
      py::arg("val_out"),
      py::arg("out_nnz"),
      py::arg("m"),
      py::arg("pivot"),
      py::arg("seed"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  py::class_<QmcWorkspace>(m, "QmcWorkspace")
      .def(py::init<int, int>(), py::arg("max_n"), py::arg("max_m"))
      .def("release", &QmcWorkspace::release)
      .def_property_readonly("max_n", [](const QmcWorkspace& self) { return self.max_n; })
      .def_property_readonly("max_m", [](const QmcWorkspace& self) { return self.max_m; })
      .def(
          "coalesce_coo_i32_f64_inplace_device",
          &QmcWorkspace::coalesce_coo_i32_f64_inplace_device,
          py::arg("idx_in"),
          py::arg("val_in"),
          py::arg("idx_out"),
          py::arg("val_out"),
          py::arg("out_nnz"),
          py::arg("n"),
          py::arg("threads") = 256,
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "phi_pivot_resample_i32_f64_inplace_device",
          &QmcWorkspace::phi_pivot_resample_i32_f64_inplace_device,
          py::arg("idx_in"),
          py::arg("val_in"),
          py::arg("idx_out"),
          py::arg("val_out"),
          py::arg("out_nnz"),
          py::arg("n_in"),
          py::arg("m"),
          py::arg("pivot"),
          py::arg("seed"),
          py::arg("threads") = 256,
          py::arg("stream") = 0,
          py::arg("sync") = true);

  py::class_<Kernel25Workspace>(m, "Kernel25Workspace")
      .def(py::init<int, int>(), py::arg("max_tasks"), py::arg("max_nnz_in"))
      .def("release", &Kernel25Workspace::release)
      .def_property_readonly("max_tasks", [](const Kernel25Workspace& self) { return self.max_tasks; })
      .def_property_readonly("max_nnz_in", [](const Kernel25Workspace& self) { return self.max_nnz_in; })
      .def(
          "build_from_jrs_allpairs_deterministic_inplace_device",
          &Kernel25Workspace::build_from_jrs_allpairs_deterministic_inplace_device,
          py::arg("drt"),
          py::arg("state"),
          py::arg("j_start"),
          py::arg("j_count"),
          py::arg("row_j"),
          py::arg("row_k"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("overflow"),
          py::arg("threads") = 128,
          py::arg("coalesce") = true,
          py::arg("stream") = 0,
          py::arg("sync") = true,
          py::arg("check_overflow") = true,
          py::arg("check_overflow_mode") = 1,
          py::arg("use_fused_count_write") = false,
          py::arg("profile") = py::none())
      .def(
          "build_from_tasks_deterministic_inplace_device",
          &Kernel25Workspace::build_from_tasks_deterministic_inplace_device,
          py::arg("drt"),
          py::arg("state"),
          py::arg("task_csf"),
          py::arg("task_p"),
          py::arg("task_q"),
          py::arg("row_j"),
          py::arg("row_k"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("overflow"),
          py::arg("threads") = 128,
          py::arg("coalesce") = true,
          py::arg("stream") = 0,
          py::arg("sync") = true,
          py::arg("check_overflow") = true);

  py::class_<Kernel3BuildGGemmExWorkspace>(m, "Kernel3BuildGWorkspace")
      .def(py::init<int, int>(), py::arg("nops"), py::arg("max_nrows"))
      .def_property_readonly("nops", [](const Kernel3BuildGGemmExWorkspace& self) { return self.nops; })
      .def_property_readonly("max_nrows", [](const Kernel3BuildGGemmExWorkspace& self) { return self.max_nrows; })
      .def("cublas_emulation_info", &Kernel3BuildGGemmExWorkspace::cublas_emulation_info)
      .def("set_cublas_math_mode", &Kernel3BuildGGemmExWorkspace::set_cublas_math_mode, py::arg("mode"))
      .def("set_cublas_workspace_bytes", &Kernel3BuildGGemmExWorkspace::set_cublas_workspace_bytes, py::arg("bytes"))
      .def("cublas_workspace_bytes", &Kernel3BuildGGemmExWorkspace::get_cublas_workspace_bytes)
      .def("gemm_backend", &Kernel3BuildGGemmExWorkspace::gemm_backend)
      .def("set_gemm_backend", &Kernel3BuildGGemmExWorkspace::set_gemm_backend, py::arg("backend"))
      .def("gemm_algo", &Kernel3BuildGGemmExWorkspace::gemm_algo_int)
      .def("set_gemm_algo", &Kernel3BuildGGemmExWorkspace::set_gemm_algo_int, py::arg("algo"))
      .def("set_cublas_emulation_strategy", &Kernel3BuildGGemmExWorkspace::set_cublas_emulation_strategy, py::arg("strategy"))
      .def(
          "set_cublas_emulation_special_values_support",
          &Kernel3BuildGGemmExWorkspace::set_cublas_emulation_special_values_support,
          py::arg("mask"))
      .def(
          "set_cublas_fixed_point_mantissa_control",
          &Kernel3BuildGGemmExWorkspace::set_cublas_fixed_point_mantissa_control,
          py::arg("control"))
      .def(
          "set_cublas_fixed_point_max_mantissa_bits",
          &Kernel3BuildGGemmExWorkspace::set_cublas_fixed_point_max_mantissa_bits,
          py::arg("max_bits"))
      .def(
          "set_cublas_fixed_point_mantissa_bit_offset",
          &Kernel3BuildGGemmExWorkspace::set_cublas_fixed_point_mantissa_bit_offset,
          py::arg("bit_offset"))
      .def(
          "gemm_dense",
          &Kernel3BuildGGemmExWorkspace::gemm_w_eri_mat_inplace_device,
          py::arg("eri_mat"),
          py::arg("w_dense"),
          py::arg("g_out"),
          py::arg("half") = 0.5,
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "build",
          &Kernel3BuildGGemmExWorkspace::build_g_from_csr_eri_mat_inplace_device,
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("eri_mat"),
          py::arg("g_out"),
          py::arg("threads") = 256,
          py::arg("half") = 0.5,
          py::arg("stream") = 0,
          py::arg("sync") = true);

  py::class_<Kernel3BuildGDFGemmExWorkspace>(m, "Kernel3BuildGDFWorkspace")
      .def(py::init<int, int, int>(), py::arg("nops"), py::arg("naux"), py::arg("max_nrows"))
      .def_property_readonly("nops", [](const Kernel3BuildGDFGemmExWorkspace& self) { return self.nops; })
      .def_property_readonly("naux", [](const Kernel3BuildGDFGemmExWorkspace& self) { return self.naux; })
      .def_property_readonly("max_nrows", [](const Kernel3BuildGDFGemmExWorkspace& self) { return self.max_nrows; })
      .def("cublas_emulation_info", &Kernel3BuildGDFGemmExWorkspace::cublas_emulation_info)
      .def("set_cublas_math_mode", &Kernel3BuildGDFGemmExWorkspace::set_cublas_math_mode, py::arg("mode"))
      .def("set_cublas_workspace_bytes", &Kernel3BuildGDFGemmExWorkspace::set_cublas_workspace_bytes, py::arg("bytes"))
      .def("cublas_workspace_bytes", &Kernel3BuildGDFGemmExWorkspace::get_cublas_workspace_bytes)
      .def("gemm_backend", &Kernel3BuildGDFGemmExWorkspace::gemm_backend)
      .def("set_gemm_backend", &Kernel3BuildGDFGemmExWorkspace::set_gemm_backend, py::arg("backend"))
      .def("gemm_algo", &Kernel3BuildGDFGemmExWorkspace::gemm_algo_int)
      .def("set_gemm_algo", &Kernel3BuildGDFGemmExWorkspace::set_gemm_algo_int, py::arg("algo"))
      .def(
          "set_cublas_emulation_strategy", &Kernel3BuildGDFGemmExWorkspace::set_cublas_emulation_strategy, py::arg("strategy"))
      .def(
          "set_cublas_emulation_special_values_support",
          &Kernel3BuildGDFGemmExWorkspace::set_cublas_emulation_special_values_support,
          py::arg("mask"))
      .def(
          "set_cublas_fixed_point_mantissa_control",
          &Kernel3BuildGDFGemmExWorkspace::set_cublas_fixed_point_mantissa_control,
          py::arg("control"))
      .def(
          "set_cublas_fixed_point_max_mantissa_bits",
          &Kernel3BuildGDFGemmExWorkspace::set_cublas_fixed_point_max_mantissa_bits,
          py::arg("max_bits"))
      .def(
          "set_cublas_fixed_point_mantissa_bit_offset",
          &Kernel3BuildGDFGemmExWorkspace::set_cublas_fixed_point_mantissa_bit_offset,
          py::arg("bit_offset"))
      .def(
          "gemm_dense",
          &Kernel3BuildGDFGemmExWorkspace::gemm_w_l_full_inplace_device,
          py::arg("l_full"),
          py::arg("w_dense"),
          py::arg("g_out"),
          py::arg("half") = 0.5,
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "build",
          &Kernel3BuildGDFGemmExWorkspace::build_g_from_csr_l_full_inplace_device,
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("l_full"),
          py::arg("g_out"),
          py::arg("threads") = 256,
          py::arg("half") = 0.5,
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "build_range",
          &Kernel3BuildGDFGemmExWorkspace::build_g_from_csr_l_full_range_inplace_device,
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("row_start"),
          py::arg("nrows"),
          py::arg("l_full"),
          py::arg("g_out"),
          py::arg("threads") = 256,
          py::arg("half") = 0.5,
          py::arg("stream") = 0,
          py::arg("sync") = true);

  py::class_<RDMGramGemmExWorkspace>(m, "RDMGramWorkspace")
      .def(py::init<int>(), py::arg("nops"))
      .def_property_readonly("nops", [](const RDMGramGemmExWorkspace& self) { return self.nops; })
      .def("cublas_emulation_info", &RDMGramGemmExWorkspace::cublas_emulation_info)
      .def("set_cublas_math_mode", &RDMGramGemmExWorkspace::set_cublas_math_mode, py::arg("mode"))
      .def("set_cublas_workspace_bytes", &RDMGramGemmExWorkspace::set_cublas_workspace_bytes, py::arg("bytes"))
      .def("cublas_workspace_bytes", &RDMGramGemmExWorkspace::get_cublas_workspace_bytes)
      .def("gemm_backend", &RDMGramGemmExWorkspace::gemm_backend)
      .def("set_gemm_backend", &RDMGramGemmExWorkspace::set_gemm_backend, py::arg("backend"))
      .def("gemm_algo", &RDMGramGemmExWorkspace::gemm_algo_int)
      .def("set_gemm_algo", &RDMGramGemmExWorkspace::set_gemm_algo_int, py::arg("algo"))
      .def("set_cublas_emulation_strategy", &RDMGramGemmExWorkspace::set_cublas_emulation_strategy, py::arg("strategy"))
      .def(
          "set_cublas_emulation_special_values_support",
          &RDMGramGemmExWorkspace::set_cublas_emulation_special_values_support,
          py::arg("mask"))
      .def(
          "set_cublas_fixed_point_mantissa_control",
          &RDMGramGemmExWorkspace::set_cublas_fixed_point_mantissa_control,
          py::arg("control"))
      .def(
          "set_cublas_fixed_point_max_mantissa_bits",
          &RDMGramGemmExWorkspace::set_cublas_fixed_point_max_mantissa_bits,
          py::arg("max_bits"))
      .def(
          "set_cublas_fixed_point_mantissa_bit_offset",
          &RDMGramGemmExWorkspace::set_cublas_fixed_point_mantissa_bit_offset,
          py::arg("bit_offset"))
      .def(
          "compute",
          &RDMGramGemmExWorkspace::gram_and_dm1_inplace_device,
          py::arg("t"),
          py::arg("c"),
          py::arg("dm1_out"),
          py::arg("gram_out"),
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "compute_csf_major",
          &RDMGramGemmExWorkspace::gram_and_dm1_csf_major_inplace_device,
          py::arg("t"),
          py::arg("c"),
          py::arg("dm1_out"),
          py::arg("gram_out"),
          py::arg("stream") = 0,
          py::arg("sync") = true,
          py::arg("accumulate") = false)
      .def(
          "compute_cross",
          &RDMGramGemmExWorkspace::gram_cross_and_dm1_inplace_device,
          py::arg("t_bra"),
          py::arg("t_ket"),
          py::arg("c_bra"),
          py::arg("dm1_out"),
          py::arg("gram_out"),
          py::arg("stream") = 0,
          py::arg("sync") = true)
      .def(
          "compute_cross_csf_major",
          &RDMGramGemmExWorkspace::gram_cross_and_dm1_csf_major_inplace_device,
          py::arg("t_bra"),
          py::arg("t_ket"),
          py::arg("c_bra"),
          py::arg("dm1_out"),
          py::arg("gram_out"),
          py::arg("stream") = 0,
          py::arg("sync") = true,
          py::arg("accumulate") = false);

  m.def("make_device_drt", &make_device_drt, py::arg("norb"), py::arg("child"), py::arg("node_twos"),
        py::arg("child_prefix"));

  m.def("make_device_state_cache", &make_device_state_cache, py::arg("drt"), py::arg("steps"), py::arg("nodes"));

  m.def(
      "epq_contribs_one_debug",
      &epq_contribs_one_debug,
      py::arg("drt"),
      py::arg("csf_idx"),
      py::arg("steps"),
      py::arg("nodes"),
      py::arg("p"),
      py::arg("q"),
      py::arg("max_out") = 100000);

  m.def(
      "epq_contribs_many_deterministic",
      &epq_contribs_many_deterministic,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("threads") = 128,
      py::arg("max_total_out") = (int64_t)-1);

  m.def(
      "epq_apply_weighted_many_atomic",
      &epq_apply_weighted_many_atomic,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("task_wgt"),
      py::arg("task_scale") = py::none(),
      py::arg("y0") = py::none(),
      py::arg("threads") = 128,
      py::arg("return_y") = true);

  m.def(
      "scatter_embed_inplace_device",
      &scatter_embed_inplace_device,
      py::arg("x_sub"),
      py::arg("sub_to_full"),
      py::arg("x_full"),
      py::arg("stream") = 0,
      py::arg("threads") = 128);

  m.def(
      "gather_project_inplace_device",
      &gather_project_inplace_device,
      py::arg("y_full"),
      py::arg("sub_to_full"),
      py::arg("y_sub"),
      py::arg("stream") = 0,
      py::arg("threads") = 128);

  m.def(
      "epq_apply_gather_inplace_device",
      &epq_apply_gather_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("p"),
      py::arg("q"),
      py::arg("x"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("alpha") = 1.0,
      py::arg("threads") = 256,
      py::arg("add") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "epq_contribs_many_count_tasks_inplace_device",
      &epq_contribs_many_count_tasks_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("counts"),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "epq_contribs_many_write_tasks_inplace_device",
      &epq_contribs_many_write_tasks_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("offsets"),
      py::arg("out_idx"),
      py::arg("out_coeff"),
      py::arg("out_task_csf") = py::none(),
      py::arg("out_task_pq") = py::none(),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "epq_contribs_many_count_allpairs_inplace_device",
      &epq_contribs_many_count_allpairs_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("counts"),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "epq_contribs_many_write_allpairs_inplace_device",
      &epq_contribs_many_write_allpairs_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("offsets"),
      py::arg("out_idx"),
      py::arg("out_coeff"),
      py::arg("out_task_csf") = py::none(),
      py::arg("out_task_pq") = py::none(),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "epq_contribs_many_count_allpairs_recompute_inplace_device",
      &epq_contribs_many_count_allpairs_recompute_inplace_device,
      py::arg("drt"),
      py::arg("ncsf"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("counts"),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("warp_coop") = false);

  m.def(
      "epq_contribs_many_write_allpairs_recompute_inplace_device",
      &epq_contribs_many_write_allpairs_recompute_inplace_device,
      py::arg("drt"),
      py::arg("ncsf"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("offsets"),
      py::arg("out_idx"),
      py::arg("out_coeff"),
      py::arg("out_task_csf") = py::none(),
      py::arg("out_task_pq") = py::none(),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("warp_coop") = false);

  m.def(
      "apply_g_flat_scatter_atomic",
      &apply_g_flat_scatter_atomic,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("y0") = py::none(),
      py::arg("threads") = 256,
      py::arg("return_y") = true);

  m.def(
      "apply_g_flat_scatter_atomic_inplace_device",
      &apply_g_flat_scatter_atomic_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_y") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  // 10.16.3 / 10.18: Warp-cooperative kernel binding (optimized for reduced local memory)
  m.def(
      "apply_g_flat_scatter_atomic_warp_coop_inplace_device",
      &apply_g_flat_scatter_atomic_warp_coop_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_y") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "apply_g_flat_scatter_atomic_epq_table_inplace_device",
      &apply_g_flat_scatter_atomic_epq_table_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("epq_data"),
      py::arg("task_csf"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_y") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "apply_g_flat_scatter_atomic_epq_table_tile_inplace_device",
      &apply_g_flat_scatter_atomic_epq_table_tile_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("local_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("epq_data"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_y") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("use_kahan") = false);

  m.def(
      "apply_g_flat_gather_epq_table_inplace_device",
      &apply_g_flat_gather_epq_table_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_t_indptr"),
      py::arg("epq_t_source"),
      py::arg("epq_t_pq"),
      py::arg("epq_t_data"),
      py::arg("task_row_by_csf"),
      py::arg("task_scale_by_csf") = py::none(),
      py::arg("task_g"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_y") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("use_kahan") = false);

  m.def(
      "apply_g_flat_gather_epq_transpose_range_inplace_device",
      &apply_g_flat_gather_epq_transpose_range_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_t_indptr"),
      py::arg("epq_t_source"),
      py::arg("epq_t_pq"),
      py::arg("epq_t_data"),
      py::arg("g_block"),
      py::arg("k_start"),
      py::arg("k_count"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("add") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("use_kahan") = false);

  m.def(
      "kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device",
      &kernel4_apply_csr_eri_mat_fused_epq_table_range_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("epq_data"),
      py::arg("row_j"),
      py::arg("row_k"),
      py::arg("csr_indptr"),
      py::arg("csr_indices"),
      py::arg("csr_data"),
      py::arg("row_start"),
      py::arg("nrows"),
      py::arg("eri_mat_t"),
      py::arg("x"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("threads") = 32,
      py::arg("zero_y") = true,
      py::arg("half") = 0.5,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("use_kahan") = false);

  m.def(
      "kernel4_build_w_from_csr_unitnnz_inplace_device",
      &kernel4_build_w_from_csr_unitnnz_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("row_j"),
      py::arg("row_k"),
      py::arg("csr_rs"),
      py::arg("csr_c"),
      py::arg("x"),
      py::arg("w_out"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "build_w_from_epq_table_inplace_device",
      &build_w_from_epq_table_inplace_device,
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("epq_data"),
      py::arg("x"),
      py::arg("w_out"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("k_start") = 0,
      py::arg("k_count") = 0);

  m.def(
      "build_w_from_epq_transpose_range_inplace_device",
      &build_w_from_epq_transpose_range_inplace_device,
      py::arg("state"),
      py::arg("epq_t_indptr"),
      py::arg("epq_t_source"),
      py::arg("epq_t_pq"),
      py::arg("epq_t_data"),
      py::arg("x"),
      py::arg("w_out"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true,
      py::arg("k_start") = 0,
      py::arg("k_count") = 0);

  m.def(
      "build_t_from_epq_table_inplace_device",
      &build_t_from_epq_table_inplace_device,
      py::arg("state"),
      py::arg("epq_indptr"),
      py::arg("epq_indices"),
      py::arg("epq_pq"),
      py::arg("epq_data"),
      py::arg("c_vec"),
      py::arg("t_out"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_out") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "build_w_diag_from_steps_inplace_device",
      &build_w_diag_from_steps_inplace_device,
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("x"),
      py::arg("w_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("relative_w") = false);

  m.def(
      "build_occ_block_from_steps_inplace_device",
      &build_occ_block_from_steps_inplace_device,
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("occ_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "build_hdiag_det_guess_from_steps_inplace_device",
      &build_hdiag_det_guess_from_steps_inplace_device,
      py::arg("state"),
      py::arg("neleca_det"),
      py::arg("h1e_diag"),
      py::arg("eri_ppqq"),
      py::arg("eri_pqqp"),
      py::arg("hdiag_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "apply_g_flat_task_sums_inplace_device",
      &apply_g_flat_task_sums_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_g"),
      py::arg("task_scale") = py::none(),
      py::arg("out_sum"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_out") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "build_t_block_epq_atomic_inplace_device",
      &build_t_block_epq_atomic_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("c"),
      py::arg("p_list"),
      py::arg("q_list"),
      py::arg("out"),
      py::arg("overflow"),
      py::arg("threads") = 256,
      py::arg("zero_out") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "kernel25_build_csr_from_triples_cuda",
      &kernel25_build_csr_from_triples_cuda,
      py::arg("j_out"),
      py::arg("k_out"),
      py::arg("rs_out"),
      py::arg("c_out"),
      py::arg("nops"),
      py::arg("coalesce") = true);

  m.def(
      "kernel25_build_csr_from_tasks_deterministic_cuda",
      &kernel25_build_csr_from_tasks_deterministic_cuda,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("threads") = 128,
      py::arg("max_total_out") = (int64_t)-1,
      py::arg("coalesce") = true);

  m.def(
      "kernel25_build_csr_from_tasks_deterministic_inplace_device",
      &kernel25_build_csr_from_tasks_deterministic_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("task_csf"),
      py::arg("task_p"),
      py::arg("task_q"),
      py::arg("row_j"),
      py::arg("row_k"),
      py::arg("indptr"),
      py::arg("indices"),
      py::arg("data"),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("coalesce") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device",
      &kernel25_build_csr_from_jrs_allpairs_deterministic_inplace_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("row_j"),
      py::arg("row_k"),
      py::arg("indptr"),
      py::arg("indices"),
      py::arg("data"),
      py::arg("overflow"),
      py::arg("threads") = 128,
      py::arg("coalesce") = true,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "kernel3_build_g_from_csr_eri_mat_cuda",
      &kernel3_build_g_from_csr_eri_mat_cuda,
      py::arg("indptr"),
      py::arg("indices"),
      py::arg("data"),
      py::arg("eri_mat"),
      py::arg("row_start") = 0,
      py::arg("nrows") = -1,
      py::arg("half") = 0.5,
      py::arg("threads") = 256);

  m.def(
      "kernel3_build_g_from_csr_eri_mat_inplace_device",
      &kernel3_build_g_from_csr_eri_mat_inplace_device,
      py::arg("indptr"),
      py::arg("indices"),
      py::arg("data"),
      py::arg("eri_mat"),
      py::arg("g_out"),
      py::arg("threads") = 256,
      py::arg("half") = 0.5,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "kernel3_build_g_from_csr_eri_mat_range_inplace_device",
      &kernel3_build_g_from_csr_eri_mat_range_inplace_device,
      py::arg("indptr"),
      py::arg("indices"),
      py::arg("data"),
      py::arg("row_start"),
      py::arg("nrows"),
      py::arg("eri_mat"),
      py::arg("g_out"),
      py::arg("threads") = 256,
      py::arg("half") = 0.5,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "epq_apply_g_debug",
      &epq_apply_g_debug,
      py::arg("drt"),
      py::arg("csf_idx"),
      py::arg("g_flat"),
      py::arg("steps"),
      py::arg("nodes"),
      py::arg("thresh_gpq") = 0.0,
      py::arg("thresh_contrib") = 0.0,
      py::arg("max_out") = 200000);

  m.def(
      "fused_hop_device",
      &fused_hop_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("x"),
      py::arg("eri_mat"),
      py::arg("h_eff_flat"),
      py::arg("y"),
      py::arg("overflow"),
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "fused_hop_phase1_device",
      &fused_hop_phase1_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("x"),
      py::arg("eri_mat"),
      py::arg("h_eff_flat"),
      py::arg("y"),
      py::arg("g_out"),
      py::arg("overflow"),
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "fused_hop_phase1_coo_device",
      &fused_hop_phase1_coo_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("x"),
      py::arg("eri_mat"),
      py::arg("h_eff_flat"),
      py::arg("y"),
      py::arg("g_out"),
      py::arg("overflow"),
      py::arg("coo_nnz_counter"),
      py::arg("coo_j_local"),
      py::arg("coo_k"),
      py::arg("coo_pq"),
      py::arg("coo_w2"),
      py::arg("max_coo"),
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "fused_hop_phase1_coo_merged_device",
      &fused_hop_phase1_coo_merged_device,
      py::arg("drt"),
      py::arg("state"),
      py::arg("j_start"),
      py::arg("j_count"),
      py::arg("x"),
      py::arg("eri_mat"),
      py::arg("h_eff_flat"),
      py::arg("y"),
      py::arg("g_out"),
      py::arg("overflow"),
      py::arg("coo_nnz_counter"),
      py::arg("coo_j_local"),
      py::arg("coo_k"),
      py::arg("coo_pq"),
      py::arg("coo_w2"),
      py::arg("max_coo"),
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("check_overflow") = true);

  m.def(
      "coo_scatter_device",
      &coo_scatter_device,
      py::arg("coo_j_local"),
      py::arg("coo_k"),
      py::arg("coo_pq"),
      py::arg("coo_w2"),
      py::arg("g_tile"),
      py::arg("nops"),
      py::arg("nnz"),
      py::arg("y"),
      py::arg("stream") = 0,
      py::arg("sync") = true);
}
