#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" void hf_df_jk_fill_lower_from_upper_f64(double* a, int n, cudaStream_t stream);
extern "C" void hf_df_jk_pack_bmnq_to_bq_f64(
    const double* b_mnq, int nao, int naux, int q0, int q, double* out_bq, cudaStream_t stream);

namespace {

struct CudaArrayView {
  void* ptr = nullptr;
  bool read_only = false;
  std::string typestr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;  // empty means "not provided"/contiguous
};

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void throw_on_cublas_error(cublasStatus_t stat, const char* what) {
  if (stat == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string((int)stat));
}

struct ScopedCublasMathMode {
  cublasHandle_t handle = nullptr;
  cublasMath_t old_mode{};
  bool active = false;
  explicit ScopedCublasMathMode(cublasHandle_t h, int math_mode) : handle(h) {
    if (!handle) return;
    if (math_mode < 0) return;
    throw_on_cublas_error(cublasGetMathMode(handle, &old_mode), "cublasGetMathMode");
    auto new_mode = static_cast<cublasMath_t>(math_mode);
    if (new_mode == old_mode) return;
    throw_on_cublas_error(cublasSetMathMode(handle, new_mode), "cublasSetMathMode");
    active = true;
  }
  ~ScopedCublasMathMode() {
    if (!active || !handle) return;
    cublasSetMathMode(handle, old_mode);
  }
};

inline void validate_cuda_device_pointer(const void* ptr, const char* what) {
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
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

inline std::string normalize_typestr(std::string t) {
  if (t.size() == 3 && (t[0] == '=' || t[0] == '|')) t[0] = '<';
  return t;
}

inline void require_typestr_f64(const CudaArrayView& a, const char* name) {
  const std::string t = normalize_typestr(a.typestr);
  if (t == "<f8") return;
  throw std::invalid_argument(std::string(name) + " must have typestr <f8 (got " + a.typestr + ")");
}

inline void require_c_contiguous(const CudaArrayView& a, const char* name, int64_t itemsize) {
  if (a.strides_bytes.empty()) return;
  if ((int64_t)a.strides_bytes.size() != (int64_t)a.shape.size()) {
    throw std::invalid_argument(std::string(name) + " has invalid strides rank");
  }
  int64_t expected = itemsize;
  for (int64_t d = (int64_t)a.shape.size() - 1; d >= 0; --d) {
    const int64_t stride = a.strides_bytes[(size_t)d];
    if (stride != expected) {
      throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
    const int64_t dim = a.shape[(size_t)d];
    expected = (dim == 0) ? expected : expected * dim;
  }
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
  bool is_empty = false;
  for (py::handle dim : shape) {
    int64_t d = py::cast<int64_t>(dim);
    if (d < 0) throw std::invalid_argument(std::string(name) + " has negative dimension in shape");
    if (d == 0) is_empty = true;
    out.shape.push_back(d);
  }

  if (cai.contains("strides")) {
    py::object strides_obj = cai["strides"];
    if (!strides_obj.is_none()) {
      py::tuple strides = strides_obj.cast<py::tuple>();
      out.strides_bytes.reserve((size_t)strides.size());
      for (py::handle s : strides) {
        int64_t sb = py::cast<int64_t>(s);
        if (sb < 0) throw std::invalid_argument(std::string(name) + " must not have negative strides");
        out.strides_bytes.push_back(sb);
      }
    }
  }

  if (out.ptr == nullptr) {
    if (!is_empty) {
      throw std::invalid_argument(std::string(name) + " has null device pointer");
    }
    return out;
  }
  validate_cuda_device_pointer(out.ptr, name);
  return out;
}

struct CublasHandle {
  cublasHandle_t h = nullptr;
  CublasHandle() { throw_on_cublas_error(cublasCreate(&h), "cublasCreate"); }
  ~CublasHandle() {
    if (h) cublasDestroy(h);
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
};

class DFJKWorkspace {
 public:
  DFJKWorkspace() = default;
  ~DFJKWorkspace() { release(); }

  void release() {
    if (v_buf_) {
      cudaFree(v_buf_);
      v_buf_ = nullptr;
    }
    v_elems_ = 0;
    if (bq_buf_) {
      cudaFree(bq_buf_);
      bq_buf_ = nullptr;
    }
    bq_elems_ = 0;
  }

  void k_from_bq_cw(
      const py::object& bq,
      const py::object& cw,
      const py::object& out_k,
      int q_block,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView bq_v = cuda_array_view_from_object(bq, "BQ");
    const CudaArrayView cw_v = cuda_array_view_from_object(cw, "Cw");
    const CudaArrayView k_v = cuda_array_view_from_object(out_k, "out");

    require_typestr_f64(bq_v, "BQ");
    require_typestr_f64(cw_v, "Cw");
    require_typestr_f64(k_v, "out");

    if (k_v.read_only) {
      throw std::invalid_argument("out must be a writable CUDA array");
    }
    if (bq_v.shape.size() != 3) {
      throw std::invalid_argument("BQ must have shape (naux, nao, nao)");
    }
    if (cw_v.shape.size() != 2) {
      throw std::invalid_argument("Cw must have shape (nao, nocc)");
    }
    if (k_v.shape.size() != 2) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }

    const int64_t naux = bq_v.shape[0];
    const int64_t nao0 = bq_v.shape[1];
    const int64_t nao1 = bq_v.shape[2];
    if (nao0 != nao1) {
      throw std::invalid_argument("BQ must have shape (naux, nao, nao)");
    }
    const int64_t nao = nao0;
    if (cw_v.shape[0] != nao) {
      throw std::invalid_argument("Cw nao mismatch with BQ");
    }
    const int64_t nocc = cw_v.shape[1];
    if (k_v.shape[0] != nao || k_v.shape[1] != nao) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }
    if (q_block <= 0) {
      throw std::invalid_argument("q_block must be > 0");
    }

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(bq_v, "BQ", itemsize);
    require_c_contiguous(cw_v, "Cw", itemsize);
    require_c_contiguous(k_v, "out", itemsize);

    auto* bq_ptr = reinterpret_cast<const double*>(bq_v.ptr);
    auto* cw_ptr = reinterpret_cast<const double*>(cw_v.ptr);
    auto* k_ptr = reinterpret_cast<double*>(k_v.ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    throw_on_cublas_error(cublasSetStream(cublas_.h, stream), "cublasSetStream");
    const ScopedCublasMathMode math_ctx(cublas_.h, math_mode);

    if (naux > (int64_t)std::numeric_limits<int>::max() || nao > (int64_t)std::numeric_limits<int>::max() ||
        nocc > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("dimensions exceed int32 limits for cuBLAS");
    }
    const int naux_i = (int)naux;
    const int nao_i = (int)nao;
    const int nocc_i = (int)nocc;
    if (naux_i < 0 || nao_i <= 0 || nocc_i < 0) throw std::invalid_argument("invalid dimensions");

    // Empty cases: always zero out out_k for predictable semantics.
    const int64_t k_elems = nao * nao;
    throw_on_cuda_error(cudaMemsetAsync(k_ptr, 0, (size_t)k_elems * sizeof(double), stream), "cudaMemsetAsync(out)");
    if (naux_i == 0 || nocc_i == 0) {
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      return;
    }

    const int q_block_i = q_block;
    const int64_t max_cols = (int64_t)q_block_i * (int64_t)nocc_i;
    const int64_t need_elems = max_cols * nao;
    ensure_v_capacity((size_t)need_elems);

    const double alpha = 1.0;
    const double beta0 = 0.0;
    const double beta = 1.0;

    // Build U directly in a layout that can be reinterpreted as column-major
    // (nao x (q*nocc)) without an explicit transpose.
    for (int q0 = 0; q0 < naux_i; q0 += q_block_i) {
      const int q = std::min(q_block_i, naux_i - q0);
      const int64_t cols = (int64_t)q * (int64_t)nocc_i;
      if (cols <= 0) continue;

      // C (q*nao x nocc, col-major) = BQ_block^T * Cw^T
      // with row-major inputs interpreted through transposed column-major views.
      // The resulting C buffer can be reinterpreted as col-major (nao x (q*nocc)).
      const int n_flat = q * nao_i;
      const double* B = bq_ptr + (int64_t)q0 * nao * nao;
      double* U = v_buf_;

      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_T,
              CUBLAS_OP_T,
              /*m=*/n_flat,
              /*n=*/nocc_i,
              /*k=*/nao_i,
              &alpha,
              B,
              /*lda=*/nao_i,
              cw_ptr,
              /*ldb=*/nocc_i,
              &beta0,
              U,
              /*ldc=*/n_flat),
          "gemm_f64(U)");

      // GEMM: accumulate full K = U @ U^T (then we enforce exact symmetry via a small kernel).
      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_N,
              CUBLAS_OP_T,
              /*m=*/nao_i,
              /*n=*/nao_i,
              /*k=*/(int)cols,
              &alpha,
              U,
              /*lda=*/nao_i,
              U,
              /*ldb=*/nao_i,
              &beta,
              k_ptr,
              /*ldc=*/nao_i),
          "gemm_f64(K)");
    }

    // Fill the remaining triangle in the Python row-major view so out_k is fully symmetric.
    hf_df_jk_fill_lower_from_upper_f64(k_ptr, nao_i, stream);

    if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  void k_from_bmnq_cw(
      const py::object& b_mnq,
      const py::object& cw,
      const py::object& out_k,
      int q_block,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView b_v = cuda_array_view_from_object(b_mnq, "B_mnQ");
    const CudaArrayView cw_v = cuda_array_view_from_object(cw, "Cw");
    const CudaArrayView k_v = cuda_array_view_from_object(out_k, "out");

    require_typestr_f64(b_v, "B_mnQ");
    require_typestr_f64(cw_v, "Cw");
    require_typestr_f64(k_v, "out");

    if (k_v.read_only) {
      throw std::invalid_argument("out must be a writable CUDA array");
    }
    if (b_v.shape.size() != 3) {
      throw std::invalid_argument("B_mnQ must have shape (nao, nao, naux)");
    }
    if (cw_v.shape.size() != 2) {
      throw std::invalid_argument("Cw must have shape (nao, nocc)");
    }
    if (k_v.shape.size() != 2) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }

    const int64_t nao0 = b_v.shape[0];
    const int64_t nao1 = b_v.shape[1];
    const int64_t naux = b_v.shape[2];
    if (nao0 != nao1) {
      throw std::invalid_argument("B_mnQ must have shape (nao, nao, naux)");
    }
    const int64_t nao = nao0;
    if (cw_v.shape[0] != nao) {
      throw std::invalid_argument("Cw nao mismatch with B_mnQ");
    }
    const int64_t nocc = cw_v.shape[1];
    if (k_v.shape[0] != nao || k_v.shape[1] != nao) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }
    if (q_block <= 0) {
      throw std::invalid_argument("q_block must be > 0");
    }

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(b_v, "B_mnQ", itemsize);
    require_c_contiguous(cw_v, "Cw", itemsize);
    require_c_contiguous(k_v, "out", itemsize);

    auto* b_ptr = reinterpret_cast<const double*>(b_v.ptr);
    auto* cw_ptr = reinterpret_cast<const double*>(cw_v.ptr);
    auto* k_ptr = reinterpret_cast<double*>(k_v.ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    throw_on_cublas_error(cublasSetStream(cublas_.h, stream), "cublasSetStream");
    const ScopedCublasMathMode math_ctx(cublas_.h, math_mode);

    if (naux > (int64_t)std::numeric_limits<int>::max() || nao > (int64_t)std::numeric_limits<int>::max() ||
        nocc > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("dimensions exceed int32 limits for cuBLAS");
    }
    const int naux_i = (int)naux;
    const int nao_i = (int)nao;
    const int nocc_i = (int)nocc;
    if (naux_i < 0 || nao_i <= 0 || nocc_i < 0) throw std::invalid_argument("invalid dimensions");

    const int64_t k_elems = nao * nao;
    throw_on_cuda_error(cudaMemsetAsync(k_ptr, 0, (size_t)k_elems * sizeof(double), stream), "cudaMemsetAsync(out)");
    if (naux_i == 0 || nocc_i == 0) {
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      return;
    }

    const int q_block_i = q_block;
    const int64_t max_rows = (int64_t)q_block_i * (int64_t)nocc_i;
    ensure_v_capacity((size_t)(max_rows * nao));
    ensure_bq_capacity((size_t)((int64_t)q_block_i * nao * nao));

    const double alpha = 1.0;
    const double beta0 = 0.0;
    const double beta = 1.0;

    for (int q0 = 0; q0 < naux_i; q0 += q_block_i) {
      const int q = std::min(q_block_i, naux_i - q0);
      const int64_t cols = (int64_t)q * (int64_t)nocc_i;
      if (cols <= 0) continue;

      // Pack a contiguous BQ block (q, nao, nao) from mnQ layout into bq_buf_.
      hf_df_jk_pack_bmnq_to_bq_f64(b_ptr, nao_i, naux_i, q0, q, bq_buf_, stream);

      // Build U directly in a layout that can be reinterpreted as
      // col-major (nao x (q*nocc)) without a transpose kernel.
      const int n_flat = q * nao_i;
      const double* B = bq_buf_;
      double* U = v_buf_;

      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_T,
              CUBLAS_OP_T,
              /*m=*/n_flat,
              /*n=*/nocc_i,
              /*k=*/nao_i,
              &alpha,
              B,
              /*lda=*/nao_i,
              cw_ptr,
              /*ldb=*/nocc_i,
              &beta0,
              U,
              /*ldc=*/n_flat),
          "gemm_f64(U)");

      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_N,
              CUBLAS_OP_T,
              /*m=*/nao_i,
              /*n=*/nao_i,
              /*k=*/(int)cols,
              &alpha,
              U,
              /*lda=*/nao_i,
              U,
              /*ldb=*/nao_i,
              &beta,
              k_ptr,
              /*ldc=*/nao_i),
          "gemm_f64(K)");
    }

    hf_df_jk_fill_lower_from_upper_f64(k_ptr, nao_i, stream);

    if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

 private:
  cublasStatus_t gemm_f64(
      cublasOperation_t transa,
      cublasOperation_t transb,
      int m,
      int n,
      int k,
      const double* alpha,
      const double* A,
      int lda,
      const double* B,
      int ldb,
      const double* beta,
      double* C,
      int ldc) {
    return cublasDgemm(cublas_.h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  void ensure_v_capacity(size_t need_elems) {
    if (need_elems <= v_elems_) return;
    if (v_buf_) {
      cudaFree(v_buf_);
      v_buf_ = nullptr;
    }
    v_elems_ = 0;
    double* ptr = nullptr;
    throw_on_cuda_error(cudaMalloc(&ptr, need_elems * sizeof(double)), "cudaMalloc(V)");
    v_buf_ = ptr;
    v_elems_ = need_elems;
  }

  void ensure_bq_capacity(size_t need_elems) {
    if (need_elems <= bq_elems_) return;
    if (bq_buf_) {
      cudaFree(bq_buf_);
      bq_buf_ = nullptr;
    }
    bq_elems_ = 0;
    double* ptr = nullptr;
    throw_on_cuda_error(cudaMalloc(&ptr, need_elems * sizeof(double)), "cudaMalloc(BQ)");
    bq_buf_ = ptr;
    bq_elems_ = need_elems;
  }

  CublasHandle cublas_;
  double* v_buf_ = nullptr;
  size_t v_elems_ = 0;
  double* bq_buf_ = nullptr;
  size_t bq_elems_ = 0;
};

}  // namespace

PYBIND11_MODULE(_hf_df_jk_cuda_ext, m) {
  m.doc() = "CUDA/cuBLAS kernels for HF DF J/K contractions.";

  py::class_<DFJKWorkspace> cls(m, "DFJKWorkspace");
  cls.def(py::init<>()).def("release", &DFJKWorkspace::release);

  cls.def(
      "k_from_bq_cw",
      [](DFJKWorkspace& self,
         const py::object& bq,
         const py::object& cw,
         const py::object& out_k,
         int q_block,
         uint64_t stream,
         int math_mode,
         bool sync) {
        self.k_from_bq_cw(bq, cw, out_k, q_block, stream, math_mode, sync);
      },
      py::arg("BQ"),
      py::arg("Cw"),
      py::arg("out"),
      py::arg("q_block") = 128,
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false);

  cls.def(
      "k_from_bmnq_cw",
      [](DFJKWorkspace& self,
         const py::object& b_mnq,
         const py::object& cw,
         const py::object& out_k,
         int q_block,
         uint64_t stream,
         int math_mode,
         bool sync) {
        self.k_from_bmnq_cw(b_mnq, cw, out_k, q_block, stream, math_mode, sync);
      },
      py::arg("B_mnQ"),
      py::arg("Cw"),
      py::arg("out"),
      py::arg("q_block") = 128,
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false);
}
