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
extern "C" void hf_df_jk_symmetrize_inplace_f64(double* a, int n, cudaStream_t stream);
extern "C" void hf_df_jk_pack_bmnq_to_bq_f64(
    const double* b_mnq, int nao, int naux, int q0, int q, double* out_bq, cudaStream_t stream);
extern "C" void hf_df_jk_unpack_qp_to_bq_f64(
    const double* b_qp, int nao, int naux, int q0, int q, double* out_bq, cudaStream_t stream);
extern "C" void hf_df_jk_extract_qp_rows_fullcols_f64(
    const double* b_qp,
    int nao,
    int naux,
    int q0,
    int q,
    int row0,
    int row_count,
    double* out_rows,
    cudaStream_t stream);
extern "C" void hf_df_jk_repack_y2d_to_yflat_f64(
    const double* y2d, int q, int row_count, int nao, double* out_yflat, cudaStream_t stream);
extern "C" void hf_df_jk_extract_qp_cols_to_zflat_t_f64(
    const double* b_qp,
    int nao,
    int naux,
    int q0,
    int q,
    int col0,
    int col_count,
    double* out_zflat_t,
    cudaStream_t stream);

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
    if (a_buf_) {
      cudaFree(a_buf_);
      a_buf_ = nullptr;
    }
    a_elems_ = 0;
    if (y_buf_) {
      cudaFree(y_buf_);
      y_buf_ = nullptr;
    }
    y_elems_ = 0;
    if (z_buf_) {
      cudaFree(z_buf_);
      z_buf_ = nullptr;
    }
    z_elems_ = 0;
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

  void density_syrk(
      const py::object& cw,
      const py::object& out,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView cw_v = cuda_array_view_from_object(cw, "Cw");
    const CudaArrayView out_v = cuda_array_view_from_object(out, "out");

    require_typestr_f64(cw_v, "Cw");
    require_typestr_f64(out_v, "out");

    if (out_v.read_only) throw std::invalid_argument("out must be a writable CUDA array");
    if (cw_v.shape.size() != 2) throw std::invalid_argument("Cw must have shape (nao, nocc)");
    if (out_v.shape.size() != 2) throw std::invalid_argument("out must have shape (nao, nao)");

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(cw_v, "Cw", itemsize);
    require_c_contiguous(out_v, "out", itemsize);

    const int64_t nao = cw_v.shape[0];
    const int64_t nocc = cw_v.shape[1];
    if (out_v.shape[0] != nao || out_v.shape[1] != nao)
      throw std::invalid_argument("out shape mismatch with Cw nao");
    if (nao > (int64_t)std::numeric_limits<int>::max() ||
        nocc > (int64_t)std::numeric_limits<int>::max())
      throw std::invalid_argument("dimensions exceed int32 limits for cuBLAS");

    const int nao_i = (int)nao;
    const int nocc_i = (int)nocc;

    auto* cw_ptr = reinterpret_cast<const double*>(cw_v.ptr);
    auto* out_ptr = reinterpret_cast<double*>(out_v.ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    throw_on_cublas_error(cublasSetStream(cublas_.h, stream), "cublasSetStream");
    const ScopedCublasMathMode math_ctx(cublas_.h, math_mode);

    throw_on_cuda_error(
        cudaMemsetAsync(out_ptr, 0, (size_t)nao_i * nao_i * sizeof(double), stream),
        "cudaMemsetAsync(out)");

    if (nao_i <= 0 || nocc_i <= 0) {
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      return;
    }

    // D = Cw @ Cw^T via DSYRK.
    // Cw is row-major (nao, nocc). In cuBLAS col-major view, it is (nocc, nao) with lda=nocc.
    // CUBLAS_FILL_MODE_LOWER in col-major = upper triangle in row-major (Python/C convention).
    // CUBLAS_OP_T: C = alpha * A^T * A where A is (nocc x nao) col-major => C = Cw_rm * Cw_rm^T.
    const double alpha = 1.0;
    const double beta_zero = 0.0;
    throw_on_cublas_error(
        cublasDsyrk(cublas_.h,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,
                    nao_i,
                    nocc_i,
                    &alpha,
                    cw_ptr,
                    nocc_i,
                    &beta_zero,
                    out_ptr,
                    nao_i),
        "cublasDsyrk");

    // Fill lower triangle from upper (row-major convention).
    hf_df_jk_fill_lower_from_upper_f64(out_ptr, nao_i, stream);

    if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  void jk_from_bq_cw(
      const py::object& bq,
      const py::object& cw,
      const py::object& d_mat,
      const py::object& out_j,
      const py::object& out_k,
      int q_block,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView bq_v = cuda_array_view_from_object(bq, "BQ");
    const CudaArrayView cw_v = cuda_array_view_from_object(cw, "Cw");
    const CudaArrayView d_v = cuda_array_view_from_object(d_mat, "D");
    const CudaArrayView j_v = cuda_array_view_from_object(out_j, "out_J");
    const CudaArrayView k_v = cuda_array_view_from_object(out_k, "out_K");

    require_typestr_f64(bq_v, "BQ");
    require_typestr_f64(cw_v, "Cw");
    require_typestr_f64(d_v, "D");
    require_typestr_f64(j_v, "out_J");
    require_typestr_f64(k_v, "out_K");

    if (j_v.read_only) throw std::invalid_argument("out_J must be a writable CUDA array");
    if (k_v.read_only) throw std::invalid_argument("out_K must be a writable CUDA array");
    if (bq_v.shape.size() != 3) throw std::invalid_argument("BQ must have shape (naux, nao, nao)");
    if (cw_v.shape.size() != 2) throw std::invalid_argument("Cw must have shape (nao, nocc)");
    if (d_v.shape.size() != 2) throw std::invalid_argument("D must have shape (nao, nao)");
    if (j_v.shape.size() != 2) throw std::invalid_argument("out_J must have shape (nao, nao)");
    if (k_v.shape.size() != 2) throw std::invalid_argument("out_K must have shape (nao, nao)");

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(bq_v, "BQ", itemsize);
    require_c_contiguous(cw_v, "Cw", itemsize);
    require_c_contiguous(d_v, "D", itemsize);
    require_c_contiguous(j_v, "out_J", itemsize);
    require_c_contiguous(k_v, "out_K", itemsize);

    const int64_t naux = bq_v.shape[0];
    const int64_t nao0 = bq_v.shape[1];
    const int64_t nao1 = bq_v.shape[2];
    if (nao0 != nao1) throw std::invalid_argument("BQ must have shape (naux, nao, nao)");
    const int64_t nao = nao0;

    if (cw_v.shape[0] != nao) throw std::invalid_argument("Cw nao mismatch with BQ");
    const int64_t nocc = cw_v.shape[1];
    if (d_v.shape[0] != nao || d_v.shape[1] != nao)
      throw std::invalid_argument("D shape mismatch with BQ nao");
    if (j_v.shape[0] != nao || j_v.shape[1] != nao)
      throw std::invalid_argument("out_J shape mismatch with BQ nao");
    if (k_v.shape[0] != nao || k_v.shape[1] != nao)
      throw std::invalid_argument("out_K shape mismatch with BQ nao");
    if (q_block <= 0) throw std::invalid_argument("q_block must be > 0");

    if (naux > (int64_t)std::numeric_limits<int>::max() ||
        nao > (int64_t)std::numeric_limits<int>::max() ||
        nocc > (int64_t)std::numeric_limits<int>::max())
      throw std::invalid_argument("dimensions exceed int32 limits for cuBLAS");

    const int naux_i = (int)naux;
    const int nao_i = (int)nao;
    const int nocc_i = (int)nocc;
    if (naux_i < 0 || nao_i <= 0 || nocc_i < 0)
      throw std::invalid_argument("invalid dimensions");

    auto* bq_ptr = reinterpret_cast<const double*>(bq_v.ptr);
    auto* cw_ptr = reinterpret_cast<const double*>(cw_v.ptr);
    auto* d_ptr = reinterpret_cast<const double*>(d_v.ptr);
    auto* j_ptr = reinterpret_cast<double*>(j_v.ptr);
    auto* k_ptr = reinterpret_cast<double*>(k_v.ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    throw_on_cublas_error(cublasSetStream(cublas_.h, stream), "cublasSetStream");
    const ScopedCublasMathMode math_ctx(cublas_.h, math_mode);

    const int64_t nn = (int64_t)nao_i * nao_i;
    throw_on_cuda_error(
        cudaMemsetAsync(j_ptr, 0, (size_t)nn * sizeof(double), stream), "cudaMemsetAsync(J)");
    throw_on_cuda_error(
        cudaMemsetAsync(k_ptr, 0, (size_t)nn * sizeof(double), stream), "cudaMemsetAsync(K)");

    if (naux_i == 0 || nocc_i == 0) {
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      return;
    }

    const int q_block_i = std::min(q_block, naux_i);
    const int nao2 = nao_i * nao_i;
    // v_buf_: K's U matrix (q*nao, nocc).
    // bq_buf_: d_q vector for J (q doubles; much smaller than q*nao*nao).
    const int64_t max_cols_k = (int64_t)q_block_i * nocc_i;
    ensure_v_capacity((size_t)(max_cols_k * nao_i));
    ensure_bq_capacity((size_t)q_block_i);

    const double alpha = 1.0;
    const double beta0 = 0.0;
    const double beta1 = 1.0;

    for (int q0 = 0; q0 < naux_i; q0 += q_block_i) {
      const int q = std::min(q_block_i, naux_i - q0);
      if (q <= 0) continue;

      const double* BQc = bq_ptr + (int64_t)q0 * nao_i * nao_i;
      double* d_q = bq_buf_;  // (q,) scratch for J
      double* U = v_buf_;     // (q*nao, nocc) scratch for K

      // --- J contribution ---
      // d_q = BQc.reshape(q, nao²) @ d_flat
      // In cuBLAS col-major: BQc is (nao², q), CUBLAS_OP_T: d_q = BQc^T @ d_flat.
      throw_on_cublas_error(
          cublasDgemv(cublas_.h, CUBLAS_OP_T,
                      nao2, q,
                      &alpha, BQc, nao2,
                      d_ptr, 1,
                      &beta0, d_q, 1),
          "cublasDgemv(J_dQ)");

      // J_flat += BQc.reshape(nao², q) @ d_q (accumulate all contributions)
      throw_on_cublas_error(
          cublasDgemv(cublas_.h, CUBLAS_OP_N,
                      nao2, q,
                      &alpha, BQc, nao2,
                      d_q, 1,
                      &beta1, j_ptr, 1),
          "cublasDgemv(J_accum)");

      // --- K contribution (same as k_from_bq_cw) ---
      const int n_flat = q * nao_i;
      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_T, CUBLAS_OP_T,
              n_flat, nocc_i, nao_i,
              &alpha, BQc, nao_i,
              cw_ptr, nocc_i,
              &beta0, U, n_flat),
          "gemm_f64(K_U)");

      const int64_t cols = (int64_t)q * nocc_i;
      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_N, CUBLAS_OP_T,
              nao_i, nao_i, (int)cols,
              &alpha, U, nao_i,
              U, nao_i,
              &beta1, k_ptr, nao_i),
          "gemm_f64(K_accum)");
    }

    // Enforce exact symmetry for both outputs.
    hf_df_jk_fill_lower_from_upper_f64(j_ptr, nao_i, stream);
    hf_df_jk_fill_lower_from_upper_f64(k_ptr, nao_i, stream);

    if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  void k_from_qp_cw(
      const py::object& b_qp,
      const py::object& cw,
      const py::object& out_k,
      int q_block,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView bqp_v = cuda_array_view_from_object(b_qp, "B_Qp");
    const CudaArrayView cw_v = cuda_array_view_from_object(cw, "Cw");
    const CudaArrayView k_v = cuda_array_view_from_object(out_k, "out");

    require_typestr_f64(bqp_v, "B_Qp");
    require_typestr_f64(cw_v, "Cw");
    require_typestr_f64(k_v, "out");

    if (k_v.read_only) {
      throw std::invalid_argument("out must be a writable CUDA array");
    }
    if (bqp_v.shape.size() != 2) {
      throw std::invalid_argument("B_Qp must have shape (naux, ntri)");
    }
    if (cw_v.shape.size() != 2) {
      throw std::invalid_argument("Cw must have shape (nao, nocc)");
    }
    if (k_v.shape.size() != 2) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }

    const int64_t naux = bqp_v.shape[0];
    const int64_t ntri = bqp_v.shape[1];
    const int64_t nao = cw_v.shape[0];
    const int64_t nocc = cw_v.shape[1];
    if (k_v.shape[0] != nao || k_v.shape[1] != nao) {
      throw std::invalid_argument("out must have shape (nao, nao)");
    }

    const int64_t expected_ntri = (nao * (nao + 1)) / 2;
    if (ntri != expected_ntri) {
      throw std::invalid_argument("B_Qp must have ntri=nao*(nao+1)//2");
    }
    if (q_block <= 0) {
      throw std::invalid_argument("q_block must be > 0");
    }

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(bqp_v, "B_Qp", itemsize);
    require_c_contiguous(cw_v, "Cw", itemsize);
    require_c_contiguous(k_v, "out", itemsize);

    auto* bqp_ptr = reinterpret_cast<const double*>(bqp_v.ptr);
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

    const int q_block_i = std::min(q_block, naux_i);
    const int64_t max_cols = (int64_t)q_block_i * (int64_t)nocc_i;
    ensure_v_capacity((size_t)(max_cols * nao));
    ensure_bq_capacity((size_t)((int64_t)q_block_i * nao * nao));

    const double alpha = 1.0;
    const double beta0 = 0.0;
    const double beta = 1.0;

    for (int q0 = 0; q0 < naux_i; q0 += q_block_i) {
      const int q = std::min(q_block_i, naux_i - q0);
      const int64_t cols = (int64_t)q * (int64_t)nocc_i;
      if (cols <= 0) continue;

      hf_df_jk_unpack_qp_to_bq_f64(bqp_ptr, nao_i, naux_i, q0, q, bq_buf_, stream);

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

  void k_block_from_qp_d(
      const py::object& b_qp,
      const py::object& d,
      const py::object& out,
      int nao,
      int row0,
      int row_count,
      int col0,
      int col_count,
      int q_block,
      int col_block,
      uint64_t stream_ptr,
      int math_mode,
      bool sync) {
    const CudaArrayView bqp_v = cuda_array_view_from_object(b_qp, "B_Qp");
    const CudaArrayView d_v = cuda_array_view_from_object(d, "D");
    const CudaArrayView out_v = cuda_array_view_from_object(out, "out");

    require_typestr_f64(bqp_v, "B_Qp");
    require_typestr_f64(d_v, "D");
    require_typestr_f64(out_v, "out");

    if (out_v.read_only) {
      throw std::invalid_argument("out must be a writable CUDA array");
    }
    if (bqp_v.shape.size() != 2) {
      throw std::invalid_argument("B_Qp must have shape (naux, ntri)");
    }
    if (d_v.shape.size() != 2 || d_v.shape[0] != d_v.shape[1]) {
      throw std::invalid_argument("D must have shape (nao, nao)");
    }
    if (out_v.shape.size() != 2) {
      throw std::invalid_argument("out must have shape (row_count, col_count)");
    }

    const int64_t naux = bqp_v.shape[0];
    const int64_t ntri = bqp_v.shape[1];
    const int64_t nao0 = d_v.shape[0];
    if ((int64_t)nao != nao0) {
      throw std::invalid_argument("nao mismatch between provided nao and D");
    }
    if (row0 < 0 || row_count < 0 || col0 < 0 || col_count < 0) {
      throw std::invalid_argument("invalid row/col arguments");
    }
    if (row0 > nao || row_count > (nao - row0)) {
      throw std::invalid_argument("row range out of bounds");
    }
    if (col0 > nao || col_count > (nao - col0)) {
      throw std::invalid_argument("col range out of bounds");
    }
    if ((int64_t)out_v.shape[0] != (int64_t)row_count || (int64_t)out_v.shape[1] != (int64_t)col_count) {
      throw std::invalid_argument("out shape mismatch with row_count/col_count");
    }

    const int64_t expected_ntri = (nao0 * (nao0 + 1)) / 2;
    if (ntri != expected_ntri) {
      throw std::invalid_argument("B_Qp must have ntri=nao*(nao+1)//2");
    }
    if (q_block <= 0) {
      throw std::invalid_argument("q_block must be > 0");
    }
    if (col_block <= 0) {
      throw std::invalid_argument("col_block must be > 0");
    }

    constexpr int64_t itemsize = (int64_t)sizeof(double);
    require_c_contiguous(bqp_v, "B_Qp", itemsize);
    require_c_contiguous(d_v, "D", itemsize);
    require_c_contiguous(out_v, "out", itemsize);

    auto* bqp_ptr = reinterpret_cast<const double*>(bqp_v.ptr);
    auto* d_ptr = reinterpret_cast<const double*>(d_v.ptr);
    auto* out_ptr = reinterpret_cast<double*>(out_v.ptr);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    throw_on_cublas_error(cublasSetStream(cublas_.h, stream), "cublasSetStream");
    const ScopedCublasMathMode math_ctx(cublas_.h, math_mode);

    if (naux > (int64_t)std::numeric_limits<int>::max() || nao > (int64_t)std::numeric_limits<int>::max()) {
      throw std::invalid_argument("dimensions exceed int32 limits for cuBLAS");
    }
    const int naux_i = (int)naux;
    const int nao_i = (int)nao;
    const int row_count_i = (int)row_count;
    const int col_count_i = (int)col_count;
    if (naux_i < 0 || nao_i <= 0 || row_count_i < 0 || col_count_i < 0) throw std::invalid_argument("invalid dimensions");

    // Empty cases: always zero out for predictable semantics.
    const int64_t out_elems = (int64_t)row_count_i * (int64_t)col_count_i;
    throw_on_cuda_error(
        cudaMemsetAsync(out_ptr, 0, (size_t)out_elems * sizeof(double), stream), "cudaMemsetAsync(out)");
    if (naux_i == 0 || row_count_i == 0 || col_count_i == 0) {
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      return;
    }

    const int q_block_i = std::min(q_block, naux_i);
    const int col_block_i = std::min(col_block, col_count_i);

    // Work buffers:
    //   a_buf_: rows (q*row_count, nao) then yflat (row_count, q*nao)
    //   y_buf_: y2d (q*row_count, nao)
    //   z_buf_: zflat_t (q*nao, cb)
    ensure_a_capacity((size_t)((int64_t)q_block_i * (int64_t)row_count_i * (int64_t)nao_i));
    ensure_y_capacity((size_t)((int64_t)q_block_i * (int64_t)row_count_i * (int64_t)nao_i));
    ensure_z_capacity((size_t)((int64_t)q_block_i * (int64_t)nao_i * (int64_t)col_block_i));

    const double alpha = 1.0;
    const double beta0 = 0.0;
    const double beta = 1.0;

    for (int q0 = 0; q0 < naux_i; q0 += q_block_i) {
      const int q = std::min(q_block_i, naux_i - q0);
      if (q <= 0) continue;
      const int64_t m_rows64 = (int64_t)q * (int64_t)row_count_i;
      const int64_t k64 = (int64_t)q * (int64_t)nao_i;
      if (m_rows64 > (int64_t)std::numeric_limits<int>::max() || k64 > (int64_t)std::numeric_limits<int>::max()) {
        throw std::invalid_argument("block sizes exceed int32 limits for cuBLAS");
      }
      const int m_rows = (int)m_rows64;
      const int k = (int)k64;

      hf_df_jk_extract_qp_rows_fullcols_f64(bqp_ptr, nao_i, naux_i, q0, q, row0, row_count_i, a_buf_, stream);

      // GEMM1: y2d (m_rows x nao) = rows (m_rows x nao) @ D (nao x nao).
      // We compute the transpose in column-major space to avoid a transpose kernel:
      //   (y2d)^T (nao x m_rows) = D^T (nao x nao) @ (rows)^T (nao x m_rows).
      throw_on_cublas_error(
          gemm_f64(
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              /*m=*/nao_i,
              /*n=*/m_rows,
              /*k=*/nao_i,
              &alpha,
              d_ptr,
              /*lda=*/nao_i,
              a_buf_,
              /*ldb=*/nao_i,
              &beta0,
              y_buf_,
              /*ldc=*/nao_i),
          "gemm_f64(y2d)");

      hf_df_jk_repack_y2d_to_yflat_f64(y_buf_, q, row_count_i, nao_i, a_buf_, stream);

      for (int c_off = 0; c_off < col_count_i; c_off += col_block_i) {
        const int cb = std::min(col_block_i, col_count_i - c_off);
        if (cb <= 0) continue;
        const int c_abs = col0 + c_off;
        hf_df_jk_extract_qp_cols_to_zflat_t_f64(bqp_ptr, nao_i, naux_i, q0, q, c_abs, cb, z_buf_, stream);

        // GEMM2: out[:, c_off:c_off+cb] += yflat (row_count x k) @ zflat_t (k x cb)
        // by writing its transpose into the column-major reinterpretation of out:
        //   out_cm[c_off:c_off+cb, :] (cb x row_count) += z_cm (cb x k) @ y_cm (k x row_count)
        double* out_block = out_ptr + (int64_t)c_off;
        throw_on_cublas_error(
            gemm_f64(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                /*m=*/cb,
                /*n=*/row_count_i,
                /*k=*/k,
                &alpha,
                z_buf_,
                /*lda=*/cb,
                a_buf_,
                /*ldb=*/k,
                &beta,
                out_block,
                /*ldc=*/col_count_i),
            "gemm_f64(out_block)");
      }
    }

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

  void ensure_a_capacity(size_t need_elems) {
    if (need_elems <= a_elems_) return;
    if (a_buf_) {
      cudaFree(a_buf_);
      a_buf_ = nullptr;
    }
    a_elems_ = 0;
    double* ptr = nullptr;
    throw_on_cuda_error(cudaMalloc(&ptr, need_elems * sizeof(double)), "cudaMalloc(A)");
    a_buf_ = ptr;
    a_elems_ = need_elems;
  }

  void ensure_y_capacity(size_t need_elems) {
    if (need_elems <= y_elems_) return;
    if (y_buf_) {
      cudaFree(y_buf_);
      y_buf_ = nullptr;
    }
    y_elems_ = 0;
    double* ptr = nullptr;
    throw_on_cuda_error(cudaMalloc(&ptr, need_elems * sizeof(double)), "cudaMalloc(Y)");
    y_buf_ = ptr;
    y_elems_ = need_elems;
  }

  void ensure_z_capacity(size_t need_elems) {
    if (need_elems <= z_elems_) return;
    if (z_buf_) {
      cudaFree(z_buf_);
      z_buf_ = nullptr;
    }
    z_elems_ = 0;
    double* ptr = nullptr;
    throw_on_cuda_error(cudaMalloc(&ptr, need_elems * sizeof(double)), "cudaMalloc(Z)");
    z_buf_ = ptr;
    z_elems_ = need_elems;
  }

  CublasHandle cublas_;
  double* v_buf_ = nullptr;
  size_t v_elems_ = 0;
  double* bq_buf_ = nullptr;
  size_t bq_elems_ = 0;
  double* a_buf_ = nullptr;
  size_t a_elems_ = 0;
  double* y_buf_ = nullptr;
  size_t y_elems_ = 0;
  double* z_buf_ = nullptr;
  size_t z_elems_ = 0;
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
      "density_syrk",
      [](DFJKWorkspace& self,
         const py::object& cw,
         const py::object& out,
         uint64_t stream,
         int math_mode,
         bool sync) { self.density_syrk(cw, out, stream, math_mode, sync); },
      py::arg("Cw"),
      py::arg("out"),
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false,
      "Compute density D = Cw @ Cw^T via cublasDsyrk (upper triangle) + fill_lower.");

  cls.def(
      "jk_from_bq_cw",
      [](DFJKWorkspace& self,
         const py::object& bq,
         const py::object& cw,
         const py::object& d_mat,
         const py::object& out_j,
         const py::object& out_k,
         int q_block,
         uint64_t stream,
         int math_mode,
         bool sync) {
        self.jk_from_bq_cw(bq, cw, d_mat, out_j, out_k, q_block, stream, math_mode, sync);
      },
      py::arg("BQ"),
      py::arg("Cw"),
      py::arg("D"),
      py::arg("out_J"),
      py::arg("out_K"),
      py::arg("q_block") = 128,
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false,
      "Fused J+K build: reads BQ once per q-block, accumulates both Coulomb J and exchange K.");

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

  cls.def(
      "k_from_qp_cw",
      [](DFJKWorkspace& self,
         const py::object& b_qp,
         const py::object& cw,
         const py::object& out_k,
         int q_block,
         uint64_t stream,
         int math_mode,
         bool sync) { self.k_from_qp_cw(b_qp, cw, out_k, q_block, stream, math_mode, sync); },
      py::arg("B_Qp"),
      py::arg("Cw"),
      py::arg("out"),
      py::arg("q_block") = 128,
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false);

  cls.def(
      "k_block_from_qp_d",
      [](DFJKWorkspace& self,
         const py::object& b_qp,
         const py::object& d,
         const py::object& out,
         int nao,
         int row0,
         int row_count,
         int col0,
         int col_count,
         int q_block,
         int col_block,
         uint64_t stream,
         int math_mode,
         bool sync) {
        self.k_block_from_qp_d(
            b_qp, d, out, nao, row0, row_count, col0, col_count, q_block, col_block, stream, math_mode, sync);
      },
      py::arg("B_Qp"),
      py::arg("D"),
      py::arg("out"),
      py::arg("nao"),
      py::arg("row0"),
      py::arg("row_count"),
      py::arg("col0"),
      py::arg("col_count"),
      py::arg("q_block") = 128,
      py::arg("col_block") = 64,
      py::arg("stream") = 0,
      py::arg("math_mode") = -1,
      py::arg("sync") = false);

  m.def(
      "symmetrize_inplace_f64",
      [](const py::object& a, uint64_t stream_ptr, bool sync) {
        const CudaArrayView a_v = cuda_array_view_from_object(a, "a");
        require_typestr_f64(a_v, "a");
        if (a_v.read_only) {
          throw std::invalid_argument("a must be a writable CUDA array");
        }
        if (a_v.shape.size() != 2 || a_v.shape[0] != a_v.shape[1]) {
          throw std::invalid_argument("a must have shape (n, n)");
        }
        constexpr int64_t itemsize = (int64_t)sizeof(double);
        require_c_contiguous(a_v, "a", itemsize);

        const int64_t n = a_v.shape[0];
        if (n > (int64_t)std::numeric_limits<int>::max()) {
          throw std::invalid_argument("n exceeds int32 limits");
        }
        auto* a_ptr = reinterpret_cast<double*>(a_v.ptr);
        validate_cuda_device_pointer(a_ptr, "a");

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        hf_df_jk_symmetrize_inplace_f64(a_ptr, (int)n, stream);
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("a"),
      py::arg("stream") = 0,
      py::arg("sync") = false,
      "Symmetrize a square float64 matrix in-place: A=(A+A^T)/2.");
}
