#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

extern "C" void guga_cuda_linalg_residual_launch(
    const double* ax,
    const double* x,
    const double* evals,
    int n,
    int nroots,
    double* out_r);

extern "C" void guga_cuda_linalg_precond_launch(
    const double* r,
    const double* diag,
    const double* evals,
    int n,
    int nroots,
    double denom_tol,
    double* out_t);

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void throw_on_cublas_error(cublasStatus_t stat, const char* what) {
  if (stat == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string((int)stat));
}

inline void throw_on_cusolver_error(cusolverStatus_t stat, const char* what) {
  if (stat == CUSOLVER_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": cusolverStatus=" + std::to_string((int)stat));
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

struct CusolverHandle {
  cusolverDnHandle_t h = nullptr;
  CusolverHandle() { throw_on_cusolver_error(cusolverDnCreate(&h), "cusolverDnCreate"); }
  ~CusolverHandle() {
    if (h) cusolverDnDestroy(h);
  }
  CusolverHandle(const CusolverHandle&) = delete;
  CusolverHandle& operator=(const CusolverHandle&) = delete;
  CusolverHandle(CusolverHandle&& other) noexcept : h(other.h) { other.h = nullptr; }
  CusolverHandle& operator=(CusolverHandle&& other) noexcept {
    if (this != &other) {
      if (h) cusolverDnDestroy(h);
      h = other.h;
      other.h = nullptr;
    }
    return *this;
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

py::dict cublas_emulation_info() {
  auto query = [](cublasHandle_t h) -> py::dict {
    int ver = 0;
    throw_on_cublas_error(cublasGetVersion(h, &ver), "cublasGetVersion");

    py::dict out;
    out["cublas_version"] = ver;

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    throw_on_cublas_error(cublasGetMathMode(h, &math_mode), "cublasGetMathMode");
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
    throw_on_cublas_error(cublasGetEmulationStrategy(h, &strategy), "cublasGetEmulationStrategy");

    cudaEmulationSpecialValuesSupport special_mask = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    throw_on_cublas_error(
        cublasGetEmulationSpecialValuesSupport(h, &special_mask), "cublasGetEmulationSpecialValuesSupport");

    cudaEmulationMantissaControl mantissa_control = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaControl(h, &mantissa_control),
        "cublasGetFixedPointEmulationMantissaControl");

    int max_bits = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMaxMantissaBitCount(h, &max_bits),
        "cublasGetFixedPointEmulationMaxMantissaBitCount");

    int bit_offset = 0;
    throw_on_cublas_error(
        cublasGetFixedPointEmulationMantissaBitOffset(h, &bit_offset),
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

    return out;
  };

  CublasHandle cublas;
  return query(cublas.h);
}

py::tuple eigh_sym(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
  auto buf = a.request();
  if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
    throw std::invalid_argument("a must be a square 2D array");
  }

  const int n = (int)buf.shape[0];
  if (n <= 0) {
    throw std::invalid_argument("matrix dimension must be >= 1");
  }

  const size_t bytes = (size_t)n * (size_t)n * sizeof(double);
  double* dA = nullptr;
  double* dW = nullptr;
  double* dWork = nullptr;
  int* dInfo = nullptr;

  throw_on_cuda_error(cudaMalloc((void**)&dA, bytes), "cudaMalloc(dA)");
  throw_on_cuda_error(cudaMalloc((void**)&dW, (size_t)n * sizeof(double)), "cudaMalloc(dW)");
  throw_on_cuda_error(cudaMalloc((void**)&dInfo, sizeof(int)), "cudaMalloc(dInfo)");

  // Treat input as column-major; for symmetric matrices this is fine even if NumPy is C-order.
  throw_on_cuda_error(cudaMemcpy(dA, buf.ptr, bytes, cudaMemcpyHostToDevice), "H2D a");

  CusolverHandle solver;
  int lwork = 0;
  throw_on_cusolver_error(
      cusolverDnDsyevd_bufferSize(
          solver.h,
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER,
          n,
          dA,
          n,
          dW,
          &lwork),
      "cusolverDnDsyevd_bufferSize");

  throw_on_cuda_error(cudaMalloc((void**)&dWork, (size_t)lwork * sizeof(double)), "cudaMalloc(dWork)");

  throw_on_cusolver_error(
      cusolverDnDsyevd(
          solver.h,
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER,
          n,
          dA,
          n,
          dW,
          dWork,
          lwork,
          dInfo),
      "cusolverDnDsyevd");

  int info = 0;
  throw_on_cuda_error(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "D2H devInfo");
  if (info != 0) {
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dWork);
    cudaFree(dInfo);
    throw std::runtime_error("cusolverDnDsyevd failed, devInfo=" + std::to_string(info));
  }

  py::array_t<double> w({n});
  throw_on_cuda_error(cudaMemcpy(w.mutable_data(), dW, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost), "D2H w");

  std::vector<double> hV((size_t)n * (size_t)n);
  throw_on_cuda_error(cudaMemcpy(hV.data(), dA, bytes, cudaMemcpyDeviceToHost), "D2H V");

  py::array_t<double> v({n, n});
  double* vptr = v.mutable_data();
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      vptr[(size_t)i * (size_t)n + (size_t)j] = hV[(size_t)i + (size_t)j * (size_t)n];
    }
  }

  cudaFree(dA);
  cudaFree(dW);
  cudaFree(dWork);
  cudaFree(dInfo);
  return py::make_tuple(w, v);
}

py::array_t<double> gemm(py::array_t<double, py::array::c_style | py::array::forcecast> a,
                         py::array_t<double, py::array::c_style | py::array::forcecast> b) {
  auto a_buf = a.request();
  auto b_buf = b.request();
  if (a_buf.ndim != 2 || b_buf.ndim != 2) {
    throw std::invalid_argument("a and b must be 2D arrays");
  }

  const int m = (int)a_buf.shape[0];
  const int k = (int)a_buf.shape[1];
  const int k2 = (int)b_buf.shape[0];
  const int n = (int)b_buf.shape[1];
  if (m < 0 || n < 0 || k < 0 || k2 < 0) {
    throw std::invalid_argument("invalid matrix shape");
  }
  if (k != k2) {
    throw std::invalid_argument("a and b have incompatible shapes");
  }
  if (m == 0 || n == 0 || k == 0) {
    return py::array_t<double>({m, n});
  }

  const size_t a_bytes = (size_t)m * (size_t)k * sizeof(double);
  const size_t b_bytes = (size_t)k * (size_t)n * sizeof(double);
  const size_t c_bytes = (size_t)m * (size_t)n * sizeof(double);

  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;
  throw_on_cuda_error(cudaMalloc((void**)&dA, a_bytes), "cudaMalloc(dA)");
  throw_on_cuda_error(cudaMalloc((void**)&dB, b_bytes), "cudaMalloc(dB)");
  throw_on_cuda_error(cudaMalloc((void**)&dC, c_bytes), "cudaMalloc(dC)");

  throw_on_cuda_error(cudaMemcpy(dA, a_buf.ptr, a_bytes, cudaMemcpyHostToDevice), "H2D a");
  throw_on_cuda_error(cudaMemcpy(dB, b_buf.ptr, b_bytes, cudaMemcpyHostToDevice), "H2D b");

  CublasHandle cublas;
  const double alpha = 1.0;
  const double beta = 0.0;

  // Row-major GEMM via column-major GEMM on transposed interpretation:
  //   C_r = A_r * B_r  (m x n, row-major)
  //   C_c = C_r^T = B_r^T * A_r^T
  // Treat A_r and B_r buffers as column-major A_c=A_r^T and B_c=B_r^T.
  // Compute C_c (n x m) in column-major into dC, which matches C_r in row-major.
  throw_on_cublas_error(
      cublasDgemm(
          cublas.h,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          dB,
          n,
          dA,
          k,
          &beta,
          dC,
          n),
      "cublasDgemm");

  py::array_t<double> c({m, n});
  throw_on_cuda_error(cudaMemcpy(c.mutable_data(), dC, c_bytes, cudaMemcpyDeviceToHost), "D2H c");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return c;
}

static void orthonormalize_columns(
    cublasHandle_t cublas,
    double* dV,
    int n,
    int nvec,
    double lindep) {
  for (int j = 0; j < nvec; j++) {
    double* vj = dV + (size_t)j * (size_t)n;
    for (int i = 0; i < j; i++) {
      const double* vi = dV + (size_t)i * (size_t)n;
      double dot = 0.0;
      throw_on_cublas_error(cublasDdot(cublas, n, vi, 1, vj, 1, &dot), "cublasDdot");
      const double alpha = -dot;
      throw_on_cublas_error(cublasDaxpy(cublas, n, &alpha, vi, 1, vj, 1), "cublasDaxpy");
    }
    double nrm = 0.0;
    throw_on_cublas_error(cublasDnrm2(cublas, n, vj, 1, &nrm), "cublasDnrm2");
    if ((double)(nrm * nrm) <= lindep) {
      throw std::runtime_error("orthonormalize_columns: linearly dependent vector encountered");
    }
    const double inv = 1.0 / nrm;
    throw_on_cublas_error(cublasDscal(cublas, n, &inv, vj, 1), "cublasDscal");
  }
}

struct DenseSymDavidsonWorkspace {
  int n = 0;
  int nroots = 0;
  int max_space = 0;
  int max_space_eff = 0;
  int ldH = 0;
  bool has_matrix = false;

  std::vector<double> h_diag;
  std::vector<int> h_guess_idx;

  CublasHandle cublas;
  CusolverHandle solver;

  bool use_gemm_ex = false;
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_64F;
  cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;

  void* dCublasWorkspace = nullptr;
  size_t cublas_workspace_bytes = 0;

  double* dA = nullptr;
  double* dDiag = nullptr;
  double* dV = nullptr;
  double* dW = nullptr;
  double* dX = nullptr;
  double* dAX = nullptr;
  double* dR = nullptr;
  double* dT = nullptr;
  double* dH = nullptr;
  double* dEigs = nullptr;
  double* dWork = nullptr;
  double* dProj = nullptr;
  int* dInfo = nullptr;
  int lwork = 0;

  void gemm(
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
      int ldc,
      const char* what) {
    if (!use_gemm_ex) {
      throw_on_cublas_error(cublasDgemm(cublas.h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), what);
      return;
    }

    throw_on_cublas_error(
        cublasGemmEx(
            cublas.h,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            A,
            CUDA_R_64F,
            lda,
            B,
            CUDA_R_64F,
            ldb,
            beta,
            C,
            CUDA_R_64F,
            ldc,
            gemm_compute_type,
            gemm_algo),
        what);
  }

  DenseSymDavidsonWorkspace(int n_, int nroots_, int max_space_)
      : n(std::max(0, n_)), nroots(std::max(1, nroots_)) {
    if (n <= 0) {
      throw std::invalid_argument("n must be >= 1");
    }
    if (nroots > n) {
      throw std::invalid_argument("nroots must be <= n");
    }
    max_space = std::max(nroots + 2, max_space_);
    max_space_eff = max_space + (nroots - 1) * 4;
    ldH = max_space_eff;

    const size_t a_bytes = (size_t)n * (size_t)n * sizeof(double);
    const size_t v_bytes = (size_t)n * (size_t)max_space_eff * sizeof(double);
    const size_t x_bytes = (size_t)n * (size_t)nroots * sizeof(double);
    const size_t h_bytes = (size_t)max_space_eff * (size_t)max_space_eff * sizeof(double);

    throw_on_cuda_error(cudaMalloc((void**)&dA, a_bytes), "cudaMalloc(dA)");
    throw_on_cuda_error(cudaMalloc((void**)&dDiag, (size_t)n * sizeof(double)), "cudaMalloc(dDiag)");
    throw_on_cuda_error(cudaMalloc((void**)&dV, v_bytes), "cudaMalloc(dV)");
    throw_on_cuda_error(cudaMalloc((void**)&dW, v_bytes), "cudaMalloc(dW)");
    throw_on_cuda_error(cudaMalloc((void**)&dX, x_bytes), "cudaMalloc(dX)");
    throw_on_cuda_error(cudaMalloc((void**)&dAX, x_bytes), "cudaMalloc(dAX)");
    throw_on_cuda_error(cudaMalloc((void**)&dR, x_bytes), "cudaMalloc(dR)");
    throw_on_cuda_error(cudaMalloc((void**)&dT, x_bytes), "cudaMalloc(dT)");
    throw_on_cuda_error(cudaMalloc((void**)&dH, h_bytes), "cudaMalloc(dH)");
    throw_on_cuda_error(cudaMalloc((void**)&dEigs, (size_t)max_space_eff * sizeof(double)), "cudaMalloc(dEigs)");
    throw_on_cuda_error(
        cudaMalloc((void**)&dProj, (size_t)max_space_eff * (size_t)nroots * sizeof(double)),
        "cudaMalloc(dProj)");
    throw_on_cuda_error(cudaMalloc((void**)&dInfo, sizeof(int)), "cudaMalloc(dInfo)");

    // Workspace sized for maximum subspace.
    throw_on_cusolver_error(
        cusolverDnDsyevd_bufferSize(
            solver.h,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            max_space_eff,
            dH,
            ldH,
            dEigs,
            &lwork),
        "cusolverDnDsyevd_bufferSize(max_space)");
    if (lwork <= 0) {
      throw std::runtime_error("cusolverDnDsyevd_bufferSize returned invalid lwork");
    }
    throw_on_cuda_error(cudaMalloc((void**)&dWork, (size_t)lwork * sizeof(double)), "cudaMalloc(dWork)");
  }

  ~DenseSymDavidsonWorkspace() { release(); }

  DenseSymDavidsonWorkspace(const DenseSymDavidsonWorkspace&) = delete;
  DenseSymDavidsonWorkspace& operator=(const DenseSymDavidsonWorkspace&) = delete;

  DenseSymDavidsonWorkspace(DenseSymDavidsonWorkspace&& other) noexcept { *this = std::move(other); }
  DenseSymDavidsonWorkspace& operator=(DenseSymDavidsonWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      n = other.n;
      nroots = other.nroots;
      max_space = other.max_space;
      max_space_eff = other.max_space_eff;
      ldH = other.ldH;
      has_matrix = other.has_matrix;
      h_diag = std::move(other.h_diag);
      h_guess_idx = std::move(other.h_guess_idx);
      cublas = std::move(other.cublas);
      solver = std::move(other.solver);
      use_gemm_ex = other.use_gemm_ex;
      gemm_compute_type = other.gemm_compute_type;
      gemm_algo = other.gemm_algo;
      dCublasWorkspace = other.dCublasWorkspace;
      cublas_workspace_bytes = other.cublas_workspace_bytes;
      dA = other.dA;
      dDiag = other.dDiag;
      dV = other.dV;
      dW = other.dW;
      dX = other.dX;
      dAX = other.dAX;
      dR = other.dR;
      dT = other.dT;
      dH = other.dH;
      dEigs = other.dEigs;
      dWork = other.dWork;
      dProj = other.dProj;
      dInfo = other.dInfo;
      lwork = other.lwork;
      other.n = 0;
      other.nroots = 0;
      other.max_space = 0;
      other.max_space_eff = 0;
      other.ldH = 0;
      other.has_matrix = false;
      other.use_gemm_ex = false;
      other.gemm_compute_type = CUBLAS_COMPUTE_64F;
      other.gemm_algo = CUBLAS_GEMM_DEFAULT;
      other.dCublasWorkspace = nullptr;
      other.cublas_workspace_bytes = 0;
      other.dA = nullptr;
      other.dDiag = nullptr;
      other.dV = nullptr;
      other.dW = nullptr;
      other.dX = nullptr;
      other.dAX = nullptr;
      other.dR = nullptr;
      other.dT = nullptr;
      other.dH = nullptr;
      other.dEigs = nullptr;
      other.dWork = nullptr;
      other.dProj = nullptr;
      other.dInfo = nullptr;
      other.lwork = 0;
    }
    return *this;
  }

  void release() noexcept {
    if (dCublasWorkspace) {
      if (cublas.h) cublasSetWorkspace(cublas.h, nullptr, 0);
      cudaFree(dCublasWorkspace);
    }
    if (dA) cudaFree(dA);
    if (dDiag) cudaFree(dDiag);
    if (dV) cudaFree(dV);
    if (dW) cudaFree(dW);
    if (dX) cudaFree(dX);
    if (dAX) cudaFree(dAX);
    if (dR) cudaFree(dR);
    if (dT) cudaFree(dT);
    if (dH) cudaFree(dH);
    if (dEigs) cudaFree(dEigs);
    if (dWork) cudaFree(dWork);
    if (dProj) cudaFree(dProj);
    if (dInfo) cudaFree(dInfo);
    dA = nullptr;
    dDiag = nullptr;
    dV = nullptr;
    dW = nullptr;
    dX = nullptr;
    dAX = nullptr;
    dR = nullptr;
    dT = nullptr;
    dH = nullptr;
    dEigs = nullptr;
    dWork = nullptr;
    dProj = nullptr;
    dInfo = nullptr;
    dCublasWorkspace = nullptr;
    cublas_workspace_bytes = 0;
  }

  void set_matrix(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    auto buf = a.request();
    if (buf.ndim != 2 || (int)buf.shape[0] != n || (int)buf.shape[1] != n) {
      throw std::invalid_argument("matrix has wrong shape");
    }

    const size_t a_bytes = (size_t)n * (size_t)n * sizeof(double);
    throw_on_cuda_error(cudaMemcpy(dA, buf.ptr, a_bytes, cudaMemcpyHostToDevice), "H2D A");

    const double* a_ptr = static_cast<const double*>(buf.ptr);
    h_diag.assign((size_t)n, 0.0);
    for (int i = 0; i < n; i++) {
      h_diag[(size_t)i] = a_ptr[(size_t)i * (size_t)n + (size_t)i];
    }
    throw_on_cuda_error(
        cudaMemcpy(dDiag, h_diag.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice), "H2D diag");

    // Initial guess indices: smallest diagonal entries.
    std::vector<int> idx((size_t)n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(),
        idx.begin() + nroots,
        idx.end(),
        [&](int i, int j) { return h_diag[(size_t)i] < h_diag[(size_t)j]; });
    h_guess_idx.assign(idx.begin(), idx.begin() + nroots);

    has_matrix = true;
  }

  py::dict get_cublas_emulation_info() const {
    // Keep this query bound to this workspace's handle (important if configured via setters).
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
    throw_on_cublas_error(cublasSetFixedPointEmulationMantissaControl(cublas.h, c), "cublasSetFixedPointEmulationMantissaControl");
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

  std::string get_gemm_backend() const {
    if (!use_gemm_ex) return "dgemm";
    if (gemm_compute_type == CUBLAS_COMPUTE_64F) return "gemmex_fp64";
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
    if (gemm_compute_type == CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) return "gemmex_emulated_fixedpoint";
#endif
    return "gemmex_unknown";
  }

  void set_gemm_backend(const std::string& backend) {
    if (backend == "dgemm") {
      use_gemm_ex = false;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      return;
    }
    if (backend == "gemmex" || backend == "gemmex_fp64") {
      use_gemm_ex = true;
      gemm_compute_type = CUBLAS_COMPUTE_64F;
      return;
    }
    if (backend == "gemmex_emulated_fixedpoint") {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 130000
      use_gemm_ex = true;
      gemm_compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
      return;
#else
      throw std::runtime_error("gemmex_emulated_fixedpoint requires CUDA 13.0+ (CUBLAS_VERSION>=130000)");
#endif
    }
    throw std::invalid_argument("unknown GEMM backend: " + backend);
  }

  int get_gemm_algo() const { return (int)gemm_algo; }

  void set_gemm_algo(int algo) { gemm_algo = (cublasGemmAlgo_t)algo; }

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
    if (dCublasWorkspace) {
      throw_on_cuda_error(cudaFree(dCublasWorkspace), "cudaFree(cublas workspace)");
      dCublasWorkspace = nullptr;
      cublas_workspace_bytes = 0;
    }
    if (bytes > 0) {
      void* p = nullptr;
      throw_on_cuda_error(cudaMalloc(&p, bytes), "cudaMalloc(cublas workspace)");
      dCublasWorkspace = p;
      cublas_workspace_bytes = bytes;
    }
    throw_on_cublas_error(cublasSetWorkspace(cublas.h, dCublasWorkspace, cublas_workspace_bytes), "cublasSetWorkspace");
  }

  py::tuple solve(int max_cycle, double tol, double lindep, double denom_tol) {
    if (!has_matrix) {
      throw std::runtime_error("matrix not set; call set_matrix(a) first");
    }

    max_cycle = std::max(1, max_cycle);
    tol = std::max(0.0, tol);
    lindep = std::max(0.0, lindep);
    denom_tol = std::max(1e-16, denom_tol);

    const double tol_residual = (tol > 0.0) ? std::sqrt(tol) : 0.0;

    const size_t x_bytes = (size_t)n * (size_t)nroots * sizeof(double);

    // Initial guess vectors (host) then upload to device.
    std::vector<double> h_v0((size_t)n * (size_t)nroots, 0.0);
    for (int r = 0; r < nroots; r++) {
      const int ii = h_guess_idx[(size_t)r];
      h_v0[(size_t)ii + (size_t)r * (size_t)n] = 1.0;
    }
    throw_on_cuda_error(cudaMemcpy(dV, h_v0.data(), x_bytes, cudaMemcpyHostToDevice), "H2D V0");

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    int m = nroots;

    // W[:,0:m] = A * V[:,0:m]
    gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, dA, n, dV, n, &zero, dW, n, "gemm(W=A*V)");

    std::vector<char> conv((size_t)nroots, 0);
    std::vector<double> evals((size_t)nroots, 0.0);
    std::vector<double> e_prev((size_t)nroots, 0.0);
    std::vector<double> e_curr((size_t)nroots, 0.0);
    std::vector<double> h_rtr((size_t)nroots * (size_t)nroots, 0.0);
    bool have_prev = false;
    int niter = 0;
    bool restart_next = false;

    for (int it = 0; it < max_cycle; it++) {
      niter = it + 1;

      if (restart_next) {
        // Restart with current approximate eigenvectors (columns of dX).
        throw_on_cuda_error(cudaMemcpy(dV, dX, x_bytes, cudaMemcpyDeviceToDevice), "D2D restart X->V");
        orthonormalize_columns(cublas.h, dV, n, nroots, lindep);
        m = nroots;
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, dA, n, dV, n, &zero, dW, n, "gemm(W=A*V restart)");
        restart_next = false;
      }

      // Hsub = V^T W (m x m)
      gemm(CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, dV, n, dW, n, &zero, dH, ldH, "gemm(H=V^T*W)");

      // Solve Hsub eigenproblem on GPU; dH is overwritten with eigenvectors.
      throw_on_cusolver_error(
          cusolverDnDsyevd(
              solver.h,
              CUSOLVER_EIG_MODE_VECTOR,
              CUBLAS_FILL_MODE_LOWER,
              m,
              dH,
              ldH,
              dEigs,
              dWork,
              lwork,
              dInfo),
          "cusolverDnDsyevd(Hsub)");

      int info = 0;
      throw_on_cuda_error(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "D2H devInfo(Hsub)");
      if (info != 0) {
        throw std::runtime_error("cusolverDnDsyevd(Hsub) failed, devInfo=" + std::to_string(info));
      }

      throw_on_cuda_error(
          cudaMemcpy(e_curr.data(), dEigs, (size_t)nroots * sizeof(double), cudaMemcpyDeviceToHost), "D2H eigs roots");

      // X = V * U_root, AX = W * U_root, where U_root is first nroots columns of dH.
      gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, nroots, m, &one, dV, n, dH, ldH, &zero, dX, n, "gemm(X=V*U)");
      gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, nroots, m, &one, dW, n, dH, ldH, &zero, dAX, n, "gemm(AX=W*U)");

      guga_cuda_linalg_residual_launch(dAX, dX, dEigs, n, nroots, dR);

      // Compute residual norms via Gram matrix: RtR = R^T R, norm_r = sqrt(diag(RtR)).
      // Note: overwrite dH scratch (eigenvectors are no longer needed after X/AX are formed).
      gemm(CUBLAS_OP_T, CUBLAS_OP_N, nroots, nroots, n, &one, dR, n, dR, n, &zero, dH, ldH, "gemm(RtR=R^T*R)");
      throw_on_cuda_error(
          cudaMemcpy2D(
              h_rtr.data(),
              (size_t)nroots * sizeof(double),
              dH,
              (size_t)ldH * sizeof(double),
              (size_t)nroots * sizeof(double),
              (size_t)nroots,
              cudaMemcpyDeviceToHost),
          "D2H RtR");

      int nconv = 0;
      for (int r = 0; r < nroots; r++) {
        const double rr = h_rtr[(size_t)r * (size_t)nroots + (size_t)r];
        const double nrm = std::sqrt(std::max(0.0, rr));
        const double de = have_prev ? (e_curr[(size_t)r] - e_prev[(size_t)r]) : e_curr[(size_t)r];
        conv[(size_t)r] = (std::abs(de) < tol && nrm < tol_residual) ? 1 : 0;
        if (conv[(size_t)r]) nconv++;
      }
      if (nconv == nroots) {
        evals = e_curr;
        break;
      }
      e_prev = e_curr;
      have_prev = true;

      // Determine which roots generate trial vectors. Match PySCF: drop
      // unconverged roots with tiny residual norm (dx_norm^2 <= lindep).
      int nraw = 0;
      for (int r = 0; r < nroots; r++) {
        if (conv[(size_t)r]) continue;
        const double rr2 = h_rtr[(size_t)r * (size_t)nroots + (size_t)r];
        if (rr2 <= lindep) continue;
        nraw++;
      }

      if (nraw == 0) {
        // No linearly independent correction vectors; fall back to residual-only
        // convergence and stop (PySCF: conv = dx_norm < sqrt(tol)).
        for (int r = 0; r < nroots; r++) {
          const double rr2 = h_rtr[(size_t)r * (size_t)nroots + (size_t)r];
          const double nrm = std::sqrt(std::max(0.0, rr2));
          conv[(size_t)r] = (nrm < tol_residual) ? 1 : 0;
        }
        evals = e_curr;
        break;
      }

      // Build preconditioned trial vectors (all roots) then pack selected ones.
      guga_cuda_linalg_precond_launch(dR, dDiag, dEigs, n, nroots, denom_tol, dT);

      int packed = 0;
      for (int r = 0; r < nroots; r++) {
        if (conv[(size_t)r]) continue;
        const double rr2 = h_rtr[(size_t)r * (size_t)nroots + (size_t)r];
        if (rr2 <= lindep) continue;
        if (packed != r) {
          const double* src = dT + (size_t)r * (size_t)n;
          double* dst = dT + (size_t)packed * (size_t)n;
          throw_on_cuda_error(
              cudaMemcpy(dst, src, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice), "D2D pack T");
        }
        packed++;
      }
      nraw = packed;

      // Block project out the existing basis: T <- T - V (V^T T).
      // dProj is (max_space_eff x nroots), so it can hold (m x nraw) with ldProj=max_space_eff.
      const int ldProj = max_space_eff;
      gemm(CUBLAS_OP_T, CUBLAS_OP_N, m, nraw, n, &one, dV, n, dT, n, &zero, dProj, ldProj, "gemm(Proj=V^T*T)");
      gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, nraw, m, &minus_one, dV, n, dProj, ldProj, &one, dT, n, "gemm(T-=V*Proj)");

      // Orthonormalize and pack new trial vectors into the first nadd columns of dT.
      int nadd = 0;
      for (int j = 0; j < nraw; j++) {
        if (nadd != j) {
          const double* src = dT + (size_t)j * (size_t)n;
          double* dst = dT + (size_t)nadd * (size_t)n;
          throw_on_cuda_error(
              cudaMemcpy(dst, src, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice), "D2D pack T");
        }
        double* t = dT + (size_t)nadd * (size_t)n;

        // Project out previously accepted new vectors.
        for (int i = 0; i < nadd; i++) {
          const double* ti = dT + (size_t)i * (size_t)n;
          double dot = 0.0;
          throw_on_cublas_error(cublasDdot(cublas.h, n, ti, 1, t, 1, &dot), "cublasDdot(ti,t)");
          const double alpha = -dot;
          throw_on_cublas_error(cublasDaxpy(cublas.h, n, &alpha, ti, 1, t, 1), "cublasDaxpy(t-=dot*ti)");
        }

        double nrm = 0.0;
        throw_on_cublas_error(cublasDnrm2(cublas.h, n, t, 1, &nrm), "cublasDnrm2(t)");
        if ((double)(nrm * nrm) <= lindep) {
          continue;
        }

        const double inv = 1.0 / nrm;
        throw_on_cublas_error(cublasDscal(cublas.h, n, &inv, t, 1), "cublasDscal(t)");
        nadd++;
      }

      if (nadd == 0) {
        // No linearly independent correction vectors after projection; fall back
        // to residual-only convergence and stop.
        for (int r = 0; r < nroots; r++) {
          const double rr2 = h_rtr[(size_t)r * (size_t)nroots + (size_t)r];
          const double nrm = std::sqrt(std::max(0.0, rr2));
          conv[(size_t)r] = (nrm < tol_residual) ? 1 : 0;
        }
        evals = e_curr;
        break;
      }

      // Append dT[:,0:nadd] into V and compute corresponding W columns.
      if (m + nadd > max_space_eff) {
        restart_next = true;
        continue;
      }

      throw_on_cuda_error(
          cudaMemcpy(
              dV + (size_t)m * (size_t)n,
              dT,
              (size_t)n * (size_t)nadd * sizeof(double),
              cudaMemcpyDeviceToDevice),
          "D2D append V");

      gemm(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          nadd,
          n,
          &one,
          dA,
          n,
          dV + (size_t)m * (size_t)n,
          n,
          &zero,
          dW + (size_t)m * (size_t)n,
          n,
          "gemm(W_new=A*V_new)");

      m += nadd;
      restart_next = (m + nroots > max_space_eff);
    }

    // Copy eigenvalues and eigenvectors back to host.
    std::vector<double> h_eigs((size_t)max_space_eff, 0.0);
    throw_on_cuda_error(cudaMemcpy(h_eigs.data(), dEigs, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost), "D2H eigs");
    for (int r = 0; r < nroots; r++) {
      evals[(size_t)r] = h_eigs[(size_t)r];
    }

    std::vector<double> hX((size_t)n * (size_t)nroots);
    throw_on_cuda_error(cudaMemcpy(hX.data(), dX, x_bytes, cudaMemcpyDeviceToHost), "D2H X");

    py::array_t<bool> conv_out({nroots});
    py::array_t<double> e_out({nroots});
    py::array_t<double> x_out({nroots, n});

    auto conv_buf = conv_out.mutable_unchecked<1>();
    auto e_buf = e_out.mutable_unchecked<1>();
    auto x_buf = x_out.mutable_unchecked<2>();
    for (int r = 0; r < nroots; r++) {
      conv_buf(r) = conv[(size_t)r] != 0;
      e_buf(r) = evals[(size_t)r];
      for (int i = 0; i < n; i++) {
        x_buf(r, i) = hX[(size_t)i + (size_t)r * (size_t)n];
      }
    }

    return py::make_tuple(conv_out, e_out, x_out, niter);
  }
};

py::tuple davidson_dense_sym(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    int nroots,
    int max_cycle,
    int max_space,
    double tol,
    double lindep,
    double denom_tol) {
  auto buf = a.request();
  if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
    throw std::invalid_argument("a must be a square 2D array");
  }
  const int n = (int)buf.shape[0];
  DenseSymDavidsonWorkspace ws(n, nroots, max_space);
  ws.set_matrix(a);
  return ws.solve(max_cycle, tol, lindep, denom_tol);
}

}  // namespace

PYBIND11_MODULE(_guga_cuda_linalg_ext, m) {
  m.doc() = "CUDA linalg baseline for a GPU-native Davidson solver (cuBLAS + cuSOLVER).";

  py::class_<DenseSymDavidsonWorkspace>(m, "DenseSymDavidsonWorkspace")
      .def(py::init<int, int, int>(), py::arg("n"), py::arg("nroots") = 1, py::arg("max_space") = 12)
      .def("cublas_emulation_info", &DenseSymDavidsonWorkspace::get_cublas_emulation_info)
      .def("set_cublas_emulation_strategy", &DenseSymDavidsonWorkspace::set_cublas_emulation_strategy, py::arg("strategy"))
      .def(
          "set_cublas_emulation_special_values_support",
          &DenseSymDavidsonWorkspace::set_cublas_emulation_special_values_support,
          py::arg("mask"))
      .def(
          "set_cublas_fixed_point_mantissa_control",
          &DenseSymDavidsonWorkspace::set_cublas_fixed_point_mantissa_control,
          py::arg("control"))
      .def(
          "set_cublas_fixed_point_max_mantissa_bits",
          &DenseSymDavidsonWorkspace::set_cublas_fixed_point_max_mantissa_bits,
          py::arg("max_bits"))
      .def(
          "set_cublas_fixed_point_mantissa_bit_offset",
          &DenseSymDavidsonWorkspace::set_cublas_fixed_point_mantissa_bit_offset,
          py::arg("bit_offset"))
      .def("gemm_backend", &DenseSymDavidsonWorkspace::get_gemm_backend)
      .def("set_gemm_backend", &DenseSymDavidsonWorkspace::set_gemm_backend, py::arg("backend"))
      .def("gemm_algo", &DenseSymDavidsonWorkspace::get_gemm_algo)
      .def("set_gemm_algo", &DenseSymDavidsonWorkspace::set_gemm_algo, py::arg("algo"))
      .def("set_cublas_math_mode", &DenseSymDavidsonWorkspace::set_cublas_math_mode, py::arg("mode"))
      .def("cublas_workspace_bytes", &DenseSymDavidsonWorkspace::get_cublas_workspace_bytes)
      .def("set_cublas_workspace_bytes", &DenseSymDavidsonWorkspace::set_cublas_workspace_bytes, py::arg("bytes"))
      .def("set_matrix", &DenseSymDavidsonWorkspace::set_matrix, py::arg("a"))
      .def(
          "solve",
          &DenseSymDavidsonWorkspace::solve,
          py::arg("max_cycle") = 50,
          py::arg("tol") = 1e-10,
          py::arg("lindep") = 1e-14,
          py::arg("denom_tol") = 1e-12)
      .def_property_readonly("n", [](const DenseSymDavidsonWorkspace& self) { return self.n; })
      .def_property_readonly("nroots", [](const DenseSymDavidsonWorkspace& self) { return self.nroots; })
      .def_property_readonly("max_space_eff", [](const DenseSymDavidsonWorkspace& self) { return self.max_space_eff; });

  m.def("device_info", &device_info);
  m.def("mem_info", &mem_info);
  m.def("cublas_emulation_info", &cublas_emulation_info);
  m.def("eigh_sym", &eigh_sym, "Symmetric eigensolve via cuSOLVER (FP64).");
  m.def("gemm", &gemm, "Row-major GEMM via cuBLAS (FP64).");
  m.def(
      "davidson_dense_sym",
      &davidson_dense_sym,
      py::arg("a"),
      py::arg("nroots") = 1,
      py::arg("max_cycle") = 50,
      py::arg("max_space") = 12,
      py::arg("tol") = 1e-10,
      py::arg("lindep") = 1e-14,
      py::arg("denom_tol") = 1e-12,
      "Dense symmetric Davidson prototype (GPU-resident; validation harness).");
}
