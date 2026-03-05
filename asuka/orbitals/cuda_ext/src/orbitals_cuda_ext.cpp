#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "orbitals_cuda_kernels_api.h"

namespace py = pybind11;

namespace {

struct CudaArrayView {
  void* ptr = nullptr;
  bool read_only = false;
  std::string typestr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;  // empty means "not provided"
};

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void throw_on_cublas_error(cublasStatus_t stat, const char* what) {
  if (stat == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string((int)stat));
}

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
    return out;
  }
  validate_cuda_device_pointer(out.ptr, name);
  return out;
}

inline std::string normalize_typestr(std::string t) {
  if (t.size() == 3 && (t[0] == '=' || t[0] == '|')) t[0] = '<';
  return t;
}

inline void require_typestr(const CudaArrayView& a, const char* name, const char* expected) {
  const std::string t = normalize_typestr(a.typestr);
  if (t == expected) return;
  throw std::invalid_argument(std::string(name) + " must have typestr " + expected + " (got " + a.typestr + ")");
}

inline bool is_c_contiguous(const CudaArrayView& a, int64_t itemsize) {
  if (a.strides_bytes.empty()) return true;
  const int nd = (int)a.shape.size();
  if ((int)a.strides_bytes.size() != nd) return false;
  int64_t stride = itemsize;
  for (int i = nd - 1; i >= 0; --i) {
    if (a.shape[i] == 0) return true;
    if (a.strides_bytes[i] != stride) return false;
    stride *= a.shape[i];
  }
  return true;
}

inline void require_c_contiguous(const CudaArrayView& a, const char* name, int64_t itemsize) {
  if (is_c_contiguous(a, itemsize)) return;
  throw std::invalid_argument(std::string(name) + " must be C-contiguous");
}

struct IngredientsWorkspace {
  int32_t max_npt = 0;
  int32_t max_nocc = 0;
  int32_t max_ncas = 0;
  int32_t lmax = 10;

  double* dPsi = nullptr;       // (max_npt,max_nocc)
  double* dGrad = nullptr;      // (max_npt,max_nocc,3)
  double* dLapl = nullptr;      // (max_npt,max_nocc)
  double* dPair = nullptr;      // (max_npt,max_ncas^2)
  double* dG = nullptr;         // (max_npt,max_ncas^2)
  double* dGy = nullptr;        // (max_npt,max_ncas^2)

  double* dRhoCore = nullptr;      // (max_npt,)
  double* dRhoAct = nullptr;       // (max_npt,)
  double* dRhoCoreGrad = nullptr;  // (max_npt,3)
  double* dRhoActGrad = nullptr;   // (max_npt,3)
  double* dRhoCoreLapl = nullptr;  // (max_npt,)
  double* dRhoActLapl = nullptr;   // (max_npt,)

  cublasHandle_t cublas = nullptr;

  IngredientsWorkspace(int32_t max_npt_, int32_t max_nocc_, int32_t max_ncas_, int32_t lmax_ = 10)
      : max_npt(std::max<int32_t>(0, max_npt_)),
        max_nocc(std::max<int32_t>(0, max_nocc_)),
        max_ncas(std::max<int32_t>(0, max_ncas_)),
        lmax(std::max<int32_t>(0, lmax_)) {
    if (max_npt <= 0) throw std::invalid_argument("max_npt must be > 0");
    if (max_nocc <= 0) throw std::invalid_argument("max_nocc must be > 0");
    if (max_ncas < 0) throw std::invalid_argument("max_ncas must be >= 0");

    const int64_t npt = max_npt;
    const int64_t nocc = max_nocc;
    const int64_t ncas = max_ncas;
    const int64_t n2 = ncas * ncas;

    auto alloc = [&](void** p, size_t bytes, const char* what) {
      if (bytes == 0) {
        *p = nullptr;
        return;
      }
      throw_on_cuda_error(cudaMalloc(p, bytes), what);
    };

    alloc(reinterpret_cast<void**>(&dPsi), size_t(npt * nocc * sizeof(double)), "cudaMalloc(dPsi)");
    alloc(reinterpret_cast<void**>(&dGrad), size_t(npt * nocc * 3 * sizeof(double)), "cudaMalloc(dGrad)");
    alloc(reinterpret_cast<void**>(&dLapl), size_t(npt * nocc * sizeof(double)), "cudaMalloc(dLapl)");

    alloc(reinterpret_cast<void**>(&dPair), size_t(npt * n2 * sizeof(double)), "cudaMalloc(dPair)");
    alloc(reinterpret_cast<void**>(&dG), size_t(npt * n2 * sizeof(double)), "cudaMalloc(dG)");
    alloc(reinterpret_cast<void**>(&dGy), size_t(npt * n2 * sizeof(double)), "cudaMalloc(dGy)");

    alloc(reinterpret_cast<void**>(&dRhoCore), size_t(npt * sizeof(double)), "cudaMalloc(dRhoCore)");
    alloc(reinterpret_cast<void**>(&dRhoAct), size_t(npt * sizeof(double)), "cudaMalloc(dRhoAct)");
    alloc(reinterpret_cast<void**>(&dRhoCoreGrad), size_t(npt * 3 * sizeof(double)), "cudaMalloc(dRhoCoreGrad)");
    alloc(reinterpret_cast<void**>(&dRhoActGrad), size_t(npt * 3 * sizeof(double)), "cudaMalloc(dRhoActGrad)");
    alloc(reinterpret_cast<void**>(&dRhoCoreLapl), size_t(npt * sizeof(double)), "cudaMalloc(dRhoCoreLapl)");
    alloc(reinterpret_cast<void**>(&dRhoActLapl), size_t(npt * sizeof(double)), "cudaMalloc(dRhoActLapl)");

    throw_on_cublas_error(cublasCreate(&cublas), "cublasCreate");
  }

  IngredientsWorkspace(const IngredientsWorkspace&) = delete;
  IngredientsWorkspace& operator=(const IngredientsWorkspace&) = delete;

  ~IngredientsWorkspace() { release(); }

  void release() noexcept {
    if (cublas) {
      cublasDestroy(cublas);
      cublas = nullptr;
    }
    if (dPsi) cudaFree(dPsi);
    if (dGrad) cudaFree(dGrad);
    if (dLapl) cudaFree(dLapl);
    if (dPair) cudaFree(dPair);
    if (dG) cudaFree(dG);
    if (dGy) cudaFree(dGy);
    if (dRhoCore) cudaFree(dRhoCore);
    if (dRhoAct) cudaFree(dRhoAct);
    if (dRhoCoreGrad) cudaFree(dRhoCoreGrad);
    if (dRhoActGrad) cudaFree(dRhoActGrad);
    if (dRhoCoreLapl) cudaFree(dRhoCoreLapl);
    if (dRhoActLapl) cudaFree(dRhoActLapl);

    dPsi = dGrad = dLapl = dPair = dG = dGy = nullptr;
    dRhoCore = dRhoAct = dRhoCoreGrad = dRhoActGrad = dRhoCoreLapl = dRhoActLapl = nullptr;
  }
};

struct BeckeGridWorkspace {
  int32_t max_nloc = 0;
  int32_t max_natm = 0;

  double* dPtsLocal = nullptr;  // (max_nloc,3)
  double* dWBase = nullptr;     // (max_nloc,)
  double* dRA = nullptr;        // (max_nloc,max_natm)
  double* dWRaw = nullptr;      // (max_nloc,max_natm)
  double* dWAtom = nullptr;     // (max_nloc,)

  BeckeGridWorkspace(int32_t max_nloc_, int32_t max_natm_)
      : max_nloc(std::max<int32_t>(0, max_nloc_)), max_natm(std::max<int32_t>(0, max_natm_)) {
    if (max_nloc <= 0) throw std::invalid_argument("max_nloc must be > 0");
    if (max_natm <= 0) throw std::invalid_argument("max_natm must be > 0");

    auto alloc = [&](void** p, size_t bytes, const char* what) {
      if (bytes == 0) {
        *p = nullptr;
        return;
      }
      throw_on_cuda_error(cudaMalloc(p, bytes), what);
    };

    alloc(reinterpret_cast<void**>(&dPtsLocal), size_t(max_nloc) * 3u * sizeof(double), "cudaMalloc(dPtsLocal)");
    alloc(reinterpret_cast<void**>(&dWBase), size_t(max_nloc) * sizeof(double), "cudaMalloc(dWBase)");
    alloc(
        reinterpret_cast<void**>(&dRA),
        size_t(max_nloc) * size_t(max_natm) * sizeof(double),
        "cudaMalloc(dRA)");
    alloc(
        reinterpret_cast<void**>(&dWRaw),
        size_t(max_nloc) * size_t(max_natm) * sizeof(double),
        "cudaMalloc(dWRaw)");
    alloc(reinterpret_cast<void**>(&dWAtom), size_t(max_nloc) * sizeof(double), "cudaMalloc(dWAtom)");
  }

  BeckeGridWorkspace(const BeckeGridWorkspace&) = delete;
  BeckeGridWorkspace& operator=(const BeckeGridWorkspace&) = delete;

  ~BeckeGridWorkspace() { release(); }

  void release() noexcept {
    if (dPtsLocal) cudaFree(dPtsLocal);
    if (dWBase) cudaFree(dWBase);
    if (dRA) cudaFree(dRA);
    if (dWRaw) cudaFree(dWRaw);
    if (dWAtom) cudaFree(dWAtom);
    dPtsLocal = dWBase = dRA = dWRaw = dWAtom = nullptr;
  }
};

inline void dgemm_pair_dm2T(
    cublasHandle_t h,
    const double* dm2,    // (n2,n2) row-major (interpreted as col-major dm2^T)
    const double* pair,   // (npt,n2) row-major (col-major n2 x npt)
    double* out_g,        // (npt,n2) row-major (col-major n2 x npt)
    int32_t npt,
    int32_t n2) {
  // We want: g = pair @ dm2^T   (row-major).
  //
  // Treat row-major memory as column-major transposes:
  // - dm2 row-major corresponds to A_col = dm2^T.
  // - pair row-major corresponds to B_col = pair^T with shape (n2,npt).
  // - out_g row-major corresponds to C_col = g^T with shape (n2,npt).
  //
  // Then g = pair @ dm2^T  <=>  g^T = dm2 @ pair^T.
  // In column-major: C = (A_col^T) * B, i.e. op(A)=T.
  const int m = int(n2);
  const int n = int(npt);
  const int k = int(n2);
  const double alpha = 1.0;
  const double beta = 0.0;
  throw_on_cublas_error(
      cublasDgemm(
          h,
          CUBLAS_OP_T,  // (dm2^T)^T = dm2
          CUBLAS_OP_N,
          m,
          n,
          k,
          &alpha,
          dm2,
          m,
          pair,
          m,
          &beta,
          out_g,
          m),
      "cublasDgemm(dm2 * pair^T)");
}

void eval_becke_atom_block_f64_inplace_device(
    const py::object& center_xyz_obj,
    const py::object& radial_r_obj,
    const py::object& radial_wr_obj,
    const py::object& angular_dirs_obj,
    const py::object& angular_w_obj,
    const py::object& atom_coords_obj,
    const py::object& RAB_obj,
    int32_t atom_index,
    int32_t becke_n,
    const py::object& pts_out_obj,
    const py::object& w_out_obj,
    BeckeGridWorkspace& ws,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  if (becke_n < 0) throw std::invalid_argument("becke_n must be >= 0");

  const CudaArrayView center_xyz = cuda_array_view_from_object(center_xyz_obj, "center_xyz");
  const CudaArrayView radial_r = cuda_array_view_from_object(radial_r_obj, "radial_r");
  const CudaArrayView radial_wr = cuda_array_view_from_object(radial_wr_obj, "radial_wr");
  const CudaArrayView angular_dirs = cuda_array_view_from_object(angular_dirs_obj, "angular_dirs");
  const CudaArrayView angular_w = cuda_array_view_from_object(angular_w_obj, "angular_w");
  const CudaArrayView atom_coords = cuda_array_view_from_object(atom_coords_obj, "atom_coords");
  const CudaArrayView RAB = cuda_array_view_from_object(RAB_obj, "RAB");
  const CudaArrayView pts_out = cuda_array_view_from_object(pts_out_obj, "pts_out");
  const CudaArrayView w_out = cuda_array_view_from_object(w_out_obj, "w_out");

  require_typestr(center_xyz, "center_xyz", "<f8");
  require_typestr(radial_r, "radial_r", "<f8");
  require_typestr(radial_wr, "radial_wr", "<f8");
  require_typestr(angular_dirs, "angular_dirs", "<f8");
  require_typestr(angular_w, "angular_w", "<f8");
  require_typestr(atom_coords, "atom_coords", "<f8");
  require_typestr(RAB, "RAB", "<f8");
  require_typestr(pts_out, "pts_out", "<f8");
  require_typestr(w_out, "w_out", "<f8");

  require_c_contiguous(center_xyz, "center_xyz", 8);
  require_c_contiguous(radial_r, "radial_r", 8);
  require_c_contiguous(radial_wr, "radial_wr", 8);
  require_c_contiguous(angular_dirs, "angular_dirs", 8);
  require_c_contiguous(angular_w, "angular_w", 8);
  require_c_contiguous(atom_coords, "atom_coords", 8);
  require_c_contiguous(RAB, "RAB", 8);
  require_c_contiguous(pts_out, "pts_out", 8);
  require_c_contiguous(w_out, "w_out", 8);

  if (center_xyz.shape.size() != 1 || center_xyz.shape[0] != 3) {
    throw std::invalid_argument("center_xyz must have shape (3,)");
  }
  if (radial_r.shape.size() != 1 || radial_wr.shape.size() != 1 || radial_r.shape[0] != radial_wr.shape[0]) {
    throw std::invalid_argument("radial_r/radial_wr must have shape (nrad,)");
  }
  if (angular_dirs.shape.size() != 2 || angular_dirs.shape[1] != 3) {
    throw std::invalid_argument("angular_dirs must have shape (nang,3)");
  }
  if (angular_w.shape.size() != 1 || angular_w.shape[0] != angular_dirs.shape[0]) {
    throw std::invalid_argument("angular_w must have shape (nang,)");
  }
  if (atom_coords.shape.size() != 2 || atom_coords.shape[1] != 3) {
    throw std::invalid_argument("atom_coords must have shape (natm,3)");
  }

  const int32_t nrad = int32_t(radial_r.shape[0]);
  const int32_t nang = int32_t(angular_dirs.shape[0]);
  const int32_t nloc = int32_t(nrad * nang);
  const int32_t natm = int32_t(atom_coords.shape[0]);
  if (nrad <= 0 || nang <= 0 || nloc <= 0) throw std::invalid_argument("nrad/nang must be > 0");
  if (natm <= 0) throw std::invalid_argument("natm must be > 0");
  if (atom_index < 0 || atom_index >= natm) throw std::invalid_argument("atom_index out of range");

  if (RAB.shape.size() != 2 || RAB.shape[0] != natm || RAB.shape[1] != natm) {
    throw std::invalid_argument("RAB must have shape (natm,natm)");
  }
  if (pts_out.shape.size() != 2 || pts_out.shape[0] != nloc || pts_out.shape[1] != 3) {
    throw std::invalid_argument("pts_out must have shape (nloc,3)");
  }
  if (w_out.shape.size() != 1 || w_out.shape[0] != nloc) {
    throw std::invalid_argument("w_out must have shape (nloc,)");
  }

  if (nloc > ws.max_nloc || natm > ws.max_natm) {
    throw std::invalid_argument("becke workspace capacity exceeded");
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  orbitals_build_atom_centered_points_weights_f64(
      static_cast<const double*>(center_xyz.ptr),
      static_cast<const double*>(radial_r.ptr),
      static_cast<const double*>(radial_wr.ptr),
      nrad,
      static_cast<const double*>(angular_dirs.ptr),
      static_cast<const double*>(angular_w.ptr),
      nang,
      ws.dPtsLocal,
      ws.dWBase,
      stream,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "orbitals_build_atom_centered_points_weights_f64 kernel launch");

  orbitals_becke_partition_atom_block_f64(
      ws.dPtsLocal,
      ws.dWBase,
      nloc,
      static_cast<const double*>(atom_coords.ptr),
      static_cast<const double*>(RAB.ptr),
      natm,
      atom_index,
      becke_n,
      ws.dRA,
      ws.dWRaw,
      ws.dWAtom,
      stream,
      threads);
  throw_on_cuda_error(cudaGetLastError(), "orbitals_becke_partition_atom_block_f64 kernel launch");

  if (ws.dPtsLocal != static_cast<double*>(pts_out.ptr)) {
    throw_on_cuda_error(
        cudaMemcpyAsync(
            pts_out.ptr,
            ws.dPtsLocal,
            size_t(nloc) * 3u * sizeof(double),
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(pts_out <- dPtsLocal)");
  }
  if (ws.dWAtom != static_cast<double*>(w_out.ptr)) {
    throw_on_cuda_error(
        cudaMemcpyAsync(
            w_out.ptr,
            ws.dWAtom,
            size_t(nloc) * sizeof(double),
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(w_out <- dWAtom)");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

void eval_aos_cart_value_f64_inplace_device(
    const py::object& shell_cxyz_obj,
    const py::object& shell_prim_start_obj,
    const py::object& shell_nprim_obj,
    const py::object& shell_l_obj,
    const py::object& shell_ao_start_obj,
    const py::object& prim_exp_obj,
    const py::object& prim_coef_obj,
    const py::object& points_obj,
    const py::object& ao_obj,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  (void)threads;

  const CudaArrayView shell_cxyz = cuda_array_view_from_object(shell_cxyz_obj, "shell_cxyz");
  const CudaArrayView shell_prim_start = cuda_array_view_from_object(shell_prim_start_obj, "shell_prim_start");
  const CudaArrayView shell_nprim = cuda_array_view_from_object(shell_nprim_obj, "shell_nprim");
  const CudaArrayView shell_l = cuda_array_view_from_object(shell_l_obj, "shell_l");
  const CudaArrayView shell_ao_start = cuda_array_view_from_object(shell_ao_start_obj, "shell_ao_start");
  const CudaArrayView prim_exp = cuda_array_view_from_object(prim_exp_obj, "prim_exp");
  const CudaArrayView prim_coef = cuda_array_view_from_object(prim_coef_obj, "prim_coef");
  const CudaArrayView points = cuda_array_view_from_object(points_obj, "points");
  const CudaArrayView ao = cuda_array_view_from_object(ao_obj, "ao");

  const int64_t nshell = shell_l.shape.empty() ? 0 : shell_l.shape[0];
  if (nshell <= 0) throw std::invalid_argument("shell arrays must be non-empty");

  require_typestr(shell_cxyz, "shell_cxyz", "<f8");
  require_typestr(shell_prim_start, "shell_prim_start", "<i4");
  require_typestr(shell_nprim, "shell_nprim", "<i4");
  require_typestr(shell_l, "shell_l", "<i4");
  require_typestr(shell_ao_start, "shell_ao_start", "<i4");
  require_typestr(prim_exp, "prim_exp", "<f8");
  require_typestr(prim_coef, "prim_coef", "<f8");
  require_typestr(points, "points", "<f8");
  require_typestr(ao, "ao", "<f8");

  require_c_contiguous(shell_cxyz, "shell_cxyz", 8);
  require_c_contiguous(shell_prim_start, "shell_prim_start", 4);
  require_c_contiguous(shell_nprim, "shell_nprim", 4);
  require_c_contiguous(shell_l, "shell_l", 4);
  require_c_contiguous(shell_ao_start, "shell_ao_start", 4);
  require_c_contiguous(prim_exp, "prim_exp", 8);
  require_c_contiguous(prim_coef, "prim_coef", 8);
  require_c_contiguous(points, "points", 8);
  require_c_contiguous(ao, "ao", 8);

  if (shell_cxyz.shape.size() != 2 || shell_cxyz.shape[0] != nshell || shell_cxyz.shape[1] != 3) {
    throw std::invalid_argument("shell_cxyz must have shape (nshell,3)");
  }
  if (shell_prim_start.shape.size() != 1 || shell_prim_start.shape[0] != nshell) {
    throw std::invalid_argument("shell_prim_start must have shape (nshell,)");
  }
  if (shell_nprim.shape.size() != 1 || shell_nprim.shape[0] != nshell) {
    throw std::invalid_argument("shell_nprim must have shape (nshell,)");
  }
  if (shell_l.shape.size() != 1 || shell_l.shape[0] != nshell) {
    throw std::invalid_argument("shell_l must have shape (nshell,)");
  }
  if (shell_ao_start.shape.size() != 1 || shell_ao_start.shape[0] != nshell) {
    throw std::invalid_argument("shell_ao_start must have shape (nshell,)");
  }

  if (prim_exp.shape.size() != 1) throw std::invalid_argument("prim_exp must be 1D");
  if (prim_coef.shape.size() != 1) throw std::invalid_argument("prim_coef must be 1D");
  if (prim_coef.shape[0] != prim_exp.shape[0]) throw std::invalid_argument("prim_exp/prim_coef length mismatch");

  if (points.shape.size() != 2 || points.shape[1] != 3) {
    throw std::invalid_argument("points must have shape (npt,3)");
  }
  const int32_t npt = int32_t(points.shape[0]);

  if (ao.read_only) throw std::invalid_argument("ao output must be writable");
  if (ao.shape.size() != 2) throw std::invalid_argument("ao must be 2D (npt,nao)");
  if (ao.shape[0] != npt) throw std::invalid_argument("ao must have shape (npt,nao)");
  const int32_t nao = int32_t(ao.shape[1]);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (npt > 0 && nao > 0) {
    orbitals_eval_aos_cart_value_f64(
        static_cast<const double*>(shell_cxyz.ptr),
        static_cast<const int32_t*>(shell_prim_start.ptr),
        static_cast<const int32_t*>(shell_nprim.ptr),
        static_cast<const int32_t*>(shell_l.ptr),
        static_cast<const int32_t*>(shell_ao_start.ptr),
        static_cast<const double*>(prim_exp.ptr),
        static_cast<const double*>(prim_coef.ptr),
        int32_t(nshell),
        nao,
        static_cast<const double*>(points.ptr),
        npt,
        static_cast<double*>(ao.ptr),
        stream);
    throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_aos_cart_value_f64 kernel launch");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

void eval_density_otpd_f64_inplace_device(
    const py::object& shell_cxyz_obj,
    const py::object& shell_prim_start_obj,
    const py::object& shell_nprim_obj,
    const py::object& shell_l_obj,
    const py::object& shell_ao_start_obj,
    const py::object& prim_exp_obj,
    const py::object& prim_coef_obj,
    const py::object& C_occ_obj,
    const py::object& points_obj,
    const py::object& dm1_obj,
    const py::object& dm2_obj,
    const py::object& rho_obj,
    const py::object& pi_obj,
    const py::object& rho_grad_obj,
    const py::object& pi_grad_obj,
    const py::object& tau_obj,
    const py::object& rho_lapl_obj,
    const py::object& pi_lapl_obj,
    int32_t ncore,
    int32_t ncas,
    int32_t deriv,
    bool compute_tau,
    bool compute_rho_laplacian,
    bool compute_pi_laplacian,
    bool symmetrize_dm2,
    IngredientsWorkspace& ws,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  (void)symmetrize_dm2;

  if (ncore < 0 || ncas < 0) throw std::invalid_argument("ncore/ncas must be >= 0");
  if (deriv < 0) throw std::invalid_argument("deriv must be >= 0");

  const bool need_grad = (deriv >= 1) || compute_tau || compute_rho_laplacian || compute_pi_laplacian;
  const bool need_lapl = compute_rho_laplacian || compute_pi_laplacian;

  const CudaArrayView shell_cxyz = cuda_array_view_from_object(shell_cxyz_obj, "shell_cxyz");
  const CudaArrayView shell_prim_start = cuda_array_view_from_object(shell_prim_start_obj, "shell_prim_start");
  const CudaArrayView shell_nprim = cuda_array_view_from_object(shell_nprim_obj, "shell_nprim");
  const CudaArrayView shell_l = cuda_array_view_from_object(shell_l_obj, "shell_l");
  const CudaArrayView shell_ao_start = cuda_array_view_from_object(shell_ao_start_obj, "shell_ao_start");
  const CudaArrayView prim_exp = cuda_array_view_from_object(prim_exp_obj, "prim_exp");
  const CudaArrayView prim_coef = cuda_array_view_from_object(prim_coef_obj, "prim_coef");
  const CudaArrayView C_occ = cuda_array_view_from_object(C_occ_obj, "C_occ");
  const CudaArrayView points = cuda_array_view_from_object(points_obj, "points");
  const CudaArrayView dm1 = cuda_array_view_from_object(dm1_obj, "dm1");
  const CudaArrayView dm2 = cuda_array_view_from_object(dm2_obj, "dm2");

  const CudaArrayView rho = cuda_array_view_from_object(rho_obj, "rho");
  const CudaArrayView pi = cuda_array_view_from_object(pi_obj, "pi");

  const int64_t nshell = shell_l.shape.empty() ? 0 : shell_l.shape[0];
  if (nshell <= 0) throw std::invalid_argument("shell arrays must be non-empty");

  require_typestr(shell_cxyz, "shell_cxyz", "<f8");
  require_typestr(shell_prim_start, "shell_prim_start", "<i4");
  require_typestr(shell_nprim, "shell_nprim", "<i4");
  require_typestr(shell_l, "shell_l", "<i4");
  require_typestr(shell_ao_start, "shell_ao_start", "<i4");
  require_typestr(prim_exp, "prim_exp", "<f8");
  require_typestr(prim_coef, "prim_coef", "<f8");
  require_typestr(C_occ, "C_occ", "<f8");
  require_typestr(points, "points", "<f8");
  require_typestr(dm1, "dm1", "<f8");
  require_typestr(dm2, "dm2", "<f8");
  require_typestr(rho, "rho", "<f8");
  require_typestr(pi, "pi", "<f8");

  require_c_contiguous(shell_cxyz, "shell_cxyz", 8);
  require_c_contiguous(shell_prim_start, "shell_prim_start", 4);
  require_c_contiguous(shell_nprim, "shell_nprim", 4);
  require_c_contiguous(shell_l, "shell_l", 4);
  require_c_contiguous(shell_ao_start, "shell_ao_start", 4);
  require_c_contiguous(prim_exp, "prim_exp", 8);
  require_c_contiguous(prim_coef, "prim_coef", 8);
  require_c_contiguous(C_occ, "C_occ", 8);
  require_c_contiguous(points, "points", 8);
  require_c_contiguous(dm1, "dm1", 8);
  require_c_contiguous(dm2, "dm2", 8);
  require_c_contiguous(rho, "rho", 8);
  require_c_contiguous(pi, "pi", 8);

  if (shell_cxyz.shape.size() != 2 || shell_cxyz.shape[0] != nshell || shell_cxyz.shape[1] != 3) {
    throw std::invalid_argument("shell_cxyz must have shape (nshell,3)");
  }
  for (const auto* name : {"shell_prim_start", "shell_nprim", "shell_l", "shell_ao_start"}) {
    (void)name;
  }
  if (shell_prim_start.shape.size() != 1 || shell_prim_start.shape[0] != nshell) {
    throw std::invalid_argument("shell_prim_start must have shape (nshell,)");
  }
  if (shell_nprim.shape.size() != 1 || shell_nprim.shape[0] != nshell) {
    throw std::invalid_argument("shell_nprim must have shape (nshell,)");
  }
  if (shell_l.shape.size() != 1 || shell_l.shape[0] != nshell) {
    throw std::invalid_argument("shell_l must have shape (nshell,)");
  }
  if (shell_ao_start.shape.size() != 1 || shell_ao_start.shape[0] != nshell) {
    throw std::invalid_argument("shell_ao_start must have shape (nshell,)");
  }

  if (C_occ.shape.size() != 2) throw std::invalid_argument("C_occ must be 2D (nao,nocc)");
  const int32_t nao = int32_t(C_occ.shape[0]);
  const int32_t nocc = int32_t(C_occ.shape[1]);
  if (nocc != ncore + ncas) {
    throw std::invalid_argument("C_occ must have nocc == ncore + ncas");
  }

  if (points.shape.size() != 2 || points.shape[1] != 3) {
    throw std::invalid_argument("points must have shape (npt,3)");
  }
  const int32_t npt = int32_t(points.shape[0]);

  if (dm1.shape.size() != 2 || dm1.shape[0] != ncas || dm1.shape[1] != ncas) {
    throw std::invalid_argument("dm1 must have shape (ncas,ncas)");
  }
  const int32_t n2 = int32_t(ncas * ncas);
  if (dm2.shape.size() != 2 || dm2.shape[0] != n2 || dm2.shape[1] != n2) {
    throw std::invalid_argument("dm2 must have shape (ncas^2,ncas^2)");
  }

  if (rho.shape.size() != 1 || rho.shape[0] != npt) throw std::invalid_argument("rho must have shape (npt,)");
  if (pi.shape.size() != 1 || pi.shape[0] != npt) throw std::invalid_argument("pi must have shape (npt,)");

  if (npt > ws.max_npt || nocc > ws.max_nocc || ncas > ws.max_ncas) {
    throw std::invalid_argument("workspace capacity exceeded");
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  throw_on_cublas_error(cublasSetStream(ws.cublas, stream), "cublasSetStream");

  // --- MO evaluation ---
  if (npt > 0 && nocc > 0) {
    if (need_lapl) {
      orbitals_eval_mos_cart_value_grad_lapl_f64(
          static_cast<const double*>(shell_cxyz.ptr),
          static_cast<const int32_t*>(shell_prim_start.ptr),
          static_cast<const int32_t*>(shell_nprim.ptr),
          static_cast<const int32_t*>(shell_l.ptr),
          static_cast<const int32_t*>(shell_ao_start.ptr),
          static_cast<const double*>(prim_exp.ptr),
          static_cast<const double*>(prim_coef.ptr),
          int32_t(nshell),
          static_cast<const double*>(C_occ.ptr),
          nao,
          nocc,
          static_cast<const double*>(points.ptr),
          npt,
          ws.dPsi,
          ws.dGrad,
          ws.dLapl,
          stream);
    } else if (need_grad) {
      orbitals_eval_mos_cart_value_grad_f64(
          static_cast<const double*>(shell_cxyz.ptr),
          static_cast<const int32_t*>(shell_prim_start.ptr),
          static_cast<const int32_t*>(shell_nprim.ptr),
          static_cast<const int32_t*>(shell_l.ptr),
          static_cast<const int32_t*>(shell_ao_start.ptr),
          static_cast<const double*>(prim_exp.ptr),
          static_cast<const double*>(prim_coef.ptr),
          int32_t(nshell),
          static_cast<const double*>(C_occ.ptr),
          nao,
          nocc,
          static_cast<const double*>(points.ptr),
          npt,
          ws.dPsi,
          ws.dGrad,
          stream);
    } else {
      orbitals_eval_mos_cart_value_f64(
          static_cast<const double*>(shell_cxyz.ptr),
          static_cast<const int32_t*>(shell_prim_start.ptr),
          static_cast<const int32_t*>(shell_nprim.ptr),
          static_cast<const int32_t*>(shell_l.ptr),
          static_cast<const int32_t*>(shell_ao_start.ptr),
          static_cast<const double*>(prim_exp.ptr),
          static_cast<const double*>(prim_coef.ptr),
          int32_t(nshell),
          static_cast<const double*>(C_occ.ptr),
          nao,
          nocc,
          static_cast<const double*>(points.ptr),
          npt,
          ws.dPsi,
          stream);
    }
  }
  throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_mos_cart_* kernel launch");

  // --- rho / tau / rho_lapl and parts ---
  {
    double* rho_grad_ptr = nullptr;
    if (deriv >= 1) {
      const CudaArrayView rho_grad = cuda_array_view_from_object(rho_grad_obj, "rho_grad");
      require_typestr(rho_grad, "rho_grad", "<f8");
      require_c_contiguous(rho_grad, "rho_grad", 8);
      if (rho_grad.shape.size() != 2 || rho_grad.shape[0] != npt || rho_grad.shape[1] != 3) {
        throw std::invalid_argument("rho_grad must have shape (npt,3)");
      }
      rho_grad_ptr = static_cast<double*>(rho_grad.ptr);
    }

    double* tau_ptr = nullptr;
    if (compute_tau) {
      const CudaArrayView tau_view = cuda_array_view_from_object(tau_obj, "tau");
      require_typestr(tau_view, "tau", "<f8");
      require_c_contiguous(tau_view, "tau", 8);
      if (tau_view.shape.size() != 1 || tau_view.shape[0] != npt) {
        throw std::invalid_argument("tau must have shape (npt,)");
      }
      tau_ptr = static_cast<double*>(tau_view.ptr);
    }

    double* rho_lapl_ptr = nullptr;
    if (compute_rho_laplacian) {
      const CudaArrayView rho_lapl_view = cuda_array_view_from_object(rho_lapl_obj, "rho_lapl");
      require_typestr(rho_lapl_view, "rho_lapl", "<f8");
      require_c_contiguous(rho_lapl_view, "rho_lapl", 8);
      if (rho_lapl_view.shape.size() != 1 || rho_lapl_view.shape[0] != npt) {
        throw std::invalid_argument("rho_lapl must have shape (npt,)");
      }
      rho_lapl_ptr = static_cast<double*>(rho_lapl_view.ptr);
    }

    orbitals_eval_rho_parts_f64(
        ws.dPsi,
        need_grad ? ws.dGrad : nullptr,
        need_lapl ? ws.dLapl : nullptr,
        static_cast<const double*>(dm1.ptr),
        ncore,
        ncas,
        npt,
        nocc,
        need_grad ? 1 : 0,
        need_lapl ? 1 : 0,
        compute_tau ? 1 : 0,
        static_cast<double*>(rho.ptr),
        rho_grad_ptr,
        tau_ptr,
        rho_lapl_ptr,
        ws.dRhoCore,
        ws.dRhoAct,
        need_grad ? ws.dRhoCoreGrad : nullptr,
        need_grad ? ws.dRhoActGrad : nullptr,
        need_lapl ? ws.dRhoCoreLapl : nullptr,
        need_lapl ? ws.dRhoActLapl : nullptr,
        stream,
        threads);
  }
  throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_rho_parts_f64 kernel launch");

  // --- pi ---
  if (ncas > 0) {
    orbitals_build_pair_x_f64(ws.dPsi, ncore, ncas, npt, nocc, ws.dPair, stream, threads);
    throw_on_cuda_error(cudaGetLastError(), "orbitals_build_pair_x_f64 kernel launch");

    if (npt > 0) {
      dgemm_pair_dm2T(ws.cublas, static_cast<const double*>(dm2.ptr), ws.dPair, ws.dG, npt, n2);
    }

    orbitals_eval_pi_f64(ws.dRhoCore, ws.dRhoAct, ws.dPair, ws.dG, npt, n2, static_cast<double*>(pi.ptr), stream, threads);
    throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_pi_f64 kernel launch");

    if (deriv >= 1) {
      const CudaArrayView pi_grad = cuda_array_view_from_object(pi_grad_obj, "pi_grad");
      require_typestr(pi_grad, "pi_grad", "<f8");
      require_c_contiguous(pi_grad, "pi_grad", 8);
      if (pi_grad.shape.size() != 2 || pi_grad.shape[0] != npt || pi_grad.shape[1] != 3) {
        throw std::invalid_argument("pi_grad must have shape (npt,3)");
      }
      orbitals_eval_pi_grad_f64(
          ws.dRhoCore,
          ws.dRhoAct,
          ws.dRhoCoreGrad,
          ws.dRhoActGrad,
          ws.dPsi,
          ws.dGrad,
          ws.dG,
          ncore,
          ncas,
          npt,
          nocc,
          static_cast<double*>(pi_grad.ptr),
          stream,
          threads);
      throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_pi_grad_f64 kernel launch");
    }

    if (compute_pi_laplacian) {
      const CudaArrayView pi_lapl = cuda_array_view_from_object(pi_lapl_obj, "pi_lapl");
      require_typestr(pi_lapl, "pi_lapl", "<f8");
      require_c_contiguous(pi_lapl, "pi_lapl", 8);
      if (pi_lapl.shape.size() != 1 || pi_lapl.shape[0] != npt) {
        throw std::invalid_argument("pi_lapl must have shape (npt,)");
      }

      orbitals_eval_pi_lapl_base_f64(
          ws.dRhoCore,
          ws.dRhoAct,
          ws.dRhoCoreGrad,
          ws.dRhoActGrad,
          ws.dRhoCoreLapl,
          ws.dRhoActLapl,
          ws.dPsi,
          ws.dGrad,
          ws.dLapl,
          ws.dG,
          ncore,
          ncas,
          npt,
          nocc,
          static_cast<double*>(pi_lapl.ptr),
          stream,
          threads);
      throw_on_cuda_error(cudaGetLastError(), "orbitals_eval_pi_lapl_base_f64 kernel launch");

      for (int32_t c = 0; c < 3; ++c) {
        orbitals_build_pair_y_f64(ws.dPsi, ws.dGrad, ncore, ncas, npt, nocc, c, ws.dPair, stream, threads);
        throw_on_cuda_error(cudaGetLastError(), "orbitals_build_pair_y_f64 kernel launch");

        if (npt > 0) {
          dgemm_pair_dm2T(ws.cublas, static_cast<const double*>(dm2.ptr), ws.dPair, ws.dGy, npt, n2);
        }

        orbitals_pi_lapl_add_termB_f64(ws.dPair, ws.dGy, npt, n2, static_cast<double*>(pi_lapl.ptr), stream, threads);
        throw_on_cuda_error(cudaGetLastError(), "orbitals_pi_lapl_add_termB_f64 kernel launch");
      }
    }

  } else {
    throw std::invalid_argument("ncas==0 is not supported by eval_density_otpd_f64_inplace_device");
  }

  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

}  // namespace

PYBIND11_MODULE(_orbitals_cuda_ext, m) {
  m.doc() = "CUDA extension for cartesian orbital and density ingredient evaluation";

  py::class_<IngredientsWorkspace>(m, "IngredientsWorkspace")
      .def(py::init<int32_t, int32_t, int32_t, int32_t>(), py::arg("max_npt"), py::arg("max_nocc"), py::arg("max_ncas"), py::arg("lmax") = 10)
      .def("release", &IngredientsWorkspace::release)
      .def_property_readonly("max_npt", [](const IngredientsWorkspace& w) { return w.max_npt; })
      .def_property_readonly("max_nocc", [](const IngredientsWorkspace& w) { return w.max_nocc; })
      .def_property_readonly("max_ncas", [](const IngredientsWorkspace& w) { return w.max_ncas; })
      .def_property_readonly("lmax", [](const IngredientsWorkspace& w) { return w.lmax; });

  py::class_<BeckeGridWorkspace>(m, "BeckeGridWorkspace")
      .def(py::init<int32_t, int32_t>(), py::arg("max_nloc"), py::arg("max_natm"))
      .def("release", &BeckeGridWorkspace::release)
      .def_property_readonly("max_nloc", [](const BeckeGridWorkspace& w) { return w.max_nloc; })
      .def_property_readonly("max_natm", [](const BeckeGridWorkspace& w) { return w.max_natm; });

  m.def(
      "eval_becke_atom_block_f64_inplace_device",
      &eval_becke_atom_block_f64_inplace_device,
      py::arg("center_xyz"),
      py::arg("radial_r"),
      py::arg("radial_wr"),
      py::arg("angular_dirs"),
      py::arg("angular_w"),
      py::arg("atom_coords"),
      py::arg("RAB"),
      py::arg("atom_index"),
      py::arg("becke_n"),
      py::arg("pts_out"),
      py::arg("w_out"),
      py::arg("workspace"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true);

  m.def(
      "eval_aos_cart_value_f64_inplace_device",
      &eval_aos_cart_value_f64_inplace_device,
      py::arg("shell_cxyz"),
      py::arg("shell_prim_start"),
      py::arg("shell_nprim"),
      py::arg("shell_l"),
      py::arg("shell_ao_start"),
      py::arg("prim_exp"),
      py::arg("prim_coef"),
      py::arg("points"),
      py::arg("ao"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true);

  m.def(
      "eval_density_otpd_f64_inplace_device",
      &eval_density_otpd_f64_inplace_device,
      py::arg("shell_cxyz"),
      py::arg("shell_prim_start"),
      py::arg("shell_nprim"),
      py::arg("shell_l"),
      py::arg("shell_ao_start"),
      py::arg("prim_exp"),
      py::arg("prim_coef"),
      py::arg("C_occ"),
      py::arg("points"),
      py::arg("dm1"),
      py::arg("dm2"),
      py::arg("rho"),
      py::arg("pi"),
      py::arg("rho_grad") = py::none(),
      py::arg("pi_grad") = py::none(),
      py::arg("tau") = py::none(),
      py::arg("rho_lapl") = py::none(),
      py::arg("pi_lapl") = py::none(),
      py::arg("ncore"),
      py::arg("ncas"),
      py::arg("deriv") = 0,
      py::arg("compute_tau") = false,
      py::arg("compute_rho_laplacian") = false,
      py::arg("compute_pi_laplacian") = false,
      py::arg("symmetrize_dm2") = true,
      py::arg("workspace"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true);
}
