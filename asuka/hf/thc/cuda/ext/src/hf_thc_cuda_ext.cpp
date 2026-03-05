#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" void hf_thc_rowwise_dot_f64(
    const double* a, const double* x, int32_t npt, int32_t nao, double* out_m, cudaStream_t stream);
extern "C" void hf_thc_scale_rows_f64(
    const double* x, const double* n, int32_t npt, int32_t nao, double* out, cudaStream_t stream);
extern "C" void hf_thc_hadamard_inplace_f64(
    double* m,
    int64_t ld_m,
    const double* z,
    int64_t ld_z,
    int32_t npt,
    int32_t nb,
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

inline int64_t leading_dim_elems_2d_rowmajor(const CudaArrayView& a, const char* name, int64_t itemsize) {
  if ((int64_t)a.shape.size() != 2) throw std::invalid_argument(std::string(name) + " must be 2D");
  const int64_t ncol = a.shape[1];
  if (a.strides_bytes.empty()) {
    return ncol;
  }
  if ((int64_t)a.strides_bytes.size() != 2) {
    throw std::invalid_argument(std::string(name) + " has invalid strides rank");
  }
  const int64_t stride_row = a.strides_bytes[0];
  const int64_t stride_col = a.strides_bytes[1];
  if (stride_col != itemsize) {
    throw std::invalid_argument(std::string(name) + " must be contiguous in the last dimension");
  }
  if (stride_row % itemsize != 0) {
    throw std::invalid_argument(std::string(name) + " has invalid row stride");
  }
  return stride_row / itemsize;
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

void rowwise_dot_f64(
    const py::object& a_obj,
    const py::object& x_obj,
    const py::object& out_m_obj,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  (void)threads;
  const CudaArrayView a = cuda_array_view_from_object(a_obj, "A");
  const CudaArrayView x = cuda_array_view_from_object(x_obj, "X");
  const CudaArrayView out_m = cuda_array_view_from_object(out_m_obj, "out");

  require_typestr_f64(a, "A");
  require_typestr_f64(x, "X");
  require_typestr_f64(out_m, "out");

  if (out_m.read_only) throw std::invalid_argument("out must be a writable CUDA array");

  require_c_contiguous(a, "A", 8);
  require_c_contiguous(x, "X", 8);
  require_c_contiguous(out_m, "out", 8);

  if (a.shape.size() != 2) throw std::invalid_argument("A must have shape (npt,nao)");
  if (x.shape.size() != 2) throw std::invalid_argument("X must have shape (npt,nao)");
  if (out_m.shape.size() != 1) throw std::invalid_argument("out must have shape (npt,)");

  const int64_t npt = a.shape[0];
  const int64_t nao = a.shape[1];
  if (x.shape[0] != npt || x.shape[1] != nao) throw std::invalid_argument("X must have the same shape as A");
  if (out_m.shape[0] != npt) throw std::invalid_argument("out must have shape (npt,) matching A");

  if (npt <= 0 || nao <= 0) return;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  hf_thc_rowwise_dot_f64(
      static_cast<const double*>(a.ptr),
      static_cast<const double*>(x.ptr),
      int32_t(npt),
      int32_t(nao),
      static_cast<double*>(out_m.ptr),
      stream);
  throw_on_cuda_error(cudaGetLastError(), "hf_thc_rowwise_dot_f64 kernel launch");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

void scale_rows_f64(
    const py::object& x_obj,
    const py::object& n_obj,
    const py::object& out_obj,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  (void)threads;
  const CudaArrayView x = cuda_array_view_from_object(x_obj, "X");
  const CudaArrayView n = cuda_array_view_from_object(n_obj, "n");
  const CudaArrayView out = cuda_array_view_from_object(out_obj, "out");

  require_typestr_f64(x, "X");
  require_typestr_f64(n, "n");
  require_typestr_f64(out, "out");

  if (out.read_only) throw std::invalid_argument("out must be a writable CUDA array");

  require_c_contiguous(x, "X", 8);
  require_c_contiguous(n, "n", 8);
  require_c_contiguous(out, "out", 8);

  if (x.shape.size() != 2) throw std::invalid_argument("X must have shape (npt,nao)");
  if (n.shape.size() != 1) throw std::invalid_argument("n must have shape (npt,)");
  if (out.shape.size() != 2) throw std::invalid_argument("out must have shape (npt,nao)");

  const int64_t npt = x.shape[0];
  const int64_t nao = x.shape[1];
  if (n.shape[0] != npt) throw std::invalid_argument("n must have shape (npt,) matching X");
  if (out.shape[0] != npt || out.shape[1] != nao) throw std::invalid_argument("out must have the same shape as X");

  if (npt <= 0 || nao <= 0) return;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  hf_thc_scale_rows_f64(
      static_cast<const double*>(x.ptr),
      static_cast<const double*>(n.ptr),
      int32_t(npt),
      int32_t(nao),
      static_cast<double*>(out.ptr),
      stream);
  throw_on_cuda_error(cudaGetLastError(), "hf_thc_scale_rows_f64 kernel launch");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

void hadamard_inplace_f64(
    const py::object& m_obj,
    const py::object& z_obj,
    int32_t threads,
    uint64_t stream_ptr,
    bool sync) {
  (void)threads;
  const CudaArrayView m = cuda_array_view_from_object(m_obj, "M");
  const CudaArrayView z = cuda_array_view_from_object(z_obj, "Z");

  require_typestr_f64(m, "M");
  require_typestr_f64(z, "Z");

  if (m.read_only) throw std::invalid_argument("M must be a writable CUDA array");

  if (m.shape.size() != 2) throw std::invalid_argument("M must have shape (npt,nb)");
  if (z.shape.size() != 2) throw std::invalid_argument("Z must have shape (npt,nb)");

  const int64_t npt = m.shape[0];
  const int64_t nb = m.shape[1];
  if (z.shape[0] != npt || z.shape[1] != nb) throw std::invalid_argument("Z must have the same shape as M");

  if (npt <= 0 || nb <= 0) return;

  const int64_t ld_m = leading_dim_elems_2d_rowmajor(m, "M", 8);
  const int64_t ld_z = leading_dim_elems_2d_rowmajor(z, "Z", 8);

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  hf_thc_hadamard_inplace_f64(
      static_cast<double*>(m.ptr),
      ld_m,
      static_cast<const double*>(z.ptr),
      ld_z,
      int32_t(npt),
      int32_t(nb),
      stream);
  throw_on_cuda_error(cudaGetLastError(), "hf_thc_hadamard_inplace_f64 kernel launch");
  if (sync) {
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

}  // namespace

PYBIND11_MODULE(_hf_thc_cuda_ext, m) {
  m.doc() = "HF THC CUDA helpers (non-GEMM kernels for THC J/K builds).";

  m.def(
      "rowwise_dot_f64",
      &rowwise_dot_f64,
      py::arg("A"),
      py::arg("X"),
      py::arg("out"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true,
      R"pbdoc(
Compute m[p] = sum_mu A[p,mu] * X[p,mu] for A,X in row-major layout.

Parameters
----------
A : cuda array, shape (npt,nao), float64, C-contiguous
X : cuda array, shape (npt,nao), float64, C-contiguous
out : cuda array, shape (npt,), float64, writable
)pbdoc");

  m.def(
      "scale_rows_f64",
      &scale_rows_f64,
      py::arg("X"),
      py::arg("n"),
      py::arg("out"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true,
      R"pbdoc(
Compute out[p,mu] = X[p,mu] * n[p] for X in row-major layout.

Parameters
----------
X : cuda array, shape (npt,nao), float64, C-contiguous
n : cuda array, shape (npt,), float64, C-contiguous
out : cuda array, shape (npt,nao), float64, writable, C-contiguous
)pbdoc");

  m.def(
      "hadamard_inplace_f64",
      &hadamard_inplace_f64,
      py::arg("M"),
      py::arg("Z"),
      py::arg("threads") = 256,
      py::arg("stream_ptr") = uint64_t(0),
      py::arg("sync") = true,
      R"pbdoc(
In-place Hadamard product: M *= Z for row-major layout.

Parameters
----------
M : cuda array, shape (npt,nb), float64, writable, C-contiguous
Z : cuda array, shape (npt,nb), float64, C-contiguous
)pbdoc");
}
