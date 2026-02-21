#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct CudaArrayView {
  void* ptr = nullptr;
  bool read_only = false;
  std::string typestr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;  // empty => not provided / C-contiguous
  uint64_t stream = 0;
};

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void validate_cuda_device_pointer(const void* ptr, const char* what) {
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear
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
        out.strides_bytes.push_back(py::cast<int64_t>(s));
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
  const std::string e = normalize_typestr(std::string(expected));
  if (t == e) return;
  throw std::invalid_argument(std::string(name) + " must have typestr " + e + " (got " + a.typestr + ")");
}

inline int64_t itemsize_from_typestr(const CudaArrayView& a, const char* name) {
  const std::string t = normalize_typestr(a.typestr);
  if (t == "<f8") return (int64_t)sizeof(double);
  if (t == "<i4") return (int64_t)sizeof(int32_t);
  throw std::invalid_argument(std::string(name) + " unsupported typestr " + a.typestr);
}

inline std::vector<int64_t> default_c_strides_bytes(const std::vector<int64_t>& shape, int64_t itemsize) {
  std::vector<int64_t> out;
  out.resize(shape.size());
  int64_t stride = itemsize;
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    out[(size_t)i] = stride;
    stride *= shape[(size_t)i];
  }
  return out;
}

inline std::vector<int64_t> strides_elems(const CudaArrayView& a, const char* name) {
  const int64_t itemsize = itemsize_from_typestr(a, name);
  std::vector<int64_t> sb = a.strides_bytes;
  if (sb.empty()) {
    sb = default_c_strides_bytes(a.shape, itemsize);
  }
  if (sb.size() != a.shape.size()) {
    throw std::invalid_argument(std::string(name) + " strides rank mismatch");
  }
  std::vector<int64_t> out;
  out.resize(sb.size());
  for (size_t i = 0; i < sb.size(); ++i) {
    if (sb[i] % itemsize != 0) {
      throw std::invalid_argument(std::string(name) + " stride is not a multiple of itemsize");
    }
    out[i] = sb[i] / itemsize;
  }
  return out;
}

inline cudaStream_t pick_stream_u64(std::initializer_list<uint64_t> streams) {
  uint64_t s = 0;
  for (uint64_t v : streams) {
    if (v) {
      s = v;
      break;
    }
  }
  return reinterpret_cast<cudaStream_t>(s);
}

extern "C" void caspt2_apply_h0diag_sr_f64_launch(
    const double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* bd,
    const double* id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double alpha,
    double beta,
    double denom_tol,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    cudaStream_t stream);

extern "C" void caspt2_apply_precond_sr_f64_launch(
    const double* r,
    int64_t r_s0,
    int64_t r_s1,
    const double* bd,
    const double* id,
    int nin,
    int nis,
    double real_shift,
    double imag_shift,
    double scale,
    double denom_tol,
    double* out,
    int64_t out_s0,
    int64_t out_s1,
    cudaStream_t stream);

extern "C" void caspt2_mltsca_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    const int32_t* lst2,
    int n2,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    double val10,
    double val11,
    double val20,
    double val21,
    cudaStream_t stream);

extern "C" void caspt2_mltdxp_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    const int32_t* lst2,
    int n2,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_a,
    double val10,
    double val11,
    double val20,
    double val21,
    cudaStream_t stream);

extern "C" void caspt2_mltmv_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int64_t y_s2,
    int len_i,
    int len_a,
    double val10,
    double val11,
    cudaStream_t stream);

extern "C" void caspt2_mltr1_f64_launch(
    int imltop,
    const int32_t* lst1,
    int n1,
    double* x,
    int64_t x_s0,
    int64_t x_s1,
    int64_t x_s2,
    const double* f,
    int64_t f_s0,
    int64_t f_s1,
    double* y,
    int64_t y_s0,
    int64_t y_s1,
    int len_p,
    int len_q,
    double val10,
    double val11,
    cudaStream_t stream);

extern "C" void caspt2_ddot_f64_launch(
    const double* x,
    const double* y,
    int64_t n,
    double* out,
    cudaStream_t stream);

void apply_h0diag_sr_f64(
    const py::object& y_obj,
    const py::object& x_obj,
    const py::object& bd_obj,
    const py::object& id_obj,
    double real_shift,
    double imag_shift,
    double alpha,
    double beta,
    double denom_tol) {
  auto y = cuda_array_view_from_object(y_obj, "y");
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto bd = cuda_array_view_from_object(bd_obj, "bd");
  auto id = cuda_array_view_from_object(id_obj, "id");

  require_typestr(y, "y", "<f8");
  require_typestr(x, "x", "<f8");
  require_typestr(bd, "bd", "<f8");
  require_typestr(id, "id", "<f8");

  if (x.shape.size() != 2 || y.shape.size() != 2) {
    throw std::invalid_argument("x and y must be 2D arrays");
  }
  if (x.shape != y.shape) {
    throw std::invalid_argument("x and y shape mismatch");
  }
  const int nin = (int)x.shape[0];
  const int nis = (int)x.shape[1];

  if (bd.shape.size() != 1 || (int)bd.shape[0] != nin) {
    throw std::invalid_argument("bd must have shape (nin,)");
  }
  if (id.shape.size() != 1 || (int)id.shape[0] != nis) {
    throw std::invalid_argument("id must have shape (nis,)");
  }

  auto xs = strides_elems(x, "x");
  auto ys = strides_elems(y, "y");
  cudaStream_t stream = pick_stream_u64({y.stream, x.stream, bd.stream, id.stream});

  caspt2_apply_h0diag_sr_f64_launch(
      (const double*)x.ptr,
      xs[0],
      xs[1],
      (const double*)bd.ptr,
      (const double*)id.ptr,
      nin,
      nis,
      real_shift,
      imag_shift,
      alpha,
      beta,
      denom_tol,
      (double*)y.ptr,
      ys[0],
      ys[1],
      stream);
}

void apply_precond_sr_f64(
    const py::object& out_obj,
    const py::object& r_obj,
    const py::object& bd_obj,
    const py::object& id_obj,
    double real_shift,
    double imag_shift,
    double scale,
    double denom_tol) {
  auto out = cuda_array_view_from_object(out_obj, "out");
  auto r = cuda_array_view_from_object(r_obj, "r");
  auto bd = cuda_array_view_from_object(bd_obj, "bd");
  auto id = cuda_array_view_from_object(id_obj, "id");

  require_typestr(out, "out", "<f8");
  require_typestr(r, "r", "<f8");
  require_typestr(bd, "bd", "<f8");
  require_typestr(id, "id", "<f8");

  if (r.shape.size() != 2 || out.shape.size() != 2) {
    throw std::invalid_argument("r and out must be 2D arrays");
  }
  if (r.shape != out.shape) {
    throw std::invalid_argument("r and out shape mismatch");
  }
  const int nin = (int)r.shape[0];
  const int nis = (int)r.shape[1];

  if (bd.shape.size() != 1 || (int)bd.shape[0] != nin) {
    throw std::invalid_argument("bd must have shape (nin,)");
  }
  if (id.shape.size() != 1 || (int)id.shape[0] != nis) {
    throw std::invalid_argument("id must have shape (nis,)");
  }

  auto rs = strides_elems(r, "r");
  auto outs = strides_elems(out, "out");
  cudaStream_t stream = pick_stream_u64({out.stream, r.stream, bd.stream, id.stream});

  caspt2_apply_precond_sr_f64_launch(
      (const double*)r.ptr,
      rs[0],
      rs[1],
      (const double*)bd.ptr,
      (const double*)id.ptr,
      nin,
      nis,
      real_shift,
      imag_shift,
      scale,
      denom_tol,
      (double*)out.ptr,
      outs[0],
      outs[1],
      stream);
}

inline void require_list_soa_i32(const CudaArrayView& lst, const char* name, int& n_out) {
  require_typestr(lst, name, "<i4");
  if (lst.shape.size() != 2) {
    throw std::invalid_argument(std::string(name) + " must be a 2D array of shape (4, n)");
  }
  if (lst.shape[0] != 4) {
    throw std::invalid_argument(std::string(name) + " must have shape[0]==4 (SoA)");
  }
  n_out = (int)lst.shape[1];
}

void mltsca_f64(
    int imltop,
    const py::object& lst1_obj,
    const py::object& lst2_obj,
    const py::object& x_obj,
    const py::object& f_obj,
    const py::object& y_obj,
    const py::tuple& val1,
    const py::tuple& val2) {
  auto lst1 = cuda_array_view_from_object(lst1_obj, "lst1");
  auto lst2 = cuda_array_view_from_object(lst2_obj, "lst2");
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto f = cuda_array_view_from_object(f_obj, "f");
  auto y = cuda_array_view_from_object(y_obj, "y");

  int n1 = 0;
  int n2 = 0;
  require_list_soa_i32(lst1, "lst1", n1);
  require_list_soa_i32(lst2, "lst2", n2);

  require_typestr(x, "x", "<f8");
  require_typestr(f, "f", "<f8");
  require_typestr(y, "y", "<f8");

  if (x.shape.size() != 2 || f.shape.size() != 2 || y.shape.size() != 2) {
    throw std::invalid_argument("mltsca: x/f/y must be 2D arrays");
  }

  auto xs = strides_elems(x, "x");
  auto fs = strides_elems(f, "f");
  auto ys = strides_elems(y, "y");

  if (val1.size() != 2 || val2.size() != 2) {
    throw std::invalid_argument("val1/val2 must be length-2 tuples");
  }
  const double v10 = val1[0].cast<double>();
  const double v11 = val1[1].cast<double>();
  const double v20 = val2[0].cast<double>();
  const double v21 = val2[1].cast<double>();

  cudaStream_t stream = pick_stream_u64({x.stream, f.stream, y.stream, lst1.stream, lst2.stream});

  caspt2_mltsca_f64_launch(
      imltop,
      (const int32_t*)lst1.ptr,
      n1,
      (const int32_t*)lst2.ptr,
      n2,
      (double*)x.ptr,
      xs[0],
      xs[1],
      (const double*)f.ptr,
      fs[0],
      fs[1],
      (double*)y.ptr,
      ys[0],
      ys[1],
      v10,
      v11,
      v20,
      v21,
      stream);
}

void mltdxp_f64(
    int imltop,
    const py::object& lst1_obj,
    const py::object& lst2_obj,
    const py::object& x_obj,
    const py::object& f_obj,
    const py::object& y_obj,
    const py::tuple& val1,
    const py::tuple& val2) {
  auto lst1 = cuda_array_view_from_object(lst1_obj, "lst1");
  auto lst2 = cuda_array_view_from_object(lst2_obj, "lst2");
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto f = cuda_array_view_from_object(f_obj, "f");
  auto y = cuda_array_view_from_object(y_obj, "y");

  int n1 = 0;
  int n2 = 0;
  require_list_soa_i32(lst1, "lst1", n1);
  require_list_soa_i32(lst2, "lst2", n2);

  require_typestr(x, "x", "<f8");
  require_typestr(f, "f", "<f8");
  require_typestr(y, "y", "<f8");

  if (x.shape.size() != 3 || y.shape.size() != 3 || f.shape.size() != 2) {
    throw std::invalid_argument("mltdxp: x/y must be 3D and f must be 2D");
  }
  if (x.shape[2] != y.shape[2]) {
    throw std::invalid_argument("mltdxp: x/y last-dimension mismatch");
  }
  const int len_a = (int)x.shape[2];

  auto xs = strides_elems(x, "x");
  auto fs = strides_elems(f, "f");
  auto ys = strides_elems(y, "y");

  if (val1.size() != 2 || val2.size() != 2) {
    throw std::invalid_argument("val1/val2 must be length-2 tuples");
  }
  const double v10 = val1[0].cast<double>();
  const double v11 = val1[1].cast<double>();
  const double v20 = val2[0].cast<double>();
  const double v21 = val2[1].cast<double>();

  cudaStream_t stream = pick_stream_u64({x.stream, f.stream, y.stream, lst1.stream, lst2.stream});

  caspt2_mltdxp_f64_launch(
      imltop,
      (const int32_t*)lst1.ptr,
      n1,
      (const int32_t*)lst2.ptr,
      n2,
      (double*)x.ptr,
      xs[0],
      xs[1],
      xs[2],
      (const double*)f.ptr,
      fs[0],
      fs[1],
      (double*)y.ptr,
      ys[0],
      ys[1],
      ys[2],
      len_a,
      v10,
      v11,
      v20,
      v21,
      stream);
}

void mltmv_f64(
    int imltop,
    const py::object& lst1_obj,
    const py::object& x_obj,
    const py::object& f_obj,
    const py::object& y_obj,
    const py::tuple& val1) {
  auto lst1 = cuda_array_view_from_object(lst1_obj, "lst1");
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto f = cuda_array_view_from_object(f_obj, "f");
  auto y = cuda_array_view_from_object(y_obj, "y");

  int n1 = 0;
  require_list_soa_i32(lst1, "lst1", n1);

  require_typestr(x, "x", "<f8");
  require_typestr(f, "f", "<f8");
  require_typestr(y, "y", "<f8");

  if (x.shape.size() != 2 || f.shape.size() != 2 || y.shape.size() != 3) {
    throw std::invalid_argument("mltmv: x/f must be 2D and y must be 3D");
  }
  const int len_i = (int)x.shape[1];
  const int len_a = (int)f.shape[1];
  if ((int)y.shape[1] != len_i || (int)y.shape[2] != len_a) {
    throw std::invalid_argument("mltmv: y shape mismatch with x/f");
  }

  auto xs = strides_elems(x, "x");
  auto fs = strides_elems(f, "f");
  auto ys = strides_elems(y, "y");

  if (val1.size() != 2) {
    throw std::invalid_argument("val1 must be a length-2 tuple");
  }
  const double v10 = val1[0].cast<double>();
  const double v11 = val1[1].cast<double>();

  cudaStream_t stream = pick_stream_u64({x.stream, f.stream, y.stream, lst1.stream});

  caspt2_mltmv_f64_launch(
      imltop,
      (const int32_t*)lst1.ptr,
      n1,
      (double*)x.ptr,
      xs[0],
      xs[1],
      (const double*)f.ptr,
      fs[0],
      fs[1],
      (double*)y.ptr,
      ys[0],
      ys[1],
      ys[2],
      len_i,
      len_a,
      v10,
      v11,
      stream);
}

void mltr1_f64(
    int imltop,
    const py::object& lst1_obj,
    const py::object& x_obj,
    const py::object& f_obj,
    const py::object& y_obj,
    const py::tuple& val1) {
  auto lst1 = cuda_array_view_from_object(lst1_obj, "lst1");
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto f = cuda_array_view_from_object(f_obj, "f");
  auto y = cuda_array_view_from_object(y_obj, "y");

  int n1 = 0;
  require_list_soa_i32(lst1, "lst1", n1);

  require_typestr(x, "x", "<f8");
  require_typestr(f, "f", "<f8");
  require_typestr(y, "y", "<f8");

  if (x.shape.size() != 3 || f.shape.size() != 2 || y.shape.size() != 2) {
    throw std::invalid_argument("mltr1: x must be 3D, f/y must be 2D");
  }
  const int len_p = (int)x.shape[1];
  const int len_q = (int)x.shape[2];
  if ((int)f.shape[1] != len_p || (int)y.shape[1] != len_q) {
    throw std::invalid_argument("mltr1: f/y shape mismatch with x");
  }

  auto xs = strides_elems(x, "x");
  auto fs = strides_elems(f, "f");
  auto ys = strides_elems(y, "y");

  if (val1.size() != 2) {
    throw std::invalid_argument("val1 must be a length-2 tuple");
  }
  const double v10 = val1[0].cast<double>();
  const double v11 = val1[1].cast<double>();

  cudaStream_t stream = pick_stream_u64({x.stream, f.stream, y.stream, lst1.stream});

  caspt2_mltr1_f64_launch(
      imltop,
      (const int32_t*)lst1.ptr,
      n1,
      (double*)x.ptr,
      xs[0],
      xs[1],
      xs[2],
      (const double*)f.ptr,
      fs[0],
      fs[1],
      (double*)y.ptr,
      ys[0],
      ys[1],
      len_p,
      len_q,
      v10,
      v11,
      stream);
}

double ddot_f64(const py::object& x_obj, const py::object& y_obj) {
  auto x = cuda_array_view_from_object(x_obj, "x");
  auto y = cuda_array_view_from_object(y_obj, "y");

  require_typestr(x, "x", "<f8");
  require_typestr(y, "y", "<f8");

  if (x.shape.size() != 1 || y.shape.size() != 1) {
    throw std::invalid_argument("ddot: x and y must be 1D arrays");
  }
  if (x.shape[0] != y.shape[0]) {
    throw std::invalid_argument("ddot: x and y size mismatch");
  }
  const int64_t n = x.shape[0];
  if (n <= 0) return 0.0;

  cudaStream_t stream = pick_stream_u64({x.stream, y.stream});

  double* d_out = nullptr;
  throw_on_cuda_error(cudaMalloc((void**)&d_out, sizeof(double)), "cudaMalloc(ddot_out)");
  throw_on_cuda_error(cudaMemsetAsync(d_out, 0, sizeof(double), stream), "cudaMemsetAsync(ddot_out)");

  caspt2_ddot_f64_launch((const double*)x.ptr, (const double*)y.ptr, n, d_out, stream);

  double h_out = 0.0;
  throw_on_cuda_error(cudaMemcpyAsync(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost, stream), "D2H ddot");
  throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ddot)");
  cudaFree(d_out);
  return h_out;
}

}  // namespace

PYBIND11_MODULE(_caspt2_cuda_ext, m) {
  m.doc() = "CASPT2 CUDA kernels (ASUKA)";

  m.def(
      "apply_h0diag_sr_f64",
      &apply_h0diag_sr_f64,
      py::arg("y"),
      py::arg("x"),
      py::arg("bd"),
      py::arg("id"),
      py::arg("real_shift") = 0.0,
      py::arg("imag_shift") = 0.0,
      py::arg("alpha") = 1.0,
      py::arg("beta") = 0.0,
      py::arg("denom_tol") = 1e-14);

  m.def(
      "apply_precond_sr_f64",
      &apply_precond_sr_f64,
      py::arg("out"),
      py::arg("r"),
      py::arg("bd"),
      py::arg("id"),
      py::arg("real_shift") = 0.0,
      py::arg("imag_shift") = 0.0,
      py::arg("scale") = 1.0,
      py::arg("denom_tol") = 1e-14);

  m.def(
      "mltsca_f64",
      &mltsca_f64,
      py::arg("imltop"),
      py::arg("lst1"),
      py::arg("lst2"),
      py::arg("x"),
      py::arg("f"),
      py::arg("y"),
      py::arg("val1"),
      py::arg("val2"));

  m.def(
      "mltdxp_f64",
      &mltdxp_f64,
      py::arg("imltop"),
      py::arg("lst1"),
      py::arg("lst2"),
      py::arg("x"),
      py::arg("f"),
      py::arg("y"),
      py::arg("val1"),
      py::arg("val2"));

  m.def(
      "mltmv_f64",
      &mltmv_f64,
      py::arg("imltop"),
      py::arg("lst1"),
      py::arg("x"),
      py::arg("f"),
      py::arg("y"),
      py::arg("val1"));

  m.def(
      "mltr1_f64",
      &mltr1_f64,
      py::arg("imltop"),
      py::arg("lst1"),
      py::arg("x"),
      py::arg("f"),
      py::arg("y"),
      py::arg("val1"));

  m.def("ddot_f64", &ddot_f64, py::arg("x"), py::arg("y"));
}

