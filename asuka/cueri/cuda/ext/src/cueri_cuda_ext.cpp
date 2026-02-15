#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "cueri_cuda_kernels_api.h"

extern "C" cudaError_t cueri_sph_coeff_sph_to_cart_launch_stream(
    const double* C_sph,
    double* C_cart,
    int nao_cart,
    int norb,
    const int32_t* ao2shell_cart,
    const int32_t* ao2local_cart,
    const int32_t* shell_ao_start_sph,
    const int32_t* shell_l,
    cudaStream_t stream,
    int threads);

namespace py = pybind11;

namespace {

struct CudaArrayView {
  void* ptr = nullptr;
  bool read_only = false;
  std::string typestr;
  std::vector<int64_t> shape;
  bool has_strides = false;  // treat any non-null strides as non-contiguous
};

inline void throw_on_cuda_error(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline void validate_cuda_device_pointer(const void* ptr, const char* what) {
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear sticky error
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
  bool is_empty = false;
  for (py::handle dim : shape) {
    int64_t d = py::cast<int64_t>(dim);
    if (d < 0) throw std::invalid_argument(std::string(name) + " has negative dimension in shape");
    out.shape.push_back(d);
    if (d == 0) is_empty = true;
  }

  if (cai.contains("strides")) {
    py::object strides_obj = cai["strides"];
    if (!strides_obj.is_none()) {
      out.has_strides = true;
    }
  }

  if (out.ptr == nullptr) {
    if (!is_empty) {
      throw std::invalid_argument(std::string(name) + " has null device pointer");
    }
    return out;
  }
  const char* validate_env = std::getenv("CUERI_VALIDATE_CUDA_POINTERS");
  if (validate_env && validate_env[0] == '1') {
    validate_cuda_device_pointer(out.ptr, name);
  }
  return out;
}

inline void require_typestr(const CudaArrayView& a, const char* name, const char* expected) {
  if (a.typestr == expected) return;
  if (a.typestr.size() == 3 && expected[0] == '<' && a.typestr[0] == '=' && a.typestr[1] == expected[1] &&
      a.typestr[2] == expected[2]) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must have typestr " + expected + " (got " + a.typestr + ")");
}

inline int64_t require_1d(const CudaArrayView& a, const char* name) {
  if (a.shape.size() != 1) throw std::invalid_argument(std::string(name) + " must be a 1D device array");
  if (a.has_strides) throw std::invalid_argument(std::string(name) + " must be contiguous (strides must be None)");
  return a.shape[0];
}

inline cudaStream_t stream_from_uint(uint64_t stream_ptr) {
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

inline void require_threads_multiple_of_32(int threads, const char* name) {
  if (threads <= 0 || threads > 1024) {
    throw std::invalid_argument(std::string(name) + " threads must be in [1,1024]");
  }
  if ((threads & 31) != 0) {
    throw std::invalid_argument(std::string(name) + " threads must be a multiple of 32");
  }
}

using EriFixedLaunchFn = cudaError_t (*)(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads);

using EriFixedMultiblockLaunchFn = cudaError_t (*)(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* partial_sums,
    int blocks_per_task,
    double* eri_out,
    cudaStream_t stream,
    int threads);

void bind_fixed_class_eri_family(
    py::module_& m,
    const char* family,
    int ncomp,
    EriFixedLaunchFn launch_block,
    EriFixedLaunchFn launch_warp,
    EriFixedMultiblockLaunchFn launch_multiblock) {
  const int64_t ncomp_i64 = static_cast<int64_t>(ncomp);
  const std::string fam = std::string(family);

  const std::string py_block_name = "eri_" + fam + "_inplace_device";
  const std::string py_warp_name = "eri_" + fam + "_warp_inplace_device";
  const std::string py_multiblock_name = "eri_" + fam + "_multiblock_inplace_device";
  const std::string c_block_name = "cueri_eri_" + fam + "_launch_stream";
  const std::string c_warp_name = "cueri_eri_" + fam + "_warp_launch_stream";
  const std::string c_multiblock_name = "cueri_eri_" + fam + "_multiblock_launch_stream";

  m.def(
      py_block_name.c_str(),
      [=](py::object task_spAB,
          py::object task_spCD,
          py::object sp_A,
          py::object sp_B,
          py::object sp_pair_start,
          py::object sp_npair,
          py::object shell_cx,
          py::object shell_cy,
          py::object shell_cz,
          py::object pair_eta,
          py::object pair_Px,
          py::object pair_Py,
          py::object pair_Pz,
          py::object pair_cK,
          py::object eri_out,
          int threads,
          uint64_t stream_ptr,
          bool sync) {
        require_threads_multiple_of_32(threads, py_block_name.c_str());
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * ncomp_i64)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*ncomp,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            launch_block(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            c_block_name.c_str());
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      py_warp_name.c_str(),
      [=](py::object task_spAB,
          py::object task_spCD,
          py::object sp_A,
          py::object sp_B,
          py::object sp_pair_start,
          py::object sp_npair,
          py::object shell_cx,
          py::object shell_cy,
          py::object shell_cz,
          py::object pair_eta,
          py::object pair_Px,
          py::object pair_Py,
          py::object pair_Pz,
          py::object pair_cK,
          py::object eri_out,
          int threads,
          uint64_t stream_ptr,
          bool sync) {
        require_threads_multiple_of_32(threads, py_warp_name.c_str());
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * ncomp_i64)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*ncomp,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        const bool warp_supported = (ncomp_i64 <= 128) && (threads <= 256);
        auto launch_fn = warp_supported ? launch_warp : launch_block;
        const char* launch_name = warp_supported ? c_warp_name.c_str() : c_block_name.c_str();
        throw_on_cuda_error(
            launch_fn(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            launch_name);
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      py_multiblock_name.c_str(),
      [=](py::object task_spAB,
          py::object task_spCD,
          py::object sp_A,
          py::object sp_B,
          py::object sp_pair_start,
          py::object sp_npair,
          py::object shell_cx,
          py::object shell_cy,
          py::object shell_cz,
          py::object pair_eta,
          py::object pair_Px,
          py::object pair_Py,
          py::object pair_Pz,
          py::object pair_cK,
          py::object partial_sums,
          int blocks_per_task,
          py::object eri_out,
          int threads,
          uint64_t stream_ptr,
          bool sync) {
        require_threads_multiple_of_32(threads, py_multiblock_name.c_str());
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * ncomp_i64)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*ncomp,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * ncomp_i64;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*ncomp,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            launch_multiblock(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            c_multiblock_name.c_str());
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);
}

}  // namespace

PYBIND11_MODULE(_cueri_cuda_ext, m) {
  m.doc() = "cuERI CUDA extension (Step 1 kernels + Step 2 utilities)";
  constexpr int kCudaLmax = 5;
  constexpr int kCudaNrootsMax = 11;
  m.attr("CUDA_MAX_L") = py::int_(kCudaLmax);
  m.attr("CUDA_MAX_NROOTS") = py::int_(kCudaNrootsMax);
  m.def("kernel_limits_device", [=]() {
    py::dict out;
    out["lmax"] = py::int_(kCudaLmax);
    out["nroots_max"] = py::int_(kCudaNrootsMax);
    return out;
  });

  m.def(
      "build_pair_tables_ss_inplace_device",
      [](py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object shell_prim_start,
         py::object shell_nprim,
         py::object prim_exp,
         py::object prim_coef,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto s_prim_start = cuda_array_view_from_object(shell_prim_start, "shell_prim_start");
        auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
        auto p_exp = cuda_array_view_from_object(prim_exp, "prim_exp");
        auto p_coef = cuda_array_view_from_object(prim_coef, "prim_coef");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");

        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(s_prim_start, "shell_prim_start", "<i4");
        require_typestr(s_nprim, "shell_nprim", "<i4");
        require_typestr(p_exp, "prim_exp", "<f8");
        require_typestr(p_coef, "prim_coef", "<f8");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }
        if (require_1d(s_prim_start, "shell_prim_start") != nShell ||
            require_1d(s_nprim, "shell_nprim") != nShell) {
          throw std::invalid_argument("shell_prim_start/shell_nprim must have shape (nShell,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(p_coef, "prim_coef");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_build_pair_tables_ss_launch_stream(
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const int32_t*>(s_prim_start.ptr),
                static_cast<const int32_t*>(s_nprim.ptr),
                static_cast<const double*>(p_exp.ptr),
                static_cast<const double*>(p_coef.ptr),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<int>(nsp),
                static_cast<double*>(eta.ptr),
                static_cast<double*>(px.ptr),
                static_cast<double*>(pyv.ptr),
                static_cast<double*>(pz.ptr),
                static_cast<double*>(ck.ptr),
                stream,
                threads),
            "cueri_build_pair_tables_ss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("shell_prim_start"),
      py::arg("shell_nprim"),
      py::arg("prim_exp"),
      py::arg("prim_coef"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "sph_coeff_sph_to_cart_device",
      [](py::object C_sph,
         py::object C_cart,
         py::object ao2shell_cart,
         py::object ao2local_cart,
         py::object shell_ao_start_sph,
         py::object shell_l,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "sph_coeff_sph_to_cart_device");
        auto Csp = cuda_array_view_from_object(C_sph, "C_sph");
        auto Cca = cuda_array_view_from_object(C_cart, "C_cart");
        auto a2s = cuda_array_view_from_object(ao2shell_cart, "ao2shell_cart");
        auto a2l = cuda_array_view_from_object(ao2local_cart, "ao2local_cart");
        auto sh0 = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto shl = cuda_array_view_from_object(shell_l, "shell_l");

        require_typestr(Csp, "C_sph", "<f8");
        require_typestr(Cca, "C_cart", "<f8");
        require_typestr(a2s, "ao2shell_cart", "<i4");
        require_typestr(a2l, "ao2local_cart", "<i4");
        require_typestr(sh0, "shell_ao_start_sph", "<i4");
        require_typestr(shl, "shell_l", "<i4");

        if (Csp.shape.size() != 2 || Csp.has_strides) {
          throw std::invalid_argument("C_sph must be a contiguous 2D device array");
        }
        if (Cca.shape.size() != 2 || Cca.has_strides) {
          throw std::invalid_argument("C_cart must be a contiguous 2D device array");
        }

        const int64_t nao_sph = Csp.shape[0];
        const int64_t norb = Csp.shape[1];
        const int64_t nao_cart = Cca.shape[0];
        if (norb < 0 || nao_sph < 0 || nao_cart < 0) throw std::invalid_argument("invalid C_sph/C_cart shape");
        if (Cca.shape[1] != norb) throw std::invalid_argument("C_sph/C_cart norb mismatch");

        if (require_1d(a2s, "ao2shell_cart") != nao_cart || require_1d(a2l, "ao2local_cart") != nao_cart) {
          throw std::invalid_argument("ao2shell_cart/ao2local_cart must have shape (nao_cart,)");
        }
        const int64_t nShell = require_1d(sh0, "shell_ao_start_sph");
        if (require_1d(shl, "shell_l") != nShell) {
          throw std::invalid_argument("shell_ao_start_sph/shell_l must have identical shape (nShell,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_sph_coeff_sph_to_cart_launch_stream(
                static_cast<const double*>(Csp.ptr),
                static_cast<double*>(Cca.ptr),
                static_cast<int>(nao_cart),
                static_cast<int>(norb),
                static_cast<const int32_t*>(a2s.ptr),
                static_cast<const int32_t*>(a2l.ptr),
                static_cast<const int32_t*>(sh0.ptr),
                static_cast<const int32_t*>(shl.ptr),
                stream,
                threads),
            "cueri_sph_coeff_sph_to_cart_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("C_sph"),
      py::arg("C_cart"),
      py::arg("ao2shell_cart"),
      py::arg("ao2local_cart"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_l"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "schwarz_ssss_inplace_device",
      [](py::object sp_pair_start,
         py::object sp_npair,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object sp_Q,
         int threads,
         uint64_t stream_ptr,
         bool sync,
         bool fast_boys) {
        require_threads_multiple_of_32(threads, "schwarz_ssss_inplace_device");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto Q = cuda_array_view_from_object(sp_Q, "sp_Q");

        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(Q, "sp_Q", "<f8");

        const int64_t nsp = require_1d(sp_npair_v, "sp_npair");
        if (require_1d(Q, "sp_Q") != nsp) throw std::invalid_argument("sp_Q must have shape (nSP,)");
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_schwarz_ssss_launch_stream(
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<int>(nsp),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(Q.ptr),
                stream,
                threads,
                fast_boys),
            "cueri_schwarz_ssss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("sp_Q"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("fast_boys") = false);

  m.def(
      "eri_ssss_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync,
         bool fast_boys) {
        require_threads_multiple_of_32(threads, "eri_ssss_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != ntasks) throw std::invalid_argument("eri_out must have shape (ntasks,)");

        const int64_t nsp = require_1d(sp_npair_v, "sp_npair");
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }
        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ssss_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads,
                fast_boys),
            "cueri_eri_ssss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("fast_boys") = false);

  m.def(
      "eri_psss_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psss_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 3)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*3,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psss_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppss_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppss_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppss_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_psps_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psps_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psps_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psps_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 27)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*27,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppps_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppps_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_pppp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_pppp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 81)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*81,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_pppp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_pppp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_dsss_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_dsss_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 6)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*6,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_dsss_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_dsss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ddss_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ddss_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 36)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*36,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ddss_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ddss_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  bind_fixed_class_eri_family(
      m, "ssdp", 18, cueri_eri_ssdp_launch_stream, cueri_eri_ssdp_warp_launch_stream, cueri_eri_ssdp_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "psds", 18, cueri_eri_psds_launch_stream, cueri_eri_psds_warp_launch_stream, cueri_eri_psds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "psdp", 54, cueri_eri_psdp_launch_stream, cueri_eri_psdp_warp_launch_stream, cueri_eri_psdp_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "psdd", 108, cueri_eri_psdd_launch_stream, cueri_eri_psdd_warp_launch_stream, cueri_eri_psdd_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ppds", 54, cueri_eri_ppds_launch_stream, cueri_eri_ppds_warp_launch_stream, cueri_eri_ppds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ppdp", 162, cueri_eri_ppdp_launch_stream, cueri_eri_ppdp_warp_launch_stream, cueri_eri_ppdp_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ppdd", 324, cueri_eri_ppdd_launch_stream, cueri_eri_ppdd_warp_launch_stream, cueri_eri_ppdd_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dsds", 36, cueri_eri_dsds_launch_stream, cueri_eri_dsds_warp_launch_stream, cueri_eri_dsds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dsdp", 108, cueri_eri_dsdp_launch_stream, cueri_eri_dsdp_warp_launch_stream, cueri_eri_dsdp_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dsdd", 216, cueri_eri_dsdd_launch_stream, cueri_eri_dsdd_warp_launch_stream, cueri_eri_dsdd_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fpss", 30, cueri_eri_fpss_launch_stream, cueri_eri_fpss_warp_launch_stream, cueri_eri_fpss_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fdss", 60, cueri_eri_fdss_launch_stream, cueri_eri_fdss_warp_launch_stream, cueri_eri_fdss_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ffss", 100, cueri_eri_ffss_launch_stream, cueri_eri_ffss_warp_launch_stream, cueri_eri_ffss_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fpps", 90, cueri_eri_fpps_launch_stream, cueri_eri_fpps_warp_launch_stream, cueri_eri_fpps_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fdps", 180, cueri_eri_fdps_launch_stream, cueri_eri_fdps_warp_launch_stream, cueri_eri_fdps_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ffps", 300, cueri_eri_ffps_launch_stream, cueri_eri_ffps_warp_launch_stream, cueri_eri_ffps_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fpds", 180, cueri_eri_fpds_launch_stream, cueri_eri_fpds_warp_launch_stream, cueri_eri_fpds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fdds", 360, cueri_eri_fdds_launch_stream, cueri_eri_fdds_warp_launch_stream, cueri_eri_fdds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ffds", 600, cueri_eri_ffds_launch_stream, cueri_eri_ffds_warp_launch_stream, cueri_eri_ffds_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ssfs", 10, cueri_eri_ssfs_launch_stream, cueri_eri_ssfs_warp_launch_stream, cueri_eri_ssfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "psfs", 30, cueri_eri_psfs_launch_stream, cueri_eri_psfs_warp_launch_stream, cueri_eri_psfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ppfs", 90, cueri_eri_ppfs_launch_stream, cueri_eri_ppfs_warp_launch_stream, cueri_eri_ppfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dsfs", 60, cueri_eri_dsfs_launch_stream, cueri_eri_dsfs_warp_launch_stream, cueri_eri_dsfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fsfs", 100, cueri_eri_fsfs_launch_stream, cueri_eri_fsfs_warp_launch_stream, cueri_eri_fsfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dpfs", 180, cueri_eri_dpfs_launch_stream, cueri_eri_dpfs_warp_launch_stream, cueri_eri_dpfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fpfs", 300, cueri_eri_fpfs_launch_stream, cueri_eri_fpfs_warp_launch_stream, cueri_eri_fpfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ddfs", 360, cueri_eri_ddfs_launch_stream, cueri_eri_ddfs_warp_launch_stream, cueri_eri_ddfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fdfs", 600, cueri_eri_fdfs_launch_stream, cueri_eri_fdfs_warp_launch_stream, cueri_eri_fdfs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fffs", 1000, cueri_eri_fffs_launch_stream, cueri_eri_fffs_warp_launch_stream, cueri_eri_fffs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ssgs", 15, cueri_eri_ssgs_launch_stream, cueri_eri_ssgs_warp_launch_stream, cueri_eri_ssgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "psgs", 45, cueri_eri_psgs_launch_stream, cueri_eri_psgs_warp_launch_stream, cueri_eri_psgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ppgs", 135, cueri_eri_ppgs_launch_stream, cueri_eri_ppgs_warp_launch_stream, cueri_eri_ppgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dsgs", 90, cueri_eri_dsgs_launch_stream, cueri_eri_dsgs_warp_launch_stream, cueri_eri_dsgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fsgs", 150, cueri_eri_fsgs_launch_stream, cueri_eri_fsgs_warp_launch_stream, cueri_eri_fsgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dpgs", 270, cueri_eri_dpgs_launch_stream, cueri_eri_dpgs_warp_launch_stream, cueri_eri_dpgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fpgs", 450, cueri_eri_fpgs_launch_stream, cueri_eri_fpgs_warp_launch_stream, cueri_eri_fpgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ddgs", 540, cueri_eri_ddgs_launch_stream, cueri_eri_ddgs_warp_launch_stream, cueri_eri_ddgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "fdgs", 900, cueri_eri_fdgs_launch_stream, cueri_eri_fdgs_warp_launch_stream, cueri_eri_fdgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "ffgs", 1500, cueri_eri_ffgs_launch_stream, cueri_eri_ffgs_warp_launch_stream, cueri_eri_ffgs_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dpdp", 324, cueri_eri_dpdp_launch_stream, cueri_eri_dpdp_warp_launch_stream, cueri_eri_dpdp_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dpdd", 648, cueri_eri_dpdd_launch_stream, cueri_eri_dpdd_warp_launch_stream, cueri_eri_dpdd_multiblock_launch_stream);
  bind_fixed_class_eri_family(
      m, "dddd", 1296, cueri_eri_dddd_launch_stream, cueri_eri_dddd_warp_launch_stream, cueri_eri_dddd_multiblock_launch_stream);

  m.def(
      "eri_psss_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psss_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 3)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*3,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psss_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psss_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_psss_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psss_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 3)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*3,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 3;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*3,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psss_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psss_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppss_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppss_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppss_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppss_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppss_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppss_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 9;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppss_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppss_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_psps_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psps_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psps_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psps_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_psps_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_psps_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 9)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*9,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 9;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*9,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_psps_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_psps_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 27)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*27,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppps_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppps_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 27)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*27,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 27;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*27,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppps_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ppps_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_pppp_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_pppp_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 81)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*81,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_pppp_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_pppp_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_pppp_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_pppp_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 81)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*81,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 81;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*81,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_pppp_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_pppp_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_dsss_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_dsss_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 6)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*6,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_dsss_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_dsss_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_dsss_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_dsss_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 6)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*6,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 6;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*6,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_dsss_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_dsss_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ddss_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ddss_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 36)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*36,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ddss_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ddss_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ddss_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ddss_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != (ntasks * 36)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*36,)");
        }
        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task) * 36;
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task*36,)");
        }

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ddss_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_ddss_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ssss_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync,
         bool fast_boys) {
        require_threads_multiple_of_32(threads, "eri_ssss_warp_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != ntasks) throw std::invalid_argument("eri_out must have shape (ntasks,)");

        const int64_t nsp = require_1d(sp_npair_v, "sp_npair");
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }
        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ssss_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads,
                fast_boys),
            "cueri_eri_ssss_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("fast_boys") = false);

  m.def(
      "eri_ssss_multiblock_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object partial_sums,
         int blocks_per_task,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync,
         bool fast_boys) {
        require_threads_multiple_of_32(threads, "eri_ssss_multiblock_inplace_device");
        if (blocks_per_task <= 0 || blocks_per_task > 65535) {
          throw std::invalid_argument("blocks_per_task must be in [1,65535]");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ps = cuda_array_view_from_object(partial_sums, "partial_sums");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ps, "partial_sums", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        if (require_1d(out, "eri_out") != ntasks) throw std::invalid_argument("eri_out must have shape (ntasks,)");

        const int64_t expected_ps = ntasks * static_cast<int64_t>(blocks_per_task);
        if (require_1d(ps, "partial_sums") != expected_ps) {
          throw std::invalid_argument("partial_sums must have shape (ntasks*blocks_per_task,)");
        }

        const int64_t nsp = require_1d(sp_npair_v, "sp_npair");
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }
        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ssss_multiblock_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<double*>(ps.ptr),
                blocks_per_task,
                static_cast<double*>(out.ptr),
                stream,
                threads,
                fast_boys),
            "cueri_eri_ssss_multiblock_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("partial_sums"),
      py::arg("blocks_per_task"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true,
      py::arg("fast_boys") = false);

  m.def(
      "count_entries_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object counts,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto c = cuda_array_view_from_object(counts, "counts");
        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(c, "counts", "<i4");
        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        (void)require_1d(c, "counts");
        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_count_entries_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<int32_t*>(c.ptr),
                stream,
                threads),
            "cueri_count_entries_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("counts"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "fill_entry_csr_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object entry_offsets,
         py::object cursor,
         py::object entry_task,
         py::object entry_widx,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto offs = cuda_array_view_from_object(entry_offsets, "entry_offsets");
        auto cur = cuda_array_view_from_object(cursor, "cursor");
        auto et = cuda_array_view_from_object(entry_task, "entry_task");
        auto ew = cuda_array_view_from_object(entry_widx, "entry_widx");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(offs, "entry_offsets", "<i4");
        require_typestr(cur, "cursor", "<i4");
        require_typestr(et, "entry_task", "<i4");
        require_typestr(ew, "entry_widx", "<i4");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");
        const int64_t nkey = require_1d(cur, "cursor");
        if (require_1d(offs, "entry_offsets") != (nkey + 1)) {
          throw std::invalid_argument("entry_offsets must have shape (nkey+1,)");
        }
        const int64_t nEntry = require_1d(et, "entry_task");
        if (require_1d(ew, "entry_widx") != nEntry) throw std::invalid_argument("entry_task/entry_widx shape mismatch");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_fill_entry_csr_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(offs.ptr),
                static_cast<int32_t*>(cur.ptr),
                static_cast<int32_t*>(et.ptr),
                static_cast<int32_t*>(ew.ptr),
                stream,
                threads),
            "cueri_fill_entry_csr_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("entry_offsets"),
      py::arg("cursor"),
      py::arg("entry_task"),
      py::arg("entry_widx"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "reduce_from_entry_csr_inplace_device",
      [](py::object entry_offsets,
         py::object entry_task,
         py::object entry_widx,
         py::object eri_task,
         py::object W,
         py::object Out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "reduce_from_entry_csr_inplace_device");
        auto offs = cuda_array_view_from_object(entry_offsets, "entry_offsets");
        auto et = cuda_array_view_from_object(entry_task, "entry_task");
        auto ew = cuda_array_view_from_object(entry_widx, "entry_widx");
        auto eri = cuda_array_view_from_object(eri_task, "eri_task");
        auto w = cuda_array_view_from_object(W, "W");
        auto out = cuda_array_view_from_object(Out, "Out");

        require_typestr(offs, "entry_offsets", "<i4");
        require_typestr(et, "entry_task", "<i4");
        require_typestr(ew, "entry_widx", "<i4");
        require_typestr(eri, "eri_task", "<f8");
        require_typestr(w, "W", "<f8");
        require_typestr(out, "Out", "<f8");

        const int64_t nEntry = require_1d(et, "entry_task");
        if (require_1d(ew, "entry_widx") != nEntry) throw std::invalid_argument("entry_task/entry_widx shape mismatch");
        (void)require_1d(eri, "eri_task");
        const int64_t nkey = require_1d(out, "Out");
        if (require_1d(offs, "entry_offsets") != (nkey + 1)) {
          throw std::invalid_argument("entry_offsets must have shape (nkey+1,)");
        }
        if (require_1d(w, "W") != nkey) throw std::invalid_argument("W must have shape (nkey,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_reduce_from_entry_csr_launch_stream(
                static_cast<const int32_t*>(offs.ptr),
                static_cast<int>(nkey),
                static_cast<const int32_t*>(et.ptr),
                static_cast<const int32_t*>(ew.ptr),
                static_cast<const double*>(eri.ptr),
                static_cast<const double*>(w.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_reduce_from_entry_csr_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("entry_offsets"),
      py::arg("entry_task"),
      py::arg("entry_widx"),
      py::arg("eri_task"),
      py::arg("W"),
      py::arg("Out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_rys_generic_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int la,
         int lb,
         int lc,
         int ld,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_rys_generic_inplace_device");
        if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("la/lb/lc/ld must be >= 0");

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t nA = (static_cast<int64_t>(la) + 1) * (static_cast<int64_t>(la) + 2) / 2;
        const int64_t nB = (static_cast<int64_t>(lb) + 1) * (static_cast<int64_t>(lb) + 2) / 2;
        const int64_t nC = (static_cast<int64_t>(lc) + 1) * (static_cast<int64_t>(lc) + 2) / 2;
        const int64_t nD = (static_cast<int64_t>(ld) + 1) * (static_cast<int64_t>(ld) + 2) / 2;
        const int64_t nAB = nA * nB;
        const int64_t nCD = nC * nD;
        const int64_t need = ntasks * nAB * nCD;
        if (require_1d(out, "eri_out") != need) {
          throw std::invalid_argument("eri_out must have shape (ntasks*nAB*nCD,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_rys_generic_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<int>(ld),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_rys_generic_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("ld"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_rys_generic_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int la,
         int lb,
         int lc,
         int ld,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_rys_generic_warp_inplace_device");
        if (threads > 256) throw std::invalid_argument("eri_rys_generic_warp_inplace_device requires threads <= 256");
        if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("la/lb/lc/ld must be >= 0");

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t nA = (static_cast<int64_t>(la) + 1) * (static_cast<int64_t>(la) + 2) / 2;
        const int64_t nB = (static_cast<int64_t>(lb) + 1) * (static_cast<int64_t>(lb) + 2) / 2;
        const int64_t nC = (static_cast<int64_t>(lc) + 1) * (static_cast<int64_t>(lc) + 2) / 2;
        const int64_t nD = (static_cast<int64_t>(ld) + 1) * (static_cast<int64_t>(ld) + 2) / 2;
        const int64_t need = ntasks * (nA * nB) * (nC * nD);
        if (require_1d(out, "eri_out") != need) {
          throw std::invalid_argument("eri_out must have shape (ntasks*nAB*nCD,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_rys_generic_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<int>(ld),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_rys_generic_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("ld"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);



m.def(
    "eri_rys_generic_deriv_contracted_inplace_device",
    [](py::object task_spAB,
       py::object task_spCD,
       py::object sp_A,
       py::object sp_B,
       py::object sp_pair_start,
       py::object sp_npair,
       int la,
       int lb,
       int lc,
       int ld,
       py::object shell_cx,
       py::object shell_cy,
       py::object shell_cz,
       py::object shell_prim_start,
       py::object shell_nprim,
       py::object prim_exp,
       py::object pair_eta,
       py::object pair_Px,
       py::object pair_Py,
       py::object pair_Pz,
       py::object pair_cK,
       py::object bar_eri,
       py::object out,
       int threads,
       uint64_t stream_ptr,
       bool sync) {
      require_threads_multiple_of_32(threads, "eri_rys_generic_deriv_contracted_inplace_device");
      if (threads > 256) {
        throw std::invalid_argument("eri_rys_generic_deriv_contracted_inplace_device requires threads <= 256");
      }
      if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("la/lb/lc/ld must be >= 0");
      if (la > 5 || lb > 5 || lc > 5 || ld > 5) {
        throw std::invalid_argument("eri_rys_generic_deriv_contracted_inplace_device supports only l<=5");
      }

      auto t_ab = cuda_array_view_from_object(task_spAB, "task_spAB");
      auto t_cd = cuda_array_view_from_object(task_spCD, "task_spCD");
      auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
      auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
      auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
      auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");

      auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
      auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
      auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
      auto s_prim_start = cuda_array_view_from_object(shell_prim_start, "shell_prim_start");
      auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
      auto p_exp = cuda_array_view_from_object(prim_exp, "prim_exp");

      auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
      auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
      auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
      auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
      auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");

      auto bar = cuda_array_view_from_object(bar_eri, "bar_eri");
      auto out_v = cuda_array_view_from_object(out, "out");

      require_typestr(t_ab, "task_spAB", "<i4");
      require_typestr(t_cd, "task_spCD", "<i4");
      require_typestr(a_sp, "sp_A", "<i4");
      require_typestr(b_sp, "sp_B", "<i4");
      require_typestr(sp_start, "sp_pair_start", "<i4");
      require_typestr(sp_npair_v, "sp_npair", "<i4");

      require_typestr(cx, "shell_cx", "<f8");
      require_typestr(cy, "shell_cy", "<f8");
      require_typestr(cz, "shell_cz", "<f8");
      require_typestr(s_prim_start, "shell_prim_start", "<i4");
      require_typestr(s_nprim, "shell_nprim", "<i4");
      require_typestr(p_exp, "prim_exp", "<f8");

      require_typestr(eta, "pair_eta", "<f8");
      require_typestr(px, "pair_Px", "<f8");
      require_typestr(pyv, "pair_Py", "<f8");
      require_typestr(pz, "pair_Pz", "<f8");
      require_typestr(ck, "pair_cK", "<f8");

      require_typestr(bar, "bar_eri", "<f8");
      require_typestr(out_v, "out", "<f8");

      const int64_t ntasks = require_1d(t_ab, "task_spAB");
      if (ntasks <= 0) return;
      if (require_1d(t_cd, "task_spCD") != ntasks) {
        throw std::invalid_argument("task_spAB and task_spCD must have identical shape (ntasks,)");
      }

      const int64_t nsp = require_1d(a_sp, "sp_A");
      if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
        throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
      }
      if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
        throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
      }

      const int64_t nShell = require_1d(cx, "shell_cx");
      if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
        throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape (nShell,)");
      }
      if (require_1d(s_prim_start, "shell_prim_start") != nShell || require_1d(s_nprim, "shell_nprim") != nShell) {
        throw std::invalid_argument("shell_prim_start/shell_nprim must have shape (nShell,)");
      }
      (void)require_1d(p_exp, "prim_exp");

      const int64_t nPair = require_1d(eta, "pair_eta");
      if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair || require_1d(pz, "pair_Pz") != nPair ||
          require_1d(ck, "pair_cK") != nPair) {
        throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
      }

      
  // bar_eri shape: (ntasks * nAB * nCD,)
  auto ncart = [](int l) -> int64_t {
    return static_cast<int64_t>(l + 1) * static_cast<int64_t>(l + 2) / 2;
  };
  const int64_t nAB = ncart(la) * ncart(lb);
  const int64_t nCD = ncart(lc) * ncart(ld);
  const int64_t nElem = nAB * nCD;

  const int64_t expected_bar = ntasks * nElem;
  const int64_t expected_out = ntasks * 12;
  if (require_1d(bar, "bar_eri") != expected_bar) {
    throw std::invalid_argument("bar_eri must have shape (ntasks*nAB*nCD,)");
  }
  if (require_1d(out_v, "out") != expected_out) {
    throw std::invalid_argument("out must have shape (ntasks*12,)");
  }

  cudaStream_t stream = stream_from_uint(stream_ptr);
  throw_on_cuda_error(
      cueri_eri_rys_generic_deriv_contracted_cart_launch_stream(
          static_cast<const int32_t*>(t_ab.ptr),
          static_cast<const int32_t*>(t_cd.ptr),
          static_cast<int>(ntasks),
          static_cast<const int32_t*>(a_sp.ptr),
          static_cast<const int32_t*>(b_sp.ptr),
          static_cast<const int32_t*>(sp_start.ptr),
          static_cast<const int32_t*>(sp_npair_v.ptr),
          static_cast<const double*>(cx.ptr),
          static_cast<const double*>(cy.ptr),
          static_cast<const double*>(cz.ptr),
          static_cast<const int32_t*>(s_prim_start.ptr),
          static_cast<const int32_t*>(s_nprim.ptr),
          static_cast<const double*>(p_exp.ptr),
          static_cast<const double*>(eta.ptr),
          static_cast<const double*>(px.ptr),
          static_cast<const double*>(pyv.ptr),
          static_cast<const double*>(pz.ptr),
          static_cast<const double*>(ck.ptr),
          static_cast<int32_t>(la),
          static_cast<int32_t>(lb),
          static_cast<int32_t>(lc),
          static_cast<int32_t>(ld),
          static_cast<const double*>(bar.ptr),
          static_cast<double*>(out_v.ptr),
          stream,
          static_cast<int32_t>(threads)),
      "cueri_eri_rys_generic_deriv_contracted_cart_launch_stream");
  if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
},
py::arg("task_spAB"),
py::arg("task_spCD"),
py::arg("sp_A"),
py::arg("sp_B"),
py::arg("sp_pair_start"),
py::arg("sp_npair"),
py::arg("la"),
py::arg("lb"),
py::arg("lc"),
py::arg("ld"),
py::arg("shell_cx"),
py::arg("shell_cy"),
py::arg("shell_cz"),
py::arg("shell_prim_start"),
py::arg("shell_nprim"),
py::arg("prim_exp"),
py::arg("pair_eta"),
py::arg("pair_Px"),
py::arg("pair_Py"),
py::arg("pair_Pz"),
py::arg("pair_cK"),
py::arg("bar_eri"),
py::arg("out"),
py::arg("threads") = 256,
py::arg("stream") = 0,
py::arg("sync") = true);

m.def(
    "eri_rys_generic_deriv_contracted_atom_grad_inplace_device",
    [](py::object task_spAB,
       py::object task_spCD,
       py::object sp_A,
       py::object sp_B,
       py::object sp_pair_start,
       py::object sp_npair,
       int la,
       int lb,
       int lc,
       int ld,
       py::object shell_cx,
       py::object shell_cy,
       py::object shell_cz,
       py::object shell_prim_start,
       py::object shell_nprim,
       py::object prim_exp,
       py::object pair_eta,
       py::object pair_Px,
       py::object pair_Py,
       py::object pair_Pz,
       py::object pair_cK,
       py::object bar_eri,
       py::object shell_atom,
       py::object grad_out,
       int threads,
       uint64_t stream_ptr,
       bool sync) {
      require_threads_multiple_of_32(threads, "eri_rys_generic_deriv_contracted_atom_grad_inplace_device");
      if (threads > 256) {
        throw std::invalid_argument("eri_rys_generic_deriv_contracted_atom_grad_inplace_device requires threads <= 256");
      }
      if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("la/lb/lc/ld must be >= 0");
      if (la > 5 || lb > 5 || lc > 5 || ld > 5) {
        throw std::invalid_argument("eri_rys_generic_deriv_contracted_atom_grad_inplace_device supports only l<=5");
      }

      auto t_ab = cuda_array_view_from_object(task_spAB, "task_spAB");
      auto t_cd = cuda_array_view_from_object(task_spCD, "task_spCD");
      auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
      auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
      auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
      auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");

      auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
      auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
      auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
      auto s_prim_start = cuda_array_view_from_object(shell_prim_start, "shell_prim_start");
      auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
      auto p_exp = cuda_array_view_from_object(prim_exp, "prim_exp");

      auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
      auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
      auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
      auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
      auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");

      auto bar = cuda_array_view_from_object(bar_eri, "bar_eri");
      auto shell_atom_v = cuda_array_view_from_object(shell_atom, "shell_atom");
      auto grad_v = cuda_array_view_from_object(grad_out, "grad_out");

      require_typestr(t_ab, "task_spAB", "<i4");
      require_typestr(t_cd, "task_spCD", "<i4");
      require_typestr(a_sp, "sp_A", "<i4");
      require_typestr(b_sp, "sp_B", "<i4");
      require_typestr(sp_start, "sp_pair_start", "<i4");
      require_typestr(sp_npair_v, "sp_npair", "<i4");

      require_typestr(cx, "shell_cx", "<f8");
      require_typestr(cy, "shell_cy", "<f8");
      require_typestr(cz, "shell_cz", "<f8");
      require_typestr(s_prim_start, "shell_prim_start", "<i4");
      require_typestr(s_nprim, "shell_nprim", "<i4");
      require_typestr(p_exp, "prim_exp", "<f8");

      require_typestr(eta, "pair_eta", "<f8");
      require_typestr(px, "pair_Px", "<f8");
      require_typestr(pyv, "pair_Py", "<f8");
      require_typestr(pz, "pair_Pz", "<f8");
      require_typestr(ck, "pair_cK", "<f8");

      require_typestr(bar, "bar_eri", "<f8");
      require_typestr(shell_atom_v, "shell_atom", "<i4");
      require_typestr(grad_v, "grad_out", "<f8");

      const int64_t ntasks = require_1d(t_ab, "task_spAB");
      if (ntasks <= 0) return;
      if (require_1d(t_cd, "task_spCD") != ntasks) {
        throw std::invalid_argument("task_spAB and task_spCD must have identical shape (ntasks,)");
      }

      const int64_t nsp = require_1d(a_sp, "sp_A");
      if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
        throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
      }
      if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
        throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
      }

      const int64_t nShell = require_1d(cx, "shell_cx");
      if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
        throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape (nShell,)");
      }
      if (require_1d(s_prim_start, "shell_prim_start") != nShell || require_1d(s_nprim, "shell_nprim") != nShell) {
        throw std::invalid_argument("shell_prim_start/shell_nprim must have shape (nShell,)");
      }
      if (require_1d(shell_atom_v, "shell_atom") != nShell) {
        throw std::invalid_argument("shell_atom must have shape (nShell,)");
      }
      (void)require_1d(p_exp, "prim_exp");

      const int64_t nPair = require_1d(eta, "pair_eta");
      if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair || require_1d(pz, "pair_Pz") != nPair ||
          require_1d(ck, "pair_cK") != nPair) {
        throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
      }

      auto ncart = [](int l) -> int64_t {
        return static_cast<int64_t>(l + 1) * static_cast<int64_t>(l + 2) / 2;
      };
      const int64_t nAB = ncart(la) * ncart(lb);
      const int64_t nCD = ncart(lc) * ncart(ld);
      const int64_t nElem = nAB * nCD;

      const int64_t expected_bar = ntasks * nElem;
      if (require_1d(bar, "bar_eri") != expected_bar) {
        throw std::invalid_argument("bar_eri must have shape (ntasks*nAB*nCD,)");
      }

      const int64_t ngrad = require_1d(grad_v, "grad_out");
      if (ngrad <= 0 || (ngrad % 3) != 0) {
        throw std::invalid_argument("grad_out must have shape (natm*3,)");
      }

      cudaStream_t stream = stream_from_uint(stream_ptr);
      throw_on_cuda_error(
          cueri_eri_rys_generic_deriv_contracted_atom_grad_inplace_cart_launch_stream(
              static_cast<const int32_t*>(t_ab.ptr),
              static_cast<const int32_t*>(t_cd.ptr),
              static_cast<int>(ntasks),
              static_cast<const int32_t*>(a_sp.ptr),
              static_cast<const int32_t*>(b_sp.ptr),
              static_cast<const int32_t*>(sp_start.ptr),
              static_cast<const int32_t*>(sp_npair_v.ptr),
              static_cast<const double*>(cx.ptr),
              static_cast<const double*>(cy.ptr),
              static_cast<const double*>(cz.ptr),
              static_cast<const int32_t*>(s_prim_start.ptr),
              static_cast<const int32_t*>(s_nprim.ptr),
              static_cast<const double*>(p_exp.ptr),
              static_cast<const double*>(eta.ptr),
              static_cast<const double*>(px.ptr),
              static_cast<const double*>(pyv.ptr),
              static_cast<const double*>(pz.ptr),
              static_cast<const double*>(ck.ptr),
              static_cast<int32_t>(la),
              static_cast<int32_t>(lb),
              static_cast<int32_t>(lc),
              static_cast<int32_t>(ld),
              static_cast<const double*>(bar.ptr),
              static_cast<const int32_t*>(shell_atom_v.ptr),
              static_cast<double*>(grad_v.ptr),
              stream,
              static_cast<int32_t>(threads)),
          "cueri_eri_rys_generic_deriv_contracted_atom_grad_inplace_cart_launch_stream");
      if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    },
    py::arg("task_spAB"),
    py::arg("task_spCD"),
    py::arg("sp_A"),
    py::arg("sp_B"),
    py::arg("sp_pair_start"),
    py::arg("sp_npair"),
    py::arg("la"),
    py::arg("lb"),
    py::arg("lc"),
    py::arg("ld"),
    py::arg("shell_cx"),
    py::arg("shell_cy"),
    py::arg("shell_cz"),
    py::arg("shell_prim_start"),
    py::arg("shell_nprim"),
    py::arg("prim_exp"),
    py::arg("pair_eta"),
    py::arg("pair_Px"),
    py::arg("pair_Py"),
    py::arg("pair_Pz"),
    py::arg("pair_cK"),
    py::arg("bar_eri"),
    py::arg("shell_atom"),
    py::arg("grad_out"),
    py::arg("threads") = 256,
    py::arg("stream") = 0,
    py::arg("sync") = true);

  m.def(
      "eri_rys_df_ld0_warp_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int la,
         int lb,
         int lc,
         py::object eri_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_rys_df_ld0_warp_inplace_device");
        if (threads > 256) throw std::invalid_argument("eri_rys_df_ld0_warp_inplace_device requires threads <= 256");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t nA = (static_cast<int64_t>(la) + 1) * (static_cast<int64_t>(la) + 2) / 2;
        const int64_t nB = (static_cast<int64_t>(lb) + 1) * (static_cast<int64_t>(lb) + 2) / 2;
        const int64_t nC = (static_cast<int64_t>(lc) + 1) * (static_cast<int64_t>(lc) + 2) / 2;
        const int64_t nAB = nA * nB;
        const int64_t need = ntasks * nAB * nC;
        if (require_1d(out, "eri_out") != need) {
          throw std::invalid_argument("eri_out must have shape (ntasks*nAB*nC,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_rys_df_ld0_warp_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_eri_rys_df_ld0_warp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("eri_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_rys_contracted_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_nprim,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object ao_shell_ao_start,
         py::object ao_shell_nctr,
         py::object ao_shell_coef_start,
         py::object ao_prim_coef,
         py::object aux_shell_ao_start,
         int n_shell_ao,
         int nao,
         int naux,
         int aux_p0_block,
         int la,
         int lb,
         int lc,
         py::object X_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_rys_contracted_inplace_device");
        if (threads > 256) throw std::invalid_argument("df_int3c2e_rys_contracted_inplace_device requires threads <= 256");
        if (n_shell_ao < 0 || nao < 0 || naux < 0 || aux_p0_block < 0) {
          throw std::invalid_argument("n_shell_ao/nao/naux/aux_p0_block must be >= 0");
        }
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ao_a0 = cuda_array_view_from_object(ao_shell_ao_start, "ao_shell_ao_start");
        auto ao_nctr_v = cuda_array_view_from_object(ao_shell_nctr, "ao_shell_nctr");
        auto ao_ck0 = cuda_array_view_from_object(ao_shell_coef_start, "ao_shell_coef_start");
        auto ao_coef = cuda_array_view_from_object(ao_prim_coef, "ao_prim_coef");
        auto aux_a0 = cuda_array_view_from_object(aux_shell_ao_start, "aux_shell_ao_start");
        auto x = cuda_array_view_from_object(X_out, "X_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(s_nprim, "shell_nprim", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ao_a0, "ao_shell_ao_start", "<i4");
        require_typestr(ao_nctr_v, "ao_shell_nctr", "<i4");
        require_typestr(ao_ck0, "ao_shell_coef_start", "<i4");
        require_typestr(ao_coef, "ao_prim_coef", "<f8");
        require_typestr(aux_a0, "aux_shell_ao_start", "<i4");
        require_typestr(x, "X_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }
        if (require_1d(s_nprim, "shell_nprim") != nShell) {
          throw std::invalid_argument("shell_nprim must have shape (nShell,)");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t n_shell_ao_ll = require_1d(ao_a0, "ao_shell_ao_start");
        if (require_1d(ao_nctr_v, "ao_shell_nctr") != n_shell_ao_ll ||
            require_1d(ao_ck0, "ao_shell_coef_start") != n_shell_ao_ll) {
          throw std::invalid_argument("ao_shell_* arrays must have identical shape (nShellAO,)");
        }
        if (n_shell_ao_ll != static_cast<int64_t>(n_shell_ao)) {
          throw std::invalid_argument("n_shell_ao does not match ao_shell_* array shape");
        }

        (void)require_1d(ao_coef, "ao_prim_coef");
        (void)require_1d(aux_a0, "aux_shell_ao_start");

        const int64_t x_len = require_1d(x, "X_out");
        const int64_t expected_x = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (x_len != expected_x) throw std::invalid_argument("X_out must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_rys_contracted_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const int32_t*>(s_nprim.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<const int32_t*>(ao_a0.ptr),
                static_cast<const int32_t*>(ao_nctr_v.ptr),
                static_cast<const int32_t*>(ao_ck0.ptr),
                static_cast<const double*>(ao_coef.ptr),
                static_cast<const int32_t*>(aux_a0.ptr),
                static_cast<int>(n_shell_ao),
                static_cast<int>(nao),
                static_cast<int>(naux),
                static_cast<int>(aux_p0_block),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<double*>(x.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_rys_contracted_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_nprim"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("ao_shell_ao_start"),
      py::arg("ao_shell_nctr"),
      py::arg("ao_shell_coef_start"),
      py::arg("ao_prim_coef"),
      py::arg("aux_shell_ao_start"),
      py::arg("n_shell_ao"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("aux_p0_block"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("X_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_rys_contracted_ctr2_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_nprim,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         py::object ao_shell_ao_start,
         py::object ao_shell_nctr,
         py::object ao_shell_coef_start,
         py::object ao_prim_coef,
         py::object aux_shell_ao_start,
         int n_shell_ao,
         int nao,
         int naux,
         int aux_p0_block,
         int la,
         int lb,
         int lc,
         py::object X_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_rys_contracted_ctr2_inplace_device");
        if (threads > 256) throw std::invalid_argument("df_int3c2e_rys_contracted_ctr2_inplace_device requires threads <= 256");
        if (n_shell_ao < 0 || nao < 0 || naux < 0 || aux_p0_block < 0) {
          throw std::invalid_argument("n_shell_ao/nao/naux/aux_p0_block must be >= 0");
        }
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");
        auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");
        auto ao_a0 = cuda_array_view_from_object(ao_shell_ao_start, "ao_shell_ao_start");
        auto ao_nctr_v = cuda_array_view_from_object(ao_shell_nctr, "ao_shell_nctr");
        auto ao_ck0 = cuda_array_view_from_object(ao_shell_coef_start, "ao_shell_coef_start");
        auto ao_coef = cuda_array_view_from_object(ao_prim_coef, "ao_prim_coef");
        auto aux_a0 = cuda_array_view_from_object(aux_shell_ao_start, "aux_shell_ao_start");
        auto x = cuda_array_view_from_object(X_out, "X_out");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");
        require_typestr(s_nprim, "shell_nprim", "<i4");
        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");
        require_typestr(ao_a0, "ao_shell_ao_start", "<i4");
        require_typestr(ao_nctr_v, "ao_shell_nctr", "<i4");
        require_typestr(ao_ck0, "ao_shell_coef_start", "<i4");
        require_typestr(ao_coef, "ao_prim_coef", "<f8");
        require_typestr(aux_a0, "aux_shell_ao_start", "<i4");
        require_typestr(x, "X_out", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) throw std::invalid_argument("task_spAB/task_spCD shape mismatch");

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }
        if (require_1d(s_nprim, "shell_nprim") != nShell) {
          throw std::invalid_argument("shell_nprim must have shape (nShell,)");
        }

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair ||
            require_1d(pz, "pair_Pz") != nPair || require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t n_shell_ao_ll = require_1d(ao_a0, "ao_shell_ao_start");
        if (require_1d(ao_nctr_v, "ao_shell_nctr") != n_shell_ao_ll ||
            require_1d(ao_ck0, "ao_shell_coef_start") != n_shell_ao_ll) {
          throw std::invalid_argument("ao_shell_* arrays must have identical shape (nShellAO,)");
        }
        if (n_shell_ao_ll != static_cast<int64_t>(n_shell_ao)) {
          throw std::invalid_argument("n_shell_ao does not match ao_shell_* array shape");
        }

        (void)require_1d(ao_coef, "ao_prim_coef");
        (void)require_1d(aux_a0, "aux_shell_ao_start");

        const int64_t x_len = require_1d(x, "X_out");
        const int64_t expected_x = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (x_len != expected_x) throw std::invalid_argument("X_out must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_rys_contracted_ctr2_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const int32_t*>(s_nprim.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<const int32_t*>(ao_a0.ptr),
                static_cast<const int32_t*>(ao_nctr_v.ptr),
                static_cast<const int32_t*>(ao_ck0.ptr),
                static_cast<const double*>(ao_coef.ptr),
                static_cast<const int32_t*>(aux_a0.ptr),
                static_cast<int>(n_shell_ao),
                static_cast<int>(nao),
                static_cast<int>(naux),
                static_cast<int>(aux_p0_block),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<double*>(x.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_rys_contracted_ctr2_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_nprim"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("ao_shell_ao_start"),
      py::arg("ao_shell_nctr"),
      py::arg("ao_shell_coef_start"),
      py::arg("ao_prim_coef"),
      py::arg("aux_shell_ao_start"),
      py::arg("n_shell_ao"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("aux_p0_block"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("X_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_deriv_contracted_cart_sp_batch_inplace_device",
      [](int spAB,
         py::object spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object shell_prim_start,
         py::object shell_nprim,
         py::object shell_ao_start,
         py::object prim_exp,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int nao,
         int naux,
         int la,
         int lb,
         int lc,
         py::object bar_X_flat,
         py::object out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_deriv_contracted_cart_sp_batch_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_sp_batch_inplace_device requires threads <= 256");
        }
        if (nao <= 0 || naux <= 0) throw std::invalid_argument("nao/naux must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_sp_batch_inplace_device supports only l<=5");

        auto cd = cuda_array_view_from_object(spCD, "spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");

        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto s_prim_start = cuda_array_view_from_object(shell_prim_start, "shell_prim_start");
        auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
        auto s_a0 = cuda_array_view_from_object(shell_ao_start, "shell_ao_start");
        auto p_exp = cuda_array_view_from_object(prim_exp, "prim_exp");

        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");

        auto bar = cuda_array_view_from_object(bar_X_flat, "bar_X_flat");
        auto out_v = cuda_array_view_from_object(out, "out");

        require_typestr(cd, "spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");

        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(s_prim_start, "shell_prim_start", "<i4");
        require_typestr(s_nprim, "shell_nprim", "<i4");
        require_typestr(s_a0, "shell_ao_start", "<i4");
        require_typestr(p_exp, "prim_exp", "<f8");

        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");

        require_typestr(bar, "bar_X_flat", "<f8");
        require_typestr(out_v, "out", "<f8");

        const int64_t ntasks = require_1d(cd, "spCD");
        if (ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }
        if (spAB < 0 || static_cast<int64_t>(spAB) >= nsp) throw std::invalid_argument("spAB out of range");

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }
        if (require_1d(s_prim_start, "shell_prim_start") != nShell || require_1d(s_nprim, "shell_nprim") != nShell ||
            require_1d(s_a0, "shell_ao_start") != nShell) {
          throw std::invalid_argument("shell_* arrays must have shape (nShell,)");
        }

        (void)require_1d(p_exp, "prim_exp");

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair || require_1d(pz, "pair_Pz") != nPair ||
            require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t bar_len = require_1d(bar, "bar_X_flat");
        const int64_t expected_bar = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_X_flat must have shape (nao*nao*naux,)");

        const int64_t out_len = require_1d(out_v, "out");
        const int64_t expected_out = ntasks * 9;
        if (out_len != expected_out) throw std::invalid_argument("out must have shape (ntasks*9,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_launch_stream(
                static_cast<int32_t>(spAB),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const int32_t*>(s_prim_start.ptr),
                static_cast<const int32_t*>(s_nprim.ptr),
                static_cast<const int32_t*>(s_a0.ptr),
                static_cast<const double*>(p_exp.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<int>(nao),
                static_cast<int>(naux),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<double*>(out_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB"),
      py::arg("spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("shell_prim_start"),
      py::arg("shell_nprim"),
      py::arg("shell_ao_start"),
      py::arg("prim_exp"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_flat"),
      py::arg("out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device",
      [](int spAB,
         py::object spCD,
         py::object sp_A,
         py::object sp_B,
         py::object sp_pair_start,
         py::object sp_npair,
         py::object shell_cx,
         py::object shell_cy,
         py::object shell_cz,
         py::object shell_prim_start,
         py::object shell_nprim,
         py::object shell_ao_start,
         py::object prim_exp,
         py::object pair_eta,
         py::object pair_Px,
         py::object pair_Py,
         py::object pair_Pz,
         py::object pair_cK,
         int nao,
         int naux,
         int la,
         int lc,
         py::object bar_V,
         py::object out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device requires threads <= 256");
        }
        if (nao < 0 || naux <= 0) throw std::invalid_argument("nao must be >=0 and naux must be >0");
        if (la < 0 || lc < 0) throw std::invalid_argument("la/lc must be >= 0");
        if (la > 5 || lc > 5) throw std::invalid_argument("df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device supports only l<=5");

        auto cd = cuda_array_view_from_object(spCD, "spCD");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto sp_start = cuda_array_view_from_object(sp_pair_start, "sp_pair_start");
        auto sp_npair_v = cuda_array_view_from_object(sp_npair, "sp_npair");

        auto cx = cuda_array_view_from_object(shell_cx, "shell_cx");
        auto cy = cuda_array_view_from_object(shell_cy, "shell_cy");
        auto cz = cuda_array_view_from_object(shell_cz, "shell_cz");
        auto s_prim_start = cuda_array_view_from_object(shell_prim_start, "shell_prim_start");
        auto s_nprim = cuda_array_view_from_object(shell_nprim, "shell_nprim");
        auto s_a0 = cuda_array_view_from_object(shell_ao_start, "shell_ao_start");
        auto p_exp = cuda_array_view_from_object(prim_exp, "prim_exp");

        auto eta = cuda_array_view_from_object(pair_eta, "pair_eta");
        auto px = cuda_array_view_from_object(pair_Px, "pair_Px");
        auto pyv = cuda_array_view_from_object(pair_Py, "pair_Py");
        auto pz = cuda_array_view_from_object(pair_Pz, "pair_Pz");
        auto ck = cuda_array_view_from_object(pair_cK, "pair_cK");

        auto bar = cuda_array_view_from_object(bar_V, "bar_V");
        auto out_v = cuda_array_view_from_object(out, "out");

        require_typestr(cd, "spCD", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(sp_start, "sp_pair_start", "<i4");
        require_typestr(sp_npair_v, "sp_npair", "<i4");

        require_typestr(cx, "shell_cx", "<f8");
        require_typestr(cy, "shell_cy", "<f8");
        require_typestr(cz, "shell_cz", "<f8");
        require_typestr(s_prim_start, "shell_prim_start", "<i4");
        require_typestr(s_nprim, "shell_nprim", "<i4");
        require_typestr(s_a0, "shell_ao_start", "<i4");
        require_typestr(p_exp, "prim_exp", "<f8");

        require_typestr(eta, "pair_eta", "<f8");
        require_typestr(px, "pair_Px", "<f8");
        require_typestr(pyv, "pair_Py", "<f8");
        require_typestr(pz, "pair_Pz", "<f8");
        require_typestr(ck, "pair_cK", "<f8");

        require_typestr(bar, "bar_V", "<f8");
        require_typestr(out_v, "out", "<f8");

        const int64_t ntasks = require_1d(cd, "spCD");
        if (ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }
        if (spAB < 0 || static_cast<int64_t>(spAB) >= nsp) throw std::invalid_argument("spAB out of range");

        const int64_t nShell = require_1d(cx, "shell_cx");
        if (require_1d(cy, "shell_cy") != nShell || require_1d(cz, "shell_cz") != nShell) {
          throw std::invalid_argument("shell_cx/shell_cy/shell_cz must have identical shape");
        }
        if (require_1d(s_prim_start, "shell_prim_start") != nShell || require_1d(s_nprim, "shell_nprim") != nShell ||
            require_1d(s_a0, "shell_ao_start") != nShell) {
          throw std::invalid_argument("shell_* arrays must have shape (nShell,)");
        }

        (void)require_1d(p_exp, "prim_exp");

        const int64_t nPair = require_1d(eta, "pair_eta");
        if (require_1d(px, "pair_Px") != nPair || require_1d(pyv, "pair_Py") != nPair || require_1d(pz, "pair_Pz") != nPair ||
            require_1d(ck, "pair_cK") != nPair) {
          throw std::invalid_argument("pair_* arrays must have identical shape (totalPairPrims,)");
        }

        const int64_t bar_len = require_1d(bar, "bar_V");
        const int64_t expected_bar = static_cast<int64_t>(naux) * static_cast<int64_t>(naux);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_V must have shape (naux*naux,)");

        const int64_t out_len = require_1d(out_v, "out");
        const int64_t expected_out = ntasks * 6;
        if (out_len != expected_out) throw std::invalid_argument("out must have shape (ntasks*6,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_metric_2c2e_deriv_contracted_cart_launch_stream(
                static_cast<int32_t>(spAB),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(a_sp.ptr),
                static_cast<const int32_t*>(b_sp.ptr),
                static_cast<const int32_t*>(sp_start.ptr),
                static_cast<const int32_t*>(sp_npair_v.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const int32_t*>(s_prim_start.ptr),
                static_cast<const int32_t*>(s_nprim.ptr),
                static_cast<const int32_t*>(s_a0.ptr),
                static_cast<const double*>(p_exp.ptr),
                static_cast<const double*>(eta.ptr),
                static_cast<const double*>(px.ptr),
                static_cast<const double*>(pyv.ptr),
                static_cast<const double*>(pz.ptr),
                static_cast<const double*>(ck.ptr),
                static_cast<int>(nao),
                static_cast<int>(naux),
                static_cast<int>(la),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<double*>(out_v.ptr),
                stream,
                threads),
            "cueri_df_metric_2c2e_deriv_contracted_cart_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB"),
      py::arg("spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("sp_pair_start"),
      py::arg("sp_npair"),
      py::arg("shell_cx"),
      py::arg("shell_cy"),
      py::arg("shell_cz"),
      py::arg("shell_prim_start"),
      py::arg("shell_nprim"),
      py::arg("shell_ao_start"),
      py::arg("prim_exp"),
      py::arg("pair_eta"),
      py::arg("pair_Px"),
      py::arg("pair_Py"),
      py::arg("pair_Pz"),
      py::arg("pair_cK"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lc"),
      py::arg("bar_V"),
      py::arg("out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "rys_roots_weights_inplace_device",
      [](py::object T, py::object roots_out, py::object weights_out, int nroots, int threads, uint64_t stream_ptr, bool sync) {
        require_threads_multiple_of_32(threads, "rys_roots_weights_inplace_device");
        if (nroots <= 0) throw std::invalid_argument("nroots must be > 0");
        auto t = cuda_array_view_from_object(T, "T");
        auto r = cuda_array_view_from_object(roots_out, "roots_out");
        auto w = cuda_array_view_from_object(weights_out, "weights_out");

        require_typestr(t, "T", "<f8");
        require_typestr(r, "roots_out", "<f8");
        require_typestr(w, "weights_out", "<f8");

        const int64_t nT = require_1d(t, "T");
        const int64_t nroots_ll = static_cast<int64_t>(nroots);
        const int64_t need = nT * nroots_ll;
        if (require_1d(r, "roots_out") != need || require_1d(w, "weights_out") != need) {
          throw std::invalid_argument("roots_out/weights_out must have shape (nT*nroots,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_rys_roots_weights_launch_stream(
                static_cast<const double*>(t.ptr),
                static_cast<int>(nT),
                static_cast<int>(nroots),
                static_cast<double*>(r.ptr),
                static_cast<double*>(w.ptr),
                stream,
                threads),
            "cueri_rys_roots_weights_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("T"),
      py::arg("roots_out"),
      py::arg("weights_out"),
      py::arg("nroots"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "scatter_df_metric_tiles_inplace_device",
      [](py::object tile,
         py::object p0,
         py::object q0,
         int naux,
         int nP,
         int nQ,
         py::object V_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_df_metric_tiles_inplace_device");
        if (naux < 0 || nP < 0 || nQ < 0) throw std::invalid_argument("naux/nP/nQ must be >= 0");
        auto t = cuda_array_view_from_object(tile, "tile");
        auto p = cuda_array_view_from_object(p0, "p0");
        auto q = cuda_array_view_from_object(q0, "q0");
        auto v = cuda_array_view_from_object(V_out, "V_out");

        require_typestr(t, "tile", "<f8");
        require_typestr(p, "p0", "<i4");
        require_typestr(q, "q0", "<i4");
        require_typestr(v, "V_out", "<f8");

        const int64_t ntasks = require_1d(p, "p0");
        if (require_1d(q, "q0") != ntasks) throw std::invalid_argument("p0/q0 must have identical shape (ntasks,)");

        const int64_t tile_len = require_1d(t, "tile");
        const int64_t expected_tile = ntasks * static_cast<int64_t>(nP) * static_cast<int64_t>(nQ);
        if (tile_len != expected_tile) throw std::invalid_argument("tile must have shape (ntasks*nP*nQ,)");

        const int64_t v_len = require_1d(v, "V_out");
        const int64_t expected_v = static_cast<int64_t>(naux) * static_cast<int64_t>(naux);
        if (v_len != expected_v) throw std::invalid_argument("V_out must have shape (naux*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_df_metric_tiles_launch_stream(
                static_cast<const double*>(t.ptr),
                static_cast<const int32_t*>(p.ptr),
                static_cast<const int32_t*>(q.ptr),
                static_cast<int>(ntasks),
                static_cast<int>(naux),
                static_cast<int>(nP),
                static_cast<int>(nQ),
                static_cast<double*>(v.ptr),
                stream,
                threads),
            "cueri_scatter_df_metric_tiles_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile"),
      py::arg("p0"),
      py::arg("q0"),
      py::arg("naux"),
      py::arg("nP"),
      py::arg("nQ"),
      py::arg("V_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "scatter_df_int3c2e_tiles_inplace_device",
      [](py::object tile,
         py::object a0,
         py::object b0,
         py::object p0,
         int nao,
         int naux,
         int nAB,
         int nB,
         int nP,
         py::object X_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_df_int3c2e_tiles_inplace_device");
        if (nao < 0 || naux < 0 || nAB < 0 || nB < 0 || nP < 0) throw std::invalid_argument("size args must be >= 0");
        auto t = cuda_array_view_from_object(tile, "tile");
        auto a = cuda_array_view_from_object(a0, "a0");
        auto b = cuda_array_view_from_object(b0, "b0");
        auto p = cuda_array_view_from_object(p0, "p0");
        auto x = cuda_array_view_from_object(X_out, "X_out");

        require_typestr(t, "tile", "<f8");
        require_typestr(a, "a0", "<i4");
        require_typestr(b, "b0", "<i4");
        require_typestr(p, "p0", "<i4");
        require_typestr(x, "X_out", "<f8");

        const int64_t ntasks = require_1d(a, "a0");
        if (require_1d(b, "b0") != ntasks || require_1d(p, "p0") != ntasks) {
          throw std::invalid_argument("a0/b0/p0 must have identical shape (ntasks,)");
        }

        const int64_t tile_len = require_1d(t, "tile");
        const int64_t expected_tile = ntasks * static_cast<int64_t>(nAB) * static_cast<int64_t>(nP);
        if (tile_len != expected_tile) throw std::invalid_argument("tile must have shape (ntasks*nAB*nP,)");

        const int64_t x_len = require_1d(x, "X_out");
        const int64_t expected_x = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (x_len != expected_x) throw std::invalid_argument("X_out must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_df_int3c2e_tiles_launch_stream(
                static_cast<const double*>(t.ptr),
                static_cast<const int32_t*>(a.ptr),
                static_cast<const int32_t*>(b.ptr),
                static_cast<const int32_t*>(p.ptr),
                static_cast<int>(ntasks),
                static_cast<int>(nao),
                static_cast<int>(naux),
                static_cast<int>(nAB),
                static_cast<int>(nB),
                static_cast<int>(nP),
                static_cast<double*>(x.ptr),
                stream,
                threads),
            "cueri_scatter_df_int3c2e_tiles_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile"),
      py::arg("a0"),
      py::arg("b0"),
      py::arg("p0"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("nAB"),
      py::arg("nB"),
      py::arg("nP"),
      py::arg("X_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "scatter_add_df_yt_tiles_inplace_device",
      [](py::object tile,
         py::object p0,
         int naux,
         int nops,
         int nP,
         py::object YT_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_add_df_yt_tiles_inplace_device");
        if (naux < 0 || nops < 0 || nP < 0) throw std::invalid_argument("naux/nops/nP must be >= 0");
        auto t = cuda_array_view_from_object(tile, "tile");
        auto p = cuda_array_view_from_object(p0, "p0");
        auto y = cuda_array_view_from_object(YT_out, "YT_out");

        require_typestr(t, "tile", "<f8");
        require_typestr(p, "p0", "<i4");
        require_typestr(y, "YT_out", "<f8");

        const int64_t ntasks = require_1d(p, "p0");

        const int64_t tile_len = require_1d(t, "tile");
        const int64_t expected_tile = ntasks * static_cast<int64_t>(nops) * static_cast<int64_t>(nP);
        if (tile_len != expected_tile) throw std::invalid_argument("tile must have shape (ntasks*nops*nP,)");

        const int64_t y_len = require_1d(y, "YT_out");
        const int64_t expected_y = static_cast<int64_t>(naux) * static_cast<int64_t>(nops);
        if (y_len != expected_y) throw std::invalid_argument("YT_out must have shape (naux*nops,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_add_df_yt_tiles_launch_stream(
                static_cast<const double*>(t.ptr),
                static_cast<const int32_t*>(p.ptr),
                static_cast<int>(ntasks),
                static_cast<int>(naux),
                static_cast<int>(nops),
                static_cast<int>(nP),
                static_cast<double*>(y.ptr),
                stream,
                threads),
            "cueri_scatter_add_df_yt_tiles_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile"),
      py::arg("p0"),
      py::arg("naux"),
      py::arg("nops"),
      py::arg("nP"),
      py::arg("YT_out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "scatter_eri_tiles_ordered_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start,
         int nao,
         int nA,
         int nB,
         int nC,
         int nD,
         py::object tile_vals,
         py::object eri_mat,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_eri_tiles_ordered_inplace_device");
        if (nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) {
          throw std::invalid_argument("nao/nA/nB/nC/nD must be > 0");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto spa = cuda_array_view_from_object(sp_A, "sp_A");
        auto spb = cuda_array_view_from_object(sp_B, "sp_B");
        auto sh0 = cuda_array_view_from_object(shell_ao_start, "shell_ao_start");
        auto tile = cuda_array_view_from_object(tile_vals, "tile_vals");
        auto out = cuda_array_view_from_object(eri_mat, "eri_mat");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(spa, "sp_A", "<i4");
        require_typestr(spb, "sp_B", "<i4");
        require_typestr(sh0, "shell_ao_start", "<i4");
        require_typestr(tile, "tile_vals", "<f8");
        require_typestr(out, "eri_mat", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) {
          throw std::invalid_argument("task_spAB/task_spCD must have identical shape (ntasks,)");
        }

        const int64_t nsp = require_1d(spa, "sp_A");
        if (require_1d(spb, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nsp,)");
        }
        (void)require_1d(sh0, "shell_ao_start");

        const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
        const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
        const int64_t need_tile = ntasks * nAB * nCD;
        if (require_1d(tile, "tile_vals") != need_tile) {
          throw std::invalid_argument("tile_vals must have shape (ntasks*nAB*nCD,)");
        }

        const int64_t nPairAO = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        const int64_t need_out = nPairAO * nPairAO;
        if (require_1d(out, "eri_mat") != need_out) {
          throw std::invalid_argument("eri_mat must have shape ((nao*nao)*(nao*nao),)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_eri_tiles_ordered_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(spa.ptr),
                static_cast<const int32_t*>(spb.ptr),
                static_cast<const int32_t*>(sh0.ptr),
                static_cast<int>(nao),
                static_cast<int>(nA),
                static_cast<int>(nB),
                static_cast<int>(nC),
                static_cast<int>(nD),
                static_cast<const double*>(tile.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_scatter_eri_tiles_ordered_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start"),
      py::arg("nao"),
      py::arg("nA"),
      py::arg("nB"),
      py::arg("nC"),
      py::arg("nD"),
      py::arg("tile_vals"),
      py::arg("eri_mat"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);
}
