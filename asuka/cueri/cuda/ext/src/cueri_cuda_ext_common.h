#pragma once
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

extern "C" cudaError_t cueri_df_bar_x_sph_to_cart_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_mnQ,
    double* out_cart_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_bar_x_sph_to_cart_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_mnQ,
    float* out_cart_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_bar_x_sph_qmn_to_cart_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_Qmn,
    double* out_cart_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_bar_x_sph_qmn_to_cart_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_sph_Qmn,
    float* out_cart_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_B_cart_to_sph_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    double* out_sph_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_B_cart_to_sph_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    float* out_sph_mnQ,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_B_cart_to_sph_qmn_sym_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    double* out_sph_Qmn,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_df_B_cart_to_sph_qmn_sym_to_f32_launch_stream(
    const int32_t* spAB_arr,
    int n_spAB,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start_cart,
    const int32_t* shell_ao_start_sph,
    int nao_cart,
    int nao_sph,
    int naux,
    int la,
    int lb,
    const double* in_cart_mnQ,
    float* out_sph_Qmn,
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

inline bool typestr_matches(const CudaArrayView& a, const char* expected) {
  if (a.typestr == expected) return true;
  if (a.typestr.size() == 3 && expected[0] == '<' && a.typestr[0] == '=' && a.typestr[1] == expected[1] &&
      a.typestr[2] == expected[2]) {
    return true;
  }
  return false;
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

// ---------------------------------------------------------------------------
// Mixed-precision ERI dispatch binding.
// precision_mode: 0=fp64, 1=f32_output, 2=mixed_fp64_output, 3=mixed_f32_output
// ---------------------------------------------------------------------------

using EriF32LaunchFn = cudaError_t (*)(
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
    float* eri_out,
    cudaStream_t stream,
    int threads);

inline void bind_mixed_precision_eri(
    py::module_& m,
    const char* family,
    int ncomp,
    EriFixedLaunchFn launch_fp64,
    EriF32LaunchFn launch_f32,
    EriFixedLaunchFn launch_mixed,
    EriF32LaunchFn launch_mixed_f32,
    EriFixedLaunchFn launch_f32accum = nullptr,
    EriF32LaunchFn launch_f32accum_f32 = nullptr) {
  const int64_t ncomp_i64 = static_cast<int64_t>(ncomp);
  const std::string fam = std::string(family);
  const std::string py_name = "eri_" + fam + "_mixed_inplace_device";
  const std::string c_name_base = "cueri_eri_" + fam;

  m.def(
      py_name.c_str(),
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
          int precision_mode,
          int threads,
          uint64_t stream_ptr,
          bool sync) {
        require_threads_multiple_of_32(threads, py_name.c_str());
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

        const bool out_f4 = (precision_mode & 1) != 0;
        if (out_f4) {
          require_typestr(out, "eri_out", "<f4");
        } else {
          require_typestr(out, "eri_out", "<f8");
        }

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

        auto call_f64 = [&](EriFixedLaunchFn fn) {
          throw_on_cuda_error(
              fn(static_cast<const int32_t*>(ab.ptr),
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
              (c_name_base + "_launch_stream").c_str());
        };
        auto call_f32 = [&](EriF32LaunchFn fn) {
          throw_on_cuda_error(
              fn(static_cast<const int32_t*>(ab.ptr),
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
                 static_cast<float*>(out.ptr),
                 stream,
                 threads),
              (c_name_base + "_f32_launch_stream").c_str());
        };

        switch (precision_mode) {
          case 0:
            call_f64(launch_fp64);
            break;
          case 1:
            if (!launch_f32) {
              throw std::invalid_argument("precision_mode=1 (f32 out) unsupported for this ERI class");
            }
            call_f32(launch_f32);
            break;
          case 2:
            call_f64(launch_mixed ? launch_mixed : launch_fp64);
            break;
          case 3:
            if (!(launch_mixed_f32 || launch_f32)) {
              throw std::invalid_argument("precision_mode=3 (mixed_f32 out) unsupported for this ERI class");
            }
            call_f32(launch_mixed_f32 ? launch_mixed_f32 : launch_f32);
            break;
          case 6:
            // Fall back in order: f32accum -> mixed -> fp64.
            call_f64(launch_f32accum ? launch_f32accum : (launch_mixed ? launch_mixed : launch_fp64));
            break;
          case 7:
            // Fall back in order: f32accum_f32 -> mixed_f32 -> f32.
            if (!(launch_f32accum_f32 || launch_mixed_f32 || launch_f32)) {
              throw std::invalid_argument("precision_mode=7 (f32accum_f32 out) unsupported for this ERI class");
            }
            call_f32(launch_f32accum_f32 ? launch_f32accum_f32 : (launch_mixed_f32 ? launch_mixed_f32 : launch_f32));
            break;
          default:
            throw std::invalid_argument("precision_mode must be 0-3 or 6-7");
        }

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
      py::arg("precision_mode") = 0,
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);
}

}  // namespace
