#include "cueri_cuda_ext_common.h"

void cueri_bind_part3(py::module_& m) {
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

  #ifndef CUERI_FAST_DEV_STEP2_ONLY
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
#endif

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

}
