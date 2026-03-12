#include "cueri_cuda_ext_common.h"

void cueri_bind_part2(py::module_& m) {
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
      "eri_ppps_tiny_warp_inplace_device",
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
        require_threads_multiple_of_32(threads, "eri_ppps_tiny_warp_inplace_device");
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
            cueri_eri_ppps_tiny_warp_launch_stream(
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
            "cueri_eri_ppps_tiny_warp_launch_stream");
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
      "eri_ppps_exact_shape_inplace_device",
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
         int npair_ab_expected,
         int npair_cd_expected,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_exact_shape_inplace_device");
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
            cueri_eri_ppps_exact_shape_launch_stream(
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
                int(npair_ab_expected),
                int(npair_cd_expected),
                stream,
                threads),
            "cueri_eri_ppps_exact_shape_launch_stream");
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
      py::arg("npair_ab_expected"),
      py::arg("npair_cd_expected"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_exact_shape_indexed_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object out_task_idx,
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
         int npair_ab_expected,
         int npair_cd_expected,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_exact_shape_indexed_inplace_device");
        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto out_idx = cuda_array_view_from_object(out_task_idx, "out_task_idx");
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
        require_typestr(out_idx, "out_task_idx", "<i4");
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
        if (require_1d(cd, "task_spCD") != ntasks || require_1d(out_idx, "out_task_idx") != ntasks) {
          throw std::invalid_argument("task_spAB/task_spCD/out_task_idx shape mismatch");
        }
        if ((require_1d(out, "eri_out") % 27) != 0) {
          throw std::invalid_argument("eri_out must have shape (ntasks_total*27,)");
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
            cueri_eri_ppps_exact_shape_indexed_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<const int32_t*>(out_idx.ptr),
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
                int(npair_ab_expected),
                int(npair_cd_expected),
                stream,
                threads),
            "cueri_eri_ppps_exact_shape_indexed_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("out_task_idx"),
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
      py::arg("npair_ab_expected"),
      py::arg("npair_cd_expected"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_prepacked_exact_inplace_device",
      [](py::object Ax,
         py::object Ay,
         py::object Az,
         py::object Bx,
         py::object By,
         py::object Bz,
         py::object Cx,
         py::object Cy,
         py::object Cz,
         py::object ab_eta,
         py::object ab_Px,
         py::object ab_Py,
         py::object ab_Pz,
         py::object ab_cK,
         py::object cd_eta,
         py::object cd_Qx,
         py::object cd_Qy,
         py::object cd_Qz,
         py::object cd_cK,
         py::object eri_out,
         int npair_ab_expected,
         int npair_cd_expected,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_prepacked_exact_inplace_device");
        auto ax = cuda_array_view_from_object(Ax, "Ax");
        auto ay = cuda_array_view_from_object(Ay, "Ay");
        auto az = cuda_array_view_from_object(Az, "Az");
        auto bx = cuda_array_view_from_object(Bx, "Bx");
        auto by = cuda_array_view_from_object(By, "By");
        auto bz = cuda_array_view_from_object(Bz, "Bz");
        auto cx = cuda_array_view_from_object(Cx, "Cx");
        auto cy = cuda_array_view_from_object(Cy, "Cy");
        auto cz = cuda_array_view_from_object(Cz, "Cz");
        auto abeta = cuda_array_view_from_object(ab_eta, "ab_eta");
        auto abpx = cuda_array_view_from_object(ab_Px, "ab_Px");
        auto abpy = cuda_array_view_from_object(ab_Py, "ab_Py");
        auto abpz = cuda_array_view_from_object(ab_Pz, "ab_Pz");
        auto abck = cuda_array_view_from_object(ab_cK, "ab_cK");
        auto cdeta = cuda_array_view_from_object(cd_eta, "cd_eta");
        auto cdqx = cuda_array_view_from_object(cd_Qx, "cd_Qx");
        auto cdqy = cuda_array_view_from_object(cd_Qy, "cd_Qy");
        auto cdqz = cuda_array_view_from_object(cd_Qz, "cd_Qz");
        auto cdck = cuda_array_view_from_object(cd_cK, "cd_cK");
        auto out = cuda_array_view_from_object(eri_out, "eri_out");

        require_typestr(ax, "Ax", "<f8");
        require_typestr(ay, "Ay", "<f8");
        require_typestr(az, "Az", "<f8");
        require_typestr(bx, "Bx", "<f8");
        require_typestr(by, "By", "<f8");
        require_typestr(bz, "Bz", "<f8");
        require_typestr(cx, "Cx", "<f8");
        require_typestr(cy, "Cy", "<f8");
        require_typestr(cz, "Cz", "<f8");
        require_typestr(abeta, "ab_eta", "<f8");
        require_typestr(abpx, "ab_Px", "<f8");
        require_typestr(abpy, "ab_Py", "<f8");
        require_typestr(abpz, "ab_Pz", "<f8");
        require_typestr(abck, "ab_cK", "<f8");
        require_typestr(cdeta, "cd_eta", "<f8");
        require_typestr(cdqx, "cd_Qx", "<f8");
        require_typestr(cdqy, "cd_Qy", "<f8");
        require_typestr(cdqz, "cd_Qz", "<f8");
        require_typestr(cdck, "cd_cK", "<f8");
        require_typestr(out, "eri_out", "<f8");

        const int64_t ntasks = require_1d(ax, "Ax");
        if (require_1d(ay, "Ay") != ntasks || require_1d(az, "Az") != ntasks ||
            require_1d(bx, "Bx") != ntasks || require_1d(by, "By") != ntasks ||
            require_1d(bz, "Bz") != ntasks || require_1d(cx, "Cx") != ntasks ||
            require_1d(cy, "Cy") != ntasks || require_1d(cz, "Cz") != ntasks) {
          throw std::invalid_argument("center arrays must all have shape (ntasks,)");
        }
        if (require_1d(out, "eri_out") != (ntasks * 27)) {
          throw std::invalid_argument("eri_out must have shape (ntasks*27,)");
        }
        const int64_t expected_ab = ntasks * static_cast<int64_t>(npair_ab_expected);
        const int64_t expected_cd = ntasks * static_cast<int64_t>(npair_cd_expected);
        if (require_1d(abeta, "ab_eta") != expected_ab || require_1d(abpx, "ab_Px") != expected_ab ||
            require_1d(abpy, "ab_Py") != expected_ab || require_1d(abpz, "ab_Pz") != expected_ab ||
            require_1d(abck, "ab_cK") != expected_ab) {
          throw std::invalid_argument("ab_* arrays must have shape (ntasks*npair_ab_expected,)");
        }
        if (require_1d(cdeta, "cd_eta") != expected_cd || require_1d(cdqx, "cd_Qx") != expected_cd ||
            require_1d(cdqy, "cd_Qy") != expected_cd || require_1d(cdqz, "cd_Qz") != expected_cd ||
            require_1d(cdck, "cd_cK") != expected_cd) {
          throw std::invalid_argument("cd_* arrays must have shape (ntasks*npair_cd_expected,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_eri_ppps_prepacked_exact_launch_stream(
                static_cast<const double*>(ax.ptr),
                static_cast<const double*>(ay.ptr),
                static_cast<const double*>(az.ptr),
                static_cast<const double*>(bx.ptr),
                static_cast<const double*>(by.ptr),
                static_cast<const double*>(bz.ptr),
                static_cast<const double*>(cx.ptr),
                static_cast<const double*>(cy.ptr),
                static_cast<const double*>(cz.ptr),
                static_cast<const double*>(abeta.ptr),
                static_cast<const double*>(abpx.ptr),
                static_cast<const double*>(abpy.ptr),
                static_cast<const double*>(abpz.ptr),
                static_cast<const double*>(abck.ptr),
                static_cast<const double*>(cdeta.ptr),
                static_cast<const double*>(cdqx.ptr),
                static_cast<const double*>(cdqy.ptr),
                static_cast<const double*>(cdqz.ptr),
                static_cast<const double*>(cdck.ptr),
                static_cast<int>(ntasks),
                static_cast<double*>(out.ptr),
                int(npair_ab_expected),
                int(npair_cd_expected),
                stream,
                threads),
            "cueri_eri_ppps_prepacked_exact_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("Ax"),
      py::arg("Ay"),
      py::arg("Az"),
      py::arg("Bx"),
      py::arg("By"),
      py::arg("Bz"),
      py::arg("Cx"),
      py::arg("Cy"),
      py::arg("Cz"),
      py::arg("ab_eta"),
      py::arg("ab_Px"),
      py::arg("ab_Py"),
      py::arg("ab_Pz"),
      py::arg("ab_cK"),
      py::arg("cd_eta"),
      py::arg("cd_Qx"),
      py::arg("cd_Qy"),
      py::arg("cd_Qz"),
      py::arg("cd_cK"),
      py::arg("eri_out"),
      py::arg("npair_ab_expected"),
      py::arg("npair_cd_expected"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "eri_ppps_runtime_dispatch_inplace_device",
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
         int tiny_work_max,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "eri_ppps_runtime_dispatch_inplace_device");
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
        if (tiny_work_max < 0) {
          throw std::invalid_argument("tiny_work_max must be >= 0");
        }
        if (ntasks == 0) return;

        cudaStream_t stream = stream_from_uint(stream_ptr);

        std::vector<int32_t> host_ab(static_cast<size_t>(ntasks));
        std::vector<int32_t> host_cd(static_cast<size_t>(ntasks));
        std::vector<int32_t> host_sp_npair(static_cast<size_t>(nsp));
        throw_on_cuda_error(cudaMemcpy(host_ab.data(), ab.ptr, static_cast<size_t>(ntasks) * sizeof(int32_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy(task_spAB D2H)");
        throw_on_cuda_error(cudaMemcpy(host_cd.data(), cd.ptr, static_cast<size_t>(ntasks) * sizeof(int32_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy(task_spCD D2H)");
        throw_on_cuda_error(cudaMemcpy(host_sp_npair.data(), sp_npair_v.ptr, static_cast<size_t>(nsp) * sizeof(int32_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy(sp_npair D2H)");

        struct ShapeGroup {
          int nab;
          int ncd;
          std::vector<int32_t> ab;
          std::vector<int32_t> cd;
          std::vector<int32_t> out_idx;
        };
        constexpr int kNumExact = 16;
        const int exact_shapes[kNumExact][2] = {
            {1, 1}, {1, 3}, {1, 6}, {1, 9},
            {3, 1}, {3, 3}, {3, 6}, {3, 9},
            {6, 1}, {6, 3}, {6, 6}, {6, 9},
            {9, 1}, {9, 3}, {9, 6}, {9, 9},
        };
        std::vector<ShapeGroup> exact_groups;
        exact_groups.reserve(kNumExact);
        for (int i = 0; i < kNumExact; ++i) {
          exact_groups.push_back(ShapeGroup{exact_shapes[i][0], exact_shapes[i][1], {}, {}, {}});
        }
        ShapeGroup tiny_group{0, 0, {}, {}, {}};
        ShapeGroup warp_group{0, 0, {}, {}, {}};

        auto exact_group_index = [](int nab, int ncd) -> int {
          if (nab == 1) {
            if (ncd == 1) return 0;
            if (ncd == 3) return 1;
            if (ncd == 6) return 2;
            if (ncd == 9) return 3;
            return -1;
          }
          if (nab == 3) {
            if (ncd == 1) return 4;
            if (ncd == 3) return 5;
            if (ncd == 6) return 6;
            if (ncd == 9) return 7;
            return -1;
          }
          if (nab == 6) {
            if (ncd == 1) return 8;
            if (ncd == 3) return 9;
            if (ncd == 6) return 10;
            if (ncd == 9) return 11;
            return -1;
          }
          if (nab == 9) {
            if (ncd == 1) return 12;
            if (ncd == 3) return 13;
            if (ncd == 6) return 14;
            if (ncd == 9) return 15;
            return -1;
          }
          return -1;
        };

        for (int64_t i = 0; i < ntasks; ++i) {
          const int32_t spab = host_ab[static_cast<size_t>(i)];
          const int32_t spcd = host_cd[static_cast<size_t>(i)];
          const int nab = host_sp_npair[static_cast<size_t>(spab)];
          const int ncd = host_sp_npair[static_cast<size_t>(spcd)];
          const int exact_idx = exact_group_index(nab, ncd);
          if (exact_idx >= 0) {
            auto& g = exact_groups[static_cast<size_t>(exact_idx)];
            g.ab.push_back(spab);
            g.cd.push_back(spcd);
            g.out_idx.push_back(static_cast<int32_t>(i));
            continue;
          }
          auto& g = (static_cast<int64_t>(nab) * static_cast<int64_t>(ncd) <= tiny_work_max) ? tiny_group : warp_group;
          g.ab.push_back(spab);
          g.cd.push_back(spcd);
          g.out_idx.push_back(static_cast<int32_t>(i));
        }

        struct DeviceBuffer {
          void* ptr{nullptr};
          cudaStream_t stream{nullptr};
          DeviceBuffer() = default;
          DeviceBuffer(size_t bytes, cudaStream_t s) : stream(s) {
            if (bytes > 0) throw_on_cuda_error(cudaMallocAsync(&ptr, bytes, s), "cudaMallocAsync");
          }
          DeviceBuffer(const DeviceBuffer&) = delete;
          DeviceBuffer& operator=(const DeviceBuffer&) = delete;
          DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), stream(other.stream) { other.ptr = nullptr; }
          DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
              if (ptr) cudaFreeAsync(ptr, stream);
              ptr = other.ptr;
              stream = other.stream;
              other.ptr = nullptr;
            }
            return *this;
          }
          ~DeviceBuffer() {
            if (ptr) cudaFreeAsync(ptr, stream);
          }
        };

        auto launch_indexed_group = [&](const std::vector<int32_t>& group_ab,
                                        const std::vector<int32_t>& group_cd,
                                        const std::vector<int32_t>& group_out_idx,
                                        auto launcher,
                                        const char* launcher_name) {
          const int count = static_cast<int>(group_ab.size());
          if (count == 0) return;
          DeviceBuffer d_ab(static_cast<size_t>(count) * sizeof(int32_t), stream);
          DeviceBuffer d_cd(static_cast<size_t>(count) * sizeof(int32_t), stream);
          DeviceBuffer d_out_idx(static_cast<size_t>(count) * sizeof(int32_t), stream);
          throw_on_cuda_error(cudaMemcpy(d_ab.ptr, group_ab.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(group_ab H2D)");
          throw_on_cuda_error(cudaMemcpy(d_cd.ptr, group_cd.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(group_cd H2D)");
          throw_on_cuda_error(cudaMemcpy(d_out_idx.ptr, group_out_idx.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(group_out_idx H2D)");
          throw_on_cuda_error(
              launcher(static_cast<const int32_t*>(d_ab.ptr),
                       static_cast<const int32_t*>(d_cd.ptr),
                       static_cast<const int32_t*>(d_out_idx.ptr),
                       count,
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
              launcher_name);
        };

        auto launch_exact_group = [&](const ShapeGroup& g) {
          const int count = static_cast<int>(g.ab.size());
          if (count == 0) return;
          DeviceBuffer d_ab(static_cast<size_t>(count) * sizeof(int32_t), stream);
          DeviceBuffer d_cd(static_cast<size_t>(count) * sizeof(int32_t), stream);
          DeviceBuffer d_out_idx(static_cast<size_t>(count) * sizeof(int32_t), stream);
          throw_on_cuda_error(cudaMemcpy(d_ab.ptr, g.ab.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(exact_ab H2D)");
          throw_on_cuda_error(cudaMemcpy(d_cd.ptr, g.cd.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(exact_cd H2D)");
          throw_on_cuda_error(cudaMemcpy(d_out_idx.ptr, g.out_idx.data(), static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice),
                              "cudaMemcpy(exact_out_idx H2D)");
          throw_on_cuda_error(
              cueri_eri_ppps_exact_shape_indexed_launch_stream(
                  static_cast<const int32_t*>(d_ab.ptr),
                  static_cast<const int32_t*>(d_cd.ptr),
                  static_cast<const int32_t*>(d_out_idx.ptr),
                  count,
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
                  g.nab,
                  g.ncd,
                  stream,
                  threads),
              "cueri_eri_ppps_exact_shape_indexed_launch_stream");
        };

        for (const auto& g : exact_groups) {
          launch_exact_group(g);
        }
        launch_indexed_group(
            tiny_group.ab,
            tiny_group.cd,
            tiny_group.out_idx,
            cueri_eri_ppps_tiny_warp_indexed_launch_stream,
            "cueri_eri_ppps_tiny_warp_indexed_launch_stream");
        launch_indexed_group(
            warp_group.ab,
            warp_group.cd,
            warp_group.out_idx,
            cueri_eri_ppps_warp_indexed_launch_stream,
            "cueri_eri_ppps_warp_indexed_launch_stream");
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
      py::arg("tiny_work_max") = 128,
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

}
