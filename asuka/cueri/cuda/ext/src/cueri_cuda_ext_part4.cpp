#include "cueri_cuda_ext_common.h"

void cueri_bind_part4(py::module_& m) {
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

#ifndef CUERI_FAST_DEV_DIRECT_JK

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

#endif
}
