#include "cueri_cuda_ext_common.h"

void cueri_bind_part8(py::module_& m) {
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

  // ── 2c2e metric allsp atomgrad ──
  m.def(
      "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device",
      [](py::object spAB_arr,
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
         py::object shell_atom,
         py::object grad_dev,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device requires threads <= 256");
        }
        if (nao < 0 || naux <= 0) throw std::invalid_argument("nao must be >=0 and naux must be >0");
        if (la < 0 || lc < 0) throw std::invalid_argument("la/lc must be >= 0");
        if (la > 5 || lc > 5) throw std::invalid_argument("df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device supports only l<=5");

        auto ab = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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
        auto s_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto g_dev = cuda_array_view_from_object(grad_dev, "grad_dev");

        require_typestr(ab, "spAB_arr", "<i4");
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
        require_typestr(s_atom, "shell_atom", "<i4");
        require_typestr(g_dev, "grad_dev", "<f8");

        const int64_t n_spAB = require_1d(ab, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t bar_len = require_1d(bar, "bar_V");
        const int64_t expected_bar = static_cast<int64_t>(naux) * static_cast<int64_t>(naux);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_V must have shape (naux*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<int>(n_spAB),
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
                static_cast<const int32_t*>(s_atom.ptr),
                static_cast<double*>(g_dev.ptr),
                stream,
                threads),
            "cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
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
      py::arg("shell_atom"),
      py::arg("grad_dev"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  // ── 2c2e metric allsp atomgrad (lower triangle only) ──
  m.def(
      "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device",
      [](py::object spAB_arr,
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
         py::object shell_atom,
         py::object grad_dev,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(
            threads, "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device requires threads <= 256");
        }
        if (nao < 0 || naux <= 0) throw std::invalid_argument("nao must be >=0 and naux must be >0");
        if (la < 0 || lc < 0) throw std::invalid_argument("la/lc must be >= 0");
        if (la > 5 || lc > 5)
          throw std::invalid_argument(
              "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device supports only l<=5");

        auto ab = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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
        auto s_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto g_dev = cuda_array_view_from_object(grad_dev, "grad_dev");

        require_typestr(ab, "spAB_arr", "<i4");
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
        require_typestr(s_atom, "shell_atom", "<i4");
        require_typestr(g_dev, "grad_dev", "<f8");

        const int64_t n_spAB = require_1d(ab, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t bar_len = require_1d(bar, "bar_V");
        const int64_t expected_bar = static_cast<int64_t>(naux) * static_cast<int64_t>(naux);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_V must have shape (naux*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<int>(n_spAB),
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
                static_cast<const int32_t*>(s_atom.ptr),
                static_cast<double*>(g_dev.ptr),
                stream,
                threads),
            "cueri_df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
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
      py::arg("shell_atom"),
      py::arg("grad_dev"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

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
      "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device",
      [](py::object tile,
         py::object a0_sph,
         py::object b0_sph,
         py::object p0,
         int nao_sph,
         int naux,
         int nAB,
         int nB,
         int nP,
         int la,
         int lb,
         py::object X_out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device");
        if (nao_sph < 0 || naux < 0 || nAB < 0 || nB < 0 || nP < 0) throw std::invalid_argument("size args must be >= 0");
        auto t = cuda_array_view_from_object(tile, "tile");
        auto a = cuda_array_view_from_object(a0_sph, "a0_sph");
        auto b = cuda_array_view_from_object(b0_sph, "b0_sph");
        auto p = cuda_array_view_from_object(p0, "p0");
        auto x = cuda_array_view_from_object(X_out, "X_out");

        require_typestr(t, "tile", "<f8");
        require_typestr(a, "a0_sph", "<i4");
        require_typestr(b, "b0_sph", "<i4");
        require_typestr(p, "p0", "<i4");
        require_typestr(x, "X_out", "<f8");

        const int64_t ntasks = require_1d(a, "a0_sph");
        if (require_1d(b, "b0_sph") != ntasks || require_1d(p, "p0") != ntasks) {
          throw std::invalid_argument("a0_sph/b0_sph/p0 must have identical shape (ntasks,)");
        }

        const int64_t tile_len = require_1d(t, "tile");
        const int64_t expected_tile = ntasks * static_cast<int64_t>(nAB) * static_cast<int64_t>(nP);
        if (tile_len != expected_tile) throw std::invalid_argument("tile must have shape (ntasks*nAB*nP,)");

        const int64_t x_len = require_1d(x, "X_out");
        const int64_t expected_x = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(naux);
        if (x_len != expected_x) throw std::invalid_argument("X_out must have shape (nao_sph*nao_sph*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_df_int3c2e_tiles_cart_to_sph_launch_stream(
                static_cast<const double*>(t.ptr),
                static_cast<const int32_t*>(a.ptr),
                static_cast<const int32_t*>(b.ptr),
                static_cast<const int32_t*>(p.ptr),
                static_cast<int>(ntasks),
                static_cast<int>(nao_sph),
                static_cast<int>(naux),
                static_cast<int>(nAB),
                static_cast<int>(nB),
                static_cast<int>(nP),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<double*>(x.ptr),
                stream,
                threads),
            "cueri_scatter_df_int3c2e_tiles_cart_to_sph_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile"),
      py::arg("a0_sph"),
      py::arg("b0_sph"),
      py::arg("p0"),
      py::arg("nao_sph"),
      py::arg("naux"),
      py::arg("nAB"),
      py::arg("nB"),
      py::arg("nP"),
      py::arg("la"),
      py::arg("lb"),
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
      "df_symmetrize_mnq_inplace_device",
      [](py::object arr_mnQ,
         int nao,
         int naux,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_symmetrize_mnq_inplace_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto arr = cuda_array_view_from_object(arr_mnQ, "arr_mnQ");
        require_typestr(arr, "arr_mnQ", "<f8");
        const int64_t n = require_1d(arr, "arr_mnQ");
        const int64_t expected = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (n != expected) throw std::invalid_argument("arr_mnQ must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_symmetrize_mnq_inplace_launch_stream(
                static_cast<double*>(arr.ptr),
                static_cast<int>(nao),
                static_cast<int>(naux),
                stream,
                threads),
            "cueri_df_symmetrize_mnq_inplace_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("arr_mnQ"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_symmetrize_mnq_to_f32_device",
      [](py::object in_mnQ,
         py::object out_mnQ,
         int nao,
         int naux,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_symmetrize_mnq_to_f32_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto in = cuda_array_view_from_object(in_mnQ, "in_mnQ");
        auto out = cuda_array_view_from_object(out_mnQ, "out_mnQ");
        require_typestr(in, "in_mnQ", "<f8");
        require_typestr(out, "out_mnQ", "<f4");

        const int64_t n_in = require_1d(in, "in_mnQ");
        const int64_t expected = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (n_in != expected) throw std::invalid_argument("in_mnQ must have shape (nao*nao*naux,)");
        const int64_t n_out = require_1d(out, "out_mnQ");
        if (n_out != expected) throw std::invalid_argument("out_mnQ must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_symmetrize_mnq_to_f32_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<float*>(out.ptr),
                static_cast<int>(nao),
                static_cast<int>(naux),
                stream,
                threads),
            "cueri_df_symmetrize_mnq_to_f32_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_mnQ"),
      py::arg("out_mnQ"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_symmetrize_qmn_inplace_device",
      [](py::object arr_Qmn,
         int naux,
         int nao,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_symmetrize_qmn_inplace_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto arr = cuda_array_view_from_object(arr_Qmn, "arr_Qmn");
        require_typestr(arr, "arr_Qmn", "<f8");
        const int64_t n = require_1d(arr, "arr_Qmn");
        const int64_t expected = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (n != expected) throw std::invalid_argument("arr_Qmn must have shape (naux*nao*nao,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_symmetrize_qmn_inplace_launch_stream(
                static_cast<double*>(arr.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                stream,
                threads),
            "cueri_df_symmetrize_qmn_inplace_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("arr_Qmn"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_symmetrize_qmn_to_mnq_device",
      [](py::object in_Qmn,
         py::object out_mnQ,
         int naux,
         int nao,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_symmetrize_qmn_to_mnq_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto in = cuda_array_view_from_object(in_Qmn, "in_Qmn");
        auto out = cuda_array_view_from_object(out_mnQ, "out_mnQ");

        require_typestr(in, "in_Qmn", "<f8");
        const bool out_is_f8 = typestr_matches(out, "<f8");
        const bool out_is_f4 = typestr_matches(out, "<f4");
        if (!out_is_f8 && !out_is_f4) {
          throw std::invalid_argument("out_mnQ must have typestr <f8 or <f4 (got " + out.typestr + ")");
        }

        const int64_t n_in = require_1d(in, "in_Qmn");
        const int64_t expected = static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (n_in != expected) throw std::invalid_argument("in_Qmn must have shape (naux*nao*nao,)");
        const int64_t n_out = require_1d(out, "out_mnQ");
        if (n_out != expected) throw std::invalid_argument("out_mnQ must have shape (naux*nao*nao,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        if (out_is_f8) {
          throw_on_cuda_error(
              cueri_df_symmetrize_qmn_to_mnq_launch_stream(
                  static_cast<const double*>(in.ptr),
                  static_cast<double*>(out.ptr),
                  static_cast<int>(naux),
                  static_cast<int>(nao),
                  stream,
                  threads),
              "cueri_df_symmetrize_qmn_to_mnq_launch_stream");
        } else {
          throw_on_cuda_error(
              cueri_df_symmetrize_qmn_to_mnq_to_f32_launch_stream(
                  static_cast<const double*>(in.ptr),
                  static_cast<float*>(out.ptr),
                  static_cast<int>(naux),
                  static_cast<int>(nao),
                  stream,
                  threads),
              "cueri_df_symmetrize_qmn_to_mnq_to_f32_launch_stream");
        }
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qmn"),
      py::arg("out_mnQ"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_pack_mnq_to_qp_device",
      [](py::object in_mnQ,
         py::object out_Qp,
         int nao,
         int naux,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_pack_mnq_to_qp_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto in = cuda_array_view_from_object(in_mnQ, "in_mnQ");
        auto out = cuda_array_view_from_object(out_Qp, "out_Qp");
        require_typestr(in, "in_mnQ", "<f8");
        require_typestr(out, "out_Qp", "<f8");

        const int64_t n_in = require_1d(in, "in_mnQ");
        const int64_t expected_in =
            static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (n_in != expected_in) throw std::invalid_argument("in_mnQ must have shape (nao*nao*naux,)");
        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
        const int64_t expected_out = static_cast<int64_t>(naux) * ntri;
        const int64_t n_out = require_1d(out, "out_Qp");
        if (n_out != expected_out) throw std::invalid_argument("out_Qp must have shape (naux*ntri,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_pack_mnq_to_qp_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(nao),
                static_cast<int>(naux),
                stream,
                threads),
            "cueri_df_pack_mnq_to_qp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_mnQ"),
      py::arg("out_Qp"),
      py::arg("nao"),
      py::arg("naux"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_pack_qmn_to_qp_device",
      [](py::object in_Qmn,
         py::object out_Qp,
         int naux,
         int nao,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_pack_qmn_to_qp_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto in = cuda_array_view_from_object(in_Qmn, "in_Qmn");
        auto out = cuda_array_view_from_object(out_Qp, "out_Qp");
        require_typestr(in, "in_Qmn", "<f8");
        require_typestr(out, "out_Qp", "<f8");

        const int64_t n_in = require_1d(in, "in_Qmn");
        const int64_t expected_in =
            static_cast<int64_t>(naux) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (n_in != expected_in) throw std::invalid_argument("in_Qmn must have shape (naux*nao*nao,)");
        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
        const int64_t expected_out = static_cast<int64_t>(naux) * ntri;
        const int64_t n_out = require_1d(out, "out_Qp");
        if (n_out != expected_out) throw std::invalid_argument("out_Qp must have shape (naux*ntri,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_pack_qmn_to_qp_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                stream,
                threads),
            "cueri_df_pack_qmn_to_qp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qmn"),
      py::arg("out_Qp"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

	  m.def(
	      "df_pack_qmn_block_to_qp_device",
	      [](py::object in_Qmn_block,
	         py::object out_Qp_block,
         int q_count,
         int nao,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_pack_qmn_block_to_qp_device");
        if (nao < 0 || q_count < 0) throw std::invalid_argument("nao/q_count must be >= 0");
        auto in = cuda_array_view_from_object(in_Qmn_block, "in_Qmn_block");
        auto out = cuda_array_view_from_object(out_Qp_block, "out_Qp_block");
        require_typestr(in, "in_Qmn_block", "<f8");
        require_typestr(out, "out_Qp_block", "<f8");

        const int64_t n_in = require_1d(in, "in_Qmn_block");
        const int64_t expected_in =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (n_in != expected_in) throw std::invalid_argument("in_Qmn_block must have shape (q_count*nao*nao,)");
        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
        const int64_t expected_out = static_cast<int64_t>(q_count) * ntri;
        const int64_t n_out = require_1d(out, "out_Qp_block");
        if (n_out != expected_out) throw std::invalid_argument("out_Qp_block must have shape (q_count*ntri,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_pack_qmn_block_to_qp_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(q_count),
                static_cast<int>(nao),
                stream,
                threads),
            "cueri_df_pack_qmn_block_to_qp_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qmn_block"),
      py::arg("out_Qp_block"),
      py::arg("q_count"),
      py::arg("nao"),
	      py::arg("threads") = 256,
	      py::arg("stream") = 0,
	      py::arg("sync") = false);

	  m.def(
	      "df_pack_lf_block_to_qp_device",
	      [](py::object in_Lf_block,
	         py::object out_Qp,
	         int naux,
	         int nao,
	         int q0,
	         int q_count,
	         int threads,
	         uint64_t stream_ptr,
	         bool sync) {
	        require_threads_multiple_of_32(threads, "df_pack_lf_block_to_qp_device");
	        if (nao < 0 || naux < 0 || q0 < 0 || q_count < 0) throw std::invalid_argument("invalid nao/naux/q0/q_count");
	        if (q0 > naux || q_count > (naux - q0)) throw std::invalid_argument("q0/q_count must satisfy q0+q_count<=naux");
	        auto in = cuda_array_view_from_object(in_Lf_block, "in_Lf_block");
	        auto out = cuda_array_view_from_object(out_Qp, "out_Qp");
	        require_typestr(in, "in_Lf_block", "<f8");
	        require_typestr(out, "out_Qp", "<f8");

	        const int64_t n_in = require_1d(in, "in_Lf_block");
	        const int64_t expected_in =
	            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
	        if (n_in != expected_in) throw std::invalid_argument("in_Lf_block must have shape (q_count*nao*nao,)");

	        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
	        const int64_t expected_out = static_cast<int64_t>(naux) * ntri;
	        const int64_t n_out = require_1d(out, "out_Qp");
	        if (n_out != expected_out) throw std::invalid_argument("out_Qp must have shape (naux*ntri,)");

	        cudaStream_t stream = stream_from_uint(stream_ptr);
	        throw_on_cuda_error(
	            cueri_df_pack_lf_block_to_qp_launch_stream(
	                static_cast<const double*>(in.ptr),
	                static_cast<double*>(out.ptr),
	                static_cast<int>(naux),
	                static_cast<int>(nao),
	                static_cast<int>(q0),
	                static_cast<int>(q_count),
	                stream,
	                threads),
	            "cueri_df_pack_lf_block_to_qp_launch_stream");
	        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
	      },
	      py::arg("in_Lf_block"),
	      py::arg("out_Qp"),
	      py::arg("naux"),
	      py::arg("nao"),
	      py::arg("q0"),
	      py::arg("q_count"),
	      py::arg("threads") = 256,
	      py::arg("stream") = 0,
	      py::arg("sync") = false);

	  m.def(
	      "df_unpack_qp_to_qmn_block_device",
	      [](py::object in_Qp,
	         py::object out_Qmn_block,
         int naux,
         int nao,
         int q0,
         int q_count,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_unpack_qp_to_qmn_block_device");
        if (nao < 0 || naux < 0 || q0 < 0 || q_count < 0) throw std::invalid_argument("invalid nao/naux/q0/q_count");
        if (q0 > naux || q_count > (naux - q0)) throw std::invalid_argument("q0/q_count must satisfy q0+q_count<=naux");
        auto in = cuda_array_view_from_object(in_Qp, "in_Qp");
        auto out = cuda_array_view_from_object(out_Qmn_block, "out_Qmn_block");
        require_typestr(in, "in_Qp", "<f8");
        require_typestr(out, "out_Qmn_block", "<f8");

        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
        const int64_t expected_in = static_cast<int64_t>(naux) * ntri;
        const int64_t n_in = require_1d(in, "in_Qp");
        if (n_in != expected_in) throw std::invalid_argument("in_Qp must have shape (naux*ntri,)");

        const int64_t expected_out =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        const int64_t n_out = require_1d(out, "out_Qmn_block");
        if (n_out != expected_out) throw std::invalid_argument("out_Qmn_block must have shape (q_count*nao*nao,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_unpack_qp_to_qmn_block_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                static_cast<int>(q0),
                static_cast<int>(q_count),
                stream,
                threads),
            "cueri_df_unpack_qp_to_qmn_block_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qp"),
      py::arg("out_Qmn_block"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("q0"),
      py::arg("q_count"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_unpack_qp_to_mnq_device",
      [](py::object in_Qp,
         py::object out_mnQ,
         int naux,
         int nao,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_unpack_qp_to_mnq_device");
        if (nao < 0 || naux < 0) throw std::invalid_argument("nao/naux must be >= 0");
        auto in = cuda_array_view_from_object(in_Qp, "in_Qp");
        auto out = cuda_array_view_from_object(out_mnQ, "out_mnQ");
        require_typestr(in, "in_Qp", "<f8");
        require_typestr(out, "out_mnQ", "<f8");

        const int64_t ntri = static_cast<int64_t>(nao) * static_cast<int64_t>(nao + 1) / 2;
        const int64_t expected_in = static_cast<int64_t>(naux) * ntri;
        const int64_t n_in = require_1d(in, "in_Qp");
        if (n_in != expected_in) throw std::invalid_argument("in_Qp must have shape (naux*ntri,)");

        const int64_t expected_out =
            static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        const int64_t n_out = require_1d(out, "out_mnQ");
        if (n_out != expected_out) throw std::invalid_argument("out_mnQ must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_unpack_qp_to_mnq_launch_stream(
                static_cast<const double*>(in.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                stream,
                threads),
            "cueri_df_unpack_qp_to_mnq_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qp"),
      py::arg("out_mnQ"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

}
