#include "cueri_cuda_ext_common.h"

void cueri_bind_part7(py::module_& m) {
  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn_chunk,
         py::object shell_ao_start_sph,
         py::object shell_atom,
         int q_offset,
         int q_count,
         py::object grad_dev,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(
            threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_inplace_device requires threads <= 256");
        }
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("supports only l<=5");
        if (q_offset < 0 || q_count <= 0 || q_offset + q_count > naux) {
          throw std::invalid_argument("q_offset/q_count must define a valid range within [0, naux)");
        }

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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

        auto bar = cuda_array_view_from_object(bar_X_sph_Qmn_chunk, "bar_X_sph_Qmn_chunk");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto sh_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto grad_v = cuda_array_view_from_object(grad_dev, "grad_dev");

        require_typestr(ab_arr, "spAB_arr", "<i4");
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
        require_typestr(bar, "bar_X_sph_Qmn_chunk", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(sh_atom, "shell_atom", "<i4");
        require_typestr(grad_v, "grad_dev", "<f8");

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");
        (void)require_1d(grad_v, "grad_dev");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn_chunk");
        const int64_t expected_bar =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_len != expected_bar) {
          throw std::invalid_argument("bar_X_sph_Qmn_chunk must have shape (q_count*nao_sph*nao_sph,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_launch_stream(
                static_cast<const int32_t*>(ab_arr.ptr),
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(q_offset),
                static_cast<int>(q_count),
                static_cast<double*>(grad_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn_chunk"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_atom"),
      py::arg("q_offset"),
      py::arg("q_count"),
      py::arg("grad_dev"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn_chunk_multi,
         int64_t bar_stride,
         int nbar,
         py::object shell_ao_start_sph,
         py::object shell_atom,
         int q_offset,
         int q_count,
         py::object grad_dev_multi,
         int64_t grad_stride,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(
            threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_inplace_device requires threads <= 256");
        }
        if (nbar <= 0) throw std::invalid_argument("nbar must be > 0");
        if (bar_stride <= 0 || grad_stride <= 0) throw std::invalid_argument("bar_stride/grad_stride must be > 0");
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("supports only l<=5");
        if (q_offset < 0 || q_count <= 0 || q_offset + q_count > naux) {
          throw std::invalid_argument("q_offset/q_count must define a valid range within [0, naux)");
        }
        const int64_t min_bar_stride =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_stride < min_bar_stride) {
          throw std::invalid_argument("bar_stride is too small for q_count*nao_sph*nao_sph");
        }

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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

        auto bar = cuda_array_view_from_object(bar_X_sph_Qmn_chunk_multi, "bar_X_sph_Qmn_chunk_multi");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto sh_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto grad_v = cuda_array_view_from_object(grad_dev_multi, "grad_dev_multi");

        require_typestr(ab_arr, "spAB_arr", "<i4");
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
        require_typestr(bar, "bar_X_sph_Qmn_chunk_multi", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(sh_atom, "shell_atom", "<i4");
        require_typestr(grad_v, "grad_dev_multi", "<f8");

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn_chunk_multi");
        const int64_t expected_bar = static_cast<int64_t>(nbar) * static_cast<int64_t>(bar_stride);
        if (bar_len != expected_bar) {
          throw std::invalid_argument("bar_X_sph_Qmn_chunk_multi must have shape (nbar*bar_stride,)");
        }

        const int64_t grad_len = require_1d(grad_v, "grad_dev_multi");
        const int64_t expected_grad = static_cast<int64_t>(nbar) * static_cast<int64_t>(grad_stride);
        if (grad_len != expected_grad) {
          throw std::invalid_argument("grad_dev_multi must have shape (nbar*grad_stride,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_launch_stream(
                static_cast<const int32_t*>(ab_arr.ptr),
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<int64_t>(bar_stride),
                static_cast<int>(nbar),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(q_offset),
                static_cast<int>(q_count),
                static_cast<double*>(grad_v.ptr),
                static_cast<int64_t>(grad_stride),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn_chunk_multi"),
      py::arg("bar_stride"),
      py::arg("nbar"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_atom"),
      py::arg("q_offset"),
      py::arg("q_count"),
      py::arg("grad_dev_multi"),
      py::arg("grad_stride"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn,
         py::object shell_ao_start_sph,
         py::object shell_atom,
         py::object grad_dev,
         int cd_tile,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device requires threads <= 256");
        }
        if (cd_tile < 1 || cd_tile > 16) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device requires cd_tile in [1,16]");
        }
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("supports only l<=5");

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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

        auto bar = cuda_array_view_from_object(bar_X_sph_Qmn, "bar_X_sph_Qmn");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto sh_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto grad_v = cuda_array_view_from_object(grad_dev, "grad_dev");

        require_typestr(ab_arr, "spAB_arr", "<i4");
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
        require_typestr(bar, "bar_X_sph_Qmn", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(sh_atom, "shell_atom", "<i4");
        require_typestr(grad_v, "grad_dev", "<f8");

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");
        (void)require_1d(grad_v, "grad_dev");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn");
        const int64_t expected_bar =
            static_cast<int64_t>(naux) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_len != expected_bar) {
          throw std::invalid_argument("bar_X_sph_Qmn must have shape (naux*nao_sph*nao_sph,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_launch_stream(
                static_cast<const int32_t*>(ab_arr.ptr),
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(cd_tile),
                static_cast<double*>(grad_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_atom"),
      py::arg("grad_dev"),
      py::arg("cd_tile") = 8,
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn_chunk,
         py::object shell_ao_start_sph,
         py::object shell_atom,
         int q_offset,
         int q_count,
         py::object grad_dev,
         int cd_tile,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(
            threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_inplace_device requires threads <= 256");
        }
        if (cd_tile < 1 || cd_tile > 16) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_inplace_device requires cd_tile in [1,16]");
        }
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("supports only l<=5");
        if (q_offset < 0 || q_count <= 0 || q_offset + q_count > naux) {
          throw std::invalid_argument("q_offset/q_count must define a valid range within [0, naux)");
        }

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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

        auto bar = cuda_array_view_from_object(bar_X_sph_Qmn_chunk, "bar_X_sph_Qmn_chunk");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto sh_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto grad_v = cuda_array_view_from_object(grad_dev, "grad_dev");

        require_typestr(ab_arr, "spAB_arr", "<i4");
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
        require_typestr(bar, "bar_X_sph_Qmn_chunk", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(sh_atom, "shell_atom", "<i4");
        require_typestr(grad_v, "grad_dev", "<f8");

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");
        (void)require_1d(grad_v, "grad_dev");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn_chunk");
        const int64_t expected_bar =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_len != expected_bar) {
          throw std::invalid_argument("bar_X_sph_Qmn_chunk must have shape (q_count*nao_sph*nao_sph,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_launch_stream(
                static_cast<const int32_t*>(ab_arr.ptr),
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(q_offset),
                static_cast<int>(q_count),
                static_cast<int>(cd_tile),
                static_cast<double*>(grad_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn_chunk"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_atom"),
      py::arg("q_offset"),
      py::arg("q_count"),
      py::arg("grad_dev"),
      py::arg("cd_tile") = 8,
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn_chunk_multi,
         int64_t bar_stride,
         int nbar,
         py::object shell_ao_start_sph,
         py::object shell_atom,
         int q_offset,
         int q_count,
         int cd_tile,
         py::object grad_dev_multi,
         int64_t grad_stride,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(
            threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_inplace_device requires threads <= 256");
        }
        if (cd_tile < 1 || cd_tile > 16) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_inplace_device requires cd_tile in [1,16]");
        }
        if (nbar <= 0) throw std::invalid_argument("nbar must be > 0");
        if (bar_stride <= 0 || grad_stride <= 0) throw std::invalid_argument("bar_stride/grad_stride must be > 0");
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5) throw std::invalid_argument("supports only l<=5");
        if (q_offset < 0 || q_count <= 0 || q_offset + q_count > naux) {
          throw std::invalid_argument("q_offset/q_count must define a valid range within [0, naux)");
        }
        const int64_t min_bar_stride =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_stride < min_bar_stride) {
          throw std::invalid_argument("bar_stride is too small for q_count*nao_sph*nao_sph");
        }

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
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

        auto bar = cuda_array_view_from_object(bar_X_sph_Qmn_chunk_multi, "bar_X_sph_Qmn_chunk_multi");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto sh_atom = cuda_array_view_from_object(shell_atom, "shell_atom");
        auto grad_v = cuda_array_view_from_object(grad_dev_multi, "grad_dev_multi");

        require_typestr(ab_arr, "spAB_arr", "<i4");
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
        require_typestr(bar, "bar_X_sph_Qmn_chunk_multi", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(sh_atom, "shell_atom", "<i4");
        require_typestr(grad_v, "grad_dev_multi", "<f8");

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        const int64_t ntasks = require_1d(cd, "spCD");
        if (n_spAB <= 0 || ntasks <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp || require_1d(sp_npair_v, "sp_npair") != nsp) {
          throw std::invalid_argument("sp_A/sp_B/sp_npair must have identical shape (nSP,)");
        }
        if (require_1d(sp_start, "sp_pair_start") != (nsp + 1)) {
          throw std::invalid_argument("sp_pair_start must have shape (nSP+1,)");
        }

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn_chunk_multi");
        const int64_t expected_bar = static_cast<int64_t>(nbar) * static_cast<int64_t>(bar_stride);
        if (bar_len != expected_bar) {
          throw std::invalid_argument("bar_X_sph_Qmn_chunk_multi must have shape (nbar*bar_stride,)");
        }

        const int64_t grad_len = require_1d(grad_v, "grad_dev_multi");
        const int64_t expected_grad = static_cast<int64_t>(nbar) * static_cast<int64_t>(grad_stride);
        if (grad_len != expected_grad) {
          throw std::invalid_argument("grad_dev_multi must have shape (nbar*grad_stride,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_launch_stream(
                static_cast<const int32_t*>(ab_arr.ptr),
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<int64_t>(bar_stride),
                static_cast<int>(nbar),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(q_offset),
                static_cast<int>(q_count),
                static_cast<int>(cd_tile),
                static_cast<double*>(grad_v.ptr),
                static_cast<int64_t>(grad_stride),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn_chunk_multi"),
      py::arg("bar_stride"),
      py::arg("nbar"),
      py::arg("shell_ao_start_sph"),
      py::arg("shell_atom"),
      py::arg("q_offset"),
      py::arg("q_count"),
      py::arg("cd_tile") = 8,
      py::arg("grad_dev_multi"),
      py::arg("grad_stride"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "int1e_dS_deriv_contracted_sph_inplace_device",
      [](py::object dS_sph_flat,
         py::object M_sph_flat,
         int natm,
         int nao_sph,
         py::object grad_flat,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "int1e_dS_deriv_contracted_sph_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("int1e_dS_deriv_contracted_sph_inplace_device requires threads <= 256");
        }
        if (natm <= 0 || nao_sph <= 0) throw std::invalid_argument("natm/nao_sph must be > 0");

        auto dS = cuda_array_view_from_object(dS_sph_flat, "dS_sph_flat");
        auto M = cuda_array_view_from_object(M_sph_flat, "M_sph_flat");
        auto g = cuda_array_view_from_object(grad_flat, "grad_flat");

        require_typestr(dS, "dS_sph_flat", "<f8");
        require_typestr(M, "M_sph_flat", "<f8");
        require_typestr(g, "grad_flat", "<f8");

        const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        const int64_t expected_dS = static_cast<int64_t>(natm) * 3 * nao2;
        const int64_t expected_M = nao2;
        const int64_t expected_g = static_cast<int64_t>(natm) * 3;
        if (require_1d(dS, "dS_sph_flat") != expected_dS) {
          throw std::invalid_argument("dS_sph_flat must have shape (natm*3*nao_sph*nao_sph,)");
        }
        if (require_1d(M, "M_sph_flat") != expected_M) {
          throw std::invalid_argument("M_sph_flat must have shape (nao_sph*nao_sph,)");
        }
        if (require_1d(g, "grad_flat") != expected_g) {
          throw std::invalid_argument("grad_flat must have shape (natm*3,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_int1e_dS_deriv_contracted_sph_launch_stream(
                static_cast<const double*>(dS.ptr),
                static_cast<const double*>(M.ptr),
                static_cast<int>(natm),
                static_cast<int>(nao_sph),
                static_cast<double*>(g.ptr),
                stream,
                threads),
            "cueri_int1e_dS_deriv_contracted_sph_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("dS_sph_flat"),
      py::arg("M_sph_flat"),
      py::arg("natm"),
      py::arg("nao_sph"),
      py::arg("grad_flat"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "int1e_dhcore_deriv_contracted_sph_inplace_device",
      [](py::object dT_sph_flat,
         py::object dV_sph_flat,
         py::object M_sph_flat,
         int natm,
         int nao_sph,
         py::object grad_flat,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "int1e_dhcore_deriv_contracted_sph_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("int1e_dhcore_deriv_contracted_sph_inplace_device requires threads <= 256");
        }
        if (natm <= 0 || nao_sph <= 0) throw std::invalid_argument("natm/nao_sph must be > 0");

        auto dT = cuda_array_view_from_object(dT_sph_flat, "dT_sph_flat");
        auto dV = cuda_array_view_from_object(dV_sph_flat, "dV_sph_flat");
        auto M = cuda_array_view_from_object(M_sph_flat, "M_sph_flat");
        auto g = cuda_array_view_from_object(grad_flat, "grad_flat");

        require_typestr(dT, "dT_sph_flat", "<f8");
        require_typestr(dV, "dV_sph_flat", "<f8");
        require_typestr(M, "M_sph_flat", "<f8");
        require_typestr(g, "grad_flat", "<f8");

        const int64_t nao2 = static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        const int64_t expected_d = static_cast<int64_t>(natm) * 3 * nao2;
        const int64_t expected_M = nao2;
        const int64_t expected_g = static_cast<int64_t>(natm) * 3;
        if (require_1d(dT, "dT_sph_flat") != expected_d) {
          throw std::invalid_argument("dT_sph_flat must have shape (natm*3*nao_sph*nao_sph,)");
        }
        if (require_1d(dV, "dV_sph_flat") != expected_d) {
          throw std::invalid_argument("dV_sph_flat must have shape (natm*3*nao_sph*nao_sph,)");
        }
        if (require_1d(M, "M_sph_flat") != expected_M) {
          throw std::invalid_argument("M_sph_flat must have shape (nao_sph*nao_sph,)");
        }
        if (require_1d(g, "grad_flat") != expected_g) {
          throw std::invalid_argument("grad_flat must have shape (natm*3,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_int1e_dhcore_deriv_contracted_sph_launch_stream(
                static_cast<const double*>(dT.ptr),
                static_cast<const double*>(dV.ptr),
                static_cast<const double*>(M.ptr),
                static_cast<int>(natm),
                static_cast<int>(nao_sph),
                static_cast<double*>(g.ptr),
                stream,
                threads),
            "cueri_int1e_dhcore_deriv_contracted_sph_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("dT_sph_flat"),
      py::arg("dV_sph_flat"),
      py::arg("M_sph_flat"),
      py::arg("natm"),
      py::arg("nao_sph"),
      py::arg("grad_flat"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

}
