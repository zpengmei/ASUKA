#include "cueri_cuda_ext_common.h"

void cueri_bind_part9(py::module_& m) {
  m.def(
      "df_bar_x_sph_to_cart_sym_device",
      [](py::object spAB_arr,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_cart,
         py::object shell_ao_start_sph,
         int nao_cart,
         int nao_sph,
         int naux,
         int la,
         int lb,
         py::object in_sph_mnQ,
         py::object out_cart_mnQ,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_bar_x_sph_to_cart_sym_device");
        if (threads > 256) {
          throw std::invalid_argument("df_bar_x_sph_to_cart_sym_device requires threads <= 256");
        }
        if (nao_cart <= 0 || nao_sph <= 0 || naux <= 0) {
          throw std::invalid_argument("nao_cart/nao_sph/naux must be > 0");
        }
        if (la < 0 || lb < 0) throw std::invalid_argument("la/lb must be >= 0");
        if (la > 5 || lb > 5) throw std::invalid_argument("df_bar_x_sph_to_cart_sym_device supports only l<=5");

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto a0c = cuda_array_view_from_object(shell_ao_start_cart, "shell_ao_start_cart");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto in = cuda_array_view_from_object(in_sph_mnQ, "in_sph_mnQ");
        auto out = cuda_array_view_from_object(out_cart_mnQ, "out_cart_mnQ");

        require_typestr(ab_arr, "spAB_arr", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(a0c, "shell_ao_start_cart", "<i4");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(in, "in_sph_mnQ", "<f8");
        const bool out_is_f8 = typestr_matches(out, "<f8");
        const bool out_is_f4 = typestr_matches(out, "<f4");
        if (!out_is_f8 && !out_is_f4) {
          throw std::invalid_argument("out_cart_mnQ must have typestr <f8 or <f4 (got " + out.typestr + ")");
        }

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        if (n_spAB <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nSP,)");
        }
        (void)require_1d(a0c, "shell_ao_start_cart");
        (void)require_1d(a0s, "shell_ao_start_sph");

        const int64_t in_len = require_1d(in, "in_sph_mnQ");
        const int64_t expected_in =
            static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(naux);
        if (in_len != expected_in) throw std::invalid_argument("in_sph_mnQ must have shape (nao_sph*nao_sph*naux,)");

        const int64_t out_len = require_1d(out, "out_cart_mnQ");
        const int64_t expected_out =
            static_cast<int64_t>(nao_cart) * static_cast<int64_t>(nao_cart) * static_cast<int64_t>(naux);
        if (out_len != expected_out) throw std::invalid_argument("out_cart_mnQ must have shape (nao_cart*nao_cart*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        if (out_is_f8) {
          throw_on_cuda_error(
              cueri_df_bar_x_sph_to_cart_sym_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<double*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_bar_x_sph_to_cart_sym_launch_stream");
        } else {
          throw_on_cuda_error(
              cueri_df_bar_x_sph_to_cart_sym_to_f32_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<float*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_bar_x_sph_to_cart_sym_to_f32_launch_stream");
        }
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_cart"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_cart"),
      py::arg("nao_sph"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("in_sph_mnQ"),
      py::arg("out_cart_mnQ"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_bar_x_sph_qmn_to_cart_sym_device",
      [](py::object spAB_arr,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_cart,
         py::object shell_ao_start_sph,
         int nao_cart,
         int nao_sph,
         int naux,
         int la,
         int lb,
         py::object in_sph_Qmn,
         py::object out_cart_mnQ,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_bar_x_sph_qmn_to_cart_sym_device");
        if (threads > 256) {
          throw std::invalid_argument("df_bar_x_sph_qmn_to_cart_sym_device requires threads <= 256");
        }
        if (nao_cart <= 0 || nao_sph <= 0 || naux <= 0) {
          throw std::invalid_argument("nao_cart/nao_sph/naux must be > 0");
        }
        if (la < 0 || lb < 0) throw std::invalid_argument("la/lb must be >= 0");
        if (la > 5 || lb > 5) throw std::invalid_argument("df_bar_x_sph_qmn_to_cart_sym_device supports only l<=5");

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto a0c = cuda_array_view_from_object(shell_ao_start_cart, "shell_ao_start_cart");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto in = cuda_array_view_from_object(in_sph_Qmn, "in_sph_Qmn");
        auto out = cuda_array_view_from_object(out_cart_mnQ, "out_cart_mnQ");

        require_typestr(ab_arr, "spAB_arr", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(a0c, "shell_ao_start_cart", "<i4");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(in, "in_sph_Qmn", "<f8");
        const bool out_is_f8 = typestr_matches(out, "<f8");
        const bool out_is_f4 = typestr_matches(out, "<f4");
        if (!out_is_f8 && !out_is_f4) {
          throw std::invalid_argument("out_cart_mnQ must have typestr <f8 or <f4 (got " + out.typestr + ")");
        }

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        if (n_spAB <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nSP,)");
        }
        (void)require_1d(a0c, "shell_ao_start_cart");
        (void)require_1d(a0s, "shell_ao_start_sph");

        const int64_t in_len = require_1d(in, "in_sph_Qmn");
        const int64_t expected_in =
            static_cast<int64_t>(naux) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (in_len != expected_in) throw std::invalid_argument("in_sph_Qmn must have shape (naux*nao_sph*nao_sph,)");

        const int64_t out_len = require_1d(out, "out_cart_mnQ");
        const int64_t expected_out =
            static_cast<int64_t>(nao_cart) * static_cast<int64_t>(nao_cart) * static_cast<int64_t>(naux);
        if (out_len != expected_out) throw std::invalid_argument("out_cart_mnQ must have shape (nao_cart*nao_cart*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        if (out_is_f8) {
          throw_on_cuda_error(
              cueri_df_bar_x_sph_qmn_to_cart_sym_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<double*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_bar_x_sph_qmn_to_cart_sym_launch_stream");
        } else {
          throw_on_cuda_error(
              cueri_df_bar_x_sph_qmn_to_cart_sym_to_f32_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<float*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_bar_x_sph_qmn_to_cart_sym_to_f32_launch_stream");
        }
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_cart"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_cart"),
      py::arg("nao_sph"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("in_sph_Qmn"),
      py::arg("out_cart_mnQ"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_B_cart_to_sph_sym_device",
      [](py::object spAB_arr,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_cart,
         py::object shell_ao_start_sph,
         int nao_cart,
         int nao_sph,
         int naux,
         int la,
         int lb,
         py::object in_cart_mnQ,
         py::object out_sph_mnQ,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_B_cart_to_sph_sym_device");
        if (threads > 256) {
          throw std::invalid_argument("df_B_cart_to_sph_sym_device requires threads <= 256");
        }
        if (nao_cart <= 0 || nao_sph <= 0 || naux <= 0) {
          throw std::invalid_argument("nao_cart/nao_sph/naux must be > 0");
        }
        if (la < 0 || lb < 0) throw std::invalid_argument("la/lb must be >= 0");
        if (la > 5 || lb > 5) throw std::invalid_argument("df_B_cart_to_sph_sym_device supports only l<=5");

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto a0c = cuda_array_view_from_object(shell_ao_start_cart, "shell_ao_start_cart");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto in = cuda_array_view_from_object(in_cart_mnQ, "in_cart_mnQ");
        auto out = cuda_array_view_from_object(out_sph_mnQ, "out_sph_mnQ");

        require_typestr(ab_arr, "spAB_arr", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(a0c, "shell_ao_start_cart", "<i4");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(in, "in_cart_mnQ", "<f8");
        const bool out_is_f8 = typestr_matches(out, "<f8");
        const bool out_is_f4 = typestr_matches(out, "<f4");
        if (!out_is_f8 && !out_is_f4) {
          throw std::invalid_argument("out_sph_mnQ must have typestr <f8 or <f4 (got " + out.typestr + ")");
        }

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        if (n_spAB <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nSP,)");
        }
        (void)require_1d(a0c, "shell_ao_start_cart");
        (void)require_1d(a0s, "shell_ao_start_sph");

        const int64_t in_len = require_1d(in, "in_cart_mnQ");
        const int64_t expected_in =
            static_cast<int64_t>(nao_cart) * static_cast<int64_t>(nao_cart) * static_cast<int64_t>(naux);
        if (in_len != expected_in) throw std::invalid_argument("in_cart_mnQ must have shape (nao_cart*nao_cart*naux,)");

        const int64_t out_len = require_1d(out, "out_sph_mnQ");
        const int64_t expected_out =
            static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(naux);
        if (out_len != expected_out) throw std::invalid_argument("out_sph_mnQ must have shape (nao_sph*nao_sph*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        if (out_is_f8) {
          throw_on_cuda_error(
              cueri_df_B_cart_to_sph_sym_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<double*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_B_cart_to_sph_sym_launch_stream");
        } else {
          throw_on_cuda_error(
              cueri_df_B_cart_to_sph_sym_to_f32_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<float*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_B_cart_to_sph_sym_to_f32_launch_stream");
        }
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_cart"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_cart"),
      py::arg("nao_sph"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("in_cart_mnQ"),
      py::arg("out_sph_mnQ"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_B_cart_to_sph_qmn_sym_device",
      [](py::object spAB_arr,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_cart,
         py::object shell_ao_start_sph,
         int nao_cart,
         int nao_sph,
         int naux,
         int la,
         int lb,
         py::object in_cart_mnQ,
         py::object out_sph_Qmn,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_B_cart_to_sph_qmn_sym_device");
        if (threads > 256) {
          throw std::invalid_argument("df_B_cart_to_sph_qmn_sym_device requires threads <= 256");
        }
        if (nao_cart <= 0 || nao_sph <= 0 || naux <= 0) {
          throw std::invalid_argument("nao_cart/nao_sph/naux must be > 0");
        }
        if (la < 0 || lb < 0) throw std::invalid_argument("la/lb must be >= 0");
        if (la > 5 || lb > 5) throw std::invalid_argument("df_B_cart_to_sph_qmn_sym_device supports only l<=5");

        auto ab_arr = cuda_array_view_from_object(spAB_arr, "spAB_arr");
        auto a_sp = cuda_array_view_from_object(sp_A, "sp_A");
        auto b_sp = cuda_array_view_from_object(sp_B, "sp_B");
        auto a0c = cuda_array_view_from_object(shell_ao_start_cart, "shell_ao_start_cart");
        auto a0s = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto in = cuda_array_view_from_object(in_cart_mnQ, "in_cart_mnQ");
        auto out = cuda_array_view_from_object(out_sph_Qmn, "out_sph_Qmn");

        require_typestr(ab_arr, "spAB_arr", "<i4");
        require_typestr(a_sp, "sp_A", "<i4");
        require_typestr(b_sp, "sp_B", "<i4");
        require_typestr(a0c, "shell_ao_start_cart", "<i4");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
        require_typestr(in, "in_cart_mnQ", "<f8");
        const bool out_is_f8 = typestr_matches(out, "<f8");
        const bool out_is_f4 = typestr_matches(out, "<f4");
        if (!out_is_f8 && !out_is_f4) {
          throw std::invalid_argument("out_sph_Qmn must have typestr <f8 or <f4 (got " + out.typestr + ")");
        }

        const int64_t n_spAB = require_1d(ab_arr, "spAB_arr");
        if (n_spAB <= 0) return;

        const int64_t nsp = require_1d(a_sp, "sp_A");
        if (require_1d(b_sp, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nSP,)");
        }
        (void)require_1d(a0c, "shell_ao_start_cart");
        (void)require_1d(a0s, "shell_ao_start_sph");

        const int64_t in_len = require_1d(in, "in_cart_mnQ");
        const int64_t expected_in =
            static_cast<int64_t>(nao_cart) * static_cast<int64_t>(nao_cart) * static_cast<int64_t>(naux);
        if (in_len != expected_in) throw std::invalid_argument("in_cart_mnQ must have shape (nao_cart*nao_cart*naux,)");

        const int64_t out_len = require_1d(out, "out_sph_Qmn");
        const int64_t expected_out =
            static_cast<int64_t>(naux) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (out_len != expected_out) throw std::invalid_argument("out_sph_Qmn must have shape (naux*nao_sph*nao_sph,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        if (out_is_f8) {
          throw_on_cuda_error(
              cueri_df_B_cart_to_sph_qmn_sym_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<double*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_B_cart_to_sph_qmn_sym_launch_stream");
        } else {
          throw_on_cuda_error(
              cueri_df_B_cart_to_sph_qmn_sym_to_f32_launch_stream(
                  static_cast<const int32_t*>(ab_arr.ptr),
                  static_cast<int>(n_spAB),
                  static_cast<const int32_t*>(a_sp.ptr),
                  static_cast<const int32_t*>(b_sp.ptr),
                  static_cast<const int32_t*>(a0c.ptr),
                  static_cast<const int32_t*>(a0s.ptr),
                  static_cast<int>(nao_cart),
                  static_cast<int>(nao_sph),
                  static_cast<int>(naux),
                  static_cast<int>(la),
                  static_cast<int>(lb),
                  static_cast<const double*>(in.ptr),
                  static_cast<float*>(out.ptr),
                  stream,
                  threads),
              "cueri_df_B_cart_to_sph_qmn_sym_to_f32_launch_stream");
        }
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("spAB_arr"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_cart"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_cart"),
      py::arg("nao_sph"),
      py::arg("naux"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("in_cart_mnQ"),
      py::arg("out_sph_Qmn"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);

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

  // Direct J/K contraction: contract ERI tiles with density D → J, K.
  m.def(
      "contract_jk_tiles_ordered_inplace_device",
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
         py::object D_mat,
         py::object J_mat,
         py::object K_mat,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "contract_jk_tiles_ordered_inplace_device");
        if (nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) {
          throw std::invalid_argument("nao/nA/nB/nC/nD must be > 0");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto spa = cuda_array_view_from_object(sp_A, "sp_A");
        auto spb = cuda_array_view_from_object(sp_B, "sp_B");
        auto sh0 = cuda_array_view_from_object(shell_ao_start, "shell_ao_start");
        auto tile = cuda_array_view_from_object(tile_vals, "tile_vals");
        auto d_arr = cuda_array_view_from_object(D_mat, "D_mat");
        auto j_arr = cuda_array_view_from_object(J_mat, "J_mat");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(spa, "sp_A", "<i4");
        require_typestr(spb, "sp_B", "<i4");
        require_typestr(sh0, "shell_ao_start", "<i4");
        require_typestr(tile, "tile_vals", "<f8");
        require_typestr(d_arr, "D_mat", "<f8");
        require_typestr(j_arr, "J_mat", "<f8");

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

        const int64_t nao2 = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (require_1d(d_arr, "D_mat") != nao2) {
          throw std::invalid_argument("D_mat must have shape (nao*nao,)");
        }
        if (require_1d(j_arr, "J_mat") != nao2) {
          throw std::invalid_argument("J_mat must have shape (nao*nao,)");
        }

        // K_mat is optional (pass None → nullptr to skip exchange)
        double* k_ptr = nullptr;
        if (!K_mat.is_none()) {
          auto k_arr = cuda_array_view_from_object(K_mat, "K_mat");
          require_typestr(k_arr, "K_mat", "<f8");
          if (require_1d(k_arr, "K_mat") != nao2) {
            throw std::invalid_argument("K_mat must have shape (nao*nao,)");
          }
          k_ptr = static_cast<double*>(k_arr.ptr);
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_contract_jk_tiles_ordered_launch_stream(
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
                static_cast<const double*>(d_arr.ptr),
                static_cast<double*>(j_arr.ptr),
                k_ptr,
                stream,
                threads),
            "cueri_contract_jk_tiles_ordered_launch_stream");
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
      py::arg("D_mat"),
      py::arg("J_mat"),
      py::arg("K_mat"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // Multi-density direct J/K contraction: contracts tiles with (Da, Db) → (Ja, Ka, Jb, Kb).
  m.def(
      "contract_jk_tiles_ordered_multi2_inplace_device",
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
         py::object Da_mat,
         py::object Db_mat,
         py::object Ja_mat,
         py::object Ka_mat,
         py::object Jb_mat,
         py::object Kb_mat,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "contract_jk_tiles_ordered_multi2_inplace_device");
        if (nao <= 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) {
          throw std::invalid_argument("nao/nA/nB/nC/nD must be > 0");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto spa = cuda_array_view_from_object(sp_A, "sp_A");
        auto spb = cuda_array_view_from_object(sp_B, "sp_B");
        auto sh0 = cuda_array_view_from_object(shell_ao_start, "shell_ao_start");
        auto tile = cuda_array_view_from_object(tile_vals, "tile_vals");
        auto da = cuda_array_view_from_object(Da_mat, "Da_mat");
        auto db = cuda_array_view_from_object(Db_mat, "Db_mat");
        auto ja = cuda_array_view_from_object(Ja_mat, "Ja_mat");
        auto jb = cuda_array_view_from_object(Jb_mat, "Jb_mat");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(spa, "sp_A", "<i4");
        require_typestr(spb, "sp_B", "<i4");
        require_typestr(sh0, "shell_ao_start", "<i4");
        require_typestr(tile, "tile_vals", "<f8");
        require_typestr(da, "Da_mat", "<f8");
        require_typestr(db, "Db_mat", "<f8");
        require_typestr(ja, "Ja_mat", "<f8");
        require_typestr(jb, "Jb_mat", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) {
          throw std::invalid_argument("task_spAB/task_spCD must have identical shape");
        }
        const int64_t nsp = require_1d(spa, "sp_A");
        if (require_1d(spb, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape");
        }
        (void)require_1d(sh0, "shell_ao_start");

        const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
        const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
        const int64_t need_tile = ntasks * nAB * nCD;
        if (require_1d(tile, "tile_vals") != need_tile) {
          throw std::invalid_argument("tile_vals must have shape (ntasks*nAB*nCD,)");
        }

        const int64_t nao2 = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (require_1d(da, "Da_mat") != nao2) throw std::invalid_argument("Da_mat must have shape (nao*nao,)");
        if (require_1d(db, "Db_mat") != nao2) throw std::invalid_argument("Db_mat must have shape (nao*nao,)");
        if (require_1d(ja, "Ja_mat") != nao2) throw std::invalid_argument("Ja_mat must have shape (nao*nao,)");
        if (require_1d(jb, "Jb_mat") != nao2) throw std::invalid_argument("Jb_mat must have shape (nao*nao,)");

        // Ka, Kb optional (pass None → nullptr to skip exchange)
        double* ka_ptr = nullptr;
        if (!Ka_mat.is_none()) {
          auto ka = cuda_array_view_from_object(Ka_mat, "Ka_mat");
          require_typestr(ka, "Ka_mat", "<f8");
          if (require_1d(ka, "Ka_mat") != nao2) throw std::invalid_argument("Ka_mat shape mismatch");
          ka_ptr = static_cast<double*>(ka.ptr);
        }
        double* kb_ptr = nullptr;
        if (!Kb_mat.is_none()) {
          auto kb = cuda_array_view_from_object(Kb_mat, "Kb_mat");
          require_typestr(kb, "Kb_mat", "<f8");
          if (require_1d(kb, "Kb_mat") != nao2) throw std::invalid_argument("Kb_mat shape mismatch");
          kb_ptr = static_cast<double*>(kb.ptr);
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_contract_jk_tiles_ordered_multi2_launch_stream(
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
                static_cast<const double*>(da.ptr),
                static_cast<const double*>(db.ptr),
                static_cast<double*>(ja.ptr),
                ka_ptr,
                static_cast<double*>(jb.ptr),
                kb_ptr,
                stream,
                threads),
            "cueri_contract_jk_tiles_ordered_multi2_launch_stream");
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
      py::arg("Da_mat"),
      py::arg("Db_mat"),
      py::arg("Ja_mat"),
      py::arg("Ka_mat"),
      py::arg("Jb_mat"),
      py::arg("Kb_mat"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // ERI tile cart->sph transform.
  m.def(
      "cart2sph_eri_right_device",
      [](py::object tile_cart,
         py::object tile_tmp,
         int la,
         int lb,
         int lc,
         int ld,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "cart2sph_eri_right_device");
        auto tin = cuda_array_view_from_object(tile_cart, "tile_cart");
        auto tout = cuda_array_view_from_object(tile_tmp, "tile_tmp");
        require_typestr(tin, "tile_cart", "<f8");
        require_typestr(tout, "tile_tmp", "<f8");
        const int64_t n_in = require_1d(tin, "tile_cart");
        const int64_t n_out = require_1d(tout, "tile_tmp");

        if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("negative l is invalid");
        const int64_t nA_cart = (static_cast<int64_t>(la + 1) * static_cast<int64_t>(la + 2)) >> 1;
        const int64_t nB_cart = (static_cast<int64_t>(lb + 1) * static_cast<int64_t>(lb + 2)) >> 1;
        const int64_t nC_cart = (static_cast<int64_t>(lc + 1) * static_cast<int64_t>(lc + 2)) >> 1;
        const int64_t nD_cart = (static_cast<int64_t>(ld + 1) * static_cast<int64_t>(ld + 2)) >> 1;
        const int64_t nAB_cart = nA_cart * nB_cart;
        const int64_t nCD_cart = nC_cart * nD_cart;
        const int64_t nCD_sph = static_cast<int64_t>(2 * lc + 1) * static_cast<int64_t>(2 * ld + 1);
        if (nAB_cart <= 0 || nCD_cart <= 0 || nCD_sph <= 0) throw std::invalid_argument("invalid tile dims");
        const int64_t per_task_in = nAB_cart * nCD_cart;
        if (n_in % per_task_in != 0) throw std::invalid_argument("tile_cart length mismatch for (la,lb,lc,ld)");
        const int64_t ntasks = n_in / per_task_in;
        const int64_t per_task_out = nAB_cart * nCD_sph;
        if (n_out != ntasks * per_task_out) throw std::invalid_argument("tile_tmp length mismatch");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_cart2sph_eri_right_launch_stream(
                static_cast<const double*>(tin.ptr),
                static_cast<double*>(tout.ptr),
                static_cast<int>(ntasks),
                la,
                lb,
                lc,
                ld,
                stream,
                threads),
            "cueri_cart2sph_eri_right_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile_cart"),
      py::arg("tile_tmp"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("ld"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "cart2sph_eri_left_device",
      [](py::object tile_tmp,
         py::object tile_sph,
         int la,
         int lb,
         int lc,
         int ld,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "cart2sph_eri_left_device");
        auto tin = cuda_array_view_from_object(tile_tmp, "tile_tmp");
        auto tout = cuda_array_view_from_object(tile_sph, "tile_sph");
        require_typestr(tin, "tile_tmp", "<f8");
        require_typestr(tout, "tile_sph", "<f8");
        const int64_t n_in = require_1d(tin, "tile_tmp");
        const int64_t n_out = require_1d(tout, "tile_sph");

        if (la < 0 || lb < 0 || lc < 0 || ld < 0) throw std::invalid_argument("negative l is invalid");
        const int64_t nA_cart = (static_cast<int64_t>(la + 1) * static_cast<int64_t>(la + 2)) >> 1;
        const int64_t nB_cart = (static_cast<int64_t>(lb + 1) * static_cast<int64_t>(lb + 2)) >> 1;
        const int64_t nAB_cart = nA_cart * nB_cart;
        const int64_t nAB_sph = static_cast<int64_t>(2 * la + 1) * static_cast<int64_t>(2 * lb + 1);
        const int64_t nCD_sph = static_cast<int64_t>(2 * lc + 1) * static_cast<int64_t>(2 * ld + 1);
        if (nAB_cart <= 0 || nAB_sph <= 0 || nCD_sph <= 0) throw std::invalid_argument("invalid tile dims");
        const int64_t per_task_in = nAB_cart * nCD_sph;
        if (n_in % per_task_in != 0) throw std::invalid_argument("tile_tmp length mismatch for (la,lb,lc,ld)");
        const int64_t ntasks = n_in / per_task_in;
        const int64_t per_task_out = nAB_sph * nCD_sph;
        if (n_out != ntasks * per_task_out) throw std::invalid_argument("tile_sph length mismatch");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_cart2sph_eri_left_launch_stream(
                static_cast<const double*>(tin.ptr),
                static_cast<double*>(tout.ptr),
                static_cast<int>(ntasks),
                la,
                lb,
                lc,
                ld,
                stream,
                threads),
            "cueri_cart2sph_eri_left_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("tile_tmp"),
      py::arg("tile_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("ld"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // Scatter spherical ERI tiles into packed AO formats.
  m.def(
      "scatter_eri_tiles_sph_s8_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_sph,
         int nao_sph,
         int nA,
         int nB,
         int nC,
         int nD,
         py::object tile_vals,
         py::object out_s8,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_eri_tiles_sph_s8_inplace_device");
        if (nao_sph < 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) {
          throw std::invalid_argument("nao_sph must be >= 0 and nA/nB/nC/nD must be > 0");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto spa = cuda_array_view_from_object(sp_A, "sp_A");
        auto spb = cuda_array_view_from_object(sp_B, "sp_B");
        auto sh0 = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto tile = cuda_array_view_from_object(tile_vals, "tile_vals");
        auto out = cuda_array_view_from_object(out_s8, "out_s8");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(spa, "sp_A", "<i4");
        require_typestr(spb, "sp_B", "<i4");
        require_typestr(sh0, "shell_ao_start_sph", "<i4");
        require_typestr(tile, "tile_vals", "<f8");
        require_typestr(out, "out_s8", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) {
          throw std::invalid_argument("task_spAB/task_spCD must have identical shape (ntasks,)");
        }
        const int64_t nsp = require_1d(spa, "sp_A");
        if (require_1d(spb, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nsp,)");
        }
        (void)require_1d(sh0, "shell_ao_start_sph");

        const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
        const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
        const int64_t need_tile = ntasks * nAB * nCD;
        if (require_1d(tile, "tile_vals") != need_tile) {
          throw std::invalid_argument("tile_vals must have shape (ntasks*nAB*nCD,)");
        }

        const int64_t nao_pair = (static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph + 1)) / 2;
        const int64_t need_out = (nao_pair * (nao_pair + 1)) / 2;
        if (require_1d(out, "out_s8") != need_out) {
          throw std::invalid_argument("out_s8 must have shape (nao_pair*(nao_pair+1)/2,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_eri_tiles_sph_s8_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(spa.ptr),
                static_cast<const int32_t*>(spb.ptr),
                static_cast<const int32_t*>(sh0.ptr),
                static_cast<int>(nao_sph),
                static_cast<int>(nA),
                static_cast<int>(nB),
                static_cast<int>(nC),
                static_cast<int>(nD),
                static_cast<const double*>(tile.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_scatter_eri_tiles_sph_s8_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_sph"),
      py::arg("nA"),
      py::arg("nB"),
      py::arg("nC"),
      py::arg("nD"),
      py::arg("tile_vals"),
      py::arg("out_s8"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "scatter_eri_tiles_sph_s4_inplace_device",
      [](py::object task_spAB,
         py::object task_spCD,
         py::object sp_A,
         py::object sp_B,
         py::object shell_ao_start_sph,
         int nao_sph,
         int nA,
         int nB,
         int nC,
         int nD,
         py::object tile_vals,
         py::object out_s4,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "scatter_eri_tiles_sph_s4_inplace_device");
        if (nao_sph < 0 || nA <= 0 || nB <= 0 || nC <= 0 || nD <= 0) {
          throw std::invalid_argument("nao_sph must be >= 0 and nA/nB/nC/nD must be > 0");
        }

        auto ab = cuda_array_view_from_object(task_spAB, "task_spAB");
        auto cd = cuda_array_view_from_object(task_spCD, "task_spCD");
        auto spa = cuda_array_view_from_object(sp_A, "sp_A");
        auto spb = cuda_array_view_from_object(sp_B, "sp_B");
        auto sh0 = cuda_array_view_from_object(shell_ao_start_sph, "shell_ao_start_sph");
        auto tile = cuda_array_view_from_object(tile_vals, "tile_vals");
        auto out = cuda_array_view_from_object(out_s4, "out_s4");

        require_typestr(ab, "task_spAB", "<i4");
        require_typestr(cd, "task_spCD", "<i4");
        require_typestr(spa, "sp_A", "<i4");
        require_typestr(spb, "sp_B", "<i4");
        require_typestr(sh0, "shell_ao_start_sph", "<i4");
        require_typestr(tile, "tile_vals", "<f8");
        require_typestr(out, "out_s4", "<f8");

        const int64_t ntasks = require_1d(ab, "task_spAB");
        if (require_1d(cd, "task_spCD") != ntasks) {
          throw std::invalid_argument("task_spAB/task_spCD must have identical shape (ntasks,)");
        }
        const int64_t nsp = require_1d(spa, "sp_A");
        if (require_1d(spb, "sp_B") != nsp) {
          throw std::invalid_argument("sp_A/sp_B must have identical shape (nsp,)");
        }
        (void)require_1d(sh0, "shell_ao_start_sph");

        const int64_t nAB = static_cast<int64_t>(nA) * static_cast<int64_t>(nB);
        const int64_t nCD = static_cast<int64_t>(nC) * static_cast<int64_t>(nD);
        const int64_t need_tile = ntasks * nAB * nCD;
        if (require_1d(tile, "tile_vals") != need_tile) {
          throw std::invalid_argument("tile_vals must have shape (ntasks*nAB*nCD,)");
        }

        const int64_t nao_pair = (static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph + 1)) / 2;
        const int64_t need_out = nao_pair * nao_pair;
        if (require_1d(out, "out_s4") != need_out) {
          throw std::invalid_argument("out_s4 must have shape (nao_pair*nao_pair,)");
        }

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_scatter_eri_tiles_sph_s4_launch_stream(
                static_cast<const int32_t*>(ab.ptr),
                static_cast<const int32_t*>(cd.ptr),
                static_cast<int>(ntasks),
                static_cast<const int32_t*>(spa.ptr),
                static_cast<const int32_t*>(spb.ptr),
                static_cast<const int32_t*>(sh0.ptr),
                static_cast<int>(nao_sph),
                static_cast<int>(nA),
                static_cast<int>(nB),
                static_cast<int>(nC),
                static_cast<int>(nD),
                static_cast<const double*>(tile.ptr),
                static_cast<double*>(out.ptr),
                stream,
                threads),
            "cueri_scatter_eri_tiles_sph_s4_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("task_spAB"),
      py::arg("task_spCD"),
      py::arg("sp_A"),
      py::arg("sp_B"),
      py::arg("shell_ao_start_sph"),
      py::arg("nao_sph"),
      py::arg("nA"),
      py::arg("nB"),
      py::arg("nC"),
      py::arg("nD"),
      py::arg("tile_vals"),
      py::arg("out_s4"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  m.def(
      "df_fused_qp_l_act_device",
      [](py::object in_Qp,
         py::object in_C_act,
         py::object out_L_act,
         int naux,
         int nao,
         int ncas,
         int ntri,
         int q0,
         int q_count,
         int tile,
         uint64_t stream_ptr,
         bool sync) {
        if (nao < 0 || naux < 0 || ncas < 0 || ntri < 0 || q0 < 0 || q_count < 0)
          throw std::invalid_argument("invalid nao/naux/ncas/ntri/q0/q_count");
        if (tile <= 0 || tile > 32) throw std::invalid_argument("tile must be in [1,32]");
        auto b = cuda_array_view_from_object(in_Qp, "in_Qp");
        auto c = cuda_array_view_from_object(in_C_act, "in_C_act");
        auto l = cuda_array_view_from_object(out_L_act, "out_L_act");
        require_typestr(b, "in_Qp", "<f8");
        require_typestr(c, "in_C_act", "<f8");
        require_typestr(l, "out_L_act", "<f8");

        const int64_t ntri_i64 = static_cast<int64_t>(ntri);
        const int64_t expected_b = static_cast<int64_t>(naux) * ntri_i64;
        if (require_1d(b, "in_Qp") != expected_b)
          throw std::invalid_argument("in_Qp must have shape (naux*ntri,)");
        const int64_t expected_c = static_cast<int64_t>(nao) * static_cast<int64_t>(ncas);
        if (require_1d(c, "in_C_act") != expected_c)
          throw std::invalid_argument("in_C_act must have shape (nao*ncas,)");
        const int64_t expected_l =
            static_cast<int64_t>(q_count) * static_cast<int64_t>(ncas) * static_cast<int64_t>(ncas);
        if (require_1d(l, "out_L_act") != expected_l)
          throw std::invalid_argument("out_L_act must have shape (q_count*ncas*ncas,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_fused_qp_l_act_launch_stream(
                static_cast<const double*>(b.ptr),
                static_cast<const double*>(c.ptr),
                static_cast<double*>(l.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                static_cast<int>(ncas),
                static_cast<int>(ntri),
                static_cast<int>(q0),
                static_cast<int>(q_count),
                stream,
                static_cast<int>(tile)),
            "cueri_df_fused_qp_l_act_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qp"),
      py::arg("in_C_act"),
      py::arg("out_L_act"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("ncas"),
      py::arg("ntri"),
      py::arg("q0"),
      py::arg("q_count"),
      py::arg("tile") = 16,
      py::arg("stream") = 0,
      py::arg("sync") = false);

  m.def(
      "df_fused_qp_exchange_sym_device",
      [](py::object in_Qp,
         py::object in_D1,
         py::object in_D2,
         py::object out_Qp,
         int naux,
         int nao,
         int ntri,
         int q0,
         int q_count,
         double alpha,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_fused_qp_exchange_sym_device");
        if (nao < 0 || naux < 0 || ntri < 0 || q0 < 0 || q_count < 0)
          throw std::invalid_argument("invalid nao/naux/ntri/q0/q_count");
        auto b = cuda_array_view_from_object(in_Qp, "in_Qp");
        auto d1 = cuda_array_view_from_object(in_D1, "in_D1");
        auto d2 = cuda_array_view_from_object(in_D2, "in_D2");
        auto out = cuda_array_view_from_object(out_Qp, "out_Qp");
        require_typestr(b, "in_Qp", "<f8");
        require_typestr(d1, "in_D1", "<f8");
        require_typestr(d2, "in_D2", "<f8");
        require_typestr(out, "out_Qp", "<f8");

        const int64_t ntri_i64 = static_cast<int64_t>(ntri);
        const int64_t expected_b = static_cast<int64_t>(naux) * ntri_i64;
        if (require_1d(b, "in_Qp") != expected_b)
          throw std::invalid_argument("in_Qp must have shape (naux*ntri,)");
        const int64_t expected_d = static_cast<int64_t>(nao) * static_cast<int64_t>(nao);
        if (require_1d(d1, "in_D1") != expected_d)
          throw std::invalid_argument("in_D1 must have shape (nao*nao,)");
        if (require_1d(d2, "in_D2") != expected_d)
          throw std::invalid_argument("in_D2 must have shape (nao*nao,)");
        if (require_1d(out, "out_Qp") != expected_b)
          throw std::invalid_argument("out_Qp must have shape (naux*ntri,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_fused_qp_exchange_sym_launch_stream(
                static_cast<const double*>(b.ptr),
                static_cast<const double*>(d1.ptr),
                static_cast<const double*>(d2.ptr),
                static_cast<double*>(out.ptr),
                static_cast<int>(naux),
                static_cast<int>(nao),
                static_cast<int>(ntri),
                static_cast<int>(q0),
                static_cast<int>(q_count),
                static_cast<double>(alpha),
                stream,
                static_cast<int>(threads)),
            "cueri_df_fused_qp_exchange_sym_launch_stream");
        if (sync) throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      },
      py::arg("in_Qp"),
      py::arg("in_D1"),
      py::arg("in_D2"),
      py::arg("out_Qp"),
      py::arg("naux"),
      py::arg("nao"),
      py::arg("ntri"),
      py::arg("q0"),
      py::arg("q_count"),
      py::arg("alpha") = 1.0,
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = false);
}
