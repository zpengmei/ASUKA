#include "cueri_cuda_ext_common.h"

void cueri_bind_part1(py::module_& m) {
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

#if !defined(CUERI_FAST_DEV_DIRECT_JK) && !defined(CUERI_FAST_DEV_STEP2_ONLY)
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
#endif

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

  #ifndef CUERI_FAST_DEV_STEP2_ONLY
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
#endif

  // Mixed-precision ERI bindings for hand-written s/p kernels.
  bind_mixed_precision_eri(m, "psss", 3,
      cueri_eri_psss_launch_stream, cueri_eri_psss_f32_launch_stream,
      cueri_eri_psss_mixed_launch_stream, cueri_eri_psss_mixed_f32_launch_stream,
      cueri_eri_psss_f32accum_launch_stream, cueri_eri_psss_f32accum_f32_launch_stream);
  bind_mixed_precision_eri(m, "ppss", 9,
      cueri_eri_ppss_launch_stream, cueri_eri_ppss_f32_launch_stream,
      cueri_eri_ppss_mixed_launch_stream, cueri_eri_ppss_mixed_f32_launch_stream,
      cueri_eri_ppss_f32accum_launch_stream, cueri_eri_ppss_f32accum_f32_launch_stream);
  bind_mixed_precision_eri(m, "psps", 9,
      cueri_eri_psps_launch_stream, cueri_eri_psps_f32_launch_stream,
      cueri_eri_psps_mixed_launch_stream, cueri_eri_psps_mixed_f32_launch_stream,
      cueri_eri_psps_f32accum_launch_stream, cueri_eri_psps_f32accum_f32_launch_stream);
  bind_mixed_precision_eri(m, "dsss", 6,
      cueri_eri_dsss_launch_stream, cueri_eri_dsss_f32_launch_stream,
      cueri_eri_dsss_mixed_launch_stream, cueri_eri_dsss_mixed_f32_launch_stream,
      cueri_eri_dsss_f32accum_launch_stream, cueri_eri_dsss_f32accum_f32_launch_stream);
  bind_mixed_precision_eri(m, "ppps", 27,
      cueri_eri_ppps_launch_stream, cueri_eri_ppps_f32_launch_stream,
      cueri_eri_ppps_mixed_launch_stream, cueri_eri_ppps_mixed_f32_launch_stream,
      cueri_eri_ppps_f32accum_launch_stream, cueri_eri_ppps_f32accum_f32_launch_stream);
  bind_mixed_precision_eri(m, "pppp", 81,
      cueri_eri_pppp_launch_stream, cueri_eri_pppp_f32_launch_stream,
      cueri_eri_pppp_mixed_launch_stream, cueri_eri_pppp_mixed_f32_launch_stream,
      cueri_eri_pppp_f32accum_launch_stream, cueri_eri_pppp_f32accum_f32_launch_stream);

#ifndef CUERI_FAST_DEV_STEP2_ONLY
  // Mixed-precision ERI bindings for generated d/f/g-shell kernels.
#define BIND_MIXED_GEN(NAME, NC) \
  bind_mixed_precision_eri(m, #NAME, NC, \
      cueri_eri_##NAME##_launch_stream, cueri_eri_##NAME##_f32_launch_stream, \
      cueri_eri_##NAME##_mixed_launch_stream, cueri_eri_##NAME##_mixed_f32_launch_stream)

  BIND_MIXED_GEN(ssdp, 18);
  BIND_MIXED_GEN(psds, 18);
  BIND_MIXED_GEN(psdp, 54);
  BIND_MIXED_GEN(psdd, 108);
  BIND_MIXED_GEN(ppds, 54);
  BIND_MIXED_GEN(ppdp, 162);
  BIND_MIXED_GEN(ppdd, 324);
  BIND_MIXED_GEN(dsds, 36);
  BIND_MIXED_GEN(dsdp, 108);
  BIND_MIXED_GEN(dsdd, 216);
  BIND_MIXED_GEN(dpdp, 324);
  BIND_MIXED_GEN(dpdd, 648);
  BIND_MIXED_GEN(dddd, 1296);
  BIND_MIXED_GEN(ddss, 36);
  BIND_MIXED_GEN(fpss, 30);
  BIND_MIXED_GEN(fdss, 60);
  BIND_MIXED_GEN(ffss, 100);
  BIND_MIXED_GEN(fpps, 90);
  BIND_MIXED_GEN(fdps, 180);
  BIND_MIXED_GEN(ffps, 300);
  BIND_MIXED_GEN(fpds, 180);
  BIND_MIXED_GEN(fdds, 360);
  BIND_MIXED_GEN(ffds, 600);
  BIND_MIXED_GEN(ssfs, 10);
  BIND_MIXED_GEN(psfs, 30);
  BIND_MIXED_GEN(ppfs, 90);
  BIND_MIXED_GEN(dsfs, 60);
  BIND_MIXED_GEN(fsfs, 100);
  BIND_MIXED_GEN(dpfs, 180);
  BIND_MIXED_GEN(fpfs, 300);
  BIND_MIXED_GEN(ddfs, 360);
  BIND_MIXED_GEN(fdfs, 600);
  BIND_MIXED_GEN(fffs, 1000);
  BIND_MIXED_GEN(ssgs, 15);
  BIND_MIXED_GEN(psgs, 45);
  BIND_MIXED_GEN(ppgs, 135);
  BIND_MIXED_GEN(dsgs, 90);
  BIND_MIXED_GEN(fsgs, 150);
  BIND_MIXED_GEN(dpgs, 270);
  BIND_MIXED_GEN(fpgs, 450);
  BIND_MIXED_GEN(ddgs, 540);
  BIND_MIXED_GEN(fdgs, 900);
  BIND_MIXED_GEN(ffgs, 1500);

#undef BIND_MIXED_GEN
#endif

}
