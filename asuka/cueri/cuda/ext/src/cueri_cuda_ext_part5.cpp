#include "cueri_cuda_ext_common.h"

void cueri_bind_part5(py::module_& m) {
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

  // Spherical bar_X variant of df_int3c2e_deriv_contracted_cart_sp_batch_inplace_device.
  // Takes bar_X in spherical AO basis in Qmn layout and applies cart<-sph inside the kernel.
  m.def(
      "df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_inplace_device",
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
         int nao_sph,
         int la,
         int lb,
         int lc,
         py::object bar_X_sph_Qmn,
         py::object shell_ao_start_sph,
         py::object out,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument(
              "df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_inplace_device requires threads <= 256");
        }
        if (nao <= 0 || naux <= 0 || nao_sph <= 0) throw std::invalid_argument("nao/naux/nao_sph must be > 0");
        if (la < 0 || lb < 0 || lc < 0) throw std::invalid_argument("la/lb/lc must be >= 0");
        if (la > 5 || lb > 5 || lc > 5)
          throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_inplace_device supports only l<=5");

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

        require_typestr(bar, "bar_X_sph_Qmn", "<f8");
        require_typestr(a0s, "shell_ao_start_sph", "<i4");
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

        const int64_t nShellTotal = require_1d(s_a0, "shell_ao_start");
        if (require_1d(a0s, "shell_ao_start_sph") != nShellTotal) {
          throw std::invalid_argument("shell_ao_start_sph must have shape (nShellTotal,)");
        }

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");

        const int64_t bar_len = require_1d(bar, "bar_X_sph_Qmn");
        const int64_t expected_bar =
            static_cast<int64_t>(naux) * static_cast<int64_t>(nao_sph) * static_cast<int64_t>(nao_sph);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_X_sph_Qmn must have shape (naux*nao_sph*nao_sph,)");

        const int64_t out_len = require_1d(out_v, "out");
        const int64_t expected_out = ntasks * 9;
        if (out_len != expected_out) throw std::invalid_argument("out must have shape (ntasks*9,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_sphbar_qmn_launch_stream(
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
                static_cast<int>(nao_sph),
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<const int32_t*>(a0s.ptr),
                static_cast<double*>(out_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_sphbar_qmn_launch_stream");
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
      py::arg("nao_sph"),
      py::arg("la"),
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_sph_Qmn"),
      py::arg("shell_ao_start_sph"),
      py::arg("out"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // Batched variant: one launch for all spAB in a (la,lb) class × all spCD in a lq class.
  // Accumulates into grad_dev (shape natm*3) via atomicAdd — no output buffer needed.
  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device",
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
         int lb,
         int lc,
         py::object bar_X_flat,
         py::object shell_atom,
         py::object grad_dev,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device requires threads <= 256");
        }
        if (nao <= 0 || naux <= 0) throw std::invalid_argument("nao/naux must be > 0");
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

        auto bar = cuda_array_view_from_object(bar_X_flat, "bar_X_flat");
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

	        const bool bar_is_f8 = typestr_matches(bar, "<f8");
	        const bool bar_is_f4 = typestr_matches(bar, "<f4");
	        if (!bar_is_f8 && !bar_is_f4) {
	          throw std::invalid_argument("bar_X_flat must have typestr <f8 or <f4 (got " + bar.typestr + ")");
	        }
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

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");
        (void)require_1d(grad_v, "grad_dev");

        const int64_t bar_len = require_1d(bar, "bar_X_flat");
        const int64_t expected_bar = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
	        if (bar_len != expected_bar) throw std::invalid_argument("bar_X_flat must have shape (nao*nao*naux,)");

	        cudaStream_t stream = stream_from_uint(stream_ptr);
	        if (bar_is_f8) {
	          throw_on_cuda_error(
	              cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream(
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
	                  static_cast<int>(la),
	                  static_cast<int>(lb),
	                  static_cast<int>(lc),
	                  static_cast<const double*>(bar.ptr),
	                  static_cast<const int32_t*>(sh_atom.ptr),
	                  static_cast<double*>(grad_v.ptr),
	                  stream,
	                  threads),
	              "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream");
	        } else {
	          throw_on_cuda_error(
	              cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_f32bar_launch_stream(
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
	                  static_cast<int>(la),
	                  static_cast<int>(lb),
	                  static_cast<int>(lc),
	                  static_cast<const float*>(bar.ptr),
	                  static_cast<const int32_t*>(sh_atom.ptr),
	                  static_cast<double*>(grad_v.ptr),
	                  stream,
	                  threads),
	              "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_f32bar_launch_stream");
	        }
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
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_flat"),
      py::arg("shell_atom"),
      py::arg("grad_dev"),
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // AB-tiled batched variant: reduces A/B atomic pressure by processing multiple spCD
  // tasks per block (cd_tile) and atomically flushing A/B once per block.
  m.def(
      "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device",
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
         int lb,
         int lc,
         py::object bar_X_flat,
         py::object shell_atom,
         py::object grad_dev,
         int cd_tile,
         int threads,
         uint64_t stream_ptr,
         bool sync) {
        require_threads_multiple_of_32(threads, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device");
        if (threads > 256) {
          throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device requires threads <= 256");
        }
        if (cd_tile < 1 || cd_tile > 16) {
          throw std::invalid_argument("df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device requires cd_tile in [1,16]");
        }
        if (nao <= 0 || naux <= 0) throw std::invalid_argument("nao/naux must be > 0");
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

        auto bar = cuda_array_view_from_object(bar_X_flat, "bar_X_flat");
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
        require_typestr(bar, "bar_X_flat", "<f8");
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

        (void)require_1d(cx, "shell_cx");
        (void)require_1d(p_exp, "prim_exp");
        (void)require_1d(eta, "pair_eta");
        (void)require_1d(sh_atom, "shell_atom");
        (void)require_1d(grad_v, "grad_dev");

        const int64_t bar_len = require_1d(bar, "bar_X_flat");
        const int64_t expected_bar = static_cast<int64_t>(nao) * static_cast<int64_t>(nao) * static_cast<int64_t>(naux);
        if (bar_len != expected_bar) throw std::invalid_argument("bar_X_flat must have shape (nao*nao*naux,)");

        cudaStream_t stream = stream_from_uint(stream_ptr);
        throw_on_cuda_error(
            cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_launch_stream(
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
                static_cast<int>(la),
                static_cast<int>(lb),
                static_cast<int>(lc),
                static_cast<const double*>(bar.ptr),
                static_cast<const int32_t*>(sh_atom.ptr),
                static_cast<int>(cd_tile),
                static_cast<double*>(grad_v.ptr),
                stream,
                threads),
            "cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_launch_stream");
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
      py::arg("lb"),
      py::arg("lc"),
      py::arg("bar_X_flat"),
      py::arg("shell_atom"),
      py::arg("grad_dev"),
      py::arg("cd_tile") = 8,
      py::arg("threads") = 256,
      py::arg("stream") = 0,
      py::arg("sync") = true);

  // Batched variant with spherical bar_X adjoint in Qmn layout: avoids materializing the
  // full Cartesian bar_X tensor when mol.cart=False / spherical AOs are used upstream.
}
