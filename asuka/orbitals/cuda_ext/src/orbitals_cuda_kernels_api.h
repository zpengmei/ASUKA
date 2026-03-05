#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {

// --- Becke atom-centered grid helpers ---

void orbitals_build_atom_centered_points_weights_f64(
    const double* center_xyz,   // (3,)
    const double* radial_r,     // (nrad,)
    const double* radial_wr,    // (nrad,)
    int32_t nrad,
    const double* angular_dirs, // (nang,3)
    const double* angular_w,    // (nang,)
    int32_t nang,
    double* pts_local,          // (nrad*nang,3)
    double* w_base,             // (nrad*nang,)
    cudaStream_t stream,
    int32_t threads);

void orbitals_becke_partition_atom_block_f64(
    const double* pts_local,   // (nloc,3)
    const double* w_base,      // (nloc,)
    int32_t nloc,
    const double* atom_coords, // (natm,3)
    const double* RAB,         // (natm,natm)
    int32_t natm,
    int32_t atom_index,
    int32_t becke_n,
    double* rA,                // (nloc,natm)
    double* w_raw,             // (nloc,natm)
    double* w_atom,            // (nloc,)
    cudaStream_t stream,
    int32_t threads);

// --- AO evaluation ---

void orbitals_eval_aos_cart_value_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    int32_t nao,
    const double* points,
    int32_t npt,
    double* ao,
    cudaStream_t stream);

// --- MO evaluation ---

void orbitals_eval_mos_cart_value_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    cudaStream_t stream);

void orbitals_eval_mos_cart_value_grad_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    double* psi_grad,
    cudaStream_t stream);

void orbitals_eval_mos_cart_value_grad_lapl_f64(
    const double* shell_cxyz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_l,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* prim_coef,
    int32_t nshell,
    const double* C_occ,
    int32_t nao,
    int32_t nocc,
    const double* points,
    int32_t npt,
    double* psi,
    double* psi_grad,
    double* psi_lapl,
    cudaStream_t stream);

// --- rho/tau/lapl parts ---

void orbitals_eval_rho_parts_f64(
    const double* psi,
    const double* psi_grad,  // (npt,nocc,3) or nullptr if need_grad=false
    const double* psi_lapl,  // (npt,nocc) or nullptr if need_lapl=false
    const double* dm1,       // (ncas,ncas)
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t need_grad,
    int32_t need_lapl,
    int32_t compute_tau,
    double* rho,               // (npt,)
    double* rho_grad,          // (npt,3) or nullptr if deriv==0
    double* tau,               // (npt,) or nullptr if compute_tau==0
    double* rho_lapl,          // (npt,) or nullptr if compute_rho_laplacian==0
    double* rho_core,          // (npt,)
    double* rho_act,           // (npt,)
    double* rho_core_grad,     // (npt,3) or nullptr if need_grad==0
    double* rho_act_grad,      // (npt,3) or nullptr if need_grad==0
    double* rho_core_lapl,     // (npt,) or nullptr if need_lapl==0
    double* rho_act_lapl,      // (npt,) or nullptr if need_lapl==0
    cudaStream_t stream,
    int32_t threads);

// --- pair buffers ---

void orbitals_build_pair_x_f64(
    const double* psi,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pair_buf,  // (npt,ncas^2)
    cudaStream_t stream,
    int32_t threads);

void orbitals_build_pair_y_f64(
    const double* psi,
    const double* psi_grad,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    int32_t comp,      // 0|1|2
    double* pair_buf,  // (npt,ncas^2)
    cudaStream_t stream,
    int32_t threads);

// --- pi outputs ---

void orbitals_eval_pi_f64(
    const double* rho_core,
    const double* rho_act,
    const double* pair_buf,  // x_flat
    const double* g_buf,     // dm2*x_flat
    int32_t npt,
    int32_t n2,
    double* pi,  // (npt,)
    cudaStream_t stream,
    int32_t threads);

void orbitals_eval_pi_grad_f64(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* psi,
    const double* psi_grad,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_grad,  // (npt,3)
    cudaStream_t stream,
    int32_t threads);

void orbitals_eval_pi_lapl_base_f64(
    const double* rho_core,
    const double* rho_act,
    const double* rho_core_grad,
    const double* rho_act_grad,
    const double* rho_core_lapl,
    const double* rho_act_lapl,
    const double* psi,
    const double* psi_grad,
    const double* psi_lapl,
    const double* g_buf,
    int32_t ncore,
    int32_t ncas,
    int32_t npt,
    int32_t nocc,
    double* pi_lapl,  // (npt,) output overwritten
    cudaStream_t stream,
    int32_t threads);

void orbitals_pi_lapl_add_termB_f64(
    const double* pair_buf,  // y_flat
    const double* gy_buf,    // dm2*y_flat
    int32_t npt,
    int32_t n2,
    double* pi_lapl,  // (npt,) incremented
    cudaStream_t stream,
    int32_t threads);
}
