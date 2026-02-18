#ifndef CUERI_CUDA_KERNELS_API_H
#define CUERI_CUDA_KERNELS_API_H

#include <cuda_runtime.h>

#include <cstdint>

extern "C" cudaError_t cueri_build_pair_tables_ss_launch_stream(
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* prim_coef,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    int nsp,
    double* pair_eta,
    double* pair_Px,
    double* pair_Py,
    double* pair_Pz,
    double* pair_cK,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_schwarz_ssss_launch_stream(
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    int nsp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* sp_Q,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys);

extern "C" cudaError_t cueri_eri_ssss_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys);

extern "C" cudaError_t cueri_eri_psss_launch_stream(
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
    double* eri_out,  // size ntasks * 3 (A=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_psss_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 3 (A=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_psss_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 3
    int blocks_per_task,
    double* eri_out,  // size ntasks * 3 (A=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppss_launch_stream(
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
    double* eri_out,  // size ntasks * 9 (A=x,y,z; B=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppss_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 9 (A=x,y,z; B=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppss_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 9
    int blocks_per_task,
    double* eri_out,  // size ntasks * 9 (A=x,y,z; B=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_psps_launch_stream(
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
    double* eri_out,  // size ntasks * 9 (A=x,y,z; C=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_psps_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 9 (A=x,y,z; C=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_psps_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 9
    int blocks_per_task,
    double* eri_out,  // size ntasks * 9 (A=x,y,z; C=x,y,z)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppps_launch_stream(
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
    double* eri_out,  // size ntasks * 27 (AB=9, CD=3)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppps_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 27 (AB=9, CD=3)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ppps_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 27
    int blocks_per_task,
    double* eri_out,  // size ntasks * 27 (AB=9, CD=3)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_pppp_launch_stream(
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
    double* eri_out,  // size ntasks * 81 (AB=9, CD=9)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_pppp_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 81 (AB=9, CD=9)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_pppp_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 81
    int blocks_per_task,
    double* eri_out,  // size ntasks * 81 (AB=9, CD=9)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_dsss_launch_stream(
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
    double* eri_out,  // size ntasks * 6 (A=xx,xy,xz,yy,yz,zz)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_dsss_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 6 (A=xx,xy,xz,yy,yz,zz)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_dsss_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 6
    int blocks_per_task,
    double* eri_out,  // size ntasks * 6 (A=xx,xy,xz,yy,yz,zz)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ddss_launch_stream(
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
    double* eri_out,  // size ntasks * 36 (AB=6, CD=6)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ddss_warp_launch_stream(
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
    double* eri_out,  // size ntasks * 36 (AB=6, CD=6)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_eri_ddss_multiblock_launch_stream(
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
    double* partial_sums,  // size ntasks * blocks_per_task * 36
    int blocks_per_task,
    double* eri_out,  // size ntasks * 36 (AB=6, CD=6)
    cudaStream_t stream,
    int threads);

#define CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(NAME, NCOMP)                                                        \
  extern "C" cudaError_t cueri_eri_##NAME##_launch_stream(                                                          \
      const int32_t* task_spAB,                                                                                      \
      const int32_t* task_spCD,                                                                                      \
      int ntasks,                                                                                                    \
      const int32_t* sp_A,                                                                                           \
      const int32_t* sp_B,                                                                                           \
      const int32_t* sp_pair_start,                                                                                  \
      const int32_t* sp_npair,                                                                                       \
      const double* shell_cx,                                                                                        \
      const double* shell_cy,                                                                                        \
      const double* shell_cz,                                                                                        \
      const double* pair_eta,                                                                                        \
      const double* pair_Px,                                                                                         \
      const double* pair_Py,                                                                                         \
      const double* pair_Pz,                                                                                         \
      const double* pair_cK,                                                                                         \
      double* eri_out, /* size ntasks * NCOMP */                                                                     \
      cudaStream_t stream,                                                                                           \
      int threads);                                                                                                  \
                                                                                                                     \
  extern "C" cudaError_t cueri_eri_##NAME##_warp_launch_stream(                                                     \
      const int32_t* task_spAB,                                                                                      \
      const int32_t* task_spCD,                                                                                      \
      int ntasks,                                                                                                    \
      const int32_t* sp_A,                                                                                           \
      const int32_t* sp_B,                                                                                           \
      const int32_t* sp_pair_start,                                                                                  \
      const int32_t* sp_npair,                                                                                       \
      const double* shell_cx,                                                                                        \
      const double* shell_cy,                                                                                        \
      const double* shell_cz,                                                                                        \
      const double* pair_eta,                                                                                        \
      const double* pair_Px,                                                                                         \
      const double* pair_Py,                                                                                         \
      const double* pair_Pz,                                                                                         \
      const double* pair_cK,                                                                                         \
      double* eri_out, /* size ntasks * NCOMP */                                                                     \
      cudaStream_t stream,                                                                                           \
      int threads);                                                                                                  \
                                                                                                                     \
  extern "C" cudaError_t cueri_eri_##NAME##_multiblock_launch_stream(                                               \
      const int32_t* task_spAB,                                                                                      \
      const int32_t* task_spCD,                                                                                      \
      int ntasks,                                                                                                    \
      const int32_t* sp_A,                                                                                           \
      const int32_t* sp_B,                                                                                           \
      const int32_t* sp_pair_start,                                                                                  \
      const int32_t* sp_npair,                                                                                       \
      const double* shell_cx,                                                                                        \
      const double* shell_cy,                                                                                        \
      const double* shell_cz,                                                                                        \
      const double* pair_eta,                                                                                        \
      const double* pair_Px,                                                                                         \
      const double* pair_Py,                                                                                         \
      const double* pair_Pz,                                                                                         \
      const double* pair_cK,                                                                                         \
      double* partial_sums, /* size ntasks * blocks_per_task * NCOMP */                                             \
      int blocks_per_task,                                                                                           \
      double* eri_out, /* size ntasks * NCOMP */                                                                     \
      cudaStream_t stream,                                                                                           \
      int threads)

CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ssdp, 18);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(psds, 18);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(psdp, 54);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(psdd, 108);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ppds, 54);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ppdp, 162);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ppdd, 324);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dsds, 36);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dsdp, 108);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dsdd, 216);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fpss, 30);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fdss, 60);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ffss, 100);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fpps, 90);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fdps, 180);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ffps, 300);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fpds, 180);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fdds, 360);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ffds, 600);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ssfs, 10);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(psfs, 30);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ppfs, 90);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dsfs, 60);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fsfs, 100);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dpfs, 180);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fpfs, 300);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ddfs, 360);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fdfs, 600);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fffs, 1000);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ssgs, 15);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(psgs, 45);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ppgs, 135);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dsgs, 90);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fsgs, 150);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dpgs, 270);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fpgs, 450);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ddgs, 540);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(fdgs, 900);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(ffgs, 1500);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dpdp, 324);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dpdd, 648);
CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS(dddd, 1296);

#undef CUERI_DECLARE_ERI_FIXED_CLASS_LAUNCHERS

extern "C" cudaError_t cueri_eri_rys_generic_launch_stream(
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
    int la,
    int lb,
    int lc,
    int ld,
    double* eri_out,  // size ntasks * nAB * nCD
    cudaStream_t stream,
    int threads);

// Warp-per-task generic kernel for small tiles (general la/lb/lc/ld).
extern "C" cudaError_t cueri_eri_rys_generic_warp_launch_stream(
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
    int la,
    int lb,
    int lc,
    int ld,
    double* eri_out,  // size ntasks * nAB * nCD
    cudaStream_t stream,
    int threads);

// DF-oriented generic kernel for (la,lb,lc,0) classes (ld fixed to 0).
//
// Designed for small nElem regimes where block-wide synchronization dominates
// the reference-oriented generic kernel. Uses a warp-per-task execution model
// with only warp-level synchronization.
extern "C" cudaError_t cueri_eri_rys_df_ld0_warp_launch_stream(
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
    int la,
    int lb,
    int lc,
    double* eri_out,  // size ntasks * nAB * nC (nD=1)
    cudaStream_t stream,
    int threads);

// DF int3c2e kernel that supports non-expanded AO contractions.
//
// Computes X(mu,nu,P) = (mu nu | P) for the tasks encoded as (spAB, spCD) where:
// - spAB indexes AO shell pairs (A,B)
// - spCD indexes (aux_shell, dummy_s) pairs (C,0)
//
// The AO basis uses contraction matrices (nctr per shell) passed via:
//   ao_shell_nctr, ao_shell_coef_start, ao_prim_coef
// where ao_prim_coef stores coefficients (including primitive norms) in prim-major order:
//   coef(shell, iprim, ictr) = ao_prim_coef[shell_coef_start + iprim*shell_nctr + ictr]
//
// Note: current implementation is intentionally DF-specialized (ld=0 only).
extern "C" cudaError_t cueri_df_int3c2e_rys_contracted_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,  // AO: Kab_geom, aux: coef (dummy has exp=0 so Kab=1)
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,  // size nao * nao * naux
    cudaStream_t stream,
    int threads);

// Specialization for common contracted AO bases with nctr<=2.
//
// This has identical semantics to `cueri_df_int3c2e_rys_contracted_launch_stream`, but
// uses smaller per-thread accumulator arrays (CTR_MAX=2) to reduce register pressure.
extern "C" cudaError_t cueri_df_int3c2e_rys_contracted_ctr2_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const int32_t* shell_nprim,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,  // AO: Kab_geom, aux: coef (dummy has exp=0 so Kab=1)
    const int32_t* ao_shell_ao_start,
    const int32_t* ao_shell_nctr,
    const int32_t* ao_shell_coef_start,
    const double* ao_prim_coef,
    const int32_t* aux_shell_ao_start,
    int n_shell_ao,
    int nao,
    int naux,
    int aux_p0_block,
    int la,
    int lb,
    int lc,
    double* X_out,  // size nao * nao * naux
    cudaStream_t stream,
    int threads);

// DF int3c2e derivative contraction (analytic) for expanded (nctr==1) cartesian bases.
//
// Contracts dX/dR against `bar_X_flat` for a fixed AO shell-pair spAB and a batch of aux spCD tasks:
//   X(mu,nu,P) = (mu nu | P)
// where each spCD indexes (aux_shell, dummy_s) pairs (C,0) with ld=0.
//
// Output `out` is a float64 array of shape (ntasks, 3, 3) flattened as (ntasks*9,) with:
//   out[t, center, coord] for center in (A,B,C) and coord in (x,y,z).
//
// Notes
// - This kernel assumes expanded AO and aux bases where primitive coefficients are already folded into `pair_cK`.
// - la/lb/lc are required to determine nroots and cartesian component layouts; caller must ensure they match the tasks.
extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_launch_stream(
    int32_t spAB,
    const int32_t* spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,  // size (nao*nao*naux)
    double* out,               // size (ntasks*9)
    cudaStream_t stream,
    int threads);

// Batched DF int3c2e derivative contraction — processes all AO shell pairs in one (la,lb) class
// × all aux shell pairs in one lq class with a single 2D kernel launch.
// Accumulates gradient contributions directly into grad_dev via atomicAdd (no output buffer).
//
// Grid: dim3(ntasks_cd, n_spAB).  Each block handles one (spAB, spCD) pair.
// spAB_arr[n_spAB]: GPU array of AO shell-pair indices, all sharing the same (la, lb).
// shell_atom[nShellTotal]: combined AO+aux shell-to-atom map (AO shells first, then aux).
// grad_dev[natm*3]: pre-zeroed gradient accumulator; results atomicAdd'd into it.
extern "C" cudaError_t cueri_df_int3c2e_deriv_contracted_cart_allsp_atomgrad_launch_stream(
    const int32_t* spAB_arr,   // [n_spAB]
    int n_spAB,
    const int32_t* spCD,       // [ntasks]
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lb,
    int lc,
    const double* bar_X_flat,  // size (nao*nao*naux)
    const int32_t* shell_atom, // size (nAOshells+nAuxShells)
    double* grad_dev,          // size (natm*3) — atomicAdd target
    cudaStream_t stream,
    int threads);

// DF metric 2c2e derivative contraction (analytic) for expanded (nctr==1) aux bases.
//
// Contracts dV/dR against `bar_V` for a fixed aux spAB and a batch of aux spCD tasks:
//   V(P,Q) = (P | Q) == (P*1 | Q*1)
// where spAB/spCD index (aux_shell, dummy_s) pairs with lb=ld=0.
//
// Output `out` is a float64 array of shape (ntasks, 2, 3) flattened as (ntasks*6,) with:
//   out[t, center, coord] for center in (A=P, C=Q) and coord in (x,y,z).
extern "C" cudaError_t cueri_df_metric_2c2e_deriv_contracted_cart_launch_stream(
    int32_t spAB,
    const int32_t* spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* shell_cx,
    const double* shell_cy,
    const double* shell_cz,
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const int32_t* shell_ao_start,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int nao,
    int naux,
    int la,
    int lc,
    const double* bar_V,  // size (naux*naux)
    double* out,          // size (ntasks*6)
    cudaStream_t stream,
    int threads);

// 4c ERI derivative contraction (analytic) for expanded (nctr==1) cartesian bases.
//
// Contracts d(μν|λσ)/dR against a per-task adjoint tile `bar_eri` (stacked) and outputs
// per-task (4,3) center/coord contributions, flattened as (ntasks*12,).
//
// bar_eri layout must match the value tile layout produced by `cueri_eri_rys_generic_launch_stream`
// for the same (la,lb,lc,ld): row-major over (nAB, nCD) with:
//   row = ia*nB + ib, col = ic*nD + id.
//
// Notes
// - This kernel requires la/lb/lc/ld be constant across the launch; caller should batch by task class.
// - Primitive coefficients must already be folded into `pair_cK` (expanded basis).
extern "C" cudaError_t cueri_eri_rys_generic_deriv_contracted_cart_launch_stream(
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
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int32_t la,
    int32_t lb,
    int32_t lc,
    int32_t ld,
    const double* bar_eri, // size (ntasks * nAB * nCD)
    double* out,           // size (ntasks * 12)
    cudaStream_t stream,
    int32_t threads);

// 4c ERI derivative contraction (analytic) with direct atom-gradient accumulation.
//
// Computes per-task contracted derivatives like
// `cueri_eri_rys_generic_deriv_contracted_cart_launch_stream`, then accumulates
// the 12 center components directly into `grad_out` by shell->atom map:
//   grad_out[atom,xyz] += [Axyz, Bxyz, Cxyz, Dxyz].
//
// `shell_atom` shape: (nShell,), int32
// `grad_out` shape:   (natm*3,), float64 (caller zero-inits / accumulates across chunks)
extern "C" cudaError_t cueri_eri_rys_generic_deriv_contracted_atom_grad_inplace_cart_launch_stream(
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
    const int32_t* shell_prim_start,
    const int32_t* shell_nprim,
    const double* prim_exp,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    int32_t la,
    int32_t lb,
    int32_t lc,
    int32_t ld,
    const double* bar_eri,     // size (ntasks * nAB * nCD)
    const int32_t* shell_atom, // size (nShell,)
    double* grad_out,          // size (natm*3,)
    cudaStream_t stream,
    int32_t threads);


extern "C" cudaError_t cueri_eri_ssss_warp_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys);

extern "C" cudaError_t cueri_eri_ssss_multiblock_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_pair_start,
    const int32_t* sp_npair,
    const double* pair_eta,
    const double* pair_Px,
    const double* pair_Py,
    const double* pair_Pz,
    const double* pair_cK,
    double* partial_sums,
    int blocks_per_task,
    double* eri_out,
    cudaStream_t stream,
    int threads,
    bool use_fast_boys);

extern "C" cudaError_t cueri_count_entries_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    int32_t* counts,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_fill_entry_csr_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* entry_offsets,
    int32_t* cursor,
    int32_t* entry_task,
    int32_t* entry_widx,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_reduce_from_entry_csr_launch_stream(
    const int32_t* entry_offsets,
    int nkey,
    const int32_t* entry_task,
    const int32_t* entry_widx,
    const double* eri_task,
    const double* W,
    double* Out,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_rys_roots_weights_launch_stream(
    const double* T,
    int nT,
    int nroots,
    double* roots_out,   // size nT * nroots
    double* weights_out, // size nT * nroots
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_scatter_df_metric_tiles_launch_stream(
    const double* tile,  // size ntasks * nP * nQ
    const int32_t* p0,   // size ntasks
    const int32_t* q0,   // size ntasks
    int ntasks,
    int naux,
    int nP,
    int nQ,
    double* V_out,  // size naux * naux (row-major)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_scatter_df_int3c2e_tiles_launch_stream(
    const double* tile,  // size ntasks * nAB * nP
    const int32_t* a0,   // size ntasks
    const int32_t* b0,   // size ntasks
    const int32_t* p0,   // size ntasks (relative to current aux block)
    int ntasks,
    int nao,
    int naux,
    int nAB,
    int nB,
    int nP,
    double* X_out,  // size nao * nao * naux (row-major)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_scatter_add_df_yt_tiles_launch_stream(
    const double* tile,  // size ntasks * nops * nP
    const int32_t* p0,   // size ntasks (absolute aux AO index for this aux shell)
    int ntasks,
    int naux,
    int nops,
    int nP,
    double* YT_out,  // size naux * nops (row-major)
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t cueri_scatter_eri_tiles_ordered_launch_stream(
    const int32_t* task_spAB,
    const int32_t* task_spCD,
    int ntasks,
    const int32_t* sp_A,
    const int32_t* sp_B,
    const int32_t* shell_ao_start,
    int nao,
    int nA,
    int nB,
    int nC,
    int nD,
    const double* tile_vals,  // size ntasks * (nA*nB) * (nC*nD)
    double* eri_mat,          // size (nao*nao) * (nao*nao), row-major
    cudaStream_t stream,
    int threads);

#endif  // CUERI_CUDA_KERNELS_API_H
