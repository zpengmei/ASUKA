from __future__ import annotations

import sys
from typing import Any

import numpy as np


_COMMON_FLAG_KEYS = (
    "conv_tol",
    "lindep",
    "max_cycle",
    "max_space",
    "pspace_size",
    "max_memory",
)

_RDM_FLAG_KEYS = (
    "rdm_backend",
    "rdm_block_nops",
    "rdm_tmpdir",
    "rdm_cuda_build_threads",
    "rdm_cuda_enable_fp64_emulation",
    "rdm_cuda_gemm_backend",
    "rdm_cuda_math_mode",
    "rdm_cuda_cublas_workspace_mb",
    "rdm_cuda_emulation_strategy",
    "rdm_cuda_fixed_point_mantissa_control",
    "rdm_cuda_fixed_point_max_mantissa_bits",
    "rdm_cuda_fixed_point_mantissa_bit_offset",
    "rdm_cuda_symmetrize_gram",
    "rdm_cuda_streaming_ncsf_cutoff",
)

_MATVEC_FLAG_KEYS = (
    "strict_gpu",
    "matvec_cuda_j_tile",
    "matvec_cuda_epq_build_nthreads",
    "matvec_cuda_epq_build_device",
    "matvec_cuda_epq_build_j_tile",
    "matvec_cuda_epq_streaming",
    "matvec_cuda_epq_stream_j_tile",
    "matvec_cuda_epq_stream_use_recompute",
    "matvec_cuda_fixed_ell_max_ncsf",
    "matvec_cuda_fixed_ell_max_width",
    "matvec_cuda_fixed_ell_row_oracle",
    "matvec_cuda_fixed_ell_threads_spmv",
    "matvec_cuda_csr_capacity_mult",
    "matvec_cuda_csr_host_cache",
    "matvec_cuda_csr_host_cache_budget_gib",
    "matvec_cuda_csr_host_cache_min_ncsf",
    "matvec_cuda_csr_pipeline_streams",
    "matvec_cuda_csr_pipeline_min_ncsf",
    "matvec_cuda_prefilter_trivial_tasks",
    "matvec_cuda_prefilter_trivial_tasks_min_ncsf",
    "matvec_cuda_threads_enum",
    "matvec_cuda_threads_g",
    "matvec_cuda_threads_w",
    "matvec_cuda_threads_apply",
    "matvec_cuda_max_g_mib",
    "matvec_cuda_mem_hard_cap_gib",
    "matvec_cuda_coalesce",
    "matvec_cuda_include_diagonal_rs",
    "matvec_cuda_fuse_count_write",
    "matvec_cuda_path_mode",
    "matvec_cuda_use_fused_hop",
    "matvec_cuda_fp32_coeff_data",
    "matvec_cuda_aggregate_offdiag",
    "matvec_cuda_enable_fp64_emulation",
    "matvec_cuda_gemm_backend",
    "matvec_cuda_emulation_strategy",
    "matvec_cuda_cublas_workspace_cap_mb",
    "matvec_cuda_apply_mode",
    "matvec_cuda_policy",
    "matvec_cuda_accuracy_mode",
    "matvec_cuda_memory_cap_gib",
    "approx_cuda_dtype",
    "matvec_cuda_dtype",
    "matvec_cuda_mixed_threshold",
    "matvec_cuda_mixed_force_final_full_hop",
    "matvec_cuda_mixed_final_full_subspace_refresh",
    "matvec_cuda_mixed_low_precision_max_iter",
    "matvec_cuda_davidson_subspace_eigh_cpu",
    "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff",
    "matvec_cuda_davidson_subspace_eigh_cpu_max_m",
    "matvec_cuda_make_hdiag_cpu",
    "matvec_cuda_make_hdiag_cpu_ncsf_cutoff",
)


def dump_flags(solver: Any, *, verbose: int | None = None) -> Any:
    v = solver.verbose if verbose is None else int(verbose)
    if v <= 0:
        return solver

    out = getattr(solver, "stdout", sys.stdout)
    print("GUGAFCISolver (CSF/GUGA)", file=out)
    print(f"twos = {None if solver.twos is None else int(solver.twos)}", file=out)
    print(f"wfnsym = {None if solver.wfnsym is None else int(solver.wfnsym)}", file=out)
    if solver.orbsym is not None:
        print(f"orbsym = {np.asarray(solver.orbsym, dtype=np.int32).ravel().tolist()}", file=out)

    for key in _COMMON_FLAG_KEYS:
        if hasattr(solver, key):
            print(f"{key} = {getattr(solver, key)}", file=out)

    if hasattr(solver, "kernel_blas_nthreads"):
        print(f"kernel_blas_nthreads = {getattr(solver, 'kernel_blas_nthreads')}", file=out)

    for key in _RDM_FLAG_KEYS:
        if hasattr(solver, key):
            print(f"{key} = {getattr(solver, key)}", file=out)

    for key in _MATVEC_FLAG_KEYS:
        if hasattr(solver, key):
            print(f"{key} = {getattr(solver, key)}", file=out)
    return solver
