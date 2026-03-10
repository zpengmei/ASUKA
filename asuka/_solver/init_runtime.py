from __future__ import annotations

import os
from typing import Any, Callable


def configure_solver_runtime_defaults(
    solver: Any,
    *,
    normalize_ws_cache_fraction_fn: Callable[[Any], float],
    auto_gpu_mem_hard_cap_fn: Callable[[], float],
    env_get_fn: Callable[[str], Any] | None = None,
) -> None:
    """Initialize runtime/default policy knobs on a solver instance.

    The caller provides normalization helpers so this module stays decoupled
    from solver/cuda-policy import wiring.
    """

    if env_get_fn is None:
        env_get_fn = os.environ.get

    # `0` means "auto": use PySCF's process-wide thread count (`lib.num_threads()`).
    solver.contract_nthreads = 0
    # BLAS thread limit for the dense pair-space GEMMs inside the contract backend.
    # None means "auto" (see kernel()).
    solver.contract_blas_nthreads = None
    # Limit BLAS/OpenMP threads during Davidson iterations. This prevents
    # oversubscription (when the contract backend uses Python threads) and
    # avoids slow multi-threaded BLAS-1/2 kernels on small/moderate CI spaces.
    #
    # None means "auto" (see kernel()).
    solver.kernel_blas_nthreads = getattr(solver, "kernel_blas_nthreads", None)
    # Safety net for pathological Davidson failures: for small CSF spaces we can
    # deterministically recover roots by explicit full-space diagonalization.
    solver.dense_eigh_ncsf_threshold = int(getattr(solver, "dense_eigh_ncsf_threshold", 0))
    solver.unconverged_fallback_full_diag = getattr(solver, "unconverged_fallback_full_diag", True)
    solver.unconverged_fallback_ncsf_max = int(getattr(solver, "unconverged_fallback_ncsf_max", 512))
    if solver.unconverged_fallback_ncsf_max < 1:
        solver.unconverged_fallback_ncsf_max = 1
    solver.raise_on_unconverged = getattr(solver, "raise_on_unconverged", False)
    solver.rdm_backend = getattr(solver, "rdm_backend", "auto")
    # `0` means "auto": use process-wide `lib.num_threads()`.
    solver.rdm_nthreads = getattr(solver, "rdm_nthreads", 0)
    solver.rdm_blas_nthreads = getattr(solver, "rdm_blas_nthreads", None)
    solver.rdm_block_nops = getattr(solver, "rdm_block_nops", 8)
    solver.rdm_tmpdir = getattr(solver, "rdm_tmpdir", None)
    solver.rdm_cuda_build_threads = getattr(solver, "rdm_cuda_build_threads", 256)
    solver.rdm_cuda_enable_fp64_emulation = getattr(solver, "rdm_cuda_enable_fp64_emulation", False)
    solver.rdm_cuda_gemm_backend = getattr(solver, "rdm_cuda_gemm_backend", "gemmex_fp64")
    solver.rdm_cuda_math_mode = getattr(solver, "rdm_cuda_math_mode", "default")
    solver.rdm_cuda_cublas_workspace_mb = getattr(solver, "rdm_cuda_cublas_workspace_mb", 0)
    solver.rdm_cuda_emulation_strategy = getattr(solver, "rdm_cuda_emulation_strategy", None)
    solver.rdm_cuda_fixed_point_mantissa_control = getattr(solver, "rdm_cuda_fixed_point_mantissa_control", None)
    solver.rdm_cuda_fixed_point_max_mantissa_bits = getattr(solver, "rdm_cuda_fixed_point_max_mantissa_bits", None)
    solver.rdm_cuda_fixed_point_mantissa_bit_offset = getattr(solver, "rdm_cuda_fixed_point_mantissa_bit_offset", None)
    solver.rdm_cuda_symmetrize_gram = getattr(solver, "rdm_cuda_symmetrize_gram", True)
    # CUDA RDM streaming policy: force tiled T-matrix builds when ncsf is large.
    # <=0 disables this size-based force and leaves the memory heuristic as-is.
    solver.rdm_cuda_streaming_ncsf_cutoff = getattr(solver, "rdm_cuda_streaming_ncsf_cutoff", 2_000_000)
    solver.matvec_backend = getattr(solver, "matvec_backend", "contract")
    # Strict GPU mode: disallow CPU fallbacks in CUDA workflows.
    solver.strict_gpu = getattr(solver, "strict_gpu", False)
    # High-level CUDA policy:
    # - matvec_cuda_policy: auto/on/off
    # - matvec_cuda_accuracy_mode: fast/balanced/strict
    # - matvec_cuda_memory_cap_gib: alias for matvec_cuda_mem_hard_cap_gib
    solver.matvec_cuda_policy = getattr(solver, "matvec_cuda_policy", "auto")
    solver.matvec_cuda_accuracy_mode = getattr(solver, "matvec_cuda_accuracy_mode", "balanced")
    solver.matvec_cuda_memory_cap_gib = getattr(solver, "matvec_cuda_memory_cap_gib", None)
    # Fraction of the active GPU budget reserved for cached CUDA matvec workspaces.
    # Can be overridden by `ASUKA_GPU_WS_CACHE_FRAC` or per-call kwargs.
    ws_cache_frac_env = env_get_fn("ASUKA_GPU_WS_CACHE_FRAC")
    if ws_cache_frac_env is None:
        ws_cache_frac_env = getattr(solver, "matvec_cuda_ws_cache_fraction", 0.2)
    solver.matvec_cuda_ws_cache_fraction = normalize_ws_cache_fraction_fn(ws_cache_frac_env)
    # PySCF determinant-style FCI uses absorb_h1e(..., fac=0.5) to fold the 1e part into a 2e tensor.
    # For cuGUGA we can optionally bypass that allocation-heavy path and pass (h1e,eri,fac) through as a
    # lightweight wrapper, enabling CUDA matvec without reconstructing (or inverting) the absorbed tensor.
    solver.absorb_h1e_mode = getattr(solver, "absorb_h1e_mode", "tensor")  # "tensor" (default) | "direct"
    # Optional override for `contract_2e` dispatch; when None, falls back to `matvec_backend`.
    solver.contract_2e_backend = getattr(solver, "contract_2e_backend", None)
    # Debug/profiling knob: collect a per-hop breakdown (one-body, CSR build, kernel4, ...)
    # from the CUDA matvec workspace into the kernel profile dict.
    solver.matvec_cuda_hop_profile = getattr(solver, "matvec_cuda_hop_profile", False)
    # CUDA Davidson subspace eigensolve policy:
    # - None: auto (enable CPU path; `subspace_eigh_cpu_max_m` decides CPU vs GPU by subspace size m).
    # - bool: force CPU/GPU subspace eigh in `cuda_davidson`.
    solver.matvec_cuda_davidson_subspace_eigh_cpu = getattr(solver, "matvec_cuda_davidson_subspace_eigh_cpu", None)
    solver.matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff = getattr(
        solver, "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff", 100_000_000
    )
    # For CUDA Davidson CPU-subspace mode, only use host eigh when subspace size m is small.
    solver.matvec_cuda_davidson_subspace_eigh_cpu_max_m = getattr(
        solver, "matvec_cuda_davidson_subspace_eigh_cpu_max_m", 64
    )
    # CUDA make_hdiag policy:
    # - None: auto (CPU for small ncsf, GPU for large ncsf).
    # - bool: force CPU/GPU determinant-diagonal guess build.
    solver.matvec_cuda_make_hdiag_cpu = getattr(solver, "matvec_cuda_make_hdiag_cpu", None)
    solver.matvec_cuda_make_hdiag_cpu_ncsf_cutoff = getattr(solver, "matvec_cuda_make_hdiag_cpu_ncsf_cutoff", 10_000)
    # Approximate CI solver (used by PySCF CASSCF orbital updates via solve_approx_ci)
    # Caps for the approximate CI solve used during CASSCF orbital micro-iterations.
    # When left as None, approx_kernel selects backend-dependent defaults.
    solver.approx_kernel_max_cycle = getattr(solver, "approx_kernel_max_cycle", None)
    solver.approx_kernel_max_space = getattr(solver, "approx_kernel_max_space", None)
    # CUDA matvec precision controls:
    # - approx_cuda_dtype: precision for approx_kernel CUDA hops ("float32" or "float64")
    # - matvec_cuda_dtype: precision mode for full kernel CUDA Davidson
    #   ("float64" | "float32" | "mixed")
    # - matvec_cuda_mixed_threshold: residual threshold for switching mixed mode
    #   from low-precision hop to full-precision hop.
    solver.approx_cuda_dtype = getattr(solver, "approx_cuda_dtype", "float32")
    solver.matvec_cuda_dtype = getattr(solver, "matvec_cuda_dtype", "float64")
    solver.matvec_cuda_mixed_threshold = getattr(solver, "matvec_cuda_mixed_threshold", 1e-5)
    solver.matvec_cuda_mixed_force_final_full_hop = getattr(solver, "matvec_cuda_mixed_force_final_full_hop", True)
    solver.matvec_cuda_mixed_final_full_subspace_refresh = getattr(
        solver, "matvec_cuda_mixed_final_full_subspace_refresh", False
    )
    solver.matvec_cuda_mixed_low_precision_max_iter = getattr(solver, "matvec_cuda_mixed_low_precision_max_iter", 2)
    # Set <=0 to auto-tune j_tile based on norb/ncsf.
    solver.matvec_cuda_j_tile = getattr(solver, "matvec_cuda_j_tile", 0)
    solver.matvec_cuda_target_ntasks = getattr(solver, "matvec_cuda_target_ntasks", 1_500_000)
    solver.matvec_cuda_j_tile_align = getattr(solver, "matvec_cuda_j_tile_align", 256)
    solver.matvec_cuda_csr_capacity_mult = getattr(solver, "matvec_cuda_csr_capacity_mult", 2.0)
    solver.matvec_cuda_csr_host_cache = getattr(solver, "matvec_cuda_csr_host_cache", "auto")
    solver.matvec_cuda_csr_host_cache_budget_gib = getattr(solver, "matvec_cuda_csr_host_cache_budget_gib", 4.0)
    solver.matvec_cuda_csr_host_cache_min_ncsf = getattr(solver, "matvec_cuda_csr_host_cache_min_ncsf", 1_000_000)
    solver.matvec_cuda_csr_pipeline_streams = getattr(solver, "matvec_cuda_csr_pipeline_streams", "auto")
    solver.matvec_cuda_csr_pipeline_min_ncsf = getattr(solver, "matvec_cuda_csr_pipeline_min_ncsf", 1_000_000)
    solver.matvec_cuda_prefilter_trivial_tasks = getattr(solver, "matvec_cuda_prefilter_trivial_tasks", "auto")
    solver.matvec_cuda_prefilter_trivial_tasks_min_ncsf = getattr(
        solver, "matvec_cuda_prefilter_trivial_tasks_min_ncsf", 1_000_000
    )
    solver.matvec_cuda_threads_enum = getattr(solver, "matvec_cuda_threads_enum", 128)
    solver.matvec_cuda_threads_g = getattr(solver, "matvec_cuda_threads_g", 256)
    # Kernel4-side helper kernel for the k-aggregated off-diagonal path (Kernel4W).
    # <=0 means "auto" (currently defaults to threads_g).
    solver.matvec_cuda_threads_w = getattr(solver, "matvec_cuda_threads_w", 0)
    # <=0 means "auto" (chosen in kernel() based on epq_table usage and problem size).
    solver.matvec_cuda_threads_apply = getattr(solver, "matvec_cuda_threads_apply", 0)
    solver.matvec_cuda_max_g_mib = getattr(solver, "matvec_cuda_max_g_mib", 256.0)
    # Hard cap for CUDA private-memory budgeting (<=0 disables cap).
    # Auto policies (EPQ enable, max_g sizing) use this budget to avoid OOM on
    # constrained private-GPU environments.
    solver.matvec_cuda_mem_hard_cap_gib = getattr(
        solver,
        "matvec_cuda_mem_hard_cap_gib",
        auto_gpu_mem_hard_cap_fn(),
    )
    solver.matvec_cuda_coalesce = getattr(solver, "matvec_cuda_coalesce", True)
    solver.matvec_cuda_include_diagonal_rs = getattr(solver, "matvec_cuda_include_diagonal_rs", True)
    solver.matvec_cuda_fuse_count_write = getattr(solver, "matvec_cuda_fuse_count_write", True)
    # CUDA matvec path selector.
    # auto: choose among fused_coo / fused_epq_hybrid / epq_blocked
    solver.matvec_cuda_path_mode = getattr(solver, "matvec_cuda_path_mode", "auto")
    # Controls whether the fused-hop kernel path is allowed when eligible.
    # Set False to force legacy EPQ/CSR paths for benchmarking or debugging.
    solver.matvec_cuda_use_fused_hop = getattr(solver, "matvec_cuda_use_fused_hop", True)
    solver.matvec_cuda_fp32_coeff_data = getattr(solver, "matvec_cuda_fp32_coeff_data", False)
    # CUDA: k-aggregated off-diagonal matvec (can be much faster, but allocates O(ncsf*nops) device memory).
    # None means "auto" (chosen in kernel() based on norb/ncsf and epq_table availability).
    solver.matvec_cuda_aggregate_offdiag = getattr(solver, "matvec_cuda_aggregate_offdiag", None)
    # cuBLAS FP64 fixed-point emulation for the off-diagonal ERI_mat GEMM (CUDA 13.x; aggregate-offdiag path).
    solver.matvec_cuda_enable_fp64_emulation = getattr(solver, "matvec_cuda_enable_fp64_emulation", False)
    # cuBLAS backend for the off-diagonal ERI_mat/DF GEMM path (CUDA 13.x: optional cuBLASLt).
    solver.matvec_cuda_gemm_backend = getattr(solver, "matvec_cuda_gemm_backend", "gemmex_fp64")
    solver.matvec_cuda_emulation_strategy = getattr(solver, "matvec_cuda_emulation_strategy", "performant")
    solver.matvec_cuda_cublas_workspace_cap_mb = getattr(solver, "matvec_cuda_cublas_workspace_cap_mb", 2048)
    # EPQ apply mode for CUDA matvec one-body / table-based apply:
    # "auto" (heuristic), "scatter" (legacy default), or "gather" (forced).
    solver.matvec_cuda_apply_mode = getattr(solver, "matvec_cuda_apply_mode", "auto")
    # CPU threading for building the combined E_pq action table used by the CUDA matvec fast path.
    # <=0 means "auto" (currently caps at 8).
    solver.matvec_cuda_epq_build_nthreads = getattr(solver, "matvec_cuda_epq_build_nthreads", 0)
    # Build the combined E_pq action table on GPU (avoids large CPU setup cost).
    # None means "auto".
    solver.matvec_cuda_epq_build_device = getattr(solver, "matvec_cuda_epq_build_device", None)
    # Tile size for GPU epq_table build; <=0 means "auto" (currently defaults to matvec j_tile).
    solver.matvec_cuda_epq_build_j_tile = getattr(solver, "matvec_cuda_epq_build_j_tile", 0)
    # Stream E_pq tiles on-demand instead of materializing a full table.
    # None/"auto" enables policy-based activation (e.g., large FP32 + mem-cap pressure).
    solver.matvec_cuda_epq_streaming = getattr(solver, "matvec_cuda_epq_streaming", "auto")
    solver.matvec_cuda_epq_stream_j_tile = getattr(solver, "matvec_cuda_epq_stream_j_tile", 0)
    solver.matvec_cuda_epq_stream_use_recompute = getattr(solver, "matvec_cuda_epq_stream_use_recompute", "auto")
    # Fixed sparse matvec backend (build full H once, then apply with fixed-pattern SpMV/SpMM).
    solver.matvec_cuda_fixed_ell_max_ncsf = getattr(solver, "matvec_cuda_fixed_ell_max_ncsf", 50_000)
    solver.matvec_cuda_fixed_ell_max_width = getattr(solver, "matvec_cuda_fixed_ell_max_width", 256)
    solver.matvec_cuda_fixed_ell_row_oracle = getattr(solver, "matvec_cuda_fixed_ell_row_oracle", "sparse")
    solver.matvec_cuda_fixed_ell_threads_spmv = getattr(solver, "matvec_cuda_fixed_ell_threads_spmv", 128)
    solver._warned_row_oracle_df = False
    solver.kernel_profile = getattr(solver, "kernel_profile", False)
    solver.kernel_profile_cuda_sync = getattr(solver, "kernel_profile_cuda_sync", False)
    solver.kernel_profile_print = getattr(solver, "kernel_profile_print", False)
    solver._last_kernel_profile = None
    # Solver-attached warm state (CI-first reuse, plus optional MO/CAS metadata).
    solver._warm_state = None
    solver._last_warm_start_info = None
