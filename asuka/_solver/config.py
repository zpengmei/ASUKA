from __future__ import annotations

import os
from typing import Any


def auto_num_threads() -> int:
    """Return a process-wide thread hint.

    Prefers explicit environment variables, then falls back to the hardware
    core count.
    """

    for key in ("CUGUGA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = os.environ.get(key)
        if not val:
            continue
        try:
            n = int(val)
        except Exception:
            continue
        if n > 0:
            return n

    return max(1, int(os.cpu_count() or 1))


def resolve_kernel_solver_controls(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
) -> dict[str, float | int]:
    """Resolve and pop common solver-control knobs from kernel kwargs."""

    tol = float(kwargs.pop("tol", getattr(defaults, "conv_tol", 1e-10)))
    lindep = float(kwargs.pop("lindep", getattr(defaults, "lindep", 1e-14)))
    max_cycle = int(kwargs.pop("max_cycle", getattr(defaults, "max_cycle", 100)))
    max_space = int(kwargs.pop("max_space", getattr(defaults, "max_space", 12)))
    max_memory = float(kwargs.pop("max_memory", getattr(defaults, "max_memory", 4000.0)))
    pspace_size = int(kwargs.pop("pspace_size", kwargs.pop("pspace", getattr(defaults, "pspace_size", 0))))
    return {
        "tol": tol,
        "lindep": lindep,
        "max_cycle": max_cycle,
        "max_space": max_space,
        "max_memory": max_memory,
        "pspace_size": pspace_size,
    }


def resolve_kernel_runtime_controls(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
    matvec_backend: str,
    auto_num_threads_fn: Any,
) -> dict[str, Any]:
    """Resolve and pop kernel runtime-control knobs from kwargs/defaults."""

    ne_constraints = kwargs.pop("ne_constraints", getattr(defaults, "ne_constraints", None))
    row_oracle_use_state_cache = bool(kwargs.pop("row_oracle_use_state_cache", False))
    precompute_epq = bool(kwargs.pop("precompute_epq", False))
    if str(matvec_backend) != "contract":
        precompute_epq = False
    max_out = int(kwargs.pop("max_out", 200_000))
    unconverged_fallback_full_diag = bool(
        kwargs.pop(
            "unconverged_fallback_full_diag",
            getattr(defaults, "unconverged_fallback_full_diag", True),
        )
    )
    unconverged_fallback_ncsf_max = int(
        kwargs.pop(
            "unconverged_fallback_ncsf_max",
            getattr(defaults, "unconverged_fallback_ncsf_max", 512),
        )
    )
    if unconverged_fallback_ncsf_max < 1:
        unconverged_fallback_ncsf_max = 1
    raise_on_unconverged = bool(
        kwargs.pop("raise_on_unconverged", getattr(defaults, "raise_on_unconverged", False))
    )
    warn_on_unconverged = bool(
        kwargs.pop("warn_on_unconverged", getattr(defaults, "warn_on_unconverged", True))
    )
    contract_nthreads = int(kwargs.pop("contract_nthreads", int(getattr(defaults, "contract_nthreads", 0))))
    if contract_nthreads <= 0:
        contract_nthreads = int(auto_num_threads_fn())
    contract_blas_nthreads = kwargs.pop("contract_blas_nthreads", getattr(defaults, "contract_blas_nthreads", None))
    if contract_blas_nthreads is not None:
        contract_blas_nthreads = int(contract_blas_nthreads)
        if contract_blas_nthreads <= 0:
            contract_blas_nthreads = None
    if contract_blas_nthreads is None and str(matvec_backend) == "contract":
        contract_blas_nthreads = int(contract_nthreads)
    kernel_blas_nthreads = kwargs.pop("kernel_blas_nthreads", getattr(defaults, "kernel_blas_nthreads", None))
    if kernel_blas_nthreads is not None:
        kernel_blas_nthreads = int(kernel_blas_nthreads)
        if kernel_blas_nthreads <= 0:
            kernel_blas_nthreads = None
    return {
        "ne_constraints": ne_constraints,
        "row_oracle_use_state_cache": row_oracle_use_state_cache,
        "precompute_epq": precompute_epq,
        "max_out": max_out,
        "unconverged_fallback_full_diag": unconverged_fallback_full_diag,
        "unconverged_fallback_ncsf_max": unconverged_fallback_ncsf_max,
        "raise_on_unconverged": raise_on_unconverged,
        "warn_on_unconverged": warn_on_unconverged,
        "contract_nthreads": contract_nthreads,
        "contract_blas_nthreads": contract_blas_nthreads,
        "kernel_blas_nthreads": kernel_blas_nthreads,
    }


def resolve_kernel_frontend_controls(
    *,
    kwargs: dict[str, Any],
    defaults: Any,
) -> dict[str, Any]:
    """Resolve/pull front-end kernel controls from kwargs/defaults."""

    kernel_profile = bool(kwargs.pop("kernel_profile", getattr(defaults, "kernel_profile", False)))
    kernel_profile_cuda_sync = bool(
        kwargs.pop("kernel_profile_cuda_sync", getattr(defaults, "kernel_profile_cuda_sync", False))
    )
    kernel_profile_print = bool(kwargs.pop("kernel_profile_print", getattr(defaults, "kernel_profile_print", False)))
    matvec_cuda_hop_profile = bool(
        kwargs.pop("matvec_cuda_hop_profile", getattr(defaults, "matvec_cuda_hop_profile", False))
    )
    matvec_cuda_davidson_subspace_eigh_cpu_in = kwargs.pop(
        "matvec_cuda_davidson_subspace_eigh_cpu",
        getattr(defaults, "matvec_cuda_davidson_subspace_eigh_cpu", None),
    )
    matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff = int(
        kwargs.pop(
            "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff",
            getattr(defaults, "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff", 100_000_000),
        )
    )
    if matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff < 0:
        matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff = 0
    matvec_cuda_davidson_subspace_eigh_cpu_max_m = int(
        kwargs.pop(
            "matvec_cuda_davidson_subspace_eigh_cpu_max_m",
            getattr(defaults, "matvec_cuda_davidson_subspace_eigh_cpu_max_m", 64),
        )
    )
    if matvec_cuda_davidson_subspace_eigh_cpu_max_m < 0:
        matvec_cuda_davidson_subspace_eigh_cpu_max_m = 0
    dry_run = bool(kwargs.pop("dry_run", False))
    warm_state_enable = bool(kwargs.pop("warm_state_enable", True))
    warm_state_update = bool(kwargs.pop("warm_state_update", True))
    warm_state_context_in = kwargs.pop("warm_state_context", None)
    warm_state_mo_coeff = kwargs.pop("mo_coeff", None)
    warm_state_mo_occ = kwargs.pop("mo_occ", None)
    orbsym = kwargs.pop("orbsym", getattr(defaults, "orbsym", None))
    wfnsym = kwargs.pop("wfnsym", getattr(defaults, "wfnsym", None))
    matvec_backend = str(kwargs.pop("matvec_backend", getattr(defaults, "matvec_backend", "contract"))).strip().lower()
    strict_gpu = bool(kwargs.pop("strict_gpu", getattr(defaults, "strict_gpu", False)))
    return {
        "kernel_profile": kernel_profile,
        "kernel_profile_cuda_sync": kernel_profile_cuda_sync,
        "kernel_profile_print": kernel_profile_print,
        "matvec_cuda_hop_profile": matvec_cuda_hop_profile,
        "matvec_cuda_davidson_subspace_eigh_cpu_in": matvec_cuda_davidson_subspace_eigh_cpu_in,
        "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff": matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff,
        "matvec_cuda_davidson_subspace_eigh_cpu_max_m": matvec_cuda_davidson_subspace_eigh_cpu_max_m,
        "dry_run": dry_run,
        "warm_state_enable": warm_state_enable,
        "warm_state_update": warm_state_update,
        "warm_state_context_in": warm_state_context_in,
        "warm_state_mo_coeff": warm_state_mo_coeff,
        "warm_state_mo_occ": warm_state_mo_occ,
        "orbsym": orbsym,
        "wfnsym": wfnsym,
        "matvec_backend": matvec_backend,
        "strict_gpu": strict_gpu,
    }
