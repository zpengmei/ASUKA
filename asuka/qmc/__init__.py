"""Stochastic projector methods (FCIQMC / FCI-FRI) in the CSF (GUGA/DRT) basis.

Top-level exports are intentionally limited to scalable or label-generic APIs.
Legacy small-space helper modules have been removed from the production surface.
"""

from __future__ import annotations

from .estimators import choose_reference_index, projected_energy_ref, rayleigh_energy_ref
from .fcifri import FCIFRIRun, run_fcifri_block, run_fcifri_ground
from .fciqmc import FCIQMCRun, run_fciqmc, update_shift
from .omp import have_openmp, maybe_set_openmp_threads, openmp_max_threads, openmp_set_num_threads
from .initiator import initiator_threshold
from .sparse import (
    SparseVector,
    coalesce_coo_auto_f64,
    coalesce_coo_i32_f64,
    coalesce_coo_i64_f64,
    coalesce_coo_u64_f64,
    sparse_abs_l1_on_support,
    sparse_dot_sorted,
    sparse_lookup_value,
)

_HAVE_DEBUG = False
try:
    from .debug import (
        build_molecule_selected_ci_case,
        build_sparse_trial_from_dense,
        QMCProjectorDiagnostics,
        QMCMoleculeDebugCase,
        run_fcifri_debug,
        run_projector_diagnostics,
    )
    _HAVE_DEBUG = True
except ModuleNotFoundError:
    # Keep core QMC APIs importable when optional debug helpers are absent.
    pass

__all__ = [
    "FCIQMCRun",
    "FCIFRIRun",
    "SparseVector",
    "choose_reference_index",
    "coalesce_coo_auto_f64",
    "coalesce_coo_i32_f64",
    "coalesce_coo_i64_f64",
    "coalesce_coo_u64_f64",
    "have_openmp",
    "initiator_threshold",
    "maybe_set_openmp_threads",
    "openmp_max_threads",
    "openmp_set_num_threads",
    "projected_energy_ref",
    "rayleigh_energy_ref",
    "run_fcifri_ground",
    "run_fcifri_block",
    "run_fciqmc",
    "sparse_abs_l1_on_support",
    "sparse_dot_sorted",
    "sparse_lookup_value",
    "update_shift",
]

if _HAVE_DEBUG:
    __all__.extend(
        [
            "build_molecule_selected_ci_case",
            "build_sparse_trial_from_dense",
            "QMCProjectorDiagnostics",
            "QMCMoleculeDebugCase",
            "run_fcifri_debug",
            "run_projector_diagnostics",
        ]
    )
