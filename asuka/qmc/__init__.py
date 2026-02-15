"""Stochastic projector methods (FCIQMC / FCI-FRI) in the CSF (GUGA/DRT) basis.

This package is intentionally staged:
- first milestone: CPU-only primitives + validation (sampling + compression),
- later milestones: stochastic H·x spawn engine, full FCIQMC/FCI-FRI loops, CUDA backends.
"""

from __future__ import annotations

from .compress import compress_phi_pivot_resample, compress_phi_pivotal
from .compress_guided import compress_phi_pivot_resample_guided
from .estimators import choose_reference_index, projected_energy_ref, rayleigh_energy_ref
from .epq_sample import EpqSample, sample_epq_from_arrays, sample_epq_one
from .fcifri import FCIFRIRun, FCIFRISubspaceRun, run_fcifri_ground, run_fcifri_subspace
from .fciqmc import FCIQMCRun, run_fciqmc, update_shift
from .omp import have_openmp, maybe_set_openmp_threads, openmp_max_threads, openmp_set_num_threads
from .projector import initiator_threshold, projector_step
from .rsi import FCIFRIRSIResult, run_fcifri_rsi
from .spawn import spawn_hamiltonian_events, spawn_one_body_events, spawn_two_body_events
from .spawn_guided import spawn_hamiltonian_events_guided_row, spawn_hamiltonian_events_guided_thinning
from .sparse import SparseVector, coalesce_coo_i32_f64

__all__ = [
    "EpqSample",
    "FCIQMCRun",
    "FCIFRIRun",
    "FCIFRISubspaceRun",
    "SparseVector",
    "choose_reference_index",
    "coalesce_coo_i32_f64",
    "compress_phi_pivot_resample",
    "compress_phi_pivotal",
    "compress_phi_pivot_resample_guided",
    "FCIFRIRSIResult",
    "have_openmp",
    "initiator_threshold",
    "maybe_set_openmp_threads",
    "openmp_max_threads",
    "openmp_set_num_threads",
    "projector_step",
    "projected_energy_ref",
    "rayleigh_energy_ref",
    "run_fcifri_ground",
    "run_fcifri_rsi",
    "run_fcifri_subspace",
    "run_fciqmc",
    "sample_epq_from_arrays",
    "sample_epq_one",
    "spawn_hamiltonian_events",
    "spawn_hamiltonian_events_guided_row",
    "spawn_hamiltonian_events_guided_thinning",
    "spawn_one_body_events",
    "spawn_two_body_events",
    "update_shift",
]
