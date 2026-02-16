"""Multi-Configuration Self-Consistent Field (MCSCF) methods.

This module provides implementations for various MCSCF methods, including:
- CASCI (Complete Active Space Configuration Interaction)
- CASSCF (Complete Active Space Self-Consistent Field)
- SACASSCF (State-Averaged CASSCF)
- Gradient and Non-adiabatic coupling calculations.

It supports both density fitting (DF) and dense integral strategies, with CPU and GPU acceleration.
"""
from __future__ import annotations

from .casci import (
    CASCIResult,
    casci_orbital_gradient_dense,
    casci_orbital_gradient_df,
    eval_casci_energy_df,
    run_casci,
    run_casci_dense_cpu,
    run_casci_dense_gpu,
    run_casci_df,
    run_casci_df_cpu,
)
from .casscf import (
    CASSCFResult,
    casscf_orbital_gradient_dense,
    casscf_orbital_gradient_df,
    run_casscf,
    run_casscf_dense_cpu,
    run_casscf_df,
)
from .nuc_grad import NucGradResult, casci_nuc_grad, casscf_nuc_grad
from .nuc_grad_df import DFNucGradResult, DFNucGradMultirootResult, casscf_nuc_grad_df, casscf_nuc_grad_df_per_root, casci_nuc_grad_df_relaxed, casci_nuc_grad_df_unrelaxed
from .nac import (
    sacasscf_nonadiabatic_couplings_df,
    sacasscf_nonadiabatic_couplings_df_densez,
    sacasscf_nonadiabatic_couplings_dense,
)
from .orbital_grad import allowed_rotation_mask, orbital_gradient_dense, orbital_gradient_df
from .zvector import (
    MCSCFZVectorResult,
    build_ci_gradient_from_effective_integrals,
    build_orbital_gradient_from_effective_densities,
    ensure_real_ci_rhs,
    effective_active_rdms_from_ci_zvector,
    pack_orbital_gradient,
    prepare_ci_rhs_for_zvector,
    project_ci_rhs_normalized,
    solve_mcscf_zvector,
)
from .ras_gas_mcscf import attach_gas_orbital_rotation_mask, attach_ras_orbital_rotation_mask

__all__ = [
    "CASCIResult",
    "CASSCFResult",
    "MCSCFZVectorResult",
    "run_casci",
    "run_casci_dense_cpu",
    "run_casci_dense_gpu",
    "run_casci_df",
    "run_casci_df_cpu",
    "casci_orbital_gradient_df",
    "casci_orbital_gradient_dense",
    "run_casscf",
    "run_casscf_dense_cpu",
    "run_casscf_df",
    "casscf_orbital_gradient_df",
    "casscf_orbital_gradient_dense",
    "NucGradResult",
    "casci_nuc_grad",
    "casscf_nuc_grad",
    "sacasscf_nonadiabatic_couplings_df",
    "sacasscf_nonadiabatic_couplings_df_densez",
    "sacasscf_nonadiabatic_couplings_dense",
    "DFNucGradResult",
    "DFNucGradMultirootResult",
    "casscf_nuc_grad_df",
    "casscf_nuc_grad_df_per_root",
    "casci_nuc_grad_df_relaxed",
    "casci_nuc_grad_df_unrelaxed",
    "allowed_rotation_mask",
    "orbital_gradient_df",
    "orbital_gradient_dense",
    "solve_mcscf_zvector",
    "build_ci_gradient_from_effective_integrals",
    "build_orbital_gradient_from_effective_densities",
    "ensure_real_ci_rhs",
    "effective_active_rdms_from_ci_zvector",
    "pack_orbital_gradient",
    "prepare_ci_rhs_for_zvector",
    "project_ci_rhs_normalized",
    "attach_gas_orbital_rotation_mask",
    "attach_ras_orbital_rotation_mask",
]
