"""Selected/Selective CI utilities built on the native GUGA/DRT CSF stack.

Top-level exports are limited to the scalable sparse-row-oracle workflows that
remain supported in production.
"""

from asuka.sci.gpu_cipsi import CIPSITrialSpaceResult, build_cipsi_trials_from_scf, run_cipsi_trials
from asuka.sci.hb_integrals import HeatBathIntegralIndex, build_hb_index, build_hb_index_from_df
from asuka.sci.hb_selection import adaptive_epsilon, heat_bath_select_and_pt2_sparse

__all__ = [
    "adaptive_epsilon",
    "build_cipsi_trials_from_scf",
    "build_hb_index",
    "build_hb_index_from_df",
    "CIPSITrialSpaceResult",
    "heat_bath_select_and_pt2_sparse",
    "HeatBathIntegralIndex",
    "run_cipsi_trials",
]
