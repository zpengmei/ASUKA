"""Selected/Selective CI utilities built on the native GUGA/DRT CSF stack.

Top-level exports favor the scalable sparse-row-oracle workflows. Small-space
reference helpers remain available from their defining modules but are not
re-exported here when they are no longer part of the supported production path.
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
