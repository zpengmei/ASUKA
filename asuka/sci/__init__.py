"""Selected/Selective CI utilities built on the native GUGA/DRT CSF stack.

This subpackage provides a lightweight *selected CI* (SCI) driver which operates
directly in the CSF basis defined by a :class:`asuka.cuguga.drt.DRT`.

The implementation is conservative and reference-oriented: it reuses the existing
sparse row-oracles to assemble the variational Hamiltonian and to generate the
external space used for selection.
"""

from asuka.sci.gpu_cipsi import CIPSITrialSpaceResult, build_cipsi_trials_from_scf, run_cipsi_trials
from asuka.sci.selected_ci import GUGASelectedCISolver, SCIResult, selected_ci

__all__ = [
    "build_cipsi_trials_from_scf",
    "CIPSITrialSpaceResult",
    "GUGASelectedCISolver",
    "run_cipsi_trials",
    "SCIResult",
    "selected_ci",
]
