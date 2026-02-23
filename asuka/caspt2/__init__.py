"""Internally contracted CASPT2 (SS/MS/XMS) on GPU (DF, C1, FP64).

This module implements the 13-case OpenMolcas formalism for IC-CASPT2
with support for both CPU (NumPy + full ERIs) and GPU (CuPy + DF) backends.
Primary entry points for end-to-end ASUKA workflows are:
  - :func:`run_caspt2` (ASUKA CASCIResult/CASSCFResult)

See ``asuka/caspt2/README.md`` for a detailed description of the
computational workflow, module structure, and conventions.
"""

from asuka.caspt2.driver_asuka import run_caspt2, run_caspt2_soc, run_caspt2_soc_multispin
from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.fock import CASPT2Fock, build_caspt2_fock
from asuka.caspt2.multistate import build_heff, diagonalize_heff
from asuka.caspt2.overlap import SBDecomposition, sbdiag
from asuka.caspt2.result import (
    CASPT2EnergyResult,
    CASPT2GradResult,
    CASPT2Result,
    CASPT2SOCResult,
    CASPT2SOCResultMultiSpin,
)
from asuka.caspt2.superindex import CASOrbitals, SuperindexMap, build_superindex
from asuka.caspt2.xms import xms_rotate_states

__all__ = [
    "run_caspt2",
    "run_caspt2_soc",
    "run_caspt2_soc_multispin",
    "caspt2_energy_ss",
    "CASPT2Fock",
    "build_caspt2_fock",
    "build_heff",
    "diagonalize_heff",
    "SBDecomposition",
    "sbdiag",
    "CASPT2EnergyResult",
    "CASPT2GradResult",
    "CASPT2Result",
    "CASPT2SOCResult",
    "CASPT2SOCResultMultiSpin",
    "CASOrbitals",
    "SuperindexMap",
    "build_superindex",
    "xms_rotate_states",
]
