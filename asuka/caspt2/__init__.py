"""Internally contracted CASPT2 (SS/MS/XMS) with analytic gradients.

This module implements the 13-case OpenMolcas formalism for IC-CASPT2
using conventional ERIs on CPU/NumPy.
"""

from asuka.caspt2.driver import caspt2_from_mc
from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.fock import CASPT2Fock, build_caspt2_fock
from asuka.caspt2.multistate import build_heff, diagonalize_heff
from asuka.caspt2.overlap import SBDecomposition, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult, CASPT2GradResult, CASPT2Result
from asuka.caspt2.superindex import CASOrbitals, SuperindexMap, build_superindex
from asuka.caspt2.xms import xms_rotate_states

# Gradient support depends on optional subpackages (e.g. SOC tools). Keep CASPT2
# energies importable even when those extras are unavailable.
try:  # pragma: no cover
    from asuka.caspt2.gradient.driver import caspt2_gradient_from_mc  # type: ignore[import-not-found]
    from asuka.caspt2.gradient.types import CASPT2LagSnapshot  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    caspt2_gradient_from_mc = None  # type: ignore[assignment]
    CASPT2LagSnapshot = None  # type: ignore[assignment]

__all__ = [
    "caspt2_from_mc",
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
    "CASOrbitals",
    "SuperindexMap",
    "build_superindex",
    "xms_rotate_states",
]

if caspt2_gradient_from_mc is not None:
    __all__.append("caspt2_gradient_from_mc")
if CASPT2LagSnapshot is not None:
    __all__.append("CASPT2LagSnapshot")
