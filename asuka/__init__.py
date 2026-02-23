"""ASUKA — GUGA/DRT CSF solver with CUDA acceleration."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

from asuka.cuguga import (
    DRT,
    build_drt,
)
from asuka.active_space import GASSpec, RASSpec, gas_ne_constraints, merge_ne_constraints, ras_ne_constraints
from asuka.cuguga.blas_threads import asuka_thread_limit
from asuka.solver import GUGAFCISolver
from asuka.mcscf.casci import autotune_casci_df as autotune

try:
    __version__ = _dist_version("asuka")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core classes
    "DRT",
    "GUGAFCISolver",
    "GASSpec",
    "RASSpec",
    # Core functions
    "build_drt",
    "gas_ne_constraints",
    "merge_ne_constraints",
    "ras_ne_constraints",
    "asuka_thread_limit",
    "autotune",
]
