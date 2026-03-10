"""ASUKA — GUGA/DRT CSF solver with CUDA acceleration."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import Any

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


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DRT": ("asuka.cuguga", "DRT"),
    "build_drt": ("asuka.cuguga", "build_drt"),
    "GASSpec": ("asuka.active_space", "GASSpec"),
    "RASSpec": ("asuka.active_space", "RASSpec"),
    "gas_ne_constraints": ("asuka.active_space", "gas_ne_constraints"),
    "merge_ne_constraints": ("asuka.active_space", "merge_ne_constraints"),
    "ras_ne_constraints": ("asuka.active_space", "ras_ne_constraints"),
    "asuka_thread_limit": ("asuka.cuguga.blas_threads", "asuka_thread_limit"),
    "GUGAFCISolver": ("asuka.solver", "GUGAFCISolver"),
    "autotune": ("asuka.mcscf.casci", "autotune_casci_df"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    modname, attr = target
    value = getattr(import_module(modname), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
