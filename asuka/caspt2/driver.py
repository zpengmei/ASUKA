"""Deprecated PySCF-based CASPT2 driver (removed).

ASUKA CASPT2 is now driven from ASUKA-native SCF/CASCI/CASSCF objects via:
  - :func:`asuka.caspt2.driver_asuka.caspt2_from_casci`
  - :func:`asuka.caspt2.driver_asuka.caspt2_from_casscf`

The old `caspt2_from_mc` entry point (PySCF CASCI/CASSCF object) has been
removed to keep the CASPT2 stack end-to-end ASUKA and GPU-focused.
"""

from __future__ import annotations

from typing import Any


def caspt2_from_mc(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "PySCF CASPT2 driver was removed. Use asuka.caspt2.caspt2_from_casci / "
        "asuka.caspt2.caspt2_from_casscf, or the ASUKA pipeline helpers under asuka.pipeline."
    )


# ---------------------------------------------------------------------------
# Stubs kept for internal legacy/gradient test monkeypatching.
# ---------------------------------------------------------------------------


def _build_h1e_mo(*args: Any, **kwargs: Any):
    raise RuntimeError("PySCF-based integral builders were removed from asuka.caspt2.driver")


def _build_mo_integrals(*args: Any, **kwargs: Any):
    raise RuntimeError("PySCF-based integral builders were removed from asuka.caspt2.driver")


def _get_ci_and_rdms(*args: Any, **kwargs: Any):
    raise RuntimeError("PySCF-based CI/RDM extraction was removed from asuka.caspt2.driver")


__all__ = ["caspt2_from_mc"]

