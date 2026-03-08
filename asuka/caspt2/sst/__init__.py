"""SST-CASPT2 backend for ASUKA.

SST (Supporting Subspace Technique) CASPT2 decomposes the perturbation
equations into (Song, JCP 160, 2024):

  E_PT2 = E^MP2_dressed - E^eff_S + E^eff_T

where:
  * ``E^MP2_dressed``: dressed MP2 energy over all hole×particle pairs
  * ``E^eff_S``: supporting subspace correction (IC cases A, B±, C, D, F±)
  * ``E^eff_T``: trailing subspace correction (IC cases 1-11)

Public API
----------

The main entry point is :func:`~asuka.caspt2.sst.sst_caspt2_energy_ss`.

Two modes are available:
  * ``sst_mode="full"`` (default): Full SST decomposition
  * ``sst_mode="ic"``: Original IC-based split (E_H± + E_cases_1-11)

Both give exact IC-CASPT2 energies.
"""

from __future__ import annotations

from asuka.caspt2.sst.energy_sst import sst_caspt2_energy_ss
from asuka.caspt2.sst.multistate import sst_caspt2_energy_ms, sst_caspt2_energy_ms_cuda
from asuka.caspt2.sst.types import (
    SSTCaseContext,
    SSTConfig,
    SSTInput,
    SSTNativeContext,
    SSTResult,
)

__all__ = [
    "SSTConfig",
    "SSTCaseContext",
    "SSTInput",
    "SSTNativeContext",
    "SSTResult",
    "sst_caspt2_energy_ss",
    "sst_caspt2_energy_ms",
    "sst_caspt2_energy_ms_cuda",
]
