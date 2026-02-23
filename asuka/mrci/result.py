from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.integrals.df_integrals import DeviceDFMOIntegrals
from asuka.mrci.ic_mrcisd import ICMRCISDResult
from asuka.mrci.mrcisd import MRCISDResult, MRCISDResultMulti
from asuka.soc.si import SpinFreeState


@dataclass(frozen=True)
class MRCIResult:
    """Result container for a single-root MRCI calculation (ASUKA-native driver)."""

    method: str
    e_ref: float
    e_tot: float
    e_corr: float
    e_tot_plus_q: float | None
    plus_q_diag: dict[str, float] | None
    result: MRCISDResult | ICMRCISDResult
    df_integrals: DeviceDFMOIntegrals | None = None


@dataclass(frozen=True)
class MRCIStatesResult:
    """Result container for multi-root MRCI calculations (ASUKA-native driver)."""

    method: str
    states: list[int]
    nroots: int
    e_ref: np.ndarray
    mrci: MRCISDResultMulti
    ecore: float
    ncore: int
    n_act: int
    n_virt: int
    nelec: int
    twos: int
    df_integrals: DeviceDFMOIntegrals | None = None


@dataclass(frozen=True)
class MRCISOCResult:
    """Result of a SOC-SI calculation on top of a spin-free MRCI result."""

    mrci: MRCIStatesResult
    spinfree_states: list[SpinFreeState]
    so_energies: np.ndarray  # (nss,), float64
    so_vectors: np.ndarray  # (nss,nss), complex128
    so_basis: list[tuple[int, int]]  # (spinfree_state_index, tm=2M)
    h_si: np.ndarray | None = None  # (nss,nss), complex Hermitian (optional debug)

