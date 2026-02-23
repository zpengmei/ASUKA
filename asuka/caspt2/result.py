from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import numpy as np

from asuka.soc.si import SpinFreeState


@dataclass(frozen=True)
class CASPT2EnergyResult:
    """Result of a single-state CASPT2 energy calculation."""

    e_ref: float
    e_pt2: float
    e_tot: float
    amplitudes: list[np.ndarray]
    breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CASPT2Result:
    """Result of SS/MS/XMS-CASPT2 calculation."""

    e_ref: float | list[float]
    e_pt2: float | list[float]
    e_tot: float | list[float]
    heff: np.ndarray | None = None
    ueff: np.ndarray | None = None
    amplitudes: list[np.ndarray] | list[list[np.ndarray]] = field(default_factory=list)
    method: str = "SS"
    breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CASPT2SOCResult:
    """Result of a SOC-SI calculation on top of a CASPT2 spin-free result."""

    caspt2: CASPT2Result
    spinfree_states: list[SpinFreeState]
    so_energies: np.ndarray  # (nss,), float64
    so_vectors: np.ndarray  # (nss,nss), complex128
    so_basis: list[tuple[int, int]]  # (spinfree_state_index, tm=2M)
    h_si: np.ndarray | None = None  # (nss,nss), complex Hermitian (optional debug)


@dataclass(frozen=True)
class CASPT2SOCResultMultiSpin:
    """Result of a SOC-SI calculation on top of multiple CASPT2 spin manifolds.

    This is intended for cross-spin SOC mixing (e.g., singlet-triplet) in the
    common-orbital case (same active MO basis).
    """

    caspt2: list[CASPT2Result]
    spinfree_states: list[SpinFreeState]
    spinfree_labels: list[tuple[int, int]]  # (ref_index, root_index_within_ref)
    so_energies: np.ndarray  # (nss,), float64
    so_vectors: np.ndarray  # (nss,nss), complex128
    so_basis: list[tuple[int, int]]  # (spinfree_state_index, tm=2M)
    h_si: np.ndarray | None = None  # (nss,nss), complex Hermitian (optional debug)


@dataclass(frozen=True)
class CASPT2GradResult:
    """Result of CASPT2 nuclear gradient calculation."""

    e_tot: float
    e_ref: float
    e_pt2: float
    grad: np.ndarray
    # Target-state metadata
    method: str = "SS"
    iroot: int = 0
    # PT2 lagrangian objects
    clag: np.ndarray | None = None
    olag: np.ndarray | None = None
    slag: np.ndarray | None = None
    wlag: np.ndarray | None = None
    dpt2_1rdm: np.ndarray | None = None
    dpt2_2rdm: np.ndarray | None = None
    breakdown: dict[str, Any] = field(default_factory=dict)
