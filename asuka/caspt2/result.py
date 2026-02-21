from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import numpy as np


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
