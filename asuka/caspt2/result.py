r"""CASPT2 result data structures.

This module defines frozen dataclasses that encapsulate the outputs of
CASPT2 calculations at various levels of theory:

Result Hierarchy
----------------
- ``CASPT2EnergyResult``: Single-state (SS) CASPT2 energy result.
  Stores the reference energy :math:`E_{\text{ref}}`, the PT2 correlation
  energy :math:`E_{\text{PT2}} = \sum_c \langle V_c | T_c \rangle`, and the
  total energy :math:`E_{\text{tot}} = E_{\text{ref}} + E_{\text{PT2}}`,
  along with the per-case amplitude vectors :math:`T_c`.

- ``CASPT2Result``: Unified SS/MS/XMS result. For multi-state calculations
  this includes the effective Hamiltonian :math:`H_{\text{eff}}` and its
  eigenvectors :math:`U_{\text{eff}}`.

- ``CASPT2SOCResult`` / ``CASPT2SOCResultMultiSpin``: Spin-orbit coupling
  results on top of CASPT2 spin-free states, including spin-orbit energies
  and complex eigenvectors.

- ``CASPT2GradResult``: Nuclear gradient result containing the gradient
  vector, Lagrangian intermediates (CLag, OLag, SLag, WLag), and PT2
  density matrices. Also stores finite-difference validation diagnostics.
"""
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
    nstates: int | None = None
    # PT2 lagrangian objects
    clag: np.ndarray | None = None
    olag: np.ndarray | None = None
    slag: np.ndarray | None = None
    wlag: np.ndarray | None = None
    dpt2_1rdm: np.ndarray | None = None
    dpt2_2rdm: np.ndarray | None = None
    # Validation/diagnostic metadata
    fd_error_abs: float | None = None
    fd_error_rel: float | None = None
    molcas_delta_abs: float | None = None
    molcas_delta_rel: float | None = None
    convergence_flags: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)
