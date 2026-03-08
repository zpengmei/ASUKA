"""Data types for the SST-CASPT2 backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from asuka.caspt2.f3 import CASPT2CIContext
    from asuka.caspt2.fock import CASPT2Fock
    from asuka.caspt2.superindex import SuperindexMap


@dataclass(frozen=True)
class SSTConfig:
    """Configuration for SST-CASPT2 computation."""

    imag_shift: float = 0.0
    real_shift: float = 0.0
    tol: float = 1e-8
    maxiter: int = 200
    threshold: float = 1e-10
    threshold_s: float = 1e-8
    laplace_npts: int = 20
    verbose: int = 0


@dataclass(frozen=True)
class SSTCaseContext:
    """Case-local solve data captured from the native SST equations."""

    case: int
    sector: str
    rhs_sr: np.ndarray
    h0_diag: np.ndarray
    amplitudes_sr: np.ndarray
    energy: float = 0.0
    shift_correction: float = 0.0
    smat: np.ndarray | None = None
    transform: np.ndarray | None = None
    b_diag: np.ndarray | None = None
    nindep: int = 0


@dataclass(frozen=True)
class SSTNativeContext:
    """Native SST solve objects needed by the future analytic gradient path."""

    mode: str
    reduced_cases: dict[int, SSTCaseContext] = field(default_factory=dict)
    hpm_cases: dict[int, SSTCaseContext] = field(default_factory=dict)
    supporting_cases: dict[int, SSTCaseContext] = field(default_factory=dict)
    trailing_cases: dict[int, SSTCaseContext] = field(default_factory=dict)
    dressed: Any | None = None
    terms: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SSTInput:
    """All data the SST backend needs, pre-extracted from ASUKA objects.

    Attributes
    ----------
    ncore, ncas, nvirt : int
        Orbital partition sizes.
    mo_coeff : ndarray (nao, nmo)
        Full MO coefficient matrix.
    dm1_act : ndarray (ncas, ncas)
        Active-space 1-RDM.
    dm2_act : ndarray (ncas, ncas, ncas, ncas)
        Active-space 2-RDM.
    fock : CASPT2Fock
        MO-basis Fock matrices (from build_caspt2_fock_ao).
    semicanonical : SemicanonicalCoreVirt
        Semicanonicalized core/virtual orbitals and energies.
    e_ref : float
        Reference CASSCF energy.
    e_nuc : float
        Nuclear repulsion energy.
    B_ao : ndarray (nao, nao, naux) or None
        AO DF factors.
    eri_mo : ndarray or None
        Full MO ERIs for brute-force validation (Stage 0 only).
    """

    ncore: int
    ncas: int
    nvirt: int
    mo_coeff: np.ndarray
    dm1_act: np.ndarray
    dm2_act: np.ndarray
    fock: CASPT2Fock
    semicanonical: Any  # SemicanonicalCoreVirt | None
    e_ref: float
    e_nuc: float
    dm3_act: np.ndarray | None = None
    ci_context: CASPT2CIContext | None = None
    smap: SuperindexMap | None = None
    B_ao: np.ndarray | None = None
    eri_mo: np.ndarray | None = None


@dataclass(frozen=True)
class SSTResult:
    """Result of SST-CASPT2 energy computation."""

    e_mp2_like: float
    e_active: float
    e_pt2: float
    e_tot: float
    amplitudes_active: np.ndarray | None = None
    pcg_converged: bool = True
    pcg_niter: int = 0
    breakdown: dict[str, Any] = field(default_factory=dict)
    native_context: SSTNativeContext | None = None
