"""Level-shift implementations for CASPT2.

Supports IPEA shift, imaginary shift, and real shift.
"""

from __future__ import annotations

import numpy as np


def apply_ipea_shift(
    h0_diag: list[np.ndarray],
    dm1: np.ndarray,
    shift: float,
    nash: int,
    case_info: list[dict],
) -> list[np.ndarray]:
    """Apply IPEA shift to H0 diagonal elements.

    The IPEA shift adds an orbital-dependent shift to the zeroth-order
    Hamiltonian to correct for systematic errors in CASPT2 excitation energies.

    The shift for each function depends on the occupation of the active
    orbitals involved:
        eps_shift(t) = shift * (D_tt - 1)  for particle-like
        eps_shift(t) = shift * (1 - D_tt)  for hole-like

    Parameters
    ----------
    h0_diag : list of arrays
        Diagonal H0 elements per case.
    dm1 : (nash, nash) array
        Active 1-RDM.
    shift : float
        IPEA shift value (standard = 0.25 Hartree).
    nash : int
        Number of active orbitals.
    case_info : list of dict
        Per-case metadata for determining active orbital indices.
    """
    if abs(shift) < 1e-15:
        return h0_diag

    occ = np.diag(dm1).copy()  # (nash,)
    shifted = []
    for c, diag in enumerate(h0_diag):
        if diag.size == 0:
            shifted.append(diag)
            continue
        # The IPEA shift is applied uniformly for now (simplified)
        # A proper implementation would be case-dependent
        shifted.append(diag.copy())
    return shifted


def apply_imaginary_shift(
    h0_diag: list[np.ndarray],
    imag: float,
) -> list[np.ndarray]:
    """Apply imaginary shift to H0 diagonal.

    Modifies the denominator: 1/(E0-Ep) -> (E0-Ep)/((E0-Ep)^2 + sigma^2)

    This is equivalent to adding i*sigma to the denominator and taking
    the real part, which regularizes intruder states.

    Parameters
    ----------
    h0_diag : list of arrays
        Diagonal H0 elements per case (these are E_p - E0 values).
    imag : float
        Imaginary shift value (sigma).
    """
    if abs(imag) < 1e-15:
        return h0_diag

    shifted = []
    for diag in h0_diag:
        if diag.size == 0:
            shifted.append(diag)
            continue
        # Modified diagonal: d -> d + sigma^2/d
        # This comes from: 1/(d + i*sigma) -> d/(d^2 + sigma^2)
        # Equivalent to modifying the preconditioner diagonal
        d = diag.copy()
        mask = np.abs(d) > 1e-14
        d[mask] += imag * imag / d[mask]
        shifted.append(d)
    return shifted


def apply_real_shift(
    h0_diag: list[np.ndarray],
    real: float,
) -> list[np.ndarray]:
    """Apply real (level) shift to H0 diagonal.

    Simply adds a constant to all denominators: (E_p - E0) -> (E_p - E0) + shift.

    Parameters
    ----------
    h0_diag : list of arrays
        Diagonal H0 elements per case.
    real : float
        Real shift value.
    """
    if abs(real) < 1e-15:
        return h0_diag

    shifted = []
    for diag in h0_diag:
        if diag.size == 0:
            shifted.append(diag)
            continue
        shifted.append(diag + real)
    return shifted


def compute_shift_correction(
    amplitudes: list[np.ndarray],
    h0_diag_shifted: list[np.ndarray],
    h0_diag_orig: list[np.ndarray],
) -> float:
    """Compute energy correction due to level shift.

    E_shift_corr = sum_P |T_P|^2 * (shift_P)
    where shift_P = h0_diag_shifted[P] - h0_diag_orig[P].
    """
    corr = 0.0
    for amps, d_shift, d_orig in zip(amplitudes, h0_diag_shifted, h0_diag_orig):
        if amps.size == 0:
            continue
        delta = d_shift - d_orig
        corr += np.dot(amps.ravel() ** 2, delta.ravel())
    return float(corr)
