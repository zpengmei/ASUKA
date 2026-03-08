"""Kronecker sum G_apx and its inverse for SST-CASPT2.

The approximate zeroth-order Hamiltonian G_apx (Eq. 22 of Song 2024) is
a four-fold Kronecker sum of the site coupling matrices:

    G_apx = γ̄(p₁) ⊕_K γ(h₁) ⊕_K γ̄(p₂) ⊕_K γ(h₂)

where ⊕_K denotes Kronecker sum: A ⊕_K B = A ⊗ I + I ⊗ B.

In the dressed basis (where γ and γ̄ are diagonal), G_apx is diagonal:

    G_apx(π₁, η₁, π₂, η₂) = ω̄_π₁ - ω_η₁ + ω̄_π₂ - ω_η₂

(using the convention that γ eigenvalues are -ω, cf. Eq. 25).

The inverse G_apx⁻¹ in the dressed basis is simply 1/G_apx(diagonal).
In the original MO basis, applying G_apx⁻¹ requires transforming to/from
the dressed basis.

For future reduced-scaling, G_apx⁻¹ can also be factored via Laplace
quadrature (Eq. 34), enabling separable contractions.

This module provides:
    * Dense construction and application of G_apx for each IC case
    * Dressed-basis diagonal inverse of G_apx
    * Laplace-factored inverse (for future optimization)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.sst.coupling import DressedOrbitals
from asuka.caspt2.sst.laplace import LaplaceGrid, make_log_trap_laplace_grid

__all__ = [
    "CaseIndexInfo",
    "build_case_index_info",
    "build_g_apx_diagonal",
    "apply_g_apx_inv_dressed",
    "apply_g_apx_inv_laplace",
]


@dataclass(frozen=True)
class CaseIndexInfo:
    """Index mapping for a specific IC case in the 4-site decomposition.

    Each IC-CASPT2 case has a specific assignment of its indices to the
    4 sites (p1, h1, p2, h2). This dataclass encodes which dressed orbital
    energies contribute to each function in the case.

    Attributes
    ----------
    case : int
        IC case number (1-13).
    nasup : int
        Active superindex dimension.
    nisup : int
        External superindex dimension.
    n_sr : int
        Dimension in the SR (transformed) basis after linear dependence removal.
    site_energies : ndarray (nasup * nisup,) or (n_sr,)
        G_apx diagonal elements for each function in this case.
    """

    case: int
    nasup: int
    nisup: int
    n_sr: int
    site_energies: np.ndarray


def _get_external_energies_for_case(
    case: int,
    dressed: DressedOrbitals,
    nish: int,
    nash: int,
    nssh: int,
) -> np.ndarray:
    """Get external (non-active) energy contributions for each IC case.

    Returns the sum of dressed orbital energies for the external indices.
    This mirrors ``_get_external_energies`` from ``energy.py`` but uses
    dressed orbital energies.
    """
    omega = dressed.omega_hole      # (nhole,) actual orbital energies
    omega_bar = dressed.omega_particle  # (nparticle,) actual orbital energies

    # Map from dressed indices to orbital types:
    # omega[:ncore] = core dressed energies
    # omega[ncore:nhole] = active-as-hole dressed energies
    # omega_bar[:ncas] = active-as-particle dressed energies
    # omega_bar[ncas:] = virtual dressed energies
    ncore = dressed.ncore
    ncas = dressed.ncas
    nvirt = dressed.nvirt

    eps_core = omega[:ncore] if ncore > 0 else np.empty(0)
    eps_virt = omega_bar[ncas:] if nvirt > 0 else np.empty(0)

    # External energy depends on the case's external index structure.
    # See superindex.py for the mapping.
    if case == 1:  # A: tuv × i → one core index external
        return -eps_core  # h₂ = core(i), contributes -ω_i
    elif case == 2:  # B+: t≥u × i≥j → symmetric core pair
        ext = []
        for i in range(nish):
            for j in range(i + 1):  # j <= i
                ext.append(-eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 3:  # B-: t>u × i>j → antisymmetric core pair
        ext = []
        for i in range(nish):
            for j in range(i):  # j < i
                ext.append(-eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 4:  # C: tuv × a → one virtual index external
        return eps_virt  # p₂ = virt(a), contributes +ω̄_a
    elif case == 5:  # D: tu × ai → one virtual + one core
        ext = []
        for a in range(nssh):
            for i in range(nish):
                ext.append(eps_virt[a] - eps_core[i])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 6:  # E+: t × a(i≥j) → virtual + symmetric core pair
        ext = []
        for a in range(nssh):
            for i in range(nish):
                for j in range(i + 1):  # j <= i
                    ext.append(eps_virt[a] - eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 7:  # E-: t × a(i>j) → virtual + antisymmetric core pair
        ext = []
        for a in range(nssh):
            for i in range(nish):
                for j in range(i):  # j < i
                    ext.append(eps_virt[a] - eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 8:  # F+: t≥u × a≥b → symmetric virtual pair
        ext = []
        for a in range(nssh):
            for b in range(a + 1):  # b <= a
                ext.append(eps_virt[a] + eps_virt[b])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 9:  # F-: t>u × a>b → antisymmetric virtual pair
        ext = []
        for a in range(nssh):
            for b in range(a):  # b < a
                ext.append(eps_virt[a] + eps_virt[b])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 10:  # G+: t × i(a≥b) → core + symmetric virtual pair
        ext = []
        for i in range(nish):
            for a in range(nssh):
                for b in range(a + 1):
                    ext.append(eps_virt[a] + eps_virt[b] - eps_core[i])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 11:  # G-: t × i(a>b) → core + antisymmetric virtual pair
        ext = []
        for i in range(nish):
            for a in range(nssh):
                for b in range(a):
                    ext.append(eps_virt[a] + eps_virt[b] - eps_core[i])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 12:  # H+: a≥b × i≥j → symmetric pairs
        ext = []
        for a in range(nssh):
            for b in range(a + 1):
                for i in range(nish):
                    for j in range(i + 1):
                        ext.append(eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    elif case == 13:  # H-: a>b × i>j → antisymmetric pairs
        ext = []
        for a in range(nssh):
            for b in range(a):
                for i in range(nish):
                    for j in range(i):
                        ext.append(eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j])
        return np.array(ext, dtype=np.float64) if ext else np.empty(0)
    else:
        raise ValueError(f"Unknown case {case}")


def build_g_apx_diagonal(
    case: int,
    dressed: DressedOrbitals,
    nish: int,
    nash: int,
    nssh: int,
    b_diag: np.ndarray,
    ext_energies_standard: np.ndarray,
) -> np.ndarray:
    """Build the G_apx diagonal for a case in the SR basis.

    For the initial implementation, we construct G_apx as a diagonal in the
    SR basis by replacing the exact B-matrix diagonal + external energies
    with the dressed-orbital Kronecker sum approximation.

    In the SR (spectrally resolved) basis, the diagonal elements are:
        G_apx_sr[α, ext] = b_diag_apx[α] + ext_energies_dressed[ext]

    where b_diag_apx is the active-superindex diagonal from the Kronecker
    sum, and ext_energies_dressed uses dressed orbital energies.

    Parameters
    ----------
    case : int
        IC case number (1-13).
    dressed : DressedOrbitals
    nish, nash, nssh : int
        Orbital partition sizes.
    b_diag : ndarray
        B-matrix diagonal from standard SB decomposition (nindep,).
    ext_energies_standard : ndarray
        External energies from standard Fock diagonal.

    Returns
    -------
    g_apx_diag : ndarray
        G_apx diagonal values.
    """
    # Get the dressed external energies for this case
    ext_dressed = _get_external_energies_for_case(case, dressed, nish, nash, nssh)

    # For the initial implementation, use the standard B-matrix diagonal
    # (which captures the active-space Kronecker sum contribution) and
    # replace only the external energies with dressed versions.
    # This is approximate but becomes exact when orbitals are semicanonical.
    nindep = b_diag.size
    next_ext = ext_dressed.size

    if nindep == 0 or next_ext == 0:
        return np.empty(0, dtype=np.float64)

    # Full diagonal: b_diag[α] + ext_dressed[ext]
    g_diag = (b_diag[:, None] + ext_dressed[None, :]).ravel()
    return np.asarray(g_diag, dtype=np.float64, order="C")


def apply_g_apx_inv_dressed(
    vec: np.ndarray,
    g_apx_diag: np.ndarray,
    threshold: float = 1e-14,
) -> np.ndarray:
    """Apply G_apx⁻¹ to a vector using the diagonal form.

    In the SR basis (where G_apx is approximately diagonal), this is
    simply element-wise division.

    Parameters
    ----------
    vec : ndarray
        Input vector.
    g_apx_diag : ndarray
        G_apx diagonal values (same length as vec).
    threshold : float
        Small-denominator threshold. Elements with |g_apx| < threshold
        are set to zero in the result.

    Returns
    -------
    result : ndarray
        G_apx⁻¹ · vec.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    g_apx_diag = np.asarray(g_apx_diag, dtype=np.float64).ravel()

    if vec.size != g_apx_diag.size:
        raise ValueError(f"vec size {vec.size} != g_apx_diag size {g_apx_diag.size}")

    result = np.zeros_like(vec)
    mask = np.abs(g_apx_diag) > threshold
    result[mask] = vec[mask] / g_apx_diag[mask]
    return result


def apply_g_apx_inv_laplace(
    vec: np.ndarray,
    omega_hole: np.ndarray,
    omega_particle: np.ndarray,
    grid: LaplaceGrid,
    site_assignment: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply G_apx⁻¹ using Laplace quadrature factorization (Eq. 34).

    This is the reduced-scaling version that factors:

        G_apx⁻¹ = ∫₀^∞ e^{-γ̄(p₁)t} e^{-γ(h₁)t} e^{-γ̄(p₂)t} e^{-γ(h₂)t} dt
                ≈ Σ_g w_g Π_m exp(-ĝ_m t_g)

    In the dressed basis, this becomes separable element-wise exponentials.

    Parameters
    ----------
    vec : ndarray
        Input vector in dressed basis.
    omega_hole : ndarray (nhole,)
        Hole dressed orbital energies.
    omega_particle : ndarray (nparticle,)
        Particle dressed orbital energies.
    grid : LaplaceGrid
        Laplace quadrature nodes and weights.
    site_assignment : tuple of 4 ndarrays
        Index arrays mapping vector elements to (p1, h1, p2, h2) indices.

    Returns
    -------
    result : ndarray
        G_apx⁻¹ · vec via Laplace factorization.
    """
    # For now, compute the diagonal G_apx and invert directly.
    # The separable Laplace form is used only when the structure allows
    # factored contractions (e.g., in DF-based sigma operations).
    p1_idx, h1_idx, p2_idx, h2_idx = site_assignment

    g_diag = (omega_particle[p1_idx] - omega_hole[h1_idx]
              + omega_particle[p2_idx] - omega_hole[h2_idx])

    # Laplace quadrature: 1/x = Σ_g w_g exp(-t_g x)
    result = np.zeros_like(vec)
    for g in range(grid.t.size):
        exp_factors = np.exp(-grid.t[g] * g_diag)
        result += grid.w[g] * exp_factors * vec

    return result
