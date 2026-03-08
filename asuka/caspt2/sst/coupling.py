"""Coupling matrices and dressed orbitals for SST-CASPT2.

Builds the hole coupling matrix γ (Eq. 17) and particle coupling matrix γ̄
(Eq. 18) from Song, JCP 160, 2024.  These encode the spin-averaged
one-particle matrix elements of Dyall's Hamiltonian and are the key
objects for the SST factorization.

Diagonalizing γ → ω_η (hole dressed energies) and γ̄ → ω̄_π (particle
dressed energies) yields the dressed orbital basis used throughout the SST
algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.mrpt2.semicanonical import molcas_diafck_eigh

__all__ = [
    "DressedOrbitals",
    "build_coupling_matrices",
    "diagonalize_coupling",
]


@dataclass(frozen=True)
class DressedOrbitals:
    """Dressed orbital basis from coupling matrix diagonalization.

    Attributes
    ----------
    omega_hole : (nhole,) where nhole = ncore + ncas
        Hole dressed orbital energies (eigenvalues of γ).
    omega_particle : (nparticle,) where nparticle = ncas + nvirt
        Particle dressed orbital energies (eigenvalues of γ̄).
    u_hole : (nhole, nhole)
        Transformation from MO hole orbitals to dressed hole orbitals.
        C_dressed_hole = C_MO[:, :nhole] @ u_hole
    u_particle : (nparticle, nparticle)
        Transformation from MO particle orbitals to dressed particle orbitals.
        C_dressed_particle = C_MO[:, ncore:] @ u_particle
    gamma : (nhole, nhole)
        Original hole coupling matrix (before diagonalization).
    gamma_bar : (nparticle, nparticle)
        Original particle coupling matrix (before diagonalization).
    ncore : int
    ncas : int
    nvirt : int
    """

    omega_hole: np.ndarray
    omega_particle: np.ndarray
    u_hole: np.ndarray
    u_particle: np.ndarray
    gamma: np.ndarray
    gamma_bar: np.ndarray
    ncore: int
    ncas: int
    nvirt: int


def build_coupling_matrices(
    fock: CASPT2Fock,
    dm1: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nvirt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build hole coupling γ and particle coupling γ̄ matrices.

    The coupling matrices are spin-averaged one-particle matrix elements
    of the Dyall Hamiltonian (Eq. 17-18 of Song 2024).

    Parameters
    ----------
    fock
        CASPT2 Fock object (provides ``fifa`` = full MO Fock).
    dm1
        Active-space 1-RDM, shape (ncas, ncas).
    ncore, ncas, nvirt
        Orbital partition sizes.

    Returns
    -------
    gamma : (nhole, nhole) where nhole = ncore + ncas
        Hole coupling matrix.
    gamma_bar : (nparticle, nparticle) where nparticle = ncas + nvirt
        Particle coupling matrix.
    """
    fifa = np.asarray(fock.fifa, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    ncore = int(ncore)
    ncas = int(ncas)
    nvirt = int(nvirt)
    nocc = ncore + ncas
    nmo = ncore + ncas + nvirt

    if fifa.shape != (nmo, nmo):
        raise ValueError(f"fifa shape {fifa.shape} != ({nmo}, {nmo})")
    if dm1.shape != (ncas, ncas):
        raise ValueError(f"dm1 shape {dm1.shape} != ({ncas}, {ncas})")

    nhole = ncore + ncas
    nparticle = ncas + nvirt

    # ── Hole coupling γ ──
    # γ encodes the hole (occupied) part of Dyall's H0.
    # Core-core: γ[i,j] = -fifa[i,j]
    # Core-active: γ[i,t] = -fifa[i, ncore+t]
    # Active-active: γ[t,u] = -Σ_v fifa[ncore+t, ncore+v] * dm1[v,u]
    gamma = np.zeros((nhole, nhole), dtype=np.float64)

    # Core-core block
    if ncore > 0:
        gamma[:ncore, :ncore] = -fifa[:ncore, :ncore]

    # Core-active block (and transpose)
    if ncore > 0 and ncas > 0:
        gamma[:ncore, ncore:nhole] = -fifa[:ncore, ncore:nocc]
        gamma[ncore:nhole, :ncore] = -fifa[ncore:nocc, :ncore]

    # Active-active block: γ[t,u] = -Σ_v f[t,v] * dm1[v,u]
    if ncas > 0:
        f_act = fifa[ncore:nocc, ncore:nocc]  # (ncas, ncas)
        gamma[ncore:nhole, ncore:nhole] = -(f_act @ dm1)

    # Symmetrize (should already be symmetric for exact dm1/Fock, but enforce)
    gamma = 0.5 * (gamma + gamma.T)

    # ── Particle coupling γ̄ ──
    # γ̄ encodes the particle (unoccupied) part of Dyall's H0.
    # Virtual-virtual: γ̄[a,b] = +fifa[nocc+a, nocc+b]
    # Virtual-active: γ̄[a,t] = +fifa[nocc+a, ncore+t]
    # Active-active: γ̄[t,u] = +Σ_v fifa[ncore+t, ncore+v] * (δ[v,u] - dm1[v,u])
    gamma_bar = np.zeros((nparticle, nparticle), dtype=np.float64)

    # Active-active block: γ̄[t,u] = +Σ_v f[t,v] * (δ[v,u] - dm1[v,u])
    if ncas > 0:
        f_act = fifa[ncore:nocc, ncore:nocc]
        hole_complement = np.eye(ncas, dtype=np.float64) - dm1
        gamma_bar[:ncas, :ncas] = f_act @ hole_complement

    # Active-virtual block (and transpose)
    if ncas > 0 and nvirt > 0:
        gamma_bar[:ncas, ncas:nparticle] = fifa[ncore:nocc, nocc:nmo]
        gamma_bar[ncas:nparticle, :ncas] = fifa[nocc:nmo, ncore:nocc]

    # Virtual-virtual block
    if nvirt > 0:
        gamma_bar[ncas:nparticle, ncas:nparticle] = fifa[nocc:nmo, nocc:nmo]

    # Symmetrize
    gamma_bar = 0.5 * (gamma_bar + gamma_bar.T)

    return (
        np.asarray(gamma, dtype=np.float64, order="C"),
        np.asarray(gamma_bar, dtype=np.float64, order="C"),
    )


def diagonalize_coupling(
    gamma: np.ndarray,
    gamma_bar: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nvirt: int,
    real_shift: float = 0.0,
    deg_tol: float = 1e-10,
) -> DressedOrbitals:
    """Diagonalize coupling matrices to obtain dressed orbitals.

    Parameters
    ----------
    gamma
        Hole coupling matrix (nhole, nhole).
    gamma_bar
        Particle coupling matrix (nparticle, nparticle).
    ncore, ncas, nvirt
        Orbital partition sizes.
    real_shift
        Real level shift ε_shift. Distributed as +ε_shift/4 to each
        dressed orbital energy (4 sites in the Kronecker sum).
    deg_tol
        Degeneracy tolerance for eigendecomposition gauge fixing.

    Returns
    -------
    DressedOrbitals
        Dressed orbital energies and transformation matrices.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    gamma_bar = np.asarray(gamma_bar, dtype=np.float64)
    nhole = ncore + ncas
    nparticle = ncas + nvirt

    if gamma.shape != (nhole, nhole):
        raise ValueError(f"gamma shape {gamma.shape} != ({nhole}, {nhole})")
    if gamma_bar.shape != (nparticle, nparticle):
        raise ValueError(f"gamma_bar shape {gamma_bar.shape} != ({nparticle}, {nparticle})")

    # Diagonalize hole coupling.
    # Per Eq. 25: γ = U diag(-ω) U^T, so eigenvalues of γ are -ω.
    # We negate to get the actual dressed orbital energies ω.
    if nhole > 0:
        neg_omega_hole, u_hole = molcas_diafck_eigh(gamma, deg_tol=deg_tol)
        omega_hole = -neg_omega_hole  # ω = -(eigenvalue of γ)
    else:
        omega_hole = np.zeros(0, dtype=np.float64)
        u_hole = np.zeros((0, 0), dtype=np.float64)

    # Diagonalize particle coupling.
    # Per Eq. 26: γ̄ = U diag(ω̄) U^T, so eigenvalues of γ̄ are ω̄ directly.
    if nparticle > 0:
        omega_particle, u_particle = molcas_diafck_eigh(gamma_bar, deg_tol=deg_tol)
    else:
        omega_particle = np.zeros(0, dtype=np.float64)
        u_particle = np.zeros((0, 0), dtype=np.float64)

    # Apply real level shift: ε_shift distributed across 4 Kronecker sum sites.
    # Shift lowers hole energies and raises particle energies, so the
    # denominator (ω̄ - ω) increases by ε_shift total.
    shift_per_site = real_shift / 4.0
    if abs(shift_per_site) > 1e-15:
        omega_hole = omega_hole - shift_per_site
        omega_particle = omega_particle + shift_per_site

    return DressedOrbitals(
        omega_hole=np.asarray(omega_hole, dtype=np.float64, order="C"),
        omega_particle=np.asarray(omega_particle, dtype=np.float64, order="C"),
        u_hole=np.asarray(u_hole, dtype=np.float64, order="C"),
        u_particle=np.asarray(u_particle, dtype=np.float64, order="C"),
        gamma=np.asarray(gamma, dtype=np.float64, order="C"),
        gamma_bar=np.asarray(gamma_bar, dtype=np.float64, order="C"),
        ncore=ncore,
        ncas=ncas,
        nvirt=nvirt,
    )
