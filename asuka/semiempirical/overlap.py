"""Analytic STO overlap matrix for NDDO semiempirical methods.

Computes exact Slater-type orbital overlaps using Mulliken's A/B auxiliary
integrals in confocal ellipsoidal coordinates. Supports 1s, 2s, and 2p STOs
(sufficient for the H/C/N/O sp basis).

The overlap is computed in the LOCAL diatomic frame (bond along z) and then
rotated to the GLOBAL Cartesian frame using direction cosines.

Reference: Mulliken, Rieke, Orloff & Orloff, JCP 17, 1248 (1949)
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .basis import build_ao_offsets, nao_for_Z
from .params import ElementParams

# STO normalization: R_nl(r) = N_nl * r^(n-1) * exp(-zeta*r)
# Normalized so that integral |R_nl|^2 r^2 dr = 1
# N_1s = 2*zeta^(3/2)
# N_2s = N_2p = 2*zeta^(5/2) / sqrt(3)


def _A(k: int, x: float) -> float:
    """Mulliken A integral: A_k(x) = integral_1^inf t^k * exp(-x*t) dt.

    Computed by upward recursion: A_k = (k * A_{k-1} + exp(-x)) / x
    """
    if x < 1e-15:
        # A_k(0) = 1/(k+1)  (diverges, but shouldn't be called with x=0)
        return 1.0 / (k + 1)
    ex = math.exp(-x)
    a = ex / x  # A_0
    for j in range(1, k + 1):
        a = (j * a + ex) / x
    return a


def _B(k: int, x: float) -> float:
    """Mulliken B integral: B_k(x) = integral_{-1}^{1} t^k * exp(-x*t) dt.

    Computed by upward recursion:
      B_0 = 2*sinh(x)/x
      B_k = [(-1)^k * exp(x) - exp(-x)] / x + k * B_{k-1} / x
    """
    if abs(x) < 1e-10:
        # Taylor expansion: B_k(0) = 2/(k+1) for k even, 0 for k odd
        if k % 2 == 0:
            return 2.0 / (k + 1)
        return 0.0

    ep = math.exp(x)
    em = math.exp(-x)
    b = (ep - em) / x  # B_0 = 2*sinh(x)/x
    for j in range(1, k + 1):
        b = ((-1) ** j * ep - em + j * b) / x
    return b


def _overlap_1s_1s(za: float, zb: float, R: float) -> float:
    """Analytic overlap S(1s_A, 1s_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 1.5
    Nb = 2.0 * zb ** 1.5
    prefactor = Na * Nb / 2.0 * (R / 2.0) ** 3
    return prefactor * (_A(2, p) * _B(0, q) - _A(0, p) * _B(2, q))


def _overlap_1s_2s(za: float, zb: float, R: float) -> float:
    """Analytic overlap S(1s_A, 2s_B). A has 1s (n=1), B has 2s (n=2)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 1.5
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    prefactor = Na * Nb / 2.0 * (R / 2.0) ** 4
    return prefactor * (
        _A(3, p) * _B(0, q) - _A(2, p) * _B(1, q)
        - _A(1, p) * _B(2, q) + _A(0, p) * _B(3, q)
    )


def _overlap_2s_1s(za: float, zb: float, R: float) -> float:
    """Analytic overlap S(2s_A, 1s_B). A has 2s (n=2), B has 1s (n=1)."""
    # Same formula as 1s-2s but with swapped roles
    # (n_A=2, n_B=1): (xi+eta)^2*(xi-eta)^1
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 1.5
    prefactor = Na * Nb / 2.0 * (R / 2.0) ** 4
    # (xi+eta)^2*(xi-eta)^1 = xi^3 + xi^2*eta - xi*eta^2 - eta^3
    return prefactor * (
        _A(3, p) * _B(0, q) + _A(2, p) * _B(1, q)
        - _A(1, p) * _B(2, q) - _A(0, p) * _B(3, q)
    )


def _overlap_2s_2s(za: float, zb: float, R: float) -> float:
    """Analytic overlap S(2s_A, 2s_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    prefactor = Na * Nb / 2.0 * (R / 2.0) ** 5
    # (xi+eta)^2*(xi-eta)^2 = xi^4 - 2*xi^2*eta^2 + eta^4
    return prefactor * (
        _A(4, p) * _B(0, q) - 2.0 * _A(2, p) * _B(2, q) + _A(0, p) * _B(4, q)
    )


def _overlap_1s_2ps(za: float, zb: float, R: float) -> float:
    """Analytic sigma overlap S(1s_A, 2p_sigma_B). B has pz along bond."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 1.5
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    # Angular factor sqrt(3)/2 for s-psigma
    prefactor = Na * Nb * math.sqrt(3.0) / 2.0 * (R / 2.0) ** 4
    # (xi+eta)^1*(xi-eta)^1*(xi*eta-1) = xi^3*eta - xi^2 - xi*eta^3 + eta^2
    return prefactor * (
        _A(3, p) * _B(1, q) - _A(2, p) * _B(0, q)
        - _A(1, p) * _B(3, q) + _A(0, p) * _B(2, q)
    )


def _overlap_2ps_1s(za: float, zb: float, R: float) -> float:
    """Analytic sigma overlap S(2p_sigma_A, 1s_B). A has pz along bond."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 1.5
    prefactor = Na * Nb * math.sqrt(3.0) / 2.0 * (R / 2.0) ** 4
    # (xi+eta)^1*(xi-eta)^1*(1+xi*eta) = xi^2 + xi^3*eta - eta^2 - xi*eta^3
    return prefactor * (
        _A(2, p) * _B(0, q) + _A(3, p) * _B(1, q)
        - _A(0, p) * _B(2, q) - _A(1, p) * _B(3, q)
    )


def _overlap_2s_2ps(za: float, zb: float, R: float) -> float:
    """Analytic sigma overlap S(2s_A, 2p_sigma_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    prefactor = Na * Nb * math.sqrt(3.0) / 2.0 * (R / 2.0) ** 5
    # (xi+eta)^2*(xi-eta)*(xi*eta-1)
    # = xi^4*eta + xi^3*eta^2 - xi^2*eta^3 - xi*eta^4 - xi^3 - xi^2*eta + xi*eta^2 + eta^3
    return prefactor * (
        _A(4, p) * _B(1, q) + _A(3, p) * _B(2, q)
        - _A(2, p) * _B(3, q) - _A(1, p) * _B(4, q)
        - _A(3, p) * _B(0, q) - _A(2, p) * _B(1, q)
        + _A(1, p) * _B(2, q) + _A(0, p) * _B(3, q)
    )


def _overlap_2ps_2s(za: float, zb: float, R: float) -> float:
    """Analytic sigma overlap S(2p_sigma_A, 2s_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    prefactor = Na * Nb * math.sqrt(3.0) / 2.0 * (R / 2.0) ** 5
    # (xi+eta)*(xi-eta)^2*(1+xi*eta)
    # = xi^3 + xi^4*eta - xi^2*eta - xi^3*eta^2 - xi*eta^2 - xi^2*eta^3 + eta^3 + xi*eta^4
    return prefactor * (
        _A(3, p) * _B(0, q) + _A(4, p) * _B(1, q)
        - _A(2, p) * _B(1, q) - _A(3, p) * _B(2, q)
        - _A(1, p) * _B(2, q) - _A(2, p) * _B(3, q)
        + _A(0, p) * _B(3, q) + _A(1, p) * _B(4, q)
    )


def _overlap_2ps_2ps(za: float, zb: float, R: float) -> float:
    """Analytic sigma overlap S(2p_sigma_A, 2p_sigma_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    # Angular factor 3/2 for psigma-psigma
    prefactor = Na * Nb * 3.0 / 2.0 * (R / 2.0) ** 5
    # (xi^2 - eta^2)*(xi^2*eta^2 - 1) = xi^4*eta^2 - xi^2 - xi^2*eta^4 + eta^2
    return prefactor * (
        _A(4, p) * _B(2, q) - _A(2, p) * _B(0, q)
        - _A(2, p) * _B(4, q) + _A(0, p) * _B(2, q)
    )


def _overlap_2pp_2pp(za: float, zb: float, R: float) -> float:
    """Analytic pi overlap S(2p_pi_A, 2p_pi_B)."""
    p = (za + zb) * R / 2.0
    q = (za - zb) * R / 2.0
    Na = 2.0 * za ** 2.5 / math.sqrt(3.0)
    Nb = 2.0 * zb ** 2.5 / math.sqrt(3.0)
    # Angular factor 3/4 for ppi-ppi
    prefactor = Na * Nb * 3.0 / 4.0 * (R / 2.0) ** 5
    # (xi^2-eta^2)*(xi^2-1)*(1-eta^2)
    # = xi^4 - xi^4*eta^2 - xi^2 + xi^2*eta^4 + eta^2 - eta^4
    return prefactor * (
        _A(4, p) * _B(0, q) - _A(4, p) * _B(2, q)
        - _A(2, p) * _B(0, q) + _A(2, p) * _B(4, q)
        + _A(0, p) * _B(2, q) - _A(0, p) * _B(4, q)
    )


def _compute_diatomic_overlaps(
    nao_A: int, nao_B: int,
    zs_A: float, zp_A: float,
    zs_B: float, zp_B: float,
    R: float,
) -> tuple:
    """Compute overlap components in the local diatomic frame.

    Returns (S_ss, S_spσ, S_pσs, S_pσpσ, S_pπpπ) where applicable.
    Components involving p orbitals are zero if the atom has nao=1.
    """
    if nao_A == 1 and nao_B == 1:
        # H-H: 1s-1s only
        return (_overlap_1s_1s(zs_A, zs_B, R), 0, 0, 0, 0)

    if nao_A == 1 and nao_B == 4:
        # H-Heavy: 1s-2s and 1s-2pσ
        Sss = _overlap_1s_2s(zs_A, zs_B, R)
        Ssp = _overlap_1s_2ps(zs_A, zp_B, R)
        return (Sss, Ssp, 0, 0, 0)

    if nao_A == 4 and nao_B == 1:
        # Heavy-H: 2s-1s and 2pσ-1s
        Sss = _overlap_2s_1s(zs_A, zs_B, R)
        Sps = _overlap_2ps_1s(zp_A, zs_B, R)
        return (Sss, 0, Sps, 0, 0)

    # Heavy-Heavy: all 5 components
    Sss = _overlap_2s_2s(zs_A, zs_B, R)
    Ssp = _overlap_2s_2ps(zs_A, zp_B, R)
    Sps = _overlap_2ps_2s(zp_A, zs_B, R)
    Spp_sig = _overlap_2ps_2ps(zp_A, zp_B, R)
    Spp_pi = _overlap_2pp_2pp(zp_A, zp_B, R)
    return (Sss, Ssp, Sps, Spp_sig, Spp_pi)


def _rotate_overlaps_to_global(
    nao_A: int, nao_B: int,
    S_ss: float, S_sp: float, S_ps: float,
    S_pp_sig: float, S_pp_pi: float,
    bond_dir: np.ndarray,
) -> np.ndarray:
    """Rotate local-frame overlaps to global Cartesian frame.

    Parameters
    ----------
    nao_A, nao_B : int
        Number of AOs on each atom (1 or 4).
    S_ss, S_sp, S_ps, S_pp_sig, S_pp_pi : float
        Local-frame overlap components.
    bond_dir : (3,) array
        Unit vector from A to B.

    Returns
    -------
    S_block : (nao_A, nao_B) array
    """
    S_block = np.zeros((nao_A, nao_B), dtype=float)
    c = bond_dir  # direction cosines

    # s_A - s_B
    S_block[0, 0] = S_ss

    if nao_B == 4:
        # s_A - p_i^B: S = S_spσ * c_i
        for i in range(3):
            S_block[0, 1 + i] = S_sp * c[i]

    if nao_A == 4:
        # p_i^A - s_B: S = S_pσs * c_i
        for i in range(3):
            S_block[1 + i, 0] = S_ps * c[i]

    if nao_A == 4 and nao_B == 4:
        # p_i^A - p_j^B:
        # S = S_ppσ * c_i * c_j + S_ppπ * (δ_ij - c_i * c_j)
        for i in range(3):
            for j in range(3):
                S_block[1 + i, 1 + j] = (
                    S_pp_sig * c[i] * c[j]
                    + S_pp_pi * (float(i == j) - c[i] * c[j])
                )

    return S_block


def build_pair_overlap_block(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    iA: int,
    iB: int,
    elem_params: Dict[int, ElementParams],
) -> np.ndarray:
    """Build one inter-atomic overlap block ``S_AB`` in molecular frame.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers for all atoms.
    coords_bohr : (N, 3) array
        Cartesian coordinates in Bohr.
    iA, iB : int
        Pair atom indices with ``iA != iB``.
    elem_params : dict
        Element parameters keyed by atomic number.

    Returns
    -------
    S_block : (nao_A, nao_B) array
        Overlap block between atoms A and B.
    """
    if int(iA) == int(iB):
        raise ValueError("build_pair_overlap_block requires iA != iB")

    iA = int(iA)
    iB = int(iB)
    Z_A = int(atomic_numbers[iA])
    Z_B = int(atomic_numbers[iB])
    ep_A = elem_params[Z_A]
    ep_B = elem_params[Z_B]
    nao_A = nao_for_Z(Z_A)
    nao_B = nao_for_Z(Z_B)

    dR = np.asarray(coords_bohr, dtype=float)[iB] - np.asarray(coords_bohr, dtype=float)[iA]
    R = float(np.linalg.norm(dR))
    if R < 1e-14:
        raise ValueError("build_pair_overlap_block requires non-coincident nuclei")
    bond_dir = dR / R

    Sss, Ssp, Sps, Spp_sig, Spp_pi = _compute_diatomic_overlaps(
        nao_A,
        nao_B,
        ep_A.zeta_s,
        ep_A.zeta_p,
        ep_B.zeta_s,
        ep_B.zeta_p,
        R,
    )
    return _rotate_overlaps_to_global(
        nao_A,
        nao_B,
        Sss,
        Ssp,
        Sps,
        Spp_sig,
        Spp_pi,
        bond_dir,
    )


def build_overlap_matrix(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    elem_params: Dict[int, ElementParams],
) -> np.ndarray:
    """Build the full STO overlap matrix using analytic STO formulas.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers for each atom.
    coords_bohr : (N, 3) array
        Atomic coordinates in Bohr.
    elem_params : dict
        ElementParams keyed by atomic number.

    Returns
    -------
    S : (nao, nao) array
        Overlap matrix.
    """
    N = len(atomic_numbers)
    offsets = build_ao_offsets(atomic_numbers)
    nao_total = offsets[-1]
    S = np.eye(nao_total, dtype=float)

    for iA in range(N):
        Z_A = atomic_numbers[iA]
        nao_A = nao_for_Z(Z_A)
        i0A = offsets[iA]

        for iB in range(iA + 1, N):
            Z_B = atomic_numbers[iB]
            nao_B = nao_for_Z(Z_B)
            i0B = offsets[iB]

            # Bond vector and distance
            S_block = build_pair_overlap_block(
                atomic_numbers=atomic_numbers,
                coords_bohr=coords_bohr,
                iA=iA,
                iB=iB,
                elem_params=elem_params,
            )

            # Place into full matrix
            idxA = slice(i0A, i0A + nao_A)
            idxB = slice(i0B, i0B + nao_B)
            S[idxA, idxB] = S_block
            S[idxB, idxA] = S_block.T

    return S
