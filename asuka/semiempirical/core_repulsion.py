"""Core-core repulsion for AM1 semiempirical methods.

MNDO baseline plus AM1 Gaussian corrections.

Reference: MOPAC ccrep.F90
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .basis import valence_electrons
from .multipole import MultipoleParams
from .params import ElementParams, ANGSTROM_TO_BOHR, EV_TO_HARTREE


def core_core_repulsion(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_r: np.ndarray,
    W_list,
    elem_params: Dict[int, ElementParams],
) -> float:
    """Compute total core-core repulsion energy.

    E_core = sum_{A<B} E_N(A,B)

    Parameters
    ----------
    atomic_numbers : sequence of int
    coords_bohr : (N, 3) array
    pair_i, pair_j : (npairs,) int arrays
    pair_r : (npairs,) distances in Bohr
    W_list : list of two-center integral arrays (from build_two_center_integrals)
    elem_params : dict of ElementParams keyed by Z

    Returns
    -------
    E_core : float
        Total core-core repulsion in Hartree.
    """
    E_core = 0.0
    npairs = len(pair_i)

    for k in range(npairs):
        iA = pair_i[k]
        iB = pair_j[k]
        Z_A = atomic_numbers[iA]
        Z_B = atomic_numbers[iB]
        R = pair_r[k]
        ep_A = elem_params[Z_A]
        ep_B = elem_params[Z_B]
        zval_A = valence_electrons(Z_A)
        zval_B = valence_electrons(Z_B)

        # The ss|ss two-center integral at this distance
        gamma_ss = W_list[k][0, 0, 0, 0]

        E_core += _pair_core_repulsion(
            Z_A, Z_B, R, zval_A, zval_B, gamma_ss, ep_A, ep_B
        )

    return E_core


def core_core_repulsion_from_gamma_ss(
    atomic_numbers: Sequence[int],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_r: np.ndarray,
    gamma_ss: np.ndarray,
    elem_params: Dict[int, ElementParams],
) -> float:
    """Compute total core-core repulsion from packed ``(ss|ss)`` pair terms."""
    E_core = 0.0
    npairs = len(pair_i)

    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        Z_A = atomic_numbers[iA]
        Z_B = atomic_numbers[iB]
        R = float(pair_r[k])
        ep_A = elem_params[Z_A]
        ep_B = elem_params[Z_B]
        zval_A = valence_electrons(Z_A)
        zval_B = valence_electrons(Z_B)
        E_core += _pair_core_repulsion(
            Z_A,
            Z_B,
            R,
            zval_A,
            zval_B,
            float(gamma_ss[k]),
            ep_A,
            ep_B,
        )
    return E_core


def _pair_core_repulsion(
    Z_A: int,
    Z_B: int,
    R: float,
    zval_A: int,
    zval_B: int,
    gamma_ss: float,
    ep_A: ElementParams,
    ep_B: ElementParams,
) -> float:
    """Core-core repulsion for one atom pair.

    MNDO baseline:
      E_N(A,B) = Z_A * Z_B * gamma_ss * (1 + f_A(R) + f_B(R))

    where f_X(R) = exp(-alpha_X * R) for general pairs,
    and   f_X(R) = R * exp(-alpha_X * R) for X-H pairs (X = N or O).

    AM1 Gaussian correction:
      E_corr = (Z_A*Z_B/R_ang) * [sum_n K_nA * exp(-L_nA*(R_ang-M_nA)^2)
                                  + sum_n K_nB * exp(-L_nB*(R_ang-M_nB)^2)]
    with R_ang in Angstrom, K in eV, L in Ang^-2, M in Ang.
    """
    R_ang = R * 0.529177210903  # Bohr -> Angstrom

    # MNDO baseline
    alpha_A = ep_A.alpha
    alpha_B = ep_B.alpha

    # Special cases for N-H and O-H pairs (MOPAC convention)
    # Note: MOPAC uses R in Angstrom for the R*exp(-alpha*R) prefactor.
    # The exponential argument alpha*R is dimensionless (same in Bohr or Ang).
    if _is_NH_or_OH(Z_A, Z_B):
        # For X-H pairs where X is N(7) or O(8):
        # The heavy atom uses R_ang*exp(-alpha*R), H uses exp(-alpha*R)
        if Z_A == 1:
            # A is H, B is heavy
            f_A = math.exp(-alpha_A * R)
            f_B = R_ang * math.exp(-alpha_B * R)
        else:
            # A is heavy, B is H
            f_A = R_ang * math.exp(-alpha_A * R)
            f_B = math.exp(-alpha_B * R)
    else:
        f_A = math.exp(-alpha_A * R)
        f_B = math.exp(-alpha_B * R)

    E_mndo = zval_A * zval_B * gamma_ss * (1.0 + f_A + f_B)

    # AM1 Gaussian corrections
    E_gauss = 0.0
    if ep_A.gaussians or ep_B.gaussians:
        gauss_sum = 0.0
        for g in ep_A.gaussians:
            gauss_sum += g.k * math.exp(-g.l * (R_ang - g.m) ** 2)
        for g in ep_B.gaussians:
            gauss_sum += g.k * math.exp(-g.l * (R_ang - g.m) ** 2)

        # Convert: K is in eV, multiply by Z_A*Z_B/R_ang
        # Result needs to be in Hartree
        if R_ang > 1e-10:
            E_gauss = zval_A * zval_B * gauss_sum * EV_TO_HARTREE / R_ang
        # Note: gauss_sum is already in eV (from K_n), so we need EV_TO_HARTREE
        # The formula is E_corr = Z_A*Z_B * sum(K_n*exp(...)) / R_ang [in eV]

    return E_mndo + E_gauss


def _is_NH_or_OH(Z_A: int, Z_B: int) -> bool:
    """Check if pair is N-H or O-H (special core-core treatment)."""
    pair = frozenset([Z_A, Z_B])
    return pair == frozenset([1, 7]) or pair == frozenset([1, 8])
