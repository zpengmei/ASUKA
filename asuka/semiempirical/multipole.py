"""MOPAC multipole charge-distribution model for NDDO two-center integrals.

Derives per-element multipole parameters (AM, AD, AQ, D1, D2) from
the one-center empirical integrals, and provides the screened Coulomb
(gamma) function used in the two-center integral assembly.

Reference: MOPAC calpar.F90, mndod.F90
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .params import ElementParams, EV_TO_HARTREE

# Hartree to eV for internal use
_EV = 1.0 / EV_TO_HARTREE


@dataclass
class MultipoleParams:
    """Derived multipole parameters for one element (all in Bohr)."""
    Z: int
    dd: float   # Dipole charge separation D1
    qq: float   # Quadrupole charge separation D2
    am: float   # Monopole additive term (rho_0)
    ad: float   # Dipole additive term (rho_1)
    aq: float   # Quadrupole additive term (rho_2)


def derive_multipole_params(ep: ElementParams) -> MultipoleParams:
    """Derive multipole parameters from one-center ERIs.

    Follows MOPAC calpar.F90 exactly.
    """
    Z = ep.Z

    if Z == 1:
        # Hydrogen: only monopole
        am = 0.5 / _f0(ep.gss)
        return MultipoleParams(Z=Z, dd=0.0, qq=0.0, am=am, ad=0.0, aq=0.0)

    # Principal quantum number for second-row atoms
    qn = 2.0

    zs = ep.zeta_s
    zp = ep.zeta_p

    # DD: dipole charge separation
    dd = ((2 * qn + 1) * (4 * zs * zp) ** (qn + 0.5)
          / (zs + zp) ** (2 * qn + 2) / math.sqrt(3.0))

    # QQ: quadrupole charge separation
    qq = math.sqrt((4 * qn**2 + 6 * qn + 2) / 20.0) / zp

    # AM: monopole additive term from gss
    am = 0.5 / _f0(ep.gss)

    # AD: dipole additive term, Newton-Raphson iteration
    # Solve: hsp_au = 0.5*ad - 0.5/sqrt(4*dd^2 + 1/ad^2)
    hsp_au = ep.hsp
    gdd1 = (hsp_au / dd**2) ** (1.0 / 3.0)
    d1, d2 = gdd1, gdd1 + 0.04
    for _ in range(5):
        hsp1 = 0.5 * d1 - 0.5 / math.sqrt(4 * dd**2 + 1.0 / d1**2)
        hsp2 = 0.5 * d2 - 0.5 / math.sqrt(4 * dd**2 + 1.0 / d2**2)
        if abs(hsp2 - hsp1) < 1e-30:
            break
        d3 = d1 + (d2 - d1) * (hsp_au - hsp1) / (hsp2 - hsp1)
        d1, d2 = d2, d3
    # Convert from MOPAC additive term to screening radius: rho = 0.5/ad
    ad = 0.5 / d2

    # AQ: quadrupole additive term, Newton-Raphson
    # hpp = 0.5*(gpp - gp2)
    # Solve: hpp_au = 0.25*aq - 0.5/sqrt(4*qq^2+1/aq^2) + 0.25/sqrt(8*qq^2+1/aq^2)
    hpp_au = 0.5 * (ep.gpp - ep.gp2)
    gqq = (16.0 * hpp_au / (48.0 * qq**4)) ** 0.2
    q1, q2 = gqq, gqq + 0.04
    for _ in range(5):
        hpp1 = (0.25 * q1
                - 0.5 / math.sqrt(4 * qq**2 + 1.0 / q1**2)
                + 0.25 / math.sqrt(8 * qq**2 + 1.0 / q1**2))
        hpp2 = (0.25 * q2
                - 0.5 / math.sqrt(4 * qq**2 + 1.0 / q2**2)
                + 0.25 / math.sqrt(8 * qq**2 + 1.0 / q2**2))
        if abs(hpp2 - hpp1) < 1e-30:
            break
        q3 = q1 + (q2 - q1) * (hpp_au - hpp1) / (hpp2 - hpp1)
        q1, q2 = q2, q3
    # Convert from MOPAC additive term to screening radius: rho = 0.5/aq
    aq = 0.5 / q2

    return MultipoleParams(Z=Z, dd=dd, qq=qq, am=am, ad=ad, aq=aq)


def _f0(gss: float) -> float:
    """Compute 1/(2*am) from gss: gss = 1/(2*am), so am = 1/(2*f0(gss)).

    Actually f0 = gss, so am = 0.5/gss. This helper just returns gss.
    """
    return gss


def gamma_screened(R: float, rho_A: float, rho_B: float) -> float:
    """Screened Coulomb interaction between two charge distributions.

    gamma(R, rho_A, rho_B) = 1/sqrt(R^2 + (rho_A + rho_B)^2)

    All quantities in Bohr/Hartree.
    """
    return 1.0 / math.sqrt(R * R + (rho_A + rho_B) ** 2)


def compute_all_multipole_params(
    elem_params: Dict[int, ElementParams],
) -> Dict[int, MultipoleParams]:
    """Derive multipole params for all elements in the parameter set."""
    return {Z: derive_multipole_params(ep) for Z, ep in elem_params.items()}
