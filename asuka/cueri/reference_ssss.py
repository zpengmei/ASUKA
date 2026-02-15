from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from .basis import BasisSoA
from .boys import boys_f0


@dataclass(frozen=True)
class PairTable:
    eta: np.ndarray  # float64, shape (nPair,)
    P: np.ndarray  # float64, shape (nPair, 3)
    cK: np.ndarray  # float64, shape (nPair,)


def _build_pair_table_ss(basis: BasisSoA, A: int, B: int) -> PairTable:
    cA = basis.shell_cxyz[A]
    cB = basis.shell_cxyz[B]
    AB2 = float(np.dot(cA - cB, cA - cB))

    sA = int(basis.shell_prim_start[A])
    sB = int(basis.shell_prim_start[B])
    nA = int(basis.shell_nprim[A])
    nB = int(basis.shell_nprim[B])
    expA = basis.prim_exp[sA : sA + nA]
    expB = basis.prim_exp[sB : sB + nB]
    coefA = basis.prim_coef[sA : sA + nA]
    coefB = basis.prim_coef[sB : sB + nB]

    eta = np.empty((nA * nB,), dtype=np.float64)
    P = np.empty((nA * nB, 3), dtype=np.float64)
    cK = np.empty((nA * nB,), dtype=np.float64)

    idx = 0
    for ia in range(nA):
        a = float(expA[ia])
        ca = float(coefA[ia])
        for ib in range(nB):
            b = float(expB[ib])
            cb = float(coefB[ib])
            e = a + b
            inv_e = 1.0 / e
            mu = a * b * inv_e
            Px = (a * cA[0] + b * cB[0]) * inv_e
            Py = (a * cA[1] + b * cB[1]) * inv_e
            Pz = (a * cA[2] + b * cB[2]) * inv_e
            Kab = np.exp(-mu * AB2)

            eta[idx] = e
            P[idx, 0] = Px
            P[idx, 1] = Py
            P[idx, 2] = Pz
            cK[idx] = (ca * cb) * Kab
            idx += 1

    return PairTable(eta=eta, P=P, cK=cK)


def eri_ssss(basis: BasisSoA, A: int, B: int, C: int, D: int) -> float:
    """Closed-form contracted (ss|ss) ERI under the BasisSoA convention."""

    pairAB = _build_pair_table_ss(basis, A, B)
    pairCD = _build_pair_table_ss(basis, C, D)

    eta = pairAB.eta
    zeta = pairCD.eta
    P = pairAB.P
    Q = pairCD.P

    # Prefactor constant.
    c = 2.0 * (pi**2.5)

    out = 0.0
    for i in range(eta.shape[0]):
        ei = float(eta[i])
        cKi = float(pairAB.cK[i])
        Pi = P[i]
        for j in range(zeta.shape[0]):
            zj = float(zeta[j])
            cKj = float(pairCD.cK[j])
            Qj = Q[j]
            PQ2 = float(np.dot(Pi - Qj, Pi - Qj))
            denom = ei + zj
            omega = ei * zj / denom
            T = omega * PQ2
            pref = c / (ei * zj * sqrt(denom))
            out += pref * cKi * cKj * boys_f0(T)
    return float(out)


def schwarz_ssss(basis: BasisSoA, A: int, B: int) -> float:
    """Exact Schwarz bound Q_AB = sqrt((AB|AB)) for s-shell pairs."""

    val = eri_ssss(basis, A, B, A, B)
    return float(sqrt(max(val, 0.0)))


__all__ = ["eri_ssss", "schwarz_ssss"]

