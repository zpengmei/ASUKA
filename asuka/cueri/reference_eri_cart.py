from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from .basis_cart import BasisCartSoA
from .boys import boys_fm_list
from .cart import ncart


@dataclass(frozen=True)
class PairTable:
    eta: np.ndarray  # float64, shape (nPair,)
    P: np.ndarray  # float64, shape (nPair, 3)
    cK: np.ndarray  # float64, shape (nPair,)


def _build_pair_table(basis: BasisCartSoA, A: int, B: int) -> PairTable:
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


def _k_two_pi_to_five_halves() -> float:
    return 2.0 * (pi**2.5)


def _boys_f0_to_f4(T: float) -> tuple[float, float, float, float, float]:
    F = boys_fm_list(T, 4)
    return float(F[0]), float(F[1]), float(F[2]), float(F[3]), float(F[4])


def _t3_component(i: int, j: int, k: int, d: tuple[float, float, float], term_f2: float, term_f3: float) -> float:
    di = float(d[i])
    dj = float(d[j])
    dk = float(d[k])
    t_f2 = term_f2 * ((dj if i == k else 0.0) + (di if j == k else 0.0) + (dk if i == j else 0.0))
    t_f3 = term_f3 * (di * dj * dk)
    return t_f2 + t_f3


def _t4_component(
    i: int,
    j: int,
    k: int,
    l: int,
    d: tuple[float, float, float],
    term_f2: float,
    term_f3: float,
    term_f4: float,
) -> float:
    di = float(d[i])
    dj = float(d[j])
    dk = float(d[k])
    dl = float(d[l])
    t_f2 = term_f2 * float((i == j and k == l) + (i == k and j == l) + (i == l and j == k))
    t_f3 = term_f3 * (
        (dk * dl if i == j else 0.0)
        + (dj * dl if i == k else 0.0)
        + (dj * dk if i == l else 0.0)
        + (di * dl if j == k else 0.0)
        + (di * dk if j == l else 0.0)
        + (di * dj if k == l else 0.0)
    )
    t_f4 = term_f4 * (di * dj * dk * dl)
    return t_f2 + t_f3 + t_f4


def eri_int2e_cart_tile(
    basis: BasisCartSoA,
    shellA: int,
    shellB: int,
    shellC: int,
    shellD: int,
) -> np.ndarray:
    """Internal CPU reference tile for selected Step-2 class coverage (cart=True).

    Supported explicit class orders:
    - (0,0,0,0) ssss
    - (1,0,0,0) psss
    - (1,1,0,0) ppss
    - (1,0,1,0) psps
    - (1,1,1,0) ppps
    - (1,1,1,1) pppp
    - (2,0,0,0) dsss

    Returns
    - tile: float64 array of shape (ncartA, ncartB, ncartC, ncartD)
    """

    A = int(shellA)
    B = int(shellB)
    C = int(shellC)
    D = int(shellD)

    la = int(basis.shell_l[A])
    lb = int(basis.shell_l[B])
    lc = int(basis.shell_l[C])
    ld = int(basis.shell_l[D])

    if la < 0 or lb < 0 or lc < 0 or ld < 0:
        raise ValueError("shell angular momentum must be >= 0")

    nA = ncart(la)
    nB = ncart(lb)
    nC = ncart(lc)
    nD = ncart(ld)

    pairAB = _build_pair_table(basis, A, B)
    pairCD = _build_pair_table(basis, C, D)

    eta = pairAB.eta
    zeta = pairCD.eta
    P = pairAB.P
    Q = pairCD.P

    # Prefactor constant for 4-center Coulomb ERIs.
    c = _k_two_pi_to_five_halves()

    # (ss|ss)
    if (la, lb, lc, ld) == (0, 0, 0, 0):
        out = 0.0
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Pi = P[i]
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qj = Q[j]
                dx = float(Pi[0] - Qj[0])
                dy = float(Pi[1] - Qj[1])
                dz = float(Pi[2] - Qj[2])
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                F0, _, _, _, _ = _boys_f0_to_f4(T)
                out += pref * cKi * cKj * F0
        return np.asarray(out, dtype=np.float64).reshape((1, 1, 1, 1))

    # Common center coords.
    Ax, Ay, Az = map(float, basis.shell_cxyz[A])
    Bx, By, Bz = map(float, basis.shell_cxyz[B])
    Cx, Cy, Cz = map(float, basis.shell_cxyz[C])
    Dx, Dy, Dz = map(float, basis.shell_cxyz[D])

    # (ps|ss)
    if (la, lb, lc, ld) == (1, 0, 0, 0):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, _, _, _ = _boys_f0_to_f4(T)
                q_over = q / denom  # omega/p
                sx += base * (-(Ax - Px) * F0 - q_over * dx * F1)
                sy += base * (-(Ay - Py) * F0 - q_over * dy * F1)
                sz += base * (-(Az - Pz) * F0 - q_over * dz * F1)
        # A(x,y,z)
        return np.asarray([sx, sy, sz], dtype=np.float64).reshape((3, 1, 1, 1))

    # (pp|ss)
    if (la, lb, lc, ld) == (1, 1, 0, 0):
        s = np.zeros((3, 3), dtype=np.float64)
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, F2, _, _ = _boys_f0_to_f4(T)

                I = base * F0
                omega_over_p = omega / p
                Jx = -omega_over_p * base * F1 * dx
                Jy = -omega_over_p * base * F1 * dy
                Jz = -omega_over_p * base * F1 * dz

                inv4p2 = 1.0 / (4.0 * p * p)
                w2 = omega * omega
                t4 = 4.0 * w2 * F2
                t2 = 2.0 * omega * F1

                Kxx = (base * (t4 * dx * dx - t2) + 2.0 * p * I) * inv4p2
                Kyy = (base * (t4 * dy * dy - t2) + 2.0 * p * I) * inv4p2
                Kzz = (base * (t4 * dz * dz - t2) + 2.0 * p * I) * inv4p2
                Kxy = (base * (t4 * dx * dy)) * inv4p2
                Kxz = (base * (t4 * dx * dz)) * inv4p2
                Kyz = (base * (t4 * dy * dz)) * inv4p2

                PAx = Px - Ax
                PAy = Py - Ay
                PAz = Pz - Az
                PBx = Px - Bx
                PBy = Py - By
                PBz = Pz - Bz

                s[0, 0] += Kxx + PAx * Jx + PBx * Jx + (PAx * PBx) * I
                s[0, 1] += Kxy + PAx * Jy + PBy * Jx + (PAx * PBy) * I
                s[0, 2] += Kxz + PAx * Jz + PBz * Jx + (PAx * PBz) * I

                s[1, 0] += Kxy + PAy * Jx + PBx * Jy + (PAy * PBx) * I
                s[1, 1] += Kyy + PAy * Jy + PBy * Jy + (PAy * PBy) * I
                s[1, 2] += Kyz + PAy * Jz + PBz * Jy + (PAy * PBz) * I

                s[2, 0] += Kxz + PAz * Jx + PBx * Jz + (PAz * PBx) * I
                s[2, 1] += Kyz + PAz * Jy + PBy * Jz + (PAz * PBy) * I
                s[2, 2] += Kzz + PAz * Jz + PBz * Jz + (PAz * PBz) * I

        # AB is (A-major, B-minor). CD is scalar.
        return s.reshape((3, 3, 1, 1))

    # (ps|ps)
    if (la, lb, lc, ld) == (1, 0, 1, 0):
        s = np.zeros((3, 3), dtype=np.float64)
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, F2, _, _ = _boys_f0_to_f4(T)

                I = base * F0
                omega_over_p = omega / p
                omega_over_q = omega / q
                Jpx = -omega_over_p * base * F1 * dx
                Jpy = -omega_over_p * base * F1 * dy
                Jpz = -omega_over_p * base * F1 * dz
                Jqx = omega_over_q * base * F1 * dx
                Jqy = omega_over_q * base * F1 * dy
                Jqz = omega_over_q * base * F1 * dz

                inv4pq = 1.0 / (4.0 * p * q)
                w2 = omega * omega
                t4 = 4.0 * w2 * F2
                t2 = 2.0 * omega * F1
                Hxx = base * (t4 * dx * dx - t2)
                Hyy = base * (t4 * dy * dy - t2)
                Hzz = base * (t4 * dz * dz - t2)
                Hxy = base * (t4 * dx * dy)
                Hxz = base * (t4 * dx * dz)
                Hyz = base * (t4 * dy * dz)
                Lxx = -(Hxx) * inv4pq
                Lyy = -(Hyy) * inv4pq
                Lzz = -(Hzz) * inv4pq
                Lxy = -(Hxy) * inv4pq
                Lxz = -(Hxz) * inv4pq
                Lyz = -(Hyz) * inv4pq

                PAx = Px - Ax
                PAy = Py - Ay
                PAz = Pz - Az
                QCx = Qx - Cx
                QCy = Qy - Cy
                QCz = Qz - Cz

                # (p_i s | p_k s) : M_ik + (Q_k-C_k)Jp_i + (P_i-A_i)Jq_k + (P_i-A_i)(Q_k-C_k) I
                s[0, 0] += Lxx + QCx * Jpx + PAx * Jqx + (PAx * QCx) * I
                s[0, 1] += Lxy + QCy * Jpx + PAx * Jqy + (PAx * QCy) * I
                s[0, 2] += Lxz + QCz * Jpx + PAx * Jqz + (PAx * QCz) * I

                s[1, 0] += Lxy + QCx * Jpy + PAy * Jqx + (PAy * QCx) * I
                s[1, 1] += Lyy + QCy * Jpy + PAy * Jqy + (PAy * QCy) * I
                s[1, 2] += Lyz + QCz * Jpy + PAy * Jqz + (PAy * QCz) * I

                s[2, 0] += Lxz + QCx * Jpz + PAz * Jqx + (PAz * QCx) * I
                s[2, 1] += Lyz + QCy * Jpz + PAz * Jqy + (PAz * QCy) * I
                s[2, 2] += Lzz + QCz * Jpz + PAz * Jqz + (PAz * QCz) * I

        return s.reshape((3, 1, 3, 1))

    # (pp|ps) and (pp|pp) are more verbose; for reference baseline we only support the explicit CUDA-covered classes.
    if (la, lb, lc, ld) == (1, 1, 1, 0):
        s = np.zeros((9, 3), dtype=np.float64)  # AB=9, CD=3
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                dvec = (dx, dy, dz)
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, F2, F3, _F4 = _boys_f0_to_f4(T)
                I = base * F0

                omega_over_p = omega / p
                omega_over_q = omega / q
                Jp = (
                    -omega_over_p * base * F1 * dx,
                    -omega_over_p * base * F1 * dy,
                    -omega_over_p * base * F1 * dz,
                )
                Jq = (
                    omega_over_q * base * F1 * dx,
                    omega_over_q * base * F1 * dy,
                    omega_over_q * base * F1 * dz,
                )

                w2 = omega * omega
                w3 = w2 * omega
                inv4p2 = 1.0 / (4.0 * p * p)
                inv4pq = 1.0 / (4.0 * p * q)

                t4 = 4.0 * w2 * F2
                t2 = 2.0 * omega * F1
                H = (
                    (base * (t4 * dx * dx - t2), base * (t4 * dx * dy), base * (t4 * dx * dz)),
                    (base * (t4 * dy * dx), base * (t4 * dy * dy - t2), base * (t4 * dy * dz)),
                    (base * (t4 * dz * dx), base * (t4 * dz * dy), base * (t4 * dz * dz - t2)),
                )

                Kp = [[0.0, 0.0, 0.0] for _ in range(3)]
                L = [[0.0, 0.0, 0.0] for _ in range(3)]
                for a in range(3):
                    for b in range(3):
                        dij = 1.0 if a == b else 0.0
                        Kp[a][b] = (H[a][b] + 2.0 * p * I * dij) * inv4p2
                        L[a][b] = -(H[a][b]) * inv4pq

                PA = (Px - Ax, Py - Ay, Pz - Az)
                PB = (Px - Bx, Py - By, Pz - Bz)
                QC = (Qx - Cx, Qy - Cy, Qz - Cz)

                term_t3_f2 = 4.0 * w2 * base * F2
                term_t3_f3 = -8.0 * w3 * base * F3

                for ia in range(3):
                    for ib in range(3):
                        ab = ia * 3 + ib
                        a = float(PA[ia])
                        b = float(PB[ib])
                        dij = 1.0 if ia == ib else 0.0
                        Kp_ij = float(Kp[ia][ib])
                        for ic in range(3):
                            c_qc = float(QC[ic])
                            T3_ijk = _t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3)
                            M_ijk = (-T3_ijk + 4.0 * p * q * dij * float(Jq[ic])) / (8.0 * p * p * q)
                            val = (
                                M_ijk
                                + c_qc * Kp_ij
                                + b * float(L[ia][ic])
                                + b * c_qc * float(Jp[ia])
                                + a * float(L[ib][ic])
                                + a * c_qc * float(Jp[ib])
                                + a * b * float(Jq[ic])
                                + a * b * c_qc * I
                            )
                            s[ab, ic] += val

        return s.reshape((3, 3, 3, 1))

    if (la, lb, lc, ld) == (1, 1, 1, 1):
        s = np.zeros((9, 9), dtype=np.float64)  # AB=9, CD=9
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                dvec = (dx, dy, dz)
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, F2, F3, F4 = _boys_f0_to_f4(T)
                I = base * F0

                omega_over_p = omega / p
                omega_over_q = omega / q
                Jp = (
                    -omega_over_p * base * F1 * dx,
                    -omega_over_p * base * F1 * dy,
                    -omega_over_p * base * F1 * dz,
                )
                Jq = (
                    omega_over_q * base * F1 * dx,
                    omega_over_q * base * F1 * dy,
                    omega_over_q * base * F1 * dz,
                )

                w2 = omega * omega
                w3 = w2 * omega
                w4 = w2 * w2
                inv4p2 = 1.0 / (4.0 * p * p)
                inv4q2 = 1.0 / (4.0 * q * q)
                inv4pq = 1.0 / (4.0 * p * q)

                t4 = 4.0 * w2 * F2
                t2 = 2.0 * omega * F1
                H = (
                    (base * (t4 * dx * dx - t2), base * (t4 * dx * dy), base * (t4 * dx * dz)),
                    (base * (t4 * dy * dx), base * (t4 * dy * dy - t2), base * (t4 * dy * dz)),
                    (base * (t4 * dz * dx), base * (t4 * dz * dy), base * (t4 * dz * dz - t2)),
                )

                Kp = [[0.0, 0.0, 0.0] for _ in range(3)]
                Kq = [[0.0, 0.0, 0.0] for _ in range(3)]
                L = [[0.0, 0.0, 0.0] for _ in range(3)]
                for a in range(3):
                    for b in range(3):
                        dij = 1.0 if a == b else 0.0
                        Kp[a][b] = (H[a][b] + 2.0 * p * I * dij) * inv4p2
                        Kq[a][b] = (H[a][b] + 2.0 * q * I * dij) * inv4q2
                        L[a][b] = -(H[a][b]) * inv4pq

                PA = (Px - Ax, Py - Ay, Pz - Az)
                PB = (Px - Bx, Py - By, Pz - Bz)
                QC = (Qx - Cx, Qy - Cy, Qz - Cz)
                QD = (Qx - Dx, Qy - Dy, Qz - Dz)

                term_t3_f2 = 4.0 * w2 * base * F2
                term_t3_f3 = -8.0 * w3 * base * F3
                term_t4_f2 = 4.0 * w2 * base * F2
                term_t4_f3 = -8.0 * w3 * base * F3
                term_t4_f4 = 16.0 * w4 * base * F4

                for ia in range(3):
                    for ib in range(3):
                        ab = ia * 3 + ib
                        a = float(PA[ia])
                        b = float(PB[ib])
                        dij = 1.0 if ia == ib else 0.0
                        Kp_ij = float(Kp[ia][ib])
                        for ic in range(3):
                            for id in range(3):
                                cd = ic * 3 + id
                                c_qc = float(QC[ic])
                                d_qd = float(QD[id])
                                dkl = 1.0 if ic == id else 0.0
                                Kq_kl = float(Kq[ic][id])

                                T3_ijk = _t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3)
                                T3_ijl = _t3_component(ia, ib, id, dvec, term_t3_f2, term_t3_f3)
                                T3_i_kl = _t3_component(ia, ic, id, dvec, term_t3_f2, term_t3_f3)
                                T3_j_kl = _t3_component(ib, ic, id, dvec, term_t3_f2, term_t3_f3)

                                M_ijk = (-T3_ijk + 4.0 * p * q * dij * float(Jq[ic])) / (8.0 * p * p * q)
                                M_ijl = (-T3_ijl + 4.0 * p * q * dij * float(Jq[id])) / (8.0 * p * p * q)
                                N_i_kl = (T3_i_kl + 4.0 * p * q * dkl * float(Jp[ia])) / (8.0 * p * q * q)
                                N_j_kl = (T3_j_kl + 4.0 * p * q * dkl * float(Jp[ib])) / (8.0 * p * q * q)

                                T4_ijkl = _t4_component(ia, ib, ic, id, dvec, term_t4_f2, term_t4_f3, term_t4_f4)
                                M4_ij_kl = (
                                    T4_ijkl
                                    + 8.0 * p * p * q * dkl * Kp_ij
                                    + 8.0 * p * q * q * dij * Kq_kl
                                    - 4.0 * p * q * dij * dkl * I
                                ) / (16.0 * p * p * q * q)

                                val = (
                                    M4_ij_kl
                                    + d_qd * M_ijk
                                    + c_qc * M_ijl
                                    + c_qc * d_qd * Kp_ij
                                    + b * N_i_kl
                                    + b * d_qd * float(L[ia][ic])
                                    + b * c_qc * float(L[ia][id])
                                    + b * c_qc * d_qd * float(Jp[ia])
                                    + a * N_j_kl
                                    + a * d_qd * float(L[ib][ic])
                                    + a * c_qc * float(L[ib][id])
                                    + a * c_qc * d_qd * float(Jp[ib])
                                    + a * b * Kq_kl
                                    + a * b * d_qd * float(Jq[ic])
                                    + a * b * c_qc * float(Jq[id])
                                    + a * b * c_qc * d_qd * I
                                )
                                s[ab, cd] += val

        return s.reshape((3, 3, 3, 3))

    # (ds|ss)
    if (la, lb, lc, ld) == (2, 0, 0, 0):
        # d components in cart order: xx,xy,xz,yy,yz,zz
        out = np.zeros((6,), dtype=np.float64)
        for i in range(int(eta.shape[0])):
            p = float(eta[i])
            cKi = float(pairAB.cK[i])
            Px, Py, Pz = map(float, P[i])
            for j in range(int(zeta.shape[0])):
                q = float(zeta[j])
                cKj = float(pairCD.cK[j])
                Qx, Qy, Qz = map(float, Q[j])
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                denom = p + q
                omega = p * q / denom
                T = omega * PQ2
                pref = c / (p * q * sqrt(denom))
                base = pref * cKi * cKj
                F0, F1, F2, _, _ = _boys_f0_to_f4(T)

                I = base * F0
                omega_over_p = omega / p
                Jx = -omega_over_p * base * F1 * dx
                Jy = -omega_over_p * base * F1 * dy
                Jz = -omega_over_p * base * F1 * dz

                inv4p2 = 1.0 / (4.0 * p * p)
                w2 = omega * omega
                t4 = 4.0 * w2 * F2
                t2 = 2.0 * omega * F1
                Kxx = (base * (t4 * dx * dx - t2) + 2.0 * p * I) * inv4p2
                Kyy = (base * (t4 * dy * dy - t2) + 2.0 * p * I) * inv4p2
                Kzz = (base * (t4 * dz * dz - t2) + 2.0 * p * I) * inv4p2
                Kxy = (base * (t4 * dx * dy)) * inv4p2
                Kxz = (base * (t4 * dx * dz)) * inv4p2
                Kyz = (base * (t4 * dy * dz)) * inv4p2

                PAx = Px - Ax
                PAy = Py - Ay
                PAz = Pz - Az

                # (d|ss): second-order moment on A: (r_i-A_i)(r_j-A_j) = K_ij + (P_i-A_i)J_j + (P_j-A_j)J_i + (P_i-A_i)(P_j-A_j)I
                d_xx = Kxx + 2.0 * PAx * Jx + (PAx * PAx) * I
                d_xy = Kxy + PAx * Jy + PAy * Jx + (PAx * PAy) * I
                d_xz = Kxz + PAx * Jz + PAz * Jx + (PAx * PAz) * I
                d_yy = Kyy + 2.0 * PAy * Jy + (PAy * PAy) * I
                d_yz = Kyz + PAy * Jz + PAz * Jy + (PAy * PAz) * I
                d_zz = Kzz + 2.0 * PAz * Jz + (PAz * PAz) * I

                out[0] += d_xx
                out[1] += d_xy
                out[2] += d_xz
                out[3] += d_yy
                out[4] += d_yz
                out[5] += d_zz

        return out.reshape((6, 1, 1, 1))

    raise NotImplementedError(f"unsupported shell quartet class (la,lb,lc,ld)=({la},{lb},{lc},{ld})")


__all__ = ["eri_int2e_cart_tile"]
