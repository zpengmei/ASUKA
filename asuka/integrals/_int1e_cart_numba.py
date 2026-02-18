from __future__ import annotations

"""Numba-accelerated AO 1e integrals for Cartesian GTOs.

This module is imported lazily by `asuka.integrals.int1e_cart` when the backend
is set to "numba" (or "auto" with numba available).
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

import numba as nb  # type: ignore

HAS_NUMBA = True


def build_comp_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened (lx,ly,lz) tables for all l<=lmax in PySCF cart order."""

    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be >= 0")

    # start offset per l in the flattened comp arrays
    start = np.zeros((lmax + 2,), dtype=np.int32)
    total = 0
    for l in range(lmax + 1):
        start[l] = total
        total += (l + 1) * (l + 2) // 2
    start[lmax + 1] = total

    lx = np.empty((total,), dtype=np.int16)
    ly = np.empty((total,), dtype=np.int16)
    lz = np.empty((total,), dtype=np.int16)

    off = 0
    for l in range(lmax + 1):
        for lxi in range(l, -1, -1):
            for lyi in range(l - lxi, -1, -1):
                lzi = l - lxi - lyi
                lx[off] = lxi
                ly[off] = lyi
                lz[off] = lzi
                off += 1
    return start, lx, ly, lz


@nb.njit(cache=True)
def contract_dS_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
    M: np.ndarray,
) -> np.ndarray:
    out = np.zeros((natm, 3), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in range(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])
                Sx = _overlap_1d_table(la + 1, lb + 1, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 1, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 1, a, b, Az, Bz)
                c = ca * cb

                for i in range(nA):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])

                        S_yz = Sy[lay, lby] * Sz[laz, lbz]
                        S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                        S_xy = Sx[lax, lbx] * Sy[lay, lby]

                        dAx = 2.0 * a * Sx[lax + 1, lbx]
                        if lax:
                            dAx -= float(lax) * Sx[lax - 1, lbx]
                        dAy = 2.0 * a * Sy[lay + 1, lby]
                        if lay:
                            dAy -= float(lay) * Sy[lay - 1, lby]
                        dAz = 2.0 * a * Sz[laz + 1, lbz]
                        if laz:
                            dAz -= float(laz) * Sz[laz - 1, lbz]

                        dBx = 2.0 * b * Sx[lax, lbx + 1]
                        if lbx:
                            dBx -= float(lbx) * Sx[lax, lbx - 1]
                        dBy = 2.0 * b * Sy[lay, lby + 1]
                        if lby:
                            dBy -= float(lby) * Sy[lay, lby - 1]
                        dBz = 2.0 * b * Sz[laz, lbz + 1]
                        if lbz:
                            dBz -= float(lbz) * Sz[laz, lbz - 1]

                        vAx = c * dAx * S_yz
                        vAy = c * dAy * S_xz
                        vAz = c * dAz * S_xy
                        vBx = c * dBx * S_yz
                        vBy = c * dBy * S_xz
                        vBz = c * dBz * S_xy

                        i0 = aoA + i
                        j0 = aoB + j
                        if i0 >= nao or j0 >= nao:  # pragma: no cover
                            continue
                        m = M[i0, j0]
                        if shA != shB:
                            m += M[j0, i0]

                        out[atomA, 0] += vAx * m
                        out[atomA, 1] += vAy * m
                        out[atomA, 2] += vAz * m
                        out[atomB, 0] += vBx * m
                        out[atomB, 1] += vBy * m
                        out[atomB, 2] += vBz * m

    return out

@nb.njit(cache=True)
def contract_dT_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
    M: np.ndarray,
) -> np.ndarray:
    out = np.zeros((natm, 3), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in range(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])

                Sx = _overlap_1d_table(la + 1, lb + 3, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 3, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 3, a, b, Az, Bz)
                c = ca * cb

                for i in range(nA):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax + 1, lay, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax - 1, lay, laz, lbx, lby, lbz, b) if lax else 0.0
                        vAx = c * (2.0 * a * t_p - float(lax) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay + 1, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay - 1, laz, lbx, lby, lbz, b) if lay else 0.0
                        vAy = c * (2.0 * a * t_p - float(lay) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz + 1, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz - 1, lbx, lby, lbz, b) if laz else 0.0
                        vAz = c * (2.0 * a * t_p - float(laz) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx + 1, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx - 1, lby, lbz, b) if lbx else 0.0
                        vBx = c * (2.0 * b * t_p - float(lbx) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby + 1, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby - 1, lbz, b) if lby else 0.0
                        vBy = c * (2.0 * b * t_p - float(lby) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz + 1, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz - 1, b) if lbz else 0.0
                        vBz = c * (2.0 * b * t_p - float(lbz) * t_m)

                        i0 = aoA + i
                        j0 = aoB + j
                        if i0 >= nao or j0 >= nao:  # pragma: no cover
                            continue
                        m = M[i0, j0]
                        if shA != shB:
                            m += M[j0, i0]

                        out[atomA, 0] += vAx * m
                        out[atomA, 1] += vAy * m
                        out[atomA, 2] += vAz * m
                        out[atomB, 0] += vBx * m
                        out[atomB, 1] += vBy * m
                        out[atomB, 2] += vBz * m

    return out

@nb.njit(cache=True)
def contract_dV_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    atom_coords: np.ndarray,
    atom_charges: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
    include_operator_deriv: bool,
    M: np.ndarray,
) -> np.ndarray:
    out = np.zeros((natm, 3), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in range(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        la1 = la + 1
        lb1 = lb + 1
        L = la + lb
        nmax = L + 1

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])
                p = a + b
                inv_p = 1.0 / p
                Px = (a * Ax + b * Bx) * inv_p
                Py = (a * Ay + b * By) * inv_p
                Pz = (a * Az + b * Bz) * inv_p

                Ex = _hermite_E_1d_table(la1, lb1, a, b, Ax, Bx)
                Ey = _hermite_E_1d_table(la1, lb1, a, b, Ay, By)
                Ez = _hermite_E_1d_table(la1, lb1, a, b, Az, Bz)

                c = ca * cb
                pref = (2.0 * math.pi) * inv_p
                scale = c * pref

                for ic in range(natm):
                    Z = float(atom_charges[ic])
                    if Z == 0.0:
                        continue
                    Cx = float(atom_coords[ic, 0])
                    Cy = float(atom_coords[ic, 1])
                    Cz = float(atom_coords[ic, 2])
                    R = _build_R_coulomb(p, Px - Cx, Py - Cy, Pz - Cz, nmax)
                    R0 = R[0]

                    for i in range(nA):
                        lax = int(comp_lx[offA + i])
                        lay = int(comp_ly[offA + i])
                        laz = int(comp_lz[offA + i])
                        for j in range(nB):
                            lbx = int(comp_lx[offB + j])
                            lby = int(comp_ly[offB + j])
                            lbz = int(comp_lz[offB + j])

                            sAx = 0.0
                            sAy = 0.0
                            sAz = 0.0
                            sBx = 0.0
                            sBy = 0.0
                            sBz = 0.0
                            sCx = 0.0
                            sCy = 0.0
                            sCz = 0.0

                            for t in range(lax + lbx + 2):
                                if t >= Ex.shape[2]:
                                    break
                                ex = Ex[lax, lbx, t]
                                ex_ip1 = Ex[lax + 1, lbx, t] if (lax + 1) <= la1 else 0.0
                                ex_im1 = Ex[lax - 1, lbx, t] if lax else 0.0
                                ex_jp1 = Ex[lax, lbx + 1, t] if (lbx + 1) <= lb1 else 0.0
                                ex_jm1 = Ex[lax, lbx - 1, t] if lbx else 0.0
                                dEx_dAx = 2.0 * a * ex_ip1 - float(lax) * ex_im1
                                dEx_dBx = 2.0 * b * ex_jp1 - float(lbx) * ex_jm1

                                for u in range(lay + lby + 2):
                                    if u >= Ey.shape[2]:
                                        break
                                    ey = Ey[lay, lby, u]
                                    ey_ip1 = Ey[lay + 1, lby, u] if (lay + 1) <= la1 else 0.0
                                    ey_im1 = Ey[lay - 1, lby, u] if lay else 0.0
                                    ey_jp1 = Ey[lay, lby + 1, u] if (lby + 1) <= lb1 else 0.0
                                    ey_jm1 = Ey[lay, lby - 1, u] if lby else 0.0
                                    dEy_dAy = 2.0 * a * ey_ip1 - float(lay) * ey_im1
                                    dEy_dBy = 2.0 * b * ey_jp1 - float(lby) * ey_jm1

                                    for v in range(laz + lbz + 2):
                                        if v >= Ez.shape[2]:
                                            break
                                        ez = Ez[laz, lbz, v]
                                        ez_ip1 = Ez[laz + 1, lbz, v] if (laz + 1) <= la1 else 0.0
                                        ez_im1 = Ez[laz - 1, lbz, v] if laz else 0.0
                                        ez_jp1 = Ez[laz, lbz + 1, v] if (lbz + 1) <= lb1 else 0.0
                                        ez_jm1 = Ez[laz, lbz - 1, v] if lbz else 0.0
                                        dEz_dAz = 2.0 * a * ez_ip1 - float(laz) * ez_im1
                                        dEz_dBz = 2.0 * b * ez_jp1 - float(lbz) * ez_jm1

                                        r = R0[t, u, v]

                                        sAx += dEx_dAx * ey * ez * r
                                        sAy += ex * dEy_dAy * ez * r
                                        sAz += ex * ey * dEz_dAz * r

                                        sBx += dEx_dBx * ey * ez * r
                                        sBy += ex * dEy_dBy * ez * r
                                        sBz += ex * ey * dEz_dBz * r

                                        if include_operator_deriv:
                                            if t + 1 <= nmax:
                                                sCx += ex * ey * ez * R0[t + 1, u, v]
                                            if u + 1 <= nmax:
                                                sCy += ex * ey * ez * R0[t, u + 1, v]
                                            if v + 1 <= nmax:
                                                sCz += ex * ey * ez * R0[t, u, v + 1]

                            i0 = aoA + i
                            j0 = aoB + j
                            if i0 >= nao or j0 >= nao:  # pragma: no cover
                                continue
                            m = M[i0, j0]
                            if shA != shB:
                                m += M[j0, i0]

                            vAx = scale * (-Z) * sAx
                            vAy = scale * (-Z) * sAy
                            vAz = scale * (-Z) * sAz
                            vBx = scale * (-Z) * sBx
                            vBy = scale * (-Z) * sBy
                            vBz = scale * (-Z) * sBz

                            out[atomA, 0] += vAx * m
                            out[atomA, 1] += vAy * m
                            out[atomA, 2] += vAz * m
                            out[atomB, 0] += vBx * m
                            out[atomB, 1] += vBy * m
                            out[atomB, 2] += vBz * m

                            if include_operator_deriv:
                                vCx = scale * (+Z) * sCx
                                vCy = scale * (+Z) * sCy
                                vCz = scale * (+Z) * sCz
                                out[ic, 0] += vCx * m
                                out[ic, 1] += vCy * m
                                out[ic, 2] += vCz * m

    return out

@nb.njit(cache=True)
def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2

@nb.njit(cache=True)
def _boys_f0(T: float) -> float:
    if T < 1e-12:
        return 1.0 - (T / 3.0) + (T * T / 10.0)
    return 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))

@nb.njit(cache=True)
def _boys_fm_list(T: float, m_max: int) -> np.ndarray:
    out = np.empty((m_max + 1,), dtype=np.float64)
    if m_max == 0:
        out[0] = _boys_f0(T)
        return out
    if T < 5.0:
        term = 1.0
        Fm = 0.0
        for k in range(120):
            Fm += term / float(2 * m_max + 2 * k + 1)
            term *= -T / float(k + 1)
        out[m_max] = Fm
        e = math.exp(-T)
        for m in range(m_max, 0, -1):
            out[m - 1] = (2.0 * T * out[m] + e) / float(2 * m - 1)
        return out

    out[0] = _boys_f0(T)
    e = math.exp(-T)
    for m in range(1, m_max + 1):
        out[m] = (float(2 * m - 1) * out[m - 1] - e) / (2.0 * T)
    return out

@nb.njit(cache=True)
def _overlap_1d_table(la: int, lb: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    p = a + b
    inv_p = 1.0 / p
    mu = a * b * inv_p
    Px = (a * Ax + b * Bx) * inv_p
    PA = Px - Ax
    PB = Px - Bx
    AB = Ax - Bx
    s00 = math.sqrt(math.pi * inv_p) * math.exp(-mu * AB * AB)

    out = np.zeros((la + 1, lb + 1), dtype=np.float64)
    out[0, 0] = s00
    inv_2p = 0.5 * inv_p

    for j in range(lb):
        out[0, j + 1] = PB * out[0, j]
        if j > 0:
            out[0, j + 1] += float(j) * inv_2p * out[0, j - 1]

    for i in range(la):
        out[i + 1, 0] = PA * out[i, 0]
        if i > 0:
            out[i + 1, 0] += float(i) * inv_2p * out[i - 1, 0]
        for j in range(lb):
            out[i + 1, j + 1] = PA * out[i, j + 1]
            if i > 0:
                out[i + 1, j + 1] += float(i) * inv_2p * out[i - 1, j + 1]
            out[i + 1, j + 1] += float(j + 1) * inv_2p * out[i, j]
    return out

@nb.njit(cache=True)
def _hermite_E_1d_table(la: int, lb: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    p = a + b
    inv_p = 1.0 / p
    mu = a * b * inv_p
    Px = (a * Ax + b * Bx) * inv_p
    PA = Px - Ax
    PB = Px - Bx
    AB = Ax - Bx

    tmax = la + lb
    E = np.zeros((la + 1, lb + 1, tmax + 1), dtype=np.float64)
    E[0, 0, 0] = math.exp(-mu * AB * AB)

    inv_2p = 0.5 * inv_p

    for i in range(la):
        prev = E[i, 0]
        cur = E[i + 1, 0]
        for t in range(i + 2):
            val = PA * prev[t]
            if t > 0:
                val += inv_2p * prev[t - 1]
            if t + 1 <= i:
                val += float(t + 1) * prev[t + 1]
            cur[t] = val

    for i in range(la + 1):
        for j in range(lb):
            prev = E[i, j]
            cur = E[i, j + 1]
            tmax_ij = i + (j + 1)
            for t in range(tmax_ij + 1):
                val = PB * prev[t]
                if t > 0:
                    val += inv_2p * prev[t - 1]
                if t + 1 <= i + j:
                    val += float(t + 1) * prev[t + 1]
                cur[t] = val
    return E

@nb.njit(cache=True)
def _build_R_coulomb(p: float, PCx: float, PCy: float, PCz: float, nmax: int) -> np.ndarray:
    R = np.zeros((nmax + 1, nmax + 1, nmax + 1, nmax + 1), dtype=np.float64)
    T = p * (PCx * PCx + PCy * PCy + PCz * PCz)
    F = _boys_fm_list(T, nmax)
    fac = -2.0 * p
    pow_fac = 1.0
    for n in range(nmax + 1):
        R[n, 0, 0, 0] = pow_fac * F[n]
        pow_fac *= fac

    for n in range(nmax - 1, -1, -1):
        max_m = nmax - n
        for t in range(max_m + 1):
            for u in range(max_m - t + 1):
                for v in range(max_m - t - u + 1):
                    if t == 0 and u == 0 and v == 0:
                        continue
                    if t > 0:
                        val = PCx * R[n + 1, t - 1, u, v]
                        if t >= 2:
                            val += float(t - 1) * R[n + 1, t - 2, u, v]
                        R[n, t, u, v] = val
                    elif u > 0:
                        val = PCy * R[n + 1, t, u - 1, v]
                        if u >= 2:
                            val += float(u - 1) * R[n + 1, t, u - 2, v]
                        R[n, t, u, v] = val
                    else:
                        val = PCz * R[n + 1, t, u, v - 1]
                        if v >= 2:
                            val += float(v - 1) * R[n + 1, t, u, v - 2]
                        R[n, t, u, v] = val
    return R

@nb.njit(cache=True)
def _kin1d_from_overlap(S: np.ndarray, i: int, j: int, b: float) -> float:
    val = b * (2.0 * float(j) + 1.0) * S[i, j] - 2.0 * b * b * S[i, j + 2]
    if j >= 2:
        val -= 0.5 * float(j * (j - 1)) * S[i, j - 2]
    return val

@nb.njit(cache=True, inline="always")
def _kin3d_from_overlap(
    Sx: np.ndarray,
    Sy: np.ndarray,
    Sz: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    jx: int,
    jy: int,
    jz: int,
    b: float,
) -> float:
    Tx = _kin1d_from_overlap(Sx, ix, jx, b)
    Ty = _kin1d_from_overlap(Sy, iy, jy, b)
    Tz = _kin1d_from_overlap(Sz, iz, jz, b)
    return Tx * Sy[iy, jy] * Sz[iz, jz] + Sx[ix, jx] * Ty * Sz[iz, jz] + Sx[ix, jx] * Sy[iy, jy] * Tz

@nb.njit(cache=True)
def _fill_tile_S(
    tile: np.ndarray,
    la: int,
    lb: int,
    cA: np.ndarray,
    cB: np.ndarray,
    expA: np.ndarray,
    coefA: np.ndarray,
    expB: np.ndarray,
    coefB: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
) -> None:
    nA = _ncart(la)
    nB = _ncart(lb)
    offA = int(comp_start[la])
    offB = int(comp_start[lb])

    Ax, Ay, Az = float(cA[0]), float(cA[1]), float(cA[2])
    Bx, By, Bz = float(cB[0]), float(cB[1]), float(cB[2])

    for ia in range(expA.shape[0]):
        a = float(expA[ia])
        ca = float(coefA[ia])
        for ib in range(expB.shape[0]):
            b = float(expB[ib])
            cb = float(coefB[ib])
            Sx = _overlap_1d_table(la, lb, a, b, Ax, Bx)
            Sy = _overlap_1d_table(la, lb, a, b, Ay, By)
            Sz = _overlap_1d_table(la, lb, a, b, Az, Bz)
            c = ca * cb
            for i in range(nA):
                lax = int(comp_lx[offA + i])
                lay = int(comp_ly[offA + i])
                laz = int(comp_lz[offA + i])
                for j in range(nB):
                    lbx = int(comp_lx[offB + j])
                    lby = int(comp_ly[offB + j])
                    lbz = int(comp_lz[offB + j])
                    tile[i, j] += c * Sx[lax, lbx] * Sy[lay, lby] * Sz[laz, lbz]

@nb.njit(cache=True)
def _fill_tile_T(
    tile: np.ndarray,
    la: int,
    lb: int,
    cA: np.ndarray,
    cB: np.ndarray,
    expA: np.ndarray,
    coefA: np.ndarray,
    expB: np.ndarray,
    coefB: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
) -> None:
    nA = _ncart(la)
    nB = _ncart(lb)
    offA = int(comp_start[la])
    offB = int(comp_start[lb])

    Ax, Ay, Az = float(cA[0]), float(cA[1]), float(cA[2])
    Bx, By, Bz = float(cB[0]), float(cB[1]), float(cB[2])

    for ia in range(expA.shape[0]):
        a = float(expA[ia])
        ca = float(coefA[ia])
        for ib in range(expB.shape[0]):
            b = float(expB[ib])
            cb = float(coefB[ib])
            # overlap tables to lb+2 for kinetic on ket
            Sx = _overlap_1d_table(la, lb + 2, a, b, Ax, Bx)
            Sy = _overlap_1d_table(la, lb + 2, a, b, Ay, By)
            Sz = _overlap_1d_table(la, lb + 2, a, b, Az, Bz)
            c = ca * cb
            for i in range(nA):
                lax = int(comp_lx[offA + i])
                lay = int(comp_ly[offA + i])
                laz = int(comp_lz[offA + i])
                for j in range(nB):
                    lbx = int(comp_lx[offB + j])
                    lby = int(comp_ly[offB + j])
                    lbz = int(comp_lz[offB + j])

                    Tx = _kin1d_from_overlap(Sx, lax, lbx, b)
                    Ty = _kin1d_from_overlap(Sy, lay, lby, b)
                    Tz = _kin1d_from_overlap(Sz, laz, lbz, b)
                    tile[i, j] += c * (
                        Tx * Sy[lay, lby] * Sz[laz, lbz]
                        + Sx[lax, lbx] * Ty * Sz[laz, lbz]
                        + Sx[lax, lbx] * Sy[lay, lby] * Tz
                    )

@nb.njit(cache=True)
def _fill_tile_V(
    tile: np.ndarray,
    la: int,
    lb: int,
    cA: np.ndarray,
    cB: np.ndarray,
    expA: np.ndarray,
    coefA: np.ndarray,
    expB: np.ndarray,
    coefB: np.ndarray,
    atom_coords: np.ndarray,
    atom_charges: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
) -> None:
    nA = _ncart(la)
    nB = _ncart(lb)
    offA = int(comp_start[la])
    offB = int(comp_start[lb])

    Ax, Ay, Az = float(cA[0]), float(cA[1]), float(cA[2])
    Bx, By, Bz = float(cB[0]), float(cB[1]), float(cB[2])

    natm = int(atom_coords.shape[0])
    L = la + lb

    for ia in range(expA.shape[0]):
        a = float(expA[ia])
        ca = float(coefA[ia])
        for ib in range(expB.shape[0]):
            b = float(expB[ib])
            cb = float(coefB[ib])
            p = a + b
            inv_p = 1.0 / p
            Px = (a * Ax + b * Bx) * inv_p
            Py = (a * Ay + b * By) * inv_p
            Pz = (a * Az + b * Bz) * inv_p

            Ex = _hermite_E_1d_table(la, lb, a, b, Ax, Bx)
            Ey = _hermite_E_1d_table(la, lb, a, b, Ay, By)
            Ez = _hermite_E_1d_table(la, lb, a, b, Az, Bz)

            c = ca * cb
            pref = (2.0 * math.pi) * inv_p

            for ic in range(natm):
                Z = float(atom_charges[ic])
                if Z == 0.0:
                    continue
                Cx = float(atom_coords[ic, 0])
                Cy = float(atom_coords[ic, 1])
                Cz = float(atom_coords[ic, 2])
                R = _build_R_coulomb(p, Px - Cx, Py - Cy, Pz - Cz, L)
                R0 = R[0]

                for i in range(nA):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])
                        s = 0.0
                        for t in range(lax + lbx + 1):
                            ex = Ex[lax, lbx, t]
                            if ex == 0.0:
                                continue
                            for u in range(lay + lby + 1):
                                ey = Ey[lay, lby, u]
                                if ey == 0.0:
                                    continue
                                for v in range(laz + lbz + 1):
                                    ez = Ez[laz, lbz, v]
                                    if ez == 0.0:
                                        continue
                                    s += ex * ey * ez * R0[t, u, v]
                        tile[i, j] += c * (-Z) * pref * s

@nb.njit(cache=True, parallel=True)
def build_S_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
) -> np.ndarray:
    out = np.zeros((nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        tile = np.zeros((nA, nB), dtype=np.float64)
        _fill_tile_S(tile, la, lb, shell_cxyz[shA], shell_cxyz[shB], expA, coefA, expB, coefB, comp_start, comp_lx, comp_ly, comp_lz)

        out[aoA : aoA + nA, aoB : aoB + nB] = tile
        if shA != shB:
            out[aoB : aoB + nB, aoA : aoA + nA] = tile.T
    return out

@nb.njit(cache=True, parallel=True)
def build_T_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
) -> np.ndarray:
    out = np.zeros((nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        tile = np.zeros((nA, nB), dtype=np.float64)
        _fill_tile_T(tile, la, lb, shell_cxyz[shA], shell_cxyz[shB], expA, coefA, expB, coefB, comp_start, comp_lx, comp_ly, comp_lz)

        out[aoA : aoA + nA, aoB : aoB + nB] = tile
        if shA != shB:
            out[aoB : aoB + nB, aoA : aoA + nA] = tile.T
    return out

@nb.njit(cache=True, parallel=True)
def build_V_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    atom_coords: np.ndarray,
    atom_charges: np.ndarray,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
) -> np.ndarray:
    out = np.zeros((nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        tile = np.zeros((nA, nB), dtype=np.float64)
        _fill_tile_V(
            tile,
            la,
            lb,
            shell_cxyz[shA],
            shell_cxyz[shB],
            expA,
            coefA,
            expB,
            coefB,
            atom_coords,
            atom_charges,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
        )

        out[aoA : aoA + nA, aoB : aoB + nB] = tile
        if shA != shB:
            out[aoB : aoB + nB, aoA : aoA + nA] = tile.T
    return out

@nb.njit(cache=True, parallel=True)
def build_dS_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
) -> np.ndarray:
    dS = np.zeros((natm, 3, nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])
                # Need shifts +/-1 on either bra or ket.
                Sx = _overlap_1d_table(la + 1, lb + 1, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 1, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 1, a, b, Az, Bz)
                c = ca * cb

                for i in range(nA):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])

                        S_yz = Sy[lay, lby] * Sz[laz, lbz]
                        S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                        S_xy = Sx[lax, lbx] * Sy[lay, lby]

                        # d/dA
                        dAx = 2.0 * a * Sx[lax + 1, lbx]
                        if lax:
                            dAx -= float(lax) * Sx[lax - 1, lbx]
                        dAy = 2.0 * a * Sy[lay + 1, lby]
                        if lay:
                            dAy -= float(lay) * Sy[lay - 1, lby]
                        dAz = 2.0 * a * Sz[laz + 1, lbz]
                        if laz:
                            dAz -= float(laz) * Sz[laz - 1, lbz]

                        # d/dB
                        dBx = 2.0 * b * Sx[lax, lbx + 1]
                        if lbx:
                            dBx -= float(lbx) * Sx[lax, lbx - 1]
                        dBy = 2.0 * b * Sy[lay, lby + 1]
                        if lby:
                            dBy -= float(lby) * Sy[lay, lby - 1]
                        dBz = 2.0 * b * Sz[laz, lbz + 1]
                        if lbz:
                            dBz -= float(lbz) * Sz[laz, lbz - 1]

                        vAx = c * dAx * S_yz
                        vAy = c * dAy * S_xz
                        vAz = c * dAz * S_xy
                        vBx = c * dBx * S_yz
                        vBy = c * dBy * S_xz
                        vBz = c * dBz * S_xy

                        i0 = aoA + i
                        j0 = aoB + j
                        dS[atomA, 0, i0, j0] += vAx
                        dS[atomA, 1, i0, j0] += vAy
                        dS[atomA, 2, i0, j0] += vAz
                        dS[atomB, 0, i0, j0] += vBx
                        dS[atomB, 1, i0, j0] += vBy
                        dS[atomB, 2, i0, j0] += vBz
                        if shA != shB:
                            dS[atomA, 0, j0, i0] += vAx
                            dS[atomA, 1, j0, i0] += vAy
                            dS[atomA, 2, j0, i0] += vAz
                            dS[atomB, 0, j0, i0] += vBx
                            dS[atomB, 1, j0, i0] += vBy
                            dS[atomB, 2, j0, i0] += vBz
    return dS

@nb.njit(cache=True, parallel=True)
def build_dT_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
) -> np.ndarray:
    dT = np.zeros((natm, 3, nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])

                # Need:
                # - bra +/-1 => la+1
                # - ket +/-1 and kinetic needs j+2 => (lb+1)+2 = lb+3
                Sx = _overlap_1d_table(la + 1, lb + 3, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 3, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 3, a, b, Az, Bz)
                c = ca * cb

                for i in range(nA):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])

                        # d/dA axis
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax + 1, lay, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax - 1, lay, laz, lbx, lby, lbz, b) if lax else 0.0
                        vAx = c * (2.0 * a * t_p - float(lax) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay + 1, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay - 1, laz, lbx, lby, lbz, b) if lay else 0.0
                        vAy = c * (2.0 * a * t_p - float(lay) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz + 1, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz - 1, lbx, lby, lbz, b) if laz else 0.0
                        vAz = c * (2.0 * a * t_p - float(laz) * t_m)

                        # d/dB axis
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx + 1, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx - 1, lby, lbz, b) if lbx else 0.0
                        vBx = c * (2.0 * b * t_p - float(lbx) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby + 1, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby - 1, lbz, b) if lby else 0.0
                        vBy = c * (2.0 * b * t_p - float(lby) * t_m)

                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz + 1, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz - 1, b) if lbz else 0.0
                        vBz = c * (2.0 * b * t_p - float(lbz) * t_m)

                        i0 = aoA + i
                        j0 = aoB + j
                        dT[atomA, 0, i0, j0] += vAx
                        dT[atomA, 1, i0, j0] += vAy
                        dT[atomA, 2, i0, j0] += vAz
                        dT[atomB, 0, i0, j0] += vBx
                        dT[atomB, 1, i0, j0] += vBy
                        dT[atomB, 2, i0, j0] += vBz
                        if shA != shB:
                            dT[atomA, 0, j0, i0] += vAx
                            dT[atomA, 1, j0, i0] += vAy
                            dT[atomA, 2, j0, i0] += vAz
                            dT[atomB, 0, j0, i0] += vBx
                            dT[atomB, 1, j0, i0] += vBy
                            dT[atomB, 2, j0, i0] += vBz
    return dT

@nb.njit(cache=True, parallel=True)
def build_dV_cart_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    atom_coords: np.ndarray,
    atom_charges: np.ndarray,
    shell_atom: np.ndarray,
    natm: int,
    comp_start: np.ndarray,
    comp_lx: np.ndarray,
    comp_ly: np.ndarray,
    comp_lz: np.ndarray,
    pairA: np.ndarray,
    pairB: np.ndarray,
    nao: int,
    include_operator_deriv: bool,
) -> np.ndarray:
    dV = np.zeros((natm, 3, nao, nao), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        aoA = int(shell_ao_start[shA])
        aoB = int(shell_ao_start[shB])
        nA = _ncart(la)
        nB = _ncart(lb)

        atomA = int(shell_atom[shA])
        atomB = int(shell_atom[shB])

        sA = int(shell_prim_start[shA])
        nprimA = int(shell_nprim[shA])
        expA = prim_exp[sA : sA + nprimA]
        coefA = prim_coef[sA : sA + nprimA]

        sB = int(shell_prim_start[shB])
        nprimB = int(shell_nprim[shB])
        expB = prim_exp[sB : sB + nprimB]
        coefB = prim_coef[sB : sB + nprimB]

        offA = int(comp_start[la])
        offB = int(comp_start[lb])

        Ax, Ay, Az = float(shell_cxyz[shA, 0]), float(shell_cxyz[shA, 1]), float(shell_cxyz[shA, 2])
        Bx, By, Bz = float(shell_cxyz[shB, 0]), float(shell_cxyz[shB, 1]), float(shell_cxyz[shB, 2])

        L = la + lb
        nmax = L + 1
        natm_local = int(atom_coords.shape[0])

        for ia in range(expA.shape[0]):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(expB.shape[0]):
                b = float(expB[ib])
                cb = float(coefB[ib])
                p = a + b
                inv_p = 1.0 / p
                Px = (a * Ax + b * Bx) * inv_p
                Py = (a * Ay + b * By) * inv_p
                Pz = (a * Az + b * Bz) * inv_p

                # Hermite E tables with +1 for basis derivatives.
                Ex = _hermite_E_1d_table(la + 1, lb + 1, a, b, Ax, Bx)
                Ey = _hermite_E_1d_table(la + 1, lb + 1, a, b, Ay, By)
                Ez = _hermite_E_1d_table(la + 1, lb + 1, a, b, Az, Bz)

                c = ca * cb
                pref = (2.0 * math.pi) * inv_p
                scale = c * pref

                for ic in range(natm_local):
                    Z = float(atom_charges[ic])
                    if Z == 0.0:
                        continue
                    Cx = float(atom_coords[ic, 0])
                    Cy = float(atom_coords[ic, 1])
                    Cz = float(atom_coords[ic, 2])
                    R = _build_R_coulomb(p, Px - Cx, Py - Cy, Pz - Cz, nmax)
                    R0 = R[0]

                    for i in range(nA):
                        lax = int(comp_lx[offA + i])
                        lay = int(comp_ly[offA + i])
                        laz = int(comp_lz[offA + i])
                        for j in range(nB):
                            lbx = int(comp_lx[offB + j])
                            lby = int(comp_ly[offB + j])
                            lbz = int(comp_lz[offB + j])

                            sAx = 0.0
                            sAy = 0.0
                            sAz = 0.0
                            sBx = 0.0
                            sBy = 0.0
                            sBz = 0.0
                            sCx = 0.0
                            sCy = 0.0
                            sCz = 0.0

                            # Use +1 bounds (see python impl) to capture derivative-induced t/u/v shifts.
                            for t in range(lax + lbx + 2):
                                ex = Ex[lax, lbx, t] if t < Ex.shape[2] else 0.0
                                ex_ip1 = Ex[lax + 1, lbx, t] if (lax + 1) <= la + 1 and t < Ex.shape[2] else 0.0
                                ex_im1 = Ex[lax - 1, lbx, t] if lax and t < Ex.shape[2] else 0.0
                                ex_jp1 = Ex[lax, lbx + 1, t] if (lbx + 1) <= lb + 1 and t < Ex.shape[2] else 0.0
                                ex_jm1 = Ex[lax, lbx - 1, t] if lbx and t < Ex.shape[2] else 0.0

                                dEx_dAx = 2.0 * a * ex_ip1 - float(lax) * ex_im1
                                dEx_dBx = 2.0 * b * ex_jp1 - float(lbx) * ex_jm1

                                for u in range(lay + lby + 2):
                                    ey = Ey[lay, lby, u] if u < Ey.shape[2] else 0.0
                                    ey_ip1 = Ey[lay + 1, lby, u] if (lay + 1) <= la + 1 and u < Ey.shape[2] else 0.0
                                    ey_im1 = Ey[lay - 1, lby, u] if lay and u < Ey.shape[2] else 0.0
                                    ey_jp1 = Ey[lay, lby + 1, u] if (lby + 1) <= lb + 1 and u < Ey.shape[2] else 0.0
                                    ey_jm1 = Ey[lay, lby - 1, u] if lby and u < Ey.shape[2] else 0.0

                                    dEy_dAy = 2.0 * a * ey_ip1 - float(lay) * ey_im1
                                    dEy_dBy = 2.0 * b * ey_jp1 - float(lby) * ey_jm1

                                    for v in range(laz + lbz + 2):
                                        ez = Ez[laz, lbz, v] if v < Ez.shape[2] else 0.0
                                        ez_ip1 = Ez[laz + 1, lbz, v] if (laz + 1) <= la + 1 and v < Ez.shape[2] else 0.0
                                        ez_im1 = Ez[laz - 1, lbz, v] if laz and v < Ez.shape[2] else 0.0
                                        ez_jp1 = Ez[laz, lbz + 1, v] if (lbz + 1) <= lb + 1 and v < Ez.shape[2] else 0.0
                                        ez_jm1 = Ez[laz, lbz - 1, v] if lbz and v < Ez.shape[2] else 0.0

                                        dEz_dAz = 2.0 * a * ez_ip1 - float(laz) * ez_im1
                                        dEz_dBz = 2.0 * b * ez_jp1 - float(lbz) * ez_jm1

                                        r = R0[t, u, v]

                                        sAx += dEx_dAx * ey * ez * r
                                        sAy += ex * dEy_dAy * ez * r
                                        sAz += ex * ey * dEz_dAz * r

                                        sBx += dEx_dBx * ey * ez * r
                                        sBy += ex * dEy_dBy * ez * r
                                        sBz += ex * ey * dEz_dBz * r

                                        if include_operator_deriv:
                                            if t + 1 <= nmax:
                                                sCx += ex * ey * ez * R0[t + 1, u, v]
                                            if u + 1 <= nmax:
                                                sCy += ex * ey * ez * R0[t, u + 1, v]
                                            if v + 1 <= nmax:
                                                sCz += ex * ey * ez * R0[t, u, v + 1]

                            i0 = aoA + i
                            j0 = aoB + j

                            vAx = scale * (-Z) * sAx
                            vAy = scale * (-Z) * sAy
                            vAz = scale * (-Z) * sAz
                            vBx = scale * (-Z) * sBx
                            vBy = scale * (-Z) * sBy
                            vBz = scale * (-Z) * sBz
                            dV[atomA, 0, i0, j0] += vAx
                            dV[atomA, 1, i0, j0] += vAy
                            dV[atomA, 2, i0, j0] += vAz
                            dV[atomB, 0, i0, j0] += vBx
                            dV[atomB, 1, i0, j0] += vBy
                            dV[atomB, 2, i0, j0] += vBz

                            if include_operator_deriv:
                                vCx = scale * (+Z) * sCx
                                vCy = scale * (+Z) * sCy
                                vCz = scale * (+Z) * sCz
                                dV[ic, 0, i0, j0] += vCx
                                dV[ic, 1, i0, j0] += vCy
                                dV[ic, 2, i0, j0] += vCz

                            if shA != shB:
                                dV[atomA, 0, j0, i0] += vAx
                                dV[atomA, 1, j0, i0] += vAy
                                dV[atomA, 2, j0, i0] += vAz
                                dV[atomB, 0, j0, i0] += vBx
                                dV[atomB, 1, j0, i0] += vBy
                                dV[atomB, 2, j0, i0] += vBz
                                if include_operator_deriv:
                                    dV[ic, 0, j0, i0] += vCx
                                    dV[ic, 1, j0, i0] += vCy
                                    dV[ic, 2, j0, i0] += vCz

    return dV


__all__ = [
    "HAS_NUMBA",
    "build_comp_tables",
]

__all__ += [
    "build_S_cart_numba",
    "build_T_cart_numba",
    "build_V_cart_numba",
    "build_dS_cart_numba",
    "build_dT_cart_numba",
    "build_dV_cart_numba",
    "contract_dS_cart_numba",
    "contract_dT_cart_numba",
    "contract_dV_cart_numba",
]
