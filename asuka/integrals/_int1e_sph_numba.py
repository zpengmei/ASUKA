from __future__ import annotations

"""Numba-accelerated AO 1e derivative integrals for spherical-harmonic GTOs.

This module mirrors ``_int1e_cart_numba.py`` but produces derivative tensors
in the **spherical** AO basis, ``dX[atom, xyz, i_sph, j_sph]``.

Strategy: compute Cartesian shell-pair tiles (same primitive loops as Cart),
then contract with the block-diagonal cart→sph transform ``T_l`` per shell:

    tile_sph = T_A^T @ tile_cart @ T_B

The ``T_all`` tensor is a padded ``(lmax+1, max_ncart, max_nsph)`` array
pre-computed once and passed in; the per-shell slicing costs nothing.
"""

import math

import numpy as np

import numba as nb  # type: ignore

HAS_NUMBA = True

# Re-use low-level helpers from the Cartesian numba module.
from asuka.integrals._int1e_cart_numba import (  # noqa: E402
    _build_R_coulomb,
    _hermite_E_1d_table,
    _kin3d_from_overlap,
    _ncart,
    _overlap_1d_table,
)


# ---------------------------------------------------------------------------
# Small helpers for nsph and cart2sph inside numba
# ---------------------------------------------------------------------------
@nb.njit(cache=True, inline="always")
def _nsph(l: int) -> int:
    return 2 * l + 1


@nb.njit(cache=True)
def _transform_tile_sph(
    tile_cart: np.ndarray,  # (3, nA_c, nB_c)
    T_all: np.ndarray,      # (lmax+1, max_nc, max_ns)
    la: int,
    lb: int,
    nA_c: int,
    nB_c: int,
    nA_s: int,
    nB_s: int,
) -> np.ndarray:
    """Transform a (3, nA_cart, nB_cart) Cartesian tile to (3, nA_sph, nB_sph)."""
    out = np.zeros((3, nA_s, nB_s), dtype=np.float64)
    for x in range(3):
        # Two-step matmul: tmp = T_A^T @ tile[x], then out[x] = tmp @ T_B
        # Step 1: tmp[i_s, j_c] = sum_k T_A[k, i_s] * tile[x, k, j_c]
        for i_s in range(nA_s):
            for j_c in range(nB_c):
                s = 0.0
                for k in range(nA_c):
                    s += T_all[la, k, i_s] * tile_cart[x, k, j_c]
                # Step 2 inline: out[x, i_s, j_s] = sum_l tmp[i_s, l] * T_B[l, j_s]
                for j_s in range(nB_s):
                    out[x, i_s, j_s] += s * T_all[lb, j_c, j_s]
    return out


# ---------------------------------------------------------------------------
# build_dS_sph_numba
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def build_dS_sph_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start_sph: np.ndarray,
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
    nao_sph: int,
    T_all: np.ndarray,
) -> np.ndarray:
    dS = np.zeros((natm, 3, nao_sph, nao_sph), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        nA_c = _ncart(la)
        nB_c = _ncart(lb)
        nA_s = _nsph(la)
        nB_s = _nsph(lb)

        aoA = int(shell_ao_start_sph[shA])
        aoB = int(shell_ao_start_sph[shB])
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
        Ax = float(shell_cxyz[shA, 0])
        Ay = float(shell_cxyz[shA, 1])
        Az = float(shell_cxyz[shA, 2])
        Bx = float(shell_cxyz[shB, 0])
        By = float(shell_cxyz[shB, 1])
        Bz = float(shell_cxyz[shB, 2])

        tileA = np.zeros((3, nA_c, nB_c), dtype=np.float64)
        tileB = np.zeros((3, nA_c, nB_c), dtype=np.float64)

        for ia in range(nprimA):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(nprimB):
                b = float(expB[ib])
                cb = float(coefB[ib])
                Sx = _overlap_1d_table(la + 1, lb + 1, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 1, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 1, a, b, Az, Bz)
                c = ca * cb
                for i in range(nA_c):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB_c):
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
                        tileA[0, i, j] += c * dAx * S_yz
                        tileA[1, i, j] += c * dAy * S_xz
                        tileA[2, i, j] += c * dAz * S_xy
                        tileB[0, i, j] += c * dBx * S_yz
                        tileB[1, i, j] += c * dBy * S_xz
                        tileB[2, i, j] += c * dBz * S_xy

        # Cart → Sph transform
        sphA = _transform_tile_sph(tileA, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)
        sphB = _transform_tile_sph(tileB, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)

        for x in range(3):
            for i_s in range(nA_s):
                for j_s in range(nB_s):
                    dS[atomA, x, aoA + i_s, aoB + j_s] += sphA[x, i_s, j_s]
                    dS[atomB, x, aoA + i_s, aoB + j_s] += sphB[x, i_s, j_s]
                    if shA != shB:
                        dS[atomA, x, aoB + j_s, aoA + i_s] += sphA[x, i_s, j_s]
                        dS[atomB, x, aoB + j_s, aoA + i_s] += sphB[x, i_s, j_s]
    return dS


# ---------------------------------------------------------------------------
# build_dT_sph_numba
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def build_dT_sph_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start_sph: np.ndarray,
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
    nao_sph: int,
    T_all: np.ndarray,
) -> np.ndarray:
    dT = np.zeros((natm, 3, nao_sph, nao_sph), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        nA_c = _ncart(la)
        nB_c = _ncart(lb)
        nA_s = _nsph(la)
        nB_s = _nsph(lb)

        aoA = int(shell_ao_start_sph[shA])
        aoB = int(shell_ao_start_sph[shB])
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
        Ax = float(shell_cxyz[shA, 0])
        Ay = float(shell_cxyz[shA, 1])
        Az = float(shell_cxyz[shA, 2])
        Bx = float(shell_cxyz[shB, 0])
        By = float(shell_cxyz[shB, 1])
        Bz = float(shell_cxyz[shB, 2])

        tileA = np.zeros((3, nA_c, nB_c), dtype=np.float64)
        tileB = np.zeros((3, nA_c, nB_c), dtype=np.float64)

        for ia in range(nprimA):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(nprimB):
                b = float(expB[ib])
                cb = float(coefB[ib])
                Sx = _overlap_1d_table(la + 1, lb + 3, a, b, Ax, Bx)
                Sy = _overlap_1d_table(la + 1, lb + 3, a, b, Ay, By)
                Sz = _overlap_1d_table(la + 1, lb + 3, a, b, Az, Bz)
                c = ca * cb
                for i in range(nA_c):
                    lax = int(comp_lx[offA + i])
                    lay = int(comp_ly[offA + i])
                    laz = int(comp_lz[offA + i])
                    for j in range(nB_c):
                        lbx = int(comp_lx[offB + j])
                        lby = int(comp_ly[offB + j])
                        lbz = int(comp_lz[offB + j])
                        # d/dA
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax + 1, lay, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax - 1, lay, laz, lbx, lby, lbz, b) if lax else 0.0
                        tileA[0, i, j] += c * (2.0 * a * t_p - float(lax) * t_m)
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay + 1, laz, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay - 1, laz, lbx, lby, lbz, b) if lay else 0.0
                        tileA[1, i, j] += c * (2.0 * a * t_p - float(lay) * t_m)
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz + 1, lbx, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz - 1, lbx, lby, lbz, b) if laz else 0.0
                        tileA[2, i, j] += c * (2.0 * a * t_p - float(laz) * t_m)
                        # d/dB
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx + 1, lby, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx - 1, lby, lbz, b) if lbx else 0.0
                        tileB[0, i, j] += c * (2.0 * b * t_p - float(lbx) * t_m)
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby + 1, lbz, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby - 1, lbz, b) if lby else 0.0
                        tileB[1, i, j] += c * (2.0 * b * t_p - float(lby) * t_m)
                        t_p = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz + 1, b)
                        t_m = _kin3d_from_overlap(Sx, Sy, Sz, lax, lay, laz, lbx, lby, lbz - 1, b) if lbz else 0.0
                        tileB[2, i, j] += c * (2.0 * b * t_p - float(lbz) * t_m)

        sphA = _transform_tile_sph(tileA, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)
        sphB = _transform_tile_sph(tileB, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)

        for x in range(3):
            for i_s in range(nA_s):
                for j_s in range(nB_s):
                    dT[atomA, x, aoA + i_s, aoB + j_s] += sphA[x, i_s, j_s]
                    dT[atomB, x, aoA + i_s, aoB + j_s] += sphB[x, i_s, j_s]
                    if shA != shB:
                        dT[atomA, x, aoB + j_s, aoA + i_s] += sphA[x, i_s, j_s]
                        dT[atomB, x, aoB + j_s, aoA + i_s] += sphB[x, i_s, j_s]
    return dT


# ---------------------------------------------------------------------------
# build_dV_sph_numba
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def build_dV_sph_numba(
    shell_cxyz: np.ndarray,
    shell_prim_start: np.ndarray,
    shell_nprim: np.ndarray,
    shell_l: np.ndarray,
    shell_ao_start_sph: np.ndarray,
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
    nao_sph: int,
    include_operator_deriv: bool,
    T_all: np.ndarray,
) -> np.ndarray:
    dV = np.zeros((natm, 3, nao_sph, nao_sph), dtype=np.float64)
    npair = int(pairA.shape[0])
    for idx in nb.prange(npair):
        shA = int(pairA[idx])
        shB = int(pairB[idx])
        la = int(shell_l[shA])
        lb = int(shell_l[shB])
        nA_c = _ncart(la)
        nB_c = _ncart(lb)
        nA_s = _nsph(la)
        nB_s = _nsph(lb)

        aoA = int(shell_ao_start_sph[shA])
        aoB = int(shell_ao_start_sph[shB])
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
        Ax = float(shell_cxyz[shA, 0])
        Ay = float(shell_cxyz[shA, 1])
        Az = float(shell_cxyz[shA, 2])
        Bx = float(shell_cxyz[shB, 0])
        By = float(shell_cxyz[shB, 1])
        Bz = float(shell_cxyz[shB, 2])

        L = la + lb
        nmax = L + 1
        natm_local = int(atom_coords.shape[0])

        tileA = np.zeros((3, nA_c, nB_c), dtype=np.float64)
        tileB = np.zeros((3, nA_c, nB_c), dtype=np.float64)
        tileC = np.zeros((natm_local, 3, nA_c, nB_c), dtype=np.float64)

        for ia in range(nprimA):
            a = float(expA[ia])
            ca = float(coefA[ia])
            for ib in range(nprimB):
                b = float(expB[ib])
                cb = float(coefB[ib])
                p = a + b
                inv_p = 1.0 / p
                Px = (a * Ax + b * Bx) * inv_p
                Py = (a * Ay + b * By) * inv_p
                Pz = (a * Az + b * Bz) * inv_p
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

                    for i in range(nA_c):
                        lax = int(comp_lx[offA + i])
                        lay = int(comp_ly[offA + i])
                        laz = int(comp_lz[offA + i])
                        for j in range(nB_c):
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

                            vAx = scale * (-Z) * sAx
                            vAy = scale * (-Z) * sAy
                            vAz = scale * (-Z) * sAz
                            vBx = scale * (-Z) * sBx
                            vBy = scale * (-Z) * sBy
                            vBz = scale * (-Z) * sBz
                            tileA[0, i, j] += vAx
                            tileA[1, i, j] += vAy
                            tileA[2, i, j] += vAz
                            tileB[0, i, j] += vBx
                            tileB[1, i, j] += vBy
                            tileB[2, i, j] += vBz
                            if include_operator_deriv:
                                tileC[ic, 0, i, j] += scale * (+Z) * sCx
                                tileC[ic, 1, i, j] += scale * (+Z) * sCy
                                tileC[ic, 2, i, j] += scale * (+Z) * sCz

        # Cart → Sph transform
        sphA = _transform_tile_sph(tileA, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)
        sphB = _transform_tile_sph(tileB, T_all, la, lb, nA_c, nB_c, nA_s, nB_s)

        for x in range(3):
            for i_s in range(nA_s):
                for j_s in range(nB_s):
                    dV[atomA, x, aoA + i_s, aoB + j_s] += sphA[x, i_s, j_s]
                    dV[atomB, x, aoA + i_s, aoB + j_s] += sphB[x, i_s, j_s]
                    if shA != shB:
                        dV[atomA, x, aoB + j_s, aoA + i_s] += sphA[x, i_s, j_s]
                        dV[atomB, x, aoB + j_s, aoA + i_s] += sphB[x, i_s, j_s]

        if include_operator_deriv:
            for ic in range(natm_local):
                sphC = _transform_tile_sph(tileC[ic], T_all, la, lb, nA_c, nB_c, nA_s, nB_s)
                for x in range(3):
                    for i_s in range(nA_s):
                        for j_s in range(nB_s):
                            dV[ic, x, aoA + i_s, aoB + j_s] += sphC[x, i_s, j_s]
                            if shA != shB:
                                dV[ic, x, aoB + j_s, aoA + i_s] += sphC[x, i_s, j_s]

    return dV


# ---------------------------------------------------------------------------
# T-matrix packing utility (called from Python, not numba)
# ---------------------------------------------------------------------------
def pack_cart2sph_matrices(lmax: int) -> np.ndarray:
    """Build padded ``T_all[lmax+1, max_ncart, max_nsph]`` for numba kernels."""
    from asuka.cueri.cart import ncart as _ncart_py  # noqa: PLC0415
    from asuka.cueri.sph import cart2sph_matrix as _c2s  # noqa: PLC0415

    lmax = int(lmax)
    max_nc = max(_ncart_py(l) for l in range(lmax + 1)) if lmax >= 0 else 1
    max_ns = max(2 * l + 1 for l in range(lmax + 1)) if lmax >= 0 else 1
    T_all = np.zeros((lmax + 1, max_nc, max_ns), dtype=np.float64)
    for l in range(lmax + 1):
        nc = _ncart_py(l)
        ns = 2 * l + 1
        T_l = np.asarray(_c2s(l), dtype=np.float64)  # (ncart, nsph)
        T_all[l, :nc, :ns] = T_l
    return T_all


__all__ = [
    "HAS_NUMBA",
    "build_dS_sph_numba",
    "build_dT_sph_numba",
    "build_dV_sph_numba",
    "pack_cart2sph_matrices",
]
