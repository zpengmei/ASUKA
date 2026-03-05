from __future__ import annotations

"""AO overlap and dipole (position) 1-electron integrals (Cartesian GTOs).

This module provides the additional 1e integrals needed for LS-THC DVR grids
in the style of Parrish et al. (JCP 138, 194107 (2013)), namely the finite-
basis representation of the Cartesian position operator (x, y, z).

All routines operate on cuERI's packed Cartesian basis (`BasisCartSoA`), using
the same libcint/PySCF `cart=True` primitive normalization convention as ASUKA's
other 1e integral builders.
"""

from typing import Any

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.cart import cartesian_components, ncart
from asuka.integrals.int1e_cart import _overlap_1d_table, nao_cart_from_basis


def build_overlap_and_dipole_cart(basis: BasisCartSoA) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (S, Rx, Ry, Rz) for a Cartesian basis.

    Definitions
    ----------
    - S[p,q]  = (p|q)         = ∫  phi_p(r) phi_q(r) d^3r
    - Rx[p,q] = (p| x | q)    = ∫  phi_p(r) x phi_q(r) d^3r
    - Ry[p,q] = (p| y | q)    = ∫  phi_p(r) y phi_q(r) d^3r
    - Rz[p,q] = (p| z | q)    = ∫  phi_p(r) z phi_q(r) d^3r
    """

    nao = int(nao_cart_from_basis(basis))
    if nao <= 0:
        z = np.zeros((0, 0), dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy()

    S = np.zeros((nao, nao), dtype=np.float64)
    Rx = np.zeros((nao, nao), dtype=np.float64)
    Ry = np.zeros((nao, nao), dtype=np.float64)
    Rz = np.zeros((nao, nao), dtype=np.float64)

    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int32).ravel()
    shell_prim_start = np.asarray(basis.shell_prim_start, dtype=np.int32).ravel()
    shell_nprim = np.asarray(basis.shell_nprim, dtype=np.int32).ravel()
    shell_cxyz = np.asarray(basis.shell_cxyz, dtype=np.float64).reshape((-1, 3))
    prim_exp = np.asarray(basis.prim_exp, dtype=np.float64).ravel()
    prim_coef = np.asarray(basis.prim_coef, dtype=np.float64).ravel()

    nshell = int(shell_l.size)
    if nshell == 0:
        return S, Rx, Ry, Rz

    # Cache cart components per angular momentum for consistent ordering.
    comps_cache: dict[int, list[tuple[int, int, int]]] = {}

    for shA in range(nshell):
        la = int(shell_l[shA])
        ao0a = int(shell_ao_start[shA])
        na = int(ncart(la))
        comps_a = comps_cache.get(la)
        if comps_a is None:
            comps_a = list(cartesian_components(la))
            comps_cache[la] = comps_a

        Ax, Ay, Az = map(float, shell_cxyz[shA])
        p0a = int(shell_prim_start[shA])
        npa = int(shell_nprim[shA])
        exps_a = prim_exp[p0a : p0a + npa]
        coefs_a = prim_coef[p0a : p0a + npa]

        for shB in range(shA + 1):
            lb = int(shell_l[shB])
            ao0b = int(shell_ao_start[shB])
            nb = int(ncart(lb))
            comps_b = comps_cache.get(lb)
            if comps_b is None:
                comps_b = list(cartesian_components(lb))
                comps_cache[lb] = comps_b

            Bx, By, Bz = map(float, shell_cxyz[shB])
            p0b = int(shell_prim_start[shB])
            npb = int(shell_nprim[shB])
            exps_b = prim_exp[p0b : p0b + npb]
            coefs_b = prim_coef[p0b : p0b + npb]

            blkS = np.zeros((na, nb), dtype=np.float64)
            blkRx = np.zeros((na, nb), dtype=np.float64)
            blkRy = np.zeros((na, nb), dtype=np.float64)
            blkRz = np.zeros((na, nb), dtype=np.float64)

            # Primitive contraction.
            for a, ca in zip(exps_a.tolist(), coefs_a.tolist()):
                a = float(a)
                ca = float(ca)
                for b, cb in zip(exps_b.tolist(), coefs_b.tolist()):
                    b = float(b)
                    c = ca * float(cb)

                    # We need overlap tables up to la+1 along each axis so we
                    # can form (p|x|q) via x = (x-Ax) + Ax.
                    Sx = _overlap_1d_table(la=la + 1, lb=lb, a=a, b=b, Ax=Ax, Bx=Bx)
                    Sy = _overlap_1d_table(la=la + 1, lb=lb, a=a, b=b, Ax=Ay, Bx=By)
                    Sz = _overlap_1d_table(la=la + 1, lb=lb, a=a, b=b, Ax=Az, Bx=Bz)

                    for ia, (lxa, lya, lza) in enumerate(comps_a):
                        sx_row = Sx[lxa]
                        sy_row = Sy[lya]
                        sz_row = Sz[lza]
                        sx_row_p1 = Sx[lxa + 1]
                        sy_row_p1 = Sy[lya + 1]
                        sz_row_p1 = Sz[lza + 1]
                        for ib, (lxb, lyb, lzb) in enumerate(comps_b):
                            s = float(sx_row[lxb] * sy_row[lyb] * sz_row[lzb])
                            blkS[ia, ib] += c * s

                            rx = float((sx_row_p1[lxb] + Ax * sx_row[lxb]) * sy_row[lyb] * sz_row[lzb])
                            ry = float(sx_row[lxb] * (sy_row_p1[lyb] + Ay * sy_row[lyb]) * sz_row[lzb])
                            rz = float(sx_row[lxb] * sy_row[lyb] * (sz_row_p1[lzb] + Az * sz_row[lzb]))
                            blkRx[ia, ib] += c * rx
                            blkRy[ia, ib] += c * ry
                            blkRz[ia, ib] += c * rz

            sa = slice(ao0a, ao0a + na)
            sb = slice(ao0b, ao0b + nb)
            S[sa, sb] += blkS
            Rx[sa, sb] += blkRx
            Ry[sa, sb] += blkRy
            Rz[sa, sb] += blkRz
            if shB != shA:
                S[sb, sa] += blkS.T
                Rx[sb, sa] += blkRx.T
                Ry[sb, sa] += blkRy.T
                Rz[sb, sa] += blkRz.T

    # Enforce symmetry to remove small numerical noise.
    S = 0.5 * (S + S.T)
    Rx = 0.5 * (Rx + Rx.T)
    Ry = 0.5 * (Ry + Ry.T)
    Rz = 0.5 * (Rz + Rz.T)
    return S, Rx, Ry, Rz


__all__ = ["build_overlap_and_dipole_cart"]

