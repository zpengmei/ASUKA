from __future__ import annotations

"""Spherical-AO contractions for 1e derivative integrals.

Goal
----
When ``mol.cart=False`` (spherical working AOs), the SCF/MCSCF pipeline should
not need to back-transform AO matrices to a full Cartesian ``(nao_cart,nao_cart)``
matrix just to reuse Cartesian derivative kernels.

This module provides *contractions* of Cartesian 1e derivative integrals with a
matrix expressed in the spherical AO basis, without materializing full-sized
Cartesian AO matrices:

  g[A,x] = Σ_{μ_cart,ν_cart} (∂M_cart[μ,ν]/∂R_{A,x}) * X_cart[μ,ν]

where X_cart is obtained on-the-fly from the spherical matrix X_sph via per-shell
cart2sph blocks:

  X_cart(block A,B) = T_A @ X_sph(block A,B) @ T_B^T

Notes
-----
- The evaluation basis is still Cartesian (the packed cuERI BasisCartSoA).
- The working AO matrix is spherical with dimension ``nao_sph``.
- We intentionally provide CPU implementations first; CUDA fused versions can be
  added later if needed.
"""

from math import pi
from typing import Any

import numpy as np

from asuka.cueri.cart import cartesian_components, ncart
from asuka.cueri.sph import cart2sph_matrix, nsph
from asuka.integrals.cart2sph import compute_sph_layout_from_cart_basis
from asuka.integrals.int1e_cart import (
    _build_R_coulomb,
    _comp_tables_cached,
    _hermite_E_1d_table,
    _int1e_backend,
    _kin_cart_component,
    _overlap_1d_table,
    _shell_pairs_lower,
    nao_cart_from_basis,
    shell_to_atom_map,
)


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        return np.asarray(cp.asnumpy(a), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _cart_block_from_sph_block(M_sph_block: np.ndarray, *, la: int, lb: int) -> np.ndarray:
    """Return M_cart(block A,B) from a spherical block M_sph(block A,B)."""
    TA = np.asarray(cart2sph_matrix(int(la)), dtype=np.float64)  # (ncartA, nsphA)
    TB = np.asarray(cart2sph_matrix(int(lb)), dtype=np.float64)  # (ncartB, nsphB)
    return (TA @ M_sph_block) @ TB.T


def _validate_sph_matrix(M_sph: Any, *, nao_sph: int) -> np.ndarray:
    M = _asnumpy_f64(M_sph)
    if M.ndim != 2 or int(M.shape[0]) != int(M.shape[1]):
        raise ValueError("M_sph must be a square 2D array")
    if tuple(map(int, M.shape)) != (int(nao_sph), int(nao_sph)):
        raise ValueError("M_sph shape mismatch with nao_sph")
    return np.asarray(M, dtype=np.float64, order="C")


def _sph_block_from_cart_block(block_cart: np.ndarray, *, la: int, lb: int) -> np.ndarray:
    """Transform cartesian shell-pair derivative blocks to spherical AO blocks."""
    TA = np.asarray(cart2sph_matrix(int(la)), dtype=np.float64)  # (ncartA, nsphA)
    TB = np.asarray(cart2sph_matrix(int(lb)), dtype=np.float64)  # (ncartB, nsphB)
    blk = np.asarray(block_cart, dtype=np.float64)
    if blk.ndim == 2:
        return TA.T @ blk @ TB
    if blk.ndim == 3:
        return np.einsum("mi,xmn,nj->xij", TA, blk, TB, optimize=True)
    if blk.ndim == 4:
        return np.einsum("mi,axmn,nj->axij", TA, blk, TB, optimize=True)
    raise ValueError("block_cart must have ndim in {2,3,4}")


def _resolve_sph_layout(basis_cart, shell_ao_start_sph):
    """Return (shell_ao_start_sph, nao_sph) from basis, computing if needed."""
    if shell_ao_start_sph is None:
        return compute_sph_layout_from_cart_basis(basis_cart)
    shell_ao_start_sph = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()
    shell_l = np.asarray(basis_cart.shell_l, dtype=np.int32).ravel()
    if int(shell_l.size):
        nao_sph = int(max(int(shell_ao_start_sph[i]) + nsph(int(shell_l[i])) for i in range(int(shell_l.size))))
    else:
        nao_sph = 0
    return shell_ao_start_sph, nao_sph


def _try_numba_sph():
    """Return the numba spherical module if available, else None."""
    backend = _int1e_backend()
    if backend not in ("numba", "cython"):
        return None
    try:
        from asuka.integrals import _int1e_sph_numba as _nb  # noqa: PLC0415
    except Exception:
        return None
    if not bool(getattr(_nb, "HAS_NUMBA", False)):
        if backend == "numba":
            raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")
        return None
    return _nb


def build_dS_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
) -> np.ndarray:
    """Build overlap derivatives in spherical AO basis: ``dS[a,x,i,j]``."""
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3, 0, 0), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = _resolve_sph_layout(basis_cart, shell_ao_start_sph)

    # Fast numba path
    _nb = _try_numba_sph()
    if _nb is not None:
        lmax = int(np.max(basis_cart.shell_l)) if int(basis_cart.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        pairA, pairB = _shell_pairs_lower(int(basis_cart.shell_l.shape[0]))
        T_all = _nb.pack_cart2sph_matrices(lmax)
        return _nb.build_dS_sph_numba(
            basis_cart.shell_cxyz,
            basis_cart.shell_prim_start,
            basis_cart.shell_nprim,
            basis_cart.shell_l,
            np.asarray(shell_ao_start_sph, dtype=np.int32),
            basis_cart.prim_exp,
            basis_cart.prim_coef,
            shell_atom,
            int(natm),
            comp_start, comp_lx, comp_ly, comp_lz,
            pairA, pairB,
            int(nao_sph),
            T_all,
        )

    # Pure-Python fallback
    out = np.zeros((natm, 3, int(nao_sph), int(nao_sph)), dtype=np.float64)
    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            S_yz = Sy[lay, lby] * Sz[laz, lbz]
                            S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                            S_xy = Sx[lax, lbx] * Sy[lay, lby]

                            dAx = 2.0 * a * Sx[lax + 1, lbx]
                            if lax:
                                dAx -= float(lax) * Sx[lax - 1, lbx]
                            tileA[0, i, j] += c * dAx * S_yz

                            dAy = 2.0 * a * Sy[lay + 1, lby]
                            if lay:
                                dAy -= float(lay) * Sy[lay - 1, lby]
                            tileA[1, i, j] += c * dAy * S_xz

                            dAz = 2.0 * a * Sz[laz + 1, lbz]
                            if laz:
                                dAz -= float(laz) * Sz[laz - 1, lbz]
                            tileA[2, i, j] += c * dAz * S_xy

                            dBx = 2.0 * b * Sx[lax, lbx + 1]
                            if lbx:
                                dBx -= float(lbx) * Sx[lax, lbx - 1]
                            tileB[0, i, j] += c * dBx * S_yz

                            dBy = 2.0 * b * Sy[lay, lby + 1]
                            if lby:
                                dBy -= float(lby) * Sy[lay, lby - 1]
                            tileB[1, i, j] += c * dBy * S_xz

                            dBz = 2.0 * b * Sz[laz, lbz + 1]
                            if lbz:
                                dBz -= float(lbz) * Sz[laz, lbz - 1]
                            tileB[2, i, j] += c * dBz * S_xy

            tileA_sph = _sph_block_from_cart_block(tileA, la=la, lb=lb)
            tileB_sph = _sph_block_from_cart_block(tileB, la=la, lb=lb)

            out[atomA, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileA_sph
            out[atomB, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileB_sph
            if shA != shB:
                out[atomA, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileA_sph.transpose(0, 2, 1)
                out[atomB, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileB_sph.transpose(0, 2, 1)

    return np.asarray(out, dtype=np.float64)


def build_dT_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
) -> np.ndarray:
    """Build kinetic derivatives in spherical AO basis: ``dT[a,x,i,j]``."""
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3, 0, 0), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = _resolve_sph_layout(basis_cart, shell_ao_start_sph)

    # Fast numba path
    _nb = _try_numba_sph()
    if _nb is not None:
        lmax = int(np.max(basis_cart.shell_l)) if int(basis_cart.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        pairA, pairB = _shell_pairs_lower(int(basis_cart.shell_l.shape[0]))
        T_all = _nb.pack_cart2sph_matrices(lmax)
        return _nb.build_dT_sph_numba(
            basis_cart.shell_cxyz,
            basis_cart.shell_prim_start,
            basis_cart.shell_nprim,
            basis_cart.shell_l,
            np.asarray(shell_ao_start_sph, dtype=np.int32),
            basis_cart.prim_exp,
            basis_cart.prim_coef,
            shell_atom,
            int(natm),
            comp_start, comp_lx, comp_ly, comp_lz,
            pairA, pairB,
            int(nao_sph),
            T_all,
        )

    # Pure-Python fallback
    out = np.zeros((natm, 3, int(nao_sph), int(nao_sph)), dtype=np.float64)

    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            t_p = _kin_cart_component(
                                i_xyz=(lax + 1, lay, laz),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax - 1, lay, laz),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lax
                                else 0.0
                            )
                            tileA[0, i, j] += c * (2.0 * a * t_p - float(lax) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay + 1, laz),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay - 1, laz),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lay
                                else 0.0
                            )
                            tileA[1, i, j] += c * (2.0 * a * t_p - float(lay) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz + 1),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz - 1),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if laz
                                else 0.0
                            )
                            tileA[2, i, j] += c * (2.0 * a * t_p - float(laz) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx + 1, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx - 1, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lbx
                                else 0.0
                            )
                            tileB[0, i, j] += c * (2.0 * b * t_p - float(lbx) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx, lby + 1, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx, lby - 1, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lby
                                else 0.0
                            )
                            tileB[1, i, j] += c * (2.0 * b * t_p - float(lby) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx, lby, lbz + 1),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx, lby, lbz - 1),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lbz
                                else 0.0
                            )
                            tileB[2, i, j] += c * (2.0 * b * t_p - float(lbz) * t_m)

            tileA_sph = _sph_block_from_cart_block(tileA, la=la, lb=lb)
            tileB_sph = _sph_block_from_cart_block(tileB, la=la, lb=lb)

            out[atomA, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileA_sph
            out[atomB, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileB_sph
            if shA != shB:
                out[atomA, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileA_sph.transpose(0, 2, 1)
                out[atomB, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileB_sph.transpose(0, 2, 1)

    return np.asarray(out, dtype=np.float64)


def build_dV_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
    include_operator_deriv: bool = True,
) -> np.ndarray:
    """Build nuclear-attraction derivatives in spherical AO basis: ``dV[a,x,i,j]``."""
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")
    if natm <= 0:
        return np.zeros((0, 3, 0, 0), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = _resolve_sph_layout(basis_cart, shell_ao_start_sph)

    # Fast numba path
    _nb = _try_numba_sph()
    if _nb is not None:
        lmax = int(np.max(basis_cart.shell_l)) if int(basis_cart.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        pairA, pairB = _shell_pairs_lower(int(basis_cart.shell_l.shape[0]))
        T_all = _nb.pack_cart2sph_matrices(lmax)
        return _nb.build_dV_sph_numba(
            basis_cart.shell_cxyz,
            basis_cart.shell_prim_start,
            basis_cart.shell_nprim,
            basis_cart.shell_l,
            np.asarray(shell_ao_start_sph, dtype=np.int32),
            basis_cart.prim_exp,
            basis_cart.prim_coef,
            atom_coords_bohr,
            atom_charges,
            shell_atom,
            int(natm),
            comp_start, comp_lx, comp_ly, comp_lz,
            pairA, pairB,
            int(nao_sph),
            bool(include_operator_deriv),
            T_all,
        )

    # Pure-Python fallback
    out = np.zeros((natm, 3, int(nao_sph), int(nao_sph)), dtype=np.float64)

    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileC = np.zeros((natm, 3, nA_cart, nB_cart), dtype=np.float64) if bool(include_operator_deriv) else None

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])
                    p = a + b
                    inv_p = 1.0 / p
                    P = (a * cA + b * cB) * inv_p

                    la1 = la + 1
                    lb1 = lb + 1
                    Ex = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Ey = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Ez = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    L = la + lb
                    nmax = L + 1
                    c = ca * cb
                    pref = (2.0 * pi) * inv_p

                    for ic in range(natm):
                        Z = float(atom_charges[ic])
                        if Z == 0.0:
                            continue
                        C = atom_coords_bohr[ic]
                        R = _build_R_coulomb(p=p, PC=P - C, nmax=nmax)
                        R0 = R[0]

                        for i, (lax, lay, laz) in enumerate(compA):
                            for j, (lbx, lby, lbz) in enumerate(compB):
                                sA_x = 0.0
                                sA_y = 0.0
                                sA_z = 0.0
                                sB_x = 0.0
                                sB_y = 0.0
                                sB_z = 0.0
                                sC_x = 0.0
                                sC_y = 0.0
                                sC_z = 0.0

                                for t in range(0, lax + lbx + 2):
                                    ex = float(Ex[lax, lbx, t])
                                    ex_p = float(Ex[lax, lbx, t])
                                    ex_ip1 = float(Ex[lax + 1, lbx, t]) if (lax + 1) <= la1 else 0.0
                                    ex_im1 = float(Ex[lax - 1, lbx, t]) if lax else 0.0
                                    ex_jp1 = float(Ex[lax, lbx + 1, t]) if (lbx + 1) <= lb1 else 0.0
                                    ex_jm1 = float(Ex[lax, lbx - 1, t]) if lbx else 0.0

                                    dEx_dAx = 2.0 * a * ex_ip1 - float(lax) * ex_im1
                                    dEx_dBx = 2.0 * b * ex_jp1 - float(lbx) * ex_jm1

                                    for u in range(0, lay + lby + 2):
                                        ey = float(Ey[lay, lby, u])

                                        ey_ip1 = float(Ey[lay + 1, lby, u]) if (lay + 1) <= la1 else 0.0
                                        ey_im1 = float(Ey[lay - 1, lby, u]) if lay else 0.0
                                        ey_jp1 = float(Ey[lay, lby + 1, u]) if (lby + 1) <= lb1 else 0.0
                                        ey_jm1 = float(Ey[lay, lby - 1, u]) if lby else 0.0

                                        dEy_dAy = 2.0 * a * ey_ip1 - float(lay) * ey_im1
                                        dEy_dBy = 2.0 * b * ey_jp1 - float(lby) * ey_jm1

                                        for v in range(0, laz + lbz + 2):
                                            ez = float(Ez[laz, lbz, v])

                                            ez_ip1 = float(Ez[laz + 1, lbz, v]) if (laz + 1) <= la1 else 0.0
                                            ez_im1 = float(Ez[laz - 1, lbz, v]) if laz else 0.0
                                            ez_jp1 = float(Ez[laz, lbz + 1, v]) if (lbz + 1) <= lb1 else 0.0
                                            ez_jm1 = float(Ez[laz, lbz - 1, v]) if lbz else 0.0

                                            dEz_dAz = 2.0 * a * ez_ip1 - float(laz) * ez_im1
                                            dEz_dBz = 2.0 * b * ez_jp1 - float(lbz) * ez_jm1

                                            r = float(R0[t, u, v])

                                            sA_x += dEx_dAx * ey * ez * r
                                            sA_y += ex * dEy_dAy * ez * r
                                            sA_z += ex * ey * dEz_dAz * r

                                            sB_x += dEx_dBx * ey * ez * r
                                            sB_y += ex * dEy_dBy * ez * r
                                            sB_z += ex * ey * dEz_dBz * r

                                            if tileC is not None:
                                                if t + 1 < int(R0.shape[0]):
                                                    sC_x += ex_p * ey * ez * float(R0[t + 1, u, v])
                                                if u + 1 < int(R0.shape[1]):
                                                    sC_y += ex_p * ey * ez * float(R0[t, u + 1, v])
                                                if v + 1 < int(R0.shape[2]):
                                                    sC_z += ex_p * ey * ez * float(R0[t, u, v + 1])

                                scale = c * pref
                                tileA[0, i, j] += scale * (-Z) * sA_x
                                tileA[1, i, j] += scale * (-Z) * sA_y
                                tileA[2, i, j] += scale * (-Z) * sA_z

                                tileB[0, i, j] += scale * (-Z) * sB_x
                                tileB[1, i, j] += scale * (-Z) * sB_y
                                tileB[2, i, j] += scale * (-Z) * sB_z

                                if tileC is not None:
                                    tileC[ic, 0, i, j] += scale * (+Z) * sC_x
                                    tileC[ic, 1, i, j] += scale * (+Z) * sC_y
                                    tileC[ic, 2, i, j] += scale * (+Z) * sC_z

            tileA_sph = _sph_block_from_cart_block(tileA, la=la, lb=lb)
            tileB_sph = _sph_block_from_cart_block(tileB, la=la, lb=lb)

            out[atomA, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileA_sph
            out[atomB, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileB_sph
            if tileC is not None:
                tileC_sph = _sph_block_from_cart_block(tileC, la=la, lb=lb)
                out[:, :, aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph] += tileC_sph
            if shA != shB:
                out[atomA, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileA_sph.transpose(0, 2, 1)
                out[atomB, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileB_sph.transpose(0, 2, 1)
                if tileC is not None:
                    out[:, :, aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph] += tileC_sph.transpose(0, 1, 3, 2)

    return np.asarray(out, dtype=np.float64)


def contract_dS_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    M_sph: Any,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
) -> np.ndarray:
    """Contract the *symmetric* nuclear derivative dS/dR with a spherical AO matrix.

    Returns
    -------
    np.ndarray
        Array of shape (natm, 3).
    """
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()

    if shell_ao_start_sph is None:
        shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(basis_cart)
    else:
        shell_ao_start_sph = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()
        nao_sph = int(shell_ao_start_sph.max(initial=0) + 1) if int(shell_ao_start_sph.size) else 0
        # Correct nao_sph needs shell_l; compute properly when user passes starts.
        shell_l = np.asarray(basis_cart.shell_l, dtype=np.int32).ravel()
        if int(shell_l.size):
            nao_sph = int(max(int(shell_ao_start_sph[i]) + nsph(int(shell_l[i])) for i in range(int(shell_l.size))))

    M_sph_np = _validate_sph_matrix(M_sph, nao_sph=int(nao_sph))

    out = np.zeros((natm, 3), dtype=np.float64)
    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        aoA_cart = int(basis_cart.shell_ao_start[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            aoB_cart = int(basis_cart.shell_ao_start[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            # Build the needed cart block(s) from M_sph.
            M_sph_ab = M_sph_np[aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph]
            M_cart_ab = _cart_block_from_sph_block(M_sph_ab, la=la, lb=lb)
            if tuple(map(int, M_cart_ab.shape)) != (nA_cart, nB_cart):  # pragma: no cover
                raise RuntimeError("internal error: M_cart_ab block shape mismatch")

            if shA != shB:
                M_sph_ba = M_sph_np[aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph]
                M_cart_ba = _cart_block_from_sph_block(M_sph_ba, la=lb, lb=la)  # (nB_cart, nA_cart)
            else:
                M_cart_ba = None

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            S_yz = Sy[lay, lby] * Sz[laz, lbz]
                            S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                            S_xy = Sx[lax, lbx] * Sy[lay, lby]

                            dAx = 2.0 * a * Sx[lax + 1, lbx]
                            if lax:
                                dAx -= float(lax) * Sx[lax - 1, lbx]
                            tileA[0, i, j] += c * dAx * S_yz

                            dAy = 2.0 * a * Sy[lay + 1, lby]
                            if lay:
                                dAy -= float(lay) * Sy[lay - 1, lby]
                            tileA[1, i, j] += c * dAy * S_xz

                            dAz = 2.0 * a * Sz[laz + 1, lbz]
                            if laz:
                                dAz -= float(laz) * Sz[laz - 1, lbz]
                            tileA[2, i, j] += c * dAz * S_xy

                            dBx = 2.0 * b * Sx[lax, lbx + 1]
                            if lbx:
                                dBx -= float(lbx) * Sx[lax, lbx - 1]
                            tileB[0, i, j] += c * dBx * S_yz

                            dBy = 2.0 * b * Sy[lay, lby + 1]
                            if lby:
                                dBy -= float(lby) * Sy[lay, lby - 1]
                            tileB[1, i, j] += c * dBy * S_xz

                            dBz = 2.0 * b * Sz[laz, lbz + 1]
                            if lbz:
                                dBz -= float(lbz) * Sz[laz, lbz - 1]
                            tileB[2, i, j] += c * dBz * S_xy

            out[atomA] += np.einsum("xij,ij->x", tileA, M_cart_ab, optimize=True)
            out[atomB] += np.einsum("xij,ij->x", tileB, M_cart_ab, optimize=True)

            if M_cart_ba is not None:
                out[atomA] += np.einsum("xji,ij->x", tileA, M_cart_ba, optimize=True)
                out[atomB] += np.einsum("xji,ij->x", tileB, M_cart_ba, optimize=True)

    return np.asarray(out, dtype=np.float64)


def _contract_dT_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    M_sph_np: np.ndarray,
    shell_atom: np.ndarray,
    shell_ao_start_sph: np.ndarray,
) -> np.ndarray:
    natm = int(atom_coords_bohr.shape[0])
    out = np.zeros((natm, 3), dtype=np.float64)

    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            M_sph_ab = M_sph_np[aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph]
            M_cart_ab = _cart_block_from_sph_block(M_sph_ab, la=la, lb=lb)
            if shA != shB:
                M_sph_ba = M_sph_np[aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph]
                M_cart_ba = _cart_block_from_sph_block(M_sph_ba, la=lb, lb=la)
            else:
                M_cart_ba = None

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            t_p = _kin_cart_component(
                                i_xyz=(lax + 1, lay, laz),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax - 1, lay, laz),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lax
                                else 0.0
                            )
                            tileA[0, i, j] += c * (2.0 * a * t_p - float(lax) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay + 1, laz),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay - 1, laz),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lay
                                else 0.0
                            )
                            tileA[1, i, j] += c * (2.0 * a * t_p - float(lay) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz + 1),
                                j_xyz=(lbx, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz - 1),
                                    j_xyz=(lbx, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if laz
                                else 0.0
                            )
                            tileA[2, i, j] += c * (2.0 * a * t_p - float(laz) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx + 1, lby, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx - 1, lby, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lbx
                                else 0.0
                            )
                            tileB[0, i, j] += c * (2.0 * b * t_p - float(lbx) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx, lby + 1, lbz),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx, lby - 1, lbz),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lby
                                else 0.0
                            )
                            tileB[1, i, j] += c * (2.0 * b * t_p - float(lby) * t_m)

                            t_p = _kin_cart_component(
                                i_xyz=(lax, lay, laz),
                                j_xyz=(lbx, lby, lbz + 1),
                                Sx=Sx,
                                Sy=Sy,
                                Sz=Sz,
                                b=b,
                            )
                            t_m = (
                                _kin_cart_component(
                                    i_xyz=(lax, lay, laz),
                                    j_xyz=(lbx, lby, lbz - 1),
                                    Sx=Sx,
                                    Sy=Sy,
                                    Sz=Sz,
                                    b=b,
                                )
                                if lbz
                                else 0.0
                            )
                            tileB[2, i, j] += c * (2.0 * b * t_p - float(lbz) * t_m)

            out[atomA] += np.einsum("xij,ij->x", tileA, M_cart_ab, optimize=True)
            out[atomB] += np.einsum("xij,ij->x", tileB, M_cart_ab, optimize=True)
            if M_cart_ba is not None:
                out[atomA] += np.einsum("xji,ij->x", tileA, M_cart_ba, optimize=True)
                out[atomB] += np.einsum("xji,ij->x", tileB, M_cart_ba, optimize=True)

    return np.asarray(out, dtype=np.float64)


def _contract_dV_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    M_sph_np: np.ndarray,
    shell_atom: np.ndarray,
    shell_ao_start_sph: np.ndarray,
    include_operator_deriv: bool = True,
) -> np.ndarray:
    natm = int(atom_coords_bohr.shape[0])
    out = np.zeros((natm, 3), dtype=np.float64)

    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            M_sph_ab = M_sph_np[aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph]
            M_cart_ab = _cart_block_from_sph_block(M_sph_ab, la=la, lb=lb)
            if shA != shB:
                M_sph_ba = M_sph_np[aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph]
                M_cart_ba = _cart_block_from_sph_block(M_sph_ba, la=lb, lb=la)
            else:
                M_cart_ba = None

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileC = np.zeros((natm, 3, nA_cart, nB_cart), dtype=np.float64) if bool(include_operator_deriv) else None

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])
                    p = a + b
                    inv_p = 1.0 / p
                    P = (a * cA + b * cB) * inv_p

                    la1 = la + 1
                    lb1 = lb + 1
                    Ex = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Ey = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Ez = _hermite_E_1d_table(la=la1, lb=lb1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    L = la + lb
                    nmax = L + 1
                    c = ca * cb
                    pref = (2.0 * pi) * inv_p

                    for ic in range(natm):
                        Z = float(atom_charges[ic])
                        if Z == 0.0:
                            continue
                        C = atom_coords_bohr[ic]
                        R = _build_R_coulomb(p=p, PC=P - C, nmax=nmax)
                        R0 = R[0]

                        for i, (lax, lay, laz) in enumerate(compA):
                            for j, (lbx, lby, lbz) in enumerate(compB):
                                sA_x = 0.0
                                sA_y = 0.0
                                sA_z = 0.0
                                sB_x = 0.0
                                sB_y = 0.0
                                sB_z = 0.0
                                sC_x = 0.0
                                sC_y = 0.0
                                sC_z = 0.0

                                for t in range(0, lax + lbx + 2):
                                    ex = float(Ex[lax, lbx, t])
                                    ex_p = float(Ex[lax, lbx, t])
                                    ex_ip1 = float(Ex[lax + 1, lbx, t]) if (lax + 1) <= la1 else 0.0
                                    ex_im1 = float(Ex[lax - 1, lbx, t]) if lax else 0.0
                                    ex_jp1 = float(Ex[lax, lbx + 1, t]) if (lbx + 1) <= lb1 else 0.0
                                    ex_jm1 = float(Ex[lax, lbx - 1, t]) if lbx else 0.0

                                    dEx_dAx = 2.0 * a * ex_ip1 - float(lax) * ex_im1
                                    dEx_dBx = 2.0 * b * ex_jp1 - float(lbx) * ex_jm1

                                    for u in range(0, lay + lby + 2):
                                        ey = float(Ey[lay, lby, u])

                                        ey_ip1 = float(Ey[lay + 1, lby, u]) if (lay + 1) <= la1 else 0.0
                                        ey_im1 = float(Ey[lay - 1, lby, u]) if lay else 0.0
                                        ey_jp1 = float(Ey[lay, lby + 1, u]) if (lby + 1) <= lb1 else 0.0
                                        ey_jm1 = float(Ey[lay, lby - 1, u]) if lby else 0.0

                                        dEy_dAy = 2.0 * a * ey_ip1 - float(lay) * ey_im1
                                        dEy_dBy = 2.0 * b * ey_jp1 - float(lby) * ey_jm1

                                        for v in range(0, laz + lbz + 2):
                                            ez = float(Ez[laz, lbz, v])

                                            ez_ip1 = float(Ez[laz + 1, lbz, v]) if (laz + 1) <= la1 else 0.0
                                            ez_im1 = float(Ez[laz - 1, lbz, v]) if laz else 0.0
                                            ez_jp1 = float(Ez[laz, lbz + 1, v]) if (lbz + 1) <= lb1 else 0.0
                                            ez_jm1 = float(Ez[laz, lbz - 1, v]) if lbz else 0.0

                                            dEz_dAz = 2.0 * a * ez_ip1 - float(laz) * ez_im1
                                            dEz_dBz = 2.0 * b * ez_jp1 - float(lbz) * ez_jm1

                                            r = float(R0[t, u, v])

                                            sA_x += dEx_dAx * ey * ez * r
                                            sA_y += ex * dEy_dAy * ez * r
                                            sA_z += ex * ey * dEz_dAz * r

                                            sB_x += dEx_dBx * ey * ez * r
                                            sB_y += ex * dEy_dBy * ez * r
                                            sB_z += ex * ey * dEz_dBz * r

                                            if tileC is not None:
                                                if t + 1 < int(R0.shape[0]):
                                                    sC_x += ex_p * ey * ez * float(R0[t + 1, u, v])
                                                if u + 1 < int(R0.shape[1]):
                                                    sC_y += ex_p * ey * ez * float(R0[t, u + 1, v])
                                                if v + 1 < int(R0.shape[2]):
                                                    sC_z += ex_p * ey * ez * float(R0[t, u, v + 1])

                                scale = c * pref
                                tileA[0, i, j] += scale * (-Z) * sA_x
                                tileA[1, i, j] += scale * (-Z) * sA_y
                                tileA[2, i, j] += scale * (-Z) * sA_z

                                tileB[0, i, j] += scale * (-Z) * sB_x
                                tileB[1, i, j] += scale * (-Z) * sB_y
                                tileB[2, i, j] += scale * (-Z) * sB_z

                                if tileC is not None:
                                    tileC[ic, 0, i, j] += scale * (+Z) * sC_x
                                    tileC[ic, 1, i, j] += scale * (+Z) * sC_y
                                    tileC[ic, 2, i, j] += scale * (+Z) * sC_z

            out[atomA] += np.einsum("xij,ij->x", tileA, M_cart_ab, optimize=True)
            out[atomB] += np.einsum("xij,ij->x", tileB, M_cart_ab, optimize=True)
            if tileC is not None:
                out += np.einsum("axij,ij->ax", tileC, M_cart_ab, optimize=True)

            if M_cart_ba is not None:
                out[atomA] += np.einsum("xji,ij->x", tileA, M_cart_ba, optimize=True)
                out[atomB] += np.einsum("xji,ij->x", tileB, M_cart_ba, optimize=True)
                if tileC is not None:
                    out += np.einsum("axji,ij->ax", tileC, M_cart_ba, optimize=True)

    return np.asarray(out, dtype=np.float64)


def contract_dhcore_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    M_sph: Any,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
    include_operator_deriv: bool = True,
) -> np.ndarray:
    """Contract d(hcore)/dR with a spherical AO matrix without forming cart-sized AO matrices."""
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()

    if shell_ao_start_sph is None:
        shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(basis_cart)
    else:
        shell_ao_start_sph = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()
        shell_l = np.asarray(basis_cart.shell_l, dtype=np.int32).ravel()
        if int(shell_l.size):
            nao_sph = int(max(int(shell_ao_start_sph[i]) + nsph(int(shell_l[i])) for i in range(int(shell_l.size))))
        else:
            nao_sph = 0

    M_sph_np = _validate_sph_matrix(M_sph, nao_sph=int(nao_sph))

    gT = _contract_dT_sph(
        basis_cart,
        atom_coords_bohr=atom_coords_bohr,
        M_sph_np=M_sph_np,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
    )
    gV = _contract_dV_sph(
        basis_cart,
        atom_coords_bohr=atom_coords_bohr,
        atom_charges=atom_charges,
        M_sph_np=M_sph_np,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
        include_operator_deriv=bool(include_operator_deriv),
    )
    return np.asarray(gT + gV, dtype=np.float64)


def contract_dS_ip_sph(
    basis_cart,
    *,
    atom_coords_bohr: np.ndarray,
    M_sph: np.ndarray,
    shell_atom: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
) -> np.ndarray:
    """Contract bra-side overlap derivative with a spherical-space matrix.

    Uses per-shell-pair Cartesian↔spherical block transforms to avoid
    materializing a full ``(nao_cart, nao_cart)`` back-transformed matrix.

    Returns
    -------
    np.ndarray
        Array of shape ``(natm, 3)``.
    """
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis_cart, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()

    if shell_ao_start_sph is None:
        shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(basis_cart)
    else:
        shell_ao_start_sph = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()
        shell_l = np.asarray(basis_cart.shell_l, dtype=np.int32).ravel()
        if int(shell_l.size):
            nao_sph = int(max(int(shell_ao_start_sph[i]) + nsph(int(shell_l[i])) for i in range(int(shell_l.size))))
        else:
            nao_sph = 0

    M_sph_np = _validate_sph_matrix(M_sph, nao_sph=int(nao_sph))

    out = np.zeros((natm, 3), dtype=np.float64)
    nshell = int(np.asarray(basis_cart.shell_l).shape[0])
    for shA in range(nshell):
        la = int(basis_cart.shell_l[shA])
        nA_cart = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_cart.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        aoA_sph = int(shell_ao_start_sph[shA])
        nA_sph = int(nsph(la))

        sA = int(basis_cart.shell_prim_start[shA])
        nprimA = int(basis_cart.shell_nprim[shA])
        expA = basis_cart.prim_exp[sA : sA + nprimA]
        coefA = basis_cart.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis_cart.shell_l[shB])
            nB_cart = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_cart.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            aoB_sph = int(shell_ao_start_sph[shB])
            nB_sph = int(nsph(lb))

            # Per-shell-pair block transforms: M_sph -> M_cart.
            M_sph_ab = M_sph_np[aoA_sph : aoA_sph + nA_sph, aoB_sph : aoB_sph + nB_sph]
            M_cart_ab = _cart_block_from_sph_block(M_sph_ab, la=la, lb=lb)

            if shA != shB:
                M_sph_ba = M_sph_np[aoB_sph : aoB_sph + nB_sph, aoA_sph : aoA_sph + nA_sph]
                M_cart_ba = _cart_block_from_sph_block(M_sph_ba, la=lb, lb=la)  # (nB_cart, nA_cart)
            else:
                M_cart_ba = None

            sB = int(basis_cart.shell_prim_start[shB])
            nprimB = int(basis_cart.shell_nprim[shB])
            expB = basis_cart.prim_exp[sB : sB + nprimB]
            coefB = basis_cart.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)
            tileB = np.zeros((3, nA_cart, nB_cart), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            S_yz = Sy[lay, lby] * Sz[laz, lbz]
                            S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                            S_xy = Sx[lax, lbx] * Sy[lay, lby]

                            dAx = 2.0 * a * Sx[lax + 1, lbx]
                            if lax:
                                dAx -= float(lax) * Sx[lax - 1, lbx]
                            tileA[0, i, j] += c * dAx * S_yz

                            dAy = 2.0 * a * Sy[lay + 1, lby]
                            if lay:
                                dAy -= float(lay) * Sy[lay - 1, lby]
                            tileA[1, i, j] += c * dAy * S_xz

                            dAz = 2.0 * a * Sz[laz + 1, lbz]
                            if laz:
                                dAz -= float(laz) * Sz[laz - 1, lbz]
                            tileA[2, i, j] += c * dAz * S_xy

                            dBx = 2.0 * b * Sx[lax, lbx + 1]
                            if lbx:
                                dBx -= float(lbx) * Sx[lax, lbx - 1]
                            tileB[0, i, j] += c * dBx * S_yz

                            dBy = 2.0 * b * Sy[lay, lby + 1]
                            if lby:
                                dBy -= float(lby) * Sy[lay, lby - 1]
                            tileB[1, i, j] += c * dBy * S_xz

                            dBz = 2.0 * b * Sz[laz, lbz + 1]
                            if lbz:
                                dBz -= float(lbz) * Sz[laz, lbz - 1]
                            tileB[2, i, j] += c * dBz * S_xy

            # Bra-only contraction: dS(A)/dR_A contracted with M_cart(A,B).
            out[atomA] += np.einsum("xij,ij->x", tileA, M_cart_ab, optimize=True)

            if M_cart_ba is not None:
                # Off-diagonal: bra-derivative for transposed block M_cart(B,A).
                out[atomB] += np.einsum("xij,ji->x", tileB, M_cart_ba, optimize=True)

    return np.asarray(out, dtype=np.float64)


__all__ = [
    "build_dS_sph",
    "build_dT_sph",
    "build_dV_sph",
    "contract_dS_ip_sph",
    "contract_dS_sph",
    "contract_dhcore_sph",
]
