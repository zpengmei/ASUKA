from __future__ import annotations

"""Cross-geometry integrals for orbital tracking across geometry changes.

This module computes AO overlap matrices between two different molecular geometries,
which is essential for tracking orbital identity during molecular dynamics, geometry
optimization, and coordinate scans.
"""

from math import pi, sqrt

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.cart import cartesian_components, ncart
from asuka.integrals.int1e_cart import nao_cart_from_basis


def _overlap_1d_table(*, la: int, lb: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    """Return 1D overlap integrals S[i,j] for i<=la, j<=lb.

    This is copied from int1e_cart.py for cross-geometry overlap computation.
    """

    la = int(la)
    lb = int(lb)
    if la < 0 or lb < 0:
        raise ValueError("la/lb must be >= 0")

    p = a + b
    inv_p = 1.0 / p
    mu = a * b * inv_p
    Px = (a * Ax + b * Bx) * inv_p
    PA = Px - Ax
    PB = Px - Bx
    AB = Ax - Bx
    s00 = sqrt(pi * inv_p) * np.exp(-mu * AB * AB)

    out = np.zeros((la + 1, lb + 1), dtype=np.float64)
    out[0, 0] = float(s00)
    inv_2p = 0.5 * inv_p

    # j-recursion at i=0
    for j in range(0, lb):
        out[0, j + 1] = PB * out[0, j]
        if j > 0:
            out[0, j + 1] += float(j) * inv_2p * out[0, j - 1]

    # i-recursion
    for i in range(0, la):
        out[i + 1, 0] = PA * out[i, 0]
        if i > 0:
            out[i + 1, 0] += float(i) * inv_2p * out[i - 1, 0]
        for j in range(0, lb):
            out[i + 1, j + 1] = PA * out[i, j + 1]
            if i > 0:
                out[i + 1, j + 1] += float(i) * inv_2p * out[i - 1, j + 1]
            out[i + 1, j + 1] += float(j + 1) * inv_2p * out[i, j]

    return out


def build_S_cross(
    basis_bra: BasisCartSoA,
    basis_ket: BasisCartSoA,
    *,
    T_bra: np.ndarray | None = None,
    T_ket: np.ndarray | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Compute cross-geometry overlap, optionally in spherical AO basis.

    If ``T_bra`` and/or ``T_ket`` are provided (cart-to-sph matrices), the
    result is ``T_bra^T @ S_cart_cross @ T_ket``.
    """
    S_cart = build_S_cross_cart(basis_bra, basis_ket, backend=backend)
    if T_bra is not None:
        S_cart = np.asarray(T_bra, dtype=np.float64).T @ S_cart
    if T_ket is not None:
        S_cart = S_cart @ np.asarray(T_ket, dtype=np.float64)
    return S_cart


def build_S_cross_cart(
    basis_bra: BasisCartSoA,
    basis_ket: BasisCartSoA,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Compute cross-geometry overlap S_μν for different basis sets.

    This function computes the AO overlap matrix between two different molecular
    geometries, which is needed for orbital tracking across geometry changes.

    Parameters
    ----------
    basis_bra : BasisCartSoA
        Basis on first geometry (bra side, nao_bra AOs)
    basis_ket : BasisCartSoA
        Basis on second geometry (ket side, nao_ket AOs)
    backend : str | None
        "python" for pure Python implementation (default).
        Future: "numba" or "cython" for accelerated versions.

    Returns
    -------
    S_cross : np.ndarray
        Cross-geometry overlap matrix (nao_bra, nao_ket)

    Notes
    -----
    - Unlike build_S_cart(), this computes the full rectangular matrix without
      assuming symmetry, since the two geometries are different.
    - The basis sets must have the same structure (same atoms, same basis set),
      only the nuclear coordinates differ.
    - This is called once per geometry step, so Python implementation is acceptable
      for now. Can be optimized with Numba/Cython if needed.

    Examples
    --------
    >>> from asuka.frontend import Molecule, build_ao_basis_cart
    >>> mol1 = Molecule.from_atoms(atoms=[("H", (0,0,0)), ("H", (0,0,1.4))], basis="sto-3g")
    >>> mol2 = Molecule.from_atoms(atoms=[("H", (0,0,0)), ("H", (0,0,1.5))], basis="sto-3g")
    >>> basis1 = build_ao_basis_cart(mol1)
    >>> basis2 = build_ao_basis_cart(mol2)
    >>> S_cross = build_S_cross_cart(basis1, basis2)
    """
    if backend is not None and backend != "python":
        raise NotImplementedError(f"backend={backend!r} not yet implemented for cross-geometry overlap")

    nao_bra = nao_cart_from_basis(basis_bra)
    nao_ket = nao_cart_from_basis(basis_ket)
    out = np.zeros((nao_bra, nao_ket), dtype=np.float64)

    nshell_bra = int(basis_bra.shell_l.shape[0])
    nshell_ket = int(basis_ket.shell_l.shape[0])

    # Loop over all shell pairs (no symmetry assumption)
    for shA in range(nshell_bra):
        la = int(basis_bra.shell_l[shA])
        aoA = int(basis_bra.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis_bra.shell_cxyz[shA]

        sA = int(basis_bra.shell_prim_start[shA])
        nprimA = int(basis_bra.shell_nprim[shA])
        expA = basis_bra.prim_exp[sA : sA + nprimA]
        coefA = basis_bra.prim_coef[sA : sA + nprimA]

        for shB in range(nshell_ket):
            lb = int(basis_ket.shell_l[shB])
            aoB = int(basis_ket.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis_ket.shell_cxyz[shB]

            sB = int(basis_ket.shell_prim_start[shB])
            nprimB = int(basis_ket.shell_nprim[shB])
            expB = basis_ket.prim_exp[sB : sB + nprimB]
            coefB = basis_ket.prim_coef[sB : sB + nprimB]

            tile = np.zeros((nA, nB), dtype=np.float64)
            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])
                    Sx = _overlap_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))
                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            tile[i, j] += c * Sx[lax, lbx] * Sy[lay, lby] * Sz[laz, lbz]

            out[aoA : aoA + nA, aoB : aoB + nB] = tile

    return out
