from __future__ import annotations

"""AO 1-electron integrals (Cartesian GTOs, libcint conventions).

Scope
-----
- Overlap S
- Kinetic T
- Nuclear attraction V (sum over nuclei)

All routines operate on cuERI's packed Cartesian basis (`BasisCartSoA`), which
stores unnormalized primitives exp(-a r^2) with coefficients that already
include the primitive normalization factors used by PySCF/libcint for
`cart=True`.
"""

from dataclasses import dataclass
from math import pi, sqrt

import functools
import numpy as np
import os

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.boys import boys_fm_list
from asuka.cueri.cart import cartesian_components, ncart


def nao_cart_from_basis(basis: BasisCartSoA) -> int:
    """Return total number of AOs for a packed cart basis."""

    n_shell = int(basis.shell_l.shape[0])
    if n_shell == 0:
        return 0
    ao_max = 0
    for sh in range(n_shell):
        l = int(basis.shell_l[sh])
        ao0 = int(basis.shell_ao_start[sh])
        ao_max = max(ao_max, ao0 + int(ncart(l)))
    return int(ao_max)


def shell_to_atom_map(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Map each basis shell to an atom index by matching centers."""

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm == 0:
        raise ValueError("atom_coords_bohr must be non-empty")

    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be > 0")
    tol2 = tol * tol

    nshell = int(basis.shell_cxyz.shape[0])
    shell_atom = np.empty((nshell,), dtype=np.int32)
    for sh in range(nshell):
        c = basis.shell_cxyz[sh]
        # find closest atom
        d2 = np.sum((atom_coords_bohr - c[None, :]) ** 2, axis=1)
        ia = int(np.argmin(d2))
        if float(d2[ia]) > tol2:
            raise ValueError(f"shell center does not match any atom within tol={tol:g}")
        shell_atom[sh] = ia
    return shell_atom


def _overlap_1d_table(*, la: int, lb: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    """Return 1D overlap integrals S[i,j] for i<=la, j<=lb."""

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


def _hermite_E_1d_table(*, la: int, lb: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    """Return Hermite coefficients E[i,j,t] for one axis.

    E includes the 1D gaussian product factor exp(-mu*(Ax-Bx)^2).
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

    tmax = la + lb
    E = np.zeros((la + 1, lb + 1, tmax + 1), dtype=np.float64)
    E[0, 0, 0] = np.exp(-mu * AB * AB)

    inv_2p = 0.5 * inv_p

    # Build i for j=0
    for i in range(0, la):
        prev = E[i, 0]
        cur = E[i + 1, 0]
        # cur[t] = PA*prev[t] + inv2p*prev[t-1] + (t+1)*prev[t+1]
        for t in range(0, i + 1 + 1):  # t <= (i+1)+0
            val = PA * prev[t]
            if t > 0:
                val += inv_2p * prev[t - 1]
            if t + 1 <= i:
                val += float(t + 1) * prev[t + 1]
            cur[t] = val

    # Build j for all i
    for i in range(0, la + 1):
        for j in range(0, lb):
            prev = E[i, j]
            cur = E[i, j + 1]
            # cur[t] = PB*prev[t] + inv2p*prev[t-1] + (t+1)*prev[t+1]
            tmax_ij = i + (j + 1)
            for t in range(0, tmax_ij + 1):
                val = PB * prev[t]
                if t > 0:
                    val += inv_2p * prev[t - 1]
                if t + 1 <= i + j:
                    val += float(t + 1) * prev[t + 1]
                cur[t] = val

    return E


def _build_R_coulomb(*, p: float, PC: np.ndarray, nmax: int) -> np.ndarray:
    """Build R[n,t,u,v] auxiliary integrals for nuclear attraction (MD/OS form).

    Definition:
      R_{tuv}^n = (-2p)^n * (∂/∂P_x)^t (∂/∂P_y)^u (∂/∂P_z)^v F_n(T)
    where T = p * |P - C|^2.

    This builder fills entries where n + t + u + v <= nmax.
    """

    PC = np.asarray(PC, dtype=np.float64).reshape((3,))
    nmax = int(nmax)
    if nmax < 0:
        raise ValueError("nmax must be >= 0")

    L = nmax
    R = np.zeros((nmax + 1, L + 1, L + 1, L + 1), dtype=np.float64)

    T = float(p * float(np.dot(PC, PC)))
    F = boys_fm_list(T, nmax)

    fac = -2.0 * float(p)
    pow_fac = 1.0
    for n in range(0, nmax + 1):
        R[n, 0, 0, 0] = pow_fac * float(F[n])
        pow_fac *= fac

    X, Y, Z = map(float, PC)
    for n in range(nmax - 1, -1, -1):
        max_m = nmax - n
        for t in range(0, max_m + 1):
            for u in range(0, max_m - t + 1):
                for v in range(0, max_m - t - u + 1):
                    if t == 0 and u == 0 and v == 0:
                        continue
                    if t > 0:
                        val = X * R[n + 1, t - 1, u, v]
                        if t >= 2:
                            val += float(t - 1) * R[n + 1, t - 2, u, v]
                        R[n, t, u, v] = val
                    elif u > 0:
                        val = Y * R[n + 1, t, u - 1, v]
                        if u >= 2:
                            val += float(u - 1) * R[n + 1, t, u - 2, v]
                        R[n, t, u, v] = val
                    else:
                        val = Z * R[n + 1, t, u, v - 1]
                        if v >= 2:
                            val += float(v - 1) * R[n + 1, t, u, v - 2]
                        R[n, t, u, v] = val

    return R


@dataclass(frozen=True)
class Int1eResult:
    S: np.ndarray  # (nao, nao)
    T: np.ndarray  # (nao, nao)
    V: np.ndarray  # (nao, nao)  (sum over nuclei with -Z/|r-C|)

    @property
    def hcore(self) -> np.ndarray:
        return self.T + self.V


@dataclass(frozen=True)
class Int1eDerivResult:
    """Nuclear derivatives of AO 1e integral matrices.

    Shapes
    ------
    - dS, dT, dV: (natm, 3, nao, nao)
    """

    dS: np.ndarray
    dT: np.ndarray
    dV: np.ndarray

    @property
    def dhcore(self) -> np.ndarray:
        return self.dT + self.dV


def _int1e_backend() -> str:
    """Return selected 1e integral backend: 'python' | 'numba' | 'cython'."""

    v = os.environ.get("ASUKA_INT1E_BACKEND", "auto").strip().lower()
    if v in ("", "auto"):
        # Prefer cython if built; else use numba if available; else fall back to python.
        try:
            from asuka.integrals import _int1e_cart_cy as _cy  # noqa: F401, PLC0415

            return "cython"
        except Exception:
            pass
        try:
            from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

            if bool(getattr(_nb, "HAS_NUMBA", False)):
                return "numba"
        except Exception:
            pass
        return "python"
    if v in ("cy", "cython"):
        return "cython"
    if v in ("py", "python"):
        return "python"
    if v in ("nb", "numba"):
        return "numba"
    raise ValueError("ASUKA_INT1E_BACKEND must be one of: auto|python|numba|cython")


@functools.lru_cache(maxsize=64)
def _shell_pairs_lower(nshell: int) -> tuple[np.ndarray, np.ndarray]:
    nshell = int(nshell)
    pairA = np.empty((nshell * (nshell + 1) // 2,), dtype=np.int32)
    pairB = np.empty_like(pairA)
    idx = 0
    for shA in range(nshell):
        for shB in range(shA + 1):
            pairA[idx] = shA
            pairB[idx] = shB
            idx += 1
    pairA.setflags(write=False)
    pairB.setflags(write=False)
    return pairA, pairB


@functools.lru_cache(maxsize=64)
def _comp_tables_cached(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

    lmax = int(lmax)
    comp_start, comp_lx, comp_ly, comp_lz = _nb.build_comp_tables(lmax)
    comp_start.setflags(write=False)
    comp_lx.setflags(write=False)
    comp_ly.setflags(write=False)
    comp_lz.setflags(write=False)
    return comp_start, comp_lx, comp_ly, comp_lz


def build_S_cart(basis: BasisCartSoA) -> np.ndarray:
    """Build AO overlap S in cart basis (float64, shape (nao,nao))."""

    backend = _int1e_backend()

    if backend == "cython":
        from asuka.integrals import _int1e_cart_cy as _cy  # noqa: PLC0415

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _cy.build_S_cart_cy(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    if backend == "numba":
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):  # pragma: no cover
            raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _nb.build_S_cart_numba(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    nao = nao_cart_from_basis(basis)
    out = np.zeros((nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

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
            if shA != shB:
                out[aoB : aoB + nB, aoA : aoA + nA] = tile.T

    return out


def build_T_cart(basis: BasisCartSoA) -> np.ndarray:
    """Build AO kinetic energy T in cart basis (float64, shape (nao,nao))."""

    backend = _int1e_backend()

    if backend == "cython":
        from asuka.integrals import _int1e_cart_cy as _cy  # noqa: PLC0415

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _cy.build_T_cart_cy(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    if backend == "numba":
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):  # pragma: no cover
            raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _nb.build_T_cart_numba(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    nao = nao_cart_from_basis(basis)
    out = np.zeros((nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tile = np.zeros((nA, nB), dtype=np.float64)
            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    # Need overlap tables up to lb+2 for kinetic recurrence on ket.
                    Sx = _overlap_1d_table(la=la, lb=lb + 2, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la, lb=lb + 2, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la, lb=lb + 2, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            # 1D kinetic from overlap: T(i,j) = b*(2j+1)S(i,j) - 2b^2 S(i,j+2) - 0.5*j*(j-1)S(i,j-2)
                            Tx = b * (2.0 * float(lbx) + 1.0) * Sx[lax, lbx] - 2.0 * b * b * Sx[lax, lbx + 2]
                            if lbx >= 2:
                                Tx -= 0.5 * float(lbx * (lbx - 1)) * Sx[lax, lbx - 2]

                            Ty = b * (2.0 * float(lby) + 1.0) * Sy[lay, lby] - 2.0 * b * b * Sy[lay, lby + 2]
                            if lby >= 2:
                                Ty -= 0.5 * float(lby * (lby - 1)) * Sy[lay, lby - 2]

                            Tz = b * (2.0 * float(lbz) + 1.0) * Sz[laz, lbz] - 2.0 * b * b * Sz[laz, lbz + 2]
                            if lbz >= 2:
                                Tz -= 0.5 * float(lbz * (lbz - 1)) * Sz[laz, lbz - 2]

                            tile[i, j] += c * (
                                Tx * Sy[lay, lby] * Sz[laz, lbz]
                                + Sx[lax, lbx] * Ty * Sz[laz, lbz]
                                + Sx[lax, lbx] * Sy[lay, lby] * Tz
                            )

            out[aoA : aoA + nA, aoB : aoB + nB] = tile
            if shA != shB:
                out[aoB : aoB + nB, aoA : aoA + nA] = tile.T

    return out


def build_dS_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    shell_atom: np.ndarray | None = None,
) -> np.ndarray:
    """Build nuclear derivatives of AO overlap: dS[dAtom,xyz,μ,ν]."""

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    backend = _int1e_backend()
    if backend in ("numba", "cython"):
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):
            if backend == "numba":  # pragma: no cover
                raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")
        else:
            lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
            comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
            pairA, pairB = _shell_pairs_lower(int(basis.shell_l.shape[0]))
            nao = nao_cart_from_basis(basis)
            return _nb.build_dS_cart_numba(
                basis.shell_cxyz,
                basis.shell_prim_start,
                basis.shell_nprim,
                basis.shell_l,
                basis.shell_ao_start,
                basis.prim_exp,
                basis.prim_coef,
                shell_atom,
                int(natm),
                comp_start,
                comp_lx,
                comp_ly,
                comp_lz,
                pairA,
                pairB,
                int(nao),
            )

    nao = nao_cart_from_basis(basis)
    dS = np.zeros((natm, 3, nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA, nB), dtype=np.float64)
            tileB = np.zeros((3, nA, nB), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    # Need +/- 1 angular momentum shifts on either bra or ket.
                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            S_yz = Sy[lay, lby] * Sz[laz, lbz]
                            S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                            S_xy = Sx[lax, lbx] * Sy[lay, lby]

                            # d/dA
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

                            # d/dB
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

            dS[atomA, :, aoA : aoA + nA, aoB : aoB + nB] += tileA
            dS[atomB, :, aoA : aoA + nA, aoB : aoB + nB] += tileB
            if shA != shB:
                dS[atomA, :, aoB : aoB + nB, aoA : aoA + nA] += tileA.transpose(0, 2, 1)
                dS[atomB, :, aoB : aoB + nB, aoA : aoA + nA] += tileB.transpose(0, 2, 1)

    return dS


def _kin1d_from_overlap(*, S: np.ndarray, i: int, j: int, b: float) -> float:
    """1D kinetic integral from overlap table (ket derivative form)."""

    val = b * (2.0 * float(j) + 1.0) * float(S[i, j]) - 2.0 * b * b * float(S[i, j + 2])
    if j >= 2:
        val -= 0.5 * float(j * (j - 1)) * float(S[i, j - 2])
    return float(val)


def _kin_cart_component(
    *,
    i_xyz: tuple[int, int, int],
    j_xyz: tuple[int, int, int],
    Sx: np.ndarray,
    Sy: np.ndarray,
    Sz: np.ndarray,
    b: float,
) -> float:
    ix, iy, iz = map(int, i_xyz)
    jx, jy, jz = map(int, j_xyz)
    Tx = _kin1d_from_overlap(S=Sx, i=ix, j=jx, b=b)
    Ty = _kin1d_from_overlap(S=Sy, i=iy, j=jy, b=b)
    Tz = _kin1d_from_overlap(S=Sz, i=iz, j=jz, b=b)
    return float(
        Tx * float(Sy[iy, jy]) * float(Sz[iz, jz])
        + float(Sx[ix, jx]) * Ty * float(Sz[iz, jz])
        + float(Sx[ix, jx]) * float(Sy[iy, jy]) * Tz
    )


def build_dT_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    shell_atom: np.ndarray | None = None,
) -> np.ndarray:
    """Build nuclear derivatives of AO kinetic: dT[dAtom,xyz,μ,ν]."""

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    backend = _int1e_backend()
    if backend in ("numba", "cython"):
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):
            if backend == "numba":  # pragma: no cover
                raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")
        else:
            lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
            comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
            pairA, pairB = _shell_pairs_lower(int(basis.shell_l.shape[0]))
            nao = nao_cart_from_basis(basis)
            return _nb.build_dT_cart_numba(
                basis.shell_cxyz,
                basis.shell_prim_start,
                basis.shell_nprim,
                basis.shell_l,
                basis.shell_ao_start,
                basis.prim_exp,
                basis.prim_coef,
                shell_atom,
                int(natm),
                comp_start,
                comp_lx,
                comp_ly,
                comp_lz,
                pairA,
                pairB,
                int(nao),
            )

    nao = nao_cart_from_basis(basis)
    dT = np.zeros((natm, 3, nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA, nB), dtype=np.float64)
            tileB = np.zeros((3, nA, nB), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    # Need:
                    # - bra +/- 1  -> la+1
                    # - ket +/- 1 and kinetic needs j+2 -> (lb+1)+2 = lb+3
                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 3, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            # d/dA (basis center)
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

                            # d/dB
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

            dT[atomA, :, aoA : aoA + nA, aoB : aoB + nB] += tileA
            dT[atomB, :, aoA : aoA + nA, aoB : aoB + nB] += tileB
            if shA != shB:
                dT[atomA, :, aoB : aoB + nB, aoA : aoA + nA] += tileA.transpose(0, 2, 1)
                dT[atomB, :, aoB : aoB + nB, aoA : aoA + nA] += tileB.transpose(0, 2, 1)

    return dT


def build_V_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
) -> np.ndarray:
    """Build nuclear attraction AO matrix V (sum over nuclei, -Z/|r-C|)."""

    backend = _int1e_backend()

    if backend == "numba":
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):  # pragma: no cover
            raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")

        atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
        atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _nb.build_V_cart_numba(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            atom_coords_bohr,
            atom_charges,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    if atom_charges.shape != (int(atom_coords_bohr.shape[0]),):
        raise ValueError("atom_charges must have shape (natm,)")

    if backend == "cython":
        from asuka.integrals import _int1e_cart_cy as _cy  # noqa: PLC0415

        lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
        comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
        nshell = int(basis.shell_l.shape[0])
        pairA, pairB = _shell_pairs_lower(nshell)
        nao = nao_cart_from_basis(basis)
        return _cy.build_V_cart_cy(
            basis.shell_cxyz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.shell_l,
            basis.shell_ao_start,
            basis.prim_exp,
            basis.prim_coef,
            atom_coords_bohr,
            atom_charges,
            comp_start,
            comp_lx,
            comp_ly,
            comp_lz,
            pairA,
            pairB,
            int(nao),
        )

    natm = int(atom_coords_bohr.shape[0])
    nao = nao_cart_from_basis(basis)
    out = np.zeros((nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tile = np.zeros((nA, nB), dtype=np.float64)
            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])
                    p = a + b
                    inv_p = 1.0 / p
                    P = (a * cA + b * cB) * inv_p

                    # Hermite E tables (include 1D gaussian overlap exponential).
                    Ex = _hermite_E_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Ey = _hermite_E_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Ez = _hermite_E_1d_table(la=la, lb=lb, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    L = la + lb
                    c = ca * cb
                    pref = (2.0 * pi) * inv_p

                    for ic in range(natm):
                        Z = float(atom_charges[ic])
                        if Z == 0.0:
                            continue
                        C = atom_coords_bohr[ic]
                        R = _build_R_coulomb(p=p, PC=P - C, nmax=L)
                        R0 = R[0]

                        # Sum over t,u,v for each AO component pair.
                        for i, (lax, lay, laz) in enumerate(compA):
                            for j, (lbx, lby, lbz) in enumerate(compB):
                                s = 0.0
                                for t in range(0, lax + lbx + 1):
                                    ex = float(Ex[lax, lbx, t])
                                    if ex == 0.0:
                                        continue
                                    for u in range(0, lay + lby + 1):
                                        ey = float(Ey[lay, lby, u])
                                        if ey == 0.0:
                                            continue
                                        for v in range(0, laz + lbz + 1):
                                            ez = float(Ez[laz, lbz, v])
                                            if ez == 0.0:
                                                continue
                                            s += ex * ey * ez * float(R0[t, u, v])
                                tile[i, j] += c * (-Z) * pref * s

            out[aoA : aoA + nA, aoB : aoB + nB] = tile
            if shA != shB:
                out[aoB : aoB + nB, aoA : aoA + nA] = tile.T

    return out


def build_dV_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    shell_atom: np.ndarray | None = None,
    include_operator_deriv: bool = True,
) -> np.ndarray:
    """Build nuclear derivatives of AO nuclear attraction: dV[dAtom,xyz,μ,ν]."""

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    backend = _int1e_backend()
    if backend in ("numba", "cython"):
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if not bool(getattr(_nb, "HAS_NUMBA", False)):
            if backend == "numba":  # pragma: no cover
                raise RuntimeError("ASUKA_INT1E_BACKEND=numba but numba is unavailable")
        else:
            lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
            comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
            pairA, pairB = _shell_pairs_lower(int(basis.shell_l.shape[0]))
            nao = nao_cart_from_basis(basis)
            return _nb.build_dV_cart_numba(
                basis.shell_cxyz,
                basis.shell_prim_start,
                basis.shell_nprim,
                basis.shell_l,
                basis.shell_ao_start,
                basis.prim_exp,
                basis.prim_coef,
                atom_coords_bohr,
                atom_charges,
                shell_atom,
                int(natm),
                comp_start,
                comp_lx,
                comp_ly,
                comp_lz,
                pairA,
                pairB,
                int(nao),
                bool(include_operator_deriv),
            )

    nao = nao_cart_from_basis(basis)
    dV = np.zeros((natm, 3, nao, nao), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA, nB), dtype=np.float64)
            tileB = np.zeros((3, nA, nB), dtype=np.float64)
            tileC = np.zeros((natm, 3, nA, nB), dtype=np.float64) if bool(include_operator_deriv) else None

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])
                    p = a + b
                    inv_p = 1.0 / p
                    P = (a * cA + b * cB) * inv_p

                    # Need +/- 1 shifts on bra/ket and operator derivatives -> total order L+1.
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

                        # Sum over t,u,v for each AO component pair.
                        for i, (lax, lay, laz) in enumerate(compA):
                            for j, (lbx, lby, lbz) in enumerate(compB):
                                # Common unshifted factors
                                sA_x = 0.0
                                sA_y = 0.0
                                sA_z = 0.0
                                sB_x = 0.0
                                sB_y = 0.0
                                sB_z = 0.0
                                sC_x = 0.0
                                sC_y = 0.0
                                sC_z = 0.0

                                # Basis derivatives can increase the required Hermite order by 1 along the
                                # differentiated axis. Use +1 bounds and rely on the zero structure of
                                # E[i,j,t] for t > i+j to avoid extra work.
                                for t in range(0, lax + lbx + 2):
                                    ex = float(Ex[lax, lbx, t])
                                    if ex == 0.0:
                                        # Still need to consider x-derivative where Ex[lax+1,lbx,t] may be non-zero.
                                        pass
                                    # Operator derivative needs t+1
                                    ex_p = float(Ex[lax, lbx, t])  # same as ex, explicit for clarity
                                    ex_ip1 = float(Ex[lax + 1, lbx, t]) if (lax + 1) <= la1 else 0.0
                                    ex_im1 = float(Ex[lax - 1, lbx, t]) if lax else 0.0
                                    ex_jp1 = float(Ex[lax, lbx + 1, t]) if (lbx + 1) <= lb1 else 0.0
                                    ex_jm1 = float(Ex[lax, lbx - 1, t]) if lbx else 0.0

                                    dEx_dAx = 2.0 * a * ex_ip1 - float(lax) * ex_im1
                                    dEx_dBx = 2.0 * b * ex_jp1 - float(lbx) * ex_jm1

                                    for u in range(0, lay + lby + 2):
                                        ey = float(Ey[lay, lby, u])
                                        if ey == 0.0:
                                            pass

                                        ey_ip1 = float(Ey[lay + 1, lby, u]) if (lay + 1) <= la1 else 0.0
                                        ey_im1 = float(Ey[lay - 1, lby, u]) if lay else 0.0
                                        ey_jp1 = float(Ey[lay, lby + 1, u]) if (lby + 1) <= lb1 else 0.0
                                        ey_jm1 = float(Ey[lay, lby - 1, u]) if lby else 0.0

                                        dEy_dAy = 2.0 * a * ey_ip1 - float(lay) * ey_im1
                                        dEy_dBy = 2.0 * b * ey_jp1 - float(lby) * ey_jm1

                                        for v in range(0, laz + lbz + 2):
                                            ez = float(Ez[laz, lbz, v])
                                            if ez == 0.0:
                                                pass

                                            ez_ip1 = float(Ez[laz + 1, lbz, v]) if (laz + 1) <= la1 else 0.0
                                            ez_im1 = float(Ez[laz - 1, lbz, v]) if laz else 0.0
                                            ez_jp1 = float(Ez[laz, lbz + 1, v]) if (lbz + 1) <= lb1 else 0.0
                                            ez_jm1 = float(Ez[laz, lbz - 1, v]) if lbz else 0.0

                                            dEz_dAz = 2.0 * a * ez_ip1 - float(laz) * ez_im1
                                            dEz_dBz = 2.0 * b * ez_jp1 - float(lbz) * ez_jm1

                                            r = float(R0[t, u, v])

                                            # Basis derivatives (A/B centers): apply derivative to one axis at a time.
                                            sA_x += dEx_dAx * ey * ez * r
                                            sA_y += ex * dEy_dAy * ez * r
                                            sA_z += ex * ey * dEz_dAz * r

                                            sB_x += dEx_dBx * ey * ez * r
                                            sB_y += ex * dEy_dBy * ez * r
                                            sB_z += ex * ey * dEz_dBz * r

                                            if tileC is not None:
                                                # Operator derivatives (nuclear center C): d/dC = -d/dP => -R_{+1}
                                                if t + 1 < int(R0.shape[0]):
                                                    sC_x += ex_p * ey * ez * float(R0[t + 1, u, v])
                                                if u + 1 < int(R0.shape[1]):
                                                    sC_y += ex_p * ey * ez * float(R0[t, u + 1, v])
                                                if v + 1 < int(R0.shape[2]):
                                                    sC_z += ex_p * ey * ez * float(R0[t, u, v + 1])

                                # Apply prefactors for this nucleus.
                                # V includes -Z/|r-C|, so basis-deriv terms carry (-Z).
                                scale = c * pref
                                tileA[0, i, j] += scale * (-Z) * sA_x
                                tileA[1, i, j] += scale * (-Z) * sA_y
                                tileA[2, i, j] += scale * (-Z) * sA_z

                                tileB[0, i, j] += scale * (-Z) * sB_x
                                tileB[1, i, j] += scale * (-Z) * sB_y
                                tileB[2, i, j] += scale * (-Z) * sB_z

                                # Operator derivative: (-Z)*d(1/r)/dC = (+Z)*R-shifted.
                                if tileC is not None:
                                    # Operator derivative: (-Z)*d(1/r)/dC = (+Z)*R-shifted.
                                    tileC[ic, 0, i, j] += scale * (+Z) * sC_x
                                    tileC[ic, 1, i, j] += scale * (+Z) * sC_y
                                    tileC[ic, 2, i, j] += scale * (+Z) * sC_z

            # Accumulate to global derivative tensor
            dV[atomA, :, aoA : aoA + nA, aoB : aoB + nB] += tileA
            dV[atomB, :, aoA : aoA + nA, aoB : aoB + nB] += tileB
            if tileC is not None:
                dV[:, :, aoA : aoA + nA, aoB : aoB + nB] += tileC
            if shA != shB:
                dV[atomA, :, aoB : aoB + nB, aoA : aoA + nA] += tileA.transpose(0, 2, 1)
                dV[atomB, :, aoB : aoB + nB, aoA : aoA + nA] += tileB.transpose(0, 2, 1)
                if tileC is not None:
                    dV[:, :, aoB : aoB + nB, aoA : aoA + nA] += tileC.transpose(0, 1, 3, 2)

    return dV


def build_int1e_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
) -> Int1eResult:
    """Convenience builder for (S,T,V) in AO basis."""

    S = build_S_cart(basis)
    T = build_T_cart(basis)
    V = build_V_cart(basis, atom_coords_bohr=atom_coords_bohr, atom_charges=atom_charges)
    return Int1eResult(S=S, T=T, V=V)


def build_int1e_cart_deriv(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
) -> Int1eDerivResult:
    """Convenience builder for (dS,dT,dV) in AO basis."""

    shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    dS = build_dS_cart(basis, atom_coords_bohr=atom_coords_bohr, shell_atom=shell_atom)
    dT = build_dT_cart(basis, atom_coords_bohr=atom_coords_bohr, shell_atom=shell_atom)
    dV = build_dV_cart(basis, atom_coords_bohr=atom_coords_bohr, atom_charges=atom_charges, shell_atom=shell_atom)
    return Int1eDerivResult(dS=dS, dT=dT, dV=dV)


def contract_dS_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    M: np.ndarray,
    shell_atom: np.ndarray | None = None,
) -> np.ndarray:
    """Return gS[natm,3] = sum_{mu,nu} dS[A,x,mu,nu] * M[mu,nu] without building dS.

    Parameters
    ----------
    basis
        Packed Cartesian AO basis.
    atom_coords_bohr
        Nuclear coordinates, shape (natm,3).
    M
        AO matrix to contract with, shape (nao,nao).
    shell_atom
        Optional shell-to-atom map (use :func:`shell_to_atom_map`).
    """

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D array")
    nao = int(nao_cart_from_basis(basis))
    if tuple(M.shape) != (nao, nao):
        raise ValueError("M shape mismatch with basis nao")
    M = np.asarray(M, dtype=np.float64, order="C")

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    backend = _int1e_backend()
    if backend in ("numba", "cython"):
        from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

        if bool(getattr(_nb, "HAS_NUMBA", False)):
            lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
            comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
            pairA, pairB = _shell_pairs_lower(int(basis.shell_l.shape[0]))
            return _nb.contract_dS_cart_numba(
                basis.shell_cxyz,
                basis.shell_prim_start,
                basis.shell_nprim,
                basis.shell_l,
                basis.shell_ao_start,
                basis.prim_exp,
                basis.prim_coef,
                shell_atom,
                int(natm),
                comp_start,
                comp_lx,
                comp_ly,
                comp_lz,
                pairA,
                pairB,
                int(nao),
                M,
            )

    # Fallback: build full dS tensor and contract (memory-heavy).
    dS = build_dS_cart(basis, atom_coords_bohr=atom_coords_bohr, shell_atom=shell_atom)
    return np.einsum("axij,ij->ax", dS, M, optimize=True)


def contract_dS_ip_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    M: np.ndarray,
    shell_atom: np.ndarray | None = None,
) -> np.ndarray:
    """Contract the bra-side overlap derivative (PySCF ``get_ovlp`` convention) with a matrix.

    This returns an array ``g[natm,3]`` with

        g[A,x] = Σ_{μ in A} Σ_ν (∂S_{μν}/∂R_{A,x}) * M_{μν}

    where the derivative acts on the **bra** basis function only. In PySCF this
    object is obtained as ``-mol.intor('int1e_ipovlp', comp=3)`` and then mapped
    to atoms by slicing the **bra** AO index range for each atom.

    Notes
    -----
    - This is a one-sided derivative (bra/basis-center only), not the full
      symmetric nuclear derivative returned by :func:`contract_dS_cart`.
    - This routine is implemented in pure Python and is intended primarily for
      SA-CASSCF nonadiabatic couplings (NAC) baseline and parity tests.
    """

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D array")
    nao = int(nao_cart_from_basis(basis))
    if tuple(M.shape) != (nao, nao):
        raise ValueError("M shape mismatch with basis nao")
    M = np.asarray(M, dtype=np.float64, order="C")

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    out = np.zeros((natm, 3), dtype=np.float64)

    nshell = int(basis.shell_l.shape[0])
    for shA in range(nshell):
        la = int(basis.shell_l[shA])
        aoA = int(basis.shell_ao_start[shA])
        nA = int(ncart(la))
        compA = cartesian_components(la)
        cA = basis.shell_cxyz[shA]
        atomA = int(shell_atom[shA])

        sA = int(basis.shell_prim_start[shA])
        nprimA = int(basis.shell_nprim[shA])
        expA = basis.prim_exp[sA : sA + nprimA]
        coefA = basis.prim_coef[sA : sA + nprimA]

        for shB in range(shA + 1):
            lb = int(basis.shell_l[shB])
            aoB = int(basis.shell_ao_start[shB])
            nB = int(ncart(lb))
            compB = cartesian_components(lb)
            cB = basis.shell_cxyz[shB]
            atomB = int(shell_atom[shB])

            sB = int(basis.shell_prim_start[shB])
            nprimB = int(basis.shell_nprim[shB])
            expB = basis.prim_exp[sB : sB + nprimB]
            coefB = basis.prim_coef[sB : sB + nprimB]

            tileA = np.zeros((3, nA, nB), dtype=np.float64)
            tileB = np.zeros((3, nA, nB), dtype=np.float64)

            for ia in range(nprimA):
                a = float(expA[ia])
                ca = float(coefA[ia])
                for ib in range(nprimB):
                    b = float(expB[ib])
                    cb = float(coefB[ib])

                    # Need +/- 1 angular momentum shifts on either bra or ket.
                    Sx = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[0]), Bx=float(cB[0]))
                    Sy = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[1]), Bx=float(cB[1]))
                    Sz = _overlap_1d_table(la=la + 1, lb=lb + 1, a=a, b=b, Ax=float(cA[2]), Bx=float(cB[2]))

                    c = ca * cb
                    for i, (lax, lay, laz) in enumerate(compA):
                        for j, (lbx, lby, lbz) in enumerate(compB):
                            S_yz = Sy[lay, lby] * Sz[laz, lbz]
                            S_xz = Sx[lax, lbx] * Sz[laz, lbz]
                            S_xy = Sx[lax, lbx] * Sy[lay, lby]

                            # d/dA (bra-center derivative)
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

                            # d/dB (used for bra-derivative of the transposed block)
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

            Mab = M[aoA : aoA + nA, aoB : aoB + nB]
            out[atomA] += np.einsum("xij,ij->x", tileA, Mab, optimize=True)

            if shA != shB:
                Mba = M[aoB : aoB + nB, aoA : aoA + nA]
                out[atomB] += np.einsum("xij,ij->x", tileB.transpose(0, 2, 1), Mba, optimize=True)

    return np.asarray(out, dtype=np.float64)


def contract_dhcore_cart(
    basis: BasisCartSoA,
    *,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    M: np.ndarray,
    shell_atom: np.ndarray | None = None,
    include_operator_deriv: bool = True,
) -> np.ndarray:
    """Return gh1[natm,3] = sum_{mu,nu} d(hcore)[A,x,mu,nu] * M[mu,nu] without building dhcore.

    Notes
    -----
    Here ``hcore = T + V``. The nuclear-attraction derivative includes both:
      - basis-center derivatives (atoms owning the shells)
      - operator-center derivatives (moving each nucleus in the Coulomb operator)
    """

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D array")
    nao = int(nao_cart_from_basis(basis))
    if tuple(M.shape) != (nao, nao):
        raise ValueError("M shape mismatch with basis nao")
    M = np.asarray(M, dtype=np.float64, order="C")

    if shell_atom is None:
        shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords_bohr)
    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    if shell_atom.shape != (int(basis.shell_l.shape[0]),):
        raise ValueError("shell_atom must have shape (nShell,)")

    from asuka.integrals import _int1e_cart_numba as _nb  # noqa: PLC0415

    lmax = int(np.max(basis.shell_l)) if int(basis.shell_l.size) else 0
    comp_start, comp_lx, comp_ly, comp_lz = _comp_tables_cached(lmax)
    pairA, pairB = _shell_pairs_lower(int(basis.shell_l.shape[0]))

    gT = _nb.contract_dT_cart_numba(
        basis.shell_cxyz,
        basis.shell_prim_start,
        basis.shell_nprim,
        basis.shell_l,
        basis.shell_ao_start,
        basis.prim_exp,
        basis.prim_coef,
        shell_atom,
        int(natm),
        comp_start,
        comp_lx,
        comp_ly,
        comp_lz,
        pairA,
        pairB,
        int(nao),
        M,
    )
    gV = _nb.contract_dV_cart_numba(
        basis.shell_cxyz,
        basis.shell_prim_start,
        basis.shell_nprim,
        basis.shell_l,
        basis.shell_ao_start,
        basis.prim_exp,
        basis.prim_coef,
        atom_coords_bohr,
        atom_charges,
        shell_atom,
        int(natm),
        comp_start,
        comp_lx,
        comp_ly,
        comp_lz,
        pairA,
        pairB,
        int(nao),
        bool(include_operator_deriv),
        M,
    )
    return np.asarray(gT + gV, dtype=np.float64)


__all__ = [
    "Int1eDerivResult",
    "Int1eResult",
    "build_S_cart",
    "build_T_cart",
    "build_V_cart",
    "build_dS_cart",
    "build_dT_cart",
    "build_dV_cart",
    "build_int1e_cart",
    "build_int1e_cart_deriv",
    "contract_dS_cart",
    "contract_dS_ip_cart",
    "contract_dhcore_cart",
    "nao_cart_from_basis",
    "shell_to_atom_map",
]
