from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AMFIAtomicOccupation:
    """OpenMolcas-style atomic occupations for AMFI mean-field.

    This matches Molcas' default `getocc_ao` behavior at the level of
    - number of fully occupied subshells (per angular momentum l)
    - number of electrons in the partially occupied subshell (per l)
    """

    closed_shells: tuple[int, int, int, int]  # (s,p,d,f): number of full subshells
    open_electrons: tuple[int, int, int, int]  # (s,p,d,f): electrons in the next (open) subshell


LIGHT_SPEED_AU = 137.035999084

# OpenMolcas electron-configuration tables for AMFI (`getocc_ao.F90`), ported verbatim.
# For each Z=0..103, entries are (s,p,d,f).
_AMFI_OCC_CLOSED_OPENMOLCAS: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0),  # Z=0
    (0, 0, 0, 0),  # Z=1
    (1, 0, 0, 0),  # Z=2
    (1, 0, 0, 0),  # Z=3
    (2, 0, 0, 0),  # Z=4
    (2, 0, 0, 0),  # Z=5
    (2, 0, 0, 0),  # Z=6
    (2, 0, 0, 0),  # Z=7
    (2, 0, 0, 0),  # Z=8
    (2, 0, 0, 0),  # Z=9
    (2, 1, 0, 0),  # Z=10
    (2, 1, 0, 0),  # Z=11
    (3, 1, 0, 0),  # Z=12
    (3, 1, 0, 0),  # Z=13
    (3, 1, 0, 0),  # Z=14
    (3, 1, 0, 0),  # Z=15
    (3, 1, 0, 0),  # Z=16
    (3, 1, 0, 0),  # Z=17
    (3, 2, 0, 0),  # Z=18
    (3, 2, 0, 0),  # Z=19
    (4, 2, 0, 0),  # Z=20
    (4, 2, 0, 0),  # Z=21
    (4, 2, 0, 0),  # Z=22
    (4, 2, 0, 0),  # Z=23
    (4, 2, 0, 0),  # Z=24
    (4, 2, 0, 0),  # Z=25
    (4, 2, 0, 0),  # Z=26
    (4, 2, 0, 0),  # Z=27
    (4, 2, 0, 0),  # Z=28
    (3, 2, 1, 0),  # Z=29
    (4, 2, 1, 0),  # Z=30
    (4, 2, 1, 0),  # Z=31
    (4, 2, 1, 0),  # Z=32
    (4, 2, 1, 0),  # Z=33
    (4, 2, 1, 0),  # Z=34
    (4, 2, 1, 0),  # Z=35
    (4, 3, 1, 0),  # Z=36
    (4, 3, 1, 0),  # Z=37
    (5, 3, 1, 0),  # Z=38
    (5, 3, 1, 0),  # Z=39
    (5, 3, 1, 0),  # Z=40
    (5, 3, 1, 0),  # Z=41
    (5, 3, 1, 0),  # Z=42
    (5, 3, 1, 0),  # Z=43
    (5, 3, 1, 0),  # Z=44
    (5, 3, 1, 0),  # Z=45
    (5, 3, 1, 0),  # Z=46
    (4, 3, 2, 0),  # Z=47
    (5, 3, 2, 0),  # Z=48
    (5, 3, 2, 0),  # Z=49
    (5, 3, 2, 0),  # Z=50
    (5, 3, 2, 0),  # Z=51
    (5, 3, 2, 0),  # Z=52
    (5, 3, 2, 0),  # Z=53
    (5, 4, 2, 0),  # Z=54
    (5, 4, 2, 0),  # Z=55
    (6, 4, 2, 0),  # Z=56
    (6, 4, 2, 0),  # Z=57
    (6, 4, 2, 0),  # Z=58
    (6, 4, 2, 0),  # Z=59
    (6, 4, 2, 0),  # Z=60
    (6, 4, 2, 0),  # Z=61
    (6, 4, 2, 0),  # Z=62
    (6, 4, 2, 0),  # Z=63
    (6, 4, 2, 0),  # Z=64
    (6, 4, 2, 0),  # Z=65
    (6, 4, 2, 0),  # Z=66
    (6, 4, 2, 0),  # Z=67
    (6, 4, 2, 0),  # Z=68
    (6, 4, 2, 0),  # Z=69
    (6, 4, 2, 1),  # Z=70
    (6, 4, 2, 1),  # Z=71
    (6, 4, 2, 1),  # Z=72
    (6, 4, 2, 1),  # Z=73
    (6, 4, 2, 1),  # Z=74
    (6, 4, 2, 1),  # Z=75
    (6, 4, 2, 1),  # Z=76
    (6, 4, 2, 1),  # Z=77
    (5, 4, 2, 1),  # Z=78
    (5, 4, 3, 1),  # Z=79
    (6, 4, 3, 1),  # Z=80
    (6, 4, 3, 1),  # Z=81
    (6, 4, 3, 1),  # Z=82
    (6, 4, 3, 1),  # Z=83
    (6, 4, 3, 1),  # Z=84
    (6, 4, 3, 1),  # Z=85
    (6, 5, 3, 1),  # Z=86
    (6, 5, 3, 1),  # Z=87
    (7, 5, 3, 1),  # Z=88
    (7, 5, 3, 1),  # Z=89
    (7, 5, 3, 1),  # Z=90
    (7, 5, 3, 1),  # Z=91
    (7, 5, 3, 1),  # Z=92
    (7, 5, 3, 1),  # Z=93
    (7, 5, 3, 1),  # Z=94
    (7, 5, 3, 1),  # Z=95
    (7, 5, 3, 1),  # Z=96
    (7, 5, 3, 1),  # Z=97
    (7, 5, 3, 1),  # Z=98
    (7, 5, 3, 1),  # Z=99
    (7, 5, 3, 1),  # Z=100
    (7, 5, 3, 1),  # Z=101
    (7, 5, 3, 2),  # Z=102
    (7, 5, 3, 2),  # Z=103
)

_AMFI_OCC_OPEN_OPENMOLCAS: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0),  # Z=0
    (0, 0, 0, 0),  # Z=1
    (0, 0, 0, 0),  # Z=2
    (1, 0, 0, 0),  # Z=3
    (0, 0, 0, 0),  # Z=4
    (0, 1, 0, 0),  # Z=5
    (0, 2, 0, 0),  # Z=6
    (0, 3, 0, 0),  # Z=7
    (0, 4, 0, 0),  # Z=8
    (0, 5, 0, 0),  # Z=9
    (0, 0, 0, 0),  # Z=10
    (1, 0, 0, 0),  # Z=11
    (0, 0, 0, 0),  # Z=12
    (0, 1, 0, 0),  # Z=13
    (0, 2, 0, 0),  # Z=14
    (0, 3, 0, 0),  # Z=15
    (0, 4, 0, 0),  # Z=16
    (0, 5, 0, 0),  # Z=17
    (0, 0, 0, 0),  # Z=18
    (1, 0, 0, 0),  # Z=19
    (0, 0, 0, 0),  # Z=20
    (0, 0, 1, 0),  # Z=21
    (0, 0, 2, 0),  # Z=22
    (0, 0, 3, 0),  # Z=23
    (0, 0, 4, 0),  # Z=24
    (0, 0, 5, 0),  # Z=25
    (0, 0, 6, 0),  # Z=26
    (0, 0, 7, 0),  # Z=27
    (0, 0, 8, 0),  # Z=28
    (1, 0, 0, 0),  # Z=29
    (0, 0, 0, 0),  # Z=30
    (0, 1, 0, 0),  # Z=31
    (0, 2, 0, 0),  # Z=32
    (0, 3, 0, 0),  # Z=33
    (0, 4, 0, 0),  # Z=34
    (0, 5, 0, 0),  # Z=35
    (0, 0, 0, 0),  # Z=36
    (1, 0, 0, 0),  # Z=37
    (0, 0, 0, 0),  # Z=38
    (0, 0, 1, 0),  # Z=39
    (0, 0, 2, 0),  # Z=40
    (0, 0, 3, 0),  # Z=41
    (0, 0, 4, 0),  # Z=42
    (0, 0, 5, 0),  # Z=43
    (0, 0, 6, 0),  # Z=44
    (0, 0, 7, 0),  # Z=45
    (0, 0, 8, 0),  # Z=46
    (1, 0, 0, 0),  # Z=47
    (0, 0, 0, 0),  # Z=48
    (0, 1, 0, 0),  # Z=49
    (0, 2, 0, 0),  # Z=50
    (0, 3, 0, 0),  # Z=51
    (0, 4, 0, 0),  # Z=52
    (0, 5, 0, 0),  # Z=53
    (0, 0, 0, 0),  # Z=54
    (1, 0, 0, 0),  # Z=55
    (0, 0, 0, 0),  # Z=56
    (0, 0, 1, 0),  # Z=57
    (0, 0, 0, 2),  # Z=58
    (0, 0, 0, 3),  # Z=59
    (0, 0, 0, 4),  # Z=60
    (0, 0, 0, 5),  # Z=61
    (0, 0, 0, 6),  # Z=62
    (0, 0, 0, 7),  # Z=63
    (0, 0, 0, 8),  # Z=64
    (0, 0, 0, 9),  # Z=65
    (0, 0, 0, 10),  # Z=66
    (0, 0, 0, 11),  # Z=67
    (0, 0, 0, 12),  # Z=68
    (0, 0, 0, 13),  # Z=69
    (0, 0, 0, 0),  # Z=70
    (0, 0, 1, 0),  # Z=71
    (0, 0, 2, 0),  # Z=72
    (0, 0, 3, 0),  # Z=73
    (0, 0, 4, 0),  # Z=74
    (0, 0, 5, 0),  # Z=75
    (0, 0, 6, 0),  # Z=76
    (0, 0, 7, 0),  # Z=77
    (1, 0, 9, 0),  # Z=78
    (1, 0, 0, 0),  # Z=79
    (0, 0, 0, 0),  # Z=80
    (0, 1, 0, 0),  # Z=81
    (0, 2, 0, 0),  # Z=82
    (0, 3, 0, 0),  # Z=83
    (0, 4, 0, 0),  # Z=84
    (0, 5, 0, 0),  # Z=85
    (0, 0, 0, 0),  # Z=86
    (1, 0, 0, 0),  # Z=87
    (0, 0, 0, 0),  # Z=88
    (0, 0, 1, 0),  # Z=89
    (0, 0, 2, 0),  # Z=90
    (0, 0, 1, 2),  # Z=91
    (0, 0, 1, 3),  # Z=92
    (0, 0, 1, 4),  # Z=93
    (0, 0, 0, 6),  # Z=94
    (0, 0, 0, 7),  # Z=95
    (0, 0, 0, 8),  # Z=96
    (0, 0, 0, 9),  # Z=97
    (0, 0, 0, 10),  # Z=98
    (0, 0, 0, 11),  # Z=99
    (0, 0, 0, 12),  # Z=100
    (0, 0, 0, 13),  # Z=101
    (0, 0, 0, 0),  # Z=102
    (0, 0, 1, 0),  # Z=103
)


def amfi_atomic_occupation_openmolcas(Z: int) -> AMFIAtomicOccupation:
    """Return OpenMolcas `getocc_ao` occupations for neutral atom Z (s,p,d,f)."""

    Z = int(Z)
    if Z < 0:
        raise ValueError("Z must be non-negative")
    if Z >= len(_AMFI_OCC_CLOSED_OPENMOLCAS):
        raise ValueError(f"Z out of range for OpenMolcas AMFI table: Z={Z}")
    return AMFIAtomicOccupation(
        closed_shells=tuple(map(int, _AMFI_OCC_CLOSED_OPENMOLCAS[Z])),
        open_electrons=tuple(map(int, _AMFI_OCC_OPEN_OPENMOLCAS[Z])),
    )


def _maybe_asnumpy(x):
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def build_amfi_atomic_density_cart(
    basis_atom: "BasisCartSoA",
    occupation: AMFIAtomicOccupation,
) -> np.ndarray:
    """Build OpenMolcas-style spherical-average AO density for AMFI mean-field (cart basis)."""

    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.sph import cart2sph_matrix  # noqa: PLC0415
    from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415

    closed = occupation.closed_shells
    open_e = occupation.open_electrons

    nao = int(nao_cart_from_basis(basis_atom))
    D = np.zeros((nao, nao), dtype=np.float64)

    contr_count: dict[int, int] = {}
    nshell = int(basis_atom.shell_l.shape[0])
    for sh in range(nshell):
        l = int(basis_atom.shell_l[sh])
        contr_count[l] = int(contr_count.get(l, 0)) + 1
        ridx = int(contr_count[l])  # 1-based radial index within this l block

        occ_m = 0.0
        if 0 <= l < 4:
            if ridx <= int(closed[l]):
                occ_m = 2.0
            elif ridx == int(closed[l]) + 1 and int(open_e[l]) > 0:
                occ_m = float(open_e[l]) / float(2 * l + 1)

        if occ_m == 0.0:
            continue

        ao0 = int(basis_atom.shell_ao_start[sh])
        n = int(ncart(l))
        if l <= 1:
            D_block = np.eye(n, dtype=np.float64)
        else:
            T = np.asarray(cart2sph_matrix(l), dtype=np.float64)
            D_block = T @ T.T
        D[ao0 : ao0 + n, ao0 : ao0 + n] = float(occ_m) * D_block

    return D


def slice_basis_by_atom(
    ao_basis: "BasisCartSoA",
    mol: Any,
    ia: int,
) -> tuple["BasisCartSoA", np.ndarray]:
    """Slice a packed cart basis to one atom; returns (basis_atom, ao_map[local]->global)."""

    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

    ia = int(ia)
    coords_bohr = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64)
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords_bohr)
    shell_keep = np.nonzero(shell_atom == ia)[0].astype(np.int32, copy=False)
    if int(shell_keep.size) == 0:
        empty = BasisCartSoA(
            shell_cxyz=np.zeros((0, 3), dtype=np.float64),
            shell_prim_start=np.zeros((0,), dtype=np.int32),
            shell_nprim=np.zeros((0,), dtype=np.int32),
            shell_l=np.zeros((0,), dtype=np.int32),
            shell_ao_start=np.zeros((0,), dtype=np.int32),
            prim_exp=np.zeros((0,), dtype=np.float64),
            prim_coef=np.zeros((0,), dtype=np.float64),
        )
        return empty, np.zeros((0,), dtype=np.int64)

    shell_cxyz_list: list[np.ndarray] = []
    shell_l_list: list[int] = []
    shell_nprim_list: list[int] = []
    shell_prim_start_list: list[int] = []
    shell_ao_start_list: list[int] = []
    prim_exp_list: list[float] = []
    prim_coef_list: list[float] = []
    src_bas_id_list: list[int] | None = [] if ao_basis.source_bas_id is not None else None
    src_ctr_id_list: list[int] | None = [] if ao_basis.source_ctr_id is not None else None

    ao_map: list[int] = []
    ao_cur = 0
    prim_cur = 0
    for sh in shell_keep.tolist():
        l = int(ao_basis.shell_l[int(sh)])
        nprim = int(ao_basis.shell_nprim[int(sh)])
        s0 = int(ao_basis.shell_prim_start[int(sh)])
        exp = np.asarray(ao_basis.prim_exp[s0 : s0 + nprim], dtype=np.float64)
        coef = np.asarray(ao_basis.prim_coef[s0 : s0 + nprim], dtype=np.float64)

        shell_cxyz_list.append(np.asarray(ao_basis.shell_cxyz[int(sh)], dtype=np.float64))
        shell_l_list.append(int(l))
        shell_nprim_list.append(int(nprim))
        shell_prim_start_list.append(int(prim_cur))
        shell_ao_start_list.append(int(ao_cur))
        prim_exp_list.extend(exp.tolist())
        prim_coef_list.extend(coef.tolist())
        if src_bas_id_list is not None and ao_basis.source_bas_id is not None:
            src_bas_id_list.append(int(np.asarray(ao_basis.source_bas_id, dtype=np.int64).ravel()[int(sh)]))
        if src_ctr_id_list is not None and ao_basis.source_ctr_id is not None:
            src_ctr_id_list.append(int(np.asarray(ao_basis.source_ctr_id, dtype=np.int64).ravel()[int(sh)]))

        ao0_g = int(ao_basis.shell_ao_start[int(sh)])
        n = int(ncart(l))
        ao_map.extend(list(range(ao0_g, ao0_g + n)))

        prim_cur += int(nprim)
        ao_cur += int(n)

    basis_atom = BasisCartSoA(
        shell_cxyz=np.asarray(shell_cxyz_list, dtype=np.float64, order="C"),
        shell_prim_start=np.asarray(shell_prim_start_list, dtype=np.int32, order="C"),
        shell_nprim=np.asarray(shell_nprim_list, dtype=np.int32, order="C"),
        shell_l=np.asarray(shell_l_list, dtype=np.int32, order="C"),
        shell_ao_start=np.asarray(shell_ao_start_list, dtype=np.int32, order="C"),
        prim_exp=np.asarray(prim_exp_list, dtype=np.float64, order="C"),
        prim_coef=np.asarray(prim_coef_list, dtype=np.float64, order="C"),
        source_bas_id=np.asarray(src_bas_id_list, dtype=np.int32, order="C") if src_bas_id_list is not None else None,
        source_ctr_id=np.asarray(src_ctr_id_list, dtype=np.int32, order="C") if src_ctr_id_list is not None else None,
    )

    return basis_atom, np.asarray(ao_map, dtype=np.int64)


@dataclass(frozen=True)
class _AMFIExtendedBasis:
    basis_ext: "BasisCartSoA"
    sh_orig_ext: np.ndarray  # (nsh_orig,), int32
    sh_minus_ext: np.ndarray  # (nsh_orig,), int32 (-1 if absent)
    sh_plus_ext: np.ndarray  # (nsh_orig,), int32


@lru_cache(maxsize=None)
def _deriv_maps_cart(l: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (minus_maps, plus_maps) for cart derivatives at angular momentum l.

    minus_maps has shape (3,ncart(l-1),ncart(l)) or None for l==0.
    plus_maps has shape (3,ncart(l+1),ncart(l)).
    """

    from asuka.cueri.cart import cart_index, cartesian_components, ncart  # noqa: PLC0415

    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")

    comps = cartesian_components(l)
    n_src = int(ncart(l))

    minus = None
    if l > 0:
        minus = np.zeros((3, int(ncart(l - 1)), n_src), dtype=np.float64)
    plus = np.zeros((3, int(ncart(l + 1)), n_src), dtype=np.float64)

    for src, (lx, ly, lz) in enumerate(comps):
        if l > 0 and lx > 0 and minus is not None:
            minus[0, cart_index(lx - 1, ly, lz), src] = float(lx)
        plus[0, cart_index(lx + 1, ly, lz), src] = 1.0

        if l > 0 and ly > 0 and minus is not None:
            minus[1, cart_index(lx, ly - 1, lz), src] = float(ly)
        plus[1, cart_index(lx, ly + 1, lz), src] = 1.0

        if l > 0 and lz > 0 and minus is not None:
            minus[2, cart_index(lx, ly, lz - 1), src] = float(lz)
        plus[2, cart_index(lx, ly, lz + 1), src] = 1.0

    return minus, plus


def _build_amfi_extended_basis(basis_atom: "BasisCartSoA") -> _AMFIExtendedBasis:
    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415

    nshell = int(basis_atom.shell_l.shape[0])
    sh_orig = np.empty((nshell,), dtype=np.int32)
    sh_minus = np.full((nshell,), -1, dtype=np.int32)
    sh_plus = np.empty((nshell,), dtype=np.int32)

    shell_cxyz_list: list[np.ndarray] = []
    shell_l_list: list[int] = []
    shell_nprim_list: list[int] = []
    shell_prim_start_list: list[int] = []
    shell_ao_start_list: list[int] = []
    prim_exp_list: list[float] = []
    prim_coef_list: list[float] = []

    ao_cur = 0
    prim_cur = 0
    for sh in range(nshell):
        l = int(basis_atom.shell_l[sh])
        nprim = int(basis_atom.shell_nprim[sh])
        s0 = int(basis_atom.shell_prim_start[sh])
        exp = np.asarray(basis_atom.prim_exp[s0 : s0 + nprim], dtype=np.float64)
        coef = np.asarray(basis_atom.prim_coef[s0 : s0 + nprim], dtype=np.float64)
        cxyz = np.asarray(basis_atom.shell_cxyz[sh], dtype=np.float64)

        def _append_shell(*, l_sh: int, coef_sh: np.ndarray) -> int:
            nonlocal ao_cur, prim_cur
            idx = len(shell_l_list)
            shell_cxyz_list.append(cxyz)
            shell_l_list.append(int(l_sh))
            shell_nprim_list.append(int(nprim))
            shell_prim_start_list.append(int(prim_cur))
            shell_ao_start_list.append(int(ao_cur))
            prim_exp_list.extend(exp.tolist())
            prim_coef_list.extend(np.asarray(coef_sh, dtype=np.float64).tolist())
            prim_cur += int(nprim)
            ao_cur += int(ncart(int(l_sh)))
            return int(idx)

        sh_orig[sh] = _append_shell(l_sh=l, coef_sh=coef)
        if l > 0:
            sh_minus[sh] = _append_shell(l_sh=l - 1, coef_sh=coef)
        sh_plus[sh] = _append_shell(l_sh=l + 1, coef_sh=(-2.0 * exp) * coef)

    basis_ext = BasisCartSoA(
        shell_cxyz=np.asarray(shell_cxyz_list, dtype=np.float64, order="C"),
        shell_prim_start=np.asarray(shell_prim_start_list, dtype=np.int32, order="C"),
        shell_nprim=np.asarray(shell_nprim_list, dtype=np.int32, order="C"),
        shell_l=np.asarray(shell_l_list, dtype=np.int32, order="C"),
        shell_ao_start=np.asarray(shell_ao_start_list, dtype=np.int32, order="C"),
        prim_exp=np.asarray(prim_exp_list, dtype=np.float64, order="C"),
        prim_coef=np.asarray(prim_coef_list, dtype=np.float64, order="C"),
    )

    return _AMFIExtendedBasis(
        basis_ext=basis_ext,
        sh_orig_ext=sh_orig,
        sh_minus_ext=sh_minus,
        sh_plus_ext=sh_plus,
    )


def _build_shell_pairs_ordered(basis: "BasisCartSoA") -> "ShellPairs":
    """Build ordered shell pairs for which spAB == A*nsh + B."""

    from asuka.cueri.shell_pairs import ShellPairs  # noqa: PLC0415

    nshell = int(basis.shell_l.shape[0])
    if nshell == 0:
        return ShellPairs(
            sp_A=np.zeros((0,), dtype=np.int32),
            sp_B=np.zeros((0,), dtype=np.int32),
            sp_npair=np.zeros((0,), dtype=np.int32),
            sp_pair_start=np.asarray([0], dtype=np.int32),
        )
    sp_A = np.repeat(np.arange(nshell, dtype=np.int32), nshell)
    sp_B = np.tile(np.arange(nshell, dtype=np.int32), nshell)
    nprim = np.asarray(basis.shell_nprim, dtype=np.int64).ravel()
    sp_npair64 = nprim[sp_A.astype(np.int64)] * nprim[sp_B.astype(np.int64)]
    if int(np.max(sp_npair64, initial=0)) > np.iinfo(np.int32).max:
        raise OverflowError("sp_npair exceeds int32 range")
    sp_npair = sp_npair64.astype(np.int32)
    sp_pair_start64 = np.concatenate([[0], np.cumsum(sp_npair64, dtype=np.int64)])
    if int(sp_pair_start64[-1]) > np.iinfo(np.int32).max:
        raise OverflowError("sp_pair_start exceeds int32 range")
    sp_pair_start = sp_pair_start64.astype(np.int32)
    return ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)


def _shell_ao_range(basis: "BasisCartSoA", sh: int) -> tuple[int, int]:
    from asuka.cueri.cart import ncart  # noqa: PLC0415

    sh = int(sh)
    ao0 = int(basis.shell_ao_start[sh])
    l = int(basis.shell_l[sh])
    n = int(ncart(l))
    return ao0, ao0 + n


def build_openmolcas_amfi_xyz_ao_asuka(
    scf_out: Any,
    *,
    scale: float = 1.0,
    include_mean_field: bool = True,
    atoms: Sequence[int] | None = None,
    max_l: int = 3,
    eri_backend: str = "cpu_sp",
    symmetrize_antitrip: bool = True,
) -> np.ndarray:
    """Build OpenMolcas-style one-center AMFI integrals (ANTITRIP) in ASUKA cart AO basis.
    """

    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415
    from asuka.frontend.periodic_table import atomic_number  # noqa: PLC0415
    from asuka.integrals.int1e_cart import build_V_cart, nao_cart_from_basis  # noqa: PLC0415

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise ValueError("scf_out.mol is required")
    ao_basis = getattr(scf_out, "ao_basis", None)
    if ao_basis is None:
        raise ValueError("scf_out.ao_basis is required")
    if not isinstance(ao_basis, BasisCartSoA):
        raise TypeError("scf_out.ao_basis must be a BasisCartSoA (cart AOs required)")

    max_l_i = int(max_l)
    lmax_in = int(np.max(np.asarray(ao_basis.shell_l, dtype=np.int32), initial=0)) if int(ao_basis.shell_l.size) else 0
    if lmax_in > max_l_i:
        raise NotImplementedError(f"AMFI builder supports max_l<={max_l_i}, got lmax={lmax_in}")

    natm = int(getattr(mol, "natm"))
    coords_bohr = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64)
    if coords_bohr.shape != (natm, 3):
        raise ValueError("mol.coords_bohr has unexpected shape")

    if atoms is None:
        atoms = list(range(natm))
    atoms = [int(i) for i in atoms]

    nao = int(nao_cart_from_basis(ao_basis))
    h_xyz = np.zeros((3, nao, nao), dtype=np.float64)

    eri_backend_s = str(eri_backend).strip().lower()
    if eri_backend_s not in {"cpu_sp"}:
        raise ValueError("eri_backend must be 'cpu_sp' for now")

    prefactor = float(scale) / (4.0 * float(LIGHT_SPEED_AU) * float(LIGHT_SPEED_AU))

    for ia in atoms:
        if ia < 0 or ia >= natm:
            raise ValueError(f"atom index out of range: {ia}")

        Z = int(atomic_number(str(getattr(mol, "atom_symbol")(ia))))
        if Z == 0:
            continue

        basis_atom, ao_map = slice_basis_by_atom(ao_basis, mol, ia)
        nao_a = int(ao_map.size)
        if nao_a == 0:
            continue

        ext = _build_amfi_extended_basis(basis_atom)
        basis_ext = ext.basis_ext

        # 1e (nuclear) piece: V for this nucleus only, then ∂×V×∂ via component maps.
        coord = np.asarray(coords_bohr[int(ia)], dtype=np.float64).reshape((1, 3))
        charges = np.asarray([float(Z)], dtype=np.float64)
        V_ext = build_V_cart(basis_ext, atom_coords_bohr=coord, atom_charges=charges)
        if V_ext.ndim != 2 or int(V_ext.shape[0]) != int(V_ext.shape[1]):
            raise ValueError("build_V_cart returned non-square matrix for extended basis")

        h1 = np.zeros((3, nao_a, nao_a), dtype=np.float64)
        nshell_a = int(basis_atom.shell_l.shape[0])
        for shA in range(nshell_a):
            lA = int(basis_atom.shell_l[shA])
            nA = int((lA + 1) * (lA + 2) // 2)
            aoA = int(basis_atom.shell_ao_start[shA])
            A_minus, A_plus = _deriv_maps_cart(lA)

            for shB in range(nshell_a):
                lB = int(basis_atom.shell_l[shB])
                nB = int((lB + 1) * (lB + 2) // 2)
                aoB = int(basis_atom.shell_ao_start[shB])
                B_minus, B_plus = _deriv_maps_cart(lB)

                # I[a,b] = <∂_a A | V | ∂_b B>
                I = np.zeros((3, 3, nA, nB), dtype=np.float64)
                for a in range(3):
                    A_terms: list[tuple[np.ndarray, int]] = []
                    if A_minus is not None and int(ext.sh_minus_ext[shA]) >= 0:
                        A_terms.append((np.asarray(A_minus[a], dtype=np.float64), int(ext.sh_minus_ext[shA])))
                    A_terms.append((np.asarray(A_plus[a], dtype=np.float64), int(ext.sh_plus_ext[shA])))

                    for b in range(3):
                        B_terms: list[tuple[np.ndarray, int]] = []
                        if B_minus is not None and int(ext.sh_minus_ext[shB]) >= 0:
                            B_terms.append((np.asarray(B_minus[b], dtype=np.float64), int(ext.sh_minus_ext[shB])))
                        B_terms.append((np.asarray(B_plus[b], dtype=np.float64), int(ext.sh_plus_ext[shB])))

                        out_ab = np.zeros((nA, nB), dtype=np.float64)
                        for mapA, shA_ext in A_terms:
                            a0, a1 = _shell_ao_range(basis_ext, shA_ext)
                            for mapB, shB_ext in B_terms:
                                b0, b1 = _shell_ao_range(basis_ext, shB_ext)
                                V_blk = np.asarray(V_ext[a0:a1, b0:b1], dtype=np.float64)
                                out_ab += mapA.T @ V_blk @ mapB
                        I[a, b] = out_ab

                h1_x = I[1, 2] - I[2, 1]
                h1_y = I[2, 0] - I[0, 2]
                h1_z = I[0, 1] - I[1, 0]
                h1[0, aoA : aoA + nA, aoB : aoB + nB] += h1_x
                h1[1, aoA : aoA + nA, aoB : aoB + nB] += h1_y
                h1[2, aoA : aoA + nA, aoB : aoB + nB] += h1_z

        h_mf = np.zeros((3, nao_a, nao_a), dtype=np.float64)
        if bool(include_mean_field) and Z not in (0, 1):
            occ = amfi_atomic_occupation_openmolcas(Z)
            D = build_amfi_atomic_density_cart(basis_atom, occupation=occ)
            if float(np.linalg.norm(D)) != 0.0:
                try:
                    from asuka.cueri import _eri_rys_cpu as _eri_cpu  # noqa: PLC0415
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        "CPU ERI extension is not built (required for internal AMFI mean-field). "
                        "Build it with: python -m asuka.cueri.build_cpu_ext build_ext --inplace"
                    ) from e

                eri_tile_sp = getattr(_eri_cpu, "eri_rys_tile_cart_sp_cy", None)
                if eri_tile_sp is None:  # pragma: no cover
                    raise RuntimeError("asuka.cueri._eri_rys_cpu missing eri_rys_tile_cart_sp_cy; rebuild the extension")

                from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu  # noqa: PLC0415

                sp = _build_shell_pairs_ordered(basis_ext)
                pair_tables = build_pair_tables_cpu(basis_ext, sp, threads=0)

                # C-ordered arrays for the Cython entry point.
                shell_cxyz = np.asarray(basis_ext.shell_cxyz, dtype=np.float64, order="C")
                shell_l = np.asarray(basis_ext.shell_l, dtype=np.int32, order="C")
                sp_A = np.asarray(sp.sp_A, dtype=np.int32, order="C")
                sp_B = np.asarray(sp.sp_B, dtype=np.int32, order="C")
                sp_pair_start = np.asarray(sp.sp_pair_start, dtype=np.int32, order="C")
                sp_npair = np.asarray(sp.sp_npair, dtype=np.int32, order="C")
                pair_eta = np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C")
                pair_Px = np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C")
                pair_Py = np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C")
                pair_Pz = np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C")
                pair_cK = np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C")

                nsh_ext = int(basis_ext.shell_l.shape[0])

                def _ncart_l(l: int) -> int:
                    return int((int(l) + 1) * (int(l) + 2) // 2)

                def _g2_p1vxp1(
                    shI: int,
                    shJ: int,
                    shK: int,
                    shL: int,
                    *,
                    nK: int,
                    nL: int,
                ) -> np.ndarray:
                    """Build g2(x,i,j,k,l) for local shells (I,J|K,L) in the original AO basis.

                    Derivatives act on shells I and J (electron 1); shells K and L are non-derivative (electron 2).
                    """

                    lI = int(basis_atom.shell_l[shI])
                    lJ = int(basis_atom.shell_l[shJ])
                    nI = _ncart_l(lI)
                    nJ = _ncart_l(lJ)

                    I_minus, I_plus = _deriv_maps_cart(lI)
                    J_minus, J_plus = _deriv_maps_cart(lJ)

                    I_var: list[tuple[int, np.ndarray]] = []
                    if I_minus is not None and int(ext.sh_minus_ext[shI]) >= 0:
                        I_var.append((int(ext.sh_minus_ext[shI]), np.asarray(I_minus, dtype=np.float64)))
                    I_var.append((int(ext.sh_plus_ext[shI]), np.asarray(I_plus, dtype=np.float64)))

                    J_var: list[tuple[int, np.ndarray]] = []
                    if J_minus is not None and int(ext.sh_minus_ext[shJ]) >= 0:
                        J_var.append((int(ext.sh_minus_ext[shJ]), np.asarray(J_minus, dtype=np.float64)))
                    J_var.append((int(ext.sh_plus_ext[shJ]), np.asarray(J_plus, dtype=np.float64)))

                    shK_ext = int(ext.sh_orig_ext[shK])
                    shL_ext = int(ext.sh_orig_ext[shL])
                    spKL = int(shK_ext * nsh_ext + shL_ext)

                    tiles: dict[tuple[int, int], np.ndarray] = {}
                    for shI_ext, _mapsI in I_var:
                        lI_ext = int(basis_ext.shell_l[shI_ext])
                        nI_ext = _ncart_l(lI_ext)
                        for shJ_ext, _mapsJ in J_var:
                            lJ_ext = int(basis_ext.shell_l[shJ_ext])
                            nJ_ext = _ncart_l(lJ_ext)
                            spIJ = int(shI_ext * nsh_ext + shJ_ext)
                            tile = eri_tile_sp(
                                shell_cxyz,
                                shell_l,
                                sp_A,
                                sp_B,
                                sp_pair_start,
                                sp_npair,
                                pair_eta,
                                pair_Px,
                                pair_Py,
                                pair_Pz,
                                pair_cK,
                                int(spIJ),
                                int(spKL),
                            )
                            tiles[(shI_ext, shJ_ext)] = np.asarray(tile, dtype=np.float64).reshape(
                                (nI_ext, nJ_ext, nK, nL)
                            )

                    def _G(dirI: int, dirJ: int) -> np.ndarray:
                        out = np.zeros((nI, nJ, nK, nL), dtype=np.float64)
                        for shI_ext, mapsI in I_var:
                            mapI = np.asarray(mapsI[dirI], dtype=np.float64)
                            for shJ_ext, mapsJ in J_var:
                                mapJ = np.asarray(mapsJ[dirJ], dtype=np.float64)
                                tile = tiles[(shI_ext, shJ_ext)]
                                out += np.einsum("Pi,PQkl,Qj->ijkl", mapI, tile, mapJ, optimize=True)
                        return out

                    G_yz = _G(1, 2)
                    G_zy = _G(2, 1)
                    G_zx = _G(2, 0)
                    G_xz = _G(0, 2)
                    G_xy = _G(0, 1)
                    G_yx = _G(1, 0)
                    return np.stack([G_yz - G_zy, G_zx - G_xz, G_xy - G_yx], axis=0)

                for shA in range(nshell_a):
                    lA = int(basis_atom.shell_l[shA])
                    nA = _ncart_l(lA)
                    aoA = int(basis_atom.shell_ao_start[shA])

                    for shB in range(nshell_a):
                        lB = int(basis_atom.shell_l[shB])
                        nB = _ncart_l(lB)
                        aoB = int(basis_atom.shell_ao_start[shB])

                        J = np.zeros((3, nA, nB), dtype=np.float64)
                        K1 = np.zeros((3, nA, nB), dtype=np.float64)
                        K2 = np.zeros((3, nA, nB), dtype=np.float64)

                        for shC in range(nshell_a):
                            lC = int(basis_atom.shell_l[shC])
                            nC = _ncart_l(lC)
                            aoC = int(basis_atom.shell_ao_start[shC])
                            D_C = np.asarray(D[aoC : aoC + nC, aoC : aoC + nC], dtype=np.float64)
                            if float(np.linalg.norm(D_C)) == 0.0:
                                continue

                            g2_abcc = _g2_p1vxp1(shA, shB, shC, shC, nK=nC, nL=nC)
                            J += np.einsum("xijkl,lk->xij", g2_abcc, D_C, optimize=True)

                            g2_acbc = _g2_p1vxp1(shA, shC, shB, shC, nK=nB, nL=nC)
                            K1 += np.einsum("xikjl,lk->xij", g2_acbc, D_C, optimize=True)

                            g2_bcac = _g2_p1vxp1(shB, shC, shA, shC, nK=nA, nL=nC)
                            K2 += np.einsum("xjkil,lk->xij", g2_bcac, D_C, optimize=True)

                        h_blk = 0.5 * (2.0 * J - 3.0 * K1 + 3.0 * K2)
                        h_mf[:, aoA : aoA + nA, aoB : aoB + nB] += h_blk

        h_loc = prefactor * (h1 + h_mf)
        # Embed into global AO space (one-center blocks only).
        idx = np.ix_(ao_map, ao_map)
        h_xyz[:, idx[0], idx[1]] += np.asarray(h_loc, dtype=np.float64)

    if bool(symmetrize_antitrip):
        h_xyz = 0.5 * (h_xyz - np.swapaxes(h_xyz, -1, -2))

    return np.asarray(h_xyz, dtype=np.float64)


def build_amfi_soc_integrals_from_scf_out(
    scf_out: Any,
    *,
    mo_coeff: np.ndarray,
    ncore: int | None = None,
    ncas: int | None = None,
    rme_scale: float = 4.0,
    phase: complex = 1j,
    scale: float = 1.0,
    include_mean_field: bool = True,
    atoms: Sequence[int] | None = None,
) -> "SOCIntegrals":
    """Build SOCIntegrals in the provided MO basis from ASUKA-native one-center AMFI."""

    from asuka.soc.grad import soc_ao_to_mo  # noqa: PLC0415
    from asuka.soc.si import SOCIntegrals, soc_xyz_to_spherical  # noqa: PLC0415

    h_xyz_antitrip = build_openmolcas_amfi_xyz_ao_asuka(
        scf_out,
        scale=float(scale),
        include_mean_field=bool(include_mean_field),
        atoms=atoms,
    )
    h_xyz_hso = openmolcas_amfi_antitrip_xyz_ao_to_hso_xyz_ao(
        h_xyz_antitrip,
        rme_scale=float(rme_scale),
        phase=complex(phase),
    )
    h_m_ao = soc_xyz_to_spherical(h_xyz_hso)
    C = np.asarray(_maybe_asnumpy(mo_coeff), dtype=np.float64)
    h_m_mo = soc_ao_to_mo(h_m_ao, C)

    if ncore is not None or ncas is not None:
        if ncore is None or ncas is None:
            raise ValueError("ncore and ncas must be provided together for active slicing")
        ncore_i = int(ncore)
        ncas_i = int(ncas)
        sl = slice(ncore_i, ncore_i + ncas_i)
        h_m_mo = np.asarray(h_m_mo[:, sl, sl], dtype=np.complex128)

    return SOCIntegrals(h_m=np.asarray(h_m_mo, dtype=np.complex128))


def openmolcas_amfi_antitrip_xyz_ao_to_hso_xyz_ao(
    h_xyz_ao: np.ndarray,
    *,
    rme_scale: float = 4.0,
    phase: complex = 1j,
) -> np.ndarray:
    """Convert OpenMolcas AMFI (ANTITRIP) integrals into the SOC Hamiltonian convention used by cuGUGA.

    OpenMolcas stores AMFI as **real antisymmetric** Cartesian components (x,y,z) of an `ANTITRIP`
    operator and constructs a Hermitian SOC Hamiltonian by multiplying by the imaginary unit in the
    SOEIG assembly (see `OpenMolcas/src/rassi/soeig.f`).

    cuGUGA's SOC contractions assume the provided AO integrals represent the *Hermitian* one-electron
    SOC operator in the AO basis. Therefore, the OpenMolcas `ANTITRIP` AMFI matrices must be mapped
    to an imaginary-antisymmetric (Hermitian) form via a global phase factor (default `phase=+1j`).

    In addition, cuGUGA's current triplet TRDM/RME normalization differs from OpenMolcas' WTDM
    convention by an overall factor of 4; applying `rme_scale=4.0` yields SOC-RASSI eigenvalue parity
    for the forward pass tests.
    """

    h = np.asarray(h_xyz_ao)
    if h.ndim != 3 or int(h.shape[0]) != 3 or int(h.shape[1]) != int(h.shape[2]):
        raise ValueError("h_xyz_ao must have shape (3,nao,nao)")

    return complex(phase) * float(rme_scale) * np.asarray(h, dtype=np.complex128)
