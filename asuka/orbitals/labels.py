from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.cart import cart_comp_str, cartesian_components
from asuka.frontend.molecule import Molecule

_LSYM = "spdfghijklmn"


def _l_symbol(l: int) -> str:
    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    if l < len(_LSYM):
        return _LSYM[l]
    return f"l{l}"


@dataclass(frozen=True)
class AOInfo:
    ao: int
    shell: int
    atom: int
    element: str
    l: int
    lx: int
    ly: int
    lz: int
    label: str


def shell_to_atom_map(
    mol: Molecule,
    ao_basis: Any,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """Map each shell to an atom index by matching shell center coordinates.

    Parameters
    ----------
    mol
        ASUKA Molecule (coords in Bohr).
    ao_basis
        BasisCartSoA-like object with `shell_cxyz` (nShell,3).
    tol
        Maximum allowed distance (Bohr) between shell center and atom coordinate.
    """

    shell_cxyz = np.asarray(getattr(ao_basis, "shell_cxyz"), dtype=np.float64)
    if shell_cxyz.ndim != 2 or shell_cxyz.shape[1] != 3:
        raise ValueError("ao_basis.shell_cxyz must have shape (nShell, 3)")

    atom_xyz = np.asarray(mol.coords_bohr, dtype=np.float64).reshape((mol.natm, 3))
    if atom_xyz.shape[0] == 0:
        raise ValueError("mol has no atoms")

    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be > 0")

    out = np.empty((shell_cxyz.shape[0],), dtype=np.int32)
    for sh, c in enumerate(shell_cxyz):
        d = np.linalg.norm(atom_xyz - c[None, :], axis=1)
        ia = int(np.argmin(d))
        if float(d[ia]) > tol:
            raise ValueError(f"failed to map shell {sh} to an atom (min dist={float(d[ia])} Bohr)")
        out[int(sh)] = ia
    return out


def build_ao_info(
    mol: Molecule,
    ao_basis: Any,
    *,
    tol: float = 1e-8,
    atom_index_base: int = 1,
) -> list[AOInfo]:
    """Build per-AO metadata and human labels.

    Notes
    -----
    - Uses cartesian ordering consistent with ASUKA/PySCF cart=True.
    """

    sh_cxyz = np.asarray(getattr(ao_basis, "shell_cxyz"), dtype=np.float64)
    sh_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    sh_ao0 = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()
    if sh_l.shape[0] != sh_cxyz.shape[0] or sh_ao0.shape[0] != sh_cxyz.shape[0]:
        raise ValueError("ao_basis shell arrays have inconsistent lengths")

    sh2a = shell_to_atom_map(mol, ao_basis, tol=tol)

    infos: list[AOInfo] = []
    for sh in range(int(sh_cxyz.shape[0])):
        l = int(sh_l[sh])
        ao0 = int(sh_ao0[sh])
        ia = int(sh2a[sh])
        element = str(mol.atom_symbol(ia))

        lsym = _l_symbol(l)
        comps = cartesian_components(l)
        for ic, (lx, ly, lz) in enumerate(comps):
            ao = ao0 + int(ic)
            comp = cart_comp_str(lx, ly, lz)
            if l == 0:
                lab = f"{element}{ia + atom_index_base} {lsym}"
            else:
                lab = f"{element}{ia + atom_index_base} {lsym}({comp})"
            infos.append(
                AOInfo(
                    ao=int(ao),
                    shell=int(sh),
                    atom=int(ia),
                    element=element,
                    l=int(l),
                    lx=int(lx),
                    ly=int(ly),
                    lz=int(lz),
                    label=str(lab),
                )
            )

    aos = np.asarray([x.ao for x in infos], dtype=np.int64)
    if aos.size and (int(aos.min()) != 0 or int(aos.max()) != int(aos.size - 1)):
        raise ValueError("unexpected AO indexing: not contiguous 0..nao-1")
    return infos


__all__ = ["AOInfo", "build_ao_info", "shell_to_atom_map"]

