from __future__ import annotations

"""One-electron (AO) integral front-end helpers.

This module builds AO-basis one-electron integrals from a `frontend.Molecule`
and BSE / explicit basis specifications.
"""

from typing import Any

import numpy as np

from .basis_bse import load_basis_shells
from .basis_packer import pack_cart_basis, parse_pyscf_basis_dict
from .molecule import Molecule
from .periodic_table import atomic_number
from asuka.integrals.int1e_cart import Int1eDerivResult, Int1eResult, build_int1e_cart, build_int1e_cart_deriv


def build_ao_basis_cart(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
):
    """Build (ao_basis, basis_name) as a cuERI packed cart basis."""

    elements = sorted(set(mol.elements))
    basis_in = mol.basis if basis is None else basis

    if isinstance(basis_in, str):
        ao_shells = load_basis_shells(str(basis_in), elements=elements)
        basis_name = str(basis_in)
    elif isinstance(basis_in, dict):
        ao_shells = parse_pyscf_basis_dict(basis_in, elements=elements)
        basis_name = "<explicit>"
    else:
        raise TypeError("basis must be a string name or an explicit per-element basis dict")

    ao_basis = pack_cart_basis(list(mol.atoms_bohr), ao_shells, expand_contractions=bool(expand_contractions))
    return ao_basis, basis_name or "<unknown>"


def _atom_coords_charges_bohr(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((mol.natm, 3))
    charges = np.asarray([atomic_number(sym) for sym, _xyz in mol.atoms_bohr], dtype=np.float64)
    return coords, charges


def build_int1e_cart_from_mol(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
) -> tuple[Int1eResult, Any, str]:
    """Build (S,T,V) in AO basis from a `Molecule`."""

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    out = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)
    return out, ao_basis, basis_name


def build_int1e_cart_deriv_from_mol(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
) -> tuple[Int1eDerivResult, Any, str]:
    """Build (dS,dT,dV) in AO basis from a `Molecule`."""

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    out = build_int1e_cart_deriv(ao_basis, atom_coords_bohr=coords, atom_charges=charges)
    return out, ao_basis, basis_name


def build_int1e_from_mol(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
) -> tuple[Int1eResult, Any, str, np.ndarray | None]:
    """Build 1e integrals, optionally transforming to spherical AOs.

    Returns ``(int1e, ao_basis, basis_name, T_or_None)`` where ``T`` is the
    ``(nao_cart, nao_sph)`` cart-to-sph matrix (or ``None`` when ``mol.cart=True``).
    """
    int1e_cart, ao_basis, basis_name = build_int1e_cart_from_mol(
        mol, basis=basis, expand_contractions=bool(expand_contractions)
    )

    if bool(mol.cart):
        return int1e_cart, ao_basis, basis_name, None

    from asuka.integrals.cart2sph import (  # noqa: PLC0415
        build_cart2sph_matrix,
        compute_sph_layout_from_cart_basis,
        transform_1e_cart_to_sph,
    )

    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    nao_cart = int(int1e_cart.S.shape[0])

    T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, nao_sph)

    S_sph = transform_1e_cart_to_sph(int1e_cart.S, T)
    T_kin_sph = transform_1e_cart_to_sph(int1e_cart.T, T)
    V_sph = transform_1e_cart_to_sph(int1e_cart.V, T)

    int1e_sph = Int1eResult(S=S_sph, T=T_kin_sph, V=V_sph)
    return int1e_sph, ao_basis, basis_name, T


__all__ = [
    "build_ao_basis_cart",
    "build_int1e_cart_from_mol",
    "build_int1e_cart_deriv_from_mol",
    "build_int1e_from_mol",
]

