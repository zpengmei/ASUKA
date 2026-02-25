from __future__ import annotations

"""DF front-end helpers.

This module connects:
- a cuGUGA `Molecule` (geometry + basis specs)
- a basis library loader (BSE or explicit basis dicts)
- cuERI DF primitives
"""

from typing import Any

from .basis_bse import load_autoaux_shells, load_basis_shells
from .basis_packer import pack_cart_basis, parse_pyscf_basis_dict
from .molecule import Molecule
from asuka.integrals.cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases


def _unique_elements(mol: Molecule) -> list[str]:
    return sorted(set(mol.elements))


def build_df_bases_cart(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    expand_contractions: bool = True,
) -> tuple[Any, Any, str]:
    """Build (ao_basis, aux_basis, auxbasis_name) as cuERI packed bases."""

    elements = _unique_elements(mol)
    basis_in = mol.basis if basis is None else basis

    if isinstance(basis_in, str):
        ao_shells = load_basis_shells(basis_in, elements=elements)
        basis_name = str(basis_in)
    elif isinstance(basis_in, dict):
        ao_shells = parse_pyscf_basis_dict(basis_in, elements=elements)
        basis_name = "<explicit>"
    else:
        raise TypeError("basis must be a string name or an explicit per-element basis dict")

    ao_basis = pack_cart_basis(list(mol.atoms_bohr), ao_shells, expand_contractions=bool(expand_contractions))

    # Auxiliary basis
    auxbasis_name = ""
    if isinstance(auxbasis, str) and str(auxbasis).strip().lower() in ("auto", "autoaux"):
        if not isinstance(basis_in, str):
            raise ValueError("auxbasis='autoaux' requires basis to be a string name")
        auxbasis_name, aux_shells = load_autoaux_shells(basis_in, elements=elements)
    elif isinstance(auxbasis, str):
        auxbasis_name = str(auxbasis)
        try:
            aux_shells = load_basis_shells(auxbasis_name, elements=elements)
        except Exception:
            # Basis Set Exchange does not necessarily expose fitted aux bases as
            # standalone names (e.g. "<basis>-jkfit"). Treat common JKFIT-like
            # names as aliases for the BSE autoaux basis.
            if isinstance(basis_in, str):
                base = str(auxbasis_name).strip()
                for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                    if base.lower().endswith(suf):
                        base = base[: -len(suf)]
                        break
                base = base or str(basis_in)
                auxbasis_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
            else:
                raise
    elif isinstance(auxbasis, dict):
        auxbasis_name = "<explicit>"
        aux_shells = parse_pyscf_basis_dict(auxbasis, elements=elements)
    else:
        raise TypeError("auxbasis must be 'autoaux', a string name, or an explicit per-element basis dict")

    aux_basis = pack_cart_basis(list(mol.atoms_bohr), aux_shells, expand_contractions=bool(expand_contractions))
    return ao_basis, aux_basis, auxbasis_name or "<unknown>"


def build_df_B_cueri(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    profile: dict | None = None,
):
    """Build whitened AO DF factors B[μ,ν,Q] via cuERI from a `Molecule`."""

    ao_basis, aux_basis, aux_name = build_df_bases_cart(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
    )
    B = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, config=config, profile=profile)
    return B, ao_basis, aux_basis, aux_name


__all__ = ["build_df_B_cueri", "build_df_bases_cart"]
