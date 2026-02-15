from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.integrals.gto_cart import ncart as _ncart
from asuka.integrals.gto_cart import primitive_norm_cart_like_pyscf as _primitive_norm_cart_like_pyscf

from .basis_cart import BasisCartSoA
from .sph import nsph


@dataclass(frozen=True)
class SphMapForCartBasis:
    """Spherical AO layout aligned to an expanded cartesian evaluation basis."""

    shell_ao_start_sph: np.ndarray  # int32, one entry per expanded shell
    nao_sph: int
    nao_cart: int


def _pack_cart_shells_from_mol_core(
    mol: Any, *, expand_contractions: bool = True
) -> tuple[BasisCartSoA, SphMapForCartBasis]:
    """Internal packer returning both cart basis and spherical AO offset map."""

    shell_cxyz: list[np.ndarray] = []
    shell_prim_start: list[int] = []
    shell_nprim: list[int] = []
    shell_l: list[int] = []
    shell_ao_start_cart: list[int] = []
    shell_ao_start_sph: list[int] = []
    prim_exp: list[float] = []
    prim_coef: list[float] = []

    ao_loc_cart = np.asarray(mol.ao_loc_nr(cart=True), dtype=np.int32)
    ao_loc_sph = np.asarray(mol.ao_loc_nr(cart=False), dtype=np.int32)

    for bas_id in range(int(mol.nbas)):
        l = int(mol.bas_angular(bas_id))
        nprim = int(mol.bas_nprim(bas_id))
        nctr = int(mol.bas_nctr(bas_id))

        exp = np.asarray(mol.bas_exp(bas_id), dtype=np.float64)
        ctr_coeff = np.asarray(mol.bas_ctr_coeff(bas_id), dtype=np.float64)  # (nprim, nctr)
        if exp.shape != (nprim,):
            raise RuntimeError("unexpected bas_exp shape from mol")
        if ctr_coeff.shape != (nprim, nctr):
            raise RuntimeError("unexpected bas_ctr_coeff shape from mol")

        norm = _primitive_norm_cart_like_pyscf(l, exp)
        if np.any(~np.isfinite(norm)):
            raise ValueError("non-finite primitive normalization")

        center = np.asarray(mol.bas_coord(bas_id), dtype=np.float64)
        if center.shape != (3,):
            raise RuntimeError("unexpected bas_coord shape from mol")

        ctr_iter = range(nctr) if bool(expand_contractions) else range(1)
        for ctr_id in ctr_iter:
            shell_cxyz.append(center)
            shell_prim_start.append(len(prim_exp))
            shell_nprim.append(nprim)
            shell_l.append(l)
            shell_ao_start_cart.append(int(ao_loc_cart[bas_id] + ctr_id * _ncart(l)))
            shell_ao_start_sph.append(int(ao_loc_sph[bas_id] + ctr_id * nsph(l)))

            col = ctr_coeff[:, ctr_id]
            prim_exp.extend(exp.tolist())
            prim_coef.extend((col * norm).tolist())

    basis = BasisCartSoA(
        shell_cxyz=np.asarray(shell_cxyz, dtype=np.float64),
        shell_prim_start=np.asarray(shell_prim_start, dtype=np.int32),
        shell_nprim=np.asarray(shell_nprim, dtype=np.int32),
        shell_l=np.asarray(shell_l, dtype=np.int32),
        shell_ao_start=np.asarray(shell_ao_start_cart, dtype=np.int32),
        prim_exp=np.asarray(prim_exp, dtype=np.float64),
        prim_coef=np.asarray(prim_coef, dtype=np.float64),
    )
    if not shell_l:
        nao_cart = 0
        nao_sph = 0
    else:
        nao_cart = int(max(a0 + _ncart(int(l)) for a0, l in zip(shell_ao_start_cart, shell_l, strict=False)))
        nao_sph = int(max(a0 + nsph(int(l)) for a0, l in zip(shell_ao_start_sph, shell_l, strict=False)))

    sph_map = SphMapForCartBasis(
        shell_ao_start_sph=np.asarray(shell_ao_start_sph, dtype=np.int32),
        nao_sph=nao_sph,
        nao_cart=nao_cart,
    )
    return basis, sph_map


def pack_cart_shells_from_mol(mol: Any, *, expand_contractions: bool = True) -> BasisCartSoA:
    """Pack a mol-like object into a cuERI `BasisCartSoA` (cartesian shells).

    The input is expected to provide the PySCF-compatible basis introspection
    methods used by the existing ASUKA bridge layer, e.g.:
    `nbas`, `ao_loc_nr(cart=True)`, `bas_angular`, `bas_nprim`, `bas_nctr`,
    `bas_exp`, `bas_ctr_coeff`, and `bas_coord`.
    """

    basis, _ = _pack_cart_shells_from_mol_core(mol, expand_contractions=bool(expand_contractions))
    return basis


def pack_cart_shells_from_mol_with_sph_map(
    mol: Any, *, expand_contractions: bool = True
) -> tuple[BasisCartSoA, SphMapForCartBasis]:
    """Pack cartesian basis and a spherical AO-offset map aligned to expanded shells."""

    return _pack_cart_shells_from_mol_core(mol, expand_contractions=bool(expand_contractions))


def get_cached_or_pack_cart_ao_basis(
    *,
    cache_owner: Any | None = None,
    mol: Any | None = None,
    cache_attr: str = "_asuka_ao_basis",
    expand_contractions: bool = True,
) -> BasisCartSoA | None:
    """Return a cached packed AO basis or build one from a mol-like object.

    Returns `None` if no basis is cached and packing fails.
    """

    if cache_owner is not None:
        cached = getattr(cache_owner, cache_attr, None)
        if isinstance(cached, BasisCartSoA):
            return cached

    mol_obj = mol
    if mol_obj is None and cache_owner is not None:
        mol_obj = getattr(cache_owner, "mol", None)
    if mol_obj is None:
        return None

    try:
        ao_basis = pack_cart_shells_from_mol(mol_obj, expand_contractions=bool(expand_contractions))
    except Exception:
        return None

    if cache_owner is not None:
        try:
            setattr(cache_owner, cache_attr, ao_basis)
        except Exception:
            # Some wrappers may disallow dynamic attributes.
            pass
    return ao_basis


__all__ = [
    "SphMapForCartBasis",
    "get_cached_or_pack_cart_ao_basis",
    "pack_cart_shells_from_mol",
    "pack_cart_shells_from_mol_with_sph_map",
]
