from __future__ import annotations

"""Utilities for subsetting packed cuERI basis objects.

Local-THC and other locality-based methods often need to evaluate basis
functions for only a subset of atoms/shells. The packed `BasisCartSoA`
structure is easy to subset by shell ids by repacking the primitive arrays.
"""

from typing import Iterable

import numpy as np

from .basis_cart import BasisCartSoA
from .cart import ncart


def subset_cart_basis_by_shells(basis: BasisCartSoA, shell_ids: Iterable[int]) -> BasisCartSoA:
    """Return a new `BasisCartSoA` containing only selected shells.

    Parameters
    ----------
    basis
        Source packed Cartesian basis.
    shell_ids
        Shell indices into `basis` to keep. The *order* of `shell_ids` is
        preserved in the output basis, which is useful for building custom AO
        orderings (e.g. primary vs secondary).
    """

    shell_ids_np = np.asarray(list(shell_ids), dtype=np.int32).ravel()
    nshell = int(np.asarray(basis.shell_l, dtype=np.int32).size)
    if int(shell_ids_np.size) == 0:
        return BasisCartSoA(
            shell_cxyz=np.zeros((0, 3), dtype=np.float64),
            shell_prim_start=np.zeros((0,), dtype=np.int32),
            shell_nprim=np.zeros((0,), dtype=np.int32),
            shell_l=np.zeros((0,), dtype=np.int32),
            shell_ao_start=np.zeros((0,), dtype=np.int32),
            prim_exp=np.zeros((0,), dtype=np.float64),
            prim_coef=np.zeros((0,), dtype=np.float64),
            source_bas_id=None if basis.source_bas_id is None else np.zeros((0,), dtype=basis.source_bas_id.dtype),
            source_ctr_id=None if basis.source_ctr_id is None else np.zeros((0,), dtype=basis.source_ctr_id.dtype),
        )

    if int(np.min(shell_ids_np)) < 0 or int(np.max(shell_ids_np)) >= nshell:
        raise IndexError("shell_ids out of range for basis")

    shell_cxyz = np.ascontiguousarray(np.asarray(basis.shell_cxyz, dtype=np.float64)[shell_ids_np])
    shell_l = np.ascontiguousarray(np.asarray(basis.shell_l, dtype=np.int32)[shell_ids_np])
    shell_nprim = np.ascontiguousarray(np.asarray(basis.shell_nprim, dtype=np.int32)[shell_ids_np])

    prim_exp_src = np.asarray(basis.prim_exp, dtype=np.float64)
    prim_coef_src = np.asarray(basis.prim_coef, dtype=np.float64)
    prim_exp_parts: list[np.ndarray] = []
    prim_coef_parts: list[np.ndarray] = []
    shell_prim_start = np.empty((int(shell_ids_np.size),), dtype=np.int32)

    cursor = 0
    shell_prim_start_list: list[int] = []
    for sid in shell_ids_np.tolist():
        s0 = int(np.asarray(basis.shell_prim_start, dtype=np.int32)[sid])
        nprim = int(np.asarray(basis.shell_nprim, dtype=np.int32)[sid])
        shell_prim_start_list.append(int(cursor))
        if nprim > 0:
            prim_exp_parts.append(prim_exp_src[s0 : s0 + nprim])
            prim_coef_parts.append(prim_coef_src[s0 : s0 + nprim])
            cursor += nprim
    shell_prim_start[:] = np.asarray(shell_prim_start_list, dtype=np.int32)

    prim_exp = np.ascontiguousarray(np.concatenate(prim_exp_parts, axis=0) if prim_exp_parts else np.zeros((0,), dtype=np.float64))
    prim_coef = np.ascontiguousarray(
        np.concatenate(prim_coef_parts, axis=0) if prim_coef_parts else np.zeros((0,), dtype=np.float64)
    )

    # Recompute AO starts for the subset basis.
    shell_ao_start = np.empty((int(shell_ids_np.size),), dtype=np.int32)
    ao_cursor = 0
    for i, l in enumerate(shell_l.tolist()):
        shell_ao_start[i] = int(ao_cursor)
        ao_cursor += int(ncart(int(l)))

    source_bas_id = None
    if getattr(basis, "source_bas_id", None) is not None:
        source_bas_id = np.ascontiguousarray(np.asarray(basis.source_bas_id)[shell_ids_np])
    source_ctr_id = None
    if getattr(basis, "source_ctr_id", None) is not None:
        source_ctr_id = np.ascontiguousarray(np.asarray(basis.source_ctr_id)[shell_ids_np])

    return BasisCartSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=np.ascontiguousarray(shell_prim_start),
        shell_nprim=shell_nprim,
        shell_l=shell_l,
        shell_ao_start=np.ascontiguousarray(shell_ao_start),
        prim_exp=prim_exp,
        prim_coef=prim_coef,
        source_bas_id=source_bas_id,
        source_ctr_id=source_ctr_id,
    )


__all__ = ["subset_cart_basis_by_shells"]

