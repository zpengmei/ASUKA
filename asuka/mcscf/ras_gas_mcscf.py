from __future__ import annotations

from types import MethodType
from typing import Sequence

import numpy as np


def attach_gas_orbital_rotation_mask(
    mc,
    *,
    block_sizes: Sequence[int],
    verbose: int = 0,
    disable_canonicalization: bool = True,
):
    """Enable inter-block active-active orbital rotations for RAS/GAS.

    This helper patches ``mc.uniq_var_indices`` to allow active-active rotations
    *between* contiguous active blocks while keeping intra-block rotations
    excluded (redundant).

    Parameters
    ----------
    mc : Any
        The CASSCF object.
    block_sizes : Sequence[int]
        Sizes of GAS active blocks.
    verbose : int, optional
        Verbosity level.
    disable_canonicalization : bool, optional
        Whether to disable canonicalization (default True).

    Returns
    -------
    Any
        The modified CASSCF object.

    Notes
    -----
    - This function must remain *import-time PySCF free*; PySCF integration is
      driven by monkeypatching an existing MCSCF object instance.
    - The patch relies on PySCF's own ``uniq_var_indices`` implementation to
      apply symmetry and frozen-orbital restrictions. To preserve that behavior,
      it temporarily sets ``mc.internal_rotation = True`` when building a
      "full" active-active mask and then intersects it with the inter-block
      selection.
    """

    block_sizes_t = tuple(int(x) for x in block_sizes)
    if not block_sizes_t:
        raise ValueError("block_sizes must be non-empty")
    if any(x < 0 for x in block_sizes_t):
        raise ValueError("block_sizes entries must be >= 0")

    block_sizes_nz = tuple(x for x in block_sizes_t if x > 0)
    if not block_sizes_nz:
        raise ValueError("block_sizes must include at least one positive entry")

    if not hasattr(mc, "uniq_var_indices") or not callable(getattr(mc, "uniq_var_indices")):
        raise TypeError("mc must define a callable uniq_var_indices method")

    prev_uniq = mc.uniq_var_indices

    def _uniq_var_indices_gas(self, nmo, ncore, ncas, frozen):
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        if sum(block_sizes_nz) != ncas:
            raise ValueError("sum(block_sizes) must equal ncas")

        mask_base = np.asarray(prev_uniq(nmo, ncore, ncas, frozen), dtype=bool)
        if mask_base.shape != (nmo, nmo):
            raise ValueError("uniq_var_indices returned an unexpected shape")

        if not hasattr(self, "internal_rotation"):
            raise RuntimeError("mc must have an internal_rotation attribute for GAS/RAS masking")

        old_internal = bool(getattr(self, "internal_rotation"))
        try:
            setattr(self, "internal_rotation", True)
            mask_full = np.asarray(prev_uniq(nmo, ncore, ncas, frozen), dtype=bool)
        finally:
            setattr(self, "internal_rotation", old_internal)

        if mask_full.shape != (nmo, nmo):
            raise ValueError("uniq_var_indices returned an unexpected shape (internal_rotation=True)")

        # Select inter-block active-active rotations (lower triangle only).
        blocks: list[tuple[int, int]] = []
        start = 0
        for size in block_sizes_nz:
            stop = start + int(size)
            blocks.append((ncore + start, ncore + stop))
            start = stop

        mask_inter = np.zeros((nmo, nmo), dtype=bool)
        for bi, (si, ei) in enumerate(blocks):
            for sj, ej in (blocks[bj] for bj in range(bi + 1, len(blocks))):
                mask_inter[sj:ej, si:ei] = True

        # Respect PySCF's symmetry/frozen constraints by intersecting with the
        # "full internal rotation" mask, then add to the baseline mask.
        mask_add = mask_inter & mask_full
        return mask_base | mask_add

    mc.uniq_var_indices = MethodType(_uniq_var_indices_gas, mc)
    mc._cuguga_active_blocks = block_sizes_nz  # type: ignore[attr-defined]
    if bool(disable_canonicalization) and hasattr(mc, "canonicalization"):
        mc.canonicalization = False
    if int(verbose) > 0 and hasattr(mc, "stdout"):
        print(f"cuGUGA: patched uniq_var_indices for GAS blocks {block_sizes_nz}", file=mc.stdout)
    return mc


def attach_ras_orbital_rotation_mask(
    mc,
    *,
    nras1: int,
    nras2: int,
    nras3: int,
    verbose: int = 0,
):
    """Convenience wrapper for 3-block RAS active spaces.

    Parameters
    ----------
    mc : Any
        The CASSCF object.
    nras1 : int
        Size of RAS1 space.
    nras2 : int
        Size of RAS2 space.
    nras3 : int
        Size of RAS3 space.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    Any
        The modified CASSCF object.
    """

    n1 = int(nras1)
    n2 = int(nras2)
    n3 = int(nras3)
    if n1 < 0 or n2 < 0 or n3 < 0:
        raise ValueError("nras1/nras2/nras3 must be >= 0 for RAS masking")
    return attach_gas_orbital_rotation_mask(mc, block_sizes=(n1, n2, n3), verbose=int(verbose))
