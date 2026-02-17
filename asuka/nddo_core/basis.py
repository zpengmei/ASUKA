"""AO index bookkeeping for NDDO semiempirical methods."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Element symbol -> atomic number
_SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

# Valence electrons per element (core charge in NDDO)
_VALENCE_ELECTRONS = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7}


def symbol_to_Z(sym: str) -> int:
    """Convert element symbol to atomic number."""
    return _SYMBOL_TO_Z[sym]


def nao_for_Z(Z: int) -> int:
    """Number of AOs for element with atomic number Z.

    H: 1 (s only), heavy p-block second-row elements: 4 (s, px, py, pz).
    """
    if Z == 1:
        return 1
    if Z in (6, 7, 8, 9):
        return 4
    raise ValueError(f"Unsupported element Z={Z}")


def valence_electrons(Z: int) -> int:
    """Number of valence electrons (= core charge in NDDO)."""
    if Z not in _VALENCE_ELECTRONS:
        raise ValueError(f"Unsupported element Z={Z}")
    return _VALENCE_ELECTRONS[Z]


def build_ao_offsets(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Build prefix-sum array of AO offsets.

    Returns array of length N+1 where offsets[i] is the first AO index
    for atom i and offsets[N] is the total number of AOs.
    """
    nao_list = [nao_for_Z(Z) for Z in atomic_numbers]
    offsets = np.zeros(len(atomic_numbers) + 1, dtype=int)
    for i, n in enumerate(nao_list):
        offsets[i + 1] = offsets[i] + n
    return offsets
