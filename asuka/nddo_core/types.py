"""Typed containers for packed NDDO atom/pair data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomicData:
    """Atom-wise packed data for NDDO kernels."""

    Z: np.ndarray
    coords_bohr: np.ndarray
    nao_per_atom: np.ndarray
    ao_offset: np.ndarray


@dataclass(frozen=True)
class PairData:
    """Pair-wise packed data for NDDO kernels."""

    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_type: np.ndarray
    pair_r: np.ndarray
    pair_R: np.ndarray
