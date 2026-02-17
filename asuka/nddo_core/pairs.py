"""Pair list builder and local diatomic frame construction."""

from __future__ import annotations

import numpy as np


def build_pair_list(coords_bohr: np.ndarray):
    """Build all N(N-1)/2 atom pairs.

    Parameters
    ----------
    coords_bohr : (N, 3) array
        Atomic coordinates in Bohr.

    Returns
    -------
    pair_i, pair_j : (npairs,) int arrays
        Atom indices for each pair (i < j).
    pair_R : (npairs, 3) array
        Bond vectors R_ij = coords[j] - coords[i].
    pair_r : (npairs,) array
        Bond lengths |R_ij|.
    """
    N = len(coords_bohr)
    idx = np.triu_indices(N, k=1)
    pair_i = idx[0]
    pair_j = idx[1]
    pair_R = coords_bohr[pair_j] - coords_bohr[pair_i]
    pair_r = np.linalg.norm(pair_R, axis=1)
    return pair_i, pair_j, pair_R, pair_r


def build_local_frames(pair_R: np.ndarray, pair_r: np.ndarray) -> np.ndarray:
    """Build rotation matrices for each atom pair.

    Returns U[npairs, 3, 3] where rows are (ex, ey, ez) with ez along the bond.
    Convention: p_local = U @ p_global, sigma along z (row 2).
    """
    npairs = len(pair_r)
    U = np.zeros((npairs, 3, 3), dtype=float)

    for k in range(npairs):
        ez = pair_R[k] / pair_r[k]
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, ez)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        ex = np.cross(ref, ez)
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        U[k, 0] = ex
        U[k, 1] = ey
        U[k, 2] = ez

    return U


def block_rotation(norb: int, U_p: np.ndarray) -> np.ndarray:
    """Build block rotation for atom with norb=1 (H) or 4 (sp)."""
    if norb == 1:
        return np.eye(1)
    T = np.zeros((4, 4), dtype=float)
    T[0, 0] = 1.0
    T[1:4, 1:4] = U_p
    return T
