from __future__ import annotations

"""Orbital tracking utilities for geometry-continuous CASSCF.

This module provides functions to track orbital identity across geometry changes,
preventing active space drift during molecular dynamics, geometry optimization,
and coordinate scans.
"""

from typing import Literal, Sequence

import numpy as np


def compute_mo_overlap(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    S_cross: np.ndarray,
) -> np.ndarray:
    """Compute MO overlap matrix: O[i,j] = ⟨mo_prev[i]|mo_new[j]⟩.

    Parameters
    ----------
    C_prev : np.ndarray
        MO coefficients at previous geometry (nao_prev, nmo_prev)
    C_new : np.ndarray
        MO coefficients at new geometry (nao_new, nmo_new)
    S_cross : np.ndarray
        Cross-geometry AO overlap (nao_prev, nao_new)

    Returns
    -------
    O : np.ndarray
        MO overlap matrix (nmo_prev, nmo_new)

    Notes
    -----
    The MO overlap is computed as:
        O = C_prev^T @ S_cross @ C_new

    where S_cross[μ,ν] = ⟨χ_μ(R_prev)|χ_ν(R_new)⟩.

    Examples
    --------
    >>> C_prev = np.eye(10)
    >>> C_new = np.eye(10)
    >>> S_cross = np.eye(10)
    >>> O = compute_mo_overlap(C_prev, C_new, S_cross)
    >>> np.allclose(O, np.eye(10))
    True
    """
    C_prev = np.asarray(C_prev, dtype=np.float64)
    C_new = np.asarray(C_new, dtype=np.float64)
    S_cross = np.asarray(S_cross, dtype=np.float64)

    if C_prev.ndim != 2:
        raise ValueError("C_prev must be a 2D array")
    if C_new.ndim != 2:
        raise ValueError("C_new must be a 2D array")
    if S_cross.ndim != 2:
        raise ValueError("S_cross must be a 2D array")

    nao_prev, nmo_prev = C_prev.shape
    nao_new, nmo_new = C_new.shape

    if S_cross.shape != (nao_prev, nao_new):
        raise ValueError(
            f"S_cross shape {S_cross.shape} incompatible with C_prev {C_prev.shape} and C_new {C_new.shape}"
        )

    return C_prev.T @ S_cross @ C_new


def assign_active_orbitals_by_overlap(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    S_cross: np.ndarray,
    prev_active_idx: Sequence[int],
    ncas: int,
    *,
    method: Literal["subspace", "hungarian"] = "subspace",
) -> np.ndarray:
    """Select/reorder new orbitals to match previous active space.

    This function identifies which new orbitals at the current geometry should
    be used as the active space, based on maximum overlap with the previous
    active space.

    Parameters
    ----------
    C_prev : np.ndarray
        Previous MO coefficients (nao_prev, nmo_prev)
    C_new : np.ndarray
        New MO coefficients (nao_new, nmo_new)
    S_cross : np.ndarray
        Cross-geometry AO overlap (nao_prev, nao_new)
    prev_active_idx : Sequence[int]
        Indices of previous active orbitals
    ncas : int
        Number of active orbitals
    method : "subspace" or "hungarian"
        - "subspace": Score by total overlap with active subspace (robust, recommended)
        - "hungarian": 1-to-1 optimal assignment (good for small CAS, requires scipy)

    Returns
    -------
    new_active_idx : np.ndarray
        Indices of new orbitals to use as active space (length ncas)

    Notes
    -----
    **Subspace method** (default):
        - Computes score[j] = Σ_i∈active |O[i,j]|² for each new orbital j
        - Selects top ncas orbitals by score
        - Robust to orbital mixing and reordering
        - No external dependencies

    **Hungarian method**:
        - Solves 1-to-1 optimal assignment problem
        - Requires scipy for linear_sum_assignment
        - Better for small, well-separated active spaces
        - May fail if orbitals mix significantly

    Examples
    --------
    >>> # Mock case: previous active=[2,3,4] overlaps with new [3,4,5]
    >>> nmo = 10
    >>> C_prev = np.eye(nmo)
    >>> C_new = np.eye(nmo)
    >>> C_new[:, 3:6] = C_prev[:, 2:5]  # Shift active orbitals by 1
    >>> S = np.eye(nmo)
    >>> prev_active = [2, 3, 4]
    >>> new_active = assign_active_orbitals_by_overlap(
    ...     C_prev, C_new, S, prev_active, ncas=3, method="subspace"
    ... )
    >>> set(new_active)
    {3, 4, 5}
    """
    prev_active_idx = list(prev_active_idx)
    if len(prev_active_idx) != ncas:
        raise ValueError(f"prev_active_idx length {len(prev_active_idx)} != ncas {ncas}")

    O = compute_mo_overlap(C_prev, C_new, S_cross)

    method_s = str(method).strip().lower()
    if method_s not in ("subspace", "hungarian"):
        raise ValueError("method must be 'subspace' or 'hungarian'")

    if method_s == "subspace":
        # Score each new MO by total overlap with previous active subspace
        # score[j] = Σ_i∈active |O[i,j]|²
        overlap_active = O[prev_active_idx, :]  # (ncas, nmo_new)
        scores = np.sum(np.abs(overlap_active) ** 2, axis=0)  # (nmo_new,)
        return np.argsort(-scores)[:ncas]  # Top ncas by score

    # method == "hungarian"
    # Use 1-to-1 optimal assignment
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "method='hungarian' requires scipy. Install with: pip install scipy"
        ) from e

    overlap_active = np.abs(O[prev_active_idx, :]) ** 2  # (ncas, nmo_new)

    # Solve linear sum assignment to maximize total overlap
    # Need cost matrix (minimize), so use negative overlap
    row, col = linear_sum_assignment(-overlap_active)

    if len(col) != ncas:
        raise RuntimeError(f"Assignment failed: expected {ncas} assignments, got {len(col)}")

    return np.asarray(col, dtype=np.int64)


def align_orbital_phases(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    S_cross: np.ndarray,
    *,
    alignment_idx: Sequence[int] | None = None,
) -> np.ndarray:
    """Fix orbital phases for continuity.

    If ⟨mo_prev[i]|mo_new[i]⟩ < 0, flip sign of C_new[:, i] to ensure
    positive diagonal overlaps.

    Parameters
    ----------
    C_prev : np.ndarray
        Previous MO coefficients (nao_prev, nmo_prev)
    C_new : np.ndarray
        New MO coefficients (nao_new, nmo_new)
    S_cross : np.ndarray
        Cross-geometry AO overlap (nao_prev, nao_new)
    alignment_idx : Sequence[int] | None
        Orbital indices to align (default: all overlapping orbitals)

    Returns
    -------
    C_aligned : np.ndarray
        Phase-aligned MO coefficients (nao_new, nmo_new)

    Notes
    -----
    Orbital phases (signs) are arbitrary in quantum chemistry, but for continuity
    across geometry changes, we want to maintain consistent phases. This function
    ensures that each orbital at the new geometry has positive overlap with the
    corresponding orbital at the previous geometry.

    Examples
    --------
    >>> nmo = 5
    >>> C_prev = np.eye(nmo)
    >>> C_new = np.eye(nmo)
    >>> C_new[:, 2] *= -1  # Flip one orbital
    >>> S = np.eye(nmo)
    >>> C_aligned = align_orbital_phases(C_prev, C_new, S)
    >>> O = compute_mo_overlap(C_prev, C_aligned, S)
    >>> np.all(np.diag(O) > 0)
    True
    """
    O = compute_mo_overlap(C_prev, C_new, S_cross)
    C_aligned = np.asarray(C_new, dtype=np.float64).copy()

    if alignment_idx is None:
        # Align all orbitals that exist in both
        alignment_idx = range(min(O.shape[0], C_aligned.shape[1]))

    for i in alignment_idx:
        if i < O.shape[0] and i < O.shape[1]:
            if O[i, i] < 0:
                C_aligned[:, i] *= -1

    return C_aligned


def reorder_mo_to_active_space(
    mo_coeff: np.ndarray,
    active_idx: Sequence[int],
    ncore: int,
) -> np.ndarray:
    """Reorder MO coefficients to place active_idx into [ncore:ncore+ncas].

    This is similar to sort_mo() but works with arbitrary indices instead of
    energy ordering.

    Parameters
    ----------
    mo_coeff : np.ndarray
        MO coefficient matrix (nao, nmo)
    active_idx : Sequence[int]
        Orbital indices to place in active space
    ncore : int
        Number of core orbitals

    Returns
    -------
    mo_reordered : np.ndarray
        Reordered MO coefficients (nao, nmo)

    Notes
    -----
    This function builds a permutation that places the specified orbitals into
    the active space [ncore:ncore+ncas], preserving the order of active_idx.

    Examples
    --------
    >>> mo = np.eye(10)
    >>> # Want orbitals [5,6,7] to become the active space at positions [2,3,4]
    >>> mo_reordered = reorder_mo_to_active_space(mo, active_idx=[5,6,7], ncore=2)
    >>> # Check that positions 2,3,4 now contain original orbitals 5,6,7
    >>> np.allclose(mo_reordered[:, 2:5], mo[:, [5,6,7]])
    True
    """
    mo_coeff = np.asarray(mo_coeff, dtype=np.float64)
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array")

    nao, nmo = mo_coeff.shape
    active_idx = list(active_idx)
    ncas = len(active_idx)

    if ncore < 0:
        raise ValueError(f"ncore must be >= 0, got {ncore}")
    if ncore + ncas > nmo:
        raise ValueError(f"ncore ({ncore}) + ncas ({ncas}) > nmo ({nmo})")
    if len(set(active_idx)) != ncas:
        raise ValueError("active_idx contains duplicates")
    if any(i < 0 or i >= nmo for i in active_idx):
        raise ValueError(f"active_idx contains out-of-range indices (nmo={nmo})")

    # Build permutation: place active_idx into [ncore:ncore+ncas]
    perm = list(range(nmo))
    target_positions = list(range(ncore, ncore + ncas))

    # For each target position, swap to bring the desired orbital there
    for tgt_idx, src_idx in zip(target_positions, active_idx):
        current_pos = perm.index(src_idx)
        # Swap
        perm[current_pos], perm[tgt_idx] = perm[tgt_idx], perm[current_pos]

    return mo_coeff[:, perm]
