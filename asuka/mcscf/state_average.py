from __future__ import annotations

"""State-averaging utilities for multiroot CASCI/CASSCF.

This module provides small helpers for:
  - normalizing SA weights
  - canonicalizing multiroot CI containers
  - following/matching roots by overlap between iterations
  - building state-averaged active-space RDMs from per-root CI vectors

All functions are NumPy-first; heavy tensor contractions (AO/DF) belong in the
main MCSCF drivers.
"""

from typing import Any, Sequence

import numpy as np


def normalize_weights(weights: Sequence[float] | None, *, nroots: int) -> np.ndarray:
    """Return normalized non-negative weights of shape (nroots,).

    Parameters
    ----------
    weights : Sequence[float] | None
        Input weights (sum must be positive). If None, equal weights are used.
    nroots : int
        Number of roots.

    Returns
    -------
    np.ndarray
        Normalized weights (sum=1).
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    if weights is None:
        return np.ones((nroots,), dtype=np.float64) / float(nroots)

    w = np.asarray(list(weights), dtype=np.float64).ravel()
    if int(w.size) != int(nroots):
        raise ValueError(f"weights must have length {nroots}, got {int(w.size)}")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to > 0")
    return np.asarray(w / s, dtype=np.float64)


def ci_as_list(ci: Any, *, nroots: int) -> list[np.ndarray]:
    """Canonicalize a multiroot CI container to a list of 1D float64 arrays.

    Parameters
    ----------
    ci : Any
        Input CI vector(s). Can be list, tuple, or array.
    nroots : int
        Number of roots expected.

    Returns
    -------
    list[np.ndarray]
        List of 1D CI vectors.
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    if nroots == 1:
        return [np.asarray(ci, dtype=np.float64).ravel()]

    if isinstance(ci, (list, tuple)):
        if len(ci) != nroots:
            raise ValueError("CI list length mismatch")
        return [np.asarray(x, dtype=np.float64).ravel() for x in ci]

    arr = np.asarray(ci, dtype=np.float64)
    if arr.ndim == 2 and int(arr.shape[0]) == nroots:
        return [np.asarray(arr[i], dtype=np.float64).ravel().copy() for i in range(nroots)]

    raise TypeError("Unsupported CI format for multiroot")


def match_roots_by_overlap(prev: list[np.ndarray], cur: list[np.ndarray]) -> np.ndarray:
    """Return permutation indices mapping current roots onto previous roots.

    Solves the linear assignment problem to maximize overlap between root sets.

    Parameters
    ----------
    prev : list[np.ndarray]
        Previous iteration CI vectors.
    cur : list[np.ndarray]
        Current iteration CI vectors.

    Returns
    -------
    np.ndarray
        Permutation indices such that `cur[perm[i]]` matches `prev[i]`.
    """

    n = int(len(prev))
    if n != int(len(cur)):
        raise ValueError("prev/cur root count mismatch")
    if n <= 1:
        return np.arange(n, dtype=np.int32)

    O = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        pi = prev[i]
        for j in range(n):
            O[i, j] = abs(float(np.dot(pi, cur[j])))

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]  # noqa: PLC0415

        row, col = linear_sum_assignment(-O)
        perm = np.empty((n,), dtype=np.int32)
        perm[row.astype(np.int32, copy=False)] = col.astype(np.int32, copy=False)
        return perm
    except Exception:
        # Greedy fallback (nroots is typically small).
        used: set[int] = set()
        perm = np.empty((n,), dtype=np.int32)
        for i in range(n):
            j_best = None
            v_best = -1.0
            for j in range(n):
                if j in used:
                    continue
                v = float(O[i, j])
                if v > v_best:
                    v_best = v
                    j_best = j
            if j_best is None:  # pragma: no cover
                j_best = next(iter(set(range(n)) - used))
            used.add(int(j_best))
            perm[i] = int(j_best)
        return perm


def fix_ci_phases(prev: list[np.ndarray], cur: list[np.ndarray]) -> None:
    """Flip CI signs so that <prev_i|cur_i> >= 0 for each root i.

    Parameters
    ----------
    prev : list[np.ndarray]
        Reference CI vectors.
    cur : list[np.ndarray]
        Target CI vectors (modified in-place).
    """

    if len(prev) != len(cur):
        raise ValueError("prev/cur root count mismatch")
    for i in range(len(prev)):
        ov = float(np.dot(prev[i], cur[i]))
        if ov < 0.0:
            cur[i] *= -1.0


def make_state_averaged_rdms(
    fcisolver: Any,
    ci_list: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    ncas: int,
    nelecas: int | tuple[int, int],
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dm1_sa, dm2_sa) in the active space as NumPy float64 arrays.

    Parameters
    ----------
    fcisolver : Any
        FCI solver object.
    ci_list : Sequence[np.ndarray]
        List of CI vectors.
    weights : Sequence[float]
        State weights.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of active electrons.
    solver_kwargs : dict[str, Any] | None, optional
        Extra kwargs for `make_rdm12`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dm1_sa, dm2_sa).
    """

    if solver_kwargs is None:
        solver_kwargs = {}

    ncas = int(ncas)
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    w = np.asarray(list(weights), dtype=np.float64).ravel()
    if int(w.size) != int(len(ci_list)):
        raise ValueError("weights/ci_list length mismatch")

    dm1 = np.zeros((ncas, ncas), dtype=np.float64)
    dm2 = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)
    base_cls = getattr(fcisolver, "_base_class", None)
    if base_cls is not None and hasattr(base_cls, "make_rdm12"):
        # PySCF's state-average wrappers implement make_rdm12 with list semantics.
        # In SA contexts we need the *per-root* make_rdm12 from the base solver class.
        make_rdm12 = getattr(base_cls, "make_rdm12")
        for wi, civec in zip(w.tolist(), ci_list):
            dm1_i, dm2_i = make_rdm12(fcisolver, civec, ncas, nelecas, **solver_kwargs)
            dm1 += float(wi) * np.asarray(dm1_i, dtype=np.float64)
            dm2 += float(wi) * np.asarray(dm2_i, dtype=np.float64)
    else:
        for wi, civec in zip(w.tolist(), ci_list):
            dm1_i, dm2_i = fcisolver.make_rdm12(civec, ncas, nelecas, **solver_kwargs)
            dm1 += float(wi) * np.asarray(dm1_i, dtype=np.float64)
            dm2 += float(wi) * np.asarray(dm2_i, dtype=np.float64)
    return dm1, dm2


__all__ = [
    "ci_as_list",
    "fix_ci_phases",
    "make_state_averaged_rdms",
    "match_roots_by_overlap",
    "normalize_weights",
]
