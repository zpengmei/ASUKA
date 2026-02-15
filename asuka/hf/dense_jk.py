from __future__ import annotations

"""Dense AO-ERI Coulomb/exchange contractions.

This module contracts a full AO ordered-pair ERI matrix with one or more AO
density matrices. Input ERIs are expected in the `eri_mat[pq, rs]` layout with:
  `pq = p * nao + q`, `rs = r * nao + s`.
"""

from typing import Any

import numpy as np

from .df_jk import _as_xp, _get_xp, _symmetrize


def _as_eri_tensor(eri_mat, nao: int):
    if getattr(eri_mat, "ndim", None) != 2:
        raise ValueError("eri_mat must be a 2D square matrix")
    n2 = int(nao) * int(nao)
    if tuple(map(int, eri_mat.shape)) != (n2, n2):
        raise ValueError(f"eri_mat must have shape ({n2},{n2}), got {tuple(map(int, eri_mat.shape))}")
    return eri_mat.reshape((int(nao), int(nao), int(nao), int(nao)))


def dense_J_from_eri_mat_D(eri_mat, D):
    """Compute Coulomb matrix J from dense AO ERIs and AO density D."""

    xp, _ = _get_xp(eri_mat, D)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    g = _as_eri_tensor(eri_mat, nao)
    J = xp.einsum("mnls,ls->mn", g, D, optimize=True)
    return _symmetrize(xp, J)


def dense_K_from_eri_mat_D(eri_mat, D):
    """Compute exchange matrix K from dense AO ERIs and AO density D."""

    xp, _ = _get_xp(eri_mat, D)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    g = _as_eri_tensor(eri_mat, nao)
    # K_mn = sum_ls D_ls * (m l | n s)
    K = xp.einsum("mlns,ls->mn", g, D, optimize=True)
    return _symmetrize(xp, K)


def dense_JK_from_eri_mat_D(
    eri_mat,
    D,
    *,
    want_J: bool = True,
    want_K: bool = True,
) -> tuple[Any | None, Any | None]:
    """Compute Coulomb/exchange matrices from dense AO ERIs and AO density D."""

    if not bool(want_J) and not bool(want_K):
        return None, None

    xp, _ = _get_xp(eri_mat, D)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    g = _as_eri_tensor(eri_mat, nao)

    J = xp.einsum("mnls,ls->mn", g, D, optimize=True) if bool(want_J) else None
    K = xp.einsum("mlns,ls->mn", g, D, optimize=True) if bool(want_K) else None
    if J is not None:
        J = _symmetrize(xp, J)
    if K is not None:
        K = _symmetrize(xp, K)
    return J, K


__all__ = [
    "dense_J_from_eri_mat_D",
    "dense_JK_from_eri_mat_D",
    "dense_K_from_eri_mat_D",
]
