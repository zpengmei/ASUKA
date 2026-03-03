from __future__ import annotations

"""Packed lower-triangular ("s2") helpers for symmetric AO-pair storage.

We use the canonical packed ordering for unordered AO pairs (mu,nu) with mu>=nu:

  p(mu,nu) = mu*(mu+1)//2 + nu

This matches NumPy's `np.tril_indices(nao)` ordering and the common PySCF "s2"
convention.
"""

from typing import Any

import numpy as np


def ntri_from_nao(nao: int) -> int:
    nao_i = int(nao)
    if nao_i < 0:
        raise ValueError("nao must be >= 0")
    return nao_i * (nao_i + 1) // 2


def nao_from_ntri(ntri: int) -> int:
    """Infer nao from ntri=nao*(nao+1)//2 and validate exactness."""

    ntri_i = int(ntri)
    if ntri_i < 0:
        raise ValueError("ntri must be >= 0")
    if ntri_i == 0:
        return 0

    # Solve nao^2 + nao - 2*ntri = 0 for positive nao.
    nao_f = (np.sqrt(8.0 * float(ntri_i) + 1.0) - 1.0) * 0.5
    nao_i = int(np.floor(nao_f + 1e-12))
    if ntri_from_nao(nao_i) != ntri_i:
        raise ValueError(f"ntri={ntri_i} is not a triangular number nao*(nao+1)//2")
    return nao_i


_WEIGHTS_CACHE: dict[tuple[int, str, bool], Any] = {}


def tri_weights(xp, nao: int, *, dtype: Any | None = None):
    """Return packed-pair weights for converting full sums to packed sums.

    For symmetric matrices, sums over all ordered pairs (mu,nu) can be written
    as sums over packed unordered pairs p with:
      w[p] = 1 for mu==nu (diagonal)
      w[p] = 2 for mu!=nu (off-diagonal)
    """

    nao_i = int(nao)
    if nao_i < 0:
        raise ValueError("nao must be >= 0")
    if dtype is None:
        dtype = xp.float64
    key = (nao_i, str(dtype), xp is not np)
    hit = _WEIGHTS_CACHE.get(key)
    if hit is not None:
        return hit

    ntri = ntri_from_nao(nao_i)
    w = xp.full((int(ntri),), 2.0, dtype=dtype)
    if nao_i:
        # Packed diagonal index p(mu,mu) = mu*(mu+1)//2 + mu = mu*(mu+3)//2
        mu = xp.arange(int(nao_i), dtype=xp.int64)
        diag = (mu * (mu + 3)) // 2
        w[diag] = xp.asarray(1.0, dtype=dtype)

    _WEIGHTS_CACHE[key] = w
    return w


def pack_tril(xp, M):
    """Pack the lower triangle (mu>=nu) of a square matrix into s2 order."""

    arr = xp.asarray(M)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError("M must be a square 2D matrix")
    nao = int(arr.shape[0])
    if nao == 0:
        return xp.empty((0,), dtype=arr.dtype)

    tri_i, tri_j = (np.tril_indices(nao) if xp is np else xp.tril_indices(nao))
    return arr[tri_i, tri_j]


def unpack_tril(xp, v, *, nao: int):
    """Unpack a packed s2 vector into a full symmetric (nao,nao) matrix."""

    nao_i = int(nao)
    if nao_i < 0:
        raise ValueError("nao must be >= 0")
    vec = xp.asarray(v)
    if vec.ndim != 1:
        raise ValueError("v must be 1D")
    expected = ntri_from_nao(nao_i)
    if int(vec.size) != int(expected):
        raise ValueError(f"v has size {int(vec.size)} but expected ntri={int(expected)} for nao={nao_i}")

    out = xp.zeros((nao_i, nao_i), dtype=vec.dtype)
    if nao_i == 0:
        return out
    tri_i, tri_j = (np.tril_indices(nao_i) if xp is np else xp.tril_indices(nao_i))
    out[tri_i, tri_j] = vec
    out[tri_j, tri_i] = vec
    return out


__all__ = ["ntri_from_nao", "nao_from_ntri", "tri_weights", "pack_tril", "unpack_tril"]

