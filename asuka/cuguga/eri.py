from __future__ import annotations

from typing import Any

import numpy as np


def _npair_s2(norb: int) -> int:
    norb = int(norb)
    return norb * (norb + 1) // 2


def _pair_index_s2(p: int, q: int) -> int:
    p = int(p)
    q = int(q)
    if p < q:
        p, q = q, p
    return p * (p + 1) // 2 + q


def _pair_id_map_s2(norb: int) -> np.ndarray:
    """Return pair_id[p,q] mapping ordered pairs to packed s2 indices."""

    norb = int(norb)
    pair_id = np.empty((norb, norb), dtype=np.intp)
    for p in range(norb):
        for q in range(norb):
            pair_id[p, q] = _pair_index_s2(p, q)
    return pair_id


def _unpack_tril(x: np.ndarray, n: int) -> np.ndarray:
    """Unpack a packed lower triangle vector into a symmetric (n,n) matrix."""

    n = int(n)
    x = np.asarray(x, dtype=np.float64).ravel()
    expected = n * (n + 1) // 2
    if int(x.size) != int(expected):
        raise ValueError("packed tril has wrong length")

    out = np.zeros((n, n), dtype=np.float64)
    tri = np.tril_indices(n)
    out[tri] = x
    out = out + np.tril(out, -1).T
    return out


def restore_eri1(eri: Any, norb: int) -> np.ndarray:
    """Return ERIs in full 4-index form (sym=1), chemist notation (pq|rs).

    Accepts several packed representations and unpacks to a full
    ``(norb, norb, norb, norb)`` tensor.

    Parameters
    ----------
    eri : array_like
        Electron repulsion integrals in one of these formats:

        - ``(norb, norb, norb, norb)`` — full tensor (returned as-is).
        - ``(npair, npair)`` with ``npair = norb*(norb+1)//2`` — sym=4 pair
          matrix.
        - ``(npair*(npair+1)//2,)`` — packed lower triangle of the sym=4
          pair matrix (sym=8).
    norb : int
        Number of spatial orbitals.

    Returns
    -------
    np.ndarray
        C-contiguous float64 array of shape ``(norb, norb, norb, norb)``.
    """

    norb = int(norb)
    if norb < 0:
        raise ValueError("norb must be >= 0")

    eri_arr = np.asarray(eri, dtype=np.float64)
    if eri_arr.ndim == 4:
        if eri_arr.shape != (norb, norb, norb, norb):
            raise ValueError("eri has wrong shape")
        return np.asarray(eri_arr, dtype=np.float64, order="C")

    if norb == 0:
        return np.zeros((0, 0, 0, 0), dtype=np.float64)

    npair = _npair_s2(norb)
    eri2: np.ndarray
    if eri_arr.ndim == 2:
        if eri_arr.shape != (npair, npair):
            raise ValueError("eri pair-matrix has wrong shape")
        eri2 = np.asarray(eri_arr, dtype=np.float64)
    elif eri_arr.ndim == 1:
        eri1 = np.asarray(eri_arr, dtype=np.float64).ravel()
        if int(eri1.size) == npair * (npair + 1) // 2:
            eri2 = _unpack_tril(eri1, npair)
        elif int(eri1.size) == npair * npair:
            eri2 = eri1.reshape(npair, npair)
        elif int(eri1.size) == norb**4:
            return np.asarray(eri1.reshape(norb, norb, norb, norb), dtype=np.float64, order="C")
        else:
            raise ValueError("unsupported eri packed format")
    else:
        raise ValueError("unsupported eri rank")

    pair_id = _pair_id_map_s2(norb)
    eri4 = eri2[pair_id[:, :, None, None], pair_id[None, None, :, :]]
    return np.asarray(eri4, dtype=np.float64, order="C")


def restore_eri4(eri: Any, norb: int) -> np.ndarray:
    """Return ERIs as a sym=4 pair matrix with shape ``(npair, npair)``.

    Parameters
    ----------
    eri : array_like
        Electron repulsion integrals in any format accepted by
        :func:`restore_eri1`.
    norb : int
        Number of spatial orbitals.

    Returns
    -------
    np.ndarray
        C-contiguous float64 array of shape ``(npair, npair)`` where
        ``npair = norb * (norb + 1) // 2``.
    """

    norb = int(norb)
    if norb < 0:
        raise ValueError("norb must be >= 0")
    if norb == 0:
        return np.zeros((0, 0), dtype=np.float64)

    eri_arr = np.asarray(eri, dtype=np.float64)
    if eri_arr.ndim == 4:
        if eri_arr.shape != (norb, norb, norb, norb):
            raise ValueError("eri has wrong shape")
        eri4 = np.asarray(eri_arr, dtype=np.float64)
    else:
        eri4 = restore_eri1(eri, norb)
    npair = _npair_s2(norb)

    pairs_p: list[int] = []
    pairs_q: list[int] = []
    for p in range(norb):
        for q in range(p + 1):
            pairs_p.append(p)
            pairs_q.append(q)

    p_idx = np.asarray(pairs_p, dtype=np.intp)
    q_idx = np.asarray(pairs_q, dtype=np.intp)
    if int(p_idx.size) != int(npair):
        raise RuntimeError("internal error: pair list size mismatch")

    eri2 = eri4[p_idx[:, None], q_idx[:, None], p_idx[None, :], q_idx[None, :]]
    eri2 = 0.5 * (eri2 + eri2.T)
    return np.asarray(eri2, dtype=np.float64, order="C")
