from __future__ import annotations

"""THC J/K contractions.

We assume an LS-THC/THC factorization of AO ERIs:

  (mu nu | la si) ~= sum_{P,Q} X_{P,mu} X_{P,nu} Z_{P,Q} X_{Q,la} X_{Q,si}

with:
- X: (npt, nao) weighted AO collocation (points-major)
- Z: (npt, npt) central metric (symmetric)

This module provides J and K builds in O(N^3) using GEMMs plus a blocked K
algorithm that avoids allocating M = X D X^T as a full (npt,npt) temporary.
"""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

_HF_THC_CUDA_EXT = None


def _load_hf_thc_cuda_ext():
    global _HF_THC_CUDA_EXT
    if _HF_THC_CUDA_EXT is False:
        return None
    if _HF_THC_CUDA_EXT is not None:
        return _HF_THC_CUDA_EXT
    try:
        from asuka import _hf_thc_cuda_ext  # type: ignore[import-not-found]
    except Exception:
        _HF_THC_CUDA_EXT = False
        return None
    _HF_THC_CUDA_EXT = _hf_thc_cuda_ext
    return _HF_THC_CUDA_EXT


def _thc_jk_impl_pref() -> str:
    return str(os.environ.get("ASUKA_THC_JK_IMPL", "auto")).strip().lower()


def _get_xp(*arrays: Any):
    try:
        import cupy as cp  # type: ignore
    except Exception:  # pragma: no cover
        cp = None  # type: ignore
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _as_xp(xp, a, *, dtype):
    return xp.asarray(a, dtype=dtype)


def _symmetrize(xp, A):
    return 0.5 * (A + A.T)


@dataclass
class THCJKWork:
    """Reusable work config for THC J/K builds."""

    q_block: int = 256


def thc_J(D, X, Z):
    """Compute Coulomb matrix J[D] via THC factors.

    Inputs
    - D: (nao,nao) density (symmetric)
    - X: (npt,nao) weighted AO collocation (points-major)
    - Z: (npt,npt) central metric
    """

    xp, _is_gpu = _get_xp(D, X, Z)
    D = _as_xp(xp, D, dtype=xp.float64)
    X = _as_xp(xp, X, dtype=xp.float64)
    Z = _as_xp(xp, Z, dtype=xp.float64)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with D")
    if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
        raise ValueError("Z must have shape (npt,npt)")

    impl = _thc_jk_impl_pref()
    ext = _load_hf_thc_cuda_ext() if _is_gpu and impl in {"auto", "ext"} else None

    # A[P,nu] = sum_mu X[P,mu] D[mu,nu]
    A = X @ D  # (npt,nao)

    if ext is not None:
        # Avoid allocating the A*X intermediate; compute the rowwise dot directly.
        try:
            stream_ptr = int(xp.cuda.get_current_stream().ptr)
        except Exception:  # pragma: no cover
            stream_ptr = 0
        m = xp.empty((npt,), dtype=xp.float64)
        ext.rowwise_dot_f64(A, X, m, threads=256, stream_ptr=stream_ptr, sync=False)
    else:
        # m[P] = X[P,:] · A[P,:] = X[P,:] D X[P,:]^T
        m = xp.sum(A * X, axis=1)  # (npt,)

    # n = Z @ m
    n = Z @ m  # (npt,)

    if ext is not None:
        # Reuse A as the row-scaled buffer (A <- X * n[:,None]).
        ext.scale_rows_f64(X, n, A, threads=256, stream_ptr=stream_ptr, sync=False)
        J = X.T @ A
    else:
        # J = X^T (n*X)
        J = X.T @ (X * n[:, None])
    return _symmetrize(xp, J)


def thc_K_blocked(D, X, Z, *, q_block: int = 256):
    """Compute exchange matrix K[D] via THC factors in blocks over point index.

    Avoids allocating M = X D X^T as a full (npt,npt) temporary.
    """

    xp, _is_gpu = _get_xp(D, X, Z)
    D = _as_xp(xp, D, dtype=xp.float64)
    X = _as_xp(xp, X, dtype=xp.float64)
    Z = _as_xp(xp, Z, dtype=xp.float64)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with D")
    if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
        raise ValueError("Z must have shape (npt,npt)")

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")
    q_block = min(int(q_block), int(npt))

    impl = _thc_jk_impl_pref()
    ext = _load_hf_thc_cuda_ext() if _is_gpu and impl in {"auto", "ext"} else None
    if ext is not None:
        try:
            stream_ptr = int(xp.cuda.get_current_stream().ptr)
        except Exception:  # pragma: no cover
            stream_ptr = 0

    K = xp.zeros((nao, nao), dtype=xp.float64)

    XT = X.T
    for q0 in range(0, int(npt), int(q_block)):
        q1 = min(int(npt), int(q0) + int(q_block))
        nb = int(q1 - q0)
        if nb <= 0:
            continue

        Xq = X[int(q0) : int(q1), :]  # (nb,nao)

        # Aq = D @ Xq.T   (nao,nb)
        Aq = D @ Xq.T

        # Mq = X @ Aq = (X D X^T)[:, q0:q1]  (npt,nb)
        Mq = X @ Aq

        if ext is not None:
            # In-place Hadamard product without materializing Tq.
            Zblk = Z[:, int(q0) : int(q1)]
            ext.hadamard_inplace_f64(Mq, Zblk, threads=256, stream_ptr=stream_ptr, sync=False)
            Fq = XT @ Mq
        else:
            # Tq = Z[:, q0:q1] ⊙ Mq  (npt,nb)
            Tq = Z[:, int(q0) : int(q1)] * Mq

            # Fq = X^T @ Tq   (nao,nb)
            Fq = XT @ Tq

        # Accumulate: K += Fq @ Xq
        K += Fq @ Xq

        if ext is not None:
            del Xq, Aq, Mq, Fq
        else:
            del Xq, Aq, Mq, Tq, Fq

    return _symmetrize(xp, K)


def thc_JK(D, X, Z, *, work: THCJKWork | None = None):
    work = THCJKWork() if work is None else work
    J = thc_J(D, X, Z)
    K = thc_K_blocked(D, X, Z, q_block=int(work.q_block))
    return J, K


__all__ = ["THCJKWork", "thc_J", "thc_JK", "thc_K_blocked"]
