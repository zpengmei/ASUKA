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
from contextlib import contextmanager
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
        from asuka.kernels import hf_thc as _hf_thc_kernels  # noqa: PLC0415
    except Exception:
        _HF_THC_CUDA_EXT = False
        return None
    ext = _hf_thc_kernels.load_ext()
    if ext is None:
        _HF_THC_CUDA_EXT = False
        return None
    _HF_THC_CUDA_EXT = ext
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
    if dtype is None:
        return xp.asarray(a)
    return xp.asarray(a, dtype=dtype)


def _symmetrize(xp, A):
    return 0.5 * (A + A.T)


@dataclass
class THCJKWork:
    """Reusable work config for THC J/K builds."""

    q_block: int = 256


@dataclass(frozen=True)
class THCPrecisionPolicy:
    """Precision policy for THC contractions.

    Notes
    -----
    - `compute_dtype` controls the dtype used for the hot GEMMs and temporaries.
    - `out_dtype` controls the returned J/K dtype (typically float64).
    - When `use_tf32=True` and `compute_dtype=float32` on GPU, we attempt to
      enable TF32 tensor-core GEMMs via CuPy's global compute-type setting.
    """

    compute_dtype: Any = np.float64
    out_dtype: Any = np.float64
    use_tf32: bool = False
    # When both Z and Y are available in fp64, default to the legacy Z path to
    # minimize numerical drift. If Z is None we will always use Y regardless of
    # this flag.
    prefer_Y: bool = False


def _resolve_policy(policy: THCPrecisionPolicy | None) -> THCPrecisionPolicy:
    return THCPrecisionPolicy() if policy is None else policy


@contextmanager
def _maybe_tf32_ctx(xp, *, enabled: bool):
    """Best-effort TF32 enablement for CuPy float32 GEMMs."""

    if not bool(enabled):
        yield
        return
    try:
        import cupy as cp  # type: ignore
    except Exception:  # pragma: no cover
        cp = None  # type: ignore
    if cp is None or xp is not cp:
        yield
        return
    try:
        import cupy._core._routines_linalg as rl  # type: ignore

        old = rl.get_compute_type(cp.float32)
        rl.set_compute_type(cp.float32, rl.COMPUTE_TYPE_TF32)
        try:
            yield
        finally:
            rl.set_compute_type(cp.float32, old)
    except Exception:  # pragma: no cover
        # If TF32 toggling fails (unsupported backend, etc.), run as-is.
        yield


def _as_dtype(xp, dt):
    if dt is None:
        return None
    try:
        return xp.dtype(dt)
    except Exception:  # pragma: no cover
        return dt


def thc_J(D, X, Z=None, *, Y=None, policy: THCPrecisionPolicy | None = None):
    """Compute Coulomb matrix J[D] via THC factors.

    Inputs
    - D: (nao,nao) density (symmetric)
    - X: (npt,nao) weighted AO collocation (points-major)
    - Z: optional (npt,npt) central metric
    - Y: optional (npt,naux) factor such that Z = Y @ Y.T. If provided and
      `policy.prefer_Y` is True, the central-metric action uses Y directly.
    """

    policy = _resolve_policy(policy)
    use_y = (Y is not None) and (Z is None or bool(policy.prefer_Y))
    metric = Y if use_y else Z
    if metric is None:
        raise ValueError("must provide Z or Y")

    xp, _is_gpu = _get_xp(D, X, metric)
    compute_dtype = _as_dtype(xp, getattr(policy, "compute_dtype", np.float64))
    out_dtype = _as_dtype(xp, getattr(policy, "out_dtype", np.float64))
    D = _as_xp(xp, D, dtype=compute_dtype)
    X = _as_xp(xp, X, dtype=compute_dtype)
    if use_y:
        Y = _as_xp(xp, Y, dtype=compute_dtype)
        Z = None
    else:
        Y = None
        Z = _as_xp(xp, Z, dtype=compute_dtype)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with D")
    if Y is not None:
        if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(npt):
            raise ValueError("Y must have shape (npt,naux)")
    else:
        if Z is None:
            raise ValueError("must provide Z or Y")
        if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
            raise ValueError("Z must have shape (npt,npt)")

    impl = _thc_jk_impl_pref()
    ext = (
        _load_hf_thc_cuda_ext()
        if _is_gpu
        and impl in {"auto", "ext"}
        and compute_dtype == _as_dtype(xp, xp.float64)
        and out_dtype == _as_dtype(xp, xp.float64)
        and Y is None
        else None
    )

    with _maybe_tf32_ctx(xp, enabled=bool(getattr(policy, "use_tf32", False)) and compute_dtype == _as_dtype(xp, xp.float32)):
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

        # n = Z @ m; prefer Y when available.
        if Y is not None:
            n = Y @ (Y.T @ m)
        else:
            assert Z is not None
            n = Z @ m  # (npt,)

        if ext is not None:
            # Reuse A as the row-scaled buffer (A <- X * n[:,None]).
            ext.scale_rows_f64(X, n, A, threads=256, stream_ptr=stream_ptr, sync=False)
            J = X.T @ A
        else:
            # J = X^T (n*X)
            J = X.T @ (X * n[:, None])
        J = _as_xp(xp, J, dtype=out_dtype)
        return _symmetrize(xp, J)


def thc_J_factored(U, V, X, Z=None, *, Y=None, policy: THCPrecisionPolicy | None = None):
    """Compute Coulomb matrix J[D] via THC factors with a factored density D = U V^T.

    This avoids forming the dense AO density when its rank is small.

    Inputs
    - U: (nao,r) left factors
    - V: (nao,r) right factors
    - X: (npt,nao) weighted AO collocation (points-major)
    - Z: optional (npt,npt) central metric
    - Y: optional (npt,naux) factor such that Z = Y @ Y.T; if provided and
      `policy.prefer_Y` is True we use it for n = Z@m.
    """

    policy = _resolve_policy(policy)
    use_y = (Y is not None) and (Z is None or bool(policy.prefer_Y))
    metric = Y if use_y else Z
    if metric is None:
        raise ValueError("must provide Z or Y")

    xp, _is_gpu = _get_xp(U, V, X, metric)
    compute_dtype = _as_dtype(xp, getattr(policy, "compute_dtype", np.float64))
    out_dtype = _as_dtype(xp, getattr(policy, "out_dtype", np.float64))
    U = _as_xp(xp, U, dtype=compute_dtype)
    V = _as_xp(xp, V, dtype=compute_dtype)
    X = _as_xp(xp, X, dtype=compute_dtype)
    if use_y:
        Y = _as_xp(xp, Y, dtype=compute_dtype)
        Z = None
    else:
        Y = None
        Z = _as_xp(xp, Z, dtype=compute_dtype)

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U and V must have the same shape (nao,r)")
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with U/V")
    if Y is not None:
        if Y.ndim != 2 or int(Y.shape[0]) != int(npt):
            raise ValueError("Y must have shape (npt,naux)")
    else:
        if Z is None:
            raise ValueError("must provide Z or Y")
        if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
            raise ValueError("Z must have shape (npt,npt)")

    with _maybe_tf32_ctx(xp, enabled=bool(getattr(policy, "use_tf32", False)) and compute_dtype == _as_dtype(xp, xp.float32)):
        # P = XU, Q = XV  (npt,r)
        P = X @ U
        Q = X @ V

        # m[P] = sum_k P[P,k] Q[P,k] = X[P,:] D X[P,:]^T
        m = xp.sum(P * Q, axis=1)  # (npt,)

        # n = Z @ m; use factor Y when available (n = Y (Y^T m)).
        if Y is not None:
            n = Y @ (Y.T @ m)
        else:
            assert Z is not None
            n = Z @ m

        # J = X^T (n*X)
        J = X.T @ (X * n[:, None])
        J = _as_xp(xp, J, dtype=out_dtype)
        return _symmetrize(xp, J)


def thc_K_blocked_factored(
    U,
    V,
    X,
    Z=None,
    *,
    q_block: int = 256,
    Y=None,
    policy: THCPrecisionPolicy | None = None,
):
    """Compute exchange matrix K[D] via THC factors with a factored density D = U V^T.

    Uses the identity:
      M = X D X^T = (XU) (XV)^T

    so the expensive 'D @ Xq.T' multiply in the dense algorithm disappears.
    """

    policy = _resolve_policy(policy)
    use_y = (Y is not None) and (Z is None or bool(policy.prefer_Y))
    metric = Y if use_y else Z
    if metric is None:
        raise ValueError("must provide Z or Y")

    xp, _is_gpu = _get_xp(U, V, X, metric)
    compute_dtype = _as_dtype(xp, getattr(policy, "compute_dtype", np.float64))
    out_dtype = _as_dtype(xp, getattr(policy, "out_dtype", np.float64))
    U = _as_xp(xp, U, dtype=compute_dtype)
    V = _as_xp(xp, V, dtype=compute_dtype)
    X = _as_xp(xp, X, dtype=compute_dtype)
    if use_y:
        Y = _as_xp(xp, Y, dtype=compute_dtype)
        Z = None
    else:
        Y = None
        Z = _as_xp(xp, Z, dtype=compute_dtype)

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U and V must have the same shape (nao,r)")
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with U/V")
    if Y is not None:
        if Y.ndim != 2 or int(Y.shape[0]) != int(npt):
            raise ValueError("Y must have shape (npt,naux)")
    else:
        if Z is None:
            raise ValueError("must provide Z or Y")
        if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
            raise ValueError("Z must have shape (npt,npt)")

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")
    q_block = min(int(q_block), int(npt))

    with _maybe_tf32_ctx(xp, enabled=bool(getattr(policy, "use_tf32", False)) and compute_dtype == _as_dtype(xp, xp.float32)):
        # Project the density factors once.
        P = X @ U  # (npt,r)
        Q = X @ V  # (npt,r)

        K = xp.zeros((nao, nao), dtype=compute_dtype)
        XT = X.T

        for q0 in range(0, int(npt), int(q_block)):
            q1 = min(int(npt), int(q0) + int(q_block))
            nb = int(q1 - q0)
            if nb <= 0:
                continue

            Xq = X[int(q0) : int(q1), :]  # (nb,nao)
            Qq = Q[int(q0) : int(q1), :]  # (nb,r)

            # Mq = (X D X^T)[:, q0:q1] = P @ Qq^T   (npt,nb)
            Mq = P @ Qq.T

            # Zblk = Z[:, q0:q1]  (npt,nb). If only Y is available, build on-the-fly.
            if Y is not None:
                Yq = Y[int(q0) : int(q1), :]  # (nb,naux)
                Zblk = Y @ Yq.T  # (npt,nb)
            else:
                assert Z is not None
                Zblk = Z[:, int(q0) : int(q1)]

            # Tq = Zblk ⊙ Mq
            Tq = Zblk * Mq

            # Fq = X^T @ Tq   (nao,nb)
            Fq = XT @ Tq

            # K += Fq @ Xq
            K += Fq @ Xq

            del Xq, Qq, Mq, Zblk, Tq, Fq

        K = _as_xp(xp, K, dtype=out_dtype)
        return _symmetrize(xp, K)


def thc_JK_factored(U, V, X, Z=None, *, work: THCJKWork | None = None, Y=None, policy: THCPrecisionPolicy | None = None):
    """Compute (J,K) via THC factors with a factored density D = U V^T.

    This shares the projected point-space factors P = XU and Q = XV between
    the J and K builds.
    """

    work = THCJKWork() if work is None else work

    policy = _resolve_policy(policy)
    use_y = (Y is not None) and (Z is None or bool(policy.prefer_Y))
    metric = Y if use_y else Z
    if metric is None:
        raise ValueError("must provide Z or Y")

    xp, _is_gpu = _get_xp(U, V, X, metric)
    compute_dtype = _as_dtype(xp, getattr(policy, "compute_dtype", np.float64))
    out_dtype = _as_dtype(xp, getattr(policy, "out_dtype", np.float64))
    U = _as_xp(xp, U, dtype=compute_dtype)
    V = _as_xp(xp, V, dtype=compute_dtype)
    X = _as_xp(xp, X, dtype=compute_dtype)
    if use_y:
        Y = _as_xp(xp, Y, dtype=compute_dtype)
        Z = None
    else:
        Y = None
        Z = _as_xp(xp, Z, dtype=compute_dtype)

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U and V must have the same shape (nao,r)")
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with U/V")
    if Y is not None:
        if Y.ndim != 2 or int(Y.shape[0]) != int(npt):
            raise ValueError("Y must have shape (npt,naux)")
    else:
        if Z is None:
            raise ValueError("must provide Z or Y")
        if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
            raise ValueError("Z must have shape (npt,npt)")

    with _maybe_tf32_ctx(xp, enabled=bool(getattr(policy, "use_tf32", False)) and compute_dtype == _as_dtype(xp, xp.float32)):
        # Project the density factors once.
        P = X @ U  # (npt,r)
        Q = X @ V  # (npt,r)

        # J: m = sum_k P_k Q_k, n = Z@m, J = X^T (n*X)
        m = xp.sum(P * Q, axis=1)  # (npt,)
        if Y is not None:
            n = Y @ (Y.T @ m)
        else:
            assert Z is not None
            n = Z @ m
        J = X.T @ (X * n[:, None])

        # K: blocked build from M = P Q^T
        q_block = int(getattr(work, "q_block", 256))
        if q_block <= 0:
            raise ValueError("work.q_block must be > 0")
        q_block = min(int(q_block), int(npt))

        K = xp.zeros((nao, nao), dtype=compute_dtype)
        XT = X.T
        for q0 in range(0, int(npt), int(q_block)):
            q1 = min(int(npt), int(q0) + int(q_block))
            nb = int(q1 - q0)
            if nb <= 0:
                continue

            Xq = X[int(q0) : int(q1), :]  # (nb,nao)
            Qq = Q[int(q0) : int(q1), :]  # (nb,r)

            Mq = P @ Qq.T  # (npt,nb)
            if Y is not None:
                Yq = Y[int(q0) : int(q1), :]  # (nb,naux)
                Zblk = Y @ Yq.T  # (npt,nb)
            else:
                assert Z is not None
                Zblk = Z[:, int(q0) : int(q1)]
            Tq = Zblk * Mq
            Fq = XT @ Tq
            K += Fq @ Xq

            del Xq, Qq, Mq, Zblk, Tq, Fq

        J = _as_xp(xp, J, dtype=out_dtype)
        K = _as_xp(xp, K, dtype=out_dtype)
        return _symmetrize(xp, J), _symmetrize(xp, K)


def thc_K_blocked(D, X, Z=None, *, q_block: int = 256, Y=None, policy: THCPrecisionPolicy | None = None):
    """Compute exchange matrix K[D] via THC factors in blocks over point index.

    Avoids allocating M = X D X^T as a full (npt,npt) temporary.
    """

    policy = _resolve_policy(policy)
    use_y = (Y is not None) and (Z is None or bool(policy.prefer_Y))
    metric = Y if use_y else Z
    if metric is None:
        raise ValueError("must provide Z or Y")

    xp, _is_gpu = _get_xp(D, X, metric)
    compute_dtype = _as_dtype(xp, getattr(policy, "compute_dtype", np.float64))
    out_dtype = _as_dtype(xp, getattr(policy, "out_dtype", np.float64))
    D = _as_xp(xp, D, dtype=compute_dtype)
    X = _as_xp(xp, X, dtype=compute_dtype)
    if use_y:
        Y = _as_xp(xp, Y, dtype=compute_dtype)
        Z = None
    else:
        Y = None
        Z = _as_xp(xp, Z, dtype=compute_dtype)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao_x = map(int, X.shape)
    if nao_x != nao:
        raise ValueError("X nao mismatch with D")
    if Y is not None:
        if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(npt):
            raise ValueError("Y must have shape (npt,naux)")
    else:
        if Z is None:
            raise ValueError("must provide Z or Y")
        if Z.ndim != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
            raise ValueError("Z must have shape (npt,npt)")

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")
    q_block = min(int(q_block), int(npt))

    impl = _thc_jk_impl_pref()
    ext = (
        _load_hf_thc_cuda_ext()
        if _is_gpu
        and impl in {"auto", "ext"}
        and compute_dtype == _as_dtype(xp, xp.float64)
        and out_dtype == _as_dtype(xp, xp.float64)
        and Y is None
        else None
    )
    if ext is not None:
        try:
            stream_ptr = int(xp.cuda.get_current_stream().ptr)
        except Exception:  # pragma: no cover
            stream_ptr = 0

    with _maybe_tf32_ctx(xp, enabled=bool(getattr(policy, "use_tf32", False)) and compute_dtype == _as_dtype(xp, xp.float32)):
        K = xp.zeros((nao, nao), dtype=compute_dtype)

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
                assert Z is not None
                Zblk = Z[:, int(q0) : int(q1)]
                ext.hadamard_inplace_f64(Mq, Zblk, threads=256, stream_ptr=stream_ptr, sync=False)
                Fq = XT @ Mq
                del Zblk
            else:
                if Y is not None:
                    Yq = Y[int(q0) : int(q1), :]  # (nb,naux)
                    Zblk = Y @ Yq.T  # (npt,nb)
                else:
                    assert Z is not None
                    Zblk = Z[:, int(q0) : int(q1)]
                # Tq = Zblk ⊙ Mq  (npt,nb)
                Tq = Zblk * Mq
                # Fq = X^T @ Tq   (nao,nb)
                Fq = XT @ Tq
                del Zblk, Tq

            # Accumulate: K += Fq @ Xq
            K += Fq @ Xq

            del Xq, Aq, Mq, Fq

        K = _as_xp(xp, K, dtype=out_dtype)
        return _symmetrize(xp, K)


def thc_JK(D, X, Z=None, *, work: THCJKWork | None = None, Y=None, policy: THCPrecisionPolicy | None = None):
    work = THCJKWork() if work is None else work
    J = thc_J(D, X, Z, Y=Y, policy=policy)
    K = thc_K_blocked(D, X, Z, q_block=int(work.q_block), Y=Y, policy=policy)
    return J, K


__all__ = [
    "THCJKWork",
    "thc_J",
    "thc_J_factored",
    "thc_JK",
    "thc_JK_factored",
    "thc_K_blocked",
    "thc_K_blocked_factored",
]
