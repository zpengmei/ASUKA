from __future__ import annotations

"""Adjoints for DF whitening and the aux-metric Cholesky.

This module provides the linear-algebra backprop pieces needed to replace the
finite-difference DF nuclear-gradient contraction on whitened DF factors `B`.

Definitions (ASUKA/cuERI convention)
-----------------------------------
Let:
  - X[μ,ν,Q]  be the *unwhitened* 3-center Coulomb integrals (μν|Q)
  - V[P,Q]    be the aux Coulomb metric (P|Q)
  - L         be the lower Cholesky factor: V = L L^T

Whitening:
  - reshape X to X_flat[μν, Q]
  - compute B^T = L^{-1} X^T  (triangular solve; L lower)
  - B is the whitened DF factor used throughout ASUKA DF-J/K.

We need:
  - df_whiten_adjoint: (B, bar_B, L) -> (bar_X, bar_L)
  - chol_lower_adjoint: (L, bar_L) -> bar_V
"""

from typing import Any

import os
import numpy as np


def _get_xp(*arrays: Any):
    try:  # pragma: no cover
        import cupy as cp  # type: ignore
    except Exception:
        return np
    for a in arrays:
        if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
            return cp
    return np


def _as_xp_f64(xp, a: Any):
    return xp.asarray(a, dtype=xp.float64)


def _solve_triangular(xp, L, B, *, lower: bool, trans: str, overwrite_b: bool = False):
    from asuka.linalg.triangular import solve_triangular  # noqa: PLC0415

    return solve_triangular(L, B, lower=bool(lower), trans=str(trans), overwrite_b=bool(overwrite_b))


def _solve_triangular_qp_inplace_cupy(L: Any, B_qp: Any) -> Any:
    """In-place packed-Qp triangular solve on CUDA.

    Compute ``X = L^{-T} B`` for ``B`` shaped ``(naux, ntri)`` without allocating
    another full-sized RHS buffer. Uses a row-major/column-major reinterpretation:

    - row-major ``B_rm(naux,ntri)``
    - same memory as column-major ``B_cm(ntri,naux) = B_rm^T``
    - solve on the right: ``X_cm = B_cm * L^{-1}``
    - reinterpret back: ``X_rm = X_cm^T = L^{-T} B_rm``
    """

    import cupy as cp  # noqa: PLC0415
    from cupy.cuda import cublas  # noqa: PLC0415

    B = cp.asarray(B_qp, dtype=cp.float64)
    if not bool(getattr(B, "flags", None).c_contiguous):
        B = cp.ascontiguousarray(B, dtype=cp.float64)
    Lf = cp.asfortranarray(cp.asarray(L, dtype=cp.float64))

    if int(B.ndim) != 2:
        raise ValueError("B_qp must have shape (naux, ntri)")
    naux, ntri = map(int, B.shape)
    if int(Lf.ndim) != 2 or int(Lf.shape[0]) != int(Lf.shape[1]) or int(Lf.shape[0]) != int(naux):
        raise ValueError("L must have shape (naux, naux)")

    handle = cp.cuda.get_cublas_handle()
    prev_stream = cublas.getStream(handle)
    prev_mode = cublas.getPointerMode(handle)
    try:
        cublas.setStream(handle, int(cp.cuda.get_current_stream().ptr))
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        alpha = np.array(1.0, dtype=np.float64)
        cublas.dtrsm(
            handle,
            cublas.CUBLAS_SIDE_RIGHT,
            cublas.CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_N,
            cublas.CUBLAS_DIAG_NON_UNIT,
            int(ntri),               # m (rows of column-major view B_cm)
            int(naux),               # n (cols of B_cm)
            int(alpha.ctypes.data),  # host pointer to alpha
            int(Lf.data.ptr),
            int(Lf.shape[0]),
            int(B.data.ptr),
            int(ntri),               # ldb for B_cm
        )
    finally:
        try:
            cublas.setStream(handle, int(prev_stream))
        except Exception:
            pass
        try:
            cublas.setPointerMode(handle, int(prev_mode))
        except Exception:
            pass

    return B


def df_whiten_adjoint(
    B: Any,
    bar_B: Any,
    L: Any,
) -> tuple[Any, Any]:
    """Adjoint of DF whitening B = whiten_3c2e(X, L) with respect to (X, L).

    Parameters
    ----------
    B
        Whitened DF factors, shape (nao, nao, naux).
    bar_B
        Adjoint w.r.t. B (same shape as B).
    L
        Lower Cholesky factor of the aux metric, shape (naux, naux).

    Returns
    -------
    bar_X
        Adjoint w.r.t. unwhitened 3c2e X, shape (nao, nao, naux).
    bar_L
        Adjoint w.r.t. the Cholesky factor L, shape (naux, naux) (lower triangle used).
    """

    xp = _get_xp(B, bar_B, L)
    B = _as_xp_f64(xp, B)
    bar_B = _as_xp_f64(xp, bar_B)
    L = _as_xp_f64(xp, L)

    if B.ndim != 3:
        raise ValueError("B must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, B.shape)
    if nao0 != nao1:
        raise ValueError("B must have shape (nao, nao, naux)")
    if bar_B.shape != B.shape:
        raise ValueError("bar_B shape mismatch")
    if L.ndim != 2 or int(L.shape[0]) != int(L.shape[1]) or int(L.shape[0]) != int(naux):
        raise ValueError("L must have shape (naux, naux)")

    # Forward relation (cuERI): B^T = L^{-1} X^T, with X_flat shape (nao^2, naux).
    # Here we need the adjoint w.r.t. X and L given bar_B.
    B_flat = B.reshape((nao0 * nao0, naux))
    bar_B_flat = bar_B.reshape((nao0 * nao0, naux))

    # bar_BT = bar_B^T because B = BT^T.
    bar_BT = bar_B_flat.T  # (naux, nao^2)

    # tmp = L^{-T} bar_BT
    tmp = _solve_triangular(xp, L, bar_BT, lower=True, trans="T")  # (naux, nao^2)

    # bar_X^T = tmp  => bar_X = tmp^T
    bar_X_flat = tmp.T  # (nao^2, naux)

    # bar_L = - tmp @ BT^T = - tmp @ B_flat
    bar_L = -(tmp @ B_flat)  # (naux, naux)
    bar_L = xp.tril(bar_L)

    return bar_X_flat.reshape(B.shape), bar_L


def df_whiten_adjoint_Qmn(
    B_mnQ: Any,
    bar_L_Qmn: Any,
    L: Any,
    *,
    overwrite_bar_L: bool = False,
) -> tuple[Any, Any]:
    """Like :func:`df_whiten_adjoint` but *bar_L* is in ``(naux, nao, nao)`` layout.

    This avoids the expensive ``cp.ascontiguousarray(bar_L.transpose((1,2,0)))``
    copy that the callers would otherwise need: reshaping a C-contiguous
    ``(naux, nao, nao)`` array to ``(naux, nao*nao)`` is free.

    Parameters
    ----------
    B_mnQ
        Whitened DF factors, shape ``(nao, nao, naux)``.
    bar_L_Qmn
        Adjoint w.r.t. B in ``(naux, nao, nao)`` layout (Fortran-like Q-first).
    L
        Lower Cholesky factor of the aux metric, shape ``(naux, naux)``.
    overwrite_bar_L
        If True, the solve overwrites ``bar_L_Qmn``'s buffer in-place
        (via cuBLAS TRSM ``overwrite_b``), saving one ``(naux, nao^2)``
        allocation.  Callers must not use ``bar_L_Qmn`` after this call.

    Returns
    -------
    bar_X
        Adjoint w.r.t. unwhitened 3c2e X, shape ``(nao, nao, naux)``.
    bar_Lchol
        Adjoint w.r.t. the Cholesky factor L, shape ``(naux, naux)``.
    """

    xp = _get_xp(B_mnQ, bar_L_Qmn, L)
    B_mnQ = _as_xp_f64(xp, B_mnQ)
    bar_L_Qmn = _as_xp_f64(xp, bar_L_Qmn)
    L = _as_xp_f64(xp, L)

    if B_mnQ.ndim != 3:
        raise ValueError("B_mnQ must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, B_mnQ.shape)
    if nao0 != nao1:
        raise ValueError("B_mnQ must have shape (nao, nao, naux)")
    if bar_L_Qmn.ndim != 3 or int(bar_L_Qmn.shape[0]) != naux or int(bar_L_Qmn.shape[1]) != nao0 or int(bar_L_Qmn.shape[2]) != nao0:
        raise ValueError("bar_L_Qmn must have shape (naux, nao, nao)")
    if L.ndim != 2 or int(L.shape[0]) != naux or int(L.shape[1]) != naux:
        raise ValueError("L must have shape (naux, naux)")

    B_flat = B_mnQ.reshape(nao0 * nao0, naux)
    # Key optimisation: (naux, nao, nao) -> (naux, nao*nao) is a free reshape
    # because bar_L_Qmn is C-contiguous.  This is the bar_BT that
    # df_whiten_adjoint would compute as bar_B.reshape(nao^2,naux).T after a
    # costly transpose copy.
    bar_BT = bar_L_Qmn.reshape(naux, nao0 * nao0)

    # When overwrite_bar_L=True, TRSM writes the solution directly into
    # bar_BT's buffer (= bar_L_Qmn's memory), avoiding a full-size temporary.
    tmp = _solve_triangular(xp, L, bar_BT, lower=True, trans="T", overwrite_b=overwrite_bar_L)  # (naux, nao^2)
    bar_X_flat = tmp.T  # (nao^2, naux) — non-contiguous view, that's fine
    bar_Lchol = -(tmp @ B_flat)  # (naux, naux)
    bar_Lchol = xp.tril(bar_Lchol)

    return bar_X_flat.reshape(B_mnQ.shape), bar_Lchol


def df_whiten_adjoint_Qp(
    B_Qp: Any,
    bar_L_Qp: Any,
    L: Any,
    *,
    nao: int,
    overwrite_bar_L: bool = False,
    p_block: int | None = None,
) -> tuple[Any, Any]:
    """Adjoint of DF whitening for packed AO-pair ("s2") storage.

    This is analogous to :func:`df_whiten_adjoint_Qmn`, but with both B and the
    adjoint bar_L stored in packed Qp layout: shape ``(naux, ntri)`` where
    ``ntri=nao*(nao+1)//2``.

    Returns
    -------
    bar_X_Qp
        Adjoint w.r.t. unwhitened 3c2e X, in packed Qp layout (naux,ntri).
        This is the packed analogue of the Qmn buffer that
        :func:`df_whiten_adjoint_Qmn` overwrites in-place.
    bar_Lchol
        Adjoint w.r.t. the Cholesky factor L, shape (naux,naux) lower triangle.
    """

    from asuka.integrals.tri_packed import ntri_from_nao, tri_weights  # noqa: PLC0415

    xp = _get_xp(B_Qp, bar_L_Qp, L)
    B_Qp = _as_xp_f64(xp, B_Qp)
    bar_L_Qp = _as_xp_f64(xp, bar_L_Qp)
    L = _as_xp_f64(xp, L)

    if B_Qp.ndim != 2:
        raise ValueError("B_Qp must have shape (naux, ntri)")
    naux, ntri = map(int, B_Qp.shape)
    if bar_L_Qp.shape != (naux, ntri):
        raise ValueError("bar_L_Qp shape mismatch")
    if L.ndim != 2 or int(L.shape[0]) != int(naux) or int(L.shape[1]) != int(naux):
        raise ValueError("L must have shape (naux, naux)")

    nao_i = int(nao)
    expected_ntri = int(ntri_from_nao(nao_i))
    if int(ntri) != int(expected_ntri):
        raise ValueError(f"B_Qp has ntri={int(ntri)} but expected {int(expected_ntri)} for nao={int(nao_i)}")

    # Triangular solve in aux space: tmp = L^{-T} bar_BT, with bar_BT == bar_L_Qp.
    # For CUDA+overwrite, use an in-place TRSM variant specialized for row-major
    # packed Qp RHS to avoid a full-size temporary allocation.
    if xp is not np and bool(overwrite_bar_L):
        tmp = _solve_triangular_qp_inplace_cupy(L, bar_L_Qp)
    else:
        tmp = _solve_triangular(xp, L, bar_L_Qp, lower=True, trans="T", overwrite_b=overwrite_bar_L)  # (naux,ntri)

    # bar_Lchol = -(tmp @ B_flat) in full layout.
    # For packed symmetry, the ordered-pair sum becomes a packed sum with weights:
    #   bar_Lchol[i,j] = -sum_p w[p] * tmp[i,p] * B[j,p]
    w = tri_weights(xp, nao_i, dtype=xp.float64)  # (ntri,)

    if p_block is None:
        try:
            p_block = int(str(os.environ.get("ASUKA_DF_QP_MM_PBLOCK", "")).strip() or "0")
        except Exception:
            p_block = 0
        if int(p_block) <= 0:
            # Default: keep (naux,p_block) work buffers reasonable.
            p_block = 8192
    p_block = int(max(1, min(int(ntri), int(p_block))))

    bar_Lchol = xp.zeros((naux, naux), dtype=xp.float64)
    work = xp.empty((naux, p_block), dtype=xp.float64)
    for p0 in range(0, int(ntri), int(p_block)):
        p1 = min(int(ntri), int(p0) + int(p_block))
        pb = int(p1 - p0)
        tmp_blk = tmp[:, int(p0) : int(p1)]
        B_blk = B_Qp[:, int(p0) : int(p1)]
        w_blk = w[int(p0) : int(p1)]

        xp.multiply(tmp_blk, w_blk[None, :], out=work[:, :pb])
        bar_Lchol -= work[:, :pb] @ B_blk.T

    bar_Lchol = xp.tril(bar_Lchol)
    return tmp, bar_Lchol


def chol_lower_adjoint(L: Any, bar_L: Any) -> Any:
    """Adjoint of lower-triangular Cholesky: V = L L^T.

    Parameters
    ----------
    L
        Lower Cholesky factor, shape (n,n).
    bar_L
        Adjoint w.r.t. L, shape (n,n). Only the lower triangle is used.

    Returns
    -------
    bar_V
        Symmetric adjoint w.r.t. V, shape (n,n).
    """

    xp = _get_xp(L, bar_L)
    L = _as_xp_f64(xp, L)
    bar_L = _as_xp_f64(xp, bar_L)

    if L.ndim != 2 or int(L.shape[0]) != int(L.shape[1]):
        raise ValueError("L must be a square 2D array")
    if bar_L.shape != L.shape:
        raise ValueError("bar_L shape mismatch")

    n = int(L.shape[0])
    bar_L = xp.tril(bar_L)

    # Backprop through the differential:
    #   Phi = tril(S); diag(Phi) = 0.5*diag(S)
    #   dL = Phi @ L
    G = L.T @ bar_L
    G = xp.tril(G)
    idx = xp.arange(n)
    G[idx, idx] *= 0.5

    # bar_V = sym( L^{-T} G L^{-1} )
    tmp = _solve_triangular(xp, L, G, lower=True, trans="T")
    bar_V = _solve_triangular(xp, L, tmp.T, lower=True, trans="T").T
    bar_V = 0.5 * (bar_V + bar_V.T)
    return bar_V


__all__ = [
    "chol_lower_adjoint",
    "df_whiten_adjoint",
    "df_whiten_adjoint_Qmn",
]
