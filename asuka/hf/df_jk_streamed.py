from __future__ import annotations

"""Streamed DF J/K builders (no materialized B).

This module evaluates 3c2e DF integrals in aux-function blocks and contracts
them into J/K without storing the full whitened DF tensor B(μ,ν,Q).

Intended use:
- SCF where B(nao,nao,naux) does not fit in GPU memory.
- Exchange still uses an occupied-orbital (MO-driven) formulation.

Implementation details
- Stream unwhitened 3c2e integrals to avoid block-wise whitening artifacts:
  integrals X(μ,ν,P) and applying the metric Cholesky via triangular solves:
    B^T = L^{-1} X^T.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .df_jk import _cublas_math_mode_ctx


def _symmetrize(xp, A):
    return 0.5 * (A + A.T)


def _naux_from_aux_basis(aux_basis) -> int:
    if not hasattr(aux_basis, "shell_ao_start") or not hasattr(aux_basis, "shell_l"):
        raise TypeError("aux_basis must provide shell_ao_start and shell_l")
    from asuka.cueri.cart import ncart  # local import to keep module light

    shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int64).ravel()
    shell_l = np.asarray(aux_basis.shell_l, dtype=np.int64).ravel()
    if shell_ao_start.shape != shell_l.shape:
        raise ValueError("aux_basis.shell_ao_start and shell_l must have identical shape")
    if shell_l.size == 0:
        return 0
    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    return int(np.max(shell_ao_start + nfunc))


@dataclass
class StreamedDFJKContext:
    """Reusable streamed DF-J/K context for SCF iterations."""

    ao_basis: Any
    aux_basis: Any
    L_metric: Any  # cp.ndarray (naux,naux)
    aux_blocks: list[tuple[int, int, int, int]]  # (shell_start, shell_stop, p0, p1)
    backend: str
    threads: int
    mode: str


def make_streamed_df_jk_context(
    ao_basis: Any,
    aux_basis: Any,
    *,
    L_metric=None,
    backend: str = "gpu_rys",
    threads: int = 256,
    mode: str = "auto",
    aux_block_naux: int = 256,
    profile: dict | None = None,
) -> StreamedDFJKContext:
    """Prepare a streamed DF-J/K context (metric Cholesky + aux-block plan)."""

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for streamed DF-J/K") from e

    from asuka.cueri import df as cueri_df  # noqa: PLC0415

    backend_s = str(backend).lower().strip()
    if backend_s not in ("gpu_ss", "gpu_sp", "gpu_rys"):
        raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")

    aux_block_naux = int(aux_block_naux)
    if aux_block_naux <= 0:
        raise ValueError("aux_block_naux must be > 0")
    aux_blocks = cueri_df.plan_aux_blocks_cart(aux_basis, max_block_naux=aux_block_naux)

    if L_metric is None:
        # Cached helper exists in cueri.df, but compute explicitly here so we can
        # respect the caller's threads/mode knobs (and keep the dependency local).
        V = cueri_df.metric_2c2e_basis(aux_basis, backend=backend_s, threads=int(threads), mode=str(mode))
        L = cueri_df.cholesky_metric(V)
    else:
        L = cp.asarray(L_metric, dtype=cp.float64)

    if L.ndim != 2 or int(L.shape[0]) != int(L.shape[1]):
        raise ValueError("L_metric must be a square 2D array")

    if profile is not None:
        prof = profile.setdefault("jk_streamed", {})
        prof["aux_block_naux"] = int(aux_block_naux)
        prof["backend"] = str(backend_s)
        prof["threads"] = int(threads)
        prof["mode"] = str(mode)
        prof["naux"] = int(L.shape[0])

    return StreamedDFJKContext(
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        L_metric=L,
        aux_blocks=aux_blocks,
        backend=str(backend_s),
        threads=int(threads),
        mode=str(mode),
    )


def _int3c2e_block(
    ctx: StreamedDFJKContext,
    *,
    shell_start: int,
    shell_stop: int,
):
    """Evaluate X(μ,ν,P) for an aux-shell block; returns (nao,nao,q) float64."""

    backend = str(ctx.backend).lower().strip()
    if backend in ("gpu_rys", "gpu_sp"):
        from asuka.cueri.gpu import df_int3c2e_rys_device_block  # noqa: PLC0415

        return df_int3c2e_rys_device_block(
            ctx.ao_basis,
            ctx.aux_basis,
            aux_shell_start=int(shell_start),
            aux_shell_stop=int(shell_stop),
            threads=int(ctx.threads),
            mode=str(ctx.mode),
            stream=None,
            profile=None,
        )

    # Fallback: use the generic slice API (may be slower and may materialize internally).
    from asuka.cueri import df as cueri_df  # noqa: PLC0415

    # Convert shell indices to aux-function indices using the planned blocks.
    # (We only call this fallback for correctness; performance tuning should use gpu_rys.)
    shell_ao_start = np.asarray(ctx.aux_basis.shell_ao_start, dtype=np.int64).ravel()
    from asuka.cueri.cart import ncart  # noqa: PLC0415

    shell_l = np.asarray(ctx.aux_basis.shell_l, dtype=np.int64).ravel()
    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    shell_ao_end = shell_ao_start + nfunc
    p0 = int(shell_ao_start[int(shell_start)])
    p1 = int(shell_ao_end[int(shell_stop) - 1])
    return cueri_df.int3c2e_block(ctx.ao_basis, ctx.aux_basis, p0, p1, backend=backend, stream=None)


def df_JK_streamed(
    ctx: StreamedDFJKContext,
    D,
    C_occ,
    occ_vals,
    *,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    work: dict | None = None,
    profile: dict | None = None,
):
    """Compute (J,K) for a single RHF-like occupied set using streamed DF.

    Inputs
    - ctx: StreamedDFJKContext with L_metric and aux_blocks
    - D: (nao,nao) AO density (float64, CuPy)
    - C_occ: (nao,nocc) occupied orbitals (float64, CuPy)
    - occ_vals: (nocc,) occupancies (float64, CuPy/NumPy)

    Returns
    - J, K: (nao,nao) float64, symmetric
    """

    try:
        import cupy as cp  # noqa: PLC0415
        import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for streamed DF-J/K") from e

    D = cp.asarray(D, dtype=cp.float64)
    C_occ = cp.asarray(C_occ, dtype=cp.float64)
    occ_vals = cp.asarray(occ_vals, dtype=cp.float64).ravel()

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if C_occ.ndim != 2 or int(C_occ.shape[0]) != int(nao):
        raise ValueError("C_occ must have shape (nao, nocc)")
    nocc = int(C_occ.shape[1])
    if int(occ_vals.shape[0]) != int(nocc):
        raise ValueError("occ_vals must have shape (nocc,)")

    naux = int(ctx.L_metric.shape[0])
    if naux <= 0:
        Z = cp.zeros((nao, nao), dtype=cp.float64)
        return Z, Z

    k_q_block = int(k_q_block)
    if k_q_block <= 0:
        raise ValueError("k_q_block must be > 0")

    # Weighted occupied orbitals so K becomes V^T @ V where V stacks (Q,i) rows.
    sqrt_occ = cp.sqrt(occ_vals)
    Cw = C_occ * sqrt_occ[None, :]
    if hasattr(Cw, "flags") and not bool(Cw.flags.c_contiguous):
        Cw = cp.ascontiguousarray(Cw)

    # Work buffers (reused across iterations to reduce allocations).
    if work is None:
        work = {}
    y = work.get("y")
    if not isinstance(y, cp.ndarray) or tuple(y.shape) != (naux,) or y.dtype != cp.float64:
        y = cp.empty((naux,), dtype=cp.float64)
        work["y"] = y
    YT = work.get("YT")
    if not isinstance(YT, cp.ndarray) or tuple(YT.shape) != (naux, nocc * nao) or YT.dtype != cp.float64:
        YT = cp.empty((naux, nocc * nao), dtype=cp.float64)
        work["YT"] = YT

    # ---- Pass 1: build y = X^T @ Dvec and YT[P, i*nao+m] = Σ_n X[m,n,P] * Cw[n,i] ----
    Dvec = D.reshape((nao * nao,))
    t_int3c2e_1 = cp.cuda.Event() if profile is not None else None
    t_int3c2e_1_end = cp.cuda.Event() if profile is not None else None
    if t_int3c2e_1 is not None and t_int3c2e_1_end is not None:
        t_int3c2e_1.record(cp.cuda.get_current_stream())
    for shell_start, shell_stop, p0, p1 in ctx.aux_blocks:
        X_blk = _int3c2e_block(ctx, shell_start=int(shell_start), shell_stop=int(shell_stop))  # (nao,nao,q)
        q = int(p1 - p0)
        X2 = X_blk.reshape((nao * nao, q))
        y[int(p0) : int(p1)] = X2.T @ Dvec
        # (m,n,q) x (n,i) -> (q,i,m) so columns are i-major then m.
        tmp = cp.einsum("mnq,ni->qim", X_blk, Cw, optimize=True)
        YT[int(p0) : int(p1), :] = tmp.reshape((q, nocc * nao))
    if t_int3c2e_1 is not None and t_int3c2e_1_end is not None:
        t_int3c2e_1_end.record(cp.cuda.get_current_stream())
        t_int3c2e_1_end.synchronize()

    # ---- Metric application: d = L^{-1} y, w = L^{-T} d, and whiten YT in-place ----
    t_solve = cp.cuda.Event() if profile is not None else None
    t_solve_end = cp.cuda.Event() if profile is not None else None
    if t_solve is not None and t_solve_end is not None:
        t_solve.record(cp.cuda.get_current_stream())
    d = cpx_linalg.solve_triangular(ctx.L_metric, y, lower=True, trans="N", unit_diagonal=False, overwrite_b=False)
    w = cpx_linalg.solve_triangular(ctx.L_metric, d, lower=True, trans="T", unit_diagonal=False, overwrite_b=False)
    W = cpx_linalg.solve_triangular(ctx.L_metric, YT, lower=True, trans="N", unit_diagonal=False, overwrite_b=True)
    if t_solve is not None and t_solve_end is not None:
        t_solve_end.record(cp.cuda.get_current_stream())
        t_solve_end.synchronize()

    # ---- K: accumulate K = Σ_Q Σ_i (μi|Q)(νi|Q) as V^T @ V in Q-blocks ----
    K = cp.zeros((nao, nao), dtype=cp.float64)
    tK = cp.cuda.Event() if profile is not None else None
    tK_end = cp.cuda.Event() if profile is not None else None
    if tK is not None and tK_end is not None:
        tK.record(cp.cuda.get_current_stream())
    with _cublas_math_mode_ctx(cp, cublas_math_mode):
        for q0 in range(0, int(naux), int(k_q_block)):
            q1 = min(int(naux), int(q0) + int(k_q_block))
            Wblk = W[int(q0) : int(q1)]  # (q, nocc*nao)
            V = Wblk.reshape(((int(q1 - q0) * int(nocc)), int(nao)))  # (q*nocc, nao)
            K += V.T @ V
    if tK is not None and tK_end is not None:
        tK_end.record(cp.cuda.get_current_stream())
        tK_end.synchronize()

    # ---- Pass 2: J = X @ w (second streamed pass over X) ----
    Jvec = work.get("Jvec")
    if not isinstance(Jvec, cp.ndarray) or tuple(Jvec.shape) != (nao * nao,) or Jvec.dtype != cp.float64:
        Jvec = cp.zeros((nao * nao,), dtype=cp.float64)
        work["Jvec"] = Jvec
    else:
        Jvec.fill(0.0)

    t_int3c2e_2 = cp.cuda.Event() if profile is not None else None
    t_int3c2e_2_end = cp.cuda.Event() if profile is not None else None
    if t_int3c2e_2 is not None and t_int3c2e_2_end is not None:
        t_int3c2e_2.record(cp.cuda.get_current_stream())
    for shell_start, shell_stop, p0, p1 in ctx.aux_blocks:
        X_blk = _int3c2e_block(ctx, shell_start=int(shell_start), shell_stop=int(shell_stop))
        q = int(p1 - p0)
        X2 = X_blk.reshape((nao * nao, q))
        wblk = w[int(p0) : int(p1)]
        Jvec += X2 @ wblk
    if t_int3c2e_2 is not None and t_int3c2e_2_end is not None:
        t_int3c2e_2_end.record(cp.cuda.get_current_stream())
        t_int3c2e_2_end.synchronize()

    J = Jvec.reshape((nao, nao))

    if profile is not None:
        prof = profile.setdefault("jk_streamed", {})
        try:
            if t_int3c2e_1 is not None and t_int3c2e_1_end is not None:
                prof["pass1_int3c2e_ms"] = float(cp.cuda.get_elapsed_time(t_int3c2e_1, t_int3c2e_1_end))
            if t_solve is not None and t_solve_end is not None:
                prof["metric_solve_ms"] = float(cp.cuda.get_elapsed_time(t_solve, t_solve_end))
            if tK is not None and tK_end is not None:
                prof["k_contract_ms"] = float(cp.cuda.get_elapsed_time(tK, tK_end))
            if t_int3c2e_2 is not None and t_int3c2e_2_end is not None:
                prof["pass2_int3c2e_ms"] = float(cp.cuda.get_elapsed_time(t_int3c2e_2, t_int3c2e_2_end))
            prof["YT_nbytes"] = int(getattr(YT, "nbytes", 0))
            prof["nocc"] = int(nocc)
            prof["nao"] = int(nao)
            prof["naux"] = int(naux)
            prof["k_q_block"] = int(k_q_block)
        except Exception:
            pass

    return _symmetrize(cp, J), _symmetrize(cp, K)


def df_JKs_streamed(
    ctx: StreamedDFJKContext,
    D,
    C_occ_list: list,
    occ_vals_list: list,
    *,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    work: dict | None = None,
    profile: dict | None = None,
):
    """Compute J and multiple K matrices (one per occupied set) using streamed DF.

    This avoids duplicating the expensive streamed 3c2e evaluation when multiple
    exchange matrices are needed (e.g., UHF alpha/beta).

    Inputs
    - D: (nao,nao) AO density used for J (typically Dtot)
    - C_occ_list[j]: (nao, nocc_j)
    - occ_vals_list[j]: (nocc_j,)
    """

    try:
        import cupy as cp  # noqa: PLC0415
        import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for streamed DF-J/K") from e

    if len(C_occ_list) != len(occ_vals_list):
        raise ValueError("C_occ_list and occ_vals_list must have the same length")

    D = cp.asarray(D, dtype=cp.float64)
    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])

    # Build a single concatenated Cw = [Cw_0, Cw_1, ...] and track orbital slices.
    slices: list[tuple[int, int]] = []
    Cw_parts: list[Any] = []
    nocc_total = 0
    for C_occ, occ_vals in zip(C_occ_list, occ_vals_list, strict=True):
        C_occ = cp.asarray(C_occ, dtype=cp.float64)
        occ_vals = cp.asarray(occ_vals, dtype=cp.float64).ravel()
        if C_occ.ndim != 2 or int(C_occ.shape[0]) != int(nao):
            raise ValueError("each C_occ must have shape (nao, nocc)")
        nocc = int(C_occ.shape[1])
        if int(occ_vals.shape[0]) != int(nocc):
            raise ValueError("each occ_vals must have shape (nocc,)")
        i0 = int(nocc_total)
        i1 = int(nocc_total + nocc)
        slices.append((i0, i1))
        nocc_total = i1
        sqrt_occ = cp.sqrt(occ_vals)
        Cw = C_occ * sqrt_occ[None, :]
        if hasattr(Cw, "flags") and not bool(Cw.flags.c_contiguous):
            Cw = cp.ascontiguousarray(Cw)
        Cw_parts.append(Cw)

    if nocc_total <= 0:
        Z = cp.zeros((nao, nao), dtype=cp.float64)
        return Z, [Z.copy() for _ in range(len(C_occ_list))]

    Cw_all = cp.concatenate(Cw_parts, axis=1)
    if hasattr(Cw_all, "flags") and not bool(Cw_all.flags.c_contiguous):
        Cw_all = cp.ascontiguousarray(Cw_all)

    naux = int(ctx.L_metric.shape[0])
    if naux <= 0:
        Z = cp.zeros((nao, nao), dtype=cp.float64)
        return Z, [Z.copy() for _ in range(len(C_occ_list))]

    k_q_block = int(k_q_block)
    if k_q_block <= 0:
        raise ValueError("k_q_block must be > 0")

    if work is None:
        work = {}
    y = work.get("y")
    if not isinstance(y, cp.ndarray) or tuple(y.shape) != (naux,) or y.dtype != cp.float64:
        y = cp.empty((naux,), dtype=cp.float64)
        work["y"] = y
    YT = work.get("YT")
    if not isinstance(YT, cp.ndarray) or tuple(YT.shape) != (naux, nocc_total * nao) or YT.dtype != cp.float64:
        YT = cp.empty((naux, nocc_total * nao), dtype=cp.float64)
        work["YT"] = YT

    # ---- Pass 1 ----
    Dvec = D.reshape((nao * nao,))
    t1 = cp.cuda.Event() if profile is not None else None
    t1e = cp.cuda.Event() if profile is not None else None
    if t1 is not None and t1e is not None:
        t1.record(cp.cuda.get_current_stream())
    for shell_start, shell_stop, p0, p1 in ctx.aux_blocks:
        X_blk = _int3c2e_block(ctx, shell_start=int(shell_start), shell_stop=int(shell_stop))
        q = int(p1 - p0)
        X2 = X_blk.reshape((nao * nao, q))
        y[int(p0) : int(p1)] = X2.T @ Dvec
        tmp = cp.einsum("mnq,ni->qim", X_blk, Cw_all, optimize=True)  # (q, nocc_total, nao)
        YT[int(p0) : int(p1), :] = tmp.reshape((q, nocc_total * nao))
    if t1 is not None and t1e is not None:
        t1e.record(cp.cuda.get_current_stream())
        t1e.synchronize()

    # ---- Metric application ----
    ts = cp.cuda.Event() if profile is not None else None
    tse = cp.cuda.Event() if profile is not None else None
    if ts is not None and tse is not None:
        ts.record(cp.cuda.get_current_stream())
    d = cpx_linalg.solve_triangular(ctx.L_metric, y, lower=True, trans="N", unit_diagonal=False, overwrite_b=False)
    w = cpx_linalg.solve_triangular(ctx.L_metric, d, lower=True, trans="T", unit_diagonal=False, overwrite_b=False)
    W = cpx_linalg.solve_triangular(ctx.L_metric, YT, lower=True, trans="N", unit_diagonal=False, overwrite_b=True)
    if ts is not None and tse is not None:
        tse.record(cp.cuda.get_current_stream())
        tse.synchronize()

    # ---- K_i ----
    Ks = [cp.zeros((nao, nao), dtype=cp.float64) for _ in range(len(C_occ_list))]
    tK = cp.cuda.Event() if profile is not None else None
    tKe = cp.cuda.Event() if profile is not None else None
    if tK is not None and tKe is not None:
        tK.record(cp.cuda.get_current_stream())
    with _cublas_math_mode_ctx(cp, cublas_math_mode):
        for q0 in range(0, int(naux), int(k_q_block)):
            q1 = min(int(naux), int(q0) + int(k_q_block))
            Wblk = W[int(q0) : int(q1)]
            W3 = Wblk.reshape((int(q1 - q0), int(nocc_total), int(nao)))  # (q, i, m)
            for j, (i0, i1) in enumerate(slices):
                nocc_j = int(i1 - i0)
                if nocc_j <= 0:
                    continue
                V = W3[:, int(i0) : int(i1), :].reshape((int(q1 - q0) * nocc_j, int(nao)))
                Ks[j] += V.T @ V
    if tK is not None and tKe is not None:
        tKe.record(cp.cuda.get_current_stream())
        tKe.synchronize()

    # ---- Pass 2: J ----
    Jvec = work.get("Jvec")
    if not isinstance(Jvec, cp.ndarray) or tuple(Jvec.shape) != (nao * nao,) or Jvec.dtype != cp.float64:
        Jvec = cp.zeros((nao * nao,), dtype=cp.float64)
        work["Jvec"] = Jvec
    else:
        Jvec.fill(0.0)

    t2 = cp.cuda.Event() if profile is not None else None
    t2e = cp.cuda.Event() if profile is not None else None
    if t2 is not None and t2e is not None:
        t2.record(cp.cuda.get_current_stream())
    for shell_start, shell_stop, p0, p1 in ctx.aux_blocks:
        X_blk = _int3c2e_block(ctx, shell_start=int(shell_start), shell_stop=int(shell_stop))
        q = int(p1 - p0)
        X2 = X_blk.reshape((nao * nao, q))
        wblk = w[int(p0) : int(p1)]
        Jvec += X2 @ wblk
    if t2 is not None and t2e is not None:
        t2e.record(cp.cuda.get_current_stream())
        t2e.synchronize()

    J = Jvec.reshape((nao, nao))
    J = _symmetrize(cp, J)
    Ks = [_symmetrize(cp, K) for K in Ks]

    if profile is not None:
        prof = profile.setdefault("jk_streamed", {})
        try:
            if t1 is not None and t1e is not None:
                prof["pass1_int3c2e_ms"] = float(cp.cuda.get_elapsed_time(t1, t1e))
            if ts is not None and tse is not None:
                prof["metric_solve_ms"] = float(cp.cuda.get_elapsed_time(ts, tse))
            if tK is not None and tKe is not None:
                prof["k_contract_ms"] = float(cp.cuda.get_elapsed_time(tK, tKe))
            if t2 is not None and t2e is not None:
                prof["pass2_int3c2e_ms"] = float(cp.cuda.get_elapsed_time(t2, t2e))
            prof["YT_nbytes"] = int(getattr(YT, "nbytes", 0))
            prof["nocc_total"] = int(nocc_total)
            prof["nocc_splits"] = [int(i1 - i0) for i0, i1 in slices]
            prof["nao"] = int(nao)
            prof["naux"] = int(naux)
            prof["k_q_block"] = int(k_q_block)
        except Exception:
            pass

    return J, Ks


__all__ = ["StreamedDFJKContext", "make_streamed_df_jk_context", "df_JK_streamed", "df_JKs_streamed"]
