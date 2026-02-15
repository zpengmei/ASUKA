from __future__ import annotations

"""DF J/K helpers.

This module provides fast, SCF-friendly DF J/K contractions that can be reused
by AO-basis SCF drivers without pulling SCF control-flow into the math kernels.
"""

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


def _get_xp(*arrays: Any):
    """Return (xp, is_gpu) where xp is numpy or cupy based on array types."""

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


@contextmanager
def _cublas_math_mode_ctx(xp, cublas_math_mode: str | None) -> Iterator[None]:
    """Temporarily set cuBLAS math mode on the current CuPy handle.

    - None: no-op
    - "default": CUBLAS_DEFAULT_MATH
    - "fp64_emulated_fixedpoint": CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (requires CUDA 13 / cuBLAS 13.x)
    """

    if cublas_math_mode is None or xp is np:
        yield
        return

    import cupy as cp  # noqa: PLC0415
    from cupy_backends.cuda.libs import cublas as cublas_lib  # noqa: PLC0415

    mode = str(cublas_math_mode).lower()
    handle = int(cp.cuda.get_cublas_handle())

    if mode == "default":
        new_math_mode = int(cublas_lib.CUBLAS_DEFAULT_MATH)
    elif mode == "fp64_emulated_fixedpoint":
        ver = int(cublas_lib.getVersion(handle))
        if ver < 130000:
            raise RuntimeError(
                "cublas_math_mode='fp64_emulated_fixedpoint' requires cuBLAS 13.x (CUDA 13.0+); "
                f"detected cublas version={ver}"
            )
        # cublasMath_t: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (CUDA 13.0+)
        new_math_mode = 8
    else:
        raise ValueError("cublas_math_mode must be one of: None, 'default', 'fp64_emulated_fixedpoint'")

    old_math_mode = int(cublas_lib.getMathMode(handle))
    if new_math_mode == old_math_mode:
        yield
        return

    cublas_lib.setMathMode(handle, int(new_math_mode))
    try:
        yield
    finally:
        cublas_lib.setMathMode(handle, int(old_math_mode))


def df_J_from_B2_D(B2, D):
    """Dense DF-J from a (nao*nao, naux) view B2 and density D."""

    xp, _ = _get_xp(B2, D)
    B2 = _as_xp(xp, B2, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if B2.ndim != 2:
        raise ValueError("B2 must be a 2D array")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if int(B2.shape[0]) != int(nao * nao):
        raise ValueError("B2 and D nao mismatch")

    naux = int(B2.shape[1])
    dvec = D.reshape((nao * nao,))
    v = B2.T @ dvec  # (naux,)
    J = (B2 @ v).reshape((nao, nao))
    return J


def df_J_from_BQ_D(BQ, D):
    """DF-J from BQ layout without forming B_mnQ.

    Inputs
    - BQ: (naux, nao, nao), float64, C-contiguous preferred
    - D: (nao, nao), float64
    """

    xp, _ = _get_xp(BQ, D)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(D.shape[0])
    if nao != nao0:
        raise ValueError("BQ and D nao mismatch")

    BQ2 = BQ.reshape((naux, nao * nao))  # view if BQ is contiguous
    d = BQ2 @ D.reshape((nao * nao,))  # (naux,)
    J = (BQ2.T @ d).reshape((nao, nao))
    return J


def df_K_from_BQ_D(BQ, D, *, profile: dict | None = None):
    """DF-K from BQ layout and density D (no B_mnQ required).

    Inputs
    - BQ: (naux, nao, nao), float64
    - D: (nao, nao), float64
    """

    xp, _ = _get_xp(BQ, D)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(D.shape[0])
    if nao != nao0:
        raise ValueError("BQ and D nao mismatch")

    use_chunked = (xp is not np) and (int(naux) * int(nao) * int(nao) >= 10_000_000)
    chunk = 128
    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_impl"] = "chunked_einsum" if use_chunked else "batched_matmul"
        if use_chunked:
            jk_prof["k_chunk_naux"] = int(chunk)

    if use_chunked:
        K = xp.zeros((nao, nao), dtype=xp.float64)
        for q0 in range(0, naux, int(chunk)):
            q1 = min(int(naux), int(q0) + int(chunk))
            BQc = BQ[int(q0) : int(q1)]
            BDc = xp.matmul(BQc, D)  # (q, nao, nao)
            K += xp.einsum("qik,qjk->ij", BDc, BQc)
        BD = None
        KQ = None
    else:
        BD = xp.matmul(BQ, D)  # (naux, nao, nao)
        KQ = xp.matmul(BD, BQ.transpose((0, 2, 1)))  # (naux, nao, nao)
        K = xp.sum(KQ, axis=0)

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        try:
            jk_prof.setdefault("BD_nbytes", int(getattr(BD, "nbytes", 0) if BD is not None else 0))
            jk_prof.setdefault("KQ_nbytes", int(getattr(KQ, "nbytes", 0) if KQ is not None else 0))
        except Exception:
            pass

    return _symmetrize(xp, K)


def df_K_from_BQ_Cocc(
    BQ,
    C_occ,
    occ_vals,
    *,
    q_block: int = 128,
    cublas_math_mode: str | None = None,
):
    """Occupied-driven DF exchange (RI-K) without a dense D.

    Computes:
      K_{μν} = Σ_i occ_i Σ_Q (μi|Q) (νi|Q)
    with:
      (μi|Q) = Σ_λ B_{μλ,Q} C_{λi}

    Inputs
    - BQ: (naux, nao, nao) contiguous, float64
    - C_occ: (nao, nocc) float64 (need not be contiguous; copied if needed)
    - occ_vals: (nocc,) float64
    """

    xp, _ = _get_xp(BQ, C_occ, occ_vals)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    C_occ = _as_xp(xp, C_occ, dtype=xp.float64)
    occ_vals = _as_xp(xp, occ_vals, dtype=xp.float64).ravel()

    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if C_occ.ndim != 2:
        raise ValueError("C_occ must be 2D")
    if occ_vals.ndim != 1:
        raise ValueError("occ_vals must be 1D")

    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(nao0)
    if int(C_occ.shape[0]) != nao:
        raise ValueError("C_occ nao mismatch with BQ")
    nocc = int(C_occ.shape[1])
    if int(occ_vals.shape[0]) != nocc:
        raise ValueError(f"occ_vals must have shape ({nocc},), got {tuple(map(int, occ_vals.shape))}")
    if nocc <= 0:
        return xp.zeros((nao, nao), dtype=xp.float64)

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")

    # Weight occupied orbitals by sqrt(occ) so K becomes a simple sum of outer products.
    sqrt_occ = xp.sqrt(occ_vals)
    Cw = C_occ * sqrt_occ[None, :]
    if hasattr(Cw, "flags") and not bool(Cw.flags.c_contiguous):
        Cw = xp.ascontiguousarray(Cw)

    K = xp.zeros((nao, nao), dtype=xp.float64)
    with _cublas_math_mode_ctx(xp, cublas_math_mode):
        for q0 in range(0, int(naux), int(q_block)):
            q1 = min(int(naux), int(q0) + int(q_block))
            BQc = BQ[int(q0) : int(q1)]  # (q, nao, nao)
            q = int(q1 - q0)

            # (q, μ, ν) x (ν, i) -> (μ, q, i)
            tmp = xp.einsum("qmn,ni->mqi", BQc, Cw, optimize=True)
            if hasattr(tmp, "flags") and not bool(tmp.flags.c_contiguous):
                tmp = xp.ascontiguousarray(tmp)

            U = tmp.reshape((nao, q * nocc))
            K += U @ U.T

    return _symmetrize(xp, K)


__all__ = [
    "df_J_from_B2_D",
    "df_J_from_BQ_D",
    "df_K_from_BQ_D",
    "df_K_from_BQ_Cocc",
]
