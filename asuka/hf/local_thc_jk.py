from __future__ import annotations

"""Local-THC J/K build.

Given LocalTHCFactors (per-block X/Z and AO index lists), assemble global
Coulomb and exchange matrices by summing block contributions while avoiding
double counting via a block-ownership rule (min block id of the AO pair).

Implementation detail: each local block uses AO ordering
``[early secondary][primary][late secondary]`` and we zero:
- any output involving early secondaries (owned by earlier blocks)
- late-secondary x late-secondary outputs (owned by later blocks)
"""

from typing import Any

import numpy as np

from .local_thc_factors import LocalTHCFactors
from .thc_jk import THCJKWork, thc_J, thc_JK, thc_K_blocked


def _require_cupy():
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local-THC CUDA contractions require CuPy") from e
    return cp


def _validate_local_thc_matrix(mat, lthc: LocalTHCFactors, *, name: str):
    cp = _require_cupy()

    arr = cp.asarray(mat, dtype=cp.float64)
    nao = int(arr.shape[-1])
    if int(arr.ndim) != 2 or int(arr.shape[0]) != int(nao):
        raise ValueError(f"{name} must be (nao,nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError(f"lthc.nao mismatch with {name}")
    return arr, nao


def _mask_owned_outputs_inplace(mat_sub, *, n_early: int, n_primary: int) -> None:
    nloc = int(mat_sub.shape[0])
    n_early = int(n_early)
    n_primary = int(n_primary)
    if n_early < 0 or n_early > nloc:
        raise ValueError("invalid blk.n_early")
    if n_primary < 0 or (n_early + n_primary) > nloc:
        raise ValueError("invalid blk.n_primary")

    if n_early > 0:
        mat_sub[:n_early, :] = 0.0
        mat_sub[:, :n_early] = 0.0
    tail = int(n_early + n_primary)
    if tail < nloc:
        mat_sub[tail:, tail:] = 0.0


def local_thc_eri_apply(D, lthc: LocalTHCFactors, *, symmetrize: bool = False):
    """Apply the local-THC Coulomb-like AO operator to an arbitrary AO pair density.

    This contracts

      V[p,q] = sum_{r,s} (p q | r s)_local * D[r,s]

    using the same block ownership rules as `local_thc_J`.
    """

    cp = _require_cupy()
    D, nao = _validate_local_thc_matrix(D, lthc, name="D")

    V = cp.zeros((nao, nao), dtype=cp.float64)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue

        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_sub = D[idx[:, None], idx[None, :]]
        V_sub = thc_J(D_sub, blk.X, blk.Z)

        _mask_owned_outputs_inplace(
            V_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )
        V[idx[:, None], idx[None, :]] += V_sub

    if bool(symmetrize):
        V = 0.5 * (V + V.T)
    return V


def local_thc_eri_apply_batched(D_batch, lthc: LocalTHCFactors, *, symmetrize: bool = False):
    """Batched version of `local_thc_eri_apply` over the leading batch dimension."""

    cp = _require_cupy()
    D_batch = cp.asarray(D_batch, dtype=cp.float64)
    if int(D_batch.ndim) != 3:
        raise ValueError("D_batch must have shape (nbatch,nao,nao)")

    nbatch, nao, nao2 = map(int, D_batch.shape)
    if int(nao2) != int(nao):
        raise ValueError("D_batch must have square trailing dimensions")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with D_batch")

    out = cp.empty((nbatch, nao, nao), dtype=cp.float64)
    for ib in range(int(nbatch)):
        out[int(ib)] = local_thc_eri_apply(D_batch[int(ib)], lthc, symmetrize=bool(symmetrize))
    return out


def local_thc_J(D, lthc: LocalTHCFactors):
    """Assemble global Coulomb matrix J[D] from LocalTHCFactors."""
    return local_thc_eri_apply(D, lthc, symmetrize=True)


def local_thc_K_blocked(D, lthc: LocalTHCFactors, *, q_block: int = 256, work: THCJKWork | None = None):
    """Assemble global exchange matrix K[D] from LocalTHCFactors."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local_thc_K_blocked requires CuPy") from e

    if work is None:
        work = THCJKWork(q_block=int(q_block))

    D = cp.asarray(D, dtype=cp.float64)
    nao = int(D.shape[0])
    if D.ndim != 2 or int(D.shape[1]) != nao:
        raise ValueError("D must be (nao,nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with D")

    K = cp.zeros((nao, nao), dtype=cp.float64)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_sub = D[idx[:, None], idx[None, :]]

        K_sub = thc_K_blocked(D_sub, blk.X, blk.Z, q_block=int(work.q_block))

        _mask_owned_outputs_inplace(
            K_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )

        K[idx[:, None], idx[None, :]] += K_sub

    return 0.5 * (K + K.T)


def local_thc_JK(D, lthc: LocalTHCFactors, *, q_block: int = 256, work: THCJKWork | None = None):
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local_thc_JK requires CuPy") from e

    if work is None:
        work = THCJKWork(q_block=int(q_block))

    D = cp.asarray(D, dtype=cp.float64)
    nao = int(D.shape[0])
    if D.ndim != 2 or int(D.shape[1]) != nao:
        raise ValueError("D must be (nao,nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with D")

    J = cp.zeros((nao, nao), dtype=cp.float64)
    K = cp.zeros((nao, nao), dtype=cp.float64)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        # Extract sub-density in block AO ordering.
        D_sub = D[idx[:, None], idx[None, :]]

        J_sub, K_sub = thc_JK(D_sub, blk.X, blk.Z, work=work)

        # Ownership mask (Song & Martinez-style ordering):
        # local AO order is [early secondary][primary][late secondary], and this
        # block owns output elements where min(owner(i), owner(j)) == block_id.
        _mask_owned_outputs_inplace(
            J_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )
        _mask_owned_outputs_inplace(
            K_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )

        # Scatter-add into global matrices.
        J[idx[:, None], idx[None, :]] += J_sub
        K[idx[:, None], idx[None, :]] += K_sub

    # Enforce symmetry (numerical noise + scatter).
    J = 0.5 * (J + J.T)
    K = 0.5 * (K + K.T)
    return J, K


__all__ = [
    "local_thc_eri_apply",
    "local_thc_eri_apply_batched",
    "local_thc_J",
    "local_thc_JK",
    "local_thc_K_blocked",
]
