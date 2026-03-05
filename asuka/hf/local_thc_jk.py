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


def local_thc_J(D, lthc: LocalTHCFactors):
    """Assemble global Coulomb matrix J[D] from LocalTHCFactors."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local_thc_J requires CuPy") from e

    D = cp.asarray(D, dtype=cp.float64)
    nao = int(D.shape[0])
    if D.ndim != 2 or int(D.shape[1]) != nao:
        raise ValueError("D must be (nao,nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with D")

    J = cp.zeros((nao, nao), dtype=cp.float64)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_sub = D[idx[:, None], idx[None, :]]

        J_sub = thc_J(D_sub, blk.X, blk.Z)

        n_early = int(getattr(blk, "n_early", 0))
        nprim = int(blk.n_primary)
        nloc = int(idx_np.size)
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if nprim < 0 or (n_early + nprim) > nloc:
            raise ValueError("invalid blk.n_primary")

        if n_early > 0:
            J_sub[:n_early, :] = 0.0
            J_sub[:, :n_early] = 0.0
        tail = int(n_early + nprim)
        if tail < nloc:
            J_sub[tail:, tail:] = 0.0

        J[idx[:, None], idx[None, :]] += J_sub

    return 0.5 * (J + J.T)


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

        n_early = int(getattr(blk, "n_early", 0))
        nprim = int(blk.n_primary)
        nloc = int(idx_np.size)
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if nprim < 0 or (n_early + nprim) > nloc:
            raise ValueError("invalid blk.n_primary")

        if n_early > 0:
            K_sub[:n_early, :] = 0.0
            K_sub[:, :n_early] = 0.0
        tail = int(n_early + nprim)
        if tail < nloc:
            K_sub[tail:, tail:] = 0.0

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

        n_early = int(getattr(blk, "n_early", 0))
        nprim = int(blk.n_primary)
        nloc = int(idx_np.size)
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if nprim < 0 or (n_early + nprim) > nloc:
            raise ValueError("invalid blk.n_primary")

        # Ownership mask (Song & Martinez-style ordering):
        # local AO order is [early secondary][primary][late secondary], and this
        # block owns output elements where min(owner(i), owner(j)) == block_id.
        #
        # This corresponds to:
        # - drop any output involving early secondaries (owned by earlier blocks)
        # - drop late-secondary x late-secondary outputs (owned by later blocks)
        if n_early > 0:
            J_sub[:n_early, :] = 0.0
            J_sub[:, :n_early] = 0.0
            K_sub[:n_early, :] = 0.0
            K_sub[:, :n_early] = 0.0
        tail = int(n_early + nprim)
        if tail < nloc:
            J_sub[tail:, tail:] = 0.0
            K_sub[tail:, tail:] = 0.0

        # Scatter-add into global matrices.
        J[idx[:, None], idx[None, :]] += J_sub
        K[idx[:, None], idx[None, :]] += K_sub

    # Enforce symmetry (numerical noise + scatter).
    J = 0.5 * (J + J.T)
    K = 0.5 * (K + K.T)
    return J, K


__all__ = ["local_thc_J", "local_thc_JK", "local_thc_K_blocked"]
