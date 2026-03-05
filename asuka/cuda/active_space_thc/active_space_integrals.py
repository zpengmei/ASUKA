from __future__ import annotations

"""Builders for GPU active-space 2e integrals from THC factors.

We use the THC/LS-THC factorization in the AO basis:

  (mu nu | la si) ~= sum_{P,Q} X[P,mu] X[P,nu] Z[P,Q] X[Q,la] X[Q,si]

and assume the central metric has been factorized/stored as:

  Z = Y @ Y.T

This allows us to form DF/Cholesky-like pair vectors in an MO subspace:

  d[L,pq] = sum_P (X_mo[P,p] X_mo[P,q]) Y[P,L]

so that:

  (pq|rs) ~= sum_L d[L,pq] d[L,rs]

The resulting objects are returned as `DeviceDFMOIntegrals` for use with the
CUDA GUGA matvec backend.
"""

import time
from typing import Any

import numpy as np

from asuka.hf.local_thc_jk import local_thc_eri_apply_batched
from asuka.integrals.df_integrals import DeviceDFMOIntegrals


def build_device_dfmo_integrals_thc(
    thc_factors: Any,
    C_active,
    *,
    want_eri_mat: bool = False,
    want_pair_norm: bool = False,
    p_block: int = 8,
    profile: dict | None = None,
) -> DeviceDFMOIntegrals:
    """Build active-space DF-like integrals from global THC factors.

    Parameters
    ----------
    thc_factors
        `asuka.hf.thc_factors.THCFactors` (global THC factors). Must provide
        `.X` (npt,nao) and `.Y` (npt,naux).
    C_active
        Active MO coefficients (nao, ncas). Can be NumPy or CuPy; will be
        converted to CuPy float64.
    want_eri_mat
        If True, materialize the pair-space ERI matrix (ncas^2, ncas^2) on GPU.
        Required for `matvec_backend='cuda_eri_mat'`.
    want_pair_norm
        If True, compute `pair_norm` for the DF vectors.
    p_block
        Block size over the first active-orbital index when building pair
        products. Keeps the (npt, p_block, ncas) temporary bounded.
    profile
        Optional profiling dictionary.
    """

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("build_device_dfmo_integrals_thc requires CuPy") from e

    if profile is not None:
        profile.clear()
        t0 = time.perf_counter()
    else:
        t0 = 0.0

    X = getattr(thc_factors, "X", None)
    Y = getattr(thc_factors, "Y", None)
    if X is None or Y is None:
        raise ValueError("thc_factors must provide .X and .Y (Z factor)")

    X = cp.asarray(X, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)
    if int(getattr(X, "ndim", 0)) != 2:
        raise ValueError("thc_factors.X must be 2D (npt,nao)")
    if int(getattr(Y, "ndim", 0)) != 2:
        raise ValueError("thc_factors.Y must be 2D (npt,naux)")

    npt, nao = map(int, X.shape)
    npt_y, naux = map(int, Y.shape)
    if int(npt_y) != int(npt):
        raise ValueError("thc_factors.X/Y npt mismatch")

    C_act = cp.asarray(C_active, dtype=cp.float64)
    if int(getattr(C_act, "ndim", 0)) != 2:
        raise ValueError("C_active must have shape (nao,ncas)")
    nao_c, ncas = map(int, C_act.shape)
    if int(nao_c) != int(nao):
        raise ValueError("C_active nao mismatch with thc_factors.X")
    if ncas <= 0:
        raise ValueError("C_active must have ncas > 0")

    if hasattr(X, "flags") and not bool(X.flags.c_contiguous):
        X = cp.ascontiguousarray(X)
    if hasattr(Y, "flags") and not bool(Y.flags.c_contiguous):
        Y = cp.ascontiguousarray(Y)
    if hasattr(C_act, "flags") and not bool(C_act.flags.c_contiguous):
        C_act = cp.ascontiguousarray(C_act)

    # X_mo[P,p] = sum_mu X[P,mu] C[mu,p]
    # Shapes: X (npt,nao), C_act (nao,ncas) -> X_act (npt,ncas)
    X_act = cp.ascontiguousarray(X @ C_act, dtype=cp.float64)

    # Build l_full[pq,L] = sum_P (X_act[P,p] X_act[P,q]) Y[P,L]
    # using blocks over p to keep temporaries small.
    p_block = int(p_block)
    if p_block <= 0:
        raise ValueError("p_block must be > 0")
    p_block = min(int(p_block), int(ncas))

    nops = int(ncas) * int(ncas)
    l_full = cp.empty((nops, int(naux)), dtype=cp.float64)

    for p0 in range(0, int(ncas), int(p_block)):
        p1 = min(int(ncas), int(p0) + int(p_block))
        pb = int(p1 - p0)
        if pb <= 0:
            continue

        # pairs[P,i,q] = X_act[P,p_i] * X_act[P,q]
        U = X_act[:, int(p0) : int(p1)]  # (npt,pb)
        pairs = U[:, :, None] * X_act[:, None, :]  # (npt,pb,ncas)
        pairs2 = pairs.reshape(int(npt), int(pb) * int(ncas))  # (npt,pb*ncas)

        block_l = pairs2.T @ Y  # (pb*ncas, naux)
        for i in range(int(pb)):
            p = int(p0) + int(i)
            rows = slice(int(p) * int(ncas), (int(p) + 1) * int(ncas))
            l_full[rows, :] = block_l[int(i) * int(ncas) : (int(i) + 1) * int(ncas), :]

        # Help CuPy reuse the pool; avoid holding the broadcasted 3D array.
        pairs = None
        pairs2 = None
        block_l = None

    l_full = cp.ascontiguousarray(l_full, dtype=cp.float64)

    eri_mat = None
    if bool(want_eri_mat):
        eri_mat = cp.ascontiguousarray(l_full @ l_full.T, dtype=cp.float64)

    # J_{ps} = sum_q (p q| q s) ~= sum_{q,L} d[L,pq] d[L,qs]
    l3 = l_full.reshape(int(ncas), int(ncas), int(naux))
    j_ps = cp.ascontiguousarray(cp.einsum("pql,qsl->ps", l3, l3, optimize=True), dtype=cp.float64)

    pair_norm = None
    if bool(want_pair_norm):
        pair_norm = cp.ascontiguousarray(cp.linalg.norm(l_full, axis=1), dtype=cp.float64)

    if profile is not None:
        cp.cuda.get_current_stream().synchronize()
        profile["t_build_dfmo_s"] = float(time.perf_counter() - float(t0))
        profile["norb"] = int(ncas)
        profile["naux"] = int(naux)
        profile["npt"] = int(npt)
        profile["p_block"] = int(p_block)
        profile["eri_mat_built"] = bool(want_eri_mat)
        profile["pair_norm_built"] = bool(want_pair_norm)
        profile["l_full_nbytes"] = int(getattr(l_full, "nbytes", 0))
        profile["eri_mat_nbytes"] = int(getattr(eri_mat, "nbytes", 0)) if eri_mat is not None else 0

    return DeviceDFMOIntegrals(
        norb=int(ncas),
        l_full=l_full,
        j_ps=j_ps,
        pair_norm=pair_norm,
        eri_mat=eri_mat,
    )


def build_device_dfmo_integrals_local_thc(
    lthc: Any,
    C_active,
    *,
    want_eri_mat: bool = True,
    p_block: int = 8,
    profile: dict | None = None,
) -> DeviceDFMOIntegrals:
    """Build active-space ERIs from LocalTHCFactors via the local AO ERI operator.

    This returns a `DeviceDFMOIntegrals` with:
    - `eri_mat` materialized in ordered-pair space (required for CUDA matvec)
    - `j_ps` computed from `eri_mat`
    - `l_full=None` because each local block has its own aux dimension
    """

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("build_device_dfmo_integrals_local_thc requires CuPy") from e

    if not bool(want_eri_mat):
        raise ValueError("LocalTHC active-space build currently requires want_eri_mat=True")

    if profile is not None:
        profile.clear()
        t0 = time.perf_counter()
    else:
        t0 = 0.0

    blocks = getattr(lthc, "blocks", None)
    if blocks is None:
        raise ValueError("lthc must provide .blocks (LocalTHCFactors)")
    blocks = tuple(blocks)
    if len(blocks) == 0:
        raise ValueError("LocalTHCFactors.blocks is empty")

    C_act = cp.asarray(C_active, dtype=cp.float64)
    if int(getattr(C_act, "ndim", 0)) != 2:
        raise ValueError("C_active must have shape (nao,ncas)")
    nao, ncas = map(int, C_act.shape)
    if ncas <= 0:
        raise ValueError("C_active must have ncas > 0")

    pair_p_block = int(p_block)
    if pair_p_block <= 0:
        raise ValueError("p_block must be > 0")
    pair_p_block = min(int(pair_p_block), int(ncas))

    nops = int(ncas) * int(ncas)
    eri_mat = cp.zeros((nops, nops), dtype=cp.float64)

    nblocks_used = 0
    for r0 in range(0, int(ncas), int(pair_p_block)):
        r1 = min(int(ncas), int(r0) + int(pair_p_block))
        rb = int(r1 - r0)
        if rb <= 0:
            continue

        nbatch = int(rb) * int(ncas)
        D_batch = cp.empty((nbatch, int(nao), int(nao)), dtype=cp.float64)

        ib = 0
        for r in range(int(r0), int(r1)):
            cr = C_act[:, int(r)]
            for s in range(int(ncas)):
                cs = C_act[:, int(s)]
                D_batch[int(ib)] = 0.5 * (
                    cr[:, None] * cs[None, :] + cs[:, None] * cr[None, :]
                )
                ib += 1

        V_batch = local_thc_eri_apply_batched(D_batch, lthc, symmetrize=True)
        eri_blk = cp.einsum("mp,bmn,nq->pqb", C_act, V_batch, C_act, optimize=True)
        eri_mat[:, int(r0) * int(ncas) : int(r1) * int(ncas)] = eri_blk.reshape(int(nops), int(nbatch))

        nblocks_used += 1

        D_batch = None
        V_batch = None
        eri_blk = None

    if nblocks_used <= 0:
        raise RuntimeError("LocalTHC active-space build produced no pair blocks")

    eri_mat = 0.5 * (eri_mat + eri_mat.T)

    # J_{ps} = sum_q (p q| q s)
    g4 = eri_mat.reshape(int(ncas), int(ncas), int(ncas), int(ncas))
    j_ps = cp.ascontiguousarray(cp.einsum("pqqs->ps", g4, optimize=True), dtype=cp.float64)

    if profile is not None:
        cp.cuda.get_current_stream().synchronize()
        profile["t_build_dfmo_s"] = float(time.perf_counter() - float(t0))
        profile["norb"] = int(ncas)
        profile["nao"] = int(nao)
        profile["nblocks_used"] = int(nblocks_used)
        profile["pair_p_block"] = int(pair_p_block)
        profile["eri_mat_nbytes"] = int(getattr(eri_mat, "nbytes", 0))

    return DeviceDFMOIntegrals(
        norb=int(ncas),
        l_full=None,
        j_ps=j_ps,
        pair_norm=None,
        eri_mat=eri_mat,
    )


__all__ = ["build_device_dfmo_integrals_local_thc", "build_device_dfmo_integrals_thc"]
