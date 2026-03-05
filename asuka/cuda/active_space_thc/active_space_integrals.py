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
    """Build active-space ERIs from LocalTHCFactors blocks (ownership-masked).

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

    p_block = int(p_block)
    if p_block <= 0:
        raise ValueError("p_block must be > 0")
    p_block = min(int(p_block), int(ncas))

    # Accumulate the full ordered-pair ERI matrix in active space.
    nops = int(ncas) * int(ncas)
    eri_mat = cp.zeros((nops, nops), dtype=cp.float64)

    nblocks_used = 0
    for blk in blocks:
        X_blk = getattr(blk, "X", None)
        Y_blk = getattr(blk, "Y", None)
        if X_blk is None or Y_blk is None:
            raise ValueError("LocalTHCBlock must provide .X and .Y (Z factor)")

        X_blk = cp.asarray(X_blk, dtype=cp.float64)
        Y_blk = cp.asarray(Y_blk, dtype=cp.float64)
        if int(getattr(X_blk, "ndim", 0)) != 2:
            raise ValueError("LocalTHCBlock.X must be 2D (npt,nloc_ao)")
        if int(getattr(Y_blk, "ndim", 0)) != 2:
            raise ValueError("LocalTHCBlock.Y must be 2D (npt,naux_blk)")

        npt, nloc = map(int, X_blk.shape)
        npt_y, naux_blk = map(int, Y_blk.shape)
        if int(npt_y) != int(npt):
            raise ValueError("LocalTHCBlock.X/Y npt mismatch")

        ao_idx_global = getattr(blk, "ao_idx_global", None)
        if ao_idx_global is None:
            raise ValueError("LocalTHCBlock.ao_idx_global is missing")
        idx_np = np.asarray(ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) != int(nloc):
            raise ValueError("LocalTHCBlock.ao_idx_global size mismatch with X columns")

        n_early = int(getattr(blk, "n_early", 0))
        n_primary = int(getattr(blk, "n_primary", 0))
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if n_primary < 0 or (n_early + n_primary) > nloc:
            raise ValueError("invalid blk.n_primary")
        tail = int(n_early + n_primary)

        # Local AO slices:
        # - non-early: [primary][late]
        # - late only: [late]
        idx_nonE = idx_np[int(n_early) :]
        idx_late = idx_np[int(tail) :]
        if int(idx_nonE.size) == 0:
            continue

        C_nonE = C_act[idx_nonE, :]  # (n_nonE_ao, ncas)
        X_nonE_act = X_blk[:, int(n_early) :] @ C_nonE  # (npt,ncas)

        have_late = bool(int(idx_late.size) > 0)
        if have_late:
            C_late = C_act[idx_late, :]
            X_late_act = X_blk[:, int(tail) :] @ C_late  # (npt,ncas)
        else:
            X_late_act = None

        # Build block pair vectors d_blk[pq,L] = sum_P (X_full[p]X_full[q] - X_late[p]X_late[q]) Y[P,L]
        l_full_blk = cp.empty((nops, int(naux_blk)), dtype=cp.float64)

        for p0 in range(0, int(ncas), int(p_block)):
            p1 = min(int(ncas), int(p0) + int(p_block))
            pb = int(p1 - p0)
            if pb <= 0:
                continue

            U = X_nonE_act[:, int(p0) : int(p1)]  # (npt,pb)
            pairs_full = U[:, :, None] * X_nonE_act[:, None, :]  # (npt,pb,ncas)
            if have_late and X_late_act is not None:
                Ul = X_late_act[:, int(p0) : int(p1)]
                pairs_late = Ul[:, :, None] * X_late_act[:, None, :]
                pairs_full = pairs_full - pairs_late
                pairs_late = None
                Ul = None
            pairs2 = pairs_full.reshape(int(npt), int(pb) * int(ncas))
            block_l = pairs2.T @ Y_blk  # (pb*ncas,naux_blk)

            for i in range(int(pb)):
                p = int(p0) + int(i)
                rows = slice(int(p) * int(ncas), (int(p) + 1) * int(ncas))
                l_full_blk[rows, :] = block_l[int(i) * int(ncas) : (int(i) + 1) * int(ncas), :]

            pairs_full = None
            pairs2 = None
            block_l = None

        # Accumulate symmetric contribution to ERI matrix.
        eri_mat += l_full_blk @ l_full_blk.T
        nblocks_used += 1

        # Help the CuPy memory pool.
        l_full_blk = None
        X_nonE_act = None
        X_late_act = None

    if nblocks_used <= 0:
        raise RuntimeError("LocalTHC active-space build produced no block contributions")

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
        profile["eri_mat_nbytes"] = int(getattr(eri_mat, "nbytes", 0))

    return DeviceDFMOIntegrals(
        norb=int(ncas),
        l_full=None,
        j_ps=j_ps,
        pair_norm=None,
        eri_mat=eri_mat,
    )


__all__ = ["build_device_dfmo_integrals_local_thc", "build_device_dfmo_integrals_thc"]
