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
from .thc_jk import (
    THCJKWork,
    THCPrecisionPolicy,
    _maybe_tf32_ctx,
    thc_J,
    thc_J_factored,
    thc_JK,
    thc_JK_factored,
    thc_K_blocked,
    thc_K_blocked_factored,
)


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


def local_thc_eri_apply(
    D,
    lthc: LocalTHCFactors,
    *,
    symmetrize: bool = False,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
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
        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        V_sub = thc_J(D_sub, X_use, Z_use, Y=Y_use, policy=policy)

        _mask_owned_outputs_inplace(
            V_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )
        V[idx[:, None], idx[None, :]] += V_sub

    if bool(symmetrize):
        V = 0.5 * (V + V.T)
    return V


def local_thc_eri_apply_batched(
    D_batch,
    lthc: LocalTHCFactors,
    *,
    symmetrize: bool = False,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Batched version of `local_thc_eri_apply` over the leading batch dimension."""

    cp = _require_cupy()
    policy = THCPrecisionPolicy() if policy is None else policy
    compute_dtype = cp.dtype(getattr(policy, "compute_dtype", cp.float64))
    out_dtype = cp.dtype(getattr(policy, "out_dtype", cp.float64))
    use_tf32 = bool(getattr(policy, "use_tf32", False)) and compute_dtype == cp.float32
    prefer_Y = bool(getattr(policy, "prefer_Y", False))

    D_batch = cp.asarray(D_batch, dtype=compute_dtype)
    if int(D_batch.ndim) != 3:
        raise ValueError("D_batch must have shape (nbatch,nao,nao)")

    nbatch, nao, nao2 = map(int, D_batch.shape)
    if int(nao2) != int(nao):
        raise ValueError("D_batch must have square trailing dimensions")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with D_batch")

    out = cp.zeros((int(nbatch), int(nao), int(nao)), dtype=compute_dtype)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        nloc = int(idx_np.size)
        if nloc == 0:
            continue

        idx = cp.asarray(idx_np, dtype=cp.int32)
        D_sub = D_batch[:, idx[:, None], idx[None, :]]  # (nbatch,nloc,nloc)

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X = getattr(cache, "X_tc") if cache is not None else getattr(blk, "X", None)
        Y = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z = None if cache is not None else getattr(blk, "Z", None)

        X = cp.asarray(X, dtype=compute_dtype)
        if int(getattr(X, "ndim", 0)) != 2 or int(X.shape[1]) != int(nloc):
            raise ValueError("LocalTHCBlock.X must have shape (npt,nlocal_ao)")
        npt = int(X.shape[0])

        use_y = (Y is not None) and (Z is None or bool(prefer_Y))
        if use_y:
            Y = cp.asarray(Y, dtype=compute_dtype)
            if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Y must have shape (npt,naux)")
        else:
            if Z is None:
                raise ValueError("LocalTHCBlock.Z is missing (provide Y or store_Z=True)")
            Z = cp.asarray(Z, dtype=compute_dtype)
            if int(getattr(Z, "ndim", 0)) != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Z must have shape (npt,npt)")

        # Batched THC-J in this local AO ordering:
        #   A = X D
        #   m = sum_mu A[P,mu] X[P,mu]
        #   n = Z m   (or Y(Y^T m))
        #   J = X^T (n*X)
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            A = cp.matmul(X[None, :, :], D_sub)  # (nbatch,npt,nloc)
        m = cp.sum(A * X[None, :, :], axis=2)  # (nbatch,npt)
        if use_y:
            assert Y is not None
            tmp = m @ Y  # (nbatch,naux)
            n = tmp @ Y.T  # (nbatch,npt)
        else:
            assert Z is not None
            n = m @ Z.T  # (nbatch,npt)

        Xn = X[None, :, :] * n[:, :, None]  # (nbatch,npt,nloc)
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            J_sub = cp.matmul(X.T[None, :, :], Xn)  # (nbatch,nloc,nloc)

        # Ownership masking (in local ordering).
        n_early = int(getattr(blk, "n_early", 0))
        n_primary = int(getattr(blk, "n_primary", 0))
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if n_primary < 0 or (n_early + n_primary) > nloc:
            raise ValueError("invalid blk.n_primary")
        tail = int(n_early + n_primary)
        if n_early > 0:
            J_sub[:, :n_early, :] = 0.0
            J_sub[:, :, :n_early] = 0.0
        if tail < nloc:
            J_sub[:, tail:, tail:] = 0.0

        out[:, idx[:, None], idx[None, :]] += J_sub

        del idx, D_sub, X, Y, Z, A, m, n, Xn, J_sub
        if use_y:
            del tmp

    if bool(symmetrize):
        out = 0.5 * (out + out.transpose((0, 2, 1)))
    return cp.asarray(out, dtype=out_dtype)


def local_thc_eri_apply_pairs_mo_batched(
    c_left_batch,
    c_right_batch,
    lthc: LocalTHCFactors,
    C_left,
    C_right,
    *,
    symmetrize: bool = False,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Apply the local-THC Coulomb-like AO operator to a batch of symmetrized rank-2 pair densities and contract to MO space.

    For each batch item ``b``, define the AO density:

      D_b = 0.5 * (c_left_b c_right_b^T + c_right_b c_left_b^T)

    Let ``V_b = local_thc_eri_apply(D_b, lthc, symmetrize=True)`` (same block ownership rules as local-THC J).
    This routine returns the contracted MO-space matrices:

      out_b = C_left^T @ V_b @ C_right

    without materializing ``D_b`` or ``V_b`` as full (nao,nao) matrices.

    Shapes
    - c_left_batch: (nbatch,nao) or (nao,)
    - c_right_batch: (nbatch,nao) or (nao,)
    - C_left: (nao,nleft)
    - C_right: (nao,nright)
    - out: (nbatch,nleft,nright)

    Notes
    - This is the intended fast path for orbital/nuclear gradient pair batches where densities are rank-2 by construction.
    - The `symmetrize` flag is accepted for API parity but is effectively a no-op: the underlying THC operator is symmetric.
    """

    cp = _require_cupy()

    policy = THCPrecisionPolicy() if policy is None else policy
    compute_dtype = cp.dtype(getattr(policy, "compute_dtype", cp.float64))
    out_dtype = cp.dtype(getattr(policy, "out_dtype", cp.float64))
    use_tf32 = bool(getattr(policy, "use_tf32", False)) and compute_dtype == cp.float32
    prefer_Y = bool(getattr(policy, "prefer_Y", False))

    c_left = cp.asarray(c_left_batch, dtype=compute_dtype)
    c_right = cp.asarray(c_right_batch, dtype=compute_dtype)
    if int(c_left.ndim) == 1:
        c_left = c_left[None, :]
    if int(c_right.ndim) == 1:
        c_right = c_right[None, :]
    if int(c_left.ndim) != 2 or int(c_right.ndim) != 2:
        raise ValueError("c_left_batch/c_right_batch must have shape (nbatch,nao) or (nao,)")
    nbatch, nao = map(int, c_left.shape)
    nbatch2, nao2 = map(int, c_right.shape)
    if int(nbatch2) != int(nbatch) or int(nao2) != int(nao):
        raise ValueError("c_left_batch and c_right_batch must have the same shape")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with c_left_batch/c_right_batch")

    C_left = cp.asarray(C_left, dtype=compute_dtype)
    C_right = cp.asarray(C_right, dtype=compute_dtype)
    if int(getattr(C_left, "ndim", 0)) != 2 or int(getattr(C_right, "ndim", 0)) != 2:
        raise ValueError("C_left/C_right must be 2D arrays")
    nao_l, nleft = map(int, C_left.shape)
    nao_r, nright = map(int, C_right.shape)
    if int(nao_l) != int(nao) or int(nao_r) != int(nao):
        raise ValueError("C_left/C_right nao mismatch with c_left_batch/c_right_batch")

    # Output: (nbatch,nleft,nright)
    out = cp.zeros((int(nbatch), int(nleft), int(nright)), dtype=compute_dtype)

    # Blockwise accumulation in MO space. Each block computes V_sub(D_sub) in its
    # local AO ordering, applies the ownership mask, and contributes:
    #   out_b += C_left_sub^T @ V_sub_b @ C_right_sub
    #
    # We avoid forming V_sub_b explicitly by first computing:
    #   T_sub_b = V_sub_b @ C_right_sub
    # using point-space intermediates (n is batch-dependent but X/Y are not).
    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        nloc = int(idx_np.size)
        if nloc == 0:
            continue

        idx = cp.asarray(idx_np, dtype=cp.int32)

        # Subvectors / submatrices in this block's local AO ordering.
        cL = c_left[:, idx]  # (nbatch,nloc)
        cR = c_right[:, idx]  # (nbatch,nloc)
        CL = C_left[idx, :]  # (nloc,nleft)
        CR = C_right[idx, :]  # (nloc,nright)

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X = getattr(cache, "X_tc") if cache is not None else getattr(blk, "X", None)
        Y = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z = None if cache is not None else getattr(blk, "Z", None)

        X = cp.asarray(X, dtype=compute_dtype)
        if int(getattr(X, "ndim", 0)) != 2 or int(X.shape[1]) != int(nloc):
            raise ValueError("LocalTHCBlock.X must have shape (npt,nlocal_ao)")

        # Point-space scalar for each batch density:
        #   m[P] = X[P,:] D X[P,:]^T
        # For D = 0.5*(cL cR^T + cR cL^T), this simplifies to:
        #   m[P] = (X cL)[P] * (X cR)[P]
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            XT = X.T
            pL = cL @ XT  # (nbatch,npt)
            pR = cR @ XT  # (nbatch,npt)
            m = pL * pR  # (nbatch,npt)

            # n = Z @ m, but prefer the stored factor Y where Z = Y Y^T:
            # For row-major batch m: n = m @ Z^T = (m @ Y) @ Y^T.
            use_y = (Y is not None) and (Z is None or bool(prefer_Y))
            if use_y:
                Y = cp.asarray(Y, dtype=compute_dtype)
                if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(X.shape[0]):
                    raise ValueError("LocalTHCBlock.Y must have shape (npt,naux)")
                tmp = m @ Y  # (nbatch,naux)
                n = tmp @ Y.T  # (nbatch,npt)
            else:
                if Z is None:
                    raise ValueError("LocalTHCBlock.Z is missing (provide Y or store_Z=True)")
                Z = cp.asarray(Z, dtype=compute_dtype)
                if int(getattr(Z, "ndim", 0)) != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(X.shape[0]):
                    raise ValueError("LocalTHCBlock.Z must have shape (npt,npt)")
                n = m @ Z.T  # (nbatch,npt)

        # Ownership masking for the output AO pairs.
        n_early = int(getattr(blk, "n_early", 0))
        n_primary = int(getattr(blk, "n_primary", 0))
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if n_primary < 0 or (n_early + n_primary) > nloc:
            raise ValueError("invalid blk.n_primary")
        tail = int(n_early + n_primary)

        # Compute T = (V_mask @ CR) for each batch item without forming V_mask.
        #
        # Local AO ordering: [early secondary][primary][late secondary]
        # Mask rule: drop any output involving early secondary; drop late-late.
        #
        # Partition X into columns for primary (P) and (P∪L) and late (L).
        X_P = X[:, int(n_early) : int(tail)]  # (npt,n_primary)
        X_K = X[:, int(n_early) :]  # (npt,n_primary+n_late)
        X_L = X[:, int(tail) :]  # (npt,n_late)
        CR_P = CR[int(n_early) : int(tail), :]  # (n_primary,nright)
        CR_K = CR[int(n_early) :, :]  # (n_primary+n_late,nright)

        # R_K = X_K @ CR_K, R_P = X_P @ CR_P (point-space MO values).
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            R_K = X_K @ CR_K  # (npt,nright)
            R_P = X_P @ CR_P  # (npt,nright)

        # Primary rows: T_P = X_P^T @ (diag(n) @ R_K)
        S_K = n[:, :, None] * R_K[None, :, :]  # (nbatch,npt,nright)
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            T_P = cp.matmul(X_P.T[None, :, :], S_K)  # (nbatch,n_primary,nright)

        # Late rows: T_L = X_L^T @ (diag(n) @ R_P)  (no late-late contribution)
        if int(tail) < int(nloc):
            S_P = n[:, :, None] * R_P[None, :, :]  # (nbatch,npt,nright)
            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                T_L = cp.matmul(X_L.T[None, :, :], S_P)  # (nbatch,n_late,nright)
        else:
            T_L = None

        # Assemble local T with early rows zero.
        T = cp.zeros((int(nbatch), int(nloc), int(nright)), dtype=compute_dtype)
        if int(n_primary) > 0:
            T[:, int(n_early) : int(tail), :] = T_P
        if T_L is not None:
            T[:, int(tail) :, :] = T_L

        # out += CL^T @ T
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            out += cp.matmul(CL.T[None, :, :], T)

        # Help the CuPy memory pool reuse buffers sooner.
        if bool(symmetrize):  # no-op, but keep a side effect to silence unused-param linters.
            pass

    return cp.asarray(out, dtype=out_dtype)


def local_thc_K_factored_mo_batched(
    U_batch,
    V_batch,
    lthc: LocalTHCFactors,
    C_left,
    C_right,
    *,
    q_block: int = 64,
    batch_block: int | None = None,
    max_workspace_bytes: int = 256 * 1024 * 1024,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Batched MO contraction for the local-THC exchange build with factored densities.

    For each batch item ``b`` with density ``D_b = U_b V_b^T``, define the local-THC
    exchange matrix (with the same block ownership rule as `local_thc_K_blocked`):

      K_b = K_local[D_b]

    This routine returns the MO-contracted matrices:

      out_b = C_left^T @ K_b @ C_right

    without materializing dense AO densities ``D_b`` or dense AO exchange matrices ``K_b``.

    Shapes
    - U_batch: (nbatch,nao,r) or (nao,r)
    - V_batch: (nbatch,nao,r) or (nao,r)
    - C_left: (nao,nleft)
    - C_right: (nao,nright)
    - out: (nbatch,nleft,nright)

    Notes
    - This is intended for (approximately) symmetric densities (U and V represent the same D);
      for general non-symmetric U/V this computes the contraction with the raw K[D], not the
      symmetrized 0.5*(K+K^T).
    - This is intended for low-rank densities (small r) and/or small MO subspaces.
    - Workspace is bounded by chunking over the batch dimension (and the q-block) so the
      temporary (nbatch,npt,qb) tensor does not explode.
    """

    cp = _require_cupy()

    policy = THCPrecisionPolicy() if policy is None else policy
    compute_dtype = cp.dtype(getattr(policy, "compute_dtype", cp.float64))
    out_dtype = cp.dtype(getattr(policy, "out_dtype", cp.float64))
    use_tf32 = bool(getattr(policy, "use_tf32", False)) and compute_dtype == cp.float32
    prefer_Y = bool(getattr(policy, "prefer_Y", False))

    U = cp.asarray(U_batch, dtype=compute_dtype)
    V = cp.asarray(V_batch, dtype=compute_dtype)
    if int(U.ndim) == 2:
        U = U[None, :, :]
    if int(V.ndim) == 2:
        V = V[None, :, :]
    if int(U.ndim) != 3 or int(V.ndim) != 3:
        raise ValueError("U_batch/V_batch must have shape (nbatch,nao,r) or (nao,r)")
    nbatch, nao, r0 = map(int, U.shape)
    nbatch2, nao2, r1 = map(int, V.shape)
    if nbatch2 != nbatch or nao2 != nao or r1 != r0:
        raise ValueError("U_batch and V_batch must have the same shape (nbatch,nao,r)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with U_batch/V_batch")

    C_left = cp.asarray(C_left, dtype=compute_dtype)
    C_right = cp.asarray(C_right, dtype=compute_dtype)
    if int(getattr(C_left, "ndim", 0)) != 2 or int(getattr(C_right, "ndim", 0)) != 2:
        raise ValueError("C_left/C_right must be 2D arrays")
    nao_l, nleft = map(int, C_left.shape)
    nao_r, nright = map(int, C_right.shape)
    if nao_l != nao or nao_r != nao:
        raise ValueError("C_left/C_right nao mismatch with U_batch/V_batch")

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")

    max_workspace_bytes = int(max_workspace_bytes)
    if max_workspace_bytes <= 0:
        raise ValueError("max_workspace_bytes must be > 0")

    out = cp.zeros((int(nbatch), int(nleft), int(nright)), dtype=compute_dtype)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        nloc = int(idx_np.size)
        if nloc == 0:
            continue

        idx = cp.asarray(idx_np, dtype=cp.int32)

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X = getattr(cache, "X_tc") if cache is not None else getattr(blk, "X", None)
        Y = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z = None if cache is not None else getattr(blk, "Z", None)

        X = cp.asarray(X, dtype=compute_dtype)
        if int(getattr(X, "ndim", 0)) != 2 or int(X.shape[1]) != int(nloc):
            raise ValueError("LocalTHCBlock.X must have shape (npt,nlocal_ao)")
        npt = int(X.shape[0])

        # For exchange we need Z blocks (Hadamard with M). Prefer the stored Y
        # factor when requested (or when Z is missing) to avoid loading/casting
        # a potentially-large full Z.
        use_y = (Y is not None) and (Z is None or bool(prefer_Y))
        if use_y:
            Y = cp.asarray(Y, dtype=compute_dtype)
            if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Y must have shape (npt,naux)")
        else:
            if Z is None:
                raise ValueError("LocalTHCBlock.Z is missing (provide Y or store_Z=True)")
            Z = cp.asarray(Z, dtype=compute_dtype)
            if int(getattr(Z, "ndim", 0)) != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Z must have shape (npt,npt)")

        # Ownership masking on AO output pairs (same rule as local_thc_K_blocked).
        n_early = int(getattr(blk, "n_early", 0))
        n_primary = int(getattr(blk, "n_primary", 0))
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if n_primary < 0 or (n_early + n_primary) > nloc:
            raise ValueError("invalid blk.n_primary")
        tail = int(n_early + n_primary)

        # Output-side AO subsets for ownership:
        # - Keep K over (primary ∪ late) and subtract the late-late piece.
        X_K = X[:, int(n_early) :]
        X_L = X[:, int(tail) :]

        CL_sub = C_left[idx, :]  # (nloc,nleft)
        CR_sub = C_right[idx, :]  # (nloc,nright)
        CL_K = CL_sub[int(n_early) :, :]
        CR_K = CR_sub[int(n_early) :, :]
        CL_L = CL_sub[int(tail) :, :]
        CR_L = CR_sub[int(tail) :, :]

        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            A_K = X_K @ CL_K  # (npt,nleft)
            B_K = X_K @ CR_K  # (npt,nright)
        if int(tail) < int(nloc):
            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                A_L = X_L @ CL_L
                B_L = X_L @ CR_L
            has_late = True
        else:
            A_L = None
            B_L = None
            has_late = False

        qb = min(int(q_block), int(npt))

        # Choose a batch chunk size so the dominant temporary (bb,npt,qb) stays bounded.
        if batch_block is not None:
            bb = max(1, min(int(batch_block), int(nbatch)))
        else:
            itemsize = int(np.dtype(compute_dtype).itemsize)
            denom = int(npt) * int(qb) * int(itemsize)  # bytes for one (npt,qb) slice
            # We need at least Mq and Tq buffers; be conservative by dividing by 2.
            bb = int(max_workspace_bytes // max(int(denom) * 2, 1))
            bb = max(1, min(int(bb), int(nbatch)))

        # Precompute A^T for batched matmul.
        AT_K = A_K.T  # (nleft,npt)
        if has_late:
            assert A_L is not None
            AT_L = A_L.T  # (nleft,npt)
        else:
            AT_L = None

        # Density-side factors (include early secondaries; they contribute to M).
        U_sub_all = U[:, idx, :]  # (nbatch,nloc,r)
        V_sub_all = V[:, idx, :]  # (nbatch,nloc,r)

        for b0 in range(0, int(nbatch), int(bb)):
            b1 = min(int(nbatch), int(b0) + int(bb))
            nb = int(b1 - b0)
            if nb <= 0:
                continue

            U_sub = U_sub_all[int(b0) : int(b1), :, :]  # (nb,nloc,r)
            V_sub = V_sub_all[int(b0) : int(b1), :, :]  # (nb,nloc,r)

            # P = XU, Q = XV  (nb,npt,r)
            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                P = cp.matmul(X[None, :, :], U_sub)
                Q = cp.matmul(X[None, :, :], V_sub)

            out_chunk = out[int(b0) : int(b1)]

            for q0 in range(0, int(npt), int(qb)):
                q1 = min(int(npt), int(q0) + int(qb))
                qn = int(q1 - q0)
                if qn <= 0:
                    continue

                Qq = Q[:, int(q0) : int(q1), :]  # (nb,qn,r)
                # Mq = P @ Qq^T  (nb,npt,qn)
                with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                    Mq = cp.matmul(P, Qq.transpose((0, 2, 1)))

                if use_y:
                    assert Y is not None
                    Yq = Y[int(q0) : int(q1), :]  # (qn,naux)
                    with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                        Zblk = Y @ Yq.T  # (npt,qn)
                else:
                    assert Z is not None
                    Zblk = Z[:, int(q0) : int(q1)]

                Tq = Mq * Zblk[None, :, :]  # (nb,npt,qn)

                BK_q = B_K[int(q0) : int(q1), :]  # (qn,nright)
                with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                    SK = cp.matmul(Tq, BK_q[None, :, :])  # (nb,npt,nright)
                    out_chunk += cp.matmul(AT_K[None, :, :], SK)  # (nb,nleft,nright)

                if has_late:
                    assert B_L is not None
                    assert AT_L is not None
                    BL_q = B_L[int(q0) : int(q1), :]
                    with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                        SL = cp.matmul(Tq, BL_q[None, :, :])
                        out_chunk -= cp.matmul(AT_L[None, :, :], SL)

                del Qq, Mq, Zblk, Tq, BK_q, SK
                if has_late:
                    del BL_q, SL

            out[int(b0) : int(b1)] = out_chunk

            del U_sub, V_sub, P, Q, out_chunk

        del idx, X, Y, Z, A_K, B_K, CL_sub, CR_sub, CL_K, CR_K, CL_L, CR_L, U_sub_all, V_sub_all
        if has_late:
            del A_L, B_L, AT_L

    return cp.asarray(out, dtype=out_dtype)


def local_thc_J_factored_mo_batched(
    U_batch,
    V_batch,
    lthc: LocalTHCFactors,
    C_left,
    C_right,
    *,
    symmetrize: bool = False,
    batch_block: int | None = None,
    max_workspace_bytes: int = 256 * 1024 * 1024,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Batched MO contraction for the local-THC Coulomb build with factored densities.

    For each batch item ``b`` with density ``D_b = U_b V_b^T``, define the local-THC
    Coulomb matrix (with the same block ownership rule as `local_thc_J_factored`):

      J_b = J_local[D_b]

    This routine returns the MO-contracted matrices:

      out_b = C_left^T @ J_b @ C_right

    without materializing dense AO densities ``D_b`` or dense AO Coulomb matrices ``J_b``.

    Shapes
    - U_batch: (nbatch,nao,r) or (nao,r)
    - V_batch: (nbatch,nao,r) or (nao,r)
    - C_left: (nao,nleft)
    - C_right: (nao,nright)
    - out: (nbatch,nleft,nright)
    """

    cp = _require_cupy()

    policy = THCPrecisionPolicy() if policy is None else policy
    compute_dtype = cp.dtype(getattr(policy, "compute_dtype", cp.float64))
    out_dtype = cp.dtype(getattr(policy, "out_dtype", cp.float64))
    use_tf32 = bool(getattr(policy, "use_tf32", False)) and compute_dtype == cp.float32
    prefer_Y = bool(getattr(policy, "prefer_Y", False))

    U = cp.asarray(U_batch, dtype=compute_dtype)
    V = cp.asarray(V_batch, dtype=compute_dtype)
    if int(U.ndim) == 2:
        U = U[None, :, :]
    if int(V.ndim) == 2:
        V = V[None, :, :]
    if int(U.ndim) != 3 or int(V.ndim) != 3:
        raise ValueError("U_batch/V_batch must have shape (nbatch,nao,r) or (nao,r)")
    nbatch, nao, r0 = map(int, U.shape)
    nbatch2, nao2, r1 = map(int, V.shape)
    if nbatch2 != nbatch or nao2 != nao or r1 != r0:
        raise ValueError("U_batch and V_batch must have the same shape (nbatch,nao,r)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with U_batch/V_batch")

    C_left = cp.asarray(C_left, dtype=compute_dtype)
    C_right = cp.asarray(C_right, dtype=compute_dtype)
    if int(getattr(C_left, "ndim", 0)) != 2 or int(getattr(C_right, "ndim", 0)) != 2:
        raise ValueError("C_left/C_right must be 2D arrays")
    nao_l, nleft = map(int, C_left.shape)
    nao_r, nright = map(int, C_right.shape)
    if nao_l != nao or nao_r != nao:
        raise ValueError("C_left/C_right nao mismatch with U_batch/V_batch")

    max_workspace_bytes = int(max_workspace_bytes)
    if max_workspace_bytes <= 0:
        raise ValueError("max_workspace_bytes must be > 0")

    out = cp.zeros((int(nbatch), int(nleft), int(nright)), dtype=compute_dtype)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        nloc = int(idx_np.size)
        if nloc == 0:
            continue

        idx = cp.asarray(idx_np, dtype=cp.int32)
        U_sub_all = U[:, idx, :]  # (nbatch,nloc,r)
        V_sub_all = V[:, idx, :]  # (nbatch,nloc,r)

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X = getattr(cache, "X_tc") if cache is not None else getattr(blk, "X", None)
        Y = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z = None if cache is not None else getattr(blk, "Z", None)

        X = cp.asarray(X, dtype=compute_dtype)
        if int(getattr(X, "ndim", 0)) != 2 or int(X.shape[1]) != int(nloc):
            raise ValueError("LocalTHCBlock.X must have shape (npt,nlocal_ao)")
        npt = int(X.shape[0])

        use_y = (Y is not None) and (Z is None or bool(prefer_Y))
        if use_y:
            Y = cp.asarray(Y, dtype=compute_dtype)
            if int(getattr(Y, "ndim", 0)) != 2 or int(Y.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Y must have shape (npt,naux)")
        else:
            if Z is None:
                raise ValueError("LocalTHCBlock.Z is missing (provide Y or store_Z=True)")
            Z = cp.asarray(Z, dtype=compute_dtype)
            if int(getattr(Z, "ndim", 0)) != 2 or int(Z.shape[0]) != int(Z.shape[1]) or int(Z.shape[0]) != int(npt):
                raise ValueError("LocalTHCBlock.Z must have shape (npt,npt)")

        # Ownership masking on AO output pairs (same rule as local_thc_J_factored).
        n_early = int(getattr(blk, "n_early", 0))
        n_primary = int(getattr(blk, "n_primary", 0))
        if n_early < 0 or n_early > nloc:
            raise ValueError("invalid blk.n_early")
        if n_primary < 0 or (n_early + n_primary) > nloc:
            raise ValueError("invalid blk.n_primary")
        tail = int(n_early + n_primary)

        # Submatrices in this block's local AO ordering.
        CL = C_left[idx, :]  # (nloc,nleft)
        CR = C_right[idx, :]  # (nloc,nright)

        # Partition X and C_right to implement the ownership mask without forming J.
        X_P = X[:, int(n_early) : int(tail)]  # (npt,n_primary)
        X_K = X[:, int(n_early) :]  # (npt,n_primary+n_late)
        X_L = X[:, int(tail) :]  # (npt,n_late)
        CR_P = CR[int(n_early) : int(tail), :]  # (n_primary,nright)
        CR_K = CR[int(n_early) :, :]  # (n_primary+n_late,nright)

        # R_K = X_K @ CR_K, R_P = X_P @ CR_P (point-space MO values).
        with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
            R_K = X_K @ CR_K  # (npt,nright)
            R_P = X_P @ CR_P  # (npt,nright)

        # Choose a batch chunk size so temporaries (bb,npt,r) stay bounded.
        if batch_block is not None:
            bb = max(1, min(int(batch_block), int(nbatch)))
        else:
            itemsize = int(np.dtype(compute_dtype).itemsize)
            denom = int(npt) * int(r0) * int(itemsize)
            # Need at least P and Q buffers; be conservative by dividing by 4.
            bb = int(max_workspace_bytes // max(int(denom) * 4, 1))
            bb = max(1, min(int(bb), int(nbatch)))

        for b0 in range(0, int(nbatch), int(bb)):
            b1 = min(int(nbatch), int(b0) + int(bb))
            nb = int(b1 - b0)
            if nb <= 0:
                continue

            U_sub = U_sub_all[int(b0) : int(b1), :, :]  # (nb,nloc,r)
            V_sub = V_sub_all[int(b0) : int(b1), :, :]  # (nb,nloc,r)

            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                P = cp.matmul(X[None, :, :], U_sub)  # (nb,npt,r)
                Q = cp.matmul(X[None, :, :], V_sub)  # (nb,npt,r)

            m = cp.sum(P * Q, axis=2)  # (nb,npt)
            if use_y:
                assert Y is not None
                tmp = m @ Y  # (nb,naux)
                n = tmp @ Y.T  # (nb,npt)
            else:
                assert Z is not None
                n = m @ Z.T  # (nb,npt)

            # Primary rows: T_P = X_P^T @ (diag(n) @ R_K)
            S_K = n[:, :, None] * R_K[None, :, :]  # (nb,npt,nright)
            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                T_P = cp.matmul(X_P.T[None, :, :], S_K)  # (nb,n_primary,nright)

            # Late rows: T_L = X_L^T @ (diag(n) @ R_P)  (no late-late contribution)
            if int(tail) < int(nloc):
                S_P = n[:, :, None] * R_P[None, :, :]  # (nb,npt,nright)
                with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                    T_L = cp.matmul(X_L.T[None, :, :], S_P)  # (nb,n_late,nright)
            else:
                T_L = None

            T = cp.zeros((int(nb), int(nloc), int(nright)), dtype=compute_dtype)
            if int(n_primary) > 0:
                T[:, int(n_early) : int(tail), :] = T_P
            if T_L is not None:
                T[:, int(tail) :, :] = T_L

            with _maybe_tf32_ctx(cp, enabled=bool(use_tf32)):
                out[int(b0) : int(b1)] += cp.matmul(CL.T[None, :, :], T)

            del U_sub, V_sub, P, Q, m, n, S_K, T_P, T
            if T_L is not None:
                del S_P, T_L

        # Help the CuPy memory pool reuse buffers sooner.
        if bool(symmetrize):  # no-op, but keep a side effect to silence unused-param linters.
            pass

    return cp.asarray(out, dtype=out_dtype)


def local_thc_K_pairs_mo_batched(
    c_left_batch,
    c_right_batch,
    lthc: LocalTHCFactors,
    C_left,
    C_right,
    *,
    q_block: int = 64,
    batch_block: int | None = None,
    max_workspace_bytes: int = 256 * 1024 * 1024,
    symmetrize: bool = False,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Batched MO contraction for the local-THC exchange build with symmetrized rank-2 pair densities.

    For each batch item ``b``, define:

      D_b = 0.5 * (c_left_b c_right_b^T + c_right_b c_left_b^T)

    This routine returns:

      out_b = C_left^T @ K_local[D_b] @ C_right

    without materializing ``D_b`` or ``K_b`` as dense AO matrices.
    """

    cp = _require_cupy()
    policy = THCPrecisionPolicy() if policy is None else policy
    compute_dtype = cp.dtype(getattr(policy, "compute_dtype", cp.float64))

    c_left = cp.asarray(c_left_batch, dtype=compute_dtype)
    c_right = cp.asarray(c_right_batch, dtype=compute_dtype)
    if int(c_left.ndim) == 1:
        c_left = c_left[None, :]
    if int(c_right.ndim) == 1:
        c_right = c_right[None, :]
    if int(c_left.ndim) != 2 or int(c_right.ndim) != 2:
        raise ValueError("c_left_batch/c_right_batch must have shape (nbatch,nao) or (nao,)")
    nbatch, nao = map(int, c_left.shape)
    nbatch2, nao2 = map(int, c_right.shape)
    if nbatch2 != nbatch or nao2 != nao:
        raise ValueError("c_left_batch and c_right_batch must have the same shape")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with c_left_batch/c_right_batch")

    # Represent the symmetric rank-2 density as a (nao,2) factored density UV^T:
    #   U = [c_left, c_right], V = [0.5*c_right, 0.5*c_left]
    U = cp.stack((c_left, c_right), axis=2)
    V = cp.stack((0.5 * c_right, 0.5 * c_left), axis=2)

    out = local_thc_K_factored_mo_batched(
        U,
        V,
        lthc,
        C_left,
        C_right,
        q_block=int(q_block),
        batch_block=batch_block,
        max_workspace_bytes=int(max_workspace_bytes),
        policy=policy,
        tc_cache=tc_cache,
    )
    if bool(symmetrize):  # no-op, keep for API parity.
        pass
    return out


def local_thc_J(D, lthc: LocalTHCFactors, *, policy: THCPrecisionPolicy | None = None, tc_cache: dict[int, Any] | None = None):
    """Assemble global Coulomb matrix J[D] from LocalTHCFactors."""
    return local_thc_eri_apply(D, lthc, symmetrize=True, policy=policy, tc_cache=tc_cache)


def local_thc_J_factored(
    U,
    V,
    lthc: LocalTHCFactors,
    *,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Assemble global Coulomb matrix J[D] from LocalTHCFactors with factored density D = U V^T."""

    cp = _require_cupy()
    U = cp.asarray(U, dtype=cp.float64)
    V = cp.asarray(V, dtype=cp.float64)
    if int(getattr(U, "ndim", 0)) != 2 or int(getattr(V, "ndim", 0)) != 2:
        raise ValueError("U/V must have shape (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U/V must have the same shape (nao,r)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with U/V")

    J = cp.zeros((nao, nao), dtype=cp.float64)
    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        U_sub = U[idx, :]
        V_sub = V[idx, :]
        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        J_sub = thc_J_factored(U_sub, V_sub, X_use, Z_use, Y=Y_use, policy=policy)

        _mask_owned_outputs_inplace(
            J_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )
        J[idx[:, None], idx[None, :]] += J_sub

    return 0.5 * (J + J.T)

def local_thc_K_blocked(
    D,
    lthc: LocalTHCFactors,
    *,
    q_block: int = 256,
    work: THCJKWork | None = None,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
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

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        K_sub = thc_K_blocked(D_sub, X_use, Z_use, q_block=int(work.q_block), Y=Y_use, policy=policy)

        _mask_owned_outputs_inplace(
            K_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )

        K[idx[:, None], idx[None, :]] += K_sub

    return 0.5 * (K + K.T)


def local_thc_K_blocked_factored(
    U,
    V,
    lthc: LocalTHCFactors,
    *,
    q_block: int = 256,
    work: THCJKWork | None = None,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Assemble global exchange matrix K[D] from LocalTHCFactors with factored density D = U V^T."""

    cp = _require_cupy()

    if work is None:
        work = THCJKWork(q_block=int(q_block))

    U = cp.asarray(U, dtype=cp.float64)
    V = cp.asarray(V, dtype=cp.float64)
    if int(getattr(U, "ndim", 0)) != 2 or int(getattr(V, "ndim", 0)) != 2:
        raise ValueError("U/V must have shape (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U/V must have the same shape (nao,r)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with U/V")

    K = cp.zeros((nao, nao), dtype=cp.float64)
    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        U_sub = U[idx, :]
        V_sub = V[idx, :]
        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        K_sub = thc_K_blocked_factored(
            U_sub,
            V_sub,
            X_use,
            Z_use,
            q_block=int(work.q_block),
            Y=Y_use,
            policy=policy,
        )

        _mask_owned_outputs_inplace(
            K_sub,
            n_early=int(getattr(blk, "n_early", 0)),
            n_primary=int(blk.n_primary),
        )
        K[idx[:, None], idx[None, :]] += K_sub

    return 0.5 * (K + K.T)

def local_thc_JK(
    D,
    lthc: LocalTHCFactors,
    *,
    q_block: int = 256,
    work: THCJKWork | None = None,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
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

        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        J_sub, K_sub = thc_JK(D_sub, X_use, Z_use, work=work, Y=Y_use, policy=policy)

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


def local_thc_JK_factored(
    U,
    V,
    lthc: LocalTHCFactors,
    *,
    q_block: int = 256,
    work: THCJKWork | None = None,
    policy: THCPrecisionPolicy | None = None,
    tc_cache: dict[int, Any] | None = None,
):
    """Assemble global (J,K) from LocalTHCFactors with factored density D = U V^T."""

    cp = _require_cupy()

    if work is None:
        work = THCJKWork(q_block=int(q_block))

    U = cp.asarray(U, dtype=cp.float64)
    V = cp.asarray(V, dtype=cp.float64)
    if int(getattr(U, "ndim", 0)) != 2 or int(getattr(V, "ndim", 0)) != 2:
        raise ValueError("U/V must have shape (nao,r)")
    nao, r0 = map(int, U.shape)
    nao2, r1 = map(int, V.shape)
    if nao2 != nao or r1 != r0:
        raise ValueError("U/V must have the same shape (nao,r)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with U/V")

    J = cp.zeros((nao, nao), dtype=cp.float64)
    K = cp.zeros((nao, nao), dtype=cp.float64)

    for blk in lthc.blocks:
        idx_np = np.asarray(blk.ao_idx_global, dtype=np.int32).ravel()
        if int(idx_np.size) == 0:
            continue
        idx = cp.asarray(idx_np, dtype=cp.int32)
        U_sub = U[idx, :]
        V_sub = V[idx, :]
        cache = None if tc_cache is None else tc_cache.get(int(getattr(blk, "block_id", -1)))
        X_use = getattr(cache, "X_tc") if cache is not None else blk.X
        Y_use = getattr(cache, "Y_tc") if cache is not None else getattr(blk, "Y", None)
        Z_use = None if cache is not None else getattr(blk, "Z", None)
        J_sub, K_sub = thc_JK_factored(U_sub, V_sub, X_use, Z_use, work=work, Y=Y_use, policy=policy)

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

        J[idx[:, None], idx[None, :]] += J_sub
        K[idx[:, None], idx[None, :]] += K_sub

    J = 0.5 * (J + J.T)
    K = 0.5 * (K + K.T)
    return J, K


__all__ = [
    "local_thc_eri_apply",
    "local_thc_eri_apply_batched",
    "local_thc_eri_apply_pairs_mo_batched",
    "local_thc_J_factored_mo_batched",
    "local_thc_K_factored_mo_batched",
    "local_thc_K_pairs_mo_batched",
    "local_thc_J",
    "local_thc_J_factored",
    "local_thc_JK",
    "local_thc_JK_factored",
    "local_thc_K_blocked",
    "local_thc_K_blocked_factored",
]
