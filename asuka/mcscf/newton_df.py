from __future__ import annotations

"""DF helpers for the Newton-CASSCF operator (gen_g_hop).

This module provides:
  1) A minimal eris-like container with the attributes required by
     :func:`asuka.mcscf.newton_casscf.gen_g_hop_internal`.
  2) A minimal CASSCF-like adapter object that exposes the subset of the PySCF
     CASSCF API used by the internal Newton operator.

Design goal
-----------
Tests may compare against PySCF.
"""

from dataclasses import dataclass
from typing import Any

import os
import numpy as np

from asuka.hf import df_jk as _df_jk
from asuka.hf import df_scf as _df_scf
from asuka.mcscf.orbital_grad import cayley_update
from asuka.utils.einsum_cache import cached_einsum


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def _get_xp(*arrays: Any) -> tuple[Any, bool]:
    """Return (xp, is_gpu) where xp is numpy or cupy based on array types.

    Parameters
    ----------
    *arrays : Any
        Arrays to inspect.

    Returns
    -------
    xp : module
        The array module (numpy or cupy).
    is_gpu : bool
        Whether the arrays are on GPU.
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _as_xp_f64(xp: Any, a: Any) -> Any:
    """Convert array to float64 using the specified backend.

    Parameters
    ----------
    xp : Any
        The array module (numpy or cupy).
    a : Any
        Input array.

    Returns
    -------
    Any
        Converted array.
    """
    return xp.asarray(a, dtype=xp.float64)


@dataclass(frozen=True)
class DFNewtonERIs:
    """Minimal eris-like container for `newton_casscf.gen_g_hop_internal`.

    Attributes
    ----------
    ppaa : Any
        Integrals (nmo,nmo,ncas,ncas).
    papa : Any
        Integrals (nmo,ncas,nmo,ncas).
    vhf_c : Any
        Core HF potential (nmo,nmo).
    j_pc : Any
        Core Coulomb potential (nmo,ncore).
    k_pc : Any
        Core Exchange potential (nmo,ncore).
    L_pu : Any or None
        DF factors (nmo,ncas,naux).  When provided, ``_h_op_raw`` can
        release ``ppaa``/``papa`` after the one-time cache build and
        use these smaller factors for the per-iteration contractions.
    L_pi : Any or None
        DF factors (nmo,ncore,naux).
    L_uv : Any or None
        DF factors (ncas,ncas,naux).

    Notes
    -----
    Attribute shapes match PySCF's `mc.ao2mo(mo)` ERIS object:
      - ppaa[p,q,u,v] = (p q|u v)
      - papa[p,u,q,v] = (p u|q v)
      - vhf_c[p,q]     = Veff_core[p,q] = J[D_core] - 0.5 K[D_core] in MO basis
      - j_pc[p,i]      = (p p|i i)
      - k_pc[p,i]      = (p i|i p)

    Arrays may be NumPy or CuPy depending on the build path.
    """

    ppaa: Any
    papa: Any
    vhf_c: Any
    j_pc: Any
    k_pc: Any
    L_pu: Any = None
    L_pi: Any = None
    L_uv: Any = None
    L_pq: Any = None


def build_df_newton_eris(
    B_ao: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    mixed_precision: bool = False,
    aux_block_naux: int = 0,
) -> DFNewtonERIs:
    """Build DF ERI intermediates required by the Newton-CASSCF operator.

    Parameters
    ----------
    B_ao : Any
        Density fitting tensor in mnQ ``(nao, nao, naux)`` or packed Qp
        ``(naux, nao*(nao+1)//2)`` layout.
    mo_coeff : Any
        Molecular orbital coefficients.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    mixed_precision : bool
        If True, compute L tensor and intermediate contractions in FP32 for
        ~50% memory savings and ~1.5x speedup.  Outputs (ppaa, papa) are
        cast to FP64.  vhf_c stays FP64 for energy accuracy.
    aux_block_naux : int
        If > 0, process auxiliary indices in blocks of this size instead of
        materializing the full L(nmo,nmo,naux) tensor.  Reduces peak memory
        by ~1-2 GiB for large molecules.  0 disables blocking.

    Returns
    -------
    DFNewtonERIs
        The constructed ERI container.

    Notes
    -----
    GPU-aware: if B_ao is a CuPy array, all computation stays on GPU and
    the returned DFNewtonERIs contains CuPy arrays. Otherwise uses NumPy.
    """

    xp, _is_gpu = _get_xp(B_ao, mo_coeff)
    B = _as_xp_f64(xp, B_ao)
    mo = _as_xp_f64(xp, mo_coeff)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")
    if mo.ndim != 2:
        raise ValueError("mo_coeff must have shape (nao,nmo)")
    nao, nmo = map(int, mo.shape)
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    _cdtype = xp.float32 if mixed_precision else xp.float64
    act = slice(ncore, nocc)

    # Packed-Qp path: always use aux-blocked transform to avoid ever
    # materializing full mnQ.
    if int(B.ndim) == 2:
        naux, ntri = map(int, B.shape)
        expected_ntri = int(nao * (nao + 1) // 2)
        if int(ntri) != int(expected_ntri):
            raise ValueError(
                "packed B_ao must have shape (naux, nao*(nao+1)//2); "
                f"got ntri={int(ntri)} for nao={int(nao)}"
            )
        block_q = int(aux_block_naux)
        if block_q <= 0:
            block_q = min(int(naux), 128 if bool(_is_gpu) else 64)
        return _build_df_newton_eris_blocked_qp(
            xp,
            B,
            mo,
            ncore=ncore,
            ncas=ncas,
            nmo=nmo,
            naux=naux,
            nocc=nocc,
            act=act,
            cdtype=_cdtype,
            block_size=int(block_q),
        )

    if int(B.ndim) != 3:
        raise ValueError("B_ao must have shape (nao,nao,naux) or packed (naux,ntri)")
    nao0, nao1, naux = map(int, B.shape)
    if nao0 != nao1:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    if nao != nao0:
        raise ValueError("B_ao and mo_coeff nao mismatch")

    if aux_block_naux > 0:
        return _build_df_newton_eris_blocked(
            xp, B, mo, ncore=ncore, ncas=ncas, nmo=nmo, naux=naux,
            nocc=nocc, act=act, cdtype=_cdtype,
            block_size=int(aux_block_naux),
        )

    # ---- Monolithic path (original, optionally mixed-precision) ----

    # L[p,q,Q] = sum_{mu,nu} C[mu,p] * B[mu,nu,Q] * C[nu,q]
    # Use tensordot for GPU GEMM acceleration.
    # Free intermediates eagerly to reduce peak GPU memory.
    B_c = xp.asarray(B, dtype=_cdtype) if mixed_precision else B
    mo_c = xp.asarray(mo, dtype=_cdtype) if mixed_precision else mo
    # Half-transform: L[p,q,Q] = C^T @ B @ C, fused to avoid intermediate copy.
    # Step 1: tmp[mu,Q,q] = B[mu,nu,Q] · C[nu,q]
    tmp = xp.tensordot(B_c, mo_c, axes=([1], [0]))  # (nao, naux, nmo)
    # Step 2: L_raw[p,Q*q] = C^T · tmp.reshape(nao, naux*nmo)  — single GEMM
    L_raw = mo_c.T @ tmp.reshape(nao, naux * nmo)  # (nmo, naux*nmo)
    del tmp
    if mixed_precision:
        del B_c, mo_c
    # Reshape (nmo, naux, nmo) → transpose to (nmo, nmo, naux) → contiguous
    L = xp.ascontiguousarray(L_raw.reshape(nmo, naux, nmo).transpose(0, 2, 1))
    del L_raw
    L = xp.asarray(L, dtype=_cdtype)

    L_act = xp.ascontiguousarray(L[act, act])
    L_pu = xp.ascontiguousarray(L[:, act])
    L_pi = xp.ascontiguousarray(L[:, :ncore]) if ncore else None

    # (p q|u v) = sum_Q L[p,q,Q] L[u,v,Q]
    ppaa = cached_einsum("pqQ,uvQ->pquv", L, L_act, xp=xp)
    # (p u|q v) = sum_Q L[p,u,Q] L[q,v,Q]
    papa = cached_einsum("puQ,qvQ->puqv", L_pu, L_pu, xp=xp)

    ppaa = xp.ascontiguousarray(xp.asarray(ppaa, dtype=xp.float64))
    papa = xp.ascontiguousarray(xp.asarray(papa, dtype=xp.float64))

    # j_pc[p,i] = (p p|i i), k_pc[p,i] = (p i|i p)
    L_pp = xp.ascontiguousarray(L[xp.arange(nmo), xp.arange(nmo)])  # (nmo,naux)
    if ncore:
        L_ii = xp.ascontiguousarray(L_pp[:ncore])  # (ncore,naux)
        j_pc = xp.ascontiguousarray(xp.asarray(L_pp @ L_ii.T, dtype=xp.float64))
        k_pc = xp.ascontiguousarray(xp.asarray(
            cached_einsum("piQ,piQ->pi", L[:, :ncore], L[:, :ncore], xp=xp),
            dtype=xp.float64,
        ))
    else:
        L_ii = None
        j_pc = xp.zeros((nmo, 0), dtype=xp.float64)
        k_pc = xp.zeros((nmo, 0), dtype=xp.float64)

    # vhf_c in MO basis from core density.
    # Always compute in FP64 for energy accuracy.
    if ncore:
        L_f64 = xp.asarray(L, dtype=xp.float64)
        L_ii_f64 = xp.asarray(L_ii, dtype=xp.float64) if L_ii is not None else xp.ascontiguousarray(L_f64[xp.arange(ncore), xp.arange(ncore)])
        gamma_core = L_ii_f64.sum(axis=0)  # (naux,)
        J_mo = xp.tensordot(L_f64, gamma_core, axes=([2], [0]))  # (nmo, nmo)
        K_mo = cached_einsum("piQ,qiQ->pq", L_f64[:, :ncore], L_f64[:, :ncore], xp=xp)
        vhf_c = xp.ascontiguousarray(xp.asarray(2.0 * J_mo - K_mo, dtype=xp.float64))
        del J_mo, K_mo, gamma_core, L_f64, L_ii_f64
    else:
        vhf_c = xp.zeros((nmo, nmo), dtype=xp.float64)

    # Return L slices in FP64 for downstream use.
    L_pu_f64 = xp.ascontiguousarray(xp.asarray(L_pu, dtype=xp.float64))
    L_pi_f64 = xp.ascontiguousarray(xp.asarray(L_pi, dtype=xp.float64)) if L_pi is not None else None
    L_act_f64 = xp.ascontiguousarray(xp.asarray(L_act, dtype=xp.float64))
    del L, L_pu, L_pi, L_act

    return DFNewtonERIs(
        ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc,
        L_pu=L_pu_f64, L_pi=L_pi_f64, L_uv=L_act_f64,
    )


def _build_df_newton_eris_blocked_qp(
    xp,
    B_Qp: Any,
    mo: Any,
    *,
    ncore: int,
    ncas: int,
    nmo: int,
    naux: int,
    nocc: int,
    act: slice,
    cdtype: Any,
    block_size: int,
) -> DFNewtonERIs:
    """Aux-blocked Newton ERIs directly from packed Qp DF factors.

    This avoids materializing full mnQ and unpacks only one aux chunk at a time.
    """
    from asuka.integrals.df_packed_s2 import apply_Qp_to_C_block  # noqa: PLC0415

    nao = int(mo.shape[0])
    mo_c = xp.asarray(mo, dtype=cdtype)

    ppaa = xp.zeros((nmo, nmo, ncas, ncas), dtype=xp.float64)
    papa = xp.zeros((nmo, ncas, nmo, ncas), dtype=xp.float64)
    j_pc = xp.zeros((nmo, max(ncore, 1)), dtype=xp.float64)[:, :ncore]
    k_pc = xp.zeros((nmo, max(ncore, 1)), dtype=xp.float64)[:, :ncore]
    vhf_J = xp.zeros((nmo, nmo), dtype=xp.float64)
    vhf_K = xp.zeros((nmo, nmo), dtype=xp.float64)

    L_pu_full = xp.zeros((nmo, ncas, naux), dtype=xp.float64)
    L_pi_full = xp.zeros((nmo, ncore, naux), dtype=xp.float64) if ncore else None
    L_act_full = xp.zeros((ncas, ncas, naux), dtype=xp.float64)

    # Reuse large temporaries within the AO2MO aux-block loop. This reduces CuPy
    # allocator churn and prevents transient VRAM spikes from repeated large
    # allocations when `build_df_newton_eris()` is called many times in AH/1step.
    #
    # Only enable the workspace on GPU (CuPy). CPU runs typically have much
    # smaller `block_size` and should avoid multi-GB staging buffers.
    # B_qmn_buf and X2d_buf are removed: apply_Qp_to_C_block (Tier A) reads directly
    # from packed Qp and produces (q*nao, nmo) without a (q,nao,nao) unpack step.
    _use_ws = xp is not np
    X_t_buf = None
    L_raw_buf = None
    L_blk_buf = None
    if bool(_use_ws):
        X_t_buf = xp.empty((int(nao), int(block_size), int(nmo)), dtype=cdtype)
        L_raw_buf = xp.empty((int(nmo), int(block_size) * int(nmo)), dtype=cdtype)
        L_blk_buf = xp.empty((int(nmo), int(nmo), int(block_size)), dtype=cdtype)

    for q0 in range(0, naux, block_size):
        q1 = min(q0 + block_size, naux)
        q = int(q1 - q0)
        if q <= 0:
            continue

        # Tier A: half-transform via apply_Qp_to_C_block — reads directly from packed
        # Qp storage, no (q,nao,nao) unpack intermediate.
        #   X2d[(q*mu), p] = sum_nu B[q,mu,nu] * C[nu,p]
        X2d = apply_Qp_to_C_block(B_Qp, mo_c, nao=int(nao), q0=int(q0), q_count=int(q))  # (q,nao,nmo)
        X2d = xp.asarray(X2d.reshape(int(q) * int(nao), int(nmo)), dtype=cdtype)
        X_blk = X2d.reshape(int(q), int(nao), int(nmo))

        if X_t_buf is not None:
            X_t_blk = X_t_buf[:, : int(q)]
            xp.copyto(X_t_blk, X_blk.transpose(1, 0, 2))  # (nao,q,nmo)
        else:
            X_t_blk = xp.ascontiguousarray(X_blk.transpose(1, 0, 2))  # (nao,q,nmo)
        del X_blk

        if L_raw_buf is not None:
            L_raw = L_raw_buf[:, : int(q) * int(nmo)]
            xp.matmul(mo_c.T, X_t_blk.reshape(int(nao), int(q) * int(nmo)), out=L_raw)  # (nmo, q*nmo)
        else:
            L_raw = mo_c.T @ X_t_blk.reshape(int(nao), int(q) * int(nmo))  # (nmo, q*nmo)

        if L_blk_buf is not None:
            L_blk = L_blk_buf[:, :, : int(q)]
            xp.copyto(L_blk, L_raw.reshape(int(nmo), int(q), int(nmo)).transpose(0, 2, 1))
        else:
            L_blk = xp.ascontiguousarray(L_raw.reshape(int(nmo), int(q), int(nmo)).transpose(0, 2, 1))
        del L_raw
        if X_t_buf is None:
            del X_t_blk

        L_blk_f64 = xp.asarray(L_blk, dtype=xp.float64)

        L_act_blk = xp.ascontiguousarray(L_blk_f64[act, act])  # (ncas,ncas,q)
        L_pu_blk = xp.ascontiguousarray(L_blk_f64[:, act])  # (nmo,ncas,q)

        ppaa += cached_einsum("pqQ,uvQ->pquv", L_blk_f64, L_act_blk, xp=xp)
        papa += cached_einsum("puQ,qvQ->puqv", L_pu_blk, L_pu_blk, xp=xp)

        L_pp_blk = L_blk_f64[xp.arange(nmo), xp.arange(nmo)]  # (nmo,q)
        if ncore:
            L_ii_blk = L_pp_blk[:ncore]  # (ncore,q)
            j_pc += L_pp_blk @ L_ii_blk.T
            k_pc += cached_einsum("piQ,piQ->pi", L_blk_f64[:, :ncore], L_blk_f64[:, :ncore], xp=xp)

            gamma_blk = L_ii_blk.sum(axis=0)  # (q,)
            vhf_J += xp.tensordot(L_blk_f64, gamma_blk, axes=([2], [0]))
            vhf_K += cached_einsum("piQ,qiQ->pq", L_blk_f64[:, :ncore], L_blk_f64[:, :ncore], xp=xp)

        L_pu_full[:, :, q0:q1] = L_pu_blk
        if L_pi_full is not None:
            L_pi_full[:, :, q0:q1] = xp.ascontiguousarray(L_blk_f64[:, :ncore])
        L_act_full[:, :, q0:q1] = L_act_blk

        del L_blk_f64, L_act_blk, L_pu_blk, L_pp_blk

    ppaa = xp.ascontiguousarray(ppaa)
    papa = xp.ascontiguousarray(papa)
    j_pc = xp.ascontiguousarray(j_pc)
    k_pc = xp.ascontiguousarray(k_pc)

    if ncore:
        vhf_c = xp.ascontiguousarray(2.0 * vhf_J - vhf_K)
    else:
        vhf_c = xp.zeros((nmo, nmo), dtype=xp.float64)
    del vhf_J, vhf_K

    return DFNewtonERIs(
        ppaa=ppaa,
        papa=papa,
        vhf_c=vhf_c,
        j_pc=j_pc,
        k_pc=k_pc,
        L_pu=xp.ascontiguousarray(L_pu_full),
        L_pi=xp.ascontiguousarray(L_pi_full) if L_pi_full is not None else None,
        L_uv=xp.ascontiguousarray(L_act_full),
    )


def _build_df_newton_eris_blocked(
    xp,
    B: Any,
    mo: Any,
    *,
    ncore: int,
    ncas: int,
    nmo: int,
    naux: int,
    nocc: int,
    act: slice,
    cdtype: Any,
    block_size: int,
) -> DFNewtonERIs:
    """Aux-blocked variant: accumulate ppaa/papa/vhf_c without full L tensor.

    Instead of materializing L(nmo,nmo,naux), process auxiliary indices in
    blocks of ``block_size``.  Each block computes L_blk(nmo,nmo,blk_naux)
    and accumulates the contributions to the output tensors.
    """
    nao = int(B.shape[0])
    mo_c = xp.asarray(mo, dtype=cdtype)

    ppaa = xp.zeros((nmo, nmo, ncas, ncas), dtype=xp.float64)
    papa = xp.zeros((nmo, ncas, nmo, ncas), dtype=xp.float64)
    j_pc = xp.zeros((nmo, max(ncore, 1)), dtype=xp.float64)[:, :ncore]
    k_pc = xp.zeros((nmo, max(ncore, 1)), dtype=xp.float64)[:, :ncore]
    vhf_J = xp.zeros((nmo, nmo), dtype=xp.float64)
    vhf_K = xp.zeros((nmo, nmo), dtype=xp.float64)

    # Accumulate L_pp (diagonal of L) and L_ii for j_pc across blocks.
    L_pp_full = xp.zeros((nmo, naux), dtype=xp.float64)

    # Also need L_pu and L_pi for the return value (stored across all aux).
    L_pu_full = xp.zeros((nmo, ncas, naux), dtype=xp.float64)
    L_pi_full = xp.zeros((nmo, ncore, naux), dtype=xp.float64) if ncore else None
    L_act_full = xp.zeros((ncas, ncas, naux), dtype=xp.float64)

    for q0 in range(0, naux, block_size):
        q1 = min(q0 + block_size, naux)
        B_blk = xp.asarray(B[:, :, q0:q1], dtype=cdtype)

        # Half-transform: L_blk[p,q,Q] = C^T @ B_blk @ C (fused, no transpose copy)
        blk = q1 - q0
        tmp = xp.tensordot(B_blk, mo_c, axes=([1], [0]))  # (nao, blk, nmo)
        L_raw = mo_c.T @ tmp.reshape(nao, blk * nmo)  # (nmo, blk*nmo)
        del tmp, B_blk
        L_blk = xp.ascontiguousarray(L_raw.reshape(nmo, blk, nmo).transpose(0, 2, 1))
        del L_raw

        # Upcast to FP64 for accumulation.
        L_blk_f64 = xp.asarray(L_blk, dtype=xp.float64)
        del L_blk

        # Active slices for this block.
        L_act_blk = xp.ascontiguousarray(L_blk_f64[act, act])  # (ncas, ncas, blk)
        L_pu_blk = xp.ascontiguousarray(L_blk_f64[:, act])  # (nmo, ncas, blk)

        # Accumulate ppaa and papa.
        ppaa += cached_einsum("pqQ,uvQ->pquv", L_blk_f64, L_act_blk, xp=xp)
        papa += cached_einsum("puQ,qvQ->puqv", L_pu_blk, L_pu_blk, xp=xp)

        # j_pc, k_pc, vhf_c contributions.
        L_pp_blk = L_blk_f64[xp.arange(nmo), xp.arange(nmo)]  # (nmo, blk)
        L_pp_full[:, q0:q1] = L_pp_blk
        if ncore:
            L_ii_blk = L_pp_blk[:ncore]  # (ncore, blk)
            j_pc += L_pp_blk @ L_ii_blk.T
            k_pc += cached_einsum("piQ,piQ->pi", L_blk_f64[:, :ncore], L_blk_f64[:, :ncore], xp=xp)

            gamma_blk = L_ii_blk.sum(axis=0)  # (blk,)
            vhf_J += xp.tensordot(L_blk_f64, gamma_blk, axes=([2], [0]))
            vhf_K += cached_einsum("piQ,qiQ->pq", L_blk_f64[:, :ncore], L_blk_f64[:, :ncore], xp=xp)

        # Store L slices for return value.
        L_pu_full[:, :, q0:q1] = L_pu_blk
        if L_pi_full is not None:
            L_pi_full[:, :, q0:q1] = xp.ascontiguousarray(L_blk_f64[:, :ncore])
        L_act_full[:, :, q0:q1] = L_act_blk

        del L_blk_f64, L_act_blk, L_pu_blk, L_pp_blk

    ppaa = xp.ascontiguousarray(ppaa)
    papa = xp.ascontiguousarray(papa)
    j_pc = xp.ascontiguousarray(j_pc)
    k_pc = xp.ascontiguousarray(k_pc)

    if ncore:
        vhf_c = xp.ascontiguousarray(2.0 * vhf_J - vhf_K)
    else:
        vhf_c = xp.zeros((nmo, nmo), dtype=xp.float64)
    del vhf_J, vhf_K

    return DFNewtonERIs(
        ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc,
        L_pu=xp.ascontiguousarray(L_pu_full),
        L_pi=xp.ascontiguousarray(L_pi_full) if L_pi_full is not None else None,
        L_uv=xp.ascontiguousarray(L_act_full),
    )


def build_dense_newton_eris(
    dense_gpu_builder: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    B_ao_for_vhf: Any | None = None,
    ao_eri_for_vhf: Any | None = None,
) -> DFNewtonERIs:
    """Build Newton ERI intermediates from exact (dense) ERIs via GPU builder.

    Uses ``dense_gpu_builder.build_pq_uv_eri_mat`` for ppaa and
    ``dense_gpu_builder.build_pu_wx_eri_mat`` for papa.

    Parameters
    ----------
    dense_gpu_builder : CuERIActiveSpaceDenseGPUBuilder
        Reusable GPU builder with cached preprocessing.
    mo_coeff : Any
        Full MO coefficient matrix (nao, nmo).
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    B_ao_for_vhf : Any | None, optional
        DF B-tensor for building vhf_c. If None, falls back to ao_eri_for_vhf.
    ao_eri_for_vhf : Any | None, optional
        Materialized AO ERI tensor (nao*nao, nao*nao) for building vhf_c
        when B_ao_for_vhf is None (dense, no-DF mode).

    Returns
    -------
    DFNewtonERIs
        The constructed ERI container.
    """

    try:
        import cupy as cp
    except Exception as e:
        raise RuntimeError("CuPy required for dense Newton ERIs") from e

    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas

    mo = cp.ascontiguousarray(cp.asarray(mo_coeff, dtype=cp.float64))
    nao, nmo = map(int, mo.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    C_act = cp.ascontiguousarray(mo[:, ncore:nocc])

    # ppaa[p,q,u,v] = (pq|uv) — shape (nmo*nmo, ncas*ncas)
    ppaa_flat = dense_gpu_builder.build_pq_uv_eri_mat(mo, C_act)
    ppaa = ppaa_flat.reshape(nmo, nmo, ncas, ncas)

    # papa[p,u,q,v] = (pu|qv) — shape (nmo*ncas, nmo*ncas)
    papa_flat = dense_gpu_builder.build_pu_qv_eri_mat(mo, C_act)
    papa = papa_flat.reshape(nmo, ncas, nmo, ncas)

    # j_pc[p,i] = (pp|ii), k_pc[p,i] = (pi|pi)  where i is a CORE orbital
    if ncore:
        if B_ao_for_vhf is not None:
            # DF path: compute from L[p,q,Q]
            L_pp = cp.ascontiguousarray(L[cp.arange(nmo), cp.arange(nmo)])  # (nmo,naux)
            L_ii = cp.ascontiguousarray(L_pp[:ncore])  # (ncore,naux)
            j_pc = cp.ascontiguousarray(L_pp @ L_ii.T)  # (nmo,ncore)
            k_pc = cp.ascontiguousarray(
                cached_einsum("piQ,piQ->pi", L[:, :ncore], L[:, :ncore], xp=cp)
            )
        elif ao_eri_for_vhf is not None:
            # Dense path: compute from AO ERIs
            # j_pc[p,i] = D_p^T @ eri_2d @ D_i  where D_x[μν] = C[μ,x]*C[ν,x]
            _ao_eri_2d = cp.asarray(ao_eri_for_vhf, dtype=cp.float64)
            if _ao_eri_2d.ndim == 4:
                _ao_eri_2d = _ao_eri_2d.reshape(nao * nao, nao * nao)
            # D_diag[x, μν] = C[μ,x] * C[ν,x]
            _D_all = (mo[:, :, None] * mo[:, None, :])  # (nao, nao, nmo) — outer products
            _D_all = _D_all.transpose(2, 0, 1).reshape(nmo, nao * nao)  # (nmo, nao^2)
            _D_core = _D_all[:ncore]  # (ncore, nao^2)
            _T = _ao_eri_2d @ _D_core.T  # (nao^2, ncore)
            j_pc = cp.ascontiguousarray(_D_all @ _T)  # (nmo, ncore)
            # k_pc[p,i] = K_pi^T @ eri_2d @ K_pi  where K_pi[μν] = C[μ,p]*C[ν,i]
            mo_core = mo[:, :ncore]  # (nao, ncore)
            # K_pi[μ,ν] = C[μ,p]*C[ν,i] → K[p,i,μ,ν] = C[μ,p]*C[ν,i]
            # Vectorized: K_flat[p*ncore+i, μ*nao+ν] = C[μ,p]*C[ν,i]
            # = (C ⊗ C_core)[μν, pi]
            # Reshape for batch matmul: for each i,
            # k_pc[p,i] = sum_{μλ} C[μ,p] * W_i[μ,λ] * C[λ,p]
            # where W_i[μ,λ] = sum_{νσ} C[ν,i] * eri[μν,λσ] * C[σ,i]
            k_pc = cp.zeros((nmo, ncore), dtype=cp.float64)
            _eri_4d = _ao_eri_2d.reshape(nao, nao, nao, nao)
            for i in range(ncore):
                _ci = mo_core[:, i]  # (nao,)
                # W_i[μ,λ] = sum_{νσ} C[ν,i] * eri[μ,ν,λ,σ] * C[σ,i]
                _W = cp.einsum("n,mnls,s->ml", _ci, _eri_4d, _ci, optimize=True)
                # k_pc[p,i] = C[:,p]^T @ W @ C[:,p] for all p
                _CW = mo.T @ _W  # (nmo, nao)
                k_pc[:, i] = cp.einsum("pn,pn->p", _CW, mo.T)
            del _eri_4d, _ao_eri_2d
            k_pc = cp.ascontiguousarray(k_pc)
        else:
            raise ValueError("build_dense_newton_eris requires B_ao_for_vhf or ao_eri_for_vhf when ncore > 0")
    else:
        j_pc = cp.zeros((nmo, 0), dtype=cp.float64)
        k_pc = cp.zeros((nmo, 0), dtype=cp.float64)

    # vhf_c: core Fock potential in MO basis
    if B_ao_for_vhf is not None and ncore:
        # DF path
        B = cp.asarray(B_ao_for_vhf, dtype=cp.float64)
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        use_cocc_k = str(os.environ.get("ASUKA_MCSCF_DF_K_COCC", "1")).strip().lower() not in {"0", "false", "no", "off"}
        if use_cocc_k:
            # Build J from dense D (cheap) and K from occupied-driven factorization.
            Jc, _ = _df_scf._df_JK(B, D_core, want_J=True, want_K=False)  # noqa: SLF001
            try:
                q_block = int(os.environ.get("ASUKA_DF_JK_K_QBLOCK", "128"))
            except Exception:
                q_block = 128
            # `B` can be mnQ (nao,nao,naux) or packed Qp (naux,ntri).
            naux = int(B.shape[0]) if int(getattr(B, "ndim", 0)) == 2 else int(B.shape[2])
            q_block = max(1, min(int(naux), int(q_block)))
            occ_core = cp.full((int(ncore),), 2.0, dtype=cp.float64)
            Kc = _df_jk.df_K_from_BmnQ_Cocc(B, mo_core, occ_core, q_block=int(q_block))
        else:
            Jc, Kc = _df_scf._df_JK(B, D_core, want_J=True, want_K=True)  # noqa: SLF001
        v_ao = cp.asarray(Jc - 0.5 * Kc, dtype=cp.float64)
        vhf_c = cp.ascontiguousarray(mo.T @ v_ao @ mo)
    elif ao_eri_for_vhf is not None and ncore:
        # Dense path: use materialized AO ERIs for J/K
        from asuka.hf.dense_jk import dense_JK_from_eri_mat_D  # noqa: PLC0415

        _ao_eri = cp.asarray(ao_eri_for_vhf, dtype=cp.float64)
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = dense_JK_from_eri_mat_D(_ao_eri, D_core, want_J=True, want_K=True)
        v_ao = cp.asarray(Jc - 0.5 * Kc, dtype=cp.float64)
        vhf_c = cp.ascontiguousarray(mo.T @ v_ao @ mo)
    elif ncore:
        raise ValueError("build_dense_newton_eris requires B_ao_for_vhf or ao_eri_for_vhf to build vhf_c")
    else:
        vhf_c = cp.zeros((nmo, nmo), dtype=cp.float64)

    return DFNewtonERIs(ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc)


def build_provider_newton_eris(
    eri_provider: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
) -> DFNewtonERIs:
    """Build Newton ERIs through a generic provider."""

    probe = None
    probe_fn = getattr(eri_provider, "probe_array", None)
    if callable(probe_fn):
        probe = probe_fn()
    xp, _is_gpu = _get_xp(probe, mo_coeff)
    mo = _as_xp_f64(xp, mo_coeff)

    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas
    if mo.ndim != 2:
        raise ValueError("mo_coeff must have shape (nao,nmo)")
    nao, nmo = map(int, mo.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    C_act = xp.ascontiguousarray(mo[:, ncore:nocc])
    ppaa_flat = eri_provider.build_pq_uv(mo, C_act)
    papa_flat = eri_provider.build_pu_qv(mo, C_act)
    ppaa = xp.ascontiguousarray(xp.asarray(ppaa_flat, dtype=xp.float64)).reshape(nmo, nmo, ncas, ncas)
    papa = xp.ascontiguousarray(xp.asarray(papa_flat, dtype=xp.float64)).reshape(nmo, ncas, nmo, ncas)

    if ncore:
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = eri_provider.jk(D_core, want_J=True, want_K=True)
        if Jc is None or Kc is None:  # pragma: no cover
            raise RuntimeError("provider.jk returned None while J/K were requested")
        v_ao = xp.asarray(Jc - 0.5 * Kc, dtype=xp.float64)
        vhf_c = xp.ascontiguousarray(mo.T @ v_ao @ mo)

        def _diag_in_mo_basis(v_ao_mat: Any) -> Any:
            """Return diag(mo.T @ v_ao_mat @ mo) without materializing full MO matrix."""
            v_ao_x = xp.asarray(v_ao_mat, dtype=xp.float64)
            vm = v_ao_x @ mo
            return xp.einsum("up,up->p", mo, vm, optimize=True)

        j_pc = xp.zeros((nmo, ncore), dtype=xp.float64)
        k_pc = xp.zeros((nmo, ncore), dtype=xp.float64)
        for i0 in range(0, ncore, 2):
            if i0 + 1 < ncore:
                ci = mo_core[:, i0]
                cj = mo_core[:, i0 + 1]
                Di = ci[:, None] @ ci[None, :]
                Dj = cj[:, None] @ cj[None, :]
                Ji, Ki, Jj, Kj = eri_provider.jk_multi2(Di, Dj, want_J=True, want_K=True)
                j_pc[:, i0] = _diag_in_mo_basis(Ji)
                k_pc[:, i0] = _diag_in_mo_basis(Ki)
                j_pc[:, i0 + 1] = _diag_in_mo_basis(Jj)
                k_pc[:, i0 + 1] = _diag_in_mo_basis(Kj)
            else:
                ci = mo_core[:, i0]
                Di = ci[:, None] @ ci[None, :]
                Ji, Ki = eri_provider.jk(Di, want_J=True, want_K=True)
                j_pc[:, i0] = _diag_in_mo_basis(Ji)
                k_pc[:, i0] = _diag_in_mo_basis(Ki)
    else:
        vhf_c = xp.zeros((nmo, nmo), dtype=xp.float64)
        j_pc = xp.zeros((nmo, 0), dtype=xp.float64)
        k_pc = xp.zeros((nmo, 0), dtype=xp.float64)

    return DFNewtonERIs(ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc)


class THCERIProvider:
    """ERI provider for Newton-CASSCF using THC factors.

    Implements the ``eri_provider`` interface expected by
    :func:`build_provider_newton_eris` and :class:`DFNewtonCASSCFAdapter`.

    Methods
    -------
    build_pq_uv(mo, C_act) -> (nmo*nmo, ncas*ncas)
    build_pu_qv(mo, C_act) -> (nmo*ncas, nmo*ncas)
    jk(D, want_J, want_K) -> (J, K)
    jk_multi2(D1, D2, want_J, want_K) -> (J1, K1, J2, K2)
    probe_array() -> any device array
    """

    def __init__(self, thc_factors: Any, *, q_block: int = 256):
        self._thc = thc_factors
        self._q_block = int(q_block)

    def probe_array(self) -> Any:
        return self._thc.X

    def _get_xp(self):
        xp, _ = _df_scf._get_xp(self._thc.X, self._thc.X)
        return xp

    def jk(self, D: Any, want_J: bool = True, want_K: bool = True) -> tuple[Any, Any]:
        from asuka.hf.thc_jk import thc_JK  # noqa: PLC0415

        xp = self._get_xp()
        D_dev = xp.asarray(D, dtype=xp.float64)
        J, K = thc_JK(D_dev, self._thc.X, self._thc.Z, Y=self._thc.Y)
        if not want_J:
            J = None
        if not want_K:
            K = None
        return J, K

    def jk_multi2(
        self, D1: Any, D2: Any, want_J: bool = True, want_K: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        J1, K1 = self.jk(D1, want_J=want_J, want_K=want_K)
        J2, K2 = self.jk(D2, want_J=want_J, want_K=want_K)
        return J1, K1, J2, K2

    def _mo_collocation(self, C: Any) -> Any:
        """X_MO[P,p] = sum_mu X[P,mu] * C[mu,p]."""
        xp = self._get_xp()
        X = xp.asarray(self._thc.X, dtype=xp.float64)
        C_dev = xp.asarray(C, dtype=xp.float64)
        return X @ C_dev  # (npt, ncol)

    def _z_action(self, M: Any) -> Any:
        """Compute Z @ M where Z = Y @ Y.T (lazy)."""
        xp = self._get_xp()
        thc = self._thc
        if thc.Z is not None:
            Z = xp.asarray(thc.Z, dtype=xp.float64)
            return Z @ xp.asarray(M, dtype=xp.float64)
        Y = xp.asarray(thc.Y, dtype=xp.float64)
        M_dev = xp.asarray(M, dtype=xp.float64)
        return Y @ (Y.T @ M_dev)

    def build_pq_uv(self, mo: Any, C_act: Any) -> Any:
        """(pq|uv) via THC: sum_PQ X_MO_Pp X_MO_Pq Z_PQ X_MO_Qu X_MO_Qv."""
        xp = self._get_xp()
        X_all = self._mo_collocation(mo)          # (npt, nmo)
        X_act = self._mo_collocation(C_act)        # (npt, ncas)
        nmo = int(X_all.shape[1])
        ncas = int(X_act.shape[1])

        # B_all[P, pq] = X_all[P,p] * X_all[P,q], shape (npt, nmo*nmo)
        # B_act[Q, uv] = X_act[Q,u] * X_act[Q,v], shape (npt, ncas*ncas)
        B_all = (X_all[:, :, None] * X_all[:, None, :]).reshape(-1, nmo * nmo)
        B_act = (X_act[:, :, None] * X_act[:, None, :]).reshape(-1, ncas * ncas)

        # ppaa = B_all.T @ Z @ B_act, shape (nmo*nmo, ncas*ncas)
        ZB_act = self._z_action(B_act)  # (npt, ncas*ncas)
        ppaa = B_all.T @ ZB_act
        return ppaa

    def build_pu_qv(self, mo: Any, C_act: Any) -> Any:
        """(pu|qv) via THC: sum_PQ X_MO_Pp X_MO_Pu Z_PQ X_MO_Qq X_MO_Qv."""
        xp = self._get_xp()
        X_all = self._mo_collocation(mo)      # (npt, nmo)
        X_act = self._mo_collocation(C_act)    # (npt, ncas)
        nmo = int(X_all.shape[1])
        ncas = int(X_act.shape[1])

        # B_pu[P, p*u] = X_all[P,p] * X_act[P,u], shape (npt, nmo*ncas)
        B_pu = (X_all[:, :, None] * X_act[:, None, :]).reshape(-1, nmo * ncas)

        # papa = B_pu.T @ Z @ B_pu, shape (nmo*ncas, nmo*ncas)
        ZB = self._z_action(B_pu)  # (npt, nmo*ncas)
        papa = B_pu.T @ ZB
        return papa


@dataclass
class DFNewtonCASSCFAdapter:
    """Minimal CASSCF-like adapter for `newton_casscf.gen_g_hop_internal`.

    This is intentionally small: it only implements what the internal operator
    needs. It can wrap ASUKA's DF SCF output + CI solver without importing PySCF.

    Attributes
    ----------
    df_B : Any
        Density fitting tensor (numpy or cupy).
    hcore_ao : Any
        Core Hamiltonian in AO basis.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of active electrons.
    mo_coeff : Any
        MO coefficients.
    fcisolver : Any
        FCI solver object.
    dense_gpu_builder : Any, optional
        CuERIActiveSpaceDenseGPUBuilder for dense ERIs.
    weights : list[float] | None
        State weights for SA-CASSCF.
    frozen : Any | None
        Frozen orbitals.
    internal_rotation : bool
        Whether internal rotation (active-active) is redundant.
    extrasym : Any | None
        Symmetry constraints.

    Notes
    -----
    df_B may be a CuPy array for GPU-accelerated AH. All downstream operations
    (ao2mo, update_jk_in_ah) will auto-detect and stay on GPU.

    When df_B is None, ao_eri (materialized AO ERI tensor) is used for J/K
    computation instead. This enables fully dense (no-DF) CASSCF with CUDA.
    """

    df_B: Any  # (nao,nao,naux) — numpy or cupy, can be None for dense mode
    hcore_ao: Any
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    mo_coeff: Any
    fcisolver: Any
    dense_gpu_builder: Any = None  # CuERIActiveSpaceDenseGPUBuilder, optional
    ao_eri: Any = None  # (nao*nao, nao*nao) — for dense J/K when df_B is None
    jk_provider: Any = None
    eri_provider: Any = None

    # Optional knobs (PySCF-compatible names)
    weights: list[float] | None = None
    frozen: Any | None = None
    internal_rotation: bool = False
    extrasym: Any | None = None
    mixed_precision: bool = False
    aux_block_naux: int = 0

    def _get_2e_probe(self) -> Any:
        """Return the first non-None 2e integral source for xp detection."""
        if self.jk_provider is not None:
            probe_fn = getattr(self.jk_provider, "probe_array", None)
            if callable(probe_fn):
                probe = probe_fn()
                if probe is not None:
                    return probe
        if self.eri_provider is not None:
            probe_fn = getattr(self.eri_provider, "probe_array", None)
            if callable(probe_fn):
                probe = probe_fn()
                if probe is not None:
                    return probe
        if self.df_B is not None:
            return self.df_B
        if self.ao_eri is not None:
            return self.ao_eri
        return self.hcore_ao

    def get_hcore(self) -> Any:
        """Return the core Hamiltonian in AO basis (on correct device)."""
        xp, _is_gpu = _get_xp(self._get_2e_probe(), self.hcore_ao)
        return _as_xp_f64(xp, self.hcore_ao)

    def ao2mo(self, mo_coeff: Any) -> DFNewtonERIs:
        """Construct the ERI object for the given MOs.

        Parameters
        ----------
        mo_coeff : Any
            Molecular orbital coefficients.

        Returns
        -------
        DFNewtonERIs
            The ERI container.
        """
        if self.eri_provider is not None:
            return build_provider_newton_eris(
                self.eri_provider,
                mo_coeff,
                ncore=int(self.ncore),
                ncas=int(self.ncas),
            )
        if self.dense_gpu_builder is not None:
            return build_dense_newton_eris(
                self.dense_gpu_builder,
                mo_coeff,
                ncore=int(self.ncore),
                ncas=int(self.ncas),
                B_ao_for_vhf=self.df_B,
                ao_eri_for_vhf=self.ao_eri,
            )
        if self.df_B is None:
            raise ValueError("ao2mo requires df_B or dense_gpu_builder")
        return build_df_newton_eris(
            self.df_B, mo_coeff, ncore=int(self.ncore), ncas=int(self.ncas),
            mixed_precision=bool(self.mixed_precision),
            aux_block_naux=int(self.aux_block_naux),
        )

    def uniq_var_indices(self, nmo: int, ncore: int, ncas: int, frozen: Any | None) -> np.ndarray:
        """Return boolean mask of independent orbital rotation parameters.

        Parameters
        ----------
        nmo : int
            Number of molecular orbitals.
        ncore : int
            Number of core orbitals.
        ncas : int
            Number of active orbitals.
        frozen : Any | None
            Frozen orbitals.

        Returns
        -------
        np.ndarray
            Boolean mask (nmo, nmo) where True elements are independent parameters.
        """
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True
        if bool(self.internal_rotation):
            mask[ncore:nocc, ncore:nocc][np.tril_indices(ncas, -1)] = True
        if self.extrasym is not None:
            extrasym = np.asarray(self.extrasym)
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask & extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[: int(frozen)] = False
                mask[:, : int(frozen)] = False
            else:
                frozen_idx = np.asarray(frozen, dtype=np.int32).ravel()
                mask[frozen_idx] = False
                mask[:, frozen_idx] = False
        return mask

    def pack_uniq_var(self, mat: Any) -> np.ndarray:
        """Pack a full anti-symmetric matrix into a flat independent-parameter vector.

        Parameters
        ----------
        mat : Any
            The full matrix (numpy or CuPy).

        Returns
        -------
        numpy or CuPy array
            Flattened vector of independent parameters (same backend as input).
        """
        xp, _on_gpu = _get_xp(mat)
        mat = xp.asarray(mat, dtype=xp.float64)
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        return xp.asarray(mat[idx], dtype=xp.float64)

    def unpack_uniq_var(self, v: Any) -> np.ndarray:
        """Unpack a flat independent-parameter vector into a full anti-symmetric matrix.

        Parameters
        ----------
        v : Any
            The flattened vector (numpy or CuPy).

        Returns
        -------
        numpy or CuPy array
            The full anti-symmetric matrix (nmo, nmo), same backend as input.
        """
        xp, _on_gpu = _get_xp(v)
        v = xp.asarray(v, dtype=xp.float64).ravel()
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        mat = xp.zeros((nmo, nmo), dtype=xp.float64)
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx: Any, u0: Any = 1) -> np.ndarray:
        """Apply orbital rotation `dx` to `u0`.

        Parameters
        ----------
        dx : Any
            Parameter update vector (packed).
        u0 : Any, optional
            Current rotation matrix. Defaults to 1.

        Returns
        -------
        np.ndarray
            Updated rotation matrix.
        """
        dr = self.unpack_uniq_var(dx)
        u = cayley_update(np, dr)
        return np.dot(u0, np.asarray(u, dtype=np.float64))

    def update_jk_in_ah(
        self,
        mo: Any,
        r: Any,
        casdm1: Any,
        eris: Any | None = None,
        *,
        return_gpu: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """DF analogue of PySCF `mc1step.CASSCF.update_jk_in_ah`.

        Parameters
        ----------
        mo : Any
            Molecular orbitals.
        r : Any
            Orbital rotation matrix (anti-symmetric).
        casdm1 : Any
            Active space density matrix.
        eris : Any | None, optional
            Integral object (unused, for compatibility).

        Returns
        -------
        tuple
            (va, vc) where va is active-space potential and vc is core potential update.

        Notes
        -----
        GPU-aware: uses the same array backend as self.df_B (or self.ao_eri).
        """

        _ = eris  # unused (kept for PySCF signature compatibility)

        ncore = int(self.ncore)
        ncas = int(self.ncas)
        nocc = ncore + ncas

        xp, _is_gpu = _get_xp(self._get_2e_probe())
        mo = _as_xp_f64(xp, mo)
        r = _as_xp_f64(xp, r)
        casdm1 = _as_xp_f64(xp, casdm1)

        if mo.ndim != 2:
            raise ValueError("mo must be 2D (nao,nmo)")
        nao, nmo = map(int, mo.shape)
        if nocc > nmo:
            raise ValueError("ncore+ncas exceeds nmo")
        if r.shape != (nmo, nmo):
            raise ValueError("r must be (nmo,nmo)")
        if casdm1.shape != (ncas, ncas):
            raise ValueError("casdm1 must be (ncas,ncas)")

        # dm3 = mo_core @ r_core,rest @ mo_rest^T  (+ sym)
        dm3 = mo[:, :ncore] @ r[:ncore, ncore:] @ mo[:, ncore:].T
        dm3 = dm3 + dm3.T

        # dm4 = mo_act @ casdm1 @ r_act,all @ mo^T (+ sym)
        dm4 = mo[:, ncore:nocc] @ casdm1 @ r[ncore:nocc] @ mo.T
        dm4 = dm4 + dm4.T

        if self.jk_provider is not None:
            J0, K0, J1, K1 = self.jk_provider.jk_multi2(
                dm3,
                dm3 * 2.0 + dm4,
                want_J=True,
                want_K=True,
            )
        else:
            from asuka.mcscf.jk_util import jk_multi2_from_2e_source  # noqa: PLC0415

            _df_B_xp = _as_xp_f64(xp, self.df_B) if self.df_B is not None else None
            _ao_eri_xp = _as_xp_f64(xp, self.ao_eri) if self.ao_eri is not None else None
            J0, K0, J1, K1 = jk_multi2_from_2e_source(
                _df_B_xp,
                _ao_eri_xp,
                dm3,
                dm3 * 2.0 + dm4,
                want_J=True,
                want_K=True,
            )

        v0 = xp.asarray(J0 * 2.0 - K0, dtype=xp.float64)
        v1 = xp.asarray(J1 * 2.0 - K1, dtype=xp.float64)

        mo_act = mo[:, ncore:nocc]
        mo_core = mo[:, :ncore]

        va = casdm1 @ mo_act.T @ v0 @ mo
        vc = mo_core.T @ v1 @ mo[:, ncore:]

        va_cont = xp.ascontiguousarray(xp.asarray(va, dtype=xp.float64))
        vc_cont = xp.ascontiguousarray(xp.asarray(vc, dtype=xp.float64))

        if return_gpu:
            return va_cont, vc_cont

        return _asnumpy_f64(va_cont), _asnumpy_f64(vc_cont)
