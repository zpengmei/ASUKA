from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.superindex import SuperindexMap
from asuka.mrpt2.df_pair_block import DFPairBlock


def _as_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _df_gram_gather(L3, pairs_act, pairs_orb, sign=1):
    """Compute gathered gram-matrix elements without forming the full gram.

    Given a 3-D DF block ``L3`` of shape ``(n_orb, n_act, naux)``, the full
    gram matrix in 4-D is ``g4[i, t, j, u] = Σ_P L3[i, t, P] · L3[j, u, P]``.

    This function computes, for each active pair ``(t_k, u_k)`` from
    *pairs_act* and each orbital pair ``(i_m, j_m)`` from *pairs_orb*::

        result[k, m] = g4[i_m, t_k, j_m, u_k] + sign · g4[j_m, t_k, i_m, u_k]

    via batched GEMM through the auxiliary dimension, avoiding the
    O((n_orb · n_act)²) gram matrix.

    Parameters
    ----------
    L3 : cupy.ndarray, shape ``(n_orb, n_act, naux)``
        DF block reshaped to 3-D.
    pairs_act : cupy.ndarray, shape ``(n_act_pairs, 2)``
        Indices into axis 1 of *L3*.
    pairs_orb : cupy.ndarray, shape ``(n_orb_pairs, 2)``
        Indices into axis 0 of *L3*.
    sign : {+1, -1, 0}
        ``+1`` symmetrises (add transposed), ``-1`` antisymmetrises,
        ``0`` returns only the direct term.

    Returns
    -------
    cupy.ndarray, shape ``(n_act_pairs, n_orb_pairs)``
    """
    t = pairs_act[:, 0]
    u = pairs_act[:, 1]
    i = pairs_orb[:, 0]
    j = pairs_orb[:, 1]

    # Batched GEMM: M[k, a, b] = Σ_P L3[a, t[k], P] · L3[b, u[k], P]
    L_t = L3[:, t, :].transpose(1, 0, 2)  # (n_act_pairs, n_orb, naux)
    L_u = L3[:, u, :].transpose(1, 0, 2)
    M = L_t @ L_u.transpose(0, 2, 1)       # (n_act_pairs, n_orb, n_orb)

    result = M[:, i, j]                     # (n_act_pairs, n_orb_pairs)
    if sign != 0:
        result = result + sign * M[:, j, i]
    return result


@dataclass(frozen=True)
class CASPT2DFBlocks:
    """Minimal DF blocks for C1 CASPT2 RHS construction."""

    l_it: DFPairBlock  # (i,t)
    l_ia: DFPairBlock  # (i,a)
    l_at: DFPairBlock  # (a,t)
    l_tu: DFPairBlock  # (t,u)
    # Optional blocks for DF-based Fock build (end-to-end CUDA SS path).
    l_ii: DFPairBlock | None = None  # (i,i)
    l_ab: DFPairBlock | None = None  # (a,b)


def df_blocks_to_device(blocks: CASPT2DFBlocks, cp, *, dtype=None) -> CASPT2DFBlocks:
    dtype = cp.float64 if dtype is None else dtype

    def _to_dev(b: DFPairBlock) -> DFPairBlock:
        l_full = cp.asarray(b.l_full)
        if l_full.dtype != dtype:
            l_full = l_full.astype(dtype, copy=False)
        l_full = cp.ascontiguousarray(l_full)
        return DFPairBlock(nx=int(b.nx), ny=int(b.ny), l_full=l_full, pair_norm=None)

    return CASPT2DFBlocks(
        l_it=_to_dev(blocks.l_it),
        l_ia=_to_dev(blocks.l_ia),
        l_at=_to_dev(blocks.l_at),
        l_tu=_to_dev(blocks.l_tu),
        l_ii=_to_dev(blocks.l_ii) if blocks.l_ii is not None else None,
        l_ab=_to_dev(blocks.l_ab) if blocks.l_ab is not None else None,
    )


def _resolve_nactel(dm1: np.ndarray, nactel: int | None) -> int:
    if nactel is not None:
        return max(1, int(nactel))
    return max(1, int(round(float(np.trace(dm1)))))


def build_rhs_df_cuda(
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    df: CASPT2DFBlocks,
    dm1: np.ndarray,
    dm2: np.ndarray,
    *,
    nactel: int | None = None,
    device: int | None = None,
):
    """Build RHS block (nasup, nisup) on GPU from DF pair blocks.

    Returns a CuPy array with dtype float64, C-order.
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CASPT2 CUDA RHS") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntuv = int(smap.ntuv)
    ntu = int(smap.ntu)

    if case < 1 or case > 13:
        raise ValueError("case must be 1..13")

    # Device DF blocks (ensure CuPy).
    l_it = cp.asarray(df.l_it.l_full, dtype=cp.float64)
    l_ia = cp.asarray(df.l_ia.l_full, dtype=cp.float64)
    l_at = cp.asarray(df.l_at.l_full, dtype=cp.float64)
    l_tu = cp.asarray(df.l_tu.l_full, dtype=cp.float64)

    # Fast-exit empty shapes.
    nasup = int(smap.nasup[case - 1])
    nisup = int(smap.nisup[case - 1])
    if nasup == 0 or nisup == 0:
        return cp.zeros((0, 0), dtype=cp.float64)

    # Fock one-electron corrections (CPU->GPU).
    fimo = _as_f64(fock.fimo)
    ao = nish
    vo = nish + nash
    nactel_eff = _resolve_nactel(dm1, nactel)

    if case == 1:
        # (uv|ti) == (it|uv)
        m = l_it @ l_tu.T  # (i*t, u*v)
        m4 = m.reshape(nish, nash, nash, nash)  # (i,t,u,v)
        rhs = m4.transpose(1, 2, 3, 0).reshape(ntuv, nish)

        fimo_ti = cp.asarray(fimo[ao : ao + nash, :nish] / float(nactel_eff), dtype=cp.float64)  # (t,i)
        t_idx = cp.arange(nash, dtype=cp.int64)
        for u in range(nash):
            rows = t_idx * (nash * nash) + u * nash + u
            rhs[rows, :] += fimo_ti
        return cp.ascontiguousarray(rhs)

    if case in (2, 3):
        naux = l_it.shape[1]
        L3_it = l_it.reshape(nish, nash, naux)

        if case == 2:
            p_act = cp.asarray(np.asarray(smap.mtgeu, dtype=np.int64))
            p_orb = cp.asarray(np.asarray(smap.migej, dtype=np.int64))
            val = _df_gram_gather(L3_it, p_act, p_orb, sign=+1)  # (ntgeu, nigej)
            t = p_act[:, 0]; u = p_act[:, 1]
            i = p_orb[:, 0]; j = p_orb[:, 1]
            fac_tu = cp.where(t != u, 0.5, 0.25).astype(cp.float64)
            fac_ij = (1.0 / cp.sqrt(1.0 + (i == j).astype(cp.float64))).astype(cp.float64)
            rhs = (val * fac_tu[:, None]) * fac_ij[None, :]
            return cp.ascontiguousarray(rhs)

        p_act = cp.asarray(np.asarray(smap.mtgtu, dtype=np.int64))
        p_orb = cp.asarray(np.asarray(smap.migtj, dtype=np.int64))
        rhs = 0.5 * _df_gram_gather(L3_it, p_act, p_orb, sign=-1)
        return cp.ascontiguousarray(rhs)

    if case == 4:
        m = l_at @ l_tu.T  # (a*t, u*v) -> (at|uv)
        m4 = m.reshape(nssh, nash, nash, nash)  # (a,t,u,v)
        rhs = m4.transpose(1, 2, 3, 0).reshape(ntuv, nssh)

        # corr_sum[a,t] = sum_y (a y| y t)
        naux = int(df.l_tu.naux)
        a3 = l_at.reshape(nssh, nash, naux)
        b3 = l_tu.reshape(nash, nash, naux)
        corr_sum = cp.einsum("ayP,ytP->at", a3, b3, optimize=True)

        fimo_at = cp.asarray(fimo[vo : vo + nssh, ao : ao + nash], dtype=cp.float64)  # (a,t)
        oneadd = (fimo_at - corr_sum) / float(nactel_eff)  # (a,t)
        oneadd_ta = oneadd.T  # (t,a)
        t_idx = cp.arange(nash, dtype=cp.int64)
        for u in range(nash):
            rows = t_idx * (nash * nash) + u * nash + u
            rhs[rows, :] += oneadd_ta
        return cp.ascontiguousarray(rhs)

    if case == 5:
        # W1[tu, a*i] = (ai|tu) == (tu|ia)
        m1 = l_tu @ l_ia.T  # (t*u, i*a)
        m14 = m1.reshape(nash, nash, nish, nssh)  # (t,u,i,a)
        w1 = m14.transpose(0, 1, 3, 2).reshape(ntu, nssh * nish)  # (tu, a,i)

        fimo_ai = cp.asarray(fimo[vo : vo + nssh, :nish] / float(nactel_eff), dtype=cp.float64).reshape(nssh * nish)
        for t in range(nash):
            p = t * nash + t
            w1[p, :] += fimo_ai

        # W2[tu, a*i] = (ti|au) == (it|au)
        m2 = l_it @ l_at.T  # (i*t, a*u)
        m24 = m2.reshape(nish, nash, nssh, nash)  # (i,t,a,u)
        w2 = m24.transpose(1, 3, 2, 0).reshape(ntu, nssh * nish)  # (t,u,a,i)

        rhs = cp.concatenate([w1, w2], axis=0)
        return cp.ascontiguousarray(rhs)

    if case in (6, 7):
        m = l_ia @ l_it.T  # (i*a, j*t) -> (ia|jt) == (ai|tj)
        m4 = m.reshape(nish, nssh, nish, nash)  # (i,a,j,t)

        if case == 6:
            migej = np.asarray(smap.migej, dtype=np.int64)
            i = cp.asarray(migej[:, 0], dtype=cp.int64)
            j = cp.asarray(migej[:, 1], dtype=cp.int64)
            a = cp.arange(nssh, dtype=cp.int64)
            t = cp.arange(nash, dtype=cp.int64)
            val = (
                m4[i[:, None, None], a[None, :, None], j[:, None, None], t[None, None, :]]
                + m4[j[:, None, None], a[None, :, None], i[:, None, None], t[None, None, :]]
            )  # (igej,a,t)
            fac = (1.0 / cp.sqrt(2.0 + 2.0 * (i == j).astype(cp.float64))).astype(cp.float64)  # (igej,)
            rhs = (val * fac[:, None, None]).transpose(2, 0, 1).reshape(nash, int(smap.nigej) * nssh)
            return cp.ascontiguousarray(rhs)

        migtj = np.asarray(smap.migtj, dtype=np.int64)
        i = cp.asarray(migtj[:, 0], dtype=cp.int64)
        j = cp.asarray(migtj[:, 1], dtype=cp.int64)
        a = cp.arange(nssh, dtype=cp.int64)
        t = cp.arange(nash, dtype=cp.int64)
        val = (
            m4[i[:, None, None], a[None, :, None], j[:, None, None], t[None, None, :]]
            - m4[j[:, None, None], a[None, :, None], i[:, None, None], t[None, None, :]]
        )
        rhs = (np.sqrt(1.5) * val).transpose(2, 0, 1).reshape(nash, int(smap.nigtj) * nssh)
        return cp.ascontiguousarray(rhs)

    if case in (8, 9):
        naux = l_at.shape[1]
        L3_at = l_at.reshape(nssh, nash, naux)

        if case == 8:
            p_act = cp.asarray(np.asarray(smap.mtgeu, dtype=np.int64))
            p_orb = cp.asarray(np.asarray(smap.mageb, dtype=np.int64))
            val = _df_gram_gather(L3_at, p_act, p_orb, sign=+1)  # (ntgeu, nageb)
            t = p_act[:, 0]; u = p_act[:, 1]
            a = p_orb[:, 0]; b = p_orb[:, 1]
            fac_tu = cp.where(t != u, 0.5, 0.25).astype(cp.float64)
            fac_ab = (1.0 / cp.sqrt(1.0 + (a == b).astype(cp.float64))).astype(cp.float64)
            rhs = (val * fac_tu[:, None]) * fac_ab[None, :]
            return cp.ascontiguousarray(rhs)

        p_act = cp.asarray(np.asarray(smap.mtgtu, dtype=np.int64))
        p_orb = cp.asarray(np.asarray(smap.magtb, dtype=np.int64))
        rhs = -0.5 * _df_gram_gather(L3_at, p_act, p_orb, sign=-1)
        return cp.ascontiguousarray(rhs)

    if case in (10, 11):
        m = l_at @ l_ia.T  # (a*t, i*b) -> (at|ib) == (at|bi)
        m4 = m.reshape(nssh, nash, nish, nssh)  # (a,t,i,b)

        if case == 10:
            mageb = np.asarray(smap.mageb, dtype=np.int64)
            a = cp.asarray(mageb[:, 0], dtype=cp.int64)
            b = cp.asarray(mageb[:, 1], dtype=cp.int64)
            i = cp.arange(nish, dtype=cp.int64)
            t = cp.arange(nash, dtype=cp.int64)

            term1 = m4[a[:, None, None], t[None, None, :], i[None, :, None], b[:, None, None]]  # (ab,i,t)
            term2 = m4[b[:, None, None], t[None, None, :], i[None, :, None], a[:, None, None]]
            val = term1 + term2
            fac = (1.0 / cp.sqrt(2.0 + 2.0 * (a == b).astype(cp.float64))).astype(cp.float64)
            rhs = (val * fac[:, None, None]).transpose(2, 0, 1).reshape(nash, int(smap.nageb) * nish)
            return cp.ascontiguousarray(rhs)

        magtb = np.asarray(smap.magtb, dtype=np.int64)
        a = cp.asarray(magtb[:, 0], dtype=cp.int64)
        b = cp.asarray(magtb[:, 1], dtype=cp.int64)
        i = cp.arange(nish, dtype=cp.int64)
        t = cp.arange(nash, dtype=cp.int64)

        term1 = m4[a[:, None, None], t[None, None, :], i[None, :, None], b[:, None, None]]
        term2 = m4[b[:, None, None], t[None, None, :], i[None, :, None], a[:, None, None]]
        val = term1 - term2
        rhs = (np.sqrt(1.5) * val).transpose(2, 0, 1).reshape(nash, int(smap.nagtb) * nish)
        return cp.ascontiguousarray(rhs)

    if case in (12, 13):
        naux = l_ia.shape[1]
        L3_ia = l_ia.reshape(nish, nssh, naux)

        if case == 12:
            p_act = cp.asarray(np.asarray(smap.mageb, dtype=np.int64))
            p_orb = cp.asarray(np.asarray(smap.migej, dtype=np.int64))
            val = _df_gram_gather(L3_ia, p_act, p_orb, sign=+1)  # (nageb, nigej)
            a = p_act[:, 0]; b = p_act[:, 1]
            i = p_orb[:, 0]; j = p_orb[:, 1]
            fac_ab = (1.0 / cp.sqrt(1.0 + (a == b).astype(cp.float64))).astype(cp.float64)
            fac_ij = (1.0 / cp.sqrt(1.0 + (i == j).astype(cp.float64))).astype(cp.float64)
            rhs = (val * fac_ab[:, None]) * fac_ij[None, :]
            return cp.ascontiguousarray(rhs)

        p_act = cp.asarray(np.asarray(smap.magtb, dtype=np.int64))
        p_orb = cp.asarray(np.asarray(smap.migtj, dtype=np.int64))
        val = _df_gram_gather(L3_ia, p_act, p_orb, sign=-1)
        rhs = (np.sqrt(3.0) * val).astype(cp.float64, copy=False)
        return cp.ascontiguousarray(rhs)

    raise NotImplementedError(f"DF RHS case {case} not implemented")


def build_all_rhs_df_cuda(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    df: CASPT2DFBlocks,
    dm1: np.ndarray,
    dm2: np.ndarray,
    *,
    nactel: int | None = None,
    device: int | None = None,
):
    """Build all 13 RHS blocks on GPU from DF pair blocks."""
    out = []
    for case in range(1, 14):
        out.append(
            build_rhs_df_cuda(
                case,
                smap,
                fock,
                df,
                dm1,
                dm2,
                nactel=nactel,
                device=device,
            )
        )
    return out
