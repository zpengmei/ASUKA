"""CPU DF-native RHS construction for IC-CASPT2.

Eliminates the O(N^5) full-ERI construction by computing RHS vectors
directly from DF pair blocks via targeted GEMMs. NumPy port of
``cuda/rhs_df_cuda.py``.

Each IC case needs only specific ERI slices (e.g., (core,act|act,act)),
reconstructed on-the-fly from 4 DF pair blocks: l_it, l_ia, l_at, l_tu.
Worst-case scaling is O(N^2 * n_act^2 * n_aux) per case.
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.superindex import SuperindexMap
from asuka.mrpt2.df_pair_block import DFPairBlock

# Import the dataclass — no CuPy dependency at module level.
from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks

__all__ = ["build_rhs_df", "build_df_blocks_cpu"]


def _resolve_nactel(dm1: np.ndarray, nactel: int | None) -> int:
    if nactel is not None:
        return max(1, int(nactel))
    return max(1, int(round(float(np.trace(dm1)))))


def _df_gram_gather(
    L3: np.ndarray,
    pairs_act: np.ndarray,
    pairs_orb: np.ndarray,
    sign: int = 1,
) -> np.ndarray:
    """Compute gathered gram-matrix elements without forming the full gram.

    Given L3 of shape (n_orb, n_act, naux), computes::

        result[k, m] = g4[i_m, t_k, j_m, u_k] + sign * g4[j_m, t_k, i_m, u_k]

    where g4[i,t,j,u] = sum_P L3[i,t,P] * L3[j,u,P].

    Parameters
    ----------
    L3 : (n_orb, n_act, naux)
    pairs_act : (n_act_pairs, 2)
    pairs_orb : (n_orb_pairs, 2)
    sign : +1, -1, or 0

    Returns
    -------
    (n_act_pairs, n_orb_pairs) array
    """
    t = pairs_act[:, 0]
    u = pairs_act[:, 1]
    i = pairs_orb[:, 0]
    j = pairs_orb[:, 1]

    # Batched GEMM: M[k, a, b] = sum_P L3[a, t[k], P] * L3[b, u[k], P]
    L_t = L3[:, t, :].transpose(1, 0, 2)  # (n_act_pairs, n_orb, naux)
    L_u = L3[:, u, :].transpose(1, 0, 2)
    M = L_t @ L_u.transpose(0, 2, 1)      # (n_act_pairs, n_orb, n_orb)

    result = M[:, i, j]                    # (n_act_pairs, n_orb_pairs)
    if sign != 0:
        result = result + sign * M[:, j, i]
    return result


def build_rhs_df(
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    df: CASPT2DFBlocks,
    dm1: np.ndarray,
    dm2: np.ndarray,
    *,
    nactel: int | None = None,
) -> np.ndarray:
    """Build RHS vector for a given IC case from DF pair blocks (CPU).

    Parameters
    ----------
    case : int
        IC case number (1-13).
    df : CASPT2DFBlocks
        DF pair blocks (NumPy arrays).
    dm1, dm2 : np.ndarray
        Active 1-RDM and 2-RDM.

    Returns
    -------
    rhs : (nasup * nisup,) array
    """
    if case < 1 or case > 13:
        raise ValueError("case must be 1..13")

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntuv = int(smap.ntuv)
    ntu = int(smap.ntu)

    l_it = np.asarray(df.l_it.l_full, dtype=np.float64)
    l_ia = np.asarray(df.l_ia.l_full, dtype=np.float64)
    l_at = np.asarray(df.l_at.l_full, dtype=np.float64)
    l_tu = np.asarray(df.l_tu.l_full, dtype=np.float64)

    nasup = int(smap.nasup[case - 1])
    nisup = int(smap.nisup[case - 1])
    if nasup == 0 or nisup == 0:
        return np.zeros(0, dtype=np.float64)

    fimo = np.asarray(fock.fimo, dtype=np.float64)
    ao = nish
    vo = nish + nash
    nactel_eff = _resolve_nactel(dm1, nactel)

    # ── Case 1 (A): VJTU ──
    if case == 1:
        m = l_it @ l_tu.T                             # (nish*nash, nash*nash)
        m4 = m.reshape(nish, nash, nash, nash)         # (i,t,u,v)
        rhs = m4.transpose(1, 2, 3, 0).reshape(ntuv, nish).copy()

        fimo_ti = fimo[ao:ao + nash, :nish] / float(nactel_eff)
        t_idx = np.arange(nash)
        for u in range(nash):
            rows = t_idx * (nash * nash) + u * nash + u
            rhs[rows, :] += fimo_ti
        return rhs.ravel()

    # ── Cases 2-3 (B+/B-) ──
    if case in (2, 3):
        naux = l_it.shape[1]
        L3_it = l_it.reshape(nish, nash, naux)

        if case == 2:
            p_act = np.asarray(smap.mtgeu, dtype=np.int64)
            p_orb = np.asarray(smap.migej, dtype=np.int64)
            val = _df_gram_gather(L3_it, p_act, p_orb, sign=+1)
            t = p_act[:, 0]; u = p_act[:, 1]
            i = p_orb[:, 0]; j = p_orb[:, 1]
            fac_tu = np.where(t != u, 0.5, 0.25)
            fac_ij = 1.0 / np.sqrt(1.0 + (i == j).astype(np.float64))
            rhs = (val * fac_tu[:, None]) * fac_ij[None, :]
            return rhs.ravel()

        # case 3
        p_act = np.asarray(smap.mtgtu, dtype=np.int64)
        p_orb = np.asarray(smap.migtj, dtype=np.int64)
        rhs = 0.5 * _df_gram_gather(L3_it, p_act, p_orb, sign=-1)
        return rhs.ravel()

    # ── Case 4 (C): ATVX ──
    if case == 4:
        m = l_at @ l_tu.T                             # (nssh*nash, nash*nash)
        m4 = m.reshape(nssh, nash, nash, nash)         # (a,t,u,v)
        rhs = m4.transpose(1, 2, 3, 0).reshape(ntuv, nssh).copy()

        naux = int(df.l_tu.naux)
        a3 = l_at.reshape(nssh, nash, naux)
        b3 = l_tu.reshape(nash, nash, naux)
        corr_sum = np.einsum("ayP,ytP->at", a3, b3, optimize=True)

        fimo_at = fimo[vo:vo + nssh, ao:ao + nash]
        oneadd = (fimo_at - corr_sum) / float(nactel_eff)
        oneadd_ta = oneadd.T
        t_idx = np.arange(nash)
        for u in range(nash):
            rows = t_idx * (nash * nash) + u * nash + u
            rhs[rows, :] += oneadd_ta
        return rhs.ravel()

    # ── Case 5 (D): AIVX ──
    if case == 5:
        # W1[tu, a*i] = (ai|tu) == (tu|ia)
        m1 = l_tu @ l_ia.T                            # (nash*nash, nish*nssh)
        m14 = m1.reshape(nash, nash, nish, nssh)       # (t,u,i,a)
        w1 = m14.transpose(0, 1, 3, 2).reshape(ntu, nssh * nish)

        fimo_ai = (fimo[vo:vo + nssh, :nish] / float(nactel_eff)).reshape(nssh * nish)
        for t in range(nash):
            p = t * nash + t
            w1[p, :] += fimo_ai

        # W2[tu, a*i] = (ti|au) == (it|au)
        m2 = l_it @ l_at.T                            # (nish*nash, nssh*nash)
        m24 = m2.reshape(nish, nash, nssh, nash)       # (i,t,a,u)
        w2 = m24.transpose(1, 3, 2, 0).reshape(ntu, nssh * nish)

        rhs = np.concatenate([w1, w2], axis=0)
        return rhs.ravel()

    # ── Cases 6-7 (E+/E-) ──
    if case in (6, 7):
        m = l_ia @ l_it.T                             # (nish*nssh, nish*nash)
        m4 = m.reshape(nish, nssh, nish, nash)         # (i,a,j,t)

        if case == 6:
            migej = np.asarray(smap.migej, dtype=np.int64)
            i = migej[:, 0]
            j = migej[:, 1]
            a = np.arange(nssh, dtype=np.int64)
            t = np.arange(nash, dtype=np.int64)
            val = (
                m4[i[:, None, None], a[None, :, None], j[:, None, None], t[None, None, :]]
                + m4[j[:, None, None], a[None, :, None], i[:, None, None], t[None, None, :]]
            )  # (nigej, nssh, nash)
            fac = 1.0 / np.sqrt(2.0 + 2.0 * (i == j).astype(np.float64))
            rhs = (val * fac[:, None, None]).transpose(2, 0, 1).reshape(nash, int(smap.nigej) * nssh)
            return rhs.ravel()

        # case 7
        migtj = np.asarray(smap.migtj, dtype=np.int64)
        i = migtj[:, 0]
        j = migtj[:, 1]
        a = np.arange(nssh, dtype=np.int64)
        t = np.arange(nash, dtype=np.int64)
        val = (
            m4[i[:, None, None], a[None, :, None], j[:, None, None], t[None, None, :]]
            - m4[j[:, None, None], a[None, :, None], i[:, None, None], t[None, None, :]]
        )
        rhs = (np.sqrt(1.5) * val).transpose(2, 0, 1).reshape(nash, int(smap.nigtj) * nssh)
        return rhs.ravel()

    # ── Cases 8-9 (F+/F-) ──
    if case in (8, 9):
        naux = l_at.shape[1]
        L3_at = l_at.reshape(nssh, nash, naux)

        if case == 8:
            p_act = np.asarray(smap.mtgeu, dtype=np.int64)
            p_orb = np.asarray(smap.mageb, dtype=np.int64)
            val = _df_gram_gather(L3_at, p_act, p_orb, sign=+1)
            t = p_act[:, 0]; u = p_act[:, 1]
            a = p_orb[:, 0]; b = p_orb[:, 1]
            fac_tu = np.where(t != u, 0.5, 0.25)
            fac_ab = 1.0 / np.sqrt(1.0 + (a == b).astype(np.float64))
            rhs = (val * fac_tu[:, None]) * fac_ab[None, :]
            return rhs.ravel()

        # case 9
        p_act = np.asarray(smap.mtgtu, dtype=np.int64)
        p_orb = np.asarray(smap.magtb, dtype=np.int64)
        rhs = -0.5 * _df_gram_gather(L3_at, p_act, p_orb, sign=-1)
        return rhs.ravel()

    # ── Cases 10-11 (G+/G-) ──
    if case in (10, 11):
        m = l_at @ l_ia.T                             # (nssh*nash, nish*nssh)
        m4 = m.reshape(nssh, nash, nish, nssh)         # (a,t,i,b)

        if case == 10:
            mageb = np.asarray(smap.mageb, dtype=np.int64)
            a = mageb[:, 0]
            b = mageb[:, 1]
            i = np.arange(nish, dtype=np.int64)
            t = np.arange(nash, dtype=np.int64)
            term1 = m4[a[:, None, None], t[None, None, :], i[None, :, None], b[:, None, None]]
            term2 = m4[b[:, None, None], t[None, None, :], i[None, :, None], a[:, None, None]]
            val = term1 + term2  # (nageb, nish, nash)
            fac = 1.0 / np.sqrt(2.0 + 2.0 * (a == b).astype(np.float64))
            rhs = (val * fac[:, None, None]).transpose(2, 0, 1).reshape(nash, int(smap.nageb) * nish)
            return rhs.ravel()

        # case 11
        magtb = np.asarray(smap.magtb, dtype=np.int64)
        a = magtb[:, 0]
        b = magtb[:, 1]
        i = np.arange(nish, dtype=np.int64)
        t = np.arange(nash, dtype=np.int64)
        term1 = m4[a[:, None, None], t[None, None, :], i[None, :, None], b[:, None, None]]
        term2 = m4[b[:, None, None], t[None, None, :], i[None, :, None], a[:, None, None]]
        val = term1 - term2
        rhs = (np.sqrt(1.5) * val).transpose(2, 0, 1).reshape(nash, int(smap.nagtb) * nish)
        return rhs.ravel()

    # ── Cases 12-13 (H+/H-) ──
    if case in (12, 13):
        naux = l_ia.shape[1]
        L3_ia = l_ia.reshape(nish, nssh, naux)

        if case == 12:
            p_act = np.asarray(smap.mageb, dtype=np.int64)
            p_orb = np.asarray(smap.migej, dtype=np.int64)
            val = _df_gram_gather(L3_ia, p_act, p_orb, sign=+1)
            a = p_act[:, 0]; b = p_act[:, 1]
            i = p_orb[:, 0]; j = p_orb[:, 1]
            fac_ab = 1.0 / np.sqrt(1.0 + (a == b).astype(np.float64))
            fac_ij = 1.0 / np.sqrt(1.0 + (i == j).astype(np.float64))
            rhs = (val * fac_ab[:, None]) * fac_ij[None, :]
            return rhs.ravel()

        # case 13
        p_act = np.asarray(smap.magtb, dtype=np.int64)
        p_orb = np.asarray(smap.migtj, dtype=np.int64)
        val = _df_gram_gather(L3_ia, p_act, p_orb, sign=-1)
        rhs = np.sqrt(3.0) * val
        return rhs.ravel()

    raise NotImplementedError(f"DF RHS case {case} not implemented")


def build_df_blocks_cpu(
    B_ao: np.ndarray,
    mo_coeff: np.ndarray,
    ncore: int,
    ncas: int,
    nvirt: int,
    *,
    max_memory_mb: float = 512.0,
) -> CASPT2DFBlocks:
    """Build DF pair blocks as NumPy arrays (no CuPy dependency).

    Parameters
    ----------
    B_ao : (nao, nao, naux) ndarray
        Whitened AO DF factors.
    mo_coeff : (nao, nmo) ndarray
        MO coefficient matrix.
    ncore, ncas, nvirt : int
        Orbital partition sizes.

    Returns
    -------
    CASPT2DFBlocks with NumPy l_full arrays.
    """
    from asuka.mrpt2.df_pair_block import build_df_pair_blocks_from_df_B  # noqa: PLC0415

    B_np = np.asarray(B_ao, dtype=np.float64)
    C = np.asarray(mo_coeff, dtype=np.float64)
    if B_np.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    naux = int(B_np.shape[2])
    C_core = C[:, :ncore]
    C_act = C[:, ncore:ncore + ncas]
    C_virt = C[:, ncore + ncas:]

    def _empty_block(nx: int, ny: int) -> DFPairBlock:
        return DFPairBlock(
            nx=int(nx), ny=int(ny),
            l_full=np.zeros((int(nx) * int(ny), naux), dtype=np.float64),
            pair_norm=None,
        )

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    if ncore > 0 and ncas > 0:
        pairs.append((C_core, C_act))
        labels.append("it")
    if ncore > 0 and nvirt > 0:
        pairs.append((C_core, C_virt))
        labels.append("ia")
    if nvirt > 0 and ncas > 0:
        pairs.append((C_virt, C_act))
        labels.append("at")
    if ncas > 0:
        pairs.append((C_act, C_act))
        labels.append("tu")

    if pairs:
        built = build_df_pair_blocks_from_df_B(
            B_np, pairs,
            max_memory=int(max(1.0, float(max_memory_mb))),
            compute_pair_norm=False,
        )
        by_label = dict(zip(labels, built))
    else:
        by_label = {}

    return CASPT2DFBlocks(
        l_it=by_label.get("it", _empty_block(ncore, ncas)),
        l_ia=by_label.get("ia", _empty_block(ncore, nvirt)),
        l_at=by_label.get("at", _empty_block(nvirt, ncas)),
        l_tu=by_label.get("tu", _empty_block(ncas, ncas)),
        l_ii=None,
        l_ab=None,
    )
