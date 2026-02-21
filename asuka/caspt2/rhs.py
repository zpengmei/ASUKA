"""RHS vector construction for IC-CASPT2.

Ports OpenMolcas ``mkrhs.f`` (MKRHSA through MKRHSH).
The RHS vector V_P = <P|H|0> is the coupling between the IC basis
functions and the reference wavefunction through the full Hamiltonian.
All integrals use chemists' notation: eri_mo[p,q,r,s] = (pq|rs).
"""

from __future__ import annotations

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.superindex import SuperindexMap


def build_rhs(
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    *,
    nactel: int | None = None,
) -> np.ndarray:
    """Build RHS vector for a given IC case.

    Parameters
    ----------
    case : int
        IC case number (1-13).
    eri_mo : (nmo, nmo, nmo, nmo)
        Full MO ERIs in chemists' notation: eri_mo[p,q,r,s] = (pq|rs).
    dm1, dm2 : np.ndarray
        Active 1-RDM and 2-RDM (E-operator convention).

    Returns
    -------
    rhs : (nasup * nisup,) array
        RHS vector, stored as (active_superindex, ext_superindex) flattened.
    """
    builders = {
        1: _rhs_a, 2: _rhs_bp, 3: _rhs_bm, 4: _rhs_c, 5: _rhs_d,
        6: _rhs_ep, 7: _rhs_em, 8: _rhs_fp, 9: _rhs_fm,
        10: _rhs_gp, 11: _rhs_gm, 12: _rhs_hp, 13: _rhs_hm,
    }
    if case not in builders:
        raise ValueError(f"Invalid case: {case}")
    if case == 1:
        return _rhs_a(smap, fock, eri_mo, dm1, dm2, nactel=nactel)
    if case == 4:
        return _rhs_c(smap, fock, eri_mo, dm1, dm2, nactel=nactel)
    if case == 5:
        return _rhs_d(smap, fock, eri_mo, dm1, dm2, nactel=nactel)
    return builders[case](smap, fock, eri_mo, dm1, dm2)


def _resolve_nactel(dm1: np.ndarray, nactel: int | None) -> int:
    """Resolve active electron count used in Molcas-style RHS normalizations."""
    if nactel is not None:
        return max(1, int(nactel))
    return max(1, int(round(float(np.trace(dm1)))))


def _rhs_a(smap, fock, eri_mo, dm1, dm2, *, nactel: int | None = None):
    """Case 1 (A): VJTU - 3 active + 1 inactive.

    W[ktuv(t,u,v), i] = (uv|ti) + delta(u,v)*fimo(t,i)/nactel
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    ntuv = smap.ntuv
    if ntuv == 0 or nish == 0:
        return np.zeros(ntuv * nish, dtype=np.float64)

    ao = nish
    nactel_eff = _resolve_nactel(dm1, nactel)

    # eri_mo[ao+u, ao+v, ao+t, i] = (uv|ti), shape [u, v, t, i]
    eri_block = eri_mo[ao:ao+nash, ao:ao+nash, ao:ao+nash, :nish]
    # Transpose to [t, u, v, i] then reshape to [ntuv, nish]
    rhs = eri_block.transpose(2, 0, 1, 3).reshape(ntuv, nish).copy()

    # Add delta(u,v) * fimo[ao+t, i] / nactel
    fimo_ti = fock.fimo[ao:ao+nash, :nish] / nactel_eff  # (nash, nish)
    for u in range(nash):
        idx = np.arange(nash) * nash * nash + u * nash + u
        rhs[idx, :] += fimo_ti

    return rhs.ravel()


def _rhs_bp(smap, fock, eri_mo, dm1, dm2):
    """Case 2 (B+): VJTIP - 2 active(sym) + 2 inactive(sym).

    WP[tgeu, igej] = [(it|ju) + (jt|iu)] * (1-d_tu/2) / (2*sqrt(1+d_ij))
    """
    ntgeu = smap.ntgeu
    nigej = smap.nigej
    if ntgeu == 0 or nigej == 0:
        return np.zeros(ntgeu * nigej, dtype=np.float64)

    ao = smap.orbs.nish
    rhs = np.zeros((ntgeu, nigej), dtype=np.float64)

    for p in range(ntgeu):
        t, u = smap.mtgeu[p]
        fac_tu = 0.5 if t != u else 0.25
        for q in range(nigej):
            i, j = smap.migej[q]
            val = eri_mo[i, ao+t, j, ao+u] + eri_mo[j, ao+t, i, ao+u]
            fac_ij = 1.0 / np.sqrt(1.0 + (1.0 if i == j else 0.0))
            rhs[p, q] = val * fac_tu * fac_ij

    return rhs.ravel()


def _rhs_bm(smap, fock, eri_mo, dm1, dm2):
    """Case 3 (B-): VJTIM - 2 active(asym) + 2 inactive(asym).

    WM[tgtu, igtj] = [(it|ju) - (jt|iu)] / 2
    """
    ntgtu = smap.ntgtu
    nigtj = smap.nigtj
    if ntgtu == 0 or nigtj == 0:
        return np.zeros(ntgtu * nigtj, dtype=np.float64)

    ao = smap.orbs.nish
    rhs = np.zeros((ntgtu, nigtj), dtype=np.float64)

    for p in range(ntgtu):
        t, u = smap.mtgtu[p]
        for q in range(nigtj):
            i, j = smap.migtj[q]
            val = eri_mo[i, ao+t, j, ao+u] - eri_mo[j, ao+t, i, ao+u]
            rhs[p, q] = val * 0.5

    return rhs.ravel()


def _rhs_c(smap, fock, eri_mo, dm1, dm2, *, nactel: int | None = None):
    """Case 4 (C): ATVX - 3 active + 1 virtual.

    W[ktuv(t,u,v), a] = (at|uv) + delta(u,v)*(fimo(a,t) - sum_y(ay|yt))/nactel
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nssh = smap.orbs.nssh
    ntuv = smap.ntuv
    if ntuv == 0 or nssh == 0:
        return np.zeros(ntuv * nssh, dtype=np.float64)

    ao = nish
    vo = nish + nash
    nactel_eff = _resolve_nactel(dm1, nactel)

    # eri_mo[vo+a, ao+t, ao+u, ao+v] = (at|uv), shape [a, t, u, v]
    eri_block = eri_mo[vo:vo+nssh, ao:ao+nash, ao:ao+nash, ao:ao+nash]
    # Transpose to [t, u, v, a] then reshape to [ntuv, nssh]
    rhs = eri_block.transpose(1, 2, 3, 0).reshape(ntuv, nssh).copy()

    # Correction: delta(u,v) * (fimo(a,t) - sum_y (ay|yt)) / nactel
    fimo_at = fock.fimo[vo:vo+nssh, ao:ao+nash]  # (nssh, nash)
    corr_sum = np.zeros((nssh, nash), dtype=np.float64)
    for y in range(nash):
        corr_sum += eri_mo[vo:vo+nssh, ao+y, ao+y, ao:ao+nash]
    oneadd = (fimo_at - corr_sum) / nactel_eff  # (nssh, nash) [a, t]
    oneadd_ta = oneadd.T  # (nash, nssh) [t, a]
    for u in range(nash):
        idx = np.arange(nash) * nash * nash + u * nash + u
        rhs[idx, :] += oneadd_ta

    return rhs.ravel()


def _rhs_d(smap, fock, eri_mo, dm1, dm2, *, nactel: int | None = None):
    """Case 5 (D): AIVX - mixed ia coupling.

    W1[ktu(t,u), a*nish+i] = (ai|tu) + delta(t,u)*fimo(a,i)/nactel
    W2[ktu(t,u), a*nish+i] = (ti|au)
    Active index: [W1; W2] stacked, nasup = 2*ntu.
    External index: a*nish + i.
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nssh = smap.orbs.nssh
    ntu = smap.ntu
    nasup = 2 * ntu
    nisup = nssh * nish
    if nasup == 0 or nisup == 0:
        return np.zeros(nasup * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    nactel_eff = _resolve_nactel(dm1, nactel)

    rhs = np.zeros((nasup, nisup), dtype=np.float64)

    for p in range(ntu):
        t, u = smap.mtu[p]
        for a in range(nssh):
            for i in range(nish):
                ext = a * nish + i
                w1 = eri_mo[vo+a, i, ao+t, ao+u]
                if t == u:
                    w1 += fock.fimo[vo+a, i] / nactel_eff
                w2 = eri_mo[ao+t, i, vo+a, ao+u]
                rhs[p, ext] = w1
                rhs[ntu + p, ext] = w2

    return rhs.ravel()


def _rhs_ep(smap, fock, eri_mo, dm1, dm2):
    """Case 6 (E+): VJAIP - 1 active + (virtual, sym inactive pair).

    WP[t, igej*nssh+a] = [(ai|tj) + (aj|ti)] / sqrt(2+2*d_ij)
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nssh = smap.orbs.nssh
    nigej = smap.nigej
    nisup = nssh * nigej
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_igej in range(nigej):
        i, j = smap.migej[q_igej]
        fac = 1.0 / np.sqrt(2.0 + 2.0 * (1.0 if i == j else 0.0))
        for a in range(nssh):
            ext = q_igej * nssh + a
            for t in range(nash):
                val = eri_mo[vo+a, i, ao+t, j] + eri_mo[vo+a, j, ao+t, i]
                rhs[t, ext] = val * fac

    return rhs.ravel()


def _rhs_em(smap, fock, eri_mo, dm1, dm2):
    """Case 7 (E-): VJAIM - 1 active + (virtual, asym inactive pair).

    WM[t, igtj*nssh+a] = sqrt(3/2) * [(ai|tj) - (aj|ti)]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nssh = smap.orbs.nssh
    nigtj = smap.nigtj
    nisup = nssh * nigtj
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    sq32 = np.sqrt(1.5)
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_igtj in range(nigtj):
        i, j = smap.migtj[q_igtj]
        for a in range(nssh):
            ext = q_igtj * nssh + a
            for t in range(nash):
                val = eri_mo[vo+a, i, ao+t, j] - eri_mo[vo+a, j, ao+t, i]
                rhs[t, ext] = sq32 * val

    return rhs.ravel()


def _rhs_fp(smap, fock, eri_mo, dm1, dm2):
    """Case 8 (F+): BVATP - 2 active(sym) + 2 virtual(sym).

    WP[tgeu, ageb] = [(au|bt) + (at|bu)] * (1-d_tu/2) / (2*sqrt(1+d_ab))
    """
    ntgeu = smap.ntgeu
    nageb = smap.nageb
    if ntgeu == 0 or nageb == 0:
        return np.zeros(ntgeu * nageb, dtype=np.float64)

    ao = smap.orbs.nish
    vo = smap.orbs.nish + smap.orbs.nash
    rhs = np.zeros((ntgeu, nageb), dtype=np.float64)

    for p in range(ntgeu):
        t, u = smap.mtgeu[p]
        fac_tu = 0.5 if t != u else 0.25
        for q in range(nageb):
            a, b = smap.mageb[q]
            val = eri_mo[vo+a, ao+u, vo+b, ao+t] + eri_mo[vo+a, ao+t, vo+b, ao+u]
            fac_ab = 1.0 / np.sqrt(1.0 + (1.0 if a == b else 0.0))
            rhs[p, q] = val * fac_tu * fac_ab

    return rhs.ravel()


def _rhs_fm(smap, fock, eri_mo, dm1, dm2):
    """Case 9 (F-): BVATM - 2 active(asym) + 2 virtual(asym).

    WM[tgtu, agtb] = [(au|bt) - (at|bu)] / 2
    """
    ntgtu = smap.ntgtu
    nagtb = smap.nagtb
    if ntgtu == 0 or nagtb == 0:
        return np.zeros(ntgtu * nagtb, dtype=np.float64)

    ao = smap.orbs.nish
    vo = smap.orbs.nish + smap.orbs.nash
    rhs = np.zeros((ntgtu, nagtb), dtype=np.float64)

    for p in range(ntgtu):
        t, u = smap.mtgtu[p]
        for q in range(nagtb):
            a, b = smap.magtb[q]
            val = eri_mo[vo+a, ao+u, vo+b, ao+t] - eri_mo[vo+a, ao+t, vo+b, ao+u]
            rhs[p, q] = val * 0.5

    return rhs.ravel()


def _rhs_gp(smap, fock, eri_mo, dm1, dm2):
    """Case 10 (G+): BJATQ - 1 active + (inactive, sym virtual pair).

    WP[t, ageb*nish+i] = [(at|bi) + (ai|bt)] / sqrt(2+2*d_ab)
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nageb = smap.nageb
    nisup = nish * nageb
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_ageb in range(nageb):
        a, b = smap.mageb[q_ageb]
        fac = 1.0 / np.sqrt(2.0 + 2.0 * (1.0 if a == b else 0.0))
        for i in range(nish):
            ext = q_ageb * nish + i
            for t in range(nash):
                val = eri_mo[vo+a, ao+t, vo+b, i] + eri_mo[vo+a, i, vo+b, ao+t]
                rhs[t, ext] = val * fac

    return rhs.ravel()


def _rhs_gm(smap, fock, eri_mo, dm1, dm2):
    """Case 11 (G-): BJATM - 1 active + (inactive, asym virtual pair).

    WM[t, agtb*nish+i] = sqrt(3/2) * [(at|bi) - (ai|bt)]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nagtb = smap.nagtb
    nisup = nish * nagtb
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    sq32 = np.sqrt(1.5)
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_agtb in range(nagtb):
        a, b = smap.magtb[q_agtb]
        for i in range(nish):
            ext = q_agtb * nish + i
            for t in range(nash):
                val = eri_mo[vo+a, ao+t, vo+b, i] - eri_mo[vo+a, i, vo+b, ao+t]
                rhs[t, ext] = sq32 * val

    return rhs.ravel()


def _rhs_hp(smap, fock, eri_mo, dm1, dm2):
    """Case 12 (H+): BJAIP - 2 virtual(sym) + 2 inactive(sym).

    VP[ageb, igej] = [(ai|bj) + (aj|bi)] / sqrt((1+d_ab)*(1+d_ij))
    """
    nageb = smap.nageb
    nigej = smap.nigej
    if nageb == 0 or nigej == 0:
        return np.zeros(nageb * nigej, dtype=np.float64)

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    vo = nish + nash
    rhs = np.zeros((nageb, nigej), dtype=np.float64)

    for p in range(nageb):
        a, b = smap.mageb[p]
        for q in range(nigej):
            i, j = smap.migej[q]
            val = eri_mo[vo+a, i, vo+b, j] + eri_mo[vo+a, j, vo+b, i]
            fac = 1.0 / np.sqrt(
                (1.0 + (1.0 if a == b else 0.0))
                * (1.0 + (1.0 if i == j else 0.0))
            )
            rhs[p, q] = val * fac

    return rhs.ravel()


def _rhs_hm(smap, fock, eri_mo, dm1, dm2):
    """Case 13 (H-): BJAIM - 2 virtual(asym) + 2 inactive(asym).

    VM[agtb, igtj] = sqrt(3) * [(ai|bj) - (aj|bi)]
    """
    nagtb = smap.nagtb
    nigtj = smap.nigtj
    if nagtb == 0 or nigtj == 0:
        return np.zeros(nagtb * nigtj, dtype=np.float64)

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    vo = nish + nash
    sq3 = np.sqrt(3.0)
    rhs = np.zeros((nagtb, nigtj), dtype=np.float64)

    for p in range(nagtb):
        a, b = smap.magtb[p]
        for q in range(nigtj):
            i, j = smap.migtj[q]
            val = eri_mo[vo+a, i, vo+b, j] - eri_mo[vo+a, j, vo+b, i]
            rhs[p, q] = sq3 * val

    return rhs.ravel()
