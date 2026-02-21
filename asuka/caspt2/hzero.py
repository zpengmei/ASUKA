"""Zeroth-order Hamiltonian (H0) matrix construction for IC-CASPT2.

Ports OpenMolcas ``mkbmat.f`` and ``nadiag.f``.
The B matrix is the representation of H0_active in the IC basis for each case.

Key quantities:
    EASUM = sum_w epsa[w] * dm1[w,w]           (active Fock energy)
    FD[t,x] = sum_w epsa[w] * dm2[t,x,w,w]    (Fock-weighted 2-RDM trace)
    FP[p,q,r,s] = 0.5 * sum_w epsa[w] * dm3[p,q,r,s,w,w]
                                                (Fock-weighted 3-RDM trace, half)
    PREF[p,q,r,s] = 0.5 * dm2[p,q,r,s]        (half 2-RDM)
"""

from __future__ import annotations

import numpy as np

from asuka.caspt2.f3 import CASPT2CIContext, F3ContractionEngine
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.superindex import SuperindexMap


# ---------------------------------------------------------------------------
# Helper: precompute Fock-weighted quantities
# ---------------------------------------------------------------------------

def _precompute_fock_quantities(
    fock: CASPT2Fock, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray,
    nish: int, nash: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Precompute EASUM, FD, and FP.

    Returns
    -------
    easum : float
        sum_w epsa[w] * dm1[w,w]
    fd : (nash, nash)
        FD[t,x] = sum_w epsa[w] * dm2[t,x,w,w]
    fp : (nash, nash, nash, nash)
        FP[p,q,r,s] = 0.5 * sum_w epsa[w] * dm3[p,q,r,s,w,w]
    """
    epsa = fock.epsa  # (nash,) diagonal of F in active block

    # EASUM
    easum = float(np.sum(epsa * np.diag(dm1[:nash, :nash])))

    # FD[t,x] = sum_w epsa[w] * dm2[t,x,w,w]
    fd = np.einsum('w,txww->tx', epsa, dm2[:nash, :nash, :nash, :nash])

    # FP[p,q,r,s] = 0.5 * sum_w epsa[w] * dm3[p,q,r,s,w,w]
    if dm3 is not None and dm3.size > 0:
        fp = 0.5 * np.einsum('w,pqrsww->pqrs', epsa,
                              dm3[:nash, :nash, :nash, :nash, :nash, :nash])
    else:
        fp = np.zeros((nash, nash, nash, nash), dtype=np.float64)

    return easum, fd, fp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bmat(
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    *,
    ci_context: CASPT2CIContext | None = None,
) -> np.ndarray:
    """Build zeroth-order Hamiltonian matrix B for a given IC case.

    The B matrix is the active-superindex representation of Dyall's H0 in the
    IC basis.  For most cases (B±, D–H±) it involves only the Fock-weighted
    RDM intermediates (EASUM, FD, FP) and the 1-/2-RDMs.  Cases A and C
    additionally require F3 (DELTA3) contractions and therefore need a valid
    ``ci_context``.

    Parameters
    ----------
    case : int
        IC case number (1–13).
    smap : SuperindexMap
        Precomputed index tables.
    fock : CASPT2Fock
        MO-basis Fock matrices (provides ``epsa`` and ``fifa``).
    dm1, dm2, dm3 : np.ndarray
        Active-space RDMs in E-operator convention.
    ci_context : CASPT2CIContext | None
        Required for cases 1 (A) and 4 (C) to compute F3 contributions
        without an explicit 4-RDM.  Ignored for other cases.

    Returns
    -------
    bmat : (nasup, nasup) array
        Zeroth-order Hamiltonian matrix in the active superindex basis.
    """
    builders = {
        1: _bmat_a,
        2: _bmat_bp,
        3: _bmat_bm,
        4: _bmat_c,
        5: _bmat_d,
        6: _bmat_ep,
        7: _bmat_em,
        8: _bmat_fp,
        9: _bmat_fm,
        10: _bmat_gp,
        11: _bmat_gm,
        12: _bmat_hp,
        13: _bmat_hm,
    }
    if case not in builders:
        raise ValueError(f"Invalid case: {case}")
    if case == 1:
        return _bmat_a(smap, fock, dm1, dm2, dm3, ci_context=ci_context)
    if case == 4:
        return _bmat_c(smap, fock, dm1, dm2, dm3, ci_context=ci_context)
    return builders[case](smap, fock, dm1, dm2, dm3)


def compute_e0(fock: CASPT2Fock, dm1: np.ndarray, nish: int, nash: int) -> float:
    """Compute E0 = EASUM = sum_w epsa[w] * dm1[w,w].

    This is the active-space contribution to E0 in Dyall's Hamiltonian.
    The inactive orbital energies are handled separately via ext_energies.
    """
    epsa = fock.epsa
    return float(np.sum(epsa * np.diag(dm1[:nash, :nash])))


# ---------------------------------------------------------------------------
# Case E (ICASE=6,7): VJAI
# ---------------------------------------------------------------------------

def _bmat_ep(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 6 (E+): VJAIP.

    OpenMolcas formula (MKBE):
        BE(t,x) = -FD(t,x) + (EASUM - epsa[x] - epsa[t])*dm1[t,x]
                  + 2*d(t,x)*epsa[x]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    if nash == 0:
        return np.empty((0, 0), dtype=np.float64)

    epsa = fock.epsa
    easum, fd, _ = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((nash, nash), dtype=np.float64)
    for t in range(nash):
        for x in range(nash):
            val = -fd[t, x]
            val += (easum - epsa[x] - epsa[t]) * dm1[t, x]
            if t == x:
                val += 2.0 * epsa[x]
            bmat[t, x] = val
    return bmat


def _bmat_em(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 7 (E-): VJAIM. Same B matrix as E+."""
    return _bmat_ep(smap, fock, dm1, dm2, dm3)


# ---------------------------------------------------------------------------
# Case G (ICASE=10,11): BJAT
# ---------------------------------------------------------------------------

def _bmat_gp(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 10 (G+): BJATQ.

    OpenMolcas formula (MKBG):
        BG(t,x) = FD(t,x) - EASUM*dm1[t,x]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    if nash == 0:
        return np.empty((0, 0), dtype=np.float64)

    easum, fd, _ = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((nash, nash), dtype=np.float64)
    for t in range(nash):
        for x in range(nash):
            bmat[t, x] = fd[t, x] - easum * dm1[t, x]
    return bmat


def _bmat_gm(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 11 (G-): BJATM. Same B matrix as G+."""
    return _bmat_gp(smap, fock, dm1, dm2, dm3)


# ---------------------------------------------------------------------------
# Helper: raw BB(tu, xy) before ± splitting (from Molcas MKBB)
# ---------------------------------------------------------------------------

def _raw_bb(t: int, u: int, x: int, y: int,
            dm1: np.ndarray, dm2: np.ndarray,
            epsa: np.ndarray, easum: float,
            fd: np.ndarray, fp: np.ndarray) -> float:
    """Raw BB(tu, xy) before ± symmetrization.

    OpenMolcas formula (MKBB):
        BB(tu,xy) = 4*(FP[x,t,y,u] - (EASUM-Et-Eu-Ex-Ey)*PREF[x,t,y,u])
          + 4*d(x,t)*((EASUM-Et-Ey-Eu)*dm1[y,u] - FD[y,u])
          + 4*d(y,u)*((EASUM-Et-Ey-Ex)*dm1[x,t] - FD[x,t])
          - 2*d(y,t)*((EASUM-Et-Eu-Ex)*dm1[x,u] - FD[x,u])
          - 2*d(x,u)*((EASUM-Et-Eu-Ey)*dm1[y,t] - FD[y,t])
          + 8*d(x,t)*d(y,u)*(Et+Ey)
          - 4*d(x,u)*d(y,t)*(Et+Ex)

    Note: PREF[x,t,y,u] = 0.5*dm2[x,t,y,u], so 4*PREF = 2*dm2.
    """
    et, eu, ex, ey = epsa[t], epsa[u], epsa[x], epsa[y]
    atuxy = easum - et - eu - ex - ey

    # Leading 2-body term: 4*(FP - ATUXY*PREF) where PREF = 0.5*dm2
    val = 4.0 * (fp[x, t, y, u] - atuxy * 0.5 * dm2[x, t, y, u])

    # Delta correction terms
    if x == t:
        val += 4.0 * ((easum - et - ey - eu) * dm1[y, u] - fd[y, u])
        if y == u:
            val += 8.0 * (et + ey)
    if y == u:
        val += 4.0 * ((easum - et - ey - ex) * dm1[x, t] - fd[x, t])
    if y == t:
        val -= 2.0 * ((easum - et - eu - ex) * dm1[x, u] - fd[x, u])
        if x == u:
            val -= 4.0 * (et + ex)
    if x == u:
        val -= 2.0 * ((easum - et - eu - ey) * dm1[y, t] - fd[y, t])
    return val


# ---------------------------------------------------------------------------
# Cases B+/B- (ICASE=2,3): VJTI
# ---------------------------------------------------------------------------

def _bmat_bp(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 2 (B+): VJTIP - symmetric.

    BBP(tu,xy) = BB(tu,xy) + BB(tu,yx)  for t>=u, x>=y.
    """
    ntgeu = smap.ntgeu
    if ntgeu == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    epsa = fock.epsa
    easum, fd, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((ntgeu, ntgeu), dtype=np.float64)
    for p in range(ntgeu):
        t, u = smap.mtgeu[p]
        for q in range(p, ntgeu):
            x, y = smap.mtgeu[q]
            val = (_raw_bb(t, u, x, y, dm1, dm2, epsa, easum, fd, fp)
                   + _raw_bb(t, u, y, x, dm1, dm2, epsa, easum, fd, fp))
            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


def _bmat_bm(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 3 (B-): VJTIM - antisymmetric.

    BBM(tu,xy) = BB(tu,xy) - BB(tu,yx)  for t>u, x>y.
    """
    ntgtu = smap.ntgtu
    if ntgtu == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    epsa = fock.epsa
    easum, fd, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((ntgtu, ntgtu), dtype=np.float64)
    for p in range(ntgtu):
        t, u = smap.mtgtu[p]
        for q in range(p, ntgtu):
            x, y = smap.mtgtu[q]
            val = (_raw_bb(t, u, x, y, dm1, dm2, epsa, easum, fd, fp)
                   - _raw_bb(t, u, y, x, dm1, dm2, epsa, easum, fd, fp))
            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


# ---------------------------------------------------------------------------
# Case D (ICASE=5): AIVX - block-structured
# ---------------------------------------------------------------------------

def _bmat_d(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
            dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 5 (D): AIVX.

    OpenMolcas formula (MKBD), 2x2 block structure:
        BD11(tu,xy) = 4*(FP[u,t,x,y] + (Et+Ex-EASUM)*PREF[u,t,x,y])
                    + 2*d(x,t)*(FD[u,y] + (Et-EASUM)*dm1[u,y])
        BD21 = BD12 = -0.5 * BD11
        BD22(tu,xy) = -2*(FP[x,t,u,y] + (Et+Ex-EASUM)*PREF[x,t,u,y])
                    + 2*d(x,t)*(FD[u,y] + (Ex-EASUM)*dm1[u,y])
    """
    ntu = smap.ntu
    nasup = 2 * ntu
    if nasup == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    epsa = fock.epsa
    easum, fd, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((nasup, nasup), dtype=np.float64)
    for p in range(ntu):
        t, u = smap.mtu[p]
        for q in range(ntu):
            x, y = smap.mtu[q]

            et, ex = epsa[t], epsa[x]
            pref_utxy = 0.5 * dm2[u, t, x, y]
            pref_xtuy = 0.5 * dm2[x, t, u, y]

            # Block (1,1)
            bd11 = 4.0 * (fp[u, t, x, y] + (et + ex - easum) * pref_utxy)
            if x == t:
                bd11 += 2.0 * (fd[u, y] + (et - easum) * dm1[u, y])
            bmat[p, q] = bd11

            # Block (2,1) = Block (1,2) = -0.5 * BD11
            bd21 = -0.5 * bd11
            bmat[ntu + p, q] = bd21
            bmat[p, ntu + q] = bd21

            # Block (2,2)
            bd22 = -2.0 * (fp[x, t, u, y] + (et + ex - easum) * pref_xtuy)
            if x == t:
                bd22 += 2.0 * (fd[u, y] + (ex - easum) * dm1[u, y])
            bmat[ntu + p, ntu + q] = bd22

    return bmat


# ---------------------------------------------------------------------------
# Helper: raw BF(tu, xy) before ± splitting (from Molcas MKBF)
# ---------------------------------------------------------------------------

def _raw_bf(t: int, u: int, x: int, y: int,
            dm2: np.ndarray, easum: float,
            fp: np.ndarray) -> float:
    """Raw BF(tu, xy) = 4*(FP[t,x,u,y] - EASUM*PREF[t,x,u,y]).

    OpenMolcas formula (MKBF):
        BF(tu,xy) = 4*(FP[t,x,u,y] - EASUM*0.5*dm2[t,x,u,y])
    """
    return 4.0 * (fp[t, x, u, y] - easum * 0.5 * dm2[t, x, u, y])


# ---------------------------------------------------------------------------
# Cases F+/F- (ICASE=8,9): BVAT
# ---------------------------------------------------------------------------

def _bmat_fp(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 8 (F+): BVATP - symmetric.

    BFP(tu,xy) = BF(tu,xy) + BF(tu,yx)  for t>=u, x>=y.
    """
    ntgeu = smap.ntgeu
    if ntgeu == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    easum, _, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((ntgeu, ntgeu), dtype=np.float64)
    for p in range(ntgeu):
        t, u = smap.mtgeu[p]
        for q in range(p, ntgeu):
            x, y = smap.mtgeu[q]
            val = (_raw_bf(t, u, x, y, dm2, easum, fp)
                   + _raw_bf(t, u, y, x, dm2, easum, fp))
            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


def _bmat_fm(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 9 (F-): BVATM - antisymmetric.

    BFM(tu,xy) = BF(tu,xy) - BF(tu,yx)  for t>u, x>y.
    """
    ntgtu = smap.ntgtu
    if ntgtu == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    easum, _, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)

    bmat = np.zeros((ntgtu, ntgtu), dtype=np.float64)
    for p in range(ntgtu):
        t, u = smap.mtgtu[p]
        for q in range(p, ntgtu):
            x, y = smap.mtgtu[q]
            val = (_raw_bf(t, u, x, y, dm2, easum, fp)
                   - _raw_bf(t, u, y, x, dm2, easum, fp))
            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


# ---------------------------------------------------------------------------
# Case A (ICASE=1): VJTU
# ---------------------------------------------------------------------------

def _bmat_a(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
            dm2: np.ndarray, dm3: np.ndarray,
            ci_context: CASPT2CIContext | None = None) -> np.ndarray:
    """Case 1 (A): VJTU.

    OpenMolcas formula (MKBA): B starts from SA times energy factor,
    then adds FP/FD delta corrections and F3 (Fock-weighted 3-RDM) terms.

    BA(tuv,xyz) = (Ey+Eu+Ex+Et-EASUM)*SA(tuv,xyz)
                - F3_contribution
                + FP/FD delta corrections
    """
    ntuv = smap.ntuv
    if ntuv == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    epsa = fock.epsa
    easum, fd, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)
    if ci_context is None:
        raise ValueError(
            "Case A B-matrix requires ci_context (DRT + CSF CI vector) to build F3 contribution."
        )
    if int(ci_context.drt.norb) != nash:
        raise ValueError(
            f"ci_context.drt.norb={int(ci_context.drt.norb)} incompatible with nash={nash}"
        )
    f3_engine = F3ContractionEngine(ci_context, epsa)

    # First, build SA (the S matrix for case A)
    from asuka.caspt2.overlap import _smat_a
    sa = _smat_a(smap, dm1, dm2, dm3)

    bmat = np.zeros((ntuv, ntuv), dtype=np.float64)
    for p in range(ntuv):
        t, u, v = smap.mtuv[p]
        for q in range(p, ntuv):
            x, y, z = smap.mtuv[q]

            # Energy factor times S matrix
            val = (epsa[y] + epsa[u] + epsa[x] + epsa[t] - easum) * sa[p, q]

            # OpenMolcas MKBA_F3 contribution (DELTA3):
            #   BA(tuv,xyz) += -F3[v,u,x,t,y,z]
            val -= f3_engine.f3_case_a(v, u, x, t, y, z, dm2, dm3, fd, fp)

            # Delta corrections (from MKBA_DP):
            if t == x:
                val += 4.0 * (fp[v, u, y, z] - epsa[t] * 0.5 * dm2[v, u, y, z])
                if y == u:
                    val += 2.0 * (fd[v, z] - (epsa[t] + epsa[u]) * dm1[v, z])
            if x == u:
                val -= 2.0 * (fp[v, t, y, z] - epsa[u] * 0.5 * dm2[v, t, y, z])
                if y == t:
                    val -= (fd[v, z] - (epsa[t] + epsa[u]) * dm1[v, z])
            if y == t:
                val -= 2.0 * (fp[v, u, x, z] - epsa[t] * 0.5 * dm2[v, u, x, z])
            if y == u:
                val -= 2.0 * (fp[v, z, x, t] - epsa[u] * 0.5 * dm2[v, z, x, t])

            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


# ---------------------------------------------------------------------------
# Case C (ICASE=4): ATVX
# ---------------------------------------------------------------------------

def _bmat_c(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
            dm2: np.ndarray, dm3: np.ndarray,
            ci_context: CASPT2CIContext | None = None) -> np.ndarray:
    """Case 4 (C): ATVX.

    OpenMolcas formula (MKBC):
    BC(tuv,xyz) = (Ey+Eu-EASUM)*SC(tuv,xyz)
                + F3_contribution
                + FP/FD delta corrections
    """
    ntuv = smap.ntuv
    if ntuv == 0:
        return np.empty((0, 0), dtype=np.float64)

    nash = smap.orbs.nash
    nish = smap.orbs.nish
    epsa = fock.epsa
    easum, fd, fp = _precompute_fock_quantities(fock, dm1, dm2, dm3, nish, nash)
    if ci_context is None:
        raise ValueError(
            "Case C B-matrix requires ci_context (DRT + CSF CI vector) to build F3 contribution."
        )
    if int(ci_context.drt.norb) != nash:
        raise ValueError(
            f"ci_context.drt.norb={int(ci_context.drt.norb)} incompatible with nash={nash}"
        )
    f3_engine = F3ContractionEngine(ci_context, epsa)

    from asuka.caspt2.overlap import _smat_c
    sc = _smat_c(smap, dm1, dm2, dm3)

    bmat = np.zeros((ntuv, ntuv), dtype=np.float64)
    for p in range(ntuv):
        t, u, v = smap.mtuv[p]
        for q in range(p, ntuv):
            x, y, z = smap.mtuv[q]

            # Energy factor times S matrix
            val = (epsa[y] + epsa[u] - easum) * sc[p, q]

            # OpenMolcas MKBC_F3 contribution (DELTA3):
            #   BC(tuv,xyz) += +F3[v,u,t,x,y,z]
            val += f3_engine.f3_case_c(v, u, t, x, y, z, dm2, dm3, fd, fp)

            # Delta corrections (from MKBC_DP):
            if y == u:
                val += 2.0 * (fp[v, z, t, x] - epsa[u] * 0.5 * dm2[v, z, t, x])
            if y == x:
                val += 2.0 * (fp[v, u, t, z] - epsa[y] * 0.5 * dm2[v, u, t, z])
            if t == u:
                val += 2.0 * (fp[v, x, y, z] - epsa[u] * 0.5 * dm2[v, x, y, z])
                if y == x:
                    val += (fd[v, z] - (epsa[u] + epsa[y]) * dm1[v, z])

            bmat[p, q] = val
            bmat[q, p] = val
    return bmat


# ---------------------------------------------------------------------------
# Cases H+/H- (ICASE=12,13): BJAI - diagonal with orbital energies
# ---------------------------------------------------------------------------

def _bmat_hp(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 12 (H+): BJAIP - S = identity, B diagonal.

    OpenMolcas NADIAG stores BD(a>=b) = eps_a + eps_b for this case.
    Combined with ID(i>=j) = -(eps_i + eps_j), the denominator is:
        (eps_a + eps_b) - (eps_i + eps_j)
    """
    nageb = smap.nageb
    if nageb == 0:
        return np.empty((0, 0), dtype=np.float64)

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    vo = nish + nash

    bmat = np.zeros((nageb, nageb), dtype=np.float64)
    for p in range(nageb):
        a, b = smap.mageb[p]
        bmat[p, p] = fock.fifa[vo + a, vo + a] + fock.fifa[vo + b, vo + b]
    return bmat


def _bmat_hm(smap: SuperindexMap, fock: CASPT2Fock, dm1: np.ndarray,
             dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 13 (H-): BJAIM - S = identity, B diagonal."""
    nagtb = smap.nagtb
    if nagtb == 0:
        return np.empty((0, 0), dtype=np.float64)

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    vo = nish + nash

    bmat = np.zeros((nagtb, nagtb), dtype=np.float64)
    for p in range(nagtb):
        a, b = smap.magtb[p]
        bmat[p, p] = fock.fifa[vo + a, vo + a] + fock.fifa[vo + b, vo + b]
    return bmat
