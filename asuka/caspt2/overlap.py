"""Overlap (S) matrix construction for the 13-case IC-CASPT2 basis.

Ports OpenMolcas ``mksmat.f`` routines (MKSA through MKSG) and
``sbdiag.f`` for joint S/B diagonalization with linear-dependence removal.

RDM conventions (E-operator):
    dm1[p,q] = <E_pq>
    dm2[p,q,r,s] = <E_pq E_rs>
    dm3[p,q,r,s,t,u] = <E_pq E_rs E_tu>
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.superindex import SuperindexMap


@dataclass(frozen=True)
class SBDecomposition:
    """Result of S-B joint diagonalization for one IC case."""

    s_eigvals: np.ndarray      # S eigenvalues before truncation
    transform: np.ndarray      # (nasup, nindep) transformation to orthonormal basis
    nindep: int                # number of independent basis functions
    b_diag: np.ndarray         # (nindep,) diagonal B in orthonormal basis


def sbdiag(
    smat: np.ndarray,
    bmat: np.ndarray,
    *,
    threshold: float = 1e-10,
    threshold_norm: float | None = None,
    threshold_s: float = 1e-8,
) -> SBDecomposition:
    """Joint diagonalization of S and B matrices with linear-dependence removal.

    Mirrors OpenMolcas ``sbdiag.f``:
    1. Diagonal pre-screen/scaling by S_ii (THRSHN)
    2. Diagonalize scaled S and remove small eigenvalues (THRSHS)
    3. Form orthonormal transform C such that C^T S C = I
    4. Transform B to orthonormal basis and diagonalize

    Parameters
    ----------
    threshold : float
        Backward-compatible alias for ``threshold_norm``.
    threshold_norm : float | None
        Diagonal-norm threshold (Molcas THRSHN).
    threshold_s : float
        Eigenvalue threshold on scaled S (Molcas THRSHS).
    """
    smat = np.asarray(smat, dtype=np.float64)
    bmat = np.asarray(bmat, dtype=np.float64)
    n = smat.shape[0]
    if threshold_norm is None:
        threshold_norm = float(threshold)

    if n == 0:
        return SBDecomposition(
            s_eigvals=np.empty(0, dtype=np.float64),
            transform=np.empty((0, 0), dtype=np.float64),
            nindep=0,
            b_diag=np.empty(0, dtype=np.float64),
        )

    # Symmetrize
    smat = 0.5 * (smat + smat.T)
    bmat = 0.5 * (bmat + bmat.T)

    # Molcas-style scaling factors from diagonal S elements.
    # The tiny index-dependent factor follows sbdiag.f behavior and helps
    # avoid exact degeneracies in the scaled metric.
    sdiag = np.diag(smat).copy()
    idx = np.arange(1, n + 1, dtype=np.float64)
    sca = np.zeros(n, dtype=np.float64)
    mask_diag = sdiag > float(threshold_norm)
    sca[mask_diag] = (1.0 + 3.0e-6 * idx[mask_diag]) / np.sqrt(sdiag[mask_diag])

    # Step 1: diagonalize scaled metric S' = D S D.
    smat_scaled = (sca[:, None] * smat) * sca[None, :]
    smat_scaled = 0.5 * (smat_scaled + smat_scaled.T)
    s_eigvals, u_s = np.linalg.eigh(smat_scaled)

    # Step 2: remove linear dependencies in scaled S.
    mask = s_eigvals >= float(threshold_s)
    nindep = int(mask.sum())
    if nindep == 0:
        return SBDecomposition(
            s_eigvals=s_eigvals,
            transform=np.empty((n, 0), dtype=np.float64),
            nindep=0,
            b_diag=np.empty(0, dtype=np.float64),
        )

    u_ind = u_s[:, mask]             # (n, nindep)
    s_ind = s_eigvals[mask]          # (nindep,)

    # Step 3: C = D * U * s^{-1/2}  (C^T S C = I).
    s_invsqrt = 1.0 / np.sqrt(s_ind)     # (nindep,)
    x_scaled = u_ind * s_invsqrt[None, :]  # (n, nindep)
    x = sca[:, None] * x_scaled

    # Step 4: Transform B into orthonormal basis
    b_orth = x.T @ bmat @ x
    b_orth = 0.5 * (b_orth + b_orth.T)

    # Step 5: Diagonalize transformed B
    b_eigvals, u_b = np.linalg.eigh(b_orth)

    # Full transform: original basis -> diagonalized basis
    transform = x @ u_b

    return SBDecomposition(
        s_eigvals=s_eigvals,
        transform=np.asarray(transform, dtype=np.float64, order="C"),
        nindep=nindep,
        b_diag=np.asarray(b_eigvals, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# S matrix builders per case
# ---------------------------------------------------------------------------

def build_smat(
    case: int,
    smap: SuperindexMap,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
) -> np.ndarray:
    """Build overlap matrix S for a given IC case.

    Parameters
    ----------
    case : int
        IC case number (1-13).
    dm1, dm2, dm3 : np.ndarray
        Active-space RDMs in E-operator convention.

    Returns
    -------
    S : (nasup, nasup) array
    """
    builders = {
        1: _smat_a,
        2: _smat_bp,
        3: _smat_bm,
        4: _smat_c,
        5: _smat_d,
        6: _smat_ep,
        7: _smat_em,
        8: _smat_fp,
        9: _smat_fm,
        10: _smat_gp,
        11: _smat_gm,
        12: _smat_hp,
        13: _smat_hm,
    }
    if case not in builders:
        raise ValueError(f"Invalid case: {case}")
    return builders[case](smap, dm1, dm2, dm3)


# ---------------------------------------------------------------------------
# Helper: +-symmetrization for pair-indexed cases (B, F)
# ---------------------------------------------------------------------------

def _symmetrize_pairs(raw_all: np.ndarray, smap: SuperindexMap, is_plus: bool) -> np.ndarray:
    """Project raw all-pairs S matrix to ordered pairs via +-symmetrization.

    Following OpenMolcas mksmat.f convention:
        S_plus[P(t>=u), Q(x>=y)] = raw[ktu[t,u], ktu[x,y]] + raw[ktu[t,u], ktu[y,x]]
        S_minus[P(t>u), Q(x>y)]  = raw[ktu[t,u], ktu[x,y]] - raw[ktu[t,u], ktu[y,x]]
    """
    if is_plus:
        nordered = smap.ntgeu
        mordered = smap.mtgeu
    else:
        nordered = smap.ntgtu
        mordered = smap.mtgtu

    if nordered == 0:
        return np.empty((0, 0), dtype=np.float64)

    ktu = smap.ktu
    proj = np.zeros((nordered, nordered), dtype=np.float64)

    for p in range(nordered):
        t, u = mordered[p]
        itu = ktu[t, u]
        for q in range(p, nordered):
            x, y = mordered[q]
            ixy = ktu[x, y]
            iyx = ktu[y, x]
            stuxy = raw_all[itu, ixy]
            stuyx = raw_all[itu, iyx]
            if is_plus:
                val = stuxy + stuyx
            else:
                val = stuxy - stuyx
            proj[p, q] = val
            proj[q, p] = val

    return proj


# ---------------------------------------------------------------------------
# Case A (ICASE=1): VJTU -- 3 active + 1 inactive
# ---------------------------------------------------------------------------

def _smat_a(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 1 (A): VJTU.

    From OpenMolcas MKSA + MKSA_DP:
        SA(tuv, xyz) = -G3[v,u,x,t,y,z]
                       - d(y,u)*G2[v,z,x,t]
                       - d(y,t)*G2[v,u,x,z]
                       - d(x,u)*G2[v,t,y,z]
                       - d(x,u)*d(y,t)*G1[v,z]
                       + 2*d(t,x)*G2[v,u,y,z]
                       + 2*d(t,x)*d(y,u)*G1[v,z]
    """
    ntuv = smap.ntuv
    if ntuv == 0:
        return np.empty((0, 0), dtype=np.float64)

    smat = np.zeros((ntuv, ntuv), dtype=np.float64)
    for p in range(ntuv):
        t, u, v = smap.mtuv[p]
        for q in range(p, ntuv):
            x, y, z = smap.mtuv[q]

            val = -dm3[v, u, x, t, y, z]

            if y == u:
                val -= dm2[v, z, x, t]
            if y == t:
                val -= dm2[v, u, x, z]
            if x == u:
                val -= dm2[v, t, y, z]
                if y == t:
                    val -= dm1[v, z]
            if t == x:
                val += 2.0 * dm2[v, u, y, z]
                if y == u:
                    val += 2.0 * dm1[v, z]

            smat[p, q] = val
            smat[q, p] = val

    return smat


# ---------------------------------------------------------------------------
# Case B (ICASE=2,3): VJTI+/- -- 2 active + 2 inactive
# ---------------------------------------------------------------------------

def _build_sb_raw(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Build raw SB matrix over all active pairs.

    From OpenMolcas MKSB:
        SB(tu, xy) = 2*G2[x,t,y,u]
                     - 4*d(x,t)*G1[y,u] + 8*d(x,t)*d(y,u)
                     - 4*d(y,u)*G1[x,t]
                     + 2*d(y,t)*G1[x,u]
                     + 2*d(x,u)*G1[y,t] - 4*d(x,u)*d(y,t)
    """
    ntu = smap.ntu
    if ntu == 0:
        return np.empty((0, 0), dtype=np.float64)

    raw = np.zeros((ntu, ntu), dtype=np.float64)
    for p in range(ntu):
        t, u = smap.mtu[p]
        for q in range(p, ntu):
            x, y = smap.mtu[q]

            val = 2.0 * dm2[x, t, y, u]

            if x == t:
                val -= 4.0 * dm1[y, u]
                if y == u:
                    val += 8.0
            if y == u:
                val -= 4.0 * dm1[x, t]
            if y == t:
                val += 2.0 * dm1[x, u]
            if x == u:
                val += 2.0 * dm1[y, t]
                if y == t:
                    val -= 4.0

            raw[p, q] = val
            raw[q, p] = val

    return raw


def _smat_bp(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 2 (B+): VJTIP -- symmetric active pair, symmetric inactive pair."""
    raw = _build_sb_raw(smap, dm1, dm2)
    return _symmetrize_pairs(raw, smap, is_plus=True)


def _smat_bm(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 3 (B-): VJTIM -- antisymmetric active pair, antisymmetric inactive pair."""
    raw = _build_sb_raw(smap, dm1, dm2)
    return _symmetrize_pairs(raw, smap, is_plus=False)


# ---------------------------------------------------------------------------
# Case C (ICASE=4): ATVX -- 3 active + 1 virtual
# ---------------------------------------------------------------------------

def _smat_c(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 4 (C): ATVX.

    From OpenMolcas MKSC + MKSC_DP:
        SC(tuv, xyz) = G3[v,u,t,x,y,z]
                       + d(y,u)*G2[v,z,t,x]
                       + d(y,x)*G2[v,u,t,z]
                       + d(t,u)*G2[v,x,y,z]
                       + d(t,u)*d(y,x)*G1[v,z]
    """
    ntuv = smap.ntuv
    if ntuv == 0:
        return np.empty((0, 0), dtype=np.float64)

    smat = np.zeros((ntuv, ntuv), dtype=np.float64)
    for p in range(ntuv):
        t, u, v = smap.mtuv[p]
        for q in range(p, ntuv):
            x, y, z = smap.mtuv[q]

            val = dm3[v, u, t, x, y, z]

            if y == u:
                val += dm2[v, z, t, x]
            if y == x:
                val += dm2[v, u, t, z]
            if t == u:
                val += dm2[v, x, y, z]
                if y == x:
                    val += dm1[v, z]

            smat[p, q] = val
            smat[q, p] = val

    return smat


# ---------------------------------------------------------------------------
# Case D (ICASE=5): AIVX -- mixed ia coupling
# ---------------------------------------------------------------------------

def _smat_d(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 5 (D): AIVX.

    2x2 block structure per (tu, xy) all-pairs.  From OpenMolcas MKSD:
        SD(tu1, xy1) =  2*(G2[u,t,x,y] + d(x,t)*G1[u,y])
        SD(tu2, xy1) =   -(G2[u,t,x,y] + d(x,t)*G1[u,y])
        SD(tu1, xy2) =   -(G2[u,t,x,y] + d(x,t)*G1[u,y])
        SD(tu2, xy2) =  -G2[x,t,u,y] + 2*d(x,t)*G1[u,y]
    """
    ntu = smap.ntu
    nasup = 2 * ntu
    if nasup == 0:
        return np.empty((0, 0), dtype=np.float64)

    smat = np.zeros((nasup, nasup), dtype=np.float64)
    for p in range(ntu):
        t, u = smap.mtu[p]
        for q in range(ntu):
            x, y = smap.mtu[q]

            gutxy = dm2[u, t, x, y]
            gxtuy = dm2[x, t, u, y]

            s11 = 2.0 * gutxy
            s22 = -gxtuy

            if x == t:
                duy = dm1[u, y]
                s11 += 2.0 * duy
                s22 += 2.0 * duy

            smat[p, q] = s11                         # block (1,1)
            smat[ntu + p, q] = -0.5 * s11            # block (2,1)
            smat[p, ntu + q] = -0.5 * s11            # block (1,2)
            smat[ntu + p, ntu + q] = s22              # block (2,2)

    return smat


# ---------------------------------------------------------------------------
# Case E (ICASE=6,7): VJAI+/- -- 1 active
# ---------------------------------------------------------------------------

def _smat_ep(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 6 (E+): VJAIP.

    From OpenMolcas MKSE:  SE(t, x) = 2*d(t,x) - G1[t,x]
    """
    nash = smap.orbs.nash
    if nash == 0:
        return np.empty((0, 0), dtype=np.float64)
    return 2.0 * np.eye(nash, dtype=np.float64) - dm1[:nash, :nash].copy()


def _smat_em(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 7 (E-): VJAIM.  Same S matrix as E+."""
    return _smat_ep(smap, dm1, dm2, dm3)


# ---------------------------------------------------------------------------
# Case F (ICASE=8,9): BVAT+/- -- 2 active + 2 virtual
# ---------------------------------------------------------------------------

def _build_sf_raw(smap: SuperindexMap, dm2: np.ndarray) -> np.ndarray:
    """Build raw SF matrix over all active pairs.

    From OpenMolcas MKSF:  SF(tu, xy) = 4*PREF(tx,uy) = 2*G2[t,x,u,y]
    """
    ntu = smap.ntu
    if ntu == 0:
        return np.empty((0, 0), dtype=np.float64)

    raw = np.zeros((ntu, ntu), dtype=np.float64)
    for p in range(ntu):
        t, u = smap.mtu[p]
        for q in range(p, ntu):
            x, y = smap.mtu[q]
            val = 2.0 * dm2[t, x, u, y]
            raw[p, q] = val
            raw[q, p] = val

    return raw


def _smat_fp(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 8 (F+): BVATP."""
    raw = _build_sf_raw(smap, dm2)
    return _symmetrize_pairs(raw, smap, is_plus=True)


def _smat_fm(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 9 (F-): BVATM."""
    raw = _build_sf_raw(smap, dm2)
    return _symmetrize_pairs(raw, smap, is_plus=False)


# ---------------------------------------------------------------------------
# Case G (ICASE=10,11): BJAT+/- -- 1 active
# ---------------------------------------------------------------------------

def _smat_gp(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 10 (G+): BJATQ.

    From OpenMolcas MKSG:  SG(t, x) = G1[t,x]
    """
    nash = smap.orbs.nash
    if nash == 0:
        return np.empty((0, 0), dtype=np.float64)
    return dm1[:nash, :nash].copy()


def _smat_gm(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 11 (G-): BJATM.  Same S matrix as G+."""
    return _smat_gp(smap, dm1, dm2, dm3)


# ---------------------------------------------------------------------------
# Case H (ICASE=12,13): BJAI+/- -- purely external
# ---------------------------------------------------------------------------

def _smat_hp(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 12 (H+): BJAIP.  S = identity."""
    nageb = smap.nageb
    if nageb == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.eye(nageb, dtype=np.float64)


def _smat_hm(smap: SuperindexMap, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Case 13 (H-): BJAIM.  S = identity."""
    nagtb = smap.nagtb
    if nagtb == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.eye(nagtb, dtype=np.float64)
