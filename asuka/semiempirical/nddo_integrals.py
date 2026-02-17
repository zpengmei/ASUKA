"""Two-center NDDO integral assembly using local-frame rotational invariants.

Computes the 22 unique two-center NDDO integrals in the local diatomic frame
(bond along z-axis), then rotates to the molecular coordinate system using
direction cosine matrices.

Reference: MOPAC mndod.F90 (reppd, rotatd, rotmat)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .basis import nao_for_Z
from .multipole import MultipoleParams

# Sign factors for the 22 rotational invariants (MOPAC convention, 0-indexed)
_NRI = np.array([
    1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
    -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
], dtype=float)
_TWO_CENTER_EINSUM_PATHS: Dict[Tuple[int, int], object] = {}


# ---------------------------------------------------------------------------
# Local-frame integral computation (MOPAC reppd)
# ---------------------------------------------------------------------------

def _reppd(R: float, mp_A: MultipoleParams, mp_B: MultipoleParams,
           si: bool, sj: bool) -> Tuple[np.ndarray, float]:
    """Compute 22 local-frame rotational invariants.

    Follows MOPAC mndod.F90:reppd exactly. All quantities in Bohr/Hartree.

    Parameters
    ----------
    R : interatomic distance in Bohr
    mp_A, mp_B : multipole parameters (am/ad/aq are screening radii = 0.5/am_mopac)
    si, sj : True if atom A, B has p orbitals

    Returns
    -------
    ri : (22,) rotational invariants in Hartree (nri signs applied)
    gab : core-core gamma in Hartree
    """
    ri = np.zeros(22)
    am_A, am_B = mp_A.am, mp_B.am

    # Core-core gamma (same screening as electron-electron for AM1)
    aee = (am_A + am_B) ** 2
    gab = 1.0 / math.sqrt(R * R + aee)
    rsq = R * R

    if not si and not sj:
        # H-H
        ri[0] = 1.0 / math.sqrt(rsq + aee)

    elif si and not sj:
        # Heavy-H
        da = mp_A.dd
        qa = mp_A.qq * 2.0
        ade = (mp_A.ad + am_B) ** 2
        aqe = (mp_A.aq + am_B) ** 2

        s1 = math.sqrt(rsq + aee)
        s2 = math.sqrt((R + da) ** 2 + ade)
        s3 = math.sqrt((R - da) ** 2 + ade)
        s4 = math.sqrt((R + qa) ** 2 + aqe)
        s5 = math.sqrt((R - qa) ** 2 + aqe)
        s6 = math.sqrt(rsq + aqe)
        s7 = math.sqrt(rsq + aqe + qa * qa)

        ee = 1.0 / s1
        ri[0] = ee
        ri[1] = 0.5 / s2 - 0.5 / s3
        ri[2] = ee + 0.25 / s4 + 0.25 / s5 - 0.5 / s6
        ri[3] = ee + 0.5 / s7 - 0.5 / s6
        ri[:4] *= _NRI[:4]

    elif not si and sj:
        # H-Heavy
        db = mp_B.dd
        qb = mp_B.qq * 2.0
        aed = (am_A + mp_B.ad) ** 2
        aeq = (am_A + mp_B.aq) ** 2

        s1 = math.sqrt(rsq + aee)
        s2 = math.sqrt((R - db) ** 2 + aed)
        s3 = math.sqrt((R + db) ** 2 + aed)
        s4 = math.sqrt((R - qb) ** 2 + aeq)
        s5 = math.sqrt((R + qb) ** 2 + aeq)
        s6 = math.sqrt(rsq + aeq)
        s7 = math.sqrt(rsq + aeq + qb * qb)

        ee = 1.0 / s1
        ri[0] = ee
        ri[4] = 0.5 / s2 - 0.5 / s3
        ri[10] = ee + 0.25 / s4 + 0.25 / s5 - 0.5 / s6
        ri[11] = ee + 0.5 / s7 - 0.5 / s6
        ri[0] *= _NRI[0]
        ri[4] *= _NRI[4]
        ri[10] *= _NRI[10]
        ri[11] *= _NRI[11]

    else:
        # Heavy-Heavy
        _reppd_heavy_heavy(R, mp_A, mp_B, ri)
        ri *= _NRI

    return ri, gab


def _reppd_heavy_heavy(R: float, mp_A: MultipoleParams, mp_B: MultipoleParams,
                        ri: np.ndarray) -> None:
    """Fill ri[0:22] for Heavy-Heavy case (before nri signs).

    Direct translation of MOPAC mndod.F90 lines 583-796.
    """
    am_A, ad_A, aq_A = mp_A.am, mp_A.ad, mp_A.aq
    am_B, ad_B, aq_B = mp_B.am, mp_B.ad, mp_B.aq
    da = mp_A.dd
    db = mp_B.dd
    qa2 = mp_A.qq * 2.0
    qb2 = mp_B.qq * 2.0

    aee = (am_A + am_B) ** 2
    ade = (ad_A + am_B) ** 2
    aqe = (aq_A + am_B) ** 2
    aed = (am_A + ad_B) ** 2
    aeq = (am_A + aq_B) ** 2
    axx = (ad_A + ad_B) ** 2
    adq = (ad_A + aq_B) ** 2
    aqd = (aq_A + ad_B) ** 2
    aqq = (aq_A + aq_B) ** 2

    r = R
    rsq = r * r

    # Compute 72 sqrt arguments (MOPAC arg(1..72) → a[0..71])
    a = np.empty(72)
    a[0] = rsq + aee
    a[1] = (r + da) ** 2 + ade
    a[2] = (r - da) ** 2 + ade
    a[3] = (r - qa2) ** 2 + aqe
    a[4] = (r + qa2) ** 2 + aqe
    a[5] = rsq + aqe
    a[6] = a[5] + qa2 * qa2
    a[7] = (r - db) ** 2 + aed
    a[8] = (r + db) ** 2 + aed
    a[9] = (r - qb2) ** 2 + aeq
    a[10] = (r + qb2) ** 2 + aeq
    a[11] = rsq + aeq
    a[12] = a[11] + qb2 * qb2
    a[13] = rsq + axx + (da - db) ** 2
    a[14] = rsq + axx + (da + db) ** 2
    a[15] = (r + da - db) ** 2 + axx
    a[16] = (r - da + db) ** 2 + axx
    a[17] = (r - da - db) ** 2 + axx
    a[18] = (r + da + db) ** 2 + axx
    a[19] = (r + da) ** 2 + adq
    a[20] = a[19] + qb2 * qb2
    a[21] = (r - da) ** 2 + adq
    a[22] = a[21] + qb2 * qb2
    a[23] = (r - db) ** 2 + aqd
    a[24] = a[23] + qa2 * qa2
    a[25] = (r + db) ** 2 + aqd
    a[26] = a[25] + qa2 * qa2
    a[27] = (r + da - qb2) ** 2 + adq
    a[28] = (r - da - qb2) ** 2 + adq
    a[29] = (r + da + qb2) ** 2 + adq
    a[30] = (r - da + qb2) ** 2 + adq
    a[31] = (r + qa2 - db) ** 2 + aqd
    a[32] = (r + qa2 + db) ** 2 + aqd
    a[33] = (r - qa2 - db) ** 2 + aqd
    a[34] = (r - qa2 + db) ** 2 + aqd
    a[35] = rsq + aqq
    a[36] = a[35] + (qa2 - qb2) ** 2
    a[37] = a[35] + (qa2 + qb2) ** 2
    a[38] = a[35] + qa2 * qa2
    a[39] = a[35] + qb2 * qb2
    a[40] = a[38] + qb2 * qb2
    a[41] = (r - qb2) ** 2 + aqq
    a[42] = a[41] + qa2 * qa2
    a[43] = (r + qb2) ** 2 + aqq
    a[44] = a[43] + qa2 * qa2
    a[45] = (r + qa2) ** 2 + aqq
    a[46] = a[45] + qb2 * qb2
    a[47] = (r - qa2) ** 2 + aqq
    a[48] = a[47] + qb2 * qb2
    a[49] = (r + qa2 - qb2) ** 2 + aqq
    a[50] = (r + qa2 + qb2) ** 2 + aqq
    a[51] = (r - qa2 - qb2) ** 2 + aqq
    a[52] = (r - qa2 + qb2) ** 2 + aqq

    # Args 53-71 use qq (not 2*qq)
    qa = mp_A.qq
    qb = mp_B.qq
    da_m_qb2 = (da - qb) ** 2
    da_p_qb2 = (da + qb) ** 2
    r_m_qb2 = (r - qb) ** 2
    r_p_qb2 = (r + qb) ** 2
    a[53] = da_m_qb2 + r_m_qb2 + adq
    a[54] = da_m_qb2 + r_p_qb2 + adq
    a[55] = da_p_qb2 + r_m_qb2 + adq
    a[56] = da_p_qb2 + r_p_qb2 + adq

    qa_m_db2 = (qa - db) ** 2
    qa_p_db2 = (qa + db) ** 2
    r_p_qa2 = (r + qa) ** 2
    r_m_qa2 = (r - qa) ** 2
    a[57] = r_p_qa2 + qa_m_db2 + aqd
    a[58] = r_m_qa2 + qa_m_db2 + aqd
    a[59] = r_p_qa2 + qa_p_db2 + aqd
    a[60] = r_m_qa2 + qa_p_db2 + aqd

    qa_m_qb2 = (qa - qb) ** 2
    qa_p_qb2 = (qa + qb) ** 2
    a[61] = a[35] + 2.0 * qa_m_qb2
    a[62] = a[35] + 2.0 * qa_p_qb2
    a[63] = a[35] + 2.0 * (qa * qa + qb * qb)

    rpqamqb2 = (r + qa - qb) ** 2
    a[64] = rpqamqb2 + qa_m_qb2 + aqq
    a[65] = rpqamqb2 + qa_p_qb2 + aqq
    rpqapqb2 = (r + qa + qb) ** 2
    a[66] = rpqapqb2 + qa_m_qb2 + aqq
    a[67] = rpqapqb2 + qa_p_qb2 + aqq
    rmqamqb2 = (r - qa - qb) ** 2
    a[68] = rmqamqb2 + qa_m_qb2 + aqq
    a[69] = rmqamqb2 + qa_p_qb2 + aqq
    rmqapqb2 = (r - qa + qb) ** 2
    a[70] = rmqapqb2 + qa_m_qb2 + aqq
    a[71] = rmqapqb2 + qa_p_qb2 + aqq

    s = np.sqrt(a)

    # Named intermediates (Hartree: ev→1, ev1→0.5, ev2→0.25, ev3→0.125, ev4→0.0625)
    ee = 1.0 / s[0]
    dze = -0.5 / s[1] + 0.5 / s[2]
    qzze = 0.25 / s[3] + 0.25 / s[4] - 0.5 / s[5]
    qxxe = 0.5 / s[6] - 0.5 / s[5]
    edz = -0.5 / s[7] + 0.5 / s[8]
    eqzz = 0.25 / s[9] + 0.25 / s[10] - 0.5 / s[11]
    eqxx = 0.5 / s[12] - 0.5 / s[11]
    dxdx = 0.5 / s[13] - 0.5 / s[14]
    dzdz = 0.25 * (1.0/s[15] + 1.0/s[16] - 1.0/s[17] - 1.0/s[18])
    dzqxx = 0.25 * (1.0/s[19] - 1.0/s[20] - 1.0/s[21] + 1.0/s[22])
    qxxdz = 0.25 * (1.0/s[23] - 1.0/s[24] - 1.0/s[25] + 1.0/s[26])
    dzqzz = (0.125 * (-1.0/s[27] + 1.0/s[28] - 1.0/s[29] + 1.0/s[30])
             + 0.25 * (-1.0/s[21] + 1.0/s[19]))
    qzzdz = (0.125 * (-1.0/s[31] + 1.0/s[32] - 1.0/s[33] + 1.0/s[34])
             + 0.25 * (1.0/s[23] - 1.0/s[25]))
    qxxqxx = (0.125 * (1.0/s[36] + 1.0/s[37])
              - 0.25 * (1.0/s[38] + 1.0/s[39]) + 0.25 / s[35])
    qxxqyy = 0.25 * (1.0/s[40] - 1.0/s[38] - 1.0/s[39] + 1.0/s[35])
    qxxqzz = (0.125 * (1.0/s[42] + 1.0/s[44] - 1.0/s[41] - 1.0/s[43])
              + 0.25 * (-1.0/s[38] + 1.0/s[35]))
    qzzqxx = (0.125 * (1.0/s[46] + 1.0/s[48] - 1.0/s[45] - 1.0/s[47])
              + 0.25 * (-1.0/s[39] + 1.0/s[35]))
    qzzqzz = (0.0625 * (1.0/s[49] + 1.0/s[50] + 1.0/s[51] + 1.0/s[52])
              - 0.125 * (1.0/s[47] + 1.0/s[45] + 1.0/s[41] + 1.0/s[43])
              + 0.25 / s[35])
    dxqxz = 0.25 * (-1.0/s[53] + 1.0/s[54] + 1.0/s[55] - 1.0/s[56])
    qxzdx = 0.25 * (-1.0/s[57] + 1.0/s[58] + 1.0/s[59] - 1.0/s[60])
    qxzqxz = 0.125 * (1.0/s[64] - 1.0/s[66] - 1.0/s[68] + 1.0/s[70]
                       - 1.0/s[65] + 1.0/s[67] + 1.0/s[69] - 1.0/s[71])

    # Assemble 22 integrals (before nri)
    ri[0] = ee
    ri[1] = -dze
    ri[2] = ee + qzze
    ri[3] = ee + qxxe
    ri[4] = -edz
    ri[5] = dzdz
    ri[6] = dxdx
    ri[7] = -edz - qzzdz
    ri[8] = -edz - qxxdz
    ri[9] = -qxzdx
    ri[10] = ee + eqzz
    ri[11] = ee + eqxx
    ri[12] = -dze - dzqzz
    ri[13] = -dze - dzqxx
    ri[14] = -dxqxz
    ri[15] = ee + eqzz + qzze + qzzqzz
    ri[16] = ee + eqzz + qxxe + qxxqzz
    ri[17] = ee + eqxx + qzze + qzzqxx
    ri[18] = ee + eqxx + qxxe + qxxqxx
    ri[19] = qxzqxz
    ri[20] = ee + eqxx + qxxe + qxxqyy
    ri[21] = 0.5 * (qxxqxx - qxxqyy)


# ---------------------------------------------------------------------------
# Local-frame tensor filling
# ---------------------------------------------------------------------------

def _fill_local_tensor(ri: np.ndarray, nao_A: int, nao_B: int) -> np.ndarray:
    """Fill local-frame integral tensor from rotational invariants.

    Local frame indices: 0=s, 1=pσ, 2=pπ1, 3=pπ2.
    """
    L = np.zeros((nao_A, nao_A, nao_B, nao_B))
    L[0, 0, 0, 0] = ri[0]  # (SS|SS)

    if nao_A >= 4:
        L[0, 1, 0, 0] = L[1, 0, 0, 0] = ri[1]              # (SO|SS)
        L[1, 1, 0, 0] = ri[2]                                 # (OO|SS)
        L[2, 2, 0, 0] = L[3, 3, 0, 0] = ri[3]               # (PP|SS)

    if nao_B >= 4:
        L[0, 0, 0, 1] = L[0, 0, 1, 0] = ri[4]              # (SS|OS)
        L[0, 0, 1, 1] = ri[10]                                # (SS|OO)
        L[0, 0, 2, 2] = L[0, 0, 3, 3] = ri[11]              # (SS|PP)

    if nao_A >= 4 and nao_B >= 4:
        # (SO|SO)
        L[0, 1, 0, 1] = L[1, 0, 0, 1] = ri[5]
        L[0, 1, 1, 0] = L[1, 0, 1, 0] = ri[5]
        # (SP|SP) - same π direction
        for p in (2, 3):
            L[0, p, 0, p] = L[p, 0, 0, p] = ri[6]
            L[0, p, p, 0] = L[p, 0, p, 0] = ri[6]
        # (OO|SO)
        L[1, 1, 0, 1] = L[1, 1, 1, 0] = ri[7]
        # (PP|SO)
        for p in (2, 3):
            L[p, p, 0, 1] = L[p, p, 1, 0] = ri[8]
        # (PO|SP) - same π direction
        for p in (2, 3):
            L[1, p, 0, p] = L[p, 1, 0, p] = ri[9]
            L[1, p, p, 0] = L[p, 1, p, 0] = ri[9]
        # (SO|OO)
        L[0, 1, 1, 1] = L[1, 0, 1, 1] = ri[12]
        # (SO|PP)
        for p in (2, 3):
            L[0, 1, p, p] = L[1, 0, p, p] = ri[13]
        # (SP|OP) - same π direction
        for p in (2, 3):
            L[0, p, 1, p] = L[p, 0, 1, p] = ri[14]
            L[0, p, p, 1] = L[p, 0, p, 1] = ri[14]
        # (OO|OO)
        L[1, 1, 1, 1] = ri[15]
        # (PP|OO)
        for p in (2, 3):
            L[p, p, 1, 1] = ri[16]
        # (OO|PP)
        for p in (2, 3):
            L[1, 1, p, p] = ri[17]
        # (PP|PP) - same direction
        L[2, 2, 2, 2] = L[3, 3, 3, 3] = ri[18]
        # (PO|PO) - same π direction
        for p in (2, 3):
            L[1, p, 1, p] = L[p, 1, 1, p] = ri[19]
            L[1, p, p, 1] = L[p, 1, p, 1] = ri[19]
        # (PP|P*P*) - different directions
        L[2, 2, 3, 3] = L[3, 3, 2, 2] = ri[20]
        # (P*P|P*P) - cross product
        L[2, 3, 2, 3] = L[3, 2, 3, 2] = ri[21]
        L[2, 3, 3, 2] = L[3, 2, 2, 3] = ri[21]

    return L


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------

def _build_transform(c: np.ndarray, nao: int) -> np.ndarray:
    """Build local-to-molecular frame transformation matrix.

    Parameters
    ----------
    c : (3,) unit vector along bond (A→B)
    nao : 1 for H, 4 for sp atoms

    Returns
    -------
    T : (nao, nao) transformation matrix, T[mol_idx, loc_idx]
    """
    if nao == 1:
        return np.array([[1.0]])

    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
    rxy = math.sqrt(cx * cx + cy * cy)

    if rxy > 1e-10:
        ca = cx / rxy
        sa = cy / rxy
        e1 = np.array([ca * cz, sa * cz, -rxy])
        e2 = np.array([-sa, ca, 0.0])
    else:
        sign = 1.0 if cz > 0 else -1.0
        e1 = np.array([sign, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])

    T = np.zeros((4, 4))
    T[0, 0] = 1.0
    T[1:, 1] = c     # σ components
    T[1:, 2] = e1    # π1 components
    T[1:, 3] = e2    # π2 components
    return T


def _pack_transform_4x4(T: np.ndarray, nao: int) -> np.ndarray:
    """Pad a local->molecular transform to a packed 4x4 tensor."""
    out = np.zeros((4, 4), dtype=np.float64)
    out[:nao, :nao] = T
    return out


def _two_center_einsum_path(nao_A: int, nao_B: int):
    """Return cached contraction path for full two-center rotation."""
    key = (int(nao_A), int(nao_B))
    path = _TWO_CENTER_EINSUM_PATHS.get(key)
    if path is not None:
        return path
    TA = np.zeros((nao_A, nao_A), dtype=np.float64)
    TB = np.zeros((nao_B, nao_B), dtype=np.float64)
    L = np.zeros((nao_A, nao_A, nao_B, nao_B), dtype=np.float64)
    path, _ = np.einsum_path(
        "ma,nb,abcd,kc,ld->mnkl",
        TA,
        TA,
        L,
        TB,
        TB,
        optimize="greedy",
    )
    _TWO_CENTER_EINSUM_PATHS[key] = path
    return path


def build_pair_two_center_tensor(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    iA: int,
    iB: int,
    mp_params: Dict[int, MultipoleParams],
) -> Tuple[np.ndarray, float]:
    """Build one two-center pair tensor ``W`` in molecular frame.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers for all atoms.
    coords_bohr : (N, 3) array
        Cartesian coordinates in Bohr.
    iA, iB : int
        Pair atom indices with ``iA != iB``.
    mp_params : dict
        Multipole parameters keyed by atomic number.

    Returns
    -------
    W : (nao_A, nao_A, nao_B, nao_B) array
        Two-center electron-repulsion tensor for pair ``(A,B)``.
    gamma_ss : float
        ``(ss|ss)`` term for the pair.
    """
    if int(iA) == int(iB):
        raise ValueError("build_pair_two_center_tensor requires iA != iB")

    iA = int(iA)
    iB = int(iB)
    ZA = int(atomic_numbers[iA])
    ZB = int(atomic_numbers[iB])
    nao_A = nao_for_Z(ZA)
    nao_B = nao_for_Z(ZB)
    mp_A = mp_params[ZA]
    mp_B = mp_params[ZB]

    coords = np.asarray(coords_bohr, dtype=float)
    R_vec = coords[iB] - coords[iA]
    R = float(np.linalg.norm(R_vec))
    if R < 1e-14:
        raise ValueError("build_pair_two_center_tensor requires non-coincident nuclei")
    c_hat = R_vec / R

    si = nao_A >= 4
    sj = nao_B >= 4
    ri, gab = _reppd(R, mp_A, mp_B, si, sj)
    L = _fill_local_tensor(ri, nao_A, nao_B)
    T_A = _build_transform(c_hat, nao_A)
    T_B = _build_transform(c_hat, nao_B)
    W = np.einsum(
        "ma,nb,abcd,kc,ld->mnkl",
        T_A,
        T_A,
        L,
        T_B,
        T_B,
        optimize=_two_center_einsum_path(nao_A, nao_B),
    )
    return W, float(gab)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_two_center_integrals(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mp_params: Dict[int, MultipoleParams],
) -> List[np.ndarray]:
    """Build two-center integral tensors for all atom pairs.

    Returns
    -------
    W_list : list of (nA, nA, nB, nB) arrays in molecular frame (Hartree)
    """
    npairs = len(pair_i)
    W_list = []

    for k in range(npairs):
        W, _ = build_pair_two_center_tensor(
            atomic_numbers=atomic_numbers,
            coords_bohr=coords_bohr,
            iA=int(pair_i[k]),
            iB=int(pair_j[k]),
            mp_params=mp_params,
        )
        W_list.append(W)

    return W_list


def build_pair_ri_payload(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mp_params: Dict[int, MultipoleParams],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build reduced pair payload for AM1 CUDA setup without full W tensors.

    Returns
    -------
    ri_pack : (npairs, 22) float64
        Rotational invariants per pair.
    ta_pack, tb_pack : (npairs, 16) float64
        Flattened padded 4x4 local->molecular transforms.
    vaa_pack : (npairs, 16) float64
        Flattened padded 4x4 block for ``W[:,:,0,0]``.
    vbb_pack : (npairs, 16) float64
        Flattened padded 4x4 block for ``W[0,0,:,:]``.
    gamma_ss : (npairs,) float64
        ``(ss|ss)`` two-center term used by core-core repulsion.
    """
    npairs = len(pair_i)
    ri_pack = np.zeros((npairs, 22), dtype=np.float64)
    ta_pack = np.zeros((npairs, 16), dtype=np.float64)
    tb_pack = np.zeros((npairs, 16), dtype=np.float64)
    vaa_pack = np.zeros((npairs, 16), dtype=np.float64)
    vbb_pack = np.zeros((npairs, 16), dtype=np.float64)
    gamma_ss = np.zeros((npairs,), dtype=np.float64)

    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        ZA = int(atomic_numbers[iA])
        ZB = int(atomic_numbers[iB])
        nao_A = nao_for_Z(ZA)
        nao_B = nao_for_Z(ZB)
        mp_A = mp_params[ZA]
        mp_B = mp_params[ZB]

        R_vec = coords_bohr[iB] - coords_bohr[iA]
        R = float(np.linalg.norm(R_vec))
        c_hat = R_vec / R
        si = nao_A >= 4
        sj = nao_B >= 4

        ri, gab = _reppd(R, mp_A, mp_B, si, sj)
        ri_pack[k, :] = ri
        gamma_ss[k] = gab

        T_A = _build_transform(c_hat, nao_A)
        T_B = _build_transform(c_hat, nao_B)
        TA4 = _pack_transform_4x4(T_A, nao_A)
        TB4 = _pack_transform_4x4(T_B, nao_B)
        ta_pack[k, :] = TA4.ravel()
        tb_pack[k, :] = TB4.ravel()

        # Build only the two one-center attraction blocks needed for H-core setup.
        L = _fill_local_tensor(ri, nao_A, nao_B)
        # s orbitals are not rotated, so the core-attraction slices reduce to
        # local-frame ``L[:,:,0,0]`` and ``L[0,0,:,:]`` respectively.
        lab = L[:, :, 0, 0]
        lcd = L[0, 0, :, :]
        vaa = T_A @ lab @ T_A.T
        vbb = T_B @ lcd @ T_B.T

        VAA4 = np.zeros((4, 4), dtype=np.float64)
        VBB4 = np.zeros((4, 4), dtype=np.float64)
        VAA4[:nao_A, :nao_A] = vaa
        VBB4[:nao_B, :nao_B] = vbb
        vaa_pack[k, :] = VAA4.ravel()
        vbb_pack[k, :] = VBB4.ravel()

    return ri_pack, ta_pack, tb_pack, vaa_pack, vbb_pack, gamma_ss


def extract_electron_nuclear(W: np.ndarray, Z_B: int) -> np.ndarray:
    """Extract electron-nuclear attraction from two-center integrals.

    V_A<-B[mu, nu] = -Z_B * W[mu, nu, 0, 0]
    """
    return -Z_B * W[:, :, 0, 0]
