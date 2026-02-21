"""MS-CASPT2 coupling contraction (OpenMolcas `hcoup.f` port, C1 only).

OpenMolcas evaluates off-diagonal effective Hamiltonian elements via:

  HEL = < ROOT1 | H * OMEGA | ROOT2 >

using:
  - a contravariant RHS vector (H|ROOT1>) stored on disk (`IVECW`)
  - a contravariant solution vector (|OMEGA ROOT2>) stored on disk (`IVECC`)
  - active transition densities (TG1/TG2/TG3) and overlap (OVL) between the
    two reference states.

This module ports the per-case kernels from `OpenMolcas/src/caspt2/hcoup.f`
and contracts them against precomputed row-wise dot products:

  row_dots[IAS, JAS] = dot(V1[IAS, :], V2[JAS, :])

where `V1` is the RHS block and `V2` is the solution block for the *ket* state.

Notes
-----
The overlap builders in `asuka.caspt2.overlap` implement the *SBMAT* (S-metric)
formulae, which are symmetric by construction. For MS couplings OpenMolcas uses
the `HCOUP` kernels, which:
  - are generally *not* symmetric for transition densities, and
  - use a different index layout in case 4 (see `hcoup.f`, CASE(4)).
"""

from __future__ import annotations

import numpy as np

from asuka.caspt2.superindex import SuperindexMap


def hcoup_case_contribution(
    case: int,
    smap: SuperindexMap,
    row_dots: np.ndarray,
    tg1: np.ndarray,
    tg2: np.ndarray,
    tg3: np.ndarray,
    *,
    ovl: float,
) -> float:
    """Contract one HCOUP case kernel against `row_dots`.

    Parameters
    ----------
    case
        IC case number (1-13).
    row_dots
        (nasup, nasup) matrix with entries `dot(V1[IAS,:], V2[JAS,:])`.
    tg1, tg2, tg3
        Active transition densities in E-operator convention:
          tg1[p,q] = <E_pq>
          tg2[p,q,r,s] = <E_pq E_rs>
          tg3[p,q,r,s,t,u] = <E_pq E_rs E_tu>
        (tg3 is dense in ASUKA; OpenMolcas stores it packed with pair-permutation
        symmetry).
    ovl
        Overlap <ROOT1|ROOT2>.
    """
    case = int(case)
    if case < 1 or case > 13:
        raise ValueError(f"Invalid case: {case}")

    row_dots = np.asarray(row_dots, dtype=np.float64, order="C")
    tg1 = np.asarray(tg1, dtype=np.float64, order="C")
    tg2 = np.asarray(tg2, dtype=np.float64, order="C")
    tg3 = np.asarray(tg3, dtype=np.float64, order="C")
    ovl = float(ovl)

    if row_dots.size == 0:
        return 0.0

    # CASE(1): SA(tuv,xyz)
    if case == 1:
        ntuv = int(smap.ntuv)
        if ntuv == 0:
            return 0.0
        if row_dots.shape != (ntuv, ntuv):
            raise ValueError(f"case 1 row_dots shape mismatch: {row_dots.shape} vs {(ntuv, ntuv)}")

        he = 0.0
        mtuv = smap.mtuv
        for ias in range(ntuv):
            t, u, v = (int(x) for x in mtuv[ias])
            for jas in range(ntuv):
                x, y, z = (int(x) for x in mtuv[jas])

                tmp = tg3[v, u, x, t, y, z]
                if y == u:
                    tmp += tg2[v, z, x, t]
                if y == t:
                    tmp += tg2[v, u, x, z]
                    if x == u:
                        tmp += tg1[v, z]
                if x == u:
                    tmp += tg2[v, t, y, z]

                sa = -tmp
                if x == t:
                    sa += 2.0 * tg2[v, u, y, z]
                    if y == u:
                        sa += 2.0 * tg1[v, z]

                he += sa * float(row_dots[ias, jas])
        return float(he)

    # CASE(2): SBP(tu,xy)
    if case == 2:
        ntgeu = int(smap.ntgeu)
        if ntgeu == 0:
            return 0.0
        if row_dots.shape != (ntgeu, ntgeu):
            raise ValueError(f"case 2 row_dots shape mismatch: {row_dots.shape} vs {(ntgeu, ntgeu)}")

        he = 0.0
        mtgeu = smap.mtgeu
        for ias in range(ntgeu):
            t, u = (int(x) for x in mtgeu[ias])
            for jas in range(ntgeu):
                x, y = (int(x) for x in mtgeu[jas])

                sbtuxy = 2.0 * tg2[x, t, y, u]
                sbtuyx = 2.0 * tg2[y, t, x, u]
                if x == t:
                    sbtuxy -= 4.0 * tg1[y, u]
                    sbtuyx += 2.0 * tg1[y, u]
                    if y == u:
                        sbtuxy += 8.0 * ovl
                        sbtuyx -= 4.0 * ovl
                if y == u:
                    sbtuxy -= 4.0 * tg1[x, t]
                    sbtuyx += 2.0 * tg1[x, t]
                if y == t:
                    sbtuxy += 2.0 * tg1[x, u]
                    sbtuyx -= 4.0 * tg1[x, u]
                    if x == u:
                        sbtuxy -= 4.0 * ovl
                        sbtuyx += 8.0 * ovl
                if x == u:
                    sbtuxy += 2.0 * tg1[y, t]
                    sbtuyx -= 4.0 * tg1[y, t]

                sbp = sbtuxy + sbtuyx
                he += sbp * float(row_dots[ias, jas])
        return float(he)

    # CASE(3): SBM(tu,xy)
    if case == 3:
        ntgtu = int(smap.ntgtu)
        if ntgtu == 0:
            return 0.0
        if row_dots.shape != (ntgtu, ntgtu):
            raise ValueError(f"case 3 row_dots shape mismatch: {row_dots.shape} vs {(ntgtu, ntgtu)}")

        he = 0.0
        mtgtu = smap.mtgtu
        for ias in range(ntgtu):
            t, u = (int(x) for x in mtgtu[ias])
            for jas in range(ntgtu):
                x, y = (int(x) for x in mtgtu[jas])

                sbtuxy = 2.0 * tg2[x, t, y, u]
                sbtuyx = 2.0 * tg2[y, t, x, u]
                if x == t:
                    sbtuxy -= 4.0 * tg1[y, u]
                    sbtuyx += 2.0 * tg1[y, u]
                    if y == u:
                        sbtuxy += 8.0 * ovl
                        sbtuyx -= 4.0 * ovl
                if y == u:
                    sbtuxy -= 4.0 * tg1[x, t]
                    sbtuyx += 2.0 * tg1[x, t]
                if y == t:
                    sbtuxy += 2.0 * tg1[x, u]
                    sbtuyx -= 4.0 * tg1[x, u]
                    if x == u:
                        sbtuxy -= 4.0 * ovl
                        sbtuyx += 8.0 * ovl
                if x == u:
                    sbtuxy += 2.0 * tg1[y, t]
                    sbtuyx -= 4.0 * tg1[y, t]

                sbm = sbtuxy - sbtuyx
                he += sbm * float(row_dots[ias, jas])
        return float(he)

    # CASE(4): SC(xuv,tyz)  (note the swapped active-index layout in OpenMolcas)
    if case == 4:
        ntuv = int(smap.ntuv)
        if ntuv == 0:
            return 0.0
        if row_dots.shape != (ntuv, ntuv):
            raise ValueError(f"case 4 row_dots shape mismatch: {row_dots.shape} vs {(ntuv, ntuv)}")

        he = 0.0
        mtuv = smap.mtuv
        for ias in range(ntuv):
            x, u, v = (int(x) for x in mtuv[ias])
            for jas in range(ntuv):
                t, y, z = (int(x) for x in mtuv[jas])

                tmp = tg3[v, u, x, t, y, z]
                if y == u:
                    tmp += tg2[v, z, x, t]
                if y == t:
                    tmp += tg2[v, u, x, z]
                    if x == u:
                        tmp += tg1[v, z]
                if x == u:
                    tmp += tg2[v, t, y, z]

                he += tmp * float(row_dots[ias, jas])
        return float(he)

    # CASE(5): SD 2x2 block over active pairs (tu)
    if case == 5:
        ntu = int(smap.ntu)
        nasup = 2 * ntu
        if ntu == 0:
            return 0.0
        if row_dots.shape != (nasup, nasup):
            raise ValueError(f"case 5 row_dots shape mismatch: {row_dots.shape} vs {(nasup, nasup)}")

        he = 0.0
        mtu = smap.mtu
        for p in range(ntu):
            t, u = (int(x) for x in mtu[p])
            ias1 = p
            ias2 = p + ntu
            for q in range(ntu):
                x, y = (int(x) for x in mtu[q])
                jas1 = q
                jas2 = q + ntu

                gutxy = tg2[u, t, x, y]
                sd11 = 2.0 * gutxy
                sd12 = -gutxy
                sd21 = -gutxy
                sd22 = -tg2[x, t, u, y]
                if t == x:
                    guy = tg1[u, y]
                    sd11 += 2.0 * guy
                    sd12 -= guy
                    sd21 -= guy
                    sd22 += 2.0 * guy

                he += sd11 * float(row_dots[ias1, jas1])
                he += sd12 * float(row_dots[ias1, jas2])
                he += sd21 * float(row_dots[ias2, jas1])
                he += sd22 * float(row_dots[ias2, jas2])
        return float(he)

    # CASE(6,7): SE(t,x) = 2*dxt - Dxt  (transition: scaled by OVL on delta)
    if case in (6, 7):
        nash = int(smap.orbs.nash)
        if nash == 0:
            return 0.0
        if row_dots.shape != (nash, nash):
            raise ValueError(f"case {case} row_dots shape mismatch: {row_dots.shape} vs {(nash, nash)}")

        he = 0.0
        for t in range(nash):
            for x in range(nash):
                se = -tg1[x, t]
                if x == t:
                    se += 2.0 * ovl
                he += se * float(row_dots[t, x])
        return float(he)

    # CASE(8): SFP(tu,xy)
    if case == 8:
        ntgeu = int(smap.ntgeu)
        if ntgeu == 0:
            return 0.0
        if row_dots.shape != (ntgeu, ntgeu):
            raise ValueError(f"case 8 row_dots shape mismatch: {row_dots.shape} vs {(ntgeu, ntgeu)}")

        he = 0.0
        mtgeu = smap.mtgeu
        for ias in range(ntgeu):
            t, u = (int(x) for x in mtgeu[ias])
            for jas in range(ntgeu):
                x, y = (int(x) for x in mtgeu[jas])
                sftuxy = 2.0 * tg2[t, x, u, y]
                sftuyx = 2.0 * tg2[t, y, u, x]
                sfp = sftuxy + sftuyx
                he += sfp * float(row_dots[ias, jas])
        return float(he)

    # CASE(9): SFM(tu,xy)
    if case == 9:
        ntgtu = int(smap.ntgtu)
        if ntgtu == 0:
            return 0.0
        if row_dots.shape != (ntgtu, ntgtu):
            raise ValueError(f"case 9 row_dots shape mismatch: {row_dots.shape} vs {(ntgtu, ntgtu)}")

        he = 0.0
        mtgtu = smap.mtgtu
        for ias in range(ntgtu):
            t, u = (int(x) for x in mtgtu[ias])
            for jas in range(ntgtu):
                x, y = (int(x) for x in mtgtu[jas])
                sftuxy = 2.0 * tg2[t, x, u, y]
                sftuyx = 2.0 * tg2[t, y, u, x]
                sfm = sftuxy - sftuyx
                he += sfm * float(row_dots[ias, jas])
        return float(he)

    # CASE(10,11): SG(t,x) = Gtx
    if case in (10, 11):
        nash = int(smap.orbs.nash)
        if nash == 0:
            return 0.0
        if row_dots.shape != (nash, nash):
            raise ValueError(f"case {case} row_dots shape mismatch: {row_dots.shape} vs {(nash, nash)}")

        he = 0.0
        for t in range(nash):
            for x in range(nash):
                sg = tg1[t, x]
                he += sg * float(row_dots[t, x])
        return float(he)

    # CASE(12,13): overlap-only (purely external) kernels
    if case in (12, 13):
        if abs(ovl) < 1.0e-12:
            return 0.0
        return float(ovl * float(np.trace(row_dots)))

    raise RuntimeError(f"Unhandled case: {case}")

