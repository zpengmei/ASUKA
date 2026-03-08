r"""Native C1 SIGDER OFFDIAG builder for SS-CASPT2 PT2_Lag parity.

This module ports the core OpenMolcas ``sigder.f`` logic needed to build the
per-case OFFDIAG matrices consumed by CLagDX.

Mathematical Background
-----------------------
The SIGDER routine computes the derivative of the sigma-vector operator
with respect to CI coefficients, producing OFFDIAG matrices that couple
different IC cases. These are needed for the configuration Lagrangian (CLag).

The OFFDIAG matrix for case *c* accumulates contributions from all
coupling channels (*c*, *c'*) defined in the 24-channel IFCOUP table:

.. math::

    \text{OFFDIAG}^{(c)}[\mu,\nu] = \sum_{\text{channels}}
        \langle \Phi_\mu^{(c)} | \hat{F}_{\text{coupling}} | \Phi_\nu^{(c)} \rangle \cdot
        \text{amplitude/RHS terms}

Two passes are performed (following OpenMolcas):

- **IMLTOP=0 + C1S1DER**: Forward sigma application on high-case
  amplitude vectors, projected onto the low-case SR basis.
- **IMLTOP=1 + C2DER**: Backward pass applying the S-metric weighted
  low-case vectors, projected onto the high-case SR basis.

The implementation reuses ASUKA's C1 SGM tensor kernels from ``sigma.py``
(``_mltsca``, ``_mltmv``, ``_mltdxp``, ``_mltr1``, ``_spec1*`` routines).
All inputs/outputs are in ASUKA superindex ordering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.caspt2.sigma import (
    _build_lists_c1,
    _mltdxp,
    _mltmv,
    _mltr1,
    _mltsca,
    _spec1a_forward,
    _spec1a_reverse,
    _spec1c_forward,
    _spec1c_reverse,
    _spec1d_forward,
    _spec1d_reverse,
)


_COUPLINGS_C1: list[tuple[int, int, int]] = [
    (1, 2, 1),
    (1, 3, 2),
    (1, 5, 3),
    (1, 6, 4),
    (1, 7, 5),
    (2, 6, 6),
    (3, 7, 7),
    (4, 5, 8),
    (4, 8, 9),
    (4, 9, 10),
    (4, 10, 11),
    (4, 11, 12),
    (5, 6, 13),
    (5, 7, 14),
    (5, 10, 15),
    (5, 11, 16),
    (6, 12, 17),
    (7, 13, 18),
    (8, 10, 19),
    (9, 11, 20),
    (10, 12, 21),
    (11, 13, 22),
    (5, 12, 23),
    (5, 13, 24),
]


def _vec_for_loop(base: np.ndarray, add: np.ndarray, *, iloop: int, add_on_loop: int, vecrot: float) -> np.ndarray:
    if base.size == 0:
        return base
    if iloop != int(add_on_loop):
        return base
    out = np.asarray(float(vecrot) * base, dtype=np.float64)
    if add.shape == base.shape and add.size != 0:
        out = np.asarray(out + add, dtype=np.float64)
    return out


def _project_sr_to_mo(
    *,
    trans: np.ndarray,
    smat: np.ndarray,
    v_mo: np.ndarray,
    ityp: int,
) -> np.ndarray:
    """Molcas RHS_SR2C pair: MO -> IC then IC -> MO.

    This mirrors the exact two-step calls used in OpenMolcas SIGDER:
      - ITYP=0:  V_ic = T^T V_mo ; V_mo <- T V_ic
      - ITYP=1:  V_ic = (S*T)^T V_mo ; V_mo <- T V_ic

    Note that ITYP=1 does *not* apply `(S*T)` on the return leg.
    """
    if trans.size == 0 or v_mo.size == 0:
        return np.zeros_like(v_mo, dtype=np.float64)
    t = np.asarray(trans, dtype=np.float64)
    if int(ityp) == 1:
        st = np.asarray(smat @ t, dtype=np.float64)
        v_ic = np.asarray(st.T @ v_mo, dtype=np.float64)
        return np.asarray(t @ v_ic, dtype=np.float64)
    if int(ityp) == 0:
        v_ic = np.asarray(t.T @ v_mo, dtype=np.float64)
        return np.asarray(t @ v_ic, dtype=np.float64)
    raise ValueError(f"invalid ITYP={ityp}")  # pragma: no cover


def _build_case_vectors_asuka(
    *,
    smap: Any,
    smat_by_case: dict[int, np.ndarray],
    trans_by_case: dict[int, np.ndarray],
    t_sr_by_case: dict[int, np.ndarray],
    lbd_sr_by_case: dict[int, np.ndarray] | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    x_t: list[np.ndarray] = []
    x_r: list[np.ndarray] = []
    smats: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for case in range(1, 14):
        idx = int(case) - 1
        nas = int(smap.nasup[idx])
        nis = int(smap.nisup[idx])
        sm = np.asarray(smat_by_case.get(int(case), np.empty((0, 0))), dtype=np.float64)
        tr = np.asarray(trans_by_case.get(int(case), np.empty((0, 0))), dtype=np.float64)
        t_sr = np.asarray(t_sr_by_case.get(int(case), np.empty((0, 0))), dtype=np.float64)
        l_sr = np.asarray(
            (lbd_sr_by_case or {}).get(int(case), np.empty((0, 0))),
            dtype=np.float64,
        )

        if (
            nas <= 0
            or nis <= 0
            or sm.shape != (nas, nas)
            or tr.ndim != 2
            or tr.shape[0] != nas
            or t_sr.ndim != 2
            or t_sr.shape[1] != nis
            or tr.shape[1] != t_sr.shape[0]
        ):
            smats.append(np.empty((0, 0), dtype=np.float64))
            trans.append(np.empty((0, 0), dtype=np.float64))
            x_t.append(np.empty((0, 0), dtype=np.float64))
            x_r.append(np.empty((0, 0), dtype=np.float64))
            continue

        smats.append(np.asarray(sm, dtype=np.float64))
        trans.append(np.asarray(tr, dtype=np.float64))
        x_t.append(np.asarray(tr @ t_sr, dtype=np.float64))

        if l_sr.shape == t_sr.shape:
            x_r.append(np.asarray(tr @ l_sr, dtype=np.float64))
        else:
            x_r.append(np.zeros((nas, nis), dtype=np.float64))
    return x_t, x_r, smats, trans


def build_sigder_offdiag_asuka_c1(
    *,
    smap: Any,
    fock: Any,
    smat_by_case: dict[int, np.ndarray],
    trans_by_case: dict[int, np.ndarray],
    t_sr_by_case: dict[int, np.ndarray],
    lbd_sr_by_case: dict[int, np.ndarray] | None = None,
    nactel: int = 1,
    vecrot: float = 1.0,
    allowed_kods: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """Build per-case OFFDIAG in ASUKA ordering from native C1 data.

    Returns OFFDIAG for cases 1..11 (the cases consumed by CLagDX).
    """
    nash = int(getattr(smap.orbs, "nash"))
    nish = int(getattr(smap.orbs, "nish"))
    nssh = int(getattr(smap.orbs, "nssh"))
    if nash <= 0:
        return {}

    x_t, x_r, smats, trans = _build_case_vectors_asuka(
        smap=smap,
        smat_by_case=smat_by_case,
        trans_by_case=trans_by_case,
        t_sr_by_case=t_sr_by_case,
        lbd_sr_by_case=lbd_sr_by_case,
    )

    out: dict[int, np.ndarray] = {}
    for case in range(1, 12):
        idx = int(case) - 1
        nas = int(smap.nasup[idx])
        nis = int(smap.nisup[idx])
        if nas > 0 and nis > 0 and x_t[idx].size != 0:
            out[int(case)] = np.zeros((nas, nas), dtype=np.float64)

    if not out:
        return {}

    lists = _build_lists_c1(smap)
    ntu = int(smap.ntu)
    ntuv = int(smap.ntuv)
    nigej = int(smap.nigej)
    nigtj = int(smap.nigtj)
    nageb = int(smap.nageb)
    nagtb = int(smap.nagtb)

    sqr2 = float(np.sqrt(2.0))
    sqr3 = float(np.sqrt(3.0))
    sqr6 = float(np.sqrt(6.0))
    sqri2 = 1.0 / sqr2
    sqri6 = 1.0 / sqr6
    sqr32 = float(np.sqrt(1.5))
    fact = 1.0 / float(max(1, int(nactel)))

    fifa = np.asarray(getattr(fock, "fifa"), dtype=np.float64)
    ao = nish
    vo = nish + nash
    so = vo + nssh
    f_ti = np.asarray(fifa[ao:vo, :ao], dtype=np.float64, order="C")
    f_it = np.asarray(fifa[:ao, ao:vo], dtype=np.float64, order="C")
    f_ia = np.asarray(fifa[:ao, vo:so], dtype=np.float64, order="C")
    f_ai = np.asarray(fifa[vo:so, :ao], dtype=np.float64, order="C")
    f_ta = np.asarray(fifa[ao:vo, vo:so], dtype=np.float64, order="C")
    f_at = np.asarray(fifa[vo:so, ao:vo], dtype=np.float64, order="C")

    # SIGDER loops twice (bra/ket ordering). For IVEC=iVecX and JVEC=iVecR:
    # - ILOOP=1 adds SCAL*JVEC on low-case CX in C1S1DER/C2DER source.
    # - ILOOP=2 adds SCAL*JVEC on high-case CX in SGM/C2DER sink.
    for iloop in (1, 2):
        # IMLTOP=0 + C1S1DER contribution.
        for low in range(1, 12):
            low_idx = int(low) - 1
            nas1 = int(smap.nasup[low_idx])
            nis1 = int(smap.nisup[low_idx])
            if nas1 <= 0 or nis1 <= 0 or x_t[low_idx].size == 0:
                continue
            if int(low) not in out:
                continue
            if smats[low_idx].shape != (nas1, nas1) or trans[low_idx].shape[0] != nas1:
                continue

            sgm2 = np.zeros((nas1, nis1), dtype=np.float64)
            sgm1_ti = np.zeros((nash, nish), dtype=np.float64) if low == 1 and nish > 0 else None
            sgm1_ta = np.zeros((nash, nssh), dtype=np.float64) if low == 4 and nssh > 0 else None
            sgm1_ia = np.zeros((nish, nssh), dtype=np.float64) if low == 5 and (nish * nssh) > 0 else None

            for c1, c2, kod in _COUPLINGS_C1:
                if int(c1) != int(low):
                    continue
                if allowed_kods is not None and int(kod) not in allowed_kods:
                    continue
                high_idx = int(c2) - 1
                y2 = x_t[high_idx]
                if y2.size == 0:
                    continue
                y2 = _vec_for_loop(y2, x_r[high_idx], iloop=iloop, add_on_loop=2, vecrot=vecrot)

                if kod == 1:
                    _mltsca(0, lists[12], lists[14], sgm1_ti, f_ti, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                    _mltsca(0, lists[3], lists[14], sgm2, f_ti, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                elif kod == 2:
                    _mltsca(0, lists[13], lists[15], sgm1_ti, f_ti, y2, val1=(3.0, -3.0), val2=(1.0, -1.0))
                    _mltsca(0, lists[4], lists[15], sgm2, f_ti, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 3:
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    _mltmv(0, lists[1], sgm2, f_ta, y3, val1=(1.0, 1.0))
                elif kod == 4:
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)
                    _mltmv(0, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
                elif kod == 5:
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)
                    _mltmv(0, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
                elif kod == 6:
                    y3 = y2.reshape(nash, nigej, nssh, order="C")
                    _mltmv(0, lists[9], sgm2, f_ta, y3, val1=(sqri2, sqri2))
                elif kod == 7:
                    y3 = y2.reshape(nash, nigtj, nssh, order="C")
                    _mltmv(0, lists[10], sgm2, f_ta, y3, val1=(sqri6, -sqri6))
                elif kod == 8:
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C")
                    _mltmv(0, lists[11], sgm1_ta, f_ti, y3, val1=(2.0, 1.0))
                    _mltmv(0, lists[2], sgm2, f_ti, y3, val1=(-1.0, -1.0))
                elif kod == 9:
                    _mltsca(0, lists[12], lists[16], sgm1_ta, f_ta, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                    _mltsca(0, lists[5], lists[16], sgm2, f_ta, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                elif kod == 10:
                    _mltsca(0, lists[13], lists[17], sgm1_ta, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                    _mltsca(0, lists[6], lists[17], sgm2, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 11:
                    x_view = sgm1_ta.T
                    y3 = y2.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)
                    _mltmv(0, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
                elif kod == 12:
                    x_view = sgm1_ta.T
                    y3 = y2.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)
                    _mltmv(0, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
                elif kod == 13:
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)
                    _mltmv(0, lists[14], sgm1_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = y2.reshape(nash, nigej, nssh, order="C")
                    _mltdxp(0, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
                elif kod == 14:
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)
                    _mltmv(0, lists[15], sgm1_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = y2.reshape(nash, nigtj, nssh, order="C")
                    _mltdxp(0, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 15:
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = y2.reshape(nash, nageb, nish, order="C")
                    _mltdxp(0, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
                elif kod == 16:
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = y2.reshape(nash, nagtb, nish, order="C")
                    _mltdxp(0, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 17:
                    x_r1 = sgm2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)
                    _mltr1(0, lists[16], x_r1, f_ta.T, y2, val1=(sqri2, 1.0))
                elif kod == 18:
                    x_r1 = sgm2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)
                    _mltr1(0, lists[17], x_r1, f_ta.T, y2, val1=(sqri2, -sqri2))
                elif kod == 19:
                    y3 = y2.reshape(nash, nageb, nish, order="C")
                    _mltmv(0, lists[9], sgm2, f_ti, y3, val1=(-sqri2, -sqri2))
                elif kod == 20:
                    y3 = y2.reshape(nash, nagtb, nish, order="C")
                    _mltmv(0, lists[10], sgm2, f_ti, y3, val1=(-sqri6, sqri6))
                elif kod == 21:
                    x_r1 = sgm2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)
                    _mltr1(0, lists[14], x_r1, f_ti.T, y2.T, val1=(-sqri2, -1.0))
                elif kod == 22:
                    x_r1 = sgm2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)
                    _mltr1(0, lists[15], x_r1, f_ti.T, y2.T, val1=(sqri2, -sqri2))
                elif kod == 23:
                    _mltsca(0, lists[14], lists[16], sgm1_ia, f_ia, y2.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
                elif kod == 24:
                    half_sqr3 = 0.5 * sqr3
                    _mltsca(0, lists[15], lists[17], sgm1_ia, f_ia, y2.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))

            if low == 1 and sgm1_ti is not None:
                _spec1a_forward(smap.ktuv, nash, sgm2, sgm1_ti, fact=fact)
            elif low == 4 and sgm1_ta is not None:
                _spec1c_forward(smap.ktuv, nash, sgm2, sgm1_ta, fact=fact)
            elif low == 5 and sgm1_ia is not None:
                _spec1d_forward(smap.ktu, nash, sgm2, sgm1_ia.T.reshape(-1, order="C"), fact=fact)

            x_low = _vec_for_loop(x_t[low_idx], x_r[low_idx], iloop=iloop, add_on_loop=1, vecrot=vecrot)
            sder = np.asarray(2.0 * (x_low @ sgm2.T), dtype=np.float64)
            sgm2_proj = _project_sr_to_mo(
                trans=np.asarray(trans[low_idx], dtype=np.float64),
                smat=np.asarray(smats[low_idx], dtype=np.float64),
                v_mo=np.asarray(sgm2, dtype=np.float64),
                ityp=1,
            )
            sder += np.asarray(-1.0 * (x_low @ sgm2_proj.T), dtype=np.float64)
            out[int(low)] = np.asarray(out[int(low)] + sder, dtype=np.float64)

        # IMLTOP=1 + C2DER contribution.
        for low in range(1, 12):
            low_idx = int(low) - 1
            nas1 = int(smap.nasup[low_idx])
            nis1 = int(smap.nisup[low_idx])
            if nas1 <= 0 or nis1 <= 0 or x_t[low_idx].size == 0:
                continue
            if smats[low_idx].shape != (nas1, nas1):
                continue

            x_low = _vec_for_loop(x_t[low_idx], x_r[low_idx], iloop=iloop, add_on_loop=1, vecrot=vecrot)
            d2 = np.asarray(smats[low_idx] @ x_low, dtype=np.float64)
            d1: np.ndarray | None = None
            if low == 1:
                d1 = _spec1a_reverse(smap.ktuv, nash, nish, d2, fact=fact)
            elif low == 4:
                d1 = _spec1c_reverse(smap.ktuv, nash, nssh, d2, fact=fact)
            elif low == 5:
                d1 = _spec1d_reverse(smap.ktu, nash, d2, fact=fact)

            for c1, c2, kod in _COUPLINGS_C1:
                if int(c1) != int(low) or int(c2) > 11:
                    continue
                if allowed_kods is not None and int(kod) not in allowed_kods:
                    continue
                high_idx = int(c2) - 1
                nas2 = int(smap.nasup[high_idx])
                nis2 = int(smap.nisup[high_idx])
                if nas2 <= 0 or nis2 <= 0 or x_t[high_idx].size == 0:
                    continue
                if trans[high_idx].size == 0:
                    continue

                sgmx = np.zeros((nas2, nis2), dtype=np.float64)

                if kod == 1:
                    _mltsca(1, lists[12], lists[14], d1, f_ti, sgmx, val1=(1.0, 2.0), val2=(1.0, sqr2))
                    _mltsca(1, lists[3], lists[14], d2, f_ti, sgmx, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                elif kod == 2:
                    _mltsca(1, lists[13], lists[15], d1, f_ti, sgmx, val1=(3.0, -3.0), val2=(1.0, -1.0))
                    _mltsca(1, lists[4], lists[15], d2, f_ti, sgmx, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 3:
                    y3 = sgmx.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    _mltmv(1, lists[1], d2, f_ta, y3, val1=(1.0, 1.0))
                elif kod == 4:
                    x_view = d1.T
                    y3 = sgmx.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)
                    _mltmv(1, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
                elif kod == 5:
                    x_view = d1.T
                    y3 = sgmx.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)
                    _mltmv(1, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
                elif kod == 6:
                    y3 = sgmx.reshape(nash, nigej, nssh, order="C")
                    _mltmv(1, lists[9], d2, f_ta, y3, val1=(sqri2, sqri2))
                elif kod == 7:
                    y3 = sgmx.reshape(nash, nigtj, nssh, order="C")
                    _mltmv(1, lists[10], d2, f_ta, y3, val1=(sqri6, -sqri6))
                elif kod == 8:
                    y3 = sgmx.reshape(2 * ntu, nssh, nish, order="C")
                    _mltmv(1, lists[11], d1, f_ti, y3, val1=(2.0, 1.0))
                    _mltmv(1, lists[2], d2, f_ti, y3, val1=(-1.0, -1.0))
                elif kod == 9:
                    _mltsca(1, lists[12], lists[16], d1, f_ta, sgmx, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                    _mltsca(1, lists[5], lists[16], d2, f_ta, sgmx, val1=(1.0, 2.0), val2=(1.0, sqr2))
                elif kod == 10:
                    _mltsca(1, lists[13], lists[17], d1, f_ta, sgmx, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                    _mltsca(1, lists[6], lists[17], d2, f_ta, sgmx, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 11:
                    x_view = d1.T
                    y3 = sgmx.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)
                    _mltmv(1, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
                elif kod == 12:
                    x_view = d1.T
                    y3 = sgmx.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)
                    _mltmv(1, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
                elif kod == 13:
                    y3 = sgmx.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)
                    x_ia = d1.reshape(nssh, nish, order="C").T
                    _mltmv(1, lists[14], x_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                    x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = sgmx.reshape(nash, nigej, nssh, order="C")
                    _mltdxp(1, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
                elif kod == 14:
                    y3 = sgmx.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)
                    x_ia = d1.reshape(nssh, nish, order="C").T
                    _mltmv(1, lists[15], x_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                    x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = sgmx.reshape(nash, nigtj, nssh, order="C")
                    _mltdxp(1, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 15:
                    x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = sgmx.reshape(nash, nageb, nish, order="C")
                    _mltdxp(1, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
                elif kod == 16:
                    x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = sgmx.reshape(nash, nagtb, nish, order="C")
                    _mltdxp(1, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 17:
                    x_r1 = d2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)
                    _mltr1(1, lists[16], x_r1, f_ta.T, sgmx, val1=(sqri2, 1.0))
                elif kod == 18:
                    x_r1 = d2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)
                    _mltr1(1, lists[17], x_r1, f_ta.T, sgmx, val1=(sqri2, -sqri2))
                elif kod == 19:
                    y3 = sgmx.reshape(nash, nageb, nish, order="C")
                    _mltmv(1, lists[9], d2, f_ti, y3, val1=(-sqri2, -sqri2))
                elif kod == 20:
                    y3 = sgmx.reshape(nash, nagtb, nish, order="C")
                    _mltmv(1, lists[10], d2, f_ti, y3, val1=(-sqri6, sqri6))
                elif kod == 21:
                    x_r1 = d2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)
                    _mltr1(1, lists[14], x_r1, f_ti.T, sgmx.T, val1=(-sqri2, -1.0))
                elif kod == 22:
                    x_r1 = d2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)
                    _mltr1(1, lists[15], x_r1, f_ti.T, sgmx.T, val1=(sqri2, -sqri2))
                elif kod == 23:
                    x_ia = d1.reshape(nssh, nish, order="C").T
                    _mltsca(1, lists[14], lists[16], x_ia, f_ia, sgmx.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
                elif kod == 24:
                    half_sqr3 = 0.5 * sqr3
                    x_ia = d1.reshape(nssh, nish, order="C").T
                    _mltsca(1, lists[15], lists[17], x_ia, f_ia, sgmx.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))

                sgmx_proj = _project_sr_to_mo(
                    trans=np.asarray(trans[high_idx], dtype=np.float64),
                    smat=np.asarray(smats[high_idx], dtype=np.float64),
                    v_mo=np.asarray(sgmx, dtype=np.float64),
                    ityp=0,
                )
                x_high = _vec_for_loop(x_t[high_idx], x_r[high_idx], iloop=iloop, add_on_loop=2, vecrot=vecrot)
                out[int(c2)] = np.asarray(out[int(c2)] + (-1.0 * (x_high @ sgmx_proj.T)), dtype=np.float64)

    return {int(k): np.asarray(v, dtype=np.float64) for k, v in out.items()}
