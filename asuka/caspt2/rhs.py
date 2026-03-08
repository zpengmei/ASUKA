r"""RHS vector construction for IC-CASPT2.

Ports OpenMolcas ``mkrhs.f`` (MKRHSA through MKRHSH).
The RHS vector :math:`V_P = \langle\Phi_P|\hat{H}|0\rangle` is the coupling
between the internally contracted (IC) basis functions and the reference
wavefunction through the full Hamiltonian.  All integrals use chemists'
notation: ``eri_mo[p,q,r,s]`` = :math:`(pq|rs)`.

Mathematical Definitions
------------------------
The RHS vector couples the reference :math:`|0\rangle` to each IC
perturber :math:`|\Phi_P\rangle` through the two-electron part
of the Hamiltonian.  The one-electron part enters only through
Fock-matrix corrections proportional to :math:`F^I_{pt}/N_{\text{act}}`.

Per-Case RHS Formulas
~~~~~~~~~~~~~~~~~~~~~
Using :math:`(pq|rs)` for chemists' notation ERIs and
:math:`\delta_{pq}` for Kronecker delta:

* **Case A (VJTU)** — 3 active + 1 inactive:

  .. math::

      V_A(tuv, i) = (uv|ti)
                    + \delta_{uv}\,\frac{F^I_{ti}}{N_{\text{act}}}

* **Case B+ (VJTIP)** — symmetric active/inactive pairs:

  .. math::

      V_{B+}(t{\ge}u, i{\ge}j) =
          \frac{[(it|ju) + (jt|iu)]\,(1 - \tfrac{1}{2}\delta_{tu})}
               {2\sqrt{1 + \delta_{ij}}}

* **Case B− (VJTIM)** — antisymmetric active/inactive pairs:

  .. math::

      V_{B-}(t{>}u, i{>}j) = \frac{(it|ju) - (jt|iu)}{2}

* **Case C (ATVX)** — 3 active + 1 virtual:

  .. math::

      V_C(tuv, a) = (at|uv)
                    + \delta_{uv}\,\frac{F^I_{at} - \sum_y (ay|yt)}{N_{\text{act}}}

* **Case D (AIVX)** — mixed inactive-virtual:

  .. math::

      V_{D1}(tu, ai) &= (ai|tu) + \delta_{tu}\,\frac{F^I_{ai}}{N_{\text{act}}} \\
      V_{D2}(tu, ai) &= (ti|au)

* **Cases E± (VJAI)** — 1 active, virtual + inactive pair:

  .. math::

      V_{E+}(t, i{\ge}j, a) &= \frac{(ai|tj) + (aj|ti)}{\sqrt{2 + 2\delta_{ij}}} \\
      V_{E-}(t, i{>}j, a) &= \sqrt{\tfrac{3}{2}}\,[(ai|tj) - (aj|ti)]

* **Cases F± (BVAT)** — symmetric/antisymmetric active + virtual pairs:

  .. math::

      V_{F+}(t{\ge}u, a{\ge}b) &= \frac{[(au|bt) + (at|bu)]\,(1 - \tfrac{1}{2}\delta_{tu})}
                                        {2\sqrt{1 + \delta_{ab}}} \\
      V_{F-}(t{>}u, a{>}b) &= \frac{(au|bt) - (at|bu)}{2}

* **Cases G± (BJAT)** — 1 active, inactive + virtual pair:

  .. math::

      V_{G+}(t, a{\ge}b, i) &= \frac{(at|bi) + (ai|bt)}{\sqrt{2 + 2\delta_{ab}}} \\
      V_{G-}(t, a{>}b, i) &= \sqrt{\tfrac{3}{2}}\,[(at|bi) - (ai|bt)]

* **Cases H± (BJAI)** — virtual + inactive pairs only:

  .. math::

      V_{H+}(a{\ge}b, i{\ge}j) &= \frac{(ai|bj) + (aj|bi)}
                                        {\sqrt{(1+\delta_{ab})(1+\delta_{ij})}} \\
      V_{H-}(a{>}b, i{>}j) &= \sqrt{3}\,[(ai|bj) - (aj|bi)]
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


def _rhs_b(smap, fock, eri_mo, dm1, dm2, *, sign: int):
    """Cases 2/3 (B±): 2 active + 2 inactive pairs.

    B+: WP[tgeu, igej] = [(it|ju) + (jt|iu)] * (1-d_tu/2) / (2*sqrt(1+d_ij))
    B-: WM[tgtu, igtj] = [(it|ju) - (jt|iu)] / 2
    """
    act_map = smap.mtgeu if sign > 0 else smap.mtgtu
    ext_map = smap.migej if sign > 0 else smap.migtj
    nact = smap.ntgeu if sign > 0 else smap.ntgtu
    next_ = smap.nigej if sign > 0 else smap.nigtj
    if nact == 0 or next_ == 0:
        return np.zeros(nact * next_, dtype=np.float64)

    ao = smap.orbs.nish
    rhs = np.zeros((nact, next_), dtype=np.float64)

    for p in range(nact):
        t, u = act_map[p]
        for q in range(next_):
            i, j = ext_map[q]
            val = eri_mo[i, ao+t, j, ao+u] + sign * eri_mo[j, ao+t, i, ao+u]
            if sign > 0:
                fac_tu = 0.5 if t != u else 0.25
                fac_ij = 1.0 / np.sqrt(1.0 + (1.0 if i == j else 0.0))
                rhs[p, q] = val * fac_tu * fac_ij
            else:
                rhs[p, q] = val * 0.5

    return rhs.ravel()


def _rhs_bp(smap, fock, eri_mo, dm1, dm2):
    return _rhs_b(smap, fock, eri_mo, dm1, dm2, sign=+1)


def _rhs_bm(smap, fock, eri_mo, dm1, dm2):
    return _rhs_b(smap, fock, eri_mo, dm1, dm2, sign=-1)


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


def _rhs_e(smap, fock, eri_mo, dm1, dm2, *, sign: int):
    """Cases 6/7 (E±): 1 active + (virtual, inactive pair).

    E+: WP[t, igej*nssh+a] = [(ai|tj) + (aj|ti)] / sqrt(2+2*d_ij)
    E-: WM[t, igtj*nssh+a] = sqrt(3/2) * [(ai|tj) - (aj|ti)]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    nssh = smap.orbs.nssh
    ext_map = smap.migej if sign > 0 else smap.migtj
    next_ = smap.nigej if sign > 0 else smap.nigtj
    nisup = nssh * next_
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_ext in range(next_):
        i, j = ext_map[q_ext]
        for a in range(nssh):
            ext = q_ext * nssh + a
            for t in range(nash):
                val = eri_mo[vo+a, i, ao+t, j] + sign * eri_mo[vo+a, j, ao+t, i]
                if sign > 0:
                    fac = 1.0 / np.sqrt(2.0 + 2.0 * (1.0 if i == j else 0.0))
                    rhs[t, ext] = val * fac
                else:
                    rhs[t, ext] = np.sqrt(1.5) * val

    return rhs.ravel()


def _rhs_ep(smap, fock, eri_mo, dm1, dm2):
    return _rhs_e(smap, fock, eri_mo, dm1, dm2, sign=+1)


def _rhs_em(smap, fock, eri_mo, dm1, dm2):
    return _rhs_e(smap, fock, eri_mo, dm1, dm2, sign=-1)


def _rhs_f(smap, fock, eri_mo, dm1, dm2, *, sign: int):
    """Cases 8/9 (F±): 2 active + 2 virtual pairs.

    F+: WP[tgeu, ageb] = [(au|bt) + (at|bu)] * (1-d_tu/2) / (2*sqrt(1+d_ab))
    F-: WM[tgtu, agtb] = [(au|bt) - (at|bu)] / 2
    """
    act_map = smap.mtgeu if sign > 0 else smap.mtgtu
    ext_map = smap.mageb if sign > 0 else smap.magtb
    nact = smap.ntgeu if sign > 0 else smap.ntgtu
    next_ = smap.nageb if sign > 0 else smap.nagtb
    if nact == 0 or next_ == 0:
        return np.zeros(nact * next_, dtype=np.float64)

    ao = smap.orbs.nish
    vo = smap.orbs.nish + smap.orbs.nash
    rhs = np.zeros((nact, next_), dtype=np.float64)

    for p in range(nact):
        t, u = act_map[p]
        for q in range(next_):
            a, b = ext_map[q]
            val = eri_mo[vo+a, ao+u, vo+b, ao+t] + sign * eri_mo[vo+a, ao+t, vo+b, ao+u]
            if sign > 0:
                fac_tu = 0.5 if t != u else 0.25
                fac_ab = 1.0 / np.sqrt(1.0 + (1.0 if a == b else 0.0))
                rhs[p, q] = val * fac_tu * fac_ab
            else:
                rhs[p, q] = val * 0.5

    return rhs.ravel()


def _rhs_fp(smap, fock, eri_mo, dm1, dm2):
    return _rhs_f(smap, fock, eri_mo, dm1, dm2, sign=+1)


def _rhs_fm(smap, fock, eri_mo, dm1, dm2):
    return _rhs_f(smap, fock, eri_mo, dm1, dm2, sign=-1)


def _rhs_g(smap, fock, eri_mo, dm1, dm2, *, sign: int):
    """Cases 10/11 (G±): 1 active + (inactive, virtual pair).

    G+: WP[t, ageb*nish+i] = [(at|bi) + (ai|bt)] / sqrt(2+2*d_ab)
    G-: WM[t, agtb*nish+i] = sqrt(3/2) * [(at|bi) - (ai|bt)]
    """
    nash = smap.orbs.nash
    nish = smap.orbs.nish
    ext_map = smap.mageb if sign > 0 else smap.magtb
    next_ = smap.nageb if sign > 0 else smap.nagtb
    nisup = nish * next_
    if nash == 0 or nisup == 0:
        return np.zeros(nash * nisup, dtype=np.float64)

    ao = nish
    vo = nish + nash
    rhs = np.zeros((nash, nisup), dtype=np.float64)

    for q_ext in range(next_):
        a, b = ext_map[q_ext]
        for i in range(nish):
            ext = q_ext * nish + i
            for t in range(nash):
                val = eri_mo[vo+a, ao+t, vo+b, i] + sign * eri_mo[vo+a, i, vo+b, ao+t]
                if sign > 0:
                    fac = 1.0 / np.sqrt(2.0 + 2.0 * (1.0 if a == b else 0.0))
                    rhs[t, ext] = val * fac
                else:
                    rhs[t, ext] = np.sqrt(1.5) * val

    return rhs.ravel()


def _rhs_gp(smap, fock, eri_mo, dm1, dm2):
    return _rhs_g(smap, fock, eri_mo, dm1, dm2, sign=+1)


def _rhs_gm(smap, fock, eri_mo, dm1, dm2):
    return _rhs_g(smap, fock, eri_mo, dm1, dm2, sign=-1)


def _rhs_h(smap, fock, eri_mo, dm1, dm2, *, sign: int):
    """Cases 12/13 (H±): virtual + inactive pairs only.

    H+: VP[ageb, igej] = [(ai|bj) + (aj|bi)] / sqrt((1+d_ab)*(1+d_ij))
    H-: VM[agtb, igtj] = sqrt(3) * [(ai|bj) - (aj|bi)]
    """
    act_map = smap.mageb if sign > 0 else smap.magtb
    ext_map = smap.migej if sign > 0 else smap.migtj
    nact = smap.nageb if sign > 0 else smap.nagtb
    next_ = smap.nigej if sign > 0 else smap.nigtj
    if nact == 0 or next_ == 0:
        return np.zeros(nact * next_, dtype=np.float64)

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    vo = nish + nash
    rhs = np.zeros((nact, next_), dtype=np.float64)

    for p in range(nact):
        a, b = act_map[p]
        for q in range(next_):
            i, j = ext_map[q]
            val = eri_mo[vo+a, i, vo+b, j] + sign * eri_mo[vo+a, j, vo+b, i]
            if sign > 0:
                fac = 1.0 / np.sqrt(
                    (1.0 + (1.0 if a == b else 0.0))
                    * (1.0 + (1.0 if i == j else 0.0))
                )
                rhs[p, q] = val * fac
            else:
                rhs[p, q] = np.sqrt(3.0) * val

    return rhs.ravel()


def _rhs_hp(smap, fock, eri_mo, dm1, dm2):
    return _rhs_h(smap, fock, eri_mo, dm1, dm2, sign=+1)


def _rhs_hm(smap, fock, eri_mo, dm1, dm2):
    return _rhs_h(smap, fock, eri_mo, dm1, dm2, sign=-1)
