"""Sigma vector (H0 action) for IC-CASPT2.

OpenMolcas solves the CASPT2 equations with a sigma-vector operator
(``SIGMA_CASPT2`` in ``sigma.f``) rather than a pure diagonal divide.

Even after S/B diagonalization (SR basis), non-diagonal blocks of the
Fock matrix couple different excitation *cases* (see ``sgm.f`` and the
``IFCOUP`` table in ``eqsolv.F90``). For parity with OpenMolcas, ASUKA
must include these couplings when computing ``sigma = (H0 - E0) x``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.overlap import SBDecomposition
from asuka.caspt2.superindex import SuperindexMap


def sigma_caspt2_diagonal(
    vec_in: list[np.ndarray],
    sb_decomp: list[SBDecomposition],
    h0_diag: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute sigma = (H0 - E0)|v> in the diagonalized basis.

    In the diagonalized S/B basis, H0 is diagonal, so this is just
    element-wise multiplication.

    Parameters
    ----------
    vec_in : list of arrays
        Input amplitudes per case, in the diagonalized basis.
        Each array has shape (nindep * nisup,) or similar.
    sb_decomp : list of SBDecomposition
        S/B decomposition results per case.
    h0_diag : list of arrays
        Diagonal of (H0 - E0) in the diagonalized basis per case.

    Returns
    -------
    sigma_out : list of arrays
        Result vectors per case.
    """
    sigma_out: list[np.ndarray] = []
    for v, decomp, diag in zip(vec_in, sb_decomp, h0_diag):
        if v.size == 0 or decomp.nindep == 0:
            sigma_out.append(np.zeros_like(v))
            continue
        sigma_out.append(diag * v)
    return sigma_out


@dataclass(frozen=True)
class SigmaC1ActiveVirtualCoupling:
    """Sigma operator with Molcas-style active-virtual case couplings (C1 only).

    This implements the missing OpenMolcas coupling that makes the
    `ATVX` (case 4) and `BVAT` (cases 8/9) amplitudes *not* equal to
    `-rhs/denom` when the active-virtual Fock block is non-zero.

    Currently implemented couplings (OpenMolcas ``sgm.f``):
      - KOD=9:  case 4 <-> case 8  (C <-> F+), uses FTA
      - KOD=10: case 4 <-> case 9  (C <-> F-), uses FTA

    This coupling exists regardless of `nish`. For general systems additional
    couplings (inactive-related KODs in `sgm.f`) are still needed for full
    OpenMolcas parity.
    """

    smap: SuperindexMap
    fock: CASPT2Fock
    smats: list[np.ndarray]
    sb_decomp: list[SBDecomposition]
    h0_diag: list[np.ndarray]
    nactel: int

    def __call__(self, vec_in: list[np.ndarray]) -> list[np.ndarray]:
        nish = int(self.smap.orbs.nish)

        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        if nash == 0:
            return [np.zeros_like(v) for v in vec_in]

        # Diagonal (H0(diag)-E0) contribution in SR basis.
        sigma_sr = sigma_caspt2_diagonal(vec_in, self.sb_decomp, self.h0_diag)

        # Convert input vectors to standard (C) contravariant blocks:
        #   x_C = T * x_SR   where T=decomp.transform (nasup, nindep)
        x_c_contrav: list[np.ndarray] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            if v.size == 0 or decomp.nindep == 0 or nisup == 0:
                x_c_contrav.append(np.empty((0, 0), dtype=np.float64))
                continue
            x_sr = v.reshape(decomp.nindep, nisup)
            x_c_contrav.append(np.asarray(decomp.transform @ x_sr, dtype=np.float64, order="C"))

        # Convert diagonal sigma to standard (C) covariant blocks:
        #   sigma_C = S * T * sigma_SR
        sigma_c_covar: list[np.ndarray] = []
        for case_idx, (v_sig, decomp) in enumerate(zip(sigma_sr, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            if v_sig.size == 0 or decomp.nindep == 0 or nisup == 0:
                sigma_c_covar.append(np.empty((0, 0), dtype=np.float64))
                continue
            sig_sr = v_sig.reshape(decomp.nindep, nisup)
            sig_c_contrav = decomp.transform @ sig_sr  # (nasup, nisup)
            smat = self.smats[case_idx]
            sigma_c_covar.append(np.asarray(smat @ sig_c_contrav, dtype=np.float64, order="C"))

        # Active-virtual Fock block (Molcas FTA): active x virtual
        ao = nish
        vo = nish + nash
        fta = np.asarray(self.fock.fifa[ao:ao + nash, vo:vo + nssh], dtype=np.float64, order="C")

        # If the coupling block is numerically zero, we can skip.
        if float(np.max(np.abs(fta))) > 0.0:
            self._apply_couplings_c1(
                fta=fta,
                x_c_contrav=x_c_contrav,
                sigma_c_covar=sigma_c_covar,
            )

        # Transform sigma back to SR basis (covariant):
        #   sigma_SR = T^T * sigma_C
        sigma_out: list[np.ndarray] = []
        for case_idx, (decomp, sig_c) in enumerate(zip(self.sb_decomp, sigma_c_covar)):
            nisup = int(self.smap.nisup[case_idx])
            if sig_c.size == 0 or decomp.nindep == 0 or nisup == 0:
                sigma_out.append(np.zeros_like(vec_in[case_idx]))
                continue
            sig_sr = decomp.transform.T @ sig_c  # (nindep, nisup)
            sigma_out.append(np.asarray(sig_sr.ravel(), dtype=np.float64))
        return sigma_out

    def _apply_couplings_c1(
        self,
        *,
        fta: np.ndarray,  # (nash, nssh)
        x_c_contrav: list[np.ndarray],
        sigma_c_covar: list[np.ndarray],
    ) -> None:
        """Apply case couplings in C representation (Molcas SIGMA/SGM layout)."""
        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        ntuv = int(self.smap.ntuv)
        ntgeu = int(self.smap.ntgeu)
        ntgtu = int(self.smap.ntgtu)
        nageb = int(self.smap.nageb)
        nagtb = int(self.smap.nagtb)

        # Case indices are 0-based here (case 4 -> idx 3, case 8 -> idx 7, case 9 -> idx 8).
        i_case_c = 3
        i_case_fp = 7
        i_case_fm = 8

        if nssh == 0:
            return

        list5 = _list5_by_t(self.smap)
        list6 = _list6_by_t(self.smap)

        # Only apply if the blocks exist and survived lin-dep removal.
        smat_c = self.smats[i_case_c] if len(self.smats) > i_case_c else None
        if smat_c is None or smat_c.size == 0:
            return

        x_c = x_c_contrav[i_case_c]
        if x_c.size == 0:
            return

        fact = 1.0 / float(max(1, int(self.nactel)))

        # Precompute (a,b)->pair maps and coefficients used by the Molcas lists 16/17.
        ab_plus_idx, ab_plus_coef = _pair_map_plus(self.smap.kageb, nssh, eq_coef=np.sqrt(2.0))
        ab_minus_idx, ab_minus_coef = _pair_map_minus(self.smap.kagtb, nssh)

        # Precompute (t,u)->pair maps for Molcas lists 12/13.
        tu_plus_idx = _active_pair_plus_idx(self.smap.ktgeu, nash)
        tu_plus_coef = _active_pair_plus_coef_kod9(nash)  # -1 / -2
        tu_minus_idx = _active_pair_minus_idx(self.smap.ktgtu, nash)
        tu_minus_coef = _active_pair_minus_coef_kod10(nash)  # -1 / +1 / 0

        # ---------- KOD=9: case 4 <-> case 8 (C <-> F+) ----------
        if ntgeu > 0 and nageb > 0:
            x_fp = x_c_contrav[i_case_fp]
            if x_fp.size != 0:
                # IMLTOP=0 contribution: sigma(case 4) from X(case 8).
                sgm2 = _sgm_c_from_fp_imltop0(
                    smap=self.smap,
                    fta=fta,
                    x_fp=x_fp,
                    list5=list5,
                    tu_plus_idx=tu_plus_idx,
                    tu_plus_coef=tu_plus_coef,
                    ab_plus_idx=ab_plus_idx,
                    ab_plus_coef=ab_plus_coef,
                    fact=fact,
                )
                sigma_c_covar[i_case_c] += smat_c @ sgm2

            # IMLTOP=1 contribution: sigma(case 8) from X(case 4).
            # D2 = S * X(case 4)  and D1 from SPEC1C (IFC=1).
            d2 = smat_c @ x_c  # (ntuv, nssh)
            d1 = _spec1c_reverse(self.smap.ktuv, nash, nssh, d2, fact=fact)  # (nash, nssh)
            delta_fp = _sgm_fp_from_c_imltop1(
                smap=self.smap,
                fta=fta,
                d1=d1,
                d2=d2,
                list5=list5,
                tu_plus_idx=tu_plus_idx,
                tu_plus_coef=tu_plus_coef,
                ab_plus_idx=ab_plus_idx,
                ab_plus_coef=ab_plus_coef,
            )
            if sigma_c_covar[i_case_fp].size != 0:
                sigma_c_covar[i_case_fp] += delta_fp

        # ---------- KOD=10: case 4 <-> case 9 (C <-> F-) ----------
        if ntgtu > 0 and nagtb > 0:
            x_fm = x_c_contrav[i_case_fm]
            if x_fm.size != 0:
                sgm2 = _sgm_c_from_fm_imltop0(
                    smap=self.smap,
                    fta=fta,
                    x_fm=x_fm,
                    list6=list6,
                    tu_minus_idx=tu_minus_idx,
                    tu_minus_coef=tu_minus_coef,
                    ab_minus_idx=ab_minus_idx,
                    ab_minus_coef=ab_minus_coef,
                    fact=fact,
                )
                sigma_c_covar[i_case_c] += smat_c @ sgm2

            d2 = smat_c @ x_c
            d1 = _spec1c_reverse(self.smap.ktuv, nash, nssh, d2, fact=fact)
            delta_fm = _sgm_fm_from_c_imltop1(
                smap=self.smap,
                fta=fta,
                d1=d1,
                d2=d2,
                list6=list6,
                tu_minus_idx=tu_minus_idx,
                tu_minus_coef=tu_minus_coef,
                ab_minus_idx=ab_minus_idx,
                ab_minus_coef=ab_minus_coef,
            )
            if sigma_c_covar[i_case_fm].size != 0:
                sigma_c_covar[i_case_fm] += delta_fm


def _active_pair_plus_idx(ktgeu: np.ndarray, nash: int) -> np.ndarray:
    out = np.empty((nash, nash), dtype=np.int64)
    for t in range(nash):
        for u in range(nash):
            hi = t if t >= u else u
            lo = u if t >= u else t
            out[t, u] = int(ktgeu[hi, lo])
    return out


def _active_pair_minus_idx(ktgtu: np.ndarray, nash: int) -> np.ndarray:
    out = np.empty((nash, nash), dtype=np.int64)
    for t in range(nash):
        for u in range(nash):
            if t == u:
                out[t, u] = 0
                continue
            hi = t if t >= u else u
            lo = u if t >= u else t
            out[t, u] = int(ktgtu[hi, lo])
    return out


def _active_pair_plus_coef_kod9(nash: int) -> np.ndarray:
    # KOD=9 (C<->F+) uses list12 with VAL1 = [-1, -2] (neq / eq).
    out = -np.ones((nash, nash), dtype=np.float64)
    for t in range(nash):
        out[t, t] = -2.0
    return out


def _active_pair_minus_coef_kod10(nash: int) -> np.ndarray:
    # KOD=10 (C<->F-) uses list13 with VAL1 = [-1, +1] (t>u / t<u).
    out = np.zeros((nash, nash), dtype=np.float64)
    for t in range(nash):
        for u in range(nash):
            if t == u:
                continue
            out[t, u] = -1.0 if t > u else 1.0
    return out


def _pair_map_plus(kplus: np.ndarray, n: int, *, eq_coef: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.empty((n, n), dtype=np.int64)
    coef = np.ones((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            hi = a if a >= b else b
            lo = b if a >= b else a
            idx[a, b] = int(kplus[hi, lo])
            if a == b:
                coef[a, b] = float(eq_coef)
    return idx, coef


def _pair_map_minus(kminus: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.empty((n, n), dtype=np.int64)
    coef = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if a == b:
                idx[a, b] = 0
                coef[a, b] = 0.0
                continue
            hi = a if a >= b else b
            lo = b if a >= b else a
            idx[a, b] = int(kminus[hi, lo])
            coef[a, b] = 1.0 if a > b else -1.0
    return idx, coef


def _spec1c_forward(ktuv: np.ndarray, nash: int, sgm2: np.ndarray, sgm1: np.ndarray, *, fact: float) -> None:
    # Mirrors OpenMolcas spec1c.f for IFC=0: add (t,a) into (t,u,u,a).
    for t in range(nash):
        y = sgm1[t, :]
        if not np.any(y):
            continue
        for u in range(nash):
            ituu = int(ktuv[t, u, u])
            sgm2[ituu, :] += fact * y


def _spec1c_reverse(ktuv: np.ndarray, nash: int, nssh: int, d2: np.ndarray, *, fact: float) -> np.ndarray:
    # Mirrors OpenMolcas spec1c.f for IFC=1: accumulate from (t,u,u,a) into (t,a).
    d1 = np.zeros((nash, nssh), dtype=np.float64)
    for t in range(nash):
        acc = np.zeros(nssh, dtype=np.float64)
        for u in range(nash):
            ituu = int(ktuv[t, u, u])
            acc += d2[ituu, :]
        d1[t, :] = fact * acc
    return d1


def _list5_by_t(smap: SuperindexMap) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Molcas list 5 (TUV/TU+) specialized to C1.

    Returns per-t arrays: (triple_idx[m], uv_plus_idx[m], coef[m]).
    """
    nash = int(smap.orbs.nash)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(nash):
        triples: list[int] = []
        uv_idx: list[int] = []
        coef: list[float] = []
        for p in range(int(smap.ntgeu)):
            u, v = (int(x) for x in smap.mtgeu[p])
            if u == v:
                triples.append(int(smap.ktuv[u, t, v]))
                uv_idx.append(p)
                coef.append(2.0)
            else:
                triples.append(int(smap.ktuv[u, t, v]))
                uv_idx.append(p)
                coef.append(1.0)
                triples.append(int(smap.ktuv[v, t, u]))
                uv_idx.append(p)
                coef.append(1.0)
        out.append((np.asarray(triples, dtype=np.int64), np.asarray(uv_idx, dtype=np.int64), np.asarray(coef, dtype=np.float64)))
    return out


def _list6_by_t(smap: SuperindexMap) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Molcas list 6 (TUV/TU-) specialized to C1.

    Returns per-t arrays: (triple_idx[m], uv_minus_idx[m], coef[m]) where coef is -1/+1.
    """
    nash = int(smap.orbs.nash)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(nash):
        triples: list[int] = []
        uv_idx: list[int] = []
        coef: list[float] = []
        for p in range(int(smap.ntgtu)):
            u, v = (int(x) for x in smap.mtgtu[p])
            # First entry (V=1): -1
            triples.append(int(smap.ktuv[u, t, v]))
            uv_idx.append(p)
            coef.append(-1.0)
            # Second entry (V=2): +1
            triples.append(int(smap.ktuv[v, t, u]))
            uv_idx.append(p)
            coef.append(1.0)
        out.append((np.asarray(triples, dtype=np.int64), np.asarray(uv_idx, dtype=np.int64), np.asarray(coef, dtype=np.float64)))
    return out


def _sgm_c_from_fp_imltop0(
    *,
    smap: SuperindexMap,
    fta: np.ndarray,  # (nash,nssh)
    x_fp: np.ndarray,  # (ntgeu,nageb)
    list5: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tu_plus_idx: np.ndarray,  # (nash,nash) -> ntgeu
    tu_plus_coef: np.ndarray,  # (nash,nash)
    ab_plus_idx: np.ndarray,  # (nssh,nssh) -> nageb
    ab_plus_coef: np.ndarray,  # (nssh,nssh)
    fact: float,
) -> np.ndarray:
    """KOD=9, IMLTOP=0: build SGM2 for sigma(case4) from X(case8)."""
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntuv = int(smap.ntuv)

    sgm1 = np.zeros((nash, nssh), dtype=np.float64)
    sgm2 = np.zeros((ntuv, nssh), dtype=np.float64)

    # One-el part (list12/list16), accumulated into SGM1(t,a).
    for u in range(nash):
        rows = tu_plus_idx[:, u]
        row_coef = tu_plus_coef[:, u]
        for b in range(nssh):
            cols = ab_plus_idx[:, b]
            col_coef = ab_plus_coef[:, b]
            y_sel = x_fp[np.ix_(rows, cols)]
            sgm1 += fta[u, b] * (row_coef[:, None] * y_sel) * (col_coef[None, :])

    # Two-el part (list5/list16), accumulated into SGM2(tuv,a).
    for t in range(nash):
        triple_idx, uv_idx, coef_uv = list5[t]
        for b in range(nssh):
            cols = ab_plus_idx[:, b]
            col_coef = ab_plus_coef[:, b]
            y_rows = x_fp[np.ix_(uv_idx, cols)]
            contrib = fta[t, b] * (coef_uv[:, None] * y_rows) * (col_coef[None, :])
            sgm2[triple_idx, :] += contrib

    _spec1c_forward(smap.ktuv, nash, sgm2, sgm1, fact=fact)
    return sgm2


def _sgm_c_from_fm_imltop0(
    *,
    smap: SuperindexMap,
    fta: np.ndarray,  # (nash,nssh)
    x_fm: np.ndarray,  # (ntgtu,nagtb)
    list6: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tu_minus_idx: np.ndarray,  # (nash,nash) -> ntgtu (t!=u)
    tu_minus_coef: np.ndarray,  # (nash,nash) (-1/+1/0)
    ab_minus_idx: np.ndarray,  # (nssh,nssh) -> nagtb (a!=b)
    ab_minus_coef: np.ndarray,  # (nssh,nssh) (+1/-1/0)
    fact: float,
) -> np.ndarray:
    """KOD=10, IMLTOP=0: build SGM2 for sigma(case4) from X(case9)."""
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntuv = int(smap.ntuv)

    sgm1 = np.zeros((nash, nssh), dtype=np.float64)
    sgm2 = np.zeros((ntuv, nssh), dtype=np.float64)

    # One-el part (list13/list17) -> SGM1(t,a).
    for u in range(nash):
        rows = tu_minus_idx[:, u]
        row_coef = tu_minus_coef[:, u]
        for b in range(nssh):
            cols = ab_minus_idx[:, b]
            col_coef = ab_minus_coef[:, b]
            y_sel = x_fm[np.ix_(rows, cols)]
            sgm1 += fta[u, b] * (row_coef[:, None] * y_sel) * (col_coef[None, :])

    # Two-el part (list6/list17) -> SGM2(tuv,a).
    for t in range(nash):
        triple_idx, uv_idx, coef_uv = list6[t]
        for b in range(nssh):
            cols = ab_minus_idx[:, b]
            col_coef = ab_minus_coef[:, b]
            y_rows = x_fm[np.ix_(uv_idx, cols)]
            contrib = fta[t, b] * (coef_uv[:, None] * y_rows) * (col_coef[None, :])
            sgm2[triple_idx, :] += contrib

    _spec1c_forward(smap.ktuv, nash, sgm2, sgm1, fact=fact)
    return sgm2


def _sgm_fp_from_c_imltop1(
    *,
    smap: SuperindexMap,
    fta: np.ndarray,  # (nash,nssh)
    d1: np.ndarray,  # (nash,nssh)
    d2: np.ndarray,  # (ntuv,nssh)
    list5: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tu_plus_idx: np.ndarray,
    tu_plus_coef: np.ndarray,
    ab_plus_idx: np.ndarray,
    ab_plus_coef: np.ndarray,
) -> np.ndarray:
    """KOD=9, IMLTOP=1: sigma(case8) update from D1/D2 of case4."""
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntgeu = int(smap.ntgeu)
    nageb = int(smap.nageb)

    out = np.zeros((ntgeu, nageb), dtype=np.float64)

    # One-el part: out[tu+,ab+] += FTA(u,b) * coeff(t,u) * coeff(a,b) * D1(t,a)
    for u in range(nash):
        rows = tu_plus_idx[:, u]
        row_coef = tu_plus_coef[:, u]
        for b in range(nssh):
            cols = ab_plus_idx[:, b]
            col_coef = ab_plus_coef[:, b]
            contrib = fta[u, b] * (row_coef[:, None] * d1) * (col_coef[None, :])
            out[np.ix_(rows, cols)] += contrib

    # Two-el part: aggregate D2(triple,a) -> rows in TU+ space for each middle index t.
    all_uv = np.arange(ntgeu, dtype=np.int64)
    for t in range(nash):
        triple_idx, uv_idx, coef_uv = list5[t]
        agg = np.zeros((ntgeu, nssh), dtype=np.float64)
        np.add.at(agg, uv_idx, coef_uv[:, None] * d2[triple_idx, :])
        for b in range(nssh):
            cols = ab_plus_idx[:, b]
            col_coef = ab_plus_coef[:, b]
            out[np.ix_(all_uv, cols)] += fta[t, b] * agg * (col_coef[None, :])

    return out


def _sgm_fm_from_c_imltop1(
    *,
    smap: SuperindexMap,
    fta: np.ndarray,  # (nash,nssh)
    d1: np.ndarray,  # (nash,nssh)
    d2: np.ndarray,  # (ntuv,nssh)
    list6: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tu_minus_idx: np.ndarray,
    tu_minus_coef: np.ndarray,
    ab_minus_idx: np.ndarray,
    ab_minus_coef: np.ndarray,
) -> np.ndarray:
    """KOD=10, IMLTOP=1: sigma(case9) update from D1/D2 of case4."""
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    ntgtu = int(smap.ntgtu)
    nagtb = int(smap.nagtb)

    out = np.zeros((ntgtu, nagtb), dtype=np.float64)

    # One-el part: out[tu-,ab-] += FTA(u,b) * coeff(t,u) * coeff(a,b) * D1(t,a)
    for u in range(nash):
        rows = tu_minus_idx[:, u]
        row_coef = tu_minus_coef[:, u]
        for b in range(nssh):
            cols = ab_minus_idx[:, b]
            col_coef = ab_minus_coef[:, b]
            contrib = fta[u, b] * (row_coef[:, None] * d1) * (col_coef[None, :])
            out[np.ix_(rows, cols)] += contrib

    all_uv = np.arange(ntgtu, dtype=np.int64)
    for t in range(nash):
        triple_idx, uv_idx, coef_uv = list6[t]
        agg = np.zeros((ntgtu, nssh), dtype=np.float64)
        np.add.at(agg, uv_idx, coef_uv[:, None] * d2[triple_idx, :])
        for b in range(nssh):
            cols = ab_minus_idx[:, b]
            col_coef = ab_minus_coef[:, b]
            out[np.ix_(all_uv, cols)] += fta[t, b] * agg * (col_coef[None, :])

    return out


# ---------------------------------------------------------------------------
# Full Molcas SIGMA_CASPT2 couplings for C1 (nish>0 support)
# ---------------------------------------------------------------------------


def _spec1a_forward(ktuv: np.ndarray, nash: int, sgm2: np.ndarray, sgm1: np.ndarray, *, fact: float) -> None:
    """Mirror OpenMolcas `spec1a.f` for IFC=0 (C1).

    Adds 1-electron sigma contributions `SGM1(t,i)` into the `tuu` triples of
    `SGM2(t,u,v,i)` with a factor `fact=1/NACTEL`.
    """
    for t in range(nash):
        y = sgm1[t, :]
        if not np.any(y):
            continue
        for u in range(nash):
            ituu = int(ktuv[t, u, u])
            sgm2[ituu, :] += fact * y


def _spec1a_reverse(ktuv: np.ndarray, nash: int, nish: int, d2: np.ndarray, *, fact: float) -> np.ndarray:
    """Mirror OpenMolcas `spec1a.f` for IFC=1 (C1)."""
    d1 = np.zeros((nash, nish), dtype=np.float64)
    for t in range(nash):
        acc = np.zeros(nish, dtype=np.float64)
        for u in range(nash):
            ituu = int(ktuv[t, u, u])
            acc += d2[ituu, :]
        d1[t, :] = fact * acc
    return d1


def _spec1d_forward(ktu: np.ndarray, nash: int, sgm2: np.ndarray, sgm1: np.ndarray, *, fact: float) -> None:
    """Mirror OpenMolcas `spec1d.f` for IFC=0 (C1).

    Case D has an extra 1-electron part over the external `ia` indices.
    It is added into the *first* (W1) block of the D active dimension on
    the diagonal `tt` pairs.
    """
    for t in range(nash):
        itt = int(ktu[t, t])
        sgm2[itt, :] += fact * sgm1


def _spec1d_reverse(ktu: np.ndarray, nash: int, d2: np.ndarray, *, fact: float) -> np.ndarray:
    """Mirror OpenMolcas `spec1d.f` for IFC=1 (C1)."""
    acc = np.zeros(d2.shape[1], dtype=np.float64)
    for t in range(nash):
        itt = int(ktu[t, t])
        acc += d2[itt, :]
    return fact * acc


def _build_lists_c1(smap: SuperindexMap) -> dict[int, np.ndarray]:
    """Build OpenMolcas MKLIST coupling lists for C1.

    Ports `OpenMolcas/src/caspt2/mklist.f` but without symmetry offsets.
    Each list entry is `(L1, L2, L3, code)` where `code` is 0/1 selecting
    `VAL(1)` or `VAL(2)` in the Molcas kernels.
    """
    nash = int(smap.orbs.nash)
    nish = int(smap.orbs.nish)
    nssh = int(smap.orbs.nssh)
    ntu = int(smap.ntu)
    ntgeu = int(smap.ntgeu)
    ntgtu = int(smap.ntgtu)

    lists: dict[int, list[list[int]]] = {k: [] for k in range(1, 18)}

    # Lists 1 and 2: TUV/TU*
    for t in range(nash):
        for iuv in range(ntu):
            u, v = (int(x) for x in smap.mtu[iuv])
            ituv = int(smap.ktuv[t, u, v])
            iutv = int(smap.ktuv[u, t, v])
            ivut = int(smap.ktuv[v, u, t])
            # List 1: code always 1
            lists[1].append([ituv, t, iuv, 0])
            lists[1].append([iutv, t, iuv + ntu, 0])
            # List 2: second entry has code=2 (select VAL1(2))
            lists[2].append([ituv, t, iuv, 0])
            lists[2].append([ivut, t, iuv + ntu, 1])

    # Lists 3 and 5: TUV/TU+
    for t in range(nash):
        for iuv in range(ntgeu):
            u, v = (int(x) for x in smap.mtgeu[iuv])
            iuvt = int(smap.ktuv[u, v, t])
            iutv = int(smap.ktuv[u, t, v])
            if u == v:
                lists[3].append([iuvt, t, iuv, 1])  # code=2
                lists[5].append([iutv, t, iuv, 1])
            else:
                ivut = int(smap.ktuv[v, u, t])
                ivtu = int(smap.ktuv[v, t, u])
                lists[3].append([iuvt, t, iuv, 0])
                lists[3].append([ivut, t, iuv, 0])
                lists[5].append([iutv, t, iuv, 0])
                lists[5].append([ivtu, t, iuv, 0])

    # Lists 4 and 6: TUV/TU-
    for t in range(nash):
        for iuv in range(ntgtu):
            u, v = (int(x) for x in smap.mtgtu[iuv])
            iuvt = int(smap.ktuv[u, v, t])
            iutv = int(smap.ktuv[u, t, v])
            ivut = int(smap.ktuv[v, u, t])
            ivtu = int(smap.ktuv[v, t, u])
            lists[4].append([iuvt, t, iuv, 0])
            lists[4].append([ivut, t, iuv, 1])  # code=2
            lists[6].append([iutv, t, iuv, 0])
            lists[6].append([ivtu, t, iuv, 1])

    # Lists 7 and 8: TU*/T  (case D active superindex)
    noff = ntu
    for t in range(nash):
        for u in range(nash):
            iut1 = int(smap.ktu[u, t])
            itu1 = int(smap.ktu[t, u])
            lists[7].append([iut1, t, u, 0])
            lists[7].append([iut1 + noff, t, u, 1])
            lists[8].append([itu1, t, u, 0])
            lists[8].append([itu1 + noff, t, u, 1])

    # Lists 9 and 10: TU+-/T
    for t in range(nash):
        for u in range(nash):
            if t > u:
                lists[9].append([int(smap.ktgeu[t, u]), t, u, 0])
                lists[10].append([int(smap.ktgtu[t, u]), t, u, 0])
            elif t < u:
                lists[9].append([int(smap.ktgeu[u, t]), t, u, 0])
                lists[10].append([int(smap.ktgtu[u, t]), t, u, 1])
            else:
                lists[9].append([int(smap.ktgeu[t, u]), t, u, 1])

    # Lists 12 and 13: T/TU+-
    for t in range(nash):
        for u in range(nash):
            if t > u:
                lists[12].append([t, u, int(smap.ktgeu[t, u]), 0])
                lists[13].append([t, u, int(smap.ktgtu[t, u]), 0])
            elif t < u:
                lists[12].append([t, u, int(smap.ktgeu[u, t]), 0])
                lists[13].append([t, u, int(smap.ktgtu[u, t]), 1])
            else:
                lists[12].append([t, u, int(smap.ktgeu[t, u]), 1])

    # List 11: T/TU*  (case D, second block only)
    for t in range(nash):
        for u in range(nash):
            iut2 = noff + int(smap.ktu[u, t])
            lists[11].append([t, u, iut2, 0])  # code=1
        for u in range(nash):
            iuu2 = noff + int(smap.ktu[u, u])
            lists[11].append([t, t, iuu2, 1])  # code=2

    # Lists 14 and 15: I/IJ+-
    for i in range(nish):
        for j in range(nish):
            if i > j:
                lists[14].append([i, j, int(smap.kigej[i, j]), 0])
                lists[15].append([i, j, int(smap.kigtj[i, j]), 0])
            elif i < j:
                lists[14].append([i, j, int(smap.kigej[j, i]), 0])
                lists[15].append([i, j, int(smap.kigtj[j, i]), 1])
            else:
                lists[14].append([i, j, int(smap.kigej[i, j]), 1])

    # Lists 16 and 17: A/AB+-
    for a in range(nssh):
        for b in range(nssh):
            if a > b:
                lists[16].append([a, b, int(smap.kageb[a, b]), 0])
                lists[17].append([a, b, int(smap.kagtb[a, b]), 0])
            elif a < b:
                lists[16].append([a, b, int(smap.kageb[b, a]), 0])
                lists[17].append([a, b, int(smap.kagtb[b, a]), 1])
            else:
                lists[16].append([a, b, int(smap.kageb[a, b]), 1])

    out: dict[int, np.ndarray] = {}
    for k in range(1, 18):
        if lists[k]:
            out[k] = np.asarray(lists[k], dtype=np.int64)
        else:
            out[k] = np.empty((0, 4), dtype=np.int64)
    return out


def _mltsca(imltop: int, lst1: np.ndarray, lst2: np.ndarray, x: np.ndarray, f: np.ndarray, y: np.ndarray,
            *, val1: tuple[float, float], val2: tuple[float, float]) -> None:
    """C1 port of `mltsca.f` with explicit ndarray indexing.

    Modes:
      - `imltop=0`: update `x`
      - `imltop=1`: update `y`
      - `imltop=2`: update `f`
    """
    if lst1.size == 0 or lst2.size == 0:
        return
    v1 = np.array(val1, dtype=np.float64)
    v2 = np.array(val2, dtype=np.float64)
    if imltop == 0:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                x[int(l11), int(l21)] += a1 * float(v2[int(c2)]) * f[int(l12), int(l22)] * y[int(l13), int(l23)]
    elif imltop == 1:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                y[int(l13), int(l23)] += a1 * float(v2[int(c2)]) * f[int(l12), int(l22)] * x[int(l11), int(l21)]
    elif imltop == 2:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                f[int(l12), int(l22)] += a1 * float(v2[int(c2)]) * x[int(l11), int(l21)] * y[int(l13), int(l23)]
    else:
        raise ValueError(f"Unsupported IMLTOP={imltop} for MLTSCA")


def _mltmv(imltop: int, lst1: np.ndarray, x: np.ndarray, f: np.ndarray, y: np.ndarray,
           *, val1: tuple[float, float]) -> None:
    """C1 port of `mltmv.f` in formal tensor order.

    Shapes:
      - x[L1, i]
      - f[L2, a]
      - y[L3, i, a]
    """
    if lst1.size == 0:
        return
    v1 = np.array(val1, dtype=np.float64)
    if imltop == 0:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            x[int(l1), :] += a * (y[int(l3), :, :] @ f[int(l2), :])
    elif imltop == 1:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            y[int(l3), :, :] += a * np.outer(x[int(l1), :], f[int(l2), :])
    elif imltop == 2:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            f[int(l2), :] += a * (x[int(l1), :] @ y[int(l3), :, :])
    else:
        raise ValueError(f"Unsupported IMLTOP={imltop} for MLTMV")


def _mltdxp(imltop: int, lst1: np.ndarray, lst2: np.ndarray, x: np.ndarray, f: np.ndarray, y: np.ndarray,
            *, val1: tuple[float, float], val2: tuple[float, float]) -> None:
    """C1 port of `mltdxp.f` in formal tensor order.

    Shapes:
      - x[L11, L21, a]
      - f[L12, L22]
      - y[L13, L23, a]
    """
    if lst1.size == 0 or lst2.size == 0:
        return
    v1 = np.array(val1, dtype=np.float64)
    v2 = np.array(val2, dtype=np.float64)
    if imltop == 0:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                a = a1 * float(v2[int(c2)]) * f[int(l12), int(l22)]
                x[int(l11), int(l21), :] += a * y[int(l13), int(l23), :]
    elif imltop == 1:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                a = a1 * float(v2[int(c2)]) * f[int(l12), int(l22)]
                y[int(l13), int(l23), :] += a * x[int(l11), int(l21), :]
    elif imltop == 2:
        for l11, l12, l13, c1 in lst1:
            a1 = float(v1[int(c1)])
            for l21, l22, l23, c2 in lst2:
                a = a1 * float(v2[int(c2)])
                f[int(l12), int(l22)] += a * float(np.dot(x[int(l11), int(l21), :], y[int(l13), int(l23), :]))
    else:
        raise ValueError(f"Unsupported IMLTOP={imltop} for MLTDXP")


def _mltr1(imltop: int, lst1: np.ndarray, x: np.ndarray, f: np.ndarray, y: np.ndarray,
           *, val1: tuple[float, float]) -> None:
    """C1 port of `mltr1.f` in formal tensor order.

    Shapes:
      - x[L1, p, q]
      - f[L2, p]
      - y[L3, q]
    """
    if lst1.size == 0:
        return
    v1 = np.array(val1, dtype=np.float64)
    if imltop == 0:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            x[int(l1), :, :] += a * (f[int(l2), :, None] * y[int(l3), None, :])
    elif imltop == 1:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            y[int(l3), :] += a * (f[int(l2), :] @ x[int(l1), :, :])
    elif imltop == 2:
        for l1, l2, l3, c in lst1:
            a = float(v1[int(c)])
            f[int(l2), :] += a * (x[int(l1), :, :] @ y[int(l3), :])
    else:
        raise ValueError(f"Unsupported IMLTOP={imltop} for MLTR1")


@dataclass(frozen=True)
class SigmaC1CaseCoupling:
    """Full Molcas SIGMA_CASPT2 inter-case coupling operator (C1 symmetry).

    Extends ``SigmaC1ActiveVirtualCoupling`` to include *all* 24 IFCOUP coupling
    channels required for systems with inactive orbitals (``nish > 0``).
    Ports KOD 1–24 from ``OpenMolcas/src/caspt2/sgm.f``.

    The algorithm operates in two passes:

    **IMLTOP=0** (forward): For each "low" case, accumulates off-diagonal
    sigma contributions from all coupled "high" cases into the low case's
    covariant sigma vector.  Uses ``_mltsca``/``_mltmv``/``_mltdxp``/``_mltr1``
    tensor contraction primitives and precomputed MKLIST coupling lists (1–17).

    **IMLTOP=1** (reverse): For each (low, high) coupling pair, propagates
    the covariant density ``D = S·X(low)`` into the high case's sigma vector
    using the same kernel types in reverse mode.

    Special 1-electron folding (``spec1a``/``spec1c``/``spec1d``) handles the
    extra Fock/nactel corrections for cases A, C, and D.

    The full coupling table (from ``eqsolv.F90`` IFCOUP):
      KOD 1–2:   A ↔ B±    KOD 9–10:  C ↔ F±    KOD 17–18: E± ↔ H±
      KOD 3:     A ↔ D     KOD 11–12: C ↔ G±    KOD 19–20: F± ↔ G±
      KOD 4–5:   A ↔ E±    KOD 13–14: D ↔ E±    KOD 21–22: G± ↔ H±
      KOD 6–7:   B± ↔ E±   KOD 15–16: D ↔ G±    KOD 23–24: D ↔ H±
      KOD 8:     C ↔ D
    """

    smap: SuperindexMap
    fock: CASPT2Fock
    smats: list[np.ndarray]
    sb_decomp: list[SBDecomposition]
    h0_diag: list[np.ndarray]
    nactel: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lists", _build_lists_c1(self.smap))

    def __call__(self, vec_in: list[np.ndarray]) -> list[np.ndarray]:
        nish = int(self.smap.orbs.nish)
        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        if nash == 0:
            return [np.zeros_like(v) for v in vec_in]

        # Diagonal contribution in SR basis.
        sigma_sr = sigma_caspt2_diagonal(vec_in, self.sb_decomp, self.h0_diag)

        # x_C = T * x_SR (contravariant) in standard basis.
        x_c: list[np.ndarray] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            if v.size == 0 or decomp.nindep == 0 or nisup == 0:
                x_c.append(np.empty((0, 0), dtype=np.float64))
                continue
            x_sr = v.reshape(decomp.nindep, nisup)
            x_c.append(np.asarray(decomp.transform @ x_sr, dtype=np.float64, order="C"))

        # sigma_C = S * T * sigma_SR (covariant) for the diagonal part.
        sigma_c: list[np.ndarray] = []
        for case_idx, (v_sig, decomp) in enumerate(zip(sigma_sr, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            if v_sig.size == 0 or decomp.nindep == 0 or nisup == 0:
                sigma_c.append(np.empty((0, 0), dtype=np.float64))
                continue
            sig_sr = v_sig.reshape(decomp.nindep, nisup)
            sig_c_contrav = decomp.transform @ sig_sr
            sigma_c.append(np.asarray(self.smats[case_idx] @ sig_c_contrav, dtype=np.float64, order="C"))

        # Off-diagonal Fock blocks (Molcas FBLOCK).
        ao = nish
        vo = nish + nash
        f_ti = np.asarray(self.fock.fifa[ao:ao + nash, :nish], dtype=np.float64, order="C")          # (t,i)
        f_it = np.asarray(self.fock.fifa[:nish, ao:ao + nash], dtype=np.float64, order="C")          # (i,t)
        f_ia = np.asarray(self.fock.fifa[:nish, vo:vo + nssh], dtype=np.float64, order="C")          # (i,a)
        f_ai = np.asarray(self.fock.fifa[vo:vo + nssh, :nish], dtype=np.float64, order="C")          # (a,i)
        f_ta = np.asarray(self.fock.fifa[ao:ao + nash, vo:vo + nssh], dtype=np.float64, order="C")   # (t,a)
        f_at = np.asarray(self.fock.fifa[vo:vo + nssh, ao:ao + nash], dtype=np.float64, order="C")   # (a,t)

        self._apply_couplings_c1(
            x_c=x_c,
            sigma_c=sigma_c,
            f_ti=f_ti,
            f_it=f_it,
            f_ia=f_ia,
            f_ai=f_ai,
            f_ta=f_ta,
            f_at=f_at,
        )

        # Back to SR basis: sigma_SR = T^T * sigma_C
        sigma_out: list[np.ndarray] = []
        for case_idx, (decomp, sig_c) in enumerate(zip(self.sb_decomp, sigma_c)):
            nisup = int(self.smap.nisup[case_idx])
            if sig_c.size == 0 or decomp.nindep == 0 or nisup == 0:
                sigma_out.append(np.zeros_like(vec_in[case_idx]))
                continue
            sig_sr = decomp.transform.T @ sig_c
            sigma_out.append(np.asarray(sig_sr.ravel(), dtype=np.float64))
        return sigma_out

    def _apply_couplings_c1(
        self,
        *,
        x_c: list[np.ndarray],
        sigma_c: list[np.ndarray],
        f_ti: np.ndarray,
        f_it: np.ndarray,
        f_ia: np.ndarray,
        f_ai: np.ndarray,
        f_ta: np.ndarray,
        f_at: np.ndarray,
    ) -> None:
        # Coupling table (ICASE1<ICASE2) for C1. See `eqsolv.F90` IFCOUP.
        couplings = [
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

        lists = getattr(self, "_lists")
        nish = int(self.smap.orbs.nish)
        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        ntuv = int(self.smap.ntuv)
        ntu = int(self.smap.ntu)
        ntgeu = int(self.smap.ntgeu)
        ntgtu = int(self.smap.ntgtu)
        nigej = int(self.smap.nigej)
        nigtj = int(self.smap.nigtj)
        nageb = int(self.smap.nageb)
        nagtb = int(self.smap.nagtb)

        sqr2 = float(np.sqrt(2.0))
        sqr3 = float(np.sqrt(3.0))
        sqr6 = float(np.sqrt(6.0))
        sqri2 = 1.0 / sqr2
        sqri6 = 1.0 / sqr6
        sqr32 = float(np.sqrt(1.5))  # sqrt(3/2)

        fact = 1.0 / float(max(1, int(self.nactel)))

        # -----------------------------
        # IMLTOP=0: sigma(low) from x(high)
        # -----------------------------
        for low in range(1, 12):
            low_idx = low - 1
            nas1 = int(self.smap.nasup[low_idx])
            nis1 = int(self.smap.nisup[low_idx])
            if nas1 == 0 or nis1 == 0 or self.sb_decomp[low_idx].nindep == 0:
                continue

            sgm2 = np.zeros((nas1, nis1), dtype=np.float64)
            sgm1_ti = None
            sgm1_ta = None
            sgm1_ia = None
            if low == 1 and nish > 0:
                sgm1_ti = np.zeros((nash, nish), dtype=np.float64)
            if low == 4 and nssh > 0:
                sgm1_ta = np.zeros((nash, nssh), dtype=np.float64)
            if low == 5 and (nish * nssh) > 0:
                sgm1_ia = np.zeros((nish, nssh), dtype=np.float64)

            for c1, c2, kod in couplings:
                if c1 != low:
                    continue
                high_idx = c2 - 1
                y2 = x_c[high_idx]
                if y2.size == 0:
                    continue

                # Prepare per-kod views and apply kernels.
                if kod == 1:
                    # A <-> B+
                    _mltsca(0, lists[12], lists[14], sgm1_ti, f_ti, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                    _mltsca(0, lists[3], lists[14], sgm2, f_ti, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                elif kod == 2:
                    # A <-> B-
                    _mltsca(0, lists[13], lists[15], sgm1_ti, f_ti, y2, val1=(3.0, -3.0), val2=(1.0, -1.0))
                    _mltsca(0, lists[4], lists[15], sgm2, f_ti, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 3:
                    # A <-> D  (two-el)
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)  # (2*ntu, i, a)
                    _mltmv(0, lists[1], sgm2, f_ta, y3, val1=(1.0, 1.0))
                elif kod == 4:
                    # A <-> E+ (one-el)
                    # x: (i,t)
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)  # (igej, t, a)
                    _mltmv(0, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
                elif kod == 5:
                    # A <-> E- (one-el)
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)  # (igtj, t, a)
                    _mltmv(0, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
                elif kod == 6:
                    # B+ <-> E+ (two-el)
                    y3 = y2.reshape(nash, nigej, nssh, order="C")  # (u, igej, a)
                    _mltmv(0, lists[9], sgm2, f_ta, y3, val1=(sqri2, sqri2))
                elif kod == 7:
                    # B- <-> E- (two-el)
                    y3 = y2.reshape(nash, nigtj, nssh, order="C")  # (u, igtj, a)
                    _mltmv(0, lists[10], sgm2, f_ta, y3, val1=(sqri6, -sqri6))
                elif kod == 8:
                    # C <-> D (one-el + two-el)
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C")  # (tu*, a, i)
                    _mltmv(0, lists[11], sgm1_ta, f_ti, y3, val1=(2.0, 1.0))
                    _mltmv(0, lists[2], sgm2, f_ti, y3, val1=(-1.0, -1.0))
                elif kod == 9:
                    # C <-> F+ (one-el + two-el)
                    _mltsca(0, lists[12], lists[16], sgm1_ta, f_ta, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                    _mltsca(0, lists[5], lists[16], sgm2, f_ta, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                elif kod == 10:
                    # C <-> F- (one-el + two-el)
                    _mltsca(0, lists[13], lists[17], sgm1_ta, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                    _mltsca(0, lists[6], lists[17], sgm2, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 11:
                    # C <-> G+ (one-el)
                    x_view = sgm1_ta.T  # (a,t)
                    y3 = y2.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)  # (ageb, t, i)
                    _mltmv(0, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
                elif kod == 12:
                    # C <-> G- (one-el)
                    x_view = sgm1_ta.T
                    y3 = y2.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)  # (agtb, t, i)
                    _mltmv(0, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
                elif kod == 13:
                    # D <-> E+ (one-el + two-el)
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)  # (igej, a, t)
                    _mltmv(0, lists[14], sgm1_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)  # (tu*, i, a)
                    ydxp = y2.reshape(nash, nigej, nssh, order="C")  # (u, igej, a)
                    _mltdxp(0, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
                elif kod == 14:
                    # D <-> E- (one-el + two-el)
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)  # (igtj, a, t)
                    _mltmv(0, lists[15], sgm1_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = y2.reshape(nash, nigtj, nssh, order="C")
                    _mltdxp(0, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 15:
                    # D <-> G+ (two-el)
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")  # (tu*, a, i)
                    ydxp = y2.reshape(nash, nageb, nish, order="C")  # (t, ageb, i)
                    _mltdxp(0, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
                elif kod == 16:
                    # D <-> G- (two-el)
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = y2.reshape(nash, nagtb, nish, order="C")
                    _mltdxp(0, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 17:
                    # E+ <-> H+ (two-el, rank-1): list16 with FTA
                    x_r1 = sgm2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)  # (a, t, igej)
                    _mltr1(0, lists[16], x_r1, f_ta.T, y2, val1=(sqri2, 1.0))
                elif kod == 18:
                    # E- <-> H- (two-el, rank-1): list17 with FTA
                    x_r1 = sgm2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)  # (a, t, igtj)
                    _mltr1(0, lists[17], x_r1, f_ta.T, y2, val1=(sqri2, -sqri2))
                elif kod == 19:
                    # F+ <-> G+ (two-el): list9 with FIT
                    y3 = y2.reshape(nash, nageb, nish, order="C")  # (u, ageb, i)
                    _mltmv(0, lists[9], sgm2, f_ti, y3, val1=(-sqri2, -sqri2))
                elif kod == 20:
                    # F- <-> G- (two-el): list10 with FIT
                    y3 = y2.reshape(nash, nagtb, nish, order="C")  # (u, agtb, i)
                    _mltmv(0, lists[10], sgm2, f_ti, y3, val1=(-sqri6, sqri6))
                elif kod == 21:
                    # G+ <-> H+ (two-el, rank-1): list14 with FTI
                    x_r1 = sgm2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)  # (i, t, ageb)
                    _mltr1(0, lists[14], x_r1, f_ti.T, y2.T, val1=(-sqri2, -1.0))
                elif kod == 22:
                    # G- <-> H- (two-el, rank-1): list15 with FTI
                    x_r1 = sgm2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)  # (i, t, agtb)
                    _mltr1(0, lists[15], x_r1, f_ti.T, y2.T, val1=(sqri2, -sqri2))
                elif kod == 23:
                    # D <-> H+ (one-el): list14/list16 with FIA
                    _mltsca(0, lists[14], lists[16], sgm1_ia, f_ia, y2.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
                elif kod == 24:
                    # D <-> H- (one-el): list15/list17 with FIA
                    half_sqr3 = 0.5 * sqr3
                    _mltsca(0, lists[15], lists[17], sgm1_ia, f_ia, y2.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))
                else:
                    raise ValueError(f"Unsupported KOD={kod}")

            # Special 1-electron folding for cases A/C/D.
            if low == 1 and sgm1_ti is not None:
                _spec1a_forward(self.smap.ktuv, nash, sgm2, sgm1_ti, fact=fact)
            elif low == 4 and sgm1_ta is not None:
                _spec1c_forward(self.smap.ktuv, nash, sgm2, sgm1_ta, fact=fact)
            elif low == 5 and sgm1_ia is not None:
                # Spec1D uses external ia vector; flatten in our storage (i,a) -> (a*nish+i).
                _spec1d_forward(self.smap.ktu, nash, sgm2, sgm1_ia.T.reshape(-1, order="C"), fact=fact)

            # Add S * sgm2 into covariant sigma for this case.
            sigma_c[low_idx] += self.smats[low_idx] @ sgm2

        # -----------------------------
        # IMLTOP=1: sigma(high) from D(low)=S*X(low) (+ SPEC1 reverse)
        # -----------------------------
        for low, high, kod in couplings:
            low_idx = low - 1
            high_idx = high - 1
            x_low = x_c[low_idx]
            if x_low.size == 0:
                continue
            sig_high = sigma_c[high_idx]
            if sig_high.size == 0:
                continue

            # D2 = S * X (covariant) for the low case.
            d2 = self.smats[low_idx] @ x_low
            d1 = None
            if low == 1:
                d1 = _spec1a_reverse(self.smap.ktuv, nash, nish, d2, fact=fact)
            elif low == 4:
                d1 = _spec1c_reverse(self.smap.ktuv, nash, nssh, d2, fact=fact)
            elif low == 5:
                d1 = _spec1d_reverse(self.smap.ktu, nash, d2, fact=fact)

            # Apply kernels in reverse direction (IMLTOP=1) using the same views.
            if kod == 1:
                _mltsca(1, lists[12], lists[14], d1, f_ti, sig_high, val1=(1.0, 2.0), val2=(1.0, sqr2))
                _mltsca(1, lists[3], lists[14], d2, f_ti, sig_high, val1=(-1.0, -2.0), val2=(1.0, sqr2))
            elif kod == 2:
                _mltsca(1, lists[13], lists[15], d1, f_ti, sig_high, val1=(3.0, -3.0), val2=(1.0, -1.0))
                _mltsca(1, lists[4], lists[15], d2, f_ti, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
            elif kod == 3:
                y3 = sig_high.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                _mltmv(1, lists[1], d2, f_ta, y3, val1=(1.0, 1.0))
            elif kod == 4:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)
                _mltmv(1, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
            elif kod == 5:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)
                _mltmv(1, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
            elif kod == 6:
                y3 = sig_high.reshape(nash, nigej, nssh, order="C")
                _mltmv(1, lists[9], d2, f_ta, y3, val1=(sqri2, sqri2))
            elif kod == 7:
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C")
                _mltmv(1, lists[10], d2, f_ta, y3, val1=(sqri6, -sqri6))
            elif kod == 8:
                y3 = sig_high.reshape(2 * ntu, nssh, nish, order="C")
                _mltmv(1, lists[11], d1, f_ti, y3, val1=(2.0, 1.0))
                _mltmv(1, lists[2], d2, f_ti, y3, val1=(-1.0, -1.0))
            elif kod == 9:
                _mltsca(1, lists[12], lists[16], d1, f_ta, sig_high, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                _mltsca(1, lists[5], lists[16], d2, f_ta, sig_high, val1=(1.0, 2.0), val2=(1.0, sqr2))
            elif kod == 10:
                _mltsca(1, lists[13], lists[17], d1, f_ta, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                _mltsca(1, lists[6], lists[17], d2, f_ta, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
            elif kod == 11:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)
                _mltmv(1, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
            elif kod == 12:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)
                _mltmv(1, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
            elif kod == 13:
                y3 = sig_high.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)
                x_ia = d1.reshape(nssh, nish, order="C").T  # (i, a)
                _mltmv(1, lists[14], x_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                ydxp = sig_high.reshape(nash, nigej, nssh, order="C")
                _mltdxp(1, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
            elif kod == 14:
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)
                x_ia = d1.reshape(nssh, nish, order="C").T
                _mltmv(1, lists[15], x_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                ydxp = sig_high.reshape(nash, nigtj, nssh, order="C")
                _mltdxp(1, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
            elif kod == 15:
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                ydxp = sig_high.reshape(nash, nageb, nish, order="C")
                _mltdxp(1, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
            elif kod == 16:
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                ydxp = sig_high.reshape(nash, nagtb, nish, order="C")
                _mltdxp(1, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
            elif kod == 17:
                x_r1 = d2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)  # (a, t, igej)
                _mltr1(1, lists[16], x_r1, f_ta.T, sig_high, val1=(sqri2, 1.0))
            elif kod == 18:
                x_r1 = d2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)  # (a, t, igtj)
                _mltr1(1, lists[17], x_r1, f_ta.T, sig_high, val1=(sqri2, -sqri2))
            elif kod == 19:
                y3 = sig_high.reshape(nash, nageb, nish, order="C")
                _mltmv(1, lists[9], d2, f_ti, y3, val1=(-sqri2, -sqri2))
            elif kod == 20:
                y3 = sig_high.reshape(nash, nagtb, nish, order="C")
                _mltmv(1, lists[10], d2, f_ti, y3, val1=(-sqri6, sqri6))
            elif kod == 21:
                x_r1 = d2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)  # (i, t, ageb)
                _mltr1(1, lists[14], x_r1, f_ti.T, sig_high.T, val1=(-sqri2, -1.0))
            elif kod == 22:
                x_r1 = d2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)  # (i, t, agtb)
                _mltr1(1, lists[15], x_r1, f_ti.T, sig_high.T, val1=(sqri2, -sqri2))
            elif kod == 23:
                x_ia = d1.reshape(nssh, nish, order="C").T
                _mltsca(1, lists[14], lists[16], x_ia, f_ia, sig_high.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
            elif kod == 24:
                half_sqr3 = 0.5 * sqr3
                x_ia = d1.reshape(nssh, nish, order="C").T
                _mltsca(1, lists[15], lists[17], x_ia, f_ia, sig_high.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))
            else:
                raise NotImplementedError(f"KOD={kod} reverse coupling not implemented")
