from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.cuda import kernels
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.overlap import SBDecomposition
from asuka.caspt2.sigma import _build_lists_c1
from asuka.caspt2.superindex import SuperindexMap


def _lists_to_device_soa(smap: SuperindexMap, cp):
    """Build MKLIST tables on CPU and move to GPU as SoA int32 arrays (4, n)."""
    lists = _build_lists_c1(smap)
    out: dict[int, object] = {}
    for k, a in lists.items():
        arr = np.asarray(a, dtype=np.int64)
        if arr.size == 0:
            out[k] = cp.empty((4, 0), dtype=cp.int32)
            continue
        # (n,4) -> (4,n)
        out[k] = cp.ascontiguousarray(cp.asarray(arr, dtype=cp.int32).T)
    return out


def _spec1a_forward(cp, *, nash: int, sgm2, sgm1, fact: float) -> None:
    # ktuv[t,u,u] in C1 is t*nash*nash + u*nash + u.
    u = cp.arange(nash, dtype=cp.int64)
    diag_u = u * nash + u
    for t in range(nash):
        rows = int(t) * (nash * nash) + diag_u
        sgm2[rows, :] += float(fact) * sgm1[t, :][None, :]


def _spec1a_reverse(cp, *, nash: int, d2, fact: float):
    u = cp.arange(nash, dtype=cp.int64)
    diag_u = u * nash + u
    out = cp.zeros((nash, int(d2.shape[1])), dtype=cp.float64)
    for t in range(nash):
        rows = int(t) * (nash * nash) + diag_u
        out[t, :] = float(fact) * cp.sum(d2[rows, :], axis=0)
    return out


def _spec1c_forward(cp, *, nash: int, sgm2, sgm1, fact: float) -> None:
    # Same indexing as spec1a (C and A share TUV active superindex).
    _spec1a_forward(cp, nash=nash, sgm2=sgm2, sgm1=sgm1, fact=fact)


def _spec1c_reverse(cp, *, nash: int, d2, fact: float):
    return _spec1a_reverse(cp, nash=nash, d2=d2, fact=fact)


def _spec1d_forward(cp, *, nash: int, sgm2, sgm1_flat, fact: float) -> None:
    # ktu[t,t] in C1 ordered-pair indexing is t*nash + t, and Spec1D updates only the first (W1) block.
    t = cp.arange(nash, dtype=cp.int64)
    rows = t * nash + t
    sgm2[rows, :] += float(fact) * sgm1_flat[None, :]


def _spec1d_reverse(cp, *, nash: int, d2, fact: float):
    t = cp.arange(nash, dtype=cp.int64)
    rows = t * nash + t
    return float(fact) * cp.sum(d2[rows, :], axis=0)


@dataclass
class SigmaC1CaseCouplingCuda:
    """GPU analogue of `asuka.caspt2.sigma.SigmaC1CaseCoupling` (C1)."""

    smap: SuperindexMap
    fock: CASPT2Fock
    smats: list[np.ndarray]
    sb_decomp: list[SBDecomposition]
    bd: list[np.ndarray]  # (nindep,) per case
    id: list[np.ndarray]  # (nisup,) per case
    nactel: int
    real_shift: float = 0.0
    imag_shift: float = 0.0
    device: int | None = None

    def __post_init__(self) -> None:
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for CASPT2 CUDA sigma") from e

        if self.device is not None:
            cp.cuda.Device(int(self.device)).use()

        self._cp = cp
        self._lists = _lists_to_device_soa(self.smap, cp)

        # Device-side S matrices and SR transforms.
        self._smats_d: list[object] = []
        self._t_d: list[object] = []
        self._tt_d: list[object] = []
        for smat, decomp in zip(self.smats, self.sb_decomp):
            self._smats_d.append(cp.ascontiguousarray(cp.asarray(smat, dtype=cp.float64)))
            self._t_d.append(cp.ascontiguousarray(cp.asarray(decomp.transform, dtype=cp.float64)))
            self._tt_d.append(cp.ascontiguousarray(cp.asarray(decomp.transform.T, dtype=cp.float64)))

        self._bd_d = [cp.ascontiguousarray(cp.asarray(b, dtype=cp.float64).ravel()) for b in self.bd]
        self._id_d = [cp.ascontiguousarray(cp.asarray(i, dtype=cp.float64).ravel()) for i in self.id]

        # Off-diagonal Fock blocks on device.
        nish = int(self.smap.orbs.nish)
        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        ao = nish
        vo = nish + nash
        fifa = np.asarray(self.fock.fifa, dtype=np.float64, order="C")
        self._f_ti = cp.ascontiguousarray(cp.asarray(fifa[ao : ao + nash, :nish], dtype=cp.float64))
        self._f_it = cp.ascontiguousarray(cp.asarray(fifa[:nish, ao : ao + nash], dtype=cp.float64))
        self._f_ia = cp.ascontiguousarray(cp.asarray(fifa[:nish, vo : vo + nssh], dtype=cp.float64))
        self._f_ai = cp.ascontiguousarray(cp.asarray(fifa[vo : vo + nssh, :nish], dtype=cp.float64))
        self._f_ta = cp.ascontiguousarray(cp.asarray(fifa[ao : ao + nash, vo : vo + nssh], dtype=cp.float64))
        self._f_at = cp.ascontiguousarray(cp.asarray(fifa[vo : vo + nssh, ao : ao + nash], dtype=cp.float64))

    def __call__(self, vec_in: list[object]) -> list[object]:
        cp = self._cp
        nash = int(self.smap.orbs.nash)
        nish = int(self.smap.orbs.nish)
        nssh = int(self.smap.orbs.nssh)
        if nash == 0:
            return [cp.zeros_like(v) for v in vec_in]

        # Diagonal (H0(diag)-E0) contribution in SR basis.
        sigma_sr: list[object] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_sr.append(cp.zeros_like(v))
                continue
            x = v.reshape(nin, nisup)
            y = cp.empty_like(x)
            kernels.apply_h0diag_sr(
                y=y,
                x=x,
                bd=self._bd_d[case_idx],
                id=self._id_d[case_idx],
                real_shift=float(self.real_shift),
                imag_shift=float(self.imag_shift),
                alpha=1.0,
                beta=0.0,
            )
            sigma_sr.append(y.ravel())

        # x_C = T * x_SR (contravariant) in standard basis.
        x_c: list[object] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                x_c.append(cp.empty((0, 0), dtype=cp.float64))
                continue
            x_sr = v.reshape(nin, nisup)
            x_c.append(self._t_d[case_idx] @ x_sr)

        # sigma_C = S * T * sigma_SR (covariant) for the diagonal part.
        sigma_c: list[object] = []
        for case_idx, (v_sig, decomp) in enumerate(zip(sigma_sr, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_c.append(cp.empty((0, 0), dtype=cp.float64))
                continue
            sig_sr = v_sig.reshape(nin, nisup)
            sig_c_contrav = self._t_d[case_idx] @ sig_sr
            sigma_c.append(self._smats_d[case_idx] @ sig_c_contrav)

        self._apply_couplings_c1(
            x_c=x_c,
            sigma_c=sigma_c,
            f_ti=self._f_ti,
            f_it=self._f_it,
            f_ia=self._f_ia,
            f_ai=self._f_ai,
            f_ta=self._f_ta,
            f_at=self._f_at,
        )

        # Back to SR basis: sigma_SR = T^T * sigma_C
        sigma_out: list[object] = []
        for case_idx, (decomp, sig_c) in enumerate(zip(self.sb_decomp, sigma_c)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_out.append(cp.zeros_like(vec_in[case_idx]))
                continue
            sig_sr = self._tt_d[case_idx] @ sig_c
            sigma_out.append(sig_sr.ravel())
        return sigma_out

    def profile(self, vec_in: list[object]) -> tuple[list[object], dict[str, float]]:
        """Profile a single sigma application (one matvec).

        Returns (sigma_out, stats) where stats is a dict of coarse stage timings
        in seconds measured with CUDA events on the default stream.
        """
        cp = self._cp
        nash = int(self.smap.orbs.nash)
        if nash == 0:
            return [cp.zeros_like(v) for v in vec_in], {
                "h0diag_s": 0.0,
                "sr_to_c_s": 0.0,
                "diag_to_cov_s": 0.0,
                "couplings_s": 0.0,
                "c_to_sr_s": 0.0,
                "total_s": 0.0,
            }

        stream = cp.cuda.Stream.null
        ev = [cp.cuda.Event() for _ in range(6)]
        ev[0].record(stream)

        # Stage 1: diagonal apply (SR basis)
        sigma_sr: list[object] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_sr.append(cp.zeros_like(v))
                continue
            x = v.reshape(nin, nisup)
            y = cp.empty_like(x)
            kernels.apply_h0diag_sr(
                y=y,
                x=x,
                bd=self._bd_d[case_idx],
                id=self._id_d[case_idx],
                real_shift=float(self.real_shift),
                imag_shift=float(self.imag_shift),
                alpha=1.0,
                beta=0.0,
            )
            sigma_sr.append(y.ravel())
        ev[1].record(stream)

        # Stage 2: SR->C transforms (contravariant)
        x_c: list[object] = []
        for case_idx, (v, decomp) in enumerate(zip(vec_in, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                x_c.append(cp.empty((0, 0), dtype=cp.float64))
                continue
            x_sr = v.reshape(nin, nisup)
            x_c.append(self._t_d[case_idx] @ x_sr)
        ev[2].record(stream)

        # Stage 3: diagonal part to covariant in C basis: sigma_C = S * T * sigma_SR
        sigma_c: list[object] = []
        for case_idx, (v_sig, decomp) in enumerate(zip(sigma_sr, self.sb_decomp)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_c.append(cp.empty((0, 0), dtype=cp.float64))
                continue
            sig_sr = v_sig.reshape(nin, nisup)
            sig_c_contrav = self._t_d[case_idx] @ sig_sr
            sigma_c.append(self._smats_d[case_idx] @ sig_c_contrav)
        ev[3].record(stream)

        # Stage 4: off-diagonal couplings (C basis)
        self._apply_couplings_c1(
            x_c=x_c,
            sigma_c=sigma_c,
            f_ti=self._f_ti,
            f_it=self._f_it,
            f_ia=self._f_ia,
            f_ai=self._f_ai,
            f_ta=self._f_ta,
            f_at=self._f_at,
        )
        ev[4].record(stream)

        # Stage 5: C->SR back-transform
        sigma_out: list[object] = []
        for case_idx, (decomp, sig_c) in enumerate(zip(self.sb_decomp, sigma_c)):
            nisup = int(self.smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                sigma_out.append(cp.zeros_like(vec_in[case_idx]))
                continue
            sig_sr = self._tt_d[case_idx] @ sig_c
            sigma_out.append(sig_sr.ravel())
        ev[5].record(stream)

        ev[5].synchronize()

        def _dt(i: int, j: int) -> float:
            return float(cp.cuda.get_elapsed_time(ev[i], ev[j])) * 1e-3

        stats = {
            "h0diag_s": _dt(0, 1),
            "sr_to_c_s": _dt(1, 2),
            "diag_to_cov_s": _dt(2, 3),
            "couplings_s": _dt(3, 4),
            "c_to_sr_s": _dt(4, 5),
            "total_s": _dt(0, 5),
        }
        return sigma_out, stats

    def _apply_couplings_c1(
        self,
        *,
        x_c: list[object],
        sigma_c: list[object],
        f_ti,
        f_it,
        f_ia,
        f_ai,
        f_ta,
        f_at,
    ) -> None:
        cp = self._cp

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

        lists = self._lists
        nish = int(self.smap.orbs.nish)
        nash = int(self.smap.orbs.nash)
        nssh = int(self.smap.orbs.nssh)
        ntu = int(self.smap.ntu)
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
            if nas1 == 0 or nis1 == 0 or int(self.sb_decomp[low_idx].nindep) == 0:
                continue

            sgm2 = cp.zeros((nas1, nis1), dtype=cp.float64)
            sgm1_ti = None
            sgm1_ta = None
            sgm1_ia = None
            if low == 1 and nish > 0:
                sgm1_ti = cp.zeros((nash, nish), dtype=cp.float64)
            if low == 4 and nssh > 0:
                sgm1_ta = cp.zeros((nash, nssh), dtype=cp.float64)
            if low == 5 and (nish * nssh) > 0:
                sgm1_ia = cp.zeros((nish, nssh), dtype=cp.float64)

            for c1, c2, kod in couplings:
                if c1 != low:
                    continue
                high_idx = c2 - 1
                y2 = x_c[high_idx]
                if int(getattr(y2, "size", 0)) == 0:
                    continue

                if kod == 1:
                    kernels.mltsca(0, lists[12], lists[14], sgm1_ti, f_ti, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                    kernels.mltsca(0, lists[3], lists[14], sgm2, f_ti, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                elif kod == 2:
                    kernels.mltsca(0, lists[13], lists[15], sgm1_ti, f_ti, y2, val1=(3.0, -3.0), val2=(1.0, -1.0))
                    kernels.mltsca(0, lists[4], lists[15], sgm2, f_ti, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 3:
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)  # (2*ntu, i, a)
                    kernels.mltmv(0, lists[1], sgm2, f_ta, y3, val1=(1.0, 1.0))
                elif kod == 4:
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)  # (igej, t, a)
                    kernels.mltmv(0, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
                elif kod == 5:
                    x_view = sgm1_ti.T
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)  # (igtj, t, a)
                    kernels.mltmv(0, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
                elif kod == 6:
                    y3 = y2.reshape(nash, nigej, nssh, order="C")  # (u, igej, a)
                    kernels.mltmv(0, lists[9], sgm2, f_ta, y3, val1=(sqri2, sqri2))
                elif kod == 7:
                    y3 = y2.reshape(nash, nigtj, nssh, order="C")  # (u, igtj, a)
                    kernels.mltmv(0, lists[10], sgm2, f_ta, y3, val1=(sqri6, -sqri6))
                elif kod == 8:
                    y3 = y2.reshape(2 * ntu, nssh, nish, order="C")  # (tu*, a, i)
                    kernels.mltmv(0, lists[11], sgm1_ta, f_ti, y3, val1=(2.0, 1.0))
                    kernels.mltmv(0, lists[2], sgm2, f_ti, y3, val1=(-1.0, -1.0))
                elif kod == 9:
                    kernels.mltsca(0, lists[12], lists[16], sgm1_ta, f_ta, y2, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                    kernels.mltsca(0, lists[5], lists[16], sgm2, f_ta, y2, val1=(1.0, 2.0), val2=(1.0, sqr2))
                elif kod == 10:
                    kernels.mltsca(0, lists[13], lists[17], sgm1_ta, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                    kernels.mltsca(0, lists[6], lists[17], sgm2, f_ta, y2, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                elif kod == 11:
                    x_view = sgm1_ta.T  # (a,t)
                    y3 = y2.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)  # (ageb, t, i)
                    kernels.mltmv(0, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
                elif kod == 12:
                    x_view = sgm1_ta.T
                    y3 = y2.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)  # (agtb, t, i)
                    kernels.mltmv(0, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
                elif kod == 13:
                    y3 = y2.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)  # (igej, a, t)
                    kernels.mltmv(0, lists[14], sgm1_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)  # (tu*, i, a)
                    ydxp = y2.reshape(nash, nigej, nssh, order="C")  # (u, igej, a)
                    kernels.mltdxp(0, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
                elif kod == 14:
                    y3 = y2.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)  # (igtj, a, t)
                    kernels.mltmv(0, lists[15], sgm1_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                    ydxp = y2.reshape(nash, nigtj, nssh, order="C")
                    kernels.mltdxp(0, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 15:
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")  # (tu*, a, i)
                    ydxp = y2.reshape(nash, nageb, nish, order="C")  # (t, ageb, i)
                    kernels.mltdxp(0, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
                elif kod == 16:
                    x3 = sgm2.reshape(2 * ntu, nssh, nish, order="C")
                    ydxp = y2.reshape(nash, nagtb, nish, order="C")
                    kernels.mltdxp(0, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
                elif kod == 17:
                    x_r1 = sgm2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)  # (a, t, igej)
                    kernels.mltr1(0, lists[16], x_r1, f_ta.T, y2, val1=(sqri2, 1.0))
                elif kod == 18:
                    x_r1 = sgm2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)  # (a, t, igtj)
                    kernels.mltr1(0, lists[17], x_r1, f_ta.T, y2, val1=(sqri2, -sqri2))
                elif kod == 19:
                    y3 = y2.reshape(nash, nageb, nish, order="C")  # (u, ageb, i)
                    kernels.mltmv(0, lists[9], sgm2, f_ti, y3, val1=(-sqri2, -sqri2))
                elif kod == 20:
                    y3 = y2.reshape(nash, nagtb, nish, order="C")  # (u, agtb, i)
                    kernels.mltmv(0, lists[10], sgm2, f_ti, y3, val1=(-sqri6, sqri6))
                elif kod == 21:
                    x_r1 = sgm2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)  # (i, t, ageb)
                    kernels.mltr1(0, lists[14], x_r1, f_ti.T, y2.T, val1=(-sqri2, -1.0))
                elif kod == 22:
                    x_r1 = sgm2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)  # (i, t, agtb)
                    kernels.mltr1(0, lists[15], x_r1, f_ti.T, y2.T, val1=(sqri2, -sqri2))
                elif kod == 23:
                    kernels.mltsca(0, lists[14], lists[16], sgm1_ia, f_ia, y2.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
                elif kod == 24:
                    half_sqr3 = 0.5 * sqr3
                    kernels.mltsca(0, lists[15], lists[17], sgm1_ia, f_ia, y2.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))
                else:
                    raise ValueError(f"Unsupported KOD={kod}")

            # Special 1-electron folding for cases A/C/D.
            if low == 1 and sgm1_ti is not None:
                _spec1a_forward(cp, nash=nash, sgm2=sgm2, sgm1=sgm1_ti, fact=fact)
            elif low == 4 and sgm1_ta is not None:
                _spec1c_forward(cp, nash=nash, sgm2=sgm2, sgm1=sgm1_ta, fact=fact)
            elif low == 5 and sgm1_ia is not None:
                _spec1d_forward(cp, nash=nash, sgm2=sgm2, sgm1_flat=sgm1_ia.T.reshape(-1, order="C"), fact=fact)

            sigma_c[low_idx] += self._smats_d[low_idx] @ sgm2

        # -----------------------------
        # IMLTOP=1: sigma(high) from D(low)=S*X(low) (+ SPEC1 reverse)
        # -----------------------------
        for low, high, kod in couplings:
            low_idx = low - 1
            high_idx = high - 1
            x_low = x_c[low_idx]
            if int(getattr(x_low, "size", 0)) == 0:
                continue
            sig_high = sigma_c[high_idx]
            if int(getattr(sig_high, "size", 0)) == 0:
                continue

            d2 = self._smats_d[low_idx] @ x_low
            d1 = None
            if low == 1:
                d1 = _spec1a_reverse(cp, nash=nash, d2=d2, fact=fact)
            elif low == 4:
                d1 = _spec1c_reverse(cp, nash=nash, d2=d2, fact=fact)
            elif low == 5:
                d1 = _spec1d_reverse(cp, nash=nash, d2=d2, fact=fact)

            if kod == 1:
                kernels.mltsca(1, lists[12], lists[14], d1, f_ti, sig_high, val1=(1.0, 2.0), val2=(1.0, sqr2))
                kernels.mltsca(1, lists[3], lists[14], d2, f_ti, sig_high, val1=(-1.0, -2.0), val2=(1.0, sqr2))
            elif kod == 2:
                kernels.mltsca(1, lists[13], lists[15], d1, f_ti, sig_high, val1=(3.0, -3.0), val2=(1.0, -1.0))
                kernels.mltsca(1, lists[4], lists[15], d2, f_ti, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
            elif kod == 3:
                y3 = sig_high.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                kernels.mltmv(1, lists[1], d2, f_ta, y3, val1=(1.0, 1.0))
            elif kod == 4:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nigej, nssh, order="C").transpose(1, 0, 2)
                kernels.mltmv(1, lists[14], x_view, f_ia, y3, val1=(1.0, sqr2))
            elif kod == 5:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C").transpose(1, 0, 2)
                kernels.mltmv(1, lists[15], x_view, f_ia, y3, val1=(-sqr3, sqr3))
            elif kod == 6:
                y3 = sig_high.reshape(nash, nigej, nssh, order="C")
                kernels.mltmv(1, lists[9], d2, f_ta, y3, val1=(sqri2, sqri2))
            elif kod == 7:
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C")
                kernels.mltmv(1, lists[10], d2, f_ta, y3, val1=(sqri6, -sqri6))
            elif kod == 8:
                y3 = sig_high.reshape(2 * ntu, nssh, nish, order="C")
                kernels.mltmv(1, lists[11], d1, f_ti, y3, val1=(2.0, 1.0))
                kernels.mltmv(1, lists[2], d2, f_ti, y3, val1=(-1.0, -1.0))
            elif kod == 9:
                kernels.mltsca(1, lists[12], lists[16], d1, f_ta, sig_high, val1=(-1.0, -2.0), val2=(1.0, sqr2))
                kernels.mltsca(1, lists[5], lists[16], d2, f_ta, sig_high, val1=(1.0, 2.0), val2=(1.0, sqr2))
            elif kod == 10:
                kernels.mltsca(1, lists[13], lists[17], d1, f_ta, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
                kernels.mltsca(1, lists[6], lists[17], d2, f_ta, sig_high, val1=(-1.0, 1.0), val2=(1.0, -1.0))
            elif kod == 11:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nageb, nish, order="C").transpose(1, 0, 2)
                kernels.mltmv(1, lists[16], x_view, f_ai, y3, val1=(sqri2, 1.0))
            elif kod == 12:
                x_view = d1.T
                y3 = sig_high.reshape(nash, nagtb, nish, order="C").transpose(1, 0, 2)
                kernels.mltmv(1, lists[17], x_view, f_ai, y3, val1=(sqr32, -sqr32))
            elif kod == 13:
                y3 = sig_high.reshape(nash, nigej, nssh, order="C").transpose(1, 2, 0)
                x_ia = d1.reshape(nssh, nish, order="C").T
                kernels.mltmv(1, lists[14], x_ia, f_ti.T, y3, val1=(sqri2, 1.0))
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                ydxp = sig_high.reshape(nash, nigej, nssh, order="C")
                kernels.mltdxp(1, lists[7], lists[14], x3, f_ti, ydxp, val1=(-1.0, -1.0), val2=(sqri2, 1.0))
            elif kod == 14:
                y3 = sig_high.reshape(nash, nigtj, nssh, order="C").transpose(1, 2, 0)
                x_ia = d1.reshape(nssh, nish, order="C").T
                kernels.mltmv(1, lists[15], x_ia, f_ti.T, y3, val1=(sqr32, -sqr32))
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C").transpose(0, 2, 1)
                ydxp = sig_high.reshape(nash, nigtj, nssh, order="C")
                kernels.mltdxp(1, lists[7], lists[15], x3, f_ti, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
            elif kod == 15:
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                ydxp = sig_high.reshape(nash, nageb, nish, order="C")
                kernels.mltdxp(1, lists[8], lists[16], x3, f_ta, ydxp, val1=(1.0, 1.0), val2=(sqri2, 1.0))
            elif kod == 16:
                x3 = d2.reshape(2 * ntu, nssh, nish, order="C")
                ydxp = sig_high.reshape(nash, nagtb, nish, order="C")
                kernels.mltdxp(1, lists[8], lists[17], x3, f_ta, ydxp, val1=(-1.0, 1.0), val2=(sqri6, -sqri6))
            elif kod == 17:
                x_r1 = d2.reshape(nash, nigej, nssh, order="C").transpose(2, 0, 1)
                kernels.mltr1(1, lists[16], x_r1, f_ta.T, sig_high, val1=(sqri2, 1.0))
            elif kod == 18:
                x_r1 = d2.reshape(nash, nigtj, nssh, order="C").transpose(2, 0, 1)
                kernels.mltr1(1, lists[17], x_r1, f_ta.T, sig_high, val1=(sqri2, -sqri2))
            elif kod == 19:
                y3 = sig_high.reshape(nash, nageb, nish, order="C")
                kernels.mltmv(1, lists[9], d2, f_ti, y3, val1=(-sqri2, -sqri2))
            elif kod == 20:
                y3 = sig_high.reshape(nash, nagtb, nish, order="C")
                kernels.mltmv(1, lists[10], d2, f_ti, y3, val1=(-sqri6, sqri6))
            elif kod == 21:
                x_r1 = d2.reshape(nash, nageb, nish, order="C").transpose(2, 0, 1)
                kernels.mltr1(1, lists[14], x_r1, f_ti.T, sig_high.T, val1=(-sqri2, -1.0))
            elif kod == 22:
                x_r1 = d2.reshape(nash, nagtb, nish, order="C").transpose(2, 0, 1)
                kernels.mltr1(1, lists[15], x_r1, f_ti.T, sig_high.T, val1=(sqri2, -sqri2))
            elif kod == 23:
                x_ia = d1.reshape(nssh, nish, order="C").T
                kernels.mltsca(1, lists[14], lists[16], x_ia, f_ia, sig_high.T, val1=(sqri2, 1.0), val2=(sqri2, 1.0))
            elif kod == 24:
                half_sqr3 = 0.5 * sqr3
                x_ia = d1.reshape(nssh, nish, order="C").T
                kernels.mltsca(1, lists[15], lists[17], x_ia, f_ia, sig_high.T, val1=(half_sqr3, -half_sqr3), val2=(1.0, -1.0))
            else:
                raise NotImplementedError(f"KOD={kod} reverse coupling not implemented")
