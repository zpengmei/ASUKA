r"""DF-based two-electron OLag/DPT2C construction for CASPT2 gradients.

This module implements the DF (Density Fitting) counterpart of the
Molcas ``olagns.f`` two-electron contribution path. Instead of forming
full MO-basis ERIs, it computes the orbital Lagrangian (OLag) and
the PT2 cumulant density (DPT2C) directly from DF pair blocks.

Mathematical Background
-----------------------
The two-electron contribution to the orbital Lagrangian is:

.. math::

    \text{OLag}^{(2e)}_{pq} = \sum_{c=1}^{13} \sum_{\mu\nu}
        T^{(c)}_{\mu\nu} \cdot \frac{\partial}{\partial \kappa_{pq}}
        \langle \Phi^{(c)}_\mu | \hat{H}_0 + \hat{H}_1 | \Phi^{(c)}_\nu \rangle

The per-case amplitude matrix ``TC`` is back-transformed from SR basis to
MO basis, then contracted with DF pair blocks ``L_{pq}^P`` using
batched GEMMs. The key helper ``_post1_df`` computes:

.. math::

    B_1 = \text{AmpL1} \cdot L_{J}^{R,P}, \quad
    B_2 = \text{AmpL1}^T \cdot L_{I}^{R,P}

and accumulates into OLag via:

.. math::

    \text{OLag}_{pR} \mathrel{+}= L_{pI}^P \cdot B_1^P + L_{pJ}^P \cdot B_2^P

A ``tildeT`` pseudo-density tensor of shape ``(nocc, nrest, naux)`` is
simultaneously constructed for the DF two-electron gradient contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import numpy as np

from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks
from asuka.mrpt2.df_pair_block import DFPairBlock


def _as_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _kTUV(t: int, u: int, v: int, nAsh: int) -> int:
    return int(t + nAsh * (u - 1) + nAsh * nAsh * (v - 1))


def _kTU(t: int, u: int, nAsh: int) -> int:
    return int(t + nAsh * (u - 1))


def _kTgeU(t: int, u: int) -> int:
    return int(t * (t - 1) // 2 + u)


def _kTgtU(t: int, u: int) -> int:
    return int((t - 1) * (t - 2) // 2 + u)


def _compute_tc(smat: np.ndarray | None, trans: np.ndarray, t_sr: np.ndarray) -> np.ndarray:
    if smat is not None:
        st = smat @ trans
        return st @ t_sr
    ttt = trans.T @ trans
    return trans @ np.linalg.inv(ttt) @ t_sr


def _empty_df_block(nx: int, ny: int, naux: int) -> DFPairBlock:
    return DFPairBlock(
        nx=int(nx),
        ny=int(ny),
        l_full=np.zeros((int(nx) * int(ny), int(naux)), dtype=np.float64),
        pair_norm=None,
    )


@dataclass(frozen=True)
class _DFViews:
    nish: int
    nash: int
    nssh: int
    naux: int
    l_ii: np.ndarray  # (nish,nish,naux)
    l_it: np.ndarray  # (nish,nash,naux)
    l_ia: np.ndarray  # (nish,nssh,naux)
    l_tu: np.ndarray  # (nash,nash,naux)
    l_at: np.ndarray  # (nssh,nash,naux)
    l_ab: np.ndarray  # (nssh,nssh,naux)

    @property
    def nocc(self) -> int:
        return int(self.nish + self.nash)

    @property
    def nmo(self) -> int:
        return int(self.nish + self.nash + self.nssh)

    @property
    def nrest(self) -> int:
        return int(self.nash + self.nssh)

    def all_for_occ(self, occ_abs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occ = int(occ_abs)
        if occ < 0 or occ >= int(self.nocc):
            raise IndexError("occupied orbital out of range")
        if occ < int(self.nish):
            i = occ
            return (
                self.l_ii[:, i, :],
                self.l_it[i, :, :],
                self.l_ia[i, :, :],
            )
        t = occ - int(self.nish)
        return (
            self.l_it[:, t, :],
            self.l_tu[:, t, :],
            self.l_at[:, t, :],
        )

    def r_for_occ(
        self,
        occ_abs: int,
        *,
        has_act: bool,
        has_vir: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        occ = int(occ_abs)
        if occ < 0 or occ >= int(self.nocc):
            raise IndexError("occupied orbital out of range")
        act = vir = None
        if occ < int(self.nish):
            i = occ
            if bool(has_act) and int(self.nash) > 0:
                act = self.l_it[i, :, :]
            if bool(has_vir) and int(self.nssh) > 0:
                vir = self.l_ia[i, :, :]
            return act, vir
        t = occ - int(self.nish)
        if bool(has_act) and int(self.nash) > 0:
            act = self.l_tu[:, t, :]
        if bool(has_vir) and int(self.nssh) > 0:
            vir = self.l_at[:, t, :]
        return act, vir


def _views_from_df_blocks(df: CASPT2DFBlocks, *, nish: int, nash: int, nssh: int) -> _DFViews:
    nish = int(nish)
    nash = int(nash)
    nssh = int(nssh)
    naux = int(df.l_it.naux) if df is not None else 0

    def _blk_or_empty(b: DFPairBlock | None, nx: int, ny: int) -> DFPairBlock:
        if b is None:
            return _empty_df_block(nx, ny, naux)
        if int(b.nx) != int(nx) or int(b.ny) != int(ny):
            raise ValueError(f"DFPairBlock shape mismatch: expected ({nx},{ny}), got ({b.nx},{b.ny})")
        if int(b.naux) != int(naux):
            raise ValueError("DFPairBlock naux mismatch")
        return b

    l_it = _as_f64(df.l_it.l_full).reshape(nish, nash, naux) if nish and nash else np.zeros((nish, nash, naux), dtype=np.float64)
    l_ia = _as_f64(df.l_ia.l_full).reshape(nish, nssh, naux) if nish and nssh else np.zeros((nish, nssh, naux), dtype=np.float64)
    l_at = _as_f64(df.l_at.l_full).reshape(nssh, nash, naux) if nssh and nash else np.zeros((nssh, nash, naux), dtype=np.float64)
    l_tu = _as_f64(df.l_tu.l_full).reshape(nash, nash, naux) if nash else np.zeros((nash, nash, naux), dtype=np.float64)

    b_ii = _blk_or_empty(df.l_ii, nish, nish)
    b_ab = _blk_or_empty(df.l_ab, nssh, nssh)
    l_ii = _as_f64(b_ii.l_full).reshape(nish, nish, naux) if nish else np.zeros((nish, nish, naux), dtype=np.float64)
    l_ab = _as_f64(b_ab.l_full).reshape(nssh, nssh, naux) if nssh else np.zeros((nssh, nssh, naux), dtype=np.float64)

    return _DFViews(
        nish=nish,
        nash=nash,
        nssh=nssh,
        naux=naux,
        l_ii=l_ii,
        l_it=l_it,
        l_ia=l_ia,
        l_tu=l_tu,
        l_at=l_at,
        l_ab=l_ab,
    )


def _post1_df(
    olag: np.ndarray,
    *,
    views: _DFViews,
    AmpL1: np.ndarray,
    nSkp: int,
    nDim: int,
    iI_occ: int,
    iJ_occ: int,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nocc = int(views.nocc)
    nmo = int(views.nmo)
    nrest = int(views.nrest)

    if int(nDim) <= 0:
        return
    if int(iI_occ) < 0 or int(iI_occ) >= nocc:
        raise IndexError("iI_occ out of range")
    if int(iJ_occ) < 0 or int(iJ_occ) >= nocc:
        raise IndexError("iJ_occ out of range")
    if tildeT.shape != (nocc, nrest, int(views.naux)):
        raise ValueError(f"tildeT shape mismatch: {tildeT.shape} vs {(nocc, nrest, int(views.naux))}")

    nSkp = int(nSkp)
    nDim = int(nDim)
    R0 = nSkp
    R1 = nSkp + nDim
    if R0 < 0 or R1 > nmo:
        raise ValueError("R slice out of range")

    if nSkp == nish and nDim == nash:
        has_act = True
        has_vir = False
        tilde_r0 = 0
        col_act = slice(nish, nish + nash)
        col_vir = None
    elif nSkp == nish and nDim == nash + nssh:
        has_act = True
        has_vir = bool(nssh > 0)
        tilde_r0 = 0
        col_act = slice(nish, nish + nash)
        col_vir = slice(nish + nash, nish + nash + nssh) if nssh > 0 else None
    elif nSkp == nocc and nDim == nssh:
        has_act = False
        has_vir = True
        tilde_r0 = nash
        col_act = None
        col_vir = slice(nocc, nocc + nssh)
    else:
        raise ValueError(f"Unsupported (nSkp,nDim)=({nSkp},{nDim}) for DF post1")

    tilde_r1 = tilde_r0 + nDim
    if tilde_r0 < 0 or tilde_r1 > nrest:
        raise ValueError("tildeT R slice out of range")

    x_core, x_act, x_vir = views.all_for_occ(int(iI_occ))
    y_core, y_act, y_vir = views.all_for_occ(int(iJ_occ))
    x_r_act, x_r_vir = views.r_for_occ(int(iI_occ), has_act=bool(has_act), has_vir=bool(has_vir))
    y_r_act, y_r_vir = views.r_for_occ(int(iJ_occ), has_act=bool(has_act), has_vir=bool(has_vir))

    AmpL1 = np.asarray(AmpL1, dtype=np.float64, order="C")
    if AmpL1.shape != (nDim, nDim):
        raise ValueError(f"AmpL1 shape mismatch: {AmpL1.shape} vs {(nDim, nDim)}")

    # B1 = AmpL1 @ Y_R  (nDim,naux)
    b1_act = b1_vir = None
    if col_act is not None and has_act and y_r_act is not None:
        if not has_vir:
            b1_act = AmpL1 @ y_r_act
        else:
            a_tt = AmpL1[:nash, :nash]
            a_ta = AmpL1[:nash, nash:]
            b1_act = a_tt @ y_r_act
            if y_r_vir is not None and a_ta.size:
                b1_act = b1_act + a_ta @ y_r_vir
    if col_vir is not None and has_vir and y_r_vir is not None:
        if not has_act:
            b1_vir = AmpL1 @ y_r_vir
        else:
            a_at = AmpL1[nash:, :nash]
            a_aa = AmpL1[nash:, nash:]
            b1_vir = a_aa @ y_r_vir
            if y_r_act is not None and a_at.size:
                b1_vir = b1_vir + a_at @ y_r_act

    # B2 = AmpL1^T @ X_R (nDim,naux)
    b2_act = b2_vir = None
    AmpL1_T = AmpL1.T
    if col_act is not None and has_act and x_r_act is not None:
        if not has_vir:
            b2_act = AmpL1_T @ x_r_act
        else:
            a_tt = AmpL1_T[:nash, :nash]
            a_ta = AmpL1_T[:nash, nash:]
            b2_act = a_tt @ x_r_act
            if x_r_vir is not None and a_ta.size:
                b2_act = b2_act + a_ta @ x_r_vir
    if col_vir is not None and has_vir and x_r_vir is not None:
        if not has_act:
            b2_vir = AmpL1_T @ x_r_vir
        else:
            a_at = AmpL1_T[nash:, :nash]
            a_aa = AmpL1_T[nash:, nash:]
            b2_vir = a_aa @ x_r_vir
            if x_r_act is not None and a_at.size:
                b2_vir = b2_vir + a_at @ x_r_act

    # Update tildeT (VVVO pseudo-density) for the two occupied columns.
    if b1_act is not None:
        tildeT[int(iI_occ), tilde_r0 : tilde_r0 + int(b1_act.shape[0]), :] += b1_act
    if b1_vir is not None:
        r0 = tilde_r0 + (nash if has_act else 0)
        tildeT[int(iI_occ), r0 : r0 + int(b1_vir.shape[0]), :] += b1_vir
    if b2_act is not None:
        tildeT[int(iJ_occ), tilde_r0 : tilde_r0 + int(b2_act.shape[0]), :] += b2_act
    if b2_vir is not None:
        r0 = tilde_r0 + (nash if has_act else 0)
        tildeT[int(iJ_occ), r0 : r0 + int(b2_vir.shape[0]), :] += b2_vir

    # OLag accumulation: olag[:,R] += X_all @ B1^T + Y_all @ B2^T
    row_core = slice(0, nish)
    row_act = slice(nish, nish + nash)
    row_vir = slice(nish + nash, nmo)

    def _add_block(rows: slice, cols: slice, left: np.ndarray, right: np.ndarray) -> None:
        if cols is None:
            return
        if int(left.shape[0]) == 0 or int(right.shape[0]) == 0:
            return
        olag[rows, cols] += left @ right.T

    if col_act is not None and b1_act is not None:
        _add_block(row_core, col_act, x_core, b1_act)
        _add_block(row_act, col_act, x_act, b1_act)
        _add_block(row_vir, col_act, x_vir, b1_act)
    if col_vir is not None and b1_vir is not None:
        _add_block(row_core, col_vir, x_core, b1_vir)
        _add_block(row_act, col_vir, x_act, b1_vir)
        _add_block(row_vir, col_vir, x_vir, b1_vir)
    if col_act is not None and b2_act is not None:
        _add_block(row_core, col_act, y_core, b2_act)
        _add_block(row_act, col_act, y_act, b2_act)
        _add_block(row_vir, col_act, y_vir, b2_act)
    if col_vir is not None and b2_vir is not None:
        _add_block(row_core, col_vir, y_core, b2_vir)
        _add_block(row_act, col_vir, y_act, b2_vir)
        _add_block(row_vir, col_vir, y_vir, b2_vir)


def _olagns_case_A_df(
    TC: np.ndarray,
    *,
    views: _DFViews,
    olag: np.ndarray,
    dpt2c: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)

    nCor = nish

    for iI in range(1, nash + 1):
        iI_orb = iI + nish - 1
        for iJ in range(1, nish + 1):
            iJabs = iJ
            iJ_orb = iJ - 1

            AmpL1 = np.zeros((nash, nash), dtype=np.float64)
            for iA in range(1, nash + 1):
                for iB in range(1, nash + 1):
                    iTabs = iB
                    iUabs = iI
                    iVabs = iA

                    iAS = _kTUV(iTabs, iUabs, iVabs, nash)
                    iIS = iJabs
                    ValA = float(TC[iAS - 1, iIS - 1]) * 2.0

                    if iUabs == iVabs:
                        iBtot = iB + nCor
                        iJtot = iJ
                        dpt2c[iBtot - 1, iJtot - 1] += ValA

                    AmpL1[iA - 1, iB - 1] += ValA

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nCor,
                nDim=nash,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_B_df(
    TCP: np.ndarray,
    TCM: np.ndarray | None,
    *,
    views: _DFViews,
    olag: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nCor = nish
    SQ2 = np.sqrt(2.0)
    SQI2 = 1.0 / SQ2
    has_minus = TCM is not None and int(getattr(TCM, "size", 0)) > 0

    for iI in range(1, nish + 1):
        for iJ in range(1, iI + 1):
            iIabs, iJabs = iI, iJ
            iI_orb, iJ_orb = iI - 1, iJ - 1

            AmpL1 = np.zeros((nash, nash), dtype=np.float64)
            for iA in range(1, nash + 1):
                for iB in range(1, nash + 1):
                    if iA > iB:
                        iTabs, iUabs = iA, iB
                    else:
                        iTabs, iUabs = iB, iA

                    iViP = _kTgeU(iIabs, iJabs)
                    iVaP = _kTgeU(iTabs, iUabs)

                    ValBP = float(TCP[iVaP - 1, iViP - 1])
                    ValBM = 0.0

                    if iA != iB and iIabs != iJabs:
                        if has_minus and TCM is not None:
                            iViM = _kTgtU(iIabs, iJabs)
                            iVaM = _kTgtU(iTabs, iUabs)
                            ValBM = float(TCM[iVaM - 1, iViM - 1])
                        if iA < iB:
                            ValBM = -ValBM

                    if iIabs == iJabs:
                        ValBP *= SQI2

                    AmpL1[iA - 1, iB - 1] += ValBP + ValBM

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nCor,
                nDim=nash,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_C_df(
    TC: np.ndarray,
    *,
    views: _DFViews,
    olag: np.ndarray,
    dpt2c: np.ndarray,
    nactel: int,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nCor = nish
    nOcc = nish + nash
    nDim = nash + nssh

    for iI in range(1, nash + 1):
        iI_orb = iI + nish - 1
        for iJ in range(1, iI + 1):
            iIabs, iJabs = iI, iJ
            iJ_orb = iJ + nish - 1

            AmpL1 = np.zeros((nDim, nDim), dtype=np.float64)
            for iA in range(1, nssh + 1):
                iAtot = iA + nOcc
                for iB in range(1, nash + 1):
                    iBtot = iB + nCor

                    iTabs = iI
                    iUabs = iB
                    iVabs = iJ

                    iAS = _kTUV(iTabs, iUabs, iVabs, nash)
                    iIS = iA

                    ValC1 = float(TC[iAS - 1, iIS - 1]) * 2.0
                    ValC2 = 0.0
                    if iIabs != iJabs:
                        iAS2 = _kTUV(iVabs, iUabs, iTabs, nash)
                        ValC2 = float(TC[iAS2 - 1, iIS - 1]) * 2.0

                    if iIabs == iJabs:
                        iAS3 = _kTUV(iB, iI, iJ, nash)
                        ONEADD = float(TC[iAS3 - 1, iIS - 1]) * 2.0
                        dpt2c[iAtot - 1, iBtot - 1] += ONEADD

                        ONEADD2 = 0.0
                        for iX in range(1, nash + 1):
                            iAS4 = _kTUV(iB, iX, iX, nash)
                            ONEADD2 += float(TC[iAS4 - 1, iIS - 1])
                        ONEADD2 = 2.0 * ONEADD2 / max(1, int(nactel))
                        AmpL1[iA + nash - 1, iB - 1] -= ONEADD2

                    AmpL1[iA + nash - 1, iB - 1] += ValC1
                    AmpL1[iB - 1, iA + nash - 1] += ValC2

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nCor,
                nDim=nDim,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_D_df(
    TC: np.ndarray,
    *,
    views: _DFViews,
    olag: np.ndarray,
    dpt2c: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nCor = nish
    nOcc = nish + nash
    nDim = nash + nssh
    nTU = nash * nash

    for iI in range(1, nash + 1):
        iI_orb = iI + nish - 1
        iItot = iI + nCor
        for iJ in range(1, nish + 1):
            iJabs = iJ
            iJ_orb = iJ - 1
            iJtot = iJ

            AmpL1 = np.zeros((nDim, nDim), dtype=np.float64)
            for iA in range(1, nssh + 1):
                iAtot = iA + nOcc
                for iB in range(1, nash + 1):
                    iBtot = iB + nCor

                    iIS = iJabs + nish * (iA - 1)
                    iAS1 = _kTU(iB, iI, nash)
                    ValD1 = float(TC[iAS1 - 1, iIS - 1]) * 2.0

                    iAS2 = iAS1 + nTU
                    ValD2 = float(TC[iAS2 - 1, iIS - 1]) * 2.0

                    if iItot == iBtot:
                        dpt2c[iAtot - 1, iJtot - 1] += ValD1

                    AmpL1[iA + nash - 1, iB - 1] += ValD2
                    AmpL1[iB - 1, iA + nash - 1] += ValD1

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nCor,
                nDim=nDim,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_E_df(
    TCP: np.ndarray,
    TCM: np.ndarray | None,
    *,
    views: _DFViews,
    olag: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nCor = nish
    nDim = nash + nssh
    SQ2 = np.sqrt(2.0)
    SQ3 = np.sqrt(3.0)
    has_minus = TCM is not None and int(getattr(TCM, "size", 0)) > 0

    for iI in range(1, nish + 1):
        for iJ in range(1, iI + 1):
            iIabs, iJabs = iI, iJ
            iI_orb, iJ_orb = iI - 1, iJ - 1

            AmpL1 = np.zeros((nDim, nDim), dtype=np.float64)
            IgeJ = _kTgeU(iIabs, iJabs)
            IgtJ = _kTgtU(iIabs, iJabs) if has_minus and iIabs > iJabs else None

            for iA in range(1, nssh + 1):
                for iB in range(1, nash + 1):
                    iISP = iA + nssh * (IgeJ - 1)
                    iASP = iB
                    ValEP = float(TCP[iASP - 1, iISP - 1])
                    ValEM = 0.0

                    if iIabs > iJabs:
                        ValEP *= SQ2
                        if has_minus and TCM is not None and IgtJ is not None:
                            iISM = iA + nssh * (IgtJ - 1)
                            iASM = iB
                            ValEM = float(TCM[iASM - 1, iISM - 1]) * SQ2 * SQ3

                    AmpL1[iA + nash - 1, iB - 1] = ValEP + ValEM
                    AmpL1[iB - 1, iA + nash - 1] = ValEP - ValEM

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nCor,
                nDim=nDim,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_F_df(
    TCP: np.ndarray,
    TCM: np.ndarray | None,
    *,
    views: _DFViews,
    olag: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nOcc = nish + nash
    SQI2 = 1.0 / np.sqrt(2.0)
    has_minus = TCM is not None and int(getattr(TCM, "size", 0)) > 0

    for iI in range(1, nash + 1):
        iI_orb = iI + nish - 1
        for iJ in range(1, iI + 1):
            iIabs, iJabs = iI, iJ
            iJ_orb = iJ + nish - 1

            AmpL1 = np.zeros((nssh, nssh), dtype=np.float64)
            for iA in range(1, nssh + 1):
                for iB in range(1, iA + 1):
                    iASP = _kTgeU(iIabs, iJabs)
                    iISP = _kTgeU(iA, iB)

                    ValFP = float(TCP[iASP - 1, iISP - 1])
                    ValFM = 0.0

                    if iIabs == iJabs:
                        ValFP *= 0.5

                    if iA != iB and iIabs != iJabs:
                        if has_minus and TCM is not None:
                            iASM = _kTgtU(iIabs, iJabs)
                            iISM = _kTgtU(iA, iB)
                            ValFM = -float(TCM[iASM - 1, iISM - 1])
                    elif iA == iB:
                        ValFP *= SQI2

                    AmpL1[iA - 1, iB - 1] += ValFP + ValFM
                    AmpL1[iB - 1, iA - 1] += ValFP - ValFM

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nOcc,
                nDim=nssh,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_G_df(
    TCP: np.ndarray,
    TCM: np.ndarray | None,
    *,
    views: _DFViews,
    olag: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nOcc = nish + nash
    SQ2 = np.sqrt(2.0)
    SQ3 = np.sqrt(3.0)
    has_minus = TCM is not None and int(getattr(TCM, "size", 0)) > 0

    for iI in range(1, nash + 1):
        iI_orb = iI + nish - 1
        for iJ in range(1, nish + 1):
            iJ_orb = iJ - 1

            AmpL1 = np.zeros((nssh, nssh), dtype=np.float64)
            for iA in range(1, nssh + 1):
                for iB in range(1, iA + 1):
                    kAB = _kTgeU(iA, iB)
                    iISP = iJ + nish * (kAB - 1)
                    iASP = iI

                    ValGP = float(TCP[iASP - 1, iISP - 1])
                    ValGM = 0.0

                    if iA != iB:
                        ValGP *= SQ2
                        if has_minus and TCM is not None:
                            kAB_m = _kTgtU(iA, iB)
                            iISM = iJ + nish * (kAB_m - 1)
                            iASM = iI
                            ValGM = float(TCM[iASM - 1, iISM - 1]) * SQ2 * SQ3

                    AmpL1[iA - 1, iB - 1] += ValGP + ValGM
                    AmpL1[iB - 1, iA - 1] += ValGP - ValGM

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nOcc,
                nDim=nssh,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def _olagns_case_H_df(
    TCP: np.ndarray,
    TCM: np.ndarray | None,
    *,
    views: _DFViews,
    olag: np.ndarray,
    tildeT: np.ndarray,
) -> None:
    nish = int(views.nish)
    nash = int(views.nash)
    nssh = int(views.nssh)
    nOcc = nish + nash
    SQ2 = np.sqrt(2.0)
    SQ3 = np.sqrt(3.0)
    nAgeB = nssh * (nssh + 1) // 2
    nAgtB = nssh * (nssh - 1) // 2
    has_minus = TCM is not None and int(getattr(TCM, "size", 0)) > 0

    tcp_f = np.asarray(TCP, dtype=np.float64).ravel(order="F")
    tcm_f = np.asarray(TCM, dtype=np.float64).ravel(order="F") if has_minus and TCM is not None else None

    for iI in range(1, nish + 1):
        for iJ in range(1, iI + 1):
            iIabs, iJabs = iI, iJ
            iI_orb, iJ_orb = iI - 1, iJ - 1

            AmpL1 = np.zeros((nssh, nssh), dtype=np.float64)
            kViHP0 = _kTgeU(iIabs, iJabs)
            kViHP_offset = nAgeB * (kViHP0 - 1)

            kViHM_offset = None
            if has_minus and iIabs > iJabs:
                kViHM0 = _kTgtU(iIabs, iJabs)
                kViHM_offset = nAgtB * (kViHM0 - 1)

            for iA in range(1, nssh + 1):
                for iB in range(1, iA + 1):
                    kVaHP = _kTgeU(iA, iB)
                    kVHP = kVaHP + kViHP_offset

                    ValHP = float(tcp_f[kVHP - 1])
                    ValHM = 0.0

                    if iIabs != iJabs:
                        if iA != iB:
                            ValHP *= 2.0
                            if has_minus and tcm_f is not None and kViHM_offset is not None:
                                kVaHM = _kTgtU(iA, iB)
                                kVHM = kVaHM + kViHM_offset
                                ValHM = float(tcm_f[kVHM - 1]) * 2.0 * SQ3
                        else:
                            ValHP *= SQ2
                    else:
                        if iA != iB:
                            ValHP *= SQ2

                    AmpL1[iA - 1, iB - 1] += ValHP + ValHM
                    AmpL1[iB - 1, iA - 1] += ValHP - ValHM

            _post1_df(
                olag,
                views=views,
                AmpL1=AmpL1,
                nSkp=nOcc,
                nDim=nssh,
                iI_occ=iI_orb,
                iJ_occ=iJ_orb,
                tildeT=tildeT,
            )


def build_olagns2_df(
    case_amps: list[Any],
    orb: Any,
    fock: Any,
    rdms: Any,
    df_blocks: CASPT2DFBlocks,
    *,
    nactel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nish = int(getattr(orb, "nish"))
    nash = int(getattr(orb, "nash"))
    nssh = int(getattr(orb, "nssh"))
    nmo = int(nish + nash + nssh)
    nocc = int(nish + nash)
    nrest = int(nash + nssh)

    views = _views_from_df_blocks(df_blocks, nish=nish, nash=nash, nssh=nssh)
    if int(views.naux) <= 0:
        raise ValueError("DF blocks appear to have naux=0")
    if int(nish) > 0 and df_blocks.l_ii is None:
        raise ValueError("DF OLagNS2 requires df_blocks.l_ii to be populated when nish>0")
    if int(nssh) > 0 and df_blocks.l_ab is None:
        raise ValueError("DF VVVO requires df_blocks.l_ab to be populated when nssh>0")

    olag = np.zeros((nmo, nmo), dtype=np.float64)
    dpt2c = np.zeros((nmo, nmo), dtype=np.float64)
    tildeT = np.zeros((nocc, nrest, int(views.naux)), dtype=np.float64)

    def _get_tc(case: int) -> np.ndarray | None:
        idx = int(case) - 1
        if idx < 0 or idx >= 13:
            return None
        ca = case_amps[idx]
        if ca is None:
            return None
        nAS = int(getattr(ca, "nAS", 0))
        nIS = int(getattr(ca, "nIS", 0))
        if nAS <= 0 or nIS <= 0:
            return None
        return _compute_tc(
            getattr(ca, "smat", None),
            _as_f64(getattr(ca, "trans")),
            _as_f64(getattr(ca, "T")).reshape(int(getattr(ca, "nIN")), nIS),
        )

    def _get_tc_pair(case_plus: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return _get_tc(case_plus), _get_tc(case_plus + 1)

    tc = _get_tc(1)
    if tc is not None:
        _olagns_case_A_df(tc, views=views, olag=olag, dpt2c=dpt2c, tildeT=tildeT)

    tcp, tcm = _get_tc_pair(2)
    if tcp is not None:
        _olagns_case_B_df(tcp, tcm, views=views, olag=olag, tildeT=tildeT)

    tc = _get_tc(4)
    if tc is not None:
        _olagns_case_C_df(tc, views=views, olag=olag, dpt2c=dpt2c, nactel=int(nactel), tildeT=tildeT)

    tc = _get_tc(5)
    if tc is not None:
        _olagns_case_D_df(tc, views=views, olag=olag, dpt2c=dpt2c, tildeT=tildeT)

    tcp, tcm = _get_tc_pair(6)
    if tcp is not None:
        _olagns_case_E_df(tcp, tcm, views=views, olag=olag, tildeT=tildeT)

    tcp, tcm = _get_tc_pair(8)
    if tcp is not None:
        _olagns_case_F_df(tcp, tcm, views=views, olag=olag, tildeT=tildeT)

    tcp, tcm = _get_tc_pair(10)
    if tcp is not None:
        _olagns_case_G_df(tcp, tcm, views=views, olag=olag, tildeT=tildeT)

    tcp, tcm = _get_tc_pair(12)
    if tcp is not None:
        _olagns_case_H_df(tcp, tcm, views=views, olag=olag, tildeT=tildeT)

    scale_mode = str(os.environ.get("ASUKA_OLAGNS2_DPT2C_SCALE_MODE", "nactel")).strip().lower()
    if scale_mode in {"none", "unity", "unscaled", "molcas_current"}:
        scale = 1.0
    else:
        scale_mode = "nactel"
        scale = 1.0 / max(1, int(nactel))
    dpt2c *= float(scale)

    build_olagns2_df._last_details = {  # type: ignore[attr-defined]
        "olag_total": np.asarray(olag, dtype=np.float64),
        "dpt2c_total": np.asarray(dpt2c, dtype=np.float64),
        "nactel": int(nactel),
        "dpt2c_scale": float(scale),
        "dpt2c_scale_mode": str(scale_mode),
    }
    return olag, dpt2c, tildeT


def olagvvvo_df(
    olag: np.ndarray,
    dpt2: np.ndarray,
    dpt2c: np.ndarray,
    *,
    orb: Any,
    df_blocks: CASPT2DFBlocks,
    tildeT: np.ndarray,
    cmo: np.ndarray,
    s_ao: np.ndarray | None,
    B_ao: np.ndarray,
    aux_block_naux: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from tools.molcas_caspt2_grad.translation.density import dpt2_trf  # noqa: PLC0415
    from asuka.mrpt2.semicanonical import build_vhf_df  # noqa: PLC0415

    nish = int(getattr(orb, "nish"))
    nash = int(getattr(orb, "nash"))
    nssh = int(getattr(orb, "nssh"))
    nocc = int(nish + nash)
    nmo = int(nish + nash + nssh)
    nrest = int(nash + nssh)

    dpt2_sym = 0.5 * (_as_f64(dpt2) + _as_f64(dpt2).T)
    dpt2c_sym = 0.5 * (_as_f64(dpt2c) + _as_f64(dpt2c).T)

    dpt2_ao = dpt2_trf(dpt2_sym, _as_f64(cmo))
    dpt2c_ao = dpt2_trf(dpt2c_sym, _as_f64(cmo))

    fpt2_ao = build_vhf_df(dpt2_ao, b_ao=_as_f64(B_ao))
    fpt2c_ao = build_vhf_df(dpt2c_ao, b_ao=_as_f64(B_ao))

    if tildeT is None or int(getattr(tildeT, "size", 0)) == 0:
        return fpt2_ao, fpt2c_ao, olag
    if tildeT.shape != (nocc, nrest, int(df_blocks.l_it.naux)):
        raise ValueError(
            f"tildeT shape mismatch: {tildeT.shape} vs {(nocc, nrest, int(df_blocks.l_it.naux))}"
        )

    if int(nssh) > 0 and df_blocks.l_ab is None:
        raise ValueError("DF VVVO requires df_blocks.l_ab to be populated when nssh>0")

    l_it = _as_f64(df_blocks.l_it.l_full).reshape(nish, nash, -1) if nish and nash else np.zeros((nish, nash, int(df_blocks.l_it.naux)), dtype=np.float64)
    l_ia = _as_f64(df_blocks.l_ia.l_full).reshape(nish, nssh, -1) if nish and nssh else np.zeros((nish, nssh, int(df_blocks.l_it.naux)), dtype=np.float64)
    l_tu = _as_f64(df_blocks.l_tu.l_full).reshape(nash, nash, -1) if nash else np.zeros((nash, nash, int(df_blocks.l_it.naux)), dtype=np.float64)
    l_at = _as_f64(df_blocks.l_at.l_full).reshape(nssh, nash, -1) if nssh and nash else np.zeros((nssh, nash, int(df_blocks.l_it.naux)), dtype=np.float64)
    l_ab = (
        _as_f64(df_blocks.l_ab.l_full).reshape(nssh, nssh, -1)
        if nssh and df_blocks.l_ab is not None
        else np.zeros((nssh, nssh, int(df_blocks.l_it.naux)), dtype=np.float64)
    )

    t_act = np.asarray(tildeT[:, :nash, :], dtype=np.float64, order="C") if nash else np.zeros((nocc, 0, int(df_blocks.l_it.naux)), dtype=np.float64)
    t_vir = np.asarray(tildeT[:, nash:, :], dtype=np.float64, order="C") if nssh else np.zeros((nocc, 0, int(df_blocks.l_it.naux)), dtype=np.float64)

    hole_core = np.zeros((nish, nocc), dtype=np.float64)
    hole_act = np.zeros((nash, nocc), dtype=np.float64)
    hole_vir = np.zeros((nssh, nocc), dtype=np.float64)

    naux = int(df_blocks.l_it.naux)
    nb = int(max(1, int(aux_block_naux)))
    for q0 in range(0, naux, nb):
        q1 = min(naux, q0 + nb)
        qb = int(q1 - q0)

        if nash:
            t_act_blk = t_act[:, :, q0:q1].reshape(nocc, nash * qb)
            if nish:
                l_it_blk = l_it[:, :, q0:q1].reshape(nish, nash * qb)
                hole_core += l_it_blk @ t_act_blk.T
            l_tu_blk = l_tu[:, :, q0:q1].reshape(nash, nash * qb)
            hole_act += l_tu_blk @ t_act_blk.T
            if nssh:
                l_at_blk = l_at[:, :, q0:q1].reshape(nssh, nash * qb)
                hole_vir += l_at_blk @ t_act_blk.T

        if nssh:
            t_vir_blk = t_vir[:, :, q0:q1].reshape(nocc, nssh * qb)
            if nish:
                l_ia_blk = l_ia[:, :, q0:q1].reshape(nish, nssh * qb)
                hole_core += l_ia_blk @ t_vir_blk.T
            if nash:
                l_ta_blk = l_at[:, :, q0:q1].transpose(1, 0, 2).reshape(nash, nssh * qb)
                hole_act += l_ta_blk @ t_vir_blk.T
            l_ab_blk = l_ab[:, :, q0:q1].reshape(nssh, nssh * qb)
            hole_vir += l_ab_blk @ t_vir_blk.T

    hole = np.vstack([hole_core, hole_act, hole_vir]) if nmo else np.zeros((0, 0), dtype=np.float64)
    if hole.shape != (nmo, nocc):
        raise RuntimeError(f"hole-lag shape mismatch: {hole.shape} vs {(nmo, nocc)}")
    olag[:, :nocc] += hole

    return fpt2_ao, fpt2c_ao, olag

