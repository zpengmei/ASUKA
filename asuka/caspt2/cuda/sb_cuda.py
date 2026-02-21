from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.superindex import SuperindexMap


@dataclass(frozen=True)
class SBDecompositionDevice:
    """Device-side analogue of `asuka.caspt2.overlap.SBDecomposition`."""

    s_eigvals: Any
    transform: Any
    nindep: int
    b_diag: Any


def _as_i64(cp, a) -> Any:
    return cp.asarray(a, dtype=cp.int64)


def precompute_fock_quantities_cuda(
    fock: CASPT2Fock,
    dm1_d,
    dm2_d,
    dm3_d,
    *,
    cp,
) -> tuple[Any, Any, Any, Any]:
    """Return (epsa, easum, fd, fp) on device."""
    epsa = cp.ascontiguousarray(cp.asarray(np.asarray(fock.epsa, dtype=np.float64), dtype=cp.float64).ravel())
    easum = cp.sum(epsa * cp.diagonal(dm1_d).astype(cp.float64, copy=False))
    fd = cp.einsum("w,txww->tx", epsa, dm2_d, optimize=True)
    if dm3_d is not None and int(getattr(dm3_d, "size", 0)) > 0:
        fp = 0.5 * cp.einsum("w,pqrsww->pqrs", epsa, dm3_d, optimize=True)
    else:
        nash = int(epsa.shape[0])
        fp = cp.zeros((nash, nash, nash, nash), dtype=cp.float64)
    return epsa, easum, cp.ascontiguousarray(fd), cp.ascontiguousarray(fp)


def build_smat_a_cuda(smap: SuperindexMap, dm1_d, dm2_d, dm3_d, *, cp) -> Any:
    """Case 1 (A) overlap matrix on GPU: SA(tuv,xyz)."""
    ntuv = int(smap.ntuv)
    if ntuv == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    mtuv = _as_i64(cp, np.asarray(smap.mtuv, dtype=np.int64))
    t = mtuv[:, 0]
    u = mtuv[:, 1]
    v = mtuv[:, 2]
    x = t
    y = u
    z = v

    t_row = t
    u_row = u
    v_row = v
    x_col = x
    y_col = y
    z_col = z

    # Base: -G3[v,u,x,t,y,z]
    smat = -dm3_d[
        v_row[:, None],
        u_row[:, None],
        x_col[None, :],
        t_row[:, None],
        y_col[None, :],
        z_col[None, :],
    ]

    mask_yu = (y_col[None, :] == u_row[:, None])
    mask_yt = (y_col[None, :] == t_row[:, None])
    mask_xu = (x_col[None, :] == u_row[:, None])
    mask_tx = (t_row[:, None] == x_col[None, :])

    # - d(y,u)*G2[v,z,x,t]
    smat = smat - mask_yu * dm2_d[v_row[:, None], z_col[None, :], x_col[None, :], t_row[:, None]]
    # - d(y,t)*G2[v,u,x,z]
    smat = smat - mask_yt * dm2_d[v_row[:, None], u_row[:, None], x_col[None, :], z_col[None, :]]
    # - d(x,u)*G2[v,t,y,z]
    smat = smat - mask_xu * dm2_d[v_row[:, None], t_row[:, None], y_col[None, :], z_col[None, :]]

    # - d(x,u)*d(y,t)*G1[v,z]
    smat = smat - (mask_xu & mask_yt) * dm1_d[v_row[:, None], z_col[None, :]]

    # + 2*d(t,x)*G2[v,u,y,z]
    smat = smat + (2.0 * mask_tx) * dm2_d[v_row[:, None], u_row[:, None], y_col[None, :], z_col[None, :]]

    # + 2*d(t,x)*d(y,u)*G1[v,z]
    smat = smat + (2.0 * (mask_tx & mask_yu)) * dm1_d[v_row[:, None], z_col[None, :]]

    smat = cp.ascontiguousarray(0.5 * (smat + smat.T))
    return smat


def build_smat_c_cuda(smap: SuperindexMap, dm1_d, dm2_d, dm3_d, *, cp) -> Any:
    """Case 4 (C) overlap matrix on GPU: SC(tuv,xyz)."""
    ntuv = int(smap.ntuv)
    if ntuv == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    mtuv = _as_i64(cp, np.asarray(smap.mtuv, dtype=np.int64))
    t = mtuv[:, 0]
    u = mtuv[:, 1]
    v = mtuv[:, 2]
    x = t
    y = u
    z = v

    t_row = t
    u_row = u
    v_row = v
    x_col = x
    y_col = y
    z_col = z

    smat = dm3_d[
        v_row[:, None],
        u_row[:, None],
        t_row[:, None],
        x_col[None, :],
        y_col[None, :],
        z_col[None, :],
    ]

    mask_yu = (y_col[None, :] == u_row[:, None])
    mask_yx = (y_col[None, :] == x_col[None, :])
    mask_tu_vec = (t_row == u_row)
    mask_tu = mask_tu_vec[:, None]

    # + d(y,u)*G2[v,z,t,x]
    smat = smat + mask_yu * dm2_d[v_row[:, None], z_col[None, :], t_row[:, None], x_col[None, :]]
    # + d(y,x)*G2[v,u,t,z]
    smat = smat + mask_yx * dm2_d[v_row[:, None], u_row[:, None], t_row[:, None], z_col[None, :]]
    # + d(t,u)*G2[v,x,y,z]
    smat = smat + mask_tu * dm2_d[v_row[:, None], x_col[None, :], y_col[None, :], z_col[None, :]]
    # + d(t,u)*d(y,x)*G1[v,z]
    smat = smat + (mask_tu & mask_yx) * dm1_d[v_row[:, None], z_col[None, :]]

    smat = cp.ascontiguousarray(0.5 * (smat + smat.T))
    return smat


def build_bmat_a_cuda(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    dm1_d,
    dm2_d,
    dm3_d,
    *,
    f3_engine,
    smat_d=None,
    easum_d=None,
    fd_d=None,
    fp_d=None,
    cp,
) -> Any:
    """Case 1 (A) B matrix on GPU (Dyall H0 active-space block)."""
    ntuv = int(smap.ntuv)
    nash = int(smap.orbs.nash)
    if ntuv == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    if nash <= 0:
        return cp.empty((0, 0), dtype=cp.float64)

    if smat_d is None:
        smat_d = build_smat_a_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)

    if easum_d is None or fd_d is None or fp_d is None:
        epsa_d, easum_d, fd_d, fp_d = precompute_fock_quantities_cuda(fock, dm1_d, dm2_d, dm3_d, cp=cp)
    else:
        epsa_d = cp.ascontiguousarray(cp.asarray(np.asarray(fock.epsa, dtype=np.float64), dtype=cp.float64).ravel())

    mtuv = _as_i64(cp, np.asarray(smap.mtuv, dtype=np.int64))
    t = mtuv[:, 0]
    u = mtuv[:, 1]
    v = mtuv[:, 2]
    x = t
    y = u
    z = v

    t_row = t
    u_row = u
    v_row = v
    x_col = x
    y_col = y
    z_col = z

    # Energy factor times S.
    factor = (
        epsa_d[u_row][:, None]
        + epsa_d[t_row][:, None]
        + epsa_d[x_col][None, :]
        + epsa_d[y_col][None, :]
        - easum_d
    )
    bmat = factor * smat_d

    # F3 contribution: BA += -F3_case_a = corr - raw (Molcas mkfg3 convention).
    vu_idx = (v_row * nash + u_row).astype(cp.int64, copy=False)
    t_row_i64 = t_row.astype(cp.int64, copy=False)
    u_row_i64 = u_row.astype(cp.int64, copy=False)
    v_row_i64 = v_row.astype(cp.int64, copy=False)
    x_vec = cp.arange(nash, dtype=cp.int64)

    # Precompute all F3raw yz matrices once (uses the mm-scaled EPQ transpose-range kernels if available).
    precompute = getattr(f3_engine, "precompute_f3raw_all", None)
    if precompute is not None:
        precompute()

    corr = cp.empty((ntuv, nash), dtype=cp.float64)
    for yfix in range(nash):
        for zfix in range(nash):
            cols = x_vec * (nash * nash) + int(yfix) * nash + int(zfix)
            mat_yz = f3_engine._matrix_for_yz_f3raw(int(yfix), int(zfix))

            xt_idx = x_vec[None, :] * nash + t_row_i64[:, None]
            raw_block = mat_yz[vu_idx[:, None], xt_idx]

            # Corrections from mkfg3.f (see F3ContractionEngine.f3_molcas).
            mask_y_eq_x = (t_row_i64 == int(yfix))  # y == x (x = t_row)
            mask_v_eq_u = (u_row_i64[:, None] == x_vec[None, :])  # v == u (v = x_col)
            mask_y_eq_u = (u_row_i64 == int(yfix))

            corr.fill(0.0)

            # if y == x: corr += 2*fp[t,u,v,z] + epsa[u]*dm2[t,u,v,z]
            term = 2.0 * fp_d[v_row_i64[:, None], u_row_i64[:, None], x_vec[None, :], int(zfix)]
            term = term + epsa_d[u_row_i64][:, None] * dm2_d[
                v_row_i64[:, None], u_row_i64[:, None], x_vec[None, :], int(zfix)
            ]
            corr += mask_y_eq_x[:, None] * term

            # nested if v == u: corr += fd[t,z]
            fd_tz = fd_d[v_row_i64, int(zfix)]
            corr += (mask_y_eq_x[:, None] & mask_v_eq_u) * fd_tz[:, None]

            # if v == u: corr += 2*fp[t,x,y,z] + epsa[y]*dm2[t,x,y,z]  (x = t_row)
            fp_tx = 2.0 * fp_d[v_row_i64, t_row_i64, int(yfix), int(zfix)]
            dm2_tx = dm2_d[v_row_i64, t_row_i64, int(yfix), int(zfix)]
            term = (fp_tx + epsa_d[int(yfix)] * dm2_tx)[:, None]
            corr += mask_v_eq_u * term

            # if y == u: corr += 2*fp[v,x,t,z] + epsa[u]*dm2[v,x,t,z]  (v = x_vec, x = t_row, t = v_row)
            term = 2.0 * fp_d[x_vec[None, :], t_row_i64[:, None], v_row_i64[:, None], int(zfix)]
            term = term + epsa_d[u_row_i64][:, None] * dm2_d[
                x_vec[None, :], t_row_i64[:, None], v_row_i64[:, None], int(zfix)
            ]
            corr += mask_y_eq_u[:, None] * term

            # Always: corr += (epsa[u] + epsa[y]) * dm3[t,u,v,x,y,z]
            corr += (epsa_d[u_row_i64][:, None] + epsa_d[int(yfix)]) * dm3_d[
                v_row_i64[:, None],
                u_row_i64[:, None],
                x_vec[None, :],
                t_row_i64[:, None],
                int(yfix),
                int(zfix),
            ]

            bmat[:, cols] = bmat[:, cols] + (corr - raw_block)

    # Delta corrections (MKBA_DP).
    mask_tx = (t_row[:, None] == x_col[None, :])
    mask_xu = (x_col[None, :] == u_row[:, None])
    mask_yt = (y_col[None, :] == t_row[:, None])
    mask_yu = (y_col[None, :] == u_row[:, None])

    # if t == x: + 4*(fp[v,u,y,z] - epsa[t]*0.5*dm2[v,u,y,z])
    term = 4.0 * (
        fp_d[v_row[:, None], u_row[:, None], y_col[None, :], z_col[None, :]]
        - epsa_d[t_row][:, None] * (0.5 * dm2_d[v_row[:, None], u_row[:, None], y_col[None, :], z_col[None, :]])
    )
    bmat = bmat + mask_tx * term

    # nested if y == u: + 2*(fd[v,z] - (epsa[t]+epsa[u])*dm1[v,z])
    fd_vz = fd_d[v_row[:, None], z_col[None, :]]
    dm1_vz = dm1_d[v_row[:, None], z_col[None, :]]
    term = 2.0 * (fd_vz - (epsa_d[t_row][:, None] + epsa_d[u_row][:, None]) * dm1_vz)
    bmat = bmat + (mask_tx & mask_yu) * term

    # if x == u: -2*(fp[v,t,y,z] - epsa[u]*0.5*dm2[v,t,y,z])
    term = -2.0 * (
        fp_d[v_row[:, None], t_row[:, None], y_col[None, :], z_col[None, :]]
        - epsa_d[u_row][:, None] * (0.5 * dm2_d[v_row[:, None], t_row[:, None], y_col[None, :], z_col[None, :]])
    )
    bmat = bmat + mask_xu * term

    # nested if y == t: -(fd[v,z] - (epsa[t]+epsa[u])*dm1[v,z])
    term = -(fd_vz - (epsa_d[t_row][:, None] + epsa_d[u_row][:, None]) * dm1_vz)
    bmat = bmat + (mask_xu & mask_yt) * term

    # if y == t: -2*(fp[v,u,x,z] - epsa[t]*0.5*dm2[v,u,x,z])
    term = -2.0 * (
        fp_d[v_row[:, None], u_row[:, None], x_col[None, :], z_col[None, :]]
        - epsa_d[t_row][:, None] * (0.5 * dm2_d[v_row[:, None], u_row[:, None], x_col[None, :], z_col[None, :]])
    )
    bmat = bmat + mask_yt * term

    # if y == u: -2*(fp[v,z,x,t] - epsa[u]*0.5*dm2[v,z,x,t])
    term = -2.0 * (
        fp_d[v_row[:, None], z_col[None, :], x_col[None, :], t_row[:, None]]
        - epsa_d[u_row][:, None] * (0.5 * dm2_d[v_row[:, None], z_col[None, :], x_col[None, :], t_row[:, None]])
    )
    bmat = bmat + mask_yu * term

    bmat = cp.ascontiguousarray(0.5 * (bmat + bmat.T))
    return bmat


def build_bmat_c_cuda(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    dm1_d,
    dm2_d,
    dm3_d,
    *,
    f3_engine,
    smat_d=None,
    easum_d=None,
    fd_d=None,
    fp_d=None,
    cp,
) -> Any:
    """Case 4 (C) B matrix on GPU (Dyall H0 active-space block)."""
    ntuv = int(smap.ntuv)
    nash = int(smap.orbs.nash)
    if ntuv == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    if nash <= 0:
        return cp.empty((0, 0), dtype=cp.float64)

    if smat_d is None:
        smat_d = build_smat_c_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)

    if easum_d is None or fd_d is None or fp_d is None:
        epsa_d, easum_d, fd_d, fp_d = precompute_fock_quantities_cuda(fock, dm1_d, dm2_d, dm3_d, cp=cp)
    else:
        epsa_d = cp.ascontiguousarray(cp.asarray(np.asarray(fock.epsa, dtype=np.float64), dtype=cp.float64).ravel())

    mtuv = _as_i64(cp, np.asarray(smap.mtuv, dtype=np.int64))
    t = mtuv[:, 0]
    u = mtuv[:, 1]
    v = mtuv[:, 2]
    x = t
    y = u
    z = v

    t_row = t
    u_row = u
    v_row = v
    x_col = x
    y_col = y
    z_col = z

    factor = epsa_d[u_row][:, None] + epsa_d[y_col][None, :] - easum_d
    bmat = factor * smat_d

    # F3 contribution: BC += +F3_case_c = raw - corr.
    vu_idx = (v_row * nash + u_row).astype(cp.int64, copy=False)
    t_row_i64 = t_row.astype(cp.int64, copy=False)
    u_row_i64 = u_row.astype(cp.int64, copy=False)
    v_row_i64 = v_row.astype(cp.int64, copy=False)
    x_vec = cp.arange(nash, dtype=cp.int64)

    mask_v_eq_u_row = (t_row_i64 == u_row_i64)
    precompute = getattr(f3_engine, "precompute_f3raw_all", None)
    if precompute is not None:
        precompute()

    corr = cp.empty((ntuv, nash), dtype=cp.float64)
    for yfix in range(nash):
        for zfix in range(nash):
            cols = x_vec * (nash * nash) + int(yfix) * nash + int(zfix)
            mat_yz = f3_engine._matrix_for_yz_f3raw(int(yfix), int(zfix))

            tx_idx = t_row_i64[:, None] * nash + x_vec[None, :]
            raw_block = mat_yz[vu_idx[:, None], tx_idx]

            mask_y_eq_x = (x_vec == int(yfix))  # y == x (x = x_vec)
            mask_y_eq_u = (u_row_i64 == int(yfix))

            corr.fill(0.0)

            # if y == x: corr += 2*fp[t,u,v,z] + epsa[u]*dm2[t,u,v,z]  (v=t_row)
            fp_tuvz = 2.0 * fp_d[v_row_i64, u_row_i64, t_row_i64, int(zfix)]
            dm2_tuvz = dm2_d[v_row_i64, u_row_i64, t_row_i64, int(zfix)]
            term = (fp_tuvz + epsa_d[u_row_i64] * dm2_tuvz)[:, None]
            corr += mask_y_eq_x[None, :] * term

            # nested if v == u: corr += fd[t,z]
            fd_tz = fd_d[v_row_i64, int(zfix)]
            corr += (mask_y_eq_x[None, :] & mask_v_eq_u_row[:, None]) * fd_tz[:, None]

            # if v == u: corr += 2*fp[t,x,y,z] + epsa[y]*dm2[t,x,y,z]
            fp_txyz = 2.0 * fp_d[v_row_i64[:, None], x_vec[None, :], int(yfix), int(zfix)]
            dm2_txyz = dm2_d[v_row_i64[:, None], x_vec[None, :], int(yfix), int(zfix)]
            corr += mask_v_eq_u_row[:, None] * (fp_txyz + epsa_d[int(yfix)] * dm2_txyz)

            # if y == u: corr += 2*fp[v,x,t,z] + epsa[u]*dm2[v,x,t,z]  (v=t_row, t=v_row)
            fp_vxtz = 2.0 * fp_d[t_row_i64[:, None], x_vec[None, :], v_row_i64[:, None], int(zfix)]
            dm2_vxtz = dm2_d[t_row_i64[:, None], x_vec[None, :], v_row_i64[:, None], int(zfix)]
            corr += mask_y_eq_u[:, None] * (fp_vxtz + epsa_d[u_row_i64][:, None] * dm2_vxtz)

            # Always: corr += (epsa[u] + epsa[y]) * dm3[t,u,v,x,y,z]
            corr += (epsa_d[u_row_i64][:, None] + epsa_d[int(yfix)]) * dm3_d[
                v_row_i64[:, None],
                u_row_i64[:, None],
                t_row_i64[:, None],
                x_vec[None, :],
                int(yfix),
                int(zfix),
            ]

            bmat[:, cols] = bmat[:, cols] + (raw_block - corr)

    # Delta corrections (MKBC_DP).
    mask_yu = (y_col[None, :] == u_row[:, None])
    mask_yx = (y_col[None, :] == x_col[None, :])
    mask_tu = (t_row == u_row)[:, None]

    term = 2.0 * (
        fp_d[v_row[:, None], z_col[None, :], t_row[:, None], x_col[None, :]]
        - epsa_d[u_row][:, None] * (0.5 * dm2_d[v_row[:, None], z_col[None, :], t_row[:, None], x_col[None, :]])
    )
    bmat = bmat + mask_yu * term

    term = 2.0 * (
        fp_d[v_row[:, None], u_row[:, None], t_row[:, None], z_col[None, :]]
        - epsa_d[y_col][None, :] * (0.5 * dm2_d[v_row[:, None], u_row[:, None], t_row[:, None], z_col[None, :]])
    )
    bmat = bmat + mask_yx * term

    term = 2.0 * (
        fp_d[v_row[:, None], x_col[None, :], y_col[None, :], z_col[None, :]]
        - epsa_d[u_row][:, None] * (0.5 * dm2_d[v_row[:, None], x_col[None, :], y_col[None, :], z_col[None, :]])
    )
    bmat = bmat + mask_tu * term

    fd_vz = fd_d[v_row[:, None], z_col[None, :]]
    dm1_vz = dm1_d[v_row[:, None], z_col[None, :]]
    term = fd_vz - (epsa_d[u_row][:, None] + epsa_d[y_col][None, :]) * dm1_vz
    bmat = bmat + (mask_tu & mask_yx) * term

    bmat = cp.ascontiguousarray(0.5 * (bmat + bmat.T))
    return bmat


def _raw_sb_matrix_cuda(t, u, x, y, dm1_d, dm2_d, *, cp):
    """Raw SB(tu,xy) over all ordered active pairs (ntu,ntu)."""
    t = cp.asarray(t, dtype=cp.int64)
    u = cp.asarray(u, dtype=cp.int64)
    x = cp.asarray(x, dtype=cp.int64)
    y = cp.asarray(y, dtype=cp.int64)

    # Base: 2*G2[x,t,y,u]
    val = 2.0 * dm2_d[x[None, :], t[:, None], y[None, :], u[:, None]]

    mask_xt = x[None, :] == t[:, None]
    mask_yu = y[None, :] == u[:, None]
    mask_yt = y[None, :] == t[:, None]
    mask_xu = x[None, :] == u[:, None]

    # -4*d(x,t)*G1[y,u]
    val = val - (4.0 * mask_xt) * dm1_d[y[None, :], u[:, None]]
    # +8*d(x,t)*d(y,u)
    val = val + 8.0 * (mask_xt & mask_yu)
    # -4*d(y,u)*G1[x,t]
    val = val - (4.0 * mask_yu) * dm1_d[x[None, :], t[:, None]]
    # +2*d(y,t)*G1[x,u]
    val = val + (2.0 * mask_yt) * dm1_d[x[None, :], u[:, None]]
    # +2*d(x,u)*G1[y,t] -4*d(x,u)*d(y,t)
    val = val + (2.0 * mask_xu) * dm1_d[y[None, :], t[:, None]]
    val = val - 4.0 * (mask_xu & mask_yt)

    return val


def _smat_b_pm_cuda(smap: SuperindexMap, dm1_d, dm2_d, *, is_plus: bool, cp) -> Any:
    """Case 2/3 overlap: SB raw plus/minus projection on ordered active pairs."""
    if bool(is_plus):
        m = np.asarray(smap.mtgeu, dtype=np.int64)
    else:
        m = np.asarray(smap.mtgtu, dtype=np.int64)
    if int(m.shape[0]) == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    t = cp.asarray(m[:, 0], dtype=cp.int64)
    u = cp.asarray(m[:, 1], dtype=cp.int64)
    x = t
    y = u

    raw_xy = _raw_sb_matrix_cuda(t, u, x, y, dm1_d, dm2_d, cp=cp)
    raw_yx = _raw_sb_matrix_cuda(t, u, y, x, dm1_d, dm2_d, cp=cp)
    smat = raw_xy + raw_yx if bool(is_plus) else raw_xy - raw_yx
    return cp.ascontiguousarray(0.5 * (smat + smat.T))


def _smat_f_pm_cuda(smap: SuperindexMap, dm2_d, *, is_plus: bool, cp) -> Any:
    """Case 8/9 overlap: SF raw plus/minus projection on ordered active pairs."""
    if bool(is_plus):
        m = np.asarray(smap.mtgeu, dtype=np.int64)
    else:
        m = np.asarray(smap.mtgtu, dtype=np.int64)
    if int(m.shape[0]) == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    t = cp.asarray(m[:, 0], dtype=cp.int64)
    u = cp.asarray(m[:, 1], dtype=cp.int64)
    x = t
    y = u

    raw_xy = 2.0 * dm2_d[t[:, None], x[None, :], u[:, None], y[None, :]]
    raw_yx = 2.0 * dm2_d[t[:, None], y[None, :], u[:, None], x[None, :]]
    smat = raw_xy + raw_yx if bool(is_plus) else raw_xy - raw_yx
    return cp.ascontiguousarray(0.5 * (smat + smat.T))


def build_smat_d_cuda(smap: SuperindexMap, dm1_d, dm2_d, *, cp) -> Any:
    """Case 5 (D) overlap matrix on GPU: SD (2*ntu x 2*ntu)."""
    ntu = int(smap.ntu)
    if ntu == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    mtu = _as_i64(cp, np.asarray(smap.mtu, dtype=np.int64))
    t = mtu[:, 0]
    u = mtu[:, 1]
    x = t
    y = u

    gutxy = dm2_d[u[:, None], t[:, None], x[None, :], y[None, :]]
    gxtuy = dm2_d[x[None, :], t[:, None], u[:, None], y[None, :]]
    mask_xt = x[None, :] == t[:, None]
    duy = dm1_d[u[:, None], y[None, :]]

    s11 = 2.0 * gutxy + (2.0 * mask_xt) * duy
    s22 = -gxtuy + (2.0 * mask_xt) * duy

    nasup = 2 * ntu
    smat = cp.empty((nasup, nasup), dtype=cp.float64)
    smat[:ntu, :ntu] = s11
    smat[ntu:, :ntu] = -0.5 * s11
    smat[:ntu, ntu:] = -0.5 * s11
    smat[ntu:, ntu:] = s22
    return cp.ascontiguousarray(0.5 * (smat + smat.T))


def build_smat_e_cuda(dm1_d, *, cp) -> Any:
    """Case 6/7 overlap: SE = 2*I - dm1."""
    nash = int(dm1_d.shape[0])
    if nash == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    return cp.ascontiguousarray(2.0 * cp.eye(nash, dtype=cp.float64) - dm1_d)


def build_smat_g_cuda(dm1_d, *, cp) -> Any:
    """Case 10/11 overlap: SG = dm1."""
    nash = int(dm1_d.shape[0])
    if nash == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    return cp.ascontiguousarray(dm1_d)


def build_smat_h_cuda(n: int, *, cp) -> Any:
    """Case 12/13 overlap: identity."""
    n = int(n)
    if n <= 0:
        return cp.empty((0, 0), dtype=cp.float64)
    return cp.eye(n, dtype=cp.float64)


def build_smat_case_cuda(case: int, smap: SuperindexMap, dm1_d, dm2_d, dm3_d, *, cp) -> Any:
    """GPU S-matrix builder dispatch for cases 1-13 (C1 only)."""
    case = int(case)
    if case == 1:
        return build_smat_a_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)
    if case == 2:
        return _smat_b_pm_cuda(smap, dm1_d, dm2_d, is_plus=True, cp=cp)
    if case == 3:
        return _smat_b_pm_cuda(smap, dm1_d, dm2_d, is_plus=False, cp=cp)
    if case == 4:
        return build_smat_c_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)
    if case == 5:
        return build_smat_d_cuda(smap, dm1_d, dm2_d, cp=cp)
    if case in (6, 7):
        return build_smat_e_cuda(dm1_d, cp=cp)
    if case == 8:
        return _smat_f_pm_cuda(smap, dm2_d, is_plus=True, cp=cp)
    if case == 9:
        return _smat_f_pm_cuda(smap, dm2_d, is_plus=False, cp=cp)
    if case in (10, 11):
        return build_smat_g_cuda(dm1_d, cp=cp)
    if case == 12:
        return build_smat_h_cuda(int(smap.nageb), cp=cp)
    if case == 13:
        return build_smat_h_cuda(int(smap.nagtb), cp=cp)
    raise ValueError("case must be 1..13")


def _raw_bb_matrix_cuda(t, u, x, y, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, *, cp):
    """Raw BB(tu,xy) before ± symmetrization (Molcas MKBB)."""
    t = cp.asarray(t, dtype=cp.int64)
    u = cp.asarray(u, dtype=cp.int64)
    x = cp.asarray(x, dtype=cp.int64)
    y = cp.asarray(y, dtype=cp.int64)

    et = epsa_d[t][:, None]
    eu = epsa_d[u][:, None]
    ex = epsa_d[x][None, :]
    ey = epsa_d[y][None, :]
    atuxy = easum_d - et - eu - ex - ey

    fp_xtyu = fp_d[x[None, :], t[:, None], y[None, :], u[:, None]]
    dm2_xtyu = dm2_d[x[None, :], t[:, None], y[None, :], u[:, None]]
    val = 4.0 * (fp_xtyu - atuxy * (0.5 * dm2_xtyu))

    mask_xt = x[None, :] == t[:, None]
    mask_yu = y[None, :] == u[:, None]
    mask_yt = y[None, :] == t[:, None]
    mask_xu = x[None, :] == u[:, None]

    # x == t:
    term = 4.0 * ((easum_d - et - ey - eu) * dm1_d[y[None, :], u[:, None]] - fd_d[y[None, :], u[:, None]])
    val = val + mask_xt * term
    val = val + (mask_xt & mask_yu) * (8.0 * (et + ey))

    # y == u:
    term = 4.0 * ((easum_d - et - ey - ex) * dm1_d[x[None, :], t[:, None]] - fd_d[x[None, :], t[:, None]])
    val = val + mask_yu * term

    # y == t:
    term = 2.0 * ((easum_d - et - eu - ex) * dm1_d[x[None, :], u[:, None]] - fd_d[x[None, :], u[:, None]])
    val = val - mask_yt * term
    val = val - (mask_yt & mask_xu) * (4.0 * (et + ex))

    # x == u:
    term = 2.0 * ((easum_d - et - eu - ey) * dm1_d[y[None, :], t[:, None]] - fd_d[y[None, :], t[:, None]])
    val = val - mask_xu * term

    return val


def _bmat_b_pm_cuda(smap: SuperindexMap, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, *, is_plus: bool, cp) -> Any:
    if bool(is_plus):
        m = np.asarray(smap.mtgeu, dtype=np.int64)
    else:
        m = np.asarray(smap.mtgtu, dtype=np.int64)
    if int(m.shape[0]) == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    t = cp.asarray(m[:, 0], dtype=cp.int64)
    u = cp.asarray(m[:, 1], dtype=cp.int64)
    x = t
    y = u

    raw_xy = _raw_bb_matrix_cuda(t, u, x, y, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, cp=cp)
    raw_yx = _raw_bb_matrix_cuda(t, u, y, x, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, cp=cp)
    bmat = raw_xy + raw_yx if bool(is_plus) else raw_xy - raw_yx
    return cp.ascontiguousarray(0.5 * (bmat + bmat.T))


def build_bmat_d_cuda(
    smap: SuperindexMap,
    dm1_d,
    dm2_d,
    *,
    epsa_d,
    easum_d,
    fd_d,
    fp_d,
    cp,
) -> Any:
    """Case 5 (D) B matrix on GPU (2*ntu x 2*ntu)."""
    ntu = int(smap.ntu)
    if ntu == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    mtu = _as_i64(cp, np.asarray(smap.mtu, dtype=np.int64))
    t = mtu[:, 0]
    u = mtu[:, 1]
    x = t
    y = u

    et = epsa_d[t][:, None]
    ex = epsa_d[x][None, :]
    factor = et + ex - easum_d
    mask_xt = x[None, :] == t[:, None]

    pref_utxy = 0.5 * dm2_d[u[:, None], t[:, None], x[None, :], y[None, :]]
    pref_xtuy = 0.5 * dm2_d[x[None, :], t[:, None], u[:, None], y[None, :]]

    bd11 = 4.0 * (fp_d[u[:, None], t[:, None], x[None, :], y[None, :]] + factor * pref_utxy)
    bd11 = bd11 + mask_xt * (
        2.0 * (fd_d[u[:, None], y[None, :]] + (et - easum_d) * dm1_d[u[:, None], y[None, :]])
    )

    bd22 = -2.0 * (fp_d[x[None, :], t[:, None], u[:, None], y[None, :]] + factor * pref_xtuy)
    bd22 = bd22 + mask_xt * (
        2.0 * (fd_d[u[:, None], y[None, :]] + (ex - easum_d) * dm1_d[u[:, None], y[None, :]])
    )

    nasup = 2 * ntu
    bmat = cp.empty((nasup, nasup), dtype=cp.float64)
    bmat[:ntu, :ntu] = bd11
    bmat[ntu:, :ntu] = -0.5 * bd11
    bmat[:ntu, ntu:] = -0.5 * bd11
    bmat[ntu:, ntu:] = bd22
    return cp.ascontiguousarray(0.5 * (bmat + bmat.T))


def build_bmat_e_cuda(dm1_d, *, epsa_d, easum_d, fd_d, cp) -> Any:
    """Case 6/7 B matrix on GPU: BE(t,x)."""
    nash = int(dm1_d.shape[0])
    if nash == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    bmat = -fd_d + (easum_d - epsa_d[None, :] - epsa_d[:, None]) * dm1_d
    bmat = bmat + 2.0 * cp.diag(epsa_d)
    return cp.ascontiguousarray(bmat)


def build_bmat_g_cuda(dm1_d, *, easum_d, fd_d, cp) -> Any:
    """Case 10/11 B matrix on GPU: BG(t,x)."""
    nash = int(dm1_d.shape[0])
    if nash == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    return cp.ascontiguousarray(fd_d - easum_d * dm1_d)


def _bmat_f_pm_cuda(smap: SuperindexMap, dm2_d, *, easum_d, fp_d, is_plus: bool, cp) -> Any:
    if bool(is_plus):
        m = np.asarray(smap.mtgeu, dtype=np.int64)
    else:
        m = np.asarray(smap.mtgtu, dtype=np.int64)
    if int(m.shape[0]) == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    t = cp.asarray(m[:, 0], dtype=cp.int64)
    u = cp.asarray(m[:, 1], dtype=cp.int64)
    x = t
    y = u

    dm2_xy = dm2_d[t[:, None], x[None, :], u[:, None], y[None, :]]
    dm2_yx = dm2_d[t[:, None], y[None, :], u[:, None], x[None, :]]
    fp_xy = fp_d[t[:, None], x[None, :], u[:, None], y[None, :]]
    fp_yx = fp_d[t[:, None], y[None, :], u[:, None], x[None, :]]

    raw_xy = 4.0 * (fp_xy - easum_d * (0.5 * dm2_xy))
    raw_yx = 4.0 * (fp_yx - easum_d * (0.5 * dm2_yx))
    bmat = raw_xy + raw_yx if bool(is_plus) else raw_xy - raw_yx
    return cp.ascontiguousarray(0.5 * (bmat + bmat.T))


def build_bmat_h_cuda(case: int, smap: SuperindexMap, fock: CASPT2Fock, *, cp) -> Any:
    """Case 12/13 B matrix on GPU: diagonal eps_a+eps_b for virtual pairs."""
    case = int(case)
    nssh = int(smap.orbs.nssh)
    if nssh == 0:
        return cp.empty((0, 0), dtype=cp.float64)

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    vo = int(nish + nash)
    diag = np.asarray(np.diag(np.asarray(fock.fifa, dtype=np.float64)), dtype=np.float64)
    eps_virt = cp.ascontiguousarray(cp.asarray(diag[vo : vo + nssh], dtype=cp.float64))

    if case == 12:
        pairs = np.asarray(smap.mageb, dtype=np.int64)
    elif case == 13:
        pairs = np.asarray(smap.magtb, dtype=np.int64)
    else:
        raise ValueError("case must be 12 or 13")
    if int(pairs.shape[0]) == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    a = cp.asarray(pairs[:, 0], dtype=cp.int64)
    b = cp.asarray(pairs[:, 1], dtype=cp.int64)
    bd = eps_virt[a] + eps_virt[b]
    return cp.diag(cp.ascontiguousarray(bd))


def build_bmat_case_cuda(
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    dm1_d,
    dm2_d,
    dm3_d,
    *,
    f3_engine,
    smat_d,
    epsa_d,
    easum_d,
    fd_d,
    fp_d,
    cp,
) -> Any:
    """GPU B-matrix builder dispatch for cases 1-13 (C1 only)."""
    case = int(case)
    if case == 1:
        return build_bmat_a_cuda(
            smap,
            fock,
            dm1_d,
            dm2_d,
            dm3_d,
            f3_engine=f3_engine,
            smat_d=smat_d,
            easum_d=easum_d,
            fd_d=fd_d,
            fp_d=fp_d,
            cp=cp,
        )
    if case == 2:
        return _bmat_b_pm_cuda(smap, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, is_plus=True, cp=cp)
    if case == 3:
        return _bmat_b_pm_cuda(smap, dm1_d, dm2_d, epsa_d, easum_d, fd_d, fp_d, is_plus=False, cp=cp)
    if case == 4:
        return build_bmat_c_cuda(
            smap,
            fock,
            dm1_d,
            dm2_d,
            dm3_d,
            f3_engine=f3_engine,
            smat_d=smat_d,
            easum_d=easum_d,
            fd_d=fd_d,
            fp_d=fp_d,
            cp=cp,
        )
    if case == 5:
        return build_bmat_d_cuda(smap, dm1_d, dm2_d, epsa_d=epsa_d, easum_d=easum_d, fd_d=fd_d, fp_d=fp_d, cp=cp)
    if case in (6, 7):
        return build_bmat_e_cuda(dm1_d, epsa_d=epsa_d, easum_d=easum_d, fd_d=fd_d, cp=cp)
    if case == 8:
        return _bmat_f_pm_cuda(smap, dm2_d, easum_d=easum_d, fp_d=fp_d, is_plus=True, cp=cp)
    if case == 9:
        return _bmat_f_pm_cuda(smap, dm2_d, easum_d=easum_d, fp_d=fp_d, is_plus=False, cp=cp)
    if case in (10, 11):
        return build_bmat_g_cuda(dm1_d, easum_d=easum_d, fd_d=fd_d, cp=cp)
    if case in (12, 13):
        return build_bmat_h_cuda(case, smap, fock, cp=cp)
    raise ValueError("case must be 1..13")


def sbdiag_cuda(
    smat_d,
    bmat_d,
    *,
    threshold_norm: float,
    threshold_s: float,
    cp,
) -> SBDecompositionDevice:
    """GPU port of `asuka.caspt2.overlap.sbdiag` using CuPy/CuSolver."""

    smat = cp.asarray(smat_d, dtype=cp.float64)
    bmat = cp.asarray(bmat_d, dtype=cp.float64)
    n = int(smat.shape[0])
    if n == 0:
        return SBDecompositionDevice(
            s_eigvals=cp.empty((0,), dtype=cp.float64),
            transform=cp.empty((0, 0), dtype=cp.float64),
            nindep=0,
            b_diag=cp.empty((0,), dtype=cp.float64),
        )

    smat = 0.5 * (smat + smat.T)
    bmat = 0.5 * (bmat + bmat.T)

    sdiag = cp.diagonal(smat).copy()
    idx = cp.arange(1, n + 1, dtype=cp.float64)
    sca = cp.zeros((n,), dtype=cp.float64)
    mask_diag = sdiag > float(threshold_norm)
    sca[mask_diag] = (1.0 + 3.0e-6 * idx[mask_diag]) / cp.sqrt(sdiag[mask_diag])

    smat_scaled = (sca[:, None] * smat) * sca[None, :]
    smat_scaled = 0.5 * (smat_scaled + smat_scaled.T)
    s_eigvals, u_s = cp.linalg.eigh(smat_scaled)

    mask = s_eigvals >= float(threshold_s)
    nindep = int(cp.asnumpy(cp.count_nonzero(mask)))
    if nindep == 0:
        return SBDecompositionDevice(
            s_eigvals=s_eigvals,
            transform=cp.empty((n, 0), dtype=cp.float64),
            nindep=0,
            b_diag=cp.empty((0,), dtype=cp.float64),
        )

    u_ind = u_s[:, mask]
    s_ind = s_eigvals[mask]
    s_invsqrt = 1.0 / cp.sqrt(s_ind)
    x_scaled = u_ind * s_invsqrt[None, :]
    x = sca[:, None] * x_scaled

    b_orth = x.T @ bmat @ x
    b_orth = 0.5 * (b_orth + b_orth.T)
    b_eigvals, u_b = cp.linalg.eigh(b_orth)

    transform = x @ u_b

    return SBDecompositionDevice(
        s_eigvals=cp.ascontiguousarray(s_eigvals),
        transform=cp.ascontiguousarray(transform),
        nindep=int(nindep),
        b_diag=cp.ascontiguousarray(b_eigvals),
    )
