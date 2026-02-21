"""MS/XMS CASPT2 HCOUP contractions on CUDA (CuPy, C1, FP64).

This mirrors :func:`asuka.caspt2.hcoup.hcoup_case_contribution` but operates on
device-resident arrays and avoids host transfers of the large ``row_dots`` blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.caspt2.superindex import SuperindexMap


_INDEX_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


def _get_index_cache(cp, smap: SuperindexMap) -> dict[str, Any]:
    nash = int(smap.orbs.nash)
    try:
        dev = int(cp.cuda.runtime.getDevice())
    except Exception:
        dev = 0
    key = (dev, nash)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    cached = {}
    cached["mtuv"] = cp.asarray(np.asarray(smap.mtuv, dtype=np.int32), dtype=cp.int32)
    cached["mtgeu"] = cp.asarray(np.asarray(smap.mtgeu, dtype=np.int32), dtype=cp.int32)
    cached["mtgtu"] = cp.asarray(np.asarray(smap.mtgtu, dtype=np.int32), dtype=cp.int32)
    cached["mtu"] = cp.asarray(np.asarray(smap.mtu, dtype=np.int32), dtype=cp.int32)
    _INDEX_CACHE[key] = cached
    return cached


def hcoup_case_contribution_cuda(
    case: int,
    smap: SuperindexMap,
    row_dots: Any,
    tg1: Any,
    tg2: Any,
    tg3: Any,
    *,
    ovl: float,
):
    """CUDA analogue of `hcoup_case_contribution` (returns a CuPy scalar)."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for hcoup_case_contribution_cuda") from e

    case = int(case)
    if case < 1 or case > 13:
        raise ValueError(f"Invalid case: {case}")
    ovl = float(ovl)

    row_dots = cp.asarray(row_dots, dtype=cp.float64)
    if int(row_dots.size) == 0:
        return cp.float64(0.0)
    tg1 = cp.asarray(tg1, dtype=cp.float64)
    tg2 = cp.asarray(tg2, dtype=cp.float64)
    tg3 = cp.asarray(tg3, dtype=cp.float64)

    idx = _get_index_cache(cp, smap)

    # CASE(1): SA(tuv,xyz)
    if case == 1:
        ntuv = int(smap.ntuv)
        if ntuv == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntuv, ntuv):
            raise ValueError(f"case 1 row_dots shape mismatch: {row_dots.shape} vs {(ntuv, ntuv)}")

        mtuv = idx["mtuv"]
        t = mtuv[:, 0].astype(cp.int32, copy=False)
        u = mtuv[:, 1].astype(cp.int32, copy=False)
        v = mtuv[:, 2].astype(cp.int32, copy=False)
        x = t
        y = u
        z = v
        n = int(smap.orbs.nash)

        # tmp = tg3[v,u,x,t,y,z] with x from jas and t from ias -> swap x/t axes.
        tg3_mat = tg3.transpose(0, 1, 3, 2, 4, 5).reshape(n * n * n, n * n * n)
        left = (v * (n * n) + u * n + t).astype(cp.int64)
        right = (x * (n * n) + y * n + z).astype(cp.int64)
        ker = tg3_mat[left][:, right]  # (ntuv,ntuv)

        # Masks
        mask_yu = (u[:, None] == y[None, :])
        mask_yt = (t[:, None] == y[None, :])
        mask_xu = (u[:, None] == x[None, :])
        mask_xt = (t[:, None] == x[None, :])

        # y == u: + tg2[v,z,x,t]
        term = tg2[v[:, None], z[None, :], x[None, :], t[:, None]]
        cp.multiply(term, mask_yu, out=term)
        cp.add(ker, term, out=ker)

        # y == t: + tg2[v,u,x,z]
        term = tg2[v[:, None], u[:, None], x[None, :], z[None, :]]
        cp.multiply(term, mask_yt, out=term)
        cp.add(ker, term, out=ker)

        # y == t and x == u: + tg1[v,z]
        mask = cp.logical_and(mask_yt, mask_xu)
        term = tg1[v[:, None], z[None, :]]
        cp.multiply(term, mask, out=term)
        cp.add(ker, term, out=ker)

        # x == u: + tg2[v,t,y,z]
        term = tg2[v[:, None], t[:, None], y[None, :], z[None, :]]
        cp.multiply(term, mask_xu, out=term)
        cp.add(ker, term, out=ker)

        # sa = -tmp
        ker *= -1.0

        # x == t: + 2*tg2[v,u,y,z]
        term = 2.0 * tg2[v[:, None], u[:, None], y[None, :], z[None, :]]
        cp.multiply(term, mask_xt, out=term)
        cp.add(ker, term, out=ker)

        # x == t and y == u: + 2*tg1[v,z]
        mask = cp.logical_and(mask_xt, mask_yu)
        term = 2.0 * tg1[v[:, None], z[None, :]]
        cp.multiply(term, mask, out=term)
        cp.add(ker, term, out=ker)

        return cp.sum(ker * row_dots)

    # CASE(2): SBP(tu,xy)
    if case == 2:
        ntgeu = int(smap.ntgeu)
        if ntgeu == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntgeu, ntgeu):
            raise ValueError(f"case 2 row_dots shape mismatch: {row_dots.shape} vs {(ntgeu, ntgeu)}")

        mt = idx["mtgeu"]
        t = mt[:, 0].astype(cp.int32, copy=False)
        u = mt[:, 1].astype(cp.int32, copy=False)
        x = t
        y = u

        sbtuxy = 2.0 * tg2[x[None, :], t[:, None], y[None, :], u[:, None]]
        sbtuyx = 2.0 * tg2[y[None, :], t[:, None], x[None, :], u[:, None]]

        mask_xt = (t[:, None] == x[None, :])
        mask_yu = (u[:, None] == y[None, :])
        mask_yt = (t[:, None] == y[None, :])
        mask_xu = (u[:, None] == x[None, :])

        term = tg1[y[None, :], u[:, None]]
        sbtuxy = sbtuxy + (-4.0) * term * mask_xt
        sbtuyx = sbtuyx + (2.0) * term * mask_xt
        sbtuxy = sbtuxy + (8.0 * ovl) * (mask_xt & mask_yu)
        sbtuyx = sbtuyx + (-4.0 * ovl) * (mask_xt & mask_yu)

        term = tg1[x[None, :], t[:, None]]
        sbtuxy = sbtuxy + (-4.0) * term * mask_yu
        sbtuyx = sbtuyx + (2.0) * term * mask_yu

        term = tg1[x[None, :], u[:, None]]
        sbtuxy = sbtuxy + (2.0) * term * mask_yt
        sbtuyx = sbtuyx + (-4.0) * term * mask_yt
        sbtuxy = sbtuxy + (-4.0 * ovl) * (mask_yt & mask_xu)
        sbtuyx = sbtuyx + (8.0 * ovl) * (mask_yt & mask_xu)

        term = tg1[y[None, :], t[:, None]]
        sbtuxy = sbtuxy + (2.0) * term * mask_xu
        sbtuyx = sbtuyx + (-4.0) * term * mask_xu

        sbp = sbtuxy + sbtuyx
        return cp.sum(sbp * row_dots)

    # CASE(3): SBM(tu,xy)
    if case == 3:
        ntgtu = int(smap.ntgtu)
        if ntgtu == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntgtu, ntgtu):
            raise ValueError(f"case 3 row_dots shape mismatch: {row_dots.shape} vs {(ntgtu, ntgtu)}")

        mt = idx["mtgtu"]
        t = mt[:, 0].astype(cp.int32, copy=False)
        u = mt[:, 1].astype(cp.int32, copy=False)
        x = t
        y = u

        sbtuxy = 2.0 * tg2[x[None, :], t[:, None], y[None, :], u[:, None]]
        sbtuyx = 2.0 * tg2[y[None, :], t[:, None], x[None, :], u[:, None]]

        mask_xt = (t[:, None] == x[None, :])
        mask_yu = (u[:, None] == y[None, :])
        mask_yt = (t[:, None] == y[None, :])
        mask_xu = (u[:, None] == x[None, :])

        term = tg1[y[None, :], u[:, None]]
        sbtuxy = sbtuxy + (-4.0) * term * mask_xt
        sbtuyx = sbtuyx + (2.0) * term * mask_xt
        sbtuxy = sbtuxy + (8.0 * ovl) * (mask_xt & mask_yu)
        sbtuyx = sbtuyx + (-4.0 * ovl) * (mask_xt & mask_yu)

        term = tg1[x[None, :], t[:, None]]
        sbtuxy = sbtuxy + (-4.0) * term * mask_yu
        sbtuyx = sbtuyx + (2.0) * term * mask_yu

        term = tg1[x[None, :], u[:, None]]
        sbtuxy = sbtuxy + (2.0) * term * mask_yt
        sbtuyx = sbtuyx + (-4.0) * term * mask_yt
        sbtuxy = sbtuxy + (-4.0 * ovl) * (mask_yt & mask_xu)
        sbtuyx = sbtuyx + (8.0 * ovl) * (mask_yt & mask_xu)

        term = tg1[y[None, :], t[:, None]]
        sbtuxy = sbtuxy + (2.0) * term * mask_xu
        sbtuyx = sbtuyx + (-4.0) * term * mask_xu

        sbm = sbtuxy - sbtuyx
        return cp.sum(sbm * row_dots)

    # CASE(4): SC(xuv,tyz)
    if case == 4:
        ntuv = int(smap.ntuv)
        if ntuv == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntuv, ntuv):
            raise ValueError(f"case 4 row_dots shape mismatch: {row_dots.shape} vs {(ntuv, ntuv)}")

        mtuv = idx["mtuv"]
        x_r = mtuv[:, 0].astype(cp.int32, copy=False)
        u_r = mtuv[:, 1].astype(cp.int32, copy=False)
        v_r = mtuv[:, 2].astype(cp.int32, copy=False)
        t_c = x_r
        y_c = u_r
        z_c = v_r
        n = int(smap.orbs.nash)

        tg3_mat = tg3.reshape(n * n * n, n * n * n)
        left = (v_r * (n * n) + u_r * n + x_r).astype(cp.int64)
        right = (t_c * (n * n) + y_c * n + z_c).astype(cp.int64)
        ker = tg3_mat[left][:, right]

        mask_yu = (u_r[:, None] == y_c[None, :])
        mask_yt_col = (y_c == t_c)  # (ntuv,)
        mask_xu_row = (x_r == u_r)  # (ntuv,)

        # y == u: + tg2[v,z,x,t]  (x is row)
        term = tg2[v_r[:, None], z_c[None, :], x_r[:, None], t_c[None, :]]
        cp.multiply(term, mask_yu, out=term)
        cp.add(ker, term, out=ker)

        # y == t: + tg2[v,u,x,z]
        term = tg2[v_r[:, None], u_r[:, None], x_r[:, None], z_c[None, :]]
        cp.multiply(term, mask_yt_col[None, :], out=term)
        cp.add(ker, term, out=ker)

        # y == t and x == u: + tg1[v,z]
        mask = cp.logical_and(mask_xu_row[:, None], mask_yt_col[None, :])
        term = tg1[v_r[:, None], z_c[None, :]]
        cp.multiply(term, mask, out=term)
        cp.add(ker, term, out=ker)

        # x == u: + tg2[v,t,y,z]
        term = tg2[v_r[:, None], t_c[None, :], y_c[None, :], z_c[None, :]]
        cp.multiply(term, mask_xu_row[:, None], out=term)
        cp.add(ker, term, out=ker)

        return cp.sum(ker * row_dots)

    # CASE(5): SD 2x2 block over active pairs (tu)
    if case == 5:
        ntu = int(smap.ntu)
        nasup = 2 * ntu
        if ntu == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (nasup, nasup):
            raise ValueError(f"case 5 row_dots shape mismatch: {row_dots.shape} vs {(nasup, nasup)}")

        mtu = idx["mtu"]
        t = mtu[:, 0].astype(cp.int32, copy=False)
        u = mtu[:, 1].astype(cp.int32, copy=False)
        x = t
        y = u

        gutxy = tg2[u[:, None], t[:, None], x[None, :], y[None, :]]
        sd11 = 2.0 * gutxy
        sd12 = -gutxy
        sd21 = -gutxy
        sd22 = -tg2[x[None, :], t[:, None], u[:, None], y[None, :]]

        mask_tx = (t[:, None] == x[None, :])
        guy = tg1[u[:, None], y[None, :]]
        sd11 = sd11 + (2.0 * guy) * mask_tx
        sd12 = sd12 + (-guy) * mask_tx
        sd21 = sd21 + (-guy) * mask_tx
        sd22 = sd22 + (2.0 * guy) * mask_tx

        # Assemble 2x2 block contribution without Python loops.
        rd = row_dots
        blk11 = rd[:ntu, :ntu]
        blk12 = rd[:ntu, ntu:]
        blk21 = rd[ntu:, :ntu]
        blk22 = rd[ntu:, ntu:]
        return cp.sum(sd11 * blk11 + sd12 * blk12 + sd21 * blk21 + sd22 * blk22)

    # CASE(6,7): SE(t,x) = -tg1[x,t] + 2*ovl*delta(x,t)
    if case in (6, 7):
        nash = int(smap.orbs.nash)
        if nash == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (nash, nash):
            raise ValueError(f"case {case} row_dots shape mismatch: {row_dots.shape} vs {(nash, nash)}")
        se = -tg1.T  # se[t,x] = -tg1[x,t]
        se = se + (2.0 * ovl) * cp.eye(nash, dtype=cp.float64)
        return cp.sum(se * row_dots)

    # CASE(8): SFP(tu,xy)
    if case == 8:
        ntgeu = int(smap.ntgeu)
        if ntgeu == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntgeu, ntgeu):
            raise ValueError(f"case 8 row_dots shape mismatch: {row_dots.shape} vs {(ntgeu, ntgeu)}")
        mt = idx["mtgeu"]
        t = mt[:, 0].astype(cp.int32, copy=False)
        u = mt[:, 1].astype(cp.int32, copy=False)
        x = t
        y = u
        sftuxy = 2.0 * tg2[t[:, None], x[None, :], u[:, None], y[None, :]]
        sftuyx = 2.0 * tg2[t[:, None], y[None, :], u[:, None], x[None, :]]
        sfp = sftuxy + sftuyx
        return cp.sum(sfp * row_dots)

    # CASE(9): SFM(tu,xy)
    if case == 9:
        ntgtu = int(smap.ntgtu)
        if ntgtu == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (ntgtu, ntgtu):
            raise ValueError(f"case 9 row_dots shape mismatch: {row_dots.shape} vs {(ntgtu, ntgtu)}")
        mt = idx["mtgtu"]
        t = mt[:, 0].astype(cp.int32, copy=False)
        u = mt[:, 1].astype(cp.int32, copy=False)
        x = t
        y = u
        sftuxy = 2.0 * tg2[t[:, None], x[None, :], u[:, None], y[None, :]]
        sftuyx = 2.0 * tg2[t[:, None], y[None, :], u[:, None], x[None, :]]
        sfm = sftuxy - sftuyx
        return cp.sum(sfm * row_dots)

    # CASE(10,11): SG(t,x) = tg1[t,x]
    if case in (10, 11):
        nash = int(smap.orbs.nash)
        if nash == 0:
            return cp.float64(0.0)
        if tuple(row_dots.shape) != (nash, nash):
            raise ValueError(f"case {case} row_dots shape mismatch: {row_dots.shape} vs {(nash, nash)}")
        return cp.sum(tg1 * row_dots)

    # CASE(12,13): overlap-only (purely external)
    if case in (12, 13):
        if abs(float(ovl)) < 1.0e-12:
            return cp.float64(0.0)
        return cp.float64(ovl) * cp.trace(row_dots)

    raise RuntimeError(f"Unhandled case: {case}")

