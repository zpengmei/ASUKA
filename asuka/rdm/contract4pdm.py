from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _get_epq_action_cache
from asuka.rdm.rdm123 import _STEP_TO_OCC_F64, _fill_epq_vec

try:  # optional Cython in-place CSC @ dense kernels
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
    )
except Exception:  # pragma: no cover
    _csc_matmul_dense_inplace_cy = None

try:  # optional SciPy-backed sparse matmul for E_pq applications
    from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
    from asuka.contract import _sp as _sp  # noqa: PLC0415
except Exception:  # pragma: no cover
    _sp = None
    _epq_spmat_list = None  # type: ignore[assignment]


def _apply_epq_dense(
    drt: DRT,
    cache,
    occ: np.ndarray,
    mats: list[object | None] | None,
    *,
    p: int,
    q: int,
    x: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill ``out[:] = E_pq @ x`` for dense RHS matrix ``x`` (CSF basis)."""

    p = int(p)
    q = int(q)
    if x.ndim != 2 or out.shape != x.shape:
        raise ValueError("x/out must have the same 2D shape")

    if p == q:
        np.multiply(occ[:, p][:, None], x, out=out)
        return

    if mats is not None:
        mat = mats[p * int(drt.norb) + q]
        if mat is None:
            raise AssertionError("missing E_pq sparse matrix")
        if _csc_matmul_dense_inplace_cy is not None:
            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                mat.indptr, mat.indices, mat.data, x, out
            )
        else:
            out[:] = mat.dot(x)  # type: ignore[operator]
        return

    # Fallback: apply to each RHS column.
    ncol = int(x.shape[1])
    for col in range(ncol):
        _fill_epq_vec(drt, cache, x[:, col], p=p, q=q, out=out[:, col])


def _build_t1(
    drt: DRT,
    c: np.ndarray,
    *,
    occ: np.ndarray,
    mats: list[object | None] | None,
) -> np.ndarray:
    """Return T[pq, :] = E_pq |c> (row-major in pq)."""

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nops = norb * norb

    cache = _get_epq_action_cache(drt)
    t1 = np.empty((nops, ncsf), dtype=np.float64)
    c_col = c.reshape(ncsf, 1)
    out_col = np.empty((ncsf, 1), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            out = t1[pq]
            if p == q:
                np.multiply(occ[:, p], c, out=out)
            elif mats is not None:
                mat = mats[pq]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    out_col[:, 0].fill(0.0)
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, c_col, out_col
                    )
                    out[:] = out_col[:, 0]
                else:
                    out[:] = mat.dot(c)  # type: ignore[operator]
            else:
                _fill_epq_vec(drt, cache, c, p=p, q=q, out=out)
    return np.asarray(t1, dtype=np.float64, order="C")


def _compute_g_cedf_aedf(
    drt: DRT,
    eri: np.ndarray,
    *,
    x_df: np.ndarray,
    occ: np.ndarray,
    mats: list[object | None] | None,
) -> np.ndarray:
    """Compute G[:,ac] = sum_{d,f,e} (df|ce) E_ae E_df |c> (vectorized over c)."""

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nops = norb * norb

    g = np.zeros((ncsf, nops), dtype=np.float64)
    cache = _get_epq_action_cache(drt)

    y = np.empty((ncsf, nops), dtype=np.float64)
    for e in range(norb):
        # W[df,c] = (df|c e)
        w = np.asarray(eri[:, :, :, e].reshape(nops, norb), dtype=np.float64, order="C")
        for a in range(norb):
            _apply_epq_dense(drt, cache, occ, mats, p=a, q=e, x=x_df, out=y)
            contrib = y @ w  # (ncsf, norb)
            for c in range(norb):
                g[:, a * norb + c] += contrib[:, c]
    return np.asarray(g, dtype=np.float64, order="C")


def _compute_g_aedf_ecdf(
    drt: DRT,
    eri: np.ndarray,
    *,
    x_df: np.ndarray,
    occ: np.ndarray,
    mats: list[object | None] | None,
) -> np.ndarray:
    """Compute G[:,ac] = sum_{d,f,e} (df|ea) E_ec E_df |c> (vectorized over a)."""

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nops = norb * norb

    g = np.zeros((ncsf, nops), dtype=np.float64)
    cache = _get_epq_action_cache(drt)

    y = np.empty((ncsf, nops), dtype=np.float64)
    for e in range(norb):
        # W[df,a] = (df|e a)
        w = np.asarray(eri[:, :, e, :].reshape(nops, norb), dtype=np.float64, order="C")
        for c in range(norb):
            _apply_epq_dense(drt, cache, occ, mats, p=e, q=c, x=x_df, out=y)
            contrib = y @ w  # (ncsf, norb)
            for a in range(norb):
                g[:, a * norb + c] += contrib[:, a]
    return np.asarray(g, dtype=np.float64, order="C")


def _compute_f3_std(
    drt: DRT,
    g: np.ndarray,
    *,
    b_pq: np.ndarray,
    occ: np.ndarray,
    mats: list[object | None] | None,
) -> np.ndarray:
    """Compute f3[p,q,r,s,a,c] = <c|E_pq E_rs O_ac|c> from G[:,ac] = (O_ac|c>)."""

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nops = norb * norb

    if g.shape != (ncsf, nops) or b_pq.shape != (nops, ncsf):
        raise ValueError("shape mismatch for G or bra vectors")

    cache = _get_epq_action_cache(drt)
    f3_flat = np.empty((nops, nops, nops), dtype=np.float64)
    y = np.empty((ncsf, nops), dtype=np.float64)
    m = np.empty((nops, nops), dtype=np.float64)
    for rs in range(nops):
        r = rs // norb
        s = rs - r * norb
        _apply_epq_dense(drt, cache, occ, mats, p=r, q=s, x=g, out=y)
        np.matmul(b_pq, y, out=m)
        f3_flat[:, rs, :] = m
    return np.asarray(f3_flat.reshape(norb, norb, norb, norb, norb, norb), dtype=np.float64, order="C")


def _make_f3ca_f3ac_pyscf(
    drt: DRT,
    civec: np.ndarray,
    eri: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (f3ca, f3ac) for a CSF wavefunction in PySCF `_contract4pdm` conventions.

    Notes
    -----
    This is a **slow-but-correct** CPU implementation intended for small active
    spaces (validation / fallback).  It matches the shapes used by PySCF:
    - `eri[p,q,r,s]` in chemist order, shape (norb,norb,norb,norb)
    - return `f3ca` and `f3ac`, each shape (norb,norb,norb,norb,norb,norb)
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nops = norb * norb

    c = np.asarray(civec, dtype=np.float64).ravel()
    if c.size != ncsf:
        raise ValueError("civec has wrong length")
    eri = np.asarray(eri, dtype=np.float64)
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError("eri must have shape (norb,norb,norb,norb)")

    max_memory_bytes = float(max_memory_mb) * 1e6
    if max_memory_bytes <= 0:
        max_memory_bytes = 1.0

    # Rough memory guard: (t1, g, scratch y, f3_flat) dominate.
    est_bytes = (
        float(nops) * float(ncsf) * 8.0 * 2.0  # t1 + bra
        + float(ncsf) * float(nops) * 8.0 * 2.0  # g + scratch y
        + float(nops) * float(nops) * float(nops) * 8.0  # f3_flat
    )
    if est_bytes > max_memory_bytes:
        raise MemoryError(
            f"contract4pdm allocation would require ~{est_bytes/1e6:.1f} MB "
            f"(norb={norb}, ncsf={ncsf}); increase max_memory_mb or use a smaller active space"
        )

    cache = _get_epq_action_cache(drt)
    occ = np.asarray(_STEP_TO_OCC_F64[cache.steps], dtype=np.float64, order="C")

    mats = None
    if _sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)

    # T1[pq,:] = E_pq |c> (row-major in pq).
    t1 = _build_t1(drt, c, occ=occ, mats=mats)

    # Bra vectors: B[pq,:] = E_qp |c>.
    b_pq = t1.reshape(norb, norb, ncsf).transpose(1, 0, 2).reshape(nops, ncsf)
    b_pq = np.asarray(b_pq, dtype=np.float64, order="C")

    # Dense matrix of ket vectors: X[:,df] = E_df |c>.
    x_df = np.ascontiguousarray(t1.T)  # (ncsf, nops)

    # f3ca: kernel NEVPTkern_cedf_aedf -> output in PySCF order (p,q,r,s,a,c).
    g_cedf_aedf = _compute_g_cedf_aedf(drt, eri, x_df=x_df, occ=occ, mats=mats)
    f3ca = _compute_f3_std(drt, g_cedf_aedf, b_pq=b_pq, occ=occ, mats=mats)

    # f3ac: kernel NEVPTkern_aedf_ecdf -> output in PySCF order (p,q,r,s,a,c).
    g_aedf_ecdf = _compute_g_aedf_ecdf(drt, eri, x_df=x_df, occ=occ, mats=mats)
    f3ac = _compute_f3_std(drt, g_aedf_ecdf, b_pq=b_pq, occ=occ, mats=mats)

    return f3ca, np.asarray(f3ac, dtype=np.float64, order="C")


def _contract4pdm_csf_pyscf(
    drt: DRT,
    civec: np.ndarray,
    eri: np.ndarray,
    *,
    kern: str,
    max_memory_mb: float = 4000.0,
) -> np.ndarray:
    """Compute one of PySCF's contracted-4PDM tensors (`f3ca` or `f3ac`) from a CSF CI vector."""

    kern = str(kern)
    if kern == "NEVPTkern_cedf_aedf":
        f3ca, _f3ac = _make_f3ca_f3ac_pyscf(drt, civec, eri, max_memory_mb=max_memory_mb)
        return f3ca
    if kern == "NEVPTkern_aedf_ecdf":
        _f3ca, f3ac = _make_f3ca_f3ac_pyscf(drt, civec, eri, max_memory_mb=max_memory_mb)
        return f3ac
    raise ValueError(f"unsupported kern={kern!r}")
