"""Heat-bath integral index: pre-sorted integral tables for HB-SCI screening.

Reference: Holmes, Sharma, Umrigar, JCTC 2017, 13, 1595.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HeatBathIntegralIndex:
    """Pre-sorted integral tables for heat-bath screening.

    One-body integrals (h_eff) are sorted by absolute value descending.
    Two-body integrals are stored in CSR format keyed by (p,q) flat index,
    with (r,s) entries sorted by |v_{pq,rs}| descending within each row.
    """

    # One-body: |h_eff_pq| sorted descending
    h1_pq: np.ndarray  # int32 [n_h1, 2] — (p,q) pairs
    h1_abs: np.ndarray  # float64 [n_h1] — |h_eff_pq| descending
    h1_signed: np.ndarray  # float64 [n_h1] — signed h_eff_pq values

    # Two-body CSR: for each pq_flat, rs sorted by |v_{pq,rs}| desc
    pq_ptr: np.ndarray  # int64 [norb^2 + 1]
    rs_idx: np.ndarray  # int32 [nnz_2e] — flat r*norb+s
    v_abs: np.ndarray  # float64 [nnz_2e]
    v_signed: np.ndarray  # float64 [nnz_2e] — signed values

    # Row-level max for fast skip
    pq_max_v: np.ndarray  # float64 [norb^2]

    # Diagonal ERI: eri_diag_t[r, pq] = v[pq, r*norb+r]  (for occupancy-weighted g)
    eri_diag_t: np.ndarray  # float64 [norb, norb^2]

    norb: int

    @property
    def n_h1(self) -> int:
        return int(self.h1_abs.shape[0])

    @property
    def nnz_2e(self) -> int:
        return int(self.v_abs.shape[0])


def build_hb_index(h1e_eff: np.ndarray, eri_4d: np.ndarray, norb: int) -> HeatBathIntegralIndex:
    """Build sorted integral tables from pre-materialized integrals.

    Uses vectorized NumPy argsort — no Python loops over integral entries.

    Parameters
    ----------
    h1e_eff : ndarray, shape (norb, norb)
        Effective one-body integrals (h - 0.5 * J_ps).
    eri_4d : ndarray, shape (norb, norb, norb, norb)
        Full 4-index ERI in chemist notation (pq|rs).
    norb : int
        Number of active orbitals.
    """
    norb = int(norb)
    nops = norb * norb

    h1e_eff = np.asarray(h1e_eff, dtype=np.float64).reshape(norb, norb)
    eri_4d = np.asarray(eri_4d, dtype=np.float64).reshape(norb, norb, norb, norb)

    # --- One-body sort ---
    h1_flat = h1e_eff.ravel()
    h1_abs_all = np.abs(h1_flat)
    # Sort descending, filter zeros
    order_h1 = np.argsort(h1_abs_all)[::-1]
    mask_h1 = h1_abs_all[order_h1] > 0.0
    order_h1 = order_h1[mask_h1]

    h1_pq = np.column_stack([order_h1 // norb, order_h1 % norb]).astype(np.int32)
    h1_abs = h1_abs_all[order_h1].astype(np.float64)
    h1_signed = h1_flat[order_h1].astype(np.float64)

    # --- Two-body CSR: vectorized sort across all rows ---
    eri_2d = eri_4d.reshape(nops, nops)  # [pq, rs]
    abs_eri = np.abs(eri_2d)  # [nops, nops]

    # Sort each row descending by |v|  (NumPy axis-wise argsort)
    order_2d = np.argsort(-abs_eri, axis=1)  # [nops, nops]

    row_idx = np.arange(nops, dtype=np.intp)[:, None]
    v_abs_sorted = abs_eri[row_idx, order_2d]     # [nops, nops] sorted |v|
    v_signed_sorted = eri_2d[row_idx, order_2d]   # [nops, nops] signed v

    # Max per row (first entry after sort)
    pq_max_v = v_abs_sorted[:, 0].copy()  # [nops]

    # Build CSR: only keep non-zero entries
    nonzero_mask = v_abs_sorted > 0.0  # [nops, nops]
    nnz_per_row = nonzero_mask.sum(axis=1)  # [nops]

    pq_ptr = np.zeros(nops + 1, dtype=np.int64)
    np.cumsum(nnz_per_row, out=pq_ptr[1:])
    total_nnz = int(pq_ptr[-1])

    if total_nnz > 0:
        rs_idx = order_2d[nonzero_mask].astype(np.int32)
        v_abs = v_abs_sorted[nonzero_mask].astype(np.float64)
        v_signed = v_signed_sorted[nonzero_mask].astype(np.float64)
    else:
        rs_idx = np.zeros(0, dtype=np.int32)
        v_abs = np.zeros(0, dtype=np.float64)
        v_signed = np.zeros(0, dtype=np.float64)

    # --- Diagonal ERI: eri_diag_t[r, pq] = v[pq, rr] ---
    # Used for vectorized occupancy-weighted g_flat: g_diag = occ @ eri_diag_t
    diag_ids = np.arange(norb) * (norb + 1)  # flat indices of (r,r): r*norb+r
    eri_diag_t = eri_2d[:, diag_ids].T.copy()  # [norb, nops]

    return HeatBathIntegralIndex(
        h1_pq=h1_pq,
        h1_abs=h1_abs,
        h1_signed=h1_signed,
        pq_ptr=pq_ptr,
        rs_idx=rs_idx,
        v_abs=v_abs,
        v_signed=v_signed,
        pq_max_v=pq_max_v,
        eri_diag_t=eri_diag_t,
        norb=norb,
    )


def build_hb_index_from_df(h1e_eff: np.ndarray, l_full: np.ndarray, norb: int) -> HeatBathIntegralIndex:
    """Build HB index from DF 3-index integrals by materializing the ERI.

    Parameters
    ----------
    h1e_eff : ndarray, shape (norb, norb)
        Effective one-body integrals.
    l_full : ndarray, shape (norb^2, naux)
        DF 3-index integrals (Cholesky vectors).
    norb : int
        Number of active orbitals.
    """
    norb = int(norb)
    nops = norb * norb
    l_full = np.asarray(l_full, dtype=np.float64)
    if l_full.shape[0] != nops:
        raise ValueError(f"l_full.shape[0]={l_full.shape[0]} != norb^2={nops}")
    # Materialize (pq|rs) = L_pq^P L_rs^P  (nops × nops GEMM)
    eri_2d = l_full @ l_full.T  # [nops, nops]
    eri_4d = eri_2d.reshape(norb, norb, norb, norb)
    return build_hb_index(h1e_eff, eri_4d, norb)


def build_g_base(
    hb_index: HeatBathIntegralIndex,
    cutoff: float,
) -> np.ndarray:
    """Compute the cutoff-fixed part of g_flat (same for all source CSFs at this cutoff).

    This includes:
    - h_eff contributions above cutoff
    - 0.5 × sum of off-diagonal v_{pq,rs} (r≠s) with |v| above cutoff

    The occ-dependent diagonal part is computed separately via a GEMM
    (occ_batch @ eri_diag_t) and added per-CSF.

    Parameters
    ----------
    hb_index : HeatBathIntegralIndex
    cutoff : float
        Absolute value cutoff.

    Returns
    -------
    g_base : ndarray, shape (norb^2,)
        Fixed part of g_flat valid for all source CSFs with this cutoff level.
    """
    norb = hb_index.norb
    nops = norb * norb
    g_base = np.zeros(nops, dtype=np.float64)

    # One-body: vectorized binary search on sorted h1_abs
    if hb_index.n_h1 > 0 and hb_index.h1_abs[0] >= cutoff:
        k_end = int(np.searchsorted(-hb_index.h1_abs, -cutoff))
        if k_end > 0:
            pq_arr = (hb_index.h1_pq[:k_end, 0].astype(np.intp) * norb
                      + hb_index.h1_pq[:k_end, 1].astype(np.intp))
            np.add.at(g_base, pq_arr, hb_index.h1_signed[:k_end])

    # Two-body off-diagonal (r≠s): fully vectorized using add.reduceat.
    # Avoids np.repeat(arange, counts) by using CSR row boundaries directly.
    # Strategy:
    #   1. Compute per-entry mask: |v| >= cutoff AND r != s.
    #   2. Zero-out filtered entries in a working array.
    #   3. np.add.reduceat over CSR row boundaries → per-pq partial sums.
    # Memory: 2 × nnz float64 temporaries (no O(nnz) index array).
    nnz = hb_index.nnz_2e
    if nnz > 0 and hb_index.pq_max_v.max() >= cutoff:
        rs_all = hb_index.rs_idx   # [nnz] int32
        r_all = rs_all // norb
        s_all = rs_all % norb
        offdiag_above = (hb_index.v_abs >= cutoff) & (r_all != s_all)  # [nnz] bool

        if offdiag_above.any():
            v_filtered = np.where(offdiag_above, 0.5 * hb_index.v_signed, 0.0)  # [nnz]
            # add.reduceat: for each pq, sum v_filtered[pq_ptr[pq]:pq_ptr[pq+1]]
            starts = hb_index.pq_ptr[:-1].astype(np.intp)
            row_sums = np.add.reduceat(v_filtered, starts)  # [nops]
            g_base += row_sums

    return g_base


def upload_hb_index(hb_index: HeatBathIntegralIndex, cp: Any) -> dict:
    """Upload all HB index arrays to GPU as CuPy arrays."""
    return {
        "h1_pq": cp.asarray(hb_index.h1_pq, dtype=cp.int32),
        "h1_abs": cp.asarray(hb_index.h1_abs, dtype=cp.float64),
        "h1_signed": cp.asarray(hb_index.h1_signed, dtype=cp.float64),
        "pq_ptr": cp.asarray(hb_index.pq_ptr, dtype=cp.int64),
        "rs_idx": cp.asarray(hb_index.rs_idx, dtype=cp.int32),
        "v_abs": cp.asarray(hb_index.v_abs, dtype=cp.float64),
        "v_signed": cp.asarray(hb_index.v_signed, dtype=cp.float64),
        "pq_max_v": cp.asarray(hb_index.pq_max_v, dtype=cp.float64),
        "eri_diag_t": cp.asarray(hb_index.eri_diag_t, dtype=cp.float64),
        "norb": int(hb_index.norb),
    }


def build_g_base_gpu(hb_dev: dict, cutoff: float, norb: int, cp: Any) -> Any:
    """GPU equivalent of build_g_base using CuPy.

    Mirrors the CPU ``build_g_base`` logic but operates entirely on GPU arrays
    in ``hb_dev`` (as returned by :func:`upload_hb_index`).  Lazily caches
    derived index arrays (``h1_pq_flat``, ``rs_r_all``, ``rs_s_all``) inside
    ``hb_dev`` so they are computed at most once per run.

    Parameters
    ----------
    hb_dev : dict
        Dict of CuPy arrays from :func:`upload_hb_index`.
    cutoff : float
        Absolute value cutoff (same as passed to :func:`build_g_base`).
    norb : int
        Number of active orbitals.
    cp : module
        CuPy module.

    Returns
    -------
    g_base : cp.ndarray, shape (norb^2,)
        Fixed part of g_flat (same for all source CSFs at this cutoff level).
    """
    nops = norb * norb
    g_base = cp.zeros(nops, dtype=cp.float64)

    # --- One-body ---
    h1_abs = hb_dev["h1_abs"]  # float64 [n_h1], sorted descending
    if h1_abs.size > 0 and float(h1_abs[0]) >= cutoff:
        k_end = int(cp.searchsorted(-h1_abs, -cutoff))
        if k_end > 0:
            if "h1_pq_flat" not in hb_dev:
                h1_pq = hb_dev["h1_pq"]  # int32 (n_h1, 2)
                hb_dev["h1_pq_flat"] = (
                    h1_pq[:, 0].astype(cp.int64) * norb + h1_pq[:, 1].astype(cp.int64)
                )
            cp.add.at(g_base, hb_dev["h1_pq_flat"][:k_end], hb_dev["h1_signed"][:k_end])

    # --- Two-body off-diagonal (r != s) ---
    v_abs = hb_dev["v_abs"]
    nnz = int(v_abs.size)
    if nnz > 0 and float(hb_dev["pq_max_v"].max()) >= cutoff:
        rs_all = hb_dev["rs_idx"]
        if "rs_r_all" not in hb_dev:
            hb_dev["rs_r_all"] = rs_all // norb
            hb_dev["rs_s_all"] = rs_all % norb
        r_all = hb_dev["rs_r_all"]
        s_all = hb_dev["rs_s_all"]
        offdiag_above = (v_abs >= cutoff) & (r_all != s_all)
        if bool(offdiag_above.any()):
            v_filtered = cp.where(offdiag_above, 0.5 * hb_dev["v_signed"], 0.0)
            starts = hb_dev["pq_ptr"][:-1].astype(cp.intp)
            g_base += cp.add.reduceat(v_filtered, starts)

    return g_base


def materialize_eri_4d_from_df_gpu(l_full_d: Any, norb: int, cp: Any) -> Any:
    """Materialize full 4D ERI on GPU from DF 3-index integrals.

    Parameters
    ----------
    l_full_d : cupy.ndarray, shape (norb^2, naux)
    norb : int
    cp : cupy module

    Returns
    -------
    cupy.ndarray, shape (norb, norb, norb, norb)
    """
    nops = norb * norb
    eri_2d = cp.dot(l_full_d, l_full_d.T)  # [nops, nops] via cuBLAS GEMM
    return eri_2d.reshape(norb, norb, norb, norb)
