"""Streaming / semistochastic PT2 energy correction.

Provides memory-bounded alternatives to the exact external-space PT2
accumulation in ``sparse_support._accumulate_and_score_external_sparse``.

Reference: Sharma, Holmes, Jeanmairet, Alavi, Umrigar, JCTC 2017, 13, 1595.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache
from asuka.sci.sparse_support import ConnectedRowCache, DiagonalGuessLookup, _connected_row_cached


@dataclass(frozen=True)
class StreamingPT2Result:
    """Result container for streaming / semistochastic PT2."""

    e_pt2: np.ndarray  # (nroots,) total PT2
    e_pt2_det: np.ndarray  # (nroots,) deterministic part
    e_pt2_stoch: np.ndarray  # (nroots,) stochastic part (zeros if exact)
    e_pt2_error: np.ndarray  # (nroots,) estimated error (zeros if exact)
    n_external_det: int
    n_external_stoch: int
    n_batches: int
    wall_time_s: float


def streaming_pt2_deterministic(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    selected_set: set[int],
    c_sel: np.ndarray,
    e_var: np.ndarray,
    hdiag_lookup: DiagonalGuessLookup,
    denom_floor: float = 1e-12,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    state_cache: DRTStateCache | None = None,
    row_cache: ConnectedRowCache | None = None,
    bucket_size: int = 500_000,
) -> StreamingPT2Result:
    """Exact streaming PT2 — same result as full accumulation, bounded memory.

    Divides the external CSF space ``[0, ncsf)`` into buckets of
    ``bucket_size``.  For each bucket, accumulates numerators from all
    sources, computes PT2 contributions, and discards the bucket.

    Parameters
    ----------
    drt, h1e, eri : DRT, integrals
        Same as for ``heat_bath_select_and_pt2_sparse``.
    sel : sequence of int
        Selected CSF indices.
    selected_set : set of int
        Set view of ``sel`` for O(1) membership.
    c_sel : ndarray, shape (nsel, nroots)
        CI coefficients for the selected space.
    e_var : ndarray, shape (nroots,)
        Variational energies.
    hdiag_lookup : DiagonalGuessLookup
        Diagonal Hamiltonian element lookup.
    denom_floor : float
        Minimum denominator magnitude.
    max_out : int
        Max connected CSFs per oracle call.
    screening, state_cache, row_cache : optional
        Oracle acceleration structures.
    bucket_size : int
        Number of external CSFs per bucket.

    Returns
    -------
    StreamingPT2Result
    """
    t0 = time.perf_counter()
    ncsf = int(drt.ncsf)
    nroots = int(e_var.size)
    c_sel = np.asarray(c_sel, dtype=np.float64)
    e_var = np.asarray(e_var, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    sel_list = [int(s) for s in sel]
    denom_floor = float(denom_floor)
    bucket_size = max(1, int(bucket_size))

    # Pre-compute all connected rows (populates row_cache if provided)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for col, j in enumerate(sel_list):
        cj = c_sel[col, :]
        max_cj = float(np.max(np.abs(cj)))
        if max_cj == 0.0:
            rows.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), cj))
            continue
        i_idx, hij = _connected_row_cached(
            drt, h1e, eri, int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )
        rows.append((np.asarray(i_idx, dtype=np.int64), np.asarray(hij, dtype=np.float64), cj))

    # Bucket-streaming PT2
    e_pt2_total = np.zeros(nroots, dtype=np.float64)
    n_external = 0
    n_batches = 0

    for lo in range(0, ncsf, bucket_size):
        hi = min(lo + bucket_size, ncsf)
        n_batches += 1

        # Accumulate numerators for this bucket
        # Use a dict for sparse accumulation (most buckets are sparse)
        bucket: dict[int, np.ndarray] = {}
        for i_idx, hij, cj in rows:
            if i_idx.size == 0:
                continue
            for ii_raw, vv_raw in zip(i_idx, hij):
                ii = int(ii_raw)
                if ii < lo or ii >= hi or ii in selected_set:
                    continue
                vv = float(vv_raw)
                acc = bucket.get(ii)
                if acc is None:
                    bucket[ii] = vv * cj
                else:
                    acc += vv * cj

        # Score PT2 for this bucket
        for ii, p in bucket.items():
            hdiag_i = float(hdiag_lookup.get(ii))
            denom = e_var - hdiag_i
            if denom_floor > 0.0:
                small = np.abs(denom) < denom_floor
                if np.any(small):
                    denom = denom.copy()
                    denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)
            e_pt2_total += (p * p) / denom
        n_external += len(bucket)

    wall_time = time.perf_counter() - t0
    return StreamingPT2Result(
        e_pt2=e_pt2_total,
        e_pt2_det=e_pt2_total.copy(),
        e_pt2_stoch=np.zeros(nroots, dtype=np.float64),
        e_pt2_error=np.zeros(nroots, dtype=np.float64),
        n_external_det=n_external,
        n_external_stoch=0,
        n_batches=n_batches,
        wall_time_s=wall_time,
    )


def semistochastic_pt2(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    selected_set: set[int],
    c_sel: np.ndarray,
    e_var: np.ndarray,
    hdiag_lookup: DiagonalGuessLookup,
    denom_floor: float = 1e-12,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    state_cache: DRTStateCache | None = None,
    row_cache: ConnectedRowCache | None = None,
    bucket_size: int = 500_000,
    n_det_sources: int | None = None,
    n_stoch_samples: int = 1000,
    n_stoch_batches: int = 10,
    seed: int | None = None,
) -> StreamingPT2Result:
    """SHCI-style semistochastic PT2 correction.

    Splits sources into deterministic (largest-|c_j|) and stochastic sets.
    The deterministic contribution is computed exactly; the stochastic part
    is estimated via importance sampling with error bars.

    Parameters
    ----------
    n_det_sources : int or None
        Number of sources for exact treatment.  Default:
        ``min(nsel, max(100, nsel // 10))``.
    n_stoch_samples : int
        Number of stochastic source samples per batch.
    n_stoch_batches : int
        Number of independent batches for error estimation.
    seed : int or None
        RNG seed for reproducibility.
    """
    t0 = time.perf_counter()
    ncsf = int(drt.ncsf)
    nroots = int(e_var.size)
    c_sel = np.asarray(c_sel, dtype=np.float64)
    e_var = np.asarray(e_var, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    sel_list = [int(s) for s in sel]
    nsel = len(sel_list)
    denom_floor = float(denom_floor)
    bucket_size = max(1, int(bucket_size))

    if n_det_sources is None:
        n_det_sources = min(nsel, max(100, nsel // 10))
    n_det_sources = min(int(n_det_sources), nsel)
    n_stoch_batches = max(1, int(n_stoch_batches))
    n_stoch_samples = max(1, int(n_stoch_samples))

    # Sort sources by max |c_j| descending
    max_c = np.array([float(np.max(np.abs(c_sel[col, :]))) for col in range(nsel)])
    order = np.argsort(-max_c)
    det_idx = order[:n_det_sources]
    stoch_idx = order[n_det_sources:]

    rng = np.random.default_rng(seed)

    # Pre-compute connected rows for deterministic sources
    def _get_row(col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        j = sel_list[col]
        cj = c_sel[col, :]
        if float(np.max(np.abs(cj))) == 0.0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), cj
        i_idx, hij = _connected_row_cached(
            drt, h1e, eri, int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            row_cache=row_cache,
        )
        return np.asarray(i_idx, dtype=np.int64), np.asarray(hij, dtype=np.float64), cj

    # Build importance sampling weights for stochastic sources
    stoch_weights = max_c[stoch_idx] ** 2
    if stoch_weights.sum() > 0:
        stoch_probs = stoch_weights / stoch_weights.sum()
    else:
        stoch_probs = np.ones(len(stoch_idx), dtype=np.float64) / max(1, len(stoch_idx))

    e_pt2_det_total = np.zeros(nroots, dtype=np.float64)
    n_external_det = 0
    n_external_stoch = 0

    # Deterministic pass: bucket-streaming over deterministic sources
    for lo in range(0, ncsf, bucket_size):
        hi = min(lo + bucket_size, ncsf)
        bucket: dict[int, np.ndarray] = {}
        for col in det_idx:
            i_idx, hij, cj = _get_row(int(col))
            if i_idx.size == 0:
                continue
            for ii_raw, vv_raw in zip(i_idx, hij):
                ii = int(ii_raw)
                if ii < lo or ii >= hi or ii in selected_set:
                    continue
                vv = float(vv_raw)
                acc = bucket.get(ii)
                if acc is None:
                    bucket[ii] = vv * cj
                else:
                    acc += vv * cj

        for ii, p in bucket.items():
            hdiag_i = float(hdiag_lookup.get(ii))
            denom = e_var - hdiag_i
            if denom_floor > 0.0:
                small = np.abs(denom) < denom_floor
                if np.any(small):
                    denom = denom.copy()
                    denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)
            e_pt2_det_total += (p * p) / denom
        n_external_det += len(bucket)

    # Stochastic pass
    deltas = np.zeros((n_stoch_batches, nroots), dtype=np.float64)
    if len(stoch_idx) > 0:
        for b in range(n_stoch_batches):
            # Sample sources with replacement
            sampled = rng.choice(len(stoch_idx), size=n_stoch_samples, replace=True, p=stoch_probs)
            # Importance weight: 1/(n_samples * p_k)
            for lo in range(0, ncsf, bucket_size):
                hi = min(lo + bucket_size, ncsf)
                # Deterministic numerators for this bucket (from det sources)
                num_D: dict[int, np.ndarray] = {}
                for col in det_idx:
                    i_idx, hij, cj = _get_row(int(col))
                    if i_idx.size == 0:
                        continue
                    for ii_raw, vv_raw in zip(i_idx, hij):
                        ii = int(ii_raw)
                        if ii < lo or ii >= hi or ii in selected_set:
                            continue
                        vv = float(vv_raw)
                        acc = num_D.get(ii)
                        if acc is None:
                            num_D[ii] = vv * cj
                        else:
                            acc += vv * cj

                # Stochastic numerators
                num_S: dict[int, np.ndarray] = {}
                for s_local in sampled:
                    col = int(stoch_idx[s_local])
                    w_is = 1.0 / (n_stoch_samples * stoch_probs[s_local])
                    i_idx, hij, cj = _get_row(col)
                    if i_idx.size == 0:
                        continue
                    for ii_raw, vv_raw in zip(i_idx, hij):
                        ii = int(ii_raw)
                        if ii < lo or ii >= hi or ii in selected_set:
                            continue
                        vv = float(vv_raw)
                        contrib = w_is * vv * cj
                        acc = num_S.get(ii)
                        if acc is None:
                            num_S[ii] = contrib.copy()
                        else:
                            acc += contrib

                # Compute stochastic correction: sum [2*num_D*num_S + num_S^2] / denom
                all_keys = set(num_S.keys())
                for ii in all_keys:
                    hdiag_i = float(hdiag_lookup.get(ii))
                    denom = e_var - hdiag_i
                    if denom_floor > 0.0:
                        small = np.abs(denom) < denom_floor
                        if np.any(small):
                            denom = denom.copy()
                            denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)
                    s = num_S[ii]
                    d = num_D.get(ii, np.zeros(nroots, dtype=np.float64))
                    deltas[b] += (2.0 * d * s + s * s) / denom
                    n_external_stoch += 1

    e_pt2_stoch = np.mean(deltas, axis=0) if n_stoch_batches > 0 else np.zeros(nroots, dtype=np.float64)
    e_pt2_error = (
        np.std(deltas, axis=0, ddof=1) / np.sqrt(n_stoch_batches)
        if n_stoch_batches > 1
        else np.zeros(nroots, dtype=np.float64)
    )
    e_pt2_total = e_pt2_det_total + e_pt2_stoch

    wall_time = time.perf_counter() - t0
    return StreamingPT2Result(
        e_pt2=e_pt2_total,
        e_pt2_det=e_pt2_det_total,
        e_pt2_stoch=e_pt2_stoch,
        e_pt2_error=e_pt2_error,
        n_external_det=n_external_det,
        n_external_stoch=n_external_stoch,
        n_batches=n_stoch_batches,
        wall_time_s=wall_time,
    )
