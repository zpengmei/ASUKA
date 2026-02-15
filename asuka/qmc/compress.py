from __future__ import annotations

import numpy as np

from .sparse import coalesce_coo_i32_f64


def _choose_pivotal_d(abs_u: np.ndarray, *, m: int) -> int:
    """Choose deterministic keep count `d` for pivotal compression.

    The pivotal compression described in the Randomized Subspace Iteration (RSI)
    paper keeps a (data-dependent) deterministic set D of the largest-magnitude
    entries so that the remaining inclusion probabilities are all <= 1.

    Let abs_u be the absolute values sorted in descending order:
        abs_u[0] >= abs_u[1] >= ...
    and let total = sum(abs_u).

    For a candidate d, define the remainder sum:
        rem = total - sum_{i<d} abs_u[i]
        K = m - d
    The remaining inclusion probabilities are p_i = K * abs_u[i] / rem.
    The maximum p_i occurs at the first remaining entry abs_u[d]. The condition
    p_i <= 1 for all remaining entries is therefore:
        abs_u[d] <= rem / K.

    This routine returns the smallest d that satisfies this condition.
    """

    abs_u = np.asarray(abs_u, dtype=np.float64).ravel()
    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    L = int(abs_u.size)
    if L == 0 or m == 0:
        return 0

    total = float(np.sum(abs_u))
    if total == 0.0:
        return 0

    # Clamp: at most keep m entries deterministically.
    d = 0
    prefix = 0.0
    while d < L:
        K = m - d
        if K <= 0:
            return m
        rem = total - prefix
        if rem <= 0.0:
            return d
        if float(abs_u[d]) <= rem / float(K):
            return d
        prefix += float(abs_u[d])
        d += 1

    return d


def _pivotal_sample_from_inclusion_probs(pi: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Ordered pivotal sampling without replacement.

    Parameters
    ----------
    pi
        Inclusion probabilities (0<=pi<=1) with sum(pi) an integer K.
        Output selects exactly K indices.

    Returns
    -------
    sel
        Boolean mask of selected indices (True means included).

    Notes
    -----
    This is an implementation of ordered pivotal sampling (Deville & Tillé).
    It iteratively fuses pairs of fractional probabilities until all are 0/1.
    """

    pi = np.asarray(pi, dtype=np.float64).ravel().copy()
    if pi.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if np.any(pi < -1e-15) or np.any(pi > 1.0 + 1e-15):
        raise ValueError("pi must satisfy 0<=pi<=1")

    # Clamp tiny numerical noise.
    pi = np.clip(pi, 0.0, 1.0)
    K = float(np.sum(pi))
    K_round = int(np.rint(K))
    if abs(K - float(K_round)) > 1e-9:
        raise ValueError("sum(pi) must be an integer for pivotal sampling")

    n = int(pi.size)
    p = pi
    sel = np.zeros(n, dtype=np.bool_)

    def is_frac(x: float) -> bool:
        return (x > 1e-15) and (x < 1.0 - 1e-15)

    # Find first fractional position.
    i = 0
    while i < n and not is_frac(float(p[i])):
        if float(p[i]) >= 1.0 - 1e-15:
            sel[i] = True
        i += 1

    j = i + 1
    while i < n and j < n:
        while j < n and not is_frac(float(p[j])):
            if float(p[j]) >= 1.0 - 1e-15:
                sel[j] = True
            j += 1
        if j >= n:
            break

        a = float(p[i])
        b = float(p[j])
        s = a + b

        if s <= 1.0 + 1e-15:
            # Case a+b <= 1: one becomes 0.
            if s <= 1e-15:
                p[i] = 0.0
                p[j] = 0.0
            else:
                prob_i = a / s
                if float(rng.random()) < prob_i:
                    p[i] = s
                    p[j] = 0.0
                else:
                    p[j] = s
                    p[i] = 0.0
        else:
            # Case a+b > 1: one becomes 1.
            denom = 2.0 - s
            if denom <= 1e-15:
                p[i] = 1.0
                p[j] = 1.0
            else:
                prob_i_one = (1.0 - b) / denom
                if float(rng.random()) < prob_i_one:
                    p[i] = 1.0
                    p[j] = s - 1.0
                else:
                    p[j] = 1.0
                    p[i] = s - 1.0

        # Finalize any integer outcomes at i and j.
        if float(p[i]) >= 1.0 - 1e-15:
            sel[i] = True
            p[i] = 1.0
        elif float(p[i]) <= 1e-15:
            p[i] = 0.0

        if float(p[j]) >= 1.0 - 1e-15:
            sel[j] = True
            p[j] = 1.0
        elif float(p[j]) <= 1e-15:
            p[j] = 0.0

        # Advance i to next fractional.
        if not is_frac(float(p[i])):
            i += 1
            while i < n and not is_frac(float(p[i])):
                if float(p[i]) >= 1.0 - 1e-15:
                    sel[i] = True
                i += 1
            j = i + 1
        else:
            j += 1

    # Finalize any remaining entries.
    for t in range(n):
        if float(p[t]) >= 1.0 - 1e-15:
            sel[t] = True

    # Safety: enforce exact K selections if numerical noise produced off-by-one.
    want = K_round
    got = int(np.count_nonzero(sel))
    if got != want:
        frac = np.abs(pi - 0.5)
        order = np.argsort(frac)
        if got > want:
            drop = [int(ii) for ii in order if sel[int(ii)]]
            for i_drop in drop[: got - want]:
                sel[i_drop] = False
        else:
            add = [int(ii) for ii in order if not sel[int(ii)]]
            for i_add in add[: want - got]:
                sel[i_add] = True

    return sel


def compress_phi_pivotal(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """RSI-style pivotal compression Φ to a budget `m` nonzeros.

    Matches the pivotal compression described in the RSI paper.
    """

    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")

    idx_u, val_u = coalesce_coo_i32_f64(idx, val)
    L = int(idx_u.size)
    if m == 0 or L == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    if L <= m:
        return idx_u, val_u

    abs_u = np.abs(val_u)
    order = np.argsort(abs_u)[::-1]
    abs_s = np.asarray(abs_u[order], dtype=np.float64, order="C")

    d = _choose_pivotal_d(abs_s, m=m)
    d = int(min(d, m, L))
    K = int(m - d)

    det_pos = order[:d]
    idx_det = np.asarray(idx_u[det_pos], dtype=np.int32, order="C")
    val_det = np.asarray(val_u[det_pos], dtype=np.float64, order="C")

    if K <= 0:
        return coalesce_coo_i32_f64(idx_det, val_det)

    rem_pos = order[d:]
    if rem_pos.size == 0:
        return coalesce_coo_i32_f64(idx_det, val_det)

    idx_rem = np.asarray(idx_u[rem_pos], dtype=np.int32, order="C")
    val_rem = np.asarray(val_u[rem_pos], dtype=np.float64, order="C")
    abs_rem = np.abs(val_rem)
    W = float(np.sum(abs_rem))
    if W == 0.0:
        return coalesce_coo_i32_f64(idx_det, val_det)

    # Inclusion probabilities for the remainder.
    pi = float(K) * (abs_rem / W)
    pi = np.clip(pi, 0.0, 1.0)
    s = float(np.sum(pi))
    if s <= 0.0:
        return coalesce_coo_i32_f64(idx_det, val_det)
    pi *= float(K) / s
    pi = np.clip(pi, 0.0, 1.0)

    sel = _pivotal_sample_from_inclusion_probs(pi, rng=rng)
    if int(np.count_nonzero(sel)) == 0:
        return coalesce_coo_i32_f64(idx_det, val_det)

    idx_samp = np.asarray(idx_rem[sel], dtype=np.int32, order="C")
    val_samp = np.asarray(val_rem[sel] / pi[sel], dtype=np.float64, order="C")

    idx_out = np.concatenate((idx_det, idx_samp))
    val_out = np.concatenate((val_det, val_samp))
    return coalesce_coo_i32_f64(idx_out, val_out)


def compress_phi_pivot_resample(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    m: int,
    pivot: int = 256,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """FRI-style stochastic compression Φ to a budget `m` nonzeros.

    Implementation: keep top-|v| pivots deterministically, resample the remainder
    via systematic resampling (with replacement) with weights ∝ |v|.

    This is unbiased in the sense that E[Φ(v)] = v (up to floating-point roundoff).
    """

    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    pivot = int(pivot)
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    idx_u, val_u = coalesce_coo_i32_f64(idx, val)
    L = int(idx_u.size)
    if m == 0 or L == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    if L <= m:
        return idx_u, val_u

    abs_u = np.abs(val_u)
    p = min(pivot, m, L)
    if p <= 0:
        piv_mask = np.zeros(L, dtype=np.bool_)
        idx_p = np.zeros(0, dtype=np.int32)
        val_p = np.zeros(0, dtype=np.float64)
    else:
        piv_pos = np.argpartition(abs_u, -p)[-p:]
        piv_mask = np.zeros(L, dtype=np.bool_)
        piv_mask[piv_pos] = True
        idx_p = np.asarray(idx_u[piv_pos], dtype=np.int32, order="C")
        val_p = np.asarray(val_u[piv_pos], dtype=np.float64, order="C")

    K = m - p
    idx_r = idx_u[~piv_mask]
    val_r = val_u[~piv_mask]
    abs_r = np.abs(val_r)
    W = float(np.sum(abs_r))
    if K <= 0 or W == 0.0 or idx_r.size == 0:
        return coalesce_coo_i32_f64(idx_p, val_p)

    step = W / float(K)
    u0 = float(rng.random()) * step
    u = u0 + step * np.arange(K, dtype=np.float64)
    cdf = np.cumsum(abs_r, dtype=np.float64)
    pos = np.searchsorted(cdf, u, side="left")
    pos = np.asarray(pos, dtype=np.int64)
    pos = np.clip(pos, 0, idx_r.size - 1)

    samp_idx = np.asarray(idx_r[pos], dtype=np.int32, order="C")
    samp_val = np.sign(val_r[pos]).astype(np.float64, copy=False) * step

    idx_out = np.concatenate((idx_p, samp_idx))
    val_out = np.concatenate((val_p, samp_val))
    return coalesce_coo_i32_f64(idx_out, val_out)

