from __future__ import annotations

"""Guided / importance-sampled Φ compression utilities.

This module generalizes :func:`asuka.qmc.compress.compress_phi_pivot_resample` by
allowing a *guiding distribution* ``q(i)`` to influence the stochastic resampling step.

The resampling distribution
---------------------------
For the non-pivot remainder entries ``r_i`` (with values ``v_i``), we choose

.. math::

    w_i = |v_i|^{1-\alpha} \; q_i^{\alpha},\qquad p_i = w_i / \sum_j w_j

and sample ``K = m - pivot`` items **with replacement** from ``p`` using
systematic resampling. Each sampled contribution is reweighted to remain
unbiased:

.. math::

    \hat v_i = v_i \; \frac{\sum_j w_j}{K\, w_i}.

Special cases:
* ``alpha = 0`` recovers the standard FRI Φ compression (weights ∝ |v|).
* ``alpha = 1`` ignores |v| in the proposal (weights ∝ q), but is still unbiased.

In practice, ``alpha`` in ``[0, 1]`` interpolates between the two.
"""

from collections.abc import Callable

import numpy as np

from .sparse import coalesce_coo_i32_f64


def compress_phi_pivot_resample_guided(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    m: int,
    pivot: int = 256,
    rng: np.random.Generator,
    logq_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    alpha: float = 0.0,
    pivot_alpha: float = 0.0,
    q_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Guided Φ compression to a budget ``m`` nonzeros.

    Parameters
    ----------
    idx, val
        COO representation of a sparse vector (may contain duplicates).
    m
        Target sparsity (maximum number of nonzeros returned).
    pivot
        Number of largest-magnitude entries to keep deterministically.
    rng
        NumPy RNG for systematic resampling.
    logq_fn
        Callable returning ``log q(i)`` for a batch of CSF indices ``i``.
        Only required when ``alpha != 0`` or ``pivot_alpha != 0``.
    alpha
        Interpolation parameter controlling how strongly the guide affects the
        *stochastic resampling* step.
    pivot_alpha
        Interpolation parameter controlling how strongly the guide affects the
        *deterministic pivot* selection.
    q_floor
        Minimum value used when converting ``log q`` to ``q``.

    Returns
    -------
    (idx_out, val_out)
        Coalesced COO arrays (sorted unique indices).
    """

    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    pivot = int(pivot)
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    alpha = float(alpha)
    pivot_alpha = float(pivot_alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(pivot_alpha):
        raise ValueError("pivot_alpha must be finite")
    if alpha != 0.0 or pivot_alpha != 0.0:
        if logq_fn is None:
            raise ValueError("logq_fn must be provided when alpha or pivot_alpha are non-zero")

    q_floor = float(q_floor)
    if q_floor <= 0.0 or not np.isfinite(q_floor):
        raise ValueError("q_floor must be a finite positive number")

    idx_u, val_u = coalesce_coo_i32_f64(idx, val)
    L = int(idx_u.size)

    if m == 0 or L == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    if L <= m:
        return idx_u, val_u

    abs_u = np.abs(val_u)

    # --- deterministic pivots ---
    p = min(int(pivot), int(m), int(L))
    if p <= 0:
        piv_mask = np.zeros(L, dtype=np.bool_)
        idx_p = np.zeros(0, dtype=np.int32)
        val_p = np.zeros(0, dtype=np.float64)
    else:
        if pivot_alpha == 0.0:
            score = abs_u
        else:
            # score_i = |v_i|^(1-a) * q_i^a
            logq = np.asarray(logq_fn(idx_u.astype(np.int64, copy=False)), dtype=np.float64).ravel()
            if logq.shape != (L,):
                raise ValueError("logq_fn returned wrong shape")
            q = np.exp(logq)
            q = np.maximum(q, q_floor)
            score = np.power(abs_u, 1.0 - pivot_alpha) * np.power(q, pivot_alpha)

        piv_pos = np.argpartition(score, -p)[-p:]
        piv_mask = np.zeros(L, dtype=np.bool_)
        piv_mask[piv_pos] = True
        idx_p = np.asarray(idx_u[piv_pos], dtype=np.int32, order="C")
        val_p = np.asarray(val_u[piv_pos], dtype=np.float64, order="C")

    # --- resample remainder ---
    K = int(m) - int(p)
    idx_r = idx_u[~piv_mask]
    val_r = val_u[~piv_mask]
    if K <= 0 or idx_r.size == 0:
        return coalesce_coo_i32_f64(idx_p, val_p)

    abs_r = np.abs(val_r)

    if alpha == 0.0:
        w = abs_r
    else:
        logq_r = np.asarray(logq_fn(idx_r.astype(np.int64, copy=False)), dtype=np.float64).ravel()
        if logq_r.shape != (int(idx_r.size),):
            raise ValueError("logq_fn returned wrong shape")
        q_r = np.exp(logq_r)
        q_r = np.maximum(q_r, q_floor)
        # w_i = |v_i|^(1-a) * q_i^a
        w = np.power(abs_r, 1.0 - alpha) * np.power(q_r, alpha)

    W = float(np.sum(w))
    if W == 0.0 or not np.isfinite(W):
        # No stochastic part; just return pivots.
        return coalesce_coo_i32_f64(idx_p, val_p)

    # systematic resampling over weights w
    step = W / float(K)
    u0 = float(rng.random()) * step
    u = u0 + step * np.arange(K, dtype=np.float64)
    cdf = np.cumsum(w, dtype=np.float64)
    pos = np.searchsorted(cdf, u, side="left")
    pos = np.asarray(pos, dtype=np.int64)
    pos = np.clip(pos, 0, idx_r.size - 1)

    samp_idx = np.asarray(idx_r[pos], dtype=np.int32, order="C")
    w_pos = np.asarray(w[pos], dtype=np.float64, order="C")
    # Unbiased reweighting: v_i * (W/K) / w_i = v_i * step / w_i
    samp_val = np.asarray(val_r[pos], dtype=np.float64, order="C") * (step / w_pos)

    idx_out = np.concatenate((idx_p, samp_idx))
    val_out = np.concatenate((val_p, samp_val))
    return coalesce_coo_i32_f64(idx_out, val_out)
