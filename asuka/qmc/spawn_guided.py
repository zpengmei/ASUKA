from __future__ import annotations

"""Guided spawning utilities for FRI/FCIQMC-style projector steps.

This module provides a *row-oracle based* spawn engine that can be biased by a
guiding distribution ``q(i)``。

Why a row-oracle spawn path?
---------------------------
The high-performance spawn engine in :mod:`asuka.qmc.spawn` samples children
through sequential GUGA generator actions (E_pq, etc.). That path is the one we
want for production, but it is difficult to incorporate an external guide
distribution without expensive per-proposal NN evaluations.

For *reference-oriented* integration (and for small CI spaces), it is useful to
have a spawn engine that:

* enumerates the connected row ``{(i, H_ij)}`` via :func:`asuka.cuguga.oracle.sparse.connected_row_sparse`;
* samples spawned targets ``i`` from a guide-aware proposal distribution;
* remains **unbiased**, i.e. the expected spawn contributions equal ``-eps * H * x``.

The routines here are intended for testing, validation, and early hybrid
experimentation. They are not optimized for large CI spaces.
"""

from collections.abc import Callable

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache
from asuka.cuguga.oracle.sparse import connected_row_sparse
from .spawn import spawn_hamiltonian_events
from .sparse import coalesce_coo_i32_f64


def spawn_hamiltonian_events_guided_row(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
    logq_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    alpha: float = 0.0,
    q_floor: float = 1e-12,
    max_out: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Spawn Hamiltonian events using a row oracle and (optional) guide bias.

    Returns COO arrays ``(evt_idx, evt_val)`` representing an unbiased estimator
    of ``-eps * H * x``.

    Parameters
    ----------
    drt, h1e, eri
        Hamiltonian definition.
    x_idx, x_val
        Sparse vector ``x`` (COO). Duplicates are allowed; they are coalesced.
    eps
        Projector timestep.
    nspawn_one, nspawn_two
        Kept for API compatibility with :func:`asuka.qmc.spawn.spawn_hamiltonian_events`.
        Here we simply use ``K = nspawn_one + nspawn_two`` spawn samples per parent.
    initiator_t
        If > 0, parents with ``|x_j| < initiator_t`` may only spawn to
        configurations already present in the current support of ``x``.
    logq_fn
        Function returning ``log q(i)`` for a batch of CSF indices. Required if
        ``alpha != 0``.
    alpha
        Interpolation parameter controlling how strongly the guide affects the
        proposal distribution.

        For a connected row ``(i, H_ij)``, we define weights

        ``w_i = |H_ij|^(1-alpha) * q_i^alpha``.

        Special cases:
        * ``alpha = 0`` -> weights proportional to ``|H_ij|``.
        * ``alpha = 1`` -> weights proportional to ``q_i``.

    Notes
    -----
    Unbiasedness is achieved by sampling from the proposal and reweighting each
    spawned contribution by ``1/p(i|j)`` (with an overall ``1/K`` factor).
    Systematic resampling is used to reduce variance.
    """

    eps = float(eps)
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha != 0.0 and logq_fn is None:
        raise ValueError("logq_fn must be provided when alpha != 0")
    q_floor = float(q_floor)
    if q_floor <= 0.0 or not np.isfinite(q_floor):
        raise ValueError("q_floor must be a finite positive number")

    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one and nspawn_two must be >= 0")
    K = int(nspawn_one + nspawn_two)
    if K <= 0:
        raise ValueError("at least one of nspawn_one or nspawn_two must be > 0")

    # Coalesce input (safety): we assume sorted unique downstream for initiator gating.
    x_idx_u, x_val_u = coalesce_coo_i32_f64(x_idx, x_val)
    m = int(x_idx_u.size)
    if m == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    t = float(initiator_t)

    # Preallocate worst-case output.
    evt_idx = np.empty(m * K, dtype=np.int32)
    evt_val = np.empty(m * K, dtype=np.float64)
    out = 0

    for j, xj in zip(x_idx_u.tolist(), x_val_u.tolist()):
        if xj == 0.0:
            continue

        allow_new = True
        if t > 0.0 and abs(xj) < t:
            allow_new = False

        i_idx, hij = connected_row_sparse(
            drt,
            h1e,
            eri,
            int(j),
            max_out=int(max_out),
            state_cache=state_cache,
        )
        if i_idx.size == 0:
            continue

        i_idx = np.asarray(i_idx, dtype=np.int32, order="C")
        hij = np.asarray(hij, dtype=np.float64, order="C")
        if i_idx.size != hij.size:
            raise RuntimeError("row oracle returned mismatched i_idx/hij sizes")

        # Initiator gating: restrict to current support.
        if not allow_new:
            pos = np.searchsorted(x_idx_u, i_idx)
            keep = (pos < m) & (x_idx_u[pos] == i_idx)
            if not np.any(keep):
                continue
            i_idx = i_idx[keep]
            hij = hij[keep]

        abs_h = np.abs(hij)
        if alpha == 0.0:
            w = abs_h
        else:
            logq = np.asarray(logq_fn(i_idx.astype(np.int64, copy=False)), dtype=np.float64).ravel()
            if logq.shape != (int(i_idx.size),):
                raise ValueError("logq_fn returned wrong shape")
            q = np.exp(logq)
            q = np.maximum(q, q_floor)
            w = np.power(abs_h, 1.0 - alpha) * np.power(q, alpha)

        # Drop zero-weight entries for robustness.
        keep_w = w > 0.0
        if not np.any(keep_w):
            continue
        if keep_w.size != 0 and np.any(~keep_w):
            i_idx = i_idx[keep_w]
            hij = hij[keep_w]
            w = w[keep_w]

        W = float(np.sum(w))
        if W <= 0.0 or not np.isfinite(W):
            continue

        step = W / float(K)

        # Systematic resampling.
        u0 = float(rng.random()) * step
        u = u0 + step * np.arange(K, dtype=np.float64)
        cdf = np.cumsum(w, dtype=np.float64)
        pos = np.searchsorted(cdf, u, side="left")
        pos = np.asarray(pos, dtype=np.int64)
        pos = np.clip(pos, 0, i_idx.size - 1)

        samp_idx = i_idx[pos]
        samp_hij = hij[pos]
        w_pos = w[pos]

        # Unbiased reweighting: Hij / p(i|j) with 1/K factor folded into `step`.
        # Contribution per spawned sample:
        #   (-eps) * xj * Hij * (W/K) / w_i
        evt_idx[out : out + K] = samp_idx
        evt_val[out : out + K] = (-eps) * float(xj) * samp_hij * (step / w_pos)
        out += K

    return evt_idx[:out], evt_val[:out]


def spawn_hamiltonian_events_guided_thinning(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
    logq_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    alpha: float = 0.0,
    q_scale: float = 1.0,
    min_accept: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Spawn events using the production spawner, then apply guide-based thinning.

    This is a reference-oriented guided spawner that avoids querying ``q(i)`` for
    the full connected row. Instead, we:
      1) propose events using :func:`asuka.qmc.spawn.spawn_hamiltonian_events`,
      2) accept each proposed event ``i`` with probability ``a(i)``,
      3) reweight accepted contributions by ``1/a(i)`` to remain unbiased.

    The thinning acceptance is:
        a(i) = clip((q(i) * q_scale)**alpha, [min_accept, 1]).

    Unbiasedness follows from:
        E[ 1{accept} * val / a ] = val.
    """

    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha != 0.0 and logq_fn is None:
        raise ValueError("logq_fn must be provided when alpha != 0")

    q_scale = float(q_scale)
    if q_scale <= 0.0 or not np.isfinite(q_scale):
        raise ValueError("q_scale must be a finite positive number")

    min_accept = float(min_accept)
    if min_accept < 0.0 or min_accept > 1.0 or not np.isfinite(min_accept):
        raise ValueError("min_accept must be a finite number in [0, 1]")

    evt_idx, evt_val = spawn_hamiltonian_events(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        eps=float(eps),
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        rng=rng,
        initiator_t=float(initiator_t),
        state_cache=state_cache,
    )
    if evt_idx.size == 0:
        return evt_idx, evt_val

    if alpha == 0.0:
        return evt_idx, evt_val

    ind = evt_idx.astype(np.int64, copy=False)
    logq = np.asarray(logq_fn(ind), dtype=np.float64).ravel()
    if logq.shape != (int(evt_idx.size),):
        raise ValueError("logq_fn returned wrong shape")

    # a(i) = clip((q(i) * q_scale)**alpha, [min_accept, 1])
    loga = alpha * (logq + np.log(q_scale))
    if min_accept > 0.0:
        loga = np.maximum(loga, np.log(min_accept))
    loga = np.minimum(loga, 0.0)
    a = np.exp(loga)

    u = rng.random(size=int(a.size))
    accept = u < a
    if not np.any(accept):
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    evt_idx = np.asarray(evt_idx[accept], dtype=np.int32, order="C")
    evt_val = np.asarray(evt_val[accept], dtype=np.float64, order="C")
    a_acc = np.asarray(a[accept], dtype=np.float64, order="C")

    # Unbiased reweighting.
    evt_val = evt_val / a_acc
    return evt_idx, evt_val
