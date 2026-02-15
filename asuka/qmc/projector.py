from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache
from asuka.cuguga.oracle.sparse import connected_row_sparse
from .compress import compress_phi_pivot_resample
from .spawn import spawn_hamiltonian_events
from .sparse import coalesce_coo_i32_f64


def initiator_threshold(*, l1_norm: float, m: int, na: float) -> float:
    """Column-scaled initiator threshold: t = na * ||x||_1 / (m - 1)."""

    na = float(na)
    if na <= 0.0:
        return 0.0
    m = int(m)
    if m <= 1:
        raise ValueError("m must be > 1 when initiator_na > 0")
    return na * float(l1_norm) / float(m - 1)


def projector_step(
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
    m: int | None = None,
    pivot: int = 256,
    initiator_na: float = 0.0,
    scale_identity: float = 1.0,
    state_cache: DRTStateCache | None = None,
    backend: str = "stochastic",
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner_kwargs: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one stochastic projector step `y ≈ (scale_identity*I - eps * H) x` (CPU reference path).

    If `m` is not None, returns `Φ(y)` compressed to exactly `m` nonzeros (unless y has fewer).
    """

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")
    scale_identity = float(scale_identity)

    backend = str(backend).lower()
    if backend not in ("stochastic", "exact"):
        raise ValueError("backend must be 'stochastic' or 'exact'")

    if backend == "exact" and (spawner is not None or spawner_kwargs is not None):
        raise ValueError("spawner/spawner_kwargs are only used in backend='stochastic'")

    if backend == "exact":
        if float(initiator_na) != 0.0:
            raise ValueError("initiator_na is not supported in backend='exact' (set it to 0)")

        idx_chunks: list[np.ndarray] = [x_idx_i32]
        val_chunks: list[np.ndarray] = [float(scale_identity) * x_val_f64]
        for j, xj in zip(x_idx_i32.tolist(), x_val_f64.tolist()):
            if xj == 0.0:
                continue
            i_idx, hij = connected_row_sparse(
                drt,
                h1e,
                eri,
                int(j),
                max_out=200_000,
                state_cache=state_cache,
            )
            if i_idx.size == 0:
                continue
            idx_chunks.append(np.asarray(i_idx, dtype=np.int32, order="C"))
            val_chunks.append((-float(eps) * float(xj)) * np.asarray(hij, dtype=np.float64, order="C"))

        idx_all = np.concatenate(idx_chunks)
        val_all = np.concatenate(val_chunks)
    else:
        if m is None:
            initiator_t = 0.0
        else:
            initiator_t = initiator_threshold(l1_norm=float(np.sum(np.abs(x_val_f64))), m=int(m), na=float(initiator_na))

        # Allow swapping the spawn engine
        if spawner is None:
            spawner = spawn_hamiltonian_events
        if spawner_kwargs is None:
            spawner_kwargs = {}

        evt_i, evt_v = spawner(
            drt,
            h1e,
            eri,
            x_idx_i32,
            x_val_f64,
            eps=float(eps),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            rng=rng,
            initiator_t=float(initiator_t),
            state_cache=state_cache,
            **dict(spawner_kwargs),
        )

        if evt_i.size:
            idx_all = np.concatenate((x_idx_i32, evt_i))
            val_all = np.concatenate((float(scale_identity) * x_val_f64, evt_v))
        else:
            idx_all = x_idx_i32
            val_all = float(scale_identity) * x_val_f64

    if m is None:
        return coalesce_coo_i32_f64(idx_all, val_all)

    if compressor is None:
        return compress_phi_pivot_resample(idx_all, val_all, m=int(m), pivot=int(pivot), rng=rng)

    return compressor(idx_all, val_all, m=int(m), pivot=int(pivot), rng=rng)
