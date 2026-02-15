from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _EPQ_ACTION_CACHE, _get_epq_action_cache


@dataclass(frozen=True)
class DRTStateCache:
    """Cached CSF states for a fixed DRT.

    Attributes
    ----------
    steps
        Array of shape ``(ncsf, norb)`` (dtype int8) storing the step-vector
        encoding (E,U,L,D) as integers 0..3.
    nodes
        Array of shape ``(ncsf, norb+1)`` (dtype int32) storing DRT node ids
        along each CSF path (including the root at position 0).

    Notes
    -----
    This cache is intentionally limited to *per-CSF* data. It does **not**
    precompute any global ``E_pq`` actions (CSR matrices). Internally it reuses
    :func:`asuka.cuguga.oracle._get_epq_action_cache` to avoid duplicating
    the step/node tables.
    """

    steps: np.ndarray
    nodes: np.ndarray


_STATE_CACHE: weakref.WeakKeyDictionary[DRT, DRTStateCache] = weakref.WeakKeyDictionary()


def get_state_cache(drt: DRT) -> DRTStateCache:
    """Return a cached (steps,nodes) table for this DRT."""

    cached = _STATE_CACHE.get(drt)
    if cached is not None:
        return cached

    epq_cache = _get_epq_action_cache(drt)
    out = DRTStateCache(steps=epq_cache.steps, nodes=epq_cache.nodes)
    _STATE_CACHE[drt] = out
    return out


def clear_state_cache(drt: DRT | None = None) -> None:
    """Clear cached CSF state tables.

    If ``drt`` is None, clears all cached state tables for all DRTs.

    Note: because :func:`get_state_cache` reuses the E_pq action cache's
    underlying ``steps/nodes`` tables, clearing this cache also clears those
    tables (and any associated CSR matrices) to actually release memory.
    """

    if drt is None:
        _STATE_CACHE.clear()
        _EPQ_ACTION_CACHE.clear()
        return

    _STATE_CACHE.pop(drt, None)
    _EPQ_ACTION_CACHE.pop(drt, None)
