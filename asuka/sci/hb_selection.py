"""Heat-bath selected-CI helpers.

The production path is the scalable sparse selector implemented in
``heat_bath_select_and_pt2_sparse``. Legacy dense/frontier-hash CUDA variants
have been removed from the supported path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache
from asuka.sci.hb_integrals import HeatBathIntegralIndex
from asuka.sci.selected_ci import ConnectedRowCache, DiagonalGuessLookup, _select_external_sparse


def heat_bath_select_and_pt2_sparse(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel_idx: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    max_add: int,
    epsilon: float,
    denom_floor: float,
    hdiag_lookup: DiagonalGuessLookup,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    row_cache: ConnectedRowCache | None = None,
) -> tuple[list[int], np.ndarray]:
    """Scalable heat-bath-style selection using exact matrix-element screening."""

    eps = float(epsilon)
    if eps < 0.0:
        raise ValueError("epsilon must be >= 0")
    return _select_external_sparse(
        drt,
        np.asarray(h1e, dtype=np.float64),
        eri,
        sel=np.asarray(sel_idx, dtype=np.int64).ravel().tolist(),
        selected_set={int(ii) for ii in np.asarray(sel_idx, dtype=np.int64).ravel().tolist()},
        c_sel=np.asarray(c_sel, dtype=np.float64),
        e_var=np.asarray(e_var, dtype=np.float64),
        hdiag_lookup=hdiag_lookup,
        max_add=int(max_add),
        select_threshold=None,
        denom_floor=float(denom_floor),
        max_out=int(max_out),
        screening=screening,
        state_cache=state_cache,
        select_screen_contrib=float(eps),
        row_cache=row_cache,
    )


def _python_build_screened_g_flat(
    hb_index: HeatBathIntegralIndex,
    occ: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Build the legacy collapsed screened ``g_flat`` helper for debugging/tests."""

    norb = hb_index.norb
    nops = norb * norb
    g_flat = np.zeros((norb, norb), dtype=np.float64)

    for k in range(hb_index.n_h1):
        if hb_index.h1_abs[k] < cutoff:
            break
        p, q = int(hb_index.h1_pq[k, 0]), int(hb_index.h1_pq[k, 1])
        g_flat[p, q] = hb_index.h1_signed[k]

    for pq in range(nops):
        if hb_index.pq_max_v[pq] < cutoff:
            continue
        p, q = pq // norb, pq % norb
        lo = int(hb_index.pq_ptr[pq])
        hi = int(hb_index.pq_ptr[pq + 1])
        if lo >= hi:
            continue

        for k in range(lo, hi):
            if hb_index.v_abs[k] < cutoff:
                break
            rs = int(hb_index.rs_idx[k])
            r, s = rs // norb, rs % norb
            v = hb_index.v_signed[k]
            if r == s:
                g_flat[p, q] += 0.5 * float(occ[r]) * v
            else:
                g_flat[p, q] += 0.5 * v

    return g_flat


def heat_bath_select_and_pt2(*args, **kwargs) -> tuple[list[int], np.ndarray]:
    """Legacy HB selector entrypoint removed from the supported path."""

    _ = (args, kwargs)
    raise NotImplementedError(
        "heat_bath_select_and_pt2 has been removed from the supported path; use "
        "heat_bath_select_and_pt2_sparse or run_cipsi_trials(..., selection_mode='heat_bath') instead"
    )


def adaptive_epsilon(
    iteration: int,
    nsel: int,
    nsel_target: int,
    eps_init: float = 1e-3,
    eps_final: float = 1e-6,
) -> float:
    """Compute adaptive epsilon for heat-bath screening."""

    _ = int(iteration)
    if nsel_target <= 0:
        return float(eps_final)
    frac = min(1.0, float(nsel) / float(nsel_target))
    if eps_init <= 0 or eps_final <= 0 or eps_init <= eps_final:
        return float(eps_final)
    return float(eps_init) * (float(eps_final) / float(eps_init)) ** frac


def semistochastic_pt2(*args, **kwargs):
    """Placeholder for the removed legacy semistochastic PT2 helper."""

    _ = (args, kwargs)
    raise NotImplementedError(
        "semistochastic_pt2 is not implemented; use heat_bath_select_and_pt2_sparse(..., max_add=0) "
        "or run_cipsi_trials(..., selection_mode='heat_bath') for deterministic PT2 evaluation"
    )
