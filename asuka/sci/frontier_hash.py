from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache
from asuka.cuguga.screening import RowScreening
from asuka.sci.selected_ci import DiagonalGuessLookup, _select_external_sparse


@dataclass
class FrontierHashStats:
    hash_cap: int
    nnz_out: int
    overflow_retries: int
    timings_ms: dict[str, float]
    memory: dict[str, Any]


class SparseFrontierSelector:
    """Scalable exact selector using sparse row-oracle accumulation.

    This is the only supported frontier selector for the public CIPSI driver.
    """

    def __init__(
        self,
        drt: DRT,
        h1e: np.ndarray,
        eri: Any,
        *,
        hdiag_lookup: DiagonalGuessLookup,
        denom_floor: float,
        max_out: int,
        screening: RowScreening | None,
        state_cache: DRTStateCache | None,
        select_screen_contrib: float = 0.0,
    ) -> None:
        self.drt = drt
        self.h1e = np.asarray(h1e, dtype=np.float64)
        self.eri = eri
        self.hdiag_lookup = hdiag_lookup
        self.denom_floor = float(denom_floor)
        self.max_out = int(max_out)
        self.screening = screening
        self.state_cache = state_cache
        self.select_screen_contrib = float(select_screen_contrib)
        self._selected: set[int] = set()

    def reset_selected_mask(self, sel_idx: np.ndarray) -> None:
        self._selected = {int(ii) for ii in np.asarray(sel_idx, dtype=np.int64).ravel().tolist()}

    def mark_selected(self, new_idx: np.ndarray | list[int]) -> None:
        for ii in np.asarray(new_idx, dtype=np.int64).ravel().tolist():
            self._selected.add(int(ii))

    def build_and_score(
        self,
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        e_var: np.ndarray,
        max_add: int,
        select_threshold: float | None = None,
        profile: bool = False,
    ) -> tuple[list[int], np.ndarray, FrontierHashStats]:
        _ = bool(profile)
        new_idx, e_pt2 = _select_external_sparse(
            self.drt,
            self.h1e,
            self.eri,
            sel=np.asarray(sel_idx, dtype=np.int64).ravel().tolist(),
            selected_set=set(self._selected),
            c_sel=np.asarray(c_sel, dtype=np.float64),
            e_var=np.asarray(e_var, dtype=np.float64),
            hdiag_lookup=self.hdiag_lookup,
            max_add=int(max_add),
            select_threshold=select_threshold,
            denom_floor=float(self.denom_floor),
            max_out=int(self.max_out),
            screening=self.screening,
            state_cache=self.state_cache,
            select_screen_contrib=float(self.select_screen_contrib),
        )
        stats = FrontierHashStats(
            hash_cap=0,
            nnz_out=int(len(new_idx)),
            overflow_retries=0,
            timings_ms={},
            memory={},
        )
        return new_idx, np.asarray(e_pt2, dtype=np.float64), stats


class FrontierHashSelector:
    """Legacy CUDA frontier-hash selector removed from the supported path."""

    def __init__(self, *args, **kwargs) -> None:
        _ = (args, kwargs)
        raise NotImplementedError(
            "FrontierHashSelector has been removed from the supported path; use SparseFrontierSelector "
            "or run_cipsi_trials(..., selection_mode='frontier_hash') instead"
        )
