"""Hamiltonian oracle for GUGA/CSF-based quantum chemistry.

This package provides segment value evaluation, E_pq generator caching,
and connected-row generation for the spin-free Hamiltonian in the GUGA
framework. The implementation is split into focused submodules:

- :mod:`._segment` -- Segment value computation (Dobrautz Table 1).
- :mod:`._cache` -- Occupancy tables, E_pq CSR caches, and precomputation.
- :mod:`._connected` -- Connected row oracles and diagonal element guesses.
"""

from __future__ import annotations

# --- _segment exports ---
from asuka.cuguga.oracle._segment import (
    _SEG_LUT_MAX_B,
    _SV_BY_CODE,
    _A,
    _C,
    _Q_FROM_STR,
    _Q_L,
    _Q_R,
    _Q_W,
    _Q_oL,
    _Q_oR,
    _Q_uL,
    _Q_uR,
    _segment_value,
    _segment_value_int,
    _segment_value_int_fallback,
)

# --- _cache exports ---
from asuka.cuguga.oracle._cache import (
    OccGroups,
    _CSR,
    _EPQ_ACTION_CACHE,
    _EPQActionCache,
    _STEP_TO_OCC,
    _child_prefix_walks,
    _csr_for_epq,
    _e_pq_contribs_from_csf_index,
    _e_pq_contribs_from_csf_index_arrays,
    _epq_contribs_cached,
    _get_epq_action_cache,
    _path_nodes,
    epq_cache_nbytes,
    occ_groups,
    occ_table,
    precompute_epq_actions,
)

# --- _connected exports ---
from asuka.cuguga.oracle._connected import (
    _assign_spin_occupations,
    _restore_eri_4d,
    connected_row,
    connected_row_structure_only,
    diagonal_element_det_guess,
)

__all__ = [
    "OccGroups",
    "connected_row",
    "connected_row_structure_only",
    "diagonal_element_det_guess",
    "epq_cache_nbytes",
    "occ_groups",
    "occ_table",
    "precompute_epq_actions",
]
