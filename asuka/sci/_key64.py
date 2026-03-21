"""Key64 encoding/decoding and state-rep helpers (internalized from qmc)."""

from __future__ import annotations

from typing import Any

import numpy as np


_STATE_REP_VALUES = {"auto", "i32", "i64", "key64"}


def normalize_state_rep(state_rep: str) -> str:
    rep = str(state_rep).strip().lower()
    if rep == "idx64":
        rep = "i64"
    if rep not in _STATE_REP_VALUES:
        raise ValueError("state_rep must be 'auto', 'i32', 'i64' (or 'idx64'), or 'key64'")
    return rep


def pack_steps_to_key64_host(steps: np.ndarray) -> np.ndarray:
    """Pack a step table (0..3 per orbital) into Key64 representation (2 bits per orbital).

    Parameters
    ----------
    steps
        int/uint array of shape ``(n, norb)`` with entries in ``{0,1,2,3}``.

    Returns
    -------
    np.ndarray
        uint64 array of shape ``(n,)``.
    """
    steps_u8 = np.asarray(steps, dtype=np.uint8)
    if steps_u8.ndim != 2:
        raise ValueError("steps must have shape (n, norb)")
    n, norb = int(steps_u8.shape[0]), int(steps_u8.shape[1])
    if norb < 0 or norb > 32:
        raise ValueError("Key64 packing requires 0 <= norb <= 32")
    if n == 0:
        return np.zeros((0,), dtype=np.uint64)
    key = np.zeros((n,), dtype=np.uint64)
    for k in range(norb):
        key |= (steps_u8[:, k].astype(np.uint64) & np.uint64(3)) << np.uint64(2 * k)
    return key


def csf_idx_to_key64_host(drt: Any, idx: np.ndarray, *, state_cache: Any | None = None) -> np.ndarray:
    """Convert CSF indices to Key64 (uint64) using cached step tables if available."""
    idx_i64 = np.asarray(idx, dtype=np.int64).ravel()
    norb = int(drt.norb)
    if norb > 32:
        raise ValueError("Key64 representation requires drt.norb <= 32")
    if idx_i64.size == 0:
        return np.zeros((0,), dtype=np.uint64)

    if state_cache is not None:
        steps_table = np.asarray(state_cache.steps, dtype=np.int8, order="C")
        steps = steps_table[idx_i64]
    else:
        steps = np.stack([np.asarray(drt.index_to_path(int(i)), dtype=np.int8, order="C") for i in idx_i64], axis=0)
    return pack_steps_to_key64_host(steps)
