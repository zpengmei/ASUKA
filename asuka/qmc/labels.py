from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT

from .sparse import coalesce_coo_i32_f64, coalesce_coo_i64_f64


_STATE_REP_VALUES = {"auto", "i32", "i64", "key64"}


def normalize_state_rep(state_rep: str) -> str:
    rep = str(state_rep).strip().lower()
    if rep == "idx64":
        rep = "i64"
    if rep not in _STATE_REP_VALUES:
        raise ValueError("state_rep must be 'auto', 'i32', 'i64' (or 'idx64'), or 'key64'")
    return rep


def requires_int64_labels(drt: DRT, idx: np.ndarray | None = None) -> bool:
    if int(drt.ncsf) > np.iinfo(np.int32).max:
        return True
    if idx is None:
        return False
    idx_arr = np.asarray(idx)
    if idx_arr.size == 0:
        return False
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError("idx must have an integer dtype")
    if idx_arr.dtype.kind == "u":
        return int(np.max(idx_arr)) > np.iinfo(np.int32).max
    return int(np.min(idx_arr)) < np.iinfo(np.int32).min or int(np.max(idx_arr)) > np.iinfo(np.int32).max


def resolve_label_array(
    drt: DRT,
    *,
    idx: np.ndarray | None,
    key: np.ndarray | None,
    name: str,
) -> np.ndarray | None:
    if idx is not None and key is not None:
        raise ValueError(f"{name} must be provided as either idx or key64, not both")
    if key is not None:
        if int(drt.norb) > 32:
            raise ValueError(f"{name} key64 input requires drt.norb <= 32")
        from .cuda_backend import key64_to_csf_idx64_host  # noqa: PLC0415

        return key64_to_csf_idx64_host(drt, np.asarray(key, dtype=np.uint64).ravel(), strict=True)
    if idx is None:
        return None
    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} idx must have an integer dtype")
    if idx_arr.dtype.kind == "u":
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
    return idx_arr


def resolve_optional_label_index(
    drt: DRT,
    *,
    idx: int | np.integer | None,
    key: int | np.integer | None,
    name: str,
) -> int | None:
    if idx is not None and key is not None:
        raise ValueError(f"{name} must be provided as either idx or key64, not both")
    if key is None:
        return None if idx is None else int(idx)
    key_arr = np.asarray([key], dtype=np.uint64)
    idx_arr = resolve_label_array(drt, idx=None, key=key_arr, name=name)
    if idx_arr is None or idx_arr.size != 1:
        raise RuntimeError(f"internal error: failed to resolve {name}")
    return int(idx_arr[0])


def coalesce_sparse_state(
    drt: DRT,
    *,
    idx: np.ndarray | None,
    key: np.ndarray | None,
    val: np.ndarray | None,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    idx_arr = resolve_label_array(drt, idx=idx, key=key, name=name)
    if idx_arr is None:
        if val is not None:
            raise ValueError(f"{name} values were provided without labels")
        return None, None
    if val is None:
        raise ValueError(f"{name} values are required when labels are provided")
    val_arr = np.asarray(val, dtype=np.float64).ravel()
    if idx_arr.size != val_arr.size:
        raise ValueError(f"{name} labels and values must have the same length")
    if requires_int64_labels(drt, idx_arr):
        return coalesce_coo_i64_f64(idx_arr, val_arr)
    return coalesce_coo_i32_f64(idx_arr, val_arr)
