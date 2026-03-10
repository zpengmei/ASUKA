from __future__ import annotations

from collections import OrderedDict
import json
import threading
from typing import Any

import numpy as np


_HF_CACHE_LOCK = threading.Lock()


def cache_get(cache: OrderedDict, key: tuple[Any, ...]):
    with _HF_CACHE_LOCK:
        hit = cache.get(key)
        if hit is None:
            return None
        cache.move_to_end(key)
        return hit


def cache_put(cache: OrderedDict, key: tuple[Any, ...], val: Any, max_size: int):
    if int(max_size) <= 0:
        return
    with _HF_CACHE_LOCK:
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > int(max_size):
            cache.popitem(last=False)


def cache_clear_all(*caches: OrderedDict) -> None:
    with _HF_CACHE_LOCK:
        for cache in caches:
            cache.clear()


def normalize_basis_key(spec: Any) -> tuple[str, str]:
    if isinstance(spec, str):
        return ("str", str(spec).strip().lower())
    if isinstance(spec, dict):
        try:
            txt = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            txt = repr(spec)
        return ("dict", txt)
    return (type(spec).__name__, repr(spec))


def mol_cache_key(mol: Any) -> tuple[Any, ...]:
    atoms = []
    for sym, xyz in mol.atoms_bohr:
        x, y, z = map(float, np.asarray(xyz, dtype=np.float64).reshape((3,)))
        atoms.append((str(sym), round(x, 12), round(y, 12), round(z, 12)))
    return (tuple(atoms), int(mol.charge), int(mol.spin), bool(mol.cart))


def cuda_device_id_or_neg1() -> int:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        return -1
    try:
        return int(cp.cuda.Device().id)
    except Exception:
        return -1
