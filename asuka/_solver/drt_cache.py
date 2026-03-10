from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def orbsym_to_tuple(orbsym: Any | None) -> tuple[int, ...] | None:
    if orbsym is None:
        return None
    return tuple(int(x) for x in np.asarray(orbsym).ravel().tolist())


def ne_constraints_to_key(
    ne_constraints: dict[int, tuple[int, int]] | None,
) -> tuple[tuple[int, int, int], ...] | None:
    if not ne_constraints:
        return None
    items: list[tuple[int, int, int]] = []
    for k, bounds in dict(ne_constraints).items():
        kk = int(k)
        if kk < 0:
            raise ValueError("ne_constraints keys must be >= 0")
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"ne_constraints[{kk}] must be a (ne_min, ne_max) tuple")
        ne_min = int(bounds[0])
        ne_max = int(bounds[1])
        if ne_min < 0 or ne_max < 0:
            raise ValueError(f"ne_constraints[{kk}] bounds must be >= 0")
        if ne_min > ne_max:
            raise ValueError(f"ne_constraints[{kk}] must satisfy ne_min <= ne_max")
        items.append((kk, ne_min, ne_max))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def ne_constraints_key_to_dict(
    ne_constraints_key: tuple[tuple[int, int, int], ...] | None,
) -> dict[int, tuple[int, int]] | None:
    if ne_constraints_key is None:
        return None
    out: dict[int, tuple[int, int]] = {}
    for k, ne_min, ne_max in ne_constraints_key:
        out[int(k)] = (int(ne_min), int(ne_max))
    return out


@dataclass(frozen=True)
class DRTKey:
    norb: int
    nelec_total: int
    twos: int
    wfnsym: int | None
    orbsym: tuple[int, ...] | None
    ne_constraints_key: tuple[tuple[int, int, int], ...] | None
