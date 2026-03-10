from __future__ import annotations

from typing import Any

from asuka.cuguga.drt import DRT, build_drt

from .drt_cache import DRTKey, ne_constraints_to_key, orbsym_to_tuple


def drt_key(
    norb: int,
    nelec_total: int,
    twos: int,
    orbsym: Any | None,
    wfnsym: int | None,
    *,
    ne_constraints: dict[int, tuple[int, int]] | None = None,
) -> DRTKey:
    return DRTKey(
        norb=int(norb),
        nelec_total=int(nelec_total),
        twos=int(twos),
        wfnsym=None if wfnsym is None else int(wfnsym),
        orbsym=orbsym_to_tuple(orbsym),
        ne_constraints_key=ne_constraints_to_key(ne_constraints),
    )


def get_or_build_drt(
    cache: dict[DRTKey, DRT],
    *,
    norb: int,
    nelec_total: int,
    twos: int,
    orbsym: Any | None = None,
    wfnsym: int | None = None,
    ne_constraints: dict[int, tuple[int, int]] | None = None,
) -> tuple[DRTKey, DRT]:
    key = drt_key(norb, nelec_total, twos, orbsym, wfnsym, ne_constraints=ne_constraints)
    drt = cache.get(key)
    if drt is None:
        drt = build_drt(
            norb=norb,
            nelec=nelec_total,
            twos_target=twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        cache[key] = drt
    return key, drt
