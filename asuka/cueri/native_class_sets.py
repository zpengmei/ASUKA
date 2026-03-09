from __future__ import annotations

from .tasks import eri_class_id


def _cid(la: int, lb: int, lc: int, ld: int) -> int:
    return int(eri_class_id(int(la), int(lb), int(lc), int(ld)))


# Native Step-2 quartet classes available in cuERI GPU kernels (including ff*).
_STEP2_NATIVE_CLASS_TUPLES: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0),  # ssss
    (1, 0, 0, 0),  # psss
    (1, 1, 0, 0),  # ppss
    (1, 0, 1, 0),  # psps
    (1, 1, 1, 0),  # ppps
    (1, 1, 1, 1),  # pppp
    (2, 0, 0, 0),  # dsss
    (2, 2, 0, 0),  # ddss
    (0, 0, 2, 1),  # ssdp
    (1, 0, 2, 0),  # psds
    (1, 0, 2, 1),  # psdp
    (1, 0, 2, 2),  # psdd
    (1, 1, 2, 0),  # ppds
    (1, 1, 2, 1),  # ppdp
    (1, 1, 2, 2),  # ppdd
    (2, 0, 2, 0),  # dsds
    (2, 0, 2, 1),  # dsdp
    (2, 0, 2, 2),  # dsdd
    (3, 1, 0, 0),  # fpss
    (3, 2, 0, 0),  # fdss
    (3, 3, 0, 0),  # ffss
    (3, 1, 1, 0),  # fpps
    (3, 2, 1, 0),  # fdps
    (3, 3, 1, 0),  # ffps
    (3, 1, 2, 0),  # fpds
    (3, 2, 2, 0),  # fdds
    (3, 3, 2, 0),  # ffds
    (0, 0, 3, 0),  # ssfs
    (1, 0, 3, 0),  # psfs
    (1, 1, 3, 0),  # ppfs
    (2, 0, 3, 0),  # dsfs
    (3, 0, 3, 0),  # fsfs
    (2, 1, 3, 0),  # dpfs
    (3, 1, 3, 0),  # fpfs
    (2, 2, 3, 0),  # ddfs
    (3, 2, 3, 0),  # fdfs
    (3, 3, 3, 0),  # fffs
    (0, 0, 4, 0),  # ssgs
    (1, 0, 4, 0),  # psgs
    (1, 1, 4, 0),  # ppgs
    (2, 0, 4, 0),  # dsgs
    (3, 0, 4, 0),  # fsgs
    (2, 1, 4, 0),  # dpgs
    (3, 1, 4, 0),  # fpgs
    (2, 2, 4, 0),  # ddgs
    (3, 2, 4, 0),  # fdgs
    (3, 3, 4, 0),  # ffgs
    (2, 1, 2, 1),  # dpdp
    (2, 1, 2, 2),  # dpdd
    (2, 2, 2, 2),  # dddd
)

# ff* classes are currently opt-in in the 4e dispatcher, but always native for
# contracted-AO Step-2 eligibility checks.
_FF_NATIVE_CLASS_TUPLES: tuple[tuple[int, int, int, int], ...] = (
    (3, 3, 0, 0),  # ffss
    (3, 3, 1, 0),  # ffps
    (3, 3, 2, 0),  # ffds
    (3, 3, 3, 0),  # fffs
    (3, 3, 4, 0),  # ffgs
)

STEP2_NATIVE_CLASS_IDS: frozenset[int] = frozenset(_cid(*q) for q in _STEP2_NATIVE_CLASS_TUPLES)
FF_NATIVE_CLASS_IDS: frozenset[int] = frozenset(_cid(*q) for q in _FF_NATIVE_CLASS_TUPLES)
DISPATCH_NATIVE_CLASS_IDS: frozenset[int] = frozenset(int(cid) for cid in STEP2_NATIVE_CLASS_IDS if int(cid) not in FF_NATIVE_CLASS_IDS)


__all__ = ["DISPATCH_NATIVE_CLASS_IDS", "FF_NATIVE_CLASS_IDS", "STEP2_NATIVE_CLASS_IDS"]
