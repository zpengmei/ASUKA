from __future__ import annotations

from typing import Iterable

import numpy as np

from asuka.cuguga.drt import DRT


def csf_printable_to_drt_steps(printable: str) -> tuple[str, ...]:
    """Map csf_fci `printable_csfstring` output to the toy DRT step alphabet.

    csf_fci uses:
    - '0' empty orbital
    - '2' doubly occupied orbital
    - 'u'/'d' singly occupied orbital with a genealogical coupling bit

    For the toy DRT we use per-orbital steps in {'E','D','U','L'}.

    Note: csf_fci prints occupation strings with the highest orbital on the left
    (orbital 0 is the right-most character). We reverse the string so that the
    returned step sequence uses orbital index order 0..norb-1.
    """

    s = str(printable)[::-1]
    steps: list[str] = []
    for ch in s:
        if ch == "0":
            steps.append("E")
        elif ch == "2":
            steps.append("D")
        elif ch == "u":
            steps.append("U")
        elif ch == "d":
            steps.append("L")
        else:
            raise ValueError(f"unexpected csf_fci printable character {ch!r}")
    return tuple(steps)


def csfaddr_to_drt_index(transformer, drt: DRT, csfaddr: int) -> int:
    printable = transformer.printable_csfstring(int(csfaddr))
    steps = csf_printable_to_drt_steps(str(printable))
    return int(drt.path_to_index(steps))


def build_csfaddr_to_drt_perm(transformer, drt: DRT) -> np.ndarray:
    ncsf = int(transformer.ncsf)
    if int(drt.ncsf) != ncsf:
        raise ValueError(f"ncsf mismatch: transformer={ncsf} drt={int(drt.ncsf)}")
    perm = np.empty(ncsf, dtype=np.int32)
    for csfaddr in range(ncsf):
        perm[csfaddr] = csfaddr_to_drt_index(transformer, drt, csfaddr)
    return perm


def invert_permutation(perm: Iterable[int], n: int | None = None) -> np.ndarray:
    perm = np.asarray(list(perm), dtype=np.int64)
    if perm.ndim != 1:
        raise ValueError("perm must be 1D")
    n = int(perm.size) if n is None else int(n)
    if perm.size != n:
        raise ValueError("perm length mismatch")
    if perm.min(initial=0) < 0 or perm.max(initial=-1) >= n:
        raise ValueError("perm values out of range")
    inv = np.empty(n, dtype=np.int32)
    inv.fill(-1)
    for i, p in enumerate(perm.tolist()):
        if inv[p] != -1:
            raise ValueError("perm is not a bijection")
        inv[p] = i
    if np.any(inv < 0):
        raise ValueError("perm is not a bijection")
    return inv
