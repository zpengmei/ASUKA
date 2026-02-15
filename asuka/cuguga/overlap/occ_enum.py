from __future__ import annotations

from collections.abc import Callable

import numpy as np

from asuka.cuguga.drt import DRT


def for_each_csf_with_occupancy_fixed_outside_union(
    drt: DRT,
    *,
    occ: np.ndarray,
    ket_steps: np.ndarray,
    union_start: int,
    union_end: int,
    max_visits: int | None = None,
    visit: Callable[[int, np.ndarray], None],
) -> int:
    """Enumerate CSFs with a fixed occupancy pattern, forcing ket steps outside a union range.

    Parameters
    ----------
    occ
        Target spatial occupancy pattern (0/1/2 per orbital).
    ket_steps
        Reference ket DRT steps. Outside `[union_start, union_end]` the bra steps
        are forced to equal `ket_steps` (required for nonzero generator products).
    union_start, union_end
        Inclusive union range where bra steps may differ from ket, subject to `occ`.
    max_visits
        Optional cap on the number of yielded CSFs. If exceeded, raises ValueError.
    visit
        Callback invoked as `visit(bra_idx, bra_steps)` for each enumerated CSF.
        The `bra_steps` array is a mutable workspace; do not store it.

    Returns
    -------
    int
        Number of visited CSFs.
    """

    norb = int(drt.norb)
    occ = np.asarray(occ, dtype=np.int8).ravel()
    ket_steps = np.asarray(ket_steps, dtype=np.int8).ravel()
    if int(occ.size) != norb:
        raise ValueError(f"occ has wrong length: {int(occ.size)} (expected {norb})")
    if int(ket_steps.size) != norb:
        raise ValueError(f"ket_steps has wrong length: {int(ket_steps.size)} (expected {norb})")

    union_start = int(union_start)
    union_end = int(union_end)
    if union_start < 0 or union_end < 0 or union_start >= norb or union_end >= norb:
        raise ValueError("union_start/union_end out of range")
    if union_end < union_start:
        raise ValueError("union_end must be >= union_start")

    # Mutable bra steps workspace; outside union we keep ket steps.
    bra_steps = ket_steps.copy()

    def occ_allows_step(occ_k: int, step: int) -> bool:
        if occ_k == 0:
            return step == 0  # E
        if occ_k == 2:
            return step == 3  # D
        if occ_k == 1:
            return step in (1, 2)  # U/L
        return False

    # Compute prefix node and index offset for the fixed ket prefix [0, union_start).
    node = int(drt.root)
    idx0 = 0
    for k in range(union_start):
        sidx = int(ket_steps[k])
        if not occ_allows_step(int(occ[k]), sidx):
            return 0
        for prior in range(sidx):
            child = int(drt.child[node, prior])
            if child >= 0:
                idx0 += int(drt.nwalks[child])
        child = int(drt.child[node, sidx])
        if child < 0:
            return 0
        node = child

    visits = 0

    def follow_suffix(node_in: int, idx_in: int, start_k: int) -> None:
        nonlocal visits
        node2 = int(node_in)
        idx2 = int(idx_in)
        for k in range(start_k, norb):
            sidx = int(ket_steps[k])
            if not occ_allows_step(int(occ[k]), sidx):
                return
            for prior in range(sidx):
                child = int(drt.child[node2, prior])
                if child >= 0:
                    idx2 += int(drt.nwalks[child])
            child = int(drt.child[node2, sidx])
            if child < 0:
                return
            node2 = child
        if node2 != int(drt.leaf):
            return
        visits += 1
        if max_visits is not None and visits > int(max_visits):
            raise ValueError(f"occupancy enumeration exceeded max_visits={int(max_visits)}")
        visit(int(idx2), bra_steps)

    def dfs(k: int, node_in: int, idx_in: int) -> None:
        if k == union_end + 1:
            follow_suffix(node_in, idx_in, k)
            return

        occ_k = int(occ[k])
        allowed: tuple[int, ...]
        if occ_k == 0:
            allowed = (0,)
        elif occ_k == 2:
            allowed = (3,)
        elif occ_k == 1:
            allowed = (1, 2)
        else:
            return

        old = int(bra_steps[k])
        for sidx in allowed:
            child = int(drt.child[int(node_in), int(sidx)])
            if child < 0:
                continue
            off = 0
            for prior in range(int(sidx)):
                cprior = int(drt.child[int(node_in), prior])
                if cprior >= 0:
                    off += int(drt.nwalks[cprior])
            bra_steps[k] = int(sidx)
            dfs(k + 1, child, int(idx_in) + off)
        bra_steps[k] = old

    dfs(union_start, node, idx0)
    return int(visits)
