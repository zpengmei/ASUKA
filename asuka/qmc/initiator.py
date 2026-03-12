from __future__ import annotations


def initiator_threshold(*, l1_norm: float, m: int, na: float) -> float:
    """Column-scaled initiator threshold: t = na * ||x||_1 / (m - 1)."""

    na = float(na)
    if na <= 0.0:
        return 0.0
    m = int(m)
    if m <= 1:
        raise ValueError("m must be > 1 when initiator_na > 0")
    return na * float(l1_norm) / float(m - 1)
