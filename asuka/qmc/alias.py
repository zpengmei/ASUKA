from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AliasTable:
    """O(1) discrete sampler via Vose's alias method.

    Attributes
    ----------
    prob
        float32 array of shape (n,) with values in [0, 1].
    alias
        int32 array of shape (n,) with alias indices in [0, n).
    weight_sum
        Sum of the (clipped-to-nonnegative) input weights as float64.
    """

    prob: np.ndarray
    alias: np.ndarray
    weight_sum: float


def build_alias_table_from_weights(weights: np.ndarray) -> AliasTable:
    """Build an alias table for sampling indices with probability ∝ weights.

    Notes
    -----
    - Negative weights are clipped to zero.
    - If the total weight is zero, this returns a uniform sampler.
    """

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    n = int(w.size)
    if n <= 0:
        raise ValueError("weights must be non-empty")

    w = np.maximum(w, 0.0)
    w_sum = float(w.sum())

    prob = np.empty((n,), dtype=np.float32)
    alias = np.empty((n,), dtype=np.int32)

    if not np.isfinite(w_sum) or w_sum <= 0.0:
        prob.fill(np.float32(1.0))
        alias[:] = np.arange(n, dtype=np.int32)
        return AliasTable(prob=prob, alias=alias, weight_sum=0.0)

    p = w / w_sum
    q = p * float(n)

    small: list[int] = []
    large: list[int] = []
    for i in range(n):
        if q[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = np.float32(q[s])
        alias[s] = np.int32(l)
        q[l] = q[l] - (1.0 - q[s])
        if q[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in large:
        prob[i] = np.float32(1.0)
        alias[i] = np.int32(i)
    for i in small:
        prob[i] = np.float32(1.0)
        alias[i] = np.int32(i)

    return AliasTable(prob=prob, alias=alias, weight_sum=w_sum)

