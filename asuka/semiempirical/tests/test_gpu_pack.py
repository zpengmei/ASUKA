from __future__ import annotations

import numpy as np

from asuka.semiempirical.gpu.kernels import build_pair_buckets


def test_build_pair_buckets_hcnof_mix():
    # Atom AO sizes: H(1), C(4), O(4), H(1)
    atomic_numbers = [1, 6, 8, 1]
    # Pairs in arbitrary order to exercise all buckets.
    pair_i = np.asarray([0, 0, 1, 2], dtype=np.int32)
    pair_j = np.asarray([1, 3, 3, 1], dtype=np.int32)
    # Buckets by (naoA, naoB):
    # 0-1 -> 14, 0-3 -> 11, 1-3 -> 41, 2-1 -> 44
    buckets = build_pair_buckets(atomic_numbers, pair_i, pair_j)

    assert buckets["11"].tolist() == [1]
    assert buckets["14"].tolist() == [0]
    assert buckets["41"].tolist() == [2]
    assert buckets["44"].tolist() == [3]
