"""P4: Test batched scatter embed / gather project CUDA kernels match per-vector reference."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy as cp

    _has_cupy = True
except Exception:
    _has_cupy = False

try:
    from asuka.cuda.cuda_backend import (
        has_sym_pair_pack_device,
        scatter_embed_batched_inplace_device,
        gather_project_batched_inplace_device,
    )

    _has_kernel = has_sym_pair_pack_device()
except Exception:
    _has_kernel = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (_has_cupy and _has_kernel),
        reason="CuPy or batched scatter/gather CUDA kernel not available",
    ),
]


@pytest.mark.parametrize("nsub,nfull,nvec", [
    (10, 50, 1),
    (10, 50, 4),
    (100, 500, 8),
    (50, 200, 16),
])
def test_scatter_embed_batched_matches_pervec(nsub, nfull, nvec):
    rng = np.random.default_rng(42 + nsub + nvec)

    # Random subset mapping
    sub_to_full = np.sort(rng.choice(nfull, size=nsub, replace=False)).astype(np.int64)
    x_sub = rng.standard_normal((nsub, nvec))

    # Reference: per-vector scatter
    x_full_ref = np.zeros((nfull, nvec))
    for v in range(nvec):
        x_full_ref[sub_to_full, v] = x_sub[:, v]

    # GPU batched
    x_full_d = cp.zeros((nfull, nvec), dtype=cp.float64)
    scatter_embed_batched_inplace_device(
        cp.asarray(x_sub, dtype=cp.float64),
        cp.asarray(sub_to_full, dtype=cp.int64),
        x_full_d,
        sync=True,
    )

    np.testing.assert_allclose(cp.asnumpy(x_full_d), x_full_ref, atol=1e-15)


@pytest.mark.parametrize("nsub,nfull,nvec", [
    (10, 50, 1),
    (10, 50, 4),
    (100, 500, 8),
    (50, 200, 16),
])
def test_gather_project_batched_matches_pervec(nsub, nfull, nvec):
    rng = np.random.default_rng(99 + nsub + nvec)

    sub_to_full = np.sort(rng.choice(nfull, size=nsub, replace=False)).astype(np.int64)
    y_full = rng.standard_normal((nfull, nvec))

    # Reference: per-vector gather
    y_sub_ref = np.zeros((nsub, nvec))
    for v in range(nvec):
        y_sub_ref[:, v] = y_full[sub_to_full, v]

    # GPU batched
    y_sub_d = cp.empty((nsub, nvec), dtype=cp.float64)
    gather_project_batched_inplace_device(
        cp.asarray(y_full, dtype=cp.float64),
        cp.asarray(sub_to_full, dtype=cp.int64),
        y_sub_d,
        sync=True,
    )

    np.testing.assert_allclose(cp.asnumpy(y_sub_d), y_sub_ref, atol=1e-15)


def test_scatter_gather_roundtrip():
    """Scatter then gather should recover the original sub-vector."""
    nsub, nfull, nvec = 30, 100, 5
    rng = np.random.default_rng(7)

    sub_to_full = np.sort(rng.choice(nfull, size=nsub, replace=False)).astype(np.int64)
    x_sub = rng.standard_normal((nsub, nvec))

    x_sub_d = cp.asarray(x_sub, dtype=cp.float64)
    stf_d = cp.asarray(sub_to_full, dtype=cp.int64)

    x_full_d = cp.zeros((nfull, nvec), dtype=cp.float64)
    scatter_embed_batched_inplace_device(x_sub_d, stf_d, x_full_d, sync=True)

    y_sub_d = cp.empty((nsub, nvec), dtype=cp.float64)
    gather_project_batched_inplace_device(x_full_d, stf_d, y_sub_d, sync=True)

    np.testing.assert_allclose(cp.asnumpy(y_sub_d), x_sub, atol=1e-15)
