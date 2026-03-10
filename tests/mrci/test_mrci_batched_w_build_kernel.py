"""P1A: Test batched W-build + diagonal CUDA kernels match per-vector reference."""
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
        has_build_w_batched_device,
        build_w_from_epq_table_batched_inplace_device,
        build_w_diag_batched_inplace_device,
    )

    _has_kernel = has_build_w_batched_device()
except Exception:
    _has_kernel = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (_has_cupy and _has_kernel),
        reason="CuPy or batched W-build CUDA kernel not available",
    ),
]


def _random_epq_table(ncsf, norb, density=0.3, rng=None):
    """Build a random source-major EPQ CSR table (offdiag only)."""
    if rng is None:
        rng = np.random.default_rng(42)
    nops = norb * norb

    indptr = [0]
    indices = []
    pq_ids = []
    data = []

    for j in range(ncsf):
        nnz_row = rng.binomial(ncsf, density)
        # random destination CSFs (excluding self for offdiag)
        dests = rng.choice(ncsf, size=nnz_row, replace=True)
        for k in dests:
            p = rng.integers(0, norb)
            q = rng.integers(0, norb)
            if p == q:
                continue  # offdiag only
            pq = p * norb + q
            coef = rng.standard_normal()
            indices.append(k)
            pq_ids.append(pq)
            data.append(coef)
        indptr.append(len(indices))

    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)
    pq_ids = np.array(pq_ids, dtype=np.int32)
    data = np.array(data, dtype=np.float64)
    return indptr, indices, pq_ids, data


def _reference_w_build(epq_table, x_np, norb, k_start, k_count, nvec, v_start):
    """NumPy reference: per-vector W-build from EPQ table."""
    indptr, indices, pq_ids, data = epq_table
    ncsf = len(indptr) - 1
    nops = norb * norb
    W_stack = np.zeros((nvec * k_count, nops))

    for v in range(nvec):
        for j in range(ncsf):
            for e in range(indptr[j], indptr[j + 1]):
                k = indices[e]
                if k < k_start or k >= k_start + k_count:
                    continue
                pq = pq_ids[e]
                coef = data[e]
                k_local = k - k_start
                W_stack[v * k_count + k_local, pq] += coef * x_np[j, v_start + v]
    return W_stack


def _random_steps_table(ncsf, norb, rng=None):
    """Random steps table: values in {0, 1, 2, 3}."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, 4, size=(ncsf, norb), dtype=np.int8)


def _reference_w_diag(steps_table, x_np, norb, j_start, j_count, nvec, v_start):
    """NumPy reference: per-vector diagonal W-build."""
    nops = norb * norb
    W_stack = np.zeros((nvec * j_count, nops))

    step_to_occ = {0: 0, 1: 1, 2: 1, 3: 2}

    for v in range(nvec):
        for j_local in range(j_count):
            j = j_start + j_local
            for r in range(norb):
                step = int(steps_table[j, r])
                occ_r = step_to_occ[step]
                if occ_r == 0:
                    continue
                rr = r * norb + r
                W_stack[v * j_count + j_local, rr] = occ_r * x_np[j, v_start + v]
    return W_stack


@pytest.mark.parametrize("norb,ncsf", [(4, 20), (6, 50), (8, 100)])
@pytest.mark.parametrize("nvec", [1, 4, 8])
def test_batched_w_build_matches_reference(norb, ncsf, nvec):
    rng = np.random.default_rng(42 + norb * 100 + ncsf + nvec)
    epq_table_np = _random_epq_table(ncsf, norb, density=0.2, rng=rng)

    nvec_total = nvec + 2  # extra columns to test v_start offset
    v_start = 1
    x_np = rng.standard_normal((ncsf, nvec_total))

    k_start = ncsf // 4
    k_count = min(ncsf // 2, ncsf - k_start)

    W_ref = _reference_w_build(epq_table_np, x_np, norb, k_start, k_count, nvec, v_start)

    epq_table_d = tuple(cp.asarray(a) for a in epq_table_np)
    x_d = cp.asarray(x_np, dtype=cp.float64)
    nops = norb * norb
    W_d = cp.zeros((nvec * k_count, nops), dtype=cp.float64)

    build_w_from_epq_table_batched_inplace_device(
        epq_table_d,
        x_d,
        ncsf=ncsf,
        norb=norb,
        w_out=W_d,
        nvec=nvec,
        v_start=v_start,
        k_start=k_start,
        k_count=k_count,
        sync=True,
    )

    W_out = cp.asnumpy(W_d)
    np.testing.assert_allclose(W_out, W_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("norb,ncsf", [(4, 30), (6, 50), (8, 80)])
@pytest.mark.parametrize("nvec", [1, 4, 8])
def test_batched_w_diag_matches_reference(norb, ncsf, nvec):
    rng = np.random.default_rng(99 + norb * 100 + ncsf + nvec)
    steps_table_np = _random_steps_table(ncsf, norb, rng=rng)

    nvec_total = nvec + 3
    v_start = 2
    x_np = rng.standard_normal((ncsf, nvec_total))

    j_start = ncsf // 3
    j_count = min(ncsf // 3, ncsf - j_start)

    W_ref = _reference_w_diag(steps_table_np, x_np, norb, j_start, j_count, nvec, v_start)

    from asuka.cuguga.state_cache import DRTStateCache

    # We need a mock state_dev with steps_table_dev attribute
    class MockStateDev:
        def __init__(self, steps):
            self.steps_table_dev = steps

    steps_d = cp.asarray(steps_table_np, dtype=cp.int8)
    state_dev = MockStateDev(steps_d)

    x_d = cp.asarray(x_np, dtype=cp.float64)
    nops = norb * norb
    W_d = cp.zeros((nvec * j_count, nops), dtype=cp.float64)

    build_w_diag_batched_inplace_device(
        state_dev,
        x_d,
        w_out=W_d,
        ncsf=ncsf,
        norb=norb,
        j_start=j_start,
        j_count=j_count,
        nvec=nvec,
        v_start=v_start,
        sync=True,
    )

    W_out = cp.asnumpy(W_d)
    np.testing.assert_allclose(W_out, W_ref, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("norb", [4, 6])
def test_batched_offdiag_plus_diag_matches_pervec(norb):
    """Combined off-diagonal + diagonal batched matches per-vector sequential build."""
    ncsf = 40
    nvec = 4
    v_start = 0
    k_start = 0
    k_count = ncsf
    nops = norb * norb

    rng = np.random.default_rng(7 + norb)
    epq_table_np = _random_epq_table(ncsf, norb, density=0.15, rng=rng)
    steps_table_np = _random_steps_table(ncsf, norb, rng=rng)
    x_np = rng.standard_normal((ncsf, nvec))

    # Reference: per-vector
    W_ref = _reference_w_build(epq_table_np, x_np, norb, k_start, k_count, nvec, v_start)
    W_diag = _reference_w_diag(steps_table_np, x_np, norb, k_start, k_count, nvec, v_start)
    W_ref += W_diag

    # GPU batched
    epq_table_d = tuple(cp.asarray(a) for a in epq_table_np)
    x_d = cp.asarray(x_np, dtype=cp.float64)

    W_d = cp.zeros((nvec * k_count, nops), dtype=cp.float64)
    build_w_from_epq_table_batched_inplace_device(
        epq_table_d, x_d,
        ncsf=ncsf, norb=norb, w_out=W_d,
        nvec=nvec, v_start=v_start,
        k_start=k_start, k_count=k_count, sync=True,
    )

    class MockStateDev:
        def __init__(self, steps):
            self.steps_table_dev = steps

    steps_d = cp.asarray(steps_table_np, dtype=cp.int8)
    build_w_diag_batched_inplace_device(
        MockStateDev(steps_d), x_d,
        w_out=W_d, ncsf=ncsf, norb=norb,
        j_start=k_start, j_count=k_count,
        nvec=nvec, v_start=v_start, sync=True,
    )

    W_out = cp.asnumpy(W_d)
    np.testing.assert_allclose(W_out, W_ref, atol=1e-12, rtol=1e-12)
