"""P2: Test RDM12 reorder + delta CUDA kernel matches NumPy reference."""
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
        rdm12_reorder_delta_inplace_device,
    )

    _has_kernel = has_sym_pair_pack_device()
except Exception:
    _has_kernel = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (_has_cupy and _has_kernel),
        reason="CuPy or rdm12_reorder_delta CUDA kernel not available",
    ),
]


def _reference_rdm12_reorder_delta(dm1_pq, gram0, norb):
    """NumPy reference for the RDM12 reorder + delta correction."""
    nops = norb * norb
    # dm1[q,p] = dm1_pq[p*norb+q]
    dm1 = dm1_pq.reshape(norb, norb).T.copy()

    # gram0 is [nops, nops]. Apply row-swap (transpose indices).
    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]
    dm2 = gram.reshape(norb, norb, norb, norb)

    # Delta correction
    for p in range(norb):
        for q in range(norb):
            dm2[p, q, q, :] -= dm1[:, p]

    return dm1, dm2


@pytest.mark.parametrize("norb", [2, 3, 4, 6, 8, 10])
def test_rdm12_reorder_delta_matches_reference(norb):
    nops = norb * norb
    rng = np.random.default_rng(123 + norb)

    dm1_pq_np = rng.standard_normal(nops)
    gram0_np = rng.standard_normal((nops, nops))

    dm1_ref, dm2_ref = _reference_rdm12_reorder_delta(dm1_pq_np, gram0_np, norb)

    dm1_pq_d = cp.asarray(dm1_pq_np, dtype=cp.float64)
    gram0_d = cp.asarray(gram0_np, dtype=cp.float64)

    dm1_out, dm2_out = rdm12_reorder_delta_inplace_device(
        dm1_pq_d, gram0_d, norb=norb, sync=True,
    )

    dm1_gpu = cp.asnumpy(dm1_out)
    dm2_gpu = cp.asnumpy(dm2_out)

    np.testing.assert_allclose(dm1_gpu, dm1_ref, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(dm2_gpu, dm2_ref, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("norb", [2, 4, 6])
def test_rdm12_symmetric_gram_input(norb):
    """Test with a symmetric gram matrix (typical in practice)."""
    nops = norb * norb
    rng = np.random.default_rng(456 + norb)

    dm1_pq_np = rng.standard_normal(nops)
    gram0_np = rng.standard_normal((nops, nops))
    gram0_np = 0.5 * (gram0_np + gram0_np.T)

    dm1_ref, dm2_ref = _reference_rdm12_reorder_delta(dm1_pq_np, gram0_np, norb)

    dm1_out, dm2_out = rdm12_reorder_delta_inplace_device(
        cp.asarray(dm1_pq_np), cp.asarray(gram0_np), norb=norb, sync=True,
    )

    np.testing.assert_allclose(cp.asnumpy(dm1_out), dm1_ref, atol=1e-14)
    np.testing.assert_allclose(cp.asnumpy(dm2_out), dm2_ref, atol=1e-13)


def test_rdm12_small_norb2():
    """Explicit 2-orbital case for easy manual verification."""
    norb = 2
    dm1_pq = np.array([1.0, 0.2, 0.2, 0.8])
    gram0 = np.eye(4) * 0.5

    dm1_ref, dm2_ref = _reference_rdm12_reorder_delta(dm1_pq, gram0, norb)

    dm1_out, dm2_out = rdm12_reorder_delta_inplace_device(
        cp.asarray(dm1_pq), cp.asarray(gram0), norb=norb, sync=True,
    )

    np.testing.assert_allclose(cp.asnumpy(dm1_out), dm1_ref, atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(dm2_out), dm2_ref, atol=1e-15)
