"""P1C: Test sym-pair pack/unpack CUDA kernels match CuPy cp.take reference."""
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
        sym_pair_pack_inplace_device,
        sym_pair_unpack_inplace_device,
    )

    _has_kernel = has_sym_pair_pack_device()
except Exception:
    _has_kernel = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (_has_cupy and _has_kernel),
        reason="CuPy or sym_pair CUDA kernel not available",
    ),
]


def _build_sym_pair_maps(norb: int):
    """Build pair_pq, pair_qp, full_to_pair index arrays (NumPy)."""
    nops = norb * norb
    pair_pq = []
    pair_qp = []
    full_to_pair = np.empty(nops, dtype=np.int32)
    u = 0
    for p in range(norb):
        for q in range(p, norb):
            pq = p * norb + q
            qp = q * norb + p
            pair_pq.append(pq)
            pair_qp.append(qp)
            full_to_pair[pq] = u
            full_to_pair[qp] = u
            u += 1
    pair_pq = np.asarray(pair_pq, dtype=np.int32)
    pair_qp = np.asarray(pair_qp, dtype=np.int32)
    npair = u
    return pair_pq, pair_qp, full_to_pair, npair


def _reference_pack(W_np, pair_pq, pair_qp):
    """Reference pack using NumPy indexing."""
    W_pair = W_np[:, pair_pq] + W_np[:, pair_qp]
    diag_mask = pair_pq == pair_qp
    W_pair[:, diag_mask] *= 0.5
    return W_pair


def _reference_unpack(G_pair_np, full_to_pair):
    """Reference unpack using NumPy indexing."""
    return G_pair_np[:, full_to_pair]


@pytest.mark.parametrize("norb", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("nrows", [1, 10, 100, 512])
def test_sym_pair_pack_matches_reference(norb, nrows):
    nops = norb * norb
    pair_pq, pair_qp, full_to_pair, npair = _build_sym_pair_maps(norb)

    rng = np.random.default_rng(42 + norb * 1000 + nrows)
    W_np = rng.standard_normal((nrows, nops))

    W_ref = _reference_pack(W_np, pair_pq, pair_qp)

    W_d = cp.asarray(W_np, dtype=cp.float64)
    W_pair_d = cp.empty((nrows, npair), dtype=cp.float64)

    sym_pair_pack_inplace_device(
        W_d,
        W_pair_d,
        cp.asarray(pair_pq),
        cp.asarray(pair_qp),
        nrows=nrows,
        nops=nops,
        npair=npair,
        sync=True,
    )

    W_pair_out = cp.asnumpy(W_pair_d)
    np.testing.assert_allclose(W_pair_out, W_ref, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("norb", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("nrows", [1, 10, 100, 512])
def test_sym_pair_unpack_matches_reference(norb, nrows):
    nops = norb * norb
    pair_pq, pair_qp, full_to_pair, npair = _build_sym_pair_maps(norb)

    rng = np.random.default_rng(99 + norb * 1000 + nrows)
    G_pair_np = rng.standard_normal((nrows, npair))

    G_ref = _reference_unpack(G_pair_np, full_to_pair)

    G_pair_d = cp.asarray(G_pair_np, dtype=cp.float64)
    G_d = cp.empty((nrows, nops), dtype=cp.float64)

    sym_pair_unpack_inplace_device(
        G_pair_d,
        G_d,
        cp.asarray(full_to_pair),
        nrows=nrows,
        nops=nops,
        npair=npair,
        sync=True,
    )

    G_out = cp.asnumpy(G_d)
    np.testing.assert_allclose(G_out, G_ref, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("norb", [4, 8])
def test_pack_gemm_unpack_roundtrip(norb):
    """Test that pack → GEMM → unpack gives same result as full GEMM.

    Requires ERI with (pq)↔(qp) symmetry: eri[p*n+q, rs] = eri[q*n+p, rs].
    """
    nops = norb * norb
    pair_pq, pair_qp, full_to_pair, npair = _build_sym_pair_maps(norb)

    rng = np.random.default_rng(7)
    nrows = 64

    W_np = rng.standard_normal((nrows, nops))
    # Build ERI with (pq)↔(qp) symmetry on both bra and ket indices
    eri_raw = rng.standard_normal((nops, nops))
    eri_full = np.zeros((nops, nops))
    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            qp = q * norb + p
            for r in range(norb):
                for s in range(norb):
                    rs = r * norb + s
                    sr = s * norb + r
                    val = 0.25 * (eri_raw[pq, rs] + eri_raw[qp, rs]
                                  + eri_raw[pq, sr] + eri_raw[qp, sr])
                    eri_full[pq, rs] = val
                    eri_full[qp, rs] = val
                    eri_full[pq, sr] = val
                    eri_full[qp, sr] = val

    # Full path: G = W @ eri_full
    G_full_ref = W_np @ eri_full

    # Sym-pair path with the pair-subsetted ERI
    eri_pair = eri_full[np.ix_(pair_pq, pair_pq)]
    W_pair_np = _reference_pack(W_np, pair_pq, pair_qp)
    G_pair_np = W_pair_np @ eri_pair
    G_sym = _reference_unpack(G_pair_np, full_to_pair)

    np.testing.assert_allclose(G_sym, G_full_ref, atol=1e-10, rtol=1e-10)

    # Same on GPU
    W_d = cp.asarray(W_np, dtype=cp.float64)
    W_pair_d = cp.empty((nrows, npair), dtype=cp.float64)

    sym_pair_pack_inplace_device(
        W_d, W_pair_d, cp.asarray(pair_pq), cp.asarray(pair_qp),
        nrows=nrows, nops=nops, npair=npair, sync=True,
    )

    eri_pair_d = cp.asarray(eri_pair, dtype=cp.float64)
    G_pair_d = cp.matmul(W_pair_d, eri_pair_d)

    G_d = cp.empty((nrows, nops), dtype=cp.float64)
    sym_pair_unpack_inplace_device(
        G_pair_d, G_d, cp.asarray(full_to_pair),
        nrows=nrows, nops=nops, npair=npair, sync=True,
    )

    G_gpu = cp.asnumpy(G_d)
    np.testing.assert_allclose(G_gpu, G_full_ref, atol=1e-10, rtol=1e-10)
