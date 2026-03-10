import numpy as np
import pytest


def _dot_sorted(idx_a: np.ndarray, val_a: np.ndarray, idx_b: np.ndarray, val_b: np.ndarray) -> float:
    ia = 0
    ib = 0
    na = int(idx_a.size)
    nb = int(idx_b.size)
    s = 0.0
    while ia < na and ib < nb:
        a = int(idx_a[ia])
        b = int(idx_b[ib])
        if a == b:
            s += float(val_a[ia]) * float(val_b[ib])
            ia += 1
            ib += 1
        elif a < b:
            ia += 1
        else:
            ib += 1
    return float(s)


@pytest.mark.cuda
def test_qmc_sparse_dot_many_sorted_matches_cpu():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _guga_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"QMC CUDA extension not available ({type(e).__name__}: {e})")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    if not hasattr(ext.QmcWorkspace, "sparse_dot_many_sorted_i32_f64_inplace_device"):
        pytest.skip("batched sparse dot entry point not available in extension")

    rng = np.random.default_rng(0)

    nA = 5
    nB = 4
    max_a = 17
    max_b = 19
    dim = 64

    a_nnz = rng.integers(low=0, high=max_a + 1, size=nA, dtype=np.int32)
    b_nnz = rng.integers(low=0, high=max_b + 1, size=nB, dtype=np.int32)

    a_idx = np.zeros((nA, max_a), dtype=np.int32)
    a_val = np.zeros((nA, max_a), dtype=np.float64)
    b_idx = np.zeros((nB, max_b), dtype=np.int32)
    b_val = np.zeros((nB, max_b), dtype=np.float64)

    for i in range(nA):
        nnz = int(a_nnz[i])
        if nnz == 0:
            continue
        idx = np.sort(rng.choice(dim, size=nnz, replace=False).astype(np.int32, copy=False))
        val = rng.normal(size=nnz).astype(np.float64, copy=False)
        a_idx[i, :nnz] = idx
        a_val[i, :nnz] = val

    for j in range(nB):
        nnz = int(b_nnz[j])
        if nnz == 0:
            continue
        idx = np.sort(rng.choice(dim, size=nnz, replace=False).astype(np.int32, copy=False))
        val = rng.normal(size=nnz).astype(np.float64, copy=False)
        b_idx[j, :nnz] = idx
        b_val[j, :nnz] = val

    expected = np.zeros((nA, nB), dtype=np.float64)
    for i in range(nA):
        for j in range(nB):
            na = int(a_nnz[i])
            nb = int(b_nnz[j])
            expected[i, j] = _dot_sorted(a_idx[i, :na], a_val[i, :na], b_idx[j, :nb], b_val[j, :nb])

    ws = ext.QmcWorkspace(128, 64)
    try:
        a_idx_dev = cp.asarray(a_idx, dtype=cp.int32)
        a_val_dev = cp.asarray(a_val, dtype=cp.float64)
        b_idx_dev = cp.asarray(b_idx, dtype=cp.int32)
        b_val_dev = cp.asarray(b_val, dtype=cp.float64)
        a_nnz_dev = cp.asarray(a_nnz, dtype=cp.int32)
        b_nnz_dev = cp.asarray(b_nnz, dtype=cp.int32)

        out = cp.empty((nA, nB), dtype=cp.float64)
        stream = int(cp.cuda.get_current_stream().ptr)

        ws.sparse_dot_many_sorted_i32_f64_inplace_device(
            a_idx_dev,
            a_val_dev,
            a_nnz_dev,
            b_idx_dev,
            b_val_dev,
            b_nnz_dev,
            out,
            256,
            stream,
            True,
        )

        got = cp.asnumpy(out).astype(np.float64, copy=False)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

        # Also test the 1D "single-vector" B interface used by hk dot products.
        out2 = cp.empty((nA, 1), dtype=cp.float64)
        ws.sparse_dot_many_sorted_i32_f64_inplace_device(
            a_idx_dev,
            a_val_dev,
            a_nnz_dev,
            b_idx_dev[0],
            b_val_dev[0],
            b_nnz_dev[:1],
            out2,
            256,
            stream,
            True,
        )
        got2 = cp.asnumpy(out2).astype(np.float64, copy=False)
        np.testing.assert_allclose(got2[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    finally:
        ws.release()

