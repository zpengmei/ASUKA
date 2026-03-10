import numpy as np
import pytest


@pytest.mark.cuda
def test_qmc_coalesce_prunes_exact_zero_gpu_u64():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _guga_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"QMC CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "QmcWorkspaceU64"):
        pytest.skip("QmcWorkspaceU64 not available in extension")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    ws = ext.QmcWorkspaceU64(16, 16)

    key_in = cp.asarray(np.asarray([7, 7], dtype=np.uint64))
    val_in = cp.asarray(np.asarray([1.0, -1.0], dtype=np.float64))
    key_out = cp.empty_like(key_in)
    val_out = cp.empty_like(val_in)
    out_nnz = cp.empty(1, dtype=cp.int32)

    ws.coalesce_coo_u64_f64_inplace_device(
        key_in,
        val_in,
        key_out,
        val_out,
        out_nnz,
        n=int(key_in.size),
        threads=128,
        stream=int(cp.cuda.get_current_stream().ptr),
        sync=True,
    )

    got = int(cp.asnumpy(out_nnz)[0])
    assert got == 0


@pytest.mark.cuda
def test_qmc_phi_u64_copies_when_n_le_m_gpu():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _guga_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"QMC CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "QmcWorkspaceU64"):
        pytest.skip("QmcWorkspaceU64 not available in extension")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    n_in = 5
    m = 8
    ws = ext.QmcWorkspaceU64(64, 64)

    key_in = cp.asarray(np.asarray([17, 3, 99, 42, 5], dtype=np.uint64))
    val_in = cp.asarray(np.asarray([1.0, -2.0, 3.0, -4.0, 0.5], dtype=np.float64))
    key_out = cp.empty(m, dtype=cp.uint64)
    val_out = cp.empty(m, dtype=cp.float64)
    out_nnz = cp.empty(1, dtype=cp.int32)

    ws.phi_pivot_resample_u64_f64_inplace_device(
        key_in,
        val_in,
        key_out,
        val_out,
        out_nnz,
        n_in=int(n_in),
        m=int(m),
        pivot=0,
        seed=123,
        threads=128,
        stream=int(cp.cuda.get_current_stream().ptr),
        sync=True,
    )

    got = int(cp.asnumpy(out_nnz)[0])
    assert got == n_in
    assert np.array_equal(cp.asnumpy(key_out[:n_in]).astype(np.uint64, copy=False), np.asarray([17, 3, 99, 42, 5], dtype=np.uint64))
    assert np.array_equal(cp.asnumpy(val_out[:n_in]).astype(np.float64, copy=False), np.asarray([1.0, -2.0, 3.0, -4.0, 0.5], dtype=np.float64))


@pytest.mark.cuda
def test_qmc_phi_u64_pivots_only_matches_reference_gpu():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _guga_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"QMC CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "QmcWorkspaceU64"):
        pytest.skip("QmcWorkspaceU64 not available in extension")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    # Deterministic path: pivot == m => no sampling.
    key_in_h = np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint64)
    val_in_h = np.asarray([1.0, -8.0, 2.0, 4.0, -0.5, 16.0, 0.25, -32.0], dtype=np.float64)
    n_in = int(key_in_h.size)
    m = 4
    pivot = 4

    # Top-|val| pivots: 17(-32), 15(16), 11(-8), 13(4). Coalesce sorts by key.
    exp_key_h = np.asarray([11, 13, 15, 17], dtype=np.uint64)
    exp_val_h = np.asarray([-8.0, 4.0, 16.0, -32.0], dtype=np.float64)

    ws = ext.QmcWorkspaceU64(64, 64)
    key_in = cp.asarray(key_in_h)
    val_in = cp.asarray(val_in_h)
    key_out = cp.empty(m, dtype=cp.uint64)
    val_out = cp.empty(m, dtype=cp.float64)
    out_nnz = cp.empty(1, dtype=cp.int32)

    ws.phi_pivot_resample_u64_f64_inplace_device(
        key_in,
        val_in,
        key_out,
        val_out,
        out_nnz,
        n_in=int(n_in),
        m=int(m),
        pivot=int(pivot),
        seed=0,
        threads=128,
        stream=int(cp.cuda.get_current_stream().ptr),
        sync=True,
    )

    got = int(cp.asnumpy(out_nnz)[0])
    assert got == m
    assert np.array_equal(cp.asnumpy(key_out[:m]).astype(np.uint64, copy=False), exp_key_h)
    assert np.array_equal(cp.asnumpy(val_out[:m]).astype(np.float64, copy=False), exp_val_h)

