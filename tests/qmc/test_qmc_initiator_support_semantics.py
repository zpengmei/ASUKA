import numpy as np
import pytest

from asuka.qmc.sparse import coalesce_coo_i32_f64


def test_qmc_support_absent_after_exact_cancellation_cpu():
    # A determinant whose contributions cancel to exactly zero must not remain in sparse support.
    idx = np.asarray([5, 5, 9], dtype=np.int32)
    val = np.asarray([0.25, -0.25, 1.0], dtype=np.float64)
    idx_u, val_u = coalesce_coo_i32_f64(idx, val)

    assert np.all(val_u != 0.0)
    assert 5 not in set(idx_u.tolist())
    assert 9 in set(idx_u.tolist())


@pytest.mark.cuda
def test_qmc_support_absent_after_exact_cancellation_gpu():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _guga_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"QMC CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "QmcWorkspace"):
        pytest.skip("QmcWorkspace not available in extension")
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    ws = ext.QmcWorkspace(32, 32)

    idx_in = cp.asarray(np.asarray([5, 5, 9], dtype=np.int32))
    val_in = cp.asarray(np.asarray([0.25, -0.25, 1.0], dtype=np.float64))

    idx_out = cp.empty_like(idx_in)
    val_out = cp.empty_like(val_in)
    out_nnz = cp.empty(1, dtype=cp.int32)

    ws.coalesce_coo_i32_f64_inplace_device(
        idx_in,
        val_in,
        idx_out,
        val_out,
        out_nnz,
        n=int(idx_in.size),
        threads=128,
        stream=int(cp.cuda.get_current_stream().ptr),
        sync=True,
    )

    nnz = int(cp.asnumpy(out_nnz)[0])
    idx_h = cp.asnumpy(idx_out[:nnz]).astype(np.int32, copy=False)
    val_h = cp.asnumpy(val_out[:nnz]).astype(np.float64, copy=False)

    assert np.all(val_h != 0.0)
    assert 5 not in set(idx_h.tolist())
    assert 9 in set(idx_h.tolist())
