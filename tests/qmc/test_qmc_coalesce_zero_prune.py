import numpy as np
import pytest

from asuka.qmc.sparse import coalesce_coo_i32_f64


def test_qmc_coalesce_prunes_exact_zero_cpu():
    idx = np.asarray([7, 7], dtype=np.int32)
    val = np.asarray([1.0, -1.0], dtype=np.float64)
    idx_u, val_u = coalesce_coo_i32_f64(idx, val)
    assert idx_u.size == 0
    assert val_u.size == 0


def test_qmc_coalesce_prunes_zero_singleton_cpu():
    idx = np.asarray([3], dtype=np.int32)
    val = np.asarray([0.0], dtype=np.float64)
    idx_u, val_u = coalesce_coo_i32_f64(idx, val)
    assert idx_u.size == 0
    assert val_u.size == 0


@pytest.mark.cuda
def test_qmc_coalesce_prunes_exact_zero_gpu():
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

    ws = ext.QmcWorkspace(16, 16)

    idx_in = cp.asarray(np.asarray([7, 7], dtype=np.int32))
    val_in = cp.asarray(np.asarray([1.0, -1.0], dtype=np.float64))
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

    got = int(cp.asnumpy(out_nnz)[0])
    assert got == 0
