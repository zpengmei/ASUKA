import numpy as np
import pytest


@pytest.mark.cuda
def test_hf_df_jk_cuda_ext_symmetrize_inplace_matches_reference():
    cp = pytest.importorskip("cupy")
    try:
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    try:
        from asuka import _hf_df_jk_cuda_ext as ext  # noqa: PLC0415
    except Exception:
        pytest.skip("HF DF-JK CUDA extension is unavailable")

    if not hasattr(ext, "symmetrize_inplace_f64"):
        pytest.skip("HF DF-JK CUDA extension lacks symmetrize_inplace_f64")

    rng = np.random.default_rng(123)
    n = 128
    a0 = rng.standard_normal((n, n), dtype=np.float64)
    a = cp.asarray(a0, dtype=cp.float64)
    ref = 0.5 * (a + a.T)

    work = a.copy()
    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    ext.symmetrize_inplace_f64(work, stream_ptr, True)

    # Exact equality is not required; the reference uses CuPy elementwise kernels.
    assert bool(cp.allclose(work, ref, rtol=0.0, atol=0.0))

