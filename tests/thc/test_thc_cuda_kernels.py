import pytest


pytestmark = pytest.mark.cuda


def _require_ext():
    cp = pytest.importorskip("cupy")
    try:
        from asuka import _hf_thc_cuda_ext as ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._hf_thc_cuda_ext is not available (build via python -m asuka.build.hf_thc_cuda_ext)")
    return cp, ext


def test_thc_rowwise_dot_f64_matches_cupy():
    cp, ext = _require_ext()
    rng = cp.random.default_rng(0)

    npt = 257
    nao = 193
    A = rng.standard_normal((npt, nao), dtype=cp.float64)
    X = rng.standard_normal((npt, nao), dtype=cp.float64)

    m_ref = cp.sum(A * X, axis=1)
    m = cp.empty((npt,), dtype=cp.float64)

    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    ext.rowwise_dot_f64(A, X, m, threads=256, stream_ptr=stream_ptr, sync=True)

    cp.testing.assert_allclose(m, m_ref, rtol=1e-12, atol=1e-12)


def test_thc_scale_rows_f64_matches_cupy():
    cp, ext = _require_ext()
    rng = cp.random.default_rng(1)

    npt = 123
    nao = 77
    X = rng.standard_normal((npt, nao), dtype=cp.float64)
    n = rng.standard_normal((npt,), dtype=cp.float64)

    out = cp.empty_like(X)
    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    ext.scale_rows_f64(X, n, out, threads=256, stream_ptr=stream_ptr, sync=True)

    ref = X * n[:, None]
    cp.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_thc_hadamard_inplace_f64_strided_Z_block_matches_cupy():
    cp, ext = _require_ext()
    rng = cp.random.default_rng(2)

    npt = 211
    nb = 33
    M0 = rng.standard_normal((npt, nb), dtype=cp.float64)
    M = M0.copy()

    # Use a column-slice view to exercise the strided-Z path (Z[:, q0:q1]).
    Zbig = rng.standard_normal((npt, nb + 7), dtype=cp.float64)
    Zblk = Zbig[:, 3 : 3 + nb]

    ref = M0 * Zblk

    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    ext.hadamard_inplace_f64(M, Zblk, threads=256, stream_ptr=stream_ptr, sync=True)

    cp.testing.assert_allclose(M, ref, rtol=1e-12, atol=1e-12)

