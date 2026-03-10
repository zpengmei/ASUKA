import numpy as np
import pytest


@pytest.mark.cuda
def test_cuda_block_apply_right_matrix_phi_fused_matches_legacy_when_phi_is_copy():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    try:
        from asuka.qmc.cuda_backend import cuda_block_apply_right_matrix_phi_ws, make_cuda_block_projector_context
    except Exception as e:
        pytest.skip(f"QMC CUDA backend unavailable ({type(e).__name__}: {e})")

    from asuka.cuguga.drt import build_drt

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(0)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    nroots = 3
    m = 64
    pivot = 16

    ctx_a = make_cuda_block_projector_context(
        drt,
        h1e,
        eri,
        nroots=nroots,
        m=m,
        pivot=pivot,
        nspawn_one=1,
        nspawn_two=1,
    )
    ctx_b = make_cuda_block_projector_context(
        drt,
        h1e,
        eri,
        nroots=nroots,
        m=m,
        pivot=pivot,
        nspawn_one=1,
        nspawn_two=1,
    )
    ctx_c = make_cuda_block_projector_context(
        drt,
        h1e,
        eri,
        nroots=nroots,
        m=m,
        pivot=pivot,
        nspawn_one=1,
        nspawn_two=1,
    )

    try:
        # Disjoint supports so the coalesce result is insensitive to reduction order.
        cols = [
            (np.asarray([0, 1], dtype=np.int32), np.asarray([1.0, -0.2], dtype=np.float64)),
            (np.asarray([2, 3], dtype=np.int32), np.asarray([0.3, 0.4], dtype=np.float64)),
            (np.asarray([4, 5], dtype=np.int32), np.asarray([-0.5, 0.6], dtype=np.float64)),
        ]
        ctx_a.set_cols(cols)
        ctx_b.set_cols(cols)
        ctx_c.set_cols(cols)

        mat = np.asarray(
            [
                [1.0, 0.2, -0.1],
                [0.3, 1.0, 0.05],
                [-0.2, 0.1, 1.0],
            ],
            dtype=np.float64,
        )

        # Choose small nnz so the intermediate always satisfies n_in <= m, making Φ a pure copy.
        seeds = rng.integers(0, np.iinfo(np.int64).max, size=nroots * (nroots - 1), dtype=np.int64)

        cuda_block_apply_right_matrix_phi_ws(
            ctx_a,
            mat=mat,
            seeds_phi=seeds,
            sync=True,
            use_fused=True,
            use_composite_keys=True,
        )
        cuda_block_apply_right_matrix_phi_ws(
            ctx_b,
            mat=mat,
            seeds_phi=seeds,
            sync=True,
            use_fused=False,
        )
        cuda_block_apply_right_matrix_phi_ws(
            ctx_c,
            mat=mat,
            seeds_phi=seeds,
            sync=True,
            use_fused=True,
            use_composite_keys=False,
        )

        idx_a, val_a, nnz_a = ctx_a.get_cols_packed()
        idx_b, val_b, nnz_b = ctx_b.get_cols_packed()
        idx_c, val_c, nnz_c = ctx_c.get_cols_packed()

        assert np.array_equal(nnz_a, nnz_b)
        assert np.array_equal(nnz_a, nnz_c)
        for k in range(nroots):
            nnz = int(nnz_a[k])
            assert np.array_equal(idx_a[k, :nnz], idx_b[k, :nnz])
            np.testing.assert_allclose(val_a[k, :nnz], val_b[k, :nnz], rtol=0, atol=0)
            assert np.array_equal(idx_a[k, :nnz], idx_c[k, :nnz])
            np.testing.assert_allclose(val_a[k, :nnz], val_c[k, :nnz], rtol=0, atol=0)
    finally:
        ctx_a.release()
        ctx_b.release()
        ctx_c.release()
