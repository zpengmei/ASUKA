import numpy as np
import pytest


@pytest.mark.cuda
def test_cuda_projector_step_fused_matches_unfused():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    try:
        from asuka.qmc.cuda_backend import cuda_projector_step_hamiltonian_ws, make_cuda_projector_context
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

    m = 64
    pivot = 16
    nspawn_one = 4
    nspawn_two = 3

    ctx_a = make_cuda_projector_context(
        drt,
        h1e,
        eri,
        m=m,
        pivot=pivot,
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
    )
    ctx_b = make_cuda_projector_context(
        drt,
        h1e,
        eri,
        m=m,
        pivot=pivot,
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
    )

    try:
        x_idx = np.asarray([0, 1, 2], dtype=np.int32)
        x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)
        nnz0 = int(x_idx.size)

        ctx_a.x_idx[:nnz0] = cp.asarray(x_idx)
        ctx_a.x_val[:nnz0] = cp.asarray(x_val)
        ctx_a.nnz = nnz0

        ctx_b.x_idx[:nnz0] = cp.asarray(x_idx)
        ctx_b.x_val[:nnz0] = cp.asarray(x_val)
        ctx_b.nnz = nnz0

        eps = 0.01
        seed_spawn = 12345
        seed_phi = 54321
        initiator_t_dev = cp.asarray(np.asarray(0.1, dtype=np.float64))

        cuda_projector_step_hamiltonian_ws(
            ctx_a,
            eps=eps,
            initiator_t=0.0,
            initiator_t_dev=initiator_t_dev,
            seed_spawn=seed_spawn,
            seed_phi=seed_phi,
            scale_identity=1.0,
            sync=True,
            use_fused=True,
        )

        cuda_projector_step_hamiltonian_ws(
            ctx_b,
            eps=eps,
            initiator_t=0.0,
            initiator_t_dev=initiator_t_dev,
            seed_spawn=seed_spawn,
            seed_phi=seed_phi,
            scale_identity=1.0,
            sync=True,
            use_fused=False,
        )

        assert int(ctx_a.nnz) == int(ctx_b.nnz)
        nnz = int(ctx_a.nnz)
        idx_a = cp.asnumpy(ctx_a.x_idx[:nnz]).astype(np.int32, copy=False)
        idx_b = cp.asnumpy(ctx_b.x_idx[:nnz]).astype(np.int32, copy=False)
        val_a = cp.asnumpy(ctx_a.x_val[:nnz]).astype(np.float64, copy=False)
        val_b = cp.asnumpy(ctx_b.x_val[:nnz]).astype(np.float64, copy=False)

        assert np.array_equal(idx_a, idx_b)
        assert np.array_equal(val_a, val_b)
    finally:
        ctx_a.release()
        ctx_b.release()

