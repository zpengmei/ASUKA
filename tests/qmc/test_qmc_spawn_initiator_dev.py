import numpy as np
import pytest


@pytest.mark.cuda
def test_qmc_spawn_hamiltonian_initiator_dev_matches_host_threshold():
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

    if not hasattr(ext, "qmc_spawn_hamiltonian_inplace_device_initiator_dev"):
        pytest.skip("device-initiator spawn entry point not available in extension")

    from asuka.cuguga.drt import build_drt
    from asuka.cuguga.oracle import _child_prefix_walks
    from asuka.cuguga.state_cache import get_state_cache

    # Tiny space; just enough to exercise the kernel path.
    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)
    cache = get_state_cache(drt)
    child_prefix = _child_prefix_walks(drt)

    drt_dev = ext.make_device_drt(
        int(drt.norb),
        np.asarray(drt.child, dtype=np.int32, order="C"),
        np.asarray(drt.node_twos, dtype=np.int16, order="C"),
        np.asarray(child_prefix, dtype=np.int64, order="C"),
    )
    state_dev = ext.make_device_state_cache(
        drt_dev,
        np.asarray(cache.steps, dtype=np.int8, order="C"),
        np.asarray(cache.nodes, dtype=np.int32, order="C"),
    )

    try:
        rng = np.random.default_rng(0)
        nops = norb * norb
        h_base_flat = cp.asarray(rng.normal(size=(nops,)).astype(np.float64), dtype=cp.float64)
        eri_mat = cp.asarray(rng.normal(size=(nops, nops)).astype(np.float64), dtype=cp.float64)

        # Two parents, one below threshold to exercise initiator gating.
        x_idx = cp.asarray(np.asarray([0, 1], dtype=np.int32))
        x_val = cp.asarray(np.asarray([1.0, 0.05], dtype=np.float64))
        m = int(x_idx.size)

        nspawn_one = 2
        nspawn_two = 1
        out_len = m * (nspawn_one + nspawn_two)

        out_idx_a = cp.empty(out_len, dtype=cp.int32)
        out_val_a = cp.empty(out_len, dtype=cp.float64)
        out_idx_b = cp.empty(out_len, dtype=cp.int32)
        out_val_b = cp.empty(out_len, dtype=cp.float64)

        eps = 0.01
        seed = 12345
        initiator_t = 0.1
        initiator_t_dev = cp.asarray(np.asarray(initiator_t, dtype=np.float64))

        stream = int(cp.cuda.get_current_stream().ptr)

        ext.qmc_spawn_hamiltonian_inplace_device(
            drt_dev,
            state_dev,
            x_idx,
            x_val,
            h_base_flat,
            eri_mat,
            out_idx_a,
            out_val_a,
            float(eps),
            int(nspawn_one),
            int(nspawn_two),
            int(seed),
            float(initiator_t),
            128,
            stream,
            True,
        )

        ext.qmc_spawn_hamiltonian_inplace_device_initiator_dev(
            drt_dev,
            state_dev,
            x_idx,
            x_val,
            h_base_flat,
            eri_mat,
            out_idx_b,
            out_val_b,
            float(eps),
            int(nspawn_one),
            int(nspawn_two),
            int(seed),
            initiator_t_dev,
            128,
            stream,
            True,
        )

        idx_a = cp.asnumpy(out_idx_a).astype(np.int32, copy=False)
        idx_b = cp.asnumpy(out_idx_b).astype(np.int32, copy=False)
        val_a = cp.asnumpy(out_val_a).astype(np.float64, copy=False)
        val_b = cp.asnumpy(out_val_b).astype(np.float64, copy=False)

        assert np.array_equal(idx_a, idx_b)
        assert np.array_equal(val_a, val_b)
    finally:
        state_dev.release()
        drt_dev.release()

