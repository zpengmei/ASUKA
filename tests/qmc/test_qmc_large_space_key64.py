from __future__ import annotations

import numpy as np
import pytest


def test_key64_roundtrip_preserves_large_csf_indices():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.cuda_backend import csf_idx_to_key64_host, key64_to_csf_idx64_host

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    idx = np.asarray([0, 17, int(drt.ncsf) - 1], dtype=np.int64)
    key = csf_idx_to_key64_host(drt, idx, state_cache=None)
    idx_rt = key64_to_csf_idx64_host(drt, key, strict=True)

    assert key.dtype == np.uint64
    assert idx_rt.dtype == np.int64
    assert np.array_equal(idx_rt, idx)


def test_selected_ci_rejects_large_space_dense_path():
    from asuka.cuguga.drt import build_drt
    from asuka.sci.selected_ci import selected_ci

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="selected_ci does not yet support|2\\^31-1"):
        selected_ci(
            drt,
            h1e,
            eri,
            nroots=1,
            init_ncsf=1,
            max_ncsf=1,
            add_ncsf=1,
            max_iter=1,
        )


def test_fciqmc_large_space_rejects_rayleigh_estimator_and_diagnostics():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int64)
    x_val = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="projected|rayleigh"):
        run_fciqmc(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            dt=1e-4,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=17,
            backend="cuda_key64",
            energy_estimator="rayleigh",
        )

    with pytest.raises(ValueError, match="rayleigh_stride"):
        run_fciqmc(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            dt=1e-4,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=17,
            backend="cuda_key64",
            energy_estimator="projected",
            rayleigh_stride=1,
        )


def test_fcifri_large_space_rejects_rayleigh_and_removes_legacy_subspace():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_block, run_fcifri_ground, run_fcifri_subspace
    from asuka.qmc.rsi import run_fcifri_rsi

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int64)
    x_val = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="projected|rayleigh"):
        run_fcifri_ground(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            m=4,
            eps=0.01,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=19,
            backend="cuda_key64",
            energy_estimator="rayleigh",
        )

    block = run_fcifri_block(
        drt,
        h1e,
        eri,
        nroots=1,
        m=1,
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=31,
        backend="auto",
        x0=[(x_idx, x_val)],
    )
    assert block.energies.shape == (1, 1)
    assert block.idx and block.idx[0].dtype == np.int64
    assert block.val and block.val[0].dtype == np.float64

    with pytest.raises(NotImplementedError, match="removed from the scalable production path"):
        run_fcifri_subspace(
            drt,
            h1e,
            eri,
            nroots=1,
            m=1,
            eps=0.01,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=23,
            method="rsi",
            x0=[(x_idx, x_val)],
        )

    with pytest.raises(NotImplementedError, match="removed from the scalable production path"):
        run_fcifri_rsi(
            drt,
            h1e,
            eri,
            U0=[(x_idx, x_val)],
            m=1,
            eps=0.01,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=29,
        )


def test_low_level_cpu_qmc_helpers_reject_large_space_labels():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.compress import compress_phi_pivot_resample, compress_phi_pivotal
    from asuka.qmc.compress_guided import compress_phi_pivot_resample_guided
    from asuka.qmc.epq_sample import sample_epq_from_arrays, sample_epq_one
    from asuka.qmc.projector import projector_step
    from asuka.qmc.spawn_guided import spawn_hamiltonian_events_guided_row
    from asuka.qmc.spawn import spawn_hamiltonian_events

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int64)
    x_val = np.asarray([1.0], dtype=np.float64)
    x_idx_big = np.asarray([np.iinfo(np.int32).max + 5], dtype=np.int64)
    rng = np.random.default_rng(31)

    with pytest.raises(NotImplementedError, match="large-space / key64 labels"):
        spawn_hamiltonian_events(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            eps=0.01,
            nspawn_one=1,
            nspawn_two=1,
            rng=rng,
        )

    with pytest.raises(NotImplementedError, match="large-space / key64 labels"):
        projector_step(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            eps=0.01,
            nspawn_one=1,
            nspawn_two=1,
            rng=rng,
        )

    with pytest.raises(NotImplementedError, match="int32-addressable labels"):
        compress_phi_pivot_resample(x_idx_big, x_val, m=1, rng=rng)

    with pytest.raises(NotImplementedError, match="int32-addressable labels"):
        compress_phi_pivotal(x_idx_big, x_val, m=1, rng=rng)

    with pytest.raises(NotImplementedError, match="int32-addressable labels"):
        compress_phi_pivot_resample_guided(
            x_idx_big,
            x_val,
            m=1,
            rng=rng,
            logq_fn=lambda idx: np.zeros_like(np.asarray(idx, dtype=np.float64)),
            alpha=0.5,
        )

    with pytest.raises(NotImplementedError, match="large-space / key64 labels"):
        spawn_hamiltonian_events_guided_row(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            eps=0.01,
            nspawn_one=1,
            nspawn_two=1,
            rng=rng,
        )

    with pytest.raises(NotImplementedError, match="int32-addressable labels"):
        sample_epq_from_arrays(x_idx_big, np.asarray([1.0], dtype=np.float64), rng)

    with pytest.raises(NotImplementedError, match="large-space / key64 labels"):
        sample_epq_one(drt, int(x_idx[0]), 0, 0, rng)


def test_root_package_surfaces_prefer_scalable_entrypoints():
    import asuka.qmc as qmc
    import asuka.sci as sci

    assert hasattr(qmc, "run_fciqmc")
    assert hasattr(qmc, "run_fcifri_ground")
    assert not hasattr(qmc, "run_fcifri_subspace")
    assert not hasattr(qmc, "run_fcifri_rsi")
    assert not hasattr(qmc, "spawn_hamiltonian_events")
    assert not hasattr(qmc, "projector_step")

    assert hasattr(sci, "run_cipsi_trials")
    assert hasattr(sci, "heat_bath_select_and_pt2_sparse")
    assert not hasattr(sci, "heat_bath_select_and_pt2")
    assert not hasattr(sci, "semistochastic_pt2")


@pytest.mark.cuda
def test_fciqmc_cuda_key64_large_space_zero_iter_smoke():
    cp = pytest.importorskip("cupy")
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    norb = 28
    drt = build_drt(norb=norb, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    rng = np.random.default_rng(7)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    x_idx = np.asarray([0, int(drt.ncsf) - 1], dtype=np.int64)
    x_val = np.asarray([1.0, -0.1], dtype=np.float64)

    res = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=1e-4,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=17,
        backend="cuda_key64",
        energy_stride=1,
        shift_damping=0.0,
        max_walker=16,
    )

    assert res.idx.dtype == np.int64
    assert res.label_kind == "key64"
    assert res.key_u64 is not None and res.key_u64.dtype == np.uint64
    assert res.ref_idx.dtype == np.int64
    assert np.array_equal(res.idx, x_idx)
