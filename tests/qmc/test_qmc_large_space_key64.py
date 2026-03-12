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


def test_selected_ci_symbol_is_absent():
    import importlib
    import asuka.sci as sci

    assert not hasattr(sci, "selected_ci")
    assert importlib.util.find_spec("asuka.sci.selected_ci") is None


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
    from asuka.qmc.fcifri import run_fcifri_block, run_fcifri_ground
    import asuka.qmc.fcifri as qmc_fcifri

    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int64)
    x_val = np.asarray([1.0], dtype=np.float64)
    x0_block = [
        (np.asarray([0], dtype=np.int64), np.asarray([1.0], dtype=np.float64)),
        (np.asarray([1], dtype=np.int64), np.asarray([1.0], dtype=np.float64)),
    ]

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
        nroots=2,
        m=1,
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=31,
        backend="auto",
        x0=x0_block,
    )
    assert block.energies.shape == (1, 2)
    assert len(block.idx) == 2
    assert len(block.val) == 2
    assert all(col.dtype == np.int64 for col in block.idx)
    assert all(col.dtype == np.float64 for col in block.val)

    import importlib

    assert not hasattr(qmc_fcifri, "run_fcifri_subspace")
    assert importlib.util.find_spec("asuka.qmc.rsi") is None


def test_low_level_cpu_qmc_helper_modules_are_absent():
    import importlib

    removed_modules = [
        "asuka.qmc.compress",
        "asuka.qmc.compress_guided",
        "asuka.qmc.epq_sample",
        "asuka.qmc.projector",
        "asuka.qmc.spawn",
        "asuka.qmc.spawn_guided",
        "asuka.qmc.subspace",
    ]

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None, module_name


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


@pytest.mark.cuda
def test_fcifri_block_cuda_key64_large_space_multi_root_smoke():
    cp = pytest.importorskip("cupy")
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_block

    norb = 28
    drt = build_drt(norb=norb, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    x0 = [
        (np.asarray([0], dtype=np.int64), np.asarray([1.0], dtype=np.float64)),
        (np.asarray([1], dtype=np.int64), np.asarray([1.0], dtype=np.float64)),
    ]

    res = run_fcifri_block(
        drt,
        h1e,
        eri,
        nroots=2,
        m=2,
        eps=1e-4,
        niter=1,
        nspawn_one=1,
        nspawn_two=1,
        seed=23,
        backend="cuda_key64",
        x0=x0,
        ortho_stride=1,
        ritz_stride=1,
    )

    assert res.backend == "cuda_key64"
    assert res.energies.shape == (2, 2)
    assert res.iters.tolist() == [0, 1]
    assert len(res.idx) == 2
    assert all(col.dtype == np.int64 for col in res.idx)
    assert all(col.size >= 1 for col in res.idx)
