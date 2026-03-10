import numpy as np
import pytest


def test_fcifri_ground_accepts_key64_input_and_reference_on_host():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.cuda_backend import csf_idx_to_key64_host
    from asuka.qmc.fcifri import run_fcifri_ground

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(12)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)
    x_key = csf_idx_to_key64_host(drt, x_idx, state_cache=None)
    ref_key = int(csf_idx_to_key64_host(drt, np.asarray([1], dtype=np.int32), state_cache=None)[0])

    res_idx = run_fcifri_ground(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        m=64,
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=123,
        backend="auto",
        preferred_ref_idx=1,
    )
    res_key = run_fcifri_ground(
        drt,
        h1e,
        eri,
        None,
        x_val,
        m=64,
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=123,
        backend="auto",
        x_key=x_key,
        preferred_ref_key=ref_key,
    )

    assert res_key.idx.dtype == np.int32
    assert res_key.label_kind == "key64"
    assert res_key.key_u64 is not None and res_key.key_u64.dtype == np.uint64
    np.testing.assert_array_equal(res_key.idx, res_idx.idx)
    np.testing.assert_allclose(res_key.val, res_idx.val, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.energies, res_idx.energies, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(res_key.ref_idx, res_idx.ref_idx)


def test_fcifri_ground_accepts_idx64_zero_iter_on_host():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_ground

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)
    rng = np.random.default_rng(21)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )
    x_idx = np.asarray([0, 1, 2], dtype=np.int64)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fcifri_ground(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        m=64,
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=123,
        backend="cuda_idx64",
    )

    assert res.label_kind == "idx64"
    assert res.key_u64 is not None and res.key_u64.dtype == np.uint64
    np.testing.assert_array_equal(np.asarray(res.key_u64, dtype=np.int64), np.asarray(res.idx, dtype=np.int64))


@pytest.mark.cuda
def test_fcifri_ground_cuda_key64_smoke():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_ground

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

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fcifri_ground(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        m=64,
        eps=0.01,
        niter=2,
        nspawn_one=4,
        nspawn_two=3,
        seed=1234,
        backend="cuda_key64",
        pivot=16,
        initiator_na=0.0,
        energy_stride=1,
    )

    assert res.idx.dtype == np.int32
    assert res.val.dtype == np.float64
    assert res.idx.size == res.val.size
    assert res.idx.size > 0
    assert res.idx.size <= 64
    assert np.all(np.isfinite(res.val))
    assert np.all(res.idx[:-1] <= res.idx[1:])


@pytest.mark.cuda
def test_fcifri_ground_auto_smoke():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_ground

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

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fcifri_ground(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        m=64,
        eps=0.01,
        niter=2,
        nspawn_one=4,
        nspawn_two=3,
        seed=1234,
        backend="auto",
        pivot=16,
        initiator_na=0.0,
        energy_stride=1,
    )

    assert res.idx.dtype == np.int32
    assert res.val.dtype == np.float64
    assert res.idx.size == res.val.size
    assert res.idx.size > 0
    assert res.idx.size <= 64
    assert np.all(np.isfinite(res.val))
    assert np.all(res.idx[:-1] <= res.idx[1:])


def test_fcifri_ground_rejects_legacy_i32_backend():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fcifri import run_fcifri_ground

    drt = build_drt(norb=4, nelec=4, twos_target=0)
    h1e = np.zeros((4, 4), dtype=np.float64)
    eri = np.zeros((4, 4, 4, 4), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int32)
    x_val = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="backend must be 'auto', 'cuda_key64', or 'cuda_idx64'"):
        run_fcifri_ground(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            m=1,
            eps=0.01,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=7,
            backend="cuda_i32",
        )


@pytest.mark.cuda
def test_fciqmc_cuda_key64_smoke():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(1)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=0.01,
        niter=2,
        nspawn_one=4,
        nspawn_two=3,
        seed=4321,
        backend="cuda_key64",
        energy_stride=1,
        shift_damping=0.0,
    )

    assert res.idx.dtype == np.int32
    assert res.val.dtype == np.float64
    assert res.idx.size == res.val.size
    assert res.idx.size > 0
    assert np.all(np.isfinite(res.val))
    assert np.all(res.idx[:-1] <= res.idx[1:])
    assert res.sample_iters.shape == res.energies.shape
    assert res.energies_projected_fixed.shape == res.sample_iters.shape
    assert res.energies_projected_dynamic.shape == res.sample_iters.shape
    assert res.dynamic_ref_idx.shape == res.sample_iters.shape
    assert res.fixed_ref_alive.shape == res.sample_iters.shape
    assert np.all(np.isfinite(res.energies_projected_dynamic))
    assert int(res.fixed_ref_idx) == 0


@pytest.mark.cuda
def test_fciqmc_auto_smoke():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(1)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=0.01,
        niter=2,
        nspawn_one=4,
        nspawn_two=3,
        seed=4321,
        backend="auto",
        energy_stride=1,
        shift_damping=0.0,
    )

    assert res.idx.dtype == np.int32
    assert res.val.dtype == np.float64
    assert res.idx.size == res.val.size
    assert res.idx.size > 0
    assert np.all(np.isfinite(res.val))
    assert np.all(res.idx[:-1] <= res.idx[1:])
    assert res.sample_iters.shape == res.energies.shape
    assert res.energies_projected_fixed.shape == res.sample_iters.shape
    assert res.energies_projected_dynamic.shape == res.sample_iters.shape
    assert res.dynamic_ref_idx.shape == res.sample_iters.shape
    assert res.fixed_ref_alive.shape == res.sample_iters.shape
    assert np.all(np.isfinite(res.energies_projected_dynamic))
    assert int(res.fixed_ref_idx) == 0


def test_fciqmc_rejects_legacy_i32_backend():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    drt = build_drt(norb=4, nelec=4, twos_target=0)
    h1e = np.zeros((4, 4), dtype=np.float64)
    eri = np.zeros((4, 4, 4, 4), dtype=np.float64)
    x_idx = np.asarray([0], dtype=np.int32)
    x_val = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="backend must be 'auto', 'cuda_key64', or 'cuda_idx64'"):
        run_fciqmc(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            dt=0.01,
            niter=0,
            nspawn_one=1,
            nspawn_two=1,
            seed=7,
            backend="cuda_i32",
        )


def test_fciqmc_accepts_idx64_zero_iter_on_host():
    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)
    rng = np.random.default_rng(22)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )
    x_idx = np.asarray([0, 1, 2], dtype=np.int64)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)

    res = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=7,
        backend="cuda_idx64",
        energy_stride=1,
        shift_damping=0.0,
    )

    assert res.label_kind == "idx64"
    assert res.key_u64 is not None and res.key_u64.dtype == np.uint64
    np.testing.assert_array_equal(np.asarray(res.key_u64, dtype=np.int64), np.asarray(res.idx, dtype=np.int64))


@pytest.mark.cuda
def test_fciqmc_cuda_trial_anchored_det_subspace_smoke():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.cuguga.drt import build_drt
    from asuka.qmc.fciqmc import run_fciqmc

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(9)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )

    x_idx = np.asarray([0, 1, 2], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2, 0.05], dtype=np.float64)
    det_idx = np.asarray([0, 1], dtype=np.int32)

    for backend in ("cuda_key64", "auto"):
        res = run_fciqmc(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            dt=0.01,
            niter=2,
            nspawn_one=4,
            nspawn_two=3,
            seed=4321,
            backend=backend,
            energy_stride=1,
            shift_damping=0.0,
            trial_idx=x_idx,
            trial_val=x_val,
            reference_policy="fixed_trial_max_abs",
            deterministic_subspace_idx=det_idx,
        )

        assert res.sample_iters.shape == (3,)
        assert np.all(np.isfinite(res.energies_projected_fixed))
        assert np.all(np.isfinite(res.energies_projected_dynamic))
        assert np.all(res.fixed_ref_alive)
        assert np.all(res.det_subspace_l1_frac >= 0.0)
