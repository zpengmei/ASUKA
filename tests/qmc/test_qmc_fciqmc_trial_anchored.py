from __future__ import annotations

import numpy as np


def _toy_case(seed: int = 0):
    from asuka.cuguga.drt import build_drt

    norb = 4
    drt = build_drt(norb=norb, nelec=4, twos_target=0)

    rng = np.random.default_rng(seed)
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
    )
    return drt, h1e, eri


def test_fciqmc_fixed_idx_strict_reports_nan_instead_of_falling_back():
    from asuka.qmc.fciqmc import run_fciqmc

    drt, h1e, eri = _toy_case(seed=1)
    x_idx = np.asarray([0, 1], dtype=np.int32)
    x_val = np.asarray([1.0, -0.2], dtype=np.float64)

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
        seed=11,
        reference_policy="fixed_idx_strict",
        preferred_ref_idx=3,
    )

    assert res.fixed_ref_idx == 3
    assert bool(res.fixed_ref_alive[0]) is False
    assert np.isnan(res.energies_projected_fixed[0])
    assert np.isnan(res.energies[0])
    assert int(res.dynamic_ref_idx[0]) == 0
    assert np.isfinite(res.energies_projected_dynamic[0])


def test_fciqmc_trial_diagnostics_are_scale_invariant():
    from asuka.qmc.fciqmc import run_fciqmc

    drt, h1e, eri = _toy_case(seed=2)
    x_idx = np.asarray([0, 2, 3], dtype=np.int32)
    x_val = np.asarray([1.0, -0.25, 0.1], dtype=np.float64)
    trial_idx = np.asarray([0, 2, 3], dtype=np.int32)
    trial_val = np.asarray([0.8, -0.3, 0.15], dtype=np.float64)
    det_idx = np.asarray([0, 3], dtype=np.int32)

    res_a = run_fciqmc(
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
        trial_idx=trial_idx,
        trial_val=trial_val,
        deterministic_subspace_idx=det_idx,
    )
    res_b = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        17.0 * x_val,
        dt=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=7,
        trial_idx=trial_idx,
        trial_val=trial_val,
        deterministic_subspace_idx=det_idx,
    )

    np.testing.assert_allclose(res_a.trial_cosine, res_b.trial_cosine, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_a.trial_support_l1_frac, res_b.trial_support_l1_frac, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_a.det_subspace_l1_frac, res_b.det_subspace_l1_frac, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_a.energies_projected_fixed, res_b.energies_projected_fixed, rtol=0.0, atol=1e-12)


def test_fciqmc_accepts_key64_inputs_for_state_trial_ref_and_det_subspace():
    from asuka.qmc.cuda_backend import csf_idx_to_key64_host
    from asuka.qmc.fciqmc import run_fciqmc

    drt, h1e, eri = _toy_case(seed=3)
    x_idx = np.asarray([0, 2, 3], dtype=np.int32)
    x_val = np.asarray([1.0, -0.25, 0.1], dtype=np.float64)
    trial_idx = np.asarray([0, 2, 3], dtype=np.int32)
    trial_val = np.asarray([0.8, -0.3, 0.15], dtype=np.float64)
    det_idx = np.asarray([0, 3], dtype=np.int32)

    x_key = csf_idx_to_key64_host(drt, x_idx, state_cache=None)
    trial_key = csf_idx_to_key64_host(drt, trial_idx, state_cache=None)
    preferred_ref_key = int(csf_idx_to_key64_host(drt, np.asarray([3], dtype=np.int32), state_cache=None)[0])
    det_key = csf_idx_to_key64_host(drt, det_idx, state_cache=None)

    res_idx = run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=9,
        reference_policy="fixed_idx_strict",
        preferred_ref_idx=3,
        trial_idx=trial_idx,
        trial_val=trial_val,
        deterministic_subspace_idx=det_idx,
    )
    res_key = run_fciqmc(
        drt,
        h1e,
        eri,
        None,
        x_val,
        dt=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=9,
        x_key=x_key,
        reference_policy="fixed_idx_strict",
        preferred_ref_key=preferred_ref_key,
        trial_key=trial_key,
        trial_val=trial_val,
        deterministic_subspace_key=det_key,
    )

    assert res_key.idx.dtype == np.int32
    np.testing.assert_array_equal(res_key.idx, res_idx.idx)
    np.testing.assert_allclose(res_key.val, res_idx.val, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.energies, res_idx.energies, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.energies_projected_fixed, res_idx.energies_projected_fixed, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.energies_projected_dynamic, res_idx.energies_projected_dynamic, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.trial_cosine, res_idx.trial_cosine, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.trial_support_l1_frac, res_idx.trial_support_l1_frac, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_key.det_subspace_l1_frac, res_idx.det_subspace_l1_frac, rtol=0.0, atol=1e-12)
