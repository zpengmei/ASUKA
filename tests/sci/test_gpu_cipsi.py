from __future__ import annotations

import numpy as np
import pytest

from asuka.cuguga.drt import build_drt
from asuka.qmc.fcifri import run_fcifri_block
from asuka.qmc.sparse import SparseVector
from asuka.sci.gpu_cipsi import CIPSITrialSpaceResult, build_cipsi_trials_from_scf, run_cipsi_trials


def _make_symmetric_test_integrals(norb: int, seed: int = 7):
    rng = np.random.default_rng(int(seed))
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(2, 3, 0, 1)
    )
    return np.asarray(h1e, dtype=np.float64), np.asarray(eri, dtype=np.float64)


def _require_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) <= 0:
            pytest.skip("no CUDA device available")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable: {e}")
    return cupy


def _cupy_asarray_or_skip(cupy, arr):
    try:
        return cupy.asarray(arr)
    except Exception as e:  # pragma: no cover - runtime/device specific
        if "misaligned address" in str(e).lower():
            pytest.skip(f"CUDA runtime unstable for device transfer: {e}")
        raise


def test_cipsi_trial_result_roundtrip(tmp_path):
    sel_idx = np.asarray([0, 2, 5], dtype=np.int32)
    ci_sel = np.asarray(
        [
            [0.75, 0.00],
            [-0.25, 0.50],
            [0.00, -0.60],
        ],
        dtype=np.float64,
    )
    roots = [
        SparseVector(np.asarray([0, 2], dtype=np.int32), np.asarray([0.75, -0.25], dtype=np.float64)),
        SparseVector(np.asarray([2, 5], dtype=np.int32), np.asarray([0.50, -0.60], dtype=np.float64)),
    ]
    res = CIPSITrialSpaceResult(
        e_var=np.asarray([-1.0, -0.5], dtype=np.float64),
        e_pt2=np.asarray([-0.1, -0.05], dtype=np.float64),
        e_tot=np.asarray([-1.1, -0.55], dtype=np.float64),
        sel_idx=sel_idx,
        ci_sel=ci_sel,
        roots=roots,
        history=[{"iter": 1, "nsel": 3}],
        profile={"epq_mode": "materialized_epq"},
        epq_mode="materialized_epq",
        ncsf=6,
    )

    x0 = res.to_qmc_x0()
    assert len(x0) == 2
    assert np.array_equal(x0[0][0], np.asarray([0, 2], dtype=np.int32))
    assert np.allclose(x0[1][1], np.asarray([0.50, -0.60], dtype=np.float64))

    out = tmp_path / "trial_space.npz"
    res.save(out)
    loaded = CIPSITrialSpaceResult.load(out)
    assert loaded.epq_mode == "materialized_epq"
    assert loaded.ncsf == 6
    assert np.allclose(loaded.e_tot, res.e_tot)
    assert np.array_equal(loaded.sel_idx, res.sel_idx)
    assert len(loaded.roots) == 2
    assert np.array_equal(loaded.roots[0].idx, roots[0].idx)
    assert np.allclose(loaded.roots[1].val, roots[1].val)


def test_cipsi_trial_result_roundtrip_preserves_int64_sel_idx(tmp_path):
    base = np.int64(np.iinfo(np.int32).max) + np.int64(11)
    sel_idx = np.asarray([base, base + 4], dtype=np.int64)
    sel_key_u64 = np.asarray([17, 29], dtype=np.uint64)
    ci_sel = np.asarray([[1.0], [-0.25]], dtype=np.float64)
    roots = [SparseVector(sel_idx.copy(), np.asarray([1.0, -0.25], dtype=np.float64))]
    res = CIPSITrialSpaceResult(
        e_var=np.asarray([-1.0], dtype=np.float64),
        e_pt2=np.asarray([-0.1], dtype=np.float64),
        e_tot=np.asarray([-1.1], dtype=np.float64),
        sel_idx=sel_idx,
        ci_sel=ci_sel,
        roots=roots,
        history=[],
        profile={},
        epq_mode="no_epq_support_aware",
        ncsf=int(base + 5),
        sel_key_u64=sel_key_u64,
        label_kind="key64",
    )

    out = tmp_path / "trial_space_i64.npz"
    res.save(out)
    loaded = CIPSITrialSpaceResult.load(out)

    assert loaded.sel_idx.dtype == np.int64
    assert np.array_equal(loaded.sel_idx, sel_idx)
    assert loaded.roots[0].idx.dtype == np.int64
    assert np.array_equal(loaded.roots[0].idx, sel_idx)
    assert loaded.sel_key_u64 is not None
    assert loaded.label_kind == "key64"
    assert np.array_equal(loaded.sel_key_u64, sel_key_u64)


def test_run_cipsi_trials_large_space_key64_smoke():
    drt = build_drt(norb=28, nelec=12, twos_target=0)
    assert int(drt.ncsf) > np.iinfo(np.int32).max

    norb = int(drt.norb)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)

    x_idx = np.asarray([0], dtype=np.int64)
    x_val = np.asarray([1.0], dtype=np.float64)
    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        ci0=[(x_idx, x_val)],
        init_ncsf=1,
        max_ncsf=1,
        grow_by=0,
        max_iter=0,
        epq_mode="no_epq_support_aware",
        selection_mode="heat_bath",
        hb_epsilon=0.0,
        state_rep="key64",
    )

    assert res.sel_idx.dtype == np.int64
    assert res.sel_key_u64 is not None and res.sel_key_u64.dtype == np.uint64
    assert res.label_kind == "key64"
    assert len(res.roots) == 1
    assert res.roots[0].idx.dtype == np.int64
    assert np.all(np.isfinite(res.e_var))


def test_legacy_frontier_selector_is_explicitly_removed():
    from asuka.sci.frontier_hash import FrontierHashSelector

    with pytest.raises(NotImplementedError, match="removed from the supported path"):
        FrontierHashSelector()


@pytest.mark.parametrize("epq_mode", ["no_epq_support_aware"])
@pytest.mark.parametrize("state_rep", ["auto", "key64"])
@pytest.mark.cuda
def test_run_cipsi_trials_gpu_smoke_and_fcifri_hook(epq_mode, state_rep):
    cupy = _require_cuda()

    drt = build_drt(norb=2, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(2)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=2,
        max_ncsf=int(drt.ncsf),
        grow_by=1,
        max_iter=2,
        epq_mode=epq_mode,
        state_rep=state_rep,
        davidson_max_cycle=8,
        davidson_max_space=4,
        davidson_tol=1e-6,
    )

    assert res.epq_mode == epq_mode
    assert res.sel_idx.ndim == 1
    if state_rep == "key64":
        assert res.sel_key_u64 is not None
        assert res.label_kind == "key64"
        assert res.sel_key_u64.shape == res.sel_idx.shape
    else:
        assert res.sel_key_u64 is None
        assert res.label_kind == "csf_idx"
    assert res.ci_sel.shape[1] == 1
    assert len(res.roots) == 1
    assert isinstance(res.roots[0], SparseVector)
    assert np.all(np.diff(res.roots[0].idx) > 0) if res.roots[0].idx.size > 1 else True

    fri = run_fcifri_block(
        drt,
        h1e,
        eri,
        nroots=1,
        m=max(1, int(res.roots[0].nnz)),
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=11,
        backend="auto",
        x0=res,
        rsi_min_nsample=0,
    )
    assert fri.energies.shape == (1, 1)


@pytest.mark.cuda
def test_run_cipsi_trials_gpu_frontier_hash_smoke():
    cupy = _require_cuda()

    from asuka.cuda.cuda_backend import has_cipsi_frontier_hash_device

    if not bool(has_cipsi_frontier_hash_device()):
        pytest.skip("CUDA extension missing frontier-hash CIPSI kernels")

    drt = build_drt(norb=3, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(3, seed=19)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(4, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 32),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=8,
        davidson_max_space=4,
        davidson_tol=1e-6,
    )

    assert res.profile.get("selection_mode") == "frontier_hash"
    assert res.epq_mode == "no_epq_support_aware"
    assert res.e_var.shape == (1,)
    assert res.e_pt2.shape == (1,)
    assert np.all(np.isfinite(res.e_var))
    assert np.all(np.isfinite(res.e_pt2))


@pytest.mark.cuda
def test_run_cipsi_trials_streamed_epq_rejected_for_scalable_modes():
    cupy = _require_cuda()

    drt = build_drt(norb=2, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(2, seed=41)

    with pytest.raises(RuntimeError, match="requires epq_mode='no_epq_support_aware'"):
        run_cipsi_trials(
            drt,
            h1e,
            eri,
            nroots=1,
            init_ncsf=2,
            max_ncsf=int(drt.ncsf),
            grow_by=1,
            max_iter=1,
            epq_mode="streamed_epq",
            selection_mode="frontier_hash",
            davidson_max_cycle=8,
            davidson_max_space=4,
            davidson_tol=1e-6,
        )


def test_run_cipsi_trials_dense_mode_removed():
    drt = build_drt(norb=2, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(2, seed=53)

    with pytest.raises(ValueError, match="selection_mode='dense' has been removed"):
        run_cipsi_trials(
            drt,
            h1e,
            eri,
            nroots=1,
            init_ncsf=2,
            max_ncsf=int(drt.ncsf),
            grow_by=1,
            max_iter=1,
            epq_mode="no_epq_support_aware",
            selection_mode="dense",
            davidson_max_cycle=8,
            davidson_max_space=4,
            davidson_tol=1e-6,
        )


def test_run_cipsi_trials_backend_contract_and_key64_labels():
    drt = build_drt(norb=3, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(3, seed=59)

    with pytest.raises(ValueError, match="backend must be 'auto', 'cpu_sparse', 'cuda_key64', or 'cuda_idx64'"):
        run_cipsi_trials(
            drt,
            h1e,
            eri,
            nroots=1,
            init_ncsf=2,
            max_ncsf=min(int(drt.ncsf), 8),
            grow_by=1,
            max_iter=1,
            epq_mode="no_epq_support_aware",
            selection_mode="frontier_hash",
            backend="cuda",
        )

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=2,
        max_ncsf=min(int(drt.ncsf), 8),
        grow_by=1,
        max_iter=1,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        backend="cuda_key64",
    )
    assert res.label_kind == "key64"
    assert res.sel_key_u64 is not None
    assert res.profile.get("backend_requested") == "cuda_key64"
    assert res.profile.get("backend_effective") == "cuda_key64"


def test_run_cipsi_trials_backend_idx64_contract_and_labels():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=61)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=2,
        max_ncsf=min(int(drt.ncsf), 10),
        grow_by=2,
        max_iter=1,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        backend="cuda_idx64",
        state_rep="i64",
    )
    assert res.label_kind == "idx64"
    assert res.sel_key_u64 is not None and res.sel_key_u64.dtype == np.uint64
    np.testing.assert_array_equal(np.asarray(res.sel_key_u64, dtype=np.int64), np.asarray(res.sel_idx, dtype=np.int64))
    assert res.profile.get("backend_requested") == "cuda_idx64"
    assert res.profile.get("backend_effective") == "cuda_idx64"


def test_build_cipsi_trials_from_scf_backend_contract():
    with pytest.raises(ValueError, match="backend must be 'auto', 'cpu_sparse', 'cuda_key64', or 'cuda_idx64'"):
        build_cipsi_trials_from_scf(
            object(),
            backend="cuda",
            ncore=0,
            ncas=2,
            nelecas=2,
        )


@pytest.mark.cuda
def test_frontier_hash_single_root_offdiag_aggregation_parity(monkeypatch):
    cupy = _require_cuda()

    from asuka.cuda.cuda_backend import has_cipsi_frontier_hash_device

    if not bool(has_cipsi_frontier_hash_device()):
        pytest.skip("CUDA extension missing frontier-hash CIPSI kernels")

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=97)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 64),
        grow_by=4,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
    )

    monkeypatch.setenv("ASUKA_FH_OFFDIAG_AGG_SINGLE_ROOT", "0")
    res_base = run_cipsi_trials(drt, h1e, eri, **kwargs)
    monkeypatch.setenv("ASUKA_FH_OFFDIAG_AGG_SINGLE_ROOT", "1")
    res_agg = run_cipsi_trials(drt, h1e, eri, **kwargs)

    np.testing.assert_allclose(res_agg.e_var, res_base.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_agg.e_pt2, res_base.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_agg.e_tot, res_base.e_tot, rtol=1e-7, atol=1e-8)
    assert np.array_equal(np.sort(res_agg.sel_idx), np.sort(res_base.sel_idx))


@pytest.mark.cuda
def test_make_hdiag_guess_device_df_matches_host_df():
    cupy = _require_cuda()

    from asuka.cuguga.state_cache import get_state_cache
    from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
    from asuka.sci.selected_ci import _make_hdiag_guess

    norb = 3
    drt = build_drt(norb=norb, nelec=2, twos_target=0)
    h1e, _ = _make_symmetric_test_integrals(norb, seed=123)

    rng = np.random.default_rng(1234)
    nops = norb * norb
    naux = 7
    l_full = np.asarray(rng.normal(size=(nops, naux)), dtype=np.float64, order="C")
    pair_norm = np.asarray(np.linalg.norm(l_full, axis=1), dtype=np.float64, order="C")
    j_ps = np.zeros((norb, norb), dtype=np.float64)
    eri_host = DFMOIntegrals(
        norb=norb,
        l_full=l_full,
        j_ps=j_ps,
        pair_norm=pair_norm,
    )
    eri_dev = DeviceDFMOIntegrals(
        norb=norb,
        l_full=_cupy_asarray_or_skip(cupy, l_full),
        j_ps=_cupy_asarray_or_skip(cupy, j_ps),
        pair_norm=_cupy_asarray_or_skip(cupy, pair_norm),
        eri_mat=None,
    )
    state_cache = get_state_cache(drt)

    h_host = _make_hdiag_guess(drt, h1e, eri_host, state_cache=state_cache)
    h_dev = _make_hdiag_guess(drt, h1e, eri_dev, state_cache=state_cache)
    np.testing.assert_allclose(h_dev, h_host, rtol=1e-12, atol=1e-12)


def test_diagonal_guess_lookup_device_df_accepts_l_full_with_eri_mat():
    from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
    from asuka.sci.selected_ci import DiagonalGuessLookup

    drt = build_drt(norb=2, nelec=2, twos_target=0)
    h1e = np.asarray([[-1.1, 0.0], [0.0, -0.4]], dtype=np.float64)

    l_full = np.asarray(
        [
            [0.8, -0.1, 0.2],
            [0.3, 0.5, -0.4],
            [0.3, 0.5, -0.4],
            [0.6, -0.2, 0.1],
        ],
        dtype=np.float64,
        order="C",
    )
    pair_norm = np.asarray(np.linalg.norm(l_full, axis=1), dtype=np.float64, order="C")
    j_ps = np.zeros((2, 2), dtype=np.float64)
    eri_mat = np.asarray(l_full @ l_full.T, dtype=np.float64, order="C")

    eri_host = DFMOIntegrals(
        norb=2,
        l_full=l_full,
        j_ps=j_ps,
        pair_norm=pair_norm,
        _eri_mat=eri_mat,
    )
    eri_dev = DeviceDFMOIntegrals(
        norb=2,
        l_full=l_full,
        j_ps=j_ps,
        pair_norm=pair_norm,
        eri_mat=eri_mat,
    )

    lookup_host = DiagonalGuessLookup(drt, h1e, eri_host)
    lookup_dev = DiagonalGuessLookup(drt, h1e, eri_dev)

    for idx in range(int(drt.ncsf)):
        np.testing.assert_allclose(lookup_dev.get(idx), lookup_host.get(idx), rtol=1e-12, atol=1e-12)
