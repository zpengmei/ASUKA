from __future__ import annotations

import numpy as np
import pytest

from asuka.cuguga.drt import build_drt
from asuka.qmc.fcifri import run_fcifri_block
from asuka.qmc.sparse import SparseVector
from asuka.sci.gpu_cipsi import CIPSITrialSpaceResult, build_cipsi_trials_from_scf, run_cipsi_trials
from asuka.sci.projected_apply import ExactExternalProjectedApply, ExactSelectedProjectedHop
from asuka.sci.sparse_support import (
    DiagonalGuessLookup,
    IncrementalVariationalHamiltonianBuilder,
    _accumulate_and_score_external_sparse,
    _build_variational_hamiltonian_sparse,
)


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


def test_legacy_frontier_selector_symbol_is_absent():
    import asuka.sci.frontier_hash as frontier_hash

    assert not hasattr(frontier_hash, "FrontierHashSelector")


@pytest.mark.parametrize("epq_mode", ["no_epq_support_aware"])
@pytest.mark.parametrize("state_rep", ["auto", "key64"])
@pytest.mark.parametrize("nroots", [1, 2])
@pytest.mark.cuda
def test_run_cipsi_trials_gpu_smoke_and_fcifri_hook(epq_mode, state_rep, nroots):
    cupy = _require_cuda()

    drt = build_drt(norb=2, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(2)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=nroots,
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
    assert res.ci_sel.shape[1] == nroots
    assert len(res.roots) == nroots
    for root in res.roots:
        assert isinstance(root, SparseVector)
        assert np.all(np.diff(root.idx) > 0) if root.idx.size > 1 else True

    fri = run_fcifri_block(
        drt,
        h1e,
        eri,
        nroots=nroots,
        m=max(1, max(int(root.nnz) for root in res.roots)),
        eps=0.01,
        niter=0,
        nspawn_one=1,
        nspawn_two=1,
        seed=11,
        backend="auto",
        x0=res,
        rsi_min_nsample=0,
    )
    assert fri.energies.shape == (1, nroots)


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


def test_run_cipsi_trials_frontier_hash_bucketed_parity(monkeypatch):
    import asuka.sci.sparse_support as sparse_support

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=123)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=4,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cpu_sparse",
    )

    res_base = run_cipsi_trials(drt, h1e, eri, **kwargs)
    monkeypatch.setattr(sparse_support, "SELECTOR_BUCKET_EDGE_THRESHOLD", 1)
    res_bucketed = run_cipsi_trials(drt, h1e, eri, **kwargs)

    np.testing.assert_allclose(res_bucketed.e_var, res_base.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_bucketed.e_pt2, res_base.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_bucketed.e_tot, res_base.e_tot, rtol=1e-7, atol=1e-8)
    assert np.array_equal(np.sort(res_bucketed.sel_idx), np.sort(res_base.sel_idx))
    assert bool(res_bucketed.profile.get("selector_bucketed_any", False))


def test_run_cipsi_trials_frontier_hash_bucket_split_parity(monkeypatch):
    import asuka.sci.sparse_support as sparse_support

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=124)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=4,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cpu_sparse",
    )

    res_base = run_cipsi_trials(drt, h1e, eri, **kwargs)
    monkeypatch.setattr(
        sparse_support,
        "_plan_selector_buckets",
        lambda drt, h1e, eri, *, sel, c_sel, max_out, screening, state_cache, row_cache: sparse_support.SelectorBucketPlan(
            bucketed=True,
            nbuckets=1,
            bucket_bounds=((0, int(drt.ncsf)),),
            active_frontier_edges=1,
        ),
    )
    monkeypatch.setattr(
        sparse_support,
        "_maybe_split_bucket_range",
        lambda label_lo, label_hi, *, cand_count, max_add: (
            ((int(label_lo), int(label_lo) + max(1, (int(label_hi) - int(label_lo)) // 2)), (int(label_lo) + max(1, (int(label_hi) - int(label_lo)) // 2), int(label_hi)))
            if int(label_hi) - int(label_lo) > 1
            else ((int(label_lo), int(label_hi)),)
        ),
    )
    res_split = run_cipsi_trials(drt, h1e, eri, **kwargs)

    np.testing.assert_allclose(res_split.e_var, res_base.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_split.e_pt2, res_base.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_split.e_tot, res_base.e_tot, rtol=1e-7, atol=1e-8)
    assert np.array_equal(np.sort(res_split.sel_idx), np.sort(res_base.sel_idx))
    split_stats = [rec.get("selector", {}) for rec in res_split.history if "selector" in rec]
    assert any(int(stats.get("selector_bucket_splits", 0)) > 0 for stats in split_stats)


@pytest.mark.cuda
def test_run_cipsi_trials_cuda_bucketed_frontier_hash_parity(monkeypatch):
    _require_cuda()
    import asuka.sci.sparse_support as sparse_support
    import asuka.sci.gpu_cipsi as gpu_cipsi

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=211)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=4,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
    )

    res_base = run_cipsi_trials(drt, h1e, eri, **kwargs)
    monkeypatch.setattr(sparse_support, "SELECTOR_BUCKET_EDGE_THRESHOLD", 1)
    monkeypatch.setattr(gpu_cipsi, "SELECTOR_BUCKET_EDGE_THRESHOLD", 1)
    res_bucketed = run_cipsi_trials(drt, h1e, eri, **kwargs)

    np.testing.assert_allclose(res_bucketed.e_var, res_base.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_bucketed.e_pt2, res_base.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_bucketed.e_tot, res_base.e_tot, rtol=1e-7, atol=1e-8)
    assert np.array_equal(np.sort(res_bucketed.sel_idx), np.sort(res_base.sel_idx))
    assert bool(res_bucketed.profile.get("selector_bucketed_any", False)) or bool(
        res_bucketed.profile.get("exact_external_selector_effective", False)
    )
    assert res_bucketed.profile.get("driver") == "cuda_cas36_hb_compact_u64"
    assert "cuda_selector_step_fallback_reason" not in res_bucketed.profile


def test_run_cipsi_trials_solver_reorder_parity(monkeypatch):
    import asuka.sci.sparse_support as sparse_support

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=125)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=4,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cpu_sparse",
    )

    res_base = run_cipsi_trials(drt, h1e, eri, **kwargs)
    monkeypatch.setattr(sparse_support, "SOLVER_REORDER_MIN_NSEL", 1)
    res_reordered = run_cipsi_trials(drt, h1e, eri, **kwargs)

    np.testing.assert_allclose(res_reordered.e_var, res_base.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_reordered.e_pt2, res_base.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_reordered.e_tot, res_base.e_tot, rtol=1e-7, atol=1e-8)
    assert any(bool(rec.get("solver_reordered", False)) for rec in res_reordered.history)
    assert bool(res_reordered.profile.get("solver_reordered_final", False))


def test_run_cipsi_trials_macro_growth_cpu_smoke():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=131)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cpu_sparse",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    assert bool(res.profile.get("macro_schedule_enabled", False))
    assert int(res.profile.get("macro_growth_steps", 0)) == 2
    assert int(res.profile.get("solve_count", 0)) >= 1
    assert len(res.history) >= 1
    assert all("macro_iter" in rec and "micro_iter" in rec for rec in res.history)
    assert bool(res.history[0].get("solve_refreshed", False))
    assert np.all(np.isfinite(res.e_var))
    assert np.all(np.isfinite(res.e_pt2))


def test_run_cipsi_trials_macro_growth_cpu_exact_external_optin_preserves_results():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=177)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cpu_sparse",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    res_ref = run_cipsi_trials(drt, h1e, eri, **kwargs)
    res_exact = run_cipsi_trials(
        drt,
        h1e,
        eri,
        **{
            **kwargs,
            "workspace_kwargs": {
                **dict(kwargs["workspace_kwargs"]),
                "exact_external_projected_selector": True,
            },
        },
    )

    np.testing.assert_allclose(res_exact.e_var, res_ref.e_var, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(res_exact.e_pt2, res_ref.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_exact.e_tot, res_ref.e_tot, rtol=1e-5, atol=1e-4)
    assert np.array_equal(np.sort(res_exact.sel_idx), np.sort(res_ref.sel_idx))
    assert bool(res_exact.profile.get("exact_external_selector_requested", False))
    assert bool(res_exact.profile.get("exact_external_selector_effective", False))
    assert "exact_external_selector_fallback_reason" not in res_exact.profile
    assert any(
        str(rec.get("selector", {}).get("selector_backend", "")).startswith("exact_external_")
        for rec in res_exact.history
    )


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_smoke():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=211)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    assert bool(res.profile.get("macro_schedule_enabled", False))
    assert int(res.profile.get("macro_growth_steps", 0)) == 2
    assert int(res.profile.get("solve_count", 0)) >= 1
    assert len(res.history) >= 1
    assert all("macro_iter" in rec and "micro_iter" in rec for rec in res.history)
    assert bool(res.history[0].get("solve_refreshed", False))
    assert res.profile.get("driver") == "cuda_cas36_hb_compact_u64"
    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert res.profile.get("projected_solver_backend") in (
        "cuda_davidson_projected_exact_tuples_device",
        "cuda_davidson_projected_exact_tuples",
    )
    assert "threshold" in [str(x) for x in res.profile.get("selection_policy_history", [])]
    assert any(
        str(rec.get("selector", {}).get("selector_backend", "")).startswith("exact_external_device_emit_")
        for rec in res.history
    )
    assert np.all(np.isfinite(res.e_var))


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_idx64_smoke():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=212)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_idx64",
        state_rep="i64",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    assert bool(res.profile.get("macro_schedule_enabled", False))
    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert res.profile.get("projected_solver_backend") == "cuda_davidson_projected_exact_tuples"
    assert res.label_kind == "idx64"
    assert np.all(np.isfinite(res.e_var))


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_df_exact_tuple_smoke():
    _require_cuda()

    from asuka.integrals.df_integrals import DFMOIntegrals

    norb = 4
    drt = build_drt(norb=norb, nelec=2, twos_target=0)
    h1e, _ = _make_symmetric_test_integrals(norb, seed=213)
    rng = np.random.default_rng(4213)
    nops = norb * norb
    naux = 9
    l_full = np.asarray(rng.normal(size=(nops, naux)), dtype=np.float64, order="C")
    pair_norm = np.asarray(np.linalg.norm(l_full, axis=1), dtype=np.float64, order="C")
    l3 = l_full.reshape(norb, norb, naux)
    j_ps = np.asarray(np.einsum("pql,qsl->ps", l3, l3, optimize=True), dtype=np.float64, order="C")
    eri = DFMOIntegrals(
        norb=norb,
        l_full=l_full,
        j_ps=j_ps,
        pair_norm=pair_norm,
    )

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    assert bool(res.profile.get("macro_schedule_enabled", False))
    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert res.profile.get("projected_solver_backend") == "cuda_davidson_projected_exact_tuples"
    assert res.label_kind == "key64"
    assert np.all(np.isfinite(res.e_var))


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_exact_external_optin_preserves_results():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=237)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    res_ref = run_cipsi_trials(drt, h1e, eri, **kwargs)
    res_exact = run_cipsi_trials(
        drt,
        h1e,
        eri,
        **{
            **kwargs,
            "workspace_kwargs": {
                **dict(kwargs["workspace_kwargs"]),
                "exact_external_projected_selector": True,
            },
        },
    )

    np.testing.assert_allclose(res_exact.e_var, res_ref.e_var, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(res_exact.e_pt2, res_ref.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_exact.e_tot, res_ref.e_tot, rtol=1e-5, atol=1e-4)
    assert np.array_equal(np.sort(res_exact.sel_idx), np.sort(res_ref.sel_idx))
    assert bool(res_exact.profile.get("exact_external_selector_requested", False))
    assert bool(res_exact.profile.get("exact_external_selector_effective", False))
    assert "exact_external_selector_fallback_reason" not in res_exact.profile
    assert any(
        str(rec.get("selector", {}).get("selector_backend", "")).startswith("exact_external_device_emit_")
        for rec in res_exact.history
    )


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_multi_root_exact_external_threshold_path():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=281)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=2,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 2},
    )

    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert res.profile.get("projected_solver_backend") in (
        "cuda_davidson_projected_exact_tuples_device",
        "cuda_davidson_projected_exact_tuples",
    )
    assert "threshold" in [str(x) for x in res.profile.get("selection_policy_history", [])]
    assert any(
        str(rec.get("selector", {}).get("selector_backend", "")).startswith("exact_external_device_emit_")
        for rec in res.history
    )
    assert len(res.roots) == 2
    assert np.all(np.isfinite(res.e_var))


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_parallel_emit_streams_smoke():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=282)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={
            "macro_growth_steps": 2,
            "external_emit_streams": 2,
            "external_emit_chunk_min_nsel": 1,
        },
    )

    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert int(res.profile.get("exact_external_emit_streams_effective", 0)) >= 2
    assert "parallel_emit_overflow_retry" not in str(res.profile.get("exact_external_emit_streams_fallback_reason", ""))
    assert any(
        str(rec.get("selector", {}).get("selector_backend", "")).startswith("exact_external_device_emit_")
        for rec in res.history
    )
    assert np.all(np.isfinite(res.e_var))


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_exact_macro_trim_records_profile():
    _require_cuda()

    drt = build_drt(norb=12, nelec=6, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(12, seed=778)

    res = run_cipsi_trials(
        drt,
        h1e,
        eri,
        nroots=1,
        init_ncsf=8,
        max_ncsf=400,
        grow_by=200,
        max_iter=1,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=8,
        davidson_max_space=8,
        davidson_tol=1e-6,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 4},
    )

    trim_sizes = [int(x) for x in res.profile.get("macro_trim_sizes", [])]
    assert bool(res.profile.get("exact_external_selector_effective", False))
    assert "threshold" in [str(x) for x in res.profile.get("selection_policy_history", [])]
    assert trim_sizes and max(trim_sizes) > 0
    assert int(len(res.sel_idx)) < max(int(rec.get("nsel", 0)) for rec in res.history)


@pytest.mark.cuda
def test_run_cipsi_trials_macro_growth_cuda_matrix_free_optin_preserves_results():
    _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=233)
    kwargs = dict(
        nroots=1,
        init_ncsf=min(8, int(drt.ncsf)),
        max_ncsf=min(int(drt.ncsf), 16),
        grow_by=2,
        max_iter=2,
        epq_mode="no_epq_support_aware",
        selection_mode="frontier_hash",
        davidson_max_cycle=10,
        davidson_max_space=8,
        davidson_tol=1e-7,
        backend="cuda_key64",
        state_rep="key64",
        workspace_kwargs={"macro_growth_steps": 2, "projected_solver_gpu": True},
    )

    res_sell = run_cipsi_trials(
        drt,
        h1e,
        eri,
        **{
            **kwargs,
            "workspace_kwargs": {
                **dict(kwargs["workspace_kwargs"]),
                "projected_solver_matrix_free": False,
            },
        },
    )
    res_proj = run_cipsi_trials(
        drt,
        h1e,
        eri,
        **{
            **kwargs,
            "workspace_kwargs": {
                **dict(kwargs["workspace_kwargs"]),
                "projected_solver_matrix_free": True,
            },
        },
    )

    assert res_sell.profile.get("projected_solver_backend") in (
        "cuda_davidson_projected_exact_tuples_device",
        "cuda_davidson_projected_exact_tuples",
    )
    np.testing.assert_allclose(res_proj.e_var, res_sell.e_var, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_proj.e_pt2, res_sell.e_pt2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(res_proj.e_tot, res_sell.e_tot, rtol=1e-7, atol=1e-8)
    assert np.array_equal(np.sort(res_proj.sel_idx), np.sort(res_sell.sel_idx))
    assert bool(res_proj.profile.get("projected_solver_matrix_free_requested", False))
    assert res_proj.profile.get("projected_solver_backend") in (
        "cuda_davidson_projected_exact_tuples_device",
        "cuda_davidson_projected_exact_tuples",
    )


def test_incremental_variational_hamiltonian_builder_matches_full_rebuild():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=321)
    max_out = 200_000
    state_cache = None
    screening = None
    row_cache = {}

    sel0 = [0, 1]
    loc_map0 = {int(ii): pos for pos, ii in enumerate(sel0)}
    builder = IncrementalVariationalHamiltonianBuilder(
        drt,
        h1e,
        eri,
        sel=sel0,
        loc_map=loc_map0,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
        row_cache=row_cache,
    )
    full0 = _build_variational_hamiltonian_sparse(
        drt,
        h1e,
        eri,
        sel=sel0,
        loc_map=loc_map0,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
        row_cache=row_cache,
    )
    np.testing.assert_allclose(builder.to_csr().toarray(), full0.toarray(), rtol=1e-12, atol=1e-12)

    added = [2, 3]
    builder.extend(added)
    sel1 = sel0 + added
    loc_map1 = {int(ii): pos for pos, ii in enumerate(sel1)}
    full1 = _build_variational_hamiltonian_sparse(
        drt,
        h1e,
        eri,
        sel=sel1,
        loc_map=loc_map1,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
        row_cache=row_cache,
    )
    np.testing.assert_allclose(builder.to_csr().toarray(), full1.toarray(), rtol=1e-12, atol=1e-12)


def test_exact_selected_projected_hop_host_matches_csr():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=401)
    sel = [0, 1, 2, 3]
    loc_map = {int(ii): pos for pos, ii in enumerate(sel)}
    builder = IncrementalVariationalHamiltonianBuilder(
        drt,
        h1e,
        eri,
        sel=sel,
        loc_map=loc_map,
        max_out=200_000,
        screening=None,
        state_cache=None,
        row_cache={},
    )
    h_csr = builder.to_csr()
    op = ExactSelectedProjectedHop.from_csr(np.asarray(sel, dtype=np.int64), h_csr)

    x = np.asarray([0.3, -0.2, 0.5, 0.1], dtype=np.float64)
    y = op.hop_host(x)
    np.testing.assert_allclose(y, np.asarray(h_csr @ x, dtype=np.float64), rtol=0.0, atol=1e-12)

    x_block = np.column_stack([x, np.asarray([0.1, 0.2, -0.1, 0.4], dtype=np.float64)])
    y_block = op.hop_host(x_block)
    np.testing.assert_allclose(y_block, np.asarray(h_csr @ x_block, dtype=np.float64), rtol=0.0, atol=1e-12)


@pytest.mark.cuda
def test_exact_selected_projected_hop_gpu_matches_csr():
    cupy = _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=402)
    sel = [0, 1, 2, 3]
    loc_map = {int(ii): pos for pos, ii in enumerate(sel)}
    builder = IncrementalVariationalHamiltonianBuilder(
        drt,
        h1e,
        eri,
        sel=sel,
        loc_map=loc_map,
        max_out=200_000,
        screening=None,
        state_cache=None,
        row_cache={},
    )
    h_csr = builder.to_csr()
    op = ExactSelectedProjectedHop.from_csr(np.asarray(sel, dtype=np.int64), h_csr)

    x = np.asarray([0.25, -0.15, 0.05, 0.3], dtype=np.float64)
    x_d = _cupy_asarray_or_skip(cupy, x)
    y_d = op.hop_gpu(x_d)
    y = np.asarray(cupy.asnumpy(y_d), dtype=np.float64)
    np.testing.assert_allclose(y, np.asarray(h_csr @ x, dtype=np.float64), rtol=1e-12, atol=1e-12)

    x_block = np.column_stack([x, np.asarray([0.4, 0.1, -0.2, 0.3], dtype=np.float64)])
    x_block_d = _cupy_asarray_or_skip(cupy, x_block)
    y_block_d = op.hop_gpu(x_block_d)
    y_block = np.asarray(cupy.asnumpy(y_block_d), dtype=np.float64)
    np.testing.assert_allclose(y_block, np.asarray(h_csr @ x_block, dtype=np.float64), rtol=1e-12, atol=1e-12)


def test_exact_external_projected_apply_host_matches_sparse_selector_reference():
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=403)
    sel = np.asarray([0, 1, 2, 3], dtype=np.int64)
    c_sel = np.asarray(
        [
            [0.45, -0.10],
            [-0.25, 0.30],
            [0.15, 0.40],
            [0.05, -0.20],
        ],
        dtype=np.float64,
    )
    e_var = np.asarray([-1.2, -0.7], dtype=np.float64)
    hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)
    row_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    op = ExactExternalProjectedApply(
        drt=drt,
        h1e=np.asarray(h1e, dtype=np.float64),
        eri=eri,
        max_out=200_000,
        screening=None,
        state_cache=None,
        row_cache=row_cache,
    )
    idx, p = op.accumulate_host(sel_idx=sel, c_sel=c_sel, selected_set=set(int(x) for x in sel.tolist()))
    c1, w, e_pt2 = op.score_host(idx=idx, vals_root_major=p, e_var=e_var, hdiag_lookup=hdiag_lookup, denom_floor=0.0)

    idx_ref, w_ref, e_pt2_ref, _cand_count_ref, c1_ref = _accumulate_and_score_external_sparse(
        drt,
        np.asarray(h1e, dtype=np.float64),
        eri,
        sel=sel.tolist(),
        selected_set=set(int(x) for x in sel.tolist()),
        c_sel=c_sel,
        e_var=e_var,
        hdiag_lookup=hdiag_lookup,
        max_add=10_000,
        select_threshold=None,
        denom_floor=0.0,
        max_out=200_000,
        screening=None,
        state_cache=None,
        select_screen_contrib=0.0,
        row_cache=row_cache,
        label_lo=0,
        label_hi=int(drt.ncsf),
        return_c1=True,
    )

    ref_pos = {int(ii): pos for pos, ii in enumerate(np.asarray(idx_ref, dtype=np.int64).tolist())}
    assert set(int(ii) for ii in idx.tolist()) == set(ref_pos.keys())
    np.testing.assert_allclose(e_pt2, e_pt2_ref, rtol=1e-12, atol=1e-12)
    for pos, ii in enumerate(idx.tolist()):
        ref_i = int(ref_pos[int(ii)])
        np.testing.assert_allclose(c1[pos, :], np.asarray(c1_ref[ref_i, :], dtype=np.float64), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w[pos], np.asarray(w_ref[ref_i], dtype=np.float64), rtol=1e-12, atol=1e-12)


@pytest.mark.cuda
def test_exact_external_projected_apply_gpu_accumulate_matches_host():
    cupy = _require_cuda()

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=404)
    sel = np.asarray([0, 1, 2, 3], dtype=np.int64)
    c_sel = np.asarray(
        [
            [0.35, -0.05],
            [-0.15, 0.20],
            [0.10, 0.30],
            [0.08, -0.12],
        ],
        dtype=np.float64,
    )
    row_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    op = ExactExternalProjectedApply(
        drt=drt,
        h1e=np.asarray(h1e, dtype=np.float64),
        eri=eri,
        max_out=200_000,
        screening=None,
        state_cache=None,
        row_cache=row_cache,
    )

    idx_h, vals_h = op.accumulate_host(
        sel_idx=sel,
        c_sel=c_sel,
        selected_set=set(int(x) for x in sel.tolist()),
    )
    idx_d, vals_d = op.accumulate_gpu(
        sel_idx=sel,
        c_sel=c_sel,
    )
    idx_gpu = np.asarray(cupy.asnumpy(idx_d), dtype=np.uint64).astype(np.int64, copy=False)
    vals_gpu = np.asarray(cupy.asnumpy(vals_d), dtype=np.float64)

    np.testing.assert_array_equal(idx_gpu, idx_h)
    np.testing.assert_allclose(vals_gpu, vals_h, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda
def test_exact_selected_tuple_emitter_surface_matches_selected_only_surface():
    cupy = _require_cuda()

    from asuka.cuda.cuda_backend import (
        cas36_exact_selected_emit_tuples_u64_inplace_device,
        cas36_hb_emit_tuples_u64_inplace_device,
        has_cas36_exact_selected_emit_tuples_u64_device,
        make_device_drt,
    )
    from asuka.sci.gpu_cipsi import _build_hb_index_and_diag_inputs, upload_hb_index

    assert bool(has_cas36_exact_selected_emit_tuples_u64_device())

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=405)
    sel = np.arange(min(8, int(drt.ncsf)), dtype=np.int64)
    hb_index, *_ = _build_hb_index_and_diag_inputs(drt, h1e, eri)
    hb_dev = upload_hb_index(hb_index, cupy)
    if int(hb_dev["h1_abs"].size) == 0:
        hb_dev["h1_pq"] = cupy.zeros((1, 2), dtype=cupy.int32)
        hb_dev["h1_abs"] = cupy.zeros((1,), dtype=cupy.float64)
        hb_dev["h1_signed"] = cupy.zeros((1,), dtype=cupy.float64)
    if int(hb_dev["rs_idx"].size) == 0:
        hb_dev["rs_idx"] = cupy.zeros((1,), dtype=cupy.int32)
        hb_dev["v_abs"] = cupy.zeros((1,), dtype=cupy.float64)
        hb_dev["v_signed"] = cupy.zeros((1,), dtype=cupy.float64)
    drt_dev = make_device_drt(drt)
    sel_u64 = cupy.asarray(sel.astype(np.uint64, copy=False))
    sel_sorted = cupy.sort(sel_u64)
    c_bound = cupy.ones((int(sel.size),), dtype=cupy.float64)
    cap = 4096

    def _run(fn):
        out_keys = cupy.empty((cap,), dtype=cupy.uint64)
        out_src = cupy.empty((cap,), dtype=cupy.int32)
        out_hij = cupy.empty((cap,), dtype=cupy.float64)
        out_n = cupy.zeros((1,), dtype=cupy.int32)
        overflow = cupy.zeros((1,), dtype=cupy.int32)
        fn(
            drt,
            drt_dev,
            sel_u64,
            c_bound,
            nsel=int(sel.size),
            h1_pq=hb_dev["h1_pq"],
            h1_abs=hb_dev["h1_abs"],
            h1_signed=hb_dev["h1_signed"],
            n_h1=int(hb_dev["h1_abs"].size),
            pq_ptr=hb_dev["pq_ptr"],
            rs_idx=hb_dev["rs_idx"],
            v_abs=hb_dev["v_abs"],
            v_signed=hb_dev["v_signed"],
            pq_max_v=hb_dev["pq_max_v"],
            out_keys_u64=out_keys,
            out_src=out_src,
            out_hij=out_hij,
            cap=cap,
            selected_idx_sorted_u64=sel_sorted,
            out_n=out_n,
            overflow=overflow,
            sync=True,
        )
        n = int(cupy.asnumpy(out_n)[0])
        return (
            np.asarray(cupy.asnumpy(out_keys[:n]), dtype=np.uint64),
            np.asarray(cupy.asnumpy(out_src[:n]), dtype=np.int32),
            np.asarray(cupy.asnumpy(out_hij[:n]), dtype=np.float64),
            int(cupy.asnumpy(overflow)[0]),
        )

    keys_exact, src_exact, hij_exact, of_exact = _run(cas36_exact_selected_emit_tuples_u64_inplace_device)

    def _hb_wrapper(*args, **kwargs):
        return cas36_hb_emit_tuples_u64_inplace_device(*args, eps=0.0, target_mode="selected_only", **kwargs)

    keys_ref, src_ref, hij_ref, of_ref = _run(_hb_wrapper)
    assert of_exact == 0
    assert of_ref == 0
    order_exact = np.lexsort((src_exact.astype(np.int64, copy=False), keys_exact.astype(np.int64, copy=False)))
    order_ref = np.lexsort((src_ref.astype(np.int64, copy=False), keys_ref.astype(np.int64, copy=False)))
    np.testing.assert_array_equal(keys_exact[order_exact], keys_ref[order_ref])
    np.testing.assert_array_equal(src_exact[order_exact], src_ref[order_ref])
    np.testing.assert_allclose(hij_exact[order_exact], hij_ref[order_ref], rtol=0.0, atol=0.0)


@pytest.mark.cuda
def test_exact_selected_dense_tuple_emitter_matches_host_selected_rows():
    cupy = _require_cuda()

    from asuka.cuda.cuda_backend import (
        cas36_exact_selected_emit_tuples_dense_u64_inplace_device,
        has_cas36_exact_selected_emit_tuples_dense_u64_device,
        make_device_drt,
    )
    from asuka.cuguga.oracle import _restore_eri_4d
    from asuka.sci.sparse_support import _connected_row_cached

    assert bool(has_cas36_exact_selected_emit_tuples_dense_u64_device())

    drt = build_drt(norb=4, nelec=2, twos_target=0)
    h1e, eri = _make_symmetric_test_integrals(4, seed=406)
    eri4 = np.asarray(_restore_eri_4d(eri, int(drt.norb)), dtype=np.float64, order="C")
    h_base = np.asarray(
        np.asarray(h1e, dtype=np.float64, order="C") - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True),
        dtype=np.float64,
        order="C",
    )

    sel = np.arange(min(8, int(drt.ncsf)), dtype=np.int64)
    sel_set = set(int(x) for x in sel.tolist())
    drt_dev = make_device_drt(drt)
    sel_u64 = cupy.asarray(sel.astype(np.uint64, copy=False))
    sel_sorted = cupy.sort(sel_u64)
    c_bound = cupy.ones((int(sel.size),), dtype=cupy.float64)
    h_base_d = cupy.asarray(np.asarray(h_base, dtype=np.float64, order="C").ravel())
    eri4_d = cupy.asarray(np.asarray(eri4, dtype=np.float64, order="C").ravel())
    cap = 16384

    out_keys = cupy.empty((cap,), dtype=cupy.uint64)
    out_src = cupy.empty((cap,), dtype=cupy.int32)
    out_hij = cupy.empty((cap,), dtype=cupy.float64)
    out_n = cupy.zeros((1,), dtype=cupy.int32)
    overflow = cupy.zeros((1,), dtype=cupy.int32)

    cas36_exact_selected_emit_tuples_dense_u64_inplace_device(
        drt,
        drt_dev,
        sel_u64,
        c_bound,
        nsel=int(sel.size),
        h_base=h_base_d,
        eri4=eri4_d,
        out_keys_u64=out_keys,
        out_src=out_src,
        out_hij=out_hij,
        cap=cap,
        selected_idx_sorted_u64=sel_sorted,
        out_n=out_n,
        overflow=overflow,
        sync=True,
    )
    assert int(cupy.asnumpy(overflow)[0]) == 0
    n = int(cupy.asnumpy(out_n)[0])
    keys_gpu = np.asarray(cupy.asnumpy(out_keys[:n]), dtype=np.uint64).astype(np.int64, copy=False)
    src_gpu = np.asarray(cupy.asnumpy(out_src[:n]), dtype=np.int32)
    hij_gpu = np.asarray(cupy.asnumpy(out_hij[:n]), dtype=np.float64)

    label_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []
    hij_parts: list[np.ndarray] = []
    row_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pos, j in enumerate(sel.tolist()):
        i_idx, hij = _connected_row_cached(
            drt,
            np.asarray(h1e, dtype=np.float64),
            eri,
            int(j),
            max_out=200_000,
            screening=None,
            state_cache=None,
            row_cache=row_cache,
        )
        keep_labels = []
        keep_vals = []
        for ii, vv in zip(np.asarray(i_idx, dtype=np.int64).tolist(), np.asarray(hij, dtype=np.float64).tolist(), strict=False):
            if int(ii) == int(j):
                continue
            if int(ii) in sel_set:
                keep_labels.append(int(ii))
                keep_vals.append(float(vv))
        if keep_labels:
            label_parts.append(np.asarray(keep_labels, dtype=np.int64))
            src_parts.append(np.full((len(keep_labels),), int(pos), dtype=np.int32))
            hij_parts.append(np.asarray(keep_vals, dtype=np.float64))

    keys_h = np.asarray(np.concatenate(label_parts), dtype=np.int64, order="C") if label_parts else np.zeros((0,), dtype=np.int64)
    src_h = np.asarray(np.concatenate(src_parts), dtype=np.int32, order="C") if src_parts else np.zeros((0,), dtype=np.int32)
    hij_h = np.asarray(np.concatenate(hij_parts), dtype=np.float64, order="C") if hij_parts else np.zeros((0,), dtype=np.float64)

    def _reduce_pairs(keys, src, hij):
        if int(keys.size) == 0:
            return keys, src, hij
        order = np.lexsort((src.astype(np.int64, copy=False), keys.astype(np.int64, copy=False)))
        keys_s = np.asarray(keys[order], dtype=np.int64, order="C")
        src_s = np.asarray(src[order], dtype=np.int32, order="C")
        hij_s = np.asarray(hij[order], dtype=np.float64, order="C")
        if int(keys_s.size) == 1:
            return keys_s, src_s, hij_s
        change = (keys_s[1:] != keys_s[:-1]) | (src_s[1:] != src_s[:-1])
        starts = np.concatenate(([0], np.nonzero(change)[0] + 1)).astype(np.int64, copy=False)
        return (
            np.asarray(keys_s[starts], dtype=np.int64, order="C"),
            np.asarray(src_s[starts], dtype=np.int32, order="C"),
            np.asarray(np.add.reduceat(hij_s, starts), dtype=np.float64, order="C"),
        )

    keys_gpu_r, src_gpu_r, hij_gpu_r = _reduce_pairs(keys_gpu, src_gpu, hij_gpu)
    keys_h_r, src_h_r, hij_h_r = _reduce_pairs(keys_h, src_h, hij_h)
    np.testing.assert_array_equal(keys_gpu_r, keys_h_r)
    np.testing.assert_array_equal(src_gpu_r, src_h_r)
    np.testing.assert_allclose(hij_gpu_r, hij_h_r, rtol=1e-11, atol=1e-11)


@pytest.mark.cuda
def test_make_hdiag_guess_device_df_matches_host_df():
    cupy = _require_cuda()

    from asuka.cuguga.state_cache import get_state_cache
    from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
    from asuka.sci.sparse_support import _make_hdiag_guess

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
    from asuka.sci.sparse_support import DiagonalGuessLookup

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
