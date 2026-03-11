from __future__ import annotations

from collections import OrderedDict
import io
from types import SimpleNamespace
import numpy as np
import pytest

from asuka._solver.dump_flags_runtime import dump_flags
from asuka._solver.init_runtime import configure_solver_runtime_defaults
from asuka._solver.kernel_runtime import (
    autotune_cuda_max_g_mib_for_large_cas,
    build_cuda_hamiltonian_inputs,
    build_kernel_dry_run_result,
    maybe_restore_contract_eri,
    normalize_row_screening,
    prepare_kernel_precompute_and_state_cache,
    resolve_kernel_nroots,
    resolve_kernel_warm_start,
    run_kernel_dense_eigh_fastpath,
)
from asuka._solver.kernel_cuda_runtime import (
    apply_low_precision_and_workspace_policy,
    auto_select_use_epq_table,
    resolve_epq_streaming_controls,
    resolve_cuda_workspace_policy_common,
    validate_low_precision_cuda_path,
)
from asuka._solver.matvec_cache_runtime import (
    configure_matvec_cuda_ws_cache,
    matvec_cuda_ws_cache_drop,
    matvec_cuda_ws_cache_enforce_budget,
    matvec_cuda_ws_cache_get,
    matvec_cuda_ws_cache_profile,
    matvec_cuda_ws_cache_put,
    matvec_cuda_ws_cache_touch,
    release_matvec_cuda_ws_cache,
)
from asuka._solver.cuda_policy import (
    apply_cuda_user_policy,
    cap_cuda_cublas_workspace_cap_mb_by_hard_cap,
    cap_cuda_max_g_mib_by_hard_cap,
    maybe_promote_cuda_apply_mode_scatter,
    normalize_cuda_accuracy_mode,
    normalize_cuda_user_policy_mode,
    normalize_matvec_cuda_path_mode,
    normalize_ws_cache_fraction,
    resolve_cuda_apply_mode,
    resolve_cuda_cublas_workspace_cap_mb,
    resolve_cuda_epq_build_device,
    resolve_cuda_memory_controls,
    resolve_cuda_max_g_mib,
    resolve_kernel_cuda_policy,
    resolve_epq_overbudget_action,
)
from asuka._solver.config import (
    auto_num_threads,
    resolve_kernel_frontend_controls,
    resolve_kernel_runtime_controls,
    resolve_kernel_solver_controls,
)
from asuka._solver.drt_cache import (
    DRTKey,
    ne_constraints_key_to_dict,
    ne_constraints_to_key,
    orbsym_to_tuple,
)
from asuka._solver.drt_runtime import drt_key, get_or_build_drt
from asuka._solver.matvec_runtime import (
    CUDA_MATVEC_BACKENDS,
    estimate_matvec_cuda_workspace_bytes,
    get_or_create_cuda_matvec_state,
    release_matvec_cuda_workspace,
    resolve_approx_cuda_frontend,
    resolve_approx_kernel_iteration_caps,
    resolve_cuda_cache_csr_tiles,
    resolve_cuda_j_tile,
    resolve_cuda_threads_apply,
    resolve_cuda_threads_w,
    resolve_cuda_workspace_controls,
    resolve_matvec_cuda_ws_cache_budget_bytes,
    resolve_kernel_cuda_execution_mode,
    tune_cuda_threads_for_large_cas_noepq,
    ws_needs_rebuild,
)
from asuka._solver.warm_state import (
    WARM_CUDA_MATVEC_BACKENDS,
    WARM_STATE_FORMAT_VERSION,
    normalize_warm_cas_metadata,
    warm_cas_metadata_from_jsonable,
    warm_cas_metadata_to_jsonable,
)
from asuka._solver.warm_state_runtime import (
    allowed_ci_devices_for_backend,
    load_warm_state,
    save_warm_state,
    update_warm_state,
    warm_state_ci0_if_compatible,
    warm_state_summary,
)


def _make_ws_cache_solver_stub(*, budget: int = 0, fraction: float = 0.2):
    released: list[object] = []
    solver = SimpleNamespace(
        _matvec_cuda_ws_cache={},
        _matvec_cuda_ws_cache_sizes={},
        _matvec_cuda_ws_cache_lru=OrderedDict(),
        _matvec_cuda_ws_cache_bytes=0,
        _matvec_cuda_ws_cache_budget_bytes=int(budget),
        _matvec_cuda_ws_cache_hits=0,
        _matvec_cuda_ws_cache_misses=0,
        _matvec_cuda_ws_cache_evictions=0,
        matvec_cuda_ws_cache_fraction=float(fraction),
        _release_matvec_cuda_workspace=lambda ws: released.append(ws),
        _estimate_matvec_cuda_workspace_bytes=lambda ws: int(getattr(ws, "nbytes", 0)),
    )
    return solver, released


def test_warm_state_constants_and_metadata_roundtrip():
    assert isinstance(WARM_STATE_FORMAT_VERSION, int)
    assert "cuda" in WARM_CUDA_MATVEC_BACKENDS

    meta = normalize_warm_cas_metadata(
        {
            "ncore": 1,
            "ncas": 6,
            "nelecas": [3, 3],
            "cas_orbsym": [1, 2, 3],
            "active_orbital_indices": [5, 6, 7],
        },
        default_ncas=4,
        default_nelecas=(2, 2),
    )
    encoded = warm_cas_metadata_to_jsonable(meta)
    decoded = warm_cas_metadata_from_jsonable(encoded)
    assert decoded == meta


def test_drt_cache_key_helpers_roundtrip():
    assert orbsym_to_tuple(None) is None
    assert orbsym_to_tuple([1, 2, 3]) == (1, 2, 3)

    key = ne_constraints_to_key({2: (1, 3), 0: (0, 2)})
    assert key == ((0, 0, 2), (2, 1, 3))
    assert ne_constraints_key_to_dict(key) == {0: (0, 2), 2: (1, 3)}
    assert ne_constraints_key_to_dict(None) is None

    drt_key = DRTKey(
        norb=6,
        nelec_total=6,
        twos=0,
        wfnsym=None,
        orbsym=(1, 1, 2, 2, 3, 3),
        ne_constraints_key=key,
    )
    assert drt_key.norb == 6


def test_drt_runtime_build_and_cache():
    cache = {}
    key1 = drt_key(2, 2, 0, None, None, ne_constraints=None)
    key2, drt1 = get_or_build_drt(
        cache,
        norb=2,
        nelec_total=2,
        twos=0,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
    )
    key3, drt2 = get_or_build_drt(
        cache,
        norb=2,
        nelec_total=2,
        twos=0,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
    )
    assert key1 == key2 == key3
    assert drt1 is drt2
    assert int(drt1.norb) == 2


def test_matvec_runtime_kernel_cuda_execution_mode_defaults_for_non_cuda():
    kwargs = {"matvec_cuda_dtype": "mixed", "matvec_cuda_mixed_threshold": 1e-4}
    out = resolve_kernel_cuda_execution_mode(
        kwargs=kwargs,
        defaults=object(),
        matvec_backend="contract",
        strict_gpu=False,
        ncsf=10,
        matvec_cuda_davidson_subspace_eigh_cpu_in=None,
        matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff=100,
    )
    assert "cuda" in CUDA_MATVEC_BACKENDS
    assert out["matvec_cuda_dtype"] == "float64"
    assert out["matvec_cuda_make_hdiag_cpu"] is None
    assert out["matvec_cuda_davidson_subspace_eigh_cpu"] is None


def test_matvec_runtime_kernel_cuda_execution_mode_strict_gpu_guard():
    kwargs = {}
    with pytest.raises(ValueError, match="strict_gpu=True is incompatible"):
        resolve_kernel_cuda_execution_mode(
            kwargs=kwargs,
            defaults=object(),
            matvec_backend="cuda",
            strict_gpu=True,
            ncsf=10,
            matvec_cuda_davidson_subspace_eigh_cpu_in=True,
            matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff=100,
        )


def test_matvec_runtime_resolve_approx_cuda_frontend():
    kwargs = {"approx_cuda_dtype": "fp64"}
    out = resolve_approx_cuda_frontend(
        kwargs=kwargs,
        defaults=object(),
        matvec_backend="contract",
    )
    assert out["approx_cuda_dtype"] == "float64"
    assert out["matvec_cuda_aggregate_offdiag_preview"] is None

    kwargs2 = {"approx_cuda_dtype": "float32", "matvec_cuda_aggregate_offdiag": False}
    with pytest.raises(ValueError, match="forbidden by hard guard"):
        resolve_approx_cuda_frontend(
            kwargs=kwargs2,
            defaults=object(),
            matvec_backend="cuda",
        )


def test_matvec_runtime_resolve_approx_kernel_iteration_caps():
    kwargs = {"max_cycle": 9, "max_space": 11}
    defaults = SimpleNamespace(approx_kernel_max_cycle=3, approx_kernel_max_space=8)
    out = resolve_approx_kernel_iteration_caps(
        kwargs=kwargs,
        defaults=defaults,
        nroots=2,
        matvec_backend="cuda",
    )
    assert out["nroots"] == 2
    assert out["max_cycle"] == 3
    assert out["max_space"] == 8

    out2 = resolve_approx_kernel_iteration_caps(
        kwargs={},
        defaults=SimpleNamespace(approx_kernel_max_cycle=None, approx_kernel_max_space=None),
        nroots=1,
        matvec_backend="contract",
    )
    assert out2["max_cycle"] == 2
    assert out2["max_space"] == 4


def test_matvec_runtime_resolve_cuda_j_tile_and_csr_cache():
    auto_j = resolve_cuda_j_tile(
        requested_j_tile=0,
        target_ntasks=1_500_000,
        j_tile_align=256,
        norb=12,
        ncsf=5000,
    )
    assert auto_j > 0
    manual_j = resolve_cuda_j_tile(
        requested_j_tile=777,
        target_ntasks=1_500_000,
        j_tile_align=256,
        norb=12,
        ncsf=5000,
    )
    assert manual_j == 777

    auto_cache_fast = resolve_cuda_cache_csr_tiles(
        cache_csr_tiles_in=None,
        aggregate_offdiag=True,
        use_epq_table=True,
        ncsf=5000,
        j_tile=auto_j,
        norb=12,
        csr_capacity_mult=2.0,
    )
    assert auto_cache_fast is False
    auto_cache_general = resolve_cuda_cache_csr_tiles(
        cache_csr_tiles_in=None,
        aggregate_offdiag=False,
        use_epq_table=False,
        ncsf=5000,
        j_tile=auto_j,
        norb=12,
        csr_capacity_mult=2.0,
    )
    assert isinstance(auto_cache_general, bool)
    forced_cache = resolve_cuda_cache_csr_tiles(
        cache_csr_tiles_in="on",
        aggregate_offdiag=False,
        use_epq_table=False,
        ncsf=5000,
        j_tile=auto_j,
        norb=12,
        csr_capacity_mult=2.0,
    )
    assert forced_cache is True


def test_matvec_runtime_resolve_cuda_workspace_controls_consume_modes():
    defaults = SimpleNamespace(
        matvec_cuda_target_ntasks=10,
        matvec_cuda_j_tile_align=16,
        matvec_cuda_j_tile=4,
        matvec_cuda_csr_capacity_mult=3.0,
        matvec_cuda_csr_host_cache="off",
        matvec_cuda_csr_host_cache_budget_gib=2.5,
        matvec_cuda_csr_host_cache_min_ncsf=12,
        matvec_cuda_csr_pipeline_streams="auto",
        matvec_cuda_csr_pipeline_min_ncsf=13,
        matvec_cuda_prefilter_trivial_tasks="on",
        matvec_cuda_prefilter_trivial_tasks_min_ncsf=14,
        matvec_cuda_threads_enum=96,
        matvec_cuda_threads_g=160,
        matvec_cuda_threads_w=80,
        matvec_cuda_threads_apply=0,
        matvec_cuda_coalesce=True,
        matvec_cuda_include_diagonal_rs=True,
        matvec_cuda_cache_csr_tiles=None,
        matvec_cuda_fuse_count_write=True,
        matvec_cuda_fp32_coeff_data=False,
        matvec_cuda_use_epq_table=None,
        matvec_cuda_aggregate_offdiag=None,
    )
    kwargs = {
        "matvec_cuda_threads_enum": 128,
        "matvec_cuda_threads_apply": 32,
        "matvec_cuda_csr_pipeline_streams": 3,
        "matvec_cuda_aggregate_offdiag": True,
    }
    out = resolve_cuda_workspace_controls(
        kwargs=kwargs,
        defaults=defaults,
        consume=True,
        context="kernel(cuda)",
    )
    assert out["threads_enum_forced"] is True
    assert out["threads_g_forced"] is False
    assert out["matvec_cuda_threads_enum"] == 128
    assert out["matvec_cuda_threads_apply"] == 32
    assert out["threads_apply_auto"] is False
    assert out["matvec_cuda_csr_pipeline_streams_mode"] == "manual"
    assert out["matvec_cuda_csr_pipeline_streams_value"] == 3
    assert out["matvec_cuda_aggregate_offdiag"] is True
    assert kwargs == {}

    kwargs2 = {"matvec_cuda_aggregate_offdiag": True, "matvec_cuda_threads_apply": 0}
    out2 = resolve_cuda_workspace_controls(
        kwargs=kwargs2,
        defaults=defaults,
        consume=False,
        context="approx_kernel(cuda)",
    )
    assert out2["threads_apply_auto"] is True
    assert kwargs2["matvec_cuda_aggregate_offdiag"] is True
    assert kwargs2["matvec_cuda_threads_apply"] == 0

    with pytest.raises(ValueError, match="forbidden by hard guard"):
        resolve_cuda_workspace_controls(
            kwargs={"matvec_cuda_aggregate_offdiag": False},
            defaults=defaults,
            consume=False,
            context="approx_kernel(cuda)",
        )


def test_matvec_runtime_get_or_create_cuda_matvec_state():
    calls = {"drt": 0, "state": 0}

    def _mk_drt(drt):
        calls["drt"] += 1
        return ("drt_dev", drt)

    def _mk_state(drt, drt_dev):
        calls["state"] += 1
        return ("state_dev", drt, drt_dev)

    cache = {}
    s1 = get_or_create_cuda_matvec_state(
        state_cache=cache,
        ws_key=("k",),
        drt="drt_obj",
        make_device_drt_fn=_mk_drt,
        make_device_state_cache_fn=_mk_state,
    )
    s2 = get_or_create_cuda_matvec_state(
        state_cache=cache,
        ws_key=("k",),
        drt="drt_obj",
        make_device_drt_fn=_mk_drt,
        make_device_state_cache_fn=_mk_state,
    )
    assert s1 == s2
    assert calls["drt"] == 1
    assert calls["state"] == 1
    assert cache[("k",)] == s1


def test_matvec_runtime_thread_tuning_helpers():
    tuned = tune_cuda_threads_for_large_cas_noepq(
        threads_enum=128,
        threads_g=256,
        threads_enum_forced=False,
        threads_g_forced=False,
        eri_mat_present=True,
        use_epq_table=False,
        aggregate_offdiag=False,
        nops=256,
        ncsf=1_500_000,
        dtype_mode="float32",
    )
    assert tuned["threads_enum"] == 256
    assert tuned["threads_g"] == 128

    tuned_forced = tune_cuda_threads_for_large_cas_noepq(
        threads_enum=128,
        threads_g=192,
        threads_enum_forced=True,
        threads_g_forced=True,
        eri_mat_present=True,
        use_epq_table=False,
        aggregate_offdiag=False,
        nops=256,
        ncsf=1_500_000,
        dtype_mode="float64",
    )
    assert tuned_forced["threads_enum"] == 128
    assert tuned_forced["threads_g"] == 192

    assert resolve_cuda_threads_w(threads_w=0, threads_g=160) == 160
    assert resolve_cuda_threads_w(threads_w=64, threads_g=160) == 64

    assert resolve_cuda_threads_apply(
        threads_apply=0,
        use_epq_table=True,
        dtype_mode="float32",
        ncsf=2_000_000,
        nops=100,
        noepq_large_ncsf_uses_64=True,
    ) == 64
    assert resolve_cuda_threads_apply(
        threads_apply=0,
        use_epq_table=False,
        dtype_mode="float64",
        ncsf=2_000_000,
        nops=100,
        noepq_large_ncsf_uses_64=True,
    ) == 64
    assert resolve_cuda_threads_apply(
        threads_apply=0,
        use_epq_table=False,
        dtype_mode="float64",
        ncsf=2_000_000,
        nops=100,
        noepq_large_ncsf_uses_64=False,
    ) == 32


def test_matvec_runtime_ws_needs_rebuild_and_budget_helpers():
    ws = SimpleNamespace(
        dtype=np.float64,
        j_tile=128,
        csr_capacity_mult=2.0,
        threads_enum=128,
        threads_g=256,
        threads_w=128,
        threads_apply=256,
        max_g_bytes=1_048_576,
        coalesce=True,
        include_diagonal_rs=True,
        fuse_count_write=True,
        fp32_coeff_data=False,
        path_mode_requested="auto",
        path_mode="auto",
        use_fused_hop=True,
        use_epq_table=True,
        aggregate_offdiag_k=True,
        l_full=object(),
        offdiag_enable_fp64_emulation=False,
        gemm_backend="cublaslt_fp64",
        offdiag_emulation_strategy="none",
        offdiag_cublas_workspace_cap_mb=256,
        apply_mode="dense",
        epq_build_device=True,
        epq_build_j_tile=128,
        epq_streaming=False,
        epq_stream_j_tile=128,
        epq_stream_use_recompute="auto",
        cache_csr_tiles=True,
        csr_host_cache_mode="off",
        csr_host_cache_budget_gib=1.0,
        csr_host_cache_min_ncsf=1000,
        csr_pipeline_streams_mode="off",
        csr_pipeline_streams=2,
        csr_pipeline_min_ncsf=1000,
        prefilter_trivial_tasks_mode="auto",
        prefilter_trivial_tasks_min_ncsf=2000,
        naux=5,
    )
    l_full = SimpleNamespace(shape=(36, 5))
    params = dict(
        expected_dtype=np.float64,
        j_tile=128,
        csr_capacity_mult=2.0,
        threads_enum=128,
        threads_g=256,
        threads_w=128,
        threads_apply=256,
        max_g_bytes=1_048_576,
        coalesce=True,
        include_diagonal_rs=True,
        fuse_count_write=True,
        fp32_coeff_data=False,
        path_mode="auto",
        use_fused_hop=True,
        use_epq_table=True,
        aggregate_offdiag_k=True,
        l_full_d=l_full,
        enable_fp64_emulation=False,
        gemm_backend="cublaslt_fp64",
        emulation_strategy="none",
        cublas_workspace_cap_mb=256,
        apply_mode="dense",
        epq_build_device=True,
        epq_build_j_tile=128,
        epq_streaming=False,
        epq_stream_j_tile=128,
        epq_stream_use_recompute="auto",
        cache_csr_tiles=True,
        csr_host_cache_mode="off",
        csr_host_cache_budget_gib=1.0,
        csr_host_cache_min_ncsf=1000,
        csr_pipeline_streams_mode="off",
        csr_pipeline_streams_value=2,
        csr_pipeline_min_ncsf=1000,
        prefilter_trivial_tasks_mode="auto",
        prefilter_trivial_tasks_min_ncsf=2000,
    )
    assert ws_needs_rebuild(ws, **params) is False
    assert ws_needs_rebuild(ws, **(params | {"j_tile": 256})) is True
    assert ws_needs_rebuild(None, **params) is True

    class _Runtime:
        @staticmethod
        def memGetInfo():
            return (int(2 * 1024**3), int(8 * 1024**3))

    cp_mod = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime()))
    budget = resolve_matvec_cuda_ws_cache_budget_bytes(
        cp_mod=cp_mod,
        hard_cap_gib=0.0,
        fraction=0.25,
    )
    expected = int((8 * 1024**3 - int(1.5 * 1024**3)) * 0.25)
    assert budget == expected
    assert resolve_matvec_cuda_ws_cache_budget_bytes(cp_mod=cp_mod, hard_cap_gib=2.0, fraction=0.25) < budget


def test_matvec_runtime_workspace_release_and_estimate_helpers():
    class _WsWithEstimate:
        def workspace_nbytes_estimate(self):
            return 1234

    assert estimate_matvec_cuda_workspace_bytes(_WsWithEstimate()) == 1234

    class _Buffer:
        nbytes = 64

    ws_tree = {
        "g": _Buffer(),
        "w": {"a": _Buffer(), "b": _Buffer()},
        "cache": [_Buffer()],
    }
    assert estimate_matvec_cuda_workspace_bytes(ws_tree) == 256

    class _WsWithRelease:
        def __init__(self):
            self.calls = 0

        def release(self):
            self.calls += 1

    ws_release = _WsWithRelease()
    release_matvec_cuda_workspace(ws_release)
    assert ws_release.calls == 1

    ws_fallback = SimpleNamespace(
        _g_buf=object(),
        _w_block=object(),
        _w_offdiag=object(),
        _csr_tile_cache=object(),
    )
    release_matvec_cuda_workspace(ws_fallback)
    assert ws_fallback._g_buf is None
    assert ws_fallback._w_block is None
    assert ws_fallback._w_offdiag is None
    assert ws_fallback._csr_tile_cache is None


def test_matvec_cache_runtime_lifecycle_budget_and_profile():
    solver, released = _make_ws_cache_solver_stub(budget=50, fraction=0.25)
    ws_a = SimpleNamespace(nbytes=30)
    ws_b = SimpleNamespace(nbytes=40)

    matvec_cuda_ws_cache_touch(solver, "a")
    assert list(solver._matvec_cuda_ws_cache_lru.keys()) == ["a"]

    matvec_cuda_ws_cache_put(solver, "a", ws_a)
    assert solver._matvec_cuda_ws_cache_bytes == 30
    assert solver._matvec_cuda_ws_cache_sizes["a"] == 30

    assert matvec_cuda_ws_cache_get(solver, "a") is ws_a
    assert matvec_cuda_ws_cache_get(solver, "missing") is None
    assert solver._matvec_cuda_ws_cache_hits == 1
    assert solver._matvec_cuda_ws_cache_misses == 1

    matvec_cuda_ws_cache_put(solver, "b", ws_b)
    assert solver._matvec_cuda_ws_cache_bytes == 40
    assert "a" not in solver._matvec_cuda_ws_cache
    assert "b" in solver._matvec_cuda_ws_cache
    assert solver._matvec_cuda_ws_cache_evictions == 1
    assert released == [ws_a]

    matvec_cuda_ws_cache_drop(solver, "b")
    assert solver._matvec_cuda_ws_cache_bytes == 0
    assert released == [ws_a, ws_b]
    assert matvec_cuda_ws_cache_profile(solver)["matvec_cuda_ws_cache_entries"] == 0


def test_matvec_cache_runtime_keep_keys_release_and_configure():
    solver, released = _make_ws_cache_solver_stub(budget=80, fraction=0.2)
    ws_a = SimpleNamespace(nbytes=45)
    ws_b = SimpleNamespace(nbytes=45)
    ws_c = SimpleNamespace(nbytes=45)
    matvec_cuda_ws_cache_put(solver, "a", ws_a)
    matvec_cuda_ws_cache_put(solver, "b", ws_b)
    matvec_cuda_ws_cache_put(solver, "c", ws_c, keep_keys=("a",))

    assert "a" not in solver._matvec_cuda_ws_cache
    assert "c" in solver._matvec_cuda_ws_cache
    assert solver._matvec_cuda_ws_cache_evictions == 2
    assert released == [ws_a, ws_b]
    assert solver._matvec_cuda_ws_cache_bytes <= solver._matvec_cuda_ws_cache_budget_bytes

    total_before_release = release_matvec_cuda_ws_cache(solver)
    assert total_before_release == 45
    assert solver._matvec_cuda_ws_cache == {}
    assert released[-1] is ws_c

    calls = {}

    def _norm(v):
        calls["fraction_in"] = v
        return 0.6

    def _resolve(*, cp_mod, hard_cap_gib, fraction):
        calls["resolve"] = (cp_mod, hard_cap_gib, fraction)
        return 123

    out_budget = configure_matvec_cuda_ws_cache(
        solver,
        cp_mod="cp",
        hard_cap_gib=4.0,
        fraction=0.4,
        normalize_ws_cache_fraction_fn=_norm,
        resolve_budget_bytes_fn=_resolve,
    )
    assert out_budget == 123
    assert solver.matvec_cuda_ws_cache_fraction == 0.6
    assert solver._matvec_cuda_ws_cache_budget_bytes == 123
    assert calls["fraction_in"] == 0.4
    assert calls["resolve"] == ("cp", 4.0, 0.6)

    solver._matvec_cuda_ws_cache_bytes = 999
    solver._matvec_cuda_ws_cache_budget_bytes = 0
    matvec_cuda_ws_cache_enforce_budget(solver)
    assert solver._matvec_cuda_ws_cache_bytes == 999


def test_dump_flags_runtime_silent_and_verbose_output():
    sink = io.StringIO()
    solver = SimpleNamespace(
        verbose=0,
        stdout=sink,
        twos=0,
        wfnsym=1,
        orbsym=np.asarray([1, 1, 2], dtype=np.int32),
        conv_tol=1e-9,
        kernel_blas_nthreads=4,
        rdm_backend="auto",
        matvec_cuda_policy="auto",
    )

    out = dump_flags(solver, verbose=None)
    assert out is solver
    assert sink.getvalue() == ""

    dump_flags(solver, verbose=1)
    text = sink.getvalue()
    assert "GUGAFCISolver (CSF/GUGA)" in text
    assert "twos = 0" in text
    assert "wfnsym = 1" in text
    assert "orbsym = [1, 1, 2]" in text
    assert "conv_tol = 1e-09" in text
    assert "kernel_blas_nthreads = 4" in text
    assert "rdm_backend = auto" in text
    assert "matvec_cuda_policy = auto" in text


def test_kernel_runtime_resolve_nroots_and_warm_start_and_dry_run():
    assert resolve_kernel_nroots(requested_nroots=None, defaults=SimpleNamespace(nroots=2), ncsf=3) == 2
    with pytest.raises(ValueError, match="nroots must be >= 1"):
        resolve_kernel_nroots(requested_nroots=0, defaults=object(), ncsf=3)
    with pytest.raises(ValueError, match="> ncsf="):
        resolve_kernel_nroots(requested_nroots=4, defaults=object(), ncsf=3)

    calls = {}

    def _warm_fn(**kwargs):
        calls["warm"] = kwargs
        return ([np.asarray([1.0, 0.0])], "applied")

    out = resolve_kernel_warm_start(
        ci0=None,
        warm_state_enable=True,
        warm_state_ci0_if_compatible_fn=_warm_fn,
        norb=2,
        nelec_total=2,
        twos=0,
        nroots=1,
        ncsf=2,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
        matvec_backend="cuda",
        cas_metadata={"ncas": 2, "nelecas": (1, 1)},
    )
    assert out["warm_applied"] is True
    assert out["warm_reason"] == "applied"
    assert out["ci0"] is not None and len(out["ci0"]) == 1
    assert calls["warm"]["ncsf"] == 2

    out2 = resolve_kernel_warm_start(
        ci0=[np.asarray([0.0, 1.0])],
        warm_state_enable=True,
        warm_state_ci0_if_compatible_fn=lambda **_kw: (_kw, "unused"),
        norb=2,
        nelec_total=2,
        twos=0,
        nroots=1,
        ncsf=2,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
        matvec_backend="contract",
        cas_metadata={},
    )
    assert out2["warm_applied"] is False
    assert out2["warm_reason"] == "ci0_provided"

    dry = build_kernel_dry_run_result(
        ci0=None,
        nroots=2,
        ncsf=3,
        ecore=1.25,
        normalize_ci0_fn=lambda *_args, **_kwargs: None,
    )
    assert dry["e"].tolist() == [1.25, 1.25]
    assert np.allclose(dry["ci"][0], np.asarray([1.0, 0.0, 0.0]))
    assert np.allclose(dry["ci"][1], np.asarray([0.0, 1.0, 0.0]))

    marker = {"called": False}

    def _norm(ci0, *, nroots, ncsf):
        marker["called"] = True
        assert nroots == 1 and ncsf == 2
        return [np.asarray(ci0[0], dtype=np.float64)]

    dry2 = build_kernel_dry_run_result(
        ci0=[np.asarray([0.2, 0.8])],
        nroots=1,
        ncsf=2,
        ecore=0.0,
        normalize_ci0_fn=_norm,
    )
    assert marker["called"] is True
    assert np.allclose(dry2["ci"][0], np.asarray([0.2, 0.8]))


def test_kernel_runtime_dense_fastpath_and_precompute_state_cache():
    calls = {"precompute": 0, "state_cache": 0, "warm": 0}

    class _StubSolver:
        dense_eigh_ncsf_threshold = 10

        def pspace(self, *_args, **_kwargs):
            addr = np.asarray([0, 1], dtype=np.int64)
            h = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
            return addr, h

        def _update_warm_state(self, **_kwargs):
            calls["warm"] += 1

    solver = _StubSolver()
    out = run_kernel_dense_eigh_fastpath(
        solver=solver,
        h1e=np.zeros((2, 2)),
        eri=np.zeros((2, 2, 2, 2)),
        norb=2,
        nelec=2,
        ncsf=2,
        nroots=1,
        ecore=0.5,
        max_out=10,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
        drt_key=("k",),
        warm_state_update=True,
        nelec_total=2,
        twos=0,
        cas_metadata={"ncas": 2},
        warm_state_mo_coeff=None,
        warm_state_mo_occ=None,
        t_kernel0=0.0,
    )
    assert out is not None
    assert out["e"].shape == (1,)
    assert out["e"][0] == pytest.approx(1.5)
    assert np.allclose(out["ci"][0], np.asarray([1.0, 0.0]))
    assert calls["warm"] == 1

    out2 = run_kernel_dense_eigh_fastpath(
        solver=solver,
        h1e=np.zeros((2, 2)),
        eri=np.zeros((2, 2, 2, 2)),
        norb=2,
        nelec=2,
        ncsf=20,
        nroots=1,
        ecore=0.0,
        max_out=10,
        orbsym=None,
        wfnsym=None,
        ne_constraints=None,
        drt_key=("k",),
        warm_state_update=False,
        nelec_total=2,
        twos=0,
        cas_metadata={},
        warm_state_mo_coeff=None,
        warm_state_mo_occ=None,
        t_kernel0=0.0,
    )
    assert out2 is None

    state_cache = prepare_kernel_precompute_and_state_cache(
        precompute_epq=True,
        drt="drt",
        matvec_backend="row_oracle_df",
        row_oracle_use_state_cache=True,
        precompute_epq_actions_fn=lambda _drt: calls.__setitem__("precompute", calls["precompute"] + 1),
        get_state_cache_fn=lambda _drt: calls.__setitem__("state_cache", calls["state_cache"] + 1) or {"ok": True},
    )
    assert state_cache == {"ok": True}
    assert calls["precompute"] == 1
    assert calls["state_cache"] == 1


def test_kernel_runtime_row_screening_and_contract_eri_restore_helpers():
    class _RowScreening:
        pass

    rs = _RowScreening()
    assert normalize_row_screening(row_screening=rs, row_screening_type=_RowScreening) is rs
    assert normalize_row_screening(row_screening=None, row_screening_type=_RowScreening) is None
    with pytest.raises(TypeError, match="row_screening must be a RowScreening or None"):
        normalize_row_screening(row_screening=object(), row_screening_type=_RowScreening)

    restored = np.zeros((2, 2, 2, 2), dtype=np.float64)
    calls = {"restore": 0}

    def _restore(_eri, _norb):
        calls["restore"] += 1
        return restored

    eri_pair = np.zeros((3, 3), dtype=np.float64)
    out = maybe_restore_contract_eri(
        matvec_backend="contract",
        eri=eri_pair,
        norb=2,
        df_types=(),
        restore_eri1_fn=_restore,
    )
    assert calls["restore"] == 1
    assert out is restored

    calls["restore"] = 0
    out2 = maybe_restore_contract_eri(
        matvec_backend="contract",
        eri=np.zeros((2, 2, 2, 2), dtype=np.float64),
        norb=2,
        df_types=(),
        restore_eri1_fn=_restore,
    )
    assert calls["restore"] == 0
    assert out2.shape == (2, 2, 2, 2)

    calls["restore"] = 0
    out3 = maybe_restore_contract_eri(
        matvec_backend="cuda",
        eri=eri_pair,
        norb=2,
        df_types=(),
        restore_eri1_fn=_restore,
    )
    assert calls["restore"] == 0
    assert out3 is eri_pair


def test_kernel_runtime_build_cuda_hamiltonian_inputs_df_path():
    class _FakeCP:
        float64 = np.float64

        @staticmethod
        def asarray(arr, dtype=None):
            return np.asarray(arr, dtype=dtype)

        @staticmethod
        def ascontiguousarray(arr):
            return np.ascontiguousarray(arr)

        @staticmethod
        def dot(a, b):
            return np.dot(a, b)

    class _DF:
        def __init__(self):
            self.l_full = np.eye(4, dtype=np.float64)
            self.j_ps = np.ones((2, 2), dtype=np.float64)

    out = build_cuda_hamiltonian_inputs(
        cp=_FakeCP(),
        eri=_DF(),
        h1e=np.eye(2, dtype=np.float64),
        norb=2,
        df_eri_mat_max_bytes=1024 * 1024,
        df_type=_DF,
        device_df_type=type("_DeviceDF", (), {}),
        restore_eri_4d_fn=lambda *_a, **_k: None,
    )
    assert out["eri_mat_d"] is not None
    assert out["l_full_d"] is None
    assert tuple(np.asarray(out["h_eff_d"]).shape) == (2, 2)


def test_kernel_runtime_autotune_cuda_max_g_mib_for_large_cas_guards():
    # Guarded path: forced setting should bypass tuning.
    out_forced = autotune_cuda_max_g_mib_for_large_cas(
        max_g_mib=256.0,
        max_g_forced=True,
        aggregate_offdiag=True,
        ncsf=2_000_000,
        norb=12,
        matvec_cuda_dtype="float64",
        eri_mat_present=True,
        mem_hard_cap_gib=0.0,
        cuda_budget_free_bytes_fn=lambda *_a, **_k: None,
    )
    assert out_forced == 256.0

    # Non-large-CAS path should also bypass tuning.
    out_small = autotune_cuda_max_g_mib_for_large_cas(
        max_g_mib=256.0,
        max_g_forced=False,
        aggregate_offdiag=True,
        ncsf=500_000,
        norb=12,
        matvec_cuda_dtype="float64",
        eri_mat_present=True,
        mem_hard_cap_gib=0.0,
        cuda_budget_free_bytes_fn=lambda *_a, **_k: None,
    )
    assert out_small == 256.0


def test_kernel_cuda_runtime_auto_select_use_epq_table():
    class _Pool:
        @staticmethod
        def total_bytes():
            return 0

        @staticmethod
        def used_bytes():
            return 0

    class _Runtime:
        @staticmethod
        def memGetInfo():
            return (6 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024)

    cp_mod = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime()), get_default_memory_pool=lambda: _Pool())
    assert auto_select_use_epq_table(
        cp=cp_mod,
        norb=14,
        ncsf=200_000,
        aggregate_offdiag=True,
        has_epq_table_device_build=True,
        mem_hard_cap_gib=11.5,
        dtype_mode="float64",
        eri_mat_present=True,
    )
    assert not auto_select_use_epq_table(
        cp=cp_mod,
        norb=28,
        ncsf=200_000,
        aggregate_offdiag=True,
        has_epq_table_device_build=True,
        mem_hard_cap_gib=11.5,
        dtype_mode="float64",
        eri_mat_present=True,
    )


def test_kernel_cuda_runtime_common_policy_helpers():
    out = resolve_cuda_workspace_policy_common(
        apply_mode="auto",
        apply_mode_forced=False,
        use_epq_table=False,
        dtype_mode="float64",
        ncsf=1_200_000,
        nops=196,
        threads_enum=96,
        threads_g=128,
        threads_w=0,
        threads_apply=0,
        threads_enum_forced=False,
        threads_g_forced=False,
        threads_apply_auto=True,
        eri_mat_present=True,
        aggregate_offdiag=True,
        max_g_mib=4096.0,
        mem_hard_cap_gib=11.5,
        cache_csr_tiles_in=None,
        j_tile=256,
        norb=14,
        csr_capacity_mult=2.0,
        noepq_large_ncsf_uses_64=True,
    )
    assert out["threads_enum"] > 0
    assert out["threads_g"] > 0
    assert out["threads_w"] > 0
    assert out["threads_apply"] == 64
    assert out["max_g_mib"] <= 512.0
    assert out["cache_csr_tiles"] in (True, False)


def test_kernel_cuda_runtime_validate_low_precision_path_guards():
    out = validate_low_precision_cuda_path(
        context="test(cuda)",
        dtype_mode="mixed",
        use_epq_table=True,
        aggregate_offdiag=True,
        ncsf=1_200_000,
        eri_mat_present=False,
        enable_fp64_emulation=False,
        use_graph=True,
    )
    assert out["use_graph"] is False

    with pytest.raises(ValueError, match="requires dense eri_mat"):
        validate_low_precision_cuda_path(
            context="test(cuda)",
            dtype_mode="float32",
            use_epq_table=False,
            aggregate_offdiag=True,
            ncsf=100,
            eri_mat_present=False,
            enable_fp64_emulation=False,
            use_graph=False,
        )


def test_kernel_cuda_runtime_resolve_epq_streaming_controls():
    out = resolve_epq_streaming_controls(
        epq_streaming_in="on",
        epq_stream_j_tile_in=-7,
        epq_stream_use_recompute_in="off",
    )
    assert out["streaming_mode"] == "on"
    assert out["streaming"] is True
    assert out["stream_j_tile"] == 0
    assert out["stream_use_recompute"] is False

    out2 = resolve_epq_streaming_controls(
        epq_streaming_in=None,
        epq_stream_j_tile_in=64,
        epq_stream_use_recompute_in=None,
    )
    assert out2["streaming_mode"] == "auto"
    assert out2["streaming"] is False
    assert out2["stream_j_tile"] == 64
    assert out2["stream_use_recompute"] == "auto"

    with pytest.raises(ValueError, match="matvec_cuda_epq_streaming"):
        resolve_epq_streaming_controls(
            epq_streaming_in="maybe",
            epq_stream_j_tile_in=0,
            epq_stream_use_recompute_in="auto",
        )
    with pytest.raises(ValueError, match="matvec_cuda_epq_stream_use_recompute"):
        resolve_epq_streaming_controls(
            epq_streaming_in="auto",
            epq_stream_j_tile_in=0,
            epq_stream_use_recompute_in="sometimes",
        )


def test_kernel_cuda_runtime_apply_low_precision_and_workspace_policy():
    out = apply_low_precision_and_workspace_policy(
        context="kernel(cuda)",
        dtype_mode="mixed",
        use_epq_table=True,
        aggregate_offdiag=True,
        ncsf=1_200_000,
        eri_mat_present=False,
        enable_fp64_emulation=False,
        use_graph=True,
        apply_mode="auto",
        apply_mode_forced=False,
        nops=196,
        threads_enum=96,
        threads_g=128,
        threads_w=0,
        threads_apply=0,
        threads_enum_forced=False,
        threads_g_forced=False,
        threads_apply_auto=True,
        max_g_mib=4096.0,
        mem_hard_cap_gib=11.5,
        cache_csr_tiles_in=None,
        j_tile=256,
        norb=14,
        csr_capacity_mult=2.0,
        noepq_large_ncsf_uses_64=True,
    )
    assert out["graph_disabled"] is True
    assert out["use_graph"] is False
    assert out["threads_enum"] > 0
    assert out["threads_g"] > 0
    assert out["threads_w"] > 0
    assert out["threads_apply"] == 64
    assert out["max_g_mib"] <= 512.0
    assert out["cache_csr_tiles"] in (True, False)


def test_init_runtime_defaults_env_override_and_normalization():
    seen: dict[str, object] = {}

    def _norm(v):
        seen["norm_in"] = v
        return 0.37

    def _auto_cap():
        seen["auto_cap_called"] = True
        return 6.5

    solver = SimpleNamespace(
        matvec_cuda_ws_cache_fraction=0.19,
        matvec_cuda_mem_hard_cap_gib=3.0,
        unconverged_fallback_ncsf_max=0,
    )
    configure_solver_runtime_defaults(
        solver,
        normalize_ws_cache_fraction_fn=_norm,
        auto_gpu_mem_hard_cap_fn=_auto_cap,
        env_get_fn=lambda key: "0.91" if key == "ASUKA_GPU_WS_CACHE_FRAC" else None,
    )

    assert seen["norm_in"] == "0.91"
    assert "auto_cap_called" in seen
    assert solver.matvec_cuda_ws_cache_fraction == 0.37
    # Existing explicit value is preserved via getattr default path.
    assert solver.matvec_cuda_mem_hard_cap_gib == 3.0
    assert solver.unconverged_fallback_ncsf_max == 1
    assert solver.matvec_backend == "contract"
    assert solver.contract_nthreads == 0


def test_init_runtime_defaults_fallback_to_attr_when_env_missing():
    seen: dict[str, object] = {}

    def _norm(v):
        seen["norm_in"] = v
        return float(v)

    solver = SimpleNamespace(matvec_cuda_ws_cache_fraction=0.23)
    configure_solver_runtime_defaults(
        solver,
        normalize_ws_cache_fraction_fn=_norm,
        auto_gpu_mem_hard_cap_fn=lambda: 7.0,
        env_get_fn=lambda _key: None,
    )
    assert seen["norm_in"] == 0.23
    assert solver.matvec_cuda_ws_cache_fraction == 0.23
    assert solver.matvec_cuda_mem_hard_cap_gib == 7.0
    assert solver.kernel_profile is False
    assert solver._last_warm_start_info is None


def test_warm_state_metadata_validation():
    with pytest.raises(TypeError):
        normalize_warm_cas_metadata("bad", default_ncas=4, default_nelecas=(2, 2))
    with pytest.raises(ValueError):
        normalize_warm_cas_metadata({"nelecas": [2]}, default_ncas=4, default_nelecas=(2, 2))
    with pytest.raises(ValueError):
        warm_cas_metadata_from_jsonable({"nelecas": [2]})
    with pytest.raises(ValueError):
        ne_constraints_to_key({-1: (0, 1)})
    with pytest.raises(ValueError):
        ne_constraints_to_key({1: (2, 1)})


def test_cuda_policy_normalizers_and_path_mode():
    assert normalize_cuda_user_policy_mode(None) == "auto"
    assert normalize_cuda_user_policy_mode(True) == "on"
    assert normalize_cuda_user_policy_mode("off") == "off"
    assert normalize_cuda_accuracy_mode("performance") == "fast"
    assert normalize_matvec_cuda_path_mode("epq") == "epq_blocked"
    assert normalize_matvec_cuda_path_mode("fused-epq-hybrid") == "fused_epq_hybrid"
    with pytest.raises(ValueError):
        normalize_matvec_cuda_path_mode("fused_coo")


def test_apply_cuda_user_policy_defaults_only_for_cuda_backends():
    kwargs = {}
    applied, reason, resolved = apply_cuda_user_policy(
        matvec_backend="cuda",
        policy_mode="on",
        accuracy_mode="balanced",
        dtype_hint="fp64",
        memory_cap_gib=8.0,
        kwargs=kwargs,
        policy_explicit=True,
        policy_configured=False,
    )
    assert applied is True
    assert reason == "policy_on"
    assert "matvec_cuda_mem_hard_cap_gib" in resolved
    assert kwargs["matvec_cuda_gemm_backend"] == "cublaslt_fp64"

    non_cuda = {}
    applied2, reason2, resolved2 = apply_cuda_user_policy(
        matvec_backend="cpu_sparse",
        policy_mode="on",
        accuracy_mode="balanced",
        dtype_hint="fp64",
        memory_cap_gib=8.0,
        kwargs=non_cuda,
        policy_explicit=True,
        policy_configured=False,
    )
    assert applied2 is False
    assert reason2 == "non_cuda_backend"
    assert resolved2 == {}
    assert non_cuda == {}


def test_resolve_kernel_cuda_policy_helper_behavior():
    kwargs = {"matvec_cuda_policy": "on", "matvec_cuda_aggregate_offdiag": True}
    defaults = SimpleNamespace(
        matvec_cuda_policy="auto",
        matvec_cuda_accuracy_mode="balanced",
        matvec_cuda_memory_cap_gib=None,
        matvec_cuda_dtype="float64",
        matvec_cuda_aggregate_offdiag=None,
    )
    out = resolve_kernel_cuda_policy(
        kwargs=kwargs,
        defaults=defaults,
        matvec_backend="cuda",
        strict_gpu=False,
    )
    assert out["matvec_cuda_aggregate_offdiag_preview"] is True
    assert out["profile"]["matvec_cuda_policy"] == "on"
    assert out["profile"]["matvec_cuda_policy_applied"] is True
    assert kwargs["matvec_cuda_use_epq_table"] is True

    with pytest.raises(ValueError, match="must be > 0"):
        resolve_kernel_cuda_policy(
            kwargs={"matvec_cuda_memory_cap_gib": 0.0},
            defaults=defaults,
            matvec_backend="cuda",
            strict_gpu=False,
        )
    with pytest.raises(ValueError, match="forbidden by hard guard"):
        resolve_kernel_cuda_policy(
            kwargs={"matvec_cuda_aggregate_offdiag": False},
            defaults=defaults,
            matvec_backend="cuda",
            strict_gpu=False,
        )

    out2 = resolve_kernel_cuda_policy(
        kwargs={},
        defaults=defaults,
        matvec_backend="contract",
        strict_gpu=True,
    )
    assert out2["matvec_cuda_aggregate_offdiag_preview"] is None
    assert out2["profile"]["strict_gpu"] is True


def test_resolve_cuda_memory_controls_consume_and_readonly():
    defaults = SimpleNamespace(
        matvec_cuda_mem_hard_cap_gib=9.5,
        matvec_cuda_ws_cache_fraction=0.2,
    )
    kwargs = {"matvec_cuda_mem_hard_cap_gib": "6.0", "matvec_cuda_ws_cache_fraction": "0.4"}
    out = resolve_cuda_memory_controls(kwargs=kwargs, defaults=defaults, consume=False)
    assert out["matvec_cuda_mem_hard_cap_gib"] == pytest.approx(6.0)
    assert out["matvec_cuda_ws_cache_fraction"] == pytest.approx(0.4)
    assert "matvec_cuda_mem_hard_cap_gib" in kwargs
    assert "matvec_cuda_ws_cache_fraction" in kwargs

    out2 = resolve_cuda_memory_controls(kwargs=kwargs, defaults=defaults, consume=True)
    assert out2["matvec_cuda_mem_hard_cap_gib"] == pytest.approx(6.0)
    assert out2["matvec_cuda_ws_cache_fraction"] == pytest.approx(0.4)
    assert kwargs == {}

    out3 = resolve_cuda_memory_controls(kwargs={}, defaults=defaults, consume=False)
    assert out3["matvec_cuda_mem_hard_cap_gib"] == pytest.approx(9.5)
    assert out3["matvec_cuda_ws_cache_fraction"] == pytest.approx(0.2)


def test_resolve_cuda_max_g_and_hard_cap_clamp():
    defaults = SimpleNamespace(matvec_cuda_max_g_mib=256.0)
    kwargs = {"matvec_cuda_max_g_mib": "1024"}
    out = resolve_cuda_max_g_mib(kwargs=kwargs, defaults=defaults, consume=False)
    assert out["max_g_forced"] is True
    assert out["matvec_cuda_max_g_mib"] == pytest.approx(1024.0)
    assert "matvec_cuda_max_g_mib" in kwargs

    out2 = resolve_cuda_max_g_mib(kwargs=kwargs, defaults=defaults, consume=True)
    assert out2["max_g_forced"] is True
    assert out2["matvec_cuda_max_g_mib"] == pytest.approx(1024.0)
    assert kwargs == {}

    out3 = resolve_cuda_max_g_mib(kwargs={}, defaults=SimpleNamespace(matvec_cuda_max_g_mib=-1.0), consume=False)
    assert out3["max_g_forced"] is False
    assert out3["matvec_cuda_max_g_mib"] == pytest.approx(256.0)

    assert cap_cuda_max_g_mib_by_hard_cap(max_g_mib=2048.0, hard_cap_gib=12.0) == pytest.approx(512.0)
    assert cap_cuda_max_g_mib_by_hard_cap(max_g_mib=2048.0, hard_cap_gib=16.0) == pytest.approx(1024.0)
    assert cap_cuda_max_g_mib_by_hard_cap(max_g_mib=768.0, hard_cap_gib=0.0) == pytest.approx(768.0)


def test_resolve_cuda_apply_mode_and_epq_build_device():
    defaults = SimpleNamespace(matvec_cuda_apply_mode="auto")
    kwargs = {"matvec_cuda_apply_mode": "scatter"}
    out = resolve_cuda_apply_mode(kwargs=kwargs, defaults=defaults, consume=False)
    assert out["apply_mode_forced"] is True
    assert out["matvec_cuda_apply_mode"] == "scatter"
    assert "matvec_cuda_apply_mode" in kwargs

    out2 = resolve_cuda_apply_mode(kwargs=kwargs, defaults=defaults, consume=True)
    assert out2["apply_mode_forced"] is True
    assert kwargs == {}
    with pytest.raises(ValueError, match="must be one of"):
        resolve_cuda_apply_mode(kwargs={"matvec_cuda_apply_mode": "bad"}, defaults=defaults, consume=False)

    assert maybe_promote_cuda_apply_mode_scatter(
        apply_mode="auto",
        apply_mode_forced=False,
        use_epq_table=True,
        dtype_mode="float32",
        ncsf=2_000_000,
    ) == "scatter"
    assert maybe_promote_cuda_apply_mode_scatter(
        apply_mode="auto",
        apply_mode_forced=True,
        use_epq_table=True,
        dtype_mode="float32",
        ncsf=2_000_000,
    ) == "auto"

    assert resolve_cuda_epq_build_device(
        epq_build_device=None,
        use_epq_table=True,
        has_epq_table_device_build=True,
    ) is True
    assert resolve_cuda_epq_build_device(
        epq_build_device=False,
        use_epq_table=True,
        has_epq_table_device_build=True,
    ) is False
    with pytest.raises(RuntimeError, match="requires a rebuilt CUDA extension"):
        resolve_cuda_epq_build_device(
            epq_build_device=True,
            use_epq_table=True,
            has_epq_table_device_build=False,
        )


def test_resolve_cuda_cublas_workspace_cap_mb_and_clamp():
    defaults = SimpleNamespace(matvec_cuda_cublas_workspace_cap_mb=2048)
    kwargs = {"matvec_cuda_cublas_workspace_cap_mb": "1024"}
    out = resolve_cuda_cublas_workspace_cap_mb(
        kwargs=kwargs,
        defaults=defaults,
        hard_cap_gib=16.0,
        consume=False,
    )
    assert out == 512
    assert "matvec_cuda_cublas_workspace_cap_mb" in kwargs

    out2 = resolve_cuda_cublas_workspace_cap_mb(
        kwargs=kwargs,
        defaults=defaults,
        hard_cap_gib=12.0,
        consume=True,
    )
    assert out2 == 256
    assert kwargs == {}

    out3 = resolve_cuda_cublas_workspace_cap_mb(
        kwargs={},
        defaults=defaults,
        hard_cap_gib=0.0,
        consume=False,
    )
    assert out3 == 2048
    assert cap_cuda_cublas_workspace_cap_mb_by_hard_cap(
        cublas_workspace_cap_mb=4096,
        hard_cap_gib=12.0,
    ) == 256


def test_ws_fraction_and_epq_overbudget_resolution():
    assert normalize_ws_cache_fraction(-1) == 0.0
    assert normalize_ws_cache_fraction(2.5) == 0.8
    assert normalize_ws_cache_fraction("bad") == 0.2

    action, why = resolve_epq_overbudget_action(
        matvec_cuda_dtype="mixed",
        matvec_cuda_aggregate_offdiag=True,
        ncsf=2_000_000,
        epq_table_forced=False,
        epq_streaming_mode="off",
        has_epq_table_device_build=True,
    )
    assert action == "keep_materialized"
    assert why == "mixed_guarded_keep_materialized"


def test_auto_num_threads_is_positive():
    assert int(auto_num_threads()) >= 1


def test_resolve_kernel_frontend_controls_defaults_and_clamps():
    defaults = SimpleNamespace(
        kernel_profile=True,
        kernel_profile_cuda_sync=False,
        kernel_profile_print=False,
        matvec_cuda_hop_profile=True,
        matvec_cuda_davidson_subspace_eigh_cpu=True,
        matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff=10,
        matvec_cuda_davidson_subspace_eigh_cpu_max_m=8,
        orbsym=(1, 2),
        wfnsym=0,
        matvec_backend="Contract",
        strict_gpu=False,
    )
    kwargs = {
        "matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff": -1,
        "matvec_cuda_davidson_subspace_eigh_cpu_max_m": -5,
        "matvec_backend": "CUDA",
        "strict_gpu": 1,
        "dry_run": 1,
        "warm_state_enable": 0,
        "warm_state_update": 1,
        "warm_state_context": {"ncas": 2},
        "mo_coeff": "C",
        "mo_occ": "occ",
        "orbsym": (3, 3),
        "wfnsym": 2,
    }
    out = resolve_kernel_frontend_controls(kwargs=kwargs, defaults=defaults)
    assert out["kernel_profile"] is True
    assert out["matvec_cuda_hop_profile"] is True
    assert out["matvec_cuda_davidson_subspace_eigh_cpu_in"] is True
    assert out["matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff"] == 0
    assert out["matvec_cuda_davidson_subspace_eigh_cpu_max_m"] == 0
    assert out["matvec_backend"] == "cuda"
    assert out["strict_gpu"] is True
    assert out["dry_run"] is True
    assert out["warm_state_enable"] is False
    assert out["warm_state_update"] is True
    assert out["warm_state_context_in"] == {"ncas": 2}
    assert out["warm_state_mo_coeff"] == "C"
    assert out["warm_state_mo_occ"] == "occ"
    assert out["orbsym"] == (3, 3)
    assert out["wfnsym"] == 2
    assert kwargs == {}


def test_resolve_kernel_solver_controls_defaults_and_alias():
    kwargs = {"pspace": 9}
    defaults = SimpleNamespace(conv_tol=1e-8, lindep=2e-13, max_cycle=44, max_space=13, max_memory=1234.0, pspace_size=7)
    out = resolve_kernel_solver_controls(kwargs=kwargs, defaults=defaults)
    assert out == {
        "tol": 1e-8,
        "lindep": 2e-13,
        "max_cycle": 44,
        "max_space": 13,
        "max_memory": 1234.0,
        "pspace_size": 9,
    }
    assert kwargs == {}


def test_resolve_kernel_solver_controls_explicit_pspace_size_takes_precedence():
    kwargs = {
        "tol": "1e-9",
        "lindep": "1e-12",
        "max_cycle": "20",
        "max_space": "11",
        "max_memory": "2048.0",
        "pspace_size": 5,
        "pspace": 99,
    }
    out = resolve_kernel_solver_controls(kwargs=kwargs, defaults=object())
    assert out["tol"] == pytest.approx(1e-9)
    assert out["lindep"] == pytest.approx(1e-12)
    assert out["max_cycle"] == 20
    assert out["max_space"] == 11
    assert out["max_memory"] == pytest.approx(2048.0)
    assert out["pspace_size"] == 5
    # Keep existing nested-pop semantics from solver.py (pspace is consumed too).
    assert kwargs == {}


def test_resolve_kernel_runtime_controls_contract_and_noncontract():
    defaults = SimpleNamespace(
        ne_constraints={"x": (0, 1)},
        unconverged_fallback_full_diag=True,
        unconverged_fallback_ncsf_max=512,
        raise_on_unconverged=False,
        warn_on_unconverged=True,
        contract_nthreads=0,
        contract_blas_nthreads=None,
        kernel_blas_nthreads=None,
    )
    kwargs = {
        "precompute_epq": True,
        "contract_nthreads": 0,
        "contract_blas_nthreads": 0,
        "kernel_blas_nthreads": -1,
        "unconverged_fallback_ncsf_max": 0,
    }
    out = resolve_kernel_runtime_controls(
        kwargs=kwargs,
        defaults=defaults,
        matvec_backend="contract",
        auto_num_threads_fn=lambda: 13,
    )
    assert out["precompute_epq"] is True
    assert out["contract_nthreads"] == 13
    assert out["contract_blas_nthreads"] == 13
    assert out["kernel_blas_nthreads"] is None
    assert out["unconverged_fallback_ncsf_max"] == 1
    assert kwargs == {}

    out2 = resolve_kernel_runtime_controls(
        kwargs={"precompute_epq": True, "contract_nthreads": 2},
        defaults=defaults,
        matvec_backend="cuda",
        auto_num_threads_fn=lambda: 9,
    )
    assert out2["precompute_epq"] is False
    assert out2["contract_nthreads"] == 2
    assert out2["contract_blas_nthreads"] is None


def test_warm_state_runtime_roundtrip(tmp_path):
    ci = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    state = {
        "format_version": int(WARM_STATE_FORMAT_VERSION),
        "norb": 2,
        "nelec_total": 2,
        "twos": 0,
        "nroots": 2,
        "ncsf": 2,
        "wfnsym": None,
        "orbsym": (1, 1),
        "ne_constraints_key": None,
        "ci_dtype": "float64",
        "ci_device": "cpu",
        "ci": ci,
        "mo_coeff": np.eye(2, dtype=np.float64),
        "mo_occ": np.asarray([2.0, 0.0]),
        "cas_metadata": {"ncas": 2, "nelecas": (1, 1)},
    }
    out = tmp_path / "warm_state.npz"
    save_warm_state(out, state=state, include_ci=True, include_mo=True)
    loaded = load_warm_state(out, require_ci=True)
    summary = warm_state_summary(loaded)
    assert summary is not None
    assert summary["nroots"] == 2
    assert summary["has_ci"] is True
    assert loaded["ci"].shape == (2, 2)


def test_warm_state_runtime_ci0_compat_and_update():
    def _norm(ci, *, nroots: int, ncsf: int):
        arr = np.asarray(ci, dtype=np.float64)
        return [np.ascontiguousarray(arr[i], dtype=np.float64) for i in range(int(nroots))]

    state = update_warm_state(
        prev_state=None,
        normalize_ci0_fn=_norm,
        ci=np.asarray([[1.0, 0.0]], dtype=np.float64),
        norb=2,
        nelec_total=2,
        twos=0,
        nroots=1,
        ncsf=2,
        orbsym_key=(1, 1),
        wfnsym=None,
        ne_constraints_key=None,
        cas_metadata={"ncas": 2, "nelecas": (1, 1)},
        mo_coeff=None,
        mo_occ=None,
    )
    ci0, reason = warm_state_ci0_if_compatible(
        state=state,
        norb=2,
        nelec_total=2,
        twos=0,
        nroots=1,
        ncsf=2,
        orbsym_key=(1, 1),
        wfnsym=None,
        ne_constraints_key=None,
        matvec_backend="cuda",
        cas_metadata={"ncas": 2, "nelecas": (1, 1)},
    )
    assert reason == "applied"
    assert ci0 is not None and len(ci0) == 1
    assert allowed_ci_devices_for_backend("cuda") == ("cpu", "cuda")
