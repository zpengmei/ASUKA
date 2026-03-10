from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from asuka._solver.cuda_policy import (
    apply_cuda_user_policy,
    normalize_cuda_accuracy_mode,
    normalize_cuda_user_policy_mode,
    normalize_matvec_cuda_path_mode,
    normalize_ws_cache_fraction,
    resolve_epq_overbudget_action,
)
from asuka._solver.config import auto_num_threads
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
    release_matvec_cuda_workspace,
    resolve_approx_cuda_frontend,
    resolve_matvec_cuda_ws_cache_budget_bytes,
    resolve_kernel_cuda_execution_mode,
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
