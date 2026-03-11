from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from asuka.cuda import cuda_backend
from asuka.cuda._backend_caps import (
    device_info,
    has_build_w_from_epq_transpose_range_mm,
    has_build_w_from_epq_transpose_range_mm_scaled,
    has_cuda_ext,
    has_epq_table_device_build,
    has_epq_table_device_build_recompute,
    has_epq_table_gather_apply_device,
    has_t_from_epq_table_device_build,
    mem_info,
)
from asuka.cuda._backend_epq_config import (
    EPQ_I32_MAX_NNZ,
    epq_indptr_cp_dtype_for_total_nnz,
    normalize_epq_blocked_transpose_mode,
    normalize_epq_indptr_mode,
    normalize_matvec_cuda_path_mode,
    resolve_epq_blocked_transpose_mode_with_env,
    resolve_epq_blocked_transpose_reserve_mib_with_env,
)
from asuka.cuda._backend_workspace_config import (
    resolve_apply_warp_coop,
    resolve_check_overflow_mode,
    resolve_epq_stream_double_buffer_mode,
    resolve_epq_stream_panic_mode,
    resolve_epq_stream_use_recompute,
)
from asuka.cuda._backend_workspace_policy import (
    resolve_cache_csr_tiles,
    resolve_csr_host_cache_enabled,
    resolve_csr_host_cache_mode,
    resolve_csr_pipeline_enabled,
    resolve_csr_pipeline_streams,
    resolve_epq_apply_cache_budget_gib,
    resolve_epq_apply_cache_enabled,
    resolve_epq_apply_cache_mode,
    resolve_fp32_csr_cache,
    resolve_prefilter_trivial_tasks_enabled,
    resolve_prefilter_trivial_tasks_mode,
    resolve_skip_zero_x_tiles_enabled,
    resolve_skip_zero_x_tiles_mode,
)
from asuka.cuda._backend_epq_table import init_workspace_epq_table
from asuka.cuda._backend_workspace_alloc import (
    build_workspace_cache_state,
    make_occ_buf_dtype,
    resolve_workspace_nrows_block,
)
from asuka.cuda._backend_csr_pipeline import (
    grow_csr_pipeline_slot,
    init_csr_pipeline_slots,
)
from asuka.cuda._backend_offdiag_setup import (
    init_sym_pair_setup,
    resolve_w_offdiag_prefer_blocked,
)
from asuka.cuda._backend_offdiag_workspace import (
    autoset_offdiag_cublas_workspace_with_backoff,
    configure_offdiag_gemm_workspace,
)
from asuka.cuda._backend_workspace_utils import (
    csr_cache_ready,
    estimate_object_nbytes,
    release_tile_csr_scratch,
    release_workspace_resources,
    workspace_nbytes_estimate_from_dict,
)
from asuka.cuda._backend_cache_runtime import (
    csr_host_cache_load_tile,
    csr_host_cache_store_tile,
    csr_host_cache_try_admit,
    csr_host_entry_bytes,
    epq_apply_cache_load,
    epq_apply_cache_store,
    epq_apply_ensure_staging,
)
from asuka.cuda._backend_hop_runtime import (
    build_epq_stream_tile,
    build_epq_stream_tile_profiled,
    finalize_one_body_phase,
    finalize_hop_profile,
    increment_profile_counter,
    normalize_hop_x,
    prepare_hop_runtime_inputs,
    resolve_epq_table_for_tile,
    resolve_epq_streaming_runtime,
    resolve_hop_runtime_flags,
    resolve_one_body_mode,
    resolve_tile_runtime_flags,
    resolve_x_tile_skip_policy,
    run_one_body_epq_streaming,
    run_one_body_single_scatter,
    stamp_csr_prefilter_profile,
    stamp_csr_prefilter_tile_profile,
    set_matvec_path_profile,
    try_cuda_graph_fast_path,
)


def test_backend_caps_helpers_with_stub_extension():
    assert has_cuda_ext(None) is False
    ext = SimpleNamespace(
        epq_contribs_many_count_allpairs_inplace_device=True,
        epq_contribs_many_count_allpairs_recompute_inplace_device=True,
        build_t_from_epq_table_inplace_device=True,
        apply_g_flat_gather_epq_table_inplace_device=True,
        build_w_from_epq_transpose_range_mm_scaled_inplace_device=True,
        build_w_from_epq_transpose_range_mm_inplace_device=True,
        device_info=lambda: {"name": "stub"},
        mem_info=lambda: {"free": 1, "total": 2},
    )
    assert has_cuda_ext(ext) is True
    assert has_epq_table_device_build(ext) is True
    assert has_epq_table_device_build_recompute(ext) is True
    assert has_t_from_epq_table_device_build(ext) is True
    assert has_epq_table_gather_apply_device(ext) is True
    assert has_build_w_from_epq_transpose_range_mm_scaled(ext) is True
    assert has_build_w_from_epq_transpose_range_mm(ext) is True
    assert device_info(ext) == {"name": "stub"}
    assert mem_info(ext) == {"free": 1, "total": 2}

    with pytest.raises(RuntimeError, match="CUDA extension not available"):
        device_info(None)
    with pytest.raises(RuntimeError, match="CUDA extension not available"):
        mem_info(None)


def test_cuda_backend_public_caps_wrappers(monkeypatch):
    ext = SimpleNamespace(
        epq_contribs_many_count_allpairs_inplace_device=True,
        epq_contribs_many_count_allpairs_recompute_inplace_device=True,
        build_t_from_epq_table_inplace_device=True,
        apply_g_flat_gather_epq_table_inplace_device=True,
        build_w_from_epq_transpose_range_mm_scaled_inplace_device=True,
        build_w_from_epq_transpose_range_mm_inplace_device=True,
        device_info=lambda: {"device": "stub"},
        mem_info=lambda: {"free": 3, "total": 4},
    )
    monkeypatch.setattr(cuda_backend, "_ext", ext, raising=False)

    assert cuda_backend.has_cuda_ext() is True
    assert cuda_backend.has_epq_table_device_build() is True
    assert cuda_backend.has_epq_table_device_build_recompute() is True
    assert cuda_backend.has_t_from_epq_table_device_build() is True
    assert cuda_backend.has_epq_table_gather_apply_device() is True
    assert cuda_backend.has_build_w_from_epq_transpose_range_mm_scaled() is True
    assert cuda_backend.has_build_w_from_epq_transpose_range_mm() is True
    assert cuda_backend.device_info() == {"device": "stub"}
    assert cuda_backend.mem_info() == {"free": 3, "total": 4}


def test_backend_epq_config_helpers():
    assert normalize_epq_indptr_mode(None) == "auto"
    assert normalize_epq_indptr_mode("i4") == "int32"
    assert normalize_epq_indptr_mode(np.int64) == "int64"
    with pytest.raises(ValueError, match="indptr_dtype must be one of"):
        normalize_epq_indptr_mode(np.float64)

    assert normalize_epq_blocked_transpose_mode(True) == "on"
    assert normalize_epq_blocked_transpose_mode("disable") == "off"
    assert resolve_epq_blocked_transpose_mode_with_env(None, "on") == "on"
    assert resolve_epq_blocked_transpose_reserve_mib_with_env(None, "768") == 768
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_epq_blocked_transpose_reserve_mib_with_env(None, "bad")

    assert normalize_matvec_cuda_path_mode("epq") == "epq_blocked"
    assert normalize_matvec_cuda_path_mode("fused_epq") == "fused_epq_hybrid"
    with pytest.raises(ValueError, match="disabled"):
        normalize_matvec_cuda_path_mode("fused_coo")


def test_backend_epq_config_indptr_cp_dtype_selector():
    cp = SimpleNamespace(int32="i4", int64="i8")
    assert epq_indptr_cp_dtype_for_total_nnz(cp, mode="auto", total_nnz=0) == "i4"
    assert epq_indptr_cp_dtype_for_total_nnz(cp, mode="int64", total_nnz=1) == "i8"
    assert epq_indptr_cp_dtype_for_total_nnz(cp, mode="auto", total_nnz=EPQ_I32_MAX_NNZ + 1) == "i8"
    with pytest.raises(ValueError, match="total_nnz must be >= 0"):
        epq_indptr_cp_dtype_for_total_nnz(cp, mode="auto", total_nnz=-1)
    with pytest.raises(ValueError, match="requires total_nnz <="):
        epq_indptr_cp_dtype_for_total_nnz(cp, mode="int32", total_nnz=EPQ_I32_MAX_NNZ + 1)


def test_backend_workspace_config_mode_resolvers():
    assert resolve_epq_stream_double_buffer_mode(None, "") == "auto"
    assert resolve_epq_stream_double_buffer_mode(None, "on") == "on"
    assert resolve_epq_stream_double_buffer_mode(False, "on") == "off"
    with pytest.raises(ValueError, match="EPQ_STREAM_DOUBLE_BUFFER"):
        resolve_epq_stream_double_buffer_mode(None, "bad")

    assert resolve_epq_stream_panic_mode(None, "off") == "off"
    assert resolve_epq_stream_panic_mode("auto", "on") == "auto"
    with pytest.raises(ValueError, match="epq_stream_panic_mode must be bool"):
        resolve_epq_stream_panic_mode("bad", "")

    assert resolve_epq_stream_use_recompute(None, "") == "auto"
    assert resolve_epq_stream_use_recompute(None, "1") is True
    assert resolve_epq_stream_use_recompute(False, "1") is False
    with pytest.raises(ValueError, match="EPQ_STREAM_RECOMPUTE"):
        resolve_epq_stream_use_recompute(None, "bad")
    with pytest.raises(ValueError, match="must be bool or 'auto'"):
        resolve_epq_stream_use_recompute("on", "")

    assert resolve_apply_warp_coop(None, "") == "auto"
    assert resolve_apply_warp_coop(None, "0") is False
    assert resolve_apply_warp_coop(True, "") is True
    with pytest.raises(ValueError, match="APPLY_WARP_COOP"):
        resolve_apply_warp_coop(None, "bad")
    with pytest.raises(ValueError, match="must be bool or 'auto'"):
        resolve_apply_warp_coop("on", "")

    assert resolve_check_overflow_mode("none") == 0
    assert resolve_check_overflow_mode("deferred") == 1
    assert resolve_check_overflow_mode("per-stage") == 2
    assert resolve_check_overflow_mode(2) == 2
    with pytest.raises(ValueError, match="must be one of: none, deferred, per-stage"):
        resolve_check_overflow_mode("bad")
    with pytest.raises(ValueError, match="must be 0 \\(none\\), 1 \\(deferred\\), or 2"):
        resolve_check_overflow_mode(7)


def test_backend_workspace_policy_helpers():
    assert resolve_cache_csr_tiles("auto", j_tile=128, ncsf=1000, norb=12, csr_capacity_mult=2.0) is True
    assert resolve_cache_csr_tiles("auto", j_tile=1000, ncsf=1000, norb=12, csr_capacity_mult=2.0) is False
    assert resolve_cache_csr_tiles("off", j_tile=128, ncsf=10000, norb=12, csr_capacity_mult=2.0) is True

    assert resolve_fp32_csr_cache("auto", dtype_is_float32=True) is True
    assert resolve_fp32_csr_cache("auto", dtype_is_float32=False) is False

    assert resolve_csr_host_cache_mode("auto") == "auto"
    assert resolve_csr_host_cache_mode(True) == "on"
    with pytest.raises(ValueError, match="csr_host_cache must be bool or one of"):
        resolve_csr_host_cache_mode("bad")

    assert resolve_csr_host_cache_enabled(
        mode="auto",
        j_tile=128,
        ncsf=100000,
        budget_bytes=1,
        min_ncsf=1000,
        cache_csr_tiles=False,
        aggregate_offdiag_k=False,
        use_epq_table=False,
    ) is True

    assert resolve_epq_apply_cache_budget_gib(None, "8.0") == 8.0
    with pytest.raises(ValueError, match="could not convert string to float"):
        resolve_epq_apply_cache_budget_gib("bad", "")
    with pytest.raises(ValueError, match="BUDGET_GIB must be a float"):
        resolve_epq_apply_cache_budget_gib(None, "bad")

    assert resolve_epq_apply_cache_mode(None, "on") == "on"
    assert resolve_epq_apply_cache_mode(False, "") == "off"
    with pytest.raises(ValueError, match="epq_apply_cache must be bool or one of"):
        resolve_epq_apply_cache_mode("bad", "")

    assert resolve_epq_apply_cache_enabled(
        mode="on",
        aggregate_offdiag_k=True,
        use_epq_table=True,
        budget_bytes=1,
        has_epq_table_device_build=True,
        csr_host_cache_mode="off",
    ) is True
    assert resolve_epq_apply_cache_enabled(
        mode="auto",
        aggregate_offdiag_k=True,
        use_epq_table=False,
        budget_bytes=1,
        has_epq_table_device_build=True,
        csr_host_cache_mode="on",
    ) is False

    assert resolve_csr_pipeline_streams("auto") == ("auto", 2)
    assert resolve_csr_pipeline_streams("off") == ("off", 0)
    assert resolve_csr_pipeline_streams("3") == ("manual", 3)
    assert resolve_csr_pipeline_enabled(
        streams_mode="auto",
        streams=2,
        j_tile=128,
        ncsf=100000,
        min_ncsf=1000,
        cache_csr_tiles=False,
        csr_host_cache_enabled=False,
        aggregate_offdiag_k=False,
        use_epq_table=False,
    ) is True
    assert resolve_csr_pipeline_enabled(
        streams_mode="auto",
        streams=0,
        j_tile=128,
        ncsf=100000,
        min_ncsf=1000,
        cache_csr_tiles=False,
        csr_host_cache_enabled=False,
        aggregate_offdiag_k=False,
        use_epq_table=False,
    ) is False

    assert resolve_prefilter_trivial_tasks_mode("auto") == "auto"
    assert resolve_prefilter_trivial_tasks_mode(True) == "on"
    with pytest.raises(ValueError, match="prefilter_trivial_tasks must be bool or one of"):
        resolve_prefilter_trivial_tasks_mode("bad")
    assert resolve_prefilter_trivial_tasks_enabled(
        mode="auto",
        j_tile=128,
        ncsf=100000,
        min_ncsf=1000,
        use_epq_table=False,
        aggregate_offdiag_k=False,
    ) is True
    assert resolve_prefilter_trivial_tasks_enabled(
        mode="auto",
        j_tile=128,
        ncsf=100000,
        min_ncsf=1000,
        use_epq_table=False,
        aggregate_offdiag_k=True,
    ) is False

    assert resolve_skip_zero_x_tiles_mode("on") == "on"
    with pytest.raises(ValueError, match="skip_zero_x_tiles must be bool or one of"):
        resolve_skip_zero_x_tiles_mode("bad")
    assert resolve_skip_zero_x_tiles_enabled(
        mode="auto",
        j_tile=128,
        ncsf=100000,
        min_ncsf=1000,
    ) is True


class _FakeStream:
    def __init__(self, counter):
        self._counter = counter

    def synchronize(self):
        self._counter["sync"] += 1


class _FakeCP:
    int32 = np.int32
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64

    def __init__(self):
        self.counter = {"sync": 0}
        self.cuda = SimpleNamespace(
            get_current_stream=lambda: _FakeStream(self.counter),
            Stream=lambda non_blocking=True: SimpleNamespace(non_blocking=bool(non_blocking)),
        )

    @staticmethod
    def asarray(arr, dtype=None):
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=None):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def ascontiguousarray(arr):
        return np.ascontiguousarray(arr)

    @staticmethod
    def dtype(dt):
        return np.dtype(dt)

    @staticmethod
    def asnumpy(arr):
        return np.asarray(arr)


def test_backend_epq_table_helper_fallback_and_normalize():
    cp = _FakeCP()
    host_calls = {"nt": None}

    def _host_build(_drt, precompute_nthreads):
        host_calls["nt"] = int(precompute_nthreads)
        return (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([1.0, 2.0], dtype=np.float64),
        )

    out = init_workspace_epq_table(
        cp=cp,
        drt=SimpleNamespace(),
        drt_dev=object(),
        state_dev=object(),
        use_epq_table=True,
        epq_streaming=False,
        epq_build_device=True,
        epq_build_j_tile=0,
        j_tile=32,
        norb=8,
        ncsf=2,
        threads_enum=64,
        epq_recompute_warp_coop=False,
        dtype=np.float64,
        epq_indptr_dtype="int32",
        epq_build_nthreads=3,
        ext=SimpleNamespace(),  # missing device builder symbol -> host fallback
        build_device_tiled_fn=lambda *_a, **_k: None,
        build_device_fn=lambda *_a, **_k: None,
        build_host_fn=_host_build,
        indptr_dtype_resolver_fn=lambda cp_mod, **_kw: cp_mod.int32,
        as_indptr_array_fn=lambda _cp, arr, **_kw: np.asarray(arr),
        as_pq_array_fn=lambda _cp, arr, **_kw: np.asarray(arr),
        epq_i32_max_nnz=np.iinfo(np.int32).max,
    )
    assert out["epq_build_device"] is False
    assert host_calls["nt"] == 3
    assert out["epq_table"] is not None
    assert np.dtype(out["epq_table"][0].dtype) == np.dtype(np.int32)
    assert cp.counter["sync"] == 1
    assert out["epq_table_build_s"] >= 0.0


def test_backend_epq_table_helper_cast_guard():
    cp = _FakeCP()

    with pytest.raises(RuntimeError, match="Cannot cast EPQ indptr to int32"):
        init_workspace_epq_table(
            cp=cp,
            drt=SimpleNamespace(),
            drt_dev=object(),
            state_dev=object(),
            use_epq_table=True,
            epq_streaming=False,
            epq_build_device=False,
            epq_build_j_tile=0,
            j_tile=32,
            norb=8,
            ncsf=2,
            threads_enum=64,
            epq_recompute_warp_coop=False,
            dtype=np.float64,
            epq_indptr_dtype="int32",
            epq_build_nthreads=1,
            ext=None,
            build_device_tiled_fn=lambda *_a, **_k: None,
            build_device_fn=lambda *_a, **_k: None,
            build_host_fn=lambda _drt, precompute_nthreads: (
                np.asarray([0, np.iinfo(np.int32).max + 1], dtype=np.int64),
                np.asarray([0], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([1.0], dtype=np.float64),
            ),
            indptr_dtype_resolver_fn=lambda cp_mod, **_kw: cp_mod.int64,
            as_indptr_array_fn=lambda _cp, arr, **_kw: np.asarray(arr),
            as_pq_array_fn=lambda _cp, arr, **_kw: np.asarray(arr),
            epq_i32_max_nnz=np.iinfo(np.int32).max,
        )


def test_backend_workspace_alloc_helpers():
    cp = _FakeCP()
    st = build_workspace_cache_state(
        cp=cp,
        cache_csr_tiles=True,
        j_tile=128,
        ncsf=1000,
        csr_capacity_mult=2.0,
        rs_n_pairs=132,
        norb=12,
        csr_data_dtype=np.float32,
    )
    assert isinstance(st["_csr_tile_cache"], dict)
    assert isinstance(st["_epq_apply_tile_cache"], dict)
    assert st["_tile_csr_capacity"] > 0
    assert tuple(st["_tile_csr_row_j"].shape) == (st["_tile_csr_capacity"],)
    assert tuple(st["_tile_csr_indptr"].shape) == (st["_tile_csr_capacity"] + 1,)

    st2 = build_workspace_cache_state(
        cp=cp,
        cache_csr_tiles=False,
        j_tile=128,
        ncsf=1000,
        csr_capacity_mult=2.0,
        rs_n_pairs=None,
        norb=12,
        csr_data_dtype=np.float64,
    )
    assert st2["_tile_csr_capacity"] == 0
    assert st2["_tile_csr_row_j"] is None

    assert make_occ_buf_dtype(cp=cp, dtype=np.float64, j_tile=32, norb=8) is None
    occ = make_occ_buf_dtype(cp=cp, dtype=np.float32, j_tile=32, norb=8)
    assert tuple(occ.shape) == (32, 8)
    assert np.dtype(occ.dtype) == np.dtype(np.float32)

    n1 = resolve_workspace_nrows_block(
        max_g_bytes=1024,
        nops=16,
        itemsize=8,
        ncsf=5,
        eri_mat_present=True,
        l_full_present=False,
        naux=0,
    )
    assert n1 == 5
    n2 = resolve_workspace_nrows_block(
        max_g_bytes=1024,
        nops=16,
        itemsize=8,
        ncsf=100,
        eri_mat_present=False,
        l_full_present=True,
        naux=10,
    )
    assert n2 == 4


def test_backend_csr_pipeline_helpers_init_and_grow():
    class _FakeK25Ws:
        def __init__(self, max_tasks, cap):
            self.max_tasks = int(max_tasks)
            self.cap = int(cap)
            self.released = 0

        def release(self):
            self.released += 1

    class _FakeExt:
        Kernel25Workspace = _FakeK25Ws

    cp = _FakeCP()

    slots0, apply0 = init_csr_pipeline_slots(
        cp=cp,
        ext=_FakeExt(),
        enabled=False,
        n_slots=3,
        initial_capacity=64,
        j_tile=16,
        rs_n_pairs=20,
        csr_data_dtype=np.float64,
    )
    assert slots0 == []
    assert apply0 is None

    slots1, apply1 = init_csr_pipeline_slots(
        cp=cp,
        ext=_FakeExt(),
        enabled=True,
        n_slots=1,
        initial_capacity=64,
        j_tile=16,
        rs_n_pairs=20,
        csr_data_dtype=np.float64,
    )
    assert slots1 == []
    assert apply1 is None

    slots, apply_stream = init_csr_pipeline_slots(
        cp=cp,
        ext=_FakeExt(),
        enabled=True,
        n_slots=2,
        initial_capacity=64,
        j_tile=16,
        rs_n_pairs=20,
        csr_data_dtype=np.float64,
    )
    assert len(slots) == 2
    assert apply_stream is not None
    assert tuple(slots[0]["row_j"].shape) == (64,)
    assert int(slots[0]["ws"].max_tasks) == 320
    assert int(slots[0]["ws"].cap) == 64

    old_ws = slots[0]["ws"]
    grow_csr_pipeline_slot(
        cp=cp,
        ext=_FakeExt(),
        slots=slots,
        slot_idx=0,
        new_cap=128,
        j_tile=16,
        rs_n_pairs=20,
        csr_data_dtype=np.float64,
    )
    assert int(old_ws.released) == 1
    assert int(slots[0]["cap"]) == 128
    assert tuple(slots[0]["indices"].shape) == (128,)

    # no-op grow should preserve current slot capacity
    grow_csr_pipeline_slot(
        cp=cp,
        ext=_FakeExt(),
        slots=slots,
        slot_idx=0,
        new_cap=127,
        j_tile=16,
        rs_n_pairs=20,
        csr_data_dtype=np.float64,
    )
    assert int(slots[0]["cap"]) == 128

    with pytest.raises(IndexError, match="slot index out of range"):
        grow_csr_pipeline_slot(
            cp=cp,
            ext=_FakeExt(),
            slots=slots,
            slot_idx=3,
            new_cap=256,
            j_tile=16,
            rs_n_pairs=20,
            csr_data_dtype=np.float64,
        )


def test_backend_offdiag_setup_helpers(monkeypatch):
    assert resolve_w_offdiag_prefer_blocked(
        dtype_is_float32=False,
        ncsf=2_000_000,
        epq_table_present=True,
        use_epq_table=True,
        eri_mat_present=True,
    ) is False
    assert resolve_w_offdiag_prefer_blocked(
        dtype_is_float32=True,
        ncsf=2_000_000,
        epq_table_present=False,
        use_epq_table=False,
        eri_mat_present=True,
    ) is True
    assert resolve_w_offdiag_prefer_blocked(
        dtype_is_float32=True,
        ncsf=2_000_000,
        epq_table_present=True,
        use_epq_table=True,
        eri_mat_present=False,
    ) is True

    cp = _FakeCP()
    # Disabled path returns defaults.
    out0 = init_sym_pair_setup(
        cp=cp,
        aggregate_offdiag_k=False,
        has_sym_pair_pack_device=True,
        norb=12,
        nops=144,
        dtype=np.float64,
        kernel3_workspace_ctor=lambda *_a, **_k: object(),
    )
    assert out0["_sym_pair_npair"] == 0
    assert out0["_sym_pair_pair_pq"] is None

    # Enabled path builds maps and gemm workspace with fallback backend.
    monkeypatch.setenv("CUGUGA_SYM_PAIR_MIN_NORB", "10")
    monkeypatch.setenv("CUGUGA_SYM_PAIR_GEMM_BACKEND", "preferred_backend")
    calls = []

    def _ctor(npair, *, max_nrows, dtype, gemm_backend):
        calls.append((npair, max_nrows, np.dtype(dtype), gemm_backend))
        if gemm_backend == "preferred_backend":
            raise RuntimeError("fail preferred")
        return {"npair": npair, "backend": gemm_backend}

    out = init_sym_pair_setup(
        cp=cp,
        aggregate_offdiag_k=True,
        has_sym_pair_pack_device=True,
        norb=12,
        nops=144,
        dtype=np.float64,
        kernel3_workspace_ctor=_ctor,
    )
    assert out["_sym_pair_npair"] == 78
    assert tuple(out["_sym_pair_pair_pq"].shape) == (78,)
    assert tuple(out["_sym_pair_pair_qp"].shape) == (78,)
    assert tuple(out["_sym_pair_full_to_pair"].shape) == (144,)
    assert out["_sym_pair_gemm_ws"] == {"npair": 78, "backend": "gemmex_fp64"}
    assert calls[0][3] == "preferred_backend"
    assert calls[1][3] == "gemmex_fp64"


class _FakeOffdiagWs:
    def __init__(self, *, fail_above_bytes: int | None = None, fail_text: str = "out of memory"):
        self.gemm_backend = None
        self.math_mode = None
        self.strategy = None
        self._ws_bytes = 0
        self._fail_above_bytes = fail_above_bytes
        self._fail_text = fail_text
        self.calls = []

    def set_gemm_backend(self, backend):
        self.gemm_backend = str(backend)
        self.calls.append(("set_gemm_backend", str(backend)))

    def set_cublas_math_mode(self, mode):
        self.math_mode = str(mode)
        self.calls.append(("set_cublas_math_mode", str(mode)))

    def set_cublas_emulation_strategy(self, strategy):
        self.strategy = str(strategy)
        self.calls.append(("set_cublas_emulation_strategy", str(strategy)))

    def cublas_emulation_info(self):
        return {"backend": self.gemm_backend or "unset"}

    def set_cublas_workspace_bytes(self, bytes_):
        b = int(bytes_)
        self.calls.append(("set_cublas_workspace_bytes", b))
        if self._fail_above_bytes is not None and b > int(self._fail_above_bytes):
            raise RuntimeError(self._fail_text)
        self._ws_bytes = b

    def cublas_workspace_bytes(self):
        return int(self._ws_bytes)


def test_backend_offdiag_workspace_config_and_backoff(monkeypatch):
    ws = _FakeOffdiagWs()
    configure_offdiag_gemm_workspace(
        ws=ws,
        gemm_backend="gemmex_fp64",
        offdiag_enable_fp64_emulation=False,
        offdiag_emulation_strategy="performant",
    )
    assert ws.gemm_backend == "gemmex_fp64"
    assert ws.math_mode == "default"

    ws2 = _FakeOffdiagWs()
    configure_offdiag_gemm_workspace(
        ws=ws2,
        gemm_backend="gemmex_fp64",
        offdiag_enable_fp64_emulation=True,
        offdiag_emulation_strategy="performant",
    )
    assert ws2.gemm_backend == "gemmex_emulated_fixedpoint"
    assert ws2.math_mode == "fp64_emulated_fixedpoint"
    assert ws2.strategy == "performant"

    monkeypatch.delenv("CUGUGA_ALLOW_CUBLAS_EAGER", raising=False)
    with pytest.raises(RuntimeError, match="CUGUGA_ALLOW_CUBLAS_EAGER=1"):
        configure_offdiag_gemm_workspace(
            ws=_FakeOffdiagWs(),
            gemm_backend="gemmex_fp64",
            offdiag_enable_fp64_emulation=True,
            offdiag_emulation_strategy="eager",
        )

    ws3 = _FakeOffdiagWs(fail_above_bytes=64 * 1024 * 1024)
    with pytest.warns(RuntimeWarning, match="workspace reduced"):
        out = autoset_offdiag_cublas_workspace_with_backoff(
            ws=ws3,
            nops=10,
            nrows_eff=20,
            cap_mb=0,
            recommend_fn=lambda **_kw: 256 * 1024 * 1024,
        )
    assert out["requested_ws"] == 256 * 1024 * 1024
    assert out["applied_ws"] == 64 * 1024 * 1024
    assert out["workspace_bytes"] == 64 * 1024 * 1024

    ws4 = _FakeOffdiagWs(fail_above_bytes=1, fail_text="driver lost")
    with pytest.raises(RuntimeError, match="driver lost"):
        autoset_offdiag_cublas_workspace_with_backoff(
            ws=ws4,
            nops=10,
            nrows_eff=20,
            cap_mb=0,
            recommend_fn=lambda **_kw: 2,
        )


def test_backend_workspace_utils_cache_ready_and_tile_release():
    assert csr_cache_ready(
        j_tile=10,
        ncsf=10,
        cache_csr_tiles=False,
        csr_single_tile_cache=object(),
        csr_tile_cache={},
    ) is True
    assert csr_cache_ready(
        j_tile=4,
        ncsf=10,
        cache_csr_tiles=False,
        csr_single_tile_cache=None,
        csr_tile_cache={0: object()},
    ) is False
    assert csr_cache_ready(
        j_tile=4,
        ncsf=10,
        cache_csr_tiles=True,
        csr_single_tile_cache=None,
        csr_tile_cache={0: object(), 4: object(), 8: object()},
    ) is True

    ws = SimpleNamespace(
        _tile_csr_row_j=object(),
        _tile_csr_row_k=object(),
        _tile_csr_indptr=object(),
        _tile_csr_indices=object(),
        _tile_csr_data=object(),
        _tile_csr_overflow=object(),
        _tile_csr_capacity=123,
    )
    release_tile_csr_scratch(ws)
    assert ws._tile_csr_row_j is None
    assert ws._tile_csr_row_k is None
    assert ws._tile_csr_indptr is None
    assert ws._tile_csr_indices is None
    assert ws._tile_csr_data is None
    assert ws._tile_csr_overflow is None
    assert ws._tile_csr_capacity == 0


def test_backend_workspace_utils_nbytes_and_release():
    shared = np.empty((8,), dtype=np.float64)
    seen: set[int] = set()
    assert estimate_object_nbytes(shared, seen) == int(shared.nbytes)
    assert estimate_object_nbytes(shared, seen) == 0  # dedup

    state = {
        "a": shared,
        "b": {"x": shared, "y": [np.empty((4,), dtype=np.float64)]},
        "__private": np.empty((2,), dtype=np.float64),
    }
    est = workspace_nbytes_estimate_from_dict(state)
    assert est == int(shared.nbytes + 4 * np.dtype(np.float64).itemsize)

    class _Rel:
        def __init__(self):
            self.calls = 0

        def release(self):
            self.calls += 1

    slot_ws = _Rel()
    w1 = _Rel()
    w2 = _Rel()
    ws = SimpleNamespace(
        _released=False,
        workspace_nbytes_estimate=lambda: 77,
        _cuda_graph=object(),
        _cuda_graph_x=object(),
        _cuda_graph_y=object(),
        _csr_pipeline_slots=[{"ws": slot_ws}],
        _csr_pipeline_apply_stream=object(),
        _k25_ws=w1,
        _offdiag_gemm_ws=w2,
        _gdf_ws=None,
        _g_buf=object(),
        _task_scale_rows=object(),
        _diag_g_cache={},
        _g_diag_buf=object(),
        _diag_w_buf=object(),
        _occ_buf=object(),
        _occ_buf_dtype=object(),
        _w_offdiag=object(),
        _w_block=object(),
        _l_full_t=object(),
        _offdiag_df_t=object(),
        _eri_diag_t=object(),
        _eri_mat_t=object(),
        _eri_mat_t_cache={},
        _task_scale_j=object(),
        _overflow_w=object(),
        overflow_apply=object(),
        _csr_row_j=object(),
        _csr_row_k=object(),
        _csr_indptr=object(),
        _csr_indices=object(),
        _csr_data=object(),
        _csr_overflow=object(),
        _csr_single_tile_cache=object(),
        _csr_tile_cache={},
        _csr_host_tile_cache={},
        _tile_csr_row_j=object(),
        _tile_csr_row_k=object(),
        _tile_csr_indptr=object(),
        _tile_csr_indices=object(),
        _tile_csr_data=object(),
        _tile_csr_overflow=object(),
        _epq_table=object(),
        _epq_apply_tile_cache={},
        _epq_apply_staging_indptr=object(),
        _epq_apply_staging_indices=object(),
        _epq_apply_staging_pq_ids=object(),
        _epq_apply_staging_data=object(),
        task_csf_all=object(),
        eri_mat=object(),
        l_full=object(),
        h_eff_flat=object(),
        _rs_r_d=object(),
        _rs_s_d=object(),
        _csr_host_cache_bytes=5,
        _epq_apply_cache_bytes=6,
        _epq_apply_staging_capacity=7,
        _tile_csr_capacity=8,
    )

    freed = release_workspace_resources(ws)
    assert freed == 77
    assert ws._released is True
    assert ws._csr_pipeline_slots == []
    assert ws._k25_ws is None and ws._offdiag_gemm_ws is None
    assert ws._g_buf is None and ws.eri_mat is None
    assert ws._csr_host_cache_bytes == 0
    assert slot_ws.calls == 1
    assert w1.calls == 1 and w2.calls == 1

    # idempotent
    assert release_workspace_resources(ws) == 0


def test_backend_cache_runtime_entry_bytes_and_admit_policy():
    tile_bytes = csr_host_entry_bytes(nrows=10, nnz=20, data_itemsize=8)
    assert tile_bytes == (10 * 4 + 10 * 4 + 11 * 8 + 20 * 4 + 20 * 8)

    ws = SimpleNamespace(
        csr_host_cache_budget_bytes=100,
        _csr_host_cache_bytes=0,
        _csr_host_tile_cache={},
        _csr_host_cache_evictions=0,
    )
    assert csr_host_cache_try_admit(ws, tile_bytes=64, score=5.0) is True
    assert csr_host_cache_try_admit(ws, tile_bytes=101, score=5.0) is False

    ws2 = SimpleNamespace(
        csr_host_cache_budget_bytes=100,
        _csr_host_cache_bytes=90,
        _csr_host_tile_cache={
            0: {"bytes": 40, "score": 1.0},
            64: {"bytes": 30, "score": 2.0},
        },
        _csr_host_cache_evictions=0,
    )
    assert csr_host_cache_try_admit(ws2, tile_bytes=50, score=3.0) is True
    assert ws2._csr_host_cache_evictions >= 1


def test_backend_cache_runtime_miss_and_early_return_paths():
    ws = SimpleNamespace(
        csr_host_cache_enabled=False,
        _csr_host_cache_store_attempts=0,
        _csr_host_tile_cache={},
        _csr_host_cache_misses=0,
        _epq_apply_cache_enabled=False,
        _epq_apply_tile_cache={},
        _epq_apply_cache_misses=0,
        _epq_apply_staging_capacity=0,
        _epq_apply_staging_indptr=None,
        _epq_apply_staging_indices=None,
        _epq_apply_staging_pq_ids=None,
        _epq_apply_staging_data=None,
        _dtype=np.float64,
        norb=6,
    )

    # Disabled store path returns before importing CuPy.
    csr_host_cache_store_tile(
        ws,
        j0=0,
        row_j_d=None,
        row_k_d=None,
        indptr_d=None,
        indices_d=None,
        data_d=None,
        nrows=4,
        nnz=8,
        stream=None,
        profile=None,
    )
    assert ws._csr_host_cache_store_attempts == 0

    # Miss path does not require CuPy and increments miss counters.
    out = csr_host_cache_load_tile(ws, j0=123, stream=None, profile={})
    assert out is None
    assert ws._csr_host_cache_misses == 1

    epq_apply_cache_store(
        ws,
        k0=0,
        indptr_d=None,
        indices_d=None,
        pq_ids_d=None,
        data_d=None,
        j_count=4,
        nnz=8,
        stream=None,
    )
    assert ws._epq_apply_tile_cache == {}

    out2 = epq_apply_cache_load(
        ws,
        k0=5,
        stream=None,
        epq_pq_dtype_for_norb_fn=lambda *_a, **_k: np.int32,
    )
    assert out2 is None
    assert ws._epq_apply_cache_misses == 1

    # Ensure-staging is indirectly covered in CUDA-backed tests; keep API import sanity here.
    assert callable(epq_apply_ensure_staging)


def test_backend_hop_runtime_flag_resolution():
    ws = SimpleNamespace(
        use_fused_hop=True,
        _fused_hop_kernel_available=True,
        norb=14,
        eri_mat=np.eye(4),
        l_full=None,
        aggregate_offdiag_k=True,
    )
    out = resolve_hop_runtime_flags(ws, path_mode="fused_epq_hybrid", eri_mat=None)
    assert out["use_fused_hop"] is True
    assert out["use_aggregate_offdiag"] is False


def test_backend_hop_runtime_graph_fast_path_noop():
    cp_mod = SimpleNamespace(
        asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
        ascontiguousarray=lambda arr: np.ascontiguousarray(arr),
        copyto=np.copyto,
        cuda=SimpleNamespace(get_current_stream=lambda: object()),
    )
    ws = SimpleNamespace(
        use_cuda_graph=False,
        _cuda_graph=None,
        _cuda_graph_x=None,
        _cuda_graph_y=None,
        _dtype=np.float64,
        ncsf=4,
    )
    out = try_cuda_graph_fast_path(
        ws,
        cp=cp_mod,
        x=np.ones(4),
        y=None,
        eri_mat=None,
        h_eff=None,
        stream=None,
        sync=True,
        check_overflow=False,
        profile=None,
    )
    assert out["executed"] is False


def test_backend_hop_runtime_prepare_inputs_and_streaming_flags():
    cp_mod = SimpleNamespace(
        asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
        ascontiguousarray=lambda arr: np.ascontiguousarray(arr),
        empty=lambda shape, dtype=None: np.empty(shape, dtype=dtype),
        dot=np.dot,
    )
    ws = SimpleNamespace(
        _dtype=np.float64,
        ncsf=4,
        nops=4,
        eri_mat=None,
        l_full=np.ones((4, 2), dtype=np.float64),
        use_epq_table=True,
        epq_streaming=True,
        _epq_table=None,
        epq_stream_panic_mode="off",
        h_eff_flat=np.arange(4, dtype=np.float64),
        _as_h_eff_flat=lambda h: np.asarray(h, dtype=np.float64).ravel(),
    )
    out = prepare_hop_runtime_inputs(
        ws,
        cp=cp_mod,
        y=None,
        eri_mat=None,
        h_eff=None,
        use_fused_hop=True,
    )
    assert tuple(out["y"].shape) == (4,)
    assert out["eri_mat_use"] is not None
    assert out["use_df"] is True
    assert out["use_epq_streaming"] is True
    assert out["use_epq_streaming_tiles"] is True


def test_backend_hop_runtime_build_epq_stream_tile_forwards_args():
    calls = {}

    def _build(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return ("ok",)

    ws = SimpleNamespace(
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        threads_enum=128,
        epq_stream_use_recompute="auto",
        epq_recompute_warp_coop=False,
        epq_stream_pq_block=0,
        _dtype=np.float64,
    )
    out = build_epq_stream_tile(
        ws,
        build_epq_action_table_tile_device_fn=_build,
        stream="s",
        sync=True,
        check_overflow=False,
        j_start=10,
        j_count=5,
    )
    assert out == ("ok",)
    assert calls["kwargs"]["j_start"] == 10
    assert calls["kwargs"]["j_count"] == 5
    assert calls["kwargs"]["stream"] == "s"


def test_backend_hop_runtime_build_epq_stream_tile_profiled_updates_profile():
    calls = {}

    def _build(*args, **kwargs):
        calls["kwargs"] = kwargs
        return ("tile",)

    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    stream = _Stream()
    ws = SimpleNamespace(
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        threads_enum=64,
        epq_stream_use_recompute=True,
        epq_recompute_warp_coop=True,
        epq_stream_pq_block=0,
        _dtype=np.float64,
    )
    profile = {}
    out = build_epq_stream_tile_profiled(
        ws,
        build_epq_action_table_tile_device_fn=_build,
        stream=stream,
        sync=True,
        check_overflow=False,
        profile=profile,
        j_start=0,
        j_count=8,
    )
    assert out == ("tile",)
    assert calls["kwargs"]["j_start"] == 0
    assert calls["kwargs"]["j_count"] == 8
    assert stream.nsync == 1
    assert profile["epq_stream_build_s"] >= 0.0
    assert profile["tile_apply_build_s"] >= 0.0


def test_backend_hop_runtime_set_matvec_path_profile():
    profile = {}
    set_matvec_path_profile(
        profile=profile,
        path_mode="fused_epq_hybrid",
        effective_mode="fused_coo",
        fallback_reason="",
    )
    assert profile["matvec_path_mode"] == "fused_epq_hybrid"
    assert profile["matvec_path_mode_effective"] == "fused_coo"
    assert profile["matvec_path_fallback_reason"] == ""


def test_backend_hop_runtime_resolve_epq_streaming_runtime_records_profile():
    ws = SimpleNamespace(
        ncsf=128,
        epq_stream_j_tile=32,
        epq_stream_double_buffer_mode="on",
        epq_stream_pq_block=16,
    )
    profile = {}
    out = resolve_epq_streaming_runtime(
        ws,
        use_epq_streaming=True,
        check_overflow=False,
        profile=profile,
        epq_stream_panic_requested=False,
        epq_stream_panic_active=False,
    )
    assert out["epq_stream_j_tile"] == 32
    assert out["epq_stream_db_requested"] is True
    assert out["epq_stream_db_active"] is False  # profiling lane disables overlap
    assert profile["epq_streaming"] == 1.0
    assert profile["epq_stream_j_tile"] == 32.0
    assert profile["epq_stream_pq_block"] == 16.0
    assert profile["epq_stream_double_buffer_requested"] == 1.0


def test_backend_hop_runtime_resolve_one_body_mode():
    assert resolve_one_body_mode(
        use_fused_hop=True,
        use_epq_streaming=True,
        epq_stream_panic_active=False,
        epq_stream_db_active=False,
    ) == "fused_zero"
    assert resolve_one_body_mode(
        use_fused_hop=False,
        use_epq_streaming=True,
        epq_stream_panic_active=True,
        epq_stream_db_active=False,
    ) == "epq_panic"
    assert resolve_one_body_mode(
        use_fused_hop=False,
        use_epq_streaming=True,
        epq_stream_panic_active=False,
        epq_stream_db_active=True,
    ) == "epq_double_buffer"
    assert resolve_one_body_mode(
        use_fused_hop=False,
        use_epq_streaming=True,
        epq_stream_panic_active=False,
        epq_stream_db_active=False,
    ) == "epq_single_buffer"
    assert resolve_one_body_mode(
        use_fused_hop=False,
        use_epq_streaming=False,
        epq_stream_panic_active=False,
        epq_stream_db_active=False,
    ) == "single_scatter"


def test_backend_hop_runtime_run_one_body_epq_streaming_panic():
    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    cp_mod = SimpleNamespace(
        asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
    )
    ws = SimpleNamespace(
        ncsf=4,
        task_csf_all=np.arange(4, dtype=np.int32),
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        overflow_apply="overflow",
        threads_apply=64,
        _dtype=np.float64,
        kahan_compensation=False,
    )
    stream = _Stream()
    calls = []

    def _apply(*args, **kwargs):
        _ = args
        calls.append(kwargs["zero_y"])

    run_one_body_epq_streaming(
        ws,
        cp=cp_mod,
        one_body_mode="epq_panic",
        x=np.ones(4),
        y=np.zeros(4),
        h_eff_flat=np.ones(4),
        stream=stream,
        sync=True,
        check_overflow=False,
        profile={},
        epq_stream_j_tile=2,
        build_epq_action_table_tile_device_fn=lambda *a, **k: None,
        apply_g_flat_scatter_atomic_inplace_device_fn=_apply,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=lambda *a, **k: None,
    )
    assert calls == [True, False]
    assert stream.nsync == 2


def test_backend_hop_runtime_run_one_body_epq_streaming_single_buffer():
    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    cp_mod = SimpleNamespace(
        asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
    )
    ws = SimpleNamespace(
        ncsf=4,
        task_csf_all=np.arange(4, dtype=np.int32),
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        overflow_apply="overflow",
        threads_apply=64,
        threads_enum=64,
        epq_stream_use_recompute=True,
        epq_recompute_warp_coop=False,
        epq_stream_pq_block=0,
        _dtype=np.float64,
        kahan_compensation=False,
    )
    stream = _Stream()
    calls = []

    def _build(*args, **kwargs):
        _ = args
        assert kwargs["j_count"] == 2
        return (np.asarray([0, 1], dtype=np.int64), np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32), np.asarray([1.0]))

    def _apply(*args, **kwargs):
        _ = args
        calls.append((kwargs["j_start"], kwargs["j_count"], kwargs["zero_y"]))

    profile = {}
    run_one_body_epq_streaming(
        ws,
        cp=cp_mod,
        one_body_mode="epq_single_buffer",
        x=np.ones(4),
        y=np.zeros(4),
        h_eff_flat=np.ones(4),
        stream=stream,
        sync=True,
        check_overflow=False,
        profile=profile,
        epq_stream_j_tile=2,
        build_epq_action_table_tile_device_fn=_build,
        apply_g_flat_scatter_atomic_inplace_device_fn=lambda *a, **k: None,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=_apply,
    )
    assert calls == [(0, 2, True), (2, 2, False)]
    assert profile["tile_apply_apply_s"] >= 0.0


def test_backend_hop_runtime_run_one_body_epq_streaming_double_buffer():
    class _Event:
        def record(self, _stream):
            return None

    class _Stream:
        def __init__(self):
            self.waited = 0

        def wait_event(self, _evt):
            self.waited += 1

        def synchronize(self):
            return None

    class _Cuda:
        @staticmethod
        def Stream(non_blocking=True):  # noqa: FBT002
            _ = non_blocking
            return _Stream()

        @staticmethod
        def Event(disable_timing=True):  # noqa: FBT002
            _ = disable_timing
            return _Event()

    cp_mod = SimpleNamespace(
        asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
        cuda=_Cuda(),
    )
    ws = SimpleNamespace(
        ncsf=4,
        task_csf_all=np.arange(4, dtype=np.int32),
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        overflow_apply="overflow",
        threads_apply=64,
        threads_enum=64,
        epq_stream_use_recompute=True,
        epq_recompute_warp_coop=False,
        epq_stream_pq_block=0,
        _dtype=np.float64,
        kahan_compensation=False,
    )
    stream = _Stream()
    calls = []

    def _build(*args, **kwargs):
        _ = args
        return (np.asarray([0, 1], dtype=np.int64), np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32), np.asarray([1.0]))

    def _apply(*args, **kwargs):
        _ = args
        calls.append((kwargs["j_start"], kwargs["j_count"], kwargs["zero_y"]))

    run_one_body_epq_streaming(
        ws,
        cp=cp_mod,
        one_body_mode="epq_double_buffer",
        x=np.ones(4),
        y=np.zeros(4),
        h_eff_flat=np.ones(4),
        stream=stream,
        sync=False,
        check_overflow=False,
        profile=None,
        epq_stream_j_tile=2,
        build_epq_action_table_tile_device_fn=_build,
        apply_g_flat_scatter_atomic_inplace_device_fn=lambda *a, **k: None,
        apply_g_flat_scatter_atomic_epq_table_tile_inplace_device_fn=_apply,
    )
    assert calls == [(0, 2, True), (2, 2, False)]
    assert stream.waited == 1


def test_backend_hop_runtime_run_one_body_single_scatter_overlap_and_direct():
    class _Stream:
        def __init__(self):
            self.tag = "main"

    class _Cuda:
        @staticmethod
        def Stream(non_blocking=True):  # noqa: FBT002
            _ = non_blocking
            return SimpleNamespace(tag="overlap")

    cp_mod = SimpleNamespace(cuda=_Cuda())
    ws = SimpleNamespace(
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        task_csf_all=np.arange(4, dtype=np.int32),
        apply_mode="scatter",
        overflow_apply="overflow",
        threads_apply=32,
        _dtype=np.float64,
        kahan_compensation=False,
        _w_offdiag=None,
    )
    calls = []

    def _apply(*args, **kwargs):
        _ = args
        calls.append(
            {
                "stream_tag": getattr(kwargs["stream"], "tag", "unknown"),
                "sync": bool(kwargs["sync"]),
                "check_overflow": bool(kwargs["check_overflow"]),
                "zero_y": bool(kwargs["zero_y"]),
            }
        )

    main_stream = _Stream()
    out_overlap = run_one_body_single_scatter(
        ws,
        cp=cp_mod,
        x=np.ones(4),
        y=np.zeros(4),
        h_eff_flat=np.ones(4),
        stream=main_stream,
        sync=True,
        check_overflow=True,
        profile=None,
        use_aggregate_offdiag=True,
        epq_table=("epq",),
        apply_g_flat_scatter_atomic_inplace_device_fn=_apply,
    )
    assert out_overlap["one_body_overlap"] is True
    assert getattr(out_overlap["stream_onebody"], "tag") == "overlap"
    assert calls[-1]["stream_tag"] == "overlap"
    assert calls[-1]["sync"] is False
    assert calls[-1]["check_overflow"] is False
    assert calls[-1]["zero_y"] is True

    out_direct = run_one_body_single_scatter(
        ws,
        cp=cp_mod,
        x=np.ones(4),
        y=np.zeros(4),
        h_eff_flat=np.ones(4),
        stream=main_stream,
        sync=True,
        check_overflow=True,
        profile={"p": 1.0},
        use_aggregate_offdiag=True,
        epq_table=("epq",),
        apply_g_flat_scatter_atomic_inplace_device_fn=_apply,
    )
    assert out_direct["one_body_overlap"] is False
    assert out_direct["stream_onebody"] is None
    assert calls[-1]["stream_tag"] == "main"
    assert calls[-1]["sync"] is True
    assert calls[-1]["check_overflow"] is True


def test_backend_hop_runtime_finalize_hop_profile_aggregates_metrics():
    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    stream = _Stream()
    profile = {
        "offdiag_gemm_s": 2.0,
        "offdiag_gemm_flops": 4.0e12,
        "offdiag_df_gemm_s": 4.0,
        "offdiag_df_gemm_flops": 2.0e12,
    }
    import time as _time

    t0 = _time.perf_counter() - 0.001
    finalize_hop_profile(
        profile=profile,
        stream=stream,
        t_total0=t0,
        csr_host_cache_hits=3,
        csr_host_cache_misses=1,
        epq_apply_cache_hits=5,
        epq_apply_cache_misses=5,
        epq_apply_cache_bytes=128,
    )
    assert profile["offdiag_gemm_tflops"] == 2.0
    assert profile["offdiag_df_gemm_tflops"] == 0.5
    assert profile["csr_host_cache_hit_rate"] == 0.75
    assert profile["epq_apply_cache_hit_rate"] == 0.5
    assert profile["epq_apply_cache_bytes"] == 128.0
    assert profile["total_s"] > 0.0
    assert stream.nsync == 1


def test_backend_hop_runtime_resolve_epq_table_for_tile_passthrough():
    ws = SimpleNamespace(_epq_table=("indptr", "indices", "pq", "data"))
    out = resolve_epq_table_for_tile(
        ws,
        use_epq_streaming_tiles=False,
        j_start=0,
        j_count=4,
        stream=None,
        sync=True,
        check_overflow=False,
        profile=None,
        build_epq_action_table_tile_device_fn=lambda *a, **k: ("tile",),
    )
    assert out == ("indptr", "indices", "pq", "data")


def test_backend_hop_runtime_resolve_epq_table_for_tile_streaming_build():
    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    calls = {}

    def _build(*args, **kwargs):
        _ = args
        calls["kwargs"] = kwargs
        return ("tile",)

    ws = SimpleNamespace(
        drt="drt",
        drt_dev="drt_dev",
        state_dev="state_dev",
        threads_enum=32,
        epq_stream_use_recompute=True,
        epq_recompute_warp_coop=False,
        epq_stream_pq_block=0,
        _dtype=np.float64,
    )
    out = resolve_epq_table_for_tile(
        ws,
        use_epq_streaming_tiles=True,
        j_start=5,
        j_count=3,
        stream=_Stream(),
        sync=False,
        check_overflow=False,
        profile={},
        build_epq_action_table_tile_device_fn=_build,
    )
    assert out == ("tile",)
    assert calls["kwargs"]["j_start"] == 5
    assert calls["kwargs"]["j_count"] == 3


def test_backend_hop_runtime_finalize_one_body_phase_records_event_and_profile():
    class _Event:
        def __init__(self):
            self.recorded = None

        def record(self, stream):
            self.recorded = stream

    class _Stream:
        def __init__(self):
            self.nsync = 0

        def synchronize(self):
            self.nsync += 1

    class _Cuda:
        @staticmethod
        def Event(disable_timing=True):  # noqa: FBT002
            _ = disable_timing
            return _Event()

    cp_mod = SimpleNamespace(cuda=_Cuda())
    main_stream = _Stream()
    overlap_stream = _Stream()
    profile = {}
    import time as _time

    evt = finalize_one_body_phase(
        cp=cp_mod,
        stream=main_stream,
        one_body_overlap=True,
        stream_onebody=overlap_stream,
        profile=profile,
        t0=_time.perf_counter() - 0.001,
    )
    assert evt.recorded is overlap_stream
    assert main_stream.nsync == 1
    assert overlap_stream.nsync == 1
    assert profile["one_body_s"] > 0.0


def test_backend_hop_runtime_resolve_tile_runtime_flags():
    ws = SimpleNamespace(
        check_overflow_mode=2,
        check_overflow_first_tile_only=True,
    )
    first = resolve_tile_runtime_flags(
        ws,
        j0=0,
        check_overflow=True,
        sync=True,
        use_csr_pipeline=True,
    )
    assert first["check_overflow_tile"] is True
    assert first["check_overflow_mode_tile"] == 2
    assert first["check_overflow_apply_tile"] is True
    assert first["tile_sync_apply"] is True

    later = resolve_tile_runtime_flags(
        ws,
        j0=128,
        check_overflow=True,
        sync=True,
        use_csr_pipeline=True,
    )
    assert later["check_overflow_tile"] is False
    assert later["check_overflow_mode_tile"] == 0
    assert later["check_overflow_apply_tile"] is False
    assert later["tile_sync_apply"] is False


def test_backend_hop_runtime_stamp_csr_prefilter_profile():
    profile = {}
    stamp_csr_prefilter_profile(
        profile=profile,
        use_csr_pipeline=True,
        pipeline_nslots=3,
        use_prefilter_trivial=False,
    )
    assert profile["csr_pipeline_active"] == 1.0
    assert profile["csr_pipeline_slots"] == 3.0
    assert profile["csr_prefilter_active"] == 0.0

    # No-op path
    stamp_csr_prefilter_profile(
        profile=None,
        use_csr_pipeline=False,
        pipeline_nslots=0,
        use_prefilter_trivial=True,
    )


def test_backend_hop_runtime_increment_profile_counter():
    profile = {}
    increment_profile_counter(profile=profile, key="k", delta=2.5)
    increment_profile_counter(profile=profile, key="k", delta=1.0)
    assert profile["k"] == 3.5
    increment_profile_counter(profile=None, key="k", delta=1.0)


def test_backend_hop_runtime_stamp_csr_prefilter_tile_profile():
    profile = {}
    out = stamp_csr_prefilter_tile_profile(profile=profile, total_tasks=10, kept_tasks=4)
    assert out["skipped_tasks"] == 6
    assert out["is_zero_tile"] is False
    assert profile["csr_prefilter_tiles"] == 1.0
    assert profile["csr_prefilter_total_tasks"] == 10.0
    assert profile["csr_prefilter_kept_tasks"] == 4.0
    assert profile["csr_prefilter_skipped_tasks"] == 6.0
    assert "csr_prefilter_zero_tiles" not in profile

    out2 = stamp_csr_prefilter_tile_profile(profile=profile, total_tasks=5, kept_tasks=0)
    assert out2["is_zero_tile"] is True
    assert profile["csr_prefilter_zero_tiles"] == 1.0


def test_backend_hop_runtime_resolve_x_tile_skip_policy_inactive():
    ws = SimpleNamespace(
        skip_zero_x_tiles_enabled=True,
        j_tile=8,
        ncsf=8,
    )
    cp_mod = SimpleNamespace(asnumpy=lambda x: np.asarray(x))
    profile = {}
    out = resolve_x_tile_skip_policy(
        ws,
        cp=cp_mod,
        x=np.ones(8),
        use_fused_hop=False,
        profile=profile,
    )
    assert out["tile_active_mask"] is None
    assert out["skip_two_body_tiles"] is False
    assert profile["x_tile_skip_policy_active"] == 0.0


def test_backend_hop_runtime_resolve_x_tile_skip_policy_zero_x():
    ws = SimpleNamespace(
        skip_zero_x_tiles_enabled=True,
        j_tile=4,
        ncsf=8,
    )
    cp_mod = SimpleNamespace(asnumpy=lambda x: np.asarray(x))
    profile = {}
    out = resolve_x_tile_skip_policy(
        ws,
        cp=cp_mod,
        x=np.zeros(8),
        use_fused_hop=False,
        profile=profile,
    )
    mask = out["tile_active_mask"]
    assert out["skip_two_body_tiles"] is True
    assert mask is not None
    assert tuple(mask.shape) == (2,)
    assert int(np.count_nonzero(mask)) == 0
    assert profile["x_tile_skip_policy_active"] == 1.0
    assert profile["x_tile_skip_mask_active"] == 1.0
    assert profile["x_tile_skipped"] == 2.0
