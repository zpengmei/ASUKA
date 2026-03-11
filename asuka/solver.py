from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence
import warnings

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.blas_threads import blas_thread_limit, openblas_get_num_threads
from asuka.cuguga.davidson import davidson1 as davidson1_sym
from asuka.cuguga.davidson import davidson1_result as davidson1_sym_result
from asuka.integrals.df_diag import diagonal_element_det_guess_df
from asuka.cuguga.oracle.matvec import matvec_df_row_oracle
from asuka.cuguga.oracle import (
    _STEP_TO_OCC,
    _csr_for_epq,
    _get_epq_action_cache,
    _restore_eri_4d,
    precompute_epq_actions,
)
from asuka.rdm.stream import make_rdm12_streaming, trans_rdm12_streaming
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import get_state_cache
from asuka._solver.warm_state import (
    normalize_warm_cas_metadata as _normalize_warm_cas_metadata,
)
from asuka._solver.cuda_policy import (
    apply_cuda_pool_hard_cap as _apply_cuda_pool_hard_cap,
    auto_gpu_mem_hard_cap as _auto_gpu_mem_hard_cap,
    cap_cuda_max_g_mib_by_hard_cap as _cap_cuda_max_g_mib_by_hard_cap_runtime,
    cuda_budget_free_bytes as _cuda_budget_free_bytes,
    enforce_cuda_aggregate_offdiag_guard as _enforce_cuda_aggregate_offdiag_guard,
    enforce_cuda_fp32_large_cas_epq_policy as _enforce_cuda_fp32_large_cas_epq_policy,
    estimate_epq_peak_bytes as _estimate_epq_peak_bytes,
    normalize_csr_host_cache_mode as _normalize_csr_host_cache_mode,
    normalize_matvec_cuda_path_mode as _normalize_matvec_cuda_path_mode,
    maybe_promote_cuda_apply_mode_scatter as _maybe_promote_cuda_apply_mode_scatter_runtime,
    normalize_prefilter_trivial_tasks_mode as _normalize_prefilter_trivial_tasks_mode,
    resolve_cuda_apply_mode as _resolve_cuda_apply_mode_runtime,
    resolve_cuda_cublas_workspace_cap_mb as _resolve_cuda_cublas_workspace_cap_mb_runtime,
    resolve_cuda_epq_build_device as _resolve_cuda_epq_build_device_runtime,
    resolve_cuda_memory_controls as _resolve_cuda_memory_controls_runtime,
    resolve_cuda_max_g_mib as _resolve_cuda_max_g_mib_runtime,
    resolve_kernel_cuda_policy as _resolve_kernel_cuda_policy_runtime,
    normalize_ws_cache_fraction as _normalize_ws_cache_fraction,
    resolve_epq_overbudget_action as _resolve_epq_overbudget_action,
    resolve_mixed_low_workspace_oom_fallback as _resolve_mixed_low_workspace_oom_fallback,
)
from asuka._solver.drt_cache import (
    DRTKey as _DRTKey,
    ne_constraints_key_to_dict as _ne_constraints_key_to_dict,
    ne_constraints_to_key as _ne_constraints_to_key,
    orbsym_to_tuple as _orbsym_to_tuple,
)
from asuka._solver.config import (
    auto_num_threads as _auto_num_threads,
    resolve_kernel_frontend_controls as _resolve_kernel_frontend_controls_runtime,
    resolve_kernel_runtime_controls as _resolve_kernel_runtime_controls_runtime,
    resolve_kernel_solver_controls as _resolve_kernel_solver_controls_runtime,
)
from asuka._solver.warm_state_runtime import (
    allowed_ci_devices_for_backend as _allowed_ci_devices_for_backend_runtime,
    load_warm_state as _load_warm_state_runtime,
    save_warm_state as _save_warm_state_runtime,
    update_warm_state as _update_warm_state_runtime,
    warm_state_ci0_if_compatible as _warm_state_ci0_if_compatible_runtime,
    warm_state_summary as _warm_state_summary_runtime,
)
from asuka._solver.drt_runtime import (
    drt_key as _drt_key_runtime,
    get_or_build_drt as _get_or_build_drt_runtime,
)
from asuka._solver.matvec_runtime import (
    estimate_matvec_cuda_workspace_bytes as _estimate_matvec_cuda_workspace_bytes_runtime,
    get_or_create_cuda_matvec_state as _get_or_create_cuda_matvec_state_runtime,
    release_matvec_cuda_workspace as _release_matvec_cuda_workspace_runtime,
    resolve_matvec_cuda_ws_cache_budget_bytes as _resolve_matvec_cuda_ws_cache_budget_bytes_runtime,
    resolve_cuda_workspace_controls as _resolve_cuda_workspace_controls_runtime,
    resolve_kernel_cuda_execution_mode as _resolve_kernel_cuda_execution_mode_runtime,
    resolve_approx_cuda_frontend as _resolve_approx_cuda_frontend_runtime,
    resolve_approx_kernel_iteration_caps as _resolve_approx_kernel_iteration_caps_runtime,
    resolve_cuda_cache_csr_tiles as _resolve_cuda_cache_csr_tiles_runtime,
    resolve_cuda_j_tile as _resolve_cuda_j_tile_runtime,
    resolve_cuda_threads_apply as _resolve_cuda_threads_apply_runtime,
    resolve_cuda_threads_w as _resolve_cuda_threads_w_runtime,
    ws_needs_rebuild as _ws_needs_rebuild_runtime,
    tune_cuda_threads_for_large_cas_noepq as _tune_cuda_threads_for_large_cas_noepq_runtime,
)
from asuka._solver.kernel_runtime import (
    autotune_cuda_max_g_mib_for_large_cas as _autotune_cuda_max_g_mib_for_large_cas_runtime,
    build_cuda_hamiltonian_inputs as _build_cuda_hamiltonian_inputs_runtime,
    build_kernel_dry_run_result as _build_kernel_dry_run_result_runtime,
    maybe_restore_contract_eri as _maybe_restore_contract_eri_runtime,
    normalize_row_screening as _normalize_row_screening_runtime,
    prepare_kernel_precompute_and_state_cache as _prepare_kernel_precompute_and_state_cache_runtime,
    resolve_kernel_nroots as _resolve_kernel_nroots_runtime,
    resolve_kernel_warm_start as _resolve_kernel_warm_start_runtime,
    run_kernel_dense_eigh_fastpath as _run_kernel_dense_eigh_fastpath_runtime,
)
from asuka._solver.kernel_cuda_runtime import (
    apply_low_precision_and_workspace_policy as _apply_low_precision_and_workspace_policy_runtime,
    auto_select_use_epq_table as _auto_select_use_epq_table_runtime,
    resolve_epq_streaming_controls as _resolve_epq_streaming_controls_runtime,
)
from asuka._solver.matvec_cache_runtime import (
    configure_matvec_cuda_ws_cache as _configure_matvec_cuda_ws_cache_runtime,
    matvec_cuda_ws_cache_drop as _matvec_cuda_ws_cache_drop_runtime,
    matvec_cuda_ws_cache_enforce_budget as _matvec_cuda_ws_cache_enforce_budget_runtime,
    matvec_cuda_ws_cache_get as _matvec_cuda_ws_cache_get_runtime,
    matvec_cuda_ws_cache_profile as _matvec_cuda_ws_cache_profile_runtime,
    matvec_cuda_ws_cache_put as _matvec_cuda_ws_cache_put_runtime,
    matvec_cuda_ws_cache_touch as _matvec_cuda_ws_cache_touch_runtime,
    release_matvec_cuda_ws_cache as _release_matvec_cuda_ws_cache_runtime,
)
from asuka._solver.dump_flags_runtime import dump_flags as _dump_flags_runtime
from asuka._solver.init_runtime import (
    configure_solver_runtime_defaults as _configure_solver_runtime_defaults_runtime,
)

if TYPE_CHECKING:  # pragma: no cover
    from asuka.contract import ContractWorkspace

try:  # optional Cython in-place CSC @ dense kernels
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_bilinear_form_cy as _csc_bilinear_form_cy,
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
        csc_quadratic_form_cy as _csc_quadratic_form_cy,
    )
except Exception:  # pragma: no cover
    _csc_bilinear_form_cy = None
    _csc_matmul_dense_inplace_cy = None
    _csc_quadratic_form_cy = None

@dataclass(frozen=True)
class H1E2EContractOp:
    """Container for passing (h1e, eri) to `contract_2e` without forming an absorbed 2e tensor.

    Notes
    -----
    PySCF Newton-CASSCF calls `fcisolver.absorb_h1e(h1, h2, ..., fac=0.5)` and then passes the
    return value to `fcisolver.contract_2e(op, civec, ...)`.  For cuGUGA we can bypass the
    determinant-style "absorb" trick and instead pass the explicit (h1e, eri) pair through.
    """

    h1e: Any  # np.ndarray or cp.ndarray
    eri: Any
    fac: float = 1.0

class _StreamObject:
    """Minimal solver base class for cuguga."""

    verbose: int
    stdout: Any

    def __init__(self) -> None:
        self.verbose = 0
        self.stdout = sys.stdout

class GUGAFCISolver(_StreamObject):
    """FCI-like solver interface for a native GUGA/CSF engine.
    """

    @staticmethod
    def _ws_needs_rebuild(
        ws,
        *,
        expected_dtype,
        j_tile,
        csr_capacity_mult,
        threads_enum,
        threads_g,
        threads_w,
        threads_apply,
        max_g_bytes,
        coalesce,
        include_diagonal_rs,
        fuse_count_write,
        fp32_coeff_data,
        path_mode,
        use_fused_hop,
        use_epq_table,
        aggregate_offdiag_k,
        l_full_d,
        enable_fp64_emulation,
        gemm_backend,
        emulation_strategy,
        cublas_workspace_cap_mb,
        apply_mode,
        epq_build_device,
        epq_build_j_tile,
        epq_streaming,
        epq_stream_j_tile,
        epq_stream_use_recompute,
        cache_csr_tiles,
        csr_host_cache_mode,
        csr_host_cache_budget_gib,
        csr_host_cache_min_ncsf,
        csr_pipeline_streams_mode,
        csr_pipeline_streams_value,
        csr_pipeline_min_ncsf,
        prefilter_trivial_tasks_mode,
        prefilter_trivial_tasks_min_ncsf,
    ) -> bool:
        """Return True if the cached workspace must be rebuilt."""
        return _ws_needs_rebuild_runtime(
            ws,
            expected_dtype=expected_dtype,
            j_tile=j_tile,
            csr_capacity_mult=csr_capacity_mult,
            threads_enum=threads_enum,
            threads_g=threads_g,
            threads_w=threads_w,
            threads_apply=threads_apply,
            max_g_bytes=max_g_bytes,
            coalesce=coalesce,
            include_diagonal_rs=include_diagonal_rs,
            fuse_count_write=fuse_count_write,
            fp32_coeff_data=fp32_coeff_data,
            path_mode=path_mode,
            use_fused_hop=use_fused_hop,
            use_epq_table=use_epq_table,
            aggregate_offdiag_k=aggregate_offdiag_k,
            l_full_d=l_full_d,
            enable_fp64_emulation=enable_fp64_emulation,
            gemm_backend=gemm_backend,
            emulation_strategy=emulation_strategy,
            cublas_workspace_cap_mb=cublas_workspace_cap_mb,
            apply_mode=apply_mode,
            epq_build_device=epq_build_device,
            epq_build_j_tile=epq_build_j_tile,
            epq_streaming=epq_streaming,
            epq_stream_j_tile=epq_stream_j_tile,
            epq_stream_use_recompute=epq_stream_use_recompute,
            cache_csr_tiles=cache_csr_tiles,
            csr_host_cache_mode=csr_host_cache_mode,
            csr_host_cache_budget_gib=csr_host_cache_budget_gib,
            csr_host_cache_min_ncsf=csr_host_cache_min_ncsf,
            csr_pipeline_streams_mode=csr_pipeline_streams_mode,
            csr_pipeline_streams_value=csr_pipeline_streams_value,
            csr_pipeline_min_ncsf=csr_pipeline_min_ncsf,
            prefilter_trivial_tasks_mode=prefilter_trivial_tasks_mode,
            prefilter_trivial_tasks_min_ncsf=prefilter_trivial_tasks_min_ncsf,
        )

    @staticmethod
    def _release_matvec_cuda_workspace(ws: Any) -> None:
        _release_matvec_cuda_workspace_runtime(ws)

    @staticmethod
    def _estimate_matvec_cuda_workspace_bytes(ws: Any) -> int:
        return _estimate_matvec_cuda_workspace_bytes_runtime(ws)

    def _resolve_matvec_cuda_ws_cache_budget_bytes(
        self,
        *,
        cp_mod: Any | None,
        hard_cap_gib: float,
        fraction: float | None = None,
    ) -> int:
        frac = self.matvec_cuda_ws_cache_fraction if fraction is None else fraction
        return _resolve_matvec_cuda_ws_cache_budget_bytes_runtime(
            cp_mod=cp_mod,
            hard_cap_gib=hard_cap_gib,
            fraction=frac,
        )

    def _matvec_cuda_ws_cache_touch(self, key: Any) -> None:
        _matvec_cuda_ws_cache_touch_runtime(self, key)

    def _matvec_cuda_ws_cache_get(self, key: Any) -> Any:
        return _matvec_cuda_ws_cache_get_runtime(self, key)

    def _matvec_cuda_ws_cache_drop(self, key: Any) -> None:
        _matvec_cuda_ws_cache_drop_runtime(self, key)

    def _matvec_cuda_ws_cache_enforce_budget(self, *, keep_keys: Sequence[Any] = ()) -> None:
        _matvec_cuda_ws_cache_enforce_budget_runtime(self, keep_keys=keep_keys)

    def _matvec_cuda_ws_cache_put(
        self,
        key: Any,
        ws: Any,
        *,
        keep_keys: Sequence[Any] = (),
    ) -> None:
        _matvec_cuda_ws_cache_put_runtime(self, key, ws, keep_keys=keep_keys)

    def release_matvec_cuda_ws_cache(self) -> int:
        """Release cached CUDA matvec workspaces (to reduce peak VRAM).

        This drops entries from the internal `_matvec_cuda_ws_cache` and calls
        the best-effort release hook on each workspace. The next CUDA matvec
        will rebuild workspaces on demand.

        Returns
        -------
        int
            Best-effort estimate of bytes held by the cache before release.
        """

        return _release_matvec_cuda_ws_cache_runtime(self)

    def _configure_matvec_cuda_ws_cache(
        self,
        *,
        cp_mod: Any | None,
        hard_cap_gib: float,
        fraction: float | None = None,
    ) -> int:
        return _configure_matvec_cuda_ws_cache_runtime(
            self,
            cp_mod=cp_mod,
            hard_cap_gib=hard_cap_gib,
            fraction=fraction,
            normalize_ws_cache_fraction_fn=_normalize_ws_cache_fraction,
            resolve_budget_bytes_fn=self._resolve_matvec_cuda_ws_cache_budget_bytes,
        )

    def _matvec_cuda_ws_cache_profile(self) -> dict[str, Any]:
        return _matvec_cuda_ws_cache_profile_runtime(self)

    def __init__(
        self,
        *,
        twos: int | None = None,
        orbsym: Optional[Sequence[int]] = None,
        wfnsym: int | None = None,
        ne_constraints: dict[int, tuple[int, int]] | None = None,
        nroots: int = 1,
    ) -> None:
        super().__init__()
        # Match PySCF-style defaults where possible, but keep the CPU path fast:
        # building the CSF pspace Hamiltonian via row-oracle calls is expensive in Python,
        # so disable it by default (users can still enable it explicitly).
        self.conv_tol = getattr(self, "conv_tol", 1e-10)
        self.pspace_size = getattr(self, "pspace_size", 0)
        self.twos = twos
        if twos is not None:
            # Keep a PySCF-compatible `spin` attribute (interpreted as 2*Sz when `nelec` is an int).
            self.spin = int(twos)
        self.orbsym = None if orbsym is None else np.asarray(orbsym, dtype=np.int32)
        self.wfnsym = wfnsym
        self.ne_constraints = None if not ne_constraints else dict(ne_constraints)
        self.nroots = int(nroots)
        if self.nroots < 1:
            raise ValueError("nroots must be >= 1")

        self._drt_cache: dict[_DRTKey, DRT] = {}
        self._tables_cache: dict[_DRTKey, Any] = {}
        self._rdm_cuda_ws_cache: dict[_DRTKey, Any] = {}
        self._matvec_cuda_state_cache: dict[_DRTKey, Any] = {}
        self._matvec_cuda_ws_cache: dict[Any, Any] = {}
        self._matvec_cuda_ws_cache_sizes: dict[Any, int] = {}
        self._matvec_cuda_ws_cache_lru: OrderedDict[Any, None] = OrderedDict()
        self._matvec_cuda_ws_cache_bytes: int = 0
        self._matvec_cuda_ws_cache_budget_bytes: int = 0
        self._matvec_cuda_ws_cache_hits: int = 0
        self._matvec_cuda_ws_cache_misses: int = 0
        self._matvec_cuda_ws_cache_evictions: int = 0
        # Keep the CPU contract workspace cache typed as `Any` so the GPU backend
        # does not import `asuka.contract` at module import time.
        self._contract_ws_cache: dict[_DRTKey, Any] = {}
        # Lightweight per-instance cache for repeated make_rdm12/make_rdm1 calls on the
        # same CI vector (common inside PySCF CASSCF/grad loops).
        self._rdm12_cache_key: tuple[object, ...] | None = None
        self._rdm12_cache_val: tuple[np.ndarray, np.ndarray] | None = None
        # Remember the most recent DRT metadata used in `kernel()`.  PySCF's symmetry-enabled
        # CASSCF calls `make_rdm12(ci, ...)` without passing `wfnsym/orbsym`, so we need a
        # fallback to avoid rebuilding an incompatible (full-symmetry) DRT.
        self._last_drt_key: _DRTKey | None = None
        _configure_solver_runtime_defaults_runtime(
            self,
            normalize_ws_cache_fraction_fn=_normalize_ws_cache_fraction,
            auto_gpu_mem_hard_cap_fn=_auto_gpu_mem_hard_cap,
        )

    def dump_flags(self, verbose: int | None = None):
        return _dump_flags_runtime(self, verbose=verbose)

    def view(self, cls):  # noqa: D401
        """Return a lightweight view of this solver as `cls`.

        This mirrors `pyscf.lib.view`:
        return a new object of type `cls` that shares the same attributes.
        """

        new_obj = cls.__new__(cls)
        new_obj.__dict__.update(self.__dict__)
        return new_obj

    def _normalize_nelec(self, nelec: int | tuple[int, int]) -> tuple[int, int, int, int]:
        if isinstance(nelec, (tuple, list, np.ndarray)):
            if len(nelec) != 2:
                raise ValueError("nelec tuple must be (neleca, nelecb)")
            neleca, nelecb = (int(nelec[0]), int(nelec[1]))
            if neleca < 0 or nelecb < 0:
                raise ValueError("neleca/nelecb must be >= 0")
            nelec_total = neleca + nelecb
        else:
            nelec_total = int(nelec)
            if nelec_total < 0:
                raise ValueError("nelec must be >= 0")
            sz_twos = int(getattr(self, "spin", 0) or 0)
            if (nelec_total + sz_twos) % 2 != 0:
                raise ValueError("nelec and 2*Sz parity mismatch")
            neleca = (nelec_total + sz_twos) // 2
            nelecb = nelec_total - neleca
        sz_twos = int(neleca - nelecb)
        return neleca, nelecb, nelec_total, sz_twos

    def _get_twos_target(self, neleca: int, nelecb: int) -> int:
        if self.twos is not None:
            twos = int(self.twos)
        else:
            twos = abs(int(neleca - nelecb))
        if twos < 0:
            raise ValueError("twos must be >= 0")
        return twos

    def _drt_key(
        self,
        norb: int,
        nelec_total: int,
        twos: int,
        orbsym: Any | None,
        wfnsym: int | None,
        *,
        ne_constraints: dict[int, tuple[int, int]] | None = None,
    ) -> _DRTKey:
        return _drt_key_runtime(
            int(norb),
            int(nelec_total),
            int(twos),
            orbsym,
            wfnsym,
            ne_constraints=ne_constraints,
        )

    def _get_drt(
        self,
        norb: int,
        nelec_total: int,
        twos: int,
        orbsym: Any | None = None,
        wfnsym: int | None = None,
        *,
        ne_constraints: dict[int, tuple[int, int]] | None = None,
    ) -> DRT:
        _, drt = _get_or_build_drt_runtime(
            self._drt_cache,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        return drt

    def _warm_state_summary(self) -> dict[str, Any] | None:
        return _warm_state_summary_runtime(self._warm_state)

    def _allowed_ci_devices_for_backend(self, matvec_backend: str) -> tuple[str, ...]:
        return _allowed_ci_devices_for_backend_runtime(matvec_backend)

    def _warm_state_ci0_if_compatible(
        self,
        *,
        norb: int,
        nelec_total: int,
        twos: int,
        nroots: int,
        ncsf: int,
        orbsym: Any | None,
        wfnsym: int | None,
        ne_constraints: dict[int, tuple[int, int]] | None,
        matvec_backend: str,
        cas_metadata: dict[str, Any],
    ) -> tuple[list[np.ndarray] | None, str]:
        return _warm_state_ci0_if_compatible_runtime(
            state=self._warm_state,
            norb=int(norb),
            nelec_total=int(nelec_total),
            twos=int(twos),
            nroots=int(nroots),
            ncsf=int(ncsf),
            orbsym_key=_orbsym_to_tuple(orbsym),
            wfnsym=None if wfnsym is None else int(wfnsym),
            ne_constraints_key=_ne_constraints_to_key(ne_constraints),
            matvec_backend=matvec_backend,
            cas_metadata=cas_metadata,
        )

    def _update_warm_state(
        self,
        *,
        ci: Any,
        norb: int,
        nelec_total: int,
        twos: int,
        nroots: int,
        ncsf: int,
        orbsym: Any | None,
        wfnsym: int | None,
        ne_constraints: dict[int, tuple[int, int]] | None,
        cas_metadata: dict[str, Any],
        mo_coeff: Any | None,
        mo_occ: Any | None,
    ) -> None:
        self._warm_state = _update_warm_state_runtime(
            prev_state=self._warm_state,
            normalize_ci0_fn=_normalize_ci0,
            ci=ci,
            norb=int(norb),
            nelec_total=int(nelec_total),
            twos=int(twos),
            nroots=int(nroots),
            ncsf=int(ncsf),
            orbsym_key=_orbsym_to_tuple(orbsym),
            wfnsym=None if wfnsym is None else int(wfnsym),
            ne_constraints_key=_ne_constraints_to_key(ne_constraints),
            cas_metadata=cas_metadata,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
        )

    def save_warm_state(
        self,
        path: str | os.PathLike[str],
        *,
        include_ci: bool = True,
        include_mo: bool = True,
    ) -> str:
        return _save_warm_state_runtime(
            path,
            state=self._warm_state,
            include_ci=bool(include_ci),
            include_mo=bool(include_mo),
        )

    def load_warm_state(
        self,
        path: str | os.PathLike[str],
        *,
        require_ci: bool = False,
    ) -> dict[str, Any]:
        self._warm_state = _load_warm_state_runtime(path, require_ci=bool(require_ci))
        self._last_warm_start_info = None
        summary = self._warm_state_summary()
        assert summary is not None
        return summary

    def clear_warm_state(self) -> None:
        self._warm_state = None
        self._last_warm_start_info = None

    def _call_davidson_gpu(
        self,
        *,
        hop_fn,
        hop_low_fn,
        hop_prof,
        hop_prof_low,
        ci0,
        hdiag,
        nroots,
        max_cycle,
        max_space,
        tol,
        lindep,
        cuda_cp,
        kprof,
        kernel_profile,
        kernel_profile_cuda_sync,
        matvec_cuda_dtype,
        matvec_cuda_mixed_threshold,
        matvec_cuda_mixed_low_precision_max_iter,
        matvec_cuda_mixed_force_final_full_hop,
        matvec_cuda_mixed_final_full_subspace_refresh,
        matvec_cuda_davidson_subspace_eigh_cpu,
        matvec_cuda_davidson_subspace_eigh_cpu_max_m,
    ):
        """Unified GPU Davidson call with profiling and mixed-precision support."""
        from asuka.cuda.cuda_davidson import davidson_sym_gpu  # noqa: PLC0415

        is_mixed = str(matvec_cuda_dtype) == "mixed"

        if kprof is not None:
            t0 = time.perf_counter()
            kprof["davidson_subspace_eigh_cpu"] = bool(matvec_cuda_davidson_subspace_eigh_cpu)
            kprof["davidson_subspace_eigh_cpu_max_m"] = int(matvec_cuda_davidson_subspace_eigh_cpu_max_m)
            kprof["davidson_hop_dtype"] = str(matvec_cuda_dtype)
            if matvec_cuda_mixed_threshold is not None:
                kprof["davidson_mixed_threshold"] = float(matvec_cuda_mixed_threshold)
            if is_mixed:
                kprof["davidson_mixed_force_final_full_hop"] = bool(matvec_cuda_mixed_force_final_full_hop)
                kprof["davidson_mixed_final_full_subspace_refresh"] = bool(
                    matvec_cuda_mixed_final_full_subspace_refresh
                )
                kprof["davidson_mixed_low_precision_max_iter"] = (
                    None
                    if matvec_cuda_mixed_low_precision_max_iter is None
                    else int(matvec_cuda_mixed_low_precision_max_iter)
                )

        res = davidson_sym_gpu(
            hop_fn,
            hop_low_precision=hop_low_fn,
            hop_low_precision_threshold=(
                float(matvec_cuda_mixed_threshold) if is_mixed else None
            ),
            hop_low_precision_max_iter=(
                matvec_cuda_mixed_low_precision_max_iter if is_mixed else None
            ),
            force_final_full_precision_hop=(
                bool(matvec_cuda_mixed_force_final_full_hop) if is_mixed else False
            ),
            force_final_full_subspace_refresh=(
                bool(matvec_cuda_mixed_final_full_subspace_refresh) if is_mixed else False
            ),
            x0=ci0,
            hdiag=hdiag,
            nroots=nroots,
            max_cycle=max_cycle,
            max_space=max_space,
            tol=tol,
            lindep=lindep,
            denom_tol=1e-12,
            stream=cuda_cp.cuda.get_current_stream(),
            profile=kernel_profile,
            profile_cuda_sync=kernel_profile_cuda_sync,
            subspace_eigh_cpu=bool(matvec_cuda_davidson_subspace_eigh_cpu),
            subspace_eigh_cpu_max_m=int(matvec_cuda_davidson_subspace_eigh_cpu_max_m),
        )

        if kprof is not None:
            if kernel_profile_cuda_sync:
                cuda_cp.cuda.get_current_stream().synchronize()
            kprof["davidson_s"] = time.perf_counter() - t0
            kprof["davidson_niter"] = int(getattr(res, "niter", -1))
            kprof["davidson_stats"] = getattr(res, "stats", None)
            if is_mixed:
                _dstats = dict(getattr(res, "stats", {}) or {})
                kprof["davidson_mixed_final_full_correction_executed"] = bool(
                    _dstats.get("hop_final_full_precision_correction_executed", False)
                )
                kprof["davidson_mixed_final_full_subspace_refresh_executed"] = bool(
                    _dstats.get("hop_final_full_subspace_refresh_executed", False)
                )
                kprof["davidson_mixed_final_full_subspace_refresh_basis_size"] = float(
                    _dstats.get("hop_final_full_subspace_refresh_basis_size", -1.0)
                )
                kprof["davidson_mixed_final_mode"] = str(
                    _dstats.get("hop_low_precision_final_mode", "unknown")
                )
                kprof["davidson_mixed_switch_residual_max"] = float(
                    _dstats.get("hop_low_precision_switch_residual_max", -1.0)
                )
                kprof["davidson_mixed_switch_reason"] = str(
                    _dstats.get("hop_low_precision_switch_reason", "none")
                )
                kprof["matvec_cuda_hop_profile"] = {
                    "full_precision": hop_prof,
                    "low_precision": hop_prof_low,
                }
            else:
                kprof["matvec_cuda_hop_profile"] = hop_prof

        return res

    def kernel(
        self,
        h1e,
        eri,
        norb: int,
        nelec: int | tuple[int, int],
        ci0=None,
        ecore: float = 0.0,
        nroots: int | None = None,
        **kwargs,
    ):
        _frontend_controls = _resolve_kernel_frontend_controls_runtime(kwargs=kwargs, defaults=self)
        kernel_profile = bool(_frontend_controls["kernel_profile"])
        kernel_profile_cuda_sync = bool(_frontend_controls["kernel_profile_cuda_sync"])
        kernel_profile_print = bool(_frontend_controls["kernel_profile_print"])
        matvec_cuda_hop_profile = bool(_frontend_controls["matvec_cuda_hop_profile"])
        matvec_cuda_davidson_subspace_eigh_cpu_in = _frontend_controls[
            "matvec_cuda_davidson_subspace_eigh_cpu_in"
        ]
        matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff = int(
            _frontend_controls["matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff"]
        )
        matvec_cuda_davidson_subspace_eigh_cpu_max_m = int(
            _frontend_controls["matvec_cuda_davidson_subspace_eigh_cpu_max_m"]
        )
        kprof: dict[str, Any] | None = {} if kernel_profile else None
        t_kernel0 = time.perf_counter()
        dry_run = bool(_frontend_controls["dry_run"])
        warm_state_enable = bool(_frontend_controls["warm_state_enable"])
        warm_state_update = bool(_frontend_controls["warm_state_update"])
        warm_state_context_in = _frontend_controls["warm_state_context_in"]
        warm_state_mo_coeff = _frontend_controls["warm_state_mo_coeff"]
        warm_state_mo_occ = _frontend_controls["warm_state_mo_occ"]

        orbsym = _frontend_controls["orbsym"]
        wfnsym = _frontend_controls["wfnsym"]
        matvec_backend = str(_frontend_controls["matvec_backend"])
        strict_gpu = bool(_frontend_controls["strict_gpu"])
        _cuda_policy = _resolve_kernel_cuda_policy_runtime(
            kwargs=kwargs,
            defaults=self,
            matvec_backend=matvec_backend,
            strict_gpu=bool(strict_gpu),
        )
        matvec_cuda_aggregate_offdiag_preview = _cuda_policy["matvec_cuda_aggregate_offdiag_preview"]
        if kprof is not None:
            kprof.update(dict(_cuda_policy["profile"]))
        row_screening = _normalize_row_screening_runtime(
            row_screening=kwargs.pop("row_screening", None),
            row_screening_type=RowScreening,
        )
        _runtime_controls = _resolve_kernel_runtime_controls_runtime(
            kwargs=kwargs,
            defaults=self,
            matvec_backend=matvec_backend,
            auto_num_threads_fn=_auto_num_threads,
        )
        ne_constraints = _runtime_controls["ne_constraints"]
        row_oracle_use_state_cache = bool(_runtime_controls["row_oracle_use_state_cache"])
        precompute_epq = bool(_runtime_controls["precompute_epq"])
        max_out = int(_runtime_controls["max_out"])
        unconverged_fallback_full_diag = bool(_runtime_controls["unconverged_fallback_full_diag"])
        unconverged_fallback_ncsf_max = int(_runtime_controls["unconverged_fallback_ncsf_max"])
        raise_on_unconverged = bool(_runtime_controls["raise_on_unconverged"])
        warn_on_unconverged = bool(_runtime_controls["warn_on_unconverged"])
        contract_nthreads = int(_runtime_controls["contract_nthreads"])
        contract_blas_nthreads = _runtime_controls["contract_blas_nthreads"]
        kernel_blas_nthreads = _runtime_controls["kernel_blas_nthreads"]
        # For the dense CPU contract backend, PySCF commonly supplies CAS ERIs in sym=4 pair-matrix
        # form (npair,npair). Restoring to a 4-index tensor inside each matvec hop would be extremely
        # costly; restore once up-front when needed.
        def _restore_eri1_lazy(eri_in, norb_in):
            from asuka.cuguga.eri import restore_eri1  # noqa: PLC0415

            return restore_eri1(eri_in, int(norb_in))

        eri = _maybe_restore_contract_eri_runtime(
            matvec_backend=matvec_backend,
            eri=eri,
            norb=int(norb),
            df_types=(DFMOIntegrals, DeviceDFMOIntegrals),
            restore_eri1_fn=_restore_eri1_lazy,
        )

        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        warm_cas_metadata = _normalize_warm_cas_metadata(
            warm_state_context_in,
            default_ncas=int(norb),
            default_nelecas=(int(neleca), int(nelecb)),
        )
        drt_key = self._drt_key(
            norb,
            nelec_total,
            twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        drt = self._get_drt(
            norb,
            nelec_total,
            twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        ncsf = int(drt.ncsf)
        _cuda_mode_cfg = _resolve_kernel_cuda_execution_mode_runtime(
            kwargs=kwargs,
            defaults=self,
            matvec_backend=matvec_backend,
            strict_gpu=bool(strict_gpu),
            ncsf=int(ncsf),
            matvec_cuda_davidson_subspace_eigh_cpu_in=matvec_cuda_davidson_subspace_eigh_cpu_in,
            matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff=int(
                matvec_cuda_davidson_subspace_eigh_cpu_ncsf_cutoff
            ),
        )
        matvec_cuda_davidson_subspace_eigh_cpu = _cuda_mode_cfg["matvec_cuda_davidson_subspace_eigh_cpu"]
        matvec_cuda_make_hdiag_cpu = _cuda_mode_cfg["matvec_cuda_make_hdiag_cpu"]
        matvec_cuda_make_hdiag_cpu_ncsf_cutoff = _cuda_mode_cfg["matvec_cuda_make_hdiag_cpu_ncsf_cutoff"]
        matvec_cuda_dtype = _cuda_mode_cfg["matvec_cuda_dtype"]
        matvec_cuda_mixed_threshold = _cuda_mode_cfg["matvec_cuda_mixed_threshold"]
        matvec_cuda_mixed_force_final_full_hop = _cuda_mode_cfg["matvec_cuda_mixed_force_final_full_hop"]
        matvec_cuda_mixed_final_full_subspace_refresh = _cuda_mode_cfg[
            "matvec_cuda_mixed_final_full_subspace_refresh"
        ]
        matvec_cuda_mixed_low_precision_max_iter = _cuda_mode_cfg["matvec_cuda_mixed_low_precision_max_iter"]

        if matvec_backend in ("cuda_eri_mat", "cuda"):
            matvec_cuda_use_epq_preview_in = kwargs.get(
                "matvec_cuda_use_epq_table",
                getattr(self, "matvec_cuda_use_epq_table", None),
            )
            if matvec_cuda_use_epq_preview_in is not None:
                _enforce_cuda_fp32_large_cas_epq_policy(
                    context="kernel(cuda)",
                    matvec_cuda_dtype=str(matvec_cuda_dtype),
                    matvec_cuda_use_epq_table=bool(matvec_cuda_use_epq_preview_in),
                    matvec_cuda_aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag_preview),
                    ncsf=int(ncsf),
                )

        nroots = _resolve_kernel_nroots_runtime(
            requested_nroots=nroots,
            defaults=self,
            ncsf=int(ncsf),
        )

        _dense_fast = _run_kernel_dense_eigh_fastpath_runtime(
            solver=self,
            h1e=h1e,
            eri=eri,
            norb=int(norb),
            nelec=nelec,
            ncsf=int(ncsf),
            nroots=int(nroots),
            ecore=float(ecore),
            max_out=int(max_out),
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
            drt_key=drt_key,
            warm_state_update=bool(warm_state_update),
            nelec_total=int(nelec_total),
            twos=int(twos),
            cas_metadata=warm_cas_metadata,
            warm_state_mo_coeff=warm_state_mo_coeff,
            warm_state_mo_occ=warm_state_mo_occ,
            t_kernel0=float(t_kernel0),
        )
        if _dense_fast is not None:
            self.converged = _dense_fast["converged"]
            e = np.asarray(_dense_fast["e"], dtype=np.float64)
            ci = _dense_fast["ci"]
            if kprof is not None:
                kprof["dense_eigh_used"] = True
                kprof["dense_eigh_ncsf"] = int(_dense_fast["dense_eigh_ncsf"])
                kprof["dense_eigh_s"] = float(_dense_fast["dense_eigh_s"])
                kprof["total_s"] = float(_dense_fast["total_s"])
                self._last_kernel_profile = kprof
            if nroots == 1:
                self.converged = bool(self.converged[0])
                self.eci, self.ci = float(e[0]), np.ascontiguousarray(ci[0])
                self._last_drt_key = drt_key
                return self.eci, self.ci
            self.eci, self.ci = e, [np.ascontiguousarray(v) for v in ci]
            self._last_drt_key = drt_key
            return self.eci, self.ci

        _warm = _resolve_kernel_warm_start_runtime(
            ci0=ci0,
            warm_state_enable=bool(warm_state_enable),
            warm_state_ci0_if_compatible_fn=self._warm_state_ci0_if_compatible,
            norb=int(norb),
            nelec_total=int(nelec_total),
            twos=int(twos),
            nroots=int(nroots),
            ncsf=int(ncsf),
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
            matvec_backend=matvec_backend,
            cas_metadata=warm_cas_metadata,
        )
        ci0 = _warm["ci0"]
        warm_applied = bool(_warm["warm_applied"])
        warm_reason = str(_warm["warm_reason"])

        self._last_warm_start_info = {"applied": bool(warm_applied), "reason": str(warm_reason)}
        if kprof is not None:
            kprof["warm_start_applied"] = bool(warm_applied)
            kprof["warm_start_reason"] = str(warm_reason)

        if dry_run:
            _dry = _build_kernel_dry_run_result_runtime(
                ci0=ci0,
                nroots=int(nroots),
                ncsf=int(ncsf),
                ecore=float(ecore),
                normalize_ci0_fn=_normalize_ci0,
            )
            e = np.asarray(_dry["e"], dtype=np.float64)
            ci = _dry["ci"]
            if warm_state_update:
                self._update_warm_state(
                    ci=ci,
                    norb=int(norb),
                    nelec_total=int(nelec_total),
                    twos=int(twos),
                    nroots=int(nroots),
                    ncsf=int(ncsf),
                    orbsym=orbsym,
                    wfnsym=wfnsym,
                    ne_constraints=ne_constraints,
                    cas_metadata=warm_cas_metadata,
                    mo_coeff=warm_state_mo_coeff,
                    mo_occ=warm_state_mo_occ,
                )
            if nroots == 1:
                return float(e[0]), ci[0]
            return e, ci

        # Solver controls (match common PySCF naming where possible)
        _solver_controls = _resolve_kernel_solver_controls_runtime(kwargs=kwargs, defaults=self)
        tol = float(_solver_controls["tol"])
        lindep = float(_solver_controls["lindep"])
        max_cycle = int(_solver_controls["max_cycle"])
        max_space = int(_solver_controls["max_space"])
        max_memory = float(_solver_controls["max_memory"])
        pspace_size = int(_solver_controls["pspace_size"])

        state_cache = _prepare_kernel_precompute_and_state_cache_runtime(
            precompute_epq=bool(precompute_epq),
            drt=drt,
            matvec_backend=matvec_backend,
            row_oracle_use_state_cache=bool(row_oracle_use_state_cache),
            precompute_epq_actions_fn=precompute_epq_actions,
            get_state_cache_fn=get_state_cache,
        )

        # Cache the most recent Hamiltonian for `contract_2e` convenience.
        self._h1e = np.asarray(h1e, dtype=np.float64)

        cuda_matvec_ws = None
        cuda_ws_low = None
        cuda_fixed_ell_ws = None
        cuda_fixed_sell_ws = None
        cuda_cp = None
        if matvec_backend in ("cuda_eri_mat", "cuda"):
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception as e:  # pragma: no cover
                raise RuntimeError("matvec_backend='cuda_eri_mat' requires CuPy") from e

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                GugaMatvecEriMatWorkspace,
                has_cuda_ext,
                has_epq_table_device_build,
                make_device_drt,
                make_device_state_cache,
            )

            _cuda_mem_ctl = _resolve_cuda_memory_controls_runtime(
                kwargs=kwargs,
                defaults=self,
                consume=True,
            )
            matvec_cuda_mem_hard_cap_gib = float(_cuda_mem_ctl["matvec_cuda_mem_hard_cap_gib"])
            matvec_cuda_ws_cache_fraction = float(_cuda_mem_ctl["matvec_cuda_ws_cache_fraction"])
            _cuda_pool_limit_b = _apply_cuda_pool_hard_cap(cp, float(matvec_cuda_mem_hard_cap_gib))
            _cuda_ws_cache_budget_b = self._configure_matvec_cuda_ws_cache(
                cp_mod=cp,
                hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                fraction=float(matvec_cuda_ws_cache_fraction),
            )

            if not has_cuda_ext():
                raise RuntimeError(
                    "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
                )

            if kprof is not None:
                if _cuda_pool_limit_b is not None:
                    kprof["matvec_cuda_pool_limit_bytes"] = int(_cuda_pool_limit_b)
                kprof["matvec_cuda_ws_cache_budget_bytes"] = int(_cuda_ws_cache_budget_b)
                kprof["matvec_cuda_ws_cache_fraction"] = float(self.matvec_cuda_ws_cache_fraction)

            ws_key = self._drt_key(
                norb,
                nelec_total,
                twos,
                orbsym=orbsym,
                wfnsym=wfnsym,
                ne_constraints=ne_constraints,
            )
            drt_dev, state_dev = _get_or_create_cuda_matvec_state_runtime(
                state_cache=self._matvec_cuda_state_cache,
                ws_key=ws_key,
                drt=drt,
                make_device_drt_fn=make_device_drt,
                make_device_state_cache_fn=make_device_state_cache,
            )

            norb_i = int(drt.norb)
            nops = norb_i * norb_i

            matvec_cuda_df_eri_mat_max_mib = float(
                kwargs.pop(
                    "matvec_cuda_df_eri_mat_max_mib",
                    getattr(self, "matvec_cuda_df_eri_mat_max_mib", 256.0),
                )
            )
            matvec_cuda_df_eri_mat_max_bytes = int(max(0.0, float(matvec_cuda_df_eri_mat_max_mib)) * 1024 * 1024)

            # GPU-resident coefficients (built once per solve call).
            if kprof is not None:
                t0 = time.perf_counter()
            _ham_inputs = _build_cuda_hamiltonian_inputs_runtime(
                cp=cp,
                eri=eri,
                h1e=h1e,
                norb=int(norb_i),
                df_eri_mat_max_bytes=int(matvec_cuda_df_eri_mat_max_bytes),
                df_type=DFMOIntegrals,
                device_df_type=DeviceDFMOIntegrals,
                restore_eri_4d_fn=_restore_eri_4d,
            )
            eri_mat_d = _ham_inputs["eri_mat_d"]
            l_full_d = _ham_inputs["l_full_d"]
            h_eff_d = _ham_inputs["h_eff_d"]
            if matvec_backend == "cuda_eri_mat" and eri_mat_d is None:
                raise RuntimeError(
                    "matvec_backend='cuda_eri_mat' requires eri_mat; use matvec_backend='cuda' to run the DF L_full path"
                )
            if kprof is not None:
                if kernel_profile_cuda_sync:
                    cp.cuda.get_current_stream().synchronize()
                kprof["h_eff_eri_to_gpu_s"] = time.perf_counter() - t0

            _cuda_ws_controls = _resolve_cuda_workspace_controls_runtime(
                kwargs=kwargs,
                defaults=self,
                consume=True,
                context="kernel(cuda)",
            )
            matvec_cuda_target_ntasks = int(_cuda_ws_controls["matvec_cuda_target_ntasks"])
            matvec_cuda_j_tile_align = int(_cuda_ws_controls["matvec_cuda_j_tile_align"])
            matvec_cuda_j_tile = int(_cuda_ws_controls["matvec_cuda_j_tile_requested"])
            matvec_cuda_j_tile = _resolve_cuda_j_tile_runtime(
                requested_j_tile=matvec_cuda_j_tile,
                target_ntasks=matvec_cuda_target_ntasks,
                j_tile_align=matvec_cuda_j_tile_align,
                norb=norb_i,
                ncsf=int(drt.ncsf),
            )
            matvec_cuda_csr_capacity_mult = float(_cuda_ws_controls["matvec_cuda_csr_capacity_mult"])
            matvec_cuda_csr_host_cache_mode = str(_cuda_ws_controls["matvec_cuda_csr_host_cache_mode"])
            matvec_cuda_csr_host_cache_budget_gib = float(_cuda_ws_controls["matvec_cuda_csr_host_cache_budget_gib"])
            matvec_cuda_csr_host_cache_min_ncsf = int(_cuda_ws_controls["matvec_cuda_csr_host_cache_min_ncsf"])
            matvec_cuda_csr_pipeline_streams_mode = str(_cuda_ws_controls["matvec_cuda_csr_pipeline_streams_mode"])
            matvec_cuda_csr_pipeline_streams_value = _cuda_ws_controls["matvec_cuda_csr_pipeline_streams_value"]
            if matvec_cuda_csr_pipeline_streams_value is None:
                matvec_cuda_csr_pipeline_streams = str(matvec_cuda_csr_pipeline_streams_mode)
            else:
                matvec_cuda_csr_pipeline_streams = int(matvec_cuda_csr_pipeline_streams_value)
            matvec_cuda_csr_pipeline_min_ncsf = int(_cuda_ws_controls["matvec_cuda_csr_pipeline_min_ncsf"])
            matvec_cuda_prefilter_trivial_tasks_mode = str(
                _cuda_ws_controls["matvec_cuda_prefilter_trivial_tasks_mode"]
            )
            matvec_cuda_prefilter_trivial_tasks_min_ncsf = int(
                _cuda_ws_controls["matvec_cuda_prefilter_trivial_tasks_min_ncsf"]
            )
            threads_enum_forced = bool(_cuda_ws_controls["threads_enum_forced"])
            threads_g_forced = bool(_cuda_ws_controls["threads_g_forced"])
            matvec_cuda_threads_enum = int(_cuda_ws_controls["matvec_cuda_threads_enum"])
            matvec_cuda_threads_g = int(_cuda_ws_controls["matvec_cuda_threads_g"])
            matvec_cuda_threads_w = int(_cuda_ws_controls["matvec_cuda_threads_w"])
            matvec_cuda_threads_apply = int(_cuda_ws_controls["matvec_cuda_threads_apply"])
            threads_apply_auto = bool(_cuda_ws_controls["threads_apply_auto"])
            matvec_cuda_coalesce = bool(_cuda_ws_controls["matvec_cuda_coalesce"])
            matvec_cuda_include_diagonal_rs = bool(_cuda_ws_controls["matvec_cuda_include_diagonal_rs"])
            matvec_cuda_cache_csr_tiles = _cuda_ws_controls["matvec_cuda_cache_csr_tiles"]
            matvec_cuda_fuse_count_write = bool(_cuda_ws_controls["matvec_cuda_fuse_count_write"])
            matvec_cuda_path_mode = _normalize_matvec_cuda_path_mode(
                kwargs.pop(
                    "matvec_cuda_path_mode",
                    getattr(self, "matvec_cuda_path_mode", "auto"),
                )
            )
            matvec_cuda_use_fused_hop = bool(
                kwargs.pop(
                    "matvec_cuda_use_fused_hop",
                    getattr(self, "matvec_cuda_use_fused_hop", True),
                )
            )
            matvec_cuda_fp32_coeff_data = bool(_cuda_ws_controls["matvec_cuda_fp32_coeff_data"])
            matvec_cuda_use_epq_table_in = _cuda_ws_controls["matvec_cuda_use_epq_table_in"]
            epq_table_forced = matvec_cuda_use_epq_table_in is not None
            matvec_cuda_aggregate_offdiag = bool(_cuda_ws_controls["matvec_cuda_aggregate_offdiag"])
            _max_g_cfg = _resolve_cuda_max_g_mib_runtime(
                kwargs=kwargs,
                defaults=self,
                consume=True,
            )
            matvec_cuda_max_g_mib = float(_max_g_cfg["matvec_cuda_max_g_mib"])
            max_g_forced = bool(_max_g_cfg["max_g_forced"])
            matvec_cuda_max_g_mib = _autotune_cuda_max_g_mib_for_large_cas_runtime(
                max_g_mib=float(matvec_cuda_max_g_mib),
                max_g_forced=bool(max_g_forced),
                aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                ncsf=int(drt.ncsf),
                norb=int(norb_i),
                matvec_cuda_dtype=str(matvec_cuda_dtype),
                eri_mat_present=bool(eri_mat_d is not None),
                mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                cuda_budget_free_bytes_fn=_cuda_budget_free_bytes,
            )
            matvec_cuda_enable_fp64_emulation = bool(
                kwargs.pop(
                    "matvec_cuda_enable_fp64_emulation",
                    getattr(self, "matvec_cuda_enable_fp64_emulation", False),
                )
            )
            gemm_backend_forced = "matvec_cuda_gemm_backend" in kwargs
            matvec_cuda_gemm_backend = str(
                kwargs.pop(
                    "matvec_cuda_gemm_backend",
                    getattr(self, "matvec_cuda_gemm_backend", "gemmex_fp64"),
                )
            ).strip().lower()
            matvec_cuda_gemm_backend_low = str(matvec_cuda_gemm_backend)
            fp64_gemm_backends = ("gemmex_fp64", "cublaslt_fp64")
            fp32_gemm_backends = ("gemmex_fp32", "gemmex_tf32", "cublaslt_fp32", "cublaslt_tf32")
            if (
                (not gemm_backend_forced)
                and str(matvec_cuda_dtype) == "float32"
                and str(matvec_cuda_gemm_backend) in fp64_gemm_backends
            ):
                matvec_cuda_gemm_backend = "gemmex_tf32"
                matvec_cuda_gemm_backend_low = "gemmex_tf32"
            if str(matvec_cuda_dtype) == "mixed":
                if str(matvec_cuda_gemm_backend) in fp32_gemm_backends:
                    if gemm_backend_forced:
                        raise ValueError(
                            "matvec_cuda_dtype='mixed' requires matvec_cuda_gemm_backend in {gemmex_fp64, cublaslt_fp64} "
                            "for the full-precision workspace"
                        )
                    matvec_cuda_gemm_backend = "gemmex_fp64"
                matvec_cuda_gemm_backend_low = "gemmex_fp32"
            matvec_cuda_emulation_strategy = str(
                kwargs.pop(
                    "matvec_cuda_emulation_strategy",
                    getattr(self, "matvec_cuda_emulation_strategy", "performant"),
                )
            )
            matvec_cuda_cublas_workspace_cap_mb = _resolve_cuda_cublas_workspace_cap_mb_runtime(
                kwargs=kwargs,
                defaults=self,
                hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                consume=True,
            )
            _apply_mode_cfg = _resolve_cuda_apply_mode_runtime(
                kwargs=kwargs,
                defaults=self,
                consume=True,
            )
            apply_mode_forced = bool(_apply_mode_cfg["apply_mode_forced"])
            matvec_cuda_apply_mode = str(_apply_mode_cfg["matvec_cuda_apply_mode"])
            matvec_cuda_use_graph = bool(
                kwargs.pop(
                    "matvec_cuda_use_graph",
                    getattr(self, "matvec_cuda_use_graph", False),
                )
            )
            matvec_cuda_graph_warmup = bool(
                kwargs.pop(
                    "matvec_cuda_graph_warmup",
                    getattr(self, "matvec_cuda_graph_warmup", True),
                )
            )
            matvec_cuda_epq_build_nthreads = int(
                kwargs.pop(
                    "matvec_cuda_epq_build_nthreads",
                    getattr(self, "matvec_cuda_epq_build_nthreads", 0),
                )
            )
            matvec_cuda_epq_build_device = kwargs.pop(
                "matvec_cuda_epq_build_device",
                getattr(self, "matvec_cuda_epq_build_device", None),
            )
            matvec_cuda_epq_build_j_tile = int(
                kwargs.pop(
                    "matvec_cuda_epq_build_j_tile",
                    getattr(self, "matvec_cuda_epq_build_j_tile", 0),
                )
            )
            _epq_stream_cfg = _resolve_epq_streaming_controls_runtime(
                epq_streaming_in=kwargs.pop(
                    "matvec_cuda_epq_streaming",
                    getattr(self, "matvec_cuda_epq_streaming", "auto"),
                ),
                epq_stream_j_tile_in=kwargs.pop(
                    "matvec_cuda_epq_stream_j_tile",
                    getattr(self, "matvec_cuda_epq_stream_j_tile", 0),
                ),
                epq_stream_use_recompute_in=kwargs.pop(
                    "matvec_cuda_epq_stream_use_recompute",
                    getattr(self, "matvec_cuda_epq_stream_use_recompute", "auto"),
                ),
            )
            matvec_cuda_epq_streaming_mode = str(_epq_stream_cfg["streaming_mode"])
            matvec_cuda_epq_streaming = bool(_epq_stream_cfg["streaming"])
            matvec_cuda_epq_stream_j_tile = int(_epq_stream_cfg["stream_j_tile"])
            matvec_cuda_epq_stream_use_recompute = _epq_stream_cfg["stream_use_recompute"]
            if matvec_cuda_use_epq_table_in is None:
                matvec_cuda_use_epq_table = _auto_select_use_epq_table_runtime(
                    cp=cp,
                    norb=int(norb_i),
                    ncsf=int(drt.ncsf),
                    aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                    has_epq_table_device_build=bool(has_epq_table_device_build()),
                    mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                    dtype_mode=str(matvec_cuda_dtype),
                    eri_mat_present=bool(eri_mat_d is not None),
                )
            else:
                matvec_cuda_use_epq_table = bool(matvec_cuda_use_epq_table_in)

            _policy_apply = _apply_low_precision_and_workspace_policy_runtime(
                context="kernel(cuda)",
                dtype_mode=str(matvec_cuda_dtype),
                use_epq_table=bool(matvec_cuda_use_epq_table),
                aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                ncsf=int(drt.ncsf),
                eri_mat_present=bool(eri_mat_d is not None),
                enable_fp64_emulation=bool(matvec_cuda_enable_fp64_emulation),
                use_graph=bool(matvec_cuda_use_graph),
                apply_mode=str(matvec_cuda_apply_mode),
                apply_mode_forced=bool(apply_mode_forced),
                nops=int(nops),
                threads_enum=int(matvec_cuda_threads_enum),
                threads_g=int(matvec_cuda_threads_g),
                threads_w=int(matvec_cuda_threads_w),
                threads_apply=int(matvec_cuda_threads_apply),
                threads_enum_forced=bool(threads_enum_forced),
                threads_g_forced=bool(threads_g_forced),
                threads_apply_auto=bool(threads_apply_auto),
                max_g_mib=float(matvec_cuda_max_g_mib),
                mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                cache_csr_tiles_in=matvec_cuda_cache_csr_tiles,
                j_tile=int(matvec_cuda_j_tile),
                norb=int(norb_i),
                csr_capacity_mult=float(matvec_cuda_csr_capacity_mult),
                noepq_large_ncsf_uses_64=True,
            )
            if bool(_policy_apply["graph_disabled"]):
                warnings.warn("matvec_cuda_dtype float32/mixed: disabling CUDA Graph capture (float64-only path)")
            matvec_cuda_use_graph = bool(_policy_apply["use_graph"])
            matvec_cuda_apply_mode = str(_policy_apply["apply_mode"])
            matvec_cuda_threads_enum = int(_policy_apply["threads_enum"])
            matvec_cuda_threads_g = int(_policy_apply["threads_g"])
            matvec_cuda_threads_w = int(_policy_apply["threads_w"])
            matvec_cuda_threads_apply = int(_policy_apply["threads_apply"])
            matvec_cuda_max_g_mib = float(_policy_apply["max_g_mib"])
            matvec_cuda_cache_csr_tiles = _policy_apply["cache_csr_tiles"]

            matvec_cuda_epq_build_device = _resolve_cuda_epq_build_device_runtime(
                epq_build_device=matvec_cuda_epq_build_device,
                use_epq_table=bool(matvec_cuda_use_epq_table),
                has_epq_table_device_build=bool(has_epq_table_device_build()),
            )

            epq_overbudget_action = "none"
            epq_overbudget_reason = "none"
            if bool(matvec_cuda_use_epq_table) and float(matvec_cuda_mem_hard_cap_gib) > 0.0:
                epq_peak_est_bytes = _estimate_epq_peak_bytes(int(drt.ncsf), int(norb_i))
                try:
                    epq_budget_free = _cuda_budget_free_bytes(cp, float(matvec_cuda_mem_hard_cap_gib))
                except Exception:
                    epq_budget_free = 0
                if (not epq_budget_free) or int(epq_peak_est_bytes) > int(float(epq_budget_free) * 0.85):
                    epq_overbudget_action, epq_overbudget_reason = _resolve_epq_overbudget_action(
                        matvec_cuda_dtype=str(matvec_cuda_dtype),
                        matvec_cuda_aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                        ncsf=int(drt.ncsf),
                        epq_table_forced=bool(epq_table_forced),
                        epq_streaming_mode=str(matvec_cuda_epq_streaming_mode),
                        has_epq_table_device_build=bool(has_epq_table_device_build()),
                    )
                    if str(epq_overbudget_action) == "streaming":
                        matvec_cuda_use_epq_table = True
                        matvec_cuda_epq_build_device = True
                        matvec_cuda_epq_streaming = True
                        if matvec_cuda_epq_stream_j_tile <= 0:
                            matvec_cuda_epq_stream_j_tile = int(matvec_cuda_j_tile)
                    elif str(epq_overbudget_action) == "disable_epq":
                        matvec_cuda_use_epq_table = False
                        matvec_cuda_epq_build_device = False
                        matvec_cuda_epq_streaming = False
                        if threads_apply_auto:
                            matvec_cuda_threads_apply = 64 if int(drt.ncsf) >= 1_000_000 else 32

            if int(matvec_cuda_epq_stream_j_tile) <= 0:
                matvec_cuda_epq_stream_j_tile = int(matvec_cuda_j_tile)

            _policy_apply = _apply_low_precision_and_workspace_policy_runtime(
                context="kernel(cuda)",
                dtype_mode=str(matvec_cuda_dtype),
                use_epq_table=bool(matvec_cuda_use_epq_table),
                aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                ncsf=int(drt.ncsf),
                eri_mat_present=bool(eri_mat_d is not None),
                enable_fp64_emulation=bool(matvec_cuda_enable_fp64_emulation),
                use_graph=bool(matvec_cuda_use_graph),
                apply_mode=str(matvec_cuda_apply_mode),
                apply_mode_forced=bool(apply_mode_forced),
                nops=int(nops),
                threads_enum=int(matvec_cuda_threads_enum),
                threads_g=int(matvec_cuda_threads_g),
                threads_w=int(matvec_cuda_threads_w),
                threads_apply=int(matvec_cuda_threads_apply),
                threads_enum_forced=bool(threads_enum_forced),
                threads_g_forced=bool(threads_g_forced),
                threads_apply_auto=bool(threads_apply_auto),
                max_g_mib=float(matvec_cuda_max_g_mib),
                mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                cache_csr_tiles_in=matvec_cuda_cache_csr_tiles,
                j_tile=int(matvec_cuda_j_tile),
                norb=int(norb_i),
                csr_capacity_mult=float(matvec_cuda_csr_capacity_mult),
                noepq_large_ncsf_uses_64=True,
            )
            if bool(_policy_apply["graph_disabled"]):
                warnings.warn("matvec_cuda_dtype float32/mixed: disabling CUDA Graph capture (float64-only path)")
            matvec_cuda_use_graph = bool(_policy_apply["use_graph"])
            matvec_cuda_apply_mode = str(_policy_apply["apply_mode"])
            matvec_cuda_threads_enum = int(_policy_apply["threads_enum"])
            matvec_cuda_threads_g = int(_policy_apply["threads_g"])
            matvec_cuda_threads_w = int(_policy_apply["threads_w"])
            matvec_cuda_threads_apply = int(_policy_apply["threads_apply"])
            matvec_cuda_max_g_mib = float(_policy_apply["max_g_mib"])
            matvec_cuda_cache_csr_tiles = _policy_apply["cache_csr_tiles"]

            if l_full_d is not None:
                # DF L_full path currently does not support CUDA Graph capture.
                if bool(matvec_cuda_use_graph):
                    warnings.warn("CUDA DF (L_full): disabling CUDA Graph capture (requires ERI_mat)")
                    matvec_cuda_use_graph = False

            # Mixed mode keeps a float32 low-precision workspace for throughput.
            matvec_cuda_aggregate_offdiag_main = bool(matvec_cuda_aggregate_offdiag)
            matvec_cuda_use_epq_table_main = bool(matvec_cuda_use_epq_table)
            matvec_cuda_epq_build_device_main = bool(matvec_cuda_epq_build_device)
            matvec_cuda_epq_streaming_main = bool(matvec_cuda_epq_streaming)
            matvec_cuda_epq_stream_j_tile_main = int(matvec_cuda_epq_stream_j_tile)
            want_max_g_bytes_main = int(matvec_cuda_max_g_mib * 1024 * 1024)
            want_max_g_bytes_low = int(matvec_cuda_max_g_mib * 1024 * 1024)
            mixed_guarded_requires_epq = bool(
                str(matvec_cuda_dtype) == "mixed"
                and bool(matvec_cuda_aggregate_offdiag)
                and int(getattr(drt, "ncsf", 0)) >= 1_000_000
            )
            if (
                str(matvec_cuda_dtype) == "mixed"
                and int(getattr(drt, "ncsf", 0)) >= 1_000_000
                and float(matvec_cuda_mem_hard_cap_gib) > 0.0
                and float(matvec_cuda_mem_hard_cap_gib) <= 12.0
            ):
                # Full-precision mixed hops are infrequent; keep auxiliary buffer pressure low
                # under tight VRAM caps, but keep EPQ policy aligned with the low-precision path.
                want_max_g_bytes_main = min(int(want_max_g_bytes_main), 128 * 1024 * 1024)
                want_max_g_bytes_low = min(int(want_max_g_bytes_low), 256 * 1024 * 1024)
            mixed_low_switch_target = "none"
            mixed_low_switch_reason = "none"

            if kprof is not None:
                kprof["matvec_cuda_target_ntasks"] = int(matvec_cuda_target_ntasks)
                kprof["matvec_cuda_j_tile_align"] = int(matvec_cuda_j_tile_align)
                kprof["matvec_cuda_j_tile"] = int(matvec_cuda_j_tile)
                kprof["matvec_cuda_csr_capacity_mult"] = float(matvec_cuda_csr_capacity_mult)
                kprof["matvec_cuda_cache_csr_tiles"] = bool(matvec_cuda_cache_csr_tiles)
                kprof["matvec_cuda_csr_host_cache"] = str(matvec_cuda_csr_host_cache_mode)
                kprof["matvec_cuda_csr_host_cache_budget_gib"] = float(matvec_cuda_csr_host_cache_budget_gib)
                kprof["matvec_cuda_csr_host_cache_min_ncsf"] = int(matvec_cuda_csr_host_cache_min_ncsf)
                if matvec_cuda_csr_pipeline_streams_value is None:
                    kprof["matvec_cuda_csr_pipeline_streams"] = str(matvec_cuda_csr_pipeline_streams_mode)
                else:
                    kprof["matvec_cuda_csr_pipeline_streams"] = int(matvec_cuda_csr_pipeline_streams_value)
                kprof["matvec_cuda_csr_pipeline_min_ncsf"] = int(matvec_cuda_csr_pipeline_min_ncsf)
                kprof["matvec_cuda_prefilter_trivial_tasks"] = str(matvec_cuda_prefilter_trivial_tasks_mode)
                kprof["matvec_cuda_prefilter_trivial_tasks_min_ncsf"] = int(
                    matvec_cuda_prefilter_trivial_tasks_min_ncsf
                )
                kprof["matvec_cuda_epq_build_nthreads"] = int(matvec_cuda_epq_build_nthreads)
                kprof["matvec_cuda_epq_build_device"] = bool(matvec_cuda_epq_build_device)
                kprof["matvec_cuda_epq_build_j_tile"] = int(matvec_cuda_epq_build_j_tile)
                kprof["matvec_cuda_epq_streaming"] = bool(matvec_cuda_epq_streaming)
                kprof["matvec_cuda_epq_stream_j_tile"] = int(matvec_cuda_epq_stream_j_tile)
                kprof["matvec_cuda_epq_stream_use_recompute"] = str(matvec_cuda_epq_stream_use_recompute)
                kprof["matvec_cuda_threads_enum"] = int(matvec_cuda_threads_enum)
                kprof["matvec_cuda_threads_g"] = int(matvec_cuda_threads_g)
                kprof["matvec_cuda_threads_w"] = int(matvec_cuda_threads_w)
                kprof["matvec_cuda_threads_apply"] = int(matvec_cuda_threads_apply)
                kprof["matvec_cuda_max_g_mib"] = float(matvec_cuda_max_g_mib)
                kprof["matvec_cuda_mem_hard_cap_gib"] = float(matvec_cuda_mem_hard_cap_gib)
                kprof["matvec_cuda_coalesce"] = bool(matvec_cuda_coalesce)
                kprof["matvec_cuda_include_diagonal_rs"] = bool(matvec_cuda_include_diagonal_rs)
                kprof["matvec_cuda_path_mode"] = str(matvec_cuda_path_mode)
                kprof["matvec_cuda_use_fused_hop"] = bool(matvec_cuda_use_fused_hop)
                kprof["matvec_cuda_fp32_coeff_data"] = bool(matvec_cuda_fp32_coeff_data)
                kprof["matvec_cuda_use_epq_table"] = bool(matvec_cuda_use_epq_table)
                kprof["matvec_cuda_aggregate_offdiag"] = bool(matvec_cuda_aggregate_offdiag)
                kprof["matvec_cuda_epq_overbudget_action"] = str(epq_overbudget_action)
                kprof["matvec_cuda_epq_overbudget_reason"] = str(epq_overbudget_reason)
                kprof["matvec_cuda_enable_fp64_emulation"] = bool(matvec_cuda_enable_fp64_emulation)
                kprof["matvec_cuda_gemm_backend"] = str(matvec_cuda_gemm_backend)
                if str(matvec_cuda_dtype) == "mixed":
                    kprof["matvec_cuda_gemm_backend_low"] = str(matvec_cuda_gemm_backend_low)
                kprof["matvec_cuda_emulation_strategy"] = str(matvec_cuda_emulation_strategy)
                kprof["matvec_cuda_cublas_workspace_cap_mb"] = int(matvec_cuda_cublas_workspace_cap_mb)
                kprof["matvec_cuda_apply_mode"] = str(matvec_cuda_apply_mode)
                kprof["matvec_cuda_dtype"] = str(matvec_cuda_dtype)
                if matvec_cuda_mixed_threshold is not None:
                    kprof["matvec_cuda_mixed_threshold"] = float(matvec_cuda_mixed_threshold)
                if str(matvec_cuda_dtype) == "mixed":
                    kprof["matvec_cuda_mixed_force_final_full_hop"] = bool(matvec_cuda_mixed_force_final_full_hop)
                    kprof["matvec_cuda_mixed_final_full_subspace_refresh"] = bool(
                        matvec_cuda_mixed_final_full_subspace_refresh
                    )
                    kprof["matvec_cuda_mixed_low_precision_max_iter"] = (
                        None
                        if matvec_cuda_mixed_low_precision_max_iter is None
                        else int(matvec_cuda_mixed_low_precision_max_iter)
                    )
                    kprof["matvec_cuda_mixed_main_use_epq_table"] = bool(matvec_cuda_use_epq_table_main)
                    kprof["matvec_cuda_mixed_main_epq_streaming"] = bool(matvec_cuda_epq_streaming_main)
                    kprof["matvec_cuda_mixed_main_max_g_bytes"] = int(want_max_g_bytes_main)
                kprof["matvec_cuda_use_graph"] = bool(matvec_cuda_use_graph)
                kprof["matvec_cuda_graph_warmup"] = bool(matvec_cuda_graph_warmup)
                kprof["matvec_cuda_make_hdiag_cpu"] = bool(matvec_cuda_make_hdiag_cpu)
                kprof["matvec_cuda_make_hdiag_cpu_ncsf_cutoff"] = int(matvec_cuda_make_hdiag_cpu_ncsf_cutoff)

            main_ws_dtype = "float32" if str(matvec_cuda_dtype) == "float32" else "float64"
            ws_cache_key = (ws_key, str(main_ws_dtype))
            def _init_cuda_ws(
                *,
                use_epq_table: bool,
                epq_build_device: bool,
                epq_streaming: bool,
                epq_stream_j_tile: int,
                epq_stream_use_recompute: bool | str,
                ws_dtype: str,
                use_graph: bool,
                gemm_backend: str,
                aggregate_offdiag_k: bool,
                max_g_bytes: int,
            ):
                ws_dtype_obj = cp.float32 if str(ws_dtype) == "float32" else cp.float64
                return GugaMatvecEriMatWorkspace(
                    drt,
                    drt_dev=drt_dev,
                    state_dev=state_dev,
                    eri_mat=eri_mat_d,
                    l_full=l_full_d,
                    h_eff=h_eff_d,
                    j_tile=matvec_cuda_j_tile,
                    csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                    cache_csr_tiles=bool(matvec_cuda_cache_csr_tiles),
                    csr_host_cache=str(matvec_cuda_csr_host_cache_mode),
                    csr_host_cache_budget_gib=float(matvec_cuda_csr_host_cache_budget_gib),
                    csr_host_cache_min_ncsf=int(matvec_cuda_csr_host_cache_min_ncsf),
                    csr_pipeline_streams=matvec_cuda_csr_pipeline_streams,
                    csr_pipeline_min_ncsf=int(matvec_cuda_csr_pipeline_min_ncsf),
                    prefilter_trivial_tasks=str(matvec_cuda_prefilter_trivial_tasks_mode),
                    prefilter_trivial_tasks_min_ncsf=int(matvec_cuda_prefilter_trivial_tasks_min_ncsf),
                    threads_enum=matvec_cuda_threads_enum,
                    threads_g=matvec_cuda_threads_g,
                    threads_w=matvec_cuda_threads_w,
                    threads_apply=matvec_cuda_threads_apply,
                    max_g_bytes=int(max_g_bytes),
                    coalesce=matvec_cuda_coalesce,
                    include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                    fuse_count_write=bool(matvec_cuda_fuse_count_write),
                    path_mode=str(matvec_cuda_path_mode),
                    use_fused_hop=bool(matvec_cuda_use_fused_hop),
                    fp32_coeff_data=bool(matvec_cuda_fp32_coeff_data),
                    use_epq_table=bool(use_epq_table),
                    aggregate_offdiag_k=bool(aggregate_offdiag_k),
                    offdiag_enable_fp64_emulation=bool(matvec_cuda_enable_fp64_emulation),
                    offdiag_emulation_strategy=str(matvec_cuda_emulation_strategy),
                    offdiag_cublas_workspace_cap_mb=int(matvec_cuda_cublas_workspace_cap_mb),
                    gemm_backend=str(gemm_backend),
                    apply_mode=str(matvec_cuda_apply_mode),
                    epq_build_nthreads=int(matvec_cuda_epq_build_nthreads),
                    epq_build_device=bool(epq_build_device),
                    epq_build_j_tile=int(matvec_cuda_epq_build_j_tile),
                    epq_streaming=bool(epq_streaming),
                    epq_stream_j_tile=int(epq_stream_j_tile),
                    epq_stream_use_recompute=epq_stream_use_recompute,
                    use_cuda_graph=bool(use_graph),
                    dtype=ws_dtype_obj,
                )
            def _collect_ws_rebuild_mismatches(
                ws,
                *,
                expected_dtype,
                j_tile,
                csr_capacity_mult,
                threads_enum,
                threads_g,
                threads_w,
                threads_apply,
                max_g_bytes,
                coalesce,
                include_diagonal_rs,
                fuse_count_write,
                fp32_coeff_data,
                path_mode,
                use_fused_hop,
                use_epq_table,
                aggregate_offdiag_k,
                l_full_d,
                gemm_backend,
                apply_mode,
                epq_build_device,
                epq_build_j_tile,
                epq_streaming,
                epq_stream_j_tile,
                epq_stream_use_recompute,
                cache_csr_tiles,
                csr_host_cache_mode,
                csr_host_cache_budget_gib,
                csr_host_cache_min_ncsf,
                prefilter_trivial_tasks_mode,
                prefilter_trivial_tasks_min_ncsf,
            ) -> list[str]:
                if ws is None:
                    return ["ws_none"]
                ws_path_mode = str(getattr(ws, "path_mode", "auto"))
                out: list[str] = []
                if np.dtype(getattr(ws, "dtype", np.float64)) != np.dtype(expected_dtype):
                    out.append("dtype")
                if int(getattr(ws, "j_tile", -1)) != int(j_tile):
                    out.append("j_tile")
                if float(getattr(ws, "csr_capacity_mult", -1.0)) != float(csr_capacity_mult):
                    out.append("csr_capacity_mult")
                if int(getattr(ws, "threads_enum", -1)) != int(threads_enum):
                    out.append("threads_enum")
                if int(getattr(ws, "threads_g", -1)) != int(threads_g):
                    out.append("threads_g")
                if int(getattr(ws, "threads_w", -1)) != int(threads_w):
                    out.append("threads_w")
                if int(getattr(ws, "threads_apply", -1)) != int(threads_apply):
                    out.append("threads_apply")
                if int(getattr(ws, "max_g_bytes", -1)) != int(max_g_bytes):
                    out.append("max_g_bytes")
                if bool(getattr(ws, "coalesce", False)) != bool(coalesce):
                    out.append("coalesce")
                if bool(getattr(ws, "include_diagonal_rs", False)) != bool(include_diagonal_rs):
                    out.append("include_diagonal_rs")
                if bool(getattr(ws, "fuse_count_write", False)) != bool(fuse_count_write):
                    out.append("fuse_count_write")
                if bool(getattr(ws, "fp32_coeff_data", False)) != bool(fp32_coeff_data):
                    out.append("fp32_coeff_data")
                if str(getattr(ws, "path_mode_requested", "auto")) != str(path_mode):
                    out.append("path_mode")
                if bool(getattr(ws, "use_fused_hop", True)) != bool(use_fused_hop):
                    out.append("use_fused_hop")
                if ws_path_mode != "fused_coo" and bool(getattr(ws, "use_epq_table", False)) != bool(use_epq_table):
                    out.append("use_epq_table")
                if (
                    ws_path_mode not in ("fused_coo", "fused_epq_hybrid")
                    and bool(getattr(ws, "aggregate_offdiag_k", False)) != bool(aggregate_offdiag_k)
                ):
                    out.append("aggregate_offdiag_k")
                if bool(getattr(ws, "l_full", None) is not None) != bool(l_full_d is not None):
                    out.append("l_full_presence")
                if (
                    l_full_d is not None
                    and int(getattr(ws, "naux", 0)) != int(getattr(l_full_d, "shape", (0, 0))[1])
                ):
                    out.append("naux")
                if str(getattr(ws, "gemm_backend", "")) != str(gemm_backend):
                    out.append("gemm_backend")
                if str(getattr(ws, "apply_mode", "")) != str(apply_mode):
                    out.append("apply_mode")
                if bool(getattr(ws, "epq_build_device", False)) != bool(epq_build_device):
                    out.append("epq_build_device")
                if int(getattr(ws, "epq_build_j_tile", 0)) != int(epq_build_j_tile):
                    out.append("epq_build_j_tile")
                if bool(getattr(ws, "epq_streaming", False)) != bool(epq_streaming):
                    out.append("epq_streaming")
                if int(getattr(ws, "epq_stream_j_tile", 0)) != int(epq_stream_j_tile):
                    out.append("epq_stream_j_tile")
                if str(getattr(ws, "epq_stream_use_recompute", "auto")) != str(epq_stream_use_recompute):
                    out.append("epq_stream_use_recompute")
                if bool(getattr(ws, "cache_csr_tiles", False)) != bool(cache_csr_tiles):
                    out.append("cache_csr_tiles")
                if str(getattr(ws, "csr_host_cache_mode", "off")) != str(csr_host_cache_mode):
                    out.append("csr_host_cache_mode")
                if float(getattr(ws, "csr_host_cache_budget_gib", -1.0)) != float(csr_host_cache_budget_gib):
                    out.append("csr_host_cache_budget_gib")
                if int(getattr(ws, "csr_host_cache_min_ncsf", -1)) != int(csr_host_cache_min_ncsf):
                    out.append("csr_host_cache_min_ncsf")
                if str(getattr(ws, "prefilter_trivial_tasks_mode", "off")) != str(prefilter_trivial_tasks_mode):
                    out.append("prefilter_trivial_tasks_mode")
                if int(getattr(ws, "prefilter_trivial_tasks_min_ncsf", -1)) != int(prefilter_trivial_tasks_min_ncsf):
                    out.append("prefilter_trivial_tasks_min_ncsf")
                return out
            cuda_ws = self._matvec_cuda_ws_cache_get(ws_cache_key)
            main_ws_needs_rebuild = self._ws_needs_rebuild(
                cuda_ws,
                expected_dtype=np.float32 if str(main_ws_dtype) == "float32" else np.float64,
                j_tile=matvec_cuda_j_tile,
                csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                threads_enum=matvec_cuda_threads_enum,
                threads_g=matvec_cuda_threads_g,
                threads_w=matvec_cuda_threads_w,
                threads_apply=matvec_cuda_threads_apply,
                max_g_bytes=want_max_g_bytes_main,
                coalesce=matvec_cuda_coalesce,
                include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                fuse_count_write=matvec_cuda_fuse_count_write,
                fp32_coeff_data=matvec_cuda_fp32_coeff_data,
                path_mode=matvec_cuda_path_mode,
                use_fused_hop=matvec_cuda_use_fused_hop,
                use_epq_table=matvec_cuda_use_epq_table_main,
                aggregate_offdiag_k=matvec_cuda_aggregate_offdiag_main,
                l_full_d=l_full_d,
                enable_fp64_emulation=matvec_cuda_enable_fp64_emulation,
                gemm_backend=matvec_cuda_gemm_backend,
                emulation_strategy=matvec_cuda_emulation_strategy,
                cublas_workspace_cap_mb=matvec_cuda_cublas_workspace_cap_mb,
                apply_mode=matvec_cuda_apply_mode,
                epq_build_device=matvec_cuda_epq_build_device_main,
                epq_build_j_tile=matvec_cuda_epq_build_j_tile,
                epq_streaming=matvec_cuda_epq_streaming_main,
                epq_stream_j_tile=matvec_cuda_epq_stream_j_tile_main,
                epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                cache_csr_tiles=matvec_cuda_cache_csr_tiles,
                csr_host_cache_mode=matvec_cuda_csr_host_cache_mode,
                csr_host_cache_budget_gib=matvec_cuda_csr_host_cache_budget_gib,
                csr_host_cache_min_ncsf=matvec_cuda_csr_host_cache_min_ncsf,
                csr_pipeline_streams_mode=matvec_cuda_csr_pipeline_streams_mode,
                csr_pipeline_streams_value=matvec_cuda_csr_pipeline_streams_value,
                csr_pipeline_min_ncsf=matvec_cuda_csr_pipeline_min_ncsf,
                prefilter_trivial_tasks_mode=matvec_cuda_prefilter_trivial_tasks_mode,
                prefilter_trivial_tasks_min_ncsf=matvec_cuda_prefilter_trivial_tasks_min_ncsf,
            )
            if main_ws_needs_rebuild:
                if kprof is not None:
                    kprof["matvec_cuda_ws_reused"] = False
                    kprof["matvec_cuda_ws_rebuild_mismatches"] = _collect_ws_rebuild_mismatches(
                        cuda_ws,
                        expected_dtype=np.float32 if str(main_ws_dtype) == "float32" else np.float64,
                        j_tile=matvec_cuda_j_tile,
                        csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                        threads_enum=matvec_cuda_threads_enum,
                        threads_g=matvec_cuda_threads_g,
                        threads_w=matvec_cuda_threads_w,
                        threads_apply=matvec_cuda_threads_apply,
                        max_g_bytes=want_max_g_bytes_main,
                        coalesce=matvec_cuda_coalesce,
                        include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                        fuse_count_write=matvec_cuda_fuse_count_write,
                        fp32_coeff_data=matvec_cuda_fp32_coeff_data,
                        path_mode=matvec_cuda_path_mode,
                        use_fused_hop=matvec_cuda_use_fused_hop,
                        use_epq_table=matvec_cuda_use_epq_table_main,
                        aggregate_offdiag_k=matvec_cuda_aggregate_offdiag_main,
                        l_full_d=l_full_d,
                        gemm_backend=matvec_cuda_gemm_backend,
                        apply_mode=matvec_cuda_apply_mode,
                        epq_build_device=matvec_cuda_epq_build_device_main,
                        epq_build_j_tile=matvec_cuda_epq_build_j_tile,
                        epq_streaming=matvec_cuda_epq_streaming_main,
                        epq_stream_j_tile=matvec_cuda_epq_stream_j_tile_main,
                        epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                        cache_csr_tiles=matvec_cuda_cache_csr_tiles,
                        csr_host_cache_mode=matvec_cuda_csr_host_cache_mode,
                        csr_host_cache_budget_gib=matvec_cuda_csr_host_cache_budget_gib,
                        csr_host_cache_min_ncsf=matvec_cuda_csr_host_cache_min_ncsf,
                        prefilter_trivial_tasks_mode=matvec_cuda_prefilter_trivial_tasks_mode,
                        prefilter_trivial_tasks_min_ncsf=matvec_cuda_prefilter_trivial_tasks_min_ncsf,
                    )
                t_ws0 = time.perf_counter() if kprof is not None else None
                cuda_ws = None

                try:
                    cuda_ws = _init_cuda_ws(
                        use_epq_table=bool(matvec_cuda_use_epq_table_main),
                        epq_build_device=bool(matvec_cuda_epq_build_device_main),
                        epq_streaming=bool(matvec_cuda_epq_streaming_main),
                        epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_main),
                        epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                        ws_dtype=str(main_ws_dtype),
                        use_graph=bool(matvec_cuda_use_graph),
                        gemm_backend=str(matvec_cuda_gemm_backend),
                        aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag_main),
                        max_g_bytes=int(want_max_g_bytes_main),
                    )
                except Exception as e:
                    # If epq_table was enabled automatically but the build ran out of memory, fall back
                    # to the CSR-based path instead of hard failing. This can be substantially slower.
                    msg = str(e).lower()
                    oom = "out of memory" in msg or "alloc" in msg or "memoryerror" in msg
                    streaming_explicit_main = str(matvec_cuda_epq_streaming_mode) in ("on", "manual")
                    can_stream_fallback_main = bool(
                        bool(matvec_cuda_use_epq_table_main)
                        and (not bool(matvec_cuda_epq_streaming_main))
                        and str(matvec_cuda_epq_streaming_mode) != "off"
                        and has_epq_table_device_build()
                    )
                    can_noepq_fallback_main = bool(
                        eri_mat_d is not None
                        and bool(matvec_cuda_aggregate_offdiag_main)
                    )
                    guarded_mixed_requires_epq_main = bool(
                        str(matvec_cuda_dtype) == "mixed"
                        and bool(matvec_cuda_aggregate_offdiag_main)
                        and int(drt.ncsf) >= 1_000_000
                    )
                    stream_required_main = bool(
                        guarded_mixed_requires_epq_main or (not bool(can_noepq_fallback_main))
                    )
                    if oom and can_stream_fallback_main and (streaming_explicit_main or stream_required_main):
                        warnings.warn(
                            "CUDA matvec: EPQ table materialization failed under memory cap; falling back to EPQ streaming."
                        )
                        matvec_cuda_use_epq_table_main = True
                        matvec_cuda_epq_build_device_main = True
                        matvec_cuda_epq_streaming_main = True
                        if int(matvec_cuda_epq_stream_j_tile_main) <= 0:
                            matvec_cuda_epq_stream_j_tile_main = int(matvec_cuda_j_tile)
                        if kprof is not None:
                            kprof["matvec_cuda_epq_streaming_oom_fallback"] = True
                            kprof["matvec_cuda_epq_streaming_oom_fallback_reason"] = (
                                "explicit_streaming_mode"
                                if bool(streaming_explicit_main)
                                else "stream_required_no_noepq_path"
                            )
                        cuda_ws = _init_cuda_ws(
                            use_epq_table=True,
                            epq_build_device=True,
                            epq_streaming=True,
                            epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_main),
                            epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                            ws_dtype=str(main_ws_dtype),
                            use_graph=bool(matvec_cuda_use_graph),
                            gemm_backend=str(matvec_cuda_gemm_backend),
                            aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag_main),
                            max_g_bytes=int(want_max_g_bytes_main),
                        )
                    elif oom and can_noepq_fallback_main and (not bool(guarded_mixed_requires_epq_main)):
                        warnings.warn(
                            "CUDA matvec: epq_table build failed (likely OOM); falling back to no-EPQ CSR path."
                        )
                        matvec_cuda_use_epq_table_main = False
                        matvec_cuda_epq_build_device_main = False
                        matvec_cuda_epq_streaming_main = False
                        if threads_apply_auto:
                            matvec_cuda_threads_apply = 64 if int(drt.ncsf) >= 1_000_000 else 32
                        if kprof is not None:
                            kprof["matvec_cuda_noepq_oom_fallback_main"] = True
                        cuda_ws = _init_cuda_ws(
                            use_epq_table=False,
                            epq_build_device=False,
                            epq_streaming=False,
                            epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_main),
                            epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                            ws_dtype=str(main_ws_dtype),
                            use_graph=bool(matvec_cuda_use_graph),
                            gemm_backend=str(matvec_cuda_gemm_backend),
                            aggregate_offdiag_k=True,
                            max_g_bytes=int(want_max_g_bytes_main),
                        )
                    else:
                        raise
                if cuda_ws is None:  # pragma: no cover
                    raise RuntimeError("internal error: failed to initialize CUDA matvec workspace")
                self._matvec_cuda_ws_cache_put(
                    ws_cache_key,
                    cuda_ws,
                    keep_keys=(ws_cache_key,),
                )
                if kprof is not None and t_ws0 is not None:
                    kprof["matvec_cuda_ws_init_s"] = time.perf_counter() - t_ws0
                    try:
                        kprof["matvec_cuda_epq_table_build_s"] = float(getattr(cuda_ws, "epq_table_build_s", 0.0))
                    except Exception:  # pragma: no cover
                        pass
            else:
                if kprof is not None:
                    kprof["matvec_cuda_ws_reused"] = True
                ws_dtype_obj = np.dtype(getattr(cuda_ws, "dtype", np.float64))
                cuda_ws.eri_mat = None if eri_mat_d is None else cp.ascontiguousarray(cp.asarray(eri_mat_d, dtype=ws_dtype_obj))
                cuda_ws.l_full = None if l_full_d is None else cp.ascontiguousarray(cp.asarray(l_full_d, dtype=ws_dtype_obj))
                h_eff_flat_new = cuda_ws._as_h_eff_flat(h_eff_d)
                if getattr(cuda_ws, "h_eff_flat", None) is None or tuple(getattr(cuda_ws.h_eff_flat, "shape", ())) != tuple(
                    getattr(h_eff_flat_new, "shape", ())
                ):
                    cuda_ws.h_eff_flat = h_eff_flat_new
                    if getattr(cuda_ws, "_cuda_graph", None) is not None:
                        # Graph capture pointers depend on stable `h_eff_flat`; fall back to
                        # re-capturing if the buffer had to be replaced.
                        cuda_ws._cuda_graph = None
                        cuda_ws._cuda_graph_x = None
                        cuda_ws._cuda_graph_y = None
                else:
                    cp.copyto(cuda_ws.h_eff_flat, h_eff_flat_new)

                # Diagonal-rs contribution depends on `eri_diag_t` extracted from `eri_mat`.
                # Invalidate it whenever the Hamiltonian changes.
                cuda_ws._eri_diag_t = None

                # Keep `eri_mat_t` pointer-stable for CUDA Graph reuse.
                if eri_mat_d is not None:
                    if getattr(cuda_ws, "_eri_mat_t", None) is not None:
                        cp.copyto(cuda_ws._eri_mat_t, cuda_ws.eri_mat.T)
                    elif getattr(cuda_ws, "_cuda_graph", None) is not None:
                        cuda_ws._cuda_graph = None
                        cuda_ws._cuda_graph_x = None
                        cuda_ws._cuda_graph_y = None
                else:
                    cuda_ws._eri_mat_t = None
                    if getattr(cuda_ws, "_cuda_graph", None) is not None:
                        cuda_ws._cuda_graph = None
                        cuda_ws._cuda_graph_x = None
                        cuda_ws._cuda_graph_y = None

                # If a CUDA Graph exists, its diagonal-rs contribution uses cached g_diag buffers.
                # Refresh them in place when the Hamiltonian changes.
                if (
                    bool(matvec_cuda_use_graph)
                    and getattr(cuda_ws, "_cuda_graph", None) is not None
                    and bool(getattr(cuda_ws, "include_diagonal_rs", False))
                ):
                    cuda_ws._build_diag_g_cache()

                cuda_ws.use_cuda_graph = bool(matvec_cuda_use_graph)
            if kprof is not None:
                kprof["matvec_cuda_use_epq_table_effective"] = bool(getattr(cuda_ws, "use_epq_table", False))
                kprof["matvec_cuda_epq_streaming_effective"] = bool(getattr(cuda_ws, "epq_streaming", False))
                kprof["matvec_cuda_path_mode_effective"] = str(getattr(cuda_ws, "path_mode", "auto"))
                kprof["matvec_cuda_path_fallback_reason"] = str(
                    getattr(cuda_ws, "path_mode_fallback_reason", "") or ""
                )
                if str(matvec_cuda_dtype) == "mixed":
                    kprof["matvec_cuda_mixed_main_use_epq_table"] = bool(getattr(cuda_ws, "use_epq_table", False))
                    kprof["matvec_cuda_mixed_main_epq_streaming"] = bool(getattr(cuda_ws, "epq_streaming", False))
                    kprof["matvec_cuda_mixed_main_max_g_bytes"] = int(getattr(cuda_ws, "max_g_bytes", 0))

            cuda_ws_low = None
            if str(matvec_cuda_dtype) == "mixed":
                matvec_cuda_use_epq_table_low = bool(getattr(cuda_ws, "use_epq_table", False))
                matvec_cuda_epq_build_device_low = bool(getattr(cuda_ws, "epq_build_device", False))
                matvec_cuda_epq_streaming_low = bool(getattr(cuda_ws, "epq_streaming", False))
                matvec_cuda_epq_stream_j_tile_low = int(
                    getattr(cuda_ws, "epq_stream_j_tile", int(matvec_cuda_epq_stream_j_tile_main))
                )
                if bool(matvec_cuda_epq_streaming_low):
                    mixed_low_switch_target = "streaming"
                    if str(epq_overbudget_action) == "streaming":
                        mixed_low_switch_reason = str(epq_overbudget_reason)
                    elif str(matvec_cuda_epq_streaming_mode) in ("on", "manual"):
                        mixed_low_switch_reason = "explicit_streaming_mode"
                    else:
                        mixed_low_switch_reason = "streaming_enabled"
                elif not bool(matvec_cuda_use_epq_table_low):
                    mixed_low_switch_target = "no_epq"
                    if str(epq_overbudget_action) == "disable_epq":
                        mixed_low_switch_reason = str(epq_overbudget_reason)
                    else:
                        mixed_low_switch_reason = "no_epq_policy"
                ws_cache_key_low = (ws_key, "float32")
                cuda_ws_low = self._matvec_cuda_ws_cache_get(ws_cache_key_low)
                low_ws_needs_rebuild = self._ws_needs_rebuild(
                    cuda_ws_low,
                    expected_dtype=np.float32,
                    j_tile=matvec_cuda_j_tile,
                    csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                    threads_enum=matvec_cuda_threads_enum,
                    threads_g=matvec_cuda_threads_g,
                    threads_w=matvec_cuda_threads_w,
                    threads_apply=matvec_cuda_threads_apply,
                    max_g_bytes=want_max_g_bytes_low,
                    coalesce=matvec_cuda_coalesce,
                    include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                    fuse_count_write=matvec_cuda_fuse_count_write,
                    fp32_coeff_data=matvec_cuda_fp32_coeff_data,
                    path_mode=matvec_cuda_path_mode,
                    use_fused_hop=matvec_cuda_use_fused_hop,
                    use_epq_table=matvec_cuda_use_epq_table_low,
                    aggregate_offdiag_k=matvec_cuda_aggregate_offdiag,
                    l_full_d=l_full_d,
                    enable_fp64_emulation=False,
                    gemm_backend=matvec_cuda_gemm_backend_low,
                    emulation_strategy=matvec_cuda_emulation_strategy,
                    cublas_workspace_cap_mb=matvec_cuda_cublas_workspace_cap_mb,
                    apply_mode=matvec_cuda_apply_mode,
                    epq_build_device=matvec_cuda_epq_build_device_low,
                    epq_build_j_tile=matvec_cuda_epq_build_j_tile,
                    epq_streaming=matvec_cuda_epq_streaming_low,
                    epq_stream_j_tile=matvec_cuda_epq_stream_j_tile_low,
                    epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                    cache_csr_tiles=matvec_cuda_cache_csr_tiles,
                    csr_host_cache_mode=matvec_cuda_csr_host_cache_mode,
                    csr_host_cache_budget_gib=matvec_cuda_csr_host_cache_budget_gib,
                    csr_host_cache_min_ncsf=matvec_cuda_csr_host_cache_min_ncsf,
                    csr_pipeline_streams_mode=matvec_cuda_csr_pipeline_streams_mode,
                    csr_pipeline_streams_value=matvec_cuda_csr_pipeline_streams_value,
                    csr_pipeline_min_ncsf=matvec_cuda_csr_pipeline_min_ncsf,
                    prefilter_trivial_tasks_mode=matvec_cuda_prefilter_trivial_tasks_mode,
                    prefilter_trivial_tasks_min_ncsf=matvec_cuda_prefilter_trivial_tasks_min_ncsf,
                )
                if low_ws_needs_rebuild:
                    if kprof is not None:
                        kprof["matvec_cuda_ws_low_reused"] = False
                    try:
                        cuda_ws_low = _init_cuda_ws(
                            use_epq_table=bool(matvec_cuda_use_epq_table_low),
                            epq_build_device=bool(matvec_cuda_epq_build_device_low),
                            epq_streaming=bool(matvec_cuda_epq_streaming_low),
                            epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_low),
                            epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                            ws_dtype="float32",
                            use_graph=False,
                            gemm_backend=str(matvec_cuda_gemm_backend_low),
                            aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag),
                            max_g_bytes=int(want_max_g_bytes_low),
                        )
                    except Exception as e:
                        msg = str(e).lower()
                        oom = "out of memory" in msg or "alloc" in msg or "memoryerror" in msg
                        can_stream_fallback_low = bool(
                            bool(matvec_cuda_use_epq_table_low)
                            and (not bool(matvec_cuda_epq_streaming_low))
                            and str(matvec_cuda_epq_streaming_mode) != "off"
                            and has_epq_table_device_build()
                        )
                        can_noepq_fallback_low = bool(eri_mat_d is not None and bool(matvec_cuda_aggregate_offdiag))
                        fallback_action, fallback_reason = _resolve_mixed_low_workspace_oom_fallback(
                            can_stream_fallback=bool(can_stream_fallback_low),
                            can_noepq_fallback=bool(can_noepq_fallback_low),
                            guarded_requires_epq=bool(mixed_guarded_requires_epq),
                        )
                        if oom and str(fallback_action) == "no_epq":
                            warnings.warn(
                                "CUDA matvec (mixed low workspace): EPQ table materialization failed under memory cap; "
                                "falling back to no-EPQ low workspace."
                            )
                            matvec_cuda_use_epq_table_low = False
                            matvec_cuda_epq_build_device_low = False
                            matvec_cuda_epq_streaming_low = False
                            mixed_low_switch_target = "no_epq"
                            mixed_low_switch_reason = str(fallback_reason)
                            if kprof is not None:
                                kprof["matvec_cuda_noepq_oom_fallback_low"] = True
                            cuda_ws_low = _init_cuda_ws(
                                use_epq_table=False,
                                epq_build_device=False,
                                epq_streaming=False,
                                epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_low),
                                epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                                ws_dtype="float32",
                                use_graph=False,
                                gemm_backend=str(matvec_cuda_gemm_backend_low),
                                aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag),
                                max_g_bytes=int(want_max_g_bytes_low),
                            )
                        elif oom and str(fallback_action) == "streaming":
                            warnings.warn(
                                "CUDA matvec (mixed low workspace): EPQ table materialization failed under memory cap; "
                                "falling back to EPQ streaming."
                            )
                            matvec_cuda_use_epq_table_low = True
                            matvec_cuda_epq_build_device_low = True
                            matvec_cuda_epq_streaming_low = True
                            if int(matvec_cuda_epq_stream_j_tile_low) <= 0:
                                matvec_cuda_epq_stream_j_tile_low = int(matvec_cuda_j_tile)
                            mixed_low_switch_target = "streaming"
                            mixed_low_switch_reason = str(fallback_reason)
                            if kprof is not None:
                                kprof["matvec_cuda_epq_streaming_oom_fallback_low"] = True
                            cuda_ws_low = _init_cuda_ws(
                                use_epq_table=True,
                                epq_build_device=True,
                                epq_streaming=True,
                                epq_stream_j_tile=int(matvec_cuda_epq_stream_j_tile_low),
                                epq_stream_use_recompute=matvec_cuda_epq_stream_use_recompute,
                                ws_dtype="float32",
                                use_graph=False,
                                gemm_backend=str(matvec_cuda_gemm_backend_low),
                                aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag),
                                max_g_bytes=int(want_max_g_bytes_low),
                            )
                        else:
                            if oom:
                                mixed_low_switch_reason = str(fallback_reason)
                            raise
                    self._matvec_cuda_ws_cache_put(
                        ws_cache_key_low,
                        cuda_ws_low,
                        keep_keys=(ws_cache_key, ws_cache_key_low),
                    )
                else:
                    if kprof is not None:
                        kprof["matvec_cuda_ws_low_reused"] = True
                    ws_dtype_obj_low = np.dtype(getattr(cuda_ws_low, "dtype", np.float32))
                    cuda_ws_low.eri_mat = None if eri_mat_d is None else cp.ascontiguousarray(cp.asarray(eri_mat_d, dtype=ws_dtype_obj_low))
                    cuda_ws_low.l_full = None if l_full_d is None else cp.ascontiguousarray(cp.asarray(l_full_d, dtype=ws_dtype_obj_low))
                    h_eff_flat_new_low = cuda_ws_low._as_h_eff_flat(h_eff_d)
                    if getattr(cuda_ws_low, "h_eff_flat", None) is None or tuple(getattr(cuda_ws_low.h_eff_flat, "shape", ())) != tuple(
                        getattr(h_eff_flat_new_low, "shape", ())
                    ):
                        cuda_ws_low.h_eff_flat = h_eff_flat_new_low
                    else:
                        cp.copyto(cuda_ws_low.h_eff_flat, h_eff_flat_new_low)
                    cuda_ws_low._eri_diag_t = None
                    if eri_mat_d is not None and getattr(cuda_ws_low, "_eri_mat_t", None) is not None:
                        cp.copyto(cuda_ws_low._eri_mat_t, cuda_ws_low.eri_mat.T)
                    elif eri_mat_d is None:
                        cuda_ws_low._eri_mat_t = None
                    cuda_ws_low.use_cuda_graph = False
                if kprof is not None:
                    kprof["matvec_cuda_mixed_low_use_epq_table_effective"] = bool(
                        getattr(cuda_ws_low, "use_epq_table", False)
                    )
                    kprof["matvec_cuda_mixed_low_epq_streaming_effective"] = bool(
                        getattr(cuda_ws_low, "epq_streaming", False)
                    )
                    kprof["matvec_cuda_mixed_low_max_g_bytes"] = int(getattr(cuda_ws_low, "max_g_bytes", 0))
                    kprof["matvec_cuda_mixed_low_switch_target"] = str(mixed_low_switch_target)
                    kprof["matvec_cuda_mixed_low_switch_reason"] = str(mixed_low_switch_reason)
                mixed_shared_epq = False
                mixed_shared_epq_reason = "not_attempted"
                try:
                    main_epq = getattr(cuda_ws, "_epq_table", None)
                    low_epq = getattr(cuda_ws_low, "_epq_table", None)
                    if main_epq is None or low_epq is None:
                        mixed_shared_epq_reason = "missing_materialized_table"
                    elif bool(getattr(cuda_ws, "epq_streaming", False)) or bool(getattr(cuda_ws_low, "epq_streaming", False)):
                        mixed_shared_epq_reason = "streaming_active"
                    else:
                        low_data_dtype = np.dtype(getattr(low_epq[3], "dtype", np.float64))
                        main_dtype = np.dtype(getattr(cuda_ws, "dtype", np.float64))
                        if low_data_dtype != np.dtype(np.float32):
                            mixed_shared_epq_reason = "low_table_not_fp32"
                        elif main_dtype != np.dtype(np.float64):
                            mixed_shared_epq_reason = "main_not_fp64"
                        else:
                            cuda_ws._epq_table = low_epq
                            mixed_shared_epq = True
                            mixed_shared_epq_reason = "shared_fp32_table"
                except Exception as _share_exc:
                    mixed_shared_epq = False
                    mixed_shared_epq_reason = f"share_failed:{_share_exc}"
                if kprof is not None:
                    kprof["matvec_cuda_mixed_shared_epq_table"] = bool(mixed_shared_epq)
                    kprof["matvec_cuda_mixed_shared_epq_table_reason"] = str(mixed_shared_epq_reason)

            cuda_matvec_ws = cuda_ws
            cuda_cp = cp

            if matvec_cuda_use_graph:
                t_graph0 = time.perf_counter() if kprof is not None else None
                try:
                    cuda_ws.enable_cuda_graph(warmup=bool(matvec_cuda_graph_warmup))
                    if kprof is not None:
                        kprof["matvec_cuda_graph_enabled"] = True
                except Exception as e:
                    # Graph capture is optional; fall back to the regular hop path if unsupported.
                    if kprof is not None:
                        kprof["matvec_cuda_graph_enabled"] = False
                        kprof["matvec_cuda_graph_error"] = str(e)
                finally:
                    if kprof is not None and t_graph0 is not None:
                        kprof["matvec_cuda_graph_capture_s"] = time.perf_counter() - t_graph0
        elif matvec_backend in ("cuda_fixed_ell", "cuda_ell"):
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception as e:  # pragma: no cover
                raise RuntimeError("matvec_backend='cuda_fixed_ell' requires CuPy") from e

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                GugaMatvecFixedEllWorkspace,
                build_h_ell_from_row_oracle_host,
                has_cuda_ext,
            )

            if not has_cuda_ext():
                raise RuntimeError(
                    "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
                )

            max_ncsf = int(kwargs.pop("matvec_cuda_fixed_ell_max_ncsf", getattr(self, "matvec_cuda_fixed_ell_max_ncsf", 50_000)))
            max_width = int(kwargs.pop("matvec_cuda_fixed_ell_max_width", getattr(self, "matvec_cuda_fixed_ell_max_width", 256)))
            row_oracle = kwargs.pop(
                "matvec_cuda_fixed_ell_row_oracle",
                getattr(self, "matvec_cuda_fixed_ell_row_oracle", "sparse"),
            )
            threads_spmv = int(
                kwargs.pop(
                    "matvec_cuda_fixed_ell_threads_spmv",
                    getattr(self, "matvec_cuda_fixed_ell_threads_spmv", 128),
                )
            )

            ncsf_i = int(drt.ncsf)
            if ncsf_i > max_ncsf:
                raise ValueError(f"cuda_fixed_ell disabled: ncsf={ncsf_i} exceeds max_ncsf={max_ncsf}")

            if kprof is not None:
                kprof["matvec_cuda_fixed_ell_max_ncsf"] = int(max_ncsf)
                kprof["matvec_cuda_fixed_ell_max_width"] = int(max_width)
                kprof["matvec_cuda_fixed_ell_row_oracle"] = str(row_oracle)
                kprof["matvec_cuda_fixed_ell_threads_spmv"] = int(threads_spmv)

            if kprof is not None:
                t0 = time.perf_counter()
            col_h, val_h, stats = build_h_ell_from_row_oracle_host(drt, h1e, eri, row_oracle=row_oracle)
            width = int(stats.get("width_max", 0))
            if width > max_width:
                raise ValueError(f"cuda_fixed_ell disabled: ELL width={width} exceeds max_width={max_width}")

            col_d = cp.asarray(col_h, dtype=cp.int32)
            val_d = cp.asarray(val_h, dtype=cp.float64)
            cuda_fixed_ell_ws = GugaMatvecFixedEllWorkspace(col_d, val_d, threads_spmv=int(threads_spmv))

            if kprof is not None:
                cp.cuda.get_current_stream().synchronize()
                kprof["matvec_cuda_fixed_ell_build_s"] = time.perf_counter() - t0
                kprof["matvec_cuda_fixed_ell_nnz_total"] = int(stats.get("nnz_total", 0))
                kprof["matvec_cuda_fixed_ell_width"] = int(width)

            cuda_cp = cp
        elif matvec_backend in ("cuda_fixed_sell", "cuda_sell"):
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception as e:  # pragma: no cover
                raise RuntimeError("matvec_backend='cuda_fixed_sell' requires CuPy") from e

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                GugaMatvecFixedSellWorkspace,
                build_h_sell_from_row_oracle_host,
                has_cuda_ext,
            )

            if not has_cuda_ext():
                raise RuntimeError(
                    "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
                )

            max_ncsf = int(
                kwargs.pop("matvec_cuda_fixed_sell_max_ncsf", getattr(self, "matvec_cuda_fixed_sell_max_ncsf", 50_000))
            )
            max_nelems = int(
                kwargs.pop(
                    "matvec_cuda_fixed_sell_max_nelems",
                    getattr(self, "matvec_cuda_fixed_sell_max_nelems", 20_000_000),
                )
            )
            slice_height = int(
                kwargs.pop(
                    "matvec_cuda_fixed_sell_slice_height",
                    getattr(self, "matvec_cuda_fixed_sell_slice_height", 32),
                )
            )
            row_oracle = kwargs.pop(
                "matvec_cuda_fixed_sell_row_oracle",
                getattr(self, "matvec_cuda_fixed_sell_row_oracle", "sparse"),
            )
            threads_spmv = int(
                kwargs.pop(
                    "matvec_cuda_fixed_sell_threads_spmv",
                    getattr(self, "matvec_cuda_fixed_sell_threads_spmv", 128),
                )
            )
            threads_spmm = int(
                kwargs.pop(
                    "matvec_cuda_fixed_sell_threads_spmm",
                    getattr(self, "matvec_cuda_fixed_sell_threads_spmm", 128),
                )
            )

            if slice_height <= 0:
                raise ValueError("cuda_fixed_sell requires slice_height > 0")

            ncsf_i = int(drt.ncsf)
            if ncsf_i > max_ncsf:
                raise ValueError(f"cuda_fixed_sell disabled: ncsf={ncsf_i} exceeds max_ncsf={max_ncsf}")

            if kprof is not None:
                kprof["matvec_cuda_fixed_sell_max_ncsf"] = int(max_ncsf)
                kprof["matvec_cuda_fixed_sell_max_nelems"] = int(max_nelems)
                kprof["matvec_cuda_fixed_sell_slice_height"] = int(slice_height)
                kprof["matvec_cuda_fixed_sell_row_oracle"] = str(row_oracle)
                kprof["matvec_cuda_fixed_sell_threads_spmv"] = int(threads_spmv)
                kprof["matvec_cuda_fixed_sell_threads_spmm"] = int(threads_spmm)

            if kprof is not None:
                t0 = time.perf_counter()
            ptr_h, width_h, col_h, val_h, stats = build_h_sell_from_row_oracle_host(
                drt, h1e, eri, slice_height=int(slice_height), row_oracle=row_oracle
            )
            nelems = int(stats.get("nelems", int(col_h.size)))
            if nelems > max_nelems:
                raise ValueError(f"cuda_fixed_sell disabled: nelems={nelems} exceeds max_nelems={max_nelems}")

            ptr_d = cp.asarray(ptr_h, dtype=cp.int64)
            width_d = cp.asarray(width_h, dtype=cp.int32)
            col_d = cp.asarray(col_h, dtype=cp.int32)
            val_d = cp.asarray(val_h, dtype=cp.float64)
            cuda_fixed_sell_ws = GugaMatvecFixedSellWorkspace(
                ptr_d,
                width_d,
                col_d,
                val_d,
                nrows=int(ncsf_i),
                slice_height=int(slice_height),
                threads_spmv=int(threads_spmv),
                threads_spmm=int(threads_spmm),
            )

            if kprof is not None:
                cp.cuda.get_current_stream().synchronize()
                kprof["matvec_cuda_fixed_sell_build_s"] = time.perf_counter() - t0
                kprof["matvec_cuda_fixed_sell_nnz_total"] = int(stats.get("nnz_total", 0))
                kprof["matvec_cuda_fixed_sell_width_max"] = int(stats.get("width_max", 0))
                kprof["matvec_cuda_fixed_sell_nelems"] = int(nelems)

            cuda_cp = cp

        hdiag = None
        hdiag_d = None
        if kprof is not None:
            t0 = time.perf_counter()
            try:
                kprof["eri_type"] = f"{type(eri).__module__}.{type(eri).__qualname__}"
            except Exception:  # pragma: no cover
                kprof["eri_type"] = str(type(eri))
            kprof["eri_is_device_df"] = bool(isinstance(eri, DeviceDFMOIntegrals))
            kprof["matvec_backend"] = str(matvec_backend)
        if (
            matvec_backend in ("cuda_eri_mat", "cuda")
            and cuda_cp is not None
            and (eri_mat_d is not None or l_full_d is not None)
            and not bool(matvec_cuda_make_hdiag_cpu)
        ):
            try:
                from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                    build_hdiag_det_guess_from_steps_inplace_device as _build_hdiag_det_guess,
                )
                neleca_det = (int(nelec_total) + int(twos)) // 2
                norb_i = int(drt.norb)
                # Extract small integral slices needed by the diagonal guess.
                p = cuda_cp.arange(norb_i, dtype=cuda_cp.int32)
                idx_pp = p * int(norb_i + 1)

                h1e_diag_d = cuda_cp.asarray(np.diag(np.asarray(h1e, dtype=np.float64)), dtype=cuda_cp.float64)

                if eri_mat_d is not None:
                    eri_ppqq_d = eri_mat_d[idx_pp[:, None], idx_pp[None, :]].copy()
                    pp = p[:, None]
                    qq = p[None, :]
                    idx_pq = (pp * norb_i + qq).ravel()
                    idx_qp = (qq * norb_i + pp).ravel()
                    eri_pqqp_d = eri_mat_d[idx_pq, idx_qp].reshape(norb_i, norb_i).copy()
                else:
                    if l_full_d is None:  # pragma: no cover
                        raise RuntimeError("internal error: l_full_d is None in DF hdiag path")
                    # DF path: compute the small diagonal slices directly from L_full.
                    #
                    #   (pp|qq) = sum_L L_full[pp,L] * L_full[qq,L]
                    #   (pq|qp) = sum_L L_full[pq,L] * L_full[qp,L]
                    #
                    # without materializing the full ERI_mat.
                    l_pp = cuda_cp.ascontiguousarray(l_full_d[idx_pp])
                    eri_ppqq_d = cuda_cp.ascontiguousarray(cuda_cp.dot(l_pp, l_pp.T))

                    naux = int(l_full_d.shape[1])
                    eri_pqqp_d = cuda_cp.empty((norb_i, norb_i), dtype=cuda_cp.float64)
                    for p_i in range(norb_i):
                        l_pq = l_full_d[int(p_i) * int(norb_i) : (int(p_i) + 1) * int(norb_i)]
                        if l_pq.shape != (norb_i, naux):  # pragma: no cover
                            raise RuntimeError("internal error: unexpected DF l_pq shape")
                        l_qp = l_full_d[int(p_i) :: int(norb_i)]
                        if l_qp.shape != (norb_i, naux):  # pragma: no cover
                            raise RuntimeError("internal error: unexpected DF l_qp shape")
                        eri_pqqp_d[int(p_i)] = cuda_cp.sum(l_pq * l_qp, axis=1)
                    eri_pqqp_d = cuda_cp.ascontiguousarray(eri_pqqp_d)

                hdiag_d = _build_hdiag_det_guess(
                    state_dev,
                    neleca_det=neleca_det,
                    h1e_diag=h1e_diag_d,
                    eri_ppqq=eri_ppqq_d,
                    eri_pqqp=eri_pqqp_d,
                    threads=256,
                    stream=cuda_cp.cuda.get_current_stream(),
                    sync=False,
                )
                if kernel_profile_cuda_sync:
                    cuda_cp.cuda.get_current_stream().synchronize()
                if kprof is not None:
                    kprof["hdiag_backend"] = "cuda_det_guess" if eri_mat_d is not None else "cuda_det_guess_df_l_full"
            except Exception as e:
                if kprof is not None:
                    kprof["make_hdiag_cuda_error"] = f"{type(e).__name__}: {e}"
                    kprof["hdiag_backend"] = "cpu_fallback"
                if strict_gpu:
                    raise RuntimeError(
                        "strict_gpu=True forbids CUDA make_hdiag fallback to CPU"
                    ) from e
                hdiag_d = None
                hdiag = self.make_hdiag(h1e, eri, norb, nelec, orbsym=orbsym, wfnsym=wfnsym)
        else:
            if strict_gpu and matvec_backend in ("cuda_eri_mat", "cuda"):
                raise RuntimeError("strict_gpu=True requires CUDA hdiag path (CPU path reached)")
            if kprof is not None:
                kprof["hdiag_backend"] = "cpu"
            hdiag = self.make_hdiag(h1e, eri, norb, nelec, orbsym=orbsym, wfnsym=wfnsym)
        if kprof is not None:
            kprof["make_hdiag_s"] = time.perf_counter() - t0

        if hdiag is not None:
            def precond(dx, e, _x0):
                denom = hdiag - float(e)
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                return dx / denom

        if ci0 is None and pspace_size > 0:
            if kprof is not None:
                t0 = time.perf_counter()
            addr, h0 = self.pspace(
                h1e,
                eri,
                norb,
                nelec,
                npsp=max(pspace_size, nroots),
                max_out=max_out,
                orbsym=orbsym,
                wfnsym=wfnsym,
            )
            w, v = np.linalg.eigh(h0)
            order = np.argsort(w)[:nroots]
            ci0_list: list[np.ndarray] = []
            for col in order.tolist():
                vec = np.zeros(ncsf, dtype=np.float64)
                vec[addr] = v[:, col]
                ci0_list.append(vec)
            ci0_norm = ci0_list
            if kprof is not None:
                kprof["pspace_s"] = time.perf_counter() - t0
        elif ci0 is None:
            if hdiag_d is not None and cuda_cp is not None:
                if int(nroots) == 1:
                    idx = np.asarray([int(cuda_cp.argmin(hdiag_d).get())], dtype=np.int64)
                else:
                    idx_d = cuda_cp.argpartition(hdiag_d, int(nroots) - 1)[: int(nroots)]
                    idx = np.asarray(cuda_cp.asnumpy(idx_d), dtype=np.int64)
            else:
                assert hdiag is not None
                idx = np.argsort(hdiag)[:nroots]
            ci0_norm = []
            for j in idx.tolist():
                v = np.zeros(ncsf, dtype=np.float64)
                v[int(j)] = 1.0
                ci0_norm.append(v)
        else:
            ci0_norm = _normalize_ci0(ci0, nroots=nroots, ncsf=ncsf)

        contract_ws: ContractWorkspace | None = None
        if matvec_backend == "contract":
            from asuka.contract import ContractWorkspace as _ContractWorkspace  # noqa: PLC0415

            contract_ws = self._contract_ws_cache.get(drt_key)
            if contract_ws is None:
                contract_ws = _ContractWorkspace()
                self._contract_ws_cache[drt_key] = contract_ws

        contract_prof_tot: dict[str, float] | None = None
        if kprof is not None and matvec_backend == "contract":
            contract_prof_tot = {}

        def hop(xs: list[np.ndarray]) -> list[np.ndarray]:
            if matvec_backend == "contract":
                from asuka.contract import (  # noqa: PLC0415
                    contract_eri_epq_eqrs_multi as _contract_eri_epq_eqrs_multi,
                    contract_h_csf_multi as _contract_h_csf_multi,
                )
                from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_h_csf_multi_df  # noqa: PLC0415

                if isinstance(eri, DFMOIntegrals):
                    prof = {} if contract_prof_tot is not None else None
                    ys = _contract_h_csf_multi_df(
                        drt,
                        h1e,
                        eri,
                        xs,
                        precompute_epq=False,
                        nthreads=contract_nthreads,
                        blas_nthreads=contract_blas_nthreads,
                        workspace=contract_ws,
                        profile_out=prof,
                    )
                    if prof is not None and contract_prof_tot is not None:
                        for kk, vv in prof.items():
                            contract_prof_tot[kk] = float(contract_prof_tot.get(kk, 0.0)) + float(vv)
                    return ys
                prof = {} if contract_prof_tot is not None else None
                ys = _contract_h_csf_multi(
                    drt,
                    h1e,
                    eri,
                    xs,
                    precompute_epq=False,
                    nthreads=contract_nthreads,
                    blas_nthreads=contract_blas_nthreads,
                    workspace=contract_ws,
                    profile_out=prof,
                )
                if prof is not None and contract_prof_tot is not None:
                    for kk, vv in prof.items():
                        contract_prof_tot[kk] = float(contract_prof_tot.get(kk, 0.0)) + float(vv)
                return ys

            if matvec_backend == "row_oracle_df":
                if not isinstance(eri, DFMOIntegrals):
                    raise TypeError("matvec_backend='row_oracle_df' requires eri=DFMOIntegrals")
                return matvec_df_row_oracle(
                    drt,
                    h1e,
                    eri,
                    xs,
                    max_out=max_out,
                    screening=row_screening,
                    state_cache=state_cache,
                )

            raise ValueError(f"unsupported matvec_backend={matvec_backend!r}")

        verbose_in = kwargs.pop("verbose", self.verbose)
        if hasattr(verbose_in, "verbose"):
            verbose = int(getattr(verbose_in, "verbose", 0))
        else:
            try:
                verbose = int(verbose_in)
            except Exception:
                verbose = 0
        if matvec_backend == "row_oracle_df" and not bool(getattr(self, "_warned_row_oracle_df", False)):
            warnings.warn(
                "matvec_backend='row_oracle_df' is a single-threaded Python reference backend; "
                "for better CPU utilization use matvec_backend='contract'"
            )
            self._warned_row_oracle_df = True
        if matvec_backend in ("cuda_eri_mat", "cuda"):
                if cuda_matvec_ws is None or cuda_cp is None:
                    raise RuntimeError("internal error: CUDA matvec workspace is not initialized")
                hop_prof: dict[str, float] | None = {} if bool(matvec_cuda_hop_profile) else None
                hop_prof_low: dict[str, float] | None = (
                    {} if bool(matvec_cuda_hop_profile) and str(matvec_cuda_dtype) == "mixed" else None
                )

                def hop_d(v_d):
                    return cuda_matvec_ws.hop(v_d, sync=False, check_overflow=False, profile=hop_prof)

                hop_d_low = None
                if str(matvec_cuda_dtype) == "mixed":
                    if cuda_ws_low is None:
                        raise RuntimeError("internal error: missing low-precision CUDA workspace for mixed mode")

                    def hop_d_low(v_d):
                        return cuda_ws_low.hop(v_d, sync=False, check_overflow=False, profile=hop_prof_low)

                res = self._call_davidson_gpu(
                    hop_fn=hop_d,
                    hop_low_fn=hop_d_low,
                    hop_prof=hop_prof,
                    hop_prof_low=hop_prof_low,
                    ci0=ci0_norm,
                    hdiag=hdiag_d if hdiag_d is not None else hdiag,
                    nroots=nroots,
                    max_cycle=max_cycle,
                    max_space=max_space,
                    tol=tol,
                    lindep=lindep,
                    cuda_cp=cuda_cp,
                    kprof=kprof,
                    kernel_profile=kernel_profile,
                    kernel_profile_cuda_sync=kernel_profile_cuda_sync,
                    matvec_cuda_dtype=matvec_cuda_dtype,
                    matvec_cuda_mixed_threshold=matvec_cuda_mixed_threshold,
                    matvec_cuda_mixed_low_precision_max_iter=matvec_cuda_mixed_low_precision_max_iter,
                    matvec_cuda_mixed_force_final_full_hop=matvec_cuda_mixed_force_final_full_hop,
                    matvec_cuda_mixed_final_full_subspace_refresh=matvec_cuda_mixed_final_full_subspace_refresh,
                    matvec_cuda_davidson_subspace_eigh_cpu=matvec_cuda_davidson_subspace_eigh_cpu,
                    matvec_cuda_davidson_subspace_eigh_cpu_max_m=matvec_cuda_davidson_subspace_eigh_cpu_max_m,
                )
                self.converged, e, ci = res.converged, res.e, res.x
        elif matvec_backend in ("cuda_fixed_ell", "cuda_ell"):
                if cuda_fixed_ell_ws is None or cuda_cp is None:
                    raise RuntimeError("internal error: CUDA fixed-ELL workspace is not initialized")

                def hop_d(v_d):
                    if int(getattr(v_d, "ndim", 1)) == 1:
                        return cuda_fixed_ell_ws.hop(v_d, sync=False)
                    return cuda_fixed_ell_ws.hop_many(v_d, sync=False)

                res = self._call_davidson_gpu(
                    hop_fn=hop_d,
                    hop_low_fn=None,
                    hop_prof=None,
                    hop_prof_low=None,
                    ci0=ci0_norm,
                    hdiag=hdiag,
                    nroots=nroots,
                    max_cycle=max_cycle,
                    max_space=max_space,
                    tol=tol,
                    lindep=lindep,
                    cuda_cp=cuda_cp,
                    kprof=kprof,
                    kernel_profile=kernel_profile,
                    kernel_profile_cuda_sync=kernel_profile_cuda_sync,
                    matvec_cuda_dtype="float64",
                    matvec_cuda_mixed_threshold=None,
                    matvec_cuda_mixed_low_precision_max_iter=None,
                    matvec_cuda_mixed_force_final_full_hop=False,
                    matvec_cuda_mixed_final_full_subspace_refresh=False,
                    matvec_cuda_davidson_subspace_eigh_cpu=matvec_cuda_davidson_subspace_eigh_cpu,
                    matvec_cuda_davidson_subspace_eigh_cpu_max_m=matvec_cuda_davidson_subspace_eigh_cpu_max_m,
                )
                self.converged, e, ci = res.converged, res.e, res.x
        elif matvec_backend in ("cuda_fixed_sell", "cuda_sell"):
                if cuda_fixed_sell_ws is None or cuda_cp is None:
                    raise RuntimeError("internal error: CUDA fixed-SELL workspace is not initialized")

                def hop_d(v_d):
                    if int(getattr(v_d, "ndim", 1)) == 1:
                        return cuda_fixed_sell_ws.hop(v_d, sync=False)
                    return cuda_fixed_sell_ws.hop_many(v_d, sync=False)

                res = self._call_davidson_gpu(
                    hop_fn=hop_d,
                    hop_low_fn=None,
                    hop_prof=None,
                    hop_prof_low=None,
                    ci0=ci0_norm,
                    hdiag=hdiag,
                    nroots=nroots,
                    max_cycle=max_cycle,
                    max_space=max_space,
                    tol=tol,
                    lindep=lindep,
                    cuda_cp=cuda_cp,
                    kprof=kprof,
                    kernel_profile=kernel_profile,
                    kernel_profile_cuda_sync=kernel_profile_cuda_sync,
                    matvec_cuda_dtype="float64",
                    matvec_cuda_mixed_threshold=None,
                    matvec_cuda_mixed_low_precision_max_iter=None,
                    matvec_cuda_mixed_force_final_full_hop=False,
                    matvec_cuda_mixed_final_full_subspace_refresh=False,
                    matvec_cuda_davidson_subspace_eigh_cpu=matvec_cuda_davidson_subspace_eigh_cpu,
                    matvec_cuda_davidson_subspace_eigh_cpu_max_m=matvec_cuda_davidson_subspace_eigh_cpu_max_m,
                )
                self.converged, e, ci = res.converged, res.e, res.x
        else:
            if kprof is not None:
                t0 = time.perf_counter()
            if kernel_blas_nthreads is None:
                # Auto policy:
                # - If the contract backend uses Python threads, keep BLAS single-threaded to
                #   avoid oversubscription and expensive OpenBLAS thread switching.
                # - For small CI spaces, multi-threaded BLAS-1/2 kernels are often slower.
                if matvec_backend == "contract" and contract_nthreads > 1:
                    kernel_blas_nthreads = 1
                elif ncsf < 200_000:
                    kernel_blas_nthreads = 1
            blas_cm = (
                contextlib.nullcontext()
                if kernel_blas_nthreads is None
                else blas_thread_limit(int(kernel_blas_nthreads))
            )
            # OpenMP thread settings are per OS thread. The threaded contract backend
            # configures each Python worker thread once (see `asuka.contract`), so avoid
            # toggling OpenMP settings in the main thread here.
            with blas_cm:
                if kprof is not None:
                    res = davidson1_sym_result(
                        hop,
                        ci0_norm,
                        precond,
                        tol=tol,
                        lindep=lindep,
                        max_cycle=max_cycle,
                        max_space=max_space,
                        nroots=nroots,
                        profile=True,
                        **kwargs,
                    )
                    self.converged, e, ci = res.converged, res.e, res.x
                    kprof["davidson_niter"] = int(res.niter)
                    kprof["davidson_stats"] = res.stats
                else:
                    self.converged, e, ci = davidson1_sym(
                        hop,
                        ci0_norm,
                        precond,
                        tol=tol,
                        lindep=lindep,
                        max_cycle=max_cycle,
                        max_space=max_space,
                        nroots=nroots,
                        **kwargs,
                    )
            if kprof is not None:
                kprof["davidson_s"] = time.perf_counter() - t0
                if contract_prof_tot is not None:
                    kprof["matvec_contract_profile"] = contract_prof_tot

        conv_arr = np.asarray(self.converged, dtype=np.bool_).ravel()
        if int(conv_arr.size) == 1 and int(nroots) > 1:
            conv_arr = np.repeat(conv_arr, int(nroots))
        if int(conv_arr.size) != int(nroots):
            conv_arr = np.zeros((int(nroots),), dtype=np.bool_)
        self.converged = conv_arr

        if not bool(np.all(self.converged)):
            fallback_used = False
            if bool(unconverged_fallback_full_diag) and int(ncsf) <= int(unconverged_fallback_ncsf_max):
                if verbose >= 1:
                    print(
                        "  Davidson did not converge; falling back to full-CSF diagonalization "
                        f"(ncsf={int(ncsf)})."
                    )
                hdiag_ps = None if hdiag is None else np.asarray(hdiag, dtype=np.float64)
                addr_full, h_full = self.pspace(
                    h1e,
                    eri,
                    norb,
                    nelec,
                    npsp=int(ncsf),
                    max_out=int(max_out),
                    hdiag=hdiag_ps,
                    orbsym=orbsym,
                    wfnsym=wfnsym,
                    ne_constraints=ne_constraints,
                )
                if int(np.asarray(addr_full).size) != int(ncsf):
                    raise RuntimeError(
                        "full-space fallback failed: pspace address size mismatch "
                        f"(got {int(np.asarray(addr_full).size)}, expected {int(ncsf)})"
                    )
                h_full = np.asarray(h_full, dtype=np.float64)
                if h_full.shape != (int(ncsf), int(ncsf)):
                    raise RuntimeError(
                        "full-space fallback failed: pspace matrix shape mismatch "
                        f"(got {h_full.shape}, expected {(int(ncsf), int(ncsf))})"
                    )
                h_full = 0.5 * (h_full + h_full.T)
                evals, evecs = np.linalg.eigh(h_full)
                order = np.argsort(np.asarray(evals, dtype=np.float64))[: int(nroots)]
                e = np.asarray(evals, dtype=np.float64)[order]
                addr_idx = np.asarray(addr_full, dtype=np.int64).ravel()
                ci_full: list[np.ndarray] = []
                for col in order.tolist():
                    v_sub = np.asarray(evecs[:, int(col)], dtype=np.float64).ravel()
                    v_full = np.zeros((int(ncsf),), dtype=np.float64)
                    v_full[addr_idx] = v_sub
                    ci_full.append(np.ascontiguousarray(v_full))
                ci = ci_full
                self.converged = np.ones((int(nroots),), dtype=np.bool_)
                fallback_used = True
                if kprof is not None:
                    kprof["unconverged_fallback_used"] = True
                    kprof["unconverged_fallback_ncsf"] = int(ncsf)
            if not fallback_used:
                msg = (
                    "GUGAFCISolver.kernel did not converge for all requested roots "
                    f"(converged={self.converged.tolist()}, nroots={int(nroots)})."
                )
                if kprof is not None:
                    kprof["unconverged_fallback_used"] = False
                    kprof["unconverged_fallback_ncsf"] = int(ncsf)
                if bool(raise_on_unconverged):
                    raise RuntimeError(
                        msg
                        + " Increase max_cycle/max_space, or enable unconverged_fallback_full_diag "
                        "for small CSF spaces."
                    )
                if warn_on_unconverged:
                    warnings.warn(msg)

        e = np.asarray(e, dtype=np.float64) + float(ecore)
        if warm_state_update:
            self._update_warm_state(
                ci=ci,
                norb=int(norb),
                nelec_total=int(nelec_total),
                twos=int(twos),
                nroots=int(nroots),
                ncsf=int(ncsf),
                orbsym=orbsym,
                wfnsym=wfnsym,
                ne_constraints=ne_constraints,
                cas_metadata=warm_cas_metadata,
                mo_coeff=warm_state_mo_coeff,
                mo_occ=warm_state_mo_occ,
            )
        if nroots == 1:
            self.converged = bool(self.converged[0])
            self.eci, self.ci = float(e[0]), np.ascontiguousarray(ci[0])
            self._last_drt_key = drt_key
            if kprof is not None:
                kprof.update(self._matvec_cuda_ws_cache_profile())
                kprof["total_s"] = time.perf_counter() - t_kernel0
                self._last_kernel_profile = kprof
                if kernel_profile_print:
                    print("GUGAFCISolver.kernel profile:", kprof)
            return self.eci, self.ci

        self.eci, self.ci = e, [np.ascontiguousarray(v) for v in ci]
        self._last_drt_key = drt_key
        if kprof is not None:
            kprof.update(self._matvec_cuda_ws_cache_profile())
            kprof["total_s"] = time.perf_counter() - t_kernel0
            self._last_kernel_profile = kprof
            if kernel_profile_print:
                print("GUGAFCISolver.kernel profile:", kprof)
        return self.eci, self.ci

    def _drt_for_ci(
        self,
        civec: Any,
        *,
        norb: int,
        nelec_total: int,
        twos: int,
        orbsym: Any | None,
        wfnsym: int | None,
        ne_constraints: dict[int, tuple[int, int]] | None,
    ) -> tuple[_DRTKey, DRT]:
        """Return (drt_key, drt) consistent with *civec*.

        PySCF symmetry-enabled CASSCF may call `make_rdm12(ci, ...)` without passing
        `orbsym/wfnsym`, even though the most recent `kernel()` call did.  In that
        case, fall back to the last kernel DRT metadata if it matches the CI size.
        """

        drt_key = self._drt_key(
            norb,
            nelec_total,
            twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        drt = self._get_drt(
            norb,
            nelec_total,
            twos,
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
        )
        try:
            _validate_civec_shape(civec, int(drt.ncsf))
        except ValueError:
            # Only attempt the fallback when symmetry metadata was *not* explicitly provided.
            if orbsym is not None or wfnsym is not None:
                raise

            last = getattr(self, "_last_drt_key", None)
            if last is None:
                raise
            if int(last.norb) != int(norb) or int(last.nelec_total) != int(nelec_total) or int(last.twos) != int(twos):
                raise

            # Use the last kernel DRT only if it matches the CI vector length.
            ncsf_ci = int(civec.size) if hasattr(civec, 'size') else int(np.asarray(civec).size)
            drt_last = self._drt_cache.get(last)
            if drt_last is None:
                # Rebuild (should be cheap for cached tables; required for external callers).
                drt_last = self._get_drt(
                    norb,
                    nelec_total,
                    twos,
                    orbsym=last.orbsym,
                    wfnsym=last.wfnsym,
                    ne_constraints=_ne_constraints_key_to_dict(last.ne_constraints_key),
                )
            if int(drt_last.ncsf) != ncsf_ci:
                raise

            _validate_civec_shape(civec, int(drt_last.ncsf))
            return last, drt_last

        return drt_key, drt

    def approx_kernel(
        self,
        h1e,
        eri,
        norb: int,
        nelec: int | tuple[int, int],
        ci0=None,
        ecore: float = 0.0,
        nroots: int = 1,
        **kwargs,
    ):
        """Approximate CI solve used inside PySCF CASSCF orbital updates.

        PySCF's CASSCF `solve_approx_ci` calls `fcisolver.approx_kernel` without
        passing `max_cycle`, so we cap iterations here by default to avoid doing
        a full CI solve during the orbital optimization micro-steps.
        """

        # Avoid expensive pspace construction for approximate solves.
        kwargs["pspace_size"] = 0

        matvec_backend = str(kwargs.get("matvec_backend", getattr(self, "matvec_backend", "contract"))).strip().lower()
        _approx_cuda_frontend = _resolve_approx_cuda_frontend_runtime(
            kwargs=kwargs,
            defaults=self,
            matvec_backend=matvec_backend,
        )
        approx_cuda_dtype = _approx_cuda_frontend["approx_cuda_dtype"]
        matvec_cuda_aggregate_offdiag_preview = _approx_cuda_frontend["matvec_cuda_aggregate_offdiag_preview"]
        _approx_caps = _resolve_approx_kernel_iteration_caps_runtime(
            kwargs=kwargs,
            defaults=self,
            nroots=nroots,
            matvec_backend=matvec_backend,
        )
        nroots_i = int(_approx_caps["nroots"])
        max_cycle = int(_approx_caps["max_cycle"])
        max_space = int(_approx_caps["max_space"])

        # Fast path for the GPU backend (single-root): a small Arnoldi/Rayleigh-Ritz
        # solve with a bounded subspace, to avoid running a full Davidson solve
        # inside `mc.update_casdm`.
        if matvec_backend in ("cuda_eri_mat", "cuda") and ci0 is not None and 1 <= nroots_i <= 4:
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception:
                cp = None

            if cp is not None:
                _approx_cuda_mem_ctl = _resolve_cuda_memory_controls_runtime(
                    kwargs=kwargs,
                    defaults=self,
                    consume=False,
                )
                approx_mem_hard_cap_gib = float(_approx_cuda_mem_ctl["matvec_cuda_mem_hard_cap_gib"])
                approx_ws_cache_fraction = float(_approx_cuda_mem_ctl["matvec_cuda_ws_cache_fraction"])
                _apply_cuda_pool_hard_cap(cp, float(approx_mem_hard_cap_gib))
                self._configure_matvec_cuda_ws_cache(
                    cp_mod=cp,
                    hard_cap_gib=float(approx_mem_hard_cap_gib),
                    fraction=float(approx_ws_cache_fraction),
                )
                orbsym = kwargs.get("orbsym", self.orbsym)
                wfnsym = kwargs.get("wfnsym", self.wfnsym)
                ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))

                norb_i = int(norb)
                neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
                twos = self._get_twos_target(neleca, nelecb)
                drt = self._get_drt(
                    norb_i,
                    nelec_total,
                    twos,
                    orbsym=orbsym,
                    wfnsym=wfnsym,
                    ne_constraints=ne_constraints,
                )
                ncsf = int(drt.ncsf)
                matvec_cuda_use_epq_preview_in = kwargs.get(
                    "matvec_cuda_use_epq_table",
                    getattr(self, "matvec_cuda_use_epq_table", None),
                )
                if matvec_cuda_use_epq_preview_in is not None:
                    _enforce_cuda_fp32_large_cas_epq_policy(
                        context="approx_kernel(cuda)",
                        matvec_cuda_dtype=str(approx_cuda_dtype),
                        matvec_cuda_use_epq_table=bool(matvec_cuda_use_epq_preview_in),
                        matvec_cuda_aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag_preview),
                        ncsf=int(ncsf),
                    )
                if ncsf > 0:
                    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                        GugaMatvecEriMatWorkspace,
                        has_cuda_ext,
                        has_epq_table_device_build,
                        make_device_drt,
                        make_device_state_cache,
                    )

                    if not has_cuda_ext():
                        raise RuntimeError(
                            "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
                        )

                    ws_key = self._drt_key(
                        norb_i,
                        nelec_total,
                        twos,
                        orbsym=orbsym,
                        wfnsym=wfnsym,
                        ne_constraints=ne_constraints,
                    )
                    drt_dev, state_dev = _get_or_create_cuda_matvec_state_runtime(
                        state_cache=self._matvec_cuda_state_cache,
                        ws_key=ws_key,
                        drt=drt,
                        make_device_drt_fn=make_device_drt,
                        make_device_state_cache_fn=make_device_state_cache,
                    )

                    nops = norb_i * norb_i
                    if isinstance(eri, DeviceDFMOIntegrals):
                        if eri.eri_mat is None:
                            raise RuntimeError("DeviceDFMOIntegrals requires eri_mat for matvec_backend='cuda_eri_mat'")
                        eri_mat_d = cp.asarray(eri.eri_mat, dtype=cp.float64)
                        eri_mat_d = cp.ascontiguousarray(eri_mat_d)
                        j_ps_d = cp.asarray(eri.j_ps, dtype=cp.float64)
                        j_ps_d = cp.ascontiguousarray(j_ps_d)
                        h1e_d = cp.asarray(np.asarray(h1e, dtype=np.float64), dtype=cp.float64)
                        h_eff_d = h1e_d - 0.5 * j_ps_d
                    else:
                        if isinstance(eri, DFMOIntegrals):
                            eri_mat_host = eri._maybe_build_eri_mat(eri_mat_max_bytes=1 << 62)
                            if eri_mat_host is None:
                                raise RuntimeError("DFMOIntegrals could not materialize ERI_mat")
                            j_ps = np.asarray(eri.j_ps, dtype=np.float64, order="C")
                        else:
                            eri4 = _restore_eri_4d(eri, norb_i).astype(np.float64, copy=False)
                            eri_mat_host = np.asarray(eri4.reshape(nops, nops), dtype=np.float64, order="C")
                            j_ps = np.einsum("pqqs->ps", eri4).astype(np.float64, copy=False)

                        h_eff = np.asarray(h1e, dtype=np.float64) - 0.5 * np.asarray(j_ps, dtype=np.float64)
                        eri_mat_d = cp.asarray(eri_mat_host, dtype=cp.float64)
                        h_eff_d = cp.asarray(h_eff, dtype=cp.float64)

                    # Match the kernel() CUDA workspace heuristics to avoid tiny j-tiles and repeated CSR builds.
                    _approx_cuda_ws_controls = _resolve_cuda_workspace_controls_runtime(
                        kwargs=kwargs,
                        defaults=self,
                        consume=False,
                        context="approx_kernel(cuda)",
                    )
                    matvec_cuda_target_ntasks = int(_approx_cuda_ws_controls["matvec_cuda_target_ntasks"])
                    matvec_cuda_j_tile_align = int(_approx_cuda_ws_controls["matvec_cuda_j_tile_align"])
                    matvec_cuda_j_tile = int(_approx_cuda_ws_controls["matvec_cuda_j_tile_requested"])
                    matvec_cuda_j_tile = _resolve_cuda_j_tile_runtime(
                        requested_j_tile=matvec_cuda_j_tile,
                        target_ntasks=matvec_cuda_target_ntasks,
                        j_tile_align=matvec_cuda_j_tile_align,
                        norb=norb_i,
                        ncsf=int(drt.ncsf),
                    )
                    matvec_cuda_csr_capacity_mult = float(_approx_cuda_ws_controls["matvec_cuda_csr_capacity_mult"])
                    matvec_cuda_csr_host_cache_mode = str(_approx_cuda_ws_controls["matvec_cuda_csr_host_cache_mode"])
                    matvec_cuda_csr_host_cache_budget_gib = float(
                        _approx_cuda_ws_controls["matvec_cuda_csr_host_cache_budget_gib"]
                    )
                    matvec_cuda_csr_host_cache_min_ncsf = int(
                        _approx_cuda_ws_controls["matvec_cuda_csr_host_cache_min_ncsf"]
                    )
                    matvec_cuda_csr_pipeline_streams_mode = str(
                        _approx_cuda_ws_controls["matvec_cuda_csr_pipeline_streams_mode"]
                    )
                    matvec_cuda_csr_pipeline_streams_value = _approx_cuda_ws_controls[
                        "matvec_cuda_csr_pipeline_streams_value"
                    ]
                    if matvec_cuda_csr_pipeline_streams_value is None:
                        matvec_cuda_csr_pipeline_streams = str(matvec_cuda_csr_pipeline_streams_mode)
                    else:
                        matvec_cuda_csr_pipeline_streams = int(matvec_cuda_csr_pipeline_streams_value)
                    matvec_cuda_csr_pipeline_min_ncsf = int(
                        _approx_cuda_ws_controls["matvec_cuda_csr_pipeline_min_ncsf"]
                    )
                    matvec_cuda_prefilter_trivial_tasks_mode = str(
                        _approx_cuda_ws_controls["matvec_cuda_prefilter_trivial_tasks_mode"]
                    )
                    matvec_cuda_prefilter_trivial_tasks_min_ncsf = int(
                        _approx_cuda_ws_controls["matvec_cuda_prefilter_trivial_tasks_min_ncsf"]
                    )
                    _approx_cuda_mem_ctl2 = _resolve_cuda_memory_controls_runtime(
                        kwargs=kwargs,
                        defaults=self,
                        consume=False,
                    )
                    matvec_cuda_mem_hard_cap_gib = float(_approx_cuda_mem_ctl2["matvec_cuda_mem_hard_cap_gib"])
                    threads_enum_forced = bool(_approx_cuda_ws_controls["threads_enum_forced"])
                    threads_g_forced = bool(_approx_cuda_ws_controls["threads_g_forced"])
                    matvec_cuda_threads_enum = int(_approx_cuda_ws_controls["matvec_cuda_threads_enum"])
                    matvec_cuda_threads_g = int(_approx_cuda_ws_controls["matvec_cuda_threads_g"])
                    matvec_cuda_threads_w = int(_approx_cuda_ws_controls["matvec_cuda_threads_w"])
                    matvec_cuda_threads_apply = int(_approx_cuda_ws_controls["matvec_cuda_threads_apply"])
                    threads_apply_auto = bool(_approx_cuda_ws_controls["threads_apply_auto"])
                    _max_g_cfg2 = _resolve_cuda_max_g_mib_runtime(
                        kwargs=kwargs,
                        defaults=self,
                        consume=False,
                    )
                    max_g_forced = bool(_max_g_cfg2["max_g_forced"])
                    matvec_cuda_max_g_mib = float(_max_g_cfg2["matvec_cuda_max_g_mib"])
                    matvec_cuda_coalesce = bool(_approx_cuda_ws_controls["matvec_cuda_coalesce"])
                    matvec_cuda_include_diagonal_rs = bool(_approx_cuda_ws_controls["matvec_cuda_include_diagonal_rs"])
                    matvec_cuda_cache_csr_tiles = _approx_cuda_ws_controls["matvec_cuda_cache_csr_tiles"]
                    matvec_cuda_fuse_count_write = bool(_approx_cuda_ws_controls["matvec_cuda_fuse_count_write"])
                    matvec_cuda_fp32_coeff_data = bool(_approx_cuda_ws_controls["matvec_cuda_fp32_coeff_data"])
                    matvec_cuda_use_epq_table = _approx_cuda_ws_controls["matvec_cuda_use_epq_table_in"]
                    matvec_cuda_aggregate_offdiag = bool(_approx_cuda_ws_controls["matvec_cuda_aggregate_offdiag"])
                    matvec_cuda_max_g_mib = _autotune_cuda_max_g_mib_for_large_cas_runtime(
                        max_g_mib=float(matvec_cuda_max_g_mib),
                        max_g_forced=bool(max_g_forced),
                        aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                        ncsf=int(drt.ncsf),
                        norb=int(norb),
                        matvec_cuda_dtype=str(approx_cuda_dtype),
                        eri_mat_present=bool(eri_mat_d is not None),
                        mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                        cuda_budget_free_bytes_fn=_cuda_budget_free_bytes,
                    )
                    matvec_cuda_enable_fp64_emulation = bool(
                        kwargs.get(
                            "matvec_cuda_enable_fp64_emulation",
                            getattr(self, "matvec_cuda_enable_fp64_emulation", False),
                        )
                    )
                    matvec_cuda_gemm_backend = str(
                        kwargs.get(
                            "matvec_cuda_gemm_backend",
                            getattr(self, "matvec_cuda_gemm_backend", "gemmex_fp64"),
                        )
                    ).strip()
                    if str(approx_cuda_dtype) == "float32" and str(matvec_cuda_gemm_backend).lower() in ("gemmex_fp64", "cublaslt_fp64"):
                        matvec_cuda_gemm_backend = "gemmex_tf32"
                    matvec_cuda_emulation_strategy = str(
                        kwargs.get(
                            "matvec_cuda_emulation_strategy",
                            getattr(self, "matvec_cuda_emulation_strategy", "performant"),
                        )
                    )
                    matvec_cuda_cublas_workspace_cap_mb = _resolve_cuda_cublas_workspace_cap_mb_runtime(
                        kwargs=kwargs,
                        defaults=self,
                        hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                        consume=False,
                    )
                    _apply_mode_cfg2 = _resolve_cuda_apply_mode_runtime(
                        kwargs=kwargs,
                        defaults=self,
                        consume=False,
                    )
                    apply_mode_forced = bool(_apply_mode_cfg2["apply_mode_forced"])
                    matvec_cuda_apply_mode = str(_apply_mode_cfg2["matvec_cuda_apply_mode"])
                    matvec_cuda_epq_build_nthreads = int(
                        kwargs.get(
                            "matvec_cuda_epq_build_nthreads",
                            getattr(self, "matvec_cuda_epq_build_nthreads", 0),
                        )
                    )
                    matvec_cuda_use_graph = bool(
                        kwargs.get(
                            "matvec_cuda_use_graph",
                            getattr(self, "matvec_cuda_use_graph", False),
                        )
                    )
                    matvec_cuda_graph_warmup = bool(
                        kwargs.get(
                            "matvec_cuda_graph_warmup",
                            getattr(self, "matvec_cuda_graph_warmup", True),
                        )
                    )

                    if matvec_cuda_use_epq_table is None:
                        matvec_cuda_use_epq_table = _auto_select_use_epq_table_runtime(
                            cp=cp,
                            norb=int(norb_i),
                            ncsf=int(drt.ncsf),
                            aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                            has_epq_table_device_build=bool(has_epq_table_device_build()),
                            mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                            dtype_mode=str(approx_cuda_dtype),
                            eri_mat_present=bool(eri_mat_d is not None),
                        )
                    else:
                        matvec_cuda_use_epq_table = bool(matvec_cuda_use_epq_table)

                    if bool(matvec_cuda_use_epq_table) and float(matvec_cuda_mem_hard_cap_gib) > 0.0:
                        epq_peak_est_bytes = _estimate_epq_peak_bytes(int(drt.ncsf), int(norb_i))
                        try:
                            epq_budget_free = _cuda_budget_free_bytes(cp, float(matvec_cuda_mem_hard_cap_gib))
                        except Exception:
                            epq_budget_free = 0
                        if (not epq_budget_free) or int(epq_peak_est_bytes) > int(float(epq_budget_free) * 0.85):
                            matvec_cuda_use_epq_table = False
                            if threads_apply_auto:
                                matvec_cuda_threads_apply = 64 if int(drt.ncsf) >= 1_000_000 else 32

                    _policy_apply2 = _apply_low_precision_and_workspace_policy_runtime(
                        context="approx_kernel(cuda)",
                        dtype_mode=str(approx_cuda_dtype),
                        use_epq_table=bool(matvec_cuda_use_epq_table),
                        aggregate_offdiag=bool(matvec_cuda_aggregate_offdiag),
                        ncsf=int(drt.ncsf),
                        eri_mat_present=bool(eri_mat_d is not None),
                        enable_fp64_emulation=bool(matvec_cuda_enable_fp64_emulation),
                        use_graph=bool(matvec_cuda_use_graph),
                        apply_mode=str(matvec_cuda_apply_mode),
                        apply_mode_forced=bool(apply_mode_forced),
                        nops=int(nops),
                        threads_enum=int(matvec_cuda_threads_enum),
                        threads_g=int(matvec_cuda_threads_g),
                        threads_w=int(matvec_cuda_threads_w),
                        threads_apply=int(matvec_cuda_threads_apply),
                        threads_enum_forced=bool(threads_enum_forced),
                        threads_g_forced=bool(threads_g_forced),
                        threads_apply_auto=bool(threads_apply_auto),
                        max_g_mib=float(matvec_cuda_max_g_mib),
                        mem_hard_cap_gib=float(matvec_cuda_mem_hard_cap_gib),
                        cache_csr_tiles_in=matvec_cuda_cache_csr_tiles,
                        j_tile=int(matvec_cuda_j_tile),
                        norb=int(norb_i),
                        csr_capacity_mult=float(matvec_cuda_csr_capacity_mult),
                        noepq_large_ncsf_uses_64=False,
                    )
                    matvec_cuda_use_graph = bool(_policy_apply2["use_graph"])
                    matvec_cuda_apply_mode = str(_policy_apply2["apply_mode"])
                    matvec_cuda_threads_enum = int(_policy_apply2["threads_enum"])
                    matvec_cuda_threads_g = int(_policy_apply2["threads_g"])
                    matvec_cuda_threads_w = int(_policy_apply2["threads_w"])
                    matvec_cuda_threads_apply = int(_policy_apply2["threads_apply"])
                    matvec_cuda_max_g_mib = float(_policy_apply2["max_g_mib"])
                    matvec_cuda_cache_csr_tiles = _policy_apply2["cache_csr_tiles"]

                    ws_cache_key = (ws_key, str(approx_cuda_dtype))
                    cuda_ws = self._matvec_cuda_ws_cache_get(ws_cache_key)
                    want_max_g_bytes = int(matvec_cuda_max_g_mib * 1024 * 1024)
                    if self._ws_needs_rebuild(
                        cuda_ws,
                        expected_dtype=np.float32 if str(approx_cuda_dtype) == "float32" else np.float64,
                        j_tile=matvec_cuda_j_tile,
                        csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                        threads_enum=matvec_cuda_threads_enum,
                        threads_g=matvec_cuda_threads_g,
                        threads_w=matvec_cuda_threads_w,
                        threads_apply=matvec_cuda_threads_apply,
                        max_g_bytes=want_max_g_bytes,
                        coalesce=matvec_cuda_coalesce,
                        include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                        fuse_count_write=matvec_cuda_fuse_count_write,
                        fp32_coeff_data=matvec_cuda_fp32_coeff_data,
                        path_mode=_normalize_matvec_cuda_path_mode(
                            getattr(self, "matvec_cuda_path_mode", "auto")
                        ),
                        use_fused_hop=bool(getattr(self, "matvec_cuda_use_fused_hop", True)),
                        use_epq_table=matvec_cuda_use_epq_table,
                        aggregate_offdiag_k=matvec_cuda_aggregate_offdiag,
                        l_full_d=None,
                        enable_fp64_emulation=matvec_cuda_enable_fp64_emulation,
                        gemm_backend=matvec_cuda_gemm_backend,
                        emulation_strategy=matvec_cuda_emulation_strategy,
                        cublas_workspace_cap_mb=matvec_cuda_cublas_workspace_cap_mb,
                        apply_mode=matvec_cuda_apply_mode,
                        epq_build_device=bool(getattr(cuda_ws, "epq_build_device", False)),
                        epq_build_j_tile=int(getattr(cuda_ws, "epq_build_j_tile", 0)),
                        epq_streaming=bool(getattr(cuda_ws, "epq_streaming", False)),
                        epq_stream_j_tile=int(getattr(cuda_ws, "epq_stream_j_tile", 0)),
                        epq_stream_use_recompute=str(getattr(cuda_ws, "epq_stream_use_recompute", "auto")),
                        cache_csr_tiles=matvec_cuda_cache_csr_tiles,
                        csr_host_cache_mode=matvec_cuda_csr_host_cache_mode,
                        csr_host_cache_budget_gib=matvec_cuda_csr_host_cache_budget_gib,
                        csr_host_cache_min_ncsf=matvec_cuda_csr_host_cache_min_ncsf,
                        csr_pipeline_streams_mode=matvec_cuda_csr_pipeline_streams_mode,
                        csr_pipeline_streams_value=matvec_cuda_csr_pipeline_streams_value,
                        csr_pipeline_min_ncsf=matvec_cuda_csr_pipeline_min_ncsf,
                        prefilter_trivial_tasks_mode=matvec_cuda_prefilter_trivial_tasks_mode,
                        prefilter_trivial_tasks_min_ncsf=matvec_cuda_prefilter_trivial_tasks_min_ncsf,
                    ):
                        epq_table_forced = "matvec_cuda_use_epq_table" in kwargs

                        def _init_cuda_ws(*, use_epq_table: bool, max_g_bytes: int):
                            ws_dtype_obj = cp.float32 if str(approx_cuda_dtype) == "float32" else cp.float64
                            return GugaMatvecEriMatWorkspace(
                                drt,
                                drt_dev=drt_dev,
                                state_dev=state_dev,
                                eri_mat=eri_mat_d,
                                h_eff=h_eff_d,
                                j_tile=matvec_cuda_j_tile,
                                csr_capacity_mult=matvec_cuda_csr_capacity_mult,
                                cache_csr_tiles=bool(matvec_cuda_cache_csr_tiles),
                                csr_host_cache=str(matvec_cuda_csr_host_cache_mode),
                                csr_host_cache_budget_gib=float(matvec_cuda_csr_host_cache_budget_gib),
                                csr_host_cache_min_ncsf=int(matvec_cuda_csr_host_cache_min_ncsf),
                                csr_pipeline_streams=matvec_cuda_csr_pipeline_streams,
                                csr_pipeline_min_ncsf=int(matvec_cuda_csr_pipeline_min_ncsf),
                                prefilter_trivial_tasks=str(matvec_cuda_prefilter_trivial_tasks_mode),
                                prefilter_trivial_tasks_min_ncsf=int(matvec_cuda_prefilter_trivial_tasks_min_ncsf),
                                threads_enum=matvec_cuda_threads_enum,
                                threads_g=matvec_cuda_threads_g,
                                threads_w=matvec_cuda_threads_w,
                                threads_apply=matvec_cuda_threads_apply,
                                max_g_bytes=int(max_g_bytes),
                                coalesce=matvec_cuda_coalesce,
                                include_diagonal_rs=matvec_cuda_include_diagonal_rs,
                                fuse_count_write=bool(matvec_cuda_fuse_count_write),
                                path_mode=_normalize_matvec_cuda_path_mode(
                                    getattr(self, "matvec_cuda_path_mode", "auto")
                                ),
                                use_fused_hop=bool(getattr(self, "matvec_cuda_use_fused_hop", True)),
                                fp32_coeff_data=bool(matvec_cuda_fp32_coeff_data),
                                use_epq_table=bool(use_epq_table),
                                aggregate_offdiag_k=bool(matvec_cuda_aggregate_offdiag),
                                offdiag_enable_fp64_emulation=bool(matvec_cuda_enable_fp64_emulation),
                                offdiag_emulation_strategy=str(matvec_cuda_emulation_strategy),
                                offdiag_cublas_workspace_cap_mb=int(matvec_cuda_cublas_workspace_cap_mb),
                                gemm_backend=str(matvec_cuda_gemm_backend),
                                apply_mode=str(matvec_cuda_apply_mode),
                                epq_build_nthreads=int(matvec_cuda_epq_build_nthreads),
                                use_cuda_graph=bool(matvec_cuda_use_graph),
                                dtype=ws_dtype_obj,
                            )

                        try:
                            cuda_ws = _init_cuda_ws(
                                use_epq_table=bool(matvec_cuda_use_epq_table),
                                max_g_bytes=int(want_max_g_bytes),
                            )
                        except Exception as e:
                            msg = str(e).lower()
                            oom = "out of memory" in msg or "alloc" in msg or "memoryerror" in msg
                            if (
                                (not epq_table_forced)
                                and bool(matvec_cuda_use_epq_table)
                                and oom
                                and str(approx_cuda_dtype) == "float64"
                            ):
                                warnings.warn(
                                    "CUDA approx_kernel: epq_table build failed (likely OOM); falling back to CSR path. "
                                    "Set matvec_cuda_use_epq_table=0 to silence."
                                )
                                matvec_cuda_use_epq_table = False
                                if threads_apply_auto:
                                    matvec_cuda_threads_apply = 64 if int(drt.ncsf) >= 1_000_000 else 32
                                cuda_ws = _init_cuda_ws(use_epq_table=False, max_g_bytes=int(want_max_g_bytes))
                            else:
                                raise
                        self._matvec_cuda_ws_cache_put(
                            ws_cache_key,
                            cuda_ws,
                            keep_keys=(ws_cache_key,),
                        )
                    else:
                        ws_dtype_obj = np.dtype(getattr(cuda_ws, "dtype", np.float64))
                        cuda_ws.eri_mat = None if eri_mat_d is None else cp.ascontiguousarray(cp.asarray(eri_mat_d, dtype=ws_dtype_obj))
                        h_eff_flat_new = cuda_ws._as_h_eff_flat(h_eff_d)
                        if getattr(cuda_ws, "h_eff_flat", None) is None or tuple(getattr(cuda_ws.h_eff_flat, "shape", ())) != tuple(
                            getattr(h_eff_flat_new, "shape", ())
                        ):
                            cuda_ws.h_eff_flat = h_eff_flat_new
                            if getattr(cuda_ws, "_cuda_graph", None) is not None:
                                # Graph capture pointers depend on stable `h_eff_flat`; fall back to
                                # re-capturing if the buffer had to be replaced.
                                cuda_ws._cuda_graph = None
                                cuda_ws._cuda_graph_x = None
                                cuda_ws._cuda_graph_y = None
                        else:
                            cp.copyto(cuda_ws.h_eff_flat, h_eff_flat_new)

                        # Diagonal-rs contribution depends on `eri_diag_t` extracted from `eri_mat`.
                        # Invalidate it whenever the Hamiltonian changes.
                        cuda_ws._eri_diag_t = None

                        # Keep `eri_mat_t` pointer-stable for CUDA Graph reuse.
                        if getattr(cuda_ws, "_eri_mat_t", None) is not None:
                            cp.copyto(cuda_ws._eri_mat_t, cuda_ws.eri_mat.T)
                        else:
                            cuda_ws._eri_mat_t = cuda_ws.eri_mat.T.copy()
                            if getattr(cuda_ws, "_cuda_graph", None) is not None:
                                cuda_ws._cuda_graph = None
                                cuda_ws._cuda_graph_x = None
                                cuda_ws._cuda_graph_y = None

                        # If a CUDA Graph was captured, update the diagonal-rs cache in place so the
                        # captured pointers remain valid and the matvec stays correct.
                        if (
                            bool(matvec_cuda_use_graph)
                            and getattr(cuda_ws, "_cuda_graph", None) is not None
                            and bool(getattr(cuda_ws, "include_diagonal_rs", False))
                        ):
                            cuda_ws._build_diag_g_cache()

                        cuda_ws.use_cuda_graph = bool(matvec_cuda_use_graph)

                    if bool(matvec_cuda_use_graph):
                        # Capture once and reuse across many approx_kernel calls (Hamiltonian updates
                        # update buffers in place, so the captured graph remains valid).
                        try:
                            cuda_ws.enable_cuda_graph(warmup=bool(matvec_cuda_graph_warmup))
                        except Exception:
                            pass

                    # Use a small subspace:
                    # - nroots==1: interpret `max_cycle` as the maximum number of Arnoldi steps
                    # - nroots>1: interpret `max_space` as the maximum subspace dimension
                    if nroots_i == 1:
                        subspace_dim = max(1, min(int(max_cycle), int(max_space), 8))
                    else:
                        subspace_dim = max(nroots_i, min(int(max_space), 8))

                    x0_list = _normalize_ci0(ci0, nroots=nroots_i, ncsf=ncsf)
                    v_list: list[Any] = []
                    av_list: list[Any] = []

                    # Build the initial orthonormal block.
                    for x0 in x0_list:
                        v = cp.asarray(x0, dtype=cp.float64)
                        vn = cp.linalg.norm(v)
                        if float(vn) == 0.0:
                            continue
                        v = v / vn
                        for u in v_list:
                            v = v - cp.vdot(u, v) * u
                        vn2 = cp.linalg.norm(v)
                        if float(vn2) < 1e-12:
                            continue
                        v = v / vn2
                        v_list.append(v)
                        av_list.append(cp.asarray(cuda_ws.hop(v, sync=False, check_overflow=False), dtype=cp.float64))

                    if len(v_list) < nroots_i:
                        raise RuntimeError("failed to build enough linearly independent initial vectors for approx_kernel")

                    # Expand subspace using a simple multi-start Arnoldi scheme:
                    # iterate over existing basis vectors, using w = A*v as the next candidate direction.
                    expand_cursor = 0
                    while len(v_list) < subspace_dim and expand_cursor < len(v_list):
                        w = av_list[expand_cursor]
                        expand_cursor += 1
                        for u in v_list:
                            w = w - cp.vdot(u, w) * u
                        wn = cp.linalg.norm(w)
                        if float(wn) < 1e-12:
                            continue
                        v_new = w / wn
                        v_list.append(v_new)
                        av_list.append(cp.asarray(cuda_ws.hop(v_new, sync=False, check_overflow=False), dtype=cp.float64))

                    vmat = cp.stack(v_list, axis=1)  # (ncsf,m)
                    avmat = cp.stack(av_list, axis=1)  # (ncsf,m)
                    hsub = vmat.T.conj() @ avmat  # (m,m)
                    hsub = 0.5 * (hsub + hsub.T.conj())

                    # For tiny subspace matrices, host eigh is typically faster than cuSOLVER dispatch.
                    hsub_h = cp.asnumpy(hsub)
                    evals_h, evecs_h = np.linalg.eigh(hsub_h)
                    order = np.argsort(evals_h)[:nroots_i]
                    evals_h = np.asarray(evals_h[order], dtype=np.float64)
                    u_r = cp.asarray(evecs_h[:, order], dtype=cp.float64)

                    xmat = vmat @ u_r  # (ncsf, nroots)
                    xnorm = cp.linalg.norm(xmat, axis=0)
                    xmat = xmat / xnorm[None, :]

                    e_out = evals_h + float(ecore)
                    xmat_h = cp.asnumpy(xmat)
                    if nroots_i == 1:
                        return float(e_out[0]), np.ascontiguousarray(xmat_h[:, 0])
                    return np.asarray(e_out, dtype=np.float64), [np.ascontiguousarray(xmat_h[:, i]) for i in range(nroots_i)]

        # Fallback: run a truncated solve (supports multi-root).
        return GUGAFCISolver.kernel(
            self,
            h1e,
            eri,
            norb,
            nelec,
            ci0=ci0,
            ecore=ecore,
            nroots=nroots,
            max_cycle=max_cycle,
            max_space=max_space,
            **kwargs,
        )

    def absorb_h1e(self, h1e, eri, norb: int, nelec: int | tuple[int, int], fac: float = 1):
        """Return an effective 2e tensor with the 1e part absorbed.

        This follows the same convention as :func:`pyscf.fci.direct_spin1.absorb_h1e`.
        """
        backend = getattr(self, "contract_2e_backend", None) or getattr(self, "matvec_backend", "contract")
        absorb_h1e_mode = str(getattr(self, "absorb_h1e_mode", "tensor")).lower().strip()
        try:
            import cupy as _cp_absorb  # type: ignore[import-not-found]
        except Exception:
            _cp_absorb = None
        # Force direct path when h1e is on GPU (tensor path requires numpy).
        _h1e_is_cupy = _cp_absorb is not None and isinstance(h1e, _cp_absorb.ndarray)
        if absorb_h1e_mode in ("direct", "op", "wrapper") or backend in ("cuda_eri_mat", "cuda") or _h1e_is_cupy:
            # Keep h1e as-is (numpy or CuPy) — contract_2e CUDA path handles both.
            if _h1e_is_cupy:
                _h1e_stored = _cp_absorb.asarray(h1e, dtype=_cp_absorb.float64)
            else:
                _h1e_stored = np.asarray(h1e, dtype=np.float64)
            return H1E2EContractOp(h1e=_h1e_stored, eri=eri, fac=float(fac))
        if isinstance(eri, (DFMOIntegrals, DeviceDFMOIntegrals)):
            raise NotImplementedError(
                "absorb_h1e is not supported for DFMOIntegrals/DeviceDFMOIntegrals; pass h1e explicitly to "
                "kernel/contract_2e, or materialize dense ERIs via df_eri.to_eri4() for validation"
            )
        from asuka.cuguga.eri import restore_eri1, restore_eri4  # noqa: PLC0415

        h1e = np.asarray(h1e, dtype=np.float64)
        # Do not mutate the input `eri` in-place. PySCF's Newton-CASSCF builds
        # a Hessian operator closure that reuses the same ERI buffers across
        # many `h_op` calls; in-place modification would make the operator
        # stateful/non-linear and break Z-vector / AH solves.
        eri_arr = np.asarray(eri, dtype=np.float64)
        if eri_arr.ndim == 4:
            if eri_arr.shape != (int(norb), int(norb), int(norb), int(norb)):
                raise ValueError("eri has wrong shape")
            eri4 = np.array(eri_arr, dtype=np.float64, copy=True, order="C")
        else:
            eri4 = restore_eri1(eri_arr, int(norb))

        if not isinstance(nelec, (int, np.integer)):
            nelec_total = int(np.sum(np.asarray(nelec, dtype=np.int64)))
        else:
            nelec_total = int(nelec)

        # Match PySCF `direct_spin1.absorb_h1e` convention.
        f1e = h1e - np.einsum("jiik->jk", eri4, optimize=True) * 0.5
        f1e = f1e * (1.0 / (float(nelec_total) + 1e-100))
        for k in range(int(norb)):
            eri4[k, k, :, :] += f1e
            eri4[:, :, k, k] += f1e
        return restore_eri4(eri4, int(norb)) * float(fac)

    def contract_2e(self, eri, civec, norb: int, nelec: int | tuple[int, int], *, return_cupy: bool = False, **kwargs):
        scale = 1.0
        if isinstance(eri, H1E2EContractOp):
            # Our CSF "absorbed-ERI" contraction backend (`contract_eri_epq_eqrs_multi`) implements
            #   Σ_pqrs eri[pqrs] E_pq E_rs
            # i.e. it does *not* include the Hamiltonian's usual 1/2 prefactor on the two-body term.
            # In PySCF, `absorb_h1e(..., fac=0.5)` is used to supply that missing 1/2. When we bypass
            # the absorbed tensor and contract the physical Hamiltonian directly, we therefore need
            # to rescale by (fac / 0.5) to reproduce the absorbed-tensor convention.
            scale = float(eri.fac) / 0.5
            kwargs = dict(kwargs)
            kwargs.setdefault("h1e", eri.h1e)
            eri = eri.eri

        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        # PySCF's symmetry-enabled Newton-CASSCF calls `contract_2e` without passing `orbsym/wfnsym`
        # even though the preceding `kernel()` call did.  Default to the solver attributes and fall
        # back to the last-kernel DRT metadata if needed to keep CI-vector sizes consistent.
        drt_key, drt = self._drt_for_ci(
            civec,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        # If h1e is provided (non-standard path), evaluate the full Hamiltonian.
        # Otherwise assume `eri` has already absorbed the 1e part (PySCF convention).
        h1e = kwargs.get("h1e", None)
        backend = (
            kwargs.get("contract_2e_backend", None)
            or getattr(self, "contract_2e_backend", None)
            or getattr(self, "matvec_backend", "contract")
        )
        use_aggregate_offdiag: bool | None = None
        if backend in ("cuda_eri_mat", "cuda"):
            aggregate_offdiag_in = kwargs.get(
                "matvec_cuda_aggregate_offdiag",
                getattr(self, "matvec_cuda_aggregate_offdiag", None),
            )
            if aggregate_offdiag_in is None:
                use_aggregate_offdiag = True
            else:
                use_aggregate_offdiag = bool(aggregate_offdiag_in)
            use_aggregate_offdiag = _enforce_cuda_aggregate_offdiag_guard(
                bool(use_aggregate_offdiag),
                context="contract_2e(cuda)",
            )
        contract_nthreads = int(getattr(self, "contract_nthreads", 0))
        if contract_nthreads <= 0:
            contract_nthreads = _auto_num_threads()
        contract_blas_nthreads = getattr(self, "contract_blas_nthreads", None)
        if contract_blas_nthreads is not None:
            contract_blas_nthreads = int(contract_blas_nthreads)
            if contract_blas_nthreads <= 0:
                contract_blas_nthreads = None
        contract_ws = self._contract_ws_cache.get(drt_key)
        if contract_ws is None:
            from asuka.contract import ContractWorkspace as _ContractWorkspace  # noqa: PLC0415

            contract_ws = _ContractWorkspace()
            self._contract_ws_cache[drt_key] = contract_ws

        if backend in ("cuda_eri_mat", "cuda"):
            if h1e is None:
                raise ValueError(
                    "contract_2e backend 'cuda_eri_mat' requires explicit h1e; set "
                    "`fcisolver.absorb_h1e_mode='direct'` so PySCF passes (h1e,eri) through."
                )
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception as e:  # pragma: no cover
                raise RuntimeError("contract_2e backend 'cuda_eri_mat' requires CuPy") from e

            _contract_cuda_mem_ctl = _resolve_cuda_memory_controls_runtime(
                kwargs=kwargs,
                defaults=self,
                consume=False,
            )
            contract_mem_hard_cap_gib = float(_contract_cuda_mem_ctl["matvec_cuda_mem_hard_cap_gib"])
            contract_ws_cache_fraction = float(_contract_cuda_mem_ctl["matvec_cuda_ws_cache_fraction"])
            _apply_cuda_pool_hard_cap(cp, float(contract_mem_hard_cap_gib))
            self._configure_matvec_cuda_ws_cache(
                cp_mod=cp,
                hard_cap_gib=float(contract_mem_hard_cap_gib),
                fraction=float(contract_ws_cache_fraction),
            )

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                GugaMatvecEriMatWorkspace,
                has_cuda_ext,
                make_device_drt,
                make_device_state_cache,
            )

            if not has_cuda_ext():
                raise RuntimeError(
                    "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
                )

            ws_state_key = drt_key
            drt_dev, state_dev = _get_or_create_cuda_matvec_state_runtime(
                state_cache=self._matvec_cuda_state_cache,
                ws_key=ws_state_key,
                drt=drt,
                make_device_drt_fn=make_device_drt,
                make_device_state_cache_fn=make_device_state_cache,
            )

            norb_i = int(drt.norb)
            nops = norb_i * norb_i

            # Mixed-precision: when fp32_coeff_data is set, build workspace in FP32
            # for TF32 GEMM acceleration (used by AH orbital optimizer).
            _use_fp32 = bool(getattr(self, "matvec_cuda_fp32_coeff_data", False))
            _ws_dtype = cp.float32 if _use_fp32 else cp.float64
            _ws_dtype_name = "float32" if _use_fp32 else "float64"
            ws_key = (drt_key, str(_ws_dtype_name), "contract_2e")

            # Avoid repeated device uploads when PySCF calls absorb_h1e/contract_2e inside a fixed Hessian operator.
            integral_key = (id(eri), id(h1e))
            cuda_ws = self._matvec_cuda_ws_cache_get(ws_key)
            _path_mode_req = _normalize_matvec_cuda_path_mode(getattr(self, "matvec_cuda_path_mode", "auto"))
            if (
                cuda_ws is None
                or getattr(cuda_ws, "_contract_2e_integral_key", None) != integral_key
                or str(getattr(cuda_ws, "path_mode_requested", "auto")) != str(_path_mode_req)
            ):
                if isinstance(eri, DeviceDFMOIntegrals):
                    if eri.eri_mat is None:
                        raise ValueError("DeviceDFMOIntegrals requires eri_mat for CUDA contract_2e")
                    eri_mat_d = cp.ascontiguousarray(cp.asarray(eri.eri_mat, dtype=_ws_dtype))
                    j_ps_d = cp.ascontiguousarray(cp.asarray(eri.j_ps, dtype=_ws_dtype))
                    # h1e may be numpy or CuPy — cp.asarray handles both.
                    h1e_d = cp.asarray(h1e, dtype=_ws_dtype)
                    h_eff_d = h1e_d - 0.5 * j_ps_d
                elif isinstance(eri, cp.ndarray) and eri.ndim == 4:
                    # CuPy 4D dense eri — stay on device, no host roundtrip.
                    eri4_d = cp.ascontiguousarray(eri.astype(cp.float64, copy=False))
                    eri_mat_d = cp.ascontiguousarray(eri4_d.reshape(nops, nops).astype(_ws_dtype, copy=False))
                    j_ps_d = cp.einsum("pqqs->ps", eri4_d, optimize=True).astype(_ws_dtype, copy=False)
                    h1e_d = cp.asarray(h1e, dtype=_ws_dtype)
                    h_eff_d = h1e_d - 0.5 * j_ps_d
                else:
                    if isinstance(eri, DFMOIntegrals):
                        eri_mat_host = eri._maybe_build_eri_mat(eri_mat_max_bytes=1 << 62)
                        if eri_mat_host is None:
                            raise RuntimeError("DFMOIntegrals could not materialize ERI_mat for CUDA contract_2e")
                        j_ps = np.asarray(eri.j_ps, dtype=np.float64, order="C")
                    else:
                        eri4 = _restore_eri_4d(eri, norb_i).astype(np.float64, copy=False)
                        eri_mat_host = np.asarray(eri4.reshape(nops, nops), dtype=np.float64, order="C")
                        j_ps = np.einsum("pqqs->ps", eri4).astype(np.float64, copy=False)

                    h_eff = np.asarray(h1e, dtype=np.float64) - 0.5 * np.asarray(j_ps, dtype=np.float64)
                    eri_mat_d = cp.ascontiguousarray(cp.asarray(eri_mat_host, dtype=_ws_dtype))
                    h_eff_d = cp.asarray(h_eff, dtype=_ws_dtype)

                use_epq_table_in = getattr(self, "matvec_cuda_use_epq_table", None)
                if use_epq_table_in is None:
                    use_epq_table = bool(int(norb_i) <= 16 and int(drt.ncsf) <= 300_000)
                else:
                    use_epq_table = bool(use_epq_table_in)
                if use_aggregate_offdiag is None:  # pragma: no cover
                    raise RuntimeError("internal error: aggregate-offdiag guard state not initialized")
                if cuda_ws is None:
                    cuda_ws = GugaMatvecEriMatWorkspace(
                        drt,
                        drt_dev=drt_dev,
                        state_dev=state_dev,
                        eri_mat=eri_mat_d,
                        h_eff=h_eff_d,
                        j_tile=int(getattr(self, "matvec_cuda_j_tile", 1024)) if int(getattr(self, "matvec_cuda_j_tile", 0)) > 0 else 1024,
                        csr_capacity_mult=float(getattr(self, "matvec_cuda_csr_capacity_mult", 2.0)),
                        cache_csr_tiles=bool(getattr(self, "matvec_cuda_cache_csr_tiles", False)),
                        csr_host_cache=_normalize_csr_host_cache_mode(getattr(self, "matvec_cuda_csr_host_cache", "auto")),
                        csr_host_cache_budget_gib=max(0.0, float(getattr(self, "matvec_cuda_csr_host_cache_budget_gib", 4.0))),
                        csr_host_cache_min_ncsf=max(1, int(getattr(self, "matvec_cuda_csr_host_cache_min_ncsf", 1_000_000))),
                        csr_pipeline_streams=getattr(self, "matvec_cuda_csr_pipeline_streams", "auto"),
                        csr_pipeline_min_ncsf=max(1, int(getattr(self, "matvec_cuda_csr_pipeline_min_ncsf", 1_000_000))),
                        prefilter_trivial_tasks=_normalize_prefilter_trivial_tasks_mode(
                            getattr(self, "matvec_cuda_prefilter_trivial_tasks", "auto")
                        ),
                        prefilter_trivial_tasks_min_ncsf=max(
                            1, int(getattr(self, "matvec_cuda_prefilter_trivial_tasks_min_ncsf", 1_000_000))
                        ),
                        threads_enum=int(getattr(self, "matvec_cuda_threads_enum", 128)),
                        threads_g=int(getattr(self, "matvec_cuda_threads_g", 256)),
                        threads_w=int(getattr(self, "matvec_cuda_threads_w", 0)) if int(getattr(self, "matvec_cuda_threads_w", 0)) > 0 else int(getattr(self, "matvec_cuda_threads_g", 256)),
                        threads_apply=int(getattr(self, "matvec_cuda_threads_apply", 32)) if int(getattr(self, "matvec_cuda_threads_apply", 0)) > 0 else 32,
                        max_g_bytes=int(float(getattr(self, "matvec_cuda_max_g_mib", 256.0)) * 1024 * 1024),
                        coalesce=bool(getattr(self, "matvec_cuda_coalesce", True)),
                        include_diagonal_rs=bool(getattr(self, "matvec_cuda_include_diagonal_rs", True)),
                        fuse_count_write=bool(getattr(self, "matvec_cuda_fuse_count_write", True)),
                        path_mode=str(_path_mode_req),
                        use_fused_hop=bool(getattr(self, "matvec_cuda_use_fused_hop", True)),
                        fp32_coeff_data=bool(getattr(self, "matvec_cuda_fp32_coeff_data", False)),
                        use_epq_table=use_epq_table,
                        aggregate_offdiag_k=bool(use_aggregate_offdiag),
                        offdiag_enable_fp64_emulation=bool(getattr(self, "matvec_cuda_enable_fp64_emulation", False)),
                        offdiag_emulation_strategy=str(getattr(self, "matvec_cuda_emulation_strategy", "performant")),
                        offdiag_cublas_workspace_cap_mb=int(getattr(self, "matvec_cuda_cublas_workspace_cap_mb", 2048)),
                        gemm_backend=str(getattr(self, "matvec_cuda_gemm_backend", "gemmex_fp64")),
                        dtype=_ws_dtype,
                        apply_mode=str(getattr(self, "matvec_cuda_apply_mode", "auto")),
                        epq_build_nthreads=int(getattr(self, "matvec_cuda_epq_build_nthreads", 0)),
                        epq_build_device=bool(getattr(self, "matvec_cuda_epq_build_device", False)),
                        epq_build_j_tile=int(getattr(self, "matvec_cuda_epq_build_j_tile", 0)),
                        use_cuda_graph=bool(getattr(self, "matvec_cuda_use_graph", False)),
                    )
                    self._matvec_cuda_ws_cache_put(
                        ws_key,
                        cuda_ws,
                        keep_keys=(ws_key,),
                    )
                else:
                    cuda_ws.eri_mat = eri_mat_d
                    h_eff_flat_new = cuda_ws._as_h_eff_flat(h_eff_d)
                    if getattr(cuda_ws, "h_eff_flat", None) is None or tuple(getattr(cuda_ws.h_eff_flat, "shape", ())) != tuple(
                        getattr(h_eff_flat_new, "shape", ())
                    ):
                        cuda_ws.h_eff_flat = h_eff_flat_new
                        if getattr(cuda_ws, "_cuda_graph", None) is not None:
                            cuda_ws._cuda_graph = None
                            cuda_ws._cuda_graph_x = None
                            cuda_ws._cuda_graph_y = None
                    else:
                        cp.copyto(cuda_ws.h_eff_flat, h_eff_flat_new)
                    cuda_ws._eri_diag_t = None
                    if getattr(cuda_ws, "_eri_mat_t", None) is not None:
                        cp.copyto(cuda_ws._eri_mat_t, cuda_ws.eri_mat.T)
                    elif getattr(cuda_ws, "_cuda_graph", None) is not None:
                        cuda_ws._cuda_graph = None
                        cuda_ws._cuda_graph_x = None
                        cuda_ws._cuda_graph_y = None

                cuda_ws._contract_2e_integral_key = integral_key

            # Accept both numpy and CuPy civec — avoid roundtrip if already on device.
            if isinstance(civec, cp.ndarray):
                x_d = cp.ascontiguousarray(civec.astype(_ws_dtype, copy=False).ravel())
            else:
                x_d = cp.ascontiguousarray(cp.asarray(np.asarray(civec, dtype=np.float64), dtype=_ws_dtype).ravel())
            y_d = cuda_ws.hop(x_d, sync=True, check_overflow=True)
            if return_cupy:
                out_d = y_d.astype(cp.float64, copy=False)
                if scale != 1.0:
                    out_d = out_d * scale
                return out_d
            out = cp.asnumpy(y_d)
            if scale != 1.0:
                out = np.asarray(out, dtype=np.float64) * scale
            return np.asarray(out, dtype=np.float64)

        if isinstance(eri, DeviceDFMOIntegrals):
            if h1e is None:
                h1e = getattr(self, "_h1e", None)
            if h1e is None:
                raise ValueError("DeviceDFMOIntegrals requires h1e (pass `h1e=` or call `kernel` first)")
            if eri.eri_mat is None:
                raise ValueError("DeviceDFMOIntegrals requires eri_mat for contract_2e")
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception:
                cp = None
            eri_mat = eri.eri_mat
            if cp is not None and isinstance(eri_mat, cp.ndarray):
                eri_mat_h = cp.asnumpy(eri_mat)
            else:
                eri_mat_h = np.asarray(eri_mat)
            eri4 = np.asarray(eri_mat_h, dtype=np.float64, order="C").reshape(norb, norb, norb, norb)
            from asuka.contract import contract_h_csf_multi as _contract_h_csf_multi  # noqa: PLC0415

            out = _contract_h_csf_multi(
                drt,
                h1e,
                eri4,
                [np.asarray(civec)],
                precompute_epq=True,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
                workspace=contract_ws,
            )[0]
            if scale != 1.0:
                out = np.asarray(out, dtype=np.float64) * scale
            return out
        if isinstance(eri, DFMOIntegrals):
            if h1e is None:
                h1e = getattr(self, "_h1e", None)
            if h1e is None:
                raise ValueError("DFMOIntegrals requires h1e (pass `h1e=` or call `kernel` first)")
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_h_csf_multi_df  # noqa: PLC0415

            out = _contract_h_csf_multi_df(
                drt,
                h1e,
                eri,
                [np.asarray(civec)],
                precompute_epq=True,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
            )[0]
        elif h1e is None:
            from asuka.contract import contract_eri_epq_eqrs_multi as _contract_eri_epq_eqrs_multi  # noqa: PLC0415

            out = _contract_eri_epq_eqrs_multi(
                drt,
                eri,
                [np.asarray(civec)],
                precompute_epq=True,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
                workspace=contract_ws,
            )[0]
        else:
            from asuka.contract import contract_h_csf_multi as _contract_h_csf_multi  # noqa: PLC0415

            out = _contract_h_csf_multi(
                drt,
                h1e,
                eri,
                [np.asarray(civec)],
                precompute_epq=True,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
                workspace=contract_ws,
            )[0]
        if scale != 1.0:
            out = np.asarray(out, dtype=np.float64) * scale
        return out

    def make_hdiag(self, h1e, eri, norb: int, nelec: int | tuple[int, int], **kwargs) -> np.ndarray:
        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        drt = self._get_drt(
            norb,
            nelec_total,
            twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        ncsf = int(drt.ncsf)
        if isinstance(eri, DFMOIntegrals):
            # Fast vectorized diagonal guess for DF integrals, matching df_diag.diagonal_element_det_guess_df.
            # This avoids an O(ncsf) Python loop (which can dominate runtime when using DFMOIntegrals).
            l_full = np.asarray(eri.l_full, dtype=np.float64, order="C")
            pair_norm = np.asarray(eri.pair_norm, dtype=np.float64, order="C")

            diag_ids = (np.arange(norb, dtype=np.int32) * (norb + 1)).astype(np.int32, copy=False)
            l_diag = np.asarray(l_full[diag_ids], dtype=np.float64, order="C")
            eri_ppqq = l_diag @ l_diag.T  # (p p| q q)
            # In the symmetric DF representation built from s2, d[L,pq]==d[L,qp], so:
            #   (p q| q p) = sum_L d[L,pq] * d[L,qp] = sum_L d[L,pq]^2 = ||d[:,pq]||^2
            eri_pqqp = np.square(pair_norm.reshape(norb, norb))

            cache = get_state_cache(drt)
            steps = np.asarray(cache.steps, dtype=np.int8)
            if steps.shape != (ncsf, norb):
                raise RuntimeError("internal error: invalid cached steps table shape")
            occ = _STEP_TO_OCC[steps]

            doubly = occ == 2
            singles = occ == 1

            neleca_det = (nelec_total + twos) // 2
            nelecb_det = nelec_total - neleca_det

            ndoubly = np.sum(doubly, axis=1, dtype=np.int32)
            alpha_need = np.asarray(neleca_det, dtype=np.int32) - ndoubly

            single_prefix = np.cumsum(singles, axis=1, dtype=np.int32)
            alpha_single = singles & (single_prefix <= alpha_need[:, None])
            beta_single = singles & (~alpha_single)

            alpha = (doubly | alpha_single).astype(np.float64)
            beta = (doubly | beta_single).astype(np.float64)
            n = alpha + beta

            h1e = np.asarray(h1e, dtype=np.float64)
            if h1e.shape != (norb, norb):
                raise ValueError("h1e has wrong shape")
            h1e_diag = np.diag(h1e)

            hdiag = n @ h1e_diag
            tmp = n @ eri_ppqq
            hdiag += 0.5 * np.sum(tmp * n, axis=1)
            tmp_a = alpha @ eri_pqqp
            hdiag += -0.5 * np.sum(tmp_a * alpha, axis=1)
            tmp_b = beta @ eri_pqqp
            hdiag += -0.5 * np.sum(tmp_b * beta, axis=1)
            return np.asarray(hdiag, dtype=np.float64)

        if isinstance(eri, DeviceDFMOIntegrals):
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except Exception:
                cp = None

            if eri.eri_mat is None:
                l_full = eri.l_full
                pair_norm = eri.pair_norm
                if l_full is None:
                    raise ValueError("DeviceDFMOIntegrals requires either eri_mat or l_full for make_hdiag")

                if cp is not None and isinstance(l_full, cp.ndarray):
                    l_full = cp.asnumpy(l_full)
                if pair_norm is not None and cp is not None and isinstance(pair_norm, cp.ndarray):
                    pair_norm = cp.asnumpy(pair_norm)

                l_full = np.asarray(l_full, dtype=np.float64, order="C")
                if pair_norm is None:
                    pair_norm = np.linalg.norm(l_full.reshape(norb, norb, -1), axis=2)
                else:
                    pair_norm = np.asarray(pair_norm, dtype=np.float64, order="C")

                diag_ids = (np.arange(norb, dtype=np.int32) * (norb + 1)).astype(np.int32, copy=False)
                l_diag = np.asarray(l_full[diag_ids], dtype=np.float64, order="C")
                eri_ppqq = l_diag @ l_diag.T
                eri_pqqp = np.square(pair_norm.reshape(norb, norb))
            else:
                eri_mat = eri.eri_mat
                if cp is not None and isinstance(eri_mat, cp.ndarray):
                    eri_mat_h = cp.asnumpy(eri_mat)
                else:
                    eri_mat_h = np.asarray(eri_mat)
                eri4 = np.asarray(eri_mat_h, dtype=np.float64, order="C").reshape(norb, norb, norb, norb)
                eri_ppqq = np.einsum("iijj->ij", eri4)
                eri_pqqp = np.einsum("ijji->ij", eri4)
        else:
            eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)
            eri_ppqq = np.einsum("iijj->ij", eri4)
            eri_pqqp = np.einsum("ijji->ij", eri4)

        # Fast vectorized diagonal guess from cached DRT steps, matching oracle.diagonal_element_det_guess.
        # This avoids per-CSF Python loops + drt.index_to_path overhead.
        cache = get_state_cache(drt)
        steps = np.asarray(cache.steps, dtype=np.int8)
        if steps.shape != (ncsf, norb):
            raise RuntimeError("internal error: invalid cached steps table shape")
        occ = _STEP_TO_OCC[steps]

        doubly = occ == 2
        singles = occ == 1

        neleca_det = (nelec_total + twos) // 2
        nelecb_det = nelec_total - neleca_det

        ndoubly = np.sum(doubly, axis=1, dtype=np.int32)
        alpha_need = np.asarray(neleca_det, dtype=np.int32) - ndoubly

        # Assign the first `alpha_need` single-occupied orbitals (in orbital index order) to alpha; the rest to beta.
        single_prefix = np.cumsum(singles, axis=1, dtype=np.int32)
        alpha_single = singles & (single_prefix <= alpha_need[:, None])
        beta_single = singles & (~alpha_single)

        alpha = (doubly | alpha_single).astype(np.float64)
        beta = (doubly | beta_single).astype(np.float64)
        n = alpha + beta

        h1e = np.asarray(h1e, dtype=np.float64)
        if h1e.shape != (norb, norb):
            raise ValueError("h1e has wrong shape")
        h1e_diag = np.diag(h1e)

        hdiag = n @ h1e_diag
        tmp = n @ eri_ppqq
        hdiag += 0.5 * np.sum(tmp * n, axis=1)
        tmp_a = alpha @ eri_pqqp
        hdiag += -0.5 * np.sum(tmp_a * alpha, axis=1)
        tmp_b = beta @ eri_pqqp
        hdiag += -0.5 * np.sum(tmp_b * beta, axis=1)
        return np.asarray(hdiag, dtype=np.float64)

    def pspace(self, h1e, eri, norb: int, nelec: int | tuple[int, int], **kwargs):
        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        drt = self._get_drt(
            norb,
            nelec_total,
            twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        if isinstance(eri, DFMOIntegrals):
            from asuka.integrals.oracle_df import connected_row_df as connected_row
        elif isinstance(eri, DeviceDFMOIntegrals):
            # DeviceDFMOIntegrals lacks the CPU DF interface needed by connected_row_df.
            # Materialize dense ERIs from eri_mat and use the standard connected_row.
            try:
                import cupy as _cp_ps  # type: ignore[import-not-found]
            except Exception:
                _cp_ps = None
            if eri.eri_mat is not None:
                _eri_mat = eri.eri_mat
                if _cp_ps is not None and isinstance(_eri_mat, _cp_ps.ndarray):
                    _eri_mat = _cp_ps.asnumpy(_eri_mat)
                eri = np.asarray(_eri_mat, dtype=np.float64).reshape(norb, norb, norb, norb)
            else:
                raise ValueError("DeviceDFMOIntegrals requires eri_mat for pspace")
            from asuka.cuguga.oracle import connected_row
        else:
            from asuka.cuguga.oracle import connected_row

        ncsf = int(drt.ncsf)
        npsp = int(kwargs.get("npsp", kwargs.get("pspace_size", 0)))
        if npsp <= 0:
            return np.zeros(0, dtype=np.int32), np.zeros((0, 0), dtype=np.float64)

        max_out = int(kwargs.get("max_out", 200_000))

        hdiag = kwargs.get("hdiag")
        if hdiag is None:
            hdiag = self.make_hdiag(h1e, eri, norb, nelec, **kwargs)

        npsp = min(npsp, ncsf)
        addr = np.argsort(np.asarray(hdiag, dtype=np.float64))[:npsp].astype(np.int32, copy=False)

        loc = -np.ones(ncsf, dtype=np.int32)
        loc[addr] = np.arange(npsp, dtype=np.int32)

        h0 = np.zeros((npsp, npsp), dtype=np.float64)
        for col, j in enumerate(addr.tolist()):
            i_idx, hij = connected_row(drt, h1e, eri, int(j), max_out=max_out)
            for i, v in zip(i_idx.tolist(), hij.tolist()):
                row = int(loc[int(i)])
                if row >= 0:
                    h0[row, col] = float(v)
        h0 = 0.5 * (h0 + h0.T)
        return addr, h0

    def make_rdm12(self, civec, norb: int, nelec: int | tuple[int, int], **kwargs):
        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        drt_key, drt = self._drt_for_ci(
            civec,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        rdm_backend = str(kwargs.get("rdm_backend", getattr(self, "rdm_backend", "auto"))).strip().lower()
        matvec_backend = str(getattr(self, "matvec_backend", "contract")).strip().lower()
        strict_gpu = bool(kwargs.get("strict_gpu", getattr(self, "strict_gpu", False)))
        rdm_block_nops = int(kwargs.get("rdm_block_nops", getattr(self, "rdm_block_nops", 8)))
        rdm_tmpdir = kwargs.get("rdm_tmpdir", getattr(self, "rdm_tmpdir", None))

        max_memory = float(kwargs.get("max_memory", getattr(self, "max_memory", 4000.0)))
        ncsf = int(drt.ncsf)
        nops = norb * norb
        t_bytes = float(nops) * float(ncsf) * 8.0
        auto_selected_cuda = False

        if strict_gpu and matvec_backend.startswith("cuda"):
            if rdm_backend == "auto":
                rdm_backend = "cuda"
                auto_selected_cuda = True
            elif rdm_backend != "cuda":
                raise ValueError("strict_gpu=True requires rdm_backend='cuda' for CUDA matvec workflows")

        if rdm_backend == "auto":
            # Prefer GPU RDM builds when the operator space is large enough that CPU setup
            # dominates end-to-end runtime.
            #
            # Keep a conservative threshold: for tiny problems the CPU path can be faster due
            # to GPU launch/transfer overhead.
            if matvec_backend.startswith("cuda"):
                want_cuda = bool(ncsf >= 4096 or t_bytes >= 16.0 * 1024.0 * 1024.0)
                headroom_mult = 1.10
            else:
                want_cuda = bool(ncsf >= 65_536 or t_bytes >= 64.0 * 1024.0 * 1024.0)
                headroom_mult = 1.30

            if want_cuda:
                try:
                    from asuka.cuda.cuda_backend import has_cuda_ext  # noqa: PLC0415

                    if has_cuda_ext():
                        import cupy as cp  # noqa: PLC0415

                        try:
                            ndev = int(cp.cuda.runtime.getDeviceCount())
                        except Exception:
                            ndev = 0

                        if ndev > 0:
                            free_bytes = 0
                            try:
                                free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
                            except Exception:
                                free_bytes = 0

                            if not free_bytes or t_bytes * float(headroom_mult) <= float(free_bytes):
                                rdm_backend = "cuda"
                                auto_selected_cuda = True
                except Exception:
                    pass

        if rdm_backend == "auto":
            # If t_pq fits comfortably in the memory budget, keep the legacy
            # in-RAM path (usually faster on tiny test cases). Otherwise use
            # the streaming/memmap implementation.
            max_bytes = max_memory * 1e6
            rdm_backend = "legacy" if t_bytes <= 0.7 * max_bytes else "stream"

        if rdm_backend == "cuda":
            from asuka.cuda.rdm_gpu import make_rdm12_cuda, make_rdm12_cuda_workspace

            build_threads = int(
                kwargs.get("rdm_cuda_build_threads", getattr(self, "rdm_cuda_build_threads", 256))
            )
            use_epq_table = kwargs.get("rdm_cuda_use_epq_table", getattr(self, "rdm_cuda_use_epq_table", None))
            if use_epq_table is None:
                matvec_backend = str(getattr(self, "matvec_backend", "contract")).strip().lower()
                if matvec_backend.startswith("cuda") and bool(getattr(self, "matvec_cuda_use_epq_table", False)):
                    use_epq_table = True
            enable_emulation = bool(
                kwargs.get(
                    "rdm_cuda_enable_fp64_emulation",
                    getattr(self, "rdm_cuda_enable_fp64_emulation", False),
                )
            )
            gemm_backend = str(kwargs.get("rdm_cuda_gemm_backend", getattr(self, "rdm_cuda_gemm_backend", "gemmex_fp64")))
            math_mode = str(kwargs.get("rdm_cuda_math_mode", getattr(self, "rdm_cuda_math_mode", "default")))
            if enable_emulation:
                gemm_backend = "gemmex_emulated_fixedpoint"
                math_mode = "fp64_emulated_fixedpoint"

            cublas_workspace_mb = int(
                kwargs.get("rdm_cuda_cublas_workspace_mb", getattr(self, "rdm_cuda_cublas_workspace_mb", 0))
            )
            emulation_strategy = kwargs.get(
                "rdm_cuda_emulation_strategy",
                getattr(self, "rdm_cuda_emulation_strategy", None),
            )
            mantissa_control = kwargs.get(
                "rdm_cuda_fixed_point_mantissa_control",
                getattr(self, "rdm_cuda_fixed_point_mantissa_control", None),
            )
            max_bits = kwargs.get(
                "rdm_cuda_fixed_point_max_mantissa_bits",
                getattr(self, "rdm_cuda_fixed_point_max_mantissa_bits", None),
            )
            bit_offset = kwargs.get(
                "rdm_cuda_fixed_point_mantissa_bit_offset",
                getattr(self, "rdm_cuda_fixed_point_mantissa_bit_offset", None),
            )
            symmetrize_gram = bool(
                kwargs.get("rdm_cuda_symmetrize_gram", getattr(self, "rdm_cuda_symmetrize_gram", True))
            )
            streaming_ncsf_cutoff = int(
                kwargs.get(
                    "rdm_cuda_streaming_ncsf_cutoff",
                    getattr(self, "rdm_cuda_streaming_ncsf_cutoff", 2_000_000),
                )
            )
            if streaming_ncsf_cutoff < 0:
                streaming_ncsf_cutoff = 0

            cache_key = (
                "cuda",
                drt_key,
                int(id(civec)),
                int(build_threads),
                None if use_epq_table is None else bool(use_epq_table),
                str(gemm_backend),
                str(math_mode),
                int(cublas_workspace_mb),
                None if emulation_strategy is None else str(emulation_strategy),
                None if mantissa_control is None else str(mantissa_control),
                None if max_bits is None else int(max_bits),
                None if bit_offset is None else int(bit_offset),
                bool(symmetrize_gram),
                int(streaming_ncsf_cutoff),
            )
            if self._rdm12_cache_key == cache_key and self._rdm12_cache_val is not None:
                return self._rdm12_cache_val

            # Ensure we have (steps,nodes) available (but avoid building the per-(p,q) CSR cache).
            get_state_cache(drt)

            ws = self._rdm_cuda_ws_cache.get(drt_key)
            if ws is None:
                ws = make_rdm12_cuda_workspace(
                    drt,
                    gemm_backend=gemm_backend,
                    math_mode=math_mode,
                    cublas_workspace_mb=cublas_workspace_mb,
                    emulation_strategy=None if emulation_strategy is None else str(emulation_strategy),
                    fixed_point_mantissa_control=None if mantissa_control is None else str(mantissa_control),
                    fixed_point_max_mantissa_bits=None if max_bits is None else int(max_bits),
                    fixed_point_mantissa_bit_offset=None if bit_offset is None else int(bit_offset),
                )
                self._rdm_cuda_ws_cache[drt_key] = ws

            try:
                out = make_rdm12_cuda(
                    drt,
                    np.asarray(civec, dtype=np.float64),
                    workspace=ws,
                    block_nops=rdm_block_nops,
                    build_threads=build_threads,
                    use_epq_table=None if use_epq_table is None else bool(use_epq_table),
                    gemm_backend=gemm_backend,
                    math_mode=math_mode,
                    cublas_workspace_mb=cublas_workspace_mb,
                    emulation_strategy=None if emulation_strategy is None else str(emulation_strategy),
                    fixed_point_mantissa_control=None if mantissa_control is None else str(mantissa_control),
                    fixed_point_max_mantissa_bits=None if max_bits is None else int(max_bits),
                    fixed_point_mantissa_bit_offset=None if bit_offset is None else int(bit_offset),
                    symmetrize_gram=symmetrize_gram,
                    streaming_ncsf_cutoff=int(streaming_ncsf_cutoff),
                )
            except Exception as e:
                if strict_gpu and matvec_backend.startswith("cuda"):
                    raise RuntimeError("strict_gpu=True forbids CUDA RDM fallback to CPU") from e
                if not auto_selected_cuda:
                    raise
                # Fallback to CPU for auto-selected CUDA backends (e.g. OOM on small GPUs).
                max_bytes = max_memory * 1e6
                rdm_backend = "legacy" if t_bytes <= 0.7 * max_bytes else "stream"
            else:
                self._rdm12_cache_key = cache_key
                self._rdm12_cache_val = out
                return out

        if strict_gpu and matvec_backend.startswith("cuda") and rdm_backend != "cuda":
            raise RuntimeError("strict_gpu=True forbids non-CUDA RDM backends in CUDA matvec workflows")

        # CPU backends rely on the cached per-(p,q) actions.
        rdm_nthreads = int(kwargs.get("rdm_nthreads", getattr(self, "rdm_nthreads", 0)))
        if rdm_nthreads <= 0:
            rdm_nthreads = _auto_num_threads()

        # Threading policy for CPU RDM builds:
        # - dm1 and dm2 use many small/medium BLAS calls; letting OpenBLAS use a very
        #   large global thread count (e.g. defaulting to all cores) can be *much*
        #   slower due to thread management overhead.
        # - Auto: cap BLAS threads for small operator spaces to a modest number.
        rdm_blas_nthreads = kwargs.get("rdm_blas_nthreads", getattr(self, "rdm_blas_nthreads", None))
        if rdm_blas_nthreads is not None:
            rdm_blas_nthreads = int(rdm_blas_nthreads)
            if rdm_blas_nthreads <= 0:
                rdm_blas_nthreads = None
        if rdm_blas_nthreads is None:
            global_blas = openblas_get_num_threads()
            if global_blas is not None:
                if nops <= 512:
                    rdm_blas_nthreads = min(int(global_blas), 8)
                else:
                    rdm_blas_nthreads = int(global_blas)
            elif nops <= 512:
                rdm_blas_nthreads = 1

        cache_key = (
            "cpu",
            str(rdm_backend),
            drt_key,
            int(id(civec)),
            int(rdm_block_nops),
            float(max_memory),
            None if rdm_tmpdir is None else str(rdm_tmpdir),
            int(rdm_nthreads),
            None if rdm_blas_nthreads is None else int(rdm_blas_nthreads),
        )
        if self._rdm12_cache_key == cache_key and self._rdm12_cache_val is not None:
            return self._rdm12_cache_val

        precompute_epq_actions(drt, nthreads=rdm_nthreads)

        if rdm_backend == "stream":
            if rdm_blas_nthreads is None:
                out = make_rdm12_streaming(
                    drt,
                    np.asarray(civec, dtype=np.float64),
                    max_memory_mb=max_memory,
                    block_nops=rdm_block_nops,
                    tmpdir=None if rdm_tmpdir is None else str(rdm_tmpdir),
                )
            else:
                with blas_thread_limit(int(rdm_blas_nthreads)):
                    out = make_rdm12_streaming(
                        drt,
                        np.asarray(civec, dtype=np.float64),
                        max_memory_mb=max_memory,
                        block_nops=rdm_block_nops,
                        tmpdir=None if rdm_tmpdir is None else str(rdm_tmpdir),
                    )
            self._rdm12_cache_key = cache_key
            self._rdm12_cache_val = out
            return out

        if rdm_backend != "legacy":
            raise ValueError(f"unsupported rdm_backend={rdm_backend!r}")

        def _legacy_rdm12() -> tuple[np.ndarray, np.ndarray]:
            # Legacy (reference) implementation: builds t_pq in RAM and uses a Gram product.
            # This path is kept for small systems / debugging; it still avoids occ_table by
            # using the E_pq cache's step table for the diagonal E_pp contribution.
            c = np.asarray(civec, dtype=np.float64).ravel()
            cache_obj = _get_epq_action_cache(drt)
            step_to_occ = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D

            # Fast path: use cached SciPy sparse matrices (when available) for E_pq|c>.
            try:
                from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
                from asuka.contract import _sp as _contract_sp  # noqa: PLC0415
            except Exception:  # pragma: no cover
                _contract_sp = None

            if _contract_sp is not None:
                # SciPy path: we overwrite every row, so avoid the cost of zero-fill.
                t_pq = np.empty((nops, c.size), dtype=np.float64)
                mats = _epq_spmat_list(drt, cache_obj)
                occ = step_to_occ[cache_obj.steps]
                c_col = c.reshape(c.size, 1)
                for p in range(norb):
                    for q in range(norb):
                        pq = p * norb + q
                        out = t_pq[pq]
                        if p == q:
                            np.multiply(occ[:, p], c, out=out)
                            continue
                        mat = mats[pq]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix")
                        if _csc_matmul_dense_inplace_cy is not None:
                            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                mat.indptr, mat.indices, mat.data, c_col, out.reshape(c.size, 1)
                            )
                        else:
                            out[:] = mat.dot(c)  # type: ignore[operator]
            else:
                t_pq = np.zeros((nops, c.size), dtype=np.float64)
                for p in range(norb):
                    for q in range(norb):
                        pq = p * norb + q
                        out = t_pq[pq]
                        if p == q:
                            out[:] = step_to_occ[cache_obj.steps[:, p]] * c
                            continue
                        csr = _csr_for_epq(cache_obj, drt, p, q)
                        for j in range(c.size):
                            start = int(csr.indptr[j])
                            end = int(csr.indptr[j + 1])
                            if start == end:
                                continue
                            out[csr.indices[start:end]] += csr.data[start:end] * c[j]

            # Adjoint convention matches make_rdm12_streaming:
            # dm1[p,q] = <E_{q p}>. For real CSFs this is symmetric.
            dm1 = (t_pq @ c).reshape(norb, norb).T

            # gram[pq,rs] = <E_pq E_rs> = dot((E_{q p}|c>), (E_rs|c>)).
            # Avoid the large fancy-indexing copy `t_pq[swap]` by swapping the first
            # index after forming the (small) Gram matrix.
            swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
            gram0 = t_pq @ t_pq.T
            gram = gram0[swap]
            dm2 = gram.reshape(norb, norb, norb, norb)
            for p in range(norb):
                for q in range(norb):
                    dm2[p, q, q, :] -= dm1[:, p]
            return dm1, dm2

        if rdm_blas_nthreads is None:
            out = _legacy_rdm12()
        else:
            with blas_thread_limit(int(rdm_blas_nthreads)):
                out = _legacy_rdm12()
        self._rdm12_cache_key = cache_key
        self._rdm12_cache_val = out
        return out

    def make_rdm123(self, civec, norb: int, nelec: int | tuple[int, int], **kwargs):
        """Compute (dm1, dm2, dm3) in the same conventions as :meth:`make_rdm12`.

        Notes
        -----
        This is a reference-oriented CPU implementation intended for small active spaces.

        The returned tensors follow the `make_rdm12` conventions:
          - dm1[p,q] = <E_{q p}>
          - dm2 matches `make_rdm12` (delta-corrected / reordered if `reorder=True`)
          - dm3 matches `asuka.rdm.rdm123._make_rdm123_pyscf` with the same `reorder` flag
        """

        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        drt_key, drt = self._drt_for_ci(
            civec,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )

        max_memory = float(kwargs.get("max_memory", getattr(self, "max_memory", 4000.0)))
        reorder = bool(kwargs.get("reorder", True))

        # CPU backend: rely on cached per-(p,q) actions (same policy as make_rdm12).
        rdm_nthreads = int(kwargs.get("rdm_nthreads", getattr(self, "rdm_nthreads", 0)))
        if rdm_nthreads <= 0:
            rdm_nthreads = _auto_num_threads()

        rdm_blas_nthreads = kwargs.get("rdm_blas_nthreads", getattr(self, "rdm_blas_nthreads", None))
        if rdm_blas_nthreads is not None:
            rdm_blas_nthreads = int(rdm_blas_nthreads)
            if rdm_blas_nthreads <= 0:
                rdm_blas_nthreads = None
        if rdm_blas_nthreads is None:
            global_blas = openblas_get_num_threads()
            if global_blas is not None:
                if int(norb * norb) <= 512:
                    rdm_blas_nthreads = min(int(global_blas), 8)
                else:
                    rdm_blas_nthreads = int(global_blas)
            elif int(norb * norb) <= 512:
                rdm_blas_nthreads = 1

        precompute_epq_actions(drt, nthreads=rdm_nthreads)

        from asuka.rdm.rdm123 import _make_rdm123_pyscf  # noqa: PLC0415

        if rdm_blas_nthreads is None:
            dm1, dm2, dm3 = _make_rdm123_pyscf(
                drt,
                np.asarray(civec, dtype=np.float64),
                max_memory_mb=max_memory,
                reorder=reorder,
            )
        else:
            with blas_thread_limit(int(rdm_blas_nthreads)):
                dm1, dm2, dm3 = _make_rdm123_pyscf(
                    drt,
                    np.asarray(civec, dtype=np.float64),
                    max_memory_mb=max_memory,
                    reorder=reorder,
                )

        # Match make_rdm12's adjoint convention: dm1[p,q] = <E_{q p}>.
        dm1 = np.asarray(dm1, dtype=np.float64, order="C").T
        dm2 = np.asarray(dm2, dtype=np.float64, order="C")
        dm3 = np.asarray(dm3, dtype=np.float64, order="C")

        # Keep the DRT key hot in the cache for future calls.
        self._last_drt_key = drt_key
        return dm1, dm2, dm3

    def make_rdm1(self, civec, norb: int, nelec: int | tuple[int, int], **kwargs) -> np.ndarray:
        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        _drt_key, drt = self._drt_for_ci(
            civec,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )

        rdm_backend = str(kwargs.get("rdm_backend", getattr(self, "rdm_backend", "auto"))).strip().lower()
        matvec_backend = str(getattr(self, "matvec_backend", "contract")).strip().lower()
        strict_gpu = bool(kwargs.get("strict_gpu", getattr(self, "strict_gpu", False)))
        if strict_gpu and matvec_backend.startswith("cuda"):
            if rdm_backend == "auto":
                rdm_backend = "cuda"
            elif rdm_backend != "cuda":
                raise ValueError("strict_gpu=True requires rdm_backend='cuda' for CUDA matvec workflows")
        if rdm_backend == "auto":
            # Match `make_rdm12` auto selection: when running a CUDA matvec backend on a
            # sufficiently large CSF space, default to the CUDA RDM build.
            ncsf = int(drt.ncsf)
            nops = norb * norb
            t_bytes = float(nops) * float(ncsf) * 8.0
            if matvec_backend.startswith("cuda") and (ncsf >= 4096 or t_bytes >= 16.0 * 1024.0 * 1024.0):
                try:
                    from asuka.cuda.cuda_backend import has_cuda_ext  # noqa: PLC0415

                    if has_cuda_ext():
                        import cupy as cp  # noqa: F401

                        rdm_backend = "cuda"
                except Exception:
                    pass

        # The CUDA backend is implemented in `make_rdm12`, so delegate when requested.
        if rdm_backend == "cuda":
            # Use the class method to bypass `StateAverageFCISolver` mixins which override
            # `make_rdm12` and expect a list/array of CI vectors.
            kwargs2 = dict(kwargs)
            kwargs2["rdm_backend"] = "cuda"
            dm1, _dm2 = GUGAFCISolver.make_rdm12(self, civec, norb, nelec, **kwargs2)
            return dm1

        if strict_gpu and matvec_backend.startswith("cuda"):
            raise RuntimeError("strict_gpu=True forbids non-CUDA make_rdm1 backends in CUDA matvec workflows")

        cache_obj = _get_epq_action_cache(drt)
        c = np.asarray(civec, dtype=np.float64).ravel()

        step_to_occ = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D
        occ = step_to_occ[cache_obj.steps]

        # dm1[p,q] = <c|E_{q p}|c>. For a real CI vector c, the scalar quadratic form
        # satisfies c.T @ A @ c == c.T @ A.T @ c, so we only need one direction (p<q)
        # and can fill the symmetric partner without building both E_pq and E_qp.
        dm1 = np.empty((norb, norb), dtype=np.float64)

        w = c * c
        dm1[np.diag_indices(norb)] = np.sum(occ * w[:, None], axis=0)

        if _csc_quadratic_form_cy is None:
            tmp = np.empty_like(c)
            c_col = c.reshape(c.size, 1)
            tmp_col = tmp.reshape(tmp.size, 1)

        for p in range(norb):
            for q in range(p + 1, norb):
                csr = _csr_for_epq(cache_obj, drt, int(p), int(q))
                if _csc_quadratic_form_cy is not None:
                    val = float(
                        _csc_quadratic_form_cy(  # type: ignore[misc]
                            csr.indptr, csr.indices, csr.data, c
                        )
                    )
                elif _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        csr.indptr, csr.indices, csr.data, c_col, tmp_col
                    )
                    val = float(np.dot(tmp, c))
                else:
                    indptr = csr.indptr
                    indices = csr.indices
                    data = csr.data
                    acc = 0.0
                    for j in range(int(c.size)):
                        cj = float(c[j])
                        if cj == 0.0:
                            continue
                        start = int(indptr[j])
                        end = int(indptr[j + 1])
                        for k in range(start, end):
                            acc += float(data[k]) * float(c[int(indices[k])]) * cj
                    val = float(acc)

                dm1[p, q] = val
                dm1[q, p] = val

        return np.ascontiguousarray(dm1)

    def make_rdm2(self, civec, norb: int, nelec: int | tuple[int, int], **kwargs) -> np.ndarray:
        _dm1, dm2 = GUGAFCISolver.make_rdm12(self, civec, norb, nelec, **kwargs)
        return dm2

    def trans_rdm1(self, ci_bra, ci_ket, norb: int, nelec: int | tuple[int, int], **kwargs) -> np.ndarray:
        """Transition 1-RDM in the active MO basis (spin-free generator convention).

        Convention matches `make_rdm12`:
          dm1[p,q] = <bra| E_{q p} |ket>
        """

        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        _drt_key, drt = self._drt_for_ci(
            ci_bra,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        _validate_civec_shape(ci_ket, int(drt.ncsf))
        cbra = np.asarray(ci_bra, dtype=np.float64).ravel()
        cket = np.asarray(ci_ket, dtype=np.float64).ravel()

        # Ensure the state-cache for diagonal E_pp (steps table) exists. This also enables
        # `_csr_for_epq` to build per-(p,q) actions on demand.
        cache_obj = _get_epq_action_cache(drt)

        step_to_occ = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D
        occ = step_to_occ[cache_obj.steps]

        # dm1[p,q] = <bra|E_{q p}|ket>. Use E_pq for p<q and its transpose relation:
        #   <bra|E_{q p}|ket> = <bra|E_{p q}^T|ket> = <ket|E_{p q}|bra>
        dm1 = np.empty((norb, norb), dtype=np.float64)
        w = cbra * cket
        dm1[np.diag_indices(norb)] = np.sum(occ * w[:, None], axis=0)

        if _csc_bilinear_form_cy is None:
            tmp = np.empty_like(cket)
            cket_col = cket.reshape(cket.size, 1)
            tmp_col = tmp.reshape(tmp.size, 1)

        for p in range(norb):
            for q in range(p + 1, norb):
                csr = _csr_for_epq(cache_obj, drt, int(p), int(q))

                if _csc_bilinear_form_cy is not None:
                    val_pq = float(
                        _csc_bilinear_form_cy(  # type: ignore[misc]
                            csr.indptr, csr.indices, csr.data, cket, cbra
                        )
                    )
                    val_qp = float(
                        _csc_bilinear_form_cy(  # type: ignore[misc]
                            csr.indptr, csr.indices, csr.data, cbra, cket
                        )
                    )
                elif _csc_matmul_dense_inplace_cy is not None:
                    # Build (E_pq|ket>) into tmp, then dot with bra.
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        csr.indptr, csr.indices, csr.data, cket_col, tmp_col
                    )
                    val_pq = float(np.dot(tmp, cbra))
                    # Reuse the same operator for the transpose bilinear: (E_pq|bra>) dotted with ket.
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        csr.indptr, csr.indices, csr.data, cbra.reshape(cbra.size, 1), tmp_col
                    )
                    val_qp = float(np.dot(tmp, cket))
                else:
                    indptr = csr.indptr
                    indices = csr.indices
                    data = csr.data
                    acc_pq = 0.0
                    acc_qp = 0.0
                    for j in range(int(cket.size)):
                        xj = float(cket[j])
                        yj = float(cbra[j])
                        start = int(indptr[j])
                        end = int(indptr[j + 1])
                        for k in range(start, end):
                            i = int(indices[k])
                            aik = float(data[k])
                            acc_pq += aik * float(cbra[i]) * xj
                            acc_qp += aik * float(cket[i]) * yj
                    val_pq = float(acc_pq)
                    val_qp = float(acc_qp)

                dm1[q, p] = val_pq
                dm1[p, q] = val_qp

        return np.ascontiguousarray(dm1)

    def trans_rdm12(
        self,
        ci_bra,
        ci_ket,
        norb: int,
        nelec: int | tuple[int, int],
        *,
        return_cupy: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transition (dm1, dm2) between CI vectors in the active MO basis.

        Convention matches `make_rdm12`:
          dm1[p,q] = <bra| E_{q p} |ket>

        The returned `dm2` follows the same spin-free δ-term correction as `make_rdm12`.
        """

        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        ne_constraints = kwargs.get("ne_constraints", getattr(self, "ne_constraints", None))
        drt_key, drt = self._drt_for_ci(
            ci_bra,
            norb=norb,
            nelec_total=nelec_total,
            twos=twos,
            orbsym=kwargs.get("orbsym", self.orbsym),
            wfnsym=kwargs.get("wfnsym", self.wfnsym),
            ne_constraints=ne_constraints,
        )
        _validate_civec_shape(ci_ket, int(drt.ncsf))

        rdm_backend = str(kwargs.get("rdm_backend", getattr(self, "rdm_backend", "auto"))).strip().lower()
        matvec_backend = str(getattr(self, "matvec_backend", "contract")).strip().lower()
        strict_gpu = bool(kwargs.get("strict_gpu", getattr(self, "strict_gpu", False)))
        rdm_block_nops = int(kwargs.get("rdm_block_nops", getattr(self, "rdm_block_nops", 8)))
        rdm_tmpdir = kwargs.get("rdm_tmpdir", getattr(self, "rdm_tmpdir", None))
        max_memory = float(kwargs.get("max_memory", getattr(self, "max_memory", 4000.0)))

        if strict_gpu and matvec_backend.startswith("cuda"):
            if rdm_backend == "auto":
                rdm_backend = "cuda"
            elif rdm_backend != "cuda":
                raise ValueError("strict_gpu=True requires rdm_backend='cuda' for CUDA matvec workflows")

        if rdm_backend == "cuda":
            from asuka.cuda.rdm_gpu import make_rdm12_cuda_workspace, trans_rdm12_cuda

            build_threads = int(
                kwargs.get("rdm_cuda_build_threads", getattr(self, "rdm_cuda_build_threads", 256))
            )
            use_epq_table = kwargs.get("rdm_cuda_use_epq_table", getattr(self, "rdm_cuda_use_epq_table", None))
            if use_epq_table is None:
                matvec_backend = str(getattr(self, "matvec_backend", "contract")).strip().lower()
                if matvec_backend.startswith("cuda") and bool(getattr(self, "matvec_cuda_use_epq_table", False)):
                    use_epq_table = True
            enable_emulation = bool(
                kwargs.get(
                    "rdm_cuda_enable_fp64_emulation",
                    getattr(self, "rdm_cuda_enable_fp64_emulation", False),
                )
            )
            gemm_backend = str(kwargs.get("rdm_cuda_gemm_backend", getattr(self, "rdm_cuda_gemm_backend", "gemmex_fp64")))
            math_mode = str(kwargs.get("rdm_cuda_math_mode", getattr(self, "rdm_cuda_math_mode", "default")))
            if enable_emulation:
                gemm_backend = "gemmex_emulated_fixedpoint"
                math_mode = "fp64_emulated_fixedpoint"

            cublas_workspace_mb = int(
                kwargs.get("rdm_cuda_cublas_workspace_mb", getattr(self, "rdm_cuda_cublas_workspace_mb", 0))
            )
            emulation_strategy = kwargs.get(
                "rdm_cuda_emulation_strategy",
                getattr(self, "rdm_cuda_emulation_strategy", None),
            )
            mantissa_control = kwargs.get(
                "rdm_cuda_fixed_point_mantissa_control",
                getattr(self, "rdm_cuda_fixed_point_mantissa_control", None),
            )
            max_bits = kwargs.get(
                "rdm_cuda_fixed_point_max_mantissa_bits",
                getattr(self, "rdm_cuda_fixed_point_max_mantissa_bits", None),
            )
            bit_offset = kwargs.get(
                "rdm_cuda_fixed_point_mantissa_bit_offset",
                getattr(self, "rdm_cuda_fixed_point_mantissa_bit_offset", None),
            )
            streaming_ncsf_cutoff = int(
                kwargs.get(
                    "rdm_cuda_streaming_ncsf_cutoff",
                    getattr(self, "rdm_cuda_streaming_ncsf_cutoff", 2_000_000),
                )
            )
            if streaming_ncsf_cutoff < 0:
                streaming_ncsf_cutoff = 0

            get_state_cache(drt)

            ws_key = drt_key
            ws = self._rdm_cuda_ws_cache.get(ws_key)
            if ws is None:
                ws = make_rdm12_cuda_workspace(
                    drt,
                    gemm_backend=gemm_backend,
                    math_mode=math_mode,
                    cublas_workspace_mb=cublas_workspace_mb,
                    emulation_strategy=None if emulation_strategy is None else str(emulation_strategy),
                    fixed_point_mantissa_control=None if mantissa_control is None else str(mantissa_control),
                    fixed_point_max_mantissa_bits=None if max_bits is None else int(max_bits),
                    fixed_point_mantissa_bit_offset=None if bit_offset is None else int(bit_offset),
                )
                self._rdm_cuda_ws_cache[ws_key] = ws

            # Pass ci vectors as-is (numpy or CuPy); trans_rdm12_cuda handles both.
            try:
                import cupy as _cp_rdm  # type: ignore[import-not-found]
            except Exception:
                _cp_rdm = None
            if _cp_rdm is not None and isinstance(ci_bra, _cp_rdm.ndarray):
                _bra = ci_bra.astype(_cp_rdm.float64, copy=False)
            else:
                _bra = np.asarray(ci_bra, dtype=np.float64)
            if _cp_rdm is not None and isinstance(ci_ket, _cp_rdm.ndarray):
                _ket = ci_ket.astype(_cp_rdm.float64, copy=False)
            else:
                _ket = np.asarray(ci_ket, dtype=np.float64)
            return trans_rdm12_cuda(
                drt,
                _bra,
                _ket,
                workspace=ws,
                block_nops=rdm_block_nops,
                build_threads=build_threads,
                use_epq_table=None if use_epq_table is None else bool(use_epq_table),
                gemm_backend=gemm_backend,
                math_mode=math_mode,
                cublas_workspace_mb=cublas_workspace_mb,
                emulation_strategy=None if emulation_strategy is None else str(emulation_strategy),
                fixed_point_mantissa_control=None if mantissa_control is None else str(mantissa_control),
                fixed_point_max_mantissa_bits=None if max_bits is None else int(max_bits),
                fixed_point_mantissa_bit_offset=None if bit_offset is None else int(bit_offset),
                streaming_ncsf_cutoff=int(streaming_ncsf_cutoff),
                return_cupy=return_cupy,
            )

        # Keep behavior consistent with `make_rdm12`: ensure the E_pq cache exists.
        if strict_gpu and matvec_backend.startswith("cuda"):
            raise RuntimeError("strict_gpu=True forbids non-CUDA trans_rdm12 backends in CUDA matvec workflows")
        rdm_nthreads = int(kwargs.get("rdm_nthreads", getattr(self, "rdm_nthreads", 0)))
        if rdm_nthreads <= 0:
            rdm_nthreads = _auto_num_threads()
        precompute_epq_actions(drt, nthreads=rdm_nthreads)

        nops = norb * norb
        rdm_blas_nthreads = kwargs.get("rdm_blas_nthreads", getattr(self, "rdm_blas_nthreads", None))
        if rdm_blas_nthreads is not None:
            rdm_blas_nthreads = int(rdm_blas_nthreads)
            if rdm_blas_nthreads <= 0:
                rdm_blas_nthreads = None
        if rdm_blas_nthreads is None:
            global_blas = openblas_get_num_threads()
            if global_blas is not None:
                if nops <= 512:
                    rdm_blas_nthreads = min(int(global_blas), 8)
                else:
                    rdm_blas_nthreads = int(global_blas)
            elif nops <= 512:
                rdm_blas_nthreads = 1

        force_memmap = None
        if rdm_backend == "auto":
            force_memmap = None
        elif rdm_backend == "stream":
            force_memmap = True
        elif rdm_backend == "legacy":
            force_memmap = False
        else:
            raise ValueError(f"unsupported rdm_backend={rdm_backend!r}")

        if rdm_blas_nthreads is None:
            return trans_rdm12_streaming(
                drt,
                np.asarray(ci_bra, dtype=np.float64),
                np.asarray(ci_ket, dtype=np.float64),
                max_memory_mb=max_memory,
                block_nops=rdm_block_nops,
                tmpdir=None if rdm_tmpdir is None else str(rdm_tmpdir),
                force_memmap=force_memmap,
            )
        with blas_thread_limit(int(rdm_blas_nthreads)):
            return trans_rdm12_streaming(
                drt,
                np.asarray(ci_bra, dtype=np.float64),
                np.asarray(ci_ket, dtype=np.float64),
                max_memory_mb=max_memory,
                block_nops=rdm_block_nops,
                tmpdir=None if rdm_tmpdir is None else str(rdm_tmpdir),
                force_memmap=force_memmap,
            )


def _get_drt_epq_cache(drt: DRT):
    cache = _get_epq_action_cache(drt)

    def get(p: int, q: int):
        return _csr_for_epq(cache, drt, int(p), int(q))

    return get


def _normalize_ci0(ci0, *, nroots: int, ncsf: int) -> list[np.ndarray]:
    if isinstance(ci0, np.ndarray):
        arr = np.asarray(ci0, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != ncsf:
                raise ValueError("ci0 has wrong length")
            return [np.ascontiguousarray(arr)]
        if arr.ndim == 2:
            if arr.shape != (nroots, ncsf):
                raise ValueError("ci0 has wrong shape")
            return [np.ascontiguousarray(arr[i]) for i in range(nroots)]
        raise ValueError("ci0 must be 1D or 2D ndarray")

    if isinstance(ci0, (list, tuple)):
        if len(ci0) != nroots:
            raise ValueError("ci0 list length must equal nroots")
        out: list[np.ndarray] = []
        for v in ci0:
            vv = np.asarray(v, dtype=np.float64).ravel()
            if vv.size != ncsf:
                raise ValueError("ci0 vector has wrong length")
            out.append(np.ascontiguousarray(vv))
        return out

    raise ValueError("unsupported ci0 type")


def _validate_civec_shape(civec, ncsf: int) -> None:
    # Support both NumPy and CuPy arrays — use .ndim/.size/.shape directly
    # rather than np.asarray() which fails for CuPy.
    if hasattr(civec, 'ndim') and hasattr(civec, 'size'):
        if civec.ndim == 1 and int(civec.size) == ncsf:
            return
        if civec.ndim == 2 and civec.shape == (1, ncsf):
            return
        raise ValueError(f"expected civec shape ({ncsf},) or (1,{ncsf}), got {civec.shape}")
    arr = np.asarray(civec)
    if arr.ndim == 1 and arr.size == ncsf:
        return
    if arr.ndim == 2 and arr.shape == (1, ncsf):
        return
    raise ValueError(f"expected civec shape ({ncsf},) or (1,{ncsf}), got {arr.shape}")
