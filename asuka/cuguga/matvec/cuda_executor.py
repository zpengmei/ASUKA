"""CUDA executor for the GUGA matvec — wraps CuPy + CUDA extension calls.

This is the *minimal* executor that implements the blocked-EPQ path.
All optional optimizations (CSR cache, CUDA graph, sym-pair, fused hop, etc.)
are NOT part of this module — they will be layered on top later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache, get_state_cache
from asuka.kernels import guga as guga_kernels

_ext = guga_kernels.load_ext()


class CudaGugaExecutor:
    """Minimal CUDA executor for the blocked-EPQ matvec path.

    Owns:
      - Device buffers (W, g, overflow)
      - GEMM workspace (Kernel3BuildGWorkspace / Kernel3BuildGDFWorkspace)
      - EPQ table (pre-built or streamed)
      - DRT device state

    Does NOT own:
      - Tiling loop (controller's job)
      - Path selection logic (controller's job)
      - Optional cache / pipeline / graph layers (future mixins)
    """

    def __init__(
        self,
        drt: DRT,
        *,
        drt_dev: Any | None = None,
        state_dev: Any | None = None,
        state_cache: DRTStateCache | None = None,
        eri_mat: Any | None = None,
        l_full: Any | None = None,
        h_eff: Any | None = None,
        dtype: Any | None = None,
        max_g_bytes: int = 256 * 1024 * 1024,
        threads_w: int = 256,
        threads_apply: int = 32,
        epq_build_device: bool = False,
        epq_indptr_dtype: str | None = "auto",
        gemm_backend: str = "gemmex_fp64",
        kahan_compensation: bool = False,
    ) -> None:
        if _ext is None:
            raise RuntimeError(
                "CUDA extension not available; build with python -m asuka.build.guga_cuda_ext"
            )

        import cupy as cp

        self._cp = cp
        self.drt = drt
        self._norb = int(drt.norb)
        self._ncsf = int(drt.ncsf)
        self._nops = self._norb * self._norb

        self._dtype = cp.dtype(cp.float64 if dtype is None else dtype)
        if self._dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
            raise ValueError("dtype must be float32 or float64")

        self.threads_w = int(threads_w)
        self.threads_apply = int(threads_apply)
        self.kahan_compensation = bool(kahan_compensation)

        # --- DRT device state ---
        from asuka.cuda.cuda_backend import make_device_drt, make_device_state_cache

        if state_cache is None:
            state_cache = get_state_cache(drt)
        if drt_dev is None:
            drt_dev = make_device_drt(drt)
        if state_dev is None:
            state_dev = make_device_state_cache(drt, drt_dev, cache=state_cache)
        self.state_cache = state_cache
        self.state_dev = state_dev
        self.drt_dev = drt_dev

        # --- Integrals ---
        self._eri_mat = cp.asarray(eri_mat, dtype=self._dtype) if eri_mat is not None else None
        self._l_full = cp.asarray(l_full, dtype=self._dtype) if l_full is not None else None
        self._use_df = self._l_full is not None
        if self._eri_mat is None and self._l_full is None:
            raise ValueError("Either eri_mat or l_full must be provided")

        # --- h_eff ---
        self._h_eff_flat: Any | None = None
        if h_eff is not None:
            h = cp.asarray(h_eff, dtype=self._dtype)
            self._h_eff_flat = h.ravel() if h.ndim == 2 else h

        # --- Cached index arrays ---
        self._task_csf_all = cp.arange(self._ncsf, dtype=cp.int32)

        # --- EPQ table ---
        from asuka.cuda._backend_epq_config import normalize_epq_indptr_mode
        self.epq_indptr_dtype = normalize_epq_indptr_mode(epq_indptr_dtype)
        self._epq_table = self._build_epq_table(epq_build_device)
        self._epq_table_t: Any | None = None  # lazy

        # --- Block size and buffers ---
        self._nrows_block = self._compute_block_size(max_g_bytes)
        self._w_buf = cp.empty((self._nrows_block, self._nops), dtype=self._dtype)
        self._g_buf = cp.empty((self._nrows_block, self._nops), dtype=self._dtype)
        self._overflow_w = cp.zeros((1,), dtype=cp.int32)
        self._overflow_apply = cp.zeros((1,), dtype=cp.int32)

        # --- GEMM workspace ---
        self._gemm_backend = str(gemm_backend)
        self._gemm_ws = self._init_gemm_workspace()

        # --- Cached transposes for DF ---
        self._l_full_t: Any | None = None
        if self._use_df and self._l_full is not None:
            self._l_full_t = self._l_full.T.copy()

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def norb(self) -> int:
        return self._norb

    @property
    def ncsf(self) -> int:
        return self._ncsf

    @property
    def nops(self) -> int:
        return self._nops

    # ------------------------------------------------------------------
    # Buffer lifecycle
    # ------------------------------------------------------------------

    def alloc_y(self, *, like: Any) -> Any:
        cp = self._cp
        return cp.zeros_like(like)

    def memzero(self, buf: Any, *, stream: Any) -> None:
        cp = self._cp
        cp.cuda.runtime.memsetAsync(
            int(buf.data.ptr), 0, int(buf.size) * int(buf.itemsize), int(stream.ptr),
        )

    # ------------------------------------------------------------------
    # One-body
    # ------------------------------------------------------------------

    def one_body_scatter(
        self,
        x: Any,
        y: Any,
        *,
        stream: Any,
    ) -> None:
        """y = one-body H*x via single EPQ-table scatter.

        Passes h_eff_flat as a 1D (nops,) g-vector with task_scale=x,
        so the kernel broadcasts internally — zero extra allocation.
        """
        from asuka.cuda.cuda_backend import apply_g_flat_scatter_atomic_inplace_device

        h = self._h_eff_flat
        if h is None:
            self.memzero(y, stream=stream)
            return

        apply_g_flat_scatter_atomic_inplace_device(
            self.drt,
            self.drt_dev,
            self.state_dev,
            self._task_csf_all,
            h,
            task_scale=x,
            epq_table=self._epq_table,
            apply_mode="auto",
            y=y,
            overflow=self._overflow_apply,
            threads=self.threads_apply,
            zero_y=True,
            stream=stream,
            sync=False,
            check_overflow=False,
            dtype=self._dtype,
            use_kahan=self.kahan_compensation,
        )

    # ------------------------------------------------------------------
    # EPQ table
    # ------------------------------------------------------------------

    def has_epq_table(self) -> bool:
        return self._epq_table is not None

    def build_epq_transpose(self) -> Any:
        if self._epq_table_t is not None:
            return self._epq_table_t
        if self._epq_table is None:
            raise RuntimeError("No EPQ table available")

        from asuka.cuda.cuda_backend import build_epq_action_table_transpose_device

        self._epq_table_t = build_epq_action_table_transpose_device(
            self.drt,
            self._epq_table,
            dtype=self._dtype,
            indptr_dtype=self.epq_indptr_dtype,
            use_cache=True,
        )
        return self._epq_table_t

    # ------------------------------------------------------------------
    # Two-body blocked path
    # ------------------------------------------------------------------

    def build_w_diag(
        self,
        x: Any,
        k_start: int,
        k_count: int,
        w_out: Any,
        *,
        stream: Any,
    ) -> None:
        from asuka.cuda.cuda_backend import build_w_diag_from_steps_inplace_device

        w_slice = w_out[:k_count]
        build_w_diag_from_steps_inplace_device(
            self.state_dev,
            j_start=k_start,
            j_count=k_count,
            x=x,
            w_out=w_slice,
            threads=256,
            stream=stream,
            sync=False,
            relative_w=True,
        )

    def build_w_offdiag(
        self,
        epq_table_t: Any,
        x: Any,
        k_start: int,
        k_count: int,
        w_out: Any,
        *,
        stream: Any,
        check_overflow: bool = True,
    ) -> None:
        from asuka.cuda.cuda_backend import build_w_from_epq_transpose_range_inplace_device

        w_slice = w_out[:k_count]
        build_w_from_epq_transpose_range_inplace_device(
            self.drt,
            self.state_dev,
            epq_table_t,
            x,
            w_out=w_slice,
            overflow=self._overflow_w,
            threads=self.threads_w,
            stream=stream,
            sync=False,
            check_overflow=False,  # defer overflow checking to avoid forced sync
            k_start=k_start,
            k_count=k_count,
            dtype=self._dtype,
        )

    def contract_w_eri(
        self,
        w: Any,
        k_count: int,
        g_out: Any,
        *,
        stream: Any,
    ) -> None:
        cp = self._cp
        w_slice = w[:k_count]
        g_slice = g_out[:k_count]

        if self._use_df:
            self._contract_w_eri_df(w_slice, k_count, g_slice, stream=stream)
        else:
            self._contract_w_eri_dense(w_slice, k_count, g_slice, stream=stream)

    def apply_g(
        self,
        epq_table_t: Any,
        g_block: Any,
        k_start: int,
        k_count: int,
        y: Any,
        *,
        stream: Any,
        check_overflow: bool = True,
    ) -> None:
        from asuka.cuda.cuda_backend import apply_g_flat_gather_epq_transpose_range_inplace_device

        g_slice = g_block[:k_count]
        apply_g_flat_gather_epq_transpose_range_inplace_device(
            self.drt,
            self.drt_dev,
            self.state_dev,
            epq_table_t,
            g_slice,
            k_start=k_start,
            k_count=k_count,
            y=y,
            overflow=self._overflow_apply,
            threads=self.threads_apply,
            add=True,
            stream=stream,
            sync=False,
            check_overflow=False,
            dtype=self._dtype,
            use_kahan=self.kahan_compensation,
        )

    # ------------------------------------------------------------------
    # Block-size query
    # ------------------------------------------------------------------

    def block_size(self) -> int:
        return self._nrows_block

    def alloc_w_block(self) -> Any:
        return self._w_buf

    def alloc_g_block(self) -> Any:
        return self._g_buf

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------

    def sync_stream(self, stream: Any) -> None:
        stream.synchronize()

    def get_current_stream(self) -> Any:
        cp = self._cp
        return cp.cuda.get_current_stream()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> int:
        freed = 0
        for attr in ("_w_buf", "_g_buf", "_overflow_w", "_overflow_apply",
                      "_eri_mat", "_l_full", "_l_full_t"):
            buf = getattr(self, attr, None)
            if buf is not None:
                freed += int(buf.nbytes) if hasattr(buf, "nbytes") else 0
                setattr(self, attr, None)
        self._epq_table = None
        self._epq_table_t = None
        self._gemm_ws = None
        return freed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_epq_table(self, device_build: bool) -> Any:
        """Build the EPQ action table (source-major)."""
        from asuka.cuda._backend_epq_table import init_workspace_epq_table
        from asuka.cuda._backend_epq_config import (
            epq_indptr_cp_dtype_for_total_nnz,
            EPQ_I32_MAX_NNZ,
        )
        from asuka.cuda.cuda_backend import (
            build_epq_action_table_combined_device,
            build_epq_action_table_combined_device_tiled,
            _get_epq_action_table_combined_host,
            _as_epq_indptr_array,
            _as_epq_pq_array,
        )

        cp = self._cp
        result = init_workspace_epq_table(
            cp=cp,
            drt=self.drt,
            drt_dev=self.drt_dev,
            state_dev=self.state_dev,
            use_epq_table=True,
            epq_streaming=False,
            epq_build_device=device_build,
            epq_build_j_tile=0,
            j_tile=min(1024, self._ncsf),
            norb=self._norb,
            ncsf=self._ncsf,
            threads_enum=128,
            epq_recompute_warp_coop=False,
            dtype=self._dtype,
            epq_indptr_dtype=self.epq_indptr_dtype,
            epq_build_nthreads=0,
            ext=_ext,
            build_device_tiled_fn=build_epq_action_table_combined_device_tiled,
            build_device_fn=build_epq_action_table_combined_device,
            build_host_fn=_get_epq_action_table_combined_host,
            indptr_dtype_resolver_fn=epq_indptr_cp_dtype_for_total_nnz,
            as_indptr_array_fn=_as_epq_indptr_array,
            as_pq_array_fn=_as_epq_pq_array,
            epq_i32_max_nnz=int(EPQ_I32_MAX_NNZ),
        )
        return result["epq_table"]

    def _compute_block_size(self, max_g_bytes: int) -> int:
        """Compute nrows_block from memory budget."""
        row_bytes = int(self._nops) * int(self._dtype.itemsize)
        # Two buffers (W + g) per block
        nrows = max(1, max_g_bytes // (2 * row_bytes))
        return min(nrows, self._ncsf)

    def _init_gemm_workspace(self) -> Any:
        """Initialize GEMM workspace for dense ERI or DF contraction."""
        if self._use_df:
            return self._init_gemm_workspace_df()
        else:
            return self._init_gemm_workspace_dense()

    def _init_gemm_workspace_dense(self) -> Any:
        from asuka.cuda.cuda_backend import Kernel3BuildGWorkspace

        try:
            ws = Kernel3BuildGWorkspace(
                self._nops,
                max_nrows=self._nrows_block,
            )
            ws.set_gemm_backend(self._gemm_backend)
            return ws
        except Exception:
            return None

    def _init_gemm_workspace_df(self) -> Any:
        from asuka.cuda.cuda_backend import Kernel3BuildGDFWorkspace

        if self._l_full is None:
            return None
        naux = int(self._l_full.shape[1])
        try:
            ws = Kernel3BuildGDFWorkspace(
                self._nops,
                naux,
                max_nrows=self._nrows_block,
            )
            ws.set_gemm_backend(self._gemm_backend)
            ws.set_cublas_math_mode("default")
            return ws
        except Exception:
            return None

    def _contract_w_eri_dense(
        self, w_slice: Any, k_count: int, g_slice: Any, *, stream: Any,
    ) -> None:
        if self._gemm_ws is not None and self._eri_mat is not None:
            self._gemm_ws.gemm_w_eri_mat_inplace_device(
                w_slice,
                self._eri_mat,
                g_out=g_slice,
                dtype=self._dtype,
                half=0.5,
                stream=stream,
                sync=False,
            )
        elif self._eri_mat is not None:
            cp = self._cp
            cp.matmul(w_slice, self._eri_mat, out=g_slice)
            g_slice *= 0.5
        else:
            raise RuntimeError("No ERI matrix available for dense contraction")

    def _contract_w_eri_df(
        self, w_slice: Any, k_count: int, g_slice: Any, *, stream: Any,
    ) -> None:
        if self._gemm_ws is not None:
            self._gemm_ws.gemm_w_l_full_inplace_device(
                w_slice,
                self._l_full,
                g_out=g_slice,
                half=0.5,
                stream=stream,
                sync=False,
            )
        elif self._l_full is not None and self._l_full_t is not None:
            cp = self._cp
            naux = int(self._l_full.shape[1])
            t_buf = cp.empty((k_count, naux), dtype=self._dtype)
            cp.matmul(w_slice, self._l_full, out=t_buf)
            t_buf *= 0.5
            cp.matmul(t_buf, self._l_full_t, out=g_slice)
        else:
            raise RuntimeError("No DF factors available for contraction")
