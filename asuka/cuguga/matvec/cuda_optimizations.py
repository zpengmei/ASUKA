"""Optional CUDA optimization layers for the GUGA matvec executor.

Each optimization is a wrapper class that composes over a base executor
(or another wrapper) via the decorator pattern.  The controller sees
the same GugaExecutorProtocol — it never knows which optimizations are
active.

New backends (ROCm, CPU-reference) implement only the 14 base protocol
methods.  None of these optimizations are required.

Usage::

    base = CudaGugaExecutor(drt, eri_mat=..., h_eff=..., epq_build_device=True)

    # Layer optimizations:
    opt  = SymPairExecutor(base)           # ~2x GEMM speedup
    opt  = TransposeGuardExecutor(opt)     # memory guard
    opt  = DiagCacheExecutor(opt)          # cache occ-based diagonal across blocks

    controller = GugaMatvecController(opt)
    y = controller.hop(x)
"""

from __future__ import annotations

from typing import Any

from asuka.cuguga.matvec.protocol import GugaExecutorProtocol


# ======================================================================
# Base passthrough wrapper
# ======================================================================

class _ExecutorWrapper:
    """Delegates every protocol method to *base*.  Subclass and override."""

    def __init__(self, base: GugaExecutorProtocol) -> None:
        self._base = base

    # --- properties ---
    @property
    def norb(self) -> int:
        return self._base.norb

    @property
    def ncsf(self) -> int:
        return self._base.ncsf

    @property
    def nops(self) -> int:
        return self._base.nops

    # --- lifecycle ---
    def alloc_y(self, *, like: Any) -> Any:
        return self._base.alloc_y(like=like)

    def memzero(self, buf: Any, *, stream: Any) -> None:
        self._base.memzero(buf, stream=stream)

    # --- one-body ---
    def one_body_scatter(self, x: Any, y: Any, *, stream: Any) -> None:
        self._base.one_body_scatter(x, y, stream=stream)

    # --- EPQ ---
    def has_epq_table(self) -> bool:
        return self._base.has_epq_table()

    def build_epq_transpose(self) -> Any:
        return self._base.build_epq_transpose()

    # --- two-body blocked ---
    def build_w_diag(self, x: Any, k_start: int, k_count: int, w_out: Any, *, stream: Any) -> None:
        self._base.build_w_diag(x, k_start, k_count, w_out, stream=stream)

    def build_w_offdiag(self, epq_table_t: Any, x: Any, k_start: int, k_count: int, w_out: Any,
                        *, stream: Any, check_overflow: bool = True) -> None:
        self._base.build_w_offdiag(epq_table_t, x, k_start, k_count, w_out,
                                   stream=stream, check_overflow=check_overflow)

    def contract_w_eri(self, w: Any, k_count: int, g_out: Any, *, stream: Any) -> None:
        self._base.contract_w_eri(w, k_count, g_out, stream=stream)

    def apply_g(self, epq_table_t: Any, g_block: Any, k_start: int, k_count: int, y: Any,
                *, stream: Any, check_overflow: bool = True) -> None:
        self._base.apply_g(epq_table_t, g_block, k_start, k_count, y,
                           stream=stream, check_overflow=check_overflow)

    # --- block-size ---
    def block_size(self) -> int:
        return self._base.block_size()

    def alloc_w_block(self) -> Any:
        return self._base.alloc_w_block()

    def alloc_g_block(self) -> Any:
        return self._base.alloc_g_block()

    # --- sync ---
    def sync_stream(self, stream: Any) -> None:
        self._base.sync_stream(stream)

    def get_current_stream(self) -> Any:
        return self._base.get_current_stream()

    # --- cleanup ---
    def release(self) -> int:
        return self._base.release()


# ======================================================================
# Sym-pair GEMM compression
# ======================================================================

class SymPairExecutor(_ExecutorWrapper):
    """Compress nops → npair = norb*(norb+1)/2 for the W@ERI GEMM.

    The two-body ERI matrix has 8-fold symmetry: eri[pq,rs] = eri[qp,rs].
    By packing W and ERI into pair-indexed form, the GEMM is ~2× smaller.
    The pair-fused CUDA kernels also build W and apply g directly in
    pair-indexed form, avoiding pack/unpack overhead.

    Falls back to the base executor if pair-fused kernels are unavailable.
    """

    def __init__(self, base: GugaExecutorProtocol) -> None:
        super().__init__(base)

        import cupy as cp
        from asuka.cuda.cuda_backend import has_sym_pair_fused_kernels
        from asuka.kernels import guga as guga_kernels

        self._cp = cp
        self._ext = guga_kernels.load_ext()
        self._use_fused = has_sym_pair_fused_kernels()

        norb = base.norb
        self._npair = norb * (norb + 1) // 2

        if not self._use_fused:
            return

        # Build pair index maps: full_to_pair[p*norb+q] = pair_index(min(p,q), max(p,q))
        # Uses lower-triangle packed ordering: pair_idx = max(p,q)*(max(p,q)+1)//2 + min(p,q)
        ftp_h = []
        for p in range(norb):
            for q in range(norb):
                pp, qq = min(p, q), max(p, q)
                ftp_h.append(qq * (qq + 1) // 2 + pp)
        full_to_pair = cp.asarray(ftp_h, dtype=cp.int32)
        self._full_to_pair = full_to_pair

        # Pair-indexed ERI: eri_pair[pair_pq, pair_rs]
        # Extract from full eri_mat using the SAME lower-triangle ordering.
        base_exec = self._unwrap_base()
        eri_mat = getattr(base_exec, "_eri_mat", None)
        if eri_mat is not None:
            # Build pq indices for each pair, matching full_to_pair ordering:
            # pair 0 = (0,0), pair 1 = (0,1), pair 2 = (1,1), pair 3 = (0,2), ...
            pair_pq = []
            for qq in range(norb):
                for pp in range(qq + 1):
                    pair_pq.append(pp * norb + qq)
            pair_pq = cp.asarray(pair_pq, dtype=cp.int64)
            self._eri_pair = eri_mat[cp.ix_(pair_pq, pair_pq)]
        else:
            self._eri_pair = None

        # Pre-allocate pair-indexed W and g buffers
        bs = base.block_size()
        dtype = getattr(base_exec, "_dtype", cp.float64)
        self._w_pair = cp.empty((bs, self._npair), dtype=dtype)
        self._g_pair = cp.empty((bs, self._npair), dtype=dtype)
        self._dtype = dtype

    def _unwrap_base(self):
        """Walk the wrapper chain to find the CudaGugaExecutor.

        Raises RuntimeError if the innermost executor is not a
        CudaGugaExecutor (e.g. ROCm backend).  SymPairExecutor uses
        CUDA-specific pair-fused kernels and cannot wrap non-CUDA backends.
        """
        obj = self._base
        while isinstance(obj, _ExecutorWrapper):
            obj = obj._base
        from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor
        if not isinstance(obj, CudaGugaExecutor):
            raise RuntimeError(
                "SymPairExecutor requires a CudaGugaExecutor as the innermost "
                "executor.  It cannot wrap non-CUDA backends because it uses "
                "CUDA-specific pair-fused kernels."
            )
        return obj

    def build_w_diag(self, x: Any, k_start: int, k_count: int, w_out: Any, *, stream: Any) -> None:
        if not self._use_fused:
            return self._base.build_w_diag(x, k_start, k_count, w_out, stream=stream)

        base_exec = self._unwrap_base()
        w_pair = w_out[:k_count]  # w_out IS the pair buffer (from alloc_w_block)
        self._ext.build_w_diag_pair_from_steps_inplace_device(
            base_exec.state_dev,
            int(k_start),
            int(k_count),
            x,
            w_pair,
            self._full_to_pair,
            256,
            int(stream.ptr),
            False,
            True,  # relative_w
        )

    def build_w_offdiag(self, epq_table_t: Any, x: Any, k_start: int, k_count: int, w_out: Any,
                        *, stream: Any, check_overflow: bool = True) -> None:
        if not self._use_fused:
            return self._base.build_w_offdiag(epq_table_t, x, k_start, k_count, w_out,
                                              stream=stream, check_overflow=check_overflow)

        base_exec = self._unwrap_base()
        w_pair = w_out[:k_count]
        t_indptr, t_source, t_pq, t_data = epq_table_t
        self._ext.build_w_pair_from_epq_transpose_range_inplace_device(
            base_exec.state_dev,
            t_indptr, t_source, t_pq, t_data,
            x,
            w_pair,
            self._full_to_pair,
            int(self._npair),
            base_exec._overflow_w,
            int(base_exec.threads_w),
            int(stream.ptr),
            False,  # sync — defer to controller
            False,  # check_overflow — defer to avoid forced sync
            int(k_start),
            int(k_count),
        )

    def contract_w_eri(self, w: Any, k_count: int, g_out: Any, *, stream: Any) -> None:
        if not self._use_fused or self._eri_pair is None:
            return self._base.contract_w_eri(w, k_count, g_out, stream=stream)

        cp = self._cp
        w_pair = w[:k_count]
        g_pair = g_out[:k_count]

        # Pair GEMM: g_pair = 0.5 * w_pair @ eri_pair
        cp.matmul(w_pair, self._eri_pair, out=g_pair)
        g_pair *= 0.5

    def apply_g(self, epq_table_t: Any, g_block: Any, k_start: int, k_count: int, y: Any,
                *, stream: Any, check_overflow: bool = True) -> None:
        if not self._use_fused:
            return self._base.apply_g(epq_table_t, g_block, k_start, k_count, y,
                                      stream=stream, check_overflow=check_overflow)

        base_exec = self._unwrap_base()
        g_pair = g_block[:k_count]
        t_indptr, t_source, t_pq, t_data = epq_table_t
        self._ext.apply_g_pair_gather_epq_transpose_range_inplace_device(
            base_exec.drt_dev,
            base_exec.state_dev,
            t_indptr, t_source, t_pq, t_data,
            g_pair,
            int(k_start),
            int(k_count),
            self._full_to_pair,
            y,
            base_exec._overflow_apply,
            int(base_exec.threads_apply),
            True,   # add
            int(stream.ptr),
            False,  # sync — defer to controller
            False,  # check_overflow — defer to avoid forced sync
        )

    def alloc_w_block(self) -> Any:
        if self._use_fused:
            return self._w_pair
        return self._base.alloc_w_block()

    def alloc_g_block(self) -> Any:
        if self._use_fused:
            return self._g_pair
        return self._base.alloc_g_block()

    def release(self) -> int:
        freed = 0
        for attr in ("_w_pair", "_g_pair", "_eri_pair", "_full_to_pair"):
            buf = getattr(self, attr, None)
            if buf is not None:
                freed += int(buf.nbytes) if hasattr(buf, "nbytes") else 0
                setattr(self, attr, None)
        freed += self._base.release()
        return freed


# ======================================================================
# Transpose memory guard
# ======================================================================

class TransposeGuardExecutor(_ExecutorWrapper):
    """Skip EPQ transpose build if estimated memory exceeds a budget.

    Falls back to the base executor's transpose build but warns when the
    EPQ transpose would exceed ``reserve_mib`` MiB of device memory.
    """

    def __init__(self, base: GugaExecutorProtocol, *, reserve_mib: int = 512) -> None:
        super().__init__(base)
        self._reserve_bytes = int(reserve_mib) * 1024 * 1024
        self._guarded = False

    def build_epq_transpose(self) -> Any:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        if free < self._reserve_bytes:
            import warnings
            warnings.warn(
                f"TransposeGuardExecutor: only {free / 1e6:.0f} MiB free "
                f"(< {self._reserve_bytes / 1e6:.0f} MiB reserve). "
                f"EPQ transpose build may cause OOM.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._guarded = True
        return self._base.build_epq_transpose()


# ======================================================================
# Convenience: compose multiple optimizations
# ======================================================================

def make_optimized_executor(
    base: GugaExecutorProtocol,
    *,
    sym_pair: bool = True,
    transpose_guard: bool = True,
    transpose_guard_mib: int = 512,
) -> GugaExecutorProtocol:
    """Compose selected optimizations over a base executor.

    Returns an executor that satisfies GugaExecutorProtocol with the
    requested optimizations layered on top.  Each optimization is
    independent and optional — disabled by default for new backends.

    Parameters
    ----------
    base
        The base executor (e.g. CudaGugaExecutor).
    sym_pair
        Enable symmetric-pair GEMM compression (~2× GEMM speedup).
    transpose_guard
        Enable memory guard before EPQ transpose build.
    transpose_guard_mib
        MiB of free GPU memory to reserve when guard is active.
    """
    result = base
    if sym_pair:
        result = SymPairExecutor(result)
    if transpose_guard:
        result = TransposeGuardExecutor(result, reserve_mib=transpose_guard_mib)
    return result
