from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # optional
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:  # optional CUDA extension
    from asuka.kernels import guga as _guga_kernels  # noqa: PLC0415

    _guga_cuda_ext = _guga_kernels.load_ext()
except Exception:  # pragma: no cover
    _guga_cuda_ext = None  # type: ignore


def have_cuda_qmc() -> bool:
    return (cp is not None) and (_guga_cuda_ext is not None)


def qmc_spawn_hamiltonian_events_u64_device(
    drt_dev: Any,
    x_key_dev: Any,
    x_val_dev: Any,
    h_base_flat_dev: Any,
    eri_mat_dev: Any,
    pair_alias_prob_dev: Any | None = None,
    pair_alias_idx_dev: Any | None = None,
    pair_norm_dev: Any | None = None,
    pair_norm_sum: float = 0.0,
    pair_sampling_mode: int = 0,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    initiator_t: float = 0.0,
    threads: int = 128,
    stream: int | None = None,
    sync: bool = True,
):
    """Spawn full-H events on GPU into a fixed-size COO buffer (Key64 space).

    Returns
    -------
    out_key_dev, out_val_dev
        CuPy arrays of shape ``(m*(nspawn_one+nspawn_two),)`` with `key==uint64(-1)`
        as sentinel.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")

    x_key_dev = cp.asarray(x_key_dev, dtype=cp.uint64)
    x_val_dev = cp.asarray(x_val_dev, dtype=cp.float64)
    h_base_flat_dev = cp.asarray(h_base_flat_dev, dtype=cp.float64).ravel()
    eri_mat_dev = cp.asarray(eri_mat_dev, dtype=cp.float64)
    pair_sampling_mode = int(pair_sampling_mode)
    pair_norm_sum = float(pair_norm_sum)
    if pair_sampling_mode != 0:
        if pair_sampling_mode != 1:
            raise ValueError("pair_sampling_mode must be 0 (uniform) or 1 (pair_norm alias)")
        if pair_alias_prob_dev is None or pair_alias_idx_dev is None or pair_norm_dev is None:
            raise ValueError("pair_alias_prob_dev/pair_alias_idx_dev/pair_norm_dev must be provided when pair_sampling_mode!=0")
        if not np.isfinite(pair_norm_sum) or pair_norm_sum <= 0.0:
            raise ValueError("pair_norm_sum must be finite and > 0 when pair_sampling_mode!=0")
        pair_alias_prob_dev = cp.asarray(pair_alias_prob_dev, dtype=cp.float32).ravel()
        pair_alias_idx_dev = cp.asarray(pair_alias_idx_dev, dtype=cp.int32).ravel()
        pair_norm_dev = cp.asarray(pair_norm_dev, dtype=cp.float64).ravel()

    m = int(x_key_dev.size)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    out_len = m * (nspawn_one + nspawn_two)
    out_key_dev = cp.empty(out_len, dtype=cp.uint64)
    out_val_dev = cp.empty(out_len, dtype=cp.float64)

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    fn = getattr(_guga_cuda_ext, "qmc_spawn_hamiltonian_u64_inplace_device", None)
    if fn is None:  # pragma: no cover
        raise RuntimeError(
            "Key64 spawn kernel is unavailable in this build of the cuGUGA CUDA extension "
            "(missing qmc_spawn_hamiltonian_u64_inplace_device). "
            "Rebuild the CUDA extension with Key64 spawning enabled, or use the index-based spawn backend."
        )

    fn(
        drt_dev,
        x_key_dev,
        x_val_dev,
        h_base_flat_dev,
        eri_mat_dev,
        out_key_dev,
        out_val_dev,
        float(eps),
        int(nspawn_one),
        int(nspawn_two),
        int(seed),
        float(initiator_t),
        int(threads),
        int(stream),
        bool(sync),
        pair_alias_prob_dev,
        pair_alias_idx_dev,
        pair_norm_dev,
        float(pair_norm_sum),
        int(pair_sampling_mode),
    )
    return out_key_dev, out_val_dev


def pack_steps_to_key64_host(steps: np.ndarray) -> np.ndarray:
    """Pack a step table (0..3 per orbital) into Key64 representation (2 bits per orbital).

    Parameters
    ----------
    steps
        int/uint array of shape ``(n, norb)`` with entries in ``{0,1,2,3}``.

    Returns
    -------
    np.ndarray
        uint64 array of shape ``(n,)``.
    """

    steps_u8 = np.asarray(steps, dtype=np.uint8)
    if steps_u8.ndim != 2:
        raise ValueError("steps must have shape (n, norb)")
    n, norb = int(steps_u8.shape[0]), int(steps_u8.shape[1])
    if norb < 0 or norb > 32:
        raise ValueError("Key64 packing requires 0 <= norb <= 32")
    if n == 0:
        return np.zeros((0,), dtype=np.uint64)
    key = np.zeros((n,), dtype=np.uint64)
    for k in range(norb):
        key |= (steps_u8[:, k].astype(np.uint64) & np.uint64(3)) << np.uint64(2 * k)
    return key


def csf_idx_to_key64_host(drt: Any, idx: np.ndarray, *, state_cache: Any | None = None) -> np.ndarray:
    """Convert CSF indices to Key64 (uint64) using cached step tables if available."""

    idx_i64 = np.asarray(idx, dtype=np.int64).ravel()
    norb = int(drt.norb)
    if norb > 32:
        raise ValueError("Key64 representation requires drt.norb <= 32")
    if idx_i64.size == 0:
        return np.zeros((0,), dtype=np.uint64)

    if state_cache is not None:
        steps_table = np.asarray(state_cache.steps, dtype=np.int8, order="C")
        steps = steps_table[idx_i64]
    else:
        steps = np.stack([np.asarray(drt.index_to_path(int(i)), dtype=np.int8, order="C") for i in idx_i64], axis=0)
    return pack_steps_to_key64_host(steps)


def key64_to_csf_idx64_host(drt: Any, key: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Convert Key64 (uint64) to CSF indices (int64) via DRT traversal.

    This is vectorized across keys and runs in O(len(key) * norb).
    """

    key_u64 = np.asarray(key, dtype=np.uint64).ravel()
    norb = int(drt.norb)
    if norb > 32:
        raise ValueError("Key64 representation requires drt.norb <= 32")
    if key_u64.size == 0:
        return np.zeros((0,), dtype=np.int64)

    from asuka.cuguga.oracle import _child_prefix_walks  # noqa: PLC0415

    child = np.asarray(drt.child, dtype=np.int32, order="C")
    child_prefix = np.asarray(_child_prefix_walks(drt), dtype=np.int64, order="C")  # (nnodes, 5)

    node = np.zeros((int(key_u64.size),), dtype=np.int32)
    idx = np.zeros((int(key_u64.size),), dtype=np.int64)

    for k in range(norb):
        step = ((key_u64 >> np.uint64(2 * k)) & np.uint64(3)).astype(np.intp, copy=False)
        idx += child_prefix[node, step]
        node = child[node, step]

    if bool(strict):
        if np.any(node < 0):
            raise ValueError("invalid Key64 path (child node < 0)")
        leaf = int(getattr(drt, "leaf", -1))
        if leaf >= 0 and np.any(node != leaf):
            raise ValueError("invalid Key64 path (does not terminate at leaf)")

    if np.any(idx < 0):
        raise ValueError("CSF index must be non-negative")
    return np.asarray(idx, dtype=np.int64)


def key64_to_csf_idx_host(drt: Any, key: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Backward-compatible alias for Key64 -> int64 CSF index conversion."""

    return key64_to_csf_idx64_host(drt, key, strict=bool(strict))


def _compact_spawn_events_u64(evt_key: Any, evt_val: Any) -> tuple[Any, Any, int]:
    """Compact valid uint64-COO spawn events on device.

    Valid events satisfy ``key != UINT64_MAX`` and ``val != 0``.
    """

    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is required")
    n_evt = int(evt_key.size)
    if n_evt <= 0:
        return evt_key[:0], evt_val[:0], 0
    invalid = np.uint64(0xFFFFFFFFFFFFFFFF)
    keep = (evt_key != invalid) & (evt_val != 0.0)
    n_keep = int(cp.count_nonzero(keep).get())
    if n_keep <= 0:
        return evt_key[:0], evt_val[:0], 0
    if n_keep == n_evt:
        return evt_key, evt_val, n_keep
    return evt_key[keep], evt_val[keep], n_keep


def filter_event_buffer_to_host(out_idx_dev: Any, out_val_dev: Any) -> tuple[np.ndarray, np.ndarray]:
    """Filter a GPU event buffer (idx==-1 sentinel) and return host COO arrays."""

    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is required")

    idx = cp.asnumpy(out_idx_dev).ravel().astype(np.int32, copy=False)
    val = cp.asnumpy(out_val_dev).ravel().astype(np.float64, copy=False)
    keep = idx >= 0
    return np.asarray(idx[keep], dtype=np.int32, order="C"), np.asarray(val[keep], dtype=np.float64, order="C")


def qmc_coalesce_coo_i32_f64_device(
    idx_in_dev: Any,
    val_in_dev: Any,
    *,
    threads: int = 256,
    stream: int | None = None,
    sync: bool = True,
):
    """Coalesce COO pairs (device): sort by idx and sum duplicates.

    Parameters
    ----------
    idx_in_dev, val_in_dev
        CuPy device arrays with matching shape ``(n,)``. Invalid entries may use
        ``idx < 0`` (those contribute zero).

    Returns
    -------
    idx_out_dev, val_out_dev, nnz_dev
        Output arrays of shape ``(n,)`` and a device scalar ``int32[1]`` holding
        the number of valid coalesced entries written to the prefix of the outputs.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")

    idx_in_dev = cp.asarray(idx_in_dev, dtype=cp.int32).ravel()
    val_in_dev = cp.asarray(val_in_dev, dtype=cp.float64).ravel()
    if idx_in_dev.size != val_in_dev.size:
        raise ValueError("idx_in_dev and val_in_dev must have the same size")

    n = int(idx_in_dev.size)
    idx_out_dev = cp.empty(n, dtype=cp.int32)
    val_out_dev = cp.empty(n, dtype=cp.float64)
    nnz_dev = cp.empty(1, dtype=cp.int32)

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    _guga_cuda_ext.qmc_coalesce_coo_i32_f64_inplace_device(
        idx_in_dev,
        val_in_dev,
        idx_out_dev,
        val_out_dev,
        nnz_dev,
        int(threads),
        int(stream),
        bool(sync),
    )
    return idx_out_dev, val_out_dev, nnz_dev


def qmc_phi_pivot_resample_i32_f64_device(
    idx_in_dev: Any,
    val_in_dev: Any,
    *,
    m: int,
    pivot: int,
    seed: int,
    threads: int = 256,
    stream: int | None = None,
    sync: bool = True,
):
    """Apply Φ compression on GPU (pivot + systematic resampling).

    Parameters
    ----------
    idx_in_dev, val_in_dev
        Coalesced COO vector on device (sorted unique indices).
    m
        Target budget (output nnz <= m).
    pivot
        Number of pivotal entries to keep deterministically (clamped to <= m).

    Returns
    -------
    idx_out_dev, val_out_dev, nnz_dev
        Output arrays of shape ``(m,)`` and a device scalar ``int32[1]`` holding
        the number of coalesced entries written to the prefix of the outputs.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")

    idx_in_dev = cp.asarray(idx_in_dev, dtype=cp.int32).ravel()
    val_in_dev = cp.asarray(val_in_dev, dtype=cp.float64).ravel()
    if idx_in_dev.size != val_in_dev.size:
        raise ValueError("idx_in_dev and val_in_dev must have the same size")

    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    pivot = int(pivot)
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    idx_out_dev = cp.empty(m, dtype=cp.int32)
    val_out_dev = cp.empty(m, dtype=cp.float64)
    nnz_dev = cp.empty(1, dtype=cp.int32)

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    _guga_cuda_ext.qmc_phi_pivot_resample_i32_f64_inplace_device(
        idx_in_dev,
        val_in_dev,
        idx_out_dev,
        val_out_dev,
        nnz_dev,
        int(m),
        int(pivot),
        int(seed),
        int(threads),
        int(stream),
        bool(sync),
    )
    return idx_out_dev, val_out_dev, nnz_dev


@dataclass
class CudaProjectorContextKey64:
    """Reusable CUDA projector context in Key64 walker space (uint64 keys)."""

    drt_dev: Any
    ws: Any

    h_base_flat_dev: Any
    eri_mat_dev: Any

    m: int
    pivot: int
    nspawn_one: int
    nspawn_two: int
    max_n: int
    max_evt: int

    threads_spawn: int = 128
    threads_qmc: int = 256
    stream: int | None = None

    # Optional alias/pair-norm sampling inputs for the Key64 spawn kernel.
    pair_alias_prob_dev: Any | None = None
    pair_alias_idx_dev: Any | None = None
    pair_norm_dev: Any | None = None
    pair_norm_sum: float = 0.0
    pair_sampling_mode: int = 0
    label_mode: str = "key64"
    ncsf_u64: int = 0

    # Ping-pong sparse-vector buffers (capacity m; only prefix [:nnz] is valid).
    x_key_a: Any | None = None
    x_val_a: Any | None = None
    x_key_b: Any | None = None
    x_val_b: Any | None = None
    nnz: int = 0
    use_a: bool = True

    # Preallocated step buffers.
    key_all: Any | None = None
    val_all: Any | None = None
    key_u: Any | None = None
    val_u: Any | None = None
    nnz_u: Any | None = None
    nnz_out: Any | None = None

    def release(self) -> None:
        if self.ws is not None:
            self.ws.release()
        if self.drt_dev is not None:
            self.drt_dev.release()
        self.ws = None
        self.drt_dev = None

    @property
    def x_key(self):
        return self.x_key_a if self.use_a else self.x_key_b

    @property
    def x_val(self):
        return self.x_val_a if self.use_a else self.x_val_b

    @property
    def x_key_next(self):
        return self.x_key_b if self.use_a else self.x_key_a

    @property
    def x_val_next(self):
        return self.x_val_b if self.use_a else self.x_val_a


@dataclass
class CudaBlockProjectorContext:
    """Reusable multi-root CUDA context for repeated projector steps.

    This is a memory-optimized variant for subspace iteration:
    - immutable device resources (DRT/state/integrals/workspace) are shared once,
    - sparse-vector storage is packed as `(nroots, m)` arrays,
    - scratch buffers are reused while stepping roots sequentially.
    """

    drt_dev: Any
    state_dev: Any
    ws: Any

    h_base_flat_dev: Any
    eri_mat_dev: Any

    nroots: int
    m: int
    pivot: int
    nspawn_one: int
    nspawn_two: int
    max_n: int
    max_evt: int

    threads_spawn: int = 128
    threads_qmc: int = 256
    stream: int | None = None

    # Packed sparse-vector buffers: (nroots, m); only prefix [:nnz[k]] is valid per root.
    x_idx_a: Any | None = None
    x_val_a: Any | None = None
    x_idx_b: Any | None = None
    x_val_b: Any | None = None
    nnz: np.ndarray | None = None
    nnz_next: np.ndarray | None = None
    use_a: bool = True

    # Shared scratch buffers (reused while stepping roots sequentially).
    evt_idx: Any | None = None
    evt_val: Any | None = None
    idx_all: Any | None = None
    val_all: Any | None = None
    idx_u: Any | None = None
    val_u: Any | None = None
    nnz_u: Any | None = None
    nnz_out: Any | None = None

    # Optional semi-stochastic deterministic subspace.
    # Host structures are only construction-time artifacts; runtime correction stays on device.
    det_idx_host: np.ndarray | None = None  # sorted global CSF indices (int32)
    det_idx_dev: Any | None = None  # device mirror of det_idx_host
    det_cols: list[tuple[np.ndarray, np.ndarray]] | None = None  # per det-col: (i_pos:int32[], hij:float64[])
    det_hdd_csr_dev: Any | None = None  # cupyx CSR for H_DD in det-subspace ordering
    det_x_buf: Any | None = None  # device x_D scratch (len |D|)
    det_y_buf: Any | None = None  # device y_D scratch (len |D|)
    det_spawn_slot_offsets: Any | None = None  # device [0..nspawn_total) scratch for slot indexing
    det_idx_buf: Any | None = None  # device scratch for appending det events (global idx)
    det_val_buf: Any | None = None  # device scratch for appending det values

    # Optional larger scratch/workspace for fused multi-root linear-combination ops
    # (e.g., apply_right_matrix / fused MGS updates).
    ws_big: Any | None = None
    idx_all_big: Any | None = None
    val_all_big: Any | None = None
    idx_u_big: Any | None = None
    val_u_big: Any | None = None
    nnz_u_big: Any | None = None
    nnz_out_big: Any | None = None

    # Optional composite-key (u64) workspace/buffers for segmented multi-root operators.
    ws_u64: Any | None = None
    key_all_u64: Any | None = None
    val_all_u64: Any | None = None
    key_u_u64: Any | None = None
    val_u_u64: Any | None = None
    nnz_u_u64: Any | None = None

    def release(self) -> None:
        if self.ws is not None:
            self.ws.release()
        if self.ws_big is not None:
            self.ws_big.release()
        if self.ws_u64 is not None:
            self.ws_u64.release()
        if self.state_dev is not None:
            self.state_dev.release()
        if self.drt_dev is not None:
            self.drt_dev.release()
        self.ws = None
        self.ws_big = None
        self.ws_u64 = None
        self.state_dev = None
        self.drt_dev = None

    @property
    def x_idx(self):
        return self.x_idx_a if self.use_a else self.x_idx_b

    @property
    def x_val(self):
        return self.x_val_a if self.use_a else self.x_val_b

    @property
    def x_idx_next(self):
        return self.x_idx_b if self.use_a else self.x_idx_a

    @property
    def x_val_next(self):
        return self.x_val_b if self.use_a else self.x_val_a

    def set_cols(self, cols: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Upload host sparse columns into the current device buffers."""

        if cp is None:  # pragma: no cover
            raise RuntimeError("CuPy is required")
        if self.x_idx is None or self.x_val is None or self.nnz is None:
            raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
        if len(cols) != int(self.nroots):
            raise ValueError(f"cols must have length nroots={int(self.nroots)}")

        for k in range(int(self.nroots)):
            idx_k, val_k = cols[k]
            idx_k = np.asarray(idx_k, dtype=np.int32).ravel()
            val_k = np.asarray(val_k, dtype=np.float64).ravel()
            if idx_k.size != val_k.size:
                raise ValueError(f"column {k} idx/val size mismatch")
            nnz_k = int(idx_k.size)
            if nnz_k <= 0:
                raise ValueError(f"column {k} is empty")
            if nnz_k > int(self.m):
                raise ValueError(f"column {k} nnz exceeds m")
            self.x_idx[k, :nnz_k] = cp.asarray(idx_k, dtype=cp.int32)
            self.x_val[k, :nnz_k] = cp.asarray(val_k, dtype=cp.float64)
            self.nnz[k] = np.int32(nnz_k)

    def get_cols(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Download current columns to host."""

        if cp is None:  # pragma: no cover
            raise RuntimeError("CuPy is required")
        if self.x_idx is None or self.x_val is None or self.nnz is None:
            raise RuntimeError("CudaBlockProjectorContext buffers not initialized")

        cols: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(int(self.nroots)):
            nnz_k = int(self.nnz[k])
            idx_k = cp.asnumpy(self.x_idx[k, :nnz_k]).astype(np.int32, copy=False)
            val_k = cp.asnumpy(self.x_val[k, :nnz_k]).astype(np.float64, copy=False)
            cols.append((idx_k, val_k))
        return cols

    def get_cols_packed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Download packed (nroots, m) arrays to host in one transfer per array."""

        if cp is None:  # pragma: no cover
            raise RuntimeError("CuPy is required")
        if self.x_idx is None or self.x_val is None or self.nnz is None:
            raise RuntimeError("CudaBlockProjectorContext buffers not initialized")

        idx = cp.asnumpy(self.x_idx).astype(np.int32, copy=False)
        val = cp.asnumpy(self.x_val).astype(np.float64, copy=False)
        nnz = np.asarray(self.nnz, dtype=np.int32).copy()
        return idx, val, nnz

    def set_cols_packed(self, idx: np.ndarray, val: np.ndarray, nnz: np.ndarray) -> None:
        """Upload packed (nroots, m) arrays to the current device buffers."""

        if cp is None:  # pragma: no cover
            raise RuntimeError("CuPy is required")
        if self.x_idx is None or self.x_val is None or self.nnz is None:
            raise RuntimeError("CudaBlockProjectorContext buffers not initialized")

        idx = np.asarray(idx, dtype=np.int32)
        val = np.asarray(val, dtype=np.float64)
        nnz = np.asarray(nnz, dtype=np.int32).ravel()
        if idx.shape != (int(self.nroots), int(self.m)):
            raise ValueError(f"idx has wrong shape: {idx.shape} (expected {(int(self.nroots), int(self.m))})")
        if val.shape != (int(self.nroots), int(self.m)):
            raise ValueError(f"val has wrong shape: {val.shape} (expected {(int(self.nroots), int(self.m))})")
        if nnz.size != int(self.nroots):
            raise ValueError(f"nnz has wrong size: {nnz.size} (expected {int(self.nroots)})")
        if int(nnz.min(initial=0)) < 0 or int(nnz.max(initial=0)) > int(self.m):
            raise ValueError("nnz entries must satisfy 0 <= nnz[k] <= m")

        self.x_idx[...] = cp.asarray(idx, dtype=cp.int32)
        self.x_val[...] = cp.asarray(val, dtype=cp.float64)
        self.nnz[:] = nnz.astype(np.int32, copy=False)


def _build_det_subspace_cols_dense(
    *,
    drt: Any,
    h1e: np.ndarray,
    eri,
    det_idx: np.ndarray,
    max_out: int,
    state_cache: Any | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build column-wise connectivity for the deterministic subspace D (dense-ERI path).

    Returns list over j in D (same order as det_idx), where each item is:
      (i_pos, hij) with i_pos in [0,|D|) indexing into det_idx.
    """

    from asuka.cuguga.oracle.sparse import connected_row_sparse  # noqa: PLC0415

    det_idx = np.asarray(det_idx, dtype=np.int64).ravel()
    ndet = int(det_idx.size)
    if ndet <= 0:
        return []
    max_out = int(max_out)

    cols: list[tuple[np.ndarray, np.ndarray]] = []
    for j_pos in range(ndet):
        j = int(det_idx[j_pos])
        i_idx, hij = connected_row_sparse(
            drt,
            np.asarray(h1e, dtype=np.float64),
            eri,
            int(j),
            max_out=int(max_out),
            state_cache=state_cache,
        )
        if i_idx.size == 0:
            cols.append((np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)))
            continue
        i_idx = np.asarray(i_idx, dtype=np.int64, order="C").ravel()
        hij = np.asarray(hij, dtype=np.float64, order="C").ravel()

        # Filter to i in D and map to positions.
        pos = np.searchsorted(det_idx, i_idx)
        inr = (pos >= 0) & (pos < ndet)
        keep = np.zeros_like(inr)
        if np.any(inr):
            pos2 = pos[inr]
            keep[inr] = det_idx[pos2] == i_idx[inr]
        if not np.any(keep):
            cols.append((np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)))
            continue
        i_pos = np.asarray(pos[keep], dtype=np.int32, order="C")
        cols.append((i_pos, np.asarray(hij[keep], dtype=np.float64, order="C")))
    return cols


def _det_subspace_matvec_cols(*, det_cols: list[tuple[np.ndarray, np.ndarray]], x_det: np.ndarray) -> np.ndarray:
    """Compute y = H_DD @ x in the deterministic subspace using prebuilt columns."""

    x_det = np.asarray(x_det, dtype=np.float64).ravel()
    ndet = int(x_det.size)
    if ndet == 0:
        return np.zeros(0, dtype=np.float64)
    if int(len(det_cols)) != ndet:
        raise ValueError("det_cols length mismatch")

    y = np.zeros(ndet, dtype=np.float64)
    for j_pos in range(ndet):
        xj = float(x_det[j_pos])
        if xj == 0.0:
            continue
        i_pos, hij = det_cols[j_pos]
        if i_pos.size:
            y[i_pos] += np.asarray(hij, dtype=np.float64) * xj
    return y


def _det_subspace_coo_from_cols(
    det_cols: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert column-wise deterministic-subspace connectivity to COO triplets.

    Returns
    -------
    row, col, data
        Host COO arrays such that ``H_DD[row[t], col[t]] = data[t]``.
    """

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    dat_parts: list[np.ndarray] = []
    for j_pos, (i_pos, hij) in enumerate(det_cols):
        i_pos_i32 = np.asarray(i_pos, dtype=np.int32).ravel()
        hij_f64 = np.asarray(hij, dtype=np.float64).ravel()
        if i_pos_i32.size != hij_f64.size:
            raise ValueError("det_cols entry has idx/val size mismatch")
        if i_pos_i32.size == 0:
            continue
        row_parts.append(i_pos_i32)
        col_parts.append(np.full(i_pos_i32.size, np.int32(j_pos), dtype=np.int32))
        dat_parts.append(hij_f64)
    if not row_parts:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(np.concatenate(row_parts), dtype=np.int32, order="C"),
        np.asarray(np.concatenate(col_parts), dtype=np.int32, order="C"),
        np.asarray(np.concatenate(dat_parts), dtype=np.float64, order="C"),
    )


def _apply_det_subspace_correction_gpu(
    *,
    ctx: CudaBlockProjectorContext,
    x_idx: Any,
    x_val: Any,
    evt_idx: Any,
    evt_val: Any,
    eps: float,
    nspawn_total: int,
) -> int:
    """Apply deterministic-subspace correction fully on GPU.

    This performs the same logic as the legacy host path:
    1) build ``x_D`` from parents in ``D``;
    2) compute ``y_D = -eps * H_DD x_D`` on device;
    3) append nonzero ``y_D`` events;
    4) remove spawned ``D->D`` events from parents in ``D`` to avoid double counting.

    Returns
    -------
    int
        Number of deterministic correction events to append.
    """

    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is required")
    if (
        ctx.det_idx_dev is None
        or ctx.det_hdd_csr_dev is None
        or ctx.det_x_buf is None
        or ctx.det_y_buf is None
        or ctx.det_idx_buf is None
        or ctx.det_val_buf is None
    ):
        return 0

    ndet = int(ctx.det_idx_dev.size)
    if ndet <= 0:
        return 0

    # Identify parents that are in D and map them to deterministic-subspace positions.
    pos = cp.searchsorted(ctx.det_idx_dev, x_idx)
    inr = pos < ndet
    pos_clip = cp.clip(pos, 0, ndet - 1)
    parent_is_det = inr & (ctx.det_idx_dev[pos_clip] == x_idx)
    parent_pos_det = cp.nonzero(parent_is_det)[0].astype(cp.int64, copy=False)

    # Build x_D on device.
    x_det = ctx.det_x_buf
    x_det[:] = 0.0
    if int(parent_pos_det.size) > 0:
        det_pos = pos[parent_pos_det].astype(cp.int64, copy=False)
        x_det[det_pos] = x_val[parent_pos_det]

    # Exact deterministic-subspace matvec on device.
    y_det = ctx.det_y_buf
    y_det[:] = ctx.det_hdd_csr_dev.dot(x_det)
    y_det *= -float(eps)

    nz = cp.nonzero(y_det != 0.0)[0].astype(cp.int64, copy=False)
    nnz_det = int(nz.size)
    if nnz_det > 0:
        ctx.det_idx_buf[:nnz_det] = ctx.det_idx_dev[nz]
        ctx.det_val_buf[:nnz_det] = y_det[nz]

    # Remove spawned D->D events for parents in D.
    if int(parent_pos_det.size) > 0:
        if ctx.det_spawn_slot_offsets is None or int(ctx.det_spawn_slot_offsets.size) != int(nspawn_total):
            ctx.det_spawn_slot_offsets = cp.arange(int(nspawn_total), dtype=cp.int64)
        slot = (parent_pos_det[:, None] * int(nspawn_total) + ctx.det_spawn_slot_offsets[None, :]).reshape(-1)
        evt_sel = evt_idx[slot]
        pos2 = cp.searchsorted(ctx.det_idx_dev, evt_sel)
        inr2 = (evt_sel >= 0) & (pos2 < ndet)
        pos2_clip = cp.clip(pos2, 0, ndet - 1)
        match = inr2 & (ctx.det_idx_dev[pos2_clip] == evt_sel)
        evt_val[slot[match]] = 0.0

    return nnz_det


def _apply_det_subspace_correction_gpu_u64(
    *,
    ctx: Any,
    x_key: Any,
    x_val: Any,
    evt_key: Any,
    evt_val: Any,
    eps: float,
    nspawn_total: int,
) -> int:
    """Apply deterministic-subspace correction for Key64 single-root FCIQMC."""

    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is required")
    if (
        ctx.det_key_dev is None
        or ctx.det_hdd_csr_dev is None
        or ctx.det_x_buf is None
        or ctx.det_y_buf is None
        or ctx.det_key_buf is None
        or ctx.det_val_buf is None
    ):
        return 0

    ndet = int(ctx.det_key_dev.size)
    if ndet <= 0:
        return 0

    pos = cp.searchsorted(ctx.det_key_dev, x_key)
    inr = pos < ndet
    pos_clip = cp.clip(pos, 0, ndet - 1)
    parent_is_det = inr & (ctx.det_key_dev[pos_clip] == x_key)
    parent_pos_det = cp.nonzero(parent_is_det)[0].astype(cp.int64, copy=False)

    x_det = ctx.det_x_buf
    x_det[:] = 0.0
    if int(parent_pos_det.size) > 0:
        det_pos = pos[parent_pos_det].astype(cp.int64, copy=False)
        x_det[det_pos] = x_val[parent_pos_det]

    y_det = ctx.det_y_buf
    y_det[:] = ctx.det_hdd_csr_dev.dot(x_det)
    y_det *= -float(eps)

    nz = cp.nonzero(y_det != 0.0)[0].astype(cp.int64, copy=False)
    nnz_det = int(nz.size)
    if nnz_det > 0:
        ctx.det_key_buf[:nnz_det] = ctx.det_key_dev[nz]
        ctx.det_val_buf[:nnz_det] = y_det[nz]

    if int(parent_pos_det.size) > 0:
        if ctx.det_spawn_slot_offsets is None or int(ctx.det_spawn_slot_offsets.size) != int(nspawn_total):
            ctx.det_spawn_slot_offsets = cp.arange(int(nspawn_total), dtype=cp.int64)
        slot = (parent_pos_det[:, None] * int(nspawn_total) + ctx.det_spawn_slot_offsets[None, :]).reshape(-1)
        evt_sel = evt_key[slot]
        pos2 = cp.searchsorted(ctx.det_key_dev, evt_sel)
        inr2 = (evt_sel != np.uint64(np.iinfo(np.uint64).max)) & (pos2 < ndet)
        pos2_clip = cp.clip(pos2, 0, ndet - 1)
        match = inr2 & (ctx.det_key_dev[pos2_clip] == evt_sel)
        evt_val[slot[match]] = 0.0

    return nnz_det


def make_cuda_projector_context_key64(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    m: int,
    pivot: int,
    nspawn_one: int,
    nspawn_two: int,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
    pair_alias_prob: Any | None = None,
    pair_alias_idx: Any | None = None,
    pair_norm: Any | None = None,
    pair_norm_sum: float = 0.0,
    pair_sampling_mode: int = 0,
    label_mode: str = "key64",
    ncsf_u64: int | None = None,
) -> CudaProjectorContextKey64:
    """Build a reusable CUDA projector context in Key64 walker space.

    Notes
    -----
    ``label_mode='key64'`` requires ``drt.norb <= 32`` so each CSF path can be
    packed into a 64-bit key (2 bits per orbital step).
    ``label_mode='idx64'`` accepts ``drt.norb <= 64`` and stores global CSF
    indices in the same uint64 buffers/workspace.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not hasattr(_guga_cuda_ext, "QmcWorkspaceU64"):
        raise RuntimeError("Key64 QMC workspace is unavailable (missing _guga_cuda_ext.QmcWorkspaceU64); rebuild the CUDA extension")

    from asuka.cuguga.oracle import _child_prefix_walks  # noqa: PLC0415

    label_mode_s = str(label_mode).strip().lower()
    if label_mode_s not in ("key64", "idx64"):
        raise ValueError("label_mode must be 'key64' or 'idx64'")

    norb = int(drt.norb)
    if label_mode_s == "key64":
        if norb > 32:
            raise ValueError("Key64 projector context requires drt.norb <= 32")
    else:
        if norb > 64:
            raise ValueError("idx64 projector context requires drt.norb <= 64")
    ncsf_u64_eff = int(int(drt.ncsf) if ncsf_u64 is None else int(ncsf_u64))
    if ncsf_u64_eff <= 0:
        raise ValueError("ncsf_u64 must be > 0")
    nops = norb * norb

    h1e = np.asarray(h1e, dtype=np.float64).reshape(norb, norb)
    eri = np.asarray(eri, dtype=np.float64)
    if eri.ndim == 4:
        if eri.shape != (norb, norb, norb, norb):
            raise ValueError("eri4 has wrong shape")
        eri_mat = eri.reshape(nops, nops)
        eri4 = eri
    elif eri.ndim == 2:
        if eri.shape != (nops, nops):
            raise ValueError("eri_mat has wrong shape")
        eri_mat = eri
        eri4 = eri.reshape(norb, norb, norb, norb)
    else:
        raise ValueError("eri must be eri_mat[pq,rs] (2D) or eri4[p,q,r,s] (4D) for CUDA projector")

    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)
    h_base_flat = h_base.ravel(order="C")

    child_prefix = _child_prefix_walks(drt)
    drt_dev = _guga_cuda_ext.make_device_drt(
        int(drt.norb),
        np.asarray(drt.child),
        np.asarray(drt.node_twos),
        np.asarray(child_prefix),
    )

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    m = int(m)
    pivot = int(pivot)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if m < 1:
        raise ValueError("m must be >= 1")
    if pivot < 0:
        raise ValueError("pivot must be >= 0")
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    pair_sampling_mode = int(pair_sampling_mode)
    pair_norm_sum = float(pair_norm_sum)
    pair_alias_prob_dev = None
    pair_alias_idx_dev = None
    pair_norm_dev = None
    if pair_sampling_mode != 0:
        if pair_sampling_mode != 1:
            raise ValueError("pair_sampling_mode must be 0 (uniform) or 1 (pair_norm alias)")
        if pair_alias_prob is None or pair_alias_idx is None or pair_norm is None:
            raise ValueError("pair_alias_prob/pair_alias_idx/pair_norm must be provided when pair_sampling_mode!=0")
        if not np.isfinite(pair_norm_sum) or pair_norm_sum <= 0.0:
            raise ValueError("pair_norm_sum must be finite and > 0 when pair_sampling_mode!=0")

        pair_alias_prob_dev = cp.asarray(pair_alias_prob, dtype=cp.float32).ravel()
        pair_alias_idx_dev = cp.asarray(pair_alias_idx, dtype=cp.int32).ravel()
        pair_norm_dev = cp.asarray(pair_norm, dtype=cp.float64).ravel()
        if int(pair_alias_prob_dev.size) != nops or int(pair_alias_idx_dev.size) != nops or int(pair_norm_dev.size) != nops:
            raise ValueError(f"pair alias/norm arrays must have length nops={nops}")

    nspawn_total = nspawn_one + nspawn_two
    max_evt = m * nspawn_total
    max_n = m * (1 + nspawn_total)

    ws = _guga_cuda_ext.QmcWorkspaceU64(int(max_n), int(m))

    ctx = CudaProjectorContextKey64(
        drt_dev=drt_dev,
        ws=ws,
        h_base_flat_dev=cp.asarray(h_base_flat, dtype=cp.float64),
        eri_mat_dev=cp.asarray(eri_mat, dtype=cp.float64),
        m=int(m),
        pivot=int(pivot),
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        max_n=int(max_n),
        max_evt=int(max_evt),
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=int(stream),
        pair_alias_prob_dev=pair_alias_prob_dev,
        pair_alias_idx_dev=pair_alias_idx_dev,
        pair_norm_dev=pair_norm_dev,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
        label_mode=str(label_mode_s),
        ncsf_u64=int(ncsf_u64_eff),
    )

    # Preallocate buffers.
    ctx.x_key_a = cp.empty(m, dtype=cp.uint64)
    ctx.x_val_a = cp.empty(m, dtype=cp.float64)
    ctx.x_key_b = cp.empty(m, dtype=cp.uint64)
    ctx.x_val_b = cp.empty(m, dtype=cp.float64)
    ctx.key_all = cp.empty(max_n, dtype=cp.uint64)
    ctx.val_all = cp.empty(max_n, dtype=cp.float64)
    ctx.key_u = cp.empty(max_n, dtype=cp.uint64)
    ctx.val_u = cp.empty(max_n, dtype=cp.float64)
    ctx.nnz_u = cp.empty(1, dtype=cp.int32)
    ctx.nnz_out = cp.empty(1, dtype=cp.int32)
    return ctx


def make_cuda_projector_context_idx64(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    m: int,
    pivot: int,
    nspawn_one: int,
    nspawn_two: int,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
    pair_alias_prob: Any | None = None,
    pair_alias_idx: Any | None = None,
    pair_norm: Any | None = None,
    pair_norm_sum: float = 0.0,
    pair_sampling_mode: int = 0,
    ncsf_u64: int | None = None,
) -> CudaProjectorContextKey64:
    """Build a reusable CUDA projector context in idx64 walker space (uint64 CSF indices)."""

    return make_cuda_projector_context_key64(
        drt,
        h1e,
        eri,
        m=int(m),
        pivot=int(pivot),
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=stream,
        pair_alias_prob=pair_alias_prob,
        pair_alias_idx=pair_alias_idx,
        pair_norm=pair_norm,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
        label_mode="idx64",
        ncsf_u64=int(int(drt.ncsf) if ncsf_u64 is None else int(ncsf_u64)),
    )


def make_cuda_block_projector_context(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    nroots: int,
    m: int,
    pivot: int,
    nspawn_one: int,
    nspawn_two: int,
    det_idx: np.ndarray | None = None,
    det_max_out: int = 200_000,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
) -> CudaBlockProjectorContext:
    """Build a reusable CUDA block-projector context for dense-ERI Hamiltonians."""

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")

    from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415
    from asuka.cuguga.oracle import _child_prefix_walks  # noqa: PLC0415

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    norb = int(drt.norb)
    nops = norb * norb

    h1e = np.asarray(h1e, dtype=np.float64).reshape(norb, norb)
    eri = np.asarray(eri, dtype=np.float64)
    if eri.ndim == 4:
        if eri.shape != (norb, norb, norb, norb):
            raise ValueError("eri4 has wrong shape")
        eri_mat = eri.reshape(nops, nops)
        eri4 = eri
    elif eri.ndim == 2:
        if eri.shape != (nops, nops):
            raise ValueError("eri_mat has wrong shape")
        eri_mat = eri
        eri4 = eri.reshape(norb, norb, norb, norb)
    else:
        raise ValueError("eri must be eri_mat[pq,rs] (2D) or eri4[p,q,r,s] (4D) for CUDA projector")

    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)
    h_base_flat = h_base.ravel(order="C")

    child_prefix = _child_prefix_walks(drt)
    drt_dev = _guga_cuda_ext.make_device_drt(int(drt.norb), np.asarray(drt.child), np.asarray(drt.node_twos), np.asarray(child_prefix))
    cache = get_state_cache(drt)
    state_dev = _guga_cuda_ext.make_device_state_cache(drt_dev, np.asarray(cache.steps, dtype=np.int8, order="C"), np.asarray(cache.nodes, dtype=np.int32, order="C"))

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    m = int(m)
    pivot = int(pivot)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if m < 1:
        raise ValueError("m must be >= 1")
    if pivot < 0:
        raise ValueError("pivot must be >= 0")
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    nspawn_total = nspawn_one + nspawn_two
    max_evt = m * nspawn_total
    det_n = 0
    if det_idx is not None:
        det_n = int(np.asarray(det_idx).size)
    max_n = m * (1 + nspawn_total) + int(det_n)

    ws = _guga_cuda_ext.QmcWorkspace(int(max_n), int(m))

    ctx = CudaBlockProjectorContext(
        drt_dev=drt_dev,
        state_dev=state_dev,
        ws=ws,
        h_base_flat_dev=cp.asarray(h_base_flat, dtype=cp.float64),
        eri_mat_dev=cp.asarray(eri_mat, dtype=cp.float64),
        nroots=nroots,
        m=m,
        pivot=pivot,
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
        max_n=max_n,
        max_evt=max_evt,
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=int(stream),
    )

    ctx.x_idx_a = cp.empty((nroots, m), dtype=cp.int32)
    ctx.x_val_a = cp.empty((nroots, m), dtype=cp.float64)
    ctx.x_idx_b = cp.empty((nroots, m), dtype=cp.int32)
    ctx.x_val_b = cp.empty((nroots, m), dtype=cp.float64)
    ctx.nnz = np.zeros(nroots, dtype=np.int32)
    ctx.nnz_next = np.zeros(nroots, dtype=np.int32)

    ctx.evt_idx = cp.empty(max_evt, dtype=cp.int32)
    ctx.evt_val = cp.empty(max_evt, dtype=cp.float64)
    ctx.idx_all = cp.empty(max_n, dtype=cp.int32)
    ctx.val_all = cp.empty(max_n, dtype=cp.float64)
    ctx.idx_u = cp.empty(max_n, dtype=cp.int32)
    ctx.val_u = cp.empty(max_n, dtype=cp.float64)
    ctx.nnz_u = cp.empty(1, dtype=cp.int32)
    ctx.nnz_out = cp.empty(1, dtype=cp.int32)

    # Optional semi-stochastic deterministic subspace.
    if det_idx is not None:
        det_idx_i32 = np.asarray(det_idx, dtype=np.int32).ravel()
        if det_idx_i32.size:
            det_idx_i32 = np.unique(det_idx_i32)
            det_idx_i32.sort()
            ctx.det_idx_host = det_idx_i32
            ctx.det_idx_dev = cp.asarray(det_idx_i32, dtype=cp.int32)
            ctx.det_idx_buf = cp.empty(int(det_idx_i32.size), dtype=cp.int32)
            ctx.det_val_buf = cp.empty(int(det_idx_i32.size), dtype=cp.float64)
            ctx.det_cols = _build_det_subspace_cols_dense(
                drt=drt,
                h1e=np.asarray(h1e, dtype=np.float64),
                eri=eri4,
                det_idx=det_idx_i32,
                max_out=int(det_max_out),
                state_cache=cache,
            )
            # Build a device CSR once so runtime deterministic correction stays on GPU.
            try:
                import cupyx.scipy.sparse as cpx_sparse  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("deterministic-subspace CUDA correction requires cupyx.scipy.sparse") from e
            row_h, col_h, dat_h = _det_subspace_coo_from_cols(ctx.det_cols)
            ndet = int(det_idx_i32.size)
            if int(dat_h.size) == 0:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix((ndet, ndet), dtype=cp.float64)
            else:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix(
                    (
                        cp.asarray(dat_h, dtype=cp.float64),
                        (
                            cp.asarray(row_h, dtype=cp.int32),
                            cp.asarray(col_h, dtype=cp.int32),
                        ),
                    ),
                    shape=(ndet, ndet),
                    dtype=cp.float64,
                )
            ctx.det_x_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_y_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_spawn_slot_offsets = cp.arange(int(nspawn_total), dtype=cp.int64)
    return ctx


def cuda_projector_step_hamiltonian_u64_ws(
    ctx: CudaProjectorContextKey64,
    *,
    eps: float,
    initiator_t: float = 0.0,
    initiator_t_dev: Any | None = None,
    seed_spawn: int,
    seed_phi: int,
    scale_identity: float = 1.0,
    sync: bool = True,
) -> int:
    """One Key64 projector step on GPU: spawn + identity merge + coalesce + Φ (workspace path).

    Notes
    -----
    Requires ``sync=True`` because the host reads device-written nnz counters
    before swapping buffers.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not bool(sync):
        raise ValueError("cuda_projector_step_hamiltonian_u64_ws currently requires sync=True")
    if ctx.ws is None or ctx.drt_dev is None:
        raise RuntimeError("CudaProjectorContextKey64 is released")
    if ctx.x_key is None or ctx.x_val is None or ctx.x_key_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaProjectorContextKey64 buffers not initialized")
    if ctx.key_all is None or ctx.val_all is None:
        raise RuntimeError("CudaProjectorContextKey64 buffers not initialized")
    if ctx.key_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaProjectorContextKey64 buffers not initialized")

    nnz = int(ctx.nnz)
    if nnz <= 0:
        raise ValueError("current x is empty")
    if nnz > int(ctx.m):
        raise ValueError("current nnz exceeds context m")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    out_len = nnz * nspawn_total
    all_len = nnz + out_len
    if all_len > int(ctx.max_n):
        raise RuntimeError(f"merged buffer length {all_len} exceeds max_n={int(ctx.max_n)}")

    x_key = ctx.x_key[:nnz]
    x_val = ctx.x_val[:nnz]

    key_all = ctx.key_all
    val_all = ctx.val_all
    evt_key = key_all[nnz:all_len]
    evt_val = val_all[nnz:all_len]

    label_mode = str(getattr(ctx, "label_mode", "key64")).strip().lower()
    if label_mode not in ("key64", "idx64"):
        raise ValueError("ctx.label_mode must be 'key64' or 'idx64'")

    if initiator_t_dev is not None:
        if float(initiator_t) != 0.0:
            raise ValueError("provide either initiator_t or initiator_t_dev (not both)")
        initiator_t_dev = cp.asarray(initiator_t_dev, dtype=cp.float64)
        if int(initiator_t_dev.size) != 1:
            raise ValueError("initiator_t_dev must be a device scalar (shape () or (1,))")
        if label_mode == "idx64":
            _guga_cuda_ext.qmc_spawn_hamiltonian_idx64_u64_inplace_device_initiator_dev(
                ctx.drt_dev,
                int(ctx.ncsf_u64),
                x_key,
                x_val,
                ctx.h_base_flat_dev,
                ctx.eri_mat_dev,
                evt_key,
                evt_val,
                float(eps),
                int(ctx.nspawn_one),
                int(ctx.nspawn_two),
                int(seed_spawn),
                initiator_t_dev,
                int(ctx.threads_spawn),
                int(ctx.stream),
                False,
                ctx.pair_alias_prob_dev,
                ctx.pair_alias_idx_dev,
                ctx.pair_norm_dev,
                float(ctx.pair_norm_sum),
                int(ctx.pair_sampling_mode),
            )
        else:
            _guga_cuda_ext.qmc_spawn_hamiltonian_u64_inplace_device_initiator_dev(
                ctx.drt_dev,
                x_key,
                x_val,
                ctx.h_base_flat_dev,
                ctx.eri_mat_dev,
                evt_key,
                evt_val,
                float(eps),
                int(ctx.nspawn_one),
                int(ctx.nspawn_two),
                int(seed_spawn),
                initiator_t_dev,
                int(ctx.threads_spawn),
                int(ctx.stream),
                False,
                ctx.pair_alias_prob_dev,
                ctx.pair_alias_idx_dev,
                ctx.pair_norm_dev,
                float(ctx.pair_norm_sum),
                int(ctx.pair_sampling_mode),
            )
    else:
        if label_mode == "idx64":
            _guga_cuda_ext.qmc_spawn_hamiltonian_idx64_u64_inplace_device(
                ctx.drt_dev,
                int(ctx.ncsf_u64),
                x_key,
                x_val,
                ctx.h_base_flat_dev,
                ctx.eri_mat_dev,
                evt_key,
                evt_val,
                float(eps),
                int(ctx.nspawn_one),
                int(ctx.nspawn_two),
                int(seed_spawn),
                float(initiator_t),
                int(ctx.threads_spawn),
                int(ctx.stream),
                False,
                ctx.pair_alias_prob_dev,
                ctx.pair_alias_idx_dev,
                ctx.pair_norm_dev,
                float(ctx.pair_norm_sum),
                int(ctx.pair_sampling_mode),
            )
        else:
            _guga_cuda_ext.qmc_spawn_hamiltonian_u64_inplace_device(
                ctx.drt_dev,
                x_key,
                x_val,
                ctx.h_base_flat_dev,
                ctx.eri_mat_dev,
                evt_key,
                evt_val,
                float(eps),
                int(ctx.nspawn_one),
                int(ctx.nspawn_two),
                int(seed_spawn),
                float(initiator_t),
                int(ctx.threads_spawn),
                int(ctx.stream),
                False,
                ctx.pair_alias_prob_dev,
                ctx.pair_alias_idx_dev,
                ctx.pair_norm_dev,
                float(ctx.pair_norm_sum),
                int(ctx.pair_sampling_mode),
            )

    # Compact spawn events before coalesce to reduce sort/reduce volume.
    evt_key_c, evt_val_c, n_evt = _compact_spawn_events_u64(evt_key, evt_val)
    all_len_eff = int(nnz + n_evt)

    # Merge identity and (possibly compacted) events.
    key_all[:nnz] = x_key
    val_all[:nnz] = float(scale_identity) * x_val
    if n_evt > 0:
        key_all[nnz:all_len_eff] = evt_key_c
        val_all[nnz:all_len_eff] = evt_val_c

    # Coalesce into (key_u, val_u) with out_nnz in nnz_u.
    ctx.ws.coalesce_coo_u64_f64_inplace_device(
        key_all,
        val_all,
        ctx.key_u,
        ctx.val_u,
        ctx.nnz_u,
        int(all_len_eff),
        int(ctx.threads_qmc),
        int(ctx.stream),
        False,
    )
    n_in = int(cp.asnumpy(ctx.nnz_u)[0])
    if n_in < 0:
        n_in = 0

    # Φ compression into the next x buffer.
    ctx.ws.phi_pivot_resample_u64_f64_inplace_device(
        ctx.key_u,
        ctx.val_u,
        ctx.x_key_next,
        ctx.x_val_next,
        ctx.nnz_out,
        int(n_in),
        int(ctx.m),
        int(ctx.pivot),
        int(seed_phi),
        int(ctx.threads_qmc),
        int(ctx.stream),
        bool(sync),
    )
    nnz_out = int(cp.asnumpy(ctx.nnz_out)[0])
    if nnz_out < 0:
        nnz_out = 0

    ctx.nnz = nnz_out
    ctx.use_a = not ctx.use_a
    return nnz_out


def cuda_projector_step_hamiltonian_idx64_ws(
    ctx: CudaProjectorContextKey64,
    *,
    eps: float,
    initiator_t: float = 0.0,
    initiator_t_dev: Any | None = None,
    seed_spawn: int,
    seed_phi: int,
    scale_identity: float = 1.0,
    sync: bool = True,
) -> int:
    """One idx64 projector step on GPU (uint64 global CSF indices)."""

    if str(getattr(ctx, "label_mode", "key64")).strip().lower() != "idx64":
        raise ValueError("cuda_projector_step_hamiltonian_idx64_ws requires ctx.label_mode='idx64'")
    return cuda_projector_step_hamiltonian_u64_ws(
        ctx,
        eps=float(eps),
        initiator_t=float(initiator_t),
        initiator_t_dev=initiator_t_dev,
        seed_spawn=int(seed_spawn),
        seed_phi=int(seed_phi),
        scale_identity=float(scale_identity),
        sync=bool(sync),
    )


def cuda_block_projector_step_hamiltonian_ws(
    ctx: CudaBlockProjectorContext,
    *,
    eps: float,
    initiator_t: list[float] | np.ndarray,
    seed_spawn: list[int] | np.ndarray,
    seed_phi: list[int] | np.ndarray,
    scale_identity: float | list[float] | np.ndarray = 1.0,
    sync: bool = True,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | list[Callable[..., tuple[np.ndarray, np.ndarray]]] | tuple[Callable[..., tuple[np.ndarray, np.ndarray]], ...] | None = None,
) -> np.ndarray:
    """Projector step for all roots: spawn + identity merge + coalesce + Φ (workspace path).

    Notes
    -----
    This workspace path currently requires ``sync=True`` because the host code
    consumes device-written nnz counters before advancing each root.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not bool(sync):
        raise ValueError("cuda_block_projector_step_hamiltonian_ws currently requires sync=True")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None or ctx.nnz_next is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.evt_idx is None or ctx.evt_val is None or ctx.idx_all is None or ctx.val_all is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    initiator_t = np.asarray(initiator_t, dtype=np.float64).ravel()
    seed_spawn = np.asarray(seed_spawn, dtype=np.int64).ravel()
    seed_phi = np.asarray(seed_phi, dtype=np.int64).ravel()
    if initiator_t.size != int(ctx.nroots) or seed_spawn.size != int(ctx.nroots) or seed_phi.size != int(ctx.nroots):
        raise ValueError("initiator_t, seed_spawn, seed_phi must have length nroots")

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    nnz_out = np.zeros(int(ctx.nroots), dtype=np.int32)

    if isinstance(scale_identity, (list, tuple, np.ndarray)):
        scale_identity_arr = np.asarray(scale_identity, dtype=np.float64).ravel()
        if scale_identity_arr.size != int(ctx.nroots):
            raise ValueError("scale_identity array must have length nroots")
    else:
        scale_identity_arr = None
        scale_identity = float(scale_identity)

    for k in range(int(ctx.nroots)):
        si_k = float(scale_identity_arr[k]) if scale_identity_arr is not None else float(scale_identity)
        nnz_k = int(ctx.nnz[k])
        if nnz_k <= 0:
            raise ValueError(f"column {k} is empty")
        if nnz_k > int(ctx.m):
            raise ValueError(f"column {k} nnz exceeds m")

        out_len = nnz_k * nspawn_total

        # Spawn events directly into the merged buffer tail to avoid an extra D2D copy.
        idx_all = ctx.idx_all
        val_all = ctx.val_all
        evt_idx = idx_all[nnz_k : nnz_k + out_len]
        evt_val = val_all[nnz_k : nnz_k + out_len]
        x_idx = ctx.x_idx[k, :nnz_k]
        x_val = ctx.x_val[k, :nnz_k]

        _guga_cuda_ext.qmc_spawn_hamiltonian_inplace_device(
            ctx.drt_dev,
            ctx.state_dev,
            x_idx,
            x_val,
            ctx.h_base_flat_dev,
            ctx.eri_mat_dev,
            evt_idx,
            evt_val,
            float(eps),
            int(ctx.nspawn_one),
            int(ctx.nspawn_two),
            int(seed_spawn[k]),
            float(initiator_t[k]),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
        )

        # Optional semi-stochastic deterministic subspace correction (fully on GPU).
        nnz_det = _apply_det_subspace_correction_gpu(
            ctx=ctx,
            x_idx=x_idx,
            x_val=x_val,
            evt_idx=evt_idx,
            evt_val=evt_val,
            eps=float(eps),
            nspawn_total=int(nspawn_total),
        )

        all_len = nnz_k + out_len + int(nnz_det)
        idx_all[:nnz_k] = x_idx
        val_all[:nnz_k] = float(si_k) * x_val
        if nnz_det:
            if ctx.det_idx_buf is None or ctx.det_val_buf is None:
                raise RuntimeError("deterministic subspace buffers not initialized")
            idx_all[nnz_k + out_len : all_len] = ctx.det_idx_buf[:nnz_det]
            val_all[nnz_k + out_len : all_len] = ctx.det_val_buf[:nnz_det]

        ctx.ws.coalesce_coo_i32_f64_inplace_device(
            idx_all,
            val_all,
            ctx.idx_u,
            ctx.val_u,
            ctx.nnz_u,
            int(all_len),
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )
        n_in = int(cp.asnumpy(ctx.nnz_u)[0])
        if n_in < 0:
            n_in = 0

        comp_k = compressor
        if isinstance(compressor, (list, tuple)):
            if k >= len(compressor):
                raise ValueError("compressor sequence is shorter than number of roots")
            comp_k = compressor[k]

        if comp_k is not None:
            if not bool(sync):
                raise ValueError("host compressor requires sync=True")

            from .sparse import coalesce_coo_i32_f64  # noqa: PLC0415

            idx_u_host = cp.asnumpy(ctx.idx_u[:n_in]).astype(np.int32, copy=False)
            val_u_host = cp.asnumpy(ctx.val_u[:n_in]).astype(np.float64, copy=False)
            rng_phi = np.random.default_rng(int(seed_phi[k]))
            idx_out_host, val_out_host = comp_k(
                idx_u_host,
                val_u_host,
                m=int(ctx.m),
                pivot=int(ctx.pivot),
                rng=rng_phi,
            )
            idx_out_host, val_out_host = coalesce_coo_i32_f64(idx_out_host, val_out_host)

            nnz_k_out = int(idx_out_host.size)
            if nnz_k_out < 0:
                nnz_k_out = 0
            if nnz_k_out > int(ctx.m):
                raise RuntimeError(f"compressor returned nnz={nnz_k_out} (>m={int(ctx.m)})")

            if nnz_k_out > 0:
                ctx.x_idx_next[k, :nnz_k_out] = cp.asarray(idx_out_host, dtype=cp.int32)
                ctx.x_val_next[k, :nnz_k_out] = cp.asarray(val_out_host, dtype=cp.float64)

            ctx.nnz_next[k] = np.int32(nnz_k_out)
            nnz_out[k] = np.int32(nnz_k_out)
            continue

        ctx.ws.phi_pivot_resample_i32_f64_inplace_device(
            ctx.idx_u,
            ctx.val_u,
            ctx.x_idx_next[k],
            ctx.x_val_next[k],
            ctx.nnz_out,
            int(n_in),
            int(ctx.m),
            int(ctx.pivot),
            int(seed_phi[k]),
            int(ctx.threads_qmc),
            int(ctx.stream),
            bool(sync),
        )
        nnz_k_out = int(cp.asnumpy(ctx.nnz_out)[0])
        if nnz_k_out < 0:
            nnz_k_out = 0
        ctx.nnz_next[k] = np.int32(nnz_k_out)
        nnz_out[k] = np.int32(nnz_k_out)

    # Swap buffers.
    ctx.nnz[:] = ctx.nnz_next
    ctx.use_a = not ctx.use_a
    return nnz_out


def cuda_block_orthonormalize_mgs_ws(
    ctx: CudaBlockProjectorContext,
    *,
    seeds_phi: list[int] | np.ndarray,
    sync: bool = True,
    use_fused: bool = True,
) -> None:
    """Modified Gram-Schmidt orthonormalization on GPU with Φ compression.

    This performs the same packed sparse-column MGS orthonormalization used by
    the scalable block FCI-FRI path inside `CudaBlockProjectorContext`.

    If `use_fused=True` (default), overlaps against all previous columns are computed
    in one batched dot call and the projection is applied as a single sparse linear
    combination (one `coalesce + Φ` per output column).

    If `use_fused=False`, a legacy reference-oriented sequential MGS update is used
    (projection + `coalesce + Φ` after each overlap).

    Notes
    -----
    This helper currently requires `sync=True`. It reads device-side nnz counters
    back to the host after workspace kernels, so an asynchronous contract would be
    misleading until the implementation is refactored.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None or ctx.nnz_next is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.idx_all is None or ctx.val_all is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")
    if not sync:
        raise ValueError("cuda_block_orthonormalize_mgs_ws currently requires sync=True")

    seeds_phi = np.asarray(seeds_phi, dtype=np.int64).ravel()
    nroots = int(ctx.nroots)
    if bool(use_fused):
        need = nroots  # one Φ seed per output column (upper bound; used only when projections occur)
    else:
        need = nroots * (nroots - 1) // 2  # legacy sequential path
    if seeds_phi.size < need:
        raise ValueError(f"seeds_phi has wrong size: {seeds_phi.size} (need >= {need})")

    m = int(ctx.m)
    pivot = int(ctx.pivot)
    if m < 1:
        raise ValueError("m must be >= 1")
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    if bool(use_fused):
        # Fused path: for each k, compute all overlaps r_ik = <q_i, x_k> in one batched dot call,
        # then form y_k = x_k - sum_i r_ik q_i as a single sparse linear combination (one coalesce+Φ).
        if not hasattr(ctx.ws, "sparse_dot_many_sorted_i32_f64_inplace_device"):
            raise RuntimeError("QmcWorkspace.sparse_dot_many_sorted_i32_f64_inplace_device is unavailable (rebuild extension)")

        seed_pos = 0
        ctx.nnz_next[:] = 0

        # Worst-case pack length: nnz(x_k) + sum_i nnz(q_i) <= (k+1)*m <= nroots*m.
        max_all_len = int(nroots) * int(m)
        ws = ctx.ws
        idx_all = ctx.idx_all
        val_all = ctx.val_all
        idx_u = ctx.idx_u
        val_u = ctx.val_u
        nnz_u = ctx.nnz_u
        nnz_out = ctx.nnz_out

        if max_all_len > int(ws.max_n):
            if ctx.ws_big is None or int(ctx.ws_big.max_n) < int(max_all_len):
                ctx.ws_big = _guga_cuda_ext.QmcWorkspace(int(max_all_len), int(m))
            ws = ctx.ws_big

            if ctx.idx_all_big is None or ctx.val_all_big is None or int(ctx.idx_all_big.size) < int(max_all_len):
                ctx.idx_all_big = cp.empty(int(max_all_len), dtype=cp.int32)
                ctx.val_all_big = cp.empty(int(max_all_len), dtype=cp.float64)
                ctx.idx_u_big = cp.empty(int(max_all_len), dtype=cp.int32)
                ctx.val_u_big = cp.empty(int(max_all_len), dtype=cp.float64)
            if ctx.idx_u_big is None or ctx.val_u_big is None:
                raise RuntimeError("internal: big coalesce buffers missing")
            if ctx.nnz_u_big is None or ctx.nnz_out_big is None:
                ctx.nnz_u_big = cp.empty(1, dtype=cp.int32)
                ctx.nnz_out_big = cp.empty(1, dtype=cp.int32)

            idx_all = ctx.idx_all_big
            val_all = ctx.val_all_big
            idx_u = ctx.idx_u_big
            val_u = ctx.val_u_big
            nnz_u = ctx.nnz_u_big
            nnz_out = ctx.nnz_out_big

        # Reused buffers for overlap computation.
        ov_buf = cp.empty((max(nroots - 1, 1), 1), dtype=cp.float64)
        nnz_b = cp.empty((1,), dtype=cp.int32)

        for k in range(nroots):
            nnz_k_in = int(ctx.nnz[k])
            if nnz_k_in <= 0:
                raise ValueError(f"column {k} is empty")
            if nnz_k_in > m:
                raise ValueError(f"column {k} nnz exceeds m")

            # First column: just normalize.
            if k == 0:
                ctx.x_idx_next[0, :nnz_k_in] = ctx.x_idx[0, :nnz_k_in]
                ctx.x_val_next[0, :nnz_k_in] = ctx.x_val[0, :nnz_k_in]
                n2 = float(cp.linalg.norm(ctx.x_val_next[0, :nnz_k_in]).get())
                if n2 == 0.0:
                    raise RuntimeError("column 0 collapsed to zero norm during orthonormalization")
                ctx.x_val_next[0, :nnz_k_in] /= n2
                ctx.nnz_next[0] = np.int32(nnz_k_in)
                continue

            # Overlaps r_ik = <q_i, x_k> for i<k (q_i are in the output buffer already).
            q_nnz_dev = cp.asarray(np.asarray(ctx.nnz_next[:k], dtype=np.int32), dtype=cp.int32)
            nnz_b[0] = np.int32(nnz_k_in)
            ctx.ws.sparse_dot_many_sorted_i32_f64_inplace_device(
                ctx.x_idx_next[:k],
                ctx.x_val_next[:k],
                q_nnz_dev,
                ctx.x_idx[k],
                ctx.x_val[k],
                nnz_b,
                ov_buf[:k],
                int(ctx.threads_qmc),
                int(ctx.stream),
                False,
            )
            ov_host = cp.asnumpy(ov_buf[:k, 0]).astype(np.float64, copy=False)
            nz = np.nonzero(ov_host != 0.0)[0]
            if nz.size == 0:
                # Already orthogonal in the sparse support sense; just normalize and move on.
                ctx.x_idx_next[k, :nnz_k_in] = ctx.x_idx[k, :nnz_k_in]
                ctx.x_val_next[k, :nnz_k_in] = ctx.x_val[k, :nnz_k_in]
                n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_k_in]).get())
                if n2 == 0.0:
                    raise RuntimeError(f"column {k} collapsed to zero norm during orthonormalization")
                ctx.x_val_next[k, :nnz_k_in] /= n2
                ctx.nnz_next[k] = np.int32(nnz_k_in)
                continue

            # Pack y_k = x_k - sum_{i<k} ov_i * q_i.
            all_len = nnz_k_in
            for ii in nz:
                all_len += int(ctx.nnz_next[int(ii)])
            if all_len > int(ws.max_n):
                raise RuntimeError("MGS fused update exceeds workspace capacity (increase max_n)")

            idx_all[:nnz_k_in] = ctx.x_idx[k, :nnz_k_in]
            val_all[:nnz_k_in] = ctx.x_val[k, :nnz_k_in]
            pos = nnz_k_in
            for ii in nz.tolist():
                i = int(ii)
                nnz_i = int(ctx.nnz_next[i])
                if nnz_i <= 0:
                    raise RuntimeError(f"column {i} is empty during orthonormalization")
                idx_all[pos : pos + nnz_i] = ctx.x_idx_next[i, :nnz_i]
                val_all[pos : pos + nnz_i] = (-float(ov_host[i])) * ctx.x_val_next[i, :nnz_i]
                pos += nnz_i
            if pos != all_len:
                raise RuntimeError("internal: MGS fused pack length mismatch")

            ws.coalesce_coo_i32_f64_inplace_device(
                idx_all,
                val_all,
                idx_u,
                val_u,
                nnz_u,
                int(all_len),
                int(ctx.threads_qmc),
                int(ctx.stream),
                False,
            )
            n_in = int(cp.asnumpy(nnz_u)[0])
            if n_in <= 0:
                raise RuntimeError(f"column {k} annihilated during orthonormalization (after fused projection)")

            seed_phi = int(seeds_phi[seed_pos])
            seed_pos += 1
            ws.phi_pivot_resample_i32_f64_inplace_device(
                idx_u,
                val_u,
                ctx.x_idx_next[k],
                ctx.x_val_next[k],
                nnz_out,
                int(n_in),
                int(m),
                int(pivot),
                int(seed_phi),
                int(ctx.threads_qmc),
                int(ctx.stream),
                bool(sync),
            )
            nnz_cur = int(cp.asnumpy(nnz_out)[0])
            if nnz_cur <= 0:
                raise RuntimeError(f"column {k} annihilated during orthonormalization (after Φ; fused)")

            n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_cur]).get())
            if n2 == 0.0:
                raise RuntimeError(f"column {k} collapsed to zero norm during orthonormalization")
            ctx.x_val_next[k, :nnz_cur] /= n2
            ctx.nnz_next[k] = np.int32(nnz_cur)

        # Swap buffers.
        ctx.nnz[:] = ctx.nnz_next
        ctx.use_a = not ctx.use_a
        return

    if not hasattr(ctx.ws, "sparse_dot_many_sorted_i32_f64_inplace_device"):
        raise RuntimeError("QmcWorkspace.sparse_dot_many_sorted_i32_f64_inplace_device is unavailable (rebuild extension)")

    _dot_out = cp.empty((1, 1), dtype=cp.float64)
    _dot_nnz_a = cp.empty((1,), dtype=cp.int32)
    _dot_nnz_b = cp.empty((1,), dtype=cp.int32)

    def dot_sparse_sorted(a_idx, a_val, b_idx, b_val) -> float:
        a_idx = a_idx.ravel()
        a_val = a_val.ravel()
        b_idx = b_idx.ravel()
        b_val = b_val.ravel()
        na = int(a_idx.size)
        nb = int(b_idx.size)
        if na == 0 or nb == 0:
            return 0.0

        _dot_nnz_a[0] = np.int32(na)
        _dot_nnz_b[0] = np.int32(nb)
        ctx.ws.sparse_dot_many_sorted_i32_f64_inplace_device(
            a_idx,
            a_val,
            _dot_nnz_a,
            b_idx,
            b_val,
            _dot_nnz_b,
            _dot_out,
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )
        return float(_dot_out[0, 0].get())

    seed_pos = 0
    ctx.nnz_next[:] = 0

    # Build orthonormalized columns sequentially into the alternate buffer.
    for k in range(nroots):
        nnz_k_in = int(ctx.nnz[k])
        if nnz_k_in <= 0:
            raise ValueError(f"column {k} is empty")
        if nnz_k_in > m:
            raise ValueError(f"column {k} nnz exceeds m")

        # Initialize current column in the output buffer.
        ctx.x_idx_next[k, :nnz_k_in] = ctx.x_idx[k, :nnz_k_in]
        ctx.x_val_next[k, :nnz_k_in] = ctx.x_val[k, :nnz_k_in]
        nnz_cur = nnz_k_in

        for i in range(k):
            nnz_i = int(ctx.nnz_next[i])
            if nnz_i <= 0:
                raise RuntimeError(f"column {i} is empty during orthonormalization")

            a_idx = ctx.x_idx_next[i, :nnz_i]
            a_val = ctx.x_val_next[i, :nnz_i]
            b_idx = ctx.x_idx_next[k, :nnz_cur]
            b_val = ctx.x_val_next[k, :nnz_cur]
            ov = dot_sparse_sorted(a_idx, a_val, b_idx, b_val)

            seed_phi = int(seeds_phi[seed_pos])
            seed_pos += 1

            if ov != 0.0:
                all_len = nnz_cur + nnz_i
                if all_len > int(ctx.max_n):
                    raise RuntimeError("MGS update exceeds workspace capacity (increase max_n)")

                ctx.idx_all[:nnz_cur] = b_idx
                ctx.idx_all[nnz_cur:all_len] = a_idx
                ctx.val_all[:nnz_cur] = b_val
                ctx.val_all[nnz_cur:all_len] = (-float(ov)) * a_val

                ctx.ws.coalesce_coo_i32_f64_inplace_device(
                    ctx.idx_all,
                    ctx.val_all,
                    ctx.idx_u,
                    ctx.val_u,
                    ctx.nnz_u,
                    int(all_len),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    False,
                )
                n_in = int(cp.asnumpy(ctx.nnz_u)[0])
                if n_in <= 0:
                    raise RuntimeError(f"column {k} annihilated during orthonormalization (proj on {i})")

                ctx.ws.phi_pivot_resample_i32_f64_inplace_device(
                    ctx.idx_u,
                    ctx.val_u,
                    ctx.x_idx_next[k],
                    ctx.x_val_next[k],
                    ctx.nnz_out,
                    int(n_in),
                    int(m),
                    int(pivot),
                    int(seed_phi),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    bool(sync),
                )
                nnz_cur = int(cp.asnumpy(ctx.nnz_out)[0])
                if nnz_cur <= 0:
                    raise RuntimeError(f"column {k} annihilated during orthonormalization (after Φ; proj on {i})")

        # Normalize.
        n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_cur]).get())
        if n2 == 0.0:
            raise RuntimeError(f"column {k} collapsed to zero norm during orthonormalization")
        ctx.x_val_next[k, :nnz_cur] /= n2
        ctx.nnz_next[k] = np.int32(nnz_cur)

    # Swap buffers.
    ctx.nnz[:] = ctx.nnz_next
    ctx.use_a = not ctx.use_a


def cuda_block_build_tmat_hamiltonian_stochastic_ws(
    ctx: CudaBlockProjectorContext,
    *,
    seeds_spawn: list[int] | np.ndarray,
    eps: float = 1.0,
    initiator_t: float = 0.0,
    sync: bool = True,
) -> np.ndarray:
    """Build a stochastic Ritz matrix T = X^T H X on GPU (dense ERI path).

    This computes an unbiased stochastic estimator of Hx_k for each column x_k using
    the existing QMC spawn engine (events-only, no Φ), then forms:
        T[i,k] = <x_i, H x_k>

    Notes
    -----
    - This is a reference-oriented implementation used by the scalable block
      FCI-FRI Ritz build.
    - `initiator_t` defaults to 0.0 to avoid adding initiator bias to the Ritz estimator.
    - This helper returns a host NumPy array, so it will synchronize before returning.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.evt_idx is None or ctx.evt_val is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")

    seeds_spawn = np.asarray(seeds_spawn, dtype=np.int64).ravel()
    nroots = int(ctx.nroots)
    if seeds_spawn.size != nroots:
        raise ValueError(f"seeds_spawn must have length nroots={nroots}")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    # Accumulate T on device, then transfer once.
    tmat = cp.zeros((nroots, nroots), dtype=cp.float64)

    if not hasattr(ctx.ws, "sparse_dot_many_sorted_i32_f64_inplace_device"):
        raise RuntimeError("QmcWorkspace.sparse_dot_many_sorted_i32_f64_inplace_device is unavailable (rebuild extension)")

    x_nnz_dev = cp.asarray(np.asarray(ctx.nnz, dtype=np.int32), dtype=cp.int32)
    dots = cp.empty((nroots, 1), dtype=cp.float64)

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    for k in range(nroots):
        nnz_k = int(ctx.nnz[k])
        if nnz_k <= 0:
            raise ValueError(f"column {k} is empty")
        if nnz_k > int(ctx.m):
            raise ValueError(f"column {k} nnz exceeds m")

        out_len = nnz_k * nspawn_total
        evt_idx = ctx.evt_idx[:out_len]
        evt_val = ctx.evt_val[:out_len]
        x_idx = ctx.x_idx[k, :nnz_k]
        x_val = ctx.x_val[k, :nnz_k]

        _guga_cuda_ext.qmc_spawn_hamiltonian_inplace_device(
            ctx.drt_dev,
            ctx.state_dev,
            x_idx,
            x_val,
            ctx.h_base_flat_dev,
            ctx.eri_mat_dev,
            evt_idx,
            evt_val,
            float(eps),
            int(ctx.nspawn_one),
            int(ctx.nspawn_two),
            int(seeds_spawn[k]),
            float(initiator_t),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
        )

        # Coalesce events into (idx_u,val_u) with out_nnz in nnz_u.
        ctx.ws.coalesce_coo_i32_f64_inplace_device(
            evt_idx,
            evt_val,
            ctx.idx_u,
            ctx.val_u,
            ctx.nnz_u,
            int(out_len),
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )

        # events represent (-eps * H x_k); for eps=1: hk = -(H x_k)
        # => <x_i, H x_k> = - <x_i, hk>
        # Avoid host sync: use device nnz_u and the full (idx_u,val_u) buffers.
        ctx.ws.sparse_dot_many_sorted_i32_f64_inplace_device(
            ctx.x_idx,
            ctx.x_val,
            x_nnz_dev,
            ctx.idx_u,
            ctx.val_u,
            ctx.nnz_u,
            dots,
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )
        tmat[:, k] = -dots[:, 0]

    tmat = 0.5 * (tmat + tmat.T)
    return cp.asnumpy(tmat).astype(np.float64, copy=False)


def cuda_block_build_sk_uthx_stochastic_ws(
    ctx: CudaBlockProjectorContext,
    *,
    u_cols: list[tuple[np.ndarray, np.ndarray]],
    seeds_spawn: list[int] | np.ndarray,
    eps: float = 1.0,
    initiator_t: float = 0.0,
    sync: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RSI evaluation matrices on GPU: S=U^T X and stochastic K=U^T H X.

    Parameters
    ----------
    u_cols
        Trial vectors U as host sparse columns. Length must equal ``ctx.nroots``.
    seeds_spawn
        One spawn seed per root used to build a stochastic estimator of ``H x_k``.
    eps
        Spawn estimator scale. Must be non-zero.

    Notes
    -----
    This helper returns host NumPy arrays, so it will synchronize before returning.
    If the optional deterministic subspace correction is enabled, `sync=True` is required.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.evt_idx is None or ctx.evt_val is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")

    eps = float(eps)
    if eps == 0.0:
        raise ValueError("eps must be non-zero")

    nroots = int(ctx.nroots)
    if len(u_cols) != nroots:
        raise ValueError(f"u_cols must have length nroots={nroots}")

    seeds_spawn = np.asarray(seeds_spawn, dtype=np.int64).ravel()
    if seeds_spawn.size != nroots:
        raise ValueError(f"seeds_spawn must have length nroots={nroots}")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    if not hasattr(ctx.ws, "sparse_dot_many_sorted_i32_f64_inplace_device"):
        raise RuntimeError("QmcWorkspace.sparse_dot_many_sorted_i32_f64_inplace_device is unavailable (rebuild extension)")

    # Upload U columns once per call as packed (nroots, max_u) arrays.
    u_nnz_host = np.zeros(nroots, dtype=np.int32)
    max_u = 0
    u_cols_i: list[np.ndarray] = []
    u_cols_v: list[np.ndarray] = []
    for i, (idx_i, val_i) in enumerate(u_cols):
        idx_i = np.asarray(idx_i, dtype=np.int32).ravel()
        val_i = np.asarray(val_i, dtype=np.float64).ravel()
        if idx_i.size != val_i.size:
            raise ValueError(f"u_cols[{i}] idx/val size mismatch")
        if idx_i.size <= 0:
            raise ValueError(f"u_cols[{i}] is empty")
        u_cols_i.append(idx_i)
        u_cols_v.append(val_i)
        u_nnz_host[i] = np.int32(idx_i.size)
        max_u = max(max_u, int(idx_i.size))

    u_idx_host = np.zeros((nroots, max_u), dtype=np.int32)
    u_val_host = np.zeros((nroots, max_u), dtype=np.float64)
    for i in range(nroots):
        nnz_i = int(u_nnz_host[i])
        u_idx_host[i, :nnz_i] = u_cols_i[i]
        u_val_host[i, :nnz_i] = u_cols_v[i]

    u_idx_dev = cp.asarray(u_idx_host, dtype=cp.int32)
    u_val_dev = cp.asarray(u_val_host, dtype=cp.float64)
    u_nnz_dev = cp.asarray(u_nnz_host, dtype=cp.int32)

    # Upload packed nnz for X once.
    for j in range(nroots):
        if int(ctx.nnz[j]) <= 0:
            raise ValueError(f"column {j} is empty")
    x_nnz_dev = cp.asarray(np.asarray(ctx.nnz, dtype=np.int32), dtype=cp.int32)

    smat = cp.empty((nroots, nroots), dtype=cp.float64)
    kmat = cp.zeros((nroots, nroots), dtype=cp.float64)

    ctx.ws.sparse_dot_many_sorted_i32_f64_inplace_device(
        u_idx_dev,
        u_val_dev,
        u_nnz_dev,
        ctx.x_idx,
        ctx.x_val,
        x_nnz_dev,
        smat,
        int(ctx.threads_qmc),
        int(ctx.stream),
        False,
    )

    dots = cp.empty((nroots, 1), dtype=cp.float64)

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    for k in range(nroots):
        nnz_k = int(ctx.nnz[k])
        if nnz_k <= 0:
            raise ValueError(f"column {k} is empty")
        if nnz_k > int(ctx.m):
            raise ValueError(f"column {k} nnz exceeds m")

        out_len = nnz_k * nspawn_total
        evt_idx = ctx.evt_idx[:out_len]
        evt_val = ctx.evt_val[:out_len]
        x_idx = ctx.x_idx[k, :nnz_k]
        x_val = ctx.x_val[k, :nnz_k]

        _guga_cuda_ext.qmc_spawn_hamiltonian_inplace_device(
            ctx.drt_dev,
            ctx.state_dev,
            x_idx,
            x_val,
            ctx.h_base_flat_dev,
            ctx.eri_mat_dev,
            evt_idx,
            evt_val,
            float(eps),
            int(ctx.nspawn_one),
            int(ctx.nspawn_two),
            int(seeds_spawn[k]),
            float(initiator_t),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
        )

        # Optional semi-stochastic deterministic subspace correction (fully on GPU).
        nnz_det = _apply_det_subspace_correction_gpu(
            ctx=ctx,
            x_idx=x_idx,
            x_val=x_val,
            evt_idx=evt_idx,
            evt_val=evt_val,
            eps=float(eps),
            nspawn_total=int(nspawn_total),
        )

        # Coalesce events (with optional det merge) into (idx_u,val_u) with out_nnz in nnz_u.
        if nnz_det:
            if ctx.idx_all is None or ctx.val_all is None or ctx.det_idx_buf is None or ctx.det_val_buf is None:
                raise RuntimeError("deterministic subspace buffers not initialized")
            all_len = int(out_len) + int(nnz_det)
            if all_len > int(ctx.max_n):
                raise RuntimeError("eval event merge exceeds workspace capacity (increase max_n)")
            ctx.idx_all[:out_len] = evt_idx
            ctx.val_all[:out_len] = evt_val
            ctx.idx_all[out_len:all_len] = ctx.det_idx_buf[:nnz_det]
            ctx.val_all[out_len:all_len] = ctx.det_val_buf[:nnz_det]
            in_idx = ctx.idx_all[:all_len]
            in_val = ctx.val_all[:all_len]
            in_len = int(all_len)
        else:
            in_idx = evt_idx
            in_val = evt_val
            in_len = int(out_len)

        ctx.ws.coalesce_coo_i32_f64_inplace_device(
            in_idx,
            in_val,
            ctx.idx_u,
            ctx.val_u,
            ctx.nnz_u,
            int(in_len),
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )

        ctx.ws.sparse_dot_many_sorted_i32_f64_inplace_device(
            u_idx_dev,
            u_val_dev,
            u_nnz_dev,
            ctx.idx_u,
            ctx.val_u,
            ctx.nnz_u,
            dots,
            int(ctx.threads_qmc),
            int(ctx.stream),
            False,
        )
        kmat[:, k] = -(dots[:, 0] / eps)

    return (
        cp.asnumpy(smat).astype(np.float64, copy=False),
        cp.asnumpy(kmat).astype(np.float64, copy=False),
    )


def cuda_block_apply_right_matrix_phi_ws(
    ctx: CudaBlockProjectorContext,
    *,
    mat: np.ndarray,
    seeds_phi: list[int] | np.ndarray,
    sync: bool = True,
    use_fused: bool = True,
    use_composite_keys: bool = True,
) -> None:
    """Apply `X <- Φ(X @ mat)` on GPU (column-wise), writing into the alternate buffer then swapping.

    If `use_fused=True` (default), each output column is built as a single sparse
    linear combination and then compressed once (`coalesce` + `Φ`). This reduces
    `Φ` calls from O(nroots^2) to O(nroots).

    If `use_composite_keys=True` (default), the fused path additionally performs a
    single global coalesce over 64-bit composite keys ``(out_col << 32) | idx``,
    then runs one `Φ` per output-column segment. This removes O(nroots) separate
    coalesce calls.

    If `use_fused=False`, the legacy incremental update is used:
      `y <- Φ(y + w*x_j)`
    which applies `Φ` after each contributing input column.

    Notes
    -----
    This helper currently requires `sync=True`. It reads device-side nnz counters
    back to the host after workspace kernels, so an asynchronous contract would be
    misleading until the implementation is refactored.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if ctx.ws is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None or ctx.nnz_next is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.idx_all is None or ctx.val_all is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")
    if not sync:
        raise ValueError("cuda_block_apply_right_matrix_phi_ws currently requires sync=True")

    mat = np.asarray(mat, dtype=np.float64)
    nroots = int(ctx.nroots)
    if mat.shape != (nroots, nroots):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(nroots, nroots)})")

    seeds_phi = np.asarray(seeds_phi, dtype=np.int64).ravel()
    if bool(use_fused):
        need = nroots  # one Φ seed per output column
    else:
        need = nroots * (nroots - 1)  # legacy incremental path
    if seeds_phi.size < need:
        raise ValueError(f"seeds_phi has wrong size: {seeds_phi.size} (need >= {need})")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    m = int(ctx.m)
    pivot = int(ctx.pivot)
    if m < 1:
        raise ValueError("m must be >= 1")
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    seed_pos = 0
    ctx.nnz_next[:] = 0

    if bool(use_fused):
        # Fused path: build each output column as a single linear combination, then coalesce+Φ once.
        # This reduces Φ calls from O(nroots^2) to O(nroots).
        nnz_host = np.asarray(ctx.nnz, dtype=np.int32).ravel()

        all_len_by_k = np.zeros(nroots, dtype=np.int64)
        max_all_len = 0
        for k in range(nroots):
            total = 0
            for j in range(nroots):
                if float(mat[j, k]) == 0.0:
                    continue
                nnz_j = int(nnz_host[j])
                if nnz_j <= 0:
                    continue
                total += nnz_j
            all_len_by_k[k] = np.int64(total)
            if total > max_all_len:
                max_all_len = total

        if max_all_len <= 0:
            raise RuntimeError("apply_right_matrix: all output columns are empty (mat may be all zeros)")

        ws_phi = ctx.ws
        idx_tmp = ctx.idx_u
        nnz_out = ctx.nnz_out
        if idx_tmp is None or nnz_out is None:
            raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")

        if max_all_len > int(ws_phi.max_n):
            # Lazily allocate a larger i32 workspace for Φ input columns.
            if ctx.ws_big is None or int(ctx.ws_big.max_n) < int(max_all_len):
                ctx.ws_big = _guga_cuda_ext.QmcWorkspace(int(max_all_len), int(m))
            ws_phi = ctx.ws_big

            if ctx.idx_u_big is None or int(ctx.idx_u_big.size) < int(max_all_len):
                ctx.idx_u_big = cp.empty(int(max_all_len), dtype=cp.int32)
            if ctx.nnz_out_big is None:
                ctx.nnz_out_big = cp.empty(1, dtype=cp.int32)
            if ctx.idx_u_big is None or ctx.nnz_out_big is None:
                raise RuntimeError("internal: big Φ buffers missing")
            idx_tmp = ctx.idx_u_big
            nnz_out = ctx.nnz_out_big

        if bool(use_composite_keys) and not hasattr(_guga_cuda_ext, "QmcWorkspaceU64"):
            # Keep compatibility with older CUDA extension builds: use the fused i32 fallback.
            use_composite_keys = False

        if bool(use_composite_keys):

            total_all_len = int(np.sum(all_len_by_k, dtype=np.int64))
            if total_all_len <= 0:
                raise RuntimeError("apply_right_matrix: all output columns are empty (mat may be all zeros)")

            # Lazily allocate/reuse u64 workspace and buffers for global composite-key coalesce.
            if ctx.ws_u64 is None or int(ctx.ws_u64.max_n) < int(total_all_len):
                if ctx.ws_u64 is not None:
                    ctx.ws_u64.release()
                ctx.ws_u64 = _guga_cuda_ext.QmcWorkspaceU64(int(total_all_len), int(m))

            if (
                ctx.key_all_u64 is None
                or ctx.val_all_u64 is None
                or int(ctx.key_all_u64.size) < int(total_all_len)
            ):
                ctx.key_all_u64 = cp.empty(int(total_all_len), dtype=cp.uint64)
                ctx.val_all_u64 = cp.empty(int(total_all_len), dtype=cp.float64)
            if (
                ctx.key_u_u64 is None
                or ctx.val_u_u64 is None
                or int(ctx.key_u_u64.size) < int(total_all_len)
            ):
                ctx.key_u_u64 = cp.empty(int(total_all_len), dtype=cp.uint64)
                ctx.val_u_u64 = cp.empty(int(total_all_len), dtype=cp.float64)
            if ctx.nnz_u_u64 is None:
                ctx.nnz_u_u64 = cp.empty(1, dtype=cp.int32)

            if (
                ctx.ws_u64 is None
                or ctx.key_all_u64 is None
                or ctx.val_all_u64 is None
                or ctx.key_u_u64 is None
                or ctx.val_u_u64 is None
                or ctx.nnz_u_u64 is None
            ):
                raise RuntimeError("internal: composite-key fused buffers missing")

            key_all_u64 = ctx.key_all_u64
            val_all_u64 = ctx.val_all_u64

            # Pack all (k,j) contributions with composite keys: (k << 32) | idx.
            pos = 0
            low_mask = np.uint64(0xFFFFFFFF)
            for k in range(nroots):
                col_prefix = np.uint64(k) << np.uint64(32)
                for j in range(nroots):
                    w = float(mat[j, k])
                    if w == 0.0:
                        continue
                    nnz_j = int(nnz_host[j])
                    if nnz_j <= 0:
                        continue
                    idx_j_u64 = (ctx.x_idx[j, :nnz_j].astype(cp.uint64, copy=False) & low_mask)
                    key_all_u64[pos : pos + nnz_j] = cp.uint64(col_prefix) | idx_j_u64
                    val_all_u64[pos : pos + nnz_j] = float(w) * ctx.x_val[j, :nnz_j]
                    pos += nnz_j
            if pos != total_all_len:
                raise RuntimeError("internal: apply_right_matrix composite pack length mismatch")

            ctx.ws_u64.coalesce_coo_u64_f64_inplace_device(
                key_all_u64,
                val_all_u64,
                ctx.key_u_u64,
                ctx.val_u_u64,
                ctx.nnz_u_u64,
                int(total_all_len),
                int(ctx.threads_qmc),
                int(ctx.stream),
                False,
            )
            n_compact = int(cp.asnumpy(ctx.nnz_u_u64)[0])
            if n_compact <= 0:
                raise RuntimeError("apply_right_matrix produced an empty global intermediate column set")

            key_u_u64 = ctx.key_u_u64[:n_compact]
            val_u_u64 = ctx.val_u_u64[:n_compact]
            bounds = cp.searchsorted(
                key_u_u64,
                cp.arange(nroots + 1, dtype=cp.uint64) << cp.uint64(32),
            )

            for k in range(nroots):
                lo = int(cp.asnumpy(bounds[k : k + 1])[0])
                hi = int(cp.asnumpy(bounds[k + 1 : k + 2])[0])
                n_in = hi - lo
                if n_in <= 0:
                    raise RuntimeError("apply_right_matrix produced an empty intermediate column")

                # Decode low 32-bit local keys back to int32 indices for i32 Φ.
                idx_tmp[:n_in] = (key_u_u64[lo:hi] & low_mask).astype(cp.int32, copy=False)

                seed_phi = int(seeds_phi[seed_pos])
                seed_pos += 1

                ws_phi.phi_pivot_resample_i32_f64_inplace_device(
                    idx_tmp[:n_in],
                    val_u_u64[lo:hi],
                    ctx.x_idx_next[k],
                    ctx.x_val_next[k],
                    nnz_out,
                    int(n_in),
                    int(m),
                    int(pivot),
                    int(seed_phi),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    bool(sync),
                )
                nnz_cur = int(cp.asnumpy(nnz_out)[0])
                if nnz_cur <= 0:
                    raise RuntimeError("apply_right_matrix annihilated a column during Φ compression")

                n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_cur]).get())
                if n2 == 0.0:
                    raise RuntimeError("apply_right_matrix produced a zero-norm column")
                ctx.x_val_next[k, :nnz_cur] /= n2
                ctx.nnz_next[k] = np.int32(nnz_cur)
        else:
            ws = ctx.ws
            idx_all = ctx.idx_all
            val_all = ctx.val_all
            idx_u = ctx.idx_u
            val_u = ctx.val_u
            nnz_u = ctx.nnz_u
            if idx_all is None or val_all is None or idx_u is None or val_u is None or nnz_u is None:
                raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")
            nnz_out = ctx.nnz_out
            if nnz_out is None:
                raise RuntimeError("CudaBlockProjectorContext nnz_out buffer not initialized")

            if max_all_len > int(ws.max_n):
                # Lazily allocate a larger workspace + scratch buffers for this operation.
                if ctx.ws_big is None or int(ctx.ws_big.max_n) < int(max_all_len):
                    ctx.ws_big = _guga_cuda_ext.QmcWorkspace(int(max_all_len), int(m))
                ws = ctx.ws_big

                if ctx.idx_all_big is None or ctx.val_all_big is None or int(ctx.idx_all_big.size) < int(max_all_len):
                    ctx.idx_all_big = cp.empty(int(max_all_len), dtype=cp.int32)
                    ctx.val_all_big = cp.empty(int(max_all_len), dtype=cp.float64)
                    ctx.idx_u_big = cp.empty(int(max_all_len), dtype=cp.int32)
                    ctx.val_u_big = cp.empty(int(max_all_len), dtype=cp.float64)
                if ctx.idx_u_big is None or ctx.val_u_big is None:
                    raise RuntimeError("internal: big coalesce buffers missing")
                if ctx.nnz_u_big is None or ctx.nnz_out_big is None:
                    ctx.nnz_u_big = cp.empty(1, dtype=cp.int32)
                    ctx.nnz_out_big = cp.empty(1, dtype=cp.int32)

                idx_all = ctx.idx_all_big
                val_all = ctx.val_all_big
                idx_u = ctx.idx_u_big
                val_u = ctx.val_u_big
                nnz_u = ctx.nnz_u_big
                nnz_out = ctx.nnz_out_big

            for k in range(nroots):
                all_len = int(all_len_by_k[k])
                if all_len <= 0:
                    raise RuntimeError("apply_right_matrix produced an empty column")
                if all_len > int(ws.max_n):
                    raise RuntimeError("apply_right_matrix fused path exceeded workspace capacity (increase max_n)")

                # Pack all contributions y <- sum_j mat[j,k] * x_j into (idx_all,val_all).
                pos = 0
                for j in range(nroots):
                    w = float(mat[j, k])
                    if w == 0.0:
                        continue
                    nnz_j = int(nnz_host[j])
                    if nnz_j <= 0:
                        continue
                    idx_all[pos : pos + nnz_j] = ctx.x_idx[j, :nnz_j]
                    val_all[pos : pos + nnz_j] = float(w) * ctx.x_val[j, :nnz_j]
                    pos += nnz_j
                if pos != all_len:
                    raise RuntimeError("internal: apply_right_matrix pack length mismatch")

                ws.coalesce_coo_i32_f64_inplace_device(
                    idx_all,
                    val_all,
                    idx_u,
                    val_u,
                    nnz_u,
                    int(all_len),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    False,
                )
                n_in = int(cp.asnumpy(nnz_u)[0])
                if n_in <= 0:
                    raise RuntimeError("apply_right_matrix produced an empty intermediate column")

                seed_phi = int(seeds_phi[seed_pos])
                seed_pos += 1

                ws.phi_pivot_resample_i32_f64_inplace_device(
                    idx_u,
                    val_u,
                    ctx.x_idx_next[k],
                    ctx.x_val_next[k],
                    nnz_out,
                    int(n_in),
                    int(m),
                    int(pivot),
                    int(seed_phi),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    bool(sync),
                )
                nnz_cur = int(cp.asnumpy(nnz_out)[0])
                if nnz_cur <= 0:
                    raise RuntimeError("apply_right_matrix annihilated a column during Φ compression")

                n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_cur]).get())
                if n2 == 0.0:
                    raise RuntimeError("apply_right_matrix produced a zero-norm column")
                ctx.x_val_next[k, :nnz_cur] /= n2
                ctx.nnz_next[k] = np.int32(nnz_cur)
    else:
        # Legacy incremental path: y <- Φ(y + w*x_j).
        for k in range(nroots):
            nnz_cur = 0
            for j in range(nroots):
                w = float(mat[j, k])
                if w == 0.0:
                    continue
                nnz_j = int(ctx.nnz[j])
                if nnz_j <= 0:
                    continue

                if nnz_cur == 0:
                    ctx.x_idx_next[k, :nnz_j] = ctx.x_idx[j, :nnz_j]
                    ctx.x_val_next[k, :nnz_j] = float(w) * ctx.x_val[j, :nnz_j]
                    nnz_cur = nnz_j
                    continue

                all_len = nnz_cur + nnz_j
                if all_len > int(ctx.max_n):
                    raise RuntimeError("apply_right_matrix update exceeds workspace capacity (increase max_n)")

                ctx.idx_all[:nnz_cur] = ctx.x_idx_next[k, :nnz_cur]
                ctx.idx_all[nnz_cur:all_len] = ctx.x_idx[j, :nnz_j]
                ctx.val_all[:nnz_cur] = ctx.x_val_next[k, :nnz_cur]
                ctx.val_all[nnz_cur:all_len] = float(w) * ctx.x_val[j, :nnz_j]

                ctx.ws.coalesce_coo_i32_f64_inplace_device(
                    ctx.idx_all,
                    ctx.val_all,
                    ctx.idx_u,
                    ctx.val_u,
                    ctx.nnz_u,
                    int(all_len),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    False,
                )
                n_in = int(cp.asnumpy(ctx.nnz_u)[0])
                if n_in <= 0:
                    raise RuntimeError("apply_right_matrix produced an empty intermediate column")

                seed_phi = int(seeds_phi[seed_pos])
                seed_pos += 1

                ctx.ws.phi_pivot_resample_i32_f64_inplace_device(
                    ctx.idx_u,
                    ctx.val_u,
                    ctx.x_idx_next[k],
                    ctx.x_val_next[k],
                    ctx.nnz_out,
                    int(n_in),
                    int(m),
                    int(pivot),
                    int(seed_phi),
                    int(ctx.threads_qmc),
                    int(ctx.stream),
                    bool(sync),
                )
                nnz_cur = int(cp.asnumpy(ctx.nnz_out)[0])
                if nnz_cur <= 0:
                    raise RuntimeError("apply_right_matrix annihilated a column during Φ compression")

            if nnz_cur <= 0:
                raise RuntimeError("apply_right_matrix produced an empty column")

            n2 = float(cp.linalg.norm(ctx.x_val_next[k, :nnz_cur]).get())
            if n2 == 0.0:
                raise RuntimeError("apply_right_matrix produced a zero-norm column")
            ctx.x_val_next[k, :nnz_cur] /= n2
            ctx.nnz_next[k] = np.int32(nnz_cur)

    ctx.nnz[:] = ctx.nnz_next
    ctx.use_a = not ctx.use_a


# ---------------------------------------------------------------------------
# CUDA FCIQMC context (spawn + coalesce, no FRI compression)
# ---------------------------------------------------------------------------


@dataclass
class CudaFCIQMCContext:
    """Reusable GPU context for repeated FCIQMC steps (spawn + coalesce, no compression).

    The walker population is not bounded by a fixed ``m``. It can grow up to
    ``max_walker``, and the shift-based population control in the FCIQMC driver
    keeps the count finite.
    """

    drt_dev: Any
    state_dev: Any
    ws: Any

    h_base_flat_dev: Any
    eri_mat_dev: Any

    nspawn_one: int
    nspawn_two: int
    max_walker: int
    max_n: int
    max_evt: int

    threads_spawn: int = 128
    threads_qmc: int = 256
    stream: int | None = None

    # Ping-pong merged buffers (capacity max_n):
    # - prefix [:nnz] holds the current sparse vector x (sorted unique),
    # - tail [nnz:nnz+nnz*(nspawn_one+nspawn_two)] is used for spawned events in-place.
    x_idx_a: Any | None = None
    x_val_a: Any | None = None
    x_idx_b: Any | None = None
    x_val_b: Any | None = None
    nnz: int = 0
    use_a: bool = True

    # Device scalar written by coalesce kernels.
    nnz_u: Any | None = None

    # Optional semi-stochastic deterministic subspace (same construction as the
    # block-projector path, but applied to the single-root FCIQMC step).
    det_idx_host: np.ndarray | None = None
    det_idx_dev: Any | None = None
    det_cols: list[tuple[np.ndarray, np.ndarray]] | None = None
    det_hdd_csr_dev: Any | None = None
    det_x_buf: Any | None = None
    det_y_buf: Any | None = None
    det_spawn_slot_offsets: Any | None = None
    det_idx_buf: Any | None = None
    det_val_buf: Any | None = None

    def release(self) -> None:
        if self.ws is not None:
            self.ws.release()
        if self.state_dev is not None:
            self.state_dev.release()
        if self.drt_dev is not None:
            self.drt_dev.release()
        self.ws = None
        self.state_dev = None
        self.drt_dev = None

    @property
    def x_idx(self):
        return self.x_idx_a if self.use_a else self.x_idx_b

    @property
    def x_val(self):
        return self.x_val_a if self.use_a else self.x_val_b

    @property
    def x_idx_next(self):
        return self.x_idx_b if self.use_a else self.x_idx_a

    @property
    def x_val_next(self):
        return self.x_val_b if self.use_a else self.x_val_a


@dataclass
class CudaFCIQMCContextKey64:
    """Reusable GPU context for repeated Key64 FCIQMC steps (spawn + coalesce, no compression)."""

    drt_dev: Any
    ws: Any

    h_base_flat_dev: Any
    eri_mat_dev: Any

    nspawn_one: int
    nspawn_two: int
    max_walker: int
    max_n: int
    max_evt: int

    threads_spawn: int = 128
    threads_qmc: int = 256
    stream: int | None = None

    # Optional alias/pair-norm sampling inputs for the Key64 spawn kernel.
    pair_alias_prob_dev: Any | None = None
    pair_alias_idx_dev: Any | None = None
    pair_norm_dev: Any | None = None
    pair_norm_sum: float = 0.0
    pair_sampling_mode: int = 0
    label_mode: str = "key64"
    ncsf_u64: int = 0

    # Ping-pong merged buffers (capacity max_n):
    # - prefix [:nnz] holds the current sparse vector x (sorted unique by key),
    # - tail [nnz:nnz+nnz*(nspawn_one+nspawn_two)] is used for spawned events in-place.
    x_key_a: Any | None = None
    x_val_a: Any | None = None
    x_key_b: Any | None = None
    x_val_b: Any | None = None
    nnz: int = 0
    use_a: bool = True

    # Device scalar written by coalesce kernels.
    nnz_u: Any | None = None

    # Optional semi-stochastic deterministic subspace in CSF order with a Key64
    # mirror for on-device support tests and event insertion.
    det_idx_host: np.ndarray | None = None
    det_key_dev: Any | None = None
    det_cols: list[tuple[np.ndarray, np.ndarray]] | None = None
    det_hdd_csr_dev: Any | None = None
    det_x_buf: Any | None = None
    det_y_buf: Any | None = None
    det_spawn_slot_offsets: Any | None = None
    det_key_buf: Any | None = None
    det_val_buf: Any | None = None

    def release(self) -> None:
        if self.ws is not None:
            self.ws.release()
        if self.drt_dev is not None:
            self.drt_dev.release()
        self.ws = None
        self.drt_dev = None

    @property
    def x_key(self):
        return self.x_key_a if self.use_a else self.x_key_b

    @property
    def x_val(self):
        return self.x_val_a if self.use_a else self.x_val_b

    @property
    def x_key_next(self):
        return self.x_key_b if self.use_a else self.x_key_a

    @property
    def x_val_next(self):
        return self.x_val_b if self.use_a else self.x_val_a


def make_cuda_fciqmc_context(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    max_walker: int,
    nspawn_one: int,
    nspawn_two: int,
    det_idx: np.ndarray | None = None,
    det_max_out: int = 200_000,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
) -> CudaFCIQMCContext:
    """Build a reusable CUDA FCIQMC context for dense-ERI Hamiltonians."""

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")

    from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415
    from asuka.cuguga.oracle import _child_prefix_walks  # noqa: PLC0415

    norb = int(drt.norb)
    nops = norb * norb

    h1e = np.asarray(h1e, dtype=np.float64).reshape(norb, norb)
    eri = np.asarray(eri, dtype=np.float64)
    if eri.ndim == 4:
        if eri.shape != (norb, norb, norb, norb):
            raise ValueError("eri4 has wrong shape")
        eri_mat = eri.reshape(nops, nops)
        eri4 = eri
    elif eri.ndim == 2:
        if eri.shape != (nops, nops):
            raise ValueError("eri_mat has wrong shape")
        eri_mat = eri
        eri4 = eri.reshape(norb, norb, norb, norb)
    else:
        raise ValueError("eri must be eri_mat[pq,rs] (2D) or eri4[p,q,r,s] (4D)")

    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)

    child_prefix = _child_prefix_walks(drt)
    drt_dev = _guga_cuda_ext.make_device_drt(
        int(drt.norb), np.asarray(drt.child), np.asarray(drt.node_twos), np.asarray(child_prefix)
    )
    cache = get_state_cache(drt)
    state_dev = _guga_cuda_ext.make_device_state_cache(
        drt_dev,
        np.asarray(cache.steps, dtype=np.int8, order="C"),
        np.asarray(cache.nodes, dtype=np.int32, order="C"),
    )

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    max_walker = int(max_walker)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if max_walker < 1:
        raise ValueError("max_walker must be >= 1")
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    nspawn_total = nspawn_one + nspawn_two
    max_evt = max_walker * nspawn_total
    det_n = 0 if det_idx is None else int(np.asarray(det_idx).size)
    max_n = max_walker + max_evt + det_n

    ws = _guga_cuda_ext.QmcWorkspace(int(max_n), int(max_walker))

    ctx = CudaFCIQMCContext(
        drt_dev=drt_dev,
        state_dev=state_dev,
        ws=ws,
        h_base_flat_dev=cp.asarray(h_base.ravel(order="C"), dtype=cp.float64),
        eri_mat_dev=cp.asarray(eri_mat, dtype=cp.float64),
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
        max_walker=max_walker,
        max_n=max_n,
        max_evt=max_evt,
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=int(stream),
    )

    ctx.x_idx_a = cp.empty(max_n, dtype=cp.int32)
    ctx.x_val_a = cp.empty(max_n, dtype=cp.float64)
    ctx.x_idx_b = cp.empty(max_n, dtype=cp.int32)
    ctx.x_val_b = cp.empty(max_n, dtype=cp.float64)
    ctx.nnz_u = cp.empty(1, dtype=cp.int32)

    if det_idx is not None:
        det_idx_i32 = np.asarray(det_idx, dtype=np.int32).ravel()
        if det_idx_i32.size:
            det_idx_i32 = np.unique(det_idx_i32)
            det_idx_i32.sort()
            ctx.det_idx_host = det_idx_i32
            ctx.det_idx_dev = cp.asarray(det_idx_i32, dtype=cp.int32)
            ctx.det_idx_buf = cp.empty(int(det_idx_i32.size), dtype=cp.int32)
            ctx.det_val_buf = cp.empty(int(det_idx_i32.size), dtype=cp.float64)
            ctx.det_cols = _build_det_subspace_cols_dense(
                drt=drt,
                h1e=np.asarray(h1e, dtype=np.float64),
                eri=eri4,
                det_idx=det_idx_i32,
                max_out=int(det_max_out),
                state_cache=cache,
            )
            try:
                import cupyx.scipy.sparse as cpx_sparse  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("deterministic-subspace CUDA correction requires cupyx.scipy.sparse") from e
            row_h, col_h, dat_h = _det_subspace_coo_from_cols(ctx.det_cols)
            ndet = int(det_idx_i32.size)
            if int(dat_h.size) == 0:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix((ndet, ndet), dtype=cp.float64)
            else:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix(
                    (
                        cp.asarray(dat_h, dtype=cp.float64),
                        (
                            cp.asarray(row_h, dtype=cp.int32),
                            cp.asarray(col_h, dtype=cp.int32),
                        ),
                    ),
                    shape=(ndet, ndet),
                    dtype=cp.float64,
                )
            ctx.det_x_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_y_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_spawn_slot_offsets = cp.arange(int(nspawn_total), dtype=cp.int64)
    return ctx


def make_cuda_fciqmc_context_key64(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    max_walker: int,
    nspawn_one: int,
    nspawn_two: int,
    det_idx: np.ndarray | None = None,
    det_max_out: int = 200_000,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
    pair_alias_prob: Any | None = None,
    pair_alias_idx: Any | None = None,
    pair_norm: Any | None = None,
    pair_norm_sum: float = 0.0,
    pair_sampling_mode: int = 0,
    label_mode: str = "key64",
    ncsf_u64: int | None = None,
) -> CudaFCIQMCContextKey64:
    """Build a reusable CUDA FCIQMC context in uint64 walker space.

    Notes
    -----
    ``label_mode='key64'`` requires ``drt.norb <= 32`` so each CSF path can be
    packed into a 64-bit key (2 bits per orbital step).
    ``label_mode='idx64'`` accepts ``drt.norb <= 64`` and stores global CSF
    indices in uint64 buffers/workspace.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not hasattr(_guga_cuda_ext, "QmcWorkspaceU64"):
        raise RuntimeError("Key64 QMC workspace is unavailable (missing _guga_cuda_ext.QmcWorkspaceU64); rebuild the CUDA extension")

    from asuka.cuguga.oracle import _child_prefix_walks  # noqa: PLC0415
    from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415

    label_mode_s = str(label_mode).strip().lower()
    if label_mode_s not in ("key64", "idx64"):
        raise ValueError("label_mode must be 'key64' or 'idx64'")

    norb = int(drt.norb)
    if label_mode_s == "key64":
        if norb > 32:
            raise ValueError("Key64 FCIQMC context requires drt.norb <= 32")
    else:
        if norb > 64:
            raise ValueError("idx64 FCIQMC context requires drt.norb <= 64")
    ncsf_u64_eff = int(int(drt.ncsf) if ncsf_u64 is None else int(ncsf_u64))
    if ncsf_u64_eff <= 0:
        raise ValueError("ncsf_u64 must be > 0")
    nops = norb * norb

    h1e = np.asarray(h1e, dtype=np.float64).reshape(norb, norb)
    eri = np.asarray(eri, dtype=np.float64)
    if eri.ndim == 4:
        if eri.shape != (norb, norb, norb, norb):
            raise ValueError("eri4 has wrong shape")
        eri_mat = eri.reshape(nops, nops)
        eri4 = eri
    elif eri.ndim == 2:
        if eri.shape != (nops, nops):
            raise ValueError("eri_mat has wrong shape")
        eri_mat = eri
        eri4 = eri.reshape(norb, norb, norb, norb)
    else:
        raise ValueError("eri must be eri_mat[pq,rs] (2D) or eri4[p,q,r,s] (4D)")

    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)

    child_prefix = _child_prefix_walks(drt)
    drt_dev = _guga_cuda_ext.make_device_drt(
        int(drt.norb),
        np.asarray(drt.child),
        np.asarray(drt.node_twos),
        np.asarray(child_prefix),
    )

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    max_walker = int(max_walker)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if max_walker < 1:
        raise ValueError("max_walker must be >= 1")
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    pair_sampling_mode = int(pair_sampling_mode)
    pair_norm_sum = float(pair_norm_sum)
    pair_alias_prob_dev = None
    pair_alias_idx_dev = None
    pair_norm_dev = None
    if pair_sampling_mode != 0:
        if pair_sampling_mode != 1:
            raise ValueError("pair_sampling_mode must be 0 (uniform) or 1 (pair_norm alias)")
        if pair_alias_prob is None or pair_alias_idx is None or pair_norm is None:
            raise ValueError("pair_alias_prob/pair_alias_idx/pair_norm must be provided when pair_sampling_mode!=0")
        if not np.isfinite(pair_norm_sum) or pair_norm_sum <= 0.0:
            raise ValueError("pair_norm_sum must be finite and > 0 when pair_sampling_mode!=0")

        pair_alias_prob_dev = cp.asarray(pair_alias_prob, dtype=cp.float32).ravel()
        pair_alias_idx_dev = cp.asarray(pair_alias_idx, dtype=cp.int32).ravel()
        pair_norm_dev = cp.asarray(pair_norm, dtype=cp.float64).ravel()
        if int(pair_alias_prob_dev.size) != nops or int(pair_alias_idx_dev.size) != nops or int(pair_norm_dev.size) != nops:
            raise ValueError(f"pair alias/norm arrays must have length nops={nops}")

    nspawn_total = nspawn_one + nspawn_two
    max_evt = max_walker * nspawn_total
    det_n = 0 if det_idx is None else int(np.asarray(det_idx).size)
    max_n = max_walker + max_evt + det_n

    # FCIQMC does not use Φ compression; keep max_m small to avoid allocating large Φ scratch.
    ws = _guga_cuda_ext.QmcWorkspaceU64(int(max_n), 1)

    ctx = CudaFCIQMCContextKey64(
        drt_dev=drt_dev,
        ws=ws,
        h_base_flat_dev=cp.asarray(h_base.ravel(order="C"), dtype=cp.float64),
        eri_mat_dev=cp.asarray(eri_mat, dtype=cp.float64),
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
        max_walker=max_walker,
        max_n=max_n,
        max_evt=max_evt,
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=int(stream),
        pair_alias_prob_dev=pair_alias_prob_dev,
        pair_alias_idx_dev=pair_alias_idx_dev,
        pair_norm_dev=pair_norm_dev,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
        label_mode=str(label_mode_s),
        ncsf_u64=int(ncsf_u64_eff),
    )

    ctx.x_key_a = cp.empty(max_n, dtype=cp.uint64)
    ctx.x_val_a = cp.empty(max_n, dtype=cp.float64)
    ctx.x_key_b = cp.empty(max_n, dtype=cp.uint64)
    ctx.x_val_b = cp.empty(max_n, dtype=cp.float64)
    ctx.nnz_u = cp.empty(1, dtype=cp.int32)

    if det_idx is not None:
        det_idx_i64 = np.asarray(det_idx, dtype=np.int64).ravel()
        if det_idx_i64.size:
            det_idx_i64 = np.unique(det_idx_i64)
            det_idx_i64.sort()
            if det_idx_i64[0] < 0:
                raise ValueError("det_idx entries must be non-negative")
            if det_idx_i64[-1] >= int(ncsf_u64_eff):
                raise ValueError("det_idx entries must be < ncsf_u64")
            cache = None if int(drt.ncsf) > np.iinfo(np.int32).max else get_state_cache(drt)
            ctx.det_idx_host = det_idx_i64
            if label_mode_s == "key64":
                det_label_u64 = np.asarray(csf_idx_to_key64_host(drt, det_idx_i64, state_cache=cache), dtype=np.uint64, order="C")
            else:
                det_label_u64 = np.asarray(det_idx_i64, dtype=np.uint64, order="C")
            ctx.det_key_dev = cp.asarray(det_label_u64, dtype=cp.uint64)
            ctx.det_key_buf = cp.empty(int(det_idx_i64.size), dtype=cp.uint64)
            ctx.det_val_buf = cp.empty(int(det_idx_i64.size), dtype=cp.float64)
            ctx.det_cols = _build_det_subspace_cols_dense(
                drt=drt,
                h1e=np.asarray(h1e, dtype=np.float64),
                eri=eri4,
                det_idx=det_idx_i64,
                max_out=int(det_max_out),
                state_cache=cache,
            )
            try:
                import cupyx.scipy.sparse as cpx_sparse  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("deterministic-subspace CUDA correction requires cupyx.scipy.sparse") from e
            row_h, col_h, dat_h = _det_subspace_coo_from_cols(ctx.det_cols)
            ndet = int(det_idx_i64.size)
            if int(dat_h.size) == 0:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix((ndet, ndet), dtype=cp.float64)
            else:
                ctx.det_hdd_csr_dev = cpx_sparse.csr_matrix(
                    (
                        cp.asarray(dat_h, dtype=cp.float64),
                        (
                            cp.asarray(row_h, dtype=cp.int32),
                            cp.asarray(col_h, dtype=cp.int32),
                        ),
                    ),
                    shape=(ndet, ndet),
                    dtype=cp.float64,
                )
            ctx.det_x_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_y_buf = cp.zeros(ndet, dtype=cp.float64)
            ctx.det_spawn_slot_offsets = cp.arange(int(nspawn_total), dtype=cp.int64)
    return ctx


def make_cuda_fciqmc_context_idx64(
    drt: Any,
    h1e: Any,
    eri: Any,
    *,
    max_walker: int,
    nspawn_one: int,
    nspawn_two: int,
    det_idx: np.ndarray | None = None,
    det_max_out: int = 200_000,
    threads_spawn: int = 128,
    threads_qmc: int = 256,
    stream: int | None = None,
    pair_alias_prob: Any | None = None,
    pair_alias_idx: Any | None = None,
    pair_norm: Any | None = None,
    pair_norm_sum: float = 0.0,
    pair_sampling_mode: int = 0,
    ncsf_u64: int | None = None,
) -> CudaFCIQMCContextKey64:
    """Build a reusable CUDA FCIQMC context in idx64 walker space (uint64 CSF indices)."""

    return make_cuda_fciqmc_context_key64(
        drt,
        h1e,
        eri,
        max_walker=int(max_walker),
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        det_idx=det_idx,
        det_max_out=int(det_max_out),
        threads_spawn=int(threads_spawn),
        threads_qmc=int(threads_qmc),
        stream=stream,
        pair_alias_prob=pair_alias_prob,
        pair_alias_idx=pair_alias_idx,
        pair_norm=pair_norm,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
        label_mode="idx64",
        ncsf_u64=int(int(drt.ncsf) if ncsf_u64 is None else int(ncsf_u64)),
    )


def cuda_fciqmc_step_hamiltonian_ws(
    ctx: CudaFCIQMCContext,
    *,
    dt: float,
    shift: float,
    initiator_t: float = 0.0,
    seed_spawn: int,
    sync: bool = True,
    use_fused: bool = True,
) -> int:
    """One FCIQMC step on GPU: spawn + shift-scale + coalesce (no FRI compression).

    Applies ``x <- (1 + dt*S)*x - dt*H*x`` entirely on device.

    Returns the new ``nnz`` after coalescing.

    Notes
    -----
    Requires ``sync=True`` because the host reads ``nnz_u`` to update ``ctx.nnz``.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not bool(sync):
        raise ValueError("cuda_fciqmc_step_hamiltonian_ws currently requires sync=True")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaFCIQMCContext is released")

    nnz = int(ctx.nnz)
    if nnz <= 0:
        raise ValueError("current x is empty")
    if nnz > int(ctx.max_walker):
        raise ValueError(f"current nnz={nnz} exceeds max_walker={ctx.max_walker}")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None or ctx.nnz_u is None:
        raise RuntimeError("CudaFCIQMCContext buffers not initialized")

    use_det = (
        ctx.det_idx_dev is not None
        and ctx.det_hdd_csr_dev is not None
        and ctx.det_x_buf is not None
        and ctx.det_y_buf is not None
        and ctx.det_idx_buf is not None
        and ctx.det_val_buf is not None
    )

    if bool(use_fused) and (not use_det) and hasattr(ctx.ws, "fciqmc_step_shift_i32_f64_inplace_device"):
        n_out = int(
            ctx.ws.fciqmc_step_shift_i32_f64_inplace_device(
                ctx.drt_dev,
                ctx.state_dev,
                ctx.x_idx,
                ctx.x_val,
                int(nnz),
                ctx.h_base_flat_dev,
                ctx.eri_mat_dev,
                ctx.x_idx_next,
                ctx.x_val_next,
                ctx.nnz_u,
                float(dt),
                float(shift),
                float(initiator_t),
                int(ctx.nspawn_one),
                int(ctx.nspawn_two),
                int(seed_spawn),
                int(ctx.threads_spawn),
                int(ctx.threads_qmc),
                int(ctx.stream),
                bool(sync),
            )
        )
        ctx.nnz = n_out
        ctx.use_a = not ctx.use_a
        return n_out

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    out_len = nnz * nspawn_total
    all_len = nnz + out_len

    x_idx = ctx.x_idx
    x_val = ctx.x_val
    x_idx_cur = x_idx[:nnz]
    x_val_cur = x_val[:nnz]

    # Spawn events directly into the tail of the current buffer (merged input).
    evt_idx = x_idx[nnz:all_len]
    evt_val = x_val[nnz:all_len]

    # Spawn: fills evt_idx/evt_val with -dt*H*x events (-1 sentinel for invalid slots).
    _guga_cuda_ext.qmc_spawn_hamiltonian_inplace_device(
        ctx.drt_dev,
        ctx.state_dev,
        x_idx_cur,
        x_val_cur,
        ctx.h_base_flat_dev,
        ctx.eri_mat_dev,
        evt_idx,
        evt_val,
        float(dt),
        int(ctx.nspawn_one),
        int(ctx.nspawn_two),
        int(seed_spawn),
        float(initiator_t),
        int(ctx.threads_spawn),
        int(ctx.stream),
        False,
    )

    nnz_det = 0
    if use_det:
        nnz_det = _apply_det_subspace_correction_gpu(
            ctx=ctx,
            x_idx=x_idx_cur,
            x_val=x_val_cur,
            evt_idx=evt_idx,
            evt_val=evt_val,
            eps=float(dt),
            nspawn_total=int(nspawn_total),
        )
    # Merge: scale the identity term in-place in the prefix; events are already in the tail.
    x_val_cur *= float(1.0 + dt * shift)
    if use_det:
        all_len += int(nnz_det)
        if all_len > int(ctx.max_n):
            raise RuntimeError("deterministic-subspace merge exceeds workspace capacity (increase max_walker)")
        x_idx[nnz + out_len : all_len] = ctx.det_idx_buf[:nnz_det]
        x_val[nnz + out_len : all_len] = ctx.det_val_buf[:nnz_det]

    # Coalesce (sort + reduce; idx < 0 sentinels contribute zero and are excluded).
    ctx.ws.coalesce_coo_i32_f64_inplace_device(
        x_idx,
        x_val,
        ctx.x_idx_next,
        ctx.x_val_next,
        ctx.nnz_u,
        int(all_len),
        int(ctx.threads_qmc),
        int(ctx.stream),
        bool(sync),
    )
    n_out = int(cp.asnumpy(ctx.nnz_u)[0])
    if n_out < 0:
        n_out = 0
    if n_out > int(ctx.max_walker):
        raise RuntimeError(
            f"walker population after coalescing ({n_out}) exceeds max_walker ({ctx.max_walker}). "
            "Increase max_walker or tighten population control."
        )

    ctx.nnz = n_out
    ctx.use_a = not ctx.use_a
    return n_out


def cuda_fciqmc_step_hamiltonian_u64_ws(
    ctx: CudaFCIQMCContextKey64,
    *,
    dt: float,
    shift: float,
    initiator_t: float = 0.0,
    seed_spawn: int,
    sync: bool = True,
) -> int:
    """One Key64 FCIQMC step on GPU: spawn + shift-scale + coalesce (no compression).

    Applies ``x <- (1 + dt*S)*x - dt*H*x`` entirely on device.

    Notes
    -----
    Requires ``sync=True`` because the host reads ``nnz_u`` to update ``ctx.nnz``.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and cuGUGA CUDA extension)")
    if not bool(sync):
        raise ValueError("cuda_fciqmc_step_hamiltonian_u64_ws currently requires sync=True")
    if ctx.ws is None or ctx.drt_dev is None:
        raise RuntimeError("CudaFCIQMCContextKey64 is released")

    nnz = int(ctx.nnz)
    if nnz <= 0:
        raise ValueError("current x is empty")
    if nnz > int(ctx.max_walker):
        raise ValueError(f"current nnz={nnz} exceeds max_walker={ctx.max_walker}")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    if ctx.x_key is None or ctx.x_val is None or ctx.x_key_next is None or ctx.x_val_next is None or ctx.nnz_u is None:
        raise RuntimeError("CudaFCIQMCContextKey64 buffers not initialized")

    use_det = (
        ctx.det_key_dev is not None
        and ctx.det_hdd_csr_dev is not None
        and ctx.det_x_buf is not None
        and ctx.det_y_buf is not None
        and ctx.det_key_buf is not None
        and ctx.det_val_buf is not None
    )

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    out_len = nnz * nspawn_total
    all_len = nnz + out_len

    x_key = ctx.x_key
    x_val = ctx.x_val
    x_key_cur = x_key[:nnz]
    x_val_cur = x_val[:nnz]

    # Spawn events directly into the tail of the current buffer (merged input).
    evt_key = x_key[nnz:all_len]
    evt_val = x_val[nnz:all_len]

    label_mode = str(getattr(ctx, "label_mode", "key64")).strip().lower()
    if label_mode not in ("key64", "idx64"):
        raise ValueError("ctx.label_mode must be 'key64' or 'idx64'")

    # Spawn: fills evt_key/evt_val with -dt*H*x events (UINT64_MAX sentinel for invalid slots).
    if label_mode == "idx64":
        _guga_cuda_ext.qmc_spawn_hamiltonian_idx64_u64_inplace_device(
            ctx.drt_dev,
            int(ctx.ncsf_u64),
            x_key_cur,
            x_val_cur,
            ctx.h_base_flat_dev,
            ctx.eri_mat_dev,
            evt_key,
            evt_val,
            float(dt),
            int(ctx.nspawn_one),
            int(ctx.nspawn_two),
            int(seed_spawn),
            float(initiator_t),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
            ctx.pair_alias_prob_dev,
            ctx.pair_alias_idx_dev,
            ctx.pair_norm_dev,
            float(ctx.pair_norm_sum),
            int(ctx.pair_sampling_mode),
        )
    else:
        _guga_cuda_ext.qmc_spawn_hamiltonian_u64_inplace_device(
            ctx.drt_dev,
            x_key_cur,
            x_val_cur,
            ctx.h_base_flat_dev,
            ctx.eri_mat_dev,
            evt_key,
            evt_val,
            float(dt),
            int(ctx.nspawn_one),
            int(ctx.nspawn_two),
            int(seed_spawn),
            float(initiator_t),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
            ctx.pair_alias_prob_dev,
            ctx.pair_alias_idx_dev,
            ctx.pair_norm_dev,
            float(ctx.pair_norm_sum),
            int(ctx.pair_sampling_mode),
        )

    nnz_det = 0
    if use_det:
        nnz_det = _apply_det_subspace_correction_gpu_u64(
            ctx=ctx,
            x_key=x_key_cur,
            x_val=x_val_cur,
            evt_key=evt_key,
            evt_val=evt_val,
            eps=float(dt),
            nspawn_total=int(nspawn_total),
        )
    # Merge: scale the identity term in-place in the prefix; events are already in the tail.
    x_val_cur *= float(1.0 + dt * shift)
    if use_det:
        all_len += int(nnz_det)
        if all_len > int(ctx.max_n):
            raise RuntimeError("deterministic-subspace merge exceeds workspace capacity (increase max_walker)")
        x_key[nnz + out_len : all_len] = ctx.det_key_buf[:nnz_det]
        x_val[nnz + out_len : all_len] = ctx.det_val_buf[:nnz_det]

    # Coalesce (sort + reduce; UINT64_MAX sentinel contributes zero and is excluded via zero-pruning).
    ctx.ws.coalesce_coo_u64_f64_inplace_device(
        x_key,
        x_val,
        ctx.x_key_next,
        ctx.x_val_next,
        ctx.nnz_u,
        int(all_len),
        int(ctx.threads_qmc),
        int(ctx.stream),
        bool(sync),
    )
    n_out = int(cp.asnumpy(ctx.nnz_u)[0])
    if n_out < 0:
        n_out = 0
    if n_out > int(ctx.max_walker):
        raise RuntimeError(
            f"walker population after coalescing ({n_out}) exceeds max_walker ({ctx.max_walker}). "
            "Increase max_walker or tighten population control."
        )

    ctx.nnz = n_out
    ctx.use_a = not ctx.use_a
    return n_out


def cuda_fciqmc_step_hamiltonian_idx64_ws(
    ctx: CudaFCIQMCContextKey64,
    *,
    dt: float,
    shift: float,
    initiator_t: float = 0.0,
    seed_spawn: int,
    sync: bool = True,
) -> int:
    """One idx64 FCIQMC step on GPU (uint64 global CSF indices)."""

    if str(getattr(ctx, "label_mode", "key64")).strip().lower() != "idx64":
        raise ValueError("cuda_fciqmc_step_hamiltonian_idx64_ws requires ctx.label_mode='idx64'")
    return cuda_fciqmc_step_hamiltonian_u64_ws(
        ctx,
        dt=float(dt),
        shift=float(shift),
        initiator_t=float(initiator_t),
        seed_spawn=int(seed_spawn),
        sync=bool(sync),
    )
