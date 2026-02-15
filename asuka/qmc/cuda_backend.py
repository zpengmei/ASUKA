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
    from asuka import _guga_cuda_ext  # type: ignore
except Exception:  # pragma: no cover
    _guga_cuda_ext = None  # type: ignore


def have_cuda_qmc() -> bool:
    return (cp is not None) and (_guga_cuda_ext is not None)


def qmc_spawn_one_body_events_device(
    drt_dev: Any,
    state_dev: Any,
    x_idx_dev: Any,
    x_val_dev: Any,
    h_eff_flat_dev: Any,
    *,
    eps: float,
    nspawn: int,
    seed: int,
    initiator_t: float = 0.0,
    threads: int = 128,
    stream: int | None = None,
    sync: bool = True,
):
    """Spawn one-body events on GPU into a fixed-size COO buffer.

    Returns
    -------
    out_idx_dev, out_val_dev
        CuPy arrays of shape ``(m*nspawn,)`` with ``out_idx_dev == -1`` marking
        unused slots.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

    x_idx_dev = cp.asarray(x_idx_dev, dtype=cp.int32)
    x_val_dev = cp.asarray(x_val_dev, dtype=cp.float64)
    h_eff_flat_dev = cp.asarray(h_eff_flat_dev, dtype=cp.float64).ravel()

    m = int(x_idx_dev.size)
    nspawn = int(nspawn)
    if nspawn <= 0:
        raise ValueError("nspawn must be >= 1")

    out_idx_dev = cp.empty(m * nspawn, dtype=cp.int32)
    out_val_dev = cp.empty(m * nspawn, dtype=cp.float64)

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    _guga_cuda_ext.qmc_spawn_one_body_inplace_device(
        drt_dev,
        state_dev,
        x_idx_dev,
        x_val_dev,
        h_eff_flat_dev,
        out_idx_dev,
        out_val_dev,
        float(eps),
        int(nspawn),
        int(seed),
        float(initiator_t),
        int(threads),
        int(stream),
        bool(sync),
    )
    return out_idx_dev, out_val_dev


def qmc_spawn_hamiltonian_events_device(
    drt_dev: Any,
    state_dev: Any,
    x_idx_dev: Any,
    x_val_dev: Any,
    h_base_flat_dev: Any,
    eri_mat_dev: Any,
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
    """Spawn full-H events on GPU into a fixed-size COO buffer.

    This matches the CPU decomposition used by `asuka.qmc.spawn.spawn_hamiltonian_events`:
    - state-dependent effective one-body term (includes r==s slice + contraction),
    - two-body product term sampled with r!=s via sequential E_rs then E_pq.

    Returns
    -------
    out_idx_dev, out_val_dev
        CuPy arrays of shape ``(m*(nspawn_one+nspawn_two),)`` with `idx==-1` as sentinel.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

    x_idx_dev = cp.asarray(x_idx_dev, dtype=cp.int32)
    x_val_dev = cp.asarray(x_val_dev, dtype=cp.float64)
    h_base_flat_dev = cp.asarray(h_base_flat_dev, dtype=cp.float64).ravel()
    eri_mat_dev = cp.asarray(eri_mat_dev, dtype=cp.float64)

    m = int(x_idx_dev.size)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one/nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one/nspawn_two must be > 0")

    out_len = m * (nspawn_one + nspawn_two)
    out_idx_dev = cp.empty(out_len, dtype=cp.int32)
    out_val_dev = cp.empty(out_len, dtype=cp.float64)

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    _guga_cuda_ext.qmc_spawn_hamiltonian_inplace_device(
        drt_dev,
        state_dev,
        x_idx_dev,
        x_val_dev,
        h_base_flat_dev,
        eri_mat_dev,
        out_idx_dev,
        out_val_dev,
        float(eps),
        int(nspawn_one),
        int(nspawn_two),
        int(seed),
        float(initiator_t),
        int(threads),
        int(stream),
        bool(sync),
    )
    return out_idx_dev, out_val_dev


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
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

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
            "Key64 spawn kernel is unavailable in this build of asuka._guga_cuda_ext "
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
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

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
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

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


def qmc_projector_step_hamiltonian_device(
    drt_dev: Any,
    state_dev: Any,
    x_idx_dev: Any,
    x_val_dev: Any,
    h_base_flat_dev: Any,
    eri_mat_dev: Any,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    scale_identity: float = 1.0,
    initiator_t: float = 0.0,
    threads_spawn: int = 128,
    threads_coalesce: int = 256,
    stream: int | None = None,
    sync: bool = True,
):
    """Compute one stochastic projector step on GPU and coalesce the COO output.

    This returns an estimator for:
        y ≈ scale_identity * x + (-eps * H * x)

    This function does not apply Φ compression; it only spawns + annihilates.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

    x_idx_dev = cp.asarray(x_idx_dev, dtype=cp.int32).ravel()
    x_val_dev = cp.asarray(x_val_dev, dtype=cp.float64).ravel()
    if x_idx_dev.size != x_val_dev.size:
        raise ValueError("x_idx_dev and x_val_dev must have the same size")

    if stream is None:
        stream = int(cp.cuda.get_current_stream().ptr)

    evt_idx_dev, evt_val_dev = qmc_spawn_hamiltonian_events_device(
        drt_dev,
        state_dev,
        x_idx_dev,
        x_val_dev,
        h_base_flat_dev,
        eri_mat_dev,
        eps=float(eps),
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        seed=int(seed),
        initiator_t=float(initiator_t),
        threads=int(threads_spawn),
        stream=int(stream),
        sync=False,
    )

    m = int(x_idx_dev.size)
    n_evt = int(evt_idx_dev.size)
    idx_all = cp.empty(m + n_evt, dtype=cp.int32)
    val_all = cp.empty(m + n_evt, dtype=cp.float64)
    idx_all[:m] = x_idx_dev
    idx_all[m:] = evt_idx_dev
    val_all[:m] = float(scale_identity) * x_val_dev
    val_all[m:] = evt_val_dev

    return qmc_coalesce_coo_i32_f64_device(idx_all, val_all, threads=int(threads_coalesce), stream=int(stream), sync=bool(sync))


@dataclass
class CudaProjectorContext:
    """Reusable GPU context for repeated `(I - eps*H)` projector steps.

    This owns device-resident DRT/state tables, Hamiltonian inputs, a persistent
    `QmcWorkspace`, and preallocated buffers sized for the worst-case step.
    """

    drt_dev: Any
    state_dev: Any
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

    # Ping-pong sparse-vector buffers (capacity m; only prefix [:nnz] is valid).
    x_idx_a: Any | None = None
    x_val_a: Any | None = None
    x_idx_b: Any | None = None
    x_val_b: Any | None = None
    nnz: int = 0
    use_a: bool = True

    # Preallocated step buffers.
    evt_idx: Any | None = None
    evt_val: Any | None = None
    idx_all: Any | None = None
    val_all: Any | None = None
    idx_u: Any | None = None
    val_u: Any | None = None
    nnz_u: Any | None = None
    nnz_out: Any | None = None

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

    # Optional semi-stochastic deterministic subspace (host-built).
    det_idx_host: np.ndarray | None = None  # sorted global CSF indices (int32)
    det_idx_dev: Any | None = None  # device mirror of det_idx_host
    det_cols: list[tuple[np.ndarray, np.ndarray]] | None = None  # per det-col: (i_pos:int32[], hij:float64[])
    det_idx_buf: Any | None = None  # device scratch for appending det events (global idx)
    det_val_buf: Any | None = None  # device scratch for appending det values

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

    det_idx = np.asarray(det_idx, dtype=np.int32).ravel()
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
        i_idx = np.asarray(i_idx, dtype=np.int32, order="C").ravel()
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


def make_cuda_projector_context(
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
):
    """Build a reusable CUDA projector context for dense-ERI Hamiltonians."""

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

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
    max_n = m * (1 + nspawn_total)

    # Persistent workspace for coalesce + Φ.
    ws = _guga_cuda_ext.QmcWorkspace(int(max_n), int(m))

    ctx = CudaProjectorContext(
        drt_dev=drt_dev,
        state_dev=state_dev,
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
    )

    # Preallocate buffers.
    ctx.x_idx_a = cp.empty(m, dtype=cp.int32)
    ctx.x_val_a = cp.empty(m, dtype=cp.float64)
    ctx.x_idx_b = cp.empty(m, dtype=cp.int32)
    ctx.x_val_b = cp.empty(m, dtype=cp.float64)
    ctx.evt_idx = cp.empty(max_evt, dtype=cp.int32)
    ctx.evt_val = cp.empty(max_evt, dtype=cp.float64)
    ctx.idx_all = cp.empty(max_n, dtype=cp.int32)
    ctx.val_all = cp.empty(max_n, dtype=cp.float64)
    ctx.idx_u = cp.empty(max_n, dtype=cp.int32)
    ctx.val_u = cp.empty(max_n, dtype=cp.float64)
    ctx.nnz_u = cp.empty(1, dtype=cp.int32)
    ctx.nnz_out = cp.empty(1, dtype=cp.int32)
    return ctx


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
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")

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

    # Optional semi-stochastic deterministic subspace (CPU-built, used on GPU step).
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
    return ctx


def cuda_projector_step_hamiltonian_ws(
    ctx: CudaProjectorContext,
    *,
    eps: float,
    initiator_t: float,
    seed_spawn: int,
    seed_phi: int,
    scale_identity: float = 1.0,
    sync: bool = True,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> int:
    """One projector step on GPU: spawn + identity merge + coalesce + Φ (workspace path).

    If `compressor` is provided, `Φ(...)` is evaluated on host (reference-oriented):
    - coalesce is still performed on GPU,
    - the coalesced COO vector is downloaded to host and passed to `compressor`,
    - the compressed vector is uploaded back to the output buffers on GPU.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaProjectorContext buffers not initialized")
    if ctx.evt_idx is None or ctx.evt_val is None or ctx.idx_all is None or ctx.val_all is None:
        raise RuntimeError("CudaProjectorContext buffers not initialized")
    if ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaProjectorContext buffers not initialized")

    nnz = int(ctx.nnz)
    if nnz <= 0:
        raise ValueError("current x is empty")
    if nnz > int(ctx.m):
        raise ValueError("current nnz exceeds context m")

    if ctx.stream is None:
        ctx.stream = int(cp.cuda.get_current_stream().ptr)

    nspawn_total = int(ctx.nspawn_one + ctx.nspawn_two)
    out_len = nnz * nspawn_total

    # Spawn events into a length-(nnz*nspawn_total) view of the preallocated buffers.
    evt_idx = ctx.evt_idx[:out_len]
    evt_val = ctx.evt_val[:out_len]
    x_idx = ctx.x_idx[:nnz]
    x_val = ctx.x_val[:nnz]

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
        int(seed_spawn),
        float(initiator_t),
        int(ctx.threads_spawn),
        int(ctx.stream),
        False,
    )

    # Merge identity and events.
    all_len = nnz + out_len
    idx_all = ctx.idx_all
    val_all = ctx.val_all
    idx_all[:nnz] = x_idx
    idx_all[nnz:all_len] = evt_idx
    val_all[:nnz] = float(scale_identity) * x_val
    val_all[nnz:all_len] = evt_val

    # Coalesce into (idx_u, val_u) with out_nnz in nnz_u.
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

    if compressor is not None:
        if not bool(sync):
            raise ValueError("host compressor requires sync=True")

        from .sparse import coalesce_coo_i32_f64  # noqa: PLC0415

        idx_u_host = cp.asnumpy(ctx.idx_u[:n_in]).astype(np.int32, copy=False)
        val_u_host = cp.asnumpy(ctx.val_u[:n_in]).astype(np.float64, copy=False)

        rng_phi = np.random.default_rng(int(seed_phi))
        idx_out, val_out = compressor(idx_u_host, val_u_host, m=int(ctx.m), pivot=int(ctx.pivot), rng=rng_phi)
        idx_out, val_out = coalesce_coo_i32_f64(idx_out, val_out)

        nnz_out = int(idx_out.size)
        if nnz_out < 0:
            nnz_out = 0
        if nnz_out > int(ctx.m):
            raise RuntimeError(f"compressor returned nnz={nnz_out} (>m={int(ctx.m)})")

        if nnz_out > 0:
            ctx.x_idx_next[:nnz_out] = cp.asarray(idx_out, dtype=cp.int32)
            ctx.x_val_next[:nnz_out] = cp.asarray(val_out, dtype=cp.float64)

        ctx.nnz = nnz_out
        ctx.use_a = not ctx.use_a
        return nnz_out

    # Φ compression into the next x buffer.
    ctx.ws.phi_pivot_resample_i32_f64_inplace_device(
        ctx.idx_u,
        ctx.val_u,
        ctx.x_idx_next,
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
    nnz_out = int(cp.asnumpy(ctx.nnz_out)[0]) if sync else int(cp.asnumpy(ctx.nnz_out)[0])
    if nnz_out < 0:
        nnz_out = 0

    ctx.nnz = nnz_out
    ctx.use_a = not ctx.use_a
    return nnz_out


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
    """Projector step for all roots: spawn + identity merge + coalesce + Φ (workspace path)."""

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
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
            int(seed_spawn[k]),
            float(initiator_t[k]),
            int(ctx.threads_spawn),
            int(ctx.stream),
            False,
        )

        # Optional semi-stochastic deterministic subspace correction:
        # - add exact D->D contributions deterministically,
        # - remove spawned events corresponding to D->D from parents in D (avoid double count).
        nnz_det = 0
        det_idx_nz: np.ndarray | None = None
        det_val_nz: np.ndarray | None = None
        if ctx.det_idx_host is not None and ctx.det_cols is not None and ctx.det_idx_dev is not None:
            if not bool(sync):
                raise ValueError("deterministic subspace requires sync=True")

            det_idx_host = np.asarray(ctx.det_idx_host, dtype=np.int32).ravel()
            ndet = int(det_idx_host.size)
            if ndet > 0:
                x_idx_h = cp.asnumpy(x_idx).astype(np.int32, copy=False)
                x_val_h = cp.asnumpy(x_val).astype(np.float64, copy=False)

                pos = np.searchsorted(det_idx_host, x_idx_h)
                inr = pos < ndet
                parent_is_det = np.zeros_like(inr, dtype=np.bool_)
                if np.any(inr):
                    parent_is_det[inr] = det_idx_host[pos[inr]] == x_idx_h[inr]

                # Build x_D in deterministic-subspace ordering.
                x_det = np.zeros(ndet, dtype=np.float64)
                if np.any(parent_is_det):
                    x_det[np.asarray(pos[parent_is_det], dtype=np.int64)] = np.asarray(x_val_h[parent_is_det], dtype=np.float64)

                # Exact deterministic subspace matvec: y_D = -eps * H_DD x_D.
                y_det = -float(eps) * _det_subspace_matvec_cols(det_cols=ctx.det_cols, x_det=x_det)
                nz = np.nonzero(y_det != 0.0)[0]
                if nz.size:
                    det_idx_nz = np.asarray(det_idx_host[nz], dtype=np.int32, order="C")
                    det_val_nz = np.asarray(y_det[nz], dtype=np.float64, order="C")
                    nnz_det = int(det_idx_nz.size)

                # Remove spawned D->D events from *parents in D* (keep outside->D stochastic contributions).
                parent_pos_det = np.nonzero(parent_is_det)[0].astype(np.int64, copy=False)
                if parent_pos_det.size:
                    # Event buffer is laid out as blocks of length nspawn_total per parent position.
                    slot = (parent_pos_det[:, None] * int(nspawn_total) + np.arange(int(nspawn_total), dtype=np.int64)[None, :]).reshape(-1)
                    slot_dev = cp.asarray(slot, dtype=cp.int64)
                    evt_sel = evt_idx[slot_dev]
                    pos2 = cp.searchsorted(ctx.det_idx_dev, evt_sel)
                    pos2 = cp.clip(pos2, 0, ndet - 1)
                    match = (evt_sel >= 0) & (ctx.det_idx_dev[pos2] == evt_sel)
                    if bool(cp.any(match)):
                        evt_val[slot_dev[match]] = 0.0

        all_len = nnz_k + out_len + int(nnz_det)
        idx_all = ctx.idx_all
        val_all = ctx.val_all
        idx_all[:nnz_k] = x_idx
        idx_all[nnz_k : nnz_k + out_len] = evt_idx
        val_all[:nnz_k] = float(si_k) * x_val
        val_all[nnz_k : nnz_k + out_len] = evt_val
        if nnz_det:
            if ctx.det_idx_buf is None or ctx.det_val_buf is None:
                raise RuntimeError("deterministic subspace buffers not initialized")
            assert det_idx_nz is not None and det_val_nz is not None
            ctx.det_idx_buf[:nnz_det] = cp.asarray(det_idx_nz, dtype=cp.int32)
            ctx.det_val_buf[:nnz_det] = cp.asarray(det_val_nz, dtype=cp.float64)
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
) -> None:
    """Modified Gram-Schmidt orthonormalization on GPU with Φ compression.

    This mirrors `asuka.qmc.subspace.orthonormalize_mgs`, operating on the packed
    `(nroots, m)` sparse-column storage inside `CudaBlockProjectorContext`.

    Implementation details (reference-oriented):
    - overlaps are computed with `cupy.searchsorted` against sorted sparse indices,
    - updates use concatenate → coalesce (workspace) → Φ (workspace),
    - output columns are written to the alternate ping-pong buffer, then swapped.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
    if ctx.ws is None or ctx.drt_dev is None or ctx.state_dev is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None or ctx.nnz_next is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.idx_all is None or ctx.val_all is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")

    seeds_phi = np.asarray(seeds_phi, dtype=np.int64).ravel()
    nroots = int(ctx.nroots)
    need = nroots * (nroots - 1) // 2
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

    def dot_sparse_sorted(a_idx, a_val, b_idx, b_val) -> float:
        a_idx = a_idx.ravel()
        a_val = a_val.ravel()
        b_idx = b_idx.ravel()
        b_val = b_val.ravel()
        na = int(a_idx.size)
        nb = int(b_idx.size)
        if na == 0 or nb == 0:
            return 0.0
        if na <= nb:
            pos = cp.searchsorted(b_idx, a_idx)
            inb = pos < nb
            if not bool(cp.any(inb)):
                return 0.0
            pos2 = pos[inb]
            a_idx2 = a_idx[inb]
            a_val2 = a_val[inb]
            match = b_idx[pos2] == a_idx2
            if not bool(cp.any(match)):
                return 0.0
            pos3 = pos2[match]
            return float(cp.sum(a_val2[match] * b_val[pos3]).get())

        # Swap to keep the left side smaller.
        pos = cp.searchsorted(a_idx, b_idx)
        ina = pos < na
        if not bool(cp.any(ina)):
            return 0.0
        pos2 = pos[ina]
        b_idx2 = b_idx[ina]
        b_val2 = b_val[ina]
        match = a_idx[pos2] == b_idx2
        if not bool(cp.any(match)):
            return 0.0
        pos3 = pos2[match]
        return float(cp.sum(b_val2[match] * a_val[pos3]).get())

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
    - This is a reference-oriented implementation intended to replace the CPU row-oracle
      Ritz build in `run_fcifri_subspace(backend="cuda")`.
    - `initiator_t` defaults to 0.0 to avoid adding initiator bias to the Ritz estimator.
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
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

    def dot_sparse_sorted_scalar(a_idx, a_val, b_idx, b_val):
        # Return a 0-d device scalar for the dot product of sorted sparse vectors.
        a_idx = a_idx.ravel()
        a_val = a_val.ravel()
        b_idx = b_idx.ravel()
        b_val = b_val.ravel()
        na = int(a_idx.size)
        nb = int(b_idx.size)
        if na == 0 or nb == 0:
            return cp.asarray(0.0, dtype=cp.float64)

        if na <= nb:
            pos = cp.searchsorted(b_idx, a_idx)
            inb = pos < nb
            pos2 = pos[inb]
            a_idx2 = a_idx[inb]
            a_val2 = a_val[inb]
            match = b_idx[pos2] == a_idx2
            pos3 = pos2[match]
            return cp.sum(a_val2[match] * b_val[pos3])

        # Swap to keep the left side smaller.
        pos = cp.searchsorted(a_idx, b_idx)
        ina = pos < na
        pos2 = pos[ina]
        b_idx2 = b_idx[ina]
        b_val2 = b_val[ina]
        match = a_idx[pos2] == b_idx2
        pos3 = pos2[match]
        return cp.sum(b_val2[match] * a_val[pos3])

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
            bool(sync),
        )
        n_hk = int(cp.asnumpy(ctx.nnz_u)[0])
        if n_hk <= 0:
            # Hx_k estimate is empty; leave column k of T as zeros.
            continue

        hk_idx = ctx.idx_u[:n_hk]
        hk_val = ctx.val_u[:n_hk]

        # events represent (-eps * H x_k); for eps=1: hk = -(H x_k)
        # => <x_i, H x_k> = - <x_i, hk>
        for i in range(nroots):
            nnz_i = int(ctx.nnz[i])
            if nnz_i <= 0:
                raise ValueError(f"column {i} is empty")
            dot = dot_sparse_sorted_scalar(ctx.x_idx[i, :nnz_i], ctx.x_val[i, :nnz_i], hk_idx, hk_val)
            tmat[i, k] = -dot

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
    """

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
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

    # Upload U columns once per call (nroots is small in RSI usage).
    u_idx_dev: list[Any] = []
    u_val_dev: list[Any] = []
    for i, (idx_i, val_i) in enumerate(u_cols):
        idx_i = np.asarray(idx_i, dtype=np.int32).ravel()
        val_i = np.asarray(val_i, dtype=np.float64).ravel()
        if idx_i.size != val_i.size:
            raise ValueError(f"u_cols[{i}] idx/val size mismatch")
        if idx_i.size <= 0:
            raise ValueError(f"u_cols[{i}] is empty")
        u_idx_dev.append(cp.asarray(idx_i, dtype=cp.int32))
        u_val_dev.append(cp.asarray(val_i, dtype=cp.float64))

    def dot_sparse_sorted_scalar(a_idx, a_val, b_idx, b_val):
        # Return a 0-d device scalar for the dot product of sorted sparse vectors.
        a_idx = a_idx.ravel()
        a_val = a_val.ravel()
        b_idx = b_idx.ravel()
        b_val = b_val.ravel()
        na = int(a_idx.size)
        nb = int(b_idx.size)
        if na == 0 or nb == 0:
            return cp.asarray(0.0, dtype=cp.float64)

        if na <= nb:
            pos = cp.searchsorted(b_idx, a_idx)
            inb = pos < nb
            pos2 = pos[inb]
            a_idx2 = a_idx[inb]
            a_val2 = a_val[inb]
            match = b_idx[pos2] == a_idx2
            pos3 = pos2[match]
            return cp.sum(a_val2[match] * b_val[pos3])

        # Swap to keep the left side smaller.
        pos = cp.searchsorted(a_idx, b_idx)
        ina = pos < na
        pos2 = pos[ina]
        b_idx2 = b_idx[ina]
        b_val2 = b_val[ina]
        match = a_idx[pos2] == b_idx2
        pos3 = pos2[match]
        return cp.sum(b_val2[match] * a_val[pos3])

    smat = cp.zeros((nroots, nroots), dtype=cp.float64)
    kmat = cp.zeros((nroots, nroots), dtype=cp.float64)

    for i in range(nroots):
        for j in range(nroots):
            nnz_j = int(ctx.nnz[j])
            if nnz_j <= 0:
                raise ValueError(f"column {j} is empty")
            smat[i, j] = dot_sparse_sorted_scalar(u_idx_dev[i], u_val_dev[i], ctx.x_idx[j, :nnz_j], ctx.x_val[j, :nnz_j])

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

        # Optional semi-stochastic deterministic subspace correction (same logic as the projector step):
        # - add exact D->D contributions deterministically,
        # - remove spawned events corresponding to D->D from parents in D (avoid double count).
        nnz_det = 0
        det_idx_nz: np.ndarray | None = None
        det_val_nz: np.ndarray | None = None
        if ctx.det_idx_host is not None and ctx.det_cols is not None and ctx.det_idx_dev is not None:
            if not bool(sync):
                raise ValueError("deterministic subspace requires sync=True")
            det_idx_host = np.asarray(ctx.det_idx_host, dtype=np.int32).ravel()
            ndet = int(det_idx_host.size)
            if ndet > 0:
                x_idx_h = cp.asnumpy(x_idx).astype(np.int32, copy=False)
                x_val_h = cp.asnumpy(x_val).astype(np.float64, copy=False)

                pos = np.searchsorted(det_idx_host, x_idx_h)
                inr = pos < ndet
                parent_is_det = np.zeros_like(inr, dtype=np.bool_)
                if np.any(inr):
                    parent_is_det[inr] = det_idx_host[pos[inr]] == x_idx_h[inr]

                x_det = np.zeros(ndet, dtype=np.float64)
                if np.any(parent_is_det):
                    x_det[np.asarray(pos[parent_is_det], dtype=np.int64)] = np.asarray(x_val_h[parent_is_det], dtype=np.float64)

                y_det = -float(eps) * _det_subspace_matvec_cols(det_cols=ctx.det_cols, x_det=x_det)
                nz = np.nonzero(y_det != 0.0)[0]
                if nz.size:
                    det_idx_nz = np.asarray(det_idx_host[nz], dtype=np.int32, order="C")
                    det_val_nz = np.asarray(y_det[nz], dtype=np.float64, order="C")
                    nnz_det = int(det_idx_nz.size)

                parent_pos_det = np.nonzero(parent_is_det)[0].astype(np.int64, copy=False)
                if parent_pos_det.size:
                    slot = (parent_pos_det[:, None] * int(nspawn_total) + np.arange(int(nspawn_total), dtype=np.int64)[None, :]).reshape(-1)
                    slot_dev = cp.asarray(slot, dtype=cp.int64)
                    evt_sel = evt_idx[slot_dev]
                    pos2 = cp.searchsorted(ctx.det_idx_dev, evt_sel)
                    pos2 = cp.clip(pos2, 0, ndet - 1)
                    match = (evt_sel >= 0) & (ctx.det_idx_dev[pos2] == evt_sel)
                    if bool(cp.any(match)):
                        evt_val[slot_dev[match]] = 0.0

        # Coalesce events (with optional det merge) into (idx_u,val_u) with out_nnz in nnz_u.
        if nnz_det:
            if ctx.idx_all is None or ctx.val_all is None or ctx.det_idx_buf is None or ctx.det_val_buf is None:
                raise RuntimeError("deterministic subspace buffers not initialized")
            assert det_idx_nz is not None and det_val_nz is not None
            all_len = int(out_len) + int(nnz_det)
            if all_len > int(ctx.max_n):
                raise RuntimeError("eval event merge exceeds workspace capacity (increase max_n)")
            ctx.idx_all[:out_len] = evt_idx
            ctx.val_all[:out_len] = evt_val
            ctx.det_idx_buf[:nnz_det] = cp.asarray(det_idx_nz, dtype=cp.int32)
            ctx.det_val_buf[:nnz_det] = cp.asarray(det_val_nz, dtype=cp.float64)
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
            bool(sync),
        )
        n_hk = int(cp.asnumpy(ctx.nnz_u)[0])
        if n_hk <= 0:
            continue

        hk_idx = ctx.idx_u[:n_hk]
        hk_val = ctx.val_u[:n_hk]

        for i in range(nroots):
            dot = dot_sparse_sorted_scalar(u_idx_dev[i], u_val_dev[i], hk_idx, hk_val)
            kmat[i, k] = -(dot / eps)

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
) -> None:
    """Apply `X <- Φ(X @ mat)` on GPU (column-wise), writing into the alternate buffer then swapping."""

    if cp is None or _guga_cuda_ext is None:  # pragma: no cover
        raise RuntimeError("CUDA QMC backend unavailable (requires cupy and asuka._guga_cuda_ext)")
    if ctx.ws is None:
        raise RuntimeError("CudaBlockProjectorContext is released")
    if ctx.x_idx is None or ctx.x_val is None or ctx.x_idx_next is None or ctx.x_val_next is None:
        raise RuntimeError("CudaBlockProjectorContext buffers not initialized")
    if ctx.nnz is None or ctx.nnz_next is None:
        raise RuntimeError("CudaBlockProjectorContext nnz buffers not initialized")
    if ctx.idx_all is None or ctx.val_all is None or ctx.idx_u is None or ctx.val_u is None or ctx.nnz_u is None or ctx.nnz_out is None:
        raise RuntimeError("CudaBlockProjectorContext scratch buffers not initialized")

    mat = np.asarray(mat, dtype=np.float64)
    nroots = int(ctx.nroots)
    if mat.shape != (nroots, nroots):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(nroots, nroots)})")

    seeds_phi = np.asarray(seeds_phi, dtype=np.int64).ravel()
    need = nroots * (nroots - 1)
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

    # Build each output column incrementally: y <- Φ(y + w*x_j).
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
