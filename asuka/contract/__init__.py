from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
import os
import threading
import time
import weakref

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.blas_threads import blas_thread_limit
from asuka.cuguga.oracle import (
    _csr_for_epq,
    _get_epq_action_cache,
    _restore_eri_4d,
    connected_row as default_connected_row,
    occ_table,
    precompute_epq_actions,
)

try:  # optional SciPy-backed sparse matmul for E_pq applications
    from scipy import sparse as _sp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _sp = None

try:  # optional Cython sparse @ dense kernels (avoid allocations in hot loops)
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csr_matmul_dense_add_cy as _csr_matmul_dense_add_cy,
        csr_matmul_dense_add_scaled_cy as _csr_matmul_dense_add_scaled_cy,
        csr_matmul_dense_inplace_cy as _csr_matmul_dense_inplace_cy,
        csc_matmul_dense_add_cy as _csc_matmul_dense_add_cy,
        csc_matmul_dense_add_scaled_cy as _csc_matmul_dense_add_scaled_cy,
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
        openmp_set_num_threads as _openmp_set_num_threads,
    )
except Exception:  # pragma: no cover
    _csr_matmul_dense_add_cy = None
    _csr_matmul_dense_add_scaled_cy = None
    _csr_matmul_dense_inplace_cy = None
    _csc_matmul_dense_add_cy = None
    _csc_matmul_dense_add_scaled_cy = None
    _csc_matmul_dense_inplace_cy = None
    _openmp_set_num_threads = None

try:  # optional fused symmetric kernels (may be missing in older builds)
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_matmul_dense_sym_add_cy as _csc_matmul_dense_sym_add_cy,
        csc_matmul_dense_sym_inplace_cy as _csc_matmul_dense_sym_inplace_cy,
    )
except Exception:  # pragma: no cover
    _csc_matmul_dense_sym_add_cy = None
    _csc_matmul_dense_sym_inplace_cy = None

try:  # optional OpenMP pair-parallel helpers (avoid Python ThreadPoolExecutor)
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        have_openmp as _have_openmp,
        openmp_max_threads as _openmp_max_threads,
        csc_matmul_dense_add_many_indexed_omp_cy as _csc_matmul_dense_add_many_indexed_omp_cy,
        csc_matmul_dense_inplace_many_indexed_cy as _csc_matmul_dense_inplace_many_indexed_cy,
        csc_matmul_dense_sym_add_many_indexed_omp_cy as _csc_matmul_dense_sym_add_many_indexed_omp_cy,
        csc_matmul_dense_sym_inplace_many_indexed_cy as _csc_matmul_dense_sym_inplace_many_indexed_cy,
    )
except Exception:  # pragma: no cover
    _have_openmp = None
    _openmp_max_threads = None
    _csc_matmul_dense_add_many_indexed_omp_cy = None
    _csc_matmul_dense_inplace_many_indexed_cy = None
    _csc_matmul_dense_sym_add_many_indexed_omp_cy = None
    _csc_matmul_dense_sym_inplace_many_indexed_cy = None


_THREAD_LOCAL = threading.local()


def _maybe_limit_openmp_in_worker(force: bool) -> None:
    """Ensure OpenMP is limited to 1 thread in this Python worker thread."""

    if not force or _openmp_set_num_threads is None:
        return
    if bool(getattr(_THREAD_LOCAL, "openmp_single", False)):
        return
    _openmp_set_num_threads(1)  # type: ignore[misc]
    _THREAD_LOCAL.openmp_single = True


def _maybe_set_openmp_wait_policy() -> None:
    """Best-effort OpenMP runtime tuning to reduce oversubscription.

    Notes
    -----
    Setting ``OMP_WAIT_POLICY=PASSIVE`` can reduce CPU burn/spin-wait from idle
    OpenMP worker threads, which otherwise can interfere with BLAS-heavy phases
    (e.g. Davidson orth/subspace) in mixed workloads.
    """

    if os.environ.get("OMP_WAIT_POLICY", "").strip():
        return
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

_EPQ_SPMAT_CACHE: weakref.WeakKeyDictionary[DRT, dict[int, object]] = weakref.WeakKeyDictionary()
_EPQ_SPMAT_CSR_CACHE: weakref.WeakKeyDictionary[DRT, dict[int, object]] = weakref.WeakKeyDictionary()
_EPQ_SYM_SPMAT_CACHE: weakref.WeakKeyDictionary[
    DRT, tuple[np.ndarray, np.ndarray, list[object | None]]
] = weakref.WeakKeyDictionary()
_EPQ_SYM_SPMAT_CSR_CACHE: weakref.WeakKeyDictionary[
    DRT, tuple[np.ndarray, np.ndarray, list[object | None]]
] = weakref.WeakKeyDictionary()


@dataclass
class ContractWorkspace:
    """Reusable scratch buffers for dense-intermediate contraction backends.

    Notes
    -----
    Instances are safe to reuse across repeated calls, but not safe to share
    across concurrent calls without external synchronization.
    """

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Symmetric-pair fastpath buffers (npair x m).
    sym_npair: int = 0
    sym_m_capacity: int = 0
    sym_t_buf: np.ndarray | None = None
    sym_g_buf: np.ndarray | None = None
    sym_h1_buf: np.ndarray | None = None
    sym_parts_nthreads: int = 0
    sym_parts_m_capacity: int = 0
    sym_parts_buf: np.ndarray | None = None

    # General pair-space buffers (nops x m).
    gen_nops: int = 0
    gen_m_capacity: int = 0
    gen_t_buf: np.ndarray | None = None
    gen_g_buf: np.ndarray | None = None

    # Cached occupancy table derived from EPQ action cache steps (0/1/2), shape (ncsf, norb).
    # Keep int8 to reduce memory for larger CSF spaces.
    occ_shape: tuple[int, int] = (0, 0)
    occ_i8: np.ndarray | None = None

    hdiag_n: int = 0
    hdiag_buf: np.ndarray | None = None
    hdiag_tmp_buf: np.ndarray | None = None
    executor: ThreadPoolExecutor | None = field(default=None, repr=False)
    executor_max_workers: int = 0

    def _ensure_occ(self, *, steps: np.ndarray) -> np.ndarray:
        steps = np.asarray(steps)
        if steps.ndim != 2:
            raise ValueError("steps must be 2D")
        shape = (int(steps.shape[0]), int(steps.shape[1]))
        if self.occ_i8 is None or tuple(self.occ_shape) != tuple(shape):
            # steps codes: 0=empty, 1=up, 2=down, 3=double -> occ 0/1/1/2
            step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int8)
            self.occ_i8 = np.asarray(step_to_occ[steps], dtype=np.int8, order="F")
            self.occ_shape = shape
        return self.occ_i8

    def _ensure_hdiag(self, *, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if self.hdiag_buf is None or self.hdiag_tmp_buf is None or int(self.hdiag_n) < n:
            self.hdiag_n = max(int(self.hdiag_n), n)
            self.hdiag_buf = np.empty(int(self.hdiag_n), dtype=np.float64)
            self.hdiag_tmp_buf = np.empty(int(self.hdiag_n), dtype=np.float64)
        return self.hdiag_buf[:n], self.hdiag_tmp_buf[:n]

    def _ensure_sym(self, *, npair: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        npair = int(npair)
        m = int(m)
        if npair < 0 or m < 0:
            raise ValueError("npair/m must be >= 0")
        if self.sym_npair != npair:
            self.sym_npair = npair
            self.sym_m_capacity = 0
            self.sym_t_buf = None
            self.sym_g_buf = None
            self.sym_h1_buf = None

        if m > self.sym_m_capacity or self.sym_t_buf is None or self.sym_g_buf is None or self.sym_h1_buf is None:
            self.sym_m_capacity = max(int(self.sym_m_capacity), m)
            self.sym_t_buf = np.empty(npair * self.sym_m_capacity, dtype=np.float64)
            self.sym_g_buf = np.empty(npair * self.sym_m_capacity, dtype=np.float64)
            self.sym_h1_buf = np.empty(self.sym_m_capacity, dtype=np.float64)

        t_flat = self.sym_t_buf[: npair * m].reshape(npair, m)
        g_flat = self.sym_g_buf[: npair * m].reshape(npair, m)
        h1_flat = self.sym_h1_buf[:m]
        return t_flat, g_flat, h1_flat

    def _ensure_sym_parts(self, *, nthreads: int, m: int) -> np.ndarray:
        nthreads = int(nthreads)
        m = int(m)
        if nthreads < 1 or m < 0:
            raise ValueError("nthreads must be >= 1 and m must be >= 0")
        if self.sym_parts_nthreads != nthreads:
            self.sym_parts_nthreads = nthreads
            self.sym_parts_m_capacity = 0
            self.sym_parts_buf = None
        if m > self.sym_parts_m_capacity or self.sym_parts_buf is None:
            self.sym_parts_m_capacity = max(int(self.sym_parts_m_capacity), m)
            self.sym_parts_buf = np.empty(nthreads * self.sym_parts_m_capacity, dtype=np.float64)
        return self.sym_parts_buf[: nthreads * m].reshape(nthreads, m)

    def _ensure_gen(self, *, nops: int, m: int) -> tuple[np.ndarray, np.ndarray]:
        nops = int(nops)
        m = int(m)
        if nops < 0 or m < 0:
            raise ValueError("nops/m must be >= 0")
        if self.gen_nops != nops:
            self.gen_nops = nops
            self.gen_m_capacity = 0
            self.gen_t_buf = None
            self.gen_g_buf = None

        if m > self.gen_m_capacity or self.gen_t_buf is None or self.gen_g_buf is None:
            self.gen_m_capacity = max(int(self.gen_m_capacity), m)
            self.gen_t_buf = np.empty(nops * self.gen_m_capacity, dtype=np.float64)
            self.gen_g_buf = np.empty(nops * self.gen_m_capacity, dtype=np.float64)

        t_flat = self.gen_t_buf[: nops * m].reshape(nops, m)
        g_flat = self.gen_g_buf[: nops * m].reshape(nops, m)
        return t_flat, g_flat

    def _ensure_executor(self, *, max_workers: int) -> ThreadPoolExecutor:
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.executor is None or self.executor_max_workers != max_workers:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.executor_max_workers = max_workers
        return self.executor

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.executor_max_workers = 0


def contract_h_py(
    drt: DRT,
    h1e,
    eri,
    x,
    *,
    row_oracle: Callable[[DRT, object, object, int], tuple[np.ndarray, np.ndarray]]
    | None = None,
    max_out: int = 200_000,
) -> np.ndarray:
    """Row-driven contraction y = H*x in DRT/CSF basis (pure Python prototype)."""

    ncsf = int(drt.ncsf)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size != ncsf:
        raise ValueError(f"x has wrong length: {x.size} (expected {ncsf})")

    if row_oracle is None:
        row_oracle = default_connected_row

    y = np.zeros(ncsf, dtype=np.float64)
    for j in range(ncsf):
        i_idx, hij = row_oracle(drt, h1e, eri, j, max_out=max_out)
        y[i_idx] += hij * x[j]
    return y


def _split_chunks(n: int, chunks: int) -> list[tuple[int, int]]:
    n = int(n)
    chunks = int(chunks)
    if n < 0:
        raise ValueError("n must be >= 0")
    if chunks < 1:
        raise ValueError("chunks must be >= 1")
    out: list[tuple[int, int]] = []
    for i in range(chunks):
        start = (i * n) // chunks
        stop = ((i + 1) * n) // chunks
        if start != stop:
            out.append((start, stop))
    return out


def _epq_spmat(drt: DRT, cache, p: int, q: int):
    if _sp is None:
        raise RuntimeError("SciPy is required for the threaded contract backend")
    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    pair_id = int(p) * norb + int(q)
    per_drt = _EPQ_SPMAT_CACHE.get(drt)
    if per_drt is None:
        per_drt = {}
        _EPQ_SPMAT_CACHE[drt] = per_drt
    mat = per_drt.get(pair_id)
    if mat is None:
        csr = _csr_for_epq(cache, drt, int(p), int(q))
        mat = _sp.csc_matrix((csr.data, csr.indices, csr.indptr), shape=(ncsf, ncsf))
        per_drt[pair_id] = mat
    return mat


def _epq_spmat_csr(drt: DRT, cache, p: int, q: int):
    if _sp is None:
        raise RuntimeError("SciPy is required for the threaded contract backend")
    norb = int(drt.norb)
    pair_id = int(p) * norb + int(q)
    per_drt = _EPQ_SPMAT_CSR_CACHE.get(drt)
    if per_drt is None:
        per_drt = {}
        _EPQ_SPMAT_CSR_CACHE[drt] = per_drt
    mat = per_drt.get(pair_id)
    if mat is None:
        mat_csc = _epq_spmat(drt, cache, int(p), int(q))
        mat = mat_csc.tocsr()  # type: ignore[operator]
        per_drt[pair_id] = mat
    return mat


def _epq_spmat_list(drt: DRT, cache) -> list[object | None]:
    if _sp is None:
        raise RuntimeError("SciPy is required for the threaded contract backend")
    norb = int(drt.norb)
    mats: list[object | None] = [None] * (norb * norb)
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            mats[p * norb + q] = _epq_spmat(drt, cache, int(p), int(q))
    return mats


def _epq_spmat_list_csr(drt: DRT, cache) -> list[object | None]:
    if _sp is None:
        raise RuntimeError("SciPy is required for the threaded contract backend")
    norb = int(drt.norb)
    mats: list[object | None] = [None] * (norb * norb)
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            mats[p * norb + q] = _epq_spmat_csr(drt, cache, int(p), int(q))
    return mats


def _epq_sym_spmat_list(
    drt: DRT,
    cache,
    *,
    nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[object | None]]:
    """Return (p_idx, q_idx, sym_mats) for unordered pairs p<=q.

    For p<q, the symmetrized operator is:
        S_pq = E_pq + E_qp
    while for p==q we treat S_pp = E_pp (diagonal occupancy action, no sparse matrix).

    Notes
    -----
    For p<q, we cache only the base operator ``E_pq`` (as a CSC sparse matrix).
    The symmetric operator is applied on the fly as ``S_pq @ x = E_pq @ x + E_pq.T @ x``.
    """

    cached = _EPQ_SYM_SPMAT_CACHE.get(drt)
    if cached is not None:
        return cached

    norb = int(drt.norb)
    # Build only the upper-triangular (p<q) pairs needed by the symmetric-pair
    # fastpath. The Cython E_pq builder releases the GIL, so this can parallelize
    # across threads and avoid a large single-thread "cache build" spike.
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    pairs = [(p, q) for p in range(norb) for q in range(p + 1, norb)]
    if pairs:
        precompute_epq_actions(drt, nthreads=nthreads, pairs=pairs)
    npair = norb * (norb + 1) // 2
    p_idx = np.empty(npair, dtype=np.int32)
    q_idx = np.empty(npair, dtype=np.int32)
    sym_mats: list[object | None] = [None] * npair

    k = 0
    for p in range(norb):
        for q in range(p, norb):
            p_idx[k] = int(p)
            q_idx[k] = int(q)
            if p != q:
                # Store the base operator E_pq; the symmetric operator is applied
                # as S_pq @ x = E_pq @ x + E_pq.T @ x to avoid constructing and
                # caching an explicit S_pq sparse matrix.
                sym_mats[k] = _epq_spmat(drt, cache, int(p), int(q))
            k += 1

    out = (p_idx, q_idx, sym_mats)
    _EPQ_SYM_SPMAT_CACHE[drt] = out
    return out


def _epq_sym_spmat_list_csr(
    drt: DRT,
    cache,
    *,
    nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[object | None]]:
    if _sp is None:
        raise RuntimeError("SciPy is required for the threaded contract backend")
    cached = _EPQ_SYM_SPMAT_CSR_CACHE.get(drt)
    if cached is not None:
        return cached

    p_idx, q_idx, sym_mats = _epq_sym_spmat_list(drt, cache, nthreads=int(nthreads))
    sym_mats_csr: list[object | None] = [None] * len(sym_mats)
    for i, mat in enumerate(sym_mats):
        if mat is None:
            continue
        sym_mats_csr[i] = mat.tocsr()  # type: ignore[operator]
    out = (p_idx, q_idx, sym_mats_csr)
    _EPQ_SYM_SPMAT_CSR_CACHE[drt] = out
    return out


def _pick_spmm_backend(spmm_backend: str, *, nthreads: int) -> str:
    """Pick sparse matmul backend ('csc' or 'csr') for a single contract call."""

    s = str(spmm_backend).strip().lower()
    if s in ("", "auto"):
        # Never nest OpenMP inside Python-thread parallel loops.
        if int(nthreads) > 1:
            return "csc"

        env = os.environ.get("GUGA_SPMM_BACKEND", "").strip().lower()
        if env in ("csc", "csr"):
            return env
        return "csc"
    if s in ("csc", "csr"):
        return s
    raise ValueError(f"unknown spmm_backend={spmm_backend!r}")

def contract_h_csf(
    drt: DRT,
    h1e,
    eri,
    x,
    *,
    precompute_epq: bool = True,
) -> np.ndarray:
    """Contract y = H*x in the CSF/DRT basis using cached E_pq actions.

    This routine evaluates the spin-free Hamiltonian in generator form

        H = Σ_pq h_pq E_pq + 1/2 Σ_pqrs (pq|rs) (E_pq E_rs - δ_qr E_ps)

    via repeated sparse applications of the one-body generators E_pq.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size != ncsf:
        raise ValueError(f"x has wrong length: {x.size} (expected {ncsf})")

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    if precompute_epq:
        precompute_epq_actions(drt)

    cache = _get_epq_action_cache(drt)
    occ = occ_table(drt).astype(np.float64, copy=False)
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    # Fold the contraction term (-δ_qr E_ps) into an effective 1-body coefficient matrix.
    h_eff = h1e - 0.5 * np.einsum("pqqs->ps", eri4)

    y = np.zeros(ncsf, dtype=np.float64)

    # One-body diagonal: Σ_p h_eff[p,p] occ_p |j>
    diag_h = np.diag(h_eff).astype(np.float64, copy=False)
    y += (occ @ diag_h) * x

    # One-body off-diagonal: Σ_{p!=q} h_eff[p,q] E_pq |x>
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            hpq = float(h_eff[p, q])
            if hpq == 0.0:
                continue
            csr = _csr_for_epq(cache, drt, p, q)
            for j in range(ncsf):
                start = int(csr.indptr[j])
                end = int(csr.indptr[j + 1])
                if start == end:
                    continue
                y[csr.indices[start:end]] += hpq * csr.data[start:end] * x[j]

    # Two-body product terms: 1/2 Σ_{pqrs} (pq|rs) E_pq E_rs |x>
    nops = norb * norb
    t_rs = np.zeros((nops, ncsf), dtype=np.float64)
    for r in range(norb):
        for s in range(norb):
            rs = r * norb + s
            if r == s:
                t_rs[rs] = occ[:, r] * x
                continue
            csr = _csr_for_epq(cache, drt, r, s)
            out = t_rs[rs]
            for j in range(ncsf):
                start = int(csr.indptr[j])
                end = int(csr.indptr[j + 1])
                if start == end:
                    continue
                out[csr.indices[start:end]] += csr.data[start:end] * x[j]

    eri_mat = (0.5 * eri4.reshape(nops, nops)).astype(np.float64, copy=False)
    g_pq = eri_mat @ t_rs  # (p,q) -> vector over CSFs

    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            g = g_pq[pq]
            if p == q:
                y += occ[:, p] * g
                continue
            csr = _csr_for_epq(cache, drt, p, q)
            for j in range(ncsf):
                start = int(csr.indptr[j])
                end = int(csr.indptr[j + 1])
                if start == end:
                    continue
                y[csr.indices[start:end]] += csr.data[start:end] * g[j]

    return y


def contract_eri_epq_eqrs(
    drt: DRT,
    eri,
    x,
    *,
    precompute_epq: bool = True,
) -> np.ndarray:
    """Contract y = (Σ_pqrs eri[pqrs] E_pq E_rs) x in the CSF/DRT basis.

    This matches the operator form used by PySCF's determinant FCI `contract_2e`
    after calling `absorb_h1e(..., fac=.5)`, i.e. `eri` is expected to already
    include the 1e contributions and any required scaling.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size != ncsf:
        raise ValueError(f"x has wrong length: {x.size} (expected {ncsf})")

    if precompute_epq:
        precompute_epq_actions(drt)

    cache = _get_epq_action_cache(drt)
    occ = occ_table(drt).astype(np.float64, copy=False)
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    y = np.zeros(ncsf, dtype=np.float64)

    nops = norb * norb
    t_rs = np.zeros((nops, ncsf), dtype=np.float64)
    for r in range(norb):
        for s in range(norb):
            rs = r * norb + s
            if r == s:
                t_rs[rs] = occ[:, r] * x
                continue
            csr = _csr_for_epq(cache, drt, r, s)
            out = t_rs[rs]
            for j in range(ncsf):
                start = int(csr.indptr[j])
                end = int(csr.indptr[j + 1])
                if start == end:
                    continue
                out[csr.indices[start:end]] += csr.data[start:end] * x[j]

    eri_mat = eri4.reshape(nops, nops)
    g_pq = eri_mat @ t_rs

    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            g = g_pq[pq]
            if p == q:
                y += occ[:, p] * g
                continue
            csr = _csr_for_epq(cache, drt, p, q)
            for j in range(ncsf):
                start = int(csr.indptr[j])
                end = int(csr.indptr[j + 1])
                if start == end:
                    continue
                y[csr.indices[start:end]] += csr.data[start:end] * g[j]

    return y


def contract_eri_epq_eqrs_multi(
    drt: DRT,
    eri,
    xs: list[np.ndarray] | np.ndarray,
    *,
    precompute_epq: bool = True,
    nthreads: int = 1,
    blas_nthreads: int | None = None,
    executor: ThreadPoolExecutor | None = None,
    workspace: ContractWorkspace | None = None,
    spmm_backend: str = "auto",
) -> list[np.ndarray]:
    """Vectorized (multi-vector) `contract_eri_epq_eqrs` with optional threading.

    This is the CSF/DRT counterpart to PySCF determinant-FCI `contract_2e`:
    `eri` is expected to be the result of `absorb_h1e(..., fac=.5)` (i.e. the
    1e part is already folded into `eri` and any scaling is included there).

    Notes
    -----
    If ``blas_nthreads`` is provided, BLAS threads are temporarily limited during
    the dense pair-space GEMM; otherwise the BLAS library default settings are
    used.

    If ``executor`` is provided, it is reused for the internal threaded loops,
    which can reduce overhead when this function is called repeatedly (e.g. by
    Davidson iterations).
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    if blas_nthreads is not None:
        blas_nthreads = int(blas_nthreads)
        if blas_nthreads < 1:
            raise ValueError("blas_nthreads must be >= 1")

    xmat = np.asarray(xs, dtype=np.float64)
    if xmat.ndim == 1:
        xmat = xmat.reshape(1, -1)
    if xmat.shape[1] != ncsf:
        raise ValueError(f"xs has wrong shape: {xmat.shape} (expected (*, {ncsf}))")

    nvec = int(xmat.shape[0])

    if precompute_epq:
        precompute_epq_actions(drt)

    cache = _get_epq_action_cache(drt)
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    if _sp is None:
        # Slow fallback (kept for environments without SciPy).
        occ = occ_table(drt).astype(np.float64, copy=False)
        y = np.zeros((nvec, ncsf), dtype=np.float64)

        nops = norb * norb
        t_rs = np.zeros((nops, ncsf, nvec), dtype=np.float64)
        for r in range(norb):
            for s in range(norb):
                rs = r * norb + s
                if r == s:
                    t_rs[rs] = (occ[:, r][:, None] * xmat.T).astype(np.float64, copy=False)
                    continue
                csr = _csr_for_epq(cache, drt, r, s)
                out = t_rs[rs]
                for j in range(ncsf):
                    start = int(csr.indptr[j])
                    end = int(csr.indptr[j + 1])
                    if start == end:
                        continue
                    out[csr.indices[start:end]] += csr.data[start:end][:, None] * xmat[:, j][None, :]

        eri_mat = eri4.reshape(nops, nops).astype(np.float64, copy=False)
        g_flat = eri_mat @ t_rs.reshape(nops, ncsf * nvec)
        g_pq = g_flat.reshape(nops, ncsf, nvec)

        for p in range(norb):
            for q in range(norb):
                pq = p * norb + q
                g = g_pq[pq]  # (ncsf, nvec)
                if p == q:
                    y += (occ[:, p][:, None] * g).T
                    continue
                csr = _csr_for_epq(cache, drt, p, q)
                for j in range(ncsf):
                    start = int(csr.indptr[j])
                    end = int(csr.indptr[j + 1])
                    if start == end:
                        continue
                    y[:, csr.indices[start:end]] += g[j][:, None] * csr.data[start:end][None, :]

        return [np.ascontiguousarray(y[i]) for i in range(nvec)]

    # OpenMP thread settings are per OS thread. When the threaded contract backend is
    # enabled, each worker thread is configured once via `_maybe_limit_openmp_in_worker`,
    # so we avoid toggling OpenMP settings in the main thread here.
    with contextlib.nullcontext():
        # SciPy sparse matmul path (supports thread-parallel execution).
        spmm_backend = _pick_spmm_backend(spmm_backend, nthreads=nthreads)
        if spmm_backend == "csr":
            mats = _epq_spmat_list_csr(drt, cache)
            spmm_inplace = _csr_matmul_dense_inplace_cy
            spmm_add = _csr_matmul_dense_add_cy
        else:
            mats = _epq_spmat_list(drt, cache)
            spmm_inplace = _csc_matmul_dense_inplace_cy
            spmm_add = _csc_matmul_dense_add_cy
        if workspace is not None:
            occ = workspace._ensure_occ(steps=cache.steps)
        else:
            step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int8)
            occ = np.asarray(step_to_occ[cache.steps], dtype=np.int8, order="F")

        x = np.ascontiguousarray(xmat.T)  # (ncsf, nvec)
        y = np.zeros((ncsf, nvec), dtype=np.float64)

        nops = norb * norb
        m = ncsf * nvec
        ws_lock = None if workspace is None else workspace.lock
        if ws_lock is not None:
            ws_lock.acquire()
        try:
            if workspace is None:
                t_flat = np.empty((nops, m), dtype=np.float64)
                g_flat = np.empty((nops, m), dtype=np.float64)
            else:
                t_flat, g_flat = workspace._ensure_gen(nops=nops, m=m)

            force_omp_single = nthreads > 1 and _openmp_set_num_threads is not None

            def t_worker(start_stop: tuple[int, int]) -> None:
                _maybe_limit_openmp_in_worker(force_omp_single)
                start, stop = start_stop
                for rs in range(start, stop):
                    r, s = divmod(int(rs), norb)
                    out = t_flat[rs].reshape(ncsf, nvec)
                    if r == s:
                        np.multiply(occ[:, r][:, None], x, out=out)
                        continue
                    mat = mats[rs]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if spmm_inplace is not None:
                        spmm_inplace(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                    else:
                        out[:] = mat.dot(x)  # type: ignore[operator]

            def g_worker(start_stop: tuple[int, int]) -> np.ndarray:
                _maybe_limit_openmp_in_worker(force_omp_single)
                start, stop = start_stop
                out = np.zeros((ncsf, nvec), dtype=np.float64)
                for pq in range(start, stop):
                    p, q = divmod(int(pq), norb)
                    g = g_flat[pq].reshape(ncsf, nvec)
                    if p == q:
                        out += occ[:, p][:, None] * g
                        continue
                    mat = mats[pq]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if spmm_add is not None:
                        spmm_add(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                    else:
                        out += mat.dot(g)  # type: ignore[operator]
                return out

            rs_chunks = _split_chunks(nops, min(nthreads, nops))
            pq_chunks = _split_chunks(nops, min(nthreads, nops))
            max_workers = max([1, len(rs_chunks), len(pq_chunks)])

            eri_mat = eri4.reshape(nops, nops).astype(np.float64, copy=False)

            if max_workers <= 1:
                t_worker(rs_chunks[0])
                if blas_nthreads is None:
                    np.matmul(eri_mat, t_flat, out=g_flat)
                else:
                    with blas_thread_limit(int(blas_nthreads)):
                        np.matmul(eri_mat, t_flat, out=g_flat)
                y += g_worker(pq_chunks[0])
            else:
                if executor is None:
                    if workspace is not None:
                        pool = workspace._ensure_executor(max_workers=max_workers)
                        shutdown = False
                    else:
                        pool_cm: object = ThreadPoolExecutor(max_workers=max_workers)
                        pool = pool_cm.__enter__()  # type: ignore[assignment]
                        shutdown = True
                else:
                    pool = executor
                    shutdown = False

                try:
                    if len(rs_chunks) == 1:
                        t_worker(rs_chunks[0])
                    else:
                        with blas_thread_limit(1):
                            list(pool.map(t_worker, rs_chunks))

                    if blas_nthreads is None:
                        np.matmul(eri_mat, t_flat, out=g_flat)
                    else:
                        with blas_thread_limit(int(blas_nthreads)):
                            np.matmul(eri_mat, t_flat, out=g_flat)

                    if len(pq_chunks) == 1:
                        y += g_worker(pq_chunks[0])
                    else:
                        with blas_thread_limit(1):
                            for part in pool.map(g_worker, pq_chunks):
                                y += part
                finally:
                    if shutdown:
                        pool_cm.__exit__(None, None, None)  # type: ignore[attr-defined]
        finally:
            if ws_lock is not None:
                ws_lock.release()

        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]


def contract_h_csf_multi(
    drt: DRT,
    h1e,
    eri,
    xs: list[np.ndarray] | np.ndarray,
    *,
    precompute_epq: bool = True,
    nthreads: int = 1,
    blas_nthreads: int | None = None,
    executor: ThreadPoolExecutor | None = None,
    workspace: ContractWorkspace | None = None,
    spmm_backend: str = "auto",
    profile_out: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Vectorized variant of :func:`contract_h_csf` for multiple trial vectors.

    Notes
    -----
    If ``blas_nthreads`` is provided, BLAS threads are temporarily limited during
    the dense pair-space GEMM; otherwise the BLAS library default settings are
    used.

    If ``executor`` is provided, it is reused for the internal threaded loops,
    which can reduce overhead when this function is called repeatedly (e.g. by
    Davidson iterations).
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    if blas_nthreads is not None:
        blas_nthreads = int(blas_nthreads)
        if blas_nthreads < 1:
            raise ValueError("blas_nthreads must be >= 1")

    xmat = np.asarray(xs, dtype=np.float64)
    if xmat.ndim == 1:
        xmat = xmat.reshape(1, -1)
    if xmat.shape[1] != ncsf:
        raise ValueError(f"xs has wrong shape: {xmat.shape} (expected (*, {ncsf}))")

    nvec = int(xmat.shape[0])
    t_total0 = 0.0
    if profile_out is not None:
        profile_out.clear()
        profile_out["calls"] = 1.0
        profile_out["nvec"] = float(nvec)
        profile_out["nthreads"] = float(nthreads)
        profile_out["blas_nthreads"] = float(0 if blas_nthreads is None else int(blas_nthreads))
        t_total0 = time.perf_counter()

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    if precompute_epq:
        precompute_epq_actions(drt)

    cache = _get_epq_action_cache(drt)
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    h_eff = h1e - 0.5 * np.einsum("pqqs->ps", eri4)

    if _sp is None:
        # Slow fallback (kept for environments without SciPy).
        occ = occ_table(drt).astype(np.float64, copy=False)
        y = np.zeros((nvec, ncsf), dtype=np.float64)

        diag_h = np.diag(h_eff).astype(np.float64, copy=False)
        y += (occ @ diag_h)[None, :] * xmat

        for p in range(norb):
            for q in range(norb):
                if p == q:
                    continue
                hpq = float(h_eff[p, q])
                if hpq == 0.0:
                    continue
                csr = _csr_for_epq(cache, drt, p, q)
                for j in range(ncsf):
                    start = int(csr.indptr[j])
                    end = int(csr.indptr[j + 1])
                    if start == end:
                        continue
                    y[:, csr.indices[start:end]] += (
                        hpq * csr.data[start:end][None, :] * xmat[:, j][:, None]
                    )

        nops = norb * norb
        t_rs = np.zeros((nops, ncsf, nvec), dtype=np.float64)
        for r in range(norb):
            for s in range(norb):
                rs = r * norb + s
                if r == s:
                    t_rs[rs] = (occ[:, r][:, None] * xmat.T).astype(np.float64, copy=False)
                    continue
                csr = _csr_for_epq(cache, drt, r, s)
                out = t_rs[rs]
                for j in range(ncsf):
                    start = int(csr.indptr[j])
                    end = int(csr.indptr[j + 1])
                    if start == end:
                        continue
                    out[csr.indices[start:end]] += csr.data[start:end][:, None] * xmat[:, j][None, :]

        eri_mat = (0.5 * eri4.reshape(nops, nops)).astype(np.float64, copy=False)
        g_flat = eri_mat @ t_rs.reshape(nops, ncsf * nvec)
        g_pq = g_flat.reshape(nops, ncsf, nvec)

        for p in range(norb):
            for q in range(norb):
                pq = p * norb + q
                g = g_pq[pq]  # (ncsf, nvec)
                if p == q:
                    y += (occ[:, p][:, None] * g).T
                    continue
                csr = _csr_for_epq(cache, drt, p, q)
                for j in range(ncsf):
                    start = int(csr.indptr[j])
                    end = int(csr.indptr[j + 1])
                    if start == end:
                        continue
                    y[:, csr.indices[start:end]] += g[j][:, None] * csr.data[start:end][None, :]

        return [np.ascontiguousarray(y[i]) for i in range(nvec)]

    # See note above: rely on per-worker OpenMP limiting, avoid main-thread toggling.
    with contextlib.nullcontext():
        # SciPy sparse matmul path (supports thread-parallel execution).
        spmm_backend = _pick_spmm_backend(spmm_backend, nthreads=nthreads)
        if profile_out is not None:
            profile_out["spmm_backend_csr"] = 1.0 if spmm_backend == "csr" else 0.0
        t0 = time.perf_counter() if profile_out is not None else 0.0
        if workspace is not None:
            occ = workspace._ensure_occ(steps=cache.steps)
        else:
            step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int8)
            occ = np.asarray(step_to_occ[cache.steps], dtype=np.int8, order="F")
        if profile_out is not None:
            profile_out["occ_s"] = time.perf_counter() - t0

        x = np.ascontiguousarray(xmat.T)  # (ncsf, nvec)
        y = np.zeros((ncsf, nvec), dtype=np.float64)

        t0 = time.perf_counter() if profile_out is not None else 0.0
        diag_h = np.diag(h_eff).astype(np.float64, copy=False)
        if workspace is None:
            occ_diag = np.zeros((ncsf,), dtype=np.float64)
            tmp = np.empty((ncsf,), dtype=np.float64)
        else:
            occ_diag, tmp = workspace._ensure_hdiag(n=ncsf)
            occ_diag.fill(0.0)
        for p in range(norb):
            coeff = float(diag_h[p])
            if coeff == 0.0:
                continue
            np.multiply(occ[:, p], coeff, out=tmp, casting="unsafe")
            occ_diag += tmp
        np.multiply(x, occ_diag[:, None], out=y)
        if profile_out is not None:
            profile_out["hdiag_apply_s"] = time.perf_counter() - t0

        # Symmetric-pair fastpath (dense ERIs): reduce pair space from norb^2 to norb*(norb+1)/2
        # by grouping (p,q) and (q,p) together using S_pq = E_pq + E_qp. This is safe when:
        #   - h_eff is symmetric (Hermitian 1e part), and
        #   - ERIs satisfy (pq|rs)=(qp|rs) and (pq|rs)=(pq|sr).
        sym_tol = 1e-12
        if (
            float(np.max(np.abs(h_eff - h_eff.T))) <= sym_tol
            and float(np.max(np.abs(eri4 - eri4.transpose(1, 0, 2, 3)))) <= sym_tol
            and float(np.max(np.abs(eri4 - eri4.transpose(0, 1, 3, 2)))) <= sym_tol
        ):
            if profile_out is not None:
                profile_out["sym_path_calls"] = 1.0
            if spmm_backend == "csr":
                p_idx, q_idx, sym_mats = _epq_sym_spmat_list_csr(drt, cache, nthreads=nthreads)
                spmm_inplace = _csr_matmul_dense_inplace_cy
                spmm_add = _csr_matmul_dense_add_cy
                spmm_add_t = _csc_matmul_dense_add_cy
            else:
                p_idx, q_idx, sym_mats = _epq_sym_spmat_list(drt, cache, nthreads=nthreads)
                spmm_inplace = _csc_matmul_dense_inplace_cy
                spmm_add = _csc_matmul_dense_add_cy
                spmm_add_t = _csr_matmul_dense_add_cy
            # Optional fused kernel for the symmetric operator S = A + A.T (cuts sparse passes in half).
            spmm_sym_inplace = _csc_matmul_dense_sym_inplace_cy
            spmm_sym_add = _csc_matmul_dense_sym_add_cy
            if profile_out is not None and spmm_sym_inplace is not None and spmm_sym_add is not None:
                profile_out["sym_fused_calls"] = 1.0
            npair = int(p_idx.size)
            m = ncsf * nvec

            # Build t_flat[u] = S_pq |x> for all unordered pairs (p<=q), and accumulate
            # one-body off-diagonal contributions y += Σ_{p<q} h_eff[p,q] * S_pq |x>.
            ws_lock = None if workspace is None else workspace.lock
            if ws_lock is not None:
                ws_lock.acquire()
            try:
                if workspace is None:
                    t_flat = np.empty((npair, m), dtype=np.float64)
                    g_flat = np.empty((npair, m), dtype=np.float64)
                    h1_flat = np.empty(m, dtype=np.float64)
                else:
                    t_flat, g_flat, h1_flat = workspace._ensure_sym(npair=npair, m=m)

                offdiag = p_idx != q_idx

                # Prefer OpenMP pair-parallel kernels to avoid Python ThreadPoolExecutor overhead
                # and reduce the risk of oversubscription. Can be overridden via env var:
                #   CUGUGA_CONTRACT_PARALLEL_BACKEND=python|openmp|auto
                backend_env = os.environ.get("CUGUGA_CONTRACT_PARALLEL_BACKEND", "").strip().lower()
                prefer_openmp = backend_env in ("", "auto", "openmp", "omp")
                force_python = backend_env in ("python", "threadpool", "threads")
                openmp_ok = (
                    not force_python
                    and bool(prefer_openmp)
                    and int(nthreads) > 1
                    and spmm_backend != "csr"
                    and _have_openmp is not None
                    and bool(_have_openmp())
                    and _csc_matmul_dense_sym_inplace_many_indexed_cy is not None
                    and _csc_matmul_dense_sym_add_many_indexed_omp_cy is not None
                )

                if bool(openmp_ok) and np.any(offdiag):
                    _maybe_set_openmp_wait_policy()
                    if profile_out is not None:
                        profile_out["sym_pair_parallel_backend_openmp"] = 1.0

                    off_u = np.asarray(np.nonzero(offdiag)[0], dtype=np.int32)
                    diag_u = np.asarray(np.nonzero(~offdiag)[0], dtype=np.int32)
                    mats_off = []
                    indptr_list = []
                    indices_list = []
                    data_list = []
                    for u in off_u.tolist():
                        mat = sym_mats[int(u)]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix for symmetric fastpath")
                        mats_off.append(mat)
                        indptr_list.append(mat.indptr)
                        indices_list.append(mat.indices)
                        data_list.append(mat.data)

                    t_build0 = time.perf_counter() if profile_out is not None else 0.0
                    if diag_u.size:
                        for u in diag_u.tolist():
                            p = int(p_idx[int(u)])
                            out = t_flat[int(u)].reshape(ncsf, nvec)
                            np.multiply(occ[:, p][:, None], x, out=out)
                    _csc_matmul_dense_sym_inplace_many_indexed_cy(  # type: ignore[misc]
                        indptr_list,
                        indices_list,
                        data_list,
                        x,
                        t_flat,
                        off_u,
                        int(nthreads),
                    )
                    if profile_out is not None:
                        profile_out["sym_t_build_s"] = time.perf_counter() - t_build0

                    if np.any(offdiag):
                        t_h10 = time.perf_counter() if profile_out is not None else 0.0
                        h_coeff = np.zeros(npair, dtype=np.float64)
                        h_coeff[offdiag] = h_eff[p_idx[offdiag], q_idx[offdiag]]
                        if blas_nthreads is None:
                            np.matmul(h_coeff, t_flat, out=h1_flat)
                        else:
                            with blas_thread_limit(int(blas_nthreads)):
                                np.matmul(h_coeff, t_flat, out=h1_flat)
                        y += h1_flat.reshape(ncsf, nvec)
                        if profile_out is not None:
                            profile_out["sym_h1_s"] = time.perf_counter() - t_h10

                    t_eri0 = time.perf_counter() if profile_out is not None else 0.0
                    eri_pair = 0.5 * np.asarray(
                        eri4[p_idx[:, None], q_idx[:, None], p_idx[None, :], q_idx[None, :]],
                        dtype=np.float64,
                        order="C",
                    )
                    if profile_out is not None:
                        profile_out["sym_eri_pair_s"] = time.perf_counter() - t_eri0

                    t_gemm0 = time.perf_counter() if profile_out is not None else 0.0
                    if blas_nthreads is None:
                        np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                    else:
                        with blas_thread_limit(int(blas_nthreads)):
                            np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                    if profile_out is not None:
                        profile_out["sym_gemm_s"] = time.perf_counter() - t_gemm0

                    t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                    # Diagonal pairs (p==q) are simple occupancy scalings.
                    if diag_u.size:
                        for u in diag_u.tolist():
                            p = int(p_idx[int(u)])
                            g = g_flat[int(u)].reshape(ncsf, nvec)
                            y += occ[:, p][:, None] * g

                    # Off-diagonal pairs: accumulate into per-thread buffers to avoid write races.
                    if workspace is None:
                        y_parts = np.empty((int(nthreads), m), dtype=np.float64)
                    else:
                        y_parts = workspace._ensure_sym_parts(nthreads=int(nthreads), m=m)
                    y_parts.fill(0.0)
                    _csc_matmul_dense_sym_add_many_indexed_omp_cy(  # type: ignore[misc]
                        indptr_list,
                        indices_list,
                        data_list,
                        g_flat,
                        y_parts,
                        off_u,
                        int(nthreads),
                    )
                    y_flat = y.reshape(-1)
                    for t in range(int(nthreads)):
                        y_flat += y_parts[t]

                    if profile_out is not None:
                        profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                        profile_out["total_s"] = time.perf_counter() - t_total0
                    return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]

                force_omp_single = nthreads > 1 and _openmp_set_num_threads is not None

                pair_chunks = _split_chunks(npair, min(nthreads, npair))

                def t_worker(start_stop: tuple[int, int]) -> None:
                    _maybe_limit_openmp_in_worker(force_omp_single)
                    start, stop = start_stop
                    for u in range(start, stop):
                        p = int(p_idx[u])
                        q = int(q_idx[u])
                        out = t_flat[u].reshape(ncsf, nvec)
                        if p == q:
                            np.multiply(occ[:, p][:, None], x, out=out)
                            continue
                        mat = sym_mats[u]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix for symmetric fastpath")
                        if spmm_sym_inplace is not None:
                            spmm_sym_inplace(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                        else:
                            if spmm_inplace is not None:
                                spmm_inplace(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                            else:
                                out[:] = mat.dot(x)  # type: ignore[operator]
                            # Add the transpose contribution: out += E_pq.T @ x.
                            if spmm_add_t is not None:
                                spmm_add_t(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                            else:
                                out += mat.T.dot(x)  # type: ignore[operator]

                apply_chunks = _split_chunks(npair, min(nthreads, npair))

                max_workers = max([1, len(pair_chunks), len(apply_chunks)])

                pool = executor
                pool_cm = None
                if max_workers > 1 and pool is None:
                    if workspace is not None:
                        pool = workspace._ensure_executor(max_workers=max_workers)
                    else:
                        pool_cm = ThreadPoolExecutor(max_workers=max_workers)
                        pool = pool_cm.__enter__()

                try:
                    t_build0 = time.perf_counter() if profile_out is not None else 0.0
                    if max_workers <= 1:
                        t_worker(pair_chunks[0])
                    else:
                        if len(pair_chunks) == 1:
                            t_worker(pair_chunks[0])
                        else:
                            with blas_thread_limit(1):
                                list(pool.map(t_worker, pair_chunks))
                    if profile_out is not None:
                        profile_out["sym_t_build_s"] = time.perf_counter() - t_build0

                    if np.any(offdiag):
                        t_h10 = time.perf_counter() if profile_out is not None else 0.0
                        h_coeff = np.zeros(npair, dtype=np.float64)
                        h_coeff[offdiag] = h_eff[p_idx[offdiag], q_idx[offdiag]]
                        if blas_nthreads is None:
                            np.matmul(h_coeff, t_flat, out=h1_flat)
                        else:
                            with blas_thread_limit(int(blas_nthreads)):
                                np.matmul(h_coeff, t_flat, out=h1_flat)
                        y += h1_flat.reshape(ncsf, nvec)
                        if profile_out is not None:
                            profile_out["sym_h1_s"] = time.perf_counter() - t_h10

                    t_eri0 = time.perf_counter() if profile_out is not None else 0.0
                    eri_pair = 0.5 * np.asarray(
                        eri4[p_idx[:, None], q_idx[:, None], p_idx[None, :], q_idx[None, :]],
                        dtype=np.float64,
                        order="C",
                    )
                    if profile_out is not None:
                        profile_out["sym_eri_pair_s"] = time.perf_counter() - t_eri0

                    t_gemm0 = time.perf_counter() if profile_out is not None else 0.0
                    if blas_nthreads is None:
                        np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                    else:
                        with blas_thread_limit(int(blas_nthreads)):
                            np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                    if profile_out is not None:
                        profile_out["sym_gemm_s"] = time.perf_counter() - t_gemm0

                    if max_workers <= 1:
                        t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                        for u in range(npair):
                            p = int(p_idx[u])
                            q = int(q_idx[u])
                            g = g_flat[u].reshape(ncsf, nvec)
                            if p == q:
                                y += occ[:, p][:, None] * g
                                continue
                            mat = sym_mats[u]
                            if mat is None:
                                raise AssertionError("missing E_pq sparse matrix for symmetric fastpath")
                            if spmm_sym_add is not None:
                                spmm_sym_add(mat.indptr, mat.indices, mat.data, g, y)  # type: ignore[attr-defined]
                            else:
                                if spmm_add is not None:
                                    spmm_add(mat.indptr, mat.indices, mat.data, g, y)  # type: ignore[attr-defined]
                                else:
                                    y += mat.dot(g)  # type: ignore[operator]
                                if spmm_add_t is not None:
                                    spmm_add_t(mat.indptr, mat.indices, mat.data, g, y)  # type: ignore[attr-defined]
                                else:
                                    y += mat.T.dot(g)  # type: ignore[operator]
                        if profile_out is not None:
                            profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                            profile_out["total_s"] = time.perf_counter() - t_total0
                        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]

                    def g_worker(start_stop: tuple[int, int]) -> np.ndarray:
                        _maybe_limit_openmp_in_worker(force_omp_single)
                        start, stop = start_stop
                        out = np.zeros((ncsf, nvec), dtype=np.float64)
                        for u in range(start, stop):
                            p = int(p_idx[u])
                            q = int(q_idx[u])
                            g = g_flat[u].reshape(ncsf, nvec)
                            if p == q:
                                out += occ[:, p][:, None] * g
                                continue
                            mat = sym_mats[u]
                            if mat is None:
                                raise AssertionError("missing E_pq sparse matrix for symmetric fastpath")
                            if spmm_sym_add is not None:
                                spmm_sym_add(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                            else:
                                if spmm_add is not None:
                                    spmm_add(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                                else:
                                    out += mat.dot(g)  # type: ignore[operator]
                                if spmm_add_t is not None:
                                    spmm_add_t(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                                else:
                                    out += mat.T.dot(g)  # type: ignore[operator]
                        return out

                    t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                    if len(apply_chunks) == 1:
                        y += g_worker(apply_chunks[0])
                    else:
                        with blas_thread_limit(1):
                            for part in pool.map(g_worker, apply_chunks):
                                y += part

                    if profile_out is not None:
                        profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                        profile_out["total_s"] = time.perf_counter() - t_total0
                    return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]
                finally:
                    if pool_cm is not None:
                        pool_cm.__exit__(None, None, None)
            finally:
                if ws_lock is not None:
                    ws_lock.release()

            raise RuntimeError("internal error: unexpected sym fastpath fallthrough")

        if profile_out is not None:
            profile_out["gen_path_calls"] = 1.0
        if spmm_backend == "csr":
            mats = _epq_spmat_list_csr(drt, cache)
            spmm_inplace = _csr_matmul_dense_inplace_cy
            spmm_add = _csr_matmul_dense_add_cy
        else:
            mats = _epq_spmat_list(drt, cache)
            spmm_inplace = _csc_matmul_dense_inplace_cy
            spmm_add = _csc_matmul_dense_add_cy

        force_omp_single = nthreads > 1 and _openmp_set_num_threads is not None

        nops = norb * norb
        m = ncsf * nvec
        ws_lock = None if workspace is None else workspace.lock
        if ws_lock is not None:
            ws_lock.acquire()
        try:
            if workspace is None:
                t_flat = np.empty((nops, m), dtype=np.float64)
                g_flat = np.empty((nops, m), dtype=np.float64)
            else:
                t_flat, g_flat = workspace._ensure_gen(nops=nops, m=m)

            diag_rs = np.arange(0, nops, norb + 1, dtype=np.int32)
            rs_all = np.arange(nops, dtype=np.int32)
            off_rs = rs_all[rs_all % (norb + 1) != 0]

            # Prefer OpenMP pair-parallel kernels to avoid Python ThreadPoolExecutor overhead
            # and reduce the risk of oversubscription. Can be overridden via env var:
            #   CUGUGA_CONTRACT_PARALLEL_BACKEND=python|openmp|auto
            backend_env = os.environ.get("CUGUGA_CONTRACT_PARALLEL_BACKEND", "").strip().lower()
            prefer_openmp = backend_env in ("", "auto", "openmp", "omp")
            force_python = backend_env in ("python", "threadpool", "threads")
            openmp_ok = (
                not force_python
                and bool(prefer_openmp)
                and int(nthreads) > 1
                and spmm_backend != "csr"
                and _have_openmp is not None
                and bool(_have_openmp())
                and _csc_matmul_dense_inplace_many_indexed_cy is not None
                and _csc_matmul_dense_add_many_indexed_omp_cy is not None
            )

            if bool(openmp_ok) and off_rs.size:
                _maybe_set_openmp_wait_policy()
                if profile_out is not None:
                    profile_out["gen_pair_parallel_backend_openmp"] = 1.0

                indptr_list = []
                indices_list = []
                data_list = []
                for rs in off_rs.tolist():
                    mat = mats[int(rs)]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    indptr_list.append(mat.indptr)
                    indices_list.append(mat.indices)
                    data_list.append(mat.data)

                t_build0 = time.perf_counter() if profile_out is not None else 0.0
                if diag_rs.size:
                    for rs in diag_rs.tolist():
                        r = int(rs) // int(norb)
                        out = t_flat[int(rs)].reshape(ncsf, nvec)
                        np.multiply(occ[:, r][:, None], x, out=out)
                _csc_matmul_dense_inplace_many_indexed_cy(  # type: ignore[misc]
                    indptr_list,
                    indices_list,
                    data_list,
                    x,
                    t_flat,
                    off_rs,
                    int(nthreads),
                )
                if profile_out is not None:
                    profile_out["gen_t_build_s"] = time.perf_counter() - t_build0

                t_h10 = time.perf_counter() if profile_out is not None else 0.0
                h_coeff = np.asarray(h_eff, dtype=np.float64).reshape(-1).copy()
                h_coeff[:: (norb + 1)] = 0.0
                h1_flat = np.empty(m, dtype=np.float64)
                if blas_nthreads is None:
                    np.matmul(h_coeff, t_flat, out=h1_flat)
                else:
                    with blas_thread_limit(int(blas_nthreads)):
                        np.matmul(h_coeff, t_flat, out=h1_flat)
                y += h1_flat.reshape(ncsf, nvec)
                if profile_out is not None:
                    profile_out["gen_h1_s"] = time.perf_counter() - t_h10

                t_gemm0 = time.perf_counter() if profile_out is not None else 0.0
                eri_mat = (0.5 * eri4.reshape(nops, nops)).astype(np.float64, copy=False)
                if blas_nthreads is None:
                    np.matmul(eri_mat, t_flat, out=g_flat)
                else:
                    with blas_thread_limit(int(blas_nthreads)):
                        np.matmul(eri_mat, t_flat, out=g_flat)
                if profile_out is not None:
                    profile_out["gen_gemm_s"] = time.perf_counter() - t_gemm0

                t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                # Diagonal pairs (p==q) are simple occupancy scalings.
                if diag_rs.size:
                    for rs in diag_rs.tolist():
                        p = int(rs) // int(norb)
                        g = g_flat[int(rs)].reshape(ncsf, nvec)
                        y += occ[:, p][:, None] * g

                # Off-diagonal pairs: accumulate into per-thread buffers to avoid write races.
                if workspace is None:
                    y_parts = np.empty((int(nthreads), m), dtype=np.float64)
                else:
                    y_parts = workspace._ensure_sym_parts(nthreads=int(nthreads), m=m)
                y_parts.fill(0.0)
                _csc_matmul_dense_add_many_indexed_omp_cy(  # type: ignore[misc]
                    indptr_list,
                    indices_list,
                    data_list,
                    g_flat,
                    y_parts,
                    off_rs,
                    int(nthreads),
                )
                y_flat = y.reshape(-1)
                for t in range(int(nthreads)):
                    y_flat += y_parts[t]

                if profile_out is not None:
                    profile_out["gen_apply_s"] = time.perf_counter() - t_apply0
                    profile_out["total_s"] = time.perf_counter() - t_total0
                return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]

            rs_ids = list(range(nops))

            def t_worker(start_stop: tuple[int, int]) -> None:
                _maybe_limit_openmp_in_worker(force_omp_single)
                start, stop = start_stop
                for rs in rs_ids[start:stop]:
                    r, s = divmod(int(rs), norb)
                    out = t_flat[rs].reshape(ncsf, nvec)
                    if r == s:
                        np.multiply(occ[:, r][:, None], x, out=out)
                        continue
                    mat = mats[rs]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if spmm_inplace is not None:
                        spmm_inplace(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                    else:
                        out[:] = mat.dot(x)  # type: ignore[operator]

            def g_worker(start_stop: tuple[int, int]) -> np.ndarray:
                _maybe_limit_openmp_in_worker(force_omp_single)
                start, stop = start_stop
                out = np.zeros((ncsf, nvec), dtype=np.float64)
                for pq in range(start, stop):
                    p, q = divmod(int(pq), norb)
                    g = g_flat[pq].reshape(ncsf, nvec)
                    if p == q:
                        out += occ[:, p][:, None] * g
                        continue
                    mat = mats[pq]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if spmm_add is not None:
                        spmm_add(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                    else:
                        out += mat.dot(g)  # type: ignore[operator]
                return out

            rs_chunks = _split_chunks(len(rs_ids), min(nthreads, len(rs_ids)))
            pq_chunks = _split_chunks(nops, min(nthreads, nops))
            max_workers = max([1, len(rs_chunks), len(pq_chunks)])

            if max_workers <= 1:
                t_build0 = time.perf_counter() if profile_out is not None else 0.0
                t_worker(rs_chunks[0])
                if profile_out is not None:
                    profile_out["gen_t_build_s"] = time.perf_counter() - t_build0

                t_h10 = time.perf_counter() if profile_out is not None else 0.0
                h_coeff = np.asarray(h_eff, dtype=np.float64).reshape(-1).copy()
                h_coeff[:: (norb + 1)] = 0.0
                h1_flat = np.empty(m, dtype=np.float64)
                if blas_nthreads is None:
                    np.matmul(h_coeff, t_flat, out=h1_flat)
                else:
                    with blas_thread_limit(int(blas_nthreads)):
                        np.matmul(h_coeff, t_flat, out=h1_flat)
                y += h1_flat.reshape(ncsf, nvec)
                if profile_out is not None:
                    profile_out["gen_h1_s"] = time.perf_counter() - t_h10

                t_gemm0 = time.perf_counter() if profile_out is not None else 0.0
                eri_mat = (0.5 * eri4.reshape(nops, nops)).astype(np.float64, copy=False)
                if blas_nthreads is None:
                    np.matmul(eri_mat, t_flat, out=g_flat)
                else:
                    with blas_thread_limit(int(blas_nthreads)):
                        np.matmul(eri_mat, t_flat, out=g_flat)
                if profile_out is not None:
                    profile_out["gen_gemm_s"] = time.perf_counter() - t_gemm0
                t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                y += g_worker(pq_chunks[0])
                if profile_out is not None:
                    profile_out["gen_apply_s"] = time.perf_counter() - t_apply0
            else:
                pool = executor
                pool_cm = None
                if pool is None:
                    if workspace is not None:
                        pool = workspace._ensure_executor(max_workers=max_workers)
                    else:
                        pool_cm = ThreadPoolExecutor(max_workers=max_workers)
                        pool = pool_cm.__enter__()
                try:
                    t_build0 = time.perf_counter() if profile_out is not None else 0.0
                    if len(rs_chunks) == 1:
                        t_worker(rs_chunks[0])
                    else:
                        with blas_thread_limit(1):
                            list(pool.map(t_worker, rs_chunks))
                    if profile_out is not None:
                        profile_out["gen_t_build_s"] = time.perf_counter() - t_build0

                    t_h10 = time.perf_counter() if profile_out is not None else 0.0
                    h_coeff = np.asarray(h_eff, dtype=np.float64).reshape(-1).copy()
                    h_coeff[:: (norb + 1)] = 0.0
                    h1_flat = np.empty(m, dtype=np.float64)
                    if blas_nthreads is None:
                        np.matmul(h_coeff, t_flat, out=h1_flat)
                    else:
                        with blas_thread_limit(int(blas_nthreads)):
                            np.matmul(h_coeff, t_flat, out=h1_flat)
                    y += h1_flat.reshape(ncsf, nvec)
                    if profile_out is not None:
                        profile_out["gen_h1_s"] = time.perf_counter() - t_h10

                    t_gemm0 = time.perf_counter() if profile_out is not None else 0.0
                    eri_mat = (0.5 * eri4.reshape(nops, nops)).astype(np.float64, copy=False)
                    if blas_nthreads is None:
                        np.matmul(eri_mat, t_flat, out=g_flat)
                    else:
                        with blas_thread_limit(int(blas_nthreads)):
                            np.matmul(eri_mat, t_flat, out=g_flat)
                    if profile_out is not None:
                        profile_out["gen_gemm_s"] = time.perf_counter() - t_gemm0

                    t_apply0 = time.perf_counter() if profile_out is not None else 0.0
                    if len(pq_chunks) == 1:
                        y += g_worker(pq_chunks[0])
                    else:
                        with blas_thread_limit(1):
                            for part in pool.map(g_worker, pq_chunks):
                                y += part
                    if profile_out is not None:
                        profile_out["gen_apply_s"] = time.perf_counter() - t_apply0
                finally:
                    if pool_cm is not None:
                        pool_cm.__exit__(None, None, None)
        finally:
            if ws_lock is not None:
                ws_lock.release()

        if profile_out is not None:
            profile_out["total_s"] = time.perf_counter() - t_total0
        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]
