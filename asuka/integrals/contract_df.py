from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import os
import time
import threading
import weakref

import numpy as np

from asuka.cuguga.blas_threads import blas_thread_limit, openmp_thread_limit
from asuka.contract import ContractWorkspace
from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import (
    _csr_for_epq,
    _get_epq_action_cache,
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
        csc_matmul_dense_add_cy as _csc_matmul_dense_add_cy,
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
        openmp_set_num_threads as _openmp_set_num_threads,
    )
except Exception:  # pragma: no cover
    _csr_matmul_dense_add_cy = None
    _csc_matmul_dense_add_cy = None
    _csc_matmul_dense_inplace_cy = None
    _openmp_set_num_threads = None

try:  # optional OpenMP pair-parallel helpers (avoid Python ThreadPoolExecutor)
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        have_openmp as _have_openmp,
        openmp_max_threads as _openmp_max_threads,
        csc_matmul_dense_sym_inplace_many_indexed_cy as _csc_matmul_dense_sym_inplace_many_indexed_cy,
        csc_matmul_dense_sym_add_many_indexed_omp_cy as _csc_matmul_dense_sym_add_many_indexed_omp_cy,
    )
except Exception:  # pragma: no cover
    _have_openmp = None
    _openmp_max_threads = None
    _csc_matmul_dense_sym_inplace_many_indexed_cy = None
    _csc_matmul_dense_sym_add_many_indexed_omp_cy = None


_THREAD_LOCAL = threading.local()


def _maybe_limit_openmp_in_worker(force: bool) -> None:
    if not force or _openmp_set_num_threads is None:
        return
    if bool(getattr(_THREAD_LOCAL, "openmp_single", False)):
        return
    _openmp_set_num_threads(1)  # type: ignore[misc]
    _THREAD_LOCAL.openmp_single = True

_EPQ_SPMAT_CACHE: weakref.WeakKeyDictionary[DRT, dict[int, object]] = weakref.WeakKeyDictionary()
_EPQ_SYM_SPMAT_CACHE: weakref.WeakKeyDictionary[
    DRT, tuple[np.ndarray, np.ndarray, list[object | None]]
] = weakref.WeakKeyDictionary()


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
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    # Build only the upper-triangular (p<q) pairs needed by the symmetric-pair fastpath.
    # The Cython E_pq builder releases the GIL, so this can parallelize across threads
    # and avoid a large single-thread "cache build" spike.
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




def contract_h_csf_multi_df(
    drt: DRT,
    h1e,
    df_eri: DFMOIntegrals,
    xs: list[np.ndarray] | np.ndarray,
    *,
    precompute_epq: bool = True,
    nthreads: int = 1,
    blas_nthreads: int | None = None,
    executor: ThreadPoolExecutor | None = None,
    workspace: ContractWorkspace | None = None,
    profile_out: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Vectorized DF-backed contraction y = H*x for multiple CSF vectors.

    This mirrors :func:`asuka.contract.contract_h_csf_multi`, but
    replaces dense 4-index ERI usage with a DF/Cholesky-vector representation
    of the 2e integrals:

        (pq|rs) ~= sum_L d[L,pq] d[L,rs]

    Parameters
    ----------
    df_eri:
        DF MO integrals for the active orbital space (norb must match drt.norb).
        This path avoids materializing the dense (norb^4) ERI tensor.

    Notes
    -----
    - Numerical results match the dense path when `eri4 = df_eri.to_eri4()`
      (up to floating point roundoff).
    - This is still a prototype; it can allocate O(norb^2 * ncsf * nvec) for
      intermediate generator applications, matching the dense implementation.
    - If ``blas_nthreads`` is provided, BLAS threads are temporarily limited
      during the dense DF pair-space matmuls; otherwise the BLAS library default
      settings are used.
    - If ``executor`` is provided, it is reused for the internal threaded loops,
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

    if int(df_eri.norb) != norb:
        raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")

    xmat = np.asarray(xs, dtype=np.float64)
    if xmat.ndim == 1:
        xmat = xmat.reshape(1, -1)
    if xmat.shape[1] != ncsf:
        raise ValueError(f"xs has wrong shape: {xmat.shape} (expected (*, {ncsf}))")

    nvec = int(xmat.shape[0])

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    t_total0 = time.perf_counter() if profile_out is not None else None
    if profile_out is not None:
        profile_out.clear()
        profile_out["calls"] = 1.0
        profile_out["df_path_calls"] = 1.0
        profile_out["nvec"] = float(nvec)
        profile_out["nthreads"] = float(nthreads)
        profile_out["blas_nthreads"] = float(0 if blas_nthreads is None else int(blas_nthreads))
        profile_out["have_openmp"] = 1.0 if (_have_openmp is not None and bool(_have_openmp())) else 0.0

    if precompute_epq:
        t0 = time.perf_counter() if profile_out is not None else None
        precompute_epq_actions(drt)
        if profile_out is not None and t0 is not None:
            profile_out["precompute_epq_s"] = time.perf_counter() - t0

    cache = _get_epq_action_cache(drt)

    # Fold the contraction term (-δ_qr E_ps) into an effective 1-body coefficient matrix:
    #   h_eff = h1e - 1/2 Σ_q (p q| q s)
    h_eff = h1e - 0.5 * np.asarray(df_eri.j_ps, dtype=np.float64)

    if _sp is None:
        # Slow fallback (kept for environments without SciPy).
        occ = occ_table(drt).astype(np.float64, copy=False)
        y = np.zeros((nvec, ncsf), dtype=np.float64)

        # One-body diagonal: Σ_p h_eff[p,p] occ_p |j>
        diag_h = np.diag(h_eff).astype(np.float64, copy=False)
        y += (occ @ diag_h)[None, :] * xmat

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
                    y[:, csr.indices[start:end]] += (
                        hpq * csr.data[start:end][None, :] * xmat[:, j][:, None]
                    )

        # Two-body product terms: 1/2 Σ_{pqrs} (pq|rs) E_pq E_rs |x>
        nops = norb * norb

        # Build t_rs = E_rs |x> for all r,s.
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

        # Apply the ERI matrix in pair space via DF:
        #   eri_mat = 1/2 * (pq|rs) = 1/2 * (L_full @ L_full^T)
        # so:
        #   g_flat = eri_mat @ t_flat = 1/2 * L_full @ (L_full^T @ t_flat)
        t_flat = t_rs.reshape(nops, ncsf * nvec)
        tmp = df_eri.l_full.T @ t_flat  # (naux, ncsf*nvec)
        g_flat = 0.5 * (df_eri.l_full @ tmp)  # (nops, ncsf*nvec)
        g_pq = g_flat.reshape(nops, ncsf, nvec)

        # Apply Σ_{pq} g_pq E_pq |.>
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

    t0 = time.perf_counter() if profile_out is not None else None
    if workspace is not None:
        occ = workspace._ensure_occ(steps=cache.steps)
    else:
        step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int8)
        occ = np.asarray(step_to_occ[cache.steps], dtype=np.int8, order="F")
    if profile_out is not None and t0 is not None:
        profile_out["occ_s"] = time.perf_counter() - t0

    x = np.ascontiguousarray(xmat.T)  # (ncsf, nvec)
    y = np.zeros((ncsf, nvec), dtype=np.float64)

    t0 = time.perf_counter() if profile_out is not None else None
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
    if profile_out is not None and t0 is not None:
        profile_out["hdiag_apply_s"] = time.perf_counter() - t0

    # Symmetric-pair fastpath: reduce pair space from norb^2 to norb*(norb+1)/2 by grouping
    # (p,q) and (q,p) together via S_pq = E_pq + E_qp, and using the corresponding unique DF
    # pair vectors (rows of df_eri.l_full are identical for (p,q) and (q,p) when built from s2).
    sym_tol = 1e-12
    if float(np.max(np.abs(h_eff - h_eff.T))) <= sym_tol:
        if profile_out is not None:
            profile_out["sym_path_calls"] = 1.0
        p_idx, q_idx, sym_mats = _epq_sym_spmat_list(drt, cache, nthreads=nthreads)

        # Prefer an explicit DF ERI matrix in pair space when it fits. For typical CAS sizes,
        # this removes the naux-scaling cost of the 2-step DF contraction inside the Davidson loop.
        eri_mat_max_mb = int(os.environ.get("CUGUGA_CONTRACT_DF_ERI_MAT_MAX_MB", "256"))
        t0 = time.perf_counter() if profile_out is not None else None
        eri_mat = df_eri._maybe_build_eri_mat(int(eri_mat_max_mb) * 1024 * 1024)
        if profile_out is not None and t0 is not None:
            profile_out["df_eri_mat_s"] = time.perf_counter() - t0
        if eri_mat is not None:
            if profile_out is not None:
                profile_out["df_eri_mat_used"] = 1.0
            pair_ids = (p_idx * norb + q_idx).astype(np.int32, copy=False)
            npair = int(p_idx.size)
            m = ncsf * nvec

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
                # OpenMP thread settings are per OS thread. Worker threads are configured once
                # via `_maybe_limit_openmp_in_worker`, so avoid toggling OpenMP settings in the
                # main thread here.
                with contextlib.nullcontext():
                    have_omp = _have_openmp is not None and bool(_have_openmp())
                    use_pair_kernels = (
                        bool(int(os.environ.get("CUGUGA_CONTRACT_DF_USE_OMP_PAIRS", "1")))
                        and have_omp
                        and int(nthreads) > 1
                        and _csc_matmul_dense_sym_inplace_many_indexed_cy is not None
                        and _csc_matmul_dense_sym_add_many_indexed_omp_cy is not None
                    )
                    if use_pair_kernels:
                        nthreads_omp = int(nthreads) if have_omp else 1
                        nthreads_apply = int(
                            os.environ.get("CUGUGA_CONTRACT_DF_OMP_APPLY_THREADS", str(nthreads_omp))
                        )
                        if nthreads_apply < 1:
                            nthreads_apply = 1
                        if nthreads_apply > nthreads_omp:
                            nthreads_apply = nthreads_omp

                        if profile_out is not None:
                            profile_out["sym_pair_backend_cython"] = 1.0
                            profile_out["sym_pair_backend_openmp"] = 1.0 if have_omp else 0.0
                            profile_out["sym_pair_t_build_nthreads"] = float(nthreads_omp)
                            profile_out["sym_pair_apply_nthreads"] = float(nthreads_apply)

                        diag_u = np.flatnonzero(p_idx == q_idx).astype(np.int32, copy=False)
                        offdiag_u = np.flatnonzero(p_idx != q_idx).astype(np.int32, copy=False)
                        diag_list = diag_u.tolist()
                        offdiag_list = offdiag_u.tolist()

                        indptr_list: list[np.ndarray] = []
                        indices_list: list[np.ndarray] = []
                        data_list: list[np.ndarray] = []
                        for uu in offdiag_list:
                            mat = sym_mats[int(uu)]
                            if mat is None:
                                raise AssertionError(
                                    "missing E_pq sparse matrix for symmetric fastpath"
                                )
                            indptr_list.append(mat.indptr)
                            indices_list.append(mat.indices)
                            data_list.append(mat.data)

                        t_build0 = time.perf_counter() if profile_out is not None else None
                        for uu in diag_list:
                            p = int(p_idx[int(uu)])
                            out = t_flat[int(uu)].reshape(ncsf, nvec)
                            np.multiply(occ[:, p][:, None], x, out=out)
                        if offdiag_u.size:
                            with openmp_thread_limit(int(nthreads_omp)):
                                _csc_matmul_dense_sym_inplace_many_indexed_cy(
                                    indptr_list,
                                    indices_list,
                                    data_list,
                                    x,
                                    t_flat,
                                    offdiag_u,
                                    nthreads=0,
                                )
                        if profile_out is not None and t_build0 is not None:
                            profile_out["sym_t_build_s"] = time.perf_counter() - t_build0

                        if offdiag_u.size:
                            t_h10 = time.perf_counter() if profile_out is not None else None
                            h_coeff = np.zeros(npair, dtype=np.float64)
                            h_coeff[offdiag_u] = h_eff[p_idx[offdiag_u], q_idx[offdiag_u]]
                            if blas_nthreads is None:
                                np.matmul(h_coeff, t_flat, out=h1_flat)
                            else:
                                with blas_thread_limit(int(blas_nthreads)):
                                    np.matmul(h_coeff, t_flat, out=h1_flat)
                            y += h1_flat.reshape(ncsf, nvec)
                            if profile_out is not None and t_h10 is not None:
                                profile_out["sym_h1_s"] = time.perf_counter() - t_h10

                        t_eri0 = time.perf_counter() if profile_out is not None else None
                        eri_pair = 0.5 * np.asarray(
                            eri_mat[pair_ids[:, None], pair_ids[None, :]],
                            dtype=np.float64,
                            order="C",
                        )
                        if profile_out is not None and t_eri0 is not None:
                            profile_out["sym_eri_pair_s"] = time.perf_counter() - t_eri0
                        t_gemm0 = time.perf_counter() if profile_out is not None else None
                        if blas_nthreads is None:
                            np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        else:
                            with blas_thread_limit(int(blas_nthreads)):
                                np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        if profile_out is not None and t_gemm0 is not None:
                            profile_out["sym_gemm_s"] = time.perf_counter() - t_gemm0

                        t_apply0 = time.perf_counter() if profile_out is not None else None
                        for uu in diag_list:
                            p = int(p_idx[int(uu)])
                            g = g_flat[int(uu)].reshape(ncsf, nvec)
                            y += occ[:, p][:, None] * g
                        if offdiag_u.size:
                            if workspace is None:
                                y_parts = np.empty((int(nthreads_apply), m), dtype=np.float64)
                            else:
                                y_parts = workspace._ensure_sym_parts(nthreads=int(nthreads_apply), m=m)
                            y_parts[:] = 0.0
                            with openmp_thread_limit(int(nthreads_apply)):
                                _csc_matmul_dense_sym_add_many_indexed_omp_cy(
                                    indptr_list,
                                    indices_list,
                                    data_list,
                                    g_flat,
                                    y_parts,
                                    offdiag_u,
                                    nthreads=0,
                                )
                            y_flat = y.reshape(m)
                            for tid in range(int(nthreads_apply)):
                                y_flat += y_parts[tid]
                        if profile_out is not None and t_apply0 is not None:
                            profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                            if t_total0 is not None:
                                profile_out["total_s"] = time.perf_counter() - t_total0

                        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]

                    pair_chunks = _split_chunks(npair, min(nthreads, npair))
                    apply_chunks = _split_chunks(npair, min(nthreads, npair))
                    max_workers = max([1, len(pair_chunks), len(apply_chunks)])

                    spmm_inplace = _csc_matmul_dense_inplace_cy
                    spmm_add = _csc_matmul_dense_add_cy
                    spmm_add_t = _csr_matmul_dense_add_cy
                    force_omp_single = nthreads > 1 and _openmp_set_num_threads is not None

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
                            if spmm_inplace is not None:
                                spmm_inplace(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                            else:
                                out[:] = mat.dot(x)  # type: ignore[operator]
                            # Add the transpose contribution: out += E_pq.T @ x.
                            if spmm_add_t is not None:
                                spmm_add_t(mat.indptr, mat.indices, mat.data, x, out)  # type: ignore[attr-defined]
                            else:
                                out += mat.T.dot(x)  # type: ignore[operator]

                    if max_workers <= 1:
                        t_build0 = time.perf_counter() if profile_out is not None else None
                        t_worker(pair_chunks[0])
                        if profile_out is not None and t_build0 is not None:
                            profile_out["sym_t_build_s"] = time.perf_counter() - t_build0

                        offdiag = p_idx != q_idx
                        if np.any(offdiag):
                            t_h10 = time.perf_counter() if profile_out is not None else None
                            h_coeff = np.zeros(npair, dtype=np.float64)
                            h_coeff[offdiag] = h_eff[p_idx[offdiag], q_idx[offdiag]]
                            if blas_nthreads is None:
                                np.matmul(h_coeff, t_flat, out=h1_flat)
                            else:
                                with blas_thread_limit(int(blas_nthreads)):
                                    np.matmul(h_coeff, t_flat, out=h1_flat)
                            y += h1_flat.reshape(ncsf, nvec)
                            if profile_out is not None and t_h10 is not None:
                                profile_out["sym_h1_s"] = time.perf_counter() - t_h10

                        t_eri0 = time.perf_counter() if profile_out is not None else None
                        eri_pair = 0.5 * np.asarray(
                            eri_mat[pair_ids[:, None], pair_ids[None, :]],
                            dtype=np.float64,
                            order="C",
                        )
                        if profile_out is not None and t_eri0 is not None:
                            profile_out["sym_eri_pair_s"] = time.perf_counter() - t_eri0
                        t_gemm0 = time.perf_counter() if profile_out is not None else None
                        if blas_nthreads is None:
                            np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        else:
                            with blas_thread_limit(int(blas_nthreads)):
                                np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        if profile_out is not None and t_gemm0 is not None:
                            profile_out["sym_gemm_s"] = time.perf_counter() - t_gemm0

                        t_apply0 = time.perf_counter() if profile_out is not None else None
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
                            if spmm_add is not None:
                                spmm_add(mat.indptr, mat.indices, mat.data, g, y)  # type: ignore[attr-defined]
                            else:
                                y += mat.dot(g)  # type: ignore[operator]
                            if spmm_add_t is not None:
                                spmm_add_t(mat.indptr, mat.indices, mat.data, g, y)  # type: ignore[attr-defined]
                            else:
                                y += mat.T.dot(g)  # type: ignore[operator]

                        if profile_out is not None and t_apply0 is not None:
                            profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                            if t_total0 is not None:
                                profile_out["total_s"] = time.perf_counter() - t_total0
                        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]

                    pool = executor
                    pool_cm = None
                    if pool is None:
                        if workspace is not None:
                            pool = workspace._ensure_executor(max_workers=max_workers)
                        else:
                            pool_cm = ThreadPoolExecutor(max_workers=max_workers)
                            pool = pool_cm.__enter__()
                    try:
                        t_build0 = time.perf_counter() if profile_out is not None else None
                        if len(pair_chunks) == 1:
                            t_worker(pair_chunks[0])
                        else:
                            with blas_thread_limit(1):
                                list(pool.map(t_worker, pair_chunks))
                        if profile_out is not None and t_build0 is not None:
                            profile_out["sym_t_build_s"] = time.perf_counter() - t_build0

                        offdiag = p_idx != q_idx
                        if np.any(offdiag):
                            t_h10 = time.perf_counter() if profile_out is not None else None
                            h_coeff = np.zeros(npair, dtype=np.float64)
                            h_coeff[offdiag] = h_eff[p_idx[offdiag], q_idx[offdiag]]
                            if blas_nthreads is None:
                                np.matmul(h_coeff, t_flat, out=h1_flat)
                            else:
                                with blas_thread_limit(int(blas_nthreads)):
                                    np.matmul(h_coeff, t_flat, out=h1_flat)
                            y += h1_flat.reshape(ncsf, nvec)
                            if profile_out is not None and t_h10 is not None:
                                profile_out["sym_h1_s"] = time.perf_counter() - t_h10

                        t_eri0 = time.perf_counter() if profile_out is not None else None
                        eri_pair = 0.5 * np.asarray(
                            eri_mat[pair_ids[:, None], pair_ids[None, :]],
                            dtype=np.float64,
                            order="C",
                        )
                        if profile_out is not None and t_eri0 is not None:
                            profile_out["sym_eri_pair_s"] = time.perf_counter() - t_eri0
                        t_gemm0 = time.perf_counter() if profile_out is not None else None
                        if blas_nthreads is None:
                            np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        else:
                            with blas_thread_limit(int(blas_nthreads)):
                                np.matmul(eri_pair.reshape(npair, npair), t_flat, out=g_flat)
                        if profile_out is not None and t_gemm0 is not None:
                            profile_out["sym_gemm_s"] = time.perf_counter() - t_gemm0

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
                                if spmm_add is not None:
                                    spmm_add(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                                else:
                                    out += mat.dot(g)  # type: ignore[operator]
                                if spmm_add_t is not None:
                                    spmm_add_t(mat.indptr, mat.indices, mat.data, g, out)  # type: ignore[attr-defined]
                                else:
                                    out += mat.T.dot(g)  # type: ignore[operator]
                            return out

                        t_apply0 = time.perf_counter() if profile_out is not None else None
                        if len(apply_chunks) == 1:
                            y += g_worker(apply_chunks[0])
                        else:
                            with blas_thread_limit(1):
                                for part in pool.map(g_worker, apply_chunks):
                                    y += part
                        if profile_out is not None and t_apply0 is not None:
                            profile_out["sym_apply_s"] = time.perf_counter() - t_apply0
                            if t_total0 is not None:
                                profile_out["total_s"] = time.perf_counter() - t_total0

                        return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]
                    finally:
                        if pool_cm is not None:
                            pool_cm.__exit__(None, None, None)
            finally:
                if ws_lock is not None:
                    ws_lock.release()

    if profile_out is not None:
        profile_out["gen_path_calls"] = 1.0
    t0 = time.perf_counter() if profile_out is not None else None
    mats = _epq_spmat_list(drt, cache)
    if profile_out is not None and t0 is not None:
        profile_out["gen_spmat_s"] = time.perf_counter() - t0

    h1_pairs: list[tuple[int, float]] = []
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            hpq = float(h_eff[p, q])
            if hpq != 0.0:
                h1_pairs.append((p * norb + q, hpq))

    def h1_worker(start_stop: tuple[int, int]) -> np.ndarray:
        start, stop = start_stop
        out = np.zeros((ncsf, nvec), dtype=np.float64)
        for pq, hpq in h1_pairs[start:stop]:
            mat = mats[pq]
            if mat is None:
                continue
            out += float(hpq) * mat.dot(x)  # type: ignore[operator]
        return out

    nops = norb * norb
    m = ncsf * nvec
    t_flat = np.empty((nops, m), dtype=np.float64)
    rs_ids = list(range(nops))

    def t_worker(start_stop: tuple[int, int]) -> None:
        start, stop = start_stop
        for rs in rs_ids[start:stop]:
            r, s = divmod(int(rs), norb)
            if r == s:
                t_flat[rs] = (occ[:, r][:, None] * x).reshape(m)
                continue
            mat = mats[rs]
            if mat is None:
                raise AssertionError("missing E_pq sparse matrix")
            t_flat[rs] = mat.dot(x).reshape(m)  # type: ignore[operator]

    def g_worker(start_stop: tuple[int, int]) -> np.ndarray:
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
            out += mat.dot(g)  # type: ignore[operator]
        return out

    h1_chunks = _split_chunks(len(h1_pairs), min(nthreads, len(h1_pairs))) if h1_pairs else []
    rs_chunks = _split_chunks(len(rs_ids), min(nthreads, len(rs_ids)))
    pq_chunks = _split_chunks(nops, min(nthreads, nops))
    max_workers = max([1, len(h1_chunks), len(rs_chunks), len(pq_chunks)])

    if max_workers <= 1:
        if h1_pairs:
            t_h10 = time.perf_counter() if profile_out is not None else None
            y += h1_worker(h1_chunks[0])
            if profile_out is not None and t_h10 is not None:
                profile_out["gen_h1_s"] = time.perf_counter() - t_h10
        t_build0 = time.perf_counter() if profile_out is not None else None
        t_worker(rs_chunks[0])
        if profile_out is not None and t_build0 is not None:
            profile_out["gen_t_build_s"] = time.perf_counter() - t_build0
        t_gemm0 = time.perf_counter() if profile_out is not None else None
        if blas_nthreads is None:
            tmp = df_eri.l_full.T @ t_flat  # (naux, ncsf*nvec)
            g_flat = 0.5 * (df_eri.l_full @ tmp)  # (nops, ncsf*nvec)
        else:
            with blas_thread_limit(int(blas_nthreads)):
                tmp = df_eri.l_full.T @ t_flat  # (naux, ncsf*nvec)
                g_flat = 0.5 * (df_eri.l_full @ tmp)  # (nops, ncsf*nvec)
        if profile_out is not None and t_gemm0 is not None:
            profile_out["gen_gemm_s"] = time.perf_counter() - t_gemm0
        t_apply0 = time.perf_counter() if profile_out is not None else None
        y += g_worker(pq_chunks[0])
        if profile_out is not None and t_apply0 is not None:
            profile_out["gen_apply_s"] = time.perf_counter() - t_apply0
            if t_total0 is not None:
                profile_out["total_s"] = time.perf_counter() - t_total0
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
            if h1_pairs:
                t_h10 = time.perf_counter() if profile_out is not None else None
                if len(h1_chunks) == 1:
                    y += h1_worker(h1_chunks[0])
                else:
                    with blas_thread_limit(1):
                        for part in pool.map(h1_worker, h1_chunks):
                            y += part
                if profile_out is not None and t_h10 is not None:
                    profile_out["gen_h1_s"] = time.perf_counter() - t_h10

            t_build0 = time.perf_counter() if profile_out is not None else None
            if len(rs_chunks) == 1:
                t_worker(rs_chunks[0])
            else:
                with blas_thread_limit(1):
                    list(pool.map(t_worker, rs_chunks))
            if profile_out is not None and t_build0 is not None:
                profile_out["gen_t_build_s"] = time.perf_counter() - t_build0

            t_gemm0 = time.perf_counter() if profile_out is not None else None
            if blas_nthreads is None:
                tmp = df_eri.l_full.T @ t_flat  # (naux, ncsf*nvec)
                g_flat = 0.5 * (df_eri.l_full @ tmp)  # (nops, ncsf*nvec)
            else:
                with blas_thread_limit(int(blas_nthreads)):
                    tmp = df_eri.l_full.T @ t_flat  # (naux, ncsf*nvec)
                    g_flat = 0.5 * (df_eri.l_full @ tmp)  # (nops, ncsf*nvec)
            if profile_out is not None and t_gemm0 is not None:
                profile_out["gen_gemm_s"] = time.perf_counter() - t_gemm0

            t_apply0 = time.perf_counter() if profile_out is not None else None
            if len(pq_chunks) == 1:
                y += g_worker(pq_chunks[0])
            else:
                with blas_thread_limit(1):
                    for part in pool.map(g_worker, pq_chunks):
                        y += part
            if profile_out is not None and t_apply0 is not None:
                profile_out["gen_apply_s"] = time.perf_counter() - t_apply0
                if t_total0 is not None:
                    profile_out["total_s"] = time.perf_counter() - t_total0
        finally:
            if pool_cm is not None:
                pool_cm.__exit__(None, None, None)

    return [np.ascontiguousarray(y[:, i]) for i in range(nvec)]
