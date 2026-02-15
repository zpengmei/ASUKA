# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from __future__ import annotations

import numpy as np

cimport numpy as cnp

from cython.parallel cimport prange
from libc.stdint cimport int32_t


cdef extern from *:
    """
    #ifdef CUERI_USE_OPENMP
    static inline int cueri_pair_coeff_openmp_enabled(void) { return 1; }
    #else
    static inline int cueri_pair_coeff_openmp_enabled(void) { return 0; }
    #endif
    """
    int cueri_pair_coeff_openmp_enabled() noexcept nogil


def openmp_enabled() -> bool:
    return bool(cueri_pair_coeff_openmp_enabled())


cdef inline void _build_pair_coeff_single_row(
    const double* CA_data,
    const double* CB_data,
    const int32_t* tri_p_data,
    const int32_t* tri_q_data,
    double* out_data,
    int nB,
    int norb,
    int npair,
    bint same_shell,
    int row,
) noexcept nogil:
    cdef int ia = row // nB
    cdef int ib = row - ia * nB
    cdef const double* CA_row = CA_data + ia * norb
    cdef const double* CB_row = CB_data + ib * norb
    cdef double* out_row = out_data + row * npair

    cdef int pair, p, q
    cdef double v
    for pair in range(npair):
        p = <int>tri_p_data[pair]
        q = <int>tri_q_data[pair]
        v = CA_row[p] * CB_row[q]
        if not same_shell:
            v += CA_row[q] * CB_row[p]
        out_row[pair] = v


cdef inline void _build_pair_coeff_batch_one(
    const double* C_data,
    const int32_t* shell_ao_start,
    const int32_t* shellA,
    const int32_t* shellB,
    const int32_t* tri_p_data,
    const int32_t* tri_q_data,
    double* out_data,
    int nA,
    int nB,
    int norb,
    int npair,
    int t,
) noexcept nogil:
    cdef int a0 = <int>shell_ao_start[<int>shellA[t]]
    cdef int b0 = <int>shell_ao_start[<int>shellB[t]]
    cdef bint same_shell = (<bint>(shellA[t] == shellB[t]))

    cdef double* out_t = out_data + t * (nA * nB * npair)

    cdef int ia, ib, row, pair, p, q
    cdef const double* CA_row
    cdef const double* CB_row
    cdef double v

    for ia in range(nA):
        CA_row = C_data + (a0 + ia) * norb
        for ib in range(nB):
            CB_row = C_data + (b0 + ib) * norb
            row = ia * nB + ib
            for pair in range(npair):
                p = <int>tri_p_data[pair]
                q = <int>tri_q_data[pair]
                v = CA_row[p] * CB_row[q]
                if not same_shell:
                    v += CA_row[q] * CB_row[p]
                out_t[row * npair + pair] = v


def build_pair_coeff_packed_tri_cy(
    CA: np.ndarray,
    CB: np.ndarray,
    tri_p: np.ndarray,
    tri_q: np.ndarray,
    *,
    same_shell: bool,
    threads: int = 0,
) -> np.ndarray:
    """Build packed-pair coefficients K[(μν),pq] for a single shell pair (A,B).

    Parameters
    ----------
    CA, CB
        AO->active coefficient blocks for shells A and B, shapes (nA,norb) and (nB,norb).
    tri_p, tri_q
        Packed-pair indices from `np.tril_indices(norb)` (converted to int32 internally).
    same_shell
        Whether A==B (packed symmetry, omit the swap term).
    threads
        OpenMP threads hint (if built with CUERI_USE_OPENMP=1).
    """

    cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] CA_c = np.asarray(CA, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] CB_c = np.asarray(CB, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] tri_p_c = np.asarray(tri_p, dtype=np.int32, order="C").ravel()
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] tri_q_c = np.asarray(tri_q, dtype=np.int32, order="C").ravel()

    if CA_c.ndim != 2 or CB_c.ndim != 2:
        raise ValueError("CA/CB must be 2D arrays")
    if tri_p_c.ndim != 1 or tri_q_c.ndim != 1 or tri_p_c.shape[0] != tri_q_c.shape[0]:
        raise ValueError("tri_p/tri_q must be 1D arrays with identical shape")

    cdef int nA = <int>CA_c.shape[0]
    cdef int nB = <int>CB_c.shape[0]
    cdef int norb = <int>CA_c.shape[1]
    if <int>CB_c.shape[1] != norb:
        raise ValueError("CA/CB must have the same norb dimension")
    cdef int npair = <int>tri_p_c.shape[0]

    cdef int threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.empty((nA * nB, npair), dtype=np.float64)

    cdef const double* CA_data = <const double*>CA_c.data
    cdef const double* CB_data = <const double*>CB_c.data
    cdef const int32_t* tri_p_data = <const int32_t*>tri_p_c.data
    cdef const int32_t* tri_q_data = <const int32_t*>tri_q_c.data
    cdef double* out_data = <double*>out.data

    cdef bint same_shell_i = <bint>same_shell
    cdef int row

    cdef int nrow = nA * nB
    cdef long long work = <long long>nrow * <long long>npair
    cdef int use_threads = threads_i
    if use_threads > nrow:
        use_threads = nrow
    if use_threads <= 1 or work < (1 << 16):
        use_threads = 1

    with nogil:
        if use_threads > 1:
            for row in prange(nrow, schedule="static", num_threads=use_threads):
                _build_pair_coeff_single_row(
                    CA_data,
                    CB_data,
                    tri_p_data,
                    tri_q_data,
                    out_data,
                    nB,
                    norb,
                    npair,
                    same_shell_i,
                    row,
                )
        else:
            for row in range(nrow):
                _build_pair_coeff_single_row(
                    CA_data,
                    CB_data,
                    tri_p_data,
                    tri_q_data,
                    out_data,
                    nB,
                    norb,
                    npair,
                    same_shell_i,
                    row,
                )

    return out


def build_pair_coeff_packed_batch_tri_cy(
    C_active: np.ndarray,
    shell_ao_start: np.ndarray,
    shellA: np.ndarray,
    shellB: np.ndarray,
    nA: int,
    nB: int,
    tri_p: np.ndarray,
    tri_q: np.ndarray,
    *,
    threads: int = 0,
) -> np.ndarray:
    """Build packed-pair coefficients for a batch of shell pairs.

    Returns `K[t,(μν),pq]` with shape `(nt, nA*nB, npair)` for tasks (shellA[t], shellB[t]).
    """

    cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] C_c = np.asarray(C_active, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sh_start_c = np.asarray(shell_ao_start, dtype=np.int32, order="C").ravel()
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shA_c = np.asarray(shellA, dtype=np.int32, order="C").ravel()
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shB_c = np.asarray(shellB, dtype=np.int32, order="C").ravel()
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] tri_p_c = np.asarray(tri_p, dtype=np.int32, order="C").ravel()
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] tri_q_c = np.asarray(tri_q, dtype=np.int32, order="C").ravel()

    if C_c.ndim != 2:
        raise ValueError("C_active must be a 2D array with shape (nao, norb)")
    if shA_c.shape[0] != shB_c.shape[0]:
        raise ValueError("shellA/shellB must have identical shape")
    if tri_p_c.shape[0] != tri_q_c.shape[0]:
        raise ValueError("tri_p/tri_q must have identical shape")

    cdef int nt = <int>shA_c.shape[0]
    cdef int nA_i = int(nA)
    cdef int nB_i = int(nB)
    if nA_i < 0 or nB_i < 0:
        raise ValueError("nA/nB must be >= 0")
    if nt == 0:
        return np.empty((0, nA_i * nB_i, int(tri_p_c.shape[0])), dtype=np.float64)

    cdef int norb = <int>C_c.shape[1]
    cdef int npair = <int>tri_p_c.shape[0]

    cdef int threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.empty((nt, nA_i * nB_i, npair), dtype=np.float64)

    cdef const double* C_data = <const double*>C_c.data
    cdef const int32_t* sh_start = <const int32_t*>sh_start_c.data
    cdef const int32_t* shA = <const int32_t*>shA_c.data
    cdef const int32_t* shB = <const int32_t*>shB_c.data
    cdef const int32_t* tri_p_data = <const int32_t*>tri_p_c.data
    cdef const int32_t* tri_q_data = <const int32_t*>tri_q_c.data
    cdef double* out_data = <double*>out.data

    cdef int t
    cdef long long work = <long long>nt * <long long>(nA_i * nB_i) * <long long>npair
    cdef int use_threads = threads_i
    if use_threads > nt:
        use_threads = nt
    if use_threads <= 1 or nt <= 1 or work < (1 << 16):
        use_threads = 1

    # Memory layout: out[t,row,pair] contiguous in `pair`.
    with nogil:
        if use_threads > 1:
            for t in prange(nt, schedule="static", num_threads=use_threads):
                _build_pair_coeff_batch_one(
                    C_data,
                    sh_start,
                    shA,
                    shB,
                    tri_p_data,
                    tri_q_data,
                    out_data,
                    nA_i,
                    nB_i,
                    norb,
                    npair,
                    t,
                )
        else:
            for t in range(nt):
                _build_pair_coeff_batch_one(
                    C_data,
                    sh_start,
                    shA,
                    shB,
                    tri_p_data,
                    tri_q_data,
                    out_data,
                    nA_i,
                    nB_i,
                    norb,
                    npair,
                    t,
                )

    return out


__all__ = [
    "build_pair_coeff_packed_batch_tri_cy",
    "build_pair_coeff_packed_tri_cy",
    "openmp_enabled",
]
