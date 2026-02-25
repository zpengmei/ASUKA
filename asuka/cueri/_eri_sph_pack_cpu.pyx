# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""CPU reference packers for spherical ERI tiles.

These routines are primarily intended for testing/validation of the CUDA
scatter kernels and for small systems where a CPU fallback is acceptable.

They assume the input tiles are already in spherical AO order per shell.
"""

from libc.stdint cimport int64_t

import numpy as np
cimport numpy as np


cdef inline int64_t pair_index(int i, int j) nogil:
    cdef int hi = i
    cdef int lo = j
    if lo > hi:
        hi = j
        lo = i
    return (<int64_t>hi * <int64_t>(hi + 1)) // 2 + <int64_t>lo


cdef inline int64_t pairpair_index(int64_t p, int64_t q) nogil:
    cdef int64_t hi = p
    cdef int64_t lo = q
    if lo > hi:
        hi = q
        lo = p
    return (hi * (hi + 1)) // 2 + lo


def scatter_eri_tiles_sph_s8_cpu(
    np.ndarray[np.int32_t, ndim=1] task_spAB,
    np.ndarray[np.int32_t, ndim=1] task_spCD,
    np.ndarray[np.int32_t, ndim=1] sp_A,
    np.ndarray[np.int32_t, ndim=1] sp_B,
    np.ndarray[np.int32_t, ndim=1] shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    np.ndarray[np.float64_t, ndim=3] tile_vals,
    np.ndarray[np.float64_t, ndim=1] out_s8,
):
    """Scatter spherical tiles into packed s8 vector on CPU."""

    cdef Py_ssize_t ntasks = task_spAB.shape[0]
    if task_spCD.shape[0] != ntasks:
        raise ValueError("task_spAB/task_spCD must have the same length")
    if tile_vals.shape[0] != ntasks:
        raise ValueError("tile_vals first axis must be ntasks")
    if tile_vals.shape[1] != nA * nB or tile_vals.shape[2] != nC * nD:
        raise ValueError("tile_vals has incompatible shape")

    cdef int64_t nao_pair = (<int64_t>nao_sph * <int64_t>(nao_sph + 1)) // 2
    cdef int64_t need_out = (nao_pair * (nao_pair + 1)) // 2
    if out_s8.shape[0] != need_out:
        raise ValueError("out_s8 has incompatible length")

    cdef Py_ssize_t t
    cdef int spab, spcd, A, B, C, D
    cdef int ia, ib, ic, id
    cdef int a, b, c, d
    cdef int iab, icd
    cdef int64_t p, q, out_idx
    cdef double v

    for t in range(ntasks):
        spab = <int>task_spAB[t]
        spcd = <int>task_spCD[t]
        A = <int>sp_A[spab]
        B = <int>sp_B[spab]
        C = <int>sp_A[spcd]
        D = <int>sp_B[spcd]

        for iab in range(nA * nB):
            ia = iab // nB
            ib = iab - ia * nB
            if A == B and ia < ib:
                continue
            a = <int>shell_ao_start_sph[A] + ia
            b = <int>shell_ao_start_sph[B] + ib
            p = pair_index(a, b)

            for icd in range(nC * nD):
                ic = icd // nD
                id = icd - ic * nD
                if C == D and ic < id:
                    continue
                c = <int>shell_ao_start_sph[C] + ic
                d = <int>shell_ao_start_sph[D] + id
                q = pair_index(c, d)

                if spab == spcd and p < q:
                    continue

                v = tile_vals[t, iab, icd]
                if v == 0.0:
                    continue
                out_idx = pairpair_index(p, q)
                out_s8[out_idx] = v

    return out_s8


def scatter_eri_tiles_sph_s4_cpu(
    np.ndarray[np.int32_t, ndim=1] task_spAB,
    np.ndarray[np.int32_t, ndim=1] task_spCD,
    np.ndarray[np.int32_t, ndim=1] sp_A,
    np.ndarray[np.int32_t, ndim=1] sp_B,
    np.ndarray[np.int32_t, ndim=1] shell_ao_start_sph,
    int nao_sph,
    int nA,
    int nB,
    int nC,
    int nD,
    np.ndarray[np.float64_t, ndim=3] tile_vals,
    np.ndarray[np.float64_t, ndim=2] out_s4,
):
    """Scatter spherical tiles into packed s4 (nao_pair x nao_pair) matrix on CPU."""

    cdef Py_ssize_t ntasks = task_spAB.shape[0]
    if task_spCD.shape[0] != ntasks:
        raise ValueError("task_spAB/task_spCD must have the same length")
    if tile_vals.shape[0] != ntasks:
        raise ValueError("tile_vals first axis must be ntasks")
    if tile_vals.shape[1] != nA * nB or tile_vals.shape[2] != nC * nD:
        raise ValueError("tile_vals has incompatible shape")

    cdef int64_t nao_pair = (<int64_t>nao_sph * <int64_t>(nao_sph + 1)) // 2
    if out_s4.shape[0] != nao_pair or out_s4.shape[1] != nao_pair:
        raise ValueError("out_s4 has incompatible shape")

    cdef Py_ssize_t t
    cdef int spab, spcd, A, B, C, D
    cdef int ia, ib, ic, id
    cdef int a, b, c, d
    cdef int iab, icd
    cdef int64_t p, q
    cdef double v

    for t in range(ntasks):
        spab = <int>task_spAB[t]
        spcd = <int>task_spCD[t]
        A = <int>sp_A[spab]
        B = <int>sp_B[spab]
        C = <int>sp_A[spcd]
        D = <int>sp_B[spcd]

        for iab in range(nA * nB):
            ia = iab // nB
            ib = iab - ia * nB
            if A == B and ia < ib:
                continue
            a = <int>shell_ao_start_sph[A] + ia
            b = <int>shell_ao_start_sph[B] + ib
            p = pair_index(a, b)

            for icd in range(nC * nD):
                ic = icd // nD
                id = icd - ic * nD
                if C == D and ic < id:
                    continue
                c = <int>shell_ao_start_sph[C] + ic
                d = <int>shell_ao_start_sph[D] + id
                q = pair_index(c, d)

                if spab == spcd and p < q:
                    continue

                v = tile_vals[t, iab, icd]
                if v == 0.0:
                    continue
                out_s4[p, q] = v
                out_s4[q, p] = v

    return out_s4


__all__ = [
    "scatter_eri_tiles_sph_s8_cpu",
    "scatter_eri_tiles_sph_s4_cpu",
]
