# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from __future__ import annotations

import numpy as np

cimport numpy as cnp

from libc.math cimport erf, exp, sqrt
from libc.stdlib cimport free, malloc


cdef double _PI = 3.141592653589793238462643383279502884


cdef inline int _ncart(int l) noexcept nogil:
    return (l + 1) * (l + 2) // 2


cdef inline double _boys_f0(double T) noexcept nogil:
    if T < 1e-12:
        return 1.0 - (T / 3.0) + (T * T / 10.0)
    cdef double sT = sqrt(T)
    return 0.5 * sqrt(_PI / T) * erf(sT)


cdef inline void _boys_fm_list(double T, int m_max, double* out) noexcept nogil:
    cdef int m, k
    cdef double term, Fm, e

    if m_max <= 0:
        out[0] = _boys_f0(T)
        return

    if T < 5.0:
        term = 1.0
        Fm = 0.0
        for k in range(120):
            Fm += term / <double>(2 * m_max + 2 * k + 1)
            term *= -T / <double>(k + 1)
        out[m_max] = Fm
        e = exp(-T)
        for m in range(m_max, 0, -1):
            out[m - 1] = (2.0 * T * out[m] + e) / <double>(2 * m - 1)
        return

    out[0] = _boys_f0(T)
    e = exp(-T)
    for m in range(1, m_max + 1):
        out[m] = (<double>(2 * m - 1) * out[m - 1] - e) / (2.0 * T)


cdef inline void _overlap_1d_table(
    int la,
    int lb,
    double a,
    double b,
    double Ax,
    double Bx,
    double* out,
    int stride,
) noexcept nogil:
    cdef int i, j
    cdef double p = a + b
    cdef double inv_p = 1.0 / p
    cdef double mu = a * b * inv_p
    cdef double Px = (a * Ax + b * Bx) * inv_p
    cdef double PA = Px - Ax
    cdef double PB = Px - Bx
    cdef double AB = Ax - Bx
    cdef double s00 = sqrt(_PI * inv_p) * exp(-mu * AB * AB)
    cdef double inv_2p = 0.5 * inv_p

    out[0] = s00

    # j recursion at i=0
    for j in range(0, lb):
        out[j + 1] = PB * out[j]
        if j > 0:
            out[j + 1] += <double>j * inv_2p * out[j - 1]

    # i recursion
    for i in range(0, la):
        out[(i + 1) * stride] = PA * out[i * stride]
        if i > 0:
            out[(i + 1) * stride] += <double>i * inv_2p * out[(i - 1) * stride]
        for j in range(0, lb):
            out[(i + 1) * stride + (j + 1)] = PA * out[i * stride + (j + 1)]
            if i > 0:
                out[(i + 1) * stride + (j + 1)] += <double>i * inv_2p * out[(i - 1) * stride + (j + 1)]
            out[(i + 1) * stride + (j + 1)] += <double>(j + 1) * inv_2p * out[i * stride + j]


cdef inline double _kin1d_from_overlap(const double* S, int stride, int i, int j, double b) noexcept nogil:
    cdef double val = b * (2.0 * <double>j + 1.0) * S[i * stride + j] - 2.0 * b * b * S[i * stride + (j + 2)]
    if j >= 2:
        val -= 0.5 * <double>(j * (j - 1)) * S[i * stride + (j - 2)]
    return val


cdef inline void _hermite_E_1d_table(
    int la,
    int lb,
    double a,
    double b,
    double Ax,
    double Bx,
    double* E,
    int ij_stride,
    int t_stride,
) noexcept nogil:
    cdef int i, j, t
    cdef int tmax_ij
    cdef int tmax
    cdef double* prev
    cdef double* cur
    cdef double val
    cdef double p = a + b
    cdef double inv_p = 1.0 / p
    cdef double mu = a * b * inv_p
    cdef double Px = (a * Ax + b * Bx) * inv_p
    cdef double PA = Px - Ax
    cdef double PB = Px - Bx
    cdef double AB = Ax - Bx
    cdef double inv_2p = 0.5 * inv_p

    # Zero required region; recurrence relies on unset values being 0.
    tmax = la + lb
    for i in range(0, la + 1):
        for j in range(0, lb + 1):
            cur = E + (i * ij_stride + j) * t_stride
            for t in range(0, tmax + 1):
                cur[t] = 0.0

    E[0] = exp(-mu * AB * AB)  # E[0,0,0]

    # Build i for j=0
    for i in range(0, la):
        prev = E + (i * ij_stride + 0) * t_stride
        cur = E + ((i + 1) * ij_stride + 0) * t_stride
        # cur[t] = PA*prev[t] + inv2p*prev[t-1] + (t+1)*prev[t+1]
        for t in range(0, i + 2):  # t <= (i+1)+0
            val = PA * prev[t]
            if t > 0:
                val += inv_2p * prev[t - 1]
            if t + 1 <= i:
                val += <double>(t + 1) * prev[t + 1]
            cur[t] = val

    # Build j for all i
    for i in range(0, la + 1):
        for j in range(0, lb):
            prev = E + (i * ij_stride + j) * t_stride
            cur = E + (i * ij_stride + (j + 1)) * t_stride
            tmax_ij = i + (j + 1)
            for t in range(0, tmax_ij + 1):
                val = PB * prev[t]
                if t > 0:
                    val += inv_2p * prev[t - 1]
                if t + 1 <= i + j:
                    val += <double>(t + 1) * prev[t + 1]
                cur[t] = val


cdef inline void _build_R_coulomb(
    double p,
    double PCx,
    double PCy,
    double PCz,
    int nmax,
    double* R,
    int stride,
    double* boys,
) noexcept nogil:
    cdef int n, t, u, v
    cdef int dim = stride
    cdef int max_m
    cdef double val

    cdef double T = p * (PCx * PCx + PCy * PCy + PCz * PCz)
    _boys_fm_list(T, nmax, boys)

    cdef double fac = -2.0 * p
    cdef double pow_fac = 1.0
    for n in range(0, nmax + 1):
        R[((n * dim + 0) * dim + 0) * dim + 0] = pow_fac * boys[n]
        pow_fac *= fac

    for n in range(nmax - 1, -1, -1):
        max_m = nmax - n
        for t in range(0, max_m + 1):
            for u in range(0, max_m - t + 1):
                for v in range(0, max_m - t - u + 1):
                    if t == 0 and u == 0 and v == 0:
                        continue
                    if t > 0:
                        val = PCx * R[(((n + 1) * dim + (t - 1)) * dim + u) * dim + v]
                        if t >= 2:
                            val += <double>(t - 1) * R[(((n + 1) * dim + (t - 2)) * dim + u) * dim + v]
                        R[((n * dim + t) * dim + u) * dim + v] = val
                    elif u > 0:
                        val = PCy * R[(((n + 1) * dim + t) * dim + (u - 1)) * dim + v]
                        if u >= 2:
                            val += <double>(u - 1) * R[(((n + 1) * dim + t) * dim + (u - 2)) * dim + v]
                        R[((n * dim + t) * dim + u) * dim + v] = val
                    else:
                        val = PCz * R[(((n + 1) * dim + t) * dim + u) * dim + (v - 1)]
                        if v >= 2:
                            val += <double>(v - 1) * R[(((n + 1) * dim + t) * dim + u) * dim + (v - 2)]
                        R[((n * dim + t) * dim + u) * dim + v] = val


def build_S_cart_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] comp_start,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lx,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_ly,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairA,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairB,
    int nao,
):
    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nshell = <int>shell_l.shape[0]
    cdef Py_ssize_t npair = pairA.shape[0]
    if pairB.shape[0] != npair:
        raise ValueError("pairA and pairB must have the same length")

    cdef int sh
    cdef int lmax = 0
    cdef const cnp.int32_t* shell_l_data = <const cnp.int32_t*>shell_l.data
    for sh in range(nshell):
        if shell_l_data[sh] > lmax:
            lmax = shell_l_data[sh]

    cdef int stride = lmax + 1
    cdef int nrow = lmax + 1
    cdef size_t s_size = <size_t>(nrow * stride) * sizeof(double)
    cdef double* Sx = <double*>malloc(s_size)
    cdef double* Sy = <double*>malloc(s_size)
    cdef double* Sz = <double*>malloc(s_size)
    if Sx == NULL or Sy == NULL or Sz == NULL:
        if Sx != NULL:
            free(Sx)
        if Sy != NULL:
            free(Sy)
        if Sz != NULL:
            free(Sz)
        raise MemoryError()

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((nao, nao), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data
    cdef const cnp.int32_t* comp_start_data = <const cnp.int32_t*>comp_start.data
    cdef const cnp.int16_t* comp_lx_data = <const cnp.int16_t*>comp_lx.data
    cdef const cnp.int16_t* comp_ly_data = <const cnp.int16_t*>comp_ly.data
    cdef const cnp.int16_t* comp_lz_data = <const cnp.int16_t*>comp_lz.data
    cdef const cnp.int32_t* pairA_data = <const cnp.int32_t*>pairA.data
    cdef const cnp.int32_t* pairB_data = <const cnp.int32_t*>pairB.data

    cdef Py_ssize_t idx, ia, ib, i, j
    cdef int shA, shB, la, lb, nA, nB, aoA, aoB
    cdef int offA, offB
    cdef int sA, sB, nprimA, nprimB
    cdef double Ax, Ay, Az, Bx, By, Bz
    cdef double a, b, ca, cb, c
    cdef int lax, lay, laz, lbx, lby, lbz
    cdef double v
    try:
        with nogil:
            for idx in range(npair):
                shA = pairA_data[idx]
                shB = pairB_data[idx]

                la = shell_l_data[shA]
                lb = shell_l_data[shB]
                aoA = shell_ao_start_data[shA]
                aoB = shell_ao_start_data[shB]
                nA = _ncart(la)
                nB = _ncart(lb)
                offA = comp_start_data[la]
                offB = comp_start_data[lb]

                Ax = shell_cxyz_data[shA * 3 + 0]
                Ay = shell_cxyz_data[shA * 3 + 1]
                Az = shell_cxyz_data[shA * 3 + 2]
                Bx = shell_cxyz_data[shB * 3 + 0]
                By = shell_cxyz_data[shB * 3 + 1]
                Bz = shell_cxyz_data[shB * 3 + 2]

                sA = shell_prim_start_data[shA]
                nprimA = shell_nprim_data[shA]
                sB = shell_prim_start_data[shB]
                nprimB = shell_nprim_data[shB]

                for ia in range(nprimA):
                    a = prim_exp_data[sA + ia]
                    ca = prim_coef_data[sA + ia]
                    for ib in range(nprimB):
                        b = prim_exp_data[sB + ib]
                        cb = prim_coef_data[sB + ib]

                        _overlap_1d_table(la, lb, a, b, Ax, Bx, Sx, stride)
                        _overlap_1d_table(la, lb, a, b, Ay, By, Sy, stride)
                        _overlap_1d_table(la, lb, a, b, Az, Bz, Sz, stride)
                        c = ca * cb

                        for i in range(nA):
                            lax = comp_lx_data[offA + i]
                            lay = comp_ly_data[offA + i]
                            laz = comp_lz_data[offA + i]
                            for j in range(nB):
                                lbx = comp_lx_data[offB + j]
                                lby = comp_ly_data[offB + j]
                                lbz = comp_lz_data[offB + j]
                                v = c * Sx[lax * stride + lbx] * Sy[lay * stride + lby] * Sz[laz * stride + lbz]
                                out_data[(aoA + i) * nao + (aoB + j)] += v
                                if shA != shB:
                                    out_data[(aoB + j) * nao + (aoA + i)] += v
    finally:
        free(Sx)
        free(Sy)
        free(Sz)

    return out


def build_T_cart_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] comp_start,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lx,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_ly,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairA,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairB,
    int nao,
):
    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nshell = <int>shell_l.shape[0]
    cdef Py_ssize_t npair = pairA.shape[0]
    if pairB.shape[0] != npair:
        raise ValueError("pairA and pairB must have the same length")

    cdef int sh
    cdef int lmax = 0
    cdef const cnp.int32_t* shell_l_data = <const cnp.int32_t*>shell_l.data
    for sh in range(nshell):
        if shell_l_data[sh] > lmax:
            lmax = shell_l_data[sh]

    cdef int stride = lmax + 3  # lb+2 => (lb+3) columns; stride must cover max lb+2
    cdef int nrow = lmax + 1
    cdef size_t s_size = <size_t>(nrow * stride) * sizeof(double)
    cdef double* Sx = <double*>malloc(s_size)
    cdef double* Sy = <double*>malloc(s_size)
    cdef double* Sz = <double*>malloc(s_size)
    if Sx == NULL or Sy == NULL or Sz == NULL:
        if Sx != NULL:
            free(Sx)
        if Sy != NULL:
            free(Sy)
        if Sz != NULL:
            free(Sz)
        raise MemoryError()

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((nao, nao), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data
    cdef const cnp.int32_t* comp_start_data = <const cnp.int32_t*>comp_start.data
    cdef const cnp.int16_t* comp_lx_data = <const cnp.int16_t*>comp_lx.data
    cdef const cnp.int16_t* comp_ly_data = <const cnp.int16_t*>comp_ly.data
    cdef const cnp.int16_t* comp_lz_data = <const cnp.int16_t*>comp_lz.data
    cdef const cnp.int32_t* pairA_data = <const cnp.int32_t*>pairA.data
    cdef const cnp.int32_t* pairB_data = <const cnp.int32_t*>pairB.data

    cdef Py_ssize_t idx, ia, ib, i, j
    cdef int shA, shB, la, lb, nA, nB, aoA, aoB
    cdef int offA, offB
    cdef int sA, sB, nprimA, nprimB
    cdef double Ax, Ay, Az, Bx, By, Bz
    cdef double a, b, ca, cb, c
    cdef int lax, lay, laz, lbx, lby, lbz
    cdef double Tx, Ty, Tz, v
    try:
        with nogil:
            for idx in range(npair):
                shA = pairA_data[idx]
                shB = pairB_data[idx]

                la = shell_l_data[shA]
                lb = shell_l_data[shB]
                aoA = shell_ao_start_data[shA]
                aoB = shell_ao_start_data[shB]
                nA = _ncart(la)
                nB = _ncart(lb)
                offA = comp_start_data[la]
                offB = comp_start_data[lb]

                Ax = shell_cxyz_data[shA * 3 + 0]
                Ay = shell_cxyz_data[shA * 3 + 1]
                Az = shell_cxyz_data[shA * 3 + 2]
                Bx = shell_cxyz_data[shB * 3 + 0]
                By = shell_cxyz_data[shB * 3 + 1]
                Bz = shell_cxyz_data[shB * 3 + 2]

                sA = shell_prim_start_data[shA]
                nprimA = shell_nprim_data[shA]
                sB = shell_prim_start_data[shB]
                nprimB = shell_nprim_data[shB]

                for ia in range(nprimA):
                    a = prim_exp_data[sA + ia]
                    ca = prim_coef_data[sA + ia]
                    for ib in range(nprimB):
                        b = prim_exp_data[sB + ib]
                        cb = prim_coef_data[sB + ib]

                        _overlap_1d_table(la, lb + 2, a, b, Ax, Bx, Sx, stride)
                        _overlap_1d_table(la, lb + 2, a, b, Ay, By, Sy, stride)
                        _overlap_1d_table(la, lb + 2, a, b, Az, Bz, Sz, stride)
                        c = ca * cb

                        for i in range(nA):
                            lax = comp_lx_data[offA + i]
                            lay = comp_ly_data[offA + i]
                            laz = comp_lz_data[offA + i]
                            for j in range(nB):
                                lbx = comp_lx_data[offB + j]
                                lby = comp_ly_data[offB + j]
                                lbz = comp_lz_data[offB + j]
                                Tx = _kin1d_from_overlap(Sx, stride, lax, lbx, b)
                                Ty = _kin1d_from_overlap(Sy, stride, lay, lby, b)
                                Tz = _kin1d_from_overlap(Sz, stride, laz, lbz, b)

                                v = c * (
                                    Tx * Sy[lay * stride + lby] * Sz[laz * stride + lbz]
                                    + Sx[lax * stride + lbx] * Ty * Sz[laz * stride + lbz]
                                    + Sx[lax * stride + lbx] * Sy[lay * stride + lby] * Tz
                                )
                                out_data[(aoA + i) * nao + (aoB + j)] += v
                                if shA != shB:
                                    out_data[(aoB + j) * nao + (aoA + i)] += v
    finally:
        free(Sx)
        free(Sy)
        free(Sz)

    return out


def build_V_cart_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] atom_coords_bohr,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] atom_charges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] comp_start,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lx,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_ly,
    cnp.ndarray[cnp.int16_t, ndim=1, mode="c"] comp_lz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairA,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] pairB,
    int nao,
):
    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")
    if atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    if atom_charges.shape[0] != atom_coords_bohr.shape[0]:
        raise ValueError("atom_charges must have shape (natm,)")

    cdef int nshell = <int>shell_l.shape[0]
    cdef int natm = <int>atom_coords_bohr.shape[0]
    cdef Py_ssize_t npair = pairA.shape[0]
    if pairB.shape[0] != npair:
        raise ValueError("pairA and pairB must have the same length")

    cdef int sh
    cdef int lmax = 0
    cdef const cnp.int32_t* shell_l_data = <const cnp.int32_t*>shell_l.data
    for sh in range(nshell):
        if shell_l_data[sh] > lmax:
            lmax = shell_l_data[sh]

    cdef int ij_stride = lmax + 1
    cdef int t_stride = 2 * lmax + 1
    cdef size_t e_size = <size_t>(ij_stride * ij_stride * t_stride) * sizeof(double)
    cdef double* Ex = <double*>malloc(e_size)
    cdef double* Ey = <double*>malloc(e_size)
    cdef double* Ez = <double*>malloc(e_size)
    if Ex == NULL or Ey == NULL or Ez == NULL:
        if Ex != NULL:
            free(Ex)
        if Ey != NULL:
            free(Ey)
        if Ez != NULL:
            free(Ez)
        raise MemoryError()

    cdef int r_stride = 2 * lmax + 1
    cdef size_t r_size = <size_t>(r_stride * r_stride * r_stride * r_stride) * sizeof(double)
    cdef double* R = <double*>malloc(r_size)
    cdef double* boys = <double*>malloc(<size_t>(r_stride) * sizeof(double))
    if R == NULL or boys == NULL:
        if Ex != NULL:
            free(Ex)
        if Ey != NULL:
            free(Ey)
        if Ez != NULL:
            free(Ez)
        if R != NULL:
            free(R)
        if boys != NULL:
            free(boys)
        raise MemoryError()

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((nao, nao), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data
    cdef const double* atom_coords_data = <const double*>atom_coords_bohr.data
    cdef const double* atom_charges_data = <const double*>atom_charges.data
    cdef const cnp.int32_t* comp_start_data = <const cnp.int32_t*>comp_start.data
    cdef const cnp.int16_t* comp_lx_data = <const cnp.int16_t*>comp_lx.data
    cdef const cnp.int16_t* comp_ly_data = <const cnp.int16_t*>comp_ly.data
    cdef const cnp.int16_t* comp_lz_data = <const cnp.int16_t*>comp_lz.data
    cdef const cnp.int32_t* pairA_data = <const cnp.int32_t*>pairA.data
    cdef const cnp.int32_t* pairB_data = <const cnp.int32_t*>pairB.data

    cdef Py_ssize_t idx, ia, ib, i, j
    cdef int shA, shB, la, lb, nA, nB, aoA, aoB
    cdef int offA, offB
    cdef int sA, sB, nprimA, nprimB
    cdef double Ax, Ay, Az, Bx, By, Bz
    cdef double a, b, ca, cb, c
    cdef double p, inv_p, Px, Py, Pz
    cdef double pref, Z, Cx, Cy, Cz
    cdef int L
    cdef int lax, lay, laz, lbx, lby, lbz
    cdef int t, u, v
    cdef double s, ex, ey, ez, val
    try:
        with nogil:
            for idx in range(npair):
                shA = pairA_data[idx]
                shB = pairB_data[idx]

                la = shell_l_data[shA]
                lb = shell_l_data[shB]
                aoA = shell_ao_start_data[shA]
                aoB = shell_ao_start_data[shB]
                nA = _ncart(la)
                nB = _ncart(lb)
                offA = comp_start_data[la]
                offB = comp_start_data[lb]

                Ax = shell_cxyz_data[shA * 3 + 0]
                Ay = shell_cxyz_data[shA * 3 + 1]
                Az = shell_cxyz_data[shA * 3 + 2]
                Bx = shell_cxyz_data[shB * 3 + 0]
                By = shell_cxyz_data[shB * 3 + 1]
                Bz = shell_cxyz_data[shB * 3 + 2]

                sA = shell_prim_start_data[shA]
                nprimA = shell_nprim_data[shA]
                sB = shell_prim_start_data[shB]
                nprimB = shell_nprim_data[shB]

                L = la + lb

                for ia in range(nprimA):
                    a = prim_exp_data[sA + ia]
                    ca = prim_coef_data[sA + ia]
                    for ib in range(nprimB):
                        b = prim_exp_data[sB + ib]
                        cb = prim_coef_data[sB + ib]

                        p = a + b
                        inv_p = 1.0 / p
                        Px = (a * Ax + b * Bx) * inv_p
                        Py = (a * Ay + b * By) * inv_p
                        Pz = (a * Az + b * Bz) * inv_p

                        _hermite_E_1d_table(la, lb, a, b, Ax, Bx, Ex, ij_stride, t_stride)
                        _hermite_E_1d_table(la, lb, a, b, Ay, By, Ey, ij_stride, t_stride)
                        _hermite_E_1d_table(la, lb, a, b, Az, Bz, Ez, ij_stride, t_stride)

                        c = ca * cb
                        pref = (2.0 * _PI) * inv_p

                        for sh in range(natm):
                            Z = atom_charges_data[sh]
                            if Z == 0.0:
                                continue
                            Cx = atom_coords_data[sh * 3 + 0]
                            Cy = atom_coords_data[sh * 3 + 1]
                            Cz = atom_coords_data[sh * 3 + 2]

                            _build_R_coulomb(p, Px - Cx, Py - Cy, Pz - Cz, L, R, r_stride, boys)

                            for i in range(nA):
                                lax = comp_lx_data[offA + i]
                                lay = comp_ly_data[offA + i]
                                laz = comp_lz_data[offA + i]
                                for j in range(nB):
                                    lbx = comp_lx_data[offB + j]
                                    lby = comp_ly_data[offB + j]
                                    lbz = comp_lz_data[offB + j]

                                    s = 0.0
                                    for t in range(0, lax + lbx + 1):
                                        ex = Ex[(lax * ij_stride + lbx) * t_stride + t]
                                        if ex == 0.0:
                                            continue
                                        for u in range(0, lay + lby + 1):
                                            ey = Ey[(lay * ij_stride + lby) * t_stride + u]
                                            if ey == 0.0:
                                                continue
                                            for v in range(0, laz + lbz + 1):
                                                ez = Ez[(laz * ij_stride + lbz) * t_stride + v]
                                                if ez == 0.0:
                                                    continue
                                                s += ex * ey * ez * R[(t * r_stride + u) * r_stride + v]

                                    val = c * (-Z) * pref * s
                                    out_data[(aoA + i) * nao + (aoB + j)] += val
                                    if shA != shB:
                                        out_data[(aoB + j) * nao + (aoA + i)] += val
    finally:
        free(Ex)
        free(Ey)
        free(Ez)
        free(R)
        free(boys)

    return out


__all__ = ["build_S_cart_cy", "build_T_cart_cy", "build_V_cart_cy"]
