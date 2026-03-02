# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from __future__ import annotations

import numpy as np

cimport numpy as cnp

from cython.parallel cimport prange
from libc.math cimport acos, copysign, cos, erf, exp, fabs, sqrt
from libc.stdlib cimport free, malloc
from libc.stdint cimport int8_t


cdef double kPi = 3.141592653589793238462643383279502884
cdef double kTwoPiToFiveHalves = 2.0 * kPi * kPi * 1.772453850905516027298167483341145182  # 2*pi^(5/2)

# Set by Cython compile-time env (`CY_K_LMAX`), defaulted by build scripts.
DEF K_LMAX = CY_K_LMAX
DEF K_STRIDE = 2 * K_LMAX + 1
DEF K_GSIZE = K_STRIDE * K_STRIDE
DEF K_NCART_MAX = (K_LMAX + 1) * (K_LMAX + 2) // 2
DEF K_NROOTS_MAX = 2 * K_LMAX + 1

# For DF gradient contractions via basis-derivative relations, we need to
# evaluate shifts with component exponents up to (l+1).
DEF K_LMAX_D = K_LMAX + 1
DEF K_STRIDE_D = 2 * K_LMAX_D + 1
DEF K_GSIZE_D = K_STRIDE_D * K_STRIDE_D

# Real spherical support is limited by the available cart2sph tables.
DEF K_SPH_LMAX = 5
DEF K_NSPH_MAX = 2 * K_SPH_LMAX + 1


cdef double _CART2SPH_SP_L0[1]
cdef double _CART2SPH_SP_L1[9]
cdef double _CART2SPH_SP_L2[30]
cdef double _CART2SPH_SP_L3[70]
cdef double _CART2SPH_SP_L4[135]
cdef double _CART2SPH_SP_L5[231]
cdef bint _CART2SPH_SP_INIT = False


cdef void _init_cart2sph_sp_tables() except *:
    global _CART2SPH_SP_INIT
    if _CART2SPH_SP_INIT:
        return

    # Fill tables from the canonical Python reference implementation to avoid
    # hard-coding large numeric arrays in Cython.
    from asuka.cueri.sph import cart2sph_matrix as _cart2sph_py  # noqa: PLC0415

    cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] flat
    cdef const double* src
    cdef Py_ssize_t i

    flat = np.asarray(_cart2sph_py(0), dtype=np.float64).ravel()
    if flat.size != 1:
        raise RuntimeError("unexpected cart2sph_matrix(0) size")
    src = <const double*>flat.data
    for i in range(1):
        _CART2SPH_SP_L0[i] = src[i]

    flat = np.asarray(_cart2sph_py(1), dtype=np.float64).ravel()
    if flat.size != 9:
        raise RuntimeError("unexpected cart2sph_matrix(1) size")
    src = <const double*>flat.data
    for i in range(9):
        _CART2SPH_SP_L1[i] = src[i]

    flat = np.asarray(_cart2sph_py(2), dtype=np.float64).ravel()
    if flat.size != 30:
        raise RuntimeError("unexpected cart2sph_matrix(2) size")
    src = <const double*>flat.data
    for i in range(30):
        _CART2SPH_SP_L2[i] = src[i]

    flat = np.asarray(_cart2sph_py(3), dtype=np.float64).ravel()
    if flat.size != 70:
        raise RuntimeError("unexpected cart2sph_matrix(3) size")
    src = <const double*>flat.data
    for i in range(70):
        _CART2SPH_SP_L3[i] = src[i]

    flat = np.asarray(_cart2sph_py(4), dtype=np.float64).ravel()
    if flat.size != 135:
        raise RuntimeError("unexpected cart2sph_matrix(4) size")
    src = <const double*>flat.data
    for i in range(135):
        _CART2SPH_SP_L4[i] = src[i]

    flat = np.asarray(_cart2sph_py(5), dtype=np.float64).ravel()
    if flat.size != 231:
        raise RuntimeError("unexpected cart2sph_matrix(5) size")
    src = <const double*>flat.data
    for i in range(231):
        _CART2SPH_SP_L5[i] = src[i]

    _CART2SPH_SP_INIT = True

cdef inline int _nsph(int l) noexcept nogil:
    return 2 * l + 1


cdef inline const double* _cart2sph_sp_ptr(int l) noexcept nogil:
    if l == 0:
        return &_CART2SPH_SP_L0[0]
    if l == 1:
        return &_CART2SPH_SP_L1[0]
    if l == 2:
        return &_CART2SPH_SP_L2[0]
    if l == 3:
        return &_CART2SPH_SP_L3[0]
    if l == 4:
        return &_CART2SPH_SP_L4[0]
    if l == 5:
        return &_CART2SPH_SP_L5[0]
    return <const double*>0


cdef extern from *:
    """
    #ifdef CUERI_USE_OPENMP
    static inline int cueri_cpu_openmp_enabled(void) { return 1; }
    #else
    static inline int cueri_cpu_openmp_enabled(void) { return 0; }
    #endif
    """
    int cueri_cpu_openmp_enabled() noexcept nogil


def openmp_enabled() -> bool:
    return bool(cueri_cpu_openmp_enabled())


def kernel_limits_cy() -> dict[str, int]:
    """Return compiled CPU ERI kernel limits."""
    return {
        "lmax": int(K_LMAX),
        "nroots_max": int(K_NROOTS_MAX),
        "lmax_deriv": int(K_LMAX_D),
    }


cdef inline int _ncart(int l) noexcept nogil:
    return (l + 1) * (l + 2) // 2


cdef inline double _binom_coeff(int n, int k) noexcept nogil:
    cdef int i, kk
    cdef double out
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    kk = k
    if kk > (n - kk):
        kk = n - kk
    out = 1.0
    for i in range(1, kk + 1):
        out *= <double>(n - kk + i) / <double>i
    return out


cdef inline void _fill_cart_comp(int l, int8_t* lx, int8_t* ly, int8_t* lz) noexcept nogil:
    cdef int idx = 0
    cdef int x, y, z, rest
    for x in range(l, -1, -1):
        rest = l - x
        for y in range(rest, -1, -1):
            z = rest - y
            lx[idx] = <int8_t>x
            ly[idx] = <int8_t>y
            lz[idx] = <int8_t>z
            idx += 1


cdef inline double _clamp_double(double x, double lo, double hi) noexcept nogil:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline double _boys_f0(double T) noexcept nogil:
    if T < 1e-12:
        return 1.0 - (T / 3.0) + (T * T / 10.0)
    cdef double sT = sqrt(T)
    return 0.5 * sqrt(kPi / T) * erf(sT)


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

cdef inline void _rys_roots_weights_1(double T, double* r, double* w) noexcept nogil:
    cdef double F[2]
    _boys_fm_list(T, 1, &F[0])
    cdef double F0 = F[0]
    cdef double F1 = F[1]
    w[0] = F0
    r[0] = (F1 / F0) if (F0 > 0.0) else 0.0
    r[0] = _clamp_double(r[0], 0.0, 1.0)


cdef inline void _rys_roots_weights_2(double T, double* r, double* w) noexcept nogil:
    cdef double mu[4]
    _boys_fm_list(T, 3, &mu[0])  # mu[0..3] = F0..F3
    cdef double mu0 = mu[0]
    cdef double mu1 = mu[1]
    cdef double mu2 = mu[2]
    cdef double mu3 = mu[3]

    cdef double a0 = (mu1 / mu0) if (mu0 > 0.0) else 0.0
    cdef double h1 = mu2 - (mu1 * mu1) / (mu0 if (mu0 > 1e-300) else 1e-300)
    if h1 < 0.0:
        h1 = 0.0
    cdef double b1 = (h1 / mu0) if (mu0 > 0.0) else 0.0
    cdef double a1_num = mu3 - 2.0 * a0 * mu2 + a0 * a0 * mu1
    cdef double a1 = (a1_num / h1) if (h1 > 0.0) else a0
    cdef double s1 = sqrt(b1) if (b1 > 0.0) else 0.0

    # eigenvalues of [[a0,s1],[s1,a1]]
    cdef double tr = a0 + a1
    cdef double det = a0 * a1 - b1
    cdef double disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    cdef double root = sqrt(disc)
    cdef double l0 = 0.5 * (tr - root)
    cdef double l1 = 0.5 * (tr + root)
    if l0 > l1:
        l0, l1 = l1, l0
    r[0] = _clamp_double(l0, 0.0, 1.0)
    r[1] = _clamp_double(l1, 0.0, 1.0)

    # weights via eigenvector first component:
    # v ∝ [s1, l-a0]
    cdef double v00 = s1
    cdef double v01 = (l0 - a0)
    cdef double n0 = v00 * v00 + v01 * v01
    w[0] = (mu0 * (v00 * v00) / n0) if (n0 > 0.0) else 0.0

    cdef double v10 = s1
    cdef double v11 = (l1 - a0)
    cdef double n1 = v10 * v10 + v11 * v11
    w[1] = (mu0 * (v10 * v10) / n1) if (n1 > 0.0) else 0.0


cdef inline void _rys_roots_weights_3(double T, double* r, double* w) noexcept nogil:
    cdef double mu[6]
    _boys_fm_list(T, 5, &mu[0])  # mu[0..5] = F0..F5

    cdef double mu0 = mu[0]
    cdef double mu1 = mu[1]
    cdef double mu2 = mu[2]
    cdef double mu3 = mu[3]
    cdef double mu4 = mu[4]
    cdef double mu5 = mu[5]

    cdef double a0 = (mu1 / mu0) if (mu0 > 0.0) else 0.0
    cdef double h1 = mu2 - (mu1 * mu1) / (mu0 if (mu0 > 1e-300) else 1e-300)
    if h1 < 0.0:
        h1 = 0.0
    cdef double b1 = (h1 / mu0) if (mu0 > 0.0) else 0.0
    cdef double a1_num = mu3 - 2.0 * a0 * mu2 + a0 * a0 * mu1
    cdef double a1 = (a1_num / h1) if (h1 > 0.0) else a0

    # P2(x) = (x-a1)(x-a0) - b1 = x^2 + c1 x + c0
    cdef double c1 = -(a0 + a1)
    cdef double c0 = a0 * a1 - b1

    # h2 = <P2,P2>
    cdef double h2 = (
        mu4
        + 2.0 * c1 * mu3
        + 2.0 * c0 * mu2
        + (c1 * c1) * mu2
        + 2.0 * c1 * c0 * mu1
        + (c0 * c0) * mu0
    )
    if h2 < 0.0:
        h2 = 0.0
    cdef double b2 = (h2 / h1) if (h1 > 0.0) else 0.0

    # a2 = <x P2, P2> / h2
    cdef double a2_num = (
        mu5
        + 2.0 * c1 * mu4
        + 2.0 * c0 * mu3
        + (c1 * c1) * mu3
        + 2.0 * c1 * c0 * mu2
        + (c0 * c0) * mu1
    )
    cdef double a2 = (a2_num / h2) if (h2 > 0.0) else a1

    cdef double s1 = sqrt(b1) if (b1 > 0.0) else 0.0
    cdef double s2 = sqrt(b2) if (b2 > 0.0) else 0.0

    # Symmetric 3x3 eigenvalues (stable closed form), for Jacobi matrix:
    # [[a0,s1,0],[s1,a1,s2],[0,s2,a2]]
    cdef double a11 = a0
    cdef double a22 = a1
    cdef double a33 = a2
    cdef double a12 = s1
    cdef double a13 = 0.0
    cdef double a23 = s2

    cdef double p1 = a12 * a12 + a13 * a13 + a23 * a23
    cdef double l0 = a11
    cdef double l1 = a22
    cdef double l2 = a33
    cdef double q, p2, p
    cdef double b11, b22, b33, b12, b13, b23
    cdef double detB, rr, phi
    if p1 > 0.0:
        q = (a11 + a22 + a33) / 3.0
        p2 = (
            (a11 - q) * (a11 - q)
            + (a22 - q) * (a22 - q)
            + (a33 - q) * (a33 - q)
            + 2.0 * p1
        )
        p = sqrt(p2 / 6.0) if (p2 > 0.0) else 0.0
        if p <= 0.0:
            p = 0.0
        else:

            b11 = (a11 - q) / p
            b22 = (a22 - q) / p
            b33 = (a33 - q) / p
            b12 = a12 / p
            b13 = a13 / p
            b23 = a23 / p

            detB = (
                b11 * (b22 * b33 - b23 * b23)
                - b12 * (b12 * b33 - b23 * b13)
                + b13 * (b12 * b23 - b22 * b13)
            )
            rr = _clamp_double(detB * 0.5, -1.0, 1.0)
            phi = acos(rr) / 3.0

            l2 = q + 2.0 * p * cos(phi)
            l0 = q + 2.0 * p * cos(phi + (2.0 * kPi / 3.0))
            l1 = 3.0 * q - l0 - l2

    # sort ascending
    cdef double e0 = l0
    cdef double e1 = l1
    cdef double e2 = l2
    cdef double tmp
    if e0 > e1:
        tmp = e0
        e0 = e1
        e1 = tmp
    if e1 > e2:
        tmp = e1
        e1 = e2
        e2 = tmp
    if e0 > e1:
        tmp = e0
        e0 = e1
        e1 = tmp

    r[0] = _clamp_double(e0, 0.0, 1.0)
    r[1] = _clamp_double(e1, 0.0, 1.0)
    r[2] = _clamp_double(e2, 0.0, 1.0)

    # weights via eigenvector first component:
    # v ∝ cross(row0,row2) for (J - λI)
    cdef double r0x, r2z, v0, v1, v2, n2
    r0x = a0 - e0
    r2z = a2 - e0
    v0 = s1 * r2z
    v1 = -r0x * r2z
    v2 = r0x * s2
    n2 = v0 * v0 + v1 * v1 + v2 * v2
    w[0] = (mu0 * (v0 * v0) / n2) if (n2 > 0.0) else 0.0

    r0x = a0 - e1
    r2z = a2 - e1
    v0 = s1 * r2z
    v1 = -r0x * r2z
    v2 = r0x * s2
    n2 = v0 * v0 + v1 * v1 + v2 * v2
    w[1] = (mu0 * (v0 * v0) / n2) if (n2 > 0.0) else 0.0

    r0x = a0 - e2
    r2z = a2 - e2
    v0 = s1 * r2z
    v1 = -r0x * r2z
    v2 = r0x * s2
    n2 = v0 * v0 + v1 * v1 + v2 * v2
    w[2] = (mu0 * (v0 * v0) / n2) if (n2 > 0.0) else 0.0


cdef inline double _poly_inner_moments(
    const double* a,
    int deg_a,
    const double* b,
    int deg_b,
    const double* mu,
    int shift,
) noexcept nogil:
    cdef int i, j
    cdef double s = 0.0
    cdef double ai
    for i in range(deg_a + 1):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(deg_b + 1):
            s += ai * b[j] * mu[i + j + shift]
    return s


cdef inline void _tridiag_eigh_ql(int n, double* d, double* e, double* z0) noexcept nogil:
    # Symmetric tridiagonal eigen decomposition via implicit QL iterations.
    #
    # Inputs:
    #   d[0..n-1]: diagonal
    #   e[0..n-2]: off-diagonal
    # Outputs:
    #   d[] : eigenvalues (ascending)
    #   z0[]: first component of each eigenvector (aligned with d[])
    #
    # Adapted from the classic tqli routine (Numerical Recipes), rewritten for 0-based indexing.

    cdef int kMaxIter = 128
    cdef double kEps = 1e-15

    cdef int i, j, k, l, m, iter
    cdef double dd, g, r, s, c, p, f, b, rr
    cdef double z_ip1, z_i
    cdef bint cont

    if n <= 0:
        return

    # Extend e with trailing zero (algorithm accesses e[m]).
    e[n - 1] = 0.0

    # First row of identity.
    for i in range(n):
        z0[i] = 1.0 if (i == 0) else 0.0

    for l in range(n):
        iter = 0
        while True:
            m = l
            for m in range(l, n - 1):
                dd = fabs(d[m]) + fabs(d[m + 1])
                if dd == 0.0:
                    dd = 1.0
                if fabs(e[m]) <= kEps * dd:
                    break
            else:
                # Mirror C/CUDA for-loop semantics: when the loop terminates without
                # a break, `m` is logically `n-1` (not `n-2` as in a Python `for` loop).
                m = n - 1
            if m == l:
                break
            if iter >= kMaxIter:
                break
            iter += 1

            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = sqrt(g * g + 1.0)
            g = d[m] - d[l] + e[l] / (g + copysign(r, g))

            s = 1.0
            c = 1.0
            p = 0.0

            cont = False
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = sqrt(f * f + g * g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    cont = True
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                rr = (d[i] - g) * s + 2.0 * c * b
                p = s * rr
                d[i + 1] = g + p
                g = c * rr - b

                # Update first row of eigenvector matrix.
                z_ip1 = z0[i + 1]
                z_i = z0[i]
                z0[i + 1] = s * z_i + c * z_ip1
                z0[i] = c * z_i - s * z_ip1

            if cont:
                continue
            d[l] -= p
            e[l] = g
            e[m] = 0.0

    # Sort by ascending eigenvalue, swapping z0 accordingly.
    for i in range(n - 1):
        k = i
        p = d[i]
        for j in range(i + 1, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            dd = d[i]
            d[i] = d[k]
            d[k] = dd
            dd = z0[i]
            z0[i] = z0[k]
            z0[k] = dd


cdef inline void _rys_roots_weights_generic(int nroots, double T, double* r, double* w) noexcept nogil:
    # Generic Stieltjes + Jacobi tridiagonal eigen method.
    cdef int i, k
    if nroots <= 0:
        return

    cdef bint use_heap = nroots > K_NROOTS_MAX

    cdef double mu_stack[2 * K_NROOTS_MAX]
    cdef double alpha_stack[K_NROOTS_MAX]
    cdef double beta_stack[K_NROOTS_MAX]
    cdef double p_prev_stack[K_NROOTS_MAX + 1]
    cdef double p_curr_stack[K_NROOTS_MAX + 1]
    cdef double p_next_stack[K_NROOTS_MAX + 1]
    cdef double d_stack[K_NROOTS_MAX]
    cdef double e_stack[K_NROOTS_MAX]
    cdef double z0_stack[K_NROOTS_MAX]

    cdef double* mu = &mu_stack[0]
    cdef double mu0, h_curr, h_next, a_num, beta_k
    cdef double* alpha = &alpha_stack[0]
    cdef double* beta = &beta_stack[0]  # beta[0] unused
    cdef double* p_prev = &p_prev_stack[0]
    cdef double* p_curr = &p_curr_stack[0]
    cdef double* p_next = &p_next_stack[0]
    cdef double* d = &d_stack[0]
    cdef double* e = &e_stack[0]
    cdef double* z0 = &z0_stack[0]

    if use_heap:
        mu = <double*>malloc((2 * nroots) * sizeof(double))
        alpha = <double*>malloc(nroots * sizeof(double))
        beta = <double*>malloc(nroots * sizeof(double))
        p_prev = <double*>malloc((nroots + 1) * sizeof(double))
        p_curr = <double*>malloc((nroots + 1) * sizeof(double))
        p_next = <double*>malloc((nroots + 1) * sizeof(double))
        d = <double*>malloc(nroots * sizeof(double))
        e = <double*>malloc(nroots * sizeof(double))
        z0 = <double*>malloc(nroots * sizeof(double))
        if (
            mu == NULL or alpha == NULL or beta == NULL
            or p_prev == NULL or p_curr == NULL or p_next == NULL
            or d == NULL or e == NULL or z0 == NULL
        ):
            for i in range(nroots):
                r[i] = 0.0
                w[i] = 0.0
            if mu != NULL: free(mu)
            if alpha != NULL: free(alpha)
            if beta != NULL: free(beta)
            if p_prev != NULL: free(p_prev)
            if p_curr != NULL: free(p_curr)
            if p_next != NULL: free(p_next)
            if d != NULL: free(d)
            if e != NULL: free(e)
            if z0 != NULL: free(z0)
            return

    _boys_fm_list(T, 2 * nroots - 1, mu)
    mu0 = mu[0]
    if not (mu0 > 0.0):
        for i in range(nroots):
            r[i] = 0.0
            w[i] = 0.0
        if use_heap:
            free(mu); free(alpha); free(beta)
            free(p_prev); free(p_curr); free(p_next)
            free(d); free(e); free(z0)
        return

    for i in range(nroots + 1):
        p_prev[i] = 0.0
        p_curr[i] = 0.0
        p_next[i] = 0.0
    p_curr[0] = 1.0  # P0(x) = 1

    h_curr = mu0
    alpha[0] = mu[1] / mu0
    beta[0] = 0.0

    for k in range(nroots - 1):
        beta_k = beta[k] if (k >= 1 and h_curr > 0.0) else 0.0

        # p_next = (x - alpha[k]) p_curr - beta_k p_prev
        for i in range(nroots + 1):
            p_next[i] = 0.0
        for i in range(k + 1):
            p_next[i + 1] += p_curr[i]
            p_next[i] -= alpha[k] * p_curr[i]
        if k >= 1:
            for i in range(k):
                p_next[i] -= beta_k * p_prev[i]

        h_next = _poly_inner_moments(p_next, k + 1, p_next, k + 1, mu, 0)
        if h_next < 0.0:
            h_next = 0.0
        beta[k + 1] = (h_next / h_curr) if (h_curr > 0.0) else 0.0

        a_num = _poly_inner_moments(p_next, k + 1, p_next, k + 1, mu, 1)
        alpha[k + 1] = (a_num / h_next) if (h_next > 0.0) else alpha[k]

        # Shift (P_{k-1}, P_k) <- (P_k, P_{k+1})
        for i in range(nroots + 1):
            p_prev[i] = p_curr[i]
            p_curr[i] = p_next[i]
        h_curr = h_next

    for i in range(nroots):
        d[i] = alpha[i]
        e[i] = 0.0
    for i in range(nroots - 1):
        e[i] = sqrt(beta[i + 1]) if (beta[i + 1] > 0.0) else 0.0

    _tridiag_eigh_ql(nroots, d, e, z0)

    for i in range(nroots):
        r[i] = _clamp_double(d[i], 0.0, 1.0)
        w[i] = mu0 * z0[i] * z0[i]
        if w[i] < 0.0:
            w[i] = 0.0

    if use_heap:
        free(mu); free(alpha); free(beta)
        free(p_prev); free(p_curr); free(p_next)
        free(d); free(e); free(z0)


cdef inline void _rys_roots_weights(int nroots, double T, double* r, double* w) noexcept nogil:
    if nroots == 1:
        _rys_roots_weights_1(T, r, w)
        return
    if nroots == 2:
        _rys_roots_weights_2(T, r, w)
        return
    if nroots == 3:
        _rys_roots_weights_3(T, r, w)
        return
    _rys_roots_weights_generic(nroots, T, r, w)


cdef inline void _compute_G(
    double* G,
    int nmax,
    int mmax,
    double C,
    double Cp,
    double B0,
    double B1,
    double B1p,
) noexcept nogil:
    # Normalized PyQuante2 / GAMESS recurrence:
    # G[0,0]=1 and all other entries are dimensionless polynomials.

    cdef int a, b

    G[0] = 1.0
    if nmax > 0:
        G[1 * K_STRIDE + 0] = C
    if mmax > 0:
        G[0 * K_STRIDE + 1] = Cp

    for a in range(2, nmax + 1):
        G[a * K_STRIDE + 0] = B1 * <double>(a - 1) * G[(a - 2) * K_STRIDE + 0] + C * G[(a - 1) * K_STRIDE + 0]
    for b in range(2, mmax + 1):
        G[0 * K_STRIDE + b] = B1p * <double>(b - 1) * G[0 * K_STRIDE + (b - 2)] + Cp * G[0 * K_STRIDE + (b - 1)]

    if mmax == 0 or nmax == 0:
        return

    for a in range(1, nmax + 1):
        G[a * K_STRIDE + 1] = <double>a * B0 * G[(a - 1) * K_STRIDE + 0] + Cp * G[a * K_STRIDE + 0]
        for b in range(2, mmax + 1):
            G[a * K_STRIDE + b] = (
                B1p * <double>(b - 1) * G[a * K_STRIDE + (b - 2)]
                + <double>a * B0 * G[(a - 1) * K_STRIDE + (b - 1)]
                + Cp * G[a * K_STRIDE + (b - 1)]
            )


cdef inline void _compute_G_d(
    double* G,
    int nmax,
    int mmax,
    double C,
    double Cp,
    double B0,
    double B1,
    double B1p,
) noexcept nogil:
    # Same recurrence as _compute_G but with a larger stride to support
    # component exponents up to 6 in DF derivative contractions.

    cdef int a, b

    G[0] = 1.0
    if nmax > 0:
        G[1 * K_STRIDE_D + 0] = C
    if mmax > 0:
        G[0 * K_STRIDE_D + 1] = Cp

    for a in range(2, nmax + 1):
        G[a * K_STRIDE_D + 0] = B1 * <double>(a - 1) * G[(a - 2) * K_STRIDE_D + 0] + C * G[(a - 1) * K_STRIDE_D + 0]
    for b in range(2, mmax + 1):
        G[0 * K_STRIDE_D + b] = B1p * <double>(b - 1) * G[0 * K_STRIDE_D + (b - 2)] + Cp * G[0 * K_STRIDE_D + (b - 1)]

    if mmax == 0 or nmax == 0:
        return

    for a in range(1, nmax + 1):
        G[a * K_STRIDE_D + 1] = <double>a * B0 * G[(a - 1) * K_STRIDE_D + 0] + Cp * G[a * K_STRIDE_D + 0]
        for b in range(2, mmax + 1):
            G[a * K_STRIDE_D + b] = (
                B1p * <double>(b - 1) * G[a * K_STRIDE_D + (b - 2)]
                + <double>a * B0 * G[(a - 1) * K_STRIDE_D + (b - 1)]
                + Cp * G[a * K_STRIDE_D + (b - 1)]
            )


cdef inline void _compute_G_runtime(
    double* G,
    int nmax,
    int mmax,
    int stride,
    double C,
    double Cp,
    double B0,
    double B1,
    double B1p,
) noexcept nogil:
    # Runtime-stride variant used by dynamic high-l fallbacks.
    cdef int a, b

    G[0] = 1.0
    if nmax > 0:
        G[1 * stride + 0] = C
    if mmax > 0:
        G[0 * stride + 1] = Cp

    for a in range(2, nmax + 1):
        G[a * stride + 0] = B1 * <double>(a - 1) * G[(a - 2) * stride + 0] + C * G[(a - 1) * stride + 0]
    for b in range(2, mmax + 1):
        G[0 * stride + b] = B1p * <double>(b - 1) * G[0 * stride + (b - 2)] + Cp * G[0 * stride + (b - 1)]

    if mmax == 0 or nmax == 0:
        return

    for a in range(1, nmax + 1):
        G[a * stride + 1] = <double>a * B0 * G[(a - 1) * stride + 0] + Cp * G[a * stride + 0]
        for b in range(2, mmax + 1):
            G[a * stride + b] = (
                B1p * <double>(b - 1) * G[a * stride + (b - 2)]
                + <double>a * B0 * G[(a - 1) * stride + (b - 1)]
                + Cp * G[a * stride + (b - 1)]
            )


cdef inline double _shift_from_G_ld0_d(
    const double* G,
    int i,
    int j,
    int k,
    const double* xij_pow,
) noexcept nogil:
    # Specialized shift for ld=0 (DF int3c2e / metric_2c2e): l=0 => outer sum collapses.
    cdef int n
    cdef double out = 0.0
    for n in range(0, j + 1):
        out += _binom_coeff(j, n) * xij_pow[j - n] * G[(n + i) * K_STRIDE_D + k]
    return out


cdef inline double _shift_from_G(
    const double* G,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow,
) noexcept nogil:
    cdef int m, n
    cdef double ijkl = 0.0
    cdef double ijm0
    for m in range(0, l + 1):
        ijm0 = 0.0
        for n in range(0, j + 1):
            ijm0 += _binom_coeff(j, n) * xij_pow[j - n] * G[(n + i) * K_STRIDE + (m + k)]
        ijkl += _binom_coeff(l, m) * xkl_pow[l - m] * ijm0
    return ijkl


cdef inline double _shift_from_G_runtime(
    const double* G,
    int stride,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow,
) noexcept nogil:
    cdef int m, n
    cdef double ijkl = 0.0
    cdef double ijm0
    for m in range(0, l + 1):
        ijm0 = 0.0
        for n in range(0, j + 1):
            ijm0 += _binom_coeff(j, n) * xij_pow[j - n] * G[(n + i) * stride + (m + k)]
        ijkl += _binom_coeff(l, m) * xkl_pow[l - m] * ijm0
    return ijkl



cdef inline double _shift_from_G_d(
    const double* G,
    int i,
    int j,
    int k,
    int l,
    const double* xij_pow,
    const double* xkl_pow,
) noexcept nogil:
    # Derivative-stride shift: supports j,l up to K_LMAX_D using K_STRIDE_D.
    cdef int m, n
    cdef double ijkl = 0.0
    cdef double ijm0
    for m in range(0, l + 1):
        ijm0 = 0.0
        for n in range(0, j + 1):
            ijm0 += _binom_coeff(j, n) * xij_pow[j - n] * G[(n + i) * K_STRIDE_D + (m + k)]
        ijkl += _binom_coeff(l, m) * xkl_pow[l - m] * ijm0
    return ijkl

cdef inline void _eri_rys_tile_cart_kernel(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_prim_start_data,
    const cnp.int32_t* shell_nprim_data,
    const cnp.int32_t* shell_l_data,
    const double* prim_exp_data,
    const double* prim_coef_data,
    int shellA,
    int shellB,
    int shellC,
    int shellD,
    double* out_data,
) noexcept nogil:
    cdef int la = <int>shell_l_data[shellA]
    cdef int lb = <int>shell_l_data[shellB]
    cdef int lc = <int>shell_l_data[shellC]
    cdef int ld = <int>shell_l_data[shellD]

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nCD = nC * nD

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]
    cdef double Dx = shell_cxyz_data[shellD * 3 + 0]
    cdef double Dy = shell_cxyz_data[shellD * 3 + 1]
    cdef double Dz = shell_cxyz_data[shellD * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz
    cdef double AB2 = ABx * ABx + ABy * ABy + ABz * ABz

    cdef double CDx = Cx - Dx
    cdef double CDy = Cy - Dy
    cdef double CDz = Cz - Dz
    cdef double CD2 = CDx * CDx + CDy * CDy + CDz * CDz

    cdef double xij_pow[K_LMAX + 1]
    cdef double yij_pow[K_LMAX + 1]
    cdef double zij_pow[K_LMAX + 1]
    cdef double xkl_pow[K_LMAX + 1]
    cdef double ykl_pow[K_LMAX + 1]
    cdef double zkl_pow[K_LMAX + 1]
    cdef int pwr

    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    xkl_pow[0] = 1.0
    ykl_pow[0] = 1.0
    zkl_pow[0] = 1.0
    for pwr in range(1, K_LMAX + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz
        xkl_pow[pwr] = xkl_pow[pwr - 1] * CDx
        ykl_pow[pwr] = ykl_pow[pwr - 1] * CDy
        zkl_pow[pwr] = zkl_pow[pwr - 1] * CDz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    cdef int8_t D_lx[K_NCART_MAX]
    cdef int8_t D_ly[K_NCART_MAX]
    cdef int8_t D_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])
    _fill_cart_comp(ld, &D_lx[0], &D_ly[0], &D_lz[0])

    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sB = <int>shell_prim_start_data[shellB]
    cdef int sC = <int>shell_prim_start_data[shellC]
    cdef int sD = <int>shell_prim_start_data[shellD]
    cdef int nprimA = <int>shell_nprim_data[shellA]
    cdef int nprimB = <int>shell_nprim_data[shellB]
    cdef int nprimC = <int>shell_nprim_data[shellC]
    cdef int nprimD = <int>shell_nprim_data[shellD]

    cdef int nmax = la + lb
    cdef int mmax = lc + ld
    cdef int L_total = la + lb + lc + ld
    cdef int nroots = (L_total // 2) + 1

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE]
    cdef double Gy[K_GSIZE]
    cdef double Gz[K_GSIZE]

    cdef int ipA, ipB, ipC, ipD
    cdef int ia, ib, ic, id, u
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz, idx, idy, idz
    cdef int row, col

    cdef double a, b, cexp, dexp
    cdef double ca, cb, cc, cd
    cdef double p, q, inv_p, inv_q, mu, Kab, Kcd
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz

    for ipA in range(nprimA):
        a = prim_exp_data[sA + ipA]
        ca = prim_coef_data[sA + ipA]
        for ipB in range(nprimB):
            b = prim_exp_data[sB + ipB]
            cb = prim_coef_data[sB + ipB]

            p = a + b
            inv_p = 1.0 / p
            mu = a * b * inv_p
            Px = (a * Ax + b * Bx) * inv_p
            Py = (a * Ay + b * By) * inv_p
            Pz = (a * Az + b * Bz) * inv_p
            Kab = exp(-mu * AB2)
            cKab = (ca * cb) * Kab

            for ipC in range(nprimC):
                cexp = prim_exp_data[sC + ipC]
                cc = prim_coef_data[sC + ipC]
                for ipD in range(nprimD):
                    dexp = prim_exp_data[sD + ipD]
                    cd = prim_coef_data[sD + ipD]

                    q = cexp + dexp
                    inv_q = 1.0 / q
                    mu = cexp * dexp * inv_q
                    Qx = (cexp * Cx + dexp * Dx) * inv_q
                    Qy = (cexp * Cy + dexp * Dy) * inv_q
                    Qz = (cexp * Cz + dexp * Dz) * inv_q
                    Kcd = exp(-mu * CD2)
                    cKcd = (cc * cd) * Kcd

                    denom = p + q
                    inv_denom = 1.0 / denom
                    dx = Px - Qx
                    dy = Py - Qy
                    dz = Pz - Qz
                    PQ2 = dx * dx + dy * dy + dz * dz
                    omega = p * q * inv_denom
                    T = omega * PQ2

                    base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                    _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                    for u in range(nroots):
                        x = roots[u]
                        w = weights[u]

                        B0 = x * 0.5 * inv_denom
                        B1 = (1.0 - x) * 0.5 / p + B0
                        B1p = (1.0 - x) * 0.5 / q + B0

                        Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                        Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                        Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                        Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                        Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                        Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                        _compute_G(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                        _compute_G(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                        _compute_G(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                        scale = base * w

                        for ia in range(nA):
                            iax = <int>A_lx[ia]
                            iay = <int>A_ly[ia]
                            iaz = <int>A_lz[ia]
                            for ib in range(nB):
                                ibx = <int>B_lx[ib]
                                iby = <int>B_ly[ib]
                                ibz = <int>B_lz[ib]
                                row = ia * nB + ib
                                for ic in range(nC):
                                    icx = <int>C_lx[ic]
                                    icy = <int>C_ly[ic]
                                    icz = <int>C_lz[ic]
                                    for id in range(nD):
                                        idx = <int>D_lx[id]
                                        idy = <int>D_ly[id]
                                        idz = <int>D_lz[id]
                                        col = ic * nD + id

                                        Ix = _shift_from_G(&Gx[0], iax, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                        Iy = _shift_from_G(&Gy[0], iay, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                        Iz = _shift_from_G(&Gz[0], iaz, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])
                                        out_data[row * nCD + col] += scale * (Ix * Iy * Iz)


cdef inline void _eri_rys_tile_cart_sp_kernel(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_l_data,
    const cnp.int32_t* sp_A_data,
    const cnp.int32_t* sp_B_data,
    const cnp.int32_t* sp_pair_start_data,
    const cnp.int32_t* sp_npair_data,
    const double* pair_eta_data,
    const double* pair_Px_data,
    const double* pair_Py_data,
    const double* pair_Pz_data,
    const double* pair_cK_data,
    int spAB,
    int spCD,
    double* out_data,
) noexcept nogil:
    cdef int shellA = <int>sp_A_data[spAB]
    cdef int shellB = <int>sp_B_data[spAB]
    cdef int shellC = <int>sp_A_data[spCD]
    cdef int shellD = <int>sp_B_data[spCD]

    cdef int la = <int>shell_l_data[shellA]
    cdef int lb = <int>shell_l_data[shellB]
    cdef int lc = <int>shell_l_data[shellC]
    cdef int ld = <int>shell_l_data[shellD]

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nCD = nC * nD

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]
    cdef double Dx = shell_cxyz_data[shellD * 3 + 0]
    cdef double Dy = shell_cxyz_data[shellD * 3 + 1]
    cdef double Dz = shell_cxyz_data[shellD * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz

    cdef double CDx = Cx - Dx
    cdef double CDy = Cy - Dy
    cdef double CDz = Cz - Dz

    cdef double xij_pow[K_LMAX + 1]
    cdef double yij_pow[K_LMAX + 1]
    cdef double zij_pow[K_LMAX + 1]
    cdef double xkl_pow[K_LMAX + 1]
    cdef double ykl_pow[K_LMAX + 1]
    cdef double zkl_pow[K_LMAX + 1]
    cdef int pwr

    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    xkl_pow[0] = 1.0
    ykl_pow[0] = 1.0
    zkl_pow[0] = 1.0
    for pwr in range(1, K_LMAX + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz
        xkl_pow[pwr] = xkl_pow[pwr - 1] * CDx
        ykl_pow[pwr] = ykl_pow[pwr - 1] * CDy
        zkl_pow[pwr] = zkl_pow[pwr - 1] * CDz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    cdef int8_t D_lx[K_NCART_MAX]
    cdef int8_t D_ly[K_NCART_MAX]
    cdef int8_t D_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])
    _fill_cart_comp(ld, &D_lx[0], &D_ly[0], &D_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int baseCD = <int>sp_pair_start_data[spCD]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimCD = <int>sp_npair_data[spCD]

    cdef int nmax = la + lb
    cdef int mmax = lc + ld
    cdef int L_total = la + lb + lc + ld
    cdef int nroots = (L_total // 2) + 1

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE]
    cdef double Gy[K_GSIZE]
    cdef double Gz[K_GSIZE]

    cdef int iAB, iCD
    cdef int ia, ib, ic, id, u
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz, idx, idy, idz
    cdef int row, col

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz

    for iAB in range(nprimAB):
        p = pair_eta_data[baseAB + iAB]
        Px = pair_Px_data[baseAB + iAB]
        Py = pair_Py_data[baseAB + iAB]
        Pz = pair_Pz_data[baseAB + iAB]
        cKab = pair_cK_data[baseAB + iAB]

        for iCD in range(nprimCD):
            q = pair_eta_data[baseCD + iCD]
            Qx = pair_Px_data[baseCD + iCD]
            Qy = pair_Py_data[baseCD + iCD]
            Qz = pair_Pz_data[baseCD + iCD]
            cKcd = pair_cK_data[baseCD + iCD]

            denom = p + q
            inv_denom = 1.0 / denom
            dx = Px - Qx
            dy = Py - Qy
            dz = Pz - Qz
            PQ2 = dx * dx + dy * dy + dz * dz
            omega = p * q * inv_denom
            T = omega * PQ2

            base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

            _rys_roots_weights(nroots, T, &roots[0], &weights[0])

            for u in range(nroots):
                x = roots[u]
                w = weights[u]

                B0 = x * 0.5 * inv_denom
                B1 = (1.0 - x) * 0.5 / p + B0
                B1p = (1.0 - x) * 0.5 / q + B0

                Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                _compute_G(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                _compute_G(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                _compute_G(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                scale = base * w

                for ia in range(nA):
                    iax = <int>A_lx[ia]
                    iay = <int>A_ly[ia]
                    iaz = <int>A_lz[ia]
                    for ib in range(nB):
                        ibx = <int>B_lx[ib]
                        iby = <int>B_ly[ib]
                        ibz = <int>B_lz[ib]
                        row = ia * nB + ib
                        for ic in range(nC):
                            icx = <int>C_lx[ic]
                            icy = <int>C_ly[ic]
                            icz = <int>C_lz[ic]
                            for id in range(nD):
                                idx = <int>D_lx[id]
                                idy = <int>D_ly[id]
                                idz = <int>D_lz[id]
                                col = ic * nD + id

                                Ix = _shift_from_G(&Gx[0], iax, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                Iy = _shift_from_G(&Gy[0], iay, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                Iz = _shift_from_G(&Gz[0], iaz, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])
                                out_data[row * nCD + col] += scale * (Ix * Iy * Iz)


cdef inline void _eri_rys_tile_cart_sp_kernel_dynamic(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_l_data,
    const cnp.int32_t* sp_A_data,
    const cnp.int32_t* sp_B_data,
    const cnp.int32_t* sp_pair_start_data,
    const cnp.int32_t* sp_npair_data,
    const double* pair_eta_data,
    const double* pair_Px_data,
    const double* pair_Py_data,
    const double* pair_Pz_data,
    const double* pair_cK_data,
    int spAB,
    int spCD,
    double* out_data,
) noexcept nogil:
    # Functional high-l fallback with runtime-sized workspaces.
    cdef int shellA = <int>sp_A_data[spAB]
    cdef int shellB = <int>sp_B_data[spAB]
    cdef int shellC = <int>sp_A_data[spCD]
    cdef int shellD = <int>sp_B_data[spCD]

    cdef int la = <int>shell_l_data[shellA]
    cdef int lb = <int>shell_l_data[shellB]
    cdef int lc = <int>shell_l_data[shellC]
    cdef int ld = <int>shell_l_data[shellD]

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nCD = nC * nD

    cdef int max_ij = la if la > lb else lb
    cdef int max_kl = lc if lc > ld else ld
    cdef int nmax = la + lb
    cdef int mmax = lc + ld
    cdef int stride = mmax + 1
    cdef int gsize = (nmax + 1) * stride
    cdef int nroots = ((la + lb + lc + ld) // 2) + 1

    cdef double* xij_pow = <double*>malloc((max_ij + 1) * sizeof(double))
    cdef double* yij_pow = <double*>malloc((max_ij + 1) * sizeof(double))
    cdef double* zij_pow = <double*>malloc((max_ij + 1) * sizeof(double))
    cdef double* xkl_pow = <double*>malloc((max_kl + 1) * sizeof(double))
    cdef double* ykl_pow = <double*>malloc((max_kl + 1) * sizeof(double))
    cdef double* zkl_pow = <double*>malloc((max_kl + 1) * sizeof(double))

    cdef int8_t* A_lx = <int8_t*>malloc(nA * sizeof(int8_t))
    cdef int8_t* A_ly = <int8_t*>malloc(nA * sizeof(int8_t))
    cdef int8_t* A_lz = <int8_t*>malloc(nA * sizeof(int8_t))
    cdef int8_t* B_lx = <int8_t*>malloc(nB * sizeof(int8_t))
    cdef int8_t* B_ly = <int8_t*>malloc(nB * sizeof(int8_t))
    cdef int8_t* B_lz = <int8_t*>malloc(nB * sizeof(int8_t))
    cdef int8_t* C_lx = <int8_t*>malloc(nC * sizeof(int8_t))
    cdef int8_t* C_ly = <int8_t*>malloc(nC * sizeof(int8_t))
    cdef int8_t* C_lz = <int8_t*>malloc(nC * sizeof(int8_t))
    cdef int8_t* D_lx = <int8_t*>malloc(nD * sizeof(int8_t))
    cdef int8_t* D_ly = <int8_t*>malloc(nD * sizeof(int8_t))
    cdef int8_t* D_lz = <int8_t*>malloc(nD * sizeof(int8_t))

    cdef double* roots = <double*>malloc(nroots * sizeof(double))
    cdef double* weights = <double*>malloc(nroots * sizeof(double))
    cdef double* Gx = <double*>malloc(gsize * sizeof(double))
    cdef double* Gy = <double*>malloc(gsize * sizeof(double))
    cdef double* Gz = <double*>malloc(gsize * sizeof(double))

    cdef bint alloc_ok = (
        xij_pow != NULL and yij_pow != NULL and zij_pow != NULL
        and xkl_pow != NULL and ykl_pow != NULL and zkl_pow != NULL
        and A_lx != NULL and A_ly != NULL and A_lz != NULL
        and B_lx != NULL and B_ly != NULL and B_lz != NULL
        and C_lx != NULL and C_ly != NULL and C_lz != NULL
        and D_lx != NULL and D_ly != NULL and D_lz != NULL
        and roots != NULL and weights != NULL
        and Gx != NULL and Gy != NULL and Gz != NULL
    )
    if not alloc_ok:
        if xij_pow != NULL: free(xij_pow)
        if yij_pow != NULL: free(yij_pow)
        if zij_pow != NULL: free(zij_pow)
        if xkl_pow != NULL: free(xkl_pow)
        if ykl_pow != NULL: free(ykl_pow)
        if zkl_pow != NULL: free(zkl_pow)
        if A_lx != NULL: free(A_lx)
        if A_ly != NULL: free(A_ly)
        if A_lz != NULL: free(A_lz)
        if B_lx != NULL: free(B_lx)
        if B_ly != NULL: free(B_ly)
        if B_lz != NULL: free(B_lz)
        if C_lx != NULL: free(C_lx)
        if C_ly != NULL: free(C_ly)
        if C_lz != NULL: free(C_lz)
        if D_lx != NULL: free(D_lx)
        if D_ly != NULL: free(D_ly)
        if D_lz != NULL: free(D_lz)
        if roots != NULL: free(roots)
        if weights != NULL: free(weights)
        if Gx != NULL: free(Gx)
        if Gy != NULL: free(Gy)
        if Gz != NULL: free(Gz)
        return

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]
    cdef double Dx = shell_cxyz_data[shellD * 3 + 0]
    cdef double Dy = shell_cxyz_data[shellD * 3 + 1]
    cdef double Dz = shell_cxyz_data[shellD * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz
    cdef double CDx = Cx - Dx
    cdef double CDy = Cy - Dy
    cdef double CDz = Cz - Dz

    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, max_ij + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz
    xkl_pow[0] = 1.0
    ykl_pow[0] = 1.0
    zkl_pow[0] = 1.0
    for pwr in range(1, max_kl + 1):
        xkl_pow[pwr] = xkl_pow[pwr - 1] * CDx
        ykl_pow[pwr] = ykl_pow[pwr - 1] * CDy
        zkl_pow[pwr] = zkl_pow[pwr - 1] * CDz

    _fill_cart_comp(la, A_lx, A_ly, A_lz)
    _fill_cart_comp(lb, B_lx, B_ly, B_lz)
    _fill_cart_comp(lc, C_lx, C_ly, C_lz)
    _fill_cart_comp(ld, D_lx, D_ly, D_lz)

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int baseCD = <int>sp_pair_start_data[spCD]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimCD = <int>sp_npair_data[spCD]

    cdef int iAB, iCD
    cdef int ia, ib, ic, id, u
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz, idx, idy, idz
    cdef int row, col

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz

    for iAB in range(nprimAB):
        p = pair_eta_data[baseAB + iAB]
        Px = pair_Px_data[baseAB + iAB]
        Py = pair_Py_data[baseAB + iAB]
        Pz = pair_Pz_data[baseAB + iAB]
        cKab = pair_cK_data[baseAB + iAB]

        for iCD in range(nprimCD):
            q = pair_eta_data[baseCD + iCD]
            Qx = pair_Px_data[baseCD + iCD]
            Qy = pair_Py_data[baseCD + iCD]
            Qz = pair_Pz_data[baseCD + iCD]
            cKcd = pair_cK_data[baseCD + iCD]

            denom = p + q
            inv_denom = 1.0 / denom
            dx = Px - Qx
            dy = Py - Qy
            dz = Pz - Qz
            PQ2 = dx * dx + dy * dy + dz * dz
            omega = p * q * inv_denom
            T = omega * PQ2

            base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd
            _rys_roots_weights(nroots, T, roots, weights)

            for u in range(nroots):
                x = roots[u]
                w = weights[u]

                B0 = x * 0.5 * inv_denom
                B1 = (1.0 - x) * 0.5 / p + B0
                B1p = (1.0 - x) * 0.5 / q + B0

                Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                _compute_G_runtime(Gx, nmax, mmax, stride, Cx_, Cpx_, B0, B1, B1p)
                _compute_G_runtime(Gy, nmax, mmax, stride, Cy_, Cpy_, B0, B1, B1p)
                _compute_G_runtime(Gz, nmax, mmax, stride, Cz_, Cpz_, B0, B1, B1p)

                scale = base * w
                for ia in range(nA):
                    iax = <int>A_lx[ia]
                    iay = <int>A_ly[ia]
                    iaz = <int>A_lz[ia]
                    for ib in range(nB):
                        ibx = <int>B_lx[ib]
                        iby = <int>B_ly[ib]
                        ibz = <int>B_lz[ib]
                        row = ia * nB + ib
                        for ic in range(nC):
                            icx = <int>C_lx[ic]
                            icy = <int>C_ly[ic]
                            icz = <int>C_lz[ic]
                            for id in range(nD):
                                idx = <int>D_lx[id]
                                idy = <int>D_ly[id]
                                idz = <int>D_lz[id]
                                col = ic * nD + id
                                Ix = _shift_from_G_runtime(Gx, stride, iax, ibx, icx, idx, xij_pow, xkl_pow)
                                Iy = _shift_from_G_runtime(Gy, stride, iay, iby, icy, idy, yij_pow, ykl_pow)
                                Iz = _shift_from_G_runtime(Gz, stride, iaz, ibz, icz, idz, zij_pow, zkl_pow)
                                out_data[row * nCD + col] += scale * (Ix * Iy * Iz)

    free(xij_pow); free(yij_pow); free(zij_pow)
    free(xkl_pow); free(ykl_pow); free(zkl_pow)
    free(A_lx); free(A_ly); free(A_lz)
    free(B_lx); free(B_ly); free(B_lz)
    free(C_lx); free(C_ly); free(C_lz)
    free(D_lx); free(D_ly); free(D_lz)
    free(roots); free(weights)
    free(Gx); free(Gy); free(Gz)


cdef inline double _eri_rys_spab_diag_max_kernel(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_l_data,
    const cnp.int32_t* sp_A_data,
    const cnp.int32_t* sp_B_data,
    const cnp.int32_t* sp_pair_start_data,
    const cnp.int32_t* sp_npair_data,
    const double* pair_eta_data,
    const double* pair_Px_data,
    const double* pair_Py_data,
    const double* pair_Pz_data,
    const double* pair_cK_data,
    int spAB,
) noexcept nogil:
    """Compute max_{μν in AB} (μν|μν) for one shell pair using pair tables.

    This is a diagonal-only evaluator for the Schwarz bound:
        Q_AB = sqrt( max_{μν} (μν|μν) )
    """

    cdef int shellA = <int>sp_A_data[spAB]
    cdef int shellB = <int>sp_B_data[spAB]

    cdef int la = <int>shell_l_data[shellA]
    cdef int lb = <int>shell_l_data[shellB]

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nAB = nA * nB

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz

    cdef double xij_pow[K_LMAX + 1]
    cdef double yij_pow[K_LMAX + 1]
    cdef double zij_pow[K_LMAX + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int nprimAB = <int>sp_npair_data[spAB]

    # CD == AB for Schwarz diagonal (AB|AB).
    cdef int nmax = la + lb
    cdef int L_total = 2 * (la + lb)
    cdef int nroots = (L_total // 2) + 1

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE]
    cdef double Gy[K_GSIZE]
    cdef double Gz[K_GSIZE]

    cdef double diag[K_NCART_MAX * K_NCART_MAX]
    cdef int row
    for row in range(nAB):
        diag[row] = 0.0

    cdef int iAB, iCD, ia, ib, u
    cdef int iax, iay, iaz, ibx, iby, ibz

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double factor
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz

    for iAB in range(nprimAB):
        p = pair_eta_data[baseAB + iAB]
        Px = pair_Px_data[baseAB + iAB]
        Py = pair_Py_data[baseAB + iAB]
        Pz = pair_Pz_data[baseAB + iAB]
        cKab = pair_cK_data[baseAB + iAB]

        # Primitive-pair symmetry: (iAB,iCD) and (iCD,iAB) contribute equally to (AB|AB).
        for iCD in range(iAB + 1):
            factor = 1.0 if iCD == iAB else 2.0
            q = pair_eta_data[baseAB + iCD]
            Qx = pair_Px_data[baseAB + iCD]
            Qy = pair_Py_data[baseAB + iCD]
            Qz = pair_Pz_data[baseAB + iCD]
            cKcd = pair_cK_data[baseAB + iCD]

            denom = p + q
            inv_denom = 1.0 / denom
            dx = Px - Qx
            dy = Py - Qy
            dz = Pz - Qz
            PQ2 = dx * dx + dy * dy + dz * dz
            omega = p * q * inv_denom
            T = omega * PQ2

            base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

            _rys_roots_weights(nroots, T, &roots[0], &weights[0])

            for u in range(nroots):
                x = roots[u]
                w = weights[u]

                B0 = x * 0.5 * inv_denom
                B1 = (1.0 - x) * 0.5 / p + B0
                B1p = (1.0 - x) * 0.5 / q + B0

                # Here, C == A and D == B (AB|AB).
                Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                Cpx_ = (Qx - Ax) + (p * inv_denom) * x * (Px - Qx)
                Cpy_ = (Qy - Ay) + (p * inv_denom) * x * (Py - Qy)
                Cpz_ = (Qz - Az) + (p * inv_denom) * x * (Pz - Qz)

                _compute_G(&Gx[0], nmax, nmax, Cx_, Cpx_, B0, B1, B1p)
                _compute_G(&Gy[0], nmax, nmax, Cy_, Cpy_, B0, B1, B1p)
                _compute_G(&Gz[0], nmax, nmax, Cz_, Cpz_, B0, B1, B1p)

                scale = base * w * factor

                for ia in range(nA):
                    iax = <int>A_lx[ia]
                    iay = <int>A_ly[ia]
                    iaz = <int>A_lz[ia]
                    for ib in range(nB):
                        ibx = <int>B_lx[ib]
                        iby = <int>B_ly[ib]
                        ibz = <int>B_lz[ib]
                        row = ia * nB + ib

                        Ix = _shift_from_G(&Gx[0], iax, ibx, iax, ibx, &xij_pow[0], &xij_pow[0])
                        Iy = _shift_from_G(&Gy[0], iay, iby, iay, iby, &yij_pow[0], &yij_pow[0])
                        Iz = _shift_from_G(&Gz[0], iaz, ibz, iaz, ibz, &zij_pow[0], &zij_pow[0])
                        diag[row] += scale * (Ix * Iy * Iz)

    cdef double max_diag = 0.0
    cdef double v
    for row in range(nAB):
        v = diag[row]
        if v > max_diag:
            max_diag = v
    if max_diag < 0.0:
        max_diag = 0.0
    return max_diag


def eri_rys_tile_cart_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    int shellA,
    int shellB,
    int shellC,
    int shellD,
):
    """Compute one cartesian ERI tile (AB|CD) using a CPU Rys microkernel (s/p baseline).

    Returns
    - out: float64 array of shape (nA*nB, nC*nD), where nX = ncart(lX)

    Notes
    - This baseline implementation supports only l<=1 per shell (NROOTS<=3).
    - Ordering matches cuERI/PySCF `cart=True` component enumeration.
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nshell = <int>shell_l.shape[0]
    if shellA < 0 or shellA >= nshell or shellB < 0 or shellB >= nshell or shellC < 0 or shellC >= nshell or shellD < 0 or shellD >= nshell:
        raise ValueError("shell indices out of range")

    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    cdef int lc = <int>shell_l[shellC]
    cdef int ld = <int>shell_l[shellD]

    if la > K_LMAX or lb > K_LMAX or lc > K_LMAX or ld > K_LMAX:
        raise NotImplementedError(f"eri_rys_tile_cart_cy currently supports only l<={K_LMAX} per shell")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((nAB, nCD), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]
    cdef double Dx = shell_cxyz_data[shellD * 3 + 0]
    cdef double Dy = shell_cxyz_data[shellD * 3 + 1]
    cdef double Dz = shell_cxyz_data[shellD * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz
    cdef double AB2 = ABx * ABx + ABy * ABy + ABz * ABz

    cdef double CDx = Cx - Dx
    cdef double CDy = Cy - Dy
    cdef double CDz = Cz - Dz
    cdef double CD2 = CDx * CDx + CDy * CDy + CDz * CDz

    cdef double xij_pow[K_LMAX + 1]
    cdef double yij_pow[K_LMAX + 1]
    cdef double zij_pow[K_LMAX + 1]
    cdef double xkl_pow[K_LMAX + 1]
    cdef double ykl_pow[K_LMAX + 1]
    cdef double zkl_pow[K_LMAX + 1]
    cdef int pwr

    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    xkl_pow[0] = 1.0
    ykl_pow[0] = 1.0
    zkl_pow[0] = 1.0
    for pwr in range(1, K_LMAX + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz
        xkl_pow[pwr] = xkl_pow[pwr - 1] * CDx
        ykl_pow[pwr] = ykl_pow[pwr - 1] * CDy
        zkl_pow[pwr] = zkl_pow[pwr - 1] * CDz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    cdef int8_t D_lx[K_NCART_MAX]
    cdef int8_t D_ly[K_NCART_MAX]
    cdef int8_t D_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])
    _fill_cart_comp(ld, &D_lx[0], &D_ly[0], &D_lz[0])

    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sB = <int>shell_prim_start_data[shellB]
    cdef int sC = <int>shell_prim_start_data[shellC]
    cdef int sD = <int>shell_prim_start_data[shellD]
    cdef int nprimA = <int>shell_nprim_data[shellA]
    cdef int nprimB = <int>shell_nprim_data[shellB]
    cdef int nprimC = <int>shell_nprim_data[shellC]
    cdef int nprimD = <int>shell_nprim_data[shellD]

    cdef int nmax = la + lb
    cdef int mmax = lc + ld
    cdef int L_total = la + lb + lc + ld
    cdef int nroots = (L_total // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise NotImplementedError(f"eri_rys_tile_cart_cy currently supports only NROOTS in [1,{K_NROOTS_MAX}]")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE]
    cdef double Gy[K_GSIZE]
    cdef double Gz[K_GSIZE]

    cdef int ipA, ipB, ipC, ipD
    cdef int ia, ib, ic, id, u
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz, idx, idy, idz
    cdef int row, col

    cdef double a, b, cexp, dexp
    cdef double ca, cb, cc, cd
    cdef double p, q, inv_p, inv_q, mu, Kab, Kcd
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz

    with nogil:
        for ipA in range(nprimA):
            a = prim_exp_data[sA + ipA]
            ca = prim_coef_data[sA + ipA]
            for ipB in range(nprimB):
                b = prim_exp_data[sB + ipB]
                cb = prim_coef_data[sB + ipB]

                p = a + b
                inv_p = 1.0 / p
                mu = a * b * inv_p
                Px = (a * Ax + b * Bx) * inv_p
                Py = (a * Ay + b * By) * inv_p
                Pz = (a * Az + b * Bz) * inv_p
                Kab = exp(-mu * AB2)
                cKab = (ca * cb) * Kab

                for ipC in range(nprimC):
                    cexp = prim_exp_data[sC + ipC]
                    cc = prim_coef_data[sC + ipC]
                    for ipD in range(nprimD):
                        dexp = prim_exp_data[sD + ipD]
                        cd = prim_coef_data[sD + ipD]

                        q = cexp + dexp
                        inv_q = 1.0 / q
                        mu = cexp * dexp * inv_q
                        Qx = (cexp * Cx + dexp * Dx) * inv_q
                        Qy = (cexp * Cy + dexp * Dy) * inv_q
                        Qz = (cexp * Cz + dexp * Dz) * inv_q
                        Kcd = exp(-mu * CD2)
                        cKcd = (cc * cd) * Kcd

                        denom = p + q
                        inv_denom = 1.0 / denom
                        dx = Px - Qx
                        dy = Py - Qy
                        dz = Pz - Qz
                        PQ2 = dx * dx + dy * dy + dz * dz
                        omega = p * q * inv_denom
                        T = omega * PQ2

                        base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                        _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                        for u in range(nroots):
                            x = roots[u]
                            w = weights[u]

                            B0 = x * 0.5 * inv_denom
                            B1 = (1.0 - x) * 0.5 / p + B0
                            B1p = (1.0 - x) * 0.5 / q + B0

                            Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                            Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                            Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                            Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                            Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                            Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                            _compute_G(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                            _compute_G(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                            _compute_G(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                            scale = base * w

                            for ia in range(nA):
                                iax = <int>A_lx[ia]
                                iay = <int>A_ly[ia]
                                iaz = <int>A_lz[ia]
                                for ib in range(nB):
                                    ibx = <int>B_lx[ib]
                                    iby = <int>B_ly[ib]
                                    ibz = <int>B_lz[ib]
                                    row = ia * nB + ib
                                    for ic in range(nC):
                                        icx = <int>C_lx[ic]
                                        icy = <int>C_ly[ic]
                                        icz = <int>C_lz[ic]
                                        for id in range(nD):
                                            idx = <int>D_lx[id]
                                            idy = <int>D_ly[id]
                                            idz = <int>D_lz[id]
                                            col = ic * nD + id

                                            Ix = _shift_from_G(&Gx[0], iax, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                            Iy = _shift_from_G(&Gy[0], iay, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                            Iz = _shift_from_G(&Gz[0], iaz, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])
                                            out_data[row * nCD + col] += scale * (Ix * Iy * Iz)

    return out


def eri_rys_tile_cart_batch_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    int shellA,
    int shellB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shellC,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shellD,
):
    """Compute a batch of cartesian ERI tiles (AB|C_t D_t) using the CPU Rys microkernel.

    Returns
    - out: float64 array of shape (nt, nA*nB, nC*nD)

    Notes
    - `shellA/shellB` are fixed across the batch.
    - `shellC/shellD` must have the same length and all tasks must share the same (lc,ld),
      so that `nC*nD` is constant and the output is a dense 3D array.
    - Baseline scope: s/p-only (l<=1, NROOTS<=3).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")
    if shellC.ndim != 1 or shellD.ndim != 1 or <int>shellC.shape[0] != <int>shellD.shape[0]:
        raise ValueError("shellC/shellD must be 1D arrays with identical shape")

    cdef int nshell = <int>shell_l.shape[0]
    if shellA < 0 or shellA >= nshell or shellB < 0 or shellB >= nshell:
        raise ValueError("shellA/shellB out of range")

    cdef int nt = <int>shellC.shape[0]
    if nt == 0:
        return np.empty((0, 0, 0), dtype=np.float64)

    if np.any(shellC < 0) or np.any(shellC >= nshell) or np.any(shellD < 0) or np.any(shellD >= nshell):
        raise ValueError("shellC/shellD entries out of range")

    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    if la > K_LMAX or lb > K_LMAX:
        raise NotImplementedError(f"eri_rys_tile_cart_batch_cy currently supports only l<={K_LMAX} per shell")

    cdef int shellC0 = <int>shellC[0]
    cdef int shellD0 = <int>shellD[0]
    cdef int lc0 = <int>shell_l[shellC0]
    cdef int ld0 = <int>shell_l[shellD0]
    if lc0 > K_LMAX or ld0 > K_LMAX:
        raise NotImplementedError(f"eri_rys_tile_cart_batch_cy currently supports only l<={K_LMAX} per shell")

    cdef int t, csh, dsh
    cdef int lc, ld
    for t in range(nt):
        csh = <int>shellC[t]
        dsh = <int>shellD[t]
        lc = <int>shell_l[csh]
        ld = <int>shell_l[dsh]
        if lc != lc0 or ld != ld0:
            raise ValueError("all tasks in eri_rys_tile_cart_batch_cy must share the same (lc,ld)")
        if lc > K_LMAX or ld > K_LMAX:
            raise NotImplementedError(f"eri_rys_tile_cart_batch_cy currently supports only l<={K_LMAX} per shell")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc0)
    cdef int nD = _ncart(ld0)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, nAB, nCD), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data
    cdef const cnp.int32_t* shell_l_data = <const cnp.int32_t*>shell_l.data
    cdef const cnp.int32_t* shellC_data = <const cnp.int32_t*>shellC.data
    cdef const cnp.int32_t* shellD_data = <const cnp.int32_t*>shellD.data

    cdef Py_ssize_t stride = <Py_ssize_t>(nAB * nCD)

    with nogil:
        for t in range(nt):
            _eri_rys_tile_cart_kernel(
                shell_cxyz_data,
                shell_prim_start_data,
                shell_nprim_data,
                shell_l_data,
                prim_exp_data,
                prim_coef_data,
                shellA,
                shellB,
                <int>shellC_data[t],
                <int>shellD_data[t],
                out_data + (<Py_ssize_t>t) * stride,
            )

    return out


cdef inline void _build_pair_table_sp(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_prim_start_data,
    const cnp.int32_t* shell_nprim_data,
    const double* prim_exp_data,
    const double* prim_coef_data,
    const cnp.int32_t* sp_A_data,
    const cnp.int32_t* sp_B_data,
    const cnp.int32_t* sp_pair_start_data,
    const cnp.int32_t* sp_npair_data,
    double* pair_eta_data,
    double* pair_Px_data,
    double* pair_Py_data,
    double* pair_Pz_data,
    double* pair_cK_data,
    int sp,
) noexcept nogil:
    cdef int A = <int>sp_A_data[sp]
    cdef int B = <int>sp_B_data[sp]
    cdef int base = <int>sp_pair_start_data[sp]
    cdef int npair = <int>sp_npair_data[sp]

    cdef int nprimB = <int>shell_nprim_data[B]
    cdef int startA = <int>shell_prim_start_data[A]
    cdef int startB = <int>shell_prim_start_data[B]

    cdef double Ax = shell_cxyz_data[A * 3 + 0]
    cdef double Ay = shell_cxyz_data[A * 3 + 1]
    cdef double Az = shell_cxyz_data[A * 3 + 2]
    cdef double Bx = shell_cxyz_data[B * 3 + 0]
    cdef double By = shell_cxyz_data[B * 3 + 1]
    cdef double Bz = shell_cxyz_data[B * 3 + 2]

    cdef double dx = Ax - Bx
    cdef double dy = Ay - By
    cdef double dz = Az - Bz
    cdef double AB2 = dx * dx + dy * dy + dz * dz

    cdef int idx, ia, ib, pA, pB
    cdef double alpha, beta, eta, inv_eta, Px, Py, Pz, mu, Kab, cK

    for idx in range(npair):
        ia = idx // nprimB
        ib = idx - ia * nprimB
        pA = startA + ia
        pB = startB + ib

        alpha = prim_exp_data[pA]
        beta = prim_exp_data[pB]
        eta = alpha + beta
        inv_eta = 1.0 / eta
        Px = (alpha * Ax + beta * Bx) * inv_eta
        Py = (alpha * Ay + beta * By) * inv_eta
        Pz = (alpha * Az + beta * Bz) * inv_eta

        mu = (alpha * beta) * inv_eta
        Kab = exp(-mu * AB2)
        cK = prim_coef_data[pA] * prim_coef_data[pB] * Kab

        pair_eta_data[base + idx] = eta
        pair_Px_data[base + idx] = Px
        pair_Py_data[base + idx] = Py
        pair_Pz_data[base + idx] = Pz
        pair_cK_data[base + idx] = cK


def build_pair_tables_cart_inplace_cpu(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_coef,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int threads=0,
):
    """Build primitive-pair tables for all shell pairs (CPU).

    The layout follows `sp_pair_start/sp_npair`, and the primitive-pair ordering matches the
    CUDA implementation: `idx = ia*nprimB + ib` (B fastest).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int n_shell = <int>shell_cxyz.shape[0]
    if (<int>shell_prim_start.shape[0]) != n_shell or (<int>shell_nprim.shape[0]) != n_shell:
        raise ValueError("shell_prim_start/shell_nprim must have shape (nShell,)")

    cdef int nsp = <int>sp_A.shape[0]
    if (<int>sp_B.shape[0]) != nsp or (<int>sp_npair.shape[0]) != nsp:
        raise ValueError("sp_A/sp_B/sp_npair must have shape (nSP,)")
    if (<int>sp_pair_start.shape[0]) != (nsp + 1):
        raise ValueError("sp_pair_start must have shape (nSP+1,)")
    if int(sp_pair_start[0]) != 0:
        raise ValueError("sp_pair_start[0] must be 0")
    if int(sp_pair_start[nsp]) != int(pair_eta.shape[0]):
        raise ValueError("pair_* arrays must have shape (sp_pair_start[-1],)")
    # Note: For typed `cnp.ndarray` objects, `.shape` is a `npy_intp*` pointer, so
    # comparing `arr.shape != other.shape` compares pointers, not dimensions.
    if (
        int(pair_Px.shape[0]) != int(pair_eta.shape[0])
        or int(pair_Py.shape[0]) != int(pair_eta.shape[0])
        or int(pair_Pz.shape[0]) != int(pair_eta.shape[0])
        or int(pair_cK.shape[0]) != int(pair_eta.shape[0])
    ):
        raise ValueError("pair_* arrays must have identical length")

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* prim_coef_data = <const double*>prim_coef.data
    cdef const cnp.int32_t* sp_A_data = <const cnp.int32_t*>sp_A.data
    cdef const cnp.int32_t* sp_B_data = <const cnp.int32_t*>sp_B.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data

    cdef double* pair_eta_data = <double*>pair_eta.data
    cdef double* pair_Px_data = <double*>pair_Px.data
    cdef double* pair_Py_data = <double*>pair_Py.data
    cdef double* pair_Pz_data = <double*>pair_Pz.data
    cdef double* pair_cK_data = <double*>pair_cK.data

    cdef int threads_i = <int>threads
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    cdef int use_threads = threads_i
    if use_threads > nsp:
        use_threads = nsp

    cdef int sp
    with nogil:
        if use_threads > 1:
            for sp in prange(nsp, schedule="static", num_threads=use_threads):
                _build_pair_table_sp(
                    shell_cxyz_data,
                    shell_prim_start_data,
                    shell_nprim_data,
                    prim_exp_data,
                    prim_coef_data,
                    sp_A_data,
                    sp_B_data,
                    sp_pair_start_data,
                    sp_npair_data,
                    pair_eta_data,
                    pair_Px_data,
                    pair_Py_data,
                    pair_Pz_data,
                    pair_cK_data,
                    sp,
                )
        else:
            for sp in range(nsp):
                _build_pair_table_sp(
                    shell_cxyz_data,
                    shell_prim_start_data,
                    shell_nprim_data,
                    prim_exp_data,
                    prim_coef_data,
                    sp_A_data,
                    sp_B_data,
                    sp_pair_start_data,
                    sp_npair_data,
                    pair_eta_data,
                    pair_Px_data,
                    pair_Py_data,
                    pair_Pz_data,
                    pair_cK_data,
                    sp,
                )

    return None


def eri_rys_tile_cart_sp_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    int spCD,
):
    """Compute one cartesian ERI tile (spAB|spCD) using prebuilt primitive-pair tables."""

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp or spCD < 0 or spCD >= nsp:
        raise ValueError("spAB/spCD out of range")

    cdef int A = <int>sp_A[spAB]
    cdef int B = <int>sp_B[spAB]
    cdef int C = <int>sp_A[spCD]
    cdef int D = <int>sp_B[spCD]

    cdef int la = <int>shell_l[A]
    cdef int lb = <int>shell_l[B]
    cdef int lc = <int>shell_l[C]
    cdef int ld = <int>shell_l[D]
    cdef int nroots = ((la + lb + lc + ld) // 2) + 1
    cdef bint use_dynamic = (la > K_LMAX or lb > K_LMAX or lc > K_LMAX or ld > K_LMAX or nroots > K_NROOTS_MAX)

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((nAB, nCD), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    with nogil:
        if use_dynamic:
            _eri_rys_tile_cart_sp_kernel_dynamic(
                <const double*>shell_cxyz.data,
                <const cnp.int32_t*>shell_l.data,
                <const cnp.int32_t*>sp_A.data,
                <const cnp.int32_t*>sp_B.data,
                <const cnp.int32_t*>sp_pair_start.data,
                <const cnp.int32_t*>sp_npair.data,
                <const double*>pair_eta.data,
                <const double*>pair_Px.data,
                <const double*>pair_Py.data,
                <const double*>pair_Pz.data,
                <const double*>pair_cK.data,
                spAB,
                spCD,
                out_data,
            )
        else:
            _eri_rys_tile_cart_sp_kernel(
                <const double*>shell_cxyz.data,
                <const cnp.int32_t*>shell_l.data,
                <const cnp.int32_t*>sp_A.data,
                <const cnp.int32_t*>sp_B.data,
                <const cnp.int32_t*>sp_pair_start.data,
                <const cnp.int32_t*>sp_npair.data,
                <const double*>pair_eta.data,
                <const double*>pair_Px.data,
                <const double*>pair_Py.data,
                <const double*>pair_Pz.data,
                <const double*>pair_cK.data,
                spAB,
                spCD,
                out_data,
            )

    return out


def eri_rys_tile_cart_sp_batch_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spCD,
    int threads=0,
):
    """Compute a batch of cartesian ERI tiles (spAB|spCD[t]) using primitive-pair tables.

    Returns
    - out: float64 array of shape (nt, nA*nB, nC*nD)
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp:
        raise ValueError("spAB out of range")

    cdef int nt = <int>spCD.shape[0]
    if nt == 0:
        return np.empty((0, 0, 0), dtype=np.float64)

    if np.any(spCD < 0) or np.any(spCD >= nsp):
        raise ValueError("spCD entries out of range")

    cdef int A = <int>sp_A[spAB]
    cdef int B = <int>sp_B[spAB]
    cdef int la = <int>shell_l[A]
    cdef int lb = <int>shell_l[B]

    cdef int spCD0 = <int>spCD[0]
    cdef int C0 = <int>sp_A[spCD0]
    cdef int D0 = <int>sp_B[spCD0]
    cdef int lc0 = <int>shell_l[C0]
    cdef int ld0 = <int>shell_l[D0]

    cdef int t, sp_i, Ci, Di, lc, ld
    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        Di = <int>sp_B[sp_i]
        lc = <int>shell_l[Ci]
        ld = <int>shell_l[Di]
        if lc != lc0 or ld != ld0:
            raise ValueError("all spCD tasks in eri_rys_tile_cart_sp_batch_cy must share the same (lc,ld)")

    cdef int nroots = ((la + lb + lc0 + ld0) // 2) + 1
    cdef bint use_dynamic = (la > K_LMAX or lb > K_LMAX or lc0 > K_LMAX or ld0 > K_LMAX or nroots > K_NROOTS_MAX)

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc0)
    cdef int nD = _ncart(ld0)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, nAB, nCD), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef Py_ssize_t stride = <Py_ssize_t>(nAB * nCD)

    cdef int threads_i = <int>threads
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    with nogil:
        if threads_i > 1 and nt > 1:
            for t in prange(nt, schedule="static", num_threads=threads_i):
                if use_dynamic:
                    _eri_rys_tile_cart_sp_kernel_dynamic(
                        <const double*>shell_cxyz.data,
                        <const cnp.int32_t*>shell_l.data,
                        <const cnp.int32_t*>sp_A.data,
                        <const cnp.int32_t*>sp_B.data,
                        <const cnp.int32_t*>sp_pair_start.data,
                        <const cnp.int32_t*>sp_npair.data,
                        <const double*>pair_eta.data,
                        <const double*>pair_Px.data,
                        <const double*>pair_Py.data,
                        <const double*>pair_Pz.data,
                        <const double*>pair_cK.data,
                        spAB,
                        <int>(<const cnp.int32_t*>spCD.data)[t],
                        out_data + (<Py_ssize_t>t) * stride,
                    )
                else:
                    _eri_rys_tile_cart_sp_kernel(
                        <const double*>shell_cxyz.data,
                        <const cnp.int32_t*>shell_l.data,
                        <const cnp.int32_t*>sp_A.data,
                        <const cnp.int32_t*>sp_B.data,
                        <const cnp.int32_t*>sp_pair_start.data,
                        <const cnp.int32_t*>sp_npair.data,
                        <const double*>pair_eta.data,
                        <const double*>pair_Px.data,
                        <const double*>pair_Py.data,
                        <const double*>pair_Pz.data,
                        <const double*>pair_cK.data,
                        spAB,
                        <int>(<const cnp.int32_t*>spCD.data)[t],
                        out_data + (<Py_ssize_t>t) * stride,
                    )
        else:
            for t in range(nt):
                if use_dynamic:
                    _eri_rys_tile_cart_sp_kernel_dynamic(
                        <const double*>shell_cxyz.data,
                        <const cnp.int32_t*>shell_l.data,
                        <const cnp.int32_t*>sp_A.data,
                        <const cnp.int32_t*>sp_B.data,
                        <const cnp.int32_t*>sp_pair_start.data,
                        <const cnp.int32_t*>sp_npair.data,
                        <const double*>pair_eta.data,
                        <const double*>pair_Px.data,
                        <const double*>pair_Py.data,
                        <const double*>pair_Pz.data,
                        <const double*>pair_cK.data,
                        spAB,
                        <int>(<const cnp.int32_t*>spCD.data)[t],
                        out_data + (<Py_ssize_t>t) * stride,
                    )
                else:
                    _eri_rys_tile_cart_sp_kernel(
                        <const double*>shell_cxyz.data,
                        <const cnp.int32_t*>shell_l.data,
                        <const cnp.int32_t*>sp_A.data,
                        <const cnp.int32_t*>sp_B.data,
                        <const cnp.int32_t*>sp_pair_start.data,
                        <const cnp.int32_t*>sp_npair.data,
                        <const double*>pair_eta.data,
                        <const double*>pair_Px.data,
                        <const double*>pair_Py.data,
                        <const double*>pair_Pz.data,
                        <const double*>pair_cK.data,
                        spAB,
                        <int>(<const cnp.int32_t*>spCD.data)[t],
                        out_data + (<Py_ssize_t>t) * stride,
                    )

    return out


def schwarz_shellpairs_cart_sp_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int threads=0,
):
    """Compute rigorous Schwarz bounds Q_AB for all shell pairs using pair tables.

    For each shell pair AB:
      Q_AB = sqrt( max_{μ in A, ν in B} (μν|μν) )
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if (<int>sp_B.shape[0]) != nsp or (<int>sp_npair.shape[0]) != nsp:
        raise ValueError("sp_A/sp_B/sp_npair must have shape (nSP,)")
    if (<int>sp_pair_start.shape[0]) != (nsp + 1):
        raise ValueError("sp_pair_start must have shape (nSP+1,)")
    if int(sp_pair_start[0]) != 0:
        raise ValueError("sp_pair_start[0] must be 0")

    cdef int total_pair_prims = int(sp_pair_start[nsp])
    if total_pair_prims != int(pair_eta.shape[0]):
        raise ValueError("pair_* arrays must have shape (sp_pair_start[-1],)")
    if (
        int(pair_Px.shape[0]) != total_pair_prims
        or int(pair_Py.shape[0]) != total_pair_prims
        or int(pair_Pz.shape[0]) != total_pair_prims
        or int(pair_cK.shape[0]) != total_pair_prims
    ):
        raise ValueError("pair_* arrays must have identical length")

    cdef int threads_i = <int>threads
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    cdef int nshell = <int>shell_l.shape[0]
    cdef int sh
    cdef bint has_high_l = False
    for sh in range(nshell):
        if <int>shell_l[sh] > K_LMAX:
            has_high_l = True
            break

    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty((nsp,), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef cnp.ndarray[cnp.double_t, ndim=2] tile
    cdef int sp_i
    cdef double max_diag

    if has_high_l:
        for sp_i in range(nsp):
            tile = eri_rys_tile_cart_sp_cy(
                shell_cxyz,
                shell_l,
                sp_A,
                sp_B,
                sp_pair_start,
                sp_npair,
                pair_eta,
                pair_Px,
                pair_Py,
                pair_Pz,
                pair_cK,
                sp_i,
                sp_i,
            )
            if tile.size == 0:
                out[sp_i] = 0.0
                continue
            max_diag = float(np.max(np.diag(tile)))
            out[sp_i] = sqrt(max_diag) if max_diag > 0.0 else 0.0
        return out

    cdef int use_threads = threads_i
    if use_threads > nsp:
        use_threads = nsp

    cdef int sp
    with nogil:
        if use_threads > 1:
            for sp in prange(nsp, schedule="static", num_threads=use_threads):
                max_diag = _eri_rys_spab_diag_max_kernel(
                    <const double*>shell_cxyz.data,
                    <const cnp.int32_t*>shell_l.data,
                    <const cnp.int32_t*>sp_A.data,
                    <const cnp.int32_t*>sp_B.data,
                    <const cnp.int32_t*>sp_pair_start.data,
                    <const cnp.int32_t*>sp_npair.data,
                    <const double*>pair_eta.data,
                    <const double*>pair_Px.data,
                    <const double*>pair_Py.data,
                    <const double*>pair_Pz.data,
                    <const double*>pair_cK.data,
                    sp,
                )
                out_data[sp] = sqrt(max_diag) if max_diag > 0.0 else 0.0
        else:
            for sp in range(nsp):
                max_diag = _eri_rys_spab_diag_max_kernel(
                    <const double*>shell_cxyz.data,
                    <const cnp.int32_t*>shell_l.data,
                    <const cnp.int32_t*>sp_A.data,
                    <const cnp.int32_t*>sp_B.data,
                    <const cnp.int32_t*>sp_pair_start.data,
                    <const cnp.int32_t*>sp_npair.data,
                    <const double*>pair_eta.data,
                    <const double*>pair_Px.data,
                    <const double*>pair_Py.data,
                    <const double*>pair_Pz.data,
                    <const double*>pair_cK.data,
                    sp,
                )
                out_data[sp] = sqrt(max_diag) if max_diag > 0.0 else 0.0

    return out


def rys_roots_weights_cy(nroots: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Rys quadrature roots/weights for a given `nroots` and `T`.

    This is primarily intended for debugging/validation of the CPU Rys implementation.
    """

    cdef int nroots_i = int(nroots)
    cdef double T_d = float(T)
    if nroots_i < 1:
        raise ValueError("nroots must be >= 1")
    if T_d < 0.0:
        raise ValueError("T must be >= 0")

    cdef double* r = <double*>malloc(nroots_i * sizeof(double))
    cdef double* w = <double*>malloc(nroots_i * sizeof(double))
    cdef int i
    if r == NULL or w == NULL:
        if r != NULL:
            free(r)
        if w != NULL:
            free(w)
        raise MemoryError("failed to allocate Rys roots/weights buffers")

    with nogil:
        _rys_roots_weights(nroots_i, T_d, r, w)

    roots = np.empty((nroots_i,), dtype=np.float64)
    weights = np.empty((nroots_i,), dtype=np.float64)
    for i in range(nroots_i):
        roots[i] = r[i]
        weights[i] = w[i]
    free(r)
    free(w)
    return roots, weights


def df_int3c2e_deriv_contracted_cart_sp_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    int spCD,
    int nao,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] bar_X_flat,
):
    """Contract d(μν|P)/dR against `bar_X_flat` for one (spAB|spCD) DF tile (ld=0).

    Parameters
    ----------
    shell_ao_start:
        In the combined DF basis (AO + aux + dummy), aux shells use an AO-start offset of +nao.
        This function subtracts `nao` to index the aux dimension of `bar_X_flat`.
    bar_X_flat:
        Adjoint w.r.t. unwhitened 3c2e integrals X, shape (nao*nao, naux).

    Returns
    -------
    out:
        float64 array of shape (3,3) where rows correspond to (A,B,C) centers and columns to (x,y,z).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp or spCD < 0 or spCD >= nsp:
        raise ValueError("spAB/spCD out of range")

    cdef int shellA = <int>sp_A[spAB]
    cdef int shellB = <int>sp_B[spAB]
    cdef int shellC = <int>sp_A[spCD]
    cdef int shellD = <int>sp_B[spCD]

    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    cdef int lc = <int>shell_l[shellC]
    cdef int ld = <int>shell_l[shellD]
    if la > K_LMAX or lb > K_LMAX or lc > K_LMAX or ld > K_LMAX:
        raise NotImplementedError(f"df_int3c2e_deriv_contracted_cart_sp_cy supports only l<={K_LMAX} per shell")
    if ld != 0:
        raise ValueError("df_int3c2e_deriv_contracted_cart_sp_cy requires ld=0 (DF 3c2e with dummy s-shell)")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nAB = nA * nB

    nao = int(nao)
    if nao <= 0:
        raise ValueError("nao must be > 0")
    cdef int naux = <int>bar_X_flat.shape[1]
    if (<int>bar_X_flat.shape[0]) != (nao * nao):
        raise ValueError("bar_X_flat must have shape (nao*nao, naux)")

    cdef int c0 = <int>shell_ao_start[shellC] - <int>nao
    if c0 < 0 or (c0 + nC) > naux:
        raise ValueError("aux index range mismatch between shell_ao_start and bar_X_flat")

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((3, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* pair_eta_data = <const double*>pair_eta.data
    cdef const double* pair_Px_data = <const double*>pair_Px.data
    cdef const double* pair_Py_data = <const double*>pair_Py.data
    cdef const double* pair_Pz_data = <const double*>pair_Pz.data
    cdef const double* pair_cK_data = <const double*>pair_cK.data
    cdef const double* bar_data = <const double*>bar_X_flat.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX_D + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int baseCD = <int>sp_pair_start_data[spCD]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimCD = <int>sp_npair_data[spCD]

    cdef int nprimB = <int>shell_nprim_data[shellB]
    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sB = <int>shell_prim_start_data[shellB]
    cdef int sC = <int>shell_prim_start_data[shellC]

    cdef int a0 = <int>shell_ao_start_data[shellA]
    cdef int b0 = <int>shell_ao_start_data[shellB]

    cdef int nmax = la + lb + 1
    cdef int mmax = lc + 1
    cdef int L_total = la + lb + lc  # ld=0
    cdef int nroots = ((L_total + 1) // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise RuntimeError("unsupported nroots in df_int3c2e_deriv_contracted_cart_sp_cy")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE_D]
    cdef double Gy[K_GSIZE_D]
    cdef double Gz[K_GSIZE_D]

    cdef int iAB, iCD
    cdef int ia, ib, ic
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz
    cdef int row_local
    cdef int row_idx, col_idx

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double aexp, bexp, cexp
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_

    cdef double Ix, Iy, Iz
    cdef double Ix_p, Ix_m, dIx
    cdef double Iy_p, Iy_m, dIy
    cdef double Iz_p, Iz_m, dIz
    cdef double bar

    with nogil:
        for iAB in range(nprimAB):
            p = pair_eta_data[baseAB + iAB]
            Px = pair_Px_data[baseAB + iAB]
            Py = pair_Py_data[baseAB + iAB]
            Pz = pair_Pz_data[baseAB + iAB]
            cKab = pair_cK_data[baseAB + iAB]

            ia = iAB // nprimB
            ib = iAB - ia * nprimB
            aexp = prim_exp_data[sA + ia]
            bexp = prim_exp_data[sB + ib]

            for iCD in range(nprimCD):
                q = pair_eta_data[baseCD + iCD]
                Qx = pair_Px_data[baseCD + iCD]
                Qy = pair_Py_data[baseCD + iCD]
                Qz = pair_Pz_data[baseCD + iCD]
                cKcd = pair_cK_data[baseCD + iCD]
                cexp = prim_exp_data[sC + iCD]

                denom = p + q
                inv_denom = 1.0 / denom
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                omega = p * q * inv_denom
                T = omega * PQ2

                base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                for u in range(nroots):
                    x = roots[u]
                    w = weights[u]

                    B0 = x * 0.5 * inv_denom
                    B1 = (1.0 - x) * 0.5 / p + B0
                    B1p = (1.0 - x) * 0.5 / q + B0

                    Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                    Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                    Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                    Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                    Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                    Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                    _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                    _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                    _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                    scale = base * w

                    for ia in range(nA):
                        iax = <int>A_lx[ia]
                        iay = <int>A_ly[ia]
                        iaz = <int>A_lz[ia]
                        for ib in range(nB):
                            ibx = <int>B_lx[ib]
                            iby = <int>B_ly[ib]
                            ibz = <int>B_lz[ib]
                            row_local = ia * nB + ib
                            row_idx = (a0 + ia) * nao + (b0 + ib)
                            for ic in range(nC):
                                icx = <int>C_lx[ic]
                                icy = <int>C_ly[ic]
                                icz = <int>C_lz[ic]
                                col_idx = c0 + ic
                                bar = bar_data[row_idx * naux + col_idx]
                                if bar == 0.0:
                                    continue

                                Ix = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx, &xij_pow[0])
                                Iy = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy, &yij_pow[0])
                                Iz = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz, &zij_pow[0])

                                # Center A derivatives
                                Ix_m = _shift_from_G_ld0_d(&Gx[0], iax - 1, ibx, icx, &xij_pow[0]) if iax > 0 else 0.0
                                Ix_p = _shift_from_G_ld0_d(&Gx[0], iax + 1, ibx, icx, &xij_pow[0])
                                dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                                out_data[0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = _shift_from_G_ld0_d(&Gy[0], iay - 1, iby, icy, &yij_pow[0]) if iay > 0 else 0.0
                                Iy_p = _shift_from_G_ld0_d(&Gy[0], iay + 1, iby, icy, &yij_pow[0])
                                dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                                out_data[0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz - 1, ibz, icz, &zij_pow[0]) if iaz > 0 else 0.0
                                Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz + 1, ibz, icz, &zij_pow[0])
                                dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                                out_data[0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # Center B derivatives
                                Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx - 1, icx, &xij_pow[0]) if ibx > 0 else 0.0
                                Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx + 1, icx, &xij_pow[0])
                                dIx = (-<double>ibx) * Ix_m + (2.0 * bexp) * Ix_p
                                out_data[1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby - 1, icy, &yij_pow[0]) if iby > 0 else 0.0
                                Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby + 1, icy, &yij_pow[0])
                                dIy = (-<double>iby) * Iy_m + (2.0 * bexp) * Iy_p
                                out_data[1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz - 1, icz, &zij_pow[0]) if ibz > 0 else 0.0
                                Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz + 1, icz, &zij_pow[0])
                                dIz = (-<double>ibz) * Iz_m + (2.0 * bexp) * Iz_p
                                out_data[1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # Center C derivatives (aux)
                                Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx - 1, &xij_pow[0]) if icx > 0 else 0.0
                                Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx + 1, &xij_pow[0])
                                dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                                out_data[2 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy - 1, &yij_pow[0]) if icy > 0 else 0.0
                                Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy + 1, &yij_pow[0])
                                dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                                out_data[2 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz - 1, &zij_pow[0]) if icz > 0 else 0.0
                                Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz + 1, &zij_pow[0])
                                dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                                out_data[2 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

    return out


def df_int3c2e_deriv_contracted_cart_sp_batch_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spCD,
    int nao,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] bar_X_flat,
):
    """Batch version of :func:`df_int3c2e_deriv_contracted_cart_sp_cy` over spCD tasks.

    Returns
    -------
    out:
        float64 array of shape (nt, 3, 3) where nt = len(spCD). Each entry uses the same center ordering
        (A,B,C) and coordinate ordering (x,y,z) as the single-tile function.

    Notes
    -----
    - All spCD tasks must share the same (lc,ld) so the output tile shape is uniform.
    - Intended for DF 3c2e where ld=0 (dummy s-shell).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp:
        raise ValueError("spAB out of range")

    cdef int nt = <int>spCD.shape[0]
    if nt == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    if np.any(spCD < 0) or np.any(spCD >= nsp):
        raise ValueError("spCD entries out of range")

    cdef int shellA = <int>sp_A[spAB]
    cdef int shellB = <int>sp_B[spAB]
    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    if la > K_LMAX or lb > K_LMAX:
        raise NotImplementedError(f"df_int3c2e_deriv_contracted_cart_sp_batch_cy supports only l<={K_LMAX} per shell")

    nao = int(nao)
    if nao <= 0:
        raise ValueError("nao must be > 0")
    cdef int naux = <int>bar_X_flat.shape[1]
    if (<int>bar_X_flat.shape[0]) != (nao * nao):
        raise ValueError("bar_X_flat must have shape (nao*nao, naux)")

    cdef int spCD0 = <int>spCD[0]
    cdef int shellC0 = <int>sp_A[spCD0]
    cdef int shellD0 = <int>sp_B[spCD0]
    cdef int lc0 = <int>shell_l[shellC0]
    cdef int ld0 = <int>shell_l[shellD0]
    if lc0 > K_LMAX or ld0 > K_LMAX:
        raise NotImplementedError(f"df_int3c2e_deriv_contracted_cart_sp_batch_cy supports only l<={K_LMAX} per shell")
    if ld0 != 0:
        raise ValueError("df_int3c2e_deriv_contracted_cart_sp_batch_cy requires ld=0")

    cdef int t, sp_i, Ci, Di, lc, ld
    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        Di = <int>sp_B[sp_i]
        lc = <int>shell_l[Ci]
        ld = <int>shell_l[Di]
        if lc != lc0 or ld != ld0:
            raise ValueError("all spCD tasks in df_int3c2e_deriv_contracted_cart_sp_batch_cy must share the same (lc,ld)")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc0)

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, 3, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef Py_ssize_t stride_out = <Py_ssize_t>(9)  # 3*3 per task

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* pair_eta_data = <const double*>pair_eta.data
    cdef const double* pair_Px_data = <const double*>pair_Px.data
    cdef const double* pair_Py_data = <const double*>pair_Py.data
    cdef const double* pair_Pz_data = <const double*>pair_Pz.data
    cdef const double* pair_cK_data = <const double*>pair_cK.data
    cdef const double* bar_data = <const double*>bar_X_flat.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX_D + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc0, &C_lx[0], &C_ly[0], &C_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimB = <int>shell_nprim_data[shellB]
    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sB = <int>shell_prim_start_data[shellB]

    cdef int a0 = <int>shell_ao_start_data[shellA]
    cdef int b0 = <int>shell_ao_start_data[shellB]

    cdef int nmax = la + lb + 1
    cdef int mmax = lc0 + 1
    cdef int L_total = la + lb + lc0  # ld=0
    cdef int nroots = ((L_total + 1) // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise RuntimeError("unsupported nroots in df_int3c2e_deriv_contracted_cart_sp_batch_cy")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE_D]
    cdef double Gy[K_GSIZE_D]
    cdef double Gz[K_GSIZE_D]

    cdef int iAB, iCD, u
    cdef int ia, ib, ic
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz
    cdef int row_idx, col_idx, c0, row_out
    cdef int baseCD, nprimCD, sC

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double aexp, bexp, cexp
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Cx, Cy, Cz

    cdef double Ix, Iy, Iz
    cdef double Ix_p, Ix_m, dIx
    cdef double Iy_p, Iy_m, dIy
    cdef double Iz_p, Iz_m, dIz
    cdef double bar

    # Precompute shellC-dependent arrays for each task: baseCD, nprimCD, sC, c0, and C center coords.
    cdef cnp.ndarray[cnp.int32_t, ndim=1] baseCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] nprimCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sC_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] c0_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.double_t, ndim=2] Cxyz_arr = np.empty((nt, 3), dtype=np.float64)

    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        baseCD_arr[t] = <cnp.int32_t>sp_pair_start_data[sp_i]
        nprimCD_arr[t] = <cnp.int32_t>sp_npair_data[sp_i]
        sC_arr[t] = <cnp.int32_t>shell_prim_start_data[Ci]
        c0_arr[t] = <cnp.int32_t>(<int>shell_ao_start_data[Ci] - nao)
        Cxyz_arr[t, 0] = shell_cxyz_data[Ci * 3 + 0]
        Cxyz_arr[t, 1] = shell_cxyz_data[Ci * 3 + 1]
        Cxyz_arr[t, 2] = shell_cxyz_data[Ci * 3 + 2]
        if (<int>c0_arr[t]) < 0 or (<int>c0_arr[t] + nC) > naux:
            raise ValueError("aux index range mismatch between shell_ao_start and bar_X_flat (batch)")

    cdef cnp.int32_t* baseCD_ptr = <cnp.int32_t*>baseCD_arr.data
    cdef cnp.int32_t* nprimCD_ptr = <cnp.int32_t*>nprimCD_arr.data
    cdef cnp.int32_t* sC_ptr = <cnp.int32_t*>sC_arr.data
    cdef cnp.int32_t* c0_ptr = <cnp.int32_t*>c0_arr.data
    cdef double* Cxyz_ptr = <double*>Cxyz_arr.data

    with nogil:
        for t in range(nt):
            row_out = <int>(t * stride_out)
            baseCD = <int>baseCD_ptr[t]
            nprimCD = <int>nprimCD_ptr[t]
            sC = <int>sC_ptr[t]
            c0 = <int>c0_ptr[t]
            Cx = Cxyz_ptr[t * 3 + 0]
            Cy = Cxyz_ptr[t * 3 + 1]
            Cz = Cxyz_ptr[t * 3 + 2]

            for iAB in range(nprimAB):
                p = pair_eta_data[baseAB + iAB]
                Px = pair_Px_data[baseAB + iAB]
                Py = pair_Py_data[baseAB + iAB]
                Pz = pair_Pz_data[baseAB + iAB]
                cKab = pair_cK_data[baseAB + iAB]

                ia = iAB // nprimB
                ib = iAB - ia * nprimB
                aexp = prim_exp_data[sA + ia]
                bexp = prim_exp_data[sB + ib]

                for iCD in range(nprimCD):
                    q = pair_eta_data[baseCD + iCD]
                    Qx = pair_Px_data[baseCD + iCD]
                    Qy = pair_Py_data[baseCD + iCD]
                    Qz = pair_Pz_data[baseCD + iCD]
                    cKcd = pair_cK_data[baseCD + iCD]
                    cexp = prim_exp_data[sC + iCD]

                    denom = p + q
                    inv_denom = 1.0 / denom
                    dx = Px - Qx
                    dy = Py - Qy
                    dz = Pz - Qz
                    PQ2 = dx * dx + dy * dy + dz * dz
                    omega = p * q * inv_denom
                    T = omega * PQ2

                    base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                    _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                    for u in range(nroots):
                        x = roots[u]
                        w = weights[u]

                        B0 = x * 0.5 * inv_denom
                        B1 = (1.0 - x) * 0.5 / p + B0
                        B1p = (1.0 - x) * 0.5 / q + B0

                        Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                        Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                        Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                        Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                        Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                        Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                        _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                        _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                        _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                        scale = base * w

                        for ia in range(nA):
                            iax = <int>A_lx[ia]
                            iay = <int>A_ly[ia]
                            iaz = <int>A_lz[ia]
                            for ib in range(nB):
                                ibx = <int>B_lx[ib]
                                iby = <int>B_ly[ib]
                                ibz = <int>B_lz[ib]
                                row_idx = (a0 + ia) * nao + (b0 + ib)
                                for ic in range(nC):
                                    icx = <int>C_lx[ic]
                                    icy = <int>C_ly[ic]
                                    icz = <int>C_lz[ic]
                                    col_idx = c0 + ic
                                    bar = bar_data[row_idx * naux + col_idx]
                                    if bar == 0.0:
                                        continue

                                    Ix = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx, &xij_pow[0])
                                    Iy = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy, &yij_pow[0])
                                    Iz = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz, &zij_pow[0])

                                    # Center A derivatives
                                    Ix_m = _shift_from_G_ld0_d(&Gx[0], iax - 1, ibx, icx, &xij_pow[0]) if iax > 0 else 0.0
                                    Ix_p = _shift_from_G_ld0_d(&Gx[0], iax + 1, ibx, icx, &xij_pow[0])
                                    dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                                    out_data[row_out + 0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                    Iy_m = _shift_from_G_ld0_d(&Gy[0], iay - 1, iby, icy, &yij_pow[0]) if iay > 0 else 0.0
                                    Iy_p = _shift_from_G_ld0_d(&Gy[0], iay + 1, iby, icy, &yij_pow[0])
                                    dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                                    out_data[row_out + 0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                    Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz - 1, ibz, icz, &zij_pow[0]) if iaz > 0 else 0.0
                                    Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz + 1, ibz, icz, &zij_pow[0])
                                    dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                                    out_data[row_out + 0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                    # Center B derivatives
                                    Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx - 1, icx, &xij_pow[0]) if ibx > 0 else 0.0
                                    Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx + 1, icx, &xij_pow[0])
                                    dIx = (-<double>ibx) * Ix_m + (2.0 * bexp) * Ix_p
                                    out_data[row_out + 1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                    Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby - 1, icy, &yij_pow[0]) if iby > 0 else 0.0
                                    Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby + 1, icy, &yij_pow[0])
                                    dIy = (-<double>iby) * Iy_m + (2.0 * bexp) * Iy_p
                                    out_data[row_out + 1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                    Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz - 1, icz, &zij_pow[0]) if ibz > 0 else 0.0
                                    Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz + 1, icz, &zij_pow[0])
                                    dIz = (-<double>ibz) * Iz_m + (2.0 * bexp) * Iz_p
                                    out_data[row_out + 1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                    # Center C derivatives (aux)
                                    Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx - 1, &xij_pow[0]) if icx > 0 else 0.0
                                    Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx + 1, &xij_pow[0])
                                    dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                                    out_data[row_out + 2 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                    Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy - 1, &yij_pow[0]) if icy > 0 else 0.0
                                    Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy + 1, &yij_pow[0])
                                    dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                                    out_data[row_out + 2 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                    Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz - 1, &zij_pow[0]) if icz > 0 else 0.0
                                    Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz + 1, &zij_pow[0])
                                    dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                                    out_data[row_out + 2 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

    return out


def df_symmetrize_qmn_inplace_cy(
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] x_flat,
    int naux,
    int nao,
):
    """In-place symmetrization for a Qmn-layout tensor.

    Parameters
    ----------
    x_flat
        Flat view of ``x[Q,m,n]`` with total length ``naux*nao*nao``.
    naux
        Number of auxiliary functions.
    nao
        Number of (working) AO functions on m/n indices.
    """
    naux = int(naux)
    nao = int(nao)
    if naux < 0 or nao < 0:
        raise ValueError("naux and nao must be >= 0")
    cdef Py_ssize_t expected = (<Py_ssize_t>naux) * (<Py_ssize_t>nao) * (<Py_ssize_t>nao)
    if (<Py_ssize_t>x_flat.shape[0]) != expected:
        raise ValueError("x_flat length mismatch with naux*nao*nao")

    cdef double* data = <double*>x_flat.data
    cdef Py_ssize_t stride_q = (<Py_ssize_t>nao) * (<Py_ssize_t>nao)
    cdef Py_ssize_t stride_m = <Py_ssize_t>nao
    cdef int q, m, n
    cdef Py_ssize_t base, idx_mn, idx_nm
    cdef double avg

    with nogil:
        for q in range(naux):
            base = (<Py_ssize_t>q) * stride_q
            for m in range(nao):
                for n in range(m + 1, nao):
                    idx_mn = base + (<Py_ssize_t>m) * stride_m + (<Py_ssize_t>n)
                    idx_nm = base + (<Py_ssize_t>n) * stride_m + (<Py_ssize_t>m)
                    avg = 0.5 * (data[idx_mn] + data[idx_nm])
                    data[idx_mn] = avg
                    data[idx_nm] = avg


def df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spCD,
    int nao_cart,
    int naux,
    int nao_sph,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] bar_X_sph_Qmn_flat,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start_sph_all,
):
    """DF 3c2e derivative contraction with spherical bar_X in Qmn layout.

    This variant avoids materializing a full Cartesian ``bar_X_flat`` tensor.
    It transforms spherical ``bar_X_sph[Q,mu_sph,nu_sph]`` to the needed
    Cartesian shell-pair block on-the-fly using cart2sph tables.
    """
    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp:
        raise ValueError("spAB out of range")

    cdef int nt = <int>spCD.shape[0]
    if nt == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    if np.any(spCD < 0) or np.any(spCD >= nsp):
        raise ValueError("spCD entries out of range")

    cdef int shellA = <int>sp_A[spAB]
    cdef int shellB = <int>sp_B[spAB]
    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    if la > K_LMAX or lb > K_LMAX:
        raise NotImplementedError(f"df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy supports only l<={K_LMAX} per shell")
    if la > K_SPH_LMAX or lb > K_SPH_LMAX:
        raise NotImplementedError(
            f"df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy supports only l<={K_SPH_LMAX} for cart2sph (got la={la}, lb={lb})"
        )

    nao_cart = int(nao_cart)
    if nao_cart <= 0:
        raise ValueError("nao_cart must be > 0")
    naux = int(naux)
    if naux <= 0:
        raise ValueError("naux must be > 0")
    nao_sph = int(nao_sph)
    if nao_sph <= 0:
        raise ValueError("nao_sph must be > 0")

    cdef Py_ssize_t expected_bar = (<Py_ssize_t>naux) * (<Py_ssize_t>nao_sph) * (<Py_ssize_t>nao_sph)
    if (<Py_ssize_t>bar_X_sph_Qmn_flat.shape[0]) != expected_bar:
        raise ValueError("bar_X_sph_Qmn_flat length mismatch with naux*nao_sph*nao_sph")
    if (<Py_ssize_t>shell_ao_start_sph_all.shape[0]) != (<Py_ssize_t>shell_ao_start.shape[0]):
        raise ValueError("shell_ao_start_sph_all length must match shell_ao_start length")

    cdef int spCD0 = <int>spCD[0]
    cdef int shellC0 = <int>sp_A[spCD0]
    cdef int shellD0 = <int>sp_B[spCD0]
    cdef int lc0 = <int>shell_l[shellC0]
    cdef int ld0 = <int>shell_l[shellD0]
    if lc0 > K_LMAX or ld0 > K_LMAX:
        raise NotImplementedError(f"df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy supports only l<={K_LMAX} per shell")
    if ld0 != 0:
        raise ValueError("df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy requires ld=0")

    cdef int t, sp_i, Ci, Di, lc, ld
    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        Di = <int>sp_B[sp_i]
        lc = <int>shell_l[Ci]
        ld = <int>shell_l[Di]
        if lc != lc0 or ld != ld0:
            raise ValueError("all spCD tasks must share the same (lc,ld)")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc0)
    cdef int nsphA = _nsph(la)
    cdef int nsphB = _nsph(lb)

    _init_cart2sph_sp_tables()

    cdef const double* TA = _cart2sph_sp_ptr(la)
    cdef const double* TB = _cart2sph_sp_ptr(lb)
    if TA == <const double*>0 or TB == <const double*>0:  # pragma: no cover
        raise NotImplementedError("cart2sph table missing for requested l")

    cdef const cnp.int32_t* shell_ao_start_sph_data = <const cnp.int32_t*>shell_ao_start_sph_all.data
    cdef int a0_sph = <int>shell_ao_start_sph_data[shellA]
    cdef int b0_sph = <int>shell_ao_start_sph_data[shellB]
    if a0_sph < 0 or b0_sph < 0:
        raise ValueError("invalid spherical AO offsets")
    if (a0_sph + nsphA) > nao_sph or (b0_sph + nsphB) > nao_sph:
        raise ValueError("spherical AO offset out of range for nao_sph")

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, 3, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef Py_ssize_t stride_out = <Py_ssize_t>(9)  # 3*3 per task

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* pair_eta_data = <const double*>pair_eta.data
    cdef const double* pair_Px_data = <const double*>pair_Px.data
    cdef const double* pair_Py_data = <const double*>pair_Py.data
    cdef const double* pair_Pz_data = <const double*>pair_Pz.data
    cdef const double* pair_cK_data = <const double*>pair_cK.data
    cdef const double* bar_sph_data = <const double*>bar_X_sph_Qmn_flat.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Bx = shell_cxyz_data[shellB * 3 + 0]
    cdef double By = shell_cxyz_data[shellB * 3 + 1]
    cdef double Bz = shell_cxyz_data[shellB * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX_D + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc0, &C_lx[0], &C_ly[0], &C_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimB = <int>shell_nprim_data[shellB]
    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sB = <int>shell_prim_start_data[shellB]

    cdef int nmax = la + lb + 1
    cdef int mmax = lc0 + 1
    cdef int L_total = la + lb + lc0  # ld=0
    cdef int nroots = ((L_total + 1) // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise RuntimeError("unsupported nroots")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE_D]
    cdef double Gy[K_GSIZE_D]
    cdef double Gz[K_GSIZE_D]

    cdef cnp.ndarray[cnp.int32_t, ndim=1] baseCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] nprimCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sC_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] c0_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.double_t, ndim=2] Cxyz_arr = np.empty((nt, 3), dtype=np.float64)

    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        baseCD_arr[t] = <cnp.int32_t>sp_pair_start_data[sp_i]
        nprimCD_arr[t] = <cnp.int32_t>sp_npair_data[sp_i]
        sC_arr[t] = <cnp.int32_t>shell_prim_start_data[Ci]
        c0_arr[t] = <cnp.int32_t>(<int>shell_ao_start_data[Ci] - nao_cart)
        Cxyz_arr[t, 0] = shell_cxyz_data[Ci * 3 + 0]
        Cxyz_arr[t, 1] = shell_cxyz_data[Ci * 3 + 1]
        Cxyz_arr[t, 2] = shell_cxyz_data[Ci * 3 + 2]
        if (<int>c0_arr[t]) < 0 or (<int>c0_arr[t] + nC) > naux:
            raise ValueError("aux index range mismatch between shell_ao_start and bar_X_sph")

    cdef cnp.int32_t* baseCD_ptr = <cnp.int32_t*>baseCD_arr.data
    cdef cnp.int32_t* nprimCD_ptr = <cnp.int32_t*>nprimCD_arr.data
    cdef cnp.int32_t* sC_ptr = <cnp.int32_t*>sC_arr.data
    cdef cnp.int32_t* c0_ptr = <cnp.int32_t*>c0_arr.data
    cdef double* Cxyz_ptr = <double*>Cxyz_arr.data

    cdef Py_ssize_t nao_sph_sq = (<Py_ssize_t>nao_sph) * (<Py_ssize_t>nao_sph)
    cdef double tmpAB[K_NCART_MAX * K_NSPH_MAX]
    cdef Py_ssize_t bar_cart_size = (<Py_ssize_t>nC) * (<Py_ssize_t>nA) * (<Py_ssize_t>nB)
    cdef double* bar_cart_buf = <double*>malloc(bar_cart_size * sizeof(double))
    if bar_cart_buf == NULL:
        raise MemoryError("failed to allocate bar_cart_buf")

    cdef int row_out
    cdef int baseCD, nprimCD, sC, c0, Q
    cdef double Cx, Cy, Cz
    cdef int iAB, iCD, u
    cdef int ia, ib, ic
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz
    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double aexp, bexp, cexp
    cdef double denom, inv_denom, Tval, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz
    cdef double Ix_p, Ix_m, dIx
    cdef double Iy_p, Iy_m, dIy
    cdef double Iz_p, Iz_m, dIz
    cdef double bar
    cdef int ia_cart, ib_cart, isphA, jsphB
    cdef double s

    try:
        with nogil:
            for t in range(nt):
                row_out = <int>(t * stride_out)
                baseCD = <int>baseCD_ptr[t]
                nprimCD = <int>nprimCD_ptr[t]
                sC = <int>sC_ptr[t]
                c0 = <int>c0_ptr[t]
                Cx = Cxyz_ptr[t * 3 + 0]
                Cy = Cxyz_ptr[t * 3 + 1]
                Cz = Cxyz_ptr[t * 3 + 2]

                # Precompute bar_cart_buf for this aux shell task t.
                for ic in range(nC):
                    Q = c0 + ic
                    for ia_cart in range(nA):
                        for jsphB in range(nsphB):
                            s = 0.0
                            for isphA in range(nsphA):
                                s += TA[ia_cart * nsphA + isphA] * bar_sph_data[
                                    (<Py_ssize_t>Q) * nao_sph_sq
                                    + (<Py_ssize_t>(a0_sph + isphA)) * (<Py_ssize_t>nao_sph)
                                    + (<Py_ssize_t>(b0_sph + jsphB))
                                ]
                            tmpAB[ia_cart * K_NSPH_MAX + jsphB] = s
                    for ia_cart in range(nA):
                        for ib_cart in range(nB):
                            s = 0.0
                            for jsphB in range(nsphB):
                                s += tmpAB[ia_cart * K_NSPH_MAX + jsphB] * TB[ib_cart * nsphB + jsphB]
                            bar_cart_buf[(<Py_ssize_t>ic) * (<Py_ssize_t>nA) * (<Py_ssize_t>nB) + (<Py_ssize_t>ia_cart) * (<Py_ssize_t>nB) + (<Py_ssize_t>ib_cart)] = s

                for iAB in range(nprimAB):
                    p = pair_eta_data[baseAB + iAB]
                    Px = pair_Px_data[baseAB + iAB]
                    Py = pair_Py_data[baseAB + iAB]
                    Pz = pair_Pz_data[baseAB + iAB]
                    cKab = pair_cK_data[baseAB + iAB]

                    ia = iAB // nprimB
                    ib = iAB - ia * nprimB
                    aexp = prim_exp_data[sA + ia]
                    bexp = prim_exp_data[sB + ib]

                    for iCD in range(nprimCD):
                        q = pair_eta_data[baseCD + iCD]
                        Qx = pair_Px_data[baseCD + iCD]
                        Qy = pair_Py_data[baseCD + iCD]
                        Qz = pair_Pz_data[baseCD + iCD]
                        cKcd = pair_cK_data[baseCD + iCD]
                        cexp = prim_exp_data[sC + iCD]

                        denom = p + q
                        inv_denom = 1.0 / denom
                        dx = Px - Qx
                        dy = Py - Qy
                        dz = Pz - Qz
                        Tval = (p * q) * inv_denom * (dx * dx + dy * dy + dz * dz)

                        base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd
                        _rys_roots_weights(nroots, Tval, &roots[0], &weights[0])

                        for u in range(nroots):
                            x = roots[u]
                            w = weights[u]

                            B0 = x * 0.5 * inv_denom
                            B1 = (1.0 - x) * 0.5 / p + B0
                            B1p = (1.0 - x) * 0.5 / q + B0

                            Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                            Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                            Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                            Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                            Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                            Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                            _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                            _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                            _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                            scale = base * w

                            for ia in range(nA):
                                iax = <int>A_lx[ia]
                                iay = <int>A_ly[ia]
                                iaz = <int>A_lz[ia]
                                for ib in range(nB):
                                    ibx = <int>B_lx[ib]
                                    iby = <int>B_ly[ib]
                                    ibz = <int>B_lz[ib]
                                    for ic in range(nC):
                                        bar = bar_cart_buf[(<Py_ssize_t>ic) * (<Py_ssize_t>nA) * (<Py_ssize_t>nB) + (<Py_ssize_t>ia) * (<Py_ssize_t>nB) + (<Py_ssize_t>ib)]
                                        if bar == 0.0:
                                            continue
                                        icx = <int>C_lx[ic]
                                        icy = <int>C_ly[ic]
                                        icz = <int>C_lz[ic]

                                        Ix = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx, &xij_pow[0])
                                        Iy = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy, &yij_pow[0])
                                        Iz = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz, &zij_pow[0])

                                        # Center A derivatives
                                        Ix_m = _shift_from_G_ld0_d(&Gx[0], iax - 1, ibx, icx, &xij_pow[0]) if iax > 0 else 0.0
                                        Ix_p = _shift_from_G_ld0_d(&Gx[0], iax + 1, ibx, icx, &xij_pow[0])
                                        dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                                        out_data[row_out + 0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                        Iy_m = _shift_from_G_ld0_d(&Gy[0], iay - 1, iby, icy, &yij_pow[0]) if iay > 0 else 0.0
                                        Iy_p = _shift_from_G_ld0_d(&Gy[0], iay + 1, iby, icy, &yij_pow[0])
                                        dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                                        out_data[row_out + 0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                        Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz - 1, ibz, icz, &zij_pow[0]) if iaz > 0 else 0.0
                                        Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz + 1, ibz, icz, &zij_pow[0])
                                        dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                                        out_data[row_out + 0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                        # Center B derivatives
                                        Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx - 1, icx, &xij_pow[0]) if ibx > 0 else 0.0
                                        Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx + 1, icx, &xij_pow[0])
                                        dIx = (-<double>ibx) * Ix_m + (2.0 * bexp) * Ix_p
                                        out_data[row_out + 1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                        Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby - 1, icy, &yij_pow[0]) if iby > 0 else 0.0
                                        Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby + 1, icy, &yij_pow[0])
                                        dIy = (-<double>iby) * Iy_m + (2.0 * bexp) * Iy_p
                                        out_data[row_out + 1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                        Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz - 1, icz, &zij_pow[0]) if ibz > 0 else 0.0
                                        Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz + 1, icz, &zij_pow[0])
                                        dIz = (-<double>ibz) * Iz_m + (2.0 * bexp) * Iz_p
                                        out_data[row_out + 1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                        # Center C derivatives (aux)
                                        Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx - 1, &xij_pow[0]) if icx > 0 else 0.0
                                        Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, ibx, icx + 1, &xij_pow[0])
                                        dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                                        out_data[row_out + 2 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                        Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy - 1, &yij_pow[0]) if icy > 0 else 0.0
                                        Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, iby, icy + 1, &yij_pow[0])
                                        dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                                        out_data[row_out + 2 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                        Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz - 1, &zij_pow[0]) if icz > 0 else 0.0
                                        Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, ibz, icz + 1, &zij_pow[0])
                                        dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                                        out_data[row_out + 2 * 3 + 2] += bar * scale * (Ix * Iy * dIz)
    finally:
        free(bar_cart_buf)

    return out


def df_metric_2c2e_deriv_contracted_cart_sp_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    int spCD,
    int nao,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] bar_V,
):
    """Contract d(P|Q)/dR against `bar_V` for one DF metric tile (P*1|Q*1).

    Returns a float64 array of shape (2,3) for centers (A=P, C=Q) and coordinates (x,y,z).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp or spCD < 0 or spCD >= nsp:
        raise ValueError("spAB/spCD out of range")

    cdef int shellA = <int>sp_A[spAB]
    cdef int shellB = <int>sp_B[spAB]
    cdef int shellC = <int>sp_A[spCD]
    cdef int shellD = <int>sp_B[spCD]

    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    cdef int lc = <int>shell_l[shellC]
    cdef int ld = <int>shell_l[shellD]
    if la > K_LMAX or lb > K_LMAX or lc > K_LMAX or ld > K_LMAX:
        raise NotImplementedError(f"df_metric_2c2e_deriv_contracted_cart_sp_cy supports only l<={K_LMAX} per shell")
    if lb != 0 or ld != 0:
        raise ValueError("df_metric_2c2e_deriv_contracted_cart_sp_cy requires lb=ld=0 (dummy s-shells)")

    cdef int nA = _ncart(la)
    cdef int nC = _ncart(lc)

    nao = int(nao)
    if nao < 0:
        raise ValueError("nao must be >= 0")
    cdef int naux = <int>bar_V.shape[0]
    if bar_V.ndim != 2 or (<int>bar_V.shape[1]) != naux:
        raise ValueError("bar_V must be square")

    cdef int a0 = <int>shell_ao_start[shellA] - <int>nao
    cdef int c0 = <int>shell_ao_start[shellC] - <int>nao
    if a0 < 0 or (a0 + nA) > naux or c0 < 0 or (c0 + nC) > naux:
        raise ValueError("aux index range mismatch between shell_ao_start and bar_V")

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((2, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_nprim_data = <const cnp.int32_t*>shell_nprim.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* pair_eta_data = <const double*>pair_eta.data
    cdef const double* pair_Px_data = <const double*>pair_Px.data
    cdef const double* pair_Py_data = <const double*>pair_Py.data
    cdef const double* pair_Pz_data = <const double*>pair_Pz.data
    cdef const double* pair_cK_data = <const double*>pair_cK.data
    cdef const double* bar_data = <const double*>bar_V.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]
    cdef double Cx = shell_cxyz_data[shellC * 3 + 0]
    cdef double Cy = shell_cxyz_data[shellC * 3 + 1]
    cdef double Cz = shell_cxyz_data[shellC * 3 + 2]

    cdef double ABx = Ax  # dummy at origin
    cdef double ABy = Ay
    cdef double ABz = Az

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX_D + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int baseCD = <int>sp_pair_start_data[spCD]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int nprimCD = <int>sp_npair_data[spCD]

    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int sC = <int>shell_prim_start_data[shellC]

    cdef int nmax = la + 1  # lb=0
    cdef int mmax = lc + 1  # ld=0
    cdef int L_total = la + lc
    cdef int nroots = ((L_total + 1) // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise RuntimeError("unsupported nroots in df_metric_2c2e_deriv_contracted_cart_sp_cy")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]

    cdef double Gx[K_GSIZE_D]
    cdef double Gy[K_GSIZE_D]
    cdef double Gz[K_GSIZE_D]

    cdef int iAB, iCD
    cdef int ia, ic
    cdef int iax, iay, iaz, icx, icy, icz
    cdef int row_idx, col_idx

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double aexp, cexp
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_

    cdef double Ix, Iy, Iz
    cdef double Ix_p, Ix_m, dIx
    cdef double Iy_p, Iy_m, dIy
    cdef double Iz_p, Iz_m, dIz
    cdef double bar

    with nogil:
        for iAB in range(nprimAB):
            p = pair_eta_data[baseAB + iAB]
            Px = pair_Px_data[baseAB + iAB]
            Py = pair_Py_data[baseAB + iAB]
            Pz = pair_Pz_data[baseAB + iAB]
            cKab = pair_cK_data[baseAB + iAB]
            aexp = prim_exp_data[sA + iAB]  # nprimB=1

            for iCD in range(nprimCD):
                q = pair_eta_data[baseCD + iCD]
                Qx = pair_Px_data[baseCD + iCD]
                Qy = pair_Py_data[baseCD + iCD]
                Qz = pair_Pz_data[baseCD + iCD]
                cKcd = pair_cK_data[baseCD + iCD]
                cexp = prim_exp_data[sC + iCD]

                denom = p + q
                inv_denom = 1.0 / denom
                dx = Px - Qx
                dy = Py - Qy
                dz = Pz - Qz
                PQ2 = dx * dx + dy * dy + dz * dz
                omega = p * q * inv_denom
                T = omega * PQ2

                base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                for u in range(nroots):
                    x = roots[u]
                    w = weights[u]

                    B0 = x * 0.5 * inv_denom
                    B1 = (1.0 - x) * 0.5 / p + B0
                    B1p = (1.0 - x) * 0.5 / q + B0

                    Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                    Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                    Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                    Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                    Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                    Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                    _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                    _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                    _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                    scale = base * w

                    for ia in range(nA):
                        iax = <int>A_lx[ia]
                        iay = <int>A_ly[ia]
                        iaz = <int>A_lz[ia]
                        row_idx = a0 + ia
                        for ic in range(nC):
                            icx = <int>C_lx[ic]
                            icy = <int>C_ly[ic]
                            icz = <int>C_lz[ic]
                            col_idx = c0 + ic
                            bar = bar_data[row_idx * naux + col_idx]
                            if bar == 0.0:
                                continue

                            Ix = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx, &xij_pow[0])
                            Iy = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy, &yij_pow[0])
                            Iz = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz, &zij_pow[0])

                            # Center A (P) derivatives
                            Ix_m = _shift_from_G_ld0_d(&Gx[0], iax - 1, 0, icx, &xij_pow[0]) if iax > 0 else 0.0
                            Ix_p = _shift_from_G_ld0_d(&Gx[0], iax + 1, 0, icx, &xij_pow[0])
                            dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                            out_data[0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                            Iy_m = _shift_from_G_ld0_d(&Gy[0], iay - 1, 0, icy, &yij_pow[0]) if iay > 0 else 0.0
                            Iy_p = _shift_from_G_ld0_d(&Gy[0], iay + 1, 0, icy, &yij_pow[0])
                            dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                            out_data[0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                            Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz - 1, 0, icz, &zij_pow[0]) if iaz > 0 else 0.0
                            Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz + 1, 0, icz, &zij_pow[0])
                            dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                            out_data[0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                            # Center C (Q) derivatives
                            Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx - 1, &xij_pow[0]) if icx > 0 else 0.0
                            Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx + 1, &xij_pow[0])
                            dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                            out_data[1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                            Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy - 1, &yij_pow[0]) if icy > 0 else 0.0
                            Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy + 1, &yij_pow[0])
                            dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                            out_data[1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                            Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz - 1, &zij_pow[0]) if icz > 0 else 0.0
                            Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz + 1, &zij_pow[0])
                            dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                            out_data[1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

    return out


def df_metric_2c2e_deriv_contracted_cart_sp_batch_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_ao_start,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spCD,
    int nao,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] bar_V,
):
    """Batch version of :func:`df_metric_2c2e_deriv_contracted_cart_sp_cy` over spCD tasks.

    Returns a float64 array of shape (nt, 2, 3) for centers (A=P, C=Q) and coordinates (x,y,z).
    """

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp:
        raise ValueError("spAB out of range")

    cdef int nt = <int>spCD.shape[0]
    if nt == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    if np.any(spCD < 0) or np.any(spCD >= nsp):
        raise ValueError("spCD entries out of range")

    cdef int shellA = <int>sp_A[spAB]
    cdef int shellB = <int>sp_B[spAB]
    cdef int la = <int>shell_l[shellA]
    cdef int lb = <int>shell_l[shellB]
    if la > K_LMAX or lb > K_LMAX:
        raise NotImplementedError(f"df_metric_2c2e_deriv_contracted_cart_sp_batch_cy supports only l<={K_LMAX} per shell")
    if lb != 0:
        raise ValueError("df_metric_2c2e_deriv_contracted_cart_sp_batch_cy requires lb=0")

    nao = int(nao)
    if nao < 0:
        raise ValueError("nao must be >= 0")
    cdef int naux = <int>bar_V.shape[0]
    if bar_V.ndim != 2 or (<int>bar_V.shape[1]) != naux:
        raise ValueError("bar_V must be square")

    cdef int spCD0 = <int>spCD[0]
    cdef int shellC0 = <int>sp_A[spCD0]
    cdef int shellD0 = <int>sp_B[spCD0]
    cdef int lc0 = <int>shell_l[shellC0]
    cdef int ld0 = <int>shell_l[shellD0]
    if lc0 > K_LMAX or ld0 > K_LMAX:
        raise NotImplementedError(f"df_metric_2c2e_deriv_contracted_cart_sp_batch_cy supports only l<={K_LMAX} per shell")
    if ld0 != 0:
        raise ValueError("df_metric_2c2e_deriv_contracted_cart_sp_batch_cy requires ld=0")

    cdef int t, sp_i, Ci, Di, lc, ld
    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        Di = <int>sp_B[sp_i]
        lc = <int>shell_l[Ci]
        ld = <int>shell_l[Di]
        if lc != lc0 or ld != ld0:
            raise ValueError("all spCD tasks in df_metric_2c2e_deriv_contracted_cart_sp_batch_cy must share the same (lc,ld)")

    cdef int nA = _ncart(la)
    cdef int nC = _ncart(lc0)

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, 2, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef Py_ssize_t stride_out = <Py_ssize_t>(6)  # 2*3 per task

    cdef const double* shell_cxyz_data = <const double*>shell_cxyz.data
    cdef const cnp.int32_t* shell_prim_start_data = <const cnp.int32_t*>shell_prim_start.data
    cdef const cnp.int32_t* shell_ao_start_data = <const cnp.int32_t*>shell_ao_start.data
    cdef const cnp.int32_t* sp_pair_start_data = <const cnp.int32_t*>sp_pair_start.data
    cdef const cnp.int32_t* sp_npair_data = <const cnp.int32_t*>sp_npair.data
    cdef const double* prim_exp_data = <const double*>prim_exp.data
    cdef const double* pair_eta_data = <const double*>pair_eta.data
    cdef const double* pair_Px_data = <const double*>pair_Px.data
    cdef const double* pair_Py_data = <const double*>pair_Py.data
    cdef const double* pair_Pz_data = <const double*>pair_Pz.data
    cdef const double* pair_cK_data = <const double*>pair_cK.data
    cdef const double* bar_data = <const double*>bar_V.data

    cdef double Ax = shell_cxyz_data[shellA * 3 + 0]
    cdef double Ay = shell_cxyz_data[shellA * 3 + 1]
    cdef double Az = shell_cxyz_data[shellA * 3 + 2]

    cdef double ABx = Ax  # dummy at origin
    cdef double ABy = Ay
    cdef double ABz = Az

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef int pwr
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    for pwr in range(1, K_LMAX_D + 1):
        xij_pow[pwr] = xij_pow[pwr - 1] * ABx
        yij_pow[pwr] = yij_pow[pwr - 1] * ABy
        zij_pow[pwr] = zij_pow[pwr - 1] * ABz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lc0, &C_lx[0], &C_ly[0], &C_lz[0])

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int nprimAB = <int>sp_npair_data[spAB]
    cdef int sA = <int>shell_prim_start_data[shellA]
    cdef int a0 = <int>shell_ao_start_data[shellA] - nao
    if a0 < 0 or (a0 + nA) > naux:
        raise ValueError("aux index range mismatch between shell_ao_start and bar_V (batch)")

    cdef int nmax = la + 1  # lb=0
    cdef int mmax = lc0 + 1
    cdef int L_total = la + lc0
    cdef int nroots = ((L_total + 1) // 2) + 1
    if nroots < 1 or nroots > K_NROOTS_MAX:
        raise RuntimeError("unsupported nroots in df_metric_2c2e_deriv_contracted_cart_sp_batch_cy")

    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]
    cdef double Gx[K_GSIZE_D]
    cdef double Gy[K_GSIZE_D]
    cdef double Gz[K_GSIZE_D]

    cdef int iAB, iCD, u
    cdef int ia, ic
    cdef int iax, iay, iaz, icx, icy, icz
    cdef int row_idx, col_idx, c0, row_out
    cdef int baseCD, nprimCD, sC

    cdef double p, q
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double cKab, cKcd
    cdef double aexp, cexp
    cdef double denom, inv_denom, omega, T, PQ2, dx, dy, dz
    cdef double base, scale
    cdef double x, w
    cdef double B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Cx, Cy, Cz

    cdef double Ix, Iy, Iz
    cdef double Ix_p, Ix_m, dIx
    cdef double Iy_p, Iy_m, dIy
    cdef double Iz_p, Iz_m, dIz
    cdef double bar

    cdef cnp.ndarray[cnp.int32_t, ndim=1] baseCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] nprimCD_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sC_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] c0_arr = np.empty((nt,), dtype=np.int32)
    cdef cnp.ndarray[cnp.double_t, ndim=2] Cxyz_arr = np.empty((nt, 3), dtype=np.float64)

    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        baseCD_arr[t] = <cnp.int32_t>sp_pair_start_data[sp_i]
        nprimCD_arr[t] = <cnp.int32_t>sp_npair_data[sp_i]
        sC_arr[t] = <cnp.int32_t>shell_prim_start_data[Ci]
        c0_arr[t] = <cnp.int32_t>(<int>shell_ao_start_data[Ci] - nao)
        Cxyz_arr[t, 0] = shell_cxyz_data[Ci * 3 + 0]
        Cxyz_arr[t, 1] = shell_cxyz_data[Ci * 3 + 1]
        Cxyz_arr[t, 2] = shell_cxyz_data[Ci * 3 + 2]
        if (<int>c0_arr[t]) < 0 or (<int>c0_arr[t] + nC) > naux:
            raise ValueError("aux index range mismatch between shell_ao_start and bar_V (batch)")

    cdef cnp.int32_t* baseCD_ptr = <cnp.int32_t*>baseCD_arr.data
    cdef cnp.int32_t* nprimCD_ptr = <cnp.int32_t*>nprimCD_arr.data
    cdef cnp.int32_t* sC_ptr = <cnp.int32_t*>sC_arr.data
    cdef cnp.int32_t* c0_ptr = <cnp.int32_t*>c0_arr.data
    cdef double* Cxyz_ptr = <double*>Cxyz_arr.data

    with nogil:
        for t in range(nt):
            row_out = <int>(t * stride_out)
            baseCD = <int>baseCD_ptr[t]
            nprimCD = <int>nprimCD_ptr[t]
            sC = <int>sC_ptr[t]
            c0 = <int>c0_ptr[t]
            Cx = Cxyz_ptr[t * 3 + 0]
            Cy = Cxyz_ptr[t * 3 + 1]
            Cz = Cxyz_ptr[t * 3 + 2]

            for iAB in range(nprimAB):
                p = pair_eta_data[baseAB + iAB]
                Px = pair_Px_data[baseAB + iAB]
                Py = pair_Py_data[baseAB + iAB]
                Pz = pair_Pz_data[baseAB + iAB]
                cKab = pair_cK_data[baseAB + iAB]
                aexp = prim_exp_data[sA + iAB]

                for iCD in range(nprimCD):
                    q = pair_eta_data[baseCD + iCD]
                    Qx = pair_Px_data[baseCD + iCD]
                    Qy = pair_Py_data[baseCD + iCD]
                    Qz = pair_Pz_data[baseCD + iCD]
                    cKcd = pair_cK_data[baseCD + iCD]
                    cexp = prim_exp_data[sC + iCD]

                    denom = p + q
                    inv_denom = 1.0 / denom
                    dx = Px - Qx
                    dy = Py - Qy
                    dz = Pz - Qz
                    PQ2 = dx * dx + dy * dy + dz * dz
                    omega = p * q * inv_denom
                    T = omega * PQ2

                    base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

                    _rys_roots_weights(nroots, T, &roots[0], &weights[0])

                    for u in range(nroots):
                        x = roots[u]
                        w = weights[u]

                        B0 = x * 0.5 * inv_denom
                        B1 = (1.0 - x) * 0.5 / p + B0
                        B1p = (1.0 - x) * 0.5 / q + B0

                        Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                        Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                        Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                        Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                        Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                        Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                        _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                        _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                        _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                        scale = base * w

                        for ia in range(nA):
                            iax = <int>A_lx[ia]
                            iay = <int>A_ly[ia]
                            iaz = <int>A_lz[ia]
                            row_idx = a0 + ia
                            for ic in range(nC):
                                icx = <int>C_lx[ic]
                                icy = <int>C_ly[ic]
                                icz = <int>C_lz[ic]
                                col_idx = c0 + ic
                                bar = bar_data[row_idx * naux + col_idx]
                                if bar == 0.0:
                                    continue

                                Ix = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx, &xij_pow[0])
                                Iy = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy, &yij_pow[0])
                                Iz = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz, &zij_pow[0])

                                # Center A derivatives
                                Ix_m = _shift_from_G_ld0_d(&Gx[0], iax - 1, 0, icx, &xij_pow[0]) if iax > 0 else 0.0
                                Ix_p = _shift_from_G_ld0_d(&Gx[0], iax + 1, 0, icx, &xij_pow[0])
                                dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                                out_data[row_out + 0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = _shift_from_G_ld0_d(&Gy[0], iay - 1, 0, icy, &yij_pow[0]) if iay > 0 else 0.0
                                Iy_p = _shift_from_G_ld0_d(&Gy[0], iay + 1, 0, icy, &yij_pow[0])
                                dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                                out_data[row_out + 0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz - 1, 0, icz, &zij_pow[0]) if iaz > 0 else 0.0
                                Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz + 1, 0, icz, &zij_pow[0])
                                dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                                out_data[row_out + 0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # Center C derivatives
                                Ix_m = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx - 1, &xij_pow[0]) if icx > 0 else 0.0
                                Ix_p = _shift_from_G_ld0_d(&Gx[0], iax, 0, icx + 1, &xij_pow[0])
                                dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                                out_data[row_out + 1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy - 1, &yij_pow[0]) if icy > 0 else 0.0
                                Iy_p = _shift_from_G_ld0_d(&Gy[0], iay, 0, icy + 1, &yij_pow[0])
                                dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                                out_data[row_out + 1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz - 1, &zij_pow[0]) if icz > 0 else 0.0
                                Iz_p = _shift_from_G_ld0_d(&Gz[0], iaz, 0, icz + 1, &zij_pow[0])
                                dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                                out_data[row_out + 1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

    return out


cdef inline void _eri_rys_deriv_contracted_cart_sp_kernel(
    const double* shell_cxyz_data,
    const cnp.int32_t* shell_l_data,
    const cnp.int32_t* shell_prim_start_data,
    const cnp.int32_t* shell_nprim_data,
    const double* prim_exp_data,
    const cnp.int32_t* sp_A_data,
    const cnp.int32_t* sp_B_data,
    const cnp.int32_t* sp_pair_start_data,
    const cnp.int32_t* sp_npair_data,
    const double* pair_eta_data,
    const double* pair_Px_data,
    const double* pair_Py_data,
    const double* pair_Pz_data,
    const double* pair_cK_data,
    int spAB,
    int spCD,
    const double* bar_data,  # shape (nAB*nCD,)
    double* out_data,        # shape (12,)
) noexcept nogil:
    # Contract analytic 4c ERI nuclear derivatives against a per-tile adjoint `bar_data`.
    #
    # Computes (d/dR_center_coord) (mu nu | la si) contracted with bar(mu nu, la si)
    # for all 4 centers (A,B,C,D) and 3 Cartesian components.
    #
    # Layout:
    #   out_data[(center*3 + xyz)] with center order [A,B,C,D] and xyz order [x,y,z].

    cdef int A = <int>sp_A_data[spAB]
    cdef int B = <int>sp_B_data[spAB]
    cdef int C = <int>sp_A_data[spCD]
    cdef int D = <int>sp_B_data[spCD]

    cdef int la = <int>shell_l_data[A]
    cdef int lb = <int>shell_l_data[B]
    cdef int lc = <int>shell_l_data[C]
    cdef int ld = <int>shell_l_data[D]

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    cdef double Ax = shell_cxyz_data[A * 3 + 0]
    cdef double Ay = shell_cxyz_data[A * 3 + 1]
    cdef double Az = shell_cxyz_data[A * 3 + 2]
    cdef double Bx = shell_cxyz_data[B * 3 + 0]
    cdef double By = shell_cxyz_data[B * 3 + 1]
    cdef double Bz = shell_cxyz_data[B * 3 + 2]
    cdef double Cx = shell_cxyz_data[C * 3 + 0]
    cdef double Cy = shell_cxyz_data[C * 3 + 1]
    cdef double Cz = shell_cxyz_data[C * 3 + 2]
    cdef double Dx = shell_cxyz_data[D * 3 + 0]
    cdef double Dy = shell_cxyz_data[D * 3 + 1]
    cdef double Dz = shell_cxyz_data[D * 3 + 2]

    cdef double ABx = Ax - Bx
    cdef double ABy = Ay - By
    cdef double ABz = Az - Bz
    cdef double CDx = Cx - Dx
    cdef double CDy = Cy - Dy
    cdef double CDz = Cz - Dz

    cdef double xij_pow[K_LMAX_D + 1]
    cdef double yij_pow[K_LMAX_D + 1]
    cdef double zij_pow[K_LMAX_D + 1]
    cdef double xkl_pow[K_LMAX_D + 1]
    cdef double ykl_pow[K_LMAX_D + 1]
    cdef double zkl_pow[K_LMAX_D + 1]

    cdef int m
    xij_pow[0] = 1.0
    yij_pow[0] = 1.0
    zij_pow[0] = 1.0
    xkl_pow[0] = 1.0
    ykl_pow[0] = 1.0
    zkl_pow[0] = 1.0
    for m in range(1, K_LMAX_D + 1):
        xij_pow[m] = xij_pow[m - 1] * ABx
        yij_pow[m] = yij_pow[m - 1] * ABy
        zij_pow[m] = zij_pow[m - 1] * ABz
        xkl_pow[m] = xkl_pow[m - 1] * CDx
        ykl_pow[m] = ykl_pow[m - 1] * CDy
        zkl_pow[m] = zkl_pow[m - 1] * CDz

    cdef int8_t A_lx[K_NCART_MAX]
    cdef int8_t A_ly[K_NCART_MAX]
    cdef int8_t A_lz[K_NCART_MAX]
    cdef int8_t B_lx[K_NCART_MAX]
    cdef int8_t B_ly[K_NCART_MAX]
    cdef int8_t B_lz[K_NCART_MAX]
    cdef int8_t C_lx[K_NCART_MAX]
    cdef int8_t C_ly[K_NCART_MAX]
    cdef int8_t C_lz[K_NCART_MAX]
    cdef int8_t D_lx[K_NCART_MAX]
    cdef int8_t D_ly[K_NCART_MAX]
    cdef int8_t D_lz[K_NCART_MAX]
    _fill_cart_comp(la, &A_lx[0], &A_ly[0], &A_lz[0])
    _fill_cart_comp(lb, &B_lx[0], &B_ly[0], &B_lz[0])
    _fill_cart_comp(lc, &C_lx[0], &C_ly[0], &C_lz[0])
    _fill_cart_comp(ld, &D_lx[0], &D_ly[0], &D_lz[0])

    cdef int L_total = la + lb + lc + ld
    cdef int nroots = ((L_total + 1) // 2) + 1
    cdef double roots[K_NROOTS_MAX]
    cdef double weights[K_NROOTS_MAX]
    cdef double Gx[K_STRIDE_D * K_STRIDE_D]
    cdef double Gy[K_STRIDE_D * K_STRIDE_D]
    cdef double Gz[K_STRIDE_D * K_STRIDE_D]

    cdef int nmax = la + lb + 1
    cdef int mmax = lc + ld + 1

    cdef int baseAB = <int>sp_pair_start_data[spAB]
    cdef int baseCD = <int>sp_pair_start_data[spCD]
    cdef int nPairAB = <int>sp_npair_data[spAB]
    cdef int nPairCD = <int>sp_npair_data[spCD]

    cdef int sA = <int>shell_prim_start_data[A]
    cdef int sB = <int>shell_prim_start_data[B]
    cdef int sC = <int>shell_prim_start_data[C]
    cdef int sD = <int>shell_prim_start_data[D]
    cdef int nprimB = <int>shell_nprim_data[B]
    cdef int nprimD = <int>shell_nprim_data[D]

    cdef int iAB, iCD, ipA, ipB, ipC, ipD
    cdef double p, q, denom, inv_denom
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double dx, dy, dz, PQ2, omega, T
    cdef double aexp, bexp, cexp, dexp
    cdef double cKab, cKcd, base, scale
    cdef double x, w, B0, B1, B1p
    cdef double Cx_, Cy_, Cz_, Cpx_, Cpy_, Cpz_
    cdef double Ix, Iy, Iz, Ix_m, Ix_p, Iy_m, Iy_p, Iz_m, Iz_p
    cdef double dIx, dIy, dIz
    cdef int ia, ib, ic, id, row, col
    cdef int iax, iay, iaz, ibx, iby, ibz, icx, icy, icz, idx, idy, idz
    cdef double bar

    # Zero output.
    for m in range(12):
        out_data[m] = 0.0

    for iAB in range(nPairAB):
        p = pair_eta_data[baseAB + iAB]
        Px = pair_Px_data[baseAB + iAB]
        Py = pair_Py_data[baseAB + iAB]
        Pz = pair_Pz_data[baseAB + iAB]
        cKab = pair_cK_data[baseAB + iAB]

        ipA = iAB // nprimB
        ipB = iAB - ipA * nprimB
        aexp = prim_exp_data[sA + ipA]
        bexp = prim_exp_data[sB + ipB]

        for iCD in range(nPairCD):
            q = pair_eta_data[baseCD + iCD]
            Qx = pair_Px_data[baseCD + iCD]
            Qy = pair_Py_data[baseCD + iCD]
            Qz = pair_Pz_data[baseCD + iCD]
            cKcd = pair_cK_data[baseCD + iCD]

            ipC = iCD // nprimD
            ipD = iCD - ipC * nprimD
            cexp = prim_exp_data[sC + ipC]
            dexp = prim_exp_data[sD + ipD]

            denom = p + q
            inv_denom = 1.0 / denom
            dx = Px - Qx
            dy = Py - Qy
            dz = Pz - Qz
            PQ2 = dx * dx + dy * dy + dz * dz
            omega = p * q * inv_denom
            T = omega * PQ2

            base = kTwoPiToFiveHalves / (p * q * sqrt(denom)) * cKab * cKcd

            _rys_roots_weights(nroots, T, &roots[0], &weights[0])

            for m in range(nroots):
                x = roots[m]
                w = weights[m]
                B0 = x * 0.5 * inv_denom
                B1 = (1.0 - x) * 0.5 / p + B0
                B1p = (1.0 - x) * 0.5 / q + B0

                Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px)
                Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py)
                Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz)

                Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx)
                Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy)
                Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz)

                _compute_G_d(&Gx[0], nmax, mmax, Cx_, Cpx_, B0, B1, B1p)
                _compute_G_d(&Gy[0], nmax, mmax, Cy_, Cpy_, B0, B1, B1p)
                _compute_G_d(&Gz[0], nmax, mmax, Cz_, Cpz_, B0, B1, B1p)

                scale = base * w

                for ia in range(nA):
                    iax = <int>A_lx[ia]
                    iay = <int>A_ly[ia]
                    iaz = <int>A_lz[ia]
                    for ib in range(nB):
                        ibx = <int>B_lx[ib]
                        iby = <int>B_ly[ib]
                        ibz = <int>B_lz[ib]
                        row = ia * nB + ib
                        for ic in range(nC):
                            icx = <int>C_lx[ic]
                            icy = <int>C_ly[ic]
                            icz = <int>C_lz[ic]
                            for id in range(nD):
                                idx = <int>D_lx[id]
                                idy = <int>D_ly[id]
                                idz = <int>D_lz[id]
                                col = ic * nD + id

                                bar = bar_data[row * nCD + col]
                                if bar == 0.0:
                                    continue

                                Ix = _shift_from_G_d(&Gx[0], iax, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                Iy = _shift_from_G_d(&Gy[0], iay, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                Iz = _shift_from_G_d(&Gz[0], iaz, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])

                                # ---- center A derivatives ----
                                Ix_m = (
                                    _shift_from_G_d(&Gx[0], iax - 1, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                    if iax > 0
                                    else 0.0
                                )
                                Ix_p = _shift_from_G_d(&Gx[0], iax + 1, ibx, icx, idx, &xij_pow[0], &xkl_pow[0])
                                dIx = (-<double>iax) * Ix_m + (2.0 * aexp) * Ix_p
                                out_data[0 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = (
                                    _shift_from_G_d(&Gy[0], iay - 1, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                    if iay > 0
                                    else 0.0
                                )
                                Iy_p = _shift_from_G_d(&Gy[0], iay + 1, iby, icy, idy, &yij_pow[0], &ykl_pow[0])
                                dIy = (-<double>iay) * Iy_m + (2.0 * aexp) * Iy_p
                                out_data[0 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = (
                                    _shift_from_G_d(&Gz[0], iaz - 1, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])
                                    if iaz > 0
                                    else 0.0
                                )
                                Iz_p = _shift_from_G_d(&Gz[0], iaz + 1, ibz, icz, idz, &zij_pow[0], &zkl_pow[0])
                                dIz = (-<double>iaz) * Iz_m + (2.0 * aexp) * Iz_p
                                out_data[0 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # ---- center B derivatives ----
                                Ix_m = (
                                    _shift_from_G_d(&Gx[0], iax, ibx - 1, icx, idx, &xij_pow[0], &xkl_pow[0])
                                    if ibx > 0
                                    else 0.0
                                )
                                Ix_p = _shift_from_G_d(&Gx[0], iax, ibx + 1, icx, idx, &xij_pow[0], &xkl_pow[0])
                                dIx = (-<double>ibx) * Ix_m + (2.0 * bexp) * Ix_p
                                out_data[1 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = (
                                    _shift_from_G_d(&Gy[0], iay, iby - 1, icy, idy, &yij_pow[0], &ykl_pow[0])
                                    if iby > 0
                                    else 0.0
                                )
                                Iy_p = _shift_from_G_d(&Gy[0], iay, iby + 1, icy, idy, &yij_pow[0], &ykl_pow[0])
                                dIy = (-<double>iby) * Iy_m + (2.0 * bexp) * Iy_p
                                out_data[1 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = (
                                    _shift_from_G_d(&Gz[0], iaz, ibz - 1, icz, idz, &zij_pow[0], &zkl_pow[0])
                                    if ibz > 0
                                    else 0.0
                                )
                                Iz_p = _shift_from_G_d(&Gz[0], iaz, ibz + 1, icz, idz, &zij_pow[0], &zkl_pow[0])
                                dIz = (-<double>ibz) * Iz_m + (2.0 * bexp) * Iz_p
                                out_data[1 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # ---- center C derivatives ----
                                Ix_m = (
                                    _shift_from_G_d(&Gx[0], iax, ibx, icx - 1, idx, &xij_pow[0], &xkl_pow[0])
                                    if icx > 0
                                    else 0.0
                                )
                                Ix_p = _shift_from_G_d(&Gx[0], iax, ibx, icx + 1, idx, &xij_pow[0], &xkl_pow[0])
                                dIx = (-<double>icx) * Ix_m + (2.0 * cexp) * Ix_p
                                out_data[2 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = (
                                    _shift_from_G_d(&Gy[0], iay, iby, icy - 1, idy, &yij_pow[0], &ykl_pow[0])
                                    if icy > 0
                                    else 0.0
                                )
                                Iy_p = _shift_from_G_d(&Gy[0], iay, iby, icy + 1, idy, &yij_pow[0], &ykl_pow[0])
                                dIy = (-<double>icy) * Iy_m + (2.0 * cexp) * Iy_p
                                out_data[2 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = (
                                    _shift_from_G_d(&Gz[0], iaz, ibz, icz - 1, idz, &zij_pow[0], &zkl_pow[0])
                                    if icz > 0
                                    else 0.0
                                )
                                Iz_p = _shift_from_G_d(&Gz[0], iaz, ibz, icz + 1, idz, &zij_pow[0], &zkl_pow[0])
                                dIz = (-<double>icz) * Iz_m + (2.0 * cexp) * Iz_p
                                out_data[2 * 3 + 2] += bar * scale * (Ix * Iy * dIz)

                                # ---- center D derivatives ----
                                Ix_m = (
                                    _shift_from_G_d(&Gx[0], iax, ibx, icx, idx - 1, &xij_pow[0], &xkl_pow[0])
                                    if idx > 0
                                    else 0.0
                                )
                                Ix_p = _shift_from_G_d(&Gx[0], iax, ibx, icx, idx + 1, &xij_pow[0], &xkl_pow[0])
                                dIx = (-<double>idx) * Ix_m + (2.0 * dexp) * Ix_p
                                out_data[3 * 3 + 0] += bar * scale * (dIx * Iy * Iz)

                                Iy_m = (
                                    _shift_from_G_d(&Gy[0], iay, iby, icy, idy - 1, &yij_pow[0], &ykl_pow[0])
                                    if idy > 0
                                    else 0.0
                                )
                                Iy_p = _shift_from_G_d(&Gy[0], iay, iby, icy, idy + 1, &yij_pow[0], &ykl_pow[0])
                                dIy = (-<double>idy) * Iy_m + (2.0 * dexp) * Iy_p
                                out_data[3 * 3 + 1] += bar * scale * (Ix * dIy * Iz)

                                Iz_m = (
                                    _shift_from_G_d(&Gz[0], iaz, ibz, icz, idz - 1, &zij_pow[0], &zkl_pow[0])
                                    if idz > 0
                                    else 0.0
                                )
                                Iz_p = _shift_from_G_d(&Gz[0], iaz, ibz, icz, idz + 1, &zij_pow[0], &zkl_pow[0])
                                dIz = (-<double>idz) * Iz_m + (2.0 * dexp) * Iz_p
                                out_data[3 * 3 + 2] += bar * scale * (Ix * Iy * dIz)


def eri_rys_deriv_contracted_cart_sp_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    int spCD,
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] bar_tile,
):
    # Contract analytic 4c ERI nuclear derivatives for a single shell-pair quartet (spAB|spCD).
    #
    # Returns: float64 array (4,3) with centers [A,B,C,D] and xyz.

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int A = <int>sp_A[spAB]
    cdef int B = <int>sp_B[spAB]
    cdef int C = <int>sp_A[spCD]
    cdef int D = <int>sp_B[spCD]
    cdef int la = <int>shell_l[A]
    cdef int lb = <int>shell_l[B]
    cdef int lc = <int>shell_l[C]
    cdef int ld = <int>shell_l[D]
    if la > K_LMAX or lb > K_LMAX or lc > K_LMAX or ld > K_LMAX:
        raise NotImplementedError(f"eri_rys_deriv_contracted_cart_sp_cy currently supports only l<={K_LMAX} per shell")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc)
    cdef int nD = _ncart(ld)
    if bar_tile.shape[0] != nA * nB or bar_tile.shape[1] != nC * nD:
        raise ValueError("bar_tile shape mismatch for (nA*nB, nC*nD)")

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((4, 3), dtype=np.float64)

    with nogil:
        _eri_rys_deriv_contracted_cart_sp_kernel(
            <const double*>shell_cxyz.data,
            <const cnp.int32_t*>shell_l.data,
            <const cnp.int32_t*>shell_prim_start.data,
            <const cnp.int32_t*>shell_nprim.data,
            <const double*>prim_exp.data,
            <const cnp.int32_t*>sp_A.data,
            <const cnp.int32_t*>sp_B.data,
            <const cnp.int32_t*>sp_pair_start.data,
            <const cnp.int32_t*>sp_npair.data,
            <const double*>pair_eta.data,
            <const double*>pair_Px.data,
            <const double*>pair_Py.data,
            <const double*>pair_Pz.data,
            <const double*>pair_cK.data,
            spAB,
            spCD,
            <const double*>bar_tile.data,
            <double*>out.data,
        )
    return out


def eri_rys_deriv_contracted_cart_sp_batch_cy(
    cnp.ndarray[cnp.double_t, ndim=2, mode="c"] shell_cxyz,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_l,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_prim_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] shell_nprim,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prim_exp,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_A,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_B,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_pair_start,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] sp_npair,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_eta,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Px,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Py,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_Pz,
    cnp.ndarray[cnp.double_t, ndim=1, mode="c"] pair_cK,
    int spAB,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spCD,
    cnp.ndarray[cnp.double_t, ndim=3, mode="c"] bar_tile,
    int threads=0,
):
    # Batch version for fixed spAB and multiple spCD tasks.
    #
    # Returns: float64 array (nt,4,3).
    # Requires all spCD tasks share (lc,ld) so bar_tile has consistent trailing dims.

    if shell_cxyz.shape[1] != 3:
        raise ValueError("shell_cxyz must have shape (nShell, 3)")

    cdef int nsp = <int>sp_A.shape[0]
    if spAB < 0 or spAB >= nsp:
        raise ValueError("spAB out of range")

    cdef int nt = <int>spCD.shape[0]
    if nt == 0:
        return np.empty((0, 4, 3), dtype=np.float64)

    if np.any(spCD < 0) or np.any(spCD >= nsp):
        raise ValueError("spCD entries out of range")

    cdef int A = <int>sp_A[spAB]
    cdef int B = <int>sp_B[spAB]
    cdef int la = <int>shell_l[A]
    cdef int lb = <int>shell_l[B]
    if la > K_LMAX or lb > K_LMAX:
        raise NotImplementedError(f"eri_rys_deriv_contracted_cart_sp_batch_cy currently supports only l<={K_LMAX} per shell")

    cdef int spCD0 = <int>spCD[0]
    cdef int C0 = <int>sp_A[spCD0]
    cdef int D0 = <int>sp_B[spCD0]
    cdef int lc0 = <int>shell_l[C0]
    cdef int ld0 = <int>shell_l[D0]
    if lc0 > K_LMAX or ld0 > K_LMAX:
        raise NotImplementedError(f"eri_rys_deriv_contracted_cart_sp_batch_cy currently supports only l<={K_LMAX} per shell")

    cdef int t, sp_i, Ci, Di, lc, ld
    for t in range(nt):
        sp_i = <int>spCD[t]
        Ci = <int>sp_A[sp_i]
        Di = <int>sp_B[sp_i]
        lc = <int>shell_l[Ci]
        ld = <int>shell_l[Di]
        if lc != lc0 or ld != ld0:
            raise ValueError("all spCD tasks must share the same (lc,ld)")

    cdef int nA = _ncart(la)
    cdef int nB = _ncart(lb)
    cdef int nC = _ncart(lc0)
    cdef int nD = _ncart(ld0)
    cdef int nAB = nA * nB
    cdef int nCD = nC * nD

    if bar_tile.shape[0] != nt or bar_tile.shape[1] != nAB or bar_tile.shape[2] != nCD:
        raise ValueError("bar_tile must have shape (nt, nA*nB, nC*nD)")

    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((nt, 4, 3), dtype=np.float64)
    cdef double* out_data = <double*>out.data
    cdef Py_ssize_t stride_out = <Py_ssize_t>(12)

    cdef int threads_i = <int>threads
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    cdef const double* bar_data = <const double*>bar_tile.data
    cdef Py_ssize_t stride_bar = <Py_ssize_t>(nAB * nCD)

    with nogil:
        if threads_i > 1 and nt > 1:
            for t in prange(nt, schedule="static", num_threads=threads_i):
                _eri_rys_deriv_contracted_cart_sp_kernel(
                    <const double*>shell_cxyz.data,
                    <const cnp.int32_t*>shell_l.data,
                    <const cnp.int32_t*>shell_prim_start.data,
                    <const cnp.int32_t*>shell_nprim.data,
                    <const double*>prim_exp.data,
                    <const cnp.int32_t*>sp_A.data,
                    <const cnp.int32_t*>sp_B.data,
                    <const cnp.int32_t*>sp_pair_start.data,
                    <const cnp.int32_t*>sp_npair.data,
                    <const double*>pair_eta.data,
                    <const double*>pair_Px.data,
                    <const double*>pair_Py.data,
                    <const double*>pair_Pz.data,
                    <const double*>pair_cK.data,
                    spAB,
                    <int>(<const cnp.int32_t*>spCD.data)[t],
                    bar_data + (<Py_ssize_t>t) * stride_bar,
                    out_data + (<Py_ssize_t>t) * stride_out,
                )
        else:
            for t in range(nt):
                _eri_rys_deriv_contracted_cart_sp_kernel(
                    <const double*>shell_cxyz.data,
                    <const cnp.int32_t*>shell_l.data,
                    <const cnp.int32_t*>shell_prim_start.data,
                    <const cnp.int32_t*>shell_nprim.data,
                    <const double*>prim_exp.data,
                    <const cnp.int32_t*>sp_A.data,
                    <const cnp.int32_t*>sp_B.data,
                    <const cnp.int32_t*>sp_pair_start.data,
                    <const cnp.int32_t*>sp_npair.data,
                    <const double*>pair_eta.data,
                    <const double*>pair_Px.data,
                    <const double*>pair_Py.data,
                    <const double*>pair_Pz.data,
                    <const double*>pair_cK.data,
                    spAB,
                    <int>(<const cnp.int32_t*>spCD.data)[t],
                    bar_data + (<Py_ssize_t>t) * stride_bar,
                    out_data + (<Py_ssize_t>t) * stride_out,
                )

    return out


__all__ = [
    "build_pair_tables_cart_inplace_cpu",
    "df_int3c2e_deriv_contracted_cart_sp_cy",
    "df_int3c2e_deriv_contracted_cart_sp_batch_cy",
    "df_int3c2e_deriv_contracted_cart_sp_batch_sphbar_qmn_cy",
    "df_metric_2c2e_deriv_contracted_cart_sp_cy",
    "df_metric_2c2e_deriv_contracted_cart_sp_batch_cy",
    "df_symmetrize_qmn_inplace_cy",
    "eri_rys_deriv_contracted_cart_sp_cy",
    "eri_rys_deriv_contracted_cart_sp_batch_cy",
    "eri_rys_tile_cart_batch_cy",
    "eri_rys_tile_cart_cy",
    "eri_rys_tile_cart_sp_batch_cy",
    "eri_rys_tile_cart_sp_cy",
    "openmp_enabled",
    "rys_roots_weights_cy",
    "schwarz_shellpairs_cart_sp_cy",
]
