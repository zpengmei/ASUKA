cdef object _oracle_mod = None
cdef object _seg_fallback = None
cdef object _sv_by_code_arr = None
cdef double[:, ::1] _sv_by_code
cdef int _seg_lut_max_b = 0
cdef object _seg_tbl_arr = None
cdef double[::1] _seg_tbl
cdef int _seg_tbl_nb = 0

# Per-process cache for the prefix-walk counts table (keyed by the most recent DRT).
cdef object _child_prefix_cached_drt = None
cdef object _child_prefix_cached_arr = None
cdef cnp.int64_t[:, ::1] _child_prefix_cached


cdef int _Q_W = 0
cdef int _Q_uR = 1
cdef int _Q_R = 2
cdef int _Q_oR = 3
cdef int _Q_uL = 4
cdef int _Q_L = 5
cdef int _Q_oL = 6

# Segment value "codes" (must match oracle.py)
cdef int _SV_ZERO = 0
cdef int _SV_ONE = 1
cdef int _SV_TWO = 2
cdef int _SV_NEG_ONE = 3
cdef int _SV_A01 = 4
cdef int _SV_A10 = 5
cdef int _SV_A12 = 6
cdef int _SV_A21 = 7
cdef int _SV_INV_B = 8
cdef int _SV_INV_BP1 = 9
cdef int _SV_INV_BP2 = 10
cdef int _SV_NEG_INV_BP1 = 11
cdef int _SV_NEG_INV_BP2 = 12
cdef int _SV_C0 = 13
cdef int _SV_C1 = 14
cdef int _SV_C2 = 15


ctypedef signed char i8
ctypedef uint64_t u64

include "_epq_cy_qmc_rng.pxi"



cdef inline int _step_to_occ(i8 step) noexcept nogil:
    if step == 0:
        return 0
    if step == 3:
        return 2
    return 1


cdef inline int _candidate_dprimes(int qk, int d_k, int* dp0, int* dp1) noexcept nogil:
    """Return number of candidate d' for (qk, d), writing into dp0/dp1.

    The segment-value LUTs imply that most (d', d) combinations are identically
    zero. Restricting the inner loop to the (at most) two viable d' values
    reduces branch checks and table lookups in the DFS hot loop.
    """

    # Weighted (diagonal) segment: only d'==d can possibly contribute.
    if qk == _Q_W:
        dp0[0] = d_k
        return 1

    # Start of raising (uR) and end of lowering (oL) share the same LUT form.
    # Nonzero cases:
    #   d=0  -> d' in {1,2}
    #   d=1,2-> d'=3
    if qk == _Q_uR or qk == _Q_oL:
        if d_k == 0:
            dp0[0] = 1
            dp1[0] = 2
            return 2
        if d_k == 1 or d_k == 2:
            dp0[0] = 3
            return 1
        return 0

    # End of raising (oR) and start of lowering (uL) share the same LUT form.
    # Nonzero cases:
    #   d=1,2-> d'=0
    #   d=3  -> d' in {1,2}
    if qk == _Q_oR or qk == _Q_uL:
        if d_k == 1 or d_k == 2:
            dp0[0] = 0
            return 1
        if d_k == 3:
            dp0[0] = 1
            dp1[0] = 2
            return 2
        return 0

    # Middle segments for raising/lowering generators.
    if qk == _Q_R:
        if d_k == 1 or d_k == 2:
            dp0[0] = 1
            dp1[0] = 2
            return 2
        dp0[0] = d_k
        return 1
    if qk == _Q_L:
        if d_k == 0:
            dp0[0] = 0
            return 1
        if d_k == 3:
            dp0[0] = 3
            return 1
        dp0[0] = 1
        dp1[0] = 2
        return 2

    # Fallback: unknown segment type (should not happen).
    return 4


include "_epq_cy_qmc_epq_sample_nogil.pxi"


