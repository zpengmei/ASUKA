cdef void _ensure_tables() except *:
    global _oracle_mod, _seg_fallback, _sv_by_code_arr, _sv_by_code, _seg_lut_max_b
    global _seg_tbl_arr, _seg_tbl, _seg_tbl_nb
    if _sv_by_code_arr is not None:
        return

    import asuka.cuguga.oracle as _oracle_mod  # type: ignore[import-not-found]

    _seg_lut_max_b = int(_oracle_mod._SEG_LUT_MAX_B)
    _seg_fallback = _oracle_mod._segment_value_int_fallback
    _sv_by_code_arr = np.asarray(_oracle_mod._SV_BY_CODE, dtype=np.float64, order="C")
    _sv_by_code = _sv_by_code_arr

    # Precompute full segment values for b <= _SEG_LUT_MAX_B:
    #   seg[q, dbi, dprime, d, b]
    # where dbi: 0=unused, 1=db=-1, 2=db=+1.
    _seg_tbl_nb = int(_seg_lut_max_b) + 1
    _seg_tbl_arr = np.zeros((7, 3, 4, 4, _seg_tbl_nb), dtype=np.float64, order="C")
    cdef double[::1] seg_flat = _seg_tbl_arr.reshape(-1)
    cdef int qq, dbi, dprime, d, b, db
    cdef Py_ssize_t idx = 0
    for qq in range(7):
        for dbi in range(3):
            if dbi == 1:
                db = -1
            elif dbi == 2:
                db = +1
            else:
                db = 0
            for dprime in range(4):
                for d in range(4):
                    for b in range(_seg_tbl_nb):
                        seg_flat[idx] = _segment_value_int_lut(qq, dprime, d, db, b)
                        idx += 1
    _seg_tbl = _seg_tbl_arr.reshape(-1)


cdef inline cnp.int64_t[:, ::1] _get_child_prefix(drt) except *:
    """Return cached child-prefix-walks table for this DRT (avoid Python call overhead)."""
    global _child_prefix_cached_drt, _child_prefix_cached_arr, _child_prefix_cached
    if _child_prefix_cached_drt is drt and _child_prefix_cached_arr is not None:
        return _child_prefix_cached
    cdef cnp.ndarray[cnp.int64_t, ndim=2] arr = _oracle_mod._child_prefix_walks(drt)  # type: ignore[attr-defined]
    _child_prefix_cached_drt = drt
    _child_prefix_cached_arr = arr
    _child_prefix_cached = arr
    return _child_prefix_cached


cdef inline double _sv_lookup(int code, int b) noexcept nogil:
    return _sv_by_code[code, b]


cdef inline int _sv_lut_w(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_W
    if dprime == 1 and d == 1:
        return _SV_ONE
    if dprime == 2 and d == 2:
        return _SV_ONE
    if dprime == 3 and d == 3:
        return _SV_TWO
    return _SV_ZERO


cdef inline int _sv_lut_uR(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_uR
    if dprime == 1 and d == 0:
        return _SV_ONE
    if dprime == 2 and d == 0:
        return _SV_ONE
    if dprime == 3 and d == 1:
        return _SV_A10
    if dprime == 3 and d == 2:
        return _SV_A12
    return _SV_ZERO


cdef inline int _sv_lut_oR(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_oR
    if dprime == 0 and (d == 1 or d == 2):
        return _SV_ONE
    if dprime == 1 and d == 3:
        return _SV_A01
    if dprime == 2 and d == 3:
        return _SV_A21
    return _SV_ZERO


cdef inline int _sv_lut_uL(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_uL
    if dprime == 0 and (d == 1 or d == 2):
        return _SV_ONE
    if dprime == 1 and d == 3:
        return _SV_A21
    if dprime == 2 and d == 3:
        return _SV_A01
    return _SV_ZERO


cdef inline int _sv_lut_oL(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_oL
    if dprime == 1 and d == 0:
        return _SV_ONE
    if dprime == 2 and d == 0:
        return _SV_ONE
    if dprime == 3 and d == 1:
        return _SV_A01
    if dprime == 3 and d == 2:
        return _SV_A21
    return _SV_ZERO


cdef inline int _sv_lut_R_dbm1(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_R_DBM1
    if dprime == 0 and d == 0:
        return _SV_ONE
    if dprime == 1 and d == 1:
        return _SV_NEG_ONE
    if dprime == 1 and d == 2:
        return _SV_NEG_INV_BP2
    if dprime == 2 and d == 2:
        return _SV_C2
    if dprime == 3 and d == 3:
        return _SV_NEG_ONE
    return _SV_ZERO


cdef inline int _sv_lut_R_dbp1(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_R_DBP1
    if dprime == 0 and d == 0:
        return _SV_ONE
    if dprime == 1 and d == 1:
        return _SV_C0
    if dprime == 2 and d == 1:
        return _SV_INV_B
    if dprime == 2 and d == 2:
        return _SV_NEG_ONE
    if dprime == 3 and d == 3:
        return _SV_NEG_ONE
    return _SV_ZERO


cdef inline int _sv_lut_L_dbm1(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_L_DBM1
    if dprime == 0 and d == 0:
        return _SV_ONE
    if dprime == 1 and d == 1:
        return _SV_C1
    if dprime == 1 and d == 2:
        return _SV_INV_BP1
    if dprime == 2 and d == 2:
        return _SV_NEG_ONE
    if dprime == 3 and d == 3:
        return _SV_NEG_ONE
    return _SV_ZERO


cdef inline int _sv_lut_L_dbp1(int dprime, int d) noexcept:
    # matches oracle._SV_LUT_L_DBP1
    if dprime == 0 and d == 0:
        return _SV_ONE
    if dprime == 1 and d == 1:
        return _SV_NEG_ONE
    if dprime == 2 and d == 1:
        return _SV_NEG_INV_BP1
    if dprime == 2 and d == 2:
        return _SV_C1
    if dprime == 3 and d == 3:
        return _SV_NEG_ONE
    return _SV_ZERO


cdef inline double _segment_value_int_lut(int q, int dprime, int d, int db, int b) noexcept:
    # LUT-only variant (no Python fallback); valid for b <= _seg_lut_max_b.
    if q == _Q_W:
        return _sv_lookup(_sv_lut_w(dprime, d), b)
    if q == _Q_uR:
        return _sv_lookup(_sv_lut_uR(dprime, d), b)
    if q == _Q_oR:
        return _sv_lookup(_sv_lut_oR(dprime, d), b)
    if q == _Q_uL:
        return _sv_lookup(_sv_lut_uL(dprime, d), b)
    if q == _Q_oL:
        return _sv_lookup(_sv_lut_oL(dprime, d), b)
    if q == _Q_R:
        if db == -1:
            return _sv_lookup(_sv_lut_R_dbm1(dprime, d), b)
        if db == +1:
            return _sv_lookup(_sv_lut_R_dbp1(dprime, d), b)
        return 0.0
    if q == _Q_L:
        if db == -1:
            return _sv_lookup(_sv_lut_L_dbm1(dprime, d), b)
        if db == +1:
            return _sv_lookup(_sv_lut_L_dbp1(dprime, d), b)
        return 0.0
    return 0.0


cdef inline double _segment_value_int_tbl(int q, int dprime, int d, int db, int b) noexcept nogil:
    """Fast table lookup for segment values (assumes 0 <= b <= _seg_lut_max_b)."""
    cdef int dbi = 0
    if q == _Q_R or q == _Q_L:
        if db == -1:
            dbi = 1
        elif db == +1:
            dbi = 2
        else:
            return 0.0
    return _seg_tbl[
        (((((q * 3 + dbi) * 4 + dprime) * 4 + d) * _seg_tbl_nb) + b)
    ]


cdef inline double _segment_value_int(int q, int dprime, int d, int db, int b) except *:
    # Mirrors oracle._segment_value_int (LUT + fallback for large b), but uses a
    # precomputed 5D segment-value table for the LUT range.
    if b <= _seg_lut_max_b:
        return _segment_value_int_tbl(q, dprime, d, db, b)

    # Rare path: call the Python fallback to preserve exact behavior.
    return float(_seg_fallback(q, dprime, d, db, b))


