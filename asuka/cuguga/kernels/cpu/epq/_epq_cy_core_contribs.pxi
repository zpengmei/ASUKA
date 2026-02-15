def epq_contribs_from_csf_index_arrays_cy(
    drt,
    int csf_idx,
    int p,
    int q,
    cnp.ndarray[cnp.int8_t, ndim=1] steps,
    cnp.ndarray[cnp.int32_t, ndim=1] nodes,
):
    """
    Return (idx:int32[], coeff:float64[]) for E_pq|csf_idx>, using Cython.

    This is a drop-in replacement for:
      asuka.cuguga.oracle._e_pq_contribs_from_csf_index_arrays(...)
    """
    _ensure_tables()

    cdef int norb = int(drt.norb)
    if int(drt.nelec) > _seg_lut_max_b:
        raise ValueError("DRT nelec exceeds segment-value LUT max; increase _SEG_LUT_MAX_B")
    if p < 0 or p >= norb or q < 0 or q >= norb:
        raise ValueError("orbital indices out of range")
    if p == q:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    cdef i8[::1] steps_v = steps
    cdef cnp.int32_t[::1] nodes_v = nodes

    cdef int occ_p = _step_to_occ(<i8>steps_v[p])
    cdef int occ_q = _step_to_occ(<i8>steps_v[q])
    if occ_q <= 0 or occ_p >= 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    cdef int start
    cdef int end
    cdef int q_start
    cdef int q_mid
    cdef int q_end
    if p < q:
        start = p
        end = q
        q_start = _Q_uR
        q_mid = _Q_R
        q_end = _Q_oR
    else:
        start = q
        end = p
        q_start = _Q_uL
        q_mid = _Q_L
        q_end = _Q_oL

    cdef cnp.int32_t[:, ::1] child = drt.child
    cdef cnp.int16_t[::1] node_twos = drt.node_twos

    # Cached prefix-walk counts.
    cdef cnp.int64_t[:, ::1] child_prefix = _get_child_prefix(drt)

    cdef int node_start = <int>nodes_v[start]
    cdef int node_end_target = <int>nodes_v[end + 1]

    # Compute prefix offsets for the reference path (no per-neighbor path_to_index).
    cdef long long idx = 0
    cdef long long prefix_offset = 0
    cdef long long prefix_endplus1 = 0
    cdef int kk
    cdef int node_kk
    cdef int step_kk
    for kk in range(end + 1):
        if kk == start:
            prefix_offset = idx
        node_kk = <int>nodes_v[kk]
        step_kk = <int>steps_v[kk]
        idx += <long long>child_prefix[node_kk, step_kk]
        if kk == end:
            prefix_endplus1 = idx

    cdef long long suffix_offset = (<long long>csf_idx) - prefix_endplus1

    # DFS stack (SOA vectors).
    cdef vector[int] st_k
    cdef vector[int] st_node
    cdef vector[double] st_w
    cdef vector[long long] st_seg
    st_k.push_back(start)
    st_node.push_back(node_start)
    st_w.push_back(1.0)
    st_seg.push_back(0)

    cdef vector[int] out_idx
    cdef vector[double] out_coeff

    cdef int k
    cdef int node_k
    cdef double w
    cdef long long seg_idx
    cdef int is_first
    cdef int is_last
    cdef int qk
    cdef int d_k
    cdef int b_k
    cdef int k_next
    cdef int dprime
    cdef int dp0
    cdef int dp1
    cdef int ndp
    cdef int child_k
    cdef int bprime
    cdef int db
    cdef double seg
    cdef double w2
    cdef long long seg_idx2
    cdef long long csf_i_ll
    cdef int csf_i

    if 1:
        while st_k.size() != 0:
            k = st_k.back()
            st_k.pop_back()
            node_k = st_node.back()
            st_node.pop_back()
            w = st_w.back()
            st_w.pop_back()
            seg_idx = st_seg.back()
            st_seg.pop_back()

            is_first = 1 if k == start else 0
            is_last = 1 if k == end else 0
            qk = q_start if is_first else (q_end if is_last else q_mid)

            d_k = <int>steps_v[k]
            b_k = <int>node_twos[<int>nodes_v[k + 1]]
            k_next = k + 1

            ndp = _candidate_dprimes(qk, d_k, &dp0, &dp1)
            if ndp == 0:
                continue
            if ndp == 1:
                dprime = dp0
                child_k = <int>child[node_k, dprime]
                if child_k >= 0:
                    bprime = <int>node_twos[child_k]
                    db = b_k - bprime
                    seg = _segment_value_int_tbl(qk, dprime, d_k, db, b_k)
                    if seg != 0.0:
                        w2 = w * seg
                        seg_idx2 = seg_idx + <long long>child_prefix[node_k, dprime]
                        if is_last:
                            if child_k == node_end_target:
                                csf_i_ll = prefix_offset + seg_idx2 + suffix_offset
                                csf_i = <int>csf_i_ll
                                if csf_i != csf_idx and w2 != 0.0:
                                    out_idx.push_back(csf_i)
                                    out_coeff.push_back(w2)
                        else:
                            st_k.push_back(k_next)
                            st_node.push_back(child_k)
                            st_w.push_back(w2)
                            st_seg.push_back(seg_idx2)
            elif ndp == 2:
                dprime = dp0
                child_k = <int>child[node_k, dprime]
                if child_k >= 0:
                    bprime = <int>node_twos[child_k]
                    db = b_k - bprime
                    seg = _segment_value_int_tbl(qk, dprime, d_k, db, b_k)
                    if seg != 0.0:
                        w2 = w * seg
                        seg_idx2 = seg_idx + <long long>child_prefix[node_k, dprime]
                        if is_last:
                            if child_k == node_end_target:
                                csf_i_ll = prefix_offset + seg_idx2 + suffix_offset
                                csf_i = <int>csf_i_ll
                                if csf_i != csf_idx and w2 != 0.0:
                                    out_idx.push_back(csf_i)
                                    out_coeff.push_back(w2)
                        else:
                            st_k.push_back(k_next)
                            st_node.push_back(child_k)
                            st_w.push_back(w2)
                            st_seg.push_back(seg_idx2)

                dprime = dp1
                child_k = <int>child[node_k, dprime]
                if child_k >= 0:
                    bprime = <int>node_twos[child_k]
                    db = b_k - bprime
                    seg = _segment_value_int_tbl(qk, dprime, d_k, db, b_k)
                    if seg != 0.0:
                        w2 = w * seg
                        seg_idx2 = seg_idx + <long long>child_prefix[node_k, dprime]
                        if is_last:
                            if child_k == node_end_target:
                                csf_i_ll = prefix_offset + seg_idx2 + suffix_offset
                                csf_i = <int>csf_i_ll
                                if csf_i != csf_idx and w2 != 0.0:
                                    out_idx.push_back(csf_i)
                                    out_coeff.push_back(w2)
                        else:
                            st_k.push_back(k_next)
                            st_node.push_back(child_k)
                            st_w.push_back(w2)
                            st_seg.push_back(seg_idx2)
            else:
                for dprime in range(4):
                    child_k = <int>child[node_k, dprime]
                    if child_k < 0:
                        continue
                    bprime = <int>node_twos[child_k]
                    db = b_k - bprime
                    seg = _segment_value_int_tbl(qk, dprime, d_k, db, b_k)
                    if seg == 0.0:
                        continue
                    w2 = w * seg
                    seg_idx2 = seg_idx + <long long>child_prefix[node_k, dprime]
                    if is_last:
                        if child_k != node_end_target:
                            continue
                        csf_i_ll = prefix_offset + seg_idx2 + suffix_offset
                        csf_i = <int>csf_i_ll
                        if csf_i != csf_idx and w2 != 0.0:
                            out_idx.push_back(csf_i)
                            out_coeff.push_back(w2)
                    else:
                        st_k.push_back(k_next)
                        st_node.push_back(child_k)
                        st_w.push_back(w2)
                        st_seg.push_back(seg_idx2)

    cdef Py_ssize_t n = <Py_ssize_t>out_idx.size()
    cdef cnp.ndarray[cnp.int32_t, ndim=1] idx_arr = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeff_arr = np.empty(n, dtype=np.float64)
    cdef cnp.int32_t[::1] idx_view = idx_arr
    cdef double[::1] coeff_view = coeff_arr
    cdef Py_ssize_t i
    for i in range(n):
        idx_view[i] = <cnp.int32_t>out_idx[i]
        coeff_view[i] = <double>out_coeff[i]
    return idx_arr, coeff_arr


