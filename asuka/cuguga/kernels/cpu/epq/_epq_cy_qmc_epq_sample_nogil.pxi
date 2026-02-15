cdef inline int _epq_sample_one_nogil(
    int norb,
    int csf_idx,
    int p,
    int q,
    i8* steps_ptr,
    cnp.int32_t* nodes_ptr,
    cnp.int32_t[:, ::1] child,
    cnp.int16_t[::1] node_twos,
    cnp.int64_t[:, ::1] child_prefix,
    u64* rng_state,
    int* child_out,
    double* coeff_out,
    double* inv_p_out,
) noexcept nogil:
    """Sample one child from E_pq|csf_idx> with p(i) ∝ |c_i| (nogil).

    Returns 1 on success and writes (child, coeff, inv_p). Returns 0 if the
    action is identically zero.
    """

    cdef int occ_p
    cdef int occ_q
    cdef int occ_pp

    if p < 0 or p >= norb or q < 0 or q >= norb:
        return 0

    if p == q:
        occ_pp = _step_to_occ(steps_ptr[p])
        if occ_pp <= 0:
            return 0
        child_out[0] = csf_idx
        coeff_out[0] = <double>occ_pp
        inv_p_out[0] = 1.0
        return 1

    occ_p = _step_to_occ(steps_ptr[p])
    occ_q = _step_to_occ(steps_ptr[q])
    if occ_q <= 0 or occ_p >= 2:
        return 0

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

    cdef int node_start = <int>nodes_ptr[start]
    cdef int node_end_target = <int>nodes_ptr[end + 1]

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
        node_kk = <int>nodes_ptr[kk]
        step_kk = <int>steps_ptr[kk]
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

    # Weighted reservoir sample over the enumerated children.
    cdef double tot = 0.0
    cdef int picked_child = 0
    cdef double picked_coeff = 0.0
    cdef double picked_wabs = 0.0

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
    cdef double wabs

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

        d_k = <int>steps_ptr[k]
        b_k = <int>node_twos[<int>nodes_ptr[k + 1]]
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
                                wabs = fabs(w2)
                                if wabs != 0.0:
                                    tot += wabs
                                    if _rand_u01(rng_state) * tot < wabs:
                                        picked_child = csf_i
                                        picked_coeff = w2
                                        picked_wabs = wabs
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
                                wabs = fabs(w2)
                                if wabs != 0.0:
                                    tot += wabs
                                    if _rand_u01(rng_state) * tot < wabs:
                                        picked_child = csf_i
                                        picked_coeff = w2
                                        picked_wabs = wabs
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
                                wabs = fabs(w2)
                                if wabs != 0.0:
                                    tot += wabs
                                    if _rand_u01(rng_state) * tot < wabs:
                                        picked_child = csf_i
                                        picked_coeff = w2
                                        picked_wabs = wabs
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
                        wabs = fabs(w2)
                        if wabs != 0.0:
                            tot += wabs
                            if _rand_u01(rng_state) * tot < wabs:
                                picked_child = csf_i
                                picked_coeff = w2
                                picked_wabs = wabs
                else:
                    st_k.push_back(k_next)
                    st_node.push_back(child_k)
                    st_w.push_back(w2)
                    st_seg.push_back(seg_idx2)

    if picked_wabs == 0.0 or tot == 0.0:
        return 0

    child_out[0] = picked_child
    coeff_out[0] = picked_coeff
    inv_p_out[0] = tot / picked_wabs
    return 1

