def epq_apply_weighted_many_cy(
    drt,
    int csf_idx,
    cnp.ndarray[cnp.int32_t, ndim=1] p_idx,
    cnp.ndarray[cnp.int32_t, ndim=1] q_idx,
    cnp.ndarray[cnp.float64_t, ndim=1] weights,
    cnp.ndarray[cnp.int8_t, ndim=1] steps,
    cnp.ndarray[cnp.int32_t, ndim=1] nodes,
    double thresh_contrib=0.0,
    bint trusted=False,
):
    """Apply Σ_t weights[t] * E_{p_idx[t],q_idx[t]} to |csf_idx> and return COO arrays.

    Returns
    -------
    (idx, val)
        Concatenated contributions to neighbor CSF indices:
          val[j] = weights[t] * <idx[j] | E_pq | csf_idx>
        with the same excitation-range segment-walk logic as
        :func:`epq_contribs_from_csf_index_arrays_cy`.
    """

    _ensure_tables()

    cdef Py_ssize_t n_pairs = p_idx.shape[0]
    if q_idx.shape[0] != n_pairs or weights.shape[0] != n_pairs:
        raise ValueError("p_idx, q_idx, and weights must have the same length")

    cdef int norb = int(drt.norb)
    if int(drt.nelec) > _seg_lut_max_b:
        raise ValueError("DRT nelec exceeds segment-value LUT max; increase _SEG_LUT_MAX_B")

    cdef i8[::1] steps_v = steps
    cdef cnp.int32_t[::1] nodes_v = nodes
    cdef cnp.int32_t[::1] p_v = p_idx
    cdef cnp.int32_t[::1] q_v = q_idx
    cdef double[::1] w_v = weights

    cdef cnp.int32_t[:, ::1] child = drt.child
    cdef cnp.int16_t[::1] node_twos = drt.node_twos

    # Cached prefix-walk counts (shared across all pairs for this DRT).
    cdef cnp.int64_t[:, ::1] child_prefix = _get_child_prefix(drt)

    cdef vector[int] out_idx
    cdef vector[double] out_val
    if n_pairs > 0:
        # Heuristic reserve to reduce vector re-allocations (typical ~2-4 nnz per (p,q)).
        out_idx.reserve(<size_t>(n_pairs * 4))
        out_val.reserve(<size_t>(n_pairs * 4))

    # Prefix offsets for the reference path:
    #   idx_prefix[k] = sum_{t=0..k-1} child_prefix[nodes[t], steps[t]]
    # so we can get:
    #   prefix_offset    = idx_prefix[start]
    #   prefix_endplus1  = idx_prefix[end+1]
    # without an O(end) scan per (p,q).
    cdef vector[long long] idx_prefix
    idx_prefix.resize(norb + 1)
    idx_prefix[0] = 0

    # Reusable DFS stacks (SOA vectors).
    cdef vector[int] st_k
    cdef vector[int] st_node
    cdef vector[double] st_w
    cdef vector[long long] st_seg

    cdef Py_ssize_t t
    cdef int p
    cdef int q
    cdef double wgt
    cdef int occ_p
    cdef int occ_q
    cdef int start
    cdef int end
    cdef int q_start
    cdef int q_mid
    cdef int q_end
    cdef int node_start
    cdef int node_end_target
    cdef long long prefix_offset
    cdef long long prefix_endplus1
    cdef long long suffix_offset
    cdef int kk
    cdef int node_kk
    cdef int step_kk

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
    cdef double val

    for kk in range(norb):
        node_kk = <int>nodes_v[kk]
        step_kk = <int>steps_v[kk]
        idx_prefix[kk + 1] = idx_prefix[kk] + <long long>child_prefix[node_kk, step_kk]

    if not trusted:
        for t in range(n_pairs):
            p = <int>p_v[t]
            q = <int>q_v[t]
            if p < 0 or p >= norb or q < 0 or q >= norb:
                raise ValueError("orbital indices out of range")

    if 1:
        for t in range(n_pairs):
            p = <int>p_v[t]
            q = <int>q_v[t]
            wgt = <double>w_v[t]
            if wgt == 0.0:
                continue

            if not trusted:
                if p == q:
                    continue
                occ_p = _step_to_occ(<i8>steps_v[p])
                occ_q = _step_to_occ(<i8>steps_v[q])
                if occ_q <= 0 or occ_p >= 2:
                    continue

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

            node_start = <int>nodes_v[start]
            node_end_target = <int>nodes_v[end + 1]

            prefix_offset = idx_prefix[start]
            prefix_endplus1 = idx_prefix[end + 1]

            suffix_offset = (<long long>csf_idx) - prefix_endplus1

            # Reset DFS stack.
            st_k.clear()
            st_node.clear()
            st_w.clear()
            st_seg.clear()

            st_k.push_back(start)
            st_node.push_back(node_start)
            st_w.push_back(1.0)
            st_seg.push_back(0)

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
                                    if csf_i != csf_idx:
                                        val = wgt * w2
                                        if thresh_contrib <= 0.0 or fabs(val) > thresh_contrib:
                                            if val != 0.0:
                                                out_idx.push_back(csf_i)
                                                out_val.push_back(val)
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
                                    if csf_i != csf_idx:
                                        val = wgt * w2
                                        if thresh_contrib <= 0.0 or fabs(val) > thresh_contrib:
                                            if val != 0.0:
                                                out_idx.push_back(csf_i)
                                                out_val.push_back(val)
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
                                    if csf_i != csf_idx:
                                        val = wgt * w2
                                        if thresh_contrib <= 0.0 or fabs(val) > thresh_contrib:
                                            if val != 0.0:
                                                out_idx.push_back(csf_i)
                                                out_val.push_back(val)
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
                            if csf_i == csf_idx:
                                continue
                            val = wgt * w2
                            if thresh_contrib > 0.0 and fabs(val) <= thresh_contrib:
                                continue
                            if val != 0.0:
                                out_idx.push_back(csf_i)
                                out_val.push_back(val)
                        else:
                            st_k.push_back(k_next)
                            st_node.push_back(child_k)
                            st_w.push_back(w2)
                            st_seg.push_back(seg_idx2)

    cdef Py_ssize_t n_out = <Py_ssize_t>out_idx.size()
    cdef cnp.ndarray[cnp.int32_t, ndim=1] idx_arr = np.empty(n_out, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] val_arr = np.empty(n_out, dtype=np.float64)
    cdef cnp.int32_t[::1] idx_view = idx_arr
    cdef double[::1] val_view = val_arr
    cdef Py_ssize_t i
    for i in range(n_out):
        idx_view[i] = <cnp.int32_t>out_idx[i]
        val_view[i] = <double>out_val[i]
    return idx_arr, val_arr


