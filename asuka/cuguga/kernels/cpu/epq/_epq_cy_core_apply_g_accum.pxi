def epq_apply_g_accum_cy(
    drt,
    int csf_idx,
    cnp.ndarray[cnp.float64_t, ndim=1] g_flat,
    cnp.ndarray[cnp.int8_t, ndim=1] steps,
    cnp.ndarray[cnp.int32_t, ndim=1] nodes,
    cnp.ndarray[cnp.float64_t, ndim=1] row,
    cnp.ndarray[cnp.npy_bool, ndim=1] mask,
    double thresh_gpq=0.0,
    double thresh_contrib=0.0,
):
    """Apply Σ_{p,q} g[p,q] * E_pq to |csf_idx> and accumulate into (row, mask).

    This is an in-place variant of :func:`epq_apply_g_cy` intended for the dense-row
    scratch path in the sparse row oracle. It avoids allocating and returning
    (idx,val) COO arrays.

    Returns
    -------
    (n_pairs, n_out)
        n_pairs: number of off-diagonal (p,q) pairs processed.
        n_out: number of nonzero contributions added to `row` (after thresholds).
    """

    _ensure_tables()

    cdef int norb = int(drt.norb)
    if int(drt.nelec) > _seg_lut_max_b:
        raise ValueError("DRT nelec exceeds segment-value LUT max; increase _SEG_LUT_MAX_B")
    cdef Py_ssize_t nops = <Py_ssize_t>(norb * norb)
    if g_flat.shape[0] != nops:
        raise ValueError("g_flat has wrong length for this DRT")

    cdef int ncsf = int(drt.ncsf)
    if row.shape[0] != ncsf or mask.shape[0] != ncsf:
        raise ValueError("row/mask have wrong length for this DRT")

    cdef i8[::1] steps_v = steps
    cdef cnp.int32_t[::1] nodes_v = nodes
    cdef double[::1] g_v = g_flat
    cdef double[::1] row_v = row
    cdef cnp.npy_bool[::1] mask_v = mask

    cdef cnp.int32_t[:, ::1] child = drt.child
    cdef cnp.int16_t[::1] node_twos = drt.node_twos

    # Cached prefix-walk counts (shared across all pairs for this DRT).
    cdef cnp.ndarray[cnp.int64_t, ndim=2] child_prefix_arr = _oracle_mod._child_prefix_walks(drt)  # type: ignore[attr-defined]
    cdef cnp.int64_t[:, ::1] child_prefix = child_prefix_arr

    # Prefix offsets for the reference path:
    cdef vector[long long] idx_prefix
    idx_prefix.resize(norb + 1)
    idx_prefix[0] = 0

    # Precompute reference d_k and b_k (ket path) once per state.
    cdef vector[int] d_ref
    cdef vector[int] b_ref
    cdef vector[int] occ_ref
    d_ref.resize(norb)
    b_ref.resize(norb)
    occ_ref.resize(norb)

    # Reusable DFS stacks (SOA vectors).
    cdef vector[int] st_k
    cdef vector[int] st_node
    cdef vector[double] st_w
    cdef vector[long long] st_seg

    cdef int kk
    cdef int node_kk
    cdef int step_kk
    for kk in range(norb):
        node_kk = <int>nodes_v[kk]
        step_kk = <int>steps_v[kk]
        idx_prefix[kk + 1] = idx_prefix[kk] + <long long>child_prefix[node_kk, step_kk]
        d_ref[kk] = step_kk
        b_ref[kk] = <int>node_twos[<int>nodes_v[kk + 1]]
        occ_ref[kk] = _step_to_occ(<i8>step_kk)

    cdef double tg = float(thresh_gpq)
    cdef double tc = float(thresh_contrib)

    cdef long long n_out = 0

    # Diagonal contribution: Σ_p g[p,p] * occ[p]
    cdef double diag_contrib = 0.0
    cdef int p
    cdef double wgt
    cdef int occ_p
    for p in range(norb):
        wgt = <double>g_v[<Py_ssize_t>(p * norb + p)]
        if wgt == 0.0:
            continue
        if tg > 0.0 and fabs(wgt) <= tg:
            continue
        occ_p = occ_ref[p]
        if occ_p != 0:
            diag_contrib += wgt * <double>occ_p

    if diag_contrib != 0.0:
        if tc > 0.0 and fabs(diag_contrib) <= tc:
            diag_contrib = 0.0
        if diag_contrib != 0.0:
            row_v[csf_idx] += diag_contrib
            mask_v[csf_idx] = 1
            n_out += 1

    cdef int q
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

    cdef long long n_pairs = 0

    for p in range(norb):
        occ_p = occ_ref[p]
        if occ_p >= 2:
            continue
        for q in range(norb):
            if q == p:
                continue
            occ_q = occ_ref[q]
            if occ_q <= 0:
                continue

            wgt = <double>g_v[<Py_ssize_t>(p * norb + q)]
            if wgt == 0.0:
                continue
            if tg > 0.0 and fabs(wgt) <= tg:
                continue

            n_pairs += 1

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

                d_k = d_ref[k]
                b_k = b_ref[k]
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
                                        if tc <= 0.0 or fabs(val) > tc:
                                            if val != 0.0:
                                                row_v[csf_i] += val
                                                mask_v[csf_i] = 1
                                                n_out += 1
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
                                        if tc <= 0.0 or fabs(val) > tc:
                                            if val != 0.0:
                                                row_v[csf_i] += val
                                                mask_v[csf_i] = 1
                                                n_out += 1
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
                                        if tc <= 0.0 or fabs(val) > tc:
                                            if val != 0.0:
                                                row_v[csf_i] += val
                                                mask_v[csf_i] = 1
                                                n_out += 1
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
                            if tc > 0.0 and fabs(val) <= tc:
                                continue
                            if val != 0.0:
                                row_v[csf_i] += val
                                mask_v[csf_i] = 1
                                n_out += 1
                        else:
                            st_k.push_back(k_next)
                            st_node.push_back(child_k)
                            st_w.push_back(w2)
                            st_seg.push_back(seg_idx2)

    return <int>n_pairs, <int>n_out


