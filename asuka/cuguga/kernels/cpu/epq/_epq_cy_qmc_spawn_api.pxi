def qmc_spawn_hamiltonian_events_cy(
    drt,
    cnp.ndarray[cnp.float64_t, ndim=2] h1e,
    cnp.ndarray[cnp.float64_t, ndim=2] eri_mat,
    cnp.ndarray[cnp.int8_t, ndim=2] steps,
    cnp.ndarray[cnp.int32_t, ndim=2] nodes,
    cnp.ndarray[cnp.int32_t, ndim=1] x_idx,
    cnp.ndarray[cnp.float64_t, ndim=1] x_val,
    double eps,
    int nspawn_one,
    int nspawn_two,
    u64 seed,
    double initiator_t,
):
    """Spawn events for `-eps * H * x` (CPU compiled backend; OpenMP-capable).

    This implements the same proposal logic as `asuka.qmc.spawn.spawn_hamiltonian_events`,
    but runs the parent/event loops in Cython and can parallelize over parents.

    Notes
    -----
    - Output is a fixed-size COO buffer of length `m*(nspawn_one+nspawn_two)` where
      unused slots have `idx=-1` and `val=0`.
    - The caller is expected to filter out `idx < 0` entries.
    """

    _ensure_tables()

    cdef int norb = int(drt.norb)
    cdef int ncsf = int(drt.ncsf)
    cdef int m = int(x_idx.shape[0])
    if x_val.shape[0] != m:
        raise ValueError("x_idx and x_val must have the same length")
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one and nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one or nspawn_two must be > 0")

    if h1e.shape[0] != norb or h1e.shape[1] != norb:
        raise ValueError("h1e has wrong shape")
    cdef int nops = norb * norb
    if eri_mat.shape[0] != nops or eri_mat.shape[1] != nops:
        raise ValueError("eri_mat has wrong shape")
    if steps.shape[0] != ncsf or steps.shape[1] != norb:
        raise ValueError("steps has wrong shape for this DRT")
    if nodes.shape[0] != ncsf or nodes.shape[1] != norb + 1:
        raise ValueError("nodes has wrong shape for this DRT")

    if int(drt.nelec) > _seg_lut_max_b:
        raise ValueError("DRT nelec exceeds segment-value LUT max; increase _SEG_LUT_MAX_B")

    # Ensure child-prefix cache is available (Python call).
    cdef cnp.int64_t[:, ::1] child_prefix = _get_child_prefix(drt)

    cdef cnp.int32_t[:, ::1] child = drt.child
    cdef cnp.int16_t[::1] node_twos = drt.node_twos

    cdef double[:, ::1] h1e_v = h1e
    cdef double[:, ::1] eri_v = eri_mat
    cdef i8[:, ::1] steps_v = steps
    cdef cnp.int32_t[:, ::1] nodes_v = nodes
    cdef cnp.int32_t[::1] x_idx_v = x_idx
    cdef double[::1] x_val_v = x_val

    cdef int nspawn_total = nspawn_one + nspawn_two
    cdef cnp.ndarray[cnp.int32_t, ndim=1] evt_idx_arr = np.empty(m * nspawn_total, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] evt_val_arr = np.zeros(m * nspawn_total, dtype=np.float64)
    evt_idx_arr.fill(-1)

    cdef cnp.int32_t[::1] evt_idx = evt_idx_arr
    cdef double[::1] evt_val = evt_val_arr

    # h_base[p,q] = h1e[p,q] - 0.5 * Σ_t (p t | t q)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] h_base_arr = np.empty((norb, norb), dtype=np.float64)
    cdef double[:, ::1] h_base = h_base_arr
    cdef int p
    cdef int q
    cdef int t
    cdef double acc
    with nogil:
        for p in range(norb):
            for q in range(norb):
                acc = 0.0
                for t in range(norb):
                    acc += eri_v[p * norb + t, t * norb + q]
                h_base[p, q] = h1e_v[p, q] - 0.5 * acc

    cdef double scale_one = (-eps / <double>nspawn_one) if nspawn_one else 0.0
    cdef double scale_two = (-eps / <double>nspawn_two) if nspawn_two else 0.0
    cdef double thr = float(initiator_t)

    # Fixed-size orbital index work arrays (per thread stack); should cover any realistic CAS.
    cdef int MAX_ORB = 512
    if norb > MAX_ORB:
        raise ValueError("norb exceeds MAX_ORB for the compiled QMC spawn backend")

    cdef int parent_pos
    cdef int slot0
    cdef int slot
    cdef int j
    cdef double xj
    cdef i8* steps_j_ptr
    cdef cnp.int32_t* nodes_j_ptr
    cdef i8* steps_k_ptr
    cdef cnp.int32_t* nodes_k_ptr
    cdef i8 step_k
    cdef int src[512]
    cdef int dst[512]
    cdef int src_k_list[512]
    cdef int ns
    cdef int nd
    cdef int ns_k
    cdef int orb
    cdef bint allow_new
    cdef u64 st
    cdef int q_orb
    cdef int p_orb
    cdef int s_orb
    cdef int r_orb
    cdef int occ_r
    cdef int occ_s
    cdef int pq_id
    cdef int rs_id
    cdef int rr_id
    cdef double w_eff
    cdef double sum_rr
    cdef double v_pqrs
    cdef double inv_p_pair_one
    cdef double inv_p_pair_rs
    cdef double inv_p_pair_pq
    cdef int k_csf
    cdef int child_out
    cdef double coeff_out
    cdef double inv_p_out
    cdef int ok
    cdef int ok2
    cdef int ok3
    cdef double coeff_rs
    cdef double inv_p_rs_epq
    cdef double coeff_pq
    cdef double inv_p_pq_epq
    cdef int i_csf

    with nogil:
        for parent_pos in prange(m, schedule="static"):
            j = <int>x_idx_v[parent_pos]
            xj = x_val_v[parent_pos]
            if xj == 0.0:
                continue

            steps_j_ptr = &steps_v[j, 0]
            nodes_j_ptr = &nodes_v[j, 0]

            ns = 0
            nd = 0
            for orb in range(norb):
                step_k = steps_j_ptr[orb]
                if step_k != 0:
                    src[ns] = orb
                    ns = ns + 1
                if step_k != 3:
                    dst[nd] = orb
                    nd = nd + 1
            if ns == 0:
                continue

            allow_new = True
            if thr > 0.0 and fabs(xj) < thr:
                allow_new = False

            # Deterministic per-parent RNG stream (independent of thread scheduling).
            st = seed ^ (<u64>(<unsigned int>j) * <u64>0xD1B54A32D192ED03) ^ (<u64>(<unsigned int>parent_pos) * <u64>0x94D049BB133111EB)

            slot0 = parent_pos * nspawn_total

            # One-body part.
            if nspawn_one:
                inv_p_pair_one = <double>(ns * norb)
                for slot in range(slot0, slot0 + nspawn_one):
                    q_orb = src[_rand_below(&st, ns)]
                    p_orb = _rand_below(&st, norb)

                    pq_id = p_orb * norb + q_orb
                    w_eff = h_base[p_orb, q_orb]

                    # w_eff += 0.5 * Σ_r (p q | r r) occ_j[r], summed over occupied orbitals only.
                    sum_rr = 0.0
                    for t in range(ns):
                        r_orb = src[t]
                        occ_r = 2 if steps_j_ptr[r_orb] == 3 else 1
                        rr_id = r_orb * norb + r_orb
                        sum_rr = sum_rr + eri_v[pq_id, rr_id] * (<double>occ_r)
                    w_eff = w_eff + 0.5 * sum_rr

                    if w_eff == 0.0:
                        continue

                    ok = _epq_sample_one_nogil(
                        norb,
                        j,
                        p_orb,
                        q_orb,
                        steps_j_ptr,
                        nodes_j_ptr,
                        child,
                        node_twos,
                        child_prefix,
                        &st,
                        &child_out,
                        &coeff_out,
                        &inv_p_out,
                    )
                    if ok == 0:
                        continue

                    i_csf = child_out
                    if not allow_new:
                        if not _contains_sorted_i32(x_idx_v, i_csf):
                            continue

                    evt_idx[slot] = <cnp.int32_t>i_csf
                    evt_val[slot] = scale_one * xj * w_eff * coeff_out * inv_p_out * inv_p_pair_one

            # Two-body r!=s product term.
            if nspawn_two:
                if norb <= 1 or nd == 0:
                    continue
                for slot in range(slot0 + nspawn_one, slot0 + nspawn_total):
                    s_orb = src[_rand_below(&st, ns)]
                    occ_s = 2 if steps_j_ptr[s_orb] == 3 else 1

                    if occ_s == 1:
                        if nd <= 1:
                            continue
                        # Sample r from dst until r != s (uniform over dst \\ {s}).
                        while True:
                            r_orb = dst[_rand_below(&st, nd)]
                            if r_orb != s_orb:
                                break
                        inv_p_pair_rs = <double>(ns * (nd - 1))
                    else:
                        # occ_s == 2 => s not in dst.
                        r_orb = dst[_rand_below(&st, nd)]
                        inv_p_pair_rs = <double>(ns * nd)

                    rs_id = r_orb * norb + s_orb

                    ok2 = _epq_sample_one_nogil(
                        norb,
                        j,
                        r_orb,
                        s_orb,
                        steps_j_ptr,
                        nodes_j_ptr,
                        child,
                        node_twos,
                        child_prefix,
                        &st,
                        &child_out,
                        &coeff_out,
                        &inv_p_out,
                    )
                    if ok2 == 0:
                        continue
                    k_csf = child_out
                    coeff_rs = coeff_out
                    inv_p_rs_epq = inv_p_out

                    # Build occupied list for k.
                    steps_k_ptr = &steps_v[k_csf, 0]
                    nodes_k_ptr = &nodes_v[k_csf, 0]
                    ns_k = 0
                    for orb in range(norb):
                        if steps_k_ptr[orb] != 0:
                            src_k_list[ns_k] = orb
                            ns_k = ns_k + 1
                    if ns_k == 0:
                        continue

                    q_orb = src_k_list[_rand_below(&st, ns_k)]
                    p_orb = _rand_below(&st, norb)
                    pq_id = p_orb * norb + q_orb

                    v_pqrs = 0.5 * eri_v[pq_id, rs_id]
                    if v_pqrs == 0.0:
                        continue

                    ok3 = _epq_sample_one_nogil(
                        norb,
                        k_csf,
                        p_orb,
                        q_orb,
                        steps_k_ptr,
                        nodes_k_ptr,
                        child,
                        node_twos,
                        child_prefix,
                        &st,
                        &child_out,
                        &coeff_out,
                        &inv_p_out,
                    )
                    if ok3 == 0:
                        continue

                    i_csf = child_out
                    if not allow_new:
                        if not _contains_sorted_i32(x_idx_v, i_csf):
                            continue

                    coeff_pq = coeff_out
                    inv_p_pq_epq = inv_p_out
                    inv_p_pair_pq = <double>(ns_k * norb)

                    evt_idx[slot] = <cnp.int32_t>i_csf
                    evt_val[slot] = (
                        scale_two
                        * xj
                        * v_pqrs
                        * coeff_rs
                        * inv_p_rs_epq
                        * inv_p_pair_rs
                        * coeff_pq
                        * inv_p_pq_epq
                        * inv_p_pair_pq
                    )

    return evt_idx_arr, evt_val_arr

