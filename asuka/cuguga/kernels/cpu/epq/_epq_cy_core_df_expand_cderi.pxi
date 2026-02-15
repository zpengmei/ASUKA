def expand_cderi_s2_to_full_pairs_and_norm_cy(cderi_s2, int norb):
    """Expand packed s2 MO CDERIs to full ordered pairs and compute pair norms.

    Parameters
    ----------
    cderi_s2:
        2D float64 array with shape (naux, npair_s2) or (npair_s2, naux).
    norb:
        Number of MOs in the subspace.

    Returns
    -------
    l_full:
        (norb*norb, naux) float64 C-ordered array of ordered-pair DF vectors.
    pair_norm:
        (norb*norb,) float64 array of 2-norms for each ordered pair.
    """

    norb = int(norb)
    if norb <= 0:
        raise ValueError("norb must be > 0")

    cderi = np.asarray(cderi_s2, dtype=np.float64)
    if cderi.ndim != 2:
        raise ValueError("expected a 2D array for cderi_s2")

    cdef int npair = norb * (norb + 1) // 2
    cdef int naux
    cdef bint src_pair_first = False

    if cderi.shape[1] == npair:
        # (naux, npair_s2)
        cderi = np.ascontiguousarray(cderi)
        naux = <int>cderi.shape[0]
        src_pair_first = False
    elif cderi.shape[0] == npair:
        # (npair_s2, naux)
        cderi = np.ascontiguousarray(cderi)
        naux = <int>cderi.shape[1]
        src_pair_first = True
    else:
        raise ValueError(f"unexpected cderi_s2 shape {cderi.shape} for norb={norb}")

    cdef int nops = norb * norb
    l_full = np.empty((nops, naux), dtype=np.float64, order="C")
    pair_norm = np.empty((nops,), dtype=np.float64)

    cdef double[:, ::1] src = cderi
    cdef double[:, ::1] out = l_full
    cdef double[::1] norm = pair_norm
    cdef int i, p, q, a, b, pid, L
    cdef double v, acc

    if src_pair_first:
        with nogil:
            for i in range(nops):
                p = i // norb
                q = i - p * norb
                if p >= q:
                    a = p
                    b = q
                else:
                    a = q
                    b = p
                pid = a * (a + 1) // 2 + b
                acc = 0.0
                for L in range(naux):
                    v = src[pid, L]
                    out[i, L] = v
                    acc += v * v
                norm[i] = sqrt(acc)
    else:
        with nogil:
            for i in range(nops):
                p = i // norb
                q = i - p * norb
                if p >= q:
                    a = p
                    b = q
                else:
                    a = q
                    b = p
                pid = a * (a + 1) // 2 + b
                acc = 0.0
                for L in range(naux):
                    v = src[L, pid]
                    out[i, L] = v
                    acc += v * v
                norm[i] = sqrt(acc)

    return l_full, pair_norm



