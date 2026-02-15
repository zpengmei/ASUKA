def csc_matmul_dense_inplace_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] = A @ x`` for a CSC matrix A given by (indptr, indices, data).

    Notes
    -----
    - A is interpreted as shape (nrow, ncol) with ncol inferred as ``indptr.size-1``.
    - ``x`` must have shape (ncol, nvec) and C-contiguous (last axis contiguous).
    - ``out`` must have shape (nrow, nvec) and C-contiguous (last axis contiguous).
    - Releases the GIL for the hot loops to allow Python-level threading.
    """

    cdef Py_ssize_t ncol = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nrow = <Py_ssize_t>out.shape[0]
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]

    if x.shape[0] != ncol or x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != ncol + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double coeff
    cdef Py_ssize_t n = nrow * nvec
    cdef Py_ssize_t joff

    with nogil:
        for i in range(n):
            out_p[i] = 0.0

        if nvec == 1:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                coeff = x_p[j]
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    out_p[row] += data_p[k] * coeff
        else:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                joff = j * nvec
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    coeff = data_p[k]
                    i = row * nvec
                    for v in range(nvec):
                        out_p[i + v] += coeff * x_p[joff + v]


def csc_matmul_dense_add_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] += A @ x`` for a CSC matrix A given by (indptr, indices, data).

    See :func:`csc_matmul_dense_inplace_cy` for shape/contiguity requirements.
    Releases the GIL for the hot loops to allow Python-level threading.
    """

    cdef Py_ssize_t ncol = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]

    if x.shape[0] != ncol or x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != ncol + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double coeff
    cdef Py_ssize_t joff
    cdef Py_ssize_t i

    with nogil:
        if nvec == 1:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                coeff = x_p[j]
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    out_p[row] += data_p[k] * coeff
        else:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                joff = j * nvec
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    coeff = data_p[k]
                    i = row * nvec
                    for v in range(nvec):
                        out_p[i + v] += coeff * x_p[joff + v]


def csc_matmul_dense_add_scaled_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
    double alpha,
):
    """Compute ``out[:] += alpha * (A @ x)`` for a CSC matrix A given by (indptr, indices, data).

    See :func:`csc_matmul_dense_inplace_cy` for shape/contiguity requirements.
    Releases the GIL for the hot loops to allow Python-level threading.
    """

    cdef Py_ssize_t ncol = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]
    cdef double alpha_v = <double>alpha

    if x.shape[0] != ncol or x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != ncol + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double coeff
    cdef Py_ssize_t joff
    cdef Py_ssize_t i

    with nogil:
        if nvec == 1:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                coeff = alpha_v * x_p[j]
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    out_p[row] += data_p[k] * coeff
        else:
            for j in range(ncol):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                joff = j * nvec
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    coeff = alpha_v * data_p[k]
                    i = row * nvec
                    for v in range(nvec):
                        out_p[i + v] += coeff * x_p[joff + v]


cdef inline void _csc_matmul_dense_inplace_ptr(
    cnp.int32_t* indptr_p,
    cnp.int32_t* indices_p,
    double* data_p,
    double* x_p,
    double* out_p,
    Py_ssize_t n,
    Py_ssize_t nvec,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double coeff
    cdef Py_ssize_t n_tot = n * nvec
    cdef Py_ssize_t joff

    for i in range(n_tot):
        out_p[i] = 0.0

    if nvec == 1:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            coeff = x_p[j]
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                out_p[row] += data_p[k] * coeff
        return

    for j in range(n):
        start = <Py_ssize_t>indptr_p[j]
        end = <Py_ssize_t>indptr_p[j + 1]
        if start == end:
            continue
        joff = j * nvec
        for k in range(start, end):
            row = <Py_ssize_t>indices_p[k]
            coeff = data_p[k]
            i = row * nvec
            for v in range(nvec):
                out_p[i + v] += coeff * x_p[joff + v]


cdef inline void _csc_matmul_dense_add_ptr(
    cnp.int32_t* indptr_p,
    cnp.int32_t* indices_p,
    double* data_p,
    double* x_p,
    double* out_p,
    Py_ssize_t n,
    Py_ssize_t nvec,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double coeff
    cdef Py_ssize_t joff
    cdef Py_ssize_t i

    if nvec == 1:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            coeff = x_p[j]
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                out_p[row] += data_p[k] * coeff
        return

    for j in range(n):
        start = <Py_ssize_t>indptr_p[j]
        end = <Py_ssize_t>indptr_p[j + 1]
        if start == end:
            continue
        joff = j * nvec
        for k in range(start, end):
            row = <Py_ssize_t>indices_p[k]
            coeff = data_p[k]
            i = row * nvec
            for v in range(nvec):
                out_p[i + v] += coeff * x_p[joff + v]


def csc_matmul_dense_inplace_many_indexed_cy(
    indptr_list,
    indices_list,
    data_list,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
    cnp.ndarray[cnp.int32_t, ndim=1] out_rows,
    int nthreads=0,
):
    """Compute ``out[out_rows[k]] = A_k @ x`` for a batch of CSC operators.

    Parameters
    ----------
    indptr_list, indices_list, data_list
        Sequences of CSC arrays (int32 indptr/indices, float64 data) of equal length.
    x
        Dense input array of shape (n, nvec), C-contiguous.
    out
        Dense row-major array of shape (nout, n*nvec), C-contiguous.
    out_rows
        int32 array of length batch mapping each operator to a row in `out`.
    nthreads
        If >0, sets OpenMP thread count. Otherwise uses the OpenMP runtime default.
    """

    cdef Py_ssize_t batch = <Py_ssize_t>out_rows.shape[0]
    if batch <= 0:
        return
    if len(indptr_list) != batch or len(indices_list) != batch or len(data_list) != batch:
        raise ValueError("indptr_list/indices_list/data_list must match out_rows length")

    # Validate row indices and infer shapes.
    cdef Py_ssize_t k
    cdef Py_ssize_t row
    for k in range(batch):
        row = <Py_ssize_t>out_rows[k]
        if row < 0 or row >= <Py_ssize_t>out.shape[0]:
            raise ValueError("out_rows contains out-of-range indices")

    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr0 = np.asarray(indptr_list[0], dtype=np.int32)
    cdef Py_ssize_t n = <Py_ssize_t>(indptr0.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>x.shape[1]
    if x.shape[0] != n:
        raise ValueError("x has wrong shape for these CSC operators")
    cdef Py_ssize_t m = n * nvec
    if out.shape[1] != m:
        raise ValueError("out must have shape (nout, n*nvec)")

    cdef vector[cnp.int32_t*] indptr_ptrs
    cdef vector[cnp.int32_t*] indices_ptrs
    cdef vector[double*] data_ptrs
    indptr_ptrs.reserve(<size_t>batch)
    indices_ptrs.reserve(<size_t>batch)
    data_ptrs.reserve(<size_t>batch)

    keepalive = []
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data
    for k in range(batch):
        indptr = np.asarray(indptr_list[k], dtype=np.int32)
        indices = np.asarray(indices_list[k], dtype=np.int32)
        data = np.asarray(data_list[k], dtype=np.float64)
        if indptr.shape[0] != n + 1:
            raise ValueError("indptr has wrong length for this CSC operator batch")
        if indices.shape[0] != data.shape[0]:
            raise ValueError("indices/data length mismatch for this CSC operator batch")
        indptr_ptrs.push_back(<cnp.int32_t*>indptr.data)
        indices_ptrs.push_back(<cnp.int32_t*>indices.data)
        data_ptrs.push_back(<double*>data.data)
        keepalive.append(indptr)
        keepalive.append(indices)
        keepalive.append(data)

    if nthreads > 0:
        guga_openmp_set_num_threads(int(nthreads))

    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data
    cdef cnp.int32_t* rows_p = <cnp.int32_t*>out_rows.data

    with nogil:
        for k in prange(batch, schedule="static"):
            row = <Py_ssize_t>rows_p[k]
            _csc_matmul_dense_inplace_ptr(
                indptr_ptrs[k],
                indices_ptrs[k],
                data_ptrs[k],
                x_p,
                out_p + row * m,
                n,
                nvec,
            )


def csc_matmul_dense_add_many_indexed_omp_cy(
    indptr_list,
    indices_list,
    data_list,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x_rows,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out_parts,
    cnp.ndarray[cnp.int32_t, ndim=1] row_ids,
    int nthreads=0,
):
    """Compute ``out_parts[tid] += A_k @ x_rows[row_ids[k]]`` in parallel.

    This is intended for pair-parallel accumulation without write races: each OpenMP thread writes
    only to its private `out_parts[tid]` buffer.

    Parameters
    ----------
    indptr_list, indices_list, data_list
        Sequences of CSC arrays (int32 indptr/indices, float64 data) of equal length.
    x_rows
        Dense row-major array of shape (nrow, n*nvec), C-contiguous.
    out_parts
        Per-thread output buffers of shape (nthreads, n*nvec), C-contiguous. Must be zeroed by caller.
    row_ids
        int32 array of length batch mapping each operator to a row in `x_rows`.
    nthreads
        If >0, sets OpenMP thread count; otherwise uses `out_parts.shape[0]`.
    """

    cdef Py_ssize_t batch = <Py_ssize_t>row_ids.shape[0]
    if batch <= 0:
        return
    if len(indptr_list) != batch or len(indices_list) != batch or len(data_list) != batch:
        raise ValueError("indptr_list/indices_list/data_list must match row_ids length")

    cdef Py_ssize_t nt = <Py_ssize_t>out_parts.shape[0]
    if nt <= 0:
        raise ValueError("out_parts must have at least one thread buffer")
    if nthreads > 0:
        nt = <Py_ssize_t>nthreads
        if out_parts.shape[0] < nt:
            raise ValueError("out_parts has fewer rows than requested nthreads")
        guga_openmp_set_num_threads(int(nthreads))

    # Validate row indices and infer shapes.
    cdef Py_ssize_t k
    cdef Py_ssize_t row
    for k in range(batch):
        row = <Py_ssize_t>row_ids[k]
        if row < 0 or row >= <Py_ssize_t>x_rows.shape[0]:
            raise ValueError("row_ids contains out-of-range indices")

    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr0 = np.asarray(indptr_list[0], dtype=np.int32)
    cdef Py_ssize_t n = <Py_ssize_t>(indptr0.shape[0] - 1)
    if n <= 0:
        raise ValueError("invalid CSC operator dimension")
    cdef Py_ssize_t m = <Py_ssize_t>x_rows.shape[1]
    if out_parts.shape[1] != m:
        raise ValueError("out_parts must have the same second dimension as x_rows")
    if m % n != 0:
        raise ValueError("x_rows second dimension must be divisible by n")
    cdef Py_ssize_t nvec = m // n

    cdef vector[cnp.int32_t*] indptr_ptrs
    cdef vector[cnp.int32_t*] indices_ptrs
    cdef vector[double*] data_ptrs
    indptr_ptrs.reserve(<size_t>batch)
    indices_ptrs.reserve(<size_t>batch)
    data_ptrs.reserve(<size_t>batch)

    keepalive = []
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data
    for k in range(batch):
        indptr = np.asarray(indptr_list[k], dtype=np.int32)
        indices = np.asarray(indices_list[k], dtype=np.int32)
        data = np.asarray(data_list[k], dtype=np.float64)
        if indptr.shape[0] != n + 1:
            raise ValueError("indptr has wrong length for this CSC operator batch")
        if indices.shape[0] != data.shape[0]:
            raise ValueError("indices/data length mismatch for this CSC operator batch")
        indptr_ptrs.push_back(<cnp.int32_t*>indptr.data)
        indices_ptrs.push_back(<cnp.int32_t*>indices.data)
        data_ptrs.push_back(<double*>data.data)
        keepalive.append(indptr)
        keepalive.append(indices)
        keepalive.append(data)

    cdef double* x_p = <double*>x_rows.data
    cdef double* out_p = <double*>out_parts.data
    cdef cnp.int32_t* rows_p = <cnp.int32_t*>row_ids.data

    cdef Py_ssize_t tid
    with nogil:
        for k in prange(batch, schedule="static"):
            tid = <Py_ssize_t>threadid()
            row = <Py_ssize_t>rows_p[k]
            _csc_matmul_dense_add_ptr(
                indptr_ptrs[k],
                indices_ptrs[k],
                data_ptrs[k],
                x_p + row * m,
                out_p + tid * m,
                n,
                nvec,
            )


def csc_matmul_dense_sym_inplace_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] = (A + A.T) @ x`` for a CSC matrix A given by (indptr, indices, data).

    Notes
    -----
    - A is interpreted as shape (n, n) with n inferred as ``indptr.size-1``.
    - ``x`` must have shape (n, nvec) and C-contiguous (last axis contiguous).
    - ``out`` must have shape (n, nvec) and C-contiguous (last axis contiguous).
    - Releases the GIL for the hot loops to allow Python-level threading.
    """

    cdef Py_ssize_t n = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]

    if out.shape[0] != n:
        raise ValueError("out has wrong shape for this CSC operator")
    if x.shape[0] != n or x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != n + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double a
    cdef Py_ssize_t n_tot = n * nvec
    cdef Py_ssize_t joff
    cdef Py_ssize_t roff
    cdef double acc
    cdef double xj
    cdef double acc_buf[8]

    with nogil:
        for i in range(n_tot):
            out_p[i] = 0.0

        if nvec == 1:
            for j in range(n):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                xj = x_p[j]
                acc = 0.0
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    a = data_p[k]
                    out_p[row] += a * xj
                    acc += a * x_p[row]
                out_p[j] += acc
        else:
            if nvec <= 8:
                for j in range(n):
                    start = <Py_ssize_t>indptr_p[j]
                    end = <Py_ssize_t>indptr_p[j + 1]
                    if start == end:
                        continue
                    joff = j * nvec
                    for v in range(nvec):
                        acc_buf[v] = 0.0
                    for k in range(start, end):
                        row = <Py_ssize_t>indices_p[k]
                        a = data_p[k]
                        roff = row * nvec
                        for v in range(nvec):
                            out_p[roff + v] += a * x_p[joff + v]
                            acc_buf[v] += a * x_p[roff + v]
                    for v in range(nvec):
                        out_p[joff + v] += acc_buf[v]
            else:
                for j in range(n):
                    start = <Py_ssize_t>indptr_p[j]
                    end = <Py_ssize_t>indptr_p[j + 1]
                    if start == end:
                        continue
                    joff = j * nvec
                    for k in range(start, end):
                        row = <Py_ssize_t>indices_p[k]
                        a = data_p[k]
                        roff = row * nvec
                        for v in range(nvec):
                            out_p[roff + v] += a * x_p[joff + v]
                            out_p[joff + v] += a * x_p[roff + v]


def csc_matmul_dense_sym_add_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] += (A + A.T) @ x`` for a CSC matrix A given by (indptr, indices, data).

    See :func:`csc_matmul_dense_sym_inplace_cy` for shape/contiguity requirements.
    Releases the GIL for the hot loops to allow Python-level threading.
    """

    cdef Py_ssize_t n = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]

    if out.shape[0] != n:
        raise ValueError("out has wrong shape for this CSC operator")
    if x.shape[0] != n or x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != n + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double a
    cdef Py_ssize_t joff
    cdef Py_ssize_t roff
    cdef double acc
    cdef double xj
    cdef double acc_buf[8]

    with nogil:
        if nvec == 1:
            for j in range(n):
                start = <Py_ssize_t>indptr_p[j]
                end = <Py_ssize_t>indptr_p[j + 1]
                if start == end:
                    continue
                xj = x_p[j]
                acc = 0.0
                for k in range(start, end):
                    row = <Py_ssize_t>indices_p[k]
                    a = data_p[k]
                    out_p[row] += a * xj
                    acc += a * x_p[row]
                out_p[j] += acc
        else:
            if nvec <= 8:
                for j in range(n):
                    start = <Py_ssize_t>indptr_p[j]
                    end = <Py_ssize_t>indptr_p[j + 1]
                    if start == end:
                        continue
                    joff = j * nvec
                    for v in range(nvec):
                        acc_buf[v] = 0.0
                    for k in range(start, end):
                        row = <Py_ssize_t>indices_p[k]
                        a = data_p[k]
                        roff = row * nvec
                        for v in range(nvec):
                            out_p[roff + v] += a * x_p[joff + v]
                            acc_buf[v] += a * x_p[roff + v]
                    for v in range(nvec):
                        out_p[joff + v] += acc_buf[v]
            else:
                for j in range(n):
                    start = <Py_ssize_t>indptr_p[j]
                    end = <Py_ssize_t>indptr_p[j + 1]
                    if start == end:
                        continue
                    joff = j * nvec
                    for k in range(start, end):
                        row = <Py_ssize_t>indices_p[k]
                        a = data_p[k]
                        roff = row * nvec
                        for v in range(nvec):
                            out_p[roff + v] += a * x_p[joff + v]
                            out_p[joff + v] += a * x_p[roff + v]


cdef inline void _csc_matmul_dense_sym_inplace_ptr(
    cnp.int32_t* indptr_p,
    cnp.int32_t* indices_p,
    double* data_p,
    double* x_p,
    double* out_p,
    Py_ssize_t n,
    Py_ssize_t nvec,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double a
    cdef Py_ssize_t n_tot = n * nvec
    cdef Py_ssize_t joff
    cdef Py_ssize_t roff
    cdef double acc
    cdef double xj
    cdef double acc_buf[8]

    for i in range(n_tot):
        out_p[i] = 0.0

    if nvec == 1:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            xj = x_p[j]
            acc = 0.0
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                a = data_p[k]
                out_p[row] += a * xj
                acc += a * x_p[row]
            out_p[j] += acc
        return

    if nvec <= 8:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            joff = j * nvec
            for v in range(nvec):
                acc_buf[v] = 0.0
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                a = data_p[k]
                roff = row * nvec
                for v in range(nvec):
                    out_p[roff + v] += a * x_p[joff + v]
                    acc_buf[v] += a * x_p[roff + v]
            for v in range(nvec):
                out_p[joff + v] += acc_buf[v]
        return

    for j in range(n):
        start = <Py_ssize_t>indptr_p[j]
        end = <Py_ssize_t>indptr_p[j + 1]
        if start == end:
            continue
        joff = j * nvec
        for k in range(start, end):
            row = <Py_ssize_t>indices_p[k]
            a = data_p[k]
            roff = row * nvec
            for v in range(nvec):
                out_p[roff + v] += a * x_p[joff + v]
                out_p[joff + v] += a * x_p[roff + v]


cdef inline void _csc_matmul_dense_sym_add_ptr(
    cnp.int32_t* indptr_p,
    cnp.int32_t* indices_p,
    double* data_p,
    double* x_p,
    double* out_p,
    Py_ssize_t n,
    Py_ssize_t nvec,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double a
    cdef Py_ssize_t joff
    cdef Py_ssize_t roff
    cdef double acc
    cdef double xj
    cdef double acc_buf[8]

    if nvec == 1:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            xj = x_p[j]
            acc = 0.0
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                a = data_p[k]
                out_p[row] += a * xj
                acc += a * x_p[row]
            out_p[j] += acc
        return

    if nvec <= 8:
        for j in range(n):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            joff = j * nvec
            for v in range(nvec):
                acc_buf[v] = 0.0
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                a = data_p[k]
                roff = row * nvec
                for v in range(nvec):
                    out_p[roff + v] += a * x_p[joff + v]
                    acc_buf[v] += a * x_p[roff + v]
            for v in range(nvec):
                out_p[joff + v] += acc_buf[v]
        return

    for j in range(n):
        start = <Py_ssize_t>indptr_p[j]
        end = <Py_ssize_t>indptr_p[j + 1]
        if start == end:
            continue
        joff = j * nvec
        for k in range(start, end):
            row = <Py_ssize_t>indices_p[k]
            a = data_p[k]
            roff = row * nvec
            for v in range(nvec):
                out_p[roff + v] += a * x_p[joff + v]
                out_p[joff + v] += a * x_p[roff + v]


def csc_matmul_dense_sym_inplace_many_indexed_cy(
    indptr_list,
    indices_list,
    data_list,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
    cnp.ndarray[cnp.int32_t, ndim=1] out_rows,
    int nthreads=0,
):
    """Compute ``out[out_rows[k]] = (A_k + A_k.T) @ x`` for a batch of CSC matrices.

    Parameters
    ----------
    indptr_list, indices_list, data_list
        Sequences of CSC arrays (int32 indptr/indices, float64 data) of equal length.
    x
        Dense input array of shape (n, nvec), C-contiguous.
    out
        Dense output buffer of shape (nout, n*nvec), C-contiguous; each selected row is treated
        as a (n, nvec) C-contiguous matrix.
    out_rows
        int32 array of length batch mapping each operator to a row in `out`.
    nthreads
        If >0 and OpenMP is enabled for this extension (build with ``GUGA_USE_OPENMP=1``),
        sets the OpenMP thread count for the internal `prange`.
    """

    cdef Py_ssize_t batch = <Py_ssize_t>out_rows.shape[0]
    if batch <= 0:
        return
    if len(indptr_list) != batch or len(indices_list) != batch or len(data_list) != batch:
        raise ValueError("indptr_list/indices_list/data_list must match out_rows length")

    # Validate row indices in Python (avoid exceptions in nogil region).
    cdef Py_ssize_t k
    cdef Py_ssize_t row
    for k in range(batch):
        row = <Py_ssize_t>out_rows[k]
        if row < 0 or row >= <Py_ssize_t>out.shape[0]:
            raise ValueError("out_rows contains out-of-range indices")

    # Infer n from the first operator.
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr0 = np.asarray(indptr_list[0], dtype=np.int32)
    cdef Py_ssize_t n = <Py_ssize_t>(indptr0.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>x.shape[1]
    if x.shape[0] != n:
        raise ValueError("x has wrong shape for these CSC operators")
    cdef Py_ssize_t m = n * nvec
    if out.shape[1] != m:
        raise ValueError("out must have shape (nout, n*nvec)")

    cdef vector[cnp.int32_t*] indptr_ptrs
    cdef vector[cnp.int32_t*] indices_ptrs
    cdef vector[double*] data_ptrs
    indptr_ptrs.reserve(<size_t>batch)
    indices_ptrs.reserve(<size_t>batch)
    data_ptrs.reserve(<size_t>batch)

    keepalive = []
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data
    for k in range(batch):
        indptr = np.asarray(indptr_list[k], dtype=np.int32)
        indices = np.asarray(indices_list[k], dtype=np.int32)
        data = np.asarray(data_list[k], dtype=np.float64)
        if indptr.shape[0] != n + 1:
            raise ValueError("indptr has wrong length for this CSC operator batch")
        if indices.shape[0] != data.shape[0]:
            raise ValueError("indices/data length mismatch for this CSC operator batch")
        indptr_ptrs.push_back(<cnp.int32_t*>indptr.data)
        indices_ptrs.push_back(<cnp.int32_t*>indices.data)
        data_ptrs.push_back(<double*>data.data)
        keepalive.append(indptr)
        keepalive.append(indices)
        keepalive.append(data)

    if nthreads > 0:
        guga_openmp_set_num_threads(int(nthreads))

    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data
    cdef cnp.int32_t* rows_p = <cnp.int32_t*>out_rows.data

    with nogil:
        for k in prange(batch, schedule="static"):
            row = <Py_ssize_t>rows_p[k]
            _csc_matmul_dense_sym_inplace_ptr(
                indptr_ptrs[k],
                indices_ptrs[k],
                data_ptrs[k],
                x_p,
                out_p + row * m,
                n,
                nvec,
            )


def csc_matmul_dense_sym_add_many_indexed_omp_cy(
    indptr_list,
    indices_list,
    data_list,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x_rows,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out_parts,
    cnp.ndarray[cnp.int32_t, ndim=1] row_ids,
    int nthreads=0,
):
    """Compute ``out_parts[tid] += (A_k + A_k.T) @ x_rows[row_ids[k]]`` in parallel.

    This is intended for pair-parallel accumulation without write races: each OpenMP thread writes
    only to its private `out_parts[tid]` buffer.

    Parameters
    ----------
    indptr_list, indices_list, data_list
        Sequences of CSC arrays (int32 indptr/indices, float64 data) of equal length.
    x_rows
        Dense row-major array of shape (nrow, n*nvec), C-contiguous.
    out_parts
        Per-thread output buffers of shape (nthreads, n*nvec), C-contiguous. Must be zeroed by caller.
    row_ids
        int32 array of length batch mapping each operator to a row in `x_rows`.
    nthreads
        If >0, sets OpenMP thread count; otherwise uses `out_parts.shape[0]`.
    """

    cdef Py_ssize_t batch = <Py_ssize_t>row_ids.shape[0]
    if batch <= 0:
        return
    if len(indptr_list) != batch or len(indices_list) != batch or len(data_list) != batch:
        raise ValueError("indptr_list/indices_list/data_list must match row_ids length")

    cdef Py_ssize_t nt = <Py_ssize_t>out_parts.shape[0]
    if nt <= 0:
        raise ValueError("out_parts must have at least one thread buffer")
    if nthreads > 0:
        nt = <Py_ssize_t>nthreads
        if out_parts.shape[0] < nt:
            raise ValueError("out_parts has fewer rows than requested nthreads")
        guga_openmp_set_num_threads(int(nthreads))

    # Validate row indices and infer shapes.
    cdef Py_ssize_t k
    cdef Py_ssize_t row
    for k in range(batch):
        row = <Py_ssize_t>row_ids[k]
        if row < 0 or row >= <Py_ssize_t>x_rows.shape[0]:
            raise ValueError("row_ids contains out-of-range indices")

    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr0 = np.asarray(indptr_list[0], dtype=np.int32)
    cdef Py_ssize_t n = <Py_ssize_t>(indptr0.shape[0] - 1)
    if n <= 0:
        raise ValueError("invalid CSC operator dimension")
    cdef Py_ssize_t m = <Py_ssize_t>x_rows.shape[1]
    if out_parts.shape[1] != m:
        raise ValueError("out_parts must have the same second dimension as x_rows")
    if m % n != 0:
        raise ValueError("x_rows second dimension must be divisible by n")
    cdef Py_ssize_t nvec = m // n

    cdef vector[cnp.int32_t*] indptr_ptrs
    cdef vector[cnp.int32_t*] indices_ptrs
    cdef vector[double*] data_ptrs
    indptr_ptrs.reserve(<size_t>batch)
    indices_ptrs.reserve(<size_t>batch)
    data_ptrs.reserve(<size_t>batch)

    keepalive = []
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data
    for k in range(batch):
        indptr = np.asarray(indptr_list[k], dtype=np.int32)
        indices = np.asarray(indices_list[k], dtype=np.int32)
        data = np.asarray(data_list[k], dtype=np.float64)
        if indptr.shape[0] != n + 1:
            raise ValueError("indptr has wrong length for this CSC operator batch")
        if indices.shape[0] != data.shape[0]:
            raise ValueError("indices/data length mismatch for this CSC operator batch")
        indptr_ptrs.push_back(<cnp.int32_t*>indptr.data)
        indices_ptrs.push_back(<cnp.int32_t*>indices.data)
        data_ptrs.push_back(<double*>data.data)
        keepalive.append(indptr)
        keepalive.append(indices)
        keepalive.append(data)

    cdef double* x_p = <double*>x_rows.data
    cdef double* out_p = <double*>out_parts.data
    cdef cnp.int32_t* rows_p = <cnp.int32_t*>row_ids.data

    cdef Py_ssize_t tid
    with nogil:
        for k in prange(batch, schedule="static"):
            tid = <Py_ssize_t>threadid()
            row = <Py_ssize_t>rows_p[k]
            _csc_matmul_dense_sym_add_ptr(
                indptr_ptrs[k],
                indices_ptrs[k],
                data_ptrs[k],
                x_p + row * m,
                out_p + tid * m,
                n,
                nvec,
            )


def csc_quadratic_form_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x,
):
    """Return ``x.T @ A @ x`` for a CSC matrix A given by (indptr, indices, data).

    Notes
    -----
    - A is interpreted as shape (nrow, ncol) with ncol inferred as ``indptr.size-1``.
    - ``x`` must have shape (ncol,) and be C-contiguous.
    - Releases the GIL for the hot loops.
    """

    cdef Py_ssize_t ncol = <Py_ssize_t>(indptr.shape[0] - 1)
    if x.shape[0] != ncol:
        raise ValueError("x has wrong shape for this CSC operator")
    if indptr.shape[0] != ncol + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data

    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double acc = 0.0
    cdef double xj

    with nogil:
        for j in range(ncol):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            xj = x_p[j]
            if xj == 0.0:
                continue
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                acc += data_p[k] * x_p[row] * xj
    return <double>acc


def csc_bilinear_form_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] y,
):
    """Return ``y.T @ A @ x`` for a CSC matrix A given by (indptr, indices, data).

    Notes
    -----
    - A is interpreted as shape (nrow, ncol) with ncol inferred as ``indptr.size-1``.
    - ``x`` and ``y`` must have shape (ncol,) and be C-contiguous.
    - Releases the GIL for the hot loops.
    """

    cdef Py_ssize_t ncol = <Py_ssize_t>(indptr.shape[0] - 1)
    if x.shape[0] != ncol or y.shape[0] != ncol:
        raise ValueError("x/y have wrong shape for this CSC operator")
    if indptr.shape[0] != ncol + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* y_p = <double*>y.data

    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t row
    cdef double acc = 0.0
    cdef double xj

    with nogil:
        for j in range(ncol):
            start = <Py_ssize_t>indptr_p[j]
            end = <Py_ssize_t>indptr_p[j + 1]
            if start == end:
                continue
            xj = x_p[j]
            if xj == 0.0:
                continue
            for k in range(start, end):
                row = <Py_ssize_t>indices_p[k]
                acc += data_p[k] * y_p[row] * xj
    return <double>acc


def csr_matmul_dense_inplace_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] = A @ x`` for a CSR matrix A given by (indptr, indices, data).

    Notes
    -----
    - A is interpreted as shape (nrow, ncol) with nrow inferred as ``indptr.size-1``.
    - ``x`` must have shape (ncol, nvec) and C-contiguous (last axis contiguous).
    - ``out`` must have shape (nrow, nvec) and C-contiguous (last axis contiguous).
    - Uses OpenMP parallelism when compiled with ``GUGA_USE_OPENMP=1`` and the compiler supports it.
    """

    cdef Py_ssize_t nrow = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]
    cdef Py_ssize_t ncol = <Py_ssize_t>x.shape[0]

    if out.shape[0] != nrow:
        raise ValueError("out has wrong shape for this CSR operator")
    if x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSR operator")
    if indptr.shape[0] != nrow + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t col
    cdef double coeff
    cdef Py_ssize_t ioff
    cdef Py_ssize_t coff
    cdef bint use_openmp = False

    # Avoid the overhead of entering an OpenMP parallel region when the runtime
    # is configured for a single thread (common when we parallelize at the
    # Python level with ThreadPoolExecutor).
    if guga_have_openmp() and guga_openmp_max_threads() > 1:
        use_openmp = True

    with nogil:
        if nvec == 1:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    out_p[i] = 0.0
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * x_p[col]
            else:
                for i in range(nrow):
                    out_p[i] = 0.0
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * x_p[col]
        else:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    ioff = i * nvec
                    for v in range(nvec):
                        out_p[ioff + v] = 0.0
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]
            else:
                for i in range(nrow):
                    ioff = i * nvec
                    for v in range(nvec):
                        out_p[ioff + v] = 0.0
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]


def csr_matmul_dense_add_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
):
    """Compute ``out[:] += A @ x`` for a CSR matrix A given by (indptr, indices, data)."""

    cdef Py_ssize_t nrow = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]
    cdef Py_ssize_t ncol = <Py_ssize_t>x.shape[0]

    if out.shape[0] != nrow:
        raise ValueError("out has wrong shape for this CSR operator")
    if x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSR operator")
    if indptr.shape[0] != nrow + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t col
    cdef double coeff
    cdef Py_ssize_t ioff
    cdef Py_ssize_t coff
    cdef bint use_openmp = False

    if guga_have_openmp() and guga_openmp_max_threads() > 1:
        use_openmp = True

    with nogil:
        if nvec == 1:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * x_p[col]
            else:
                for i in range(nrow):
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * x_p[col]
        else:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    ioff = i * nvec
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]
            else:
                for i in range(nrow):
                    ioff = i * nvec
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]


def csr_matmul_dense_add_scaled_cy(
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.float64_t, ndim=1] data,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] out,
    double alpha,
):
    """Compute ``out[:] += alpha * (A @ x)`` for a CSR matrix A given by (indptr, indices, data)."""

    cdef Py_ssize_t nrow = <Py_ssize_t>(indptr.shape[0] - 1)
    cdef Py_ssize_t nvec = <Py_ssize_t>out.shape[1]
    cdef Py_ssize_t ncol = <Py_ssize_t>x.shape[0]
    cdef double alpha_v = <double>alpha

    if out.shape[0] != nrow:
        raise ValueError("out has wrong shape for this CSR operator")
    if x.shape[1] != nvec:
        raise ValueError("x has wrong shape for this CSR operator")
    if indptr.shape[0] != nrow + 1:
        raise ValueError("indptr has wrong length")

    cdef cnp.int32_t* indptr_p = <cnp.int32_t*>indptr.data
    cdef cnp.int32_t* indices_p = <cnp.int32_t*>indices.data
    cdef double* data_p = <double*>data.data
    cdef double* x_p = <double*>x.data
    cdef double* out_p = <double*>out.data

    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t v
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t col
    cdef double coeff
    cdef Py_ssize_t ioff
    cdef Py_ssize_t coff
    cdef bint use_openmp = False

    if guga_have_openmp() and guga_openmp_max_threads() > 1:
        use_openmp = True

    with nogil:
        if nvec == 1:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * (alpha_v * x_p[col])
            else:
                for i in range(nrow):
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        out_p[i] += data_p[k] * (alpha_v * x_p[col])
        else:
            if use_openmp:
                for i in prange(nrow, schedule="static"):
                    ioff = i * nvec
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = alpha_v * data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]
            else:
                for i in range(nrow):
                    ioff = i * nvec
                    start = <Py_ssize_t>indptr_p[i]
                    end = <Py_ssize_t>indptr_p[i + 1]
                    for k in range(start, end):
                        col = <Py_ssize_t>indices_p[k]
                        coff = col * nvec
                        coeff = alpha_v * data_p[k]
                        for v in range(nvec):
                            out_p[ioff + v] += coeff * x_p[coff + v]


def dense_row_mask_add_many_cy(
    cnp.ndarray[cnp.float64_t, ndim=1] row,
    cnp.ndarray[cnp.npy_bool, ndim=1] mask,
    cnp.ndarray[cnp.int32_t, ndim=1] idx,
    cnp.ndarray[cnp.float64_t, ndim=1] val,
):
    """Add COO contributions into a dense row scratch (plus touched mask).

    This is a small helper for row-oracle assembly to avoid the overhead of
    `np.add.at` + `mask[idx]=True` in Python for large COO arrays.
    """

    cdef Py_ssize_t n = idx.shape[0]
    if val.shape[0] != n:
        raise ValueError("idx and val must have the same size")
    if n == 0:
        return None

    cdef double[::1] row_v = row
    cdef cnp.npy_bool[::1] mask_v = mask
    cdef cnp.int32_t[::1] idx_v = idx
    cdef double[::1] val_v = val

    cdef Py_ssize_t i
    cdef cnp.int32_t ii

    # No bounds checks: callers must guarantee 0 <= idx < row.size.
    with nogil:
        for i in range(n):
            ii = idx_v[i]
            row_v[ii] += val_v[i]
            mask_v[ii] = 1

    return None
