from __future__ import annotations

import os
import tempfile
from typing import Final

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _csr_for_epq, _get_epq_action_cache

_STEP_TO_OCC_F64: Final[np.ndarray] = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D

try:  # optional Cython in-place CSC @ dense kernels
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
    )
except Exception:  # pragma: no cover
    _csc_matmul_dense_inplace_cy = None


def _fill_t_row(
    drt: DRT,
    cache,
    c: np.ndarray,
    *,
    p: int,
    q: int,
    out: np.ndarray,
) -> None:
    """Fill `out[:] = (E_pq |c>)` using the cached CSC-like representation."""

    p = int(p)
    q = int(q)
    if p == q:
        # Diagonal generator: E_pp |c> = occ_p * c
        out[:] = _STEP_TO_OCC_F64[cache.steps[:, p]] * c
        return

    csr = _csr_for_epq(cache, drt, p, q)
    if _csc_matmul_dense_inplace_cy is not None:
        c_col = c.reshape(int(c.size), 1)
        out_col = out.reshape(int(out.size), 1)
        _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
            csr.indptr, csr.indices, csr.data, c_col, out_col
        )
        return

    out.fill(0.0)
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    for j in range(int(c.size)):
        cj = float(c[j])
        if cj == 0.0:
            continue
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        out[indices[start:end]] += data[start:end] * cj


def make_rdm12_streaming(
    drt: DRT,
    civec: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
    block_nops: int = 8,
    tmpdir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (dm1, dm2) without allocating an in-RAM `t_pq` matrix.

    This routine is exact but may use an on-disk `np.memmap` for temporary storage
    when `t_pq` does not fit in the requested memory budget.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nops = norb * norb

    c = np.asarray(civec, dtype=np.float64).ravel()
    if c.size != ncsf:
        raise ValueError("civec has wrong length")

    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    max_memory_bytes = float(max_memory_mb) * 1e6
    if max_memory_bytes <= 0:
        max_memory_bytes = 1.0

    # We store T[pq, :] = (E_pq |c>). This is potentially huge:
    # bytes = (norb^2) * ncsf * 8.
    t_bytes = float(nops) * float(ncsf) * 8.0

    cache = _get_epq_action_cache(drt)

    dm1 = np.zeros((norb, norb), dtype=np.float64)

    # Optional fast path: use cached SciPy sparse matrices for E_pq|c> when available.
    try:
        from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
        from asuka.contract import _sp as _contract_sp  # noqa: PLC0415
    except Exception:  # pragma: no cover
        _contract_sp = None
        _epq_spmat_list = None  # type: ignore[assignment]

    mats = None
    occ = None
    if _contract_sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)
        occ = _STEP_TO_OCC_F64[cache.steps]

    # Choose in-RAM vs memmap storage.
    # Heuristic: if the full T fits comfortably in max_memory, keep it in memory;
    # otherwise use a disk-backed memmap to bound RSS.
    use_memmap = t_bytes > 0.7 * max_memory_bytes

    if not use_memmap:
        if mats is not None and occ is not None:
            T = np.empty((nops, ncsf), dtype=np.float64)
            c_col = c.reshape(ncsf, 1)
            for p in range(norb):
                for q in range(norb):
                    pq = p * norb + q
                    out = T[pq]
                    if p == q:
                        np.multiply(occ[:, p], c, out=out)
                    else:
                        mat = mats[pq]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix")
                        if _csc_matmul_dense_inplace_cy is not None:
                            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                mat.indptr, mat.indices, mat.data, c_col, out.reshape(ncsf, 1)
                            )
                        else:
                            out[:] = mat.dot(c)  # type: ignore[operator]
            dm1[:] = (T @ c).reshape(norb, norb).T
        else:
            T = np.empty((nops, ncsf), dtype=np.float64)
            for p in range(norb):
                for q in range(norb):
                    pq = p * norb + q
                    out = T[pq]
                    _fill_t_row(drt, cache, c, p=p, q=q, out=out)
            dm1[:] = (T @ c).reshape(norb, norb).T

        gram0 = T @ T.T
    else:
        with tempfile.TemporaryDirectory(dir=tmpdir) as tdir:
            path = os.path.join(tdir, "t_pq.f64")
            Tm = np.memmap(path, dtype=np.float64, mode="w+", shape=(nops, ncsf))
            c_col = c.reshape(ncsf, 1)
            for p in range(norb):
                for q in range(norb):
                    pq = p * norb + q
                    out = Tm[pq]
                    if mats is not None and occ is not None:
                        if p == q:
                            np.multiply(occ[:, p], c, out=out)
                        else:
                            mat = mats[pq]
                            if mat is None:
                                raise AssertionError("missing E_pq sparse matrix")
                            if _csc_matmul_dense_inplace_cy is not None:
                                _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                    mat.indptr, mat.indices, mat.data, c_col, out.reshape(ncsf, 1)
                                )
                            else:
                                out[:] = mat.dot(c)  # type: ignore[operator]
                    else:
                        _fill_t_row(drt, cache, c, p=p, q=q, out=out)
            Tm.flush()
            dm1[:] = (Tm @ c).reshape(norb, norb).T

            gram0 = np.empty((nops, nops), dtype=np.float64)
            for i0 in range(0, nops, block_nops):
                i1 = min(nops, i0 + block_nops)
                A = Tm[i0:i1]
                for j0 in range(0, i0 + 1, block_nops):
                    j1 = min(nops, j0 + block_nops)
                    B = Tm[j0:j1]
                    blk = A @ B.T
                    gram0[i0:i1, j0:j1] = blk
                    if i0 != j0:
                        gram0[j0:j1, i0:i1] = blk.T

    # Adjoint convention:
    # <E_pq E_rs> = dot(t_qp, t_rs) == (T @ T.T)[qp, rs].
    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]

    dm2 = gram.reshape(norb, norb, norb, norb)
    for p in range(norb):
        for q in range(norb):
            dm2[p, q, q, :] -= dm1[:, p]
    return dm1, dm2


def trans_rdm12_streaming(
    drt: DRT,
    ci_bra: np.ndarray,
    ci_ket: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
    block_nops: int = 8,
    tmpdir: str | None = None,
    force_memmap: bool | None = None,
    reorder: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition (dm1, dm2) between two CI vectors without allocating a full in-RAM `t_pq`.

    Convention matches `make_rdm12_streaming`:
    - dm1[p,q] = <bra| E_{q p} |ket>
    - dm2 is built from <bra|E_pq E_rs|ket>.

    By default (`reorder=True`), apply the standard δ-term contraction used in
    `make_rdm12_streaming` / PySCF-style spin-free Hamiltonian algebra:

      <E_pq E_rs>  ->  <E_pq E_rs - δ_{qr} E_ps>

    Setting `reorder=False` returns the raw (unreordered) <E_pq E_rs> tensor.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nops = norb * norb

    cbra = np.asarray(ci_bra, dtype=np.float64).ravel()
    cket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if cbra.size != ncsf or cket.size != ncsf:
        raise ValueError("ci_bra/ci_ket have wrong length")

    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    max_memory_bytes = float(max_memory_mb) * 1e6
    if max_memory_bytes <= 0:
        max_memory_bytes = 1.0

    # We store T_ket[pq, :] = (E_pq |ket>). This is potentially huge:
    # bytes = (norb^2) * ncsf * 8.
    t_bytes = float(nops) * float(ncsf) * 8.0

    cache = _get_epq_action_cache(drt)

    dm1 = np.zeros((norb, norb), dtype=np.float64)

    # Optional fast path: use cached SciPy sparse matrices for E_pq|c> when available.
    try:
        from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
        from asuka.contract import _sp as _contract_sp  # noqa: PLC0415
    except Exception:  # pragma: no cover
        _contract_sp = None
        _epq_spmat_list = None  # type: ignore[assignment]

    mats = None
    occ = None
    if _contract_sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)
        occ = _STEP_TO_OCC_F64[cache.steps]

    if force_memmap is None:
        use_memmap = t_bytes > 0.7 * max_memory_bytes
    else:
        use_memmap = bool(force_memmap)

    # Workspace for building E_pq|bra> blocks.
    A = np.empty((block_nops, ncsf), dtype=np.float64)

    # <bra|E_pq E_rs|ket> will be accumulated as gram_qp[qp, rs] = dot(E_qp|bra>, E_rs|ket>),
    # then reordered to pq by swapping the first index.
    gram_qp = np.empty((nops, nops), dtype=np.float64)

    if not use_memmap:
        T = np.empty((nops, ncsf), dtype=np.float64)
        cket_col = cket.reshape(ncsf, 1)
        cbra_col = cbra.reshape(ncsf, 1)
        for p in range(norb):
            for q in range(norb):
                pq = p * norb + q
                out = T[pq]
                if mats is not None and occ is not None:
                    if p == q:
                        np.multiply(occ[:, p], cket, out=out)
                    else:
                        mat = mats[pq]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix")
                        if _csc_matmul_dense_inplace_cy is not None:
                            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                mat.indptr, mat.indices, mat.data, cket_col, out.reshape(ncsf, 1)
                            )
                        else:
                            out[:] = mat.dot(cket)  # type: ignore[operator]
                else:
                    _fill_t_row(drt, cache, cket, p=p, q=q, out=out)
        dm1[:] = (T @ cbra).reshape(norb, norb).T

        # Build gram in blocks: A = (E_qp|bra>) rows, B = (E_rs|ket>) rows.
        for i0 in range(0, nops, block_nops):
            i1 = min(nops, i0 + block_nops)
            for qp in range(i0, i1):
                q = qp // norb
                p = qp - q * norb
                if mats is not None and occ is not None:
                    if q == p:
                        np.multiply(occ[:, q], cbra, out=A[qp - i0])
                    else:
                        mat = mats[q * norb + p]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix")
                        if _csc_matmul_dense_inplace_cy is not None:
                            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                mat.indptr,
                                mat.indices,
                                mat.data,
                                cbra_col,
                                A[qp - i0].reshape(ncsf, 1),
                            )
                        else:
                            A[qp - i0][:] = mat.dot(cbra)  # type: ignore[operator]
                else:
                    _fill_t_row(drt, cache, cbra, p=q, q=p, out=A[qp - i0])
            Ablk = A[: i1 - i0]
            for j0 in range(0, nops, block_nops):
                j1 = min(nops, j0 + block_nops)
                B = T[j0:j1]
                gram_qp[i0:i1, j0:j1] = Ablk @ B.T
    else:
        with tempfile.TemporaryDirectory(dir=tmpdir) as tdir:
            path = os.path.join(tdir, "t_pq_ket.f64")
            Tm = np.memmap(path, dtype=np.float64, mode="w+", shape=(nops, ncsf))
            cket_col = cket.reshape(ncsf, 1)
            cbra_col = cbra.reshape(ncsf, 1)
            for p in range(norb):
                for q in range(norb):
                    pq = p * norb + q
                    out = Tm[pq]
                    if mats is not None and occ is not None:
                        if p == q:
                            np.multiply(occ[:, p], cket, out=out)
                        else:
                            mat = mats[pq]
                            if mat is None:
                                raise AssertionError("missing E_pq sparse matrix")
                            if _csc_matmul_dense_inplace_cy is not None:
                                _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                    mat.indptr, mat.indices, mat.data, cket_col, out.reshape(ncsf, 1)
                                )
                            else:
                                out[:] = mat.dot(cket)  # type: ignore[operator]
                    else:
                        _fill_t_row(drt, cache, cket, p=p, q=q, out=out)
            Tm.flush()
            dm1[:] = (Tm @ cbra).reshape(norb, norb).T

            for i0 in range(0, nops, block_nops):
                i1 = min(nops, i0 + block_nops)
                for qp in range(i0, i1):
                    q = qp // norb
                    p = qp - q * norb
                    if mats is not None and occ is not None:
                        if q == p:
                            np.multiply(occ[:, q], cbra, out=A[qp - i0])
                        else:
                            mat = mats[q * norb + p]
                            if mat is None:
                                raise AssertionError("missing E_pq sparse matrix")
                            if _csc_matmul_dense_inplace_cy is not None:
                                _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                    mat.indptr,
                                    mat.indices,
                                    mat.data,
                                    cbra_col,
                                    A[qp - i0].reshape(ncsf, 1),
                                )
                            else:
                                A[qp - i0][:] = mat.dot(cbra)  # type: ignore[operator]
                    else:
                        _fill_t_row(drt, cache, cbra, p=q, q=p, out=A[qp - i0])
                Ablk = A[: i1 - i0]
                for j0 in range(0, nops, block_nops):
                    j1 = min(nops, j0 + block_nops)
                    B = Tm[j0:j1]
                    gram_qp[i0:i1, j0:j1] = Ablk @ B.T

    # Map qp->pq on the first index: gram[pq,rs] = gram_qp[qp,rs]
    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram_qp[swap]

    dm2 = gram.reshape(norb, norb, norb, norb)
    if bool(reorder):
        for p in range(norb):
            for q in range(norb):
                dm2[p, q, q, :] -= dm1[:, p]
    return dm1, dm2


def trans_rdm1_streaming(
    drt: DRT,
    ci_bra: np.ndarray,
    ci_ket: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,  # kept for API symmetry; not used by dm1-only path
) -> np.ndarray:
    """Compute the transition 1-RDM between two CI vectors without building dm2.

    Convention matches `trans_rdm12_streaming`:
    - dm1[p,q] = <bra| E_{q p} |ket>

    Notes
    -----
    This is the dm1-only analogue of `trans_rdm12_streaming` and is intended for
    multistate workflows (e.g., XMS rotation) where only dm1 is required.
    """

    _ = float(max_memory_mb)  # API placeholder for future workspace heuristics

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)

    cbra = np.asarray(ci_bra, dtype=np.float64).ravel()
    cket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if cbra.size != ncsf or cket.size != ncsf:
        raise ValueError("ci_bra/ci_ket have wrong length")

    cache = _get_epq_action_cache(drt)

    dm1 = np.empty((norb, norb), dtype=np.float64)
    out = np.empty(ncsf, dtype=np.float64)

    # Optional fast path: use cached SciPy sparse matrices for E_pq|c> when available.
    try:
        from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
        from asuka.contract import _sp as _contract_sp  # noqa: PLC0415
    except Exception:  # pragma: no cover
        _contract_sp = None
        _epq_spmat_list = None  # type: ignore[assignment]

    mats = None
    occ = None
    if _contract_sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)
        occ = _STEP_TO_OCC_F64[cache.steps]

    cket_col = cket.reshape(ncsf, 1)
    cbra_col = cbra.reshape(ncsf, 1)
    out_col = out.reshape(ncsf, 1)

    for q in range(norb):
        for p in range(norb):
            # dm1[p,q] = <bra|E_{q p}|ket>
            if mats is not None and occ is not None:
                if q == p:
                    np.multiply(occ[:, q], cket, out=out)
                else:
                    mat = mats[q * norb + p]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if _csc_matmul_dense_inplace_cy is not None:
                        _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                            mat.indptr, mat.indices, mat.data, cket_col, out_col
                        )
                    else:
                        out[:] = mat.dot(cket)  # type: ignore[operator]
            else:
                _fill_t_row(drt, cache, cket, p=q, q=p, out=out)
            dm1[p, q] = float(np.dot(cbra, out))

    return dm1


def make_rdm1_streaming(
    drt: DRT,
    civec: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
) -> np.ndarray:
    """Compute the state 1-RDM (dm1 only) in the same convention as `make_rdm12_streaming`.

    Convention:
    - dm1[p,q] = <c| E_{q p} |c>
    """

    return trans_rdm1_streaming(drt, civec, civec, max_memory_mb=float(max_memory_mb))


def trans_rdm1_all_streaming(
    drt: DRT,
    ci_list_bra: list[np.ndarray],
    ci_list_ket: list[np.ndarray] | None = None,
    *,
    block_nops: int = 8,
) -> np.ndarray:
    """Compute a full transition dm1 tensor for many bras/kets with one E_pq sweep per ket.

    Convention matches `trans_rdm1_streaming`:
    - dm1[bra,ket,p,q] = <bra| E_{q p} |ket>

    Parameters
    ----------
    ci_list_bra:
        List of CI vectors used as bras.
    ci_list_ket:
        List of CI vectors used as kets. If None, uses `ci_list_bra`.
    block_nops:
        Block size for the `E_pq|ket>` row workspace (controls memory).

    Returns
    -------
    dm1:
        Array of shape (nbra, nket, norb, norb) in float64.

    Notes
    -----
    This is the recommended building block for XMS workflows: it avoids the
    `O(nbra*nket*norb^2)` repeated application cost of calling
    `trans_rdm1_streaming` in a double loop by applying all generators once per
    ket and contracting against all bras in a single GEMM per block.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nops = norb * norb

    if ci_list_ket is None:
        ci_list_ket = ci_list_bra

    if len(ci_list_bra) == 0 or len(ci_list_ket) == 0:
        raise ValueError("ci_list_bra/ci_list_ket must be non-empty")

    cbra = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in ci_list_bra], axis=1)
    cket_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list_ket]
    if cbra.shape[0] != ncsf:
        raise ValueError("ci_list_bra vectors have wrong length")
    for c in cket_list:
        if int(c.size) != ncsf:
            raise ValueError("ci_list_ket vectors have wrong length")

    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    cache = _get_epq_action_cache(drt)

    # Optional fast path: use cached SciPy sparse matrices for E_pq|c> when available.
    try:
        from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
        from asuka.contract import _sp as _contract_sp  # noqa: PLC0415
    except Exception:  # pragma: no cover
        _contract_sp = None
        _epq_spmat_list = None  # type: ignore[assignment]

    mats = None
    occ = None
    if _contract_sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)
        occ = _STEP_TO_OCC_F64[cache.steps]

    nbra = int(cbra.shape[1])
    nket = int(len(cket_list))
    if mats is not None and occ is not None and nket > 1:
        # Multi-ket fast path: apply each E_pq to all ket vectors at once, then contract vs all bras.
        # This reduces the number of sparse matvecs by ~nket and uses the Cython CSC @ dense kernel
        # when available.
        out_bytes = int(ncsf) * int(nket) * np.dtype(np.float64).itemsize
        if out_bytes <= 256 * 1024**2:
            cket = np.stack(cket_list, axis=1)  # (ncsf, nket)
            out = np.empty((ncsf, nket), dtype=np.float64)
            v = np.empty((nops, nket, nbra), dtype=np.float64)
            for pq in range(nops):
                p = pq // norb
                q = pq - p * norb
                if p == q:
                    out[:] = occ[:, p][:, None] * cket
                else:
                    mat = mats[pq]
                    if mat is None:
                        raise AssertionError("missing E_pq sparse matrix")
                    if _csc_matmul_dense_inplace_cy is not None:
                        _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                            mat.indptr, mat.indices, mat.data, cket, out
                        )
                    else:
                        out[:] = mat.dot(cket)  # type: ignore[operator]
                v[pq] = out.T @ cbra

            swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()  # pq -> qp
            v = v[swap]
            return v.transpose(2, 1, 0).reshape(nbra, nket, norb, norb)

    cket_cols = [c.reshape(ncsf, 1) for c in cket_list]
    out = np.empty(ncsf, dtype=np.float64)
    out_col = out.reshape(ncsf, 1)

    # Workspace A holds a block of rows of T[pq,:] = E_pq|ket>.
    A = np.empty((block_nops, ncsf), dtype=np.float64)

    # dm1[bra,ket,p,q] = <bra|E_{q p}|ket> = (T @ bra)[qp].
    dm1 = np.empty((cbra.shape[1], len(cket_list), norb, norb), dtype=np.float64)
    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()  # pq -> qp

    for k, cket in enumerate(cket_list):
        # Build V[pq,bra] = <bra|E_pq|ket> in blocks.
        V = np.empty((nops, cbra.shape[1]), dtype=np.float64)
        for i0 in range(0, nops, block_nops):
            i1 = min(nops, i0 + block_nops)
            nb = int(i1 - i0)

            for t, pq in enumerate(range(i0, i1)):
                p = pq // norb
                q = pq - p * norb
                if mats is not None and occ is not None:
                    if p == q:
                        np.multiply(occ[:, p], cket, out=A[t])
                    else:
                        mat = mats[pq]
                        if mat is None:
                            raise AssertionError("missing E_pq sparse matrix")
                        if _csc_matmul_dense_inplace_cy is not None:
                            _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                                mat.indptr, mat.indices, mat.data, cket_cols[k], A[t].reshape(ncsf, 1)
                            )
                        else:
                            A[t][:] = mat.dot(cket)  # type: ignore[operator]
                else:
                    _fill_t_row(drt, cache, cket, p=p, q=q, out=A[t])

            V[i0:i1] = A[:nb] @ cbra

        # Apply the qp swap and reshape to (bra, p, q).
        dm1[:, k] = V[swap].T.reshape(cbra.shape[1], norb, norb)

    return dm1
