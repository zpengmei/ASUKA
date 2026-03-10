"""Tests for Kernel 2.5 width-aware bucketing and hybrid dispatch."""

import pytest
import numpy as np

pytestmark = pytest.mark.cuda

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from asuka.cuda import cuda_backend as cb

    _ext = cb._ext
    HAS_EXT = _ext is not None
except Exception:
    HAS_EXT = False

try:
    from asuka.cuguga.drt import DRT, build_drt

    HAS_GUGA = True
except ImportError:
    HAS_GUGA = False

skipif_no_cuda = pytest.mark.skipif(
    not (HAS_CUPY and HAS_EXT and HAS_GUGA), reason="CUDA extension, CuPy, or GUGA module not available"
)


def _make_csr_from_drt(drt, j_start=0, j_count=None, coalesce=True):
    """Helper: build CSR via Kernel25Workspace and return arrays + workspace."""
    ncsf = drt.ncsf
    norb = drt.norb
    if j_count is None:
        j_count = ncsf
    drt_dev = cb.make_device_drt(drt)
    state_dev = cb.make_device_state_cache(drt, drt_dev)

    n_pairs = norb * (norb - 1)
    max_tasks = j_count * n_pairs
    max_nnz = max(max_tasks * 8, 2048)
    ws = _ext.Kernel25Workspace(max_tasks, max_nnz)

    row_j = cp.empty((max_nnz,), dtype=cp.int32)
    row_k = cp.empty((max_nnz,), dtype=cp.int32)
    indptr = cp.empty((max_nnz + 1,), dtype=cp.int64)
    indices = cp.empty((max_nnz,), dtype=cp.int32)
    data = cp.empty((max_nnz,), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)

    profile = {}
    nrows, nnz, nnz_in = ws.build_from_jrs_allpairs_deterministic_inplace_device(
        drt_dev,
        state_dev,
        int(j_start),
        int(j_count),
        row_j,
        row_k,
        indptr,
        indices,
        data,
        overflow,
        128,
        coalesce,
        0,
        True,
        True,
        1,
        False,
        profile,
    )
    nrows = int(nrows)
    nnz = int(nnz)
    return (
        ws,
        drt_dev,
        state_dev,
        row_j[:nrows],
        row_k[:nrows],
        indptr[: nrows + 1],
        indices[:nnz],
        data[:nnz],
        nrows,
        nnz,
        profile,
    )


@skipif_no_cuda
def test_bucket_info_after_build():
    """Verify that get_bucket_info returns valid metadata after a CSR build."""
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    if drt.ncsf < 2:
        pytest.skip("ncsf too small")

    ws, drt_dev, state_dev, row_j_d, row_k_d, indptr_d, indices_d, data_d, nrows, nnz, profile = _make_csr_from_drt(drt)

    if nrows == 0:
        pytest.skip("no CSR rows produced")

    # Check bucket info.
    valid, offsets, counts, perm_ptr = ws.get_bucket_info()
    assert valid, "bucketing_valid should be True after packed CSR build"
    assert len(offsets) == 5
    assert len(counts) == 4

    # Offsets are cumulative sums of counts.
    total = 0
    for b in range(4):
        assert int(offsets[b]) == total
        total += int(counts[b])
    assert int(offsets[4]) == total
    assert total == nrows, f"sum of bucket counts ({total}) must equal nrows ({nrows})"

    # perm_ptr is a non-null device pointer.
    assert int(perm_ptr) != 0

    # Read the row_perm array back and validate.
    perm_d = cp.ndarray(
        shape=(nrows,),
        dtype=cp.int32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(int(perm_ptr), nrows * 4, None), 0
        ),
    )
    perm_h = perm_d.get()

    # row_perm should be a valid permutation of [0..nrows-1].
    assert set(perm_h.tolist()) == set(range(nrows)), "row_perm must be a permutation"

    # Verify bucket width constraints using CSR indptr.
    indptr_h = indptr_d.get()
    widths = np.diff(indptr_h).astype(np.int32)

    bucket_ranges = [(0, 1), (2, 4), (5, 16), (17, 10**9)]
    for b in range(4):
        start = int(offsets[b])
        end = int(offsets[b + 1])
        w_min, w_max = bucket_ranges[b]
        for idx in range(start, end):
            row = perm_h[idx]
            w = widths[row]
            assert w_min <= w <= w_max, (
                f"bucket {b}: row {row} has width {w}, expected [{w_min},{w_max}]"
            )

    # Profile should contain bucket_ms and bucket_counts.
    assert "bucket_ms" in profile
    assert "bucket_counts" in profile
    assert len(profile["bucket_counts"]) == 4


@skipif_no_cuda
def test_bucket_determinism():
    """Two builds with identical input produce identical row_perm."""
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    if drt.ncsf < 2:
        pytest.skip("ncsf too small")

    def run_build():
        ws, drt_dev, state_dev, *_, nrows, nnz, _ = _make_csr_from_drt(drt)
        valid, offsets, counts, perm_ptr = ws.get_bucket_info()
        if not valid or nrows == 0:
            return None, None
        perm_d = cp.ndarray(
            shape=(nrows,), dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(int(perm_ptr), nrows * 4, None), 0
            ),
        )
        return perm_d.get().copy(), tuple(int(x) for x in offsets)

    perm1, off1 = run_build()
    perm2, off2 = run_build()
    if perm1 is None:
        pytest.skip("no CSR rows produced")
    np.testing.assert_array_equal(perm1, perm2, err_msg="row_perm must be deterministic")
    assert off1 == off2, "bucket_offsets must be deterministic"


@skipif_no_cuda
def test_bucket0_w_aggregate_kernel():
    """Verify bucket0 W-aggregate kernel produces correct results for width-1 rows."""
    drt = build_drt(norb=4, nelec=2, twos_target=0)
    if drt.ncsf < 2:
        pytest.skip("ncsf too small")

    ws, drt_dev, state_dev, row_j_d, row_k_d, indptr_d, indices_d, data_d, nrows, nnz, _ = _make_csr_from_drt(
        drt, coalesce=False
    )
    if nrows == 0:
        pytest.skip("no CSR rows produced")

    valid, offsets, counts, perm_ptr = ws.get_bucket_info()
    if not valid:
        pytest.skip("bucketing not valid")
    n_b0 = int(offsets[1])
    if n_b0 == 0:
        pytest.skip("no width-1 rows in this DRT")

    ncsf = drt.ncsf
    nops = drt.norb * drt.norb

    rng = np.random.default_rng(123)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    # Reference: manually compute w[k,rs] += x[j] * c for width-1 rows.
    perm_d = cp.ndarray(
        shape=(nrows,), dtype=cp.int32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(int(perm_ptr), nrows * 4, None), 0
        ),
    )
    perm_h = perm_d.get()
    row_j_h = row_j_d.get()
    row_k_h = row_k_d.get()
    indptr_h = indptr_d.get()
    indices_h = indices_d.get()
    data_h = data_d.get()

    w_ref = np.zeros((ncsf, nops), dtype=np.float64)
    for idx in range(n_b0):
        row = perm_h[idx]
        j = row_j_h[row]
        k = row_k_h[row]
        start = int(indptr_h[row])
        rs = indices_h[start]
        c = data_h[start]
        w_ref[k, rs] += x_h[j] * c

    # GPU: bucket0 kernel.
    w_gpu = cp.zeros((ncsf, nops), dtype=cp.float64)
    overflow = cp.zeros((1,), dtype=cp.int32)

    _ext.kernel4_build_w_from_csr_bucket0_rawptr(
        int(perm_ptr),
        n_b0,
        row_j_d,
        row_k_d,
        indptr_d,
        indices_d,
        data_d,
        x_d,
        w_gpu,
        overflow,
        ncsf,
        nops,
        256,
        0,
        True,
        True,
    )

    w_gpu_h = w_gpu.get()
    max_err = np.max(np.abs(w_ref - w_gpu_h))
    assert max_err < 1e-12, f"Bucket0 W-aggregate max error: {max_err}"


@skipif_no_cuda
def test_narrow_kernel_synthetic():
    """Verify narrow kernel (MAX_WIDTH=4) matches general fused kernel on synthetic CSR with width 2-4."""
    drt = build_drt(norb=6, nelec=4, twos_target=0)
    if drt.ncsf < 2:
        pytest.skip("ncsf too small")

    ncsf = drt.ncsf
    norb = drt.norb
    nops = norb * norb
    rng = np.random.default_rng(42)

    # Build EPQ table.
    from asuka.cuda.cuda_backend import build_epq_action_table_combined_device
    drt_dev = cb.make_device_drt(drt)
    state_dev = cb.make_device_state_cache(drt, drt_dev)
    epq_table = build_epq_action_table_combined_device(drt, drt_dev, state_dev, indptr_dtype="int64")
    epq_indptr_d, epq_indices_d, epq_pq_d, epq_data_d = epq_table

    # Build a real CSR (all width 1), then manually construct wider rows
    # by combining width-1 rows that share the same (j,k) or by creating synthetic rows.
    ws, drt_dev, state_dev, row_j_d_orig, row_k_d_orig, indptr_d_orig, indices_d_orig, data_d_orig, nrows_orig, nnz_orig, _ = _make_csr_from_drt(drt, coalesce=True)
    if nrows_orig < 8:
        pytest.skip("not enough rows for synthetic test")

    # Create synthetic CSR with widths [2, 3, 4, 2] (4 rows).
    widths = [2, 3, 4, 2]
    n_synth = len(widths)
    total_nnz = sum(widths)

    row_j_h = row_j_d_orig[:n_synth].get()
    row_k_h = row_k_d_orig[:n_synth].get()

    indptr_h = np.zeros(n_synth + 1, dtype=np.int64)
    for i, w in enumerate(widths):
        indptr_h[i + 1] = indptr_h[i] + w

    # Generate random column indices (rs) and data.
    indices_h = np.zeros(total_nnz, dtype=np.int32)
    data_h = np.zeros(total_nnz, dtype=np.float64)
    for i, w in enumerate(widths):
        start = int(indptr_h[i])
        # Pick unique rs values.
        rs_vals = rng.choice(nops, size=w, replace=False).astype(np.int32)
        rs_vals.sort()
        indices_h[start:start + w] = rs_vals
        data_h[start:start + w] = rng.standard_normal(w)

    row_j_d = cp.asarray(row_j_h, dtype=cp.int32)
    row_k_d = cp.asarray(row_k_h, dtype=cp.int32)
    indptr_d = cp.asarray(indptr_h, dtype=cp.int64)
    indices_d = cp.asarray(indices_h, dtype=cp.int32)
    data_d = cp.asarray(data_h, dtype=cp.float64)
    row_perm_d = cp.arange(n_synth, dtype=cp.int32)

    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)
    eri_mat_h = rng.standard_normal((nops, nops))
    eri_mat_h = 0.5 * (eri_mat_h + eri_mat_h.T)
    eri_mat_t_d = cp.asarray(eri_mat_h, dtype=cp.float64)

    perm_ptr = int(row_perm_d.data.ptr)

    # Reference: general fused kernel.
    y_ref = cp.zeros(ncsf, dtype=cp.float64)
    overflow_ref = cp.zeros(1, dtype=cp.int32)
    _ext.kernel4_apply_csr_fused_perm_rawptr(
        perm_ptr, n_synth, state_dev,
        epq_indptr_d, epq_indices_d, epq_pq_d, epq_data_d,
        row_j_d, row_k_d, indptr_d, indices_d, data_d,
        eri_mat_t_d, x_d, y_ref, overflow_ref,
        nops, 32, 0, True, True,
    )

    # Narrow kernel.
    y_narrow = cp.zeros(ncsf, dtype=cp.float64)
    overflow_narrow = cp.zeros(1, dtype=cp.int32)
    _ext.kernel4_apply_csr_narrow_fused_perm_rawptr(
        perm_ptr, n_synth, state_dev,
        epq_indptr_d, epq_indices_d, epq_pq_d, epq_data_d,
        row_j_d, row_k_d, indptr_d, indices_d, data_d,
        eri_mat_t_d, x_d, y_narrow, overflow_narrow,
        nops, 32, 0, True, True,
    )

    y_ref_h = y_ref.get()
    y_narrow_h = y_narrow.get()
    max_err = np.max(np.abs(y_ref_h - y_narrow_h))
    assert max_err < 1e-12, f"Narrow kernel vs fused kernel max error: {max_err}"


@skipif_no_cuda
def test_bucket_info_not_valid_without_build():
    """Verify that a fresh workspace reports bucketing as invalid."""
    ws = _ext.Kernel25Workspace(100, 1000)
    valid, offsets, counts, perm_ptr = ws.get_bucket_info()
    assert not valid
    assert offsets is None
    assert counts is None
    assert int(perm_ptr) == 0
