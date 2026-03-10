"""Tests for EPQ blocked aggregate optimizations (symmetric-pair GEMM)."""

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
    not (HAS_CUPY and HAS_EXT and HAS_GUGA),
    reason="CUDA extension, CuPy, or GUGA module not available",
)


def _make_symmetric_eri(norb, rng):
    """Build a random ERI matrix with full 8-fold symmetry: (rs|pq)=(sr|pq)=(rs|qp)=(pq|rs)."""
    nops = norb * norb
    # Build from DF-like factorization to ensure all symmetries.
    naux = max(norb * 3, 20)
    # L[pq, Q] must satisfy L[pq,Q] = L[qp,Q] for pair permutation symmetry.
    l_raw = rng.standard_normal((norb, norb, naux))
    l_sym = 0.5 * (l_raw + l_raw.transpose(1, 0, 2))  # symmetrize p<->q
    l_full = l_sym.reshape(nops, naux)
    eri = l_full @ l_full.T  # This gives full 8-fold symmetry.
    return eri, l_full


def _build_workspace_with_eri(norb, nelec, twos_target=0, use_df=False):
    """Build a GugaMatvecEriMatWorkspace with random ERI or DF L_full."""
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos_target)
    ncsf = drt.ncsf
    nops = norb * norb
    rng = np.random.default_rng(42)

    # Random h_eff.
    h_eff_h = rng.standard_normal((norb, norb))
    h_eff_h = 0.5 * (h_eff_h + h_eff_h.T)

    eri_h, l_full_h = _make_symmetric_eri(norb, rng)

    if use_df:
        ws = cb.GugaMatvecEriMatWorkspace(
            drt,
            l_full=cp.asarray(l_full_h, dtype=cp.float64),
            h_eff=cp.asarray(h_eff_h, dtype=cp.float64),
            aggregate_offdiag_k=True,
            use_epq_table=True,
            epq_build_device=True,
        )
    else:
        ws = cb.GugaMatvecEriMatWorkspace(
            drt,
            eri_mat=cp.asarray(eri_h, dtype=cp.float64),
            h_eff=cp.asarray(h_eff_h, dtype=cp.float64),
            aggregate_offdiag_k=True,
            use_epq_table=True,
            epq_build_device=True,
        )

    return ws, drt, ncsf


@skipif_no_cuda
def test_sym_pair_maps_initialized():
    """Verify symmetric-pair index maps are built when aggregate_offdiag_k=True."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=6, nelec=4)
    norb = int(drt.norb)
    npair = norb * (norb + 1) // 2

    if int(getattr(ws, "_sym_pair_npair", 0)) <= 0:
        pytest.skip("Symmetric-pair compression path is disabled in this CUDA backend build")

    assert ws._sym_pair_npair == npair, f"Expected npair={npair}, got {ws._sym_pair_npair}"
    assert ws._sym_pair_pair_pq is not None
    assert ws._sym_pair_pair_qp is not None
    assert ws._sym_pair_full_to_pair is not None

    pair_pq = ws._sym_pair_pair_pq.get()
    pair_qp = ws._sym_pair_pair_qp.get()
    full_to_pair = ws._sym_pair_full_to_pair.get()

    assert pair_pq.shape == (npair,)
    assert pair_qp.shape == (npair,)
    assert full_to_pair.shape == (norb * norb,)

    # Verify that full_to_pair maps both (p,q) and (q,p) to the same pair index.
    for u in range(npair):
        pq = pair_pq[u]
        qp = pair_qp[u]
        assert full_to_pair[pq] == u
        assert full_to_pair[qp] == u


@skipif_no_cuda
def test_sym_pair_eri_compression():
    """Verify compressed ERI pair matrix has correct symmetry."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=6, nelec=4)
    norb = int(drt.norb)
    nops = norb * norb
    npair = ws._sym_pair_npair

    if int(npair) <= 0:
        pytest.skip("Symmetric-pair compression path is disabled in this CUDA backend build")

    # Build compressed ERI.
    eri_pair = ws._sym_pair_get_eri_pair(ws.eri_mat)
    if eri_pair is None:
        pytest.skip("Symmetric-pair ERI compression helper unavailable in this CUDA backend build")
    assert eri_pair.shape == (npair, npair)

    # Verify: ERI_pair should be symmetric.
    eri_pair_h = eri_pair.get()
    assert np.allclose(eri_pair_h, eri_pair_h.T, atol=1e-14)


@skipif_no_cuda
def test_sym_pair_gemm_dense_matches_full():
    """Verify that symmetric-pair GEMM produces same result as full GEMM."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=6, nelec=4)
    if ncsf < 2:
        pytest.skip("ncsf too small")

    rng = np.random.default_rng(123)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    # Run hop with sym_pair (default when aggregate_offdiag_k=True and has_sym_pair_pack).
    profile_sym = {}
    y_sym = ws.hop(x_d, profile=profile_sym)
    y_sym_h = y_sym.get().copy()

    # Now disable sym_pair and run again.
    # Save and restore pair maps.
    saved_npair = ws._sym_pair_npair
    ws._sym_pair_npair = 0
    profile_full = {}
    y_full = ws.hop(x_d, profile=profile_full)
    y_full_h = y_full.get().copy()
    ws._sym_pair_npair = saved_npair

    max_err = np.max(np.abs(y_sym_h - y_full_h))
    assert max_err < 1e-10, f"Sym-pair vs full GEMM max error: {max_err}"

    # Check that sym_pair was actually used.
    if "sym_pair_active" in profile_sym:
        assert profile_sym["sym_pair_active"] == 1.0


@skipif_no_cuda
def test_sym_pair_gemm_df_matches_full():
    """Verify that symmetric-pair DF GEMM produces same result as full DF GEMM."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=6, nelec=4, use_df=True)
    if ncsf < 2:
        pytest.skip("ncsf too small")

    rng = np.random.default_rng(456)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    # Run hop with sym_pair.
    profile_sym = {}
    y_sym = ws.hop(x_d, profile=profile_sym)
    y_sym_h = y_sym.get().copy()

    # Disable sym_pair and run again.
    saved_npair = ws._sym_pair_npair
    ws._sym_pair_npair = 0
    profile_full = {}
    y_full = ws.hop(x_d, profile=profile_full)
    y_full_h = y_full.get().copy()
    ws._sym_pair_npair = saved_npair

    max_err = np.max(np.abs(y_sym_h - y_full_h))
    assert max_err < 1e-10, f"Sym-pair DF vs full DF GEMM max error: {max_err}"


@skipif_no_cuda
def test_epq_transpose_sorted_by_source():
    """Verify EPQ transpose entries are sorted by source within each row."""
    drt = build_drt(norb=6, nelec=4, twos_target=0)
    if drt.ncsf < 2:
        pytest.skip("ncsf too small")

    ws, _, ncsf = _build_workspace_with_eri(norb=6, nelec=4)

    # Force transpose build.
    epq_table = ws._epq_table
    assert epq_table is not None

    from asuka.cuda.cuda_backend import build_epq_action_table_transpose_device
    epq_table_t = build_epq_action_table_transpose_device(
        drt, epq_table, dtype=cp.float64, use_cache=False,
    )
    t_indptr, t_source, t_pq, t_data = epq_table_t

    t_indptr_h = t_indptr.get()
    t_source_h = t_source.get()

    # Check that within each row, t_source is non-decreasing.
    for i in range(ncsf):
        s = int(t_indptr_h[i])
        e = int(t_indptr_h[i + 1])
        if e - s <= 1:
            continue
        row_sources = t_source_h[s:e]
        assert np.all(row_sources[:-1] <= row_sources[1:]), (
            f"Row {i}: t_source not sorted: {row_sources[:10]}"
        )


@skipif_no_cuda
def test_sym_pair_gemm_determinism():
    """Two hops with sym_pair produce identical results."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=6, nelec=4)
    if ncsf < 2:
        pytest.skip("ncsf too small")

    rng = np.random.default_rng(789)
    x_d = cp.asarray(rng.standard_normal(ncsf), dtype=cp.float64)

    y1 = ws.hop(x_d).get().copy()
    y2 = ws.hop(x_d).get().copy()

    max_diff = np.max(np.abs(y1 - y2))
    assert max_diff < 1e-12, f"sym-pair hop not reproducible: max diff = {max_diff}"


@skipif_no_cuda
def test_pair_fused_kernels_match_full():
    """Verify pair-fused W-build/Apply kernels match full-nops path (norb>=10)."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=10, nelec=8)
    if ncsf < 2:
        pytest.skip("ncsf too small")

    from asuka.cuda.cuda_backend import has_sym_pair_fused_kernels
    if not has_sym_pair_fused_kernels():
        pytest.skip("pair-fused kernels not available")

    rng = np.random.default_rng(999)
    x_d = cp.asarray(rng.standard_normal(ncsf), dtype=cp.float64)

    # Run with pair-fused (active when norb >= 10 and fused kernels exist).
    assert ws._sym_pair_npair > 0, "sym_pair should be active for norb=10"
    y_fused = ws.hop(x_d).get().copy()

    # Disable sym_pair entirely and run full-nops path.
    saved_npair = ws._sym_pair_npair
    ws._sym_pair_npair = 0
    y_full = ws.hop(x_d).get().copy()
    ws._sym_pair_npair = saved_npair

    max_err = np.max(np.abs(y_fused - y_full))
    assert max_err < 1e-10, f"Pair-fused vs full max error: {max_err}"


@skipif_no_cuda
def test_pair_fused_determinism():
    """Two hops with pair-fused kernels produce identical results (norb>=10)."""
    ws, drt, ncsf = _build_workspace_with_eri(norb=10, nelec=8)
    if ncsf < 2:
        pytest.skip("ncsf too small")

    from asuka.cuda.cuda_backend import has_sym_pair_fused_kernels
    if not has_sym_pair_fused_kernels():
        pytest.skip("pair-fused kernels not available")

    assert ws._sym_pair_npair > 0
    rng = np.random.default_rng(1234)
    x_d = cp.asarray(rng.standard_normal(ncsf), dtype=cp.float64)

    y1 = ws.hop(x_d).get().copy()
    y2 = ws.hop(x_d).get().copy()

    max_diff = np.max(np.abs(y1 - y2))
    assert max_diff < 1e-12, f"pair-fused hop not reproducible: max diff = {max_diff}"
