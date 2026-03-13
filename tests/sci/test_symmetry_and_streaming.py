"""Tests for irrep screening (Part A) and streaming PT2 (Part B).

Part A tests:
  - Symmetry-filtered HB index construction
  - Energy parity with/without orbsym
  - C1 no-op

Part B tests:
  - Streaming PT2 matches exact
  - Multiroot streaming
  - Semistochastic convergence
  - CIPSI pt2_mode="streaming" integration
"""

from __future__ import annotations

import numpy as np
import pytest

from asuka.cuguga.drt import DRT, build_drt
from asuka.sci.hb_integrals import HeatBathIntegralIndex, build_hb_index


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _h2_integrals():
    """H2 minimal basis: 2 electrons, 2 orbitals."""
    norb = 2
    h1e = np.array([[-1.252, 0.0], [0.0, -0.475]], dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    eri[0, 0, 0, 0] = 0.674
    eri[1, 1, 1, 1] = 0.697
    eri[0, 0, 1, 1] = 0.181
    eri[1, 1, 0, 0] = 0.181
    eri[0, 1, 0, 1] = 0.15
    eri[1, 0, 1, 0] = 0.15
    return norb, h1e, eri


def _be_integrals():
    """Be atom: 2 electrons in 4 orbitals."""
    norb = 4
    rng = np.random.default_rng(42)
    h1e = rng.standard_normal((norb, norb)) * 0.5
    h1e = (h1e + h1e.T) / 2
    h1e[np.diag_indices(norb)] = np.array([-1.5, -0.8, -0.3, 0.1])
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    v = rng.standard_normal() * 0.2
                    eri[p, q, r, s] = v
                    eri[r, s, p, q] = v
                    eri[q, p, s, r] = v
                    eri[s, r, q, p] = v
    eri = (eri + eri.transpose(2, 3, 0, 1)) / 2
    eri = (eri + eri.transpose(1, 0, 3, 2)) / 2
    return norb, h1e, eri


def _c2v_h2o_like_integrals():
    """4-orbital C2v-like system with orbsym=[0, 0, 1, 1] (A1, A1, B1, B1)."""
    norb = 4
    orbsym = np.array([0, 0, 1, 1], dtype=np.int32)
    rng = np.random.default_rng(99)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    # Populate only symmetry-allowed integrals
    for p in range(norb):
        for q in range(norb):
            if (orbsym[p] ^ orbsym[q]) != 0:
                continue
            h1e[p, q] = rng.standard_normal() * 0.5
            for r in range(norb):
                for s in range(norb):
                    if (orbsym[r] ^ orbsym[s]) != 0:
                        continue
                    if (orbsym[p] ^ orbsym[q]) != (orbsym[r] ^ orbsym[s]):
                        continue
                    v = rng.standard_normal() * 0.2
                    eri[p, q, r, s] = v
                    eri[r, s, p, q] = v
                    eri[q, p, s, r] = v
                    eri[s, r, q, p] = v
    h1e = (h1e + h1e.T) / 2
    eri = (eri + eri.transpose(2, 3, 0, 1)) / 2
    eri = (eri + eri.transpose(1, 0, 3, 2)) / 2
    return norb, h1e, eri, orbsym


# ===========================================================================
# Part A: Symmetry filtering tests
# ===========================================================================


class TestHBIndexSymmetryFiltering:
    """Test that orbsym filtering works correctly in build_hb_index."""

    def test_hb_index_symmetry_filtering_reduces_entries(self):
        """C2v with orbsym=[0,0,1,1]: filtered index has fewer h1/CSR entries."""
        norb, h1e, eri, orbsym = _c2v_h2o_like_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps

        idx_unfilt = build_hb_index(h_eff, eri, norb)
        idx_filt = build_hb_index(h_eff, eri, norb, orbsym=orbsym)

        # Filtered should have <= entries
        assert idx_filt.n_h1 <= idx_unfilt.n_h1
        assert idx_filt.nnz_2e <= idx_unfilt.nnz_2e

        # Check no forbidden pairs in h1
        for k in range(idx_filt.n_h1):
            p, q = int(idx_filt.h1_pq[k, 0]), int(idx_filt.h1_pq[k, 1])
            assert (orbsym[p] ^ orbsym[q]) == 0, f"Forbidden h1 pair ({p}, {q})"

        # Check sym_pq_allowed is set
        assert idx_filt.sym_pq_allowed is not None
        assert idx_filt.sym_pq_allowed.shape == (norb * norb,)

    def test_hb_index_symmetry_filtering_no_forbidden_csr(self):
        """No symmetry-forbidden pairs in CSR after filtering."""
        norb, h1e, eri, orbsym = _c2v_h2o_like_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb, orbsym=orbsym)

        for pq in range(norb * norb):
            p, q = pq // norb, pq % norb
            lo = int(idx.pq_ptr[pq])
            hi = int(idx.pq_ptr[pq + 1])
            if lo >= hi:
                continue
            # If pq is forbidden, there should be no CSR entries
            if (orbsym[p] ^ orbsym[q]) != 0:
                assert lo == hi, f"Forbidden pq={pq} has CSR entries"

    def test_hb_symmetry_c1_noop(self):
        """All-zero orbsym gives identical index to no orbsym."""
        norb, h1e, eri = _be_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps

        idx_none = build_hb_index(h_eff, eri, norb)
        idx_c1 = build_hb_index(h_eff, eri, norb, orbsym=np.zeros(norb, dtype=np.int32))

        np.testing.assert_array_equal(idx_none.h1_pq, idx_c1.h1_pq)
        np.testing.assert_array_equal(idx_none.h1_abs, idx_c1.h1_abs)
        np.testing.assert_array_equal(idx_none.pq_ptr, idx_c1.pq_ptr)
        np.testing.assert_array_equal(idx_none.rs_idx, idx_c1.rs_idx)
        np.testing.assert_allclose(idx_none.v_abs, idx_c1.v_abs, atol=1e-15)

    def test_hb_symmetry_energy_parity(self):
        """Full CIPSI with and without orbsym on a C2v molecule gives same E."""
        norb, h1e, eri, orbsym = _c2v_h2o_like_integrals()
        nelec = 2

        # Without orbsym
        drt1 = build_drt(norb=norb, nelec=nelec, twos_target=0)
        # With orbsym (wfnsym=0 for A1 ground state)
        drt2 = build_drt(norb=norb, nelec=nelec, twos_target=0, orbsym=orbsym, wfnsym=0)

        from asuka.sci.gpu_cipsi import run_cipsi_trials

        res1 = run_cipsi_trials(
            drt1, h1e, eri, nroots=1,
            init_ncsf=drt1.ncsf, max_ncsf=drt1.ncsf,
            backend="cpu_sparse",
        )
        res2 = run_cipsi_trials(
            drt2, h1e, eri, nroots=1,
            init_ncsf=drt2.ncsf, max_ncsf=drt2.ncsf,
            backend="cpu_sparse",
        )
        # The symmetry-filtered DRT may have fewer CSFs, but FCI energy
        # in the A1 sector should match the lowest A1 eigenvalue from
        # the full space.
        np.testing.assert_allclose(res2.e_var, res1.e_var, atol=1e-10)


class TestDRTOrbsym:
    """Test that DRT correctly stores orbsym."""

    def test_drt_stores_orbsym(self):
        orbsym = np.array([0, 0, 1, 1], dtype=np.int32)
        drt = build_drt(norb=4, nelec=2, twos_target=0, orbsym=orbsym, wfnsym=0)
        assert drt.orbsym is not None
        np.testing.assert_array_equal(drt.orbsym, orbsym)

    def test_drt_orbsym_none_by_default(self):
        drt = build_drt(norb=2, nelec=2, twos_target=0)
        assert drt.orbsym is None

    def test_drt_orbsym_validation(self):
        with pytest.raises(ValueError, match="orbsym length"):
            DRT(
                norb=2, nelec=2, twos_target=0,
                node_k=np.array([0, 1], dtype=np.int16),
                node_ne=np.array([0, 0], dtype=np.int16),
                node_twos=np.array([0, 0], dtype=np.int16),
                node_sym=np.array([0, 0], dtype=np.int16),
                nwalks=np.array([1, 1], dtype=np.int64),
                child=np.full((2, 4), -1, dtype=np.int32),
                root=0, leaf=1, ncsf=1,
                orbsym=np.array([0, 0, 0], dtype=np.int32),  # Wrong length!
            )


# ===========================================================================
# Part B: Streaming PT2 tests
# ===========================================================================


def _run_exact_pt2(drt, h1e, eri, nroots=1):
    """Run CIPSI to get exact PT2 for comparison."""
    from asuka.sci.gpu_cipsi import run_cipsi_trials

    return run_cipsi_trials(
        drt, h1e, eri, nroots=nroots,
        init_ncsf=min(8, drt.ncsf),
        max_ncsf=min(8, drt.ncsf),
        grow_by=4,
        backend="cpu_sparse",
    )


class TestStreamingPT2:
    """Tests for streaming_pt2_deterministic."""

    def test_streaming_matches_exact_h2(self):
        """Streaming PT2 with bucket_size=1 matches exact for H2."""
        norb, h1e, eri = _h2_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        ref = _run_exact_pt2(drt, h1e, eri)

        from asuka.sci.sparse_support import DiagonalGuessLookup, ConnectedRowCache
        from asuka.sci.streaming_pt2 import streaming_pt2_deterministic

        sel = ref.sel_idx.tolist()
        sel_set = set(sel)
        c_sel = ref.ci_sel
        if c_sel.ndim == 1:
            c_sel = c_sel[:, None]

        hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)
        row_cache = ConnectedRowCache()

        res = streaming_pt2_deterministic(
            drt, h1e, eri,
            sel=sel, selected_set=sel_set,
            c_sel=c_sel, e_var=ref.e_var,
            hdiag_lookup=hdiag_lookup,
            row_cache=row_cache,
            bucket_size=1,
        )
        np.testing.assert_allclose(res.e_pt2, ref.e_pt2, atol=1e-14)
        assert res.n_batches > 0
        assert res.wall_time_s >= 0.0

    def test_streaming_matches_exact_be(self):
        """Streaming PT2 matches exact for Be with various bucket sizes."""
        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        ref = _run_exact_pt2(drt, h1e, eri)

        from asuka.sci.sparse_support import DiagonalGuessLookup, ConnectedRowCache
        from asuka.sci.streaming_pt2 import streaming_pt2_deterministic

        sel = ref.sel_idx.tolist()
        sel_set = set(sel)
        c_sel = ref.ci_sel
        if c_sel.ndim == 1:
            c_sel = c_sel[:, None]
        hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)

        for bs in [1, 3, 5, drt.ncsf]:
            res = streaming_pt2_deterministic(
                drt, h1e, eri,
                sel=sel, selected_set=sel_set,
                c_sel=c_sel, e_var=ref.e_var,
                hdiag_lookup=hdiag_lookup,
                bucket_size=bs,
            )
            np.testing.assert_allclose(
                res.e_pt2, ref.e_pt2, atol=1e-12,
                err_msg=f"bucket_size={bs} mismatch",
            )

    def test_streaming_multiroot(self):
        """Streaming PT2 works with 2 roots."""
        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)

        from asuka.sci.gpu_cipsi import run_cipsi_trials

        ref = run_cipsi_trials(
            drt, h1e, eri, nroots=2,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=min(4, drt.ncsf),
            backend="cpu_sparse",
        )

        from asuka.sci.sparse_support import DiagonalGuessLookup
        from asuka.sci.streaming_pt2 import streaming_pt2_deterministic

        sel = ref.sel_idx.tolist()
        sel_set = set(sel)
        c_sel = ref.ci_sel
        if c_sel.ndim == 1:
            c_sel = c_sel[:, None]
        hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)

        res = streaming_pt2_deterministic(
            drt, h1e, eri,
            sel=sel, selected_set=sel_set,
            c_sel=c_sel, e_var=ref.e_var,
            hdiag_lookup=hdiag_lookup,
            bucket_size=3,
        )
        assert res.e_pt2.shape == (2,)
        np.testing.assert_allclose(res.e_pt2, ref.e_pt2, atol=1e-12)

    def test_cipsi_pt2_mode_streaming(self):
        """Integration: run_cipsi_trials(pt2_mode='streaming') works."""
        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)

        from asuka.sci.gpu_cipsi import run_cipsi_trials

        res_exact = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=min(4, drt.ncsf),
            backend="cpu_sparse",
        )
        res_stream = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=min(4, drt.ncsf),
            backend="cpu_sparse",
            pt2_mode="streaming",
            pt2_bucket_size=3,
        )
        np.testing.assert_allclose(res_stream.e_pt2, res_exact.e_pt2, atol=1e-12)
        np.testing.assert_allclose(res_stream.e_var, res_exact.e_var, atol=1e-12)


class TestSemistochasticPT2:
    """Tests for semistochastic PT2."""

    def test_semistochastic_converges(self):
        """Increasing samples makes semistochastic approach exact."""
        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        ref = _run_exact_pt2(drt, h1e, eri)

        from asuka.sci.sparse_support import DiagonalGuessLookup
        from asuka.sci.streaming_pt2 import semistochastic_pt2

        sel = ref.sel_idx.tolist()
        sel_set = set(sel)
        c_sel = ref.ci_sel
        if c_sel.ndim == 1:
            c_sel = c_sel[:, None]
        hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)

        # With all sources deterministic, should be exact
        res = semistochastic_pt2(
            drt, h1e, eri,
            sel=sel, selected_set=sel_set,
            c_sel=c_sel, e_var=ref.e_var,
            hdiag_lookup=hdiag_lookup,
            n_det_sources=len(sel),  # all deterministic
            seed=42,
        )
        np.testing.assert_allclose(res.e_pt2, ref.e_pt2, atol=1e-12)

    def test_semistochastic_error_estimate(self):
        """Reported error is nonnegative and finite."""
        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        ref = _run_exact_pt2(drt, h1e, eri)

        from asuka.sci.sparse_support import DiagonalGuessLookup
        from asuka.sci.streaming_pt2 import semistochastic_pt2

        sel = ref.sel_idx.tolist()
        sel_set = set(sel)
        c_sel = ref.ci_sel
        if c_sel.ndim == 1:
            c_sel = c_sel[:, None]
        hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri)

        res = semistochastic_pt2(
            drt, h1e, eri,
            sel=sel, selected_set=sel_set,
            c_sel=c_sel, e_var=ref.e_var,
            hdiag_lookup=hdiag_lookup,
            n_det_sources=max(1, len(sel) // 2),
            n_stoch_samples=50,
            n_stoch_batches=5,
            seed=42,
        )
        assert np.all(res.e_pt2_error >= 0.0)
        assert np.all(np.isfinite(res.e_pt2_error))
        assert np.all(np.isfinite(res.e_pt2))
