"""Tests for Heat-Bath Selected Configuration Interaction (HB-SCI).

Tests cover:
1. Integral index construction (sorting, CSR structure)
2. Correctness: HB-SCI with eps→0 matches frontier-hash CIPSI
3. PT2 correction accuracy
4. Energy convergence with decreasing epsilon
5. Adaptive epsilon schedule
"""

from __future__ import annotations

import numpy as np
import pytest

from asuka.cuguga.drt import build_drt
from asuka.sci.hb_integrals import HeatBathIntegralIndex, build_g_base, build_hb_index, build_hb_index_from_df
from asuka.sci.hb_selection import adaptive_epsilon, heat_bath_select_and_pt2, semistochastic_pt2


# ---------------------------------------------------------------------------
# Helper: build small molecule integrals for testing
# ---------------------------------------------------------------------------

def _h2_integrals():
    """H2 minimal basis (STO-3G like): 2 electrons, 2 orbitals."""
    norb = 2
    # Simple model Hamiltonian for H2
    h1e = np.array([[-1.252, 0.0], [0.0, -0.475]], dtype=np.float64)
    # Two-electron integrals (chemist notation)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    eri[0, 0, 0, 0] = 0.674
    eri[1, 1, 1, 1] = 0.697
    eri[0, 0, 1, 1] = 0.663
    eri[1, 1, 0, 0] = 0.663
    eri[0, 1, 0, 1] = 0.181
    eri[0, 1, 1, 0] = 0.181
    eri[1, 0, 0, 1] = 0.181
    eri[1, 0, 1, 0] = 0.181
    return norb, h1e, eri


def _lih_integrals():
    """LiH minimal basis (STO-3G like): 2 electrons in 2 active orbitals."""
    norb = 2
    h1e = np.array([[-1.10, 0.05], [0.05, -0.50]], dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    eri[0, 0, 0, 0] = 0.55
    eri[1, 1, 1, 1] = 0.60
    eri[0, 0, 1, 1] = 0.48
    eri[1, 1, 0, 0] = 0.48
    eri[0, 1, 0, 1] = 0.15
    eri[0, 1, 1, 0] = 0.15
    eri[1, 0, 0, 1] = 0.15
    eri[1, 0, 1, 0] = 0.15
    return norb, h1e, eri


def _be_integrals():
    """Be atom minimal: 2 electrons in 4 orbitals — larger space for convergence tests."""
    norb = 4
    rng = np.random.default_rng(42)
    # Random symmetric h1e
    h1e = rng.standard_normal((norb, norb)) * 0.5
    h1e = (h1e + h1e.T) / 2
    h1e[np.diag_indices(norb)] = np.array([-1.5, -0.8, -0.3, 0.1])
    # Random symmetric ERI with correct symmetry
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
    # Symmetrize
    eri = (eri + eri.transpose(2, 3, 0, 1)) / 2
    eri = (eri + eri.transpose(1, 0, 3, 2)) / 2
    return norb, h1e, eri


# ---------------------------------------------------------------------------
# Phase 1a: Integral index tests
# ---------------------------------------------------------------------------

class TestHBIntegralIndex:
    def test_build_basic(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        assert isinstance(idx, HeatBathIntegralIndex)
        assert idx.norb == norb

    def test_h1_sorted_descending(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        # h1_abs must be sorted descending
        for i in range(len(idx.h1_abs) - 1):
            assert idx.h1_abs[i] >= idx.h1_abs[i + 1], \
                f"h1_abs not sorted descending at {i}: {idx.h1_abs[i]} < {idx.h1_abs[i+1]}"

    def test_v_abs_sorted_descending_per_pq(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        nops = norb * norb
        for pq in range(nops):
            lo = int(idx.pq_ptr[pq])
            hi = int(idx.pq_ptr[pq + 1])
            for k in range(lo, hi - 1):
                assert idx.v_abs[k] >= idx.v_abs[k + 1], \
                    f"v_abs not sorted descending for pq={pq} at {k}"

    def test_csr_structure_valid(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        nops = norb * norb
        assert idx.pq_ptr.shape == (nops + 1,)
        assert idx.pq_ptr[0] == 0
        assert idx.pq_ptr[-1] == idx.nnz_2e
        # Monotonically increasing
        for i in range(nops):
            assert idx.pq_ptr[i] <= idx.pq_ptr[i + 1]

    def test_pq_max_v_correct(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        nops = norb * norb
        for pq in range(nops):
            lo = int(idx.pq_ptr[pq])
            hi = int(idx.pq_ptr[pq + 1])
            if hi > lo:
                assert idx.pq_max_v[pq] == idx.v_abs[lo], \
                    f"pq_max_v[{pq}] != first v_abs element"
            else:
                assert idx.pq_max_v[pq] == 0.0

    def test_h1_signed_matches_abs(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        np.testing.assert_allclose(np.abs(idx.h1_signed), idx.h1_abs)

    def test_build_from_df(self):
        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        nops = norb * norb

        # Create pseudo DF integrals: L such that L @ L^T ≈ eri_2d
        eri_2d = eri.reshape(nops, nops)
        # Use eigendecomposition for exact factorization
        eigvals, eigvecs = np.linalg.eigh(eri_2d)
        mask = eigvals > 1e-14
        l_full = eigvecs[:, mask] * np.sqrt(eigvals[mask])

        idx_df = build_hb_index_from_df(h_eff, l_full, norb)
        idx_direct = build_hb_index(h_eff, eri, norb)

        # Same structure
        assert idx_df.norb == idx_direct.norb
        assert idx_df.n_h1 == idx_direct.n_h1

    def test_larger_system(self):
        norb, h1e, eri = _be_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        assert idx.norb == norb
        assert idx.nnz_2e > 0

    def test_build_g_base_handles_empty_rows(self):
        norb = 2
        h_eff = np.zeros((norb, norb), dtype=np.float64)
        eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
        eri[0, 0, 0, 1] = 1.0
        eri[0, 0, 1, 0] = 1.0
        eri[0, 1, 0, 0] = 1.0
        eri[1, 0, 0, 0] = 1.0
        idx = build_hb_index(h_eff, eri, norb)

        g_base = build_g_base(idx, cutoff=0.5)

        np.testing.assert_allclose(g_base, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))


# ---------------------------------------------------------------------------
# Phase 1b: Python screening function tests
# ---------------------------------------------------------------------------

class TestScreenedGFlat:
    def test_zero_cutoff_recovers_full(self):
        """With cutoff=0, screened g_flat should equal the full g_flat."""
        from asuka.sci.hb_selection import _python_build_screened_g_flat

        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        # Test with occupation [2, 0] (doubly occupied first orbital)
        occ = np.array([2, 0], dtype=np.int8)
        g_flat = _python_build_screened_g_flat(idx, occ, cutoff=0.0)

        # Build reference g_flat manually
        g_ref = h_eff.copy()
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        if r == s:
                            g_ref[p, q] += 0.5 * occ[r] * eri[p, q, r, s]
                        else:
                            g_ref[p, q] += 0.5 * eri[p, q, r, s]

        np.testing.assert_allclose(g_flat, g_ref, atol=1e-12)

    def test_large_cutoff_zeros_out(self):
        """With a very large cutoff, g_flat should be all zeros."""
        from asuka.sci.hb_selection import _python_build_screened_g_flat

        norb, h1e, eri = _h2_integrals()
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps
        idx = build_hb_index(h_eff, eri, norb)

        occ = np.array([2, 0], dtype=np.int8)
        g_flat = _python_build_screened_g_flat(idx, occ, cutoff=1e10)

        np.testing.assert_allclose(g_flat, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Phase 5: Adaptive epsilon tests
# ---------------------------------------------------------------------------

class TestAdaptiveEpsilon:
    def test_initial(self):
        eps = adaptive_epsilon(1, 100, 10000, eps_init=1e-3, eps_final=1e-6)
        assert eps > 1e-6
        assert eps <= 1e-3

    def test_at_target(self):
        eps = adaptive_epsilon(10, 10000, 10000, eps_init=1e-3, eps_final=1e-6)
        np.testing.assert_allclose(eps, 1e-6, rtol=1e-6)

    def test_beyond_target(self):
        eps = adaptive_epsilon(20, 20000, 10000, eps_init=1e-3, eps_final=1e-6)
        np.testing.assert_allclose(eps, 1e-6, rtol=1e-6)

    def test_monotonically_decreasing(self):
        epsilons = []
        for nsel in [100, 500, 1000, 5000, 10000]:
            eps = adaptive_epsilon(1, nsel, 10000, eps_init=1e-3, eps_final=1e-6)
            epsilons.append(eps)
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i + 1]


class TestSemistochasticPT2:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            semistochastic_pt2(
                None,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 1), dtype=np.float64),
                np.zeros((1,), dtype=np.float64),
                None,
                None,
                {},
                1,
                0,
                1e-12,
                np.zeros((0,), dtype=np.float64),
            )

    def test_legacy_heat_bath_selector_removed(self):
        with pytest.raises(NotImplementedError):
            heat_bath_select_and_pt2()


# ---------------------------------------------------------------------------
# GPU-dependent tests (require CUDA)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.mark.cuda
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestHBSCIGPU:
    """Integration tests that require GPU."""

    def _run_cipsi(self, norb, h1e, eri, nelec, selection_mode, **kwargs):
        """Helper to run GPU CIPSI with given parameters."""
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        drt = build_drt(norb=norb, nelec=nelec, twos_target=0)
        return run_cipsi_trials(
            drt, h1e, eri,
            nroots=1,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=drt.ncsf,
            grow_by=max(1, drt.ncsf // 4),
            max_iter=10,
            epq_mode="no_epq_support_aware",
            selection_mode=selection_mode,
            verbose=0,
            **kwargs,
        )

    def test_hb_energy_convergence(self):
        """HB-SCI variational energy improves with decreasing epsilon."""
        norb, h1e, eri = _be_integrals()

        energies = []
        for eps in [1e-1, 1e-2, 1e-4]:
            res = self._run_cipsi(norb, h1e, eri, 2, "heat_bath", hb_epsilon=eps)
            energies.append(float(res.e_var[0]))

        # Energy should decrease (improve) or stay same with smaller eps
        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i + 1] - 1e-6, \
                f"Energy did not improve: eps series gave {energies}"

    def test_hb_adaptive_schedule(self):
        """HB-SCI with adaptive epsilon schedule runs without error."""
        norb, h1e, eri = _be_integrals()
        res = self._run_cipsi(
            norb, h1e, eri, 2, "heat_bath",
            hb_eps_schedule="adaptive",
            hb_eps_init=1e-1,
            hb_eps_final=1e-6,
        )
        assert res.e_var is not None
        assert res.e_var.shape == (1,)

    def test_hb_pt2_nonzero(self):
        """HB-SCI produces non-zero PT2 correction."""
        norb, h1e, eri = _be_integrals()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        res = run_cipsi_trials(
            drt, h1e, eri,
            nroots=1,
            init_ncsf=2,
            max_ncsf=min(10, drt.ncsf),
            grow_by=2,
            max_iter=3,
            epq_mode="no_epq_support_aware",
            selection_mode="heat_bath",
            hb_epsilon=1e-4,
        )
        # PT2 should be non-zero if space is not complete
        if res.sel_idx.size < drt.ncsf:
            assert np.any(res.e_pt2 != 0.0), "PT2 is zero despite incomplete space"

    def test_hb_matches_frontier_hash_h2(self):
        """HB-SCI with small eps matches frontier_hash CIPSI for H2."""
        norb, h1e, eri = _h2_integrals()

        try:
            res_fh = self._run_cipsi(norb, h1e, eri, 2, "frontier_hash")
        except RuntimeError:
            pytest.skip("frontier_hash not available")

        res_hb = self._run_cipsi(norb, h1e, eri, 2, "heat_bath", hb_epsilon=0.0)

        np.testing.assert_allclose(
            res_hb.e_var, res_fh.e_var, atol=1e-8,
            err_msg="HB-SCI variational energy differs from frontier_hash CIPSI"
        )

    def test_hb_unscreened_matches_frontier_hash_reference(self):
        """Unscreened HB-SCI matches frontier-hash on an incomplete variational space."""
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        kwargs = dict(
            nroots=1,
            init_ncsf=2,
            max_ncsf=4,
            grow_by=2,
            max_iter=1,
            epq_mode="no_epq_support_aware",
            verbose=0,
        )

        res_fh = run_cipsi_trials(drt, h1e, eri, selection_mode="frontier_hash", **kwargs)
        res_hb = run_cipsi_trials(drt, h1e, eri, selection_mode="heat_bath", hb_epsilon=0.0, **kwargs)

        assert np.array_equal(res_hb.sel_idx, res_fh.sel_idx)
        np.testing.assert_allclose(res_hb.e_var, res_fh.e_var, atol=1e-10)
        np.testing.assert_allclose(res_hb.e_pt2, res_fh.e_pt2, atol=1e-10)

    def test_hb_backend_parity_multiroot_history(self, monkeypatch):
        """Python and CUDA backend selectors must agree on the screened frontier history."""
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        kwargs = dict(
            nroots=2,
            init_ncsf=2,
            max_ncsf=4,
            grow_by=2,
            max_iter=1,
            epq_mode="no_epq_support_aware",
            selection_mode="heat_bath",
            hb_epsilon=1e-4,
            verbose=0,
        )

        monkeypatch.setenv("ASUKA_HB_SCI_BACKEND", "python")
        res_py = run_cipsi_trials(drt, h1e, eri, **kwargs)

        monkeypatch.setenv("ASUKA_HB_SCI_BACKEND", "cuda")
        res_cuda = run_cipsi_trials(drt, h1e, eri, **kwargs)

        assert np.array_equal(res_py.sel_idx, res_cuda.sel_idx)
        np.testing.assert_allclose(res_py.history[0]["e_var"], res_cuda.history[0]["e_var"], atol=1e-10)
        np.testing.assert_allclose(res_py.history[0]["e_pt2"], res_cuda.history[0]["e_pt2"], atol=1e-10)

    def test_hb_backend_parity_cuda_fused_default(self, monkeypatch):
        """Default cuda_fused backend must match the exact CUDA screened path."""
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb, h1e, eri = _be_integrals()
        drt = build_drt(norb=norb, nelec=2, twos_target=0)
        kwargs = dict(
            nroots=1,
            init_ncsf=2,
            max_ncsf=6,
            grow_by=2,
            max_iter=2,
            epq_mode="no_epq_support_aware",
            selection_mode="heat_bath",
            hb_epsilon=1e-4,
            verbose=0,
        )

        monkeypatch.delenv("ASUKA_HB_SCI_FUSED_IMPL", raising=False)
        monkeypatch.setenv("ASUKA_HB_SCI_BACKEND", "cuda")
        res_cuda = run_cipsi_trials(drt, h1e, eri, **kwargs)

        monkeypatch.setenv("ASUKA_HB_SCI_BACKEND", "cuda_fused")
        res_fused = run_cipsi_trials(drt, h1e, eri, **kwargs)

        assert np.array_equal(res_cuda.sel_idx, res_fused.sel_idx)
        np.testing.assert_allclose(res_cuda.e_var, res_fused.e_var, atol=1e-10)
        np.testing.assert_allclose(res_cuda.e_pt2, res_fused.e_pt2, atol=1e-10)
        np.testing.assert_allclose(res_cuda.e_tot, res_fused.e_tot, atol=1e-10)
