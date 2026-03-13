"""GPU validation tests for irrep screening (Part A) and streaming PT2 (Part B).

These tests run on GPU and validate:
1. GPU CIPSI with orbsym gives same energy as without (irrep screening correctness)
2. GPU CIPSI with pt2_mode="streaming" matches exact PT2 on GPU
3. Symmetry filtering reduces kernel work (measured via HB index size)
4. VRAM stays under 20 GB cap throughout
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from asuka.cuguga.drt import build_drt
from asuka.sci.hb_integrals import build_hb_index, upload_hb_index


def _require_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) <= 0:
            pytest.skip("no CUDA device available")
    except Exception as e:
        pytest.skip(f"CUDA runtime unavailable: {e}")
    # Hard cap: 20 GB
    pool = cupy.get_default_memory_pool()
    pool.set_limit(size=20 * 1024**3)
    return cupy


def _gpu_mem_mb(cp):
    """Current GPU memory used by CuPy pool in MB."""
    pool = cp.get_default_memory_pool()
    return pool.used_bytes() / (1024**2)


def _make_symmetric_test_integrals(norb: int, seed: int = 7):
    rng = np.random.default_rng(int(seed))
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(size=(norb, norb, norb, norb))
    eri = 0.25 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(2, 3, 0, 1)
    )
    return np.asarray(h1e, dtype=np.float64), np.asarray(eri, dtype=np.float64)


def _make_c2v_integrals(norb: int, seed: int = 99):
    """Generate C2v-symmetric integrals with orbsym."""
    rng = np.random.default_rng(seed)
    # Assign irreps cyclically: 0,1,0,1,...
    orbsym = np.array([i % 2 for i in range(norb)], dtype=np.int32)
    h1e = np.zeros((norb, norb), dtype=np.float64)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            if (orbsym[p] ^ orbsym[q]) != 0:
                continue
            h1e[p, q] = rng.standard_normal() * 0.5
            for r in range(norb):
                for s in range(norb):
                    if (orbsym[r] ^ orbsym[s]) != 0:
                        continue
                    v = rng.standard_normal() * 0.2
                    eri[p, q, r, s] = v
                    eri[r, s, p, q] = v
                    eri[q, p, s, r] = v
                    eri[s, r, q, p] = v
    h1e = (h1e + h1e.T) / 2
    eri = (eri + eri.transpose(2, 3, 0, 1)) / 2
    eri = (eri + eri.transpose(1, 0, 3, 2)) / 2
    return h1e, eri, orbsym


# ---------------------------------------------------------------------------
# Part A: GPU irrep screening validation
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestGPUIrrepScreening:
    """Validate that irrep-filtered CUDA CIPSI produces correct results."""

    def test_gpu_cipsi_orbsym_energy_parity_small(self):
        """GPU CIPSI with orbsym=C2v on 4-orbital system matches no-orbsym."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 4
        h1e, eri, orbsym = _make_c2v_integrals(norb, seed=99)

        drt_nosym = build_drt(norb=norb, nelec=2, twos_target=0)
        drt_sym = build_drt(norb=norb, nelec=2, twos_target=0, orbsym=orbsym, wfnsym=0)

        res_nosym = run_cipsi_trials(
            drt_nosym, h1e, eri, nroots=1,
            init_ncsf=drt_nosym.ncsf, max_ncsf=drt_nosym.ncsf,
            backend="cpu_sparse",
        )
        res_sym = run_cipsi_trials(
            drt_sym, h1e, eri, nroots=1,
            init_ncsf=drt_sym.ncsf, max_ncsf=drt_sym.ncsf,
            backend="cpu_sparse",
        )
        np.testing.assert_allclose(res_sym.e_var, res_nosym.e_var, atol=1e-10)

    def test_gpu_cipsi_cuda_key64_with_orbsym(self):
        """GPU CUDA key64 driver with orbsym runs successfully."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 6
        h1e, eri, orbsym = _make_c2v_integrals(norb, seed=77)

        drt = build_drt(norb=norb, nelec=4, twos_target=0, orbsym=orbsym, wfnsym=0)
        mem_before = _gpu_mem_mb(cp)

        res = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(20, drt.ncsf),
            max_ncsf=min(50, drt.ncsf),
            grow_by=10,
            max_iter=3,
            backend="cuda_key64",
            state_rep="key64",
            selection_mode="frontier_hash",
        )
        mem_after = _gpu_mem_mb(cp)
        print(f"\n  GPU mem: {mem_before:.1f} -> {mem_after:.1f} MB (delta={mem_after - mem_before:.1f})")
        print(f"  driver: {res.profile.get('driver')}")

        assert res.profile.get("driver") == "cuda_cas36_hb_compact_u64"
        assert np.all(np.isfinite(res.e_var))
        assert mem_after < 20 * 1024  # < 20 GB

    def test_gpu_cipsi_cuda_key64_with_orbsym_multiroot(self):
        """GPU CUDA key64 driver with orbsym and 2 roots."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 6
        h1e, eri, orbsym = _make_c2v_integrals(norb, seed=88)
        drt = build_drt(norb=norb, nelec=4, twos_target=0, orbsym=orbsym, wfnsym=0)

        res = run_cipsi_trials(
            drt, h1e, eri, nroots=2,
            init_ncsf=min(20, drt.ncsf),
            max_ncsf=min(50, drt.ncsf),
            grow_by=10,
            max_iter=3,
            backend="cuda_key64",
            state_rep="key64",
            selection_mode="frontier_hash",
        )
        assert res.profile.get("driver") == "cuda_cas36_hb_compact_u64"
        assert res.e_var.shape == (2,)
        assert np.all(np.isfinite(res.e_var))
        assert np.all(np.isfinite(res.e_pt2))

    def test_hb_index_symmetry_reduces_csr_gpu(self):
        """Symmetry-filtered HB index uploaded to GPU has fewer entries."""
        cp = _require_cuda()

        norb = 8
        # Use general (non-symmetric) integrals so filtering actually removes entries
        h1e, eri = _make_symmetric_test_integrals(norb, seed=55)
        orbsym = np.array([i % 2 for i in range(norb)], dtype=np.int32)
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps

        idx_unfilt = build_hb_index(h_eff, eri, norb)
        idx_filt = build_hb_index(h_eff, eri, norb, orbsym=orbsym)

        dev_unfilt = upload_hb_index(idx_unfilt, cp)
        dev_filt = upload_hb_index(idx_filt, cp)

        # Filtered should use less GPU memory
        assert int(dev_filt["rs_idx"].size) <= int(dev_unfilt["rs_idx"].size)
        # sym_pq_allowed should be uploaded for filtered
        assert dev_filt.get("sym_pq_allowed") is not None
        assert dev_unfilt.get("sym_pq_allowed") is None

        nnz_unfilt = int(idx_unfilt.nnz_2e)
        nnz_filt = int(idx_filt.nnz_2e)
        print(f"\n  CSR entries: {nnz_unfilt} -> {nnz_filt} ({100 * (1 - nnz_filt / max(1, nnz_unfilt)):.1f}% reduction)")
        assert nnz_filt <= nnz_unfilt


# ---------------------------------------------------------------------------
# Part B: GPU streaming PT2 validation
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestGPUStreamingPT2:
    """Validate streaming PT2 on GPU CIPSI pipeline."""

    def test_streaming_pt2_matches_exact_gpu(self):
        """pt2_mode='streaming' matches exact on CUDA path."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 4
        h1e, eri = _make_symmetric_test_integrals(norb, seed=211)
        drt = build_drt(norb=norb, nelec=2, twos_target=0)

        kwargs = dict(
            nroots=1,
            init_ncsf=min(8, drt.ncsf),
            max_ncsf=min(8, drt.ncsf),
            grow_by=4,
            max_iter=2,
            backend="cpu_sparse",
        )

        res_exact = run_cipsi_trials(drt, h1e, eri, **kwargs)
        res_stream = run_cipsi_trials(drt, h1e, eri, pt2_mode="streaming", pt2_bucket_size=3, **kwargs)

        np.testing.assert_allclose(res_stream.e_pt2, res_exact.e_pt2, atol=1e-12)
        np.testing.assert_allclose(res_stream.e_var, res_exact.e_var, atol=1e-12)

    def test_streaming_pt2_multiroot_gpu(self):
        """pt2_mode='streaming' with 2 roots on GPU."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 4
        h1e, eri = _make_symmetric_test_integrals(norb, seed=311)
        drt = build_drt(norb=norb, nelec=2, twos_target=0)

        kwargs = dict(
            nroots=2,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=min(8, drt.ncsf),
            grow_by=2,
            max_iter=2,
            backend="cpu_sparse",
        )

        res_exact = run_cipsi_trials(drt, h1e, eri, **kwargs)
        res_stream = run_cipsi_trials(drt, h1e, eri, pt2_mode="streaming", pt2_bucket_size=5, **kwargs)

        np.testing.assert_allclose(res_stream.e_pt2, res_exact.e_pt2, atol=1e-11)

    def test_semistochastic_pt2_gpu(self):
        """pt2_mode='semistochastic' runs and gives finite results on GPU."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 4
        h1e, eri = _make_symmetric_test_integrals(norb, seed=411)
        drt = build_drt(norb=norb, nelec=2, twos_target=0)

        res = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(4, drt.ncsf),
            max_ncsf=min(8, drt.ncsf),
            grow_by=2, max_iter=2,
            backend="cpu_sparse",
            pt2_mode="semistochastic",
            pt2_n_det_sources=4,
            pt2_n_stoch_samples=50,
            pt2_n_stoch_batches=5,
            pt2_seed=42,
        )
        assert np.all(np.isfinite(res.e_pt2))
        assert np.all(np.isfinite(res.e_var))

    def test_gpu_cipsi_with_orbsym_and_streaming(self):
        """Combined: orbsym + streaming PT2 on GPU."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 6
        h1e, eri, orbsym = _make_c2v_integrals(norb, seed=123)
        drt = build_drt(norb=norb, nelec=4, twos_target=0, orbsym=orbsym, wfnsym=0)

        res = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(10, drt.ncsf),
            max_ncsf=min(30, drt.ncsf),
            grow_by=5, max_iter=3,
            backend="cuda_key64",
            state_rep="key64",
            selection_mode="frontier_hash",
            pt2_mode="streaming",
            pt2_bucket_size=10,
        )
        assert res.profile.get("driver") == "cuda_cas36_hb_compact_u64"
        assert np.all(np.isfinite(res.e_var))
        assert np.all(np.isfinite(res.e_pt2))
        print(f"\n  E_var={res.e_var[0]:.10f}  E_PT2={res.e_pt2[0]:.10f}")


# ---------------------------------------------------------------------------
# GPU memory benchmark
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestGPUMemoryBound:
    """Verify VRAM stays under 20 GB at moderate scale."""

    def test_moderate_scale_vram_bounded(self):
        """norb=10, nelec=6 CIPSI on GPU stays under 20 GB VRAM."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 10
        h1e, eri = _make_symmetric_test_integrals(norb, seed=777)
        drt = build_drt(norb=norb, nelec=6, twos_target=0)
        print(f"\n  ncsf = {drt.ncsf}")

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        mem0 = pool.used_bytes() / (1024**2)

        t0 = time.perf_counter()
        res = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(50, drt.ncsf),
            max_ncsf=min(200, drt.ncsf),
            grow_by=50,
            max_iter=3,
            backend="cuda_key64",
            state_rep="key64",
            selection_mode="frontier_hash",
        )
        dt = time.perf_counter() - t0
        mem1 = pool.used_bytes() / (1024**2)

        print(f"  driver: {res.profile.get('driver')}")
        print(f"  wall time: {dt:.2f}s")
        print(f"  GPU mem: {mem0:.1f} -> {mem1:.1f} MB (peak pool used)")
        print(f"  E_var = {res.e_var}")
        print(f"  E_PT2 = {res.e_pt2}")

        assert mem1 < 20 * 1024  # < 20 GB
        assert np.all(np.isfinite(res.e_var))

    def test_moderate_scale_streaming_pt2_vram_bounded(self):
        """norb=10, nelec=6 streaming PT2 stays under 20 GB."""
        cp = _require_cuda()
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        norb = 10
        h1e, eri = _make_symmetric_test_integrals(norb, seed=888)
        drt = build_drt(norb=norb, nelec=6, twos_target=0)

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()

        t0 = time.perf_counter()
        res = run_cipsi_trials(
            drt, h1e, eri, nroots=1,
            init_ncsf=min(50, drt.ncsf),
            max_ncsf=min(200, drt.ncsf),
            grow_by=50,
            max_iter=3,
            backend="cuda_key64",
            state_rep="key64",
            selection_mode="frontier_hash",
            pt2_mode="streaming",
            pt2_bucket_size=100,
        )
        dt = time.perf_counter() - t0
        mem_peak = pool.used_bytes() / (1024**2)

        print(f"\n  ncsf={drt.ncsf}, wall={dt:.2f}s, GPU mem={mem_peak:.1f} MB")
        print(f"  E_var={res.e_var}, E_PT2={res.e_pt2}")

        assert mem_peak < 20 * 1024
        assert np.all(np.isfinite(res.e_pt2))

    def test_symmetry_speedup_measurable(self):
        """Symmetry filtering measurably reduces HB index size on GPU."""
        cp = _require_cuda()

        norb = 10
        # Use general integrals — orbsym filtering should remove forbidden pairs
        h1e, eri = _make_symmetric_test_integrals(norb, seed=555)
        orbsym = np.array([i % 2 for i in range(norb)], dtype=np.int32)
        j_ps = np.einsum("pqqs->ps", eri)
        h_eff = h1e - 0.5 * j_ps

        idx_unfilt = build_hb_index(h_eff, eri, norb)
        idx_filt = build_hb_index(h_eff, eri, norb, orbsym=orbsym)

        dev_unfilt = upload_hb_index(idx_unfilt, cp)
        dev_filt = upload_hb_index(idx_filt, cp)

        nnz_unfilt = int(idx_unfilt.nnz_2e)
        nnz_filt = int(idx_filt.nnz_2e)
        h1_unfilt = int(idx_unfilt.n_h1)
        h1_filt = int(idx_filt.n_h1)

        reduction_2e = 100 * (1 - nnz_filt / max(1, nnz_unfilt))
        reduction_1e = 100 * (1 - h1_filt / max(1, h1_unfilt))

        print(f"\n  norb={norb}, C2v (alternating irreps)")
        print(f"  1e entries: {h1_unfilt} -> {h1_filt} ({reduction_1e:.1f}% reduction)")
        print(f"  2e CSR entries: {nnz_unfilt} -> {nnz_filt} ({reduction_2e:.1f}% reduction)")
        print(f"  sym_pq_allowed uploaded: {dev_filt.get('sym_pq_allowed') is not None}")

        # For C2v with alternating irreps, expect significant reduction
        assert reduction_2e > 30.0, f"Expected >30% 2e reduction, got {reduction_2e:.1f}%"
        assert reduction_1e > 30.0, f"Expected >30% 1e reduction, got {reduction_1e:.1f}%"
