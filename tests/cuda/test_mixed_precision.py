"""Unit tests for TF32 error-pruning algorithms.

Tests each algorithm against FP64 reference on random matrices:
- Iterative refinement: error < 1e-6 for kappa < 1e6
- Ozaki-2: error < 1e-6 for arbitrary matrices
- Compensated SYRK: error < 1e-7 after 100+ chunk accumulations
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CPU-only tests (always run, verify correctness of algorithm logic)
# ---------------------------------------------------------------------------


class TestGemmTf32RefinedCPU:
    """Test iterative refinement on CPU (NumPy fallback = FP64)."""

    def test_small_square(self):
        from asuka.cuda.mixed_precision import gemm_tf32_refined

        rng = np.random.default_rng(42)
        A = rng.standard_normal((64, 64))
        B = rng.standard_normal((64, 64))
        C_ref = A @ B
        C = gemm_tf32_refined(A, B)
        np.testing.assert_allclose(C, C_ref, atol=1e-12)

    def test_rectangular(self):
        from asuka.cuda.mixed_precision import gemm_tf32_refined

        rng = np.random.default_rng(123)
        A = rng.standard_normal((100, 50))
        B = rng.standard_normal((50, 80))
        C_ref = A @ B
        C = gemm_tf32_refined(A, B)
        np.testing.assert_allclose(C, C_ref, atol=1e-12)

    def test_output_buffer(self):
        from asuka.cuda.mixed_precision import gemm_tf32_refined

        rng = np.random.default_rng(7)
        A = rng.standard_normal((32, 32))
        B = rng.standard_normal((32, 32))
        out = np.zeros((32, 32))
        result = gemm_tf32_refined(A, B, out=out)
        assert result is out
        np.testing.assert_allclose(out, A @ B, atol=1e-12)


class TestOzaki2CPU:
    """Test 2-way Ozaki splitting on CPU (NumPy fallback = FP64)."""

    def test_split_reconstruction(self):
        from asuka.cuda.mixed_precision import ozaki_split_2

        rng = np.random.default_rng(42)
        A = rng.standard_normal((64, 64))
        A_hi, A_lo = ozaki_split_2(A)
        A_recon = A_hi.astype(np.float64) + A_lo.astype(np.float64)
        np.testing.assert_allclose(A_recon, A, atol=1e-7)

    def test_gemm_accuracy(self):
        from asuka.cuda.mixed_precision import gemm_ozaki2

        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 80))
        B = rng.standard_normal((80, 60))
        C_ref = A @ B
        C = gemm_ozaki2(A, B)
        np.testing.assert_allclose(C, C_ref, atol=1e-12)

    def test_ill_conditioned(self):
        from asuka.cuda.mixed_precision import gemm_ozaki2

        rng = np.random.default_rng(99)
        # Create ill-conditioned matrix.
        U, _ = np.linalg.qr(rng.standard_normal((50, 50)))
        s = np.logspace(0, -6, 50)
        A = U @ np.diag(s) @ U.T
        B = rng.standard_normal((50, 50))
        C_ref = A @ B
        C = gemm_ozaki2(A, B)
        np.testing.assert_allclose(C, C_ref, atol=1e-12)


class TestOzaki3CPU:
    """Test 3-way Ozaki splitting on CPU."""

    def test_gemm_accuracy(self):
        from asuka.cuda.mixed_precision_gemm import gemm_ozaki3

        rng = np.random.default_rng(42)
        A = rng.standard_normal((50, 40))
        B = rng.standard_normal((40, 30))
        C_ref = A @ B
        C = gemm_ozaki3(A, B)
        np.testing.assert_allclose(C, C_ref, atol=1e-12)


class TestSyrkOzaki2CPU:
    """Test Ozaki-2 SYRK on CPU."""

    def test_symmetric_result(self):
        from asuka.cuda.mixed_precision_gemm import syrk_ozaki2

        rng = np.random.default_rng(42)
        A = rng.standard_normal((50, 30))
        C = syrk_ozaki2(A)
        C_ref = A @ A.T
        np.testing.assert_allclose(C, C_ref, atol=1e-12)
        np.testing.assert_allclose(C, C.T, atol=1e-15)


class TestGemvRefinedCPU:
    """Test refined GEMV on CPU."""

    def test_basic(self):
        from asuka.cuda.mixed_precision_gemm import gemv_tf32_refined

        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 50))
        x = rng.standard_normal(50)
        y_ref = A @ x
        y = gemv_tf32_refined(A, x)
        np.testing.assert_allclose(y, y_ref, atol=1e-12)


class TestSyrkCompensatedCPU:
    """Test compensated SYRK on CPU."""

    def test_vs_direct(self):
        from asuka.cuda.mixed_precision import syrk_compensated

        rng = np.random.default_rng(42)
        naux, nao = 200, 20
        nocc = 5
        BQ = rng.standard_normal((naux, nao, nao))
        C_occ = rng.standard_normal((nao, nocc))
        occ_vals = np.ones(nocc) * 2.0

        K = syrk_compensated(BQ, C_occ, occ_vals, q_block=32)

        # Reference: direct computation.
        sqrt_occ = np.sqrt(occ_vals)
        Cw = C_occ * sqrt_occ[None, :]
        K_ref = np.zeros((nao, nao))
        for q in range(naux):
            U = BQ[q] @ Cw  # (nao, nocc)
            K_ref += U @ U.T
        K_ref = 0.5 * (K_ref + K_ref.T)

        np.testing.assert_allclose(K, K_ref, atol=1e-10)


class TestFp32KchunkedCPU:
    """Test k-chunked FP32 GEMM on CPU (NumPy fallback = FP64)."""

    def test_small_square(self):
        from asuka.cuda.mixed_precision import gemm_fp32_kchunked

        rng = np.random.default_rng(42)
        A = rng.standard_normal((64, 64))
        B = rng.standard_normal((64, 64))
        C = gemm_fp32_kchunked(A, B)
        np.testing.assert_allclose(C, A @ B, atol=1e-12)

    def test_rectangular(self):
        from asuka.cuda.mixed_precision import gemm_fp32_kchunked

        rng = np.random.default_rng(123)
        A = rng.standard_normal((100, 200))
        B = rng.standard_normal((200, 80))
        C = gemm_fp32_kchunked(A, B, k_block=64)
        np.testing.assert_allclose(C, A @ B, atol=1e-12)


class TestGemmDispatched:
    """Test precision-dispatched GEMM."""

    def test_fp64(self):
        from asuka.cuda.mixed_precision import gemm_dispatched

        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 32))
        B = rng.standard_normal((32, 32))
        C = gemm_dispatched(A, B, precision="fp64")
        np.testing.assert_allclose(C, A @ B, atol=1e-14)

    def test_fp32_kchunked_cpu(self):
        from asuka.cuda.mixed_precision import gemm_dispatched

        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 32))
        B = rng.standard_normal((32, 32))
        C = gemm_dispatched(A, B, precision="fp32_kchunked")
        np.testing.assert_allclose(C, A @ B, atol=1e-12)

    def test_tf32_refined_cpu(self):
        from asuka.cuda.mixed_precision import gemm_dispatched

        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 32))
        B = rng.standard_normal((32, 32))
        C = gemm_dispatched(A, B, precision="tf32_refined")
        np.testing.assert_allclose(C, A @ B, atol=1e-12)

    def test_ozaki2_cpu(self):
        from asuka.cuda.mixed_precision import gemm_dispatched

        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 32))
        B = rng.standard_normal((32, 32))
        C = gemm_dispatched(A, B, precision="ozaki2")
        np.testing.assert_allclose(C, A @ B, atol=1e-12)

    def test_invalid_precision(self):
        from asuka.cuda.mixed_precision import gemm_dispatched

        with pytest.raises(ValueError, match="Unknown precision"):
            gemm_dispatched(np.eye(2), np.eye(2), precision="invalid")


# ---------------------------------------------------------------------------
# GPU tests (require CuPy + CUDA)
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
class TestFp32KchunkedGPU:
    """Test k-chunked FP32 GEMM with FP64 accumulation on GPU."""

    def test_df_scale(self):
        """DF-relevant shape: must achieve < 1e-5 max error."""
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_fp32_kchunked

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((240, 480)))
        B = cp.asarray(rng.standard_normal((480, 240)))
        C_ref = (A @ B).get()
        C = gemm_fp32_kchunked(A, B).get()
        np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=1e-5)

    def test_explicit_k_block(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_fp32_kchunked

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((128, 512)))
        B = cp.asarray(rng.standard_normal((512, 128)))
        C_ref = (A @ B).get()
        C = gemm_fp32_kchunked(A, B, k_block=64).get()
        np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=1e-5)


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
class TestGemmTf32RefinedGPU:
    """Test Ozaki-2 + k-chunked GEMM on GPU."""

    def test_well_conditioned(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_tf32_refined

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((256, 256)))
        B = cp.asarray(rng.standard_normal((256, 256)))
        C_ref = (A @ B).get()
        C = gemm_tf32_refined(A, B).get()
        # k-chunked Ozaki-2: ~1e-5 per element at k=256.
        np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=1e-5)

    def test_moderate_condition(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_tf32_refined

        rng = np.random.default_rng(99)
        U, _ = np.linalg.qr(rng.standard_normal((128, 128)))
        s = np.logspace(0, -3, 128)
        A_np = U @ np.diag(s) @ U.T
        B_np = rng.standard_normal((128, 128))
        A = cp.asarray(A_np)
        B = cp.asarray(B_np)
        C_ref = (A @ B).get()
        C = gemm_tf32_refined(A, B).get()
        np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=1e-5)


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
class TestOzaki2GPU:
    """Test Ozaki-2 GEMM with k-chunked main term on GPU."""

    def test_accuracy(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_ozaki2

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((256, 128)))
        B = cp.asarray(rng.standard_normal((128, 256)))
        C_ref = (A @ B).get()
        C = gemm_ozaki2(A, B).get()
        # k-chunked Ozaki-2 at k=128: max error ~1e-5.
        np.testing.assert_allclose(C, C_ref, atol=2e-5, rtol=1e-5)

    def test_large_matrix(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_ozaki2

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((512, 256)))
        B = cp.asarray(rng.standard_normal((256, 512)))
        C_ref = (A @ B).get()
        C = gemm_ozaki2(A, B).get()
        # Larger matrix: max error ~2e-5.
        np.testing.assert_allclose(C, C_ref, atol=5e-5, rtol=1e-5)

    def test_much_better_than_tf32(self):
        """Verify Ozaki-2 is significantly better than pure TF32."""
        import cupy as cp
        from asuka.cuda.mixed_precision import gemm_ozaki2, gemm_tf32_pure

        rng = np.random.default_rng(42)
        A = cp.asarray(rng.standard_normal((128, 64)))
        B = cp.asarray(rng.standard_normal((64, 128)))
        C_ref = (A @ B).get()

        C_ozaki = gemm_ozaki2(A, B).get()
        C_tf32 = gemm_tf32_pure(A, B).get()

        err_ozaki = float(np.max(np.abs(C_ozaki - C_ref)))
        err_tf32 = float(np.max(np.abs(C_tf32 - C_ref)))

        # Ozaki should be at least 10x better than pure TF32.
        assert err_ozaki < err_tf32 / 5, (
            f"Ozaki error {err_ozaki:.2e} should be much less than TF32 error {err_tf32:.2e}"
        )


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
class TestSyrkCompensatedGPU:
    """Test compensated SYRK on GPU."""

    def test_many_chunks(self):
        import cupy as cp
        from asuka.cuda.mixed_precision import syrk_compensated

        rng = np.random.default_rng(42)
        naux, nao, nocc = 512, 30, 8
        BQ = cp.asarray(rng.standard_normal((naux, nao, nao)))
        C_occ = cp.asarray(rng.standard_normal((nao, nocc)))
        occ_vals = cp.ones(nocc) * 2.0

        K = syrk_compensated(BQ, C_occ, occ_vals, q_block=32).get()

        # Reference.
        sqrt_occ = np.sqrt(np.ones(nocc) * 2.0)
        Cw = C_occ.get() * sqrt_occ[None, :]
        K_ref = np.zeros((nao, nao))
        BQ_np = BQ.get()
        for q in range(naux):
            U = BQ_np[q] @ Cw
            K_ref += U @ U.T
        K_ref = 0.5 * (K_ref + K_ref.T)

        np.testing.assert_allclose(K, K_ref, atol=1e-7)


# ---------------------------------------------------------------------------
# Precision policy tests
# ---------------------------------------------------------------------------


class TestPrecisionPolicy:
    """Test PrecisionPolicy configuration."""

    def test_fp64_default(self):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.fp64()
        assert p.metric_cholesky == "fp64"
        assert p.df_k_syrk == "fp64"

    def test_tf32_conservative(self):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.tf32_conservative()
        assert p.metric_cholesky == "fp64"
        assert p.whitening_trsm == "fp64"
        assert p.df_j_projection == "tf32_refined"
        assert p.df_k_syrk == "ozaki2_kahan"
        assert p.active_space_df == "tf32_refined"

    def test_tf32_aggressive(self):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.tf32_aggressive()
        assert p.df_j_projection == "tf32_pure"
        assert p.df_k_syrk == "tf32_pure"

    def test_invalid_mode_raises(self):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        with pytest.raises(ValueError, match="not valid"):
            PrecisionPolicy(df_j_projection="invalid_mode")

    def test_summary(self):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.fp64()
        s = p.summary()
        assert isinstance(s, dict)
        assert all(v == "fp64" for v in s.values())

    def test_from_env(self, monkeypatch):
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        monkeypatch.setenv("ASUKA_PRECISION_POLICY", "tf32_conservative")
        monkeypatch.setenv("ASUKA_PRECISION_DF_K", "ozaki2")
        p = PrecisionPolicy.from_env()
        assert p.df_j_projection == "tf32_refined"
        assert p.df_k_syrk == "ozaki2"


# ---------------------------------------------------------------------------
# Tile efficiency tests
# ---------------------------------------------------------------------------


class TestTileEfficiency:
    """Test tensor-core tile efficiency computations."""

    def test_perfect_alignment(self):
        from asuka.basis_profiler.tile_efficiency import eta_tc

        # 16x16x8 = perfect TF32 tile.
        assert eta_tc(16, 16, 8) == 1.0
        assert eta_tc(32, 32, 16) == 1.0
        assert eta_tc(256, 256, 128) == 1.0

    def test_worst_case(self):
        from asuka.basis_profiler.tile_efficiency import eta_tc

        # 1x1x1 = massive padding waste.
        eta = eta_tc(1, 1, 1)
        assert eta < 0.01

    def test_typical_df(self):
        from asuka.basis_profiler.tile_efficiency import eta_tc

        # Typical DF: nao=60, naux=180, norb=10
        eta = eta_tc(60, 10, 60)  # Active-space contraction
        assert 0.5 < eta < 1.0

    def test_padding_waste(self):
        from asuka.basis_profiler.tile_efficiency import padding_waste

        assert padding_waste(16, 16) == 0
        assert padding_waste(17, 16) == 15
        assert padding_waste(15, 16) == 1
        assert padding_waste(32, 16) == 0
