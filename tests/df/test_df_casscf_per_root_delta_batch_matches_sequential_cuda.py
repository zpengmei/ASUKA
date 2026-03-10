import numpy as np
import pytest


@pytest.mark.cuda
def test_df_casscf_per_root_delta_batch_matches_sequential_cuda():
    """End-to-end check: per-root DF-CASSCF gradients batched vs sequential.

    This verifies the delta-batching + SA+delta fusion logic does not change
    per-root gradients compared to the sequential contraction path (fp64).
    """
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    try:
        from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    except Exception:
        pytest.skip("cuERI CUDA extension is unavailable")

    # Require packed-Qp + multibar kernels for the batched path.
    if not hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_multibar3_inplace_device"):
        pytest.skip("CUDA extension lacks packed-Qp multibar3 DF gradient kernel")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df_per_root

    monkey = pytest.MonkeyPatch()
    try:
        # Force packed-Qp DF and deterministic fp64 contractions for comparison.
        monkey.setenv("ASUKA_DF_AO_PACKED_S2", "1")
        monkey.setenv("ASUKA_DF_HYBRID_MODE", "off")
        monkey.setenv("ASUKA_DF_FUSED_CONTRACT_PRECISION", "fp64")
        monkey.setenv("ASUKA_DF_GRAD_PATH_LOG", "0")

        mol = Molecule.from_atoms(
            atoms=[
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ],
            basis="6-31g*",
            cart=False,
        )

        scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=50, conv_tol=1e-10, conv_tol_dm=1e-8)
        assert bool(getattr(scf_out.scf, "converged", False))

        mc = run_casscf(
            scf_out,
            ncore=4,
            ncas=2,
            nelecas=2,
            backend="cuda",
            max_cycle_macro=30,
            nroots=2,
            root_weights=[0.5, 0.5],
        )
        assert bool(getattr(mc, "converged", False))

        # Sequential path (no delta batching).
        monkey.setenv("ASUKA_DF_DELTA_BATCH_CONTRACT", "0")
        g_seq = casscf_nuc_grad_df_per_root(scf_out, mc, df_backend="cuda")

        # Batched path (fp64 adjoints).
        monkey.setenv("ASUKA_DF_DELTA_BATCH_CONTRACT", "1")
        monkey.setenv("ASUKA_DF_DELTA_BATCH_NBAR", "2")
        monkey.setenv("ASUKA_DF_DELTA_BATCH_PRECISION", "fp64")
        g_batched = casscf_nuc_grad_df_per_root(scf_out, mc, df_backend="cuda")

        grads_seq = np.asarray(g_seq.grads, dtype=np.float64)
        grads_batched = np.asarray(g_batched.grads, dtype=np.float64)
        assert grads_seq.shape == grads_batched.shape

        # Allow small differences due to atomic reduction ordering.
        assert np.allclose(grads_batched, grads_seq, rtol=5e-7, atol=5e-8)
    finally:
        monkey.undo()

