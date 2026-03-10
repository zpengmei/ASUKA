import numpy as np
import pytest


@pytest.mark.cuda
def test_df_grad_qp_multibar_matches_single_cuda():
    """Validate packed-Qp multibar DF gradient kernels against single-bar launches.

    This isolates the fused multibar CUDA kernels by comparing their output to
    stacking N independent single-bar contractions for the same adjoints.
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

    # Require packed-Qp + multibar kernels.
    if not hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_inplace_device"):
        pytest.skip("CUDA extension lacks packed-Qp sphbar DF gradient kernel")
    if not hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_multibar3_inplace_device"):
        pytest.skip("CUDA extension lacks packed-Qp multibar3 DF gradient kernel")

    from asuka.frontend import Molecule, run_hf
    from asuka.integrals.df_grad_context import DFGradContractionContext

    # Force packed-Qp DF factors for this test (independent of user env).
    monkey = pytest.MonkeyPatch()
    monkey.setenv("ASUKA_DF_AO_PACKED_S2", "1")
    try:
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

        B_sph = scf_out.df_B
        assert int(getattr(B_sph, "ndim", 0)) == 2, "expected packed-Qp DF factors (naux, ntri)"

        ctx = DFGradContractionContext.build(
            scf_out.ao_basis,
            scf_out.aux_basis,
            atom_coords_bohr=np.asarray(mol.coords_bohr, dtype=np.float64),
            backend="cuda",
            df_threads=0,
            L_chol=getattr(scf_out, "df_L", None),
        )
        ctx.ensure_cuda()

        naux, ntri = map(int, B_sph.shape)
        rng = np.random.default_rng(123)
        bar_L_list = [cp.asarray(rng.standard_normal((naux, ntri)), dtype=cp.float64) for _ in range(3)]

        bar_X_list = []
        bar_V_list = []
        nao_sph_val = None
        for bar_L in bar_L_list:
            bar_X, bar_V, nao_sph = ctx._adjoints_device_from_sph_qp(  # noqa: SLF001
                B_sph=B_sph,
                bar_L_sph=bar_L,
                precision="fp64",
            )
            bar_X_list.append(bar_X)
            bar_V_list.append(bar_V)
            if nao_sph_val is None:
                nao_sph_val = int(nao_sph)
            else:
                assert int(nao_sph_val) == int(nao_sph)

        # Fused multibar (nbar=3)
        g_multi = ctx._contract_device_from_adjoints_sph_qp_multibar(  # noqa: SLF001
            bar_X_list,
            bar_V_list,
            nao_sph=int(nao_sph_val or 0),
        )

        # Reference: stack N independent single-bar launches.
        g_ref = cp.stack(
            [
                ctx._contract_device_from_adjoints_sph_qp(  # noqa: SLF001
                    bx,
                    bv,
                    nao_sph=int(nao_sph_val or 0),
                    grad_dev=None,
                )
                for bx, bv in zip(bar_X_list, bar_V_list)
            ],
            axis=0,
        )

        g_multi_np = np.asarray(cp.asnumpy(g_multi), dtype=np.float64)
        g_ref_np = np.asarray(cp.asnumpy(g_ref), dtype=np.float64)
        assert g_multi_np.shape == g_ref_np.shape

        # Allow small differences due to atomic reduction ordering.
        assert np.allclose(g_multi_np, g_ref_np, rtol=1e-8, atol=1e-8)
    finally:
        monkey.undo()

