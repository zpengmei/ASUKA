import numpy as np
import pytest


@pytest.mark.cuda
def test_df_casscf_spherical_energy_grad_cuda_smoke():
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

    # Required kernels for cart-free spherical DF build + gradient on CUDA.
    if not hasattr(_ext, "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device"):
        pytest.skip("CUDA extension lacks spherical DF int3c2e scatter support")
    if not hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_inplace_device"):
        pytest.skip("CUDA extension lacks sphbar-qmn DF gradient kernel")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df

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
    assert getattr(scf_out, "sph_map", None) is not None
    _T, nao_cart, nao_sph = scf_out.sph_map
    assert int(nao_sph) < int(nao_cart)

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

    g = casscf_nuc_grad_df(scf_out, mc, df_backend="cuda")
    assert isinstance(g.e_tot, float)
    assert g.grad.shape == (int(mol.natm), 3)
    assert bool(np.isfinite(g.grad).all())

