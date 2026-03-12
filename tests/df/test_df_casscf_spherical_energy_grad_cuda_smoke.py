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


@pytest.mark.cuda
def test_df_casscf_short_run_returns_finite_outer_state_cuda():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert bool(getattr(scf_out.scf, "converged", False))

    mc = run_casscf(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        orbital_optimizer="1step",
        max_cycle_macro=1,
        tol=1e-10,
        conv_tol_grad=1e-6,
    )

    assert np.isfinite(float(mc.e_tot))
    assert np.isfinite(np.asarray(mc.e_roots, dtype=np.float64)).all()
    assert np.isfinite(float(mc.grad_norm))
    assert int(mc.niter) == 1


@pytest.mark.cuda
def test_dense_gpu_casscf_short_run_returns_finite_outer_state_cuda():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf.casscf import run_casscf_df

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert bool(getattr(scf_out.scf, "converged", False))

    mc = run_casscf_df(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        casci_backend="dense_gpu",
        matvec_backend="cuda_eri_mat",
        orbital_optimizer="1step",
        max_cycle_macro=1,
        tol=1e-10,
        conv_tol_grad=1e-6,
    )

    assert np.isfinite(float(mc.e_tot))
    assert np.isfinite(np.asarray(mc.e_roots, dtype=np.float64)).all()
    assert np.isfinite(float(mc.grad_norm))
    assert int(mc.niter) == 1


@pytest.mark.cuda
def test_cuda_make_rdm12_accepts_cupy_ci_and_returns_cupy():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf.casci import run_casci
    from asuka.solver import GUGAFCISolver

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert bool(getattr(scf_out.scf, "converged", False))

    solver = GUGAFCISolver(twos=0, nroots=1)
    cas = run_casci(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        matvec_backend="cuda",
        fcisolver=solver,
    )
    ci_dev = cp.asarray(np.asarray(cas.ci, dtype=np.float64).ravel(), dtype=cp.float64)
    dm1, dm2 = solver.make_rdm12(
        ci_dev,
        2,
        2,
        rdm_backend="cuda",
        return_cupy=True,
        strict_gpu=True,
    )
    assert isinstance(dm1, cp.ndarray)
    assert isinstance(dm2, cp.ndarray)
    assert dm1.shape == (2, 2)
    assert dm2.shape == (2, 2, 2, 2)
    assert bool(cp.isfinite(dm1).all())
    assert bool(cp.isfinite(dm2).all())


@pytest.mark.cuda
def test_cuda_casci_df_can_return_cupy_ci():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf.casci import run_casci
    from asuka.solver import GUGAFCISolver

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    solver = GUGAFCISolver(twos=0, nroots=1)
    cas = run_casci(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        matvec_backend="cuda",
        fcisolver=solver,
        return_cupy=True,
    )
    assert isinstance(cas.ci, cp.ndarray)
    assert bool(cp.isfinite(cas.ci).all())


@pytest.mark.cuda
def test_cuda_make_hdiag_df_returns_cupy():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf.casci import run_casci
    from asuka.solver import GUGAFCISolver

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert bool(getattr(scf_out.scf, "converged", False))

    solver = GUGAFCISolver(twos=0, nroots=1)
    cas = run_casci(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        matvec_backend="cuda",
        fcisolver=solver,
    )
    hdiag = solver.make_hdiag(cas.h1eff, cas.eri, 2, 2, return_cupy=True)
    assert isinstance(hdiag, cp.ndarray)
    assert hdiag.ndim == 1
    assert int(hdiag.size) > 0
    assert bool(cp.isfinite(hdiag).all())
