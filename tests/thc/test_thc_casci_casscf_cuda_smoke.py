import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def _skip_if_cuda_unavailable():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    # Orbitals CUDA extension is required for THC factor construction (AO eval).
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # cuERI CUDA backend is required for THC DF warmup/reference.
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    # GUGA CUDA backend is required for matvec_backend='cuda_eri_mat'.
    try:
        from asuka.cuda.cuda_backend import has_cuda_ext as has_guga_cuda_ext

        if not bool(has_guga_cuda_ext()):
            pytest.skip("GUGA CUDA extension not available")
    except Exception:
        pytest.skip("GUGA CUDA extension not available")


def test_thc_casci_smoke_matches_df_reasonably():
    _skip_if_cuda_unavailable()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_df, run_rhf_thc
    from asuka.mcscf.casci import run_casci
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -0.757, 0.587)),
            ("H", (0.0, 0.757, 0.587)),
        ],
        unit="Angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )

    df = run_rhf_df(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(df.converged)

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    thc = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_grid_spec=grid,
        thc_grid_kind="rdvr",
        thc_dvr_basis="aux",
        thc_npt=2000,
        thc_solve_method="fit_metric_qr",
        use_density_difference=True,
        df_warmup_cycles=2,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(thc.converged)
    assert thc.thc_factors is not None

    # CAS(2,2) for water (10 e-): ncore = (10-2)/2 = 4
    cas_df = run_casci(df, ncore=4, ncas=2, nelecas=2, backend="cuda", df=True, matvec_backend="cuda_eri_mat")
    cas_thc = run_casci(thc, ncore=4, ncas=2, nelecas=2, backend="cuda", df=True, matvec_backend="cuda_eri_mat")

    assert np.isfinite(float(cas_df.e_tot))
    assert np.isfinite(float(cas_thc.e_tot))

    # THC is an approximation to DF; on this tiny system it should be reasonably close.
    err = float(abs(float(cas_thc.e_tot) - float(cas_df.e_tot)))
    assert err < 2e-1


def test_thc_casscf_smoke_runs():
    _skip_if_cuda_unavailable()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -0.757, 0.587)),
            ("H", (0.0, 0.757, 0.587)),
        ],
        unit="Angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    scf_out = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_grid_spec=grid,
        thc_grid_kind="rdvr",
        thc_dvr_basis="aux",
        thc_npt=2000,
        thc_solve_method="fit_metric_qr",
        use_density_difference=True,
        df_warmup_cycles=2,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(scf_out.converged)

    mc = run_casscf(
        scf_out,
        ncore=4,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=20,
        tol=1e-8,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    assert np.isfinite(float(mc.e_tot))
    assert np.isfinite(float(mc.grad_norm))


def test_local_thc_casci_smoke_runs():
    """Smoke test: local-THC SCF -> CASCI runs on CUDA (multi-block)."""
    _skip_if_cuda_unavailable()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf.casci import run_casci
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 3.015)),
        ],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )

    grid = DeviceGridSpec(radial_n=12, angular_n=50, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    scf_out = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_mode="local",
        # Force at least 2 blocks for LiH(STO-3G): Li has 5 AOs, H has 1 AO.
        thc_local_config={"block_max_ao": 5, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0},
        thc_grid_spec=grid,
        thc_grid_kind="rdvr",
        thc_dvr_basis="aux",
        thc_npt=1200,
        thc_solve_method="fit_metric_qr",
        use_density_difference=True,
        df_warmup_cycles=2,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(scf_out.converged)

    # LiH (4 e-): CAS(2,2) -> ncore = (4-2)/2 = 1
    cas = run_casci(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        matvec_backend="cuda_eri_mat",
    )
    assert np.isfinite(float(cas.e_tot))


def test_local_thc_casscf_smoke_runs():
    """Smoke test: local-THC SCF -> CASSCF runs on CUDA (multi-block)."""
    _skip_if_cuda_unavailable()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 3.015)),
        ],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )

    grid = DeviceGridSpec(radial_n=12, angular_n=50, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    scf_out = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_mode="local",
        thc_local_config={"block_max_ao": 5, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0},
        thc_grid_spec=grid,
        thc_grid_kind="rdvr",
        thc_dvr_basis="aux",
        thc_npt=1200,
        thc_solve_method="fit_metric_qr",
        use_density_difference=True,
        df_warmup_cycles=2,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(scf_out.converged)

    mc = run_casscf(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=10,
        tol=1e-8,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    assert np.isfinite(float(mc.e_tot))
    assert np.isfinite(float(mc.grad_norm))
