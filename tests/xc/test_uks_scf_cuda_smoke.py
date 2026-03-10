import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def _require_cuda_stacks():
    pytest.importorskip("cupy")

    # AO-on-grid eval + THC factor build require the orbitals CUDA extension.
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # DF factor build / DF warmup requires cuERI CUDA backend.
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")


def test_uks_df_smoke():
    _require_cuda_stacks()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_hf

    mol = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0))],
        unit="Bohr",
        charge=0,
        spin=1,
        basis="sto-3g",
        cart=False,
    )

    out = run_hf(
        mol,
        method="uks",
        backend="cuda",
        df=True,
        functional="m06-l",
        max_cycle=40,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
        grid_radial_n=16,
        grid_angular_n=74,
        xc_batch_size=20000,
    )
    assert bool(out.scf.converged)
    assert str(out.scf.method).upper() == "UKS"
    assert np.isfinite(float(out.scf.e_tot))


def test_uks_thc_global_smoke():
    _require_cuda_stacks()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_hf
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0))],
        unit="Bohr",
        charge=0,
        spin=1,
        basis="sto-3g",
        cart=False,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    out = run_hf(
        mol,
        method="uks",
        backend="cuda",
        two_e_backend="thc",
        functional="m06-l",
        thc_mode="global",
        thc_grid_spec=grid,
        thc_grid_kind="becke",
        thc_npt=1000,
        thc_solve_method="inv_metric",
        use_density_difference=True,
        df_warmup_cycles=3,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
        grid_radial_n=16,
        grid_angular_n=74,
        xc_batch_size=20000,
    )
    assert bool(out.scf.converged)
    assert str(out.scf.method).upper() == "UKS-THC"
    assert np.isfinite(float(out.scf.e_tot))


def test_uks_thc_local_smoke():
    _require_cuda_stacks()

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_hf
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0))],
        unit="Bohr",
        charge=0,
        spin=1,
        basis="sto-3g",
        cart=False,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    out = run_hf(
        mol,
        method="uks",
        backend="cuda",
        two_e_backend="thc",
        functional="m06-l",
        thc_mode="local",
        thc_grid_spec=grid,
        thc_grid_kind="becke",
        thc_npt=1000,
        thc_solve_method="inv_metric",
        use_density_difference=True,
        df_warmup_cycles=3,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
        grid_radial_n=16,
        grid_angular_n=74,
        xc_batch_size=20000,
    )
    assert bool(out.scf.converged)
    assert str(out.scf.method).upper() == "UKS-LTHC"
    assert np.isfinite(float(out.scf.e_tot))
