import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def test_rhf_thc_rdvr_smoke_matches_df_reasonably():
    cp = pytest.importorskip("cupy")

    # Require the orbitals CUDA extension (AO eval).
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # Require cuERI CUDA backend for DF warmup/reference.
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_df, run_rhf_thc
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
    assert bool(df.scf.converged)

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
    assert bool(thc.scf.converged)
    assert np.isfinite(float(thc.scf.e_tot))

    err = float(abs(float(thc.scf.e_tot) - float(df.scf.e_tot)))
    assert err < 1e-1

    assert thc.thc_factors is not None
    X = thc.thc_factors.X
    Z = thc.thc_factors.Z
    assert isinstance(X, cp.ndarray) and isinstance(Z, cp.ndarray)
    assert X.ndim == 2 and Z.ndim == 2
