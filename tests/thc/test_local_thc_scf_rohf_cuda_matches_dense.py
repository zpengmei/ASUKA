import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def test_rohf_local_thc_matches_dense_hf_reasonably():
    cp = pytest.importorskip("cupy")

    # Require the orbitals CUDA extension (grid build + AO eval).
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # Require cuERI CUDA backend (dense HF + DF warmup/reference).
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rohf_dense, run_rohf_thc
    from asuka.density import DeviceGridSpec

    mol = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0))],
        unit="Bohr",
        charge=0,
        spin=1,
        basis="sto-3g",
        cart=False,
    )

    dense = run_rohf_dense(
        mol,
        basis="sto-3g",
        backend="cuda",
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(dense.scf.converged)

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)
    lthc = run_rohf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_mode="local",
        thc_local_config={"block_max_ao": 10**9, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0},
        thc_grid_spec=grid,
        thc_grid_kind="becke",
        thc_npt=2000,
        thc_solve_method="inv_metric",
        use_density_difference=True,
        df_warmup_cycles=5,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(lthc.scf.converged)
    assert np.isfinite(float(lthc.scf.e_tot))

    err = float(abs(float(lthc.scf.e_tot) - float(dense.scf.e_tot)))
    assert err < 2e-3

    assert lthc.thc_factors is not None
    assert lthc.df_L is not None
    assert isinstance(lthc.df_L, cp.ndarray)
