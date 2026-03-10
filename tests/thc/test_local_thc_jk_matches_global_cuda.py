import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def test_local_thc_jk_single_block_matches_global_thc_jk():
    cp = pytest.importorskip("cupy")

    # Require the orbitals CUDA extension (grid build + AO eval).
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # Require cuERI CUDA backend (aux metric build).
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.hf.local_thc_factors import LocalTHCFactors
    from asuka.hf.local_thc_jk import local_thc_JK
    from asuka.hf.thc_factors import THCFactors
    from asuka.hf.thc_jk import THCJKWork, thc_JK
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
        cart=False,  # spherical AOs
    )

    grid = DeviceGridSpec(radial_n=10, angular_n=50, rmax=8.0, becke_n=3, prune_tol=1e-14, threads=256)
    # Use all grid points by requesting npt >= full grid size.
    thc_npt = 10**9

    g = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_mode="global",
        thc_grid_spec=grid,
        thc_grid_kind="becke",
        thc_npt=int(thc_npt),
        thc_solve_method="inv_metric",
        use_density_difference=False,
        max_cycle=1,
        diis=False,
        init_fock_cycles=0,
    )
    assert g.thc_factors is not None
    assert isinstance(g.thc_factors, THCFactors)

    l = run_rhf_thc(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        thc_mode="local",
        thc_local_config={"block_max_ao": 10**9, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0},
        thc_grid_spec=grid,
        thc_grid_kind="becke",
        thc_npt=int(thc_npt),
        thc_solve_method="inv_metric",
        use_density_difference=False,
        max_cycle=1,
        diis=False,
        init_fock_cycles=0,
    )
    assert l.thc_factors is not None
    assert isinstance(l.thc_factors, LocalTHCFactors)

    Xg = g.thc_factors.X
    Zg = g.thc_factors.Z
    assert isinstance(Xg, cp.ndarray) and isinstance(Zg, cp.ndarray)
    nao = int(Xg.shape[1])

    rng = np.random.default_rng(0)
    D = rng.normal(size=(nao, nao))
    D = 0.5 * (D + D.T)
    D = cp.asarray(D, dtype=cp.float64)

    Jg, Kg = thc_JK(D, Xg, Zg, work=THCJKWork(q_block=64))
    Jl, Kl = local_thc_JK(D, l.thc_factors, q_block=64)

    cp.testing.assert_allclose(Jl, Jg, rtol=1e-10, atol=1e-10)
    cp.testing.assert_allclose(Kl, Kg, rtol=1e-10, atol=1e-10)
