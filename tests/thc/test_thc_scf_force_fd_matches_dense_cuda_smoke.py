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

    # cuERI CUDA backend is required for dense HF and aux metric build.
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")


def _mol_from_coords(coords_bohr: np.ndarray):
    from asuka.frontend.molecule import Molecule

    coords_bohr = np.asarray(coords_bohr, dtype=np.float64).reshape((3, 3))
    atoms = [
        ("O", coords_bohr[0]),
        ("H", coords_bohr[1]),
        ("H", coords_bohr[2]),
    ]
    return Molecule.from_atoms(atoms, unit="Bohr", charge=0, spin=0, basis="sto-3g", cart=False)


def _fd_grad_component(energy_fn, coords0: np.ndarray, *, ia: int, xyz: int, h: float) -> float:
    coords0 = np.asarray(coords0, dtype=np.float64).reshape((-1, 3))
    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[int(ia), int(xyz)] += float(h)
    coords_m[int(ia), int(xyz)] -= float(h)
    e_p = float(energy_fn(coords_p))
    e_m = float(energy_fn(coords_m))
    return float((e_p - e_m) / (2.0 * float(h)))


def test_thc_scf_fd_force_component_matches_dense_reasonably():
    """Smoke check: FD force from THC-SCF energy is close to dense HF FD force."""
    _skip_if_cuda_unavailable()

    from asuka.frontend.scf import run_rhf_dense, run_rhf_thc
    from asuka.density import DeviceGridSpec

    # Water geometry (Angstrom) converted to Bohr inside Molecule.
    mol0 = _mol_from_coords(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.757 * 1.8897259886, 0.587 * 1.8897259886],
                [0.0, 0.757 * 1.8897259886, 0.587 * 1.8897259886],
            ],
            dtype=np.float64,
        )
    )
    coords0 = mol0.coords_bohr

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=1e-14, threads=256)

    def e_dense(coords_bohr: np.ndarray) -> float:
        mol = _mol_from_coords(coords_bohr)
        out = run_rhf_dense(
            mol,
            basis="sto-3g",
            backend="cuda",
            max_cycle=30,
            conv_tol=1e-10,
            conv_tol_dm=1e-8,
            diis=True,
        )
        assert bool(out.scf.converged)
        return float(out.scf.e_tot)

    def e_thc(coords_bohr: np.ndarray, *, thc_mode: str) -> float:
        mol = _mol_from_coords(coords_bohr)
        out = run_rhf_thc(
            mol,
            basis="sto-3g",
            auxbasis="autoaux",
            thc_mode=str(thc_mode),
            thc_local_config={"block_max_ao": 5, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0}
            if str(thc_mode).strip().lower() == "local"
            else None,
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
        assert bool(out.scf.converged)
        return float(out.scf.e_tot)

    # One component (H1 z) to keep runtime bounded.
    ia = 1
    xyz = 2
    h = 1e-3
    g_dense = _fd_grad_component(e_dense, coords0, ia=ia, xyz=xyz, h=h)
    g_thc_global = _fd_grad_component(lambda c: e_thc(c, thc_mode="global"), coords0, ia=ia, xyz=xyz, h=h)
    g_thc_local = _fd_grad_component(lambda c: e_thc(c, thc_mode="local"), coords0, ia=ia, xyz=xyz, h=h)

    assert np.isfinite(g_dense)
    assert np.isfinite(g_thc_global)
    assert np.isfinite(g_thc_local)

    # THC is approximate; require only a loose agreement on this FD force component.
    assert abs(float(g_thc_global) - float(g_dense)) < 1e-3
    assert abs(float(g_thc_local) - float(g_dense)) < 1e-3

