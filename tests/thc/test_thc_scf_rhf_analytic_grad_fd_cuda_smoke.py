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

    # Orbitals CUDA extension is required for THC factor construction (AO eval + gradients).
    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    # cuERI CUDA backend is required for aux metric + metric derivative contraction.
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


def test_thc_rhf_analytic_grad_matches_fd_component_reasonably():
    _skip_if_cuda_unavailable()

    from asuka.frontend.scf import run_rhf_thc
    from asuka.density import DeviceGridSpec
    from asuka.hf.nuc_grad_thc import rhf_nuc_grad_thc

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.757 * 1.8897259886, 0.587 * 1.8897259886],
            [0.0, 0.757 * 1.8897259886, 0.587 * 1.8897259886],
        ],
        dtype=np.float64,
    )

    # Use the RDVR grid for stable fit-metric THC factors (required for meaningful energies).
    # Avoid point downselect so the analytic gradient path is well-defined.
    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol = _mol_from_coords(coords_bohr)
        out = run_rhf_thc(
            mol,
            basis="sto-3g",
            auxbasis="autoaux",
            thc_mode="global",
            thc_grid_spec=grid,
            thc_grid_kind="rdvr",
            thc_dvr_basis="aux",
            thc_npt=None,
            thc_solve_method="fit_metric_gram",
            # The analytic gradient currently supports only the pure THC energy.
            use_density_difference=False,
            df_warmup_cycles=0,
            max_cycle=50,
            conv_tol=1e-12,
            conv_tol_dm=1e-10,
            diis=True,
            init_guess="core",
            mo_coeff0=mo_coeff0,
        )
        assert bool(out.scf.converged)
        return out

    out0 = run(coords0)
    g = np.asarray(rhf_nuc_grad_thc(out0), dtype=np.float64)
    assert g.shape == (3, 3)

    # One force component (H1 z) to keep runtime bounded.
    ia, xyz = 1, 2
    h = 1e-3
    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[ia, xyz] += h
    coords_m[ia, xyz] -= h

    # Warm start the displaced SCFs with the converged orbitals to reduce SCF noise.
    mo0 = getattr(out0.scf, "mo_coeff", None)
    out_p = run(coords_p, mo_coeff0=mo0)
    out_m = run(coords_m, mo_coeff0=mo0)

    e_p = float(out_p.scf.e_tot)
    e_m = float(out_m.scf.e_tot)
    fd = (e_p - e_m) / (2.0 * h)

    assert np.isfinite(fd)
    assert np.isfinite(g[ia, xyz])

    # Tight-ish tolerance: with tight SCF convergence, this is usually ~1e-5.
    assert abs(float(fd) - float(g[ia, xyz])) < 1e-4
