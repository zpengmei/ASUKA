from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def _skip_if_cuda_unavailable():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            pytest.skip("CuPy is present but a CUDA device is unavailable")
        _ = int(cp.cuda.runtime.getDevice())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available")

    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    try:
        from asuka.cuda.cuda_backend import has_cuda_ext as has_guga_cuda_ext

        if not bool(has_guga_cuda_ext()):
            pytest.skip("GUGA CUDA extension not available")
    except Exception:
        pytest.skip("GUGA CUDA extension not available")


def _require_slow_tests():
    if str(os.environ.get("ASUKA_RUN_SLOW_TESTS", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set ASUKA_RUN_SLOW_TESTS=1 to run THC per-root relaxed FD checks")


def _build_global_h2o_ref():
    from asuka.density import DeviceGridSpec
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.757 * 1.8897259886, 0.587 * 1.8897259886],
            [0.0, 0.757 * 1.8897259886, 0.587 * 1.8897259886],
        ],
        dtype=np.float64,
    )
    mol = Molecule.from_atoms(
        [("O", coords0[0]), ("H", coords0[1]), ("H", coords0[2])],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )
    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol_now = Molecule.from_atoms(
            [("O", coords_bohr[0]), ("H", coords_bohr[1]), ("H", coords_bohr[2])],
            unit="Bohr",
            charge=0,
            spin=0,
            basis="sto-3g",
            cart=False,
        )
        out = run_rhf_thc(
            mol_now,
            basis="sto-3g",
            auxbasis="autoaux",
            thc_mode="global",
            thc_grid_spec=grid,
            thc_grid_kind="rdvr",
            thc_dvr_basis="aux",
            thc_npt=None,
            thc_solve_method="fit_metric_gram",
            use_density_difference=False,
            df_warmup_cycles=0,
            max_cycle=50,
            conv_tol=1e-12,
            conv_tol_dm=1e-10,
            diis=True,
            init_guess="core",
            mo_coeff0=mo_coeff0,
        )
        assert bool(out.converged)
        return out

    def run_mc(scf_out, *, guess_mc=None):
        mc = run_casscf(
            scf_out,
            ncore=4,
            ncas=2,
            nelecas=2,
            backend="cuda",
            df=True,
            max_cycle_macro=500,
            tol=1e-7,
            orbital_optimizer="lbfgs",
            matvec_backend="cuda_eri_mat",
            nroots=2,
            root_weights=[0.5, 0.5],
            guess=guess_mc,
        )
        assert bool(mc.converged)
        return mc

    return coords0, run_scf, run_mc


def _build_local_lih_ref():
    from asuka.density import DeviceGridSpec
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.015],
        ],
        dtype=np.float64,
    )
    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol_now = Molecule.from_atoms(
            [("Li", coords_bohr[0]), ("H", coords_bohr[1])],
            unit="Bohr",
            charge=0,
            spin=0,
            basis="sto-3g",
            cart=False,
        )
        out = run_rhf_thc(
            mol_now,
            basis="sto-3g",
            auxbasis="autoaux",
            thc_mode="local",
            thc_local_config={
                "block_max_ao": 5,
                "aux_schwarz_thr": 0.0,
                "sec_overlap_thr": 0.0,
                "no_point_downselect": True,
            },
            thc_grid_spec=grid,
            thc_grid_kind="rdvr",
            thc_dvr_basis="aux",
            thc_npt=None,
            thc_solve_method="fit_metric_gram",
            use_density_difference=False,
            df_warmup_cycles=0,
            max_cycle=50,
            conv_tol=1e-12,
            conv_tol_dm=1e-10,
            diis=True,
            init_guess="core",
            mo_coeff0=mo_coeff0,
        )
        assert bool(out.converged)
        return out

    def run_mc(scf_out, *, guess_mc=None):
        mc = run_casscf(
            scf_out,
            ncore=1,
            ncas=2,
            nelecas=2,
            backend="cuda",
            df=True,
            max_cycle_macro=200,
            tol=1e-6,
            orbital_optimizer="lbfgs",
            matvec_backend="cuda_eri_mat",
            nroots=2,
            root_weights=[0.5, 0.5],
            guess=guess_mc,
        )
        assert bool(mc.converged)
        return mc

    return coords0, run_scf, run_mc


def _root_follow_fd_errors(coords0, run_scf, run_mc):
    from asuka.mcscf.nuc_grad_thc import casscf_nuc_grad_thc_per_root
    from asuka.mcscf.state_average import ci_as_list
    from asuka.mrci.common import assign_roots_by_overlap

    scf0 = run_scf(coords0)
    mc0 = run_mc(scf0)
    res = casscf_nuc_grad_thc_per_root(scf0, mc0)

    ia, xyz = 1, 2
    h = 1e-3
    ci0 = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mc0.ci, nroots=2)]

    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[ia, xyz] += h
    coords_m[ia, xyz] -= h

    scf_p = run_scf(coords_p, mo_coeff0=getattr(scf0, "mo_coeff", None))
    mc_p = run_mc(scf_p, guess_mc=mc0)
    scf_m = run_scf(coords_m, mo_coeff0=getattr(scf0, "mo_coeff", None))
    mc_m = run_mc(scf_m, guess_mc=mc0)

    ci_p = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mc_p.ci, nroots=2)]
    ci_m = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mc_m.ci, nroots=2)]

    ov_p = np.zeros((2, 2), dtype=np.float64)
    ov_m = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            ov_p[i, j] = abs(float(np.dot(ci0[i], ci_p[j]))) ** 2
            ov_m[i, j] = abs(float(np.dot(ci0[i], ci_m[j]))) ** 2

    map_p = assign_roots_by_overlap(ov_p)
    map_m = assign_roots_by_overlap(ov_m)
    e_p = np.asarray(mc_p.e_roots, dtype=np.float64)
    e_m = np.asarray(mc_m.e_roots, dtype=np.float64)

    errs = []
    for k in range(2):
        fd = float((e_p[int(map_p[k])] - e_m[int(map_m[k])]) / (2.0 * h))
        analytic = float(np.asarray(res.grads, dtype=np.float64)[k, ia, xyz])
        errs.append(abs(analytic - fd))
    return errs


@pytest.mark.parametrize(
    ("builder", "tol"),
    [
        (_build_global_h2o_ref, 5.0e-4),
        (_build_local_lih_ref, 7.0e-4),
    ],
)
def test_thc_casscf_per_root_relaxed_fd_component(builder, tol):
    _require_slow_tests()
    _skip_if_cuda_unavailable()

    coords0, run_scf, run_mc = builder()
    errs = _root_follow_fd_errors(coords0, run_scf, run_mc)
    assert max(errs) < float(tol), f"max root error {max(errs):.6e} exceeds tol {float(tol):.6e}"
