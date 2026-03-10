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
        pytest.skip("set ASUKA_RUN_SLOW_TESTS=1 to run THC Newton smoke checks")


def _asnumpy_f64(a):
    try:
        import cupy as cp

        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


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
    scf = run_rhf_thc(
        mol,
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
    )
    mc = run_casscf(
        scf,
        ncore=4,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=200,
        tol=1e-6,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    if not bool(scf.converged) or not bool(mc.converged):
        pytest.skip("global THC H2O reference did not converge")
    return scf, mc


def _build_local_lih_ref():
    from asuka.density import DeviceGridSpec
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf

    mol = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 3.015))],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )
    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)
    scf = run_rhf_thc(
        mol,
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
    )
    mc = run_casscf(
        scf,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=200,
        tol=1e-6,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    if not bool(scf.converged) or not bool(mc.converged):
        pytest.skip("local THC LiH reference did not converge")
    return scf, mc


@pytest.mark.parametrize(
    "builder",
    [
        _build_global_h2o_ref,
        _build_local_lih_ref,
    ],
)
def test_thc_newton_eris_matches_active_space_casci_slice(builder):
    _require_slow_tests()
    _skip_if_cuda_unavailable()

    from asuka.mcscf.newton_thc import THCNewtonCASSCFAdapter
    from asuka.solver import GUGAFCISolver

    scf, mc = builder()
    fcisolver = GUGAFCISolver(twos=int(getattr(scf.mol, "spin", 0)), nroots=int(mc.nroots))
    adapter = THCNewtonCASSCFAdapter(
        scf_out=scf,
        hcore_ao=scf.int1e.hcore,
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
        nelecas=mc.nelecas,
        mo_coeff=mc.mo_coeff,
        fcisolver=fcisolver,
    )
    eris = adapter.ao2mo(mc.mo_coeff)

    ppaa_act = _asnumpy_f64(eris.ppaa[int(mc.ncore) : int(mc.ncore + mc.ncas), int(mc.ncore) : int(mc.ncore + mc.ncas)])
    eri_ref = getattr(getattr(mc, "casci"), "eri", None)
    if eri_ref is None or getattr(eri_ref, "eri_mat", None) is None:
        pytest.skip("CASCI THC result does not expose eri_mat for validation")
    eri_ref_4d = _asnumpy_f64(eri_ref.eri_mat).reshape(int(mc.ncas), int(mc.ncas), int(mc.ncas), int(mc.ncas))
    np.testing.assert_allclose(ppaa_act, eri_ref_4d, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize(
    "builder",
    [
        _build_global_h2o_ref,
        _build_local_lih_ref,
    ],
)
def test_thc_newton_gradient_vector_is_small_at_converged_reference(builder):
    _require_slow_tests()
    _skip_if_cuda_unavailable()

    from asuka.mcscf.newton_casscf import compute_mcscf_gradient_vector
    from asuka.mcscf.newton_thc import THCNewtonCASSCFAdapter
    from asuka.solver import GUGAFCISolver

    scf, mc = builder()
    fcisolver = GUGAFCISolver(twos=int(getattr(scf.mol, "spin", 0)), nroots=int(mc.nroots))
    adapter = THCNewtonCASSCFAdapter(
        scf_out=scf,
        hcore_ao=scf.int1e.hcore,
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
        nelecas=mc.nelecas,
        mo_coeff=mc.mo_coeff,
        fcisolver=fcisolver,
    )
    eris = adapter.ao2mo(mc.mo_coeff)
    g = compute_mcscf_gradient_vector(adapter, mc.mo_coeff, mc.ci, eris)
    gnorm = float(np.linalg.norm(_asnumpy_f64(g)))
    assert gnorm < 3.0e-3, f"THC Newton gradient vector too large at converged reference: {gnorm:.6e}"


@pytest.mark.parametrize(
    "builder",
    [
        _build_global_h2o_ref,
        _build_local_lih_ref,
    ],
)
def test_thc_newton_zero_rhs_zvector_returns_zero(builder):
    _require_slow_tests()
    _skip_if_cuda_unavailable()

    from asuka.mcscf.newton_thc import THCNewtonCASSCFAdapter
    from asuka.mcscf.zvector import build_mcscf_hessian_operator, solve_mcscf_zvector
    from asuka.solver import GUGAFCISolver

    scf, mc = builder()
    fcisolver = GUGAFCISolver(twos=int(getattr(scf.mol, "spin", 0)), nroots=int(mc.nroots))
    adapter = THCNewtonCASSCFAdapter(
        scf_out=scf,
        hcore_ao=scf.int1e.hcore,
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
        nelecas=mc.nelecas,
        mo_coeff=mc.mo_coeff,
        fcisolver=fcisolver,
    )
    eris = adapter.ao2mo(mc.mo_coeff)
    hess = build_mcscf_hessian_operator(
        adapter,
        mo_coeff=mc.mo_coeff,
        ci=mc.ci,
        eris=eris,
        use_newton_hessian=True,
    )
    z = solve_mcscf_zvector(
        adapter,
        rhs_orb=np.zeros((int(hess.n_orb),), dtype=np.float64),
        rhs_ci=None,
        hessian_op=hess,
        tol=1.0e-10,
        maxiter=20,
    )
    znorm = float(np.linalg.norm(np.asarray(z.z_packed, dtype=np.float64)))
    assert znorm < 1.0e-12
