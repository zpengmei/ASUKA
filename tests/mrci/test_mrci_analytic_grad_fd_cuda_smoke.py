from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.cuda
_REF_CACHE: dict[str, tuple[object, object]] | None = None


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
        pytest.skip("set ASUKA_RUN_SLOW_TESTS=1 to run MRCI analytic-vs-FD CUDA checks")


def _build_lih_refs():
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE

    from asuka.density import DeviceGridSpec
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_df, run_rhf_thc
    from asuka.mcscf import run_casscf

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

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    scf_df = run_rhf_df(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        max_cycle=60,
        conv_tol=1e-12,
        conv_tol_dm=1e-10,
    )
    mc_df = run_casscf(
        scf_df,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=200,
        tol=1e-6,
        orbital_optimizer="1step",
        matvec_backend="cuda_eri_mat",
    )
    if not bool(scf_df.converged) or not bool(mc_df.converged):
        pytest.skip("DF LiH validation reference did not converge")

    thc_common = dict(
        mol=mol,
        basis="sto-3g",
        auxbasis="autoaux",
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

    scf_thc_global = run_rhf_thc(thc_mode="global", **thc_common)
    mc_thc_global = run_casscf(
        scf_thc_global,
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
    if not bool(scf_thc_global.converged) or not bool(mc_thc_global.converged):
        pytest.skip("global THC LiH validation reference did not converge")

    scf_thc_local = run_rhf_thc(
        thc_mode="local",
        thc_local_config={"block_max_ao": 5, "aux_schwarz_thr": 0.0, "sec_overlap_thr": 0.0, "no_point_downselect": True},
        **thc_common,
    )
    mc_thc_local = run_casscf(
        scf_thc_local,
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
    if not bool(scf_thc_local.converged) or not bool(mc_thc_local.converged):
        pytest.skip("local THC LiH validation reference did not converge")

    _REF_CACHE = {
        "df": (scf_df, mc_df),
        "thc_global": (scf_thc_global, mc_thc_global),
        "thc_local": (scf_thc_local, mc_thc_local),
    }
    return _REF_CACHE


def _gradient_pair(scf_out, mc, *, method: str, integrals_backend: str, mrci_backend: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    from asuka.mrci.grad_driver import mrci_grad_from_ref

    kwargs = dict(
        method=str(method),
        state=0,
        integrals_backend=str(integrals_backend),
        hop_backend="cuda",
    )
    if mrci_backend is not None:
        kwargs["mrci_backend"] = str(mrci_backend)

    grad_analytic = mrci_grad_from_ref(
        mc,
        scf_out=scf_out,
        backend="analytic",
        **kwargs,
    )
    grad_fd = mrci_grad_from_ref(
        mc,
        scf_out=scf_out,
        backend="fd",
        fd_step_bohr=5.0e-4,
        **kwargs,
    )
    return (
        np.asarray(grad_analytic.grad_tot, dtype=np.float64),
        np.asarray(grad_fd.grad_tot, dtype=np.float64),
    )

@pytest.mark.parametrize(
    ("ref_key", "method", "integrals_backend", "mrci_backend", "tol", "label"),
    [
        ("df", "mrcisd", "df_B", None, 3.0e-3, "DF uncontracted"),
        ("df", "ic_mrcisd", "df_B", "semi_direct", 3.0e-3, "DF contracted semi-direct"),
        ("df", "ic_mrcisd", "df_B", "rdm", 3.0e-3, "DF contracted rdm"),
        ("thc_global", "mrcisd", "thc", None, 1.0e-3, "THC global uncontracted"),
        ("thc_global", "ic_mrcisd", "thc", "semi_direct", 1.0e-3, "THC global contracted semi-direct"),
        ("thc_global", "ic_mrcisd", "thc", "rdm", 1.0e-3, "THC global contracted rdm"),
        ("thc_local", "mrcisd", "thc", None, 1.0e-3, "THC local uncontracted"),
        ("thc_local", "ic_mrcisd", "thc", "semi_direct", 1.0e-3, "THC local contracted semi-direct"),
        ("thc_local", "ic_mrcisd", "thc", "rdm", 1.0e-3, "THC local contracted rdm"),
    ],
)
def test_mrci_analytic_fd_full_gradient_matches_tiny_fd(ref_key, method, integrals_backend, mrci_backend, tol, label):
    _require_slow_tests()
    _skip_if_cuda_unavailable()

    refs = _build_lih_refs()
    scf_out, mc = refs[ref_key]
    grad_analytic, grad_fd = _gradient_pair(
        scf_out,
        mc,
        method=method,
        integrals_backend=integrals_backend,
        mrci_backend=mrci_backend,
    )
    err = float(np.max(np.abs(np.asarray(grad_analytic - grad_fd, dtype=np.float64))))
    assert err < float(tol), f"{label}: max_abs_err={err:.12f} tol={float(tol):.12f}"
