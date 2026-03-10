from __future__ import annotations

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


def _build_local_thc_ref():
    from asuka.density import DeviceGridSpec
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
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
        thc_solve_method="fit_metric_gram",
        use_density_difference=True,
        df_warmup_cycles=1,
        max_cycle=20,
        conv_tol=1e-9,
        conv_tol_dm=1e-7,
        diis=True,
    )
    mc = run_casscf(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=True,
        max_cycle_macro=5,
        tol=1e-7,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    return scf_out, mc


def test_local_thc_mrci_energy_smoke():
    _skip_if_cuda_unavailable()

    from asuka.mrci.driver_asuka import mrci_from_ref

    scf_out, mc = _build_local_thc_ref()
    res = mrci_from_ref(
        mc,
        scf_out=scf_out,
        method="mrcisd",
        state=0,
        integrals_backend="thc",
        hop_backend="cuda",
        max_cycle=20,
        tol=1e-8,
    )
    assert np.isfinite(float(res.e_tot))


def test_local_thc_ic_mrci_rdm_analytic_grad_smoke():
    _skip_if_cuda_unavailable()

    from asuka.mrci.grad_driver import mrci_grad_from_ref

    scf_out, mc = _build_local_thc_ref()
    res = mrci_grad_from_ref(
        mc,
        scf_out=scf_out,
        method="ic_mrcisd",
        backend="analytic",
        mrci_backend="rdm",
        state=0,
        integrals_backend="thc",
        hop_backend="cuda",
        max_cycle=20,
        tol=1e-8,
    )
    grad = np.asarray(res.grad_tot, dtype=np.float64)
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))
