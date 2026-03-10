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

    # Orbitals CUDA extension is required for THC factor construction + VJP kernels.
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

    # GUGA CUDA backend is required for CASSCF matvec.
    try:
        from asuka.cuda.cuda_backend import has_cuda_ext as has_guga_cuda_ext

        if not bool(has_guga_cuda_ext()):
            pytest.skip("GUGA CUDA extension not available")
    except Exception:
        pytest.skip("GUGA CUDA extension not available")


def _mol_h2o_from_coords(coords_bohr: np.ndarray):
    from asuka.frontend.molecule import Molecule

    coords_bohr = np.asarray(coords_bohr, dtype=np.float64).reshape((3, 3))
    atoms = [
        ("O", coords_bohr[0]),
        ("H", coords_bohr[1]),
        ("H", coords_bohr[2]),
    ]
    return Molecule.from_atoms(atoms, unit="Bohr", charge=0, spin=0, basis="sto-3g", cart=False)


def _mol_lih_from_coords(coords_bohr: np.ndarray):
    from asuka.frontend.molecule import Molecule

    coords_bohr = np.asarray(coords_bohr, dtype=np.float64).reshape((2, 3))
    atoms = [
        ("Li", coords_bohr[0]),
        ("H", coords_bohr[1]),
    ]
    return Molecule.from_atoms(atoms, unit="Bohr", charge=0, spin=0, basis="sto-3g", cart=False)


def _fd_component(energy_fn, coords0: np.ndarray, *, ia: int, xyz: int, h: float) -> float:
    coords0 = np.asarray(coords0, dtype=np.float64).reshape((-1, 3))
    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[int(ia), int(xyz)] += float(h)
    coords_m[int(ia), int(xyz)] -= float(h)
    e_p = float(energy_fn(coords_p))
    e_m = float(energy_fn(coords_m))
    return float((e_p - e_m) / (2.0 * float(h)))


def _require_casscf_thc_ref_data(scf0, mc0):
    """Build reference CASSCF densities/RDMs on GPU.

    Used for a low-noise fixed-reference finite-difference validation of THC
    analytic gradients without re-optimizing displaced CASSCF wavefunctions.
    """

    import cupy as cp

    from asuka.mcscf.nuc_grad_thc import _dm2_sym_flat
    from asuka.mcscf.state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
    from asuka.solver import GUGAFCISolver

    ncore = int(getattr(mc0, "ncore"))
    ncas = int(getattr(mc0, "ncas"))
    nocc = int(ncore + ncas)

    nroots = int(getattr(mc0, "nroots", 1))
    weights = normalize_weights(getattr(mc0, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(mc0, "ci"), nroots=nroots)

    twos = int(getattr(getattr(scf0, "mol", None), "spin", 0))
    fcisolver = GUGAFCISolver(twos=twos, nroots=nroots)

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=ncas,
        nelecas=getattr(mc0, "nelecas"),
    )

    C = cp.asarray(getattr(mc0, "mo_coeff"), dtype=cp.float64)
    nao = int(C.shape[0])
    if ncore:
        C_core = C[:, :ncore]
        D_core = 2.0 * (C_core @ C_core.T)
    else:
        D_core = cp.zeros((nao, nao), dtype=cp.float64)

    C_act = C[:, ncore:nocc]
    dm1 = cp.asarray(dm1_act, dtype=cp.float64)
    dm1 = 0.5 * (dm1 + dm1.T)
    D_act = C_act @ dm1 @ C_act.T
    D_tot = D_core + D_act
    D_w = D_act + 0.5 * D_core

    dm2_flat_sym = _dm2_sym_flat(cp, dm2_act, ncas=ncas)

    return {
        "ncore": ncore,
        "ncas": ncas,
        "nocc": nocc,
        "dm1_act": dm1_act,
        "dm2_act": dm2_act,
        "dm2_flat_sym": dm2_flat_sym,
        "C_act": C_act,
        "D_core": D_core,
        "D_w": D_w,
        "D_tot": D_tot,
    }


def _fixed_reference_value_thc_casscf(scf_out, ref) -> float:
    import cupy as cp

    from asuka.cuda.active_space_thc import build_device_dfmo_integrals_local_thc, build_device_dfmo_integrals_thc
    from asuka.hf.thc_factors import THCFactors
    from asuka.hf.local_thc_factors import LocalTHCFactors

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("expected scf_out.thc_factors (THCFactors or LocalTHCFactors)")

    h_ao = cp.asarray(getattr(getattr(scf_out, "int1e"), "hcore"), dtype=cp.float64)

    D_tot = ref["D_tot"]
    D_core = ref["D_core"]
    D_w = ref["D_w"]
    C_act = ref["C_act"]
    dm2_flat_sym = ref["dm2_flat_sym"]

    e1 = float(cp.sum(D_tot * h_ao).item())

    if isinstance(thc, THCFactors):
        from asuka.hf.thc_jk import THCJKWork, thc_JK

        Jc, Kc = thc_JK(D_core, thc.X, thc.Z, work=THCJKWork(q_block=256))
        Vc = Jc - 0.5 * Kc
        e_mean = float(cp.sum(D_w * Vc).item())
        eri = build_device_dfmo_integrals_thc(thc, C_act, want_eri_mat=True)
    else:
        from asuka.hf.local_thc_jk import local_thc_JK

        Jc, Kc = local_thc_JK(D_core, thc, q_block=256)
        Vc = Jc - 0.5 * Kc
        e_mean = float(cp.sum(D_w * Vc).item())
        eri = build_device_dfmo_integrals_local_thc(thc, C_act, want_eri_mat=True)

    e_aa = float((0.5 * cp.sum(dm2_flat_sym * cp.asarray(eri.eri_mat, dtype=cp.float64))).item())

    e_nuc = float(scf_out.mol.energy_nuc())
    return e_nuc + e1 + e_mean + e_aa


def _run_relaxed_fd_component(coords0, *, ia: int, xyz: int, h: float, run_scf, run_mc) -> tuple[float, dict[int, tuple[float, bool, bool, int, int]]]:
    coords0 = np.asarray(coords0, dtype=np.float64).reshape((-1, 3))
    ref_scf = run_scf(coords0)
    ref_mc = run_mc(ref_scf)

    prev_mo = getattr(ref_scf.scf, "mo_coeff", None)
    prev_mc = ref_mc
    e_pm: dict[int, tuple[float, bool, bool, int, int]] = {}
    for sign in (1, -1):
        coords = coords0.copy()
        coords[int(ia), int(xyz)] += float(sign) * float(h)
        scf_d = run_scf(coords, mo_coeff0=prev_mo)
        mc_d = run_mc(scf_d, guess_mc=prev_mc)
        e_pm[int(sign)] = (
            float(mc_d.e_tot),
            bool(scf_d.converged),
            bool(mc_d.converged),
            int(scf_d.niter),
            int(getattr(mc_d, "niter", -1)),
        )
        prev_mo = getattr(scf_d, "mo_coeff", prev_mo)
        prev_mc = mc_d

    fd = (float(e_pm[1][0]) - float(e_pm[-1][0])) / (2.0 * float(h))
    return fd, e_pm


def test_thc_casscf_analytic_grad_matches_fixed_reference_fd_component_global():
    _skip_if_cuda_unavailable()

    from asuka.density import DeviceGridSpec
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad import casscf_nuc_grad

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.757 * 1.8897259886, 0.587 * 1.8897259886],
            [0.0, 0.757 * 1.8897259886, 0.587 * 1.8897259886],
        ],
        dtype=np.float64,
    )

    # RDVR grid with fit-metric THC factors (stable energies), no point downselect.
    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol = _mol_h2o_from_coords(coords_bohr)
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
            max_cycle_macro=200,
            tol=1e-6,
            orbital_optimizer="lbfgs",
            matvec_backend="cuda_eri_mat",
            guess=guess_mc,
        )
        assert bool(mc.converged)
        return mc

    scf0 = run_scf(coords0)
    mc0 = run_mc(scf0)
    g = casscf_nuc_grad(scf0, mc0, backend="thc").grad
    assert g.shape == (3, 3)

    ia, xyz = 1, 2
    h = 1e-3

    ref = _require_casscf_thc_ref_data(scf0, mc0)

    mo0 = getattr(scf0, "mo_coeff", None)
    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[ia, xyz] += h
    coords_m[ia, xyz] -= h
    scf_p = run_scf(coords_p, mo_coeff0=mo0)
    scf_m = run_scf(coords_m, mo_coeff0=mo0)
    fd = (_fixed_reference_value_thc_casscf(scf_p, ref) - _fixed_reference_value_thc_casscf(scf_m, ref)) / (2.0 * h)

    assert np.isfinite(fd)
    assert np.isfinite(g[ia, xyz])

    assert abs(float(fd) - float(g[ia, xyz])) < 5e-4


def test_thc_casscf_analytic_grad_matches_fixed_reference_fd_component_local():
    _skip_if_cuda_unavailable()

    from asuka.density import DeviceGridSpec
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad import casscf_nuc_grad

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.015],
        ],
        dtype=np.float64,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol = _mol_lih_from_coords(coords_bohr)
        out = run_rhf_thc(
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
            guess=guess_mc,
        )
        # Local-THC CASSCF convergence can be sensitive; the low-noise
        # fixed-reference validation below avoids reoptimizing displaced CI/MOs.
        assert np.isfinite(float(mc.e_tot))
        return mc

    scf0 = run_scf(coords0)
    mc0 = run_mc(scf0)
    g = casscf_nuc_grad(scf0, mc0, backend="thc").grad
    assert g.shape == (2, 3)

    ia, xyz = 1, 2
    h = 1e-3

    ref = _require_casscf_thc_ref_data(scf0, mc0)

    mo0 = getattr(scf0, "mo_coeff", None)
    coords_p = coords0.copy()
    coords_m = coords0.copy()
    coords_p[ia, xyz] += h
    coords_m[ia, xyz] -= h
    scf_p = run_scf(coords_p, mo_coeff0=mo0)
    scf_m = run_scf(coords_m, mo_coeff0=mo0)
    fd = (_fixed_reference_value_thc_casscf(scf_p, ref) - _fixed_reference_value_thc_casscf(scf_m, ref)) / (2.0 * h)

    assert np.isfinite(fd)
    assert np.isfinite(g[ia, xyz])

    assert abs(float(fd) - float(g[ia, xyz])) < 1e-3


def test_thc_casscf_analytic_grad_matches_relaxed_fd_component_global():
    _skip_if_cuda_unavailable()

    from asuka.density import DeviceGridSpec
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad import casscf_nuc_grad

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.757 * 1.8897259886, 0.587 * 1.8897259886],
            [0.0, 0.757 * 1.8897259886, 0.587 * 1.8897259886],
        ],
        dtype=np.float64,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol = _mol_h2o_from_coords(coords_bohr)
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
            max_cycle_macro=200,
            tol=1e-6,
            orbital_optimizer="lbfgs",
            matvec_backend="cuda_eri_mat",
            guess=guess_mc,
        )
        assert bool(mc.converged)
        return mc

    scf0 = run_scf(coords0)
    mc0 = run_mc(scf0)
    g = casscf_nuc_grad(scf0, mc0, backend="thc").grad

    ia, xyz = 1, 2
    h = 1e-3
    fd, e_pm = _run_relaxed_fd_component(coords0, ia=ia, xyz=xyz, h=h, run_scf=run_scf, run_mc=run_mc)

    assert np.isfinite(fd)
    assert np.isfinite(g[ia, xyz])
    assert bool(e_pm[1][1]) and bool(e_pm[1][2])
    assert bool(e_pm[-1][1]) and bool(e_pm[-1][2])
    assert abs(float(fd) - float(g[ia, xyz])) < 5e-4


def test_thc_casscf_analytic_grad_matches_relaxed_fd_component_local():
    _skip_if_cuda_unavailable()

    from asuka.density import DeviceGridSpec
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad import casscf_nuc_grad

    coords0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.015],
        ],
        dtype=np.float64,
    )

    grid = DeviceGridSpec(radial_n=16, angular_n=74, rmax=10.0, becke_n=3, prune_tol=0.0, threads=256)

    def run_scf(coords_bohr: np.ndarray, *, mo_coeff0=None):
        mol = _mol_lih_from_coords(coords_bohr)
        out = run_rhf_thc(
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
            guess=guess_mc,
        )
        assert bool(mc.converged)
        return mc

    scf0 = run_scf(coords0)
    mc0 = run_mc(scf0)
    g = casscf_nuc_grad(scf0, mc0, backend="thc").grad

    ia, xyz = 1, 2
    h = 1e-3
    fd, e_pm = _run_relaxed_fd_component(coords0, ia=ia, xyz=xyz, h=h, run_scf=run_scf, run_mc=run_mc)

    assert np.isfinite(fd)
    assert np.isfinite(g[ia, xyz])
    assert bool(e_pm[1][1]) and bool(e_pm[1][2])
    assert bool(e_pm[-1][1]) and bool(e_pm[-1][2])
    assert abs(float(fd) - float(g[ia, xyz])) < 1e-3
