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
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")


def test_thc_g_dm2_matches_explicit_contraction_small():
    """Validate THC g_dm2 contraction against an explicit (pu|wx) sum.

    This is an internal consistency test for the `orbital_gradient_thc` g_dm2
    path: the optimized contraction must match the direct definition

      g_dm2[p,v] = sum_{u,w,x} (p u | w x) * dm2[w,x,u,v]

    where the ERIs are approximated by the same THC factors.
    """

    _skip_if_cuda_unavailable()

    import cupy as cp

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf.casci import run_casci
    from asuka.density import DeviceGridSpec
    from asuka.solver import GUGAFCISolver

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
        thc_grid_spec=grid,
        thc_grid_kind="rdvr",
        thc_dvr_basis="aux",
        thc_npt=1200,
        thc_solve_method="fit_metric_qr",
        use_density_difference=True,
        df_warmup_cycles=2,
        max_cycle=30,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
        diis=True,
    )
    assert bool(scf_out.scf.converged)
    assert scf_out.thc_factors is not None

    # LiH (4 e-): CAS(2,2) -> ncore = (4-2)/2 = 1
    ncore = 1
    ncas = 2
    nelecas = 2
    cas = run_casci(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        backend="cuda",
        df=True,
        matvec_backend="cuda_eri_mat",
    )
    assert np.isfinite(float(cas.e_tot))

    fci = GUGAFCISolver(twos=0, nroots=1)
    dm1_act, dm2_act = fci.make_rdm12(cas.ci, int(ncas), nelecas)
    dm2 = np.asarray(dm2_act, dtype=np.float64)
    assert dm2.shape == (int(ncas), int(ncas), int(ncas), int(ncas))

    thc = scf_out.thc_factors
    X = cp.asarray(thc.X, dtype=cp.float64)
    Y = cp.asarray(thc.Y, dtype=cp.float64)
    C = cp.asarray(scf_out.scf.mo_coeff, dtype=cp.float64)
    X_mo = X @ C
    X_act = X_mo[:, int(ncore) : int(ncore) + int(ncas)]

    npt = int(X_mo.shape[0])
    nmo = int(C.shape[1])
    naux = int(Y.shape[1])

    # Build active-pair vectors d[wx,L] = sum_P X_act[P,w]*X_act[P,x]*Y[P,L]
    d_wx = cp.empty((int(ncas) * int(ncas), int(naux)), dtype=cp.float64)
    for w in range(int(ncas)):
        for x in range(int(ncas)):
            v = X_act[:, int(w)] * X_act[:, int(x)]
            d_wx[int(w) * int(ncas) + int(x)] = v.T @ Y

    # "Fast" contraction (matches orbital_gradient_thc implementation).
    dm2_flat = cp.asarray(dm2.reshape(int(ncas) * int(ncas), int(ncas) * int(ncas)), dtype=cp.float64)
    T_flat = d_wx.T @ dm2_flat  # (naux,ncas^2)
    S_flat = Y @ T_flat  # (npt,ncas^2)
    S = S_flat.reshape(int(npt), int(ncas), int(ncas))
    t_pv = cp.einsum("Pu,Puv->Pv", X_act, S, optimize=True)
    g_fast = X_mo.T @ t_pv  # (nmo,ncas)
    g_fast_h = cp.asnumpy(g_fast)

    # Explicit reference: g[p,v] = sum_{u,w,x} (p u|w x) dm2[w,x,u,v]
    dm2_h = np.asarray(dm2, dtype=np.float64)
    g_ref = np.zeros((int(nmo), int(ncas)), dtype=np.float64)
    for p in range(int(nmo)):
        x_p = X_mo[:, int(p)]
        for v in range(int(ncas)):
            acc = 0.0
            for u in range(int(ncas)):
                d_pu = (x_p * X_act[:, int(u)]).T @ Y  # (naux,)
                eri_pu_wx = d_wx @ d_pu  # (ncas^2,)
                for w in range(int(ncas)):
                    for x in range(int(ncas)):
                        wx = int(w) * int(ncas) + int(x)
                        acc += float(eri_pu_wx[wx].item()) * float(dm2_h[int(w), int(x), int(u), int(v)])
            g_ref[int(p), int(v)] = acc

    assert np.allclose(g_fast_h, g_ref, rtol=1e-9, atol=1e-9)

