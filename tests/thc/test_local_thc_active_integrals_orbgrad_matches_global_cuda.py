import dataclasses

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

    # cuERI CUDA backend is required for THC auxiliary metric build.
    try:
        from asuka.cueri.gpu import has_cuda_ext

        if not bool(has_cuda_ext()):
            pytest.skip("cuERI CUDA extension not available")
    except Exception:
        pytest.skip("cuERI CUDA extension not available")

    # GUGA CUDA backend is required for matvec_backend='cuda_eri_mat'.
    try:
        from asuka.cuda.cuda_backend import has_cuda_ext as has_guga_cuda_ext

        if not bool(has_guga_cuda_ext()):
            pytest.skip("GUGA CUDA extension not available")
    except Exception:
        pytest.skip("GUGA CUDA extension not available")


def _make_synthetic_local_thc_from_global(thc, *, nao: int, ao_rep: str):
    """Wrap global THCFactors in a single-block LocalTHCFactors for equivalence checks."""
    from asuka.hf.local_thc_factors import LocalTHCBlock, LocalTHCFactors

    blk = LocalTHCBlock(
        block_id=0,
        ao_idx_global=np.arange(int(nao), dtype=np.int32),
        n_early=0,
        n_primary=int(nao),
        atoms_primary=(),
        atoms_secondary_early=(),
        atoms_secondary_late=(),
        atoms_aux=(),
        X=thc.X,
        Y=thc.Y,
        Z=thc.Z,
        points=thc.points,
        weights=thc.weights,
        L_metric=thc.L_metric,
        meta={"synthetic": True},
    )
    return LocalTHCFactors(
        blocks=(blk,),
        nao=int(nao),
        ao_rep=str(ao_rep),
        L_metric_full=thc.L_metric,
        meta={"synthetic": True},
    )


def test_synthetic_local_thc_matches_global_for_active_integrals_and_orbgrad():
    """Local-THC code paths must match global THC when factors are identical.

    This is a strong consistency check for:
    - active-space integral builds (eri_mat + j_ps)
    - orbital_gradient_thc, including the g_dm2 contraction
    """
    _skip_if_cuda_unavailable()

    import cupy as cp

    from asuka.cuda.active_space_thc import build_device_dfmo_integrals_local_thc, build_device_dfmo_integrals_thc
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_rhf_thc
    from asuka.mcscf.casci import run_casci
    from asuka.mcscf.orbital_grad import orbital_gradient_thc
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
        thc_mode="global",
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
    assert bool(scf_out.converged)
    assert scf_out.thc_factors is not None

    thc = scf_out.thc_factors
    nao = int(getattr(thc.X, "shape", (0, 0))[1])
    ao_rep = "cart" if bool(getattr(mol, "cart", False)) else "sph"
    lthc = _make_synthetic_local_thc_from_global(thc, nao=int(nao), ao_rep=str(ao_rep))
    scf_out_l = dataclasses.replace(scf_out, thc_factors=lthc)

    # LiH (4 e-): CAS(2,2) -> ncore = (4-2)/2 = 1
    ncore = 1
    ncas = 2
    nelecas = 2
    C = cp.asarray(scf_out.mo_coeff, dtype=cp.float64)
    C_act = C[:, int(ncore) : int(ncore) + int(ncas)]

    # Active-space integrals must match exactly (same X/Y factors).
    eri_g = build_device_dfmo_integrals_thc(thc, C_act, want_eri_mat=True)
    eri_l = build_device_dfmo_integrals_local_thc(lthc, C_act, want_eri_mat=True)
    cp.testing.assert_allclose(eri_l.eri_mat, eri_g.eri_mat, rtol=1e-10, atol=1e-10)
    cp.testing.assert_allclose(eri_l.j_ps, eri_g.j_ps, rtol=1e-10, atol=1e-10)

    # CASCI -> RDMs
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

    # Orbital gradients must match (global vs local path) for identical factors.
    G_g, gn_g, eps_g = orbital_gradient_thc(
        scf_out,
        C=C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        q_block=64,
        pair_p_block=4,
    )
    G_l, gn_l, eps_l = orbital_gradient_thc(
        scf_out_l,
        C=C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        q_block=64,
        pair_p_block=4,
    )

    # The outputs are GPU arrays (CuPy) here.
    cp.testing.assert_allclose(cp.asarray(G_l), cp.asarray(G_g), rtol=1e-10, atol=1e-10)
    cp.testing.assert_allclose(cp.asarray(eps_l), cp.asarray(eps_g), rtol=1e-12, atol=1e-12)
    assert abs(float(gn_l) - float(gn_g)) < 1e-10
