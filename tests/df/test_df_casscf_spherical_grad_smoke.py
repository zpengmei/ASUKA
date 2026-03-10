import numpy as np
import pytest


@pytest.mark.cuda
def test_df_casscf_spherical_energy_grad_smoke():
    """DF-CASSCF energy + analytic DF gradient should run with mol.cart=False."""
    pytest.importorskip("cupy")  # DF build path depends on CuPy being installed

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df

    # 6-31g* includes d-functions on O -> spherical AO dimension differs from Cartesian.
    mol = Molecule.from_atoms(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.43233673, 1.10715266)),
            ("H", (0.0, 1.43233673, 1.10715266)),
        ],
        basis="6-31g*",
        cart=False,
    )

    scf_out = run_hf(mol, method="rhf", backend="cpu", max_cycle=50)
    assert bool(getattr(scf_out.scf, "converged", False))
    assert getattr(scf_out, "sph_map", None) is not None
    _T, nao_cart, nao_sph = scf_out.sph_map
    assert int(nao_sph) < int(nao_cart)

    mc = run_casscf(
        scf_out,
        ncore=4,
        ncas=2,
        nelecas=2,
        backend="cpu",
        max_cycle_macro=30,
        nroots=2,
        root_weights=[0.5, 0.5],
    )
    assert bool(getattr(mc, "converged", False))

    g = casscf_nuc_grad_df(scf_out, mc, df_backend="cpu")
    assert isinstance(g.e_tot, float)
    assert g.grad.shape == (int(mol.natm), 3)
    assert bool(np.isfinite(g.grad).all())
