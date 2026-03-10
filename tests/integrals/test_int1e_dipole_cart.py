import numpy as np


def test_int1e_dipole_cart_s_shell_at_origin_is_zero():
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_dipole_cart import build_overlap_and_dipole_cart

    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0))],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=True,
    )
    basis, _name = build_ao_basis_cart(mol, basis="sto-3g", expand_contractions=True)

    S, Rx, Ry, Rz = build_overlap_and_dipole_cart(basis)
    assert S.shape[0] == S.shape[1]
    assert Rx.shape == S.shape and Ry.shape == S.shape and Rz.shape == S.shape
    assert np.all(np.isfinite(S))
    np.testing.assert_allclose(S, S.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Rx, Rx.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Ry, Ry.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Rz, Rz.T, rtol=0.0, atol=1e-12)

    # For an s function centered at the origin, <s|x|s> = <s|y|s> = <s|z|s> = 0.
    np.testing.assert_allclose(Rx, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Ry, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Rz, 0.0, rtol=0.0, atol=1e-12)

