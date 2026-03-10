import numpy as np

from asuka.frontend import Molecule, build_ao_basis_cart
from asuka.frontend.periodic_table import atomic_number
from asuka.integrals.cart2sph import compute_sph_layout_from_cart_basis
from asuka.integrals.int1e_cart import shell_to_atom_map
from asuka.integrals.int1e_sph import (
    build_dS_sph,
    build_dT_sph,
    build_dV_sph,
    contract_dS_sph,
    contract_dhcore_sph,
)

_BASIS_O_SPD = {
    "O": [
        [0, [1.0, 1.0]],
        [1, [0.8, 1.0]],
        [2, [0.6, 1.0]],
    ],
}


def _small_o_basis():
    mol = Molecule.from_atoms([("O", (0.0, 0.0, 0.0))], unit="bohr", basis=_BASIS_O_SPD, cart=False)
    ao_basis, _ = build_ao_basis_cart(mol, basis=_BASIS_O_SPD)
    coords = np.asarray(mol.coords_bohr, dtype=np.float64)
    charges = np.asarray([atomic_number("O")], dtype=np.float64)
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    return ao_basis, coords, charges, shell_atom, shell_ao_start_sph, int(nao_sph)


def test_build_dS_sph_matches_contract_dS_sph():
    ao_basis, coords, _charges, shell_atom, shell_ao_start_sph, nao_sph = _small_o_basis()
    rng = np.random.default_rng(12)
    M = rng.normal(size=(nao_sph, nao_sph))
    M = np.asarray(0.5 * (M + M.T), dtype=np.float64)

    dS_sph = build_dS_sph(
        ao_basis,
        atom_coords_bohr=coords,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
    )
    got = np.einsum("axij,ij->ax", dS_sph, M, optimize=True)
    ref = contract_dS_sph(
        ao_basis,
        atom_coords_bohr=coords,
        M_sph=M,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_build_dhcore_components_sph_match_contract_dhcore_sph():
    ao_basis, coords, charges, shell_atom, shell_ao_start_sph, nao_sph = _small_o_basis()
    rng = np.random.default_rng(19)
    M = rng.normal(size=(nao_sph, nao_sph))
    M = np.asarray(0.5 * (M + M.T), dtype=np.float64)

    dT_sph = build_dT_sph(
        ao_basis,
        atom_coords_bohr=coords,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
    )
    dV_sph = build_dV_sph(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
        include_operator_deriv=True,
    )
    got = np.einsum("axij,ij->ax", dT_sph + dV_sph, M, optimize=True)
    ref = contract_dhcore_sph(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        M_sph=M,
        shell_atom=shell_atom,
        shell_ao_start_sph=shell_ao_start_sph,
        include_operator_deriv=True,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-11)
