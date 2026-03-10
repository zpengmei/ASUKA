import numpy as np
import pytest

from asuka.frontend import Molecule, build_ao_basis_cart
from asuka.frontend.periodic_table import atomic_number
from asuka.integrals.int1e_cart import shell_to_atom_map
from asuka.integrals.int1e_sph import build_dS_sph, build_dT_sph, build_dV_sph
from asuka.integrals.int1e_sph_cuda import (
    contract_dS_sph_prebuilt_cuda,
    contract_dhcore_sph_prebuilt_cuda,
    has_int1e_sph_cuda_kernels,
)

_BASIS_O_SPD = {
    "O": [
        [0, [1.0, 1.0]],
        [1, [0.8, 1.0]],
        [2, [0.6, 1.0]],
    ],
}


@pytest.mark.cuda
def test_int1e_sph_cuda_prebuilt_contractions_match_cpu():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")
    if not has_int1e_sph_cuda_kernels():
        pytest.skip("CUDA extension lacks spherical int1e contraction kernels")

    mol = Molecule.from_atoms([("O", (0.0, 0.0, 0.0))], unit="bohr", basis=_BASIS_O_SPD, cart=False)
    ao_basis, _ = build_ao_basis_cart(mol, basis=_BASIS_O_SPD)
    coords = np.asarray(mol.coords_bohr, dtype=np.float64)
    charges = np.asarray([atomic_number("O")], dtype=np.float64)
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    dS = build_dS_sph(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
    dT = build_dT_sph(ao_basis, atom_coords_bohr=coords, shell_atom=shell_atom)
    dV = build_dV_sph(
        ao_basis,
        atom_coords_bohr=coords,
        atom_charges=charges,
        shell_atom=shell_atom,
        include_operator_deriv=True,
    )

    nao_sph = int(dS.shape[2])
    rng = np.random.default_rng(73)
    M = rng.normal(size=(nao_sph, nao_sph))
    M = np.asarray(0.5 * (M + M.T), dtype=np.float64)

    ref_dS = np.einsum("axij,ij->ax", dS, M, optimize=True)
    ref_h1 = np.einsum("axij,ij->ax", dT + dV, M, optimize=True)

    got_dS = contract_dS_sph_prebuilt_cuda(dS, M)
    got_h1 = contract_dhcore_sph_prebuilt_cuda(dT, dV, M)

    np.testing.assert_allclose(got_dS, ref_dS, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got_h1, ref_h1, rtol=1e-10, atol=1e-10)
